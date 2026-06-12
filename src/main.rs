//! Dictata — 100% local system-wide voice dictation.
//!
//! The application is "tray-only": the dock is the primary eframe viewport,
//! hidden when idle and shown during recording / transcription. A global
//! hotkey triggers the cycle; transcription runs on a worker thread, then
//! the text (optionally reformatted by an LLM mode) is pasted into the
//! active application.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use dictata::audio::Recorder;
use dictata::config::{self, Config};
use dictata::dock::{Dock, DockState};
use dictata::hotkey::{self, HotKeyState, Hotkeys};
use dictata::settings::{self, SettingsState};
use dictata::transcriber::Transcriber;
use dictata::tray::{Tray, TrayAction};
use dictata::i18n::tr;
use dictata::streaming::{StreamParams, StreamingSession};
use dictata::{history, i18n, modes, paste, platform};

/// Path of the ggml model file from the configured name (e.g. "tiny").
fn model_file_path(cfg: &Config) -> PathBuf {
    dictata::models::model_path(&cfg.model_dir, &cfg.model)
}

/// Result returned by the transcription thread.
enum Worker {
    Ok { text: String, status: &'static str },
    Err(String),
}

enum State {
    Idle,
    Recording { since: Instant },
    Transcribing,
    Flash { until: Instant },
    /// "Position the dock" mode: visible and draggable with the mouse.
    Positioning { until: Instant },
}

struct App {
    cfg: Config,
    dock: Dock,
    state: State,
    recorder: Recorder,
    _hotkeys: Hotkeys, // keeps the manager alive (RAII)
    hotkey_id: Option<u32>,
    tray: Option<Tray>,
    transcriber: Arc<Mutex<Option<Transcriber>>>,
    tx: Sender<Worker>,
    rx: Receiver<Worker>,
    no_activate_done: bool,
    dock_shown: bool,
    settings: Option<SettingsState>,
    /// True once the settings viewport has been rendered at least once.
    settings_live: bool,
    /// Frames to wait before creating the settings viewport (see render_settings).
    settings_warmup: u8,
    /// Streaming session in progress (continuous mode, raw type only).
    stream: Option<StreamingSession>,
    /// Worker of the last streaming session: joined before starting another.
    stream_join: Option<std::thread::JoinHandle<()>>,
}

impl App {
    fn new(cfg: Config) -> Self {
        let mut hotkeys = Hotkeys::new().expect("init raccourci global");
        let hotkey_id = match hotkeys.set(&cfg.hotkey) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("raccourci: {e}");
                None
            }
        };
        let tray = match Tray::new() {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("tray: {e}");
                None
            }
        };
        let mut dock = Dock::new();
        dock.scale = cfg.dock_scale.clamp(0.7, 1.6);
        dock.opacity = cfg.dock_opacity.clamp(0.4, 1.0);
        if let Some(m) = cfg.modes.get(&cfg.active_mode) {
            dock.mode_label = m.label.clone();
        }
        let recorder = Recorder::new(cfg.input_device.clone(), cfg.audio_source.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let settings = if std::env::var("DICTATA_OPEN_SETTINGS").is_ok() {
            Some(SettingsState::new(cfg.clone()))
        } else {
            None
        };
        App {
            cfg,
            dock,
            state: State::Idle,
            recorder,
            _hotkeys: hotkeys,
            hotkey_id,
            tray,
            transcriber: Arc::new(Mutex::new(None)),
            tx,
            rx,
            no_activate_done: false,
            dock_shown: false,
            settings,
            settings_live: false,
            settings_warmup: 0,
            stream: None,
            stream_join: None,
        }
    }

    /// Renders the settings window (secondary viewport). The window works
    /// on a copy of the config; it is only applied on clicking "Save".
    fn render_settings(&mut self, ctx: &egui::Context) {
        if self.settings.is_none() {
            return;
        }
        // eframe panics ("user callback was never called") if an immediate
        // viewport is created while the parent window is actually hidden.
        // Workaround: make the parent visible (it paints nothing and is fully
        // transparent, so nothing shows on screen), wait one frame for the
        // command to take effect, create the child, then hide the parent again.
        if !self.settings_live {
            if self.settings_warmup == 0 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                self.settings_warmup = 2;
                ctx.request_repaint();
                return;
            }
            self.settings_warmup -= 1;
            if self.settings_warmup > 0 {
                ctx.request_repaint();
                return;
            }
        }
        let mut closed_x = false;
        let mut result = settings::RenderResult::default();
        let builder = egui::ViewportBuilder::default()
            .with_title("Dictata \u{2014} Reglages")
            .with_inner_size([880.0, 620.0])
            .with_min_inner_size([720.0, 520.0]);
        {
            let st = self.settings.as_mut().unwrap();
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("dictata_settings"),
                builder,
                |ui, _class| {
                    settings::apply_theme(ui.ctx());
                    result = settings::render(ui, st);
                    if ui.ctx().input(|i| i.viewport().close_requested()) {
                        closed_x = true;
                    }
                },
            );
        }
        if !self.settings_live {
            // Viewport created: the parent can go back to hidden.
            self.settings_live = true;
            if !self.dock_shown {
                ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            }
        }
        if result.save {
            let new_cfg = self.settings.as_ref().unwrap().cfg.clone();
            self.apply_config(new_cfg);
            if let Some(st) = self.settings.as_mut() {
                st.set_status(tr("saved_ok"));
            }
        }
        if result.close || closed_x {
            self.settings = None;
            self.settings_live = false;
            self.settings_warmup = 0;
            // The window was tracking its working copy: revert to the real config.
            i18n::set_lang(&self.cfg.ui_lang);
        }
        // "Position the dock" request: shows the dock, draggable for 6 s.
        let requested = self
            .settings
            .as_mut()
            .map(|st| std::mem::take(&mut st.position_dock_request))
            .unwrap_or(false);
        if requested && !matches!(self.state, State::Recording { .. } | State::Transcribing) {
            self.dock.state = DockState::Idle;
            self.dock.status_text = tr("dock_drag").into();
            self.state = State::Positioning {
                until: Instant::now() + Duration::from_secs(6),
            };
            self.show_dock(ctx, true);
        }
    }

    /// Applies an edited config: save + side effects (hotkey,
    /// transcriber to reload if the model changes, active mode label).
    fn apply_config(&mut self, new: Config) {
        let hotkey_changed = self.cfg.hotkey != new.hotkey;
        let model_changed =
            self.cfg.model != new.model || self.cfg.gpu != new.gpu || self.cfg.model_dir != new.model_dir;
        self.cfg = new;
        config::save(&self.cfg);
        i18n::set_lang(&self.cfg.ui_lang);
        if hotkey_changed {
            match self._hotkeys.set(&self.cfg.hotkey) {
                Ok(id) => self.hotkey_id = Some(id),
                Err(e) => eprintln!("raccourci: {e}"),
            }
        }
        if model_changed {
            *self.transcriber.lock().unwrap_or_else(|p| p.into_inner()) = None; // lazy reload
        }
        if let Some(m) = self.cfg.modes.get(&self.cfg.active_mode) {
            self.dock.mode_label = m.label.clone();
        }
        self.dock.scale = self.cfg.dock_scale.clamp(0.7, 1.6);
        self.dock.opacity = self.cfg.dock_opacity.clamp(0.4, 1.0);
    }

    /// Dock size in points, based on `dock_scale`.
    fn dock_size(&self) -> egui::Vec2 {
        let s = self.cfg.dock_scale.clamp(0.7, 1.6);
        egui::vec2(dictata::dock::WIDTH * s, dictata::dock::HEIGHT * s)
    }

    fn show_dock(&mut self, ctx: &egui::Context, visible: bool) {
        if visible != self.dock_shown {
            if visible {
                let size = self.dock_size();
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
                let pos = if let Some([x, y]) = self.cfg.dock_pos {
                    egui::pos2(x, y)
                } else if let Some(monitor) = ctx.input(|i| i.viewport().monitor_size) {
                    egui::pos2((monitor.x - size.x) / 2.0, monitor.y - size.y - 60.0)
                } else {
                    egui::pos2(0.0, 0.0)
                };
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(visible));
            self.dock_shown = visible;
        }
    }

    fn flash(&mut self, ctx: &egui::Context, state: DockState, status: &str, secs: f32) {
        self.dock.state = state;
        self.dock.status_text = status.to_string();
        self.state = State::Flash {
            until: Instant::now() + Duration::from_secs_f32(secs),
        };
        self.show_dock(ctx, true);
    }

    fn toggle(&mut self, ctx: &egui::Context) {
        match self.state {
            State::Idle | State::Flash { .. } => self.start_recording(ctx),
            State::Recording { .. } => self.stop_and_transcribe(ctx),
            State::Transcribing => {}     // busy: ignore
            State::Positioning { .. } => {} // positioning in progress: ignore
        }
    }

    /// Ends positioning: stores the window's current position.
    fn finish_positioning(&mut self, ctx: &egui::Context) {
        if let Some(rect) = ctx.input(|i| i.viewport().outer_rect) {
            self.cfg.dock_pos = Some([rect.left(), rect.top()]);
            config::save(&self.cfg);
            if let Some(st) = self.settings.as_mut() {
                st.cfg.dock_pos = self.cfg.dock_pos;
                st.set_status(tr("cfg_dock_saved"));
            }
        }
        self.state = State::Idle;
        self.dock.state = DockState::Idle;
        self.dock.status_text.clear();
        self.show_dock(ctx, false);
    }

    fn start_recording(&mut self, ctx: &egui::Context) {
        // No new session while the previous streaming worker is still
        // alive (otherwise two workers share the transcriber and pasting).
        if let Some(h) = self.stream_join.take() {
            if h.is_finished() {
                let _ = h.join();
            } else {
                self.stream_join = Some(h);
                self.flash(ctx, DockState::Error, tr("status_busy"), 1.2);
                return;
            }
        }
        self.recorder = Recorder::new(self.cfg.input_device.clone(), self.cfg.audio_source.clone());
        match self.recorder.start() {
            Ok(()) => {
                // Continuous mode: transcription/insertion as pauses occur
                // (only for raw-type modes, like the v1).
                let mode_is_raw = self
                    .cfg
                    .modes
                    .get(&self.cfg.active_mode)
                    .map(|m| m.kind == "raw")
                    .unwrap_or(true);
                if self.cfg.streaming && mode_is_raw {
                    let lang = self
                        .cfg
                        .modes
                        .get(&self.cfg.active_mode)
                        .and_then(|m| m.language.clone())
                        .or_else(|| self.cfg.language.clone());
                    let params = StreamParams {
                        model_path: model_file_path(&self.cfg),
                        gpu: self.cfg.gpu != "cpu",
                        language: lang,
                        vocab_prompt: modes::build_initial_prompt(&self.cfg.vocabulary, ""),
                        beam_size: self.cfg.beam_size,
                    };
                    let auto_paste = self.cfg.auto_paste;
                    let restore_delay = self.cfg.paste_restore_delay;
                    let emit = move |s: &str| {
                        if auto_paste {
                            if let Err(e) = paste::paste_text_with_delay(s, restore_delay) {
                                eprintln!("collage (stream): {e}");
                            }
                        }
                    };
                    let tx = self.tx.clone();
                    let ctx2 = ctx.clone();
                    let done = move |res: Result<String, String>| {
                        let _ = match res {
                            Ok(text) => tx.send(Worker::Ok {
                                text,
                                status: "stream",
                            }),
                            Err(e) => tx.send(Worker::Err(e)),
                        };
                        ctx2.request_repaint();
                    };
                    self.stream = Some(StreamingSession::start(
                        self.recorder.drain_handle(),
                        self.transcriber.clone(),
                        params,
                        emit,
                        done,
                    ));
                }
                self.dock.reset_levels();
                self.dock.state = DockState::Recording;
                self.dock.status_text = "0:00".into();
                self.state = State::Recording {
                    since: Instant::now(),
                };
                self.show_dock(ctx, true);
                eprintln!("[etat] enregistrement demarre");
            }
            Err(e) => {
                eprintln!("micro: {e}");
                self.flash(ctx, DockState::Error, tr("status_mic_ko"), 1.6);
            }
        }
    }

    fn stop_and_transcribe(&mut self, ctx: &egui::Context) {
        // Streaming session: the worker transcribes the remainder then signals completion.
        if let Some(session) = self.stream.take() {
            let tail = self.recorder.stop();
            eprintln!("[etat] arret streaming ({} echantillons restants)", tail.len());
            self.stream_join = Some(session.finish(tail));
            self.dock.state = DockState::Transcribing;
            self.dock.status_text = "\u{2026}".into();
            self.state = State::Transcribing;
            return;
        }
        let audio = self.recorder.stop();
        eprintln!(
            "[etat] arret -> transcription ({} echantillons, {:.2}s)",
            audio.len(),
            audio.len() as f32 / 16000.0
        );
        self.dock.state = DockState::Transcribing;
        self.dock.status_text = "\u{2026}".into();
        self.state = State::Transcribing;

        let cfg = self.cfg.clone();
        let mode = cfg
            .modes
            .get(&cfg.active_mode)
            .cloned()
            .unwrap_or_else(|| cfg.modes.values().next().cloned().expect("au moins un mode"));
        let model_path = model_file_path(&cfg);
        let gpu = cfg.gpu != "cpu";
        let tr = self.transcriber.clone();
        let tx = self.tx.clone();
        let ctx2 = ctx.clone();

        std::thread::spawn(move || {
            if audio.len() < 4000 {
                let _ = tx.send(Worker::Ok {
                    text: String::new(),
                    status: "raw",
                });
                ctx2.request_repaint();
                return;
            }
            let mut guard = tr.lock().unwrap_or_else(|p| p.into_inner());
            if guard.is_none() {
                match Transcriber::load(&model_path, gpu) {
                    Ok(t) => *guard = Some(t),
                    Err(e) => {
                        let _ = tx.send(Worker::Err(e));
                        ctx2.request_repaint();
                        return;
                    }
                }
            }
            let t = guard.as_ref().unwrap();
            let lang = mode.language.clone().or_else(|| cfg.language.clone());
            let translate = mode.task == "translate";
            let prompt = modes::build_initial_prompt(&cfg.vocabulary, "");
            let prompt_opt = if prompt.is_empty() {
                None
            } else {
                Some(prompt.as_str())
            };
            match t.transcribe(&audio, lang.as_deref(), translate, prompt_opt, cfg.beam_size) {
                Ok(raw) => {
                    let (text, status) = modes::apply_mode(&raw, &mode, &cfg);
                    if !text.trim().is_empty() {
                        let dur = audio.len() as f64 / 16000.0;
                        history::add_entry(&text, &cfg.active_mode, lang.as_deref(), Some(dur));
                    }
                    let _ = tx.send(Worker::Ok { text, status });
                }
                Err(e) => {
                    let _ = tx.send(Worker::Err(e));
                }
            }
            ctx2.request_repaint();
        });
    }

    fn handle_worker(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Worker::Ok { text, status } => {
                    eprintln!("[worker] statut={status}, texte={text:?}");
                    if text.trim().is_empty() {
                        self.flash(ctx, DockState::Done, tr("status_empty"), 0.9);
                    } else if status == "stream" {
                        // Already pasted chunk by chunk: just history + flash.
                        history::add_entry(&text, &self.cfg.active_mode, None, None);
                        self.flash(ctx, DockState::Done, tr("status_pasted"), 0.9);
                    } else {
                        let mut paste_failed = false;
                        if self.cfg.auto_paste {
                            if let Err(e) =
                                paste::paste_text_with_delay(&text, self.cfg.paste_restore_delay)
                            {
                                eprintln!("collage: {e}");
                                paste_failed = true;
                            }
                        }
                        if paste_failed {
                            // Text stays in the clipboard: surface the failure
                            // instead of pretending it was pasted.
                            self.flash(ctx, DockState::Error, tr("status_paste_error"), 1.6);
                            continue;
                        }
                        let label = match status {
                            "llm" => tr("status_reformulated"),
                            "llm_fallback" => tr("status_raw_fallback"),
                            _ => tr("status_pasted"),
                        };
                        self.flash(ctx, DockState::Done, label, 0.9);
                    }
                }
                Worker::Err(e) => {
                    eprintln!("transcription: {e}");
                    self.flash(ctx, DockState::Error, tr("status_error"), 1.6);
                }
            }
        }
    }
}

impl eframe::App for App {
    fn clear_color(&self, _v: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let t = ctx.input(|i| i.time);

        // No-focus-steal applied once (the HWND exists even when hidden).
        if !self.no_activate_done {
            platform::make_no_activate(frame);
            // eframe does not honor with_visible(false) for the primary viewport:
            // force hiding the idle dock from the first frame.
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.no_activate_done = true;
        }

        // Global hotkey events.
        for (id, st) in hotkey::poll_events() {
            if st == HotKeyState::Pressed && Some(id) == self.hotkey_id {
                self.toggle(&ctx);
            }
        }

        // Tray menu events.
        if let Some(tray) = &self.tray {
            for action in tray.poll_actions() {
                match action {
                    TrayAction::OpenSettings => {
                        if self.settings.is_none() {
                            self.settings = Some(SettingsState::new(self.cfg.clone()));
                        }
                    }
                    TrayAction::Quit => {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
        }

        // Transcription results.
        self.handle_worker(&ctx);

        // Update according to state.
        let active = match self.state {
            State::Recording { since } => {
                let lvl = self.recorder.level();
                self.dock.push_level(lvl);
                let secs = since.elapsed().as_secs();
                self.dock.status_text = format!("{}:{:02}", secs / 60, secs % 60);
                // Recording duration cap (bounds memory usage).
                if secs >= self.cfg.max_record_seconds as u64 {
                    eprintln!("[etat] duree max atteinte -> arret");
                    self.stop_and_transcribe(&ctx);
                }
                true
            }
            State::Transcribing => true,
            State::Flash { until } => {
                if Instant::now() >= until {
                    self.state = State::Idle;
                    self.dock.state = DockState::Idle;
                    self.show_dock(&ctx, false);
                    false
                } else {
                    true
                }
            }
            State::Positioning { until } => {
                // Drag and drop: a held click moves the window, and
                // each interaction extends the positioning window.
                if ctx.input(|i| i.pointer.primary_pressed()) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    self.state = State::Positioning {
                        until: Instant::now() + Duration::from_secs(6),
                    };
                }
                if Instant::now() >= until && !ctx.input(|i| i.pointer.primary_down()) {
                    self.finish_positioning(&ctx);
                    false
                } else {
                    true
                }
            }
            State::Idle => {
                self.show_dock(&ctx, false);
                false
            }
        };

        if self.dock_shown {
            self.dock.paint(ui.painter(), ctx.content_rect(), t);
        }

        // Settings window (secondary viewport) if open.
        self.render_settings(&ctx);

        // Polls the hotkey/tray even when idle, faster when active.
        let delay = if active || self.settings.is_some() { 16 } else { 80 };
        ctx.request_repaint_after(Duration::from_millis(delay));
    }
}

fn main() -> eframe::Result {
    // Redirects whisper.cpp/ggml logs to the `log` crate (silent
    // without a configured logger) instead of polluting stderr.
    whisper_rs::install_logging_hooks();

    let cfg = config::load();
    i18n::set_lang(&cfg.ui_lang);
    eprintln!(
        "Dictata — modele={}, hotkey={}, mode={}",
        cfg.model, cfg.hotkey, cfg.active_mode
    );

    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([dictata::dock::WIDTH, dictata::dock::HEIGHT])
        .with_decorations(false)
        .with_transparent(true)
        .with_resizable(false)
        .with_always_on_top()
        .with_taskbar(false)
        .with_active(false)
        .with_visible(false)
        .with_title("Dictata");
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "Dictata",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(cfg)))),
    )
}
