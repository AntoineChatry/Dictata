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
use dictata::{history, i18n, modes, paste, platform, settings_logic};

/// Path of the ggml model file from the configured name (e.g. "tiny").
fn model_file_path(cfg: &Config) -> PathBuf {
    dictata::models::model_path(&cfg.model_dir, &cfg.model)
}

/// Result returned by the transcription thread.
///
/// Carries the take it belongs to: results arrive one or more frames after the
/// take ended, by which time a new take may already have replaced the
/// application's current mode. Reading `take_mode` on reception logged the
/// wrong mode in the history; `id` likewise makes cancellation exact instead of
/// dropping whichever message happens to arrive first.
enum Worker {
    Ok {
        id: u64,
        text: String,
        status: &'static str,
        mode: String,
    },
    Err(u64, String),
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
    /// Mode key resolved for the current take. Auto-mode may override
    /// `cfg.active_mode` based on the foreground app captured at record start.
    take_mode: String,
    /// Monotonic id of the current take, attached to its worker results.
    take_id: u64,
    /// Id of a cancelled take whose worker still owes one result, so exactly
    /// that message is dropped and no other.
    cancelled_take: Option<u64>,
    /// Escape-hold gesture for the take in progress (see `platform`).
    cancel_gesture: platform::CancelGesture,
    /// Foreground window seen at the last record start, surfaced in the
    /// settings so a rule can be written against what is actually detected.
    last_window: Option<(String, String)>,
    /// One-off message to flash on the dock shortly after startup (model
    /// recovered after a crash). Shown from the second frame, so it does not
    /// race the initial "hide the dock" viewport command.
    startup_notice: Option<&'static str>,
}

impl App {
    fn new(cfg: Config) -> Self {
        let mut hotkeys = Hotkeys::new().expect("global hotkey init");
        let hotkey_id = match hotkeys.set(&cfg.hotkey) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("hotkey: {e}");
                None
            }
        };
        let mode_list = tray_mode_list(&cfg);
        let tray = match Tray::new(&mode_list, &cfg.active_mode) {
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
        let take_mode = cfg.active_mode.clone();
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
            take_mode,
            take_id: 0,
            cancelled_take: None,
            cancel_gesture: platform::CancelGesture::new(false),
            last_window: None,
            startup_notice: None,
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
                Err(e) => eprintln!("hotkey: {e}"),
            }
        }
        if model_changed {
            *self.transcriber.lock().unwrap_or_else(|p| p.into_inner()) = None; // lazy reload
        }
        if let Some(m) = self.cfg.modes.get(&self.cfg.active_mode) {
            self.dock.mode_label = m.label.clone();
        }
        // Modes may have been renamed/added/removed in settings: rebuild the
        // tray Mode submenu so it stays in sync.
        let mode_list = tray_mode_list(&self.cfg);
        let active = self.cfg.active_mode.clone();
        if let Some(tray) = &mut self.tray {
            if let Err(e) = tray.refresh(&mode_list, &active) {
                eprintln!("tray refresh: {e}");
            }
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

    /// Push-to-talk, key down: starts a take, with the same guards as
    /// [`Self::toggle`]. A take already running is left alone.
    fn ptt_press(&mut self, ctx: &egui::Context) {
        match self.state {
            State::Idle | State::Flash { .. } => self.start_recording(ctx),
            State::Recording { .. } => {}   // already recording: ignore
            State::Transcribing => {}       // busy: ignore
            State::Positioning { .. } => {} // positioning in progress: ignore
        }
    }

    /// Push-to-talk, key up: ends the take. A release received in any other
    /// state (take refused because busy, shortcut changed mid-press) is
    /// ignored; `max_record_seconds` remains the safety net if it never comes.
    fn ptt_release(&mut self, ctx: &egui::Context) {
        if matches!(self.state, State::Recording { .. }) {
            self.stop_and_transcribe(ctx);
        }
    }

    /// Escape during a take: drop the audio without transcribing or pasting.
    /// In streaming mode the chunks already pasted stay pasted (they left the
    /// application as they were produced); only the remainder is dropped.
    fn cancel_recording(&mut self, ctx: &egui::Context) {
        self.abort_take(ctx, "Escape", tr("status_cancelled"), 1.0);
    }

    /// Ends the current take without transcribing or pasting, and shows
    /// `status` on the dock. `reason` is a short diagnostic tag; it never
    /// carries dictated content.
    fn abort_take(&mut self, ctx: &egui::Context, reason: &str, status: &str, secs: f32) {
        if !matches!(self.state, State::Recording { .. }) {
            return;
        }
        if let Some(session) = self.stream.take() {
            let _ = self.recorder.stop();
            // The worker still reports back once; swallow that one result so the
            // "cancelled" feedback is not overwritten by an "empty take" one.
            self.cancelled_take = Some(self.take_id);
            self.stream_join = Some(session.cancel());
        } else {
            let _ = self.recorder.stop();
        }
        eprintln!("[state] take aborted ({reason})");
        self.flash(ctx, DockState::Error, status, secs);
    }

    /// Mode key to use for the next take. With auto-mode on, the foreground
    /// application (captured now, while it still holds focus) may select a
    /// mapped mode; otherwise the manually chosen `active_mode` is kept.
    /// LLM-type modes still fall back to raw unless the local LLM is enabled.
    fn effective_mode_key(&mut self) -> String {
        if self.cfg.auto_mode {
            if let Some((exe, title)) = platform::foreground_app() {
                self.last_window = Some((exe.clone(), title.clone()));
                if let Some(st) = self.settings.as_mut() {
                    st.last_window = Some((exe.clone(), title.clone()));
                }
                let matched = settings_logic::match_app_mode(&self.cfg.app_modes, &exe, &title);
                // The window title is user content (document names, email
                // subjects) and stays out of the logs; it is shown in the
                // auto-mode settings card, which is where it is needed.
                eprintln!(
                    "[auto-mode] exe={exe:?} -> {}",
                    matched.map(|s| s.as_str()).unwrap_or("(no rule)")
                );
                if let Some(key) = matched {
                    if self.cfg.modes.contains_key(key) {
                        return key.clone();
                    }
                    eprintln!("[auto-mode] unknown mode {key:?}, ignored");
                }
            }
        }
        self.cfg.active_mode.clone()
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
        // Checked before anything is written to `self`: a refused take must
        // leave the previous one's mode and id untouched.
        if let Some(h) = self.stream_join.take() {
            if h.is_finished() {
                let _ = h.join();
            } else {
                self.stream_join = Some(h);
                self.flash(ctx, DockState::Error, tr("status_busy"), 1.2);
                return;
            }
        }
        // Resolve the take's mode now, while the target application still has
        // focus (auto-mode reads the foreground app before the dock appears).
        let take_mode = self.effective_mode_key();
        self.recorder = Recorder::new(self.cfg.input_device.clone(), self.cfg.audio_source.clone());
        match self.recorder.start() {
            Ok(()) => {
                self.take_mode = take_mode;
                self.take_id = self.take_id.wrapping_add(1);
                // Escape held right now belongs to the app that had focus, not
                // to this take: it must be released before it can abort.
                self.cancel_gesture = platform::CancelGesture::new(platform::escape_down());
                // Continuous mode: transcription/insertion as pauses occur
                // (only for raw-type modes, like the v1).
                let mode_is_raw = self
                    .cfg
                    .modes
                    .get(&self.take_mode)
                    .map(|m| m.kind == "raw")
                    .unwrap_or(true);
                if self.cfg.streaming && mode_is_raw {
                    let lang = self
                        .cfg
                        .modes
                        .get(&self.take_mode)
                        .and_then(|m| m.language.clone())
                        .or_else(|| self.cfg.language.clone());
                    let params = StreamParams {
                        model_path: model_file_path(&self.cfg),
                        gpu: self.cfg.gpu != "cpu",
                        language: lang,
                        vocab_prompt: modes::build_initial_prompt(&self.cfg.vocabulary, ""),
                        beam_size: self.cfg.beam_size,
                        low_voice: self.cfg.low_voice,
                    };
                    let auto_paste = self.cfg.auto_paste;
                    let restore_delay = self.cfg.paste_restore_delay;
                    let emit = move |s: &str| {
                        if auto_paste {
                            if let Err(e) = paste::paste_text_with_delay(s, restore_delay) {
                                eprintln!("paste (stream): {e}");
                            }
                        }
                    };
                    let tx = self.tx.clone();
                    let ctx2 = ctx.clone();
                    // Captured now: by the time the worker reports back, the
                    // application may already be on another take.
                    let id = self.take_id;
                    let take_mode = self.take_mode.clone();
                    let done = move |res: Result<String, String>| {
                        let _ = match res {
                            Ok(text) => tx.send(Worker::Ok {
                                id,
                                text,
                                status: "stream",
                                mode: take_mode,
                            }),
                            Err(e) => tx.send(Worker::Err(id, e)),
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
                eprintln!("[state] recording started");
            }
            Err(e) => {
                eprintln!("mic: {e}");
                self.flash(ctx, DockState::Error, tr("status_mic_ko"), 1.6);
            }
        }
    }

    fn stop_and_transcribe(&mut self, ctx: &egui::Context) {
        // Streaming session: the worker transcribes the remainder then signals completion.
        if let Some(session) = self.stream.take() {
            let tail = self.recorder.stop();
            eprintln!("[state] streaming stopped ({} samples left)", tail.len());
            self.stream_join = Some(session.finish(tail));
            self.dock.state = DockState::Transcribing;
            self.dock.status_text = "\u{2026}".into();
            self.state = State::Transcribing;
            return;
        }
        let mut audio = self.recorder.stop();
        eprintln!(
            "[state] stopped -> transcribing ({} samples, {:.2}s)",
            audio.len(),
            audio.len() as f32 / 16000.0
        );
        // Soft-voice dictation: amplify a quiet take before transcription.
        if self.cfg.low_voice {
            dictata::audio::boost_quiet(&mut audio);
        }
        self.dock.state = DockState::Transcribing;
        self.dock.status_text = "\u{2026}".into();
        self.state = State::Transcribing;

        let cfg = self.cfg.clone();
        let take_key = self.take_mode.clone();
        let mode = cfg
            .modes
            .get(&take_key)
            .cloned()
            .unwrap_or_else(|| cfg.modes.values().next().cloned().expect("at least one mode"));
        let model_path = model_file_path(&cfg);
        let gpu = cfg.gpu != "cpu";
        let tr = self.transcriber.clone();
        let tx = self.tx.clone();
        let ctx2 = ctx.clone();
        // Captured now: the result is handled frames later, possibly after a
        // new take has already replaced `take_mode`.
        let id = self.take_id;

        std::thread::spawn(move || {
            if audio.len() < 4000 {
                let _ = tx.send(Worker::Ok {
                    id,
                    text: String::new(),
                    status: "raw",
                    mode: take_key,
                });
                ctx2.request_repaint();
                return;
            }
            // VAD (one-shot only): if the user enabled it, fetch the small
            // model on first use, then hand its path to the transcription.
            // Computed before locking so a download never blocks streaming.
            let vad_path = if cfg.vad {
                let p = dictata::models::vad_model_path(&cfg.model_dir);
                if !p.exists() {
                    let _ = dictata::models::download_vad(&cfg.model_dir, |_, _| {});
                }
                p.exists().then_some(p)
            } else {
                None
            };
            let mut guard = tr.lock().unwrap_or_else(|p| p.into_inner());
            if guard.is_none() {
                match Transcriber::load(&model_path, gpu) {
                    Ok(t) => *guard = Some(t),
                    Err(e) => {
                        let _ = tx.send(Worker::Err(id, e));
                        ctx2.request_repaint();
                        return;
                    }
                }
            }
            let t = guard.as_mut().unwrap();
            let lang = mode.language.clone().or_else(|| cfg.language.clone());
            let translate = mode.task == "translate";
            let prompt = modes::build_initial_prompt(&cfg.vocabulary, "");
            let prompt_opt = if prompt.is_empty() {
                None
            } else {
                Some(prompt.as_str())
            };
            match t.transcribe(&audio, lang.as_deref(), translate, prompt_opt, cfg.beam_size, vad_path.as_deref()) {
                Ok(raw) => {
                    // Whisper's own detection, so the LLM is told which
                    // language to answer in even when `language` is "auto".
                    let detected = t.detected_language();
                    eprintln!("[transcribe] detected language = {detected:?}");
                    let (text, status) = modes::apply_mode(&raw, &mode, &cfg, detected);
                    if !text.trim().is_empty() {
                        let dur = audio.len() as f64 / 16000.0;
                        history::add_entry(&cfg, &text, &take_key, lang.as_deref(), Some(dur));
                    }
                    let _ = tx.send(Worker::Ok {
                        id,
                        text,
                        status,
                        mode: take_key,
                    });
                }
                Err(e) => {
                    let _ = tx.send(Worker::Err(id, e));
                }
            }
            ctx2.request_repaint();
        });
    }

    fn handle_worker(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            // Drop the one result owed by a cancelled take, matched on its id so
            // an unrelated result still in flight is never discarded instead.
            let id = match &msg {
                Worker::Ok { id, .. } | Worker::Err(id, _) => *id,
            };
            if self.cancelled_take == Some(id) {
                self.cancelled_take = None;
                eprintln!("[worker] take {id} result dropped (cancelled)");
                continue;
            }
            match msg {
                Worker::Ok {
                    id,
                    text,
                    status,
                    mode,
                } => {
                    // Metadata only: the transcription itself is user content
                    // and must never reach a log.
                    eprintln!(
                        "[worker] take {id} status={status} mode={mode} chars={}",
                        text.chars().count()
                    );
                    if text.trim().is_empty() {
                        self.flash(ctx, DockState::Done, tr("status_empty"), 0.9);
                    } else if status == "stream" {
                        // Already pasted chunk by chunk: just history + flash.
                        history::add_entry(&self.cfg, &text, &mode, None, None);
                        self.flash(ctx, DockState::Done, tr("status_pasted"), 0.9);
                    } else {
                        let mut paste_failed = false;
                        if self.cfg.auto_paste {
                            if let Err(e) =
                                paste::paste_text_with_delay(&text, self.cfg.paste_restore_delay)
                            {
                                eprintln!("paste: {e}");
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
                Worker::Err(id, e) => {
                    eprintln!("[worker] take {id} transcription failed: {e}");
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

        // Startup notice, from the second frame onwards: on the first one the
        // dock is still being hidden, and showing it in the same frame would
        // race that viewport command.
        if self.no_activate_done
            && let Some(msg) = self.startup_notice.take()
        {
            self.flash(&ctx, DockState::Error, msg, 3.0);
        }

        // No-focus-steal applied once (the HWND exists even when hidden).
        if !self.no_activate_done {
            platform::make_no_activate(frame);
            // eframe does not honor with_visible(false) for the primary viewport:
            // force hiding the idle dock from the first frame.
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.no_activate_done = true;
        }

        // Global hotkey events. In push-to-talk the take follows the key
        // (down starts, up stops); in toggle mode only the press counts.
        for (id, st) in hotkey::poll_events() {
            if Some(id) != self.hotkey_id {
                continue;
            }
            if self.cfg.activation == "push_to_talk" {
                match st {
                    HotKeyState::Pressed => self.ptt_press(&ctx),
                    HotKeyState::Released => self.ptt_release(&ctx),
                }
            } else if st == HotKeyState::Pressed {
                self.toggle(&ctx);
            }
        }

        // Tray menu events. Drain first (owned Vec) so the loop body can
        // mutate `self` freely.
        let actions = self
            .tray
            .as_ref()
            .map(|t| t.poll_actions())
            .unwrap_or_default();
        for action in actions {
            match action {
                TrayAction::SetMode(key) => {
                    if key != self.cfg.active_mode {
                        self.cfg.active_mode = key;
                        config::save(&self.cfg);
                        if let Some(m) = self.cfg.modes.get(&self.cfg.active_mode) {
                            self.dock.mode_label = m.label.clone();
                        }
                    }
                    if let Some(tray) = &self.tray {
                        tray.set_active_mode(&self.cfg.active_mode);
                    }
                }
                TrayAction::OpenSettings => {
                    if self.settings.is_none() {
                        let mut st = SettingsState::new(self.cfg.clone());
                        st.last_window = self.last_window.clone();
                        self.settings = Some(st);
                    }
                }
                TrayAction::Quit => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
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
                // A capture stream that died (device unplugged, permission
                // revoked) must not leave the user dictating into nothing.
                if self.recorder.stream_failed() {
                    self.abort_take(&ctx, "audio stream lost", tr("status_mic_lost"), 1.6);
                }
                // Escape aborts the take. Polled here rather than through egui
                // input: the dock never holds focus, so it never sees the key
                // itself. A hold is required, because the key is read globally
                // and a tap usually belongs to the focused application.
                else if self
                    .cancel_gesture
                    .poll(platform::escape_down(), Instant::now())
                {
                    self.cancel_recording(&ctx);
                } else if secs >= self.cfg.max_record_seconds as u64 {
                    // Recording duration cap (bounds memory usage).
                    eprintln!("[state] max duration reached -> stopping");
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

/// (key, label) pairs for the tray Mode submenu, in config order.
fn tray_mode_list(cfg: &Config) -> Vec<(String, String)> {
    cfg.modes
        .iter()
        .map(|(k, m)| (k.clone(), m.label.clone()))
        .collect()
}

fn main() -> eframe::Result {
    // Redirects whisper.cpp/ggml logs to the `log` crate (silent
    // without a configured logger) instead of polluting stderr.
    whisper_rs::install_logging_hooks();

    let mut cfg = config::load();
    i18n::set_lang(&cfg.ui_lang);

    // A marker left behind means the previous run died loading that model
    // (whisper.cpp aborts on a malformed ggml file). Keeping the model selected
    // would crash again on the next take, with no console to explain it, so
    // fall back to the default. The model file itself is left alone.
    let sentinel = std::fs::read_to_string(config::model_sentinel_path()).ok();
    let recovered = config::recover_from_sentinel(sentinel.as_deref(), &cfg.model);
    if let Some(faulty) = &recovered {
        eprintln!("[startup] previous run died loading {faulty}; falling back to the default model");
        cfg.model = config::default_model();
        config::save(&cfg);
    }
    let _ = std::fs::remove_file(config::model_sentinel_path());

    eprintln!(
        "Dictata — model={}, hotkey={}, mode={}",
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
        Box::new(move |_cc| {
            let mut app = App::new(cfg);
            if recovered.is_some() {
                app.startup_notice = Some(tr("status_model_recovered"));
            }
            Ok(Box::new(app))
        }),
    )
}
