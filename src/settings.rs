//! SuperWhisper-style settings window: icon sidebar + pages,
//! dark theme, cards, toggles, shortcut chips, model library
//! (ggml catalog + download) and history.
//!
//! Modeled on the Python v1 (`freewhisper/settings_window.py`), adapted to the
//! whisper.cpp backend (ggml models, GPU mode instead of CTranslate2 compute_type).
//!
//! The window works on a **copy** of the config (`SettingsState::cfg`):
//! nothing is applied until the user clicks "Save"
//! (`RenderResult::save`). "Close" / the X return `RenderResult::close`.

use eframe::egui::{self, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Stroke, StrokeKind, Vec2};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::config::{Config, Mode};
use crate::i18n::tr;
use crate::settings_logic::{capitalize, normalize_mode_key, parse_repl, parse_vocab, pretty_token, transcribe_file};
use crate::{audio, hardware, history, llm, models, paste};

// ---------- palette (modern dark, purple accent — Raycast/Linear style) ----------
const BG: Color32 = Color32::from_rgb(0x0f, 0x0f, 0x14);
const SIDEBAR: Color32 = Color32::from_rgb(0x0a, 0x0a, 0x0e);
const CARD: Color32 = Color32::from_rgb(0x17, 0x17, 0x1e);
const CARD_BORDER: Color32 = Color32::from_rgb(0x25, 0x25, 0x2f);
const ROW_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x26);
const INPUT_BORDER: Color32 = Color32::from_rgb(0x32, 0x32, 0x3e);
const NAV_SEL: Color32 = Color32::from_rgba_premultiplied(0x33, 0x29, 0x55, 0xff);
const NAV_HOVER: Color32 = Color32::from_rgb(0x17, 0x17, 0x1e);
const TEXT: Color32 = Color32::from_rgb(0xec, 0xec, 0xf1);
const DIM: Color32 = Color32::from_rgb(0x8d, 0x8d, 0x9b);
const ACCENT: Color32 = Color32::from_rgb(0x8b, 0x7c, 0xff);
const ACCENT_SOFT: Color32 = Color32::from_rgb(0xb4, 0xab, 0xff);
const OK: Color32 = Color32::from_rgb(0x5a, 0xd2, 0x8c);
const BAD: Color32 = Color32::from_rgb(0xe0, 0x55, 0x55);
const DARK_ON_ACCENT: Color32 = Color32::from_rgb(0x12, 0x0c, 0x2e);

// (i18n key, transcription language code)
const LANGUAGES: &[(&str, Option<&str>)] = &[
    ("lang_auto", None), ("lang_fr", Some("fr")), ("lang_en", Some("en")),
    ("lang_es", Some("es")), ("lang_de", Some("de")), ("lang_it", Some("it")),
    ("lang_pt", Some("pt")), ("lang_nl", Some("nl")), ("lang_ru", Some("ru")),
    ("lang_zh", Some("zh")), ("lang_ja", Some("ja")), ("lang_ko", Some("ko")),
    ("lang_ar", Some("ar")),
];

const UI_LANGS: &[(&str, &str)] = &[("Français", "fr"), ("English", "en"), ("Español", "es")];

const GPU_MODES: &[(&str, &str)] = &[
    ("Auto", "auto"),
    ("CPU uniquement", "cpu"),
    ("Vulkan (AMD / Intel)", "vulkan"),
    ("CUDA (NVIDIA)", "cuda"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Modes,
    Vocabulary,
    Configuration,
    Sound,
    Models,
    Llm,
    History,
}

#[derive(Clone, Copy)]
enum IconKind {
    Home,
    Modes,
    Book,
    Gear,
    Sound,
    Cube,
    Chip,
    Clock,
}

// Per-page tint (duotone tile in the nav: translucent background + stroke).
const PAGES: &[(Page, &str, IconKind, Color32)] = &[
    (Page::Home, "nav_home", IconKind::Home, Color32::from_rgb(0xe0, 0x8a, 0x5a)),
    (Page::Modes, "nav_modes", IconKind::Modes, Color32::from_rgb(0x8b, 0x7c, 0xff)),
    (Page::Vocabulary, "nav_vocab", IconKind::Book, Color32::from_rgb(0x5a, 0x9b, 0xe0)),
    (Page::Configuration, "nav_config", IconKind::Gear, Color32::from_rgb(0xa0, 0x98, 0xc0)),
    (Page::Sound, "nav_sound", IconKind::Sound, Color32::from_rgb(0x5a, 0xd2, 0x8c)),
    (Page::Models, "nav_models", IconKind::Cube, Color32::from_rgb(0x4c, 0xc4, 0xbc)),
    (Page::Llm, "nav_llm", IconKind::Chip, Color32::from_rgb(0xd0, 0x6f, 0xb8)),
    (Page::History, "nav_history", IconKind::Clock, Color32::from_rgb(0xd8, 0xb4, 0x5a)),
];

struct Download {
    model: String,
    received: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    finished: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
}

pub struct SettingsState {
    /// Working copy: applied to the app only on "Save".
    pub cfg: Config,
    page: Page,
    hw: hardware::HwInfo,
    reco: String,
    mics: Vec<String>,
    selected_mode: String,
    new_mode_name: String,
    vocab_text: String,
    repl_text: String,
    capturing_hotkey: bool,
    download: Option<Download>,
    llm_status: Option<bool>,
    history: Vec<history::Entry>,
    status: String,
    /// "Reposition dock" request (consumed by the host application).
    pub position_dock_request: bool,
    /// Audio/video file transcription in progress (History page).
    filetx: Option<FileTx>,
    /// HuggingFace search (Models page).
    hf_query: String,
    hf_repos: Vec<String>,
    hf_files: Vec<models::HfFile>,
    hf_error: Option<String>,
    hf_task: Option<Arc<Mutex<Option<HfOutcome>>>>,
}

/// Result of a HuggingFace fetch (search or repo file listing).
enum HfOutcome {
    Repos(Result<Vec<String>, String>),
    Files(Result<Vec<models::HfFile>, String>),
}

/// "Transcribe a file" worker: the result arrives via `done`.
struct FileTx {
    done: Arc<Mutex<Option<Result<String, String>>>>,
}

impl SettingsState {
    pub fn new(cfg: Config) -> Self {
        let hw = hardware::detect();
        let reco = hardware::recommended(&hw, hardware::gpu_active(&hw, &cfg.gpu)).to_string();
        let mics = audio::list_input_devices();
        let selected_mode = if cfg.modes.contains_key(&cfg.active_mode) {
            cfg.active_mode.clone()
        } else {
            cfg.modes.keys().next().cloned().unwrap_or_default()
        };
        let vocab_text = cfg.vocabulary.join("\n");
        let repl_text = cfg
            .replacements
            .iter()
            .map(|(k, v)| format!("{k} = {v}"))
            .collect::<Vec<_>>()
            .join("\n");
        let history = history::read_entries(100);
        SettingsState {
            cfg,
            page: Page::Home,
            hw,
            reco,
            mics,
            selected_mode,
            new_mode_name: String::new(),
            vocab_text,
            repl_text,
            capturing_hotkey: false,
            download: None,
            llm_status: None,
            history,
            status: String::new(),
            position_dock_request: false,
            filetx: None,
            hf_query: String::new(),
            hf_repos: Vec::new(),
            hf_files: Vec::new(),
            hf_error: None,
            hf_task: None,
        }
    }

    /// Sets a status message (shown in the footer).
    pub fn set_status(&mut self, msg: &str) {
        self.status = msg.to_string();
    }
}

/// What the render reports back to the host application.
#[derive(Default)]
pub struct RenderResult {
    pub save: bool,
    pub close: bool,
}

// ---------- theme ----------
pub fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = BG;
    visuals.extreme_bg_color = Color32::from_rgb(0x14, 0x14, 0x1b); // field background
    visuals.override_text_color = Some(TEXT);
    visuals.widgets.noninteractive.bg_fill = CARD;
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(5);
    // Flat SuperWhisper-style buttons: medium gray, no border.
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(0x35, 0x35, 0x3e);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(0x35, 0x35, 0x3e);
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
    visuals.widgets.inactive.corner_radius = CornerRadius::same(5);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x42, 0x42, 0x4d);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x42, 0x42, 0x4d);
    visuals.widgets.hovered.bg_stroke = Stroke::NONE;
    visuals.widgets.hovered.corner_radius = CornerRadius::same(5);
    visuals.widgets.hovered.expansion = 0.0;
    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.corner_radius = CornerRadius::same(5);
    visuals.widgets.open.corner_radius = CornerRadius::same(5);
    visuals.selection.bg_fill = ACCENT.gamma_multiply(0.45);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6], blur: 18, spread: 0,
        color: Color32::from_black_alpha(120),
    };
    ctx.set_visuals(visuals);
    // Generous button/combo padding (SuperWhisper look).
    ctx.style_mut(|s| s.spacing.button_padding = egui::vec2(12.0, 5.0));
}

// ---------- small widgets ----------

/// iOS-style toggle switch.
fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let size = egui::vec2(40.0, 22.0);
    let (rect, mut resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let how = ui.ctx().animate_bool(resp.id, *on);
    let off = Color32::from_rgb(0x44, 0x44, 0x4c);
    let bg = off.lerp_to_gamma(ACCENT, how);
    let r = rect.height() / 2.0;
    ui.painter().rect_filled(rect, r, bg);
    let cx = egui::lerp((rect.left() + r)..=(rect.right() - r), how);
    ui.painter()
        .circle_filled(egui::pos2(cx, rect.center().y), r - 3.0, Color32::WHITE);
    resp
}

/// Card (background + rounded border). Optional title as a header.
fn card<R>(ui: &mut egui::Ui, title: Option<&str>, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::NONE
        .fill(CARD)
        .stroke(Stroke::new(1.0, CARD_BORDER))
        .corner_radius(14.0)
        .inner_margin(18.0)
        .shadow(egui::epaint::Shadow {
            offset: [0, 4], blur: 16, spread: 0,
            color: Color32::from_black_alpha(70),
        })
        .show(ui, |ui| {
            ui.style_mut().spacing.item_spacing.y = 10.0;
            if let Some(t) = title {
                ui.label(RichText::new(t).size(12.0).color(ACCENT_SOFT).strong());
                ui.add_space(4.0);
            }
            add(ui)
        })
        .inner
}

/// "label (+ hint) ............ widget" line, right-aligned,
/// laid on a full-width row with a subtle background.
fn row(ui: &mut egui::Ui, label: &str, hint: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(ROW_BG.gamma_multiply(0.55))
        .corner_radius(8.0)
        .inner_margin(egui::Margin { left: 12, right: 12, top: 9, bottom: 9 })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            row_inner(ui, label, hint, add);
        });
}

fn row_inner(ui: &mut egui::Ui, label: &str, hint: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        // Reserve room for the widget on the right: the hint wraps
        // instead of going under the combo.
        let left_w = (ui.available_width() - 320.0).max(140.0);
        ui.vertical(|ui| {
            ui.set_width(left_w);
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(RichText::new(label).strong());
            if !hint.is_empty() {
                ui.add(egui::Label::new(RichText::new(hint).size(11.0).color(DIM)).wrap());
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| add(ui));
    });
}

/// Draws a small monochrome (line) icon inside `r`.
fn draw_icon(p: &egui::Painter, r: Rect, kind: IconKind, col: Color32) {
    let s = r.width();
    let o = r.min;
    let pt = |fx: f32, fy: f32| Pos2::new(o.x + fx * s, o.y + fy * s);
    let st = Stroke::new(1.6, col);
    let line = |a: Pos2, b: Pos2| p.line_segment([a, b], st);
    match kind {
        IconKind::Home => {
            line(pt(0.12, 0.5), pt(0.5, 0.12));
            line(pt(0.5, 0.12), pt(0.88, 0.5));
            p.rect_stroke(
                Rect::from_min_max(pt(0.27, 0.5), pt(0.73, 0.9)),
                CornerRadius::same(2),
                st,
                StrokeKind::Inside,
            );
        }
        IconKind::Modes => {
            for (i, y) in [0.28f32, 0.5, 0.72].iter().enumerate() {
                line(pt(0.42, *y), pt(0.85, *y));
                let _ = i;
                p.circle_filled(pt(0.22, *y), 1.7, col);
            }
        }
        IconKind::Book => {
            p.rect_stroke(
                Rect::from_min_max(pt(0.22, 0.18), pt(0.78, 0.82)),
                CornerRadius::same(2),
                st,
                StrokeKind::Inside,
            );
            line(pt(0.5, 0.18), pt(0.5, 0.82));
        }
        IconKind::Gear => {
            p.circle_stroke(pt(0.5, 0.5), s * 0.18, st);
            for k in 0..8 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4;
                line(
                    pt(0.5 + a.cos() * 0.3, 0.5 + a.sin() * 0.3),
                    pt(0.5 + a.cos() * 0.42, 0.5 + a.sin() * 0.42),
                );
            }
        }
        IconKind::Sound => {
            let tri = vec![
                pt(0.2, 0.4), pt(0.37, 0.4), pt(0.52, 0.25),
                pt(0.52, 0.75), pt(0.37, 0.6), pt(0.2, 0.6),
            ];
            p.add(egui::Shape::convex_polygon(tri, col, Stroke::NONE));
            p.circle_stroke(pt(0.55, 0.5), s * 0.22, st);
        }
        IconKind::Cube => {
            for y in [0.2f32, 0.45, 0.7] {
                p.rect_stroke(
                    Rect::from_min_max(pt(0.22, y), pt(0.78, y + 0.16)),
                    CornerRadius::same(2),
                    st,
                    StrokeKind::Inside,
                );
            }
        }
        IconKind::Chip => {
            p.rect_stroke(
                Rect::from_min_max(pt(0.3, 0.3), pt(0.7, 0.7)),
                CornerRadius::same(2),
                st,
                StrokeKind::Inside,
            );
            for off in [0.42f32, 0.58] {
                line(pt(off, 0.12), pt(off, 0.3));
                line(pt(off, 0.7), pt(off, 0.88));
                line(pt(0.12, off), pt(0.3, off));
                line(pt(0.7, off), pt(0.88, off));
            }
        }
        IconKind::Clock => {
            p.circle_stroke(pt(0.5, 0.5), s * 0.34, st);
            line(pt(0.5, 0.5), pt(0.5, 0.28));
            line(pt(0.5, 0.5), pt(0.68, 0.5));
        }
    }
}

/// Navigation item (icon + label), highlighted when selected.
fn nav_item(ui: &mut egui::Ui, label: &str, icon: IconKind, tint: Color32, selected: bool) -> egui::Response {
    let h = 36.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::click());
    let bg = if selected {
        NAV_SEL
    } else if resp.hovered() {
        NAV_HOVER
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        ui.painter().rect_filled(rect, 6.0, bg);
    }
    if selected {
        // Accent bar on the left of the selected pill.
        let bar = Rect::from_min_size(Pos2::new(rect.left() + 2.0, rect.center().y - 8.0), Vec2::new(3.0, 16.0));
        ui.painter().rect_filled(bar, 2.0, ACCENT);
    }
    // Duotone tile: translucent tinted background + line icon in the tint
    // (more subtle than SuperWhisper's solid badges).
    let tile = Rect::from_min_size(Pos2::new(rect.left() + 10.0, rect.center().y - 11.0), Vec2::splat(22.0));
    let tile_alpha = if selected { 0.30 } else { 0.16 };
    ui.painter().rect_filled(tile, 6.0, tint.gamma_multiply(tile_alpha));
    let icon_col = if selected { tint.lerp_to_gamma(Color32::WHITE, 0.25) } else { tint.gamma_multiply(0.85) };
    let icon_rect = tile.shrink(3.0);
    draw_icon(ui.painter(), icon_rect, icon, icon_col);
    let txt_col = if selected { Color32::WHITE } else { DIM };
    ui.painter().text(
        Pos2::new(rect.left() + 42.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        FontId::proportional(14.0),
        txt_col,
    );
    resp
}

/// Read-only key chips ("Keyboard shortcuts" list).
fn key_chips(ui: &mut egui::Ui, tokens: &[&str]) {
    let fid = FontId::proportional(12.0);
    for tok in tokens.iter().rev() {
        let label = pretty_token(tok);
        let galley = ui.painter().layout_no_wrap(label.clone(), fid.clone(), TEXT);
        let (chip, _) = ui.allocate_exact_size(Vec2::new(galley.size().x + 16.0, 22.0), egui::Sense::hover());
        ui.painter().rect_filled(chip, 5.0, Color32::from_rgb(0x34, 0x34, 0x3c));
        ui.painter().rect_stroke(chip, CornerRadius::same(5), Stroke::new(1.0, INPUT_BORDER), StrokeKind::Inside);
        ui.painter().text(chip.center(), egui::Align2::CENTER_CENTER, label, fid.clone(), TEXT);
    }
}

/// Shortcut field: Ctrl/Alt/… chips; click -> captures a combination.
/// Returns `true` if the combination changed.
fn shortcut_edit(ui: &mut egui::Ui, st: &mut SettingsState) -> bool {
    let mut changed = false;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(320.0, 40.0), egui::Sense::click());
    ui.painter().rect_filled(rect, 5.0, ROW_BG);
    ui.painter()
        .rect_stroke(rect, CornerRadius::same(5), Stroke::new(1.0, if st.capturing_hotkey { ACCENT } else { CARD_BORDER }), StrokeKind::Inside);

    if resp.clicked() {
        st.capturing_hotkey = true;
    }

    if st.capturing_hotkey {
        ui.painter().text(
            Pos2::new(rect.left() + 12.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            tr("shortcut_press_key"),
            FontId::proportional(13.0),
            ACCENT,
        );
        // Captures the first non-modifier key.
        if let Some(combo) = ui.input(|i| capture_combo(i)) {
            if combo.parse::<global_hotkey::hotkey::HotKey>().is_ok() {
                if combo != st.cfg.hotkey {
                    st.cfg.hotkey = combo;
                    changed = true;
                }
                st.capturing_hotkey = false;
            }
        }
        // Escape cancels.
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            st.capturing_hotkey = false;
        }
        ui.ctx().request_repaint();
    } else {
        // Token chips.
        let mut x = rect.left() + 10.0;
        let fid = FontId::proportional(12.0);
        for tok in st.cfg.hotkey.split('+').filter(|t| !t.is_empty()) {
            let label = pretty_token(tok);
            let galley = ui.painter().layout_no_wrap(label.clone(), fid.clone(), TEXT);
            let w = galley.size().x + 16.0;
            let chip = Rect::from_min_size(Pos2::new(x, rect.center().y - 11.0), Vec2::new(w, 22.0));
            ui.painter().rect_filled(chip, 5.0, Color32::from_rgb(0x34, 0x34, 0x3c));
            ui.painter().rect_stroke(chip, CornerRadius::same(5), Stroke::new(1.0, INPUT_BORDER), StrokeKind::Inside);
            ui.painter().text(chip.center(), egui::Align2::CENTER_CENTER, label, fid.clone(), TEXT);
            x += w + 6.0;
        }
        ui.painter().text(
            Pos2::new(rect.right() - 10.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            tr("shortcut_click_edit"),
            FontId::proportional(11.0),
            DIM,
        );
    }
    changed
}

/// Reads the keyboard state and returns a "mod+mod+key" combination if a
/// main key is pressed.
fn capture_combo(i: &egui::InputState) -> Option<String> {
    for ev in &i.events {
        if let egui::Event::Key { key, pressed: true, modifiers, .. } = ev {
            if matches!(
                key,
                egui::Key::Escape
            ) {
                continue;
            }
            let mut parts: Vec<String> = Vec::new();
            if modifiers.ctrl || modifiers.command {
                parts.push("ctrl".into());
            }
            if modifiers.alt {
                parts.push("alt".into());
            }
            if modifiers.shift {
                parts.push("shift".into());
            }
            if modifiers.mac_cmd {
                parts.push("super".into());
            }
            parts.push(key_token(*key));
            return Some(parts.join("+"));
        }
    }
    None
}

/// Key name -> token accepted by global-hotkey (lowercase).
fn key_token(key: egui::Key) -> String {
    use egui::Key::*;
    match key {
        Space => "space".into(),
        Enter => "enter".into(),
        Tab => "tab".into(),
        Backspace => "backspace".into(),
        Delete => "delete".into(),
        Insert => "insert".into(),
        Home => "home".into(),
        End => "end".into(),
        PageUp => "pageup".into(),
        PageDown => "pagedown".into(),
        ArrowUp => "up".into(),
        ArrowDown => "down".into(),
        ArrowLeft => "left".into(),
        ArrowRight => "right".into(),
        Num0 => "0".into(), Num1 => "1".into(), Num2 => "2".into(), Num3 => "3".into(),
        Num4 => "4".into(), Num5 => "5".into(), Num6 => "6".into(), Num7 => "7".into(),
        Num8 => "8".into(), Num9 => "9".into(),
        other => format!("{other:?}").to_lowercase(),
    }
}


// ---------- main render ----------
pub fn render(ui: &mut egui::Ui, st: &mut SettingsState) -> RenderResult {
    let mut res = RenderResult::default();
    let ctx = ui.ctx().clone();
    // The UI language follows the working copy: instant change.
    crate::i18n::set_lang(&st.cfg.ui_lang);

    // Sidebar (full height).
    egui::SidePanel::left("nav")
        .exact_width(210.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(SIDEBAR).inner_margin(egui::Margin {
            left: 12, right: 8, top: 16, bottom: 12,
        }))
        .show_inside(ui, |ui| {
            ui.label(RichText::new("Dictata").size(16.0).strong().color(TEXT));
            ui.label(RichText::new(tr("brand_sub")).size(11.0).color(DIM));
            ui.add_space(12.0);
            for (page, label, icon, tint) in PAGES {
                if nav_item(ui, tr(label), *icon, *tint, st.page == *page).clicked() {
                    st.page = *page;
                }
                ui.add_space(2.0);
            }
        });

    // Footer (status + Close / Save), below the content.
    egui::TopBottomPanel::bottom("footer")
        .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin {
            left: 20, right: 20, top: 8, bottom: 12,
        }))
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&st.status).color(ACCENT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let save = egui::Button::new(RichText::new(tr("btn_save")).color(DARK_ON_ACCENT).strong())
                        .fill(ACCENT)
                        .corner_radius(5.0);
                    if ui.add(save).clicked() {
                        res.save = true;
                    }
                    if ui.button(tr("btn_close")).clicked() {
                        res.close = true;
                    }
                });
            });
        });

    // Page content.
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin {
            left: 28, right: 28, top: 24, bottom: 16,
        }))
        .show_inside(ui, |ui| {
            ui.style_mut().spacing.item_spacing.y = 14.0;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match st.page {
                    Page::Home => page_home(ui, st),
                    Page::Modes => page_modes(ui, st),
                    Page::Vocabulary => page_vocab(ui, st),
                    Page::Configuration => page_config(ui, st),
                    Page::Sound => page_sound(ui, st),
                    Page::Models => page_models(ui, st, &ctx),
                    Page::Llm => page_llm(ui, st),
                    Page::History => page_history(ui, st),
                });
        });

    // Applies the text editors (vocab/replacements) to the working copy.
    if res.save {
        st.cfg.vocabulary = parse_vocab(&st.vocab_text);
        st.cfg.replacements = parse_repl(&st.repl_text);
        if !st.cfg.modes.contains_key(&st.cfg.active_mode) {
            st.cfg.active_mode = "raw".into();
        }
    }
    res
}

fn h1(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).size(22.0).strong().color(TEXT));
    ui.add_space(4.0);
}

// ---------- Home ----------
fn page_home(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_home"));
    card(ui, Some(tr("home_hw")), |ui| {
        ui.label(&st.hw.cpu_name);
        ui.label(hardware::summary(&st.hw));
        ui.horizontal(|ui| {
            ui.label(tr("home_reco"));
            ui.label(RichText::new(&st.reco).color(ACCENT).strong());
        });
    });
    card(ui, Some(tr("home_state")), |ui| {
        let mode = st
            .cfg
            .modes
            .get(&st.cfg.active_mode)
            .map(|m| m.label.clone())
            .unwrap_or_else(|| st.cfg.active_mode.clone());
        ui.label(RichText::new(format!("{} {}", tr("home_active_model"), st.cfg.model)));
        ui.label(RichText::new(format!("{} {mode}", tr("home_active_mode"))));
        ui.label(RichText::new(format!("{} {}", tr("home_models_folder"), st.cfg.model_dir)).color(DIM));
        if ui.button(tr("home_open_folder")).clicked() {
            let _ = std::fs::create_dir_all(&st.cfg.model_dir);
            let _ = open_path(&st.cfg.model_dir);
        }
    });
    card(ui, Some(tr("home_howto_title")), |ui| {
        ui.label(
            RichText::new(tr("home_howto_text"))
            .color(DIM),
        );
    });
}

#[cfg(windows)]
fn open_path(path: &str) -> std::io::Result<()> {
    std::process::Command::new("explorer").arg(path).spawn().map(|_| ())
}
#[cfg(not(windows))]
fn open_path(path: &str) -> std::io::Result<()> {
    std::process::Command::new("xdg-open").arg(path).spawn().map(|_| ())
}

// ---------- Modes ----------
fn page_modes(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_modes"));

    // Active mode.
    card(ui, None, |ui| {
        row(ui, tr("modes_active"), tr("modes_active_hint"), |ui| {
            let cur = st
                .cfg
                .modes
                .get(&st.cfg.active_mode)
                .map(|m| m.label.clone())
                .unwrap_or_else(|| st.cfg.active_mode.clone());
            let opts: Vec<(String, String)> =
                st.cfg.modes.iter().map(|(k, m)| (k.clone(), m.label.clone())).collect();
            egui::ComboBox::from_id_salt("active_mode")
                .selected_text(cur)
                .width(220.0)
                .show_ui(ui, |ui| {
                    for (k, label) in opts {
                        ui.selectable_value(&mut st.cfg.active_mode, k, label);
                    }
                });
        });
    });

    // List + form.
    card(ui, None, |ui| {
        ui.horizontal_top(|ui| {
            // List column.
            ui.vertical(|ui| {
                ui.set_width(200.0);
                let keys: Vec<String> = st.cfg.modes.keys().cloned().collect();
                egui::Frame::NONE
                    .fill(Color32::from_rgb(0x2a, 0x2a, 0x31))
                    .stroke(Stroke::new(1.0, INPUT_BORDER))
                    .corner_radius(8.0)
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.set_width(188.0);
                        for k in &keys {
                            ui.selectable_value(&mut st.selected_mode, k.clone(), k.as_str());
                        }
                    });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut st.new_mode_name).hint_text(tr("modes_key_ph")).desired_width(96.0));
                    if ui.button(tr("modes_add")).clicked() {
                        if let Some(name) = normalize_mode_key(&st.new_mode_name)
                            .filter(|n| !st.cfg.modes.contains_key(n))
                        {
                            st.cfg.modes.insert(
                                name.clone(),
                                Mode {
                                    label: capitalize(&name),
                                    kind: "llm".into(),
                                    task: "transcribe".into(),
                                    language: None,
                                    prompt: String::new(),
                                },
                            );
                            st.selected_mode = name;
                            st.new_mode_name.clear();
                        }
                    }
                });
                if ui.button(tr("modes_del")).clicked() && st.selected_mode != "raw" {
                    st.cfg.modes.shift_remove(&st.selected_mode);
                    st.selected_mode = st.cfg.modes.keys().next().cloned().unwrap_or_default();
                }
            });

            ui.add_space(14.0);

            // Form column.
            let sel = st.selected_mode.clone();
            ui.vertical(|ui| {
                if let Some(m) = st.cfg.modes.get_mut(&sel) {
                    egui::Grid::new("mode_form")
                        .num_columns(2)
                        .spacing([12.0, 10.0])
                        .show(ui, |ui| {
                            ui.label(tr("modes_label"));
                            ui.add(egui::TextEdit::singleline(&mut m.label).desired_width(f32::INFINITY));
                            ui.end_row();

                            ui.label(tr("modes_type"));
                            egui::ComboBox::from_id_salt("mode_type")
                                .selected_text(m.kind.clone())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut m.kind, "raw".into(), "raw");
                                    ui.selectable_value(&mut m.kind, "llm".into(), "llm");
                                });
                            ui.end_row();

                            ui.label(tr("modes_task"));
                            egui::ComboBox::from_id_salt("mode_task")
                                .selected_text(m.task.clone())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut m.task, "transcribe".into(), "transcribe");
                                    ui.selectable_value(&mut m.task, "translate".into(), "translate");
                                });
                            ui.end_row();
                        });
                    ui.add_space(8.0);
                    ui.label(RichText::new(tr("modes_prompt_ph")).color(DIM).size(12.0));
                    ui.add(
                        egui::TextEdit::multiline(&mut m.prompt)
                            .desired_rows(6)
                            .desired_width(f32::INFINITY),
                    );
                } else {
                    ui.label(RichText::new(tr("modes_none_sel")).color(DIM));
                }
            });
        });
    });
}


// ---------- Vocabulary ----------
fn page_vocab(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_vocab"));
    card(ui, Some(tr("vocab_title")), |ui| {
        ui.label(RichText::new(tr("vocab_hint")).color(DIM).size(12.0));
        ui.add(
            egui::TextEdit::multiline(&mut st.vocab_text)
                .desired_rows(6)
                .desired_width(f32::INFINITY),
        );
    });
    card(ui, Some(tr("repl_title")), |ui| {
        ui.label(RichText::new(tr("repl_hint")).color(DIM).size(12.0));
        ui.add(
            egui::TextEdit::multiline(&mut st.repl_text)
                .desired_rows(5)
                .desired_width(f32::INFINITY),
        );
    });
}

// ---------- Configuration ----------
fn page_config(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_config"));
    card(ui, Some(tr("cfg_card")), |ui| {
        row(ui, tr("cfg_hotkey"), tr("cfg_hotkey_hint"), |ui| {
            shortcut_edit(ui, st);
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_cancel"), tr("cfg_cancel_hint"), |ui| {
            key_chips(ui, &["esc"]);
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_activation"), "", |ui| {
            let cur = if st.cfg.activation == "push_to_talk" {
                tr("cfg_activation_ptt")
            } else {
                tr("cfg_activation_toggle")
            };
            egui::ComboBox::from_id_salt("activation")
                .selected_text(cur)
                .width(240.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.cfg.activation, "toggle".into(), tr("cfg_activation_toggle"));
                    ui.selectable_value(
                        &mut st.cfg.activation,
                        "push_to_talk".into(),
                        tr("cfg_activation_ptt"),
                    );
                });
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_autopaste"), tr("cfg_autopaste_hint"), |ui| {
            toggle(ui, &mut st.cfg.auto_paste);
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_streaming"), tr("cfg_streaming_hint"), |ui| {
            toggle(ui, &mut st.cfg.streaming);
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_ui_lang"), tr("cfg_ui_lang_hint"), |ui| {
            let cur = UI_LANGS
                .iter()
                .find(|(_, c)| *c == st.cfg.ui_lang)
                .map(|(l, _)| *l)
                .unwrap_or("Français");
            egui::ComboBox::from_id_salt("ui_lang")
                .selected_text(cur)
                .width(160.0)
                .show_ui(ui, |ui| {
                    for (label, code) in UI_LANGS {
                        ui.selectable_value(&mut st.cfg.ui_lang, code.to_string(), *label);
                    }
                });
        });
    });
    ui.add_space(10.0);
    card(ui, Some(tr("cfg_dock_card")), |ui| {
        row(ui, tr("cfg_dock_size"), "", |ui| {
            let mut pct = (st.cfg.dock_scale * 100.0).round() as i32;
            if ui
                .add(egui::DragValue::new(&mut pct).range(70..=160).suffix(" %").speed(1))
                .changed()
            {
                st.cfg.dock_scale = pct as f32 / 100.0;
            }
        });
        ui.add_space(4.0);
        row(ui, tr("cfg_dock_opacity"), "", |ui| {
            let mut pct = (st.cfg.dock_opacity * 100.0).round() as i32;
            if ui
                .add(egui::DragValue::new(&mut pct).range(40..=100).suffix(" %").speed(1))
                .changed()
            {
                st.cfg.dock_opacity = pct as f32 / 100.0;
            }
        });
        ui.add_space(8.0);
        ui.label(RichText::new(tr("cfg_dock_position_hint")).color(DIM).size(12.0));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button(tr("cfg_dock_position_btn")).clicked() {
                st.position_dock_request = true;
            }
            if ui.button(tr("cfg_dock_reset_btn")).clicked() {
                st.cfg.dock_pos = None;
                let msg = tr("cfg_dock_reset_done").to_string();
                st.set_status(&msg);
            }
        });
    });
}

// ---------- Sound ----------
fn page_sound(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_sound"));
    card(ui, Some(tr("sound_card")), |ui| {
        row(ui, tr("source_label"), tr("source_hint"), |ui| {
            let sources = [
                ("mic", tr("source_mic")),
                ("system", tr("source_system")),
                ("mix", tr("source_mix")),
            ];
            let cur = sources
                .iter()
                .find(|(c, _)| *c == st.cfg.audio_source)
                .map(|(_, l)| *l)
                .unwrap_or(tr("source_mic"));
            egui::ComboBox::from_id_salt("audio_source")
                .selected_text(cur)
                .width(280.0)
                .show_ui(ui, |ui| {
                    for (code, label) in sources {
                        ui.selectable_value(&mut st.cfg.audio_source, code.to_string(), label);
                    }
                });
        });
        ui.add_space(4.0);
        row(ui, tr("sound_mic"), "", |ui| {
            let cur = st.cfg.input_device.clone().unwrap_or_else(|| tr("sound_default_mic").into());
            let mics = st.mics.clone();
            egui::ComboBox::from_id_salt("mic")
                .selected_text(cur)
                .width(280.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.cfg.input_device, None, tr("sound_default_mic"));
                    for m in mics {
                        ui.selectable_value(&mut st.cfg.input_device, Some(m.clone()), m);
                    }
                });
        });
        ui.add_space(4.0);
        row(ui, tr("sound_beeps"), tr("sound_beeps_hint"), |ui| {
            toggle(ui, &mut st.cfg.play_sounds);
        });
    });
}

// ---------- Models ----------
fn page_models(ui: &mut egui::Ui, st: &mut SettingsState, ctx: &egui::Context) {
    h1(ui, tr("nav_models"));

    // Download in progress.
    if let Some(dl) = &st.download {
        let recv = dl.received.load(Ordering::Relaxed);
        let total = dl.total.load(Ordering::Relaxed);
        let done = dl.finished.load(Ordering::Relaxed);
        let err = dl.error.lock().unwrap().clone();
        let name = dl.model.clone();
        card(ui, None, |ui| {
            ui.label(format!("{} {name}", tr("models_downloading")));
            let frac = if total > 0 { recv as f32 / total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            if done {
                if let Some(e) = &err {
                    ui.colored_label(BAD, e);
                } else {
                    ui.colored_label(OK, tr("models_done"));
                }
            }
        });
        if done {
            st.download = None;
            let msg = match &err {
                Some(e) => format!("{} {e}", tr("models_dl_error")),
                None => tr("models_installed_ok").to_string(),
            };
            st.set_status(&msg);
        } else {
            ctx.request_repaint();
        }
    }

    // Model parameters.
    card(ui, Some(tr("models_params")), |ui| {
        row(ui, tr("models_default_lang"), "", |ui| {
            let cur = LANGUAGES
                .iter()
                .find(|(_, c)| c.map(String::from) == st.cfg.language)
                .map(|(l, _)| tr(l))
                .unwrap_or(tr("lang_auto"));
            egui::ComboBox::from_id_salt("lang")
                .selected_text(cur)
                .width(220.0)
                .show_ui(ui, |ui| {
                    for (label, code) in LANGUAGES {
                        ui.selectable_value(&mut st.cfg.language, code.map(String::from), tr(label));
                    }
                });
        });
        ui.add_space(4.0);
        row(ui, tr("models_beam"), tr("models_beam_hint"), |ui| {
            ui.add(egui::DragValue::new(&mut st.cfg.beam_size).range(1..=10).speed(1));
        });
        ui.add_space(4.0);
        row(ui, tr("models_accel"), tr("models_accel_hint"), |ui| {
            let cur = GPU_MODES
                .iter()
                .find(|(_, v)| *v == st.cfg.gpu)
                .map(|(l, _)| *l)
                .unwrap_or("Auto");
            egui::ComboBox::from_id_salt("gpu")
                .selected_text(cur)
                .width(220.0)
                .show_ui(ui, |ui| {
                    for (label, value) in GPU_MODES {
                        ui.selectable_value(&mut st.cfg.gpu, value.to_string(), *label);
                    }
                });
        });
    });

    // Installed models.
    let dir = st.cfg.model_dir.clone();
    let installed = models::list_installed(&dir);
    card(ui, Some(tr("models_installed")), |ui| {
        if installed.is_empty() {
            ui.label(RichText::new(tr("models_none")).color(DIM));
        }
        for name in &installed {
            let active = st.cfg.model == *name;
            model_row(ui, st, name, &format!("{name}"), "", active, true, None);
        }
    });

    // Recommended catalog.
    let reco = st.reco.clone();
    card(ui, Some(tr("models_lib")), |ui| {
        ui.label(RichText::new(tr("models_reco_hint")).color(DIM).size(12.0));
        ui.add_space(2.0);
        for c in models::CATALOG {
            let active = st.cfg.model == c.name;
            let inst = models::is_installed(&dir, c.name);
            let rating = hardware::rate(c.name, &st.hw, hardware::gpu_active(&st.hw, &st.cfg.gpu));
            let badge = if !rating.label.is_empty() {
                Some((rating.label.clone(), rating_color(rating.level)))
            } else {
                None
            };
            let _ = &reco;
            let sub = format!("{} Mo \u{00b7} {}", c.size_mb, rating.detail);
            model_row(ui, st, c.name, c.label, &sub, active, inst, badge);
        }
    });

    // HuggingFace search.
    if let Some(task) = &st.hf_task {
        let outcome = task.lock().unwrap().take();
        if let Some(outcome) = outcome {
            st.hf_task = None;
            st.hf_error = None;
            match outcome {
                HfOutcome::Repos(Ok(r)) => {
                    st.hf_files.clear();
                    st.hf_repos = r;
                }
                HfOutcome::Files(Ok(f)) => {
                    st.hf_repos.clear();
                    st.hf_files = f;
                }
                HfOutcome::Repos(Err(e)) | HfOutcome::Files(Err(e)) => st.hf_error = Some(e),
            }
        }
    }
    card(ui, Some(tr("hf_card")), |ui| {
        ui.label(RichText::new(tr("hf_hint")).color(DIM).size(12.0));
        ui.add_space(4.0);
        let mut go = false;
        ui.horizontal(|ui| {
            let resp = ui.add(egui::TextEdit::singleline(&mut st.hf_query).desired_width(320.0));
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                go = true;
            }
            if ui.button(tr("hf_search")).clicked() {
                go = true;
            }
            if st.hf_task.is_some() {
                ui.label(RichText::new(tr("hf_searching")).color(DIM));
            }
        });
        if go && !st.hf_query.trim().is_empty() && st.hf_task.is_none() {
            st.hf_error = None;
            match models::parse_hf_query(&st.hf_query) {
                models::HfQuery::FileUrl(url, fname) => {
                    if st.download.is_none() {
                        st.download = Some(spawn_download_url(&dir, &url, &fname, &fname, ui.ctx().clone()));
                    }
                }
                q => st.hf_task = Some(spawn_hf(q, ui.ctx().clone())),
            }
        }
        if let Some(e) = &st.hf_error {
            ui.colored_label(BAD, e);
        }
        // Search results: repos to browse.
        if !st.hf_repos.is_empty() {
            ui.add_space(4.0);
            let repos = st.hf_repos.clone();
            for repo in &repos {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(repo).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if st.hf_task.is_none() && ui.button(tr("hf_browse")).clicked() {
                            st.hf_task =
                                Some(spawn_hf(models::HfQuery::Repo(repo.clone()), ui.ctx().clone()));
                        }
                    });
                });
            }
        }
        // Repo files: downloadable .bin.
        if !st.hf_files.is_empty() {
            ui.add_space(4.0);
            let files = st.hf_files.clone();
            ui.label(RichText::new(&files[0].repo).color(DIM).size(12.0));
            for f in &files {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&f.fname).strong());
                    if let Some(s) = f.size {
                        ui.label(RichText::new(format!("{} Mo", s / 1_000_000)).color(DIM).size(11.0));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if models::is_installed(&dir, &f.fname) {
                            ui.label(RichText::new(tr("models_done")).color(OK));
                        } else if st.download.is_none() {
                            if ui.button(tr("models_download")).clicked() {
                                st.download = Some(spawn_download_url(
                                    &dir,
                                    &f.url(),
                                    &f.fname,
                                    &f.fname,
                                    ui.ctx().clone(),
                                ));
                            }
                        } else {
                            ui.add_enabled(false, egui::Button::new(tr("models_download")));
                        }
                    });
                });
            }
        }
    });
}

fn rating_color(level: hardware::Level) -> Color32 {
    match level {
        hardware::Level::Ideal => ACCENT,
        hardware::Level::Good => OK,
        hardware::Level::Heavy => Color32::from_rgb(0xe0, 0xa2, 0x3c),
        hardware::Level::TooBig => BAD,
        hardware::Level::Unknown => DIM,
    }
}

#[allow(clippy::too_many_arguments)]
fn model_row(
    ui: &mut egui::Ui,
    st: &mut SettingsState,
    model_name: &str,
    title: &str,
    subtitle: &str,
    active: bool,
    installed: bool,
    badge: Option<(String, Color32)>,
) {
    egui::Frame::NONE
        .fill(ROW_BG)
        .stroke(Stroke::new(1.0, CARD_BORDER))
        .corner_radius(8.0)
        .inner_margin(egui::Margin { left: 12, right: 12, top: 8, bottom: 8 })
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(title).strong());
                        if let Some((label, col)) = &badge {
                            ui.label(RichText::new(label).size(11.0).color(*col));
                        }
                    });
                    if !subtitle.is_empty() {
                        ui.label(RichText::new(subtitle).size(11.0).color(DIM));
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if active {
                        ui.label(RichText::new(tr("models_active")).color(ACCENT).strong());
                    } else if installed {
                        if ui.button(tr("models_delete")).clicked() {
                            let msg = match models::delete(&st.cfg.model_dir, model_name) {
                                Ok(()) => tr("models_deleted").to_string(),
                                Err(e) => format!("{} {e}", tr("models_del_error")),
                            };
                            st.set_status(&msg);
                        }
                        let b = egui::Button::new(RichText::new(tr("models_use")).color(DARK_ON_ACCENT).strong())
                            .fill(ACCENT)
                            .corner_radius(5.0);
                        if ui.add(b).clicked() {
                            st.cfg.model = model_name.to_string();
                        }
                    } else if st.download.is_none() {
                        if ui.button(tr("models_download")).clicked() {
                            st.download = Some(spawn_download(&st.cfg.model_dir, model_name, ui.ctx().clone()));
                        }
                    } else {
                        ui.add_enabled(false, egui::Button::new(tr("models_download")));
                    }
                });
            });
        });
    ui.add_space(8.0);
}

// ---------- LLM ----------
fn page_llm(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_llm"));
    card(ui, Some(tr("llm_card")), |ui| {
        row(ui, tr("llm_enable"), tr("llm_enable_hint"), |ui| {
            toggle(ui, &mut st.cfg.llm.enabled);
        });
        ui.add_space(4.0);
        row(ui, tr("llm_url"), tr("llm_url_hint"), |ui| {
            ui.add(egui::TextEdit::singleline(&mut st.cfg.llm.base_url).desired_width(280.0));
        });
        ui.add_space(4.0);
        row(ui, tr("llm_model"), "", |ui| {
            ui.add(egui::TextEdit::singleline(&mut st.cfg.llm.model).desired_width(280.0));
        });
        ui.add_space(4.0);
        row(ui, tr("llm_temp"), "", |ui| {
            ui.add(egui::DragValue::new(&mut st.cfg.llm.temperature).speed(0.05).range(0.0..=1.5));
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(tr("llm_test")).clicked() {
                st.llm_status = Some(llm::is_available(&st.cfg.llm));
            }
            match st.llm_status {
                Some(true) => {
                    ui.label(RichText::new(tr("llm_ok")).color(OK));
                }
                Some(false) => {
                    ui.label(RichText::new(tr("llm_ko")).color(BAD));
                }
                None => {}
            }
        });
    });
}

// ---------- History ----------
fn page_history(ui: &mut egui::Ui, st: &mut SettingsState) {
    h1(ui, tr("nav_history"));
    ui.horizontal(|ui| {
        if ui.button(tr("hist_refresh")).clicked() {
            st.history = history::read_entries(100);
        }
        if ui.button(tr("hist_clear")).clicked() {
            history::clear();
            st.history.clear();
        }
        if st.filetx.is_some() {
            ui.add_enabled(false, egui::Button::new(tr("filetx_running")));
            ui.spinner();
        } else if ui.button(tr("hist_from_file")).clicked() {
            let picked = rfd::FileDialog::new()
                .add_filter(
                    "Audio/Video",
                    &["wav", "mp3", "m4a", "flac", "ogg", "opus", "mp4", "mkv", "mov", "webm", "aac"],
                )
                .pick_file();
            if let Some(path) = picked {
                st.filetx = Some(spawn_filetx(&st.cfg, path, ui.ctx().clone()));
                st.status = tr("filetx_running").into();
            }
        }
        ui.label(RichText::new(tr("hist_hint")).color(DIM).size(12.0));
    });
    // Collects the file transcription result.
    if let Some(ft) = &st.filetx {
        let done = ft.done.lock().unwrap().take();
        if let Some(res) = done {
            st.filetx = None;
            match res {
                Ok(text) => {
                    let _ = paste::set_clipboard(&text);
                    history::add_entry(&text, &st.cfg.active_mode, None, None);
                    st.history = history::read_entries(100);
                    st.status = tr("filetx_done").into();
                }
                Err(e) => {
                    st.status = format!("{} : {}", tr("filetx_error"), e.chars().take(120).collect::<String>());
                }
            }
        }
    }
    ui.add_space(6.0);
    card(ui, None, |ui| {
        if st.history.is_empty() {
            ui.label(RichText::new(tr("hist_none")).color(DIM));
        }
        for e in &st.history {
            let preview: String = e.text.chars().take(80).collect();
            let line = format!("[{}] ({}) {}", e.datetime, e.mode, preview);
            if ui
                .add(egui::Label::new(line).sense(egui::Sense::click()))
                .on_hover_text(tr("hist_copy_hover"))
                .clicked()
            {
                let _ = paste::set_clipboard(&e.text);
                st.status = tr("hist_copied").into();
            }
        }
    });
}

/// Transcribes an audio/video file in the background (loads its own
/// transcriber: the settings window has no access to the app's).
fn spawn_filetx(cfg: &Config, path: std::path::PathBuf, ctx: egui::Context) -> FileTx {
    let done: Arc<Mutex<Option<Result<String, String>>>> = Arc::new(Mutex::new(None));
    let d2 = done.clone();
    let cfg = cfg.clone();
    std::thread::spawn(move || {
        let res = transcribe_file(&cfg, &path);
        *d2.lock().unwrap_or_else(|p| p.into_inner()) = Some(res);
        ctx.request_repaint();
    });
    FileTx { done }
}

// ---------- download ----------
fn spawn_download(dir: &str, model: &str, ctx: egui::Context) -> Download {
    let fname = models::file_name(model);
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{fname}");
    spawn_download_url(dir, &url, &fname, model, ctx)
}

/// Downloads an arbitrary URL (HuggingFace search) on a worker thread.
fn spawn_download_url(dir: &str, url: &str, fname: &str, label: &str, ctx: egui::Context) -> Download {
    let received = Arc::new(AtomicU64::new(0));
    let total = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicBool::new(false));
    let error = Arc::new(Mutex::new(None));
    let (d, u, f) = (dir.to_string(), url.to_string(), fname.to_string());
    let (r2, t2, f2, e2) = (received.clone(), total.clone(), finished.clone(), error.clone());
    std::thread::spawn(move || {
        let r3 = r2.clone();
        let res = models::download_url(&d, &u, &f, |recu, tot| {
            r3.store(recu, Ordering::Relaxed);
            if let Some(t) = tot {
                t2.store(t, Ordering::Relaxed);
            }
            ctx.request_repaint();
        });
        if let Err(e) = res {
            *e2.lock().unwrap() = Some(e);
        }
        f2.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
    Download {
        model: label.to_string(),
        received,
        total,
        finished,
        error,
    }
}

/// Runs a HuggingFace fetch (search or file listing) on a worker thread.
fn spawn_hf(query: models::HfQuery, ctx: egui::Context) -> Arc<Mutex<Option<HfOutcome>>> {
    let slot: Arc<Mutex<Option<HfOutcome>>> = Arc::new(Mutex::new(None));
    let slot2 = slot.clone();
    std::thread::spawn(move || {
        let outcome = match query {
            models::HfQuery::Search(q) => HfOutcome::Repos(models::hf_search(&q)),
            models::HfQuery::Repo(r) => HfOutcome::Files(models::hf_list_files(&r)),
            models::HfQuery::FileUrl(..) => unreachable!("handled by the caller"),
        };
        *slot2.lock().unwrap() = Some(outcome);
        ctx.request_repaint();
    });
    slot
}
