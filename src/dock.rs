//! SuperWhisper-style floating dock, rendered with egui.
//!
//! Small translucent rounded pill: status dot on the left, animated central
//! area (waveform while recording, dots while transcribing, a line when idle),
//! mode label + status on the right. Reuses the colors and proportions of the
//! Python v1 (`freewhisper/dock.py`).

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use std::collections::VecDeque;

pub const WIDTH: f32 = 300.0;
pub const HEIGHT: f32 = 64.0;
const RADIUS: f32 = 18.0;
const BARS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DockState {
    Idle,
    Recording,
    Transcribing,
    Done,
    Error,
}

fn accent(state: DockState) -> Color32 {
    match state {
        DockState::Recording => Color32::from_rgb(94, 170, 255),
        DockState::Transcribing => Color32::from_rgb(190, 150, 255),
        DockState::Done => Color32::from_rgb(90, 210, 140),
        DockState::Error => Color32::from_rgb(240, 110, 110),
        DockState::Idle => Color32::from_rgb(150, 150, 160),
    }
}

fn with_alpha(c: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (alpha.clamp(0.0, 1.0) * 255.0) as u8)
}

pub struct Dock {
    pub state: DockState,
    pub mode_label: String,
    pub status_text: String,
    /// Size factor (0.7 - 1.6), see `dock_scale` in the config.
    pub scale: f32,
    /// Opacity of the pill background (0.4 - 1.0), see `dock_opacity`.
    pub opacity: f32,
    levels: VecDeque<f32>,
}

impl Default for Dock {
    fn default() -> Self {
        Self::new()
    }
}

impl Dock {
    pub fn new() -> Self {
        Dock {
            state: DockState::Idle,
            mode_label: "Brut".to_string(),
            status_text: String::new(),
            scale: 1.0,
            opacity: 0.93,
            levels: VecDeque::from(vec![0.0; BARS]),
        }
    }

    /// Adds an RMS level (0..1) to the waveform (sliding window of 32).
    pub fn push_level(&mut self, level: f32) {
        if self.levels.len() >= BARS {
            self.levels.pop_front();
        }
        self.levels.push_back(level.clamp(0.0, 1.0));
    }

    pub fn reset_levels(&mut self) {
        self.levels = VecDeque::from(vec![0.0; BARS]);
    }

    /// Draws the dock into `rect`. `time` (seconds) animates pulses/dots.
    pub fn paint(&self, painter: &egui::Painter, rect: Rect, time: f64) {
        let phase = (time * 5.5) as f32;
        let acc = accent(self.state);
        // Uniform scale: all drawing follows the actual window size.
        let s = (rect.height() / HEIGHT).max(0.1);
        let bg_a = (self.opacity.clamp(0.4, 1.0) * 255.0) as u8;

        // Dark pill background + subtle border.
        painter.rect_filled(rect, RADIUS * s, Color32::from_rgba_unmultiplied(26, 26, 30, bg_a));
        painter.rect_stroke(
            rect,
            RADIUS * s,
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 26)),
            StrokeKind::Inside,
        );

        // Status dot (left), with a pulsing halo while recording.
        let cx = rect.left() + 22.0 * s;
        let cy = rect.center().y;
        let dot_r = 5.0 * s;
        if self.state == DockState::Recording {
            let pulse = 0.6 + 0.4 * phase.sin().abs();
            painter.circle_filled(Pos2::new(cx, cy), dot_r * 2.4, with_alpha(acc, 0.25 * pulse));
        }
        painter.circle_filled(Pos2::new(cx, cy), dot_r, acc);

        // Central area.
        let area = Rect::from_min_max(
            Pos2::new(rect.left() + 40.0 * s, rect.top() + 8.0 * s),
            Pos2::new(rect.right() - 70.0 * s, rect.bottom() - 8.0 * s),
        );
        match self.state {
            DockState::Recording => self.paint_waveform(painter, area, acc),
            DockState::Transcribing => self.paint_thinking(painter, area, acc, phase, s),
            _ => self.paint_static(painter, area, acc),
        }

        // Right-hand text: mode (bold-ish) + status.
        let tx = rect.right() - 8.0 * s;
        painter.text(
            Pos2::new(tx, rect.top() + 20.0 * s),
            Align2::RIGHT_CENTER,
            &self.mode_label,
            FontId::proportional(13.0 * s),
            Color32::from_rgba_unmultiplied(235, 235, 240, 230),
        );
        if !self.status_text.is_empty() {
            painter.text(
                Pos2::new(tx, rect.top() + 40.0 * s),
                Align2::RIGHT_CENTER,
                &self.status_text,
                FontId::proportional(11.0 * s),
                Color32::from_rgba_unmultiplied(170, 170, 180, 220),
            );
        }
    }

    fn paint_waveform(&self, painter: &egui::Painter, area: Rect, acc: Color32) {
        let n = self.levels.len().max(1);
        let slot = area.width() / n as f32;
        let bar_w = (slot * 0.55).max(2.0);
        let mid = area.center().y;
        let max_h = area.height();
        for (i, &lvl) in self.levels.iter().enumerate() {
            let h = (lvl * max_h).max(2.0);
            let x = area.left() + i as f32 * slot + (slot - bar_w) / 2.0;
            let color = with_alpha(acc, 0.45 + 0.55 * (lvl * 1.4).min(1.0));
            let r = Rect::from_min_size(Pos2::new(x, mid - h / 2.0), Vec2::new(bar_w, h));
            painter.rect_filled(r, bar_w / 2.0, color);
        }
    }

    fn paint_thinking(&self, painter: &egui::Painter, area: Rect, acc: Color32, phase: f32, scale: f32) {
        let cy = area.center().y;
        let spacing = 16.0 * scale;
        let start_x = area.center().x - spacing;
        for i in 0..3 {
            let ph = phase - i as f32 * 0.6;
            let s = 0.5 + 0.5 * ph.sin();
            let r = (3.0 + 3.0 * s) * scale;
            painter.circle_filled(Pos2::new(start_x + i as f32 * spacing, cy), r, with_alpha(acc, 0.4 + 0.6 * s));
        }
    }

    fn paint_static(&self, painter: &egui::Painter, area: Rect, acc: Color32) {
        let cy = area.center().y;
        let r = Rect::from_min_size(Pos2::new(area.left(), cy - 2.0), Vec2::new(area.width(), 4.0));
        painter.rect_filled(r, 2.0, with_alpha(acc, 0.8));
    }
}
