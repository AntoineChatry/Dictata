//! Preview of the floating dock: borderless, transparent window, always on
//! top, that never steals focus (WS_EX_NOACTIVATE). Cycles Recording ->
//! Transcribing -> Done with animated waveform. `cargo run --example docktest`

use eframe::egui;
use dictata::dock::{Dock, DockState, HEIGHT, WIDTH};

fn main() -> eframe::Result {
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([WIDTH, HEIGHT])
        .with_decorations(false)
        .with_transparent(true)
        .with_resizable(false)
        .with_always_on_top()
        .with_taskbar(false)
        .with_active(false)
        .with_title("Dictata Dock");
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "Dictata Dock",
        options,
        Box::new(|_cc| Ok(Box::new(DemoApp::new()))),
    )
}

struct DemoApp {
    dock: Dock,
    positioned: bool,
    no_activate_done: bool,
}

impl DemoApp {
    fn new() -> Self {
        let mut dock = Dock::new();
        dock.mode_label = "Email".into();
        dock.state = DockState::Recording;
        DemoApp {
            dock,
            positioned: false,
            no_activate_done: false,
        }
    }
}

impl eframe::App for DemoApp {
    fn clear_color(&self, _v: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let t = ctx.input(|i| i.time);

        // Reposition at bottom-center once the monitor size is known.
        if !self.positioned {
            if let Some(monitor) = ctx.input(|i| i.viewport().monitor_size) {
                let x = (monitor.x - WIDTH) / 2.0;
                let y = monitor.y - HEIGHT - 60.0;
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x, y)));
                self.positioned = true;
            }
        }

        // Apply WS_EX_NOACTIVATE once (no focus steal).
        #[cfg(windows)]
        if !self.no_activate_done {
            apply_no_activate(_frame);
            self.no_activate_done = true;
        }

        // Demo: state cycle over 7 s + waveform/animation.
        let cycle = (t % 7.0) as f32;
        self.dock.state = if cycle < 3.0 {
            DockState::Recording
        } else if cycle < 5.0 {
            DockState::Transcribing
        } else {
            DockState::Done
        };
        self.dock.status_text = match self.dock.state {
            DockState::Recording => "0:03".into(),
            DockState::Transcribing => "\u{2026}".into(),
            DockState::Done => "Colle".into(),
            _ => String::new(),
        };
        if self.dock.state == DockState::Recording {
            let lvl = 0.3 + 0.45 * (t * 9.0).sin().abs() as f32 + 0.2 * (t * 23.0).sin().abs() as f32;
            self.dock.push_level(lvl.min(1.0));
        }

        self.dock.paint(ui.painter(), ctx.content_rect(), t);
        ctx.request_repaint();
    }
}

#[cfg(windows)]
fn apply_no_activate(frame: &eframe::Frame) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetWindowLongPtrW, SetWindowLongPtrW, GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
    };
    if let Ok(h) = frame.window_handle() {
        if let RawWindowHandle::Win32(w) = h.as_raw() {
            let hwnd = w.hwnd.get() as HWND;
            unsafe {
                let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                SetWindowLongPtrW(
                    hwnd,
                    GWL_EXSTYLE,
                    ex | WS_EX_NOACTIVATE as isize | WS_EX_TOOLWINDOW as isize,
                );
            }
        }
    }
}
