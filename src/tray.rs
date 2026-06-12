//! Notification icon (`tray-icon` crate) + context menu.
//!
//! Like the global hotkey, the tray relies on the message loop of the thread
//! that creates it (eframe/winit in the app). Its events are drained via
//! [`Tray::poll_actions`] from that same thread.

use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::i18n::tr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayAction {
    OpenSettings,
    Quit,
}

pub struct Tray {
    _tray: TrayIcon,
    settings_id: MenuId,
    quit_id: MenuId,
}

impl Tray {
    pub fn new() -> Result<Self, String> {
        let menu = Menu::new();
        let settings = MenuItem::new(tr("tray_settings"), true, None);
        let quit = MenuItem::new(tr("tray_quit"), true, None);
        menu.append(&settings).map_err(|e| format!("menu: {e}"))?;
        menu.append(&PredefinedMenuItem::separator())
            .map_err(|e| format!("menu: {e}"))?;
        menu.append(&quit).map_err(|e| format!("menu: {e}"))?;

        let tray = TrayIconBuilder::new()
            .with_tooltip("Dictata")
            .with_icon(make_icon())
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| format!("tray: {e}"))?;

        Ok(Tray {
            _tray: tray,
            settings_id: settings.id().clone(),
            quit_id: quit.id().clone(),
        })
    }

    /// Drain menu events and return the corresponding actions.
    pub fn poll_actions(&self) -> Vec<TrayAction> {
        let rx = MenuEvent::receiver();
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if ev.id == self.settings_id {
                out.push(TrayAction::OpenSettings);
            } else if ev.id == self.quit_id {
                out.push(TrayAction::Quit);
            }
        }
        out
    }
}

/// 32x32 RGBA icon: blue disc on transparent background (placeholder).
fn make_icon() -> Icon {
    let (w, h) = (32u32, 32u32);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let (cx, cy, r) = (15.5f32, 15.5f32, 13.0f32);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            if dx * dx + dy * dy <= r * r {
                let i = ((y * w + x) * 4) as usize;
                rgba[i] = 0x4a;
                rgba[i + 1] = 0x90;
                rgba[i + 2] = 0xe2;
                rgba[i + 3] = 0xff;
            }
        }
    }
    Icon::from_rgba(rgba, w, h).expect("icone rgba")
}
