//! Notification icon (`tray-icon` crate) + context menu.
//!
//! Like the global hotkey, the tray relies on the message loop of the thread
//! that creates it (eframe/winit in the app). Its events are drained via
//! [`Tray::poll_actions`] from that same thread.

use tray_icon::menu::{
    CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::i18n::tr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    SetMode(String),
    OpenSettings,
    Quit,
}

pub struct Tray {
    tray: TrayIcon,
    settings_id: MenuId,
    quit_id: MenuId,
    // (mode key, checkable item) in display order; the item id maps back to the
    // key in `poll_actions`, and `set_active_mode` toggles the check marks.
    mode_items: Vec<(String, CheckMenuItem)>,
}

impl Tray {
    /// `modes` = (key, label) pairs in display order; `active` = active mode key.
    pub fn new(modes: &[(String, String)], active: &str) -> Result<Self, String> {
        let (menu, settings_id, quit_id, mode_items) = build_menu(modes, active)?;
        let tray = TrayIconBuilder::new()
            .with_tooltip("Dictata")
            .with_icon(make_icon())
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| format!("tray: {e}"))?;

        Ok(Tray {
            tray,
            settings_id,
            quit_id,
            mode_items,
        })
    }

    /// Rebuilds the whole menu after the mode list changes (e.g. a mode is
    /// renamed/added/removed in settings).
    pub fn refresh(&mut self, modes: &[(String, String)], active: &str) -> Result<(), String> {
        let (menu, settings_id, quit_id, mode_items) = build_menu(modes, active)?;
        self.tray.set_menu(Some(Box::new(menu)));
        self.settings_id = settings_id;
        self.quit_id = quit_id;
        self.mode_items = mode_items;
        Ok(())
    }

    /// Updates the check marks so only `key` is ticked (exclusive selection).
    pub fn set_active_mode(&self, key: &str) {
        for (k, item) in &self.mode_items {
            item.set_checked(k == key);
        }
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
            } else if let Some((k, _)) = self.mode_items.iter().find(|(_, it)| it.id() == &ev.id) {
                out.push(TrayAction::SetMode(k.clone()));
            }
        }
        out
    }
}

/// Builds the context menu (Mode submenu above Settings) and returns it with
/// the ids/items needed to route and update menu events.
fn build_menu(
    modes: &[(String, String)],
    active: &str,
) -> Result<(Menu, MenuId, MenuId, Vec<(String, CheckMenuItem)>), String> {
    let menu = Menu::new();

    let mode_menu = Submenu::new(tr("tray_mode"), true);
    let mut mode_items = Vec::with_capacity(modes.len());
    for (key, label) in modes {
        let item = CheckMenuItem::new(label, true, key == active, None);
        mode_menu.append(&item).map_err(|e| format!("menu: {e}"))?;
        mode_items.push((key.clone(), item));
    }
    menu.append(&mode_menu).map_err(|e| format!("menu: {e}"))?;
    menu.append(&PredefinedMenuItem::separator())
        .map_err(|e| format!("menu: {e}"))?;

    let settings = MenuItem::new(tr("tray_settings"), true, None);
    let quit = MenuItem::new(tr("tray_quit"), true, None);
    menu.append(&settings).map_err(|e| format!("menu: {e}"))?;
    menu.append(&PredefinedMenuItem::separator())
        .map_err(|e| format!("menu: {e}"))?;
    menu.append(&quit).map_err(|e| format!("menu: {e}"))?;

    Ok((
        menu,
        settings.id().clone(),
        quit.id().clone(),
        mode_items,
    ))
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
