//! Global keyboard shortcut (`global-hotkey` crate).
//!
//! IMPORTANT: on Windows, `GlobalHotKeyManager::new()` creates a hidden window
//! on the calling thread; `WM_HOTKEY` events are only dispatched if that
//! thread runs a message loop. In the application it is eframe/winit (main
//! thread) that pumps the messages. This manager must therefore be created
//! and [`poll_events`] called from the UI thread.

use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager};

pub use global_hotkey::HotKeyState;

pub struct Hotkeys {
    manager: GlobalHotKeyManager,
    current: Option<HotKey>,
}

impl Hotkeys {
    pub fn new() -> Result<Self, String> {
        let manager =
            GlobalHotKeyManager::new().map_err(|e| format!("init raccourci global: {e}"))?;
        Ok(Hotkeys {
            manager,
            current: None,
        })
    }

    /// (Re)register the shortcut from a string like "ctrl+alt+space".
    /// Returns the identifier to compare with received events.
    pub fn set(&mut self, spec: &str) -> Result<u32, String> {
        let hk: HotKey = spec
            .parse()
            .map_err(|e| format!("raccourci invalide '{spec}': {e:?}"))?;
        if let Some(old) = self.current.take() {
            let _ = self.manager.unregister(old);
        }
        self.manager
            .register(hk)
            .map_err(|e| format!("enregistrement '{spec}': {e}"))?;
        self.current = Some(hk);
        Ok(hk.id())
    }

    /// Identifier of the active shortcut (to filter events).
    pub fn current_id(&self) -> Option<u32> {
        self.current.map(|h| h.id())
    }
}

/// Drain pending hotkey events. Call regularly from
/// the UI thread (the message loop must run there).
pub fn poll_events() -> Vec<(u32, HotKeyState)> {
    let rx = GlobalHotKeyEvent::receiver();
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        out.push((ev.id, ev.state));
    }
    out
}
