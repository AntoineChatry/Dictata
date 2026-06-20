//! Pasting the transcribed text into the active application.
//!
//! Strategy (like v1): put the text in the clipboard then
//! simulate Ctrl+V (Cmd+V on macOS). The previous clipboard text content is
//! restored afterwards so as not to overwrite what the user had copied.

use arboard::Clipboard;
use enigo::{
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Settings,
};
use std::time::Duration;

#[cfg(target_os = "macos")]
const MOD_KEY: Key = Key::Meta;
#[cfg(not(target_os = "macos"))]
const MOD_KEY: Key = Key::Control;

/// Copy `text` to the clipboard (without pasting).
pub fn set_clipboard(text: &str) -> Result<(), String> {
    let mut cb = Clipboard::new().map_err(|e| format!("presse-papier: {e}"))?;
    cb.set_text(text.to_owned())
        .map_err(|e| format!("copie: {e}"))
}

/// Simulate pressing Ctrl/Cmd+V in the foreground application.
fn press_paste() -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| format!("enigo: {e}"))?;
    enigo
        .key(MOD_KEY, Press)
        .map_err(|e| format!("press mod: {e}"))?;
    enigo
        .key(Key::Unicode('v'), Click)
        .map_err(|e| format!("touche v: {e}"))?;
    enigo
        .key(MOD_KEY, Release)
        .map_err(|e| format!("release mod: {e}"))?;
    Ok(())
}

/// Copy `text` then paste it into the active app, restoring the previous
/// clipboard text content afterwards (default delay).
pub fn paste_text(text: &str) -> Result<(), String> {
    paste_text_with_delay(text, 0.12)
}

/// Like [`paste_text`], with `restore_delay` (s) before restoring the
/// clipboard (see `paste_restore_delay` in the config).
pub fn paste_text_with_delay(text: &str, restore_delay: f32) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    let previous = Clipboard::new().ok().and_then(|mut c| c.get_text().ok());

    set_clipboard(text)?;
    // Let the clipboard propagate before pasting.
    std::thread::sleep(Duration::from_millis(40));
    press_paste()?;

    // Restore the previous content after the paste has been consumed. Electron
    // apps (e.g. LM Studio) read the clipboard *asynchronously* on Ctrl+V, so a
    // synchronous restore can win the race and make them paste the previous
    // content (Notepad reads synchronously, hence no issue there). Restore
    // off-thread and only if our text is still on the clipboard: a generous
    // delay then neither blocks the caller (next streaming chunk / one-shot)
    // nor clobbers a more recent paste.
    if let Some(prev) = previous {
        let pasted = text.to_owned();
        let delay = restore_delay.clamp(0.0, 5.0);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f32(delay));
            if let Ok(mut c) = Clipboard::new() {
                if c.get_text().ok().as_deref() == Some(pasted.as_str()) {
                    let _ = c.set_text(prev);
                }
            }
        });
    }
    Ok(())
}

/// Type `text` character by character (without the clipboard).
pub fn type_text(text: &str) -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| format!("enigo: {e}"))?;
    enigo.text(text).map_err(|e| format!("frappe: {e}"))
}
