//! Small platform-specific adjustments.

/// Apply `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW` to the eframe `frame` window
/// so it never steals focus (essential for the dock: otherwise the
/// paste Ctrl+V would land in the dock). No-op outside Windows.
#[cfg(windows)]
pub fn make_no_activate(frame: &eframe::Frame) {
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

#[cfg(not(windows))]
pub fn make_no_activate(_frame: &eframe::Frame) {}

/// Executable name (lowercased leaf, e.g. "firefox.exe") and window title
/// (lowercased) of the foreground window, used to auto-select a dictation
/// mode. The title matters because a browser is a single executable for
/// every site: only the title tells a webmail from a code review.
/// Captured at record start, while the target app still holds focus (the
/// dock is a `WS_EX_NOACTIVATE` window and never steals it).
/// Returns `None` on failure or on platforms where it is not implemented
/// (non-Windows: the caller keeps the manually selected mode).
#[cfg(windows)]
pub fn foreground_app() -> Option<(String, String)> {
    use windows_sys::Win32::Foundation::{CloseHandle, MAX_PATH};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId,
    };
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return None;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        if pid == 0 {
            return None;
        }
        // Title first: it is the cheap call and the one that discriminates
        // between two tabs of the same browser. An empty title is not fatal.
        let mut tbuf = [0u16; 512];
        let tlen = GetWindowTextW(hwnd, tbuf.as_mut_ptr(), tbuf.len() as i32);
        let title = if tlen > 0 {
            String::from_utf16_lossy(&tbuf[..tlen as usize]).to_lowercase()
        } else {
            String::new()
        };
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut buf = [0u16; MAX_PATH as usize];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(handle, 0, buf.as_mut_ptr(), &mut len);
        CloseHandle(handle);
        if ok == 0 || len == 0 {
            return None;
        }
        let full = String::from_utf16_lossy(&buf[..len as usize]);
        let leaf = full.rsplit(['\\', '/']).next().unwrap_or(&full);
        Some((leaf.to_lowercase(), title))
    }
}

/// Non-Windows: not implemented (Wayland forbids querying another app's
/// focus; X11 support would need an extra dependency). The caller keeps
/// the manually selected mode.
#[cfg(not(windows))]
pub fn foreground_app() -> Option<(String, String)> {
    None
}

/// True while the Escape key is physically held down, whatever window has
/// focus. Polled per frame during a take to cancel it: the dock is a
/// `WS_EX_NOACTIVATE` window, so it never receives key events itself and
/// egui's own input cannot see Escape. Reading the key state (rather than
/// registering a hotkey) keeps Escape working normally in the focused
/// application instead of swallowing it.
#[cfg(windows)]
pub fn escape_down() -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_ESCAPE};
    // The high bit means "currently down"; the low bit ("pressed since the
    // last call") is shared process-wide and unreliable, so it is ignored.
    unsafe { (GetAsyncKeyState(VK_ESCAPE as i32) as u16 & 0x8000) != 0 }
}

#[cfg(not(windows))]
pub fn escape_down() -> bool {
    false
}

/// How long Escape must be held to abort a take.
///
/// [`escape_down`] reads the key globally, so a plain tap would also cancel the
/// dictation when it was meant for the focused application — closing an
/// autocomplete popup, dismissing a dialog. Requiring a deliberate hold keeps
/// Escape usable in the target app while still offering a way out.
pub const CANCEL_HOLD: std::time::Duration = std::time::Duration::from_millis(400);

/// Turns the raw global Escape state into a deliberate "abort this take"
/// gesture. Pure state machine: [`poll`](Self::poll) takes the key state and
/// the current instant, so it is testable without a keyboard.
pub struct CancelGesture {
    /// False while a key held from before the take started is still down: that
    /// press was not meant for us, so it must be released first.
    armed: bool,
    held_since: Option<std::time::Instant>,
}

impl CancelGesture {
    /// `escape_already_down` is the key state at the moment the take starts.
    pub fn new(escape_already_down: bool) -> Self {
        CancelGesture {
            armed: !escape_already_down,
            held_since: None,
        }
    }

    /// Returns true exactly once, when the hold has lasted [`CANCEL_HOLD`].
    pub fn poll(&mut self, down: bool, now: std::time::Instant) -> bool {
        if !down {
            self.armed = true;
            self.held_since = None;
            return false;
        }
        if !self.armed {
            return false;
        }
        let since = *self.held_since.get_or_insert(now);
        if now.duration_since(since) >= CANCEL_HOLD {
            // Disarm so a key still held after the abort cannot fire again.
            self.armed = false;
            self.held_since = None;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod gesture_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_short_tap_does_not_cancel() {
        // The common case: Escape pressed for the focused application.
        let mut g = CancelGesture::new(false);
        let t0 = Instant::now();
        assert!(!g.poll(true, t0));
        assert!(!g.poll(true, t0 + Duration::from_millis(200)));
        assert!(!g.poll(false, t0 + Duration::from_millis(250)));
        // And the aborted attempt leaves no residue.
        assert!(!g.poll(true, t0 + Duration::from_millis(300)));
    }

    #[test]
    fn a_long_hold_cancels_once() {
        let mut g = CancelGesture::new(false);
        let t0 = Instant::now();
        assert!(!g.poll(true, t0));
        assert!(!g.poll(true, t0 + CANCEL_HOLD - Duration::from_millis(1)));
        assert!(g.poll(true, t0 + CANCEL_HOLD));
        // Still held: must not fire a second time.
        assert!(!g.poll(true, t0 + CANCEL_HOLD * 3));
    }

    #[test]
    fn a_key_held_from_before_the_take_is_ignored_until_released() {
        // Escape already down when recording starts: that press was not aimed
        // at the take and must not abort it.
        let mut g = CancelGesture::new(true);
        let t0 = Instant::now();
        assert!(!g.poll(true, t0 + CANCEL_HOLD * 2));
        // Released, then pressed again: now it counts.
        assert!(!g.poll(false, t0 + CANCEL_HOLD * 2));
        assert!(!g.poll(true, t0 + CANCEL_HOLD * 3));
        assert!(g.poll(true, t0 + CANCEL_HOLD * 4));
    }

    #[test]
    fn releasing_restarts_the_hold() {
        let mut g = CancelGesture::new(false);
        let t0 = Instant::now();
        assert!(!g.poll(true, t0));
        assert!(!g.poll(false, t0 + Duration::from_millis(300)));
        // The earlier 300 ms must not count towards the new press.
        assert!(!g.poll(true, t0 + Duration::from_millis(310)));
        assert!(!g.poll(true, t0 + Duration::from_millis(600)));
        assert!(g.poll(true, t0 + Duration::from_millis(710)));
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn foreground_app_ffi_is_sound() {
        // The foreground window is non-deterministic in a test run, so we
        // don't assert a specific app; this exercises the OpenProcess /
        // QueryFullProcessImageNameW FFI path without crashing, and checks
        // that any returned name is a lowercased, non-empty leaf.
        if let Some((exe, title)) = foreground_app() {
            assert!(!exe.is_empty());
            assert!(!exe.contains('\\') && !exe.contains('/'));
            assert_eq!(exe, exe.to_lowercase());
            assert_eq!(title, title.to_lowercase());
        }
    }
}
