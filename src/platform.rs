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
