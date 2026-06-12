//! Notification icon test: build the tray and run a
//! message loop for ~6 s (the icon should appear in the notification area).
//! `cargo run --example traytest`  (Windows)

#[cfg(not(windows))]
fn main() {
    eprintln!("Test tray reserve a Windows pour l'instant.");
}

#[cfg(windows)]
fn main() {
    use dictata::tray::{Tray, TrayAction};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };

    let tray = match Tray::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ECHEC construction tray: {e}");
            std::process::exit(1);
        }
    };
    println!("Tray construit. Icone visible ~6 s (clique le menu pour tester)…");

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(6) {
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        for action in tray.poll_actions() {
            println!("Action menu: {action:?}");
            if action == TrayAction::Quit {
                println!("Quitter demande.");
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    println!("OK tray (construit + boucle de messages sans erreur)");
}
