//! Real test of the global hotkey: register a combination, simulate it via
//! enigo from a helper thread, and verify that `WM_HOTKEY` does show up in
//! the receiver while the message loop is being pumped.
//! `cargo run --example hotkeytest`  (Windows)

#[cfg(not(windows))]
fn main() {
    eprintln!("Test raccourci reserve a Windows.");
}

#[cfg(windows)]
fn main() {
    use enigo::{
        Direction::{Click, Press, Release},
        Enigo, Key, Keyboard, Settings,
    };
    use dictata::hotkey::{poll_events, HotKeyState, Hotkeys};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
    };

    const SPEC: &str = "ctrl+shift+alt+f8";

    let mut hk = Hotkeys::new().expect("Hotkeys::new");
    let id = hk.set(SPEC).unwrap_or_else(|e| {
        eprintln!("set: {e}");
        std::process::exit(1);
    });
    println!("Raccourci '{SPEC}' enregistre (id={id}). Simulation dans 1 s…");

    // Helper thread: simulate the combination after 1 s.
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_millis(1000));
        let mut e = Enigo::new(&Settings::default()).expect("enigo");
        e.key(Key::Control, Press).unwrap();
        e.key(Key::Shift, Press).unwrap();
        e.key(Key::Alt, Press).unwrap();
        e.key(Key::F8, Click).unwrap();
        e.key(Key::Alt, Release).unwrap();
        e.key(Key::Shift, Release).unwrap();
        e.key(Key::Control, Release).unwrap();
    });

    // Message loop (~4 s): pump WM_HOTKEY then drain the events.
    let mut got_pressed = false;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(4) {
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        for (eid, state) in poll_events() {
            println!("Evenement: id={eid} state={state:?}");
            if eid == id && state == HotKeyState::Pressed {
                got_pressed = true;
            }
        }
        if got_pressed {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    if got_pressed {
        println!("OK raccourci global (WM_HOTKEY recu pour la combinaison simulee)");
    } else {
        eprintln!("ECHEC: aucun evenement Pressed recu pour id={id}");
        std::process::exit(1);
    }
}
