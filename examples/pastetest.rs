//! Real paste test: open Notepad, force it to the foreground, paste a probe
//! text via `paste_text`, then read back Notepad's content (Ctrl+A / Ctrl+C)
//! to verify it is indeed there.
//! `cargo run --example pastetest`  (Windows; closes Notepad without saving)

#[cfg(not(windows))]
fn main() {
    eprintln!("Test Notepad reserve a Windows.");
}

#[cfg(windows)]
fn main() {
    use arboard::Clipboard;
    use enigo::{
        Direction::{Click, Press, Release},
        Enigo, Key, Keyboard, Settings,
    };
    use dictata::paste;
    use std::process::Command;
    use std::thread::sleep;
    use std::time::Duration;

    const SENTINEL: &str = "SENTINEL_BEFORE_PASTE";
    const PROBE: &str = "DICTATA_TEST_4242";

    // 1. Clipboard round-trip (arboard).
    paste::set_clipboard(SENTINEL).expect("set_clipboard");
    let back = Clipboard::new().unwrap().get_text().unwrap();
    assert_eq!(back, SENTINEL, "round-trip presse-papier casse");
    println!("Presse-papier round-trip : OK");

    // 2. Open Notepad and wait for its creation.
    println!("Ouverture de Notepad…");
    Command::new("notepad.exe").spawn().expect("lancement notepad");
    sleep(Duration::from_millis(1500));

    // 3. Force Notepad to the foreground (bypasses the Windows focus lock).
    if !focus_notepad() {
        eprintln!("ECHEC: fenetre Notepad introuvable");
        std::process::exit(3);
    }
    sleep(Duration::from_millis(400));

    // 4. Paste the probe text via the production function.
    paste::paste_text(PROBE).expect("paste_text");
    sleep(Duration::from_millis(400));

    // 5. Select all + copy to read back what Notepad contains.
    {
        let mut enigo = Enigo::new(&Settings::default()).expect("enigo");
        enigo.key(Key::Control, Press).unwrap();
        enigo.key(Key::Unicode('a'), Click).unwrap();
        enigo.key(Key::Unicode('c'), Click).unwrap();
        enigo.key(Key::Control, Release).unwrap();
    }
    sleep(Duration::from_millis(300));
    let in_notepad = Clipboard::new().unwrap().get_text().unwrap_or_default();

    // 6. Close Notepad without saving.
    let _ = Command::new("taskkill")
        .args(["/F", "/IM", "notepad.exe"])
        .output();

    println!("Contenu lu dans Notepad : {in_notepad:?}");
    if in_notepad.trim() == PROBE {
        println!("OK collage reel (Ctrl+V a insere le texte dans l'app active)");
    } else if in_notepad == SENTINEL {
        eprintln!("ECHEC: Notepad vide -> le collage n'a pas atteint l'app");
        std::process::exit(1);
    } else {
        eprintln!("ECHEC: contenu inattendu (autre fenetre au premier plan ?)");
        std::process::exit(2);
    }
}

/// Reliably bring the classic Notepad window to the foreground.
#[cfg(windows)]
fn focus_notepad() -> bool {
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::System::Threading::AttachThreadInput;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, FindWindowW, GetForegroundWindow, GetWindowThreadProcessId,
        SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    unsafe {
        let class: Vec<u16> = "Notepad".encode_utf16().chain(std::iter::once(0)).collect();
        let hwnd: HWND = FindWindowW(class.as_ptr(), std::ptr::null());
        if hwnd.is_null() {
            return false;
        }
        let fg = GetForegroundWindow();
        let fg_tid = GetWindowThreadProcessId(fg, std::ptr::null_mut());
        let np_tid = GetWindowThreadProcessId(hwnd, std::ptr::null_mut());
        AttachThreadInput(fg_tid, np_tid, 1);
        ShowWindow(hwnd, SW_RESTORE);
        BringWindowToTop(hwnd);
        SetForegroundWindow(hwnd);
        AttachThreadInput(fg_tid, np_tid, 0);
        true
    }
}
