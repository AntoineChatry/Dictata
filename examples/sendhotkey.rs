//! Send the ctrl+alt+space combination once (to drive the running app).
//! `cargo run --example sendhotkey`

#[cfg(not(windows))]
fn main() {
    eprintln!("Windows uniquement.");
}

#[cfg(windows)]
fn main() {
    use enigo::{
        Direction::{Click, Press, Release},
        Enigo, Key, Keyboard, Settings,
    };
    let mut e = Enigo::new(&Settings::default()).expect("enigo");
    e.key(Key::Control, Press).unwrap();
    e.key(Key::Alt, Press).unwrap();
    e.key(Key::Space, Click).unwrap();
    e.key(Key::Alt, Release).unwrap();
    e.key(Key::Control, Release).unwrap();
    println!("ctrl+alt+space envoye");
}
