//! Loopback capture test (system audio): record ~2 s and print
//! stats. Play a sound during the test to verify the capture.
//! `cargo run --example loopbacktest [system|mix]`

use dictata::audio::Recorder;
use std::time::Duration;

fn main() {
    let source = std::env::args().nth(1).unwrap_or_else(|| "system".into());
    let mut rec = Recorder::new(None, source.clone());
    match rec.start() {
        Ok(()) => println!("Capture '{source}' 2 s\u{2026} (joue un son !)"),
        Err(e) => {
            eprintln!("Echec demarrage capture {source} : {e}");
            std::process::exit(1);
        }
    }
    std::thread::sleep(Duration::from_secs(2));
    let samples = rec.stop();
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    println!("Echantillons 16k : {}", samples.len());
    println!("Duree (s)        : {:.2}", samples.len() as f32 / 16000.0);
    println!("RMS global       : {rms:.5}");
}
