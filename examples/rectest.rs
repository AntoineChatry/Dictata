//! Microphone capture test: record ~1.5 s and print stats.
//! `cargo run --example rectest`

use dictata::audio::Recorder;
use std::time::Duration;

fn main() {
    println!("Peripheriques d'entree :");
    for d in dictata::audio::list_input_devices() {
        println!("  - {d}");
    }

    let mut rec = Recorder::new(None, "mic".into());
    match rec.start() {
        Ok(()) => println!("Enregistrement 1,5 s… (parle !)"),
        Err(e) => {
            eprintln!("Echec demarrage micro : {e}");
            std::process::exit(1);
        }
    }

    let mut peak = 0.0f32;
    for _ in 0..15 {
        std::thread::sleep(Duration::from_millis(100));
        let l = rec.level();
        if l > peak {
            peak = l;
        }
    }

    let samples = rec.stop();
    let dur = samples.len() as f32 / 16000.0;
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    println!("Echantillons 16k : {}", samples.len());
    println!("Duree (s)        : {dur:.2}");
    println!("Niveau crete     : {peak:.3}");
    println!("RMS global       : {rms:.4}");
    if samples.len() > 8000 {
        println!("OK capture");
    } else {
        println!("ATTENTION: tres peu d'echantillons");
    }
}
