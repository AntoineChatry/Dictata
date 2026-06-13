//! Deterministic transcription test on the jfk.wav sample.
//! `cargo run --example transcribe`
//!
//! Expected (tiny model): the famous JFK excerpt
//! "And so my fellow Americans, ask not what your country can do for you..."

use dictata::transcriber::Transcriber;
use std::path::Path;
use std::time::Instant;

/// Minimal WAV reader: 16-bit mono PCM. Returns (f32 samples, rate).
fn read_wav_pcm16(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("lecture wav: {e}"))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("entete RIFF/WAVE invalide".into());
    }
    let mut sample_rate = 0u32;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            sample_rate = u32::from_le_bytes([bytes[body + 4], bytes[body + 5], bytes[body + 6], bytes[body + 7]]);
        } else if id == b"data" {
            let end = (body + size).min(bytes.len());
            data = Some(&bytes[body..end]);
        }
        pos = body + size + (size & 1); // chunks aligned on 2 bytes
    }
    let data = data.ok_or("chunk data introuvable")?;
    let samples: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    Ok((samples, sample_rate))
}

fn main() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let model = base.join("ggml-tiny.bin");
    let wav = base.join("jfk.wav");

    let (audio, rate) = read_wav_pcm16(&wav).unwrap_or_else(|e| {
        eprintln!("WAV: {e}");
        std::process::exit(1);
    });
    println!("WAV: {} echantillons @ {} Hz ({:.2} s)", audio.len(), rate, audio.len() as f32 / rate as f32);
    if rate != 16000 {
        eprintln!("ATTENTION: rate != 16000, whisper attend du 16 kHz");
    }

    let t0 = Instant::now();
    let gpu = std::env::var("DICTATA_GPU").map(|v| v == "1").unwrap_or(false);
    println!("GPU: {gpu}");
    let mut tr = Transcriber::load(&model, gpu).unwrap_or_else(|e| {
        eprintln!("Chargement modele: {e}");
        std::process::exit(1);
    });
    println!("Modele charge en {:.2} s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let text = tr.transcribe(&audio, Some("en"), false, None, 5, None).unwrap_or_else(|e| {
        eprintln!("Transcription: {e}");
        std::process::exit(1);
    });
    println!("Transcrit en {:.2} s", t1.elapsed().as_secs_f32());
    println!("---");
    println!("{text}");
    println!("---");

    let low = text.to_lowercase();
    if low.contains("fellow americans") && low.contains("country") {
        println!("OK transcription (mots cles JFK presents)");
    } else {
        eprintln!("ECHEC: texte attendu non trouve");
        std::process::exit(2);
    }
}
