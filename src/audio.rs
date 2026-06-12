//! Audio capture (cpal) -> mono 16 kHz f32 for whisper.cpp.
//!
//! Three sources (see `audio_source` in the config):
//! - "mic"    : microphone (default);
//! - "system" : system audio via WASAPI loopback (cpal enables loopback
//!   automatically when an input stream is opened on an output device);
//! - "mix"    : mic + system audio, mixed by addition after resampling
//!   (meeting mode).
//!
//! Each stream captures at the device's native format (often 48 kHz,
//! multiple channels), folds to mono in the callback, then is resampled
//! to 16 kHz when drained/stopped. The RMS level is exposed continuously
//! to animate the dock waveform.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

pub const TARGET_RATE: u32 = 16000;

/// Lists the names of the available input devices.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut out = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(desc) = d.description() {
                out.push(desc.name().to_string());
            }
        }
    }
    out
}

/// Decodes any audio/video file to mono f32 16 kHz via ffmpeg
/// (port of the Python `audio.load_audio_file`).
pub fn load_audio_file(path: &str) -> Result<Vec<f32>, String> {
    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args([
        "-nostdin", "-threads", "0", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", "16000",
        "-loglevel", "error", "-",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd.output().map_err(|e| format!("ffmpeg introuvable: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg a echoue: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn pick_device(name: &Option<String>) -> Option<cpal::Device> {
    let host = cpal::default_host();
    if let Some(want) = name {
        if let Ok(devices) = host.input_devices() {
            for d in devices {
                if d.description()
                    .map(|desc| desc.name() == want.as_str())
                    .unwrap_or(false)
                {
                    return Some(d);
                }
            }
        }
    }
    host.default_input_device()
}

/// Shared buffer of one capture (one per stream).
struct Shared {
    buffer: Mutex<Vec<f32>>, // mono, at native rate
    level: AtomicU32,        // f32 stored as raw bits (to_bits/from_bits), 0.0..1.0
    rate: AtomicU32,         // native stream rate
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Shared {
            buffer: Mutex::new(Vec::new()),
            level: AtomicU32::new(0),
            rate: AtomicU32::new(TARGET_RATE),
        })
    }

    /// Empties the buffer and returns it resampled to 16 kHz.
    fn drain_16k(&self) -> Vec<f32> {
        let native = std::mem::take(&mut *self.buffer.lock().unwrap());
        resample_to_16k(&native, self.rate.load(Ordering::Relaxed))
    }
}

pub struct Recorder {
    pub device_name: Option<String>,
    pub source: String, // "mic" | "system" | "mix"
    streams: Vec<cpal::Stream>,
    caps: Vec<Arc<Shared>>,
}

impl Recorder {
    pub fn new(device_name: Option<String>, source: String) -> Self {
        Recorder {
            device_name,
            source,
            streams: Vec::new(),
            caps: Vec::new(),
        }
    }

    /// Current RMS level (0.0..1.0) for the waveform (max across captures).
    pub fn level(&self) -> f32 {
        self.caps
            .iter()
            .map(|c| f32::from_bits(c.level.load(Ordering::Relaxed)))
            .fold(0.0, f32::max)
    }

    pub fn is_recording(&self) -> bool {
        !self.streams.is_empty()
    }

    /// Starts the capture. Returns a readable error if the source is unavailable.
    pub fn start(&mut self) -> Result<(), String> {
        if !self.streams.is_empty() {
            return Ok(());
        }
        let host = cpal::default_host();
        let want_mic = self.source != "system";
        let want_system = self.source == "system" || self.source == "mix";

        if want_mic {
            let device = pick_device(&self.device_name).ok_or("aucun micro disponible")?;
            let cfg = device
                .default_input_config()
                .map_err(|e| format!("config micro: {e}"))?;
            let shared = Shared::new();
            let stream = open_capture(&device, &cfg, shared.clone())?;
            self.streams.push(stream);
            self.caps.push(shared);
        }
        if want_system {
            // WASAPI loopback: input stream opened on the default output device.
            let device = host
                .default_output_device()
                .ok_or("aucune sortie audio disponible (loopback)")?;
            let cfg = device
                .default_output_config()
                .map_err(|e| format!("config loopback: {e}"))?;
            let shared = Shared::new();
            let stream = open_capture(&device, &cfg, shared.clone())?;
            self.streams.push(stream);
            self.caps.push(shared);
        }
        Ok(())
    }

    /// Stops the capture and returns mono 16 kHz f32 audio (sources mixed).
    pub fn stop(&mut self) -> Vec<f32> {
        self.streams.clear(); // drop -> stops the streams
        let parts: Vec<Vec<f32>> = self.caps.iter().map(|c| c.drain_16k()).collect();
        for c in &self.caps {
            c.level.store(0, Ordering::Relaxed);
        }
        mix(parts)
    }

    /// Drain handle shareable across threads (the `Recorder` itself is not
    /// `Send` because of the cpal streams). Create it after `start()`.
    pub fn drain_handle(&self) -> DrainHandle {
        DrainHandle {
            caps: self.caps.clone(),
        }
    }

    /// Duration (s) of the audio captured so far (first capture).
    pub fn duration(&self) -> f32 {
        self.caps
            .first()
            .map(|c| {
                let n = c.buffer.lock().unwrap().len();
                let rate = c.rate.load(Ordering::Relaxed);
                if rate == 0 { 0.0 } else { n as f32 / rate as f32 }
            })
            .unwrap_or(0.0)
    }
}

/// Opens a capture stream on `device` with format `cfg`, feeding `shared`.
fn open_capture(
    device: &cpal::Device,
    cfg: &cpal::SupportedStreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, String> {
    shared.rate.store(cfg.sample_rate(), Ordering::Relaxed);
    let channels = cfg.channels() as usize;
    let sample_format = cfg.sample_format();
    let stream_cfg: cpal::StreamConfig = cfg.clone().into();
    let err_fn = |e| eprintln!("erreur flux audio: {e}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            stream_cfg,
            move |data: &[f32], _| feed(&shared, data, channels),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            stream_cfg,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                feed(&shared, &f, channels)
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            stream_cfg,
            move |data: &[u16], _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0 - 1.0).collect();
                feed(&shared, &f, channels)
            },
            err_fn,
            None,
        ),
        other => return Err(format!("format audio non gere: {other:?}")),
    }
    .map_err(|e| format!("ouverture flux: {e}"))?;

    stream.play().map_err(|e| format!("demarrage flux: {e}"))?;
    Ok(stream)
}

/// Mixes several 16 kHz tracks by addition (clamped to [-1, 1]).
fn mix(mut parts: Vec<Vec<f32>>) -> Vec<f32> {
    match parts.len() {
        0 => Vec::new(),
        1 => parts.pop().unwrap(),
        _ => {
            let len = parts.iter().map(Vec::len).max().unwrap_or(0);
            let mut out = vec![0.0f32; len];
            for p in &parts {
                for (o, s) in out.iter_mut().zip(p.iter()) {
                    *o += s;
                }
            }
            for o in &mut out {
                *o = o.clamp(-1.0, 1.0);
            }
            out
        }
    }
}

/// Access to the capture buffers from another thread (streaming mode):
/// `drain()` empties the audio accumulated so far without stopping capture.
pub struct DrainHandle {
    caps: Vec<Arc<Shared>>,
}

impl DrainHandle {
    /// Empties the accumulated buffers and returns them mixed as mono 16 kHz f32.
    pub fn drain(&self) -> Vec<f32> {
        mix(self.caps.iter().map(|c| c.drain_16k()).collect())
    }
}

/// Folds to mono, accumulates, and updates the RMS level.
fn feed(shared: &Arc<Shared>, data: &[f32], channels: usize) {
    if channels == 0 {
        return;
    }
    let mut sum_sq = 0.0f32;
    let mut mono = Vec::with_capacity(data.len() / channels);
    for frame in data.chunks(channels) {
        let m: f32 = frame.iter().copied().sum::<f32>() / channels as f32;
        sum_sq += m * m;
        mono.push(m);
    }
    if !mono.is_empty() {
        let rms = (sum_sq / mono.len() as f32).sqrt();
        let level = (rms * 8.0).clamp(0.0, 1.0);
        shared.level.store(level.to_bits(), Ordering::Relaxed);
        shared.buffer.lock().unwrap().extend_from_slice(&mono);
    }
}

/// Linear resampling to 16 kHz (sufficient for speech/whisper).
pub fn resample_to_16k(input: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == TARGET_RATE || input.is_empty() {
        return input.to_vec();
    }
    let ratio = TARGET_RATE as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate_is_identity() {
        let v = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_to_16k(&v, TARGET_RATE), v);
    }

    #[test]
    fn resample_48k_to_16k_thirds_length() {
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = resample_to_16k(&input, 48000);
        // 48k -> 16k: ~1/3 of the samples
        assert!((out.len() as i32 - 1600).abs() <= 1, "len={}", out.len());
    }

    #[test]
    fn mix_pads_and_clamps() {
        let out = mix(vec![vec![0.8, 0.8], vec![0.8]]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-6); // 1.6 clamped to 1.0
        assert!((out[1] - 0.8).abs() < 1e-6);
    }
}
