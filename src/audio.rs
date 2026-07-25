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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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

/// Amplifies quiet recordings so soft or low-volume speech reaches a usable
/// level before transcription. Peak-normalizes toward `TARGET` with a capped
/// gain; leaves normal-volume buffers untouched, and near-silent buffers
/// alone so the noise floor is not amplified into hallucinations.
pub fn boost_quiet(samples: &mut [f32]) {
    const TARGET: f32 = 0.5;
    const MAX_GAIN: f32 = 8.0;
    const NOISE_FLOOR: f32 = 0.005;
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak < NOISE_FLOOR || peak >= TARGET {
        return;
    }
    let gain = (TARGET / peak).min(MAX_GAIN);
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
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
    /// Set by cpal's error callback: the stream is dead (device unplugged,
    /// permission revoked). Without it the capture goes silent while the UI
    /// keeps counting, and the user dictates into a stream nobody reads.
    dead: AtomicBool,
    /// Resampling phase, carried across drains (see [`Resampler`]).
    resampler: Mutex<Resampler>,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Shared {
            buffer: Mutex::new(Vec::new()),
            level: AtomicU32::new(0),
            rate: AtomicU32::new(TARGET_RATE),
            dead: AtomicBool::new(false),
            resampler: Mutex::new(Resampler::new()),
        })
    }

    /// Locks the sample buffer, recovering a poisoned mutex instead of panicking.
    ///
    /// One of the holders is `feed`, which runs in cpal's real-time callback: a
    /// panic there would unwind across the WASAPI FFI boundary. A `Vec` of
    /// samples cannot be left in an inconsistent state by a panicking holder,
    /// so ignoring the poison flag is sound, and it keeps every lock site on
    /// this buffer uniformly panic-free.
    fn lock_buffer(&self) -> std::sync::MutexGuard<'_, Vec<f32>> {
        self.buffer.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Empties the buffer and returns it resampled to 16 kHz.
    ///
    /// Streaming calls this every 200 ms, so the resampler state has to persist
    /// between calls: restarting at phase zero on each slice drops the
    /// fractional remainder and puts a discontinuity at every slice boundary.
    fn drain_16k(&self) -> Vec<f32> {
        let native = {
            let mut guard = self.lock_buffer();
            // Copy out, then `clear()` — deliberately not `mem::take`, which
            // swaps in a capacity-0 `Vec` and so throws away the preallocation
            // made in `open_capture`. Streaming drains every 200 ms, so the
            // real-time callback would then be growing the buffer from zero
            // again after the very first drain. The copy costs one memcpy of
            // the drained slice, on this thread rather than in the callback.
            let native = guard.clone();
            guard.clear();
            native
        };
        let rate = self.rate.load(Ordering::Relaxed);
        let mut r = self.resampler.lock().unwrap_or_else(|p| p.into_inner());
        r.process(&native, rate)
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

    /// True once any capture stream has reported a fatal error (device
    /// unplugged, permission revoked). Polled by the UI during a take: cpal
    /// reports these on its own thread, and a release build has no console, so
    /// this is the only way the failure reaches the user.
    pub fn stream_failed(&self) -> bool {
        self.caps.iter().any(|c| c.dead.load(Ordering::Relaxed))
    }

    /// Starts the capture. Returns a readable error if the source is unavailable.
    ///
    /// All or nothing: in `mix` mode the microphone opens before the loopback,
    /// and a failure there would otherwise leave the mic capturing silently
    /// until the next take — recording light on, device held — after the UI has
    /// already reported the take as failed.
    pub fn start(&mut self) -> Result<(), String> {
        if !self.streams.is_empty() {
            return Ok(());
        }
        if let Err(e) = self.start_streams() {
            self.streams.clear();
            self.caps.clear();
            return Err(e);
        }
        Ok(())
    }

    fn start_streams(&mut self) -> Result<(), String> {
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
                let n = c.lock_buffer().len();
                let rate = c.rate.load(Ordering::Relaxed);
                if rate == 0 { 0.0 } else { n as f32 / rate as f32 }
            })
            .unwrap_or(0.0)
    }
}

/// Audio reserved in the capture buffer before the stream starts.
///
/// `feed` appends to that buffer from cpal's real-time callback, so a
/// reallocation there means copying the whole buffer (tens of MB on a long
/// take) inside a few-milliseconds budget — an underrun, i.e. a hole in the
/// recording. Reserving up front moves that cost out of the callback.
///
/// Only the first minute is covered: `max_record_seconds` defaults to 600, and
/// reserving for the worst case would pin ~115 MB per stream at 48 kHz. Takes
/// longer than this reallocate as before.
///
/// The reservation only survives because [`Shared::drain_16k`] empties the
/// buffer with `clear()`; anything that replaces the `Vec` silently undoes it.
const PREALLOC_SECONDS: usize = 60;

/// Opens a capture stream on `device` with format `cfg`, feeding `shared`.
fn open_capture(
    device: &cpal::Device,
    cfg: &cpal::SupportedStreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, String> {
    shared.rate.store(cfg.sample_rate(), Ordering::Relaxed);
    shared
        .lock_buffer()
        .reserve(cfg.sample_rate() as usize * PREALLOC_SECONDS);
    let channels = cfg.channels() as usize;
    let sample_format = cfg.sample_format();
    let stream_cfg: cpal::StreamConfig = cfg.clone().into();
    let failed = shared.clone();
    let err_fn = move |e| {
        eprintln!("erreur flux audio: {e}");
        failed.dead.store(true, Ordering::Relaxed);
    };

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
        shared.lock_buffer().extend_from_slice(&mono);
    }
}

/// Linear resampler to 16 kHz that keeps its phase between calls.
///
/// [`resample_to_16k`] treats its input as a complete signal, which is right
/// for a one-shot buffer but wrong for streaming: there the capture is drained
/// every 200 ms and each slice would restart at phase zero, discarding the
/// fractional remainder and stitching the slices with a small phase jump five
/// times a second.
///
/// Positions are expressed in input samples relative to the start of the slice
/// being processed; index -1 denotes the previous slice's last sample, which is
/// what makes interpolation continuous across the boundary.
struct Resampler {
    /// Position of the next output sample. In `[-1, 0)` between two slices.
    pos: f64,
    /// Last sample of the previous slice (index -1 above).
    prev: Option<f32>,
}

impl Resampler {
    fn new() -> Self {
        Resampler {
            pos: 0.0,
            prev: None,
        }
    }

    fn process(&mut self, input: &[f32], from_rate: u32) -> Vec<f32> {
        // Identity: pass through untouched and leave the phase alone, so a
        // 16 kHz device never accumulates rounding error.
        if from_rate == TARGET_RATE || from_rate == 0 {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }
        let step = from_rate as f64 / TARGET_RATE as f64; // input samples per output sample
        let len = input.len() as f64;
        let prev = self.prev;
        let at = |i: isize| -> Option<f32> {
            if i < 0 {
                prev
            } else {
                input.get(i as usize).copied()
            }
        };
        let mut out = Vec::with_capacity((len / step).ceil() as usize + 1);
        while self.pos < len {
            let idx = self.pos.floor() as isize;
            // Stop one short of the end: interpolating the last position needs
            // the first sample of the next slice, which has not arrived yet.
            let (Some(a), Some(b)) = (at(idx), at(idx + 1)) else {
                break;
            };
            let frac = (self.pos - idx as f64) as f32;
            out.push(a + (b - a) * frac);
            self.pos += step;
        }
        // Re-express the pending position in the next slice's coordinates.
        self.pos -= len;
        self.prev = input.last().copied();
        out
    }
}

/// Linear resampling to 16 kHz of a complete buffer (one-shot path, file
/// transcription). For the streaming path see [`Resampler`], which keeps its
/// phase across successive slices.
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
    fn resampler_is_continuous_across_slices() {
        // A ramp makes any phase jump obvious: a correct resampler produces a
        // strictly regular output whatever the slicing.
        let input: Vec<f32> = (0..44_100).map(|i| i as f32).collect();

        let mut whole = Resampler::new();
        let reference = whole.process(&input, 44_100);

        let mut streamed = Resampler::new();
        let mut chunked = Vec::new();
        // 200 ms at 44.1 kHz is 8820 samples: deliberately not a whole number
        // of output samples, which is exactly where the phase used to be lost.
        for slice in input.chunks(8_820) {
            chunked.extend(streamed.process(slice, 44_100));
        }

        assert_eq!(reference.len(), chunked.len(), "longueurs differentes");
        for (i, (a, b)) in reference.iter().zip(&chunked).enumerate() {
            assert!((a - b).abs() < 1e-3, "echantillon {i}: {a} vs {b}");
        }
    }

    #[test]
    fn resampler_keeps_a_regular_step() {
        // On a ramp the output must advance by exactly `from_rate / 16000`
        // input units per sample, across slice boundaries included.
        let input: Vec<f32> = (0..48_000).map(|i| i as f32).collect();
        let mut r = Resampler::new();
        let mut out = Vec::new();
        for slice in input.chunks(9_600) {
            out.extend(r.process(slice, 48_000));
        }
        assert!(out.len() > 15_000, "sortie trop courte: {}", out.len());
        for w in out.windows(2) {
            assert!((w[1] - w[0] - 3.0).abs() < 1e-2, "pas irregulier: {w:?}");
        }
    }

    #[test]
    fn resampler_passes_through_at_the_target_rate() {
        let mut r = Resampler::new();
        let v = vec![0.1, 0.2, 0.3];
        assert_eq!(r.process(&v, TARGET_RATE), v);
        // And an empty slice is a no-op rather than a phase disturbance.
        assert!(r.process(&[], 48_000).is_empty());
    }

    #[test]
    fn mix_pads_and_clamps() {
        let out = mix(vec![vec![0.8, 0.8], vec![0.8]]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-6); // 1.6 clamped to 1.0
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn a_failed_start_leaves_no_stream_behind() {
        // `mix` needs an output device for the loopback. On a machine that has
        // one the start succeeds and there is nothing to assert; what must
        // never happen is a half-started recorder, so check the invariant that
        // holds either way: `is_recording()` agrees with the returned result.
        let mut rec = Recorder::new(Some("::device::inexistant::".into()), "mix".into());
        match rec.start() {
            Err(_) => assert!(
                !rec.is_recording(),
                "un demarrage en echec ne doit laisser aucun flux ouvert"
            ),
            Ok(()) => {
                assert!(rec.is_recording());
                let _ = rec.stop();
            }
        }
    }

    #[test]
    fn stream_failure_is_reported_to_the_caller() {
        let mut rec = Recorder::new(None, "mic".into());
        assert!(!rec.stream_failed(), "aucun flux ouvert: pas d'echec");
        // Stand in for cpal's error callback, which cannot be triggered without
        // real hardware (see the manual test for a device unplugged mid-take).
        let shared = Shared::new();
        rec.caps.push(shared.clone());
        assert!(!rec.stream_failed());
        shared.dead.store(true, Ordering::Relaxed);
        assert!(rec.stream_failed(), "l'echec de flux doit remonter");
    }

    #[test]
    fn feed_survives_a_poisoned_buffer() {
        // `feed` runs in cpal's real-time callback: a panic there unwinds across
        // the WASAPI FFI boundary. It must keep working on a poisoned mutex.
        let shared = Shared::new();
        let other = shared.clone();
        let poisoner = std::thread::spawn(move || {
            let _guard = other.buffer.lock().unwrap();
            panic!("empoisonnement volontaire du mutex");
        });
        assert!(poisoner.join().is_err(), "le thread devait paniquer");
        assert!(shared.buffer.is_poisoned(), "le mutex devait etre empoisonne");

        feed(&shared, &[0.5, 0.5], 1);
        assert_eq!(shared.lock_buffer().as_slice(), &[0.5, 0.5]);
    }

    #[test]
    fn feed_downmixes_and_ignores_degenerate_input() {
        // Interleaved stereo folds to the per-frame mean.
        let shared = Shared::new();
        feed(&shared, &[1.0, -1.0, 0.5, 0.5], 2);
        assert_eq!(shared.lock_buffer().as_slice(), &[0.0, 0.5]);

        // A zero channel count would divide by zero: guarded, nothing appended.
        let zero = Shared::new();
        feed(&zero, &[1.0, 2.0], 0);
        assert!(zero.lock_buffer().is_empty());

        // Empty callback buffer: no level update, no append, no panic.
        let empty = Shared::new();
        feed(&empty, &[], 1);
        assert!(empty.lock_buffer().is_empty());
    }

    #[test]
    fn boost_quiet_amplifies_soft_speech() {
        // Peak 0.05 -> capped gain (8x) brings it to 0.4, still under TARGET.
        let mut s = vec![0.05, -0.04, 0.03];
        boost_quiet(&mut s);
        assert!((s[0] - 0.4).abs() < 1e-6, "s[0]={}", s[0]);
        assert!((s[1] + 0.32).abs() < 1e-6, "s[1]={}", s[1]);
    }

    #[test]
    fn boost_quiet_leaves_normal_and_silence_untouched() {
        // Already loud enough (peak >= TARGET): unchanged.
        let mut loud = vec![0.6, -0.7];
        boost_quiet(&mut loud);
        assert_eq!(loud, vec![0.6, -0.7]);
        // Near-silence (below the noise floor): not amplified into garbage.
        let mut hush = vec![0.001, -0.002];
        boost_quiet(&mut hush);
        assert_eq!(hush, vec![0.001, -0.002]);
    }
}
