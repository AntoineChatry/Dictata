//! SenseVoice backend (feature `sensevoice`) via ONNX Runtime.
//!
//! Runs the **SenseVoiceSmall** multilingual CTC model exported by
//! [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Chinese, English,
//! Japanese, Korean, Cantonese). ONNX Runtime is pulled through `ort` directly,
//! like the shared VAD, so this backend needs neither whisper.cpp nor Parakeet.
//!
//! The model expects a FunASR front-end that this module reimplements in pure
//! Rust: **kaldi fbank (80 mel) -> LFR stacking -> CMVN**. Its CMVN statistics,
//! LFR window and language ids are read from the ONNX model metadata (the values
//! sherpa-onnx bakes in at export time), so nothing is hard-coded to one model.
//!
//! Like the other backends, it has no beam search, no decoding prompt, no
//! translation and no internal VAD — [`AsrEngine::capabilities`] says so.
//!
//! # Model
//!
//! Point [`EngineConfig::model_path`] at the **directory** of a sherpa-onnx
//! SenseVoice export, containing `model.int8.onnx` (or `model.onnx`) and
//! `tokens.txt`.

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use realfft::RealFftPlanner;

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    EngineCapabilities, EngineConfig, LanguageSupport, TranscribeControl, TranscribeOptions,
    TranscriptionResult,
};

// --- kaldi fbank parameters (FunASR SenseVoice front-end) -------------------
// The model was trained with torchaudio/kaldi fbank using a hamming window, no
// dither, 0.97 pre-emphasis and 80 mel bins at 16 kHz. These must match training
// or the features (and therefore the transcription) are wrong.
const FBANK_SR: f32 = 16_000.0;
/// Frame length in samples (25 ms at 16 kHz).
const FRAME_LEN: usize = 400;
/// Frame shift in samples (10 ms at 16 kHz).
const FRAME_SHIFT: usize = 160;
/// FFT size: the next power of two above `FRAME_LEN`.
const FFT_SIZE: usize = 512;
/// Number of real-FFT output bins (`FFT_SIZE / 2 + 1`).
const NUM_FFT_BINS: usize = FFT_SIZE / 2 + 1;
/// Mel filterbank size before LFR stacking.
const NUM_MEL: usize = 80;
const PREEMPH: f32 = 0.97;
const MEL_LOW_HZ: f32 = 20.0;
const MEL_HIGH_HZ: f32 = 8_000.0;

/// SenseVoice CTC backend. See [`AsrEngine`].
pub struct SenseVoiceEngine {
    session: Session,
    caps: EngineCapabilities,
    /// Token id -> symbol, from `tokens.txt`.
    tokens: Vec<String>,
    /// CMVN: applied as `(feat + neg_mean) * inv_stddev` per LFR frame.
    neg_mean: Vec<f32>,
    inv_stddev: Vec<f32>,
    /// LFR stacking window and shift (from model metadata).
    lfr_m: usize,
    lfr_n: usize,
    /// Stacked feature dimension (`NUM_MEL * lfr_m`, i.e. 560 for the default).
    feat_dim: usize,
    /// Whether the model wants samples already normalized to `[-1, 1]`. When
    /// false (the SenseVoice default) the waveform is scaled back to i16 range.
    normalize_samples: bool,
    /// CTC blank id (0 for the sherpa-onnx SenseVoice export).
    blank_id: usize,
    /// Language ids from the model's `lid_dict` (metadata).
    lang_auto: i32,
    lang_zh: i32,
    lang_en: i32,
    lang_ja: i32,
    lang_ko: i32,
    lang_yue: i32,
    /// Text-normalization id feeding the `text_norm` input (with inverse text
    /// normalization on).
    with_itn: i32,
    /// Model input node names, resolved at load.
    x_name: String,
    xlen_name: String,
    lang_name: String,
    tnorm_name: String,
    /// Default language when a call does not force one (`None` = auto).
    default_language: Option<String>,
}

impl SenseVoiceEngine {
    /// Load a sherpa-onnx SenseVoice export from the **directory**
    /// [`EngineConfig::model_path`].
    ///
    /// Fails with [`AsrError::ModelNotFound`] when the directory, the model file
    /// or `tokens.txt` is absent, and [`AsrError::Load`] when the graph is not a
    /// recognizable SenseVoice export (missing CMVN metadata or wrong input set).
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let dir = config.model_path.as_path();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "sensevoice model directory not found: {}",
                dir.display()
            )));
        }
        // Windows MAX_PATH (~260) guard, mirroring the Parakeet backend: ONNX
        // Runtime opens the model file inside this directory and fails with an
        // opaque error past the limit.
        if cfg!(windows)
            && let Some(p) = dir.to_str()
            && p.len() > 230
        {
            return Err(AsrError::Load(format!(
                "sensevoice model directory path too long ({} chars, Windows limit ~260): {}",
                p.len(),
                dir.display()
            )));
        }

        let model_file = ["model.int8.onnx", "model.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.is_file())
            .ok_or_else(|| {
                AsrError::ModelNotFound(format!(
                    "no model.int8.onnx or model.onnx in {}",
                    dir.display()
                ))
            })?;
        let tokens_file = dir.join("tokens.txt");
        if !tokens_file.is_file() {
            return Err(AsrError::ModelNotFound(format!(
                "tokens.txt not found in {}",
                dir.display()
            )));
        }
        let tokens = load_tokens(&tokens_file)?;

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 8);

        let session: Session = Session::builder()
            .map_err(|e| AsrError::Load(format!("{e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AsrError::Load(format!("{e}")))?
            .with_intra_threads(n_threads)
            .map_err(|e| AsrError::Load(format!("{e}")))?
            .commit_from_file(&model_file)
            .map_err(|e| AsrError::Load(format!("sensevoice load: {e}")))?;

        let meta = session
            .metadata()
            .map_err(|e| AsrError::Load(format!("sensevoice metadata: {e}")))?;

        let lfr_m = meta_usize(&meta, "lfr_window_size", 7);
        let lfr_n = meta_usize(&meta, "lfr_window_shift", 6);
        validate_lfr(lfr_m, lfr_n).map_err(AsrError::Load)?;
        let feat_dim = NUM_MEL * lfr_m;
        let neg_mean = parse_float_vec(meta.custom("neg_mean"), "neg_mean")?;
        let inv_stddev = parse_float_vec(meta.custom("inv_stddev"), "inv_stddev")?;
        if neg_mean.len() != feat_dim || inv_stddev.len() != feat_dim {
            return Err(AsrError::Load(format!(
                "sensevoice CMVN size mismatch: expected {feat_dim}, got neg_mean={} inv_stddev={}",
                neg_mean.len(),
                inv_stddev.len()
            )));
        }
        let normalize_samples = meta
            .custom("normalize_samples")
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        let blank_id = meta_usize(&meta, "blank_id", 0);
        // Read the remaining metadata into owned values now: `meta` borrows
        // `session`, and that borrow must end before `session` moves into the
        // struct below.
        let lang_auto = meta_i32(&meta, "lang_auto", 0);
        let lang_zh = meta_i32(&meta, "lang_zh", 3);
        let lang_en = meta_i32(&meta, "lang_en", 4);
        let lang_ja = meta_i32(&meta, "lang_ja", 11);
        let lang_ko = meta_i32(&meta, "lang_ko", 12);
        let lang_yue = meta_i32(&meta, "lang_yue", 13);
        let with_itn = meta_i32(&meta, "with_itn", 14);

        // Input node names: the SenseVoice graph has exactly x, x_length,
        // language, text_norm. Match by name, fall back to declared order.
        let in_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        if in_names.len() != 4 {
            return Err(AsrError::Load(format!(
                "expected a SenseVoice model with 4 inputs (x, x_length, language, text_norm), got {}: {in_names:?}",
                in_names.len()
            )));
        }
        let find = |wanted: &str, pos: usize| {
            in_names
                .iter()
                .find(|n| n.as_str() == wanted)
                .cloned()
                .unwrap_or_else(|| in_names[pos].clone())
        };

        // Release the metadata's borrow of `session` before moving `session`.
        drop(meta);

        Ok(SenseVoiceEngine {
            session,
            caps: SenseVoiceEngine::capabilities_static(),
            tokens,
            neg_mean,
            inv_stddev,
            lfr_m,
            lfr_n,
            feat_dim,
            normalize_samples,
            blank_id,
            lang_auto,
            lang_zh,
            lang_en,
            lang_ja,
            lang_ko,
            lang_yue,
            with_itn,
            x_name: find("x", 0),
            xlen_name: find("x_length", 1),
            lang_name: find("language", 2),
            tnorm_name: find("text_norm", 3),
            default_language: config.default_language.clone(),
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "sensevoice",
            supports_prompt: false,
            supports_beam: false,
            supports_translate: false,
            supports_internal_vad: false,
            languages: LanguageSupport::Set(
                ["chinese", "english", "japanese", "korean", "cantonese"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
        }
    }

    /// Map a requested language (or the default) to the model's integer id.
    /// Unknown or absent languages fall back to auto-detection.
    fn lang_id_for(&self, requested: Option<&str>) -> i32 {
        let name = requested
            .or(self.default_language.as_deref())
            .map(|s| s.to_ascii_lowercase());
        match name.as_deref() {
            None | Some("") | Some("auto") => self.lang_auto,
            Some("zh") | Some("chinese") | Some("mandarin") => self.lang_zh,
            Some("en") | Some("english") => self.lang_en,
            Some("ja") | Some("japanese") => self.lang_ja,
            Some("ko") | Some("korean") => self.lang_ko,
            Some("yue") | Some("cantonese") => self.lang_yue,
            _ => self.lang_auto,
        }
    }
}

impl AsrEngine for SenseVoiceEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError> {
        if audio.is_empty() {
            return Ok(TranscriptionResult::default());
        }
        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }
        control.report_progress(0);

        // Enforce the audio contract; a NaN/Inf sample poisons the fbank.
        let audio = super::sanitize_audio(audio);
        let audio = audio.as_ref();

        // The model was trained on i16-range samples (metadata normalize_samples
        // = 0), while our contract is [-1, 1]; scale back so the CMVN statistics
        // line up with training.
        let scale = if self.normalize_samples { 1.0 } else { 32_768.0 };

        // Constant across windows: the language / text-norm ids and the input node
        // names do not depend on the audio, so resolve them once. Cloning the
        // names also frees the immutable borrow of `self` before the mutable
        // `self.session.run` inside the loop.
        let lang_id = self.lang_id_for(opts.language.as_deref());
        let with_itn = self.with_itn;
        let blank = self.blank_id;
        let x_name = self.x_name.clone();
        let xlen_name = self.xlen_name.clone();
        let lang_name = self.lang_name.clone();
        let tnorm_name = self.tnorm_name.clone();

        // SenseVoice is a short-form CTC model that runs the WHOLE sequence through
        // ONNX Runtime in one pass: on a long clip the arena balloons (it never
        // returns its peak to the OS) and the model, trained on <= 30 s, degrades.
        // Window the audio into bounded chunks (<= 32 s, cut on the least-bad
        // energy dip) exactly like the Parakeet backend, decode each and
        // concatenate. Every sample is transcribed exactly once (no overlap).
        let mut text = String::new();
        let mut detected_language: Option<String> = None;
        let total = audio.len();
        let mut start = 0;
        while start < audio.len() {
            // Cooperative cancellation, checked before the expensive pass; progress
            // is the fraction of samples already consumed.
            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            control.report_progress((start as u64 * 100 / total as u64) as u8);

            let rest = &audio[start..];
            let cut = super::find_min_rms_cut(rest);
            let window = &rest[..cut];
            start += cut;

            let (fbank, frames) = compute_fbank(window, scale);
            if frames == 0 {
                // Fewer than one full frame in this window: nothing to decode.
                continue;
            }
            let (mut x, t_lfr) = apply_lfr(&fbank, frames, NUM_MEL, self.lfr_m, self.lfr_n);
            apply_cmvn(&mut x, t_lfr, self.feat_dim, &self.neg_mean, &self.inv_stddev);

            let x_val = Value::from_array(([1_i64, t_lfr as i64, self.feat_dim as i64], x))
                .map_err(|e| AsrError::Transcribe(format!("sensevoice x tensor: {e}")))?;
            let xlen_val = Value::from_array(([1_i64], vec![t_lfr as i32]))
                .map_err(|e| AsrError::Transcribe(format!("sensevoice x_length tensor: {e}")))?;
            let lang_val = Value::from_array(([1_i64], vec![lang_id]))
                .map_err(|e| AsrError::Transcribe(format!("sensevoice language tensor: {e}")))?;
            let tnorm_val = Value::from_array(([1_i64], vec![with_itn]))
                .map_err(|e| AsrError::Transcribe(format!("sensevoice text_norm tensor: {e}")))?;

            // Extract token ids inside a block so the borrow of `session` ends
            // before we read `self.tokens` below.
            let ids = {
                let outputs = self
                    .session
                    .run(ort::inputs![
                        x_name.as_str() => x_val,
                        xlen_name.as_str() => xlen_val,
                        lang_name.as_str() => lang_val,
                        tnorm_name.as_str() => tnorm_val,
                    ])
                    .map_err(|e| AsrError::Transcribe(format!("sensevoice inference: {e}")))?;

                let (shape, logits) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AsrError::Transcribe(format!("sensevoice logits: {e}")))?;
                let dims = shape.as_ref();
                if dims.len() != 3 {
                    return Err(AsrError::Transcribe(format!(
                        "sensevoice expected rank-3 logits [1, T, vocab], got shape {dims:?}"
                    )));
                }
                let out_frames = dims[1] as usize;
                let vocab = dims[2] as usize;
                greedy_ctc(logits, out_frames, vocab, blank)
            };

            // The first decoded token is the language tag (`<|en|>` etc.); keep the
            // one from the first window that yields it (the clip is one language).
            if detected_language.is_none() {
                detected_language = ids
                    .first()
                    .and_then(|&id| self.tokens.get(id))
                    .and_then(|tag| lang_name_from_tag(tag));
            }
            // SenseVoice prepends 4 meta tokens (language, emotion, event, itn)
            // before the transcript; skip them.
            let piece = tokens_to_text(&ids, &self.tokens, 4);
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&piece);
            }
        }

        // A cancellation requested during the last non-preemptible run is reported
        // rather than a full Ok.
        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }
        control.report_progress(100);
        Ok(TranscriptionResult {
            text,
            detected_language,
            // CTC greedy decoding carries no reliable per-token timing, so no
            // timed segments are emitted (the empty-segments contract).
            segments: Vec::new(),
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

// --- metadata helpers -------------------------------------------------------

fn meta_usize(meta: &ort::session::ModelMetadata, key: &str, default: usize) -> usize {
    meta.custom(key)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn meta_i32(meta: &ort::session::ModelMetadata, key: &str, default: i32) -> i32 {
    meta.custom(key)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Parse a float vector stored in model metadata. sherpa-onnx SenseVoice writes
/// the CMVN vectors comma-separated; other exports may use whitespace. Accept
/// either by splitting on commas and whitespace, ignoring empty tokens (so a
/// trailing separator does not produce a spurious parse error).
fn parse_float_vec(value: Option<String>, key: &str) -> Result<Vec<f32>, AsrError> {
    let value = value.ok_or_else(|| {
        AsrError::Load(format!(
            "sensevoice metadata missing '{key}' (not a sherpa-onnx SenseVoice export?)"
        ))
    })?;
    value
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f32>()
                .map_err(|e| AsrError::Load(format!("sensevoice bad '{key}' value '{t}': {e}")))
        })
        .collect()
}

/// Load `tokens.txt` (`<symbol> <id>` per line) into an id-indexed table.
fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| AsrError::Load(format!("sensevoice tokens.txt: {e}")))?;
    let mut table: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        // The symbol is everything before the last whitespace-separated field
        // (the id); SenseVoice pieces contain no spaces.
        let Some((sym, id)) = line.rsplit_once([' ', '\t']) else {
            continue;
        };
        let Ok(id) = id.trim().parse::<usize>() else {
            continue;
        };
        if id >= table.len() {
            table.resize(id + 1, String::new());
        }
        table[id] = sym.to_string();
    }
    if table.is_empty() {
        return Err(AsrError::Load(
            "sensevoice tokens.txt is empty or malformed".into(),
        ));
    }
    Ok(table)
}

// --- feature extraction (pure, testable) ------------------------------------

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Triangular mel filterbank as a dense `NUM_MEL * NUM_FFT_BINS` matrix (kaldi
/// mel scale, `[MEL_LOW_HZ, MEL_HIGH_HZ]`).
fn mel_filterbank() -> Vec<f32> {
    let mel_low = hz_to_mel(MEL_LOW_HZ);
    let mel_high = hz_to_mel(MEL_HIGH_HZ);
    let delta = (mel_high - mel_low) / (NUM_MEL + 1) as f32;
    let mut bank = vec![0.0f32; NUM_MEL * NUM_FFT_BINS];
    for m in 0..NUM_MEL {
        let left = mel_low + m as f32 * delta;
        let center = mel_low + (m + 1) as f32 * delta;
        let right = mel_low + (m + 2) as f32 * delta;
        for i in 0..NUM_FFT_BINS {
            let freq = i as f32 * FBANK_SR / FFT_SIZE as f32;
            let mel = hz_to_mel(freq);
            let w = if mel > left && mel < right {
                if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                }
            } else {
                0.0
            };
            bank[m * NUM_FFT_BINS + i] = w;
        }
    }
    bank
}

/// Compute log-mel fbank features (kaldi/FunASR: hamming window, DC removal,
/// 0.97 pre-emphasis, power spectrum, 80 mel bins). Returns the features
/// row-major as `frames * NUM_MEL` and the frame count. `scale` is applied to
/// the samples first (i16 rescaling). Fewer than one frame yields no rows.
fn compute_fbank(samples: &[f32], scale: f32) -> (Vec<f32>, usize) {
    if samples.len() < FRAME_LEN {
        return (Vec::new(), 0);
    }
    let num_frames = 1 + (samples.len() - FRAME_LEN) / FRAME_SHIFT;
    let hamming: Vec<f32> = (0..FRAME_LEN)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LEN - 1) as f32).cos())
        .collect();
    let bank = mel_filterbank();

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(FFT_SIZE);
    let mut buf = vec![0.0f32; FFT_SIZE];
    let mut spectrum = r2c.make_output_vec();

    let mut out = Vec::with_capacity(num_frames * NUM_MEL);
    let mut frame = [0.0f32; FRAME_LEN];
    for f in 0..num_frames {
        let start = f * FRAME_SHIFT;
        for i in 0..FRAME_LEN {
            frame[i] = samples[start + i] * scale;
        }
        // Remove DC offset.
        let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
        for x in frame.iter_mut() {
            *x -= mean;
        }
        // Pre-emphasis (kaldi order: after DC removal, before windowing).
        for i in (1..FRAME_LEN).rev() {
            frame[i] -= PREEMPH * frame[i - 1];
        }
        frame[0] -= PREEMPH * frame[0];
        // Hamming window, then zero-pad to the FFT size.
        for i in 0..FRAME_LEN {
            buf[i] = frame[i] * hamming[i];
        }
        for b in buf.iter_mut().skip(FRAME_LEN) {
            *b = 0.0;
        }
        r2c.process(&mut buf, &mut spectrum)
            .expect("realfft length invariant");
        for m in 0..NUM_MEL {
            let row = &bank[m * NUM_FFT_BINS..(m + 1) * NUM_FFT_BINS];
            let mut energy = 0.0f32;
            for (i, c) in spectrum.iter().enumerate() {
                energy += (c.re * c.re + c.im * c.im) * row[i];
            }
            out.push(energy.max(f32::EPSILON).ln());
        }
    }
    (out, num_frames)
}

/// Reject non-positive LFR parameters read from model metadata (semi-trusted
/// input). `m == 0` would underflow `(m - 1) / 2` and `n == 0` would divide by
/// zero in `apply_lfr`, both turning a corrupt/crafted model into a panic or OOM
/// at transcribe time. Kept pure so the guard is unit-testable.
fn validate_lfr(m: usize, n: usize) -> Result<(), String> {
    if m == 0 || n == 0 {
        return Err(format!(
            "sensevoice invalid LFR metadata: window_size={m}, window_shift={n} (both must be >= 1)"
        ));
    }
    Ok(())
}

/// Low-frame-rate stacking: stack `m` consecutive frames with shift `n`,
/// left-padding by `(m-1)/2` copies of the first frame and right-padding the
/// tail with the last frame. Input is `frames * dim`; output is `t_lfr * (dim*m)`.
fn apply_lfr(feats: &[f32], frames: usize, dim: usize, m: usize, n: usize) -> (Vec<f32>, usize) {
    if frames == 0 {
        return (Vec::new(), 0);
    }
    let t_lfr = (frames as f32 / n as f32).ceil() as usize;
    let left_pad = (m - 1) / 2;
    let padded_frames = frames + left_pad;
    let mut padded = vec![0.0f32; padded_frames * dim];
    for i in 0..left_pad {
        padded[i * dim..(i + 1) * dim].copy_from_slice(&feats[0..dim]);
    }
    padded[left_pad * dim..].copy_from_slice(&feats[..frames * dim]);

    let feat_dim = dim * m;
    let mut out = vec![0.0f32; t_lfr * feat_dim];
    let last = &padded[(padded_frames - 1) * dim..padded_frames * dim];
    for i in 0..t_lfr {
        let start = i * n;
        let end = (start + m).min(padded_frames);
        let avail = end - start;
        out[i * feat_dim..i * feat_dim + avail * dim]
            .copy_from_slice(&padded[start * dim..end * dim]);
        for j in avail..m {
            out[i * feat_dim + j * dim..i * feat_dim + (j + 1) * dim].copy_from_slice(last);
        }
    }
    (out, t_lfr)
}

/// Apply CMVN in place: `(feat + neg_mean) * inv_stddev` per row.
fn apply_cmvn(feats: &mut [f32], t: usize, dim: usize, neg_mean: &[f32], inv_stddev: &[f32]) {
    for row in 0..t {
        for d in 0..dim {
            feats[row * dim + d] = (feats[row * dim + d] + neg_mean[d]) * inv_stddev[d];
        }
    }
}

// --- decoding (pure, testable) ----------------------------------------------

/// CTC greedy decoding: per-frame argmax, drop the blank id, collapse
/// consecutive duplicates (matching sherpa-onnx's greedy search).
fn greedy_ctc(logits: &[f32], frames: usize, vocab: usize, blank: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut prev: i64 = -1;
    for t in 0..frames {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let mut best = 0usize;
        let mut best_val = row[0];
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = j;
            }
        }
        if best != blank && best as i64 != prev {
            out.push(best);
        }
        prev = best as i64;
    }
    out
}

/// Join decoded token ids into text, skipping the first `skip` (SenseVoice's
/// meta tokens) and turning the BPE word-boundary marker `▁` into a space.
fn tokens_to_text(ids: &[usize], tokens: &[String], skip: usize) -> String {
    let mut s = String::new();
    for &id in ids.iter().skip(skip) {
        if let Some(tok) = tokens.get(id) {
            s.push_str(tok);
        }
    }
    s.replace('\u{2581}', " ").trim().to_string()
}

/// Turn a SenseVoice language tag token (`<|en|>`) into an English language
/// name, or `None` if it is not a recognized tag.
fn lang_name_from_tag(tag: &str) -> Option<String> {
    let inner = tag.trim_start_matches("<|").trim_end_matches("|>");
    let name = match inner {
        "zh" => "chinese",
        "en" => "english",
        "ja" => "japanese",
        "ko" => "korean",
        "yue" => "cantonese",
        _ => return None,
    };
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::DevicePreference;

    #[test]
    fn load_reports_missing_model_directory() {
        let cfg = EngineConfig {
            model_path: "::/no/such/sensevoice/dir".into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            SenseVoiceEngine::load(&cfg),
            Err(AsrError::ModelNotFound(_))
        ));
    }

    #[test]
    fn capabilities_declare_a_closed_language_set() {
        let caps = SenseVoiceEngine::capabilities_static();
        assert_eq!(caps.name, "sensevoice");
        assert!(!caps.supports_prompt);
        assert!(!caps.supports_beam);
        assert!(!caps.supports_translate);
        assert!(!caps.supports_internal_vad);
        match caps.languages {
            LanguageSupport::Set(langs) => {
                assert_eq!(langs.len(), 5);
                assert!(langs.iter().any(|l| l == "english"));
            }
            LanguageSupport::Any => panic!("SenseVoice has a closed language set"),
        }
    }

    fn sine(freq: f32, secs: f32) -> Vec<f32> {
        let n = (FBANK_SR * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / FBANK_SR).sin() * 0.5)
            .collect()
    }

    #[test]
    fn fbank_frame_count_follows_the_kaldi_formula() {
        // 1 s at 16 kHz: 1 + (16000 - 400) / 160 = 98 frames.
        let (feats, frames) = compute_fbank(&sine(440.0, 1.0), 1.0);
        assert_eq!(frames, 98);
        assert_eq!(feats.len(), frames * NUM_MEL);
    }

    #[test]
    fn fbank_is_empty_below_one_frame() {
        let (feats, frames) = compute_fbank(&vec![0.1; FRAME_LEN - 1], 1.0);
        assert_eq!(frames, 0);
        assert!(feats.is_empty());
    }

    #[test]
    fn fbank_peak_mel_bin_rises_with_tone_frequency() {
        // The dominant mel bin of a high tone must sit above that of a low tone:
        // a direction check on the mel mapping without asserting exact energies.
        let peak = |freq: f32| {
            let (feats, frames) = compute_fbank(&sine(freq, 0.5), 32_768.0);
            // Average each mel bin across frames, take the argmax bin.
            let mut sums = vec![0.0f32; NUM_MEL];
            for f in 0..frames {
                for m in 0..NUM_MEL {
                    sums[m] += feats[f * NUM_MEL + m];
                }
            }
            sums.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert!(
            peak(500.0) < peak(3000.0),
            "a 3 kHz tone should peak in a higher mel bin than a 500 Hz tone"
        );
    }

    #[test]
    fn lfr_params_must_be_positive() {
        // Corrupt/crafted metadata with a zero LFR knob is rejected at load,
        // never reaching apply_lfr where it would panic or OOM.
        assert!(validate_lfr(0, 6).is_err());
        assert!(validate_lfr(7, 0).is_err());
        assert!(validate_lfr(0, 0).is_err());
        assert!(validate_lfr(7, 6).is_ok());
    }

    #[test]
    fn lfr_shape_and_left_padding() {
        // 4 frames of dim 2, m=3, n=2 -> t_lfr = ceil(4/2) = 2, feat_dim = 6.
        // Frames: [0,1],[2,3],[4,5],[6,7]; left_pad = 1 (copy of frame 0).
        let feats = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (out, t_lfr) = apply_lfr(&feats, 4, 2, 3, 2);
        assert_eq!(t_lfr, 2);
        assert_eq!(out.len(), 2 * 6);
        // Row 0 starts at padded index 0: [pad(f0), f0, f1] = [0,1, 0,1, 2,3].
        assert_eq!(&out[0..6], &[0.0, 1.0, 0.0, 1.0, 2.0, 3.0]);
        // Row 1 starts at padded index 2: [f1, f2, f3] = [2,3, 4,5, 6,7].
        assert_eq!(&out[6..12], &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn lfr_right_pads_tail_with_last_frame() {
        // 2 frames, dim 1, m=3, n=2 -> t_lfr = 1. left_pad = 1 (copy of f0).
        // padded = [f0, f0, f1] = [10, 10, 20]; row0 = [10,10,20], full.
        let (out, t_lfr) = apply_lfr(&[10.0, 20.0], 2, 1, 3, 2);
        assert_eq!(t_lfr, 1);
        assert_eq!(out, vec![10.0, 10.0, 20.0]);
    }

    #[test]
    fn cmvn_applies_shift_then_scale() {
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        apply_cmvn(&mut feats, 2, 2, &[1.0, -1.0], &[2.0, 0.5]);
        // (1+1)*2=4, (2-1)*0.5=0.5, (3+1)*2=8, (4-1)*0.5=1.5
        assert_eq!(feats, vec![4.0, 0.5, 8.0, 1.5]);
    }

    #[test]
    fn parse_float_vec_accepts_comma_and_whitespace() {
        // sherpa-onnx SenseVoice stores the CMVN vectors comma-separated.
        assert_eq!(
            parse_float_vec(Some("-8.31,-8.60,-9.62".into()), "neg_mean").unwrap(),
            vec![-8.31, -8.60, -9.62]
        );
        // Whitespace-separated and mixed/padded inputs parse the same, and a
        // trailing separator does not add a bogus entry.
        assert_eq!(
            parse_float_vec(Some("1.0 2.0\t3.0".into()), "inv_stddev").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            parse_float_vec(Some(" 1.0, 2.0 , 3.0, ".into()), "neg_mean").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        // A genuinely non-numeric token is still rejected.
        assert!(parse_float_vec(Some("1.0,x,3.0".into()), "neg_mean").is_err());
        // Missing metadata is reported, not silently empty.
        assert!(parse_float_vec(None, "neg_mean").is_err());
    }

    #[test]
    fn greedy_ctc_collapses_and_drops_blank() {
        // vocab 3, blank = 0. Frames (argmax): 1,1,0,2,0,2 -> 1,2,2.
        let vocab = 3;
        let frame = |best: usize| {
            let mut r = vec![0.0f32; vocab];
            r[best] = 1.0;
            r
        };
        let mut logits = Vec::new();
        for b in [1, 1, 0, 2, 0, 2] {
            logits.extend(frame(b));
        }
        assert_eq!(greedy_ctc(&logits, 6, vocab, 0), vec![1, 2, 2]);
    }

    #[test]
    fn tokens_to_text_skips_meta_and_maps_underscore() {
        let tokens: Vec<String> = ["<blank>", "<|en|>", "<|HAPPY|>", "<|Speech|>", "<|withitn|>", "\u{2581}hello", "\u{2581}world"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // ids: 4 meta tokens then two words.
        let ids = vec![1, 2, 3, 4, 5, 6];
        assert_eq!(tokens_to_text(&ids, &tokens, 4), "hello world");
    }

    #[test]
    fn language_tag_parses_to_name() {
        assert_eq!(lang_name_from_tag("<|en|>").as_deref(), Some("english"));
        assert_eq!(lang_name_from_tag("<|yue|>").as_deref(), Some("cantonese"));
        assert_eq!(lang_name_from_tag("<|Speech|>"), None);
    }
}
