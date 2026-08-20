//! Zipformer transducer backend (feature `zipformer`) via ONNX Runtime.
//!
//! Runs an offline **RNN-T (transducer)** zipformer model exported by
//! [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) / icefall, split into
//! three ONNX graphs — `encoder`, `decoder` (a stateless Conv1d over the last
//! `context_size` tokens) and `joiner` — plus a `tokens.txt`. ONNX Runtime is
//! pulled through `ort` directly, like the shared VAD, so this backend needs
//! neither whisper.cpp nor Parakeet.
//!
//! The front-end is a **kaldi-compatible 80-bin log-mel fbank** (povey window,
//! `snip_edges = false`, no dither, 0.97 pre-emphasis, mel range 20 Hz..7600 Hz)
//! provided by the pure-Rust `kaldi-native-fbank` port — the exact features
//! icefall trained on. Samples stay in `[-1, 1]` (icefall's `normalize_samples`),
//! so they are not rescaled to i16 range.
//!
//! Decoding is **greedy** (one symbol per encoder frame): the joiner runs at each
//! encoder time step, the arg-max token is emitted when it is neither blank
//! (id 0) nor `<unk>`, and the stateless decoder is re-run only after an
//! emission. The initial decoder context is `context_size` blanks, matching
//! icefall's training-time padding.
//!
//! Like the other backends, it has no beam search, no decoding prompt, no
//! translation and no internal VAD — [`AsrEngine::capabilities`] says so.
//!
//! # Model
//!
//! Point [`EngineConfig::model_path`] at the **directory** of a sherpa-onnx
//! offline zipformer transducer export: an `*encoder*.onnx`, `*decoder*.onnx`,
//! `*joiner*.onnx` (int8 variants preferred when present) and `tokens.txt`.

use std::path::{Path, PathBuf};

use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use kaldi_native_fbank::{FbankComputer, FbankOptions};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    EngineCapabilities, EngineConfig, LanguageSupport, TranscribeControl, TranscribeOptions,
    TranscriptionResult, SAMPLE_RATE,
};

/// Number of mel bins the fbank front-end produces (icefall zipformer standard).
const NUM_MEL: usize = 80;
/// RNN-T blank id (icefall convention).
const BLANK_ID: i64 = 0;

/// Zipformer transducer backend. See [`AsrEngine`].
pub struct ZipformerEngine {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    caps: EngineCapabilities,
    /// Token id -> symbol, from `tokens.txt`.
    tokens: Vec<String>,
    /// Fbank configuration, cloned into a fresh (stateful) extractor per call.
    fbank_opts: FbankOptions,
    /// Stateless-decoder context width (from decoder metadata, default 2).
    context_size: usize,
    /// `<unk>` id to skip during greedy search, or `-1` when the model has none.
    unk_id: i64,
    /// Encoder input node names `[x, x_lens]`.
    enc_in: Vec<String>,
    /// Decoder input node name `[y]`.
    dec_in: String,
    /// Joiner input node names `[encoder_out, decoder_out]`.
    joiner_in: Vec<String>,
}

impl ZipformerEngine {
    /// Load a sherpa-onnx offline zipformer transducer export from the
    /// **directory** [`EngineConfig::model_path`].
    ///
    /// Fails with [`AsrError::ModelNotFound`] when the directory, an
    /// encoder/decoder/joiner `.onnx` or `tokens.txt` is absent, and
    /// [`AsrError::Load`] when a graph does not have the expected input arity of a
    /// transducer export.
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let dir = config.model_path.as_path();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "zipformer model directory not found: {}",
                dir.display()
            )));
        }
        // Windows MAX_PATH (~260) guard, mirroring the other ONNX backends.
        if cfg!(windows)
            && let Some(p) = dir.to_str()
            && p.len() > 230
        {
            return Err(AsrError::Load(format!(
                "zipformer model directory path too long ({} chars, Windows limit ~260): {}",
                p.len(),
                dir.display()
            )));
        }

        let encoder_file = find_onnx(dir, "encoder")?;
        let decoder_file = find_onnx(dir, "decoder")?;
        let joiner_file = find_onnx(dir, "joiner")?;

        let tokens_file = dir.join("tokens.txt");
        if !tokens_file.is_file() {
            return Err(AsrError::ModelNotFound(format!(
                "tokens.txt not found in {}",
                dir.display()
            )));
        }
        let tokens = load_tokens(&tokens_file)?;

        let encoder = build_session(&encoder_file)?;
        let decoder = build_session(&decoder_file)?;
        let joiner = build_session(&joiner_file)?;

        // The stateless decoder carries `context_size` and `vocab_size` in its
        // metadata (icefall/sherpa export). Read context_size into an owned value
        // now; the borrow of `decoder` must end before it moves into the struct.
        let context_size = {
            let meta = decoder
                .metadata()
                .map_err(|e| AsrError::Load(format!("zipformer decoder metadata: {e}")))?;
            let cs = meta
                .custom("context_size")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(2);
            drop(meta);
            cs
        };
        if context_size == 0 {
            return Err(AsrError::Load(
                "zipformer decoder reports context_size = 0".into(),
            ));
        }

        let enc_in = input_names(&encoder);
        if enc_in.len() != 2 {
            return Err(AsrError::Load(format!(
                "zipformer encoder expects 2 inputs (features, feature_lengths), got {}: {enc_in:?}",
                enc_in.len()
            )));
        }
        let dec_in_names = input_names(&decoder);
        if dec_in_names.len() != 1 {
            return Err(AsrError::Load(format!(
                "zipformer decoder expects 1 input, got {}: {dec_in_names:?}",
                dec_in_names.len()
            )));
        }
        let joiner_in = input_names(&joiner);
        if joiner_in.len() != 2 {
            return Err(AsrError::Load(format!(
                "zipformer joiner expects 2 inputs (encoder_out, decoder_out), got {}: {joiner_in:?}",
                joiner_in.len()
            )));
        }

        // `<unk>` is skipped during greedy search like the blank; -1 means "none".
        let unk_id = tokens
            .iter()
            .position(|t| t == "<unk>")
            .and_then(|p| i64::try_from(p).ok())
            .unwrap_or(-1);

        Ok(ZipformerEngine {
            encoder,
            decoder,
            joiner,
            caps: ZipformerEngine::capabilities_static(),
            tokens,
            fbank_opts: build_fbank_opts(),
            context_size,
            unk_id,
            enc_in,
            dec_in: dec_in_names.into_iter().next().unwrap(),
            joiner_in,
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "zipformer",
            supports_prompt: false,
            supports_beam: false,
            supports_translate: false,
            supports_internal_vad: false,
            // A transducer emits whatever tokens its `tokens.txt` defines; coverage
            // depends entirely on the loaded model (English, multilingual, ...), so
            // the backend advertises broad coverage rather than a fixed set.
            languages: LanguageSupport::Any,
        }
    }

    /// Compute kaldi-compatible 80-bin log-mel fbank features for `audio`,
    /// returned row-major as `frames * NUM_MEL` plus the frame count.
    fn compute_features(&self, audio: &[f32]) -> Result<(Vec<f32>, usize), AsrError> {
        let computer = FbankComputer::new(self.fbank_opts.clone())
            .map_err(|e| AsrError::Transcribe(format!("zipformer fbank init: {e}")))?;
        let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
        // Our contract already guarantees 16 kHz; accept_waveform panics on a
        // mismatch, which cannot happen here.
        online.accept_waveform(SAMPLE_RATE as f32, audio);
        online.input_finished();

        let frames = online.num_frames_ready();
        let mut feats = Vec::with_capacity(frames * NUM_MEL);
        for i in 0..frames {
            let frame = online.get_frame(i).ok_or_else(|| {
                AsrError::Transcribe("zipformer: fbank frame vanished mid-read".into())
            })?;
            feats.extend_from_slice(frame);
        }
        Ok((feats, frames))
    }

    /// Run the stateless decoder over the last `context_size` tokens and return
    /// its output vector (`decoder_out`, flattened).
    fn run_decoder(&mut self, context: &[i64]) -> Result<Vec<f32>, AsrError> {
        let dec_in = Value::from_array(([1_i64, context.len() as i64], context.to_vec()))
            .map_err(|e| AsrError::Transcribe(format!("zipformer decoder input: {e}")))?;
        let outputs = self
            .decoder
            .run(ort::inputs![self.dec_in.as_str() => dec_in])
            .map_err(|e| AsrError::Transcribe(format!("zipformer decoder run: {e}")))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AsrError::Transcribe(format!("zipformer decoder output: {e}")))?;
        Ok(data.to_vec())
    }

    /// Run the joiner on one encoder frame and the current decoder output, and
    /// return the raw logits over the vocabulary.
    fn run_joiner(&mut self, encoder_frame: &[f32], decoder_out: &[f32]) -> Result<Vec<f32>, AsrError> {
        let enc_val = Value::from_array(([1_i64, encoder_frame.len() as i64], encoder_frame.to_vec()))
            .map_err(|e| AsrError::Transcribe(format!("zipformer joiner encoder input: {e}")))?;
        let dec_val = Value::from_array(([1_i64, decoder_out.len() as i64], decoder_out.to_vec()))
            .map_err(|e| AsrError::Transcribe(format!("zipformer joiner decoder input: {e}")))?;
        let outputs = self
            .joiner
            .run(ort::inputs![
                self.joiner_in[0].as_str() => enc_val,
                self.joiner_in[1].as_str() => dec_val,
            ])
            .map_err(|e| AsrError::Transcribe(format!("zipformer joiner run: {e}")))?;
        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AsrError::Transcribe(format!("zipformer joiner output: {e}")))?;
        Ok(logits.to_vec())
    }
}

impl AsrEngine for ZipformerEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        _opts: &TranscribeOptions,
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

        // 1) fbank features -> [1, T, NUM_MEL].
        let (feats, frames) = self.compute_features(audio)?;
        if frames == 0 {
            // Fewer than one full frame of audio: nothing to transcribe.
            return Ok(TranscriptionResult::default());
        }

        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }

        // 2) encoder: (features, feature_lengths) -> encoder_out [1, T', C].
        let x_val = Value::from_array(([1_i64, frames as i64, NUM_MEL as i64], feats))
            .map_err(|e| AsrError::Transcribe(format!("zipformer features tensor: {e}")))?;
        let xlen_val = Value::from_array(([1_i64], vec![frames as i64]))
            .map_err(|e| AsrError::Transcribe(format!("zipformer feature_lengths tensor: {e}")))?;
        let (enc_frames, enc_dim, encoder_out) = {
            let outputs = self
                .encoder
                .run(ort::inputs![
                    self.enc_in[0].as_str() => x_val,
                    self.enc_in[1].as_str() => xlen_val,
                ])
                .map_err(|e| AsrError::Transcribe(format!("zipformer encoder run: {e}")))?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| AsrError::Transcribe(format!("zipformer encoder output: {e}")))?;
            let dims = shape.as_ref();
            if dims.len() != 3 {
                return Err(AsrError::Transcribe(format!(
                    "zipformer expected rank-3 encoder_out [1, T, C], got shape {dims:?}"
                )));
            }
            (dims[1] as usize, dims[2] as usize, data.to_vec())
        };

        // 3) greedy transducer search. The initial decoder context is
        // `context_size` blanks (icefall's training-time padding); the decoder is
        // re-run only after emitting a non-blank token.
        let mut hyp: Vec<i64> = vec![BLANK_ID; self.context_size];
        let mut decoder_out = self.run_decoder(&hyp)?;

        for t in 0..enc_frames {
            // Cancellation is cheap; check every frame. Progress is coarse (every
            // 64 frames ~= 2.5 s of audio at the usual 40 ms encoder stride).
            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            if t % 64 == 0 {
                control.report_progress((t as u64 * 100 / enc_frames as u64) as u8);
            }

            let frame = &encoder_out[t * enc_dim..(t + 1) * enc_dim];
            let logits = self.run_joiner(frame, &decoder_out)?;
            let y = argmax(&logits) as i64;
            if y != BLANK_ID && y != self.unk_id {
                hyp.push(y);
                let ctx = &hyp[hyp.len() - self.context_size..];
                decoder_out = self.run_decoder(ctx)?;
            }
        }
        control.report_progress(100);

        // Strip the initial `context_size` blanks; the rest is the transcript.
        let text = tokens_to_text(&hyp[self.context_size..], &self.tokens);
        Ok(TranscriptionResult {
            text,
            // A transducer does not detect or expose a language tag.
            detected_language: None,
            // Greedy transducer timestamps exist (per encoder frame) but need the
            // model's subsampling stride to map to seconds; not emitted yet.
            segments: Vec::new(),
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

// --- session / model helpers ------------------------------------------------

/// Build an ONNX Runtime session with the shared optimization/threading setup.
fn build_session(path: &Path) -> Result<Session, AsrError> {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8);
    Session::builder()
        .map_err(|e| AsrError::Load(format!("{e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| AsrError::Load(format!("{e}")))?
        .with_intra_threads(n_threads)
        .map_err(|e| AsrError::Load(format!("{e}")))?
        .commit_from_file(path)
        .map_err(|e| AsrError::Load(format!("zipformer load {}: {e}", path.display())))
}

/// Find the newest `.onnx` file in `dir` whose name contains `keyword`,
/// preferring an int8-quantized variant when both are present. Zipformer exports
/// do not use fixed file names (e.g. `encoder-epoch-99-avg-1.int8.onnx`), so we
/// match by substring rather than an exact stem.
fn find_onnx(dir: &Path, keyword: &str) -> Result<PathBuf, AsrError> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| AsrError::Load(format!("zipformer read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let is_onnx = p.extension().and_then(|s| s.to_str()) == Some("onnx");
            let named = p
                .file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.to_ascii_lowercase().contains(keyword))
                .unwrap_or(false);
            is_onnx && named
        })
        .collect();
    if candidates.is_empty() {
        return Err(AsrError::ModelNotFound(format!(
            "no *{keyword}*.onnx in {}",
            dir.display()
        )));
    }
    // Prefer an int8 file: its key is `false` and sorts before non-int8 (`true`).
    candidates.sort_by_key(|p| {
        let n = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        !n.contains("int8")
    });
    Ok(candidates.remove(0))
}

fn input_names(session: &Session) -> Vec<String> {
    session.inputs().iter().map(|i| i.name().to_string()).collect()
}

/// The kaldi fbank configuration icefall zipformer models were trained on:
/// 80 povey-windowed mel bins over 20 Hz..(Nyquist-400 = 7600 Hz), 0.97
/// pre-emphasis, DC removal, no dither and `snip_edges = false`, on samples kept
/// in `[-1, 1]` (no i16 rescaling). `use_energy` is forced off so the feature
/// dimension is exactly 80 (the crate defaults it on, which would append a 81st
/// energy coefficient and break the model's input).
fn build_fbank_opts() -> FbankOptions {
    let mut opts = FbankOptions::default();
    opts.frame_opts.samp_freq = SAMPLE_RATE as f32;
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.snip_edges = false;
    opts.mel_opts.num_bins = NUM_MEL;
    opts.mel_opts.low_freq = 20.0;
    opts.mel_opts.high_freq = -400.0;
    opts.use_energy = false;
    opts
}

// --- decoding (pure, testable) ----------------------------------------------

/// Index of the maximum value (first on ties), or 0 for an empty slice.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Detokenize BPE/SentencePiece ids into text: `▁` becomes a leading space and
/// byte tokens `<0xNN>` are reassembled into raw bytes, then decoded as UTF-8.
/// Character-level tokens (no `▁`) concatenate directly (e.g. CJK).
fn tokens_to_text(ids: &[i64], tokens: &[String]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        let Some(tok) = usize::try_from(id).ok().and_then(|i| tokens.get(i)) else {
            continue;
        };
        if let Some(b) = parse_byte_token(tok) {
            bytes.push(b);
            continue;
        }
        for ch in tok.chars() {
            if ch == '\u{2581}' {
                bytes.push(b' ');
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).trim().to_string()
}

/// Parse a SentencePiece byte token `<0xNN>` into its byte value.
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// Load `tokens.txt` (`<symbol> <id>` per line) into an id-indexed table.
fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| AsrError::Load(format!("zipformer tokens.txt: {e}")))?;
    let mut table: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        // The symbol is everything before the last whitespace-separated field
        // (the id); BPE pieces contain no spaces.
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
            "zipformer tokens.txt is empty or malformed".into(),
        ));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::DevicePreference;

    #[test]
    fn load_reports_missing_model_directory() {
        let cfg = EngineConfig {
            model_path: "::/no/such/zipformer/dir".into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            ZipformerEngine::load(&cfg),
            Err(AsrError::ModelNotFound(_))
        ));
    }

    #[test]
    fn capabilities_declare_no_whisper_only_options() {
        let caps = ZipformerEngine::capabilities_static();
        assert_eq!(caps.name, "zipformer");
        assert!(!caps.supports_prompt);
        assert!(!caps.supports_beam);
        assert!(!caps.supports_translate);
        assert!(!caps.supports_internal_vad);
        assert!(matches!(caps.languages, LanguageSupport::Any));
    }

    #[test]
    fn fbank_opts_yield_80_pure_mel_bins() {
        // use_energy MUST be off, otherwise the crate appends an energy coefficient
        // and the feature dim becomes 81 — a silent model-breaking mismatch.
        let opts = build_fbank_opts();
        assert!(!opts.use_energy);
        assert_eq!(opts.mel_opts.num_bins, NUM_MEL);
        let comp = FbankComputer::new(opts).expect("fbank opts valid");
        assert_eq!(comp.dim(), NUM_MEL, "feature dim must be exactly 80");
    }

    #[test]
    fn fbank_produces_finite_80dim_frames_on_a_tone() {
        // End-to-end exercise of the real kaldi-native-fbank port (no model
        // needed): a 1 s tone yields >0 frames, each of width 80 and all finite.
        let engine_opts = build_fbank_opts();
        let comp = FbankComputer::new(engine_opts).unwrap();
        let mut online = OnlineFeature::new(FeatureComputer::Fbank(comp));
        let sine: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5)
            .collect();
        online.accept_waveform(SAMPLE_RATE as f32, &sine);
        online.input_finished();
        assert!(online.num_frames_ready() > 50, "expected ~100 frames for 1 s");
        let frame = online.get_frame(0).unwrap();
        assert_eq!(frame.len(), NUM_MEL);
        assert!(frame.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn argmax_picks_the_highest_index() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        // Ties keep the first maximum (strict `>` comparison).
        assert_eq!(argmax(&[1.0, 1.0, 0.5]), 0);
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn tokens_to_text_maps_underscore_and_reassembles_bytes() {
        let tokens: Vec<String> = ["<blk>", "<unk>", "\u{2581}he", "llo", "\u{2581}world"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // "▁he" "llo" "▁world" -> "hello world" (only ▁ introduces a space).
        assert_eq!(tokens_to_text(&[2, 3, 4], &tokens), "hello world");
        // 'é' = U+00E9 = bytes 0xC3 0xA9, split across two byte tokens.
        let bt: Vec<String> = ["<0xC3>", "<0xA9>", "\u{2581}caf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(tokens_to_text(&[2, 0, 1], &bt), "caf\u{e9}");
    }

    #[test]
    fn parse_byte_token_only_when_well_formed() {
        assert_eq!(parse_byte_token("<0x20>"), Some(0x20));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("\u{2581}the"), None);
    }
}
