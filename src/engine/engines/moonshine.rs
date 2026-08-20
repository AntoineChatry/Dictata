//! Moonshine backend (feature `moonshine`) via ONNX Runtime.
//!
//! Runs the **Moonshine** English speech-to-text model (Useful Sensors), an
//! encoder-decoder transformer with autoregressive greedy decoding. It uses the
//! [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) "v1" export, split into
//! four ONNX graphs — `preprocess`, `encode`, `uncached_decode`, `cached_decode`
//! — plus a SentencePiece `tokens.txt`. ONNX Runtime is pulled through `ort`
//! directly, like the shared VAD, so this backend needs neither whisper.cpp nor
//! Parakeet.
//!
//! The pipeline is: raw waveform -> `preprocess` (learned conv front-end) ->
//! `encode` -> autoregressive decode. Decoding starts from the SOS token through
//! `uncached_decode` (which also emits the initial KV-cache states), then loops
//! `cached_decode` — feeding the previous token and threading the KV cache — and
//! takes the arg-max token each step until the EOS token or a duration-derived
//! step cap. The KV-cache tensors are moved between calls without copying.
//!
//! Moonshine is English-only, has no beam search, no decoding prompt, no
//! translation and no internal VAD — [`AsrEngine::capabilities`] says so.
//!
//! # Model
//!
//! Point [`EngineConfig::model_path`] at the **directory** of a sherpa-onnx
//! Moonshine export, containing `preprocess.onnx`, `encode(.int8).onnx`,
//! `uncached_decode(.int8).onnx`, `cached_decode(.int8).onnx` and `tokens.txt`.

use std::borrow::Cow;
use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Value};

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    EngineCapabilities, EngineConfig, LanguageSupport, TranscribeControl, TranscribeOptions,
    TranscriptionResult,
};

/// Below this many samples the learned conv front-end has too little context to
/// produce a frame; such clips yield an empty transcription instead of an error.
const MIN_SAMPLES: usize = 512;
/// Encoder frame stride in samples (the front-end downsamples the 16 kHz
/// waveform by this factor). Used only to derive the decoding step cap.
const FRAME_STRIDE: f64 = 384.0;
/// Upper bound on decoded tokens per second of audio (matches sherpa-onnx).
const TOKENS_PER_SEC: f64 = 6.0;

/// Moonshine encoder-decoder backend. See [`AsrEngine`].
pub struct MoonshineEngine {
    preprocess: Session,
    encode: Session,
    uncached: Session,
    cached: Session,
    caps: EngineCapabilities,
    /// Token id -> SentencePiece symbol, from `tokens.txt`.
    tokens: Vec<String>,
    /// Preprocess input (audio) and output (features) node names.
    prep_in: String,
    prep_out: String,
    /// Encode input names (features, then optional features_len) and output name.
    enc_in: Vec<String>,
    enc_out: String,
    /// Uncached decoder input names `[token, encoder_out, seq_len]` and output
    /// names `[logits, state_0, ...]`.
    unc_in: Vec<String>,
    unc_out: Vec<String>,
    /// Cached decoder input names `[token, encoder_out, seq_len, state_0, ...]`
    /// and output names `[logits, state_0, ...]`.
    cac_in: Vec<String>,
    cac_out: Vec<String>,
    /// Start- and end-of-sequence token ids (from `tokens.txt`, default 1/2).
    sos: i32,
    eos: i32,
}

impl MoonshineEngine {
    /// Load a sherpa-onnx Moonshine export from the **directory**
    /// [`EngineConfig::model_path`].
    ///
    /// Fails with [`AsrError::ModelNotFound`] when the directory, a model file or
    /// `tokens.txt` is absent, and [`AsrError::Load`] when a graph does not have
    /// the expected input/output arity of a Moonshine v1 export.
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let dir = config.model_path.as_path();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "moonshine model directory not found: {}",
                dir.display()
            )));
        }
        // Windows MAX_PATH (~260) guard, mirroring the other ONNX backends.
        if cfg!(windows)
            && let Some(p) = dir.to_str()
            && p.len() > 230
        {
            return Err(AsrError::Load(format!(
                "moonshine model directory path too long ({} chars, Windows limit ~260): {}",
                p.len(),
                dir.display()
            )));
        }

        let preprocess_file = pick_model(dir, "preprocess")?;
        let encode_file = pick_model(dir, "encode")?;
        let uncached_file = pick_model(dir, "uncached_decode")?;
        let cached_file = pick_model(dir, "cached_decode")?;

        let tokens_file = dir.join("tokens.txt");
        if !tokens_file.is_file() {
            return Err(AsrError::ModelNotFound(format!(
                "tokens.txt not found in {}",
                dir.display()
            )));
        }
        let tokens = load_tokens(&tokens_file)?;

        let preprocess = build_session(&preprocess_file)?;
        let encode = build_session(&encode_file)?;
        let uncached = build_session(&uncached_file)?;
        let cached = build_session(&cached_file)?;

        // Resolve node names positionally: the sherpa-onnx export binds inputs in
        // a fixed order — preprocess(audio) -> features; encode(features,
        // features_len) -> encoder_out; decoders(token, encoder_out, seq_len,
        // states...) -> (logits, states...).
        let prep_in = input_names(&preprocess);
        let prep_out = output_names(&preprocess);
        if prep_in.len() != 1 || prep_out.len() != 1 {
            return Err(AsrError::Load(format!(
                "moonshine preprocess expects 1 input and 1 output, got {} / {}",
                prep_in.len(),
                prep_out.len()
            )));
        }

        let enc_in = input_names(&encode);
        let enc_out_names = output_names(&encode);
        if !(1..=2).contains(&enc_in.len()) || enc_out_names.len() != 1 {
            return Err(AsrError::Load(format!(
                "moonshine encode expects 1-2 inputs and 1 output, got {} / {}",
                enc_in.len(),
                enc_out_names.len()
            )));
        }

        let unc_in = input_names(&uncached);
        let unc_out = output_names(&uncached);
        if unc_in.len() != 3 || unc_out.len() < 2 {
            return Err(AsrError::Load(format!(
                "moonshine uncached_decode expects 3 inputs (token, encoder_out, seq_len) and >=2 outputs (logits, states), got {} / {}",
                unc_in.len(),
                unc_out.len()
            )));
        }

        let cac_in = input_names(&cached);
        let cac_out = output_names(&cached);
        // Cached decoder: the same 3 primary inputs plus one input per state, and
        // it must consume exactly the states the uncached decoder produced.
        if cac_in.len() != 3 + (unc_out.len() - 1) || cac_out.len() != unc_out.len() {
            return Err(AsrError::Load(format!(
                "moonshine cached_decode arity mismatch: inputs {} (expected {}), outputs {} (expected {})",
                cac_in.len(),
                3 + (unc_out.len() - 1),
                cac_out.len(),
                unc_out.len()
            )));
        }

        let sos = token_id(&tokens, "<s>").unwrap_or(1);
        let eos = token_id(&tokens, "</s>").unwrap_or(2);

        Ok(MoonshineEngine {
            preprocess,
            encode,
            uncached,
            cached,
            caps: MoonshineEngine::capabilities_static(),
            tokens,
            prep_in: prep_in.into_iter().next().unwrap(),
            prep_out: prep_out.into_iter().next().unwrap(),
            enc_in,
            enc_out: enc_out_names.into_iter().next().unwrap(),
            unc_in,
            unc_out,
            cac_in,
            cac_out,
            sos,
            eos,
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "moonshine",
            supports_prompt: false,
            supports_beam: false,
            supports_translate: false,
            supports_internal_vad: false,
            languages: LanguageSupport::Set(vec!["english".to_string()]),
        }
    }

    /// Run the full preprocess -> encode -> greedy-decode pipeline on a single
    /// window (already `>= MIN_SAMPLES`) and return its detokenized text. Called
    /// once per window by [`AsrEngine::transcribe`]; borrows `self` mutably and
    /// releases it on return, so the ONNX arena settles between windows.
    fn decode_window(
        &mut self,
        audio: &[f32],
        control: &TranscribeControl,
    ) -> Result<String, AsrError> {
        // 1) preprocess: raw waveform (1, N) -> features (1, T, dim).
        let audio_val = Value::from_array(([1_i64, audio.len() as i64], audio.to_vec()))
            .map_err(|e| AsrError::Transcribe(format!("moonshine audio tensor: {e}")))?;
        let features: DynValue = {
            let mut out = self
                .preprocess
                .run(vec![(
                    Cow::from(self.prep_in.clone()),
                    SessionInputValue::from(audio_val),
                )])
                .map_err(|e| AsrError::Transcribe(format!("moonshine preprocess: {e}")))?;
            out.remove(&self.prep_out)
                .ok_or_else(|| AsrError::Transcribe("moonshine preprocess: missing output".into()))?
        };
        let feat_frames = shape_dim(&features, 1);

        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }

        // 2) encode: (features, features_len) -> encoder_out (1, T, dim).
        let encoder_out: DynValue = {
            let mut inputs: Vec<(Cow<str>, SessionInputValue)> =
                vec![(Cow::from(self.enc_in[0].clone()), (&features).into())];
            if self.enc_in.len() == 2 {
                let len_val = Value::from_array(([1_i64], vec![feat_frames as i32]))
                    .map_err(|e| AsrError::Transcribe(format!("moonshine features_len: {e}")))?;
                inputs.push((Cow::from(self.enc_in[1].clone()), len_val.into()));
            }
            let mut out = self
                .encode
                .run(inputs)
                .map_err(|e| AsrError::Transcribe(format!("moonshine encode: {e}")))?;
            out.remove(&self.enc_out)
                .ok_or_else(|| AsrError::Transcribe("moonshine encode: missing output".into()))?
        };
        let enc_frames = shape_dim(&encoder_out, 1);
        let max_len = decode_step_cap(enc_frames);

        // 3) autoregressive greedy decode.
        let mut generated: Vec<i32> = Vec::new();
        let mut seq_len: i32 = 1;

        // Uncached step: feed the SOS token, collect the first logits and states.
        let (mut next_id, mut states) = {
            let token_val = Value::from_array(([1_i64, 1_i64], vec![self.sos]))
                .map_err(|e| AsrError::Transcribe(format!("moonshine token tensor: {e}")))?;
            let seqlen_val = Value::from_array(([1_i64], vec![seq_len]))
                .map_err(|e| AsrError::Transcribe(format!("moonshine seq_len tensor: {e}")))?;
            let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
                (Cow::from(self.unc_in[0].clone()), token_val.into()),
                (Cow::from(self.unc_in[1].clone()), (&encoder_out).into()),
                (Cow::from(self.unc_in[2].clone()), seqlen_val.into()),
            ];
            let mut out = self
                .uncached
                .run(inputs)
                .map_err(|e| AsrError::Transcribe(format!("moonshine uncached decode: {e}")))?;
            let id = argmax_logits(&out, &self.unc_out[0])?;
            let states = take_states(&mut out, &self.unc_out[1..])?;
            (id, states)
        };

        // Cached loop: thread the previous token and KV cache until EOS or cap.
        while generated.len() < max_len {
            if next_id == self.eos {
                break;
            }
            generated.push(next_id);

            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            seq_len += 1;

            let token_val = Value::from_array(([1_i64, 1_i64], vec![next_id]))
                .map_err(|e| AsrError::Transcribe(format!("moonshine token tensor: {e}")))?;
            let seqlen_val = Value::from_array(([1_i64], vec![seq_len]))
                .map_err(|e| AsrError::Transcribe(format!("moonshine seq_len tensor: {e}")))?;
            let mut inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(self.cac_in.len());
            inputs.push((Cow::from(self.cac_in[0].clone()), token_val.into()));
            inputs.push((Cow::from(self.cac_in[1].clone()), (&encoder_out).into()));
            inputs.push((Cow::from(self.cac_in[2].clone()), seqlen_val.into()));
            for (name, state) in self.cac_in[3..].iter().zip(states.into_iter()) {
                inputs.push((Cow::from(name.clone()), state.into()));
            }

            let mut out = self
                .cached
                .run(inputs)
                .map_err(|e| AsrError::Transcribe(format!("moonshine cached decode: {e}")))?;
            next_id = argmax_logits(&out, &self.cac_out[0])?;
            states = take_states(&mut out, &self.cac_out[1..])?;
        }

        Ok(tokens_to_text(&generated, &self.tokens))
    }
}

impl AsrEngine for MoonshineEngine {
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

        // Enforce the audio contract; a NaN/Inf sample poisons the conv front-end.
        let audio = super::sanitize_audio(audio);
        let audio = audio.as_ref();
        if audio.len() < MIN_SAMPLES {
            return Ok(TranscriptionResult::default());
        }

        // Moonshine is a short-form model: it encodes the WHOLE clip in one pass
        // (ONNX Runtime's arena never returns its peak to the OS) and its decode
        // step cap grows with duration, so a long clip balloons the RAM and makes
        // the model hallucinate. Window the audio into bounded chunks (<= 32 s,
        // cut on the least-bad energy dip) like the Parakeet backend, decode each
        // and concatenate. Every sample is decoded exactly once (no overlap).
        let mut text = String::new();
        let total = audio.len();
        let mut start = 0;
        while start < audio.len() {
            // Cancellation checked before each expensive window; progress is the
            // fraction of samples already consumed.
            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            control.report_progress((start as u64 * 100 / total as u64) as u8);

            let rest = &audio[start..];
            let cut = super::find_min_rms_cut(rest);
            let window = &rest[..cut];
            start += cut;

            // A trailing window can fall below the conv front-end's minimum; it
            // carries no decodable content, so skip it.
            if window.len() < MIN_SAMPLES {
                continue;
            }
            let piece = self.decode_window(window, control)?;
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&piece);
            }
        }

        control.report_progress(100);
        Ok(TranscriptionResult {
            text,
            // Moonshine is English-only; report it rather than leaving it unknown.
            detected_language: Some("english".to_string()),
            // Greedy decoding carries no reliable per-token timing.
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
        .map_err(|e| AsrError::Load(format!("moonshine load {}: {e}", path.display())))
}

/// Find `<stem>.int8.onnx` (preferred) or `<stem>.onnx` in `dir`.
fn pick_model(dir: &Path, stem: &str) -> Result<std::path::PathBuf, AsrError> {
    [format!("{stem}.int8.onnx"), format!("{stem}.onnx")]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            AsrError::ModelNotFound(format!(
                "no {stem}.int8.onnx or {stem}.onnx in {}",
                dir.display()
            ))
        })
}

fn input_names(session: &Session) -> Vec<String> {
    session.inputs().iter().map(|i| i.name().to_string()).collect()
}

fn output_names(session: &Session) -> Vec<String> {
    session.outputs().iter().map(|o| o.name().to_string()).collect()
}

/// Read dimension `idx` of a value's shape (0 if out of range).
fn shape_dim(value: &DynValue, idx: usize) -> usize {
    value
        .shape()
        .as_ref()
        .get(idx)
        .map(|&d| d.max(0) as usize)
        .unwrap_or(0)
}

/// Duration-derived cap on decoded tokens: `frames * 384/16000 * 6`, at least 1.
fn decode_step_cap(enc_frames: usize) -> usize {
    let cap = enc_frames as f64 * FRAME_STRIDE / 16_000.0 * TOKENS_PER_SEC;
    (cap as usize).max(1)
}

/// Arg-max over the vocab dimension of a `(1, 1, vocab)` logits output.
fn argmax_logits(out: &ort::session::SessionOutputs, name: &str) -> Result<i32, AsrError> {
    let (_, logits) = out[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| AsrError::Transcribe(format!("moonshine logits: {e}")))?;
    argmax(logits)
        .ok_or_else(|| AsrError::Transcribe("moonshine logits: empty vocab dimension".into()))
}

/// Move the named KV-cache state outputs out of `out` (Arc-shared, no data copy)
/// so they can feed the next decoder call.
fn take_states(
    out: &mut ort::session::SessionOutputs,
    names: &[String],
) -> Result<Vec<DynValue>, AsrError> {
    names
        .iter()
        .map(|n| {
            out.remove(n)
                .ok_or_else(|| AsrError::Transcribe(format!("moonshine missing state output '{n}'")))
        })
        .collect()
}

// --- decoding (pure, testable) ----------------------------------------------

/// Index of the maximum value, or `None` for an empty slice.
fn argmax(values: &[f32]) -> Option<i32> {
    let mut best = 0usize;
    let mut best_val = *values.first()?;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Some(best as i32)
}

/// Detokenize SentencePiece ids into text: `▁` becomes a leading space and byte
/// tokens `<0xNN>` are reassembled into raw bytes, then decoded as UTF-8.
fn tokens_to_text(ids: &[i32], tokens: &[String]) -> String {
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

/// Look up a special token's id by its exact symbol.
fn token_id(tokens: &[String], symbol: &str) -> Option<i32> {
    tokens
        .iter()
        .position(|t| t == symbol)
        .and_then(|p| i32::try_from(p).ok())
}

/// Load `tokens.txt` (`<symbol> <id>` per line) into an id-indexed table.
fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| AsrError::Load(format!("moonshine tokens.txt: {e}")))?;
    let mut table: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        // The symbol is everything before the last whitespace-separated field
        // (the id); SentencePiece pieces contain no spaces.
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
            "moonshine tokens.txt is empty or malformed".into(),
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
            model_path: "::/no/such/moonshine/dir".into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            MoonshineEngine::load(&cfg),
            Err(AsrError::ModelNotFound(_))
        ));
    }

    #[test]
    fn capabilities_declare_english_only() {
        let caps = MoonshineEngine::capabilities_static();
        assert_eq!(caps.name, "moonshine");
        assert!(!caps.supports_prompt);
        assert!(!caps.supports_beam);
        assert!(!caps.supports_translate);
        assert!(!caps.supports_internal_vad);
        match caps.languages {
            LanguageSupport::Set(langs) => {
                assert_eq!(langs, vec!["english".to_string()]);
            }
            LanguageSupport::Any => panic!("Moonshine has a closed language set"),
        }
    }

    #[test]
    fn argmax_picks_the_highest_index() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), Some(1));
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), Some(0));
        // Ties keep the first maximum (strict `>` comparison).
        assert_eq!(argmax(&[1.0, 1.0, 0.5]), Some(0));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn decode_step_cap_scales_with_frames() {
        // 750 frames ~= 18 s of audio -> 750*384/16000*6 = 108 tokens.
        assert_eq!(decode_step_cap(750), 108);
        // A tiny clip still allows at least one decode step.
        assert_eq!(decode_step_cap(0), 1);
    }

    #[test]
    fn byte_tokens_parse_only_when_well_formed() {
        assert_eq!(parse_byte_token("<0x20>"), Some(0x20));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("\u{2581}the"), None);
    }

    #[test]
    fn tokens_to_text_maps_underscore_to_space() {
        let tokens: Vec<String> = ["<unk>", "<s>", "</s>", "\u{2581}hello", "\u{2581}world", "!"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // "▁hello" "▁world" "!" -> "hello world!" (only ▁ introduces a space).
        assert_eq!(tokens_to_text(&[3, 4, 5], &tokens), "hello world!");
    }

    #[test]
    fn tokens_to_text_reassembles_byte_tokens_as_utf8() {
        // 'é' is U+00E9 = bytes 0xC3 0xA9 in UTF-8, split across two byte tokens.
        let tokens: Vec<String> = ["<0xC3>", "<0xA9>", "\u{2581}caf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(tokens_to_text(&[2, 0, 1], &tokens), "caf\u{e9}");
    }

    #[test]
    fn token_id_finds_special_symbols() {
        let tokens: Vec<String> = ["<unk>", "<s>", "</s>"].iter().map(|s| s.to_string()).collect();
        assert_eq!(token_id(&tokens, "<s>"), Some(1));
        assert_eq!(token_id(&tokens, "</s>"), Some(2));
        assert_eq!(token_id(&tokens, "<pad>"), None);
    }
}
