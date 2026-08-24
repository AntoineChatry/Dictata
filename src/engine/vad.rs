//! Shared Silero voice-activity detection (feature `vad`).
//!
//! A backend-agnostic pre-filter: run it on a mono 16 kHz buffer to find the
//! speech regions, then feed only those to whichever [`crate::engine::AsrEngine`] you
//! use. It pulls ONNX Runtime through `ort` **directly**, so it works even in a
//! whisper-only build (no Parakeet needed), and it is version-pinned to the same
//! `ort` as `parakeet-rs`, so Cargo unifies both onto a single ONNX Runtime.
//!
//! # Model
//!
//! Bring a **Silero VAD v5** ONNX model (`silero_vad.onnx`, ~2 MB, from
//! [snakers4/silero-vad](https://github.com/snakers4/silero-vad)). It is not
//! bundled — download it separately, like the ASR models. The v5 graph takes a
//! 576-sample input (64 samples of lookback context + a 512-sample hop) plus a
//! recurrent `state`, and a sample-rate input.
//!
//! # What it is not
//!
//! This is *external* VAD, independent of whisper.cpp's own internal VAD. It is
//! the one VAD every backend can share; Parakeet has none of its own, and
//! whisper's is only reachable through whisper.cpp.

use crate::engine::error::AsrError;
use crate::engine::types::SAMPLE_RATE;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

/// New samples consumed per inference step (the hop). Silero v5 advances 512
/// samples at 16 kHz per window.
const WINDOW_SAMPLES: usize = 512;

/// Lookback context prepended to every window. Silero v5 feeds the model
/// `CONTEXT_SAMPLES + WINDOW_SAMPLES` samples (64 + 512 = 576): the 64 tail
/// samples of the previous window, then the 512 new ones. **Without this
/// context the model silently returns ~0 even on clear speech** — the current
/// v5 export (producer `spox`) requires it, unlike older 512-only exports.
const CONTEXT_SAMPLES: usize = 64;

/// Length of the recurrent state tensor: shape `[2, 1, 128]` = 256 floats.
const STATE_LEN: usize = 2 * 128;

/// Tuning knobs for turning per-window speech probabilities into segments.
///
/// The defaults mirror the reference `get_speech_timestamps` from the Silero
/// project; they are a good starting point for conversational 16 kHz audio.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Speech starts when the window probability reaches this value (`0.0..=1.0`).
    pub threshold: f32,
    /// Speech shorter than this is discarded as a blip (milliseconds).
    pub min_speech_ms: u32,
    /// A silence gap shorter than this does not split a segment (milliseconds).
    pub min_silence_ms: u32,
    /// Padding added to each side of a segment so words are not clipped
    /// (milliseconds).
    pub speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// A detected speech region, as **sample indices** into the buffer passed to
/// [`SileroVad::segments`]. Convert to seconds by dividing by
/// [`crate::engine::SAMPLE_RATE`]. `end` is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    /// First speech sample (inclusive).
    pub start: usize,
    /// One past the last speech sample (exclusive).
    pub end: usize,
}

impl SpeechSegment {
    /// Length of the segment in samples.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// True when the segment carries no samples.
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// A loaded Silero VAD model.
///
/// Stateful **within** one call (the recurrent `state` carries context between
/// windows) but reset at the start of every [`segments`](Self::segments) call,
/// so successive calls are independent — matching the rest of the crate's
/// stateless-between-calls contract.
pub struct SileroVad {
    session: Session,
    config: VadConfig,
    // Input node names resolved from the model at load, so we tolerate the exact
    // graph naming instead of hard-coding positions.
    audio_input: String,
    state_input: String,
    sr_input: String,
    // Recurrent state, carried window to window; shape [2, 1, 128].
    state: Vec<f32>,
    // Lookback samples prepended to the next window; `CONTEXT_SAMPLES` long.
    context: Vec<f32>,
}

impl SileroVad {
    /// Load a Silero v5 ONNX model from `model_path` with the given config.
    ///
    /// Fails with [`AsrError::ModelNotFound`] if the file is absent, or
    /// [`AsrError::Load`] if the graph is not a recognizable Silero v5 model
    /// (wrong input set).
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        config: VadConfig,
    ) -> Result<Self, AsrError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(AsrError::ModelNotFound(path.display().to_string()));
        }

        let session: Session = Session::builder()
            .map_err(|e| AsrError::Load(format!("{e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AsrError::Load(format!("{e}")))?
            .commit_from_file(path)
            .map_err(|e| AsrError::Load(format!("{e}")))?;

        // Resolve the three input names. Silero v5 names them "input", "state",
        // "sr"; match by those, falling back to the graph's declared order
        // (audio, state, sr) when a model uses different labels.
        let names: Vec<String> =
            session.inputs().iter().map(|i| i.name().to_string()).collect();
        if names.len() != 3 {
            return Err(AsrError::Load(format!(
                "expected a Silero v5 model with 3 inputs (audio, state, sr), got {}: {names:?}",
                names.len()
            )));
        }
        let find = |wanted: &str| names.iter().find(|n| n.as_str() == wanted).cloned();
        let audio_input = find("input").unwrap_or_else(|| names[0].clone());
        let state_input = find("state").unwrap_or_else(|| names[1].clone());
        let sr_input = find("sr").unwrap_or_else(|| names[2].clone());

        Ok(SileroVad {
            session,
            config,
            audio_input,
            state_input,
            sr_input,
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT_SAMPLES],
        })
    }

    /// Detect speech regions in a mono 16 kHz buffer.
    ///
    /// Returns the segments in order, already padded and merged per the config.
    /// An empty or all-silence buffer yields an empty vector.
    pub fn segments(&mut self, audio: &[f32]) -> Result<Vec<SpeechSegment>, AsrError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let probs = self.window_probabilities(audio)?;
        Ok(probs_to_segments(&probs, WINDOW_SAMPLES, audio.len(), &self.config))
    }

    /// Convenience: concatenate only the speech regions of `audio` into a new
    /// buffer, dropping the silence. Useful to hand a backend a compact clip.
    pub fn collect_speech(&mut self, audio: &[f32]) -> Result<Vec<f32>, AsrError> {
        let segments = self.segments(audio)?;
        let total: usize = segments.iter().map(SpeechSegment::len).sum();
        let mut out = Vec::with_capacity(total);
        for seg in segments {
            out.extend_from_slice(&audio[seg.start..seg.end]);
        }
        Ok(out)
    }

    /// Run the model over every 512-sample window and collect the per-window
    /// speech probability. The recurrent state is reset first, so the result
    /// depends only on `audio`.
    fn window_probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>, AsrError> {
        self.state.iter_mut().for_each(|s| *s = 0.0);
        self.context.iter_mut().for_each(|s| *s = 0.0);
        let window_count = audio.len().div_ceil(WINDOW_SAMPLES);
        let mut probs = Vec::with_capacity(window_count);
        let mut window = [0.0f32; WINDOW_SAMPLES];
        for w in 0..window_count {
            let start = w * WINDOW_SAMPLES;
            let end = (start + WINDOW_SAMPLES).min(audio.len());
            let n = end - start;
            window[..n].copy_from_slice(&audio[start..end]);
            // Zero-pad the trailing partial window (silence reads as non-speech).
            window[n..].iter_mut().for_each(|s| *s = 0.0);
            probs.push(self.process_window(&window)?);
        }
        Ok(probs)
    }

    /// Feed one 512-sample window through the model, update the recurrent state,
    /// and return the speech probability for that window.
    fn process_window(&mut self, window: &[f32; WINDOW_SAMPLES]) -> Result<f32, AsrError> {
        // Disjoint borrows: `session` mutably for `run`, the name/state fields
        // immutably to build the inputs — split off `self` so the borrow checker
        // sees them as separate.
        let Self {
            session,
            audio_input,
            state_input,
            sr_input,
            state,
            context,
            ..
        } = self;

        // Prepend the lookback context: the model wants CONTEXT_SAMPLES + 512
        // samples (see `CONTEXT_SAMPLES`). Feeding a bare 512-sample window makes
        // it return ~0 on speech.
        let mut input = Vec::with_capacity(CONTEXT_SAMPLES + WINDOW_SAMPLES);
        input.extend_from_slice(context);
        input.extend_from_slice(window);
        let audio_val =
            Value::from_array(([1_i64, (CONTEXT_SAMPLES + WINDOW_SAMPLES) as i64], input))
                .map_err(|e| AsrError::Transcribe(format!("vad audio tensor: {e}")))?;
        let state_val = Value::from_array(([2_i64, 1, 128], state.clone()))
            .map_err(|e| AsrError::Transcribe(format!("vad state tensor: {e}")))?;
        let sr_val = Value::from_array(([1_i64], vec![SAMPLE_RATE as i64]))
            .map_err(|e| AsrError::Transcribe(format!("vad sr tensor: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                audio_input.as_str() => audio_val,
                state_input.as_str() => state_val,
                sr_input.as_str() => sr_val,
            ])
            .map_err(|e| AsrError::Transcribe(format!("vad inference: {e}")))?;

        // Silero v5 emits two outputs in order: the speech probability, then the
        // updated recurrent state. Index positionally — output names vary but the
        // order does not.
        let (_, prob) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AsrError::Transcribe(format!("vad probability: {e}")))?;
        let prob = *prob
            .first()
            .ok_or_else(|| AsrError::Transcribe("vad returned an empty probability".into()))?;

        let (_, new_state) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| AsrError::Transcribe(format!("vad state output: {e}")))?;
        if new_state.len() == STATE_LEN {
            state.copy_from_slice(new_state);
        }

        // Next window's lookback = the last CONTEXT_SAMPLES of this window.
        context.copy_from_slice(&window[WINDOW_SAMPLES - CONTEXT_SAMPLES..]);

        Ok(prob)
    }
}

/// Turn per-window speech probabilities into padded, merged speech segments.
///
/// Faithful port of Silero's reference `get_speech_timestamps` hysteresis: a
/// segment opens once the probability crosses `threshold`, and only closes after
/// the probability stays below `threshold - 0.15` for at least `min_silence_ms`
/// (short dips do not split a word). Segments shorter than `min_speech_ms` are
/// dropped, then each is padded by `speech_pad_ms` and neighbouring pads are
/// split so they never overlap. Kept pure (no model) so it is unit-testable.
fn probs_to_segments(
    probs: &[f32],
    window: usize,
    total_samples: usize,
    cfg: &VadConfig,
) -> Vec<SpeechSegment> {
    let sr = SAMPLE_RATE as usize;
    let min_speech = sr * cfg.min_speech_ms as usize / 1000;
    let min_silence = sr * cfg.min_silence_ms as usize / 1000;
    let pad = sr * cfg.speech_pad_ms as usize / 1000;
    let neg_threshold = cfg.threshold - 0.15;

    let mut segments: Vec<SpeechSegment> = Vec::new();
    let mut triggered = false;
    let mut current_start = 0usize;
    // 0 means "no pending silence"; otherwise the sample where silence began.
    let mut temp_end = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        let pos = i * window;
        if p >= cfg.threshold && temp_end != 0 {
            temp_end = 0;
        }
        if p >= cfg.threshold && !triggered {
            triggered = true;
            current_start = pos;
            continue;
        }
        if p < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = pos;
            }
            if pos - temp_end < min_silence {
                continue;
            }
            if temp_end - current_start > min_speech {
                segments.push(SpeechSegment { start: current_start, end: temp_end });
            }
            temp_end = 0;
            triggered = false;
        }
    }
    // Close a segment still open at the end of the audio.
    if triggered && total_samples - current_start > min_speech {
        segments.push(SpeechSegment { start: current_start, end: total_samples });
    }

    apply_padding(&mut segments, pad, total_samples);
    segments
}

/// Pad each segment by `pad` samples per side, clamped to `[0, total_samples]`.
/// When two segments are closer than `2 * pad`, split the gap between them so
/// the padded segments meet without overlapping (mirrors the Silero reference).
fn apply_padding(segments: &mut [SpeechSegment], pad: usize, total_samples: usize) {
    let n = segments.len();
    for i in 0..n {
        if i == 0 {
            segments[i].start = segments[i].start.saturating_sub(pad);
        }
        if i + 1 < n {
            let gap = segments[i + 1].start.saturating_sub(segments[i].end);
            if gap < 2 * pad {
                let half = gap / 2;
                segments[i].end += half;
                segments[i + 1].start = segments[i + 1].start.saturating_sub(half);
            } else {
                segments[i].end = (segments[i].end + pad).min(total_samples);
                segments[i + 1].start = segments[i + 1].start.saturating_sub(pad);
            }
        } else {
            segments[i].end = (segments[i].end + pad).min(total_samples);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 30 ms of padding at 16 kHz = 480 samples; 250 ms min speech = 4000; 100 ms
    // min silence = 1600. Windows are 512 samples.
    const WIN: usize = WINDOW_SAMPLES;

    fn cfg() -> VadConfig {
        VadConfig::default()
    }

    #[test]
    fn all_silence_yields_no_segments() {
        let probs = vec![0.0f32; 40];
        let segs = probs_to_segments(&probs, WIN, 40 * WIN, &cfg());
        assert!(segs.is_empty());
    }

    #[test]
    fn a_short_blip_is_discarded() {
        // Two speech windows (~1024 samples) is well under the 4000-sample floor.
        let mut probs = vec![0.0f32; 40];
        probs[10] = 0.9;
        probs[11] = 0.9;
        let segs = probs_to_segments(&probs, WIN, 40 * WIN, &cfg());
        assert!(segs.is_empty());
    }

    #[test]
    fn one_sustained_region_becomes_one_segment() {
        // Windows 10..30 are speech (~20*512 = 10240 samples > min_speech).
        let mut probs = vec![0.0f32; 40];
        for p in probs.iter_mut().take(30).skip(10) {
            *p = 0.9;
        }
        let total = 40 * WIN;
        let segs = probs_to_segments(&probs, WIN, total, &cfg());
        assert_eq!(segs.len(), 1);
        // Start padded left by 480, from window 10 * 512 = 5120.
        assert_eq!(segs[0].start, 10 * WIN - 480);
        // End padded right by 480, from the silence onset at window 30 = 15360.
        assert_eq!(segs[0].end, 30 * WIN + 480);
    }

    #[test]
    fn a_short_dip_does_not_split_a_segment() {
        // Speech 10..40 with a single silent window at 25: the ~512-sample gap is
        // below the 1600-sample min silence, so it stays one segment.
        let mut probs = vec![0.9f32; 50];
        for p in probs.iter_mut().take(10) {
            *p = 0.0;
        }
        probs[25] = 0.0;
        for p in probs.iter_mut().skip(40) {
            *p = 0.0;
        }
        let total = 50 * WIN;
        let segs = probs_to_segments(&probs, WIN, total, &cfg());
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn a_long_silence_splits_into_two_segments() {
        // Two speech blocks separated by ~10 silent windows (5120 > 1600).
        let mut probs = vec![0.0f32; 60];
        for p in probs.iter_mut().take(20).skip(5) {
            *p = 0.9;
        }
        for p in probs.iter_mut().take(55).skip(40) {
            *p = 0.9;
        }
        let total = 60 * WIN;
        let segs = probs_to_segments(&probs, WIN, total, &cfg());
        assert_eq!(segs.len(), 2);
        assert!(segs[0].end <= segs[1].start);
    }

    #[test]
    fn padding_never_exceeds_the_buffer() {
        // Speech runs to the very last window; padded end must clamp to total.
        let mut probs = vec![0.9f32; 30];
        for p in probs.iter_mut().take(5) {
            *p = 0.0;
        }
        let total = 30 * WIN;
        let segs = probs_to_segments(&probs, WIN, total, &cfg());
        assert_eq!(segs.len(), 1);
        assert!(segs[0].end <= total);
        assert!(segs[0].start < segs[0].end);
    }

    #[test]
    fn segment_len_and_is_empty_are_consistent() {
        let s = SpeechSegment { start: 100, end: 500 };
        assert_eq!(s.len(), 400);
        assert!(!s.is_empty());
        assert!(SpeechSegment { start: 5, end: 5 }.is_empty());
    }
}
