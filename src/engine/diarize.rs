//! Speaker diarization (feature `diarize`).
//!
//! "Who spoke when", as a standalone pass — deliberately **outside** the
//! [`crate::engine::AsrEngine`] trait. Diarization produces speaker-labelled time spans,
//! not text, so it does not fit `transcribe`; like the shared [`crate::engine::vad`] it
//! lives beside the trait, and the caller composes the two — diarize to get the
//! spans, then transcribe each span — when it wants speaker-attributed text.
//!
//! Backed by NVIDIA's streaming Sortformer v2 (4-speaker) through `parakeet-rs`,
//! so this pass requires the Parakeet backend. Bring a
//! `diar_streaming_sortformer_4spk-v2` ONNX model (from
//! [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2));
//! it is not bundled.
//!
//! # Turn-taking, not overlap
//!
//! Sortformer supports up to 4 concurrent speakers, but this offline pass is
//! tuned for **turn-taking** speech (meetings, interviews, multi-speaker
//! dictation). Heavy cross-talk is not its target.

use crate::engine::error::AsrError;
use crate::engine::types::SAMPLE_RATE;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use std::path::Path;

/// A speaker identifier, `0..4` for the 4-speaker Sortformer model.
///
/// A shared alias so a future speaker-attributed transcription API can label its
/// output with the same type, without either side depending on the other.
pub type SpeakerId = usize;

/// Post-processing knobs for turning raw per-frame speaker probabilities into
/// segments. The defaults mirror NVIDIA's tuned CallHome preset for Sortformer
/// v2; the remaining internal knobs (padding, median smoothing) keep that
/// preset's values.
#[derive(Debug, Clone, Copy)]
pub struct DiarizeConfig {
    /// A speaker turn starts when its probability reaches this value (`0.0..=1.0`).
    pub onset: f32,
    /// A speaker turn ends when its probability falls below this value.
    pub offset: f32,
    /// A turn shorter than this is discarded (milliseconds).
    pub min_speech_ms: u32,
    /// Two turns of the same speaker closer than this are merged (milliseconds).
    pub min_gap_ms: u32,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        // NVIDIA's CallHome v2 preset (diar_streaming_sortformer_4spk-v2).
        DiarizeConfig {
            onset: 0.641,
            offset: 0.561,
            min_speech_ms: 511,
            min_gap_ms: 296,
        }
    }
}

impl From<DiarizeConfig> for DiarizationConfig {
    /// Start from the tuned CallHome preset and override only the exposed knobs,
    /// so the padding and median-smoothing defaults are preserved.
    fn from(c: DiarizeConfig) -> Self {
        let mut base = DiarizationConfig::callhome();
        base.onset = c.onset;
        base.offset = c.offset;
        base.min_duration_on = c.min_speech_ms as f32 / 1000.0;
        base.min_duration_off = c.min_gap_ms as f32 / 1000.0;
        base
    }
}

/// A single-speaker time span, as **sample indices** at [`crate::engine::SAMPLE_RATE`]
/// into the buffer passed to [`Diarizer::diarize`]. Convert to seconds by
/// dividing by [`crate::engine::SAMPLE_RATE`]. `end` is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerSegment {
    /// First sample of the turn (inclusive).
    pub start: usize,
    /// One past the last sample of the turn (exclusive).
    pub end: usize,
    /// Which speaker is talking.
    pub speaker: SpeakerId,
}

impl SpeakerSegment {
    /// Length of the segment in samples.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// True when the segment carries no samples.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Offline speaker diarizer over NVIDIA Sortformer v2.
///
/// Load once, then call [`diarize`](Diarizer::diarize) per clip. It holds the
/// ONNX session; Sortformer resets its streaming state at the start of each
/// `diarize`, so successive calls are independent.
pub struct Diarizer {
    inner: Sortformer,
}

impl Diarizer {
    /// Load a Sortformer ONNX model and configure post-processing.
    ///
    /// Returns [`AsrError::ModelNotFound`] when `model_path` does not exist, and
    /// [`AsrError::Load`] when the model exists but cannot be loaded. Runs on
    /// CPU (the `parakeet-rs` default execution provider).
    pub fn load(model_path: impl AsRef<Path>, config: DiarizeConfig) -> Result<Self, AsrError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(AsrError::ModelNotFound(path.display().to_string()));
        }
        let inner = Sortformer::with_config(path, None, config.into())
            .map_err(|e| AsrError::Load(format!("{e}")))?;
        Ok(Diarizer { inner })
    }

    /// Diarize a full **mono 16 kHz `f32`** buffer, returning the speaker turns
    /// in start-time order.
    ///
    /// An empty buffer yields an empty vector without error. Inference failure
    /// surfaces as [`AsrError::Transcribe`].
    pub fn diarize(&mut self, audio: &[f32]) -> Result<Vec<SpeakerSegment>, AsrError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let raw = self
            .inner
            .diarize(audio.to_vec(), SAMPLE_RATE, 1)
            .map_err(|e| AsrError::Transcribe(format!("{e}")))?;
        Ok(raw
            .into_iter()
            .map(|s| SpeakerSegment {
                start: s.start as usize,
                end: s.end as usize,
                speaker: s.speaker_id,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_a_missing_model_reports_model_not_found() {
        let result = Diarizer::load("does/not/exist_sortformer.onnx", DiarizeConfig::default());
        assert!(matches!(result, Err(AsrError::ModelNotFound(_))));
    }

    #[test]
    fn default_config_matches_the_callhome_preset() {
        let c = DiarizeConfig::default();
        assert!((c.onset - 0.641).abs() < 1e-6);
        assert!((c.offset - 0.561).abs() < 1e-6);
        assert_eq!(c.min_speech_ms, 511);
        assert_eq!(c.min_gap_ms, 296);
    }

    #[test]
    fn config_maps_onto_the_parakeet_preset_fields() {
        let mapped: DiarizationConfig = DiarizeConfig {
            onset: 0.6,
            offset: 0.4,
            min_speech_ms: 300,
            min_gap_ms: 150,
        }
        .into();
        assert!((mapped.onset - 0.6).abs() < 1e-6);
        assert!((mapped.offset - 0.4).abs() < 1e-6);
        assert!((mapped.min_duration_on - 0.3).abs() < 1e-6);
        assert!((mapped.min_duration_off - 0.15).abs() < 1e-6);
        // An untouched knob keeps the tuned CallHome default.
        assert_eq!(mapped.median_window, 11);
    }

    #[test]
    fn segment_len_and_is_empty_are_consistent() {
        let s = SpeakerSegment {
            start: 100,
            end: 900,
            speaker: 2,
        };
        assert_eq!(s.len(), 800);
        assert!(!s.is_empty());

        let empty = SpeakerSegment {
            start: 500,
            end: 500,
            speaker: 0,
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }
}
