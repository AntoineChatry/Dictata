//! In-memory, model-free engine: the trait's test double and reference
//! implementation.

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    EngineCapabilities, LanguageSupport, TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// Returns a canned text for any non-empty audio.
///
/// Lets the caller wire up and test their integration (one-shot, streaming,
/// dispatch) without loading a real model or depending on a native backend.
pub struct MockEngine {
    canned: String,
    caps: EngineCapabilities,
}

impl MockEngine {
    /// Creates a mock that returns `canned` for any non-empty clip.
    pub fn new(canned: impl Into<String>) -> Self {
        MockEngine {
            canned: canned.into(),
            caps: EngineCapabilities {
                name: "mock",
                supports_prompt: true,
                supports_beam: false,
                supports_translate: false,
                supports_internal_vad: false,
                languages: LanguageSupport::Any,
            },
        }
    }
}

impl AsrEngine for MockEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError> {
        // Honor the control contract even in the test double: bail on a
        // pre-set cancel flag, and always end at 100 % on success.
        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }
        if audio.is_empty() {
            return Ok(TranscriptionResult::default());
        }
        control.report_progress(100);
        Ok(TranscriptionResult {
            text: self.canned.clone(),
            // Echo the requested language, else a default: enough to exercise
            // the caller's `detected_language` path.
            detected_language: opts
                .language
                .clone()
                .or_else(|| Some("english".to_string())),
            // The mock models no timing: a single segment covering the whole
            // text, enough to exercise the caller's `segments` path.
            segments: vec![crate::engine::types::Segment {
                text: self.canned.clone(),
                start: 0.0,
                end: audio.len() as f32 / crate::engine::types::SAMPLE_RATE as f32,
            }],
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_audio_yields_empty_result() {
        let mut m = MockEngine::new("must not appear");
        let out = m
            .transcribe(&[], &TranscribeOptions::default(), &TranscribeControl::none())
            .unwrap();
        assert!(out.text.is_empty());
        assert!(out.detected_language.is_none());
        assert!(out.segments.is_empty());
    }

    #[test]
    fn non_empty_audio_returns_canned_text() {
        let mut m = MockEngine::new("bonjour");
        let out = m
            .transcribe(
                &[0.2, 0.3],
                &TranscribeOptions::default(),
                &TranscribeControl::none(),
            )
            .unwrap();
        assert_eq!(out.text, "bonjour");
        assert_eq!(out.detected_language.as_deref(), Some("english"));
        // One segment covering the whole canned text, timed against the input.
        assert_eq!(out.segments.len(), 1);
        assert_eq!(out.segments[0].text, "bonjour");
        assert_eq!(out.segments[0].start, 0.0);
    }

    #[test]
    fn requested_language_is_echoed_back() {
        let mut m = MockEngine::new("hola");
        let opts = TranscribeOptions {
            language: Some("spanish".into()),
            ..Default::default()
        };
        let out = m
            .transcribe(&[0.1], &opts, &TranscribeControl::none())
            .unwrap();
        assert_eq!(out.detected_language.as_deref(), Some("spanish"));
    }

    #[test]
    fn a_pre_set_cancel_flag_yields_cancelled() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let mut m = MockEngine::new("should not appear");
        let control = TranscribeControl {
            cancel: Some(Arc::new(AtomicBool::new(true))),
            on_progress: None,
        };
        assert!(matches!(
            m.transcribe(&[0.2, 0.3], &TranscribeOptions::default(), &control),
            Err(AsrError::Cancelled)
        ));
    }

    #[test]
    fn progress_reaches_100_on_success() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU8, Ordering};
        let seen = Arc::new(AtomicU8::new(0));
        let seen2 = seen.clone();
        let control = TranscribeControl {
            cancel: None,
            on_progress: Some(Arc::new(move |p| seen2.store(p, Ordering::Relaxed))),
        };
        let mut m = MockEngine::new("bonjour");
        m.transcribe(&[0.2, 0.3], &TranscribeOptions::default(), &control)
            .unwrap();
        assert_eq!(seen.load(Ordering::Relaxed), 100);
    }
}
