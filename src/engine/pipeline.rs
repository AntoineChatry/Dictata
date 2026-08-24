//! VAD-gated transcription pipeline (feature `vad`).
//!
//! Composes the shared Silero [`crate::engine::vad`] with any [`AsrEngine`]: detect the
//! speech regions, transcribe each one in turn, drop the silence, and stitch the
//! pieces back onto the **original** timeline. This is the single place the crate
//! wires a VAD and a backend together; like [`crate::engine::vad`] and [`crate::engine::diarize`]
//! it lives *beside* the trait, never inside it, so the engines stay unaware of
//! it and the caller opts in explicitly.
//!
//! # Timeline
//!
//! Two backend families need different handling, and this function covers both:
//! - backends that emit timed [`Segment`]s (whisper, Parakeet) — each segment is
//!   shifted from span-local time onto the global timeline;
//! - backends that emit none (Zipformer, SenseVoice, Moonshine) — one segment is
//!   synthesized per speech span, so the returned [`TranscriptionResult`] always
//!   carries a coherent timeline regardless of backend.

use crate::engine::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    SAMPLE_RATE, Segment, TranscribeControl, TranscribeOptions, TranscriptionResult,
};
use crate::engine::vad::SileroVad;

/// Transcribe `audio` (mono 16 kHz `f32`) through `engine`, gated by `vad`.
///
/// Runs the VAD to find speech regions, transcribes only those through `engine`,
/// concatenates the text and places every segment onto the original timeline
/// (see the module docs). Silence is dropped, so on sparse audio the backend
/// sees far fewer samples than the whole clip.
///
/// Empty or all-silence audio returns an empty [`TranscriptionResult`] without
/// error. Cancellation is checked before each span (and forwarded into the
/// backend, so a long span stays interruptible); progress is reported as the
/// fraction of *speech* samples already transcribed and always ends at 100 on
/// success. `detected_language` is the first one any span reports.
pub fn transcribe_with_vad(
    engine: &mut dyn AsrEngine,
    vad: &mut SileroVad,
    audio: &[f32],
    opts: &TranscribeOptions,
    control: &TranscribeControl,
) -> Result<TranscriptionResult, AsrError> {
    if audio.is_empty() {
        return Ok(TranscriptionResult::default());
    }
    let spans = vad.segments(audio)?;
    let total_speech: usize = spans.iter().map(|s| s.len()).sum();
    if total_speech == 0 {
        return Ok(TranscriptionResult::default());
    }

    let sr = SAMPLE_RATE as f32;
    let mut text = String::new();
    let mut detected_language: Option<String> = None;
    let mut segments: Vec<Segment> = Vec::new();
    let mut consumed = 0usize;

    for span in spans {
        // Cancellation between spans (fast), then forwarded into the backend so a
        // long span stays interruptible. Progress is driven here, not by the
        // backend, so its own hook is left off.
        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }
        let inner_control = TranscribeControl {
            cancel: control.cancel.clone(),
            on_progress: None,
        };

        let clip = &audio[span.start..span.end];
        let inner = engine.transcribe(clip, opts, &inner_control)?;

        place_result(
            inner,
            span.start,
            span.end,
            sr,
            &mut text,
            &mut segments,
            &mut detected_language,
        );

        consumed += span.len();
        control.report_progress((consumed as u64 * 100 / total_speech as u64) as u8);
    }
    control.report_progress(100);

    Ok(TranscriptionResult {
        text,
        detected_language,
        segments,
    })
}

/// Fold one span's [`TranscriptionResult`] into the running output, placing its
/// segments onto the global timeline.
///
/// Pure (no VAD, no engine) so the re-offset math is unit-testable on its own.
/// `span_start`/`span_end` are sample indices at [`SAMPLE_RATE`]; `sr` is that
/// rate as `f32`. Appends the span's text (space-separated) and either shifts the
/// backend's own segments by the span offset or synthesizes one covering the
/// whole span when the backend emitted none.
fn place_result(
    inner: TranscriptionResult,
    span_start: usize,
    span_end: usize,
    sr: f32,
    text: &mut String,
    segments: &mut Vec<Segment>,
    detected_language: &mut Option<String>,
) {
    let offset = span_start as f32 / sr;

    if !inner.text.is_empty() {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&inner.text);
    }
    if detected_language.is_none() {
        *detected_language = inner.detected_language;
    }

    if inner.segments.is_empty() {
        // Backend without timing: synthesize a segment spanning the speech region
        // so the timeline stays complete. Skip when the span produced no text.
        if !inner.text.is_empty() {
            segments.push(Segment {
                text: inner.text,
                start: offset,
                end: span_end as f32 / sr,
            });
        }
    } else {
        // Backend with timing: shift each segment from span-local onto global time.
        for mut seg in inner.segments {
            seg.start += offset;
            seg.end += offset;
            segments.push(seg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(t: &str, start: f32, end: f32) -> Segment {
        Segment { text: t.into(), start, end }
    }

    #[test]
    fn a_backend_without_timing_gets_a_synthesized_span_segment() {
        // Zipformer/SenseVoice/Moonshine path: no inner segments, one synthesized
        // segment covering the whole span, timed from the span offset.
        let inner = TranscriptionResult {
            text: "hello there".into(),
            detected_language: None,
            segments: Vec::new(),
        };
        let mut text = String::new();
        let mut segments = Vec::new();
        let mut lang = None;
        // Span [16000, 48000) samples = [1.0 s, 3.0 s).
        place_result(inner, 16_000, 48_000, SAMPLE_RATE as f32, &mut text, &mut segments, &mut lang);

        assert_eq!(text, "hello there");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello there");
        assert!((segments[0].start - 1.0).abs() < 1e-6);
        assert!((segments[0].end - 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_backend_with_timing_has_its_segments_shifted_onto_the_global_timeline() {
        // whisper/Parakeet path: inner segments are span-local; each must be
        // shifted by the span offset (here 1.0 s).
        let inner = TranscriptionResult {
            text: "one two".into(),
            detected_language: Some("english".into()),
            segments: vec![seg("one", 0.0, 0.5), seg("two", 0.5, 1.2)],
        };
        let mut text = String::new();
        let mut segments = Vec::new();
        let mut lang = None;
        place_result(inner, 16_000, 48_000, SAMPLE_RATE as f32, &mut text, &mut segments, &mut lang);

        assert_eq!(text, "one two");
        assert_eq!(lang.as_deref(), Some("english"));
        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 1.0).abs() < 1e-6);
        assert!((segments[0].end - 1.5).abs() < 1e-6);
        assert!((segments[1].start - 1.5).abs() < 1e-6);
        assert!((segments[1].end - 2.2).abs() < 1e-6);
    }

    #[test]
    fn spans_are_concatenated_with_a_space_and_first_language_wins() {
        let sr = SAMPLE_RATE as f32;
        let mut text = String::new();
        let mut segments = Vec::new();
        let mut lang = None;

        place_result(
            TranscriptionResult { text: "first".into(), detected_language: Some("french".into()), segments: Vec::new() },
            0, 16_000, sr, &mut text, &mut segments, &mut lang,
        );
        place_result(
            TranscriptionResult { text: "second".into(), detected_language: Some("english".into()), segments: Vec::new() },
            32_000, 48_000, sr, &mut text, &mut segments, &mut lang,
        );

        assert_eq!(text, "first second");
        // First non-None language wins; a later span does not overwrite it.
        assert_eq!(lang.as_deref(), Some("french"));
        assert_eq!(segments.len(), 2);
        assert!((segments[1].start - 2.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_span_result_adds_no_text_and_no_segment() {
        let mut text = String::from("kept");
        let mut segments = Vec::new();
        let mut lang = None;
        place_result(
            TranscriptionResult::default(),
            16_000, 32_000, SAMPLE_RATE as f32, &mut text, &mut segments, &mut lang,
        );
        // No spurious separator, no synthesized empty segment.
        assert_eq!(text, "kept");
        assert!(segments.is_empty());
    }
}
