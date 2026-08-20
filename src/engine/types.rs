//! Types shared by all backends: input, output, options, capabilities, load
//! configuration.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sample rate every engine expects, in Hz.
///
/// Audio passed to [`crate::engine::AsrEngine::transcribe`] must be **mono**, at this
/// rate, `f32` in `[-1.0, 1.0]`. The engine does not resample: that stays the
/// caller's responsibility.
pub const SAMPLE_RATE: u32 = 16_000;

/// Options for one transcription, passed **per call** (never stored on the
/// engine).
///
/// It is a *superset*: a backend cleanly ignores what it does not support (query
/// [`EngineCapabilities`] up front). The per-call design is deliberate — the
/// engine can be shared between the one-shot path and streaming, so any
/// per-usage state (VAD included) must travel through here, not the constructor.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Forced language (name or code, backend-dependent). `None` = auto-detect.
    pub language: Option<String>,
    /// Translate to English. Ignored by backends without translation.
    pub translate: bool,
    /// Decoding prompt: custom vocabulary and/or continuity with already-emitted
    /// text. Honoured by whisper-like backends; ignored otherwise.
    pub initial_prompt: Option<String>,
    /// Beam width. `<= 1` = greedy decoding. Relevant for beam-search backends
    /// (whisper); ignored otherwise.
    pub beam_size: i32,
    /// VAD model to apply before decoding (skip silence). Reserved for the
    /// one-shot path of backends with an internal VAD; leave `None` in streaming.
    pub vad_model: Option<PathBuf>,
}

/// Cooperative control for a single [`crate::engine::AsrEngine::transcribe`] call:
/// cancellation and progress. Passed **by call**, like [`TranscribeOptions`].
///
/// Both channels are optional and default to inert ([`TranscribeControl::none`]).
/// A backend polls them at its own granularity — whisper.cpp via its native
/// per-step callbacks, Parakeet between its internal windows (~30 s) — so
/// cancellation is *cooperative*, not instant, and progress is coarse.
///
/// The types are deliberately the primitives the caller already has: an
/// `Arc<AtomicBool>` for cancel (the same flag the caller's cancellation path
/// flips) and a plain `Fn(u8)` for progress. `cancel` is an `Option` on purpose: a backend
/// only wires up its (possibly allocation-leaking, on whisper.cpp) native abort
/// hook when a real flag is present, so the common uncancelled path pays nothing.
pub struct TranscribeControl {
    /// Cancellation flag. When it reads `true`, the backend stops as soon as it
    /// next checks and returns [`crate::engine::AsrError::Cancelled`] (never a partial
    /// `Ok`). `None` = never cancelled; share the *same* `Arc` you already use
    /// for app-level cancellation to unify both.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Progress sink, called with an integer percentage in `0..=100` as the
    /// backend advances (monotonic, may skip values, always ends at 100 on
    /// success). `Arc<dyn Fn>` rather than `Box` because whisper.cpp requires a
    /// `'static` callback, so it is cloned into the native hook. `None` = no
    /// progress reporting.
    pub on_progress: Option<Arc<dyn Fn(u8) + Send + Sync>>,
}

impl TranscribeControl {
    /// An inert control: never cancelled, no progress. Use it for the plain
    /// fire-and-forget transcription (`&TranscribeControl::none()`).
    pub fn none() -> Self {
        TranscribeControl {
            cancel: None,
            on_progress: None,
        }
    }

    /// `true` once the caller has requested cancellation. Cheap; a backend calls
    /// it at each check point.
    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// Report progress (`0..=100`) to the sink if one is set; a no-op otherwise.
    pub fn report_progress(&self, pct: u8) {
        if let Some(cb) = &self.on_progress {
            cb(pct.min(100));
        }
    }
}

impl Default for TranscribeControl {
    fn default() -> Self {
        TranscribeControl::none()
    }
}

/// A timed span of transcribed text (subtitle-style). Times are in **seconds**
/// from the start of the audio passed to [`crate::engine::AsrEngine::transcribe`].
///
/// Granularity is backend-defined but sentence/phrase-level in practice: whisper
/// returns its native segments, Parakeet groups tokens into sentences. Use it
/// for subtitles/SRT, alignment or click-to-seek; ignore it and read
/// [`TranscriptionResult::text`] when you only want the words.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Segment text (trimmed).
    pub text: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

/// Result of one transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionResult {
    /// Transcribed text (already trimmed of leading/trailing whitespace).
    pub text: String,
    /// Language detected by the model, when available (full English name for
    /// whisper, e.g. `"french"`). Useful to the caller when `language` was
    /// "auto" and a downstream stage needs to know the language.
    pub detected_language: Option<String>,
    /// Timed segments covering `text`, in order. Empty when the backend produced
    /// none (e.g. empty audio) or does not emit timing. Times are seconds from
    /// the start of the input audio. See [`Segment`].
    pub segments: Vec<Segment>,
}

/// A backend's language coverage.
#[derive(Debug, Clone)]
pub enum LanguageSupport {
    /// Any language / broad coverage with auto-detection (e.g. whisper).
    Any,
    /// Closed set of supported languages (e.g. Parakeet TDT — 25 languages).
    Set(Vec<String>),
}

/// What a backend can do. Drives the UI (grey out unsupported `beam`/`prompt`)
/// and dispatch, without instantiating or guessing.
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Human-readable backend name (e.g. `"whisper.cpp"`, `"parakeet-tdt"`).
    pub name: &'static str,
    /// Honours [`TranscribeOptions::initial_prompt`].
    pub supports_prompt: bool,
    /// Honours [`TranscribeOptions::beam_size`].
    pub supports_beam: bool,
    /// Honours [`TranscribeOptions::translate`].
    pub supports_translate: bool,
    /// Honours [`TranscribeOptions::vad_model`] (internal VAD).
    pub supports_internal_vad: bool,
    /// Supported languages.
    pub languages: LanguageSupport,
}

/// Compute-device preference, interpreted by each backend.
///
/// Deliberately abstract: `Gpu`/`Auto` map to Vulkan on whisper (when compiled)
/// and to DirectML on Parakeet (when available). The backend falls back to CPU
/// when it cannot satisfy the preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Best device available for this backend.
    #[default]
    Auto,
    /// Force CPU.
    Cpu,
    /// Force GPU if the backend has one; otherwise a load error.
    Gpu,
}

/// Engine load parameters.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Model location: a **file** (ggml `.bin` for whisper) or a **directory**
    /// (ONNX encoder/decoder/joiner + tokens bundle for Parakeet).
    pub model_path: PathBuf,
    /// Desired device.
    pub device: DevicePreference,
    /// Default language when a call does not specify one. `None` = auto.
    pub default_language: Option<String>,
    /// Optional crash marker. Some native backends (whisper.cpp) can `abort()`
    /// the process on a corrupt model — not catchable by `Result` nor
    /// `catch_unwind`. When this path is set, the engine writes the model name
    /// here **before** the risky load and clears it once the load returns; a
    /// file still on disk at the next startup tells the caller which model
    /// killed the process. `None` = disabled.
    pub crash_marker: Option<PathBuf>,
}
