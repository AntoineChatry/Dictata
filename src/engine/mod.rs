//! `dictata-engine` — multi-backend, offline ASR abstraction for Rust.
//!
//! A single [`AsrEngine`] trait exposes transcription; each backend (whisper.cpp
//! via `whisper-rs`, NVIDIA Parakeet via `parakeet-rs`/ONNX) is one
//! implementation, enabled by a dedicated Cargo *feature* to isolate its heavy
//! dependencies.
//!
//! # Audio contract
//!
//! Every engine consumes **mono, 16 kHz, `f32`** audio in `[-1.0, 1.0]` (see
//! [`types::SAMPLE_RATE`]). This is what the caller's capture stage produces;
//! the engine does not resample.
//!
//! # Execution model
//!
//! The trait is **synchronous**: transcription runs on a worker thread on the
//! caller's side (no async runtime). *Streaming* (splitting on pauses,
//! incremental insertion) is **orchestrated by the caller**, which calls
//! [`AsrEngine::transcribe`] per chunk; the engine stays stateless between calls.
//!
//! # Example
//!
//! ```
//! use dictata::engine::{AsrEngine, TranscribeOptions, TranscribeControl};
//! use dictata::engine::engines::MockEngine;
//!
//! let mut engine = MockEngine::new("hello world");
//! let out = engine
//!     .transcribe(&[0.1, -0.1, 0.2], &TranscribeOptions::default(), &TranscribeControl::none())
//!     .unwrap();
//! assert_eq!(out.text, "hello world");
//! ```

#[cfg(feature = "diarize")]
pub mod diarize;
pub mod engines;
pub mod error;
#[cfg(feature = "vad")]
pub mod pipeline;
#[cfg(feature = "resample")]
pub mod resample;
pub mod types;
#[cfg(feature = "vad")]
pub mod vad;

pub use engines::{AsrEngine, EngineKind, load_engine};
pub use error::AsrError;
pub use types::{
    DevicePreference, EngineCapabilities, EngineConfig, LanguageSupport, Segment,
    TranscribeControl, TranscribeOptions, TranscriptionResult, SAMPLE_RATE,
};
