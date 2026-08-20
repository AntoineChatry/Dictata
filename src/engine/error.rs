//! Unified error type for the engine layer.

use thiserror::Error;

/// Errors returned by an [`crate::engine::AsrEngine`] or by loading a backend.
///
/// A `From<AsrError> for String` conversion is provided for callers still
/// working with `Result<_, String>` at their boundaries: a
/// `.map_err(String::from)` (or `?` into `String`) hooks in without rewriting
/// the existing call sites.
#[derive(Debug, Error)]
pub enum AsrError {
    /// The model file / directory does not exist.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The model exists but failed to load (invalid format, EP unavailable,
    /// memory, etc.).
    #[error("model load failed: {0}")]
    Load(String),

    /// Inference failed.
    #[error("transcription failed: {0}")]
    Transcribe(String),

    /// Backend requested but not compiled into this binary (Cargo feature off)
    /// or unavailable on the machine (e.g. missing execution provider).
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    /// Operation unsupported by this backend (e.g. translation on a model that
    /// does not do it). Prefer silently ignoring an unsupported *option*; reserve
    /// this variant for what a caller must know.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// The caller cancelled the transcription through
    /// [`crate::engine::TranscribeControl`] before it finished. Distinct from a real
    /// failure on purpose: the caller treats it as a normal interruption, not an
    /// error, the way `io::ErrorKind::Interrupted` is handled.
    #[error("transcription cancelled")]
    Cancelled,
}

impl From<AsrError> for String {
    fn from(e: AsrError) -> Self {
        e.to_string()
    }
}
