//! Transcription facade over the multi-backend ASR engine (`crate::engine`).
//!
//! The app talks to this thin wrapper with plain `&str`/`bool`/`&Path`
//! arguments; it adapts them to the engine's `AsrEngine` trait. The backend is
//! chosen from the model artifact's shape via [`EngineKind::detect`]: a ggml
//! `.bin` selects whisper.cpp (identical to the old direct `whisper-rs` path),
//! an ONNX bundle directory selects the matching backend (Parakeet, …). The
//! whisper-specific inference details (state reuse, fitted audio context, VAD,
//! the crash sentinel) now live in the engine.

use std::path::Path;

use crate::engine::{
    AsrEngine, DevicePreference, EngineConfig, EngineKind, TranscribeControl, TranscribeOptions,
    load_engine,
};

pub struct Transcriber {
    engine: Box<dyn AsrEngine>,
    /// Language the engine detected on the last `transcribe`, cached so
    /// `detected_language` keeps its old pull-based shape for callers.
    last_detected: Option<String>,
}

impl Transcriber {
    /// Load a ggml whisper model. `gpu` requests the GPU backend (Vulkan when
    /// compiled); the engine falls back to CPU when it cannot honour it.
    pub fn load(model_path: &Path, gpu: bool) -> Result<Self, String> {
        let config = EngineConfig {
            model_path: model_path.to_path_buf(),
            device: if gpu {
                DevicePreference::Auto
            } else {
                DevicePreference::Cpu
            },
            default_language: None,
            // Preserve the crash sentinel: whisper.cpp aborts the process on a
            // malformed ggml, so a marker on disk at next startup names the
            // model that killed the previous run.
            crash_marker: Some(crate::config::model_sentinel_path()),
        };
        // Pick the backend from the model artifact's shape (a ggml `.bin`
        // resolves to whisper.cpp, an ONNX bundle directory to its matching
        // backend). Today every model_path the app builds is a `.bin`, so this
        // stays whisper — but the choice is no longer hard-coded here.
        let kind = EngineKind::detect(model_path)?;
        let engine = load_engine(kind, &config)?;
        Ok(Transcriber {
            engine,
            last_detected: None,
        })
    }

    /// Transcribe mono 16 kHz f32 audio.
    /// `language=None` => automatic detection; `translate` => translate to English.
    /// `beam_size > 1` => beam search (more accurate, slower).
    /// `vad_model=Some(path)` => skip silence via the backend's internal VAD
    /// before decoding. Meant for the one-shot path; streaming passes `None`.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: Option<&str>,
        translate: bool,
        initial_prompt: Option<&str>,
        beam_size: i32,
        vad_model: Option<&Path>,
    ) -> Result<String, String> {
        let opts = TranscribeOptions {
            language: language.map(str::to_string),
            translate,
            initial_prompt: initial_prompt.map(str::to_string),
            beam_size,
            vad_model: vad_model.map(Path::to_path_buf),
        };
        let result = self
            .engine
            .transcribe(audio, &opts, &TranscribeControl::none())?;
        self.last_detected = result.detected_language;
        Ok(result.text)
    }

    /// Full English name of the language detected on the last [`Self::transcribe`]
    /// call (e.g. "french"), from the backend's own detection. Used to tell the
    /// LLM which language to answer in. `None` before any transcription or if the
    /// backend did not report one.
    pub fn detected_language(&self) -> Option<&str> {
        self.last_detected.as_deref()
    }
}
