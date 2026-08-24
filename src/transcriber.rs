//! Transcription facade over the multi-backend ASR engine (`crate::engine`).
//!
//! The app talks to this thin wrapper with plain `&str`/`bool`/`&Path`
//! arguments; it adapts them to the engine's `AsrEngine` trait. The backend is
//! chosen from the model artifact's shape via [`EngineKind::detect`]: a ggml
//! `.bin` selects whisper.cpp (identical to the old direct `whisper-rs` path),
//! an ONNX bundle directory selects the matching backend (Parakeet, …). The
//! whisper-specific inference details (state reuse, fitted audio context, VAD,
//! the crash sentinel) now live in the engine.
//!
//! # VAD
//!
//! whisper.cpp carries its own internal VAD (fed the ggml Silero model through
//! [`TranscribeOptions::vad_model`]). The ONNX backends have none, so when the
//! caller asks for VAD on one of them this facade gates silence through the
//! shared external Silero ONNX VAD ([`crate::engine::pipeline::transcribe_with_vad`]).

use std::path::Path;

use crate::engine::{
    AsrEngine, DevicePreference, EngineConfig, EngineKind, TranscribeControl, TranscribeOptions,
    TranscriptionResult, load_engine,
};
#[cfg(feature = "vad")]
use crate::engine::vad::{SileroVad, VadConfig};

pub struct Transcriber {
    engine: Box<dyn AsrEngine>,
    /// Backend chosen at load time. whisper uses its own internal VAD; the ONNX
    /// backends have none and go through the shared external Silero VAD.
    kind: EngineKind,
    /// Language the engine detected on the last `transcribe`, cached so
    /// `detected_language` keeps its old pull-based shape for callers.
    last_detected: Option<String>,
    /// External Silero ONNX VAD, loaded on first use for the non-whisper
    /// backends and cached to reuse its ONNX session. `None` until then.
    #[cfg(feature = "vad")]
    silero: Option<SileroVad>,
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
        // backend).
        let kind = EngineKind::detect(model_path)?;
        let engine = load_engine(kind, &config)?;
        Ok(Transcriber {
            engine,
            kind,
            last_detected: None,
            #[cfg(feature = "vad")]
            silero: None,
        })
    }

    /// Transcribe mono 16 kHz f32 audio.
    /// `language=None` => automatic detection; `translate` => translate to English.
    /// `beam_size > 1` => beam search (more accurate, slower).
    /// `vad_model=Some(path)` => skip silence before decoding. For whisper this
    /// is the ggml Silero model fed to whisper.cpp's internal VAD; for the ONNX
    /// backends it is the external Silero ONNX model, applied via the shared
    /// VAD-gated pipeline. Meant for the one-shot path; streaming passes `None`.
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
            // whisper.cpp reads its internal VAD from `vad_model`; the ONNX
            // backends ignore this field and use the external VAD below.
            vad_model: if self.kind == EngineKind::Whisper {
                vad_model.map(Path::to_path_buf)
            } else {
                None
            },
        };
        let result = self.transcribe_inner(audio, &opts, vad_model)?;
        self.last_detected = result.detected_language;
        Ok(result.text)
    }

    /// Run the backend, gating silence through the external Silero VAD for the
    /// non-whisper backends when the caller supplied a VAD model.
    #[cfg_attr(not(feature = "vad"), allow(unused_variables))]
    fn transcribe_inner(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        vad_model: Option<&Path>,
    ) -> Result<TranscriptionResult, String> {
        #[cfg(feature = "vad")]
        {
            // Non-whisper backends have no internal VAD: when the caller asks
            // for VAD, gate silence through the shared external Silero model.
            if self.kind != EngineKind::Whisper
                && let Some(vad_path) = vad_model
            {
                if self.silero.is_none() {
                    self.silero = Some(SileroVad::load(vad_path, VadConfig::default())?);
                }
                let vad = self.silero.as_mut().unwrap();
                return Ok(crate::engine::pipeline::transcribe_with_vad(
                    self.engine.as_mut(),
                    vad,
                    audio,
                    opts,
                    &TranscribeControl::none(),
                )?);
            }
        }
        Ok(self
            .engine
            .transcribe(audio, opts, &TranscribeControl::none())?)
    }

    /// Full English name of the language detected on the last [`Self::transcribe`]
    /// call (e.g. "french"), from the backend's own detection. Used to tell the
    /// LLM which language to answer in. `None` before any transcription or if the
    /// backend did not report one.
    pub fn detected_language(&self) -> Option<&str> {
        self.last_detected.as_deref()
    }
}
