//! Transcription via whisper.cpp (ggml `.bin` models), whisper-rs wrapper.
//!
//! The model is loaded once (`load`), then each transcription creates an
//! ephemeral `state`. The expected audio is mono 16 kHz f32 (see `audio.rs`).

use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Transcriber {
    ctx: WhisperContext,
    n_threads: i32,
}

impl Transcriber {
    /// Load a ggml model. `gpu` enables the GPU backend if the binary is
    /// compiled with one (Vulkan/CUDA); otherwise the option has no effect.
    pub fn load(model_path: &Path, gpu: bool) -> Result<Self, String> {
        if !model_path.exists() {
            return Err(format!("model not found: {}", model_path.display()));
        }
        let path = model_path.to_str().ok_or("path model non-UTF8")?;
        // Windows MAX_PATH: past ~260 chars ggml fails to open the file with
        // an opaque error; fail early with an actionable message instead.
        if cfg!(windows) && path.len() > 230 {
            return Err(format!(
                "Model PATH too long ({} characters, Windows limit ~260) : {}",
                path.len(),
                model_path.display()
            ));
        }
        let mut params = WhisperContextParameters::default();
        params.use_gpu(gpu);
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| format!("chargement modele: {e}"))?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        Ok(Transcriber { ctx, n_threads })
    }

    /// Transcribe mono 16 kHz f32 audio.
    /// `language=None` => automatic detection; `translate` => translate to English.
    /// `beam_size > 1` => beam search (more accurate, slower).
    pub fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
        translate: bool,
        initial_prompt: Option<&str>,
        beam_size: i32,
    ) -> Result<String, String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("whisper state: {e}"))?;

        let strategy = if beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.n_threads);
        params.set_translate(translate);
        params.set_language(language);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        if let Some(p) = initial_prompt {
            if !p.is_empty() {
                params.set_initial_prompt(p);
            }
        }

        state
            .full(params, audio)
            .map_err(|e| format!("transcription: {e}"))?;

        let n = state.full_n_segments();
        let mut text = String::new();
        for i in 0..n {
            if let Some(seg) = state.get_segment(i) {
                if let Ok(s) = seg.to_str_lossy() {
                    text.push_str(&s);
                }
            }
        }
        Ok(text.trim().to_string())
    }
}
