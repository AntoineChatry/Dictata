//! Transcription via whisper.cpp (ggml `.bin` models), whisper-rs wrapper.
//!
//! The model and its inference `state` are created once (`load`) and reused
//! across calls. `set_no_context` keeps each clip independent, so reuse only
//! avoids reallocating the KV cache and compute buffers on every chunk without
//! changing the result. The expected audio is mono 16 kHz f32 (see `audio.rs`).

use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

pub struct Transcriber {
    state: WhisperState,
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
        // Flash attention shrinks the attention buffers (and is faster). Gated
        // on `gpu`: it only helps VRAM, and CPU-only builds (no Vulkan backend)
        // skip it. DTW token timestamps are not used here, so the trade-off is free.
        params.flash_attn(gpu);
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| format!("chargement modele: {e}"))?;
        // Create the inference state once and reuse it across calls. The state
        // owns an Arc to the context, so it keeps the model alive on its own and
        // we do not need to hold `ctx` separately.
        let state = ctx
            .create_state()
            .map_err(|e| format!("whisper state: {e}"))?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        Ok(Transcriber { state, n_threads })
    }

    /// Transcribe mono 16 kHz f32 audio.
    /// `language=None` => automatic detection; `translate` => translate to English.
    /// `beam_size > 1` => beam search (more accurate, slower).
    /// `vad_model=Some(path)` => skip silence with whisper.cpp VAD before
    /// decoding (less compute, fewer hallucinations). Meant for the one-shot
    /// path; the streaming path already cuts on pauses and passes `None`.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: Option<&str>,
        translate: bool,
        initial_prompt: Option<&str>,
        beam_size: i32,
        vad_model: Option<&Path>,
    ) -> Result<String, String> {
        if audio.is_empty() {
            return Ok(String::new());
        }

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
        // The state is reused across calls; disable carry-over of the previous
        // clip's tokens so each transcription stays independent (continuity is
        // handled explicitly via `initial_prompt`). Required with a reused
        // state, otherwise prior context leaks and causes repetition/drift.
        params.set_no_context(true);
        if let Some(p) = initial_prompt {
            if !p.is_empty() {
                params.set_initial_prompt(p);
            }
        }

        // Fit the encoder context to the clip and force a single segment for
        // clips that fit one window: less transient VRAM and no trailing
        // repetition hallucinated on short chunks. Clips >= 30 s keep the full
        // context, so long one-shot transcriptions are unchanged.
        params.set_audio_ctx(fitted_audio_ctx(audio.len()));
        if audio.len() as f32 / 16_000.0 <= 28.0 {
            params.set_single_segment(true);
        }

        // Optional VAD: drop silent regions before decoding. The model path
        // must be set before enabling (enable_vad panics otherwise), so guard
        // on a readable, existing file. Default Silero params are kept.
        if let Some(vad) = vad_model {
            if vad.exists() {
                if let Some(p) = vad.to_str() {
                    params.set_vad_model_path(Some(p));
                    params.enable_vad(true);
                }
            }
        }

        self.state
            .full(params, audio)
            .map_err(|e| format!("transcription: {e}"))?;

        let n = self.state.full_n_segments();
        let mut text = String::new();
        for i in 0..n {
            if let Some(seg) = self.state.get_segment(i) {
                if let Ok(s) = seg.to_str_lossy() {
                    text.push_str(&s);
                }
            }
        }
        Ok(text.trim().to_string())
    }
}

/// Encoder audio-context size (frames, 50/s) fitted to the clip duration.
///
/// whisper.cpp pads every clip to 30 s (1500 frames) and runs the encoder at
/// full size regardless of the real length. For short dictation chunks that is
/// mostly wasted compute and transient VRAM. We size the context to the next
/// 5 s step above the clip (+0.5 s safety margin so the last word is never
/// clipped), floored at 5 s (very short windows hallucinate) and capped at the
/// model maximum (1500). For clips >= 30 s this returns 1500 — identical to the
/// default, so long one-shot transcriptions are unaffected.
fn fitted_audio_ctx(n_samples: usize) -> i32 {
    const RATE: f32 = 16_000.0;
    const FRAMES_PER_SEC: f32 = 50.0;
    const FULL: f32 = 1500.0;
    let secs = n_samples as f32 / RATE + 0.5; // safety margin
    let step_s = (secs / 5.0).ceil() * 5.0; // 5 / 10 / 15 / 20 … s
    (step_s * FRAMES_PER_SEC).min(FULL) as i32
}

#[cfg(test)]
mod tests {
    use super::fitted_audio_ctx;

    const RATE: usize = 16_000;

    #[test]
    fn audio_ctx_floors_at_5s() {
        // A 1 s clip still gets a 5 s (250-frame) context.
        assert_eq!(fitted_audio_ctx(RATE), 250);
        assert_eq!(fitted_audio_ctx(RATE / 2), 250);
    }

    #[test]
    fn audio_ctx_always_covers_the_clip_with_margin() {
        // step_s >= duration + 0.5 s, so the context never clips the audio.
        for secs in 1..=29 {
            let ctx = fitted_audio_ctx(secs * RATE);
            assert!(
                ctx as f32 / 50.0 >= secs as f32 + 0.5,
                "{secs}s clip -> {ctx} frames does not cover the audio"
            );
        }
    }

    #[test]
    fn audio_ctx_caps_at_full_for_long_clips() {
        // 30 s and beyond fall back to the default full context (no regression).
        assert_eq!(fitted_audio_ctx(30 * RATE), 1500);
        assert_eq!(fitted_audio_ctx(120 * RATE), 1500);
    }
}
