//! whisper.cpp backend (ggml `.bin` models) via `whisper-rs`.
//!
//! The model and its inference `state` are created once ([`WhisperEngine::load`])
//! and reused across calls. `set_no_context` keeps each clip independent: reuse
//! only avoids reallocating the KV cache and compute buffers per chunk, without
//! changing the result. Expected audio is mono 16 kHz `f32`.

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    DevicePreference, EngineCapabilities, EngineConfig, LanguageSupport, Segment,
    TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// whisper.cpp backend. See [`AsrEngine`].
pub struct WhisperEngine {
    state: WhisperState,
    n_threads: i32,
    /// Config default language, applied as a fallback when a call does not force
    /// one (honours [`EngineConfig::default_language`]).
    default_language: Option<String>,
    caps: EngineCapabilities,
}

impl WhisperEngine {
    /// Loads a ggml model from `config`.
    ///
    /// [`EngineConfig::device`] drives the GPU backend: `Auto`/`Gpu` enable it
    /// when compiled (feature `vulkan`), `Cpu` disables it. `Gpu` does not fail
    /// on a CPU-only build — whisper.cpp simply falls back to CPU.
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let model_path = config.model_path.as_path();
        if !model_path.exists() {
            return Err(AsrError::ModelNotFound(model_path.display().to_string()));
        }
        let path = model_path
            .to_str()
            .ok_or_else(|| AsrError::Load("model path is not valid UTF-8".into()))?;
        // Windows MAX_PATH: past ~260 chars ggml fails to open the file with an
        // opaque error; fail early with a clear message.
        if cfg!(windows) && path.len() > 230 {
            return Err(AsrError::Load(format!(
                "model path too long ({} chars, Windows limit ~260): {}",
                path.len(),
                model_path.display()
            )));
        }
        // whisper.cpp `abort()`s on a malformed ggml instead of returning an
        // error: write a marker naming the model being loaded. A startup that
        // finds it knows the previous run died here. Written for all callers by
        // living in `load`. See [`EngineConfig::crash_marker`].
        if let Some(marker) = config.crash_marker.as_deref() {
            let loading = model_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let _ = std::fs::write(marker, &loading);
        }

        let gpu = matches!(config.device, DevicePreference::Auto | DevicePreference::Gpu);
        let mut params = WhisperContextParameters::default();
        params.use_gpu(gpu);
        // Flash attention shrinks the attention buffers (and speeds up). Gated on
        // `gpu`: it only helps VRAM, so a CPU-only build skips it. DTW timestamps
        // are not used here, so the trade-off is free.
        params.flash_attn(gpu);
        let ctx_result = WhisperContext::new_with_params(path, params);
        // Surviving this call proves there was no abort(): clear the marker
        // whatever the outcome (an abort() would have killed the process here,
        // before this line). A graceful Err means the process lived, so leaving
        // the marker would wrongly flag this model as a process-killer next boot.
        if let Some(marker) = config.crash_marker.as_deref() {
            let _ = std::fs::remove_file(marker);
        }
        let ctx = ctx_result.map_err(|e| AsrError::Load(format!("model load: {e}")))?;
        // Inference state created once and reused. The state holds an Arc to the
        // context, keeping the model alive on its own — no need to keep `ctx`.
        let state = ctx
            .create_state()
            .map_err(|e| AsrError::Load(format!("whisper state: {e}")))?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        Ok(WhisperEngine {
            state,
            n_threads,
            default_language: config.default_language.clone(),
            caps: WhisperEngine::capabilities_static(),
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "whisper.cpp",
            supports_prompt: true,
            supports_beam: true,
            supports_translate: true,
            supports_internal_vad: true,
            languages: LanguageSupport::Any,
        }
    }
}

impl AsrEngine for WhisperEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError> {
        if audio.is_empty() {
            return Ok(TranscriptionResult::default());
        }
        // Enforce the audio contract: a NaN/Inf sample can make whisper.cpp
        // abort() the process. Borrows through untouched when already clean.
        let audio = super::sanitize_audio(audio);
        let audio = audio.as_ref();

        let strategy = if opts.beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size: opts.beam_size,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.n_threads);
        params.set_translate(opts.translate);
        // The call's language wins; fall back to the config default; else None
        // (whisper auto-detect). Honours EngineConfig::default_language.
        let language = resolve_language(opts.language.as_deref(), self.default_language.as_deref());
        params.set_language(language);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        // Reused state: cut the carry-over of the previous clip's tokens to keep
        // each transcription independent (continuity goes explicitly through
        // `initial_prompt`). Mandatory with a reused state, otherwise the prior
        // context leaks in -> repetition/drift.
        params.set_no_context(true);
        if let Some(p) = opts.initial_prompt.as_deref()
            && !p.is_empty()
        {
            params.set_initial_prompt(p);
        }

        // Fit the encoder context to the clip length and force a single segment
        // for clips that fit in one window: less transient VRAM, no hallucinated
        // repetition at the end of a short chunk. A clip >= 30 s keeps the full
        // context (long one-shot unchanged).
        params.set_audio_ctx(fitted_audio_ctx(audio.len()));
        if audio.len() as f32 / 16_000.0 <= 28.0 {
            params.set_single_segment(true);
        }

        // Optional VAD: skip silence before decoding. The model path must be set
        // BEFORE enabling (`enable_vad` panics otherwise), hence the guard on a
        // readable, existing file. Default Silero params.
        if let Some(vad) = opts.vad_model.as_deref()
            && vad.exists()
            && let Some(p) = vad.to_str()
        {
            params.set_vad_model_path(Some(p));
            params.enable_vad(true);
        }

        // Cooperative cancellation + progress through whisper.cpp's own callbacks.
        // Both are wired ONLY when the caller actually supplies a channel:
        // whisper-rs 0.16's `set_*_callback_safe` `Box::into_raw`s the closure and
        // never reclaims it (no `Drop` on `FullParams`), so each wired call leaks
        // a small box and pins the captured `Arc`. Gating on `Some` keeps the
        // common uncancelled/no-progress path (streaming chunks) allocation-clean,
        // and limits the leak to opted-in long-file calls where it is one tiny box
        // per file. The closures must be `'static`; cloning the `Arc`s satisfies
        // that (the trait objects are `'static`).
        if let Some(cancel) = control.cancel.clone() {
            use std::sync::atomic::Ordering;
            params.set_abort_callback_safe(move || cancel.load(Ordering::Relaxed));
        }
        if let Some(on_progress) = control.on_progress.clone() {
            params.set_progress_callback_safe(move |p: i32| {
                on_progress(p.clamp(0, 100) as u8);
            });
        }

        let full_result = self.state.full(params, audio);
        // whisper.cpp aborts decoding when the abort callback returns true, which
        // surfaces here as an error. Disambiguate: if the caller cancelled, report
        // it as `Cancelled` (a normal interruption) rather than a transcription
        // failure — whatever `full` returned.
        if control.is_cancelled() {
            return Err(AsrError::Cancelled);
        }
        full_result.map_err(|e| AsrError::Transcribe(format!("{e}")))?;

        let n = self.state.full_n_segments();
        let mut text = String::new();
        let mut segments = Vec::new();
        for i in 0..n {
            if let Some(seg) = self.state.get_segment(i)
                && let Ok(s) = seg.to_str_lossy()
            {
                text.push_str(&s);
                // whisper timestamps are in centiseconds (100 = 1 s), already
                // absolute (whisper.cpp windows internally and reports on the
                // full input's timeline).
                segments.push(Segment {
                    text: s.trim().to_string(),
                    start: seg.start_timestamp() as f32 / 100.0,
                    end: seg.end_timestamp() as f32 / 100.0,
                });
            }
        }

        // Language detected by whisper itself, so the caller knows the language
        // even when `language` was "auto".
        let detected_language = {
            let id = self.state.full_lang_id_from_state();
            if id < 0 {
                None
            } else {
                whisper_rs::get_lang_str_full(id).map(str::to_string)
            }
        };

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            detected_language,
            segments,
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

/// Encoder audio-context size (frames, 50/s) fitted to the clip length.
///
/// whisper.cpp pads every clip to 30 s (1500 frames) and runs the encoder at
/// full size whatever the real length. For a short dictation chunk that is mostly
/// wasted compute and transient VRAM. We size the context to the 5 s step above
/// the clip (+0.5 s margin so the last word is never clipped), floored at 5 s
/// (very short windows hallucinate) and capped at the model max (1500). A clip
/// >= 30 s returns 1500 — identical to the default, long one-shot unchanged.
fn fitted_audio_ctx(n_samples: usize) -> i32 {
    const RATE: f32 = 16_000.0;
    const FRAMES_PER_SEC: f32 = 50.0;
    const FULL: f32 = 1500.0;
    let secs = n_samples as f32 / RATE + 0.5; // safety margin
    let step_s = (secs / 5.0).ceil() * 5.0; // 5 / 10 / 15 / 20 … s
    (step_s * FRAMES_PER_SEC).min(FULL) as i32
}

/// Effective language for a call: the call's own choice wins, otherwise the
/// engine's config default, otherwise `None` (whisper auto-detect). Kept pure so
/// the fallback precedence is unit-testable without a model.
fn resolve_language<'a>(call: Option<&'a str>, default: Option<&'a str>) -> Option<&'a str> {
    call.or(default)
}

#[cfg(test)]
mod tests {
    use super::{fitted_audio_ctx, resolve_language};

    const RATE: usize = 16_000;

    #[test]
    fn resolve_language_prefers_call_then_default_then_none() {
        // A forced call language wins over the config default.
        assert_eq!(resolve_language(Some("french"), Some("german")), Some("french"));
        // No call language: fall back to the config default (raw-forced default).
        assert_eq!(resolve_language(None, Some("german")), Some("german"));
        // Neither: None, i.e. whisper auto-detect.
        assert_eq!(resolve_language(None, None), None);
    }

    #[test]
    fn audio_ctx_floors_at_5s() {
        assert_eq!(fitted_audio_ctx(RATE), 250);
        assert_eq!(fitted_audio_ctx(RATE / 2), 250);
    }

    #[test]
    fn audio_ctx_always_covers_the_clip_with_margin() {
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
        assert_eq!(fitted_audio_ctx(30 * RATE), 1500);
        assert_eq!(fitted_audio_ctx(120 * RATE), 1500);
    }
}
