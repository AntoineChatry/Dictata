//! NVIDIA Parakeet backend (ONNX models) via `parakeet-rs` / ONNX Runtime.
//!
//! Uses the **TDT** model (`ParakeetTDT`), multilingual with language
//! auto-detection and predicted punctuation. The model is a **directory**
//! containing `encoder-model.onnx`, `decoder_joint-model.onnx` and `vocab.txt`.
//!
//! Unlike whisper.cpp, Parakeet supports neither a decoding prompt
//! (`initial_prompt`), nor beam search, nor translation, nor an internal VAD:
//! [`AsrEngine::capabilities`] says so, and the corresponding
//! [`TranscribeOptions`] are ignored.

use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};

use super::AsrEngine;
use crate::engine::error::AsrError;
use crate::engine::types::{
    DevicePreference, EngineCapabilities, EngineConfig, LanguageSupport, Segment,
    TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// Parakeet TDT backend. See [`AsrEngine`].
pub struct ParakeetEngine {
    model: ParakeetTDT,
    caps: EngineCapabilities,
}

impl ParakeetEngine {
    /// Loads a Parakeet TDT model from the **directory** [`EngineConfig::model_path`].
    ///
    /// [`EngineConfig::device`] picks the ONNX Runtime execution provider:
    /// `Cpu` forces CPU; `Auto`/`Gpu` request DirectML when the crate is built
    /// with `parakeet-directml`, otherwise fall back to CPU. GPU providers fall
    /// back to CPU anyway when the hardware/driver is missing (`ort`'s
    /// guarantee). [`EngineConfig::crash_marker`] is ignored: `ort` returns
    /// `Result`s, it does not `abort()` the process.
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let dir = config.model_path.as_path();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "parakeet model directory not found: {}",
                dir.display()
            )));
        }
        // Windows MAX_PATH (~260): parakeet-rs opens `encoder-model.onnx` and
        // `decoder_joint-model.onnx` inside this directory, so the directory path
        // plus the longest leaf (~24 chars) must stay well under the limit or
        // ONNX Runtime fails to open the file with an opaque error. Fail early
        // with a clear message, mirroring the whisper backend's guard. Only
        // checked when the path is UTF-8 measurable; a non-UTF-8 path is left to
        // the loader (parakeet-rs takes a `Path`, not a `&str`).
        if cfg!(windows)
            && let Some(p) = dir.to_str()
            && p.len() > 230
        {
            return Err(AsrError::Load(format!(
                "parakeet model directory path too long ({} chars, Windows limit ~260): {}",
                p.len(),
                dir.display()
            )));
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);

        let exec = ExecutionConfig::new()
            .with_execution_provider(select_provider(config.device))
            .with_intra_threads(n_threads as usize);

        let model = ParakeetTDT::from_pretrained(dir, Some(exec))
            .map_err(|e| AsrError::Load(format!("parakeet load: {e}")))?;

        Ok(ParakeetEngine {
            model,
            caps: ParakeetEngine::capabilities_static(),
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "parakeet-tdt",
            supports_prompt: false,
            supports_beam: false,
            supports_translate: false,
            supports_internal_vad: false,
            languages: LanguageSupport::Set(tdt_v3_languages()),
        }
    }
}

impl AsrEngine for ParakeetEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        _opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError> {
        if audio.is_empty() {
            return Ok(TranscriptionResult::default());
        }
        // Enforce the audio contract before feature extraction: a NaN/Inf sample
        // poisons Parakeet's mel spectrogram into garbage text. Borrows through
        // untouched when already clean.
        let audio = super::sanitize_audio(audio);
        let audio = audio.as_ref();

        // Parakeet runs the encoder over the WHOLE sequence at once (no internal
        // windowing, unlike whisper.cpp), and ONNX Runtime's arena never returns
        // its peak to the OS: on a long clip the RAM balloons without bound and
        // stays high. So we split into windows <= 32 s, cutting on the least-bad
        // energy dip, and free each window between passes so the arena settles on
        // a single window's peak. Every sample is transcribed exactly once — no
        // overlap — so no word is duplicated at a seam (same RMS-cut rationale as
        // the streaming path; an overlap+dedup approach is unreliable without the
        // token context Parakeet does not carry between calls).
        let mut text = String::new();
        let mut segments: Vec<Segment> = Vec::new();
        let total = audio.len();
        let mut start = 0;
        while start < audio.len() {
            // Cooperative cancellation: checked at each window boundary, i.e.
            // before the expensive encoder pass. Granularity is one window
            // (~30 s), which is the finest this backend can offer without an
            // in-model hook. Report progress as the fraction of samples already
            // consumed (0 % on the first window, 100 % once the loop ends).
            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            control.report_progress((start as u64 * 100 / total as u64) as u8);
            let rest = &audio[start..];
            let cut = super::find_min_rms_cut(rest);
            // `transcribe_samples` takes an owned `Vec<f32>`; we hand it a copy of
            // the window, freed at the end of the iteration. Mono 16 kHz is our
            // input contract. `Sentences` groups tokens into sentence-level spans.
            let out = self
                .model
                .transcribe_samples(
                    rest[..cut].to_vec(),
                    crate::engine::types::SAMPLE_RATE,
                    1,
                    Some(TimestampMode::Sentences),
                )
                .map_err(|e| AsrError::Transcribe(format!("{e}")))?;
            let piece = out.text.trim();
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            // Token times are relative to the window; shift them onto the full
            // input's timeline by the window's start offset.
            let offset = start as f32 / crate::engine::types::SAMPLE_RATE as f32;
            segments.extend(
                out.tokens
                    .into_iter()
                    .filter(|t| !t.text.trim().is_empty())
                    .map(|t| Segment {
                        text: t.text.trim().to_string(),
                        start: t.start + offset,
                        end: t.end + offset,
                    }),
            );
            start += cut;
        }
        control.report_progress(100);

        // The model auto-detects the language but does not expose it in the
        // result -> `detected_language = None`.
        Ok(TranscriptionResult {
            text,
            detected_language: None,
            segments,
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

/// Maps a [`DevicePreference`] to an ONNX Runtime execution provider.
///
/// `DirectML` is only nameable when the crate is built with the
/// `parakeet-directml` feature (the `parakeet-rs` enum variant is itself gated);
/// otherwise we stay on CPU.
fn select_provider(device: DevicePreference) -> ExecutionProvider {
    match device {
        DevicePreference::Cpu => ExecutionProvider::Cpu,
        DevicePreference::Auto | DevicePreference::Gpu => {
            #[cfg(feature = "parakeet-directml")]
            {
                ExecutionProvider::DirectML
            }
            #[cfg(not(feature = "parakeet-directml"))]
            {
                ExecutionProvider::Cpu
            }
        }
    }
}

/// Languages of the `nvidia/parakeet-tdt-0.6b-v3` model (25 European languages:
/// the official EU languages minus Irish, plus Russian and Ukrainian), as
/// lowercase English names to stay consistent with whisper's language detection.
///
/// Source: NVIDIA parakeet-tdt-0.6b-v3 model card (reconfirmed). It is a
/// capability metadatum, not a guarantee extracted programmatically from the
/// loaded model.
fn tdt_v3_languages() -> Vec<String> {
    [
        "bulgarian", "croatian", "czech", "danish", "dutch", "english", "estonian",
        "finnish", "french", "german", "greek", "hungarian", "italian", "latvian",
        "lithuanian", "maltese", "polish", "portuguese", "romanian", "russian",
        "slovak", "slovenian", "spanish", "swedish", "ukrainian",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_declare_no_whisper_only_options() {
        let caps = ParakeetEngine::capabilities_static();
        assert_eq!(caps.name, "parakeet-tdt");
        assert!(!caps.supports_prompt);
        assert!(!caps.supports_beam);
        assert!(!caps.supports_translate);
        assert!(!caps.supports_internal_vad);
        match caps.languages {
            LanguageSupport::Set(langs) => {
                assert_eq!(langs.len(), 25, "parakeet-tdt-v3 = 25 langues");
                assert!(langs.iter().any(|l| l == "french"));
            }
            LanguageSupport::Any => panic!("Parakeet has a closed language set"),
        }
    }

    #[test]
    fn load_reports_missing_model_directory() {
        let cfg = EngineConfig {
            model_path: "::/no/such/parakeet/dir".into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            ParakeetEngine::load(&cfg),
            Err(AsrError::ModelNotFound(_))
        ));
    }
}
