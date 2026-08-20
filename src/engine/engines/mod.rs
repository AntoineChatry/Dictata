//! The [`AsrEngine`] trait and the backend factory.
//!
//! Each real backend lives in its own submodule, enabled by a Cargo feature
//! (`whisper`, `parakeet`). [`MockEngine`] is always available and serves as the
//! test double and reference implementation.

mod mock;
pub use mock::MockEngine;

#[cfg(feature = "whisper")]
mod whisper;
#[cfg(feature = "whisper")]
pub use whisper::WhisperEngine;

#[cfg(feature = "parakeet")]
mod parakeet;
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetEngine;

#[cfg(feature = "sensevoice")]
mod sensevoice;
#[cfg(feature = "sensevoice")]
pub use sensevoice::SenseVoiceEngine;

#[cfg(feature = "moonshine")]
mod moonshine;
#[cfg(feature = "moonshine")]
pub use moonshine::MoonshineEngine;

#[cfg(feature = "zipformer")]
mod zipformer;
#[cfg(feature = "zipformer")]
pub use zipformer::ZipformerEngine;

#[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine",
    feature = "zipformer"
))]
use std::borrow::Cow;

use crate::engine::error::AsrError;
use crate::engine::types::{
    EngineCapabilities, EngineConfig, TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// An ASR backend: loads a model once, transcribes clips on demand.
///
/// **Stateless between calls**: each [`transcribe`](AsrEngine::transcribe) is
/// independent. Any continuity (vocabulary, already-emitted text) goes through
/// [`TranscribeOptions::initial_prompt`], never through state carried on the
/// engine — one instance can be shared between one-shot and streaming.
///
/// `Send` lets the engine move onto a worker thread; it is not `Sync`, so
/// concurrent access goes through the caller's own `Mutex`.
pub trait AsrEngine: Send {
    /// Transcribe `audio` (mono 16 kHz `f32`, see [`crate::engine::SAMPLE_RATE`]).
    ///
    /// Empty `audio` returns an empty result without error.
    ///
    /// `control` carries cooperative cancellation and progress (see
    /// [`TranscribeControl`]); pass `&TranscribeControl::none()` for a plain
    /// call. A requested cancellation returns [`AsrError::Cancelled`], never a
    /// partial `Ok`.
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError>;

    /// This backend's capabilities (honoured options, languages). Stable for the
    /// lifetime of the instance.
    fn capabilities(&self) -> &EngineCapabilities;
}

/// Identifies a backend for the [`load_engine`] factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// whisper.cpp via `whisper-rs` (feature `whisper`).
    Whisper,
    /// NVIDIA Parakeet via `parakeet-rs`/ONNX (feature `parakeet`).
    Parakeet,
    /// SenseVoice CTC via `ort`/ONNX (feature `sensevoice`).
    SenseVoice,
    /// Moonshine encoder-decoder via `ort`/ONNX (feature `moonshine`).
    Moonshine,
    /// Zipformer transducer via `ort`/ONNX (feature `zipformer`).
    Zipformer,
}

impl EngineKind {
    /// Infer the backend from the shape of the model artifact at `path`.
    ///
    /// A single `.bin` file is a whisper.cpp ggml model. A directory is one of
    /// the ONNX bundles, told apart by the graph file each backend actually
    /// loads (see the respective backend's `load`):
    ///   - `decoder_joint-model.onnx`             -> Parakeet (TDT nemo export)
    ///   - `preprocess.onnx` + `cached_decode*`   -> Moonshine (sherpa v1 export)
    ///   - a `*joiner*.onnx`                       -> Zipformer (transducer export)
    ///   - `model.onnx` / `model.int8.onnx`        -> SenseVoice (CTC export)
    ///
    /// These signatures are mutually exclusive across the four ONNX exports, so
    /// the order only breaks ties that cannot occur in a real bundle. Detection
    /// is independent of which backends are compiled: a kind whose feature is off
    /// still resolves here and surfaces later as [`AsrError::BackendUnavailable`]
    /// from [`load_engine`], which is the honest, actionable error.
    ///
    /// Returns [`AsrError::ModelNotFound`] when `path` does not exist and
    /// [`AsrError::Load`] when it exists but matches no known layout — never a
    /// silent guess, since a mislabelled backend loads garbage or aborts.
    pub fn detect(path: &std::path::Path) -> Result<EngineKind, AsrError> {
        if path.is_file() {
            return match path.extension().and_then(|e| e.to_str()) {
                Some("bin") => Ok(EngineKind::Whisper),
                _ => Err(AsrError::Load(format!(
                    "unrecognized model file (expected a .bin whisper model): {}",
                    path.display()
                ))),
            };
        }
        if !path.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "model path not found: {}",
                path.display()
            )));
        }
        // Parakeet: the joint decoder graph is unique to its nemo export.
        if path.join("decoder_joint-model.onnx").is_file() {
            return Ok(EngineKind::Parakeet);
        }
        // Moonshine: the learned conv front-end plus the cached-decode loop graph
        // (the `.int8` variant keeps the `cached_decode` stem).
        if path.join("preprocess.onnx").is_file() && dir_has_onnx_matching(path, "cached_decode") {
            return Ok(EngineKind::Moonshine);
        }
        // Zipformer: only the transducer export ships a joiner graph.
        if dir_has_onnx_matching(path, "joiner") {
            return Ok(EngineKind::Zipformer);
        }
        // SenseVoice: a single CTC graph named model(.int8).onnx.
        if path.join("model.onnx").is_file() || path.join("model.int8.onnx").is_file() {
            return Ok(EngineKind::SenseVoice);
        }
        // Name each backend's signature so the user can see which file is missing
        // from their bundle, instead of a blanket "unknown".
        Err(AsrError::Load(format!(
            "unrecognized model bundle in {}: no backend signature found. Expected one of: \
             whisper (a `.bin` file), \
             parakeet (`decoder_joint-model.onnx` + `encoder-model.onnx` + `vocab.txt`), \
             moonshine (`preprocess.onnx` + `cached_decode*.onnx` + `tokens.txt`), \
             zipformer (`*joiner*.onnx` + `*encoder*.onnx` + `*decoder*.onnx` + `tokens.txt`), \
             sensevoice (`model.onnx` or `model.int8.onnx` + `tokens.txt`)",
            path.display()
        )))
    }
}

/// `true` when `dir` holds a `*.onnx` file whose name contains `stem` — the way
/// sherpa-onnx exports name their graphs (e.g. `joiner-epoch-99-avg-1.int8.onnx`
/// contains `joiner`, `cached_decode.int8.onnx` contains `cached_decode`).
fn dir_has_onnx_matching(dir: &std::path::Path, stem: &str) -> bool {
    std::fs::read_dir(dir).is_ok_and(|rd| {
        rd.flatten().any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".onnx") && n.contains(stem))
        })
    })
}

/// Loads backend `kind` from `config`, returned behind `dyn AsrEngine` for
/// dynamic dispatch on the caller's side.
///
/// When a backend is not built (its feature is off), returns
/// [`AsrError::BackendUnavailable`]. Each compiled backend lives behind its
/// `#[cfg(feature = …)]` arm.
// `config` is consumed only by a compiled backend arm; with no backend feature
// at all it is legitimately unused.
#[cfg_attr(
    not(any(
        feature = "whisper",
        feature = "parakeet",
        feature = "sensevoice",
        feature = "moonshine",
        feature = "zipformer"
    )),
    allow(unused_variables)
)]
pub fn load_engine(
    kind: EngineKind,
    config: &EngineConfig,
) -> Result<Box<dyn AsrEngine>, AsrError> {
    match kind {
        EngineKind::Whisper => {
            #[cfg(feature = "whisper")]
            {
                WhisperEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "whisper"))]
            {
                Err(AsrError::BackendUnavailable(
                    "whisper backend not built (feature `whisper` off)".into(),
                ))
            }
        }
        EngineKind::Parakeet => {
            #[cfg(feature = "parakeet")]
            {
                ParakeetEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "parakeet"))]
            {
                Err(AsrError::BackendUnavailable(
                    "parakeet backend not built (feature `parakeet` off)".into(),
                ))
            }
        }
        EngineKind::SenseVoice => {
            #[cfg(feature = "sensevoice")]
            {
                SenseVoiceEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "sensevoice"))]
            {
                Err(AsrError::BackendUnavailable(
                    "sensevoice backend not built (feature `sensevoice` off)".into(),
                ))
            }
        }
        EngineKind::Moonshine => {
            #[cfg(feature = "moonshine")]
            {
                MoonshineEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "moonshine"))]
            {
                Err(AsrError::BackendUnavailable(
                    "moonshine backend not built (feature `moonshine` off)".into(),
                ))
            }
        }
        EngineKind::Zipformer => {
            #[cfg(feature = "zipformer")]
            {
                ZipformerEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "zipformer"))]
            {
                Err(AsrError::BackendUnavailable(
                    "zipformer backend not built (feature `zipformer` off)".into(),
                ))
            }
        }
    }
}

/// Enforce the audio contract before it reaches a native backend.
///
/// Every backend documents mono 16 kHz `f32` in `[-1.0, 1.0]`, but callers only
/// promise it — nothing checks. A single `NaN`/`±Inf` sample (corrupt file,
/// bad ffmpeg decode) can make whisper.cpp `abort()` the whole process — which
/// no `Result` nor `catch_unwind` can intercept — or poison Parakeet's mel
/// extraction into garbage text. Out-of-range values (`|s| > 1.0`) distort the
/// spectrogram. This replaces non-finite samples with silence and clamps the
/// rest into range. It borrows the input untouched when the audio is already
/// clean, so the overwhelmingly common path allocates nothing.
#[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine",
    feature = "zipformer"
))]
pub(crate) fn sanitize_audio(audio: &[f32]) -> Cow<'_, [f32]> {
    if audio.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)) {
        return Cow::Borrowed(audio);
    }
    let fixed = audio
        .iter()
        .map(|&s| if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 })
        .collect();
    Cow::Owned(fixed)
}

/// Hard cap on a transcription window's length, in samples (30 s). No window
/// ever exceeds this, which is what bounds the ONNX Runtime arena.
#[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
pub(crate) const WINDOW_SAMPLES: usize = 30 * crate::engine::types::SAMPLE_RATE as usize;
/// A clip up to this length (32 s) is transcribed in a single window. Set above
/// `WINDOW_SAMPLES` on purpose: whenever we DO split, the remainder after a cut
/// in the [25 s, 30 s] band is then always >= 2 s. A very short trailing window
/// (a few hundred ms) makes the model hallucinate; this removes that case by
/// construction, at the cost of one window occasionally reaching 32 s.
#[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
pub(crate) const SINGLE_WINDOW_MAX_SAMPLES: usize = 32 * crate::engine::types::SAMPLE_RATE as usize;
/// Start of the cut-point search band (25 s). We only look for an energy dip
/// between 25 and 30 s: past 25 s of speech we want to cut, but at the best spot
/// in that band.
#[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
pub(crate) const SEARCH_START_SAMPLES: usize = 25 * crate::engine::types::SAMPLE_RATE as usize;
/// RMS analysis block size (100 ms).
#[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
const RMS_BLOCK: usize = crate::engine::types::SAMPLE_RATE as usize / 10;

/// Cut position (in samples) of the next window within `rest`.
///
/// Shared by every backend that feeds ONNX Runtime the WHOLE sequence at once
/// (Parakeet, SenseVoice, Moonshine): the arena never returns its peak to the
/// OS, so a long clip must be split into bounded windows and freed between
/// passes. These are also short-form models (trained on <= 30 s), so windowing
/// is a correctness fix too, not only a memory one.
///
/// Returns `rest.len()` when everything fits in a single window (last pass, or a
/// short clip: identical behaviour to a single call). Otherwise it searches the
/// [25 s, 30 s] band for the 100 ms block of lowest energy (RMS) and cuts at its
/// start: we land on the least-bad breath rather than mid-word. If no dip stands
/// out (continuous speech), the hard cap at 30 s applies — the window is always
/// bounded. Always `>= 25 s` once `rest` exceeds `SINGLE_WINDOW_MAX_SAMPLES`, so
/// the calling loop always makes progress.
#[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
pub(crate) fn find_min_rms_cut(rest: &[f32]) -> usize {
    if rest.len() <= SINGLE_WINDOW_MAX_SAMPLES {
        return rest.len();
    }
    // Default hard cap: kept if the band yields no block (impossible here — the
    // band is 5 s = 50 blocks — but keeps the window bounded whatever happens).
    let mut best_start = WINDOW_SAMPLES;
    let mut best_rms = f32::INFINITY;
    let mut pos = SEARCH_START_SAMPLES;
    while pos + RMS_BLOCK <= WINDOW_SAMPLES {
        let block = &rest[pos..pos + RMS_BLOCK];
        let rms = (block.iter().map(|s| s * s).sum::<f32>() / RMS_BLOCK as f32).sqrt();
        if rms < best_rms {
            best_rms = rms;
            best_start = pos;
        }
        pos += RMS_BLOCK;
    }
    best_start
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::TranscribeOptions;

    #[test]
    fn load_engine_never_yields_an_engine_from_a_bogus_config() {
        let cfg = EngineConfig {
            model_path: "::/does/not/exist".into(),
            device: crate::engine::types::DevicePreference::Auto,
            default_language: None,
            crash_marker: None,
        };
        // Depending on the build:
        //   - Parakeet without its feature: always `BackendUnavailable`.
        //   - Whisper without its feature: `BackendUnavailable`.
        //   - Whisper WITH its feature: `ModelNotFound` (the bogus model is absent).
        // In every case an error — never an engine from a nonexistent path. No
        // `{:?}` on the `Ok`: `dyn AsrEngine` is not `Debug`.
        for kind in [EngineKind::Whisper, EngineKind::Parakeet] {
            match load_engine(kind, &cfg) {
                Err(_) => {}
                Ok(_) => panic!("a nonexistent path must never yield an engine ({kind:?})"),
            }
        }
    }

    #[cfg(not(feature = "whisper"))]
    #[test]
    fn whisper_arm_is_unavailable_without_its_feature() {
        let cfg = EngineConfig {
            model_path: "irrelevant".into(),
            device: crate::engine::types::DevicePreference::Auto,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            load_engine(EngineKind::Whisper, &cfg),
            Err(AsrError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn detect_maps_each_model_layout_to_its_backend() {
        use std::fs;
        let base = std::env::temp_dir().join(format!("dictata_detect_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        // whisper: a plain `.bin` file.
        let ggml = base.join("ggml-base.bin");
        fs::write(&ggml, b"x").unwrap();
        assert_eq!(EngineKind::detect(&ggml).unwrap(), EngineKind::Whisper);

        // A directory bundle is told apart by its signature graph file.
        let mk = |name: &str, files: &[&str]| {
            let d = base.join(name);
            fs::create_dir_all(&d).unwrap();
            for f in files {
                fs::write(d.join(f), b"x").unwrap();
            }
            d
        };
        assert_eq!(
            EngineKind::detect(&mk(
                "parakeet",
                &["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"],
            ))
            .unwrap(),
            EngineKind::Parakeet
        );
        assert_eq!(
            EngineKind::detect(&mk(
                "moonshine",
                &["preprocess.onnx", "encode.onnx", "cached_decode.int8.onnx", "tokens.txt"],
            ))
            .unwrap(),
            EngineKind::Moonshine
        );
        assert_eq!(
            EngineKind::detect(&mk(
                "zipformer",
                &["encoder-epoch-99.onnx", "decoder-epoch-99.onnx", "joiner-epoch-99.int8.onnx", "tokens.txt"],
            ))
            .unwrap(),
            EngineKind::Zipformer
        );
        assert_eq!(
            EngineKind::detect(&mk("sensevoice", &["model.int8.onnx", "tokens.txt"])).unwrap(),
            EngineKind::SenseVoice
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_rejects_missing_and_unknown_layouts() {
        use std::fs;
        let base = std::env::temp_dir().join(format!("dictata_detect_neg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        // Absent path -> ModelNotFound.
        assert!(matches!(
            EngineKind::detect(&base.join("nope.bin")),
            Err(AsrError::ModelNotFound(_))
        ));
        // A file that is not a `.bin` -> Load (unrecognized), never a guess.
        let txt = base.join("notes.txt");
        fs::write(&txt, b"x").unwrap();
        assert!(matches!(EngineKind::detect(&txt), Err(AsrError::Load(_))));
        // A directory with no known signature -> Load (unrecognized layout).
        let empty = base.join("mystery");
        fs::create_dir_all(&empty).unwrap();
        fs::write(empty.join("readme.md"), b"x").unwrap();
        assert!(matches!(EngineKind::detect(&empty), Err(AsrError::Load(_))));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn engine_is_usable_as_a_trait_object() {
        // The whole point of the exercise: drive any backend behind
        // `dyn AsrEngine`.
        let mut engine: Box<dyn AsrEngine> = Box::new(MockEngine::new("salut"));
        let out = engine
            .transcribe(
                &[0.1, -0.1],
                &TranscribeOptions::default(),
                &TranscribeControl::none(),
            )
            .unwrap();
        assert_eq!(out.text, "salut");
        assert!(engine.capabilities().supports_prompt || !engine.capabilities().supports_prompt);
    }

    #[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
    #[test]
    fn sanitize_borrows_clean_audio_without_allocating() {
        let clean = [0.0, 0.5, -0.5, 1.0, -1.0];
        // Clean audio is passed through as a borrow (no copy).
        assert!(matches!(sanitize_audio(&clean), Cow::Borrowed(_)));
    }

    #[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
    #[test]
    fn sanitize_replaces_non_finite_and_clamps_out_of_range() {
        let dirty = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0, -3.0, 0.3];
        let out = sanitize_audio(&dirty);
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(&*out, &[0.0, 0.0, 0.0, 1.0, -1.0, 0.3]);
        // Every sample is finite and in range after sanitizing.
        assert!(out.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    const RATE: usize = crate::engine::types::SAMPLE_RATE as usize;

    /// `secs` seconds of a signal at amplitude `amp` (0.0 = silence).
    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        let n = (RATE as f32 * secs) as usize;
        (0..n).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    #[test]
    fn cut_takes_the_whole_clip_when_it_fits_in_one_window() {
        // <= 32 s: a single call, cut = full length (identical to the old path).
        assert_eq!(find_min_rms_cut(&tone(10.0, 0.3)), 10 * RATE);
        assert_eq!(find_min_rms_cut(&tone(30.0, 0.3)), 30 * RATE);
        assert_eq!(find_min_rms_cut(&tone(32.0, 0.3)), 32 * RATE);
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    #[test]
    fn cut_falls_on_the_quietest_block_in_the_search_band() {
        // 33 s of loud speech with a silence gap at 26.0 s: the cut must land in
        // that gap (the lowest-energy 100 ms block of the [25 s, 30 s] band).
        let mut audio = tone(26.0, 0.3);
        audio.extend(tone(0.2, 0.0)); // silence at 26.0 s
        audio.extend(tone(6.8, 0.3)); // total 33 s
        let cut = find_min_rms_cut(&audio);
        let cut_s = cut as f32 / RATE as f32;
        assert!(
            (26.0..=26.3).contains(&cut_s),
            "cut expected near 26 s (the gap), got {cut_s} s"
        );
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    #[test]
    fn cut_stays_within_the_search_band_on_continuous_speech() {
        // Continuous speech with no dip: the cut stays bounded to [25 s, 30 s],
        // never beyond (ONNX arena guaranteed flat).
        let cut = find_min_rms_cut(&tone(40.0, 0.3));
        assert!(
            (SEARCH_START_SAMPLES..=WINDOW_SAMPLES).contains(&cut),
            "cut {cut} outside band [{SEARCH_START_SAMPLES}, {WINDOW_SAMPLES}]"
        );
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    #[test]
    fn a_split_never_leaves_a_tiny_trailing_window() {
        // Whenever the clip is split, the remainder must be >= 2 s, so the last
        // window never degrades into a hallucination-prone sub-second clip. This
        // is what the 32 s single-window threshold buys.
        for secs in [30.1f32, 32.1, 33.0, 45.0, 60.0] {
            let audio = tone(secs, 0.2);
            let cut = find_min_rms_cut(&audio);
            if cut < audio.len() {
                let tail = audio.len() - cut;
                assert!(tail >= 2 * RATE, "tail {tail} samples < 2 s at {secs} s");
            }
        }
    }

    #[cfg(any(feature = "parakeet", feature = "sensevoice", feature = "moonshine"))]
    #[test]
    fn cut_always_makes_progress() {
        // Every cut is > 0, so the windowing loop always advances.
        for secs in [1.0f32, 15.0, 29.9, 30.0, 32.1, 61.0, 120.0] {
            assert!(find_min_rms_cut(&tone(secs, 0.2)) > 0, "null cut at {secs} s");
        }
    }
}
