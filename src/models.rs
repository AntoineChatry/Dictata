//! Catalog of ggml models (whisper.cpp) + download from HuggingFace.
//!
//! Models are `ggml-{name}.bin` files from the `ggerganov/whisper.cpp` repo.
//! The `name` maps directly to `config.model` (e.g. "base", "small-q5_1").

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub struct CatalogEntry {
    /// Identifier = `config.model` (e.g. "tiny-q5_1"). File = `ggml-{name}.bin`.
    pub name: &'static str,
    pub label: &'static str,
    pub size_mb: u32,
    pub quantized: bool,
}

/// Featured models (quantized ones preferred, as per the decisions).
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry { name: "tiny-q5_1", label: "Tiny", size_mb: 31, quantized: true },
    CatalogEntry { name: "tiny", label: "Tiny", size_mb: 75, quantized: false },
    CatalogEntry { name: "base-q5_1", label: "Base", size_mb: 57, quantized: true },
    CatalogEntry { name: "base", label: "Base", size_mb: 141, quantized: false },
    CatalogEntry { name: "small-q5_1", label: "Small", size_mb: 181, quantized: true },
    CatalogEntry { name: "small", label: "Small", size_mb: 465, quantized: false },
    CatalogEntry { name: "medium-q5_0", label: "Medium", size_mb: 514, quantized: true },
    CatalogEntry { name: "medium", label: "Medium", size_mb: 1463, quantized: false },
    CatalogEntry { name: "large-v3-turbo-q5_0", label: "Large v3 Turbo", size_mb: 547, quantized: true },
    CatalogEntry { name: "large-v3-turbo", label: "Large v3 Turbo", size_mb: 1549, quantized: false },
    CatalogEntry { name: "large-v3-q5_0", label: "Large v3", size_mb: 1031, quantized: true },
    CatalogEntry { name: "large-v3", label: "Large v3", size_mb: 2952, quantized: false },
];

/// A verified ONNX bundle from HuggingFace, offered one-click in the models
/// page (the multi-file counterpart of [`CATALOG`]). Every repo here has been
/// checked to have the exact layout its backend's `load` expects, so a download
/// followed by [`crate::engine::EngineKind::detect`] always classifies.
pub struct OnnxCatalogEntry {
    /// HuggingFace repo id ("owner/name"). The bundle installs under its leaf
    /// (e.g. `parakeet-tdt-0.6b-v3-onnx`), which becomes `config.model`.
    pub repo: &'static str,
    /// i18n key for the human-readable label (translated in the UI, so the
    /// language qualifier follows the app language). See [`crate::i18n`].
    pub label_key: &'static str,
    /// Backend name, matching [`crate::engine::EngineKind`]'s `Debug`.
    pub backend: &'static str,
    /// Approximate whole-repo download size (fp32 + int8 variants), MB.
    pub size_mb: u32,
}

/// Curated ONNX bundles, one per multi-file backend. Repo ids verified against
/// the HuggingFace API (correct signature files at the repo root).
pub const ONNX_CATALOG: &[OnnxCatalogEntry] = &[
    OnnxCatalogEntry {
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        label_key: "onnx_parakeet",
        backend: "Parakeet",
        size_mb: 3070,
    },
    OnnxCatalogEntry {
        repo: "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        label_key: "onnx_sensevoice",
        backend: "SenseVoice",
        size_mb: 1120,
    },
    OnnxCatalogEntry {
        repo: "k2-fsa/sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        label_key: "onnx_zipformer",
        backend: "Zipformer",
        size_mb: 610,
    },
    OnnxCatalogEntry {
        repo: "csukuangfj/sherpa-onnx-moonshine-base-en-int8",
        label_key: "onnx_moonshine",
        backend: "Moonshine",
        size_mb: 220,
    },
];

pub fn file_name(model: &str) -> String {
    // Custom models (HuggingFace search) are stored under their full file
    // name; catalog names keep the `ggml-{name}.bin` scheme.
    // Only the last path component is kept: a hand-edited `model` must never
    // escape `model_dir`, since `model_path` feeds `delete` and `Path::join`
    // silently discards the base when joined with an absolute path.
    let leaf = Path::new(model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if leaf.ends_with(".bin") {
        leaf.to_string()
    } else {
        format!("ggml-{leaf}.bin")
    }
}

/// Sanitized last path component of `model`, used to name an ONNX **bundle
/// directory** inside `model_dir`. Unlike [`file_name`] it appends nothing: a
/// bundle is a directory (e.g. `parakeet-tdt-0.6b-v3`), not a `ggml-*.bin`.
/// Only the leaf survives, so a hand-edited `model` can never escape
/// `model_dir` (same guarantee as [`file_name`], which `model_path`/`delete`
/// rely on).
pub fn dir_name(model: &str) -> String {
    Path::new(model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string()
}

/// Resolve `model` to its on-disk location inside `model_dir`.
///
/// A model is either an ONNX **bundle directory** (multi-file backends:
/// Parakeet, SenseVoice, …) or a single ggml **`.bin` file** (whisper). The
/// directory form wins when it exists, so `model` names the bundle folder; every
/// other case falls back to the `ggml-{model}.bin` scheme, keeping the whisper
/// path byte-for-byte identical to before.
pub fn model_path(model_dir: &str, model: &str) -> PathBuf {
    let bundle = Path::new(model_dir).join(dir_name(model));
    if bundle.is_dir() {
        return bundle;
    }
    Path::new(model_dir).join(file_name(model))
}

pub fn is_installed(model_dir: &str, model: &str) -> bool {
    model_path(model_dir, model).exists()
}

/// ONNX bundle **directories** present in `model_dir` (subfolders whose layout
/// [`crate::engine::EngineKind::detect`] recognizes as a real backend bundle).
///
/// Mirror of [`installed_files`] for the multi-file backends: a bundle is a
/// directory, not a `.bin`, so the whisper-oriented `.bin` scan never sees it.
/// A subfolder that is not a valid bundle (stray directory, partial download) is
/// skipped, so it never shows up as an installed model. Returns an empty list
/// when `model_dir` cannot be read.
pub fn installed_bundles(model_dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(model_dir) {
        for e in rd.flatten() {
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && crate::engine::EngineKind::detect(&e.path()).is_ok()
                && let Some(n) = e.file_name().to_str()
            {
                out.push(n.to_string());
            }
        }
    }
    out
}

/// Raw `.bin` file names present in `model_dir`.
///
/// Lets a caller read the directory once and answer many "is this installed?"
/// questions from the result, instead of one `exists()` syscall per model per
/// UI frame. Returns an empty list if the directory cannot be read, which
/// [`is_installed_among`] then reports as "not installed" — same answer as
/// [`is_installed`] in that situation.
pub fn installed_files(model_dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(model_dir) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str()
                && n.ends_with(".bin")
            {
                out.push(n.to_string());
            }
        }
    }
    out
}

/// [`is_installed`] answered from a list obtained via [`installed_files`].
pub fn is_installed_among(files: &[String], model: &str) -> bool {
    files.contains(&file_name(model))
}

/// Delete an installed model: a `.bin` file, or an ONNX bundle **directory**
/// (removed whole).
pub fn delete(model_dir: &str, model: &str) -> Result<(), String> {
    let p = model_path(model_dir, model);
    let res = if p.is_dir() {
        std::fs::remove_dir_all(&p)
    } else {
        std::fs::remove_file(&p)
    };
    res.map_err(|e| e.to_string())
}

/// List the models installed in `model_dir`: ggml names (without `ggml-`/`.bin`)
/// **and** ONNX bundle directory names.
pub fn list_installed(model_dir: &str) -> Vec<String> {
    let mut out = list_installed_from(&installed_files(model_dir));
    out.extend(installed_bundles(model_dir));
    out.sort();
    out
}

/// [`list_installed`] from a `.bin` listing already obtained via
/// [`installed_files`]. Covers whisper models only; ONNX bundles are listed
/// separately via [`installed_bundles`] (a `.bin` scan cannot see a directory).
pub fn list_installed_from(files: &[String]) -> Vec<String> {
    let mut out = model_names(files);
    out.sort();
    out
}

/// File names -> model identifiers, as `config.model` stores them.
fn model_names(files: &[String]) -> Vec<String> {
    files
        .iter()
        .filter(|n| {
            // VAD models (`ggml-silero-*.bin`) are not transcription models:
            // selecting one as the main model crashes whisper.cpp. Hide them.
            !n.starts_with("ggml-silero-")
        })
        .map(|n| {
            match n.strip_prefix("ggml-").and_then(|s| s.strip_suffix(".bin")) {
                Some(s) => s.to_string(),
                // Custom model: identified by its full file name.
                None => n.to_string(),
            }
        })
        .collect()
}

/// Download `model` into `model_dir`. `progress(received, total_opt)` is called
/// during the transfer. Writes a `.part` first then renames at the end.
pub fn download(
    model_dir: &str,
    model: &str,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let fname = file_name(model);
    let url = format!("{HF_BASE}/{fname}");
    download_url(model_dir, &url, &fname, progress)
}

/// Download an ONNX **bundle**: every file in `files` into `model_dir/{subdir}/`.
///
/// Used for the multi-file backends (Parakeet, sherpa-onnx exports) that ship a
/// directory of `.onnx` + `tokens.txt`/`vocab.txt` rather than a single ggml
/// `.bin`. `progress(received, total)` reports **aggregate** bytes across all
/// files (total is known only when every file advertises a size). On any
/// failure the partial bundle directory is removed, so a half-download never
/// registers as an installed model. `subdir` is reduced to its leaf via
/// [`dir_name`], so it stays inside `model_dir`.
pub fn download_bundle(
    model_dir: &str,
    subdir: &str,
    files: &[HfFile],
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let leaf = dir_name(subdir);
    if leaf.is_empty() {
        return Err(format!("invalid bundle name: {subdir:?}"));
    }
    if files.is_empty() {
        return Err("empty bundle (no files to download)".into());
    }
    let bundle = Path::new(model_dir).join(&leaf);
    let bundle_str = bundle
        .to_str()
        .ok_or_else(|| "bundle path is not valid UTF-8".to_string())?
        .to_string();

    // Aggregate total: `Some` only when every file advertises its size (`Sum`
    // for `Option` folds to `None` as soon as one size is missing).
    let total: Option<u64> = files.iter().map(|f| f.size).sum();
    let mut base: u64 = 0;

    for f in files {
        let res = download_url(&bundle_str, &f.url(), &f.fname, |recu, _tot| {
            progress(base + recu, total);
        });
        match res {
            Ok(_) => base += f.size.unwrap_or(0),
            Err(e) => {
                // Never leave a partial bundle behind: detect() would otherwise
                // list it as a broken installed model.
                let _ = std::fs::remove_dir_all(&bundle);
                return Err(format!("{}: {e}", f.fname));
            }
        }
    }
    // Validate the assembled directory against a known backend layout. A repo
    // may download cleanly yet be the wrong export (a NeMo `.nemo` card with
    // only JSON sidecars, a FunASR SenseVoice instead of the sherpa-onnx one):
    // without this the folder would silently never appear as an installed
    // model. Reject it here with the per-backend "expected …" message instead.
    if let Err(e) = crate::engine::EngineKind::detect(&bundle) {
        let _ = std::fs::remove_dir_all(&bundle);
        return Err(e.to_string());
    }
    Ok(bundle)
}

// ---------- VAD model (whisper.cpp Silero) ----------

/// File name of the VAD model (whisper.cpp Silero v5), stored next to the
/// whisper models. Used by the one-shot path to skip silence before decoding.
pub const VAD_MODEL_FILE: &str = "ggml-silero-v5.1.2.bin";
const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";

pub fn vad_model_path(model_dir: &str) -> PathBuf {
    Path::new(model_dir).join(VAD_MODEL_FILE)
}

/// Download the VAD model (~2 MB) into `model_dir` (`.part`-then-rename scheme).
pub fn download_vad(
    model_dir: &str,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    download_url(model_dir, VAD_MODEL_URL, VAD_MODEL_FILE, progress)
}

// ---------- External VAD model (shared Silero ONNX) ----------

/// File name of the shared external Silero VAD (v5 ONNX), stored next to the
/// models. The non-whisper backends (Parakeet, SenseVoice, Moonshine,
/// Zipformer) have no internal VAD, so they gate silence through this shared
/// model; whisper.cpp keeps its own internal VAD ([`VAD_MODEL_FILE`]).
pub const SILERO_ONNX_FILE: &str = "silero_vad.onnx";
// The canonical Silero v5 export (producer `spox`), the one the engine's VAD
// expects (576-sample input = 64 lookback + 512 hop). From the upstream repo.
const SILERO_ONNX_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";

pub fn silero_onnx_path(model_dir: &str) -> PathBuf {
    Path::new(model_dir).join(SILERO_ONNX_FILE)
}

/// Download the external Silero ONNX VAD (~2 MB) into `model_dir`
/// (`.part`-then-rename scheme).
pub fn download_silero_onnx(
    model_dir: &str,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    download_url(model_dir, SILERO_ONNX_URL, SILERO_ONNX_FILE, progress)
}

/// Resolve the VAD model to hand to a transcription, fetching it on first use.
/// `None` when `vad_enabled` is off or the model could not be obtained.
///
/// The right VAD file depends on the backend, picked from `model_path`'s shape
/// via [`crate::engine::EngineKind`]: whisper.cpp uses its own internal ggml VAD
/// ([`VAD_MODEL_FILE`]); the ONNX backends have none and share the external
/// Silero ONNX VAD ([`SILERO_ONNX_FILE`]), which is only useful when the `vad`
/// feature is built — without it the facade ignores the model, so it is not
/// fetched. Shared by the dictation worker and the "transcribe a file" path.
pub fn resolve_vad_path(model_dir: &str, model_path: &Path, vad_enabled: bool) -> Option<PathBuf> {
    if !vad_enabled {
        return None;
    }
    let is_whisper = matches!(
        crate::engine::EngineKind::detect(model_path),
        Ok(crate::engine::EngineKind::Whisper)
    );
    if is_whisper {
        let p = vad_model_path(model_dir);
        if !p.exists() {
            let _ = download_vad(model_dir, |_, _| {});
        }
        p.exists().then_some(p)
    } else {
        #[cfg(feature = "vad")]
        {
            let p = silero_onnx_path(model_dir);
            if !p.exists() {
                let _ = download_silero_onnx(model_dir, |_, _| {});
            }
            p.exists().then_some(p)
        }
        #[cfg(not(feature = "vad"))]
        {
            None
        }
    }
}

/// Last path component of `fname`, rejecting anything that cannot name a file
/// inside a directory (traversal, empty, separator only).
///
/// `download_url` is the single point where a download touches the disk, so the
/// check lives there rather than only in the callers: `download` already goes
/// through [`file_name`], but the HuggingFace `FileUrl` path derives its name
/// straight from the URL and would otherwise reach `Path::join` unfiltered.
fn safe_file_name(fname: &str) -> Result<&str, String> {
    Path::new(fname)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| format!("nom de fichier invalide: {fname:?}"))
}

/// Inactivity tolerated on the transfer before it is declared dead.
///
/// This is the blocking client's own default, pinned explicitly because the
/// download depends on it: in `reqwest::blocking` this timeout is applied to
/// *each* read of the body with a fresh deadline (see `blocking/response.rs`,
/// `Read for Response`), so it acts as an inactivity timeout and a multi-GB
/// model on a slow link still completes as long as bytes keep arriving. Do not
/// mistake it for the async client's `timeout`, which is a total deadline and
/// would cut large downloads short.
const DOWNLOAD_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Not covered by the default above, which only starts once a connection exists.
const DOWNLOAD_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Download an arbitrary URL into `model_dir` under `fname` (same `.part`
/// then rename scheme as `download`).
pub fn download_url(
    model_dir: &str,
    url: &str,
    fname: &str,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    download_url_with_timeouts(
        model_dir,
        url,
        fname,
        progress,
        DOWNLOAD_READ_TIMEOUT,
        DOWNLOAD_CONNECT_TIMEOUT,
    )
}

/// [`download_url`] with explicit timeouts, so the stalled-transfer test does
/// not have to wait the production 30 s.
fn download_url_with_timeouts(
    model_dir: &str,
    url: &str,
    fname: &str,
    mut progress: impl FnMut(u64, Option<u64>),
    read_timeout: std::time::Duration,
    connect_timeout: std::time::Duration,
) -> Result<PathBuf, String> {
    let fname = safe_file_name(fname)?;
    std::fs::create_dir_all(model_dir).map_err(|e| format!("models directory: {e}"))?;
    let dest = Path::new(model_dir).join(fname);
    let tmp = dest.with_extension("part");

    let mut resp = reqwest::blocking::Client::builder()
        .timeout(read_timeout)
        .connect_timeout(connect_timeout)
        .build()
        .map_err(|e| format!("client http: {e}"))?
        .get(url)
        .send()
        .map_err(|e| format!("request: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} for {url}", resp.status()));
    }
    let total = resp.content_length();

    let mut file = std::fs::File::create(&tmp).map_err(|e| format!("file creation: {e}"))?;
    let transfer = (|| -> Result<(), String> {
        let mut buf = [0u8; 65536];
        let mut received = 0u64;
        loop {
            let n = resp.read(&mut buf).map_err(|e| format!("stream read: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| format!("write: {e}"))?;
            received += n as u64;
            progress(received, total);
        }
        // Belt and braces: the reader already fails on a body shorter than the
        // announced length, but a stream ending early on a valid frame would
        // otherwise be committed as a complete model.
        match total {
            Some(total) if received != total => Err(format!(
                "incomplete download ({received}/{total} bytes), model not installed"
            )),
            _ => Ok(()),
        }
    })();
    drop(file);
    // Never leave a `.part` behind: a failed download must not litter
    // `model_dir` nor leave a stale partial file for the next attempt.
    if let Err(e) = transfer {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &dest).map_err(|e| format!("rename: {e}"))?;
    Ok(dest)
}

// ---------- HuggingFace browsing ----------

/// `.bin` file of a HuggingFace repo (candidate ggml model).
#[derive(Clone)]
pub struct HfFile {
    pub repo: String,
    pub fname: String,
    pub size: Option<u64>,
}

impl HfFile {
    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, self.fname)
    }
}

fn hf_get(url: &str) -> Result<serde_json::Value, String> {
    let resp = reqwest::blocking::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .map_err(|e| format!("request: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json().map_err(|e| format!("json: {e}"))
}

/// Search repos on HuggingFace; returns repo ids ("owner/name").
pub fn hf_search(query: &str) -> Result<Vec<String>, String> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&limit=20",
        urlencode(query)
    );
    let v = hf_get(&url)?;
    Ok(v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default())
}

/// List the `.bin` files of a HuggingFace repo.
pub fn hf_list_files(repo: &str) -> Result<Vec<HfFile>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let v = hf_get(&url)?;
    let mut out = Vec::new();
    if let Some(sib) = v.get("siblings").and_then(|s| s.as_array()) {
        for f in sib {
            if let Some(name) = f.get("rfilename").and_then(|n| n.as_str()) {
                if name.ends_with(".bin") && !name.contains('/') {
                    out.push(HfFile {
                        repo: repo.to_string(),
                        fname: name.to_string(),
                        size: f.get("size").and_then(|s| s.as_u64()),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// `true` when `name` is a file an ONNX bundle is made of: a graph (`.onnx`),
/// its external weights (`*.onnx.data` / `*.onnx_data` / `*.data`, produced when
/// a model exceeds the 2 GB protobuf limit — ONNX Runtime loads it by name from
/// the model's directory, so it MUST be fetched too), or a vocabulary/config
/// sidecar (`tokens.txt`, `vocab.txt`, `*.json`, SentencePiece `*.model`).
/// Drives which repo files are pulled into a bundle download.
pub fn is_bundle_file(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.ends_with(".onnx")
        || n.ends_with(".onnx.data")
        || n.ends_with(".onnx_data")
        || n.ends_with(".data")
        || n.ends_with(".txt")
        || n.ends_with(".json")
        || n.ends_with(".model")
}

/// List a HuggingFace repo's **root-level** files, split into ggml `.bin`
/// whisper models and ONNX **bundle** artifacts ([`is_bundle_file`]).
///
/// One API call feeds both the existing `.bin` browser and the bundle
/// downloader. Nested files (name containing `/`) are skipped: the multi-file
/// backends expect a flat directory, and flattening a repo's subfolders here
/// would silently produce a broken bundle.
pub fn hf_list_repo(repo: &str) -> Result<(Vec<HfFile>, Vec<HfFile>), String> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let v = hf_get(&url)?;
    let mut bins = Vec::new();
    let mut bundle = Vec::new();
    if let Some(sib) = v.get("siblings").and_then(|s| s.as_array()) {
        for f in sib {
            if let Some(name) = f.get("rfilename").and_then(|n| n.as_str()) {
                if name.contains('/') {
                    continue;
                }
                // Big graphs are Git-LFS: the real byte size sits under
                // `lfs.size`, while the plain `size` is the tiny pointer. Prefer
                // `lfs.size` so the aggregate download total (and the progress
                // bar) is right; fall back to `size` for small non-LFS files.
                let size = f
                    .get("lfs")
                    .and_then(|l| l.get("size"))
                    .and_then(|s| s.as_u64())
                    .or_else(|| f.get("size").and_then(|s| s.as_u64()));
                let hf = HfFile {
                    repo: repo.to_string(),
                    fname: name.to_string(),
                    size,
                };
                if name.ends_with(".bin") {
                    bins.push(hf);
                } else if is_bundle_file(name) {
                    bundle.push(hf);
                }
            }
        }
    }
    Ok((bins, bundle))
}

/// Interprets the search-field input: direct file URL, repo URL, repo id
/// ("owner/name") or free-text query.
pub enum HfQuery {
    /// Direct downloadable file (url, file name).
    FileUrl(String, String),
    /// Repo whose files should be listed.
    Repo(String),
    /// Free-text search.
    Search(String),
}

pub fn parse_hf_query(input: &str) -> HfQuery {
    let s = input.trim().trim_end_matches('/');
    // https only: keeping the `http://` prefix would download the model over a
    // downgradable connection. HuggingFace serves https and redirects anyway.
    if let Some(rest) = s.strip_prefix("https://huggingface.co/") {
        // File URL: https://huggingface.co/owner/name/resolve/main/file.bin
        if rest.contains("/resolve/") || rest.contains("/blob/") {
            let fname = rest.rsplit('/').next().unwrap_or("model.bin").to_string();
            let url = s.replace("/blob/", "/resolve/");
            return HfQuery::FileUrl(url, fname);
        }
        // Repo URL: https://huggingface.co/owner/name[/tree/...]
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() >= 2 {
            return HfQuery::Repo(format!("{}/{}", parts[0], parts[1]));
        }
        return HfQuery::Search(rest.to_string());
    }
    // Bare repo id: exactly one '/', no spaces.
    if s.matches('/').count() == 1 && !s.contains(' ') && !s.starts_with("http") {
        return HfQuery::Repo(s.to_string());
    }
    HfQuery::Search(s.to_string())
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_scheme() {
        assert_eq!(file_name("base"), "ggml-base.bin");
        assert_eq!(file_name("small-q5_1"), "ggml-small-q5_1.bin");
        // Custom models keep their full file name.
        assert_eq!(file_name("whisper-large-zh.bin"), "whisper-large-zh.bin");
    }

    #[test]
    fn download_url_rejects_truncated() {
        // Server announcing 100 bytes but sending 10, then closing cleanly.
        let srv = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = srv.accept() {
                let mut junk = [0u8; 1024];
                let _ = s.read(&mut junk);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n0123456789");
                let _ = s.flush();
            }
        });
        let dir = std::env::temp_dir().join("dictata_trunc_test");
        let _ = std::fs::remove_dir_all(&dir);
        let res = download_url(
            dir.to_str().unwrap(),
            &format!("http://127.0.0.1:{port}/x.bin"),
            "x.bin",
            |_, _| {},
        );
        let err = res.expect_err("a truncated download must not succeed");
        assert!(!dir.join("x.bin").exists(), "truncated file was installed: {err}");
        assert!(!dir.join("x.part").exists(), "leftover .part: {err}");
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("truncated download rejected by: {err}");
    }

    #[test]
    fn file_name_catalog_unchanged() {
        // The whole catalog must keep the plain `ggml-{name}.bin` scheme.
        for e in CATALOG {
            assert_eq!(file_name(e.name), format!("ggml-{}.bin", e.name));
        }
        assert_eq!(file_name(VAD_MODEL_FILE), VAD_MODEL_FILE);
    }

    #[test]
    fn file_name_stays_in_model_dir() {
        // Only the last component survives: `model_path` feeds `delete`.
        assert_eq!(file_name("../../evil.bin"), "evil.bin");
        assert_eq!(file_name("..\\..\\evil.bin"), "evil.bin");
        assert_eq!(file_name("C:\\Windows\\System32\\evil.bin"), "evil.bin");
        assert_eq!(file_name("/etc/passwd.bin"), "passwd.bin");
        // A name that is only traversal resolves to no usable file.
        assert_eq!(file_name(".."), "ggml-.bin");
        for m in ["../../evil.bin", "C:\\Windows\\evil.bin", ".."] {
            let p = model_path("F:\\models", m);
            assert_eq!(p.parent(), Some(Path::new("F:\\models")), "escaped: {m}");
        }
    }

    #[test]
    fn installed_lookup_matches_the_per_file_check() {
        // The cached lookup must answer exactly like `is_installed`, including
        // for custom models stored under their full file name.
        let files = vec![
            "ggml-base.bin".to_string(),
            "whisper-large-zh.bin".to_string(),
            VAD_MODEL_FILE.to_string(),
        ];
        assert!(is_installed_among(&files, "base"));
        assert!(is_installed_among(&files, "whisper-large-zh.bin"));
        assert!(is_installed_among(&files, VAD_MODEL_FILE));
        assert!(!is_installed_among(&files, "small"));
        assert!(!is_installed_among(&files, "ggml-base.bin.bak"));
        // A traversing name resolves to its leaf, like everywhere else.
        assert!(is_installed_among(&files, "../../ggml-base.bin"));

        // And the display list hides the VAD model while keeping the rest.
        let mut names = model_names(&files);
        names.sort();
        assert_eq!(names, vec!["base".to_string(), "whisper-large-zh.bin".to_string()]);
    }

    #[test]
    fn download_url_gives_up_on_stalled_stream() {
        // Server that answers the headers then goes silent without closing: the
        // read blocks forever without a read timeout, wedging the models UI
        // until the app is restarted.
        let srv = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        let keep_alive = std::thread::spawn(move || {
            if let Ok((mut s, _)) = srv.accept() {
                let mut junk = [0u8; 1024];
                let _ = s.read(&mut junk);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n0123456789");
                let _ = s.flush();
                // Hold the connection open, sending nothing more, well past the
                // 300 ms read timeout the download is given below.
                std::thread::sleep(std::time::Duration::from_millis(900));
            }
        });
        let dir = std::env::temp_dir().join("dictata_stall_test");
        let _ = std::fs::remove_dir_all(&dir);
        let started = std::time::Instant::now();
        let res = download_url_with_timeouts(
            dir.to_str().unwrap(),
            &format!("http://127.0.0.1:{port}/x.bin"),
            "x.bin",
            |_, _| {},
            std::time::Duration::from_millis(300),
            std::time::Duration::from_secs(5),
        );
        let err = res.expect_err("un transfert gele doit echouer");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "le timeout de lecture n'a pas coupe le transfert: {err}"
        );
        assert!(!dir.join("x.bin").exists(), "fichier incomplet installe: {err}");
        assert!(!dir.join("x.part").exists(), "reste un .part: {err}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = keep_alive.join();
    }

    #[test]
    fn download_rejects_unsafe_file_names() {
        // Traversal is stripped down to the leaf, which stays inside model_dir.
        assert_eq!(safe_file_name("../../evil.bin").unwrap(), "evil.bin");
        assert_eq!(safe_file_name("..\\..\\evil.bin").unwrap(), "evil.bin");
        assert_eq!(safe_file_name("C:\\Windows\\System32\\evil.bin").unwrap(), "evil.bin");
        assert_eq!(safe_file_name("/etc/passwd.bin").unwrap(), "passwd.bin");
        // Legitimate names are untouched (the other callers rely on this).
        assert_eq!(safe_file_name("ggml-base.bin").unwrap(), "ggml-base.bin");
        assert_eq!(safe_file_name(VAD_MODEL_FILE).unwrap(), VAD_MODEL_FILE);
        // A trailing separator is normalised away, and the leaf still stays put.
        assert_eq!(safe_file_name("some/dir/").unwrap(), "dir");
        // Names that cannot designate a file are refused outright.
        for bad in ["", "..", ".", "/", "\\", "../"] {
            assert!(safe_file_name(bad).is_err(), "accepte a tort: {bad:?}");
        }
    }

    #[test]
    fn download_url_refuses_unsafe_name_before_any_request() {
        // Port 1 is closed: reaching the network would surface a request error,
        // so a "nom de fichier invalide" proves the check runs first.
        let dir = std::env::temp_dir().join("dictata_unsafe_name_test");
        let _ = std::fs::remove_dir_all(&dir);
        let err = download_url(dir.to_str().unwrap(), "http://127.0.0.1:1/x", "..", |_, _| {})
            .expect_err("un nom invalide doit etre refuse");
        assert!(err.contains("nom de fichier invalide"), "erreur inattendue: {err}");
        assert!(!dir.exists(), "le dossier ne doit pas etre cree avant la validation");
    }

    #[test]
    fn hf_query_parsing() {
        match parse_hf_query("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin") {
            HfQuery::FileUrl(url, fname) => {
                assert_eq!(fname, "ggml-base.bin");
                assert!(url.contains("/resolve/main/"));
            }
            _ => panic!("expected FileUrl"),
        }
        match parse_hf_query("https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin") {
            HfQuery::FileUrl(url, _) => assert!(url.contains("/resolve/")),
            _ => panic!("expected FileUrl"),
        }
        match parse_hf_query("https://huggingface.co/ggerganov/whisper.cpp") {
            HfQuery::Repo(r) => assert_eq!(r, "ggerganov/whisper.cpp"),
            _ => panic!("expected Repo"),
        }
        match parse_hf_query("ggerganov/whisper.cpp") {
            HfQuery::Repo(r) => assert_eq!(r, "ggerganov/whisper.cpp"),
            _ => panic!("expected Repo"),
        }
        match parse_hf_query("whisper ggml") {
            HfQuery::Search(q) => assert_eq!(q, "whisper ggml"),
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn model_path_prefers_a_bundle_directory_over_the_bin_scheme() {
        use std::fs;
        let base = std::env::temp_dir().join(format!("dictata_bundle_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let dir = base.to_str().unwrap();
        fs::create_dir_all(&base).unwrap();

        // A ggml model resolves to `ggml-{name}.bin` (unchanged behaviour).
        assert_eq!(model_path(dir, "base"), base.join("ggml-base.bin"));
        assert!(!is_installed(dir, "base"));

        // A real ONNX bundle directory: model_path returns the directory itself,
        // it counts as installed, appears in the listing, and delete removes the
        // whole folder.
        let bundle = base.join("parakeet-tdt");
        fs::create_dir_all(&bundle).unwrap();
        for f in ["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"] {
            fs::write(bundle.join(f), b"x").unwrap();
        }
        assert_eq!(model_path(dir, "parakeet-tdt"), bundle);
        assert!(is_installed(dir, "parakeet-tdt"));
        assert!(installed_bundles(dir).contains(&"parakeet-tdt".to_string()));
        assert!(list_installed(dir).contains(&"parakeet-tdt".to_string()));

        delete(dir, "parakeet-tdt").unwrap();
        assert!(!bundle.exists(), "delete must remove the whole bundle directory");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn a_non_bundle_subdirectory_is_not_listed_as_installed() {
        use std::fs;
        let base = std::env::temp_dir().join(format!("dictata_nonbundle_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let dir = base.to_str().unwrap();
        // A stray directory with no recognizable backend signature.
        fs::create_dir_all(base.join("junk")).unwrap();
        fs::write(base.join("junk").join("readme.md"), b"x").unwrap();
        assert!(installed_bundles(dir).is_empty(), "a non-bundle dir must not count as installed");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn bundle_file_filter_covers_graphs_weights_and_sidecars() {
        // Graphs, external weights (must be fetched or a >2 GB model fails to
        // load), and vocab/config sidecars are in; unrelated repo files are out.
        for yes in [
            "encoder-model.onnx",
            "encoder-model.int8.onnx",
            "encoder-model.onnx.data",
            "model.onnx_data",
            "weights.data",
            "vocab.txt",
            "tokens.txt",
            "config.json",
            "bpe.model",
        ] {
            assert!(is_bundle_file(yes), "should be a bundle file: {yes}");
        }
        for no in ["README.md", ".gitattributes", "preview.png", "ggml-base.bin"] {
            assert!(!is_bundle_file(no), "should NOT be a bundle file: {no}");
        }
    }

    #[test]
    fn dir_name_stays_in_model_dir() {
        // Like file_name, only the leaf survives — a bundle name cannot escape.
        assert_eq!(dir_name("../../evil"), "evil");
        assert_eq!(dir_name("..\\..\\evil"), "evil");
        assert_eq!(dir_name("C:\\Windows\\System32"), "System32");
        assert_eq!(dir_name("parakeet-tdt-0.6b-v3"), "parakeet-tdt-0.6b-v3");
    }

    #[test]
    fn catalog_non_empty_and_unique() {
        assert!(!CATALOG.is_empty());
        let mut names: Vec<&str> = CATALOG.iter().map(|c| c.name).collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "noms de catalogue dupliques");
    }
}
