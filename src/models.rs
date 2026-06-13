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
    CatalogEntry { name: "tiny-q5_1", label: "Tiny (quantifie)", size_mb: 31, quantized: true },
    CatalogEntry { name: "tiny", label: "Tiny", size_mb: 75, quantized: false },
    CatalogEntry { name: "base-q5_1", label: "Base (quantifie)", size_mb: 57, quantized: true },
    CatalogEntry { name: "base", label: "Base", size_mb: 141, quantized: false },
    CatalogEntry { name: "small-q5_1", label: "Small (quantifie)", size_mb: 181, quantized: true },
    CatalogEntry { name: "small", label: "Small", size_mb: 465, quantized: false },
    CatalogEntry { name: "medium-q5_0", label: "Medium (quantifie)", size_mb: 514, quantized: true },
    CatalogEntry { name: "medium", label: "Medium", size_mb: 1463, quantized: false },
    CatalogEntry { name: "large-v3-turbo-q5_0", label: "Large v3 Turbo (quantifie)", size_mb: 547, quantized: true },
    CatalogEntry { name: "large-v3-turbo", label: "Large v3 Turbo", size_mb: 1549, quantized: false },
    CatalogEntry { name: "large-v3-q5_0", label: "Large v3 (quantifie)", size_mb: 1031, quantized: true },
    CatalogEntry { name: "large-v3", label: "Large v3", size_mb: 2952, quantized: false },
];

pub fn file_name(model: &str) -> String {
    // Custom models (HuggingFace search) are stored under their full file
    // name; catalog names keep the `ggml-{name}.bin` scheme.
    if model.ends_with(".bin") {
        model.to_string()
    } else {
        format!("ggml-{model}.bin")
    }
}

pub fn model_path(model_dir: &str, model: &str) -> PathBuf {
    Path::new(model_dir).join(file_name(model))
}

pub fn is_installed(model_dir: &str, model: &str) -> bool {
    model_path(model_dir, model).exists()
}

/// Delete an installed model file.
pub fn delete(model_dir: &str, model: &str) -> Result<(), String> {
    std::fs::remove_file(model_path(model_dir, model)).map_err(|e| e.to_string())
}

/// List the ggml models present in `model_dir` (names without `ggml-`/`.bin`).
pub fn list_installed(model_dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(model_dir) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str() {
                if let Some(s) = n.strip_prefix("ggml-").and_then(|s| s.strip_suffix(".bin")) {
                    out.push(s.to_string());
                } else if n.ends_with(".bin") {
                    // Custom model: identified by its full file name.
                    out.push(n.to_string());
                }
            }
        }
    }
    out.sort();
    out
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

/// Download an arbitrary URL into `model_dir` under `fname` (same `.part`
/// then rename scheme as `download`).
pub fn download_url(
    model_dir: &str,
    url: &str,
    fname: &str,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(model_dir).map_err(|e| format!("dossier modeles: {e}"))?;
    let dest = Path::new(model_dir).join(fname);
    let tmp = dest.with_extension("part");

    // Default blocking client: no timeout (large downloads).
    let mut resp = reqwest::blocking::Client::new()
        .get(url)
        .send()
        .map_err(|e| format!("requete: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} pour {url}", resp.status()));
    }
    let total = resp.content_length();

    let mut file = std::fs::File::create(&tmp).map_err(|e| format!("creation fichier: {e}"))?;
    let mut buf = [0u8; 65536];
    let mut received = 0u64;
    loop {
        let n = resp.read(&mut buf).map_err(|e| format!("lecture flux: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| format!("ecriture: {e}"))?;
        received += n as u64;
        progress(received, total);
    }
    drop(file);
    std::fs::rename(&tmp, &dest).map_err(|e| format!("renommage: {e}"))?;
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
        .map_err(|e| format!("requete: {e}"))?;
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
    if let Some(rest) = s
        .strip_prefix("https://huggingface.co/")
        .or_else(|| s.strip_prefix("http://huggingface.co/"))
    {
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
    fn catalog_non_empty_and_unique() {
        assert!(!CATALOG.is_empty());
        let mut names: Vec<&str> = CATALOG.iter().map(|c| c.name).collect();
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "noms de catalogue dupliques");
    }
}
