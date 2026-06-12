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
    format!("ggml-{model}.bin")
}

pub fn model_path(model_dir: &str, model: &str) -> PathBuf {
    Path::new(model_dir).join(file_name(model))
}

pub fn is_installed(model_dir: &str, model: &str) -> bool {
    model_path(model_dir, model).exists()
}

/// List the ggml models present in `model_dir` (names without `ggml-`/`.bin`).
pub fn list_installed(model_dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(model_dir) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str() {
                if let Some(s) = n.strip_prefix("ggml-").and_then(|s| s.strip_suffix(".bin")) {
                    out.push(s.to_string());
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
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let fname = file_name(model);
    let url = format!("{HF_BASE}/{fname}");
    std::fs::create_dir_all(model_dir).map_err(|e| format!("dossier modeles: {e}"))?;
    let dest = Path::new(model_dir).join(&fname);
    let tmp = dest.with_extension("part");

    // Default blocking client: no timeout (large downloads).
    let mut resp = reqwest::blocking::Client::new()
        .get(&url)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_scheme() {
        assert_eq!(file_name("base"), "ggml-base.bin");
        assert_eq!(file_name("small-q5_1"), "ggml-small-q5_1.bin");
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
