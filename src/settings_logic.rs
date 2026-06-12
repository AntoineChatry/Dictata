//! Business logic of the settings window, with no UI dependency.
//!
//! Extracted from `settings.rs` (2026-06-12 audit) to be unit
//! testable: parsing of the text editors (vocabulary, replacements),
//! mode key normalization, shortcut token display, and the
//! "transcribe a file" pipeline.

use indexmap::IndexMap;
use std::path::Path;

use crate::config::Config;
use crate::transcriber::Transcriber;
use crate::{audio, models, modes};

/// Vocabulary editor -> list (one entry per non-empty line).
pub fn parse_vocab(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

/// Replacements editor -> ordered map (`key = value` per line).
pub fn parse_repl(text: &str) -> IndexMap<String, String> {
    let mut m = IndexMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim();
            if !k.is_empty() {
                m.insert(k.to_string(), v.trim().to_string());
            }
        }
    }
    m
}

/// Entered name -> valid mode key ("Mon Mode" -> "mon_mode"), or None if empty.
pub fn normalize_mode_key(name: &str) -> Option<String> {
    let key = name.trim().to_lowercase().replace(' ', "_");
    if key.is_empty() { None } else { Some(key) }
}

/// First letter uppercased (default label of a new mode).
pub fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Shortcut token -> chip label ("ctrl" -> "Ctrl", "super" -> "Win").
pub fn pretty_token(tok: &str) -> String {
    match tok.to_lowercase().as_str() {
        "ctrl" => "Ctrl".into(),
        "alt" => "Alt".into(),
        "shift" => "Shift".into(),
        "super" | "win" | "windows" | "meta" => "Win".into(),
        "space" => "Space".into(),
        "enter" => "Enter".into(),
        other => capitalize(other),
    }
}

/// "Transcribe a file" pipeline: decodes via ffmpeg, loads a dedicated
/// transcriber, applies the active mode. Blocking — call from
/// a worker thread.
pub fn transcribe_file(cfg: &Config, path: &Path) -> Result<String, String> {
    let audio_data = audio::load_audio_file(path.to_str().ok_or("chemin non-UTF8")?)?;
    let mode = cfg
        .modes
        .get(&cfg.active_mode)
        .cloned()
        .or_else(|| cfg.modes.get("raw").cloned())
        .ok_or("aucun mode")?;
    let model_path = models::model_path(&cfg.model_dir, &cfg.model);
    let t = Transcriber::load(&model_path, cfg.gpu != "cpu")?;
    let lang = mode.language.clone().or_else(|| cfg.language.clone());
    let prompt = modes::build_initial_prompt(&cfg.vocabulary, "");
    let prompt_opt = if prompt.is_empty() { None } else { Some(prompt.as_str()) };
    let raw = t.transcribe(&audio_data, lang.as_deref(), mode.task == "translate", prompt_opt, cfg.beam_size)?;
    let (text, _status) = modes::apply_mode(&raw, &mode, cfg);
    if text.trim().is_empty() {
        return Err("transcription vide".into());
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_ignores_blank_lines_and_trims() {
        let v = parse_vocab("  Rust \n\n  \nwhisper\n");
        assert_eq!(v, vec!["Rust".to_string(), "whisper".to_string()]);
    }

    #[test]
    fn repl_parses_pairs_and_skips_invalid() {
        let m = parse_repl("a = b\nsans egal\n = vide\nc=d");
        assert_eq!(m.len(), 2);
        assert_eq!(m["a"], "b");
        assert_eq!(m["c"], "d");
    }

    #[test]
    fn mode_key_normalized() {
        assert_eq!(normalize_mode_key("  Mon Mode "), Some("mon_mode".into()));
        assert_eq!(normalize_mode_key("   "), None);
    }

    #[test]
    fn tokens_prettified() {
        assert_eq!(pretty_token("ctrl"), "Ctrl");
        assert_eq!(pretty_token("super"), "Win");
        assert_eq!(pretty_token("f4"), "F4");
        assert_eq!(pretty_token(""), "");
    }
}
