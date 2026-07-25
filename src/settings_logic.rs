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

/// App-mode mapping editor -> ordered map (`pattern = mode_key` per line).
/// Patterns are lowercased to match `platform::foreground_app`.
pub fn parse_app_modes(text: &str) -> IndexMap<String, String> {
    let mut m = IndexMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim().to_lowercase();
            let v = v.trim();
            if !k.is_empty() && !v.is_empty() {
                m.insert(k, v.to_string());
            }
        }
    }
    m
}

/// First mapping entry matching the foreground window, or None.
///
/// A pattern matches when it equals the executable name or appears anywhere
/// in the window title, so `firefox.exe = message` and `gmail = email` can
/// coexist: the map is ordered, so the first line written wins and the user
/// controls priority by putting the specific patterns above the generic ones.
/// `exe` and `title` are expected lowercased (as `platform::foreground_app`
/// returns them).
pub fn match_app_mode<'a>(
    map: &'a IndexMap<String, String>,
    exe: &str,
    title: &str,
) -> Option<&'a String> {
    map.iter()
        .find(|(pat, _)| pat.as_str() == exe || (!title.is_empty() && title.contains(pat.as_str())))
        .map(|(_, mode)| mode)
}

/// Whether `base_url` provably points at this machine.
///
/// The application promises that dictations never leave the machine, but the
/// LLM endpoint is a free text field: pointing it at a remote host sends the
/// full text of every dictation there. The setting stays free — this only lets
/// the UI say so.
///
/// Returns `None` when the URL cannot be parsed or has no host: nothing can be
/// claimed either way, and the "Test" button already reports an unusable
/// endpoint. `Url` comes from `reqwest`'s public re-export, so no new
/// dependency is involved.
pub fn llm_endpoint_is_local(base_url: &str) -> Option<bool> {
    let url = reqwest::Url::parse(base_url.trim()).ok()?;
    let host = url.host_str()?;
    // `host_str` returns a slice of the serialised URL, which keeps the
    // brackets around an IPv6 literal; strip them before parsing.
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return Some(ip.is_loopback());
    }
    Some(host.eq_ignore_ascii_case("localhost"))
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
    let mut t = Transcriber::load(&model_path, cfg.gpu != "cpu")?;
    let lang = mode.language.clone().or_else(|| cfg.language.clone());
    let prompt = modes::build_initial_prompt(&cfg.vocabulary, "");
    let prompt_opt = if prompt.is_empty() { None } else { Some(prompt.as_str()) };
    let raw = t.transcribe(&audio_data, lang.as_deref(), mode.task == "translate", prompt_opt, cfg.beam_size, None)?;
    let lang_detected = t.detected_language();
    let (text, _status) = modes::apply_mode(&raw, &mode, cfg, lang_detected);
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
    fn app_modes_lowercases_keys_and_skips_invalid() {
        let m = parse_app_modes("Outlook.EXE = email\nno-equal\nchrome.exe =\n  Code.exe = raw ");
        assert_eq!(m.len(), 2);
        assert_eq!(m["outlook.exe"], "email");
        assert_eq!(m["code.exe"], "raw");
    }

    #[test]
    fn app_mode_matches_title_then_exe_in_order() {
        // The real case: one browser executable, several sites. The title
        // discriminates, and the first matching line wins.
        let m = parse_app_modes("gmail = email\nfirefox.exe = message\noutlook.exe = email");
        let gmail_title = "boite de reception (12) - gmail - mozilla firefox";
        assert_eq!(
            match_app_mode(&m, "firefox.exe", gmail_title),
            Some(&"email".to_string())
        );
        // Same executable, another site -> falls through to the exe line.
        assert_eq!(
            match_app_mode(&m, "firefox.exe", "github - mozilla firefox"),
            Some(&"message".to_string())
        );
        // Exe match with no title at all.
        assert_eq!(
            match_app_mode(&m, "outlook.exe", ""),
            Some(&"email".to_string())
        );
        // Nothing matches -> caller keeps the manual mode.
        assert_eq!(match_app_mode(&m, "code.exe", "main.rs - dictata"), None);
    }

    #[test]
    fn app_mode_empty_title_does_not_match_everything() {
        let m = parse_app_modes("gmail = email");
        assert_eq!(match_app_mode(&m, "firefox.exe", ""), None);
    }

    #[test]
    fn llm_endpoint_locality_is_detected() {
        // The defaults and the usual local setups.
        assert_eq!(llm_endpoint_is_local("http://localhost:1234/v1"), Some(true));
        assert_eq!(llm_endpoint_is_local("http://127.0.0.1:1234/v1"), Some(true));
        assert_eq!(llm_endpoint_is_local("http://127.2.3.4:11434/v1"), Some(true));
        assert_eq!(llm_endpoint_is_local("http://LocalHost:1234/v1"), Some(true));
        assert_eq!(llm_endpoint_is_local("  http://localhost:1234/v1  "), Some(true));
        // IPv6 loopback, whichever way host_str brackets it.
        assert_eq!(llm_endpoint_is_local("http://[::1]:1234/v1"), Some(true));
        // Anything else leaves the machine.
        assert_eq!(llm_endpoint_is_local("https://api.example.com/v1"), Some(false));
        assert_eq!(llm_endpoint_is_local("http://192.168.1.20:1234/v1"), Some(false));
        // A LAN address is not this machine either, even on a private network.
        assert_eq!(llm_endpoint_is_local("http://10.0.0.5:1234/v1"), Some(false));
        // Unparseable or hostless: no claim is made.
        assert_eq!(llm_endpoint_is_local(""), None);
        assert_eq!(llm_endpoint_is_local("pas une url"), None);
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
