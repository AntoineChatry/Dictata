//! Applying a mode to the raw transcription.
//!
//! Pipeline: raw text -> replacements (dictionary, case-insensitive)
//! -> (optional) LLM reformatting for the mode.

use crate::config::{Config, Mode};
use crate::llm;
use indexmap::IndexMap;

/// Replace all occurrences of `from` with `to`, ignoring case.
/// Literal replacement (no regex). Assumes lowercasing
/// preserves byte length (true for typical French).
fn replace_case_insensitive(haystack: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return haystack.to_string();
    }
    let hay_lower = haystack.to_lowercase();
    let from_lower = from.to_lowercase();
    // If the length changes when lowercasing, we cannot safely index
    // the original: give up replacing this term.
    if hay_lower.len() != haystack.len() {
        return haystack.to_string();
    }
    let step = from_lower.len();
    let mut result = String::with_capacity(haystack.len());
    let mut last = 0;
    let mut start = 0;
    while let Some(pos) = hay_lower[start..].find(&from_lower) {
        let abs = start + pos;
        result.push_str(&haystack[last..abs]);
        result.push_str(to);
        last = abs + step;
        start = last;
    }
    result.push_str(&haystack[last..]);
    result
}

/// Apply the dictionary replacements (case-insensitive).
pub fn apply_replacements(text: &str, replacements: &IndexMap<String, String>) -> String {
    let mut out = text.to_string();
    for (src, dst) in replacements {
        if src.is_empty() {
            continue;
        }
        out = replace_case_insensitive(&out, src, dst);
    }
    out
}

/// Build the Whisper `initial_prompt` from the custom vocabulary.
pub fn build_initial_prompt(vocabulary: &[String], base_prompt: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !base_prompt.is_empty() {
        parts.push(base_prompt.to_string());
    }
    if !vocabulary.is_empty() {
        parts.push(format!("Vocabulaire : {}.", vocabulary.join(", ")));
    }
    parts.join(" ").trim().to_string()
}

/// System prompt for a take, with an explicit output-language directive when
/// the language is known.
///
/// The mode prompts are written in English, and small local models follow the
/// language of the system prompt over a rule buried inside it: a French
/// dictation comes back translated into English (measured on qwen3-4b).
///
/// Both the placement and the wording were measured on qwen3-4b, because the
/// obvious phrasings silently disable the mode:
///
/// - Appended *after* the mode prompt, the directive becomes the last and
///   dominant instruction and the model returns a near verbatim copy, losing
///   the mode's formatting entirely.
/// - Any wording containing "never translate it" is read as "reproduce the
///   input faithfully" and has the same effect, even as a prefix: a short
///   dictation came back unchanged instead of being formatted as an email.
///
/// A bare setting line, before the mode prompt, sets the language without
/// implying fidelity, and leaves the mode's own rules last and strongest.
fn prompt_for_language(base: &str, language: Option<&str>) -> String {
    match language {
        Some(lang) if !lang.is_empty() => format!("Output language: {lang}.\n\n{base}"),
        _ => base.to_string(),
    }
}

/// Returns `(final_text, status)`.
/// status: "raw", "llm" (reformatted) or "llm_fallback" (LLM unavailable -> raw).
/// `language` is the full English name of the dictation language as detected
/// by whisper (e.g. "french"), or `None` when it is unknown.
pub fn apply_mode(
    raw_text: &str,
    mode: &Mode,
    cfg: &Config,
    language: Option<&str>,
) -> (String, &'static str) {
    let text = apply_replacements(raw_text, &cfg.replacements);

    if mode.kind == "llm" && !mode.prompt.is_empty() {
        if !cfg.llm.enabled {
            return (text, "llm_fallback");
        }
        // A mode that deliberately translates must not be pinned to the
        // source language.
        let sys = if mode.task == "translate" {
            mode.prompt.clone()
        } else {
            prompt_for_language(&mode.prompt, language)
        };
        return match llm::transform(&text, &sys, &cfg.llm) {
            Ok(t) if !t.is_empty() => (t, "llm"),
            Ok(_) => (text, "llm"),
            Err(_) => (text, "llm_fallback"),
        };
    }

    (text, "raw")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn replacements_case_insensitive() {
        let mut r = IndexMap::new();
        r.insert("github".into(), "GitHub".into());
        r.insert("rust".into(), "Rust".into());
        let out = apply_replacements("j'utilise Github et RUST", &r);
        assert_eq!(out, "j'utilise GitHub et Rust");
    }

    #[test]
    fn replacements_empty_noop() {
        let r = IndexMap::new();
        assert_eq!(apply_replacements("bonjour", &r), "bonjour");
    }

    #[test]
    fn initial_prompt_builds() {
        let voc = vec!["Dictata".to_string(), "egui".to_string()];
        assert_eq!(
            build_initial_prompt(&voc, "Contexte technique."),
            "Contexte technique. Vocabulaire : Dictata, egui."
        );
        assert_eq!(build_initial_prompt(&[], ""), "");
    }

    #[test]
    fn raw_mode_returns_raw() {
        let cfg = Config::default();
        let mode = cfg.modes.get("raw").unwrap().clone();
        let (text, status) = apply_mode("  bonjour  ", &mode, &cfg, None);
        assert_eq!(status, "raw");
        assert_eq!(text, "  bonjour  "); // raw does not trim whitespace
    }

    #[test]
    fn llm_mode_disabled_falls_back() {
        let cfg = Config::default(); // llm.enabled = false
        let mode = cfg.modes.get("email").unwrap().clone();
        let (text, status) = apply_mode("ceci est un test", &mode, &cfg, None);
        assert_eq!(status, "llm_fallback");
        assert_eq!(text, "ceci est un test");
    }
}
