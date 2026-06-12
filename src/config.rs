//! Configuration load/save (modeled on the Python v1).
//!
//! Simple, readable, hand-editable JSON. Missing keys are filled in with
//! defaults (`#[serde(default = ...)]`) to survive upgrades.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application base directory (next to the exe; overridden by the
/// `DICTATA_HOME` environment variable).
pub fn home_dir() -> PathBuf {
    if let Ok(h) = std::env::var("DICTATA_HOME") {
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_path() -> PathBuf {
    home_dir().join("config.json")
}

pub fn history_path() -> PathBuf {
    home_dir().join("history.jsonl")
}

pub fn default_model_dir() -> String {
    home_dir().join("models").to_string_lossy().into_owned()
}

// ---------- defaults (functions for serde) ----------
fn default_hotkey() -> String {
    "ctrl+alt+space".into()
}
fn default_activation() -> String {
    "toggle".into()
}
fn default_true() -> bool {
    true
}
fn default_model() -> String {
    "base".into()
}
fn default_gpu() -> String {
    "auto".into()
}
fn default_active_mode() -> String {
    "raw".into()
}
fn default_ui_lang() -> String {
    "fr".into()
}
fn default_dock_scale() -> f32 {
    1.0
}
fn default_dock_opacity() -> f32 {
    0.93
}
fn default_audio_source() -> String {
    "mic".into()
}
fn default_paste_restore_delay() -> f32 {
    0.5
}
fn default_max_record_seconds() -> u32 {
    600
}
fn default_beam_size() -> i32 {
    5
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Mode {
    pub label: String,
    #[serde(rename = "type")]
    pub kind: String, // "raw" | "llm"
    #[serde(default = "default_task")]
    pub task: String, // "transcribe" | "translate"
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: String,
}
fn default_task() -> String {
    "transcribe".into()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LlmConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_llm_url")]
    pub base_url: String,
    #[serde(default = "default_api_key")]
    pub api_key: String,
    #[serde(default = "default_llm_model")]
    pub model: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_temp")]
    pub temperature: f32,
}
fn default_llm_url() -> String {
    "http://localhost:1234/v1".into()
}
fn default_api_key() -> String {
    "not-needed".into()
}
fn default_llm_model() -> String {
    "local-model".into()
}
fn default_timeout() -> u64 {
    60
}
fn default_temp() -> f32 {
    0.2
}
impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            enabled: false,
            base_url: default_llm_url(),
            api_key: default_api_key(),
            model: default_llm_model(),
            timeout: default_timeout(),
            temperature: default_temp(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    #[serde(default = "default_ui_lang")]
    pub ui_lang: String, // "fr" | "en" | "es"
    #[serde(default = "default_dock_scale")]
    pub dock_scale: f32, // 0.7 - 1.6
    #[serde(default = "default_dock_opacity")]
    pub dock_opacity: f32, // 0.4 - 1.0
    #[serde(default)]
    pub dock_pos: Option<[f32; 2]>, // pixels; None = bottom-center
    #[serde(default = "default_hotkey")]
    pub hotkey: String,
    #[serde(default = "default_activation")]
    pub activation: String, // "toggle" | "push_to_talk"
    #[serde(default)]
    pub input_device: Option<String>, // device name (cpal)
    #[serde(default = "default_audio_source")]
    pub audio_source: String, // "mic" | "system" | "mix"
    #[serde(default = "default_true")]
    pub auto_paste: bool,
    #[serde(default)]
    pub streaming: bool, // progressive insertion (raw mode)
    #[serde(default = "default_true")]
    pub play_sounds: bool,
    #[serde(default = "default_paste_restore_delay")]
    pub paste_restore_delay: f32,
    #[serde(default = "default_max_record_seconds")]
    pub max_record_seconds: u32,
    #[serde(default = "default_beam_size")]
    pub beam_size: i32,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default = "default_gpu")]
    pub gpu: String, // "auto" | "cpu" | "vulkan" | "cuda"
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
    #[serde(default)]
    pub vocabulary: Vec<String>,
    #[serde(default)]
    pub replacements: IndexMap<String, String>,
    #[serde(default = "default_active_mode")]
    pub active_mode: String,
    #[serde(default = "default_modes")]
    pub modes: IndexMap<String, Mode>,
    #[serde(default)]
    pub llm: LlmConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            ui_lang: default_ui_lang(),
            dock_scale: default_dock_scale(),
            dock_opacity: default_dock_opacity(),
            dock_pos: None,
            hotkey: default_hotkey(),
            activation: default_activation(),
            input_device: None,
            audio_source: default_audio_source(),
            auto_paste: true,
            streaming: false,
            play_sounds: true,
            paste_restore_delay: default_paste_restore_delay(),
            max_record_seconds: default_max_record_seconds(),
            beam_size: default_beam_size(),
            model: default_model(),
            language: None,
            gpu: default_gpu(),
            model_dir: default_model_dir(),
            vocabulary: Vec::new(),
            replacements: IndexMap::new(),
            active_mode: default_active_mode(),
            modes: default_modes(),
            llm: LlmConfig::default(),
        }
    }
}

/// Default modes (same as the Python v1).
pub fn default_modes() -> IndexMap<String, Mode> {
    let mut m = IndexMap::new();
    m.insert("raw".into(), Mode {
        label: "Brut".into(), kind: "raw".into(), task: "transcribe".into(),
        language: None, prompt: String::new(),
    });
    m.insert("clean".into(), Mode {
        label: "Nettoye".into(), kind: "llm".into(), task: "transcribe".into(), language: None,
        prompt: "Tu es un assistant de dictee. Reecris le texte transcrit en corrigeant la \
ponctuation, les majuscules et les fautes evidentes, en supprimant les hesitations (euh, hum, \
repetitions). Conserve EXACTEMENT le sens et la langue d'origine. Ne reponds qu'avec le texte \
corrige, sans commentaire.".into(),
    });
    m.insert("email".into(), Mode {
        label: "Email".into(), kind: "llm".into(), task: "transcribe".into(), language: None,
        prompt: "Transforme la dictee suivante en un email clair et professionnel (formule \
d'appel, corps structure, formule de politesse). Garde la langue d'origine. Ne reponds qu'avec \
l'email, sans commentaire.".into(),
    });
    m.insert("message".into(), Mode {
        label: "Message".into(), kind: "llm".into(), task: "transcribe".into(), language: None,
        prompt: "Transforme la dictee suivante en un message court, naturel et bien ponctue, \
adapte a une messagerie instantanee. Garde la langue d'origine. Ne reponds qu'avec le \
message.".into(),
    });
    m.insert("list".into(), Mode {
        label: "Liste".into(), kind: "llm".into(), task: "transcribe".into(), language: None,
        prompt: "Transforme la dictee suivante en une liste a puces claire et concise. Garde la \
langue d'origine. Ne reponds qu'avec la liste.".into(),
    });
    m
}

pub fn load() -> Config {
    let path = config_path();
    if !path.exists() {
        let cfg = Config::default();
        save(&cfg);
        return cfg;
    }
    match std::fs::read_to_string(&path) {
        // Tolerate a UTF-8 BOM (files edited with Notepad/PowerShell).
        Ok(s) => match serde_json::from_str::<Config>(s.trim_start_matches('\u{feff}')) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("config.json invalide ({e}), utilisation des defauts");
                Config::default()
            }
        },
        Err(e) => {
            eprintln!("lecture config.json impossible ({e})");
            Config::default()
        }
    }
}

pub fn save(cfg: &Config) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(cfg) {
        Ok(s) => {
            // Atomic write (temp + rename) so a crash mid-write never leaves
            // a truncated config.json that load() would silently replace
            // with defaults.
            let tmp = path.with_extension("json.tmp");
            let res = std::fs::write(&tmp, s)
                .and_then(|_| std::fs::rename(&tmp, &path));
            if let Err(e) = res {
                eprintln!("ecriture config.json impossible ({e})");
                let _ = std::fs::remove_file(&tmp);
            }
        }
        Err(e) => eprintln!("serialisation config impossible ({e})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrip() {
        let cfg = Config::default();
        assert_eq!(cfg.model, "base");
        assert_eq!(cfg.modes.len(), 5);
        // mode order is preserved (IndexMap)
        let keys: Vec<&String> = cfg.modes.keys().collect();
        assert_eq!(keys, vec!["raw", "clean", "email", "message", "list"]);

        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.modes.len(), 5);
        assert_eq!(back.llm.base_url, "http://localhost:1234/v1");
    }

    #[test]
    fn partial_json_fills_defaults() {
        // a minimal JSON must be completed with the defaults
        let back: Config = serde_json::from_str(r#"{"model":"small"}"#).unwrap();
        assert_eq!(back.model, "small");
        assert_eq!(back.hotkey, "ctrl+alt+space");
        assert_eq!(back.modes.len(), 5);
        assert!(!back.llm.enabled);
    }
}
