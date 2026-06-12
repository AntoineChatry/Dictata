//! Dictata — system-wide voice dictation, 100% local (native Rust version).
//! Library exposing the modules consumed by the binary and the examples.

pub mod audio;
pub mod config;
pub mod dock;
pub mod hardware;
pub mod history;
pub mod hotkey;
pub mod i18n;
pub mod llm;
pub mod models;
pub mod modes;
pub mod paste;
pub mod platform;
pub mod settings;
pub mod settings_logic;
pub mod streaming;
pub mod transcriber;
pub mod tray;
