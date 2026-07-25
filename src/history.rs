//! Transcription history (append-only JSONL file).
//!
//! Modeled on the Python v1 (`freewhisper/history.py`). The `datetime` field is
//! a readable local date (via `GetLocalTime` on Windows, UTC elsewhere) to
//! avoid adding a time-handling dependency.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry {
    pub ts: f64,
    pub datetime: String,
    pub mode: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    pub text: String,
}

/// Extra entries tolerated above `history_limit` before the file is rewritten.
///
/// Trimming reads and rewrites the whole file, and two of the three callers run
/// on the UI thread, so it must not happen on every dictation. With this slack
/// it happens once per `RETENTION_SLACK` entries instead.
///
/// The cap is therefore a high-water mark, not an exact count: the file cycles
/// between `history_limit` and `history_limit + RETENTION_SLACK` entries.
const RETENTION_SLACK: usize = 100;

/// Append an entry to the end of the history file, unless the user turned the
/// history off. Applies the retention cap afterwards.
pub fn add_entry(
    cfg: &config::Config,
    text: &str,
    mode: &str,
    language: Option<&str>,
    duration: Option<f64>,
) {
    if !cfg.save_history {
        return;
    }
    append_entry(text, mode, language, duration);
    enforce_retention(cfg.history_limit);
}

fn append_entry(text: &str, mode: &str, language: Option<&str>, duration: Option<f64>) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let entry = Entry {
        ts,
        datetime: now_string(),
        mode: mode.to_string(),
        language: language.map(|s| s.to_string()),
        duration: duration.map(|d| (d * 100.0).round() / 100.0),
        text: text.to_string(),
    };
    let path = config::history_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = match serde_json::to_string(&entry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("serialisation historique impossible ({e})");
            return;
        }
    };
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("ecriture historique impossible ({e})");
            }
        }
        Err(e) => eprintln!("ouverture historique impossible ({e})"),
    }
}

/// Drops the oldest entries so the file holds at most `limit` of them.
///
/// Only rewrites once the file exceeds `limit + RETENTION_SLACK`, and writes
/// through a temp file then renames, so an interrupted trim cannot truncate the
/// history.
fn enforce_retention(limit: usize) {
    let limit = limit.max(1);
    let path = config::history_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() <= limit + RETENTION_SLACK {
        return;
    }
    let kept = lines[lines.len() - limit..].join("\n");
    let tmp = path.with_extension("jsonl.tmp");
    let res = std::fs::write(&tmp, format!("{kept}\n")).and_then(|_| std::fs::rename(&tmp, &path));
    if let Err(e) = res {
        eprintln!("rotation historique impossible ({e})");
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Return the last `limit` entries, most recent first.
pub fn read_entries(limit: usize) -> Vec<Entry> {
    let path = config::history_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<Entry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Entry>(l).ok())
        .collect();
    if entries.len() > limit {
        entries = entries.split_off(entries.len() - limit);
    }
    entries.reverse();
    entries
}

pub fn clear() {
    let path = config::history_path();
    let _ = std::fs::remove_file(path);
}

/// "YYYY-MM-DD HH:MM:SS" in local time (Windows) or UTC (others).
fn now_string() -> String {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::SYSTEMTIME;
        use windows_sys::Win32::System::SystemInformation::GetLocalTime;
        unsafe {
            let mut st: SYSTEMTIME = std::mem::zeroed();
            GetLocalTime(&mut st);
            return format!(
                "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond
            );
        }
    }
    #[cfg(not(windows))]
    {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (y, mo, d, h, mi, s) = civil_from_unix(secs as i64);
        format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
    }
}

/// Unix timestamp -> UTC civil date conversion (Howard Hinnant's algorithm).
#[cfg(not(windows))]
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, h, mi, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points `DICTATA_HOME` at a private directory for the duration of a test.
    ///
    /// The history tests share one process-wide env var, so they must not run
    /// concurrently; they are kept in a single `#[test]` each with a distinct
    /// directory and are serialised by `HISTORY_ENV` below.
    fn use_temp_home(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fwhist_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("DICTATA_HOME", &dir) };
        dir
    }

    static HISTORY_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn roundtrip_and_order() {
        let _lock = HISTORY_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let dir = use_temp_home("roundtrip");
        let cfg = config::Config::default();
        clear();

        add_entry(&cfg, "premier", "raw", Some("fr"), Some(1.234));
        add_entry(&cfg, "deuxieme", "email", None, None);
        let entries = read_entries(50);
        assert_eq!(entries.len(), 2);
        // most recent first
        assert_eq!(entries[0].text, "deuxieme");
        assert_eq!(entries[1].text, "premier");
        assert_eq!(entries[1].language.as_deref(), Some("fr"));
        assert_eq!(entries[1].duration, Some(1.23));
        // datetime non-empty and of the right length "YYYY-MM-DD HH:MM:SS"
        assert_eq!(entries[0].datetime.len(), 19);

        clear();
        assert!(read_entries(50).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_history_writes_nothing() {
        let _lock = HISTORY_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let dir = use_temp_home("disabled");
        let cfg = config::Config {
            save_history: false,
            ..config::Config::default()
        };
        clear();

        add_entry(&cfg, "ne doit pas etre ecrit", "raw", None, None);
        assert!(
            !config::history_path().exists(),
            "le fichier ne doit meme pas etre cree"
        );
        assert!(read_entries(50).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retention_drops_the_oldest_entries() {
        let _lock = HISTORY_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let dir = use_temp_home("retention");
        let cfg = config::Config { history_limit: 5, ..config::Config::default() };
        clear();

        // The trim fires once past limit + RETENTION_SLACK and cuts back to the
        // limit, so the file cycles between the two bounds rather than sitting
        // exactly on the limit.
        let total = 5 + RETENTION_SLACK + 3;
        for i in 0..total {
            add_entry(&cfg, &format!("entree {i}"), "raw", None, None);
        }
        let entries = read_entries(10_000);
        assert!(
            entries.len() <= 5 + RETENTION_SLACK,
            "la borne haute doit etre respectee: {}",
            entries.len()
        );
        // read_entries returns most recent first: the newest survived...
        assert_eq!(entries[0].text, format!("entree {}", total - 1));
        // ...and the oldest were dropped.
        assert!(
            !entries.iter().any(|e| e.text == "entree 0"),
            "la plus ancienne entree aurait du etre supprimee"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn retention_keeps_entries_below_the_slack() {
        let _lock = HISTORY_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let dir = use_temp_home("slack");
        let cfg = config::Config { history_limit: 5, ..config::Config::default() };
        clear();

        // Just above the limit but within the slack: nothing is rewritten yet.
        for i in 0..8 {
            add_entry(&cfg, &format!("entree {i}"), "raw", None, None);
        }
        assert_eq!(read_entries(1000).len(), 8);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
