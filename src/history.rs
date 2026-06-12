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

/// Append an entry to the end of the history file.
pub fn add_entry(text: &str, mode: &str, language: Option<&str>, duration: Option<f64>) {
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

    #[test]
    fn roundtrip_and_order() {
        // Isolate the history file in a temporary directory.
        let dir = std::env::temp_dir().join(format!("fwhist_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("DICTATA_HOME", &dir) };
        clear();

        add_entry("premier", "raw", Some("fr"), Some(1.234));
        add_entry("deuxieme", "email", None, None);
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
}
