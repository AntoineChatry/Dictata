//! Hardware detection + model recommendation, LM Studio style.
//!
//! The default binary runs whisper.cpp on **CPU** (no GPU backend
//! compiled): the recommendation is therefore based on RAM and core count.
//! Memory footprints are conservative estimates (quantized ggml).

/// Approximate memory footprint (GB) + CPU speed tier per size.
struct ModelInfo {
    ram_gb: f32,
    speed: Speed,
}

#[derive(Clone, Copy)]
enum Speed {
    Fast,
    Medium,
    Slow,
    VerySlow,
}

impl Speed {
    fn label(self) -> &'static str {
        match self {
            Speed::Fast => "rapide",
            Speed::Medium => "correct sur CPU",
            Speed::Slow => "lent sur CPU",
            Speed::VerySlow => "tres lent sur CPU",
        }
    }
}

fn model_info(key: &str) -> Option<ModelInfo> {
    let (ram_gb, speed) = match key {
        "tiny" => (0.6, Speed::Fast),
        "base" => (0.9, Speed::Fast),
        "small" => (1.6, Speed::Medium),
        "medium" => (3.0, Speed::Slow),
        "large-v3-turbo" => (3.0, Speed::Medium),
        "large-v3" => (5.0, Speed::VerySlow),
        _ => return None,
    };
    Some(ModelInfo { ram_gb, speed })
}

#[derive(Debug, Clone)]
pub struct HwInfo {
    pub ram_total_gb: Option<f32>,
    pub ram_avail_gb: Option<f32>,
    pub cpu_physical: usize,
    pub cpu_logical: usize,
    pub cpu_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Ideal,
    Good,
    Heavy,
    TooBig,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Rating {
    pub level: Level,
    pub label: String,
    pub detail: String,
    /// Indicative color (hex) for the UI.
    pub color: &'static str,
}

/// (total, available) in bytes, or None if undetectable.
#[cfg(windows)]
fn ram_bytes() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            Some((m.ullTotalPhys, m.ullAvailPhys))
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
fn ram_bytes() -> Option<(u64, u64)> {
    // Linux: /proc/meminfo (MemTotal / MemAvailable, in KiB).
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut avail = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = rest.split_whitespace().next().and_then(|n| n.parse::<u64>().ok());
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail = rest.split_whitespace().next().and_then(|n| n.parse::<u64>().ok());
        }
    }
    let total = total? * 1024;
    let avail = avail.map(|a| a * 1024).unwrap_or(total);
    Some((total, avail))
}

#[cfg(windows)]
fn cpu_name() -> String {
    // Read the registry via reg.exe to avoid an extra dependency.
    let out = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output();
    if let Ok(o) = out {
        let s = String::from_utf8_lossy(&o.stdout);
        if let Some(pos) = s.find("REG_SZ") {
            return s[pos + 6..].trim().to_string();
        }
    }
    "CPU".to_string()
}

#[cfg(not(windows))]
fn cpu_name() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some(rest) = line.to_lowercase().strip_prefix("model name") {
                if let Some(idx) = rest.find(':') {
                    return rest[idx + 1..].trim().to_string();
                }
            }
        }
    }
    "CPU".to_string()
}

pub fn detect() -> HwInfo {
    let ram = ram_bytes();
    let to_gb = |b: u64| (b as f64 / 1024f64.powi(3) * 10.0).round() as f32 / 10.0;
    HwInfo {
        ram_total_gb: ram.map(|(t, _)| to_gb(t)),
        ram_avail_gb: ram.map(|(_, a)| to_gb(a)),
        cpu_physical: num_cpus::get_physical(),
        cpu_logical: num_cpus::get(),
        cpu_name: cpu_name(),
    }
}

/// Guess the size (`model_info` key) from a name or repo.
pub fn size_key(name_or_id: &str) -> Option<&'static str> {
    let s = name_or_id.to_lowercase();
    if s.contains("turbo") {
        Some("large-v3-turbo")
    } else if s.contains("large") {
        Some("large-v3")
    } else if s.contains("medium") {
        Some("medium")
    } else if s.contains("small") {
        Some("small")
    } else if s.contains("base") {
        Some("base")
    } else if s.contains("tiny") {
        Some("tiny")
    } else {
        None
    }
}

/// Ideal model for this machine (the "star" LM Studio style).
pub fn recommended(info: &HwInfo) -> &'static str {
    let ram = info.ram_total_gb.unwrap_or(4.0);
    let cores = if info.cpu_physical > 0 {
        info.cpu_physical
    } else {
        info.cpu_logical.max(1)
    };
    if ram < 4.0 || cores <= 2 {
        "tiny"
    } else if cores >= 8 && ram >= 12.0 {
        "small"
    } else if cores >= 4 && ram >= 8.0 {
        "base"
    } else {
        "tiny"
    }
}

/// Rate a model for the machine.
pub fn rate(name_or_id: &str, info: &HwInfo) -> Rating {
    let key = match size_key(name_or_id) {
        Some(k) => k,
        None => {
            return Rating {
                level: Level::Unknown,
                label: String::new(),
                detail: "Taille inconnue".into(),
                color: "#9a9aa5",
            }
        }
    };
    let spec = model_info(key).expect("cle connue");
    let reco = recommended(info);
    let is_reco = size_key(reco) == Some(key);

    // CPU: RAM is a hard constraint, speed a soft constraint.
    if let Some(ram_total) = info.ram_total_gb {
        if spec.ram_gb > (ram_total - 1.0).max(0.0) {
            return Rating {
                level: Level::TooBig,
                label: "Trop lourd".into(),
                detail: format!("~{:.0} Go requis > RAM dispo", spec.ram_gb),
                color: "#e05555",
            };
        }
    }

    if is_reco {
        Rating {
            level: Level::Ideal,
            label: "\u{2605} Ideal pour ta machine".into(),
            detail: spec.speed.label().into(),
            color: "#5eaaff",
        }
    } else {
        match spec.speed {
            Speed::Fast | Speed::Medium => Rating {
                level: Level::Good,
                label: "Recommande".into(),
                detail: spec.speed.label().into(),
                color: "#5ad28c",
            },
            _ => Rating {
                level: Level::Heavy,
                label: "Lourd".into(),
                detail: spec.speed.label().into(),
                color: "#e0a23c",
            },
        }
    }
}

pub fn summary(info: &HwInfo) -> String {
    let mut parts = Vec::new();
    if let Some(r) = info.ram_total_gb {
        parts.push(format!("{r:.0} Go RAM"));
    }
    let cores = if info.cpu_physical > 0 {
        info.cpu_physical
    } else {
        info.cpu_logical
    };
    if cores > 0 {
        parts.push(format!("{cores} coeurs"));
    }
    parts.push("CPU".to_string());
    parts.join(" \u{00b7} ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(ram: f32, cores: usize) -> HwInfo {
        HwInfo {
            ram_total_gb: Some(ram),
            ram_avail_gb: Some(ram),
            cpu_physical: cores,
            cpu_logical: cores,
            cpu_name: "test".into(),
        }
    }

    #[test]
    fn size_key_variants() {
        assert_eq!(size_key("ggml-large-v3-turbo-q5_0.bin"), Some("large-v3-turbo"));
        assert_eq!(size_key("base"), Some("base"));
        assert_eq!(size_key("small-q5_1"), Some("small"));
        assert_eq!(size_key("inconnu"), None);
    }

    #[test]
    fn reco_tiers() {
        assert_eq!(recommended(&machine(2.0, 4)), "tiny"); // low RAM
        assert_eq!(recommended(&machine(8.0, 4)), "base");
        assert_eq!(recommended(&machine(16.0, 12)), "small");
        assert_eq!(recommended(&machine(4.0, 2)), "tiny"); // 2 cores
    }

    #[test]
    fn rate_too_big_and_ideal() {
        let m8 = machine(8.0, 4); // reco = base
        assert_eq!(rate("base", &m8).level, Level::Ideal);
        let m4 = machine(4.0, 4); // reco = tiny
        assert_eq!(rate("tiny", &m4).level, Level::Ideal);
        assert_eq!(rate("large-v3", &m4).level, Level::TooBig); // 5 GB > 3 GB available
        assert_eq!(rate("inconnu", &m4).level, Level::Unknown);
    }
}
