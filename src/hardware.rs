//! Hardware detection + model recommendation, LM Studio style.
//!
//! With the `vulkan` feature (default) and `gpu != "cpu"`, ratings are based
//! on the detected GPU VRAM; otherwise on RAM and core count (CPU mode).
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
    /// Most capable GPU detected (highest VRAM), if any.
    pub gpu_name: Option<String>,
    pub vram_gb: Option<f32>,
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

/// (name, VRAM GB) of the GPU with the most VRAM, via the display-class
/// registry key (same reg.exe approach as `cpu_name`).
#[cfg(windows)]
fn gpu_info() -> Option<(String, f32)> {
    const KEY: &str =
        r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
    // Lines of `reg query /s /v <value>`: subkey path, then the value line.
    let query = |value: &str| -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Ok(o) = std::process::Command::new("reg")
            .args(["query", KEY, "/s", "/v", value])
            .output()
        {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut cur = String::new();
            for line in s.lines() {
                if line.starts_with("HKEY_") {
                    cur = line.trim().to_string();
                } else if let Some(pos) = line.find("REG_") {
                    if let Some(data) = line[pos..].split_once(char::is_whitespace) {
                        out.push((cur.clone(), data.1.trim().to_string()));
                    }
                }
            }
        }
        out
    };
    let names: std::collections::HashMap<String, String> =
        query("DriverDesc").into_iter().collect();
    let mut best: Option<(String, f32)> = None;
    for (key, hex) in query("HardwareInformation.qwMemorySize") {
        let bytes = u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or(0);
        let gb = (bytes as f64 / 1024f64.powi(3) * 10.0).round() as f32 / 10.0;
        if gb > 0.0 && best.as_ref().is_none_or(|(_, b)| gb > *b) {
            if let Some(name) = names.get(&key) {
                best = Some((name.clone(), gb));
            }
        }
    }
    best
}

#[cfg(not(windows))]
fn gpu_info() -> Option<(String, f32)> {
    // No portable VRAM detection without an extra dependency: CPU ratings.
    None
}

pub fn detect() -> HwInfo {
    let ram = ram_bytes();
    let to_gb = |b: u64| (b as f64 / 1024f64.powi(3) * 10.0).round() as f32 / 10.0;
    let gpu = gpu_info();
    HwInfo {
        ram_total_gb: ram.map(|(t, _)| to_gb(t)),
        ram_avail_gb: ram.map(|(_, a)| to_gb(a)),
        cpu_physical: num_cpus::get_physical(),
        cpu_logical: num_cpus::get(),
        cpu_name: cpu_name(),
        gpu_name: gpu.as_ref().map(|(n, _)| n.clone()),
        vram_gb: gpu.map(|(_, v)| v),
    }
}

/// True when transcription will actually run on the GPU: Vulkan compiled in,
/// config not forcing CPU, and a GPU detected.
pub fn gpu_active(info: &HwInfo, gpu_cfg: &str) -> bool {
    cfg!(feature = "vulkan") && gpu_cfg != "cpu" && info.gpu_name.is_some()
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
pub fn recommended(info: &HwInfo, use_gpu: bool) -> &'static str {
    if use_gpu {
        if let Some(vram) = info.vram_gb {
            if vram >= 6.0 {
                return "large-v3-turbo";
            }
            if vram >= 3.0 {
                return "small";
            }
        }
    }
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

/// Rate a model for the machine. `use_gpu` = transcription runs on the GPU
/// (see `gpu_active`): ratings are then based on VRAM, not CPU speed.
pub fn rate(name_or_id: &str, info: &HwInfo, use_gpu: bool) -> Rating {
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

    // GPU: VRAM is the constraint; anything that fits runs fast.
    if use_gpu {
        if let Some(vram) = info.vram_gb {
            let gpu = info.gpu_name.as_deref().unwrap_or("GPU");
            if spec.ram_gb <= (vram - 0.5).max(0.0) {
                let is_reco = size_key(recommended(info, true)) == Some(key);
                return if is_reco {
                    Rating {
                        level: Level::Ideal,
                        label: "\u{2605} Ideal pour ta machine".into(),
                        detail: format!("rapide sur {gpu}"),
                        color: "#5eaaff",
                    }
                } else {
                    Rating {
                        level: Level::Good,
                        label: "Recommande".into(),
                        detail: format!("rapide sur {gpu}"),
                        color: "#5ad28c",
                    }
                };
            }
            // Does not fit in VRAM: rated as CPU, annotated.
            let mut r = rate_cpu(key, &spec, info);
            if r.level != Level::TooBig {
                r.detail = format!("{} \u{2014} VRAM insuffisante ({vram:.0} Go)", r.detail);
            }
            return r;
        }
    }
    rate_cpu(key, &spec, info)
}

/// CPU rating: RAM is a hard constraint, speed a soft constraint.
fn rate_cpu(key: &'static str, spec: &ModelInfo, info: &HwInfo) -> Rating {
    let reco = recommended(info, false);
    let is_reco = size_key(reco) == Some(key);

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
    match (&info.gpu_name, info.vram_gb) {
        (Some(g), Some(v)) => parts.push(format!("{g} ({v:.0} Go)")),
        (Some(g), None) => parts.push(g.clone()),
        _ => parts.push("CPU".to_string()),
    }
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
            gpu_name: None,
            vram_gb: None,
        }
    }

    fn with_gpu(mut m: HwInfo, name: &str, vram: f32) -> HwInfo {
        m.gpu_name = Some(name.into());
        m.vram_gb = Some(vram);
        m
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
        assert_eq!(recommended(&machine(2.0, 4), false), "tiny"); // low RAM
        assert_eq!(recommended(&machine(8.0, 4), false), "base");
        assert_eq!(recommended(&machine(16.0, 12), false), "small");
        assert_eq!(recommended(&machine(4.0, 2), false), "tiny"); // 2 cores
        // GPU: VRAM drives the tiers.
        let g12 = with_gpu(machine(16.0, 8), "RX 7700 XT", 12.0);
        assert_eq!(recommended(&g12, true), "large-v3-turbo");
        assert_eq!(recommended(&g12, false), "small"); // forced CPU
        let g4 = with_gpu(machine(8.0, 4), "GTX 970", 4.0);
        assert_eq!(recommended(&g4, true), "small");
    }

    #[test]
    fn rate_too_big_and_ideal() {
        let m8 = machine(8.0, 4); // reco = base
        assert_eq!(rate("base", &m8, false).level, Level::Ideal);
        let m4 = machine(4.0, 4); // reco = tiny
        assert_eq!(rate("tiny", &m4, false).level, Level::Ideal);
        assert_eq!(rate("large-v3", &m4, false).level, Level::TooBig); // 5 GB > 3 GB available
        assert_eq!(rate("inconnu", &m4, false).level, Level::Unknown);
    }

    #[test]
    fn rate_gpu() {
        let g12 = with_gpu(machine(16.0, 8), "RX 7700 XT", 12.0);
        let r = rate("large-v3-turbo", &g12, true);
        assert_eq!(r.level, Level::Ideal);
        assert!(r.detail.contains("RX 7700 XT"), "{}", r.detail);
        assert_eq!(rate("medium", &g12, true).level, Level::Good);
        // Model bigger than VRAM: falls back to the CPU rating, annotated.
        let g2 = with_gpu(machine(16.0, 8), "petit GPU", 2.0);
        let r = rate("large-v3", &g2, true);
        assert_ne!(r.level, Level::Ideal);
        assert!(r.detail.contains("VRAM insuffisante"), "{}", r.detail);
        // No GPU detected: same as CPU even with use_gpu.
        assert_eq!(rate("base", &machine(8.0, 4), true).level, Level::Ideal);
    }
}
