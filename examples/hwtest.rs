//! Hardware detection + recommendation + catalog + download test.
//! `cargo run --example hwtest`

use dictata::{hardware, models};

fn main() {
    let info = hardware::detect();
    println!("CPU       : {}", info.cpu_name);
    println!("Coeurs    : {} physiques / {} logiques", info.cpu_physical, info.cpu_logical);
    println!("RAM       : {:?} Go (dispo {:?})", info.ram_total_gb, info.ram_avail_gb);
    println!("Resume    : {}", hardware::summary(&info));
    println!("GPU       : {:?} ({:?} Go)", info.gpu_name, info.vram_gb);
    let gpu = hardware::gpu_active(&info, "auto");
    println!("Recommande: {} (gpu_active={gpu})", hardware::recommended(&info, gpu));
    println!();

    println!("Catalogue (note pour cette machine) :");
    for c in models::CATALOG {
        let r = hardware::rate(c.name, &info, gpu);
        println!(
            "  {:24} {:>5} Mo  [{:?}] {} \u{2014} {}",
            c.name, c.size_mb, r.level, r.label, r.detail
        );
    }

    let dir = format!(
        "{}\\models",
        std::env::var("DICTATA_HOME").unwrap_or_else(|_| ".".into())
    );
    println!("\nInstalles dans {dir} : {:?}", models::list_installed(&dir));

    // Real download test: tiny-q5_1 (~31 MB) if missing.
    let target = "tiny-q5_1";
    if models::is_installed(&dir, target) {
        println!("{target} deja present, telechargement ignore.");
    } else {
        println!("Telechargement de {target}…");
        let res = models::download(&dir, target, |recu, total| {
            if let Some(t) = total {
                let pct = recu as f64 / t as f64 * 100.0;
                if recu % (4 * 1024 * 1024) < 65536 {
                    print!("\r  {:.0}% ({:.1}/{:.1} Mo)   ", pct, recu as f64 / 1e6, t as f64 / 1e6);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }
        });
        match res {
            Ok(p) => {
                let sz = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                println!("\nOK telechargement : {} ({} octets)", p.display(), sz);
            }
            Err(e) => {
                eprintln!("\nECHEC telechargement : {e}");
                std::process::exit(1);
            }
        }
    }
}
