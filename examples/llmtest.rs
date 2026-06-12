//! Test of the local LLM client against LM Studio/Ollama (OpenAI-compatible).
//! `cargo run --example llmtest`  (requires a local server + a loaded model)
//!
//! Note: the 1st call on a large model loaded on demand (JIT) can be
//! slow; a large timeout is used here for the test.

use dictata::config::{Config, LlmConfig};
use dictata::{llm, modes};

fn first_model_id(cfg: &LlmConfig) -> Option<String> {
    let url = format!("{}/models", cfg.base_url.trim_end_matches('/'));
    let resp = reqwest::blocking::Client::new()
        .get(url)
        .bearer_auth(&cfg.api_key)
        .send()
        .ok()?;
    let v: serde_json::Value = resp.json().ok()?;
    v["data"][0]["id"].as_str().map(|s| s.to_string())
}

fn main() {
    let mut cfg = LlmConfig::default();
    cfg.enabled = true;
    cfg.timeout = 300; // tolerate JIT loading of a large model

    println!("base_url = {}", cfg.base_url);
    let avail = llm::is_available(&cfg);
    println!("is_available = {avail}");
    if !avail {
        eprintln!("Aucun serveur LLM local : test transform ignore.");
        return;
    }

    match first_model_id(&cfg) {
        Some(id) => {
            println!("modele detecte : {id}");
            cfg.model = id;
        }
        None => {
            eprintln!("Aucun modele expose par /models — transform ignore.");
            return;
        }
    }

    let mut full = Config::default();
    full.llm = cfg.clone();
    let clean = full.modes.get("clean").unwrap().clone();
    let raw = "euh donc voila je voulais juste euh dire que le projet avance bien hum";

    let (out, status) = modes::apply_mode(raw, &clean, &full);
    println!("status = {status}");
    println!("--- entree ---\n{raw}\n--- sortie LLM ---\n{out}\n--------------");

    if status == "llm" && !out.is_empty() {
        println!("OK LLM (reformatage effectue)");
    } else {
        eprintln!("ECHEC: statut={status} (attendu 'llm' avec sortie non vide)");
        std::process::exit(1);
    }
}
