//! Optional reformatting by a *local* LLM (OpenAI-compatible API).
//!
//! Targets LM Studio (http://localhost:1234/v1) or Ollama. No cloud calls:
//! the endpoint is local and configurable. Disabled by default. `base_url`
//! already includes the `/v1` suffix (we append `/models` and `/chat/completions`).

use crate::config::LlmConfig;
use serde_json::json;
use std::time::Duration;

fn http(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs.max(1)))
        .build()
        .expect("client http")
}

fn base(cfg: &LlmConfig) -> String {
    cfg.base_url.trim_end_matches('/').to_string()
}

/// Quickly test whether the local LLM server responds (GET /models).
pub fn is_available(cfg: &LlmConfig) -> bool {
    let url = format!("{}/models", base(cfg));
    let req = http(3).get(url).bearer_auth(&cfg.api_key);
    matches!(req.send(), Ok(r) if r.status().is_success())
}

/// Reformat `text` according to `system_prompt`. Returns the LLM text (trimmed).
pub fn transform(text: &str, system_prompt: &str, cfg: &LlmConfig) -> Result<String, String> {
    let url = format!("{}/chat/completions", base(cfg));
    let model = if cfg.model.is_empty() {
        "local-model"
    } else {
        cfg.model.as_str()
    };
    let body = json!({
        "model": model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    });
    let resp = http(cfg.timeout)
        .post(url)
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .map_err(|e| format!("requete LLM: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("LLM HTTP {}", resp.status()));
    }
    let v: serde_json::Value = resp.json().map_err(|e| format!("reponse LLM: {e}"))?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();
    Ok(content)
}
