pub mod commands;
pub mod prompt;

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// LLM settings persisted in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    /// "local" (Ollama) or "cloud" (Gemini)
    pub mode: String,
    /// Ollama model tag for local mode (e.g. "qwen3.5:9b")
    pub ollama_model: Option<String>,
    /// Ollama server base URL (default: "http://localhost:11434")
    pub ollama_url: Option<String>,
    /// API key for cloud provider (Gemini)
    pub api_key: Option<String>,
    /// Cloud model name (default: "gemini-2.5-flash")
    pub cloud_model: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            mode: "local".to_string(),
            ollama_model: Some("qwen3.5:9b".to_string()),
            ollama_url: Some("http://localhost:11434".to_string()),
            api_key: None,
            cloud_model: Some("gemini-2.5-flash".to_string()),
        }
    }
}

/// Tauri-managed state holding the current LLM settings.
pub struct LlmState(pub Mutex<LlmSettings>);

/// Generate text using the Gemini REST API.
pub async fn generate_cloud(
    prompt: &str,
    api_key: &str,
    model: &str,
) -> Result<String, String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let body = serde_json::json!({
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        }
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = response.status();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    if !status.is_success() {
        return Err(format!("API error ({}): {}", status, response_text));
    }

    // Parse the Gemini response
    let json: serde_json::Value = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    // Extract generated text from response
    let text = json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| format!("Unexpected API response structure: {}", response_text))?;

    Ok(text.to_string())
}

/// Generate text using a local Ollama server (e.g. the `qwen3.5:9b` model).
///
/// Requires `ollama serve` to be running (the Ollama desktop app starts it
/// automatically) and the given model to be pulled (`ollama pull <model>`).
pub async fn generate_ollama(
    prompt: &str,
    model: &str,
    base_url: &str,
) -> Result<String, String> {
    generate_ollama_opts(prompt, model, base_url, 0.7, 8192).await
}

/// Like `generate_ollama`, but with explicit sampling `temperature` and
/// `num_predict` (max output tokens). Use a low temperature (e.g. 0.0) for
/// deterministic, judge-style tasks where a single short answer is expected.
pub async fn generate_ollama_opts(
    prompt: &str,
    model: &str,
    base_url: &str,
    temperature: f32,
    num_predict: i32,
) -> Result<String, String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));

    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        // Disable chain-of-thought for reasoning models (e.g. qwen3.5) so the
        // answer lands in `response` rather than the separate `thinking` field.
        // Harmless for non-reasoning models.
        "think": false,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        }
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            format!(
                "Could not reach Ollama at {}. Is the Ollama server running? ({})",
                url, e
            )
        })?;

    let status = response.status();
    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Ollama response: {}", e))?;

    if !status.is_success() {
        return Err(format!("Ollama API error ({}): {}", status, response_text));
    }

    let json: serde_json::Value = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse Ollama JSON: {}", e))?;

    json["response"]
        .as_str()
        .map(|s| s.trim().to_string())
        .ok_or_else(|| format!("Unexpected Ollama response structure: {}", response_text))
}

/// Load LLM settings from the database, or return defaults.
pub fn load_settings(conn: &rusqlite::Connection) -> LlmSettings {
    let result = conn.query_row(
        "SELECT value FROM app_settings WHERE key = 'llm_settings'",
        [],
        |row| row.get::<_, String>(0),
    );

    match result {
        Ok(json_str) => serde_json::from_str(&json_str).unwrap_or_default(),
        Err(_) => LlmSettings::default(),
    }
}

/// Persist LLM settings to the database.
pub fn save_settings(
    conn: &rusqlite::Connection,
    settings: &LlmSettings,
) -> Result<(), String> {
    let json = serde_json::to_string(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;

    conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('llm_settings', ?1)",
        [&json],
    )
    .map_err(|e| format!("Failed to save settings: {}", e))?;

    Ok(())
}
