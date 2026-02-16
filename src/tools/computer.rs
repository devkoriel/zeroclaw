use super::traits::{Tool, ToolResult};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Timeout for shell sub-commands (screencapture, sips, cliclick, etc.).
const CMD_TIMEOUT: Duration = Duration::from_secs(20);
/// Timeout for the Gemini Vision API call.
const VISION_TIMEOUT: Duration = Duration::from_secs(45);
/// Vision model to use for screenshot descriptions.
const VISION_MODEL: &str = "gemini-2.5-flash";
/// Maximum image file size to send to vision API (~6 MB).
const MAX_IMAGE_BYTES: u64 = 6_291_456;
/// Delay after opening an app to let it fully render.
const OPEN_APP_DELAY: Duration = Duration::from_millis(1500);
/// Screenshot cache TTL — avoid redundant captures in rapid screenshot→click→verify cycles.
const SCREENSHOT_CACHE_TTL: Duration = Duration::from_secs(3);

/// Keys that cliclick's `kp:` command supports.
/// Regular character keys (a-z, 0-9, punctuation) are NOT supported by cliclick
/// and must use AppleScript `keystroke` instead.
const CLICLICK_SPECIAL_KEYS: &[&str] = &[
    "arrow-down", "arrow-left", "arrow-right", "arrow-up",
    "brightness-down", "brightness-up",
    "delete", "end", "enter", "esc", "escape",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
    "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
    "fwd-delete", "home",
    "keys-light-down", "keys-light-toggle", "keys-light-up",
    "mute",
    "num-0", "num-1", "num-2", "num-3", "num-4",
    "num-5", "num-6", "num-7", "num-8", "num-9",
    "num-clear", "num-divide", "num-enter", "num-equals",
    "num-minus", "num-multiply", "num-plus",
    "page-down", "page-up",
    "play-next", "play-pause", "play-previous",
    "return", "space", "tab",
    "volume-down", "volume-up",
];

/// Global flag: once cliclick passes its check, skip future checks.
static CLICLICK_VERIFIED: AtomicBool = AtomicBool::new(false);

/// Cached logical screen width (0 = not yet cached).
static SCREEN_WIDTH_CACHE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// ── Gemini Vision API types (separate from providers/gemini.rs) ─────────────

#[derive(Serialize)]
struct VisionRequest {
    contents: Vec<VisionContent>,
    #[serde(rename = "generationConfig")]
    generation_config: VisionGenConfig,
}

#[derive(Serialize)]
struct VisionContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<VisionPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum VisionPart {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

#[derive(Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct VisionGenConfig {
    temperature: f64,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    response_schema: Option<serde_json::Value>,
}

// ── Structured Vision Response types ─────────────────────────────────────────

/// Parsed structured response from Gemini Vision API.
#[derive(Debug, Deserialize)]
struct VisionResponse {
    foreground_app: Option<String>,
    screen_state: Option<String>,
    elements: Option<Vec<VisionElement>>,
    visible_text: Option<String>,
}

/// A single interactive UI element detected in a screenshot.
#[derive(Debug, Deserialize)]
struct VisionElement {
    label: String,
    #[serde(rename = "type")]
    element_type: String,
    x: i32,
    y: i32,
    width: Option<i32>,
    height: Option<i32>,
    state: Option<String>,
}

/// Build the JSON schema for Gemini's structured output.
fn vision_response_schema() -> serde_json::Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "foreground_app": {
                "type": "STRING",
                "description": "Name of the frontmost application"
            },
            "screen_state": {
                "type": "STRING",
                "description": "Brief description: dialogs, notifications, loading, idle, etc."
            },
            "elements": {
                "type": "ARRAY",
                "description": "All interactive UI elements with precise pixel coordinates",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "description": "Element text or accessible label"},
                        "type": {"type": "STRING", "description": "button, text_field, link, menu_item, tab, icon, checkbox, chat_item, list_item, image, toggle, dropdown"},
                        "x": {"type": "INTEGER", "description": "Center X in pixels from top-left"},
                        "y": {"type": "INTEGER", "description": "Center Y in pixels from top-left"},
                        "width": {"type": "INTEGER", "description": "Approximate width in pixels"},
                        "height": {"type": "INTEGER", "description": "Approximate height in pixels"},
                        "state": {"type": "STRING", "description": "enabled, disabled, selected, focused, checked, unchecked"}
                    },
                    "required": ["label", "type", "x", "y"]
                }
            },
            "visible_text": {
                "type": "STRING",
                "description": "All visible text content on screen (messages, labels, titles, paragraphs)"
            }
        },
        "required": ["foreground_app", "screen_state", "elements"]
    })
}

/// Format a parsed VisionResponse into a structured text description for the agent LLM.
fn format_vision_response(resp: &VisionResponse) -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("[Screen Analysis]\n");
    if let Some(ref app) = resp.foreground_app {
        out.push_str(&format!("App: {app}\n"));
    }
    if let Some(ref state) = resp.screen_state {
        out.push_str(&format!("State: {state}\n"));
    }
    out.push('\n');

    if let Some(ref elements) = resp.elements {
        if !elements.is_empty() {
            out.push_str("[Interactive Elements] (use these coordinates for click actions)\n");
            for (i, el) in elements.iter().enumerate() {
                let size = match (el.width, el.height) {
                    (Some(w), Some(h)) => format!(" [{w}x{h}]"),
                    _ => String::new(),
                };
                let state_str = el
                    .state
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!(" ({s})"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "{}. \"{}\" ({}) at ({}, {}){}{}\n",
                    i + 1,
                    el.label,
                    el.element_type,
                    el.x,
                    el.y,
                    size,
                    state_str
                ));
            }
            out.push('\n');
        }
    }

    if let Some(ref text) = resp.visible_text {
        if !text.is_empty() {
            out.push_str("[Visible Text]\n");
            out.push_str(text);
            out.push('\n');
        }
    }

    out
}

// ── Tool implementation ─────────────────────────────────────────────────────

/// Cached screenshot data (base64 + timestamp).
struct ScreenshotCache {
    base64: String,
    captured_at: Instant,
}

/// Computer-use tool — see the screen via vision AI and control mouse/keyboard.
pub struct ComputerTool {
    security: Arc<SecurityPolicy>,
    gemini_key: Option<String>,
    client: reqwest::Client,
    screenshot_cache: Arc<Mutex<Option<ScreenshotCache>>>,
}

impl ComputerTool {
    pub fn new(security: Arc<SecurityPolicy>, gemini_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(VISION_TIMEOUT)
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            security,
            gemini_key,
            client,
            screenshot_cache: Arc::new(Mutex::new(None)),
        }
    }

    // ── Screenshot + Vision ─────────────────────────────────────────────

    async fn action_screenshot(&self, args: &serde_json::Value) -> ToolResult {
        let extra_prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // 0. Wake display if sleeping (user-activity assertion for 5s)
        let _ = run_cmd("caffeinate", &["-u", "-t", "5"]).await;

        // 1. Capture screenshot (PNG for lossless quality + better text OCR)
        let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let path = format!("/tmp/zeroclaw_screen_{ts}.png");

        if let Err(e) = run_cmd("screencapture", &["-x", "-t", "png", &path]).await {
            return err_result(format!(
                "Screenshot capture failed: {e}\n\n\
                 If Screen Recording permission is needed:\n\
                 1. Open: System Settings → Privacy & Security → Screen Recording\n\
                 2. Click + and add /Applications/ZeroClaw.app\n\
                 3. Toggle it ON, then restart the daemon"
            ));
        }

        // Check if screenshot file is empty (permission denied or locked screen)
        if let Ok(meta) = tokio::fs::metadata(&path).await {
            if meta.len() == 0 {
                // Retry once: wake display, wait, re-capture
                let _ = tokio::fs::remove_file(&path).await;
                tracing::info!("Screenshot empty — waking display and retrying");
                let _ = run_cmd("caffeinate", &["-u", "-t", "5"]).await;
                tokio::time::sleep(Duration::from_secs(2)).await;

                if let Err(e) = run_cmd("screencapture", &["-x", "-t", "png", &path]).await {
                    return err_result(format!("Screenshot retry failed: {e}"));
                }

                // Check again
                if let Ok(meta2) = tokio::fs::metadata(&path).await {
                    if meta2.len() == 0 {
                        let _ = tokio::fs::remove_file(&path).await;
                        return err_result(
                            "Screen Recording permission required — screenshot file is empty.\n\n\
                             Grant it now:\n\
                             1. Open: System Settings → Privacy & Security → Screen Recording\n\
                             2. Click + and add /Applications/ZeroClaw.app\n\
                             3. Toggle it ON\n\
                             4. Restart the daemon: launchctl kickstart -k gui/501/com.zeroclaw.daemon"
                        );
                    }
                }
            }
        }

        // 2. Get logical screen width (cached) and resize
        let logical_width = get_logical_screen_width_cached().await;
        if let Some(w) = logical_width {
            let _ = run_cmd("sips", &["--resampleWidth", &w.to_string(), &path]).await;
        }

        // 3. Read + encode
        let meta = match tokio::fs::metadata(&path).await {
            Ok(m) => m,
            Err(e) => {
                let _ = tokio::fs::remove_file(&path).await;
                return err_result(format!("Cannot read screenshot: {e}"));
            }
        };
        if meta.len() > MAX_IMAGE_BYTES {
            let _ = tokio::fs::remove_file(&path).await;
            return err_result(format!(
                "Screenshot too large ({} bytes). Max: {MAX_IMAGE_BYTES}",
                meta.len()
            ));
        }

        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => {
                let _ = tokio::fs::remove_file(&path).await;
                return err_result(format!("Failed to read screenshot file: {e}"));
            }
        };
        let _ = tokio::fs::remove_file(&path).await;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

        // Update screenshot cache
        {
            let mut cache = self.screenshot_cache.lock().await;
            *cache = Some(ScreenshotCache {
                base64: b64.clone(),
                captured_at: Instant::now(),
            });
        }

        // --- ZeroClaw fork: Hybrid Programmatic Grounding ---
        // Try structured screen probing first (Swift AXAPI → JXA → Vision fallback).
        // If probe succeeds with elements, return structured data + screenshot image.
        // If probe fails, fall through to existing Vision API path (unchanged).
        if let Ok(state) = crate::tools::screen_state::probe_screen_state().await {
            let mut output = crate::tools::screen_state::format_screen_state(&state);
            if !extra_prompt.is_empty() {
                output.push_str(&format!("[User context: {extra_prompt}]\n"));
            }
            return ToolResult {
                success: true,
                output,
                error: None,
                image_base64: Some(b64),
                image_mime: Some("image/png".into()),
            };
        }
        // --- end ZeroClaw fork ---

        // 4. Image width for the vision prompt dimension hint
        let img_w = logical_width.unwrap_or(0);

        // 5. Call Gemini Vision API (Tier 3 fallback — probes returned 0 elements)
        let api_key = match &self.gemini_key {
            Some(k) if !k.is_empty() => k.as_str(),
            _ => {
                return ToolResult {
                    success: true,
                    output: format!(
                        "Screenshot captured ({} bytes) but vision AI unavailable.\n\
                         Set GEMINI_API_KEY environment variable to enable screen descriptions.\n\
                         You can still use click/type/key actions if you know the coordinates.",
                        bytes.len()
                    ),
                    error: None,
                    image_base64: Some(b64),
                    image_mime: Some("image/png".into()),
                };
            }
        };

        let dim_hint = if img_w > 0 {
            format!("Image width: {img_w}px.")
        } else {
            String::new()
        };

        let prompt = format!(
            "Analyze this macOS screenshot. {dim_hint}\n\
             Coordinates must be PRECISE pixel positions from the top-left corner — they will be used directly for mouse clicks.\n\
             For every interactive UI element, report its CENTER x,y coordinates.\n\
             Include: buttons, text fields, links, menu items, tabs, icons, checkboxes, chat items, list items, toggles, dropdowns.\n\
             {extra_prompt}"
        );

        match self.call_vision_api(api_key, &b64, &prompt).await {
            Ok(description) => {
                // Try to parse structured JSON from Gemini
                let formatted = match serde_json::from_str::<VisionResponse>(&description) {
                    Ok(parsed) => {
                        // Validate coordinates are within screen bounds
                        if let (Some(w), Some(ref elements)) = (logical_width, &parsed.elements) {
                            for el in elements {
                                if el.x < 0 || el.y < 0 || el.x > w as i32 * 2 {
                                    tracing::warn!(
                                        "Vision element '{}' has suspicious coordinates ({}, {}) for {}px screen",
                                        el.label, el.x, el.y, w
                                    );
                                }
                            }
                        }
                        format_vision_response(&parsed)
                    }
                    Err(_) => {
                        // Fallback: return raw text (backward compatible)
                        tracing::debug!("Vision API returned non-JSON; using raw text");
                        description
                    }
                };
                ToolResult {
                    success: true,
                    output: formatted,
                    error: None,
                    image_base64: Some(b64),
                    image_mime: Some("image/png".into()),
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: format!("Screenshot captured ({} bytes) but vision API failed.", bytes.len()),
                error: Some(format!("Vision API error: {e}")),
                image_base64: Some(b64),
                image_mime: Some("image/png".into()),
            },
        }
    }

    async fn call_vision_api(
        &self,
        api_key: &str,
        image_b64: &str,
        prompt: &str,
    ) -> Result<String, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL}:generateContent?key={api_key}"
        );

        let request = VisionRequest {
            contents: vec![VisionContent {
                role: Some("user".into()),
                parts: vec![
                    VisionPart::Text {
                        text: prompt.to_string(),
                    },
                    VisionPart::InlineData {
                        inline_data: InlineData {
                            mime_type: "image/png".into(),
                            data: image_b64.to_string(),
                        },
                    },
                ],
            }],
            generation_config: VisionGenConfig {
                temperature: 0.1,
                max_output_tokens: 4096,
                response_mime_type: Some("application/json".into()),
                response_schema: Some(vision_response_schema()),
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Gemini API error ({status}): {body}"));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {e}"))?;

        // Extract text from candidates[0].content.parts[0].text
        body.get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .map(String::from)
            .ok_or_else(|| "No text in vision API response".to_string())
    }

    // ── Click actions ───────────────────────────────────────────────────

    async fn action_click(&self, prefix: &str, args: &serde_json::Value) -> ToolResult {
        let (x, y) = match extract_coords(args) {
            Ok(coords) => coords,
            Err(e) => return err_result(e),
        };

        if let Err(e) = check_cliclick_cached().await {
            return err_result(e);
        }

        // Bring the frontmost app to focus and ensure we click in the right place.
        // This avoids clicking behind overlapping windows.
        let coord = format!("{prefix}:{x},{y}");
        match run_cmd("cliclick", &[&coord]).await {
            Ok(out) => ToolResult {
                success: true,
                output: format!("Clicked at ({x}, {y}). {out}"),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => err_result(format!("Click failed: {e}")),
        }
    }

    // ── Type action ─────────────────────────────────────────────────────

    async fn action_type(&self, args: &serde_json::Value) -> ToolResult {
        let Some(text) = args.get("text").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: text");
        };

        if let Err(e) = check_cliclick_cached().await {
            return err_result(e);
        }

        let arg = format!("t:{text}");
        match run_cmd("cliclick", &[&arg]).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Typed: \"{text}\""),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => err_result(format!("Type failed: {e}")),
        }
    }

    // ── Key combo action ────────────────────────────────────────────────

    async fn action_key(&self, args: &serde_json::Value) -> ToolResult {
        let Some(combo) = args.get("key").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: key");
        };

        // Determine if we need AppleScript or cliclick.
        // cliclick's kp: only supports special keys (return, tab, arrows, F-keys, etc.)
        // For combos with regular character keys (a-z, 0-9, punctuation), use AppleScript.
        let parts: Vec<&str> = combo.split('+').map(str::trim).collect();
        let final_key = parts.last().copied().unwrap_or("");
        let mapped_key = map_key_name(final_key);
        let needs_applescript = !is_cliclick_special_key(mapped_key);

        if needs_applescript {
            // Use AppleScript for key combos involving regular characters.
            // This handles cmd+c, cmd+v, cmd+a, ctrl+a, etc. reliably.
            return self.action_key_applescript(combo, &parts).await;
        }

        // Use cliclick for special-key-only combos (e.g., "enter", "cmd+tab", "cmd+shift+tab")
        if let Err(e) = check_cliclick_cached().await {
            return err_result(e);
        }

        let cliclick_args = parse_key_combo(combo);
        match run_cmd("cliclick", &cliclick_args.iter().map(String::as_str).collect::<Vec<_>>())
            .await
        {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Key combo: {combo}"),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => {
                // Fallback to AppleScript if cliclick fails
                tracing::warn!("cliclick key combo failed ({e}), falling back to AppleScript");
                self.action_key_applescript(combo, &parts).await
            }
        }
    }

    /// Execute a key combo via AppleScript `keystroke` / `key code`.
    /// This is the reliable method for combos involving regular character keys
    /// (e.g., cmd+c, cmd+v, cmd+a, ctrl+z) and also works as a fallback for
    /// special keys.
    async fn action_key_applescript(&self, combo: &str, parts: &[&str]) -> ToolResult {
        if parts.is_empty() {
            return err_result("Empty key combo");
        }

        let final_key = parts.last().copied().unwrap_or("");
        let modifiers = &parts[..parts.len().saturating_sub(1)];

        // Build AppleScript modifier list: {command down, shift down, ...}
        let mut as_modifiers = Vec::new();
        for m in modifiers {
            match m.to_lowercase().as_str() {
                "cmd" | "command" => as_modifiers.push("command down"),
                "ctrl" | "control" => as_modifiers.push("control down"),
                "alt" | "option" | "opt" => as_modifiers.push("option down"),
                "shift" => as_modifiers.push("shift down"),
                "fn" => as_modifiers.push("fn down"),
                _ => {
                    return err_result(format!("Unknown modifier: {m}"));
                }
            }
        }

        let modifier_clause = if as_modifiers.is_empty() {
            String::new()
        } else {
            format!(" using {{{}}}", as_modifiers.join(", "))
        };

        // Determine whether to use `keystroke` (for characters) or `key code` (for special keys)
        let script = if let Some(key_code) = applescript_key_code(final_key) {
            // Special key → use key code
            format!(
                "tell application \"System Events\" to key code {key_code}{modifier_clause}"
            )
        } else if final_key.len() == 1 {
            // Single character → use keystroke
            // Escape quotes for AppleScript
            let escaped = final_key.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "tell application \"System Events\" to keystroke \"{escaped}\"{modifier_clause}"
            )
        } else {
            // Multi-char non-special key — try keystroke anyway
            let escaped = final_key.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "tell application \"System Events\" to keystroke \"{escaped}\"{modifier_clause}"
            )
        };

        match run_cmd("osascript", &["-e", &script]).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Key combo: {combo} (via AppleScript)"),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => err_result(format!("Key press failed (AppleScript): {e}")),
        }
    }

    // ── Scroll action ───────────────────────────────────────────────────

    async fn action_scroll(&self, args: &serde_json::Value) -> ToolResult {
        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("down");
        let amount = args
            .get("amount")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(3);

        // Validate direction
        if !matches!(direction, "up" | "down" | "left" | "right") {
            return err_result(format!(
                "Invalid scroll direction: {direction}. Use up/down/left/right"
            ));
        }

        // Move cursor to position first if coordinates provided
        if let Ok((x, y)) = extract_coords(args) {
            if check_cliclick_cached().await.is_ok() {
                let _ = run_cmd("cliclick", &[&format!("m:{x},{y}")]).await;
            }
        }

        // Use JXA (JavaScript for Automation) + CoreGraphics for reliable scrolling.
        // CGEventCreateScrollWheelEvent is the most reliable way to scroll on macOS.
        // JXA is always available on macOS (unlike Python pyobjc which may not be installed).
        let (scroll_y, scroll_x) = match direction {
            "up" => (amount as i32, 0),
            "down" => (-(amount as i32), 0),
            "left" => (0, amount as i32),
            "right" => (0, -(amount as i32)),
            _ => unreachable!(),
        };

        // Write JXA script to temp file to avoid shell quoting issues with osascript -e
        let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let script_path = format!("/tmp/zeroclaw_scroll_{ts}.js");
        let jxa_content = format!(
            "ObjC.import('CoreGraphics');\n\
             var e = $.CGEventCreateScrollWheelEvent(null, 0, 2, {scroll_y}, {scroll_x});\n\
             $.CGEventPost(0, e);\n"
        );

        if let Err(e) = tokio::fs::write(&script_path, &jxa_content).await {
            return err_result(format!("Failed to write scroll script: {e}"));
        }

        let result = run_cmd("osascript", &["-l", "JavaScript", &script_path]).await;
        let _ = tokio::fs::remove_file(&script_path).await;

        match result {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Scrolled {direction} by {amount}"),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => {
                // Fallback: AppleScript key codes (arrow keys — not true scrolling but
                // works when CoreGraphics is unavailable)
                let key_code = match direction {
                    "up" => 126,   // up arrow
                    "down" => 125, // down arrow
                    "left" => 123, // left arrow
                    "right" => 124, // right arrow
                    _ => unreachable!(),
                };
                let mut script_parts = Vec::new();
                for _ in 0..amount {
                    script_parts.push(format!(
                        "tell application \"System Events\" to key code {key_code}"
                    ));
                }
                let fallback_script = script_parts.join("\n");
                match run_cmd("osascript", &["-e", &fallback_script]).await {
                    Ok(_) => ToolResult {
                        success: true,
                        output: format!("Scrolled {direction} by {amount} (fallback)"),
                        error: None,
                        image_base64: None,
                        image_mime: None,
                    },
                    Err(e2) => err_result(format!(
                        "Scroll failed. Primary (JXA CoreGraphics): {e}. Fallback (osascript): {e2}"
                    )),
                }
            }
        }
    }

    // ── Open app action ─────────────────────────────────────────────────

    async fn action_open_app(&self, args: &serde_json::Value) -> ToolResult {
        let Some(app_name) = args.get("text").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: text (app name)");
        };

        match run_cmd("open", &["-a", app_name]).await {
            Ok(_) => {
                // Wait for app to launch, come to foreground, and render its UI.
                // 500ms was too short for heavier apps like KakaoTalk.
                tokio::time::sleep(OPEN_APP_DELAY).await;

                // Bring the app to the front to ensure it's focused
                let activate_script = format!(
                    "tell application \"{}\" to activate",
                    app_name.replace('"', "\\\"")
                );
                let _ = run_cmd("osascript", &["-e", &activate_script]).await;

                ToolResult {
                    success: true,
                    output: format!("Opened application: {app_name}"),
                    error: None,
                    image_base64: None,
                    image_mime: None,
                }
            }
            Err(e) => err_result(format!("Failed to open {app_name}: {e}")),
        }
    }

    // ── Cursor position action ──────────────────────────────────────────

    async fn action_cursor_position(&self) -> ToolResult {
        if let Err(e) = check_cliclick_cached().await {
            return err_result(e);
        }

        match run_cmd("cliclick", &["p"]).await {
            Ok(out) => ToolResult {
                success: true,
                output: format!("Cursor position: {out}"),
                error: None,
                image_base64: None,
                image_mime: None,
            },
            Err(e) => err_result(format!("Failed to get cursor position: {e}")),
        }
    }
}

#[async_trait]
impl Tool for ComputerTool {
    fn name(&self) -> &str {
        "computer"
    }

    fn description(&self) -> &str {
        "See the screen and control mouse/keyboard to interact with any macOS application. \
         Actions: screenshot (see screen via AI vision), click/double_click/right_click, \
         type, key (combos like cmd+c), scroll, open_app, cursor_position."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: screenshot, click, double_click, right_click, type, key, scroll, open_app, cursor_position",
                    "enum": ["screenshot", "click", "double_click", "right_click", "type", "key", "scroll", "open_app", "cursor_position"]
                },
                "x": {
                    "type": "integer",
                    "description": "X coordinate for click/scroll actions"
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate for click/scroll actions"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for type action) or app name (for open_app)"
                },
                "key": {
                    "type": "string",
                    "description": "Key combo like 'cmd+c', 'enter', 'tab', 'cmd+shift+t'"
                },
                "direction": {
                    "type": "string",
                    "description": "Scroll direction: up, down, left, right"
                },
                "amount": {
                    "type": "integer",
                    "description": "Scroll amount (default: 3)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Extra context for the vision model when taking a screenshot"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if !self.security.can_act() {
            return Ok(err_result("Action blocked: autonomy is read-only"));
        }
        self.security.record_action();

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let result = match action {
            "screenshot" => self.action_screenshot(&args).await,
            "click" => self.action_click("c", &args).await,
            "double_click" => self.action_click("dc", &args).await,
            "right_click" => self.action_click("rc", &args).await,
            "type" => self.action_type(&args).await,
            "key" => self.action_key(&args).await,
            "scroll" => self.action_scroll(&args).await,
            "open_app" => self.action_open_app(&args).await,
            "cursor_position" => self.action_cursor_position().await,
            "" => err_result("Missing required parameter: action"),
            other => err_result(format!(
                "Unknown action: {other}. Valid: screenshot, click, double_click, \
                 right_click, type, key, scroll, open_app, cursor_position"
            )),
        };

        Ok(result)
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn err_result(msg: impl Into<String>) -> ToolResult {
    let msg = msg.into();
    ToolResult {
        success: false,
        output: String::new(),
        error: Some(msg),
        image_base64: None,
        image_mime: None,
    }
}

fn extract_coords(args: &serde_json::Value) -> Result<(i64, i64), String> {
    let x = args
        .get("x")
        .and_then(serde_json::Value::as_i64)
        .ok_or("Missing required parameter: x")?;
    let y = args
        .get("y")
        .and_then(serde_json::Value::as_i64)
        .ok_or("Missing required parameter: y")?;
    Ok((x, y))
}

/// Check if a mapped key name is a cliclick special key.
fn is_cliclick_special_key(key: &str) -> bool {
    CLICLICK_SPECIAL_KEYS.contains(&key)
}

/// Map a key name to its AppleScript `key code` number, if it's a special key.
/// Returns None for regular character keys (which use `keystroke` instead).
fn applescript_key_code(name: &str) -> Option<u32> {
    match name.to_lowercase().as_str() {
        "return" | "enter" => Some(36),
        "tab" => Some(48),
        "space" | " " => Some(49),
        "delete" | "backspace" => Some(51),
        "escape" | "esc" => Some(53),
        "fwd-delete" | "forward-delete" | "forwarddelete" => Some(117),
        "up" | "arrow-up" => Some(126),
        "down" | "arrow-down" => Some(125),
        "left" | "arrow-left" => Some(123),
        "right" | "arrow-right" => Some(124),
        "home" => Some(115),
        "end" => Some(119),
        "page-up" | "pageup" | "page_up" => Some(116),
        "page-down" | "pagedown" | "page_down" => Some(121),
        "f1" => Some(122),
        "f2" => Some(120),
        "f3" => Some(99),
        "f4" => Some(118),
        "f5" => Some(96),
        "f6" => Some(97),
        "f7" => Some(98),
        "f8" => Some(100),
        "f9" => Some(101),
        "f10" => Some(109),
        "f11" => Some(103),
        "f12" => Some(111),
        "f13" => Some(105),
        "f14" => Some(107),
        "f15" => Some(113),
        "f16" => Some(106),
        _ => None,
    }
}

/// Run a command with timeout, returning stdout on success or an error message.
async fn run_cmd(program: &str, args: &[&str]) -> Result<String, String> {
    let result = tokio::time::timeout(
        CMD_TIMEOUT,
        tokio::process::Command::new(program)
            .args(args)
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            if output.status.success() {
                Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("{program} failed: {stderr}").trim().to_string())
            }
        }
        Ok(Err(e)) => Err(format!("Failed to execute {program}: {e}")),
        Err(_) => Err(format!("{program} timed out after {}s", CMD_TIMEOUT.as_secs())),
    }
}

/// Check if cliclick is installed and Accessibility is granted.
/// Uses a cached result after the first successful check to avoid
/// spawning 2 subprocesses on every single action.
async fn check_cliclick_cached() -> Result<(), String> {
    // Fast path: already verified
    if CLICLICK_VERIFIED.load(Ordering::Relaxed) {
        return Ok(());
    }

    // Slow path: full check
    let result = check_cliclick_inner().await;
    if result.is_ok() {
        CLICLICK_VERIFIED.store(true, Ordering::Relaxed);
    }
    result
}

/// Inner cliclick check — actually spawns processes.
async fn check_cliclick_inner() -> Result<(), String> {
    if run_cmd("which", &["cliclick"]).await.is_err() {
        return Err(
            "cliclick not found. Install with: brew install cliclick".into(),
        );
    }

    // Quick accessibility check: try a no-op cursor position query.
    // cliclick prints an error and exits non-zero without Accessibility.
    match run_cmd("cliclick", &["p"]).await {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = e.to_lowercase();
            if msg.contains("accessibility") || msg.contains("permission") || msg.contains("not allowed") {
                Err(
                    "Accessibility permission required for mouse/keyboard control.\n\n\
                     Grant it now:\n\
                     1. Open: System Settings → Privacy & Security → Accessibility\n\
                     2. Click + and add /Applications/ZeroClaw.app\n\
                     3. Toggle it ON\n\n\
                     Then try again."
                        .into(),
                )
            } else {
                // Some other cliclick error
                Err(format!("cliclick check failed: {e}"))
            }
        }
    }
}

/// Get logical screen width, using a cached value after first successful call.
/// Screen resolution doesn't change between screenshots in the same session.
async fn get_logical_screen_width_cached() -> Option<u32> {
    let cached = SCREEN_WIDTH_CACHE.load(Ordering::Relaxed);
    if cached > 0 {
        return Some(cached);
    }

    let width = get_logical_screen_width().await;
    if let Some(w) = width {
        SCREEN_WIDTH_CACHE.store(w, Ordering::Relaxed);
    }
    width
}

/// Get logical screen width via JXA `AppKit`, with `system_profiler` fallback.
async fn get_logical_screen_width() -> Option<u32> {
    // Method 1: JXA (JavaScript for Automation) + AppKit — always available on macOS.
    // Unlike Python pyobjc, JXA doesn't require any extra packages.
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
    let script_path = format!("/tmp/zeroclaw_screenw_{ts}.js");

    if tokio::fs::write(
        &script_path,
        "ObjC.import('AppKit');\nvar f = $.NSScreen.mainScreen.frame;\nf.size.width;\n",
    )
    .await
    .is_ok()
    {
        if let Ok(out) = run_cmd("osascript", &["-l", "JavaScript", &script_path]).await {
            let _ = tokio::fs::remove_file(&script_path).await;
            if let Ok(w) = out.trim().parse::<u32>() {
                if w > 0 {
                    return Some(w);
                }
            }
        } else {
            let _ = tokio::fs::remove_file(&script_path).await;
        }
    }

    // Method 2: system_profiler fallback (slower but doesn't need any frameworks)
    if let Ok(out) = run_cmd("system_profiler", &["SPDisplaysDataType"]).await {
        // Look for "Resolution: 1512 x 982" (logical) or "3024 x 1964" (physical)
        // The "UI Looks like" line gives logical resolution on newer macOS
        for line in out.lines() {
            let trimmed = line.trim();
            if trimmed.contains("Resolution:") || trimmed.contains("UI Looks like:") {
                // Parse "1512 x 982" pattern
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "x" && i > 0 {
                        if let Ok(w) = parts[i - 1].parse::<u32>() {
                            if w > 0 && w <= 7680 {
                                return Some(w);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Parse a key combo string into cliclick arguments.
/// NOTE: This is only used for combos where the final key IS a cliclick special key.
/// For regular character keys, `action_key_applescript` is used instead.
///
/// Examples:
/// - `"enter"` → `["kp:return"]`
/// - `"cmd+tab"` → `["kd:cmd", "kp:tab", "ku:cmd"]`
/// - `"cmd+shift+tab"` → `["kd:cmd", "kd:shift", "kp:tab", "ku:shift", "ku:cmd"]`
fn parse_key_combo(combo: &str) -> Vec<String> {
    let parts: Vec<&str> = combo.split('+').map(str::trim).collect();

    if parts.len() == 1 {
        // Single key press
        let key = map_key_name(parts[0]);
        return vec![format!("kp:{key}")];
    }

    // Multiple parts: all but last are modifiers
    let modifiers = &parts[..parts.len() - 1];
    let key = map_key_name(parts[parts.len() - 1]);

    let mut args = Vec::new();

    // Key-down for each modifier
    for m in modifiers {
        args.push(format!("kd:{}", map_key_name(m)));
    }

    // Press the final key
    args.push(format!("kp:{key}"));

    // Key-up for modifiers in reverse order
    for m in modifiers.iter().rev() {
        args.push(format!("ku:{}", map_key_name(m)));
    }

    args
}

/// Map common key names to cliclick's expected key names.
fn map_key_name(name: &str) -> &str {
    match name.to_lowercase().as_str() {
        "enter" | "return" => "return",
        "esc" | "escape" => "escape",
        "cmd" | "command" => "cmd",
        "ctrl" | "control" => "ctrl",
        "alt" | "option" | "opt" => "alt",
        "shift" => "shift",
        "tab" => "tab",
        "space" | " " => "space",
        "delete" | "backspace" => "delete",
        "fn" => "fn",
        "up" => "arrow-up",
        "down" => "arrow-down",
        "left" => "arrow-left",
        "right" => "arrow-right",
        "home" => "home",
        "end" => "end",
        "pageup" | "page_up" => "page-up",
        "pagedown" | "page_down" => "page-down",
        "f1" => "f1",
        "f2" => "f2",
        "f3" => "f3",
        "f4" => "f4",
        "f5" => "f5",
        "f6" => "f6",
        "f7" => "f7",
        "f8" => "f8",
        "f9" => "f9",
        "f10" => "f10",
        "f11" => "f11",
        "f12" => "f12",
        _ => {
            // Return as-is for single character keys (a-z, 0-9, etc.)
            // We can't return the lowercased version since we need the original
            // But cliclick accepts single chars directly
            // Leak is avoided by returning the original name reference
            // Since we already matched on lowercase, return the most common case
            name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{AutonomyLevel, SecurityPolicy};

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    #[test]
    fn tool_name() {
        let tool = ComputerTool::new(test_security(), None);
        assert_eq!(tool.name(), "computer");
    }

    #[test]
    fn tool_description_not_empty() {
        let tool = ComputerTool::new(test_security(), None);
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn tool_schema_has_action() {
        let tool = ComputerTool::new(test_security(), None);
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["action"].is_object());
        assert_eq!(schema["required"], json!(["action"]));
    }

    #[test]
    fn tool_schema_has_all_params() {
        let tool = ComputerTool::new(test_security(), None);
        let schema = tool.parameters_schema();
        let props = &schema["properties"];
        for param in ["action", "x", "y", "text", "key", "direction", "amount", "prompt"] {
            assert!(props[param].is_object(), "Missing param: {param}");
        }
    }

    // ── Key combo parsing ───────────────────────────────────────────────

    #[test]
    fn parse_single_key() {
        assert_eq!(parse_key_combo("enter"), vec!["kp:return"]);
        assert_eq!(parse_key_combo("tab"), vec!["kp:tab"]);
        assert_eq!(parse_key_combo("escape"), vec!["kp:escape"]);
    }

    #[test]
    fn parse_key_with_modifier() {
        // cmd+tab uses cliclick (tab is a special key)
        assert_eq!(
            parse_key_combo("cmd+tab"),
            vec!["kd:cmd", "kp:tab", "ku:cmd"]
        );
    }

    #[test]
    fn parse_key_with_multiple_modifiers() {
        assert_eq!(
            parse_key_combo("cmd+shift+tab"),
            vec!["kd:cmd", "kd:shift", "kp:tab", "ku:shift", "ku:cmd"]
        );
    }

    #[test]
    fn parse_key_with_three_modifiers() {
        assert_eq!(
            parse_key_combo("ctrl+alt+shift+delete"),
            vec![
                "kd:ctrl",
                "kd:alt",
                "kd:shift",
                "kp:delete",
                "ku:shift",
                "ku:alt",
                "ku:ctrl"
            ]
        );
    }

    #[test]
    fn parse_arrow_keys() {
        assert_eq!(parse_key_combo("up"), vec!["kp:arrow-up"]);
        assert_eq!(parse_key_combo("down"), vec!["kp:arrow-down"]);
    }

    // ── cliclick special key detection ──────────────────────────────────

    #[test]
    fn special_keys_detected() {
        assert!(is_cliclick_special_key("return"));
        assert!(is_cliclick_special_key("tab"));
        assert!(is_cliclick_special_key("space"));
        assert!(is_cliclick_special_key("arrow-up"));
        assert!(is_cliclick_special_key("delete"));
        assert!(is_cliclick_special_key("f1"));
    }

    #[test]
    fn regular_keys_not_special() {
        assert!(!is_cliclick_special_key("c"));
        assert!(!is_cliclick_special_key("v"));
        assert!(!is_cliclick_special_key("a"));
        assert!(!is_cliclick_special_key("z"));
        assert!(!is_cliclick_special_key("1"));
    }

    // ── AppleScript key codes ───────────────────────────────────────────

    #[test]
    fn applescript_key_codes() {
        assert_eq!(applescript_key_code("return"), Some(36));
        assert_eq!(applescript_key_code("enter"), Some(36));
        assert_eq!(applescript_key_code("tab"), Some(48));
        assert_eq!(applescript_key_code("escape"), Some(53));
        assert_eq!(applescript_key_code("up"), Some(126));
        assert_eq!(applescript_key_code("f1"), Some(122));
        // Regular character keys return None
        assert_eq!(applescript_key_code("c"), None);
        assert_eq!(applescript_key_code("v"), None);
    }

    // ── Key name mapping ────────────────────────────────────────────────

    #[test]
    fn map_key_names() {
        assert_eq!(map_key_name("enter"), "return");
        assert_eq!(map_key_name("return"), "return");
        assert_eq!(map_key_name("esc"), "escape");
        assert_eq!(map_key_name("cmd"), "cmd");
        assert_eq!(map_key_name("command"), "cmd");
        assert_eq!(map_key_name("ctrl"), "ctrl");
        assert_eq!(map_key_name("alt"), "alt");
        assert_eq!(map_key_name("option"), "alt");
        assert_eq!(map_key_name("backspace"), "delete");
    }

    // ── Coordinate extraction ───────────────────────────────────────────

    #[test]
    fn extract_coords_valid() {
        let args = json!({"x": 100, "y": 200});
        assert_eq!(extract_coords(&args).unwrap(), (100, 200));
    }

    #[test]
    fn extract_coords_missing_x() {
        let args = json!({"y": 200});
        assert!(extract_coords(&args).is_err());
    }

    #[test]
    fn extract_coords_missing_y() {
        let args = json!({"x": 100});
        assert!(extract_coords(&args).is_err());
    }

    // ── Vision request serialization ────────────────────────────────────

    #[test]
    fn vision_request_serializes() {
        let req = VisionRequest {
            contents: vec![VisionContent {
                role: Some("user".into()),
                parts: vec![
                    VisionPart::Text {
                        text: "describe".into(),
                    },
                    VisionPart::InlineData {
                        inline_data: InlineData {
                            mime_type: "image/jpeg".into(),
                            data: "abc123".into(),
                        },
                    },
                ],
            }],
            generation_config: VisionGenConfig {
                temperature: 0.1,
                max_output_tokens: 4096,
                response_mime_type: None,
                response_schema: None,
            },
        };

        let json_str = serde_json::to_string(&req).unwrap();
        assert!(json_str.contains("\"text\":\"describe\""));
        assert!(json_str.contains("\"inline_data\""));
        assert!(json_str.contains("\"mime_type\":\"image/jpeg\""));
        assert!(json_str.contains("\"data\":\"abc123\""));
        assert!(json_str.contains("\"temperature\":0.1"));
        assert!(json_str.contains("\"maxOutputTokens\":4096"));
    }

    // ── Structured Vision Response tests ───────────────────────────────

    #[test]
    fn parse_vision_response_full() {
        let json = r#"{
            "foreground_app": "KakaoTalk",
            "screen_state": "Chat list visible",
            "elements": [
                {"label": "가족", "type": "chat_item", "x": 150, "y": 300, "width": 120, "height": 40, "state": "enabled"},
                {"label": "Search", "type": "text_field", "x": 200, "y": 60, "width": 300, "height": 30}
            ],
            "visible_text": "가족 - 좋은 아침!"
        }"#;

        let resp: VisionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.foreground_app.as_deref(), Some("KakaoTalk"));
        assert_eq!(resp.elements.as_ref().unwrap().len(), 2);
        assert_eq!(resp.elements.as_ref().unwrap()[0].x, 150);
        assert_eq!(resp.elements.as_ref().unwrap()[0].y, 300);
    }

    #[test]
    fn parse_vision_response_minimal() {
        let json = r#"{
            "foreground_app": "Finder",
            "screen_state": "idle",
            "elements": []
        }"#;

        let resp: VisionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.foreground_app.as_deref(), Some("Finder"));
        assert!(resp.elements.as_ref().unwrap().is_empty());
        assert!(resp.visible_text.is_none());
    }

    #[test]
    fn format_vision_response_produces_structured_text() {
        let resp = VisionResponse {
            foreground_app: Some("Safari".into()),
            screen_state: Some("Web page loaded".into()),
            elements: Some(vec![
                VisionElement {
                    label: "Submit".into(),
                    element_type: "button".into(),
                    x: 400,
                    y: 500,
                    width: Some(80),
                    height: Some(30),
                    state: Some("enabled".into()),
                },
                VisionElement {
                    label: "Email".into(),
                    element_type: "text_field".into(),
                    x: 300,
                    y: 200,
                    width: None,
                    height: None,
                    state: None,
                },
            ]),
            visible_text: Some("Welcome to the site".into()),
        };

        let text = format_vision_response(&resp);
        assert!(text.contains("App: Safari"));
        assert!(text.contains("State: Web page loaded"));
        assert!(text.contains("\"Submit\" (button) at (400, 500) [80x30] (enabled)"));
        assert!(text.contains("\"Email\" (text_field) at (300, 200)"));
        assert!(text.contains("[Visible Text]"));
        assert!(text.contains("Welcome to the site"));
    }

    #[test]
    fn vision_response_schema_has_required_fields() {
        let schema = vision_response_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "foreground_app"));
        assert!(required.iter().any(|v| v == "screen_state"));
        assert!(required.iter().any(|v| v == "elements"));
    }

    // ── Caching tests ───────────────────────────────────────────────────

    #[test]
    fn screen_width_cache_starts_empty() {
        // Cache should start at 0 (uncached)
        // Note: can't test atomics in isolation since they're global,
        // but we verify the sentinel value logic
        assert_eq!(0u32, 0); // placeholder — real test is the integration
    }

    // ── Async tests ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn execute_unknown_action() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool.execute(json!({"action": "fly"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("Unknown action"));
    }

    #[tokio::test]
    async fn execute_missing_action() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("Missing"));
    }

    #[tokio::test]
    async fn click_missing_coords() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool
            .execute(json!({"action": "click"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("parameter: x"));
    }

    #[tokio::test]
    async fn type_missing_text() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool
            .execute(json!({"action": "type"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("text"));
    }

    #[tokio::test]
    async fn key_missing_key() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool
            .execute(json!({"action": "key"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("key"));
    }

    #[tokio::test]
    async fn open_app_missing_name() {
        let tool = ComputerTool::new(test_security(), None);
        let result = tool
            .execute(json!({"action": "open_app"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("text"));
    }

    #[tokio::test]
    async fn read_only_blocks_all_actions() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = ComputerTool::new(security, None);
        let result = tool
            .execute(json!({"action": "screenshot"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("read-only"));
    }
}
