use super::traits::{Tool, ToolResult};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use base64::Engine;
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// Timeout for shell sub-commands (screencapture, sips, cliclick, etc.).
const CMD_TIMEOUT: Duration = Duration::from_secs(15);
/// Timeout for the Gemini Vision API call.
const VISION_TIMEOUT: Duration = Duration::from_secs(30);
/// Vision model to use for screenshot descriptions.
const VISION_MODEL: &str = "gemini-2.0-flash";
/// Maximum JPEG file size to send to vision API (~4 MB).
const MAX_JPEG_BYTES: u64 = 4_194_304;

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
}

// ── Tool implementation ─────────────────────────────────────────────────────

/// Computer-use tool — see the screen via vision AI and control mouse/keyboard.
pub struct ComputerTool {
    security: Arc<SecurityPolicy>,
    gemini_key: Option<String>,
    client: reqwest::Client,
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
        }
    }

    // ── Screenshot + Vision ─────────────────────────────────────────────

    async fn action_screenshot(&self, args: &serde_json::Value) -> ToolResult {
        let extra_prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // 1. Capture screenshot
        let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let path = format!("/tmp/zeroclaw_screen_{ts}.jpg");

        if let Err(e) = run_cmd("screencapture", &["-x", "-t", "jpg", &path]).await {
            return err_result(format!(
                "Screenshot capture failed: {e}\n\n\
                 If Screen Recording permission is needed:\n\
                 1. Open: System Settings → Privacy & Security → Screen Recording\n\
                 2. Click + and add /Applications/ZeroClaw.app\n\
                 3. Toggle it ON, then restart the daemon"
            ));
        }

        // Check if screenshot file is empty (permission denied produces 0-byte file)
        if let Ok(meta) = tokio::fs::metadata(&path).await {
            if meta.len() == 0 {
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

        // 2. Get logical screen width and resize
        let logical_width = get_logical_screen_width().await;
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
        if meta.len() > MAX_JPEG_BYTES {
            let _ = tokio::fs::remove_file(&path).await;
            return err_result(format!(
                "Screenshot too large ({} bytes). Max: {MAX_JPEG_BYTES}",
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

        // 4. Image width for the vision prompt dimension hint
        let img_w = logical_width.unwrap_or(0);

        // 5. Call Gemini Vision API
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
                };
            }
        };

        let dim_hint = if img_w > 0 {
            format!("Image width: {img_w}px.")
        } else {
            String::new()
        };

        let prompt = format!(
            "Describe this macOS screenshot in detail. {dim_hint}\n\
             For each interactive UI element (buttons, text fields, links, menu items, \
             tabs, icons, checkboxes), provide its approximate center coordinates as \
             [x, y] in pixels from the top-left corner of the image.\n\
             Format: Element 'Label' at [x, y]\n\n\
             Describe:\n\
             1. Which application is in the foreground\n\
             2. All visible text content (messages, labels, titles)\n\
             3. All interactive elements with coordinates\n\
             4. Current state (any dialogs, notifications, selections)\n\
             {extra_prompt}\n\
             Be concise but thorough. Focus on actionable elements."
        );

        match self.call_vision_api(api_key, &b64, &prompt).await {
            Ok(description) => ToolResult {
                success: true,
                output: description,
                error: None,
            },
            Err(e) => ToolResult {
                success: false,
                output: format!("Screenshot captured ({} bytes) but vision API failed.", bytes.len()),
                error: Some(format!("Vision API error: {e}")),
            },
        }
    }

    async fn call_vision_api(
        &self,
        api_key: &str,
        jpeg_b64: &str,
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
                            mime_type: "image/jpeg".into(),
                            data: jpeg_b64.to_string(),
                        },
                    },
                ],
            }],
            generation_config: VisionGenConfig {
                temperature: 0.1,
                max_output_tokens: 4096,
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

        if let Err(e) = check_cliclick().await {
            return err_result(e);
        }

        let coord = format!("{prefix}:{x},{y}");
        match run_cmd("cliclick", &[&coord]).await {
            Ok(out) => ToolResult {
                success: true,
                output: format!("Clicked at ({x}, {y}). {out}"),
                error: None,
            },
            Err(e) => err_result(format!("Click failed: {e}")),
        }
    }

    // ── Type action ─────────────────────────────────────────────────────

    async fn action_type(&self, args: &serde_json::Value) -> ToolResult {
        let Some(text) = args.get("text").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: text");
        };

        if let Err(e) = check_cliclick().await {
            return err_result(e);
        }

        let arg = format!("t:{text}");
        match run_cmd("cliclick", &[&arg]).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Typed: \"{text}\""),
                error: None,
            },
            Err(e) => err_result(format!("Type failed: {e}")),
        }
    }

    // ── Key combo action ────────────────────────────────────────────────

    async fn action_key(&self, args: &serde_json::Value) -> ToolResult {
        let Some(combo) = args.get("key").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: key");
        };

        if let Err(e) = check_cliclick().await {
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
            },
            Err(e) => err_result(format!("Key press failed: {e}")),
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

        // Use osascript for scrolling (cliclick doesn't support it)
        let scroll_amount = match direction {
            "up" | "left" => amount,
            "down" | "right" => -amount,
            _ => return err_result(format!("Invalid scroll direction: {direction}. Use up/down/left/right")),
        };

        let axis = if direction == "left" || direction == "right" {
            "horizontal scroll"
        } else {
            "scroll"
        };

        // Move cursor to position first if coordinates provided
        if let Ok((x, y)) = extract_coords(args) {
            if check_cliclick().await.is_ok() {
                let _ = run_cmd("cliclick", &[&format!("m:{x},{y}")]).await;
            }
        }

        let script = format!(
            "tell application \"System Events\" to {axis} by {scroll_amount}"
        );
        match run_cmd("osascript", &["-e", &script]).await {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Scrolled {direction} by {amount}"),
                error: None,
            },
            Err(e) => err_result(format!("Scroll failed: {e}")),
        }
    }

    // ── Open app action ─────────────────────────────────────────────────

    async fn action_open_app(&self, args: &serde_json::Value) -> ToolResult {
        let Some(app_name) = args.get("text").and_then(|v| v.as_str()) else {
            return err_result("Missing required parameter: text (app name)");
        };

        match run_cmd("open", &["-a", app_name]).await {
            Ok(_) => {
                // Wait briefly for app to launch and come to foreground
                tokio::time::sleep(Duration::from_millis(500)).await;
                ToolResult {
                    success: true,
                    output: format!("Opened application: {app_name}"),
                    error: None,
                }
            }
            Err(e) => err_result(format!("Failed to open {app_name}: {e}")),
        }
    }

    // ── Cursor position action ──────────────────────────────────────────

    async fn action_cursor_position(&self) -> ToolResult {
        if let Err(e) = check_cliclick().await {
            return err_result(e);
        }

        match run_cmd("cliclick", &["p"]).await {
            Ok(out) => ToolResult {
                success: true,
                output: format!("Cursor position: {out}"),
                error: None,
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
async fn check_cliclick() -> Result<(), String> {
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

/// Get logical screen width via Python `AppKit`, with `system_profiler` fallback.
async fn get_logical_screen_width() -> Option<u32> {
    // Method 1: Python AppKit (fast, reliable)
    if let Ok(out) = run_cmd(
        "python3",
        &[
            "-c",
            "from AppKit import NSScreen; f=NSScreen.mainScreen().frame(); print(int(f.size.width))",
        ],
    )
    .await
    {
        if let Ok(w) = out.trim().parse::<u32>() {
            if w > 0 {
                return Some(w);
            }
        }
    }

    // Method 2: system_profiler fallback
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
///
/// Examples:
/// - `"cmd+c"` → `["kd:cmd", "kp:c", "ku:cmd"]`
/// - `"cmd+shift+t"` → `["kd:cmd", "kd:shift", "kp:t", "ku:shift", "ku:cmd"]`
/// - `"enter"` → `["kp:return"]`
/// - `"tab"` → `["kp:tab"]`
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
        assert_eq!(
            parse_key_combo("cmd+c"),
            vec!["kd:cmd", "kp:c", "ku:cmd"]
        );
    }

    #[test]
    fn parse_key_with_multiple_modifiers() {
        assert_eq!(
            parse_key_combo("cmd+shift+t"),
            vec!["kd:cmd", "kd:shift", "kp:t", "ku:shift", "ku:cmd"]
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
        assert_eq!(
            parse_key_combo("cmd+left"),
            vec!["kd:cmd", "kp:arrow-left", "ku:cmd"]
        );
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
