// --- ZeroClaw fork: Hybrid Programmatic Grounding ---
//
// Cascading screen observation system:
//   Tier 1: Swift AXAPI (compiled CLI, ~50ms, most precise)
//   Tier 2: JXA System Events (built-in macOS, slower but always available)
//   Tier 3: Vision/Screenshot (existing Gemini vision — handled in computer.rs)
//
// This module handles Tiers 1 & 2. If both fail, computer.rs falls through
// to the existing Vision API path (Tier 3) unchanged.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Timeout for each probe attempt.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Timeout for Swift compilation (first use only).
const COMPILE_TIMEOUT: Duration = Duration::from_secs(60);

/// Embedded Swift source — compiled on first use, cached by content hash.
const SWIFT_SOURCE: &str = include_str!("../screen_probe.swift");

/// Embedded JXA source — written to temp file and run via osascript.
const JXA_SOURCE: &str = include_str!("../screen_probe.js");

// ── Data types (match Swift/JXA JSON output) ─────────────────────────────────

/// Bounding box in screen coordinates (origin = top-left of main display).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// A single UI element discovered by the probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIElement {
    pub id: String,
    pub role: String,
    pub name: Option<String>,
    pub value: Option<String>,
    pub bbox: Option<BBox>,
    pub interactable: bool,
}

/// Result of a screen probe — the full observable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenState {
    pub status: String,
    pub source: String,
    pub app_name: Option<String>,
    pub window_title: Option<String>,
    pub elements: Vec<UIElement>,
}

// ── Swift binary management ──────────────────────────────────────────────────

/// Compute a deterministic binary path based on Swift source content hash.
/// When ZeroClaw is redeployed with updated Swift code, a new binary is compiled.
fn swift_binary_path() -> String {
    let mut hasher = DefaultHasher::new();
    SWIFT_SOURCE.hash(&mut hasher);
    let hash = hasher.finish();
    format!("/tmp/zeroclaw_screen_probe_{hash:016x}")
}

/// Ensure the compiled Swift binary exists. Compiles on first use.
async fn ensure_swift_binary() -> Result<String, String> {
    let bin_path = swift_binary_path();

    // Fast path: binary already compiled
    if tokio::fs::metadata(&bin_path).await.is_ok() {
        return Ok(bin_path);
    }

    // Write Swift source to temp file
    let src_path = format!("{bin_path}.swift");
    tokio::fs::write(&src_path, SWIFT_SOURCE)
        .await
        .map_err(|e| format!("Failed to write Swift source: {e}"))?;

    // Compile with optimization
    tracing::info!("Compiling screen probe Swift binary (first use)...");
    let output = tokio::time::timeout(
        COMPILE_TIMEOUT,
        tokio::process::Command::new("swiftc")
            .args([
                "-O",
                "-o",
                &bin_path,
                &src_path,
                "-framework",
                "ApplicationServices",
                "-framework",
                "AppKit",
            ])
            .output(),
    )
    .await
    .map_err(|_| "Swift compilation timed out (60s)".to_string())?
    .map_err(|e| format!("Failed to run swiftc: {e}"))?;

    // Clean up source file
    let _ = tokio::fs::remove_file(&src_path).await;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Swift compilation failed: {stderr}"));
    }

    tracing::info!("Screen probe compiled: {bin_path}");
    Ok(bin_path)
}

// ── Probe execution ──────────────────────────────────────────────────────────

/// Tier 1: Run Swift AXAPI probe.
async fn probe_swift() -> Result<ScreenState, String> {
    let bin_path = ensure_swift_binary().await?;

    let output = tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::process::Command::new(&bin_path).output(),
    )
    .await
    .map_err(|_| "Swift probe timed out (10s)".to_string())?
    .map_err(|e| format!("Failed to run screen_probe: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Failed to parse Swift probe JSON: {e}"))
}

/// Tier 2: Run JXA System Events probe.
async fn probe_jxa() -> Result<ScreenState, String> {
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
    let script_path = format!("/tmp/zeroclaw_jxa_probe_{ts}.js");

    tokio::fs::write(&script_path, JXA_SOURCE)
        .await
        .map_err(|e| format!("Failed to write JXA script: {e}"))?;

    let output = tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::process::Command::new("osascript")
            .args(["-l", "JavaScript", &script_path])
            .output(),
    )
    .await
    .map_err(|_| {
        let _ = std::fs::remove_file(&script_path);
        "JXA probe timed out (10s)".to_string()
    })?
    .map_err(|e| {
        let _ = std::fs::remove_file(&script_path);
        format!("Failed to run osascript: {e}")
    })?;

    let _ = tokio::fs::remove_file(&script_path).await;

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Failed to parse JXA probe JSON: {e}"))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Probe screen state using cascading fallback:
///   1. Swift AXAPI (fast, precise)
///   2. JXA System Events (slower, always available)
///
/// Returns the first successful result with elements.
/// Returns Err if both probes fail or return 0 elements — caller should
/// fall back to Vision API (Tier 3).
pub async fn probe_screen_state() -> Result<ScreenState, String> {
    // Tier 1: Swift AXAPI
    match probe_swift().await {
        Ok(state) if state.status == "ok" && !state.elements.is_empty() => {
            tracing::debug!(
                "Screen probe: Swift AXAPI returned {} elements for {:?}",
                state.elements.len(),
                state.app_name
            );
            return Ok(state);
        }
        Ok(state) => {
            tracing::debug!(
                "Screen probe: Swift returned status={}, {} elements",
                state.status,
                state.elements.len()
            );
        }
        Err(e) => {
            tracing::debug!("Screen probe: Swift failed: {e}");
        }
    }

    // Tier 2: JXA System Events
    match probe_jxa().await {
        Ok(state) if state.status == "ok" && !state.elements.is_empty() => {
            tracing::debug!(
                "Screen probe: JXA returned {} elements for {:?}",
                state.elements.len(),
                state.app_name
            );
            return Ok(state);
        }
        Ok(state) => {
            tracing::debug!(
                "Screen probe: JXA returned status={}, {} elements",
                state.status,
                state.elements.len()
            );
        }
        Err(e) => {
            tracing::debug!("Screen probe: JXA failed: {e}");
        }
    }

    Err("All screen probes returned 0 elements — falling back to vision".into())
}

/// Format a ScreenState into a structured text block for the agent LLM.
///
/// Output format matches `format_vision_response` in computer.rs so the agent
/// can use the same coordinate-based click workflow regardless of probe source.
pub fn format_screen_state(state: &ScreenState) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("[Screen Analysis]\n");
    if let Some(ref app) = state.app_name {
        out.push_str(&format!("App: {app}\n"));
    }
    if let Some(ref title) = state.window_title {
        out.push_str(&format!("Window: {title}\n"));
    }
    out.push_str(&format!("Source: {} (programmatic)\n", state.source));
    out.push('\n');

    if !state.elements.is_empty() {
        out.push_str("[Interactive Elements] (use these coordinates for click actions)\n");
        let mut count = 0;
        for el in &state.elements {
            let Some(ref bbox) = el.bbox else { continue };

            // Compute center coordinates (what the agent should click)
            let cx = bbox.x + bbox.w / 2;
            let cy = bbox.y + bbox.h / 2;

            let name = el.name.as_deref().unwrap_or("(unnamed)");

            // Clean role prefix for readability
            let role = el.role.strip_prefix("AX").unwrap_or(&el.role);

            let value_str = el
                .value
                .as_deref()
                .filter(|v| !v.is_empty())
                .map(|v| format!(" = \"{v}\""))
                .unwrap_or_default();

            let interact = if el.interactable { " *" } else { "" };

            count += 1;
            out.push_str(&format!(
                "{n}. \"{name}\" ({role}) at ({cx}, {cy}) [{w}x{h}]{value}{interact}\n",
                n = count,
                w = bbox.w,
                h = bbox.h,
                value = value_str,
            ));

            if count >= 150 {
                let remaining = state.elements.len() - count;
                if remaining > 0 {
                    out.push_str(&format!("  ... and {remaining} more elements\n"));
                }
                break;
            }
        }
        out.push('\n');
    }

    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swift_binary_path_is_deterministic() {
        let p1 = swift_binary_path();
        let p2 = swift_binary_path();
        assert_eq!(p1, p2);
        assert!(p1.starts_with("/tmp/zeroclaw_screen_probe_"));
        assert!(p1.len() > "/tmp/zeroclaw_screen_probe_".len());
    }

    #[test]
    fn parse_screen_state_json() {
        let json = r#"{
            "status": "ok",
            "source": "swift_axapi",
            "app_name": "KakaoTalk",
            "window_title": "KakaoTalk",
            "elements": [
                {
                    "id": "ax_1",
                    "role": "AXButton",
                    "name": "Send",
                    "value": null,
                    "bbox": {"x": 360, "y": 580, "w": 80, "h": 30},
                    "interactable": true
                },
                {
                    "id": "ax_2",
                    "role": "AXTextField",
                    "name": "Message",
                    "value": "Hello",
                    "bbox": {"x": 50, "y": 580, "w": 300, "h": 30},
                    "interactable": true
                }
            ]
        }"#;

        let state: ScreenState = serde_json::from_str(json).unwrap();
        assert_eq!(state.status, "ok");
        assert_eq!(state.source, "swift_axapi");
        assert_eq!(state.app_name.as_deref(), Some("KakaoTalk"));
        assert_eq!(state.elements.len(), 2);
        assert_eq!(state.elements[0].role, "AXButton");
        assert!(state.elements[0].interactable);
        assert_eq!(state.elements[0].bbox.as_ref().unwrap().x, 360);
    }

    #[test]
    fn parse_screen_state_empty_elements() {
        let json = r#"{
            "status": "error_no_accessibility",
            "source": "swift_axapi",
            "app_name": null,
            "window_title": null,
            "elements": []
        }"#;

        let state: ScreenState = serde_json::from_str(json).unwrap();
        assert_eq!(state.status, "error_no_accessibility");
        assert!(state.elements.is_empty());
        assert!(state.app_name.is_none());
    }

    #[test]
    fn format_screen_state_with_elements() {
        let state = ScreenState {
            status: "ok".into(),
            source: "swift_axapi".into(),
            app_name: Some("Safari".into()),
            window_title: Some("Apple".into()),
            elements: vec![
                UIElement {
                    id: "ax_1".into(),
                    role: "AXButton".into(),
                    name: Some("Submit".into()),
                    value: None,
                    bbox: Some(BBox {
                        x: 360,
                        y: 480,
                        w: 80,
                        h: 30,
                    }),
                    interactable: true,
                },
                UIElement {
                    id: "ax_2".into(),
                    role: "AXTextField".into(),
                    name: Some("Email".into()),
                    value: Some("test@example.com".into()),
                    bbox: Some(BBox {
                        x: 200,
                        y: 400,
                        w: 200,
                        h: 25,
                    }),
                    interactable: true,
                },
                UIElement {
                    id: "ax_3".into(),
                    role: "AXStaticText".into(),
                    name: Some("Welcome".into()),
                    value: None,
                    bbox: Some(BBox {
                        x: 100,
                        y: 100,
                        w: 300,
                        h: 20,
                    }),
                    interactable: false,
                },
            ],
        };

        let text = format_screen_state(&state);
        assert!(text.contains("App: Safari"));
        assert!(text.contains("Window: Apple"));
        assert!(text.contains("Source: swift_axapi (programmatic)"));
        // Center of (360, 480, 80, 30) = (400, 495)
        assert!(text.contains("\"Submit\" (Button) at (400, 495) [80x30]"));
        // Center of (200, 400, 200, 25) = (300, 412)
        assert!(text.contains("\"Email\" (TextField) at (300, 412) [200x25] = \"test@example.com\""));
        // Non-interactable element should NOT have *
        assert!(text.contains("\"Welcome\" (StaticText) at (250, 110) [300x20]\n"));
        // Interactable elements should have *
        assert!(text.contains("[80x30] *"));
    }

    #[test]
    fn format_screen_state_empty() {
        let state = ScreenState {
            status: "ok".into(),
            source: "jxa_system_events".into(),
            app_name: Some("Finder".into()),
            window_title: None,
            elements: vec![],
        };

        let text = format_screen_state(&state);
        assert!(text.contains("App: Finder"));
        assert!(!text.contains("[Interactive Elements]"));
    }

    #[test]
    fn format_screen_state_skips_no_bbox() {
        let state = ScreenState {
            status: "ok".into(),
            source: "swift_axapi".into(),
            app_name: Some("Test".into()),
            window_title: None,
            elements: vec![UIElement {
                id: "ax_1".into(),
                role: "AXButton".into(),
                name: Some("Ghost".into()),
                value: None,
                bbox: None, // no position — skip in output
                interactable: true,
            }],
        };

        let text = format_screen_state(&state);
        // Element without bbox should not appear in the numbered list
        assert!(!text.contains("Ghost"));
    }

    #[test]
    fn bbox_serde_roundtrip() {
        let bbox = BBox {
            x: 100,
            y: 200,
            w: 50,
            h: 30,
        };
        let json = serde_json::to_string(&bbox).unwrap();
        let parsed: BBox = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.x, 100);
        assert_eq!(parsed.w, 50);
    }

    #[test]
    fn ui_element_serde_roundtrip() {
        let el = UIElement {
            id: "ax_42".into(),
            role: "AXButton".into(),
            name: Some("OK".into()),
            value: None,
            bbox: Some(BBox {
                x: 0,
                y: 0,
                w: 60,
                h: 24,
            }),
            interactable: true,
        };
        let json = serde_json::to_string(&el).unwrap();
        let parsed: UIElement = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "ax_42");
        assert!(parsed.interactable);
        assert_eq!(parsed.bbox.unwrap().w, 60);
    }

    #[test]
    fn embedded_sources_not_empty() {
        assert!(!SWIFT_SOURCE.is_empty());
        assert!(SWIFT_SOURCE.contains("AXUIElement"));
        assert!(!JXA_SOURCE.is_empty());
        assert!(JXA_SOURCE.contains("System Events"));
    }
}
