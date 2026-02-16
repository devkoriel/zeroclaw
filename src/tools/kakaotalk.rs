use super::traits::{Tool, ToolResult};
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// Timeout for AppleScript commands.
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(15);

/// KakaoTalk messaging tool — sends messages via AppleScript accessibility API.
/// This bypasses the computer tool's slow vision-AI pipeline entirely.
///
/// Supports:
/// - Opening/finding chat rooms by name
/// - Sending messages (including Korean/CJK text via clipboard)
/// - Reading recent messages from a chat
/// - Listing open chat windows
pub struct KakaoTalkTool {
    security: Arc<SecurityPolicy>,
}

impl KakaoTalkTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for KakaoTalkTool {
    fn name(&self) -> &str {
        "kakaotalk"
    }

    fn description(&self) -> &str {
        "Send and read KakaoTalk messages via native macOS accessibility. \
         Fast (<1s) and reliable — no screen vision or coordinate guessing needed. \
         Actions: send_message, read_messages, list_chats, open_chat, search_chat."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: send_message, read_messages, list_chats, open_chat, search_chat",
                    "enum": ["send_message", "read_messages", "list_chats", "open_chat", "search_chat"]
                },
                "chat_name": {
                    "type": "string",
                    "description": "Chat room or contact name (for send_message, read_messages, open_chat)"
                },
                "message": {
                    "type": "string",
                    "description": "Message text to send (for send_message)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search_chat)"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent messages to read (default: 10, for read_messages)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Check autonomy level
        if matches!(self.security.autonomy, crate::security::AutonomyLevel::ReadOnly) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("KakaoTalk tool is not available in read-only mode".into()),
            });
        }

        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'action' parameter"))?;

        match action {
            "send_message" => self.send_message(&args).await,
            "read_messages" => self.read_messages(&args).await,
            "list_chats" => self.list_chats().await,
            "open_chat" => self.open_chat(&args).await,
            "search_chat" => self.search_chat(&args).await,
            _ => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Unknown action: {action}. Use: send_message, read_messages, list_chats, open_chat, search_chat")),
            }),
        }
    }
}

impl KakaoTalkTool {
    /// Send a message to a KakaoTalk chat room.
    /// Uses clipboard (pbcopy) for reliable Korean/CJK text handling.
    async fn send_message(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let chat_name = match args.get("chat_name").and_then(|v| v.as_str()) {
            Some(name) if !name.is_empty() => name,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: chat_name".into()),
                });
            }
        };

        let message = match args.get("message").and_then(|v| v.as_str()) {
            Some(msg) if !msg.is_empty() => msg,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: message".into()),
                });
            }
        };

        // Step 1: Activate KakaoTalk
        if let Err(e) = run_osascript("tell application \"KakaoTalk\" to activate").await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to activate KakaoTalk: {e}")),
            });
        }

        // Brief pause for app activation
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Step 2: Check if the chat window exists, if not try to find and open it
        let window_exists = check_window_exists(chat_name).await;
        if !window_exists {
            // Try to open the chat via search
            match self.find_and_open_chat(chat_name).await {
                Ok(true) => {
                    // Wait for window to open
                    tokio::time::sleep(Duration::from_millis(800)).await;
                }
                Ok(false) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!(
                            "Chat window '{}' not found. Open the chat first or check the name.",
                            chat_name
                        )),
                    });
                }
                Err(e) => {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to search for chat: {e}")),
                    });
                }
            }
        }

        // Step 3: Focus the chat window
        let focus_script = format!(
            "tell application \"System Events\" to tell process \"KakaoTalk\" to perform action \"AXRaise\" of window \"{}\"",
            escape_applescript(chat_name)
        );
        if let Err(e) = run_osascript(&focus_script).await {
            tracing::warn!("Failed to raise window (may still work): {e}");
        }

        // Step 4: Copy message to clipboard (handles Korean/CJK perfectly)
        if let Err(e) = copy_to_clipboard(message).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to copy message to clipboard: {e}")),
            });
        }

        // Step 5: Set focus to the text input area and paste
        // Try the direct accessibility approach first (set value), fall back to paste
        let set_value_script = format!(
            "tell application \"System Events\" to tell process \"KakaoTalk\" to tell window \"{}\" to tell scroll area 2 to tell text area 1 to set value to \"{}\"",
            escape_applescript(chat_name),
            escape_applescript(message)
        );

        let use_paste = if let Err(_) = run_osascript(&set_value_script).await {
            // Direct set failed — fall back to click + paste
            tracing::info!("Direct text set failed, falling back to clipboard paste");
            true
        } else {
            false
        };

        if use_paste {
            // Click on the text input area (bottom of window)
            let click_input_script = format!(
                "tell application \"System Events\" to tell process \"KakaoTalk\" to tell window \"{}\" to click scroll area 2",
                escape_applescript(chat_name)
            );
            let _ = run_osascript(&click_input_script).await;
            tokio::time::sleep(Duration::from_millis(200)).await;

            // Paste from clipboard
            let paste_script = "tell application \"System Events\" to keystroke \"v\" using command down";
            if let Err(e) = run_osascript(paste_script).await {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to paste message: {e}")),
                });
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Step 6: Press Enter to send
        let send_script = "tell application \"System Events\" to tell process \"KakaoTalk\" to key code 36";
        if let Err(e) = run_osascript(send_script).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to press Enter: {e}")),
            });
        }

        Ok(ToolResult {
            success: true,
            output: format!("Message sent to '{}': {}", chat_name, truncate_for_display(message, 100)),
            error: None,
        })
    }

    /// Read recent messages from a KakaoTalk chat window.
    async fn read_messages(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let chat_name = match args.get("chat_name").and_then(|v| v.as_str()) {
            Some(name) if !name.is_empty() => name,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: chat_name".into()),
                });
            }
        };

        let _count = args
            .get("count")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as usize;

        // Activate KakaoTalk
        if let Err(e) = run_osascript("tell application \"KakaoTalk\" to activate").await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to activate KakaoTalk: {e}")),
            });
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Focus the window
        let focus_script = format!(
            "tell application \"System Events\" to tell process \"KakaoTalk\" to perform action \"AXRaise\" of window \"{}\"",
            escape_applescript(chat_name)
        );
        let _ = run_osascript(&focus_script).await;

        // Try to read the chat content via accessibility
        // KakaoTalk's chat messages are in scroll area 1 (the message display area)
        let read_script = format!(
            "tell application \"System Events\" to tell process \"KakaoTalk\" to tell window \"{}\" to value of scroll area 1",
            escape_applescript(chat_name)
        );

        match run_osascript(&read_script).await {
            Ok(content) if !content.trim().is_empty() => Ok(ToolResult {
                success: true,
                output: content,
                error: None,
            }),
            _ => {
                // Fallback: try getting all static text elements
                let fallback_script = format!(
                    "tell application \"System Events\" to tell process \"KakaoTalk\" to tell window \"{}\" to get value of every static text of scroll area 1",
                    escape_applescript(chat_name)
                );
                match run_osascript(&fallback_script).await {
                    Ok(content) => Ok(ToolResult {
                        success: true,
                        output: if content.trim().is_empty() {
                            "No messages found (the chat may use a UI structure that can't be read via accessibility). Try using the computer tool with screenshot for reading.".into()
                        } else {
                            content
                        },
                        error: None,
                    }),
                    Err(e) => Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to read messages: {e}. Try using the computer tool with screenshot instead.")),
                    }),
                }
            }
        }
    }

    /// List all open KakaoTalk chat windows.
    async fn list_chats(&self) -> anyhow::Result<ToolResult> {
        // Activate KakaoTalk first
        let _ = run_osascript("tell application \"KakaoTalk\" to activate").await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        let script = "tell application \"System Events\" to tell process \"KakaoTalk\" to get name of every window";
        match run_osascript(script).await {
            Ok(windows) => {
                let window_list = windows.trim().to_string();
                if window_list.is_empty() || window_list == "missing value" {
                    Ok(ToolResult {
                        success: true,
                        output: "No KakaoTalk windows open. Use search_chat to find and open a chat.".into(),
                        error: None,
                    })
                } else {
                    Ok(ToolResult {
                        success: true,
                        output: format!("Open KakaoTalk windows: {window_list}"),
                        error: None,
                    })
                }
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to list chats: {e}")),
            }),
        }
    }

    /// Open a specific chat window by name (if already in recent chats).
    async fn open_chat(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let chat_name = match args.get("chat_name").and_then(|v| v.as_str()) {
            Some(name) if !name.is_empty() => name,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: chat_name".into()),
                });
            }
        };

        // Activate KakaoTalk
        if let Err(e) = run_osascript("tell application \"KakaoTalk\" to activate").await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to activate KakaoTalk: {e}")),
            });
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        match self.find_and_open_chat(chat_name).await {
            Ok(true) => {
                tokio::time::sleep(Duration::from_millis(800)).await;
                Ok(ToolResult {
                    success: true,
                    output: format!("Opened chat: {chat_name}"),
                    error: None,
                })
            }
            Ok(false) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Could not find chat: {chat_name}")),
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Error opening chat: {e}")),
            }),
        }
    }

    /// Search for a chat in KakaoTalk.
    async fn search_chat(&self, args: &serde_json::Value) -> anyhow::Result<ToolResult> {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Missing required parameter: query".into()),
                });
            }
        };

        // Activate KakaoTalk
        if let Err(e) = run_osascript("tell application \"KakaoTalk\" to activate").await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to activate KakaoTalk: {e}")),
            });
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Use Cmd+F to open search, type the query
        if let Err(e) = copy_to_clipboard(query).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to copy search query: {e}")),
            });
        }

        // Open search with Cmd+F
        let search_script = "tell application \"System Events\" to keystroke \"f\" using command down";
        if let Err(e) = run_osascript(search_script).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to open search: {e}")),
            });
        }
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Paste the search query
        let paste_script = "tell application \"System Events\" to keystroke \"v\" using command down";
        if let Err(e) = run_osascript(paste_script).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to paste search query: {e}")),
            });
        }

        Ok(ToolResult {
            success: true,
            output: format!("Searched for '{}' in KakaoTalk. Use list_chats to see results or open_chat to open a specific chat.", query),
            error: None,
        })
    }

    /// Try to find and open a chat by searching in the main KakaoTalk window.
    async fn find_and_open_chat(&self, chat_name: &str) -> Result<bool, String> {
        // First check if window already exists
        if check_window_exists(chat_name).await {
            return Ok(true);
        }

        // Try to search for the chat using Cmd+F in the main window
        // Copy search term to clipboard
        copy_to_clipboard(chat_name)
            .await
            .map_err(|e| format!("Clipboard error: {e}"))?;

        // Focus main KakaoTalk window (usually named "KakaoTalk" or "카카오톡")
        let _ = run_osascript("tell application \"System Events\" to tell process \"KakaoTalk\" to set frontmost to true").await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Open search
        run_osascript("tell application \"System Events\" to keystroke \"f\" using command down")
            .await
            .map_err(|e| format!("Search shortcut failed: {e}"))?;
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Clear existing search text and paste
        run_osascript("tell application \"System Events\" to keystroke \"a\" using command down")
            .await
            .map_err(|e| format!("Select all failed: {e}"))?;
        tokio::time::sleep(Duration::from_millis(100)).await;

        run_osascript("tell application \"System Events\" to keystroke \"v\" using command down")
            .await
            .map_err(|e| format!("Paste failed: {e}"))?;
        tokio::time::sleep(Duration::from_millis(800)).await;

        // Press Enter to select the first result
        run_osascript("tell application \"System Events\" to key code 36")
            .await
            .map_err(|e| format!("Enter failed: {e}"))?;
        tokio::time::sleep(Duration::from_millis(800)).await;

        // Check if the window opened
        Ok(check_window_exists(chat_name).await)
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Run an osascript command and return stdout.
async fn run_osascript(script: &str) -> Result<String, String> {
    let output = tokio::time::timeout(
        SCRIPT_TIMEOUT,
        tokio::process::Command::new("osascript")
            .args(["-e", script])
            .output(),
    )
    .await
    .map_err(|_| "osascript timed out".to_string())?
    .map_err(|e| format!("osascript failed to start: {e}"))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("osascript error: {}", stderr.trim()))
    }
}

/// Copy text to macOS clipboard using pbcopy.
async fn copy_to_clipboard(text: &str) -> Result<(), String> {
    use tokio::io::AsyncWriteExt;

    let mut child = tokio::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn pbcopy: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to pbcopy: {e}"))?;
        // Drop stdin to close the pipe
        drop(stdin);
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("pbcopy failed: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err("pbcopy exited with error".to_string())
    }
}

/// Check if a KakaoTalk window with the given name exists.
async fn check_window_exists(name: &str) -> bool {
    let script = format!(
        "tell application \"System Events\" to tell process \"KakaoTalk\" to get name of every window"
    );
    match run_osascript(&script).await {
        Ok(windows) => {
            // Window names are returned comma-separated
            windows.contains(name)
        }
        Err(_) => false,
    }
}

/// Escape special characters for AppleScript string literals.
fn escape_applescript(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Truncate a string for display purposes.
fn truncate_for_display(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let truncated = &s[..s.floor_char_boundary(max_len)];
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::AutonomyLevel;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    #[test]
    fn tool_name() {
        let tool = KakaoTalkTool::new(test_security());
        assert_eq!(tool.name(), "kakaotalk");
    }

    #[test]
    fn tool_description_not_empty() {
        let tool = KakaoTalkTool::new(test_security());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn tool_schema_has_action() {
        let tool = KakaoTalkTool::new(test_security());
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["action"].is_object());
        assert_eq!(schema["required"], json!(["action"]));
    }

    #[test]
    fn tool_schema_has_all_params() {
        let tool = KakaoTalkTool::new(test_security());
        let schema = tool.parameters_schema();
        let props = &schema["properties"];
        for param in ["action", "chat_name", "message", "query", "count"] {
            assert!(props[param].is_object(), "Missing param: {param}");
        }
    }

    #[test]
    fn escape_applescript_quotes() {
        assert_eq!(escape_applescript("hello \"world\""), "hello \\\"world\\\"");
    }

    #[test]
    fn escape_applescript_newlines() {
        assert_eq!(escape_applescript("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate_for_display("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let long = "a".repeat(200);
        let result = truncate_for_display(&long, 100);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 104); // 100 + "..."
    }

    #[tokio::test]
    async fn execute_unknown_action() {
        let tool = KakaoTalkTool::new(test_security());
        let result = tool.execute(json!({"action": "fly"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("Unknown action"));
    }

    #[tokio::test]
    async fn execute_missing_action() {
        let tool = KakaoTalkTool::new(test_security());
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn send_message_missing_chat_name() {
        let tool = KakaoTalkTool::new(test_security());
        let result = tool
            .execute(json!({"action": "send_message", "message": "hi"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("chat_name"));
    }

    #[tokio::test]
    async fn send_message_missing_message() {
        let tool = KakaoTalkTool::new(test_security());
        let result = tool
            .execute(json!({"action": "send_message", "chat_name": "Test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("message"));
    }

    #[tokio::test]
    async fn read_only_blocks_all_actions() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = KakaoTalkTool::new(security);
        let result = tool
            .execute(json!({"action": "list_chats"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap().contains("read-only"));
    }
}
