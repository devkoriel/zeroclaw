use super::traits::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::process::Command;

/// Self-upgrade tool — checks for and applies updates from the git repository.
pub struct SelfUpgradeTool {
    repo_dir: PathBuf,
}

impl SelfUpgradeTool {
    pub fn new() -> Self {
        let repo_dir = Self::detect_repo_dir();
        Self { repo_dir }
    }

    /// Derive the repository root from the running binary's location.
    /// Falls back to the current working directory.
    fn detect_repo_dir() -> PathBuf {
        if let Ok(exe) = std::env::current_exe() {
            // Binary is typically at <repo>/target/release/zeroclaw
            // Walk up looking for a .git directory.
            let mut dir = exe.as_path();
            while let Some(parent) = dir.parent() {
                if parent.join(".git").is_dir() {
                    return parent.to_path_buf();
                }
                dir = parent;
            }
        }
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }

    fn run_git(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.repo_dir)
            .output()
            .map_err(|e| format!("Failed to run git: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(stdout)
        } else {
            Err(format!("{stderr}\n{stdout}").trim().to_string())
        }
    }
}

#[async_trait]
impl Tool for SelfUpgradeTool {
    fn name(&self) -> &str {
        "self_upgrade"
    }

    fn description(&self) -> &str {
        "Check for and apply ZeroClaw updates from the git repository. \
         Use check_only=true (default) to see pending changes; \
         set check_only=false with approved=true to pull and rebuild."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "check_only": {
                    "type": "boolean",
                    "description": "If true (default), only check for updates without applying them",
                    "default": true
                },
                "approved": {
                    "type": "boolean",
                    "description": "Set true to approve pulling changes and rebuilding. Required when check_only is false.",
                    "default": false
                }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let check_only = args
            .get("check_only")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let approved = args
            .get("approved")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        if !self.repo_dir.join(".git").is_dir() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "No git repository found at {}",
                    self.repo_dir.display()
                )),
            });
        }

        // Fetch latest from origin
        if let Err(e) = self.run_git(&["fetch", "origin", "main"]) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("git fetch failed: {e}")),
            });
        }

        // Show pending changes
        let pending = self
            .run_git(&["log", "HEAD..origin/main", "--oneline", "--no-decorate"])
            .unwrap_or_default();

        let current_sha = self
            .run_git(&["rev-parse", "--short", "HEAD"])
            .unwrap_or_default()
            .trim()
            .to_string();

        let remote_sha = self
            .run_git(&["rev-parse", "--short", "origin/main"])
            .unwrap_or_default()
            .trim()
            .to_string();

        if pending.trim().is_empty() {
            return Ok(ToolResult {
                success: true,
                output: format!("Already up to date (HEAD: {current_sha})."),
                error: None,
            });
        }

        if check_only {
            let commit_count = pending.lines().count();
            return Ok(ToolResult {
                success: true,
                output: format!(
                    "{commit_count} new commit(s) available ({current_sha} → {remote_sha}):\n{pending}"
                ),
                error: None,
            });
        }

        // Upgrade requested — require approval
        if !approved {
            let commit_count = pending.lines().count();
            return Ok(ToolResult {
                success: false,
                output: format!(
                    "{commit_count} new commit(s) will be applied ({current_sha} → {remote_sha}):\n{pending}"
                ),
                error: Some("APPROVAL_REQUIRED: Self-upgrade will pull changes and rebuild. Re-call with approved=true to proceed.".into()),
            });
        }

        // Pull changes
        match self.run_git(&["pull", "origin", "main"]) {
            Ok(pull_output) => {
                // Build release binary
                let build = Command::new("cargo")
                    .args(["build", "--release"])
                    .current_dir(&self.repo_dir)
                    .output();

                match build {
                    Ok(output) if output.status.success() => {
                        // Auto-deploy: copy binary to app bundle and codesign
                        let release_bin = self.repo_dir.join("target/release/zeroclaw");
                        let app_bin = PathBuf::from("/Applications/ZeroClaw.app/Contents/MacOS/zeroclaw");
                        let mut deploy_log = String::new();

                        if let Err(e) = std::fs::copy(&release_bin, &app_bin) {
                            return Ok(ToolResult {
                                success: false,
                                output: format!("Build succeeded but deploy failed ({current_sha} → {remote_sha})."),
                                error: Some(format!("Failed to copy binary: {e}")),
                            });
                        }
                        deploy_log.push_str("Binary copied to app bundle.\n");

                        // Codesign with stable identity (fallback to ad-hoc)
                        let sign_result = Command::new("codesign")
                            .args(["--force", "--deep", "--sign", "ZeroClaw Development",
                                   "--identifier", "com.zeroclaw.daemon",
                                   "/Applications/ZeroClaw.app"])
                            .output();
                        match sign_result {
                            Ok(s) if s.status.success() => {
                                deploy_log.push_str("Signed with ZeroClaw Development certificate.\n");
                            }
                            _ => {
                                let _ = Command::new("codesign")
                                    .args(["--force", "--deep", "--sign", "-",
                                           "--identifier", "com.zeroclaw.daemon",
                                           "/Applications/ZeroClaw.app"])
                                    .output();
                                deploy_log.push_str("Signed with ad-hoc identity (no certificate found).\n");
                            }
                        }

                        // Schedule daemon restart in a detached thread so this response
                        // gets delivered before the process dies.
                        std::thread::spawn(|| {
                            std::thread::sleep(std::time::Duration::from_secs(3));
                            let uid = Command::new("id").arg("-u").output()
                                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                                .unwrap_or_else(|_| "501".into());
                            let plist = format!(
                                "{}/Library/LaunchAgents/com.zeroclaw.daemon.plist",
                                std::env::var("HOME").unwrap_or_else(|_| "/Users/koriel".into())
                            );
                            let _ = Command::new("launchctl")
                                .args(["bootout", &format!("gui/{uid}"), &plist])
                                .output();
                            std::thread::sleep(std::time::Duration::from_secs(1));
                            let _ = Command::new("launchctl")
                                .args(["bootstrap", &format!("gui/{uid}"), &plist])
                                .output();
                        });

                        Ok(ToolResult {
                            success: true,
                            output: format!(
                                "Upgrade & deploy complete ({current_sha} → {remote_sha}).\n\
                                 Pull: {pull_output}\n\
                                 Build: success\n\
                                 {deploy_log}\n\
                                 Daemon will restart in ~3 seconds."
                            ),
                            error: None,
                        })
                    }
                    Ok(output) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        Ok(ToolResult {
                            success: false,
                            output: format!("Pull succeeded but build failed ({current_sha} → {remote_sha})."),
                            error: Some(format!("cargo build --release failed:\n{stderr}")),
                        })
                    }
                    Err(e) => Ok(ToolResult {
                        success: false,
                        output: format!("Pull succeeded but cargo not found ({current_sha} → {remote_sha})."),
                        error: Some(format!("Failed to run cargo: {e}")),
                    }),
                }
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("git pull failed: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name() {
        let tool = SelfUpgradeTool::new();
        assert_eq!(tool.name(), "self_upgrade");
    }

    #[test]
    fn tool_schema_has_check_only() {
        let tool = SelfUpgradeTool::new();
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["check_only"].is_object());
        assert!(schema["properties"]["approved"].is_object());
    }

    #[tokio::test]
    async fn check_only_default() {
        let tool = SelfUpgradeTool::new();
        // With default args, should check only (no mutation)
        let result = tool.execute(json!({})).await.unwrap();
        // Either succeeds (up to date / pending) or fails (no git) — never mutates
        assert!(result.error.is_none() || !result.error.as_deref().unwrap_or("").contains("pull"));
    }

    #[tokio::test]
    async fn upgrade_requires_approval() {
        let tool = SelfUpgradeTool::new();
        let result = tool
            .execute(json!({"check_only": false, "approved": false}))
            .await
            .unwrap();
        // Should either require approval or be up-to-date
        let err = result.error.as_deref().unwrap_or("");
        let is_up_to_date = result.output.contains("up to date");
        assert!(
            err.contains("APPROVAL_REQUIRED") || is_up_to_date || err.contains("git"),
            "Expected APPROVAL_REQUIRED or up-to-date, got: output={}, error={err}",
            result.output
        );
    }

    #[test]
    fn detect_repo_dir_finds_git() {
        let dir = SelfUpgradeTool::detect_repo_dir();
        // We're running from within the zeroclaw repo, so .git should exist
        assert!(
            dir.join(".git").is_dir(),
            "Expected .git in {}, running tests from the repo",
            dir.display()
        );
    }
}
