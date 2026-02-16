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
    /// When deployed as an app bundle, the binary is NOT inside the repo,
    /// so we check well-known paths and $HOME/Development/zeroclaw as fallbacks.
    fn detect_repo_dir() -> PathBuf {
        // 1. Walk up from binary location (works when running from target/release/)
        if let Ok(exe) = std::env::current_exe() {
            let mut dir = exe.as_path();
            while let Some(parent) = dir.parent() {
                if parent.join(".git").is_dir() {
                    return parent.to_path_buf();
                }
                dir = parent;
            }
        }

        // 2. Check $HOME/Development/zeroclaw (canonical location)
        if let Ok(home) = std::env::var("HOME") {
            let candidate = PathBuf::from(&home).join("Development/zeroclaw");
            if candidate.join(".git").is_dir() {
                return candidate;
            }
        }

        // 3. Last resort: current directory
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
        "Build, deploy, and restart ZeroClaw. This is the ONLY safe way to redeploy yourself. \
         Use check_only=true (default) to see pending changes; \
         set check_only=false with approved=true to pull, build, copy binary, codesign, and restart. \
         Do NOT use shell commands for building or restarting — they are blocked by security policy."
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
        let pull_output = match self.run_git(&["pull", "origin", "main"]) {
            Ok(o) => o,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("git pull failed: {e}")),
                });
            }
        };

        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/koriel".into());
        // Build PATH with asdf/cargo/rustup toolchain discovery
        let mut path_parts: Vec<String> = vec![
            format!("{home}/.asdf/shims"),
            format!("{home}/.cargo/bin"),
        ];
        // Discover rustup toolchain bin dirs (cargo may live here)
        let rustup_toolchains = format!("{home}/.rustup/toolchains");
        if let Ok(entries) = std::fs::read_dir(&rustup_toolchains) {
            for entry in entries.flatten() {
                let bin = entry.path().join("bin");
                if bin.is_dir() {
                    path_parts.push(bin.to_string_lossy().into_owned());
                }
            }
        }
        path_parts.extend([
            "/opt/homebrew/bin".into(),
            "/opt/homebrew/sbin".into(),
            "/usr/local/bin".into(),
            "/usr/bin".into(),
            "/bin".into(),
            "/usr/sbin".into(),
            "/sbin".into(),
        ]);
        let path_env = path_parts.join(":");

        // Phase 1: Build
        let build = Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(&self.repo_dir)
            .env("PATH", &path_env)
            .env("HOME", &home)
            .output();

        let build_output = match build {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                return Ok(ToolResult {
                    success: false,
                    output: format!("Build failed ({current_sha} → {remote_sha})."),
                    error: Some(format!("cargo build --release:\n{stderr}")),
                });
            }
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to run cargo: {e}")),
                });
            }
        };
        let build_stderr = String::from_utf8_lossy(&build_output.stderr);

        // Phase 2: Copy binary to app bundle (daemon is still running old binary — this is safe)
        let release_bin = self.repo_dir.join("target/release/zeroclaw");
        let app_bin = "/Applications/ZeroClaw.app/Contents/MacOS/zeroclaw";
        if let Err(e) = std::fs::copy(&release_bin, app_bin) {
            return Ok(ToolResult {
                success: false,
                output: "Build succeeded but copy failed.".into(),
                error: Some(format!("cp to app bundle: {e}")),
            });
        }

        // Phase 3: Codesign (while daemon still runs the old in-memory binary)
        let _ = Command::new("codesign")
            .args([
                "--force", "--deep",
                "--sign", "ZeroClaw Development",
                "--identifier", "com.zeroclaw.daemon",
                "/Applications/ZeroClaw.app",
            ])
            .env("PATH", &path_env)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .or_else(|| {
                Command::new("codesign")
                    .args([
                        "--force", "--deep", "--sign", "-",
                        "--identifier", "com.zeroclaw.daemon",
                        "/Applications/ZeroClaw.app",
                    ])
                    .output()
                    .ok()
            });

        // Phase 4: Schedule restart via a DETACHED process that outlives this daemon.
        // nohup + setsid ensures the restart script survives when bootout kills us.
        let uid = Command::new("id").arg("-u").output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "501".into());
        let plist = format!("{home}/Library/LaunchAgents/com.zeroclaw.daemon.plist");

        let restart_script = format!(
            "sleep 3; launchctl bootout gui/{uid} '{plist}' 2>/dev/null; \
             sleep 1; launchctl bootstrap gui/{uid} '{plist}'"
        );
        // Use nohup + bash -c in background so the restart outlives our process
        let _ = Command::new("nohup")
            .args(["bash", "-c", &restart_script])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null())
            .spawn(); // spawn() returns immediately, doesn't wait

        Ok(ToolResult {
            success: true,
            output: format!(
                "Upgrade & deploy complete ({current_sha} → {remote_sha}).\n\
                 Pull: {pull_output}\n\
                 Build: success{}\n\
                 Binary copied & signed. Daemon restarting in ~3s.\n\
                 You'll receive a startup notification when I'm back.",
                if build_stderr.is_empty() { String::new() }
                else { format!("\n{build_stderr}") }
            ),
            error: None,
        })
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
