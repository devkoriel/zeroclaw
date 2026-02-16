//! Automatic model routing based on task domain.
//!
//! Classifies incoming messages as **technical** (coding, programming, DevOps,
//! system administration) or **general** (conversation, Q&A, creative, casual).
//!
//! Technical tasks → Claude Opus (primary model) — best-in-class for coding,
//! tool use, agentic reasoning, and long-context code analysis.
//!
//! General tasks → Gemini (fast model) — fast, cheap, great for conversation,
//! Q&A, creative writing, summarization, and general knowledge.

/// Decide which model hint to use for the given user message.
///
/// Returns `Some("hint:fast")` for general/non-technical tasks (→ Gemini),
/// or `None` for technical tasks (→ Claude Opus, the default primary model).
///
/// Each message is classified independently so that switching between general
/// chat and technical work within the same conversation routes to the right
/// model. The shared conversation history provides context continuity
/// regardless of which model handles a particular turn.
///
/// **Exception**: Short approval/denial responses ("yes", "ok", "go ahead")
/// in active conversations always go to the primary model, since they may
/// be answering a tool-approval prompt from Claude.
pub fn select_model_hint(message: &str, has_prior_exchange: bool) -> Option<&'static str> {
    let lower = message.to_ascii_lowercase();

    // Short approval/denial responses in active conversations stay on primary
    // model — they're likely answering a tool-approval or action-confirmation
    // prompt that Claude issued. Without conversation tracking we can't know
    // which model asked, so we play it safe.
    if has_prior_exchange && is_approval_response(&lower) {
        return None;
    }

    // Technical / coding / programming / DevOps → Claude Opus
    if is_technical_task(message) {
        return None;
    }

    // Everything else → Gemini (general conversation, Q&A, creative, etc.)
    Some("hint:fast")
}

/// Returns `true` if the message is about coding, programming, technical
/// operations, or requires tool use / system interaction.
///
/// This is intentionally strict: only clearly technical messages go to
/// Claude Opus. General-purpose "write", "create", "summarize" requests
/// that aren't about code go to Gemini since it handles them well and
/// is much cheaper.
fn is_technical_task(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();

    // ── Direct code indicators ──

    // Code blocks or inline code
    if message.contains("```") || message.contains("` ") {
        return true;
    }

    // File paths
    if message.contains("~/") || message.contains("/usr/") || message.contains("/etc/")
        || message.contains("/var/") || message.contains("/tmp/")
        || message.contains("C:\\") || message.contains(".rs")
        || message.contains(".py") || message.contains(".ts")
        || message.contains(".js") || message.contains(".go")
        || message.contains(".sol") || message.contains(".toml")
        || message.contains(".yaml") || message.contains(".yml")
        || message.contains(".json") || message.contains(".env")
        || message.contains(".sh") || message.contains(".tf")
    {
        return true;
    }

    // Error messages / stack traces
    if lower.contains("error:") || lower.contains("traceback")
        || lower.contains("panic at") || lower.contains("stack trace")
        || lower.contains("segfault") || lower.contains("exception")
        || lower.contains("compile error") || lower.contains("build failed")
        || lower.contains("syntax error") || lower.contains("type error")
    {
        return true;
    }

    // ── macOS app / system interaction (requires tool use) ──
    // Any request to interact with applications needs Claude for proper tool use.
    // App names are safe (no false positives). Action verbs use multi-word
    // phrases to avoid matching common English ("send me a summary", "open question").
    const APP_NAMES: &[&str] = &[
        "kakaotalk", "kakao talk", "카카오톡", "카톡",
        "imessage", "messages app", "메시지",
        "telegram", "텔레그램",
        "discord", "디스코드",
        "slack", "슬랙",
        "mail app", "mail.app",
        "safari", "chrome", "brave", "firefox",
        "finder", "terminal",
        "system preferences", "system settings",
        "notes app", "reminders",
        "spotify", "music app",
        "zoom", "facetime",
    ];
    for app in APP_NAMES {
        if lower.contains(app) {
            return true;
        }
    }
    // Action phrases that imply system interaction (multi-word to avoid false positives)
    const SYSTEM_ACTIONS: &[&str] = &[
        "send a message", "send the message", "send message",
        "send a mail", "send the mail", "send an email", "send email",
        "send it to", "send this to", "send to my",
        "take a screenshot", "take screenshot", "capture screen",
        "click on", "click the", "right click", "double click",
        "type in", "type into",
        "check my email", "check email", "check my mail", "read my email",
        "play music", "play song", "play the ",
        "메일 보내", "메시지 보내", "문자 보내",
        "앱 열어", "앱 실행", "스크린샷",
    ];
    for action in SYSTEM_ACTIONS {
        if lower.contains(action) {
            return true;
        }
    }

    // Desktop app interaction verbs — "open/launch/close/quit" followed by anything
    // implies tool use (computer tool). Match the verb prefix generously since
    // the object can be any app name ("open the elgato stream deck application").
    const APP_VERBS: &[&str] = &[
        "open the ", "open a ", "open my ",
        "launch the ", "launch a ", "launch my ",
        "close the ", "close a ", "close my ",
        "quit the ", "quit a ", "quit my ",
    ];
    for verb in APP_VERBS {
        if lower.contains(verb) {
            return true;
        }
    }
    // Memory operations need tool use → Claude
    if lower.contains("remember this") || lower.contains("memorize")
        || lower.contains("recall what") || lower.contains("기억해")
        || lower.contains("저장해")
    {
        return true;
    }

    // ── Shell / DevOps commands ──
    const SHELL_SIGNALS: &[&str] = &[
        "ssh ", "scp ", "curl ", "wget ", "docker ",
        "kubectl ", "helm ", "terraform ",
        "cargo ", "rustc ", "npm ", "pnpm ", "yarn ",
        "pip ", "pip3 ", "python3 ", "node ",
        "git ", "make ", "cmake ", "gcc ", "g++ ",
        "chmod ", "chown ", "sudo ", "systemctl ",
        "launchctl ", "brew ",
    ];
    for cmd in SHELL_SIGNALS {
        if lower.contains(cmd) {
            return true;
        }
    }

    // ── Programming / technical action verbs ──
    // These specifically relate to code/system operations, NOT general actions
    const TECH_ACTIONS: &[&str] = &[
        "debug ", "deploy ", "compile ", "refactor ",
        "implement ", "migrate ", "rebase ",
        "commit ", "push ", "merge ",
        "fix the bug", "fix this bug", "fix the error", "fix this error",
        "fix the code", "fix this code",
        "read the file", "edit the file", "open the file",
        "create a file", "create the file", "delete the file",
        "list the files", "check the logs",
        "run the test", "run tests", "run cargo", "run npm",
        "build the", "build this",
    ];
    for action in TECH_ACTIONS {
        if lower.contains(action) {
            return true;
        }
    }

    // ── Technical domain keywords ──
    const TECH_KEYWORDS: &[&str] = &[
        // Programming concepts
        "function", "variable", "class ", "struct ", "enum ",
        "interface ", "trait ", "generic", "async ", "await ",
        "callback", "closure", "iterator", "pointer", "reference",
        "mutex", "semaphore", "thread", "goroutine", "coroutine",
        // Architecture & systems
        "architecture", "microservice", "monolith", "api endpoint",
        "middleware", "load balancer", "reverse proxy",
        "database", "schema", "migration", "query", "index ",
        "cache", "redis", "postgres", "mysql", "mongodb",
        // DevOps & infrastructure
        "kubernetes", "k8s", "docker", "container",
        "terraform", "ansible", "helm", "argocd",
        "ci/cd", "cicd", "pipeline", "github actions",
        "aws ", "gcp ", "azure ",
        // Security & networking
        "vulnerability", "authentication", "authorization",
        "ssl", "tls", "certificate", "firewall",
        "race condition", "deadlock", "memory leak",
        "buffer overflow", "injection",
        // Code quality
        "refactor", "optimize", "performance", "benchmark",
        "test coverage", "unit test", "integration test",
        "linter", "clippy", "eslint", "prettier",
        // Specific technologies
        "react", "nextjs", "next.js", "vue", "angular",
        "express", "fastapi", "django", "flask",
        "rust ", "golang", "typescript", "solidity",
        "smart contract", "blockchain", "web3",
        "oracle", "scribe", "chronicle",
    ];
    for keyword in TECH_KEYWORDS {
        if lower.contains(keyword) {
            return true;
        }
    }

    // ── Code-like patterns ──

    // Contains typical code symbols in combination
    if (lower.contains("()") || lower.contains("{}") || lower.contains("[]"))
        && (lower.contains("fn ") || lower.contains("def ") || lower.contains("func ")
            || lower.contains("class ") || lower.contains("const ")
            || lower.contains("let ") || lower.contains("var "))
    {
        return true;
    }

    // CJK technical action words
    if lower.contains("코드") || lower.contains("프로그") || lower.contains("개발")  // Korean: code, program, develop
        || lower.contains("コード") || lower.contains("プログラム") // Japanese: code, program
        || lower.contains("代码") || lower.contains("程序") || lower.contains("编程") // Chinese: code, program, programming
    {
        return true;
    }

    false
}

/// Check if the message is an approval/denial response to a permission request.
fn is_approval_response(lower: &str) -> bool {
    let trimmed = lower.trim();
    matches!(
        trimmed,
        "yes" | "y" | "no" | "n" | "ok" | "okay"
            | "approved" | "approve" | "denied" | "deny"
            | "go ahead" | "go for it" | "do it"
            | "cancel" | "stop" | "abort" | "nope"
            | "yes, go ahead" | "yes please" | "yes, approved"
            | "no, don't" | "no, cancel"
    )
}

/// Extract a short subject from conversation messages.
///
/// Looks at the first user message and truncates to ~100 chars at a sentence
/// or word boundary. Used to label persistent conversations.
pub fn extract_subject(messages: &[crate::providers::ChatMessage]) -> Option<String> {
    const MAX_SUBJECT_LEN: usize = 100;

    let first_user = messages.iter().find(|m| m.role == "user")?;
    let content = first_user.content.trim();
    if content.is_empty() {
        return None;
    }

    // Strip memory context prefix if present
    let text = if let Some(idx) = content.find("[Memory context]") {
        let after = &content[idx..];
        after
            .find('\n')
            .and_then(|nl| {
                let rest = after[nl..].trim_start();
                // Skip all "- key: value" memory lines
                let mut cursor = rest;
                loop {
                    if cursor.starts_with("- ") {
                        if let Some(nl2) = cursor.find('\n') {
                            cursor = cursor[nl2..].trim_start();
                        } else {
                            return None; // only memory lines, no actual message
                        }
                    } else {
                        break;
                    }
                }
                if cursor.is_empty() {
                    None
                } else {
                    Some(cursor)
                }
            })
            .unwrap_or(content)
    } else {
        content
    };

    if text.is_empty() {
        return None;
    }

    if text.len() <= MAX_SUBJECT_LEN {
        return Some(text.to_string());
    }

    // Truncate at last sentence boundary within limit
    let truncated = &text[..text.ceil_char_boundary(MAX_SUBJECT_LEN)];
    for delim in [". ", "? ", "! "] {
        if let Some(pos) = truncated.rfind(delim) {
            return Some(truncated[..pos + 1].to_string());
        }
    }

    // No sentence boundary — truncate at last word boundary
    if let Some(pos) = truncated.rfind(' ') {
        return Some(format!("{}...", &truncated[..pos]));
    }

    Some(format!("{truncated}..."))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── General tasks → hint:fast (Gemini) ──

    #[test]
    fn general_greeting() {
        assert_eq!(select_model_hint("hello", false), Some("hint:fast"));
        assert_eq!(select_model_hint("hi there", false), Some("hint:fast"));
        assert_eq!(select_model_hint("good morning", false), Some("hint:fast"));
    }

    #[test]
    fn general_factual_question() {
        assert_eq!(
            select_model_hint("what is the capital of France?", false),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("who invented the telephone?", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_casual_chat() {
        assert_eq!(
            select_model_hint("how are you?", false),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("thanks!", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_creative_writing() {
        // "write me a poem" is creative, NOT technical → Gemini
        assert_eq!(
            select_model_hint("write me a haiku about autumn", false),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("tell me a joke", false),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("write a short story about a cat", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_summarization() {
        assert_eq!(
            select_model_hint("summarize the French Revolution", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_translation() {
        assert_eq!(
            select_model_hint("translate hello world to Korean", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_recommendation() {
        assert_eq!(
            select_model_hint("recommend a good book about history", false),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_long_non_technical() {
        // Long but clearly non-technical → Gemini
        assert_eq!(
            select_model_hint(
                "I'm planning a trip to Japan next month and wondering about the best places to visit in Tokyo. What are some must-see attractions?",
                false
            ),
            Some("hint:fast")
        );
    }

    #[test]
    fn general_empty_message() {
        assert_eq!(select_model_hint("", false), Some("hint:fast"));
    }

    // ── Technical tasks → None (Claude Opus) ──

    #[test]
    fn technical_code_block() {
        assert_eq!(
            select_model_hint("fix this code:\n```rust\nfn main() { panic!() }\n```", false),
            None
        );
    }

    #[test]
    fn technical_file_path() {
        assert_eq!(
            select_model_hint("check ~/Development/zeroclaw/src/main.rs", false),
            None
        );
    }

    #[test]
    fn technical_error_message() {
        assert_eq!(
            select_model_hint("I got this error: thread 'main' panicked at 'index out of bounds'", false),
            None
        );
    }

    #[test]
    fn technical_shell_command() {
        assert_eq!(
            select_model_hint("run cargo test to check if everything passes", false),
            None
        );
    }

    #[test]
    fn technical_deployment() {
        assert_eq!(
            select_model_hint("deploy the latest build to the staging kubernetes cluster", false),
            None
        );
    }

    #[test]
    fn technical_debugging() {
        assert_eq!(
            select_model_hint("debug why the API endpoint returns 500", false),
            None
        );
    }

    #[test]
    fn technical_file_operations() {
        assert_eq!(
            select_model_hint("create a file called utils.rs with helper functions", false),
            None
        );
    }

    #[test]
    fn technical_docker() {
        assert_eq!(
            select_model_hint("set up docker compose for the project", false),
            None
        );
    }

    #[test]
    fn technical_git_operations() {
        assert_eq!(
            select_model_hint("commit and push these changes", false),
            None
        );
    }

    #[test]
    fn technical_code_concept() {
        assert_eq!(
            select_model_hint("explain how async functions work in Rust", false),
            None
        );
    }

    #[test]
    fn technical_database() {
        assert_eq!(
            select_model_hint("how do I add an index to the users table?", false),
            None
        );
    }

    #[test]
    fn technical_security() {
        assert_eq!(
            select_model_hint("check for vulnerability in the auth module", false),
            None
        );
    }

    #[test]
    fn technical_file_extension() {
        assert_eq!(
            select_model_hint("read config.toml and update the port", false),
            None
        );
    }

    #[test]
    fn technical_cjk_code() {
        // Korean: "코드를 수정해주세요" = "Please fix the code"
        assert_eq!(
            select_model_hint("코드를 수정해주세요", false),
            None
        );
    }

    // ── Approval responses ──

    #[test]
    fn approval_in_conversation_stays_primary() {
        // In active conversation, approval goes to Claude (may be tool approval)
        assert_eq!(select_model_hint("yes", true), None);
        assert_eq!(select_model_hint("go ahead", true), None);
        assert_eq!(select_model_hint("cancel", true), None);
    }

    #[test]
    fn approval_first_message_goes_to_gemini() {
        // First message "yes" with no context → just a word, route to Gemini
        assert_eq!(select_model_hint("yes", false), Some("hint:fast"));
        assert_eq!(select_model_hint("ok", false), Some("hint:fast"));
    }

    // ── Follow-up routing (per-message, not locked) ──

    #[test]
    fn followup_general_goes_to_gemini() {
        // In active conversation, general questions still go to Gemini
        assert_eq!(
            select_model_hint("what's the weather like?", true),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("tell me about cats", true),
            Some("hint:fast")
        );
    }

    #[test]
    fn followup_technical_goes_to_claude() {
        // In active conversation, technical still goes to Claude
        assert_eq!(
            select_model_hint("now deploy it to kubernetes", true),
            None
        );
        assert_eq!(
            select_model_hint("fix the bug in main.rs", true),
            None
        );
    }

    // ── App interaction → Claude (requires computer tool) ──

    #[test]
    fn technical_open_app_by_name() {
        // "open the <app name>" must route to Claude — requires computer tool
        assert_eq!(
            select_model_hint("Open the elgato stream deck application", false),
            None
        );
        assert_eq!(
            select_model_hint("open the settings", false),
            None
        );
        assert_eq!(
            select_model_hint("launch the terminal", false),
            None
        );
        assert_eq!(
            select_model_hint("close the browser", false),
            None
        );
        assert_eq!(
            select_model_hint("quit the music app", false),
            None
        );
    }

    #[test]
    fn general_open_question_not_misrouted() {
        // "open" as adjective/noun should NOT trigger app detection
        assert_eq!(
            select_model_hint("what are some open problems in physics?", false),
            Some("hint:fast")
        );
    }

    // ── Subject extraction ──

    use crate::providers::ChatMessage;

    #[test]
    fn extract_subject_from_short_message() {
        let msgs = vec![ChatMessage::user("install neomutt with brew")];
        assert_eq!(
            extract_subject(&msgs).as_deref(),
            Some("install neomutt with brew")
        );
    }

    #[test]
    fn extract_subject_truncates_at_sentence() {
        let long = "Please install neomutt. Then configure it for my email. Also set up GPG signing for outgoing messages and make sure IMAP works.";
        let msgs = vec![ChatMessage::user(long)];
        let subject = extract_subject(&msgs).unwrap();
        assert!(subject.len() <= 110); // some slack for "..."
        assert!(subject.ends_with('.') || subject.ends_with("..."));
    }

    #[test]
    fn extract_subject_skips_system_messages() {
        let msgs = vec![
            ChatMessage::system("You are ZeroClaw"),
            ChatMessage::user("hello there"),
        ];
        assert_eq!(extract_subject(&msgs).as_deref(), Some("hello there"));
    }

    #[test]
    fn extract_subject_empty_conversation() {
        let msgs: Vec<ChatMessage> = vec![];
        assert!(extract_subject(&msgs).is_none());
    }

    #[test]
    fn extract_subject_no_user_messages() {
        let msgs = vec![ChatMessage::system("system only")];
        assert!(extract_subject(&msgs).is_none());
    }
}
