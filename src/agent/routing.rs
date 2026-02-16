//! Automatic model routing based on task complexity.
//!
//! Classifies incoming messages as simple or complex to route them to the
//! most cost-effective model. Simple conversational queries go to the fast
//! model (e.g. Gemini), while complex tasks requiring tool use, code generation,
//! or multi-step reasoning stay on the primary model (e.g. Claude Opus).

/// Decide which model string to pass to the provider for a given user message.
///
/// Returns `Some("hint:fast")` for simple tasks that can be handled by the
/// fast/cheap model, or `None` to use the default (most capable) model.
///
/// When `has_prior_exchange` is true (the conversation already has assistant
/// messages), the primary model is always used to maintain context continuity.
/// Only brand-new conversations with simple messages are routed to the fast model.
pub fn select_model_hint(message: &str, has_prior_exchange: bool) -> Option<&'static str> {
    // If there's been a prior exchange in this conversation, always use the
    // primary model. Short follow-ups like "temp", "yes", "that one" are
    // context-dependent answers that the fast model can't handle without
    // the full conversation history and tool-use capabilities.
    if has_prior_exchange {
        return None; // always use primary model for follow-ups
    }

    if is_complex_task(message) {
        None // use default (Claude Opus)
    } else {
        Some("hint:fast") // route to Gemini
    }
}

/// Heuristic complexity classifier. Returns `true` if the message likely
/// requires deep reasoning, tool use, or multi-step execution.
fn is_complex_task(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    let len = message.len();

    // Long messages are usually complex requests
    if len > 300 {
        return true;
    }

    // Code blocks or file paths signal technical tasks
    if message.contains("```") || message.contains("~/") || message.contains("/usr/") {
        return true;
    }

    // Error messages / stack traces
    if lower.contains("error:") || lower.contains("traceback") || lower.contains("panic at") {
        return true;
    }

    // Approval responses should stay on same model that asked
    if is_approval_response(&lower) {
        return true;
    }

    // Tool-use / action signals
    const ACTION_SIGNALS: &[&str] = &[
        "create ", "write ", "build ", "deploy ", "install ",
        "configure ", "fix ", "debug ", "implement ", "refactor ",
        "delete ", "remove ", "run ", "execute ", "compile ",
        "test ", "update ", "upgrade ", "migrate ", "restart ",
        "start ", "stop ", "kill ", "mkdir ", "touch ",
        "commit ", "push ", "pull ", "merge ", "rebase ",
        "ssh ", "scp ", "curl ", "wget ", "docker ",
        "chmod ", "chown ", "sudo ",
        "read the file", "open the file", "edit the file",
        "read my ", "read email", "read mail", "check my ",
        "send email", "send mail", "send a message",
        "show me the", "list the files", "check the",
        "make a", "set up", "set the",
        "add a", "add the",
        "search for", "find my ", "download ", "upload ",
        "schedule ", "remind me", "summarize ", "analyze ",
    ];

    for signal in ACTION_SIGNALS {
        if lower.contains(signal) {
            return true;
        }
    }

    // Multi-step patterns
    if lower.contains("step 1")
        || lower.contains("first,")
        || lower.contains("then ")
        || lower.contains("after that")
        || lower.contains("and then")
        || lower.contains("finally ")
    {
        return true;
    }

    // Numbered lists (1. ... 2. ...)
    if lower.contains("\n1.") || lower.contains("\n2.") {
        return true;
    }

    // Technical keywords that imply complex reasoning
    const COMPLEX_KEYWORDS: &[&str] = &[
        "architecture", "refactor", "optimize", "performance",
        "security", "vulnerability", "database", "schema",
        "terraform", "kubernetes", "docker", "helm",
        "api endpoint", "middleware", "authentication",
        "race condition", "deadlock", "memory leak",
        "cicd", "ci/cd", "pipeline", "workflow",
        // System integration tasks requiring tool use
        "email", "calendar", "notification", "reminder",
        "screenshot", "clipboard", "desktop", "application",
        "browser", "safari", "chrome", "firefox",
        "important", "missed", "unread",
        "自動", "실행", // CJK action words
    ];

    for keyword in COMPLEX_KEYWORDS {
        if lower.contains(keyword) {
            return true;
        }
    }

    // Otherwise, it's likely a simple conversational query → Gemini
    false
}

/// Check if the message is an approval/denial response to a permission request.
/// These must stay on the same model (Claude) that generated the approval request.
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

    // ── Simple tasks → hint:fast ──

    #[test]
    fn simple_greeting() {
        assert_eq!(select_model_hint("hello", false), Some("hint:fast"));
        assert_eq!(select_model_hint("hi there", false), Some("hint:fast"));
        assert_eq!(select_model_hint("good morning", false), Some("hint:fast"));
    }

    #[test]
    fn simple_factual_question() {
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
    fn simple_short_question() {
        assert_eq!(
            select_model_hint("how are you?", false),
            Some("hint:fast")
        );
        assert_eq!(
            select_model_hint("thanks!", false),
            Some("hint:fast")
        );
    }

    // ── Complex tasks → None (default Claude) ──

    #[test]
    fn complex_file_operation() {
        assert_eq!(
            select_model_hint("create a new file called utils.rs with helper functions", false),
            None
        );
    }

    #[test]
    fn complex_shell_command() {
        assert_eq!(
            select_model_hint("run cargo test to check if everything passes", false),
            None
        );
    }

    #[test]
    fn complex_code_block() {
        assert_eq!(
            select_model_hint("fix this code:\n```rust\nfn main() { panic!() }\n```", false),
            None
        );
    }

    #[test]
    fn complex_error_message() {
        assert_eq!(
            select_model_hint("I got this error: thread 'main' panicked at 'index out of bounds'", false),
            None
        );
    }

    #[test]
    fn complex_multi_step() {
        assert_eq!(
            select_model_hint("first, read the config file, then update the port to 8080", false),
            None
        );
    }

    #[test]
    fn complex_long_message() {
        let long = "a".repeat(301);
        assert_eq!(select_model_hint(&long, false), None);
    }

    #[test]
    fn complex_deployment() {
        assert_eq!(
            select_model_hint("deploy the latest build to the staging kubernetes cluster", false),
            None
        );
    }

    #[test]
    fn complex_debugging() {
        assert_eq!(
            select_model_hint("debug why the API endpoint returns 500", false),
            None
        );
    }

    #[test]
    fn complex_file_path() {
        assert_eq!(
            select_model_hint("check ~/Development/zeroclaw/src/main.rs", false),
            None
        );
    }

    // ── Approval responses → None (stay on Claude) ──

    #[test]
    fn approval_yes() {
        assert_eq!(select_model_hint("yes", false), None);
        assert_eq!(select_model_hint("Yes", false), None);
        assert_eq!(select_model_hint("y", false), None);
        assert_eq!(select_model_hint("go ahead", false), None);
        assert_eq!(select_model_hint("approved", false), None);
    }

    #[test]
    fn approval_no() {
        assert_eq!(select_model_hint("no", false), None);
        assert_eq!(select_model_hint("cancel", false), None);
        assert_eq!(select_model_hint("stop", false), None);
    }

    // ── Edge cases ──

    #[test]
    fn empty_message() {
        // Empty is not complex
        assert_eq!(select_model_hint("", false), Some("hint:fast"));
    }

    #[test]
    fn borderline_action_in_question() {
        // "write" appears → complex
        assert_eq!(
            select_model_hint("write me a haiku about rust", false),
            None
        );
    }

    #[test]
    fn technical_keyword_triggers_complex() {
        assert_eq!(
            select_model_hint("explain the security implications", false),
            None
        );
        assert_eq!(
            select_model_hint("how does the authentication work?", false),
            None
        );
    }

    // ── Follow-up messages in active conversations → None (primary model) ──

    #[test]
    fn followup_short_answer_uses_primary_model() {
        // "temp" would normally route to fast model, but in an active
        // conversation it's a follow-up answer and must use primary model.
        assert_eq!(select_model_hint("temp", true), None);
        assert_eq!(select_model_hint("that one", true), None);
        assert_eq!(select_model_hint("the first option", true), None);
    }

    #[test]
    fn followup_simple_greeting_uses_primary_model() {
        // Even greetings should stay on primary in an active conversation
        assert_eq!(select_model_hint("hello", true), None);
        assert_eq!(select_model_hint("thanks!", true), None);
    }

    #[test]
    fn followup_empty_uses_primary_model() {
        assert_eq!(select_model_hint("", true), None);
    }

    #[test]
    fn followup_complex_still_uses_primary_model() {
        // Complex messages in follow-ups also stay on primary (no change)
        assert_eq!(select_model_hint("create a file called test.rs", true), None);
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
