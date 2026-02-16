/// Markdown → Telegram HTML converter.
///
/// Telegram's HTML parse mode supports: `<b>`, `<i>`, `<u>`, `<s>`, `<code>`,
/// `<pre>`, `<a href>`, `<blockquote>`.  Standard Markdown features like
/// `###`, `**`, `---`, `- list` have no native Telegram Markdown-v1 support,
/// so we convert them to the HTML equivalents.

/// Convert standard Markdown to Telegram-compatible HTML.
pub fn markdown_to_telegram_html(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + input.len() / 4);
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    let mut in_blockquote = false;

    while i < lines.len() {
        let line = lines[i];

        // ── Fenced code block ───────────────────────────────────
        if line.trim_start().starts_with("```") {
            let lang = line.trim_start().trim_start_matches('`').trim();
            let mut code_lines: Vec<&str> = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                code_lines.push(lines[i]);
                i += 1;
            }
            // skip closing ```
            if i < lines.len() {
                i += 1;
            }

            // Close any open blockquote before code block
            if in_blockquote {
                result.push_str("</blockquote>");
                in_blockquote = false;
            }

            let code_content = escape_html(&code_lines.join("\n"));
            if lang.is_empty() {
                result.push_str("<pre>");
                result.push_str(&code_content);
                result.push_str("</pre>\n");
            } else {
                result.push_str("<pre><code class=\"language-");
                result.push_str(&escape_html(lang));
                result.push_str("\">");
                result.push_str(&code_content);
                result.push_str("</code></pre>\n");
            }
            continue;
        }

        // ── Horizontal rule ─────────────────────────────────────
        let trimmed = line.trim();
        if (trimmed == "---" || trimmed == "***" || trimmed == "___")
            && trimmed.len() >= 3
        {
            if in_blockquote {
                result.push_str("</blockquote>");
                in_blockquote = false;
            }
            result.push_str("———\n");
            i += 1;
            continue;
        }

        // ── Blockquote ──────────────────────────────────────────
        if trimmed.starts_with("> ") || trimmed == ">" {
            if !in_blockquote {
                result.push_str("<blockquote>");
                in_blockquote = true;
            } else {
                result.push('\n');
            }
            let quote_text = trimmed.strip_prefix("> ").unwrap_or(
                trimmed.strip_prefix('>').unwrap_or(""),
            );
            result.push_str(&apply_inline_formatting(&escape_html(quote_text)));
            i += 1;
            continue;
        }

        // Close blockquote if previous lines were quotes
        if in_blockquote {
            result.push_str("</blockquote>\n");
            in_blockquote = false;
        }

        // ── Heading ─────────────────────────────────────────────
        if let Some(heading_text) = extract_heading(trimmed) {
            result.push_str("<b>");
            result.push_str(&apply_inline_formatting(&escape_html(heading_text)));
            result.push_str("</b>\n");
            i += 1;
            continue;
        }

        // ── Unordered list ──────────────────────────────────────
        if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            result.push_str("• ");
            result.push_str(&apply_inline_formatting(&escape_html(rest)));
            result.push('\n');
            i += 1;
            continue;
        }

        // ── Regular line ────────────────────────────────────────
        result.push_str(&apply_inline_formatting(&escape_html(line)));
        result.push('\n');
        i += 1;
    }

    // Close any trailing blockquote
    if in_blockquote {
        result.push_str("</blockquote>\n");
    }

    // Trim single trailing newline to avoid extra whitespace
    if result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Minimal Discord formatter — Discord handles standard Markdown natively.
/// Only converts horizontal rules (`---`) which Discord doesn't render.
pub fn markdown_to_discord(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            result.push_str("———\n");
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    // Match input: if it didn't end with \n, trim ours
    if !input.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Escape HTML entities in text content.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Extract heading text from a line like `### Foo` → `Some("Foo")`.
fn extract_heading(line: &str) -> Option<&str> {
    if line.starts_with("### ") {
        Some(line[4..].trim())
    } else if line.starts_with("## ") {
        Some(line[3..].trim())
    } else if line.starts_with("# ") {
        Some(line[2..].trim())
    } else {
        None
    }
}

/// Apply inline formatting to an already-HTML-escaped string.
fn apply_inline_formatting(escaped: &str) -> String {
    let mut s = escaped.to_string();

    // Inline code (must be before bold/italic to avoid conflicts)
    s = replace_inline_code(&s);

    // Bold **text**
    s = replace_paired_marker(&s, "**", "<b>", "</b>");

    // Strikethrough ~~text~~
    s = replace_paired_marker(&s, "~~", "<s>", "</s>");

    // Italic *text* (careful: must not match inside bold tags already processed)
    s = replace_single_star_italic(&s);

    // Italic _text_ (word-boundary: only match _word_ not mid_word)
    s = replace_underscore_italic(&s);

    // Links [text](url)
    s = replace_links(&s);

    s
}

/// Replace `` `code` `` with `<code>code</code>`.
fn replace_inline_code(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut code = String::new();
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == '`' {
                    found_close = true;
                    break;
                }
                code.push(inner);
            }
            if found_close && !code.is_empty() {
                result.push_str("<code>");
                result.push_str(&code);
                result.push_str("</code>");
            } else {
                result.push('`');
                result.push_str(&code);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Replace `**text**` with `<open>text<close>`.
fn replace_paired_marker(s: &str, marker: &str, open: &str, close: &str) -> String {
    let mut result = String::new();
    let mut rest = s;
    loop {
        let Some(start) = rest.find(marker) else {
            result.push_str(rest);
            break;
        };
        let after_open = start + marker.len();
        let Some(end) = rest[after_open..].find(marker) else {
            result.push_str(rest);
            break;
        };
        let inner = &rest[after_open..after_open + end];
        if inner.is_empty() {
            result.push_str(&rest[..after_open + end + marker.len()]);
            rest = &rest[after_open + end + marker.len()..];
            continue;
        }
        result.push_str(&rest[..start]);
        result.push_str(open);
        result.push_str(inner);
        result.push_str(close);
        rest = &rest[after_open + end + marker.len()..];
    }
    result
}

/// Replace `*text*` with `<i>text</i>`, avoiding already-processed bold tags.
fn replace_single_star_italic(s: &str) -> String {
    let mut result = String::new();
    let mut rest = s;
    loop {
        let Some(start) = rest.find('*') else {
            result.push_str(rest);
            break;
        };
        // Skip if this is a double ** (already handled by bold)
        if rest[start..].starts_with("**") {
            result.push_str(&rest[..start + 2]);
            rest = &rest[start + 2..];
            continue;
        }
        let after = start + 1;
        let Some(end) = rest[after..].find('*') else {
            result.push_str(rest);
            break;
        };
        // Skip if closing is **
        if rest[after + end..].starts_with("**") {
            result.push_str(&rest[..after + end + 2]);
            rest = &rest[after + end + 2..];
            continue;
        }
        let inner = &rest[after..after + end];
        if inner.is_empty() || inner.starts_with(' ') || inner.ends_with(' ') {
            result.push_str(&rest[..after + end + 1]);
            rest = &rest[after + end + 1..];
            continue;
        }
        result.push_str(&rest[..start]);
        result.push_str("<i>");
        result.push_str(inner);
        result.push_str("</i>");
        rest = &rest[after + end + 1..];
    }
    result
}

/// Replace `_text_` with `<i>text</i>` (word boundaries).
fn replace_underscore_italic(s: &str) -> String {
    let mut result = String::new();
    let mut rest = s;
    loop {
        let Some(start) = rest.find('_') else {
            result.push_str(rest);
            break;
        };
        // Check word boundary before _
        if start > 0 {
            let prev = rest.as_bytes()[start - 1];
            if prev.is_ascii_alphanumeric() {
                result.push_str(&rest[..start + 1]);
                rest = &rest[start + 1..];
                continue;
            }
        }
        let after = start + 1;
        let Some(end) = rest[after..].find('_') else {
            result.push_str(rest);
            break;
        };
        let inner = &rest[after..after + end];
        if inner.is_empty() || inner.starts_with(' ') || inner.ends_with(' ') {
            result.push_str(&rest[..after + end + 1]);
            rest = &rest[after + end + 1..];
            continue;
        }
        result.push_str(&rest[..start]);
        result.push_str("<i>");
        result.push_str(inner);
        result.push_str("</i>");
        rest = &rest[after + end + 1..];
    }
    result
}

/// Replace `[text](url)` with `<a href="url">text</a>`.
fn replace_links(s: &str) -> String {
    let mut result = String::new();
    let mut rest = s;
    loop {
        let Some(bracket_start) = rest.find('[') else {
            result.push_str(rest);
            break;
        };
        let Some(bracket_end) = rest[bracket_start..].find("](") else {
            result.push_str(rest);
            break;
        };
        let bracket_end = bracket_start + bracket_end;
        let Some(paren_end) = rest[bracket_end + 2..].find(')') else {
            result.push_str(rest);
            break;
        };
        let paren_end = bracket_end + 2 + paren_end;
        let text = &rest[bracket_start + 1..bracket_end];
        let url = &rest[bracket_end + 2..paren_end];
        result.push_str(&rest[..bracket_start]);
        result.push_str("<a href=\"");
        result.push_str(url);
        result.push_str("\">");
        result.push_str(text);
        result.push_str("</a>");
        rest = &rest[paren_end + 1..];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_unchanged() {
        assert_eq!(
            markdown_to_telegram_html("Hello world"),
            "Hello world"
        );
    }

    #[test]
    fn heading_levels() {
        assert_eq!(markdown_to_telegram_html("# Title"), "<b>Title</b>");
        assert_eq!(markdown_to_telegram_html("## Section"), "<b>Section</b>");
        assert_eq!(markdown_to_telegram_html("### Sub"), "<b>Sub</b>");
    }

    #[test]
    fn bold_text() {
        assert_eq!(
            markdown_to_telegram_html("This is **bold** text"),
            "This is <b>bold</b> text"
        );
    }

    #[test]
    fn italic_star() {
        assert_eq!(
            markdown_to_telegram_html("This is *italic* text"),
            "This is <i>italic</i> text"
        );
    }

    #[test]
    fn italic_underscore() {
        assert_eq!(
            markdown_to_telegram_html("This is _italic_ text"),
            "This is <i>italic</i> text"
        );
    }

    #[test]
    fn strikethrough() {
        assert_eq!(
            markdown_to_telegram_html("This is ~~struck~~ text"),
            "This is <s>struck</s> text"
        );
    }

    #[test]
    fn inline_code() {
        assert_eq!(
            markdown_to_telegram_html("Use `cargo test` here"),
            "Use <code>cargo test</code> here"
        );
    }

    #[test]
    fn fenced_code_block_no_lang() {
        let input = "```\nfn main() {}\n```";
        assert_eq!(
            markdown_to_telegram_html(input),
            "<pre>fn main() {}</pre>"
        );
    }

    #[test]
    fn fenced_code_block_with_lang() {
        let input = "```rust\nfn main() {}\n```";
        assert_eq!(
            markdown_to_telegram_html(input),
            "<pre><code class=\"language-rust\">fn main() {}</code></pre>"
        );
    }

    #[test]
    fn code_block_html_escaped() {
        let input = "```\n<div>&amp;</div>\n```";
        assert_eq!(
            markdown_to_telegram_html(input),
            "<pre>&lt;div&gt;&amp;amp;&lt;/div&gt;</pre>"
        );
    }

    #[test]
    fn link() {
        assert_eq!(
            markdown_to_telegram_html("See [docs](https://example.com)"),
            "See <a href=\"https://example.com\">docs</a>"
        );
    }

    #[test]
    fn unordered_list_dash() {
        let input = "- First\n- Second\n- Third";
        assert_eq!(
            markdown_to_telegram_html(input),
            "• First\n• Second\n• Third"
        );
    }

    #[test]
    fn unordered_list_star() {
        let input = "* First\n* Second";
        assert_eq!(
            markdown_to_telegram_html(input),
            "• First\n• Second"
        );
    }

    #[test]
    fn horizontal_rule() {
        assert_eq!(markdown_to_telegram_html("---"), "———");
        assert_eq!(markdown_to_telegram_html("***"), "———");
        assert_eq!(markdown_to_telegram_html("___"), "———");
    }

    #[test]
    fn blockquote() {
        let input = "> This is quoted\n> Second line";
        assert_eq!(
            markdown_to_telegram_html(input),
            "<blockquote>This is quoted\nSecond line</blockquote>"
        );
    }

    #[test]
    fn blockquote_ends_at_non_quote_line() {
        let input = "> Quoted\nNormal text";
        assert_eq!(
            markdown_to_telegram_html(input),
            "<blockquote>Quoted</blockquote>\nNormal text"
        );
    }

    #[test]
    fn html_entities_escaped() {
        assert_eq!(
            markdown_to_telegram_html("x < 5 & y > 3"),
            "x &lt; 5 &amp; y &gt; 3"
        );
    }

    #[test]
    fn mixed_formatting() {
        let input = "### Status Report\n\n**Server**: Running\n- CPU: `12%`\n- Memory: *low*\n\n---\n\nAll good.";
        let output = markdown_to_telegram_html(input);
        assert!(output.contains("<b>Status Report</b>"));
        assert!(output.contains("<b>Server</b>: Running"));
        assert!(output.contains("• CPU: <code>12%</code>"));
        assert!(output.contains("• Memory: <i>low</i>"));
        assert!(output.contains("———"));
        assert!(output.contains("All good."));
    }

    #[test]
    fn empty_input() {
        assert_eq!(markdown_to_telegram_html(""), "");
    }

    #[test]
    fn bold_and_italic_in_same_line() {
        assert_eq!(
            markdown_to_telegram_html("**bold** and *italic*"),
            "<b>bold</b> and <i>italic</i>"
        );
    }

    #[test]
    fn underscore_in_middle_of_word_not_italic() {
        assert_eq!(
            markdown_to_telegram_html("snake_case_name"),
            "snake_case_name"
        );
    }

    // ── Discord formatter tests ─────────────────────────────────

    #[test]
    fn discord_hr_converted() {
        assert_eq!(markdown_to_discord("---"), "———");
        assert_eq!(markdown_to_discord("***"), "———");
    }

    #[test]
    fn discord_other_markdown_preserved() {
        let input = "### Title\n**bold** and *italic*\n- list";
        assert_eq!(markdown_to_discord(input), input);
    }

    #[test]
    fn discord_empty() {
        assert_eq!(markdown_to_discord(""), "");
    }

    #[test]
    fn discord_preserves_trailing_newline() {
        assert_eq!(markdown_to_discord("hello\n"), "hello\n");
    }

    #[test]
    fn discord_no_trailing_newline() {
        assert_eq!(markdown_to_discord("hello"), "hello");
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn numbered_list_preserved() {
        let input = "1. First\n2. Second";
        assert_eq!(
            markdown_to_telegram_html(input),
            "1. First\n2. Second"
        );
    }

    #[test]
    fn unclosed_bold_not_converted() {
        assert_eq!(
            markdown_to_telegram_html("This is **unclosed"),
            "This is **unclosed"
        );
    }

    #[test]
    fn multiple_code_blocks() {
        let input = "```\nblock1\n```\ntext\n```\nblock2\n```";
        let output = markdown_to_telegram_html(input);
        assert!(output.contains("<pre>block1</pre>"));
        assert!(output.contains("<pre>block2</pre>"));
        assert!(output.contains("text"));
    }
}
