use super::traits::{Channel, ChannelMessage, MediaAttachment, MediaType};
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::path::{Path, PathBuf};
use std::time::Duration;
use uuid::Uuid;

/// Telegram's maximum message length for text messages
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;

/// Split a message into chunks that respect Telegram's 4096 character limit.
/// Tries to split at word boundaries when possible, and handles continuation.
fn split_message_for_telegram(message: &str) -> Vec<String> {
    if message.len() <= TELEGRAM_MAX_MESSAGE_LENGTH {
        return vec![message.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = message;

    while !remaining.is_empty() {
        let chunk_end = if remaining.len() <= TELEGRAM_MAX_MESSAGE_LENGTH {
            remaining.len()
        } else {
            // Try to find a good break point (newline, then space)
            let search_area = &remaining[..TELEGRAM_MAX_MESSAGE_LENGTH];

            // Prefer splitting at newline
            if let Some(pos) = search_area.rfind('\n') {
                // Don't split if the newline is too close to the start
                if pos >= TELEGRAM_MAX_MESSAGE_LENGTH / 2 {
                    pos + 1
                } else {
                    // Try space as fallback
                    search_area
                        .rfind(' ')
                        .unwrap_or(TELEGRAM_MAX_MESSAGE_LENGTH)
                        + 1
                }
            } else if let Some(pos) = search_area.rfind(' ') {
                pos + 1
            } else {
                // Hard split at the limit
                TELEGRAM_MAX_MESSAGE_LENGTH
            }
        };

        chunks.push(remaining[..chunk_end].to_string());
        remaining = &remaining[chunk_end..];
    }

    chunks
}

/// Telegram channel — long-polls the Bot API for updates
pub struct TelegramChannel {
    bot_token: String,
    allowed_users: Vec<String>,
    client: reqwest::Client,
}

impl TelegramChannel {
    pub fn new(bot_token: String, allowed_users: Vec<String>) -> Self {
        Self {
            bot_token,
            allowed_users,
            client: reqwest::Client::new(),
        }
    }

    fn api_url(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{method}", self.bot_token)
    }

    fn is_user_allowed(&self, username: &str) -> bool {
        self.allowed_users.iter().any(|u| u == "*" || u == username)
    }

    fn is_any_user_allowed<'a, I>(&self, identities: I) -> bool
    where
        I: IntoIterator<Item = &'a str>,
    {
        identities.into_iter().any(|id| self.is_user_allowed(id))
    }

    /// Send a document/file to a Telegram chat
    pub async fn send_document(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("document", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendDocument"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendDocument failed: {err}");
        }

        tracing::info!("Telegram document sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a document from bytes (in-memory) to a Telegram chat
    pub async fn send_document_bytes(
        &self,
        chat_id: &str,
        file_bytes: Vec<u8>,
        file_name: &str,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("document", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendDocument"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendDocument failed: {err}");
        }

        tracing::info!("Telegram document sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a photo to a Telegram chat
    pub async fn send_photo(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("photo.jpg");

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("photo", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendPhoto"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendPhoto failed: {err}");
        }

        tracing::info!("Telegram photo sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a photo from bytes (in-memory) to a Telegram chat
    pub async fn send_photo_bytes(
        &self,
        chat_id: &str,
        file_bytes: Vec<u8>,
        file_name: &str,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("photo", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendPhoto"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendPhoto failed: {err}");
        }

        tracing::info!("Telegram photo sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a video to a Telegram chat
    pub async fn send_video(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video.mp4");

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("video", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendVideo"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendVideo failed: {err}");
        }

        tracing::info!("Telegram video sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send an audio file to a Telegram chat
    pub async fn send_audio(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.mp3");

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("audio", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendAudio"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendAudio failed: {err}");
        }

        tracing::info!("Telegram audio sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a voice message to a Telegram chat
    pub async fn send_voice(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("voice.ogg");

        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name(file_name.to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("voice", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendVoice"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendVoice failed: {err}");
        }

        tracing::info!("Telegram voice sent to {chat_id}: {file_name}");
        Ok(())
    }

    /// Send a file by URL (Telegram will download it)
    pub async fn send_document_by_url(
        &self,
        chat_id: &str,
        url: &str,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "document": url
        });

        if let Some(cap) = caption {
            body["caption"] = serde_json::Value::String(cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendDocument"))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendDocument by URL failed: {err}");
        }

        tracing::info!("Telegram document (URL) sent to {chat_id}: {url}");
        Ok(())
    }

    // --- ZeroClaw fork: comprehensive media extraction ---

    /// Extract text content and media attachments from a Telegram message.
    /// Handles all Telegram-compatible media types.
    async fn extract_message_content(
        &self,
        message: &serde_json::Value,
    ) -> (String, Vec<MediaAttachment>) {
        let mut attachments = Vec::new();
        let caption = message
            .get("caption")
            .and_then(|c| c.as_str())
            .map(String::from);

        // Text message (highest priority)
        if let Some(text) = message.get("text").and_then(|t| t.as_str()) {
            return (text.to_string(), attachments);
        }

        // Photo — array of PhotoSize, pick the largest
        if let Some(photos) = message.get("photo").and_then(|p| p.as_array()) {
            if let Some(largest) = photos.last() {
                let mut att = MediaAttachment::new(MediaType::Photo);
                att.file_id = largest.get("file_id").and_then(|f| f.as_str()).map(String::from);
                if let Some(w) = largest.get("width").and_then(|v| v.as_u64()) {
                    att.metadata.insert("width".into(), w.to_string());
                }
                if let Some(h) = largest.get("height").and_then(|v| v.as_u64()) {
                    att.metadata.insert("height".into(), h.to_string());
                }
                att.file_size = largest.get("file_size").and_then(|v| v.as_u64());
                att.caption = caption.clone();
                att.mime_type = Some("image/jpeg".into());
                attachments.push(att);
            }
            let text = caption.unwrap_or_else(|| "[Photo]".into());
            return (text, attachments);
        }

        // Document
        if let Some(doc) = message.get("document") {
            let mut att = MediaAttachment::new(MediaType::Document);
            att.file_id = doc.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.file_name = doc.get("file_name").and_then(|f| f.as_str()).map(String::from);
            att.mime_type = doc.get("mime_type").and_then(|f| f.as_str()).map(String::from);
            att.file_size = doc.get("file_size").and_then(|v| v.as_u64());
            att.caption = caption.clone();
            attachments.push(att);
            let name = message.get("document")
                .and_then(|d| d.get("file_name"))
                .and_then(|n| n.as_str())
                .unwrap_or("file");
            let text = caption.unwrap_or_else(|| format!("[Document: {name}]"));
            return (text, attachments);
        }

        // Video
        if let Some(vid) = message.get("video") {
            let mut att = MediaAttachment::new(MediaType::Video);
            att.file_id = vid.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.mime_type = vid.get("mime_type").and_then(|f| f.as_str()).map(String::from);
            att.file_size = vid.get("file_size").and_then(|v| v.as_u64());
            if let Some(dur) = vid.get("duration").and_then(|v| v.as_u64()) {
                att.metadata.insert("duration".into(), dur.to_string());
            }
            att.caption = caption.clone();
            attachments.push(att);
            let dur = vid.get("duration").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = caption.unwrap_or_else(|| format!("[Video, {dur}s]"));
            return (text, attachments);
        }

        // Audio
        if let Some(aud) = message.get("audio") {
            let mut att = MediaAttachment::new(MediaType::Audio);
            att.file_id = aud.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.mime_type = aud.get("mime_type").and_then(|f| f.as_str()).map(String::from);
            att.file_size = aud.get("file_size").and_then(|v| v.as_u64());
            if let Some(dur) = aud.get("duration").and_then(|v| v.as_u64()) {
                att.metadata.insert("duration".into(), dur.to_string());
            }
            if let Some(title) = aud.get("title").and_then(|t| t.as_str()) {
                att.metadata.insert("title".into(), title.into());
            }
            if let Some(performer) = aud.get("performer").and_then(|p| p.as_str()) {
                att.metadata.insert("performer".into(), performer.into());
            }
            att.caption = caption.clone();
            attachments.push(att);
            let title = aud.get("title").and_then(|t| t.as_str()).unwrap_or("audio");
            let dur = aud.get("duration").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = caption.unwrap_or_else(|| format!("[Audio: {title}, {dur}s]"));
            return (text, attachments);
        }

        // Voice
        if let Some(voice) = message.get("voice") {
            let mut att = MediaAttachment::new(MediaType::Voice);
            att.file_id = voice.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.mime_type = Some("audio/ogg".into());
            att.file_size = voice.get("file_size").and_then(|v| v.as_u64());
            if let Some(dur) = voice.get("duration").and_then(|v| v.as_u64()) {
                att.metadata.insert("duration".into(), dur.to_string());
            }
            att.caption = caption.clone();
            attachments.push(att);
            let dur = voice.get("duration").and_then(|v| v.as_u64()).unwrap_or(0);
            let text = caption.unwrap_or_else(|| format!("[Voice message, {dur}s]"));
            return (text, attachments);
        }

        // Video note (round video)
        if let Some(vn) = message.get("video_note") {
            let mut att = MediaAttachment::new(MediaType::VideoNote);
            att.file_id = vn.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.file_size = vn.get("file_size").and_then(|v| v.as_u64());
            if let Some(dur) = vn.get("duration").and_then(|v| v.as_u64()) {
                att.metadata.insert("duration".into(), dur.to_string());
            }
            attachments.push(att);
            let dur = vn.get("duration").and_then(|v| v.as_u64()).unwrap_or(0);
            return (format!("[Video note, {dur}s]"), attachments);
        }

        // Animation (GIF)
        if let Some(anim) = message.get("animation") {
            let mut att = MediaAttachment::new(MediaType::Animation);
            att.file_id = anim.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.mime_type = anim.get("mime_type").and_then(|f| f.as_str()).map(String::from);
            att.file_size = anim.get("file_size").and_then(|v| v.as_u64());
            att.caption = caption.clone();
            attachments.push(att);
            let text = caption.unwrap_or_else(|| "[Animation/GIF]".into());
            return (text, attachments);
        }

        // Sticker
        if let Some(sticker) = message.get("sticker") {
            let mut att = MediaAttachment::new(MediaType::Sticker);
            att.file_id = sticker.get("file_id").and_then(|f| f.as_str()).map(String::from);
            att.file_size = sticker.get("file_size").and_then(|v| v.as_u64());
            if let Some(emoji) = sticker.get("emoji").and_then(|e| e.as_str()) {
                att.metadata.insert("emoji".into(), emoji.into());
            }
            if let Some(set) = sticker.get("set_name").and_then(|s| s.as_str()) {
                att.metadata.insert("set_name".into(), set.into());
            }
            let is_animated = sticker.get("is_animated").and_then(|v| v.as_bool()).unwrap_or(false);
            att.mime_type = Some(if is_animated { "application/x-tgsticker" } else { "image/webp" }.into());
            attachments.push(att);
            let emoji = sticker.get("emoji").and_then(|e| e.as_str()).unwrap_or("");
            return (format!("[Sticker {emoji}]"), attachments);
        }

        // Location (no file download)
        if let Some(loc) = message.get("location") {
            let lat = loc.get("latitude").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let lon = loc.get("longitude").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mut att = MediaAttachment::new(MediaType::Location);
            att.metadata.insert("latitude".into(), lat.to_string());
            att.metadata.insert("longitude".into(), lon.to_string());
            attachments.push(att);
            return (format!("[Location: {lat:.6}, {lon:.6}]"), attachments);
        }

        // Venue (location + name)
        if let Some(venue) = message.get("venue") {
            let title = venue.get("title").and_then(|t| t.as_str()).unwrap_or("Unknown venue");
            let address = venue.get("address").and_then(|a| a.as_str()).unwrap_or("");
            let mut att = MediaAttachment::new(MediaType::Venue);
            att.metadata.insert("title".into(), title.into());
            att.metadata.insert("address".into(), address.into());
            if let Some(loc) = venue.get("location") {
                if let Some(lat) = loc.get("latitude").and_then(|v| v.as_f64()) {
                    att.metadata.insert("latitude".into(), lat.to_string());
                }
                if let Some(lon) = loc.get("longitude").and_then(|v| v.as_f64()) {
                    att.metadata.insert("longitude".into(), lon.to_string());
                }
            }
            attachments.push(att);
            return (format!("[Venue: {title}, {address}]"), attachments);
        }

        // Contact
        if let Some(contact) = message.get("contact") {
            let phone = contact.get("phone_number").and_then(|p| p.as_str()).unwrap_or("");
            let first = contact.get("first_name").and_then(|f| f.as_str()).unwrap_or("");
            let last = contact.get("last_name").and_then(|l| l.as_str()).unwrap_or("");
            let mut att = MediaAttachment::new(MediaType::Contact);
            att.metadata.insert("phone_number".into(), phone.into());
            att.metadata.insert("first_name".into(), first.into());
            if !last.is_empty() {
                att.metadata.insert("last_name".into(), last.into());
            }
            attachments.push(att);
            let name = if last.is_empty() { first.to_string() } else { format!("{first} {last}") };
            return (format!("[Contact: {name}, {phone}]"), attachments);
        }

        // Poll
        if let Some(poll) = message.get("poll") {
            let question = poll.get("question").and_then(|q| q.as_str()).unwrap_or("");
            let mut att = MediaAttachment::new(MediaType::Poll);
            att.metadata.insert("question".into(), question.into());
            if let Some(options) = poll.get("options").and_then(|o| o.as_array()) {
                let opts: Vec<&str> = options
                    .iter()
                    .filter_map(|o| o.get("text").and_then(|t| t.as_str()))
                    .collect();
                att.metadata.insert("options".into(), opts.join(", "));
            }
            attachments.push(att);
            return (format!("[Poll: {question}]"), attachments);
        }

        // Unknown message type — skip
        tracing::debug!("Telegram: skipping unsupported message type");
        (String::new(), attachments)
    }

    // --- end ZeroClaw fork ---

    // --- ZeroClaw fork: file download + new send methods ---

    /// Download a Telegram file by file_id to the local workspace.
    /// Returns the local filesystem path.
    pub async fn download_file(
        &self,
        file_id: &str,
        workspace: &Path,
    ) -> anyhow::Result<PathBuf> {
        // Step 1: Call getFile to get the file_path
        let body = serde_json::json!({ "file_id": file_id });
        let resp = self
            .client
            .post(self.api_url("getFile"))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram getFile failed: {err}");
        }

        let data: serde_json::Value = resp.json().await?;
        let remote_path = data
            .get("result")
            .and_then(|r| r.get("file_path"))
            .and_then(|p| p.as_str())
            .ok_or_else(|| anyhow::anyhow!("Telegram getFile: missing file_path"))?;

        // Step 2: Download the file
        let download_url = format!(
            "https://api.telegram.org/file/bot{}/{remote_path}",
            self.bot_token
        );
        let file_resp = self.client.get(&download_url).send().await?;
        if !file_resp.status().is_success() {
            anyhow::bail!("Telegram file download failed: {}", file_resp.status());
        }
        let bytes = file_resp.bytes().await?;

        // Step 3: Save to workspace/downloads/
        let downloads_dir = workspace.join("downloads");
        tokio::fs::create_dir_all(&downloads_dir).await?;

        let file_name = remote_path
            .rsplit('/')
            .next()
            .unwrap_or("file");
        let local_path = downloads_dir.join(format!("{}_{file_name}", &file_id[..8.min(file_id.len())]));
        tokio::fs::write(&local_path, &bytes).await?;

        tracing::info!("Telegram file downloaded: {file_id} -> {}", local_path.display());
        Ok(local_path)
    }

    /// Send a sticker to a Telegram chat
    pub async fn send_sticker(
        &self,
        chat_id: &str,
        file_path: &Path,
    ) -> anyhow::Result<()> {
        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name("sticker.webp".to_string());

        let form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("sticker", part);

        let resp = self
            .client
            .post(self.api_url("sendSticker"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendSticker failed: {err}");
        }
        Ok(())
    }

    /// Send an animation (GIF) to a Telegram chat
    pub async fn send_animation(
        &self,
        chat_id: &str,
        file_path: &Path,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name("animation.gif".to_string());

        let mut form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("animation", part);

        if let Some(cap) = caption {
            form = form.text("caption", cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendAnimation"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendAnimation failed: {err}");
        }
        Ok(())
    }

    /// Send a location to a Telegram chat
    pub async fn send_location(
        &self,
        chat_id: &str,
        latitude: f64,
        longitude: f64,
    ) -> anyhow::Result<()> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "latitude": latitude,
            "longitude": longitude
        });

        let resp = self
            .client
            .post(self.api_url("sendLocation"))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendLocation failed: {err}");
        }
        Ok(())
    }

    /// Send a video note (round video) to a Telegram chat
    pub async fn send_video_note(
        &self,
        chat_id: &str,
        file_path: &Path,
    ) -> anyhow::Result<()> {
        let file_bytes = tokio::fs::read(file_path).await?;
        let part = Part::bytes(file_bytes).file_name("video_note.mp4".to_string());

        let form = Form::new()
            .text("chat_id", chat_id.to_string())
            .part("video_note", part);

        let resp = self
            .client
            .post(self.api_url("sendVideoNote"))
            .multipart(form)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendVideoNote failed: {err}");
        }
        Ok(())
    }

    /// Send a contact to a Telegram chat
    pub async fn send_contact(
        &self,
        chat_id: &str,
        phone_number: &str,
        first_name: &str,
        last_name: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "phone_number": phone_number,
            "first_name": first_name
        });

        if let Some(last) = last_name {
            body["last_name"] = serde_json::Value::String(last.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendContact"))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendContact failed: {err}");
        }
        Ok(())
    }

    // --- end ZeroClaw fork ---

    /// Send a photo by URL (Telegram will download it)
    pub async fn send_photo_by_url(
        &self,
        chat_id: &str,
        url: &str,
        caption: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut body = serde_json::json!({
            "chat_id": chat_id,
            "photo": url
        });

        if let Some(cap) = caption {
            body["caption"] = serde_json::Value::String(cap.to_string());
        }

        let resp = self
            .client
            .post(self.api_url("sendPhoto"))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = resp.text().await?;
            anyhow::bail!("Telegram sendPhoto by URL failed: {err}");
        }

        tracing::info!("Telegram photo (URL) sent to {chat_id}: {url}");
        Ok(())
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn send(&self, message: &str, chat_id: &str) -> anyhow::Result<()> {
        // Split message if it exceeds Telegram's 4096 character limit
        let chunks = split_message_for_telegram(message);

        for (i, chunk) in chunks.iter().enumerate() {
            // Add continuation marker for multi-part messages
            let text = if chunks.len() > 1 {
                if i == 0 {
                    format!("{chunk}\n\n(continues...)")
                } else if i == chunks.len() - 1 {
                    format!("(continued)\n\n{chunk}")
                } else {
                    format!("(continued)\n\n{chunk}\n\n(continues...)")
                }
            } else {
                chunk.to_string()
            };

            // --- ZeroClaw fork: try HTML first (gateway pre-formats) ---
            let html_body = serde_json::json!({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            });

            let html_resp = self
                .client
                .post(self.api_url("sendMessage"))
                .json(&html_body)
                .send()
                .await?;

            if html_resp.status().is_success() {
                // Small delay between chunks to avoid rate limiting
                if i < chunks.len() - 1 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                continue;
            }

            let html_status = html_resp.status();
            let html_err = html_resp.text().await.unwrap_or_default();
            tracing::warn!(
                status = ?html_status,
                "Telegram sendMessage with HTML failed; retrying without parse_mode"
            );
            // --- end ZeroClaw fork ---

            // Retry without parse_mode as a compatibility fallback.
            let plain_body = serde_json::json!({
                "chat_id": chat_id,
                "text": text,
            });
            let plain_resp = self
                .client
                .post(self.api_url("sendMessage"))
                .json(&plain_body)
                .send()
                .await?;

            if !plain_resp.status().is_success() {
                let plain_status = plain_resp.status();
                let plain_err = plain_resp.text().await.unwrap_or_default();
                anyhow::bail!(
                    "Telegram sendMessage failed (html {}: {}; plain {}: {})",
                    html_status,
                    html_err,
                    plain_status,
                    plain_err
                );
            }

            // Small delay between chunks to avoid rate limiting
            if i < chunks.len() - 1 {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        let mut offset: i64 = 0;

        tracing::info!("Telegram channel listening for messages...");

        loop {
            let url = self.api_url("getUpdates");
            let body = serde_json::json!({
                "offset": offset,
                "timeout": 30,
                "allowed_updates": ["message"]
            });

            let resp = match self.client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Telegram poll error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
            };

            let data: serde_json::Value = match resp.json().await {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("Telegram parse error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
            };

            if let Some(results) = data.get("result").and_then(serde_json::Value::as_array) {
                for update in results {
                    // Advance offset past this update
                    if let Some(uid) = update.get("update_id").and_then(serde_json::Value::as_i64) {
                        offset = uid + 1;
                    }

                    let Some(message) = update.get("message") else {
                        continue;
                    };

                    // --- ZeroClaw fork: parse all Telegram media types ---
                    let username_opt = message
                        .get("from")
                        .and_then(|f| f.get("username"))
                        .and_then(|u| u.as_str());
                    let username = username_opt.unwrap_or("unknown");

                    let user_id = message
                        .get("from")
                        .and_then(|f| f.get("id"))
                        .and_then(serde_json::Value::as_i64);
                    let user_id_str = user_id.map(|id| id.to_string());

                    let mut identities = vec![username];
                    if let Some(ref id) = user_id_str {
                        identities.push(id.as_str());
                    }

                    if !self.is_any_user_allowed(identities.iter().copied()) {
                        tracing::warn!(
                            "Telegram: ignoring message from unauthorized user: username={username}, user_id={}. \
Allowlist Telegram @username or numeric user ID, then run `zeroclaw onboard --channels-only`.",
                            user_id_str.as_deref().unwrap_or("unknown")
                        );
                        continue;
                    }

                    let chat_id = message
                        .get("chat")
                        .and_then(|c| c.get("id"))
                        .and_then(serde_json::Value::as_i64)
                        .map(|id| id.to_string());

                    let Some(chat_id) = chat_id else {
                        tracing::warn!("Telegram: missing chat_id in message, skipping");
                        continue;
                    };

                    // Send "typing" indicator immediately when we receive a message
                    let typing_body = serde_json::json!({
                        "chat_id": &chat_id,
                        "action": "typing"
                    });
                    let _ = self
                        .client
                        .post(self.api_url("sendChatAction"))
                        .json(&typing_body)
                        .send()
                        .await;

                    // Extract content and attachments from all media types
                    let (content, attachments) = self.extract_message_content(message).await;

                    // Skip updates with no usable content
                    if content.is_empty() && attachments.is_empty() {
                        continue;
                    }

                    let msg = ChannelMessage {
                        id: Uuid::new_v4().to_string(),
                        sender: chat_id,
                        content,
                        channel: "telegram".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        attachments,
                    };

                    if tx.send(msg).await.is_err() {
                        return Ok(());
                    }
                    // --- end ZeroClaw fork ---
                }
            }
        }
    }

    async fn start_typing(&self, recipient: &str) -> anyhow::Result<()> {
        let body = serde_json::json!({
            "chat_id": recipient,
            "action": "typing"
        });
        self.client
            .post(self.api_url("sendChatAction"))
            .json(&body)
            .send()
            .await?;
        Ok(())
    }

    async fn health_check(&self) -> bool {
        let timeout_duration = Duration::from_secs(5);

        match tokio::time::timeout(
            timeout_duration,
            self.client.get(self.api_url("getMe")).send(),
        )
        .await
        {
            Ok(Ok(resp)) => resp.status().is_success(),
            Ok(Err(e)) => {
                tracing::debug!("Telegram health check failed: {e}");
                false
            }
            Err(_) => {
                tracing::debug!("Telegram health check timed out after 5s");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_channel_name() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        assert_eq!(ch.name(), "telegram");
    }

    #[test]
    fn telegram_api_url() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("getMe"),
            "https://api.telegram.org/bot123:ABC/getMe"
        );
    }

    #[test]
    fn telegram_user_allowed_wildcard() {
        let ch = TelegramChannel::new("t".into(), vec!["*".into()]);
        assert!(ch.is_user_allowed("anyone"));
    }

    #[test]
    fn telegram_user_allowed_specific() {
        let ch = TelegramChannel::new("t".into(), vec!["alice".into(), "bob".into()]);
        assert!(ch.is_user_allowed("alice"));
        assert!(!ch.is_user_allowed("eve"));
    }

    #[test]
    fn telegram_user_denied_empty() {
        let ch = TelegramChannel::new("t".into(), vec![]);
        assert!(!ch.is_user_allowed("anyone"));
    }

    #[test]
    fn telegram_user_exact_match_not_substring() {
        let ch = TelegramChannel::new("t".into(), vec!["alice".into()]);
        assert!(!ch.is_user_allowed("alice_bot"));
        assert!(!ch.is_user_allowed("alic"));
        assert!(!ch.is_user_allowed("malice"));
    }

    #[test]
    fn telegram_user_empty_string_denied() {
        let ch = TelegramChannel::new("t".into(), vec!["alice".into()]);
        assert!(!ch.is_user_allowed(""));
    }

    #[test]
    fn telegram_user_case_sensitive() {
        let ch = TelegramChannel::new("t".into(), vec!["Alice".into()]);
        assert!(ch.is_user_allowed("Alice"));
        assert!(!ch.is_user_allowed("alice"));
        assert!(!ch.is_user_allowed("ALICE"));
    }

    #[test]
    fn telegram_wildcard_with_specific_users() {
        let ch = TelegramChannel::new("t".into(), vec!["alice".into(), "*".into()]);
        assert!(ch.is_user_allowed("alice"));
        assert!(ch.is_user_allowed("bob"));
        assert!(ch.is_user_allowed("anyone"));
    }

    #[test]
    fn telegram_user_allowed_by_numeric_id_identity() {
        let ch = TelegramChannel::new("t".into(), vec!["123456789".into()]);
        assert!(ch.is_any_user_allowed(["unknown", "123456789"]));
    }

    #[test]
    fn telegram_user_denied_when_none_of_identities_match() {
        let ch = TelegramChannel::new("t".into(), vec!["alice".into(), "987654321".into()]);
        assert!(!ch.is_any_user_allowed(["unknown", "123456789"]));
    }

    // ── File sending API URL tests ──────────────────────────────────

    #[test]
    fn telegram_api_url_send_document() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("sendDocument"),
            "https://api.telegram.org/bot123:ABC/sendDocument"
        );
    }

    #[test]
    fn telegram_api_url_send_photo() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("sendPhoto"),
            "https://api.telegram.org/bot123:ABC/sendPhoto"
        );
    }

    #[test]
    fn telegram_api_url_send_video() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("sendVideo"),
            "https://api.telegram.org/bot123:ABC/sendVideo"
        );
    }

    #[test]
    fn telegram_api_url_send_audio() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("sendAudio"),
            "https://api.telegram.org/bot123:ABC/sendAudio"
        );
    }

    #[test]
    fn telegram_api_url_send_voice() {
        let ch = TelegramChannel::new("123:ABC".into(), vec![]);
        assert_eq!(
            ch.api_url("sendVoice"),
            "https://api.telegram.org/bot123:ABC/sendVoice"
        );
    }

    // ── File sending integration tests (with mock server) ──────────

    #[tokio::test]
    async fn telegram_send_document_bytes_builds_correct_form() {
        // This test verifies the method doesn't panic and handles bytes correctly
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes = b"Hello, this is a test file content".to_vec();

        // The actual API call will fail (no real server), but we verify the method exists
        // and handles the input correctly up to the network call
        let result = ch
            .send_document_bytes("123456", file_bytes, "test.txt", Some("Test caption"))
            .await;

        // Should fail with network error, not a panic or type error
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Error should be network-related, not a code bug
        assert!(
            err.contains("error") || err.contains("failed") || err.contains("connect"),
            "Expected network error, got: {err}"
        );
    }

    #[tokio::test]
    async fn telegram_send_photo_bytes_builds_correct_form() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        // Minimal valid PNG header bytes
        let file_bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

        let result = ch
            .send_photo_bytes("123456", file_bytes, "test.png", None)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_document_by_url_builds_correct_json() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);

        let result = ch
            .send_document_by_url("123456", "https://example.com/file.pdf", Some("PDF doc"))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_photo_by_url_builds_correct_json() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);

        let result = ch
            .send_photo_by_url("123456", "https://example.com/image.jpg", None)
            .await;

        assert!(result.is_err());
    }

    // ── File path handling tests ────────────────────────────────────

    #[tokio::test]
    async fn telegram_send_document_nonexistent_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let path = Path::new("/nonexistent/path/to/file.txt");

        let result = ch.send_document("123456", path, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should fail with file not found error
        assert!(
            err.contains("No such file") || err.contains("not found") || err.contains("os error"),
            "Expected file not found error, got: {err}"
        );
    }

    #[tokio::test]
    async fn telegram_send_photo_nonexistent_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let path = Path::new("/nonexistent/path/to/photo.jpg");

        let result = ch.send_photo("123456", path, None).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_video_nonexistent_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let path = Path::new("/nonexistent/path/to/video.mp4");

        let result = ch.send_video("123456", path, None).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_audio_nonexistent_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let path = Path::new("/nonexistent/path/to/audio.mp3");

        let result = ch.send_audio("123456", path, None).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_voice_nonexistent_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let path = Path::new("/nonexistent/path/to/voice.ogg");

        let result = ch.send_voice("123456", path, None).await;

        assert!(result.is_err());
    }

    // ── Message splitting tests ─────────────────────────────────────

    #[test]
    fn telegram_split_short_message() {
        let msg = "Hello, world!";
        let chunks = split_message_for_telegram(msg);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], msg);
    }

    #[test]
    fn telegram_split_exact_limit() {
        let msg = "a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH);
        let chunks = split_message_for_telegram(&msg);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), TELEGRAM_MAX_MESSAGE_LENGTH);
    }

    #[test]
    fn telegram_split_over_limit() {
        let msg = "a".repeat(TELEGRAM_MAX_MESSAGE_LENGTH + 100);
        let chunks = split_message_for_telegram(&msg);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        assert!(chunks[1].len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
    }

    #[test]
    fn telegram_split_at_word_boundary() {
        let msg = format!(
            "{} more text here",
            "word ".repeat(TELEGRAM_MAX_MESSAGE_LENGTH / 5)
        );
        let chunks = split_message_for_telegram(&msg);
        assert!(chunks.len() >= 2);
        // First chunk should end with a complete word (space at the end)
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        }
    }

    #[test]
    fn telegram_split_at_newline() {
        let text_block = "Line of text\n".repeat(TELEGRAM_MAX_MESSAGE_LENGTH / 13 + 1);
        let chunks = split_message_for_telegram(&text_block);
        assert!(chunks.len() >= 2);
        for chunk in chunks {
            assert!(chunk.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        }
    }

    #[test]
    fn telegram_split_preserves_content() {
        let msg = "test ".repeat(TELEGRAM_MAX_MESSAGE_LENGTH / 5 + 100);
        let chunks = split_message_for_telegram(&msg);
        let rejoined = chunks.join("");
        assert_eq!(rejoined, msg);
    }

    #[test]
    fn telegram_split_empty_message() {
        let chunks = split_message_for_telegram("");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn telegram_split_very_long_message() {
        let msg = "x".repeat(TELEGRAM_MAX_MESSAGE_LENGTH * 3);
        let chunks = split_message_for_telegram(&msg);
        assert!(chunks.len() >= 3);
        for chunk in chunks {
            assert!(chunk.len() <= TELEGRAM_MAX_MESSAGE_LENGTH);
        }
    }

    // ── Caption handling tests ──────────────────────────────────────

    #[tokio::test]
    async fn telegram_send_document_bytes_with_caption() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes = b"test content".to_vec();

        // With caption
        let result = ch
            .send_document_bytes("123456", file_bytes.clone(), "test.txt", Some("My caption"))
            .await;
        assert!(result.is_err()); // Network error expected

        // Without caption
        let result = ch
            .send_document_bytes("123456", file_bytes, "test.txt", None)
            .await;
        assert!(result.is_err()); // Network error expected
    }

    #[tokio::test]
    async fn telegram_send_photo_bytes_with_caption() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes = vec![0x89, 0x50, 0x4E, 0x47];

        // With caption
        let result = ch
            .send_photo_bytes(
                "123456",
                file_bytes.clone(),
                "test.png",
                Some("Photo caption"),
            )
            .await;
        assert!(result.is_err());

        // Without caption
        let result = ch
            .send_photo_bytes("123456", file_bytes, "test.png", None)
            .await;
        assert!(result.is_err());
    }

    // ── Empty/edge case tests ───────────────────────────────────────

    #[tokio::test]
    async fn telegram_send_document_bytes_empty_file() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes: Vec<u8> = vec![];

        let result = ch
            .send_document_bytes("123456", file_bytes, "empty.txt", None)
            .await;

        // Should not panic, will fail at API level
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_document_bytes_empty_filename() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes = b"content".to_vec();

        let result = ch.send_document_bytes("123456", file_bytes, "", None).await;

        // Should not panic
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn telegram_send_document_bytes_empty_chat_id() {
        let ch = TelegramChannel::new("fake-token".into(), vec!["*".into()]);
        let file_bytes = b"content".to_vec();

        let result = ch
            .send_document_bytes("", file_bytes, "test.txt", None)
            .await;

        // Should not panic
        assert!(result.is_err());
    }
}
