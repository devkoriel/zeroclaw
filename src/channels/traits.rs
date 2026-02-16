use async_trait::async_trait;
use std::collections::HashMap;

// --- ZeroClaw fork: media type support for all Telegram-compatible media ---

/// Media types supported across messaging platforms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaType {
    Photo,
    Document,
    Video,
    Audio,
    Voice,
    VideoNote,
    Animation,
    Sticker,
    Location,
    Contact,
    Poll,
    Venue,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Photo => write!(f, "Photo"),
            Self::Document => write!(f, "Document"),
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Voice => write!(f, "Voice"),
            Self::VideoNote => write!(f, "VideoNote"),
            Self::Animation => write!(f, "Animation"),
            Self::Sticker => write!(f, "Sticker"),
            Self::Location => write!(f, "Location"),
            Self::Contact => write!(f, "Contact"),
            Self::Poll => write!(f, "Poll"),
            Self::Venue => write!(f, "Venue"),
        }
    }
}

impl MediaType {
    /// Whether this media type represents a downloadable file.
    pub fn is_file(&self) -> bool {
        matches!(
            self,
            Self::Photo
                | Self::Document
                | Self::Video
                | Self::Audio
                | Self::Voice
                | Self::VideoNote
                | Self::Animation
                | Self::Sticker
        )
    }

    /// Whether this media type is an image suitable for vision models.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Photo | Self::Sticker | Self::Animation)
    }
}

/// A media attachment from a messaging platform.
#[derive(Debug, Clone)]
pub struct MediaAttachment {
    pub media_type: MediaType,
    /// Platform-specific file identifier (e.g. Telegram file_id).
    pub file_id: Option<String>,
    /// Local filesystem path after download.
    pub file_path: Option<String>,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub file_size: Option<u64>,
    pub caption: Option<String>,
    /// Extra structured data (lat/lon, phone, duration, etc.).
    pub metadata: HashMap<String, String>,
}

impl MediaAttachment {
    pub fn new(media_type: MediaType) -> Self {
        Self {
            media_type,
            file_id: None,
            file_path: None,
            file_name: None,
            mime_type: None,
            file_size: None,
            caption: None,
            metadata: HashMap::new(),
        }
    }
}

// --- end ZeroClaw fork ---

/// A message received from or sent to a channel
#[derive(Debug, Clone, Default)]
pub struct ChannelMessage {
    pub id: String,
    pub sender: String,
    pub content: String,
    pub channel: String,
    pub timestamp: u64,
    // --- ZeroClaw fork ---
    pub attachments: Vec<MediaAttachment>,
    // --- end ZeroClaw fork ---
}

/// Core channel trait — implement for any messaging platform
#[async_trait]
pub trait Channel: Send + Sync {
    /// Human-readable channel name
    fn name(&self) -> &str;

    /// Send a message through this channel
    async fn send(&self, message: &str, recipient: &str) -> anyhow::Result<()>;

    /// Start listening for incoming messages (long-running)
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;

    /// Check if channel is healthy
    async fn health_check(&self) -> bool {
        true
    }

    /// Signal that the bot is processing a response (e.g. "typing" indicator).
    /// Implementations should repeat the indicator as needed for their platform.
    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Stop any active typing indicator.
    async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyChannel;

    #[async_trait]
    impl Channel for DummyChannel {
        fn name(&self) -> &str {
            "dummy"
        }

        async fn send(&self, _message: &str, _recipient: &str) -> anyhow::Result<()> {
            Ok(())
        }

        async fn listen(
            &self,
            tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        ) -> anyhow::Result<()> {
            tx.send(ChannelMessage {
                id: "1".into(),
                sender: "tester".into(),
                content: "hello".into(),
                channel: "dummy".into(),
                timestamp: 123,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
        }
    }

    #[test]
    fn channel_message_clone_preserves_fields() {
        let message = ChannelMessage {
            id: "42".into(),
            sender: "alice".into(),
            content: "ping".into(),
            channel: "dummy".into(),
            timestamp: 999,
            ..Default::default()
        };

        let cloned = message.clone();
        assert_eq!(cloned.id, "42");
        assert_eq!(cloned.sender, "alice");
        assert_eq!(cloned.content, "ping");
        assert_eq!(cloned.channel, "dummy");
        assert_eq!(cloned.timestamp, 999);
    }

    #[tokio::test]
    async fn default_trait_methods_return_success() {
        let channel = DummyChannel;

        assert!(channel.health_check().await);
        assert!(channel.start_typing("bob").await.is_ok());
        assert!(channel.stop_typing("bob").await.is_ok());
        assert!(channel.send("hello", "bob").await.is_ok());
    }

    #[tokio::test]
    async fn listen_sends_message_to_channel() {
        let channel = DummyChannel;
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        channel.listen(tx).await.unwrap();

        let received = rx.recv().await.expect("message should be sent");
        assert_eq!(received.sender, "tester");
        assert_eq!(received.content, "hello");
        assert_eq!(received.channel, "dummy");
    }
}
