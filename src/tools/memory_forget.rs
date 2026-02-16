use super::traits::{Tool, ToolResult};
use crate::memory::Memory;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Let the agent forget/delete a memory entry
pub struct MemoryForgetTool {
    memory: Arc<dyn Memory>,
}

impl MemoryForgetTool {
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemoryForgetTool {
    fn name(&self) -> &str {
        "memory_forget"
    }

    // --- ZeroClaw fork: clarified description to prevent LLM confusing file/folder
    // deletion requests with memory operations ---
    fn description(&self) -> &str {
        "Erase a stored memory entry by its exact key. ONLY use this for memory management — NOT for deleting files, folders, or other resources. Use the shell tool for file system operations."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key of the memory to forget"
                }
            },
            "required": ["key"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let key = args
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'key' parameter"))?;

        // --- ZeroClaw fork: guard against deleting internal bookkeeping entries ---
        if key.starts_with("webhook_msg_") || key.starts_with("assistant_resp_") {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Cannot forget internal entry '{key}'. Use the shell tool for file/folder operations."
                )),
            });
        }
        // --- end ZeroClaw fork ---

        match self.memory.forget(key).await {
            Ok(true) => Ok(ToolResult {
                success: true,
                output: format!("Forgot memory: {key}"),
                error: None,
            }),
            Ok(false) => Ok(ToolResult {
                success: true,
                output: format!("No memory found with key: {key}"),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to forget memory: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryCategory, SqliteMemory};
    use tempfile::TempDir;

    fn test_mem() -> (TempDir, Arc<dyn Memory>) {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();
        (tmp, Arc::new(mem))
    }

    #[test]
    fn name_and_schema() {
        let (_tmp, mem) = test_mem();
        let tool = MemoryForgetTool::new(mem);
        assert_eq!(tool.name(), "memory_forget");
        assert!(tool.parameters_schema()["properties"]["key"].is_object());
    }

    #[tokio::test]
    async fn forget_existing() {
        let (_tmp, mem) = test_mem();
        mem.store("temp", "temporary", MemoryCategory::Conversation)
            .await
            .unwrap();

        let tool = MemoryForgetTool::new(mem.clone());
        let result = tool.execute(json!({"key": "temp"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Forgot"));

        assert!(mem.get("temp").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn forget_nonexistent() {
        let (_tmp, mem) = test_mem();
        let tool = MemoryForgetTool::new(mem);
        let result = tool.execute(json!({"key": "nope"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("No memory found"));
    }

    #[tokio::test]
    async fn forget_missing_key() {
        let (_tmp, mem) = test_mem();
        let tool = MemoryForgetTool::new(mem);
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
    }
}
