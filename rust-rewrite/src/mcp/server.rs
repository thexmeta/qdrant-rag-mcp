use serde_json::json;
use tracing::info;

use crate::mcp::types::{JsonRpcRequest, JsonRpcResponse};

/// Main handler for incoming JSON-RPC requests.
pub async fn handle_request(req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    info!("Handling method: {}", req.method);

    let id = match req.id {
        Some(id) => id,
        None => {
            info!("Received notification: {}", req.method);
            return None; // Notifications don't get a response
        }
    };

    // Routing skeleton
    match req.method.as_str() {
        "initialize" => {
            // Basic MCP initialization response
            let result = json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": true
                    },
                    "resources": {
                        "subscribe": true,
                        "listChanged": true
                    },
                    "prompts": {
                        "listChanged": true
                    }
                },
                "serverInfo": {
                    "name": "qdrant-rag-mcp-rust",
                    "version": "0.1.0"
                }
            });
            Some(JsonRpcResponse::success(id, result))
        }
        "notifications/initialized" => {
            info!("Client initialized.");
            None
        }
        "ping" => Some(JsonRpcResponse::success(id, json!({}))),
        // Fallback for unhandled methods
        _ => Some(JsonRpcResponse::error(
            id,
            -32601,
            format!("Method not found: {}", req.method),
            None,
        )),
    }
}
