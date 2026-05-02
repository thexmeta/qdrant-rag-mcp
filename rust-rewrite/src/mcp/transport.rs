use futures::{SinkExt, StreamExt};
use serde_json::json;
use tokio::io::{stdin, stdout};
use tokio_util::codec::{FramedRead, FramedWrite, LinesCodec};
use tracing::{debug, error, info};

use crate::mcp::types::{JsonRpcRequest, JsonRpcResponse};

/// Runs the MCP standard input/output transport loop.
/// This reads lines from stdin using LinesCodec to prevent buffer deadlocks,
/// parses them as JSON-RPC, and dispatches them to the handler.
pub async fn run_stdio_transport() {
    // Initialize strict framing. 10MB limit for incoming JSON-RPC payloads
    let mut reader = FramedRead::new(stdin(), LinesCodec::new_with_max_length(1024 * 1024 * 10));
    let mut writer = FramedWrite::new(stdout(), LinesCodec::new());

    info!("MCP stdio transport initialized.");

    while let Some(result) = reader.next().await {
        match result {
            Ok(line) => {
                debug!("Received payload: {}", line);
                let request: Result<JsonRpcRequest, _> = serde_json::from_str(&line);

                match request {
                    Ok(req) => {
                        let id = req.id.clone();

                        // Dispatch JSON-RPC payload to async handler
                        // We use spawn to handle requests concurrently
                        let response =
                            tokio::spawn(
                                async move { crate::mcp::server::handle_request(req).await },
                            )
                            .await;

                        match response {
                            Ok(Some(resp)) => {
                                let resp_str = serde_json::to_string(&resp).unwrap();
                                if let Err(e) = writer.send(resp_str).await {
                                    error!("Failed to write response to stdout: {}", e);
                                }
                            }
                            Ok(None) => {
                                // Notification, no response needed
                            }
                            Err(e) => {
                                error!("Task panicked handling request: {}", e);
                                if let Some(req_id) = id {
                                    let err_resp = JsonRpcResponse::error(
                                        req_id,
                                        -32603,
                                        "Internal error",
                                        Some(json!(e.to_string())),
                                    );
                                    let resp_str = serde_json::to_string(&err_resp).unwrap();
                                    let _ = writer.send(resp_str).await;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse JSON-RPC request: {}", e);
                        // Send parse error
                        let err_resp = JsonRpcResponse::error(
                            serde_json::Value::Null,
                            -32700,
                            "Parse error",
                            Some(json!(e.to_string())),
                        );
                        let resp_str = serde_json::to_string(&err_resp).unwrap();
                        let _ = writer.send(resp_str).await;
                    }
                }
            }
            Err(e) => {
                error!("I/O Framing error on stdin: {:?}", e);
                break; // Break the loop on fatal I/O errors
            }
        }
    }
}
