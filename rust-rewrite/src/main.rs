mod mcp;
mod qdrant;

use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() {
    // Initialize tracing (telemetry)
    let subscriber = FmtSubscriber::builder()
        // Use stderr for logs so stdout is clean for MCP JSON-RPC
        .with_writer(std::io::stderr)
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    info!("Starting Qdrant RAG MCP Server (Rust Rewrite)");

    // Initialize Qdrant Client (pointing to default local instance for now)
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
    match qdrant::client::QdrantClientWrapper::new(&qdrant_url).await {
        Ok(client) => {
            info!("Successfully initialized Qdrant client");
            // Check health to ensure connection is working and prevent dead code warnings
            if let Err(e) = client.check_health().await {
                tracing::warn!("Health check failed after initialization: {}", e);
            } else {
                info!("Qdrant health check passed.");
            }
        }
        Err(e) => {
            tracing::error!("Failed to initialize Qdrant client: {}", e);
            // Non-fatal for now, as we might just want to answer MCP requests
        }
    }

    // Start the MCP stdio transport loop
    // This will block the main thread and process incoming messages
    mcp::transport::run_stdio_transport().await;

    info!("MCP Server exiting.");
}
