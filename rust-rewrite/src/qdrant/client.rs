use qdrant_client::Qdrant;
use qdrant_client::qdrant::HealthCheckReply;
use thiserror::Error;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum QdrantError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Initialization error: {0}")]
    InitError(String),
}

/// A thin wrapper around the official qdrant-client.
#[allow(dead_code)] // Will be used heavily in Phase 2/3
pub struct QdrantClientWrapper {
    pub client: Qdrant,
}

impl QdrantClientWrapper {
    /// Creates a new Qdrant client connection.
    pub async fn new(url: &str) -> Result<Self, QdrantError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| QdrantError::InitError(e.to_string()))?;

        info!("Created Qdrant client for URL: {}", url);

        Ok(Self { client })
    }

    /// Checks the health of the Qdrant instance.
    pub async fn check_health(&self) -> Result<HealthCheckReply, QdrantError> {
        match self.client.health_check().await {
            Ok(reply) => Ok(reply),
            Err(e) => {
                let err_msg = e.to_string();
                warn!("Qdrant health check failed: {}", err_msg);
                Err(QdrantError::ConnectionFailed(err_msg))
            }
        }
    }
}
