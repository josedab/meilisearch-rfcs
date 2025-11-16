use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Node unavailable: {0}")]
    NodeUnavailable(String),

    #[error("All nodes unavailable for shard")]
    AllNodesUnavailable,

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Coordinator error: {0}")]
    Coordinator(#[from] meilisearch_coordinator::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Merge error: {0}")]
    Merge(String),

    #[error("Missing document ID")]
    MissingDocumentId,

    #[error("Invalid document ID")]
    InvalidDocumentId,

    #[error("Sharding field missing: {0}")]
    ShardingFieldMissing(String),

    #[error("Invalid sharding field: {0}")]
    InvalidShardingField(String),
}

pub type Result<T> = std::result::Result<T, Error>;
