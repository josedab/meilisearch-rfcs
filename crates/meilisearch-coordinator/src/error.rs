use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("No nodes available in the cluster")]
    NoNodesAvailable,

    #[error("Shard not found: {0}")]
    ShardNotFound(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Quorum not reached")]
    QuorumNotReached,

    #[error("Raft error: {0}")]
    Raft(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

pub type Result<T> = std::result::Result<T, Error>;
