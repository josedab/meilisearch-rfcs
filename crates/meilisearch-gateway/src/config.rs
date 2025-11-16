use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for distributed Meilisearch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Server configuration
    pub server: ServerConfig,
    /// Cluster configuration (optional - only for distributed mode)
    pub cluster: Option<ClusterConfig>,
    /// Sharding configuration
    pub sharding: ShardingConfig,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP bind address
    pub http_addr: String,
    /// Database path
    pub db_path: PathBuf,
    /// Server mode: "single-node" or "distributed"
    pub mode: ServerMode,
}

/// Server mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServerMode {
    /// Single-node mode (default, backward compatible)
    SingleNode,
    /// Distributed mode (requires cluster configuration)
    Distributed,
}

/// Cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// This node's ID
    pub node_id: String,
    /// Seed nodes for cluster discovery
    pub seed_nodes: Vec<String>,
    /// Raft port for consensus
    pub raft_port: u16,
}

/// Sharding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Default number of shards per index
    pub default_shard_count: usize,
    /// Default replication factor
    pub default_replication_factor: usize,
    /// Default sharding strategy: "hash", "range", or "field"
    pub default_strategy: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                http_addr: "0.0.0.0:7700".to_string(),
                db_path: PathBuf::from("/var/lib/meilisearch/data.ms"),
                mode: ServerMode::SingleNode,
            },
            cluster: None,
            sharding: ShardingConfig {
                default_shard_count: 4,
                default_replication_factor: 2,
                default_strategy: "hash".to_string(),
            },
        }
    }
}

impl DistributedConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // If distributed mode is enabled, cluster config is required
        if self.server.mode == ServerMode::Distributed && self.cluster.is_none() {
            return Err(ConfigError::InvalidConfig(
                "Distributed mode requires cluster configuration".to_string(),
            ));
        }

        // Validate seed nodes in distributed mode
        if let Some(cluster) = &self.cluster {
            if cluster.seed_nodes.is_empty() {
                return Err(ConfigError::InvalidConfig(
                    "At least one seed node is required for distributed mode".to_string(),
                ));
            }
        }

        // Validate sharding configuration
        if self.sharding.default_shard_count == 0 {
            return Err(ConfigError::InvalidConfig(
                "Shard count must be greater than 0".to_string(),
            ));
        }

        if self.sharding.default_replication_factor == 0 {
            return Err(ConfigError::InvalidConfig(
                "Replication factor must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if distributed mode is enabled
    pub fn is_distributed(&self) -> bool {
        self.server.mode == ServerMode::Distributed
    }
}

/// Configuration error
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
