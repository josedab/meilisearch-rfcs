use crate::error::Result;
use crate::gateway::Gateway;
use meilisearch_coordinator::{cluster::ClusterHealth, IndexMetadata, NodeInfo};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// API handlers for cluster management endpoints
pub struct ClusterApi {
    gateway: Arc<Gateway>,
}

impl ClusterApi {
    /// Create a new cluster API handler
    pub fn new(gateway: Arc<Gateway>) -> Self {
        Self { gateway }
    }

    /// Handle GET /_cluster/health
    pub async fn get_cluster_health(&self) -> Result<ClusterHealthResponse> {
        let health = self.gateway.get_cluster_health().await;

        Ok(ClusterHealthResponse {
            status: health.status.to_string(),
            node_count: health.node_count,
            active_shards: health.active_shards,
            relocating_shards: health.relocating_shards,
            initializing_shards: health.initializing_shards,
            unassigned_shards: health.unassigned_shards,
        })
    }

    /// Handle GET /_cluster/nodes
    pub async fn get_cluster_nodes(&self) -> Result<ClusterNodesResponse> {
        let nodes = self.gateway.get_cluster_nodes().await;

        let node_responses: Vec<_> = nodes
            .into_iter()
            .map(|node| NodeResponse {
                id: node.id.to_string(),
                address: node.address,
                status: format!("{:?}", node.status),
                shard_count: node.shard_count,
                role: format!("{:?}", node.role),
            })
            .collect();

        Ok(ClusterNodesResponse {
            nodes: node_responses,
        })
    }

    /// Handle GET /indexes/{index_uid}/_shards
    pub async fn get_index_shards(&self, index_uid: &str) -> Result<IndexShardsResponse> {
        let metadata = self.gateway.get_index_shards(index_uid).await?;

        let shard_responses: Vec<_> = metadata
            .shards
            .into_iter()
            .map(|shard| ShardResponse {
                shard_id: shard.shard_id,
                primary: shard.primary.to_string(),
                replicas: shard.replicas.iter().map(|r| r.to_string()).collect(),
                state: format!("{:?}", shard.state),
            })
            .collect();

        Ok(IndexShardsResponse {
            shard_count: metadata.shard_count,
            replication_factor: metadata.replication_factor,
            shards: shard_responses,
        })
    }
}

/// Response for GET /_cluster/health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealthResponse {
    pub status: String,
    pub node_count: usize,
    pub active_shards: usize,
    pub relocating_shards: usize,
    pub initializing_shards: usize,
    pub unassigned_shards: usize,
}

/// Response for GET /_cluster/nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNodesResponse {
    pub nodes: Vec<NodeResponse>,
}

/// Node information in API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResponse {
    pub id: String,
    pub address: String,
    pub status: String,
    pub shard_count: usize,
    pub role: String,
}

/// Response for GET /indexes/{index_uid}/_shards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexShardsResponse {
    pub shard_count: usize,
    pub replication_factor: usize,
    pub shards: Vec<ShardResponse>,
}

/// Shard information in API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardResponse {
    pub shard_id: usize,
    pub primary: String,
    pub replicas: Vec<String>,
    pub state: String,
}

/// Request for POST /indexes (with distributed sharding)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexRequest {
    pub uid: String,
    #[serde(rename = "primaryKey")]
    pub primary_key: Option<String>,
    pub sharding: Option<ShardingConfig>,
}

/// Sharding configuration for index creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    #[serde(rename = "shardCount")]
    pub shard_count: usize,
    #[serde(rename = "replicationFactor")]
    pub replication_factor: usize,
    pub strategy: String,
}
