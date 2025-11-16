use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a node in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// Status of a node in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is healthy and serving requests
    Healthy,
    /// Node is degraded but still operational
    Degraded,
    /// Node is down and not responding
    Down,
    /// Node is joining the cluster
    Joining,
    /// Node is leaving the cluster
    Leaving,
}

/// Resource capacity of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// Available memory in bytes
    pub available_memory: u64,
    /// Available disk space in bytes
    pub available_disk: u64,
    /// Number of CPU cores
    pub cpu_cores: usize,
}

/// Information about a node in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique identifier for the node
    pub id: NodeId,
    /// Network address (host:port)
    pub address: String,
    /// Current status of the node
    pub status: NodeStatus,
    /// Resource capacity
    pub capacity: ResourceCapacity,
    /// Number of shards currently on this node
    pub shard_count: usize,
    /// Node role (data, gateway, coordinator)
    pub role: NodeRole,
}

/// Role of a node in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Data node - stores shards
    Data,
    /// Gateway node - routes queries
    Gateway,
    /// Coordinator node - manages cluster state
    Coordinator,
    /// Combined role - all of the above
    All,
}

/// Assignment of a shard to nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Shard identifier
    pub shard_id: usize,
    /// Primary node for this shard
    pub primary: NodeId,
    /// Replica nodes for this shard
    pub replicas: Vec<NodeId>,
    /// Current state of the shard
    pub state: ShardState,
}

/// State of a shard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    /// Shard is active and serving requests
    Active,
    /// Shard is initializing
    Initializing,
    /// Shard is relocating to another node
    Relocating,
    /// Shard is unassigned
    Unassigned,
}

/// Index UID type
pub type IndexUid = String;

/// Cluster metadata for an index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Index unique identifier
    pub uid: IndexUid,
    /// Number of shards
    pub shard_count: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Sharding strategy
    pub strategy: ShardingStrategy,
    /// Shard assignments
    pub shards: Vec<ShardAssignment>,
}

/// Sharding strategy for distributing documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding (default)
    Hash,
    /// Range-based sharding
    Range { ranges: Vec<(String, usize)> },
    /// Field-based sharding
    Field { field: String },
}
