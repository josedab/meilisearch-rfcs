use crate::types::{IndexMetadata, IndexUid, NodeId, NodeInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete state of the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    /// All nodes in the cluster
    pub nodes: HashMap<NodeId, NodeInfo>,
    /// Shard distribution for all indexes
    pub shard_map: HashMap<IndexUid, IndexMetadata>,
    /// Cluster version (monotonic counter, incremented on every state change)
    pub version: u64,
}

impl ClusterState {
    /// Create a new empty cluster state
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            shard_map: HashMap::new(),
            version: 0,
        }
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, node: NodeInfo) {
        self.nodes.insert(node.id, node);
        self.version += 1;
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<NodeInfo> {
        let node = self.nodes.remove(&node_id);
        if node.is_some() {
            self.version += 1;
        }
        node
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&NodeInfo> {
        self.nodes.get(&node_id)
    }

    /// Get mutable reference to a node
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut NodeInfo> {
        self.nodes.get_mut(&node_id)
    }

    /// Add index metadata
    pub fn add_index(&mut self, metadata: IndexMetadata) {
        self.shard_map.insert(metadata.uid.clone(), metadata);
        self.version += 1;
    }

    /// Remove index metadata
    pub fn remove_index(&mut self, index_uid: &str) -> Option<IndexMetadata> {
        let metadata = self.shard_map.remove(index_uid);
        if metadata.is_some() {
            self.version += 1;
        }
        metadata
    }

    /// Get index metadata
    pub fn get_index(&self, index_uid: &str) -> Option<&IndexMetadata> {
        self.shard_map.get(index_uid)
    }

    /// Get mutable reference to index metadata
    pub fn get_index_mut(&mut self, index_uid: &str) -> Option<&mut IndexMetadata> {
        self.shard_map.get_mut(index_uid)
    }

    /// Get all healthy nodes
    pub fn healthy_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes
            .values()
            .filter(|node| matches!(node.status, crate::types::NodeStatus::Healthy))
            .collect()
    }

    /// Get cluster health summary
    pub fn health_summary(&self) -> ClusterHealth {
        let total_nodes = self.nodes.len();
        let healthy_nodes = self
            .nodes
            .values()
            .filter(|n| matches!(n.status, crate::types::NodeStatus::Healthy))
            .count();

        let mut active_shards = 0;
        let mut relocating_shards = 0;
        let mut initializing_shards = 0;
        let mut unassigned_shards = 0;

        for metadata in self.shard_map.values() {
            for shard in &metadata.shards {
                match shard.state {
                    crate::types::ShardState::Active => active_shards += 1,
                    crate::types::ShardState::Relocating => relocating_shards += 1,
                    crate::types::ShardState::Initializing => initializing_shards += 1,
                    crate::types::ShardState::Unassigned => unassigned_shards += 1,
                }
            }
        }

        let status = if unassigned_shards > 0 {
            HealthStatus::Red
        } else if relocating_shards > 0 || initializing_shards > 0 {
            HealthStatus::Yellow
        } else if healthy_nodes == total_nodes {
            HealthStatus::Green
        } else {
            HealthStatus::Yellow
        };

        ClusterHealth {
            status,
            node_count: total_nodes,
            active_shards,
            relocating_shards,
            initializing_shards,
            unassigned_shards,
        }
    }
}

impl Default for ClusterState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All shards assigned and active
    Green,
    /// All primary shards assigned, some replicas missing
    Yellow,
    /// Some primary shards unassigned
    Red,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Green => write!(f, "green"),
            HealthStatus::Yellow => write!(f, "yellow"),
            HealthStatus::Red => write!(f, "red"),
        }
    }
}

/// Cluster health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealth {
    pub status: HealthStatus,
    pub node_count: usize,
    pub active_shards: usize,
    pub relocating_shards: usize,
    pub initializing_shards: usize,
    pub unassigned_shards: usize,
}
