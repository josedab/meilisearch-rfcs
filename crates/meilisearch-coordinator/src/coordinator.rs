use crate::cluster::ClusterState;
use crate::error::{Error, Result};
use crate::shard_allocator::ShardAllocator;
use crate::types::{IndexMetadata, IndexUid, NodeId, NodeInfo, NodeStatus};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Cluster coordinator managing cluster state and consensus
pub struct ClusterCoordinator {
    /// Current cluster state
    state: Arc<RwLock<ClusterState>>,
    /// Shard allocator
    allocator: Arc<ShardAllocator>,
    /// This node's ID
    node_id: NodeId,
}

impl ClusterCoordinator {
    /// Create a new cluster coordinator
    pub fn new(node_id: NodeId, allocator: ShardAllocator) -> Self {
        Self {
            state: Arc::new(RwLock::new(ClusterState::new())),
            allocator: Arc::new(allocator),
            node_id,
        }
    }

    /// Get the current cluster state
    pub async fn get_state(&self) -> ClusterState {
        self.state.read().await.clone()
    }

    /// Add a node to the cluster
    pub async fn add_node(&self, node: NodeInfo) -> Result<()> {
        let mut state = self.state.write().await;
        state.add_node(node);
        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: NodeId) -> Result<NodeInfo> {
        let mut state = self.state.write().await;
        state
            .remove_node(node_id)
            .ok_or_else(|| Error::NodeNotFound(node_id.to_string()))
    }

    /// Mark a node as healthy
    pub async fn mark_node_healthy(&self, node_id: NodeId) {
        let mut state = self.state.write().await;
        if let Some(node) = state.get_node_mut(node_id) {
            node.status = NodeStatus::Healthy;
        }
    }

    /// Mark a node as down
    pub async fn mark_node_down(&self, node_id: NodeId) {
        let mut state = self.state.write().await;
        if let Some(node) = state.get_node_mut(node_id) {
            node.status = NodeStatus::Down;
        }
    }

    /// Create a new distributed index
    pub async fn create_index(
        &self,
        index_uid: IndexUid,
        shard_count: usize,
        replication_factor: usize,
        strategy: crate::types::ShardingStrategy,
    ) -> Result<IndexMetadata> {
        let state = self.state.read().await;

        // Get available nodes
        let nodes: Vec<_> = state.healthy_nodes().into_iter().cloned().collect();

        if nodes.is_empty() {
            return Err(Error::NoNodesAvailable);
        }

        // Allocate shards
        let shards = self.allocator.allocate_shards(
            &index_uid,
            shard_count,
            replication_factor,
            &nodes,
        )?;

        let metadata = IndexMetadata {
            uid: index_uid.clone(),
            shard_count,
            replication_factor,
            strategy,
            shards,
        };

        drop(state);

        // Add to cluster state
        let mut state = self.state.write().await;
        state.add_index(metadata.clone());

        Ok(metadata)
    }

    /// Delete an index
    pub async fn delete_index(&self, index_uid: &str) -> Result<IndexMetadata> {
        let mut state = self.state.write().await;
        state
            .remove_index(index_uid)
            .ok_or_else(|| Error::IndexNotFound(index_uid.to_string()))
    }

    /// Get index metadata
    pub async fn get_index(&self, index_uid: &str) -> Result<IndexMetadata> {
        let state = self.state.read().await;
        state
            .get_index(index_uid)
            .cloned()
            .ok_or_else(|| Error::IndexNotFound(index_uid.to_string()))
    }

    /// Get shards assigned to a specific node
    pub async fn get_shards_on_node(&self, node_id: NodeId) -> Vec<ShardInfo> {
        let state = self.state.read().await;
        let mut shards = Vec::new();

        for (index_uid, metadata) in &state.shard_map {
            for shard in &metadata.shards {
                if shard.primary == node_id || shard.replicas.contains(&node_id) {
                    shards.push(ShardInfo {
                        index_uid: index_uid.clone(),
                        shard_id: shard.shard_id,
                        primary: shard.primary,
                        replicas: shard.replicas.clone(),
                        state: shard.state,
                    });
                }
            }
        }

        shards
    }

    /// Promote a replica to primary
    pub async fn promote_replica_to_primary(
        &self,
        index_uid: &str,
        shard_id: usize,
        new_primary: NodeId,
    ) -> Result<()> {
        let mut state = self.state.write().await;

        let metadata = state
            .get_index_mut(index_uid)
            .ok_or_else(|| Error::IndexNotFound(index_uid.to_string()))?;

        let shard = metadata
            .shards
            .iter_mut()
            .find(|s| s.shard_id == shard_id)
            .ok_or_else(|| Error::ShardNotFound(format!("{}/{}", index_uid, shard_id)))?;

        // Remove new primary from replicas
        shard.replicas.retain(|&id| id != new_primary);

        // Add old primary to replicas if it's still healthy
        if let Some(old_primary_node) = state.get_node(shard.primary) {
            if matches!(old_primary_node.status, NodeStatus::Healthy) {
                shard.replicas.push(shard.primary);
            }
        }

        // Set new primary
        shard.primary = new_primary;

        Ok(())
    }

    /// Get cluster health
    pub async fn get_cluster_health(&self) -> crate::cluster::ClusterHealth {
        let state = self.state.read().await;
        state.health_summary()
    }
}

/// Information about a shard
#[derive(Debug, Clone)]
pub struct ShardInfo {
    pub index_uid: IndexUid,
    pub shard_id: usize,
    pub primary: NodeId,
    pub replicas: Vec<NodeId>,
    pub state: crate::types::ShardState,
}
