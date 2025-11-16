use crate::error::Result;
use meilisearch_coordinator::{ClusterCoordinator, NodeId};
use std::sync::Arc;
use std::time::Duration;

/// Health monitor for cluster nodes
pub struct HealthMonitor {
    /// Cluster coordinator
    coordinator: Arc<ClusterCoordinator>,
    /// Heartbeat interval
    heartbeat_interval: Duration,
    /// Failure threshold (consecutive failures before marking node down)
    failure_threshold: usize,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(
        coordinator: Arc<ClusterCoordinator>,
        heartbeat_interval: Duration,
        failure_threshold: usize,
    ) -> Self {
        Self {
            coordinator,
            heartbeat_interval,
            failure_threshold,
        }
    }

    /// Start monitoring a node's health
    pub async fn monitor_node_health(&self, node_id: NodeId) {
        let mut consecutive_failures = 0;
        let mut interval = tokio::time::interval(self.heartbeat_interval);

        loop {
            interval.tick().await;

            match self.send_heartbeat(node_id).await {
                Ok(_) => {
                    consecutive_failures = 0;
                    self.coordinator.mark_node_healthy(node_id).await;
                }
                Err(e) => {
                    consecutive_failures += 1;
                    tracing::warn!(
                        "Heartbeat to node {} failed: {} (attempt {})",
                        node_id,
                        e,
                        consecutive_failures
                    );

                    if consecutive_failures >= self.failure_threshold {
                        tracing::error!("Node {} marked as down", node_id);
                        self.coordinator.mark_node_down(node_id).await;
                        self.initiate_failover(node_id).await;
                        break;
                    }
                }
            }
        }
    }

    /// Send heartbeat to a node
    async fn send_heartbeat(&self, node_id: NodeId) -> Result<()> {
        // In a real implementation, this would send an HTTP request to the node
        // For now, this is a placeholder
        tracing::debug!("Sending heartbeat to node {}", node_id);
        Ok(())
    }

    /// Initiate failover for a failed node
    async fn initiate_failover(&self, failed_node: NodeId) {
        tracing::info!("Initiating failover for node {}", failed_node);

        // Find all shards on failed node
        let shards = self.coordinator.get_shards_on_node(failed_node).await;

        for shard in shards {
            if shard.primary == failed_node {
                // Promote a replica to primary
                if let Some(&new_primary) = shard.replicas.first() {
                    tracing::info!(
                        "Promoting replica {} to primary for shard {}/{}",
                        new_primary,
                        shard.index_uid,
                        shard.shard_id
                    );

                    if let Err(e) = self
                        .coordinator
                        .promote_replica_to_primary(
                            &shard.index_uid,
                            shard.shard_id,
                            new_primary,
                        )
                        .await
                    {
                        tracing::error!("Failed to promote replica: {}", e);
                    }
                } else {
                    tracing::error!(
                        "No replicas available for shard {}/{}",
                        shard.index_uid,
                        shard.shard_id
                    );
                }
            } else {
                // This was a replica that failed
                tracing::info!(
                    "Replica failed for shard {}/{}, will allocate new replica",
                    shard.index_uid,
                    shard.shard_id
                );
                // In a full implementation, we would allocate a new replica here
            }
        }
    }

    /// Start monitoring all nodes in the cluster
    pub async fn start_monitoring(&self) {
        let state = self.coordinator.get_state().await;

        for node_id in state.nodes.keys() {
            let monitor = HealthMonitor {
                coordinator: self.coordinator.clone(),
                heartbeat_interval: self.heartbeat_interval,
                failure_threshold: self.failure_threshold,
            };

            let node_id = *node_id;
            tokio::spawn(async move {
                monitor.monitor_node_health(node_id).await;
            });
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self {
            coordinator: Arc::new(ClusterCoordinator::new(
                NodeId(0),
                meilisearch_coordinator::ShardAllocator::default(),
            )),
            heartbeat_interval: Duration::from_secs(5),
            failure_threshold: 3,
        }
    }
}
