use crate::error::{Error, Result};
use crate::pool::NodeConnectionPool;
use crate::types::{SearchQuery, ShardSearchResult};
use meilisearch_coordinator::ShardAssignment;
use std::sync::Arc;

/// Query router for distributed search
pub struct QueryRouter {
    /// Node connection pool
    node_pool: Arc<NodeConnectionPool>,
}

impl QueryRouter {
    /// Create a new query router
    pub fn new(node_pool: Arc<NodeConnectionPool>) -> Self {
        Self { node_pool }
    }

    /// Execute distributed search across all shards
    pub async fn execute_distributed_search(
        &self,
        index_uid: &str,
        shards: &[ShardAssignment],
        query: &SearchQuery,
    ) -> Result<Vec<ShardSearchResult>> {
        // Execute search on all shards in parallel
        let shard_futures = shards
            .iter()
            .map(|shard| self.execute_shard_search(index_uid, shard, query));

        let results = futures::future::try_join_all(shard_futures).await?;

        Ok(results)
    }

    /// Execute search on a single shard
    async fn execute_shard_search(
        &self,
        index_uid: &str,
        shard: &ShardAssignment,
        query: &SearchQuery,
    ) -> Result<ShardSearchResult> {
        // Try primary first
        match self.query_node(shard.primary, index_uid, query).await {
            Ok(result) => Ok(result),
            Err(e) if e.is_node_unavailable() => {
                // Fallback to replicas
                self.query_replica(index_uid, shard, query).await
            }
            Err(e) => Err(e),
        }
    }

    /// Query a specific node
    async fn query_node(
        &self,
        node_id: meilisearch_coordinator::NodeId,
        index_uid: &str,
        query: &SearchQuery,
    ) -> Result<ShardSearchResult> {
        self.node_pool.send_query(node_id, index_uid, query).await
    }

    /// Try querying replicas when primary fails
    async fn query_replica(
        &self,
        index_uid: &str,
        shard: &ShardAssignment,
        query: &SearchQuery,
    ) -> Result<ShardSearchResult> {
        for &replica in &shard.replicas {
            if let Ok(result) = self.query_node(replica, index_uid, query).await {
                return Ok(result);
            }
        }
        Err(Error::AllNodesUnavailable)
    }
}

impl Error {
    /// Check if this is a node unavailable error
    pub fn is_node_unavailable(&self) -> bool {
        matches!(self, Error::NodeUnavailable(_))
    }
}
