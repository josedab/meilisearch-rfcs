use crate::error::Result;
use crate::indexer::DistributedIndexer;
use crate::merger::ResultMerger;
use crate::pool::NodeConnectionPool;
use crate::router::QueryRouter;
use crate::types::{SearchQuery, SearchResult, TaskInfo};
use meilisearch_coordinator::ClusterCoordinator;
use std::sync::Arc;

/// Gateway node for distributed Meilisearch
pub struct Gateway {
    /// Query router
    router: Arc<QueryRouter>,
    /// Result merger
    merger: Arc<ResultMerger>,
    /// Distributed indexer
    indexer: Arc<DistributedIndexer>,
    /// Cluster coordinator
    coordinator: Arc<ClusterCoordinator>,
}

impl Gateway {
    /// Create a new gateway
    pub fn new(
        node_pool: Arc<NodeConnectionPool>,
        coordinator: Arc<ClusterCoordinator>,
    ) -> Self {
        let router = Arc::new(QueryRouter::new(node_pool.clone()));
        let merger = Arc::new(ResultMerger::new());
        let indexer = Arc::new(DistributedIndexer::new(node_pool, coordinator.clone()));

        Self {
            router,
            merger,
            indexer,
            coordinator,
        }
    }

    /// Handle a search request
    pub async fn handle_search_request(
        &self,
        index_uid: &str,
        query: SearchQuery,
    ) -> Result<SearchResult> {
        // 1. Get index metadata from coordinator
        let metadata = self.coordinator.get_index(index_uid).await?;

        // 2. Route query to all shards in parallel
        let shard_results = self
            .router
            .execute_distributed_search(index_uid, &metadata.shards, &query)
            .await?;

        // 3. Merge results
        let merged = self.merger.merge_search_results(shard_results, &query)?;

        Ok(merged)
    }

    /// Handle an index documents request
    pub async fn handle_index_documents(
        &self,
        index_uid: &str,
        documents: Vec<serde_json::Value>,
    ) -> Result<TaskInfo> {
        self.indexer.index_documents(index_uid, documents).await
    }

    /// Get cluster health
    pub async fn get_cluster_health(&self) -> meilisearch_coordinator::cluster::ClusterHealth {
        self.coordinator.get_cluster_health().await
    }

    /// Get cluster nodes
    pub async fn get_cluster_nodes(&self) -> Vec<meilisearch_coordinator::NodeInfo> {
        let state = self.coordinator.get_state().await;
        state.nodes.values().cloned().collect()
    }

    /// Get index shards
    pub async fn get_index_shards(
        &self,
        index_uid: &str,
    ) -> Result<meilisearch_coordinator::IndexMetadata> {
        self.coordinator.get_index(index_uid).await.map_err(Into::into)
    }
}
