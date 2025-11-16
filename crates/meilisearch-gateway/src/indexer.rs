use crate::error::{Error, Result};
use crate::pool::NodeConnectionPool;
use crate::types::{ShardTaskInfo, TaskInfo, TaskStatus};
use meilisearch_coordinator::{ClusterCoordinator, ShardingStrategy};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Distributed indexer for write operations
pub struct DistributedIndexer {
    /// Node connection pool
    pool: Arc<NodeConnectionPool>,
    /// Cluster coordinator
    coordinator: Arc<ClusterCoordinator>,
}

impl DistributedIndexer {
    /// Create a new distributed indexer
    pub fn new(pool: Arc<NodeConnectionPool>, coordinator: Arc<ClusterCoordinator>) -> Self {
        Self { pool, coordinator }
    }

    /// Index documents across shards
    pub async fn index_documents(
        &self,
        index_uid: &str,
        documents: Vec<serde_json::Value>,
    ) -> Result<TaskInfo> {
        // Get index metadata
        let metadata = self.coordinator.get_index(index_uid).await?;

        // Group documents by target shard
        let mut docs_by_shard: HashMap<usize, Vec<serde_json::Value>> = HashMap::new();

        for doc in documents {
            let doc_id = doc
                .get("id")
                .ok_or(Error::MissingDocumentId)?
                .as_str()
                .ok_or(Error::InvalidDocumentId)?;

            let shard_id = self.determine_shard(doc_id, &doc, &metadata.strategy, metadata.shard_count)?;
            docs_by_shard.entry(shard_id).or_insert_with(Vec::new).push(doc);
        }

        // Send documents to respective shards in parallel
        let index_futures = docs_by_shard.into_iter().map(|(shard_id, docs)| {
            let shard = &metadata.shards[shard_id];
            self.index_shard_documents(index_uid, shard.shard_id, shard.primary.0, docs)
        });

        let shard_tasks = futures::future::try_join_all(index_futures).await?;

        // Create aggregated task
        Ok(TaskInfo {
            task_uid: uuid::Uuid::new_v4().to_string(),
            status: TaskStatus::Enqueued,
            shard_tasks,
        })
    }

    /// Index documents on a specific shard
    async fn index_shard_documents(
        &self,
        index_uid: &str,
        shard_id: usize,
        node_id: u64,
        documents: Vec<serde_json::Value>,
    ) -> Result<ShardTaskInfo> {
        let node = meilisearch_coordinator::NodeId(node_id);
        let task = self.pool.send_index_request(node, index_uid, documents).await?;

        Ok(ShardTaskInfo {
            shard_id,
            node_id,
            task_uid: task.task_uid,
            status: task.status,
        })
    }

    /// Determine which shard a document belongs to
    fn determine_shard(
        &self,
        doc_id: &str,
        doc: &serde_json::Value,
        strategy: &ShardingStrategy,
        shard_count: usize,
    ) -> Result<usize> {
        match strategy {
            ShardingStrategy::Hash => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                doc_id.hash(&mut hasher);
                Ok((hasher.finish() as usize) % shard_count)
            }
            ShardingStrategy::Range { ranges } => {
                for (end_key, shard_id) in ranges {
                    if doc_id <= end_key {
                        return Ok(*shard_id);
                    }
                }
                Ok(ranges.last().map(|(_, id)| *id).unwrap_or(0))
            }
            ShardingStrategy::Field { field } => {
                let field_value = doc
                    .get(field)
                    .ok_or_else(|| Error::ShardingFieldMissing(field.clone()))?
                    .as_str()
                    .ok_or_else(|| Error::InvalidShardingField(field.clone()))?;

                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                field_value.hash(&mut hasher);
                Ok((hasher.finish() as usize) % shard_count)
            }
        }
    }
}
