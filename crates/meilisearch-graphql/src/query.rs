use async_graphql::{Context, Object, Result as GraphQLResult};
use index_scheduler::IndexScheduler;
use std::sync::Arc;
use time::OffsetDateTime;

use crate::error::{Error, Result};
use crate::types::*;

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get a single index by UID
    async fn index(
        &self,
        ctx: &Context<'_>,
        uid: String,
    ) -> GraphQLResult<Option<Index>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;

        // Check if index exists
        let rtxn = scheduler.env.read_txn()?;
        let index_exists = scheduler
            .index_exists(&rtxn, &uid)
            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

        if !index_exists {
            return Ok(None);
        }

        // Get index metadata
        let created_at = OffsetDateTime::now_utc();
        let updated_at = OffsetDateTime::now_utc();

        Ok(Some(Index {
            uid,
            primary_key: None,
            created_at,
            updated_at,
        }))
    }

    /// List all indexes with pagination
    async fn indexes(
        &self,
        ctx: &Context<'_>,
        offset: Option<i32>,
        limit: Option<i32>,
    ) -> GraphQLResult<IndexConnection> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let rtxn = scheduler.env.read_txn()?;

        // Get all index names
        let index_names = scheduler
            .index_names(&rtxn)
            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

        let offset = offset.unwrap_or(0).max(0) as usize;
        let limit = limit.unwrap_or(20).max(1).min(100) as usize;
        let total = index_names.len() as i32;

        let results: Vec<Index> = index_names
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|uid| Index {
                uid,
                primary_key: None,
                created_at: OffsetDateTime::now_utc(),
                updated_at: OffsetDateTime::now_utc(),
            })
            .collect();

        Ok(IndexConnection {
            results,
            offset: offset as i32,
            limit: limit as i32,
            total,
        })
    }

    /// Perform federated search across multiple indexes
    async fn federated_search(
        &self,
        ctx: &Context<'_>,
        queries: Vec<FederatedQueryInput>,
        limit: Option<i32>,
    ) -> GraphQLResult<FederatedSearchResult> {
        let _scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let limit = limit.unwrap_or(20).max(1).min(100);

        // Placeholder implementation
        // In a real implementation, this would:
        // 1. Execute searches across all specified indexes
        // 2. Merge and rank results based on weights
        // 3. Return combined results

        Ok(FederatedSearchResult {
            hits: vec![],
            processing_time_ms: 0,
            limit,
        })
    }

    /// Get a single task by UID
    async fn task(
        &self,
        ctx: &Context<'_>,
        uid: i32,
    ) -> GraphQLResult<Option<Task>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let rtxn = scheduler.env.read_txn()?;

        // Get task from scheduler
        let task = scheduler
            .get_task(&rtxn, uid as u32)
            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

        if let Some(task) = task {
            Ok(Some(convert_task(task)))
        } else {
            Ok(None)
        }
    }

    /// List tasks with filtering and pagination
    async fn tasks(
        &self,
        ctx: &Context<'_>,
        filter: Option<TaskFilterInput>,
        limit: Option<i32>,
        offset: Option<i32>,
    ) -> GraphQLResult<TaskConnection> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let rtxn = scheduler.env.read_txn()?;

        let offset = offset.unwrap_or(0).max(0) as usize;
        let limit = limit.unwrap_or(20).max(1).min(100) as usize;

        // Build query from filter
        let query = build_task_query(filter);

        // Get tasks from scheduler
        let tasks = scheduler
            .get_task_ids(&rtxn, &query)
            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

        let total = tasks.len() as i32;

        let results: Vec<Task> = tasks
            .into_iter()
            .skip(offset)
            .take(limit)
            .filter_map(|task_id| {
                scheduler
                    .get_task(&rtxn, task_id)
                    .ok()
                    .flatten()
                    .map(convert_task)
            })
            .collect();

        Ok(TaskConnection {
            results,
            offset: offset as i32,
            limit: limit as i32,
            total,
        })
    }

    /// Get global stats
    async fn stats(&self, ctx: &Context<'_>) -> GraphQLResult<Stats> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let rtxn = scheduler.env.read_txn()?;

        let stats = scheduler
            .stats()
            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

        Ok(Stats {
            database_size: stats.database_size as i64,
            last_update: None,
            indexes: serde_json::json!(stats.indexes),
        })
    }

    /// Health check endpoint
    async fn health(&self, _ctx: &Context<'_>) -> GraphQLResult<Health> {
        Ok(Health {
            status: "available".to_string(),
        })
    }
}

// Helper functions

fn convert_task(task: meilisearch_types::tasks::Task) -> Task {
    use meilisearch_types::tasks::Status;

    let status = match task.status {
        Status::Enqueued => TaskStatus::Enqueued,
        Status::Processing => TaskStatus::Processing,
        Status::Succeeded => TaskStatus::Succeeded,
        Status::Failed => TaskStatus::Failed,
        Status::Canceled => TaskStatus::Canceled,
    };

    // Simplified task type - use as_kind() method
    let task_type = TaskType::IndexUpdate; // Placeholder

    Task {
        uid: task.uid as i32,
        index_uid: task.index_uid().map(|s| s.to_string()),
        status,
        task_type,
        details: None,
        error: task.error.map(|e| TaskError {
            message: e.message,
            code: e.code.to_string(),
            error_type: "internal".to_string(),
            link: "https://www.meilisearch.com/docs".to_string(),
        }),
        duration: None,
        enqueued_at: task.enqueued_at,
        started_at: task.started_at,
        finished_at: task.finished_at,
    }
}

fn build_task_query(filter: Option<TaskFilterInput>) -> index_scheduler::Query {
    let mut query = index_scheduler::Query::default();

    if let Some(filter) = filter {
        if let Some(uids) = filter.uids {
            query.uids = Some(uids.into_iter().map(|u| u as u32).collect());
        }

        if let Some(statuses) = filter.statuses {
            use meilisearch_types::tasks::Status;
            query.statuses = Some(
                statuses
                    .into_iter()
                    .map(|s| match s {
                        TaskStatus::Enqueued => Status::Enqueued,
                        TaskStatus::Processing => Status::Processing,
                        TaskStatus::Succeeded => Status::Succeeded,
                        TaskStatus::Failed => Status::Failed,
                        TaskStatus::Canceled => Status::Canceled,
                    })
                    .collect(),
            );
        }

        if let Some(index_uids) = filter.index_uids {
            query.index_uids = Some(index_uids);
        }
    }

    query
}
