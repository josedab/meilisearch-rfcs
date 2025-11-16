use async_graphql::{Context, ErrorExtensions, Result as GraphQLResult, Subscription};
use futures_util::Stream;
use index_scheduler::IndexScheduler;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::IntervalStream;
use tokio_stream::StreamExt;

use crate::error::Error;
use crate::types::*;

pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    /// Subscribe to search results updates (polls every 100ms)
    async fn search_stream(
        &self,
        ctx: &Context<'_>,
        index_uid: String,
        query: Option<String>,
        limit: Option<i32>,
    ) -> impl Stream<Item = GraphQLResult<SearchResult>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>().ok().cloned();
        let limit = limit.unwrap_or(20).max(1).min(100);
        let query = query.unwrap_or_default();

        IntervalStream::new(tokio::time::interval(Duration::from_millis(100))).then(
            move |_| {
                let scheduler = scheduler.clone();
                let index_uid = index_uid.clone();
                let query = query.clone();

                async move {
                    if let Some(scheduler) = scheduler {
                        // Perform search
                        // This is a simplified implementation
                        // In production, this would use the actual search engine

                        Ok(SearchResult {
                            hits: vec![],
                            estimated_total_hits: 0,
                            processing_time_ms: 0,
                            query: query.clone(),
                            limit,
                            offset: 0,
                            facet_distribution: None,
                        })
                    } else {
                        Err(Error::InternalError(
                            "Index scheduler not available".to_string(),
                        )
                        .extend())
                    }
                }
            },
        )
    }

    /// Subscribe to task status updates
    async fn task_updates(
        &self,
        ctx: &Context<'_>,
        task_uid: Option<i32>,
        index_uid: Option<String>,
    ) -> impl Stream<Item = GraphQLResult<Task>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>().ok().cloned();

        IntervalStream::new(tokio::time::interval(Duration::from_millis(500))).then(
            move |_| {
                let scheduler = scheduler.clone();
                let task_uid = task_uid;
                let index_uid = index_uid.clone();

                async move {
                    if let Some(scheduler) = scheduler {
                        let rtxn = scheduler
                            .env
                            .read_txn()
                            .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

                        // Get task
                        if let Some(uid) = task_uid {
                            if let Some(task) = scheduler
                                .get_task(&rtxn, uid as u32)
                                .map_err(|e| Error::IndexSchedulerError(e.to_string()))?
                            {
                                return Ok(convert_task(task));
                            }
                        }

                        // Return a placeholder task
                        Err(Error::TaskNotFound(task_uid.unwrap_or(0) as u32).extend())
                    } else {
                        Err(Error::InternalError(
                            "Index scheduler not available".to_string(),
                        )
                        .extend())
                    }
                }
            },
        )
    }

    /// Subscribe to index stats updates
    async fn index_stats_stream(
        &self,
        ctx: &Context<'_>,
        index_uid: String,
    ) -> impl Stream<Item = GraphQLResult<IndexStats>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>().ok().cloned();

        IntervalStream::new(tokio::time::interval(Duration::from_secs(1))).then(move |_| {
            let scheduler = scheduler.clone();
            let index_uid = index_uid.clone();

            async move {
                if let Some(scheduler) = scheduler {
                    let rtxn = scheduler
                        .env
                        .read_txn()
                        .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

                    // Check if index exists
                    let exists = scheduler
                        .index_exists(&rtxn, &index_uid)
                        .map_err(|e| Error::IndexSchedulerError(e.to_string()))?;

                    if !exists {
                        return Err(Error::IndexNotFound(index_uid.clone()).extend());
                    }

                    // Get index stats
                    // This is simplified - in production would get real stats
                    Ok(IndexStats {
                        number_of_documents: 0,
                        is_indexing: false,
                        field_distribution: serde_json::json!({}),
                    })
                } else {
                    Err(
                        Error::InternalError("Index scheduler not available".to_string())
                            .extend(),
                    )
                }
            }
        })
    }
}

// Helper function to convert internal task to GraphQL task
fn convert_task(task: meilisearch_types::tasks::Task) -> Task {
    use meilisearch_types::tasks::Status;

    let status = match task.status {
        Status::Enqueued => TaskStatus::Enqueued,
        Status::Processing => TaskStatus::Processing,
        Status::Succeeded => TaskStatus::Succeeded,
        Status::Failed => TaskStatus::Failed,
        Status::Canceled => TaskStatus::Canceled,
    };

    // Simplified task type
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
