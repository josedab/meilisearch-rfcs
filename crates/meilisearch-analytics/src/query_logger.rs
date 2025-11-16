use crate::{Error, Result};
use heed::{types::*, Database, Env, RwTxn};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLog {
    /// Unique query ID
    pub query_id: String,
    /// Index queried
    pub index_uid: String,
    /// Query text
    pub query: String,
    /// Filters applied
    pub filters: Option<String>,
    /// Number of results
    pub hits_count: usize,
    /// Processing time
    pub processing_time_ms: u64,
    /// Timestamp
    pub timestamp: i64,
    /// User/session identifier
    pub user_id: Option<String>,
    /// Search type (keyword, semantic, hybrid)
    pub search_type: SearchType,
    /// Semantic ratio (for hybrid)
    pub semantic_ratio: Option<f32>,
    /// Result positions of clicked documents
    pub clicked_positions: Vec<usize>,
    /// Total clicks
    pub total_clicks: usize,
    /// Query latency percentile
    pub latency_bucket: LatencyBucket,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    Keyword,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyBucket {
    Fast,      // < 10ms
    Normal,    // 10-50ms
    Slow,      // 50-200ms
    VerySlow,  // > 200ms
}

impl LatencyBucket {
    pub fn from_millis(ms: u64) -> Self {
        match ms {
            0..=9 => Self::Fast,
            10..=49 => Self::Normal,
            50..=199 => Self::Slow,
            _ => Self::VerySlow,
        }
    }
}

pub struct QueryLogger {
    /// Channel to send query logs
    log_tx: mpsc::UnboundedSender<QueryLog>,
}

impl QueryLogger {
    pub fn new(_storage: Arc<QueryStorage>) -> (Self, mpsc::UnboundedReceiver<QueryLog>) {
        let (log_tx, log_rx) = mpsc::unbounded_channel();

        (Self { log_tx }, log_rx)
    }

    /// Log a search query
    pub fn log_query(&self, log: QueryLog) -> Result<()> {
        self.log_tx.send(log)
            .map_err(|_| Error::LogChannelClosed)?;
        Ok(())
    }

    /// Background task to persist logs
    pub async fn persist_logs(
        mut log_rx: mpsc::UnboundedReceiver<QueryLog>,
        storage: Arc<QueryStorage>,
    ) {
        let mut batch = Vec::new();
        let mut last_flush = Instant::now();

        loop {
            tokio::select! {
                Some(log) = log_rx.recv() => {
                    batch.push(log);

                    // Flush when batch size reached or 5s elapsed
                    if batch.len() >= 100 || last_flush.elapsed() > Duration::from_secs(5) {
                        if let Err(e) = storage.write_batch(&batch).await {
                            tracing::error!("Failed to write query logs: {}", e);
                        }
                        batch.clear();
                        last_flush = Instant::now();
                    }
                }

                _ = tokio::time::sleep(Duration::from_secs(5)) => {
                    if !batch.is_empty() {
                        if let Err(e) = storage.write_batch(&batch).await {
                            tracing::error!("Failed to write query logs: {}", e);
                        }
                        batch.clear();
                        last_flush = Instant::now();
                    }
                }
            }
        }
    }
}

/// Query storage backend
pub struct QueryStorage {
    /// LMDB environment for analytics
    env: Arc<Env>,
    /// Query log database
    query_db: Database<Str, SerdeBincode<QueryLog>>,
    /// Time-series metrics database
    metrics_db: Database<U64<heed::byteorder::NativeEndian>, SerdeBincode<MetricsBucket>>,
}

impl QueryStorage {
    /// Create a new QueryStorage instance
    pub fn new(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let env = unsafe {
            heed::EnvOpenOptions::new()
                .map_size(10 * 1024 * 1024 * 1024) // 10 GB
                .max_dbs(10)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;
        let query_db = env.create_database(&mut wtxn, Some("queries"))?;
        let metrics_db = env.create_database(&mut wtxn, Some("metrics"))?;
        wtxn.commit()?;

        Ok(Self {
            env: Arc::new(env),
            query_db,
            metrics_db,
        })
    }

    pub async fn write_batch(&self, logs: &[QueryLog]) -> Result<()> {
        let mut wtxn = self.env.write_txn()?;

        for log in logs {
            let key = format!("{}:{}", log.timestamp, log.query_id);
            self.query_db.put(&mut wtxn, &key, log)?;

            // Update time-series metrics
            self.update_metrics(&mut wtxn, log)?;
        }

        wtxn.commit()?;
        Ok(())
    }

    fn update_metrics(&self, wtxn: &mut RwTxn, log: &QueryLog) -> Result<()> {
        // Bucket by 5-minute intervals
        let bucket_timestamp = (log.timestamp / 300) * 300;

        let mut bucket = self.metrics_db.get(wtxn, &(bucket_timestamp as u64))?
            .unwrap_or_default();

        bucket.query_count += 1;
        bucket.total_processing_time_ms += log.processing_time_ms;
        bucket.total_clicks += log.total_clicks;
        bucket.zero_result_count += if log.hits_count == 0 { 1 } else { 0 };

        self.metrics_db.put(wtxn, &(bucket_timestamp as u64), &bucket)?;

        Ok(())
    }

    /// Get metrics for a time range
    pub fn get_metrics(&self, start_time: i64, end_time: i64) -> Result<Vec<MetricsBucket>> {
        let rtxn = self.env.read_txn()?;
        let mut metrics = Vec::new();

        let start_bucket = (start_time / 300) * 300;
        let end_bucket = (end_time / 300) * 300;

        for timestamp in (start_bucket..=end_bucket).step_by(300) {
            if let Some(bucket) = self.metrics_db.get(&rtxn, &(timestamp as u64))? {
                metrics.push(bucket);
            }
        }

        Ok(metrics)
    }

    /// Get query logs for a time range
    pub fn get_query_logs(&self, index_uid: &str, start_time: i64, end_time: i64) -> Result<Vec<QueryLog>> {
        let rtxn = self.env.read_txn()?;
        let mut logs = Vec::new();

        // Iterate through all entries and filter by index_uid and timestamp
        for result in self.query_db.iter(&rtxn)? {
            let (_key, log) = result?;
            if log.index_uid == index_uid && log.timestamp >= start_time && log.timestamp <= end_time {
                logs.push(log);
            }
        }

        Ok(logs)
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MetricsBucket {
    pub query_count: usize,
    pub total_processing_time_ms: u64,
    pub total_clicks: usize,
    pub zero_result_count: usize,
}

impl MetricsBucket {
    pub fn avg_processing_time(&self) -> f64 {
        if self.query_count == 0 {
            0.0
        } else {
            self.total_processing_time_ms as f64 / self.query_count as f64
        }
    }

    pub fn ctr(&self) -> f64 {
        if self.query_count == 0 {
            0.0
        } else {
            self.total_clicks as f64 / self.query_count as f64
        }
    }

    pub fn zero_results_rate(&self) -> f64 {
        if self.query_count == 0 {
            0.0
        } else {
            self.zero_result_count as f64 / self.query_count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_latency_bucket() {
        assert!(matches!(LatencyBucket::from_millis(5), LatencyBucket::Fast));
        assert!(matches!(LatencyBucket::from_millis(25), LatencyBucket::Normal));
        assert!(matches!(LatencyBucket::from_millis(100), LatencyBucket::Slow));
        assert!(matches!(LatencyBucket::from_millis(500), LatencyBucket::VerySlow));
    }

    #[tokio::test]
    async fn test_query_logging() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(QueryStorage::new(temp_dir.path()).unwrap());

        let log = QueryLog {
            query_id: "test_query_1".to_string(),
            index_uid: "test_index".to_string(),
            query: "laptop".to_string(),
            filters: None,
            hits_count: 10,
            processing_time_ms: 15,
            timestamp: 1699564800,
            user_id: Some("user123".to_string()),
            search_type: SearchType::Keyword,
            semantic_ratio: None,
            clicked_positions: vec![0],
            total_clicks: 1,
            latency_bucket: LatencyBucket::Normal,
        };

        storage.write_batch(&[log.clone()]).await.unwrap();

        let logs = storage.get_query_logs("test_index", 1699564700, 1699564900).unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].query, "laptop");
    }
}
