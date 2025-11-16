use crate::ctr_tracker::CTRTracker;
use crate::query_logger::{QueryStorage, SearchType};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub struct RelevancyLearner {
    ctr_tracker: Arc<CTRTracker>,
    storage: Arc<QueryStorage>,
}

impl RelevancyLearner {
    pub fn new(ctr_tracker: Arc<CTRTracker>, storage: Arc<QueryStorage>) -> Self {
        Self {
            ctr_tracker,
            storage,
        }
    }

    /// Analyze query patterns and suggest improvements
    pub async fn suggest_improvements(
        &self,
        index_uid: &str,
        time_range_days: i64,
    ) -> Result<Vec<RelevancySuggestion>> {
        let mut suggestions = Vec::new();

        let end_time = time::OffsetDateTime::now_utc().unix_timestamp();
        let start_time = end_time - (time_range_days * 24 * 60 * 60);

        // Get query logs for analysis
        let logs = self.storage.get_query_logs(index_uid, start_time, end_time)?;

        // 1. Identify high-zero-result queries
        let zero_result_queries = self.find_zero_result_queries(&logs);
        if !zero_result_queries.is_empty() {
            suggestions.push(RelevancySuggestion {
                suggestion_type: SuggestionType::AddSynonyms,
                description: format!(
                    "{} queries frequently return zero results. Consider adding synonyms or adjusting filters.",
                    zero_result_queries.len()
                ),
                affected_queries: zero_result_queries,
                expected_impact: Impact::High,
            });
        }

        // 2. Identify low-CTR queries
        let low_ctr_queries = self.find_low_ctr_queries(&logs, 0.05);
        if !low_ctr_queries.is_empty() {
            suggestions.push(RelevancySuggestion {
                suggestion_type: SuggestionType::AdjustRanking,
                description: format!(
                    "{} queries have low click-through rates (< 5%). Ranking may need adjustment.",
                    low_ctr_queries.len()
                ),
                affected_queries: low_ctr_queries,
                expected_impact: Impact::Medium,
            });
        }

        // 3. Detect hybrid search opportunities
        let keyword_only_queries = self.find_semantic_opportunities(&logs);
        if !keyword_only_queries.is_empty() {
            suggestions.push(RelevancySuggestion {
                suggestion_type: SuggestionType::EnableHybridSearch,
                description: format!(
                    "{} queries are using keyword-only search. Consider enabling hybrid search for better semantic understanding.",
                    keyword_only_queries.len()
                ),
                affected_queries: keyword_only_queries,
                expected_impact: Impact::High,
            });
        }

        // 4. Detect slow queries that might benefit from optimization
        let slow_queries = self.find_slow_queries(&logs, 100); // > 100ms
        if !slow_queries.is_empty() {
            suggestions.push(RelevancySuggestion {
                suggestion_type: SuggestionType::OptimizePerformance,
                description: format!(
                    "{} queries are slow (> 100ms). Consider optimizing searchable attributes or adding custom ranking.",
                    slow_queries.len()
                ),
                affected_queries: slow_queries,
                expected_impact: Impact::Medium,
            });
        }

        Ok(suggestions)
    }

    fn find_zero_result_queries(&self, logs: &[crate::query_logger::QueryLog]) -> Vec<String> {
        let mut query_stats: HashMap<String, (usize, usize)> = HashMap::new();

        for log in logs {
            let entry = query_stats.entry(log.query.clone()).or_insert((0, 0));
            entry.0 += 1; // Total searches
            if log.hits_count == 0 {
                entry.1 += 1; // Zero result searches
            }
        }

        query_stats
            .into_iter()
            .filter(|(_, (total, zero))| {
                // Query appears at least 5 times and has >50% zero results
                *total >= 5 && (*zero as f64 / *total as f64) > 0.5
            })
            .map(|(query, _)| query)
            .collect()
    }

    fn find_low_ctr_queries(&self, logs: &[crate::query_logger::QueryLog], threshold: f64) -> Vec<String> {
        let mut query_stats: HashMap<String, (usize, usize)> = HashMap::new();

        for log in logs {
            let entry = query_stats.entry(log.query.clone()).or_insert((0, 0));
            entry.0 += 1; // Total searches
            entry.1 += log.total_clicks; // Total clicks
        }

        query_stats
            .into_iter()
            .filter(|(_, (total, clicks))| {
                // Query appears at least 10 times
                *total >= 10 && {
                    let ctr = *clicks as f64 / *total as f64;
                    ctr < threshold && ctr > 0.0 // Has some engagement but low
                }
            })
            .map(|(query, _)| query)
            .collect()
    }

    fn find_semantic_opportunities(&self, logs: &[crate::query_logger::QueryLog]) -> Vec<String> {
        let mut keyword_queries: HashMap<String, usize> = HashMap::new();

        for log in logs {
            if matches!(log.search_type, SearchType::Keyword) {
                *keyword_queries.entry(log.query.clone()).or_insert(0) += 1;
            }
        }

        // Find queries that appear frequently (>= 20 times) and are keyword-only
        keyword_queries
            .into_iter()
            .filter(|(_, count)| *count >= 20)
            .map(|(query, _)| query)
            .collect()
    }

    fn find_slow_queries(&self, logs: &[crate::query_logger::QueryLog], threshold_ms: u64) -> Vec<String> {
        let mut query_stats: HashMap<String, (usize, u64)> = HashMap::new();

        for log in logs {
            let entry = query_stats.entry(log.query.clone()).or_insert((0, 0));
            entry.0 += 1; // Count
            entry.1 += log.processing_time_ms; // Total processing time
        }

        query_stats
            .into_iter()
            .filter(|(_, (count, total_time))| {
                *count >= 5 && (*total_time / *count as u64) > threshold_ms
            })
            .map(|(query, _)| query)
            .collect()
    }

    /// Get query performance statistics
    pub fn get_query_stats(
        &self,
        index_uid: &str,
        query: &str,
        time_range_days: i64,
    ) -> Result<QueryStats> {
        let end_time = time::OffsetDateTime::now_utc().unix_timestamp();
        let start_time = end_time - (time_range_days * 24 * 60 * 60);

        let logs = self.storage.get_query_logs(index_uid, start_time, end_time)?;

        let query_logs: Vec<_> = logs.iter()
            .filter(|log| log.query == query)
            .collect();

        if query_logs.is_empty() {
            return Ok(QueryStats::default());
        }

        let total_searches = query_logs.len();
        let total_clicks: usize = query_logs.iter().map(|log| log.total_clicks).sum();
        let total_results: usize = query_logs.iter().map(|log| log.hits_count).sum();
        let total_processing_time: u64 = query_logs.iter().map(|log| log.processing_time_ms).sum();
        let zero_result_count = query_logs.iter().filter(|log| log.hits_count == 0).count();

        Ok(QueryStats {
            total_searches,
            total_clicks,
            ctr: total_clicks as f64 / total_searches as f64,
            avg_results: total_results as f64 / total_searches as f64,
            avg_processing_time_ms: total_processing_time as f64 / total_searches as f64,
            zero_results_rate: zero_result_count as f64 / total_searches as f64,
        })
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RelevancySuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub affected_queries: Vec<String>,
    pub expected_impact: Impact,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum SuggestionType {
    AddSynonyms,
    AdjustRanking,
    EnableHybridSearch,
    ReduceTypoTolerance,
    IncreaseTypoTolerance,
    AddStopWords,
    OptimizePerformance,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Impact {
    Low,
    Medium,
    High,
}

#[derive(Default, Serialize, Deserialize)]
pub struct QueryStats {
    pub total_searches: usize,
    pub total_clicks: usize,
    pub ctr: f64,
    pub avg_results: f64,
    pub avg_processing_time_ms: f64,
    pub zero_results_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_logger::{LatencyBucket, QueryLog, SearchType};
    use crate::query_logger::QueryStorage;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_zero_result_detection() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(QueryStorage::new(temp_dir.path()).unwrap());
        let ctr_tracker = Arc::new(CTRTracker::new(storage.clone()));
        let learner = RelevancyLearner::new(ctr_tracker, storage.clone());

        // Create logs with some zero-result queries
        let mut logs = Vec::new();
        for i in 0..10 {
            logs.push(QueryLog {
                query_id: format!("query_{}", i),
                index_uid: "test_index".to_string(),
                query: "missing_term".to_string(),
                filters: None,
                hits_count: 0, // Zero results
                processing_time_ms: 15,
                timestamp: 1699564800 + i,
                user_id: Some(format!("user_{}", i)),
                search_type: SearchType::Keyword,
                semantic_ratio: None,
                clicked_positions: vec![],
                total_clicks: 0,
                latency_bucket: LatencyBucket::Normal,
            });
        }

        storage.write_batch(&logs).await.unwrap();

        let suggestions = learner.suggest_improvements("test_index", 7).await.unwrap();

        // Should suggest adding synonyms for zero-result query
        assert!(suggestions.iter().any(|s| matches!(s.suggestion_type, SuggestionType::AddSynonyms)));
    }

    #[tokio::test]
    async fn test_semantic_opportunity_detection() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(QueryStorage::new(temp_dir.path()).unwrap());
        let ctr_tracker = Arc::new(CTRTracker::new(storage.clone()));
        let learner = RelevancyLearner::new(ctr_tracker, storage.clone());

        // Create many keyword-only searches
        let mut logs = Vec::new();
        for i in 0..25 {
            logs.push(QueryLog {
                query_id: format!("query_{}", i),
                index_uid: "test_index".to_string(),
                query: "laptop".to_string(),
                filters: None,
                hits_count: 10,
                processing_time_ms: 15,
                timestamp: 1699564800 + i,
                user_id: Some(format!("user_{}", i)),
                search_type: SearchType::Keyword, // Only keyword search
                semantic_ratio: None,
                clicked_positions: vec![0],
                total_clicks: 1,
                latency_bucket: LatencyBucket::Normal,
            });
        }

        storage.write_batch(&logs).await.unwrap();

        let suggestions = learner.suggest_improvements("test_index", 7).await.unwrap();

        // Should suggest enabling hybrid search
        assert!(suggestions.iter().any(|s| matches!(s.suggestion_type, SuggestionType::EnableHybridSearch)));
    }
}
