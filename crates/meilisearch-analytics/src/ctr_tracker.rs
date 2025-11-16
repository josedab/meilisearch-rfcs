use crate::query_logger::QueryStorage;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub struct CTRTracker {
    /// Tracks clicks per query
    click_data: Arc<RwLock<HashMap<String, ClickData>>>,
    /// Storage for persistence
    storage: Arc<QueryStorage>,
}

#[derive(Clone)]
struct ClickData {
    /// Total impressions (searches)
    impressions: usize,
    /// Clicks per result position
    clicks_by_position: HashMap<usize, usize>,
    /// Total clicks
    total_clicks: usize,
    /// Last updated
    last_updated: Instant,
}

impl Default for ClickData {
    fn default() -> Self {
        Self {
            impressions: 0,
            clicks_by_position: HashMap::new(),
            total_clicks: 0,
            last_updated: Instant::now(),
        }
    }
}

impl CTRTracker {
    pub fn new(storage: Arc<QueryStorage>) -> Self {
        Self {
            click_data: Arc::new(RwLock::new(HashMap::new())),
            storage,
        }
    }

    pub fn record_impression(&self, query_id: &str, _results_count: usize) {
        let mut data = self.click_data.write().unwrap();
        let entry = data.entry(query_id.to_string()).or_default();
        entry.impressions += 1;
        entry.last_updated = Instant::now();
    }

    pub fn record_click(&self, query_id: &str, position: usize) {
        let mut data = self.click_data.write().unwrap();
        let entry = data.entry(query_id.to_string()).or_default();

        *entry.clicks_by_position.entry(position).or_insert(0) += 1;
        entry.total_clicks += 1;
        entry.last_updated = Instant::now();
    }

    /// Compute CTR for a query
    pub fn compute_ctr(&self, query_id: &str) -> Option<f64> {
        let data = self.click_data.read().unwrap();
        let entry = data.get(query_id)?;

        if entry.impressions == 0 {
            return None;
        }

        Some(entry.total_clicks as f64 / entry.impressions as f64)
    }

    /// Compute Mean Reciprocal Rank (MRR)
    pub fn compute_mrr(&self, query_id: &str) -> Option<f64> {
        let data = self.click_data.read().unwrap();
        let entry = data.get(query_id)?;

        if entry.clicks_by_position.is_empty() {
            return None;
        }

        // MRR = average of 1/rank for first clicks
        let reciprocal_ranks: Vec<f64> = entry.clicks_by_position.iter()
            .map(|(&position, &count)| (count as f64) / ((position + 1) as f64))
            .collect();

        let mrr = reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64;
        Some(mrr)
    }

    /// Get top-K most clicked positions for query
    pub fn top_clicked_positions(&self, query_id: &str, k: usize) -> Vec<(usize, usize)> {
        let data = self.click_data.read().unwrap();
        let entry = match data.get(query_id) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut positions: Vec<_> = entry.clicks_by_position.iter()
            .map(|(&pos, &count)| (pos, count))
            .collect();

        positions.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        positions.truncate(k);

        positions
    }

    /// Get total impressions for a query
    pub fn get_impressions(&self, query_id: &str) -> usize {
        let data = self.click_data.read().unwrap();
        data.get(query_id).map(|e| e.impressions).unwrap_or(0)
    }

    /// Get total clicks for a query
    pub fn get_total_clicks(&self, query_id: &str) -> usize {
        let data = self.click_data.read().unwrap();
        data.get(query_id).map(|e| e.total_clicks).unwrap_or(0)
    }

    /// Clear old data (for memory management)
    pub fn clear_old_data(&self, max_age: std::time::Duration) {
        let mut data = self.click_data.write().unwrap();
        let now = Instant::now();

        data.retain(|_, entry| {
            now.duration_since(entry.last_updated) < max_age
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_logger::QueryStorage;
    use tempfile::TempDir;

    #[test]
    fn test_ctr_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(QueryStorage::new(temp_dir.path()).unwrap());
        let tracker = CTRTracker::new(storage);

        let query_id = "test_query";

        // Record 10 impressions
        for _ in 0..10 {
            tracker.record_impression(query_id, 5);
        }

        // Record 3 clicks at different positions
        tracker.record_click(query_id, 0);
        tracker.record_click(query_id, 1);
        tracker.record_click(query_id, 0);

        // Test CTR
        let ctr = tracker.compute_ctr(query_id).unwrap();
        assert_eq!(ctr, 0.3); // 3 clicks / 10 impressions

        // Test MRR
        let mrr = tracker.compute_mrr(query_id).unwrap();
        assert!(mrr > 0.0);

        // Test top clicked positions
        let top_positions = tracker.top_clicked_positions(query_id, 2);
        assert_eq!(top_positions.len(), 2);
        assert_eq!(top_positions[0].0, 0); // Position 0 should be most clicked
        assert_eq!(top_positions[0].1, 2); // With 2 clicks
    }

    #[test]
    fn test_mrr_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Arc::new(QueryStorage::new(temp_dir.path()).unwrap());
        let tracker = CTRTracker::new(storage);

        let query_id = "test_mrr";

        tracker.record_impression(query_id, 5);
        tracker.record_click(query_id, 0); // Position 1: 1/1 = 1.0
        tracker.record_click(query_id, 2); // Position 3: 1/3 = 0.333

        let mrr = tracker.compute_mrr(query_id).unwrap();
        // MRR = (1 + 0.333) / 2 = 0.6665
        assert!((mrr - 0.6665).abs() < 0.01);
    }
}
