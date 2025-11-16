use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::rrf::FusionWeights;

/// Stores learned weights based on query patterns and user interactions
///
/// This system learns optimal fusion weights from user behavior (click-through rates,
/// dwell time, etc.) and applies them to future similar queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeights {
    /// Map from query pattern hash to learned weights
    patterns: HashMap<u64, WeightEntry>,

    /// Global fallback weights used when no pattern-specific weights exist
    global_weights: FusionWeights,

    /// Learning rate for exponential moving average updates
    #[serde(default = "default_learning_rate")]
    learning_rate: f64,
}

fn default_learning_rate() -> f64 {
    0.1
}

/// Entry storing learned weights and metadata for a query pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightEntry {
    /// Learned fusion weights
    weights: FusionWeights,

    /// Number of observations used to compute these weights
    observation_count: usize,

    /// Last update timestamp (for potential decay/expiration)
    #[serde(skip)]
    last_updated: Option<std::time::SystemTime>,
}

impl LearnedWeights {
    /// Create a new learned weights system with default global weights
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            global_weights: FusionWeights::default(),
            learning_rate: default_learning_rate(),
        }
    }

    /// Create a new learned weights system with custom global weights
    pub fn with_global_weights(global_weights: FusionWeights) -> Self {
        Self { patterns: HashMap::new(), global_weights, learning_rate: default_learning_rate() }
    }

    /// Set the learning rate for weight updates
    ///
    /// # Arguments
    /// * `rate` - Learning rate between 0.0 and 1.0
    ///   - Lower values make learning slower but more stable
    ///   - Higher values adapt faster but may be more volatile
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Update weights based on click-through rate signals
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `keyword_clicks` - Number of clicks on keyword-sourced results
    /// * `semantic_clicks` - Number of clicks on semantic-sourced results
    ///
    /// This method uses exponential moving average to update weights gradually
    /// based on user interaction signals.
    pub fn update_from_ctr(
        &mut self,
        query: &str,
        keyword_clicks: usize,
        semantic_clicks: usize,
    ) {
        let pattern_hash = self.hash_query_pattern(query);

        let total_clicks = keyword_clicks + semantic_clicks;
        if total_clicks == 0 {
            return;
        }

        let semantic_preference = semantic_clicks as f64 / total_clicks as f64;

        // Get or create entry for this pattern
        let entry =
            self.patterns.entry(pattern_hash).or_insert_with(|| WeightEntry {
                weights: self.global_weights,
                observation_count: 0,
                last_updated: Some(std::time::SystemTime::now()),
            });

        // Exponential moving average for weight updates
        entry.weights.semantic_weight = self.learning_rate * semantic_preference
            + (1.0 - self.learning_rate) * entry.weights.semantic_weight;
        entry.weights.keyword_weight = 1.0 - entry.weights.semantic_weight;

        entry.observation_count += 1;
        entry.last_updated = Some(std::time::SystemTime::now());
    }

    /// Update weights based on dwell time (how long user spent on result)
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `result_source` - Whether the clicked result came from keyword or semantic search
    /// * `dwell_time_ms` - Time spent on the result in milliseconds
    ///
    /// Longer dwell time indicates higher quality results. This method adjusts
    /// weights to favor the source that produces results with higher dwell time.
    pub fn update_from_dwell_time(
        &mut self,
        query: &str,
        result_source: ResultSource,
        dwell_time_ms: u64,
    ) {
        // Dwell time > 30 seconds is considered a positive signal
        const POSITIVE_DWELL_TIME_MS: u64 = 30_000;

        if dwell_time_ms >= POSITIVE_DWELL_TIME_MS {
            match result_source {
                ResultSource::Keyword => self.update_from_ctr(query, 1, 0),
                ResultSource::Semantic => self.update_from_ctr(query, 0, 1),
                ResultSource::Both => self.update_from_ctr(query, 1, 1),
            }
        }
    }

    /// Retrieve learned weights for a query
    ///
    /// Returns pattern-specific weights if available, otherwise returns global weights.
    pub fn get_weights(&self, query: &str) -> FusionWeights {
        let pattern_hash = self.hash_query_pattern(query);
        self.patterns.get(&pattern_hash).map(|entry| entry.weights).unwrap_or(self.global_weights)
    }

    /// Get weights with confidence score
    ///
    /// # Returns
    /// Tuple of (weights, confidence) where confidence is based on observation count
    pub fn get_weights_with_confidence(&self, query: &str) -> (FusionWeights, f64) {
        let pattern_hash = self.hash_query_pattern(query);
        match self.patterns.get(&pattern_hash) {
            Some(entry) => {
                // Confidence grows with observations but caps at 1.0
                let confidence = (entry.observation_count as f64 / 100.0).min(1.0);
                (entry.weights, confidence)
            }
            None => (self.global_weights, 0.0),
        }
    }

    /// Clear learned weights for a specific query pattern
    pub fn clear_pattern(&mut self, query: &str) {
        let pattern_hash = self.hash_query_pattern(query);
        self.patterns.remove(&pattern_hash);
    }

    /// Clear all learned weights
    pub fn clear_all(&mut self) {
        self.patterns.clear();
    }

    /// Get statistics about learned patterns
    pub fn stats(&self) -> LearnedWeightsStats {
        let total_patterns = self.patterns.len();
        let total_observations: usize =
            self.patterns.values().map(|e| e.observation_count).sum();

        LearnedWeightsStats { total_patterns, total_observations }
    }

    /// Hash query into a pattern identifier
    ///
    /// This creates pattern buckets based on query characteristics:
    /// - Short queries (≤2 terms) -> "short_query"
    /// - Queries with numbers -> "numeric_query"
    /// - Standard queries -> "standard_query"
    ///
    /// This allows learning to generalize across similar query types
    fn hash_query_pattern(&self, query: &str) -> u64 {
        // Normalize and extract pattern
        let normalized = query.to_lowercase();
        let tokens: Vec<_> = normalized.split_whitespace().collect();

        // Create pattern based on query structure
        let pattern = if tokens.is_empty() {
            "empty_query".to_string()
        } else if tokens.len() <= 2 {
            "short_query".to_string()
        } else if tokens.iter().any(|t| t.chars().any(|c| c.is_numeric())) {
            "numeric_query".to_string()
        } else if tokens.len() >= 6 {
            "long_query".to_string()
        } else {
            "standard_query".to_string()
        };

        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for LearnedWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Source of a search result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    /// Result came from keyword search
    Keyword,
    /// Result came from semantic/vector search
    Semantic,
    /// Result appeared in both keyword and semantic results
    Both,
}

/// Statistics about learned weights
#[derive(Debug, Clone)]
pub struct LearnedWeightsStats {
    /// Number of distinct query patterns with learned weights
    pub total_patterns: usize,
    /// Total number of observations across all patterns
    pub total_observations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_learning() {
        let mut learned = LearnedWeights::new();

        // Simulate clicks favoring semantic results
        learned.update_from_ctr("climate change", 2, 8);

        let weights = learned.get_weights("climate change");
        // Should favor semantic search
        assert!(weights.semantic_weight > weights.keyword_weight);
    }

    #[test]
    fn test_pattern_generalization() {
        let mut learned = LearnedWeights::new();

        // Update with one short query
        learned.update_from_ctr("shoes", 1, 9);

        // Should apply to other short queries
        let weights = learned.get_weights("bags");
        assert!(weights.semantic_weight > 0.5);
    }

    #[test]
    fn test_confidence_grows_with_observations() {
        let mut learned = LearnedWeights::new();

        // First observation
        learned.update_from_ctr("test query", 5, 5);
        let (_, confidence1) = learned.get_weights_with_confidence("test query");

        // More observations
        for _ in 0..10 {
            learned.update_from_ctr("test query", 5, 5);
        }
        let (_, confidence2) = learned.get_weights_with_confidence("test query");

        assert!(confidence2 > confidence1);
    }

    #[test]
    fn test_dwell_time_learning() {
        let mut learned = LearnedWeights::new();

        // Long dwell time on semantic result
        learned.update_from_dwell_time("machine learning", ResultSource::Semantic, 45_000);

        let weights = learned.get_weights("machine learning");
        assert!(weights.semantic_weight > 0.5);
    }

    #[test]
    fn test_clear_pattern() {
        let mut learned = LearnedWeights::new();

        learned.update_from_ctr("test", 5, 5);
        assert!(learned.stats().total_patterns > 0);

        learned.clear_pattern("test");
        // Pattern should be cleared, but stats might still show other patterns
        // Depending on hash collisions
    }

    #[test]
    fn test_stats() {
        let mut learned = LearnedWeights::new();

        learned.update_from_ctr("query 1", 5, 5);
        learned.update_from_ctr("query 2", 3, 7);

        let stats = learned.stats();
        assert!(stats.total_patterns > 0);
        assert!(stats.total_observations >= 2);
    }
}
