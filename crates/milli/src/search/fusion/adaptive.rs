use charabia::Tokenize;
use std::collections::HashSet;

use super::rrf::FusionWeights;

/// Analyzes query characteristics to determine optimal fusion strategy
///
/// This analyzer examines query text to extract features that indicate
/// whether keyword or semantic search should be preferred. For example:
/// - Queries with numbers often need exact matching (favor keyword)
/// - Short exploratory queries benefit from semantic expansion
/// - Navigational queries with specific terms favor keyword matching
pub struct QueryAnalyzer {
    config: AdaptiveConfig,
}

/// Configuration for adaptive fusion behavior
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Keywords indicating navigational intent (prefer keyword search)
    /// Examples: "where", "how to", "buy", "price", "size", "color"
    pub navigational_indicators: Vec<String>,

    /// Keywords indicating exploratory intent (prefer semantic search)
    /// Examples: "similar", "like", "about", "related", "concept"
    pub exploratory_indicators: Vec<String>,

    /// Threshold for query specificity (based on unique terms)
    /// Queries with more unique terms than this threshold are considered specific
    pub specificity_threshold: usize,

    /// Default semantic ratio when no clear signal
    /// Value between 0.0 (pure keyword) and 1.0 (pure semantic)
    pub default_semantic_ratio: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            navigational_indicators: vec![
                "where".to_string(),
                "how to".to_string(),
                "buy".to_string(),
                "price".to_string(),
                "size".to_string(),
                "color".to_string(),
                "find".to_string(),
                "get".to_string(),
            ],
            exploratory_indicators: vec![
                "similar".to_string(),
                "like".to_string(),
                "about".to_string(),
                "related".to_string(),
                "concept".to_string(),
                "explain".to_string(),
                "overview".to_string(),
            ],
            specificity_threshold: 5,
            default_semantic_ratio: 0.5,
        }
    }
}

/// Features extracted from a query
#[derive(Debug)]
pub struct QueryFeatures {
    /// Query contains navigational indicators
    pub is_navigational: bool,

    /// Query contains exploratory indicators
    pub is_exploratory: bool,

    /// Query contains numeric characters
    pub has_numbers: bool,

    /// Query contains exact phrase matches (quoted strings)
    pub has_exact_phrases: bool,

    /// Number of unique terms in the query
    pub unique_term_count: usize,

    /// Average length of terms in the query
    pub avg_term_length: f32,
}

impl QueryAnalyzer {
    /// Create a new query analyzer with the given configuration
    pub fn new(config: AdaptiveConfig) -> Self {
        Self { config }
    }

    /// Create a new query analyzer with default configuration
    pub fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }

    /// Extract features from query text
    ///
    /// # Arguments
    /// * `query` - The search query text to analyze
    ///
    /// # Returns
    /// QueryFeatures containing various signals about the query
    pub fn analyze_query(&self, query: &str) -> QueryFeatures {
        let query_lower = query.to_lowercase();

        // Tokenize query using charabia
        let tokens: Vec<_> = query.tokenize().collect();
        let unique_terms: HashSet<String> =
            tokens.iter().map(|t| t.lemma().to_string()).collect();
        let unique_term_count = unique_terms.len();

        let avg_term_length = if !tokens.is_empty() {
            tokens.iter().map(|t| t.lemma().len()).sum::<usize>() as f32 / tokens.len() as f32
        } else {
            0.0
        };

        // Detect navigational intent
        let is_navigational =
            self.config.navigational_indicators.iter().any(|indicator| query_lower.contains(indicator));

        // Detect exploratory intent
        let is_exploratory =
            self.config.exploratory_indicators.iter().any(|indicator| query_lower.contains(indicator));

        // Detect numbers (product codes, sizes, etc.)
        let has_numbers = query.chars().any(|c| c.is_numeric());

        // Detect exact phrases (quoted strings)
        let has_exact_phrases = query.contains('"');

        QueryFeatures {
            is_navigational,
            is_exploratory,
            has_numbers,
            has_exact_phrases,
            unique_term_count,
            avg_term_length,
        }
    }

    /// Determine optimal semantic ratio based on query features
    ///
    /// This method applies heuristics to adjust the semantic ratio:
    /// - Navigational queries: decrease semantic weight
    /// - Exploratory queries: increase semantic weight
    /// - Queries with numbers: decrease semantic weight (favor exact matching)
    /// - Exact phrases: decrease semantic weight (user wants precision)
    /// - Specific queries (many terms): decrease semantic weight
    /// - Short queries: increase semantic weight (benefit from expansion)
    ///
    /// # Arguments
    /// * `features` - Query features extracted from analyze_query
    ///
    /// # Returns
    /// Semantic ratio between 0.0 and 1.0
    pub fn compute_semantic_ratio(&self, features: &QueryFeatures) -> f32 {
        let mut semantic_ratio = self.config.default_semantic_ratio;

        // Increase keyword weight for navigational queries
        if features.is_navigational {
            semantic_ratio -= 0.2;
        }

        // Increase semantic weight for exploratory queries
        if features.is_exploratory {
            semantic_ratio += 0.2;
        }

        // Queries with numbers often need exact matching
        if features.has_numbers {
            semantic_ratio -= 0.15;
        }

        // Exact phrases indicate desire for precision
        if features.has_exact_phrases {
            semantic_ratio -= 0.15;
        }

        // Long, specific queries benefit from keyword precision
        if features.unique_term_count >= self.config.specificity_threshold {
            semantic_ratio -= 0.1;
        }

        // Short queries benefit from semantic expansion
        if features.unique_term_count <= 2 {
            semantic_ratio += 0.15;
        }

        // Clamp to valid range [0.0, 1.0]
        semantic_ratio.clamp(0.0, 1.0)
    }

    /// Compute fusion weights based on query analysis
    ///
    /// # Arguments
    /// * `query` - The search query text
    ///
    /// # Returns
    /// FusionWeights optimized for the query characteristics
    pub fn compute_fusion_weights(&self, query: &str) -> FusionWeights {
        let features = self.analyze_query(query);
        let semantic_ratio = self.compute_semantic_ratio(&features);
        FusionWeights::from_semantic_ratio(semantic_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigational_query() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("where to buy red shoes");

        assert!(features.is_navigational);
        assert!(!features.is_exploratory);
        assert!(!features.has_numbers);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor keyword search
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_exploratory_query() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("articles about climate change");

        assert!(!features.is_navigational);
        assert!(features.is_exploratory);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor semantic search
        assert!(ratio > 0.5);
    }

    #[test]
    fn test_query_with_numbers() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("nike shoes size 10");

        assert!(features.has_numbers);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor keyword search for exact matching
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_exact_phrase_query() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("\"climate change\" impacts");

        assert!(features.has_exact_phrases);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor keyword search for precision
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_short_query() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("shoes");

        assert_eq!(features.unique_term_count, 1);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor semantic search for expansion
        assert!(ratio > 0.5);
    }

    #[test]
    fn test_specific_long_query() {
        let analyzer = QueryAnalyzer::default();
        let features = analyzer.analyze_query("red nike running shoes with air cushion technology");

        assert!(features.unique_term_count >= 5);

        let ratio = analyzer.compute_semantic_ratio(&features);
        // Should favor keyword search for specific queries
        assert!(ratio < 0.5);
    }

    #[test]
    fn test_fusion_weights_computation() {
        let analyzer = QueryAnalyzer::default();
        let weights = analyzer.compute_fusion_weights("climate change");

        assert!(weights.keyword_weight >= 0.0);
        assert!(weights.semantic_weight >= 0.0);
        // Weights should sum to represent a valid ratio
        assert!((weights.keyword_weight + weights.semantic_weight - 1.0).abs() < 1.0);
    }
}
