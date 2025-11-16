/// Fusion strategies for combining keyword and semantic search results
///
/// This module provides advanced fusion techniques for hybrid search:
/// - RRF (Reciprocal Rank Fusion): Rank-based fusion without score normalization
/// - Adaptive fusion: Query-aware dynamic weight adjustment
/// - Learned weights: User interaction-based weight optimization

pub mod adaptive;
pub mod learning;
pub mod rrf;

pub use adaptive::{AdaptiveConfig, QueryAnalyzer};
pub use learning::LearnedWeights;
pub use rrf::{FusionWeights, RRFScorer};

/// Strategy for fusing keyword and semantic search results
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Weighted score combination (current/legacy implementation)
    ///
    /// Combines scores using: final_score = keyword_score * (1 - ratio) + semantic_score * ratio
    ///
    /// # Example
    /// ```ignore
    /// FusionStrategy::WeightedSum { semantic_ratio: 0.5 }
    /// ```
    WeightedSum {
        /// Ratio between 0.0 (pure keyword) and 1.0 (pure semantic)
        semantic_ratio: f32,
    },

    /// Reciprocal Rank Fusion
    ///
    /// Uses rank-based scoring without requiring score normalization:
    /// RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
    ///
    /// # Example
    /// ```ignore
    /// FusionStrategy::RRF {
    ///     k: 60.0,
    ///     weights: FusionWeights::balanced(),
    /// }
    /// ```
    RRF {
        /// The k parameter controls rank sensitivity (typically 60)
        k: f64,
        /// Weights for keyword and semantic results
        weights: FusionWeights,
    },

    /// Adaptive fusion - automatically determines strategy based on query
    ///
    /// Analyzes query characteristics to decide whether to favor keyword
    /// or semantic search. Uses heuristics like:
    /// - Query length
    /// - Presence of numbers
    /// - Navigational vs exploratory indicators
    ///
    /// # Example
    /// ```ignore
    /// FusionStrategy::Adaptive {
    ///     config: AdaptiveConfig::default(),
    /// }
    /// ```
    Adaptive {
        /// Configuration for adaptive behavior
        config: AdaptiveConfig,
    },

    /// Learned fusion - uses historical user interactions
    ///
    /// Applies weights learned from user behavior (clicks, dwell time)
    /// to optimize fusion for query patterns.
    ///
    /// # Example
    /// ```ignore
    /// FusionStrategy::Learned {
    ///     weights: learned_weights,
    ///     fallback: Box::new(FusionStrategy::WeightedSum { semantic_ratio: 0.5 }),
    /// }
    /// ```
    Learned {
        /// Learned weights system
        weights: LearnedWeights,
        /// Fallback strategy when no learned weights available
        fallback: Box<FusionStrategy>,
    },
}

impl FusionStrategy {
    /// Create a weighted sum strategy with equal weighting
    pub fn balanced() -> Self {
        Self::WeightedSum { semantic_ratio: 0.5 }
    }

    /// Create an RRF strategy with balanced weights and standard k value
    pub fn rrf_balanced() -> Self {
        Self::RRF { k: 60.0, weights: FusionWeights::balanced() }
    }

    /// Create an adaptive strategy with default configuration
    pub fn adaptive() -> Self {
        Self::Adaptive { config: AdaptiveConfig::default() }
    }

    /// Get the name of this strategy for logging/debugging
    pub fn name(&self) -> &'static str {
        match self {
            Self::WeightedSum { .. } => "weighted_sum",
            Self::RRF { .. } => "rrf",
            Self::Adaptive { .. } => "adaptive",
            Self::Learned { .. } => "learned",
        }
    }
}

impl Default for FusionStrategy {
    /// Default strategy is weighted sum with equal weighting
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy_names() {
        assert_eq!(FusionStrategy::balanced().name(), "weighted_sum");
        assert_eq!(FusionStrategy::rrf_balanced().name(), "rrf");
        assert_eq!(FusionStrategy::adaptive().name(), "adaptive");
    }

    #[test]
    fn test_default_strategy() {
        let default = FusionStrategy::default();
        assert_eq!(default.name(), "weighted_sum");
    }
}
