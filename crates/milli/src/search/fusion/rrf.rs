use std::collections::HashMap;

use crate::score_details::ScoreDetails;
use crate::DocumentId;

/// Reciprocal Rank Fusion scoring
///
/// RRF is a rank-based fusion method that combines results from multiple retrieval systems
/// without requiring score normalization. The formula is:
///
/// RRF_score(d) = Σ (1 / (k + rank_i(d)))
///
/// Where:
/// - `d` is a document
/// - `k` is a constant (typically 60)
/// - `rank_i(d)` is the rank of document `d` in result set `i`
pub struct RRFScorer {
    /// The k parameter controls rank sensitivity
    /// Higher values give less weight to top-ranked documents
    k: f64,
}

impl RRFScorer {
    /// Create a new RRF scorer with the given k parameter
    ///
    /// The k parameter is typically set to 60 based on research literature.
    /// Lower values give more weight to top-ranked documents.
    pub fn new(k: f64) -> Self {
        Self { k }
    }

    /// Compute RRF scores for documents appearing in multiple result lists
    ///
    /// # Arguments
    /// * `keyword_results` - Results from keyword search, ordered by relevance
    /// * `vector_results` - Results from vector/semantic search, ordered by relevance
    /// * `weights` - Fusion weights to apply to each retrieval method
    ///
    /// # Returns
    /// A vector of (DocumentId, RRF score) tuples, sorted by RRF score descending
    pub fn score(
        &self,
        keyword_results: &[(DocumentId, Vec<ScoreDetails>)],
        vector_results: &[(DocumentId, Vec<ScoreDetails>)],
        weights: &FusionWeights,
    ) -> Vec<(DocumentId, f64)> {
        let mut rrf_scores: HashMap<DocumentId, f64> = HashMap::new();

        // Score keyword results
        // Each document gets a score based on its rank: weight / (k + rank + 1)
        for (rank, (docid, _scores)) in keyword_results.iter().enumerate() {
            let rrf_contribution = weights.keyword_weight / (self.k + (rank as f64) + 1.0);
            *rrf_scores.entry(*docid).or_insert(0.0) += rrf_contribution;
        }

        // Score vector results
        // Documents appearing in both lists will have their scores summed
        for (rank, (docid, _scores)) in vector_results.iter().enumerate() {
            let rrf_contribution = weights.semantic_weight / (self.k + (rank as f64) + 1.0);
            *rrf_scores.entry(*docid).or_insert(0.0) += rrf_contribution;
        }

        // Sort by RRF score descending
        let mut results: Vec<_> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }
}

/// Weights for fusion strategies
///
/// These weights determine how much each retrieval method (keyword vs semantic)
/// contributes to the final ranking.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FusionWeights {
    /// Weight applied to keyword search results
    pub keyword_weight: f64,
    /// Weight applied to semantic/vector search results
    pub semantic_weight: f64,
}

impl FusionWeights {
    /// Create fusion weights from a semantic ratio
    ///
    /// # Arguments
    /// * `semantic_ratio` - Value between 0.0 and 1.0 indicating semantic weight
    ///   - 0.0 = pure keyword search
    ///   - 1.0 = pure semantic search
    ///   - 0.5 = equal weighting
    pub fn from_semantic_ratio(semantic_ratio: f32) -> Self {
        Self {
            keyword_weight: (1.0 - semantic_ratio) as f64,
            semantic_weight: semantic_ratio as f64,
        }
    }

    /// Create balanced fusion weights (equal weighting)
    pub fn balanced() -> Self {
        Self { keyword_weight: 1.0, semantic_weight: 1.0 }
    }
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_basic() {
        let scorer = RRFScorer::new(60.0);
        let weights = FusionWeights::balanced();

        // Document 1 appears first in keyword results
        let keyword_results = vec![(1, vec![]), (2, vec![]), (3, vec![])];

        // Document 2 appears first in vector results
        let vector_results = vec![(2, vec![]), (1, vec![]), (4, vec![])];

        let scores = scorer.score(&keyword_results, &vector_results, &weights);

        // Document 1 should have: 1/(60+0+1) + 1/(60+1+1) ≈ 0.01639 + 0.01613 ≈ 0.03252
        // Document 2 should have: 1/(60+1+1) + 1/(60+0+1) ≈ 0.01613 + 0.01639 ≈ 0.03252
        // Both should have similar scores since they appear highly ranked in both lists
        assert_eq!(scores[0].0, 1);
        assert_eq!(scores[1].0, 2);
        assert!((scores[0].1 - scores[1].1).abs() < 0.001);
    }

    #[test]
    fn test_rrf_weights() {
        let scorer = RRFScorer::new(60.0);
        // Favor keyword search
        let weights = FusionWeights { keyword_weight: 2.0, semantic_weight: 1.0 };

        let keyword_results = vec![(1, vec![])];
        let vector_results = vec![(2, vec![])];

        let scores = scorer.score(&keyword_results, &vector_results, &weights);

        // Document 1 (keyword) should score higher due to higher weight
        assert_eq!(scores[0].0, 1);
        assert!(scores[0].1 > scores[1].1);
    }

    #[test]
    fn test_fusion_weights_from_ratio() {
        let weights = FusionWeights::from_semantic_ratio(0.7);
        assert!((weights.keyword_weight - 0.3).abs() < 0.001);
        assert!((weights.semantic_weight - 0.7).abs() < 0.001);
    }
}
