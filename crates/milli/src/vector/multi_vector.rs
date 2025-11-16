use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub type DocumentId = u32;

/// Multi-vector document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDocument {
    pub doc_id: DocumentId,
    pub embeddings: HashMap<String, Vec<f32>>,
}

impl MultiVectorDocument {
    /// Create a new multi-vector document
    pub fn new(doc_id: DocumentId) -> Self {
        Self {
            doc_id,
            embeddings: HashMap::new(),
        }
    }

    /// Add an embedding to the document
    pub fn add_embedding(&mut self, name: String, embedding: Vec<f32>) {
        self.embeddings.insert(name, embedding);
    }

    /// Get an embedding by name
    pub fn get_embedding(&self, name: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(name)
    }

    /// Get all embedding names
    pub fn embedding_names(&self) -> Vec<&String> {
        self.embeddings.keys().collect()
    }

    /// Number of embeddings in this document
    pub fn num_embeddings(&self) -> usize {
        self.embeddings.len()
    }
}

/// Multi-vector search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorQuery {
    pub queries: HashMap<String, Vec<f32>>,
    pub weights: HashMap<String, f32>,
}

impl MultiVectorQuery {
    /// Create a new multi-vector query
    pub fn new() -> Self {
        Self {
            queries: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    /// Add a query vector with optional weight
    pub fn add_query(&mut self, name: String, vector: Vec<f32>, weight: Option<f32>) {
        self.queries.insert(name.clone(), vector);
        if let Some(w) = weight {
            self.weights.insert(name, w);
        }
    }

    /// Get the weight for a query (defaults to 1.0 if not specified)
    pub fn get_weight(&self, name: &str) -> f32 {
        self.weights.get(name).copied().unwrap_or(1.0)
    }
}

impl Default for MultiVectorQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Result fusion strategy for multi-vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionStrategy {
    /// Weighted sum of scores
    WeightedSum,
    /// Maximum score across all vectors
    Max,
    /// Reciprocal Rank Fusion
    RRF,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::WeightedSum
    }
}

/// Fuse results from multiple vector searches
pub fn fuse_results(
    results_per_store: HashMap<String, Vec<(DocumentId, f32)>>,
    weights: &HashMap<String, f32>,
    strategy: FusionStrategy,
    limit: usize,
) -> Vec<(DocumentId, f32)> {
    match strategy {
        FusionStrategy::WeightedSum => fuse_weighted_sum(results_per_store, weights, limit),
        FusionStrategy::Max => fuse_max(results_per_store, limit),
        FusionStrategy::RRF => fuse_rrf(results_per_store, limit),
    }
}

fn fuse_weighted_sum(
    results_per_store: HashMap<String, Vec<(DocumentId, f32)>>,
    weights: &HashMap<String, f32>,
    limit: usize,
) -> Vec<(DocumentId, f32)> {
    let mut combined_scores: HashMap<DocumentId, f32> = HashMap::new();

    for (store_name, results) in results_per_store {
        let weight = weights.get(&store_name).copied().unwrap_or(1.0);

        for (doc_id, distance) in results {
            // Convert distance to similarity score (1 - distance)
            let score = (1.0 - distance) * weight;
            *combined_scores.entry(doc_id).or_insert(0.0) += score;
        }
    }

    let mut ranked: Vec<_> = combined_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(limit);

    // Convert back to distance (1 - score)
    ranked.into_iter()
        .map(|(doc_id, score)| (doc_id, 1.0 - score))
        .collect()
}

fn fuse_max(
    results_per_store: HashMap<String, Vec<(DocumentId, f32)>>,
    limit: usize,
) -> Vec<(DocumentId, f32)> {
    let mut best_scores: HashMap<DocumentId, f32> = HashMap::new();

    for (_store_name, results) in results_per_store {
        for (doc_id, distance) in results {
            let score = 1.0 - distance;
            best_scores.entry(doc_id)
                .and_modify(|current| *current = current.max(score))
                .or_insert(score);
        }
    }

    let mut ranked: Vec<_> = best_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(limit);

    ranked.into_iter()
        .map(|(doc_id, score)| (doc_id, 1.0 - score))
        .collect()
}

fn fuse_rrf(
    results_per_store: HashMap<String, Vec<(DocumentId, f32)>>,
    limit: usize,
) -> Vec<(DocumentId, f32)> {
    const K: f32 = 60.0; // Standard RRF constant

    let mut rrf_scores: HashMap<DocumentId, f32> = HashMap::new();

    for (_store_name, results) in results_per_store {
        for (rank, (doc_id, _distance)) in results.iter().enumerate() {
            let rrf_contribution = 1.0 / (K + (rank + 1) as f32);
            *rrf_scores.entry(*doc_id).or_insert(0.0) += rrf_contribution;
        }
    }

    let mut ranked: Vec<_> = rrf_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(limit);

    // For RRF, we return the RRF score as the "distance" (inverted)
    ranked.into_iter()
        .map(|(doc_id, score)| (doc_id, 1.0 / (1.0 + score)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_vector_document() {
        let mut doc = MultiVectorDocument::new(1);
        doc.add_embedding("title".to_string(), vec![1.0, 2.0, 3.0]);
        doc.add_embedding("content".to_string(), vec![4.0, 5.0, 6.0]);

        assert_eq!(doc.num_embeddings(), 2);
        assert!(doc.get_embedding("title").is_some());
        assert!(doc.get_embedding("content").is_some());
        assert!(doc.get_embedding("missing").is_none());
    }

    #[test]
    fn test_multi_vector_query() {
        let mut query = MultiVectorQuery::new();
        query.add_query("title".to_string(), vec![1.0, 2.0], Some(2.0));
        query.add_query("content".to_string(), vec![3.0, 4.0], None);

        assert_eq!(query.get_weight("title"), 2.0);
        assert_eq!(query.get_weight("content"), 1.0);
        assert_eq!(query.get_weight("missing"), 1.0);
    }

    #[test]
    fn test_weighted_sum_fusion() {
        let mut results_per_store = HashMap::new();
        results_per_store.insert(
            "store1".to_string(),
            vec![(1, 0.1), (2, 0.2), (3, 0.3)],
        );
        results_per_store.insert(
            "store2".to_string(),
            vec![(1, 0.2), (2, 0.1), (4, 0.4)],
        );

        let mut weights = HashMap::new();
        weights.insert("store1".to_string(), 1.0);
        weights.insert("store2".to_string(), 1.0);

        let results = fuse_weighted_sum(results_per_store, &weights, 3);

        // Doc 1 should have the best combined score
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_rrf_fusion() {
        let mut results_per_store = HashMap::new();
        results_per_store.insert(
            "store1".to_string(),
            vec![(1, 0.1), (2, 0.2), (3, 0.3)],
        );
        results_per_store.insert(
            "store2".to_string(),
            vec![(2, 0.1), (1, 0.2), (4, 0.4)],
        );

        let results = fuse_rrf(results_per_store, 3);

        // Results should be ordered by RRF score
        assert!(results.len() <= 3);
    }
}
