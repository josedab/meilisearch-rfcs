use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// SPLADE sparse vector representation
///
/// SPLADE (Sparse Lexical and Expansion Model) produces sparse vectors where each term
/// has a learned weight. This combines the efficiency of sparse inverted indexes with
/// the semantic understanding of neural models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpladeVector {
    /// Word ID to weight mapping (sparse)
    /// Only stores non-zero weights to maintain sparsity
    pub weights: HashMap<u32, f32>,
    /// L2 norm for normalization
    pub norm: f32,
}

impl SpladeVector {
    /// Creates a new SPLADE vector from a weight mapping
    ///
    /// Automatically computes the L2 norm for normalization
    pub fn new(weights: HashMap<u32, f32>) -> Self {
        let norm = weights.values().map(|&w| w * w).sum::<f32>().sqrt();

        Self { weights, norm }
    }

    /// Creates a new SPLADE vector with pre-computed norm
    pub fn new_with_norm(weights: HashMap<u32, f32>, norm: f32) -> Self {
        Self { weights, norm }
    }

    /// Normalize vector to unit L2 norm
    ///
    /// This modifies the vector in-place to have unit length
    pub fn normalize(&mut self) {
        if self.norm > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= self.norm;
            }
            self.norm = 1.0;
        }
    }

    /// Get the top-K terms by weight
    ///
    /// Returns a vector of (term_id, weight) tuples sorted by weight in descending order
    pub fn top_k_terms(&self, k: usize) -> Vec<(u32, f32)> {
        let mut terms: Vec<_> =
            self.weights.iter().map(|(&term_id, &weight)| (term_id, weight)).collect();

        terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        terms.truncate(k);

        terms
    }

    /// Compute dot product similarity with another sparse vector
    ///
    /// This is the primary scoring function for SPLADE retrieval.
    /// Only iterates over non-zero terms for efficiency.
    pub fn dot_product(&self, other: &SpladeVector) -> f32 {
        let mut score = 0.0;

        // Iterate over the smaller vector for efficiency
        let (smaller, larger) = if self.weights.len() < other.weights.len() {
            (&self.weights, &other.weights)
        } else {
            (&other.weights, &self.weights)
        };

        for (&term_id, &weight) in smaller {
            if let Some(&other_weight) = larger.get(&term_id) {
                score += weight * other_weight;
            }
        }

        score
    }

    /// Get the number of active (non-zero) terms
    pub fn active_term_count(&self) -> usize {
        self.weights.len()
    }

    /// Prune the vector to keep only top-K terms
    ///
    /// This reduces memory usage and can speed up retrieval
    pub fn prune_to_top_k(&mut self, k: usize) {
        if self.weights.len() <= k {
            return;
        }

        let top_k = self.top_k_terms(k);
        self.weights = top_k.into_iter().collect();

        // Recompute norm after pruning
        self.norm = self.weights.values().map(|&w| w * w).sum::<f32>().sqrt();
    }

    /// Apply a threshold to remove low-weight terms
    ///
    /// Terms with weight below the threshold are removed
    pub fn apply_threshold(&mut self, threshold: f32) {
        self.weights.retain(|_, &mut weight| weight >= threshold);

        // Recompute norm after thresholding
        self.norm = self.weights.values().map(|&w| w * w).sum::<f32>().sqrt();
    }

    /// Get the weight for a specific term
    pub fn get_weight(&self, term_id: u32) -> f32 {
        self.weights.get(&term_id).copied().unwrap_or(0.0)
    }

    /// Check if the vector is empty (no active terms)
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Default for SpladeVector {
    fn default() -> Self {
        Self { weights: HashMap::new(), norm: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splade_vector_creation() {
        let mut weights = HashMap::new();
        weights.insert(1, 0.8);
        weights.insert(2, 0.6);
        weights.insert(3, 0.4);

        let vec = SpladeVector::new(weights.clone());

        // Check norm calculation: sqrt(0.8^2 + 0.6^2 + 0.4^2) = sqrt(1.16) ≈ 1.077
        assert!((vec.norm - 1.077).abs() < 0.01);
        assert_eq!(vec.weights, weights);
    }

    #[test]
    fn test_normalize() {
        let mut weights = HashMap::new();
        weights.insert(1, 0.8);
        weights.insert(2, 0.6);

        let mut vec = SpladeVector::new(weights);
        vec.normalize();

        assert_eq!(vec.norm, 1.0);
        // After normalization, squared weights should sum to 1
        let sum_sq: f32 = vec.weights.values().map(|&w| w * w).sum();
        assert!((sum_sq - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_dot_product() {
        let mut weights1 = HashMap::new();
        weights1.insert(1, 0.8);
        weights1.insert(2, 0.6);
        weights1.insert(3, 0.4);

        let mut weights2 = HashMap::new();
        weights2.insert(1, 0.5);
        weights2.insert(2, 0.3);
        weights2.insert(4, 0.2); // This term doesn't overlap

        let vec1 = SpladeVector::new(weights1);
        let vec2 = SpladeVector::new(weights2);

        // Dot product: 0.8*0.5 + 0.6*0.3 = 0.4 + 0.18 = 0.58
        let score = vec1.dot_product(&vec2);
        assert!((score - 0.58).abs() < 0.0001);
    }

    #[test]
    fn test_top_k_terms() {
        let mut weights = HashMap::new();
        weights.insert(1, 0.8);
        weights.insert(2, 0.6);
        weights.insert(3, 0.4);
        weights.insert(4, 0.2);

        let vec = SpladeVector::new(weights);
        let top_2 = vec.top_k_terms(2);

        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0], (1, 0.8));
        assert_eq!(top_2[1], (2, 0.6));
    }

    #[test]
    fn test_prune_to_top_k() {
        let mut weights = HashMap::new();
        weights.insert(1, 0.8);
        weights.insert(2, 0.6);
        weights.insert(3, 0.4);
        weights.insert(4, 0.2);

        let mut vec = SpladeVector::new(weights);
        vec.prune_to_top_k(2);

        assert_eq!(vec.active_term_count(), 2);
        assert!(vec.weights.contains_key(&1));
        assert!(vec.weights.contains_key(&2));
        assert!(!vec.weights.contains_key(&3));
        assert!(!vec.weights.contains_key(&4));
    }

    #[test]
    fn test_apply_threshold() {
        let mut weights = HashMap::new();
        weights.insert(1, 0.8);
        weights.insert(2, 0.6);
        weights.insert(3, 0.4);
        weights.insert(4, 0.2);

        let mut vec = SpladeVector::new(weights);
        vec.apply_threshold(0.5);

        assert_eq!(vec.active_term_count(), 2);
        assert!(vec.weights.contains_key(&1));
        assert!(vec.weights.contains_key(&2));
        assert!(!vec.weights.contains_key(&3));
        assert!(!vec.weights.contains_key(&4));
    }

    #[test]
    fn test_empty_vector() {
        let vec = SpladeVector::default();
        assert!(vec.is_empty());
        assert_eq!(vec.norm, 0.0);
        assert_eq!(vec.active_term_count(), 0);
    }
}
