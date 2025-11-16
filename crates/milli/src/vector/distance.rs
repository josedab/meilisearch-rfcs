use serde::{Deserialize, Serialize};

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorDistanceMetric {
    /// Cosine similarity (current default)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product similarity
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

impl Default for VectorDistanceMetric {
    fn default() -> Self {
        VectorDistanceMetric::Cosine
    }
}

pub trait DistanceFunction {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

impl DistanceFunction for VectorDistanceMetric {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_distance(a, b),
            Self::Euclidean => euclidean_distance(a, b),
            Self::DotProduct => dot_product_distance(a, b),
            Self::Manhattan => manhattan_distance(a, b),
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = dot / (norm_a * norm_b);
    1.0 - similarity
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // Convert to distance (higher dot product = more similar, so negate)
    -dot
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.001); // Orthogonal vectors

        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!(dist.abs() < 0.001); // Identical vectors
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001); // 3-4-5 triangle

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let dist = euclidean_distance(&a, &b);
        assert!(dist.abs() < 0.001); // Identical vectors
    }

    #[test]
    fn test_dot_product_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let dist = dot_product_distance(&a, &b);
        // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
        assert!((dist + 20.0).abs() < 0.001);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        let dist = manhattan_distance(&a, &b);
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((dist - 12.0).abs() < 0.001);
    }
}
