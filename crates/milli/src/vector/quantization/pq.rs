use serde::{Deserialize, Serialize};

use super::QuantizationError;

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of subvectors (must divide dimension evenly)
    pub num_subvectors: usize,
    /// Bits per code (typically 8 or 16)
    pub bits_per_code: usize,
    /// Number of training iterations for k-means
    pub training_iterations: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subvectors: 96,  // For 768D vectors
            bits_per_code: 8,     // 256 centroids per subspace
            training_iterations: 20,
        }
    }
}

pub struct ProductQuantizer {
    config: PQConfig,
    /// Codebooks for each subvector (num_subvectors × 2^bits_per_code × subvector_dim)
    codebooks: Vec<Vec<Vec<f32>>>,
    dimension: usize,
    subvector_dim: usize,
}

impl ProductQuantizer {
    /// Train PQ codebooks on sample vectors
    pub fn train(
        config: PQConfig,
        training_vectors: &[Vec<f32>],
    ) -> Result<Self, QuantizationError> {
        let dimension = training_vectors.first()
            .ok_or(QuantizationError::EmptyTrainingSet)?
            .len();

        if dimension % config.num_subvectors != 0 {
            return Err(QuantizationError::DimensionMismatch);
        }

        let subvector_dim = dimension / config.num_subvectors;
        let num_centroids = 1 << config.bits_per_code; // 2^bits_per_code

        let mut codebooks = Vec::with_capacity(config.num_subvectors);

        // Train codebook for each subspace
        for subspace_idx in 0..config.num_subvectors {
            let start_dim = subspace_idx * subvector_dim;
            let end_dim = start_dim + subvector_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = training_vectors.iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();

            // Run k-means clustering
            let centroids = kmeans_clustering(
                &subvectors,
                num_centroids,
                config.training_iterations,
            )?;

            codebooks.push(centroids);
        }

        Ok(Self {
            config,
            codebooks,
            dimension,
            subvector_dim,
        })
    }

    /// Encode a vector using PQ
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.num_subvectors);

        for (subspace_idx, codebook) in self.codebooks.iter().enumerate() {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find nearest centroid
            let code = find_nearest_centroid(subvector, codebook);
            codes.push(code as u8);
        }

        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimension);

        for (code, codebook) in codes.iter().zip(self.codebooks.iter()) {
            let centroid = &codebook[*code as usize];
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Asymmetric distance computation (query vector vs PQ codes)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut distance_sq = 0.0;

        for (subspace_idx, &code) in codes.iter().enumerate() {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            let centroid = &self.codebooks[subspace_idx][code as usize];

            // Squared Euclidean distance for this subspace
            for (q, c) in query_subvector.iter().zip(centroid.iter()) {
                let diff = q - c;
                distance_sq += diff * diff;
            }
        }

        distance_sq.sqrt()
    }
}

pub(crate) fn kmeans_clustering(
    vectors: &[Vec<f32>],
    k: usize,
    iterations: usize,
) -> Result<Vec<Vec<f32>>, QuantizationError> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let dim = vectors.first().ok_or(QuantizationError::EmptyTrainingSet)?.len();
    let mut rng = thread_rng();

    // Initialize centroids randomly
    let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k)
        .map(|v| v.clone())
        .collect();

    for _ in 0..iterations {
        // Assignment step
        let mut clusters: Vec<Vec<Vec<f32>>> = vec![Vec::new(); k];

        for vector in vectors {
            let nearest_idx = find_nearest_centroid(vector, &centroids);
            clusters[nearest_idx].push(vector.clone());
        }

        // Update step
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue; // Keep previous centroid
            }

            let mut new_centroid = vec![0.0; dim];
            for vector in cluster {
                for (j, &val) in vector.iter().enumerate() {
                    new_centroid[j] += val;
                }
            }

            let cluster_size = cluster.len() as f32;
            for val in &mut new_centroid {
                *val /= cluster_size;
            }

            centroids[i] = new_centroid;
        }
    }

    Ok(centroids)
}

fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids.iter()
        .enumerate()
        .map(|(i, centroid)| {
            let dist_sq: f32 = vector.iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, dist_sq)
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_encode_decode() {
        let config = PQConfig {
            num_subvectors: 4,
            bits_per_code: 4,
            training_iterations: 10,
        };

        // Create simple training vectors (12D)
        let training_vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..12).map(|j| (i + j) as f32 / 100.0).collect()
            })
            .collect();

        let pq = ProductQuantizer::train(config, &training_vectors).unwrap();

        // Test encode/decode
        let test_vector: Vec<f32> = (0..12).map(|i| i as f32 / 10.0).collect();
        let codes = pq.encode(&test_vector);
        let decoded = pq.decode(&codes);

        assert_eq!(decoded.len(), test_vector.len());
        assert_eq!(codes.len(), 4); // num_subvectors
    }

    #[test]
    fn test_pq_dimension_mismatch() {
        let config = PQConfig {
            num_subvectors: 5, // 13 is not divisible by 5
            bits_per_code: 8,
            training_iterations: 10,
        };

        let training_vectors: Vec<Vec<f32>> = vec![vec![1.0; 13]];
        let result = ProductQuantizer::train(config, &training_vectors);

        assert!(matches!(result, Err(QuantizationError::DimensionMismatch)));
    }
}
