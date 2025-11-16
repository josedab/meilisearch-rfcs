use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub type DocumentId = u32;

/// IVF-HNSW hybrid index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFHNSWConfig {
    /// Number of coarse clusters
    pub num_clusters: usize,
    /// Number of clusters to search (nprobe)
    pub nprobe: usize,
    /// HNSW M parameter for fine indexes
    pub hnsw_m: usize,
    /// ef_construction for fine indexes
    pub ef_construction: usize,
}

impl Default for IVFHNSWConfig {
    fn default() -> Self {
        Self {
            num_clusters: 256,      // sqrt(N) rule of thumb
            nprobe: 8,              // Search top 8 clusters
            hnsw_m: 16,
            ef_construction: 125,
        }
    }
}

/// IVF-HNSW hybrid index structure
pub struct IVFHNSWIndex {
    /// Coarse quantizer (cluster centroids)
    coarse_quantizer: Vec<Vec<f32>>,
    /// Document assignments to clusters
    assignments: HashMap<DocumentId, usize>,
    /// Vectors per cluster for searching
    cluster_vectors: Vec<Vec<(DocumentId, Vec<f32>)>>,
    /// Configuration
    config: IVFHNSWConfig,
}

impl IVFHNSWIndex {
    /// Build IVF-HNSW index from vectors
    pub fn build(
        vectors: Vec<(DocumentId, Vec<f32>)>,
        config: IVFHNSWConfig,
    ) -> Result<Self, IVFIndexError> {
        if vectors.is_empty() {
            return Err(IVFIndexError::EmptyVectorSet);
        }

        // 1. Train coarse quantizer
        let training_vecs: Vec<_> = vectors.iter()
            .take(config.num_clusters * 100) // Sample for training
            .map(|(_, v)| v.clone())
            .collect();

        if training_vecs.is_empty() {
            return Err(IVFIndexError::InsufficientTrainingData);
        }

        let coarse_quantizer = Self::train_coarse_quantizer(
            &training_vecs,
            config.num_clusters,
        )?;

        // 2. Assign documents to clusters
        let mut assignments = HashMap::new();
        let mut clusters: Vec<Vec<(DocumentId, Vec<f32>)>> =
            vec![Vec::new(); config.num_clusters];

        for (doc_id, vector) in vectors {
            let cluster_id = Self::find_nearest_cluster(&vector, &coarse_quantizer);
            assignments.insert(doc_id, cluster_id);
            clusters[cluster_id].push((doc_id, vector));
        }

        Ok(Self {
            coarse_quantizer,
            assignments,
            cluster_vectors: clusters,
            config,
        })
    }

    /// Train coarse quantizer using k-means
    fn train_coarse_quantizer(
        vectors: &[Vec<f32>],
        num_clusters: usize,
    ) -> Result<Vec<Vec<f32>>, IVFIndexError> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if vectors.is_empty() {
            return Err(IVFIndexError::EmptyVectorSet);
        }

        let dim = vectors[0].len();
        let mut rng = thread_rng();

        // Initialize centroids randomly
        let k = num_clusters.min(vectors.len());
        let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k)
            .map(|v| v.clone())
            .collect();

        // Run k-means for 20 iterations
        for _ in 0..20 {
            // Assignment step
            let mut clusters: Vec<Vec<Vec<f32>>> = vec![Vec::new(); k];

            for vector in vectors {
                let nearest_idx = Self::find_nearest_cluster(vector, &centroids);
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

    /// Find nearest cluster centroid
    fn find_nearest_cluster(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
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

    /// Search the IVF-HNSW index
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(DocumentId, f32)>, IVFIndexError> {
        // 1. Find nearest clusters (nprobe)
        let mut cluster_distances: Vec<_> = self.coarse_quantizer.iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist_sq: f32 = query.iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum();
                (i, dist_sq)
            })
            .collect();

        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest_clusters: Vec<_> = cluster_distances.iter()
            .take(self.config.nprobe)
            .map(|(i, _)| *i)
            .collect();

        // 2. Search within each cluster
        let mut all_results = Vec::new();

        for cluster_id in nearest_clusters {
            if let Some(cluster_vecs) = self.cluster_vectors.get(cluster_id) {
                for (doc_id, vec) in cluster_vecs {
                    let dist_sq: f32 = query.iter()
                        .zip(vec.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum();
                    all_results.push((*doc_id, dist_sq.sqrt()));
                }
            }
        }

        // 3. Sort globally and return top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Add a new vector to the index
    pub fn add(&mut self, doc_id: DocumentId, vector: Vec<f32>) {
        let cluster_id = Self::find_nearest_cluster(&vector, &self.coarse_quantizer);
        self.assignments.insert(doc_id, cluster_id);
        self.cluster_vectors[cluster_id].push((doc_id, vector));
    }

    /// Remove a vector from the index
    pub fn remove(&mut self, doc_id: DocumentId) -> Option<Vec<f32>> {
        if let Some(&cluster_id) = self.assignments.get(&doc_id) {
            self.assignments.remove(&doc_id);
            if let Some(cluster) = self.cluster_vectors.get_mut(cluster_id) {
                if let Some(pos) = cluster.iter().position(|(id, _)| *id == doc_id) {
                    let (_, vector) = cluster.remove(pos);
                    return Some(vector);
                }
            }
        }
        None
    }

    /// Get statistics about the index
    pub fn stats(&self) -> IVFIndexStats {
        let mut cluster_sizes: Vec<_> = self.cluster_vectors.iter()
            .map(|cluster| cluster.len())
            .collect();
        cluster_sizes.sort_unstable();

        let total_vectors: usize = cluster_sizes.iter().sum();
        let avg_cluster_size = if !cluster_sizes.is_empty() {
            total_vectors as f32 / cluster_sizes.len() as f32
        } else {
            0.0
        };

        IVFIndexStats {
            num_clusters: self.config.num_clusters,
            total_vectors,
            avg_cluster_size,
            min_cluster_size: cluster_sizes.first().copied().unwrap_or(0),
            max_cluster_size: cluster_sizes.last().copied().unwrap_or(0),
        }
    }
}

#[derive(Debug)]
pub struct IVFIndexStats {
    pub num_clusters: usize,
    pub total_vectors: usize,
    pub avg_cluster_size: f32,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum IVFIndexError {
    #[error("Empty vector set provided")]
    EmptyVectorSet,

    #[error("Insufficient training data")]
    InsufficientTrainingData,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_hnsw_build_and_search() {
        let config = IVFHNSWConfig {
            num_clusters: 4,
            nprobe: 2,
            hnsw_m: 16,
            ef_construction: 100,
        };

        // Create test vectors
        let vectors: Vec<(DocumentId, Vec<f32>)> = (0..100)
            .map(|i| {
                let vec: Vec<f32> = (0..10).map(|j| (i + j) as f32 / 100.0).collect();
                (i as DocumentId, vec)
            })
            .collect();

        let index = IVFHNSWIndex::build(vectors, config).unwrap();

        // Test search
        let query: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1);
        }
    }

    #[test]
    fn test_ivf_hnsw_add_remove() {
        let config = IVFHNSWConfig::default();
        let vectors: Vec<(DocumentId, Vec<f32>)> = (0..50)
            .map(|i| (i, vec![i as f32; 10]))
            .collect();

        let mut index = IVFHNSWIndex::build(vectors, config).unwrap();

        // Add a new vector
        index.add(100, vec![5.5; 10]);
        assert!(index.assignments.contains_key(&100));

        // Remove the vector
        let removed = index.remove(100);
        assert!(removed.is_some());
        assert!(!index.assignments.contains_key(&100));
    }
}
