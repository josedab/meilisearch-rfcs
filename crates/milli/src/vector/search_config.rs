use serde::{Deserialize, Serialize};

/// Configuration for vector search behavior
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSearchConfig {
    /// Exploration factor for HNSW search
    /// Higher = more accurate but slower
    /// Typical range: 50-500
    pub ef_search: Option<usize>,

    /// For IVF indexes: number of clusters to search
    pub nprobe: Option<usize>,

    /// Minimum similarity threshold (filter low-quality results)
    /// Range: 0.0 to 1.0
    pub min_similarity: Option<f32>,

    /// Maximum distance threshold
    pub max_distance: Option<f32>,
}

impl VectorSearchConfig {
    /// Create a new search configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ef_search parameter
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    /// Set the nprobe parameter
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    /// Set the minimum similarity threshold
    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }

    /// Set the maximum distance threshold
    pub fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = Some(max_distance);
        self
    }

    /// Get the effective ef_search value for a given limit
    pub fn effective_ef_search(&self, limit: usize) -> usize {
        self.ef_search.unwrap_or_else(|| (limit * 10).max(100))
    }

    /// Get the effective nprobe value
    pub fn effective_nprobe(&self) -> usize {
        self.nprobe.unwrap_or(8)
    }

    /// Check if a distance passes the configured thresholds
    pub fn passes_thresholds(&self, distance: f32) -> bool {
        if let Some(max_dist) = self.max_distance {
            if distance > max_dist {
                return false;
            }
        }

        if let Some(min_sim) = self.min_similarity {
            let similarity = 1.0 - distance;
            if similarity < min_sim {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VectorSearchConfig::new();
        assert_eq!(config.effective_ef_search(10), 100);
        assert_eq!(config.effective_ef_search(20), 200);
        assert_eq!(config.effective_nprobe(), 8);
    }

    #[test]
    fn test_custom_config() {
        let config = VectorSearchConfig::new()
            .with_ef_search(500)
            .with_min_similarity(0.8)
            .with_max_distance(0.5);

        assert_eq!(config.effective_ef_search(10), 500);
        assert!(config.passes_thresholds(0.15)); // similarity = 0.85
        assert!(!config.passes_thresholds(0.25)); // similarity = 0.75
        assert!(!config.passes_thresholds(0.6)); // distance too large
    }

    #[test]
    fn test_threshold_filtering() {
        let config = VectorSearchConfig::new()
            .with_min_similarity(0.7);

        assert!(config.passes_thresholds(0.2)); // similarity = 0.8
        assert!(config.passes_thresholds(0.3)); // similarity = 0.7
        assert!(!config.passes_thresholds(0.4)); // similarity = 0.6
    }
}
