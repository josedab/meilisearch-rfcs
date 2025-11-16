use crate::query_logger::QueryLog;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

pub struct ABTestEngine {
    /// Active experiments
    experiments: Arc<RwLock<HashMap<String, Experiment>>>,
    /// Results tracker
    results: Arc<RwLock<HashMap<String, ExperimentResults>>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub experiment_id: String,
    pub index_uid: String,
    pub variants: Vec<Variant>,
    pub traffic_split: Vec<f32>,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub status: ExperimentStatus,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Variant {
    pub variant_id: String,
    pub settings: Settings,
    pub description: String,
}

/// Simplified settings structure for A/B testing
/// In a real implementation, this would match meilisearch-types Settings
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_rules: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub searchable_attributes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub displayed_attributes: Option<Vec<String>>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    Draft,
    Running,
    Completed,
    Stopped,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    pub variant_metrics: HashMap<String, VariantMetrics>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct VariantMetrics {
    pub impressions: usize,
    pub clicks: usize,
    pub ctr: f64,
    pub mrr: f64,
    pub avg_processing_time_ms: f64,
    pub zero_results_rate: f64,
}

impl ABTestEngine {
    pub fn new() -> Self {
        Self {
            experiments: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&self, experiment: Experiment) -> crate::Result<()> {
        let mut experiments = self.experiments.write().unwrap();
        experiments.insert(experiment.experiment_id.clone(), experiment);
        Ok(())
    }

    /// Assign user to experiment variant
    pub fn assign_variant(&self, experiment_id: &str, user_id: &str) -> Option<String> {
        let experiments = self.experiments.read().unwrap();
        let experiment = experiments.get(experiment_id)?;

        if !matches!(experiment.status, ExperimentStatus::Running) {
            return None;
        }

        // Consistent hash-based assignment
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        experiment_id.hash(&mut hasher);
        let hash = hasher.finish();

        let normalized = (hash % 10000) as f32 / 10000.0;

        // Determine variant based on traffic split
        let mut cumulative = 0.0;
        for (i, &split) in experiment.traffic_split.iter().enumerate() {
            cumulative += split;
            if normalized < cumulative {
                return Some(experiment.variants[i].variant_id.clone());
            }
        }

        Some(experiment.variants.last()?.variant_id.clone())
    }

    /// Record query result for experiment
    pub fn record_query(
        &self,
        experiment_id: &str,
        variant_id: &str,
        query_log: &QueryLog,
    ) {
        let mut results = self.results.write().unwrap();
        let exp_results = results.entry(experiment_id.to_string()).or_default();
        let metrics = exp_results.variant_metrics.entry(variant_id.to_string()).or_default();

        metrics.impressions += 1;
        metrics.clicks += query_log.total_clicks;

        // Update running averages
        let n = metrics.impressions as f64;
        metrics.ctr = metrics.clicks as f64 / n;
        metrics.avg_processing_time_ms =
            (metrics.avg_processing_time_ms * (n - 1.0) + query_log.processing_time_ms as f64) / n;

        if query_log.hits_count == 0 {
            metrics.zero_results_rate =
                (metrics.zero_results_rate * (n - 1.0) + 1.0) / n;
        } else {
            metrics.zero_results_rate =
                (metrics.zero_results_rate * (n - 1.0)) / n;
        }
    }

    /// Get experiment results
    pub fn get_results(&self, experiment_id: &str) -> Option<ExperimentResults> {
        let results = self.results.read().unwrap();
        results.get(experiment_id).cloned()
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Vec<Experiment> {
        let experiments = self.experiments.read().unwrap();
        experiments.values().cloned().collect()
    }

    /// Get a specific experiment
    pub fn get_experiment(&self, experiment_id: &str) -> Option<Experiment> {
        let experiments = self.experiments.read().unwrap();
        experiments.get(experiment_id).cloned()
    }

    /// Update experiment status
    pub fn update_experiment_status(
        &self,
        experiment_id: &str,
        status: ExperimentStatus,
    ) -> crate::Result<()> {
        let mut experiments = self.experiments.write().unwrap();
        if let Some(experiment) = experiments.get_mut(experiment_id) {
            experiment.status = status;
            Ok(())
        } else {
            Err(crate::Error::LogChannelClosed) // TODO: Add proper error type
        }
    }

    /// Determine winning variant (statistical significance)
    pub fn determine_winner(&self, experiment_id: &str, confidence: f64) -> Option<String> {
        let results = self.results.read().unwrap();
        let exp_results = results.get(experiment_id)?;

        if exp_results.variant_metrics.len() < 2 {
            return None;
        }

        // Find variant with highest CTR
        let mut best_variant: Option<(&String, &VariantMetrics)> = None;

        for (variant_id, metrics) in &exp_results.variant_metrics {
            if let Some((_, best_metrics)) = best_variant {
                if metrics.ctr > best_metrics.ctr {
                    best_variant = Some((variant_id, metrics));
                }
            } else {
                best_variant = Some((variant_id, metrics));
            }
        }

        let (winner_id, winner_metrics) = best_variant?;

        // Check statistical significance using z-test
        for (variant_id, metrics) in &exp_results.variant_metrics {
            if variant_id == winner_id {
                continue;
            }

            let z_score = calculate_z_score(
                winner_metrics.ctr,
                winner_metrics.impressions,
                metrics.ctr,
                metrics.impressions,
            );

            // z-score > 1.96 = 95% confidence
            // z-score > 2.576 = 99% confidence
            let required_z = if confidence >= 0.99 { 2.576 } else { 1.96 };

            if z_score < required_z {
                // Not statistically significant
                return None;
            }
        }

        Some(winner_id.clone())
    }
}

impl Default for ABTestEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_z_score(
    ctr_a: f64,
    n_a: usize,
    ctr_b: f64,
    n_b: usize,
) -> f64 {
    let p_pooled = ((ctr_a * n_a as f64) + (ctr_b * n_b as f64)) / (n_a + n_b) as f64;
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n_a as f64 + 1.0 / n_b as f64)).sqrt();

    if se == 0.0 {
        return 0.0;
    }

    ((ctr_a - ctr_b) / se).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_logger::{LatencyBucket, SearchType};

    #[test]
    fn test_variant_assignment() {
        let engine = ABTestEngine::new();

        let experiment = Experiment {
            experiment_id: "test_exp".to_string(),
            index_uid: "test_index".to_string(),
            variants: vec![
                Variant {
                    variant_id: "control".to_string(),
                    settings: Settings {
                        ranking_rules: Some(vec!["words".to_string(), "typo".to_string()]),
                        searchable_attributes: None,
                        displayed_attributes: None,
                    },
                    description: "Control variant".to_string(),
                },
                Variant {
                    variant_id: "treatment".to_string(),
                    settings: Settings {
                        ranking_rules: Some(vec!["words".to_string(), "proximity".to_string()]),
                        searchable_attributes: None,
                        displayed_attributes: None,
                    },
                    description: "Treatment variant".to_string(),
                },
            ],
            traffic_split: vec![0.5, 0.5],
            start_time: 1699564800,
            end_time: None,
            status: ExperimentStatus::Running,
        };

        engine.create_experiment(experiment).unwrap();

        // Test consistent assignment
        let user_id = "user123";
        let variant1 = engine.assign_variant("test_exp", user_id);
        let variant2 = engine.assign_variant("test_exp", user_id);

        assert_eq!(variant1, variant2); // Same user should get same variant
        assert!(variant1.is_some());
    }

    #[test]
    fn test_experiment_tracking() {
        let engine = ABTestEngine::new();

        let experiment = Experiment {
            experiment_id: "test_tracking".to_string(),
            index_uid: "test_index".to_string(),
            variants: vec![
                Variant {
                    variant_id: "v1".to_string(),
                    settings: Settings {
                        ranking_rules: None,
                        searchable_attributes: None,
                        displayed_attributes: None,
                    },
                    description: "Variant 1".to_string(),
                },
            ],
            traffic_split: vec![1.0],
            start_time: 1699564800,
            end_time: None,
            status: ExperimentStatus::Running,
        };

        engine.create_experiment(experiment).unwrap();

        // Record some queries
        for i in 0..10 {
            let log = QueryLog {
                query_id: format!("query_{}", i),
                index_uid: "test_index".to_string(),
                query: "test".to_string(),
                filters: None,
                hits_count: 5,
                processing_time_ms: 15,
                timestamp: 1699564800 + i,
                user_id: Some(format!("user_{}", i)),
                search_type: SearchType::Keyword,
                semantic_ratio: None,
                clicked_positions: vec![0],
                total_clicks: if i % 2 == 0 { 1 } else { 0 },
                latency_bucket: LatencyBucket::Normal,
            };

            engine.record_query("test_tracking", "v1", &log);
        }

        let results = engine.get_results("test_tracking").unwrap();
        let metrics = results.variant_metrics.get("v1").unwrap();

        assert_eq!(metrics.impressions, 10);
        assert_eq!(metrics.clicks, 5); // 5 queries with clicks
        assert_eq!(metrics.ctr, 0.5);
    }

    #[test]
    fn test_z_score_calculation() {
        // Test with significant difference
        let z = calculate_z_score(0.3, 1000, 0.2, 1000);
        assert!(z > 1.96); // Should be statistically significant at 95%

        // Test with no difference
        let z = calculate_z_score(0.3, 1000, 0.3, 1000);
        assert!(z < 0.1); // Should not be significant
    }
}
