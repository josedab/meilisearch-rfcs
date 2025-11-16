use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::usage_tracker::TenantMetrics;

pub struct CostCalculator {
    pricing: Arc<RwLock<PricingModel>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PricingModel {
    /// Cost per 1000 searches
    pub search_cost_per_k: f64,

    /// Cost per 1000 documents indexed
    pub indexing_cost_per_k: f64,

    /// Cost per GB-month storage
    pub storage_cost_per_gb_month: f64,

    /// Cost per CPU-hour
    pub cpu_cost_per_hour: f64,

    /// Cost per GB network egress
    pub network_cost_per_gb: f64,
}

impl Default for PricingModel {
    fn default() -> Self {
        Self {
            search_cost_per_k: 0.50,        // $0.50 per 1K searches
            indexing_cost_per_k: 1.00,      // $1.00 per 1K documents
            storage_cost_per_gb_month: 0.10, // $0.10 per GB-month
            cpu_cost_per_hour: 0.05,        // $0.05 per CPU-hour
            network_cost_per_gb: 0.09,      // $0.09 per GB egress
        }
    }
}

impl CostCalculator {
    pub fn new() -> Self {
        Self {
            pricing: Arc::new(RwLock::new(PricingModel::default())),
        }
    }

    pub fn with_pricing(pricing: PricingModel) -> Self {
        Self {
            pricing: Arc::new(RwLock::new(pricing)),
        }
    }

    pub fn calculate_cost(&self, tenant_id: &str, metrics: &TenantMetrics) -> BillingReport {
        let pricing = self.pricing.read().unwrap();

        let search_cost = (metrics.searches_performed as f64 / 1000.0) * pricing.search_cost_per_k;
        let indexing_cost = (metrics.documents_indexed as f64 / 1000.0) * pricing.indexing_cost_per_k;

        let storage_gb_months = metrics.disk_space_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let storage_cost = storage_gb_months * pricing.storage_cost_per_gb_month;

        let cpu_hours = metrics.cpu_time_ms as f64 / (1000.0 * 3600.0);
        let cpu_cost = cpu_hours * pricing.cpu_cost_per_hour;

        let network_gb = metrics.network_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let network_cost = network_gb * pricing.network_cost_per_gb;

        let total_cost = search_cost + indexing_cost + storage_cost + cpu_cost + network_cost;

        BillingReport {
            tenant_id: tenant_id.to_string(),
            period_start: metrics.period_start,
            period_end: metrics.period_end,
            breakdown: CostBreakdown {
                search_cost,
                indexing_cost,
                storage_cost,
                cpu_cost,
                network_cost,
            },
            total_cost,
            currency: "USD".to_string(),
        }
    }

    pub fn update_pricing(&self, pricing: PricingModel) {
        let mut current_pricing = self.pricing.write().unwrap();
        *current_pricing = pricing;
    }

    pub fn get_pricing(&self) -> PricingModel {
        self.pricing.read().unwrap().clone()
    }
}

impl Default for CostCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize)]
pub struct BillingReport {
    pub tenant_id: String,
    pub period_start: i64,
    pub period_end: i64,
    pub breakdown: CostBreakdown,
    pub total_cost: f64,
    pub currency: String,
}

#[derive(Debug, Serialize)]
pub struct CostBreakdown {
    pub search_cost: f64,
    pub indexing_cost: f64,
    pub storage_cost: f64,
    pub cpu_cost: f64,
    pub network_cost: f64,
}
