use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::tenant_quotas::{ApiOperation, ResourceCost, Error};

pub struct UsageTracker {
    /// Per-tenant metrics
    metrics: Arc<RwLock<HashMap<String, TenantMetrics>>>,
    /// Metrics storage
    storage: Arc<dyn MetricsStorage>,
}

pub trait MetricsStorage: Send + Sync {
    fn get_metrics(&self, tenant_id: &str, period: TimePeriod) -> Result<TenantMetrics, Error>;
    fn save_metrics(&self, tenant_id: &str, metrics: &TenantMetrics) -> Result<(), Error>;
}

#[derive(Debug, Clone, Copy)]
pub enum TimePeriod {
    CurrentHour,
    CurrentDay,
    CurrentMonth,
    Custom { start: i64, end: i64 },
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TenantMetrics {
    /// API call counts by operation
    pub api_calls: HashMap<ApiOperation, u64>,

    /// Total documents indexed
    pub documents_indexed: u64,

    /// Total documents deleted
    pub documents_deleted: u64,

    /// Total searches performed
    pub searches_performed: u64,

    /// CPU time consumed (milliseconds)
    pub cpu_time_ms: u64,

    /// Peak memory usage
    pub peak_memory_bytes: u64,

    /// Disk space used
    pub disk_space_bytes: u64,

    /// Network bandwidth (ingress + egress)
    pub network_bytes: u64,

    /// Time period these metrics cover
    pub period_start: i64,
    pub period_end: i64,
}

impl UsageTracker {
    pub fn new(storage: Arc<dyn MetricsStorage>) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            storage,
        }
    }

    pub fn record_operation(
        &self,
        tenant_id: &str,
        operation: ApiOperation,
        cost: ResourceCost,
    ) -> Result<(), Error> {
        let mut metrics = self.metrics.write().unwrap();
        let tenant_metrics = metrics.entry(tenant_id.to_string())
            .or_default();

        // Update counters
        *tenant_metrics.api_calls.entry(operation).or_insert(0) += 1;

        match operation {
            ApiOperation::DocumentAdd => {
                tenant_metrics.documents_indexed += cost.documents as u64;
            }
            ApiOperation::Search => {
                tenant_metrics.searches_performed += 1;
            }
            _ => {}
        }

        tenant_metrics.cpu_time_ms += cost.cpu_time_ms;
        tenant_metrics.peak_memory_bytes = tenant_metrics.peak_memory_bytes.max(cost.memory_bytes);

        Ok(())
    }

    pub fn get_metrics(&self, tenant_id: &str, period: TimePeriod) -> Result<TenantMetrics, Error> {
        // Retrieve metrics for time period
        self.storage.get_metrics(tenant_id, period)
    }

    pub fn reset_metrics(&self, tenant_id: &str) -> Result<(), Error> {
        let mut metrics = self.metrics.write().unwrap();
        metrics.remove(tenant_id);
        Ok(())
    }

    pub fn save_current_metrics(&self, tenant_id: &str) -> Result<(), Error> {
        let metrics = self.metrics.read().unwrap();
        if let Some(tenant_metrics) = metrics.get(tenant_id) {
            self.storage.save_metrics(tenant_id, tenant_metrics)?;
        }
        Ok(())
    }
}
