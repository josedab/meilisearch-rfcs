use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantQuota {
    /// Tenant identifier
    pub tenant_id: String,

    /// Service tier
    pub tier: ServiceTier,

    /// Resource limits
    pub limits: ResourceLimits,

    /// Current usage
    pub usage: ResourceUsage,

    /// Soft vs hard limits
    pub enforcement: EnforcementPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceTier {
    Free,
    Hobby,
    Pro,
    Enterprise,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum documents across all indexes
    pub max_documents: Option<usize>,

    /// Maximum total index size (bytes)
    pub max_index_size: Option<u64>,

    /// Maximum number of indexes
    pub max_indexes: Option<usize>,

    /// API rate limits
    pub rate_limits: RateLimits,

    /// Maximum concurrent searches
    pub max_concurrent_searches: Option<usize>,

    /// Maximum indexing operations per day
    pub max_indexing_ops_per_day: Option<usize>,

    /// CPU quota (milliseconds per second)
    pub cpu_quota_ms: Option<u64>,

    /// Memory limit (bytes)
    pub memory_limit: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    /// Searches per minute
    pub searches_per_minute: Option<u32>,

    /// Document additions per minute
    pub documents_per_minute: Option<u32>,

    /// Settings updates per hour
    pub settings_per_hour: Option<u32>,

    /// Burst allowance (extra capacity for short periods)
    pub burst_capacity: Option<u32>,
}

impl RateLimits {
    pub fn get_limit_for(&self, operation: ApiOperation) -> Option<u32> {
        match operation {
            ApiOperation::Search => self.searches_per_minute,
            ApiOperation::DocumentAdd => self.documents_per_minute,
            ApiOperation::SettingsUpdate => self.settings_per_hour,
            ApiOperation::IndexCreate => self.settings_per_hour,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current document count
    pub document_count: usize,

    /// Current index size (bytes)
    pub index_size: u64,

    /// Number of indexes
    pub index_count: usize,

    /// API calls in current window
    pub api_calls: HashMap<ApiOperation, u32>,

    /// CPU time used (milliseconds)
    pub cpu_time_ms: u64,

    /// Memory currently allocated
    pub memory_allocated: u64,

    /// Last reset time
    pub last_reset: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApiOperation {
    Search,
    DocumentAdd,
    SettingsUpdate,
    IndexCreate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementPolicy {
    /// Block requests immediately when limit reached
    Hard,
    /// Allow burst up to N% over limit, then throttle
    Soft { burst_percent: u32 },
    /// Warn but don't enforce
    Monitor,
}

pub struct QuotaManager {
    quotas: Arc<RwLock<HashMap<String, TenantQuota>>>,
    storage: Arc<dyn QuotaStorage>,
}

pub trait QuotaStorage: Send + Sync {
    fn update_usage(&self, tenant_id: &str, usage: &ResourceUsage) -> Result<(), Error>;
    fn load_quota(&self, tenant_id: &str) -> Result<Option<TenantQuota>, Error>;
    fn save_quota(&self, quota: &TenantQuota) -> Result<(), Error>;
}

impl QuotaManager {
    pub fn new(storage: Arc<dyn QuotaStorage>) -> Self {
        Self {
            quotas: Arc::new(RwLock::new(HashMap::new())),
            storage,
        }
    }

    pub fn check_quota(
        &self,
        tenant_id: &str,
        operation: ApiOperation,
        estimated_cost: ResourceCost,
    ) -> Result<QuotaCheckResult, Error> {
        let mut quotas = self.quotas.write().unwrap();
        let quota = quotas.get_mut(tenant_id)
            .ok_or(Error::TenantNotFound)?;

        // Check each limit
        let mut violations = Vec::new();

        // 1. Check document count
        if let Some(max_docs) = quota.limits.max_documents {
            if quota.usage.document_count + estimated_cost.documents > max_docs {
                violations.push(QuotaViolation::DocumentLimit {
                    current: quota.usage.document_count,
                    requested: estimated_cost.documents,
                    limit: max_docs,
                });
            }
        }

        // 2. Check rate limits
        if let Some(limit) = quota.limits.rate_limits.get_limit_for(operation) {
            let current_rate = quota.usage.api_calls.get(&operation).copied().unwrap_or(0);

            if current_rate >= limit {
                match quota.enforcement {
                    EnforcementPolicy::Hard => {
                        return Ok(QuotaCheckResult::Denied {
                            reason: QuotaViolation::RateLimit {
                                operation,
                                current: current_rate,
                                limit,
                            },
                        });
                    }
                    EnforcementPolicy::Soft { burst_percent } => {
                        let burst_limit = limit + (limit * burst_percent / 100);
                        if current_rate >= burst_limit {
                            return Ok(QuotaCheckResult::Denied {
                                reason: QuotaViolation::RateLimit {
                                    operation,
                                    current: current_rate,
                                    limit: burst_limit,
                                },
                            });
                        }
                        violations.push(QuotaViolation::SoftLimitExceeded {
                            operation,
                            current: current_rate,
                            soft_limit: limit,
                        });
                    }
                    EnforcementPolicy::Monitor => {
                        // Just log, don't block
                        tracing::warn!(
                            tenant_id = tenant_id,
                            operation = ?operation,
                            current = current_rate,
                            limit = limit,
                            "Tenant exceeded quota (monitor mode)"
                        );
                    }
                }
            }
        }

        // If no violations or only soft limit warnings
        if violations.iter().all(|v| matches!(v, QuotaViolation::SoftLimitExceeded { .. })) {
            Ok(QuotaCheckResult::Allowed {
                warnings: violations,
            })
        } else {
            Ok(QuotaCheckResult::Denied {
                reason: violations.into_iter().next().unwrap(),
            })
        }
    }

    pub fn record_usage(
        &self,
        tenant_id: &str,
        operation: ApiOperation,
        actual_cost: ResourceCost,
    ) -> Result<(), Error> {
        let mut quotas = self.quotas.write().unwrap();
        let quota = quotas.get_mut(tenant_id)
            .ok_or(Error::TenantNotFound)?;

        // Update usage counters
        quota.usage.document_count += actual_cost.documents;
        quota.usage.cpu_time_ms += actual_cost.cpu_time_ms;
        quota.usage.memory_allocated = quota.usage.memory_allocated.max(actual_cost.memory_bytes);

        *quota.usage.api_calls.entry(operation).or_insert(0) += 1;

        // Persist to storage
        self.storage.update_usage(tenant_id, &quota.usage)?;

        Ok(())
    }

    pub fn add_tenant(&self, quota: TenantQuota) -> Result<(), Error> {
        let mut quotas = self.quotas.write().unwrap();
        let tenant_id = quota.tenant_id.clone();
        quotas.insert(tenant_id.clone(), quota.clone());
        self.storage.save_quota(&quota)?;
        Ok(())
    }

    pub fn get_tenant(&self, tenant_id: &str) -> Result<TenantQuota, Error> {
        let quotas = self.quotas.read().unwrap();
        quotas.get(tenant_id).cloned().ok_or(Error::TenantNotFound)
    }
}

#[derive(Debug)]
pub enum QuotaCheckResult {
    Allowed {
        warnings: Vec<QuotaViolation>,
    },
    Denied {
        reason: QuotaViolation,
    },
}

#[derive(Debug, Serialize)]
pub enum QuotaViolation {
    DocumentLimit {
        current: usize,
        requested: usize,
        limit: usize,
    },
    RateLimit {
        operation: ApiOperation,
        current: u32,
        limit: u32,
    },
    SoftLimitExceeded {
        operation: ApiOperation,
        current: u32,
        soft_limit: u32,
    },
    MemoryLimit {
        current: u64,
        requested: u64,
        limit: u64,
    },
}

#[derive(Debug, Default)]
pub struct ResourceCost {
    pub documents: usize,
    pub cpu_time_ms: u64,
    pub memory_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Tenant not found")]
    TenantNotFound,

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Queue is full")]
    QueueFull,
}
