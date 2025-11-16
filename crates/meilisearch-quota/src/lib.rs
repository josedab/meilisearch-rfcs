//! # Meilisearch Quota System
//!
//! This crate provides resource quota and quality of service (QoS) functionality for multi-tenant
//! Meilisearch deployments. It enables:
//!
//! - Per-tenant resource limits (CPU, memory, disk, API rate)
//! - Priority queues for different service tiers
//! - Cost attribution and billing metrics
//! - Tenant isolation guarantees
//!
//! ## Example
//!
//! ```no_run
//! use meilisearch_quota::{
//!     tenant_quotas::{QuotaManager, TenantQuota, ServiceTier, ResourceLimits, RateLimits, ResourceUsage, EnforcementPolicy},
//!     rate_limiter::TenantRateLimiter,
//!     priority_queue::PriorityQueueManager,
//!     usage_tracker::UsageTracker,
//!     billing::CostCalculator,
//! };
//!
//! // Set up quota management
//! // let quota_manager = QuotaManager::new(...);
//! // let rate_limiter = TenantRateLimiter::new();
//! // let priority_queue = PriorityQueueManager::new(1000);
//! // let usage_tracker = UsageTracker::new(...);
//! // let cost_calculator = CostCalculator::new();
//! ```

pub mod tenant_quotas;
pub mod rate_limiter;
pub mod priority_queue;
pub mod usage_tracker;
pub mod billing;

// Re-export commonly used types
pub use tenant_quotas::{
    TenantQuota, ServiceTier, ResourceLimits, RateLimits, ResourceUsage,
    ApiOperation, EnforcementPolicy, QuotaManager, QuotaCheckResult,
    QuotaViolation, ResourceCost, Error,
};

pub use rate_limiter::{TokenBucketRateLimiter, TenantRateLimiter, RateLimitResult};

pub use priority_queue::{Priority, PrioritizedRequest, RequestPayload, PriorityQueueManager};

pub use usage_tracker::{UsageTracker, TenantMetrics, TimePeriod};

pub use billing::{CostCalculator, PricingModel, BillingReport, CostBreakdown};
