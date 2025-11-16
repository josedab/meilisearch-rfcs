use meilisearch_quota::{
    billing::{CostCalculator, PricingModel},
    priority_queue::{Priority, PriorityQueueManager, PrioritizedRequest, RequestPayload},
    rate_limiter::TokenBucketRateLimiter,
    tenant_quotas::{
        ApiOperation, EnforcementPolicy, RateLimits, ResourceLimits,
        ResourceUsage, ServiceTier, TenantQuota,
    },
    usage_tracker::TenantMetrics,
};
use std::time::Instant;

#[test]
fn test_token_bucket_rate_limiter() {
    // Create a rate limiter: 60 requests per minute, burst of 100
    let mut limiter = TokenBucketRateLimiter::new(60, Some(100));

    // Should allow initial requests up to burst capacity
    for _ in 0..100 {
        assert!(limiter.try_consume(1), "Should allow requests within burst capacity");
    }

    // Should deny requests beyond capacity
    assert!(!limiter.try_consume(1), "Should deny requests beyond capacity");

    // Check time until available
    let wait_time = limiter.time_until_available(1);
    assert!(wait_time.is_some(), "Should provide wait time");
}

#[test]
fn test_priority_queue_ordering() {
    let queue = PriorityQueueManager::new(100);

    // Add requests with different priorities
    let low_priority_req = PrioritizedRequest {
        request_id: "low-1".to_string(),
        tenant_id: "tenant-1".to_string(),
        priority: Priority::Low,
        enqueued_at: Instant::now(),
        operation: ApiOperation::Search,
        payload: RequestPayload::Search("query".to_string()),
    };

    let high_priority_req = PrioritizedRequest {
        request_id: "high-1".to_string(),
        tenant_id: "tenant-2".to_string(),
        priority: Priority::High,
        enqueued_at: Instant::now(),
        operation: ApiOperation::Search,
        payload: RequestPayload::Search("query".to_string()),
    };

    let critical_priority_req = PrioritizedRequest {
        request_id: "critical-1".to_string(),
        tenant_id: "tenant-3".to_string(),
        priority: Priority::Critical,
        enqueued_at: Instant::now(),
        operation: ApiOperation::Search,
        payload: RequestPayload::Search("query".to_string()),
    };

    // Enqueue in random order
    queue.enqueue(low_priority_req).unwrap();
    queue.enqueue(high_priority_req).unwrap();
    queue.enqueue(critical_priority_req).unwrap();

    // Dequeue should return highest priority first
    let first = queue.dequeue().unwrap();
    assert_eq!(first.priority, Priority::Critical);
    assert_eq!(first.request_id, "critical-1");

    let second = queue.dequeue().unwrap();
    assert_eq!(second.priority, Priority::High);
    assert_eq!(second.request_id, "high-1");

    let third = queue.dequeue().unwrap();
    assert_eq!(third.priority, Priority::Low);
    assert_eq!(third.request_id, "low-1");

    assert!(queue.is_empty());
}

#[test]
fn test_service_tier_to_priority_conversion() {
    assert_eq!(Priority::from(ServiceTier::Free), Priority::Low);
    assert_eq!(Priority::from(ServiceTier::Hobby), Priority::Normal);
    assert_eq!(Priority::from(ServiceTier::Pro), Priority::High);
    assert_eq!(Priority::from(ServiceTier::Enterprise), Priority::Critical);
    assert_eq!(
        Priority::from(ServiceTier::Custom("custom".to_string())),
        Priority::Normal
    );
}

#[test]
fn test_cost_calculation() {
    let calculator = CostCalculator::new();

    let metrics = TenantMetrics {
        searches_performed: 10000,       // 10K searches
        documents_indexed: 5000,         // 5K documents
        disk_space_bytes: 1073741824,    // 1 GB
        cpu_time_ms: 3600000,            // 1 CPU-hour
        network_bytes: 1073741824,       // 1 GB
        ..Default::default()
    };

    let report = calculator.calculate_cost("tenant-123", &metrics);

    assert_eq!(report.tenant_id, "tenant-123");
    assert_eq!(report.currency, "USD");

    // Expected costs with default pricing:
    // Searches: 10000/1000 * $0.50 = $5.00
    // Indexing: 5000/1000 * $1.00 = $5.00
    // Storage: 1 GB * $0.10 = $0.10
    // CPU: 1 hour * $0.05 = $0.05
    // Network: 1 GB * $0.09 = $0.09
    // Total: $10.24

    assert!((report.breakdown.search_cost - 5.0).abs() < 0.01);
    assert!((report.breakdown.indexing_cost - 5.0).abs() < 0.01);
    assert!((report.breakdown.storage_cost - 0.10).abs() < 0.01);
    assert!((report.breakdown.cpu_cost - 0.05).abs() < 0.01);
    assert!((report.breakdown.network_cost - 0.09).abs() < 0.01);
    assert!((report.total_cost - 10.24).abs() < 0.01);
}

#[test]
fn test_custom_pricing_model() {
    let custom_pricing = PricingModel {
        search_cost_per_k: 1.0,
        indexing_cost_per_k: 2.0,
        storage_cost_per_gb_month: 0.20,
        cpu_cost_per_hour: 0.10,
        network_cost_per_gb: 0.15,
    };

    let calculator = CostCalculator::with_pricing(custom_pricing);

    let metrics = TenantMetrics {
        searches_performed: 1000,
        documents_indexed: 1000,
        disk_space_bytes: 1073741824,
        cpu_time_ms: 3600000,
        network_bytes: 1073741824,
        ..Default::default()
    };

    let report = calculator.calculate_cost("tenant-456", &metrics);

    // With custom pricing:
    // Searches: 1 * $1.00 = $1.00
    // Indexing: 1 * $2.00 = $2.00
    // Storage: 1 * $0.20 = $0.20
    // CPU: 1 * $0.10 = $0.10
    // Network: 1 * $0.15 = $0.15
    // Total: $3.45

    assert!((report.total_cost - 3.45).abs() < 0.01);
}

#[test]
fn test_rate_limits_get_limit_for() {
    let rate_limits = RateLimits {
        searches_per_minute: Some(100),
        documents_per_minute: Some(200),
        settings_per_hour: Some(10),
        burst_capacity: Some(150),
    };

    assert_eq!(rate_limits.get_limit_for(ApiOperation::Search), Some(100));
    assert_eq!(rate_limits.get_limit_for(ApiOperation::DocumentAdd), Some(200));
    assert_eq!(rate_limits.get_limit_for(ApiOperation::SettingsUpdate), Some(10));
    assert_eq!(rate_limits.get_limit_for(ApiOperation::IndexCreate), Some(10));
}

#[test]
fn test_tenant_quota_structure() {
    let quota = TenantQuota {
        tenant_id: "test-tenant".to_string(),
        tier: ServiceTier::Pro,
        limits: ResourceLimits {
            max_documents: Some(1_000_000),
            max_index_size: Some(10_737_418_240), // 10 GB
            max_indexes: Some(10),
            rate_limits: RateLimits {
                searches_per_minute: Some(1000),
                documents_per_minute: Some(10000),
                settings_per_hour: Some(50),
                burst_capacity: Some(2000),
            },
            max_concurrent_searches: Some(50),
            max_indexing_ops_per_day: Some(100_000),
            cpu_quota_ms: Some(500),
            memory_limit: Some(2_147_483_648), // 2 GB
        },
        usage: ResourceUsage::default(),
        enforcement: EnforcementPolicy::Soft { burst_percent: 20 },
    };

    assert_eq!(quota.tenant_id, "test-tenant");
    assert_eq!(quota.limits.max_documents, Some(1_000_000));
    assert!(matches!(quota.tier, ServiceTier::Pro));
    assert!(matches!(quota.enforcement, EnforcementPolicy::Soft { burst_percent: 20 }));
}
