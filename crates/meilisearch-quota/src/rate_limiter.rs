use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::tenant_quotas::{ApiOperation, TenantQuota, Error};

/// Token bucket rate limiter
pub struct TokenBucketRateLimiter {
    /// Maximum tokens (burst capacity)
    capacity: u32,
    /// Tokens added per second
    refill_rate: f64,
    /// Current token count
    tokens: f64,
    /// Last refill time
    last_refill: Instant,
}

impl TokenBucketRateLimiter {
    pub fn new(rate_per_minute: u32, burst_capacity: Option<u32>) -> Self {
        let capacity = burst_capacity.unwrap_or(rate_per_minute);
        let refill_rate = rate_per_minute as f64 / 60.0; // tokens per second

        Self {
            capacity,
            refill_rate,
            tokens: capacity as f64,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume N tokens
    pub fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();

        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity as f64);
        self.last_refill = now;
    }

    /// Time until N tokens available
    pub fn time_until_available(&self, tokens: u32) -> Option<Duration> {
        if self.tokens >= tokens as f64 {
            return None;
        }

        let tokens_needed = tokens as f64 - self.tokens;
        let seconds = tokens_needed / self.refill_rate;

        Some(Duration::from_secs_f64(seconds))
    }

    /// Get remaining tokens
    pub fn remaining_tokens(&self) -> u32 {
        self.tokens as u32
    }
}

/// Per-tenant rate limiter manager
pub struct TenantRateLimiter {
    limiters: Arc<RwLock<HashMap<String, HashMap<ApiOperation, TokenBucketRateLimiter>>>>,
}

impl TenantRateLimiter {
    pub fn new() -> Self {
        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn check_rate_limit(
        &self,
        tenant_id: &str,
        operation: ApiOperation,
        quota: &TenantQuota,
    ) -> Result<RateLimitResult, Error> {
        let mut limiters = self.limiters.write().unwrap();
        let tenant_limiters = limiters.entry(tenant_id.to_string())
            .or_insert_with(HashMap::new);

        let limiter = tenant_limiters.entry(operation)
            .or_insert_with(|| {
                let limit = quota.limits.rate_limits.get_limit_for(operation).unwrap_or(1000);
                let burst = quota.limits.rate_limits.burst_capacity.unwrap_or(limit * 2);
                TokenBucketRateLimiter::new(limit, Some(burst))
            });

        if limiter.try_consume(1) {
            Ok(RateLimitResult::Allowed {
                remaining: limiter.remaining_tokens(),
            })
        } else {
            let retry_after = limiter.time_until_available(1)
                .unwrap_or(Duration::from_secs(60));

            Ok(RateLimitResult::Limited {
                retry_after,
            })
        }
    }
}

impl Default for TenantRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum RateLimitResult {
    Allowed { remaining: u32 },
    Limited { retry_after: Duration },
}
