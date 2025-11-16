use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::tenant_quotas::{ApiOperation, ServiceTier, Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl From<ServiceTier> for Priority {
    fn from(tier: ServiceTier) -> Self {
        match tier {
            ServiceTier::Free => Priority::Low,
            ServiceTier::Hobby => Priority::Normal,
            ServiceTier::Pro => Priority::High,
            ServiceTier::Enterprise => Priority::Critical,
            ServiceTier::Custom(_) => Priority::Normal,
        }
    }
}

#[derive(Debug)]
pub struct PrioritizedRequest {
    pub request_id: String,
    pub tenant_id: String,
    pub priority: Priority,
    pub enqueued_at: Instant,
    pub operation: ApiOperation,
    pub payload: RequestPayload,
}

#[derive(Debug, Clone)]
pub enum RequestPayload {
    Search(String),
    DocumentAdd(Vec<u8>),
    SettingsUpdate(String),
    IndexCreate(String),
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first (BinaryHeap is a max-heap)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // FIFO within same priority (earlier requests first)
                other.enqueued_at.cmp(&self.enqueued_at)
            }
            ordering => ordering,
        }
    }
}

pub struct PriorityQueueManager {
    queue: Arc<Mutex<BinaryHeap<PrioritizedRequest>>>,
    max_queue_size: usize,
}

impl PriorityQueueManager {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            max_queue_size,
        }
    }

    pub fn enqueue(&self, request: PrioritizedRequest) -> Result<(), Error> {
        let mut queue = self.queue.lock().unwrap();

        if queue.len() >= self.max_queue_size {
            // Queue full - reject lowest priority requests
            if let Some(lowest) = queue.peek() {
                if request.priority <= lowest.priority {
                    return Err(Error::QueueFull);
                }
                queue.pop(); // Remove lowest priority
            }
        }

        queue.push(request);
        Ok(())
    }

    pub fn dequeue(&self) -> Option<PrioritizedRequest> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop()
    }

    pub fn len(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.len()
    }

    pub fn is_empty(&self) -> bool {
        let queue = self.queue.lock().unwrap();
        queue.is_empty()
    }
}
