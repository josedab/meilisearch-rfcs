use serde::{Deserialize, Serialize};

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
    pub filter: Option<String>,
    pub facets: Option<Vec<String>>,
    pub attributes_to_retrieve: Option<Vec<String>>,
    pub attributes_to_highlight: Option<Vec<String>>,
    pub show_matches_position: Option<bool>,
}

/// Search result from a single shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSearchResult {
    pub documents_ids: Vec<String>,
    pub document_scores: Vec<Vec<ScoreDetails>>,
    pub documents: Vec<serde_json::Value>,
    pub facet_distribution: Option<std::collections::HashMap<String, std::collections::HashMap<String, u64>>>,
    pub processing_time_ms: u64,
    pub query: String,
}

/// Merged search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<serde_json::Value>,
    pub query: String,
    pub processing_time_ms: u64,
    pub limit: usize,
    pub offset: usize,
    pub estimated_total_hits: usize,
    pub facet_distribution: Option<std::collections::HashMap<String, std::collections::HashMap<String, u64>>>,
}

/// Score details for ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDetails {
    pub order: usize,
    pub value: f64,
}

impl ScoreDetails {
    /// Calculate global score from score details
    pub fn global_score<'a>(scores: impl Iterator<Item = &'a ScoreDetails>) -> f64 {
        scores.map(|s| s.value).sum()
    }
}

/// Task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task_uid: String,
    pub status: TaskStatus,
    pub shard_tasks: Vec<ShardTaskInfo>,
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Enqueued,
    Processing,
    Succeeded,
    Failed,
}

/// Shard-specific task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardTaskInfo {
    pub shard_id: usize,
    pub node_id: u64,
    pub task_uid: String,
    pub status: TaskStatus,
}
