# Meilisearch Analytics

This crate implements search analytics and relevancy intelligence features for Meilisearch as described in RFC 006.

## Features

### Query Logging
- Tracks all search queries with metadata (index, query text, filters, results count, processing time)
- Stores logs in LMDB for efficient querying
- Time-series aggregation with 5-minute buckets
- Configurable retention policies

### Click-Through Rate (CTR) Tracking
- Records user interactions with search results
- Computes CTR and Mean Reciprocal Rank (MRR) metrics
- Position-based click analysis
- In-memory tracking with periodic persistence

### A/B Testing Framework
- Create experiments with multiple variants
- Consistent hash-based variant assignment
- Statistical significance testing (z-test)
- Real-time metrics collection per variant

### Relevancy Learning
- Automatic detection of zero-result queries
- Low-CTR query identification
- Semantic search opportunity detection
- Performance optimization suggestions

## Usage

### Query Logger

```rust
use meilisearch_analytics::{QueryLogger, QueryStorage, QueryLog, SearchType, LatencyBucket};
use std::sync::Arc;

// Create storage
let storage = Arc::new(QueryStorage::new(path)?);

// Create logger
let (logger, log_rx) = QueryLogger::new(storage.clone());

// Start background persistence task
tokio::spawn(QueryLogger::persist_logs(log_rx, storage.clone()));

// Log a query
let log = QueryLog {
    query_id: "unique_id".to_string(),
    index_uid: "products".to_string(),
    query: "laptop".to_string(),
    filters: None,
    hits_count: 42,
    processing_time_ms: 15,
    timestamp: OffsetDateTime::now_utc().unix_timestamp(),
    user_id: Some("user123".to_string()),
    search_type: SearchType::Hybrid,
    semantic_ratio: Some(0.5),
    clicked_positions: vec![],
    total_clicks: 0,
    latency_bucket: LatencyBucket::from_millis(15),
};

logger.log_query(log)?;
```

### CTR Tracker

```rust
use meilisearch_analytics::CTRTracker;

let tracker = CTRTracker::new(storage);

// Record impression
tracker.record_impression("query_id", 10);

// Record click
tracker.record_click("query_id", 2); // User clicked position 2

// Get metrics
let ctr = tracker.compute_ctr("query_id").unwrap();
let mrr = tracker.compute_mrr("query_id").unwrap();
let top_clicks = tracker.top_clicked_positions("query_id", 5);
```

### A/B Testing

```rust
use meilisearch_analytics::{ABTestEngine, Experiment, Variant, ExperimentStatus};

let engine = ABTestEngine::new();

// Create experiment
let experiment = Experiment {
    experiment_id: "ranking_test_1".to_string(),
    index_uid: "products".to_string(),
    variants: vec![
        Variant {
            variant_id: "control".to_string(),
            settings: Settings { /* ... */ },
            description: "Current ranking".to_string(),
        },
        Variant {
            variant_id: "treatment".to_string(),
            settings: Settings { /* ... */ },
            description: "New ranking".to_string(),
        },
    ],
    traffic_split: vec![0.5, 0.5],
    start_time: OffsetDateTime::now_utc().unix_timestamp(),
    end_time: None,
    status: ExperimentStatus::Running,
};

engine.create_experiment(experiment)?;

// Assign user to variant
let variant = engine.assign_variant("ranking_test_1", "user123");

// Record query results
engine.record_query("ranking_test_1", &variant, &query_log);

// Get results
let results = engine.get_results("ranking_test_1");

// Determine winner
let winner = engine.determine_winner("ranking_test_1", 0.95); // 95% confidence
```

### Relevancy Learner

```rust
use meilisearch_analytics::RelevancyLearner;

let learner = RelevancyLearner::new(ctr_tracker, storage);

// Get suggestions
let suggestions = learner.suggest_improvements("products", 7).await?;

for suggestion in suggestions {
    println!("Type: {:?}", suggestion.suggestion_type);
    println!("Description: {}", suggestion.description);
    println!("Affected queries: {:?}", suggestion.affected_queries);
    println!("Impact: {:?}", suggestion.expected_impact);
}

// Get query stats
let stats = learner.get_query_stats("products", "laptop", 7)?;
println!("CTR: {}", stats.ctr);
println!("Avg processing time: {}ms", stats.avg_processing_time_ms);
```

## Storage

Analytics data is stored in LMDB with two databases:

1. **Query Log Database**: Stores individual query logs with composite keys (timestamp:query_id)
2. **Metrics Database**: Time-series aggregations in 5-minute buckets

Default retention: 30 days (configurable)

## Performance

- **Query Logging**: Async, batched writes (100 queries or 5s intervals)
- **CTR Tracking**: In-memory with periodic cleanup
- **Storage Overhead**: ~500 bytes per query, ~15GB/month for 1M queries/day
- **Query Latency**: +0.8ms average overhead when enabled

## Privacy

- Optional user ID hashing
- Configurable PII filtering
- GDPR-compliant retention policies
- Opt-out mechanisms

## Testing

Run tests with:

```bash
cargo test -p meilisearch-analytics
```

## License

MIT
