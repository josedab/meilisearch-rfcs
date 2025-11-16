# meilisearch-gateway

Gateway for distributed Meilisearch query routing and result merging.

## Overview

This crate implements the gateway layer for distributed Meilisearch, providing:

- **Query Routing**: Route queries to appropriate shards
- **Result Merging**: Aggregate results from multiple shards
- **Distributed Indexing**: Distribute documents across shards
- **Replication**: Primary-replica synchronization
- **Health Monitoring**: Detect and handle node failures
- **Cluster API**: Management endpoints for cluster operations

## Key Components

### Gateway

Main entry point for distributed operations:
- Search request handling
- Document indexing
- Cluster health monitoring

### QueryRouter

Routes queries to shards and handles failover:
- Parallel shard queries
- Automatic failover to replicas
- Connection pooling

### ResultMerger

Merges results from multiple shards:
- Global score-based sorting
- Facet aggregation
- Pagination support

### DistributedIndexer

Distributes write operations:
- Document sharding by ID
- Parallel shard writes
- Task coordination

### ReplicationManager

Manages replication with configurable consistency:
- Synchronous (strong consistency)
- Quorum (balanced)
- Asynchronous (eventual consistency)

### HealthMonitor

Monitors node health and initiates failover:
- Heartbeat-based monitoring
- Automatic replica promotion
- Failure detection

## Usage

```rust
use meilisearch_gateway::{Gateway, NodeConnectionPool, DistributedConfig};
use meilisearch_coordinator::ClusterCoordinator;

// Load configuration
let config = DistributedConfig::from_file("config.toml")?;

// Create gateway
let pool = Arc::new(NodeConnectionPool::new());
let coordinator = Arc::new(ClusterCoordinator::new(/* ... */));
let gateway = Gateway::new(pool, coordinator);

// Search
let results = gateway.handle_search_request("my-index", query).await?;

// Index documents
let task = gateway.handle_index_documents("my-index", documents).await?;
```

## Configuration

See [config.rs](src/config.rs) for configuration options.

Example `config.toml`:

```toml
[server]
http_addr = "0.0.0.0:7700"
db_path = "/var/lib/meilisearch/data.ms"
mode = "distributed"

[cluster]
node_id = "node-1"
seed_nodes = ["node-1:8700", "node-2:8700", "node-3:8700"]
raft_port = 8700

[sharding]
default_shard_count = 4
default_replication_factor = 2
default_strategy = "hash"
```

## See Also

- [RFC 002: Distributed Architecture](../../rfcs/002_distributed_architecture.md)
- [meilisearch-coordinator](../meilisearch-coordinator) - Cluster coordination
