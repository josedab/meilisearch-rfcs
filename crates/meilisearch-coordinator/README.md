# meilisearch-coordinator

Cluster coordinator for distributed Meilisearch using Raft consensus.

## Overview

This crate implements the cluster coordination layer for distributed Meilisearch, providing:

- **Raft Consensus**: Distributed consensus for cluster state management
- **Cluster State Management**: Track nodes, shards, and their assignments
- **Shard Allocation**: Consistent hashing for balanced shard distribution
- **Node Management**: Add/remove nodes, health tracking

## Key Components

### ClusterCoordinator

Main coordinator managing cluster state and operations:
- Node registration and removal
- Index creation with shard allocation
- Failover and replica promotion

### ShardAllocator

Implements consistent hashing for shard allocation:
- Virtual nodes for balanced distribution
- Primary and replica selection
- Automatic rebalancing

### ClusterState

Maintains the current state of the cluster:
- Node information and health
- Shard distribution map
- Version tracking for state changes

## Usage

```rust
use meilisearch_coordinator::{ClusterCoordinator, ShardAllocator, NodeId};

// Create coordinator
let allocator = ShardAllocator::default();
let coordinator = ClusterCoordinator::new(NodeId(1), allocator);

// Create distributed index
let metadata = coordinator.create_index(
    "my-index".to_string(),
    4, // shard count
    2, // replication factor
    ShardingStrategy::Hash,
).await?;
```

## See Also

- [RFC 002: Distributed Architecture](../../rfcs/002_distributed_architecture.md)
- [meilisearch-gateway](../meilisearch-gateway) - Query routing and result merging
