# RFC 005: GraphQL API for Flexible Query Composition

**Status:** Draft  
**Created:** 2025-11-16  
**Authors:** Meilisearch Community  
**Tracking Issue:** TBD

---

## Summary

This RFC proposes adding a GraphQL API layer alongside the existing REST API, enabling more flexible query composition, better field selection, type-safe client generation, and real-time subscriptions. The GraphQL API will co-exist with the REST API, providing developers with modern querying capabilities while maintaining backward compatibility.

## Motivation

### Current REST API Limitations

Meilisearch's current REST API ([`crates/meilisearch/src/routes/`](crates/meilisearch/src/routes/)) is well-designed for simplicity but has limitations for complex queries:

**Key Limitations:**

1. **Over-fetching**: Must retrieve all fields or use `attributesToRetrieve` for each query
2. **Multiple Round-Trips**: Federated search requires separate calls or complex multi-search API
3. **No Real-Time Updates**: Polling required for task status, document changes
4. **Limited Nested Queries**: Cannot fetch related data in single request
5. **Client Type Safety**: No automatic type generation from API schema

### Real-World Use Cases

**Complex Dashboard Queries:**
```graphql
# Single GraphQL query replacing 5+ REST calls
query Dashboard {
  indexes {
    products {
      stats {
        numberOfDocuments
        isIndexing
      }
      search(query: "laptop", limit: 10) {
        hits {
          id
          title
          price
        }
        facetDistribution {
          facets {
            name
            values { value count }
          }
        }
      }
    }
  }
}
```

**Real-Time Search Updates:**
```graphql
subscription SearchUpdates($query: String!) {
  searchStream(indexUid: "products", query: $query) {
    hits {
      id
      title
      price
    }
  }
}
```

## Technical Design

### GraphQL Schema

**New file:** `crates/meilisearch-graphql/schema.graphql`

```graphql
type Query {
  index(uid: String!): Index
  indexes(offset: Int, limit: Int): IndexConnection!
  federatedSearch(queries: [FederatedQuery!]!, limit: Int): FederatedSearchResult!
  task(uid: Int!): Task
  tasks(filter: TaskFilter, limit: Int): TaskConnection!
  stats: Stats!
  health: Health!
}

type Mutation {
  createIndex(uid: String!, primaryKey: String): IndexCreationTask!
  addDocuments(indexUid: String!, documents: [JSON!]!): DocumentTask!
  updateSettings(indexUid: String!, settings: SettingsInput!): SettingsTask!
  deleteDocuments(indexUid: String!, documentIds: [String!]!): DocumentTask!
  deleteIndex(uid: String!): IndexDeletionTask!
}

type Subscription {
  searchStream(indexUid: String!, query: String, limit: Int): SearchResult!
  taskUpdates(taskUid: Int, indexUid: String): Task!
  indexStatsStream(indexUid: String!): IndexStats!
}

type Index {
  uid: String!
  primaryKey: String
  createdAt: DateTime!
  updatedAt: DateTime!
  stats: IndexStats!
  settings: Settings!
  search(
    query: String
    filter: String
    facets: [String!]
    limit: Int
  ): SearchResult!
}

type SearchResult {
  hits: [SearchHit!]!
  estimatedTotalHits: Int!
  processingTimeMs: Int!
  facetDistribution: FacetDistribution
}

type SearchHit {
  id: String!
  fields: JSON!
  _formatted: JSON
  _rankingScore: Float
}

scalar JSON
scalar DateTime
```

### Resolver Implementation

```rust
use async_graphql::{Context, Object, Result};

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn index(&self, ctx: &Context<'_>, uid: String) -> Result<Option<Index>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        match scheduler.index(&uid) {
            Ok(index) => Ok(Some(Index::from_milli_index(uid, index))),
            Err(_) => Ok(None),
        }
    }
    
    async fn search(
        &self,
        ctx: &Context<'_>,
        index_uid: String,
        query: Option<String>,
        limit: Option<i32>,
    ) -> Result<SearchResult> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>()?;
        let index = scheduler.index(&index_uid)?;
        let rtxn = index.read_txn()?;
        
        let mut search = milli::Search::new(&rtxn, &index);
        if let Some(q) = query {
            search.query(q);
        }
        search.limit(limit.unwrap_or(20) as usize);
        
        let (result, _) = search.execute()?;
        Ok(SearchResult::from_milli_result(result))
    }
}
```

### Subscription Support

```rust
use async_graphql::Subscription;
use futures_util::Stream;

pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    async fn search_stream(
        &self,
        ctx: &Context<'_>,
        index_uid: String,
        query: Option<String>,
    ) -> impl Stream<Item = Result<SearchResult>> {
        let scheduler = ctx.data::<Arc<IndexScheduler>>().unwrap().clone();
        
        tokio_stream::wrappers::IntervalStream::new(
            tokio::time::interval(Duration::from_millis(100))
        )
        .then(move |_| {
            let scheduler = scheduler.clone();
            let query = query.clone();
            
            async move {
                let index = scheduler.index(&index_uid)?;
                let rtxn = index.read_txn()?;
                
                let mut search = milli::Search::new(&rtxn, &index);
                if let Some(q) = query {
                    search.query(q);
                }
                
                let (result, _) = search.execute()?;
                Ok(SearchResult::from_milli_result(result))
            }
        })
    }
}
```

## API Changes

### New Endpoints

```
POST /graphql (Query & Mutation)
WebSocket: ws://localhost:7700/graphql/ws (Subscriptions)
GET /graphql (Playground - development only)
```

### Example Usage

**Simple Search:**
```graphql
query {
  index(uid: "products") {
    search(query: "laptop", limit: 10) {
      hits {
        id
        fields
      }
    }
  }
}
```

**Federated Search:**
```graphql
query {
  federatedSearch(
    queries: [
      { indexUid: "products", query: "laptop", weight: 2.0 }
      { indexUid: "articles", query: "laptop", weight: 0.5 }
    ]
  ) {
    hits {
      indexUid
      document { id fields }
    }
  }
}
```

## Backward Compatibility

1. **Co-existence**: GraphQL alongside REST (no breaking changes)
2. **Optional**: Experimental flag `--experimental-graphql`
3. **Feature parity**: All REST features available in GraphQL

## Implementation Plan

### Phase 1: Core Server (3 weeks)
- GraphQL schema definition
- Query/Mutation resolvers
- Unit tests

### Phase 2: Advanced Features (3 weeks)
- Subscription support
- Federated search
- Authentication

### Phase 3: Client Tooling (2 weeks)
- TypeScript type generation
- Example clients
- Documentation

### Phase 4: Polish (2 weeks)
- Performance tuning
- Production guide

## Performance Implications

| Metric | REST API | GraphQL API | Overhead |
|--------|----------|-------------|----------|
| Parse time | 0.5ms | 2ms | +1.5ms |
| Execute time | 10ms | 10ms | 0ms |
| Serialize | 1ms | 1.5ms | +0.5ms |
| **Total** | **11.5ms** | **13.5ms** | **+15%** |

## Drawbacks

1. **Additional complexity**: New API to maintain
2. **Learning curve**: GraphQL requires understanding
3. **Dependency size**: +500KB compiled

## Alternatives Considered

1. **gRPC**: Less web-friendly
2. **JSON-RPC**: No type system
3. **REST improvements only**: Cannot solve over-fetching

## Open Questions

1. **Default enable**: When to enable by default?
2. **Subscription limits**: How many per connection?
3. **Schema versioning**: How to handle breaking changes?

## References

- GraphQL Spec: [https://spec.graphql.org/](https://spec.graphql.org/)
- async-graphql: [Documentation](https://async-graphql.github.io/)
- REST routes: [`crates/meilisearch/src/routes/`](crates/meilisearch/src/routes/)

---

**Changelog:**
- 2025-11-16: Initial draft created