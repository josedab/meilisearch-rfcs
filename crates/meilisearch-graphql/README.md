# Meilisearch GraphQL API

This crate provides a GraphQL API layer for Meilisearch, enabling flexible query composition, better field selection, type-safe client generation, and real-time subscriptions.

## Features

- **Flexible Queries**: Compose complex queries with precise field selection
- **Type Safety**: Automatic client type generation from GraphQL schema
- **Real-time Updates**: WebSocket subscriptions for live data
- **Co-exists with REST**: Works alongside the existing REST API

## API Endpoints

- `POST /graphql` - GraphQL queries and mutations
- `GET /graphql` - GraphQL Playground (development only)
- `WebSocket /graphql/ws` - GraphQL subscriptions

## Usage Examples

### Simple Search Query

```graphql
query SearchProducts {
  index(uid: "products") {
    search(query: "laptop", limit: 10) {
      hits {
        id
        fields
      }
      estimatedTotalHits
      processingTimeMs
    }
  }
}
```

### Dashboard Query (Multiple Indexes)

```graphql
query Dashboard {
  indexes {
    results {
      uid
      stats {
        numberOfDocuments
        isIndexing
      }
    }
    total
  }
  stats {
    databaseSize
    indexes
  }
}
```

### Federated Search

```graphql
query FederatedSearch {
  federatedSearch(
    queries: [
      { indexUid: "products", query: "laptop", weight: 2.0 }
      { indexUid: "articles", query: "laptop", weight: 0.5 }
    ]
    limit: 20
  ) {
    hits {
      indexUid
      document {
        id
        fields
      }
      rankingScore
    }
    processingTimeMs
  }
}
```

### Create Index Mutation

```graphql
mutation CreateIndex {
  createIndex(uid: "movies", primaryKey: "id") {
    taskUid
    indexUid
    enqueuedAt
  }
}
```

### Add Documents Mutation

```graphql
mutation AddDocuments {
  addDocuments(
    indexUid: "products"
    documents: [
      {
        "id": "1"
        "title": "Laptop"
        "price": 999
      }
    ]
  ) {
    taskUid
    indexUid
    enqueuedAt
  }
}
```

### Real-time Search Subscription

```graphql
subscription SearchUpdates {
  searchStream(indexUid: "products", query: "laptop", limit: 10) {
    hits {
      id
      fields
    }
    estimatedTotalHits
  }
}
```

### Task Updates Subscription

```graphql
subscription TaskUpdates {
  taskUpdates(taskUid: 42) {
    uid
    status
    type
    indexUid
    enqueuedAt
    startedAt
    finishedAt
  }
}
```

## Integration

The GraphQL API is automatically integrated into Meilisearch when the `meilisearch-graphql` crate is included. The endpoints are available at:

- Development: `http://localhost:7700/graphql`
- Production: Configure via your deployment settings

## Schema

The complete GraphQL schema is defined in `schema.graphql`. Key types include:

- **Query**: Root query type for reading data
- **Mutation**: Root mutation type for modifying data
- **Subscription**: Root subscription type for real-time updates
- **Index**: Index metadata and operations
- **SearchResult**: Search results with hits and metadata
- **Task**: Asynchronous task status and details

## Development

### Running Tests

```bash
cargo test -p meilisearch-graphql
```

### Building

```bash
cargo build -p meilisearch-graphql
```

## Performance

GraphQL queries have a small overhead compared to REST endpoints (~15% in benchmarks) due to query parsing and validation. However, the ability to fetch exactly what you need in a single request often results in better overall performance for complex use cases.

## Authentication

GraphQL endpoints respect the same authentication mechanisms as REST endpoints. Include your API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

## Error Handling

GraphQL errors follow the GraphQL specification and include:
- Error message
- Error code
- Error type
- Location in query (if applicable)

Example error response:

```json
{
  "errors": [
    {
      "message": "Index not found: unknown_index",
      "extensions": {
        "code": "index_not_found"
      }
    }
  ]
}
```

## Client Libraries

You can use any GraphQL client library:

- **JavaScript/TypeScript**: Apollo Client, urql, graphql-request
- **Python**: gql, sgqlc
- **Rust**: cynic, graphql-client
- **Go**: gqlgen

Example with TypeScript and graphql-request:

```typescript
import { request, gql } from 'graphql-request'

const query = gql`
  query SearchProducts {
    index(uid: "products") {
      search(query: "laptop", limit: 10) {
        hits {
          id
          fields
        }
      }
    }
  }
`

const data = await request('http://localhost:7700/graphql', query)
console.log(data)
```

## License

Same as Meilisearch (MIT)
