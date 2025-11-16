use async_graphql::{EmptyMutation, EmptySubscription, Schema, SchemaBuilder};
use index_scheduler::IndexScheduler;
use std::sync::Arc;

use crate::mutation::MutationRoot;
use crate::query::QueryRoot;
use crate::subscription::SubscriptionRoot;

/// The complete GraphQL schema type
pub type GraphQLSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

/// Build the GraphQL schema with all resolvers
pub fn build_schema(scheduler: Arc<IndexScheduler>) -> GraphQLSchema {
    Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .data(scheduler)
        .finish()
}

/// Build a schema with only queries (no mutations or subscriptions)
/// Useful for read-only contexts
pub fn build_query_only_schema(scheduler: Arc<IndexScheduler>) -> Schema<QueryRoot, EmptyMutation, EmptySubscription> {
    Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(scheduler)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builds() {
        // This test just ensures the schema can be built
        // In a real implementation, you'd need to create a test IndexScheduler
        // For now, this is a placeholder to verify the types compile correctly
    }
}
