use actix_web::{web, HttpRequest, HttpResponse};
use index_scheduler::IndexScheduler;
use meilisearch_graphql::{build_schema, graphql_handler, graphql_playground, graphql_subscription, GraphQLSchema};
use std::sync::Arc;

use crate::extractors::authentication::policies::*;
use crate::extractors::authentication::GuardedData;

/// Configure GraphQL routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/graphql")
            .route(web::post().to(graphql_post_handler))
            .route(web::get().to(graphql_get_handler)),
    )
    .service(web::resource("/graphql/ws").route(web::get().to(graphql_ws_handler)));
}

/// GraphQL POST endpoint (for queries and mutations)
async fn graphql_post_handler(
    index_scheduler: GuardedData<ActionPolicy<{ actions::SEARCH }>, Data<IndexScheduler>>,
    schema: web::Data<GraphQLSchema>,
    req: async_graphql_actix_web::GraphQLRequest,
) -> async_graphql_actix_web::GraphQLResponse {
    graphql_handler(schema, req).await
}

/// GraphQL GET endpoint (for GraphQL Playground in development)
async fn graphql_get_handler(
    _index_scheduler: GuardedData<ActionPolicy<{ actions::SEARCH }>, Data<IndexScheduler>>,
) -> HttpResponse {
    graphql_playground().await
}

/// GraphQL WebSocket endpoint (for subscriptions)
async fn graphql_ws_handler(
    index_scheduler: GuardedData<ActionPolicy<{ actions::SEARCH }>, Data<IndexScheduler>>,
    schema: web::Data<GraphQLSchema>,
    req: HttpRequest,
    payload: web::Payload,
) -> actix_web::Result<HttpResponse> {
    graphql_subscription(schema, req, payload).await
}

/// Initialize GraphQL schema with the index scheduler
pub fn init_graphql_schema(index_scheduler: Arc<IndexScheduler>) -> GraphQLSchema {
    build_schema(index_scheduler)
}
