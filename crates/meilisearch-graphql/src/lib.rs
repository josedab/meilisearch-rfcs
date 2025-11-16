pub mod error;
pub mod mutation;
pub mod query;
pub mod schema;
pub mod subscription;
pub mod types;

#[cfg(test)]
mod tests;

pub use error::{GraphQLError, Result};
pub use schema::{build_schema, GraphQLSchema};

use actix_web::{web, HttpRequest, HttpResponse};
use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_actix_web::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};

/// GraphQL POST endpoint handler
pub async fn graphql_handler(
    schema: web::Data<GraphQLSchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

/// GraphQL Playground UI (for development only)
pub async fn graphql_playground() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

/// GraphQL WebSocket subscription endpoint handler
pub async fn graphql_subscription(
    schema: web::Data<GraphQLSchema>,
    req: HttpRequest,
    payload: web::Payload,
) -> actix_web::Result<HttpResponse> {
    GraphQLSubscription::new(schema.get_ref().clone()).start(&req, payload)
}
