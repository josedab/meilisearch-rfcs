use actix_web::{web, HttpResponse};
use index_scheduler::IndexScheduler;
use meilisearch_types::error::ResponseError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use time::OffsetDateTime;

use crate::extractors::authentication::policies::*;
use crate::extractors::authentication::GuardedData;

/// Get analytics overview for an index
pub async fn get_analytics_overview(
    index_scheduler: GuardedData<ActionPolicy<{ actions::INDEXES_GET }>, Data<IndexScheduler>>,
    index_uid: web::Path<String>,
    params: web::Query<AnalyticsParams>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // Calculate time range
    let end_time = params.end_time.unwrap_or_else(|| OffsetDateTime::now_utc().unix_timestamp());
    let start_time = params.start_time.unwrap_or_else(|| {
        end_time - (7 * 24 * 60 * 60) // Default to 7 days
    });

    // TODO: Fetch actual analytics data from analytics engine
    // For now, return mock data structure
    let overview = AnalyticsOverview {
        total_queries: 0,
        unique_queries: 0,
        avg_processing_time_ms: 0.0,
        p50_processing_time_ms: 0.0,
        p95_processing_time_ms: 0.0,
        p99_processing_time_ms: 0.0,
        total_clicks: 0,
        overall_ctr: 0.0,
        zero_results_rate: 0.0,
        top_queries: vec![],
        search_type_distribution: HashMap::new(),
        latency_distribution: HashMap::new(),
    };

    Ok(HttpResponse::Ok().json(overview))
}

/// Get detailed query analysis
pub async fn get_query_analysis(
    index_scheduler: GuardedData<ActionPolicy<{ actions::INDEXES_GET }>, Data<IndexScheduler>>,
    path: web::Path<(String, String)>,
) -> Result<HttpResponse, ResponseError> {
    let (index_uid, query) = path.into_inner();

    // TODO: Fetch actual query analysis from analytics engine
    let analysis = QueryAnalysis {
        query: query.clone(),
        total_searches: 0,
        unique_users: 0,
        avg_results: 0.0,
        avg_processing_time_ms: 0.0,
        ctr: 0.0,
        mrr: 0.0,
        top_clicked_documents: vec![],
        zero_results_count: 0,
        recent_searches: vec![],
    };

    Ok(HttpResponse::Ok().json(analysis))
}

/// Track a user click
pub async fn track_click(
    index_scheduler: GuardedData<ActionPolicy<{ actions::INDEXES_GET }>, Data<IndexScheduler>>,
    index_uid: web::Path<String>,
    click_event: web::Json<ClickEvent>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // TODO: Record click event in analytics engine

    Ok(HttpResponse::Accepted().json(serde_json::json!({
        "message": "Click event recorded"
    })))
}

/// Create a new A/B test experiment
pub async fn create_experiment(
    index_scheduler: GuardedData<ActionPolicy<{ actions::SETTINGS_UPDATE }>, Data<IndexScheduler>>,
    index_uid: web::Path<String>,
    experiment: web::Json<CreateExperimentRequest>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // TODO: Create experiment in A/B testing engine

    Ok(HttpResponse::Created().json(serde_json::json!({
        "experimentId": experiment.experiment_id,
        "status": "draft"
    })))
}

/// Get experiment results
pub async fn get_experiment_results(
    index_scheduler: GuardedData<ActionPolicy<{ actions::INDEXES_GET }>, Data<IndexScheduler>>,
    path: web::Path<(String, String)>,
) -> Result<HttpResponse, ResponseError> {
    let (index_uid, experiment_id) = path.into_inner();

    // TODO: Fetch experiment results from A/B testing engine
    let results = ExperimentResultsResponse {
        experiment_id: experiment_id.clone(),
        status: "running".to_string(),
        variants: HashMap::new(),
        winner: None,
        confidence: None,
    };

    Ok(HttpResponse::Ok().json(results))
}

/// Get relevancy suggestions
pub async fn get_relevancy_suggestions(
    index_scheduler: GuardedData<ActionPolicy<{ actions::INDEXES_GET }>, Data<IndexScheduler>>,
    index_uid: web::Path<String>,
    params: web::Query<SuggestionsParams>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // TODO: Fetch suggestions from relevancy learner
    let suggestions = SuggestionsResponse {
        suggestions: vec![],
    };

    Ok(HttpResponse::Ok().json(suggestions))
}

// Request/Response types

#[derive(Debug, Deserialize)]
pub struct AnalyticsParams {
    #[serde(default)]
    pub start_time: Option<i64>,
    #[serde(default)]
    pub end_time: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct AnalyticsOverview {
    pub total_queries: usize,
    pub unique_queries: usize,
    pub avg_processing_time_ms: f64,
    pub p50_processing_time_ms: f64,
    pub p95_processing_time_ms: f64,
    pub p99_processing_time_ms: f64,
    pub total_clicks: usize,
    pub overall_ctr: f64,
    pub zero_results_rate: f64,
    pub top_queries: Vec<PopularQuery>,
    pub search_type_distribution: HashMap<String, usize>,
    pub latency_distribution: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct PopularQuery {
    pub query: String,
    pub count: usize,
    pub avg_hits: f64,
    pub ctr: f64,
}

#[derive(Debug, Serialize)]
pub struct QueryAnalysis {
    pub query: String,
    pub total_searches: usize,
    pub unique_users: usize,
    pub avg_results: f64,
    pub avg_processing_time_ms: f64,
    pub ctr: f64,
    pub mrr: f64,
    pub top_clicked_documents: Vec<ClickedDocument>,
    pub zero_results_count: usize,
    pub recent_searches: Vec<RecentSearch>,
}

#[derive(Debug, Serialize)]
pub struct ClickedDocument {
    pub document_id: String,
    pub title: String,
    pub clicks: usize,
    pub avg_position: f64,
}

#[derive(Debug, Serialize)]
pub struct RecentSearch {
    pub timestamp: i64,
    pub results_count: usize,
    pub processing_time_ms: u64,
    pub clicked: bool,
}

#[derive(Debug, Deserialize)]
pub struct ClickEvent {
    pub query_id: String,
    pub document_id: String,
    pub position: usize,
    pub timestamp: i64,
}

#[derive(Debug, Deserialize)]
pub struct CreateExperimentRequest {
    pub experiment_id: String,
    pub variants: Vec<ExperimentVariant>,
    pub traffic_split: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ExperimentVariant {
    pub variant_id: String,
    pub settings: serde_json::Value,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct ExperimentResultsResponse {
    pub experiment_id: String,
    pub status: String,
    pub variants: HashMap<String, VariantMetricsResponse>,
    pub winner: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct VariantMetricsResponse {
    pub impressions: usize,
    pub clicks: usize,
    pub ctr: f64,
    pub mrr: f64,
    pub avg_processing_time_ms: f64,
}

#[derive(Debug, Deserialize)]
pub struct SuggestionsParams {
    #[serde(default = "default_time_range")]
    pub time_range_days: i64,
}

fn default_time_range() -> i64 {
    7
}

#[derive(Debug, Serialize)]
pub struct SuggestionsResponse {
    pub suggestions: Vec<RelevancySuggestionResponse>,
}

#[derive(Debug, Serialize)]
pub struct RelevancySuggestionResponse {
    pub suggestion_type: String,
    pub description: String,
    pub affected_queries: Vec<String>,
    pub expected_impact: String,
}

// Route configuration
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/indexes/{index_uid}/analytics")
            .route("/overview", web::get().to(get_analytics_overview))
            .route("/queries/{query}", web::get().to(get_query_analysis))
            .route("/clicks", web::post().to(track_click))
            .route("/suggestions", web::get().to(get_relevancy_suggestions))
    )
    .service(
        web::scope("/indexes/{index_uid}/experiments")
            .route("", web::post().to(create_experiment))
            .route("/{experiment_id}/results", web::get().to(get_experiment_results))
    );
}
