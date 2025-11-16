use async_graphql::{Enum, InputObject, Object, SimpleObject, Union};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use time::OffsetDateTime;

/// Custom JSON scalar type
pub type JSON = JsonValue;

/// Custom DateTime scalar type
pub type DateTime = OffsetDateTime;

// Index types

#[derive(Clone, Debug)]
pub struct Index {
    pub uid: String,
    pub primary_key: Option<String>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

#[Object]
impl Index {
    async fn uid(&self) -> &str {
        &self.uid
    }

    async fn primary_key(&self) -> Option<&str> {
        self.primary_key.as_deref()
    }

    async fn created_at(&self) -> OffsetDateTime {
        self.created_at
    }

    async fn updated_at(&self) -> OffsetDateTime {
        self.updated_at
    }
}

#[derive(Clone, Debug, SimpleObject)]
pub struct IndexConnection {
    pub results: Vec<Index>,
    pub offset: i32,
    pub limit: i32,
    pub total: i32,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct IndexStats {
    pub number_of_documents: i32,
    pub is_indexing: bool,
    pub field_distribution: JSON,
}

// Settings types

#[derive(Clone, Debug, SimpleObject)]
pub struct Settings {
    pub displayed_attributes: Option<Vec<String>>,
    pub searchable_attributes: Option<Vec<String>>,
    pub filterable_attributes: Option<Vec<String>>,
    pub sortable_attributes: Option<Vec<String>>,
    pub ranking_rules: Option<Vec<String>>,
    pub stop_words: Option<Vec<String>>,
    pub synonyms: Option<JSON>,
    pub distinct_attribute: Option<String>,
    pub typo_tolerance: Option<TypoTolerance>,
    pub faceting: Option<Faceting>,
    pub pagination: Option<Pagination>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct TypoTolerance {
    pub enabled: bool,
    pub min_word_size_for_typos: Option<MinWordSizeForTypos>,
    pub disable_on_words: Option<Vec<String>>,
    pub disable_on_attributes: Option<Vec<String>>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct MinWordSizeForTypos {
    pub one_typo: Option<i32>,
    pub two_typos: Option<i32>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct Faceting {
    pub max_values_per_facet: Option<i32>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct Pagination {
    pub max_total_hits: Option<i32>,
}

#[derive(Clone, Debug, InputObject)]
pub struct SettingsInput {
    pub displayed_attributes: Option<Vec<String>>,
    pub searchable_attributes: Option<Vec<String>>,
    pub filterable_attributes: Option<Vec<String>>,
    pub sortable_attributes: Option<Vec<String>>,
    pub ranking_rules: Option<Vec<String>>,
    pub stop_words: Option<Vec<String>>,
    pub synonyms: Option<JSON>,
    pub distinct_attribute: Option<String>,
    pub typo_tolerance: Option<TypoToleranceInput>,
    pub faceting: Option<FacetingInput>,
    pub pagination: Option<PaginationInput>,
}

#[derive(Clone, Debug, InputObject)]
pub struct TypoToleranceInput {
    pub enabled: Option<bool>,
    pub min_word_size_for_typos: Option<MinWordSizeForTyposInput>,
    pub disable_on_words: Option<Vec<String>>,
    pub disable_on_attributes: Option<Vec<String>>,
}

#[derive(Clone, Debug, InputObject)]
pub struct MinWordSizeForTyposInput {
    pub one_typo: Option<i32>,
    pub two_typos: Option<i32>,
}

#[derive(Clone, Debug, InputObject)]
pub struct FacetingInput {
    pub max_values_per_facet: Option<i32>,
}

#[derive(Clone, Debug, InputObject)]
pub struct PaginationInput {
    pub max_total_hits: Option<i32>,
}

// Search types

#[derive(Clone, Debug, SimpleObject)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub estimated_total_hits: i32,
    pub processing_time_ms: i32,
    pub query: String,
    pub limit: i32,
    pub offset: i32,
    pub facet_distribution: Option<FacetDistribution>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct SearchHit {
    pub id: String,
    pub fields: JSON,
    pub formatted: Option<JSON>,
    pub ranking_score: Option<f64>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct FacetDistribution {
    pub facets: Vec<Facet>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct Facet {
    pub name: String,
    pub values: Vec<FacetValue>,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct FacetValue {
    pub value: String,
    pub count: i32,
}

// Federated search types

#[derive(Clone, Debug, SimpleObject)]
pub struct FederatedSearchResult {
    pub hits: Vec<FederatedHit>,
    pub processing_time_ms: i32,
    pub limit: i32,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct FederatedHit {
    pub index_uid: String,
    pub document: SearchHit,
    pub ranking_score: Option<f64>,
}

#[derive(Clone, Debug, InputObject)]
pub struct FederatedQueryInput {
    pub index_uid: String,
    pub query: Option<String>,
    pub filter: Option<String>,
    pub limit: Option<i32>,
    pub weight: Option<f64>,
}

// Task types

#[derive(Clone, Debug, SimpleObject)]
pub struct Task {
    pub uid: i32,
    pub index_uid: Option<String>,
    pub status: TaskStatus,
    pub task_type: TaskType,
    pub details: Option<JSON>,
    pub error: Option<TaskError>,
    pub duration: Option<String>,
    pub enqueued_at: OffsetDateTime,
    pub started_at: Option<OffsetDateTime>,
    pub finished_at: Option<OffsetDateTime>,
}

#[derive(Clone, Debug, Enum, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Enqueued,
    Processing,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Clone, Debug, Enum, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    IndexCreation,
    IndexUpdate,
    IndexDeletion,
    DocumentAddition,
    DocumentDeletion,
    SettingsUpdate,
    DumpCreation,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct TaskError {
    pub message: String,
    pub code: String,
    pub error_type: String,
    pub link: String,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct TaskConnection {
    pub results: Vec<Task>,
    pub offset: i32,
    pub limit: i32,
    pub total: i32,
}

#[derive(Clone, Debug, InputObject)]
pub struct TaskFilterInput {
    pub uids: Option<Vec<i32>>,
    pub statuses: Option<Vec<TaskStatus>>,
    pub types: Option<Vec<TaskType>>,
    pub index_uids: Option<Vec<String>>,
}

// Task result types

#[derive(Clone, Debug, SimpleObject)]
pub struct IndexCreationTask {
    pub task_uid: i32,
    pub index_uid: String,
    pub enqueued_at: OffsetDateTime,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct DocumentTask {
    pub task_uid: i32,
    pub index_uid: String,
    pub enqueued_at: OffsetDateTime,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct SettingsTask {
    pub task_uid: i32,
    pub index_uid: String,
    pub enqueued_at: OffsetDateTime,
}

#[derive(Clone, Debug, SimpleObject)]
pub struct IndexDeletionTask {
    pub task_uid: i32,
    pub index_uid: String,
    pub enqueued_at: OffsetDateTime,
}

// Stats types

#[derive(Clone, Debug, SimpleObject)]
pub struct Stats {
    pub database_size: i64,
    pub last_update: Option<OffsetDateTime>,
    pub indexes: JSON,
}

// Health types

#[derive(Clone, Debug, SimpleObject)]
pub struct Health {
    pub status: String,
}
