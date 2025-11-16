use actix_web::{web, HttpResponse};
use index_scheduler::IndexScheduler;
use meilisearch_types::error::ResponseError;
use serde::Serialize;

use crate::extractors::authentication::policies::ActionPolicy;
use crate::extractors::authentication::GuardedData;

/// Interactive query builder endpoint
/// Returns metadata about an index to help build queries
pub async fn query_builder(
    index_scheduler: GuardedData<ActionPolicy<{ actions::SEARCH }>, web::Data<IndexScheduler>>,
    index_uid: web::Path<String>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // Get index from scheduler
    let index = index_scheduler.index(&index_uid)?;
    let rtxn = index.read_txn()?;

    // Gather metadata
    let searchable_attributes = index
        .searchable_fields(&rtxn)?
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let filterable_attributes = index
        .filterable_fields(&rtxn)?
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let sortable_attributes = index
        .sortable_fields(&rtxn)?
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Get faceted fields (same as filterable in most cases)
    let facetable_attributes = filterable_attributes.clone();

    // Get ranking rules
    let criteria = index.criteria(&rtxn)?;
    let ranking_rules: Vec<String> = criteria
        .iter()
        .map(|c| format!("{:?}", c).to_lowercase())
        .collect();

    // Get typo tolerance settings
    let min_typo_word_len = index
        .min_word_len_one_typo(&rtxn)
        .unwrap_or(5);

    // Get embedder names if any
    let embedders = get_embedder_names(&rtxn, &index)?;

    // Get a sample document for reference
    let example_document = get_sample_document(&rtxn, &index)?;

    let metadata = QueryBuilderMetadata {
        searchable_attributes,
        filterable_attributes,
        sortable_attributes,
        facetable_attributes,
        ranking_rules,
        typo_tolerance: min_typo_word_len,
        embedders,
        example_document,
    };

    Ok(HttpResponse::Ok().json(metadata))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryBuilderMetadata {
    pub searchable_attributes: Vec<String>,
    pub filterable_attributes: Vec<String>,
    pub sortable_attributes: Vec<String>,
    pub facetable_attributes: Vec<String>,
    pub ranking_rules: Vec<String>,
    pub typo_tolerance: u8,
    pub embedders: Vec<String>,
    pub example_document: Option<serde_json::Value>,
}

fn get_embedder_names(
    rtxn: &heed::RoTxn,
    index: &milli::Index,
) -> Result<Vec<String>, meilisearch_types::milli::Error> {
    // Get embedder configurations
    let embedding_configs = index.embedding_configs(rtxn)?;

    Ok(embedding_configs
        .into_iter()
        .map(|(name, _config)| name)
        .collect())
}

fn get_sample_document(
    rtxn: &heed::RoTxn,
    index: &milli::Index,
) -> Result<Option<serde_json::Value>, meilisearch_types::milli::Error> {
    // Get the first document from the index as an example
    let documents = index.documents(rtxn, std::iter::once(0))?;

    if let Some(Ok((_doc_id, reader))) = documents.into_iter().next() {
        // Convert obkv document to JSON
        let document = meilisearch_types::milli::obkv_to_json(
            &index.displayed_fields(rtxn)?.map(|fields| {
                fields.iter().map(|s| s.to_string()).collect()
            }).unwrap_or_default(),
            reader,
        )?;

        Ok(Some(document))
    } else {
        Ok(None)
    }
}

// Module to export actions constants
mod actions {
    pub const SEARCH: u8 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder_metadata_serialization() {
        let metadata = QueryBuilderMetadata {
            searchable_attributes: vec!["title".to_string(), "description".to_string()],
            filterable_attributes: vec!["price".to_string(), "category".to_string()],
            sortable_attributes: vec!["price".to_string(), "created_at".to_string()],
            facetable_attributes: vec!["brand".to_string(), "category".to_string()],
            ranking_rules: vec!["words".to_string(), "typo".to_string()],
            typo_tolerance: 5,
            embedders: vec!["default".to_string()],
            example_document: Some(serde_json::json!({
                "id": "1",
                "title": "Test Product",
                "price": 99.99
            })),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("searchableAttributes"));
        assert!(json.contains("filterableAttributes"));
    }
}
