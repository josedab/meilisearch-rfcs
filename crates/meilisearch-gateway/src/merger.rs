use crate::error::Result;
use crate::types::{ScoreDetails, SearchQuery, SearchResult, ShardSearchResult};
use std::collections::HashMap;

/// Result merger for aggregating shard results
pub struct ResultMerger;

impl ResultMerger {
    /// Create a new result merger
    pub fn new() -> Self {
        Self
    }

    /// Merge search results from multiple shards
    pub fn merge_search_results(
        &self,
        shard_results: Vec<ShardSearchResult>,
        query: &SearchQuery,
    ) -> Result<SearchResult> {
        if shard_results.is_empty() {
            return Ok(SearchResult {
                hits: vec![],
                query: query.q.clone().unwrap_or_default(),
                processing_time_ms: 0,
                limit: query.limit.unwrap_or(20),
                offset: query.offset.unwrap_or(0),
                estimated_total_hits: 0,
                facet_distribution: None,
            });
        }

        // Collect all documents with their scores
        let mut all_docs: Vec<(String, Vec<ScoreDetails>, serde_json::Value)> = Vec::new();

        for shard_result in &shard_results {
            for i in 0..shard_result.documents_ids.len() {
                let doc_id = shard_result.documents_ids[i].clone();
                let scores = shard_result.document_scores.get(i).cloned().unwrap_or_default();
                let doc = shard_result.documents.get(i).cloned().unwrap_or(serde_json::json!({}));

                all_docs.push((doc_id, scores, doc));
            }
        }

        // Sort by score globally
        all_docs.sort_by(|a, b| {
            ScoreDetails::global_score(b.1.iter())
                .partial_cmp(&ScoreDetails::global_score(a.1.iter()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_hits = all_docs.len();

        // Apply offset and limit
        let offset = query.offset.unwrap_or(0);
        let limit = query.limit.unwrap_or(20);
        let end = std::cmp::min(offset + limit, all_docs.len());

        let hits: Vec<_> = all_docs
            .into_iter()
            .skip(offset)
            .take(end.saturating_sub(offset))
            .map(|(_, _, doc)| doc)
            .collect();

        // Merge facets if requested
        let facet_distribution = if query.facets.is_some() {
            let facets: Vec<_> = shard_results
                .iter()
                .filter_map(|r| r.facet_distribution.clone())
                .collect();
            Some(self.merge_facet_distribution(facets))
        } else {
            None
        };

        // Sum processing times
        let processing_time_ms = shard_results.iter().map(|r| r.processing_time_ms).max().unwrap_or(0);

        Ok(SearchResult {
            hits,
            query: query.q.clone().unwrap_or_default(),
            processing_time_ms,
            limit,
            offset,
            estimated_total_hits: total_hits,
            facet_distribution,
        })
    }

    /// Merge facet distributions from multiple shards
    pub fn merge_facet_distribution(
        &self,
        shard_facets: Vec<HashMap<String, HashMap<String, u64>>>,
    ) -> HashMap<String, HashMap<String, u64>> {
        let mut merged: HashMap<String, HashMap<String, u64>> = HashMap::new();

        for shard_facet in shard_facets {
            for (facet_name, facet_values) in shard_facet {
                let merged_values = merged.entry(facet_name).or_insert_with(HashMap::new);

                for (value, count) in facet_values {
                    *merged_values.entry(value).or_insert(0) += count;
                }
            }
        }

        merged
    }
}

impl Default for ResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_empty_results() {
        let merger = ResultMerger::new();
        let query = SearchQuery {
            q: Some("test".to_string()),
            offset: None,
            limit: None,
            filter: None,
            facets: None,
            attributes_to_retrieve: None,
            attributes_to_highlight: None,
            show_matches_position: None,
        };

        let result = merger.merge_search_results(vec![], &query).unwrap();
        assert_eq!(result.hits.len(), 0);
    }

    #[test]
    fn test_merge_facets() {
        let merger = ResultMerger::new();

        let mut facet1 = HashMap::new();
        let mut genre_values1 = HashMap::new();
        genre_values1.insert("action".to_string(), 10);
        genre_values1.insert("drama".to_string(), 5);
        facet1.insert("genre".to_string(), genre_values1);

        let mut facet2 = HashMap::new();
        let mut genre_values2 = HashMap::new();
        genre_values2.insert("action".to_string(), 7);
        genre_values2.insert("comedy".to_string(), 3);
        facet2.insert("genre".to_string(), genre_values2);

        let merged = merger.merge_facet_distribution(vec![facet1, facet2]);

        assert_eq!(merged.get("genre").unwrap().get("action").unwrap(), &17);
        assert_eq!(merged.get("genre").unwrap().get("drama").unwrap(), &5);
        assert_eq!(merged.get("genre").unwrap().get("comedy").unwrap(), &3);
    }
}
