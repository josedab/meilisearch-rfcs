use actix_web::{web, HttpResponse};
use index_scheduler::IndexScheduler;
use meilisearch_types::deserr::DeserrJsonError;
use meilisearch_types::error::ResponseError;
use serde::{Deserialize, Serialize};

use crate::extractors::authentication::policies::ActionPolicy;
use crate::extractors::authentication::GuardedData;
use crate::extractors::sequential_extractor::SeqHandler;

/// Explain why a document matched or didn't match
pub async fn explain_document(
    index_scheduler: GuardedData<ActionPolicy<{ actions::SEARCH }>, web::Data<IndexScheduler>>,
    index_uid: web::Path<String>,
    params: web::Json<ExplainQuery>,
) -> Result<HttpResponse, ResponseError> {
    let index_uid = index_uid.into_inner();

    // Get index from scheduler
    let index = index_scheduler.index(&index_uid)?;
    let rtxn = index.read_txn()?;

    // Create explainer and generate explanation
    let explainer = QueryExplainer::new(&rtxn, &index);
    let explanation = explainer.explain(&params.query, &params.document_id, params.filter.as_deref())?;

    Ok(HttpResponse::Ok().json(explanation))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExplainQuery {
    pub query: String,
    pub document_id: String,
    #[serde(default)]
    pub filter: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryExplanation {
    pub document_id: String,
    pub matched: bool,
    pub score: Option<f64>,
    pub ranking_breakdown: Vec<RankingRuleExplanation>,
    pub filter_evaluation: Option<FilterExplanation>,
    pub term_matching: Vec<TermMatch>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankingRuleExplanation {
    pub rule: String,
    pub score: f64,
    pub details: String,
    pub contribution_percent: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilterExplanation {
    pub filter: String,
    pub matched: bool,
    pub evaluation: Vec<FilterClause>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilterClause {
    pub clause: String,
    pub matched: bool,
    pub field: String,
    pub operator: String,
    pub value: serde_json::Value,
    pub document_value: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TermMatch {
    pub query_term: String,
    pub matched_word: String,
    pub typo_count: u8,
    pub field: String,
    pub positions: Vec<usize>,
}

pub struct QueryExplainer<'a> {
    rtxn: &'a heed::RoTxn<'a>,
    index: &'a milli::Index,
}

impl<'a> QueryExplainer<'a> {
    pub fn new(rtxn: &'a heed::RoTxn<'a>, index: &'a milli::Index) -> Self {
        Self { rtxn, index }
    }

    pub fn explain(
        &self,
        query: &str,
        document_id: &str,
        filter: Option<&str>,
    ) -> Result<QueryExplanation, meilisearch_types::milli::Error> {
        // 1. Get document
        let external_documents_ids = self.index.external_documents_ids(self.rtxn)?;
        let doc_id = external_documents_ids
            .get(document_id)
            .ok_or_else(|| {
                meilisearch_types::milli::Error::UserError(
                    meilisearch_types::milli::UserError::UnknownInternalDocumentId {
                        document_id: document_id.to_string().into(),
                    }
                )
            })?;

        // 2. Execute search to get scoring information
        let mut search = meilisearch_types::milli::Search::new(self.rtxn, self.index);
        search.query(query);

        if let Some(filter_str) = filter {
            // Parse and set filter
            // Note: This is simplified - real implementation would need proper filter parsing
            // search.filter(filter_condition);
        }

        let search_result = search.execute()?;

        // 3. Check if document is in results and extract scoring
        let matched = search_result.documents_ids.contains(&doc_id);
        let mut score = None;
        let mut ranking_breakdown = Vec::new();

        if matched {
            // Find the document's position in results
            if let Some(position) = search_result.documents_ids.iter().position(|&id| id == doc_id) {
                // Extract score details for this document
                if let Some(score_details) = search_result.document_scores.get(position) {
                    // Process score details into ranking breakdown
                    ranking_breakdown = self.process_score_details(score_details);

                    // Calculate total score
                    let total: f64 = ranking_breakdown.iter().map(|r| r.score).sum();
                    score = Some(total);
                }
            }
        }

        // 4. Evaluate filter (if any)
        let filter_evaluation = filter.map(|f| FilterExplanation {
            filter: f.to_string(),
            matched,
            evaluation: Vec::new(), // TODO: implement detailed filter evaluation
        });

        // 5. Analyze term matching
        let term_matching = self.analyze_term_matching(query, doc_id)?;

        Ok(QueryExplanation {
            document_id: document_id.to_string(),
            matched,
            score,
            ranking_breakdown,
            filter_evaluation,
            term_matching,
        })
    }

    fn process_score_details(
        &self,
        score_details: &[meilisearch_types::milli::score_details::ScoreDetails],
    ) -> Vec<RankingRuleExplanation> {
        let mut explanations = Vec::new();

        for detail in score_details {
            use meilisearch_types::milli::score_details::ScoreDetails;

            let (rule_name, score_value, details_str) = match detail {
                ScoreDetails::Words(words) => {
                    let score = words.matching_words as f64 / words.max_matching_words.max(1) as f64;
                    (
                        "words".to_string(),
                        score,
                        format!("{} of {} words matched", words.matching_words, words.max_matching_words),
                    )
                }
                ScoreDetails::Typo(typo) => {
                    let score = 1.0 - (typo.typo_count as f64 / typo.max_typo_count.max(1) as f64);
                    (
                        "typo".to_string(),
                        score,
                        format!("{} typos (max: {})", typo.typo_count, typo.max_typo_count),
                    )
                }
                ScoreDetails::Proximity(prox) => {
                    (
                        "proximity".to_string(),
                        prox.score as f64,
                        format!("proximity score: {}", prox.score),
                    )
                }
                ScoreDetails::Attribute(attr) => {
                    (
                        "attribute".to_string(),
                        attr.rank as f64,
                        format!("attribute rank: {}", attr.rank),
                    )
                }
                ScoreDetails::Exactness(exact) => {
                    let score = exact.matching_words as f64 / exact.max_matching_words.max(1) as f64;
                    (
                        "exactness".to_string(),
                        score,
                        format!("{} exact matches", exact.matching_words),
                    )
                }
                ScoreDetails::Sort(sort) => {
                    (
                        "sort".to_string(),
                        0.0, // Sort doesn't contribute to relevance score
                        format!("sort value: {}", sort.value),
                    )
                }
                _ => continue,
            };

            explanations.push(RankingRuleExplanation {
                rule: rule_name,
                score: score_value,
                details: details_str,
                contribution_percent: 0.0, // Will be calculated after all rules
            });
        }

        // Calculate contribution percentages
        let total_score: f64 = explanations.iter().map(|e| e.score).sum();
        if total_score > 0.0 {
            for exp in &mut explanations {
                exp.contribution_percent = (exp.score / total_score) * 100.0;
            }
        }

        explanations
    }

    fn analyze_term_matching(
        &self,
        query: &str,
        doc_id: u32,
    ) -> Result<Vec<TermMatch>, meilisearch_types::milli::Error> {
        let mut matches = Vec::new();

        // Simple term extraction (in real implementation, use proper query parsing)
        let terms: Vec<&str> = query.split_whitespace().collect();

        for term in terms {
            // This is a simplified version - real implementation would need to:
            // 1. Use the query graph to find actual matches
            // 2. Check each searchable field
            // 3. Find positions in the document
            // 4. Calculate typo distances

            matches.push(TermMatch {
                query_term: term.to_string(),
                matched_word: term.to_string(),
                typo_count: 0,
                field: "unknown".to_string(), // Would need to determine actual field
                positions: vec![],
            });
        }

        Ok(matches)
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
    fn test_explain_query_deserialization() {
        let json = r#"{
            "query": "test query",
            "documentId": "doc_123",
            "filter": "price > 100"
        }"#;

        let explain_query: ExplainQuery = serde_json::from_str(json).unwrap();
        assert_eq!(explain_query.query, "test query");
        assert_eq!(explain_query.document_id, "doc_123");
        assert_eq!(explain_query.filter, Some("price > 100".to_string()));
    }
}
