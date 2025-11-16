use serde::{Deserialize, Serialize};

/// Enhanced error with actionable suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct EnhancedError {
    /// Human-readable error message
    pub message: String,

    /// Error code for programmatic handling
    pub code: String,

    /// Error type category
    #[serde(rename = "type")]
    pub error_type: ErrorType,

    /// Detailed error context
    pub details: Option<ErrorDetails>,

    /// Suggestion for fixing the error
    pub suggestion: Option<String>,

    /// Concrete fix (API call to make)
    pub how_to_fix: Option<ErrorFix>,

    /// Documentation link
    pub link: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ErrorType {
    InvalidRequest,
    IndexNotFound,
    DocumentNotFound,
    InternalError,
    AuthenticationError,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    /// Field or parameter that caused error
    pub field: Option<String>,

    /// Reason code
    pub reason: String,

    /// Current value (if applicable)
    pub current_value: Option<serde_json::Value>,

    /// Expected value or format
    pub expected: Option<String>,

    /// Related settings or configuration
    pub related_config: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorFix {
    /// HTTP method
    pub method: String,

    /// API endpoint
    pub endpoint: String,

    /// Request body
    pub body: Option<serde_json::Value>,

    /// cURL command
    pub curl_example: String,
}

impl EnhancedError {
    /// Create error for field not filterable
    pub fn field_not_filterable(field: &str, index_uid: &str) -> Self {
        Self {
            message: format!("Invalid filter: Field '{}' is not filterable", field),
            code: "field_not_filterable".to_string(),
            error_type: ErrorType::InvalidRequest,
            details: Some(ErrorDetails {
                field: Some(field.to_string()),
                reason: "not_in_filterable_attributes".to_string(),
                current_value: None,
                expected: Some("Field must be in filterableAttributes".to_string()),
                related_config: Some(vec!["filterableAttributes".to_string()]),
            }),
            suggestion: Some(format!(
                "Add '{}' to filterableAttributes in index settings",
                field
            )),
            how_to_fix: Some(ErrorFix {
                method: "PATCH".to_string(),
                endpoint: format!("/indexes/{}/settings", index_uid),
                body: Some(serde_json::json!({
                    "filterableAttributes": [field]
                })),
                curl_example: format!(
                    "curl -X PATCH 'http://localhost:7700/indexes/{}/settings' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"filterableAttributes\": [\"{}\"]}}' ",
                    index_uid, field
                ),
            }),
            link: "https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_search_results#filterable-attributes".to_string(),
        }
    }

    /// Create error for invalid filter syntax
    pub fn invalid_filter_syntax(filter: &str, position: usize, expected: &str) -> Self {
        let context = get_error_context(filter, position, 20);

        Self {
            message: format!("Invalid filter syntax at position {}", position),
            code: "invalid_filter_syntax".to_string(),
            error_type: ErrorType::InvalidRequest,
            details: Some(ErrorDetails {
                field: Some("filter".to_string()),
                reason: "syntax_error".to_string(),
                current_value: Some(serde_json::json!(filter)),
                expected: Some(expected.to_string()),
                related_config: None,
            }),
            suggestion: Some(format!(
                "Expected '{}' at position {}. Context: {}",
                expected, position, context
            )),
            how_to_fix: None,
            link: "https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_expression_reference".to_string(),
        }
    }

    /// Create error for field not sortable
    pub fn field_not_sortable(field: &str, index_uid: &str) -> Self {
        Self {
            message: format!("Invalid sort: Field '{}' is not sortable", field),
            code: "field_not_sortable".to_string(),
            error_type: ErrorType::InvalidRequest,
            details: Some(ErrorDetails {
                field: Some(field.to_string()),
                reason: "not_in_sortable_attributes".to_string(),
                current_value: None,
                expected: Some("Field must be in sortableAttributes".to_string()),
                related_config: Some(vec!["sortableAttributes".to_string()]),
            }),
            suggestion: Some(format!(
                "Add '{}' to sortableAttributes in index settings",
                field
            )),
            how_to_fix: Some(ErrorFix {
                method: "PATCH".to_string(),
                endpoint: format!("/indexes/{}/settings", index_uid),
                body: Some(serde_json::json!({
                    "sortableAttributes": [field]
                })),
                curl_example: format!(
                    "curl -X PATCH 'http://localhost:7700/indexes/{}/settings' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"sortableAttributes\": [\"{}\"]}}' ",
                    index_uid, field
                ),
            }),
            link: "https://www.meilisearch.com/docs/learn/filtering_and_sorting/sort_search_results#sortable-attributes".to_string(),
        }
    }

    /// Create error for index not found
    pub fn index_not_found(index_uid: &str) -> Self {
        Self {
            message: format!("Index '{}' not found", index_uid),
            code: "index_not_found".to_string(),
            error_type: ErrorType::IndexNotFound,
            details: Some(ErrorDetails {
                field: Some("indexUid".to_string()),
                reason: "index_does_not_exist".to_string(),
                current_value: Some(serde_json::json!(index_uid)),
                expected: Some("Existing index UID".to_string()),
                related_config: None,
            }),
            suggestion: Some(format!(
                "Create index '{}' before performing this operation",
                index_uid
            )),
            how_to_fix: Some(ErrorFix {
                method: "POST".to_string(),
                endpoint: "/indexes".to_string(),
                body: Some(serde_json::json!({
                    "uid": index_uid,
                    "primaryKey": "id"
                })),
                curl_example: format!(
                    "curl -X POST 'http://localhost:7700/indexes' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"uid\": \"{}\", \"primaryKey\": \"id\"}}' ",
                    index_uid
                ),
            }),
            link: "https://www.meilisearch.com/docs/reference/api/indexes#create-an-index".to_string(),
        }
    }

    /// Create error for document not found
    pub fn document_not_found(document_id: &str, index_uid: &str) -> Self {
        Self {
            message: format!("Document '{}' not found in index '{}'", document_id, index_uid),
            code: "document_not_found".to_string(),
            error_type: ErrorType::DocumentNotFound,
            details: Some(ErrorDetails {
                field: Some("documentId".to_string()),
                reason: "document_does_not_exist".to_string(),
                current_value: Some(serde_json::json!(document_id)),
                expected: Some("Existing document ID".to_string()),
                related_config: None,
            }),
            suggestion: Some(format!(
                "Check that document '{}' exists in index '{}' or add it first",
                document_id, index_uid
            )),
            how_to_fix: None,
            link: "https://www.meilisearch.com/docs/reference/api/documents#get-one-document".to_string(),
        }
    }
}

fn get_error_context(text: &str, position: usize, window: usize) -> String {
    let start = position.saturating_sub(window);
    let end = (position + window).min(text.len());
    let context = &text[start..end];
    let pointer_pos = position - start;

    let before: String = context.chars().take(pointer_pos).collect();
    let after: String = context.chars().skip(pointer_pos).collect();

    format!("\"{}^{}\"", before, after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_not_filterable_error() {
        let error = EnhancedError::field_not_filterable("created_at", "products");

        assert_eq!(error.code, "field_not_filterable");
        assert!(error.message.contains("created_at"));
        assert!(error.suggestion.is_some());
        assert!(error.how_to_fix.is_some());
    }

    #[test]
    fn test_invalid_filter_syntax_error() {
        let error = EnhancedError::invalid_filter_syntax("price > ", 8, "number");

        assert_eq!(error.code, "invalid_filter_syntax");
        assert!(error.details.is_some());
    }

    #[test]
    fn test_error_context() {
        let context = get_error_context("price > 100 AND category = 'electronics'", 8, 5);
        assert!(context.contains("^"));
    }
}
