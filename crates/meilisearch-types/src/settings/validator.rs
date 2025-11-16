use serde::Serialize;
use std::collections::HashSet;

/// Validate settings and provide helpful suggestions
pub struct SettingsValidator;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub suggestions: Vec<ValidationSuggestion>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub current_value: Option<serde_json::Value>,
    pub fix: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub impact: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationSuggestion {
    pub category: String,
    pub suggestion: String,
    pub benefit: String,
}

impl SettingsValidator {
    /// Validate settings and provide helpful feedback
    pub fn validate(
        searchable_attributes: Option<&[String]>,
        filterable_attributes: Option<&[String]>,
        sortable_attributes: Option<&[String]>,
        ranking_rules: Option<&[String]>,
        embedders: Option<&serde_json::Value>,
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // 1. Validate filterable attributes aren't unnecessarily in searchable
        if let (Some(filterable), Some(searchable)) = (filterable_attributes, searchable_attributes) {
            let filterable_set: HashSet<_> = filterable.iter().collect();
            let searchable_set: HashSet<_> = searchable.iter().collect();
            let overlap: Vec<_> = filterable_set.intersection(&searchable_set).collect();

            if !overlap.is_empty() {
                let overlap_strs: Vec<String> = overlap.iter().map(|s| s.to_string()).collect();
                warnings.push(ValidationWarning {
                    field: "filterableAttributes".to_string(),
                    message: format!(
                        "Fields {:?} are both filterable and searchable. This increases index size.",
                        overlap_strs
                    ),
                    impact: "Index size may be 20-30% larger than necessary".to_string(),
                });

                suggestions.push(ValidationSuggestion {
                    category: "optimization".to_string(),
                    suggestion: "Consider removing filterable fields from searchableAttributes if full-text search is not needed on them".to_string(),
                    benefit: "Reduce index size by 20-30%".to_string(),
                });
            }
        }

        // 2. Validate ranking rules order
        if let Some(ranking_rules) = ranking_rules {
            if ranking_rules.contains(&"sort".to_string()) {
                if let Some(sort_index) = ranking_rules.iter().position(|r| r == "sort") {
                    if let Some(words_index) = ranking_rules.iter().position(|r| r == "words") {
                        if sort_index < words_index {
                            warnings.push(ValidationWarning {
                                field: "rankingRules".to_string(),
                                message: "'sort' appears before 'words'. This may produce unexpected results.".to_string(),
                                impact: "Search relevancy may be poor for text queries".to_string(),
                            });

                            suggestions.push(ValidationSuggestion {
                                category: "relevancy".to_string(),
                                suggestion: "Move 'sort' after 'words', 'typo', and 'proximity' for better text search relevancy".to_string(),
                                benefit: "Improve search relevancy while maintaining custom sorting".to_string(),
                            });
                        }
                    }
                }
            }

            // Check for duplicate ranking rules
            let mut seen = HashSet::new();
            let mut duplicates = Vec::new();
            for rule in ranking_rules {
                if !seen.insert(rule) {
                    duplicates.push(rule.clone());
                }
            }

            if !duplicates.is_empty() {
                errors.push(ValidationError {
                    field: "rankingRules".to_string(),
                    message: format!("Duplicate ranking rules found: {:?}", duplicates),
                    current_value: Some(serde_json::json!(ranking_rules)),
                    fix: "Remove duplicate ranking rules".to_string(),
                });
            }
        }

        // 3. Check for unused embedders
        if embedders.is_some() {
            suggestions.push(ValidationSuggestion {
                category: "feature_usage".to_string(),
                suggestion: "Embedder configured. Ensure documents contain vector fields to utilize this feature.".to_string(),
                benefit: "Enable semantic search capabilities".to_string(),
            });
        }

        // 4. Detect potential typos in attribute names
        if let Some(searchable) = searchable_attributes {
            let common_typos = vec![
                ("titel", "title"),
                ("desciption", "description"),
                ("cateogry", "category"),
                ("prie", "price"),
                ("autor", "author"),
                ("iamge", "image"),
            ];

            for attr in searchable {
                for (typo, correct) in &common_typos {
                    if attr == typo {
                        warnings.push(ValidationWarning {
                            field: "searchableAttributes".to_string(),
                            message: format!("Possible typo: '{}' - did you mean '{}'?", typo, correct),
                            impact: "Field may not exist in documents, leading to poor search results".to_string(),
                        });
                    }
                }
            }
        }

        // 5. Check for sortable fields that should also be filterable
        if let (Some(sortable), Some(filterable)) = (sortable_attributes, filterable_attributes) {
            let filterable_set: HashSet<_> = filterable.iter().collect();
            let not_filterable: Vec<_> = sortable.iter()
                .filter(|s| !filterable_set.contains(s))
                .collect();

            if !not_filterable.is_empty() {
                suggestions.push(ValidationSuggestion {
                    category: "best_practice".to_string(),
                    suggestion: format!(
                        "Fields {:?} are sortable but not filterable. Consider making them filterable for better query capabilities.",
                        not_filterable
                    ),
                    benefit: "Enable filtering on sorted fields for more flexible queries".to_string(),
                });
            }
        }

        // 6. Validate attribute name conventions
        if let Some(searchable) = searchable_attributes {
            for attr in searchable {
                // Check for spaces in attribute names
                if attr.contains(' ') {
                    warnings.push(ValidationWarning {
                        field: "searchableAttributes".to_string(),
                        message: format!("Attribute '{}' contains spaces. This may cause issues.", attr),
                        impact: "Querying this field may be difficult or error-prone".to_string(),
                    });
                }

                // Check for very long attribute names
                if attr.len() > 100 {
                    warnings.push(ValidationWarning {
                        field: "searchableAttributes".to_string(),
                        message: format!("Attribute '{}' is very long ({} characters).", attr, attr.len()),
                        impact: "Long attribute names increase index size and may affect performance".to_string(),
                    });
                }
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            suggestions,
        }
    }

    /// Quick validation that only checks for critical errors
    pub fn validate_critical(
        ranking_rules: Option<&[String]>,
    ) -> ValidationResult {
        let mut errors = Vec::new();

        if let Some(ranking_rules) = ranking_rules {
            // Check for duplicate ranking rules
            let mut seen = HashSet::new();
            let mut duplicates = Vec::new();
            for rule in ranking_rules {
                if !seen.insert(rule) {
                    duplicates.push(rule.clone());
                }
            }

            if !duplicates.is_empty() {
                errors.push(ValidationError {
                    field: "rankingRules".to_string(),
                    message: format!("Duplicate ranking rules found: {:?}", duplicates),
                    current_value: Some(serde_json::json!(ranking_rules)),
                    fix: "Remove duplicate ranking rules".to_string(),
                });
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlapping_attributes() {
        let searchable = vec!["title".to_string(), "description".to_string(), "price".to_string()];
        let filterable = vec!["price".to_string(), "category".to_string()];

        let result = SettingsValidator::validate(
            Some(&searchable),
            Some(&filterable),
            None,
            None,
            None,
        );

        assert!(result.valid);
        assert!(!result.warnings.is_empty());
        assert!(result.warnings[0].field == "filterableAttributes");
    }

    #[test]
    fn test_ranking_rules_order() {
        let ranking_rules = vec!["sort".to_string(), "words".to_string(), "typo".to_string()];

        let result = SettingsValidator::validate(
            None,
            None,
            None,
            Some(&ranking_rules),
            None,
        );

        assert!(result.valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_duplicate_ranking_rules() {
        let ranking_rules = vec!["words".to_string(), "typo".to_string(), "words".to_string()];

        let result = SettingsValidator::validate(
            None,
            None,
            None,
            Some(&ranking_rules),
            None,
        );

        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_typo_detection() {
        let searchable = vec!["titel".to_string(), "description".to_string()];

        let result = SettingsValidator::validate(
            Some(&searchable),
            None,
            None,
            None,
            None,
        );

        assert!(result.valid);
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.message.contains("typo")));
    }

    #[test]
    fn test_attribute_with_spaces() {
        let searchable = vec!["my field".to_string()];

        let result = SettingsValidator::validate(
            Some(&searchable),
            None,
            None,
            None,
            None,
        );

        assert!(result.valid);
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.message.contains("spaces")));
    }
}
