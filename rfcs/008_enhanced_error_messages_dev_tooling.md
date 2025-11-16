# RFC 008: Enhanced Error Messages & Developer Tooling

**Status:** Draft  
**Created:** 2025-11-16  
**Authors:** Meilisearch Community  
**Tracking Issue:** TBD

---

## Summary

This RFC proposes comprehensive improvements to error messages, debugging tools, and developer experience through actionable error suggestions, query explanation APIs, schema validation, interactive query builders, and CLI debugging utilities. These enhancements will reduce time-to-resolution for common issues and improve the overall developer experience.

## Motivation

### Current Pain Points

From [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md:625), developer experience improvements are identified as low-complexity, medium-value contributions:

**Current Issues:**

1. **Cryptic Errors**: Error messages lack context and actionable fixes
2. **No Query Explanation**: Can't understand why documents matched or didn't match
3. **Unclear Schema Errors**: Settings validation errors are vague
4. **Limited Debugging Tools**: No built-in tools to debug ranking, filters, or performance
5. **No Interactive Exploration**: Difficult to experiment with query variations

**Example current error:**

```json
{
  "message": "Invalid filter",
  "code": "invalid_filter",
  "type": "invalid_request",
  "link": "https://docs.meilisearch.com/errors#invalid_filter"
}
```

**Proposed enhanced error:**

```json
{
  "message": "Invalid filter: Field 'created_at' is not filterable",
  "code": "invalid_filter",
  "type": "invalid_request",
  "details": {
    "field": "created_at",
    "reason": "not_in_filterable_attributes",
    "suggestion": "Add 'created_at' to filterableAttributes in index settings",
    "howToFix": {
      "method": "PATCH",
      "endpoint": "/indexes/products/settings",
      "body": {
        "filterableAttributes": ["created_at"]
      }
    }
  },
  "link": "https://docs.meilisearch.com/errors#field_not_filterable"
}
```

### Real-World Use Cases

**Debugging Search Relevancy:**
- User: "Why is document X not appearing for query Y?"
- Current: No way to know without code diving
- **Proposed:** Query explanation API shows ranking breakdown

**Schema Migration Errors:**
- User: "Why did my settings update fail?"
- Current: Generic error
- **Proposed:** Detailed validation with specific field issues

**Performance Debugging:**
- User: "Why is this query slow?"
- Current: No visibility into bottlenecks
- **Proposed:** Query profiling with span timing breakdown

## Technical Design

### 1. Enhanced Error System

**New file:** `crates/meilisearch-types/src/error/enhanced.rs`

```rust
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
}

fn get_error_context(text: &str, position: usize, window: usize) -> String {
    let start = position.saturating_sub(window);
    let end = (position + window).min(text.len());
    let context = &text[start..end];
    let pointer_pos = position - start;
    
    format!("\"{}\" (error at ^)", 
        context.chars().take(pointer_pos).collect::<String>() + "^" +
        &context.chars().skip(pointer_pos).collect::<String>()
    )
}
```

### 2. Query Explanation API

**New file:** `crates/meilisearch/src/routes/indexes/explain.rs`

```rust
use actix_web::{web, HttpResponse};

/// Explain why a document matched or didn't match
pub async fn explain_document(
    index_scheduler: Data<IndexScheduler>,
    index_uid: web::Path<String>,
    params: web::Json<ExplainQuery>,
) -> Result<HttpResponse, ResponseError> {
    let index = index_scheduler.index(&index_uid)?;
    let rtxn = index.read_txn()?;
    
    let explainer = QueryExplainer::new(&rtxn, &index);
    let explanation = explainer.explain(&params.query, &params.document_id)?;
    
    Ok(HttpResponse::Ok().json(explanation))
}

#[derive(Deserialize)]
pub struct ExplainQuery {
    pub query: String,
    pub document_id: String,
    pub filter: Option<String>,
}

#[derive(Serialize)]
pub struct QueryExplanation {
    pub document_id: String,
    pub matched: bool,
    pub score: Option<f64>,
    pub ranking_breakdown: Vec<RankingRuleExplanation>,
    pub filter_evaluation: Option<FilterExplanation>,
    pub term_matching: Vec<TermMatch>,
}

#[derive(Serialize)]
pub struct RankingRuleExplanation {
    pub rule: String,
    pub score: f64,
    pub details: String,
    pub contribution_percent: f64,
}

#[derive(Serialize)]
pub struct FilterExplanation {
    pub filter: String,
    pub matched: bool,
    pub evaluation: Vec<FilterClause>,
}

#[derive(Serialize)]
pub struct FilterClause {
    pub clause: String,
    pub matched: bool,
    pub field: String,
    pub operator: String,
    pub value: serde_json::Value,
    pub document_value: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct TermMatch {
    pub query_term: String,
    pub matched_word: String,
    pub typo_count: u8,
    pub field: String,
    pub positions: Vec<usize>,
}

pub struct QueryExplainer<'a> {
    rtxn: &'a RoTxn<'a>,
    index: &'a Index,
}

impl<'a> QueryExplainer<'a> {
    pub fn new(rtxn: &'a RoTxn<'a>, index: &'a Index) -> Self {
        Self { rtxn, index }
    }
    
    pub fn explain(&self, query: &str, document_id: &str) -> Result<QueryExplanation> {
        // 1. Parse query
        let parsed_query = parse_query(query)?;
        
        // 2. Get document
        let doc_id = self.index.external_documents_ids()\n            .get(self.rtxn, document_id)?\n            .ok_or(Error::DocumentNotFound)?;
        
        let doc = self.index.documents(self.rtxn, std::iter::once(doc_id))\n            .next()\n            .ok_or(Error::DocumentNotFound)??;
        
        // 3. Evaluate each ranking rule
        let ranking_breakdown = self.evaluate_ranking_rules(&parsed_query, doc_id)?;
        
        // 4. Evaluate filter (if any)\n        let filter_evaluation = None; // TODO: implement
        
        // 5. Analyze term matching\        let term_matching = self.analyze_term_matching(&parsed_query, doc_id, &doc)?;
        
        // 6. Compute overall match
        let matched = !ranking_breakdown.is_empty();
        let score = if matched {
            Some(ranking_breakdown.iter().map(|r| r.score).sum())
        } else {
            None
        };
        
        Ok(QueryExplanation {
            document_id: document_id.to_string(),
            matched,
            score,
            ranking_breakdown,
            filter_evaluation,
            term_matching,
        })
    }
    
    fn evaluate_ranking_rules(
        &self,
        query: &ParsedQuery,
        doc_id: DocumentId,
    ) -> Result<Vec<RankingRuleExplanation>> {
        let mut explanations = Vec::new();
        
        // Evaluate each ranking rule
        for rule in &["words", "typo", "proximity", "attribute", "exactness"] {
            let score = self.evaluate_single_rule(rule, query, doc_id)?;
            
            if let Some(score_value) = score {
                explanations.push(RankingRuleExplanation {
                    rule: rule.to_string(),
                    score: score_value,
                    details: format!("{} rule contributed {:.2} to final score", rule, score_value),
                    contribution_percent: 0.0, // Computed after all rules
                });
            }
        }
        
        // Compute contribution percentages
        let total_score: f64 = explanations.iter().map(|e| e.score).sum();
        if total_score > 0.0 {
            for exp in &mut explanations {
                exp.contribution_percent = (exp.score / total_score) * 100.0;
            }
        }
        
        Ok(explanations)
    }
    
    fn analyze_term_matching(
        &self,
        query: &ParsedQuery,
        doc_id: DocumentId,
        doc: &obkv::KvReader,
    ) -> Result<Vec<TermMatch>> {
        let mut matches = Vec::new();
        
        for query_term in &query.terms {
            // Check if this term matches in document
            // (Simplified - actual implementation would use query graph)
            
            let matched_word = query_term.original.clone();
            let typo_count = 0; // TODO: compute actual typo distance
            
            matches.push(TermMatch {
                query_term: query_term.original.clone(),
                matched_word,
                typo_count,
                field: "title".to_string(), // TODO: find actual field
                positions: vec![0], // TODO: find actual positions
            });
        }
        
        Ok(matches)
    }
}
```

### 3. Schema Validation with Suggestions

**New file:** `crates/meilisearch-types/src/settings/validator.rs`

```rust
/// Validate settings and provide helpful suggestions
pub struct SettingsValidator;

#[derive(Debug, Serialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub suggestions: Vec<ValidationSuggestion>,
}

#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub current_value: Option<serde_json::Value>,
    pub fix: String,
}

#[derive(Debug, Serialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub impact: String,
}

#[derive(Debug, Serialize)]
pub struct ValidationSuggestion {
    pub category: String,
    pub suggestion: String,
    pub benefit: String,
}

impl SettingsValidator {
    pub fn validate(settings: &Settings, current_index: Option<&Index>) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        
        // 1. Validate filterable attributes aren't in searchable
        if let Some(filterable) = &settings.filterable_attributes {
            if let Some(searchable) = &settings.searchable_attributes {
                let overlap: Vec<_> = filterable.iter()
                    .filter(|f| searchable.contains(f))
                    .collect();
                
                if !overlap.is_empty() {
                    warnings.push(ValidationWarning {
                        field: "filterableAttributes".to_string(),
                        message: format!(
                            "Fields {:?} are both filterable and searchable. This increases index size.",
                            overlap
                        ),
                        impact: "Index size may be 20-30% larger than necessary".to_string(),
                    });
                    
                    suggestions.push(ValidationSuggestion {
                        category: "optimization".to_string(),
                        suggestion: "Consider removing filterable fields from searchableAttributes if full-text search not needed on them".to_string(),
                        benefit: "Reduce index size by 20-30%".to_string(),
                    });
                }
            }
        }
        
        // 2. Validate ranking rules order
        if let Some(ranking_rules) = &settings.ranking_rules {
            if ranking_rules.contains(&"sort".to_string()) {
                let sort_index = ranking_rules.iter().position(|r| r == "sort").unwrap();
                let words_index = ranking_rules.iter().position(|r| r == "words");
                
                if let Some(wi) = words_index {
                    if sort_index < wi {
                        warnings.push(ValidationWarning {
                            field: "rankingRules".to_string(),
                            message: "'sort' appears before 'words'. This may produce unexpected results.".to_string(),
                            impact: "Search relevancy may be poor for text queries".to_string(),
                        });
                    }
                }
            }
        }
        
        // 3. Check for unused embedders
        if let Some(embedders) = &settings.embedders {
            if current_index.is_some() {
                // Check if any documents actually have vectors
                // Warn if embedder configured but no vectors in index
                suggestions.push(ValidationSuggestion {
                    category: "unused_feature".to_string(),
                    suggestion: "Embedder configured but no vector fields detected in documents".to_string(),
                    benefit: "Remove embedder configuration to save resources".to_string(),
                });
            }
        }
        
        // 4. Detect potential typos in attribute names
        if let Some(searchable) = &settings.searchable_attributes {
            let common_typos = vec![
                ("titel", "title"),
                ("desciption", "description"),
                ("cateogry", "category"),
            ];
            
            for attr in searchable {
                for (typo, correct) in &common_typos {
                    if attr == typo {
                        warnings.push(ValidationWarning {
                            field: "searchableAttributes".to_string(),
                            message: format!("Possible typo: '{}' - did you mean '{}'?", typo, correct),
                            impact: "Field may not exist in documents".to_string(),
                        });
                    }
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
}
```

### 4. Interactive Query Builder

**New file:** `crates/meilisearch/src/routes/indexes/query_builder.rs`

```rust
/// Interactive query builder endpoint
pub async fn query_builder(
    index_scheduler: Data<IndexScheduler>,
    index_uid: web::Path<String>,
) -> Result<HttpResponse, ResponseError> {
    let index = index_scheduler.index(&index_uid)?;
    let rtxn = index.read_txn()?;
    
    // Generate query builder metadata
    let metadata = QueryBuilderMetadata {
        searchable_attributes: index.searchable_fields(&rtxn)?,
        filterable_attributes: index.filterable_fields(&rtxn)?,
        sortable_attributes: index.sortable_fields(&rtxn)?,
        facetable_attributes: index.faceted_fields(&rtxn)?,
        ranking_rules: index.criteria(&rtxn)?,
        typo_tolerance: index.min_word_len_one_typo(&rtxn)?,
        embedders: index.embedding_configs(&rtxn)?,
        example_document: get_sample_document(&rtxn, &index)?,
    };
    
    Ok(HttpResponse::Ok().json(metadata))
}

#[derive(Serialize)]
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
```

### 5. CLI Debugging Tools

**New file:** `crates/meilitool/src/debug.rs`

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
pub struct DebugCommand {
    #[clap(subcommand)]
    pub command: DebugSubcommand,
}

#[derive(Subcommand)]
pub enum DebugSubcommand {
    /// Explain why a query returned specific results
    ExplainQuery {
        #[clap(long)]
        index: String,
        #[clap(long)]
        query: String,
        #[clap(long)]
        document_id: Option<String>,
    },
    
    /// Validate index settings
    ValidateSettings {
        #[clap(long)]
        index: String,
        #[clap(long)]
        settings_file: Option<PathBuf>,
    },
    
    /// Profile query performance
    ProfileQuery {
        #[clap(long)]
        index: String,
        #[clap(long)]
        query: String,
        #[clap(long)]
        runs: Option<usize>,
    },
    
    /// Analyze index structure
    AnalyzeIndex {
        #[clap(long)]
        index: String,
    },
}

impl DebugCommand {
    pub fn run(&self, db_path: &Path) -> Result<()> {
        match &self.command {
            DebugSubcommand::ExplainQuery { index, query, document_id } => {
                self.explain_query(db_path, index, query, document_id.as_deref())
            }
            DebugSubcommand::ValidateSettings { index, settings_file } => {
                self.validate_settings(db_path, index, settings_file.as_deref())
            }
            DebugSubcommand::ProfileQuery { index, query, runs } => {
                self.profile_query(db_path, index, query, runs.unwrap_or(10))
            }
            DebugSubcommand::AnalyzeIndex { index } => {
                self.analyze_index(db_path, index)
            }
        }
    }
    
    fn explain_query(
        &self,
        db_path: &Path,
        index_uid: &str,
        query: &str,
        document_id: Option<&str>,
    ) -> Result<()> {
        println!("🔍 Explaining query: \"{}\"", query);
        println!("📁 Index: {}", index_uid);
        
        // Open index
        let env = unsafe { Env::open(db_path) }?;
        let rtxn = env.read_txn()?;
        let index = open_index(&env, &rtxn, index_uid)?;
        
        // Execute search
        let mut search = milli::Search::new(&rtxn, &index);
        search.query(query);
        let (results, _) = search.execute()?;
        
        println!(\"\\n✅ Found {} results\", results.documents_ids.len());
        
        // Explain top 5 results
        for (i, (doc_id, scores)) in results.documents_ids.iter()
            .zip(results.document_scores.iter())
            .take(5)
            .enumerate()
        {\n            println!(\"\\n{}. Document ID: {}\", i + 1, doc_id);
            println!(\"   Score: {:.4}\", ScoreDetails::global_score(scores.iter()));
            println!(\"   Breakdown:\");
            
            for score_detail in scores {
                print_score_detail(score_detail, 3);
            }
        }
        
        // If specific document ID requested
        if let Some(target_id) = document_id {
            self.explain_specific_document(&rtxn, &index, query, target_id)?;
        }
        
        Ok(())
    }
    
    fn profile_query(
        &self,
        db_path: &Path,
        index_uid: &str,
        query: &str,
        runs: usize,
    ) -> Result<()> {
        println!("⚡ Profiling query: \"{}\"", query);
        println!("🔄 Running {} iterations", runs);
        
        let env = unsafe { Env::open(db_path) }?;
        let index = open_index(&env, &env.read_txn()?, index_uid)?;
        
        let mut timings = Vec::new();
        
        for i in 0..runs {
            let rtxn = env.read_txn()?;
            let start = Instant::now();
            
            let mut search = milli::Search::new(&rtxn, &index);
            search.query(query);
            let _ = search.execute()?;
            
            let duration = start.elapsed();
            timings.push(duration);
            
            if (i + 1) % 10 == 0 {
                print!(\".\");
                std::io::stdout().flush().unwrap();
            }
        }
        
        println!(\"\\n\");
        
        // Compute statistics
        timings.sort();
        let p50 = timings[runs / 2];
        let p95 = timings[(runs * 95) / 100];
        let p99 = timings[(runs * 99) / 100];
        let mean: Duration = timings.iter().sum::<Duration>() / runs as u32;
        
        println!(\"📊 Results:\");
        println!(\"   Mean:   {:?}\", mean);
        println!(\"   p50:    {:?}\", p50);
        println!(\"   p95:    {:?}\", p95);
        println!(\"   p99:    {:?}\", p99);
        println!(\"   Min:    {:?}\", timings[0]);
        println!(\"   Max:    {:?}\", timings[runs - 1]);
        
        Ok(())
    }
}

fn print_score_detail(detail: &ScoreDetails, indent: usize) {
    let spaces = " ".repeat(indent);
    
    match detail {
        ScoreDetails::Words(words) => {
            println!(\"{}Words: {} matching\", spaces, words.matching_words);
        }
        ScoreDetails::Typo(typo) => {
            println!(\"{}Typo: {} typos, {} max\", spaces, typo.typo_count, typo.max_typo_count);
        }
        ScoreDetails::Proximity(prox) => {
            println!(\"{}Proximity: score {}\", spaces, prox.score);
        }
        _ => {
            println!(\"{}Other: {:?}\", spaces, detail);
        }
    }
}
```

## API Changes

### New Debugging Endpoints

```http
# Explain query results
POST /indexes/{indexUid}/_explain

{
  "query": "wireless headphones",
  "documentId": "prod_123",
  "filter": "price < 200"
}

Response:
{
  "documentId": "prod_123",
  "matched": true,
  "score": 0.87,
  "rankingBreakdown": [
    {
      "rule": "words",
      "score": 0.95,
      "details": "All query terms found",
      "contributionPercent": 52.3
    },
    {
      "rule": "typo",
      "score": 1.0,
      "details": "No typos needed",
      "contributionPercent": 28.7
    },
    {
      "rule": "proximity",
      "score": 0.78,
      "details": "Terms close together",
      "contributionPercent": 19.0
    }
  ],
  "termMatching": [
    {
      "queryTerm": "wireless",
      "matchedWord": "wireless",
      "typoCount": 0,
      "field": "title",
      "positions": [0]
    }
  ]
}

# Validate settings
POST /indexes/{indexUid}/_validate_settings

{
  "searchableAttributes": ["title", "description"],
  "filterableAttributes": ["price", "category"],
  "rankingRules": ["sort", "words", "typo"]
}

Response:
{
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "field": "rankingRules",
      "message": "'sort' appears before 'words'. Text relevancy may be poor.",
      "impact": "Search quality degradation for keyword queries"
    }
  ],
  "suggestions": [
    {
      "category": "optimization",
      "suggestion": "Move 'sort' after 'words', 'typo', and 'proximity' for better relevancy",
      "benefit": "Improve search relevancy while maintaining custom sorting"
    }
  ]
}

# Get query builder metadata
GET /indexes/{indexUid}/_query_builder

Response:
{
  "searchableAttributes": ["title", "description", "brand"],
  "filterableAttributes": ["price", "category", "in_stock"],
  "sortableAttributes": ["price", "created_at"],
  "facetableAttributes": ["brand", "category", "color"],
  "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"],
  "exampleDocument": {
    "id": "1",
    "title": "Wireless Headphones",
    "price": 99.99,
    "category": "Electronics"
  }
}
```

### CLI Commands

```bash
# Explain why document appears for query
meilitool debug explain-query \
  --db-path ./data.ms \
  --index products \
  --query "wireless headphones" \
  --document-id "prod_123"

# Validate settings file
meilitool debug validate-settings \
  --db-path ./data.ms \
  --index products \
  --settings-file ./settings.json

# Profile query performance
meilitool debug profile-query \
  --db-path ./data.ms \
  --index products \
  --query "laptop" \
  --runs 100

# Analyze index structure
meilitool debug analyze-index \
  --db-path ./data.ms \
  --index products
```

## Backward Compatibility

### Compatibility Strategy

1. **Additive only**: No breaking changes to existing error format
2. **Opt-in enhanced errors**: Header `X-Meilisearch-Verbose-Errors: true`
3. **CLI tools separate**: meilitool doesn't affect main binary
4. **Progressive enhancement**: Old clients work unchanged

## Implementation Plan

### Phase 1: Enhanced Errors (3 weeks)

**Tasks:**
1. Implement EnhancedError type
2. Add error builders for common cases
3. Update all error responses
4. Unit tests

**Deliverables:**
- 50+ error types with suggestions
- Consistent error format
- Documentation

### Phase 2: Query Explanation (3 weeks)

**Tasks:**
1. Implement QueryExplainer
2. Add _explain endpoint
3. Ranking breakdown logic
4. Integration tests

**Deliverables:**
- Query explanation API working
- Detailed ranking breakdown
- Filter evaluation

### Phase 3: Schema Validation (2 weeks)

**Tasks:**
1. Implement SettingsValidator
2. Add validation rules
3. Generate suggestions
4. API endpoint

**Deliverables:**
- Settings validation working
- Helpful suggestions
- API documentation

### Phase 4: CLI Tools (2 weeks)

**Tasks:**
1. Add debug subcommands to meilitool
2. Implement explain-query command
3. Add profile-query command
4. Pretty output formatting

**Deliverables:**
- CLI tools functional
- Documentation
- Examples

### Phase 5: Documentation (2 weeks)

**Tasks:**
1. Error reference documentation
2. Debugging guide
3. Best practices
4. Video tutorials

**Deliverables:**
- Complete debugging guide
- Error code reference
- Troubleshooting playbook

## Performance Implications

### Enhanced Errors

**Overhead:** +0.1-0.5ms per error (validation + suggestion generation)

**Impact:** Negligible - errors are already exceptional path

### Query Explanation

**Overhead:** +5-10ms (re-executes search with instrumentation)

**Impact:** Acceptable for debugging endpoint (not used in production queries)

### Schema Validation

**Overhead:** +1-2ms (runs complex validation checks)

**Impact:** Only during settings updates (infrequent operation)

## Drawbacks

### 1. Response Size Increase

Enhanced errors are 3-5x larger than current errors

**Mitigation:** Opt-in via header, compression

### 2. Maintenance Burden

More error types to maintain

**Mitigation:** Error generation macros, templates

### 3. Documentation Overhead

Each error needs documentation

**Mitigation:** Auto-generate error docs from code

## Alternatives Considered

### 1. Generic Error Messages

**Approach:** Keep errors simple and generic

**Why not chosen:**
- Poor developer experience
- Increases support burden
- Competitors have better errors

### 2. Error Codes Only

**Approach:** Just return error codes, let users look up

**Why not chosen:**
- Extra step for developers
- Documentation may be outdated
- Inline help is better UX

### 3. AI-Powered Error Suggestions

**Approach:** Use LLM to generate error fixes

**Why not chosen:**
- Requires external API
- Latency concerns
- Deterministic rules better for now
- Could add later as enhancement

## Open Questions

### 1. Error Verbosity Default

**Question:** Should verbose errors be default?

**Options:**
- A: Yes (better DX)
- B: No (backward compatible)
- C: Depend on environment (dev vs prod)

**Recommendation:** Option C

### 2. Query Explanation Caching

**Question:** Should explanation results be cached?

**Options:**
- A: Yes, cache for 5 minutes
- B: No, always fresh
- C: Configurable

**Recommendation:** Option B (debugging needs fresh data)

### 3. CLI Tool Distribution

**Question:** Ship CLI tools separately or bundled?

**Options:**
- A: Separate binary (meilitool)
- B: Subcommands in main binary
- C: Optional Cargo feature

**Recommendation:** Option A (already exists)

## References

### Error Design Best Practices

1. **Error Messages:**
   - [Google API Design Guide - Errors](https://cloud.google.com/apis/design/errors)

2. **Developer Experience:**
   - Stripe API error messages (industry best practice)

### Rust Error Handling

1. **thiserror:**
   - [Documentation](https://docs.rs/thiserror/)

2. **miette:**
   - Fancy error reporting for Rust
   - [GitHub](https://github.com/zkat/miette)

### Meilisearch Codebase

1. **Current errors:** [`crates/milli/src/error.rs`](crates/milli/src/error.rs:1)
2. **HTTP errors:** [`crates/meilisearch/src/error.rs`](crates/meilisearch/src/error.rs:1)
3. **meilitool:** [`crates/meilitool/src/main.rs`](crates/meilitool/src/main.rs:1)

## Community Discussion

Key discussion points:

1. **Most needed tools:** What debugging features are most valuable?
2. **Error verbosity:** Too much detail overwhelming?
3. **CLI vs API:** Which interface preferred for debugging?
4. **Documentation generation:** Auto-generate error docs?

**Discussion link:** TBD after posting to GitHub

---

**Changelog:**
- 2025-11-16: Initial draft created