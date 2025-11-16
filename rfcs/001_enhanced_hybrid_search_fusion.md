# RFC 001: Enhanced Hybrid Search with Advanced Fusion Strategies

**Status:** Draft  
**Created:** 2025-11-10  
**Authors:** Meilisearch Community  
**Tracking Issue:** TBD

---

## Summary

This RFC proposes significant enhancements to Meilisearch's hybrid search capabilities by implementing advanced fusion strategies including Reciprocal Rank Fusion (RRF), adaptive semantic ratios, and learned fusion weights. These improvements will provide better relevance, more intelligent query-adaptive blending, and enhanced control over the balance between keyword and vector search results.

## Motivation

### Current Limitations

Meilisearch's current hybrid search implementation ([`crates/milli/src/search/hybrid.rs`](crates/milli/src/search/hybrid.rs:1)) uses a simple weighted score fusion approach:

```rust
// Current implementation: weighted sum with fixed ratio
let keyword_results = ScoreWithRatioResult::new(keyword_results, 1.0 - semantic_ratio);
let vector_results = ScoreWithRatioResult::new(vector_results, semantic_ratio);
```

**Key limitations:**

1. **Score Distribution Sensitivity**: The weighted sum approach requires careful score normalization across different embedding models
2. **Fixed Ratio**: The `semantic_ratio` parameter is query-independent, requiring manual tuning per use case
3. **No Diversity Handling**: Results can be dominated by one retrieval method, reducing coverage
4. **Limited Query Intelligence**: System doesn't adapt to query characteristics (e.g., navigational vs. exploratory)

### Real-World Use Cases

**E-Commerce Search:**
- User searches "red nike running shoes size 10"
- Keyword search: Excellent for exact matches (size, brand, color)
- Vector search: Captures semantic similarity ("athletic footwear", "sneakers")
- **Desired behavior**: Dynamically increase keyword weight for specific queries with exact attributes

**Content Discovery:**
- User searches "articles about climate change impacts"
- Keyword search: Matches "climate change" terms
- Vector search: Finds semantically related content ("global warming", "environmental effects")
- **Desired behavior**: Higher semantic weight for exploratory queries

**Multi-Language Search:**
- User searches in English for documents in multiple languages
- Vector search: Natural cross-lingual matching
- Keyword search: Limited to English documents
- **Desired behavior**: Adaptively weight based on language detection

## Technical Design

### 1. Reciprocal Rank Fusion (RRF)

#### Overview

RRF is a rank-based fusion method that combines results from multiple retrieval systems without requiring score normalization:

```
RRF_score(d) = Σ (1 / (k + rank_i(d)))
```

Where:
- `d` is a document
- `k` is a constant (typically 60)
- `rank_i(d)` is the rank of document `d` in result set `i`

#### Implementation

**New file:** `crates/milli/src/search/fusion/rrf.rs`

```rust
use roaring::RoaringBitmap;
use std::collections::HashMap;

/// Reciprocal Rank Fusion scoring
pub struct RRFScorer {
    /// The k parameter controls rank sensitivity
    k: f64,
}

impl RRFScorer {
    pub fn new(k: f64) -> Self {
        Self { k }
    }
    
    /// Compute RRF scores for documents appearing in multiple result lists
    pub fn score(
        &self,
        keyword_results: &[(DocumentId, Vec<ScoreDetails>)],
        vector_results: &[(DocumentId, Vec<ScoreDetails>)],
        weights: &FusionWeights,
    ) -> Vec<(DocumentId, f64)> {
        let mut rrf_scores: HashMap<DocumentId, f64> = HashMap::new();
        
        // Score keyword results
        for (rank, (docid, _scores)) in keyword_results.iter().enumerate() {
            let rrf_contribution = weights.keyword_weight / (self.k + (rank as f64) + 1.0);
            *rrf_scores.entry(*docid).or_insert(0.0) += rrf_contribution;
        }
        
        // Score vector results
        for (rank, (docid, _scores)) in vector_results.iter().enumerate() {
            let rrf_contribution = weights.semantic_weight / (self.k + (rank as f64) + 1.0);
            *rrf_scores.entry(*docid).or_insert(0.0) += rrf_contribution;
        }
        
        // Sort by RRF score descending
        let mut results: Vec<_> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        results
    }
}

#[derive(Debug, Clone)]
pub struct FusionWeights {
    pub keyword_weight: f64,
    pub semantic_weight: f64,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            keyword_weight: 1.0,
            semantic_weight: 1.0,
        }
    }
}
```

**Modified:** `crates/milli/src/search/hybrid.rs`

```rust
use crate::search::fusion::rrf::RRFScorer;

pub enum FusionStrategy {
    /// Weighted score combination (current implementation)
    WeightedSum { semantic_ratio: f32 },
    /// Reciprocal Rank Fusion
    RRF { k: f64, weights: FusionWeights },
    /// Adaptive fusion (decides strategy per query)
    Adaptive { config: AdaptiveConfig },
}

impl Search<'_> {
    pub fn fusion_strategy(&mut self, strategy: FusionStrategy) -> &mut Search<'_> {
        self.fusion_strategy = Some(strategy);
        self
    }
    
    #[tracing::instrument(level = "trace", skip_all, target = "search::hybrid")]
    pub fn execute_hybrid(&self, strategy: FusionStrategy) -> Result<(SearchResult, Option<u32>)> {
        // ... existing code for executing keyword and vector searches ...
        
        let merged_results = match strategy {
            FusionStrategy::WeightedSum { semantic_ratio } => {
                // Current implementation
                ScoreWithRatioResult::merge(
                    vector_results,
                    keyword_results,
                    self.offset,
                    self.limit,
                    search.distinct.as_deref(),
                    search.index,
                    search.rtxn,
                )?
            }
            FusionStrategy::RRF { k, weights } => {
                let scorer = RRFScorer::new(k);
                self.merge_with_rrf(
                    &scorer,
                    keyword_results,
                    vector_results,
                    &weights,
                )?
            }
            FusionStrategy::Adaptive { config } => {
                let strategy = self.determine_fusion_strategy(&config, &keyword_results, &vector_results)?;
                // Recursively apply determined strategy
                self.execute_hybrid(strategy)?
            }
        };
        
        Ok(merged_results)
    }
}
```

### 2. Adaptive Semantic Ratio

#### Query Analysis

**New file:** `crates/milli/src/search/fusion/adaptive.rs`

```rust
use charabia::Tokenize;

/// Analyzes query characteristics to determine optimal fusion strategy
pub struct QueryAnalyzer {
    config: AdaptiveConfig,
}

#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Keywords indicating navigational intent (prefer keyword search)
    pub navigational_indicators: Vec<String>,
    /// Keywords indicating exploratory intent (prefer semantic search)
    pub exploratory_indicators: Vec<String>,
    /// Threshold for query specificity (based on unique terms)
    pub specificity_threshold: usize,
    /// Default semantic ratio when no clear signal
    pub default_semantic_ratio: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            navigational_indicators: vec![
                "where".to_string(),
                "how to".to_string(),
                "buy".to_string(),
                "price".to_string(),
                "size".to_string(),
                "color".to_string(),
            ],
            exploratory_indicators: vec![
                "similar".to_string(),
                "like".to_string(),
                "about".to_string(),
                "related".to_string(),
                "concept".to_string(),
            ],
            specificity_threshold: 5,
            default_semantic_ratio: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct QueryFeatures {
    pub is_navigational: bool,
    pub is_exploratory: bool,
    pub has_numbers: bool,
    pub has_exact_phrases: bool,
    pub unique_term_count: usize,
    pub avg_term_length: f32,
}

impl QueryAnalyzer {
    pub fn new(config: AdaptiveConfig) -> Self {
        Self { config }
    }
    
    /// Extract features from query text
    pub fn analyze_query(&self, query: &str) -> QueryFeatures {
        let query_lower = query.to_lowercase();
        
        // Tokenize query
        let tokens: Vec<_> = query.tokenize().collect();
        let unique_term_count = tokens.iter().map(|t| t.lemma()).collect::<std::collections::HashSet<_>>().len();
        
        let avg_term_length = if !tokens.is_empty() {
            tokens.iter().map(|t| t.lemma().len()).sum::<usize>() as f32 / tokens.len() as f32
        } else {
            0.0
        };
        
        // Detect navigational intent
        let is_navigational = self.config.navigational_indicators.iter()
            .any(|indicator| query_lower.contains(indicator));
        
        // Detect exploratory intent
        let is_exploratory = self.config.exploratory_indicators.iter()
            .any(|indicator| query_lower.contains(indicator));
        
        // Detect numbers (product codes, sizes, etc.)
        let has_numbers = query.chars().any(|c| c.is_numeric());
        
        // Detect exact phrases (quoted strings)
        let has_exact_phrases = query.contains('"');
        
        QueryFeatures {
            is_navigational,
            is_exploratory,
            has_numbers,
            has_exact_phrases,
            unique_term_count,
            avg_term_length,
        }
    }
    
    /// Determine optimal semantic ratio based on query features
    pub fn compute_semantic_ratio(&self, features: &QueryFeatures) -> f32 {
        let mut semantic_ratio = self.config.default_semantic_ratio;
        
        // Increase keyword weight for navigational queries
        if features.is_navigational {
            semantic_ratio -= 0.2;
        }
        
        // Increase semantic weight for exploratory queries
        if features.is_exploratory {
            semantic_ratio += 0.2;
        }
        
        // Queries with numbers often need exact matching
        if features.has_numbers {
            semantic_ratio -= 0.15;
        }
        
        // Exact phrases indicate desire for precision
        if features.has_exact_phrases {
            semantic_ratio -= 0.15;
        }
        
        // Long, specific queries benefit from keyword precision
        if features.unique_term_count >= self.config.specificity_threshold {
            semantic_ratio -= 0.1;
        }
        
        // Short queries benefit from semantic expansion
        if features.unique_term_count <= 2 {
            semantic_ratio += 0.15;
        }
        
        // Clamp to valid range
        semantic_ratio.max(0.0).min(1.0)
    }
}
```

#### Integration

```rust
impl Search<'_> {
    fn determine_fusion_strategy(
        &self,
        config: &AdaptiveConfig,
        keyword_results: &SearchResult,
        vector_results: &SearchResult,
    ) -> Result<FusionStrategy> {
        let analyzer = QueryAnalyzer::new(config.clone());
        
        // Analyze query
        let features = analyzer.analyze_query(
            self.query.as_deref().unwrap_or("")
        );
        
        // Compute adaptive semantic ratio
        let semantic_ratio = analyzer.compute_semantic_ratio(&features);
        
        tracing::debug!(
            "Adaptive fusion: query_features={:?}, semantic_ratio={}",
            features,
            semantic_ratio
        );
        
        // Use RRF for balanced queries, weighted sum otherwise
        if (0.4..=0.6).contains(&semantic_ratio) {
            Ok(FusionStrategy::RRF {
                k: 60.0,
                weights: FusionWeights {
                    keyword_weight: 1.0 - semantic_ratio as f64,
                    semantic_weight: semantic_ratio as f64,
                },
            })
        } else {
            Ok(FusionStrategy::WeightedSum { semantic_ratio })
        }
    }
}
```

### 3. Learned Fusion Weights

#### Click-Through Rate (CTR) Based Learning

**New file:** `crates/milli/src/search/fusion/learning.rs`

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Stores learned weights based on query patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeights {
    /// Map from query pattern hash to learned weights
    patterns: HashMap<u64, FusionWeights>,
    /// Global fallback weights
    global_weights: FusionWeights,
}

impl LearnedWeights {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            global_weights: FusionWeights::default(),
        }
    }
    
    /// Update weights based on user interaction signals
    pub fn update_from_ctr(
        &mut self,
        query: &str,
        keyword_clicks: usize,
        semantic_clicks: usize,
    ) {
        let pattern_hash = self.hash_query_pattern(query);
        
        let total_clicks = keyword_clicks + semantic_clicks;
        if total_clicks == 0 {
            return;
        }
        
        let semantic_preference = semantic_clicks as f64 / total_clicks as f64;
        
        // Exponential moving average for weight updates
        let alpha = 0.1; // Learning rate
        let weights = self.patterns.entry(pattern_hash)
            .or_insert_with(FusionWeights::default);
        
        weights.semantic_weight = alpha * semantic_preference 
            + (1.0 - alpha) * weights.semantic_weight;
        weights.keyword_weight = 1.0 - weights.semantic_weight;
    }
    
    /// Retrieve weights for a query
    pub fn get_weights(&self, query: &str) -> FusionWeights {
        let pattern_hash = self.hash_query_pattern(query);
        self.patterns.get(&pattern_hash)
            .cloned()
            .unwrap_or_else(|| self.global_weights.clone())
    }
    
    fn hash_query_pattern(&self, query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Normalize and extract pattern
        let normalized = query.to_lowercase();
        let tokens: Vec<_> = normalized.split_whitespace().collect();
        
        // Create pattern based on query structure
        let pattern = if tokens.len() <= 2 {
            "short_query".to_string()
        } else if tokens.iter().any(|t| t.chars().any(|c| c.is_numeric())) {
            "numeric_query".to_string()
        } else {
            "standard_query".to_string()
        };
        
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        hasher.finish()
    }
}
```

## API Changes

### Search Request

**Current API:**
```json
{
  "q": "red nike shoes",
  "hybrid": {
    "semanticRatio": 0.5,
    "embedder": "default"
  }
}
```

**Proposed API:**
```json
{
  "q": "red nike shoes",
  "hybrid": {
    "embedder": "default",
    "fusion": {
      "strategy": "adaptive",
      "config": {
        "navigationalIndicators": ["buy", "price", "size"],
        "exploratoryIndicators": ["similar", "like", "about"],
        "defaultSemanticRatio": 0.5
      }
    }
  }
}
```

**Alternative strategies:**

```json
{
  "fusion": {
    "strategy": "rrf",
    "k": 60,
    "weights": {
      "keyword": 1.0,
      "semantic": 1.0
    }
  }
}
```

```json
{
  "fusion": {
    "strategy": "weighted",
    "semanticRatio": 0.7
  }
}
```

### Settings Update

**New index settings:**

```json
{
  "hybridSearch": {
    "defaultFusionStrategy": "adaptive",
    "adaptiveConfig": {
      "navigationalIndicators": ["where", "how", "buy"],
      "exploratoryIndicators": ["similar", "related", "about"],
      "specificityThreshold": 5,
      "defaultSemanticRatio": 0.5
    },
    "rrfConfig": {
      "k": 60
    },
    "enableLearning": false
  }
}
```

## Backward Compatibility

### Compatibility Strategy

1. **Default behavior unchanged**: Existing queries use weighted sum fusion
2. **Opt-in new features**: RRF and adaptive fusion require explicit configuration
3. **API versioning**: New `fusion` parameter coexists with legacy `semanticRatio`
4. **Migration path**:
   - v1.12: Current weighted sum only
   - v1.13: Add RRF and adaptive (opt-in)
   - v1.14: Make adaptive default for new indexes
   - v2.0: Deprecate standalone `semanticRatio`

### Deprecation Timeline

**Phase 1 (v1.13):** Introduce new API, mark `semanticRatio` as soft-deprecated
```rust
#[deprecated(since = "1.13.0", note = "Use fusion.strategy instead")]
pub semantic_ratio: Option<f32>,
```

**Phase 2 (v1.15):** Warning logs when using deprecated API

**Phase 3 (v2.0):** Remove deprecated API

## Implementation Plan

### Phase 1: Core RRF Implementation (2 weeks)

**Tasks:**
1. Create `crates/milli/src/search/fusion/` module
2. Implement `RRFScorer` with unit tests
3. Add `FusionStrategy` enum to `Search` struct
4. Implement RRF merge logic
5. Add benchmarks comparing RRF vs weighted sum

**Deliverables:**
- Working RRF implementation
- Unit tests with 90%+ coverage
- Performance benchmarks

### Phase 2: Adaptive Fusion (3 weeks)

**Tasks:**
1. Implement `QueryAnalyzer` with feature extraction
2. Add configurable adaptive parameters
3. Create integration tests with real queries
4. Document query patterns and behavior

**Deliverables:**
- Adaptive fusion working for common query patterns
- Configuration documentation
- A/B test framework for fusion strategies

### Phase 3: API Integration (2 weeks)

**Tasks:**
1. Add HTTP API endpoints for fusion configuration
2. Update search request parsing
3. Add index settings for fusion strategy
4. Update OpenAPI specifications

**Deliverables:**
- Complete API implementation
- Updated documentation
- API integration tests

### Phase 4: Learned Weights (4 weeks)

**Tasks:**
1. Design interaction tracking schema
2. Implement CTR-based weight updates
3. Add weight persistence to index settings
4. Create weight visualization tools

**Deliverables:**
- Learning system integrated
- Weight persistence working
- Monitoring dashboard

### Phase 5: Documentation & Polish (2 weeks)

**Tasks:**
1. Write comprehensive user guide
2. Create migration guide from weighted sum
3. Add practical examples for each fusion strategy
4. Performance tuning and optimization

**Deliverables:**
- Complete documentation
- Migration tools
- Performance optimization report

## Performance Implications

### Computational Complexity

**Current weighted sum:**
- Time: O(n log n) for merge sort
- Space: O(n) for result storage

**RRF:**
- Time: O(n log n) for sorting by RRF score
- Space: O(n) for score HashMap
- **Impact**: Comparable to weighted sum, negligible overhead

**Adaptive fusion:**
- Time: O(m) for query analysis (m = query tokens)
- Additional: O(n log n) for chosen strategy
- **Impact**: < 1ms overhead for query analysis

### Memory Overhead

- **RRF**: +8 bytes per document (f64 score)
- **Adaptive config**: ~1KB per index
- **Learned weights**: ~100 bytes per query pattern

**Total overhead**: < 0.1% for typical indexes

### Benchmarking Strategy

**Test scenarios:**
1. 1M documents, 1000 queries
2. Compare strategies: weighted sum, RRF (k=60), adaptive
3. Metrics: latency (p50, p95, p99), relevance (NDCG@10)

**Expected results:**
- RRF latency: +2-5ms vs weighted sum
- Adaptive overhead: +0.5-1ms
- Relevance improvement: +5-15% NDCG for exploratory queries

## Open Questions

### 1. Default Fusion Strategy

**Question:** Should adaptive fusion become the default for all new indexes?

**Options:**
- A: Keep weighted sum as default (safest, backward compatible)
- B: Make adaptive default for new indexes only
- C: Make adaptive default universally in v2.0

**Recommendation:** Option B - Balance innovation with stability

### 2. Learning System Scope

**Question:** Should learned weights be:
- A: Per-index (isolated learning)
- B: Global across all indexes (shared patterns)
- C: Configurable per deployment

**Recommendation:** Option C with per-index default

### 3. RRF Parameter Exposure

**Question:** Should the RRF `k` parameter be:
- A: Fixed at k=60 (research standard)
- B: Configurable per-query
- C: Configurable per-index with per-query override

**Recommendation:** Option C for maximum flexibility

### 4. Query Feature Engineering

**Question:** What additional query features should inform adaptive fusion?
- Query language detection
- Named entity recognition
- Query category classification
- User context (location, history)

**Discussion needed:** Community input on priority

## Alternatives Considered

### 1. CombSUM Instead of RRF

**CombSUM:** Simply sum normalized scores
```
CombSUM(d) = Σ normalized_score_i(d)
```

**Why not chosen:**
- Requires careful score normalization
- Less robust to score distribution differences
- RRF is proven in research literature

### 2. Neural Fusion Models

**Approach:** Use ML model to learn optimal fusion

**Why not chosen:**
- Requires training data and infrastructure
- Adds deployment complexity
- Difficult to explain/debug
- Future RFC may explore this

### 3. Round-Robin Interleaving

**Approach:** Alternate between keyword and semantic results

**Why not chosen:**
- Too simplistic, ignores relevance scores
- Poor user experience for biased result sets
- No research supporting effectiveness

## References

### Research Papers

1. **Reciprocal Rank Fusion:**
   - Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). "Reciprocal rank fusion outperforms condorcet and individual rank learning methods." *SIGIR 2009*.
   - [Paper Link](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

2. **Hybrid Search Strategies:**
   - Zamani, H., Dehghani, M., Croft, W. B., et al. (2018). "From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing." *CIKM 2018*.

3. **Query Intent Classification:**
   - Broder, A. (2002). "A taxonomy of web search." *ACM SIGIR Forum*.

### Existing Implementations

1. **Elasticsearch:** Uses `rrf` rank feature
   - [Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)

2. **Vespa:** Supports multiple fusion strategies
   - [Ranking Documentation](https://docs.vespa.ai/en/ranking.html)

### Meilisearch Codebase

1. **Current hybrid search:** [`crates/milli/src/search/hybrid.rs`](crates/milli/src/search/hybrid.rs:1)
2. **Score details:** [`crates/milli/src/score_details.rs`](crates/milli/src/score_details.rs:1)
3. **Vector store:** [`crates/milli/src/vector/store.rs`](crates/milli/src/vector/store.rs:1)

## Community Discussion

This RFC will be published to Meilisearch GitHub Discussions for community feedback. Key discussion points:

1. **Use case validation:** Does this solve real-world problems?
2. **API design:** Is the proposed API intuitive?
3. **Performance concerns:** Are the tradeoffs acceptable?
4. **Alternative approaches:** What are we missing?

**Discussion link:** TBD after posting to GitHub

---

**Changelog:**
- 2025-11-10: Initial draft created