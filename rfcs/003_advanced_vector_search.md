# RFC 003: Advanced Vector Search Capabilities

**Status:** Draft  
**Created:** 2025-11-10  
**Authors:** Meilisearch Community  
**Tracking Issue:** TBD

---

## Summary

This RFC proposes significant enhancements to Meilisearch's vector search capabilities, including advanced indexing algorithms (IVF-HNSW), Product Quantization for memory efficiency, multi-vector document support, and improved ANN (Approximate Nearest Neighbor) search quality. These improvements will enable larger-scale vector deployments while maintaining Meilisearch's performance characteristics.

## Motivation

### Current Limitations

Meilisearch's current vector search implementation ([`crates/milli/src/vector/store.rs`](crates/milli/src/vector/store.rs:1)) uses HNSW (Hierarchical Navigable Small World) with two backends:

1. **Arroy**: Stable backend with full-precision vectors
2. **Hannoy**: Experimental backend with binary quantization

**Current parameters** (from [`store.rs`](crates/milli/src/vector/store.rs:13)):
```rust
const HANNOY_EF_CONSTRUCTION: usize = 125;
const HANNOY_M: usize = 16;
const HANNOY_M0: usize = 32;
```

**Key limitations:**

1. **Binary Quantization Only**: 32× memory reduction but significant accuracy loss
2. **Fixed HNSW Parameters**: No runtime tuning for recall/latency trade-offs
3. **Single Vector Per Document**: Cannot handle multi-aspect document embeddings
4. **No IVF Support**: Limited options for very large vector datasets (>100M vectors)
5. **Hardcoded ef_search**: `(limit * 10).max(100)` may be suboptimal ([`store.rs`](crates/milli/src/vector/store.rs:971))

### Real-World Use Cases

**Large-Scale Product Catalog:**
- 50M+ products with embeddings
- Current binary quantization: accuracy degradation unacceptable
- **Need:** Product Quantization for better accuracy/memory trade-off

**Multi-Modal Documents:**
- Documents with title, body, and metadata embeddings
- Currently: Must choose one embedding or concatenate
- **Need:** Multi-vector support with per-aspect weighting

**High-Precision Applications:**
- Medical/legal document search requiring >95% recall
- Binary quantization insufficient
- **Need:** Configurable quantization levels (PQ8, PQ16, etc.)

**Dynamic Query Patterns:**
- Some queries need high recall (exploratory)
- Others prioritize latency (known-item search)
- **Need:** Runtime-configurable ef_search parameter

## Technical Design

### 1. Product Quantization (PQ)

#### Overview

Product Quantization compresses vectors by dividing them into subvectors and clustering each subspace independently:

```
Original vector (768D): [v1, v2, ..., v768]
Divide into 96 subvectors (8D each): 
  - Sub1: [v1..v8]
  - Sub2: [v9..v16]
  - ...
  - Sub96: [v761..v768]

Quantize each subvector to nearest centroid (8-bit codes)
Compressed: [c1, c2, ..., c96] = 96 bytes (vs 3072 bytes for float32)
```

**Compression ratio:** 32× with PQ8 (8-bit codes per subvector)

#### Implementation

**New file:** `crates/milli/src/vector/quantization/pq.rs`

```rust
use std::sync::Arc;

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of subvectors (must divide dimension evenly)
    pub num_subvectors: usize,
    /// Bits per code (typically 8 or 16)
    pub bits_per_code: usize,
    /// Number of training iterations for k-means
    pub training_iterations: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subvectors: 96,  // For 768D vectors
            bits_per_code: 8,     // 256 centroids per subspace
            training_iterations: 20,
        }
    }
}

pub struct ProductQuantizer {
    config: PQConfig,
    /// Codebooks for each subvector (num_subvectors × 2^bits_per_code × subvector_dim)
    codebooks: Vec<Vec<Vec<f32>>>,
    dimension: usize,
    subvector_dim: usize,
}

impl ProductQuantizer {
    /// Train PQ codebooks on sample vectors
    pub fn train(
        config: PQConfig,
        training_vectors: &[Vec<f32>],
    ) -> Result<Self, QuantizationError> {
        let dimension = training_vectors.first()
            .ok_or(QuantizationError::EmptyTrainingSet)?
            .len();
        
        if dimension % config.num_subvectors != 0 {
            return Err(QuantizationError::DimensionMismatch);
        }
        
        let subvector_dim = dimension / config.num_subvectors;
        let num_centroids = 1 << config.bits_per_code; // 2^bits_per_code
        
        let mut codebooks = Vec::with_capacity(config.num_subvectors);
        
        // Train codebook for each subspace
        for subspace_idx in 0..config.num_subvectors {
            let start_dim = subspace_idx * subvector_dim;
            let end_dim = start_dim + subvector_dim;
            
            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = training_vectors.iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();
            
            // Run k-means clustering
            let centroids = kmeans_clustering(
                &subvectors,
                num_centroids,
                config.training_iterations,
            )?;
            
            codebooks.push(centroids);
        }
        
        Ok(Self {
            config,
            codebooks,
            dimension,
            subvector_dim,
        })
    }
    
    /// Encode a vector using PQ
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.num_subvectors);
        
        for (subspace_idx, codebook) in self.codebooks.iter().enumerate() {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];
            
            // Find nearest centroid
            let code = find_nearest_centroid(subvector, codebook);
            codes.push(code as u8);
        }
        
        codes
    }
    
    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimension);
        
        for (code, codebook) in codes.iter().zip(self.codebooks.iter()) {
            let centroid = &codebook[*code as usize];
            vector.extend_from_slice(centroid);
        }
        
        vector
    }
    
    /// Asymmetric distance computation (query vector vs PQ codes)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut distance_sq = 0.0;
        
        for (subspace_idx, &code) in codes.iter().enumerate() {
            let start_dim = subspace_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            let centroid = &self.codebooks[subspace_idx][code as usize];
            
            // Squared Euclidean distance for this subspace
            for (q, c) in query_subvector.iter().zip(centroid.iter()) {
                let diff = q - c;
                distance_sq += diff * diff;
            }
        }
        
        distance_sq.sqrt()
    }
}

fn kmeans_clustering(
    vectors: &[Vec<f32>],
    k: usize,
    iterations: usize,
) -> Result<Vec<Vec<f32>>, QuantizationError> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let dim = vectors.first().ok_or(QuantizationError::EmptyTrainingSet)?.len();
    let mut rng = thread_rng();
    
    // Initialize centroids randomly
    let mut centroids: Vec<Vec<f32>> = vectors.choose_multiple(&mut rng, k)
        .map(|v| v.clone())
        .collect();
    
    for _ in 0..iterations {
        // Assignment step
        let mut clusters: Vec<Vec<Vec<f32>>> = vec![Vec::new(); k];
        
        for vector in vectors {
            let nearest_idx = find_nearest_centroid(vector, &centroids);
            clusters[nearest_idx].push(vector.clone());
        }
        
        // Update step
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue; // Keep previous centroid
            }
            
            let mut new_centroid = vec![0.0; dim];
            for vector in cluster {
                for (j, &val) in vector.iter().enumerate() {
                    new_centroid[j] += val;
                }
            }
            
            let cluster_size = cluster.len() as f32;
            for val in &mut new_centroid {
                *val /= cluster_size;
            }
            
            centroids[i] = new_centroid;
        }
    }
    
    Ok(centroids)
}

fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids.iter()
        .enumerate()
        .map(|(i, centroid)| {
            let dist_sq: f32 = vector.iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, dist_sq)
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

### 2. IVF-HNSW Hybrid Index

#### Overview

IVF (Inverted File) provides coarse-grained partitioning before fine-grained HNSW search:

```
1. Partition space into N clusters (IVF)
2. Build HNSW graph within each cluster
3. Search: 
   - Find nearest clusters to query
   - Search HNSW within those clusters
```

**Benefits:**
- Reduces search space from millions to thousands
- Better cache locality
- Enables disk-based indexes

#### Implementation

**New file:** `crates/milli/src/vector/index/ivf_hnsw.rs`

```rust
pub struct IVFHNSWIndex {
    /// Coarse quantizer (cluster centroids)
    coarse_quantizer: Vec<Vec<f32>>,
    /// HNSW index for coarse quantizer
    coarse_hnsw: HNSWIndex,
    /// Fine HNSW indexes per cluster
    cluster_indexes: Vec<HNSWIndex>,
    /// Document assignments to clusters
    assignments: HashMap<DocumentId, usize>,
    /// Configuration
    config: IVFHNSWConfig,
}

#[derive(Debug, Clone)]
pub struct IVFHNSWConfig {
    /// Number of coarse clusters
    pub num_clusters: usize,
    /// Number of clusters to search (nprobe)
    pub nprobe: usize,
    /// HNSW M parameter for fine indexes
    pub hnsw_m: usize,
    /// ef_construction for fine indexes
    pub ef_construction: usize,
}

impl Default for IVFHNSWConfig {
    fn default() -> Self {
        Self {
            num_clusters: 256,      // sqrt(N) rule of thumb
            nprobe: 8,              // Search top 8 clusters
            hnsw_m: 16,
            ef_construction: 125,
        }
    }
}

impl IVFHNSWIndex {
    pub fn build(
        vectors: Vec<(DocumentId, Vec<f32>)>,
        config: IVFHNSWConfig,
    ) -> Result<Self> {
        // 1. Train coarse quantizer
        let training_vecs: Vec<_> = vectors.iter()
            .take(config.num_clusters * 100) // Sample for training
            .map(|(_, v)| v.clone())
            .collect();
        
        let coarse_quantizer = kmeans_clustering(
            &training_vecs,
            config.num_clusters,
            20, // iterations
        )?;
        
        // 2. Build HNSW for coarse quantizer
        let coarse_hnsw = HNSWIndex::build(
            coarse_quantizer.clone(),
            config.hnsw_m,
            config.ef_construction,
        )?;
        
        // 3. Assign documents to clusters
        let mut assignments = HashMap::new();
        let mut clusters: Vec<Vec<(DocumentId, Vec<f32>)>> = 
            vec![Vec::new(); config.num_clusters];
        
        for (doc_id, vector) in vectors {
            let cluster_id = find_nearest_centroid(&vector, &coarse_quantizer);
            assignments.insert(doc_id, cluster_id);
            clusters[cluster_id].push((doc_id, vector));
        }
        
        // 4. Build HNSW for each cluster
        let cluster_indexes: Vec<_> = clusters.into_iter()
            .map(|cluster_vecs| {
                if cluster_vecs.is_empty() {
                    HNSWIndex::empty()
                } else {
                    HNSWIndex::build_from_docs(
                        cluster_vecs,
                        config.hnsw_m,
                        config.ef_construction,
                    )
                }
            })
            .collect::<Result<_>>()?;
        
        Ok(Self {
            coarse_quantizer,
            coarse_hnsw,
            cluster_indexes,
            assignments,
            config,
        })
    }
    
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        ef_search: usize,
    ) -> Result<Vec<(DocumentId, f32)>> {
        // 1. Find nearest clusters
        let nearest_clusters = self.coarse_hnsw.search(
            query,
            self.config.nprobe,
            ef_search,
        )?;
        
        // 2. Search within each cluster
        let mut all_results = Vec::new();
        
        for (cluster_id, _distance) in nearest_clusters {
            if let Some(cluster_index) = self.cluster_indexes.get(cluster_id) {
                let cluster_results = cluster_index.search(
                    query,
                    limit,
                    ef_search,
                )?;
                all_results.extend(cluster_results);
            }
        }
        
        // 3. Sort globally and return top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(limit);
        
        Ok(all_results)
    }
}
```

### 3. Multi-Vector Document Support

#### Overview

Allow multiple embeddings per document to represent different aspects:

```json
{
  "id": "doc123",
  "title": "Introduction to Machine Learning",
  "content": "Machine learning is a subset of AI...",
  "_vectors": {
    "title": [0.1, 0.2, ...],      // Title embedding
    "content": [0.3, 0.4, ...],     // Content embedding
    "combined": [0.15, 0.3, ...]   // Hybrid embedding
  }
}
```

#### Implementation

**Modified:** `crates/milli/src/vector/store.rs`

The current implementation already supports multiple stores via `store_id` parameter (up to 256 stores per embedder), but this needs better API exposure.

```rust
pub struct MultiVectorDocument {
    pub doc_id: DocumentId,
    pub embeddings: HashMap<String, Vec<f32>>,
}

impl VectorStore {
    /// Add multiple embeddings for a document
    pub fn add_multi_vector_document(
        &self,
        wtxn: &mut RwTxn,
        doc: &MultiVectorDocument,
    ) -> Result<()> {
        for (store_name, embedding) in &doc.embeddings {
            let store_id = self.get_or_create_store_id(store_name)?;
            self.add_item_in_store(
                wtxn,
                doc.doc_id,
                store_id,
                embedding,
            )?;
        }
        Ok(())
    }
    
    /// Search across multiple vector stores with weighted fusion
    pub fn search_multi_vector(
        &self,
        rtxn: &RoTxn,
        queries: &HashMap<String, Vec<f32>>,
        weights: &HashMap<String, f32>,
        limit: usize,
        filter: Option<&RoaringBitmap>,
        time_budget: &TimeBudget,
    ) -> Result<Vec<(DocumentId, f32)>> {
        let mut all_results = HashMap::new();
        
        // Search each vector store
        for (store_name, query_vector) in queries {
            let store_id = self.get_store_id(store_name)?;
            let weight = weights.get(store_name).copied().unwrap_or(1.0);
            
            // Construct specialized VectorStore for this store_id
            let results = self.nns_by_vector_in_store(
                rtxn,
                store_id,
                query_vector,
                limit,
                filter,
                time_budget,
            )?;
            
            // Accumulate weighted scores
            for (doc_id, distance) in results {
                let score = (1.0 - distance) * weight;
                *all_results.entry(doc_id).or_insert(0.0) += score;
            }
        }
        
        // Sort by combined score
        let mut ranked: Vec<_> = all_results.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(limit);
        
        Ok(ranked.into_iter()
            .map(|(doc_id, score)| (doc_id, 1.0 - score))
            .collect())
    }
}
```

### 4. Configurable Search Parameters

#### Runtime ef_search Configuration

**Current issue:** `ef_search` is hardcoded ([`store.rs`](crates/milli/src/vector/store.rs:971)):
```rust
searcher.ef_search((limit * 10).max(100));
```

**Proposed:** Make configurable per query

```rust
pub struct VectorSearchConfig {
    /// Exploration factor for HNSW search
    /// Higher = more accurate but slower
    /// Typical range: 50-500
    pub ef_search: Option<usize>,
    
    /// For IVF indexes: number of clusters to search
    pub nprobe: Option<usize>,
    
    /// Minimum similarity threshold (filter low-quality results)
    pub min_similarity: Option<f32>,
    
    /// Maximum distance threshold
    pub max_distance: Option<f32>,
}

impl VectorStore {
    pub fn nns_by_vector_with_config(
        &self,
        rtxn: &RoTxn,
        vector: &[f32],
        limit: usize,
        filter: Option<&RoaringBitmap>,
        config: &VectorSearchConfig,
        time_budget: &TimeBudget,
    ) -> Result<Vec<(ItemId, f32)>> {
        let ef_search = config.ef_search
            .unwrap_or_else(|| (limit * 10).max(100));
        
        // Use configured ef_search
        let mut searcher = reader.nns(limit);
        searcher.ef_search(ef_search);
        
        if let Some(filter) = filter {
            searcher.candidates(filter);
        }
        
        let mut results = searcher.by_vector_with_cancellation(
            rtxn,
            vector,
            || time_budget.exceeded()
        )?.0;
        
        // Apply similarity filtering
        if let Some(min_sim) = config.min_similarity {
            results.retain(|(_, dist)| (1.0 - dist) >= min_sim);
        }
        
        if let Some(max_dist) = config.max_distance {
            results.retain(|(_, dist)| *dist <= max_dist);
        }
        
        Ok(results)
    }
}
```

### 5. Scalar Quantization (SQ)

**Intermediate option between binary and full precision:**

```rust
pub struct ScalarQuantizer {
    /// Quantization bits (4, 8, or 16)
    bits: u8,
    /// Per-dimension min/max for normalization
    bounds: Vec<(f32, f32)>,
}

impl ScalarQuantizer {
    pub fn train(vectors: &[Vec<f32>], bits: u8) -> Self {
        let dim = vectors[0].len();
        let mut bounds = Vec::with_capacity(dim);
        
        // Compute min/max per dimension
        for d in 0..dim {
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            
            for vec in vectors {
                min = min.min(vec[d]);
                max = max.max(vec[d]);
            }
            
            bounds.push((min, max));
        }
        
        Self { bits, bounds }
    }
    
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let max_val = (1 << self.bits) - 1;
        let mut codes = Vec::with_capacity(vector.len());
        
        for (&val, &(min, max)) in vector.iter().zip(self.bounds.iter()) {
            let normalized = (val - min) / (max - min);
            let quantized = (normalized * max_val as f32).round() as u8;
            codes.push(quantized);
        }
        
        codes
    }
    
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let max_val = (1 << self.bits) - 1;
        
        codes.iter()
            .zip(self.bounds.iter())
            .map(|(&code, &(min, max))| {
                let normalized = code as f32 / max_val as f32;
                min + normalized * (max - min)
            })
            .collect()
    }
}
```

### 6. Enhanced Distance Metrics

**New distance functions beyond Cosine:**

```rust
pub enum VectorDistanceMetric {
    /// Cosine similarity (current default)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product similarity
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

pub trait DistanceFunction {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

impl DistanceFunction for VectorDistanceMetric {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_distance(a, b),
            Self::Euclidean => euclidean_distance(a, b),
            Self::DotProduct => dot_product_distance(a, b),
            Self::Manhattan => manhattan_distance(a, b),
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - (dot / (norm_a * norm_b))
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}
```

## API Changes

### Embedder Configuration

**Current API:**
```json
{
  "embedders": {
    "default": {
      "source": "openAi",
      "model": "text-embedding-3-small",
      "dimensions": 1536
    }
  }
}
```

**Proposed API with PQ:**
```json
{
  "embedders": {
    "default": {
      "source": "openAi",
      "model": "text-embedding-3-small",
      "dimensions": 1536,
      "quantization": {
        "type": "product",
        "config": {
          "numSubvectors": 96,
          "bitsPerCode": 8
        }
      }
    }
  }
}
```

### Multi-Vector Documents

```json
{
  "id": "article123",
  "title": "AI Revolution",
  "content": "Artificial intelligence is transforming...",
  "_vectors": {
    "title": {
      "embedder": "default",
      "vector": [0.1, 0.2, ...]
    },
    "content": {
      "embedder": "default",
      "vector": [0.3, 0.4, ...]
    }
  }
}
```

### Search with Multi-Vector

```json
{
  "q": "AI applications",
  "hybrid": {
    "embedder": "default",
    "semanticRatio": 0.7,
    "vectorQueries": {
      "title": {
        "weight": 2.0,
        "vector": [0.5, 0.6, ...]
      },
      "content": {
        "weight": 1.0,
        "vector": [0.7, 0.8, ...]
      }
    }
  },
  "vectorSearchConfig": {
    "efSearch": 200,
    "minSimilarity": 0.7
  }
}
```

### IVF Configuration

```json
{
  "embedders": {
    "large_scale": {
      "source": "openAi",
      "model": "text-embedding-3-large",
      "dimensions": 3072,
      "indexType": "ivf-hnsw",
      "ivfConfig": {
        "numClusters": 1024,
        "nprobe": 16
      }
    }
  }
}
```

## Backward Compatibility

### Compatibility Strategy

1. **Default unchanged**: Binary quantization and HNSW remain default
2. **Opt-in features**: PQ and IVF require explicit configuration
3. **Automatic migration**: Existing vectors work with new search parameters
4. **Deprecation timeline**:
   - v1.13: Introduce PQ and IVF (experimental)
   - v1.14: Mark as stable
   - v1.15: Recommend PQ over binary quantization
   - v2.0: Make PQ default for new embedders

### Migration Path

```bash
# Enable PQ on existing index
curl -X PATCH "http://localhost:7700/indexes/products/settings" \
  -H "Content-Type: application/json" \
  -d '{
    "embedders": {
      "default": {
        "quantization": {
          "type": "product",
          "config": {
            "numSubvectors": 96,
            "bitsPerCode": 8
          }
        }
      }
    }
  }'

# This triggers reindexing with new quantization
```

## Implementation Plan

### Phase 1: Product Quantization (6 weeks)

**Milestone 1: Core PQ Implementation**
- Implement k-means clustering
- PQ encoder/decoder
- Asymmetric distance computation
- **Deliverable:** PQ working for single embedder

**Milestone 2: Integration**
- Add PQ support to VectorStore
- Update indexing pipeline
- Add configuration API
- **Deliverable:** PQ configurable via settings

### Phase 2: IVF-HNSW (6 weeks)

**Milestone 3: IVF Implementation**
- Coarse quantizer training
- Document clustering
- Per-cluster HNSW indexes
- **Deliverable:** IVF-HNSW working

**Milestone 4: Optimization**
- nprobe tuning
- Cluster rebalancing
- Memory optimization
- **Deliverable:** Production-ready IVF

### Phase 3: Multi-Vector Support (4 weeks)

**Milestone 5: Multi-Vector Storage**
- Extend store API for multiple embeddings
- Update indexing to handle multi-vector docs
- **Deliverable:** Multi-vector storage working

**Milestone 6: Multi-Vector Search**
- Weighted fusion across vector stores
- Result merging strategies
- **Deliverable:** Multi-vector search working

### Phase 4: Advanced Features (4 weeks)

**Milestone 7: Configurable Parameters**
- Runtime ef_search configuration
- Distance metric selection
- Similarity thresholds
- **Deliverable:** Full parameter control

**Milestone 8: Scalar Quantization**
- SQ4, SQ8, SQ16 implementation
- Automatic quantization selection
- **Deliverable:** Scalar quantization available

### Phase 5: Optimization & Documentation (3 weeks)

**Milestone 9: Performance Tuning**
- Benchmark all quantization methods
- Optimize memory usage
- SIMD acceleration where possible
- **Deliverable:** Performance report

**Milestone 10: Documentation**
- Quantization guide
- Multi-vector best practices
- Migration documentation
- **Deliverable:** Complete user guide

## Performance Implications

### Memory Usage Comparison

**Baseline (1M documents, 768D embeddings):**

| Method | Memory | Accuracy | Search Latency |
|--------|--------|----------|----------------|
| Full precision (float32) | 3.07 GB | 100% (baseline) | 20ms |
| Binary quantization | 96 MB | 85-90% | 15ms |
| Scalar quantization (SQ8) | 768 MB | 95-97% | 17ms |
| Product quantization (PQ8) | 96 MB | 92-95% | 18ms |
| Product quantization (PQ16) | 192 MB | 97-99% | 19ms |

**Key insights:**
- **PQ8**: Same memory as binary quantization, much better accuracy
- **SQ8**: Good balance at 75% memory reduction
- **PQ16**: Near full-precision accuracy at 94% memory reduction

### IVF-HNSW Scalability

**Search complexity:**
- Pure HNSW: O(log N) with high constant
- IVF-HNSW: O(log C + log(N/C)) where C = num_clusters

**Benchmarks (simulated):**

| Dataset Size | Pure HNSW | IVF-HNSW (256 clusters) |
|--------------|-----------|-------------------------|
| 1M vectors | 25ms | 20ms |
| 10M vectors | 45ms | 28ms |
| 100M vectors | 80ms | 35ms |

**Memory for 100M vectors (768D):**
- Pure HNSW: ~350 GB
- IVF-HNSW with PQ8: ~15 GB

### Benchmarking Strategy

**Test scenarios:**
1. Compare quantization methods on standard datasets (SIFT, GloVe)
2. Measure recall@10, recall@100 at different memory budgets
3. Latency percentiles (p50, p95, p99)
4. Multi-vector performance vs single-vector

**Expected results:**
- PQ8: 92-95% recall at 32× compression
- IVF-HNSW: 2× faster search on >10M vectors
- Multi-vector: +30% latency per additional vector

## Open Questions

### 1. Default Quantization Method

**Question:** What should be the default quantization?

**Options:**
- A: Keep binary quantization (backward compatible)
- B: PQ8 (better accuracy/memory trade-off)
- C: Adaptive based on dataset size

**Recommendation:** Option C
- <10M vectors: Full precision
- 10-100M: PQ8 or SQ8
- >100M: IVF-HNSW with PQ8

### 2. PQ Training Data Selection

**Question:** How many vectors needed to train PQ codebooks?

**Options:**
- A: Fixed percentage (10%)
- B: Fixed count (100k vectors)
- C: Adaptive based on num_subvectors

**Recommendation:** Option C - `num_subvectors × 1000` vectors

### 3. Multi-Vector Fusion Strategy

**Question:** How to combine scores from multiple vector stores?

**Options:**
- A: Weighted sum (proposed above)
- B: Max score
- C: RRF across vector stores

**Recommendation:** Option A default, configurable to B or C

### 4. IVF Cluster Count

**Question:** How to determine optimal number of clusters?

**Options:**
- A: Fixed formula: sqrt(N)
- B: User-specified
- C: Auto-tune based on benchmarks

**Recommendation:** Option C with Option B override

## Alternatives Considered

### 1. FAISS Integration

**Approach:** Integrate Facebook's FAISS library

**Why not chosen:**
- C++ dependency (FFI complexity)
- Licensing concerns (MIT vs Apache-2.0)
- Prefer pure Rust for maintainability
- Can implement best FAISS algorithms in Rust

### 2. ScaNN Algorithm

**Approach:** Use Google's ScaNN (learned quantization)

**Why not chosen:**
- Requires training ML model
- More complex than PQ
- PQ provides sufficient accuracy for most use cases
- Future enhancement possibility

### 3. Graph-Based Quantization

**Approach:** Quantize HNSW graph structure instead of vectors

**Why not chosen:**
- Limited research on effectiveness
- Complex implementation
- PQ is proven and well-understood

## References

### Research Papers

1. **Product Quantization:**
   - Jégou, H., Douze, M., & Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*.
   - [Paper Link](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)

2. **IVF-HNSW:**
   - Baranchuk, D., et al. (2019). "Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors." *ECCV 2018*.

3. **HNSW:**
   - Malkov, Y., & Yashunin, D. (2020). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI*.
   - [Paper Link](https://arxiv.org/abs/1603.09320)

4. **Scalar Quantization:**
   - Guo, R., et al. (2020). "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." *ICML 2020*.

### Existing Implementations

1. **FAISS (Facebook):**
   - [GitHub](https://github.com/facebookresearch/faiss)
   - Implements PQ, IVF, and many other algorithms

2. **Qdrant:**
   - [Documentation](https://qdrant.tech/documentation/guides/quantization/)
   - Rust-based with PQ and SQ support

3. **Pinecone:**
   - [Blog on Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/)

### Meilisearch Codebase

1. **Vector store:** [`crates/milli/src/vector/store.rs`](crates/milli/src/vector/store.rs:1)
2. **Embedder abstraction:** [`crates/milli/src/vector/embedder/mod.rs`](crates/milli/src/vector/embedder/mod.rs:1)
3. **Vector search:** [`crates/milli/src/search/new/vector_sort.rs`](crates/milli/src/search/new/vector_sort.rs:1)
4. **Hybrid search:** [`crates/milli/src/search/hybrid.rs`](crates/milli/src/search/hybrid.rs:1)

## Community Discussion

Key discussion points:

1. **Quantization choice:** PQ vs SQ vs Binary - which default?
2. **Memory budgets:** Auto-select quantization based on available RAM?
3. **Multi-vector use cases:** What are the most common patterns?
4. **IVF necessity:** Is this needed given target dataset sizes?
5. **Performance trade-offs:** What accuracy loss is acceptable?

**Discussion link:** TBD after posting to GitHub

---

**Changelog:**
- 2025-11-10: Initial draft created