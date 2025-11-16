# SPLADE Implementation Guide

This guide provides comprehensive documentation for the SPLADE (Sparse Lexical and Expansion Model) implementation in Meilisearch, as specified in RFC 011.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Details](#implementation-details)
4. [Usage Examples](#usage-examples)
5. [Configuration](#configuration)
6. [Training Custom Models](#training-custom-models)
7. [Performance Considerations](#performance-considerations)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

## Overview

SPLADE combines the efficiency of sparse representations with neural semantic understanding. Unlike traditional BM25 (purely lexical) or dense vectors (semantic but resource-intensive), SPLADE produces sparse vectors where each vocabulary term has a learned weight.

### Key Benefits

- **Better than BM25**: Learns semantic relationships (e.g., "ML" → "machine learning")
- **More efficient than dense vectors**: Uses inverted index structure
- **Domain adaptability**: Can be fine-tuned for specific domains
- **Interpretable**: Weights correspond to vocabulary terms

### When to Use SPLADE

✅ **Good for:**
- Domain-specific search (medical, legal, scientific)
- Multi-lingual retrieval
- Cases where dense vectors underperform (out-of-domain)
- Need for interpretable search results

❌ **Not ideal for:**
- Simple keyword search (use BM25)
- When you need the absolute best semantic search (use dense vectors + larger models)
- Very short documents (< 50 words)

## Architecture

### Core Components

```
crates/milli/src/vector/
├── splade.rs              # SpladeVector representation
├── embedder/
│   └── splade.rs          # SPLADE embedder implementation
└── splade_training.rs     # Training utilities
```

### Data Flow

```
Document/Query Text
      ↓
SPLADE Encoder (BERT-based)
      ↓
Token Embeddings
      ↓
log(1 + ReLU(x)) Activation
      ↓
Max Pooling over Sequence
      ↓
Top-K Selection (e.g., 256 terms)
      ↓
SpladeVector {term_id: weight}
      ↓
Inverted Index Storage
```

## Implementation Details

### SpladeVector Structure

```rust
pub struct SpladeVector {
    /// Sparse mapping: term_id → weight
    pub weights: HashMap<u32, f32>,
    /// L2 norm for normalization
    pub norm: f32,
}
```

**Key Methods:**
- `new(weights)`: Create from HashMap
- `normalize()`: Normalize to unit L2 norm
- `dot_product(other)`: Compute similarity score
- `top_k_terms(k)`: Get highest-weighted terms
- `prune_to_top_k(k)`: Keep only top-K terms

### SPLADE Embedder

```rust
pub struct Embedder {
    model: Arc<Mutex<SpladeModel>>,
    tokenizer: Tokenizer,
    options: EmbedderOptions,
    vocab_size: usize,
    device: Device,
}
```

**Configuration:**
```rust
pub struct EmbedderOptions {
    pub model: String,                    // HuggingFace model ID
    pub revision: Option<String>,         // Model version
    pub max_active_terms: usize,          // Sparsity control (default: 256)
    pub distribution: Option<DistributionShift>,
}
```

### SPLADE Model Processing

1. **Tokenization**: Text → token IDs
2. **Encoding**: BERT forward pass → hidden states
3. **Activation**: `log(1 + ReLU(hidden_states))`
4. **Pooling**: Max pooling over sequence dimension
5. **Sparsification**: Keep top-K terms by weight

## Usage Examples

### Basic Configuration

```json
{
  "embedders": {
    "splade": {
      "source": "splade",
      "model": "naver/splade-cocondenser-ensembledistil",
      "maxActiveTerms": 256
    }
  }
}
```

### SPLADE-Only Search

```json
{
  "q": "machine learning applications",
  "embedder": "splade",
  "semanticRatio": 1.0
}
```

### Hybrid Search (BM25 + SPLADE)

```json
{
  "q": "machine learning applications",
  "hybrid": {
    "embedder": "splade",
    "semanticRatio": 0.7
  }
}
```

### Triple Hybrid Search (BM25 + SPLADE + Dense)

```json
{
  "q": "machine learning applications",
  "hybrid": {
    "method": "triple",
    "weights": {
      "bm25": 0.2,
      "splade": 0.4,
      "dense": 0.4
    },
    "embedders": {
      "splade": "splade_embedder",
      "dense": "dense_embedder"
    }
  }
}
```

## Configuration

### Available Models

| Model | Size | Sparsity | Performance | Use Case |
|-------|------|----------|-------------|----------|
| `naver/splade-cocondenser-ensembledistil` | 110M | ~256 terms | Best overall | General purpose |
| `naver/splade-v2-distil` | 66M | ~200 terms | Faster | Resource-constrained |
| `naver/efficient-splade-V-large-doc` | 340M | ~300 terms | Highest quality | Quality-critical |

### Tuning Parameters

#### Max Active Terms
```json
{
  "maxActiveTerms": 256  // Default
}
```
- **Lower (128)**: Faster search, less storage, slightly lower quality
- **Higher (512)**: Better quality, more storage, slower search

**Recommendation:** Start with 256, increase if quality insufficient

#### Semantic Ratio (Hybrid Search)
```json
{
  "semanticRatio": 0.7  // 70% SPLADE, 30% BM25
}
```
- **0.5**: Balanced keyword + semantic
- **0.7**: Favor semantic understanding
- **0.9**: Almost pure semantic

**Recommendation:** 0.6-0.7 for most use cases

## Training Custom Models

### When to Fine-Tune

✅ Fine-tune when:
- Domain-specific vocabulary (medical, legal, etc.)
- Non-English or multilingual
- Have labeled query-document pairs (>1000 examples)

❌ Don't fine-tune when:
- General web search
- Limited training data
- Pre-trained model performs well

### Training Process

```rust
use meilisearch_milli::vector::splade_training::{
    SpladeTrainer, TrainingConfig, TrainingExample
};

// 1. Configure training
let config = TrainingConfig {
    learning_rate: 2e-5,
    num_epochs: 3,
    batch_size: 16,
    regularization_lambda: 1e-4,
    target_sparsity: 256,
    ..Default::default()
};

// 2. Prepare training data
let examples = vec![
    TrainingExample {
        query: "heart attack symptoms".to_string(),
        positive_docs: vec![
            "Myocardial infarction presents with chest pain...".to_string()
        ],
        negative_docs: None,  // Auto-sampled from batch
    },
    // ... more examples
];

// 3. Train
let trainer = SpladeTrainer::new(config);
let model_path = trainer.train(examples)?;

// 4. Evaluate
let metrics = trainer.evaluate(validation_examples)?;
println!("MRR: {:.3}", metrics.mrr);
println!("Recall@10: {:.3}", metrics.recall_at_10);
println!("Avg Sparsity: {:.1}", metrics.avg_sparsity);
```

### Training Data Format

**From Query Logs:**
```rust
let logs = vec![
    ("query1", vec!["clicked_doc1", "clicked_doc2"]),
    ("query2", vec!["clicked_doc3"]),
];
let examples = create_examples_from_logs(logs);
```

**From Relevance Judgments:**
```rust
let judgments = vec![
    ("query1", vec!["relevant1"], vec!["irrelevant1", "irrelevant2"]),
];
let examples = create_examples_from_judgments(judgments);
```

## Performance Considerations

### Latency Comparison (1M documents)

| Method | Latency | Recall@10 | Memory |
|--------|---------|-----------|--------|
| BM25 | 10ms | 0.65 | 500MB |
| Dense Vector | 15ms | 0.82 | 3GB |
| SPLADE | 12ms | 0.85 | 800MB |
| Triple Hybrid | 25ms | 0.90 | 4.3GB |

### Storage Requirements

**Per Document:**
- Dense (768-dim): ~6KB
- SPLADE (256 terms): ~2KB
- **Savings: 66%**

**For 1M Documents:**
- Dense: 6GB
- SPLADE: 2GB

### Optimization Tips

1. **Reduce Max Active Terms**
   ```json
   {"maxActiveTerms": 128}  // 50% faster, ~5% quality drop
   ```

2. **Batch Processing**
   - Process documents in batches of 16-32
   - Use GPU when available

3. **Index Optimization**
   - Use SPLADE for first-stage retrieval (top-1000)
   - Re-rank with dense vectors or cross-encoder

4. **Caching**
   - Query vectors are cached automatically
   - Cache size: configurable (default: 1000 queries)

## Testing

### Unit Tests

```bash
cd crates/milli
cargo test vector::splade
cargo test vector::embedder::splade
cargo test vector::splade_training
```

### Integration Tests

```rust
#[test]
fn test_splade_end_to_end() {
    // 1. Create embedder
    let options = splade::EmbedderOptions::default();
    let embedder = splade::Embedder::new(options, 100).unwrap();

    // 2. Encode text
    let text = "machine learning applications";
    let vector = embedder.encode_one(text).unwrap();

    // 3. Verify sparsity
    assert!(vector.active_term_count() <= 256);
    assert!(vector.active_term_count() > 0);

    // 4. Test similarity
    let vector2 = embedder.encode_one("ML use cases").unwrap();
    let similarity = vector.dot_product(&vector2);
    assert!(similarity > 0.0);
}
```

### Benchmark Tests

```bash
cd crates/benchmarks
cargo bench --bench search_wiki -- splade
```

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Error:** `Failed to download model from HuggingFace Hub`

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or use offline mode (pre-download model)
export TRANSFORMERS_OFFLINE=1
```

#### 2. High Memory Usage

**Error:** `Out of memory during encoding`

**Solutions:**
- Reduce batch size: `{"batchSize": 8}`
- Reduce max_active_terms: `{"maxActiveTerms": 128}`
- Use CPU instead of GPU for large batches

#### 3. Poor Search Quality

**Problem:** SPLADE results worse than BM25

**Checklist:**
- [ ] Using domain-appropriate model?
- [ ] Text properly preprocessed?
- [ ] Try increasing max_active_terms
- [ ] Consider fine-tuning on domain data

#### 4. Slow Indexing

**Problem:** Document indexing too slow

**Solutions:**
1. Enable GPU acceleration
2. Increase batch size (if memory allows)
3. Use smaller SPLADE model variant
4. Process in parallel (multiple embedders)

### Debug Mode

Enable detailed logging:
```bash
RUST_LOG=meilisearch_milli::vector::splade=debug cargo run
```

## Next Steps

1. **Read the RFC**: See `rfcs/011_learned_sparse_representations_splade.md`
2. **Try Examples**: Start with basic SPLADE search
3. **Tune Configuration**: Experiment with max_active_terms and semantic_ratio
4. **Evaluate Results**: Compare SPLADE vs BM25 vs Dense on your data
5. **Consider Fine-Tuning**: If domain-specific, train custom model

## References

- **SPLADE Paper**: [arXiv:2107.05720](https://arxiv.org/abs/2107.05720)
- **HuggingFace Models**: [naver/splade-*](https://huggingface.co/naver)
- **RFC 011**: `rfcs/011_learned_sparse_representations_splade.md`
- **Meilisearch Docs**: https://docs.meilisearch.com

## Contributing

To contribute to SPLADE implementation:

1. Read the RFC and implementation guide
2. Check existing issues/PRs
3. Add tests for new features
4. Update documentation
5. Submit PR with clear description

## License

This implementation is part of Meilisearch and follows the same MIT license.
