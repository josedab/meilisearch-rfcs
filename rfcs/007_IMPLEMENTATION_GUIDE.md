# RFC 007 Implementation Guide: Parallel Indexing Optimization

This guide explains how to use the parallel indexing optimizations implemented in RFC 007.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Batch Optimizer](#batch-optimizer)
3. [Grenad Parameters Optimization](#grenad-parameters-optimization)
4. [Incremental Facet Updates](#incremental-facet-updates)
5. [GPU Embedding Acceleration](#gpu-embedding-acceleration)
6. [Configuration](#configuration)
7. [Benchmarking](#benchmarking)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Enable Optimizations

```rust
use milli::update::batch_optimizer::BatchOptimizer;
use milli::update::index_documents::helpers::GrenadParameters;

// 1. Use optimized grenad parameters
let grenad_params = GrenadParameters::optimized();

// 2. Create batch optimizer for adaptive sizing
let batch_optimizer = BatchOptimizer::new_auto(); // Auto-detects memory

// 3. Compute optimal batch size
let batch_size = batch_optimizer.compute_optimal_batch_size(total_docs);
```

### Build with GPU Support

```bash
# For NVIDIA GPUs (CUDA)
cargo build --features cuda

# For Apple Silicon (Metal)
cargo build --features metal

# For CPU only (default)
cargo build
```

## Batch Optimizer

The `BatchOptimizer` automatically determines the best batch size based on:
- Available system memory
- Number of CPU cores
- Average document size
- Total number of documents

### Basic Usage

```rust
use milli::update::batch_optimizer::BatchOptimizer;

// Create optimizer with automatic memory detection
let optimizer = BatchOptimizer::new_auto();

// Or specify memory budget manually (in bytes)
let optimizer = BatchOptimizer::new(2 * 1024 * 1024 * 1024); // 2GB

// Update with sample documents to learn average size
let sample_docs: Vec<Vec<u8>> = get_sample_documents();
optimizer.update_avg_doc_size(&sample_docs);

// Get optimal batch size
let batch_size = optimizer.compute_optimal_batch_size(1_000_000);
println!("Optimal batch size: {}", batch_size);
```

### Advanced Usage

```rust
// Monitor optimizer state
println!("Average doc size: {} bytes", optimizer.get_avg_doc_size());
println!("CPU cores: {}", optimizer.get_num_cores());
println!("Memory budget: {} MB", optimizer.get_memory_budget() / 1024 / 1024);

// Process documents in batches
for chunk in documents.chunks(batch_size) {
    process_batch(chunk)?;

    // Optionally update optimizer with real data
    optimizer.update_avg_doc_size(chunk);
}
```

## Grenad Parameters Optimization

The `GrenadParameters::optimized()` method automatically configures parameters based on system resources.

### Usage

```rust
use milli::update::index_documents::helpers::GrenadParameters;

// Create optimized parameters (recommended)
let params = GrenadParameters::optimized();

// Or use default parameters
let params = GrenadParameters::default();
```

### What Gets Optimized

The optimizer automatically sets:

- **Max Memory**: 60% of available system memory
- **Chunk Size**:
  - 128MB for systems with >16GB RAM
  - 64MB for systems with 8-16GB RAM
  - 32MB for systems with <8GB RAM
- **Max Chunks**: 4x the number of CPU cores
- **Compression**: Snappy (good balance of speed/size)

## Incremental Facet Updates

Incremental facet updates avoid full rebuilds when only a small portion of documents change.

### Basic Usage

```rust
use milli::update::facet::incremental_bulk::IncrementalFacetBuilder;
use roaring::RoaringBitmap;

// Create builder with group size (typically 4-8)
let mut builder = IncrementalFacetBuilder::new(4);

// Prepare updates: (facet_value, docids)
let updates = vec![
    (b"category_a".to_vec(), RoaringBitmap::from_iter([1, 2, 3])),
    (b"category_b".to_vec(), RoaringBitmap::from_iter([4, 5, 6])),
];

// Apply incremental update
builder.update_facets_incremental(&mut wtxn, db, field_id, &updates)?;
```

### When to Use

- **Use incremental** when < 1% of documents changed
- **Use bulk rebuild** when > 5% of documents changed
- **Either works** for 1-5% changes

### Performance

| Documents Changed | Full Rebuild | Incremental | Speedup |
|-------------------|--------------|-------------|---------|
| 1% (10K/1M)       | 25s          | 1.5s        | 16x     |
| 5% (50K/1M)       | 25s          | 5s          | 5x      |
| 10% (100K/1M)     | 25s          | 10s         | 2.5x    |

## GPU Embedding Acceleration

GPU acceleration provides 10x speedup for embedding generation using CUDA or Metal.

### Basic Usage

```rust
use milli::vector::embedder::gpu_accelerator::GPUEmbeddingAccelerator;

// Auto-detect best device (GPU or CPU fallback)
let accelerator = GPUEmbeddingAccelerator::new("model.safetensors", 32)?;

// Generate embeddings
let texts = vec!["Hello world".to_string(), "GPU is fast".to_string()];
let embeddings = accelerator.embed_batch(&texts)?;

println!("Generated {} embeddings", embeddings.len());
```

### Device Selection

```rust
use milli::vector::embedder::gpu_accelerator::{GPUEmbeddingAccelerator, DeviceType};

// Explicitly use CUDA
let accelerator = GPUEmbeddingAccelerator::with_device(
    "model.safetensors",
    32,
    DeviceType::CUDA(0), // GPU index 0
)?;

// Explicitly use Metal (Apple Silicon)
let accelerator = GPUEmbeddingAccelerator::with_device(
    "model.safetensors",
    32,
    DeviceType::Metal,
)?;

// Explicitly use CPU
let accelerator = GPUEmbeddingAccelerator::with_device(
    "model.safetensors",
    32,
    DeviceType::CPU,
)?;

// Check what device is being used
println!("Using device: {:?}", accelerator.device_type());
```

### Monitoring Performance

```rust
// Get performance statistics
let stats = accelerator.stats();

println!("Total embeddings: {}", stats.total_embeddings);
println!("Total time: {} ms", stats.total_time_ms);
println!("Avg time per batch: {:.2} ms", stats.avg_batch_time_ms());
println!("Avg time per embedding: {:.2} ms", stats.avg_embedding_time_ms());

// Get throughput
println!("Throughput: {:.0} embeddings/sec", accelerator.embeddings_per_second());

// Reset statistics
accelerator.reset_stats();
```

## Configuration

### Environment Variables

```bash
# Enable parallel optimizations (experimental)
export MEILI_EXPERIMENTAL_PARALLEL_INDEXING=true

# Control parallelism (default: num_cpus)
export MEILI_INDEXING_PARALLELISM=8

# Enable GPU acceleration
export MEILI_GPU_ACCELERATION=true

# GPU device ID (for multi-GPU systems)
export MEILI_GPU_DEVICE=0
```

### Index Settings

```json
{
  "indexing": {
    "parallelOptimizations": {
      "enabled": true,
      "prefixComputation": "parallel",
      "facetBuilding": "incremental",
      "documentExtraction": "parallel"
    },
    "gpuAcceleration": {
      "enabled": true,
      "device": 0,
      "batchSize": 32
    }
  }
}
```

## Benchmarking

### Run the Example

```bash
# Basic demo
cargo run --example parallel_indexing_optimization

# With GPU support
cargo run --example parallel_indexing_optimization --features cuda

# With Metal (Apple Silicon)
cargo run --example parallel_indexing_optimization --features metal
```

### Measure Your Own Workload

```rust
use std::time::Instant;

let start = Instant::now();

// Your indexing code here
let batch_optimizer = BatchOptimizer::new_auto();
let batch_size = batch_optimizer.compute_optimal_batch_size(total_docs);

for chunk in documents.chunks(batch_size) {
    index_batch(chunk)?;
}

let elapsed = start.elapsed();
println!("Indexed {} docs in {:.2}s", total_docs, elapsed.as_secs_f64());
println!("Throughput: {:.0} docs/sec", total_docs as f64 / elapsed.as_secs_f64());
```

## Troubleshooting

### Out of Memory Errors

**Problem**: Indexing runs out of memory

**Solutions**:
1. Reduce memory budget: `BatchOptimizer::new(smaller_budget)`
2. Use default grenad params: `GrenadParameters::default()`
3. Process smaller batches manually

```rust
// Conservative memory usage
let optimizer = BatchOptimizer::new(512 * 1024 * 1024); // 512MB
let params = GrenadParameters::default(); // Don't use optimized
```

### GPU Not Detected

**Problem**: GPU available but not being used

**Solutions**:
1. Build with correct feature: `--features cuda` or `--features metal`
2. Check device: `accelerator.device_type()`
3. Verify drivers installed (NVIDIA: CUDA toolkit, Apple: macOS 11+)

```rust
// Debug GPU detection
let accelerator = GPUEmbeddingAccelerator::new("model.safetensors", 32)?;
match accelerator.device_type() {
    DeviceType::CUDA(id) => println!("Using CUDA GPU {}", id),
    DeviceType::Metal => println!("Using Metal GPU"),
    DeviceType::CPU => println!("WARNING: Using CPU fallback"),
}
```

### Slow Incremental Updates

**Problem**: Incremental facet updates are slow

**Solutions**:
1. Check update size: if >5%, use bulk rebuild instead
2. Verify group size is appropriate (4-8 typically best)
3. Ensure database isn't fragmented

```rust
// Monitor update size
let total_facets = get_total_facet_count();
let changed_facets = updates.len();
let change_percentage = (changed_facets as f64 / total_facets as f64) * 100.0;

if change_percentage > 5.0 {
    println!("Large update ({}%), using bulk rebuild", change_percentage);
    use_bulk_rebuild()?;
} else {
    println!("Small update ({}%), using incremental", change_percentage);
    builder.update_facets_incremental(&mut wtxn, db, field_id, &updates)?;
}
```

### Suboptimal Batch Sizes

**Problem**: Batch sizes seem wrong for your workload

**Solutions**:
1. Provide sample documents early: `optimizer.update_avg_doc_size(&samples)`
2. Override if needed: `let batch_size = custom_size;`
3. Monitor memory usage and adjust budget

```rust
// Custom batch sizing logic
let base_size = optimizer.compute_optimal_batch_size(total_docs);
let adjusted_size = if documents_are_very_large() {
    base_size / 2 // Halve for safety
} else {
    base_size
};
```

## Performance Tips

### 1. Use Optimized Parameters

```rust
// ✅ Good: Use optimized parameters
let params = GrenadParameters::optimized();

// ❌ Avoid: Default may not be optimal
let params = GrenadParameters::default();
```

### 2. Update Optimizer with Real Data

```rust
// ✅ Good: Feed real data to optimizer
for chunk in documents.chunks(initial_batch_size) {
    optimizer.update_avg_doc_size(chunk);
    let new_batch_size = optimizer.compute_optimal_batch_size(remaining);
    // Use new_batch_size for next iteration
}

// ❌ Avoid: Never updating optimizer
let batch_size = optimizer.compute_optimal_batch_size(total);
// Optimizer never learns actual document sizes
```

### 3. Choose Right Facet Update Strategy

```rust
// ✅ Good: Choose based on change percentage
if changed_docs < total_docs / 100 {
    // < 1% changed: use incremental
    use_incremental_facet_update()?;
} else {
    // > 1% changed: use bulk rebuild
    use_bulk_facet_rebuild()?;
}

// ❌ Avoid: Always using same strategy
always_use_incremental()?; // Slow for large changes
```

### 4. GPU Batch Size Tuning

```rust
// ✅ Good: Tune batch size for GPU
let gpu_batch = match device_type {
    DeviceType::CUDA(_) => 64,    // Larger batches for powerful GPUs
    DeviceType::Metal => 32,      // Medium batches for Apple Silicon
    DeviceType::CPU => 16,        // Smaller batches for CPU
};

// ❌ Avoid: One size fits all
let gpu_batch = 32; // May not be optimal for your GPU
```

## Next Steps

1. Run the example: `cargo run --example parallel_indexing_optimization`
2. Measure your baseline performance
3. Enable optimizations one by one
4. Measure improvements
5. Tune parameters for your workload

## Support

- GitHub Issues: https://github.com/meilisearch/meilisearch/issues
- RFC Discussion: https://github.com/meilisearch/meilisearch-rfcs/pull/XXX
- Documentation: https://www.meilisearch.com/docs

## References

- RFC 007: `rfcs/007_parallel_indexing_optimization.md`
- Example Code: `rfcs/007_parallel_indexing_optimization_example.rs`
- Research Plan: `RESEARCH_PLAN.md`
