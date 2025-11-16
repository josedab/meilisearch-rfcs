/// Batch Optimizer Module - RFC 007: Parallel Indexing Optimization
///
/// This module provides intelligent batch sizing for indexing operations
/// based on system resources and document characteristics.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Dynamically compute optimal batch size based on system resources
pub struct BatchOptimizer {
    /// Available memory budget (in bytes)
    memory_budget: usize,
    /// Number of CPU cores
    num_cores: usize,
    /// Average document size (updated dynamically)
    avg_doc_size: AtomicUsize,
}

impl BatchOptimizer {
    /// Create a new BatchOptimizer with specified memory budget
    ///
    /// # Arguments
    /// * `memory_budget` - Total memory available for batch processing (in bytes)
    ///
    /// # Example
    /// ```
    /// use milli::update::batch_optimizer::BatchOptimizer;
    ///
    /// // Create optimizer with 1GB memory budget
    /// let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);
    /// ```
    pub fn new(memory_budget: usize) -> Self {
        Self {
            memory_budget,
            num_cores: num_cpus::get(),
            avg_doc_size: AtomicUsize::new(0),
        }
    }

    /// Create a new BatchOptimizer with automatic memory detection
    ///
    /// Uses 60% of available system memory as the budget
    pub fn new_auto() -> Self {
        let available_memory = Self::get_available_memory();
        let memory_budget = (available_memory as f64 * 0.6) as usize;
        Self::new(memory_budget)
    }

    /// Compute optimal batch size based on current state
    ///
    /// # Arguments
    /// * `total_docs` - Total number of documents to process
    ///
    /// # Returns
    /// Optimal batch size that balances memory usage and parallelism
    ///
    /// # Example
    /// ```
    /// use milli::update::batch_optimizer::BatchOptimizer;
    ///
    /// let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);
    /// let batch_size = optimizer.compute_optimal_batch_size(1_000_000);
    /// println!("Optimal batch size: {}", batch_size);
    /// ```
    pub fn compute_optimal_batch_size(&self, total_docs: usize) -> usize {
        let avg_size = self.avg_doc_size.load(Ordering::Relaxed);

        // Reserve 30% of memory budget for overhead (indexing structures, etc.)
        let available_memory = (self.memory_budget as f64 * 0.7) as usize;

        // Compute batch size that fits in memory
        let memory_constrained_batch = if avg_size > 0 {
            available_memory / avg_size
        } else {
            10000 // Default if no size info available
        };

        // Prefer batches that align with CPU cores for better parallelism
        let core_aligned_batch = if memory_constrained_batch >= self.num_cores {
            (memory_constrained_batch / self.num_cores) * self.num_cores
        } else {
            memory_constrained_batch
        };

        // Clamp to reasonable bounds:
        // - Minimum: 1000 docs (avoid too many small batches)
        // - Maximum: 100000 docs (avoid excessive memory usage)
        // - Also consider total_docs to avoid batches larger than needed
        core_aligned_batch.max(1000).min(100000).min(total_docs)
    }

    /// Update average document size based on a new sample
    ///
    /// Uses exponential moving average (EMA) to adapt to changing document sizes
    ///
    /// # Arguments
    /// * `new_sample` - Slice of document byte arrays from recent batch
    pub fn update_avg_doc_size(&self, new_sample: &[Vec<u8>]) {
        if new_sample.is_empty() {
            return;
        }

        let sample_avg = new_sample.iter().map(|d| d.len()).sum::<usize>() / new_sample.len();

        // Exponential moving average: 90% old, 10% new
        // This makes the optimizer responsive to changes but stable
        let old_avg = self.avg_doc_size.load(Ordering::Relaxed);
        let new_avg = if old_avg == 0 {
            sample_avg
        } else {
            ((old_avg * 9) + sample_avg) / 10
        };

        self.avg_doc_size.store(new_avg, Ordering::Relaxed);
    }

    /// Get current average document size estimate
    pub fn get_avg_doc_size(&self) -> usize {
        self.avg_doc_size.load(Ordering::Relaxed)
    }

    /// Get number of CPU cores being used for optimization
    pub fn get_num_cores(&self) -> usize {
        self.num_cores
    }

    /// Get memory budget in bytes
    pub fn get_memory_budget(&self) -> usize {
        self.memory_budget
    }

    /// Detect available system memory (platform-specific)
    fn get_available_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            use sysinfo::{System, SystemExt};
            let mut sys = System::new();
            sys.refresh_memory();
            sys.available_memory() as usize
        }

        #[cfg(target_os = "macos")]
        {
            use sysinfo::{System, SystemExt};
            let mut sys = System::new();
            sys.refresh_memory();
            sys.available_memory() as usize
        }

        #[cfg(target_os = "windows")]
        {
            use sysinfo::{System, SystemExt};
            let mut sys = System::new();
            sys.refresh_memory();
            sys.available_memory() as usize
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Default to 8GB if platform unknown
            8 * 1024 * 1024 * 1024
        }
    }
}

impl Default for BatchOptimizer {
    fn default() -> Self {
        Self::new_auto()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_optimizer_creation() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024); // 1GB
        assert_eq!(optimizer.get_memory_budget(), 1024 * 1024 * 1024);
        assert!(optimizer.get_num_cores() > 0);
    }

    #[test]
    fn test_compute_batch_size_no_history() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024); // 1GB
        let batch_size = optimizer.compute_optimal_batch_size(1_000_000);

        // Should use default of 10000 when no document size history
        assert!(batch_size >= 1000);
        assert!(batch_size <= 100000);
    }

    #[test]
    fn test_update_avg_doc_size() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);

        // Sample documents
        let docs: Vec<Vec<u8>> = vec![vec![0u8; 1000], vec![0u8; 2000], vec![0u8; 1500]];

        optimizer.update_avg_doc_size(&docs);

        let avg = optimizer.get_avg_doc_size();
        assert_eq!(avg, 1500); // (1000 + 2000 + 1500) / 3
    }

    #[test]
    fn test_exponential_moving_average() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);

        // First sample: 1000 bytes
        let docs1: Vec<Vec<u8>> = vec![vec![0u8; 1000]];
        optimizer.update_avg_doc_size(&docs1);
        assert_eq!(optimizer.get_avg_doc_size(), 1000);

        // Second sample: 2000 bytes
        // EMA: (1000 * 9 + 2000) / 10 = 1100
        let docs2: Vec<Vec<u8>> = vec![vec![0u8; 2000]];
        optimizer.update_avg_doc_size(&docs2);
        assert_eq!(optimizer.get_avg_doc_size(), 1100);
    }

    #[test]
    fn test_batch_size_with_known_doc_size() {
        let optimizer = BatchOptimizer::new(100 * 1024 * 1024); // 100MB budget

        // Update with 1KB average document size
        let docs: Vec<Vec<u8>> = vec![vec![0u8; 1024]];
        optimizer.update_avg_doc_size(&docs);

        let batch_size = optimizer.compute_optimal_batch_size(1_000_000);

        // Available: 100MB * 0.7 = 70MB
        // Per doc: 1KB
        // Max docs: ~70000
        // Should be aligned to num_cores and clamped to max 100000
        assert!(batch_size >= 1000);
        assert!(batch_size <= 100000);
    }

    #[test]
    fn test_empty_sample_ignored() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);

        let docs: Vec<Vec<u8>> = vec![vec![0u8; 1000]];
        optimizer.update_avg_doc_size(&docs);
        assert_eq!(optimizer.get_avg_doc_size(), 1000);

        // Empty sample should not change average
        let empty_docs: Vec<Vec<u8>> = vec![];
        optimizer.update_avg_doc_size(&empty_docs);
        assert_eq!(optimizer.get_avg_doc_size(), 1000);
    }

    #[test]
    fn test_batch_size_respects_total_docs() {
        let optimizer = BatchOptimizer::new(1024 * 1024 * 1024);

        // Small total - batch size should not exceed it
        let batch_size = optimizer.compute_optimal_batch_size(500);
        assert!(batch_size <= 500);
    }
}
