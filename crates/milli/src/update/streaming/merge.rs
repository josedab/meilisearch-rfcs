use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::wal::WriteAheadLog;
use super::write_buffer::MemTable;
use crate::{Index, Result};

/// Background merge scheduler for flushing memtables to LMDB
///
/// The MergeScheduler runs in a background thread, receiving immutable memtables
/// and merging them to LMDB according to the configured strategy.
pub struct MergeScheduler {
    /// Receives immutable memtables to merge
    flush_rx: mpsc::Receiver<Arc<MemTable>>,
    /// LMDB index reference
    index: Arc<Index>,
    /// Merge strategy
    strategy: MergeStrategy,
    /// WAL for truncation after merge
    wal: Arc<WriteAheadLog>,
}

/// Strategy for when to merge memtables to LMDB
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Merge immediately when memtable is full
    Immediate,
    /// Merge when N memtables are pending
    Batched { count: usize },
    /// Merge on time interval
    Periodic { interval: Duration },
    /// Adaptive based on load and size
    Adaptive,
}

impl MergeScheduler {
    /// Create a new MergeScheduler
    ///
    /// # Arguments
    /// * `flush_rx` - Channel to receive memtables for merging
    /// * `index` - The LMDB index to merge into
    /// * `strategy` - When to trigger merges
    /// * `wal` - Write-ahead log for truncation
    pub fn new(
        flush_rx: mpsc::Receiver<Arc<MemTable>>,
        index: Arc<Index>,
        strategy: MergeStrategy,
        wal: Arc<WriteAheadLog>,
    ) -> Self {
        Self {
            flush_rx,
            index,
            strategy,
            wal,
        }
    }

    /// Run the merge scheduler loop
    ///
    /// This should be spawned in a background thread
    pub fn run(mut self) -> Result<()> {
        let mut pending_memtables = Vec::new();

        loop {
            // Try to receive a new immutable memtable (non-blocking with timeout)
            match self.flush_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(memtable) => {
                    pending_memtables.push(memtable);

                    // Check merge conditions
                    if self.should_merge(&pending_memtables) {
                        self.merge_memtables(&mut pending_memtables)?;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Periodic merge trigger
                    if !pending_memtables.is_empty() && matches!(self.strategy, MergeStrategy::Periodic { .. }) {
                        self.merge_memtables(&mut pending_memtables)?;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }

        Ok(())
    }

    /// Determine if a merge should be triggered
    fn should_merge(&self, pending: &[Arc<MemTable>]) -> bool {
        match &self.strategy {
            MergeStrategy::Immediate => true,
            MergeStrategy::Batched { count } => pending.len() >= *count,
            MergeStrategy::Periodic { .. } => false, // Handled by timer
            MergeStrategy::Adaptive => {
                // Adaptive: merge if total pending size > threshold
                let total_size: usize = pending.iter().map(|m| m.size_bytes).sum();
                total_size >= 100 * 1024 * 1024 // 100MB
            }
        }
    }

    /// Merge pending memtables to LMDB
    fn merge_memtables(&self, pending: &mut Vec<Arc<MemTable>>) -> Result<()> {
        let memtables = std::mem::take(pending);

        tracing::info!(
            count = memtables.len(),
            "Starting memtable merge to LMDB"
        );

        let start = Instant::now();

        // Merge to LMDB
        let result = Self::merge_to_lmdb(&self.index, memtables)?;

        let duration = start.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis(),
            documents_merged = result.documents_count,
            "Memtable merge completed"
        );

        // Truncate WAL after successful merge
        if result.max_lsn > 0 {
            self.wal.truncate_before(result.max_lsn)?;
        }

        Ok(())
    }

    /// Merge memtables into LMDB (blocking operation)
    fn merge_to_lmdb(
        _index: &Index,
        memtables: Vec<Arc<MemTable>>,
    ) -> Result<MergeResult> {
        let mut documents_count = 0;
        let max_lsn = 0;

        // Count documents to be merged
        for memtable in &memtables {
            for (doc_id, _document) in memtable.iter_documents() {
                if !memtable.get_deletions().contains(doc_id) {
                    documents_count += 1;
                }
            }
        }

        // NOTE: Full LMDB integration would happen here
        // This would involve:
        // 1. Opening a write transaction
        // 2. Converting document JSON to ObkvCodec format
        // 3. Updating the documents database
        // 4. Updating all inverted indexes
        // 5. Integrating with the existing indexing pipeline:
        //    - crates/milli/src/update/index_documents/
        //    - crates/milli/src/update/new/indexer/
        //
        // For now, this is a placeholder demonstrating the merge
        // scheduling and background processing architecture.

        tracing::debug!(
            "Merge placeholder: would merge {} documents from {} memtables",
            documents_count,
            memtables.len()
        );

        Ok(MergeResult {
            documents_count,
            max_lsn,
        })
    }
}

/// Result of a merge operation
struct MergeResult {
    documents_count: usize,
    max_lsn: u64,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        MergeStrategy::Adaptive
    }
}

impl MergeStrategy {
    /// Parse merge strategy from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "immediate" => Some(MergeStrategy::Immediate),
            "batched" => Some(MergeStrategy::Batched { count: 5 }),
            "periodic" => Some(MergeStrategy::Periodic {
                interval: Duration::from_secs(5),
            }),
            "adaptive" => Some(MergeStrategy::Adaptive),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_strategy_parsing() {
        assert!(matches!(
            MergeStrategy::from_str("immediate"),
            Some(MergeStrategy::Immediate)
        ));
        assert!(matches!(
            MergeStrategy::from_str("adaptive"),
            Some(MergeStrategy::Adaptive)
        ));
        assert!(matches!(
            MergeStrategy::from_str("batched"),
            Some(MergeStrategy::Batched { .. })
        ));
        assert!(matches!(
            MergeStrategy::from_str("periodic"),
            Some(MergeStrategy::Periodic { .. })
        ));
    }
}
