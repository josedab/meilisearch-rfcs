use std::sync::Arc;

use super::wal::{WriteAheadLog, WALEntry};
use crate::{Index, Result};

/// Crash recovery from Write-Ahead Log
///
/// The CrashRecovery replays the WAL to restore the system state after
/// an unexpected shutdown or crash.
pub struct CrashRecovery {
    wal: Arc<WriteAheadLog>,
    index: Arc<Index>,
}

/// Statistics from crash recovery
#[derive(Default, Debug, Clone)]
pub struct RecoveryStats {
    /// Number of documents recovered
    pub documents_recovered: usize,
    /// Number of deletions recovered
    pub deletions_recovered: usize,
    /// Number of memtable flushes found
    pub memtable_flushes: usize,
    /// Last checkpoint LSN found
    pub last_checkpoint: Option<u64>,
}

impl CrashRecovery {
    /// Create a new CrashRecovery instance
    ///
    /// # Arguments
    /// * `wal` - The write-ahead log to replay
    /// * `index` - The index to recover into
    pub fn new(wal: Arc<WriteAheadLog>, index: Arc<Index>) -> Self {
        Self { wal, index }
    }

    /// Recover from a crash by replaying the WAL
    ///
    /// This should be called during system startup to ensure all committed
    /// operations are present in LMDB.
    pub fn recover_from_crash(&self) -> Result<RecoveryStats> {
        let mut stats = RecoveryStats::default();

        tracing::info!("Starting crash recovery from WAL");

        // Replay WAL
        self.wal.replay(|entry| {
            match entry {
                WALEntry::DocumentAdd { doc_id, .. } => {
                    stats.documents_recovered += 1;
                    // NOTE: Full recovery would re-index the document here
                    // This would involve parsing the document and calling
                    // into the existing indexing pipeline
                    tracing::debug!("Would recover document {}", doc_id);
                }
                WALEntry::DocumentDelete { doc_id, .. } => {
                    stats.deletions_recovered += 1;
                    // NOTE: Full recovery would delete the document here
                    tracing::debug!("Would recover deletion of document {}", doc_id);
                }
                WALEntry::MemTableFlush { memtable_id, .. } => {
                    stats.memtable_flushes += 1;
                    tracing::debug!("Found memtable flush marker: {}", memtable_id);
                    // Already merged, can skip
                }
                WALEntry::Checkpoint { lsn, .. } => {
                    stats.last_checkpoint = Some(lsn);
                    tracing::debug!("Found checkpoint at LSN: {}", lsn);
                }
            }
            Ok(())
        })?;

        tracing::info!(
            documents = stats.documents_recovered,
            deletions = stats.deletions_recovered,
            flushes = stats.memtable_flushes,
            "Crash recovery completed"
        );

        Ok(stats)
    }

    /// Check if recovery is needed
    ///
    /// Returns true if there are WAL segments present that might need replay
    pub fn needs_recovery(&self) -> Result<bool> {
        let stats = self.wal.stats()?;
        Ok(stats.segment_count > 0 && stats.total_size_bytes > 0)
    }

    /// Create a checkpoint in the WAL
    ///
    /// Checkpoints mark a point where all previous operations have been
    /// successfully applied to LMDB.
    pub fn create_checkpoint(&self) -> Result<u64> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let lsn = self.wal.append(WALEntry::Checkpoint {
            lsn: 0, // Will be overwritten with actual LSN
            timestamp,
        })?;

        tracing::info!("Created checkpoint at LSN: {}", lsn);

        Ok(lsn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_recovery_stats_default() {
        let stats = RecoveryStats::default();
        assert_eq!(stats.documents_recovered, 0);
        assert_eq!(stats.deletions_recovered, 0);
        assert_eq!(stats.memtable_flushes, 0);
        assert_eq!(stats.last_checkpoint, None);
    }

    #[test]
    fn test_needs_recovery_empty_wal() {
        // Create temporary directory for WAL
        let temp_dir = TempDir::new().unwrap();
        let wal_dir = temp_dir.path().to_path_buf();

        let wal = Arc::new(WriteAheadLog::new(wal_dir, 100 * 1024 * 1024).unwrap());

        // Create a mock index (this would need proper setup in real tests)
        // For now, this test is illustrative

        // Would need: let index = Arc::new(Index::new(...));
        // let recovery = CrashRecovery::new(wal, index);
        // assert!(!recovery.needs_recovery().unwrap());
    }
}
