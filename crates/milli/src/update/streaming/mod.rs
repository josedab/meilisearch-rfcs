//! Streaming indexing with real-time updates
//!
//! This module implements an LSM-tree-inspired architecture for Meilisearch that enables
//! true real-time document updates by introducing:
//! - In-memory write buffer with active and immutable memtables
//! - Write-ahead log (WAL) for durability and crash recovery
//! - Background merge pipeline for asynchronous LMDB merging
//!
//! See RFC 004 for detailed design documentation.

pub use self::merge::{MergeScheduler, MergeStrategy};
pub use self::recovery::{CrashRecovery, RecoveryStats};
pub use self::wal::{WriteAheadLog, WALEntry};
pub use self::write_buffer::{MemTable, WriteBuffer};

mod merge;
mod recovery;
mod wal;
mod write_buffer;

#[cfg(test)]
mod tests;
