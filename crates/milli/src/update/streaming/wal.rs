use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::{DocumentId, Result};

/// Write-Ahead Log for crash recovery
///
/// The WAL provides durability guarantees by persisting all operations before
/// they are applied to memory structures. In case of a crash, the WAL can be
/// replayed to restore the system state.
pub struct WriteAheadLog {
    /// Current WAL segment
    current_segment: Arc<Mutex<WALSegment>>,
    /// WAL directory
    wal_dir: PathBuf,
    /// Maximum segment size before rotation
    max_segment_size: u64,
    /// Segment counter
    next_segment_id: AtomicU64,
}

/// A single WAL segment file
struct WALSegment {
    segment_id: u64,
    file: BufWriter<File>,
    size: u64,
}

/// Entry types stored in the WAL
#[derive(Debug, Serialize, Deserialize)]
pub enum WALEntry {
    /// A document was added or updated
    DocumentAdd {
        doc_id: DocumentId,
        document: Vec<u8>,
        timestamp: u64,
    },
    /// A document was deleted
    DocumentDelete {
        doc_id: DocumentId,
        timestamp: u64,
    },
    /// A memtable was flushed to LMDB
    MemTableFlush {
        memtable_id: u64,
        timestamp: u64,
    },
    /// A checkpoint marker
    Checkpoint {
        lsn: u64,
        timestamp: u64,
    },
}

impl WriteAheadLog {
    /// Create a new WriteAheadLog
    ///
    /// # Arguments
    /// * `wal_dir` - Directory to store WAL segments
    /// * `max_segment_size` - Maximum size of a segment before rotation
    pub fn new(wal_dir: PathBuf, max_segment_size: u64) -> Result<Self> {
        std::fs::create_dir_all(&wal_dir)?;

        let segment_id = 0;
        let segment = WALSegment::create(&wal_dir, segment_id)?;

        Ok(Self {
            current_segment: Arc::new(Mutex::new(segment)),
            wal_dir,
            max_segment_size,
            next_segment_id: AtomicU64::new(1),
        })
    }

    /// Append an entry to the WAL
    ///
    /// Returns the Log Sequence Number (LSN) of the entry
    pub fn append(&self, entry: WALEntry) -> Result<u64> {
        let mut segment = self.current_segment.lock().unwrap();

        // Serialize entry
        let entry_bytes = bincode::serialize(&entry).map_err(crate::InternalError::BincodeError)?;
        let entry_len = entry_bytes.len() as u32;

        // Write length prefix + entry
        segment.file.write_all(&entry_len.to_le_bytes())?;
        segment.file.write_all(&entry_bytes)?;
        segment.size += 4 + entry_len as u64;

        // Sync to disk (fsync for durability)
        segment.file.flush()?;
        segment.file.get_ref().sync_data()?;

        // Calculate LSN (segment_id << 32 | offset)
        let lsn = (segment.segment_id << 32) | segment.size;

        // Rotate segment if needed
        if segment.size >= self.max_segment_size {
            drop(segment); // Release lock
            self.rotate_segment()?;
        }

        Ok(lsn)
    }

    /// Rotate to a new WAL segment
    fn rotate_segment(&self) -> Result<()> {
        let new_segment_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
        let new_segment = WALSegment::create(&self.wal_dir, new_segment_id)?;

        let mut current = self.current_segment.lock().unwrap();
        *current = new_segment;

        tracing::info!("Rotated to new WAL segment: {}", new_segment_id);

        Ok(())
    }

    /// Replay WAL for crash recovery
    ///
    /// Calls the provided callback for each entry in the WAL
    pub fn replay<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(WALEntry) -> Result<()>,
    {
        // Find all WAL segments
        let mut segments: Vec<_> = std::fs::read_dir(&self.wal_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "wal"))
            .collect();

        segments.sort_by_key(|entry| entry.path());

        // Replay in order
        for segment_entry in segments {
            let path = segment_entry.path();
            let mut file = File::open(&path)?;

            loop {
                // Read length prefix
                let mut len_buf = [0u8; 4];
                match file.read_exact(&mut len_buf) {
                    Ok(_) => {}
                    Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e.into()),
                }

                let entry_len = u32::from_le_bytes(len_buf) as usize;

                // Read entry
                let mut entry_buf = vec![0u8; entry_len];
                file.read_exact(&mut entry_buf)?;

                let entry: WALEntry = bincode::deserialize(&entry_buf).map_err(crate::InternalError::BincodeError)?;
                callback(entry)?;
            }
        }

        Ok(())
    }

    /// Truncate WAL after successful merge to LMDB
    ///
    /// Removes all segments with ID less than the one containing the given LSN
    pub fn truncate_before(&self, lsn: u64) -> Result<()> {
        let segment_id = (lsn >> 32) as u64;

        // Delete old segments
        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_stem() {
                if let Ok(id) = name.to_string_lossy().parse::<u64>() {
                    if id < segment_id {
                        std::fs::remove_file(&path)?;
                        tracing::info!("Removed WAL segment: {:?}", path);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the current WAL directory
    pub fn wal_dir(&self) -> &Path {
        &self.wal_dir
    }

    /// Get statistics about the WAL
    pub fn stats(&self) -> Result<WALStats> {
        let mut segment_count = 0;
        let mut total_size = 0u64;

        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
            if entry.path().extension().map_or(false, |ext| ext == "wal") {
                segment_count += 1;
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            }
        }

        let current_segment_id = self.current_segment.lock().unwrap().segment_id;

        Ok(WALStats {
            segment_count,
            total_size_bytes: total_size,
            current_segment_id,
        })
    }
}

impl WALSegment {
    /// Create a new WAL segment file
    fn create(wal_dir: &Path, segment_id: u64) -> Result<Self> {
        let path = wal_dir.join(format!("{:010}.wal", segment_id));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        Ok(Self {
            segment_id,
            file: BufWriter::new(file),
            size: 0,
        })
    }
}

/// Statistics about the WAL state
#[derive(Debug, Clone)]
pub struct WALStats {
    pub segment_count: usize,
    pub total_size_bytes: u64,
    pub current_segment_id: u64,
}
