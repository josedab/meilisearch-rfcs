use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use heed::RoTxn;
use roaring::RoaringBitmap;

use crate::{DocumentId, Index, Result};

/// In-memory buffer for recent document updates
///
/// The WriteBuffer manages active and immutable memtables, providing non-blocking
/// writes and real-time search across all in-memory data.
pub struct WriteBuffer {
    /// Active memtable accepting writes
    active: Arc<RwLock<MemTable>>,
    /// Immutable memtables pending merge to LMDB
    immutable: Arc<RwLock<Vec<Arc<MemTable>>>>,
    /// Maximum size before flush (bytes)
    max_size: usize,
    /// Flush trigger
    flush_tx: mpsc::Sender<Arc<MemTable>>,
}

/// In-memory table holding document updates
///
/// MemTables store documents and index updates before they are merged to LMDB.
/// Once frozen, they become immutable and are queued for background merging.
pub struct MemTable {
    /// Document storage (docid -> serialized document)
    documents: HashMap<DocumentId, Vec<u8>>,
    /// Inverted index updates (word -> docids)
    word_docids: HashMap<String, RoaringBitmap>,
    /// Deleted document IDs
    deletions: RoaringBitmap,
    /// Approximate memory usage
    pub size_bytes: usize,
    /// Creation timestamp
    created_at: Instant,
    /// Frozen flag (no more writes accepted)
    pub frozen: bool,
}

impl WriteBuffer {
    /// Create a new WriteBuffer with specified maximum size
    ///
    /// Returns the buffer and a receiver channel for flush notifications
    pub fn new(max_size: usize) -> (Self, mpsc::Receiver<Arc<MemTable>>) {
        let (flush_tx, flush_rx) = mpsc::channel();

        let buffer = Self {
            active: Arc::new(RwLock::new(MemTable::new())),
            immutable: Arc::new(RwLock::new(Vec::new())),
            max_size,
            flush_tx,
        };

        (buffer, flush_rx)
    }

    /// Add or update a document (non-blocking)
    ///
    /// This method writes to the active memtable and triggers rotation if needed
    pub fn write(&self, doc_id: DocumentId, document: Vec<u8>) -> Result<()> {
        let mut active = self.active.write().unwrap();

        // Add document to memtable
        active.add_document(doc_id, document)?;

        // Check if flush needed
        if active.size_bytes >= self.max_size {
            drop(active); // Release lock
            self.rotate_memtable()?;
        }

        Ok(())
    }

    /// Delete a document
    ///
    /// Marks the document as deleted in the active memtable
    pub fn delete(&self, doc_id: DocumentId) -> Result<()> {
        let mut active = self.active.write().unwrap();
        active.mark_deleted(doc_id);

        if active.size_bytes >= self.max_size {
            drop(active);
            self.rotate_memtable()?;
        }

        Ok(())
    }

    /// Rotate active memtable to immutable
    ///
    /// Creates a new active memtable and moves the old one to the immutable list
    fn rotate_memtable(&self) -> Result<()> {
        // Create new active memtable
        let new_active = MemTable::new();

        // Swap active with new
        let old_active = {
            let mut active_lock = self.active.write().unwrap();
            std::mem::replace(&mut *active_lock, new_active)
        };

        // Freeze old memtable
        let mut old = old_active;
        old.frozen = true;
        let old_arc = Arc::new(old);

        // Add to immutable list
        {
            let mut immutable = self.immutable.write().unwrap();
            immutable.push(old_arc.clone());
        }

        // Trigger flush
        let _ = self.flush_tx.send(old_arc);

        Ok(())
    }

    /// Search across active + immutable memtables + LMDB
    ///
    /// Returns documents matching the query, prioritizing more recent updates
    pub fn search(
        &self,
        rtxn: &RoTxn,
        query: &str,
        index: &Index,
    ) -> Result<Vec<(DocumentId, Vec<u8>)>> {
        let mut results = HashMap::new();

        // 1. Search active memtable
        {
            let active = self.active.read().unwrap();
            for (doc_id, doc) in active.search_documents(query) {
                results.insert(doc_id, doc.clone());
            }
        }

        // 2. Search immutable memtables (newest first)
        {
            let immutable = self.immutable.read().unwrap();
            for memtable in immutable.iter().rev() {
                for (doc_id, doc) in memtable.search_documents(query) {
                    results.entry(doc_id).or_insert_with(|| doc.clone());
                }
            }
        }

        // 3. Search LMDB (for documents not in memtables)
        let memtable_docids: RoaringBitmap = results.keys().copied().collect();

        // Note: This is a simplified search. In production, we would integrate
        // with the full search pipeline in crates/milli/src/search/
        let lmdb_results = self.search_lmdb(rtxn, index, query)?;

        for (doc_id, doc) in lmdb_results {
            if !memtable_docids.contains(doc_id) {
                results.insert(doc_id, doc);
            }
        }

        Ok(results.into_iter().collect())
    }

    /// Helper to search LMDB backend
    ///
    /// This is a placeholder that would integrate with the actual search implementation
    fn search_lmdb(
        &self,
        _rtxn: &RoTxn,
        _index: &Index,
        _query: &str,
    ) -> Result<Vec<(DocumentId, Vec<u8>)>> {
        // TODO: Integrate with actual search pipeline
        // This would call into the existing search infrastructure
        Ok(Vec::new())
    }

    /// Force flush of the active memtable
    pub fn force_flush(&self) -> Result<()> {
        self.rotate_memtable()
    }

    /// Get statistics about the write buffer
    pub fn stats(&self) -> WriteBufferStats {
        let active_size = self.active.read().unwrap().size_bytes;
        let immutable_count = self.immutable.read().unwrap().len();
        let immutable_total_size: usize = self
            .immutable
            .read()
            .unwrap()
            .iter()
            .map(|m| m.size_bytes)
            .sum();

        WriteBufferStats {
            active_size_bytes: active_size,
            immutable_count,
            immutable_total_size_bytes: immutable_total_size,
            total_size_bytes: active_size + immutable_total_size,
        }
    }

    /// Remove immutable memtables from the list after successful merge
    pub fn remove_immutable(&self, count: usize) {
        let mut immutable = self.immutable.write().unwrap();
        if count <= immutable.len() {
            immutable.drain(0..count);
        }
    }
}

impl MemTable {
    /// Create a new empty MemTable
    fn new() -> Self {
        Self {
            documents: HashMap::new(),
            word_docids: HashMap::new(),
            deletions: RoaringBitmap::new(),
            size_bytes: 0,
            created_at: Instant::now(),
            frozen: false,
        }
    }

    /// Add a document to the memtable
    ///
    /// Extracts searchable words and updates the inverted index
    fn add_document(&mut self, doc_id: DocumentId, document: Vec<u8>) -> Result<()> {
        if self.frozen {
            return Err(crate::Error::InternalError(crate::InternalError::AbortedIndexation));
        }

        // Extract words for inverted index
        let words = extract_words(&document)?;
        for word in words {
            self.word_docids
                .entry(word)
                .or_insert_with(RoaringBitmap::new)
                .insert(doc_id);
        }

        // Store document
        let doc_size = document.len();
        self.documents.insert(doc_id, document);
        self.size_bytes += doc_size + 32; // 32 bytes overhead estimate

        Ok(())
    }

    /// Mark a document as deleted
    fn mark_deleted(&mut self, doc_id: DocumentId) {
        self.deletions.insert(doc_id);
        self.size_bytes += 4; // Approximate size of deletion marker
    }

    /// Search for documents matching a query
    ///
    /// Returns an iterator over matching documents
    fn search_documents(&self, query: &str) -> impl Iterator<Item = (DocumentId, &Vec<u8>)> {
        let matching_docids = self
            .word_docids
            .get(query)
            .map(|bitmap| bitmap.clone())
            .unwrap_or_default();

        let deletions = self.deletions.clone();
        let docs = &self.documents;

        matching_docids
            .into_iter()
            .filter(move |doc_id| !deletions.contains(*doc_id))
            .filter_map(move |doc_id| docs.get(&doc_id).map(|doc| (doc_id, doc)))
    }

    /// Get all documents in the memtable
    pub fn iter_documents(&self) -> impl Iterator<Item = (DocumentId, &Vec<u8>)> {
        let deletions = self.deletions.clone();
        self.documents
            .iter()
            .filter(move |(doc_id, _)| !deletions.contains(**doc_id))
            .map(|(doc_id, doc)| (*doc_id, doc))
    }

    /// Get all deletions
    pub fn get_deletions(&self) -> &RoaringBitmap {
        &self.deletions
    }

    /// Get word to docids mapping
    pub fn get_word_docids(&self) -> &HashMap<String, RoaringBitmap> {
        &self.word_docids
    }
}

/// Statistics about the write buffer state
#[derive(Debug, Clone)]
pub struct WriteBufferStats {
    pub active_size_bytes: usize,
    pub immutable_count: usize,
    pub immutable_total_size_bytes: usize,
    pub total_size_bytes: usize,
}

/// Extract searchable words from a document
///
/// This is a simplified word extraction. In production, this would use
/// the full tokenization pipeline from charabia.
fn extract_words(document: &[u8]) -> Result<Vec<String>> {
    // Parse the document as JSON
    let doc: serde_json::Value = serde_json::from_slice(document).map_err(crate::InternalError::SerdeJson)?;

    let mut words = Vec::new();

    // Recursively extract string values
    extract_strings_recursive(&doc, &mut words);

    // Normalize and tokenize (simplified)
    let normalized_words: Vec<String> = words
        .into_iter()
        .flat_map(|s| {
            s.to_lowercase()
                .split_whitespace()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(normalized_words)
}

/// Recursively extract string values from JSON
fn extract_strings_recursive(value: &serde_json::Value, words: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => words.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                extract_strings_recursive(item, words);
            }
        }
        serde_json::Value::Object(obj) => {
            for (_key, val) in obj {
                extract_strings_recursive(val, words);
            }
        }
        _ => {}
    }
}
