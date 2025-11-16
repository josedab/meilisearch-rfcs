use super::*;
use std::sync::Arc;
use tempfile::TempDir;

/// Test WriteBuffer basic functionality
#[tokio::test]
async fn test_write_buffer_basic_operations() {
    let max_size = 1024 * 1024; // 1MB
    let (buffer, mut _flush_rx) = WriteBuffer::new(max_size);

    // Create a test document
    let doc_id = 1;
    let document = serde_json::json!({
        "id": 1,
        "title": "Test Document",
        "content": "This is a test document"
    });
    let doc_bytes = serde_json::to_vec(&document).unwrap();

    // Write document
    buffer.write(doc_id, doc_bytes.clone()).await.unwrap();

    // Check stats
    let stats = buffer.stats();
    assert!(stats.active_size_bytes > 0);
    assert_eq!(stats.immutable_count, 0);
}

/// Test MemTable operations
#[test]
fn test_memtable_add_and_search() {
    let mut memtable = MemTable::new();

    // Add a document
    let doc_id = 1;
    let document = serde_json::json!({
        "id": 1,
        "title": "Test",
        "content": "Hello World"
    });
    let doc_bytes = serde_json::to_vec(&document).unwrap();

    memtable.add_document(doc_id, doc_bytes).unwrap();

    // Check size increased
    assert!(memtable.size_bytes > 0);

    // Search for document
    let results: Vec<_> = memtable.search_documents("hello").collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, doc_id);
}

/// Test MemTable deletion
#[test]
fn test_memtable_deletion() {
    let mut memtable = MemTable::new();

    // Add documents
    let doc1 = serde_json::json!({"id": 1, "title": "Doc 1"});
    let doc2 = serde_json::json!({"id": 2, "title": "Doc 2"});

    memtable.add_document(1, serde_json::to_vec(&doc1).unwrap()).unwrap();
    memtable.add_document(2, serde_json::to_vec(&doc2).unwrap()).unwrap();

    // Delete doc 1
    memtable.mark_deleted(1);

    // Verify deletion
    assert!(memtable.get_deletions().contains(1));
    assert!(!memtable.get_deletions().contains(2));

    // Check iteration skips deleted
    let docs: Vec<_> = memtable.iter_documents().collect();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].0, 2);
}

/// Test frozen memtable rejects writes
#[test]
fn test_frozen_memtable_rejects_writes() {
    let mut memtable = MemTable::new();
    memtable.frozen = true;

    let doc = serde_json::json!({"id": 1});
    let result = memtable.add_document(1, serde_json::to_vec(&doc).unwrap());

    assert!(result.is_err());
}

/// Test WAL basic operations
#[test]
fn test_wal_append_and_replay() {
    let temp_dir = TempDir::new().unwrap();
    let wal_dir = temp_dir.path().to_path_buf();

    let wal = WriteAheadLog::new(wal_dir, 100 * 1024 * 1024).unwrap();

    // Append entries
    let entry1 = WALEntry::DocumentAdd {
        doc_id: 1,
        document: vec![1, 2, 3],
        timestamp: 1000,
    };
    let entry2 = WALEntry::DocumentDelete {
        doc_id: 2,
        timestamp: 2000,
    };

    let lsn1 = wal.append(entry1).unwrap();
    let lsn2 = wal.append(entry2).unwrap();

    assert!(lsn2 > lsn1);

    // Replay
    let mut count = 0;
    wal.replay(|entry| {
        count += 1;
        match entry {
            WALEntry::DocumentAdd { doc_id, .. } => assert_eq!(doc_id, 1),
            WALEntry::DocumentDelete { doc_id, .. } => assert_eq!(doc_id, 2),
            _ => panic!("Unexpected entry type"),
        }
        Ok(())
    })
    .unwrap();

    assert_eq!(count, 2);
}

/// Test WAL segment rotation
#[test]
fn test_wal_segment_rotation() {
    let temp_dir = TempDir::new().unwrap();
    let wal_dir = temp_dir.path().to_path_buf();

    // Small segment size to force rotation
    let max_segment_size = 100; // 100 bytes
    let wal = WriteAheadLog::new(wal_dir.clone(), max_segment_size).unwrap();

    // Write enough entries to trigger rotation
    for i in 0..10 {
        let entry = WALEntry::DocumentAdd {
            doc_id: i,
            document: vec![0u8; 50], // 50 bytes
            timestamp: i as u64,
        };
        wal.append(entry).unwrap();
    }

    // Check that multiple segments were created
    let stats = wal.stats().unwrap();
    assert!(stats.segment_count > 1);
}

/// Test WAL truncation
#[test]
fn test_wal_truncation() {
    let temp_dir = TempDir::new().unwrap();
    let wal_dir = temp_dir.path().to_path_buf();

    let max_segment_size = 100;
    let wal = WriteAheadLog::new(wal_dir.clone(), max_segment_size).unwrap();

    // Write entries to create multiple segments
    for i in 0..10 {
        let entry = WALEntry::DocumentAdd {
            doc_id: i,
            document: vec![0u8; 50],
            timestamp: i as u64,
        };
        wal.append(entry).unwrap();
    }

    let stats_before = wal.stats().unwrap();
    let initial_segments = stats_before.segment_count;

    // Truncate (LSN with high segment ID)
    let lsn_to_truncate = (2u64 << 32) | 100;
    wal.truncate_before(lsn_to_truncate).unwrap();

    // Verify segments were removed
    let stats_after = wal.stats().unwrap();
    assert!(stats_after.segment_count < initial_segments);
}

/// Test MergeStrategy parsing
#[test]
fn test_merge_strategy_from_str() {
    assert!(matches!(
        MergeStrategy::from_str("immediate"),
        Some(MergeStrategy::Immediate)
    ));

    assert!(matches!(
        MergeStrategy::from_str("batched"),
        Some(MergeStrategy::Batched { count: 5 })
    ));

    assert!(matches!(
        MergeStrategy::from_str("periodic"),
        Some(MergeStrategy::Periodic { .. })
    ));

    assert!(matches!(
        MergeStrategy::from_str("adaptive"),
        Some(MergeStrategy::Adaptive)
    ));

    assert!(MergeStrategy::from_str("invalid").is_none());
}

/// Test RecoveryStats
#[test]
fn test_recovery_stats_default() {
    let stats = RecoveryStats::default();
    assert_eq!(stats.documents_recovered, 0);
    assert_eq!(stats.deletions_recovered, 0);
    assert_eq!(stats.memtable_flushes, 0);
    assert_eq!(stats.last_checkpoint, None);
}

/// Test word extraction from documents
#[test]
fn test_word_extraction() {
    let doc = serde_json::json!({
        "title": "Hello World",
        "content": "This is a test document"
    });
    let doc_bytes = serde_json::to_vec(&doc).unwrap();

    let words = super::write_buffer::extract_words(&doc_bytes).unwrap();

    assert!(words.contains(&"hello".to_string()));
    assert!(words.contains(&"world".to_string()));
    assert!(words.contains(&"test".to_string()));
    assert!(words.contains(&"document".to_string()));
}

/// Integration test: Write buffer rotation
#[tokio::test]
async fn test_write_buffer_rotation() {
    let max_size = 100; // Small size to force rotation
    let (buffer, mut flush_rx) = WriteBuffer::new(max_size);

    // Write enough to trigger rotation
    let doc = serde_json::json!({"id": 1, "content": "A".repeat(200)});
    let doc_bytes = serde_json::to_vec(&doc).unwrap();

    buffer.write(1, doc_bytes).await.unwrap();

    // Check that a memtable was flushed
    let flushed = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        flush_rx.recv()
    ).await;

    assert!(flushed.is_ok());
    assert!(flushed.unwrap().is_some());

    // Verify immutable count in stats
    let stats = buffer.stats();
    assert!(stats.immutable_count > 0 || stats.active_size_bytes > 0);
}

/// Test WriteBuffer stats
#[tokio::test]
async fn test_write_buffer_stats() {
    let (buffer, _rx) = WriteBuffer::new(1024 * 1024);

    // Initial stats
    let stats = buffer.stats();
    assert_eq!(stats.active_size_bytes, 0);
    assert_eq!(stats.immutable_count, 0);
    assert_eq!(stats.total_size_bytes, 0);

    // Add a document
    let doc = serde_json::json!({"id": 1, "content": "test"});
    buffer.write(1, serde_json::to_vec(&doc).unwrap()).await.unwrap();

    // Check stats updated
    let stats = buffer.stats();
    assert!(stats.active_size_bytes > 0);
    assert_eq!(stats.total_size_bytes, stats.active_size_bytes);
}

/// Test concurrent writes to WriteBuffer
#[tokio::test]
async fn test_concurrent_writes() {
    let (buffer, _rx) = WriteBuffer::new(10 * 1024 * 1024);
    let buffer = Arc::new(buffer);

    let mut handles = vec![];

    // Spawn multiple tasks writing concurrently
    for i in 0..10 {
        let buffer_clone = buffer.clone();
        let handle = tokio::spawn(async move {
            let doc = serde_json::json!({"id": i, "content": format!("Document {}", i)});
            buffer_clone.write(i, serde_json::to_vec(&doc).unwrap()).await
        });
        handles.push(handle);
    }

    // Wait for all writes
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Verify stats
    let stats = buffer.stats();
    assert!(stats.total_size_bytes > 0);
}

/// Test WAL stats
#[test]
fn test_wal_stats() {
    let temp_dir = TempDir::new().unwrap();
    let wal = WriteAheadLog::new(temp_dir.path().to_path_buf(), 100 * 1024 * 1024).unwrap();

    let stats = wal.stats().unwrap();
    assert_eq!(stats.segment_count, 1);
    assert_eq!(stats.current_segment_id, 0);

    // Write some entries
    for i in 0..5 {
        let entry = WALEntry::DocumentAdd {
            doc_id: i,
            document: vec![0u8; 100],
            timestamp: i as u64,
        };
        wal.append(entry).unwrap();
    }

    let stats = wal.stats().unwrap();
    assert!(stats.total_size_bytes > 0);
}
