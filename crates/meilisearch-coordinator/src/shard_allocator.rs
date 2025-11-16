use crate::error::{Error, Result};
use crate::types::{IndexUid, NodeId, NodeInfo, ShardAssignment, ShardState};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

/// Shard allocator using consistent hashing
pub struct ShardAllocator {
    /// Number of shards per index (default)
    pub default_shard_count: usize,
    /// Replication factor (default)
    pub default_replication_factor: usize,
    /// Virtual nodes per physical node (for balanced distribution)
    pub virtual_nodes: usize,
}

impl ShardAllocator {
    /// Create a new shard allocator
    pub fn new(
        default_shard_count: usize,
        default_replication_factor: usize,
        virtual_nodes: usize,
    ) -> Self {
        Self {
            default_shard_count,
            default_replication_factor,
            virtual_nodes,
        }
    }

    /// Allocate shards for an index across available nodes
    pub fn allocate_shards(
        &self,
        index_uid: &IndexUid,
        shard_count: usize,
        replication_factor: usize,
        nodes: &[NodeInfo],
    ) -> Result<Vec<ShardAssignment>> {
        if nodes.is_empty() {
            return Err(Error::NoNodesAvailable);
        }

        let mut assignments = Vec::new();

        for shard_id in 0..shard_count {
            // Compute primary node using consistent hashing
            let primary = self.select_primary_node(index_uid, shard_id, nodes)?;

            // Select replicas (different from primary)
            let replicas =
                self.select_replica_nodes(index_uid, shard_id, primary, nodes, replication_factor)?;

            assignments.push(ShardAssignment {
                shard_id,
                primary,
                replicas,
                state: ShardState::Initializing,
            });
        }

        Ok(assignments)
    }

    /// Select primary node for a shard using consistent hashing
    fn select_primary_node(
        &self,
        index_uid: &IndexUid,
        shard_id: usize,
        nodes: &[NodeInfo],
    ) -> Result<NodeId> {
        if nodes.is_empty() {
            return Err(Error::NoNodesAvailable);
        }

        // Hash index_uid + shard_id to get consistent node
        let hash = self.hash_shard(index_uid, shard_id);

        // Find node with closest hash value (consistent hashing)
        let ring = self.build_hash_ring(nodes);
        let node_id = ring.find_node(hash)?;

        Ok(node_id)
    }

    /// Select replica nodes (must be different from primary)
    fn select_replica_nodes(
        &self,
        index_uid: &IndexUid,
        shard_id: usize,
        primary: NodeId,
        nodes: &[NodeInfo],
        replication_factor: usize,
    ) -> Result<Vec<NodeId>> {
        let mut replicas = Vec::new();
        let ring = self.build_hash_ring(nodes);

        // Start with hash of shard + replica index
        for replica_idx in 0..replication_factor {
            let hash = self.hash_shard_replica(index_uid, shard_id, replica_idx);

            // Find next available node that's not primary and not already a replica
            if let Some(node_id) = ring.find_next_available_node(hash, primary, &replicas) {
                replicas.push(node_id);
            } else if nodes.len() > replicas.len() + 1 {
                // Fallback: pick any node not already used
                for node in nodes {
                    if node.id != primary && !replicas.contains(&node.id) {
                        replicas.push(node.id);
                        break;
                    }
                }
            }
        }

        Ok(replicas)
    }

    /// Build consistent hash ring from nodes
    fn build_hash_ring(&self, nodes: &[NodeInfo]) -> HashRing {
        let mut ring = HashRing::new();

        for node in nodes {
            // Add virtual nodes for better distribution
            for v in 0..self.virtual_nodes {
                let virtual_hash = self.hash_virtual_node(node.id, v);
                ring.add_node(virtual_hash, node.id);
            }
        }

        ring
    }

    /// Hash a shard identifier
    fn hash_shard(&self, index_uid: &IndexUid, shard_id: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        index_uid.hash(&mut hasher);
        shard_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a shard replica identifier
    fn hash_shard_replica(&self, index_uid: &IndexUid, shard_id: usize, replica_idx: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        index_uid.hash(&mut hasher);
        shard_id.hash(&mut hasher);
        replica_idx.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a virtual node
    fn hash_virtual_node(&self, node_id: NodeId, virtual_idx: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        node_id.hash(&mut hasher);
        virtual_idx.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ShardAllocator {
    fn default() -> Self {
        Self::new(4, 2, 150)
    }
}

/// Consistent hash ring
pub struct HashRing {
    nodes: BTreeMap<u64, NodeId>,
}

impl HashRing {
    /// Create a new hash ring
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    /// Add a node to the ring
    pub fn add_node(&mut self, hash: u64, node_id: NodeId) {
        self.nodes.insert(hash, node_id);
    }

    /// Find the node responsible for a given hash
    pub fn find_node(&self, hash: u64) -> Result<NodeId> {
        // Find first node with hash >= query hash
        self.nodes
            .range(hash..)
            .next()
            .or_else(|| self.nodes.iter().next())
            .map(|(_, &node_id)| node_id)
            .ok_or(Error::NoNodesAvailable)
    }

    /// Find next available node that's not in the excluded list
    pub fn find_next_available_node(
        &self,
        hash: u64,
        primary: NodeId,
        excluded: &[NodeId],
    ) -> Option<NodeId> {
        // Try to find a node starting from hash
        for (_, &node_id) in self.nodes.range(hash..) {
            if node_id != primary && !excluded.contains(&node_id) {
                return Some(node_id);
            }
        }

        // Wrap around to beginning
        for (_, &node_id) in self.nodes.iter() {
            if node_id != primary && !excluded.contains(&node_id) {
                return Some(node_id);
            }
        }

        None
    }
}

impl Default for HashRing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeRole, NodeStatus, ResourceCapacity};

    fn create_test_nodes(count: usize) -> Vec<NodeInfo> {
        (0..count)
            .map(|i| NodeInfo {
                id: NodeId(i as u64),
                address: format!("node-{}:7700", i),
                status: NodeStatus::Healthy,
                capacity: ResourceCapacity {
                    available_memory: 1024 * 1024 * 1024,
                    available_disk: 100 * 1024 * 1024 * 1024,
                    cpu_cores: 4,
                },
                shard_count: 0,
                role: NodeRole::Data,
            })
            .collect()
    }

    #[test]
    fn test_shard_allocation() {
        let allocator = ShardAllocator::default();
        let nodes = create_test_nodes(5);

        let assignments = allocator
            .allocate_shards("test-index", 4, 2, &nodes)
            .unwrap();

        assert_eq!(assignments.len(), 4);

        for assignment in &assignments {
            // Primary and replicas should be different
            assert!(!assignment.replicas.contains(&assignment.primary));

            // Should have correct number of replicas
            assert_eq!(assignment.replicas.len(), 2);

            // Replicas should be unique
            let unique_replicas: std::collections::HashSet<_> =
                assignment.replicas.iter().collect();
            assert_eq!(unique_replicas.len(), assignment.replicas.len());
        }
    }

    #[test]
    fn test_consistent_hashing() {
        let allocator = ShardAllocator::default();
        let nodes = create_test_nodes(5);

        // Same index and shard should map to same node
        let node1 = allocator.select_primary_node("test-index", 0, &nodes).unwrap();
        let node2 = allocator.select_primary_node("test-index", 0, &nodes).unwrap();

        assert_eq!(node1, node2);
    }

    #[test]
    fn test_no_nodes_available() {
        let allocator = ShardAllocator::default();
        let nodes = vec![];

        let result = allocator.allocate_shards("test-index", 4, 2, &nodes);
        assert!(matches!(result, Err(Error::NoNodesAvailable)));
    }
}
