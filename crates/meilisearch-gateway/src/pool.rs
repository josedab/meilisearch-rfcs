use crate::error::{Error, Result};
use crate::types::{SearchQuery, ShardSearchResult};
use dashmap::DashMap;
use meilisearch_coordinator::NodeId;
use std::sync::Arc;

/// Connection pool for data nodes
pub struct NodeConnectionPool {
    /// HTTP clients for each node
    clients: Arc<DashMap<NodeId, reqwest::Client>>,
    /// Node addresses
    addresses: Arc<DashMap<NodeId, String>>,
}

impl NodeConnectionPool {
    /// Create a new node connection pool
    pub fn new() -> Self {
        Self {
            clients: Arc::new(DashMap::new()),
            addresses: Arc::new(DashMap::new()),
        }
    }

    /// Register a node with its address
    pub fn register_node(&self, node_id: NodeId, address: String) {
        self.addresses.insert(node_id, address);
        self.clients.insert(node_id, reqwest::Client::new());
    }

    /// Unregister a node
    pub fn unregister_node(&self, node_id: NodeId) {
        self.addresses.remove(&node_id);
        self.clients.remove(&node_id);
    }

    /// Send a search query to a node
    pub async fn send_query(
        &self,
        node_id: NodeId,
        index_uid: &str,
        query: &SearchQuery,
    ) -> Result<ShardSearchResult> {
        let client = self
            .clients
            .get(&node_id)
            .ok_or_else(|| Error::NodeUnavailable(node_id.to_string()))?;

        let address = self
            .addresses
            .get(&node_id)
            .ok_or_else(|| Error::NodeUnavailable(node_id.to_string()))?;

        let url = format!("http://{}/indexes/{}/search", address.value(), index_uid);

        let response = client
            .post(&url)
            .json(query)
            .send()
            .await
            .map_err(|e| Error::NodeUnavailable(format!("{}: {}", node_id, e)))?;

        if !response.status().is_success() {
            return Err(Error::NodeUnavailable(format!(
                "{}: HTTP {}",
                node_id,
                response.status()
            )));
        }

        let result = response.json().await?;
        Ok(result)
    }

    /// Send index documents request to a node
    pub async fn send_index_request(
        &self,
        node_id: NodeId,
        index_uid: &str,
        documents: Vec<serde_json::Value>,
    ) -> Result<crate::types::ShardTaskInfo> {
        let client = self
            .clients
            .get(&node_id)
            .ok_or_else(|| Error::NodeUnavailable(node_id.to_string()))?;

        let address = self
            .addresses
            .get(&node_id)
            .ok_or_else(|| Error::NodeUnavailable(node_id.to_string()))?;

        let url = format!("http://{}/indexes/{}/documents", address.value(), index_uid);

        let response = client
            .post(&url)
            .json(&documents)
            .send()
            .await
            .map_err(|e| Error::NodeUnavailable(format!("{}: {}", node_id, e)))?;

        if !response.status().is_success() {
            return Err(Error::NodeUnavailable(format!(
                "{}: HTTP {}",
                node_id,
                response.status()
            )));
        }

        let result = response.json().await?;
        Ok(result)
    }
}

impl Default for NodeConnectionPool {
    fn default() -> Self {
        Self::new()
    }
}
