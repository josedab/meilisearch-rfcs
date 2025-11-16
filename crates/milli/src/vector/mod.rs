pub mod db;
pub mod distance;
mod distribution;
pub mod embedder;
mod embeddings;
pub mod error;
pub mod extractor;
pub mod index;
pub mod json_template;
pub mod multi_vector;
pub mod parsed_vectors;
pub mod quantization;
mod runtime;
pub mod search_config;
pub mod session;
pub mod settings;
mod store;

pub use self::error::Error;

pub type Embedding = Vec<f32>;

pub use distance::{DistanceFunction, VectorDistanceMetric};
pub use distribution::DistributionShift;
pub use embedder::{Embedder, EmbedderOptions, EmbeddingConfig, SearchQuery};
pub use embeddings::Embeddings;
pub use index::{IVFHNSWConfig, IVFHNSWIndex, IVFIndexError, IVFIndexStats};
pub use multi_vector::{FusionStrategy, MultiVectorDocument, MultiVectorQuery, fuse_results};
pub use quantization::{PQConfig, ProductQuantizer, QuantizationError, QuantizationType, SQConfig, ScalarQuantizer};
pub use runtime::{RuntimeEmbedder, RuntimeEmbedders, RuntimeFragment};
pub use search_config::VectorSearchConfig;
pub use store::{VectorStore, VectorStoreBackend, VectorStoreStats};

pub const REQUEST_PARALLELISM: usize = 40;

/// Whether CUDA is supported in this version of Meilisearch.
pub const fn is_cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}
