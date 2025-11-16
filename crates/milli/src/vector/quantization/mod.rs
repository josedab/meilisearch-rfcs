pub mod pq;
pub mod sq;

use serde::{Deserialize, Serialize};

pub use pq::{PQConfig, ProductQuantizer};
pub use sq::{SQConfig, ScalarQuantizer};

/// Quantization method selection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum QuantizationType {
    /// Binary quantization (existing)
    Binary,
    /// Product quantization
    Product {
        #[serde(flatten)]
        config: PQConfig,
    },
    /// Scalar quantization
    Scalar {
        #[serde(flatten)]
        config: SQConfig,
    },
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::Binary
    }
}

/// Errors that can occur during quantization
#[derive(Debug, thiserror::Error)]
pub enum QuantizationError {
    #[error("Empty training set provided")]
    EmptyTrainingSet,

    #[error("Dimension mismatch: vector dimension must be divisible by number of subvectors")]
    DimensionMismatch,

    #[error("Invalid bits per code: must be 4, 8, or 16")]
    InvalidBitsPerCode,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
