pub mod ab_testing;
pub mod ctr_tracker;
pub mod query_logger;
pub mod relevancy_learner;

pub use ab_testing::{ABTestEngine, Experiment, ExperimentResults, ExperimentStatus, Variant, VariantMetrics};
pub use ctr_tracker::CTRTracker;
pub use query_logger::{LatencyBucket, QueryLog, QueryLogger, QueryStorage, SearchType};
pub use relevancy_learner::{Impact, RelevancyLearner, RelevancySuggestion, SuggestionType};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Log channel closed")]
    LogChannelClosed,

    #[error("Database error: {0}")]
    DatabaseError(#[from] heed::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
