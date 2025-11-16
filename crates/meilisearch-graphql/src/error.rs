use async_graphql::{Error as GraphQLError, ErrorExtensions};
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Task not found: {0}")]
    TaskNotFound(u32),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Search error: {0}")]
    SearchError(String),

    #[error("Index scheduler error: {0}")]
    IndexSchedulerError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl ErrorExtensions for Error {
    fn extend(&self) -> GraphQLError {
        GraphQLError::new(format!("{}", self)).extend_with(|_err, e| match self {
            Error::IndexNotFound(_) => {
                e.set("code", "index_not_found");
            }
            Error::TaskNotFound(_) => {
                e.set("code", "task_not_found");
            }
            Error::InvalidQuery(_) => {
                e.set("code", "invalid_query");
            }
            Error::SearchError(_) => {
                e.set("code", "search_error");
            }
            Error::IndexSchedulerError(_) => {
                e.set("code", "index_scheduler_error");
            }
            Error::InternalError(_) => {
                e.set("code", "internal_error");
            }
            Error::AuthenticationError(_) => {
                e.set("code", "authentication_error");
            }
            Error::SerializationError(_) => {
                e.set("code", "serialization_error");
            }
        })
    }
}

impl From<index_scheduler::Error> for Error {
    fn from(err: index_scheduler::Error) -> Self {
        Error::IndexSchedulerError(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::SerializationError(err.to_string())
    }
}
