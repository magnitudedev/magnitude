//! The inference service's composition-root library: model assessment (the environment and its
//! identity, and the automatic assessment pool), model residency on the engine
//! worker, the shared resolved-configuration cache, the hardware endpoint's provider and
//! contained child processes.

pub mod assessment;
pub mod build_identity;
pub mod configurations;
pub mod hardware;
pub mod memory_domains;
pub mod residency;
pub mod serving;
pub mod worker_process;

/// Run blocking work on Tokio's bounded blocking pool inside the caller's tracing span.
pub fn spawn_blocking_traced<F, R>(operation: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(operation))
}
