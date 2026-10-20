//! The inference service's composition-root library: model assessment (the environment and its
//! identity, and the automatic assessment pool), model residency on the engine worker, the model
//! store with the devices it serves, the shared resolved-configuration cache, the hardware
//! endpoint's provider and contained child processes.

pub mod assessment;
pub mod build_identity;
pub mod configurations;
pub mod hardware;
pub mod memory_domains;
pub mod model_store;
pub mod residency;
pub mod serving;
pub mod worker_process;

/// Makes the token ACN configures at runtime the only Hugging Face credential ICN presents: the
/// Hub client must never resolve one from the environment (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
/// `HF_TOKEN_PATH`) or the Hugging Face token file, and engine workers inherit none.
///
/// # Safety
///
/// Changes the process environment, so it must run before any other thread starts.
pub unsafe fn isolate_hub_credentials() {
    // SAFETY: the caller guarantees no other thread reads or writes the environment.
    unsafe {
        std::env::set_var("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
        for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"] {
            std::env::remove_var(name);
        }
    }
}

/// Run blocking work on Tokio's bounded blocking pool inside the caller's tracing span.
pub fn spawn_blocking_traced<F, R>(operation: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(operation))
}
