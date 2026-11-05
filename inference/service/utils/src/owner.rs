//! Tasks that own a part of the service's state: what they owe it is owed by no one else.

use std::future::Future;

/// Run `owner`, a task that owns part of the service's state as `name` (for the service's life,
/// or until it has brought that state where it must go), on the runtime. Its panic is an
/// invariant violation that leaves what it owns ownerless, so it ends the process (whose workers
/// end with it, as they do on any exit of their parent) rather than ending with the detached task
/// while the service serves on without it. Its abort, by whoever holds the returned handle, ends
/// it as cancellation, never as a panic.
pub fn spawn<F>(name: &'static str, owner: F) -> Owner
where
    F: Future<Output = ()> + Send + 'static,
{
    let task = tokio::spawn(owner);
    let abort = task.abort_handle();
    let ended = tokio::spawn(async move {
        if let Err(join) = task.await {
            if join.is_panic() {
                tracing::error!(owner = name, "an owner of the service panicked; ending the process");
                std::process::abort();
            }
        }
    });
    Owner { abort, ended }
}

/// A running owner ([`spawn`]): dropping the handle detaches it.
#[derive(Debug)]
pub struct Owner {
    abort: tokio::task::AbortHandle,
    /// Ends once the owner has, run to its end or cancelled; a panic ends the process first.
    ended: tokio::task::JoinHandle<()>,
}

impl Owner {
    /// End the owner as cancellation, at its next await.
    pub fn abort(&self) {
        self.abort.abort();
    }

    /// Return once the owner has ended, run to its end or cancelled: its future is dropped, so
    /// nothing of it runs after.
    pub async fn ended(self) {
        // The watch never panics, as an owner's panic ends the process first; only the
        // runtime's shutdown cancels it, which ends this wait with it.
        let _ = self.ended.await;
    }
}
