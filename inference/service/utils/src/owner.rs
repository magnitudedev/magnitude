//! Tasks that own a part of the service's state: what they owe it is owed by no one else.

use std::future::Future;

/// Run `owner`, a task that owns part of the service's state as `name` (for the service's life,
/// or until it has brought that state where it must go), on the runtime. Nobody waits for it as
/// its own work ([`seismic_threads::detached`]): its panic is an invariant violation that leaves
/// what it owns ownerless, so it ends the process at the panic, its report naming `name` (the
/// process's workers end with it, as they do on any exit of their parent), rather than ending
/// with the task while the service serves on without it. Its abort, by whoever holds the
/// returned handle, ends it as cancellation, never as a panic.
pub fn spawn<F>(name: &'static str, owner: F) -> Owner
where
    F: Future<Output = ()> + Send + 'static,
{
    Owner { task: tokio::spawn(seismic_threads::detached(name, owner)) }
}

/// A running owner ([`spawn`]): dropping the handle detaches it.
#[derive(Debug)]
pub struct Owner {
    task: tokio::task::JoinHandle<()>,
}

impl Owner {
    /// End the owner as cancellation, at its next await.
    pub fn abort(&self) {
        self.task.abort();
    }

    /// Return once the owner has ended, run to its end or cancelled: its future is dropped, so
    /// nothing of it runs after.
    pub async fn ended(self) {
        // An owner's panic ends the process at the panic, so the task ends only by returning or
        // by cancellation, which ends this wait alike.
        let _ = self.task.await;
    }
}
