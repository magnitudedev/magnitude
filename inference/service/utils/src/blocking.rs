//! Blocking work on Tokio's blocking pool, joined as the work's own value.

use std::future::Future;

/// Run `operation` on Tokio's bounded blocking pool inside the caller's tracing span, and resolve
/// to what it returns.
///
/// The task starts at once; dropping the returned future detaches it to run to completion. A
/// panic in `operation` is a defect and continues in the awaiting task. The task is never aborted,
/// and the runtime cancels a blocking task only while shutting down, when the awaiting task is
/// dropped with it: a cancelled join is an invariant violation.
pub fn run<F, R>(operation: F) -> impl Future<Output = R> + Send + 'static
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    let task = tokio::task::spawn_blocking(move || span.in_scope(operation));
    async move {
        match task.await {
            Ok(value) => value,
            Err(join) => match join.try_into_panic() {
                Ok(panic) => std::panic::resume_unwind(panic),
                Err(join) => panic!("a blocking task was cancelled while awaited: {join}"),
            },
        }
    }
}
