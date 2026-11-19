//! Assessment work-stealing pool helpers backed by [`seismic_threads::pool`].

pub use seismic_threads::pool::join;

/// Evaluate `work` on every item in `items` across the shared work pool using
/// dynamic atomic work stealing, entering a thread-local
/// [`crate::programs::graph::sizing::SizingScope`] on each participating
/// thread, and returning results (or the first error in input order) in exact
/// input order.
pub fn par_for_results<T: Sync, R: Send, E: Send, F>(
    items: &[T],
    work: F,
) -> Result<Vec<R>, E>
where
    F: Fn(&T) -> Result<R, E> + Sync,
{
    seismic_threads::pool::par_for_results_with(
        items,
        crate::programs::graph::sizing::SizingScope::enter,
        work,
    )
}
