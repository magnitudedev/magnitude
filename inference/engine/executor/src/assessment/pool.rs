//! Assessment work-stealing pool helpers backed by [`seismic_threads::pool`].

/// Run `a` and `b` on the shared work pool, propagating the calling thread's
/// bound [`crate::programs::graph::sizing::SizingScope`] device properties to
/// both branches.
pub fn join<A: Send, B>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B,
) -> (A, B) {
    let device = crate::programs::graph::sizing::current_sizing_device();
    seismic_threads::pool::join(
        move || {
            let _scope = crate::programs::graph::sizing::SizingScope::enter_with_device(device);
            a()
        },
        move || {
            let _scope = crate::programs::graph::sizing::SizingScope::enter_with_device(device);
            b()
        },
    )
}

/// Evaluate `work` on every item in `items` across the shared work pool using
/// dynamic atomic work stealing, entering a thread-local
/// [`crate::programs::graph::sizing::SizingScope`] (preserving the calling
/// thread's device properties) on each participating thread, and returning
/// results (or the first error in input order) in exact input order.
pub fn par_for_results<T: Sync, R: Send, E: Send, F>(
    items: &[T],
    work: F,
) -> Result<Vec<R>, E>
where
    F: Fn(&T) -> Result<R, E> + Sync,
{
    let device = crate::programs::graph::sizing::current_sizing_device();
    seismic_threads::pool::par_for_results_with(
        items,
        move || crate::programs::graph::sizing::SizingScope::enter_with_device(device),
        work,
    )
}
