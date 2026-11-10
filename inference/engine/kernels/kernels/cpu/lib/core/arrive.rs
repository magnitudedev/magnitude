// Arrival counting across the work items of one launch, over `sync` scratch
// (zero whenever a launch starts; see the `sync` scratch contract); the CPU
// form of `cuda/lib/core/arrive.cuh`. A work item that has finished its
// stores arrives; exactly one item, the last to arrive, is told so, and it
// restores the counter to zero, so the next launch over the same scratch
// starts from zero. No item ever waits for another.
//
// The arrival is a release and the last item's is an acquire: every item's
// stores happen before the last item's reads.

use std::sync::atomic::{AtomicU32, Ordering};

/// Whether this item is the last of `expected` to arrive at `counter`.
///
/// # Safety
/// `counter` is a `sync` scratch word every item of the launch passes for
/// the same group, `expected` is the group's item count, and each item
/// arrives exactly once.
#[inline(always)]
pub unsafe fn last(counter: *mut u32, expected: u32) -> bool {
    // SAFETY: the caller's: a live, aligned word of the launch's scratch.
    let counter = unsafe { AtomicU32::from_ptr(counter) };
    let is_last = counter.fetch_add(1, Ordering::AcqRel) == expected - 1;
    if is_last {
        counter.store(0, Ordering::Relaxed);
    }
    is_last
}
