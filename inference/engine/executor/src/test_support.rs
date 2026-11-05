//! What the crate's tests share.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// The process's CPU placement, for a test: libtest runs the crate's tests
/// in threads of one process, and a serving executor places every thread
/// of it (`seismic::ProcessPlacement`, one at a time), so what the process
/// may run on reads as that placement while it serves. A test that creates
/// a serving executor (a load with an expert bank), or sizes a pool from
/// the CPUs the process may run on (a CPU site's admission, a test's pool),
/// holds it while it does.
pub(crate) fn process_placement() -> MutexGuard<'static, ()> {
    static PLACEMENT: Mutex<()> = Mutex::new(());
    // A test that failed holding it leaves nothing to repair: its placement
    // dropped as it unwound.
    PLACEMENT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether discovery found an available executor of `backend`: the one
/// source of a backend's presence for a test that needs one the host may
/// lack. Such a test runs on each backend this holds for, and fails where
/// an available one does not open.
pub(crate) fn available(devices: &seismic::Devices, backend: seismic::BackendName) -> bool {
    devices.topology().executors().iter().any(|executor| {
        executor.backend() == backend && matches!(executor.availability(), seismic::Availability::Available)
    })
}
