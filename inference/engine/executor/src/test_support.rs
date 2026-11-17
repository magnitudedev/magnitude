//! What the crate's tests, and its fixtures (`fixtures`) for the tests
//! above it, share.

#[cfg(test)]
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The process's CPU placement, for a test: libtest runs the crate's tests
/// in threads of one process, and a serving executor places every thread
/// of it (`seismic::ProcessPlacement`, one at a time), so what the process
/// may run on reads as that placement while it serves. A test that creates
/// a serving executor (a load with an expert bank), or sizes a pool from
/// the CPUs the process may run on (a CPU site's admission, a test's pool),
/// holds it while it does.
#[cfg(test)]
pub(crate) fn process_placement() -> MutexGuard<'static, ()> {
    static PLACEMENT: Mutex<()> = Mutex::new(());
    // A test that failed holding it leaves nothing to repair: its placement
    // dropped as it unwound.
    PLACEMENT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Open executor `id` of `catalog` as every test does: forming through the
/// user store, the one store of the process (`seismic::ProgramStore::user`),
/// so each program is compiled once however many tests and test processes
/// load it, and as a device of its own over unbounded tallies of its stores
/// (`seismic::testing::open_tallied`), with its device store's tally. Tests
/// share a process, and a device of the test's own shares no store counts
/// with another test's, so neither moves the other's room.
pub(crate) fn open_tallied(
    catalog: &seismic::Devices,
    id: seismic::ExecutorId,
) -> Result<(seismic::Device, seismic::memory::Tally), seismic::OpenError> {
    // The test's process reports as an engine process does: a panic that
    // ends it (`seismic::threads`) with every thread's stack, and
    // a hung test with them on SIGUSR1.
    crate::diagnostics::install();
    seismic::testing::open_tallied(catalog, id, Default::default())
}

/// [`open_tallied`] without the tally, for a test that reads no memory.
pub(crate) fn open(catalog: &seismic::Devices, id: seismic::ExecutorId) -> Result<seismic::Device, seismic::OpenError> {
    open_tallied(catalog, id).map(|(device, _)| device)
}

/// Open the first executor of `backend` in `catalog` as every test does
/// ([`open_tallied`]).
#[cfg(test)]
pub(crate) fn open_backend_tallied(
    catalog: &seismic::Devices,
    backend: seismic::BackendName,
) -> Result<(seismic::Device, seismic::memory::Tally), seismic::OpenError> {
    crate::diagnostics::install();
    seismic::testing::open_backend_tallied(catalog, backend, Default::default())
}

/// [`open_backend_tallied`] without the tally.
#[cfg(test)]
pub(crate) fn open_backend(
    catalog: &seismic::Devices,
    backend: seismic::BackendName,
) -> Result<seismic::Device, seismic::OpenError> {
    open_backend_tallied(catalog, backend).map(|(device, _)| device)
}

/// Whether a test runs on `backend` ([`seismic::testing::selected`]): the
/// runner requests it or none, and discovery found an available executor
/// of it, which a run that requests it must ([`seismic::testing::selected`]
/// panics otherwise). Such a test runs on each backend this holds for, and
/// fails where an available one does not open.
pub(crate) fn available(devices: &seismic::Devices, backend: seismic::BackendName) -> bool {
    seismic::testing::selected(&devices.topology(), backend).is_some()
}

/// The engine's kernel tree as production checks it: every `.seismic` file
/// under `kernels/` beside the standard library, each native implementation's
/// source read where its declaration names it. A test that needs a few of
/// its entries takes this one module (checked once), never a list of files of
/// its own, which would lack whatever one of them comes to read of another.
#[cfg(test)]
pub(crate) fn check_kernels() -> seismic_lang::checked::CheckedModule {
    static MODULE: std::sync::OnceLock<seismic_lang::checked::CheckedModule> = std::sync::OnceLock::new();
    MODULE
        .get_or_init(|| {
            let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
            let kernels = std::path::Path::new(&manifest).join("../kernels/kernels");
            seismic_lang::source::load(&[kernels], seismic_std::sources())
                .expect("the kernel sources check")
                .module
        })
        .clone()
}

#[cfg(test)]
/// The device the host has, selected as if the host had exactly `host_bytes`
/// of memory free and no driver grant: the room a plan fits into is this
/// figure, never the host's reading of the moment, so a test checks the same
/// thing wherever it runs.
pub(crate) fn selected_with_room(catalog: &seismic::Devices, host_bytes: u64) -> crate::platform::SelectedDevice {
    let release = crate::platform::ReleaseRate::of_machine(None);
    let real = crate::platform::select_device(catalog, crate::ExecutionPath::Native, crate::platform::DeviceRequest::Automatic, &release)
        .expect("the host has a device");
    let mut status = real.reading.status.clone();
    status.sample.host.total = host_bytes;
    status.sample.host.supply = seismic::Supply { free_beyond_target: i64::try_from(host_bytes).expect("bytes fit i64"), clean_cache: 0, cold_reclaimable: 0 };
    status.sample.grants.clear();
    crate::platform::selected(&real.reading.topology, &status, &release, real.info, real.graph_cost)
}

/// Whether `selected` is a device backend: a test that streams experts or
/// measures a device's attention cases has nothing to run on the CPU backend
/// (which holds no expert bank), so it passes by skipping there, but never in
/// a run that requests a device backend.
#[cfg(test)]
pub(crate) fn is_device_backend(selected: &crate::platform::SelectedDevice) -> bool {
    let backend = selected.info.backend();
    let device = backend != seismic::BackendName::Cpu;
    if !device && seismic::testing::requested_backend().is_some_and(|requested| requested != seismic::BackendName::Cpu) {
        panic!("the run requests {:?}, and this host selects {backend:?}", seismic::testing::requested_backend());
    }
    device
}
