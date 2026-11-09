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

/// How every test of the process opens a device: with the one artifact
/// store the process shares ([`seismic::MemoryStore`]). A load forms its
/// programs on its device, and a device lives only while a test holds it;
/// through the store each program is compiled once in the process however
/// many tests load it, not once per load. Every open passes the same store,
/// so no test asks a device another test holds open for a different one
/// (`seismic::OpenError::ArtifactStoreConflict`).
pub(crate) fn device_options() -> seismic::DeviceOptions {
    static STORE: std::sync::OnceLock<std::sync::Arc<seismic::MemoryStore>> = std::sync::OnceLock::new();
    seismic::DeviceOptions {
        artifacts: Some(STORE.get_or_init(Default::default).clone()),
        ..Default::default()
    }
}

/// Open executor `id` of `catalog` as every test does ([`device_options`]).
pub(crate) fn open(catalog: &seismic::Devices, id: seismic::ExecutorId) -> Result<seismic::Device, seismic::OpenError> {
    catalog.open_with(id, device_options())
}

/// Open the first executor of `backend` in `catalog` as every test does
/// ([`device_options`]).
#[cfg(test)]
pub(crate) fn open_backend(
    catalog: &seismic::Devices,
    backend: seismic::BackendName,
) -> Result<seismic::Device, seismic::OpenError> {
    catalog.open_backend_with(backend, device_options())
}

/// Whether a test runs on `backend` ([`seismic::testing::selected`]): the
/// runner requests it or none, and discovery found an available executor
/// of it. Such a test runs on each backend this holds for, and fails where
/// an available one does not open. A build for a booked host names its
/// backend (`--config env.SEISMIC_TEST_BOOKED=cuda`, read when the tests
/// compile), and that backend holds or panics, naming it: a test never
/// passes by skipping the backend its run was booked for.
pub(crate) fn available(devices: &seismic::Devices, backend: seismic::BackendName) -> bool {
    let topology = devices.topology();
    let found = seismic::testing::selected(&topology, backend).is_some();
    if !found && booked() == Some(backend) {
        let listed = topology
            .executors()
            .iter()
            .filter(|executor| executor.backend() == backend)
            .map(|executor| format!("{}: {:?}", executor.selector(), executor.availability()))
            .collect::<Vec<_>>();
        panic!("the build was booked for {backend:?}, and this host has no available {backend:?} executor: {listed:?}");
    }
    found
}

/// The backend the build names as its host's (`SEISMIC_TEST_BOOKED`, at
/// compile time); a value that names no backend panics.
pub(crate) fn booked() -> Option<seismic::BackendName> {
    option_env!("SEISMIC_TEST_BOOKED").map(|name| {
        seismic::BackendName::parse(name).unwrap_or_else(|| panic!("unknown SEISMIC_TEST_BOOKED {name:?}"))
    })
}

/// `sources`, held in memory, checked with each native implementation's
/// source read from the engine's kernel tree, where the declarations name
/// them ([`seismic_lang::source::check_beside`]).
#[cfg(test)]
pub(crate) fn check_kernels(
    sources: seismic_lang::checked::SourceSet,
) -> seismic_lang::checked::CheckedModule {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let kernels = std::path::Path::new(&manifest).join("../kernels/kernels");
    seismic_lang::source::check_beside(sources, &kernels).expect("the kernel sources check")
}
