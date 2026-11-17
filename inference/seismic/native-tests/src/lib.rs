//! Fixtures of the direct-native route: one checked module whose entries
//! carry Metal, CUDA and CPU implementations with static dimensions, tuning
//! parameters, scratch, several launches and shared memory. The tests run
//! on every backend the host can open.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/fixtures.rs"));

/// Available test backends of this test's capability domain, or exactly
/// the backend explicitly requested by the runner. A test's domain states
/// the backends it runs on, so under a requested backend outside it the
/// test selects none; a requested backend inside it must be available and
/// open, as absence is never a successful empty run.
///
/// Every device forms through the user store, the one store of the process
/// (`seismic::ProgramStore::user`): a program any test of any process of
/// the user formed, for any case, device or test, loads from the store, and
/// one the process forms compiles once. A test that observes its compiles
/// opens its devices with its own store ([`devices_with`]).
pub fn devices(backends: &[seismic::BackendName]) -> Vec<seismic::Device> {
    devices_with(backends, Default::default())
}

/// What a test forms before it touches a device, for one backend it runs
/// on: each native request it will prepare, checked as preparation checks
/// it, without a device ([`setup`]).
pub struct Setup {
    backend: seismic::BackendName,
    checked: usize,
    unchecked: Vec<String>,
}

impl Setup {
    /// The backend this setup forms the test's requests for.
    pub fn backend(&self) -> seismic::BackendName {
        self.backend
    }

    /// Check one request the test will prepare on this backend, `what` naming
    /// it, by what its entry's generated `native_admission` decides: a refusal
    /// fails the test here, with preparation's own message, before any device
    /// opens. One whose decision reads a device is left to preparation, and a
    /// setup run reports the test unchecked.
    pub fn admit(
        &mut self,
        what: impl std::fmt::Display,
        admission: Result<seismic::NativeAdmission, seismic::CheckedBundleError>,
    ) {
        match admission.unwrap_or_else(|error| panic!("{what}: the checked module does not load: {error}")) {
            seismic::NativeAdmission::Admitted => self.checked += 1,
            seismic::NativeAdmission::Refused(why) => {
                panic!("{what} on {:?}: preparation refuses it: {why}", self.backend)
            }
            seismic::NativeAdmission::Unchecked(why) => {
                self.unchecked.push(format!("{what} on {:?}: {why}", self.backend))
            }
        }
    }
}

/// A test's setup, then its devices ([`devices`]): `form` runs once for
/// each backend of `domain` the test runs on (the one the runner requests,
/// where it lies in `domain`; every one of `domain` where none is), forming
/// what the test will prepare there and checking it ([`Setup::admit`]), so a
/// request the backend's declarations refuse fails before any device opens.
/// The checks read no device: they cost the test nothing it would not spend
/// preparing.
///
/// In a setup run (`seismic::testing::setup_only`) the test ends after
/// `form`, its device work unrun, with what it checked
/// (`seismic::testing::SETUP_CHECKED`), or, where a check reads a device,
/// as unchecked (`seismic::testing::SETUP_UNCHECKED`): a runner reads each
/// test's setup before it books a device.
pub fn setup(domain: &[seismic::BackendName], form: impl FnMut(&mut Setup)) -> Vec<seismic::Device> {
    formed(domain, form);
    devices(domain)
}

/// [`setup`], its devices opened with `options` ([`devices_with`]).
pub fn setup_with(
    domain: &[seismic::BackendName],
    options: seismic::DeviceOptions,
    form: impl FnMut(&mut Setup),
) -> Vec<seismic::Device> {
    formed(domain, form);
    devices_with(domain, options)
}

fn formed(domain: &[seismic::BackendName], mut form: impl FnMut(&mut Setup)) {
    let backends = match seismic::testing::requested_backend() {
        Some(requested) => domain.iter().copied().filter(|backend| *backend == requested).collect(),
        None => domain.to_vec(),
    };
    let mut checked = 0;
    let mut unchecked = Vec::new();
    for backend in &backends {
        let mut setup = Setup { backend: *backend, checked: 0, unchecked: Vec::new() };
        form(&mut setup);
        checked += setup.checked;
        unchecked.extend(setup.unchecked);
    }
    if !seismic::testing::setup_only() {
        return;
    }
    if !unchecked.is_empty() {
        panic!(
            "{}: {checked} requests admitted on {backends:?}, {} read a device: {}",
            seismic::testing::SETUP_UNCHECKED,
            unchecked.len(),
            unchecked.join("; ")
        );
    }
    // Its device work unrun: the test ends here, as a runner of setups reads it.
    panic!("{} {checked} requests admitted on {backends:?}", seismic::testing::SETUP_CHECKED);
}

/// A line of a test's progress, written to stderr past the test harness's
/// capture: a test whose host model takes minutes stays visibly alive, so its
/// silence means no progress.
pub fn progress(line: std::fmt::Arguments) {
    use std::io::Write;
    writeln!(std::io::stderr(), "{line}").unwrap();
}

/// Every available backend's device, the CPU's included, or the backend
/// the runner requests ([`devices`]): for a test every backend runs.
pub fn every_device() -> Vec<seismic::Device> {
    use seismic::BackendName::{Cpu, Cuda, Metal, Vulkan};
    devices(&[Cpu, Metal, Cuda, Vulkan])
}

/// A DeviceOnly granule pool of `device`, of 1 MiB granules in 4 MiB
/// blocks, charged to an unbounded entry of its store.
pub fn pool(device: &seismic::Device) -> seismic::Pool {
    let topology = device.topology();
    let store = topology.first_store(device.info(), seismic::Contract::DeviceOnly).expect("the device places DeviceOnly");
    let identity = topology.store_identity(store).expect("a placement names stores of its revision");
    let entry: std::sync::Arc<dyn seismic::memory::Entry> = std::sync::Arc::new(seismic::memory::Tally::unbounded(identity));
    seismic::Pool::open(&seismic::PoolCharges::one(&device.charges_in(seismic::Contract::DeviceOnly, entry)), seismic::Contract::DeviceOnly, 1 << 20, 4 << 20).expect("the pool opens")
}

/// Select test devices with the caller's artifact-store configuration. Each
/// call discovers and opens fresh devices, each with its own ledgers, so a
/// test bounds them alone beside the process's other tests; selection has
/// the same contract as [`devices`].
pub fn devices_with(
    backends: &[seismic::BackendName],
    options: seismic::DeviceOptions,
) -> Vec<seismic::Device> {
    tallied_devices(backends, options).into_iter().map(|(device, _)| device).collect()
}

/// [`devices_with`], each device with its device store's tally, which a
/// test reads and bounds.
fn tallied_devices(
    backends: &[seismic::BackendName],
    options: seismic::DeviceOptions,
) -> Vec<(seismic::Device, seismic::memory::Tally)> {
    use seismic::Devices;
    let selected = seismic::testing::requested_backend();
    if let Some(backend) = selected.filter(|backend| !backends.contains(backend)) {
        eprintln!("native test backends: none, {backend:?} is outside this test's backend domain");
        return Vec::new();
    }
    let discovery = Devices::discover(&[]).expect("test device discovery");
    let topology = discovery.topology();
    let opened = backends.iter().copied()
        .filter_map(|backend| seismic::testing::selected(&topology, backend).map(|device| (backend, device.id())))
        .map(|(backend, id)| seismic::testing::open_tallied(&discovery, id, options.clone())
            .unwrap_or_else(|error| panic!("available {backend:?} test device must open: {error}")))
        .collect::<Vec<_>>();
    if let Some(backend) = selected {
        assert!(!opened.is_empty(), "selected {backend:?} test backend is unavailable");
    }
    eprintln!("native test backends: {:?}", opened.iter().map(|(device, _)| device.backend()).collect::<Vec<_>>());
    opened
}

/// The device of `backend` a test of that one backend runs on, as
/// [`devices`] selects it, with its own ledgers: a test that bounds its
/// memory sees only its own work.
pub fn fresh_device(backend: seismic::BackendName) -> Option<seismic::Device> {
    tallied_device(backend).map(|(device, _)| device)
}

/// [`fresh_device`] with its device store's tally, which the test reads and
/// bounds.
pub fn tallied_device(backend: seismic::BackendName) -> Option<(seismic::Device, seismic::memory::Tally)> {
    tallied_devices(&[backend], seismic::DeviceOptions::default()).pop()
}

/// One device of the first backend of `preferred` the host has, then the
/// CPU, for a test that checks one accelerator beside the CPU; under a
/// requested backend ([`seismic::testing::requested_backend`]), that
/// backend alone where it lies in `domain` and none outside it, as
/// [`devices`] selects.
pub fn first_and_cpu(preferred: &[seismic::BackendName], domain: &[seismic::BackendName]) -> Vec<seismic::Device> {
    if seismic::testing::requested_backend().is_some() {
        return devices(domain);
    }
    let mut chosen = devices(preferred);
    chosen.truncate(1);
    chosen.extend(devices(&[seismic::BackendName::Cpu]));
    chosen
}
