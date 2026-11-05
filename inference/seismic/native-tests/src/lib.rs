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
pub fn devices(backends: &[seismic::BackendName]) -> Vec<seismic::Device> {
    devices_with(backends, seismic::DeviceOptions::default())
}

/// Select test devices with the caller's artifact-store configuration. Each
/// call discovers and opens fresh devices; selection has the same contract
/// as `devices`.
pub fn devices_with(
    backends: &[seismic::BackendName],
    options: seismic::DeviceOptions,
) -> Vec<seismic::Device> {
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
        .map(|(backend, id)| discovery.open_with(id, options.clone())
            .unwrap_or_else(|error| panic!("available {backend:?} test device must open: {error}")))
        .collect::<Vec<_>>();
    if let Some(backend) = selected {
        assert!(!opened.is_empty(), "selected {backend:?} test backend is unavailable");
    }
    eprintln!("native test backends: {:?}", opened.iter().map(seismic::Device::backend).collect::<Vec<_>>());
    opened
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
