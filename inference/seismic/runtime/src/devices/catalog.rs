use super::{OpenError, ResolveError};
use crate::api::device::DeviceInner;
use crate::artifacts::DeviceOptions;
use crate::memory::{DeviceStores, Store, Tally};
use seismic_discovery::{DiscoveryError, Inventory, RuntimeStatus, Status};
use seismic_topology::{
    Availability, BackendName, DeviceMemory, DeviceTopology, DiagnosticSource, ExecutorId,
    ExecutorSelector, Reported, ReportedBy,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

/// The executors this process has open, shared by every [`Devices`], held
/// across an open so that an executor opens once. An open that panics has
/// changed nothing (an executor is entered only once it opened, and the
/// entries of dropped ones are only ever removed), so the panic is that
/// open's alone: the lock does not pass it on to every later open of the
/// process as poison.
fn opened() -> MutexGuard<'static, HashMap<ExecutorId, Weak<DeviceInner>>> {
    static OPENED: OnceLock<Mutex<HashMap<ExecutorId, Weak<DeviceInner>>>> = OnceLock::new();
    OPENED
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Runtime resources over an inventory. Machine discovery and sampling
/// belong to the inventory.
pub struct Devices {
    inventory: Arc<Inventory>,
}

impl Devices {
    /// Devices over `inventory`, which discovers and samples the machine.
    pub fn new(inventory: Arc<Inventory>) -> Self {
        Self { inventory }
    }

    /// Devices over a newly discovered inventory serving `paths`, its
    /// probe answers (Metal's capability probes) read from and kept in the
    /// process's store ([`crate::artifacts::ProgramStore::user`]).
    pub fn discover(paths: &[PathBuf]) -> Result<Self, DiscoveryError> {
        Inventory::discover(paths, crate::artifacts::ProgramStore::user()).map(|inventory| Self::new(Arc::new(inventory)))
    }

    /// The inventory these devices are over.
    pub fn inventory(&self) -> &Arc<Inventory> {
        &self.inventory
    }

    /// The inventory's current topology revision.
    pub fn topology(&self) -> Arc<DeviceTopology> {
        self.inventory.topology()
    }

    /// The executor of `selector` in the current revision. Never
    /// substitutes another executor.
    pub fn resolve(&self, selector: ExecutorSelector) -> Result<ExecutorId, ResolveError> {
        self.topology()
            .resolve(selector)
            .map(|executor| executor.id())
            .ok_or(ResolveError::Missing(selector))
    }

    /// Open executor `id` of the current revision with default options;
    /// see [`Devices::open_with`].
    pub fn open(&self, id: ExecutorId) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_with(id, DeviceOptions::default())
    }

    /// Open `id` with `options`, outside any embedder's ledger: an executor
    /// already open is shared, and one not open yet charges a [`Tally`] of
    /// each store it reaches, its own. Asking an open executor for a
    /// different artifact store is an error.
    pub fn open_with(&self, id: ExecutorId, options: DeviceOptions) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_shared(id, options, None)
    }

    /// Open `id` with `options`, charging `stores`, the embedder's ledger of
    /// every store it reaches: the product's open. An executor already open
    /// is shared where it charges these same stores; asking it for others,
    /// or for a different artifact store, is an error.
    pub fn open_in(
        &self,
        id: ExecutorId,
        options: DeviceOptions,
        stores: Vec<Arc<dyn Store>>,
    ) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_shared(id, options, Some(stores))
    }

    fn open_shared(
        &self,
        id: ExecutorId,
        options: DeviceOptions,
        stores: Option<Vec<Arc<dyn Store>>>,
    ) -> Result<Arc<DeviceInner>, OpenError> {
        let topology = self.topology();
        let info = topology.executor(id).ok_or(OpenError::Stale(id))?.clone();
        let mut opened = opened();
        opened.retain(|_, device| device.strong_count() != 0);
        if let Some(device) = opened.get(&id).and_then(Weak::upgrade) {
            if options.artifacts.as_ref().is_some_and(|requested| !Arc::ptr_eq(&device.artifacts, requested)) {
                return Err(OpenError::ArtifactStoreConflict(info.selector()));
            }
            if stores.as_ref().is_some_and(|stores| !device.memory.opened_with(stores)) {
                return Err(OpenError::StoreConflict(info.selector()));
            }
            return Ok(device);
        }
        if let Availability::Unavailable { reason } = info.availability() {
            return Err(OpenError::Unavailable {
                selector: info.selector(),
                reason: reason.clone(),
            });
        }
        let embedded = stores.is_some();
        let stores = stores.unwrap_or_else(|| tallies(&topology, &info));
        let memory = DeviceStores::new(&topology, &info, stores, embedded);
        let device = crate::backends::open(info, topology, memory, options)?;
        opened.insert(id, Arc::downgrade(&device));
        Ok(device)
    }

    /// Open `id` with `options` as a device of its own, charging `stores`,
    /// shared with no other open: for a test, whose process other tests
    /// share, reading and bounding what its own stores count.
    #[doc(hidden)]
    pub fn open_own(
        &self,
        id: ExecutorId,
        options: DeviceOptions,
        stores: Vec<Arc<dyn Store>>,
    ) -> Result<Arc<DeviceInner>, OpenError> {
        let topology = self.topology();
        let info = topology.executor(id).ok_or(OpenError::Stale(id))?.clone();
        if let Availability::Unavailable { reason } = info.availability() {
            return Err(OpenError::Unavailable {
                selector: info.selector(),
                reason: reason.clone(),
            });
        }
        let memory = DeviceStores::new(&topology, &info, stores, true);
        crate::backends::open(info, topology, memory, options)
    }

    /// The measurement job's running of executor `id`'s probes: each its
    /// backend declares (`seismic_lang::device`) that `store` holds no
    /// result of for the executor's device and driver, in a probe process
    /// of the running executable, keeping each result in `store` as it
    /// comes, before `deadline` ([`crate::probes`]). Opening the executor
    /// afterwards with `store` reads them; nothing here opens it. A backend
    /// without probes runs none.
    pub fn probe(
        &self,
        id: ExecutorId,
        store: &dyn crate::artifacts::ArtifactStore,
        deadline: std::time::Instant,
    ) -> Result<crate::probes::ProbeRun, OpenError> {
        let topology = self.topology();
        let selector = topology.executor(id).ok_or(OpenError::Stale(id))?.selector();
        match selector {
            #[cfg(not(target_os = "macos"))]
            ExecutorSelector::Vulkan { uuid } => {
                let identity = topology.executor_identity(id).ok_or(OpenError::Stale(id))?;
                let unavailable = |reason: String| OpenError::Unavailable { selector, reason };
                let program = std::env::current_exe()
                    .map_err(|error| unavailable(format!("locating the running executable for its probes: {error}")))?;
                let device = seismic_vulkan::probe::argument(&uuid);
                Ok(crate::probes::run::<seismic_vulkan::Probed>(
                    store,
                    &identity,
                    seismic_vulkan::probe::REVISION,
                    &program,
                    &["vulkan", device.as_str()],
                    deadline,
                ))
            }
            #[cfg(target_os = "macos")]
            ExecutorSelector::Vulkan { .. } => unreachable!("this build discovers no Vulkan executor"),
            // Their backends declare no probes.
            ExecutorSelector::HostCpu | ExecutorSelector::Metal { .. } | ExecutorSelector::Cuda { .. } => {
                Ok(crate::probes::ProbeRun::default())
            }
        }
    }

    /// Low-level control: opens the first discovered executor of a backend.
    /// Managed callers select by requirements and open by identity.
    pub fn open_backend(&self, backend: BackendName) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_backend_with(backend, DeviceOptions::default())
    }

    /// [`Devices::open_backend`] with `options`, as [`Devices::open_with`].
    pub fn open_backend_with(&self, backend: BackendName, options: DeviceOptions) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_with(self.executor_of(backend)?, options)
    }

    /// The first discovered executor of `backend`.
    #[doc(hidden)]
    pub fn executor_of(&self, backend: BackendName) -> Result<ExecutorId, OpenError> {
        let topology = self.topology();
        let executor = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == backend)
            .ok_or_else(|| OpenError::NoDevice {
                backend,
                diagnostics: topology
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| diagnostic.source == DiagnosticSource::Backend(backend))
                    .map(|diagnostic| diagnostic.message.clone())
                    .collect(),
            })?;
        Ok(executor.id())
    }

    /// The status of `executor` of `topology`: a sample of the stores and
    /// of the executor's grants, with what this process's opened executors
    /// hold.
    pub fn status(&self, topology: &DeviceTopology, executor: ExecutorId) -> Status {
        // Read outside the lock: a driver query must not hold up an open.
        let devices = opened()
            .values()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        Status {
            sample: self.inventory.sample(topology, &[executor]),
            runtime: Self::runtime_status(&devices, topology, executor),
        }
    }

    /// The status of opened `device`, as [`Self::status`] reads it.
    pub fn device_status(&self, device: &Arc<DeviceInner>) -> Status {
        let (topology, executor) = (device.topology(), device.info().id());
        Status {
            sample: self.inventory.sample(topology, &[executor]),
            runtime: Self::runtime_status(std::slice::from_ref(device), topology, executor),
        }
    }

    /// What only opened executors know: what is left of each grant of `of`
    /// on its own memory when it is one of `devices`, an opened CUDA
    /// executor (`cuMemGetInfo`'s free). A grant on host memory is the
    /// host's figure, read by the embedder's ledger beside what it holds.
    fn runtime_status(devices: &[Arc<DeviceInner>], topology: &DeviceTopology, of: ExecutorId) -> RuntimeStatus {
        let free = devices
            .iter()
            .filter(|device| {
                topology
                    .resolve(device.info().selector())
                    .is_some_and(|executor| executor.id() == of)
            })
            .filter_map(|device| device.kind.cuda_free())
            .reduce(|a, b| match (a, b) {
                (Ok(a), Ok(b)) => Ok(a.min(b)),
                (Err(error), _) | (_, Err(error)) => Err(error),
            });
        let mut grants = Vec::new();
        if let Some(free) = free {
            let executor = topology.executor(of).expect("opened executor belongs to topology");
            let memory = topology
                .physical_device(executor.physical())
                .expect("an executor's device belongs to topology")
                .memory();
            if memory == DeviceMemory::Own {
                for path in topology.access_paths().iter().filter(|path| path.executor() == of && path.grant().is_some()) {
                    grants.push((path.id(), free.clone().map(|value| Reported::new(value, ReportedBy::CudaFree))));
                }
            }
        }
        RuntimeStatus { grants }
    }
}

/// An unbounded [`Tally`] of every store `executor` of `topology` reaches:
/// the stores of a device opened outside any embedder's ledger.
fn tallies(topology: &DeviceTopology, executor: &seismic_topology::Executor) -> Vec<Arc<dyn Store>> {
    let mut identities = Vec::new();
    let reached = topology
        .access_paths()
        .iter()
        .filter(|path| path.executor() == executor.id())
        .map(|path| path.store())
        .chain([topology.host_ram().id()]);
    for store in reached {
        let identity = topology.store_identity(store).expect("a path's store belongs to its topology");
        if !identities.contains(&identity) {
            identities.push(identity);
        }
    }
    identities.into_iter().map(|identity| Arc::new(Tally::unbounded(identity)) as Arc<dyn Store>).collect()
}

/// Vulkan is not built on macOS: there a Vulkan request is refused with the
/// Vulkan backend's diagnostic and no other source's.
#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn a_vulkan_request_is_refused_with_the_missing_runtime() {
        let devices = Devices::discover(&[]).unwrap();
        assert_eq!(
            devices.open_backend(BackendName::Vulkan).err(),
            Some(OpenError::NoDevice {
                backend: BackendName::Vulkan,
                diagnostics: vec!["this build has no Vulkan runtime".into()],
            })
        );
    }
}

#[cfg(test)]
mod opened_tests {
    use super::*;

    /// An open that panics while it holds the opened executors leaves
    /// every later open of the process working.
    #[test]
    fn a_panicked_open_leaves_later_opens_working() {
        let panicked = std::thread::spawn(|| {
            let _held = opened();
            panic!("the open failed");
        })
        .join();
        assert!(panicked.is_err());
        Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
    }
}
