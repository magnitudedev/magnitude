use super::{OpenError, ResolveError};
use crate::api::device::DeviceInner;
use crate::artifacts::DeviceOptions;
use crate::memory::{path_claim, store_ledger, DeviceLedgers};
use seismic_discovery::{DiscoveryError, Inventory, RuntimeStatus, Status};
use seismic_topology::{
    Availability, BackendName, DeviceMemory, DeviceTopology, DiagnosticSource, ExecutorId,
    ExecutorSelector, Reported, ReportedBy,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// The executors this process has open, shared by every [`Devices`].
fn opened() -> &'static Mutex<HashMap<ExecutorId, Weak<DeviceInner>>> {
    static OPENED: OnceLock<Mutex<HashMap<ExecutorId, Weak<DeviceInner>>>> = OnceLock::new();
    OPENED.get_or_init(Mutex::default)
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

    /// Devices over a newly discovered inventory serving `paths`.
    pub fn discover(paths: &[PathBuf]) -> Result<Self, DiscoveryError> {
        Inventory::discover(paths).map(|inventory| Self::new(Arc::new(inventory)))
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

    /// Open `id` with `options`. An executor already open is shared; asking
    /// it for a different artifact store is an error.
    pub fn open_with(
        &self,
        id: ExecutorId,
        options: DeviceOptions,
    ) -> Result<Arc<DeviceInner>, OpenError> {
        let topology = self.topology();
        let info = topology.executor(id).ok_or(OpenError::Stale(id))?.clone();
        let mut opened = opened().lock().expect("opened executors lock poisoned");
        opened.retain(|_, device| device.strong_count() != 0);
        if let Some(device) = opened.get(&id).and_then(Weak::upgrade) {
            let same = match (&device.artifacts, &options.artifacts) {
                (_, None) => true,
                (Some(open), Some(requested)) => Arc::ptr_eq(open, requested),
                (None, Some(_)) => false,
            };
            if !same {
                return Err(OpenError::ArtifactStoreConflict(info.selector()));
            }
            return Ok(device);
        }
        if let Availability::Unavailable { reason } = info.availability() {
            return Err(OpenError::Unavailable {
                selector: info.selector(),
                reason: reason.clone(),
            });
        }
        let memory = DeviceLedgers::new(&topology, &info);
        let device = crate::backends::open(info, topology, memory, options)?;
        opened.insert(id, Arc::downgrade(&device));
        Ok(device)
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
                crate::probes::run::<seismic_vulkan::Probed>(
                    store,
                    &identity,
                    seismic_vulkan::probe::REVISION,
                    &program,
                    &["vulkan", device.as_str()],
                    deadline,
                )
                .map_err(unavailable)
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
        self.open(executor.id())
    }

    /// The status of `executor` of `topology`: a sample of the stores and
    /// of the executor's grants, with what this process's opened executors
    /// hold.
    pub fn status(&self, topology: &DeviceTopology, executor: ExecutorId) -> Status {
        Status {
            sample: self.inventory.sample(topology, &[executor]),
            runtime: self.runtime_status(topology, executor),
        }
    }

    /// What only opened executors know: each store's ledger, and what is
    /// left of each grant of `of` when it is an opened CUDA executor.
    fn runtime_status(&self, topology: &DeviceTopology, of: ExecutorId) -> RuntimeStatus {
        let charges = topology
            .stores()
            .iter()
            .map(|store| (store.id(), store_ledger(topology, store.id()).claim()))
            .collect();
        // Read outside the lock: a driver query must not hold up an open.
        let devices = opened()
            .lock()
            .expect("opened executors lock poisoned")
            .values()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
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
            let executor = topology
                .executor(of)
                .expect("opened executor belongs to topology");
            let memory = topology
                .physical_device(executor.physical())
                .expect("an executor's device belongs to topology")
                .memory();
            for path in topology
                .access_paths()
                .iter()
                .filter(|path| path.executor() == of)
            {
                let Some(grant) = path.grant() else {
                    continue;
                };
                let remaining = match memory {
                    // CUDA's free figure for its own memory bounds the grant.
                    DeviceMemory::Own => free
                        .clone()
                        .map(|value| Reported::new(value, ReportedBy::CudaFree)),
                    // The free figure of a device whose memory is the host's
                    // excludes reclaimable page cache and is no bound: the
                    // grant less what the executor holds is.
                    DeviceMemory::Host => Ok(Reported::new(
                        grant
                            .value
                            .saturating_sub(path_claim(topology, executor, path)),
                        grant.by,
                    )),
                    DeviceMemory::Undecided => {
                        unreachable!("an executor of an undecided device has no access path")
                    }
                };
                grants.push((path.id(), remaining));
            }
        }
        RuntimeStatus { charges, grants }
    }
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
