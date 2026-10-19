use super::{OpenError, ResolveError};
use crate::api::device::DeviceInner;
use crate::artifacts::DeviceOptions;
use crate::memory::{path_claim, store_ledger, ExecutorMemory};
use seismic_topology::{
    Availability, BackendName, DeviceMemory, DeviceTopology, DiagnosticSource, DiscoveryError,
    ExecutorId, ExecutorSelector, Inventory, Reported, ReportedBy, RuntimeStatus, Status,
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
        let memory = ExecutorMemory::new(&topology, &info);
        let device = crate::backends::open(info, topology, memory, options)?;
        opened.insert(id, Arc::downgrade(&device));
        Ok(device)
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

    /// A status sample of the current topology with what this process's
    /// opened executors hold.
    pub fn status(&self) -> Status {
        let topology = self.topology();
        Status {
            sample: self.inventory.sample(&topology),
            runtime: self.runtime_status(&topology),
        }
    }

    /// What only opened executors know: each store's ledger, and what is
    /// left of each grant an opened CUDA executor reads.
    fn runtime_status(&self, topology: &DeviceTopology) -> RuntimeStatus {
        let charges = topology
            .stores()
            .iter()
            .map(|store| (store.id(), store_ledger(topology, store.id()).bytes()))
            .collect();
        // Read outside the lock: a driver query must not hold up an open.
        let devices = opened()
            .lock()
            .expect("opened executors lock poisoned")
            .values()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        let mut cuda = HashMap::new();
        for device in devices {
            let Some(executor) = topology.resolve(device.info().selector()) else {
                continue;
            };
            if let Some(free) = device.kind.cuda_free() {
                let prior = cuda.entry(executor.id()).or_insert_with(|| free.clone());
                *prior = match (prior.clone(), free) {
                    (Ok(a), Ok(b)) => Ok(a.min(b)),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                };
            }
        }
        let mut grants = Vec::new();
        for (id, free) in cuda {
            let executor = topology
                .executor(id)
                .expect("opened executor belongs to topology");
            let memory = topology
                .physical_device(executor.physical())
                .expect("an executor's device belongs to topology")
                .memory();
            for path in topology
                .access_paths()
                .iter()
                .filter(|path| path.executor() == id)
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
