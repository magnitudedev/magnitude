use super::{
    host, Availability, DeviceId, DeviceInfo, DeviceMemory, DeviceMemoryInfo, DeviceSelector,
    DeviceTopology, DiscoveryError, HostMemoryStatus, LedgerKey, MemoryPoolId, MemoryPoolInfo,
    MemoryPoolKind, ObservationError, OpenError, ResolveError,
};
use crate::api::device::DeviceInner;
use crate::artifacts::DeviceOptions;
use crate::backends::{self, DiscoveredMemory};
use crate::memory::{MemoryDomain, PoolLedger};
use seismic_lang::registry::BackendName;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

/// The process's one ledger per physical pool, shared by every opened device
/// of every catalog and revision whose allocations consume that pool.
fn pool_ledger(key: LedgerKey) -> Arc<PoolLedger> {
    static LEDGERS: OnceLock<Mutex<HashMap<LedgerKey, Weak<PoolLedger>>>> = OnceLock::new();
    let mut ledgers = LEDGERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("process pool-ledger registry lock poisoned");
    if let Some(ledger) = ledgers.get(&key).and_then(Weak::upgrade) {
        return ledger;
    }
    let ledger = PoolLedger::new();
    ledgers.insert(key, Arc::downgrade(&ledger));
    ledger
}

/// The process-local device catalog. Discovery enumerates host and enabled
/// backend facts only; opening creates contexts, queues and profiles.
pub struct Catalog {
    state: Mutex<CatalogState>,
}

struct CatalogState {
    topology: Arc<DeviceTopology>,
    opened: HashMap<DeviceId, Weak<DeviceInner>>,
}

impl Catalog {
    pub fn discover() -> Result<Self, DiscoveryError> {
        Ok(Self {
            state: Mutex::new(CatalogState {
                topology: Arc::new(inventory(0)?),
                opened: HashMap::new(),
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, CatalogState> {
        self.state
            .lock()
            .expect("device catalog lock poisoned while holding catalog state")
    }

    pub fn topology(&self) -> Arc<DeviceTopology> {
        self.state().topology.clone()
    }

    /// Re-enumerates. An unchanged inventory keeps its revision and
    /// identifiers; a changed one receives a new revision, which makes
    /// earlier identifiers stale. Live opened devices and their allocations
    /// are unaffected.
    pub fn refresh(&self) -> Result<Arc<DeviceTopology>, DiscoveryError> {
        let revision = self.state().topology.revision;
        let unchanged = inventory(revision)?;
        let mut state = self.state();
        if state.topology.revision == revision && *state.topology == unchanged {
            return Ok(state.topology.clone());
        }
        let next = state.topology.revision + 1;
        state.topology = Arc::new(unchanged.with_revision(next));
        state.opened.retain(|id, _| id.revision == next);
        Ok(state.topology.clone())
    }

    /// Resolves a selector in the current topology. A selector never falls
    /// back to an ordinal or another device.
    pub fn resolve(&self, selector: DeviceSelector) -> Result<DeviceId, ResolveError> {
        let topology = self.topology();
        let mut matches = topology
            .devices()
            .iter()
            .filter(|device| device.selector == selector);
        let found = matches.next().ok_or(ResolveError::Missing(selector))?;
        let extra = matches.count();
        if extra != 0 {
            return Err(ResolveError::Ambiguous {
                selector,
                matches: extra + 1,
            });
        }
        Ok(found.id)
    }

    pub fn open(&self, id: DeviceId) -> Result<Arc<DeviceInner>, OpenError> {
        self.open_with(id, DeviceOptions::default())
    }

    /// Open with `options`. A device already open is shared as it was
    /// opened; asking it for a different artifact store is an error.
    pub fn open_with(
        &self,
        id: DeviceId,
        options: DeviceOptions,
    ) -> Result<Arc<DeviceInner>, OpenError> {
        // Holding this lock through acquisition makes opening atomic: two
        // callers cannot create independent services or profiles for the
        // same descriptor. A dropped device may be opened and profiled
        // again; a live one is always shared.
        let mut state = self.state();
        let info = state
            .topology
            .device(id)
            .ok_or(OpenError::Stale(id))?
            .clone();
        if let Some(device) = state.opened.get(&id).and_then(Weak::upgrade) {
            let same = match (&device.artifacts, &options.artifacts) {
                (_, None) => true,
                (Some(open), Some(requested)) => Arc::ptr_eq(open, requested),
                (None, Some(_)) => false,
            };
            if !same {
                return Err(OpenError::ArtifactStoreConflict(info.selector));
            }
            return Ok(device);
        }
        if let Availability::Unavailable { reason } = &info.availability {
            return Err(OpenError::Unavailable {
                selector: info.selector,
                reason: reason.clone(),
            });
        }
        let key = match &info.memory {
            DeviceMemory::Established(memory) => {
                state
                    .topology
                    .pool(memory.allocation_pool)
                    .expect("validated topology references its own pools")
                    .ledger
            }
            DeviceMemory::Unsupported { .. } => LedgerKey::Unestablished(info.selector),
        };
        let device = backends::open(info, MemoryDomain::new(pool_ledger(key)), options)?;
        state.opened.insert(id, Arc::downgrade(&device));
        Ok(device)
    }

    /// Low-level control: opens the first discovered device of a backend in
    /// discovery order. Managed selection chooses by model requirements.
    pub fn open_backend(&self, backend: BackendName) -> Result<Arc<DeviceInner>, OpenError> {
        let topology = self.topology();
        let Some(device) = topology
            .devices()
            .iter()
            .find(|device| device.backend == backend)
        else {
            return Err(OpenError::NoDevice {
                backend,
                diagnostics: topology
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| diagnostic.backend == backend)
                    .map(|diagnostic| diagnostic.message.clone())
                    .collect(),
            });
        };
        self.open(device.id)
    }

    pub fn host_memory_status(&self) -> Result<HostMemoryStatus, ObservationError> {
        host::status()
    }
}

/// Enumerates host and backend facts into one validated topology.
fn inventory(revision: u64) -> Result<DeviceTopology, DiscoveryError> {
    let capacity = host::capacity().map_err(DiscoveryError::HostMemory)?;
    let host_pool = MemoryPoolId { revision, index: 0 };
    let mut pools = vec![MemoryPoolInfo {
        id: host_pool,
        kind: MemoryPoolKind::HostRam,
        capacity_bytes: capacity.bytes,
        basis: capacity.basis,
        ledger: LedgerKey::Host,
    }];
    let discovered = backends::discover();
    let mut devices = Vec::with_capacity(discovered.devices.len());
    for device in discovered.devices {
        let memory = match device.memory {
            DiscoveredMemory::Host {
                max_allocation_bytes,
            } => DeviceMemory::Established(DeviceMemoryInfo {
                allocation_pool: host_pool,
                host_pool,
                max_allocation_bytes,
            }),
            DiscoveredMemory::Dedicated {
                capacity_bytes,
                basis,
                ledger,
                max_allocation_bytes,
            } => {
                // Multiple views of one physical pool are merged only by
                // proven identity; distinct devices are never summed.
                let allocation_pool = match pools.iter().find(|pool| pool.ledger == ledger) {
                    Some(pool) => pool.id,
                    None => {
                        let id = MemoryPoolId {
                            revision,
                            index: u32::try_from(pools.len()).expect("pool count exceeds u32"),
                        };
                        pools.push(MemoryPoolInfo {
                            id,
                            kind: MemoryPoolKind::DeviceLocal,
                            capacity_bytes,
                            basis,
                            ledger,
                        });
                        id
                    }
                };
                DeviceMemory::Established(DeviceMemoryInfo {
                    allocation_pool,
                    host_pool,
                    max_allocation_bytes,
                })
            }
            DiscoveredMemory::Unsupported { reason } => DeviceMemory::Unsupported { reason },
        };
        devices.push(DeviceInfo {
            id: DeviceId {
                revision,
                index: u32::try_from(devices.len()).expect("device count exceeds u32"),
            },
            selector: device.selector,
            name: device.name,
            kind: device.kind,
            backend: device.backend,
            availability: device.availability,
            memory,
            descriptor: Arc::new(device.descriptor),
        });
    }
    Ok(DeviceTopology {
        revision,
        devices,
        pools,
        diagnostics: discovered.diagnostics,
    })
}

impl DeviceTopology {
    fn with_revision(mut self, revision: u64) -> Self {
        self.revision = revision;
        for device in &mut self.devices {
            device.id.revision = revision;
            if let DeviceMemory::Established(memory) = &mut device.memory {
                memory.allocation_pool.revision = revision;
                memory.host_pool.revision = revision;
            }
        }
        for pool in &mut self.pools {
            pool.id.revision = revision;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_cpu_is_always_discovered_on_the_host_pool() {
        let catalog = Catalog::discover().unwrap();
        let topology = catalog.topology();
        let cpu = topology
            .devices()
            .iter()
            .find(|device| device.selector == DeviceSelector::HostCpu)
            .unwrap();
        let DeviceMemory::Established(memory) = &cpu.memory else {
            panic!("host CPU backing is established")
        };
        assert!(memory.allocates_host_memory());
        assert_eq!(
            topology.pool(memory.allocation_pool),
            Some(topology.host_pool())
        );
        assert!(topology.host_pool().capacity_bytes > 0);
        // The address-size limit is an allocation limit, never capacity.
        assert!(memory.max_allocation_bytes > topology.host_pool().capacity_bytes);
    }

    #[test]
    fn unchanged_refresh_keeps_identifiers_and_selectors_resolve_exactly() {
        let catalog = Catalog::discover().unwrap();
        let before = catalog.topology();
        let after = catalog.refresh().unwrap();
        assert_eq!(before.revision(), after.revision());
        let id = catalog.resolve(DeviceSelector::HostCpu).unwrap();
        assert_eq!(after.device(id).unwrap().selector, DeviceSelector::HostCpu);
        assert_eq!(
            catalog.resolve(DeviceSelector::Cuda { uuid: [0xff; 16] }),
            Err(ResolveError::Missing(DeviceSelector::Cuda {
                uuid: [0xff; 16]
            }))
        );
    }

    #[test]
    fn a_changed_revision_makes_earlier_identifiers_stale() {
        let catalog = Catalog::discover().unwrap();
        let old = catalog.resolve(DeviceSelector::HostCpu).unwrap();
        {
            let mut state = catalog.state();
            let bumped = (*state.topology).clone().with_revision(old.revision + 1);
            state.topology = Arc::new(bumped);
        }
        assert!(matches!(catalog.open(old), Err(OpenError::Stale(_))));
        let current = catalog.resolve(DeviceSelector::HostCpu).unwrap();
        assert_ne!(old, current);
    }

    /// Vulkan is not built on macOS.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_vulkan_request_is_refused_with_the_missing_runtime() {
        let catalog = Catalog::discover().unwrap();
        assert_eq!(
            catalog.open_backend(BackendName::Vulkan).err(),
            Some(OpenError::NoDevice {
                backend: BackendName::Vulkan,
                diagnostics: vec!["this build has no Vulkan runtime".into()],
            })
        );
    }

    #[test]
    fn devices_on_one_pool_share_one_ledger_across_handles() {
        let catalog = Catalog::discover().unwrap();
        let cpu = catalog.open_backend(BackendName::Cpu).unwrap();
        let again = catalog.open_backend(BackendName::Cpu).unwrap();
        assert!(Arc::ptr_eq(&cpu, &again));
        let tensor = crate::api::tensor::TensorInner::zeros(
            &cpu,
            seismic_lang::registry::dense(seismic_lang::types::DType::F32),
            &[1024],
        )
        .unwrap();
        let usage = cpu.memory_usage();
        assert!(usage.charged >= 4096);
        assert!(usage.pool_charged >= usage.charged);
        // A second catalog in this process opens its own device on the same
        // physical pool: both see one shared ledger.
        let other = Catalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        assert!(!Arc::ptr_eq(&cpu, &other));
        assert_eq!(other.memory_usage().charged, 0);
        assert!(other.memory_usage().pool_charged >= usage.charged);
        drop(tensor);
        assert_eq!(cpu.memory_usage().charged, 0);
    }

    #[test]
    fn selectors_round_trip_through_text() {
        for selector in [
            DeviceSelector::HostCpu,
            DeviceSelector::Metal {
                registry_id: 0x1000_0000_0abc,
            },
            DeviceSelector::Cuda {
                uuid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            DeviceSelector::Vulkan {
                uuid: [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            },
        ] {
            assert_eq!(selector.to_string().parse::<DeviceSelector>(), Ok(selector));
        }
        assert!("cuda:00".parse::<DeviceSelector>().is_err());
    }
}
