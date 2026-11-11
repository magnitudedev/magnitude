//! The inventory: the current topology revision of the machine, refreshed
//! on request, and its status samples.

use crate::gather::{self, DiscoveryError, Gathered, Natives};
use crate::host::{self, HostStatus, ObservationError};
use crate::status::{self, Sample};
use seismic_topology::{DeviceTopology, ExecutorId, Identity, Revision};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

/// The machine as discovery last found it: the current topology revision,
/// and the native handles its status samples read. Shared by everything in
/// the process that reads the machine.
pub struct Inventory {
    /// The paths the inventory serves, whose volumes each revision holds.
    paths: Vec<PathBuf>,
    state: Mutex<State>,
}

struct State {
    topology: Arc<DeviceTopology>,
    natives: Natives,
}

impl Inventory {
    /// The machine as its backends and OS report it, with the volumes of
    /// `paths` (the model store, the kernel cache): what the backend crates
    /// report of each device, the OS's memory, CPU, adapter and filesystem
    /// interfaces, and nothing timed.
    pub fn discover(paths: &[PathBuf]) -> Result<Self, DiscoveryError> {
        seismic_lang::registry::test_device_reached("discovers devices");
        Ok(Self::from_gathered(paths.to_vec(), gather::gather(paths)?))
    }

    pub(crate) fn from_gathered(paths: Vec<PathBuf>, gathered: Gathered) -> Self {
        Self {
            paths,
            state: Mutex::new(State {
                topology: Arc::new(seismic_topology::build(Revision::next(), &gathered.reports)),
                natives: gathered.natives,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("inventory lock poisoned")
    }

    /// The current revision.
    pub fn topology(&self) -> Arc<DeviceTopology> {
        self.state().topology.clone()
    }

    /// Discover again. A machine that reads the same keeps its revision and
    /// identifiers; one that reads differently is a new revision, whose
    /// identifiers are all its own.
    pub fn refresh(&self) -> Result<Arc<DeviceTopology>, DiscoveryError> {
        let gathered = gather::gather(&self.paths)?;
        Ok(self.advance(gathered))
    }

    pub(crate) fn advance(&self, gathered: Gathered) -> Arc<DeviceTopology> {
        let mut state = self.state();
        state.natives = gathered.natives;
        let unchanged = seismic_topology::build(state.topology.revision(), &gathered.reports);
        if unchanged != *state.topology {
            state.topology = Arc::new(seismic_topology::build(Revision::next(), &gathered.reports));
        }
        state.topology.clone()
    }

    /// One reading of host RAM alone, taken now: the OS's availability
    /// within this process's limits. A [`Sample`] reads host RAM the same
    /// way, so its headroom ([`crate::Status::headroom`]) and this agree.
    pub fn host_status(&self) -> Result<HostStatus, ObservationError> {
        host::status()
    }

    /// One reading of the status of `topology`'s stores and of the grants
    /// of `executors`, host RAM read now: what a claim and every
    /// observation between claims is judged against.
    pub fn sample(&self, topology: &DeviceTopology, executors: &[ExecutorId]) -> Sample {
        let host_ram = self.host_status();
        status::sample(topology, executors, &self.state().natives, host_ram)
    }

    /// The persistent identity of an executor of the current revision, what
    /// measurements of work on it are keyed by; none for an executor of
    /// another revision.
    pub fn executor_identity(&self, executor: ExecutorId) -> Option<Identity> {
        self.topology().executor_identity(executor)
    }
}
