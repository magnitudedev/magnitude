//! The inventory: the current topology revision of the machine, refreshed
//! on request, and its status samples.

use crate::discovery::{self, DiscoveryError, Gathered, Natives};
use crate::host::{self, DisplacementHistory, HostStatus, ObservationError};
use crate::model::{DeviceTopology, Revision};
use crate::status::{self, Sample};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

/// The machine as discovery last found it: the current topology revision,
/// the native handles its status samples read, and the displacement history
/// of those samples. Shared by everything in the process that reads the
/// machine.
pub struct Inventory {
    /// The paths the inventory serves, whose volumes each revision holds.
    paths: Vec<PathBuf>,
    state: Mutex<State>,
}

struct State {
    topology: Arc<DeviceTopology>,
    natives: Natives,
    /// Displacement windows of the host samples taken through this
    /// inventory.
    displacement: DisplacementHistory,
}

impl Inventory {
    /// The machine as its backends and OS report it, with the volumes of
    /// `paths` (the model store, the kernel cache): what the backend crates
    /// report of each device, the OS's memory, CPU, adapter and filesystem
    /// interfaces, and nothing timed.
    pub fn discover(paths: &[PathBuf]) -> Result<Self, DiscoveryError> {
        Ok(Self::from_gathered(
            paths.to_vec(),
            discovery::gather(paths)?,
        ))
    }

    pub(crate) fn from_gathered(paths: Vec<PathBuf>, gathered: Gathered) -> Self {
        Self {
            paths,
            state: Mutex::new(State {
                topology: Arc::new(discovery::build(Revision::next(), &gathered.reports)),
                natives: gathered.natives,
                displacement: DisplacementHistory::default(),
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
        let gathered = discovery::gather(&self.paths)?;
        Ok(self.advance(gathered))
    }

    pub(crate) fn advance(&self, gathered: Gathered) -> Arc<DeviceTopology> {
        let mut state = self.state();
        state.natives = gathered.natives;
        let unchanged = discovery::build(state.topology.revision(), &gathered.reports);
        if unchanged != *state.topology {
            state.topology = Arc::new(discovery::build(Revision::next(), &gathered.reports));
        }
        state.topology.clone()
    }

    /// One reading of host RAM alone: the OS's availability within this
    /// process's limits, with the displacement and distress this inventory's
    /// readings have seen. A [`Sample`] reads host RAM the same way, so its
    /// headroom ([`crate::Status::headroom`]) and this agree.
    pub fn host_status(&self) -> Result<HostStatus, ObservationError> {
        host::status(&mut self.state().displacement)
    }

    /// One reading of the status of `topology`'s stores and grants.
    pub fn sample(&self, topology: &DeviceTopology) -> Sample {
        let mut state = self.state();
        let State {
            natives,
            displacement,
            ..
        } = &mut *state;
        status::sample(topology, natives, displacement)
    }
}
