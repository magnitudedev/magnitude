//! What the machine has: the reports
//! the backend crates and the OS give of it, gathered into the inventory of
//! its topology (`seismic_topology`), its status samples (host RAM at one
//! instant among them), the OS's memory pressure events, the host's
//! processors and their pinning, and the machine as its firmware names it.

#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests start threads of their own; the code they test starts its threads through seismic-threads"
    )
)]

#[cfg(windows)]
mod adapters;
#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
mod gather;
mod host;
mod inventory;
mod machine;
mod pressure;
mod processors;
mod status;
mod volume;

#[cfg(test)]
mod tests;

pub use gather::DiscoveryError;
pub use host::{
    host_sample, Counted, Harm, HostSample, Lag, ObservationError, OwnCold, OwnReclaimed, PressureLevel, Supply, WiredFigures, WiredLimit,
};
pub use inventory::Inventory;
pub use machine::{Machine, MachineFormFactor, MachineIdentity};
pub use processors::{
    device_node, pin, place_process, Core, DeviceNode, MemoryNode, PciLocation, ProcessPlacement,
    Processors, ProcessorsReportedBy,
};
pub use pressure::{PressureEvents, PressureWaker, Wake};
pub use status::{RuntimeStatus, Sample, Status};
pub use volume::{identify as volume_identity, volume_available};
