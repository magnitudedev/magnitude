//! What the machine has: the reports
//! the backend crates and the OS give of it, gathered into the inventory of
//! its topology (`seismic_topology`), its status samples, and the host's
//! processors and their pinning.

#[cfg(windows)]
mod adapters;
mod gather;
mod host;
mod inventory;
mod processors;
mod status;
mod volume;

#[cfg(test)]
mod tests;

pub use gather::DiscoveryError;
pub use host::{
    host_room, DisplacementWindow, HostDisplacement, HostHeld, HostStatus, KernelPressure, LimitVisibility,
    ObservationError, ProcessLimitKind, ProcessMemoryLimit, DISPLACEMENT_CADENCE,
    DISPLACEMENT_HISTORY,
};
pub use inventory::Inventory;
pub use processors::{
    device_node, pin, place_process, Core, DeviceNode, MemoryNode, PciLocation, ProcessPlacement,
    Processors, ProcessorsReportedBy,
};
pub use status::{RuntimeStatus, Sample, Status};
pub use volume::{identify as volume_identity, volume_available};
