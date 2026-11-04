//! What the machine has: the reports
//! the backend crates and the OS give of it, gathered into the inventory of
//! its topology (`seismic_topology`), and its status samples.

#[cfg(windows)]
mod adapters;
mod gather;
mod host;
mod inventory;
mod status;
mod volume;

#[cfg(test)]
mod tests;

pub use gather::DiscoveryError;
pub use host::{
    DisplacementWindow, HostDisplacement, HostStatus, KernelPressure, LimitVisibility,
    ObservationError, ProcessLimitKind, ProcessMemoryLimit, DISPLACEMENT_HISTORY,
};
pub use inventory::Inventory;
pub use status::{RuntimeStatus, Sample, Status};
pub use volume::volume_available;
