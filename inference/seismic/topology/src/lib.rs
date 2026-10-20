//! What the machine is (specs/26-10-18/device-topology.md): the device
//! topology, its discovery through the backend crates and the OS, and
//! status samples.
//!
//! The topology is a graph of physical devices, executors (one backend's
//! handle on a device) and stores (host RAM, device memory, volumes), with
//! access paths from executors to stores, each under its grant, transfer
//! paths between stores, and each executor's placement of every storage
//! kind. Every value is reported, carrying the API that reported it; the
//! topology measures nothing. Its executor identities key what
//! `seismic-measure` measures.

#[cfg(windows)]
mod adapters;
mod discovery;
#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
mod host;
mod inventory;
mod model;
mod normalized;
mod selector;
mod status;
mod volume;

#[cfg(test)]
mod tests;

pub use discovery::DiscoveryError;
pub use host::{
    DisplacementWindow, HostDisplacement, HostStatus, KernelPressure, LimitVisibility,
    ObservationError, ProcessLimitKind, ProcessMemoryLimit, DISPLACEMENT_HISTORY,
};
pub use inventory::Inventory;
pub use model::{
    AccessPath, AccessPathId, AccessRoute, Availability, DeviceMemory, DeviceTopology, Diagnostic,
    DiagnosticSource, Executor, ExecutorId, ExecutorKind, Identity, Limits, PhysicalDevice,
    PhysicalDeviceId, PhysicalIdentity, Reported, ReportedBy, Revision, StorageKind, Store,
    StoreId, StoreIdentity, StoreKind, TransferPath, TransferPathId, VolumeId,
};
pub use seismic_lang::registry::BackendName;
pub use selector::{ExecutorSelector, SelectorParseError};
pub use status::{RuntimeStatus, Sample, Status};
pub use volume::volume_available;
