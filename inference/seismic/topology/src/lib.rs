//! What the machine is (specs/26-10-18/device-topology.md): the device
//! topology as data, and the rules that build it from what the backends and
//! the OS report (`seismic-discovery` reads them).
//!
//! The topology is a graph of physical devices, executors (one backend's
//! handle on a device) and stores (host RAM, device memory, volumes), with
//! access paths from executors to stores, each under its grant, transfer
//! paths between stores, and each executor's placement of every storage
//! kind. Every value is reported, carrying the API that reported it; the
//! topology measures nothing. Its executor identities key what
//! `seismic-measure` measures.

#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
mod model;
mod normalized;
mod reports;
mod selector;

#[cfg(test)]
mod tests;

pub use model::{
    AccessPath, AccessPathId, AccessRoute, Availability, DeviceMemory, DeviceTopology, Diagnostic,
    DiagnosticSource, Executor, ExecutorId, ExecutorKind, GrantSource, Identity, Limits,
    PhysicalDevice, PhysicalDeviceId, PhysicalIdentity, Reported, ReportedBy, Revision,
    StorageKind, Store, StoreId, StoreIdentity, StoreKind, TransferPath, TransferPathId, VolumeId,
};
pub use reports::{
    build, AdapterReport, CpuReport, CudaMemory, CudaReport, MetalReport, Reports, VolumeReport,
    VulkanHeap, VulkanMemory, VulkanReport,
};
pub use seismic_lang::registry::BackendName;
pub use selector::{ExecutorSelector, SelectorParseError};
