//! What the machine is: the device topology as data, and the rules that build it from what the backends and
//! the OS report (`seismic-discovery` reads them).
//!
//! The topology is a graph of physical devices, executors (one backend's
//! handle on a device) and stores (host RAM, device memory, volumes), with
//! access paths from executors to stores, each under its grant, transfer
//! paths between stores, and each executor's kinds and placement of every
//! buffer contract. Every value is reported, carrying the API that reported it; the
//! topology measures nothing. Its executor identities key what
//! `seismic-measure` measures.

#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
mod kind;
mod model;
mod normalized;
mod placement;
mod processors;
mod reports;
mod selector;
mod system;

#[cfg(test)]
mod tests;

pub use kind::{HostReads, Kind, KindMemory, KindReport, NativeKind, Reach};
pub use model::{
    resolve, AccessPath, AccessPathId, AllocationLimit, Availability, CacheSize, DeviceMemory,
    DeviceTopology, Diagnostic, DiagnosticSource, Executor, ExecutorId, ExecutorKind, Grant,
    GrantKind, GrantSource, Host, Identity, KindId, Limits, NativeAllocations, PhysicalDevice,
    PhysicalDeviceId, PhysicalIdentity, Reported, ReportedBy, Revision, Store, StoreId,
    StoreIdentity, StoreKind, TransferPath, TransferPathId, VolumeId,
};
pub use placement::{place, Contract, WINDOW_BYTES};
pub use processors::{flat_nodes, Core, HostNode, MemoryNode, PciLocation, Processors};
pub use reports::{
    build, AdapterReport, CpuReport, CudaMemory, CudaReport, MetalReport, Reports, VolumeReport,
    VulkanHeap, VulkanMemory, VulkanReport,
};
pub use seismic_lang::registry::BackendName;
pub use selector::{ExecutorSelector, SelectorParseError};
pub use system::{label, parse_smbios, Firmware, FormFactor, HardwareIdentity, SystemIdentity};
