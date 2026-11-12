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
mod reports;
mod selector;

#[cfg(test)]
mod tests;

pub use kind::{
    HostReads, Kind, KindMemory, KindReport, NativeKind, Obtained, Reach, Reserving, Unreservable,
};
pub use model::{
    AccessPath, AccessPathId, AllocationLimit, Availability, DeviceAllocationBound, DeviceMemory, DeviceTopology,
    Diagnostic, DiagnosticSource, Executor, ExecutorId, ExecutorKind, GrantSource, Identity,
    KindId, Limits, Packing, PhysicalDevice, PhysicalDeviceId, PhysicalIdentity, Reported,
    ReportedBy, Reservation,
    ReservationCharges, ReservationChunks, ReservedRecords, Revision, Store, StoreId,
    StoreIdentity, StoreKind, TableCharge, TransferPath, TransferPathId, VolumeId,
};
pub use placement::{place, reserving, Contract, Placing, ReservedContract, WINDOW_BYTES};
pub use reports::{
    build, AdapterReport, CpuReport, CudaMemory, CudaReport, MetalReport, Reports, VolumeReport,
    VulkanHeap, VulkanMemory, VulkanReport,
};
pub use seismic_lang::registry::BackendName;
pub use selector::{ExecutorSelector, SelectorParseError};
