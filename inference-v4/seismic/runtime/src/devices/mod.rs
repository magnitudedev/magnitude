//! The one Seismic device inventory: host and backend discovery, device
//! identity, memory backing, scoped observations and device opening.
//!
//! Public concepts are devices and memory pools. A [`DeviceTopology`] is an
//! immutable, validated snapshot: every pool a device references exists in
//! the same snapshot, and required fields are established facts, never
//! placeholders. Configurations whose physical backing is not normalized
//! say so explicitly ([`DeviceMemory::Unsupported`]) without hiding the
//! device. Memory status is a separately sampled observation. Native handles
//! stay in private descriptors.

mod catalog;
mod host;

pub use crate::memory::MemoryUsage;
pub use catalog::Catalog;
pub use host::{
    HeadroomBasis, HeadroomEstimate, HostMeasurements, HostMemoryStatus, LimitVisibility,
    PressureLevel, ProcessLimitKind, ProcessMemoryLimit,
};

use seismic_compiler::errors::TargetError;
use seismic_lang::registry::BackendName;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::SystemTime;

/// Snapshot-local device identity, valid only within the topology revision
/// that issued it. It is not a persistent or cross-process identity; use
/// [`DeviceSelector`] for that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub(crate) revision: u64,
    pub(crate) index: u32,
}

/// Snapshot-local memory pool identity. Distinct from [`DeviceId`] so the
/// two reference kinds cannot be mixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryPoolId {
    pub(crate) revision: u64,
    pub(crate) index: u32,
}

/// Same-machine selection identity for handing a chosen device to another
/// process (for example a managed worker), which resolves it against its own
/// catalog. It names the native device identity, never an ordinal. It is not
/// guaranteed to survive hardware reconfiguration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeviceSelector {
    /// The host CPU execution target of this machine.
    HostCpu,
    /// `MTLDevice.registryID`.
    Metal { registry_id: u64 },
    /// `cuDeviceGetUuid_v2`: the exposed device or MIG partition.
    Cuda { uuid: [u8; 16] },
    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    Vulkan { uuid: [u8; 16] },
}

fn write_uuid(f: &mut fmt::Formatter<'_>, uuid: &[u8; 16]) -> fmt::Result {
    for (index, byte) in uuid.iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            f.write_str("-")?;
        }
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostCpu => f.write_str("host-cpu"),
            Self::Metal { registry_id } => write!(f, "metal:{registry_id:016x}"),
            Self::Cuda { uuid } => {
                f.write_str("cuda:")?;
                write_uuid(f, uuid)
            }
            Self::Vulkan { uuid } => {
                f.write_str("vulkan:")?;
                write_uuid(f, uuid)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectorParseError(String);

impl fmt::Display for SelectorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid device selector `{}`", self.0)
    }
}
impl std::error::Error for SelectorParseError {}

impl FromStr for DeviceSelector {
    type Err = SelectorParseError;

    /// Parses exactly the [`fmt::Display`] form.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || SelectorParseError(text.to_owned());
        if text == "host-cpu" {
            return Ok(Self::HostCpu);
        }
        if let Some(id) = text.strip_prefix("metal:") {
            return u64::from_str_radix(id, 16)
                .map(|registry_id| Self::Metal { registry_id })
                .map_err(|_| invalid());
        }
        let (vulkan, text) = match (text.strip_prefix("cuda:"), text.strip_prefix("vulkan:")) {
            (Some(uuid), _) => (false, uuid),
            (_, Some(uuid)) => (true, uuid),
            (None, None) => return Err(invalid()),
        };
        let hex = text.replace('-', "");
        if hex.len() != 32 || !hex.is_ascii() {
            return Err(invalid());
        }
        let mut uuid = [0u8; 16];
        for (index, byte) in uuid.iter_mut().enumerate() {
            *byte =
                u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map_err(|_| invalid())?;
        }
        Ok(if vulkan {
            Self::Vulkan { uuid }
        } else {
            Self::Cuda { uuid }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// Whether this runtime can open the device through its backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Available,
    Unavailable { reason: String },
}

/// A device's memory relationships. `Unsupported` is an explicit result for
/// a configuration whose physical backing is not normalized; the device may
/// still execute, but managed capacity assessment cannot use it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceMemory {
    Established(DeviceMemoryInfo),
    Unsupported { reason: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemoryInfo {
    /// The pool this device's allocations consume.
    pub allocation_pool: MemoryPoolId,
    /// The host pool used for host-side copies (loading, staging). Equal to
    /// `allocation_pool` when the device allocates host RAM.
    pub host_pool: MemoryPoolId,
    /// The single-allocation limit (`TargetLimits::max_allocation_bytes`).
    /// Neither capacity nor availability.
    pub max_allocation_bytes: u64,
}

impl DeviceMemoryInfo {
    /// Device allocations are host RAM: one backing, charged once.
    pub fn allocates_host_memory(&self) -> bool {
        self.allocation_pool == self.host_pool
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPoolKind {
    HostRam,
    DeviceLocal,
}

/// The definition of a pool's capacity figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityBasis {
    /// macOS `hw.memsize`: installed physical RAM.
    InstalledRam,
    /// Linux `MemTotal` / Windows `ullTotalPhys`: RAM usable by the OS after
    /// firmware and kernel reservations.
    OsUsableRam,
    /// CUDA `cuDeviceTotalMem`: memory of the exposed device or partition.
    CudaDeviceTotal,
    /// The size of the Vulkan device's largest device-local heap.
    VulkanDeviceLocalHeap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPoolInfo {
    pub id: MemoryPoolId,
    pub kind: MemoryPoolKind,
    pub capacity_bytes: u64,
    pub basis: CapacityBasis,
    pub(crate) ledger: LedgerKey,
}

/// Physical identity of an accounting ledger; stable across refreshes so
/// live allocations keep one shared ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LedgerKey {
    Host,
    Cuda([u8; 16]),
    #[cfg(not(target_os = "macos"))]
    Vulkan([u8; 16]),
    /// A device whose backing is not normalized accounts privately.
    Unestablished(DeviceSelector),
}

#[derive(Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub selector: DeviceSelector,
    pub name: String,
    pub kind: DeviceKind,
    pub backend: BackendName,
    pub availability: Availability,
    pub memory: DeviceMemory,
    pub(crate) descriptor: Arc<crate::backends::Descriptor>,
}

impl DeviceInfo {
    /// The device's advisory working-set ceiling, when its backend reports
    /// one without opening an execution context. This bounds stable fit on
    /// unified-memory Metal alongside the host RAM domain.
    pub fn recommended_working_set_bytes(&self) -> Option<u64> {
        #[cfg(target_os = "macos")]
        if let crate::backends::Descriptor::Metal { handle } = self.descriptor.as_ref() {
            return Some(handle.recommended_working_set_bytes());
        }
        None
    }
}

impl fmt::Debug for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceInfo")
            .field("id", &self.id)
            .field("selector", &self.selector)
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("backend", &self.backend)
            .field("availability", &self.availability)
            .field("memory", &self.memory)
            .finish()
    }
}

impl PartialEq for DeviceInfo {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.selector == other.selector
            && self.name == other.name
            && self.kind == other.kind
            && self.backend == other.backend
            && self.availability == other.availability
            && self.memory == other.memory
    }
}
impl Eq for DeviceInfo {}

/// A backend enumeration problem retained from discovery. Absent drivers
/// and devices that cannot be described are reported here, not hidden.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveryDiagnostic {
    pub backend: BackendName,
    pub message: String,
}

/// One immutable inventory revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTopology {
    revision: u64,
    devices: Vec<DeviceInfo>,
    pools: Vec<MemoryPoolInfo>,
    diagnostics: Vec<DiscoveryDiagnostic>,
}

impl DeviceTopology {
    /// Changes when identity, backing or access environment changes, not on
    /// allocation.
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }
    pub fn pools(&self) -> &[MemoryPoolInfo] {
        &self.pools
    }
    pub fn diagnostics(&self) -> &[DiscoveryDiagnostic] {
        &self.diagnostics
    }
    /// `None` for an identifier from another revision.
    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        (id.revision == self.revision).then(|| &self.devices[id.index as usize])
    }
    /// `None` for an identifier from another revision.
    pub fn pool(&self, id: MemoryPoolId) -> Option<&MemoryPoolInfo> {
        (id.revision == self.revision).then(|| &self.pools[id.index as usize])
    }
    /// The host RAM pool; every topology has exactly one.
    pub fn host_pool(&self) -> &MemoryPoolInfo {
        &self.pools[0]
    }
}

/// One sampled device-scoped observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMemoryStatus {
    pub sampled_at: SystemTime,
    pub measurements: DeviceMeasurements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceMeasurements {
    /// The device allocates host RAM with no device-specific budget; the
    /// host status is the applicable observation.
    Host,
    Metal {
        /// `recommendedMaxWorkingSetSize`: advisory ceiling for this device's
        /// working set, not additional RAM.
        recommended_working_set_bytes: u64,
        /// `currentAllocatedSize`: resources this process holds on the
        /// device, including non-Seismic resources.
        current_allocated_bytes: u64,
    },
    Cuda {
        /// `cuMemGetInfo`: device-wide free memory; this process's and other
        /// processes' allocations are already excluded.
        free_bytes: u64,
        /// `cuMemGetInfo`: total memory of the exposed device.
        total_bytes: u64,
    },
    Vulkan {
        /// `VK_EXT_memory_budget` of the device-local heap: what this
        /// process may use.
        heap_budget_bytes: u64,
        /// What this process uses of it.
        heap_usage_bytes: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    /// Host RAM capacity could not be established; no topology is complete
    /// without it.
    HostMemory(String),
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostMemory(message) => write!(f, "host memory discovery failed: {message}"),
        }
    }
}
impl std::error::Error for DiscoveryError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// The identifier belongs to an earlier topology revision.
    Stale(DeviceId),
    /// No device of `backend` was discovered; `diagnostics` are that
    /// backend's discovery diagnostics (why it has none, when it said).
    NoDevice {
        backend: BackendName,
        diagnostics: Vec<String>,
    },
    Unavailable {
        selector: DeviceSelector,
        reason: String,
    },
    /// The native device at this descriptor no longer has the discovered
    /// identity.
    IdentityChanged(DeviceSelector),
    Backend(TargetError),
    /// The device is already open with a different artifact store.
    ArtifactStoreConflict(DeviceSelector),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stale(id) => write!(
                f,
                "device id {id:?} belongs to an earlier topology revision"
            ),
            Self::NoDevice {
                backend,
                diagnostics,
            } => {
                write!(f, "no {} device was discovered", backend.as_str())?;
                if !diagnostics.is_empty() {
                    write!(f, ": {}", diagnostics.join("; "))?;
                }
                Ok(())
            }
            Self::Unavailable { selector, reason } => {
                write!(f, "device {selector} is unavailable: {reason}")
            }
            Self::IdentityChanged(selector) => {
                write!(f, "device {selector} changed identity since discovery")
            }
            Self::Backend(error) => write!(f, "{error}"),
            Self::ArtifactStoreConflict(selector) => {
                write!(
                    f,
                    "device {selector} is already open with a different artifact store"
                )
            }
        }
    }
}
impl std::error::Error for OpenError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    Missing(DeviceSelector),
    Ambiguous {
        selector: DeviceSelector,
        matches: usize,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(selector) => write!(f, "device {selector} is not present"),
            Self::Ambiguous { selector, matches } => {
                write!(f, "device {selector} matches {matches} devices")
            }
        }
    }
}
impl std::error::Error for ResolveError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationError {
    /// This platform/configuration offers no qualified facility for the
    /// observation.
    Unsupported(String),
    /// The native query failed.
    Failed(String),
}

impl fmt::Display for ObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => write!(f, "memory observation unsupported: {message}"),
            Self::Failed(message) => write!(f, "memory observation failed: {message}"),
        }
    }
}
impl std::error::Error for ObservationError {}
