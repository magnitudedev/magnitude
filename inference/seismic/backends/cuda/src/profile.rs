//! CUDA opened-device acquisition. Device-wide legality and numerical specs
//! form `DeviceDescription<Cuda>`; fixed service probes on the production
//! context/stream form its separately identity-bound `ExecutionProfile`.
//! Concrete function resources are reflected only during native reconciliation.
//!
//! Specs are exact device/driver reports except where a field's
//! documentation names it a backend rule.

use crate::capability;
use crate::driver::{Allocation, Context, Driver, DriverError, Event, Stream};
use crate::nvrtc;
use crate::{
    Cuda, CudaPtxPrograms, SERVICE_ATOMIC, SERVICE_BARRIER, SERVICE_GLOBAL_READ, SERVICE_GLOBAL_WRITE,
    SERVICE_INSTRUCTION, SERVICE_MATRIX, SERVICE_REPACK, SERVICE_SHARED_READ, SERVICE_SHARED_WRITE,
    SERVICE_SUBGROUP,
};
use seismic_compiler::errors::TargetError;
use seismic_compiler::target::{
    assemble_device_description, DiscoveredTarget, ExecutionProfileParts,
};
use seismic_estimator::{
    AnalyticalService, CoreService, CompositionQualificationCase, CompositionQualificationParts, DurationInterval, FactProvenance,
    MeasurementBatch, MeasurementSeries, ProfileAcquisitionMetrics, ResourceTopology,
    ServiceAccuracyClass, ServiceClassId, ServiceCorrelationId, ServiceCurve, ServiceCurveRegime,
    ServiceDefinition, ServiceQualificationDomain,
};
const SERVICE_SUBMISSION: ServiceClassId = ServiceClassId::new("core.submission");
const SERVICE_COPY: ServiceClassId = ServiceClassId::new("core.copy");
const SERVICE_FILL: ServiceClassId = ServiceClassId::new("core.fill");
const SERVICE_SCALAR_READ: ServiceClassId = ServiceClassId::new("core.scalar-read");
const SERVICE_ALLOCATION_INSTANCE: ServiceClassId = ServiceClassId::new("core.allocation-instance");
const SERVICE_TENSOR_PUBLICATION: ServiceClassId = ServiceClassId::new("core.tensor-publication");
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::{
    DataTypeSupport, KernelAbiAllocation, KernelAbiAllocationRole, KernelAbiFootprint,
    KernelAbiLayout, KernelAbiModel, KernelWordLayout, LocalRealization, LocalRealizationPolicy,
    NumericalEnvironment, TargetLimits, VectorSupport,
};
use seismic_lang::intrinsics::MathOp;
use seismic_lang::registry::{self, BackendName};
use seismic_lang::types::DType;
use seismic_native_target::CompatibilityIdentity;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::{c_int, CStr};
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Revision of the CUDA backend implementation: changes whenever emission,
/// resource rules, or the implemented intrinsic set change (§15.1).
pub const BACKEND_REVISION: &str = "seismic-cuda-v12";

/// CUDA kernels receive one pointer to the backend's canonical device-side
/// launch frame; every interface table is reached through that pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CudaKernelAbi;

impl KernelAbiModel<Cuda> for CudaKernelAbi {
    fn layout(&self, kernel: &Kernel<Cuda>) -> KernelAbiLayout {
        let interface = kernel.interface();
        // Core owns the sole word-table schema. The CUDA ABI sizes the
        // allocation from that schema instead of maintaining a second
        // binding/local/geometry word-count formula.
        let words = u64::from(KernelWordLayout::for_kernel(kernel).total);
        let bytes = |words: u64| {
            words
                .checked_mul(8)
                .expect("closed CUDA ABI allocation size overflows u64")
        };
        let allocations = vec![
            KernelAbiAllocation {
                role: KernelAbiAllocationRole::BufferTable,
                bytes: bytes(
                    u64::try_from(interface.bindings.len())
                        .expect("closed CUDA binding inventory exceeds u64"),
                )
                .max(8),
                alignment: 8,
            },
            KernelAbiAllocation {
                role: KernelAbiAllocationRole::WordTable,
                bytes: bytes(words),
                alignment: 8,
            },
            KernelAbiAllocation {
                role: KernelAbiAllocationRole::ScalarResults,
                bytes: bytes(
                    u64::try_from(interface.result_slots.len())
                        .expect("closed CUDA result inventory exceeds u64"),
                )
                .max(8),
                alignment: 8,
            },
            KernelAbiAllocation {
                role: KernelAbiAllocationRole::LaunchFrame,
                bytes: 5 * 8,
                alignment: 8,
            },
        ];
        KernelAbiLayout {
            footprint: KernelAbiFootprint {
                bytes: 8,
                alignment: 8,
            },
            allocations,
        }
    }
}

/// The CUDA kernel parameter space in bytes, by driver generation: 32764
/// bytes with driver API 12.1+ on Volta and later, 4096 bytes before.
const KERNEL_PARAMETER_BYTES_LARGE: u64 = 32_764;
const KERNEL_PARAMETER_BYTES_CLASSIC: u64 = 4_096;

pub(crate) const fn kernel_parameter_bytes(driver_api: DriverApiVersion) -> u64 {
    if driver_api.0 >= DriverApiVersion::LARGE_PARAMETERS_MINIMUM.0 {
        KERNEL_PARAMETER_BYTES_LARGE
    } else {
        KERNEL_PARAMETER_BYTES_CLASSIC
    }
}

/// Per-device CUDA launch limits queried once from driver attributes, shared
/// by target discovery and direct launch validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceLaunchLimits {
    pub(crate) max_grid: [u64; 3],
    pub(crate) max_block: [u64; 3],
    pub(crate) max_shared_bytes: u64,
    pub(crate) max_param_bytes: u64,
}

impl DeviceLaunchLimits {
    pub(crate) fn query(driver: &Driver, ordinal: c_int) -> Result<Self, TargetError> {
        let unsigned = |key: i32| -> Result<u64, TargetError> {
            let value = driver.attribute(key, ordinal).map_err(unavailable)?;
            u32::try_from(value).map(u64::from).map_err(|_| {
                TargetError::UnsupportedDevice(format!("device attribute {key} reports {value}"))
            })
        };
        let version = driver.version().map_err(unavailable)?;
        let driver_api = DriverApiVersion(u32::try_from(version).map_err(|_| {
            TargetError::UnsupportedDriver(format!("driver API version {version}"))
        })?);
        Ok(Self {
            max_grid: [
                unsigned(attr::MAX_GRID_DIM_X)?,
                unsigned(attr::MAX_GRID_DIM_Y)?,
                unsigned(attr::MAX_GRID_DIM_Z)?,
            ],
            max_block: [
                unsigned(attr::MAX_BLOCK_DIM_X)?,
                unsigned(attr::MAX_BLOCK_DIM_Y)?,
                unsigned(attr::MAX_BLOCK_DIM_Z)?,
            ],
            max_shared_bytes: unsigned(attr::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
            max_param_bytes: kernel_parameter_bytes(driver_api),
        })
    }
}

/// Documented per-thread local memory limit of every supported compute
/// capability (CUDA programming guide, compute capability 2.x and later).
const LOCAL_BYTES_PER_THREAD: u64 = 512 * 1024;

/// Documented per-thread register ceiling of every supported compute
/// capability: the device's `registers_per_thread`.
const MAX_REGISTERS_PER_THREAD: u64 = 255;
/// Backend emission contract (`ptx::.maxnreg`), used by cooperative-grid
/// residency constraints. This is not driver reflection.
pub const CODEGEN_REGISTERS_PER_THREAD: u32 = 64;

/// `cuMemAlloc` returns allocations aligned to at least 256 bytes.
const ALLOCATION_ALIGNMENT: u64 = 256;

/// A CUDA compute capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u16,
    pub minor: u16,
}

impl ComputeCapability {
    /// The emission floor: PTX for `sm_80` runs forward-compatibly on every
    /// later device.
    pub const SM80: Self = Self { major: 8, minor: 0 };

    /// The first member of NVIDIA's Blackwell 10.x family. Family-specific
    /// PTX is used only when the opened driver's PTX toolchain also supports
    /// that target.
    pub const SM100: Self = Self {
        major: 10,
        minor: 0,
    };
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// The integer of `cuDriverGetVersion` (`major * 1000 + minor * 10`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriverApiVersion(pub u32);

impl DriverApiVersion {
    /// The oldest driver guaranteed to load the bundled NVRTC's CUBIN: the
    /// first driver of the bundled CUDA major (minor-version compatibility;
    /// R525 for CUDA 12). It also covers the PTX 7.1 of generated kernels
    /// (CUDA 11.1).
    pub const MINIMUM: Self = Self(nvrtc::BUNDLED_MAJOR * 1000);
    /// The 32 KiB kernel parameter space arrived with CUDA 12.1.
    pub const LARGE_PARAMETERS_MINIMUM: Self = Self(12_010);
    /// CUDA 12.9 carries PTX ISA 8.8, the first PTX revision with the
    /// `sm_100f` family target and family-level tcgen05 contract.
    pub const PTX_88_MINIMUM: Self = Self(12_090);

    pub fn major(self) -> u32 {
        self.0 / 1000
    }
    pub fn minor(self) -> u32 {
        (self.0 % 1000) / 10
    }
}

/// What the backend's kernels require of a device: the `sm_80` emission
/// floor and the driver floor.
fn admit_device(
    name: &str,
    compute_capability: ComputeCapability,
    driver_api: DriverApiVersion,
) -> Result<(), TargetError> {
    if compute_capability < ComputeCapability::SM80 {
        return Err(TargetError::UnsupportedDevice(format!(
            "{name}: compute capability {compute_capability} is below the sm_80 emission floor"
        )));
    }
    admit_driver(driver_api)
}

/// The driver floor of the backend: below it, the bundled NVRTC's CUBIN is
/// not guaranteed to load.
fn admit_driver(driver_api: DriverApiVersion) -> Result<(), TargetError> {
    if driver_api < DriverApiVersion::MINIMUM {
        return Err(TargetError::UnsupportedDriver(format!(
            "driver API {driver_api} is below {}: the bundled NVRTC {} requires a CUDA {} driver",
            DriverApiVersion::MINIMUM,
            nvrtc::BUNDLED_MAJOR,
            nvrtc::BUNDLED_MAJOR,
        )));
    }
    Ok(())
}

/// Whether the device takes the `sm_100f` family target: PTX ISA 8.8 defines
/// the family for compute capabilities 10.0 and 10.3.
fn in_blackwell_100_family(
    compute_capability: ComputeCapability,
    driver_api: DriverApiVersion,
) -> bool {
    compute_capability.major == ComputeCapability::SM100.major
        && matches!(compute_capability.minor, 0 | 3)
        && driver_api >= DriverApiVersion::PTX_88_MINIMUM
}

impl fmt::Display for DriverApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

/// The PTX virtual ISA every kernel of this backend is emitted for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PtxTarget {
    pub isa_major: u16,
    pub isa_minor: u16,
    /// Numeric SM spelling: 80 is `sm_80`.
    pub architecture: u16,
    pub feature_set: PtxFeatureSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PtxFeatureSet {
    Baseline,
    Family,
}

impl PtxTarget {
    /// PTX 7.1 for `sm_80`: the scalar baseline plus `mma.sync` m16n8k16
    /// and `cvt.bf16`.
    pub const BASELINE: Self = Self {
        isa_major: 7,
        isa_minor: 1,
        architecture: 80,
        feature_set: PtxFeatureSet::Baseline,
    };

    /// PTX 8.8's Blackwell-family target. Unlike a baseline `sm_100`
    /// target, `sm_100f` admits the tcgen05/TMEM instruction family and is
    /// valid only on members of that documented family.
    pub const BLACKWELL_100_FAMILY: Self = Self {
        isa_major: 8,
        isa_minor: 8,
        architecture: 100,
        feature_set: PtxFeatureSet::Family,
    };

    pub fn header(self) -> String {
        format!(
            ".version {}.{}\n.target sm_{}{}\n.address_size 64\n",
            self.isa_major,
            self.isa_minor,
            self.architecture,
            match self.feature_set {
                PtxFeatureSet::Baseline => "",
                PtxFeatureSet::Family => "f",
            }
        )
    }
}

/// Native tensor-memory availability of the exact opened target. The
/// dimensions and allocation unit are architectural properties of the
/// `sm_100f` family, not byte-storage estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorMemory {
    Unavailable,
    Tcgen05 {
        columns: u32,
        lanes: u32,
        cell_bits: u32,
        allocation_granularity_columns: u32,
    },
}

/// Everything CUDA knows about one opened device before planning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaDeviceInfo {
    /// Driver device ordinal; the primary context of this ordinal is what
    /// native compilation and execution retain.
    pub device_ordinal: u32,
    pub compute_capability: ComputeCapability,
    pub driver_api: DriverApiVersion,
    pub ptx: PtxTarget,
    pub tensor_memory: TensorMemory,
    /// The device's properties (`seismic_lang::device::cuda`): its
    /// multiprocessors, warp, block, register and shared-memory limits, L2
    /// and cooperative launch. No reliable pre-JIT register pressure model
    /// exists for PTX, so its register limits are reported specs, not constraints.
    pub properties: Properties,
    /// Fixed physical-register ceiling emitted on every entry.
    pub codegen_registers_per_thread: u32,
    /// `cuDeviceTotalMem`.
    pub global_memory_bytes: u64,
    /// The kernel parameter space, by driver generation.
    pub kernel_parameter_bytes: u64,
}

pub use seismic_lang::device::cuda::Properties;

/// The device attributes a device's properties are established from
/// (`seismic_lang::device::cuda`), read once.
struct Attributes {
    multiprocessor_count: u64,
    warp_size: u64,
    max_threads_per_block: u64,
    max_threads_per_multiprocessor: u64,
    max_shared_memory_per_block_optin: u64,
    l2_cache_size: u64,
    cooperative_launch: bool,
    max_registers_per_block: u64,
}

impl seismic_lang::device::cuda::Queries for Attributes {
    fn multiprocessor_count(&self) -> Option<u64> {
        Some(self.multiprocessor_count)
    }
    fn warp_size(&self) -> Option<u64> {
        Some(self.warp_size)
    }
    fn max_threads_per_block(&self) -> Option<u64> {
        Some(self.max_threads_per_block)
    }
    fn max_threads_per_multiprocessor(&self) -> Option<u64> {
        Some(self.max_threads_per_multiprocessor)
    }
    fn max_shared_memory_per_block_optin(&self) -> Option<u64> {
        Some(self.max_shared_memory_per_block_optin)
    }
    fn l2_cache_size(&self) -> Option<u64> {
        Some(self.l2_cache_size)
    }
    fn cooperative_launch(&self) -> Option<bool> {
        Some(self.cooperative_launch)
    }
    fn max_registers_per_thread(&self) -> Option<u64> {
        Some(MAX_REGISTERS_PER_THREAD)
    }
    fn max_registers_per_block(&self) -> Option<u64> {
        Some(self.max_registers_per_block)
    }
}

/// Device attribute keys (`CUdevice_attribute`).
mod attr {
    pub const PCI_BUS_ID: i32 = 33;
    pub const PCI_DEVICE_ID: i32 = 34;
    pub const PCI_DOMAIN_ID: i32 = 50;
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const MAX_BLOCK_DIM_X: i32 = 2;
    pub const MAX_BLOCK_DIM_Y: i32 = 3;
    pub const MAX_BLOCK_DIM_Z: i32 = 4;
    pub const MAX_GRID_DIM_X: i32 = 5;
    pub const MAX_GRID_DIM_Y: i32 = 6;
    pub const MAX_GRID_DIM_Z: i32 = 7;
    pub const WARP_SIZE: i32 = 10;
    pub const MAX_REGISTERS_PER_BLOCK: i32 = 12;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const L2_CACHE_SIZE: i32 = 38;
    pub const MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;
    pub const COOPERATIVE_LAUNCH: i32 = 99;
    pub const INTEGRATED: i32 = 18;
    /// `CU_DEVICE_ATTRIBUTE_NUMA_CONFIG`: whether the device's memory is an
    /// OS NUMA node.
    pub const NUMA_CONFIG: i32 = 130;
}

fn unavailable(error: DriverError) -> TargetError {
    match error.is_busy() {
        true => TargetError::DeviceBusy(error.to_string()),
        false => TargetError::DeviceUnavailable(error.to_string()),
    }
}

/// The single-allocation limit of a CUDA device: one `cuMemAlloc` can at
/// most cover the device's total memory. This is the `TargetLimits` value and
/// the executor's reported allocation limit in the topology; it is not
/// available memory.
pub fn max_allocation_bytes(total_memory_bytes: u64) -> u64 {
    total_memory_bytes
}

/// Cheap unopened device identity, as discovery reads it. Constructing this
/// value does not retain a primary context, create a stream, run probes, or
/// assemble a profile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceDescriptor {
    pub ordinal: u32,
    /// `cuDeviceGetUuid_v2`: the exposed device or MIG partition, never
    /// merely its parent GPU.
    pub uuid: [u8; 16],
    pub name: String,
    /// `cuDeviceTotalMem`: memory of the exposed device or partition.
    pub total_memory_bytes: u64,
    /// `cuMemGetAllocationGranularity` (recommended) of its device memory:
    /// what one allocation's size is rounded up to.
    pub allocation_granularity: u64,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`: the device shares host memory.
    pub integrated: bool,
    /// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`: zero for a device without one.
    pub l2_cache_bytes: u64,
    /// `cuDriverGetVersion`.
    pub driver_api: DriverApiVersion,
    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` / `_MINOR`.
    pub compute_capability: ComputeCapability,
    /// `CU_DEVICE_ATTRIBUTE_NUMA_CONFIG`: the device's memory is an OS NUMA
    /// node, which the OS manages as host memory.
    pub numa_node: bool,
    /// `cuDeviceGetLuid`: the OS adapter's LUID, in `LUID` memory layout,
    /// where the driver reports one (WDDM). It identifies nothing across
    /// boots; it matches the memory the OS reports as dedicated to the
    /// adapter.
    pub luid: Option<[u8; 8]>,
    /// `CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`, `_PCI_BUS_ID`, `_PCI_DEVICE_ID`:
    /// where the device sits on its bus (domain, bus, device).
    pub pci: [u32; 3],
    /// The device's hardware properties (`seismic_lang::device::cuda`),
    /// read from CUDA device attributes without creating a context.
    pub properties: Properties,
}

impl DeviceDescriptor {
    /// Whether the backend's kernels can run on the device: the reason when
    /// they cannot.
    pub fn admission(&self) -> Result<(), TargetError> {
        admit_device(&self.name, self.compute_capability, self.driver_api)
    }

    /// Whether the device's memory is the host's: an integrated device's,
    /// or one the OS manages as a NUMA node.
    pub fn memory_is_host(&self) -> bool {
        self.integrated || self.numa_node
    }
}

fn read_compute_capability(
    driver: &Driver,
    device: c_int,
) -> Result<ComputeCapability, TargetError> {
    let unsigned = |key: i32| -> Result<u16, TargetError> {
        let value = driver.attribute(key, device).map_err(unavailable)?;
        u16::try_from(value).map_err(|_| {
            TargetError::UnsupportedDevice(format!("device attribute {key} reports {value}"))
        })
    };
    Ok(ComputeCapability {
        major: unsigned(attr::COMPUTE_CAPABILITY_MAJOR)?,
        minor: unsigned(attr::COMPUTE_CAPABILITY_MINOR)?,
    })
}

/// Number of CUDA devices the driver reports; zero (or no driver) is
/// `DeviceUnavailable`.
pub fn device_count() -> Result<u32, TargetError> {
    let driver = Driver::load().map_err(TargetError::DeviceUnavailable)?;
    let mut count = 0;
    unsafe {
        driver
            .check((driver.device_count)(&mut count), "device count")
            .map_err(unavailable)?;
    }
    u32::try_from(count)
        .map_err(|_| TargetError::DeviceUnavailable(format!("driver reports {count} devices")))
}

/// Decode a fixed-size C buffer filled by `cuDeviceGetName`, forcing a
/// trailing NUL byte before calling `CStr::from_ptr`.
fn decode_device_name(buffer: &mut [std::ffi::c_char]) -> String {
    let Some(last) = buffer.last_mut() else {
        return String::new();
    };
    *last = 0;
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub fn describe(ordinal: u32) -> Result<DeviceDescriptor, TargetError> {
    let driver = Driver::load().map_err(TargetError::DeviceUnavailable)?;
    let ordinal_c = c_int::try_from(ordinal).map_err(|_| {
        TargetError::DeviceUnavailable(format!("ordinal {ordinal} exceeds the driver ABI"))
    })?;
    let device_uuid_v2 = driver.device_uuid_v2.ok_or_else(|| {
        TargetError::UnsupportedDriver(
            "the CUDA driver lacks cuDeviceGetUuid_v2 (driver API 11.4); exposed-device identity cannot be established"
                .into(),
        )
    })?;
    let mut handle = 0;
    let mut name = [0 as std::ffi::c_char; 256];
    let mut memory = 0usize;
    let mut uuid = [0u8; 16];
    let mut version = 0;
    unsafe {
        driver
            .check((driver.device_get)(&mut handle, ordinal_c), "device selection")
            .map_err(unavailable)?;
        driver
            .check((driver.driver_version)(&mut version), "driver version")
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_name)(name.as_mut_ptr(), name.len() as c_int, handle),
                "device name",
            )
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_total_memory)(&mut memory, handle),
                "total device memory",
            )
            .map_err(unavailable)?;
        driver
            .check(device_uuid_v2(&mut uuid, handle), "device UUID")
            .map_err(unavailable)?;
    }
    let integrated = driver
        .attribute(attr::INTEGRATED, handle)
        .map_err(unavailable)?
        != 0;
    let pci = |key: c_int| {
        let value = driver.attribute(key, handle).map_err(unavailable)?;
        u32::try_from(value).map_err(|_| TargetError::DeviceUnavailable(format!("PCI attribute {key} of {value}")))
    };
    Ok(DeviceDescriptor {
        ordinal,
        uuid,
        name: decode_device_name(&mut name),
        total_memory_bytes: memory as u64,
        allocation_granularity: crate::driver::device_granularity(&driver, handle).map_err(unavailable)?,
        integrated,
        l2_cache_bytes: u64::try_from(
            driver
                .attribute(attr::L2_CACHE_SIZE, handle)
                .map_err(unavailable)?,
        )
        .map_err(|_| TargetError::DeviceUnavailable("negative L2 cache size".into()))?,
        driver_api: DriverApiVersion(u32::try_from(version).map_err(|_| {
            TargetError::UnsupportedDriver(format!("driver API version {version}"))
        })?),
        compute_capability: read_compute_capability(&driver, handle)?,
        numa_node: numa_node(&driver, handle)?,
        luid: luid(&driver, handle)?,
        pci: [pci(attr::PCI_DOMAIN_ID)?, pci(attr::PCI_BUS_ID)?, pci(attr::PCI_DEVICE_ID)?],
        properties: read_properties(&driver, handle)?,
    })
}

fn read_properties(driver: &Driver, handle: c_int) -> Result<Properties, TargetError> {
    let unsigned = |key: c_int| -> Result<u32, TargetError> {
        let value = driver.attribute(key, handle).map_err(unavailable)?;
        u32::try_from(value).map_err(|_| {
            TargetError::UnsupportedDevice(format!("device attribute {key} reports {value}"))
        })
    };
    Ok(seismic_lang::device::cuda::describe(
        &Attributes {
            multiprocessor_count: u64::from(unsigned(attr::MULTIPROCESSOR_COUNT)?),
            warp_size: u64::from(unsigned(attr::WARP_SIZE)?),
            max_threads_per_block: u64::from(unsigned(attr::MAX_THREADS_PER_BLOCK)?),
            max_threads_per_multiprocessor: u64::from(unsigned(
                attr::MAX_THREADS_PER_MULTIPROCESSOR,
            )?),
            max_shared_memory_per_block_optin: u64::from(unsigned(
                attr::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )?),
            l2_cache_size: u64::from(unsigned(attr::L2_CACHE_SIZE)?),
            cooperative_launch: unsigned(attr::COOPERATIVE_LAUNCH)? == 1,
            max_registers_per_block: u64::from(unsigned(attr::MAX_REGISTERS_PER_BLOCK)?),
        },
        &Default::default(),
    )
    .expect("every CUDA property's source is an attribute the driver reports"))
}

/// `CUDA_ERROR_INVALID_VALUE`.
const INVALID_VALUE: i32 = 1;
/// `CUDA_ERROR_NOT_SUPPORTED`.
const NOT_SUPPORTED: i32 = 801;

/// Whether the device's memory is an OS NUMA node
/// (`CU_DEVICE_ATTRIBUTE_NUMA_CONFIG` is `CU_DEVICE_NUMA_CONFIG_NUMA_NODE`).
/// A driver that does not know the attribute rejects it as an invalid value:
/// it states no NUMA node for the device.
fn numa_node(driver: &Driver, device: c_int) -> Result<bool, TargetError> {
    /// `CU_DEVICE_NUMA_CONFIG_NUMA_NODE`.
    const NUMA_NODE: i32 = 1;
    let config = match driver.attribute(attr::NUMA_CONFIG, device) {
        Ok(config) => Some(config == NUMA_NODE),
        Err(error) if error.code == INVALID_VALUE => None,
        Err(error) => return Err(unavailable(error)),
    };
    Ok(seismic_topology::resolve(
        &[("CU_DEVICE_ATTRIBUTE_NUMA_CONFIG", &|| config)],
        "default",
        || false,
    )
    .0)
}

/// `cuDeviceGetLuid`, which only a driver under WDDM supports: elsewhere the
/// device has no OS adapter LUID.
fn luid(driver: &Driver, device: c_int) -> Result<Option<[u8; 8]>, TargetError> {
    let Some(device_luid) = driver.device_luid else {
        return Ok(None);
    };
    let mut luid = [0 as std::ffi::c_char; 8];
    let mut node_mask: std::ffi::c_uint = 0;
    match driver.check(
        unsafe { device_luid(luid.as_mut_ptr(), &mut node_mask, device) },
        "device LUID",
    ) {
        Ok(()) => Ok(Some(luid.map(|byte| byte as u8))),
        Err(error) if error.code == NOT_SUPPORTED => Ok(None),
        Err(error) => Err(unavailable(error)),
    }
}

/// Gathers the complete profile parts of one device ordinal.
pub(crate) fn discover(ordinal: u32) -> Result<DiscoveredTarget<Cuda>, TargetError> {
    let driver = Driver::load().map_err(TargetError::DeviceUnavailable)?;
    let ordinal_c = c_int::try_from(ordinal).map_err(|_| {
        TargetError::DeviceUnavailable(format!("ordinal {ordinal} exceeds the driver ABI"))
    })?;
    let mut handle = 0;
    unsafe {
        driver
            .check((driver.device_get)(&mut handle, ordinal_c), "device selection")
            .map_err(unavailable)?;
    }
    let unsigned = |key: i32| -> Result<u32, TargetError> {
        let value = driver.attribute(key, handle).map_err(unavailable)?;
        u32::try_from(value).map_err(|_| {
            TargetError::UnsupportedDevice(format!("device attribute {key} reports {value}"))
        })
    };
    let mut name = [0 as std::ffi::c_char; 256];
    let mut version = 0;
    let mut memory = 0usize;
    unsafe {
        driver
            .check(
                (driver.device_name)(name.as_mut_ptr(), name.len() as c_int, handle),
                "device name",
            )
            .map_err(unavailable)?;
        driver
            .check((driver.driver_version)(&mut version), "driver version")
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_total_memory)(&mut memory, handle),
                "total device memory",
            )
            .map_err(unavailable)?;
    }
    let hardware = decode_device_name(&mut name);
    let compute_capability = read_compute_capability(&driver, handle)?;
    let driver_api =
        DriverApiVersion(u32::try_from(version).map_err(|_| {
            TargetError::UnsupportedDriver(format!("driver API version {version}"))
        })?);
    admit_device(&hardware, compute_capability, driver_api)?;
    let blackwell_100_family = in_blackwell_100_family(compute_capability, driver_api);
    let ptx = if blackwell_100_family {
        PtxTarget::BLACKWELL_100_FAMILY
    } else {
        PtxTarget::BASELINE
    };
    let tensor_memory = if blackwell_100_family {
        // PTX ISA 8.8, Tensor Memory §§9.7.18.1–9.7.18.1.2:
        // sm_100f has 512 columns by 128 32-bit lanes per CTA and
        // non-exclusive allocations are aligned in 32-column units.
        TensorMemory::Tcgen05 {
            columns: 512,
            lanes: 128,
            cell_bits: 32,
            allocation_granularity_columns: 32,
        }
    } else {
        TensorMemory::Unavailable
    };
    let properties = read_properties(&driver, handle)?;
    let kernel_parameter_bytes = kernel_parameter_bytes(driver_api);
    let global_memory_bytes = memory as u64;
    let specs = CudaDeviceInfo {
        device_ordinal: ordinal,
        compute_capability,
        driver_api,
        ptx,
        tensor_memory,
        properties,
        codegen_registers_per_thread: CODEGEN_REGISTERS_PER_THREAD,
        global_memory_bytes,
        kernel_parameter_bytes,
    };
    let limits = TargetLimits {
        max_workgroup_size: [
            u64::from(unsigned(attr::MAX_BLOCK_DIM_X)?),
            u64::from(unsigned(attr::MAX_BLOCK_DIM_Y)?),
            u64::from(unsigned(attr::MAX_BLOCK_DIM_Z)?),
        ],
        max_workgroup_threads: properties.max_group_threads.get(),
        max_grid: [
            u64::from(unsigned(attr::MAX_GRID_DIM_X)?),
            u64::from(unsigned(attr::MAX_GRID_DIM_Y)?),
            u64::from(unsigned(attr::MAX_GRID_DIM_Z)?),
        ],
        max_workgroup_bytes: properties.shared_bytes_per_group.get(),
        participant_local_bytes: Some(LOCAL_BYTES_PER_THREAD),
        // Every binding is one 8-byte pointer word plus its extents and
        // strides (`ptx::ParameterAbi`); the argument byte limit is the
        // exact bound, this count is its pointer-only projection.
        max_bindings: u32::try_from(kernel_parameter_bytes / 8).map_err(|_| {
            TargetError::UnsupportedDriver(format!(
                "kernel parameter space {kernel_parameter_bytes}"
            ))
        })?,
        max_argument_bytes: kernel_parameter_bytes,
        max_allocation_bytes: max_allocation_bytes(global_memory_bytes),
        max_allocation_alignment: ALLOCATION_ALIGNMENT,
        // Kernel index registers are 64-bit (`.address_size 64`).
        max_index_bits: 64,
        subgroup_width: Some(u32::try_from(properties.subgroup_width.get()).expect("a warp's width fits u32")),
    };
    let dtypes = DataTypeSupport {
        scalars: [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::U32,
            DType::U64,
            DType::Bool,
        ]
        .into_iter()
        .collect(),
        // Exact native/CAS implementations in the PTX emitter.
        atomics: [DType::F32, DType::I32, DType::U32].into_iter().collect(),
        // Row layouts are native-only (`ROW_LAYOUT_IS_NATIVE_ONLY`).
        representations: registry::representations()
            .iter()
            .filter(|info| info.layout == registry::Layout::Packet)
            .map(|info| info.id)
            .collect(),
    };
    let numerics = NumericalEnvironment {
        contraction_available: true,
        flush_to_zero_available: true,
        approximate_transcendentals: [
            MathOp::Exp,
            MathOp::Log,
            MathOp::Sin,
            MathOp::Cos,
            MathOp::Sqrt,
            MathOp::Rsqrt,
        ]
        .into_iter()
        .map(MathOp::name)
        .collect(),
        denormals_preserved: true,
    };
    let identity = CompatibilityIdentity {
        backend: BackendName::Cuda,
        backend_revision: BACKEND_REVISION,
        hardware,
        driver: driver_api.to_string(),
        toolchain: format!(
            "driver-jit ptx{}.{} sm_{}{}",
            specs.ptx.isa_major,
            specs.ptx.isa_minor,
            specs.ptx.architecture,
            match specs.ptx.feature_set {
                PtxFeatureSet::Baseline => "",
                PtxFeatureSet::Family => "f",
            }
        ),
        fingerprint: fingerprint(&specs, &limits, &dtypes, &numerics),
    };
    Ok(DiscoveredTarget {
        identity,
        limits,
        dtypes,
        vectors: VectorSupport::default(),
        numerics,
        device_info: specs,
        kernel_abi: CudaKernelAbi,
        local_realization: LocalRealizationPolicy {
            workgroup: LocalRealization::NativeDynamic,
            participant: LocalRealization::InvocationScratchPerParticipant,
            register: LocalRealization::InvocationScratchPerParticipant,
        },
    })
}

/// Acquires and assembles the complete profile on the exact context and
/// production stream retained by the opened device service, its probe
/// kernels formed through `programs`, the device's.
pub(crate) fn profile_for_opened(
    service: &crate::Device,
    device: Arc<seismic_native_target::DeviceDescription<Cuda>>,
    programs: &CudaPtxPrograms,
) -> Result<seismic_compiler::evaluation::AnalyticalEvaluationContext<Cuda>, TargetError> {
    let total_started = Instant::now();
    let (ordinal, context, stream) = (service.ordinal(), service.context(), service.stream());
    if context.ordinal()
        != c_int::try_from(ordinal).map_err(|_| {
            TargetError::DeviceUnavailable(format!("ordinal {ordinal} exceeds the driver ABI"))
        })?
        || !Arc::ptr_eq(context, stream.context())
    {
        panic!("CUDA profile acquisition received a context/stream for another opened device")
    }
    let (services, composition_qualification, probe_build_ns, probe_execution_ns) =
        acquire_services(service, device.device_info(), programs)?;
    let execution_parts = ExecutionProfileParts::new(
        device,
        "seismic-cuda-probes-v5-classified-host-confidence95",
        services,
        ProfileAcquisitionMetrics {
            identity_and_limits_ns: 0,
            probe_build_ns,
            probe_execution_ns,
            total_ns: u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        },
        composition_qualification,
    );
    seismic_compiler::evaluation::AnalyticalEvaluationContext::assemble(
        execution_parts,
        crate::model::CudaAnalyticalModel,
    )
}

pub(crate) fn device_for_opened(
    ordinal: u32,
) -> Result<Arc<seismic_native_target::DeviceDescription<Cuda>>, TargetError> {
    let parts = discover(ordinal)?;
    Ok(Arc::new(assemble_device_description(
        parts,
        capability::registry(),
    )?))
}

/// Fixed retained sample count paired with the normal 95% coefficient below.
/// Sixty-four observations make the estimated mean materially more stable than
/// raw extrema without turning acquisition into an unbounded convergence loop.
const GPU_PROBE_REPETITIONS: usize = 64;
const HOST_DETERMINISTIC_REPETITIONS: usize = 9;
const HOST_STOCHASTIC_REPETITIONS: usize = 64;
const GPU_CONFIDENCE_Z_95: f64 = 1.96;
/// NVIDIA's CUDA Driver API documents `cuEventElapsedTime` with approximately
/// 0.5 microsecond resolution. This is clock quantization, not the latency of
/// an empty kernel measured by that clock.
const CUDA_EVENT_TIMER_RESOLUTION_NS: u64 = 500;
const MAX_GPU_ACQUISITION_ROUNDS: usize = 3;
const MAX_HOST_ACQUISITION_ROUNDS: usize = 8;

fn acquire_services(
    service: &crate::Device,
    specs: &CudaDeviceInfo,
    programs: &CudaPtxPrograms,
) -> Result<
    (
        Vec<ServiceDefinition>,
        CompositionQualificationParts,
        u64,
        u64,
    ),
    TargetError,
> {
    let (context, stream) = (service.context(), service.stream());
    let build_started = Instant::now();
    let source = seismic_native_target::ProgramSource {
        text: probe_source(specs.ptx),
        entries: PROBE_ENTRIES.into_iter().map(seismic_native_target::ProgramEntry::named).collect(),
    };
    let formed = programs.form(std::slice::from_ref(&source)).map_err(|error| {
        TargetError::DeviceUnavailable(format!("CUDA probe formation: {error:?}"))
    })?;
    let probe = |entry: &str| {
        let (module, position) = formed[0].program(&seismic_native_target::ProgramEntry::named(entry));
        module.function(position)
    };
    let empty = probe("probe_empty");
    let probe_build_ns = u64::try_from(build_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let execution_started = Instant::now();
    let allocation = Allocation::new(context, 4096).map_err(unavailable)?;
    let timer_resolution = event_timer_resolution();
    let baseline = measure_kernel_sequence_difference(
        context,
        stream,
        empty,
        allocation.pointer,
        32_768,
        65_536,
        timer_resolution,
        2,
        "probe_empty",
    )?;
    let mut definitions = Vec::new();
    let mut add_kernel = |class,
                          entry: &'static str,
                          latency_block: [u32; 3],
                          capacity_block: [u32; 3],
                          operations_per_thread: u64,
                          participants_per_unit: u64,
                          topology: ResourceTopology|
     -> Result<(), TargetError> {
        let function = probe(entry);
        let latency_units = u64::from(latency_block[0])
            * u64::from(latency_block[1])
            * u64::from(latency_block[2])
            * operations_per_thread
            / participants_per_unit;
        let capacity_grid = [u32::try_from(specs.properties.units.get()).expect("a device's multiprocessors fit u32").max(1), 1, 1];
        let capacity_units = u64::from(capacity_grid[0])
            * u64::from(capacity_block[0])
            * u64::from(capacity_block[1])
            * u64::from(capacity_block[2])
            * operations_per_thread
            / participants_per_unit;
        let percent = match service_accuracy(class) {
            ServiceAccuracyClass::Compute => 1,
            ServiceAccuracyClass::MemoryOrTransfer => 2,
        };
        let mut accepted = None;
        let mut last = None;
        for _ in 0..MAX_GPU_ACQUISITION_ROUNDS {
            let latency = measure_kernel(
                context,
                stream,
                function,
                allocation.pointer,
                [1, 1, 1],
                latency_block,
                latency_units,
                None,
            )?;
            let saturated = measure_kernel(
                context,
                stream,
                function,
                allocation.pointer,
                capacity_grid,
                capacity_block,
                capacity_units,
                (entry == "probe_repack").then_some(64),
            )?;
            let dependency = widen_interval(
                normalized_confidence_interval(&latency.observations),
                timer_resolution,
            );
            let capacity = widen_interval(
                differential_wave_confidence_interval(
                    &saturated.observations,
                    &baseline.observations,
                    topology,
                ),
                timer_resolution,
            );
            last = Some((dependency, capacity));
            if interval_qualified(dependency, percent) && interval_qualified(capacity, percent) {
                accepted = Some((latency, saturated));
                break;
            }
        }
        let (latency, saturated) = accepted.ok_or_else(|| {
            let unavailable = DurationInterval::new(0, u64::MAX, 1);
            let (dependency, capacity) = last.unwrap_or((unavailable, unavailable));
            TargetError::DeviceUnavailable(format!(
                "CUDA service probe `{entry}` failed its {percent}% qualification after {MAX_GPU_ACQUISITION_ROUNDS} rounds (dependency {}/{}, {}/{}, capacity {}/{}, {}/{})",
                dependency.lower_numerator, dependency.denominator,
                dependency.upper_numerator, dependency.denominator,
                capacity.lower_numerator, capacity.denominator,
                capacity.upper_numerator, capacity.denominator,
            ))
        })?;
        definitions.push(measured_definition(
            class,
            topology,
            entry,
            timer_resolution,
            &baseline,
            &latency,
            &saturated,
        ));
        Ok(())
    };
    let sms = u32::try_from(specs.properties.units.get()).expect("a device's multiprocessors fit u32").max(1);
    let lanes = u32::try_from(specs.properties.threads_per_unit.get()).expect("a multiprocessor's threads fit u32").max(1);
    let compute_topology = ResourceTopology {
        resources: sms,
        max_concurrency: lanes,
    };
    add_kernel(
        SERVICE_INSTRUCTION,
        "probe_instruction",
        [1, 1, 1],
        [128, 1, 1],
        16_777_216,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_GLOBAL_READ,
        "probe_global_read",
        [1, 1, 1],
        [128, 1, 1],
        65_536,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_GLOBAL_WRITE,
        "probe_global_write",
        [1, 1, 1],
        [128, 1, 1],
        65_536,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_SHARED_READ,
        "probe_shared_read",
        [1, 1, 1],
        [128, 1, 1],
        1_048_576,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_SHARED_WRITE,
        "probe_shared_write",
        [1, 1, 1],
        [128, 1, 1],
        1_048_576,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_ATOMIC,
        "probe_atomic",
        [1, 1, 1],
        [128, 1, 1],
        65_536,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_BARRIER,
        "probe_barrier",
        [32, 1, 1],
        [128, 1, 1],
        65_536,
        1,
        compute_topology,
    )?;
    add_kernel(
        SERVICE_SUBGROUP,
        "probe_subgroup",
        [32, 1, 1],
        [128, 1, 1],
        268_435_456,
        1,
        compute_topology,
    )?;
    // One `mma.sync` is one collective warp operation, whereas ordinary
    // probes count one operation per participant.
    add_kernel(
        SERVICE_MATRIX,
        "probe_matrix",
        [32, 1, 1],
        [128, 1, 1],
        32_768,
        32,
        ResourceTopology {
            resources: sms,
            max_concurrency: (lanes / 32).max(1),
        },
    )?;
    add_kernel(
        SERVICE_REPACK,
        "probe_repack",
        [1, 1, 1],
        [128, 1, 1],
        1_048_576,
        1,
        compute_topology,
    )?;

    definitions.extend(acquire_core_services(
        service,
        &allocation,
        timer_resolution,
        &baseline,
    )?);
    let required: BTreeSet<_> =
        <seismic_estimator::CoreService as seismic_estimator::AnalyticalService>::ALL
            .iter()
            .map(|service| seismic_estimator::AnalyticalService::class(*service))
            .collect();
    debug_assert!(required
        .iter()
        .all(|class| definitions.iter().any(|item| item.class == *class)));
    let composition_observations = measure_kernel_sequence_difference(
        context,
        stream,
        empty,
        allocation.pointer,
        8_192,
        16_384,
        timer_resolution,
        5,
        "composition_empty",
    )?;
    let observed = widen_interval(
        confidence_interval(&composition_observations.observations),
        timer_resolution,
    );
    let mut qualification_digest = Sha256::new();
    for observation in &composition_observations.observations {
        qualification_digest.update((observation.nanoseconds.ceil() as u64).to_le_bytes());
    }
    let composition_qualification = CompositionQualificationParts {
        suite_revision: "seismic-cuda-composition-v6-event-resolution-confidence95",
        cases: vec![CompositionQualificationCase {
            stable_name: "adjacent-8192-empty-launches",
            demands: vec![seismic_estimator::ConcreteExecutionDemand {
                class: SERVICE_SUBMISSION,
                units: 8_192,
                mode: seismic_estimator::DemandMode::DependencyLatency,
            }],
            observed_ns: observed,
        }],
        observations_digest: qualification_digest.finalize().into(),
    };
    let probe_execution_ns =
        u64::try_from(execution_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok((
        definitions,
        composition_qualification,
        probe_build_ns,
        probe_execution_ns,
    ))
}

/// The entries of [`probe_source`], in its order.
const PROBE_ENTRIES: [&str; 11] = [
    "probe_empty",
    "probe_instruction",
    "probe_global_read",
    "probe_global_write",
    "probe_shared_read",
    "probe_shared_write",
    "probe_atomic",
    "probe_barrier",
    "probe_subgroup",
    "probe_matrix",
    "probe_repack",
];

fn probe_source(target: PtxTarget) -> String {
    format!(
        "{}\
.visible .entry probe_empty(.param .u64 output) {{ ret; }}\n\
.visible .entry probe_instruction(.param .u64 output) {{\n\
  .reg .u64 %p; .reg .u32 %i, %x; .reg .pred %p0;\n\
  ld.param.u64 %p, [output]; mov.u32 %i, 0; mov.u32 %x, %tid.x;\n\
L_instruction: add.u32 %x, %x, 1; add.u32 %i, %i, 1; setp.lt.u32 %p0, %i, 16777216; @%p0 bra L_instruction;\n\
  st.global.u32 [%p], %x; ret; }}\n\
.visible .entry probe_global_read(.param .u64 output) {{ .reg .u64 %p; .reg .u32 %i, %x; .reg .pred %q; ld.param.u64 %p, [output]; mov.u32 %i, 0; L_global_read: ld.volatile.global.u32 %x, [%p]; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 65536; @%q bra L_global_read; st.global.u32 [%p], %x; ret; }}\n\
.visible .entry probe_global_write(.param .u64 output) {{ .reg .u64 %p; .reg .u32 %i; .reg .pred %q; ld.param.u64 %p, [output]; mov.u32 %i, 0; L_global_write: st.volatile.global.u32 [%p], %i; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 65536; @%q bra L_global_write; ret; }}\n\
.visible .entry probe_shared_read(.param .u64 output) {{ .shared .align 4 .u32 s[128]; .reg .u32 %i, %x; .reg .pred %q; st.shared.u32 [s], 1; bar.sync 0; mov.u32 %i, 0; L_shared_read: ld.volatile.shared.u32 %x, [s]; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 1048576; @%q bra L_shared_read; ret; }}\n\
.visible .entry probe_shared_write(.param .u64 output) {{ .shared .align 4 .u32 s[128]; .reg .u32 %i; .reg .pred %q; mov.u32 %i, 0; L_shared_write: st.volatile.shared.u32 [s], %i; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 1048576; @%q bra L_shared_write; ret; }}\n\
.visible .entry probe_atomic(.param .u64 output) {{ .reg .u64 %p; .reg .u32 %i, %x; .reg .pred %q; ld.param.u64 %p, [output]; mov.u32 %i, 0; L_atomic: atom.global.add.u32 %x, [%p], 1; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 65536; @%q bra L_atomic; ret; }}\n\
.visible .entry probe_barrier(.param .u64 output) {{ .reg .u32 %i; .reg .pred %q; mov.u32 %i, 0; L_barrier: bar.sync 0; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 65536; @%q bra L_barrier; ret; }}\n\
.visible .entry probe_subgroup(.param .u64 output) {{ .reg .u64 %p; .reg .u32 %i, %x, %y; .reg .pred %q; ld.param.u64 %p, [output]; mov.u32 %x, %laneid; mov.u32 %i, 0; L_subgroup: shfl.sync.idx.b32 %y, %x, 0, 31, 0xffffffff; mov.b32 %x, %y; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 268435456; @%q bra L_subgroup; st.global.u32 [%p], %x; ret; }}\n\
.visible .entry probe_matrix(.param .u64 output) {{ .reg .u64 %p; .reg .b32 %a<4>, %b<2>; .reg .f32 %d<4>; .reg .u32 %i; .reg .pred %q; ld.param.u64 %p, [output]; mov.b32 %a0, 0; mov.b32 %a1, 0; mov.b32 %a2, 0; mov.b32 %a3, 0; mov.b32 %b0, 0; mov.b32 %b1, 0; mov.f32 %d0, 0f00000000; mov.f32 %d1, 0f00000000; mov.f32 %d2, 0f00000000; mov.f32 %d3, 0f00000000; mov.u32 %i, 0; L_matrix: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%d0,%d1,%d2,%d3}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{%d0,%d1,%d2,%d3}}; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 32768; @%q bra L_matrix; st.global.f32 [%p], %d0; ret; }}\n\
.visible .entry probe_repack(.param .u64 output) {{ .reg .u64 %p, %o64; .reg .u32 %i, %x, %y, %bid, %lane, %off; .reg .pred %q; ld.param.u64 %p, [output]; mov.u32 %bid, %ctaid.x; mov.u32 %lane, %tid.x; mad.lo.u32 %off, %bid, 128, %lane; cvt.u64.u32 %o64, %off; add.u64 %p, %p, %o64; ld.global.u8 %x, [%p]; mov.u32 %i, 0; L_repack: shl.b32 %y, %x, 4; or.b32 %y, %y, %x; mov.b32 %x, %y; add.u32 %i, %i, 1; setp.lt.u32 %q, %i, 1048576; @%q bra L_repack; st.global.u8 [%p], %x; ret; }}\n",
        target.header()
    )
}

#[derive(Clone, Copy)]
struct ProbeObservation {
    nanoseconds: f64,
    units: u64,
}

struct AcquiredBatch {
    observations: Vec<ProbeObservation>,
    acquisition_duration_ns: u64,
    /// How its observations were timed, which decides how they bound an
    /// interval.
    timing: BatchTiming,
}

/// How a batch's observations were timed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchTiming {
    /// By CUDA events on the device's stream: a confidence interval.
    Device,
    /// By the host's monotonic clock, its observations bounded by `spread`.
    Host(HostSpread),
}

fn interval_qualified(interval: DurationInterval, percent: u128) -> bool {
    if interval.upper_numerator == 0 {
        return true;
    }
    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
    let midpoint_twice =
        u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
    width * 100 <= midpoint_twice * percent
}

fn widen_interval(interval: DurationInterval, timer: DurationInterval) -> DurationInterval {
    let uncertainty = timer.upper_numerator.saturating_mul(2);
    DurationInterval::new(
        interval.lower_numerator.saturating_sub(uncertainty),
        interval.upper_numerator.saturating_add(uncertainty),
        interval.denominator,
    )
}

fn acquisition_failure(
    probe: &str,
    percent: u128,
    interval: DurationInterval,
    rounds: usize,
) -> TargetError {
    TargetError::DeviceUnavailable(format!(
        "CUDA service probe `{probe}` interval {}/{}..{}/{} ns exceeded its {percent}% relative half-width qualification after {rounds} fixed acquisition rounds",
        interval.lower_numerator,
        interval.denominator,
        interval.upper_numerator,
        interval.denominator,
    ))
}

fn event_timer_resolution() -> DurationInterval {
    DurationInterval::new(
        CUDA_EVENT_TIMER_RESOLUTION_NS,
        CUDA_EVENT_TIMER_RESOLUTION_NS,
        1,
    )
}

fn measure_kernel(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    function: crate::driver::Handle,
    output: u64,
    grid: [u32; 3],
    block: [u32; 3],
    units: u64,
    warmup_override: Option<usize>,
) -> Result<AcquiredBatch, TargetError> {
    // Bring the isolated stream and GPU clocks to the same steady regime as
    // the retained batch. A handful of microsecond launches is insufficient
    // on power-managed devices such as GB10.
    let warmups = warmup_override.unwrap_or_else(|| {
        if units < 8_192 {
            8_192
        } else if grid == [1, 1, 1] {
            1_024
        } else {
            8
        }
    });
    let acquisition_started = Instant::now();
    let _current = context.enter().map_err(unavailable)?;
    let mut output = output;
    let mut params = [(&mut output as *mut u64).cast::<std::ffi::c_void>()];
    for _ in 0..warmups {
        enqueue_kernel(context, stream, function, grid, block, &mut params)?;
    }
    stream.synchronize().map_err(unavailable)?;
    let mut observations = Vec::with_capacity(GPU_PROBE_REPETITIONS);
    let mut events = Vec::with_capacity(GPU_PROBE_REPETITIONS);
    for _ in 0..GPU_PROBE_REPETITIONS {
        let start = Event::new(context).map_err(unavailable)?;
        let end = Event::new(context).map_err(unavailable)?;
        start.record(stream).map_err(unavailable)?;
        enqueue_kernel(context, stream, function, grid, block, &mut params)?;
        end.record(stream).map_err(unavailable)?;
        events.push((start, end));
    }
    stream.synchronize().map_err(unavailable)?;
    for (start, end) in events {
        observations.push(ProbeObservation {
            nanoseconds: Event::elapsed_ns(&start, &end).map_err(unavailable)?,
            units,
        });
    }
    drop(_current);
    Ok(AcquiredBatch {
        observations,
        acquisition_duration_ns: u64::try_from(acquisition_started.elapsed().as_nanos())
            .unwrap_or(u64::MAX),
        timing: BatchTiming::Device,
    })
}

fn enqueue_kernel(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    function: crate::driver::Handle,
    grid: [u32; 3],
    block: [u32; 3],
    params: &mut [*mut std::ffi::c_void; 1],
) -> Result<(), TargetError> {
    let status = unsafe {
        context.enqueue().launch(false, function, grid, block, 0, stream.raw(), params.as_mut_ptr(), std::ptr::null_mut())
    };
    context.driver.check(status, "profile probe launch").map_err(unavailable)?;
    Ok(())
}

fn measure_kernel_sequence_difference(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    function: crate::driver::Handle,
    output: u64,
    small: u64,
    large: u64,
    timer_resolution: DurationInterval,
    percent: u128,
    probe: &str,
) -> Result<AcquiredBatch, TargetError> {
    let units = large
        .checked_sub(small)
        .filter(|units| *units > 0)
        .ok_or_else(|| {
            TargetError::DeviceUnavailable(
                "CUDA adjacent-launch probe requires ordered positive work counts".into(),
            )
        })?;
    let mut last = None;
    for _ in 0..MAX_GPU_ACQUISITION_ROUNDS {
        let batch = measure_kernel_sequence_difference_round(
            context, stream, function, output, small, large, units,
        )?;
        let interval = widen_interval(
            normalized_confidence_interval(&batch.observations),
            timer_resolution,
        );
        if interval_qualified(interval, percent) {
            return Ok(batch);
        }
        last = Some(interval);
    }
    Err(acquisition_failure(
        probe,
        percent,
        last.unwrap_or_else(|| DurationInterval::new(0, u64::MAX, 1)),
        MAX_GPU_ACQUISITION_ROUNDS,
    ))
}

fn measure_kernel_sequence_difference_round(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    function: crate::driver::Handle,
    output: u64,
    small: u64,
    large: u64,
    units: u64,
) -> Result<AcquiredBatch, TargetError> {
    let acquisition_started = Instant::now();
    let mut output = output;
    let mut params = [(&mut output as *mut u64).cast::<std::ffi::c_void>()];
    let _current = context.enter().map_err(unavailable)?;
    for _ in 0..large {
        enqueue_kernel(context, stream, function, [1, 1, 1], [1, 1, 1], &mut params)?;
    }
    stream.synchronize().map_err(unavailable)?;
    let mut events = Vec::with_capacity(GPU_PROBE_REPETITIONS);
    for _ in 0..GPU_PROBE_REPETITIONS {
        let small_start = Event::new(context).map_err(unavailable)?;
        let small_end = Event::new(context).map_err(unavailable)?;
        let large_start = Event::new(context).map_err(unavailable)?;
        let large_end = Event::new(context).map_err(unavailable)?;
        small_start.record(stream).map_err(unavailable)?;
        for _ in 0..small {
            enqueue_kernel(context, stream, function, [1, 1, 1], [1, 1, 1], &mut params)?;
        }
        small_end.record(stream).map_err(unavailable)?;
        large_start.record(stream).map_err(unavailable)?;
        for _ in 0..large {
            enqueue_kernel(context, stream, function, [1, 1, 1], [1, 1, 1], &mut params)?;
        }
        large_end.record(stream).map_err(unavailable)?;
        events.push((small_start, small_end, large_start, large_end));
    }
    stream.synchronize().map_err(unavailable)?;
    let mut observations = Vec::with_capacity(GPU_PROBE_REPETITIONS);
    for (small_start, small_end, large_start, large_end) in events {
        let small_ns = Event::elapsed_ns(&small_start, &small_end).map_err(unavailable)?;
        let large_ns = Event::elapsed_ns(&large_start, &large_end).map_err(unavailable)?;
        if large_ns <= small_ns {
            return Err(TargetError::DeviceUnavailable(
                "CUDA adjacent-launch timestamps were not monotone".into(),
            ));
        }
        observations.push(ProbeObservation {
            nanoseconds: large_ns - small_ns,
            units,
        });
    }
    drop(_current);
    Ok(AcquiredBatch {
        observations,
        acquisition_duration_ns: u64::try_from(acquisition_started.elapsed().as_nanos())
            .unwrap_or(u64::MAX),
        timing: BatchTiming::Device,
    })
}

fn measured_definition(
    class: seismic_estimator::ServiceClassId,
    topology: ResourceTopology,
    probe: &'static str,
    timer_resolution_ns: DurationInterval,
    baseline: &AcquiredBatch,
    latency: &AcquiredBatch,
    saturated: &AcquiredBatch,
) -> ServiceDefinition {
    let extrema = latency.timing == BatchTiming::Host(HostSpread::Extrema);
    let latency_interval = widen_interval(
        if extrema {
            normalized_observation_interval(&latency.observations)
        } else {
            normalized_confidence_interval(&latency.observations)
        },
        timer_resolution_ns,
    );
    let capacity_interval = if class == SERVICE_SUBMISSION {
        widen_interval(
            normalized_confidence_interval(&saturated.observations),
            timer_resolution_ns,
        )
    } else if extrema {
        widen_interval(
            differential_wave_interval(&saturated.observations, &baseline.observations, topology),
            timer_resolution_ns,
        )
    } else {
        widen_interval(
            differential_wave_confidence_interval(
                &saturated.observations,
                &baseline.observations,
                topology,
            ),
            timer_resolution_ns,
        )
    };
    let method = match latency.timing {
        BatchTiming::Host(HostSpread::Confidence) => "calibrated-whole-batch-v6/host-monotonic-confidence95-differential",
        BatchTiming::Host(HostSpread::Extrema) => "calibrated-whole-batch-v6/host-monotonic-extrema-differential",
        BatchTiming::Device => "bounded-whole-batch-v5/cuda-event-500ns-confidence95-differential",
    };
    let saturated_units = saturated
        .observations
        .first()
        .map(|item| item.units)
        .unwrap_or(1)
        .max(1);
    let accuracy = service_accuracy(class);
    ServiceDefinition {
        class,
        correlation: ServiceCorrelationId::new(class.stable_name()),
        qualification: ServiceQualificationDomain {
            minimum_units: 1,
            maximum_units: saturated_units.max(2),
            maximum_concurrent_uses: u64::from(topology.resources)
                .checked_mul(u64::from(topology.max_concurrency))
                .expect("validated CUDA service topology exceeds u64"),
        },
        accuracy,
        topology,
        dependency_latency: latency_interval,
        saturated_capacity: ServiceCurve {
            // Kernel launches already carry the separately measured
            // `core.submission` demand. Keeping the empty-launch interval here
            // would both double-count submission and subject a compute service
            // to the wrong acquisition budget.
            setup: DurationInterval::new(0, 0, 1),
            regimes: vec![ServiceCurveRegime {
                max_units: None,
                per_unit: capacity_interval,
            }],
        },
        provenance: FactProvenance::Measured {
            batches: vec![
                measurement_batch(
                    probe,
                    method,
                    baseline
                        .observations
                        .first()
                        .map(|item| item.units)
                        .unwrap_or(1)
                        .max(1),
                    timer_resolution_ns,
                    &baseline.observations,
                    baseline.acquisition_duration_ns,
                ),
                measurement_batch(
                    probe,
                    method,
                    latency
                        .observations
                        .first()
                        .map(|item| item.units)
                        .unwrap_or(1)
                        .max(1),
                    timer_resolution_ns,
                    &latency.observations,
                    latency.acquisition_duration_ns,
                ),
                measurement_batch(
                    probe,
                    method,
                    saturated_units,
                    timer_resolution_ns,
                    &saturated.observations,
                    saturated.acquisition_duration_ns,
                ),
            ]
            .into_boxed_slice(),
        },
    }
}

fn service_accuracy(class: seismic_estimator::ServiceClassId) -> ServiceAccuracyClass {
    if matches!(
        class,
        SERVICE_GLOBAL_READ
            | SERVICE_GLOBAL_WRITE
            | SERVICE_SHARED_READ
            | SERVICE_SHARED_WRITE
            | SERVICE_ATOMIC
            | SERVICE_REPACK
            | SERVICE_COPY
            | SERVICE_FILL
            | SERVICE_SCALAR_READ
            | SERVICE_SUBMISSION
            | SERVICE_ALLOCATION_INSTANCE
            | SERVICE_TENSOR_PUBLICATION
    ) {
        ServiceAccuracyClass::MemoryOrTransfer
    } else {
        ServiceAccuracyClass::Compute
    }
}

fn measurement_batch(
    probe: &'static str,
    method: &'static str,
    workload_units: u64,
    timer_resolution_ns: DurationInterval,
    observations: &[ProbeObservation],
    acquisition_duration_ns: u64,
) -> MeasurementBatch {
    let observations_ns = observations
        .iter()
        .map(|item| item.nanoseconds.ceil() as u64)
        .collect::<Vec<_>>();
    let mut digest = Sha256::new();
    for observation in &observations_ns {
        digest.update(observation.to_le_bytes());
    }
    MeasurementBatch {
        probe,
        method,
        timer_resolution_ns,
        series: MeasurementSeries::single(workload_units, observations_ns),
        observations_digest: digest.finalize().into(),
        acquisition_duration_ns,
    }
}

fn observation_interval(observations: &[ProbeObservation]) -> DurationInterval {
    let lower = observations
        .iter()
        .map(|observation| observation.nanoseconds)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let upper = observations
        .iter()
        .map(|observation| observation.nanoseconds)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    DurationInterval::new(
        lower.floor().max(0.0) as u64,
        upper.ceil().max(0.0) as u64,
        1,
    )
}

fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| {
            let residual = value - mean;
            residual * residual
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, (variance / values.len() as f64).sqrt())
}

fn confidence_interval(observations: &[ProbeObservation]) -> DurationInterval {
    let values = observations
        .iter()
        .map(|observation| observation.nanoseconds)
        .collect::<Vec<_>>();
    let (mean, standard_error) = mean_and_standard_error(&values);
    let radius = GPU_CONFIDENCE_Z_95 * standard_error;
    DurationInterval::new(
        (mean - radius).max(0.0).floor() as u64,
        (mean + radius).max(0.0).ceil() as u64,
        1,
    )
}

fn normalized_confidence_interval(observations: &[ProbeObservation]) -> DurationInterval {
    let units = observations
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        observations
            .iter()
            .all(|observation| observation.units == units),
        "one fixed CUDA confidence batch changed its demand units"
    );
    let total = confidence_interval(observations);
    DurationInterval::new(total.lower_numerator, total.upper_numerator, units)
}

fn normalized_observation_interval(observations: &[ProbeObservation]) -> DurationInterval {
    let units = observations
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        observations
            .iter()
            .all(|observation| observation.units == units),
        "one fixed CUDA probe batch changed its demand units"
    );
    let total = observation_interval(observations);
    DurationInterval::new(total.lower_numerator, total.upper_numerator, units)
}

fn differential_interval(
    observations: &[ProbeObservation],
    baseline: &[ProbeObservation],
) -> DurationInterval {
    let units = observations
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        observations
            .iter()
            .all(|observation| observation.units == units),
        "one fixed CUDA differential probe batch changed its demand units"
    );
    let baseline_units = baseline
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        baseline
            .iter()
            .all(|observation| observation.units == baseline_units),
        "one fixed CUDA baseline batch changed its demand units"
    );
    let baseline_min = baseline
        .iter()
        .map(|observation| observation.nanoseconds / baseline_units as f64)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let baseline_max = baseline
        .iter()
        .map(|observation| observation.nanoseconds / baseline_units as f64)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let lower = observations
        .iter()
        .map(|observation| (observation.nanoseconds - baseline_max).max(0.0))
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let upper = observations
        .iter()
        .map(|observation| (observation.nanoseconds - baseline_min).max(0.0))
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    DurationInterval::new(lower.floor() as u64, upper.ceil() as u64, units)
}

fn differential_wave_interval(
    observations: &[ProbeObservation],
    baseline: &[ProbeObservation],
    topology: ResourceTopology,
) -> DurationInterval {
    let units = observations
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        observations
            .iter()
            .all(|observation| observation.units == units),
        "one fixed CUDA saturated probe batch changed its demand units"
    );
    let concurrency = u64::from(topology.resources)
        .checked_mul(u64::from(topology.max_concurrency))
        .expect("validated CUDA service topology exceeds u64");
    assert!(
        concurrency != 0,
        "CUDA service topology has zero concurrency"
    );
    let waves = units.div_ceil(concurrency).max(1);
    let residual = differential_interval(observations, baseline);
    DurationInterval::new(residual.lower_numerator, residual.upper_numerator, waves)
}

fn differential_wave_confidence_interval(
    observations: &[ProbeObservation],
    baseline: &[ProbeObservation],
    topology: ResourceTopology,
) -> DurationInterval {
    let units = observations
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        observations
            .iter()
            .all(|observation| observation.units == units),
        "one fixed CUDA saturated confidence batch changed its demand units"
    );
    let baseline_units = baseline
        .first()
        .map(|observation| observation.units)
        .unwrap_or(1)
        .max(1);
    assert!(
        baseline
            .iter()
            .all(|observation| observation.units == baseline_units),
        "one fixed CUDA baseline confidence batch changed its demand units"
    );
    let observed_values = observations
        .iter()
        .map(|observation| observation.nanoseconds)
        .collect::<Vec<_>>();
    let baseline_values = baseline
        .iter()
        .map(|observation| observation.nanoseconds / baseline_units as f64)
        .collect::<Vec<_>>();
    let (observed_mean, observed_error) = mean_and_standard_error(&observed_values);
    let (baseline_mean, baseline_error) = mean_and_standard_error(&baseline_values);
    let mean = (observed_mean - baseline_mean).max(0.0);
    let radius = GPU_CONFIDENCE_Z_95
        * (observed_error * observed_error + baseline_error * baseline_error).sqrt();
    let concurrency = u64::from(topology.resources)
        .checked_mul(u64::from(topology.max_concurrency))
        .expect("validated CUDA service topology exceeds u64");
    assert!(
        concurrency != 0,
        "CUDA service topology has zero concurrency"
    );
    let waves = units.div_ceil(concurrency).max(1);
    DurationInterval::new(
        (mean - radius).max(0.0).floor() as u64,
        (mean + radius).max(0.0).ceil() as u64,
        waves,
    )
}

/// Every core service's definition, each measured on what this backend does
/// for it.
fn acquire_core_services(
    service: &crate::Device,
    allocation: &Allocation,
    event_resolution: DurationInterval,
    launch_baseline: &AcquiredBatch,
) -> Result<Vec<ServiceDefinition>, TargetError> {
    let (context, stream) = (service.context(), service.stream());
    let topology = ResourceTopology {
        resources: 1,
        max_concurrency: 1,
    };
    let bytes = 64 * 1024 * 1024;
    let transfer_source = Allocation::new(context, bytes).map_err(unavailable)?;
    let transfer_destination = Allocation::new(context, bytes).map_err(unavailable)?;
    let host_resolution = host_timer_resolution()?;
    let host_baseline = measure_host(
        host_resolution,
        1,
        HostSpread::Extrema,
        "probe_host_baseline",
        || {
            std::hint::black_box(());
            Ok(())
        },
    )?;
    let host = |class, probe, observations: AcquiredBatch| {
        measured_definition(
            class,
            topology,
            probe,
            host_resolution,
            &host_baseline,
            &observations,
            &observations,
        )
    };
    <CoreService as AnalyticalService>::ALL
        .iter()
        .map(|core| -> Result<ServiceDefinition, TargetError> {
            let class = core.class();
            Ok(match core {
                CoreService::Submission => measured_definition(
                    class,
                    topology,
                    "probe_empty",
                    event_resolution,
                    launch_baseline,
                    launch_baseline,
                    launch_baseline,
                ),
                CoreService::Copy => {
                    let copy = measure_stream_operation(
                        context,
                        stream,
                        bytes as u64,
                        event_resolution,
                        2,
                        "probe_device_copy",
                        || unsafe {
                            context.driver.check(
                                context.enqueue().memcpy_device_async(
                                    transfer_destination.pointer,
                                    transfer_source.pointer,
                                    bytes,
                                    stream.raw(),
                                ),
                                "profile device copy",
                            )
                        },
                    )?;
                    measured_definition(
                        class,
                        topology,
                        "probe_device_copy",
                        event_resolution,
                        launch_baseline,
                        &copy,
                        &copy,
                    )
                }
                CoreService::Fill => {
                    let fill = measure_stream_operation(
                        context,
                        stream,
                        bytes as u64,
                        event_resolution,
                        2,
                        "probe_device_fill",
                        || unsafe {
                            context.driver.check(
                                context.enqueue().memset_d8_async(
                                    transfer_source.pointer,
                                    0,
                                    bytes,
                                    stream.raw(),
                                ),
                                "profile device fill",
                            )
                        },
                    )?;
                    measured_definition(
                        class,
                        topology,
                        "probe_device_fill",
                        event_resolution,
                        launch_baseline,
                        &fill,
                        &fill,
                    )
                }
                CoreService::ScalarRead => host(
                    class,
                    "probe_scalar_read",
                    measure_host(
                        host_resolution,
                        2,
                        HostSpread::Confidence,
                        "probe_scalar_read",
                        || {
                            let mut bytes = [0u8; 4];
                            allocation.download_at(0, &mut bytes).map_err(unavailable)
                        },
                    )?,
                ),
                CoreService::ScalarMove => host(
                    class,
                    "probe_scalar_move",
                    measure_host(
                        host_resolution,
                        1,
                        HostSpread::Extrema,
                        "probe_scalar_move",
                        || {
                            let source = std::hint::black_box(0x1234_5678_u64);
                            std::hint::black_box(source);
                            Ok(())
                        },
                    )?,
                ),
                // A reached instance's backing, as the executor acquires one:
                // the device's allocation of a block range, charged to the
                // device store's own entry, released after.
                CoreService::AllocationInstance => {
                    let entry = service.stores().device.own();
                    host(
                        class,
                        "probe_allocation_instance",
                        measure_host(
                            host_resolution,
                            2,
                            HostSpread::Confidence,
                            "probe_allocation_instance",
                            || {
                                let instance =
                                    <crate::Device as seismic_compiler::executable::DeviceService<Cuda>>::allocate(
                                        service,
                                        crate::topology::DEVICE_KIND,
                                        seismic_estimator::ALLOCATION_INSTANCE_BYTES,
                                        256,
                                        &entry,
                                    )
                                    .map_err(|error| TargetError::DeviceUnavailable(error.to_string()))?;
                                drop(std::hint::black_box(instance));
                                Ok(())
                            },
                        )?,
                    )
                }
                // A publication as the schedule's execution records it.
                CoreService::TensorPublication => {
                    let mut probe = seismic_compiler::executable::TensorPublicationProbe::new();
                    host(
                        class,
                        "probe_tensor_publication",
                        measure_host(
                            host_resolution,
                            2,
                            HostSpread::Confidence,
                            "probe_tensor_publication",
                            || {
                                probe.publish();
                                Ok(())
                            },
                        )?,
                    )
                }
                CoreService::DataCheck => host(
                    class,
                    "probe_data_check",
                    measure_host(
                        host_resolution,
                        1,
                        HostSpread::Extrema,
                        "probe_data_check",
                        || {
                            let left = std::hint::black_box(0x1234_5678_u64);
                            let right = std::hint::black_box(0x1234_5678_u64);
                            std::hint::black_box(left == right);
                            Ok(())
                        },
                    )?,
                ),
            })
        })
        .collect()
}

fn measure_stream_operation(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    units: u64,
    timer_resolution: DurationInterval,
    percent: u128,
    probe: &str,
    enqueue: impl Fn() -> Result<(), DriverError>,
) -> Result<AcquiredBatch, TargetError> {
    let mut last = None;
    for _ in 0..MAX_GPU_ACQUISITION_ROUNDS {
        let acquisition_started = Instant::now();
        for _ in 0..GPU_PROBE_REPETITIONS {
            enqueue().map_err(unavailable)?;
        }
        stream.synchronize().map_err(unavailable)?;
        let mut events = Vec::with_capacity(GPU_PROBE_REPETITIONS);
        for _ in 0..GPU_PROBE_REPETITIONS {
            let start = Event::new(context).map_err(unavailable)?;
            let end = Event::new(context).map_err(unavailable)?;
            start.record(stream).map_err(unavailable)?;
            enqueue().map_err(unavailable)?;
            end.record(stream).map_err(unavailable)?;
            events.push((start, end));
        }
        stream.synchronize().map_err(unavailable)?;
        let mut observations = Vec::with_capacity(GPU_PROBE_REPETITIONS);
        for (start, end) in events {
            observations.push(ProbeObservation {
                nanoseconds: Event::elapsed_ns(&start, &end).map_err(unavailable)?,
                units,
            });
        }
        let interval = widen_interval(
            normalized_confidence_interval(&observations),
            timer_resolution,
        );
        if interval_qualified(interval, percent) {
            return Ok(AcquiredBatch {
                observations,
                acquisition_duration_ns: u64::try_from(acquisition_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX),
                timing: BatchTiming::Device,
            });
        }
        last = Some(interval);
    }
    Err(acquisition_failure(
        probe,
        percent,
        last.unwrap_or_else(|| DurationInterval::new(0, u64::MAX, 1)),
        MAX_GPU_ACQUISITION_ROUNDS,
    ))
}

/// How a host probe's observations bound its interval.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostSpread {
    /// A deterministic operation: the extrema of a few observations.
    Extrema,
    /// A stochastic operation: the 95% confidence interval of the mean.
    Confidence,
}

impl HostSpread {
    fn repetitions(self) -> usize {
        match self {
            Self::Extrema => HOST_DETERMINISTIC_REPETITIONS,
            Self::Confidence => HOST_STOCHASTIC_REPETITIONS,
        }
    }

    fn interval(self, observations: &[ProbeObservation]) -> DurationInterval {
        match self {
            Self::Extrema => normalized_observation_interval(observations),
            Self::Confidence => normalized_confidence_interval(observations),
        }
    }
}

/// Measures a host operation in observations of a unit count calibrated to
/// the duration the band needs at the host timer's resolution; a round that
/// does not qualify grows the count for the next by how far it missed,
/// within the round budget (`seismic_estimator::escalate_observation_units`).
fn measure_host(
    timer_resolution: DurationInterval,
    percent: u64,
    spread: HostSpread,
    probe: &str,
    mut operation: impl FnMut() -> Result<(), TargetError>,
) -> Result<AcquiredBatch, TargetError> {
    let mut observe = |units: u64| -> Result<u64, TargetError> {
        let start = Instant::now();
        for _ in 0..units {
            operation()?;
        }
        Ok(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
    };
    let mut units = seismic_estimator::calibrate_observation_units(
        seismic_estimator::required_observation_ns(timer_resolution.upper_numerator, percent),
        &mut observe,
    )?;
    let mut last = None;
    let mut rounds = 0;
    while rounds < MAX_HOST_ACQUISITION_ROUNDS {
        rounds += 1;
        let acquisition_started = Instant::now();
        observe(units)?;
        let mut observations = Vec::with_capacity(spread.repetitions());
        for _ in 0..spread.repetitions() {
            observations.push(ProbeObservation {
                nanoseconds: observe(units)? as f64,
                units,
            });
        }
        let interval = widen_interval(spread.interval(&observations), timer_resolution);
        if interval_qualified(interval, u128::from(percent)) {
            return Ok(AcquiredBatch {
                observations,
                acquisition_duration_ns: u64::try_from(acquisition_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX),
                timing: BatchTiming::Host(spread),
            });
        }
        last = Some(interval);
        match seismic_estimator::escalate_observation_units(units, acquisition_started.elapsed(), Some(interval), percent) {
            Some(next) => units = next,
            None => break,
        }
    }
    Err(acquisition_failure(
        probe,
        u128::from(percent),
        last.unwrap_or_else(|| DurationInterval::new(0, u64::MAX, 1)),
        rounds,
    ))
}

fn host_timer_resolution() -> Result<DurationInterval, TargetError> {
    let mut smallest = None;
    for _ in 0..256 {
        let start = Instant::now();
        let elapsed = start.elapsed().as_nanos() as u64;
        if elapsed > 0 {
            smallest = Some(smallest.map_or(elapsed, |current: u64| current.min(elapsed)));
        }
    }
    let resolution = smallest.ok_or_else(|| {
        TargetError::DeviceUnavailable(
            "host monotonic timer did not resolve the fixed CUDA host-service probes".into(),
        )
    })?;
    Ok(DurationInterval::new(resolution, resolution, 1))
}

fn fingerprint(
    specs: &CudaDeviceInfo,
    limits: &TargetLimits,
    dtypes: &DataTypeSupport,
    numerics: &NumericalEnvironment,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    put_str(&mut digest, registry::REGISTRY_REVISION);
    put_str(&mut digest, BACKEND_REVISION);
    digest.update(specs.device_ordinal.to_le_bytes());
    digest.update(specs.compute_capability.major.to_le_bytes());
    digest.update(specs.compute_capability.minor.to_le_bytes());
    digest.update(specs.driver_api.0.to_le_bytes());
    digest.update(specs.ptx.isa_major.to_le_bytes());
    digest.update(specs.ptx.isa_minor.to_le_bytes());
    digest.update(specs.ptx.architecture.to_le_bytes());
    digest.update([match specs.ptx.feature_set {
        PtxFeatureSet::Baseline => 0,
        PtxFeatureSet::Family => 1,
    }]);
    match specs.tensor_memory {
        TensorMemory::Unavailable => digest.update([0]),
        TensorMemory::Tcgen05 {
            columns,
            lanes,
            cell_bits,
            allocation_granularity_columns,
        } => {
            digest.update([1]);
            digest.update(columns.to_le_bytes());
            digest.update(lanes.to_le_bytes());
            digest.update(cell_bits.to_le_bytes());
            digest.update(allocation_granularity_columns.to_le_bytes());
        }
    }
    digest.update(specs.properties.identity().as_bytes());
    digest.update(specs.codegen_registers_per_thread.to_le_bytes());
    for value in [specs.global_memory_bytes, specs.kernel_parameter_bytes] {
        digest.update(value.to_le_bytes());
    }
    for value in limits.max_workgroup_size {
        digest.update(value.to_le_bytes());
    }
    digest.update(limits.max_workgroup_threads.to_le_bytes());
    for value in limits.max_grid {
        digest.update(value.to_le_bytes());
    }
    digest.update(limits.max_workgroup_bytes.to_le_bytes());
    put_option_u64(&mut digest, limits.participant_local_bytes);
    digest.update(limits.max_bindings.to_le_bytes());
    digest.update(limits.max_argument_bytes.to_le_bytes());
    digest.update(limits.max_allocation_bytes.to_le_bytes());
    digest.update(limits.max_allocation_alignment.to_le_bytes());
    digest.update(limits.max_index_bits.to_le_bytes());
    put_option_u32(&mut digest, limits.subgroup_width);
    for dtype in &dtypes.scalars {
        put_str(&mut digest, dtype.name());
    }
    digest.update([0xff]);
    for dtype in &dtypes.atomics {
        put_str(&mut digest, dtype.name());
    }
    digest.update([0xfe]);
    for representation in &dtypes.representations {
        put_str(
            &mut digest,
            registry::representation_info(*representation).name,
        );
    }
    digest.update([
        u8::from(numerics.contraction_available),
        u8::from(numerics.flush_to_zero_available),
        u8::from(numerics.denormals_preserved),
    ]);
    for operation in &numerics.approximate_transcendentals {
        put_str(&mut digest, operation);
    }
    digest.finalize().into()
}

fn put_str(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}
fn put_option_u64(digest: &mut Sha256, value: Option<u64>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(value.to_le_bytes());
    }
}
fn put_option_u32(digest: &mut Sha256, value: Option<u32>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(value.to_le_bytes());
    }
}

#[allow(dead_code)]
fn _specs_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<CudaDeviceInfo>();
    let _: BTreeSet<DType> = BTreeSet::new();
}

#[cfg(test)]
mod acquisition_tests {
    use super::*;

    fn observation(nanoseconds: f64, units: u64) -> ProbeObservation {
        ProbeObservation { nanoseconds, units }
    }

    #[test]
    fn qualification_is_relative_half_width() {
        assert!(interval_qualified(DurationInterval::new(99, 101, 1), 1));
        assert!(!interval_qualified(DurationInterval::new(98, 102, 1), 1));
        assert!(interval_qualified(DurationInterval::new(98, 102, 1), 2));
        assert!(!interval_qualified(DurationInterval::new(97, 103, 1), 2));
        assert!(interval_qualified(DurationInterval::new(0, 0, 1), 1));
    }

    #[test]
    fn timer_uncertainty_widens_both_bounds() {
        assert_eq!(
            widen_interval(
                DurationInterval::new(100, 110, 7),
                DurationInterval::new(2, 3, 1),
            ),
            DurationInterval::new(94, 116, 7),
        );
        assert_eq!(
            widen_interval(
                DurationInterval::new(3, 5, 1),
                DurationInterval::new(2, 2, 1),
            ),
            DurationInterval::new(0, 9, 1),
        );
    }

    #[test]
    fn empty_kernel_duration_is_not_event_timer_resolution() {
        let resolution = event_timer_resolution();
        assert_eq!(resolution, DurationInterval::new(500, 500, 1));
        // A representative 2.4 us empty-kernel observation remains a measured
        // service duration. Widening uses ±2 event ticks (±1 us), not ±4.8 us.
        assert_eq!(
            widen_interval(DurationInterval::new(2_400, 2_400, 1), resolution),
            DurationInterval::new(1_400, 3_400, 1),
        );
    }

    #[test]
    fn gpu_confidence_interval_is_fixed_95_percent_mean_interval() {
        let observations = (0..GPU_PROBE_REPETITIONS)
            .map(|index| observation(if index % 2 == 0 { 90.0 } else { 110.0 }, 8))
            .collect::<Vec<_>>();
        assert_eq!(
            confidence_interval(&observations),
            DurationInterval::new(97, 103, 1),
        );
        assert_eq!(
            normalized_confidence_interval(&observations),
            DurationInterval::new(97, 103, 8),
        );
    }

    #[test]
    fn differential_normalizes_amplified_submission_baseline() {
        let observed = [observation(210.0, 10), observation(220.0, 10)];
        let baseline = [observation(1_000.0, 100), observation(1_200.0, 100)];
        assert_eq!(
            differential_interval(&observed, &baseline),
            DurationInterval::new(198, 210, 10),
        );
    }

    #[test]
    fn host_noop_baseline_is_normalized_by_its_own_units() {
        let observed = [observation(65_536.0, 32_768)];
        let baseline = [observation(268_435_456.0, 268_435_456)];
        assert_eq!(
            differential_interval(&observed, &baseline),
            DurationInterval::new(65_535, 65_535, 32_768),
        );
    }

    #[test]
    fn host_probe_classification_separates_stochastic_downloads_from_compute() {
        let batch = |values: Vec<f64>, units, spread| AcquiredBatch {
            observations: values
                .into_iter()
                .map(|nanoseconds| observation(nanoseconds, units))
                .collect(),
            acquisition_duration_ns: 1,
            timing: BatchTiming::Host(spread),
        };
        let baseline = batch(
            vec![268_435_456.0; HOST_DETERMINISTIC_REPETITIONS],
            268_435_456,
            HostSpread::Extrema,
        );
        let stochastic_values = (0..HOST_STOCHASTIC_REPETITIONS)
            .map(|index| if index % 2 == 0 { 90.0 } else { 110.0 })
            .collect::<Vec<_>>();
        let stochastic = batch(stochastic_values.clone(), 1, HostSpread::Confidence);
        let scalar_read = measured_definition(
            SERVICE_SCALAR_READ,
            ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            "probe_scalar_read",
            DurationInterval::new(0, 0, 1),
            &baseline,
            &stochastic,
            &stochastic,
        );
        assert_eq!(
            scalar_read.dependency_latency,
            DurationInterval::new(97, 103, 1),
        );
        let FactProvenance::Measured { batches } = scalar_read.provenance else {
            panic!("scalar read must retain measured provenance")
        };
        assert!(batches[1].method.contains("host-monotonic-confidence95"));

        let deterministic = batch(stochastic_values, 1, HostSpread::Extrema);
        let scalar_move = measured_definition(
            CoreService::ScalarMove.class(),
            ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            "probe_scalar_move",
            DurationInterval::new(0, 0, 1),
            &baseline,
            &deterministic,
            &deterministic,
        );
        assert_eq!(
            scalar_move.dependency_latency,
            DurationInterval::new(90, 110, 1),
        );
        let FactProvenance::Measured { batches } = scalar_move.provenance else {
            panic!("scalar move must retain measured provenance")
        };
        assert!(batches[1].method.contains("host-monotonic-extrema"));
    }

    #[test]
    fn differential_wave_uses_topology_wave_denominator() {
        let observed = [observation(120.0, 101), observation(130.0, 101)];
        let baseline = [observation(1_000.0, 100)];
        assert_eq!(
            differential_wave_interval(
                &observed,
                &baseline,
                ResourceTopology {
                    resources: 2,
                    max_concurrency: 5,
                },
            ),
            DurationInterval::new(110, 120, 11),
        );
    }

    #[test]
    fn differential_confidence_normalizes_baseline_and_uses_wave_denominator() {
        let observed = (0..GPU_PROBE_REPETITIONS)
            .map(|_| observation(120.0, 101))
            .collect::<Vec<_>>();
        let baseline = (0..GPU_PROBE_REPETITIONS)
            .map(|_| observation(1_000.0, 100))
            .collect::<Vec<_>>();
        assert_eq!(
            differential_wave_confidence_interval(
                &observed,
                &baseline,
                ResourceTopology {
                    resources: 2,
                    max_concurrency: 5,
                },
            ),
            DurationInterval::new(110, 110, 11),
        );
    }

    #[test]
    fn bounded_failure_is_typed_and_describes_the_last_interval() {
        let error = acquisition_failure("probe_copy", 2, DurationInterval::new(90, 110, 4), 3);
        let TargetError::DeviceUnavailable(message) = error else {
            panic!("acquisition failure must remain a typed availability failure")
        };
        assert!(message.contains("`probe_copy`"));
        assert!(message.contains("90/4..110/4"));
        assert!(message.contains("2%"));
        assert!(message.contains("3 fixed acquisition rounds"));
    }

    #[test]
    fn provenance_contains_only_supplied_accepted_batches() {
        let batch = |nanoseconds, acquisition_duration_ns| AcquiredBatch {
            observations: (0..GPU_PROBE_REPETITIONS)
                .map(|_| observation(nanoseconds, 1))
                .collect(),
            acquisition_duration_ns,
            timing: BatchTiming::Device,
        };
        let baseline = batch(10.0, 11);
        let latency = batch(100.0, 22);
        let saturated = batch(200.0, 33);
        let definition = measured_definition(
            SERVICE_SUBMISSION,
            ResourceTopology {
                resources: 1,
                max_concurrency: 1,
            },
            "probe_empty",
            DurationInterval::new(1, 1, 1),
            &baseline,
            &latency,
            &saturated,
        );
        let FactProvenance::Measured { batches } = definition.provenance else {
            panic!("measured CUDA service must retain measured provenance")
        };
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].acquisition_duration_ns, 11);
        assert_eq!(batches[1].acquisition_duration_ns, 22);
        assert_eq!(batches[2].acquisition_duration_ns, 33);
        let (_, latency) = batches[0].series.single_parts().unwrap();
        assert_eq!(latency.len(), GPU_PROBE_REPETITIONS);
        assert!(latency.iter().all(|value| *value == 10));
        let (_, capacity) = batches[1].series.single_parts().unwrap();
        assert_eq!(capacity.len(), GPU_PROBE_REPETITIONS);
        assert!(capacity.iter().all(|value| *value == 100));
        let (_, setup) = batches[2].series.single_parts().unwrap();
        assert_eq!(setup.len(), GPU_PROBE_REPETITIONS);
        assert!(setup.iter().all(|value| *value == 200));
    }
}

#[cfg(test)]
mod floor_tests {
    use super::*;

    #[test]
    fn the_driver_floor_is_the_bundled_nvrtc_major() {
        assert_eq!(DriverApiVersion::MINIMUM, DriverApiVersion(12_000));
        for rejected in [11_010, 11_080, 11_090] {
            let error = admit_driver(DriverApiVersion(rejected)).unwrap_err();
            assert!(
                matches!(&error, TargetError::UnsupportedDriver(reason) if reason.contains("12.0")),
                "{error:?}"
            );
        }
        for admitted in [12_000, 12_020, 12_090, 13_000] {
            admit_driver(DriverApiVersion(admitted)).unwrap();
        }
    }

    #[test]
    fn the_blackwell_family_is_10_0_and_10_3_at_driver_12_9() {
        let capability = |minor| ComputeCapability { major: 10, minor };
        let driver = DriverApiVersion::PTX_88_MINIMUM;
        assert!(in_blackwell_100_family(capability(0), driver));
        assert!(in_blackwell_100_family(capability(3), driver));
        assert!(!in_blackwell_100_family(capability(7), driver));
        assert!(!in_blackwell_100_family(
            capability(0),
            DriverApiVersion(12_080)
        ));
        assert!(!in_blackwell_100_family(
            ComputeCapability {
                major: 12,
                minor: 0
            },
            driver
        ));
    }

    #[test]
    fn decode_device_name_forces_nul_termination_and_handles_empty() {
        let mut normal = [0 as std::ffi::c_char; 16];
        for (slot, byte) in normal.iter_mut().zip(b"RTX 4090") {
            *slot = *byte as std::ffi::c_char;
        }
        assert_eq!(decode_device_name(&mut normal), "RTX 4090");

        // Completely filled buffer with no NUL byte must be NUL-terminated at
        // the last byte before reading.
        let mut unterminated = [b'A' as std::ffi::c_char; 8];
        assert_eq!(decode_device_name(&mut unterminated), "AAAAAAA");
        assert_eq!(unterminated[7], 0);

        let mut empty: [std::ffi::c_char; 0] = [];
        assert_eq!(decode_device_name(&mut empty), "");
    }
}
