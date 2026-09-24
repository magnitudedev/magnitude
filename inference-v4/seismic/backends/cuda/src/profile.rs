//! CUDA opened-device acquisition. Device-wide legality and numerical facts
//! form `DeviceDescription<Cuda>`; fixed service probes on the production
//! context/stream form its separately identity-bound `ExecutionProfile`.
//! Concrete function resources are reflected only during native reconciliation.
//!
//! Facts are exact device/driver reports except where a field's
//! documentation names it a backend rule.

use crate::capability;
use crate::driver::{
    compile_image, load_module, module_function, Allocation, Context, Driver, DriverError, Event,
    Stream,
};
use crate::{
    Cuda, SERVICE_ATOMIC, SERVICE_BARRIER, SERVICE_GLOBAL_READ, SERVICE_GLOBAL_WRITE,
    SERVICE_INSTRUCTION, SERVICE_MATRIX, SERVICE_REPACK, SERVICE_SHARED_READ, SERVICE_SHARED_WRITE,
    SERVICE_SUBGROUP,
};
use seismic_compiler::errors::TargetError;
use seismic_compiler::target::{
    assemble_device_description, DiscoveredTarget, ExecutionProfileParts,
};
use seismic_estimator::{
    CompositionQualificationCase, CompositionQualificationParts, DurationInterval, FactProvenance,
    MeasurementBatch, MeasurementSeries, ProfileAcquisitionMetrics, ResourceTopology,
    ServiceAccuracyClass, ServiceClassId, ServiceCorrelationId, ServiceCurve, ServiceCurveRegime,
    ServiceDefinition, ServiceQualificationDomain,
};
const SERVICE_SUBMISSION: ServiceClassId = ServiceClassId::new("core.submission");
const SERVICE_COPY: ServiceClassId = ServiceClassId::new("core.copy");
const SERVICE_FILL: ServiceClassId = ServiceClassId::new("core.fill");
const SERVICE_SCALAR_READ: ServiceClassId = ServiceClassId::new("core.scalar-read");
const SERVICE_SCALAR_MOVE: ServiceClassId = ServiceClassId::new("core.scalar-move");
const SERVICE_DATA_CHECK: ServiceClassId = ServiceClassId::new("core.data-check");
use seismic_ir::kernel::Kernel;
use seismic_ir::target::{
    DataTypeSupport, KernelAbiAllocation, KernelAbiAllocationRole, KernelAbiFootprint,
    KernelAbiLayout, KernelAbiModel, KernelWordLayout, LocalRealization, LocalRealizationPolicy,
    NumericalEnvironment, TargetLimits, VectorSupport,
};
use seismic_lang::intrinsics::MathOp;
use seismic_lang::registry::{self, BackendName};
use seismic_lang::types::DType;
use seismic_target::CompatibilityIdentity;
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

/// Documented per-thread local memory limit of every supported compute
/// capability (CUDA programming guide, compute capability 2.x and later).
const LOCAL_BYTES_PER_THREAD: u64 = 512 * 1024;

/// Documented per-thread register ceiling of every supported compute
/// capability.
const MAX_REGISTERS_PER_THREAD: u32 = 255;
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
    /// PTX 7.1 (`cvt` to/from `bf16`) requires CUDA 11.1.
    pub const PTX_71_MINIMUM: Self = Self(11_010);
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
/// dimensions and allocation unit are architectural facts of the
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
pub struct CudaFacts {
    /// Driver device ordinal; the primary context of this ordinal is what
    /// native compilation and execution retain.
    pub device_ordinal: u32,
    pub compute_capability: ComputeCapability,
    pub driver_api: DriverApiVersion,
    pub ptx: PtxTarget,
    pub tensor_memory: TensorMemory,
    /// `CU_DEVICE_ATTRIBUTE_WARP_SIZE`.
    pub warp_size: u32,
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    pub multiprocessors: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`.
    pub max_threads_per_multiprocessor: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`.
    pub max_blocks_per_multiprocessor: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`.
    pub registers_per_block: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR`.
    pub registers_per_multiprocessor: u32,
    /// Documented per-thread register ceiling. No reliable pre-JIT register
    /// pressure model exists for PTX, so this is a fact, not a constraint.
    pub max_registers_per_thread: u32,
    /// Fixed physical-register ceiling emitted on every entry.
    pub codegen_registers_per_thread: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`.
    pub shared_bytes_per_multiprocessor: u64,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`: the true
    /// per-block shared-memory ceiling (native compilation opts every
    /// kernel in).
    pub shared_bytes_per_block: u64,
    /// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`.
    pub l2_cache_bytes: u64,
    /// `cuDeviceTotalMem`.
    pub global_memory_bytes: u64,
    /// `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`.
    pub cooperative_launch: bool,
    /// The kernel parameter space, by driver generation.
    pub kernel_parameter_bytes: u64,
}

/// Device attribute keys (`CUdevice_attribute`).
mod attr {
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
    pub const MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;
    pub const MAX_REGISTERS_PER_MULTIPROCESSOR: i32 = 82;
    pub const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;
    pub const COOPERATIVE_LAUNCH: i32 = 99;
    pub const MAX_BLOCKS_PER_MULTIPROCESSOR: i32 = 106;
    pub const INTEGRATED: i32 = 18;
}

fn unavailable(error: DriverError) -> TargetError {
    TargetError::DeviceUnavailable(error.to_string())
}

/// The single-allocation limit of a CUDA device: one `cuMemAlloc` can at
/// most cover the device's total memory. This is the `TargetLimits` value and
/// the catalog's published allocation limit; it is not available memory.
pub fn max_allocation_bytes(total_memory_bytes: u64) -> u64 {
    total_memory_bytes
}

/// Cheap unopened catalog identity. Constructing this value does not retain
/// a primary context, create a stream, run probes, or assemble a profile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceDescriptor {
    pub ordinal: u32,
    /// `cuDeviceGetUuid_v2`: the exposed device or MIG partition, never
    /// merely its parent GPU.
    pub uuid: [u8; 16],
    pub name: String,
    /// `cuDeviceTotalMem`: memory of the exposed device or partition.
    pub total_memory_bytes: u64,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`: the device shares host memory.
    pub integrated: bool,
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

pub fn describe(ordinal: u32) -> Result<DeviceDescriptor, TargetError> {
    let driver = Driver::load().map_err(TargetError::DeviceUnavailable)?;
    let device = c_int::try_from(ordinal).map_err(|_| {
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
    unsafe {
        driver
            .check((driver.device_get)(&mut handle, device), "device selection")
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_name)(name.as_mut_ptr(), name.len() as c_int, device),
                "device name",
            )
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_total_memory)(&mut memory, device),
                "total device memory",
            )
            .map_err(unavailable)?;
        driver
            .check(device_uuid_v2(&mut uuid, device), "device UUID")
            .map_err(unavailable)?;
    }
    let integrated = driver
        .attribute(attr::INTEGRATED, device)
        .map_err(unavailable)?
        != 0;
    Ok(DeviceDescriptor {
        ordinal,
        uuid,
        name: unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
        total_memory_bytes: memory as u64,
        integrated,
    })
}

/// Gathers the complete profile parts of one device ordinal.
pub(crate) fn discover(ordinal: u32) -> Result<DiscoveredTarget<Cuda>, TargetError> {
    let driver = Driver::load().map_err(TargetError::DeviceUnavailable)?;
    let device = c_int::try_from(ordinal).map_err(|_| {
        TargetError::DeviceUnavailable(format!("ordinal {ordinal} exceeds the driver ABI"))
    })?;
    unsafe {
        let mut handle = 0;
        driver
            .check((driver.device_get)(&mut handle, device), "device selection")
            .map_err(unavailable)?;
    }
    let unsigned = |key: i32| -> Result<u32, TargetError> {
        let value = driver.attribute(key, device).map_err(unavailable)?;
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
                (driver.device_name)(name.as_mut_ptr(), name.len() as c_int, device),
                "device name",
            )
            .map_err(unavailable)?;
        driver
            .check((driver.driver_version)(&mut version), "driver version")
            .map_err(unavailable)?;
        driver
            .check(
                (driver.device_total_memory)(&mut memory, device),
                "total device memory",
            )
            .map_err(unavailable)?;
    }
    let hardware = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let major = unsigned(attr::COMPUTE_CAPABILITY_MAJOR)?;
    let minor = unsigned(attr::COMPUTE_CAPABILITY_MINOR)?;
    let compute_capability = ComputeCapability {
        major: u16::try_from(major).map_err(|_| {
            TargetError::UnsupportedDevice(format!("compute capability major {major}"))
        })?,
        minor: u16::try_from(minor).map_err(|_| {
            TargetError::UnsupportedDevice(format!("compute capability minor {minor}"))
        })?,
    };
    if compute_capability < ComputeCapability::SM80 {
        return Err(TargetError::UnsupportedDevice(format!(
            "{hardware}: compute capability {compute_capability} is below the sm_80 emission floor"
        )));
    }
    let driver_api =
        DriverApiVersion(u32::try_from(version).map_err(|_| {
            TargetError::UnsupportedDriver(format!("driver API version {version}"))
        })?);
    if driver_api < DriverApiVersion::PTX_71_MINIMUM {
        return Err(TargetError::UnsupportedDriver(format!(
            "driver API {driver_api} is below {} (PTX 7.1)",
            DriverApiVersion::PTX_71_MINIMUM
        )));
    }
    let blackwell_100_family = compute_capability.major == ComputeCapability::SM100.major
        && matches!(compute_capability.minor, 0 | 3 | 7)
        && driver_api >= DriverApiVersion::PTX_88_MINIMUM;
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
    let warp_size = unsigned(attr::WARP_SIZE)?;
    let multiprocessors = unsigned(attr::MULTIPROCESSOR_COUNT)?;
    let cooperative_launch = unsigned(attr::COOPERATIVE_LAUNCH)? == 1;
    let kernel_parameter_bytes = if driver_api >= DriverApiVersion::LARGE_PARAMETERS_MINIMUM {
        KERNEL_PARAMETER_BYTES_LARGE
    } else {
        KERNEL_PARAMETER_BYTES_CLASSIC
    };
    let global_memory_bytes = memory as u64;
    let facts = CudaFacts {
        device_ordinal: ordinal,
        compute_capability,
        driver_api,
        ptx,
        tensor_memory,
        warp_size,
        multiprocessors,
        max_threads_per_multiprocessor: unsigned(attr::MAX_THREADS_PER_MULTIPROCESSOR)?,
        max_blocks_per_multiprocessor: unsigned(attr::MAX_BLOCKS_PER_MULTIPROCESSOR)?,
        registers_per_block: unsigned(attr::MAX_REGISTERS_PER_BLOCK)?,
        registers_per_multiprocessor: unsigned(attr::MAX_REGISTERS_PER_MULTIPROCESSOR)?,
        max_registers_per_thread: MAX_REGISTERS_PER_THREAD,
        codegen_registers_per_thread: CODEGEN_REGISTERS_PER_THREAD,
        shared_bytes_per_multiprocessor: u64::from(unsigned(
            attr::MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        )?),
        shared_bytes_per_block: u64::from(unsigned(attr::MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?),
        l2_cache_bytes: u64::from(unsigned(attr::L2_CACHE_SIZE)?),
        global_memory_bytes,
        cooperative_launch,
        kernel_parameter_bytes,
    };
    let limits = TargetLimits {
        max_workgroup_size: [
            u64::from(unsigned(attr::MAX_BLOCK_DIM_X)?),
            u64::from(unsigned(attr::MAX_BLOCK_DIM_Y)?),
            u64::from(unsigned(attr::MAX_BLOCK_DIM_Z)?),
        ],
        max_workgroup_threads: u64::from(unsigned(attr::MAX_THREADS_PER_BLOCK)?),
        max_grid: [
            u64::from(unsigned(attr::MAX_GRID_DIM_X)?),
            u64::from(unsigned(attr::MAX_GRID_DIM_Y)?),
            u64::from(unsigned(attr::MAX_GRID_DIM_Z)?),
        ],
        max_workgroup_bytes: facts.shared_bytes_per_block,
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
        subgroup_width: Some(warp_size),
    };
    let dtypes = DataTypeSupport {
        scalars: [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::U32,
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
            facts.ptx.isa_major,
            facts.ptx.isa_minor,
            facts.ptx.architecture,
            match facts.ptx.feature_set {
                PtxFeatureSet::Baseline => "",
                PtxFeatureSet::Family => "f",
            }
        ),
        fingerprint: fingerprint(&facts, &limits, &dtypes, &numerics),
    };
    Ok(DiscoveredTarget {
        identity,
        limits,
        dtypes,
        vectors: VectorSupport::default(),
        numerics,
        facts,
        kernel_abi: CudaKernelAbi,
        local_realization: LocalRealizationPolicy {
            workgroup: LocalRealization::NativeDynamic,
            participant: LocalRealization::InvocationScratchPerParticipant,
            register: LocalRealization::InvocationScratchPerParticipant,
        },
    })
}

/// Acquires and assembles the complete profile on the exact context and
/// production stream retained by the opened device service.
pub(crate) fn profile_for_opened(
    ordinal: u32,
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    device: Arc<seismic_target::DeviceDescription<Cuda>>,
) -> Result<seismic_compiler::evaluation::AnalyticalEvaluationContext<Cuda>, TargetError> {
    let total_started = Instant::now();
    if context.ordinal()
        != c_int::try_from(ordinal).map_err(|_| {
            TargetError::DeviceUnavailable(format!("ordinal {ordinal} exceeds the driver ABI"))
        })?
        || !Arc::ptr_eq(context, stream.context())
    {
        panic!("CUDA profile acquisition received a context/stream for another opened device")
    }
    let (services, composition_qualification, probe_build_ns, probe_execution_ns) =
        acquire_services(context, stream, device.facts())?;
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
) -> Result<Arc<seismic_target::DeviceDescription<Cuda>>, TargetError> {
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
const HOST_COMPUTE_AMPLIFICATION_UNITS: u64 = 268_435_456;
const GPU_CONFIDENCE_Z_95: f64 = 1.96;
/// NVIDIA's CUDA Driver API documents `cuEventElapsedTime` with approximately
/// 0.5 microsecond resolution. This is clock quantization, not the latency of
/// an empty kernel measured by that clock.
const CUDA_EVENT_TIMER_RESOLUTION_NS: u64 = 500;
const MAX_GPU_ACQUISITION_ROUNDS: usize = 3;
const MAX_HOST_ACQUISITION_ROUNDS: usize = 8;

fn acquire_services(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    facts: &CudaFacts,
) -> Result<
    (
        Vec<ServiceDefinition>,
        CompositionQualificationParts,
        u64,
        u64,
    ),
    TargetError,
> {
    let build_started = Instant::now();
    let source = probe_source(facts.ptx);
    let image = compile_image(context, &source).map_err(|error| {
        TargetError::DeviceUnavailable(format!("CUDA probe compilation: {error:?}"))
    })?;
    let (module, empty) = load_module(context, &image, "probe_empty").map_err(|error| {
        TargetError::DeviceUnavailable(format!("CUDA probe loading: {error:?}"))
    })?;
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
        let function = module_function(&module, entry).map_err(|error| {
            TargetError::DeviceUnavailable(format!("CUDA probe `{entry}` lookup: {error:?}"))
        })?;
        let latency_units = u64::from(latency_block[0])
            * u64::from(latency_block[1])
            * u64::from(latency_block[2])
            * operations_per_thread
            / participants_per_unit;
        let capacity_grid = [facts.multiprocessors.max(1), 1, 1];
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
    let sms = facts.multiprocessors.max(1);
    let lanes = facts.max_threads_per_multiprocessor.max(1);
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
        context,
        stream,
        empty,
        &allocation,
        facts,
        timer_resolution,
        &baseline,
    )?);
    let required: BTreeSet<_> =
        <seismic_estimator::CoreService as seismic_estimator::AnalyticalService>::ALL
            .iter()
            .copied()
            .map(|service| {
                ServiceClassId::new(seismic_estimator::AnalyticalService::stable_name(service))
            })
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
    unsafe {
        context
            .driver
            .check(
                (context.driver.launch)(
                    function,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "profile probe launch",
            )
            .map_err(unavailable)?;
    }
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
    let host_fact = probe.starts_with("probe_scalar_") || probe == "probe_data_check";
    let stochastic_host_fact = probe == "probe_scalar_read";
    let latency_interval = widen_interval(
        if host_fact && !stochastic_host_fact {
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
    } else if host_fact && !stochastic_host_fact {
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
    let method = if stochastic_host_fact {
        "bounded-whole-batch-v5/host-monotonic-confidence95-differential"
    } else if host_fact {
        "bounded-whole-batch-v5/host-monotonic-extrema-differential"
    } else {
        "bounded-whole-batch-v5/cuda-event-500ns-confidence95-differential"
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

fn acquire_core_services(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    _empty: crate::driver::Handle,
    allocation: &Allocation,
    _facts: &CudaFacts,
    event_resolution: DurationInterval,
    launch_baseline: &AcquiredBatch,
) -> Result<Vec<ServiceDefinition>, TargetError> {
    let topology = ResourceTopology {
        resources: 1,
        max_concurrency: 1,
    };
    let mut services = vec![measured_definition(
        SERVICE_SUBMISSION,
        topology,
        "probe_empty",
        event_resolution,
        launch_baseline,
        launch_baseline,
        launch_baseline,
    )];
    let bytes = 64 * 1024 * 1024;
    let transfer_source = Allocation::new(context, bytes).map_err(unavailable)?;
    let transfer_destination = Allocation::new(context, bytes).map_err(unavailable)?;
    let copy = measure_stream_operation(
        context,
        stream,
        bytes as u64,
        event_resolution,
        2,
        "probe_device_copy",
        || unsafe {
            context.driver.check(
                (context.driver.memcpy_device_async)(
                    transfer_destination.pointer,
                    transfer_source.pointer,
                    bytes,
                    stream.raw(),
                ),
                "profile device copy",
            )
        },
    )?;
    let fill = measure_stream_operation(
        context,
        stream,
        bytes as u64,
        event_resolution,
        2,
        "probe_device_fill",
        || unsafe {
            context.driver.check(
                (context.driver.memset_d8_async)(transfer_source.pointer, 0, bytes, stream.raw()),
                "profile device fill",
            )
        },
    )?;
    services.push(measured_definition(
        SERVICE_COPY,
        topology,
        "probe_device_copy",
        event_resolution,
        launch_baseline,
        &copy,
        &copy,
    ));
    services.push(measured_definition(
        SERVICE_FILL,
        topology,
        "probe_device_fill",
        event_resolution,
        launch_baseline,
        &fill,
        &fill,
    ));

    let host_resolution = host_timer_resolution()?;
    let host_baseline = measure_host(
        HOST_COMPUTE_AMPLIFICATION_UNITS,
        host_resolution,
        1,
        "probe_host_baseline",
        || std::hint::black_box(()),
    )?;
    let scalar_read = measure_host_result(32_768, host_resolution, 2, "probe_scalar_read", || {
        let mut bytes = [0u8; 4];
        allocation.download_at(0, &mut bytes).map_err(unavailable)
    })?;
    let scalar_move = measure_host(
        HOST_COMPUTE_AMPLIFICATION_UNITS,
        host_resolution,
        1,
        "probe_scalar_move",
        || {
            let source = std::hint::black_box(0x1234_5678_u64);
            std::hint::black_box(source);
        },
    )?;
    let data_check = measure_host(
        HOST_COMPUTE_AMPLIFICATION_UNITS,
        host_resolution,
        1,
        "probe_data_check",
        || {
            let left = std::hint::black_box(0x1234_5678_u64);
            let right = std::hint::black_box(0x1234_5678_u64);
            std::hint::black_box(left == right);
        },
    )?;
    for (class, probe, observations) in [
        (SERVICE_SCALAR_READ, "probe_scalar_read", scalar_read),
        (SERVICE_SCALAR_MOVE, "probe_scalar_move", scalar_move),
        (SERVICE_DATA_CHECK, "probe_data_check", data_check),
    ] {
        services.push(measured_definition(
            class,
            topology,
            probe,
            host_resolution,
            &host_baseline,
            &observations,
            &observations,
        ));
    }
    Ok(services)
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

fn measure_host(
    mut units: u64,
    timer_resolution: DurationInterval,
    percent: u128,
    probe: &str,
    mut operation: impl FnMut(),
) -> Result<AcquiredBatch, TargetError> {
    units = units.max(1);
    let mut last = None;
    for _ in 0..MAX_HOST_ACQUISITION_ROUNDS {
        let acquisition_started = Instant::now();
        for _ in 0..units {
            operation();
        }
        let observations = (0..HOST_DETERMINISTIC_REPETITIONS)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..units {
                    operation();
                }
                ProbeObservation {
                    nanoseconds: start.elapsed().as_nanos() as f64,
                    units,
                }
            })
            .collect::<Vec<_>>();
        let interval = widen_interval(
            normalized_observation_interval(&observations),
            timer_resolution,
        );
        if interval_qualified(interval, percent) {
            return Ok(AcquiredBatch {
                observations,
                acquisition_duration_ns: u64::try_from(acquisition_started.elapsed().as_nanos())
                    .unwrap_or(u64::MAX),
            });
        }
        last = Some(interval);
    }
    Err(acquisition_failure(
        probe,
        percent,
        last.unwrap_or_else(|| DurationInterval::new(0, u64::MAX, 1)),
        MAX_HOST_ACQUISITION_ROUNDS,
    ))
}

fn measure_host_result(
    mut units: u64,
    timer_resolution: DurationInterval,
    percent: u128,
    probe: &str,
    mut operation: impl FnMut() -> Result<(), TargetError>,
) -> Result<AcquiredBatch, TargetError> {
    units = units.max(1);
    let mut last = None;
    for _ in 0..MAX_HOST_ACQUISITION_ROUNDS {
        let acquisition_started = Instant::now();
        for _ in 0..units {
            operation()?;
        }
        let mut observations = Vec::with_capacity(HOST_STOCHASTIC_REPETITIONS);
        for _ in 0..HOST_STOCHASTIC_REPETITIONS {
            let start = Instant::now();
            for _ in 0..units {
                operation()?;
            }
            observations.push(ProbeObservation {
                nanoseconds: start.elapsed().as_nanos() as f64,
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
            });
        }
        last = Some(interval);
    }
    Err(acquisition_failure(
        probe,
        percent,
        last.unwrap_or_else(|| DurationInterval::new(0, u64::MAX, 1)),
        MAX_HOST_ACQUISITION_ROUNDS,
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
    facts: &CudaFacts,
    limits: &TargetLimits,
    dtypes: &DataTypeSupport,
    numerics: &NumericalEnvironment,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    put_str(&mut digest, registry::REGISTRY_REVISION);
    put_str(&mut digest, BACKEND_REVISION);
    digest.update(facts.device_ordinal.to_le_bytes());
    digest.update(facts.compute_capability.major.to_le_bytes());
    digest.update(facts.compute_capability.minor.to_le_bytes());
    digest.update(facts.driver_api.0.to_le_bytes());
    digest.update(facts.ptx.isa_major.to_le_bytes());
    digest.update(facts.ptx.isa_minor.to_le_bytes());
    digest.update(facts.ptx.architecture.to_le_bytes());
    digest.update([match facts.ptx.feature_set {
        PtxFeatureSet::Baseline => 0,
        PtxFeatureSet::Family => 1,
    }]);
    match facts.tensor_memory {
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
    for value in [
        facts.warp_size,
        facts.multiprocessors,
        facts.max_threads_per_multiprocessor,
        facts.max_blocks_per_multiprocessor,
        facts.registers_per_block,
        facts.registers_per_multiprocessor,
        facts.max_registers_per_thread,
        facts.codegen_registers_per_thread,
    ] {
        digest.update(value.to_le_bytes());
    }
    for value in [
        facts.shared_bytes_per_multiprocessor,
        facts.shared_bytes_per_block,
        facts.l2_cache_bytes,
        facts.global_memory_bytes,
        facts.kernel_parameter_bytes,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update([u8::from(facts.cooperative_launch)]);
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
fn _facts_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<CudaFacts>();
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
    fn host_noop_baseline_uses_the_fixed_compute_amplification_domain() {
        assert_eq!(HOST_COMPUTE_AMPLIFICATION_UNITS, 268_435_456);
        let observed = [observation(65_536.0, 32_768)];
        let baseline = [observation(
            HOST_COMPUTE_AMPLIFICATION_UNITS as f64,
            HOST_COMPUTE_AMPLIFICATION_UNITS,
        )];
        assert_eq!(
            differential_interval(&observed, &baseline),
            DurationInterval::new(65_535, 65_535, 32_768),
        );
    }

    #[test]
    fn host_probe_classification_separates_stochastic_downloads_from_compute() {
        let batch = |values: Vec<f64>, units| AcquiredBatch {
            observations: values
                .into_iter()
                .map(|nanoseconds| observation(nanoseconds, units))
                .collect(),
            acquisition_duration_ns: 1,
        };
        let baseline = batch(
            vec![HOST_COMPUTE_AMPLIFICATION_UNITS as f64; HOST_DETERMINISTIC_REPETITIONS],
            HOST_COMPUTE_AMPLIFICATION_UNITS,
        );
        let stochastic_values = (0..HOST_STOCHASTIC_REPETITIONS)
            .map(|index| if index % 2 == 0 { 90.0 } else { 110.0 })
            .collect::<Vec<_>>();
        let stochastic = batch(stochastic_values.clone(), 1);
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

        let deterministic = batch(stochastic_values, 1);
        let scalar_move = measured_definition(
            SERVICE_SCALAR_MOVE,
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
