//! Complete host discovery and the CPU target profile.
//!
//! Discovery fixes every fact used by planning or emission before a plan is
//! constructed. Native compilation only reconstructs the Cranelift ISA from
//! `HostFacts::codegen`; it never probes or narrows the target.

use crate::codegen::CodegenPolicy;
use crate::numeric::{MATH_IDENTITY, MATH_VERSION};
use crate::{registry, services, Cpu};
use seismic_compiler::errors::TargetError;
use seismic_compiler::target::{
    assemble_device_description, DiscoveredTarget, ExecutionProfileParts,
};
use seismic_estimator::ProfileAcquisitionMetrics;
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::{
    DataTypeSupport, KernelAbiFootprint, KernelAbiLayout, KernelAbiModel, LocalRealization,
    LocalRealizationPolicy, NumericalEnvironment, TargetLimits, VectorOperationClass,
    VectorSupport, VectorSupportEntry,
};
use seismic_lang::registry::{self as language_registry, BackendName, REGISTRY_REVISION};
use seismic_lang::types::DType;
use seismic_native_target::CompatibilityIdentity;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;

pub const BACKEND_REVISION: &str = "seismic-cpu-v10";
pub(crate) const SCRATCH_ALIGNMENT: u64 = 4096;

/// The host address-size limit on one allocation, aligned to the scratch
/// granule. This is the `TargetLimits` allocation limit, not RAM capacity.
pub const MAX_ALLOCATION_BYTES: u64 = (isize::MAX as u64) & !(SCRATCH_ALIGNMENT - 1);

/// The heap-backed launch-local policy. These are hard representability
/// limits, not guessed cache or stack capacities: CPU launch locals never
/// occupy the native thread stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScratchPolicy {
    pub workgroup_bytes: u64,
    pub participant_bytes: u64,
    pub alignment: u64,
}

/// Fixed-width SIMD register classes the profiled Cranelift ISA may emit.
/// Width is a target fact; factories select only from this closed set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdTier {
    Bits128,
    Bits256,
    Bits512,
}

impl SimdTier {
    pub const fn bits(self) -> u16 {
        match self {
            Self::Bits128 => 128,
            Self::Bits256 => 256,
            Self::Bits512 => 512,
        }
    }

    pub const fn f32_lanes(self) -> u16 {
        self.bits() / 32
    }
}

/// Exact ABI of the generated worker entry. The interface data is reached
/// through `LaunchFrame`; the entry itself always receives seven machine
/// words (frame, barrier, workgroup id, local id, and the workgroup,
/// participant, and register scratch bases).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostKernelAbi;

impl KernelAbiModel<Cpu> for HostKernelAbi {
    fn layout(&self, _kernel: &Kernel<Cpu>) -> KernelAbiLayout {
        KernelAbiLayout {
            footprint: KernelAbiFootprint {
                bytes: 7 * 8,
                alignment: 8,
            },
            // The host frame and its borrowed tables live for the duration
            // of the synchronous worker launch. They are not device-visible
            // physical allocations and therefore do not enter the core
            // allocation plan.
            allocations: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostFacts {
    pub workers: u32,
    pub codegen: CodegenPolicy,
    pub scratch: ScratchPolicy,
    pub simd: Vec<SimdTier>,
    /// SIMD tiers with a native single-rounded FMA. A matrix/FMA factory
    /// must select from this subset; expanding FMA into multiply-plus-add
    /// would violate the numerical contract.
    pub simd_fma: Vec<SimdTier>,
    pub math_identity: &'static str,
    pub math_version: u32,
}

pub(crate) fn profile_for_workers(
    workers: &mut crate::workers::Workers,
    device: std::sync::Arc<seismic_native_target::DeviceDescription<Cpu>>,
) -> Result<seismic_compiler::evaluation::AnalyticalEvaluationContext<Cpu>, TargetError> {
    let total_started = Instant::now();
    let probe_started = Instant::now();
    let services = services::probe_services(workers)?;
    let composition_qualification = services::qualify_compositions(workers)?;
    let probe_execution_ns = u64::try_from(probe_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let execution_parts = ExecutionProfileParts::new(
        device,
        services::probe_suite_revision(),
        services,
        ProfileAcquisitionMetrics {
            identity_and_limits_ns: 0,
            probe_build_ns: 0,
            probe_execution_ns,
            total_ns: u64::try_from(total_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        },
        composition_qualification,
    );
    seismic_compiler::evaluation::AnalyticalEvaluationContext::assemble(
        execution_parts,
        crate::model::CpuAnalyticalModel,
    )
}

pub(crate) fn device_for_workers(
    pool: &crate::workers::Workers,
) -> Result<std::sync::Arc<seismic_native_target::DeviceDescription<Cpu>>, TargetError> {
    let workers = u32::try_from(pool.count())
        .map_err(|_| TargetError::UnsupportedDevice("CPU worker count exceeds u32::MAX".into()))?;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(TargetError::UnsupportedDevice(
        "CPU reference math requires a controllable IEEE floating environment".into(),
    ));
    let codegen = CodegenPolicy::host()?;
    if workers == 0 {
        return Err(TargetError::UnsupportedDevice(
            "the opened CPU worker pool is empty".into(),
        ));
    }
    let address_limit = MAX_ALLOCATION_BYTES;
    let scratch = ScratchPolicy {
        workgroup_bytes: address_limit,
        participant_bytes: address_limit,
        alignment: SCRATCH_ALIGNMENT,
    };
    let simd = simd_tiers(&codegen);
    let simd_fma = if codegen.triple.starts_with("aarch64")
        || codegen
            .isa_flags
            .get("has_fma")
            .is_some_and(|value| value == "true")
    {
        simd.clone()
    } else {
        Vec::new()
    };
    let facts = HostFacts {
        workers,
        codegen,
        scratch,
        simd,
        simd_fma,
        math_identity: MATH_IDENTITY,
        math_version: MATH_VERSION,
    };
    let limits = TargetLimits {
        max_workgroup_size: [u64::from(workers), 1, 1],
        max_workgroup_threads: u64::from(workers),
        // CPU workgroups are linear claims. Higher-dimensional logical
        // iteration is delinearized inside the kernel, so the physical grid
        // has one unbounded axis and no cross-axis product can overflow.
        max_grid: [u64::MAX, 1, 1],
        max_workgroup_bytes: scratch.workgroup_bytes,
        participant_local_bytes: Some(scratch.participant_bytes),
        max_bindings: u32::MAX,
        max_argument_bytes: address_limit,
        max_allocation_bytes: address_limit,
        max_allocation_alignment: SCRATCH_ALIGNMENT,
        max_index_bits: 64,
        subgroup_width: None,
    };
    let scalars: BTreeSet<_> = DType::ALL.into_iter().collect();
    // Row layouts are native-only (`ROW_LAYOUT_IS_NATIVE_ONLY`).
    let representations = language_registry::representations()
        .iter()
        .filter(|info| info.layout == language_registry::Layout::Packet)
        .map(|info| info.id)
        .collect();
    let dtypes = DataTypeSupport {
        scalars: scalars.clone(),
        atomics: scalars
            .iter()
            .copied()
            .filter(|dtype| dtype.is_numeric())
            .collect(),
        representations,
    };
    let numerics = NumericalEnvironment {
        contraction_available: true,
        flush_to_zero_available: false,
        approximate_transcendentals: BTreeSet::new(),
        denormals_preserved: true,
    };
    let vectors = vector_support(&facts);
    let identity = identity(&facts, &limits, &dtypes, &vectors, &numerics);
    let parts = DiscoveredTarget {
        identity,
        limits,
        dtypes,
        vectors,
        numerics,
        facts,
        kernel_abi: HostKernelAbi,
        local_realization: LocalRealizationPolicy {
            workgroup: LocalRealization::NativeDynamic,
            participant: LocalRealization::NativeDynamic,
            register: LocalRealization::NativeDynamic,
        },
    };
    Ok(std::sync::Arc::new(assemble_device_description(
        parts,
        registry(),
    )?))
}

fn identity(
    facts: &HostFacts,
    limits: &TargetLimits,
    dtypes: &DataTypeSupport,
    vectors: &VectorSupport,
    numerics: &NumericalEnvironment,
) -> CompatibilityIdentity {
    let mut digest = Sha256::new();
    put_str(&mut digest, REGISTRY_REVISION);
    put_str(&mut digest, BACKEND_REVISION);
    put_str(&mut digest, &facts.codegen.compiler);
    put_str(&mut digest, &facts.codegen.triple);
    digest.update([call_conv_tag(facts.codegen.call_conv)]);
    for (name, value) in &facts.codegen.shared_flags {
        put_str(&mut digest, name);
        put_str(&mut digest, value);
    }
    digest.update([0xff]);
    for (name, value) in &facts.codegen.isa_flags {
        put_str(&mut digest, name);
        put_str(&mut digest, value);
    }
    digest.update([0xfe]);
    digest.update(facts.workers.to_le_bytes());
    digest.update(facts.scratch.workgroup_bytes.to_le_bytes());
    digest.update(facts.scratch.participant_bytes.to_le_bytes());
    digest.update(facts.scratch.alignment.to_le_bytes());
    for tier in &facts.simd {
        digest.update(tier.bits().to_le_bytes());
    }
    digest.update([0xfb]);
    for tier in &facts.simd_fma {
        digest.update(tier.bits().to_le_bytes());
    }
    put_str(&mut digest, facts.math_identity);
    digest.update(facts.math_version.to_le_bytes());
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
    digest.update([0xfd]);
    for dtype in &dtypes.atomics {
        put_str(&mut digest, dtype.name());
    }
    digest.update([0xfc]);
    for representation in &dtypes.representations {
        put_str(
            &mut digest,
            language_registry::representation_info(*representation).name,
        );
    }
    for entry in &vectors.entries {
        put_str(&mut digest, entry.dtype.name());
        digest.update(entry.lanes.to_le_bytes());
        digest.update((entry.operations.len() as u64).to_le_bytes());
    }
    digest.update([
        u8::from(numerics.contraction_available),
        u8::from(numerics.flush_to_zero_available),
        u8::from(numerics.denormals_preserved),
    ]);
    for operation in &numerics.approximate_transcendentals {
        put_str(&mut digest, operation);
    }
    CompatibilityIdentity {
        backend: BackendName::Cpu,
        backend_revision: BACKEND_REVISION,
        hardware: facts.codegen.triple.clone(),
        driver: "host".into(),
        toolchain: facts.codegen.compiler.clone(),
        fingerprint: digest.finalize().into(),
    }
}

fn vector_support(facts: &HostFacts) -> VectorSupport {
    use seismic_ir::kernel::ops::{BinaryOp, BitOp, UnaryOp};

    let mut entries = Vec::new();
    for tier in &facts.simd {
        let lanes = tier.f32_lanes();
        for dtype in DType::ALL {
            let mut operations = vec![
                VectorOperationClass::Splat,
                VectorOperationClass::Lane,
                VectorOperationClass::Write {
                    representation: language_registry::dense(dtype),
                },
            ];
            operations.extend(
                language_registry::representations()
                    .iter()
                    .filter(|representation| {
                        representation.decoded == dtype
                            && !matches!(
                                representation.access,
                                language_registry::RepresentationAccess::ConversionSource
                            )
                    })
                    .map(|representation| VectorOperationClass::Read {
                        representation: representation.id,
                    }),
            );
            for to in DType::ALL {
                operations.push(VectorOperationClass::Cast { to });
            }
            if dtype == DType::Bool {
                operations.extend([
                    VectorOperationClass::Bit(BitOp::And),
                    VectorOperationClass::Bit(BitOp::Or),
                    VectorOperationClass::Bit(BitOp::Xor),
                ]);
            } else {
                operations.extend([
                    VectorOperationClass::Binary(BinaryOp::Add),
                    VectorOperationClass::Binary(BinaryOp::Sub),
                    VectorOperationClass::Binary(BinaryOp::Mul),
                    VectorOperationClass::Binary(BinaryOp::Div),
                    VectorOperationClass::Binary(BinaryOp::Rem),
                    VectorOperationClass::Binary(BinaryOp::Min),
                    VectorOperationClass::Binary(BinaryOp::Max),
                    VectorOperationClass::Unary(UnaryOp::Neg),
                    VectorOperationClass::Unary(UnaryOp::Abs),
                    VectorOperationClass::ReduceAdd,
                ]);
                if !dtype.is_float() {
                    operations.extend([
                        VectorOperationClass::Bit(BitOp::And),
                        VectorOperationClass::Bit(BitOp::Or),
                        VectorOperationClass::Bit(BitOp::Xor),
                        VectorOperationClass::Bit(BitOp::Shl),
                        VectorOperationClass::Bit(BitOp::Shr),
                    ]);
                }
                if facts.simd_fma.contains(tier) && dtype.is_float() {
                    operations.push(VectorOperationClass::Fma);
                }
            }
            entries.push(VectorSupportEntry {
                dtype,
                lanes,
                operations,
            });
        }
    }
    VectorSupport { entries }
}

fn simd_tiers(codegen: &CodegenPolicy) -> Vec<SimdTier> {
    let enabled = |name: &str| {
        codegen
            .isa_flags
            .get(name)
            .is_some_and(|value| value == "true")
    };
    let mut tiers = vec![SimdTier::Bits128];
    if enabled("has_avx2") {
        tiers.push(SimdTier::Bits256);
    }
    if enabled("has_avx512f") {
        tiers.push(SimdTier::Bits512);
    }
    tiers
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

fn call_conv_tag(call_conv: cranelift_codegen::isa::CallConv) -> u8 {
    use cranelift_codegen::isa::CallConv;
    match call_conv {
        CallConv::Fast => 0,
        CallConv::Cold => 1,
        CallConv::Tail => 2,
        CallConv::SystemV => 3,
        CallConv::WindowsFastcall => 4,
        CallConv::AppleAarch64 => 5,
        CallConv::Probestack => 6,
        CallConv::Winch => 7,
    }
}
