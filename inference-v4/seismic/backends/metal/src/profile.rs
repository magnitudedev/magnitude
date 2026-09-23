//! Metal opened-device acquisition: immutable device-wide facts are assembled
//! into `DeviceDescription<Metal>`, while fixed probe observations on the
//! production service become the separately bound `ExecutionProfile<Metal>`.
//!
//! Unknown is distinct from unsupported: a dtype is advertised only when
//! its probe compiled and produced a pipeline.

use crate::device::MetalDevice;
use crate::facts::{
    LanguageVersion, MatrixCombination, MetalFacts, MetalFamily, ARGUMENT_TABLE_ENTRIES,
    MSL_GRID_INDEX_MAX, RESERVED_ARGUMENT_ENTRIES,
};
use crate::{Metal, BACKEND_REVISION};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSProcessInfo, NSString};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
};
use seismic_compiler::errors::TargetError;
use seismic_compiler::target::{
    assemble_device_description, CompilerRegistry, CompilerRegistryParts, DiscoveredTarget,
    ExecutionProfileParts,
};
use seismic_estimator::ProfileAcquisitionMetrics;
use seismic_ir::target::{
    DataTypeSupport, KernelAbiAllocation, KernelAbiAllocationRole, KernelAbiFootprint,
    KernelAbiLayout, KernelAbiModel, LocalRealization, LocalRealizationPolicy,
    NumericalEnvironment, TargetLimits, VectorSupport,
};
use seismic_lang::registry::{self, BackendName, RepresentationKind, REGISTRY_REVISION};
use seismic_lang::types::DType;
use seismic_target::{CompatibilityIdentity, DeviceDescription};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::OnceLock;
use std::time::Instant;

pub const PROBE_SUITE_REVISION: &str = "seismic-metal-probes-v2";

fn addressable_resources(_: &MetalFacts) -> Vec<seismic_ir::target::AddressableResourceClass> {
    Vec::new()
}

/// The sealed static Metal capability registry, assembled once (§4.2).
pub fn registry() -> &'static CompilerRegistry<Metal> {
    static REGISTRY: OnceLock<CompilerRegistry<Metal>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CompilerRegistry::assemble(CompilerRegistryParts {
            capabilities: crate::intrinsic::registrations(),

            native_launch_constraints: crate::native_launch_constraints,
            addressable_resources,
            emitted_intrinsics: crate::intrinsic::emitted_intrinsics(),
        })
    })
}

/// Acquires one coherent analytical context from the already-open production
/// device service. Authoritative device truth is assembled before the model's
/// required probe vocabulary is derived from it.
pub fn open_analytical(
    device: &MetalDevice,
    description: std::sync::Arc<DeviceDescription<Metal>>,
) -> Result<seismic_compiler::evaluation::AnalyticalEvaluationContext<Metal>, TargetError> {
    let execution_parts = acquire_execution_parts(device, description)?;
    seismic_compiler::evaluation::AnalyticalEvaluationContext::assemble(
        execution_parts,
        crate::model::MetalAnalyticalModel,
    )
}

pub fn open_device(
    device: &MetalDevice,
) -> Result<std::sync::Arc<DeviceDescription<Metal>>, TargetError> {
    discover_device(device).map(std::sync::Arc::new)
}

/// Gathers every Metal target fact of one device.
fn discover_device(device: &MetalDevice) -> Result<DeviceDescription<Metal>, TargetError> {
    let handle = device.handle();
    let raw = handle.raw();
    let language = probe_language_version(raw).ok_or_else(|| {
        TargetError::UnsupportedToolchain(
            "the Metal compiler accepts none of the language versions this backend emits".into(),
        )
    })?;
    let size = raw.maxThreadsPerThreadgroup();
    let facts = MetalFacts {
        device_name: handle.name(),
        architecture: raw.architecture().name().to_string(),
        operating_system: NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string(),
        registry_id: handle.registry_id(),
        unified_memory: raw.hasUnifiedMemory(),
        families: observed_families(raw),
        language,
        max_threads_per_threadgroup: [size.width as u64, size.height as u64, size.depth as u64],
        max_threadgroup_bytes: raw.maxThreadgroupMemoryLength() as u64,
        max_buffer_bytes: raw.maxBufferLength() as u64,
        buffer_alignment: raw
            .heapBufferSizeAndAlignWithLength_options(
                1,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .align as u64,
        scalar_collective_dtypes: probe_scalar_collectives(raw, language),
        bfloat_arithmetic: probe_bfloat_arithmetic(raw, language),
        matrix_dtypes: probe_matrix_dtypes(raw, language),
        matrix_combinations: probe_matrix_combinations(raw, language),
        argument_table_entries: ARGUMENT_TABLE_ENTRIES,
        reserved_argument_entries: RESERVED_ARGUMENT_ENTRIES,
        backend_revision: BACKEND_REVISION,
    };
    assemble_device_description(parts_from_facts(facts), registry())
}

fn acquire_execution_parts(
    device: &MetalDevice,
    description: std::sync::Arc<DeviceDescription<Metal>>,
) -> Result<ExecutionProfileParts<Metal>, TargetError> {
    let total_started = Instant::now();
    let mut required_services: std::collections::BTreeSet<_> =
        <seismic_estimator::CoreService as seismic_estimator::AnalyticalService>::ALL
            .iter()
            .copied()
            .map(|service| {
                seismic_estimator::ServiceClassId::new(
                    seismic_estimator::AnalyticalService::stable_name(service),
                )
            })
            .collect();
    required_services.extend(
        <seismic_estimator_metal::MetalService as seismic_estimator::AnalyticalService>::ALL
            .iter()
            .copied()
            .filter(|service| {
                seismic_estimator_metal::service_available(
                    description.facts().bfloat_arithmetic,
                    description.intrinsics(),
                    *service,
                )
            })
            .map(|service| {
                seismic_estimator::ServiceClassId::new(
                    seismic_estimator::AnalyticalService::stable_name(service),
                )
            }),
    );
    let identity_and_limits_ns = total_started.elapsed().as_nanos() as u64;
    let service_acquisition =
        crate::services::acquire(device, description.facts(), &required_services)?;
    let acquisition = ProfileAcquisitionMetrics {
        identity_and_limits_ns,
        probe_build_ns: service_acquisition.metrics.probe_build_ns,
        probe_execution_ns: service_acquisition.metrics.probe_execution_ns,
        total_ns: total_started.elapsed().as_nanos() as u64,
    };
    Ok(ExecutionProfileParts::new(
        description,
        PROBE_SUITE_REVISION,
        service_acquisition.services,
        acquisition,
        service_acquisition.composition,
    ))
}

/// Metal's direct kernel argument ABI. Each typed buffer binding and each
/// compiler-owned auxiliary table is one 64-bit resource address. Dynamic
/// shape/layout words live in the parameter buffer and therefore do not
/// consume additional kernel-argument bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetalKernelAbi;

impl KernelAbiModel<Metal> for MetalKernelAbi {
    fn layout(&self, kernel: &seismic_ir::kernel::Kernel<Metal>) -> KernelAbiLayout {
        let interface = kernel.interface();
        let entries = interface.bindings.len() as u64 + u64::from(RESERVED_ARGUMENT_ENTRIES);
        let word_bytes =
            u64::from(seismic_ir::target::KernelWordLayout::for_kernel(kernel).total) * 8;
        let result_bytes = interface.result_slots.len() as u64 * 8;
        KernelAbiLayout {
            footprint: KernelAbiFootprint {
                bytes: entries * 8,
                alignment: 8,
            },
            allocations: vec![
                KernelAbiAllocation {
                    role: KernelAbiAllocationRole::WordTable,
                    bytes: word_bytes,
                    alignment: 8,
                },
                KernelAbiAllocation {
                    role: KernelAbiAllocationRole::ScalarResults,
                    bytes: result_bytes,
                    alignment: 8,
                },
            ],
        }
    }
}

fn limits_of(facts: &MetalFacts) -> TargetLimits {
    TargetLimits {
        max_workgroup_size: facts.max_threads_per_threadgroup,
        // The exact total is pipeline-specific. This device-wide product is
        // only an outer representability bound; every admitted kernel uses
        // its reflected `maxTotalThreadsPerThreadgroup`.
        max_workgroup_threads: facts.max_threads_per_threadgroup.into_iter().product(),
        max_grid: [MSL_GRID_INDEX_MAX; 3],
        max_workgroup_bytes: facts.max_threadgroup_bytes,
        // Metal exposes no authoritative private-stack capacity. Participant
        // locals are realized as explicit invocation scratch and constrained
        // by their global allocation topology, not by a fabricated stack
        // limit.
        participant_local_bytes: None,
        max_bindings: facts.argument_table_entries - facts.reserved_argument_entries,
        // Every Metal buffer argument is one 64-bit resource address.
        max_argument_bytes: u64::from(ARGUMENT_TABLE_ENTRIES) * 8,
        max_allocation_bytes: facts.max_buffer_bytes,
        max_allocation_alignment: facts.buffer_alignment,
        max_index_bits: 64,
        // SIMD width is reflected from each concrete pipeline.
        subgroup_width: None,
    }
}

fn dtype_support_of(facts: &MetalFacts) -> DataTypeSupport {
    let mut scalars: BTreeSet<DType> =
        [DType::F32, DType::F16, DType::I32, DType::U32, DType::Bool]
            .into_iter()
            .collect();
    if facts.bfloat_arithmetic {
        scalars.insert(DType::BF16);
    }
    // Atomics: native 32-bit fetch operations for i32/u32 and a 32-bit
    // compare-exchange loop for f32. Narrow elements are not advertised:
    // updating a containing 32-bit word would require an allocation-topology
    // padding/alignment guarantee that dense f16/bf16 storage does not own.
    let atomics: BTreeSet<DType> = [DType::I32, DType::U32, DType::F32].into_iter().collect();
    // Representation geometry and packed decode recipes are canonical
    // registry facts consumed by the shared emission layout.
    let representations = registry::representations()
        .iter()
        .filter(|info| match &info.kind {
            RepresentationKind::Dense(dtype) => scalars.contains(dtype),
            RepresentationKind::Packed(layout) => {
                scalars.contains(&info.decoded)
                    && layout
                        .planes
                        .iter()
                        .all(|plane| scalars.contains(&plane.storage_dtype))
            }
            RepresentationKind::External(_) => scalars.contains(&info.decoded),
        })
        .map(|info| info.id)
        .collect();
    DataTypeSupport {
        scalars,
        atomics,
        representations,
    }
}

fn numerical_environment() -> NumericalEnvironment {
    NumericalEnvironment {
        // `fma` exists as an explicit single-rounded operation; implicit
        // contraction is disabled (`#pragma clang fp contract(off)`).
        contraction_available: true,
        // MSL permits native FP32 arithmetic to flush denormal operands and
        // results even in safe/no-fast-math mode. Relaxed implementations may
        // use that native path explicitly; exact implementations are emitted
        // through the backend's software IEEE primitive layer.
        flush_to_zero_available: true,
        // The `fast::` namespace sequences, admitted only through an
        // `Approximate` math op the transfer records.
        approximate_transcendentals: ["exp", "log", "sin", "cos", "sqrt", "rsqrt"]
            .into_iter()
            .collect(),
        denormals_preserved: false,
    }
}

fn identity_of(
    facts: &MetalFacts,
    limits: &TargetLimits,
    dtypes: &DataTypeSupport,
    numerics: &NumericalEnvironment,
) -> CompatibilityIdentity {
    let mut hash = Sha256::new();
    let mut field = |tag: &'static [u8], bytes: &[u8]| {
        hash.update((tag.len() as u32).to_le_bytes());
        hash.update(tag);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    };
    field(b"format", b"seismic-metal-target-v1");
    field(b"registry", REGISTRY_REVISION.as_bytes());
    field(b"backend", facts.backend_revision.as_bytes());
    field(b"architecture", facts.architecture.as_bytes());
    field(b"operating-system", facts.operating_system.as_bytes());
    field(b"unified", &[u8::from(facts.unified_memory)]);
    field(b"language", facts.language.label().as_bytes());
    for family in &facts.families {
        let tag = match family {
            MetalFamily::Metal3 => vec![0],
            MetalFamily::Metal4 => vec![1],
            MetalFamily::Apple(generation) => vec![2, *generation],
            MetalFamily::Mac2 => vec![3],
        };
        field(b"family", &tag);
    }
    for value in limits.max_workgroup_size {
        field(b"max-workgroup-axis", &value.to_le_bytes());
    }
    field(
        b"max-workgroup-threads",
        &limits.max_workgroup_threads.to_le_bytes(),
    );
    for value in limits.max_grid {
        field(b"max-grid-axis", &value.to_le_bytes());
    }
    field(
        b"max-workgroup-bytes",
        &limits.max_workgroup_bytes.to_le_bytes(),
    );
    encode_optional_u64(
        &mut field,
        b"participant-local-bytes",
        limits.participant_local_bytes,
    );
    field(b"max-bindings", &limits.max_bindings.to_le_bytes());
    field(
        b"max-argument-bytes",
        &limits.max_argument_bytes.to_le_bytes(),
    );
    field(
        b"max-allocation-bytes",
        &limits.max_allocation_bytes.to_le_bytes(),
    );
    field(
        b"max-allocation-alignment",
        &limits.max_allocation_alignment.to_le_bytes(),
    );
    field(b"max-index-bits", &limits.max_index_bits.to_le_bytes());
    encode_optional_u32(&mut field, b"subgroup-width", limits.subgroup_width);
    for dtype in &dtypes.scalars {
        field(b"scalar", &[dtype_tag(*dtype)]);
    }
    for dtype in &dtypes.atomics {
        field(b"atomic", &[dtype_tag(*dtype)]);
    }
    for representation in &dtypes.representations {
        field(
            b"representation",
            registry::representation_info(*representation)
                .name
                .as_bytes(),
        );
    }
    field(b"contraction", &[u8::from(numerics.contraction_available)]);
    field(b"ftz", &[u8::from(numerics.flush_to_zero_available)]);
    field(b"denormals", &[u8::from(numerics.denormals_preserved)]);
    for name in &numerics.approximate_transcendentals {
        field(b"approximate", name.as_bytes());
    }
    for dtype in &facts.scalar_collective_dtypes {
        field(b"collective-dtype", &[dtype_tag(*dtype)]);
    }
    field(b"bfloat", &[u8::from(facts.bfloat_arithmetic)]);
    for dtype in &facts.matrix_dtypes {
        field(b"matrix-dtype", &[dtype_tag(*dtype)]);
    }
    for combination in &facts.matrix_combinations {
        field(
            b"matrix-combination",
            &[
                dtype_tag(combination.accumulator),
                dtype_tag(combination.left),
                dtype_tag(combination.right),
            ],
        );
    }
    CompatibilityIdentity {
        backend: BackendName::Metal,
        backend_revision: BACKEND_REVISION,
        hardware: format!("{} ({})", facts.device_name, facts.architecture),
        driver: format!(
            "{};registry-id={}",
            facts.operating_system, facts.registry_id
        ),
        toolchain: format!(
            "metal-compiler/{};msl-{}",
            facts.operating_system,
            facts.language.label()
        ),
        fingerprint: hash.finalize().into(),
    }
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
        DType::Bool => 5,
    }
}

fn encode_optional_u64(
    field: &mut impl FnMut(&'static [u8], &[u8]),
    tag: &'static [u8],
    value: Option<u64>,
) {
    match value {
        Some(value) => {
            field(tag, &[1]);
            field(tag, &value.to_le_bytes());
        }
        None => field(tag, &[0]),
    }
}

fn encode_optional_u32(
    field: &mut impl FnMut(&'static [u8], &[u8]),
    tag: &'static [u8],
    value: Option<u32>,
) {
    match value {
        Some(value) => {
            field(tag, &[1]);
            field(tag, &value.to_le_bytes());
        }
        None => field(tag, &[0]),
    }
}

// ---------------------------------------------------------------------------
// Device queries
// ---------------------------------------------------------------------------

fn observed_families(device: &ProtocolObject<dyn MTLDevice>) -> BTreeSet<MetalFamily> {
    [
        (MTLGPUFamily::Metal3, MetalFamily::Metal3),
        (MTLGPUFamily::Metal4, MetalFamily::Metal4),
        (MTLGPUFamily::Apple1, MetalFamily::Apple(1)),
        (MTLGPUFamily::Apple2, MetalFamily::Apple(2)),
        (MTLGPUFamily::Apple3, MetalFamily::Apple(3)),
        (MTLGPUFamily::Apple4, MetalFamily::Apple(4)),
        (MTLGPUFamily::Apple5, MetalFamily::Apple(5)),
        (MTLGPUFamily::Apple6, MetalFamily::Apple(6)),
        (MTLGPUFamily::Apple7, MetalFamily::Apple(7)),
        (MTLGPUFamily::Apple8, MetalFamily::Apple(8)),
        (MTLGPUFamily::Apple9, MetalFamily::Apple(9)),
        (MTLGPUFamily::Apple10, MetalFamily::Apple(10)),
        (MTLGPUFamily::Mac2, MetalFamily::Mac2),
    ]
    .into_iter()
    .filter(|(native, _)| device.supportsFamily(*native))
    .map(|(_, family)| family)
    .collect()
}

// ---------------------------------------------------------------------------
// Compile and pipeline probes
// ---------------------------------------------------------------------------

pub(crate) fn native_language(language: LanguageVersion) -> MTLLanguageVersion {
    match language {
        LanguageVersion::V2_3 => MTLLanguageVersion::Version2_3,
        LanguageVersion::V2_4 => MTLLanguageVersion::Version2_4,
        LanguageVersion::V3_0 => MTLLanguageVersion::Version3_0,
        LanguageVersion::V3_1 => MTLLanguageVersion::Version3_1,
        LanguageVersion::V3_2 => MTLLanguageVersion::Version3_2,
        LanguageVersion::V4_0 => MTLLanguageVersion::Version4_0,
    }
}

/// Compile options every probe and every production library share: the
/// probed language version, fast math off (§9.4).
pub(crate) fn compile_options(language: LanguageVersion) -> objc2::rc::Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(native_language(language));
    // The macOS 13 API floor; default fast math may reassociate ordered
    // operations and erase publication casts.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    options
}

const PROBE_PRELUDE: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

fn probe_language_version(device: &ProtocolObject<dyn MTLDevice>) -> Option<LanguageVersion> {
    let source = NSString::from_str(&format!(
        "{PROBE_PRELUDE}kernel void seismic_language_probe(device uint* output [[buffer(0)]]) {{ output[0] = 0u; }}\n"
    ));
    [
        LanguageVersion::V4_0,
        LanguageVersion::V3_2,
        LanguageVersion::V3_1,
        LanguageVersion::V3_0,
        LanguageVersion::V2_4,
        LanguageVersion::V2_3,
    ]
    .into_iter()
    .find(|version| {
        device
            .newLibraryWithSource_options_error(&source, Some(&compile_options(*version)))
            .is_ok()
    })
}

/// Compiles one probe kernel to a pipeline; `None` when any stage refuses.
fn probe_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
    function: &str,
    body: &str,
) -> Option<objc2::rc::Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let source = NSString::from_str(&format!("{PROBE_PRELUDE}{body}"));
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&compile_options(language)))
        .ok()?;
    let function = library.newFunctionWithName(&NSString::from_str(function))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .ok()
}

#[derive(Clone, Copy)]
enum ProbeDtype {
    F16,
    BF16,
    F32,
}

impl ProbeDtype {
    const ALL: [ProbeDtype; 3] = [ProbeDtype::F16, ProbeDtype::BF16, ProbeDtype::F32];

    fn msl(self) -> &'static str {
        match self {
            ProbeDtype::F16 => "half",
            ProbeDtype::BF16 => "bfloat",
            ProbeDtype::F32 => "float",
        }
    }

    fn dtype(self) -> DType {
        match self {
            ProbeDtype::F16 => DType::F16,
            ProbeDtype::BF16 => DType::BF16,
            ProbeDtype::F32 => DType::F32,
        }
    }
}

fn probe_scalar_collectives(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
) -> BTreeSet<DType> {
    ProbeDtype::ALL
        .into_iter()
        .filter(|probe| {
            let ty = probe.msl();
            probe_pipeline(
                device,
                language,
                "seismic_scalar_probe",
                &format!(
                    "kernel void seismic_scalar_probe(device {ty}* values [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {{ {ty} x = values[0]; values[0] = simd_shuffle(simd_sum(x) + simd_max(x) + simd_min(x), lane); }}\n"
                ),
            )
            .is_some()
        })
        .map(ProbeDtype::dtype)
        .collect()
}

fn probe_bfloat_arithmetic(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
) -> bool {
    probe_pipeline(
        device,
        language,
        "seismic_bfloat_probe",
        "kernel void seismic_bfloat_probe(device bfloat* values [[buffer(0)]]) { bfloat x = values[0]; values[0] = bfloat(fma(float(x), float(x), 1.0f)) * x; }\n",
    )
    .is_some()
}

fn probe_matrix_dtypes(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
) -> BTreeSet<DType> {
    ProbeDtype::ALL
        .into_iter()
        .filter(|probe| {
            let ty = probe.msl();
            probe_pipeline(
                device,
                language,
                "seismic_matrix_transfer_probe",
                &format!(
                    "kernel void seismic_matrix_transfer_probe(device {ty}* a [[buffer(0)]], device {ty}* c [[buffer(1)]]) {{ simdgroup_matrix<{ty}, 8, 8> f; simdgroup_load(f, a, 8); simdgroup_load(f, a, 8, ulong2(0, 0), true); simdgroup_store(f, c, 8); }}\n"
                ),
            )
            .is_some()
        })
        .map(ProbeDtype::dtype)
        .collect()
}

fn probe_matrix_combinations(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
) -> BTreeSet<MatrixCombination> {
    let mut combinations = BTreeSet::new();
    for accumulator in ProbeDtype::ALL {
        for left in ProbeDtype::ALL {
            for right in ProbeDtype::ALL {
                let (acc, l, r) = (accumulator.msl(), left.msl(), right.msl());
                let body = format!(
                    "kernel void seismic_matrix_probe(device {l}* a [[buffer(0)]], device {r}* b [[buffer(1)]], device {acc}* c [[buffer(2)]]) {{ simdgroup_matrix<{l}, 8, 8> af; simdgroup_matrix<{r}, 8, 8> bf; simdgroup_matrix<{acc}, 8, 8> cf; simdgroup_matrix<{acc}, 8, 8> df; simdgroup_load(af, a, 8); simdgroup_load(bf, b, 8); simdgroup_load(cf, c, 8); simdgroup_multiply_accumulate(df, af, bf, cf); simdgroup_store(df, c, 8); }}\n"
                );
                if probe_pipeline(device, language, "seismic_matrix_probe", &body).is_some() {
                    combinations.insert(MatrixCombination {
                        accumulator: accumulator.dtype(),
                        left: left.dtype(),
                        right: right.dtype(),
                    });
                }
            }
        }
    }
    combinations
}

/// Compiles one library from full MSL source with the production options.
pub(crate) fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
    source: &str,
) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    device
        .newLibraryWithSource_options_error(
            &NSString::from_str(source),
            Some(&compile_options(language)),
        )
        .map_err(|error| error.localizedDescription().to_string())
}

/// Pure target assembly from a complete immutable description. No device
/// opening, native compilation, or timing acquisition occurs here.
pub fn device_description_from_facts(
    facts: MetalFacts,
) -> Result<DeviceDescription<Metal>, TargetError> {
    assemble_device_description(parts_from_facts(facts), registry())
}
fn parts_from_facts(facts: MetalFacts) -> DiscoveredTarget<Metal> {
    let limits = limits_of(&facts);
    let dtypes = dtype_support_of(&facts);
    let numerics = numerical_environment();
    let identity = identity_of(&facts, &limits, &dtypes, &numerics);
    DiscoveredTarget {
        identity,
        limits,
        dtypes,
        vectors: VectorSupport::default(),
        numerics,
        facts,
        kernel_abi: MetalKernelAbi,
        local_realization: LocalRealizationPolicy {
            workgroup: LocalRealization::NativeDynamic,
            participant: LocalRealization::InvocationScratchPerParticipant,
            register: LocalRealization::InvocationScratchPerParticipant,
        },
    }
}
