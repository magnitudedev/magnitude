//! Metal opened-device acquisition: immutable device-wide facts are assembled
//! into `DeviceDescription<Metal>`, while fixed probe observations on the
//! production service become the separately bound `ExecutionProfile<Metal>`.
//!
//! Unknown is distinct from unsupported: a dtype is advertised only when
//! its probe compiled and produced a pipeline.

use crate::device::MetalDevice;
use crate::facts::{
    LanguageVersion, MatrixCombination, MetalFacts, MetalFamily, Properties, ARGUMENT_TABLE_ENTRIES,
    MSL_GRID_INDEX_MAX, RESERVED_ARGUMENT_ENTRIES,
};
use crate::{Metal, BACKEND_REVISION};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSProcessInfo, NSString};
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary, MTLLibraryError,
    MTLLibraryErrorDomain, MTLMathFloatingPointFunctions, MTLMathMode,
};
use seismic_compiler::errors::TargetError;
use seismic_compiler::target::{
    assemble_device_description, CompilerRegistry, CompilerRegistryParts, DiscoveredTarget,
    ExecutionProfileParts,
};
use seismic_estimator::ProfileAcquisitionMetrics;
use seismic_ir::physical_target::{
    DataTypeSupport, KernelAbiAllocation, KernelAbiAllocationRole, KernelAbiFootprint,
    KernelAbiLayout, KernelAbiModel, LocalRealization, LocalRealizationPolicy,
    NumericalEnvironment, TargetLimits, VectorSupport,
};
use seismic_lang::registry::{self, BackendName, RepresentationKind, REGISTRY_REVISION};
use seismic_lang::types::DType;
use seismic_native_target::{
    ArtifactKey, ArtifactStore, CompatibilityIdentity, DeviceDescription, PROBES,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub const PROBE_SUITE_REVISION: &str = "seismic-metal-probes-v2";

fn addressable_resources(
    _: &MetalFacts,
) -> Vec<seismic_ir::physical_target::AddressableResourceClass> {
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
    programs: &crate::MetalPrograms,
) -> Result<seismic_compiler::evaluation::AnalyticalEvaluationContext<Metal>, TargetError> {
    let execution_parts =
        crate::pooled(|| acquire_execution_parts(device, description, programs))?;
    seismic_compiler::evaluation::AnalyticalEvaluationContext::assemble(
        execution_parts,
        crate::model::MetalAnalyticalModel,
    )
}

/// The description of `device`, its capability probes' answers read from
/// `store` and the probes run only where it lacks them ([`probe_answers`]).
pub fn open_device(
    device: &MetalDevice,
    store: &dyn ArtifactStore,
) -> Result<std::sync::Arc<DeviceDescription<Metal>>, TargetError> {
    let answers = probe_answers(device.handle(), store);
    crate::pooled(|| discover_device(device, &answers)).map(std::sync::Arc::new)
}

/// Gathers every Metal target fact of one device, its probes answered by
/// `answers`.
fn discover_device(
    device: &MetalDevice,
    answers: &MetalProbeAnswers,
) -> Result<DeviceDescription<Metal>, TargetError> {
    let handle = device.handle();
    let raw = handle.raw();
    let MetalProbeAnswers::Accepted {
        language,
        scalar_collective_dtypes,
        bfloat_arithmetic,
        matrix_dtypes,
        matrix_combinations,
        tensor_matmul_forms,
        tensor_headers_form: _,
    } = answers.clone()
    else {
        return Err(unsupported_language());
    };
    let size = raw.maxThreadsPerThreadgroup();
    let facts = MetalFacts {
        device_name: handle.name(),
        architecture: architecture_name(raw),
        operating_system: operating_system(),
        registry_id: handle.registry_id(),
        unified_memory: handle.has_unified_memory(),
        families: observed_families(raw),
        language,
        max_threads_per_threadgroup: [size.width as u64, size.height as u64, size.depth as u64],
        max_buffer_bytes: handle.max_allocation_bytes(),
        buffer_alignment: raw
            .heapBufferSizeAndAlignWithLength_options(
                1,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .align as u64,
        scalar_collective_dtypes,
        bfloat_arithmetic,
        matrix_dtypes,
        matrix_combinations,
        properties: properties_at(raw, tensor_matmul_forms),
        argument_table_entries: ARGUMENT_TABLE_ENTRIES,
        reserved_argument_entries: RESERVED_ARGUMENT_ENTRIES,
        backend_revision: BACKEND_REVISION,
    };
    assemble_device_description(parts_from_facts(facts), registry())
}

fn acquire_execution_parts(
    device: &MetalDevice,
    description: std::sync::Arc<DeviceDescription<Metal>>,
    programs: &crate::MetalPrograms,
) -> Result<ExecutionProfileParts<Metal>, TargetError> {
    let total_started = Instant::now();
    // The GPU services the device's model admits; the core services are
    // the backend's own host and queue measurements (`services::acquire`).
    let gpu_services = <seismic_estimator_metal::MetalService as seismic_estimator::AnalyticalService>::ALL
        .iter()
        .copied()
        .filter(|service| {
            seismic_estimator_metal::service_available(
                description.facts().bfloat_arithmetic,
                description.intrinsics(),
                *service,
            )
        })
        .collect::<Vec<_>>();
    let identity_and_limits_ns = total_started.elapsed().as_nanos() as u64;
    let service_acquisition =
        crate::services::acquire(device, programs, &gpu_services)?;
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
            u64::from(seismic_ir::physical_target::KernelWordLayout::for_kernel(kernel).total) * 8;
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
        max_workgroup_bytes: facts.properties.shared_bytes_per_group.get(),
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
        [DType::F32, DType::F16, DType::I32, DType::U32, DType::U64, DType::Bool]
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
            // Row layouts are native-only (`ROW_LAYOUT_IS_NATIVE_ONLY`).
            RepresentationKind::PackedRows(_) => false,
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
        DType::U64 => 6,
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
        // MTLLanguageVersion4_1, which the bindings do not name yet.
        LanguageVersion::V4_1 => MTLLanguageVersion((4 << 16) + 1),
    }
}

/// Compile options every probe and every production library share: the
/// probed language version, fast math off (§9.4).
pub(crate) fn compile_options(language: LanguageVersion) -> objc2::rc::Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(native_language(language));
    // Math mode and floating-point function selection are macOS 15 APIs and
    // set the backend's macOS 15 floor. Default fast math may reassociate
    // ordered operations and erase publication casts.
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    options
}

const PROBE_PRELUDE: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

/// The OS version and build, as the OS reports it: the build ships the
/// Metal compiler.
pub fn operating_system() -> String {
    NSProcessInfo::processInfo()
        .operatingSystemVersionString()
        .to_string()
}

/// What a device's capability probes answered: whether its compiler
/// accepts a language version the backend emits, and, at the highest it
/// accepts, what compiles to a pipeline. Each answer is determined by the
/// probes, their compile options and this build's reading of them (the
/// formation build), the device, and the Metal compiler, which the OS build
/// ships, so it is kept in the program store under the key of exactly those
/// ([`probe_answers`]) and the probes run once per key.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MetalProbeAnswers {
    /// The compiler accepts none of the language versions this backend
    /// emits.
    Unsupported,
    Accepted {
        /// The highest language version the compiler accepts.
        language: LanguageVersion,
        /// Scalar dtypes whose SIMD-group reductions and shuffles form.
        scalar_collective_dtypes: BTreeSet<DType>,
        bfloat_arithmetic: bool,
        /// Element dtypes whose SIMD-group matrices load and store.
        matrix_dtypes: BTreeSet<DType>,
        matrix_combinations: BTreeSet<MatrixCombination>,
        /// Whether a tensor-operation `matmul2d` forms a pipeline
        /// (`seismic_lang::device::metal`'s `tensor_matmul_forms`).
        tensor_matmul_forms: bool,
        /// Whether the compiler forms the tensor-operation types and
        /// cooperative destinations (Metal 4), whatever the GPU executes.
        tensor_headers_form: bool,
    },
}

impl MetalProbeAnswers {
    /// The language version the backend compiles to, or why the device
    /// cannot run the backend's kernels.
    pub fn language(&self) -> Result<LanguageVersion, TargetError> {
        match self {
            Self::Accepted { language, .. } => Ok(*language),
            Self::Unsupported => Err(unsupported_language()),
        }
    }

    /// Whether `device`, whose probes these answers are, forms tensor
    /// operations: the property its opened description holds, from the
    /// same queries and answers, so planning before a device is opened and
    /// the opened device agree.
    pub fn forms_tensor_operations(&self, device: &crate::DeviceHandle) -> bool {
        match self {
            Self::Accepted { tensor_matmul_forms, .. } => {
                properties_at(device.raw(), *tensor_matmul_forms).tensor_ops.holds()
            }
            Self::Unsupported => false,
        }
    }

    /// Whether this host's Metal compiler forms tensor-operation types and
    /// cooperative destinations, whatever the GPU executes: what forming a
    /// `SEISMIC_HAS_TENSOR_OPS` source needs.
    pub fn forms_tensor_headers(&self) -> bool {
        match self {
            Self::Accepted { tensor_headers_form, .. } => *tensor_headers_form,
            Self::Unsupported => false,
        }
    }

    /// Every probe run on `device`, at the highest language version its
    /// compiler accepts, and the probes that did not run to a verdict
    /// ([`Probe`]), each named by its kernel and language version: the
    /// answers hold beyond this process only when there are none.
    fn probe(device: &crate::DeviceHandle) -> (Self, Vec<String>) {
        let raw = device.raw();
        let mut not_run = Vec::new();
        let mut forms = |language: LanguageVersion, function: &str, body: &str, pipeline: bool| {
            match probe(raw, language, function, body, pipeline) {
                Probe::Formed => true,
                Probe::Rejected => false,
                Probe::NotRun(error) => {
                    not_run.push(format!("{function} (msl-{}: {error})", language.label()));
                    false
                }
            }
        };
        let language_probe = language_probe();
        let Some(language) = LANGUAGE_VERSIONS
            .into_iter()
            .find(|version| forms(*version, "seismic_language_probe", &language_probe, false))
        else {
            return (Self::Unsupported, not_run);
        };
        let mut collect = |dtypes: &mut dyn Iterator<Item = (ProbeDtype, String)>, function: &str| {
            dtypes
                .filter(|(_, body)| forms(language, function, body, true))
                .map(|(probe, _)| probe.dtype())
                .collect::<BTreeSet<_>>()
        };
        let scalar_collective_dtypes = collect(
            &mut ProbeDtype::ALL.into_iter().map(|probe| (probe, scalar_collective_probe(probe))),
            "seismic_scalar_probe",
        );
        let matrix_dtypes = collect(
            &mut ProbeDtype::ALL.into_iter().map(|probe| (probe, matrix_transfer_probe(probe))),
            "seismic_matrix_transfer_probe",
        );
        let bfloat_arithmetic = forms(language, "seismic_bfloat_probe", BFLOAT_PROBE, true);
        let mut matrix_combinations = BTreeSet::new();
        for accumulator in ProbeDtype::ALL {
            for left in ProbeDtype::ALL {
                for right in ProbeDtype::ALL {
                    let body = matrix_combination_probe(accumulator, left, right);
                    if forms(language, "seismic_matrix_probe", &body, true) {
                        matrix_combinations.insert(MatrixCombination {
                            accumulator: accumulator.dtype(),
                            left: left.dtype(),
                            right: right.dtype(),
                        });
                    }
                }
            }
        }
        let metal4 = language >= LanguageVersion::V4_0;
        let tensor_matmul_forms = metal4 && forms(language, "seismic_tensor_probe", TENSOR_PROBE, true);
        let tensor_headers_form =
            metal4 && forms(language, "seismic_tensor_headers", TENSOR_HEADERS_PROBE, false);
        let answers = Self::Accepted {
            language,
            scalar_collective_dtypes,
            bfloat_arithmetic,
            matrix_dtypes,
            matrix_combinations,
            tensor_matmul_forms,
            tensor_headers_form,
        };
        (answers, not_run)
    }
}

/// The name of `device`'s GPU architecture (`MTLDevice.architecture.name`);
/// where Metal states none, the device's name (`MTLDevice.name`). An Intel
/// Mac's GPU (`MTLIGAccelDevice`, read on a Mac mini 2018 with UHD Graphics
/// 630 on macOS 15.8.2) answers nil, which objc2-metal's `architecture`
/// binding takes as never null and panics on.
pub(crate) fn architecture_name(device: &ProtocolObject<dyn MTLDevice>) -> String {
    // SAFETY: `architecture` is a property getter taking no arguments that returns an `MTLArchitecture` or nil.
    let architecture: Option<objc2::rc::Retained<objc2_metal::MTLArchitecture>> = unsafe { objc2::msg_send![device, architecture] };
    crate::pooled(|| architecture.map_or_else(|| device.name(), |architecture| architecture.name()).to_string())
}

/// The answers of `device`'s capability probes, under the key of everything
/// that determines them (this build's formation code, the device's name and
/// GPU architecture, and the OS build, which ships the Metal compiler): this
/// process's answers under that key, else `store`'s, else the probes' own,
/// run here once ([`probe_runs`]). Every answer a process has is held for
/// its life, so each key is read or probed once a process, and keys probe
/// apart. Answers are kept in `store` only when every probe reached a
/// verdict; otherwise a warning names the probes that did not, and a later
/// process probes again. Bytes under the key that do not decode are a miss.
/// A process's answers are its own whatever store a later call names: the
/// key covers what determines them, not where they are kept.
pub fn probe_answers(device: &crate::DeviceHandle, store: &dyn ArtifactStore) -> MetalProbeAnswers {
    static HELD: Mutex<BTreeMap<String, Arc<OnceLock<MetalProbeAnswers>>>> = Mutex::new(BTreeMap::new());
    let architecture = architecture_name(device.raw());
    let key = ArtifactKey::of(&[
        crate::toolchain::FORMATION_BUILD.as_bytes(),
        device.name().as_bytes(),
        architecture.as_bytes(),
        operating_system().as_bytes(),
    ]);
    let held = HELD
        .lock()
        .expect("held probe answers")
        .entry(key.as_str().to_owned())
        .or_default()
        .clone();
    held.get_or_init(|| {
        if let Some(bytes) = store.get(PROBES, &key) {
            match serde_json::from_slice(&bytes) {
                Ok(answers) => return answers,
                Err(error) => {
                    PROBE_MISSES.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        device = %device.name(),
                        %error,
                        "stored Metal capability probe answers do not decode: a miss; probing again"
                    )
                }
            }
        }
        PROBE_RUNS.fetch_add(1, Ordering::Relaxed);
        let (answers, not_run) = crate::pooled(|| MetalProbeAnswers::probe(device));
        if not_run.is_empty() {
            let bytes = serde_json::to_vec(&answers).expect("probe answers serialize");
            store.put(PROBES, &key, &bytes);
        } else {
            tracing::warn!(
                device = %device.name(),
                not_run = %not_run.join(", "),
                "Metal capability probes did not run to the compiler's verdict; their answers serve \
                 this process only and are not kept, so a later process probes again"
            );
        }
        answers
    })
    .clone()
}

/// The times this process ran its devices' capability probes: once per
/// miss of both the store and the process's answers ([`probe_answers`]).
pub fn probe_runs() -> u64 {
    PROBE_RUNS.load(Ordering::Relaxed)
}

static PROBE_RUNS: AtomicU64 = AtomicU64::new(0);

/// The stored answers this process found damaged, which it probed again
/// ([`probe_answers`]).
pub fn probe_misses() -> u64 {
    PROBE_MISSES.load(Ordering::Relaxed)
}

static PROBE_MISSES: AtomicU64 = AtomicU64::new(0);

fn unsupported_language() -> TargetError {
    TargetError::UnsupportedToolchain(
        "the Metal compiler accepts none of the language versions this backend emits".into(),
    )
}

/// The language versions the backend emits, highest first.
const LANGUAGE_VERSIONS: [LanguageVersion; 7] = [
    LanguageVersion::V4_1,
    LanguageVersion::V4_0,
    LanguageVersion::V3_2,
    LanguageVersion::V3_1,
    LanguageVersion::V3_0,
    LanguageVersion::V2_4,
    LanguageVersion::V2_3,
];

/// How one probe came out.
enum Probe {
    Formed,
    /// A verdict against the probe: its library rejected by the compiler,
    /// or its pipeline refused by the device.
    Rejected,
    /// The compiler did not reach a verdict (interrupted, or its service
    /// failed): no answer. It names the error's domain and code.
    NotRun(String),
}

/// The `NSCocoaErrorDomain` codes of a broken connection to a service
/// (`NSXPCConnectionInterrupted`, `NSXPCConnectionInvalid`,
/// `NSXPCConnectionReplyInvalid`): the Metal compiler service failed, and
/// gave no verdict.
const XPC_FAILURES: [isize; 3] = [4097, 4099, 4101];

impl Probe {
    /// How a probe whose library failed to compile with `error` came out: a
    /// rejection for a compile failure or a language version the compiler
    /// does not support, both in the Metal library domain; otherwise not
    /// run.
    fn of_library(error: &NSError) -> Self {
        let library_domain = unsafe { MTLLibraryErrorDomain };
        let verdict = [MTLLibraryError::CompileFailure, MTLLibraryError::Unsupported]
            .iter()
            .any(|code| error.code() == code.0 as isize);
        match *error.domain() == *library_domain && verdict {
            true => Self::Rejected,
            false => Self::not_run(error),
        }
    }

    /// How a probe whose compiled library's pipeline failed to form with
    /// `error` came out: not run where the compiler service failed (the
    /// library domain's internal error, or a broken service connection);
    /// otherwise the device refused it, a rejection, whatever the domain
    /// its driver reports it in.
    fn of_pipeline(error: &NSError) -> Self {
        let library_domain = unsafe { MTLLibraryErrorDomain };
        let cocoa_domain = unsafe { objc2_foundation::NSCocoaErrorDomain };
        let service_failed = (*error.domain() == *library_domain
            && error.code() == MTLLibraryError::Internal.0 as isize)
            || (*error.domain() == *cocoa_domain && XPC_FAILURES.contains(&error.code()));
        if service_failed {
            return Self::not_run(error);
        }
        tracing::debug!(domain = %error.domain(), code = error.code(), "a Metal capability probe's pipeline was refused");
        Self::Rejected
    }

    fn not_run(error: &NSError) -> Self {
        Self::NotRun(format!("{} {}", error.domain(), error.code()))
    }
}

/// One probe kernel compiled at `language`, and its pipeline formed where
/// `pipeline` ([`Probe::of_library`], [`Probe::of_pipeline`]).
fn probe(
    device: &ProtocolObject<dyn MTLDevice>,
    language: LanguageVersion,
    function: &str,
    body: &str,
    pipeline: bool,
) -> Probe {
    crate::pooled(|| {
        let source = NSString::from_str(&format!("{PROBE_PRELUDE}{body}"));
        let library = match device.newLibraryWithSource_options_error(&source, Some(&compile_options(language))) {
            Ok(library) => library,
            Err(error) => return Probe::of_library(&error),
        };
        let function = library
            .newFunctionWithName(&NSString::from_str(function))
            .unwrap_or_else(|| panic!("a compiled Metal probe library lacks its kernel `{function}`"));
        if !pipeline {
            return Probe::Formed;
        }
        match device.newComputePipelineStateWithFunction_error(&function) {
            Ok(_) => Probe::Formed,
            Err(error) => Probe::of_pipeline(&error),
        }
    })
}

fn language_probe() -> String {
    "kernel void seismic_language_probe(device uint* output [[buffer(0)]]) { output[0] = 0u; }\n".into()
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

fn scalar_collective_probe(probe: ProbeDtype) -> String {
    let ty = probe.msl();
    format!(
        "kernel void seismic_scalar_probe(device {ty}* values [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {{ {ty} x = values[0]; values[0] = simd_shuffle(simd_sum(x) + simd_max(x) + simd_min(x), lane); }}\n"
    )
}

const BFLOAT_PROBE: &str = "kernel void seismic_bfloat_probe(device bfloat* values [[buffer(0)]]) { bfloat x = values[0]; values[0] = bfloat(fma(float(x), float(x), 1.0f)) * x; }\n";

fn matrix_transfer_probe(probe: ProbeDtype) -> String {
    let ty = probe.msl();
    format!(
        "kernel void seismic_matrix_transfer_probe(device {ty}* a [[buffer(0)]], device {ty}* c [[buffer(1)]]) {{ simdgroup_matrix<{ty}, 8, 8> f; simdgroup_load(f, a, 8); simdgroup_load(f, a, 8, ulong2(0, 0), true); simdgroup_store(f, c, 8); }}\n"
    )
}

fn matrix_combination_probe(accumulator: ProbeDtype, left: ProbeDtype, right: ProbeDtype) -> String {
    let (acc, l, r) = (accumulator.msl(), left.msl(), right.msl());
    format!(
        "kernel void seismic_matrix_probe(device {l}* a [[buffer(0)]], device {r}* b [[buffer(1)]], device {acc}* c [[buffer(2)]]) {{ simdgroup_matrix<{l}, 8, 8> af; simdgroup_matrix<{r}, 8, 8> bf; simdgroup_matrix<{acc}, 8, 8> cf; simdgroup_matrix<{acc}, 8, 8> df; simdgroup_load(af, a, 8); simdgroup_load(bf, b, 8); simdgroup_load(cf, c, 8); simdgroup_multiply_accumulate(df, af, bf, cf); simdgroup_store(df, c, 8); }}\n"
    )
}

/// The queries of a device's properties (`seismic_lang::device::metal`),
/// the tensor query answered by its probe ([`MetalProbeAnswers`]).
struct Queries<'d> {
    device: &'d ProtocolObject<dyn MTLDevice>,
    tensor_matmul_forms: bool,
}

impl seismic_lang::device::metal::Queries for Queries<'_> {
    fn max_threadgroup_memory_length(&self) -> Option<u64> {
        Some(self.device.maxThreadgroupMemoryLength() as u64)
    }
    /// The cores resident launches are bound by
    /// ([`crate::registry_entry::gpu_cores`]).
    fn gpu_cores(&self) -> Option<u64> {
        Some(crate::registry_entry::gpu_cores(self.device.registryID()))
    }
    fn apple10_family(&self) -> Option<bool> {
        Some(crate::pooled(|| self.device.supportsFamily(MTLGPUFamily::Apple10)))
    }
    fn tensor_matmul_forms(&self) -> Option<bool> {
        Some(self.tensor_matmul_forms)
    }
}

/// The device's properties, its tensor query answered by
/// `tensor_matmul_forms` (Metal declares no probes).
fn properties_at(device: &ProtocolObject<dyn MTLDevice>, tensor_matmul_forms: bool) -> Properties {
    seismic_lang::device::metal::describe(&Queries { device, tensor_matmul_forms }, &Default::default())
        .expect("every Metal property's query answers")
}

/// A `matmul2d` over a device and a threadgroup tensor into a cooperative
/// F32 destination: the tensor-operation facilities native sources use
/// under `SEISMIC_HAS_TENSOR_OPS`, which forms a pipeline where the GPU
/// executes them.
const TENSOR_PROBE: &str = "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
     kernel void seismic_tensor_probe(device bfloat* a [[buffer(0)]], device float* c [[buffer(1)]], uint t [[thread_index_in_threadgroup]]) { \
     threadgroup half b[256]; b[t % 256] = half(t); threadgroup_barrier(mem_flags::mem_threadgroup); \
     auto ta = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(16, 16)); \
     auto tb = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(16, 16)); \
     auto tc = tensor<device float, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(16, 16)); \
     constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(16, 16, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); \
     mpp::tensor_ops::matmul2d<d, execution_simdgroups<1>> op; \
     auto acc = op.get_destination_cooperative_tensor<decltype(ta), decltype(tb), float>(); \
     for (uint16_t i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f; \
     op.run(ta, tb, acc); acc.store(tc); }\n";

/// Tensor-operation types and a cooperative destination, compiled only:
/// whether the compiler forms them, whatever the GPU executes.
const TENSOR_HEADERS_PROBE: &str = "#include <metal_tensor>\n\
     #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
     kernel void seismic_tensor_headers(device float* out [[buffer(0)]]) { \
     using shape = metal::extents<int32_t, 16, 16>; \
     using tile = metal::tensor<threadgroup half, shape, metal::tensor_inline>; \
     constexpr auto d = mpp::tensor_ops::matmul2d_descriptor(16, 16, 16, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); \
     mpp::tensor_ops::matmul2d<d, metal::execution_simdgroups<1>> op; \
     auto acc = op.get_destination_cooperative_tensor<tile, tile, float>(); \
     out[0] = float(acc.get_capacity()); }\n";

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
