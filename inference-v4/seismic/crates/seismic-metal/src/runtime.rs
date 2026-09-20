//! Metal runtime: device, compiled pipelines, buffers, structured
//! submission, and reflection.
//!
//! The runtime consumes the retained structured execution tree: it validates
//! bindings, allocates the exact arena/results/status/slot/extent blocks,
//! evaluates retained execution expressions, skips zero-work launches
//! (zero-size native grids are never submitted), submits retained order, and
//! reports status after synchronous completion. It compiles no candidates,
//! retries nothing, and reconstructs no control flow. Alias validation is the
//! root-ABI `validate_alias_rules`; the assembly alias union is deleted.

use crate::msl::{Emitted, ExecutionItem, LaunchBinding};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use seismic_realization::executable::{BufferBindingId, ExecutionExpr, ResolvedExecutorScalar};
use seismic_realization::validate_alias_rules;
use std::collections::BTreeMap;
use std::ptr::NonNull;
mod observation;
pub use observation::{DispatchObservation, Observation};

pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    info: DeviceInfo,
    identity: std::rc::Rc<()>,
}

#[derive(Clone)]
pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    offset: usize,
    identity: std::rc::Rc<()>,
}

/// One compiled launch: its native pipeline state.
struct CompiledLaunch {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct Pipeline {
    states: Vec<Option<CompiledLaunch>>,
    /// The retained structured execution tree (never flattened away).
    pub emitted: Emitted,
    /// Native-reflected facts per launch: telemetry and evaluation of the
    /// selected native-resource contract only; they never change an
    /// algorithm, mapping, layout, or candidate.
    pub facts: Vec<PipelineFacts>,
    identity: std::rc::Rc<()>,
}

/// Native-reflected facts of one launch (telemetry only).
#[derive(Clone, Debug)]
pub struct PipelineFacts {
    pub launch: usize,
    pub kernel: String,
    pub execution_width: u64,
    pub max_threads_per_group: u64,
    pub static_threadgroup_bytes: u64,
}

/// A fully bound invocation: caller-supplied ABI parameter buffers (in
/// parameter-ordinal order) and scalar values (in ABI scalar order).
pub struct Invocation<'a> {
    pub pipeline: &'a Pipeline,
    pub buffers: Vec<&'a Buffer>,
    pub scalars: Vec<f64>,
}

/// The execution outcome: allocated/updated result buffers by ABI result
/// order, and per-dispatch telemetry when requested.
pub struct Outcome {
    pub results: Vec<Buffer>,
    pub dispatches: usize,
}

// ---------------------------------------------------------------------------
// Device observation (unchanged probes)
// ---------------------------------------------------------------------------

/// How a target fact was established. Absence of an observation is never
/// treated as proof that a feature is unsupported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    DeviceQuery,
    CompileProbe,
    ConservativeAssumption,
}

impl Provenance {
    fn fingerprint(self) -> &'static str {
        match self {
            Self::DeviceQuery => "query",
            Self::CompileProbe => "probe",
            Self::ConservativeAssumption => "assumption",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observed<T> {
    pub value: T,
    pub provenance: Provenance,
}

impl<T> Observed<T> {
    pub fn new(value: T, provenance: Provenance) -> Self {
        Self { value, provenance }
    }
}

/// Vendor families are observations used to derive capabilities; they are
/// never source-language capability names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalFamily {
    Metal3,
    Metal4,
    Apple(u8),
    Mac2,
}

impl MetalFamily {
    fn fingerprint(self) -> String {
        match self {
            Self::Metal3 => "metal3".into(),
            Self::Metal4 => "metal4".into(),
            Self::Apple(generation) => format!("apple{generation}"),
            Self::Mac2 => "mac2".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalLanguageVersion {
    V2_3,
    V2_4,
    V3_0,
    V3_1,
    V3_2,
    V4_0,
}

impl MetalLanguageVersion {
    fn native(self) -> MTLLanguageVersion {
        match self {
            Self::V2_3 => MTLLanguageVersion::Version2_3,
            Self::V2_4 => MTLLanguageVersion::Version2_4,
            Self::V3_0 => MTLLanguageVersion::Version3_0,
            Self::V3_1 => MTLLanguageVersion::Version3_1,
            Self::V3_2 => MTLLanguageVersion::Version3_2,
            Self::V4_0 => MTLLanguageVersion::Version4_0,
        }
    }

    fn fingerprint(self) -> &'static str {
        match self {
            Self::V2_3 => "2.3",
            Self::V2_4 => "2.4",
            Self::V3_0 => "3.0",
            Self::V3_1 => "3.1",
            Self::V3_2 => "3.2",
            Self::V4_0 => "4.0",
        }
    }
}

/// Facts which affect legality, emission, or qualification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalProfile {
    pub families: Observed<Vec<MetalFamily>>,
    pub language: Observed<MetalLanguageVersion>,
    pub device_limits_provenance: Provenance,
    /// Metal exposes no private-stack limit; a conservative compiler budget.
    pub private_storage_budget_bytes: Observed<u64>,
    /// Exact scalar collective dtypes accepted by a native compile probe.
    pub scalar_dtypes: Observed<Vec<seismic_lang::types::DType>>,
    /// Exact legacy SIMD-group matrix element types accepted for
    /// declaration/load/store (observed; matrix stays unregistered until
    /// fragment emission is complete).
    pub matrix_dtypes: Observed<Vec<seismic_lang::types::DType>>,
    /// Exact multiply-accumulate type combinations accepted by the native
    /// compiler (observed; unregistered).
    pub matrix_combinations: Observed<Vec<crate::target::MatrixCombination>>,
}

impl MetalProfile {
    /// Canonical capability-and-limit identity.
    pub fn fingerprint(&self) -> String {
        let mut family_values = self.families.value.clone();
        family_values.sort_unstable();
        family_values.dedup();
        let families = family_values
            .iter()
            .map(|family| (*family).fingerprint())
            .collect::<Vec<_>>()
            .join(",");
        let mut scalar = self.scalar_dtypes.value.clone();
        scalar.sort_by_key(|dtype| dtype.name());
        scalar.dedup();
        let scalar = scalar
            .iter()
            .map(|dtype| dtype.name())
            .collect::<Vec<_>>()
            .join(",");
        let mut matrix = self.matrix_dtypes.value.clone();
        matrix.sort_by_key(|dtype| dtype.name());
        matrix.dedup();
        let matrix = matrix
            .iter()
            .map(|dtype| dtype.name())
            .collect::<Vec<_>>()
            .join(",");
        let mut combinations = self.matrix_combinations.value.clone();
        combinations.sort_by_key(|combination| {
            format!(
                "{}:{}:{}",
                combination.accumulator.name(),
                combination.left.name(),
                combination.right.name()
            )
        });
        combinations.dedup();
        let combinations = combinations
            .iter()
            .map(|combination| {
                format!(
                    "{}:{}:{}",
                    combination.accumulator.name(),
                    combination.left.name(),
                    combination.right.name()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "families={families}@{};msl={}@{};scalar={scalar}@{};matrix={matrix}@{};\
             mma={combinations}@{};limits@{};private-budget={}@{}",
            self.families.provenance.fingerprint(),
            self.language.value.fingerprint(),
            self.language.provenance.fingerprint(),
            self.scalar_dtypes.provenance.fingerprint(),
            self.matrix_dtypes.provenance.fingerprint(),
            self.matrix_combinations.provenance.fingerprint(),
            self.device_limits_provenance.fingerprint(),
            self.private_storage_budget_bytes.value,
            self.private_storage_budget_bytes.provenance.fingerprint(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub architecture_name: String,
    pub registry_id: u64,
    pub unified_memory: bool,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    pub max_buffer_bytes: u64,
    pub cores: Option<u32>,
    pub recommended_working_set_bytes: u64,
    pub profile: MetalProfile,
}

impl DeviceInfo {
    /// Feature legality identity (excludes device identity).
    pub fn capability_fingerprint(&self) -> String {
        format!(
            "seismic-metal-profile-v1;{};unified={};threads={};threadgroup={};buffer={}",
            self.profile.fingerprint(),
            self.unified_memory,
            self.max_threads_per_threadgroup,
            self.max_threadgroup_bytes,
            self.max_buffer_bytes,
        )
    }

    /// Qualification and tuning identity.
    pub fn target_fingerprint(&self) -> String {
        format!(
            "{};architecture={};working-set={}",
            self.capability_fingerprint(),
            self.architecture_name,
            self.recommended_working_set_bytes,
        )
    }
}

fn observe_device(device: &ProtocolObject<dyn MTLDevice>) -> Result<DeviceInfo, String> {
    let size = device.maxThreadsPerThreadgroup();
    let families = observed_families(device);
    let language = probe_language_version(device)
        .ok_or("Metal compiler accepts none of the language versions supported by seismic-metal")?;
    let scalar_dtypes = probe_scalar_dtypes(device, language);
    let (matrix_dtypes, matrix_combinations) = probe_matrix_signatures(device, language);
    Ok(DeviceInfo {
        name: device.name().to_string(),
        architecture_name: device.architecture().name().to_string(),
        registry_id: device.registryID(),
        unified_memory: device.hasUnifiedMemory(),
        max_threads_per_threadgroup: size.width as u64,
        max_threadgroup_bytes: device.maxThreadgroupMemoryLength() as u64,
        max_buffer_bytes: device.maxBufferLength() as u64,
        cores: None,
        recommended_working_set_bytes: device.recommendedMaxWorkingSetSize(),
        profile: MetalProfile {
            families: Observed::new(families, Provenance::DeviceQuery),
            language: Observed::new(language, Provenance::CompileProbe),
            device_limits_provenance: Provenance::DeviceQuery,
            private_storage_budget_bytes: Observed::new(
                crate::mapping::CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
                Provenance::ConservativeAssumption,
            ),
            scalar_dtypes: Observed::new(scalar_dtypes, Provenance::CompileProbe),
            matrix_dtypes: Observed::new(matrix_dtypes, Provenance::CompileProbe),
            matrix_combinations: Observed::new(matrix_combinations, Provenance::CompileProbe),
        },
    })
}

fn observed_families(device: &ProtocolObject<dyn MTLDevice>) -> Vec<MetalFamily> {
    let mut families = Vec::new();
    for (native, family) in [
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
    ] {
        if device.supportsFamily(native) {
            families.push(family);
        }
    }
    families.sort_unstable();
    families
}

fn probe_language_version(device: &ProtocolObject<dyn MTLDevice>) -> Option<MetalLanguageVersion> {
    const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void seismic_language_probe(device uint* output [[buffer(0)]]) { output[0] = 0; }\n";
    let source = NSString::from_str(SOURCE);
    [
        MetalLanguageVersion::V4_0,
        MetalLanguageVersion::V3_2,
        MetalLanguageVersion::V3_1,
        MetalLanguageVersion::V3_0,
        MetalLanguageVersion::V2_4,
        MetalLanguageVersion::V2_3,
    ]
    .into_iter()
    .find(|version| {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(version.native());
        device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .is_ok()
    })
}

fn compile_probe(
    device: &ProtocolObject<dyn MTLDevice>,
    language: MetalLanguageVersion,
    function: &str,
    source: &str,
) -> bool {
    let source = NSString::from_str(source);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(language.native());
    let Ok(library) = device.newLibraryWithSource_options_error(&source, Some(&options)) else {
        return false;
    };
    let name = NSString::from_str(function);
    let Some(function) = library.newFunctionWithName(&name) else {
        return false;
    };
    device
        .newComputePipelineStateWithFunction_error(&function)
        .is_ok()
}

fn metal_dtype(dtype: seismic_lang::types::DType) -> Option<&'static str> {
    use seismic_lang::types::DType;
    match dtype {
        DType::F16 => Some("half"),
        DType::BF16 => Some("bfloat"),
        DType::F32 => Some("float"),
        _ => None,
    }
}

fn probe_scalar_dtypes(
    device: &ProtocolObject<dyn MTLDevice>,
    language: MetalLanguageVersion,
) -> Vec<seismic_lang::types::DType> {
    use seismic_lang::types::DType;
    [DType::F16, DType::BF16, DType::F32]
        .into_iter()
        .filter(|&dtype| {
            let ty = metal_dtype(dtype).expect("probe dtype is a Metal float");
            compile_probe(
                device,
                language,
                "seismic_scalar_probe",
                &format!(
                    "#include <metal_stdlib>\nusing namespace metal;\nkernel void \
                     seismic_scalar_probe(device {ty}* values [[buffer(0)]], \
                     uint lane [[thread_index_in_simdgroup]]) {{ {ty} x = values[0]; \
                     values[0] = simd_shuffle(simd_sum(x) + simd_max(x) + simd_min(x), lane); }}\n"
                ),
            )
        })
        .collect()
}

fn probe_matrix_signatures(
    device: &ProtocolObject<dyn MTLDevice>,
    language: MetalLanguageVersion,
) -> (
    Vec<seismic_lang::types::DType>,
    Vec<crate::target::MatrixCombination>,
) {
    use seismic_lang::types::DType;
    let dtypes = [DType::F16, DType::BF16, DType::F32];
    let transfer_source = |dtype: DType| {
        let ty = metal_dtype(dtype).expect("matrix transfer probe dtype");
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void \
             seismic_matrix_transfer_probe(device {ty}* a [[buffer(0)]], \
             device {ty}* c [[buffer(1)]]) {{ simdgroup_matrix<{ty}, 8, 8> f; \
             simdgroup_load(f, a, 8); simdgroup_load(f, a, 8, ulong2(0, 0), true); \
             simdgroup_store(f, c, 8); }}\n"
        )
    };
    let source = |accumulator: DType, left: DType, right: DType| {
        let (accumulator, left, right) = (
            metal_dtype(accumulator).expect("matrix probe accumulator"),
            metal_dtype(left).expect("matrix probe left"),
            metal_dtype(right).expect("matrix probe right"),
        );
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void \
             seismic_matrix_probe(device {left}* a [[buffer(0)]], device {right}* b \
             [[buffer(1)]], device {accumulator}* c [[buffer(2)]]) {{ \
             simdgroup_matrix<{left}, 8, 8> af; simdgroup_matrix<{right}, 8, 8> bf; \
             simdgroup_matrix<{accumulator}, 8, 8> cf; simdgroup_matrix<{accumulator}, 8, 8> df; \
             simdgroup_load(af, a, 8); simdgroup_load(bf, b, 8); simdgroup_load(cf, c, 8); \
             simdgroup_multiply_accumulate(df, af, bf, cf); simdgroup_store(df, c, 8); }}\n"
        )
    };
    let matrix_dtypes = dtypes
        .into_iter()
        .filter(|&dtype| {
            compile_probe(
                device,
                language,
                "seismic_matrix_transfer_probe",
                &transfer_source(dtype),
            )
        })
        .collect::<Vec<_>>();
    let mut combinations = Vec::new();
    for accumulator in dtypes {
        for left in dtypes {
            for right in dtypes {
                if compile_probe(
                    device,
                    language,
                    "seismic_matrix_probe",
                    &source(accumulator, left, right),
                ) {
                    combinations.push(crate::target::MatrixCombination {
                        accumulator,
                        left,
                        right,
                    });
                }
            }
        }
    }
    (matrix_dtypes, combinations)
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

impl Device {
    pub fn open() -> Result<Device, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device
            .newCommandQueue()
            .ok_or("could not create a command queue")?;
        let info = observe_device(&device)?;
        Ok(Device {
            device,
            queue,
            info,
            identity: std::rc::Rc::new(()),
        })
    }

    pub fn info(&self) -> DeviceInfo {
        self.info.clone()
    }

    pub fn buffer(&self, len: usize) -> Result<Buffer, String> {
        let buffer = self
            .device
            .newBufferWithLength_options(len.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or("buffer allocation failed")?;
        Ok(Buffer {
            buffer,
            len,
            offset: 0,
            identity: self.identity.clone(),
        })
    }

    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let b = self.buffer(bytes.len())?;
        b.write(bytes);
        Ok(b)
    }

    /// Compile every emitted launch once; statically skipped launches retain
    /// their identity without a native pipeline.
    pub fn compile(&self, emitted: Emitted) -> Result<Pipeline, String> {
        let statically_empty = |launch: &crate::msl::EmittedLaunch| launch.skipped;
        let any_source = emitted
            .launches
            .iter()
            .any(|launch| !statically_empty(launch));
        let mut states: Vec<Option<CompiledLaunch>> =
            (0..emitted.launches.len()).map(|_| None).collect();
        let mut facts = Vec::new();
        if any_source {
            let source = NSString::from_str(&emitted.source);
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(self.info.profile.language.value.native());
            // Keep the macOS 13 API floor. Default fast math may reassociate
            // explicitly ordered operations and erase publication casts.
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, Some(&options))
                .map_err(|e| format!("Metal compile failed: {}", e.localizedDescription()))?;
            for (index, launch) in emitted.launches.iter().enumerate() {
                if statically_empty(launch) {
                    continue;
                }
                let name = NSString::from_str(&launch.kernel);
                let function = library.newFunctionWithName(&name).ok_or_else(|| {
                    format!("kernel `{}` not found in compiled library", launch.kernel)
                })?;
                let state = self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| {
                        format!("pipeline creation failed: {}", e.localizedDescription())
                    })?;
                // Reflection: telemetry and native-contract evaluation only.
                let execution_width = state.threadExecutionWidth() as u64;
                let max_threads = state.maxTotalThreadsPerThreadgroup() as u64;
                if let ExecutionExpr::Const(participants) = launch.participants {
                    if participants == 0 || participants > max_threads {
                        return Err(format!(
                            "{} requests {participants} threads but the native pipeline \
                             permits {max_threads}",
                            launch.kernel
                        ));
                    }
                }
                if launch.threadgroup_bytes > self.device.maxThreadgroupMemoryLength() as u64 {
                    return Err(
                        "native pipeline threadgroup storage exceeds device capacity".into(),
                    );
                }
                facts.push(PipelineFacts {
                    launch: index,
                    kernel: launch.kernel.clone(),
                    execution_width,
                    max_threads_per_group: max_threads,
                    static_threadgroup_bytes: state.staticThreadgroupMemoryLength() as u64,
                });
                states[index] = Some(CompiledLaunch { state });
            }
        }
        Ok(Pipeline {
            states,
            emitted,
            facts,
            identity: self.identity.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Execution over the retained structured tree
// ---------------------------------------------------------------------------

/// One in-flight command buffer: launches within a `Repeat` visit share it;
/// visits are committed and completed before the binder slot is rebound.
struct Submission {
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
}

impl Submission {
    fn open() -> Result<Self, String> {
        Ok(Self {
            command: None,
            encoder: None,
        })
    }

    fn encoder(
        &mut self,
        device: &Device,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, String> {
        if self.encoder.is_none() {
            let command = device
                .queue
                .commandBuffer()
                .ok_or("could not create a command buffer")?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or("could not create a compute encoder")?;
            self.command = Some(command);
            self.encoder = Some(encoder);
        }
        Ok(self.encoder.clone().expect("the encoder is open"))
    }

    fn flush(&mut self) -> Result<(), String> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if let Some(command) = self.command.take() {
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(format!(
                    "command buffer failed: {}",
                    error.localizedDescription()
                ));
            }
        }
        Ok(())
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        // Never release an open encoder without ending it.
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if let Some(command) = self.command.take() {
            command.commit();
        }
    }
}

/// Retained-expression evaluation environment.
struct Exec<'a> {
    pipeline: &'a Pipeline,
    /// ABI buffer binding → resident buffer.
    bindings: BTreeMap<BufferBindingId, Buffer>,
    /// Scalar slot block (shared memory; CPU-visible).
    slots: Buffer,
    /// Runtime-extent values.
    extents: Vec<u64>,
    /// ABI scalar value for a leaf path; single-scalar layouts only
    /// (multi-scalar layouts need ordinal-qualified ABI paths).
    scalar_of_path: Box<dyn Fn(&seismic_lang::types::ValuePath) -> Result<f64, String> + 'a>,
}

impl Device {
    /// Validate bindings, allocate the exact planned blocks, evaluate the
    /// retained execution tree, submit in retained order, and report status
    /// after synchronous completion.
    pub fn run(&self, invocation: &Invocation<'_>) -> Result<Outcome, String> {
        let pipeline = invocation.pipeline;
        self.validate_pipeline(pipeline)?;
        let emitted = &pipeline.emitted;
        let abi = &emitted.abi;
        // Caller-supplied parameter buffers, in parameter-ordinal order.
        let parameters = abi
            .buffers
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.role,
                    seismic_realization::executable::AbiRole::Parameter { .. }
                )
            })
            .count();
        if invocation.buffers.len() != parameters {
            return Err(format!(
                "Metal invocation binds {} buffers; the ABI names {parameters} parameters",
                invocation.buffers.len()
            ));
        }
        let mut bindings: BTreeMap<BufferBindingId, Buffer> = BTreeMap::new();
        let mut parameter_buffers = invocation.buffers.iter();
        for buffer in &abi.buffers {
            match buffer.role {
                seismic_realization::executable::AbiRole::Parameter { .. } => {
                    let supplied = parameter_buffers
                        .next()
                        .ok_or("the invocation supplies every ABI parameter")?;
                    self.validate_buffer(supplied)?;
                    if supplied.allocation_alignment() < buffer.alignment as u64
                        || supplied.offset % buffer.alignment as usize != 0
                    {
                        return Err(format!(
                            "Metal parameter `{}` violates its {}-byte binding alignment",
                            buffer.path, buffer.alignment
                        ));
                    }
                    if supplied.len() < buffer.bytes as usize {
                        return Err(format!(
                            "Metal parameter `{}` has {} bytes; needs {}",
                            buffer.path,
                            supplied.len(),
                            buffer.bytes
                        ));
                    }
                    bindings.insert(buffer.binding, (*supplied).clone());
                }
                seismic_realization::executable::AbiRole::Result => {}
            }
        }
        // Scalar encoding (validated against the ABI layout).
        abi.scalars.encode(&invocation.scalars)?;
        // Scalar leaves resolve by path; only single-scalar layouts are
        // unambiguous today.
        let scalar_snapshot = invocation.scalars.clone();
        let scalar_of_path = move |path: &seismic_lang::types::ValuePath| -> Result<f64, String> {
            if path.0.is_empty() && scalar_snapshot.len() == 1 {
                Ok(scalar_snapshot[0])
            } else if path.0.is_empty() && scalar_snapshot.is_empty() {
                Err("compiler bug: an ABI scalar transport names no scalar field".into())
            } else {
                Err("compiler bug: an ABI scalar transport needs an ordinal-qualified path".into())
            }
        };
        // Result buffers are runtime-allocated by path/plane and bound by
        // every launch that writes them.
        for buffer in &abi.buffers {
            if matches!(
                buffer.role,
                seismic_realization::executable::AbiRole::Result
            ) {
                let allocated = self.buffer(buffer.bytes as usize)?;
                bindings.insert(buffer.binding, allocated);
            }
        }
        // Root-ABI alias validation over actual byte ranges (parameters and
        // allocated results; distinct allocations are disjoint by identity).
        // Allocation identity is the native buffer object; two views of one
        // allocation share it and their byte ranges decide the overlap.
        let locate = |binding: BufferBindingId| -> Result<Option<(u64, u64, u64)>, String> {
            let Some(buffer) = bindings.get(&binding) else {
                // A result binding: runtime-allocated, disjoint by construction.
                return Ok(None);
            };
            Ok(Some((
                buffer.raw() as *const _ as u64,
                buffer.offset as u64,
                buffer.len as u64,
            )))
        };
        validate_alias_rules(abi, locate)?;
        // Planned blocks.
        let arena = self.buffer(emitted.arena_bytes as usize)?;
        let slots = self.buffer(emitted.slot_count.saturating_mul(4))?;
        slots.write(&vec![0u8; emitted.slot_count * 4]);
        // Evaluate the retained runtime-extent expressions.
        let mut exec = Exec {
            pipeline,
            bindings,
            slots: slots.clone(),
            extents: Vec::new(),
            scalar_of_path: Box::new(scalar_of_path),
        };
        let mut extents = vec![0u64; emitted.extent_count];
        for (id, expr) in emitted.runtime_extents.clone() {
            let value = exec.expr(&expr)?;
            if (id.0 as usize) < extents.len() {
                extents[id.0 as usize] = value;
            }
        }
        exec.extents = extents.clone();
        let extent_buffer = self.buffer(emitted.extent_count.saturating_mul(8))?;
        {
            let words = unsafe {
                std::slice::from_raw_parts_mut(
                    extent_buffer.contents().as_ptr() as *mut u64,
                    emitted.extent_count,
                )
            };
            words.copy_from_slice(&extents);
        }
        let status_fields = abi
            .status
            .as_ref()
            .map(|status| status.fields.len())
            .unwrap_or(0);
        let status = self.buffer(status_fields.saturating_mul(4))?;
        if status_fields > 0 {
            status.write(&vec![0u8; status_fields * 4]);
        }
        // Submit the retained structured order. A `Repeat` rebinds its binder
        // slot between visits, so each visit's work is committed (and
        // completed) before the next visit is encoded; launches within one
        // visit share one command buffer.
        let mut submission = Submission::open()?;
        let mut dispatches = 0usize;
        let walked = self.walk(
            &exec,
            &emitted.execution,
            &mut submission,
            &arena,
            &extent_buffer,
            &status,
            &mut dispatches,
        );
        let flushed = submission.flush();
        walked?;
        flushed?;
        exec.assert_status(&status)?;
        // Status: first error reported after synchronous completion.
        if status_fields > 0 {
            let bytes = status.read(status_fields * 4);
            for (index, word) in bytes.chunks_exact(4).enumerate() {
                let code = u32::from_le_bytes(word.try_into().unwrap());
                if code != 0 {
                    let field = abi
                        .status
                        .as_ref()
                        .and_then(|status| status.fields.get(index));
                    return Err(format!(
                        "Metal invocation failed a safety check (status field#{index}, code \
                         {code}, node {:?})",
                        field.map(|f| (f.node.node.0, f.index))
                    ));
                }
            }
        }
        // Tensor results were allocated and bound above; scalar/index/range
        // results decode from the compiler-owned result scalar block, which
        // the runtime does not yet own.
        let mut results = Vec::new();
        for binding in &abi.results {
            match binding {
                seismic_realization::executable::ResultBinding::Buffer { binding, .. } => {
                    let buffer = exec
                        .bindings
                        .get(binding)
                        .cloned()
                        .ok_or_else(|| format!("result buffer#{binding:?} is not bound"))?;
                    results.push(buffer);
                }
                seismic_realization::executable::ResultBinding::Scalar { .. }
                | seismic_realization::executable::ResultBinding::Range { .. } => {
                    return Err(
                        "scalar/index/range result decoding from the result scalar block is \
                         not implemented"
                            .into(),
                    );
                }
            }
        }
        Ok(Outcome {
            results,
            dispatches,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        exec: &Exec<'_>,
        items: &[ExecutionItem],
        submission: &mut Submission,
        arena: &Buffer,
        extents: &Buffer,
        status: &Buffer,
        dispatches: &mut usize,
    ) -> Result<(), String> {
        for item in items {
            match item {
                ExecutionItem::Launch(index) => {
                    let launch = exec
                        .pipeline
                        .emitted
                        .launches
                        .get(*index)
                        .ok_or_else(|| format!("launch#{index} is absent"))?;
                    // Zero work is a retained launch condition: zero-size
                    // native grids are never submitted.
                    let work_items = exec.expr(&launch.work_items)?;
                    if work_items == 0 {
                        continue;
                    }
                    let participants = exec.expr(&launch.participants)?.max(1);
                    let workgroups = work_items.div_ceil(participants).max(1);
                    let compiled = exec
                        .pipeline
                        .states
                        .get(*index)
                        .and_then(|state| state.as_ref())
                        .ok_or_else(|| {
                            format!(
                                "launch#{index} `{}` has no compiled pipeline",
                                launch.kernel
                            )
                        })?;
                    let encoder = submission.encoder(self)?;
                    encoder.setComputePipelineState(&compiled.state);
                    // Storage members bind at sequential indices (the emitter's
                    // declaration order); the shared blocks bind at their fixed
                    // reserved indices.
                    let mut next_index = 0u32;
                    for binding in &launch.bindings {
                        match binding {
                            LaunchBinding::Buffer { binding, .. } => {
                                let buffer = exec.bindings.get(binding).ok_or_else(|| {
                                    format!("ABI buffer#{binding:?} is not bound")
                                })?;
                                unsafe {
                                    encoder.setBuffer_offset_atIndex(
                                        Some(buffer.raw()),
                                        buffer.offset,
                                        next_index as usize,
                                    )
                                };
                                next_index += 1;
                            }
                            LaunchBinding::Arena { offset, .. } => {
                                unsafe {
                                    encoder.setBuffer_offset_atIndex(
                                        Some(&arena.buffer),
                                        *offset as usize,
                                        next_index as usize,
                                    )
                                };
                                next_index += 1;
                            }
                            LaunchBinding::Slots => unsafe {
                                encoder.setBuffer_offset_atIndex(
                                    Some(&exec.slots.buffer),
                                    0,
                                    crate::msl::SLOT_BUFFER_INDEX as usize,
                                )
                            },
                            LaunchBinding::Extents => unsafe {
                                encoder.setBuffer_offset_atIndex(
                                    Some(&extents.buffer),
                                    0,
                                    crate::msl::EXTENT_BUFFER_INDEX as usize,
                                )
                            },
                            LaunchBinding::Status => unsafe {
                                encoder.setBuffer_offset_atIndex(
                                    Some(&status.buffer),
                                    0,
                                    crate::msl::STATUS_BUFFER_INDEX as usize,
                                )
                            },
                        }
                    }
                    if *dispatches > 0 {
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    let grid = MTLSize {
                        width: workgroups as usize,
                        height: 1,
                        depth: 1,
                    };
                    let group = MTLSize {
                        width: participants as usize,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
                    *dispatches += 1;
                }
                ExecutionItem::Call(children) => {
                    // Nested calls share the root buffers and arena; the child
                    // body was encoded against the same resolved storages.
                    self.walk(
                        exec, children, submission, arena, extents, status, dispatches,
                    )?;
                }
                ExecutionItem::If {
                    condition,
                    then_steps,
                    else_steps,
                } => {
                    // Exactly one retained predicate; exactly one branch.
                    if exec.scalar(condition)? != 0 {
                        self.walk(
                            exec, then_steps, submission, arena, extents, status, dispatches,
                        )?;
                    } else {
                        self.walk(
                            exec, else_steps, submission, arena, extents, status, dispatches,
                        )?;
                    }
                }
                ExecutionItem::Repeat {
                    binder,
                    start,
                    end,
                    body,
                } => {
                    // Ascending half-open range; the binder slot is rebound
                    // before each visit, and each visit's work completes
                    // before the next is encoded.
                    let start = exec.scalar(start)?;
                    let end = exec.scalar(end)?;
                    for value in start..end {
                        submission.flush()?;
                        exec.write_slot(binder.0, value as u32)?;
                        self.walk(exec, body, submission, arena, extents, status, dispatches)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_buffer(&self, buffer: &Buffer) -> Result<(), String> {
        if !std::rc::Rc::ptr_eq(&self.identity, &buffer.identity) {
            return Err("Metal buffer belongs to a different device owner".into());
        }
        Ok(())
    }
    pub(crate) fn validate_pipeline(&self, pipeline: &Pipeline) -> Result<(), String> {
        if !std::rc::Rc::ptr_eq(&self.identity, &pipeline.identity) {
            return Err("Metal pipeline belongs to a different device owner".into());
        }
        Ok(())
    }
}

impl Exec<'_> {
    /// Evaluate one retained execution expression.
    fn expr(&self, expression: &ExecutionExpr) -> Result<u64, String> {
        Ok(match expression {
            ExecutionExpr::Const(value) => *value,
            ExecutionExpr::Extent(id) => self
                .extents
                .get(id.0 as usize)
                .copied()
                .ok_or_else(|| format!("runtime extent#{id:?} has no evaluated value"))?,
            ExecutionExpr::AbiScalar { path, .. } => {
                let value = (self.scalar_of_path)(path)?;
                value as u64
            }
            ExecutionExpr::ExecutorScalar(slot) => self.slot_word(slot.0)? as u64,
            ExecutionExpr::Add(a, b) => self
                .expr(a)?
                .checked_add(self.expr(b)?)
                .ok_or("execution expression overflows")?,
            ExecutionExpr::Sub(a, b) => self
                .expr(a)?
                .checked_sub(self.expr(b)?)
                .ok_or("execution expression underflows")?,
            ExecutionExpr::Mul(a, b) => self
                .expr(a)?
                .checked_mul(self.expr(b)?)
                .ok_or("execution expression overflows")?,
            ExecutionExpr::CeilDiv(a, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                if b == 0 {
                    return Err("execution expression divides by zero".into());
                }
                a.div_ceil(b)
            }
            ExecutionExpr::Div(a, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                if b == 0 {
                    return Err("execution expression divides by zero".into());
                }
                a / b
            }
            ExecutionExpr::Rem(a, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                if b == 0 {
                    return Err("execution expression divides by zero".into());
                }
                a % b
            }
            ExecutionExpr::Min(a, b) => self.expr(a)?.min(self.expr(b)?),
        })
    }

    fn scalar(&self, scalar: &ResolvedExecutorScalar) -> Result<i64, String> {
        match scalar {
            ResolvedExecutorScalar::Abi { path, .. } => {
                let value = (self.scalar_of_path)(path)?;
                Ok(value as i64)
            }
            ResolvedExecutorScalar::Result { .. } => {
                Err("a result scalar cannot be read as an executor control value".into())
            }
            ResolvedExecutorScalar::Slot { slot, .. } => Ok(self.slot_word(slot.0)? as i32 as i64),
            ResolvedExecutorScalar::Computed { expr, .. } => Ok(self.expr(expr)? as i64),
        }
    }

    /// Write one executor-scalar slot (a `Repeat` binder rebind).
    fn write_slot(&self, slot: u64, word: u32) -> Result<(), String> {
        let contents = self.slots.contents();
        let words = unsafe {
            std::slice::from_raw_parts_mut(
                contents.as_ptr() as *mut u32,
                self.pipeline.emitted.slot_count,
            )
        };
        let entry = words
            .get_mut(slot as usize)
            .ok_or_else(|| format!("executor slot#{slot} is outside the planned block"))?;
        *entry = word;
        Ok(())
    }

    /// Report the first failing status field after synchronous completion.
    fn assert_status(&self, status: &Buffer) -> Result<(), String> {
        let Some(binding) = self.pipeline.emitted.abi.status.as_ref() else {
            return Ok(());
        };
        let bytes = status.read(binding.fields.len() * 4);
        for (index, word) in bytes.chunks_exact(4).enumerate() {
            let code = u32::from_le_bytes(word.try_into().unwrap());
            if code != 0 {
                let field = binding.fields.get(index);
                return Err(format!(
                    "Metal invocation failed a safety check (status field#{index}, code \
                     {code}, node {:?})",
                    field.map(|field| (field.node.node.0, field.index))
                ));
            }
        }
        Ok(())
    }

    fn slot_word(&self, slot: u64) -> Result<u32, String> {
        let words = unsafe {
            std::slice::from_raw_parts(
                self.slots.contents().as_ptr() as *const u32,
                self.pipeline.emitted.slot_count,
            )
        };
        words
            .get(slot as usize)
            .copied()
            .ok_or_else(|| format!("executor slot#{slot} is outside the planned block"))
    }
}

impl Buffer {
    /// The CPU-visible contents pointer at this view's offset.
    pub(crate) fn contents(&self) -> std::ptr::NonNull<u8> {
        std::ptr::NonNull::new(unsafe {
            (self.buffer.contents().as_ptr() as *mut u8).add(self.offset)
        })
        .expect("a shared Metal buffer has contents")
    }
    /// Alignment of this allocation's GPU virtual address.
    pub fn allocation_alignment(&self) -> u64 {
        let address = self.buffer.gpuAddress();
        if address == 0 {
            1
        } else {
            1u64 << address.trailing_zeros()
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len {
            return Err("Metal buffer view exceeds its parent".into());
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            len: range.end - range.start,
            offset: self
                .offset
                .checked_add(range.start)
                .ok_or("Metal view offset overflow")?,
            identity: self.identity.clone(),
        })
    }
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    pub fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.buffer.contents().as_ptr() as *mut u8).add(self.offset),
                bytes.len(),
            )
        };
    }

    pub fn read(&self, len: usize) -> Vec<u8> {
        assert!(len <= self.len, "Metal host read exceeds buffer capacity");
        let mut out = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.buffer.contents().as_ptr() as *const u8).add(self.offset),
                out.as_mut_ptr(),
                len,
            )
        };
        out
    }
}
