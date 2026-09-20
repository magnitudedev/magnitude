//! Metal runtime: device, compiled pipelines, buffers, submission, timing.

use crate::msl::{Emitted, Launch};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;
mod observation;
pub use observation::{DispatchObservation, Observation};

/// Dispatches per committed command buffer of an unprofiled batch.
const COMMIT_DISPATCHES: usize = 64;

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

pub struct Pipeline {
    states: Vec<CompiledLaunch>,
    pub emitted: Emitted,
    /// Buffers the realization needs and the caller does not supply, allocated at compile.
    scratch: Vec<Buffer>,
    identity: std::rc::Rc<()>,
    pub facts: Vec<PipelineFacts>,
}
struct CompiledLaunch {
    index: usize,
    launch: Launch,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}
impl Pipeline {
    /// Selected native dispatches. Absent retained launch slots keep their
    /// identities in the emitted artifact but do not create native pipelines.
    pub fn phase_count(&self) -> usize {
        self.states.len()
    }
}

#[derive(Clone, Debug)]
pub struct PipelineFacts {
    /// Original retained launch slot; inactive slots have no native facts.
    pub launch: usize,
    pub kernel: String,
    pub execution_width: u64,
    pub max_threads_per_group: u64,
    pub static_threadgroup_bytes: u64,
}

/// A fully bound invocation retained through synchronous batch completion.
pub struct Invocation<'a> {
    pub pipeline: &'a Pipeline,
    pub buffers: Vec<&'a Buffer>,
    pub scalars: Vec<u8>,
}

/// How a target fact was established. Absence of an observation is never treated as proof that
/// a feature is unsupported.
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

/// Vendor families are observations used to derive capabilities; they are never source-language
/// capability names.
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

/// Facts which affect legality, emission, or qualification. This deliberately contains no
/// TensorOps claims: the current emitter cannot generate those operations, and family or language
/// support alone would not prove an individual operation signature usable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalProfile {
    pub families: Observed<Vec<MetalFamily>>,
    /// Highest language version accepted by a source compilation probe using this device.
    pub language: Observed<MetalLanguageVersion>,
    /// Provenance for the quantitative fields retained directly on `DeviceInfo`.
    pub device_limits_provenance: Provenance,
    /// Metal exposes no private-stack limit. This is a conservative compiler budget, not a device
    /// limit, and its provenance must remain visible to selection and cache identity.
    pub private_storage_budget_bytes: Observed<u64>,
    /// Exact scalar collective dtypes accepted by a native compile probe.
    pub scalar_dtypes: Observed<Vec<seismic_lang::types::DType>>,
    /// Exact legacy SIMD-group matrix element types accepted for declaration/load/store.
    pub matrix_dtypes: Observed<Vec<seismic_lang::types::DType>>,
    /// Exact multiply-accumulate type combinations accepted by the native compiler.
    pub matrix_combinations: Observed<Vec<crate::target::MatrixCombination>>,
}

impl MetalProfile {
    /// Canonical capability-and-limit identity. Device marketing name and registry identity are
    /// intentionally excluded.
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
            "families={families}@{};msl={}@{};scalar={scalar}@{};matrix={matrix}@{};mma={combinations}@{};limits@{};private-budget={}@{}",
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
    /// Unavailable unless obtained from an authoritative device query.
    pub cores: Option<u32>,
    pub recommended_working_set_bytes: u64,
    pub profile: MetalProfile,
}

impl DeviceInfo {
    /// Feature legality identity. It intentionally excludes architecture and device identity so
    /// equal capability profiles compare equal across devices.
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

    /// Qualification and tuning identity. Architecture and the queried working-set budget can
    /// affect performance even when two devices expose the same capability profile.
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
        // Working-set limits do not determine execution-unit count.
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

/// Probe only the language standard itself. Successfully compiling this source does not establish
/// support for TensorOps, a data type, or any other optional intrinsic family.
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
                    "#include <metal_stdlib>\nusing namespace metal;\nkernel void seismic_scalar_probe(device {ty}* values [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {{ {ty} x = values[0]; values[0] = simd_shuffle(simd_sum(x) + simd_max(x) + simd_min(x), lane); }}\n"
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
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void seismic_matrix_transfer_probe(device {ty}* a [[buffer(0)]], device {ty}* c [[buffer(1)]]) {{ simdgroup_matrix<{ty}, 8, 8> f; simdgroup_load(f, a, 8); simdgroup_load(f, a, 8, ulong2(0, 0), true); simdgroup_store(f, c, 8); }}\n"
        )
    };
    let source = |accumulator: DType, left: DType, right: DType| {
        let (accumulator, left, right) = (
            metal_dtype(accumulator).expect("matrix probe accumulator"),
            metal_dtype(left).expect("matrix probe left"),
            metal_dtype(right).expect("matrix probe right"),
        );
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void seismic_matrix_probe(device {left}* a [[buffer(0)]], device {right}* b [[buffer(1)]], device {accumulator}* c [[buffer(2)]]) {{ simdgroup_matrix<{left}, 8, 8> af; simdgroup_matrix<{right}, 8, 8> bf; simdgroup_matrix<{accumulator}, 8, 8> cf; simdgroup_matrix<{accumulator}, 8, 8> df; simdgroup_load(af, a, 8); simdgroup_load(bf, b, 8); simdgroup_load(cf, c, 8); simdgroup_multiply_accumulate(df, af, bf, cf); simdgroup_store(df, c, 8); }}\n"
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

    pub fn compile(&self, emitted: Emitted) -> Result<Pipeline, String> {
        emitted.scalar_layout()?;
        let scalar_slot = emitted
            .buffers
            .len()
            .checked_add(emitted.scratch.len())
            .ok_or("Metal argument slot overflow")?;
        let status_slot = scalar_slot
            .checked_add(usize::from(!emitted.scalars.is_empty()))
            .ok_or("Metal argument slot overflow")?;
        if emitted.status_slot.is_some_and(|slot| slot != status_slot) {
            return Err("Metal status binding differs from the selected invocation ABI".into());
        }
        if emitted.scratch.len() != emitted.scratch_bindings.len()
            || emitted
                .scratch
                .iter()
                .zip(&emitted.scratch_bindings)
                .any(|(&bytes, binding)| bytes != binding.bytes)
        {
            return Err("Metal scratch allocations differ from their selected bindings".into());
        }
        if emitted
            .buffers
            .iter()
            .chain(&emitted.scratch_bindings)
            .any(|binding| !binding.alignment.is_power_of_two())
        {
            return Err("Metal buffer bindings require nonzero power-of-two alignment".into());
        }
        if emitted.alias_pairs.iter().any(|&(left, right, _)| {
            left >= emitted.buffers.len() || right >= emitted.buffers.len()
        }) {
            return Err("Metal alias condition names an absent invocation binding".into());
        }
        for launch in &emitted.launches {
            let absent = launch.bindings.iter().any(|binding| match *binding {
                crate::msl::Binding::Buffer(n) => n >= emitted.buffers.len(),
                crate::msl::Binding::Scratch(n) => n >= emitted.scratch.len(),
                crate::msl::Binding::Scalars => emitted.scalars.is_empty(),
                crate::msl::Binding::Status => false,
            });
            if absent || launch.bindings.len() > crate::msl::MAX_KERNEL_BUFFERS {
                return Err(format!(
                    "launch `{}` has an invalid buffer table ({} bindings)",
                    launch.kernel,
                    launch.bindings.len()
                ));
            }
            if let Some(dispatch) = &launch.dispatch {
                if *dispatch
                    != seismic_realization::dispatch::GroupDispatch::new(
                        dispatch.work_items,
                        dispatch.lanes_per_item,
                        dispatch.items_per_group,
                    )?
                    || launch.threadgroups != dispatch.groups
                    || launch.threads_per_threadgroup != dispatch.threads_per_group
                {
                    return Err("Metal launch differs from its selected dispatch geometry".into());
                }
            }
            usize::try_from(launch.threadgroups)
                .map_err(|_| "Metal grid exceeds native dimensions")?;
            usize::try_from(launch.threads_per_threadgroup)
                .map_err(|_| "Metal group exceeds native dimensions")?;
        }
        // Preserve every declared scratch slot, including empty split storage.
        // Removing one would shift the scalar and status arguments of all launches.
        let scratch = emitted
            .scratch_bindings
            .iter()
            .map(|binding| {
                let buffer = self.buffer(binding.bytes)?;
                if buffer.allocation_alignment() < binding.alignment as u64 {
                    return Err(
                        "Metal scratch allocation violates its selected binding alignment".into(),
                    );
                }
                Ok(buffer)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if emitted
            .launches
            .iter()
            .all(|launch| launch.threadgroups == 0)
        {
            return Ok(Pipeline {
                states: Vec::new(),
                emitted,
                scratch,
                identity: self.identity.clone(),
                facts: Vec::new(),
            });
        }
        let source = NSString::from_str(&emitted.source);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(self.info.profile.language.value.native());
        // Keep the macOS 13 API floor. Default Metal fast math may erase
        // publication casts and reassociate explicitly ordered operations.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = self
            .device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| format!("Metal compile failed: {}", e.localizedDescription()))?;
        let mut states = Vec::new();
        let mut facts = Vec::new();
        for (index, launch) in emitted.launches.iter().enumerate() {
            if launch.threadgroups == 0 {
                continue;
            }
            let name = NSString::from_str(&launch.kernel);
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                format!("kernel `{}` not found in compiled library", launch.kernel)
            })?;
            let state = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("pipeline creation failed: {}", e.localizedDescription()))?;
            let physical = PipelineFacts {
                launch: index,
                kernel: launch.kernel.clone(),
                execution_width: state.threadExecutionWidth() as u64,
                max_threads_per_group: state.maxTotalThreadsPerThreadgroup() as u64,
                static_threadgroup_bytes: state.staticThreadgroupMemoryLength() as u64,
            };
            if launch.threads_per_threadgroup == 0
                || launch.threads_per_threadgroup > physical.max_threads_per_group
            {
                return Err(format!(
                    "{} requests {} threads but native pipeline permits {}",
                    launch.kernel, launch.threads_per_threadgroup, physical.max_threads_per_group
                ));
            }
            if launch.declared_threadgroup_bytes > self.device.maxThreadgroupMemoryLength() as u64
                || physical.static_threadgroup_bytes
                    > self.device.maxThreadgroupMemoryLength() as u64
            {
                return Err("native pipeline threadgroup storage exceeds device capacity".into());
            }
            if let Some(dispatch) = &launch.dispatch {
                if physical.execution_width != dispatch.lanes_per_item
                    || launch.threadgroups != dispatch.groups
                    || launch.threads_per_threadgroup != dispatch.threads_per_group
                {
                    return Err(
                        "native pipeline/launch differs from declared subgroup realization".into(),
                    );
                }
            }
            facts.push(physical);
            states.push(CompiledLaunch {
                index,
                launch: launch.clone(),
                state,
            });
        }
        Ok(Pipeline {
            states,
            emitted,
            scratch,
            identity: self.identity.clone(),
            facts,
        })
    }

    /// Submit every launch of the pipeline `repeat` times, in order, in one command buffer, and
    /// wait. Returns GPU time in seconds for the whole command buffer.
    pub fn run(
        &self,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        scalars: &[u8],
        repeat: usize,
    ) -> Result<f64, String> {
        self.run_many(
            &[Invocation {
                pipeline,
                buffers: buffers.to_vec(),
                scalars: scalars.to_vec(),
            }],
            repeat,
        )
    }

    /// Validate every binding before submission, encode in source order, and
    /// retain one error status through the entire command buffer. Later dispatches
    /// cannot erase an earlier failure. Physical completion precedes return.
    pub fn run_many(&self, invocations: &[Invocation<'_>], repeat: usize) -> Result<f64, String> {
        Ok(self.submit(invocations, repeat, false)?.command_seconds)
    }

    /// Optional qualification: one sampled compute encoder per dispatch. Its
    /// stage interval differs from the uninstrumented command-buffer interval.
    /// Unsupported native counters are reported; no substituted clock is used.
    pub fn profile(
        &self,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        scalars: &[u8],
    ) -> Result<Observation, String> {
        self.submit(
            &[Invocation {
                pipeline,
                buffers: buffers.to_vec(),
                scalars: scalars.to_vec(),
            }],
            1,
            true,
        )
    }

    fn submit(
        &self,
        invocations: &[Invocation<'_>],
        repeat: usize,
        profile: bool,
    ) -> Result<Observation, String> {
        if repeat == 0 || invocations.is_empty() {
            return Err("Metal batch and repeat count must be nonempty".into());
        }
        let mut dispatch_count = 0usize;
        for invocation in invocations {
            let Invocation {
                pipeline,
                buffers,
                scalars,
            } = invocation;
            self.validate_pipeline(pipeline)?;
            if buffers.len() != pipeline.emitted.buffers.len() {
                return Err("Metal buffer binding count mismatch".into());
            }
            for (buffer, slot) in buffers.iter().zip(&pipeline.emitted.buffers) {
                self.validate_buffer(buffer)?;
                if buffer.allocation_alignment() < slot.alignment as u64
                    || !buffer.offset.is_multiple_of(slot.alignment)
                {
                    return Err("Metal resident view violates typed storage alignment".into());
                }
                if buffer.len() < slot.bytes {
                    return Err(format!(
                        "Metal buffer {}.{} has {} bytes; needs {}",
                        slot.parameter,
                        slot.plane,
                        buffer.len(),
                        slot.bytes
                    ));
                }
            }
            for (a, b, exact_allowed) in &pipeline.emitted.alias_pairs {
                let (left, right) = (buffers[*a], buffers[*b]);
                let (left_size, right_size) = (
                    pipeline.emitted.buffers[*a].bytes,
                    pipeline.emitted.buffers[*b].bytes,
                );
                if std::ptr::eq(left.raw(), right.raw())
                    && left_size != 0
                    && right_size != 0
                    && left.offset < right.offset + right_size
                    && right.offset < left.offset + left_size
                    && !(*exact_allowed && left.offset == right.offset && left_size == right_size)
                {
                    return Err("pointwise partition binding has unsafe overlapping storage".into());
                }
            }
            pipeline.emitted.scalar_layout()?.validate_bytes(scalars)?;
            dispatch_count = dispatch_count
                .checked_add(pipeline.phase_count())
                .ok_or("Metal dispatch count overflow")?;
        }
        let dispatch_count = dispatch_count
            .checked_mul(repeat)
            .ok_or("Metal repetition overflow")?;
        if dispatch_count == 0 {
            return Ok(Observation {
                command_seconds: 0.0,
                dispatches: Vec::new(),
            });
        }
        let status = self.buffer_from(&[0; 4])?;
        let mut capture = if profile {
            Some(observation::Capture::new(self, dispatch_count)?)
        } else {
            None
        };
        let open = || -> Result<_, String> {
            let command = self
                .queue
                .commandBuffer()
                .ok_or("could not create a command buffer")?;
            let encoder = if profile {
                None
            } else {
                Some(
                    command
                        .computeCommandEncoder()
                        .ok_or("could not create a compute encoder")?,
                )
            };
            Ok((command, encoder))
        };
        // Submission rule: an unprofiled batch is committed in command buffers of
        // `COMMIT_DISPATCHES` dispatches, in source order on the one queue, so the device
        // executes the head of the batch while the host encodes the rest. Completion, error
        // and status checks still cover the whole batch before return.
        let mut committed = Vec::new();
        let (mut command, mut shared_encoder) = open()?;
        let mut encoded = 0usize;
        let mut first = true;
        for _ in 0..repeat {
            for (invocation_index, invocation) in invocations.iter().enumerate() {
                let Invocation {
                    pipeline,
                    buffers,
                    scalars,
                } = invocation;
                for compiled in &pipeline.states {
                    let CompiledLaunch {
                        index: launch_index,
                        launch,
                        state,
                    } = compiled;
                    if !profile && encoded == COMMIT_DISPATCHES {
                        if let Some(encoder) = &shared_encoder {
                            encoder.endEncoding();
                        }
                        command.commit();
                        committed.push(command);
                        (command, shared_encoder) = open()?;
                        (encoded, first) = (0, true);
                    }
                    encoded += 1;
                    let encoder = match (&mut capture, &shared_encoder) {
                        (Some(capture), _) => capture.encoder(
                            &command,
                            invocation_index,
                            *launch_index,
                            &launch.kernel,
                        )?,
                        (None, Some(encoder)) => encoder.clone(),
                        (None, None) => {
                            return Err("Metal submission has no compute encoder".into())
                        }
                    };
                    if !profile && (!first || launch.after_barrier) {
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    first = false;
                    encoder.setComputePipelineState(state);
                    // Each kernel declares only the resources it references; its table maps
                    // local buffer index -> invocation resource (validated by `compile`).
                    for (local, binding) in launch.bindings.iter().enumerate() {
                        match *binding {
                            crate::msl::Binding::Buffer(n) => {
                                let b = buffers
                                    .get(n)
                                    .ok_or("Metal launch binds an absent invocation buffer")?;
                                unsafe {
                                    encoder.setBuffer_offset_atIndex(
                                        Some(&b.buffer),
                                        b.offset,
                                        local,
                                    )
                                };
                            }
                            crate::msl::Binding::Scratch(n) => {
                                let b = pipeline
                                    .scratch
                                    .get(n)
                                    .ok_or("Metal launch binds absent scratch storage")?;
                                unsafe {
                                    encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, local)
                                };
                            }
                            crate::msl::Binding::Status => unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&status.buffer), 0, local)
                            },
                            crate::msl::Binding::Scalars => {
                                let bytes = NonNull::new(scalars.as_ptr() as *mut _)
                                    .ok_or("Metal scalar block has no storage")?;
                                unsafe {
                                    encoder.setBytes_length_atIndex(bytes, scalars.len(), local)
                                };
                            }
                        }
                    }
                    let grid = MTLSize {
                        width: launch.threadgroups as usize,
                        height: 1,
                        depth: 1,
                    };
                    let group = MTLSize {
                        width: launch.threads_per_threadgroup as usize,
                        height: 1,
                        depth: 1,
                    };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
                    if profile {
                        encoder.endEncoding();
                    }
                }
            }
        }
        if let Some(encoder) = shared_encoder {
            encoder.endEncoding();
        }
        command.commit();
        committed.push(command);
        for command in &committed {
            command.waitUntilCompleted();
            if let Some(e) = command.error() {
                return Err(format!(
                    "command buffer failed: {}",
                    e.localizedDescription()
                ));
            }
        }
        if status.read(4) != [0; 4] {
            return Err("Metal invocation encountered an out-of-bounds view".into());
        }
        Ok(Observation {
            // First start to last end on the one queue, including any wait for the host.
            command_seconds: committed
                .last()
                .zip(committed.first())
                .map_or(0.0, |(last, first)| {
                    last.GPUEndTime() - first.GPUStartTime()
                }),
            dispatches: capture
                .map(|capture| capture.finish(self))
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

impl Device {
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

impl Buffer {
    /// Alignment of this allocation's GPU virtual address, independent of the
    /// CPU mapping and any retained view offset. A missing address proves only
    /// byte alignment; it does not justify an assumed page or SIMD alignment.
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

    /// Fill from bytes at an offset.
    pub fn write_at(&self, offset: usize, bytes: &[u8]) {
        assert!(offset
            .checked_add(bytes.len())
            .is_some_and(|end| end <= self.len));
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.buffer.contents().as_ptr() as *mut u8).add(self.offset + offset),
                bytes.len(),
            )
        };
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

#[cfg(test)]
mod profile_tests {
    use super::*;

    fn synthetic_info(name: &str, registry_id: u64) -> DeviceInfo {
        DeviceInfo {
            name: name.into(),
            architecture_name: "synthetic-apple".into(),
            registry_id,
            unified_memory: true,
            max_threads_per_threadgroup: 1024,
            max_threadgroup_bytes: 32 * 1024,
            max_buffer_bytes: 1 << 30,
            cores: None,
            recommended_working_set_bytes: 8 << 30,
            profile: MetalProfile {
                // Deliberately unordered and repeated: fingerprinting is canonical.
                families: Observed::new(
                    vec![
                        MetalFamily::Apple(9),
                        MetalFamily::Metal3,
                        MetalFamily::Apple(9),
                    ],
                    Provenance::DeviceQuery,
                ),
                language: Observed::new(MetalLanguageVersion::V3_2, Provenance::CompileProbe),
                device_limits_provenance: Provenance::DeviceQuery,
                private_storage_budget_bytes: Observed::new(
                    crate::mapping::CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
                    Provenance::ConservativeAssumption,
                ),
                scalar_dtypes: Observed::new(
                    vec![
                        seismic_lang::types::DType::F16,
                        seismic_lang::types::DType::F32,
                    ],
                    Provenance::CompileProbe,
                ),
                matrix_dtypes: Observed::new(
                    vec![
                        seismic_lang::types::DType::F16,
                        seismic_lang::types::DType::F32,
                    ],
                    Provenance::CompileProbe,
                ),
                matrix_combinations: Observed::new(
                    vec![crate::target::MatrixCombination {
                        accumulator: seismic_lang::types::DType::F32,
                        left: seismic_lang::types::DType::F16,
                        right: seismic_lang::types::DType::F16,
                    }],
                    Provenance::CompileProbe,
                ),
            },
        }
    }

    #[test]
    fn capability_fingerprint_excludes_device_identity() {
        let left = synthetic_info("Marketing Name A", 11);
        let right = synthetic_info("Marketing Name B", 99);
        assert_eq!(
            left.capability_fingerprint(),
            right.capability_fingerprint()
        );
        assert_eq!(left.target_fingerprint(), right.target_fingerprint());
        assert!(!left.capability_fingerprint().contains("Marketing"));

        let mut different_architecture = right;
        different_architecture.architecture_name = "synthetic-apple-next".into();
        assert_eq!(
            left.capability_fingerprint(),
            different_architecture.capability_fingerprint()
        );
        assert_ne!(
            left.target_fingerprint(),
            different_architecture.target_fingerprint()
        );
    }

    #[test]
    fn capability_fingerprint_tracks_language_and_provenance() {
        let baseline = synthetic_info("device", 1);
        let mut language = baseline.clone();
        language.profile.language.value = MetalLanguageVersion::V4_0;
        assert_ne!(
            baseline.capability_fingerprint(),
            language.capability_fingerprint()
        );

        let fingerprint = baseline.capability_fingerprint();
        assert!(fingerprint.contains("msl=3.2@probe"));
        assert!(fingerprint.contains("private-budget=131072@assumption"));
        assert!(fingerprint.contains("families=metal3,apple9@query"));
    }

    #[test]
    fn selection_uses_the_profile_bound_private_budget() {
        let mut info = synthetic_info("device", 1);
        info.profile.private_storage_budget_bytes.value = 96 * 1024;
        let limits = crate::mapping::Limits::from_device(&info);
        assert_eq!(limits.max_private_bytes, 96 * 1024);
        assert_eq!(
            info.profile.private_storage_budget_bytes.provenance,
            Provenance::ConservativeAssumption
        );
    }
}
