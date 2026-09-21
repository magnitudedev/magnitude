//! Metal runtime: the device facade (observation, buffers) and the
//! executor over prepared invocations.
//!
//! The executor consumes `PreparedInvocation` (package R1) and the sealed
//! native artifact: it allocates the exact arena/results/status/slot
//! blocks, evaluates the retained execution expressions against validated
//! invocation values and folded native facts, submits the native tree in
//! retained order, skips zero-work launches, threads join/carry values,
//! performs the host-side fills, and reports the status block after
//! synchronous completion. It compiles no candidates, retries nothing, and
//! reconstructs no control flow. Its only failures are `SafetyViolation`
//! (a retained guard or kernel check failed) and `ExternalFailure`
//! (driver, allocation, or submission); no `String` error channel exists
//! and nothing is looked up by a fallible identifier.

use crate::native::{
    CopyDestination, CopySource, NativeArtifact, NativeBinding, NativeStep, ScalarDestination,
    StorageTarget,
};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::types::{DType, ValuePath};
use seismic_realization::failure::{
    ExecutionFailure, ExternalFailure, ExternalStage, SafetyKind, SafetyViolation,
    SafetyViolationSource,
};
use seismic_realization::ids::{
    BufferSlot, DenseIndex, GuardIx, LaunchIx, ObligationRef, ResultFieldIx, ScalarSlotIx,
    StatusFieldIx, StorageIx,
};
use seismic_realization::invocation::InvocationValues;
use seismic_realization::physical::{
    ExecutionExpr, GuardPredicate, ScalarSource,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};
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
    pub(crate) fn native(self) -> MTLLanguageVersion {
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
    pub scalar_dtypes: Observed<Vec<DType>>,
    /// Exact legacy SIMD-group matrix element types accepted for
    /// declaration/load/store.
    pub matrix_dtypes: Observed<Vec<DType>>,
    /// Exact multiply-accumulate type combinations accepted by the native
    /// compiler.
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
                crate::target::CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
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

// ---------------------------------------------------------------------------
// Device and buffers
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

    pub(crate) fn handle(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub(crate) fn queue_handle(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    pub(crate) fn language_version(&self) -> MetalLanguageVersion {
        self.info.profile.language.value
    }

    pub(crate) fn max_threadgroup_bytes(&self) -> u64 {
        self.info.max_threadgroup_bytes
    }
}

impl Buffer {
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
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The executor over prepared invocations
// ---------------------------------------------------------------------------

/// The MSL float dtypes the compile probes admit. The probe input lists
/// are exactly these constants, so the dtype→MSL-type pairing is total
/// over the probe vocabulary by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeDtype {
    F16,
    BF16,
    F32,
}

impl ProbeDtype {
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

const PROBE_DTYPES: [ProbeDtype; 3] = [ProbeDtype::F16, ProbeDtype::BF16, ProbeDtype::F32];

fn probe_scalar_dtypes(
    device: &ProtocolObject<dyn MTLDevice>,
    language: MetalLanguageVersion,
) -> Vec<DType> {
    PROBE_DTYPES
        .into_iter()
        .filter(|&dtype| {
            let ty = dtype.msl();
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
        .map(ProbeDtype::dtype)
        .collect()
}

fn probe_matrix_signatures(
    device: &ProtocolObject<dyn MTLDevice>,
    language: MetalLanguageVersion,
) -> (Vec<DType>, Vec<crate::target::MatrixCombination>) {
    let transfer_source = |dtype: ProbeDtype| {
        let ty = dtype.msl();
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void \
             seismic_matrix_transfer_probe(device {ty}* a [[buffer(0)]], \
             device {ty}* c [[buffer(1)]]) {{ simdgroup_matrix<{ty}, 8, 8> f; \
             simdgroup_load(f, a, 8); simdgroup_load(f, a, 8, ulong2(0, 0), true); \
             simdgroup_store(f, c, 8); }}\n"
        )
    };
    let source = |accumulator: ProbeDtype, left: ProbeDtype, right: ProbeDtype| {
        let (accumulator, left, right) = (accumulator.msl(), left.msl(), right.msl());
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
    let matrix_dtypes = PROBE_DTYPES
        .into_iter()
        .filter(|&dtype| {
            compile_probe(
                device,
                language,
                "seismic_matrix_transfer_probe",
                &transfer_source(dtype),
            )
        })
        .map(ProbeDtype::dtype)
        .collect::<Vec<_>>();
    let mut combinations = Vec::new();
    for accumulator in PROBE_DTYPES {
        for left in PROBE_DTYPES {
            for right in PROBE_DTYPES {
                if compile_probe(
                    device,
                    language,
                    "seismic_matrix_probe",
                    &source(accumulator, left, right),
                ) {
                    combinations.push(crate::target::MatrixCombination {
                        accumulator: accumulator.dtype(),
                        left: left.dtype(),
                        right: right.dtype(),
                    });
                }
            }
        }
    }
    (matrix_dtypes, combinations)
}

/// One owned result plane of an invocation, in ABI result order.
#[derive(Clone)]
pub struct OwnedResultPlane {
    pub path: Vec<u32>,
    pub plane: String,
    pub buffer: Buffer,
}

/// One decoded compiler-owned scalar result. Range endpoints share a
/// semantic path and remain explicitly distinguished by `endpoint`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarResult {
    pub path: ValuePath,
    pub endpoint: Option<RangeEndpoint>,
    pub dtype: DType,
    pub value: f64,
}

/// The execution outcome: the invocation's result planes (the kernels'
/// outputs are already resident in them) and the decoded scalar results.
#[derive(Clone, Default)]
pub struct ExecutionOutcome {
    pub planes: Vec<OwnedResultPlane>,
    pub scalars: Vec<ScalarResult>,
    pub launches: Vec<seismic_realization::physical::LaunchExecution>,
}

/// The R1 invocation surface the Metal executor consumes. `PreparedInvocation`
/// (seismic-runtime, package R1, through its sealed `MetalSurface` adapter)
/// implements it; the dependency direction is fixed (seismic-runtime depends
/// on this crate), so the backend names the accessors it needs and R1
/// provides them. Frozen accessor set of the B1-Metal lane report:
///
/// - `metal_artifact` — the Metal arm of the sealed `CompiledArtifact`;
/// - `metal_buffer` — one validated buffer's Metal handle, by contract slot
///   order (validated device/type/size/alias facts by `prepare`);
/// - `values` — the invocation values `prepare` evaluated exactly once;
/// - `result_planes` — the owned result planes in ABI result order.
pub trait InvocationSurface {
    fn metal_artifact(&self) -> &crate::native::NativeArtifact;
    fn metal_buffer(&self, slot: seismic_realization::ids::BufferSlot) -> Option<&Buffer>;
    fn values(&self) -> &InvocationValues;
    fn result_planes(&self) -> &[OwnedResultPlane];
}

/// The Metal executor: executes one prepared invocation against its sealed
/// native artifact. Only `ExecutionFailure` outcomes: `SafetyViolation`
/// from the status block and retained guards, `ExternalFailure` for the
/// driver, allocation, and submission systems. It observes no absent
/// compiler-owned record: every launch, guard, storage, and copy it touches
/// is a folded dense record of the artifact.
pub struct Executor;

impl Executor {
    pub fn new() -> Self {
        Executor
    }

    pub fn execute(
        &self,
        invocation: &dyn InvocationSurface,
    ) -> Result<ExecutionOutcome, ExecutionFailure> {
        ExecutionContext::run(invocation)
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

fn external(stage: ExternalStage, detail: impl Into<String>) -> ExecutionFailure {
    ExecutionFailure::External(ExternalFailure {
        stage,
        detail: detail.into(),
    })
}

/// Whether evaluating one expression reads executor-produced state (the
/// producing launch must be complete first).
fn needs_flush(expression: &ExecutionExpr) -> bool {
    matches!(
        expression,
        ExecutionExpr::Guarded(_) | ExecutionExpr::ResultField(_)
    )
}

/// One in-flight command buffer: launches within a `Repeat` visit share it;
/// visits are committed and completed before a slot is rebound.
struct Submission {
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
}

impl Submission {
    fn open() -> Result<Self, ExecutionFailure> {
        Ok(Self {
            command: None,
            encoder: None,
        })
    }

    fn encoder(
        &mut self,
        context: &ExecutionContext<'_>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, ExecutionFailure> {
        if self.encoder.is_none() {
            let command = context.artifact.queue().commandBuffer().ok_or_else(|| {
                external(ExternalStage::Driver, "could not create a command buffer")
            })?;
            let encoder = command.computeCommandEncoder().ok_or_else(|| {
                external(ExternalStage::Driver, "could not create a compute encoder")
            })?;
            self.command = Some(command);
            self.encoder = Some(encoder);
        }
        match &self.encoder {
            Some(encoder) => Ok(encoder.clone()),
            None => Err(external(
                ExternalStage::Driver,
                "the compute encoder was not opened",
            )),
        }
    }

    fn flush(&mut self) -> Result<(), ExecutionFailure> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if let Some(command) = self.command.take() {
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return Err(external(
                    ExternalStage::Submission,
                    format!("command buffer failed: {}", error.localizedDescription()),
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

/// The execution environment: the sealed artifact, the validated
/// invocation surface, and the exact planned blocks allocated for this
/// execution.
struct ExecutionContext<'a> {
    artifact: &'a NativeArtifact,
    invocation: &'a dyn InvocationSurface,
    arena: Buffer,
    slots: Buffer,
    results: Buffer,
    status: Buffer,
}

impl<'a> ExecutionContext<'a> {
    fn run(invocation: &dyn InvocationSurface) -> Result<ExecutionOutcome, ExecutionFailure> {
        let artifact = invocation.metal_artifact();
        let device = Device::from_handles(artifact.device(), artifact.queue())?;
        let arena = device
            .buffer(artifact.resources().arena_bytes as usize)
            .map_err(|error| external(ExternalStage::Allocation, error))?;
        let slot_bytes = artifact.resources().scalar_slots as usize * 4;
        let slots = device
            .buffer(slot_bytes)
            .map_err(|error| external(ExternalStage::Allocation, error))?;
        if slot_bytes > 0 {
            slots.write(&vec![0u8; slot_bytes]);
        }
        let result_bytes = artifact.resources().result_bytes as usize;
        let results = device
            .buffer(result_bytes)
            .map_err(|error| external(ExternalStage::Allocation, error))?;
        if result_bytes > 0 {
            results.write(&vec![0u8; result_bytes]);
        }
        let status_bytes = artifact.resources().status_bytes as usize;
        let status = device
            .buffer(status_bytes)
            .map_err(|error| external(ExternalStage::Allocation, error))?;
        if status_bytes > 0 {
            status.write(&vec![0u8; status_bytes]);
        }
        let context = ExecutionContext {
            artifact,
            invocation,
            arena,
            slots,
            results,
            status,
        };
        let mut submission = Submission::open()?;
        let mut dispatches = 0usize;
        let mut launches = Vec::new();
        let walked = context.walk(artifact.steps(), &mut submission, &mut dispatches, &mut launches);
        let flushed = submission.flush();
        walked?;
        flushed?;
        context.report_status()?;
        Ok(ExecutionOutcome {
            planes: invocation.result_planes().to_vec(),
            scalars: context.decode_scalar_results(),
            launches,
        })
    }

    /// The native handle of one validated buffer; every buffer of a Metal
    /// artifact's invocation is a Metal buffer (validated by `prepare`
    /// against the artifact's contract).
    fn metal_buffer(
        &self,
        slot: BufferSlot,
    ) -> Result<&ProtocolObject<dyn MTLBuffer>, ExecutionFailure> {
        self.invocation
            .metal_buffer(slot)
            .map(|buffer| buffer.raw())
            .ok_or_else(|| {
                external(
                    ExternalStage::Driver,
                    format!(
                        "buffer slot {} of a Metal invocation is not a Metal buffer",
                        slot.index()
                    ),
                )
            })
    }

    // -- retained expression and scalar evaluation -------------------------

    /// Evaluate one retained execution expression against the validated
    /// invocation values, the executor slots, the result block, and the
    /// folded native facts. Every partial operation consults its
    /// dominating guard first (A1's four-closure `evaluate`: executor
    /// slots, result fields, guards); a failed guard is the
    /// `SafetyViolation` returned.
    fn eval(&self, expression: &ExecutionExpr) -> Result<u64, ExecutionFailure> {
        match expression {
            ExecutionExpr::Invocation(id) => Ok(self.invocation.values().derived[*id]),
            ExecutionExpr::Guarded(guarded) => {
                let mut guards =
                    |guard: GuardIx, kind: SafetyKind| self.check_guard(guard, kind);
                guarded
                    .evaluate(
                        self.invocation.values(),
                        &|slot| u64::from(self.slot_word(slot)),
                        &|field| self.result_word(field),
                        &mut guards,
                    )
                    .map_err(ExecutionFailure::Safety)
            }
            ExecutionExpr::NativeFact(fact) => Ok(self.artifact.native_fact(*fact)),
            ExecutionExpr::ResultField(field) => Ok(self.result_word(*field)),
        }
    }

    /// One retained executor guard: its status field (the artifact's dense
    /// guard→status pairing) must be clean — the dominating launch's
    /// checks wrote nothing. The failing site is the schedule guard.
    fn check_guard(&self, guard: GuardIx, kind: SafetyKind) -> Result<(), SafetyViolation> {
        let field = self.artifact.status_field_of_guard(guard);
        if self.status_word(field) != 0 {
            let obligation = self.artifact.status_fields()[field.index()].obligation.clone();
            Err(SafetyViolation {
                source: SafetyViolationSource::Guard(guard),
                obligation,
                kind,
            })
        } else {
            Ok(())
        }
    }

    /// The status words of the root status block, dense by
    /// `StatusFieldIx` (the block is allocated `fields × 4` bytes).
    fn status_words(&self) -> Vec<u32> {
        let bytes = self.status.read(self.artifact.resources().status_bytes as usize);
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    fn status_word(&self, field: StatusFieldIx) -> u32 {
        self.status_words()[field.index()]
    }

    /// The result words of the compiler-owned result block, dense by
    /// `ResultFieldIx` (the block is allocated `fields × 8` bytes).
    fn result_words(&self) -> Vec<u64> {
        let bytes = self.results.read(self.artifact.resources().result_bytes as usize);
        bytes
            .chunks_exact(8)
            .map(|word| {
                u64::from_le_bytes([
                    word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
                ])
            })
            .collect()
    }

    fn result_word(&self, field: ResultFieldIx) -> u64 {
        self.result_words()[field.index()]
    }

    /// The executor slot words, dense by `ScalarSlotIx` (the block is
    /// allocated `slots × 4` bytes).
    fn slot_words(&self) -> Vec<u32> {
        let bytes = self.slots.read(self.artifact.resources().scalar_slots as usize * 4);
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    fn slot_word(&self, slot: ScalarSlotIx) -> u32 {
        self.slot_words()[slot.index()]
    }

    fn write_slot_word(&self, slot: ScalarSlotIx, word: u32) {
        let mut words = self.slot_words();
        words[slot.index()] = word;
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.slots.write(&bytes);
    }

    fn write_result_word(&self, field: ResultFieldIx, word: u64) {
        let mut words = self.result_words();
        words[field.index()] = word;
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        self.results.write(&bytes);
    }

    /// The current value of one scalar source.
    fn scalar_value(&self, source: &ScalarSource) -> Result<u64, ExecutionFailure> {
        match source {
            ScalarSource::Abi(slot) => Ok(self.invocation.values().scalars[*slot].bits),
            ScalarSource::Executor(slot) => Ok(u64::from(self.slot_word(*slot))),
            ScalarSource::Invocation(id) => Ok(self.invocation.values().derived[*id]),
            ScalarSource::Result(field) => Ok(self.result_word(*field)),
        }
    }

    // -- the native tree ----------------------------------------------------

    fn walk(
        &self,
        steps: &[NativeStep],
        submission: &mut Submission,
        dispatches: &mut usize,
        launches: &mut Vec<seismic_realization::physical::LaunchExecution>,
    ) -> Result<(), ExecutionFailure> {
        for step in steps {
            match step {
                NativeStep::Launch { launch } => {
                    self.launch(*launch, submission, dispatches, launches)?;
                }
                NativeStep::Guard {
                    guard,
                    obligation,
                    status,
                    predicate,
                } => {
                    // The guard's expressions may read executor slots or
                    // result fields; their producers must be complete.
                    submission.flush()?;
                    self.eval_guard(*guard, obligation, *status, predicate)?;
                }
                NativeStep::Call(children) => {
                    self.walk(children, submission, dispatches, launches)?;
                }
                NativeStep::If {
                    condition,
                    then_steps,
                    else_steps,
                    joins,
                } => {
                    submission.flush()?;
                    let (taken, side) = if self.scalar_value(condition)? != 0 {
                        (then_steps, 0)
                    } else {
                        (else_steps, 1)
                    };
                    self.walk(taken, submission, dispatches, launches)?;
                    for join in joins {
                        let source = if side == 0 {
                            &join.then_source
                        } else {
                            &join.else_source
                        };
                        self.copy(source, &join.destination)?;
                    }
                }
                NativeStep::Repeat {
                    start,
                    end,
                    bound: _,
                    binder,
                    body,
                    carries,
                } => {
                    submission.flush()?;
                    let start = self.eval(start)?;
                    let end = self.eval(end)?;
                    for carry in carries {
                        self.copy(&carry.initial, &carry.current)?;
                    }
                    for value in start..end {
                        let word = u32::try_from(value).map_err(|_| {
                            external(
                                ExternalStage::Driver,
                                "a repeat binder value exceeds the executor slot-word domain",
                            )
                        })?;
                        self.write_slot_word(*binder, word);
                        self.walk(body, submission, dispatches, launches)?;
                        submission.flush()?;
                        for carry in carries {
                            self.copy(&carry.update, &carry.current)?;
                        }
                    }
                    for carry in carries {
                        self.copy_readback(&carry.current, &carry.result)?;
                    }
                }
                NativeStep::Fill { offset, bytes } => {
                    submission.flush()?;
                    self.fill(*offset, *bytes)?;
                }
            }
        }
        Ok(())
    }

    /// One retained executor guard predicate over retained expressions.
    /// The step carries its own status field, so no lookup is needed.
    fn eval_guard(
        &self,
        guard: GuardIx,
        obligation: &ObligationRef,
        status: StatusFieldIx,
        predicate: &GuardPredicate,
    ) -> Result<(), ExecutionFailure> {
        let kind = match predicate {
            GuardPredicate::ProductFits { .. } => SafetyKind::ShapeProductFits,
            GuardPredicate::ExtentPositive { .. } => SafetyKind::ExtentPositive,
            GuardPredicate::RangeOrdered { .. } => SafetyKind::RangeInBounds,
        };
        let holds = match predicate {
            GuardPredicate::ProductFits { factors, bits } => {
                let mut product = 1u128;
                for factor in factors {
                    product = product.saturating_mul(u128::from(self.eval(factor)?));
                }
                *bits >= 128 || product < (1u128 << *bits)
            }
            GuardPredicate::ExtentPositive { extent } => self.eval(extent)? > 0,
            GuardPredicate::RangeOrdered { start, end, bound } => {
                self.eval(start)? <= self.eval(end)? && self.eval(end)? <= self.eval(bound)?
            }
        };
        if holds {
            Ok(())
        } else {
            let _ = status;
            Err(ExecutionFailure::Safety(SafetyViolation {
                source: SafetyViolationSource::Guard(guard),
                obligation: obligation.clone(),
                kind,
            }))
        }
    }

    /// One launch: evaluate its geometry, bind its direct program, and
    /// submit. Zero work skips submission. The sealed geometry is
    /// authoritative for every traversal (serialized-policy launches are
    /// sealed `[1, 1, 1]` with one participant).
    fn launch(
        &self,
        launch: LaunchIx,
        submission: &mut Submission,
        dispatches: &mut usize,
        launches: &mut Vec<seismic_realization::physical::LaunchExecution>,
    ) -> Result<(), ExecutionFailure> {
        let started = std::time::Instant::now();
        let native = self.artifact.launch(launch);
        // Geometry that reads executor slots or result fields needs its
        // producers complete before host evaluation.
        if needs_flush(native.work_items())
            || needs_flush(native.participants())
            || native.workgroups().iter().any(needs_flush)
        {
            submission.flush()?;
        }
        let work_items = self.eval(native.work_items())?;
        if work_items == 0 {
            return Ok(());
        }
        let workgroups = [
            self.eval(&native.workgroups()[0])?,
            self.eval(&native.workgroups()[1])?,
            self.eval(&native.workgroups()[2])?,
        ];
        let participants = self.eval(native.participants())?;
        let encoder = submission.encoder(self)?;
        encoder.setComputePipelineState(native.state());
        for binding in native.bindings() {
            match binding {
                NativeBinding::Storage { index, target } => match target {
                    StorageTarget::Abi { slot } => {
                        let buffer = self.metal_buffer(*slot)?;
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(buffer), 0, *index as usize)
                        }
                    }
                    StorageTarget::Arena { offset } => {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(self.arena.raw()),
                                *offset as usize,
                                *index as usize,
                            )
                        }
                    }
                    // A staged storage never enters a device binding group
                    // (the renderer declares its array natively); a staged
                    // entry here contradicts the folded binding program.
                    StorageTarget::Staged => {
                        return Err(external(
                            ExternalStage::Driver,
                            format!(
                                "launch {} binds a staged storage at argument {}",
                                launch.index(),
                                index
                            ),
                        ))
                    }
                },
                NativeBinding::Scalar { index, source, dtype } => {
                    let value = self.scalar_value(source)?;
                    let bytes = scalar_bytes(value, *dtype);
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            binding_ptr(&bytes),
                            bytes.len(),
                            *index as usize,
                        )
                    }
                }
                NativeBinding::Total { index } => {
                    let bytes = work_items.to_le_bytes();
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            binding_ptr(&bytes),
                            bytes.len(),
                            *index as usize,
                        )
                    }
                }
                NativeBinding::Extent { index, value } => {
                    let evaluated = self.eval(value)?;
                    let bytes = evaluated.to_le_bytes();
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            binding_ptr(&bytes),
                            bytes.len(),
                            *index as usize,
                        )
                    }
                }
                NativeBinding::Slots { index } => unsafe {
                    encoder.setBuffer_offset_atIndex(Some(self.slots.raw()), 0, *index as usize)
                },
                NativeBinding::Results { index } => unsafe {
                    encoder.setBuffer_offset_atIndex(Some(self.results.raw()), 0, *index as usize)
                },
                NativeBinding::Status { index } => unsafe {
                    encoder.setBuffer_offset_atIndex(Some(self.status.raw()), 0, *index as usize)
                },
            }
        }
        if *dispatches > 0 {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        let grid = MTLSize {
            width: workgroups[0] as usize,
            height: workgroups[1] as usize,
            depth: workgroups[2] as usize,
        };
        let group = MTLSize {
            width: participants as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
        *dispatches += 1;
        // Dispatch-time geometry: emitted where the executor knows it, so a
        // running launch is observable before it completes.
        eprintln!(
            "magnitude-engine: dispatch launch {} work_items={work_items} \
             participants={participants} workgroups={:?}",
            launch.index(),
            workgroups
        );
        launches.push(seismic_realization::physical::LaunchExecution {
            launch,
            work_items,
            participants,
            workgroups,
            seconds: started.elapsed().as_secs_f64(),
        });
        Ok(())
    }

    /// Copy one folded source into one folded destination. Scalar sources
    /// are word values; tensor sources are arena slices (equal-capacity
    /// storages for the same value, so the smaller capacity covers every
    /// plane either side addresses; the same storage is a no-op).
    fn copy(
        &self,
        source: &CopySource,
        destination: &CopyDestination,
    ) -> Result<(), ExecutionFailure> {
        match (source, destination) {
            (
                CopySource::Scalar { source },
                CopyDestination::Scalar { destination },
            ) => {
                let value = self.scalar_value(source)?;
                match destination {
                    ScalarDestination::Slot(slot) => {
                        self.write_slot_word(*slot, value as u32);
                        Ok(())
                    }
                    ScalarDestination::Result(field) => {
                        self.write_result_word(*field, value);
                        Ok(())
                    }
                }
            }
            (
                CopySource::Tensor { storage, bytes },
                CopyDestination::Tensor {
                    storage: to,
                    bytes: to_bytes,
                },
            ) => {
                if storage == to {
                    return Ok(());
                }
                self.copy_arena(storage, to, (*bytes).min(*to_bytes))
            }
            // The folded records pair kinds at assembly; a mismatched pair
            // contradicts the fold.
            (
                CopySource::Scalar { .. } | CopySource::Tensor { .. },
                CopyDestination::Scalar { .. } | CopyDestination::Tensor { .. },
            ) => Err(external(
                ExternalStage::Driver,
                "a folded join or carry pairs a scalar with a tensor",
            )),
        }
    }

    /// `result := current` after the loop: the current destination is read
    /// back and written to the result destination of the same kind.
    fn copy_readback(
        &self,
        current: &CopyDestination,
        result: &CopyDestination,
    ) -> Result<(), ExecutionFailure> {
        match (current, result) {
            (
                CopyDestination::Scalar { destination: from },
                CopyDestination::Scalar { destination: to },
            ) => {
                let value = match from {
                    ScalarDestination::Slot(slot) => u64::from(self.slot_word(*slot)),
                    ScalarDestination::Result(field) => self.result_word(*field),
                };
                match to {
                    ScalarDestination::Slot(slot) => {
                        self.write_slot_word(*slot, value as u32);
                        Ok(())
                    }
                    ScalarDestination::Result(field) => {
                        self.write_result_word(*field, value);
                        Ok(())
                    }
                }
            }
            (
                CopyDestination::Tensor { storage, bytes },
                CopyDestination::Tensor {
                    storage: to,
                    bytes: to_bytes,
                },
            ) => {
                if storage == to {
                    return Ok(());
                }
                self.copy_arena(storage, to, (*bytes).min(*to_bytes))
            }
            (
                CopyDestination::Scalar { .. } | CopyDestination::Tensor { .. },
                CopyDestination::Scalar { .. } | CopyDestination::Tensor { .. },
            ) => Err(external(
                ExternalStage::Driver,
                "a folded carry pairs a scalar with a tensor",
            )),
        }
    }

    /// A device copy between two arena placements (shared memory: a host
    /// copy after the producing command buffer completed).
    fn copy_arena(
        &self,
        from: &StorageIx,
        to: &StorageIx,
        bytes: u64,
    ) -> Result<(), ExecutionFailure> {
        let (from_offset, to_offset) = match (
            self.artifact.storage_target(*from),
            self.artifact.storage_target(*to),
        ) {
            (StorageTarget::Arena { offset: from }, StorageTarget::Arena { offset: to }) => {
                (from, to)
            }
            _ => {
                return Err(external(
                    ExternalStage::Driver,
                    "a tensor join or carry crosses a non-arena residence",
                ))
            }
        };
        let arena_bytes = self.artifact.resources().arena_bytes as usize;
        let mut contents = self.arena.read(arena_bytes);
        let moved: Vec<u8> = contents
            [from_offset as usize..from_offset as usize + bytes as usize]
            .to_vec();
        contents[to_offset as usize..to_offset as usize + bytes as usize].copy_from_slice(&moved);
        self.arena.write(&contents);
        Ok(())
    }

    /// One host-side fill: zeroes `bytes` bytes of the arena at `offset`
    /// (the pull-counter reset before the launch that claims from it).
    fn fill(&self, offset: u64, bytes: u64) -> Result<(), ExecutionFailure> {
        let arena_bytes = self.artifact.resources().arena_bytes as usize;
        let mut contents = self.arena.read(arena_bytes);
        for byte in &mut contents[offset as usize..(offset + bytes) as usize] {
            *byte = 0;
        }
        self.arena.write(&contents);
        Ok(())
    }

    /// Report the first failing status field after synchronous completion.
    /// A field with a dominating schedule guard is that guard's site; a
    /// kernel-`Check` field is its own site.
    fn report_status(&self) -> Result<(), ExecutionFailure> {
        for (index, field) in self.artifact.status_fields().iter().enumerate() {
            let status = StatusFieldIx::from_index(index);
            if self.status_word(status) != 0 {
                let source = match self.artifact.status_guard(status) {
                    Some(guard) => SafetyViolationSource::Guard(guard),
                    None => SafetyViolationSource::KernelCheck(status),
                };
                return Err(ExecutionFailure::Safety(SafetyViolation {
                    source,
                    obligation: field.obligation.clone(),
                    kind: field.kind,
                }));
            }
        }
        Ok(())
    }

    /// Decode the compiler-owned scalar results from the result block.
    fn decode_scalar_results(&self) -> Vec<ScalarResult> {
        self.artifact
            .result_fields()
            .iter()
            .map(|field| ScalarResult {
                path: field.path.clone(),
                endpoint: field.endpoint,
                dtype: field.dtype,
                value: decode_result_word(self.result_word(field.index), field.dtype),
            })
            .collect()
    }
}

/// The raw pointer of a by-value binding's byte slice. A slice's pointer
/// is non-null by construction; the cast re-expresses the element type.
fn binding_ptr(bytes: &[u8]) -> NonNull<std::ffi::c_void> {
    match NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void) {
        Some(pointer) => pointer,
        None => unreachable!("a slice pointer is non-null by construction"),
    }
}

/// The ABI bytes of one scalar value in its representation.
fn scalar_bytes(value: u64, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => (value as u32).to_le_bytes().to_vec(),
        DType::F16 | DType::BF16 => (value as u16).to_le_bytes().to_vec(),
        DType::Bool => vec![u8::from(value & 1 != 0)],
    }
}

fn decode_result_word(word: u64, dtype: DType) -> f64 {
    match dtype {
        DType::Bool => f64::from((word & 1) as u8),
        DType::I32 => (word as u32 as i32) as f64,
        DType::U32 => (word as u32) as f64,
        DType::F32 => f32::from_bits(word as u32) as f64,
        DType::F16 => {
            let bits = word as u16;
            let sign = ((bits >> 15) & 1) as u32;
            let exponent = ((bits >> 10) & 0x1f) as u32;
            let fraction = (bits & 0x03ff) as u32;
            let f32_bits = if exponent == 0 {
                if fraction == 0 {
                    sign << 31
                } else {
                    let shift = fraction.leading_zeros() - 21;
                    (sign << 31)
                        | ((127 - 15 - shift) << 23)
                        | ((fraction << (shift + 1) & 0x03ff) << 13)
                }
            } else if exponent == 0x1f {
                (sign << 31) | (0xff << 23) | (fraction << 13)
            } else {
                (sign << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13)
            };
            f32::from_bits(f32_bits) as f64
        }
        DType::BF16 => f32::from_bits((word as u32 & 0xffff) << 16) as f64,
    }
}

impl Device {
    /// A device facade over the artifact-owned handles (the artifact keeps
    /// them alive; the facade only allocates this execution's blocks).
    fn from_handles(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Device, ExecutionFailure> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> = device.into();
        let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = queue.into();
        Ok(Device {
            info: DeviceInfo {
                name: String::new(),
                architecture_name: String::new(),
                registry_id: 0,
                unified_memory: true,
                max_threads_per_threadgroup: device.maxThreadsPerThreadgroup().width as u64,
                max_threadgroup_bytes: device.maxThreadgroupMemoryLength() as u64,
                max_buffer_bytes: device.maxBufferLength() as u64,
                cores: None,
                recommended_working_set_bytes: device.recommendedMaxWorkingSetSize(),
                profile: MetalProfile {
                    families: Observed::new(Vec::new(), Provenance::DeviceQuery),
                    language: Observed::new(
                        MetalLanguageVersion::V3_2,
                        Provenance::ConservativeAssumption,
                    ),
                    device_limits_provenance: Provenance::DeviceQuery,
                    private_storage_budget_bytes: Observed::new(
                        crate::target::CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
                        Provenance::ConservativeAssumption,
                    ),
                    scalar_dtypes: Observed::new(Vec::new(), Provenance::DeviceQuery),
                    matrix_dtypes: Observed::new(Vec::new(), Provenance::DeviceQuery),
                    matrix_combinations: Observed::new(Vec::new(), Provenance::DeviceQuery),
                },
            },
            device,
            queue,
            identity: std::rc::Rc::new(()),
        })
    }
}
