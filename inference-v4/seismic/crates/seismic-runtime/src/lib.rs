//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Native executables are produced only by the unified logical ->
//! family -> solve -> resolve -> encode pipeline.
//!
//! Metal, the CPU and CUDA are the backends on the structured pipeline. Each
//! backend runtime validates the retained root ABI (sizes, alignment, alias
//! rules over actual byte ranges), allocates the exact planned
//! arena/results/status blocks, binds retained IDs, evaluates the retained
//! execution expressions, skips zero-work launches, and interprets the
//! retained structured execution tree. This crate is the device facade over
//! those embedded runtimes: one `Device`/`Buffer`/`Kernel` surface with no
//! retry, candidate, or strategy API.
//!
//! The CUDA driver is loaded at run time, so opening a CUDA device fails
//! cleanly on a host without one. `BackendDevice`, `Storage`, `Executable`
//! and `DeviceFacts` are closed sums: a backend rejoins by adding one variant
//! to each and one arm to every `match` on them (all of them are in this file
//! and in `plan.rs`). The Metal variants exist on macOS only; the CPU and
//! CUDA variants exist everywhere.

mod error;
pub use error::Error;
pub mod memory;
pub mod plan;
pub use seismic_compiler::pipeline::Workload;
use seismic_compiler::pipeline::{self, Compiled};
pub use seismic_compiler::planning::{Budget, NumericalEvidence};
use seismic_lang::abi::ScalarParameter;
use seismic_lang::sir::Program;
use seismic_realization::executable::{
    self as realization, AbiRole, BufferBindingId, ExecutableDialect, ExecutionExpr,
    ResolvedLaunchId, ResolvedPlan, ResolvedSchedule, ResolvedStep,
};
use std::rc::Rc;

#[derive(Clone)]
enum BackendDevice {
    #[cfg(target_os = "macos")]
    Metal(Rc<seismic_metal::runtime::Device>),
    Cpu(Rc<CpuDevice>),
    /// A private, thread-affine driver context.
    Cuda(Rc<seismic_cuda::Device>),
}
/// The worker pool of one CPU device. Execution is synchronous, so one launch runs at a time.
struct CpuDevice {
    workers: std::cell::RefCell<seismic_cpu::Workers>,
}
/// Facts of the host a CPU device executes on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuInfo {
    /// Worker threads that execute the pieces of one independent domain.
    pub workers: u64,
}
#[derive(Clone)]
pub struct Device(BackendDevice, Rc<memory::Domain>);
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceFacts {
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::runtime::DeviceInfo),
    Cpu(CpuInfo),
    Cuda(seismic_cuda::DeviceInfo),
}
#[derive(Clone)]
enum Storage {
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::runtime::Buffer),
    /// Host memory.
    Cpu(seismic_cpu::Buffer),
    Cuda(seismic_cuda::Buffer),
}
#[derive(Clone)]
pub struct Buffer(Storage, Rc<Allocation>, usize);
struct Allocation {
    bytes: usize,
    _charge: memory::Charge,
}

/// One public buffer binding of the retained root ABI, in ABI buffer order.
/// The dense plane is presented as `""` for source-level binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub parameter: String,
    pub plane: String,
    pub role: BindingRole,
    pub binding: BufferBindingId,
    pub bytes: usize,
    pub alignment: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingRole {
    Parameter,
    Result { path: Vec<u32> },
}

/// The retained root ABI in bindable form: public buffers in ABI order,
/// result buffer bindings in ABI result order, scalar fields in encoding
/// order, and the alias rules over binding ids.
pub(crate) struct Binding {
    buffers: Vec<BufferSpec>,
    result_bindings: Vec<BufferBindingId>,
    scalars: Vec<ScalarParameter>,
    alias_rules: Vec<realization::AliasRule>,
}

enum Executable {
    #[cfg(target_os = "macos")]
    Metal {
        device: Rc<seismic_metal::runtime::Device>,
        pipeline: Box<seismic_metal::runtime::Pipeline>,
    },
    Cpu {
        device: Rc<CpuDevice>,
        kernel: Box<seismic_cpu::Kernel>,
    },
    Cuda {
        /// Keeps the driver context alive for the sequence's modules and buffers.
        _device: Rc<seismic_cuda::Device>,
        sequence: Box<seismic_cuda::PhysicalSequence>,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceTimingScope {
    /// Complete Metal command buffer interval, including its encoded dependencies.
    CommandBuffer,
    /// Wall time of the launches of a CPU kernel on the worker pool, after binding.
    CpuLaunches,
    /// Sum of the per-launch CUDA event intervals, excluding host gaps.
    CudaLaunches,
}
/// Distinct timing boundaries.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionObservation {
    /// Binding, submission, synchronous completion and status validation.
    pub host_seconds: f64,
    /// Native GPU command time; excludes host binding and readback.
    pub device_seconds: Option<f64>,
    pub device_scope: Option<DeviceTimingScope>,
}

/// One native dispatch of a profiled execution, with its selected launch geometry.
#[derive(Clone, Debug)]
pub struct DispatchProfile {
    pub launch: usize,
    pub kernel: String,
    pub threadgroups: u64,
    pub threads_per_threadgroup: u64,
    /// Native stage-timestamp interval of this dispatch's encoder.
    pub device_seconds: f64,
}

/// One selected implementation alternative at a call occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlternativeSelection {
    pub logical_alternative: u32,
    pub physical_alternative: u32,
}
/// The solver's implementation assignment, retained for reporting.
#[derive(Clone, Debug, Default)]
pub struct PlanAssignment {
    selections: Vec<(u32, AlternativeSelection)>,
}
impl PlanAssignment {
    pub fn selections(&self) -> &[(u32, AlternativeSelection)] {
        &self.selections
    }
}

/// Auditable report projected directly from the resolved plan. It is not a
/// second selection representation.
#[derive(Clone, Debug)]
pub struct Selection {
    pub entry: String,
    pub assignment: PlanAssignment,
    pub estimated_cost: i64,
    pub optimal: bool,
    pub resources: Vec<LaunchResources>,
    pub capability_fingerprint: String,
    pub numerical_assessment: seismic_lang::precision::NumericalAssessment,
    /// Identity of the evidence the assessment carries: the resolved
    /// toolchain fingerprint and evidence class.
    pub numerical_evidence_identity: Option<String>,
    /// Static shape and element bindings of the compiled specialization (its identity).
    pub shapes: Vec<(String, i64)>,
    pub elements: Vec<(String, String)>,
    /// Wall time of the single specialization/planning/emission/native pipeline.
    pub compile: std::time::Duration,
}

#[derive(Clone, Debug)]
pub struct LaunchResources {
    pub launch: ResolvedLaunchId,
    /// Grid geometry when it is statically determined; geometry that depends
    /// on runtime extents or scalars stays symbolic.
    pub workgroups: Option<[u64; 3]>,
    pub device_bytes: u64,
    pub workgroup_bytes: u64,
    pub private_bytes_per_participant: u64,
    pub bindings: u64,
    pub threads_per_group: Option<u64>,
}

pub struct Kernel {
    executable: Executable,
    binding: Binding,
    selection: Selection,
}

/// Evaluate a retained execution expression statically. Runtime-extent
/// references resolve transitively; scalar references are unknown here.
fn eval_static(
    expr: &ExecutionExpr,
    extents: &dyn Fn(seismic_lang::types::RuntimeExtentId) -> Option<ExecutionExpr>,
) -> Option<u64> {
    Some(match expr {
        ExecutionExpr::Const(value) => *value,
        ExecutionExpr::Extent(id) => {
            let resolved = extents(*id)?;
            eval_static(&resolved, extents)?
        }
        ExecutionExpr::AbiScalar { .. } | ExecutionExpr::ExecutorScalar(_) => return None,
        ExecutionExpr::Add(left, right) => {
            eval_static(left, extents)?.checked_add(eval_static(right, extents)?)?
        }
        ExecutionExpr::Sub(left, right) => {
            eval_static(left, extents)?.checked_sub(eval_static(right, extents)?)?
        }
        ExecutionExpr::Mul(left, right) => {
            eval_static(left, extents)?.checked_mul(eval_static(right, extents)?)?
        }
        ExecutionExpr::CeilDiv(left, right) => {
            eval_static(left, extents)?.div_ceil(eval_static(right, extents)?)
        }
        ExecutionExpr::Div(left, right) => {
            eval_static(left, extents)?.checked_div(eval_static(right, extents)?)?
        }
        ExecutionExpr::Rem(left, right) => {
            eval_static(left, extents)?.checked_rem(eval_static(right, extents)?)?
        }
        ExecutionExpr::Min(left, right) => {
            eval_static(left, extents)?.min(eval_static(right, extents)?)
        }
    })
}

/// Collect the retained root ABI in bindable form. Parameter names come from
/// the entry interface: ABI parameter ordinals are interface param ordinals.
fn binding<D: ExecutableDialect, A>(compiled: &Compiled<D, A>) -> Result<Binding, String> {
    let abi = &compiled.physical.abi;
    let interface = compiled
        .logical
        .choice(compiled.logical.entry_choice)
        .interface
        .clone();
    let mut buffers = Vec::with_capacity(abi.buffers.len());
    for buffer in &abi.buffers {
        let (parameter, role) = match buffer.role {
            AbiRole::Parameter { ordinal } => {
                let name = interface
                    .params
                    .get(ordinal as usize)
                    .map(|param| param.name.clone())
                    .ok_or_else(|| {
                        "compiler bug: an ABI parameter names no interface parameter".to_string()
                    })?;
                (name, BindingRole::Parameter)
            }
            AbiRole::Result => (
                buffer
                    .path
                    .0
                    .iter()
                    .map(|ordinal| ordinal.to_string())
                    .collect::<Vec<_>>()
                    .join("."),
                BindingRole::Result {
                    path: buffer.path.0.clone(),
                },
            ),
        };
        buffers.push(BufferSpec {
            parameter,
            plane: match buffer.plane.as_str() {
                "dense" => String::new(),
                plane => plane.to_string(),
            },
            role,
            binding: buffer.binding,
            bytes: usize::try_from(buffer.bytes)
                .map_err(|_| "an ABI buffer exceeds the host address range".to_string())?,
            alignment: usize::try_from(buffer.alignment)
                .map_err(|_| "an ABI alignment exceeds the host address range".to_string())?,
        });
    }
    let result_bindings = abi
        .results
        .iter()
        .filter_map(|result| match result {
            realization::ResultBinding::Buffer { binding, .. } => Some(*binding),
            realization::ResultBinding::Scalar { .. }
            | realization::ResultBinding::Range { .. } => None,
        })
        .collect();
    let scalars = abi
        .scalars
        .fields
        .iter()
        .map(|field| field.parameter.clone())
        .collect();
    Ok(Binding {
        buffers,
        result_bindings,
        scalars,
        alias_rules: abi.alias_rules.clone(),
    })
}

fn selection<D: ExecutableDialect, A>(
    compiled: &Compiled<D, A>,
    compile: std::time::Duration,
) -> Selection {
    fn resources<D: ExecutableDialect>(
        plan: &ResolvedPlan<D>,
        schedule: &ResolvedSchedule<D>,
        output: &mut Vec<LaunchResources>,
    ) {
        for step in schedule.steps.iter() {
            match step {
                ResolvedStep::Launch(launch) => {
                    let extent_of = |id: seismic_lang::types::RuntimeExtentId| {
                        plan.runtime_extents.get(id).cloned()
                    };
                    let workgroups = launch
                        .geometry
                        .workgroups
                        .iter()
                        .map(|expr| eval_static(expr, &extent_of))
                        .collect::<Option<Vec<_>>>()
                        .map(|axes| [axes[0], axes[1], axes[2]]);
                    let threads_per_group = launch
                        .geometry
                        .participants_per_workgroup
                        .iter()
                        .map(|expr| eval_static(expr, &extent_of))
                        .product();
                    output.push(LaunchResources {
                        launch: launch.id,
                        workgroups,
                        device_bytes: plan.internal_arena.bytes,
                        workgroup_bytes: launch.kernel.resources.workgroup_bytes,
                        private_bytes_per_participant: launch
                            .kernel
                            .resources
                            .private_bytes_per_participant,
                        bindings: launch.bindings.len() as u64,
                        threads_per_group,
                    });
                }
                ResolvedStep::Call(call) => resources(plan, &call.body.schedule, output),
                ResolvedStep::If(if_step) => {
                    resources(plan, &if_step.then_schedule, output);
                    resources(plan, &if_step.else_schedule, output);
                }
                ResolvedStep::Repeat(repeat) => resources(plan, &repeat.body, output),
            }
        }
    }
    let physical = &compiled.physical;
    let mut launch_resources = Vec::new();
    resources(physical, &physical.entry.schedule, &mut launch_resources);
    Selection {
        entry: compiled.logical.entry.clone(),
        assignment: PlanAssignment {
            selections: physical
                .identity
                .selections
                .iter()
                .map(|(choice, selected)| {
                    (
                        choice.0,
                        AlternativeSelection {
                            logical_alternative: selected.0,
                            physical_alternative: selected.1,
                        },
                    )
                })
                .collect(),
        },
        estimated_cost: i64::try_from(physical.estimated_cost).unwrap_or(i64::MAX),
        optimal: physical.optimal,
        resources: launch_resources,
        capability_fingerprint: compiled.logical.target.capability_fingerprint.clone(),
        numerical_assessment: physical.numerical.clone(),
        numerical_evidence_identity: Some(format!(
            "{}:{:?}",
            physical.identity.toolchain_fingerprint, physical.numerical.evidence
        )),
        shapes: compiled
            .logical
            .shapes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect(),
        elements: compiled
            .logical
            .elements
            .iter()
            .map(|(name, element)| (name.clone(), element.to_string()))
            .collect(),
        compile,
    }
}
impl Device {
    pub fn facts(&self) -> DeviceFacts {
        match &self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => DeviceFacts::Metal(device.info()),
            BackendDevice::Cpu(device) => DeviceFacts::Cpu(CpuInfo {
                workers: device.workers.borrow().count() as u64,
            }),
            BackendDevice::Cuda(device) => DeviceFacts::Cuda(device.info.clone()),
        }
    }
    /// The host CPU: one worker thread per unit of available parallelism, host memory buffers.
    pub fn cpu() -> Result<Self, String> {
        Ok(Self(
            BackendDevice::Cpu(Rc::new(CpuDevice {
                workers: std::cell::RefCell::new(seismic_cpu::Workers::host()?),
            })),
            Rc::default(),
        ))
    }
    /// CUDA device zero. Fails with the driver's reason on a host without a CUDA driver.
    pub fn cuda() -> Result<Self, String> {
        Ok(Self(
            BackendDevice::Cuda(Rc::new(
                seismic_cuda::Device::open(0).map_err(|e| format!("CUDA device: {e}"))?,
            )),
            Rc::default(),
        ))
    }
    /// Open the device a target name denotes: `metal`, `cpu` or `cuda`.
    pub fn open(target: &str) -> Result<Self, String> {
        match target {
            #[cfg(target_os = "macos")]
            "metal" => Self::metal(),
            "cpu" => Self::cpu(),
            "cuda" => Self::cuda(),
            #[cfg(not(target_os = "macos"))]
            "metal" => Err("the Metal device exists on macOS only".into()),
            other => Err(format!(
                "unknown device `{other}`; expected `metal`, `cpu` or `cuda`"
            )),
        }
    }
    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, String> {
        Ok(Self(
            BackendDevice::Metal(Rc::new(seismic_metal::runtime::Device::open()?)),
            Rc::default(),
        ))
    }
    pub fn backend(&self) -> &'static str {
        match self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(_) => "metal",
            BackendDevice::Cpu(_) => "cpu",
            BackendDevice::Cuda(_) => "cuda",
        }
    }
    pub fn memory_usage(&self) -> memory::Usage {
        self.1.usage()
    }
    /// Set the storage budget shared by this device and all its clones. Native
    /// allocator failures remain failures, not invented capacity measurements.
    pub fn set_memory_limit(&self, bytes: Option<usize>) -> Result<(), Error> {
        self.1.set_limit(bytes)
    }
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, Error> {
        let charge = self.1.charge(bytes)?;
        Ok(Buffer(
            match &self.0 {
                #[cfg(target_os = "macos")]
                BackendDevice::Metal(device) => Storage::Metal(device.buffer(bytes)?),
                BackendDevice::Cpu(_) => Storage::Cpu(seismic_cpu::Buffer::new(bytes)?),
                BackendDevice::Cuda(device) => Storage::Cuda(device.buffer(bytes)?),
            },
            Rc::new(Allocation {
                bytes,
                _charge: charge,
            }),
            0,
        ))
    }
    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    /// Compile through the sole specialization -> family -> solve -> resolve
    /// -> encode pipeline for this device. No selected or partially lowered
    /// artifact is exposed at the runtime boundary.
    pub fn compile(
        &self,
        program: &Program,
        entry: &str,
        workload: &Workload,
        budget: Budget,
    ) -> Result<Kernel, String> {
        self.compile_with_evidence(program, entry, workload, budget, &[])
    }

    /// Compile with whole-program numerical evidence keyed to complete physical
    /// assignments. Evidence for any other assignment is ignored by planning.
    pub fn compile_with_evidence(
        &self,
        program: &Program,
        entry: &str,
        workload: &Workload,
        budget: Budget,
        evidence: &[NumericalEvidence],
    ) -> Result<Kernel, String> {
        let started = std::time::Instant::now();
        let compiled = match &self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => {
                let backend = seismic_metal::mapping::MetalCompiler::from_device(device)?;
                let compiled =
                    pipeline::compile(program, entry, workload, &backend, evidence, budget)
                        .map_err(|error| error.to_string())?;
                let bound = binding(&compiled)?;
                let report = selection(&compiled, started.elapsed());
                let pipeline = compiled.native;
                Kernel {
                    binding: bound,
                    executable: Executable::Metal {
                        device: device.clone(),
                        pipeline: Box::new(pipeline),
                    },
                    selection: report,
                }
            }
            BackendDevice::Cpu(device) => {
                let workers = device.workers.borrow().count() as u64;
                let backend = seismic_cpu::mapping::Cpu::host(workers)?;
                let compiled =
                    pipeline::compile(program, entry, workload, &backend, evidence, budget)
                        .map_err(|error| error.to_string())?;
                let bound = binding(&compiled)?;
                let report = selection(&compiled, started.elapsed());
                let seismic_cpu::physical::NativeArtifact { kernel } = compiled.native;
                Kernel {
                    binding: bound,
                    executable: Executable::Cpu {
                        device: device.clone(),
                        kernel: Box::new(kernel),
                    },
                    selection: report,
                }
            }
            BackendDevice::Cuda(device) => {
                let backend = seismic_cuda::CudaCompiler::new(device)?;
                let compiled =
                    pipeline::compile(program, entry, workload, &backend, evidence, budget)
                        .map_err(|error| error.to_string())?;
                let bound = binding(&compiled)?;
                let report = selection(&compiled, started.elapsed());
                let sequence = compiled.native;
                Kernel {
                    binding: bound,
                    executable: Executable::Cuda {
                        _device: device.clone(),
                        sequence: Box::new(sequence),
                    },
                    selection: report,
                }
            }
        };
        Ok(compiled)
    }
}
impl Buffer {
    fn backend(&self) -> &'static str {
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(_) => "Metal",
            Storage::Cpu(_) => "CPU",
            Storage::Cuda(_) => "CUDA",
        }
    }
    #[cfg(target_os = "macos")]
    fn metal(&self) -> Result<&seismic_metal::runtime::Buffer, String> {
        match &self.0 {
            Storage::Metal(buffer) => Ok(buffer),
            _ => Err(format!(
                "a {} buffer is bound to a Metal kernel",
                self.backend()
            )),
        }
    }
    fn cpu(&self) -> Result<&seismic_cpu::Buffer, String> {
        match &self.0 {
            Storage::Cpu(buffer) => Ok(buffer),
            _ => Err(format!(
                "a {} buffer is bound to a CPU kernel",
                self.backend()
            )),
        }
    }
    fn cuda(&self) -> Result<seismic_cuda::Buffer, String> {
        match &self.0 {
            Storage::Cuda(buffer) => Ok(buffer.clone()),
            _ => Err(format!(
                "a {} buffer is bound to a CUDA kernel",
                self.backend()
            )),
        }
    }
    /// Resource-domain identity includes cloned device handles and retained views.
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.1._charge.belongs_to(&device.1)
    }
    /// Allocation identity survives cloning and byte views. It is distinct from
    /// logical view size and is never inferred from an exposed device address.
    pub fn shares_allocation(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.1, &other.1)
    }
    /// Physical bytes released if exactly these retained handles are dropped.
    /// Duplicate references count once; any other clone or view pins allocation.
    /// Calls/observations are synchronous, so backend invocation pins do not escape.
    pub fn reclaimable_bytes<'a>(
        buffers: impl IntoIterator<Item = &'a Self>,
    ) -> Result<usize, String> {
        let mut handles = std::collections::HashSet::new();
        let mut allocations = std::collections::HashMap::new();
        for buffer in buffers {
            if !handles.insert(buffer as *const Self) {
                continue;
            }
            let entry = allocations
                .entry(Rc::as_ptr(&buffer.1))
                .or_insert((0usize, &buffer.1));
            entry.0 += 1;
        }
        allocations
            .values()
            .try_fold(0usize, |total, (selected, allocation)| {
                let bytes = if *selected == Rc::strong_count(allocation) {
                    allocation.bytes
                } else {
                    0
                };
                total
                    .checked_add(bytes)
                    .ok_or_else(|| "reclaimable allocation total overflow".into())
            })
    }
    pub fn len(&self) -> usize {
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => b.len(),
            Storage::Cpu(b) => b.len(),
            Storage::Cuda(b) => b.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Offset relative to the retained allocation, including nested views.
    pub fn allocation_offset(&self) -> usize {
        self.2
    }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        let offset = self
            .2
            .checked_add(range.start)
            .ok_or("buffer view offset overflow")?;
        Ok(Self(
            match &self.0 {
                #[cfg(target_os = "macos")]
                Storage::Metal(b) => Storage::Metal(b.view(range)?),
                Storage::Cpu(b) => Storage::Cpu(b.view(range)?),
                Storage::Cuda(b) => Storage::Cuda(b.view(range)?),
            },
            self.1.clone(),
            offset,
        ))
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > self.len() {
            return Err("host write exceeds resident view".into());
        }
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                b.write(bytes);
                Ok(())
            }
            Storage::Cpu(b) => b.write(bytes),
            Storage::Cuda(b) => b.write(bytes),
        }
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), String> {
        if bytes.len() > self.len() {
            return Err("host read exceeds resident view".into());
        }
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                bytes.copy_from_slice(&b.read(bytes.len()));
                Ok(())
            }
            Storage::Cpu(b) => b.read(bytes),
            Storage::Cuda(b) => b.read(bytes),
        }
    }
}
impl Kernel {
    /// The resolved physical planning report retained for this kernel.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// Public buffer bindings of the retained root ABI, in ABI order.
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.binding.buffers
    }
    /// Scalar fields of the retained root ABI, in encoding order.
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.binding.scalars
    }
    /// Retained native launches of the structured schedule.
    pub fn launch_count(&self) -> usize {
        match &self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { pipeline, .. } => pipeline.emitted.launches.len(),
            Executable::Cpu { kernel, .. } => kernel.launch_count(),
            Executable::Cuda { sequence, .. } => sequence.launch_count(),
        }
    }
    /// Invocation contract of the resolved execution, checked before every
    /// submission: binding counts, byte sizes against the retained ABI, and
    /// the root alias rules over actual byte ranges. Typed alignment is
    /// checked by the native binding path.
    pub(crate) fn validate_invocation(
        &self,
        buffers: &[Buffer],
        scalars: &[f64],
    ) -> Result<(), String> {
        let specs = &self.binding.buffers;
        if buffers.len() != specs.len() || scalars.len() != self.binding.scalars.len() {
            return Err("invocation binding count differs from the retained entry ABI".into());
        }
        for (spec, buffer) in specs.iter().zip(buffers) {
            if buffer.len() < spec.bytes {
                return Err(format!(
                    "binding {}.{} needs {} bytes; {} are bound",
                    spec.parameter,
                    if spec.plane.is_empty() {
                        "dense"
                    } else {
                        &spec.plane
                    },
                    spec.bytes,
                    buffer.len()
                ));
            }
        }
        let locate = |binding: BufferBindingId| -> Option<(usize, &Buffer)> {
            specs
                .iter()
                .position(|spec| spec.binding == binding)
                .and_then(|ordinal| buffers.get(ordinal).map(|buffer| (ordinal, buffer)))
        };
        for rule in &self.binding.alias_rules {
            let (left, right) = match *rule {
                realization::AliasRule::MayOverlap { left, right } => (left, right),
                realization::AliasRule::MustDisjoint { left, right } => (left, right),
            };
            let (Some((_, left_buffer)), Some((_, right_buffer))) = (locate(left), locate(right))
            else {
                continue;
            };
            let (left_bytes, right_bytes) = (
                specs
                    .iter()
                    .find(|spec| spec.binding == left)
                    .map(|spec| spec.bytes)
                    .unwrap_or(0),
                specs
                    .iter()
                    .find(|spec| spec.binding == right)
                    .map(|spec| spec.bytes)
                    .unwrap_or(0),
            );
            if left_bytes == 0 || right_bytes == 0 {
                continue;
            }
            let overlapping = left_buffer.shares_allocation(right_buffer)
                && left_buffer.allocation_offset()
                    < right_buffer.allocation_offset().saturating_add(right_bytes)
                && right_buffer.allocation_offset()
                    < left_buffer.allocation_offset().saturating_add(left_bytes);
            if overlapping && matches!(rule, realization::AliasRule::MustDisjoint { .. }) {
                return Err(format!(
                    "bound buffers of bindings {left:?} and {right:?} overlap but must be disjoint"
                ));
            }
        }
        Ok(())
    }
    /// Synchronous completion boundary. Physical resources remain owned until
    /// completion, including failures. Tensor results are written into the
    /// invocation's result buffers.
    pub fn execute(&mut self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        self.invoke(buffers, scalars, false).map(|_| ())
    }
    pub fn execute_observed(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
    ) -> Result<ExecutionObservation, String> {
        let start = std::time::Instant::now();
        let device_seconds = self.invoke(buffers, scalars, true)?;
        Ok(ExecutionObservation {
            host_seconds: start.elapsed().as_secs_f64(),
            device_seconds,
            device_scope: Some(self.timing_scope()),
        })
    }
    /// One observed execution. The backend runtimes retain no per-dispatch
    /// native telemetry; the dispatch list is empty.
    pub fn execute_profiled(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
    ) -> Result<(ExecutionObservation, Vec<DispatchProfile>), String> {
        Ok((self.execute_observed(buffers, scalars)?, Vec::new()))
    }
    fn timing_scope(&self) -> DeviceTimingScope {
        match &self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { .. } => DeviceTimingScope::CommandBuffer,
            Executable::Cpu { .. } => DeviceTimingScope::CpuLaunches,
            Executable::Cuda { .. } => DeviceTimingScope::CudaLaunches,
        }
    }
    fn invoke(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
        timed: bool,
    ) -> Result<Option<f64>, String> {
        self.validate_invocation(buffers, scalars)?;
        let specs = self.binding.buffers.clone();
        let result_bindings = self.binding.result_bindings.clone();
        match &mut self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { device, pipeline } => {
                let parameters: Vec<&seismic_metal::runtime::Buffer> = specs
                    .iter()
                    .zip(buffers)
                    .filter(|(spec, _)| matches!(spec.role, BindingRole::Parameter))
                    .map(|(_, buffer)| buffer.metal())
                    .collect::<Result<Vec<_>, _>>()?;
                let invocation = seismic_metal::runtime::Invocation {
                    pipeline,
                    buffers: parameters,
                    scalars: scalars.to_vec(),
                };
                let outcome = device.run(&invocation)?;
                // Copy the runtime-allocated result planes into the bound buffers.
                for (binding, result) in result_bindings.iter().zip(&outcome.results) {
                    if let Some((spec, bound)) = specs
                        .iter()
                        .zip(buffers)
                        .find(|(spec, _)| spec.binding == *binding)
                    {
                        let bytes = result.read(spec.bytes);
                        bound.metal()?.write(&bytes);
                    }
                }
                Ok(None)
            }
            Executable::Cpu { device, kernel } => {
                let parameters: Vec<&seismic_cpu::Buffer> = specs
                    .iter()
                    .zip(buffers)
                    .filter(|(spec, _)| matches!(spec.role, BindingRole::Parameter))
                    .map(|(_, buffer)| buffer.cpu())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut workers = device
                    .workers
                    .try_borrow_mut()
                    .map_err(|_| "the CPU device is already executing")?;
                let start = std::time::Instant::now();
                let outputs = kernel
                    .run(&mut workers, &parameters, scalars)
                    .map_err(|failure| failure.to_string())?;
                for (path, plane, bytes) in &outputs.buffers {
                    let plane = match plane.as_str() {
                        "dense" => "",
                        other => other,
                    };
                    for (spec, bound) in specs.iter().zip(buffers) {
                        if let BindingRole::Result { path: result_path } = &spec.role {
                            if result_path.as_slice() == path.0.as_slice() && spec.plane == plane {
                                bound.cpu()?.write(bytes)?;
                            }
                        }
                    }
                }
                let _ = timed;
                Ok(Some(start.elapsed().as_secs_f64()))
            }
            Executable::Cuda { sequence, .. } => {
                let bound: Vec<seismic_cuda::Buffer> = buffers
                    .iter()
                    .map(Buffer::cuda)
                    .collect::<Result<Vec<_>, _>>()?;
                let status = sequence
                    .execute(&bound, scalars, timed)
                    .map_err(|error| error.to_string())?;
                if let Some(status) = status {
                    return Err(format!(
                        "CUDA invocation failed a safety check (status field {}, kind {})",
                        status.field, status.kind
                    ));
                }
                Ok(None)
            }
        }
    }
}
