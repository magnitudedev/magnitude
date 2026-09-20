//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Native executables are produced only by the unified logical ->
//! physical -> native compiler pipeline.
//!
//! Metal, the CPU and CUDA are the backends on the structured pipeline. The CUDA driver is
//! loaded at run time, so opening a CUDA device fails cleanly on a host without one.
//! `BackendDevice`, `Storage`, `Executable` and `DeviceFacts` are closed
//! sums: a backend rejoins by adding one variant to each and one arm to every `match` on them
//! (all of them are in this file and in `plan.rs`).
//! The Metal variants exist on macOS only; the CPU and CUDA variants exist everywhere.

mod error;
pub use error::Error;
pub mod memory;
pub mod plan;
pub use seismic_compiler::planning::{Budget, NumericalEvidence, Strategy};
use seismic_lang::abi::ScalarParameter;
use seismic_lang::family::Workload;
use seismic_lang::sir::Program;
use seismic_realization::{
    executable::{
        AbiRole, ExecutableDialect, PlanAssignment, ResolvedLaunchId, ResolvedPlan,
        ResolvedScheduleItem, StorageScope,
    },
    BufferSpec, InvocationConditions,
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
/// The worker pool of one CPU device. Execution is synchronous, so one phase runs at a time.
struct CpuDevice {
    workers: std::cell::RefCell<seismic_cpu::Workers>,
}
/// Facts of the host a CPU device executes on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuInfo {
    /// Worker threads that execute the pieces of a root `parallel` phase.
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
    /// Wall time of the phases of a CPU kernel on the worker pool, after binding.
    CpuPhases,
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

/// Auditable report projected directly from the resolved executable plan. It
/// is not a second selection representation.
#[derive(Clone, Debug)]
pub struct Selection {
    pub entry: String,
    pub assignment: PlanAssignment,
    pub estimated_cost: i64,
    pub optimal: bool,
    pub resources: Vec<LaunchResources>,
    pub capability_fingerprint: String,
    pub numerical_assessment: seismic_lang::precision::NumericalAssessment,
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
    pub workgroups: [u64; 3],
    pub device_bytes: u64,
    pub workgroup_bytes: u64,
    pub private_bytes_per_participant: u64,
    pub bindings: u64,
    pub threads_per_group: u64,
}

pub struct Kernel {
    conditions: InvocationConditions,
    executable: Executable,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    selection: Selection,
}

fn selection<D: ExecutableDialect, A>(
    compiled: &seismic_compiler::pipeline::Compiled<D, A>,
    compile: std::time::Duration,
) -> Selection {
    fn resources<D: ExecutableDialect>(
        plan: &ResolvedPlan<D>,
        output: &mut Vec<LaunchResources>,
    ) {
        let device_bytes = plan
            .device_storage()
            .allocations
            .iter()
            .filter(|storage| storage.scope == StorageScope::Device)
            .map(|storage| storage.bytes)
            .sum();
        for item in plan.items().iter() {
            match item {
                ResolvedScheduleItem::Phase(phase) => {
                    for launch in phase.launches.iter() {
                        output.push(LaunchResources {
                            launch: launch.id,
                            workgroups: launch.geometry.workgroups,
                            device_bytes,
                            workgroup_bytes: launch.kernel.resources.workgroup_bytes,
                            private_bytes_per_participant: launch.kernel.resources.private_bytes,
                            bindings: launch.binding_groups.len() as u64,
                            threads_per_group: launch
                                .geometry
                                .participants_per_workgroup
                                .iter()
                                .product(),
                        });
                    }
                }
                ResolvedScheduleItem::Subplan(subplan) => resources(&subplan.plan, output),
            }
        }
    }
    let mut launch_resources = Vec::new();
    resources(&compiled.physical, &mut launch_resources);
    let physical = &compiled.physical;
    Selection {
        entry: compiled.logical.entry.clone(),
        assignment: physical.identity().assignment.clone(),
        estimated_cost: physical.estimated_cost(),
        optimal: physical.optimal(),
        resources: launch_resources,
        capability_fingerprint: compiled.logical.capability_fingerprint.clone(),
        numerical_assessment: physical.numerical_assessment().clone(),
        numerical_evidence_identity: Some(format!(
            "{}:{}",
            physical.identity().precision.method_revision,
            physical.identity().precision.evidence_domain
        )),
        shapes: compiled
            .logical
            .shapes
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect(),
        elements: compiled
            .logical
            .elems
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
            #[cfg(target_os = "macos")]
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
    /// Compile through the sole specialization -> physical planning -> native
    /// pipeline for this device. No selected or partially lowered artifact is
    /// exposed at the runtime boundary.
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
        match &self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => {
                let backend = seismic_metal::mapping::MetalCompiler::from_device(device)
                    .map_err(|error| error.to_string())?;
                let compiled = seismic_compiler::pipeline::compile(
                    program, entry, workload, &backend, evidence, budget,
                )
                .map_err(|error| error.to_string())?;
                let selection = selection(&compiled, started.elapsed());
                let pipeline = compiled.native;
                let conditions = InvocationConditions::from_executable(
                    &compiled.physical,
                    &pipeline.emitted.buffer_ids,
                )?;
                let buffers = pipeline.emitted.buffers.clone();
                let scalars = pipeline.emitted.scalars.clone();
                Ok(Kernel {
                    conditions,
                    executable: Executable::Metal {
                        device: device.clone(),
                        pipeline: Box::new(pipeline),
                    },
                    buffers,
                    scalars,
                    selection,
                })
            }
            BackendDevice::Cpu(device) => {
                let workers = device.workers.borrow().count() as u64;
                let backend =
                    seismic_cpu::mapping::Cpu::host(workers).map_err(|error| error.to_string())?;
                let compiled = seismic_compiler::pipeline::compile(
                    program, entry, workload, &backend, evidence, budget,
                )
                .map_err(|error| error.to_string())?;
                let selection = selection(&compiled, started.elapsed());
                let seismic_cpu::physical::NativeArtifact { kernel } = compiled.native;
                let conditions = InvocationConditions::from_executable(
                    &compiled.physical,
                    kernel.external_ids(),
                )?;
                let buffers = kernel.buffers().to_vec();
                let scalars = kernel.scalars().to_vec();
                Ok(Kernel {
                    conditions,
                    executable: Executable::Cpu {
                        device: device.clone(),
                        kernel: Box::new(kernel),
                    },
                    buffers,
                    scalars,
                    selection,
                })
            }
            BackendDevice::Cuda(device) => {
                let backend =
                    seismic_cuda::CudaCompiler::new(device).map_err(|error| error.to_string())?;
                let compiled = seismic_compiler::pipeline::compile(
                    program, entry, workload, &backend, evidence, budget,
                )
                .map_err(|error| error.to_string())?;
                let selection = selection(&compiled, started.elapsed());
                let sequence = compiled.native;
                let ids = sequence
                    .external_allocations()
                    .iter()
                    .map(|allocation| allocation.id)
                    .collect::<Vec<_>>();
                let conditions = InvocationConditions::from_executable(&compiled.physical, &ids)?;
                let buffers = sequence
                    .external_allocations()
                    .iter()
                    .map(|allocation| {
                        let abi = allocation
                            .role
                            .as_ref()
                            .ok_or_else(|| "public CUDA allocation has no ABI role".to_string())?;
                        let (parameter, plane, role) = match abi {
                            AbiRole::Parameter {
                                ordinal,
                                representation_plane,
                                ..
                            } => (
                                format!("parameter_{ordinal}"),
                                representation_plane.clone().unwrap_or_default(),
                                seismic_realization::BufferRole::Parameter,
                            ),
                            AbiRole::Result {
                                ordinal,
                                path,
                                representation_plane,
                            } => (
                                format!("result_{ordinal}"),
                                representation_plane.clone().unwrap_or_default(),
                                seismic_realization::BufferRole::Result { path: path.clone() },
                            ),
                            AbiRole::InvocationResource { name } => {
                                return Err(format!(
                                    "invocation resource `{name}` escaped into the public CUDA ABI"
                                ));
                            }
                        };
                        Ok(BufferSpec {
                            parameter,
                            plane,
                            role,
                            bytes: usize::try_from(allocation.bytes)
                                .map_err(|_| "CUDA ABI allocation exceeds host address range")?,
                            alignment: usize::try_from(allocation.alignment)
                                .map_err(|_| "CUDA ABI alignment exceeds host address range")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let scalars = Vec::new();
                Ok(Kernel {
                    conditions,
                    executable: Executable::Cuda {
                        _device: device.clone(),
                        sequence: Box::new(sequence),
                    },
                    buffers,
                    scalars,
                    selection,
                })
            }
        }
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
            #[cfg(target_os = "macos")]
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
    /// The logical specialization and resolved physical planning report retained
    /// for this kernel.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// Invocation conditions of the resolved execution, checked before every submission.
    /// Byte sizes and typed alignment are checked by the native binding path.
    fn validate_invocation(&self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.buffers.len() || scalars.len() != self.scalars.len() {
            return Err("invocation binding count differs from the retained entry ABI".into());
        }
        for &index in self.conditions.independent_buffers() {
            if buffers
                .iter()
                .enumerate()
                .any(|(other, b)| other != index && buffers[index].shares_allocation(b))
            {
                return Err(format!(
                    "private intermediate {} aliases another entry argument",
                    self.buffers[index].parameter
                ));
            }
        }
        self.conditions.validate_aliases(&self.buffers, |i| {
            (
                Rc::as_ptr(&buffers[i].1) as usize as u64,
                buffers[i].2 as u64,
            )
        })
    }
    #[cfg(target_os = "macos")]
    pub fn metal_pipeline_facts(&self) -> Option<&[seismic_metal::runtime::PipelineFacts]> {
        match &self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { pipeline, .. } => Some(&pipeline.facts),
            Executable::Cpu { .. } | Executable::Cuda { .. } => None,
        }
    }
    pub fn phase_count(&self) -> usize {
        match &self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { pipeline, .. } => pipeline.phase_count(),
            Executable::Cpu { kernel, .. } => kernel.phase_count(),
            Executable::Cuda { sequence, .. } => sequence.phase_count(),
        }
    }
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.scalars
    }
    /// Synchronous completion boundary. Physical resources remain owned until
    /// completion, including failures. This is not yet a batched submission plan.
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
    /// One execution with a sampled encoder per dispatch (native stage timestamps).
    /// The per-dispatch intervals are a qualification aid; the command interval of this
    /// run includes the per-encoder boundaries and is not a throughput sample.
    pub fn execute_profiled(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
    ) -> Result<(ExecutionObservation, Vec<DispatchProfile>), String> {
        let start = std::time::Instant::now();
        self.validate_invocation(buffers, scalars)?;
        match &mut self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { device, pipeline } => {
                let buffers = buffers
                    .iter()
                    .map(Buffer::metal)
                    .collect::<Result<Vec<_>, _>>()?;
                let scalars = pipeline.emitted.encode_scalars(scalars)?;
                let observation = device.profile(pipeline, &buffers, &scalars)?;
                let dispatches = observation
                    .dispatches
                    .into_iter()
                    .map(|d| {
                        let launch = pipeline
                            .emitted
                            .launches
                            .get(d.launch)
                            .ok_or("profiled dispatch names an absent launch")?;
                        Ok(DispatchProfile {
                            launch: d.launch,
                            kernel: d.kernel,
                            threadgroups: launch.threadgroups,
                            threads_per_threadgroup: launch.threads_per_threadgroup,
                            device_seconds: d.elapsed_ns as f64 * 1e-9,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((
                    ExecutionObservation {
                        host_seconds: start.elapsed().as_secs_f64(),
                        device_seconds: Some(observation.command_seconds),
                        device_scope: Some(DeviceTimingScope::CommandBuffer),
                    },
                    dispatches,
                ))
            }
            // A CPU kernel has no native per-dispatch timestamps: the phases are timed whole.
            Executable::Cpu { device, kernel } => {
                let seconds = run_cpu(device, kernel, buffers, scalars)?;
                Ok((
                    ExecutionObservation {
                        host_seconds: start.elapsed().as_secs_f64(),
                        device_seconds: Some(seconds),
                        device_scope: Some(DeviceTimingScope::CpuPhases),
                    },
                    Vec::new(),
                ))
            }
            // CUDA reports the event intervals of its launches as one sum.
            Executable::Cuda { sequence, .. } => {
                let seconds = run_cuda(sequence, buffers, scalars, true)?;
                Ok((
                    ExecutionObservation {
                        host_seconds: start.elapsed().as_secs_f64(),
                        device_seconds: seconds,
                        device_scope: Some(DeviceTimingScope::CudaLaunches),
                    },
                    Vec::new(),
                ))
            }
        }
    }
    fn timing_scope(&self) -> DeviceTimingScope {
        match &self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { .. } => DeviceTimingScope::CommandBuffer,
            Executable::Cpu { .. } => DeviceTimingScope::CpuPhases,
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
        match &mut self.executable {
            #[cfg(target_os = "macos")]
            Executable::Metal { device, pipeline } => {
                let buffers = buffers
                    .iter()
                    .map(Buffer::metal)
                    .collect::<Result<Vec<_>, _>>()?;
                let scalars = pipeline.emitted.encode_scalars(scalars)?;
                device
                    .run(pipeline, &buffers, &scalars, 1)
                    .map(|seconds| timed.then_some(seconds))
            }
            Executable::Cpu { device, kernel } => {
                run_cpu(device, kernel, buffers, scalars).map(|seconds| timed.then_some(seconds))
            }
            Executable::Cuda { sequence, .. } => run_cuda(sequence, buffers, scalars, timed),
        }
    }
}
fn run_cuda(
    sequence: &mut seismic_cuda::PhysicalSequence,
    buffers: &[Buffer],
    scalars: &[f64],
    timed: bool,
) -> Result<Option<f64>, String> {
    let buffers = buffers
        .iter()
        .map(Buffer::cuda)
        .collect::<Result<Vec<_>, _>>()?;
    sequence.execute_with_scalars(&buffers, scalars, timed)
}
/// Run every phase of a CPU kernel to completion; returns the wall seconds of the phases.
fn run_cpu(
    device: &CpuDevice,
    kernel: &mut seismic_cpu::Kernel,
    buffers: &[Buffer],
    scalars: &[f64],
) -> Result<f64, String> {
    let buffers = buffers
        .iter()
        .map(Buffer::cpu)
        .collect::<Result<Vec<_>, _>>()?;
    let mut workers = device
        .workers
        .try_borrow_mut()
        .map_err(|_| "the CPU device is already executing")?;
    let start = std::time::Instant::now();
    kernel.run(&mut workers, &buffers, scalars)?;
    Ok(start.elapsed().as_secs_f64())
}
