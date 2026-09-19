//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Native executables are compiled only from a checked `Selected` witness.
//!
//! Metal, the CPU and CUDA are the backends on the structured pipeline. The CUDA driver is
//! loaded at run time, so opening a CUDA device fails cleanly on a host without one.
//! `BackendDevice`, `Storage`, `Executable`, `DeviceFacts` and `SelectedExecution` are closed
//! sums: a backend rejoins by adding one variant to each and one arm to every `match` on them
//! (all of them are in this file and in `plan.rs`).
//! The Metal variants exist on macOS only; the CPU and CUDA variants exist everywhere.

mod error;
pub use error::Error;
pub mod memory;
pub mod plan;
use seismic_compiler::selection::{self, Budget, ProofStatus, Selected};
/// What a `Selection` record and `plan::Settings` are made of.
pub use seismic_compiler::selection::{Phase, SearchStats, Strategy, Timings};
use seismic_lang::family::Workload;
use seismic_lang::sir::Program;
use seismic_lang::abi::ScalarParameter;
use seismic_lang::family::Witness;
use seismic_realization::{BufferSpec, InvocationConditions};
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
/// A checked selection for one backend: the only input of native compilation.
pub enum SelectedExecution {
    #[cfg(target_os = "macos")]
    Metal(Selected<seismic_metal::execution::Execution>),
    Cpu(Selected<seismic_cpu::mapping::Execution>),
    Cuda(Selected<seismic_cuda::execution::Launches>),
}
impl From<Selected<seismic_cuda::execution::Launches>> for SelectedExecution {
    fn from(selected: Selected<seismic_cuda::execution::Launches>) -> Self {
        Self::Cuda(selected)
    }
}
#[cfg(target_os = "macos")]
impl From<Selected<seismic_metal::execution::Execution>> for SelectedExecution {
    fn from(selected: Selected<seismic_metal::execution::Execution>) -> Self {
        Self::Metal(selected)
    }
}
impl From<Selected<seismic_cpu::mapping::Execution>> for SelectedExecution {
    fn from(selected: Selected<seismic_cpu::mapping::Execution>) -> Self {
        Self::Cpu(selected)
    }
}
/// Native compilation accepts only a checked selection. Prepared executions and
/// unselected candidates are not executable API inputs.
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
        sequence: Box<seismic_cuda::Sequence>,
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

/// What selection decided for a compiled kernel. Estimates are in the units of
/// `estimate_model`; they are estimates, not measurements.
#[derive(Clone, Debug)]
pub struct Selection {
    pub entry: String,
    pub witness: Witness,
    pub seed: Witness,
    pub estimate: u64,
    pub seed_estimate: u64,
    pub status: ProofStatus,
    pub estimate_model: String,
    pub lower_bound: u64,
    pub unresolved: Vec<String>,
    /// Static shape and element bindings of the compiled specialization (its identity).
    pub shapes: Vec<(String, i64)>,
    pub elements: Vec<(String, String)>,
    /// Wall time of the selection phases, and what the search did.
    pub timings: Timings,
    pub search: SearchStats,
    /// Source emission, where the backend separates it from native compilation (Metal).
    pub emit: Option<std::time::Duration>,
    /// Native compilation of the emitted execution (CPU and CUDA: emission included).
    pub native_compile: std::time::Duration,
}

pub struct Kernel {
    conditions: InvocationConditions,
    executable: Executable,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    selection: Selection,
}
impl Device {
    pub fn facts(&self) -> DeviceFacts {
        match &self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => DeviceFacts::Metal(device.info()),
            BackendDevice::Cpu(device) => DeviceFacts::Cpu(CpuInfo { workers: device.workers.borrow().count() as u64 }),
            BackendDevice::Cuda(device) => DeviceFacts::Cuda(device.info.clone()),
        }
    }
    /// The host CPU: one worker thread per unit of available parallelism, host memory buffers.
    pub fn cpu() -> Result<Self, String> {
        Ok(Self(
            BackendDevice::Cpu(Rc::new(CpuDevice { workers: std::cell::RefCell::new(seismic_cpu::Workers::host()?) })),
            Rc::default(),
        ))
    }
    /// CUDA device zero. Fails with the driver's reason on a host without a CUDA driver.
    pub fn cuda() -> Result<Self, String> {
        Ok(Self(BackendDevice::Cuda(Rc::new(seismic_cuda::Device::open(0).map_err(|e| format!("CUDA device: {e}"))?)), Rc::default()))
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
            other => Err(format!("unknown device `{other}`; expected `metal`, `cpu` or `cuda`")),
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
            Rc::new(Allocation { bytes, _charge: charge }),
            0,
        ))
    }
    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    /// Joint selection for this device's backend, built from the device's queried facts.
    /// The budget is the only tuning input.
    pub fn select(&self, program: &Program, entry: &str, workload: &Workload, budget: Budget) -> Result<SelectedExecution, String> {
        Ok(match self.facts() {
            #[cfg(target_os = "macos")]
            DeviceFacts::Metal(info) => {
                let backend = seismic_metal::mapping::Metal::from_device(&info).map_err(|e| e.to_string())?;
                selection::select(program, entry, workload, &backend, budget).map_err(|e| e.to_string())?.into()
            }
            DeviceFacts::Cpu(info) => {
                let backend = seismic_cpu::mapping::Cpu::host(info.workers).map_err(|e| e.to_string())?;
                selection::select(program, entry, workload, &backend, budget).map_err(|e| e.to_string())?.into()
            }
            DeviceFacts::Cuda(info) => {
                let backend = seismic_cuda::mapping::Cuda::from_device(&info).map_err(|e| e.to_string())?;
                selection::select(program, entry, workload, &backend, budget).map_err(|e| e.to_string())?.into()
            }
        })
    }
    /// Emit and natively compile exactly the selected execution. Nothing here selects,
    /// re-lowers or replaces any part of the witness. A selection for another backend than
    /// this device's is an error.
    pub fn compile_selected(&self, selected: impl Into<SelectedExecution>) -> Result<Kernel, String> {
        fn retained<E>(selected: Selected<E>) -> (E, Selection) {
            let Selected { execution, family, witness, estimate, seed, seed_estimate, status, estimate_model, lower_bound, unresolved, timings, search } = selected;
            let shapes = family.workload.shapes.iter().map(|(name, value)| (name.clone(), *value)).collect();
            let elements = family.workload.elems.iter().map(|(name, element)| (name.clone(), element.to_string())).collect();
            (execution, Selection { entry: family.entry.clone(), witness, seed, estimate, seed_estimate, status, estimate_model, lower_bound, unresolved, shapes, elements, timings, search, emit: None, native_compile: Default::default() })
        }
        let started = std::time::Instant::now();
        match (&self.0, selected.into()) {
            #[cfg(target_os = "macos")]
            (BackendDevice::Metal(device), SelectedExecution::Metal(selected)) => {
                let (execution, mut selection) = retained(selected);
                let conditions = InvocationConditions::from_lowered(execution.source())?;
                let emitted = seismic_metal::msl::emit_execution(&execution)?;
                let (buffers, scalars) = (emitted.buffers.clone(), emitted.scalars.clone());
                let emit = started.elapsed();
                let executable = Executable::Metal { device: device.clone(), pipeline: Box::new(device.compile(emitted)?) };
                (selection.emit, selection.native_compile) = (Some(emit), started.elapsed() - emit);
                Ok(Kernel { conditions, executable, buffers, scalars, selection })
            }
            (BackendDevice::Cpu(device), SelectedExecution::Cpu(selected)) => {
                let (execution, mut selection) = retained(selected);
                let conditions = execution.conditions().clone();
                let kernel = Box::new(seismic_cpu::compile(execution)?);
                selection.native_compile = started.elapsed();
                let (buffers, scalars) = (kernel.buffers().to_vec(), kernel.scalars().to_vec());
                Ok(Kernel { conditions, executable: Executable::Cpu { device: device.clone(), kernel }, buffers, scalars, selection })
            }
            (BackendDevice::Cuda(device), SelectedExecution::Cuda(selected)) => {
                let (launches, mut selection) = retained(selected);
                let conditions = launches.phases.first().map(|phase| phase.program().conditions.clone()).ok_or("the selected CUDA execution has no launch")?;
                let sequence = Box::new(device.compile_launches(launches)?);
                selection.native_compile = started.elapsed();
                let (buffers, scalars) = (sequence.buffers().to_vec(), sequence.scalars().to_vec());
                Ok(Kernel { conditions, executable: Executable::Cuda { _device: device.clone(), sequence }, buffers, scalars, selection })
            }
            _ => {
                Err(format!("the selection targets another backend than this `{}` device", self.backend()))
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
            _ => Err(format!("a {} buffer is bound to a Metal kernel", self.backend())),
        }
    }
    fn cpu(&self) -> Result<&seismic_cpu::Buffer, String> {
        match &self.0 {
            Storage::Cpu(buffer) => Ok(buffer),
            _ => Err(format!("a {} buffer is bound to a CPU kernel", self.backend())),
        }
    }
    fn cuda(&self) -> Result<seismic_cuda::Buffer, String> {
        match &self.0 {
            Storage::Cuda(buffer) => Ok(buffer.clone()),
            _ => Err(format!("a {} buffer is bound to a CUDA kernel", self.backend())),
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
    /// The witness this kernel was compiled from, with its proof status.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// Invocation conditions of the selected execution, checked before every submission.
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
            (Rc::as_ptr(&buffers[i].1) as usize as u64, buffers[i].2 as u64)
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
                let buffers = buffers.iter().map(Buffer::metal).collect::<Result<Vec<_>, _>>()?;
                let scalars = pipeline.emitted.encode_scalars(scalars)?;
                let observation = device.profile(pipeline, &buffers, &scalars)?;
                let dispatches = observation
                    .dispatches
                    .into_iter()
                    .map(|d| {
                        let launch = pipeline.emitted.launches.get(d.launch).ok_or("profiled dispatch names an absent launch")?;
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
                    ExecutionObservation { host_seconds: start.elapsed().as_secs_f64(), device_seconds: Some(seconds), device_scope: Some(DeviceTimingScope::CpuPhases) },
                    Vec::new(),
                ))
            }
            // CUDA reports the event intervals of its launches as one sum.
            Executable::Cuda { sequence, .. } => {
                let seconds = run_cuda(sequence, buffers, scalars, true)?;
                Ok((
                    ExecutionObservation { host_seconds: start.elapsed().as_secs_f64(), device_seconds: seconds, device_scope: Some(DeviceTimingScope::CudaLaunches) },
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
                let buffers = buffers.iter().map(Buffer::metal).collect::<Result<Vec<_>, _>>()?;
                let scalars = pipeline.emitted.encode_scalars(scalars)?;
                device
                    .run(pipeline, &buffers, &scalars, 1)
                    .map(|seconds| timed.then_some(seconds))
            }
            Executable::Cpu { device, kernel } => run_cpu(device, kernel, buffers, scalars).map(|seconds| timed.then_some(seconds)),
            Executable::Cuda { sequence, .. } => run_cuda(sequence, buffers, scalars, timed),
        }
    }
}
fn run_cuda(sequence: &mut seismic_cuda::Sequence, buffers: &[Buffer], scalars: &[f64], timed: bool) -> Result<Option<f64>, String> {
    let buffers = buffers.iter().map(Buffer::cuda).collect::<Result<Vec<_>, _>>()?;
    sequence.execute(&buffers, scalars, timed)
}
/// Run every phase of a CPU kernel to completion; returns the wall seconds of the phases.
fn run_cpu(device: &CpuDevice, kernel: &mut seismic_cpu::Kernel, buffers: &[Buffer], scalars: &[f64]) -> Result<f64, String> {
    let buffers = buffers.iter().map(Buffer::cpu).collect::<Result<Vec<_>, _>>()?;
    let mut workers = device.workers.try_borrow_mut().map_err(|_| "the CPU device is already executing")?;
    let start = std::time::Instant::now();
    kernel.run(&mut workers, &buffers, scalars)?;
    Ok(start.elapsed().as_secs_f64())
}
