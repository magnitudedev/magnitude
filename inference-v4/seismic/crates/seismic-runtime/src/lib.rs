//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Native executables require completed automatic selection.
mod error;
pub use error::Error;
pub mod execution;
pub mod memory;
pub mod plan;
pub mod tuner;
use seismic_lang::abi::ScalarParameter;
use seismic_realization::BufferSpec;
use std::rc::Rc;

#[derive(Clone)]
enum BackendDevice {
    Cpu,
    Cuda(Rc<seismic_cuda::Device>),
    #[cfg(target_os = "macos")]
    Metal(Rc<seismic_metal::runtime::Device>),
}
#[derive(Clone)]
/// Native compilation accepts only a completed compiler selection.
/// Fixed candidates and prepared executions are not executable API inputs.
///
/// ```compile_fail
/// use seismic_runtime::{Candidate, Device};
/// ```
/// ```compile_fail
/// fn bypass(device: &seismic_runtime::Device, execution: seismic_runtime::execution::Execution) {
///     device.compile_execution(execution);
/// }
/// ```
/// ```compile_fail
/// use seismic_runtime::plan::{Diagnostic, PlanCompiler};
/// ```
pub struct Device(BackendDevice, Rc<memory::Domain>);
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceFacts {
    Cpu {
        architecture: &'static str,
        operating_system: &'static str,
    },
    Cuda(seismic_cuda::DeviceInfo),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::runtime::DeviceInfo),
}
#[derive(Clone)]
enum Storage {
    Cpu(seismic_cpu::Buffer),
    Cuda(seismic_cuda::Buffer),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::runtime::Buffer),
}
#[derive(Clone)]
pub struct Buffer(Storage, Rc<Allocation>, usize);
struct Allocation {
    bytes: usize,
    _charge: memory::Charge,
}
enum Executable {
    Cpu(Box<seismic_cpu::Kernel>),
    Cuda(Box<seismic_cuda::Sequence>),
    #[cfg(target_os = "macos")]
    Metal {
        device: Rc<seismic_metal::runtime::Device>,
        pipeline: Box<seismic_metal::runtime::Pipeline>,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceTimingScope {
    /// Sum of event intervals around CUDA kernels; inter-phase host gaps excluded.
    KernelEventSum,
    /// Complete Metal command buffer interval, including its encoded dependencies.
    CommandBuffer,
}
/// Distinct timing boundaries; the device interval is absent on CPU.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionObservation {
    /// Binding, submission, synchronous completion and status validation.
    pub host_seconds: f64,
    /// Native GPU event/command time; excludes host binding and readback.
    /// CUDA sums phase kernel intervals; Metal measures its command buffer.
    pub device_seconds: Option<f64>,
    pub device_scope: Option<DeviceTimingScope>,
}

pub struct Kernel {
    conditions: seismic_realization::InvocationConditions,
    executable: Executable,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    tuning: tuner::Artifact,
}
impl Device {
    pub fn facts(&self) -> DeviceFacts {
        match &self.0 {
            BackendDevice::Cpu => DeviceFacts::Cpu {
                architecture: std::env::consts::ARCH,
                operating_system: std::env::consts::OS,
            },
            BackendDevice::Cuda(device) => DeviceFacts::Cuda(device.info.clone()),
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => DeviceFacts::Metal(device.info()),
        }
    }
    pub fn cpu() -> Self {
        Self(BackendDevice::Cpu, Rc::default())
    }
    pub fn cuda(ordinal: i32) -> Result<Self, String> {
        Ok(Self(BackendDevice::Cuda(Rc::new(
            seismic_cuda::Device::open(ordinal)?,
        )), Rc::default()))
    }
    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, String> {
        Ok(Self(BackendDevice::Metal(Rc::new(
            seismic_metal::runtime::Device::open()?,
        )), Rc::default()))
    }
    pub fn backend(&self) -> &'static str {
        match self.0 {
            BackendDevice::Cpu => "cpu",
            BackendDevice::Cuda(_) => "cuda",
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(_) => "metal",
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
                BackendDevice::Cpu => Storage::Cpu(seismic_cpu::Buffer::new(bytes)?),
                BackendDevice::Cuda(device) => Storage::Cuda(device.buffer(bytes)?),
                #[cfg(target_os = "macos")]
                BackendDevice::Metal(device) => Storage::Metal(device.buffer(bytes)?),
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
    pub fn compile_tuned(&self, tuned: tuner::TunedIr) -> Result<Kernel, String> {
        tuned.conditions().validate_device(&self.facts())?;
        let (execution, artifact) = tuned.into_parts();
        match (execution, artifact.conditions().implementation()) {
            (
                execution::Execution::Cpu(program),
                tuner::ImplementationConditions::Cpu(conditions),
            ) => {
                let invocation_conditions = program.conditions.clone();
                let native = seismic_cpu::compile_execution_with(program, &conditions.codegen)?;
                let buffers = native.buffers().to_vec();
                let scalars = native.scalars().to_vec();
                Ok(Kernel {
                    conditions: invocation_conditions,
                    executable: Executable::Cpu(Box::new(native)),
                    buffers,
                    scalars,
                    tuning: artifact,
                })
            }
            (
                execution @ execution::Execution::Cuda(_),
                tuner::ImplementationConditions::Cuda(_),
            ) => self.compile_execution(execution, artifact),
            #[cfg(target_os = "macos")]
            (
                execution @ execution::Execution::Metal(_),
                tuner::ImplementationConditions::Metal(_),
            ) => self.compile_execution(execution, artifact),
            _ => {
                return Err(
                    "selected execution and retained implementation conditions differ".into(),
                );
            }
        }
    }

    /// Native compilation consumes a prepared execution. It cannot select or
    /// replace a candidate by re-running lowering or preparation.
    fn compile_execution(&self, execution: execution::Execution, artifact: tuner::Artifact) -> Result<Kernel, String> {
        use execution::Execution;
        let conditions = match &execution {
            Execution::Cpu(p) => p.conditions.clone(),
            Execution::Cuda(phases) => {
                let conditions = phases
                    .first()
                    .map(|p| p.program().conditions.clone())
                    .unwrap_or_default();
                if phases.iter().any(|p| p.program().conditions != conditions) {
                    return Err("CUDA phases disagree on invocation conditions".into());
                }
                conditions
            }
            #[cfg(target_os = "macos")]
            Execution::Metal(e) => {
                seismic_realization::InvocationConditions::from_lowered(e.source())?
            }
        };
        let (executable, buffers, scalars) = match (&self.0, execution) {
            (BackendDevice::Cpu, Execution::Cpu(program)) => {
                let kernel = seismic_cpu::compile_execution(program)?;
                let buffers = kernel.buffers().to_vec();
                let scalars = kernel.scalars().to_vec();
                (Executable::Cpu(Box::new(kernel)), buffers, scalars)
            }
            (BackendDevice::Cuda(device), Execution::Cuda(phases)) => {
                let kernel = device.compile_executions(phases)?;
                let buffers = kernel.buffers().to_vec();
                let scalars = kernel.scalars().to_vec();
                (Executable::Cuda(Box::new(kernel)), buffers, scalars)
            }
            #[cfg(target_os = "macos")]
            (BackendDevice::Metal(device), Execution::Metal(execution)) => {
                let emitted = seismic_metal::msl::emit_execution(&execution)?;
                let buffers = emitted.buffers.clone();
                let scalars = emitted.scalars.clone();
                let pipeline = device.compile(emitted)?;
                (
                    Executable::Metal {
                        device: device.clone(),
                        pipeline: Box::new(pipeline),
                    },
                    buffers,
                    scalars,
                )
            }
            _ => return Err("selected execution and device backend differ".into()),
        };
        Ok(Kernel {
            conditions,
            executable,
            buffers,
            scalars,
            tuning: artifact,
        })
    }
}
impl Buffer {
    /// Resource-domain identity includes cloned device handles and retained views.
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.1._charge.belongs_to(&device.1)
    }
    fn model_alignment(&self) -> Result<u64, String> {
        match &self.0 {
            // The owner allocates Vec<u64>; this is a stable guaranteed minimum,
            // independent of allocator luck or the offset of this view.
            Storage::Cpu(_) => Ok(std::mem::align_of::<u64>() as u64),
            Storage::Cuda(buffer) => Ok(buffer.allocation_alignment()),
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => Ok(buffer.allocation_alignment()),
        }
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
            Storage::Cpu(b) => b.len(),
            Storage::Cuda(b) => b.len(),
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => b.len(),
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
                Storage::Cpu(b) => Storage::Cpu(b.view(range)?),
                Storage::Cuda(b) => Storage::Cuda(b.view(range)?),
                #[cfg(target_os = "macos")]
                Storage::Metal(b) => Storage::Metal(b.view(range)?),
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
            Storage::Cpu(b) => b.write(bytes),
            Storage::Cuda(b) => b.write(bytes),
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                b.write(bytes);
                Ok(())
            }
        }
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), String> {
        if bytes.len() > self.len() {
            return Err("host read exceeds resident view".into());
        }
        match &self.0 {
            Storage::Cpu(b) => b.read(bytes),
            Storage::Cuda(b) => b.read(bytes),
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                bytes.copy_from_slice(&b.read(bytes.len()));
                Ok(())
            }
        }
    }
}
impl Kernel {
    pub fn tuning(&self) -> &tuner::Artifact {
        &self.tuning
    }
    fn validate_tuning(&self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        if buffers.len() != self.buffers.len() {
            return Err("invocation buffer count differs from the retained entry ABI".into());
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
        })?;
        for (slot, binding) in self.tuning.workload().buffers.iter().enumerate() {
            if self.tuning.workload().conditions_allocation(binding.allocation)
                && !self.conditions.read_only_buffers().contains(&slot) {
                return Err("content-conditioned allocation may be modified by this execution".into());
            }
        }
        tuner::validate_bindings(self.tuning.workload(), buffers, &self.scalars, scalars)?;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    pub fn metal_pipeline_facts(&self) -> Option<&[seismic_metal::runtime::PipelineFacts]> {
        match &self.executable {
            Executable::Metal { pipeline, .. } => Some(&pipeline.facts),
            _ => None,
        }
    }
    pub fn phase_count(&self) -> usize {
        match &self.executable {
            Executable::Cpu(_) => 1,
            Executable::Cuda(sequence) => sequence.phase_count(),
            #[cfg(target_os = "macos")]
            Executable::Metal { pipeline, .. } => pipeline.phase_count(),
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
            device_scope: match &self.executable {
                Executable::Cpu(_) => None,
                Executable::Cuda(_) => Some(DeviceTimingScope::KernelEventSum),
                #[cfg(target_os = "macos")]
                Executable::Metal { .. } => Some(DeviceTimingScope::CommandBuffer),
            },
        })
    }
    fn invoke(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
        timed: bool,
    ) -> Result<Option<f64>, String> {
        self.validate_tuning(buffers, scalars)?;
        match &mut self.executable {
            Executable::Cpu(kernel) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match &b.0 {
                        Storage::Cpu(b) => Ok(b.clone()),
                        _ => Err("non-CPU buffer in CPU invocation"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                kernel.run_resident(&buffers, scalars).map(|_| None)
            }
            Executable::Cuda(kernel) => {
                let buffers = buffers
                    .iter()
                    .map(|b| match &b.0 {
                        Storage::Cuda(b) => Ok(b.clone()),
                        _ => Err("non-CUDA buffer in CUDA invocation"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                kernel.execute(&buffers, scalars, timed)
            }
            #[cfg(target_os = "macos")]
            Executable::Metal { device, pipeline } => {
                let buffers = buffers
                    .iter()
                    .map(|b| match &b.0 {
                        Storage::Metal(b) => Ok(b),
                        _ => Err("non-Metal buffer in Metal invocation"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let scalars = pipeline.emitted.encode_scalars(scalars)?;
                device
                    .run(pipeline, &buffers, &scalars, 1)
                    .map(|seconds| timed.then_some(seconds))
            }
        }
    }
}
