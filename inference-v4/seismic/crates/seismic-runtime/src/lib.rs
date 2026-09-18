//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Candidates are explicit until accounting can justify selection.
pub mod plan;
pub mod execution;
pub mod tuner;
pub mod choices;
use seismic_lang::{abi::ScalarParameter, lowered_ir::LoweredIr};
use seismic_realization::{BufferSpec, LoadStrategy, ScalarOptions};
use std::rc::Rc;

enum BackendDevice {
    Cpu,
    Cuda(Rc<seismic_cuda::Device>),
    #[cfg(target_os = "macos")]
    Metal(Rc<seismic_metal::runtime::Device>),
}
pub struct Device(BackendDevice);
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
}
/// A supplied realization, not an automatic performance preference.
#[derive(Clone)]
pub enum Candidate {
    Cpu {
        loads: LoadStrategy,
    },
    Cuda {
        options: ScalarOptions,
        threads_per_block: u32,
    },
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::execution::Config),
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
    executable: Executable,
    buffers: Vec<BufferSpec>,
    scalars: Vec<ScalarParameter>,
    tuning: Option<tuner::Artifact>,
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
        Self(BackendDevice::Cpu)
    }
    pub fn cuda(ordinal: i32) -> Result<Self, String> {
        Ok(Self(BackendDevice::Cuda(Rc::new(
            seismic_cuda::Device::open(ordinal)?,
        ))))
    }
    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, String> {
        Ok(Self(BackendDevice::Metal(Rc::new(
            seismic_metal::runtime::Device::open()?,
        ))))
    }
    pub fn backend(&self) -> &'static str {
        match self.0 {
            BackendDevice::Cpu => "cpu",
            BackendDevice::Cuda(_) => "cuda",
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(_) => "metal",
        }
    }
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String> {
        Ok(Buffer(
            match &self.0 {
                BackendDevice::Cpu => Storage::Cpu(seismic_cpu::Buffer::new(bytes)?),
                BackendDevice::Cuda(device) => Storage::Cuda(device.buffer(bytes)?),
                #[cfg(target_os = "macos")]
                BackendDevice::Metal(device) => Storage::Metal(device.buffer(bytes)?),
            },
            Rc::new(Allocation { bytes }),
            0,
        ))
    }
    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    pub fn compile_tuned(&self, tuned: tuner::TunedIr) -> Result<Kernel, String> {
        if !matches!(self.0,BackendDevice::Cpu) { return Err("CPU tuned execution requires a CPU device".into()); }
        let (program, artifact) = tuned.into_parts();
        let kernel = seismic_cpu::compile_execution_with(program,&artifact.conditions().codegen)?;
        let buffers=kernel.buffers().to_vec();
        let scalars=kernel.scalars().to_vec();
        Ok(Kernel { executable:Executable::Cpu(Box::new(kernel)),buffers,scalars,tuning:Some(artifact) })
    }
    pub fn compile(&self, lowered: &LoweredIr, candidate: Candidate) -> Result<Kernel, String> {
        let execution = execution::Execution::prepare(lowered, candidate, &self.facts())?;
        self.compile_execution(execution)
    }
    /// Native compilation consumes a prepared execution. It cannot select or
    /// replace a candidate by re-running lowering or preparation.
    pub fn compile_execution(&self, execution: execution::Execution) -> Result<Kernel, String> {
        use execution::Execution;
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
                (Executable::Metal { device: device.clone(), pipeline: Box::new(pipeline) }, buffers, scalars)
            }
            _ => return Err("selected execution and device backend differ".into()),
        };
        Ok(Kernel { executable, buffers, scalars, tuning: None })
    }
}
impl Buffer {
    fn model_alignment(&self) -> Result<u64, String> {
        match &self.0 {
            // The owner allocates Vec<u64>; this is a stable guaranteed minimum,
            // independent of allocator luck or the offset of this view.
            Storage::Cpu(_) => Ok(std::mem::align_of::<u64>() as u64),
            _ => Err("GPU buffer model admission requires its backend contract".into()),
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
        let offset = self.2.checked_add(range.start).ok_or("buffer view offset overflow")?;
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
    pub fn tuning(&self) -> Option<&tuner::Artifact> { self.tuning.as_ref() }
    fn validate_tuning(&self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        if let Some(artifact) = &self.tuning {
            tuner::validate_bindings(artifact.workload(), buffers, &self.scalars, scalars)?;
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    pub fn metal_pipeline_facts(&self) -> Option<&[seismic_metal::runtime::PipelineFacts]> {
        match &self.executable { Executable::Metal {pipeline,..}=>Some(&pipeline.facts), _=>None }
    }
    pub fn phase_count(&self) -> usize {
        match &self.executable {
            Executable::Cpu(_) => 1,
            Executable::Cuda(sequence) => sequence.phase_count(),
            #[cfg(target_os = "macos")]
            Executable::Metal { pipeline, .. } => pipeline.emitted.launches.len(),
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
