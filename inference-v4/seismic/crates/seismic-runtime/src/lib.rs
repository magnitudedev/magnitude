//! Native resource ownership and invocation. Numerical work stays in compiled
//! Seismic. Native executables are produced only by the unified semantic ->
//! logical -> plan-space -> physical-plan -> native-artifact pipeline.
//!
//! Metal, the CPU and CUDA are the backends on the structured pipeline. This
//! crate owns explicit compiled-plan preparation (`plan`), one-shot invocation
//! validation into `PreparedInvocation` (`invocation`), and submission over
//! prepared invocations through the backend executors (`submission`). Raw
//! caller bindings cannot reach an executor; every execution failure is the
//! shared invocation/safety/external taxonomy.
//!
//! The CUDA driver is loaded at run time, so opening a CUDA device fails
//! cleanly on a host without one. `Device`, `Buffer`, and `DeviceFacts` are
//! closed sums: a backend rejoins by adding one variant to each and one arm
//! to every match on them. The Metal variants exist on macOS only; the CPU
//! and CUDA variants exist everywhere.

mod error;
use error::external;
pub use error::Error;
/// Invocation validation: the only constructor of `PreparedInvocation` (R1).
pub mod invocation;
pub mod memory;
pub mod plan;
/// Submission of prepared invocations (R1).
pub mod submission;

pub use seismic_compiler::pipeline::CompileFailure;
pub use seismic_compiler::planning::Budget;
pub use seismic_realization::numerics::NumericalEvidence;

use seismic_realization::failure::{ExecutionFailure, ExternalStage};
use std::cell::RefCell;
use std::rc::Rc;

/// The handle of one open device, by backend. Total: every device has
/// exactly one arm, paired with its storage kind at construction.
pub(crate) enum DeviceHandle<'a> {
    #[cfg(target_os = "macos")]
    Metal(&'a Rc<seismic_metal::runtime::Device>),
    Cpu(&'a Rc<CpuDevice>),
    Cuda(&'a Rc<seismic_cuda::runtime::Device>),
}

#[derive(Clone)]
pub(crate) enum BackendDevice {
    #[cfg(target_os = "macos")]
    Metal(Rc<seismic_metal::runtime::Device>),
    Cpu(Rc<CpuDevice>),
    /// A private, thread-affine driver context.
    Cuda(Rc<seismic_cuda::runtime::Device>),
}

/// The worker pool of one CPU device. Execution is synchronous, so one launch
/// runs at a time.
pub(crate) struct CpuDevice {
    pub(crate) workers: RefCell<seismic_cpu::Workers>,
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
    Cuda(seismic_cuda::runtime::DeviceInfo),
}

#[derive(Clone)]
pub(crate) enum Storage {
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::runtime::Buffer),
    /// Host memory.
    Cpu(seismic_cpu::Buffer),
    Cuda(seismic_cuda::runtime::Buffer),
}

#[derive(Clone)]
pub struct Buffer(Storage, Rc<Allocation>, usize);

struct Allocation {
    bytes: usize,
    _charge: memory::Charge,
}

/// Observation of one synchronous submission: its wall time and one record
/// per launch its executor ran (dispatched geometry and per-launch wall
/// time, completion waits included). Executors that do not yet record
/// launches report an empty set.
#[derive(Clone, Debug, Default)]
pub struct ExecutionObservation {
    pub host_seconds: f64,
    pub launches: Vec<seismic_realization::physical::LaunchExecution>,
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

    /// The device's backend handle. The one total backend dispatch; a
    /// caller never pairs handles by hand.
    pub(crate) fn backend_handle(&self) -> DeviceHandle<'_> {
        match &self.0 {
            #[cfg(target_os = "macos")]
            BackendDevice::Metal(device) => DeviceHandle::Metal(device),
            BackendDevice::Cpu(device) => DeviceHandle::Cpu(device),
            BackendDevice::Cuda(device) => DeviceHandle::Cuda(device),
        }
    }

    /// The host CPU: one worker thread per unit of available parallelism, host
    /// memory buffers.
    pub fn cpu() -> Result<Self, Error> {
        let workers = seismic_cpu::Workers::host()
            .map_err(|reason| external(ExternalStage::Allocation, reason.to_string()))?;
        Ok(Self(
            BackendDevice::Cpu(Rc::new(CpuDevice {
                workers: RefCell::new(workers),
            })),
            Rc::default(),
        ))
    }

    /// CUDA device zero. Fails with the driver's reason on a host without a
    /// CUDA driver.
    pub fn cuda() -> Result<Self, Error> {
        let device = seismic_cuda::runtime::Device::open(0).map_err(|error| match error {
            seismic_cuda::runtime::OpenError::Driver(detail) => {
                external(ExternalStage::Driver, detail)
            }
            seismic_cuda::runtime::OpenError::Target(error) => {
                external(ExternalStage::Driver, error.to_string())
            }
        })?;
        Ok(Self(BackendDevice::Cuda(Rc::new(device)), Rc::default()))
    }

    /// Open the device a target name denotes: `metal`, `cpu` or `cuda`.
    pub fn open(target: &str) -> Result<Self, Error> {
        match target {
            #[cfg(target_os = "macos")]
            "metal" => Self::metal(),
            "cpu" => Self::cpu(),
            "cuda" => Self::cuda(),
            #[cfg(not(target_os = "macos"))]
            "metal" => Err(external(
                ExternalStage::Driver,
                "the Metal device exists on macOS only",
            )),
            other => Err(external(
                ExternalStage::Driver,
                format!("unknown device `{other}`; expected `metal`, `cpu` or `cuda`"),
            )),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self, Error> {
        let device = seismic_metal::runtime::Device::open()
            .map_err(|reason| external(ExternalStage::Driver, reason.to_string()))?;
        Ok(Self(BackendDevice::Metal(Rc::new(device)), Rc::default()))
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
                BackendDevice::Metal(device) => Storage::Metal(
                    device
                        .buffer(bytes)
                        .map_err(|reason| external(ExternalStage::Allocation, reason.to_string()))?,
                ),
                BackendDevice::Cpu(_) => Storage::Cpu(
                    seismic_cpu::Buffer::new(bytes)
                        .map_err(|reason| external(ExternalStage::Allocation, reason.to_string()))?,
                ),
                BackendDevice::Cuda(device) => {
                    Storage::Cuda(device.buffer(bytes).map_err(cuda_error)?)
                }
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
}

impl Buffer {
    /// The host-memory handle of a CPU buffer.
    pub fn as_cpu(&self) -> Option<&seismic_cpu::Buffer> {
        match &self.0 {
            Storage::Cpu(buffer) => Some(buffer),
            #[cfg(target_os = "macos")]
            Storage::Metal(_) => None,
            Storage::Cuda(_) => None,
        }
    }

    /// The Metal handle of a macOS Metal buffer.
    #[cfg(target_os = "macos")]
    pub fn as_metal(&self) -> Option<&seismic_metal::runtime::Buffer> {
        match &self.0 {
            Storage::Metal(buffer) => Some(buffer),
            Storage::Cpu(_) => None,
            Storage::Cuda(_) => None,
        }
    }

    /// The device-memory handle of a CUDA buffer.
    pub fn as_cuda(&self) -> Option<&seismic_cuda::runtime::Buffer> {
        match &self.0 {
            Storage::Cuda(buffer) => Some(buffer),
            #[cfg(target_os = "macos")]
            Storage::Metal(_) => None,
            Storage::Cpu(_) => None,
        }
    }

    /// The host pointer of a CPU buffer view (the CPU executor's binding
    /// path).
    pub fn host_pointer(&self) -> Option<*mut u8> {
        self.as_cpu().map(|buffer| buffer.data_pointer())
    }

    /// The device pointer of a CUDA buffer view (the CUDA executor's binding
    /// path).
    pub fn device_pointer(&self) -> Option<u64> {
        self.as_cuda().map(|buffer| buffer.device_pointer())
    }

    /// Resource-domain identity includes cloned device handles and retained
    /// views.
    pub fn belongs_to(&self, device: &Device) -> bool {
        self.1._charge.belongs_to(&device.1)
    }

    /// Allocation identity survives cloning and byte views. It is distinct
    /// from logical view size and is never inferred from an exposed device
    /// address.
    pub fn shares_allocation(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.1, &other.1)
    }

    /// Physical bytes released if exactly these retained handles are dropped.
    /// Duplicate references count once; any other clone or view pins
    /// allocation. Calls/observations are synchronous, so backend invocation
    /// pins do not escape.
    pub fn reclaimable_bytes<'a>(
        buffers: impl IntoIterator<Item = &'a Self>,
    ) -> Result<usize, Error> {
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
                total.checked_add(bytes).ok_or_else(|| {
                    external(
                        ExternalStage::Allocation,
                        "reclaimable allocation total overflow",
                    )
                })
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

    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, Error> {
        if range.start > range.end || range.end > self.len() {
            return Err(Error::Range {
                requested: range.end,
                available: self.len(),
            });
        }
        let offset = self.2.checked_add(range.start).ok_or(Error::Range {
            requested: range.start,
            available: usize::MAX - self.2,
        })?;
        Ok(Self(
            match &self.0 {
                #[cfg(target_os = "macos")]
                Storage::Metal(b) => Storage::Metal(
                    b.view(range.clone())
                        .map_err(|reason| external(ExternalStage::Driver, reason.to_string()))?,
                ),
                Storage::Cpu(b) => Storage::Cpu(
                    b.view(range.clone())
                        .map_err(|reason| external(ExternalStage::Driver, reason.to_string()))?,
                ),
                Storage::Cuda(b) => {
                    Storage::Cuda(b.view(range.clone()).map_err(cuda_error)?)
                }
            },
            self.1.clone(),
            offset,
        ))
    }

    pub fn write(&self, bytes: &[u8]) -> Result<(), Error> {
        if bytes.len() > self.len() {
            return Err(Error::Range {
                requested: bytes.len(),
                available: self.len(),
            });
        }
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                b.write(bytes);
                Ok(())
            }
            Storage::Cpu(b) => b
                .write(bytes)
                .map_err(|reason| external(ExternalStage::Driver, reason.to_string())),
            Storage::Cuda(b) => b.write(bytes).map_err(cuda_error),
        }
    }

    pub fn read(&self, bytes: &mut [u8]) -> Result<(), Error> {
        if bytes.len() > self.len() {
            return Err(Error::Range {
                requested: bytes.len(),
                available: self.len(),
            });
        }
        match &self.0 {
            #[cfg(target_os = "macos")]
            Storage::Metal(b) => {
                bytes.copy_from_slice(&b.read(bytes.len()));
                Ok(())
            }
            Storage::Cpu(b) => b
                .read(bytes)
                .map_err(|reason| external(ExternalStage::Driver, reason.to_string())),
            Storage::Cuda(b) => b.read(bytes).map_err(cuda_error),
        }
    }
}

/// A CUDA device-boundary failure of a buffer operation. Device-boundary
/// buffer operations surface only external failures; any other class a
/// buffer operation reports is the device system failing around the
/// transfer, and stays external.
fn cuda_error(failure: ExecutionFailure) -> Error {
    match failure {
        ExecutionFailure::External(failure) => Error::External(failure),
        failure @ (ExecutionFailure::Invocation(_) | ExecutionFailure::Safety(_)) => {
            external(ExternalStage::Driver, failure.to_string())
        }
    }
}
