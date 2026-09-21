//! The CUDA device facade and the executor over prepared invocations.
//!
//! The device owns the private driver context and compiles encoded
//! launches into loaded native functions (PTX→cubin→module→handle,
//! reflecting the native-resource facts). The executor runs one prepared
//! invocation over the sealed native tree with direct dense handles: it
//! allocates the compiler-owned blocks (arena, executor slots, status,
//! results, runtime-extent values, grid-barrier scratch), evaluates the
//! retained execution expressions against the validated invocation values,
//! threads joins and carries, skips zero-work launches, submits
//! cooperative launches through `cuLaunchCooperativeKernel`, and reports
//! the first recorded status error after synchronous completion.
//!
//! Execution failures are exactly `ExecutionFailure`: `Safety` from a
//! failed retained guard or a recorded kernel check; `External` for
//! allocation, driver, and submission failures. There is no String error
//! channel and no lookup by id.

use crate::driver::{Allocation, Context, Driver, Handle, Module};
use crate::intrinsics::Dialect;
use crate::native::{NativeArtifact, NativeStep};
use seismic_compiler::pipeline::AssemblyFailure;
use seismic_lang::types::DType;
use seismic_realization::failure::{
    ExecutionFailure, ExternalFailure, ExternalStage, SafetyKind, SafetyViolation,
    SafetyViolationSource,
};
use seismic_realization::ids::{
    BufferSlot, GuardIx, LaunchIx, ObligationRef, ScalarSlotIx, StorageIx,
};
use seismic_realization::invocation::{InvocationValues, ScalarWord};
use seismic_realization::physical::{
    ExecutionExpr, GuardPredicate, PhysicalPlan, PhysicalStep, ScalarSource, SealedValue,
    StoragePlacement,
};
use std::collections::BTreeSet;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub driver_version: i32,
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
    pub warp_size: u32,
    pub multiprocessors: u32,
    pub global_memory_bytes: u64,
    pub l2_cache_bytes: u64,
    pub max_threads_per_multiprocessor: u32,
    pub registers_32bit_per_multiprocessor: u32,
    pub shared_bytes_per_multiprocessor: u64,
    /// The device and driver support cooperative launch
    /// (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`).
    pub cooperative_launch: bool,
}

impl DeviceInfo {
    /// The cooperative-grid facility, when the device supports cooperative
    /// launch: the maximum number of participants that can stay
    /// simultaneously resident (every multiprocessor at its thread
    /// capacity).
    pub fn cooperative_grid(&self) -> Option<seismic_realization::target::CooperativeGrid> {
        self.cooperative_launch
            .then(|| seismic_realization::target::CooperativeGrid {
                max_resident_participants: u64::from(self.max_threads_per_multiprocessor)
                    * u64::from(self.multiprocessors),
            })
    }
}

/// A private context, deliberately thread-affine. Resource owners retain this
/// context; no allocation or module can outlive its driver.
pub struct Device {
    context: Rc<Context>,
    pub info: DeviceInfo,
    target_profile: crate::target::TargetProfile,
}

/// Reflected native resources of one compiled function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeResources {
    pub registers_per_thread: i32,
    pub local_bytes_per_thread: i32,
    pub max_threads_per_block: i32,
    pub shared_bytes_per_block: i32,
}

/// One native-assembly failure of the device boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssemblyError {
    /// PTX compilation, linking, or image loading failed. This is not
    /// evidence that another physical assignment is legal.
    Toolchain { stage: &'static str, detail: String },
    /// A CUDA driver operation failed independently of the code.
    Driver(crate::driver::DriverError),
    /// Host-side preparation failed after successful native emission.
    Preparation { stage: &'static str, detail: String },
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toolchain { stage, detail } => write!(f, "CUDA {stage} failed: {detail}"),
            Self::Driver(error) => error.fmt(f),
            Self::Preparation { stage, detail } => write!(f, "CUDA {stage} failed: {detail}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

impl From<AssemblyError> for AssemblyFailure {
    fn from(error: AssemblyError) -> Self {
        match error {
            AssemblyError::Toolchain { stage, detail } => AssemblyFailure::Toolchain(
                seismic_compiler::pipeline::ToolchainReport(format!("{stage}: {detail}")),
            ),
            AssemblyError::Driver(error) => AssemblyFailure::SystemPreparation(
                seismic_compiler::pipeline::SystemReport(error.to_string()),
            ),
            AssemblyError::Preparation { stage, detail } => AssemblyFailure::SystemPreparation(
                seismic_compiler::pipeline::SystemReport(format!("{stage}: {detail}")),
            ),
        }
    }
}

/// A checked resident byte range. Clones and subviews retain the allocation and
/// its private context. Host access is synchronous; handles remain thread-affine.
#[derive(Clone)]
pub struct Buffer {
    allocation: Rc<Allocation>,
    offset: usize,
    len: usize,
}

impl Buffer {
    /// Alignment of the retained allocation's actual device address.
    pub fn allocation_alignment(&self) -> u64 {
        let address = self.allocation.pointer;
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
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, ExecutionFailure> {
        if range.start > range.end || range.end > self.len {
            return Err(ExecutionFailure::External(ExternalFailure {
                stage: ExternalStage::Driver,
                detail: "CUDA buffer view exceeds its parent".into(),
            }));
        }
        let offset = match self.offset.checked_add(range.start) {
            Some(offset) => offset,
            None => {
                return Err(ExecutionFailure::External(ExternalFailure {
                    stage: ExternalStage::Submission,
                    detail: "CUDA buffer view offset overflows the addressable range".into(),
                }))
            }
        };
        Ok(Self {
            allocation: self.allocation.clone(),
            offset,
            len: range.end - range.start,
        })
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), ExecutionFailure> {
        self.allocation
            .upload_at(self.offset, bytes)
            .map_err(external_driver)
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), ExecutionFailure> {
        self.allocation
            .download_at(self.offset, bytes)
            .map_err(external_driver)
    }
    /// The device pointer of this buffer view (prepared invocations bind
    /// it; no other caller may rely on it).
    pub fn device_pointer(&self) -> u64 {
        self.allocation.pointer + self.offset as u64
    }
}

fn external_driver(detail: String) -> ExecutionFailure {
    ExecutionFailure::External(ExternalFailure {
        stage: ExternalStage::Driver,
        detail,
    })
}

fn external_allocation(detail: String) -> ExecutionFailure {
    ExecutionFailure::External(ExternalFailure {
        stage: ExternalStage::Allocation,
        detail,
    })
}

/// Opening a CUDA device failed: no usable driver, an unusable observed
/// target, or a driver failure while observing the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// No usable CUDA driver library on this host.
    Driver(String),
    /// The observed device cannot form a CUDA target profile.
    Target(crate::target::TargetError),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(detail) => write!(f, "CUDA driver unavailable: {detail}"),
            Self::Target(error) => write!(f, "CUDA target: {error}"),
        }
    }
}

impl std::error::Error for OpenError {}

impl Device {
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, ExecutionFailure> {
        Ok(Buffer {
            allocation: Rc::new(
                Allocation::new(&self.context, bytes).map_err(external_allocation)?,
            ),
            offset: 0,
            len: bytes,
        })
    }

    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, ExecutionFailure> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }

    /// The device's cooperative-grid facility, when it has one.
    pub fn cooperative_grid(&self) -> Option<seismic_realization::target::CooperativeGrid> {
        self.info.cooperative_grid()
    }

    pub(crate) fn context(&self) -> &Rc<Context> {
        &self.context
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }

    /// Open one CUDA device by ordinal. Fails cleanly on hosts without a
    /// driver.
    pub fn open(ordinal: i32) -> Result<Self, OpenError> {
        let driver = Driver::load().map_err(OpenError::Driver)?;
        let mut device = 0;
        unsafe {
            driver
                .check(
                    (driver.device_get)(&mut device, ordinal),
                    "device selection",
                )
                .map_err(OpenError::Driver)?;
        }
        let attribute = |key| -> Result<u32, OpenError> {
            let mut n = 0;
            unsafe {
                driver
                    .check(
                        (driver.device_attribute)(&mut n, key, device),
                        "device attribute",
                    )
                    .map_err(OpenError::Driver)?;
            }
            u32::try_from(n)
                .map_err(|_| OpenError::Driver("invalid negative device attribute".into()))
        };
        let major = attribute(75)?;
        let minor = attribute(76)?;
        let mut name = [0 as std::ffi::c_char; 256];
        let mut version = 0;
        let mut memory = 0usize;
        unsafe {
            driver
                .check(
                    (driver.device_name)(name.as_mut_ptr(), name.len() as i32, device),
                    "device name",
                )
                .map_err(OpenError::Driver)?;
            driver
                .check((driver.driver_version)(&mut version), "driver version")
                .map_err(OpenError::Driver)?;
            driver
                .check(
                    (driver.device_total_memory)(&mut memory, device),
                    "total device memory",
                )
                .map_err(OpenError::Driver)?;
        }
        let info = DeviceInfo {
            name: unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            compute_capability: (major as i32, minor as i32),
            driver_version: version,
            max_threads_per_block: attribute(1)?.min(attribute(2)?),
            max_grid_x: attribute(5)?,
            warp_size: attribute(10)?,
            multiprocessors: attribute(16)?,
            global_memory_bytes: memory as u64,
            l2_cache_bytes: u64::from(attribute(38)?),
            max_threads_per_multiprocessor: attribute(39)?,
            registers_32bit_per_multiprocessor: attribute(82)?,
            shared_bytes_per_multiprocessor: u64::from(attribute(81)?),
            // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
            cooperative_launch: attribute(99)? == 1,
        };
        let limits = crate::mapping::Limits {
            max_threads_per_block: info.max_threads_per_block,
            max_grid_x: info.max_grid_x,
            warp_size: info.warp_size,
            max_scratch_bytes: info.global_memory_bytes / 4,
        };
        let target_profile = crate::target::TargetProfile::from_observation(
            crate::target::TargetObservation::driver(info.compute_capability, info.driver_version)
                .map_err(OpenError::Target)?,
            limits,
            info.cooperative_grid(),
        )
        .map_err(OpenError::Target)?;
        Ok(Self {
            context: Context::new(driver, device).map_err(OpenError::Driver)?,
            info,
            target_profile,
        })
    }

    /// Compile one encoded launch: PTX→cubin→module→function handle, with
    /// the reflected native-resource attributes.
    pub(crate) fn compile_launch(
        &self,
        launch: &crate::encode::CudaLaunch,
    ) -> Result<crate::native::CompiledLaunch, AssemblyFailure> {
        self.target_profile
            .admits(crate::target::PtxTarget::SCALAR_BASELINE)
            .map_err(|error| {
                AssemblyFailure::from(AssemblyError::Toolchain {
                    stage: "target admission",
                    detail: error.to_string(),
                })
            })?;
        let (image, _log) =
            crate::driver::compile_image(&self.context, &launch.ptx).map_err(|detail| {
                AssemblyFailure::from(AssemblyError::Toolchain {
                    stage: "PTX linking",
                    detail,
                })
            })?;
        let _current = self.context.enter().map_err(|detail| {
            AssemblyFailure::from(AssemblyError::Preparation {
                stage: "context entry",
                detail,
            })
        })?;
        let driver = &self.context.driver;
        let mut raw = std::ptr::null_mut();
        let mut log = vec![0u8; 16384];
        let mut options = [5, 6];
        let mut values = [log.as_mut_ptr().cast::<c_void>(), log.len() as *mut c_void];
        let status = unsafe {
            (driver.module_load)(
                &mut raw,
                image.as_ptr().cast(),
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if let Err(error) = driver.check_typed(status, "native image loading") {
            let end = log.iter().position(|byte| *byte == 0).unwrap_or(log.len());
            return Err(AssemblyFailure::from(AssemblyError::Toolchain {
                stage: "native image loading",
                detail: format!("{error}\n{}", String::from_utf8_lossy(&log[..end])),
            }));
        }
        let module = Module {
            raw,
            context: self.context.clone(),
        };
        let mut function: Handle = std::ptr::null_mut();
        let function_name = CString::new(launch.name.as_str()).map_err(|_| {
            AssemblyFailure::CompilerInvariant(seismic_realization::failure::CompilerDefect::new(
                seismic_realization::failure::Package::B1Cuda,
                "a compiled launch name contains NUL",
            ))
        })?;
        unsafe {
            driver
                .check_typed(
                    (driver.module_function)(&mut function, module.raw, function_name.as_ptr()),
                    "kernel lookup",
                )
                .map_err(AssemblyError::Driver)?;
        }
        let attr = |key| -> Result<i32, AssemblyFailure> {
            let mut value = 0;
            unsafe {
                driver
                    .check_typed(
                        (driver.function_attribute)(&mut value, key, function),
                        "native function attribute",
                    )
                    .map_err(AssemblyError::Driver)?;
            }
            Ok(value)
        };
        Ok(crate::native::CompiledLaunch {
            function,
            module,
            resources: NativeResources {
                registers_per_thread: attr(4)?,
                local_bytes_per_thread: attr(3)?,
                max_threads_per_block: attr(0)?,
                shared_bytes_per_block: attr(1)?,
            },
        })
    }

    /// The per-multiprocessor block residency of one compiled launch at a
    /// block width (a capacity fact, never a performance score).
    pub(crate) fn occupancy_blocks(
        &self,
        native: &crate::native::CompiledLaunch,
        block_threads: u64,
    ) -> Result<u32, AssemblyFailure> {
        let threads = u32::try_from(block_threads).map_err(|_| {
            AssemblyFailure::CompilerInvariant(seismic_realization::failure::CompilerDefect::new(
                seismic_realization::failure::Package::B1Cuda,
                "a compiled launch's block width overflows the driver ABI",
            ))
        })?;
        // A driver-reported capacity is non-negative by the CUDA ABI; a
        // negative report is clamped to zero and caught by the residency
        // validation that follows.
        let capacity = native.resources.max_threads_per_block.max(0) as u32;
        let resident = threads.min(capacity);
        let _current = self.context.enter().map_err(|detail| {
            AssemblyFailure::from(AssemblyError::Preparation {
                stage: "context entry",
                detail,
            })
        })?;
        let driver = &self.context.driver;
        let mut active_blocks = 0;
        unsafe {
            driver
                .check_typed(
                    (driver.occupancy_blocks)(
                        &mut active_blocks,
                        native.function,
                        resident as i32,
                        0,
                    ),
                    "compiled-kernel occupancy limit",
                )
                .map_err(AssemblyError::Driver)?;
        }
        Ok(active_blocks.max(0) as u32)
    }

    /// Zero `bytes` bytes at one device address.
    fn memset(&self, pointer: u64, bytes: u64) -> Result<(), ExecutionFailure> {
        if bytes == 0 {
            return Ok(());
        }
        let _current = self.context.enter().map_err(external_driver)?;
        let driver = &self.context.driver;
        unsafe {
            driver
                .check(
                    (driver.memset_d8)(pointer, 0, bytes as usize),
                    "device memory fill",
                )
                .map_err(external_driver)?;
        }
        Ok(())
    }

    /// Copy `bytes` bytes between two device addresses.
    fn device_copy(
        &self,
        destination: u64,
        source: u64,
        bytes: u64,
    ) -> Result<(), ExecutionFailure> {
        if bytes == 0 {
            return Ok(());
        }
        let _current = self.context.enter().map_err(external_driver)?;
        let driver = &self.context.driver;
        unsafe {
            driver
                .check(
                    (driver.memcpy_device)(destination, source, bytes as usize),
                    "device-to-device copy",
                )
                .map_err(external_driver)?;
        }
        Ok(())
    }

    /// Upload one 8-byte word at a device address.
    fn write_word(&self, pointer: u64, word: u64) -> Result<(), ExecutionFailure> {
        let _current = self.context.enter().map_err(external_driver)?;
        let driver = &self.context.driver;
        unsafe {
            driver
                .check(
                    (driver.upload)(pointer, word.to_le_bytes().as_ptr().cast(), 8),
                    "device word upload",
                )
                .map_err(external_driver)?;
        }
        Ok(())
    }

    /// Download one 8-byte word from a device address.
    fn read_word(&self, pointer: u64) -> Result<u64, ExecutionFailure> {
        let mut bytes = [0u8; 8];
        let _current = self.context.enter().map_err(external_driver)?;
        let driver = &self.context.driver;
        unsafe {
            driver
                .check(
                    (driver.download)(bytes.as_mut_ptr().cast(), pointer, 8),
                    "device word download",
                )
                .map_err(external_driver)?;
        }
        Ok(u64::from_le_bytes(bytes))
    }

    /// Download one 4-byte word from a device address.
    fn read_status_word(&self, pointer: u64) -> Result<u32, ExecutionFailure> {
        let mut bytes = [0u8; 4];
        let _current = self.context.enter().map_err(external_driver)?;
        let driver = &self.context.driver;
        unsafe {
            driver
                .check(
                    (driver.download)(bytes.as_mut_ptr().cast(), pointer, 4),
                    "status word download",
                )
                .map_err(external_driver)?;
        }
        Ok(u32::from_le_bytes(bytes))
    }
}

// ---------------------------------------------------------------------------
// The executor
// ---------------------------------------------------------------------------

/// What the host runtime (R1) supplies from one `PreparedInvocation`: the
/// validated invocation values and the device pointer of every validated
/// root-ABI buffer. Prepared by `CompiledPlan::prepare`; never re-validated
/// here.
pub struct Prepared<'a> {
    pub values: &'a InvocationValues,
    /// The device pointer of one validated buffer, by contract slot.
    pub buffer_pointer: &'a dyn Fn(BufferSlot) -> u64,
}

/// One executed invocation's outcome: the compiler-owned result-block
/// words, one per `ResultFieldIx` in field order (the host runtime decodes
/// them with the plan's result-field table).
pub struct Outcome {
    pub result_words: Vec<u64>,
}

/// The executor over one device, one sealed native artifact, and its
/// physical plan. Direct dense handles only: no launch, storage, slot,
/// field, or fact is ever looked up by a fallible identifier.
pub struct Executor<'a> {
    device: &'a Device,
    artifact: &'a NativeArtifact,
    plan: &'a PhysicalPlan<Dialect>,
    /// The obligation of every sealed executor guard step, dense in
    /// `GuardIx`: the schedule's own tree order, which the pre-order walk
    /// below reproduces.
    guard_obligations: Vec<ObligationRef>,
}

impl<'a> Executor<'a> {
    pub fn new(
        device: &'a Device,
        artifact: &'a NativeArtifact,
        plan: &'a PhysicalPlan<Dialect>,
    ) -> Self {
        let mut guard_obligations = Vec::new();
        fn collect(steps: &[PhysicalStep<Dialect>], out: &mut Vec<ObligationRef>) {
            for step in steps {
                match step {
                    // `GuardIx` is dense in this walk's order.
                    PhysicalStep::Guard(guard) => out.push(guard.obligation.clone()),
                    PhysicalStep::Launch(_) | PhysicalStep::Fill(_) => {}
                    PhysicalStep::Call(call) => collect(&call.body.steps, out),
                    PhysicalStep::If(branch) => {
                        collect(&branch.then_schedule.steps, out);
                        collect(&branch.else_schedule.steps, out);
                    }
                    PhysicalStep::Repeat(repeat) => collect(&repeat.body.steps, out),
                }
            }
        }
        collect(&plan.schedule().steps, &mut guard_obligations);
        Executor {
            device,
            artifact,
            plan,
            guard_obligations,
        }
    }

    /// Execute one prepared invocation over the sealed native tree.
    pub fn execute(&self, prepared: &Prepared<'_>) -> Result<Outcome, ExecutionFailure> {
        if self.artifact.launches().is_empty() {
            // An empty schedule is the valid identity artifact.
            return Ok(Outcome {
                result_words: Vec::new(),
            });
        }
        let resources = self.plan.resources();
        let arena = Allocation::new(self.device.context(), resources.arena_bytes as usize)
            .map_err(external_allocation)?;
        let slots = Allocation::new(self.device.context(), (resources.scalar_slots as usize) * 8)
            .map_err(external_allocation)?;
        let status = Allocation::new(
            self.device.context(),
            (resources.status_bytes as usize).max(1),
        )
        .map_err(external_allocation)?;
        let results = Allocation::new(
            self.device.context(),
            (resources.result_bytes as usize).max(1),
        )
        .map_err(external_allocation)?;
        let extents = Allocation::new(
            self.device.context(),
            self.artifact.runtime_extents().len().max(1) * 8,
        )
        .map_err(external_allocation)?;
        let barriers = Allocation::new(
            self.device.context(),
            self.artifact.barrier_words().max(1) * 4,
        )
        .map_err(external_allocation)?;
        self.device.memset(status.pointer, resources.status_bytes)?;
        self.device
            .memset(barriers.pointer, (self.artifact.barrier_words() as u64) * 4)?;
        let mut run = Run {
            executor: self,
            prepared,
            arena,
            slots,
            status,
            results,
            extents,
            barriers,
            passed_guards: BTreeSet::new(),
        };
        // The runtime-extent value block: evaluate every retained runtime
        // extent once against the prepared invocation.
        for (ordinal, extent) in self.artifact.runtime_extents().iter().enumerate() {
            let value = run.eval(&self.plan.runtime_extent(*extent))?;
            run.extents
                .upload_at(ordinal * 8, &value.to_le_bytes())
                .map_err(external_allocation)?;
        }
        run.steps(self.artifact.root())?;
        // Synchronous completion: the first recorded status error wins.
        for field in self.plan.status_fields().ids() {
            let word = self
                .device
                .read_status_word(run.status.pointer + (field.index() as u64) * 4)?;
            if word != 0 {
                let status = &self.plan.status_fields()[field];
                return Err(ExecutionFailure::Safety(SafetyViolation {
                    source: SafetyViolationSource::KernelCheck(field),
                    obligation: status.obligation.clone(),
                    kind: status.kind,
                }));
            }
        }
        let result_words = self
            .plan
            .result_fields()
            .ids()
            .map(|field| {
                self.device
                    .read_word(run.results.pointer + (field.index() as u64) * 8)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Outcome { result_words })
    }
}

/// One execution's mutable state: the compiler-owned blocks and the set of
/// established executor guards.
struct Run<'a> {
    executor: &'a Executor<'a>,
    prepared: &'a Prepared<'a>,
    arena: Allocation,
    slots: Allocation,
    status: Allocation,
    results: Allocation,
    extents: Allocation,
    barriers: Allocation,
    passed_guards: BTreeSet<usize>,
}

impl Run<'_> {
    /// Interpret one subtree of the sealed native tree.
    fn steps(&mut self, steps: &[NativeStep]) -> Result<(), ExecutionFailure> {
        for step in steps {
            match step {
                NativeStep::Launch(id) => self.submit(*id)?,
                NativeStep::Guard(guard) => {
                    let holds = match &guard.predicate {
                        GuardPredicate::ProductFits { factors, bits } => {
                            let limit = if *bits >= 64 {
                                u64::MAX
                            } else {
                                (1u64 << u32::from(*bits)) - 1
                            };
                            let mut product = 1u64;
                            let mut fits = true;
                            for factor in factors {
                                let value = self.eval(factor)?;
                                product = match product.checked_mul(value) {
                                    Some(next) => next,
                                    None => {
                                        fits = false;
                                        break;
                                    }
                                };
                            }
                            fits && product <= limit
                        }
                        GuardPredicate::ExtentPositive { extent } => self.eval(extent)? > 0,
                        GuardPredicate::RangeOrdered { start, end, bound } => {
                            let (start, end, bound) =
                                (self.eval(start)?, self.eval(end)?, self.eval(bound)?);
                            start <= end && end <= bound
                        }
                    };
                    if !holds {
                        let kind = self.executor.plan.status_fields()[guard.status].kind;
                        return Err(ExecutionFailure::Safety(SafetyViolation {
                            source: SafetyViolationSource::Guard(guard.id),
                            obligation: guard.obligation.clone(),
                            kind,
                        }));
                    }
                    self.passed_guards.insert(guard.id.index());
                }
                NativeStep::Call(body) => self.steps(body)?,
                NativeStep::If {
                    condition,
                    then_steps,
                    else_steps,
                    joins,
                    ..
                } => {
                    let taken = self.scalar_source(condition)? != 0;
                    if taken {
                        self.steps(then_steps)?;
                    } else {
                        self.steps(else_steps)?;
                    }
                    // `joined` holds the taken side's value.
                    for join in joins {
                        let taken_side = if taken {
                            &join.then_value
                        } else {
                            &join.else_value
                        };
                        self.transfer(taken_side, &join.joined)?;
                    }
                }
                NativeStep::Repeat {
                    start,
                    end,
                    binder,
                    body,
                    carries,
                    ..
                } => {
                    let (start, end) = (self.eval(start)?, self.eval(end)?);
                    // The discharged `RangeOrdered` predicate (an invocation
                    // relation or the dominating guard step that already ran)
                    // established `0 <= start <= end <= bound`. `current` is
                    // rebound from `initial` before the first visit and from
                    // `update` after each visit; `result` is the final value.
                    // Carries: `current := initial` before the first visit.
                    for carry in carries {
                        self.transfer(&carry.initial, &carry.current)?;
                    }
                    for visit in start..end {
                        self.slots
                            .upload_at(binder.index() * 8, &visit.to_le_bytes())
                            .map_err(external_allocation)?;
                        self.steps(body)?;
                        // `current := update` after each visit.
                        for carry in carries {
                            self.transfer(&carry.update, &carry.current)?;
                        }
                    }
                    // `result := current` after the loop.
                    for carry in carries {
                        self.transfer(&carry.current, &carry.result)?;
                    }
                }
                NativeStep::Fill(fill) => {
                    let pointer = self.storage_pointer(fill.storage)?;
                    self.executor.device.memset(pointer, fill.bytes)?;
                }
            }
        }
        Ok(())
    }

    /// Submit one launch with its evaluated geometry; zero-work launches
    /// are skipped (zero native grids are never submitted).
    fn submit(&self, id: LaunchIx) -> Result<(), ExecutionFailure> {
        let device = self.executor.device;
        let native = self.executor.artifact.launch(id);
        let work_items = self.eval(&native.launch.work_items)?;
        if work_items == 0 {
            return Ok(());
        }
        let block = u32::try_from(self.eval(&native.launch.participants)?).map_err(|_| {
            ExecutionFailure::External(ExternalFailure {
                stage: ExternalStage::Driver,
                detail: "a launch's participant count overflows the native block width".into(),
            })
        })?;
        if block == 0 {
            return Err(ExecutionFailure::External(ExternalFailure {
                stage: ExternalStage::Driver,
                detail: "a launch resolved to zero participants".into(),
            }));
        }
        // The sealed geometry, submitted uniformly: serialized-policy
        // launches seal `workgroups = [1, 1, 1]` with `participants = 1`
        // (their single participant loops `[0, total)` internally).
        let axis = |expr: &ExecutionExpr| -> Result<u32, ExecutionFailure> {
            u32::try_from(self.eval(expr)?).map_err(|_| {
                ExecutionFailure::External(ExternalFailure {
                    stage: ExternalStage::Driver,
                    detail: "a launch's workgroup axis overflows the native grid width".into(),
                })
            })
        };
        let (grid_x, grid_y, grid_z) = (
            axis(&native.launch.workgroups[0])?,
            axis(&native.launch.workgroups[1])?,
            axis(&native.launch.workgroups[2])?,
        );
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            return Ok(());
        }
        let shared_bytes = u32::try_from(
            self.executor.plan.launches()[id.index()]
                .resources
                .workgroup_bytes,
        )
        .map_err(|_| {
            ExecutionFailure::External(ExternalFailure {
                stage: ExternalStage::Submission,
                detail: "a launch's workgroup bytes exceed the native shared-memory \
                         argument"
                    .into(),
            })
        })?;
        // The mechanical parameter list, in encoded order.
        let mut arguments = Vec::with_capacity(native.launch.params.len());
        for param in &native.launch.params {
            let value = match param {
                crate::encode::CudaParam::Storage(storage) => self.storage_pointer(*storage)?,
                crate::encode::CudaParam::Scalar { source, .. } => self.scalar_source(source)?,
                crate::encode::CudaParam::WorkTotal => work_items,
                crate::encode::CudaParam::RuntimeExtents => self.extents.pointer,
                crate::encode::CudaParam::ExecutorSlots => self.slots.pointer,
                crate::encode::CudaParam::Status => self.status.pointer,
                crate::encode::CudaParam::Results => self.results.pointer,
                crate::encode::CudaParam::GridBarrier => self.barriers.pointer,
            };
            arguments.push(value);
        }
        let mut params = arguments
            .iter_mut()
            .map(|value| (value as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        let context = device.context();
        let _current = context.enter().map_err(external_driver)?;
        let driver = &context.driver;
        let status = if native.launch.cooperative {
            unsafe {
                (driver.launch_cooperative)(
                    native.function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block,
                    1,
                    1,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        } else {
            unsafe {
                (driver.launch)(
                    native.function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block,
                    1,
                    1,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        };
        driver
            .check(status, "kernel launch")
            .map_err(external_driver)?;
        // Calling the loaded driver fn pointer is the unsafe act; the
        // private context is entered above.
        unsafe {
            driver
                .check((driver.synchronize)(), "kernel completion")
                .map_err(external_driver)?;
        }
        Ok(())
    }

    /// The device pointer of one storage: a validated root-ABI buffer, or
    /// an arena offset.
    fn storage_pointer(&self, storage: StorageIx) -> Result<u64, ExecutionFailure> {
        let mirror = &self.executor.artifact.storages()[storage.index()];
        match mirror.placement {
            StoragePlacement::Abi { slot } => Ok((self.prepared.buffer_pointer)(slot)),
            StoragePlacement::Arena { offset } => Ok(self.arena.pointer + offset),
            // Workgroup and participant storage is declared inside the
            // kernel (dynamic shared / thread-local memory) and never
            // bound as a launch parameter.
            StoragePlacement::Workgroup | StoragePlacement::Participant => {
                Err(ExecutionFailure::External(ExternalFailure {
                    stage: ExternalStage::Driver,
                    detail: format!(
                        "storage {} is kernel-local and has no device pointer",
                        storage.index()
                    ),
                }))
            }
        }
    }

    /// The 64-bit word of one scalar source.
    fn scalar_source(&self, source: &ScalarSource) -> Result<u64, ExecutionFailure> {
        let device = self.executor.device;
        match source {
            ScalarSource::Abi(slot) => Ok(self.prepared.values.scalars[*slot].bits),
            ScalarSource::Executor(slot) => {
                device.read_word(self.slots.pointer + (slot.index() as u64) * 8)
            }
            ScalarSource::Invocation(id) => Ok(self.prepared.values.derived[*id]),
            ScalarSource::Result(field) => {
                device.read_word(self.results.pointer + (field.index() as u64) * 8)
            }
        }
    }

    /// Transfer one sealed value (a join or carry): scalar sources are
    /// word copies; tensor members with distinct storages are device
    /// copies (same storage ⇒ no-op).
    fn transfer(
        &self,
        source: &SealedValue,
        destination: &SealedValue,
    ) -> Result<(), ExecutionFailure> {
        let device = self.executor.device;
        match (source, destination) {
            (SealedValue::Scalar(source), SealedValue::Scalar(destination)) => {
                let word = self.scalar_source(source)?;
                match destination {
                    ScalarSource::Executor(slot) => {
                        device.write_word(self.slots.pointer + (slot.index() as u64) * 8, word)
                    }
                    ScalarSource::Result(field) => {
                        device.write_word(self.results.pointer + (field.index() as u64) * 8, word)
                    }
                    // Root-ABI inputs and invocation values are read-only.
                    ScalarSource::Abi(_) | ScalarSource::Invocation(_) => {
                        Err(ExecutionFailure::External(ExternalFailure {
                            stage: ExternalStage::Driver,
                            detail: "a joined scalar targets a read-only source".into(),
                        }))
                    }
                }
            }
            (SealedValue::Tensor(source), SealedValue::Tensor(destination)) => {
                for (source_view, destination_view) in
                    source.as_slice().iter().zip(destination.as_slice())
                {
                    if source_view.storage == destination_view.storage {
                        continue;
                    }
                    let from = self.storage_pointer(source_view.storage)?;
                    let to = self.storage_pointer(destination_view.storage)?;
                    let bytes = self.executor.artifact.storages()[source_view.storage.index()]
                        .bytes
                        .min(
                            self.executor.artifact.storages()[destination_view.storage.index()]
                                .bytes,
                        );
                    device.device_copy(to, from, bytes)?;
                }
                Ok(())
            }
            _ => Err(ExecutionFailure::External(ExternalFailure {
                stage: ExternalStage::Driver,
                detail: "a join or carry changes its value kind".into(),
            })),
        }
    }

    /// Evaluate one retained execution expression against the prepared
    /// invocation, the folded native facts, the device result block, and
    /// the executor slot words (with the dominating guards consulted
    /// before every partial operation).
    fn eval(&self, expr: &ExecutionExpr) -> Result<u64, ExecutionFailure> {
        let device = self.executor.device;
        match expr {
            ExecutionExpr::Invocation(id) => Ok(self.prepared.values.derived[*id]),
            ExecutionExpr::NativeFact(index) => Ok(self.executor.artifact.native_fact(*index)),
            ExecutionExpr::ResultField(field) => {
                device.read_word(self.results.pointer + (field.index() as u64) * 8)
            }
            ExecutionExpr::Guarded(guarded) => {
                let values = self.prepared.values.clone();
                let slots = &self.slots;
                let results = &self.results;
                // A device read the driver fails is reported after the
                // evaluation (the frozen evaluator closures are
                // infallible).
                let read_failure: std::cell::RefCell<Option<ExecutionFailure>> =
                    std::cell::RefCell::new(None);
                let executor = |slot: ScalarSlotIx| -> u64 {
                    match device.read_word(slots.pointer + (slot.index() as u64) * 8) {
                        Ok(word) => word,
                        Err(failure) => {
                            *read_failure.borrow_mut() = Some(failure);
                            0
                        }
                    }
                };
                let result_words = |field: seismic_realization::ids::ResultFieldIx| -> u64 {
                    match device.read_word(results.pointer + (field.index() as u64) * 8) {
                        Ok(word) => word,
                        Err(failure) => {
                            *read_failure.borrow_mut() = Some(failure);
                            0
                        }
                    }
                };
                // The dominating guard step has passed (the schedule runs
                // it before every use); a guard that has not is the safety
                // violation, with its obligation from the dense guard
                // table (the schedule's own order).
                let passed = &self.passed_guards;
                let obligations = &self.executor.guard_obligations;
                let mut guards = |guard: GuardIx, kind: SafetyKind| {
                    if passed.contains(&guard.index()) {
                        Ok(())
                    } else {
                        Err(SafetyViolation {
                            source: SafetyViolationSource::Guard(guard),
                            obligation: obligations[guard.index()].clone(),
                            kind,
                        })
                    }
                };
                let evaluated = guarded.evaluate(&values, &executor, &result_words, &mut guards);
                if let Some(failure) = read_failure.into_inner() {
                    return Err(failure);
                }
                evaluated.map_err(ExecutionFailure::Safety)
            }
        }
    }
}

/// Decode one result word into its scalar value (the host runtime pairs
/// the words with the plan's result-field dtypes).
pub fn decode_result_word(word: u64, dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::from_bits(word as u32) as f64,
        DType::I32 => f64::from((word as u32) as i32),
        DType::U32 => f64::from(word as u32),
        DType::Bool => f64::from((word & 1) as u8),
        DType::F16 => seismic_lang::numeric::f16_to_f32((word & 0xffff) as u16) as f64,
        DType::BF16 => f32::from_bits(((word & 0xffff) as u32) << 16) as f64,
    }
}

/// Re-exported for the host runtime: one validated scalar word.
pub use seismic_realization::invocation::ScalarWord as ValidatedScalarWord;

/// The word of one validated scalar in its ABI representation.
pub fn scalar_word(word: &ScalarWord) -> u64 {
    word.bits
}
