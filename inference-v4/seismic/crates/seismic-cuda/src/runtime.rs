use crate::driver::{Allocation, Context, Driver, Event, Handle, Module};
use seismic_realization::executable::{AbiRole, ResolvedStorageId};
use std::{
    collections::BTreeMap,
    ffi::{c_void, CStr, CString},
    fmt,
    rc::Rc,
};

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
}
/// A private context, deliberately thread-affine. Resource owners retain this
/// context; no allocation or module can outlive its driver.
pub struct Device {
    context: Rc<Context>,
    pub info: DeviceInfo,
    target_profile: crate::target::TargetProfile,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeResources {
    pub registers_per_thread: i32,
    pub local_bytes_per_thread: i32,
    pub max_threads_per_block: i32,
    pub shared_bytes_per_block: i32,
    /// Driver occupancy limit for this compiled function and explicit block size;
    /// a capacity constraint, not observed occupancy or a performance score.
    pub max_active_blocks_per_multiprocessor: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeFinalizationError {
    /// PTX generation/linking or image loading failed. This is not evidence
    /// that another physical assignment is legal.
    CodeGeneration { stage: &'static str, detail: String },
    /// A CUDA driver operation failed independently of assignment resources.
    Driver(crate::driver::DriverError),
    /// Host-side runtime preparation failed after successful native emission.
    RuntimePreparation { stage: &'static str, detail: String },
    /// Native output contradicted a guarantee already proven by the physical
    /// program and encoded in PTX launch bounds. This is a compiler/driver
    /// defect, never a reason to select another physical assignment.
    Invariant { stage: &'static str, detail: String },
}

impl fmt::Display for NativeFinalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeGeneration { stage, detail } => {
                write!(f, "CUDA {stage} failed: {detail}")
            }
            Self::Driver(error) => error.fmt(f),
            Self::RuntimePreparation { stage, detail } => {
                write!(f, "CUDA {stage} failed: {detail}")
            }
            Self::Invariant { stage, detail } => {
                write!(f, "CUDA native invariant `{stage}` failed: {detail}")
            }
        }
    }
}

impl std::error::Error for NativeFinalizationError {}

fn validate_native_invariants(
    planned_threads_per_block: u32,
    planned_static_shared_bytes: u32,
    native: &NativeResources,
) -> Result<(), NativeFinalizationError> {
    if native.max_threads_per_block < 0
        || planned_threads_per_block > native.max_threads_per_block as u32
    {
        return Err(NativeFinalizationError::Invariant {
            stage: "launch bound",
            detail: format!(
                "planned .maxntid requires {planned_threads_per_block} threads; loaded function permits {}",
                native.max_threads_per_block
            ),
        });
    }
    if native.shared_bytes_per_block < 0
        || native.shared_bytes_per_block as u32 != planned_static_shared_bytes
    {
        return Err(NativeFinalizationError::Invariant {
            stage: "static shared memory",
            detail: format!(
                "physical program planned {planned_static_shared_bytes} bytes; loaded function reports {}",
                native.shared_bytes_per_block
            ),
        });
    }
    if native.max_active_blocks_per_multiprocessor == 0 {
        return Err(NativeFinalizationError::Invariant {
            stage: "kernel residency",
            detail: format!(
                "a launch-bounded kernel has zero residency ({} registers/thread, {} local bytes/thread, {} shared bytes/block)",
                native.registers_per_thread,
                native.local_bytes_per_thread,
                native.shared_bytes_per_block
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod native_finalization_tests {
    use super::*;

    fn resources() -> NativeResources {
        NativeResources {
            registers_per_thread: 64,
            local_bytes_per_thread: 4096,
            max_threads_per_block: 128,
            shared_bytes_per_block: 0,
            max_active_blocks_per_multiprocessor: 0,
        }
    }

    #[test]
    fn lower_native_thread_limit_is_a_fatal_invariant() {
        let error = validate_native_invariants(256, 0, &resources()).unwrap_err();
        assert!(matches!(error, NativeFinalizationError::Invariant { .. }));
        assert!(error.to_string().contains("permits 128"));
    }

    #[test]
    fn code_generation_failure_is_not_a_resource_refinement() {
        let error = NativeFinalizationError::CodeGeneration {
            stage: "PTX linking",
            detail: "invalid instruction".into(),
        };
        assert!(matches!(
            error,
            NativeFinalizationError::CodeGeneration { .. }
        ));
    }

    #[test]
    fn unexpected_static_shared_memory_is_a_fatal_invariant() {
        let mut native = resources();
        native.max_threads_per_block = 256;
        native.max_active_blocks_per_multiprocessor = 1;
        native.shared_bytes_per_block = 16;
        let error = validate_native_invariants(256, 0, &native).unwrap_err();
        assert!(matches!(error, NativeFinalizationError::Invariant { .. }));
        assert!(error.to_string().contains("planned 0 bytes"));
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
    /// Alignment of the retained allocation's actual device address. Subviews
    /// retain this base fact; their offsets are checked separately by accounting.
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
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len {
            return Err("CUDA buffer view exceeds its parent".into());
        }
        Ok(Self {
            allocation: self.allocation.clone(),
            offset: self
                .offset
                .checked_add(range.start)
                .ok_or("CUDA view offset overflow")?,
            len: range.end - range.start,
        })
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > self.len {
            return Err("CUDA write exceeds buffer view".into());
        }
        self.allocation.upload_at(self.offset, bytes)
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), String> {
        if bytes.len() > self.len {
            return Err("CUDA read exceeds buffer view".into());
        }
        self.allocation.download_at(self.offset, bytes)
    }
    fn pointer(&self) -> u64 {
        self.allocation.pointer + self.offset as u64
    }
}
impl Device {
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String> {
        Ok(Buffer {
            allocation: Rc::new(Allocation::new(&self.context, bytes)?),
            offset: 0,
            len: bytes,
        })
    }
    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let buffer = self.buffer(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    pub fn open(ordinal: i32) -> Result<Self, String> {
        let driver = Driver::load()?;
        let mut device = 0;
        unsafe {
            driver.check(
                (driver.device_get)(&mut device, ordinal),
                "device selection",
            )?;
        }
        let attribute = |key| -> Result<u32, String> {
            let mut n = 0;
            unsafe {
                driver.check(
                    (driver.device_attribute)(&mut n, key, device),
                    "device attribute",
                )?;
            }
            u32::try_from(n).map_err(|_| "invalid negative device attribute".into())
        };
        let major = attribute(75)?;
        let minor = attribute(76)?;
        let mut name = [0 as std::ffi::c_char; 256];
        let mut version = 0;
        let mut memory = 0usize;
        unsafe {
            driver.check(
                (driver.device_name)(name.as_mut_ptr(), name.len() as i32, device),
                "device name",
            )?;
            driver.check((driver.driver_version)(&mut version), "driver version")?;
            driver.check(
                (driver.device_total_memory)(&mut memory, device),
                "total device memory",
            )?;
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
        };
        let target_profile = crate::target::TargetProfile::from_observation(
            crate::target::TargetObservation::driver(info.compute_capability, info.driver_version)
                .map_err(|error| error.to_string())?,
            crate::target::TargetLimits {
                max_threads_per_block: info.max_threads_per_block,
                max_grid_x: info.max_grid_x,
                warp_size: info.warp_size,
                max_scratch_bytes: info.global_memory_bytes / 4,
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            context: Context::new(driver, device)?,
            info,
            target_profile,
        })
    }
    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }
    /// Compile the immutable result of physical resolution. No mapping,
    /// allocation, target, or launch decision is made at this boundary.
    pub fn compile_emitted(
        &self,
        emitted: crate::native::Emitted,
    ) -> Result<PhysicalSequence, NativeFinalizationError> {
        if emitted.launches.is_empty() {
            return Err(NativeFinalizationError::Invariant {
                stage: "physical launch sequence",
                detail: "resolved CUDA artifact has no launches".into(),
            });
        }
        let mut owned = BTreeMap::new();
        let mut external = Vec::new();
        for allocation in &emitted.abi {
            match allocation.role {
                Some(AbiRole::Parameter { .. } | AbiRole::Result { .. }) => {
                    external.push(allocation.clone());
                }
                Some(AbiRole::InvocationResource { .. }) | None => {
                    let bytes = usize::try_from(allocation.bytes).map_err(|_| {
                        NativeFinalizationError::RuntimePreparation {
                            stage: "resolved allocation",
                            detail: format!(
                                "allocation#{} exceeds host address range",
                                allocation.id.0
                            ),
                        }
                    })?;
                    let allocation_handle =
                        Allocation::new(&self.context, bytes).map_err(|detail| {
                            NativeFinalizationError::RuntimePreparation {
                                stage: "resolved allocation",
                                detail,
                            }
                        })?;
                    owned.insert(allocation.id, Rc::new(allocation_handle));
                }
            }
        }
        external.sort_by_key(|allocation| match allocation.role.as_ref().unwrap() {
            AbiRole::Parameter {
                ordinal,
                path,
                representation_plane,
            } => (0, *ordinal, path.clone(), representation_plane.clone()),
            AbiRole::Result {
                ordinal,
                path,
                representation_plane,
            } => (1, *ordinal, path.clone(), representation_plane.clone()),
            AbiRole::InvocationResource { name } => (2, 0, vec![], Some(name.clone())),
        });
        let phases = emitted
            .launches
            .iter()
            .map(|launch| self.compile_physical_launch(launch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PhysicalSequence {
            name: emitted.name,
            phases,
            external,
            owned,
        })
    }

    fn compile_physical_launch(
        &self,
        launch: &crate::native::CudaLaunch,
    ) -> Result<PhysicalKernel, NativeFinalizationError> {
        self.target_profile.admits(launch.target).map_err(|error| {
            NativeFinalizationError::CodeGeneration {
                stage: "target admission",
                detail: error.to_string(),
            }
        })?;
        if launch.block != launch.launch_bound {
            return Err(NativeFinalizationError::Invariant {
                stage: "launch bound",
                detail: "PTX .maxntid differs from resolved block geometry".into(),
            });
        }
        let threads = launch
            .block
            .into_iter()
            .try_fold(1u64, |value, axis| value.checked_mul(axis))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| NativeFinalizationError::Invariant {
                stage: "launch geometry",
                detail: "resolved CUDA block size overflows the driver ABI".into(),
            })?;
        if launch
            .grid
            .into_iter()
            .any(|axis| u32::try_from(axis).is_err())
        {
            return Err(NativeFinalizationError::Invariant {
                stage: "launch geometry",
                detail: "resolved CUDA grid exceeds the driver ABI".into(),
            });
        }
        let (image, compilation_log) = crate::driver::compile_image(&self.context, &launch.ptx)
            .map_err(|detail| NativeFinalizationError::CodeGeneration {
                stage: "PTX linking",
                detail,
            })?;
        let _current =
            self.context
                .enter()
                .map_err(|detail| NativeFinalizationError::RuntimePreparation {
                    stage: "context entry",
                    detail,
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
            return Err(NativeFinalizationError::CodeGeneration {
                stage: "native image loading",
                detail: format!("{error}\n{}", String::from_utf8_lossy(&log[..end])),
            });
        }
        let module = Module {
            raw,
            context: self.context.clone(),
        };
        let mut function = std::ptr::null_mut();
        let function_name =
            CString::new(launch.name.as_str()).map_err(|_| NativeFinalizationError::Invariant {
                stage: "kernel lookup",
                detail: "resolved CUDA launch name contains NUL".into(),
            })?;
        unsafe {
            driver
                .check_typed(
                    (driver.module_function)(&mut function, module.raw, function_name.as_ptr()),
                    "kernel lookup",
                )
                .map_err(NativeFinalizationError::Driver)?;
        }
        let attr = |key| -> Result<i32, NativeFinalizationError> {
            let mut value = 0;
            unsafe {
                driver
                    .check_typed(
                        (driver.function_attribute)(&mut value, key, function),
                        "native function attribute",
                    )
                    .map_err(NativeFinalizationError::Driver)?;
            }
            Ok(value)
        };
        let mut native = NativeResources {
            registers_per_thread: attr(4)?,
            local_bytes_per_thread: attr(3)?,
            max_threads_per_block: attr(0)?,
            shared_bytes_per_block: attr(1)?,
            max_active_blocks_per_multiprocessor: 0,
        };
        let mut active_blocks = 0;
        unsafe {
            driver
                .check_typed(
                    (driver.occupancy_blocks)(&mut active_blocks, function, threads as i32, 0),
                    "compiled-kernel occupancy limit",
                )
                .map_err(NativeFinalizationError::Driver)?;
        }
        native.max_active_blocks_per_multiprocessor = u32::try_from(active_blocks).unwrap_or(0);
        validate_native_invariants(threads, 0, &native)?;
        Ok(PhysicalKernel {
            module,
            function,
            launch: launch.clone(),
            native,
            image: NativeImage {
                cubin: image,
                compilation_log,
                driver_version: self.info.driver_version,
                compute_capability: self.info.compute_capability,
                target: launch.target,
                target_fingerprint: self.target_profile.fingerprint().to_owned(),
            },
            timing: [
                Event::new(&self.context).map_err(|detail| {
                    NativeFinalizationError::RuntimePreparation {
                        stage: "timing-event creation",
                        detail,
                    }
                })?,
                Event::new(&self.context).map_err(|detail| {
                    NativeFinalizationError::RuntimePreparation {
                        stage: "timing-event creation",
                        detail,
                    }
                })?,
            ],
        })
    }
}
/// The exact linked image loaded by this kernel. Developer inspection can use
/// CUDA tooling, but compilation/execution depend only on the installed driver.
pub struct NativeImage {
    pub cubin: Vec<u8>,
    pub compilation_log: String,
    pub driver_version: i32,
    pub compute_capability: (i32, i32),
    pub target: crate::target::PtxTarget,
    pub target_fingerprint: String,
}

/// A natively compiled resolved kernel. Its launch geometry and argument order
/// are copied from the physical artifact and are immutable after compilation.
pub struct PhysicalKernel {
    module: Module,
    function: Handle,
    launch: crate::native::CudaLaunch,
    pub native: NativeResources,
    pub image: NativeImage,
    timing: [Event; 2],
}

/// Runtime owner for the direct physical CUDA path. Callers bind only
/// Parameter/Result allocations; compiler-owned Invocation/Internal storage is
/// allocated once and never exposed as user ABI.
pub struct PhysicalSequence {
    name: String,
    phases: Vec<PhysicalKernel>,
    external: Vec<crate::native::CudaAbiAllocation>,
    owned: BTreeMap<ResolvedStorageId, Rc<Allocation>>,
}

impl PhysicalSequence {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }

    pub fn external_allocations(&self) -> &[crate::native::CudaAbiAllocation] {
        &self.external
    }

    pub fn ptx_sources(&self) -> impl Iterator<Item = &str> {
        self.phases.iter().map(|phase| phase.launch.ptx.as_str())
    }

    pub fn native_images(&self) -> impl Iterator<Item = &NativeImage> {
        self.phases.iter().map(|phase| &phase.image)
    }

    pub fn execute(&mut self, buffers: &[Buffer], timed: bool) -> Result<Option<f64>, String> {
        self.execute_with_scalars(buffers, &[], timed)
    }

    pub fn execute_with_scalars(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
        timed: bool,
    ) -> Result<Option<f64>, String> {
        if buffers.len() != self.external.len() {
            return Err(format!(
                "CUDA physical sequence needs {} external allocations, received {}",
                self.external.len(),
                buffers.len()
            ));
        }
        let mut pointers = BTreeMap::new();
        for (spec, buffer) in self.external.iter().zip(buffers) {
            if !Rc::ptr_eq(&buffer.allocation.context, &self.phases[0].module.context) {
                return Err("CUDA buffer belongs to a different context".into());
            }
            if buffer.len < spec.bytes as usize {
                return Err(format!(
                    "CUDA allocation#{} needs {} bytes, has {}",
                    spec.id.0, spec.bytes, buffer.len
                ));
            }
            if spec.alignment == 0 || buffer.pointer() % spec.alignment != 0 {
                return Err(format!(
                    "CUDA allocation#{} violates alignment {}",
                    spec.id.0, spec.alignment
                ));
            }
            pointers.insert(spec.id, buffer.pointer());
        }
        pointers.extend(
            self.owned
                .iter()
                .map(|(id, allocation)| (*id, allocation.pointer)),
        );
        if !scalars.is_empty() {
            return Err("CUDA physical sequence has no scalar invocation storage".into());
        }
        let mut seconds = 0.0;
        for phase in &mut self.phases {
            seconds += phase.launch(&pointers, timed)?;
        }
        Ok(timed.then_some(seconds))
    }
}

impl PhysicalKernel {
    fn launch(
        &mut self,
        allocations: &BTreeMap<ResolvedStorageId, u64>,
        timed: bool,
    ) -> Result<f64, String> {
        let mut arguments = self
            .launch
            .bindings
            .iter()
            .map(|binding| {
                allocations
                    .get(&binding.allocation)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "CUDA launch#{} binding#{} names absent allocation#{}",
                            self.launch.id.0, binding.id.0, binding.allocation.0
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut params = arguments
            .iter_mut()
            .map(|value| (value as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        let grid = self.launch.grid.map(|axis| axis as u32);
        let block = self.launch.block.map(|axis| axis as u32);
        let context = &self.module.context;
        let _current = context.enter()?;
        let driver = &context.driver;
        unsafe {
            if timed {
                driver.check(
                    (driver.event_record)(self.timing[0].raw, std::ptr::null_mut()),
                    "start timing event",
                )?;
            }
            driver.check(
                (driver.launch)(
                    self.function,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "kernel launch",
            )?;
            if timed {
                driver.check(
                    (driver.event_record)(self.timing[1].raw, std::ptr::null_mut()),
                    "end timing event",
                )?;
            }
            driver.check((driver.synchronize)(), "kernel completion")?;
        }
        if !timed {
            return Ok(0.0);
        }
        let mut milliseconds = 0.0f32;
        unsafe {
            driver.check(
                (driver.event_elapsed)(&mut milliseconds, self.timing[0].raw, self.timing[1].raw),
                "elapsed event time",
            )?;
        }
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err("invalid CUDA event duration".into());
        }
        Ok(f64::from(milliseconds) * 0.001)
    }
}
