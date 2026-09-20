use crate::driver::{Allocation, Context, Driver, Event, Handle, Module};
use seismic_lang::types::{DType, RuntimeExtentId};
use seismic_realization::executable::{self as rz};
use seismic_realization::{encode_scalars, validate_alias_rules};
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
    /// Compile the native artifact: each encoded launch's PTX is compiled
    /// once through the driver, and the reflected native-resource facts
    /// instantiate the selected bounded geometry contract (`min(preferred,
    /// native_max)` per the plan's contract; a reflected fact outside the
    /// declared domain is a fatal invariant, never a retry.
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
        let kernels = emitted
            .launches
            .iter()
            .map(|launch| self.compile_physical_launch(launch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PhysicalSequence {
            name: emitted.name,
            kernels,
            storages: emitted.storages.clone(),
            abi_buffers: emitted.abi_buffers.clone(),
            abi_scalar_fields: emitted.abi_scalar_fields.clone(),
            abi_scalar_bytes: emitted.abi_scalar_bytes,
            status_words: emitted.status_words,
            slot_words: emitted.slot_words,
            arena_bytes: emitted.arena_bytes,
            runtime_extents: emitted.runtime_extents.clone(),
            alias_rules: emitted.alias_rules.clone(),
            root: emitted.root.clone(),
        })
    }

    fn compile_physical_launch(
        &self,
        launch: &crate::native::CudaLaunch,
    ) -> Result<PhysicalKernel, NativeFinalizationError> {
        self.target_profile
            .admits(crate::target::PtxTarget::SCALAR_BASELINE)
            .map_err(|error| NativeFinalizationError::CodeGeneration {
                stage: "target admission",
                detail: error.to_string(),
            })?;
        let threads =
            u32::try_from(launch.block).map_err(|_| NativeFinalizationError::Invariant {
                stage: "launch geometry",
                detail: "resolved CUDA block size overflows the driver ABI".into(),
            })?;
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
        // The selected contract's admissible domain: at least one resident
        // participant, and the native maximum admits the planned width.
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
                target: crate::target::PtxTarget::SCALAR_BASELINE,
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

/// The exact linked image loaded by this kernel.
pub struct NativeImage {
    pub cubin: Vec<u8>,
    pub compilation_log: String,
    pub driver_version: i32,
    pub compute_capability: (i32, i32),
    pub target: crate::target::PtxTarget,
    pub target_fingerprint: String,
}

/// A natively compiled resolved kernel. Its launch geometry, parameter
/// list, and retained work condition are copied from the encoded artifact
/// and are immutable after compilation.
pub struct PhysicalKernel {
    module: Module,
    function: Handle,
    launch: crate::native::CudaLaunch,
    pub native: NativeResources,
    pub image: NativeImage,
    timing: [Event; 2],
}

/// Runtime owner for the resolved CUDA plan. Callers bind only the root
/// ABI buffers; the internal arena, executor slot block, status block, and
/// result scalar block are compiler-owned and never exposed as user ABI.
/// Execution interprets the retained structured tree — no retry or
/// candidate surface exists.
pub struct PhysicalSequence {
    name: String,
    kernels: Vec<PhysicalKernel>,
    storages: BTreeMap<u64, crate::native::StorageMirror>,
    abi_buffers: Vec<rz::BufferBinding>,
    abi_scalar_fields: Vec<(String, DType, usize)>,
    abi_scalar_bytes: usize,
    status_words: u64,
    slot_words: u64,
    arena_bytes: u64,
    runtime_extents: Vec<(RuntimeExtentId, rz::ExecutionExpr)>,
    alias_rules: Vec<rz::AliasRule>,
    root: Vec<crate::native::EncodedBodyStep>,
}

/// One invocation's runtime state: bound buffers, the scalar block, the
/// internal arena, and the compiler-owned slot/status/result blocks.
struct Invocation {
    pointers: BTreeMap<u64, u64>,
    scalar_words: Vec<u64>,
    arena: Allocation,
    slots: Allocation,
    status: Allocation,
    results: Allocation,
    extents: BTreeMap<u32, u64>,
}

/// Evaluate one retained execution expression against an invocation.
fn evaluate(expr: &rz::ExecutionExpr, invocation: &Invocation) -> Result<u64, String> {
    use rz::ExecutionExpr as Expr;
    Ok(match expr {
        Expr::Const(value) => *value,
        Expr::Extent(id) => *invocation
            .extents
            .get(&id.0)
            .ok_or_else(|| format!("runtime extent#{} is unevaluated", id.0))?,
        Expr::AbiScalar { .. } | Expr::ExecutorScalar(_) => {
            return Err(
                "compiler bug: an extent expression references an unevaluated scalar".into(),
            )
        }
        Expr::Add(left, right) => evaluate(left, invocation)?
            .checked_add(evaluate(right, invocation)?)
            .ok_or("geometry addition overflow")?,
        Expr::Sub(left, right) => evaluate(left, invocation)?
            .checked_sub(evaluate(right, invocation)?)
            .ok_or("geometry subtraction overflow")?,
        Expr::Mul(left, right) => evaluate(left, invocation)?
            .checked_mul(evaluate(right, invocation)?)
            .ok_or("geometry multiplication overflow")?,
        Expr::CeilDiv(left, right) => {
            let (left, right) = (evaluate(left, invocation)?, evaluate(right, invocation)?);
            left.div_ceil(right)
        }
        Expr::Div(left, right) => evaluate(left, invocation)?
            .checked_div(evaluate(right, invocation)?)
            .ok_or("geometry division overflow")?,
        Expr::Rem(left, right) => evaluate(left, invocation)?
            .checked_rem(evaluate(right, invocation)?)
            .ok_or("geometry remainder division by zero")?,
        Expr::Min(left, right) => evaluate(left, invocation)?.min(evaluate(right, invocation)?),
    })
}

impl PhysicalSequence {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn launch_count(&self) -> usize {
        self.kernels.len()
    }

    pub fn abi_buffers(&self) -> &[rz::BufferBinding] {
        &self.abi_buffers
    }

    /// The compiler-owned result scalar block of one execution.
    pub fn result_bytes(&self) -> usize {
        self.abi_scalar_fields.len().max(1) * 8
    }

    pub fn ptx_sources(&self) -> impl Iterator<Item = &str> {
        self.kernels.iter().map(|kernel| kernel.launch.ptx.as_str())
    }

    pub fn native_images(&self) -> impl Iterator<Item = &NativeImage> {
        self.kernels.iter().map(|kernel| &kernel.image)
    }

    /// Execute one invocation: validate the root ABI alias rules against
    /// the bound byte ranges, allocate the compiler-owned blocks, evaluate
    /// the retained runtime extents, interpret the structured execution
    /// tree, skip zero-work launches, and report the first recorded error
    /// after synchronous completion.
    pub fn execute(
        &mut self,
        buffers: &[Buffer],
        scalar_values: &[f64],
        timed: bool,
    ) -> Result<Option<NativeStatus>, String> {
        use seismic_lang::abi::ScalarParameter;
        if buffers.len() != self.abi_buffers.len() {
            return Err(format!(
                "CUDA plan needs {} ABI buffers, received {}",
                self.abi_buffers.len(),
                buffers.len()
            ));
        }
        let schema: Vec<ScalarParameter> = self
            .abi_scalar_fields
            .iter()
            .map(|(name, dtype, _)| ScalarParameter::plain(name.clone(), *dtype))
            .collect();
        let scalar_words = encode_scalars(&schema, scalar_values)
            .map_err(|error| format!("CUDA scalar encoding: {error}"))?;
        if scalar_values.len() != self.abi_scalar_fields.len() {
            return Err(format!(
                "CUDA plan needs {} ABI scalars, received {}",
                self.abi_scalar_fields.len(),
                scalar_values.len()
            ));
        }
        // Root-ABI alias validation over actual byte ranges.
        let locate = |binding: rz::BufferBindingId| -> Result<Option<(u64, u64, u64)>, String> {
            let Some(index) = self
                .abi_buffers
                .iter()
                .position(|candidate| candidate.binding == binding)
            else {
                return Err(format!(
                    "the invocation supplies no buffer for binding {}",
                    binding.0
                ));
            };
            let Some(buffer) = buffers.get(index) else {
                // A result binding: runtime-allocated, disjoint by construction.
                return Ok(None);
            };
            Ok(Some((
                buffer.identity(),
                buffer.offset(),
                buffer.len() as u64,
            )))
        };
        let abi = rz::ResolvedAbi::<crate::physical::CudaDialect> {
            buffers: self.abi_buffers.clone(),
            scalars: seismic_lang::abi::ScalarLayout::words(&schema)?,
            results: Vec::new(),
            alias_rules: self.alias_rules.clone(),
            status: None,
        };
        validate_alias_rules(&abi, locate)
            .map_err(|error| format!("CUDA invocation alias contract: {error}"))?;
        let mut invocation = self.prepare(buffers, &scalar_words)?;
        self.run_step(&self.root.clone(), &mut invocation, timed)?;
        self.finish(&mut invocation)
    }

    fn prepare(&self, buffers: &[Buffer], scalar_words: &[u64]) -> Result<Invocation, String> {
        let context = self
            .kernels
            .first()
            .map(|kernel| kernel.module_context())
            .ok_or("the plan has no kernel")?;
        let arena = Allocation::new(&context, self.arena_bytes as usize)
            .map_err(|error| format!("CUDA internal arena: {error}"))?;
        let slots = Allocation::new(&context, (self.slot_words as usize) * 8)
            .map_err(|error| format!("CUDA executor slot block: {error}"))?;
        let status = Allocation::new(&context, (self.status_words as usize) * 4)
            .map_err(|error| format!("CUDA status block: {error}"))?;
        let results = Allocation::new(&context, self.result_bytes())
            .map_err(|error| format!("CUDA result block: {error}"))?;
        let mut pointers = BTreeMap::new();
        for (storage, mirror) in &self.storages {
            let pointer = match mirror.placement {
                rz::ResolvedStoragePlacement::Abi { binding } => {
                    let index = self
                        .abi_buffers
                        .iter()
                        .position(|candidate| candidate.binding == binding)
                        .ok_or_else(|| "an ABI storage names an absent buffer".to_string())?;
                    let buffer = buffers
                        .get(index)
                        .ok_or_else(|| "an ABI buffer is not bound".to_string())?;
                    if buffer.len() as u64 + 1 <= mirror.bytes && mirror.bytes > 0 {
                        return Err(format!(
                            "CUDA ABI buffer needs {} bytes, has {}",
                            mirror.bytes,
                            buffer.len()
                        ));
                    }
                    if mirror.alignment > 1 && buffer.pointer() % mirror.alignment != 0 {
                        return Err(format!(
                            "CUDA ABI buffer violates alignment {}",
                            mirror.alignment
                        ));
                    }
                    buffer.pointer()
                }
                rz::ResolvedStoragePlacement::Arena { offset } => arena.pointer + offset,
                rz::ResolvedStoragePlacement::Workgroup
                | rz::ResolvedStoragePlacement::Participant => continue,
            };
            pointers.insert(*storage, pointer);
        }
        let mut invocation = Invocation {
            pointers,
            scalar_words: scalar_words.to_vec(),
            arena,
            slots,
            status,
            results,
            extents: BTreeMap::new(),
        };
        // Evaluate every retained runtime extent once (their expressions
        // may reference ABI scalars or device slot words).
        for (id, expr) in &self.runtime_extents {
            let value = evaluate(expr, &invocation)?;
            invocation.extents.insert(id.0, value);
        }
        Ok(invocation)
    }

    /// Interpret one subtree of the structured execution tree.
    fn run_step(
        &self,
        steps: &[crate::native::EncodedBodyStep],
        invocation: &mut Invocation,
        timed: bool,
    ) -> Result<(), String> {
        for step in steps {
            match step {
                crate::native::EncodedBodyStep::Launch(id) => {
                    let kernel = self
                        .kernels
                        .iter()
                        .find(|kernel| kernel.launch.id == *id)
                        .ok_or_else(|| format!("launch#{} is absent", id.0))?;
                    kernel.submit(invocation, timed)?;
                }
                crate::native::EncodedBodyStep::Call(body) => {
                    self.run_step(body, invocation, timed)?;
                }
                crate::native::EncodedBodyStep::If {
                    condition,
                    then_steps,
                    else_steps,
                } => {
                    let taken = self.control_value(condition, invocation)?;
                    if taken != 0 {
                        self.run_step(then_steps, invocation, timed)?;
                    } else {
                        self.run_step(else_steps, invocation, timed)?;
                    }
                }
                crate::native::EncodedBodyStep::Repeat {
                    start,
                    end,
                    binder,
                    body,
                } => {
                    let (start, end) = (
                        self.control_value(start, invocation)?,
                        self.control_value(end, invocation)?,
                    );
                    for visit in start..end {
                        invocation.write_slot(*binder, visit)?;
                        self.run_step(body, invocation, timed)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Read one control scalar (an ABI word, a device slot word, or a
    /// resolved constant).
    fn control_value(
        &self,
        source: &crate::native::ControlSource,
        invocation: &Invocation,
    ) -> Result<u64, String> {
        Ok(match source {
            crate::native::ControlSource::Const(value) => {
                u64::from_ne_bytes((*value as i64).to_ne_bytes())
            }
            crate::native::ControlSource::Abi { offset, .. } => {
                invocation.scalar_word(*offset as usize)?
            }
            crate::native::ControlSource::Slot(slot) => invocation.read_slot(*slot)?,
            crate::native::ControlSource::Computed(expr) => evaluate(expr, invocation)?,
        })
    }

    /// Synchronous completion: download the status words and report the
    /// first recorded error with its discharged-field identity.
    fn finish(&self, invocation: &mut Invocation) -> Result<Option<NativeStatus>, String> {
        if self.status_words == 0 {
            return Ok(None);
        }
        let mut words = vec![0u8; (self.status_words as usize) * 4];
        invocation
            .status
            .download_at(0, &mut words)
            .map_err(|error| format!("CUDA status download: {error}"))?;
        for (ordinal, word) in words.chunks_exact(4).enumerate() {
            let code = u32::from_le_bytes(word.try_into().expect("a status word"));
            if code != 0 {
                return Ok(Some(NativeStatus {
                    field: ordinal as u64,
                    kind: code,
                }));
            }
        }
        Ok(None)
    }
}

/// The first recorded status error of one execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeStatus {
    /// The status field that recorded the first error.
    pub field: u64,
    /// The recorded kind code.
    pub kind: u32,
}

impl Invocation {
    fn scalar_word(&self, offset: usize) -> Result<u64, String> {
        let index = offset / 8;
        self.scalar_words
            .get(index)
            .copied()
            .ok_or_else(|| format!("ABI scalar word #{index} is absent"))
    }

    fn read_slot(&self, slot: u64) -> Result<u64, String> {
        let mut bytes = [0u8; 8];
        self.slots
            .download_at((slot as usize) * 8, &mut bytes)
            .map_err(|error| format!("CUDA executor slot read: {error}"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn write_slot(&self, slot: u64, value: u64) -> Result<(), String> {
        self.slots
            .upload_at((slot as usize) * 8, &value.to_ne_bytes())
            .map_err(|error| format!("CUDA binder slot write: {error}"))
    }
}

impl Buffer {
    fn identity(&self) -> u64 {
        self.allocation.pointer
    }
    fn offset(&self) -> u64 {
        self.offset as u64
    }
}

impl PhysicalKernel {
    fn module_context(&self) -> Rc<Context> {
        self.module.context.clone()
    }

    /// Submit one launch with its evaluated geometry: zero-work launches
    /// are skipped (zero native grids are never submitted).
    fn submit(&self, invocation: &Invocation, timed: bool) -> Result<(), String> {
        let work_items = self.work_items(invocation)?;
        if work_items == 0 {
            return Ok(());
        }
        let block = self.launch.block;
        let grid = work_items.div_ceil(block);
        let mut arguments = Vec::new();
        for param in &self.launch.params {
            let value = match param {
                crate::native::CudaParam::Storage(storage) => invocation
                    .pointers
                    .get(&storage.0)
                    .copied()
                    .ok_or_else(|| format!("storage#{} is unbound", storage.0))?,
                crate::native::CudaParam::AbiScalar { offset, .. } => {
                    invocation.scalar_word(*offset as usize)?
                }
                crate::native::CudaParam::Extent(id) => *invocation
                    .extents
                    .get(&id.0)
                    .ok_or_else(|| format!("runtime extent#{} is unevaluated", id.0))?,
                crate::native::CudaParam::SlotBlock => invocation.slots.pointer,
                crate::native::CudaParam::StatusBlock => invocation.status.pointer,
                crate::native::CudaParam::ResultBlock => invocation.results.pointer,
                crate::native::CudaParam::MathCall { .. } => {
                    return Err("compiler bug: a software call reached the driver boundary".into())
                }
            };
            arguments.push(value);
        }
        let mut params = arguments
            .iter_mut()
            .map(|value| (value as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
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
                    grid as u32,
                    1,
                    1,
                    block as u32,
                    1,
                    1,
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
        Ok(())
    }

    /// The retained zero-work launch condition of this launch, evaluated
    /// against the invocation's extents and slot words.
    fn work_items(&self, invocation: &Invocation) -> Result<u64, String> {
        evaluate(&self.launch.work_items, invocation)
    }
}
