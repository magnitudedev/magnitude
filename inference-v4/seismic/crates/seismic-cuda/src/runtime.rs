use crate::{
    driver::{Allocation, Context, Driver, Event, Handle, Module},
    ptx,
    execution::{Execution, Limits},
};
use seismic_lang::abi::ScalarParameter;
use seismic_realization::{BufferSpec, ScalarProgram};
use std::{
    ffi::{c_void, CStr},
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
}
#[derive(Clone, Debug)]
pub struct NativeResources {
    pub registers_per_thread: i32,
    pub local_bytes_per_thread: i32,
    pub max_threads_per_block: i32,
    pub shared_bytes_per_block: i32,
    /// Driver occupancy limit for this compiled function and explicit block size;
    /// a capacity constraint, not observed occupancy or a performance score.
    pub max_active_blocks_per_multiprocessor: u32,
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
        if address == 0 { 1 } else { 1u64 << address.trailing_zeros() }
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
        if major < 8 {
            return Err(format!(
                "CUDA scalar PTX baseline requires SM 8.0 or newer; device is {major}.{minor}"
            ));
        }
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
        Ok(Self {
            context: Context::new(driver, device)?,
            info,
        })
    }
    /// Consume the complete selected launch sequence, without re-preparing IR.
    pub fn compile_executions(&self, executions: Vec<Execution>) -> Result<Sequence, String> {
        if executions.is_empty() {
            return Err("CUDA execution sequence must have at least one phase".into());
        }
        let first = executions[0].program();
        for execution in &executions {
            execution.validate_limits(self.execution_limits())?;
            if execution.program().public_buffer_count != first.public_buffer_count
                || execution.program().buffers != first.buffers
                || execution.program().scalars != first.scalars {
                return Err("CUDA phases must share the same invocation bindings".into());
            }
        }
        let public_buffers = first.buffers[..first.public_buffer_count].to_vec();
        let internal = first.buffers[first.public_buffer_count..].iter()
            .map(|spec| self.buffer(spec.bytes)).collect::<Result<Vec<_>, _>>()?;
        let phases = executions.into_iter()
            .map(|execution| self.compile_execution(execution))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Sequence { phases, public_buffers, internal })
    }
    fn execution_limits(&self) -> Limits {
        Limits { max_threads_per_block: self.info.max_threads_per_block, max_grid_x: self.info.max_grid_x }
    }
    fn compile_code(&self, execution: &Execution) -> Result<CompiledCode, String> {
        execution.validate_limits(self.execution_limits())?;
        let threads_per_block = execution.dispatch().threads_per_group as u32;
        let source = ptx::print(execution.target_plan());
        let context = &self.context;
        let (image, compilation_log) = crate::driver::compile_image(context, &source)?;
        let _current = context.enter()?;
        let driver = &context.driver;
        let mut raw = std::ptr::null_mut();
        let mut log = vec![0u8; 16384];
        let mut options = [5, 6]; // CU_JIT_ERROR_LOG_BUFFER, *_SIZE_BYTES
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
        if let Err(error) = driver.check(status, "native image loading") {
            let end = log.iter().position(|b| *b == 0).unwrap_or(log.len());
            return Err(format!("{error}\n{}", String::from_utf8_lossy(&log[..end])));
        }
        let module = Module {
            raw,
            context: context.clone(),
        };
        let mut function = std::ptr::null_mut();
        unsafe {
            driver.check(
                (driver.module_function)(&mut function, module.raw, c"seismic_kernel".as_ptr()),
                "kernel lookup",
            )?;
        }
        let attr = |key| -> Result<i32, String> {
            let mut n = 0;
            unsafe {
                driver.check(
                    (driver.function_attribute)(&mut n, key, function),
                    "native function attribute",
                )?;
            }
            Ok(n)
        };
        let mut active_blocks = 0;
        unsafe {
            driver.check(
                (driver.occupancy_blocks)(
                    &mut active_blocks,
                    function,
                    threads_per_block as i32,
                    0,
                ),
                "compiled-kernel occupancy limit",
            )?;
        }
        let active_blocks = u32::try_from(active_blocks)
            .ok()
            .filter(|n| *n > 0)
            .ok_or("compiled CUDA candidate cannot reside on an execution unit")?;
        let native = NativeResources {
            registers_per_thread: attr(4)?,
            local_bytes_per_thread: attr(3)?,
            max_threads_per_block: attr(0)?,
            shared_bytes_per_block: attr(1)?,
            max_active_blocks_per_multiprocessor: active_blocks,
        };
        if threads_per_block > native.max_threads_per_block as u32 {
            return Err("CUDA block size exceeds compiled kernel capability".into());
        }
        Ok(CompiledCode {
            module,
            function,
            native,
            source,
            image: NativeImage {
                cubin: image,
                compilation_log,
                driver_version: self.info.driver_version,
                compute_capability: self.info.compute_capability,
            },
        })
    }
    /// Compile and allocate exactly the prevalidated selected execution.
    pub fn compile_execution(&self, execution: Execution) -> Result<Kernel, String> {
        let CompiledCode { module, function, native, source, image } = self.compile_code(&execution)?;
        let context = &self.context;
        let storage = execution.storage();
        let table = Allocation::new(context, storage.buffer_table_bytes)?;
        let scalars = Allocation::new(context, storage.scalar_bytes)?;
        let scratch = Allocation::new(context, storage.scratch_bytes)?;
        let statuses = Allocation::new(context, storage.status_bytes)?;
        Ok(Kernel {
            module,
            function,
            execution,
            tensors: Vec::new(),
            table,
            scalars,
            scratch,
            statuses,
            native,
            ptx: source,
            image,
            ready: false,
            timing: [Event::new(context)?, Event::new(context)?],
        })
    }
}
struct CompiledCode {
    module: Module,
    function: Handle,
    native: NativeResources,
    source: String,
    image: NativeImage,
}
/// Synchronous initial invocation owner. Launch returns only after completion;
/// all submitted buffers, code and scratch therefore survive every device use.
pub struct Kernel {
    module: Module,
    function: Handle,
    execution: Execution,
    tensors: Vec<Buffer>,
    table: Allocation,
    scalars: Allocation,
    scratch: Allocation,
    statuses: Allocation,
    ready: bool,
    timing: [Event; 2],
    pub native: NativeResources,
    pub ptx: String,
    image: NativeImage,
}
/// The exact linked image loaded by this kernel. Developer inspection can use
/// CUDA tooling, but compilation/execution depend only on the installed driver.
pub struct NativeImage {
    pub cubin: Vec<u8>,
    pub compilation_log: String,
    pub driver_version: i32,
    pub compute_capability: (i32, i32),
}
impl Kernel {
    pub fn native_image(&self) -> &NativeImage {
        &self.image
    }
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.execution.program().buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        &self.execution.program().scalars
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes
    }
    pub fn work_items(&self) -> u64 {
        self.execution.program().work_items
    }
    fn validate(&self, buffers: &[&mut [u8]]) -> Result<(), String> {
        if buffers.len() != self.execution.program().buffers.len() {
            return Err("CUDA buffer binding count mismatch".into());
        }
        for (buffer, spec) in buffers.iter().zip(&self.execution.program().buffers) {
            if buffer.len() < spec.bytes {
                return Err(format!(
                    "CUDA buffer {}.{} has {} bytes; needs {}",
                    spec.parameter,
                    spec.plane,
                    buffer.len(),
                    spec.bytes
                ));
            }
        }
        Ok(())
    }
    /// Bind resident views without copying their contents. Rebinding updates only
    /// the pointer/scalar tables; ownership remains retained through every launch.
    pub fn bind(&mut self, buffers: &[Buffer], scalars: &[f64]) -> Result<(), String> {
        self.ready = false;
        if buffers.len() != self.execution.program().buffers.len() {
            return Err("CUDA resident binding count mismatch".into());
        }
        for (buffer, spec) in buffers.iter().zip(&self.execution.program().buffers) {
            if !Rc::ptr_eq(&buffer.allocation.context, &self.module.context) {
                return Err("CUDA buffer belongs to a different context".into());
            }
            if buffer.len < spec.bytes {
                return Err(format!(
                    "CUDA resident {}.{} needs {} bytes, has {}",
                    spec.parameter, spec.plane, spec.bytes, buffer.len
                ));
            }
            if spec.alignment == 0 || buffer.pointer() % spec.alignment as u64 != 0 {
                return Err("CUDA resident view violates typed storage alignment".into());
            }
        }
        self.execution.program().conditions.validate_aliases(&self.execution.program().buffers, |i| (0, buffers[i].pointer()))?;
        let words = seismic_realization::encode_scalars(&self.execution.program().scalars, scalars)?;
        let bytes = words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        self.scalars.upload(&bytes)?;
        let pointers = buffers
            .iter()
            .flat_map(|b| b.pointer().to_le_bytes())
            .collect::<Vec<_>>();
        self.table.upload(&pointers)?;
        self.tensors = buffers.to_vec();
        self.ready = true;
        Ok(())
    }
    pub fn upload(&mut self, buffers: &[&mut [u8]], scalars: &[f64]) -> Result<(), String> {
        self.ready = false;
        self.validate(buffers)?;
        // The first host invocation supplies storage. Subsequent uploads refresh
        // the current resident bindings rather than allocating per invocation.
        if self.tensors.is_empty() && !self.execution.program().buffers.is_empty() {
            self.tensors = self.execution.program()
                .buffers
                .iter()
                .map(|spec| {
                    Ok(Buffer {
                        allocation: Rc::new(Allocation::new(&self.module.context, spec.bytes)?),
                        offset: 0,
                        len: spec.bytes,
                    })
                })
                .collect::<Result<_, String>>()?;
        }
        let resident = self.tensors.clone();
        // Validate scalar values before modifying caller-owned resident contents.
        seismic_realization::encode_scalars(&self.execution.program().scalars, scalars)?;
        for ((tensor, buffer), spec) in resident.iter().zip(buffers).zip(&self.execution.program().buffers) {
            tensor.write(&buffer[..spec.bytes])?;
        }
        self.bind(&resident, scalars)
    }
    /// Release completed invocation storage while retaining reusable code.
    pub fn release_bindings(&mut self) {
        self.ready = false;
        self.tensors.clear();
    }
    pub fn launch(&mut self) -> Result<(), String> {
        self.launch_inner(false).map(|_| ())
    }
    /// CUDA event interval around one kernel in this private context. Excludes
    /// uploads, status initialization/download and host synchronization. The
    /// caller owns input reset and the conditioning/repetition protocol.
    pub fn launch_timed(&mut self) -> Result<f64, String> {
        self.launch_inner(true)
    }
    fn launch_inner(&mut self, timed: bool) -> Result<f64, String> {
        if !self.ready {
            return Err("CUDA kernel must have validated inputs uploaded before launch".into());
        }
        if self.execution.dispatch().groups as u32 == 0 {
            return Ok(0.0);
        }
        self.ready = false;
        let context = &self.module.context;
        let _current = context.enter()?;
        self.statuses.fill(255)?;
        let mut arguments = [
            self.table.pointer,
            self.scalars.pointer,
            self.scratch.pointer,
            self.statuses.pointer,
        ];
        let mut params = arguments
            .iter_mut()
            .map(|v| (v as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        let driver = &context.driver;
        unsafe {
            if timed {
                driver.check(
                    (driver.event_record)(self.timing[0].raw, std::ptr::null_mut()),
                    "start timing event",
                )?;
            }
            let launch = (driver.launch)(
                self.function,
                self.execution.dispatch().groups as u32,
                1,
                1,
                self.execution.dispatch().threads_per_group as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            // Even a submission error may report an earlier asynchronous error.
            // Drain the private context before any owner can release resources.
            let end_event = if timed {
                (driver.event_record)(self.timing[1].raw, std::ptr::null_mut())
            } else {
                0
            };
            let completion = (driver.synchronize)();
            driver.check(launch, "kernel launch")?;
            driver.check(completion, "kernel completion")?;
            driver.check(end_event, "end timing event")?;
        }
        let mut statuses = vec![0u8; self.statuses.bytes];
        self.statuses.download(&mut statuses)?;
        for (index, bytes) in statuses.chunks_exact(4).enumerate() {
            let status = i32::from_le_bytes(bytes.try_into().unwrap());
            if status != 0 {
                self.ready = false;
                return Err(format!("CUDA work item {index} failed with status {status}; outputs may be partially written"));
            }
        }
        let mut milliseconds = 0.0f32;
        if timed {
            unsafe {
                driver.check(
                    (driver.event_elapsed)(
                        &mut milliseconds,
                        self.timing[0].raw,
                        self.timing[1].raw,
                    ),
                    "elapsed event time",
                )?;
            }
            if !milliseconds.is_finite() || milliseconds < 0.0 {
                return Err("invalid CUDA event duration".into());
            }
        }
        self.ready = true;
        Ok(f64::from(milliseconds) * 0.001)
    }
    pub fn download(&self, buffers: &mut [&mut [u8]]) -> Result<(), String> {
        self.validate(buffers)?;
        if self.tensors.len() != self.execution.program().buffers.len() {
            return Err("CUDA tensors have not been bound".into());
        }
        for ((tensor, buffer), spec) in self.tensors.iter().zip(buffers).zip(&self.execution.program().buffers)
        {
            tensor.read(&mut buffer[..spec.bytes])?;
        }
        Ok(())
    }
    pub fn run(&mut self, buffers: &mut [&mut [u8]], scalars: &[f64]) -> Result<(), String> {
        self.upload(buffers, scalars)?;
        self.launch()?;
        self.download(buffers)
    }
}

/// An ordered realization of one source kernel's parallel phases. All bindings
/// validate before the first phase; failures release retained input owners only
/// after physical completion. Successful earlier writes may remain after failure.
pub struct Sequence {
    phases: Vec<Kernel>,
    public_buffers: Vec<BufferSpec>,
    /// One invocation-owned publication allocation per retained cross-phase
    /// value. Every phase receives the same suffix of the pointer table.
    internal: Vec<Buffer>,
}
impl Sequence {
    pub fn buffers(&self) -> &[BufferSpec] {
        &self.public_buffers
    }
    pub fn scalars(&self) -> &[ScalarParameter] {
        self.phases[0].scalars()
    }
    pub fn ptx_sources(&self) -> impl Iterator<Item = &str> {
        self.phases.iter().map(|kernel| kernel.ptx.as_str())
    }
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
    pub fn realizations(&self) -> impl Iterator<Item = (&ScalarProgram, &NativeResources)> {
        self.phases
            .iter()
            .map(|kernel| (kernel.execution.program(), &kernel.native))
    }
    /// Device timing is the sum of per-kernel event intervals, excluding host gaps.
    pub fn execute(
        &mut self,
        buffers: &[Buffer],
        scalars: &[f64],
        timed: bool,
    ) -> Result<Option<f64>, String> {
        if buffers.len() != self.public_buffers.len() {
            return Err("CUDA sequence public binding count mismatch".into());
        }
        let bindings = buffers.iter().chain(&self.internal).cloned().collect::<Vec<_>>();
        let result = (|| {
            for kernel in &mut self.phases {
                kernel.bind(&bindings, scalars)?;
            }
            let mut seconds = 0.0;
            for (index, kernel) in self.phases.iter_mut().enumerate() {
                if timed {
                    seconds += kernel
                        .launch_timed()
                        .map_err(|error| format!("CUDA phase {index}: {error}"))?;
                } else {
                    kernel
                        .launch()
                        .map_err(|error| format!("CUDA phase {index}: {error}"))?;
                }
            }
            Ok(timed.then_some(seconds))
        })();
        for kernel in &mut self.phases {
            kernel.release_bindings();
        }
        result
    }
}
