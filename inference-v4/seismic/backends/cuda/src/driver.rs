//! Minimal dynamically loaded CUDA driver ABI. Symbols are resolved once per
//! process; every owned object keeps its driver and primary context alive.
//! No CUDA headers or libraries are required to build this crate, so it
//! builds on hosts without CUDA and fails at device discovery instead.
//!
//! Every wrapper here has one precondition: it is called with valid handles
//! it produced itself. Violating that from Seismic's own code is a bug of
//! this module (§13.3.3); the driver's own failures are typed
//! [`DriverError`]s.
//!
//! Contexts: the backend uses the device's *primary* context, retained by
//! ordinal. The current-context binding is thread-local driver state, so
//! every entry point makes the context current for the duration of the call
//! ([`Context::enter`]); the context object itself may be shared between
//! threads, which is why the handles below are `Send + Sync`.

use libloading::Library;
use std::ffi::{c_char, c_int, c_uchar, c_uint, c_void, CStr};
use std::fmt;
use std::sync::{Arc, OnceLock};

pub type Handle = *mut c_void;
type ResultCode = c_int;

macro_rules! driver {
    (
        $( $field:ident: $ty:ty => $symbol:literal ),* $(,)?;
        optional { $( $optional:ident: $optional_ty:ty => $optional_symbol:literal ),* $(,)? }
    ) => {
        /// Required symbols fail driver loading. Optional symbols belong to
        /// newer driver API versions; their absence is reported by the one
        /// operation that needs them.
        pub(crate) struct Driver {
            $(pub $field:$ty,)*
            $(pub $optional: Option<$optional_ty>,)*
            _library: Library,
        }
        impl Driver {
            fn load_library() -> Result<Self, String> {
                #[cfg(target_os = "windows")]
                let names = &["nvcuda.dll"];
                #[cfg(not(target_os = "windows"))]
                let names = &["libcuda.so.1"];
                let mut errors = Vec::new();
                for name in names {
                    // CUDA driver symbols have the documented C ABI. Keeping the
                    // library in `Driver` preserves every copied function pointer.
                    let library = match unsafe { Library::new(name) } {
                        Ok(library) => library,
                        Err(error) => {
                            errors.push(error.to_string());
                            continue;
                        }
                    };
                    unsafe {
                        $(let $field: $ty = *library
                            .get(concat!($symbol, "\0").as_bytes())
                            .map_err(|e| format!("CUDA driver symbol {}: {e}", $symbol))?;)*
                        $(let $optional: Option<$optional_ty> = library
                            .get::<$optional_ty>(concat!($optional_symbol, "\0").as_bytes())
                            .ok()
                            .map(|symbol| *symbol);)*
                        let driver = Self { $($field,)* $($optional,)* _library: library };
                        driver.check((driver.init)(0), "initialization").map_err(|e| e.to_string())?;
                        return Ok(driver);
                    }
                }
                Err(format!("CUDA driver unavailable: {}", errors.join("; ")))
            }
        }
    }
}

driver! {
    init: unsafe extern "system" fn(c_uint) -> ResultCode => "cuInit",
    device_count: unsafe extern "system" fn(*mut c_int) -> ResultCode => "cuDeviceGetCount",
    device_get: unsafe extern "system" fn(*mut c_int, c_int) -> ResultCode => "cuDeviceGet",
    device_name: unsafe extern "system" fn(*mut c_char, c_int, c_int) -> ResultCode => "cuDeviceGetName",
    device_attribute: unsafe extern "system" fn(*mut c_int, c_int, c_int) -> ResultCode => "cuDeviceGetAttribute",
    device_total_memory: unsafe extern "system" fn(*mut usize, c_int) -> ResultCode => "cuDeviceTotalMem_v2",
    memory_info: unsafe extern "system" fn(*mut usize, *mut usize) -> ResultCode => "cuMemGetInfo_v2",
    driver_version: unsafe extern "system" fn(*mut c_int) -> ResultCode => "cuDriverGetVersion",
    primary_context_retain: unsafe extern "system" fn(*mut Handle, c_int) -> ResultCode => "cuDevicePrimaryCtxRetain",
    primary_context_release: unsafe extern "system" fn(c_int) -> ResultCode => "cuDevicePrimaryCtxRelease_v2",
    context_get: unsafe extern "system" fn(*mut Handle) -> ResultCode => "cuCtxGetCurrent",
    context_set: unsafe extern "system" fn(Handle) -> ResultCode => "cuCtxSetCurrent",
    stream_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuStreamCreate",
    stream_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuStreamDestroy_v2",
    stream_synchronize: unsafe extern "system" fn(Handle) -> ResultCode => "cuStreamSynchronize",
    event_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuEventCreate",
    event_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventDestroy_v2",
    event_record: unsafe extern "system" fn(Handle, Handle) -> ResultCode => "cuEventRecord",
    event_synchronize: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventSynchronize",
    event_query: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventQuery",
    event_elapsed_time: unsafe extern "system" fn(*mut f32, Handle, Handle) -> ResultCode => "cuEventElapsedTime",
    allocate: unsafe extern "system" fn(*mut u64, usize) -> ResultCode => "cuMemAlloc_v2",
    free: unsafe extern "system" fn(u64) -> ResultCode => "cuMemFree_v2",
    allocate_host: unsafe extern "system" fn(*mut *mut c_void, usize) -> ResultCode => "cuMemAllocHost_v2",
    free_host: unsafe extern "system" fn(*mut c_void) -> ResultCode => "cuMemFreeHost",
    host_alloc: unsafe extern "system" fn(*mut *mut c_void, usize, c_uint) -> ResultCode => "cuMemHostAlloc",
    host_device_pointer: unsafe extern "system" fn(*mut u64, *mut c_void, c_uint) -> ResultCode => "cuMemHostGetDevicePointer_v2",
    upload: unsafe extern "system" fn(u64, *const c_void, usize) -> ResultCode => "cuMemcpyHtoD_v2",
    upload_async: unsafe extern "system" fn(u64, *const c_void, usize, Handle) -> ResultCode => "cuMemcpyHtoDAsync_v2",
    download: unsafe extern "system" fn(*mut c_void, u64, usize) -> ResultCode => "cuMemcpyDtoH_v2",
    memcpy_device_async: unsafe extern "system" fn(u64, u64, usize, Handle) -> ResultCode => "cuMemcpyDtoDAsync_v2",
    memset_d8_async: unsafe extern "system" fn(u64, c_uchar, usize, Handle) -> ResultCode => "cuMemsetD8Async",
    memset_d16_async: unsafe extern "system" fn(u64, u16, usize, Handle) -> ResultCode => "cuMemsetD16Async",
    memset_d32_async: unsafe extern "system" fn(u64, c_uint, usize, Handle) -> ResultCode => "cuMemsetD32Async",
    module_load: unsafe extern "system" fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> ResultCode => "cuModuleLoadDataEx",
    module_unload: unsafe extern "system" fn(Handle) -> ResultCode => "cuModuleUnload",
    module_function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> ResultCode => "cuModuleGetFunction",
    link_create: unsafe extern "system" fn(c_uint, *mut c_int, *mut *mut c_void, *mut Handle) -> ResultCode => "cuLinkCreate_v2",
    link_add_data: unsafe extern "system" fn(Handle, c_int, *mut c_void, usize, *const c_char, c_uint, *mut c_int, *mut *mut c_void) -> ResultCode => "cuLinkAddData_v2",
    link_complete: unsafe extern "system" fn(Handle, *mut *mut c_void, *mut usize) -> ResultCode => "cuLinkComplete",
    link_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuLinkDestroy",
    function_attribute: unsafe extern "system" fn(*mut c_int, c_int, Handle) -> ResultCode => "cuFuncGetAttribute",
    function_set_attribute: unsafe extern "system" fn(Handle, c_int, c_int) -> ResultCode => "cuFuncSetAttribute",
    occupancy_max_active_blocks: unsafe extern "system" fn(*mut c_int, Handle, c_int, usize) -> ResultCode => "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    launch: unsafe extern "system" fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> ResultCode => "cuLaunchKernel",
    launch_cooperative: unsafe extern "system" fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> ResultCode => "cuLaunchCooperativeKernel",
    error_string: unsafe extern "system" fn(ResultCode, *mut *const c_char) -> ResultCode => "cuGetErrorString";
    optional {
        // Driver API 11.4+: identifies the exposed device, including a MIG
        // partition, rather than its parent GPU.
        device_uuid_v2: unsafe extern "system" fn(*mut [u8; 16], c_int) -> ResultCode => "cuDeviceGetUuid_v2",
        // Driver API 12.0+: explicitly built kernel graphs (direct-native replay).
        graph_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuGraphCreate",
        graph_add_kernel_node: unsafe extern "system" fn(*mut Handle, Handle, *const Handle, usize, *const KernelNodeParams) -> ResultCode => "cuGraphAddKernelNode_v2",
        graph_instantiate: unsafe extern "system" fn(*mut Handle, Handle, u64) -> ResultCode => "cuGraphInstantiateWithFlags",
        graph_launch: unsafe extern "system" fn(Handle, Handle) -> ResultCode => "cuGraphLaunch",
        graph_exec_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuGraphExecDestroy",
        graph_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuGraphDestroy",
    }
}

// The driver's function pointers and the loaded library are plain data; the
// CUDA driver API is documented thread-safe.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<Result<Arc<Driver>, String>> = OnceLock::new();

impl Driver {
    /// The process-wide driver, loaded on first use. A host without a usable
    /// driver reports the loader's message.
    pub fn load() -> Result<Arc<Self>, String> {
        DRIVER
            .get_or_init(|| Self::load_library().map(Arc::new))
            .clone()
    }

    pub fn check(&self, result: ResultCode, operation: &'static str) -> Result<(), DriverError> {
        if result == 0 {
            return Ok(());
        }
        let mut text = std::ptr::null();
        let description = unsafe {
            if (self.error_string)(result, &mut text) == 0 && !text.is_null() {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            } else {
                "unknown driver error".into()
            }
        };
        Err(DriverError {
            operation,
            code: result,
            description,
        })
    }

    /// One `cuDeviceGetAttribute` query.
    pub fn attribute(&self, key: c_int, device: c_int) -> Result<i32, DriverError> {
        let mut value = 0;
        unsafe {
            self.check(
                (self.device_attribute)(&mut value, key, device),
                "device attribute",
            )?;
        }
        Ok(value)
    }
}

/// One failed driver call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverError {
    pub operation: &'static str,
    pub code: i32,
    pub description: String,
}

impl DriverError {
    /// CUDA result codes that mean the device or its context is gone
    /// (`CUDA_ERROR_DEINITIALIZED`, `_ILLEGAL_ADDRESS`, `_LAUNCH_TIMEOUT`,
    /// `_LAUNCH_FAILED`, `_ASSERT`, `_HARDWARE_STACK_ERROR`,
    /// `_ILLEGAL_INSTRUCTION`, `_MISALIGNED_ADDRESS`, `_INVALID_ADDRESS_SPACE`,
    /// `_INVALID_PC`, `_ECC_UNCORRECTABLE`, `_DEVICE_UNAVAILABLE`).
    pub fn is_device_loss(&self) -> bool {
        matches!(
            self.code,
            4 | 46 | 214 | 700 | 702 | 710 | 714 | 715 | 716 | 717 | 718 | 719
        )
    }
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA {}: {} ({})",
            self.operation, self.description, self.code
        )
    }
}

impl std::error::Error for DriverError {}

/// The retained primary context of one device ordinal.
pub(crate) struct Context {
    pub driver: Arc<Driver>,
    raw: Handle,
    ordinal: c_int,
}

// The raw handle is an opaque driver object usable from any thread once made
// current there (`enter`).
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn retain(driver: Arc<Driver>, ordinal: c_int) -> Result<Arc<Self>, DriverError> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            driver.check(
                (driver.primary_context_retain)(&mut raw, ordinal),
                "primary context retain",
            )?;
        }
        Ok(Arc::new(Self {
            driver,
            raw,
            ordinal,
        }))
    }

    pub fn ordinal(&self) -> c_int {
        self.ordinal
    }

    /// Makes this context current on the calling thread until the guard
    /// drops, restoring the previous binding afterwards.
    pub fn enter(&self) -> Result<Current<'_>, DriverError> {
        let mut previous = std::ptr::null_mut();
        unsafe {
            self.driver.check(
                (self.driver.context_get)(&mut previous),
                "current context query",
            )?;
            self.driver
                .check((self.driver.context_set)(self.raw), "set current context")?;
        }
        Ok(Current {
            driver: &self.driver,
            previous,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.primary_context_release)(self.ordinal);
        }
    }
}

/// One explicit CUDA stream bound to the opened production context. The
/// executor and profiler never rely on legacy-default-stream process state.
pub(crate) struct Stream {
    raw: Handle,
    context: Arc<Context>,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new(context: &Arc<Context>) -> Result<Arc<Self>, DriverError> {
        let _current = context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            context.driver.check(
                (context.driver.stream_create)(&mut raw, 0),
                "stream creation",
            )?;
        }
        Ok(Arc::new(Self {
            raw,
            context: context.clone(),
        }))
    }

    pub fn raw(&self) -> Handle {
        self.raw
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.stream_synchronize)(self.raw),
                "stream synchronization",
            )
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.stream_destroy)(self.raw);
            }
        }
    }
}

/// CUDA event pair timing uses the device event timebase. Elapsed values are
/// returned in nanoseconds without discarding the driver's fractional
/// millisecond resolution; uncertainty is attached by profile acquisition.
pub(crate) struct Event {
    raw: Handle,
    context: Arc<Context>,
}

impl Event {
    pub fn new(context: &Arc<Context>) -> Result<Self, DriverError> {
        let _current = context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check((context.driver.event_create)(&mut raw, 0), "event creation")?;
        }
        Ok(Self {
            raw,
            context: context.clone(),
        })
    }

    pub fn record(&self, stream: &Stream) -> Result<(), DriverError> {
        assert!(
            Arc::ptr_eq(&self.context, stream.context()),
            "Event::record precondition: event and stream belong to one opened CUDA context"
        );
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.event_record)(self.raw, stream.raw()),
                "event record",
            )
        }
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.event_synchronize)(self.raw),
                "event synchronization",
            )
        }
    }

    /// `true` once every command recorded before the event has completed.
    pub fn query(&self) -> Result<bool, DriverError> {
        let _current = self.context.enter()?;
        // CUDA_ERROR_NOT_READY
        const NOT_READY: c_int = 600;
        let status = unsafe { (self.context.driver.event_query)(self.raw) };
        if status == NOT_READY {
            return Ok(false);
        }
        self.context
            .driver
            .check(status, "event query")
            .map(|()| true)
    }

    pub fn elapsed_ns(start: &Self, end: &Self) -> Result<f64, DriverError> {
        assert!(
            Arc::ptr_eq(&start.context, &end.context),
            "Event::elapsed_ns precondition: events belong to one opened CUDA context"
        );
        let _current = start.context.enter()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            start.context.driver.check(
                (start.context.driver.event_elapsed_time)(&mut milliseconds, start.raw, end.raw),
                "event elapsed time",
            )?;
        }
        Ok(f64::from(milliseconds) * 1_000_000.0)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.event_destroy)(self.raw);
            }
        }
    }
}

pub(crate) struct Current<'a> {
    driver: &'a Driver,
    previous: Handle,
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.driver.context_set)(self.previous);
        }
    }
}

/// One device allocation, freed with its context.
pub(crate) struct Allocation {
    pub pointer: u64,
    pub bytes: usize,
    pub context: Arc<Context>,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    pub fn new(context: &Arc<Context>, bytes: usize) -> Result<Self, DriverError> {
        let _current = context.enter()?;
        let mut pointer = 0;
        unsafe {
            context.driver.check(
                (context.driver.allocate)(&mut pointer, bytes.max(1)),
                "allocation",
            )?;
        }
        Ok(Self {
            pointer,
            bytes,
            context: context.clone(),
        })
    }

    /// Precondition (this module's own): `offset + bytes.len() <= self.bytes`.
    /// The generic runtime binds buffers whose extents the executable derived
    /// from the same layout roots, so a violation is a bug of this crate.
    pub fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.bytes),
            "Allocation::upload_at precondition: [{offset}, {offset}+{}) exceeds {} bytes",
            bytes.len(),
            self.bytes
        );
        if bytes.is_empty() {
            return Ok(());
        }
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.upload)(
                    self.pointer + offset as u64,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                ),
                "upload",
            )
        }
    }

    /// Precondition (this module's own): `offset + bytes.len() <= self.bytes`.
    pub fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.bytes),
            "Allocation::download_at precondition: [{offset}, {offset}+{}) exceeds {} bytes",
            bytes.len(),
            self.bytes
        );
        if bytes.is_empty() {
            return Ok(());
        }
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.download)(
                    bytes.as_mut_ptr().cast(),
                    self.pointer + offset as u64,
                    bytes.len(),
                ),
                "download",
            )
        }
    }
}

/// `CUDA_KERNEL_NODE_PARAMS_v2`.
#[repr(C)]
pub(crate) struct KernelNodeParams {
    pub function: Handle,
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared_bytes: c_uint,
    pub parameters: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kernel: Handle,
    pub context: Handle,
}

fn graph_symbol<F: Copy>(symbol: Option<F>) -> Result<F, DriverError> {
    symbol.ok_or(DriverError {
        operation: "kernel graph",
        code: 0,
        description: "the CUDA driver predates driver API 12.0 kernel graphs".into(),
    })
}

/// A kernel graph under construction: kernel nodes, each depending on the
/// previous one, so it executes in stream order.
pub(crate) struct Graph {
    raw: Handle,
    last: Option<Handle>,
    context: Arc<Context>,
}

impl Graph {
    pub fn new(context: &Arc<Context>) -> Result<Self, DriverError> {
        let create = graph_symbol(context.driver.graph_create)?;
        let _current = context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check(create(&mut raw, 0), "kernel graph creation")?;
        }
        Ok(Self {
            raw,
            last: None,
            context: context.clone(),
        })
    }

    /// Append a kernel node after every node added before it. The driver
    /// copies the parameter values the node's `parameters` point to.
    pub fn push_kernel(&mut self, node: &KernelNodeParams) -> Result<(), DriverError> {
        let add = graph_symbol(self.context.driver.graph_add_kernel_node)?;
        let _current = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        let dependencies = self.last.as_slice();
        unsafe {
            self.context.driver.check(
                add(
                    &mut raw,
                    self.raw,
                    dependencies.as_ptr(),
                    dependencies.len(),
                    node,
                ),
                "kernel graph node",
            )?;
        }
        self.last = Some(raw);
        Ok(())
    }

    pub fn instantiate(self) -> Result<GraphExec, DriverError> {
        let instantiate = graph_symbol(self.context.driver.graph_instantiate)?;
        let _current = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            self.context.driver.check(
                instantiate(&mut raw, self.raw, 0),
                "kernel graph instantiation",
            )?;
        }
        Ok(GraphExec {
            raw,
            context: self.context.clone(),
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let (Some(destroy), Ok(_current)) = (self.context.driver.graph_destroy, self.context.enter()) {
            unsafe {
                destroy(self.raw);
            }
        }
    }
}

/// An instantiated kernel graph. Launches of it already queued complete
/// even if it is destroyed.
pub(crate) struct GraphExec {
    raw: Handle,
    context: Arc<Context>,
}

// Graph executables are driver objects usable from any thread with their
// context current; launches of one are serialized by the owner.
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<(), DriverError> {
        let launch = graph_symbol(self.context.driver.graph_launch)?;
        let _current = self.context.enter()?;
        unsafe {
            self.context
                .driver
                .check(launch(self.raw, stream.raw()), "kernel graph launch")
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let (Some(destroy), Ok(_current)) =
            (self.context.driver.graph_exec_destroy, self.context.enter())
        {
            unsafe {
                destroy(self.raw);
            }
        }
    }
}

/// Pinned host memory mapped into the device address space. The host writes
/// and reads it directly; kernels address it through `device`. A host write
/// made before a launch is visible to that launch, so filling it needs no
/// driver call and never waits for the device stream.
pub(crate) struct HostMapped {
    host: *mut u8,
    pub device: u64,
    pub bytes: usize,
    context: Arc<Context>,
}

// The CUDA driver owns the pinned allocation; host access is plain memory
// access, and the owner serializes host access against device use.
unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    pub fn new(context: &Arc<Context>, bytes: usize) -> Result<Self, DriverError> {
        // CU_MEMHOSTALLOC_DEVICEMAP
        const DEVICE_MAP: c_uint = 2;
        let _current = context.enter()?;
        let mut host = std::ptr::null_mut();
        unsafe {
            context.driver.check(
                (context.driver.host_alloc)(&mut host, bytes.max(1), DEVICE_MAP),
                "mapped host allocation",
            )?;
            // Owned from here, so a failed mapping query frees it.
            let mut mapped = Self {
                host: host.cast(),
                device: 0,
                bytes,
                context: context.clone(),
            };
            context.driver.check(
                (context.driver.host_device_pointer)(&mut mapped.device, host, 0),
                "mapped host device pointer",
            )?;
            Ok(mapped)
        }
    }

    /// Precondition (this module's own): `offset + bytes.len() <= self.bytes`.
    pub fn write_at(&self, offset: usize, bytes: &[u8]) {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.bytes),
            "HostMapped::write_at precondition: [{offset}, {offset}+{}) exceeds {} bytes",
            bytes.len(),
            self.bytes
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host.add(offset), bytes.len());
        }
    }

    /// Precondition (this module's own): `offset + bytes.len() <= self.bytes`.
    pub fn read_at(&self, offset: usize, bytes: &mut [u8]) {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.bytes),
            "HostMapped::read_at precondition: [{offset}, {offset}+{}) exceeds {} bytes",
            bytes.len(),
            self.bytes
        );
        unsafe {
            std::ptr::copy_nonoverlapping(self.host.add(offset), bytes.as_mut_ptr(), bytes.len());
        }
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.free_host)(self.host.cast());
            }
        }
    }
}

/// Pinned host staging retained until every asynchronous upload that reads
/// it has completed on the submission stream.
pub(crate) struct PinnedUpload {
    pointer: *mut u8,
    bytes: usize,
    context: Arc<Context>,
}

// The CUDA driver owns the pinned allocation and accepts it from any host
// thread while its primary context is current there.
unsafe impl Send for PinnedUpload {}

impl PinnedUpload {
    pub fn new(context: &Arc<Context>, bytes: usize) -> Result<Self, DriverError> {
        let _current = context.enter()?;
        let mut pointer = std::ptr::null_mut();
        unsafe {
            context.driver.check(
                (context.driver.allocate_host)(&mut pointer, bytes.max(1)),
                "pinned host allocation",
            )?;
        }
        Ok(Self {
            pointer: pointer.cast(),
            bytes,
            context: context.clone(),
        })
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.bytes) }
    }

    /// Enqueues a copy from this retained staging allocation on the exact
    /// production stream owned by the opened CUDA service.
    pub fn upload(
        &self,
        stream: &Stream,
        source_offset: usize,
        destination: u64,
        bytes: usize,
    ) -> Result<(), DriverError> {
        assert!(
            source_offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.bytes),
            "PinnedUpload::upload source range exceeds its private staging allocation"
        );
        if bytes == 0 {
            return Ok(());
        }
        assert!(
            Arc::ptr_eq(&self.context, stream.context()),
            "PinnedUpload::upload precondition: staging and stream belong to one opened CUDA context"
        );
        let _current = self.context.enter()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.upload_async)(
                    destination,
                    self.pointer.add(source_offset).cast(),
                    bytes,
                    stream.raw(),
                ),
                "asynchronous upload",
            )
        }
    }
}

impl Drop for PinnedUpload {
    fn drop(&mut self) {
        // The executor synchronizes the stream before releasing an in-flight
        // staging allocation. Driver teardown/device loss is the only path
        // where entering the context may fail; free is then best-effort.
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.free_host)(self.pointer.cast());
            }
        }
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.free)(self.pointer);
            }
        }
    }
}

/// One loaded module; unloaded with its context.
pub(crate) struct Module {
    pub raw: Handle,
    pub context: Arc<Context>,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.module_unload)(self.raw);
            }
        }
    }
}

/// A JIT link state; retains its log buffers until destroyed.
struct Linker {
    raw: Handle,
    context: Arc<Context>,
    info: Vec<u8>,
    error: Vec<u8>,
}

impl Drop for Linker {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            unsafe {
                (self.context.driver.link_destroy)(self.raw);
            }
        }
    }
}

/// A toolchain (JIT) failure with the driver's log.
#[derive(Clone, Debug)]
pub(crate) enum JitError {
    /// The driver rejected the PTX or failed to link it.
    Toolchain { error: DriverError, log: String },
    /// A driver call unrelated to the PTX text failed.
    Driver(DriverError),
    /// The driver returned an empty or oversized image.
    MalformedImage(String),
}

fn log_text(buffer: &[u8]) -> String {
    let end = buffer.iter().position(|b| *b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

/// PTX -> cubin through the driver JIT.
pub(crate) fn compile_image(context: &Arc<Context>, source: &str) -> Result<Vec<u8>, JitError> {
    let _current = context.enter().map_err(JitError::Driver)?;
    let driver = &context.driver;
    let mut info = vec![0u8; 16384];
    let mut error = vec![0u8; 16384];
    // CUDA driver JIT options: INFO_LOG_BUFFER(3)/SIZE(4), ERROR_LOG_BUFFER(5)/SIZE(6),
    // TARGET_FROM_CUCONTEXT(8), LOG_VERBOSE(12).
    let mut options = [3, 4, 5, 6, 8, 12];
    let mut values = [
        info.as_mut_ptr().cast(),
        info.len() as *mut c_void,
        error.as_mut_ptr().cast(),
        error.len() as *mut c_void,
        std::ptr::null_mut(),
        std::ptr::without_provenance_mut::<c_void>(1),
    ];
    let mut raw = std::ptr::null_mut();
    unsafe {
        driver
            .check(
                (driver.link_create)(
                    options.len() as u32,
                    options.as_mut_ptr(),
                    values.as_mut_ptr(),
                    &mut raw,
                ),
                "JIT link creation",
            )
            .map_err(JitError::Driver)?;
    }
    let linker = Linker {
        raw,
        context: context.clone(),
        info,
        error,
    };
    // The emitter never writes a NUL byte into PTX text (all text is
    // formatted from ASCII mnemonics and decimal/hex literals).
    let mut input = std::ffi::CString::new(source)
        .expect("PTX emitter precondition: emitted text contains no NUL byte")
        .into_bytes_with_nul();
    let status = unsafe {
        // CU_JIT_INPUT_PTX = 1
        (driver.link_add_data)(
            linker.raw,
            1,
            input.as_mut_ptr().cast(),
            input.len(),
            c"seismic.ptx".as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    let toolchain = |status, operation| {
        driver
            .check(status, operation)
            .map_err(|error| JitError::Toolchain {
                error,
                log: log_text(&linker.error),
            })
    };
    toolchain(status, "PTX compilation")?;
    let mut image = std::ptr::null_mut();
    let mut size = 0;
    toolchain(
        unsafe { (driver.link_complete)(linker.raw, &mut image, &mut size) },
        "native image linking",
    )?;
    if image.is_null() || size == 0 || size > isize::MAX as usize {
        return Err(JitError::MalformedImage(format!(
            "driver returned an invalid native image ({size} bytes); log: {}",
            log_text(&linker.info)
        )));
    }
    // The image remains valid until `cuLinkDestroy`; copy before the linker
    // drops.
    Ok(unsafe { std::slice::from_raw_parts(image.cast::<u8>(), size) }.to_vec())
}

/// cubin -> loaded module + entry function handle.
pub(crate) fn load_module(
    context: &Arc<Context>,
    image: &[u8],
    entry: &str,
) -> Result<(Module, Handle), JitError> {
    let _current = context.enter().map_err(JitError::Driver)?;
    let driver = &context.driver;
    let mut raw = std::ptr::null_mut();
    let mut log = vec![0u8; 16384];
    // ERROR_LOG_BUFFER(5)/SIZE(6)
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
    driver
        .check(status, "native image loading")
        .map_err(|error| JitError::Toolchain {
            error,
            log: log_text(&log),
        })?;
    let module = Module {
        raw,
        context: context.clone(),
    };
    // Entry names are `seismic_kernel_<n>`: ASCII without NUL.
    let name = std::ffi::CString::new(entry)
        .expect("PTX emitter precondition: entry names contain no NUL byte");
    let mut function: Handle = std::ptr::null_mut();
    unsafe {
        driver
            .check(
                (driver.module_function)(&mut function, module.raw, name.as_ptr()),
                "kernel lookup",
            )
            .map_err(JitError::Driver)?;
    }
    Ok((module, function))
}

pub(crate) fn module_function(module: &Module, entry: &str) -> Result<Handle, JitError> {
    let _current = module.context.enter().map_err(JitError::Driver)?;
    let name = std::ffi::CString::new(entry)
        .expect("PTX probe entry precondition: entry names contain no NUL byte");
    let mut function = std::ptr::null_mut();
    unsafe {
        module
            .context
            .driver
            .check(
                (module.context.driver.module_function)(&mut function, module.raw, name.as_ptr()),
                "kernel lookup",
            )
            .map_err(JitError::Driver)?;
    }
    Ok(function)
}

/// Authoritative `cuFuncGetAttribute` reflection on one loaded function.
/// Native reconciliation consumes these facts before planning admission.
pub(crate) fn function_attribute(
    context: &Arc<Context>,
    function: Handle,
    key: c_int,
) -> Result<i32, DriverError> {
    let _current = context.enter()?;
    let mut value = 0;
    unsafe {
        context.driver.check(
            (context.driver.function_attribute)(&mut value, key, function),
            "native function attribute",
        )?;
    }
    Ok(value)
}

/// `cuFuncSetAttribute` on a loaded function.
pub(crate) fn set_function_attribute(
    context: &Arc<Context>,
    function: Handle,
    key: c_int,
    value: c_int,
) -> Result<(), DriverError> {
    let _current = context.enter()?;
    unsafe {
        context.driver.check(
            (context.driver.function_set_attribute)(function, key, value),
            "native function attribute update",
        )
    }
}

/// Authoritative occupancy of one concrete loaded function for an exact
/// block size and dynamic-shared byte count.
pub(crate) fn occupancy_max_active_blocks(
    context: &Arc<Context>,
    function: Handle,
    block_threads: u32,
    dynamic_shared_bytes: u64,
) -> Result<u32, DriverError> {
    let _current = context.enter()?;
    let block_threads = c_int::try_from(block_threads).map_err(|_| DriverError {
        operation: "native function occupancy query",
        code: -1,
        description: "block thread count exceeds driver ABI".into(),
    })?;
    let dynamic_shared_bytes = usize::try_from(dynamic_shared_bytes).map_err(|_| DriverError {
        operation: "native function occupancy query",
        code: -1,
        description: "dynamic shared byte count exceeds driver ABI".into(),
    })?;
    let mut blocks = 0;
    unsafe {
        context.driver.check(
            (context.driver.occupancy_max_active_blocks)(
                &mut blocks,
                function,
                block_threads,
                dynamic_shared_bytes,
            ),
            "native function occupancy query",
        )?;
    }
    u32::try_from(blocks).map_err(|_| DriverError {
        operation: "native function occupancy query",
        code: -1,
        description: "driver returned a negative active-block count".into(),
    })
}
