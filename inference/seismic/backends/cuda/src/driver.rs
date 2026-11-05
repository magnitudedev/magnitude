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

use crate::registered::{host_pages, Quiet, Registered};
use seismic_reservation::HostRange;

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
            /// The enqueue calls, reachable only while issuing
            /// ([`crate::registered::issue::Issuing`]).
            pub enqueue: crate::registered::issue::Enqueue,
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
                        let enqueue = crate::registered::issue::Enqueue::load(&library)?;
                        let driver = Self { $($field,)* $($optional,)* enqueue, _library: library };
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
    device_can_access_peer: unsafe extern "system" fn(*mut c_int, c_int, c_int) -> ResultCode
        => "cuDeviceCanAccessPeer",
    memory_info: unsafe extern "system" fn(*mut usize, *mut usize) -> ResultCode => "cuMemGetInfo_v2",
    driver_version: unsafe extern "system" fn(*mut c_int) -> ResultCode => "cuDriverGetVersion",
    primary_context_retain: unsafe extern "system" fn(*mut Handle, c_int) -> ResultCode => "cuDevicePrimaryCtxRetain",
    primary_context_release: unsafe extern "system" fn(c_int) -> ResultCode => "cuDevicePrimaryCtxRelease_v2",
    primary_context_state: unsafe extern "system" fn(c_int, *mut c_uint, *mut c_int) -> ResultCode => "cuDevicePrimaryCtxGetState",
    context_get: unsafe extern "system" fn(*mut Handle) -> ResultCode => "cuCtxGetCurrent",
    context_set: unsafe extern "system" fn(Handle) -> ResultCode => "cuCtxSetCurrent",
    stream_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuStreamCreate",
    stream_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuStreamDestroy_v2",
    stream_query: unsafe extern "system" fn(Handle) -> ResultCode => "cuStreamQuery",
    event_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuEventCreate",
    event_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventDestroy_v2",
    event_synchronize: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventSynchronize",
    event_query: unsafe extern "system" fn(Handle) -> ResultCode => "cuEventQuery",
    event_elapsed_time: unsafe extern "system" fn(*mut f32, Handle, Handle) -> ResultCode => "cuEventElapsedTime",
    allocate: unsafe extern "system" fn(*mut u64, usize) -> ResultCode => "cuMemAlloc_v2",
    free: unsafe extern "system" fn(u64) -> ResultCode => "cuMemFree_v2",
    allocate_host: unsafe extern "system" fn(*mut *mut c_void, usize) -> ResultCode => "cuMemAllocHost_v2",
    free_host: unsafe extern "system" fn(*mut c_void) -> ResultCode => "cuMemFreeHost",
    host_alloc: unsafe extern "system" fn(*mut *mut c_void, usize, c_uint) -> ResultCode => "cuMemHostAlloc",
    host_device_pointer: unsafe extern "system" fn(*mut u64, *mut c_void, c_uint) -> ResultCode => "cuMemHostGetDevicePointer_v2",
    host_register: unsafe extern "system" fn(*mut c_void, usize, c_uint) -> ResultCode => "cuMemHostRegister_v2",
    host_unregister: unsafe extern "system" fn(*mut c_void) -> ResultCode => "cuMemHostUnregister",
    module_load: unsafe extern "system" fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> ResultCode => "cuModuleLoadDataEx",
    module_unload: unsafe extern "system" fn(Handle) -> ResultCode => "cuModuleUnload",
    module_function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> ResultCode => "cuModuleGetFunction",
    module_global: unsafe extern "system" fn(*mut u64, *mut usize, Handle, *const c_char) -> ResultCode => "cuModuleGetGlobal_v2",
    link_create: unsafe extern "system" fn(c_uint, *mut c_int, *mut *mut c_void, *mut Handle) -> ResultCode => "cuLinkCreate_v2",
    link_add_data: unsafe extern "system" fn(Handle, c_int, *mut c_void, usize, *const c_char, c_uint, *mut c_int, *mut *mut c_void) -> ResultCode => "cuLinkAddData_v2",
    link_complete: unsafe extern "system" fn(Handle, *mut *mut c_void, *mut usize) -> ResultCode => "cuLinkComplete",
    link_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuLinkDestroy",
    function_attribute: unsafe extern "system" fn(*mut c_int, c_int, Handle) -> ResultCode => "cuFuncGetAttribute",
    function_set_attribute: unsafe extern "system" fn(Handle, c_int, c_int) -> ResultCode => "cuFuncSetAttribute",
    occupancy_max_active_blocks: unsafe extern "system" fn(*mut c_int, Handle, c_int, usize) -> ResultCode => "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    error_string: unsafe extern "system" fn(ResultCode, *mut *const c_char) -> ResultCode => "cuGetErrorString";
    optional {
        // Driver API 11.7+: a graph node of stream memory operations on a
        // 32-bit word (a host gate's wait, a device signal to the host), and
        // its word and value set on an executable graph.
        graph_add_batch_mem_op_node: unsafe extern "system" fn(*mut Handle, Handle, *const Handle, usize, *const BatchMemOpNodeParams) -> ResultCode => "cuGraphAddBatchMemOpNode",
        graph_exec_batch_mem_op_node_set_params: unsafe extern "system" fn(Handle, Handle, *const BatchMemOpNodeParams) -> ResultCode => "cuGraphExecBatchMemOpNodeSetParams",
        // Driver API 11.4+: identifies the exposed device, including a MIG
        // partition, rather than its parent GPU.
        device_uuid_v2: unsafe extern "system" fn(*mut [u8; 16], c_int) -> ResultCode => "cuDeviceGetUuid_v2",
        // Driver API 10.0+: the OS adapter LUID, supported under WDDM only.
        device_luid: unsafe extern "system" fn(*mut c_char, *mut c_uint, c_int) -> ResultCode
            => "cuDeviceGetLuid",
        // Driver API 12.0+: explicitly built kernel graphs (direct-native replay).
        graph_create: unsafe extern "system" fn(*mut Handle, c_uint) -> ResultCode => "cuGraphCreate",
        graph_add_kernel_node: unsafe extern "system" fn(*mut Handle, Handle, *const Handle, usize, *const KernelNodeParams) -> ResultCode => "cuGraphAddKernelNode_v2",
        graph_instantiate: unsafe extern "system" fn(*mut Handle, Handle, u64) -> ResultCode => "cuGraphInstantiateWithFlags",
        graph_exec_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuGraphExecDestroy",
        graph_destroy: unsafe extern "system" fn(Handle) -> ResultCode => "cuGraphDestroy",
        graph_exec_kernel_node_set_params: unsafe extern "system" fn(Handle, Handle, *const KernelNodeParams) -> ResultCode => "cuGraphExecKernelNodeSetParams_v2",
        // Driver API 11.1+: a node that records an event, and that event set
        // on an executable graph.
        graph_add_event_record_node: unsafe extern "system" fn(*mut Handle, Handle, *const Handle, usize, Handle) -> ResultCode => "cuGraphAddEventRecordNode",
        graph_exec_event_record_node_set_event: unsafe extern "system" fn(Handle, Handle, Handle) -> ResultCode => "cuGraphExecEventRecordNodeSetEvent",
        // Driver API 11.6+: a node of an executable graph enabled or not.
        graph_node_set_enabled: unsafe extern "system" fn(Handle, Handle, c_uint) -> ResultCode => "cuGraphNodeSetEnabled",
        // Driver API 11.1+: a graph executable's device resources set up
        // ahead of its first launch.
        graph_upload: unsafe extern "system" fn(Handle, Handle) -> ResultCode => "cuGraphUpload",
        // Driver API 12.4+: a function loaded now rather than at its first
        // launch, where modules load lazily (`CUDA_MODULE_LOADING`).
        function_load: unsafe extern "system" fn(Handle) -> ResultCode => "cuFuncLoad",
        // Driver API 12.3+: typed graph edges (programmatic dependent launch).
        graph_add_dependencies: unsafe extern "system" fn(Handle, *const Handle, *const Handle, *const EdgeData, usize) -> ResultCode => "cuGraphAddDependencies_v2",
        // Driver API 10.2+: virtual memory management (reserved address ranges).
        memory_granularity: unsafe extern "system" fn(*mut usize, *const AllocationProperties, c_uint) -> ResultCode => "cuMemGetAllocationGranularity",
        address_reserve: unsafe extern "system" fn(*mut u64, usize, usize, u64, u64) -> ResultCode => "cuMemAddressReserve",
        address_free: unsafe extern "system" fn(u64, usize) -> ResultCode => "cuMemAddressFree",
        physical_create: unsafe extern "system" fn(*mut u64, usize, *const AllocationProperties, u64) -> ResultCode => "cuMemCreate",
        physical_release: unsafe extern "system" fn(u64) -> ResultCode => "cuMemRelease",
        map: unsafe extern "system" fn(u64, usize, usize, u64, u64) -> ResultCode => "cuMemMap",
        unmap: unsafe extern "system" fn(u64, usize) -> ResultCode => "cuMemUnmap",
        set_access: unsafe extern "system" fn(u64, usize, *const AccessDescription, usize) -> ResultCode => "cuMemSetAccess",
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

    /// Whether the device's primary context exists in this process.
    #[cfg_attr(target_os = "windows", allow(dead_code))]
    pub fn primary_context_active(&self, ordinal: c_int) -> Result<bool, DriverError> {
        let (mut flags, mut active) = (0, 0);
        unsafe {
            self.check(
                (self.primary_context_state)(ordinal, &mut flags, &mut active),
                "primary context state",
            )?;
        }
        Ok(active != 0)
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
    /// A non-blocking stream for synchronous host reads. A read waits for
    /// its producer through the runtime's access fences; on the legacy
    /// default stream it would also wait for every later queued launch of
    /// the blocking execution streams (a pipelined next step).
    transfer: Handle,
    /// A non-blocking stream that runs the host's wakes behind device
    /// writes of host words (`direct::HostWord`): each waits there for its
    /// write, so no other stream's work waits for a host function.
    notify: Handle,
    /// Where the device's memory is the host's, what the driver holds for
    /// the process outside its resident set: each allocation below adds
    /// itself as it is made and takes itself off as it is freed (device
    /// memory, reservations' granules of either kind, graphs' driver memory;
    /// `cuMemHostAlloc` memory is resident shared memory, in the footprint
    /// already). Set once, at the device's opening.
    pub(crate) driver_memory: OnceLock<Arc<seismic_native_target::process::DriverMemory>>,
    /// The device's quiet: its streams, and its registered host granules
    /// awaiting unregistration ([`Quiet`]).
    pub(crate) quiet: Arc<Quiet>,
}

// The raw handle is an opaque driver object usable from any thread once made
// current there (`enter`).
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Ordinal `ordinal`'s primary context.
    pub fn retain(driver: Arc<Driver>, ordinal: c_int) -> Result<Arc<Self>, DriverError> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            driver.check(
                (driver.primary_context_retain)(&mut raw, ordinal),
                "primary context retain",
            )?;
        }
        let mut context = Self {
            driver,
            raw,
            ordinal,
            transfer: std::ptr::null_mut(),
            notify: std::ptr::null_mut(),
            driver_memory: OnceLock::new(),
            quiet: Quiet::of(ordinal),
        };
        let create = |context: &Self, what: &'static str| -> Result<Handle, DriverError> {
            let _current = context.enter()?;
            let mut stream = std::ptr::null_mut();
            unsafe { context.driver.check((context.driver.stream_create)(&mut stream, CU_STREAM_NON_BLOCKING), what)? };
            context.quiet.add_stream(stream);
            Ok(stream)
        };
        context.transfer = create(&context, "transfer stream creation")?;
        context.notify = create(&context, "notify stream creation")?;
        Ok(Arc::new(context))
    }

    /// The stream the host's wakes run on ([`Context::notify`]).
    pub(crate) fn notify_stream(&self) -> Handle {
        self.notify
    }

    /// Wait for every command queued on `stream` to complete, the thread
    /// sleeping: a blocking event ([`EVENT_BLOCKING_SYNC`]) recorded behind
    /// them. This is the one way the host waits for a stream.
    pub(crate) fn synchronize(&self, stream: Handle, what: &'static str) -> Result<(), DriverError> {
        let _current = self.enter()?;
        let mut event = std::ptr::null_mut();
        unsafe { self.driver.check((self.driver.event_create)(&mut event, EVENT_BLOCKING_SYNC), what)? };
        let waited = self
            .issuing()
            .and_then(|issuing| self.driver.check(unsafe { issuing.event_record(event, stream) }, what))
            .and_then(|()| self.driver.check(unsafe { (self.driver.event_synchronize)(event) }, what));
        unsafe { (self.driver.event_destroy)(event) };
        waited
    }

    /// The right to enqueue device work ([`crate::registered::issue::Issuing`]); refused
    /// with the device's fault once it has one (a registered granule its
    /// quiet failed to unregister: `registered::Quiet`), which ends its
    /// serving with that reason.
    pub(crate) fn issuing(&self) -> Result<crate::registered::issue::Issuing<'_>, DriverError> {
        self.quiet.issuing(&self.driver.enqueue)
    }

    pub fn ordinal(&self) -> c_int {
        self.ordinal
    }

    /// The driver allocated `bytes` outside the resident set
    /// ([`Context::driver_memory`]).
    pub(crate) fn driver_allocated(&self, bytes: u64) {
        if let Some(memory) = self.driver_memory.get() {
            memory.allocated(bytes);
        }
    }

    /// The driver freed `bytes` it allocated ([`Context::driver_allocated`]).
    pub(crate) fn driver_freed(&self, bytes: u64) {
        if let Some(memory) = self.driver_memory.get() {
            memory.freed(bytes);
        }
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
        for stream in [self.transfer, self.notify] {
            if stream.is_null() {
                continue;
            }
            if let Ok(_current) = self.enter() {
                // Its work complete before the quiet stops looking at it
                // (with the device faulted, the quiet no longer drains).
                let _ = self.synchronize(stream, "stream drain");
                self.quiet.remove_stream(stream);
                unsafe {
                    (self.driver.stream_destroy)(stream);
                }
            }
        }
        unsafe {
            (self.driver.primary_context_release)(self.ordinal);
        }
    }
}

/// `CU_STREAM_NON_BLOCKING`: a stream that never synchronizes with the
/// legacy default stream.
const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

/// `CUDA_ERROR_NOT_READY`: a queried stream or event has work still running.
pub(crate) const NOT_READY: c_int = 600;

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
        context.quiet.add_stream(raw);
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
        self.context.synchronize(self.raw, "stream synchronization")
    }

    /// `true` once every command queued on the stream has completed; it
    /// waits for nothing.
    pub fn query(&self) -> Result<bool, DriverError> {
        let _current = self.context.enter()?;
        let status = unsafe { (self.context.driver.stream_query)(self.raw) };
        if status == NOT_READY {
            return Ok(false);
        }
        self.context.driver.check(status, "stream query").map(|()| true)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            // Its work complete before the quiet stops looking at it
            // (with the device faulted, the quiet no longer drains).
            let _ = self.synchronize();
            self.context.quiet.remove_stream(self.raw);
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

// An event is a driver object usable from any thread with its context
// current, which every method makes it; its owner serializes recording it.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

/// `CU_EVENT_BLOCKING_SYNC`: a thread synchronizing on the event sleeps
/// until the device reaches it. A default event's synchronization spins a
/// core for the work's whole run, even when the primary context's flags ask
/// for blocking sync (`cuDevicePrimaryCtxSetFlags`), as measured on the
/// driving thread of a decode step. Every host wait on the device therefore
/// goes through such an event ([`Context::synchronize`], [`Event::new`]).
const EVENT_BLOCKING_SYNC: c_uint = 0x1;

impl Event {
    /// An event whose synchronization blocks ([`EVENT_BLOCKING_SYNC`]).
    pub fn new(context: &Arc<Context>) -> Result<Self, DriverError> {
        Self::with_flags(context, EVENT_BLOCKING_SYNC)
    }

    /// An event only timed, read once its work completed: nothing sleeps on
    /// it, so recording it wakes no host.
    pub fn stamp(context: &Arc<Context>) -> Result<Self, DriverError> {
        Self::with_flags(context, 0)
    }

    fn with_flags(context: &Arc<Context>, flags: c_uint) -> Result<Self, DriverError> {
        let _current = context.enter()?;
        let mut raw = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check((context.driver.event_create)(&mut raw, flags), "event creation")?;
        }
        Ok(Self {
            raw,
            context: context.clone(),
        })
    }

    pub(crate) fn raw(&self) -> Handle {
        self.raw
    }

    pub fn record(&self, stream: &Stream) -> Result<(), DriverError> {
        assert!(
            Arc::ptr_eq(&self.context, stream.context()),
            "Event::record precondition: event and stream belong to one opened CUDA context"
        );
        let _current = self.context.enter()?;
        let status = unsafe { self.context.issuing()?.event_record(self.raw, stream.raw()) };
        self.context.driver.check(status, "event record")
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
        context.driver_allocated(bytes.max(1) as u64);
        Ok(Self {
            pointer,
            bytes,
            context: context.clone(),
        })
    }

    /// Precondition (this module's own): `offset + bytes.len() <= self.bytes`.
    /// The generic runtime binds buffers whose extents the executable derived
    /// from the same layout roots, so a violation is a bug of this crate.
    pub fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.bytes),
            "Allocation::download_at precondition: [{offset}, {offset}+{}) exceeds {} bytes",
            bytes.len(),
            self.bytes
        );
        download(&self.context, self.pointer + offset as u64, bytes)
    }
}

/// `CUmemLocation`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MemoryLocation {
    kind: c_uint,
    id: c_int,
}

impl MemoryLocation {
    fn device(ordinal: c_int) -> Self {
        // CU_MEM_LOCATION_TYPE_DEVICE
        Self {
            kind: 1,
            id: ordinal,
        }
    }

    /// Host NUMA node `node`.
    fn host_numa(node: c_int) -> Self {
        // CU_MEM_LOCATION_TYPE_HOST_NUMA
        Self { kind: 3, id: node }
    }

    /// Where created `memory` of device `ordinal` lies: the device, or the
    /// host NUMA node closest to it (`CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`),
    /// node 0 where the host has no NUMA, as `cuMemCreate` requires there.
    fn of(memory: CreatedMemory, driver: &Driver, ordinal: c_int) -> Result<Self, DriverError> {
        // CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
        const HOST_NUMA_ID: c_int = 134;
        Ok(match memory {
            CreatedMemory::Device => Self::device(ordinal),
            CreatedMemory::Host => Self::host_numa(driver.attribute(HOST_NUMA_ID, ordinal)?.max(0)),
        })
    }
}

/// Memory a reservation's granules are created as (`cuMemCreate`): the
/// device's own, or pinned memory of its host node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CreatedMemory {
    Device,
    Host,
}

impl CreatedMemory {
    /// Who may read and write its granules: the device, and for host memory
    /// the host (`CU_MEM_LOCATION_TYPE_HOST_NUMA` access, whose id the
    /// driver ignores).
    fn access(self, ordinal: c_int) -> Vec<AccessDescription> {
        // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        let readwrite = |location| AccessDescription { location, flags: 3 };
        let device = readwrite(MemoryLocation::device(ordinal));
        match self {
            Self::Device => vec![device],
            Self::Host => vec![device, readwrite(MemoryLocation::host_numa(0))],
        }
    }
}

/// What memory a reservation's granules are ([`ReservedMemory::new`]): the
/// mechanism of each kind that reserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeMemory {
    /// The device's own memory, mapped for the device.
    Device,
    /// Pinned memory of the host node closest to a device whose memory is
    /// the host's (GB10), created for it (`Granules::Created`) and mapped
    /// for the device and the host at the same addresses, which it reads
    /// as its own.
    HostNode,
    /// Host pages a discrete device reads in place over its link, each
    /// granule registered for it at its host address
    /// (`Granules::Registered`). Its pages lie where the
    /// host's first touch places them: the process may not set their node
    /// in a container (`mbind` was refused on two RTX 3090 hosts), and the
    /// node made no difference to the link (a 2990WX's tier on node 2 of a
    /// device on node 0, and an EPYC 7313's over nodes 0-3 of one on node 1,
    /// read as fast as host-NUMA allocations of the device's node).
    Registered,
}

impl RangeMemory {
    /// The memory of `kind` (`crate::topology`); none for one that reserves
    /// nothing.
    pub const fn of(kind: u32) -> Option<Self> {
        match kind {
            crate::topology::DEVICE_KIND => Some(Self::Device),
            crate::topology::HOST_NODE_KIND => Some(Self::HostNode),
            crate::topology::REGISTERED_KIND => Some(Self::Registered),
            _ => None,
        }
    }

    /// Whether the host addresses the backed bytes in place: a device
    /// reservation the host reaches only by driver copies
    /// (`Buffer::host_address`), a host one in place.
    pub const fn host_addressed(self) -> bool {
        matches!(self, Self::HostNode | Self::Registered)
    }

    /// What its granules are created as; none for registered host pages.
    const fn created(self) -> Option<CreatedMemory> {
        match self {
            Self::Device => Some(CreatedMemory::Device),
            Self::HostNode => Some(CreatedMemory::Host),
            Self::Registered => None,
        }
    }
}

/// `CUmemAllocationProp`.
#[repr(C)]
pub(crate) struct AllocationProperties {
    kind: c_uint,
    requested_handle_types: c_uint,
    location: MemoryLocation,
    win32_metadata: *mut c_void,
    compression: c_uchar,
    rdma_capable: c_uchar,
    usage: u16,
    reserved: [c_uchar; 4],
}

impl AllocationProperties {
    /// Pinned memory at `location`, not exportable.
    fn at(location: MemoryLocation) -> Self {
        Self {
            // CU_MEM_ALLOCATION_TYPE_PINNED
            kind: 1,
            requested_handle_types: 0,
            location,
            win32_metadata: std::ptr::null_mut(),
            compression: 0,
            rdma_capable: 0,
            usage: 0,
            reserved: [0; 4],
        }
    }
}

/// `CUmemAccessDesc`.
#[repr(C)]
pub(crate) struct AccessDescription {
    location: MemoryLocation,
    flags: c_uint,
}

fn memory_management_symbol<F: Copy>(symbol: Option<F>) -> Result<F, DriverError> {
    symbol.ok_or(DriverError {
        operation: "virtual memory management",
        code: 0,
        description: "the CUDA driver predates driver API 10.2 virtual memory management".into(),
    })
}

/// The granularity in which the driver backs device memory
/// (`cuMemGetAllocationGranularity`, recommended): what one allocation's
/// size is rounded up to.
pub(crate) fn allocation_granularity(context: &Context) -> Result<u64, DriverError> {
    granularity(&context.driver, MemoryLocation::device(context.ordinal()))
}

/// `cuMemGetAllocationGranularity` (recommended) for pinned memory at
/// `location`. A failed query is an error, never a guessed granule: a
/// driver without it (before driver API 10.2) loads none of the kernels,
/// whose least target, `sm_80` (`PtxTarget::BASELINE`), CUDA 11.0 first
/// supports.
fn granularity(driver: &Driver, location: MemoryLocation) -> Result<u64, DriverError> {
    let granularity_of = memory_management_symbol(driver.memory_granularity)?;
    let properties = AllocationProperties::at(location);
    let mut granularity = 0usize;
    unsafe {
        // CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        driver.check(
            granularity_of(&mut granularity, &properties, 1),
            "allocation granularity",
        )?;
    }
    Ok(granularity as u64)
}

/// What device work reaches a reservation's backed bytes through chunks
/// of: none, the granules are mapped into one device virtual range.
pub const RESERVATION_CHUNKS: Option<u64> = None;

/// The first driver API that places virtual-memory allocations on a host
/// NUMA node (`CU_MEM_LOCATION_TYPE_HOST_NUMA`, CUDA 12.2), which created
/// host memory is, as `cuDriverGetVersion` reports it.
const HOST_NUMA_DRIVER: c_int = 12020;

/// The granule in which device `ordinal` backs a reserved tensor's
/// committed bytes of `memory`; `None` where it reserves no address ranges
/// of it. Created memory: the driver's allocation granularity, none
/// without virtual memory management or for host memory before
/// [`HOST_NUMA_DRIVER`]. Registered host pages: the host's registration
/// granule (`seismic_reservation::registration_granule`, a large page where it has
/// them), none where the device reads registered memory at no host address
/// (`CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`), which their reservation's
/// one base needs. Discovery reads it without a context: what it
/// advertises is what [`ReservedMemory::new`] backs.
pub fn reservation_granule(ordinal: u32, memory: RangeMemory) -> Result<Option<u64>, String> {
    let driver = Driver::load()?;
    let ordinal = c_int::try_from(ordinal).map_err(|_| "a device ordinal exceeds the driver ABI")?;
    reserving_granularity(&driver, ordinal, memory).map_err(|error| error.to_string())
}

/// A reserving device's granule of `memory` ([`granularity`]).
fn reserving_granularity(
    driver: &Driver,
    ordinal: c_int,
    memory: RangeMemory,
) -> Result<Option<u64>, DriverError> {
    let Some(memory) = memory.created() else {
        // CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        const HOST_POINTER: c_int = 91;
        if driver.attribute(HOST_POINTER, ordinal)? == 0 {
            return Ok(None);
        }
        return seismic_reservation::registration_granule().map(Some).map_err(host_pages);
    };
    // CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    const VIRTUAL_MEMORY_MANAGEMENT: c_int = 102;
    if driver.address_reserve.is_none() || driver.attribute(VIRTUAL_MEMORY_MANAGEMENT, ordinal)? == 0
    {
        return Ok(None);
    }
    if memory == CreatedMemory::Host {
        let mut version = 0;
        unsafe {
            driver.check((driver.driver_version)(&mut version), "driver version")?;
        }
        if version < HOST_NUMA_DRIVER {
            return Ok(None);
        }
    }
    let location = MemoryLocation::of(memory, driver, ordinal)?;
    granularity(driver, location).map(Some)
}

/// An address range reserved once, with memory of one kind
/// ([`RangeMemory`]) backed over a leading prefix of it in granules. The
/// base address never changes while the backed prefix grows and shrinks, so
/// work bound to it (and graphs keyed by its address) stays valid, and host
/// memory's host address with it.
pub struct ReservedMemory {
    pub(crate) base: u64,
    /// Bytes of address space, a whole number of granules.
    pub(crate) reserved: usize,
    pub(crate) granularity: usize,
    /// What the granules are.
    pub(crate) memory: RangeMemory,
    /// The account they are charged to.
    account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    granules: Granules,
    charges: std::sync::Mutex<Vec<Box<dyn seismic_compiler::executable::MemoryHold>>>,
    /// The stream its zero fills run on, its own: what it zeroes no queued
    /// device work reaches, so it waits on none of the device's.
    pub(crate) stream: Arc<Stream>,
    context: Arc<Context>,
}

/// What backs a reservation's granules, and how many are backed.
enum Granules {
    /// Physical allocations created at `location` (CUDA virtual memory
    /// management), mapped in order from the base of a reserved device
    /// address range: device memory, and host memory where the device's
    /// memory is the host's. There it reads them faster than registered
    /// host pages: on GB10 1 GiB read at 239 GB/s against 164 GB/s
    /// (`where_memory_is_the_hosts_a_host_reservation_reads_as_a_device_one`).
    Created {
        memory: CreatedMemory,
        location: MemoryLocation,
        handles: std::sync::Mutex<Vec<u64>>,
    },
    /// Host memory a device reads over its link ([`Registered`]: granules
    /// registered at their host addresses, unregistered only once the
    /// device is quiet), no driver copy or fill crossing a granule
    /// ([`ReservedMemory::copy_span`]). Physical allocations `cuMemCreate`
    /// places on a host node are small pages, which an IOMMU translating
    /// the device's reads (DMA mode) maps one by one: on an RTX 3090 their
    /// zero-copy reads fall to half the link and below, and their copies
    /// run at 10.7 of its 12.4 GB/s ([`HostRange`]).
    Registered(Arc<Registered>),
}

unsafe impl Send for ReservedMemory {}
unsafe impl Sync for ReservedMemory {}

impl ReservedMemory {
    /// Reserve address space for `bytes` with nothing backed, to be backed
    /// with `memory` in `granularity`-byte granules (`reservation_granule`),
    /// charged to `account`.
    pub(crate) fn new(
        context: &Arc<Context>,
        bytes: usize,
        granularity: usize,
        memory: RangeMemory,
        account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    ) -> Result<Self, DriverError> {
        let driver = &context.driver;
        let stream = Stream::new(context)?;
        let reserved = bytes.max(1).div_ceil(granularity) * granularity;
        let (base, granules) = match memory.created() {
            Some(created) => {
                let reserve = memory_management_symbol(driver.address_reserve)?;
                let location = MemoryLocation::of(created, driver, context.ordinal())?;
                let _current = context.enter()?;
                let mut base = 0u64;
                unsafe {
                    driver.check(
                        reserve(&mut base, reserved, granularity, 0, 0),
                        "address reservation",
                    )?;
                }
                let handles = std::sync::Mutex::new(Vec::new());
                (base, Granules::Created { memory: created, location, handles })
            }
            None => {
                let range = HostRange::reserve(reserved as u64, granularity as u64, seismic_reservation::PageSize::Large)
                    .map_err(host_pages)?;
                let base = range.base().as_ptr() as u64;
                (base, Granules::Registered(Registered::new(context, range)))
            }
        };
        Ok(Self {
            base,
            charges: std::sync::Mutex::new(Vec::new()),
            reserved,
            granularity,
            memory,
            account,
            granules,
            stream,
            context: context.clone(),
        })
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Registered granules, committed or awaiting the device's quiet.
    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn registered_granules(&self) -> Option<usize> {
        match &self.granules {
            Granules::Registered(registered) => Some(registered.registered()),
            Granules::Created { .. } => None,
        }
    }

    /// The bytes no driver copy of this reservation's crosses: registered
    /// granules are each their own registration.
    pub(crate) fn copy_span(&self) -> Option<u64> {
        matches!(self.granules, Granules::Registered(_)).then_some(self.granularity as u64)
    }

    /// Back exactly the granules covering the leading `bytes`: map new ones
    /// read-write for this device, or unmap and release the tail. Precondition
    /// for a shrink: no device work still touches the released range.
    /// Each granule backed is charged to the reservation's account before
    /// it is mapped, and
    /// released with its granule. Where its bytes read zero from: registered
    /// host pages newly backed are the host's, zeroed; created granules are
    /// the driver's, whose contents it leaves undefined, so none of `bytes`.
    pub(crate) fn commit(&self, bytes: usize) -> Result<usize, seismic_compiler::errors::ExecutionError> {
        let wanted = bytes.div_ceil(self.granularity);
        let charge = || self.account.reserve(self.granularity as u64);
        assert!(
            bytes <= self.reserved,
            "ReservedMemory::commit precondition: {bytes} bytes exceed the {}-byte reservation",
            self.reserved
        );
        match &self.granules {
            Granules::Created { memory, location, handles } => {
                let mut charges = self
                    .charges
                    .lock()
                    .expect("native reservation charge lock poisoned");
                let mut extra = Vec::new();
                for _ in charges.len()..wanted {
                    extra.push(charge()?);
                }
                self.commit_created(*memory, *location, handles, wanted)
                    .map_err(crate::executor::allocation_error(bytes as u64))?;
                charges.truncate(wanted);
                charges.extend(extra);
                Ok(bytes)
            }
            // Its granules carry their own charges, which fall when they
            // are unregistered ([`Registered::commit`]).
            Granules::Registered(registered) => registered.commit(wanted, charge).map(|fresh| fresh * self.granularity),
        }
    }

    fn commit_created(
        &self,
        memory: CreatedMemory,
        location: MemoryLocation,
        handles: &std::sync::Mutex<Vec<u64>>,
        wanted: usize,
    ) -> Result<(), DriverError> {
        let driver = &self.context.driver;
        let create = memory_management_symbol(driver.physical_create)?;
        let release = memory_management_symbol(driver.physical_release)?;
        let map = memory_management_symbol(driver.map)?;
        let unmap = memory_management_symbol(driver.unmap)?;
        let set_access = memory_management_symbol(driver.set_access)?;
        let _current = self.context.enter()?;
        let mut granules = handles
            .lock()
            .expect("reservation granule lock is never poisoned");
        let address = |index: usize| self.base + (index * self.granularity) as u64;
        let previous = granules.len();
        while granules.len() > wanted {
            let index = granules.len() - 1;
            unsafe {
                if let Err(error) = driver.check(unmap(address(index), self.granularity), "unmap") {
                    if granules.len() == previous {
                        return Err(error);
                    }
                    panic!("CUDA reservation unmap failed after partial shrink: {error}");
                }
                // Once unmapped, the physical handle is no longer usable by
                // the old view. Releasing it must succeed for the shrink to
                // return as an ordinary recoverable result.
                driver
                    .check(release(granules[index]), "physical release")
                    .unwrap_or_else(|error| {
                        panic!("CUDA reservation release failed after unmap: {error}")
                    });
            }
            granules.pop();
            self.context.driver_freed(self.granularity as u64);
        }
        if wanted < previous {
            return Ok(());
        }
        let first_new = granules.len();
        if wanted == first_new {
            return Ok(());
        }
        let properties = AllocationProperties::at(location);
        let mapped = (|| {
            while granules.len() < wanted {
                let mut handle = 0u64;
                unsafe {
                    driver.check(
                        create(&mut handle, self.granularity, &properties, 0),
                        "physical allocation",
                    )?;
                    if let Err(error) = driver.check(
                        map(address(granules.len()), self.granularity, 0, handle, 0),
                        "map",
                    ) {
                        release(handle);
                        return Err(error);
                    }
                }
                granules.push(handle);
                self.context.driver_allocated(self.granularity as u64);
            }
            let access = memory.access(self.context.ordinal());
            unsafe {
                driver.check(
                    set_access(
                        address(first_new),
                        (wanted - first_new) * self.granularity,
                        access.as_ptr(),
                        access.len(),
                    ),
                    "set access",
                )
            }
        })();
        if mapped.is_err() {
            // Leave the backing as it was: release what this call mapped.
            while granules.len() > first_new {
                let index = granules.len() - 1;
                unsafe {
                    unmap(address(index), self.granularity);
                    release(granules[index]);
                }
                granules.pop();
                self.context.driver_freed(self.granularity as u64);
            }
        }
        mapped
    }
}

impl Drop for ReservedMemory {
    fn drop(&mut self) {
        let driver = &self.context.driver;
        let Ok(_current) = self.context.enter() else {
            return;
        };
        match &mut self.granules {
            Granules::Created { handles, .. } => {
                let (Some(unmap), Some(release), Some(free)) =
                    (driver.unmap, driver.physical_release, driver.address_free)
                else {
                    return;
                };
                let granules = handles
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for (index, handle) in granules.iter().enumerate() {
                    unsafe {
                        unmap(
                            self.base + (index * self.granularity) as u64,
                            self.granularity,
                        );
                        release(*handle);
                    }
                }
                self.context
                    .driver_freed((granules.len() * self.granularity) as u64);
                unsafe {
                    free(self.base, self.reserved);
                }
            }
            // Unregistered once the device is quiet, its range freed then.
            Granules::Registered(registered) => registered.drop_reservation(),
        }
    }
}

/// Synchronous host copy into device memory at `pointer`, on the context's
/// non-blocking transfer stream: it waits for nothing but itself (callers
/// hold the destination's access fence, so no queued launch still reads or
/// writes it).
pub(crate) fn upload(context: &Context, pointer: u64, bytes: &[u8]) -> Result<(), DriverError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let _current = context.enter()?;
    unsafe {
        context.driver.check(
            context.issuing()?.upload_async(pointer, bytes.as_ptr().cast(), bytes.len(), context.transfer),
            "upload",
        )?;
    }
    context.synchronize(context.transfer, "upload synchronization")
}

/// Synchronous device-to-host copy from `pointer`, on the context's
/// non-blocking transfer stream: it waits for nothing but itself (callers
/// hold the producer's access fence).
pub(crate) fn download(
    context: &Context,
    pointer: u64,
    bytes: &mut [u8],
) -> Result<(), DriverError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let _current = context.enter()?;
    unsafe {
        context.driver.check(
            context.issuing()?.download_async(bytes.as_mut_ptr().cast(), pointer, bytes.len(), context.transfer),
            "download",
        )?;
    }
    context.synchronize(context.transfer, "download synchronization")
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

/// `CUgraphEdgeData`.
#[repr(C)]
pub(crate) struct EdgeData {
    from_port: c_uchar,
    to_port: c_uchar,
    kind: c_uchar,
    reserved: [c_uchar; 5],
}

/// How a graph node depends on the node before it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Edge {
    /// The node starts after its predecessor completes.
    Complete,
    /// The node may start once every block of its predecessor has triggered
    /// (`griddepcontrol.launch_dependents`) or exited; its kernel waits for
    /// the predecessor's completion itself (`griddepcontrol.wait`) before
    /// touching memory the predecessor may write.
    Programmatic,
}

fn graph_symbol<F: Copy>(symbol: Option<F>) -> Result<F, DriverError> {
    symbol.ok_or(DriverError {
        operation: "kernel graph",
        code: 0,
        description: "the CUDA driver predates driver API 12.0 kernel graphs".into(),
    })
}

/// `CUstreamBatchMemOpParams`: a union of 48 bytes, of which a 32-bit wait
/// or write (`CUstreamMemOpWaitValueParams_st`,
/// `CUstreamMemOpWriteValueParams_st`) uses the operation, the address, the
/// value, the flags and the alias.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct StreamMemOp {
    operation: u32,
    address: u64,
    value: u64,
    flags: u32,
    alias: u64,
    reserved: u64,
}

const _: () = assert!(size_of::<StreamMemOp>() == 48);

/// `CUDA_BATCH_MEM_OP_NODE_PARAMS`.
#[repr(C)]
pub(crate) struct BatchMemOpNodeParams {
    context: Handle,
    count: c_uint,
    operations: *mut StreamMemOp,
    flags: c_uint,
}

/// A 32-bit stream memory operation on a word at a device address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WordOperation {
    /// Wait until the word reaches the value (`CU_STREAM_WAIT_VALUE_GEQ`,
    /// compared cyclically: `word - value` as I32 at least 0).
    Wait,
    /// Write the value once the work before completes, its writes visible
    /// first (`CU_STREAM_WRITE_VALUE_DEFAULT`, with its memory barrier).
    Write,
}

impl WordOperation {
    /// Its flags, a stream's (`cuStreamWaitValue32`, `cuStreamWriteValue32`)
    /// and a graph node's alike: `CU_STREAM_WAIT_VALUE_GEQ` and
    /// `CU_STREAM_WRITE_VALUE_DEFAULT`.
    pub(crate) fn flags(self) -> u32 {
        match self {
            Self::Wait | Self::Write => 0,
        }
    }

    fn of(self, address: u64, value: u32) -> StreamMemOp {
        // CU_STREAM_MEM_OP_WAIT_VALUE_32 and _WRITE_VALUE_32.
        let operation = match self {
            Self::Wait => 1,
            Self::Write => 2,
        };
        StreamMemOp { operation, address, value: u64::from(value), flags: self.flags(), alias: 0, reserved: 0 }
    }
}

/// A kernel graph under construction: kernel nodes, each depending on the
/// earlier nodes it names, and event-record and word-operation nodes among
/// them, each after the node added before it.
pub(crate) struct Graph {
    raw: Handle,
    /// Every kernel node, in order of addition.
    nodes: Vec<Handle>,
    /// Every event-record node of a mark ([`Self::push_mark`]), in order of
    /// addition.
    marks: Vec<Handle>,
    /// Every word-operation node, in order of addition.
    operations: Vec<Handle>,
    /// The node added last.
    last: Option<Handle>,
    /// The node added last since the last kernel node, where one was: the
    /// next kernel node follows it.
    since: Option<Handle>,
    /// Whether a wait was added since the last kernel node: the next kernel
    /// node follows it to completion, whatever its edge.
    gated: bool,
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
            nodes: Vec::new(),
            marks: Vec::new(),
            operations: Vec::new(),
            last: None,
            since: None,
            gated: false,
            context: context.clone(),
        })
    }

    /// Add an event-record node of a mark, recording `event` once the node
    /// added last completes, or as the graph starts where none was
    /// ([`Self::push_event`]).
    pub fn push_mark(&mut self, event: &Event) -> Result<(), DriverError> {
        let raw = self.push_event(event)?;
        self.marks.push(raw);
        Ok(())
    }

    /// Add an event-record node recording `event` once the node added last
    /// completes, or as the graph starts where none was. A launch's chain
    /// runs through it, so the graph stays one chain, but for a programmatic
    /// edge, which only a kernel can source: there the node hangs off the
    /// chain (`push_kernel`).
    pub fn push_event(&mut self, event: &Event) -> Result<Handle, DriverError> {
        let add = graph_symbol(self.context.driver.graph_add_event_record_node)?;
        let _current = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        let dependencies = self.last;
        unsafe {
            self.context.driver.check(
                add(&mut raw, self.raw, dependencies.as_slice().as_ptr(), dependencies.as_slice().len(), event.raw()),
                "kernel graph event record node",
            )?;
        }
        self.last = Some(raw);
        self.since = Some(raw);
        Ok(raw)
    }

    /// Add a node of one word operation on the word at device address
    /// `address`, once the node added last completes, or as the graph starts
    /// where none was. A wait holds every later node of the chain, a
    /// programmatic launch's too (`push_kernel`); a write hangs off a
    /// programmatic edge as an event-record node does. Following the node
    /// added last follows the whole chain before it: a kernel on a
    /// programmatic edge waits for its predecessor's completion itself
    /// ([`Edge::Programmatic`]), so it completes only after it.
    pub fn push_word(&mut self, operation: WordOperation, address: u64, value: u32) -> Result<(), DriverError> {
        let add = graph_symbol(self.context.driver.graph_add_batch_mem_op_node)?;
        let _current = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        let dependencies = self.last;
        let mut word = operation.of(address, value);
        let params = BatchMemOpNodeParams { context: self.context.raw, count: 1, operations: &mut word, flags: 0 };
        unsafe {
            self.context.driver.check(
                add(&mut raw, self.raw, dependencies.as_slice().as_ptr(), dependencies.as_slice().len(), &params),
                "kernel graph word operation node",
            )?;
        }
        self.operations.push(raw);
        self.last = Some(raw);
        self.since = Some(raw);
        self.gated |= operation == WordOperation::Wait;
        Ok(())
    }

    /// Add a kernel node after the nodes `after` names (by order of
    /// addition), by `edge` from each, and after a node added since them,
    /// unless the edge is programmatic: it starts before they complete,
    /// which waiting on that node would undo. After a wait it follows the
    /// wait to completion, whatever its edge. The driver copies the
    /// parameter values the node's `parameters` point to.
    pub fn push_kernel(&mut self, node: &KernelNodeParams, edge: Edge, after: &[usize]) -> Result<(), DriverError> {
        let add = graph_symbol(self.context.driver.graph_add_kernel_node)?;
        let _current = self.context.enter()?;
        let mut raw = std::ptr::null_mut();
        let after = after.iter().map(|&node| self.nodes[node]).collect::<Vec<_>>();
        let gated = std::mem::take(&mut self.gated);
        let programmatic = edge == Edge::Programmatic && !after.is_empty() && !gated;
        let since = self.since.take();
        let dependencies = match since {
            _ if programmatic => &[][..],
            Some(ref since) => std::slice::from_ref(since),
            None => after.as_slice(),
        };
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
        if programmatic {
            let connect = self.context.driver.graph_add_dependencies.ok_or(DriverError {
                operation: "kernel graph edge",
                code: 0,
                description: "the CUDA driver predates driver API 12.3 typed graph edges".into(),
            })?;
            // CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC out of the predecessor,
            // CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC.
            let data = EdgeData {
                from_port: 1,
                to_port: 0,
                kind: 1,
                reserved: [0; 5],
            };
            for from in &after {
                unsafe {
                    self.context
                        .driver
                        .check(connect(self.raw, from, &raw, &data, 1), "kernel graph edge")?;
                }
            }
        }
        self.nodes.push(raw);
        self.last = Some(raw);
        Ok(())
    }

    /// Kernel node `index`, by order of addition.
    pub fn node(&self, index: usize) -> Handle {
        self.nodes[index]
    }

    /// Event-record node `index`, by order of addition.
    pub fn mark(&self, index: usize) -> Handle {
        self.marks[index]
    }

    /// Word-operation node `index`, by order of addition.
    pub fn word(&self, index: usize) -> Handle {
        self.operations[index]
    }

    /// An executable graph of the nodes added. The graph is kept by the
    /// caller while the executable's parameters may be set: they name its
    /// nodes ([`GraphExec::set_kernel`]).
    pub fn instantiate(&self) -> Result<GraphExec, DriverError> {
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

// A graph is a driver object usable from any thread with its context
// current; its owner serializes changes to it.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        if let (Some(destroy), Ok(_current)) =
            (self.context.driver.graph_destroy, self.context.enter())
        {
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
    /// Set up the executable's device resources on `stream` ahead of its
    /// first launch, which would otherwise do it: a driver call that can
    /// wait behind a kernel spinning on the host, so every graph is uploaded
    /// as it is instantiated, while no device work is queued.
    pub fn upload(&self, stream: &Stream) -> Result<(), DriverError> {
        let upload = graph_symbol(self.context.driver.graph_upload)?;
        let _current = self.context.enter()?;
        unsafe { self.context.driver.check(upload(self.raw, stream.raw()), "kernel graph upload") }
    }

    /// Set the parameters of `node`, a kernel node of the graph this
    /// executable was instantiated from, to `params` (the same function and
    /// geometry, other arguments) for every launch after this call. Launches
    /// already queued keep the parameters they were queued with, and the
    /// driver takes the new values at the next launch, in stream order, so
    /// no device work is waited for. The driver copies the values `params`
    /// points to.
    pub fn set_kernel(&self, node: Handle, params: &KernelNodeParams) -> Result<(), DriverError> {
        let set = graph_symbol(self.context.driver.graph_exec_kernel_node_set_params)?;
        let _current = self.context.enter()?;
        unsafe {
            self.context
                .driver
                .check(set(self.raw, node, params), "kernel graph node parameter update")
        }
    }

    /// Have `node`, an event-record node of the graph this executable was
    /// instantiated from, record `event` for every launch after this call.
    /// Launches already queued keep the event they were queued with.
    pub fn set_event(&self, node: Handle, event: &Event) -> Result<(), DriverError> {
        let set = graph_symbol(self.context.driver.graph_exec_event_record_node_set_event)?;
        let _current = self.context.enter()?;
        unsafe {
            self.context
                .driver
                .check(set(self.raw, node, event.raw()), "kernel graph event update")
        }
    }

    /// Have `node`, a word-operation node of the graph this executable was
    /// instantiated from, do `operation` (its own) on the word at `address`
    /// with `value` for every launch after this call. Launches already
    /// queued keep what they were queued with.
    pub fn set_word(&self, node: Handle, operation: WordOperation, address: u64, value: u32) -> Result<(), DriverError> {
        let set = graph_symbol(self.context.driver.graph_exec_batch_mem_op_node_set_params)?;
        let _current = self.context.enter()?;
        let mut word = operation.of(address, value);
        let params = BatchMemOpNodeParams { context: self.context.raw, count: 1, operations: &mut word, flags: 0 };
        unsafe {
            self.context
                .driver
                .check(set(self.raw, node, &params), "kernel graph word operation update")
        }
    }

    /// Enable `node`, a node of the graph this executable was instantiated
    /// from, or disable it, for every launch after this call: a disabled
    /// node runs as an empty one, its dependencies kept. Launches already
    /// queued are unaffected.
    pub fn set_enabled(&self, node: Handle, enabled: bool) -> Result<(), DriverError> {
        let set = graph_symbol(self.context.driver.graph_node_set_enabled)?;
        let _current = self.context.enter()?;
        unsafe {
            self.context
                .driver
                .check(set(self.raw, node, c_uint::from(enabled)), "kernel graph node enable")
        }
    }

    pub fn launch(&self, stream: &Stream) -> Result<(), DriverError> {
        let _current = self.context.enter()?;
        let Some(status) = (unsafe { self.context.issuing()?.graph_launch(self.raw, stream.raw()) }) else {
            return graph_symbol::<()>(None);
        };
        self.context.driver.check(status, "kernel graph launch")
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
    _charge: Box<dyn seismic_compiler::executable::MemoryHold>,
}

// The CUDA driver owns the pinned allocation; host access is plain memory
// access, and the owner serializes host access against device use.
unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    /// Mapped host memory of `bytes`, charged to `account`.
    pub fn new(
        context: &Arc<Context>,
        account: &Arc<dyn seismic_compiler::executable::MemoryAccount>,
        bytes: usize,
    ) -> Result<Self, seismic_compiler::errors::ExecutionError> {
        // CU_MEMHOSTALLOC_DEVICEMAP
        const DEVICE_MAP: c_uint = 2;
        let failed = || crate::executor::allocation_error(bytes as u64);
        let charge = account.reserve(bytes.max(1) as u64)?;
        let _current = context.enter().map_err(failed())?;
        let mut host = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check(
                    (context.driver.host_alloc)(&mut host, bytes.max(1), DEVICE_MAP),
                    "mapped host allocation",
                )
                .map_err(failed())?;
            // Owned from here, so a failed mapping query frees it.
            let mut mapped = Self {
                host: host.cast(),
                device: 0,
                bytes,
                context: context.clone(),
                _charge: charge,
            };
            context
                .driver
                .check(
                    (context.driver.host_device_pointer)(&mut mapped.device, host, 0),
                    "mapped host device pointer",
                )
                .map_err(failed())?;
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

    /// The allocation's first byte on the host.
    pub fn host(&self) -> *mut u8 {
        self.host
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
    _charge: Box<dyn seismic_compiler::executable::MemoryHold>,
}

// The CUDA driver owns the pinned allocation and accepts it from any host
// thread while its primary context is current there.
unsafe impl Send for PinnedUpload {}

impl PinnedUpload {
    /// Pinned host staging of `bytes`, charged to `account`.
    pub fn new(
        context: &Arc<Context>,
        account: &Arc<dyn seismic_compiler::executable::MemoryAccount>,
        bytes: usize,
    ) -> Result<Self, seismic_compiler::errors::ExecutionError> {
        let failed = || crate::executor::allocation_error(bytes as u64);
        let charge = account.reserve(bytes.max(1) as u64)?;
        let _current = context.enter().map_err(failed())?;
        let mut pointer = std::ptr::null_mut();
        unsafe {
            context
                .driver
                .check(
                    (context.driver.allocate_host)(&mut pointer, bytes.max(1)),
                    "pinned host allocation",
                )
                .map_err(failed())?;
        }
        Ok(Self {
            pointer: pointer.cast(),
            bytes,
            context: context.clone(),
            _charge: charge,
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
        let status = unsafe {
            self.context.issuing()?.upload_async(destination, self.pointer.add(source_offset).cast(), bytes, stream.raw())
        };
        self.context.driver.check(status, "asynchronous upload")
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
            self.context.driver_freed(self.bytes.max(1) as u64);
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

/// Function `entry` of `module`, loaded now: where modules load lazily
/// (the default from driver API 12.2) a function would otherwise load at its
/// first launch, a driver call that can wait behind a kernel spinning on the
/// host, so every function is loaded as its module is prepared, never while
/// serving. A driver that loads lazily but cannot load a function on demand
/// (12.2 and 12.3) loads it at its first launch.
pub(crate) fn module_function(module: &Module, entry: &str) -> Result<Handle, JitError> {
    let _current = module.context.enter().map_err(JitError::Driver)?;
    let name = std::ffi::CString::new(entry)
        .expect("PTX probe entry precondition: entry names contain no NUL byte");
    let mut function = std::ptr::null_mut();
    let driver = &module.context.driver;
    unsafe {
        driver
            .check((driver.module_function)(&mut function, module.raw, name.as_ptr()), "kernel lookup")
            .map_err(JitError::Driver)?;
        if let Some(load) = driver.function_load {
            driver.check(load(function), "kernel loading").map_err(JitError::Driver)?;
        }
    }
    Ok(function)
}

/// Whether the module defines the global variable `name`.
pub(crate) fn module_defines(module: &Module, name: &str) -> Result<bool, DriverError> {
    // CUDA_ERROR_NOT_FOUND
    const NOT_FOUND: ResultCode = 500;
    let _current = module.context.enter()?;
    let name = std::ffi::CString::new(name)
        .expect("module global precondition: names contain no NUL byte");
    let mut pointer = 0u64;
    let mut bytes = 0usize;
    let status = unsafe {
        (module.context.driver.module_global)(&mut pointer, &mut bytes, module.raw, name.as_ptr())
    };
    if status == NOT_FOUND {
        return Ok(false);
    }
    module.context.driver.check(status, "module global lookup")?;
    Ok(true)
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
