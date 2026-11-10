//! The one way device work is enqueued: the driver's enqueue calls are
//! private to [`Enqueue`], so every enqueue site goes through its methods
//! (reached as `Context::enqueue`). Each method is the driver's call
//! (`unsafe` as it is), returning its result code; an optional call returns
//! none where the driver lacks it.

use libloading::Library;
use std::ffi::{c_int, c_uchar, c_uint, c_void};

type Handle = *mut c_void;
type ResultCode = c_int;
type Launch = unsafe extern "system" fn(
    Handle,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> ResultCode;
type StreamValue = unsafe extern "system" fn(Handle, u64, u32, c_uint) -> ResultCode;
/// `CUhostFn`: a host function a stream runs once its preceding work completes.
pub(crate) type HostFunction = unsafe extern "system" fn(*mut c_void);

/// The driver's enqueue calls, private to this module.
pub(crate) struct Enqueue {
    launch: Launch,
    launch_cooperative: Launch,
    upload_async: unsafe extern "system" fn(u64, *const c_void, usize, Handle) -> ResultCode,
    download_async: unsafe extern "system" fn(*mut c_void, u64, usize, Handle) -> ResultCode,
    memcpy_device_async: unsafe extern "system" fn(u64, u64, usize, Handle) -> ResultCode,
    memset_d8_async: unsafe extern "system" fn(u64, c_uchar, usize, Handle) -> ResultCode,
    memset_d16_async: unsafe extern "system" fn(u64, u16, usize, Handle) -> ResultCode,
    memset_d32_async: unsafe extern "system" fn(u64, c_uint, usize, Handle) -> ResultCode,
    memset_d2d32_async: unsafe extern "system" fn(u64, usize, c_uint, usize, usize, Handle) -> ResultCode,
    event_record: unsafe extern "system" fn(Handle, Handle) -> ResultCode,
    stream_wait_event: unsafe extern "system" fn(Handle, Handle, c_uint) -> ResultCode,
    launch_host_function: unsafe extern "system" fn(Handle, HostFunction, *mut c_void) -> ResultCode,
    // Driver API 11.7+: stream memory operations on a 32-bit word.
    stream_wait_value_32: Option<StreamValue>,
    stream_write_value_32: Option<StreamValue>,
    // Driver API 12.0+: explicitly built kernel graphs.
    graph_launch: Option<unsafe extern "system" fn(Handle, Handle) -> ResultCode>,
}

impl Enqueue {
    /// The enqueue calls of `library` (the loaded driver), its optional
    /// ones where it has them.
    pub(crate) fn load(library: &Library) -> Result<Self, String> {
        unsafe fn required<T: Copy>(library: &Library, symbol: &str) -> Result<T, String> {
            unsafe {
                library
                    .get::<T>(format!("{symbol}\0").as_bytes())
                    .map(|symbol| *symbol)
                    .map_err(|error| format!("CUDA driver symbol {symbol}: {error}"))
            }
        }
        unsafe fn optional<T: Copy>(library: &Library, symbol: &str) -> Option<T> {
            unsafe { library.get::<T>(format!("{symbol}\0").as_bytes()).ok().map(|symbol| *symbol) }
        }
        unsafe {
            Ok(Self {
                launch: required(library, "cuLaunchKernel")?,
                launch_cooperative: required(library, "cuLaunchCooperativeKernel")?,
                upload_async: required(library, "cuMemcpyHtoDAsync_v2")?,
                download_async: required(library, "cuMemcpyDtoHAsync_v2")?,
                memcpy_device_async: required(library, "cuMemcpyDtoDAsync_v2")?,
                memset_d8_async: required(library, "cuMemsetD8Async")?,
                memset_d16_async: required(library, "cuMemsetD16Async")?,
                memset_d32_async: required(library, "cuMemsetD32Async")?,
                memset_d2d32_async: required(library, "cuMemsetD2D32Async")?,
                event_record: required(library, "cuEventRecord")?,
                stream_wait_event: required(library, "cuStreamWaitEvent")?,
                launch_host_function: required(library, "cuLaunchHostFunc")?,
                stream_wait_value_32: optional(library, "cuStreamWaitValue32_v2"),
                stream_write_value_32: optional(library, "cuStreamWriteValue32_v2"),
                graph_launch: optional(library, "cuGraphLaunch"),
            })
        }
    }

    /// Whether the driver has stream memory operations (driver API 11.7).
    pub(crate) fn stream_values(&self) -> bool {
        self.stream_wait_value_32.is_some() && self.stream_write_value_32.is_some()
    }

    /// `cuLaunchKernel`, or `cuLaunchCooperativeKernel` where `cooperative`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        cooperative: bool,
        function: Handle,
        grid: [c_uint; 3],
        block: [c_uint; 3],
        shared: c_uint,
        stream: Handle,
        parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> ResultCode {
        let launch = if cooperative { self.launch_cooperative } else { self.launch };
        unsafe { launch(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, stream, parameters, extra) }
    }
    pub unsafe fn upload_async(&self, destination: u64, source: *const c_void, bytes: usize, stream: Handle) -> ResultCode {
        unsafe { (self.upload_async)(destination, source, bytes, stream) }
    }
    pub unsafe fn download_async(&self, destination: *mut c_void, source: u64, bytes: usize, stream: Handle) -> ResultCode {
        unsafe { (self.download_async)(destination, source, bytes, stream) }
    }
    pub unsafe fn memcpy_device_async(&self, destination: u64, source: u64, bytes: usize, stream: Handle) -> ResultCode {
        unsafe { (self.memcpy_device_async)(destination, source, bytes, stream) }
    }
    pub unsafe fn memset_d8_async(&self, destination: u64, value: c_uchar, count: usize, stream: Handle) -> ResultCode {
        unsafe { (self.memset_d8_async)(destination, value, count, stream) }
    }
    pub unsafe fn memset_d16_async(&self, destination: u64, value: u16, count: usize, stream: Handle) -> ResultCode {
        unsafe { (self.memset_d16_async)(destination, value, count, stream) }
    }
    pub unsafe fn memset_d32_async(&self, destination: u64, value: c_uint, count: usize, stream: Handle) -> ResultCode {
        unsafe { (self.memset_d32_async)(destination, value, count, stream) }
    }
    pub unsafe fn memset_d2d32_async(
        &self,
        destination: u64,
        pitch: usize,
        value: c_uint,
        width: usize,
        height: usize,
        stream: Handle,
    ) -> ResultCode {
        unsafe { (self.memset_d2d32_async)(destination, pitch, value, width, height, stream) }
    }
    pub unsafe fn event_record(&self, event: Handle, stream: Handle) -> ResultCode {
        unsafe { (self.event_record)(event, stream) }
    }
    /// `cuStreamWaitEvent`: `stream`'s later work waits for `event`.
    pub unsafe fn stream_wait_event(&self, stream: Handle, event: Handle) -> ResultCode {
        unsafe { (self.stream_wait_event)(stream, event, 0) }
    }
    /// `cuLaunchHostFunc`: `stream` runs `function(data)` once its preceding
    /// work completes.
    pub unsafe fn launch_host_function(&self, stream: Handle, function: HostFunction, data: *mut c_void) -> ResultCode {
        unsafe { (self.launch_host_function)(stream, function, data) }
    }
    pub unsafe fn wait_value_32(&self, stream: Handle, address: u64, value: u32, flags: c_uint) -> Option<ResultCode> {
        self.stream_wait_value_32.map(|wait| unsafe { wait(stream, address, value, flags) })
    }
    pub unsafe fn write_value_32(&self, stream: Handle, address: u64, value: u32, flags: c_uint) -> Option<ResultCode> {
        self.stream_write_value_32.map(|write| unsafe { write(stream, address, value, flags) })
    }
    pub unsafe fn graph_launch(&self, graph: Handle, stream: Handle) -> Option<ResultCode> {
        self.graph_launch.map(|launch| unsafe { launch(graph, stream) })
    }
}
