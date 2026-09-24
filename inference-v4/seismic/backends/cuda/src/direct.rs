//! Formation and execution of explicitly authored CUDA C++ native
//! implementations: NVRTC to CUBIN for the opened device, launched on the
//! device's stream. It does not construct compiler kernels or plans.

use crate::buffer::Buffer;
use crate::driver::{self, DriverError, Event, Handle, JitError, Module};
use crate::executor::Device;
use crate::nvrtc::{self, NvrtcError};
use seismic_compiler::errors::ExecutionError;
use seismic_target::NativeCompilationError;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// cuFuncGetAttribute / cuFuncSetAttribute keys.
const MAX_THREADS_PER_BLOCK: i32 = 0;
const SHARED_SIZE_BYTES: i32 = 1;
const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

/// The kernel functions of one authored CUDA source, formed for one device.
pub struct DirectModule {
    /// Keeps the functions' module loaded.
    _module: Arc<Module>,
    functions: Vec<DirectFunction>,
}

struct DirectFunction {
    raw: Handle,
    max_threads: u64,
    static_shared: u64,
    /// Largest dynamic shared size currently admitted by the function
    /// attribute; raised on demand.
    dynamic_shared_limit: AtomicU64,
}

// Loaded functions are immutable driver objects owned by `module`.
unsafe impl Send for DirectModule {}
unsafe impl Sync for DirectModule {}

fn formation_error(error: NvrtcError) -> NativeCompilationError {
    match error {
        NvrtcError::Compilation { log } => NativeCompilationError::ToolchainFailure(log),
        other => NativeCompilationError::ToolchainFailure(other.to_string()),
    }
}

fn jit_error(error: JitError) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(match error {
        JitError::Toolchain { error, log } => format!("{error}\n{log}"),
        JitError::Driver(error) => error.to_string(),
        JitError::MalformedImage(message) => message,
    })
}

fn driver_failure(error: DriverError) -> NativeCompilationError {
    NativeCompilationError::ToolchainFailure(error.to_string())
}

fn submission(error: DriverError) -> ExecutionError {
    if error.is_device_loss() {
        ExecutionError::DeviceLost(error.to_string())
    } else {
        ExecutionError::SubmissionFailed(error.to_string())
    }
}

impl DirectModule {
    /// Form `source` with NVRTC for `sm_<architecture>` and load each named
    /// kernel, in the given order.
    pub fn compile(
        device: &Device,
        source: &str,
        name: &str,
        architecture: u32,
        kernels: &[&str],
    ) -> Result<Self, NativeCompilationError> {
        let cubin = nvrtc::compile_cubin(source, name, architecture).map_err(formation_error)?;
        let context = device.context();
        let first = kernels.first().ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput("native module has no kernels".into())
        })?;
        let (module, _) = driver::load_module(context, &cubin, first).map_err(jit_error)?;
        let mut functions = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            let raw = driver::module_function(&module, kernel).map_err(jit_error)?;
            let attribute = |key| {
                driver::function_attribute(context, raw, key)
                    .map_err(driver_failure)
                    .map(|value| u64::try_from(value).unwrap_or(0))
            };
            functions.push(DirectFunction {
                raw,
                max_threads: attribute(MAX_THREADS_PER_BLOCK)?,
                static_shared: attribute(SHARED_SIZE_BYTES)?,
                dynamic_shared_limit: AtomicU64::new(attribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)?),
            });
        }
        Ok(Self {
            _module: Arc::new(module),
            functions,
        })
    }

    pub fn max_threads_per_block(&self, function: usize) -> u64 {
        self.functions[function].max_threads
    }

    pub fn static_shared_bytes(&self, function: usize) -> u64 {
        self.functions[function].static_shared
    }
}

/// One direct launch as issued.
pub struct DirectLaunch<'a> {
    pub module: &'a DirectModule,
    pub function: usize,
    /// Buffer arguments in ABI order, as `(buffer, byte offset)`.
    pub buffers: &'a [(&'a Buffer, u64)],
    /// The argument words, passed by value as one struct parameter.
    pub words: &'a [u8],
    pub scalar_results: (&'a Buffer, u64),
    pub grid: [u64; 3],
    pub block: [u64; 3],
    pub shared_bytes: u64,
}

/// Launches issued in order on the device's stream. The stream orders each
/// launch after the previous one's writes.
pub struct DirectBatch {
    device: Device,
    start: Event,
}

impl DirectBatch {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let start = Event::new(device.context()).map_err(submission)?;
        start.record(device.stream()).map_err(submission)?;
        Ok(Self {
            device: device.clone(),
            start,
        })
    }

    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Ok(());
        }
        let function = &launch.module.functions[launch.function];
        let threads = launch.block.iter().product::<u64>();
        if threads > function.max_threads {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per block, but the function allows {}",
                function.max_threads
            )));
        }
        let context = self.device.context();
        if launch.shared_bytes > function.dynamic_shared_limit.load(Ordering::Acquire) {
            let bytes = i32::try_from(launch.shared_bytes).map_err(|_| {
                ExecutionError::SubmissionFailed("dynamic shared size exceeds driver ABI".into())
            })?;
            driver::set_function_attribute(
                context,
                function.raw,
                MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
            .map_err(submission)?;
            function
                .dynamic_shared_limit
                .fetch_max(launch.shared_bytes, Ordering::AcqRel);
        }
        let mut pointers = launch
            .buffers
            .iter()
            .map(|(buffer, offset)| buffer.pointer() + offset)
            .collect::<Vec<u64>>();
        pointers.push(launch.scalar_results.0.pointer() + launch.scalar_results.1);
        // Parameter order: buffers, words (by value), scalar results.
        let buffer_count = launch.buffers.len();
        // The prefix declares at least one word, so an entry without words
        // still passes one zeroed word.
        let mut words = launch.words.to_vec();
        if words.is_empty() {
            words = vec![0; 8];
        }
        let mut parameters: Vec<*mut c_void> = Vec::with_capacity(buffer_count + 2);
        for pointer in pointers.iter_mut().take(buffer_count) {
            parameters.push((pointer as *mut u64).cast());
        }
        parameters.push(words.as_mut_ptr().cast());
        parameters.push((&mut pointers[buffer_count] as *mut u64).cast());
        let dimension = |value: u64| {
            u32::try_from(value).map_err(|_| {
                ExecutionError::SubmissionFailed("native launch geometry exceeds driver ABI".into())
            })
        };
        let _current = context.enter().map_err(submission)?;
        let driver = &context.driver;
        let status = unsafe {
            (driver.launch)(
                function.raw,
                dimension(launch.grid[0])?,
                dimension(launch.grid[1])?,
                dimension(launch.grid[2])?,
                dimension(launch.block[0])?,
                dimension(launch.block[1])?,
                dimension(launch.block[2])?,
                dimension(launch.shared_bytes)?,
                self.device.stream().raw(),
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        driver
            .check(status, "native kernel launch")
            .map_err(submission)
    }

    /// Record completion without waiting.
    pub fn commit(self) -> Result<DirectSubmission, ExecutionError> {
        let end = Event::new(self.device.context()).map_err(submission)?;
        end.record(self.device.stream()).map_err(submission)?;
        Ok(DirectSubmission {
            start: self.start,
            end,
        })
    }
}

/// Issued direct launches, observed through their completion event.
pub struct DirectSubmission {
    start: Event,
    end: Event,
}

// Events are driver objects usable from any thread with their context made
// current, which every method does.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    pub fn is_complete(&self) -> bool {
        // A failed query means the work cannot make further progress.
        self.end.query().unwrap_or(true)
    }

    pub fn wait_complete(&self) {
        let _ = self.end.synchronize();
    }

    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.end.synchronize().map_err(submission)
    }

    /// Device execution time between the batch's start and end events.
    pub fn device_seconds(&self) -> Result<f64, ExecutionError> {
        Event::elapsed_ns(&self.start, &self.end)
            .map(|nanoseconds| nanoseconds / 1e9)
            .map_err(submission)
    }
}

/// The NVRTC identity recorded in native artifact identities.
pub fn nvrtc_version() -> Result<(u32, u32), NativeCompilationError> {
    nvrtc::version().map_err(|error| NativeCompilationError::ToolchainFailure(error.to_string()))
}
