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
    formation: nvrtc::Formation,
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
        NvrtcError::Unavailable(reason) => NativeCompilationError::ToolchainUnavailable(reason),
        NvrtcError::UnsupportedArchitecture {
            architecture,
            supported,
        } => NativeCompilationError::UnsupportedArchitecture {
            architecture: format!("sm_{architecture}"),
            supported: supported
                .into_iter()
                .map(|architecture| format!("sm_{architecture}"))
                .collect(),
        },
        NvrtcError::Compilation { log } => NativeCompilationError::ToolchainFailure(log),
        call @ NvrtcError::Call { .. } => NativeCompilationError::ToolchainFailure(call.to_string()),
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
    /// Form `source` for `sm_<architecture>` and load each named kernel, in
    /// the given order. `stored` may supply an image formed earlier under the
    /// same formation (compiler release, architecture, options); an image
    /// the driver does not load is ignored. Otherwise NVRTC compiles the
    /// source and `store` receives the new image.
    pub fn form(
        device: &Device,
        source: &str,
        name: &str,
        architecture: u32,
        kernels: &[&str],
        stored: impl FnOnce(&nvrtc::Formation) -> Option<Vec<u8>>,
        store: impl FnOnce(&nvrtc::Formation, &[u8]),
    ) -> Result<Self, NativeCompilationError> {
        let formation = nvrtc::formation(architecture).map_err(formation_error)?;
        if let Some(image) = stored(&formation) {
            if let Ok(module) = Self::load(device, &image, formation, kernels) {
                return Ok(module);
            }
        }
        let cubin = nvrtc::compile_cubin(source, name, architecture).map_err(formation_error)?;
        store(&cubin.formation, &cubin.image);
        Self::load(device, &cubin.image, cubin.formation, kernels)
    }

    /// Load a CUBIN image and each named kernel of it.
    fn load(
        device: &Device,
        image: &[u8],
        formation: nvrtc::Formation,
        kernels: &[&str],
    ) -> Result<Self, NativeCompilationError> {
        let context = device.context();
        let first = kernels.first().ok_or_else(|| {
            NativeCompilationError::MalformedToolchainOutput("native module has no kernels".into())
        })?;
        let (module, _) = driver::load_module(context, image, first).map_err(jit_error)?;
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
            formation,
        })
    }

    /// The compiler release, architecture and options that formed this
    /// module; together with the source they determine its code.
    pub fn formation(&self) -> &nvrtc::Formation {
        &self.formation
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

/// A non-empty launch checked against its function and in driver form.
struct FormedLaunch {
    function: Handle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    /// Buffer addresses in ABI order, then the scalar-result address.
    pointers: Vec<u64>,
    words: Vec<u8>,
}

impl FormedLaunch {
    /// `None` for an empty grid. Raises the function's dynamic shared-memory
    /// limit when the launch needs more.
    fn form(device: &Device, launch: &DirectLaunch<'_>) -> Result<Option<Self>, ExecutionError> {
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Ok(None);
        }
        let function = &launch.module.functions[launch.function];
        let threads = launch.block.iter().product::<u64>();
        if threads > function.max_threads {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per block, but the function allows {}",
                function.max_threads
            )));
        }
        if launch.shared_bytes > function.dynamic_shared_limit.load(Ordering::Acquire) {
            let bytes = i32::try_from(launch.shared_bytes).map_err(|_| {
                ExecutionError::SubmissionFailed("dynamic shared size exceeds driver ABI".into())
            })?;
            driver::set_function_attribute(
                device.context(),
                function.raw,
                MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
            .map_err(submission)?;
            function
                .dynamic_shared_limit
                .fetch_max(launch.shared_bytes, Ordering::AcqRel);
        }
        let mut pointers = Vec::with_capacity(launch.buffers.len() + 1);
        pointers.extend(
            launch
                .buffers
                .iter()
                .map(|(buffer, offset)| buffer.pointer() + offset),
        );
        pointers.push(launch.scalar_results.0.pointer() + launch.scalar_results.1);
        // The prefix declares at least one word, so an entry without words
        // still passes one zeroed word.
        let words = if launch.words.is_empty() {
            vec![0; 8]
        } else {
            launch.words.to_vec()
        };
        let dimension = |value: u64| {
            u32::try_from(value).map_err(|_| {
                ExecutionError::SubmissionFailed("native launch geometry exceeds driver ABI".into())
            })
        };
        let axes = |values: [u64; 3]| -> Result<[u32; 3], ExecutionError> {
            Ok([
                dimension(values[0])?,
                dimension(values[1])?,
                dimension(values[2])?,
            ])
        };
        Ok(Some(Self {
            function: function.raw,
            grid: axes(launch.grid)?,
            block: axes(launch.block)?,
            shared_bytes: dimension(launch.shared_bytes)?,
            pointers,
            words,
        }))
    }

    /// The kernel parameter array, pointing into `self`: buffers, the words
    /// struct (by value), the scalar-result address.
    fn parameters(&mut self) -> Vec<*mut c_void> {
        let (scalars, buffers) = self
            .pointers
            .split_last_mut()
            .expect("a formed launch carries its scalar-result address");
        let mut parameters: Vec<*mut c_void> = Vec::with_capacity(buffers.len() + 2);
        parameters.extend(
            buffers
                .iter_mut()
                .map(|pointer| (pointer as *mut u64).cast::<c_void>()),
        );
        parameters.push(self.words.as_mut_ptr().cast());
        parameters.push((scalars as *mut u64).cast());
        parameters
    }
}

/// Forms a [`DirectGraph`]: the launches of one submission, in order.
pub struct DirectGraphBuilder {
    device: Device,
    graph: driver::Graph,
}

impl DirectGraphBuilder {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            graph: driver::Graph::new(device.context()).map_err(submission)?,
        })
    }

    /// Append a launch after every launch added before it; an empty grid
    /// adds nothing. The argument values are copied into the graph.
    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(mut formed) = FormedLaunch::form(&self.device, launch)? else {
            return Ok(());
        };
        let mut parameters = formed.parameters();
        self.graph
            .push_kernel(&driver::KernelNodeParams {
                function: formed.function,
                grid: formed.grid,
                block: formed.block,
                shared_bytes: formed.shared_bytes,
                parameters: parameters.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kernel: std::ptr::null_mut(),
                context: std::ptr::null_mut(),
            })
            .map_err(submission)
    }

    pub fn instantiate(self) -> Result<DirectGraph, ExecutionError> {
        Ok(DirectGraph {
            exec: self.graph.instantiate().map_err(submission)?,
        })
    }
}

/// An instantiated CUDA graph of direct launches with fixed arguments. One
/// graph launch replaces its launches' individual driver calls; the device
/// runs them in order, as the stream would.
pub struct DirectGraph {
    exec: driver::GraphExec,
}

/// Launches issued in order on the device's stream. The stream orders each
/// launch after the previous one's writes.
pub struct DirectBatch {
    device: Device,
    start: Event,
    /// For a timed batch, per `launch` call, the event recorded just before
    /// the launch (absent for an empty launch).
    marks: Option<Vec<Option<Event>>>,
}

impl DirectBatch {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let start = Event::new(device.context()).map_err(submission)?;
        start.record(device.stream()).map_err(submission)?;
        Ok(Self {
            device: device.clone(),
            start,
            marks: None,
        })
    }

    /// A measurement batch that records an event before every launch, so
    /// each launch's device interval is known. The events add stream work;
    /// a timed batch attributes time but does not measure production.
    pub fn timed(device: &Device) -> Result<Self, ExecutionError> {
        let mut batch = Self::new(device)?;
        batch.marks = Some(Vec::new());
        Ok(batch)
    }

    /// Record a launch that does no device work (inactive, or an empty
    /// grid): nothing is issued, and a timed batch keeps its place with no
    /// interval.
    pub fn skip(&mut self) {
        if let Some(marks) = &mut self.marks {
            marks.push(None);
        }
    }

    pub fn launch(&mut self, launch: &DirectLaunch<'_>) -> Result<(), ExecutionError> {
        let Some(mut formed) = FormedLaunch::form(&self.device, launch)? else {
            self.skip();
            return Ok(());
        };
        if let Some(marks) = &mut self.marks {
            let mark = Event::new(self.device.context()).map_err(submission)?;
            mark.record(self.device.stream()).map_err(submission)?;
            marks.push(Some(mark));
        }
        let mut parameters = formed.parameters();
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        let driver = &context.driver;
        let status = unsafe {
            (driver.launch)(
                formed.function,
                formed.grid[0],
                formed.grid[1],
                formed.grid[2],
                formed.block[0],
                formed.block[1],
                formed.block[2],
                formed.shared_bytes,
                self.device.stream().raw(),
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        driver
            .check(status, "native kernel launch")
            .map_err(submission)
    }

    /// Queue every launch of `graph`, in its order. A timed batch launches
    /// individually instead, so it never replays.
    pub fn replay(&mut self, graph: &DirectGraph) -> Result<(), ExecutionError> {
        assert!(
            self.marks.is_none(),
            "DirectBatch::replay precondition: a timed batch times each launch"
        );
        graph
            .exec
            .launch(self.device.stream())
            .map_err(submission)
    }

    /// Record completion without waiting.
    pub fn commit(self) -> Result<DirectSubmission, ExecutionError> {
        let end = Event::new(self.device.context()).map_err(submission)?;
        end.record(self.device.stream()).map_err(submission)?;
        Ok(DirectSubmission {
            start: self.start,
            end,
            marks: self.marks,
        })
    }
}

/// A completed stream event paired with the host clock time read just after
/// it completed: the origin that places device events on the host timeline.
pub struct TimelineAnchor {
    event: Event,
    host_seconds: f64,
}

// See `DirectSubmission`.
unsafe impl Send for TimelineAnchor {}
unsafe impl Sync for TimelineAnchor {}

impl TimelineAnchor {
    /// Record an event on the device's (idle) stream, wait for it, and read
    /// `host_clock`. The pairing error is the synchronization latency.
    pub fn record(device: &Device, host_clock: impl FnOnce() -> f64) -> Result<Self, ExecutionError> {
        let event = Event::new(device.context()).map_err(submission)?;
        event.record(device.stream()).map_err(submission)?;
        event.synchronize().map_err(submission)?;
        Ok(Self {
            event,
            host_seconds: host_clock(),
        })
    }

    fn place(&self, event: &Event) -> Result<f64, ExecutionError> {
        Event::elapsed_ns(&self.event, event)
            .map(|nanoseconds| self.host_seconds + nanoseconds / 1e9)
            .map_err(submission)
    }
}

/// Issued direct launches, observed through their completion event.
pub struct DirectSubmission {
    start: Event,
    end: Event,
    marks: Option<Vec<Option<Event>>>,
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

    /// The completed batch's device interval on the anchor's host clock.
    pub fn device_interval(&self, anchor: &TimelineAnchor) -> Result<(f64, f64), ExecutionError> {
        Ok((anchor.place(&self.start)?, anchor.place(&self.end)?))
    }

    /// For a timed batch, each `launch` call's device interval on the
    /// anchor's host clock, in launch order (`None` for an empty launch). A
    /// launch ends where the next recorded launch (or the batch) ends.
    /// `None` for a production batch. Call after completion.
    pub fn launch_intervals(
        &self,
        anchor: &TimelineAnchor,
    ) -> Result<Option<Vec<Option<(f64, f64)>>>, ExecutionError> {
        let Some(marks) = &self.marks else {
            return Ok(None);
        };
        let mut intervals = Vec::with_capacity(marks.len());
        for (index, mark) in marks.iter().enumerate() {
            let Some(mark) = mark else {
                intervals.push(None);
                continue;
            };
            let end = marks[index + 1..]
                .iter()
                .flatten()
                .next()
                .unwrap_or(&self.end);
            intervals.push(Some((anchor.place(mark)?, anchor.place(end)?)));
        }
        Ok(Some(intervals))
    }
}
