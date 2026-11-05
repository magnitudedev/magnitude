//! Formation and execution of explicitly authored CUDA C++ native
//! implementations: NVRTC to CUBIN for the opened device, launched on the
//! device's stream. It does not construct compiler kernels or plans.

use crate::buffer::Buffer;
use crate::driver::{self, DriverError, Event, Handle, JitError, Module};
use crate::executor::Device;
use crate::nvrtc::NvrtcError;
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCensus;
use seismic_native_target::NativeCompilationError;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

// cuFuncGetAttribute / cuFuncSetAttribute keys.
const MAX_THREADS_PER_BLOCK: i32 = 0;
const SHARED_SIZE_BYTES: i32 = 1;
const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

/// The kernel functions of one CUDA program, loaded on one device.
pub struct DirectModule {
    /// Keeps the functions' module loaded.
    _module: Arc<Module>,
    functions: Vec<DirectFunction>,
    /// How a replayed launch of this module depends on the launch before
    /// it: programmatic when the source declares
    /// `SEISMIC_PROGRAMMATIC_DEPENDENCY` (every kernel waits for its
    /// predecessor itself) and the driver supports typed edges.
    edge: driver::Edge,
}

/// The global a source defines through `SEISMIC_PROGRAMMATIC_DEPENDENCY`.
const PROGRAMMATIC_DECLARATION: &str = "seismic_programmatic_dependency";

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

pub(crate) fn formation_error(error: NvrtcError) -> NativeCompilationError {
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
        call @ NvrtcError::Call { .. } => {
            NativeCompilationError::ToolchainFailure(call.to_string())
        }
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
    /// Whether a replayed launch of this module may start once the launch
    /// before it has triggered, not completed (programmatic dependent
    /// launch): how a graph node of it depends on the one before.
    pub fn programmatic(&self) -> bool {
        self.edge == driver::Edge::Programmatic
    }

    /// Load a CUBIN image and each named kernel of it, in order.
    pub(crate) fn load(
        device: &Device,
        image: &[u8],
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
        let edge = if context.driver.graph_add_dependencies.is_some()
            && driver::module_defines(&module, PROGRAMMATIC_DECLARATION).map_err(driver_failure)?
        {
            driver::Edge::Programmatic
        } else {
            driver::Edge::Complete
        };
        Ok(Self {
            _module: Arc::new(module),
            functions,
            edge,
        })
    }

    pub fn max_threads_per_block(&self, function: usize) -> u64 {
        self.functions[function].max_threads
    }

    pub fn static_shared_bytes(&self, function: usize) -> u64 {
        self.functions[function].static_shared
    }

    /// The blocks of `function` one multiprocessor holds at once with
    /// `threads` threads and `shared_bytes` dynamic shared bytes each.
    pub fn active_blocks(
        &self,
        device: &Device,
        function: usize,
        threads: u64,
        shared_bytes: u64,
    ) -> Result<u64, ExecutionError> {
        let function = &self.functions[function];
        function.admit_shared(device, shared_bytes)?;
        let threads = u32::try_from(threads).map_err(|_| {
            ExecutionError::SubmissionFailed("native block size exceeds driver ABI".into())
        })?;
        driver::occupancy_max_active_blocks(device.context(), function.raw, threads, shared_bytes)
            .map(u64::from)
            .map_err(submission)
    }
}

impl DirectFunction {
    /// Raises the function's dynamic shared-memory limit to `bytes` when it
    /// is lower.
    fn admit_shared(&self, device: &Device, bytes: u64) -> Result<(), ExecutionError> {
        if bytes > self.dynamic_shared_limit.load(Ordering::Acquire) {
            let value = i32::try_from(bytes).map_err(|_| {
                ExecutionError::SubmissionFailed("dynamic shared size exceeds driver ABI".into())
            })?;
            driver::set_function_attribute(device.context(), self.raw, MAX_DYNAMIC_SHARED_SIZE_BYTES, value)
                .map_err(submission)?;
            self.dynamic_shared_limit.fetch_max(bytes, Ordering::AcqRel);
        }
        Ok(())
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
    /// Scalar-result slots, passed after the words; a null pointer for an
    /// entry without scalar results.
    pub scalar_results: Option<(&'a Buffer, u64)>,
    pub grid: [u64; 3],
    pub block: [u64; 3],
    pub shared_bytes: u64,
}

impl DirectLaunch<'_> {
    /// Whether the launch does no device work: an empty grid or block.
    pub fn is_empty(&self) -> bool {
        self.grid.contains(&0) || self.block.contains(&0)
    }
}

/// A non-empty launch checked against its function and in driver form.
#[derive(PartialEq, Eq)]
struct FormedLaunch {
    function: Handle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    /// Buffer addresses in ABI order, then the scalar-result address.
    pointers: Vec<u64>,
    words: Vec<u8>,
    edge: driver::Edge,
}

// A formed launch's function is a loaded, immutable driver object
// (`DirectModule`), the rest plain values.
unsafe impl Send for FormedLaunch {}
unsafe impl Sync for FormedLaunch {}

impl FormedLaunch {
    /// The launch as a node's parameters. Raises the function's dynamic
    /// shared-memory limit when the launch needs more. A launch that does
    /// not run (`enabled` false) may do no device work: an empty axis of its
    /// grid is formed as one block, which the driver requires of any node.
    fn form(device: &Device, launch: &DirectLaunch<'_>, enabled: bool) -> Result<Self, ExecutionError> {
        let function = &launch.module.functions[launch.function];
        let threads = launch.block.iter().product::<u64>();
        if threads > function.max_threads {
            return Err(ExecutionError::SubmissionFailed(format!(
                "native launch requests {threads} threads per block, but the function allows {}",
                function.max_threads
            )));
        }
        function.admit_shared(device, launch.shared_bytes)?;
        let mut pointers = Vec::with_capacity(launch.buffers.len() + 1);
        pointers.extend(
            launch
                .buffers
                .iter()
                .map(|(buffer, offset)| buffer.pointer() + offset),
        );
        pointers.push(
            launch
                .scalar_results
                .map_or(0, |(buffer, offset)| buffer.pointer() + offset),
        );
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
        Ok(Self {
            function: function.raw,
            grid: axes(match enabled {
                true => launch.grid,
                false => launch.grid.map(|blocks| blocks.max(1)),
            })?,
            block: axes(match enabled {
                true => launch.block,
                false => launch.block.map(|threads| threads.max(1)),
            })?,
            shared_bytes: dimension(launch.shared_bytes)?,
            pointers,
            words,
            edge: launch.module.edge,
        })
    }

    /// The launch as a graph's kernel node, its arguments `parameters`
    /// ([`Self::parameters`]).
    fn node(&self, parameters: &mut [*mut c_void]) -> driver::KernelNodeParams {
        driver::KernelNodeParams {
            function: self.function,
            grid: self.grid,
            block: self.block,
            shared_bytes: self.shared_bytes,
            parameters: parameters.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kernel: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        }
    }

    /// The kernel parameter array, pointing into `self`: buffers, the words
    /// struct (by value) and the scalar-result address.
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

/// Forms a [`DirectGraph`]: the launches of one submission, each after the
/// earlier ones it depends on.
pub struct DirectGraphBuilder {
    device: Device,
    graph: driver::Graph,
    /// Each node's launch as formed, in order of addition, and whether it
    /// runs.
    formed: Vec<FormedLaunch>,
    enabled: Vec<bool>,
    /// What its marks record until a sampled run sets its own
    /// ([`DirectGraph::sample`]), where it has any.
    rest: Option<Arc<Event>>,
    /// The launches added so far ([`DirectGraph::census`]).
    launches: GraphCensus,
}

impl DirectGraphBuilder {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            graph: driver::Graph::new(device.context()).map_err(submission)?,
            formed: Vec::new(),
            enabled: Vec::new(),
            rest: None,
            launches: GraphCensus::default(),
        })
    }

    /// Add a mark after the launches added so far: an event-record node in
    /// the graph's chain, stamping its event as the launch before it
    /// completes ([`DirectGraph::sample`]). It counts as a launch of the
    /// graph's memory.
    pub fn mark(&mut self) -> Result<(), ExecutionError> {
        let rest = match &self.rest {
            Some(rest) => rest.clone(),
            None => Arc::new(Event::stamp(self.device.context()).map_err(submission)?),
        };
        self.graph.push_mark(&rest).map_err(submission)?;
        self.rest = Some(rest);
        self.launches = self.launches.plus(GraphCensus::launch(0)).ok_or_else(|| {
            ExecutionError::SubmissionFailed("a graph's launch census overflows".into())
        })?;
        Ok(())
    }

    /// Add a launch after the launches `after` names, by their order of
    /// addition: it runs once they completed, beside every launch it does
    /// not follow, by the module's edge from each, where `enabled`; a node
    /// disabled runs as an empty one, its dependencies kept, until a replay
    /// enables it ([`DirectGraph::bind`]). A disabled launch may do no
    /// device work ([`DirectLaunch::is_empty`]). The argument values are
    /// copied into the graph.
    pub fn launch(&mut self, launch: &DirectLaunch<'_>, after: &[usize], enabled: bool) -> Result<(), ExecutionError> {
        let mut formed = FormedLaunch::form(&self.device, launch, enabled)?;
        let mut parameters = formed.parameters();
        let node = formed.node(&mut parameters);
        self.graph.push_kernel(&node, formed.edge, after).map_err(submission)?;
        self.formed.push(formed);
        self.enabled.push(enabled);
        // Its memory follows no parameter byte
        // (`seismic_compiler::graph_cost::argument_bytes`).
        self.launches = self.launches.plus(GraphCensus::launch(0)).ok_or_else(|| {
            ExecutionError::SubmissionFailed("a graph's launch census overflows".into())
        })?;
        Ok(())
    }

    /// Instantiate the graph and upload it to the device
    /// (`GraphExec::upload`), so its first launch does no setup. The driver
    /// memory both take, which no allocation charges, is measured as the
    /// change in the device's free memory across them and counted as memory
    /// the device holds until the graph is dropped. The driver has taken it
    /// once the graph exists, so it is recorded to the device account, never
    /// refused.
    pub fn instantiate(self) -> Result<DirectGraph, ExecutionError> {
        let before = self.device.memory_info()?.free_bytes;
        let exec = self.graph.instantiate().map_err(submission)?;
        for (node, _) in self.enabled.iter().enumerate().filter(|(_, enabled)| !**enabled) {
            exec.set_enabled(self.graph.node(node), false).map_err(submission)?;
        }
        exec.upload(self.device.stream()).map_err(submission)?;
        let device_bytes = before.saturating_sub(self.device.memory_info()?.free_bytes);
        let context = self.device.context().clone();
        let launched = Event::new(&context).map_err(submission)?;
        let census = GraphCensus::graph(self.launches).ok_or_else(|| {
            ExecutionError::SubmissionFailed("a graph's census overflows".into())
        })?;
        let charge = self.device.accounts().device.record(device_bytes);
        let held = self.device.graph_bytes().clone();
        held.fetch_add(device_bytes, std::sync::atomic::Ordering::Relaxed);
        context.driver_allocated(device_bytes);
        Ok(DirectGraph {
            _charge: charge,
            exec,
            launched,
            sampled: Vec::new(),
            rest: self.rest,
            graph: self.graph,
            formed: self.formed,
            enabled: self.enabled,
            device_bytes,
            census,
            held,
            context,
        })
    }
}

/// An instantiated CUDA graph of direct launches with fixed arguments. One
/// graph launch replaces its launches' individual driver calls; the device
/// runs each once those it depends on completed.
pub struct DirectGraph {
    exec: driver::GraphExec,
    /// Recorded behind each launch of the graph ([`DirectBatch::replay`]):
    /// whether its last launch completed, before which no update of it is
    /// made without waiting.
    launched: Event,
    /// The events a sampled run set on its marks, in order; empty where they
    /// record `rest` ([`Self::sample`]).
    sampled: Vec<Arc<Event>>,
    /// What its marks record at rest, where it has any.
    rest: Option<Arc<Event>>,
    /// The graph `exec` was instantiated from, whose nodes name the
    /// launches a parameter update sets ([`Self::bind`]).
    graph: driver::Graph,
    /// Each node's launch as the executable binds it now, and whether it
    /// runs.
    formed: Vec<FormedLaunch>,
    enabled: Vec<bool>,
    device_bytes: u64,
    /// What the graph is made of, as its driver memory follows it.
    census: GraphCensus,
    held: std::sync::Arc<std::sync::atomic::AtomicU64>,
    _charge: Box<dyn seismic_compiler::executable::MemoryHold>,
    /// The context whose driver memory the instantiation added to.
    context: std::sync::Arc<driver::Context>,
}

impl DirectGraph {
    /// Its launches.
    pub fn census(&self) -> GraphCensus {
        self.census
    }

    /// The graph's launches, enabled or not.
    pub fn nodes(&self) -> usize {
        self.formed.len()
    }

    /// Have its marks stamp `marks`, in order, for every launch of the graph
    /// after this call, or none: the events a sampled run reads, or the
    /// graph's own at rest. A mark is a node of the graph on every run: the
    /// driver refuses to disable an event-record node
    /// (`cuGraphNodeSetEnabled`, on an RTX 3090), so a run not sampled pays
    /// its stamp and nothing more. Its owner samples only an idle graph
    /// ([`Self::bind`]). The graph holds the events it stamps until it
    /// stamps others.
    pub fn sample(&mut self, marks: &[(&Stamps, usize)]) -> Result<(), ExecutionError> {
        if marks.is_empty() && self.sampled.is_empty() {
            return Ok(());
        }
        if marks.is_empty() {
            let rest = self.rest.as_ref().expect("a graph that was sampled has marks");
            for mark in 0..self.sampled.len() {
                self.exec.set_event(self.graph.mark(mark), rest).map_err(submission)?;
            }
            self.sampled.clear();
            return Ok(());
        }
        self.sampled.clear();
        for (node, (stamps, mark)) in marks.iter().enumerate() {
            let event = stamps.events[*mark].clone();
            self.exec.set_event(self.graph.mark(node), &event).map_err(submission)?;
            self.sampled.push(event);
        }
        Ok(())
    }

    /// Whether every launch of the graph queued so far completed: an update
    /// of it then waits for nothing. Its fault where the context faulted.
    pub fn idle(&self) -> Result<bool, ExecutionError> {
        self.launched.query().map_err(submission)
    }

    /// Bind `launch` at node `node` (by order of addition) for every launch
    /// of the graph after this call, running where `enabled`: its function,
    /// geometry, shared memory and arguments set where they changed (a
    /// graph is one per layout, every class, block and place of it binding
    /// its own), and the node enabled or disabled where that changed. Only
    /// how the node follows the one before is the graph's form, so another
    /// is an invariant violation. Launches already queued run as they were
    /// queued; binding a graph with launches in flight paces the host behind
    /// them (`graph_cost::graph_update_cost_probe`), so its owner binds only
    /// an idle graph ([`Self::idle`];
    /// [`driver::GraphExec::set_kernel`], [`driver::GraphExec::set_enabled`]).
    pub fn bind(&mut self, device: &Device, node: usize, launch: &DirectLaunch<'_>, enabled: bool) -> Result<(), ExecutionError> {
        if enabled {
            let mut formed = FormedLaunch::form(device, launch, true)?;
            assert!(formed.edge == self.formed[node].edge, "a graph's launch binds only parameters: its shape fixes its edges");
            if formed != self.formed[node] {
                let mut parameters = formed.parameters();
                let params = formed.node(&mut parameters);
                self.exec.set_kernel(self.graph.node(node), &params).map_err(submission)?;
                self.formed[node] = formed;
            }
        }
        self.set_enabled(node, enabled)
    }

    /// Disable node `node` (by order of addition) for every launch of the
    /// graph after this call: a launch the run does not make, which another
    /// run of the graph's layout binds. Its parameters stay as they were.
    pub fn disable(&mut self, node: usize) -> Result<(), ExecutionError> {
        self.set_enabled(node, false)
    }

    fn set_enabled(&mut self, node: usize, enabled: bool) -> Result<(), ExecutionError> {
        if enabled != self.enabled[node] {
            self.exec.set_enabled(self.graph.node(node), enabled).map_err(submission)?;
            self.enabled[node] = enabled;
        }
        Ok(())
    }
}

impl Drop for DirectGraph {
    fn drop(&mut self) {
        self.held
            .fetch_sub(self.device_bytes, std::sync::atomic::Ordering::Relaxed);
        self.context.driver_freed(self.device_bytes);
    }
}

/// A 32-bit word in pinned host memory mapped into the device's address
/// space, which the host and the device's stream both write and wait on.
/// A host thread waiting for it sleeps ([`Self::wait`]): every write, the
/// host's or a stream's ([`DirectBatch::write_word`]), wakes it.
pub struct HostWord {
    mapped: driver::HostMapped,
    wake: Arc<Wake>,
}

/// What a [`HostWord`]'s waiters sleep on, which every write of the word
/// notifies, under its lock, so no wake is lost between a waiter's look at
/// the word and its sleep.
pub(crate) struct Wake {
    lock: Mutex<()>,
    changed: Condvar,
}

impl Wake {
    fn notify(&self) {
        drop(self.lock.lock().expect("host word wake lock"));
        self.changed.notify_all();
    }
}

/// The host function a stream runs behind a device write of a word: `data`
/// is the word's [`Wake`], an `Arc` the write leaked for it.
unsafe extern "system" fn wake_after_write(data: *mut c_void) {
    // SAFETY: `DirectBatch::write_word` leaked one reference for this call.
    let wake = unsafe { Arc::from_raw(data.cast::<Wake>().cast_const()) };
    wake.notify();
}

impl HostWord {
    /// A word at 0.
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        let mapped = driver::HostMapped::new(device.context(), &device.accounts().upload, 4)?;
        mapped.write_at(0, &0u32.to_le_bytes());
        Ok(Self { mapped, wake: Arc::new(Wake { lock: Mutex::new(()), changed: Condvar::new() }) })
    }

    /// Wake the word's waiters now: what they wait on besides the word
    /// changed (a fault of its writer).
    pub(crate) fn wake(&self) {
        self.wake.notify();
    }

    /// Wake the word's waiters once `stream`'s work queued so far completes,
    /// a device write of the word among it: an event recorded behind it,
    /// which the context's notify stream waits for before it runs their
    /// wake, so no work of `stream` waits for the host. Every device write
    /// of a host word is followed by this, so every host wait sleeps.
    pub(crate) fn wake_after(&self, stream: &driver::Stream) -> Result<(), ExecutionError> {
        let context = stream.context();
        let written = Event::new(context).map_err(submission)?;
        written.record(stream).map_err(submission)?;
        let _current = context.enter().map_err(submission)?;
        let issue = context.issuing().map_err(submission)?;
        let notify = context.notify_stream();
        // SAFETY: both handles are this context's; the event may be
        // destroyed once the wait is queued, and the wake's reference is the
        // host function's to release.
        unsafe {
            context.driver.check(issue.stream_wait_event(notify, written.raw()), "wake wait").map_err(submission)?;
            let wake = Arc::into_raw(self.wake.clone()).cast_mut().cast::<c_void>();
            let status = issue.launch_host_function(notify, wake_after_write, wake);
            if status != 0 {
                drop(Arc::from_raw(wake.cast::<Wake>().cast_const()));
            }
            context.driver.check(status, "wake launch").map_err(submission)
        }
    }

    /// Wait at most `timeout` until `reached` holds of the word, sleeping
    /// between its writes; whether it did.
    pub fn wait(&self, reached: impl Fn(u32) -> bool, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut lock = self.wake.lock.lock().expect("host word wake lock");
        loop {
            if reached(self.load()) {
                return true;
            }
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return false;
            };
            lock = self.wake.changed.wait_timeout(lock, left).expect("host word wake lock").0;
        }
    }

    fn atomic(&self) -> &std::sync::atomic::AtomicU32 {
        // SAFETY: the mapping is four aligned bytes owned by `self`; the
        // device accesses them only through the word operations.
        unsafe { &*self.mapped.host().cast::<std::sync::atomic::AtomicU32>() }
    }

    /// Set the word from the host, after every host write before it, and
    /// wake its waiters.
    pub fn store(&self, value: u32) {
        self.atomic().store(value, Ordering::Release);
        self.wake.notify();
    }

    /// The word, with every write the device made before it.
    pub fn load(&self) -> u32 {
        self.atomic().load(Ordering::Acquire)
    }

    /// The address device work writes and waits on the word at.
    pub(crate) fn device_address(&self) -> u64 {
        self.mapped.device
    }
}

/// Device timestamps at marks of work on the device's stream: an event per
/// mark, recorded as the stream passes it ([`DirectBatch::stamp`]) or as a
/// graph's mark node does ([`DirectGraph::sample`]), so the device time
/// between two marks is read once the work has passed both.
pub struct Stamps {
    events: Vec<Arc<Event>>,
}

impl Stamps {
    /// The stamps of `marks` marks on `device`.
    pub fn new(device: &Device, marks: usize) -> Result<Self, ExecutionError> {
        let events = (0..marks)
            .map(|_| Event::stamp(device.context()).map(Arc::new).map_err(submission))
            .collect::<Result<_, _>>()?;
        Ok(Self { events })
    }

    /// Who these stamps are while any graph or run holds them: what a graph
    /// sampled by them is bound to.
    pub fn identity(&self) -> u64 {
        self.events.first().map_or(0, |event| event.raw() as u64)
    }

    /// The device seconds from mark `from` to mark `to`, once the stream has
    /// passed `to` (waited for here).
    pub fn seconds(&self, from: usize, to: usize) -> Result<f64, ExecutionError> {
        self.events[to].synchronize().map_err(submission)?;
        Event::elapsed_ns(&self.events[from], &self.events[to])
            .map(|nanoseconds| nanoseconds / 1e9)
            .map_err(submission)
    }
}

/// Launches issued in order on one of the device's native streams. The
/// stream orders each launch after the previous one's writes.
pub struct DirectBatch {
    device: Device,
    stream: std::sync::Arc<driver::Stream>,
    start: Event,
    /// For a timed batch, per `launch` call, the event recorded just before
    /// the launch (absent for an empty launch).
    marks: Option<Vec<Option<Event>>>,
}

impl DirectBatch {
    /// A batch on native stream `stream`.
    pub fn new(device: &Device, stream: crate::NativeStream) -> Result<Self, ExecutionError> {
        let stream = device.native_stream(stream).clone();
        let start = Event::new(device.context()).map_err(submission)?;
        start.record(&stream).map_err(submission)?;
        Ok(Self {
            device: device.clone(),
            stream,
            start,
            marks: None,
        })
    }

    /// A measurement batch that records an event before every launch, so
    /// each launch's device interval is known. The events add stream work;
    /// a timed batch attributes time but does not measure production.
    pub fn timed(device: &Device, stream: crate::NativeStream) -> Result<Self, ExecutionError> {
        let mut batch = Self::new(device, stream)?;
        batch.marks = Some(Vec::new());
        Ok(batch)
    }

    /// The batch's following work waits until `other`, issued on
    /// another of the device's streams, has completed (`cuStreamWaitEvent`
    /// on its end event): device work, waiting for no host.
    pub fn follow(&mut self, other: &DirectSubmission) -> Result<(), ExecutionError> {
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        // SAFETY: the stream and the event are this context's, and the end
        // event was recorded when its submission committed.
        let status = unsafe { context.issuing().map_err(submission)?.stream_wait_event(self.stream.raw(), other.end.raw()) };
        context.driver.check(status, "cross-stream wait").map_err(submission)
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
        if launch.is_empty() {
            self.skip();
            return Ok(());
        }
        let mut formed = FormedLaunch::form(&self.device, launch, true)?;
        if let Some(marks) = &mut self.marks {
            let mark = Event::new(self.device.context()).map_err(submission)?;
            mark.record(&self.stream).map_err(submission)?;
            marks.push(Some(mark));
        }
        let mut parameters = formed.parameters();
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        let issue = context.issuing().map_err(submission)?;
        let driver = &context.driver;
        let status = unsafe {
            issue.launch(
                false,
                formed.function,
                formed.grid,
                formed.block,
                formed.shared_bytes,
                self.stream.raw(),
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        drop(issue);
        driver
            .check(status, "native kernel launch")
            .map_err(submission)
    }

    /// Stamp mark `mark` of `stamps` once the stream's preceding work
    /// completes ([`Stamps`]).
    pub fn stamp(&mut self, stamps: &Stamps, mark: usize) -> Result<(), ExecutionError> {
        stamps.events[mark].record(&self.stream).map_err(submission)
    }

    /// The stream's following work waits until `word` reaches `value`
    /// (`cuStreamWaitValue32`, compared cyclically: `word - value` as I32
    /// at least 0): the host gate.
    pub fn wait_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        // CU_STREAM_WAIT_VALUE_GEQ
        const GEQ: u32 = 0;
        self.word_operation(word, value, GEQ, |issue, stream, address, value, flags| unsafe {
            issue.wait_value_32(stream, address, value, flags)
        })
    }

    /// `word` becomes `value` once the stream's preceding work completes,
    /// its writes visible first (`cuStreamWriteValue32` with its memory
    /// barrier); the word's waiters wake then ([`HostWord::wake_after`]).
    pub fn write_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        // CU_STREAM_WRITE_VALUE_DEFAULT
        const DEFAULT: u32 = 0;
        self.word_operation(word, value, DEFAULT, |issue, stream, address, value, flags| unsafe {
            issue.write_value_32(stream, address, value, flags)
        })?;
        word.wake_after(&self.stream)
    }

    fn word_operation(
        &mut self,
        word: &HostWord,
        value: u32,
        flags: u32,
        operation: impl FnOnce(&crate::registered::issue::Issuing<'_>, Handle, u64, u32, std::ffi::c_uint) -> Option<std::ffi::c_int>,
    ) -> Result<(), ExecutionError> {
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        let status = operation(&context.issuing().map_err(submission)?, self.stream.raw(), word.mapped.device, value, flags)
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed(
                    "the CUDA driver predates driver API 11.7 stream memory operations".into(),
                )
            })?;
        context
            .driver
            .check(status, "stream memory operation")
            .map_err(submission)
    }

    /// Queue every launch of `graph`, in its order. A timed batch launches
    /// individually instead, so it never replays.
    pub fn replay(&mut self, graph: &DirectGraph) -> Result<(), ExecutionError> {
        assert!(
            self.marks.is_none(),
            "DirectBatch::replay precondition: a timed batch times each launch"
        );
        graph.exec.launch(&self.stream).map_err(submission)?;
        graph.launched.record(&self.stream).map_err(submission)
    }

    /// Record completion without waiting.
    pub fn commit(self) -> Result<DirectSubmission, ExecutionError> {
        self.end()
    }

    /// End a batch that failed: the work it issued runs on, and the
    /// returned submission completes behind it, or, where its end event
    /// could not be recorded, the error ([`Self::end`]).
    pub fn abandon(mut self) -> Result<DirectSubmission, ExecutionError> {
        self.marks = None;
        self.end()
    }

    /// The submission an end event recorded behind every issued launch
    /// completes. Where the event cannot be recorded the stream is waited
    /// for first, and the error returned: the issued launches have
    /// finished, or (the wait failing too, both errors returned) the
    /// context's sticky error lets nothing more run, so no storage they
    /// reach is reused while they run.
    fn end(self) -> Result<DirectSubmission, ExecutionError> {
        let end = Event::new(self.device.context()).and_then(|end| {
            end.record(&self.stream)?;
            Ok(end)
        });
        match end {
            Ok(end) => Ok(DirectSubmission {
                start: self.start,
                end,
                marks: self.marks,
                complete: std::sync::atomic::AtomicBool::new(false),
            }),
            Err(error) => Err(match self.stream.synchronize() {
                Ok(()) => submission(error),
                Err(waited) => ExecutionError::IssueAndCompletionFailed {
                    issue: Box::new(submission(error)),
                    completion: Box::new(submission(waited)),
                },
            }),
        }
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
    pub fn record(
        device: &Device,
        host_clock: impl FnOnce() -> f64,
    ) -> Result<Self, ExecutionError> {
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
    /// The work was seen complete, which it stays: no later look asks the
    /// driver, whose event calls serialize with every other thread's.
    complete: std::sync::atomic::AtomicBool,
}

// Events are driver objects usable from any thread with their context made
// current, which every method does.
unsafe impl Send for DirectSubmission {}
unsafe impl Sync for DirectSubmission {}

impl DirectSubmission {
    /// Whether the work completed, without waiting: `Ok(false)` while it
    /// runs, and its fault where it ended without completing (the context
    /// faulted, which runs none of it any more).
    pub fn completed(&self) -> Result<bool, ExecutionError> {
        use std::sync::atomic::Ordering::Relaxed;
        if self.complete.load(Relaxed) {
            return Ok(true);
        }
        let complete = self.end.query().map_err(submission)?;
        self.complete.store(complete, Relaxed);
        Ok(complete)
    }

    /// Wait for the work to end and report its outcome: its fault where it
    /// ended without completing.
    pub fn finish(&self) -> Result<(), ExecutionError> {
        self.end.synchronize().map_err(submission)?;
        self.complete.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Each launch spins (about 0.1 s) before writing `slot + 1` to
    /// `out[slot]`, so a completion not behind it is seen as an unwritten
    /// slot.
    const SOURCE: &str = r#"
struct Words { unsigned long long slot; };
extern "C" __global__ void mark(unsigned int* out, Words words, unsigned int* scalars) {
    long long start = clock64();
    while (clock64() - start < 200000000LL) {}
    out[words.slot] = (unsigned int)words.slot + 1;
}
"#;

    /// A batch whose third launch fails to form after two were issued is
    /// abandoned into a submission that completes behind both, so the
    /// storage they write is fenced until they finish.
    #[test]
    fn an_abandoned_batch_completes_behind_the_launches_it_issued() {
        let Some((
            _exclusive,
            crate::OpenedCuda {
                service: device,
                device: description,
            },
        )) = crate::test_device()
        else {
            return;
        };
        let capability = description.facts().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin =
            crate::nvrtc::compile_cubin_named(SOURCE, "abandon.cu", architecture, &["mark"])
                .unwrap();
        let module =
            DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        let out = device.allocate_mapped(8).unwrap();
        let scalars = device.allocate_mapped(4).unwrap();
        let host = out.host_address().unwrap().cast::<u32>();
        // SAFETY: `out` maps 8 bytes the device does not use yet.
        unsafe { host.as_ptr().write_bytes(0, 2) };
        let buffers = [(&out, 0)];
        let words = [0u64.to_le_bytes(), 1u64.to_le_bytes()];
        let launch = |slot: usize, threads: u64| DirectLaunch {
            module: &module,
            function: 0,
            buffers: &buffers,
            words: &words[slot],
            scalar_results: Some((&scalars, 0)),
            grid: [1, 1, 1],
            block: [threads, 1, 1],
            shared_bytes: 0,
        };
        let mut batch = DirectBatch::new(&device, crate::NativeStream::Compute).unwrap();
        batch.launch(&launch(0, 1)).unwrap();
        batch.launch(&launch(1, 1)).unwrap();
        let too_many = module.max_threads_per_block(0) + 1;
        assert!(batch.launch(&launch(0, too_many)).is_err());
        let submission = batch
            .abandon()
            .expect("the launches issued before the failure are fenced");
        submission.finish().unwrap();
        // SAFETY: the launches writing `out` have completed.
        let written = unsafe { [host.as_ptr().read_volatile(), host.as_ptr().add(1).read_volatile()] };
        assert_eq!(written, [1, 2]);
    }
}
