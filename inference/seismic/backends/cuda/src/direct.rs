//! Formation and execution of explicitly authored CUDA C++ native
//! implementations: NVRTC to CUBIN for the opened device, launched on the
//! device's stream. It does not construct compiler kernels or plans.

use crate::buffer::Buffer;
use crate::driver::{self, DriverError, Event, Handle, JitError, Module};
use crate::executor::Device;
use crate::nvrtc::NvrtcError;
use seismic_compiler::direct_launch::{DirectLaunchRequest, LaunchGrid, ValidatedLaunch};
use seismic_compiler::errors::ExecutionError;
use seismic_compiler::graph_cost::GraphCensus;
use seismic_native_target::{NativeCompilationError, ProgramStatistic, StatisticValue};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

// cuFuncGetAttribute / cuFuncSetAttribute keys.
const MAX_THREADS_PER_BLOCK: i32 = 0;
const SHARED_SIZE_BYTES: i32 = 1;
const LOCAL_SIZE_BYTES: i32 = 3;
const NUM_REGS: i32 = 4;
const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;

/// The kernel functions of one CUDA program, loaded on one device.
pub struct DirectModule {
    /// Keeps the functions' module loaded.
    module: Arc<Module>,
    functions: Vec<DirectFunction>,
    /// The bytes of the CUBIN the module was loaded from.
    image_bytes: u64,
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
    /// What the driver reports the function uses: its registers, local
    /// (spill and stack) and static shared bytes, and the block size those
    /// admit.
    statistics: Vec<ProgramStatistic>,
    /// Largest dynamic shared size currently admitted by the function
    /// attribute; raised on demand.
    dynamic_shared_limit: AtomicU64,
    /// Serializes slow-path raises of `dynamic_shared_limit` so concurrent
    /// callers never downgrade the driver attribute.
    dynamic_shared_raise: Mutex<()>,
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
    NativeCompilationError::ToolchainFailure(jit_report(error))
}

/// What a JIT failure reports: the driver's error and its log.
pub(crate) fn jit_report(error: JitError) -> String {
    match error {
        JitError::Toolchain { error, log } => format!("{error}\n{log}"),
        JitError::Driver(error) => error.to_string(),
        JitError::MalformedImage(message) => message,
    }
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
        if kernels.is_empty() {
            return Err(NativeCompilationError::MalformedToolchainOutput(
                "native module has no kernels".into(),
            ));
        }
        let (module, raw_functions) = driver::load_module(context, image, kernels).map_err(jit_error)?;
        let mut functions = Vec::with_capacity(raw_functions.len());
        for raw in raw_functions {
            let attribute = |key| {
                driver::function_attribute(context, raw, key)
                    .map_err(driver_failure)
                    .map(|value| {
                        u64::try_from(value).expect("the driver reports a non-negative function attribute")
                    })
            };
            let max_threads = attribute(MAX_THREADS_PER_BLOCK)?;
            let static_shared = attribute(SHARED_SIZE_BYTES)?;
            let statistic = |name: &str, description: &str, value| ProgramStatistic {
                name: name.into(),
                description: description.into(),
                value: StatisticValue::Uint(value),
            };
            let statistics = vec![
                statistic("registers", "CU_FUNC_ATTRIBUTE_NUM_REGS: registers of one thread", attribute(NUM_REGS)?),
                statistic(
                    "local bytes",
                    "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: local memory of one thread (spills and stack)",
                    attribute(LOCAL_SIZE_BYTES)?,
                ),
                statistic("static shared bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: static shared memory of one block", static_shared),
                statistic(
                    "max threads per block",
                    "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: the most threads a block may have, given its registers",
                    max_threads,
                ),
            ];
            functions.push(DirectFunction {
                raw,
                max_threads,
                static_shared,
                statistics,
                dynamic_shared_limit: AtomicU64::new(attribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)?),
                dynamic_shared_raise: Mutex::new(()),
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
            module: Arc::new(module),
            functions,
            image_bytes: u64::try_from(image.len()).expect("a CUBIN's size fits u64"),
            edge,
        })
    }

    /// The context the module is loaded in.
    pub(crate) fn context(&self) -> &Arc<driver::Context> {
        &self.module.context
    }

    /// The driver's handle of kernel `function`.
    pub(crate) fn function(&self, function: usize) -> Handle {
        self.functions[function].raw
    }

    /// The bytes of the CUBIN the module was loaded from.
    pub fn image_bytes(&self) -> u64 {
        self.image_bytes
    }

    pub fn max_threads_per_block(&self, function: usize) -> u64 {
        self.functions[function].max_threads
    }

    pub fn static_shared_bytes(&self, function: usize) -> u64 {
        self.functions[function].static_shared
    }

    /// What the driver reports `function` uses.
    pub fn statistics(&self, function: usize) -> &[ProgramStatistic] {
        &self.functions[function].statistics
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
        if bytes <= self.dynamic_shared_limit.load(Ordering::Acquire) {
            return Ok(());
        }
        let _guard = self
            .dynamic_shared_raise
            .lock()
            .expect("dynamic shared limit lock is never poisoned");
        if bytes <= self.dynamic_shared_limit.load(Ordering::Acquire) {
            return Ok(());
        }
        let value = i32::try_from(bytes).map_err(|_| {
            ExecutionError::SubmissionFailed("dynamic shared size exceeds driver ABI".into())
        })?;
        driver::set_function_attribute(device.context(), self.raw, MAX_DYNAMIC_SHARED_SIZE_BYTES, value)
            .map_err(submission)?;
        self.dynamic_shared_limit.fetch_max(bytes, Ordering::AcqRel);
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
    /// Buffer addresses in ABI order.
    buffers: Vec<u64>,
    /// Scalar-result address, or 0 for an entry without scalar results.
    scalar_results: u64,
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
        let grid = match enabled {
            true => launch.grid,
            false => launch.grid.map(|blocks| blocks.max(1)),
        };
        let block = match enabled {
            true => launch.block,
            false => launch.block.map(|threads| threads.max(1)),
        };
        // The prefix declares at least one word, so an entry without words
        // still passes one zeroed word.
        let words_bytes = if launch.words.is_empty() {
            8u64
        } else {
            launch.words.len() as u64
        };
        let param_bytes = (launch.buffers.len() as u64)
            .checked_add(1)
            .and_then(|pointers| pointers.checked_mul(8))
            .and_then(|pointer_bytes| pointer_bytes.checked_add(words_bytes))
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed("native launch parameter size overflowed".into())
            })?;
        let ValidatedLaunch::Active {
            grid: Some(grid),
            group: Some(block),
        } = device
            .launch_contract(function.max_threads)
            .validate(&DirectLaunchRequest {
                buffer_slots: 0,
                grid: LaunchGrid::Direct(grid),
                group: Some(block),
                static_shared_bytes: function.static_shared,
                dynamic_shared_bytes: launch.shared_bytes,
                param_bytes,
            })?
        else {
            return Err(ExecutionError::SubmissionFailed(
                "native launch has an empty grid or block".into(),
            ));
        };
        function.admit_shared(device, launch.shared_bytes)?;
        let buffers = launch
            .buffers
            .iter()
            .map(|(buffer, offset)| buffer.pointer() + offset)
            .collect();
        let scalar_results = launch
            .scalar_results
            .map_or(0, |(buffer, offset)| buffer.pointer() + offset);
        let words = if launch.words.is_empty() {
            vec![0; 8]
        } else {
            launch.words.to_vec()
        };
        let shared_bytes = u32::try_from(launch.shared_bytes).map_err(|_| {
            ExecutionError::SubmissionFailed("native launch geometry exceeds driver ABI".into())
        })?;
        Ok(Self {
            function: function.raw,
            grid,
            block,
            shared_bytes,
            buffers,
            scalar_results,
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
        let mut parameters: Vec<*mut c_void> = Vec::with_capacity(self.buffers.len() + 2);
        parameters.extend(
            self.buffers
                .iter_mut()
                .map(|pointer| (pointer as *mut u64).cast::<c_void>()),
        );
        parameters.push(self.words.as_mut_ptr().cast());
        parameters.push((&mut self.scalar_results as *mut u64).cast());
        parameters
    }
}

/// One kernel node of a [`DirectGraph`]: its launch as formed or bound, and
/// whether it runs.
struct GraphNode {
    launch: FormedLaunch,
    enabled: bool,
}

/// The marks of a [`DirectGraph`] that has at least one mark node: the event
/// they record at rest, and any sampled run's events currently bound to them.
struct GraphMarks {
    /// What its marks record at rest.
    rest: Arc<Event>,
    /// The events a sampled run set on its marks, in order; empty where they
    /// record `rest` ([`DirectGraph::sample`]).
    sampled: Vec<Arc<Event>>,
}

impl GraphMarks {
    fn new(rest: Arc<Event>) -> Self {
        Self {
            rest,
            sampled: Vec::new(),
        }
    }
}

/// Forms a [`DirectGraph`]: the launches of one submission, each after the
/// earlier ones it depends on.
pub struct DirectGraphBuilder {
    device: Device,
    graph: driver::Graph,
    /// Each kernel node as formed, in order of addition.
    nodes: Vec<GraphNode>,
    /// What its marks record until a sampled run sets its own
    /// ([`DirectGraph::sample`]), where it has any.
    rest: Option<Arc<Event>>,
    /// Each word-operation node as bound now, in order of addition.
    words: Vec<WordNode>,
    /// The launches added so far ([`DirectGraph::census`]).
    launches: GraphCensus,
}

/// A word-operation node of a [`DirectGraph`]: a host gate's wait, or a
/// write the word's host waiters wake behind.
struct WordNode {
    /// The device address of the word it binds, and the value.
    address: u64,
    value: u32,
    kind: WordKind,
}

enum WordKind {
    Wait,
    /// A write's: the event recorded as it lands, which the waiters' wake
    /// follows ([`DirectBatch::replay`]), and the wake of the word it binds
    /// ([`DirectGraph::bind_word`]).
    Write {
        written: Event,
        wake: Arc<Wake>,
    },
}

impl WordKind {
    fn operation(&self) -> driver::WordOperation {
        match self {
            Self::Wait => driver::WordOperation::Wait,
            Self::Write { .. } => driver::WordOperation::Write,
        }
    }
}

impl DirectGraphBuilder {
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Ok(Self {
            device: device.clone(),
            graph: driver::Graph::new(device.context()).map_err(submission)?,
            nodes: Vec::new(),
            rest: None,
            words: Vec::new(),
            launches: GraphCensus::default(),
        })
    }

    /// One launch more of the graph's memory.
    fn count_launch(&mut self) -> Result<(), ExecutionError> {
        self.launches = self.launches.plus(GraphCensus::launch(0)).ok_or_else(|| {
            ExecutionError::SubmissionFailed("a graph's launch census overflows".into())
        })?;
        Ok(())
    }

    /// Add a gate after the nodes added so far: the graph's later nodes run
    /// once `word` reaches `value` (compared as [`DirectBatch::wait_word`]
    /// does). A replay binds its word and value ([`DirectGraph::bind_word`]).
    /// It counts as a launch of the graph's memory.
    pub fn wait_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        self.push_word(word, value, WordKind::Wait)
    }

    fn driver_error(&self, required: u64, error: driver::DriverError) -> ExecutionError {
        graph_driver_error(required, || self.device.refusal_reading(), error)
    }

    /// Add a write after the nodes added so far: `word` becomes `value` once
    /// they complete, their writes visible first (as
    /// [`DirectBatch::write_word`]), followed by an event-record node of its
    /// own the word's waiters wake behind as each replay lands it
    /// ([`DirectBatch::replay`]). A replay binds its word and value
    /// ([`DirectGraph::bind_word`]). It counts as two launches of the
    /// graph's memory.
    pub fn write_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        let written = Event::new(self.device.context()).map_err(|error| self.driver_error(0, error))?;
        self.push_word(
            word,
            value,
            WordKind::Write {
                written,
                wake: word.wake.clone(),
            },
        )
    }

    fn push_word(&mut self, word: &HostWord, value: u32, kind: WordKind) -> Result<(), ExecutionError> {
        let address = word.device_address();
        self.graph
            .push_word(kind.operation(), address, value)
            .map_err(|error| self.driver_error(0, error))?;
        self.count_launch()?;
        if let WordKind::Write { written, .. } = &kind {
            self.graph.push_event(written).map_err(|error| self.driver_error(0, error))?;
            self.count_launch()?;
        }
        self.words.push(WordNode { address, value, kind });
        Ok(())
    }

    /// Add a mark after the launches added so far: an event-record node in
    /// the graph's chain, stamping its event as the launch before it
    /// completes ([`DirectGraph::sample`]). It counts as a launch of the
    /// graph's memory.
    pub fn mark(&mut self) -> Result<(), ExecutionError> {
        let rest = match &self.rest {
            Some(rest) => rest.clone(),
            None => Arc::new(Event::stamp(self.device.context()).map_err(|error| self.driver_error(0, error))?),
        };
        self.graph.push_mark(&rest).map_err(|error| self.driver_error(0, error))?;
        self.rest = Some(rest);
        self.count_launch()
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
        self.graph.push_kernel(&node, formed.edge, after).map_err(|error| self.driver_error(0, error))?;
        self.nodes.push(GraphNode { launch: formed, enabled });
        // Its memory follows no parameter byte
        // (`seismic_compiler::graph_cost::argument_bytes`).
        self.count_launch()
    }

    /// Instantiate the graph and upload it to the device
    /// (`GraphExec::upload`), so its first launch does no setup. The driver
    /// memory both take, which no allocation charges, is its census priced
    /// with every live graph's ([`crate::graph_cost::GraphPool`]), held until
    /// the graph is dropped. The driver has taken it once the graph exists,
    /// so it is charged to the device store's own entry as taken, never
    /// refused.
    pub fn instantiate(self) -> Result<DirectGraph, ExecutionError> {
        let census = GraphCensus::graph(self.launches).ok_or_else(|| {
            ExecutionError::SubmissionFailed("a graph's census overflows".into())
        })?;
        // The driver refusing memory for the graph is a capacity refusal of
        // what its census takes; any other failure the submission's.
        let required = crate::graph_cost::census_bytes(&self.device, census);
        let refused = |error: driver::DriverError| self.driver_error(required, error);
        // The driver's instantiation and the record of what it took are one
        // change.
        let change = self.device.graphs().entry().change();
        let exec = self.graph.instantiate().map_err(refused)?;
        for (index, node) in self.nodes.iter().enumerate() {
            if !node.enabled {
                exec.set_enabled(self.graph.node(index), false).map_err(refused)?;
            }
        }
        exec.upload(self.device.stream()).map_err(refused)?;
        // `cuGraphUpload` enqueues asynchronous work on the device stream;
        // finish it before returning so back-to-back graph recordings
        // (`cuda::record_run`) still observe an idle device.
        self.device.stream().synchronize().map_err(refused)?;
        let launched = Event::new(self.device.context()).map_err(refused)?;
        self.device.graphs().instantiated(&change, &self.device, census)?;
        Ok(DirectGraph {
            device: self.device.clone(),
            exec,
            launched,
            marks: self.rest.map(GraphMarks::new),
            graph: self.graph,
            nodes: self.nodes,
            words: self.words,
            census,
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
    /// The graph's mark nodes, where it has any.
    marks: Option<GraphMarks>,
    /// The graph `exec` was instantiated from, whose nodes name the
    /// launches a parameter update sets ([`Self::bind`]).
    graph: driver::Graph,
    /// Each kernel node as the executable binds it now.
    nodes: Vec<GraphNode>,
    /// Each word-operation node as the executable binds it now.
    words: Vec<WordNode>,
    /// What the graph is made of, as its driver memory follows it.
    census: GraphCensus,
    /// The device whose graph pool holds its driver memory.
    device: Device,
}

impl DirectGraph {
    /// Its launches.
    pub fn census(&self) -> GraphCensus {
        self.census
    }

    /// The graph's launches, enabled or not.
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The graph's word-operation nodes ([`DirectGraphBuilder::wait_word`],
    /// [`DirectGraphBuilder::write_word`]).
    pub fn words(&self) -> usize {
        self.words.len()
    }

    /// Bind word-operation node `index` (by order of addition) to `word` at
    /// `value` for every launch of the graph after this call: a write's
    /// waiters are then `word`'s. Launches already queued keep what they
    /// were queued with; its owner binds only an idle graph ([`Self::bind`]).
    pub fn bind_word(&mut self, index: usize, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        let node = &mut self.words[index];
        let address = word.device_address();
        if (node.address, node.value) != (address, value) {
            self.exec
                .set_word(self.graph.word(index), node.kind.operation(), address, value)
                .map_err(submission)?;
            (node.address, node.value) = (address, value);
        }
        if let WordKind::Write { wake, .. } = &mut node.kind {
            *wake = word.wake.clone();
        }
        Ok(())
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
        let Some(state) = &mut self.marks else {
            assert!(marks.is_empty(), "DirectGraph::sample: graph has no marks");
            return Ok(());
        };
        if marks.is_empty() && state.sampled.is_empty() {
            return Ok(());
        }
        for (node, (stamps, mark)) in marks.iter().enumerate() {
            let event = stamps.events[*mark].clone();
            self.exec.set_event(self.graph.mark(node), &event).map_err(submission)?;
            if node < state.sampled.len() {
                state.sampled[node] = event;
            } else {
                state.sampled.push(event);
            }
        }
        // Reset trailing nodes from the end so any partial failure keeps
        // every still-referenced event alive in `state.sampled`.
        while state.sampled.len() > marks.len() {
            let node = state.sampled.len() - 1;
            self.exec.set_event(self.graph.mark(node), &state.rest).map_err(submission)?;
            state.sampled.pop();
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
            let current = &mut self.nodes[node];
            assert!(
                formed.edge == current.launch.edge,
                "a graph's launch binds only parameters: its shape fixes its edges"
            );
            if formed != current.launch {
                let mut parameters = formed.parameters();
                let params = formed.node(&mut parameters);
                self.exec.set_kernel(self.graph.node(node), &params).map_err(submission)?;
                current.launch = formed;
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
        let current = &mut self.nodes[node];
        if enabled != current.enabled {
            self.exec.set_enabled(self.graph.node(node), enabled).map_err(submission)?;
            current.enabled = enabled;
        }
        Ok(())
    }
}

impl Drop for DirectGraph {
    fn drop(&mut self) {
        self.device.graphs().dropped(self.census);
    }
}

/// A 32-bit word in pinned host memory mapped into the device's address
/// space, which the host and the device's stream both write and wait on.
/// A host thread waiting for it sleeps ([`Self::wait`]): every write, the
/// host's, a stream's ([`DirectBatch::write_word`]) or a graph's
/// ([`DirectGraphBuilder::write_word`]), wakes it.
pub struct HostWord {
    mapped: driver::HostMapped,
    wake: Arc<Wake>,
}

/// What a [`HostWord`]'s waiters sleep on, which every write of the word
/// notifies. Its mutex is a leaf lock — never held across a waiter's
/// predicate — because [`wake_after_write`] runs on CUDA's `cuLaunchHostFunc`
/// thread, which driver teardown (`cuMemHostUnregister`, `cuMemFree`) waits
/// for while holding internal driver locks.
pub(crate) struct Wake {
    generation: Mutex<u64>,
    changed: Condvar,
}

impl Wake {
    fn new() -> Self {
        Self { generation: Mutex::new(0), changed: Condvar::new() }
    }

    fn notify(&self) {
        let mut generation = self.generation.lock().expect("host word wake lock");
        *generation = generation.wrapping_add(1);
        drop(generation);
        self.changed.notify_all();
    }

    fn wait(&self, reached: impl Fn() -> bool, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let seen = *self.generation.lock().expect("host word wake lock");
            if reached() {
                return true;
            }
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return false;
            };
            let guard = self.generation.lock().expect("host word wake lock");
            if *guard != seen {
                continue;
            }
            let _ = self.changed.wait_timeout(guard, left).expect("host word wake lock");
        }
    }

    /// Wake the waiters once `written`'s last record, on `writer`,
    /// completes: `writer`'s wake stream waits for it, then runs the wake,
    /// so no work of `writer` waits for the host and no other stream's work
    /// holds the wake ([`driver::Stream::wakes`]).
    fn behind(self: &Arc<Self>, writer: &driver::Stream, written: &Event) -> Result<(), ExecutionError> {
        let context = writer.context();
        let wakes_stream = writer.wakes().map_err(submission)?;
        let wakes = wakes_stream.raw();
        let _current = context.enter().map_err(submission)?;
        let issue = context.enqueue();
        // SAFETY: both handles are this context's; the event may be
        // destroyed once the wait is queued, and the wake's reference is the
        // host function's to release.
        unsafe {
            context.driver.check(issue.stream_wait_event(wakes, written.raw()), "wake wait").map_err(submission)?;
            let wake = Arc::into_raw(self.clone()).cast_mut().cast::<c_void>();
            let status = issue.launch_host_function(wakes, wake_after_write, wake);
            if status != 0 {
                drop(Arc::from_raw(wake.cast::<Wake>().cast_const()));
            }
            context.driver.check(status, "wake launch").map_err(submission)?;
        }
        wakes_stream.flush_submission().map_err(submission)
    }
}

/// The host function a stream runs behind a device write of a word: `data`
/// is the word's [`Wake`], an `Arc` leaked for it ([`Wake::behind`]).
unsafe extern "system" fn wake_after_write(data: *mut c_void) {
    // SAFETY: `Wake::behind` leaked one reference for this call.
    let wake = unsafe { Arc::from_raw(data.cast::<Wake>().cast_const()) };
    wake.notify();
}

impl HostWord {
    /// A word at 0, its page charged to the device's host store's own
    /// entry.
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        Self::on(device.context(), &device.stores().host.own())
    }

    /// A word at 0 in `context`, its page charged to `entry`, an entry of
    /// host RAM.
    pub(crate) fn on(
        context: &Arc<driver::Context>,
        entry: &Arc<dyn seismic_memory::entry::Entry>,
    ) -> Result<Self, ExecutionError> {
        let mapped = driver::HostMapped::new(context, entry, 4)?;
        mapped.write_at(0, &0u32.to_le_bytes());
        Ok(Self { mapped, wake: Arc::new(Wake::new()) })
    }

    /// Wake the word's waiters now: what they wait on besides the word
    /// changed (a fault of its writer).
    pub(crate) fn wake(&self) {
        self.wake.notify();
    }

    /// Wake the word's waiters once `stream`'s work queued so far completes,
    /// a device write of the word among it: an event recorded behind it,
    /// which `stream`'s wake stream waits for before it runs their wake, so
    /// no work of `stream` waits for the host. Every stream write
    /// of a host word is followed by this, and every graph write by its own
    /// event's wake ([`DirectBatch::replay`]), so every host wait sleeps.
    pub(crate) fn wake_after(&self, stream: &driver::Stream) -> Result<(), ExecutionError> {
        let written = Event::new(stream.context()).map_err(submission)?;
        written.record(stream).map_err(submission)?;
        self.wake.behind(stream, &written)
    }

    /// Wait at most `timeout` until `reached` holds of the word, sleeping
    /// between its writes; whether it did.
    pub fn wait(&self, reached: impl Fn(u32) -> bool, timeout: std::time::Duration) -> bool {
        self.wake.wait(|| reached(self.load()), timeout)
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
    pub fn device_address(&self) -> u64 {
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

/// Synchronizes its stream on drop when device work was enqueued on it and
/// not handed off to a [`DirectSubmission`].
struct BatchFence {
    stream: Option<Arc<driver::Stream>>,
}

impl BatchFence {
    fn idle() -> Self {
        Self { stream: None }
    }

    fn arm(&mut self, stream: &Arc<driver::Stream>) {
        if self.stream.is_none() {
            self.stream = Some(Arc::clone(stream));
        }
    }

    fn disarm(mut self) {
        self.stream = None;
    }
}

impl Drop for BatchFence {
    fn drop(&mut self) {
        if let Some(stream) = &self.stream {
            let _ = stream.synchronize();
        }
    }
}

/// Launches issued in order on one of the device's native streams. The
/// stream orders each launch after the previous one's writes.
pub struct DirectBatch {
    /// Declared first so an uncommitted drop synchronizes `stream` before
    /// `start`, `marks`, `stream`, or `device` drop.
    fence: BatchFence,
    device: Device,
    stream: Arc<driver::Stream>,
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
            fence: BatchFence::idle(),
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
        self.follow_event(&other.end)
    }

    /// The batch's following work waits until `event`, recorded on another
    /// stream of the same context (e.g. a `TransferQueue` copy stream), has
    /// completed (`cuStreamWaitEvent`): device work, waiting for no host.
    pub fn follow_event(&mut self, event: &Event) -> Result<(), ExecutionError> {
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        // SAFETY: the stream and the event are this context's, and the event
        // was recorded on its producing stream before this call.
        let status = unsafe { context.enqueue().stream_wait_event(self.stream.raw(), event.raw()) };
        context.driver.check(status, "cross-stream wait").map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
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
        let issue = context.enqueue();
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
        driver
            .check(status, "native kernel launch")
            .map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
    }

    /// Copy `bytes` bytes of the device's memory at `source` to
    /// `destination` in stream order (`cuMemcpyDtoDAsync`): after the
    /// stream's preceding work, before its following work, waiting for no
    /// host.
    pub fn copy(&mut self, destination: u64, source: u64, bytes: u64) -> Result<(), ExecutionError> {
        if bytes == 0 {
            return Ok(());
        }
        let count = usize::try_from(bytes)
            .map_err(|_| ExecutionError::SubmissionFailed("a device copy exceeds the host's address space".into()))?;
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        // SAFETY: both ranges lie in live device allocations of this
        // context, which the caller keeps until the batch completes.
        let status = unsafe { context.enqueue().memcpy_device_async(destination, source, count, self.stream.raw()) };
        context.driver.check(status, "device copy").map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
    }

    /// Copy `bytes` bytes of pinned host memory at `source` to device memory
    /// at `destination` in stream order (`cuMemcpyHtoDAsync`): after the
    /// stream's preceding work, before its following work, waiting for no
    /// host.
    ///
    /// # Safety
    /// `source` must point to at least `bytes` bytes of live pinned host
    /// memory mapped in this context, and `destination` must lie in a live
    /// device allocation of this context, both kept alive until the batch
    /// completes.
    pub unsafe fn upload(&mut self, destination: u64, source: *const u8, bytes: u64) -> Result<(), ExecutionError> {
        if bytes == 0 {
            return Ok(());
        }
        let count = usize::try_from(bytes)
            .map_err(|_| ExecutionError::SubmissionFailed("a host upload exceeds the host's address space".into()))?;
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        // SAFETY: caller upholds live pinned host `source` and device
        // `destination` for `count` bytes until the batch completes.
        let status = unsafe { context.enqueue().upload_async(destination, source.cast(), count, self.stream.raw()) };
        context.driver.check(status, "host-to-device copy").map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
    }

    /// Set the 8-byte-aligned word of the device's memory at `destination`
    /// to `value` in stream order, as two `cuMemsetD32Async` of its halves
    /// carrying the value in the call: no work of the stream runs between
    /// them, and work of another stream that reads the word orders after
    /// the batch, so no reader sees it half written.
    pub fn write_u64(&mut self, destination: u64, value: u64) -> Result<(), ExecutionError> {
        if destination % 8 != 0 {
            return Err(ExecutionError::SubmissionFailed("a device word is not 8-byte aligned".into()));
        }
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        for (offset, half) in [(0, value as u32), (4, (value >> 32) as u32)] {
            // SAFETY: a word of a live device allocation of this context.
            let status = unsafe { context.enqueue().memset_d32_async(destination + offset, half, 1, self.stream.raw()) };
            context.driver.check(status, "device word write").map_err(submission)?;
            self.fence.arm(&self.stream);
        }
        Ok(())
    }

    /// Stamp mark `mark` of `stamps` once the stream's preceding work
    /// completes ([`Stamps`]).
    pub fn stamp(&mut self, stamps: &Stamps, mark: usize) -> Result<(), ExecutionError> {
        stamps.events[mark].record(&self.stream).map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
    }

    /// The stream's following work waits until `word` reaches `value`
    /// (`cuStreamWaitValue32`, compared cyclically: `word - value` as I32
    /// at least 0): the host gate.
    pub fn wait_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        self.word_operation(word, value, driver::WordOperation::Wait.flags(), |issue, stream, address, value, flags| unsafe {
            issue.wait_value_32(stream, address, value, flags)
        })
    }

    /// `word` becomes `value` once the stream's preceding work completes,
    /// its writes visible first (`cuStreamWriteValue32` with its memory
    /// barrier); the word's waiters wake then ([`HostWord::wake_after`]).
    pub fn write_word(&mut self, word: &HostWord, value: u32) -> Result<(), ExecutionError> {
        self.word_operation(word, value, driver::WordOperation::Write.flags(), |issue, stream, address, value, flags| unsafe {
            issue.write_value_32(stream, address, value, flags)
        })?;
        word.wake_after(&self.stream)
    }

    fn word_operation(
        &mut self,
        word: &HostWord,
        value: u32,
        flags: u32,
        operation: impl FnOnce(&crate::enqueue::Enqueue, Handle, u64, u32, std::ffi::c_uint) -> Option<std::ffi::c_int>,
    ) -> Result<(), ExecutionError> {
        let context = self.device.context();
        let _current = context.enter().map_err(submission)?;
        let status = operation(context.enqueue(), self.stream.raw(), word.mapped.device, value, flags)
            .ok_or_else(|| {
                ExecutionError::SubmissionFailed(
                    "the CUDA driver predates driver API 11.7 stream memory operations".into(),
                )
            })?;
        context
            .driver
            .check(status, "stream memory operation")
            .map_err(submission)?;
        self.fence.arm(&self.stream);
        Ok(())
    }

    /// Queue every launch of `graph`, in its order, and behind each of its
    /// writes of a word the wake of that word's waiters, as the write lands
    /// ([`DirectGraphBuilder::write_word`]), not as the graph ends. A timed
    /// batch launches individually instead, so it never replays.
    pub fn replay(&mut self, graph: &DirectGraph) -> Result<(), ExecutionError> {
        assert!(
            self.marks.is_none(),
            "DirectBatch::replay precondition: a timed batch times each launch"
        );
        graph.exec.launch(&self.stream).map_err(submission)?;
        self.fence.arm(&self.stream);
        graph.launched.record(&self.stream).map_err(submission)?;
        for node in &graph.words {
            if let WordKind::Write { written, wake } = &node.kind {
                wake.behind(&self.stream, written)?;
            }
        }
        Ok(())
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
        let Self {
            fence,
            device,
            stream,
            start,
            marks,
        } = self;
        let end = Event::new(device.context()).and_then(|end| {
            end.record(&stream)?;
            stream.flush_submission()?;
            Ok(end)
        });
        match end {
            Ok(end) => {
                fence.disarm();
                Ok(DirectSubmission {
                    start,
                    end,
                    marks,
                    complete: std::sync::atomic::AtomicBool::new(false),
                })
            }
            Err(error) => {
                fence.disarm();
                Err(match stream.synchronize() {
                    Ok(()) => submission(error),
                    Err(waited) => ExecutionError::IssueAndCompletionFailed {
                        issue: Box::new(submission(error)),
                        completion: Box::new(submission(waited)),
                    },
                })
            }
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
        if !self.complete.load(std::sync::atomic::Ordering::Relaxed) {
            self.end.synchronize().map_err(submission)?;
            self.complete.store(true, std::sync::atomic::Ordering::Relaxed);
        }
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

impl Drop for DirectSubmission {
    fn drop(&mut self) {
        if !self.complete.load(Ordering::Relaxed) {
            let _ = self.end.synchronize();
        }
    }
}

fn graph_driver_error(
    required: u64,
    reading: impl FnOnce() -> seismic_compiler::errors::RefusalReading,
    error: driver::DriverError,
) -> ExecutionError {
    if error.is_device_loss() {
        ExecutionError::DeviceLost(error.to_string())
    } else if error.code == crate::executor::OUT_OF_MEMORY {
        ExecutionError::DeviceRefused { required, reading: reading() }
    } else {
        submission(error)
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
        let capability = description.device_info().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin =
            crate::nvrtc::compile_cubin_named(SOURCE, "abandon.cu", architecture, &["mark"])
                .unwrap();
        let module =
            DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        let out = device.allocate_mapped(8, &crate::test_entry()).unwrap();
        let scalars = device.allocate_mapped(4, &crate::test_entry()).unwrap();
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

    /// `Wake::wait` never holds its lock across `reached`: a predicate that
    /// blocks on another lock held by a thread waiting for `Wake::notify`
    /// (the lock-order cycle between `HostWord::wait`, a driver call, and
    /// CUDA's `cuLaunchHostFunc` callback) must not deadlock.
    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "a test starts threads of its own"
    )]
    fn wake_notify_does_not_deadlock_with_a_locking_wait_predicate() {
        use std::sync::atomic::AtomicBool;
        use std::sync::mpsc;
        use std::time::Duration;

        let wake = Arc::new(Wake::new());
        let other = Arc::new(Mutex::new(()));
        let in_reached = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        let holder = other.lock().unwrap();
        let waiter = {
            let wake = Arc::clone(&wake);
            let other = Arc::clone(&other);
            let in_reached = Arc::clone(&in_reached);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                wake.wait(
                    || {
                        in_reached.store(true, Ordering::Release);
                        let _guard = other.lock().unwrap();
                        done.load(Ordering::Acquire)
                    },
                    Duration::from_secs(2),
                )
            })
        };

        while !in_reached.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        // Simulate CUDA's `cuLaunchHostFunc` callback running `Wake::notify`
        // while `holder` still holds `other` (waiting for the callback to
        // complete before releasing `other`).
        let (tx, rx) = mpsc::sync_channel(1);
        let notifier = {
            let wake = Arc::clone(&wake);
            std::thread::spawn(move || {
                wake.notify();
                let _ = tx.send(());
            })
        };
        rx.recv_timeout(Duration::from_secs(2))
            .expect("Wake::notify deadlocked behind Wake::wait's predicate");
        notifier.join().unwrap();

        done.store(true, Ordering::Release);
        drop(holder);
        assert!(waiter.join().unwrap(), "waiter should observe done == true");
    }

    #[test]
    fn graph_driver_error_classifies_oom_device_loss_and_other() {
        use seismic_compiler::errors::{ExecutionError, RefusalReading};

        let oom = super::graph_driver_error(
            128,
            || RefusalReading::Host,
            crate::driver::DriverError {
                operation: "graph kernel node",
                code: crate::executor::OUT_OF_MEMORY,
                description: "out of memory".into(),
            },
        );
        assert!(oom.capacity_refusal().is_some(), "{oom:?}");
        assert!(matches!(
            oom,
            ExecutionError::DeviceRefused { required: 128, reading: RefusalReading::Host }
        ));

        let lost = super::graph_driver_error(
            0,
            || RefusalReading::Host,
            crate::driver::DriverError {
                operation: "graph kernel node",
                code: 700,
                description: "illegal address".into(),
            },
        );
        assert!(matches!(lost, ExecutionError::DeviceLost(_)));

        let other = super::graph_driver_error(
            0,
            || RefusalReading::Host,
            crate::driver::DriverError {
                operation: "graph kernel node",
                code: 1,
                description: "invalid value".into(),
            },
        );
        assert!(matches!(other, ExecutionError::SubmissionFailed(_)));
    }
}
