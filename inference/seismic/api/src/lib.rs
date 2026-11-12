//! The public Seismic API (spec §14): devices, tensors, prepared kernels,
//! runtime-discovered functions, and the helpers generated bindings compose.
//! This crate re-exports and
//! composes; it contains no second implementation.
//!
//! Consumers import only this crate and `seismic-build`. Nothing here
//! exposes `PlanSpace`, `FrozenPlan`, `ExecutableVariant`, target domains,
//! solver decisions, raw backend buffers, binding indices, or manual arena
//! allocation (§14.6).
//!
//! W9-B owns device/tensor internals, W9-C owns invocation, W9-A owns the
//! generated-code contract. Dynamic callers use [`dynamic`] without implementing
//! the unsafe generated entry contract.

pub use seismic_compiler::errors::{
    CheckedBundleError, ExecutionError, InvocationError, PreparationError, TargetError,
};
pub use seismic_compiler::feedback::{
    EvaluationMethod, FeedbackOptions, FeedbackReport, InvocationRange, InvocationScope,
    PreparationOptions,
};
pub use seismic_lang::checked::{
    Encoded as LaunchEncoding, NativeComparison, NativeCondition, NativeImplementation,
    NativeLaunch, NativeNatExpr, NativeParameter, NativeScratch, NativeSpecialization,
    NativeSpecializationError, DeviceBound, OnDevice, RecordPort,
};
pub use seismic_lang::precision::{Limit, PrecisionPolicy, SpecialPolicy, Tolerance};
pub use seismic_native_cpu::repack::ScaledCodeRows;
/// Numerical comparison helpers used by validation frontends, and the one
/// selector of the backends a test runs on.
pub mod testing {
    pub use seismic_compiler::numerics::{compare_element, ElementComparison};
    use super::{Availability, BackendName, DeviceTopology, Executor};

    pub use seismic_lang::registry::requested_test_backend as requested_backend;
    pub use seismic_lang::registry::{
        test_setup_only as setup_only, TEST_SETUP_CHECKED as SETUP_CHECKED,
        TEST_SETUP_UNCHECKED as SETUP_UNCHECKED,
    };

    /// Open executor `id` of `devices` with `options` as a device of its
    /// own, sharing no ledger with another open: a test's, beside other
    /// tests in one process, bounds its ledgers alone.
    #[cfg(feature = "fixtures")]
    pub fn open_own_ledgers(devices: &super::Devices, id: super::ExecutorId, options: super::DeviceOptions) -> Result<super::Device, super::OpenError> {
        devices.inner.open_own_ledgers(id, options).map(|inner| super::Device { inner })
    }

    /// [`open_own_ledgers`] of the first executor of `backend`.
    #[cfg(feature = "fixtures")]
    pub fn open_backend_own_ledgers(devices: &super::Devices, backend: BackendName, options: super::DeviceOptions) -> Result<super::Device, super::OpenError> {
        open_own_ledgers(devices, devices.inner.executor_of(backend)?, options)
    }

    /// The executor a test runs `backend` on: the first available one of
    /// it in `topology`, where the runner requests `backend` or none. A
    /// test runs on each backend this names one for, and fails where it
    /// does not open.
    pub fn selected(topology: &DeviceTopology, backend: BackendName) -> Option<&Executor> {
        if requested_backend().is_some_and(|requested| requested != backend) {
            return None;
        }
        topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == backend && matches!(executor.availability(), Availability::Available))
    }
}
pub use seismic_runtime::artifacts::{ArtifactKey, ArtifactStore, DeviceOptions, MemoryStore};
pub use seismic_runtime::api::device::Bounds;
pub use seismic_runtime::probes::{serve_probe_if_asked, KeptProbe, ProbeFailure, ProbeRun, NAMESPACE as PROBE_NAMESPACE};
/// What a native declaration reads of the device it runs on, `device.NAME`,
/// and how each backend establishes it.
pub use seismic_lang::device;
pub use seismic_runtime::formation::FormedPrograms;
pub use seismic_runtime::{backing_times, large_page_backing, unmeasured_graph_cost, BackingTimes, LargePageBacking};
/// Offline formation coverage of a generated module's native
/// implementations, and offline formation of the kernels a program requests.
#[cfg(feature = "coverage")]
pub mod coverage {
    use crate::{generated::Module, KernelRequest, NativeSpecialization};
    pub use seismic_runtime::native::coverage::{
        form, Configuration, Coverage, CoverageError, Formation, FormationFailure, Forming,
        GroupSite, InProcess, ProgramEntry, ProgramSource, Progress, RequestFailure, Target,
        Toolchain, Untaken, Verdict,
    };

    /// Every resident weight representation a weight slot of `module`'s
    /// native implementations does not take while it takes another, on
    /// `toolchain`'s backend, by preprocessing alone.
    pub fn untaken(module: &Module, toolchain: &Toolchain) -> Result<Vec<Untaken>, CoverageError> {
        seismic_runtime::native::coverage::untaken(module.checked(), toolchain)
    }

    /// Form every native implementation of `module` with `toolchain`,
    /// through `forming`, until every preprocessor group of its sources has
    /// been formed under some configuration. `progress` counts the entries
    /// searched.
    pub fn cover(
        module: &Module,
        toolchain: &Toolchain,
        forming: &dyn Forming,
        progress: &Progress,
    ) -> Result<Coverage, CoverageError> {
        seismic_runtime::native::coverage::cover(module.checked(), toolchain, forming, progress)
    }

    /// Form every kernel request at its default specialization on
    /// `toolchain`'s base configuration, in parallel through `forming`;
    /// results in request order. `progress` counts the requests formed.
    pub fn form_requests(
        module: &Module,
        toolchain: &Toolchain,
        forming: &dyn Forming,
        requests: &[KernelRequest],
        progress: &Progress,
    ) -> Vec<Result<(), RequestFailure>> {
        let requests = requests
            .iter()
            .map(|request| {
                assert_eq!(
                    request.backend,
                    toolchain.backend(),
                    "a request forms on its own backend"
                );
                seismic_runtime::native::coverage::Request {
                    entry: module
                        .checked()
                        .entry_named(request.entry)
                        .expect("a request names an entry of its module"),
                    bindings: request.elements.iter().fold(
                        seismic_lang::entry::ElementBindings::new(),
                        |bindings, (name, element)| bindings.bind(name, element.id()),
                    ),
                    statics: request
                        .statics
                        .iter()
                        .fold(NativeSpecialization::new(), |statics, (name, value)| {
                            statics.with_static(name.clone(), *value)
                        }),
                }
            })
            .collect::<Vec<_>>();
        seismic_runtime::native::coverage::form_requests(
            module.checked(),
            toolchain,
            forming,
            &requests,
            progress,
        )
    }
}
/// Replay of the tuning search against recorded surveys (development).
pub use seismic_runtime::native::replay;
pub use seismic_runtime::native::search::{
    Cost, LaunchKey, ParameterValues, SearchSettings, SearchStop,
};
pub use seismic_runtime::native::trace::{
    host_seconds, GateOpening, SubmissionTrace, TraceDetail, TraceError, TracedLaunch,
    TracedLaunchGeometry, TracedQueueSubmission, TracedSubmission,
};
pub use seismic_runtime::native::tune::{
    admit, governing_ratio, reselect, return_freed_pages, BuildError, Configuration,
    ConfigurationRecord, Constraint, CoverageCheck, CoverageOutcome, DeclaredParameter, Difference,
    Exclusion, FormationRecord, Formed, GoverningStep, Grounded, Narrowed, NumericalObservation,
    NumericalRequirement, Outcome, Performed, PieceMemory, PointInputs, PointMeasurement,
    PointRecord, PointSpec, ProcessMemory, Reselection, SearchPlan, Spent, Stage,
    Standing, StepPower, Strategy, SubjectComparison, SubjectMeasurement, SurveyPlan, SurveyProgress, TuneError,
    TuningDisposition, TuningInitializer, TuningMethod, TuningResult, TuningTime, UnitPlan,
    Violation, Work, Workers, WorkloadRotation, WrittenRows, COMPARISON_VERSION, GOVERNING_STEP,
};
pub use seismic_runtime::native::{
    Footprint, GraphReplays, NativeAdmission, NativeArtifactIdentity, NativeQueue, Observed, Quote,
};

/// Result of checking one native entry's element bindings without device
/// formation. Acceptance proves only checked source semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeBindingCheck {
    AcceptedByCheckedEntry,
    Unsupported(String),
}

/// One checked tensor port or result before device or native implementation formation.
/// The byte count uses the registered canonical representation layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTensorMetadata {
    pub element: Element,
    pub extents: Vec<u64>,
    pub canonical_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTensorParameterCheck {
    Checked(NativeTensorMetadata),
    Unsupported(String),
}

/// Checked result leaves in schema order; `None` is a scalar leaf. Native
/// graph formation still has to reject scalar results and validate the node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTensorResultsCheck {
    Checked(Vec<Option<NativeTensorMetadata>>),
    Unsupported(String),
}

/// The CPU native ABI that generated bindings wrap. Kernel authors use the
/// generated `Context` of their entry instead.
pub mod native_cpu {
    pub use seismic_runtime::native::{
        CpuInvocation, CpuKernelFn, CpuLaunchVariants, CpuNativeKernels, CpuVariant,
    };
}

/// What recorded graphs are made of and what a device's hold for it.
pub use seismic_compiler::graph_cost::{GraphBytes, GraphCensus, GraphCost, LaunchGraphs, LaunchedGraph};
pub use seismic_lang::expr::{BigInt, BigUint};
pub use seismic_lang::ids::StableEntryId;
pub use seismic_lang::reference_math::conversion::exact_f64;
pub use seismic_lang::registry::{
    f16_bits, f16_to_f32, record_table_bytes, BackendName, GgufScales, GridScale, Layout, OperandRecipe,
};
pub use seismic_lang::types::DType;
/// The Seismic CPU library every CPU native kernel builds on.
pub use seismic_native_cpu as cpu;
pub use seismic_runtime::api::tensor::{
    PreparedRecommit, RecommitRefusal, RecommitRelease, RecordPlacement, Resizer,
};
pub use seismic_runtime::api::HostRegion;
pub use seismic_runtime::api::{
    CallError, OutputError, RegionError, SharedWordError, TensorError, ViewChargeError,
    WorkflowError,
};

/// The largest block a dedicated device's allocator takes from its driver
/// for an ordinary allocation (blocks grow from 64 MiB to it).
pub use seismic_compiler::blocks::BLOCK_CAP_BYTES as DEVICE_BLOCK_CAP_BYTES;
mod flight;
pub use flight::{FlightCompletion, FlightRing};
pub mod trace;
pub use trace::{
    Argument, ArgumentWord, ClassFact, Dim, DimensionExpr, GraphRecorder, NativeGraphTraces, Reals, Table, TableExpr,
    TraceOutput,
};

pub use seismic_compiler::blocks::Packing as DeviceBlockPacking;

/// A replay of `backend`'s device block allocator, the rule its blocks are
/// placed by (`DeviceBlockPacking`), or `None` for a backend whose memory is
/// the host's. CUDA blocks are whole granules of the driver's recommended
/// allocation granularity, 2 MiB on every device measured, with ranges 256
/// aligned; Vulkan blocks are exact, capped at 1 GiB (a device whose
/// allocation limit is lower caps them lower), with ranges 16 aligned.
pub fn device_block_packing(backend: BackendName) -> Option<DeviceBlockPacking> {
    match backend {
        BackendName::Cuda => Some(DeviceBlockPacking::new(DEVICE_BLOCK_CAP_BYTES, 2 << 20, 256)),
        BackendName::Vulkan => Some(DeviceBlockPacking::new(DEVICE_BLOCK_CAP_BYTES, 1, 16)),
        BackendName::Metal | BackendName::Cpu => None,
    }
}
mod slab;
pub use seismic_discovery::{
    device_node, host_available, host_room, pin, HostAvailable, place_process, volume_available, volume_identity, Core, DeviceNode,
    DiscoveryError, HostHeld, HostStatus, Inventory, LimitVisibility, MemoryNode, ObservationError,
    PciLocation, ProcessLimitKind, ProcessMemoryLimit, ProcessPlacement, Processors, ProcessorsReportedBy,
    Reclaim, RuntimeStatus, Sample, Status,
};
pub use seismic_lang::forms::MetalPacking;
/// Measuring work on a device (`seismic-measure`): the timing session, the
/// sustained clock, the statistics and the records of measured points. The
/// runtime's measurable work is [`KernelPoint`] and [`GraphPoint`].
pub use seismic_measure as measure;
pub use seismic_runtime::devices::{
    BackendHeld, DeviceLedgerView, Ledger, Ledgers, LimitError, OpenError, PathLedgerView,
    ResolveError, StoreLedgerView,
};
pub use seismic_runtime::native::{
    GraphPending, GraphPoint, GraphTiming, KernelPoint, NativeSpans, Point, PointPending,
};
pub use seismic_topology::{
    AccessPath, AccessPathId, AllocationLimit, Availability, DeviceAllocationBound, Contract, DeviceMemory, DeviceTopology,
    Diagnostic, DiagnosticSource, Executor, ExecutorId, ExecutorKind, ExecutorSelector, HostReads, Identity,
    Kind, KindId, KindMemory, Limits, NativeKind, Obtained, Packing, PhysicalDevice,
    PhysicalDeviceId, PhysicalIdentity, Reach, Reported, ReportedBy, Reservation,
    ReservationCharges, ReservationChunks, ReservedContract, ReservedRecords, Revision, Placing, Store, StoreId,
    StoreIdentity, StoreKind, TableCharge, TransferPath, TransferPathId, Unreservable, VolumeId,
};
mod measurer;
pub use measurer::{KernelIdentity, Measurer, PointKey};
/// The recorded machines (`fixtures::recorded`), for other crates' tests.
#[cfg(feature = "fixtures")]
pub use seismic_topology::fixtures;
pub use slab::{Slab, SlabLayout, SlabRegion, SlabTensor};

use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::ids::RepresentationId;
use seismic_runtime::api::tensor::Ordinary;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// Execution resources over the machine's inventory: opening executors by
/// identity, and the status of what this process holds.
pub struct Devices {
    inner: seismic_runtime::devices::Devices,
}

impl Devices {
    /// Devices over `inventory`, which owns discovery and its revisions.
    pub fn new(inventory: Arc<Inventory>) -> Self {
        Self {
            inner: seismic_runtime::devices::Devices::new(inventory),
        }
    }

    /// Devices over a newly discovered inventory serving `paths`.
    pub fn discover(paths: &[std::path::PathBuf]) -> Result<Self, DiscoveryError> {
        seismic_runtime::devices::Devices::discover(paths).map(|inner| Self { inner })
    }

    /// The inventory the devices are opened over.
    pub fn inventory(&self) -> &Arc<Inventory> {
        self.inner.inventory()
    }

    /// The current immutable topology revision.
    pub fn topology(&self) -> Arc<DeviceTopology> {
        self.inner.topology()
    }

    /// Resolves a same-machine selector, rejecting a missing executor.
    /// Never substitutes another executor.
    pub fn resolve(&self, selector: ExecutorSelector) -> Result<ExecutorId, ResolveError> {
        self.inner.resolve(selector)
    }

    /// Open executor `id` of the current revision without an artifact
    /// store. An executor already open is shared.
    pub fn open(&self, id: ExecutorId) -> Result<Device, OpenError> {
        self.inner.open(id).map(|inner| Device { inner })
    }

    /// Open with `options` (for example an artifact store). An executor
    /// already open is shared; asking it for a different store is an error.
    pub fn open_with(&self, id: ExecutorId, options: DeviceOptions) -> Result<Device, OpenError> {
        self.inner
            .open_with(id, options)
            .map(|inner| Device { inner })
    }

    /// The measurement job's running of executor `id`'s probes, each in a
    /// probe process of the running executable, which serves it first thing
    /// in `main` ([`serve_probe_if_asked`]), keeping each result in `store`,
    /// before `deadline`; opening the executor with `store` afterwards reads
    /// them. A load never calls this.
    pub fn probe(
        &self,
        id: ExecutorId,
        store: &dyn ArtifactStore,
        deadline: std::time::Instant,
    ) -> Result<ProbeRun, OpenError> {
        self.inner.probe(id, store, deadline)
    }

    /// Low-level control: opens the first discovered executor of a backend.
    /// Managed callers select by requirements and open by identity.
    pub fn open_backend(&self, backend: BackendName) -> Result<Device, OpenError> {
        self.inner
            .open_backend(backend)
            .map(|inner| Device { inner })
    }

    /// [`Devices::open_backend`] with `options`, as [`Devices::open_with`].
    pub fn open_backend_with(&self, backend: BackendName, options: DeviceOptions) -> Result<Device, OpenError> {
        self.inner
            .open_backend_with(backend, options)
            .map(|inner| Device { inner })
    }

    /// The status of `executor` of `topology`: a sample of the stores and
    /// of the executor's grants, with what this process's opened executors
    /// hold.
    pub fn status(&self, topology: &DeviceTopology, executor: ExecutorId) -> Status {
        self.inner.status(topology, executor)
    }

    /// The status of opened `device`, with what the devices of its ledger
    /// scope hold: as [`Self::status`] reads it for every open of the
    /// product.
    pub fn device_status(&self, device: &Device) -> Status {
        self.inner.device_status(&device.inner)
    }

    /// What executor `id`'s recorded graphs hold for their census
    /// ([`Device::graph_cost`]): its backend measures it on opening, once
    /// per device in a process, so this opens the executor, sharing it
    /// where it is open. A load reads it from records and never calls this.
    pub fn graph_cost(&self, id: ExecutorId) -> Result<GraphCost, OpenError> {
        self.open(id).map(|device| device.graph_cost())
    }
}

impl fmt::Debug for Devices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Devices")
            .field("topology", &self.topology())
            .finish()
    }
}

/// One open device. Cloning shares the device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<seismic_runtime::api::device::DeviceInner>,
}

impl Device {
    /// Declare the load's charge final: a record table formed afterwards
    /// outside a [`TableSealLifted`] scope is an invariant violation (stops
    /// every build), being device memory no claim covers.
    pub fn seal_record_tables(&self) {
        self.inner.seal_record_tables()
    }

    /// Lift the seal while the guard lives, for allocations made under a
    /// claim after the load (a component's import and binding).
    pub fn lift_table_seal(&self) -> TableSealLifted {
        self.inner.lift_table_seal();
        TableSealLifted(self.clone(), std::marker::PhantomData)
    }

    pub fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    pub fn info(&self) -> &Executor {
        self.inner.info()
    }
    /// The device's properties (`seismic::device`), its probes as the
    /// measurement job stored them when it opened.
    pub fn properties(&self) -> &device::DeviceProperties {
        self.inner.properties()
    }
    #[cfg(feature = "machine-line")]
    /// Whether a Vulkan device forms the cooperative-matrix path, which a
    /// matrix program is empty without; `None` on every other backend.
    pub fn forms_matrix_path(&self) -> Option<bool> {
        self.inner.forms_matrix_path()
    }
    /// What this device's graphs cost its submissions so far: graphs
    /// recorded (a submission during which that count grew recorded one on
    /// the host, so its duration is not its launches' alone) and replays
    /// that updated a graph's parameters first. Serving records none in
    /// steady state: every shape's graph is recorded at its first use and
    /// shared by every place after.
    pub fn graph_replays(&self) -> GraphReplays {
        self.inner.graph_replays()
    }
    /// What the graphs the device keeps may be made of: what its graph
    /// families' and plans' live slots hold
    /// ([`NativeGraphFamily::graphs`]).
    pub fn graph_budget(&self) -> GraphCensus {
        self.inner.graph_budget()
    }
    /// What the device's recorded graphs hold for their census, measured
    /// when it opened.
    pub fn graph_cost(&self) -> GraphCost {
        self.inner.graph_cost()
    }
    /// Complete capability summary acquired from this opened device.
    pub fn capabilities(&self) -> &[String] {
        self.inner.capabilities()
    }
    pub fn backend(&self) -> BackendName {
        self.inner.info().backend()
    }
    /// Whether this device's toolchain keeps what it compiles in the
    /// platform's own cache (Metal's shader cache), which the process's
    /// normal exit leaves written.
    pub fn platform_cached(&self) -> bool {
        self.inner.platform_cached()
    }
    /// The host threads this device's own execution keeps busy while its
    /// work runs: a CPU device's worker pool's participants, the submitting
    /// thread among them; an accelerator's submitting thread alone.
    pub fn host_threads(&self) -> usize {
        self.inner.host_threads()
    }
    /// The host memory node the device is attached to.
    pub fn host_node(&self) -> Result<DeviceNode, String> {
        self.inner.host_node()
    }
    /// Stable key of this device's model and configuration for native
    /// tuning records.
    pub fn tuning_identity(&self) -> String {
        self.inner.tuning_identity()
    }
    /// Lend the device `workers` for the programs it forms off the calling
    /// thread, or take them back (none): whoever owns a span of formation
    /// on the device (a load's tuner) owns its threads. Without workers the
    /// device forms in turn on the thread that asks.
    pub fn staff(&self, workers: Option<Arc<Workers>>) {
        self.inner.staff(workers)
    }
    /// Whether this opened device forms Metal tensor operations (false on
    /// every other backend).
    pub fn forms_tensor_operations(&self) -> bool {
        self.inner.forms_tensor_operations()
    }
    /// Record every native submission on this device until the returned
    /// trace is dropped (measurement only; see `TraceDetail`).
    pub fn trace_submissions(&self, detail: TraceDetail) -> Result<SubmissionTrace, TraceError> {
        SubmissionTrace::start(&self.inner, detail)
    }
    /// Begin a checked direct-native graph on this opened device.
    pub fn native_graph(&self) -> NativeGraph {
        NativeGraph {
            inner: seismic_runtime::native::graph::NativeGraphDraft::new(&self.inner),
        }
    }
    /// The device's shared graph workspace of `bytes`: every family slot
    /// created in it binds the one allocation.
    pub fn execution_arena(&self, bytes: u64) -> Result<NativeExecutionArena, TensorError> {
        seismic_runtime::native::graph::NativeExecutionArena::new(&self.inner, bytes)
            .map(|inner| NativeExecutionArena { inner })
    }
    /// Begin an exact graph using a certified Seismic placement.
    pub fn native_graph_with_layout(&self, layout: &NativeGraphLayout) -> NativeGraph {
        NativeGraph {
            inner: seismic_runtime::native::graph::NativeGraphDraft::with_layout(
                &self.inner,
                layout.inner.clone(),
            ),
        }
    }
    /// An empty sequence of graph runs to submit together on `queue`.
    pub fn native_sequence(&self, queue: NativeQueue) -> NativeGraphSequence {
        NativeGraphSequence {
            inner: seismic_runtime::native::graph::NativeGraphSequence::new(&self.inner, queue),
        }
    }
    /// The native programs this device has formed (compiled, or loaded from
    /// its artifact store) and the seconds forming took: for checks that
    /// preparation forms nothing, and a load's account of its time.
    pub fn programs_formed(&self) -> FormedPrograms {
        self.inner.programs_formed()
    }
    /// The topology revision this device was opened in.
    pub fn topology(&self) -> &Arc<DeviceTopology> {
        self.inner.topology()
    }
    /// Whether this device's allocations count in the process's footprint
    /// (CPU, Metal) rather than being held by a driver beside it.
    pub fn allocations_in_process_footprint(&self) -> bool {
        self.inner.allocations_in_process_footprint()
    }
    /// The ledgers this device reaches: each store's (every executor of
    /// the process), each of its paths' accounts (every open of its
    /// executor), and its own on each store.
    pub fn ledgers(&self) -> Ledgers {
        self.inner.ledgers()
    }
    /// The store device storage is placed on first, whose own ledger is the
    /// device's charge: on a device whose memory is the host's, every charge
    /// it makes in that memory.
    pub fn device_store(&self) -> StoreId {
        self.inner.device_store()
    }
    /// The device's own ledger on its device store ([`Self::device_store`]).
    pub fn device_ledger(&self) -> DeviceLedgerView {
        self.inner.device_ledger()
    }
    /// The right to bound every ledger this device reaches
    /// ([`Bounds::bound`]): one live holder per ledger. Taking it while
    /// another holder lives panics, naming both.
    #[track_caller]
    pub fn bounds(&self) -> Bounds {
        self.inner.bounds()
    }
    /// Start the high-water mark of each of the device's own ledgers again
    /// from what it charges now: the next reading is the most charged since
    /// this call.
    pub fn restart_high_water(&self) {
        self.inner.restart_high_water()
    }
    /// What fixed buffers of `contract` can still take on this device: an
    /// allocation needing more is refused as a capacity shortfall.
    pub fn room(&self, contract: Contract) -> u64 {
        self.inner.room(contract)
    }
    /// What the backend holds for fixed buffers of `kind` beyond their
    /// allocations.
    pub fn backend_held(&self, kind: Contract) -> BackendHeld {
        self.inner.backend_held(kind)
    }
    /// A planned workflow on this device; refused on a native-only backend.
    pub fn workflow(&self) -> Result<WorkflowDraft, WorkflowError> {
        seismic_runtime::api::kernel::workflow(&self.inner).map(|inner| WorkflowDraft { inner })
    }
    pub(crate) fn inner(&self) -> &Arc<seismic_runtime::api::device::DeviceInner> {
        &self.inner
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("info", self.info()).finish()
    }
}

// ---------------------------------------------------------------------------
// Elements and tensors
// ---------------------------------------------------------------------------

/// An element representation by registry identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Element(RepresentationId);

/// An element is kept by its unique storage name ([`Element::name`]), the
/// registry's stable identity of it; a name the registry no longer holds
/// reads as no element.
impl serde::Serialize for Element {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for Element {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Element::named(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("no element is named {name}")))
    }
}

/// The integer codes of a packed representation ([`Element::integer_codes`]):
/// every code is a value in `min ..= max`, and `group` consecutive values
/// along the packing axis share one scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerCodes {
    pub min: i32,
    pub max: i32,
    pub group: u64,
}

impl Element {
    /// The storage registered under a unique storage name (`q4k`,
    /// `q4k@rows16`, `f32`, `gguf_q4_k`).
    pub fn named(name: &str) -> Option<Element> {
        seismic_lang::registry::representation(name).map(Element)
    }
    /// The element of storage `storage` (`q4k`, `nvfp4g16a4` for NVFP4's W4A4
    /// element) in `layout`.
    pub fn stored(storage: &str, layout: Layout) -> Option<Element> {
        seismic_lang::registry::storage(storage, layout).map(Element)
    }
    /// The external storage rows of `width` stored values of `encoding`,
    /// scaled as `scales` says, are read as, when the registry converts
    /// from it.
    pub fn gguf_source(
        encoding: magnitude_ggml::Encoding,
        scales: GgufScales,
        width: u64,
    ) -> Option<Element> {
        seismic_lang::registry::gguf_source(encoding, scales, width).map(Element)
    }
    /// The registered source of `encoding` scaled as `scales` says,
    /// whatever its rows' width.
    pub fn gguf_packet_source(
        encoding: magnitude_ggml::Encoding,
        scales: GgufScales,
    ) -> Option<Element> {
        seismic_lang::registry::gguf_packet_source(encoding, scales).map(Element)
    }
    /// Every external (artifact-format) storage the registry converts from.
    pub fn external_sources() -> impl Iterator<Item = Element> {
        seismic_lang::registry::representations()
            .iter()
            .filter(|info| {
                matches!(
                    info.kind,
                    seismic_lang::registry::RepresentationKind::External(_)
                )
            })
            .map(|info| Element(info.id))
    }
    /// The storage this external source's registered conversion writes in
    /// `layout` for products taking `operands`; `None` when it has none.
    pub fn converted(&self, layout: Layout, operands: OperandRecipe) -> Option<Element> {
        seismic_lang::registry::conversion_destination(self.0, layout, operands).map(Element)
    }
    /// How products over this element take their activation operands
    /// (`operands(W)`).
    pub fn operand_recipe(&self) -> OperandRecipe {
        seismic_lang::registry::representation_info(self.0).operands
    }
    /// Unique storage name.
    pub fn name(&self) -> &'static str {
        seismic_lang::registry::representation_info(self.0).name
    }
    /// This element as a weight operand of the CPU's components on `tier`,
    /// read where it lies, a device's row layout too
    /// (`cpu::WeightOperand::of`); `None` when the CPU reads no rows of it.
    pub fn cpu_operand(&self, tier: cpu::Tier) -> Option<cpu::WeightOperand> {
        cpu::WeightOperand::of(tier, self.0)
    }
    /// The logical representation this storage encodes.
    pub fn representation(&self) -> &'static str {
        seismic_lang::registry::representation_info(self.0).representation
    }
    pub fn layout(&self) -> Layout {
        seismic_lang::registry::representation_info(self.0).layout
    }
    /// Logical values per packet along the packing axis of packed or external
    /// storage; `None` for dense storage.
    pub fn logical_group(&self) -> Option<u64> {
        match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(_) => None,
            seismic_lang::registry::RepresentationKind::Packed(layout) => {
                Some(u64::from(layout.group))
            }
            seismic_lang::registry::RepresentationKind::PackedRows(layout) => {
                Some(u64::from(layout.group()))
            }
            seismic_lang::registry::RepresentationKind::External(layout) => {
                Some(u64::from(layout.logical_group))
            }
        }
    }
    /// The integer codes of packed storage, as the registry decodes them:
    /// the values a code takes and the columns that share one scale. `None`
    /// for dense storage, for floating codes and for grid codes (vector
    /// entries indexing a value grid, whose indices are not magnitudes).
    pub fn integer_codes(&self) -> Option<IntegerCodes> {
        // A pure function of the registry's recipe: derived once per
        // representation rather than re-interpreting its codes per query.
        type Codes = std::collections::HashMap<RepresentationId, Option<IntegerCodes>>;
        static CODES: std::sync::OnceLock<std::sync::RwLock<Codes>> = std::sync::OnceLock::new();
        let codes = CODES.get_or_init(Default::default);
        if let Some(known) = codes.read().unwrap().get(&self.0) {
            return *known;
        }
        let derived = self.derive_integer_codes();
        *codes.write().unwrap().entry(self.0).or_insert(derived)
    }

    /// Whether CUDA's GEMM folds this element's exact codes, scaling each
    /// group's F32 partial (`_FORM_GEMM_FOLD`), rather than rounding its
    /// values to the operand type.
    pub fn cuda_gemm_fold(&self) -> bool {
        seismic_lang::forms::cuda_gemm_fold_of(self.0)
    }

    /// This element's admission to Metal's token-packing GEMM, as Metal's
    /// native libraries read it (`_FORM_PACK`); `None` where the form does
    /// not serve it.
    pub fn metal_packing(&self) -> Option<MetalPacking> {
        seismic_lang::forms::metal_packing_of(self.0)
    }

    fn derive_integer_codes(&self) -> Option<IntegerCodes> {
        use seismic_lang::reference_math::ReferenceScalar;
        use seismic_lang::registry::{DecodeComponent, DecodeStep, QuantizedComponent};
        let recipe = seismic_lang::registry::decode_recipe(self.0, DType::F32)?;
        let DecodeComponent::Temporary(code) = recipe.component(QuantizedComponent::Code)? else {
            return None;
        };
        let grid = recipe.steps().iter().any(
            |step| matches!(step, DecodeStep::InterpretGridCode { into, .. } if *into == code),
        );
        if grid {
            return None;
        }
        let constant = |role| match recipe.component(role) {
            Some(DecodeComponent::Constant(ReferenceScalar::I32(value))) => Some(value),
            _ => None,
        };
        Some(IntegerCodes {
            min: constant(QuantizedComponent::CodeMin)?,
            max: constant(QuantizedComponent::CodeMax)?,
            group: u64::try_from(constant(QuantizedComponent::GroupSize)?).ok()?,
        })
    }
    /// Rows of one row tile of row-layout storage (`rows8`: 8); 1 for every
    /// other storage. A matrix view of tiled storage selects whole tiles, or
    /// runs to the end of its tensor.
    pub fn tile_rows(&self) -> u64 {
        match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::PackedRows(layout) => layout.tile_rows(),
            _ => 1,
        }
    }
    /// The host's repack of a tensor of `shape` from `source` into this
    /// storage, as a device's repack binding converts it: between dense
    /// float elements by elements, each decoded to F32 and rounded into
    /// this element (`import_dense`'s conversion, which no registry entry
    /// holds); otherwise by the registered conversion
    /// (`cpu::repack::convert`). `None` when neither converts the pair or
    /// `bytes` is not the source's canonical byte count.
    pub fn repack_host(self, source: Element, shape: &[u64], bytes: &[u8]) -> Option<Vec<u8>> {
        match (source.dtype(), self.dtype()) {
            (Some(from), Some(to)) if from.is_float() && to.is_float() => {
                let width = to.bytes() as usize;
                let values = source.decode_host(shape, bytes)?;
                Some(
                    values
                        .into_iter()
                        .flat_map(|value| {
                            let decoded = f64::from(value as f32);
                            let rounded = seismic_lang::reference_math::float_literal(to, decoded);
                            rounded.bits().to_le_bytes().into_iter().take(width)
                        })
                        .collect(),
                )
            }
            _ => {
                let conversion =
                    seismic_lang::registry::representation_conversion(source.0, self.0)?;
                seismic_native_cpu::repack::convert(conversion, shape, bytes)
            }
        }
    }
    /// The registry's reference of [`Element::repack_host`]'s conversion
    /// (`seismic_lang::interp::repack`), for other crates' tests of the
    /// host's conversions. `None` as `repack_host`, or for a pair of dense
    /// float elements, which no registry entry converts.
    #[cfg(feature = "fixtures")]
    pub fn repack_reference(self, source: Element, shape: &[u64], bytes: &[u8]) -> Option<Vec<u8>> {
        let conversion = seismic_lang::registry::representation_conversion(source.0, self.0)?;
        let shape = shape
            .iter()
            .map(|extent| usize::try_from(*extent).ok())
            .collect::<Option<Vec<_>>>()?;
        seismic_lang::interp::repack(conversion.id, &shape, bytes)
    }
    /// `bytes`, arbitrary bytes of whole packets of `source`, brought into
    /// the domain of the registered conversion from `source` into this
    /// storage (`RepresentationConversion::admit_source`): what a check of a
    /// repack against [`Element::repack_host`] may convert. `None` when no
    /// conversion is registered or `bytes` is not whole source packets.
    pub fn repack_source(self, source: Element, bytes: Vec<u8>) -> Option<Vec<u8>> {
        self.sourced(
            source,
            bytes,
            seismic_lang::registry::RepresentationConversion::admit_source,
        )
    }
    /// `bytes`, arbitrary bytes of whole packets of `source`, brought into
    /// the domain of the registered conversion from `source` into this
    /// storage with every number this storage reads from them finite
    /// (`RepresentationConversion::finite_source`): a record whose
    /// arithmetic compares. `None` as [`Element::repack_source`].
    pub fn finite_source(self, source: Element, bytes: Vec<u8>) -> Option<Vec<u8>> {
        self.sourced(
            source,
            bytes,
            seismic_lang::registry::RepresentationConversion::finite_source,
        )
    }
    fn sourced(
        self,
        source: Element,
        mut bytes: Vec<u8>,
        admit: fn(&seismic_lang::registry::RepresentationConversion, &mut [u8]),
    ) -> Option<Vec<u8>> {
        let conversion = seismic_lang::registry::representation_conversion(source.0, self.0)?;
        if !bytes
            .len()
            .is_multiple_of(conversion.source_layout.packet_size as usize)
        {
            return None;
        }
        admit(conversion, &mut bytes);
        Some(bytes)
    }
    /// Host reference decode of canonical packed or dense bytes of a tensor
    /// of `shape` into logical values in row-major order. `None` for external
    /// storage or a wrong byte count.
    pub fn decode_host(self, shape: &[u64], bytes: &[u8]) -> Option<Vec<f64>> {
        let shape = shape
            .iter()
            .map(|extent| usize::try_from(*extent).ok())
            .collect::<Option<Vec<_>>>()?;
        let data = match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => {
                seismic_lang::interp::TensorData::dense_from_bytes(*dtype, shape, bytes.to_vec())
            }
            seismic_lang::registry::RepresentationKind::Packed(_)
            | seismic_lang::registry::RepresentationKind::PackedRows(_) => {
                seismic_lang::interp::TensorData::encoded(self.0, shape, bytes.to_vec())
            }
            seismic_lang::registry::RepresentationKind::External(_) => return None,
        }
        .ok()?;
        data.values().ok()
    }
    /// Rows of `columns` values stored as `source`, through the registered
    /// conversion of `source` into this storage, whose values are integer
    /// codes times one stored scale (`ScaledCodeRows::admit`), placed by
    /// [`ScaledCodeRows::placement`]. `None` when no conversion is
    /// registered, this storage's values are not such codes, or the rows
    /// are not whole packets of `source`.
    pub fn scaled_code_rows(
        self,
        source: Element,
        columns: std::num::NonZeroU32,
    ) -> Option<ScaledCodeRows> {
        let conversion = seismic_lang::registry::representation_conversion(source.0, self.0)?;
        ScaledCodeRows::admit(conversion, columns)
    }
    fn id(&self) -> RepresentationId {
        self.0
    }
    pub fn dense(dtype: DType) -> Element {
        Element(seismic_lang::registry::dense(dtype))
    }
    pub fn dtype(self) -> Option<DType> {
        match &seismic_lang::registry::representation_info(self.0).kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => Some(*dtype),
            seismic_lang::registry::RepresentationKind::Packed(_)
            | seismic_lang::registry::RepresentationKind::PackedRows(_)
            | seismic_lang::registry::RepresentationKind::External(_) => None,
        }
    }
    /// Exact bytes of the runtime's canonical tensor layout for these
    /// logical extents, including per-row packets and packet alignment.
    pub fn canonical_byte_len(self, extents: &[u64]) -> Result<u64, TensorError> {
        seismic_runtime::api::tensor::TensorInner::canonical_byte_len(self.id(), extents)
    }
    pub fn f32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::F32,
        ))
    }
    pub fn f16() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::F16,
        ))
    }
    pub fn bf16() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::BF16,
        ))
    }
    pub fn i32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::I32,
        ))
    }
    pub fn u32() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::U32,
        ))
    }
    pub fn u64() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::U64,
        ))
    }
    pub fn bool() -> Element {
        Element(seismic_lang::registry::dense(
            seismic_lang::types::DType::Bool,
        ))
    }
}

impl From<DType> for Element {
    fn from(value: DType) -> Self {
        Self::dense(value)
    }
}

/// IEEE-754 binary16 scalar value, preserved as bits at the Rust boundary.
/// Seismic performs any widening explicitly according to the authored
/// kernel and selected numerical policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F16(u16);

impl F16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

/// bfloat16 scalar value, preserved as bits at the Rust boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BF16(u16);

impl BF16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

pub use seismic_runtime::native::graph::NativeTimelinePort;
pub use seismic_runtime::native::timeline::{AnsweredBy, ParkAnswer};

/// Timeline roles and node kinds (`seismic_runtime::native::timeline`).
pub mod timeline {
    pub use seismic_runtime::native::timeline::{
        Gate, Park, Parks, Publish, Reached, Signal, TimelineNodeKind, TimelineRole, ValuedRole, Wait,
    };
}

/// A monotonic value between the host and one device's native work, of role
/// `R` (`timeline::Gate`, `Reached` or `Parks`), which says who raises it and
/// who waits on it; the handle has only the operations of its role, and a
/// graph's timeline node binds only a timeline of its own role
/// (`NativeGraphBindings::set_timeline`). A sealed graph's node binds it per
/// run: at a wait node (a `Gate`) the device runs nothing after it until the
/// host has raised the timeline to the run's value; at a signal node (a
/// `Reached`) the device raises it to the run's value once the work before
/// has completed, its writes visible to the host.
///
/// So the host never waits on what only it raises, nor the device on what
/// only it raises: on Metal, CUDA and Vulkan each side is its own event,
/// word or semaphore, and such a wait would never return.
///
/// Each side has its own operations and nodes:
///
/// ```
/// use seismic::timeline::{Gate, Reached, Signal, Wait};
/// fn each_side_its_own(
///     graph: &mut seismic::NativeGraph,
///     bindings: &mut seismic::NativeGraphBindings,
///     gate: &seismic::Timeline<Gate>,
///     read: &seismic::Timeline<Reached>,
/// ) {
///     let (opened, done) = (graph.timeline(Wait), graph.timeline(Signal));
///     bindings.set_timeline(opened, gate, 1).unwrap();
///     bindings.set_timeline(done, read, 1).unwrap();
///     gate.signal(1).unwrap();
///     read.wait(1, std::time::Duration::from_secs(1)).unwrap();
/// }
/// ```
///
/// and neither the other's:
///
/// ```compile_fail
/// fn host_waits_on_its_own_gate(gate: &seismic::Timeline<seismic::timeline::Gate>) {
///     gate.wait(1, std::time::Duration::from_secs(1));
/// }
/// ```
///
/// ```compile_fail
/// fn host_signals_the_devices_side(read: &seismic::Timeline<seismic::timeline::Reached>) {
///     read.signal(1);
/// }
/// ```
///
/// ```compile_fail
/// fn gate_bound_to_a_signal_node(
///     graph: &mut seismic::NativeGraph,
///     bindings: &mut seismic::NativeGraphBindings,
///     gate: &seismic::Timeline<seismic::timeline::Gate>,
/// ) {
///     let signal = graph.timeline(seismic::timeline::Signal);
///     bindings.set_timeline(signal, gate, 1);
/// }
/// ```
///
/// ```compile_fail
/// fn park_bound_at_a_value(
///     graph: &mut seismic::NativeGraph,
///     bindings: &mut seismic::NativeGraphBindings,
///     parks: &seismic::Timeline<seismic::timeline::Parks>,
/// ) {
///     let park = graph.timeline(seismic::timeline::Park);
///     bindings.set_timeline(park, parks, 1);
/// }
/// ```
///
/// Clones are handles of one timeline. Dropping the last opens every wait a
/// run binds on it, and each such run whose wait was never reached fails its
/// completion: a teardown leaves no device work waiting. So a handle lives
/// as long as runs wait on the timeline, whoever is to raise it.
pub struct Timeline<R: timeline::TimelineRole> {
    host: Arc<seismic_runtime::native::timeline::TimelineHost<R>>,
}

impl<R: timeline::TimelineRole> Clone for Timeline<R> {
    fn clone(&self) -> Self {
        Self { host: self.host.clone() }
    }
}

impl<R: timeline::TimelineRole> Timeline<R> {
    /// A timeline at 0 on `device`.
    pub fn new(device: &Device) -> Result<Self, TensorError> {
        seismic_runtime::native::timeline::TimelineHost::new(device.inner()).map(|host| Self { host: Arc::new(host) })
    }
}

impl Timeline<timeline::Gate> {
    /// Raise the gate to `value` from the host, after every host write
    /// before it; a value it has reached already changes nothing.
    pub fn signal(&self, value: u64) -> Result<(), ExecutionError> {
        self.host.signal(value)
    }

    /// Whether the host has raised the gate to `value`; the device's loss
    /// where it cannot say.
    pub fn signalled(&self, value: u64) -> Result<bool, ExecutionError> {
        self.host.signalled(value)
    }

    /// Wait at most `timeout` for the host (another thread) to raise the
    /// gate to `value`, after which this thread sees every host write before
    /// that signal; whether it did, or the device's loss where it cannot
    /// say.
    pub fn wait_signalled(&self, value: u64, timeout: std::time::Duration) -> Result<bool, ExecutionError> {
        self.host.wait_signalled(value, timeout)
    }
}

impl Timeline<timeline::Reached> {
    /// Whether the device has raised the timeline to `value`; the device's
    /// loss where it cannot say.
    pub fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.host.reached(value)
    }

    /// The value the device has raised the timeline to; the device's loss
    /// where it cannot say.
    pub fn current(&self) -> Result<u64, ExecutionError> {
        self.host.current()
    }

    /// Wait at most `timeout` for the device to raise the timeline to
    /// `value`, after which the host sees every device write before the
    /// signal; whether it did, or the device's loss where it cannot say.
    pub fn wait(&self, value: u64, timeout: std::time::Duration) -> Result<bool, ExecutionError> {
        self.host.wait(value, timeout)
    }

    /// Raise the timeline to `value` in the device's place: the host
    /// completes a signal no device work will raise (a step queued but
    /// never run), or plays the device (a test). Precondition: no device
    /// signal of the timeline is pending.
    pub fn reach(&self, value: u64) -> Result<(), ExecutionError> {
        self.host.reach(value)
    }
}

impl Timeline<timeline::Parks> {
    /// Whether the device reached the park or publication that took `value`
    /// (`NativeGraphOutputs::park`), every device write before it visible
    /// to the host; a park's call waits there unless the host answered it.
    /// The device's loss where it cannot say.
    pub fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.host.reached(value)
    }

    /// The value the device has raised the timeline to; the device's loss
    /// where it cannot say.
    pub fn current(&self) -> Result<u64, ExecutionError> {
        self.host.current()
    }

    /// Raise what the device raises to `value`, as the device would: a host
    /// that plays the device (a test).
    pub fn reach(&self, value: u64) -> Result<(), ExecutionError> {
        self.host.reach(value)
    }

    /// Whether the host answered the park that took `value`, whether or not
    /// the device reached it; never where the dropping of the last handle
    /// opened its gate unanswered. The device's loss where it cannot say.
    pub fn answered(&self, value: u64) -> Result<bool, ExecutionError> {
        self.host.answered(value)
    }

    /// Whether the device's kernels wait for a park's answer themselves, so
    /// a park binds no operation of this timeline.
    pub fn parks_in_kernels(&self) -> bool {
        self.host.parks_in_kernels()
    }

    /// Answer the park that took `value`, after every host write the call
    /// reads, before or after the device reaches it: where the device's
    /// side is its own (Vulkan) the device runs on through it, and an
    /// answer never reads as the device reaching anything.
    /// `answer` is why the host answers now, which a trace keeps
    /// (`TracedQueueSubmission::opened`).
    pub fn answer(&self, value: u64, answer: ParkAnswer) -> Result<(), ExecutionError> {
        self.host.answer(value, answer)
    }
}

pub use seismic_runtime::native::transfer::TransferError;

/// One device's transfer queue (`seismic_runtime::native::transfer`):
/// batches of copies between host memory and the device's storage, either
/// way, that run beside its compute, each batch complete once its value is
/// reached. Where the device's every storage is
/// host-addressed (Metal) there is none, and the host writes in place.
///
/// The caller orders each batch against device work: it copies only into
/// bytes no queued work reaches, and lets work read them only after seeing
/// the batch complete (a host gate it answers then). The sources' bytes
/// stay untouched until then.
#[derive(Clone)]
pub struct Transfer {
    inner: Arc<seismic_runtime::native::transfer::TransferInner>,
}

/// One copy of a [`Transfer`] batch: `bytes` bytes of `source` from byte
/// `source_offset` of its view into `destination` from byte
/// `destination_offset` of its view: host memory into the device's storage,
/// or back.
pub struct TransferRun<'a> {
    pub source: &'a Tensor,
    pub source_offset: u64,
    pub destination: &'a Tensor,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One step of a [`Transfer`] batch, in order: a copy, or a signal: the i32
/// word `index` of `word` (the device's storage, a dense i32 tensor) set to
/// `value` once every run of the batch before it has landed, which device
/// work may wait for by equality.
pub enum TransferItem<'a> {
    Run(TransferRun<'a>),
    Signal {
        word: &'a Tensor,
        index: u64,
        value: i32,
    },
}

impl Transfer {
    /// The transfer queue of `device`; why it has none.
    pub fn new(device: &Device) -> Result<Self, TransferError> {
        seismic_runtime::native::transfer::TransferInner::new(device.inner()).map(|inner| Self {
            inner: Arc::new(inner),
        })
    }

    /// Queue `items` as one batch: the value it completes at. The caller
    /// queues it only once the device work that read the bytes it
    /// overwrites has ended.
    pub fn copy(&self, items: &[TransferItem<'_>]) -> Result<u64, TransferError> {
        use seismic_runtime::native::transfer as native;
        let items = items
            .iter()
            .map(|item| match item {
                TransferItem::Run(run) => native::TransferItem::Run(native::TransferRun {
                    source: run.source.inner(),
                    source_offset: run.source_offset,
                    destination: run.destination.inner(),
                    destination_offset: run.destination_offset,
                    bytes: run.bytes,
                }),
                TransferItem::Signal { word, index, value } => native::TransferItem::Signal {
                    word: word.inner(),
                    offset: index * 4,
                    value: *value,
                },
            })
            .collect::<Vec<_>>();
        self.inner.copy(&items)
    }

    /// Whether the batch that took `value` is complete; why none can
    /// complete (a copy faulted, or the device is lost).
    pub fn completed(&self, value: u64) -> Result<bool, TransferError> {
        self.inner.completed(value)
    }

    /// Wait at most `timeout` for the batch that took `value`; whether it
    /// completed, or why none can.
    pub fn wait(&self, value: u64, timeout: std::time::Duration) -> Result<bool, TransferError> {
        self.inner.wait(value, timeout)
    }
}

/// A device tensor: an owned allocation or a view, with device identity,
/// representation, extents and strides (§14.3). Shapes come from here.
/// One read-only host window wrapped by the device. Every tensor view shares
/// one allocation and one charge; submitted work retains that allocation and
/// its host mapping until completion.
pub struct ReadOnlyMappedRegion {
    inner: seismic_runtime::api::tensor::MappedRegionInner,
}

impl ReadOnlyMappedRegion {
    pub fn new(device: &Device, region: HostRegion) -> Result<Self, TensorError> {
        seismic_runtime::api::tensor::MappedRegionInner::new(device.inner(), region)
            .map(|inner| Self { inner })
    }

    /// Whether a tensor of `element` and `extents` over a region's bytes
    /// from `byte_offset` keeps the element's alignment: a region starts at
    /// a host page, so its bytes are aligned as their offset is.
    pub fn keeps_alignment(
        element: Element,
        extents: &[u64],
        byte_offset: u64,
    ) -> Result<bool, TensorError> {
        seismic_runtime::api::tensor::MappedRegionInner::keeps_alignment(
            element.id(),
            extents,
            byte_offset,
        )
    }

    /// A tensor of `element` and `extents` over the region's bytes from
    /// `byte_offset`, which must keep the element's alignment
    /// ([`Self::keeps_alignment`]).
    pub fn tensor(
        &self,
        element: Element,
        extents: &[u64],
        byte_offset: u64,
    ) -> Result<Tensor, TensorError> {
        self.inner
            .tensor(element.id(), extents, byte_offset)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }
}

#[derive(Clone)]
pub struct Tensor {
    inner: Arc<seismic_runtime::api::tensor::TensorInner>,
}

/// A weak view of one tensor's physical storage. It never pins the
/// allocation; callers can classify storage still charged after ownership
/// moves without maintaining a second byte ledger.
pub struct TensorStorageObserver {
    inner: seismic_runtime::api::tensor::TensorStorageObserver,
}

impl TensorStorageObserver {
    /// The observed allocation's charge to each store; `None` once released.
    pub fn store_charges(&self) -> Option<Vec<(StoreId, u64)>> {
        self.inner.store_charges()
    }
    /// The observed allocation's charge along each access path; `None`
    /// once released.
    pub fn path_charges(&self) -> Option<Vec<(AccessPathId, u64)>> {
        self.inner.path_charges()
    }
    pub fn identity(&self) -> u64 {
        self.inner.identity()
    }

    pub fn charged_bytes(&self) -> Option<u64> {
        self.inner.charged_bytes()
    }
}

impl Tensor {
    pub(crate) fn slabbed(
        table: &Ordinary,
        element: Element,
        rows_per_slab: u64,
        logical_rows: u64,
        alias_identity: u64,
        regions: Vec<Option<Ordinary>>,
        lease: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::slabbed(
            table,
            element.id(),
            rows_per_slab,
            logical_rows,
            alias_identity,
            regions,
            lease,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }

    /// This tensor as an ordinary one, unless it views a reservation.
    pub(crate) fn ordinary(&self) -> Option<Ordinary> {
        Ordinary::of(self.inner.clone())
    }

    pub(crate) fn of_ordinary(tensor: &Ordinary) -> Tensor {
        Tensor {
            inner: tensor.tensor().clone(),
        }
    }

    pub(crate) fn new_slab_alias() -> u64 {
        seismic_runtime::api::tensor::TensorInner::new_slab_alias()
    }

    /// Canonical storage footprint and required alignment of a tensor shape.
    pub fn canonical_layout(element: Element, extents: &[u64]) -> Result<(u64, u64), TensorError> {
        seismic_runtime::api::tensor::TensorInner::canonical_layout(element.id(), extents)
    }

    /// A canonical-layout tensor of `element` and `extents` viewing this
    /// tensor's bytes from `byte_offset`: it shares the allocation and its
    /// device-use lifetime. This tensor must be one contiguous byte range of
    /// an allocation (not slabbed), and the region must lie within it,
    /// starting at a multiple of its layout's alignment in the allocation
    /// ([`TensorError::Region`] otherwise). A region of a reserved tensor
    /// lies within its backed bytes and pins them; on chunked storage
    /// (Metal) its records are packed, none crossing a chunk: it starts
    /// where its first record fits in one chunk (such as the end of an
    /// earlier packed region, `seismic_topology::Packing::end`),
    /// and its records are where [`Tensor::record_placement`] says.
    pub fn region(
        &self,
        element: Element,
        extents: &[u64],
        byte_offset: u64,
    ) -> Result<Tensor, TensorError> {
        self.inner
            .region(element.id(), extents, byte_offset)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }

    /// Records `start..end` of this tensor as a region of their own, which
    /// pins a reservation's backed bytes only through the last of them
    /// (`TensorInner::record_region`): what a holder of a few records keeps,
    /// so it never holds a recommit above them.
    pub fn record_region(&self, start: u64, end: u64) -> Result<Tensor, TensorError> {
        self.inner.record_region(start, end).map(|inner| Tensor { inner: Arc::new(inner) })
    }

    /// Where this canonical tensor's records (leading-axis rows) lie:
    /// packed from its first byte on chunked reserved storage
    /// (`seismic_topology::Packing`), otherwise one after another. None for
    /// a tensor without records of their own (no rows, rows of no bytes, or
    /// rows laid out together), or on chunked storage with records larger
    /// than a chunk, which no chunk holds whole.
    pub fn record_placement(&self) -> Option<RecordPlacement> {
        self.inner.record_placement()
    }

    /// Form now the address table a native record port binds this view's
    /// records through, where its backend binds record tables (Metal,
    /// Vulkan), so
    /// binding this view later allocates nothing: for a view first bound
    /// while its device may refuse an allocation, such as a release under
    /// memory pressure. A fixed allocation holds the table with its storage,
    /// and a view of a single record later binds any table it holds with an
    /// entry at that record's address.
    pub fn hold_record_table(&self) -> Result<(), TensorError> {
        self.inner.hold_records()
    }

    #[cfg(test)]
    pub(crate) fn device_address(&self) -> u64 {
        self.inner.device_address()
    }

    /// Allocate without initializing storage. A pure-output kernel can use
    /// this to avoid a host zero-fill before writing the result.
    ///
    /// # Safety
    /// Every byte of the tensor's physical layout, including padding, must
    /// be initialized before the tensor is read or passed as a device input.
    pub unsafe fn uninitialized(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        // SAFETY: the caller upholds the complete-initialization contract.
        unsafe {
            seismic_runtime::api::tensor::TensorInner::uninitialized(
                device.inner(),
                element.id(),
                extents,
            )
        }
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }

    /// A zero-filled tensor in storage the host writes and reads in place
    /// while device work that uses it is queued or running
    /// (`Contract::Shared`): memory cached for the host's reads and stores
    /// on every backend (mapped pinned host memory on CUDA, a host-cached
    /// type on Vulkan, never a BAR window): the slots of a table the host
    /// fills while steps that read it wait at a gate (`Timeline`), or a
    /// notice the device writes for the host. On Metal it is
    /// `MTLStorageModeShared`, as is every Metal tensor.
    pub fn shared(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::shared(device.inner(), element.id(), extents)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }

    /// A zero-filled tensor of coherent words the host and running device
    /// work exchange (`Contract::Words`; [`Tensor::load_acquire`],
    /// [`Tensor::store_release`]): timelines, park and answer words, maps
    /// and directories.
    pub fn words(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::words(device.inner(), element.id(), extents).map(
            |inner| Tensor {
                inner: Arc::new(inner),
            },
        )
    }

    /// A [`Tensor::words`] tensor whose every order across the
    /// device's queues the protocol of the work that uses them gives: work
    /// of one native queue never waits for another's use of them (host
    /// access still waits for every use). A slot of an exchange that one
    /// queue's work requests and another's waits on, which the host
    /// answers.
    pub fn protocol_words(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::protocol_words(
            device.inner(),
            element.id(),
            extents,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }

    /// A zero-filled tensor in upload storage (`Contract::Upload`): what
    /// the host only writes in place while device work reads it, where a
    /// graph run's inputs lie (a large BAR's on Vulkan, whose host reads
    /// are uncached). The host never reads it.
    pub fn upload(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::upload(device.inner(), element.id(), extents)
            .map(|inner| Tensor {
                inner: Arc::new(inner),
            })
    }

    /// Write `bytes` at `offset` bytes into this view without waiting for
    /// device work that uses its storage.
    ///
    /// # Safety
    /// The tensor was made by [`Tensor::shared`] (or is a view of one), and
    /// no device work in flight reads or writes these bytes: what the device
    /// reads after it, it reads after a gate the host opens once written.
    pub unsafe fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<(), TensorError> {
        // SAFETY: the caller upholds the contract.
        unsafe { self.inner.write_shared(offset, bytes) }
    }

    /// The host address of this view's first byte, where the host addresses
    /// its storage in place (a [`Tensor::shared`] view); none otherwise.
    /// Valid while the tensor lives. Bytes written through it are written
    /// under [`Tensor::write_shared`]'s contract, in place: a read from a
    /// file landing straight in a staging record, say.
    pub fn shared_address(&self) -> Option<std::ptr::NonNull<u8>> {
        self.inner.host_address()
    }

    /// Read `into.len()` bytes at `offset` bytes into this view without
    /// waiting for device work that uses its storage.
    ///
    /// # Safety
    /// The tensor was made by [`Tensor::shared`] (or is a view of one), and
    /// no device work in flight writes these bytes: the device wrote them
    /// before a signal the host waited for.
    pub unsafe fn read_shared(&self, offset: u64, into: &mut [u8]) -> Result<(), TensorError> {
        // SAFETY: the caller upholds the contract.
        unsafe { self.inner.read_shared(offset, into) }
    }

    /// A [`Tensor::shared`] view's 32-bit words, for stores that order
    /// nothing: a device reading one sees the store in time, and no other
    /// write by it. [`TensorError::SharedWord`] as [`Tensor::load_acquire`].
    pub fn shared_words(&self) -> Result<&[std::sync::atomic::AtomicU32], TensorError> {
        self.inner.shared_words()
    }

    /// The 32-bit word at `offset` bytes into a [`Tensor::shared`] view (or
    /// any view of host-addressed storage), loaded with acquire ordering
    /// while device work that uses it may be running: what the device wrote
    /// before publishing the word (a store after a system-scope fence) is
    /// visible after it. It neither waits for nor excludes device work.
    /// [`TensorError::SharedWord`] when the word is not 4-byte aligned, lies
    /// outside the view, or its storage is not host-addressed.
    pub fn load_acquire(&self, offset: u64) -> Result<u32, TensorError> {
        self.inner.load_acquire(offset)
    }

    /// Publish `value` as the 32-bit word at `offset` bytes into a
    /// [`Tensor::shared`] view to device work that may be running: a
    /// release store, then a sequentially consistent fence. Every earlier
    /// host write is visible to a device load of the word under a
    /// system-scope acquire (a system-coherent load, then a system-scope
    /// fence), and the store is ordered before every later host load. It
    /// neither waits for nor excludes device work; errors as
    /// [`Tensor::load_acquire`].
    pub fn store_release(&self, offset: u64, value: u32) -> Result<(), TensorError> {
        self.inner.store_release(offset, value)
    }

    /// The 64-bit word at `offset` bytes, loaded as one access with acquire
    /// ordering, as [`Tensor::load_acquire`] loads a 32-bit one; the word
    /// 8-byte aligned.
    pub fn load_acquire_u64(&self, offset: u64) -> Result<u64, TensorError> {
        self.inner.load_acquire_u64(offset)
    }

    /// Publish `value` as the 64-bit word at `offset` bytes as one store, as
    /// [`Tensor::store_release`] publishes a 32-bit one: a reader never
    /// assembles a value the host never held. The word 8-byte aligned.
    pub fn store_release_u64(&self, offset: u64, value: u64) -> Result<(), TensorError> {
        self.inner.store_release_u64(offset, value)
    }

    /// Publish `update` of the 64-bit word at `offset` bytes as one
    /// read-modify-write, where it returns a value for what the word holds,
    /// as [`Tensor::store_release_u64`] publishes a store; none leaves the
    /// word as it is. Whether it published. The word 8-byte aligned.
    pub fn update_release_u64(
        &self,
        offset: u64,
        update: impl FnMut(u64) -> Option<u64>,
    ) -> Result<bool, TensorError> {
        self.inner.update_release_u64(offset, update)
    }

    /// A zero-filled owned tensor.
    pub fn zeros(
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::zeros(device.inner(), element.id(), extents).map(
            |inner| Tensor {
                inner: Arc::new(inner),
            },
        )
    }
    /// An owned tensor initialized from host bytes in the representation's
    /// canonical dense layout.
    pub fn from_host(
        device: &Device,
        element: Element,
        extents: &[u64],
        bytes: &[u8],
    ) -> Result<Tensor, TensorError> {
        seismic_runtime::api::tensor::TensorInner::from_host(
            device.inner(),
            element.id(),
            extents,
            bytes,
        )
        .map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    /// A zero-filled tensor over `extents` whose leading-axis rows
    /// `[0, committed)` are physically backed and charged, and the
    /// [`Resizer`] that backs and releases the rest of the leading extent's
    /// reserved address space in place, one resize at a time. Its owner
    /// installs a resize between submissions (never between queuing a
    /// sequence that binds the tensor and submitting that sequence). Graphs
    /// are sealed over the whole shape, while device work bound to the
    /// tensor must stay within the committed rows, and host access reaching
    /// past them is refused (views within them read and write as ordinary
    /// tensors). The CPU backend, Metal and CUDA with virtual memory
    /// management reserve; elsewhere it is refused
    /// ([`TensorError::Unreservable`]). A reserved tensor is never slabbed.
    pub fn reserved(
        device: &Device,
        element: Element,
        extents: &[u64],
        committed: u64,
    ) -> Result<(Tensor, Resizer), TensorError> {
        Self::reserved_in(
            ReservedContract::DeviceOnly,
            device,
            element,
            extents,
            committed,
            None,
        )
    }

    /// [`Tensor::reserved`] under `contract`, where the device reserves it
    /// (`Executor::reserving`); elsewhere it is refused
    /// ([`TensorError::Unreservable`]). A `Shared` range is one the host
    /// writes and reads in place while device work that uses it is queued
    /// or running, wherever the placement rule puts it; its host address
    /// never moves, and a region of its committed rows is host-addressed while it
    /// pins them ([`Tensor::shared_address`]). Its records may carry views
    /// of up to `carries` bytes whole (another form's records packed in
    /// them, [`ReservedRecords::carries`]), which no chunk boundary crosses.
    pub fn reserved_in(
        kind: ReservedContract,
        device: &Device,
        element: Element,
        extents: &[u64],
        committed: u64,
        carries: Option<std::num::NonZeroU64>,
    ) -> Result<(Tensor, Resizer), TensorError> {
        seismic_runtime::api::tensor::TensorInner::reserved(
            device.inner(),
            kind,
            element.id(),
            extents,
            committed,
            carries,
        )
        .map(|(inner, resizer)| {
            let tensor = Tensor {
                inner: Arc::new(inner),
            };
            (tensor, resizer)
        })
    }
    /// Leading rows physically backed: every row unless the tensor reaches
    /// past a reservation's committed rows.
    pub fn committed_rows(&self) -> u64 {
        self.inner.committed_rows()
    }
    pub fn read_to_host(&self) -> Result<Vec<u8>, TensorError> {
        self.inner.read_to_host()
    }
    /// Replaces the bytes of this exact tensor view.  The byte count must
    /// match its canonical representation layout; allocation-level access
    /// exclusion makes this safe even when other views share the allocation.
    pub fn write_from_host(&mut self, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_from_host(bytes)
    }
    /// Write a canonical upload tensor's physical bytes in place as `fill`
    /// hands them over, each chunk at its byte offset, holding the tensor's
    /// host access until `fill` returns: never a second whole tensor on the
    /// host. Covering every byte is `fill`'s to do; on error the tensor is
    /// only partly initialized and must not be used.
    pub fn write_chunks(
        &mut self,
        fill: impl FnOnce(&mut dyn FnMut(u64, &[u8]) -> std::io::Result<()>) -> std::io::Result<()>,
    ) -> Result<(), TensorError> {
        self.inner.write_chunks(fill)
    }
    pub fn device(&self) -> Device {
        Device {
            inner: self.inner.device().clone(),
        }
    }
    pub fn element(&self) -> Element {
        Element(self.inner.representation())
    }
    pub fn extents(&self) -> &[u64] {
        self.inner.extents()
    }
    pub fn strides(&self) -> &[u64] {
        self.inner.strides()
    }
    /// A view over a contiguous range of the leading axis.
    pub fn slice_leading(&self, start: u64, end: u64) -> Result<Tensor, TensorError> {
        self.inner.slice_leading(start, end).map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    /// A canonical-layout view with different logical extents and identical
    /// physical byte coverage.
    pub fn reshape(&self, extents: &[u64]) -> Result<Tensor, TensorError> {
        self.inner.reshape(extents).map(|inner| Tensor {
            inner: Arc::new(inner),
        })
    }
    pub fn byte_len(&self) -> u64 {
        self.inner.byte_len()
    }
    pub fn storage_bytes(&self) -> u64 {
        self.inner.storage_bytes()
    }
    /// What this tensor's allocation is charged, the record tables its
    /// storage holds included.
    pub fn charged_bytes(&self) -> u64 {
        self.inner.charged_bytes()
    }
    /// The bytes of this contiguous view charged to each store: the share
    /// of its allocation's path charges its byte range covers.
    pub fn view_store_charges(&self) -> Result<Vec<(StoreId, u64)>, TensorError> {
        self.inner.view_store_charges()
    }
    pub fn observe_storage(&self) -> TensorStorageObserver {
        TensorStorageObserver {
            inner: self.inner.observe_storage(),
        }
    }
    pub fn belongs_to(&self, device: &Device) -> bool {
        Arc::ptr_eq(self.inner.device(), device.inner())
    }
    pub fn shares_allocation(&self, other: &Tensor) -> bool {
        self.inner.shares_allocation(&other.inner)
    }

    /// Physical bytes released if exactly these tensor handles are dropped.
    /// Views and clones outside the supplied set continue to pin storage.
    pub fn reclaimable_bytes<'a>(
        tensors: impl IntoIterator<Item = &'a Tensor>,
    ) -> Result<u64, TensorError> {
        seismic_runtime::api::tensor::TensorInner::reclaimable_bytes(
            tensors.into_iter().map(|tensor| &tensor.inner),
        )
    }
    pub(crate) fn descriptor(&self) -> seismic_compiler::prepared::TensorDescriptor {
        self.inner.descriptor()
    }
    pub(crate) fn inner(&self) -> &Arc<seismic_runtime::api::tensor::TensorInner> {
        &self.inner
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("element", &self.element().name())
            .field("extents", &self.extents())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Failure of `for_device`.
#[derive(Debug)]
pub enum LoadError {
    Bundle(CheckedBundleError),
    Source(SourceLoadError),
    Preparation(PreparationError),
}

impl LoadError {
    fn from_prepare(error: seismic_runtime::api::kernel::PrepareError) -> Self {
        match error {
            seismic_runtime::api::kernel::PrepareError::Source(error) => {
                Self::Source(SourceLoadError::from_internal(error))
            }
            seismic_runtime::api::kernel::PrepareError::Preparation(error) => {
                Self::Preparation(error)
            }
        }
    }
}

/// A checked source/binding error discovered while instantiating a generated
/// entry. The checked compiler object and its internal IDs are deliberately
/// not part of the public API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLoadError {
    message: String,
}

impl SourceLoadError {
    fn from_internal(error: seismic_lang::checked::SourceError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl fmt::Display for SourceLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SourceLoadError {}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(e) => write!(f, "{e}"),
            Self::Source(e) => write!(f, "{e}"),
            Self::Preparation(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for LoadError {}

/// Symbolic tensor result owned by one workflow draft. It has no ordinary
/// tensor operations and cannot be passed to `Kernel::call`.
#[derive(Clone)]
pub struct WorkflowTensor {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
}

#[derive(Clone)]
pub struct WorkflowTensorView {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
    operations: Vec<seismic_runtime::api::kernel::ViewOperation>,
    /// Each of `operations` with its numbers as expressions over a graph
    /// trace's facts ([`trace`]); none where no fact reaches any of them.
    expressions: Option<Arc<[seismic_runtime::native::dimension::ViewExpr]>>,
}

impl WorkflowTensor {
    /// A symbolic contiguous view of the leading axis. Bounds and packet
    /// alignment are validated during whole-workflow admission, once the
    /// producer's exact result descriptor exists.
    pub fn slice_leading<S, E>(&self, start: S, end: E) -> WorkflowTensorView
    where
        Dim: From<S> + From<E>,
    {
        WorkflowTensorView {
            inner: self.inner,
            operations: Vec::new(),
            expressions: None,
        }
        .slice_leading(start, end)
    }
    /// A symbolic canonical reshape of a workflow result. Seismic checks
    /// physical byte coverage and representation before graph execution.
    pub fn reshape<D: Clone>(&self, extents: &[D]) -> WorkflowTensorView
    where
        Dim: From<D>,
    {
        WorkflowTensorView {
            inner: self.inner,
            operations: Vec::new(),
            expressions: None,
        }
        .reshape(extents)
    }
}

impl WorkflowTensorView {
    pub fn slice_leading<S, E>(&self, start: S, end: E) -> Self
    where
        Dim: From<S> + From<E>,
    {
        let (start, end) = (Dim::from(start), Dim::from(end));
        self.then(
            seismic_runtime::api::kernel::ViewOperation::LeadingSlice { start: start.recorded(), end: end.recorded() },
            seismic_runtime::native::dimension::ViewExpr::LeadingSlice {
                start: start.expression(),
                end: end.expression(),
            },
        )
    }
    pub fn reshape<D: Clone>(&self, extents: &[D]) -> Self
    where
        Dim: From<D>,
    {
        let extents = extents.iter().cloned().map(Dim::from).collect::<Vec<_>>();
        self.then(
            seismic_runtime::api::kernel::ViewOperation::Reshape { extents: extents.iter().map(Dim::recorded).collect() },
            seismic_runtime::native::dimension::ViewExpr::Reshape {
                extents: extents.iter().map(Dim::expression).collect(),
            },
        )
    }

    /// It followed by `operation`, which is `expression` at the recording's
    /// facts. Expressions are kept only once a fact reaches one.
    fn then(
        &self,
        operation: seismic_runtime::api::kernel::ViewOperation,
        expression: seismic_runtime::native::dimension::ViewExpr,
    ) -> Self {
        let expressions = match (&self.expressions, expression.is_constant()) {
            (None, true) => None,
            (Some(expressions), _) => Some(expressions.iter().cloned().chain([expression]).collect()),
            (None, false) => Some(
                self.operations
                    .iter()
                    .map(seismic_runtime::native::dimension::ViewExpr::constant)
                    .chain([expression])
                    .collect(),
            ),
        };
        let mut operations = self.operations.clone();
        operations.push(operation);
        Self { inner: self.inner, operations, expressions }
    }
}

pub enum WorkflowTensorRef<'a> {
    External(&'a Tensor),
    Result(&'a WorkflowTensor),
    View(&'a WorkflowTensorView),
}

impl<'a> From<&'a Tensor> for WorkflowTensorRef<'a> {
    fn from(value: &'a Tensor) -> Self {
        Self::External(value)
    }
}

impl<'a> From<&'a WorkflowTensor> for WorkflowTensorRef<'a> {
    fn from(value: &'a WorkflowTensor) -> Self {
        Self::Result(value)
    }
}

impl<'a> From<&'a WorkflowTensorView> for WorkflowTensorRef<'a> {
    fn from(value: &'a WorkflowTensorView) -> Self {
        Self::View(value)
    }
}

pub enum WorkflowTensorMut<'a> {
    External(&'a mut Tensor),
    Result(&'a mut WorkflowTensor),
    View(&'a mut WorkflowTensorView),
}

impl<'a> From<&'a mut Tensor> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut Tensor) -> Self {
        Self::External(value)
    }
}

impl<'a> From<&'a mut WorkflowTensor> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut WorkflowTensor) -> Self {
        Self::Result(value)
    }
}

impl<'a> From<&'a mut WorkflowTensorView> for WorkflowTensorMut<'a> {
    fn from(value: &'a mut WorkflowTensorView) -> Self {
        Self::View(value)
    }
}

enum WorkflowTensorOwnedValue {
    External(Tensor),
    Result(WorkflowTensor),
}

pub struct WorkflowTensorOwned<'a> {
    value: WorkflowTensorOwnedValue,
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> From<Tensor> for WorkflowTensorOwned<'a> {
    fn from(value: Tensor) -> Self {
        Self {
            value: WorkflowTensorOwnedValue::External(value),
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a> From<WorkflowTensor> for WorkflowTensorOwned<'a> {
    fn from(value: WorkflowTensor) -> Self {
        Self {
            value: WorkflowTensorOwnedValue::Result(value),
            marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct WorkflowScalar<T> {
    inner: seismic_runtime::api::kernel::WorkflowResultRef,
    marker: std::marker::PhantomData<fn() -> T>,
}

pub enum WorkflowScalarValue<'a, T> {
    Immediate(T),
    Result(&'a WorkflowScalar<T>),
}

/// The contract a generated entry type satisfies.
///
/// # Safety
///
/// Implementations must be emitted together with the content-addressed
/// checked bundle by `seismic-build`. `Args`, `OutputArgs`, `Results`, their
/// encoders, `decode`, and `resolve` must describe that bundle entry exactly. Safe consumers
/// never implement this trait; the unsafe boundary is what lets the runtime
/// treat a schema mismatch as a compiler/generator invariant rather than a
/// recoverable invocation condition.
pub unsafe trait Entry: 'static {
    /// Typed arguments.
    type Args<'a>;
    /// Typed results.
    type Results;
    /// Caller-owned tensor storage for checked native results.
    type OutputArgs<'a>;
    /// Typed arguments whose tensor/scalar leaves may reference results of
    /// earlier nodes in the same workflow draft.
    type WorkflowArgs<'a>;
    /// Typed symbolic results returned while constructing a workflow.
    type WorkflowResults;
    const NAME: &'static str;
    /// Fixed device bytes one prepared direct native entry holds: its
    /// scalar-result slots, none without scalar results.
    const NATIVE_SCALAR_RESULT_BYTES: u64;
    /// The checked module this entry belongs to.
    fn module() -> Result<&'static generated::Module, CheckedBundleError>;
    fn resolve(module: &generated::Module) -> Result<generated::EntryToken, CheckedBundleError>;
    fn encode(args: Self::Args<'_>) -> generated::EncodedArgs;
    fn encode_outputs(outputs: Self::OutputArgs<'_>) -> generated::EncodedOutputs;
    fn decode(results: generated::DecodedResults) -> Self::Results;
    fn encode_workflow(args: Self::WorkflowArgs<'_>) -> generated::EncodedWorkflowArgs;
    fn decode_workflow(results: generated::PendingWorkflowResults) -> Self::WorkflowResults;
    fn workflow_outputs(results: Self::WorkflowResults) -> Vec<generated::WorkflowResultRef>;
}

/// A prepared kernel for one entry on one device under one precision
/// policy. `call` validates, selects, allocates, and executes (§14.4).
pub struct Kernel<E: Entry> {
    inner: Arc<seismic_runtime::api::kernel::PreparedAny>,
    marker: std::marker::PhantomData<E>,
}

/// Mutable feedback search state. Kernels returned by this preparation own
/// their executable resources independently and are never changed by continuation.
pub struct FeedbackPreparation<'device, E: Entry> {
    inner: seismic_runtime::api::kernel::FeedbackPreparation<'device>,
    marker: std::marker::PhantomData<E>,
}

impl<'device, E: Entry> FeedbackPreparation<'device, E> {
    fn start(
        device: &'device Device,
        precision: PrecisionPolicy,
        options: FeedbackOptions,
        bindings: seismic_lang::entry::ElementBindings,
    ) -> Result<(Self, Kernel<E>), LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        let (inner, kernel) = seismic_runtime::api::kernel::start_feedback(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            precision,
            options,
        )
        .map_err(LoadError::from_prepare)?;
        Ok((
            Self {
                inner,
                marker: std::marker::PhantomData,
            },
            Kernel {
                inner: Arc::new(kernel),
                marker: std::marker::PhantomData,
            },
        ))
    }

    pub fn continue_for(
        &mut self,
        additional: std::time::Duration,
    ) -> Result<Kernel<E>, LoadError> {
        self.inner
            .continue_for(additional)
            .map(|inner| Kernel {
                inner: Arc::new(inner),
                marker: std::marker::PhantomData,
            })
            .map_err(LoadError::from_prepare)
    }

    pub fn report(&self) -> &FeedbackReport {
        self.inner.report()
    }
}

/// An explicitly selected, top-level native implementation of one entry.
/// It has the same typed call contract as [`Kernel`]. It can be composed into
/// a direct-native graph, but is not a portable workflow/compiler candidate.
pub struct NativeKernel<E: Entry> {
    inner: Arc<seismic_runtime::api::kernel::NativePreparedAny>,
    marker: std::marker::PhantomData<E>,
}

/// Tensor-result native calls from any prepared entries on one device,
/// encoded in order and completed with one device wait.
pub struct NativeTensorBatch {
    inner: seismic_runtime::api::kernel::NativeTensorBatchAny,
}

pub struct NativeTensorBatchCompletion {
    inner: seismic_runtime::api::kernel::NativeTensorBatchCompletionAny,
}

impl NativeTensorBatch {
    pub fn new(device: &Device) -> Self {
        Self {
            inner: seismic_runtime::api::kernel::NativeTensorBatchAny::new(device.inner()),
        }
    }

    pub fn push<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        args: E::Args<'_>,
        outputs: E::OutputArgs<'_>,
    ) -> Result<(), CallError> {
        self.inner
            .push(&kernel.inner, E::encode(args), E::encode_outputs(outputs))
    }

    pub fn submit(self) -> Result<NativeTensorBatchCompletion, CallError> {
        self.inner
            .submit()
            .map(|inner| NativeTensorBatchCompletion { inner })
    }
}

impl NativeTensorBatchCompletion {
    /// Whether the device has finished the batch, without waiting.
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Observe completion and any device failure. Dropping an unobserved
    /// completion still waits so its mapped inputs stay alive.
    pub fn wait(self) -> Result<(), CallError> {
        self.inner.wait()
    }
}

/// The points of one native tuning unit, whose inputs are built on request
/// and held by the tuner only while its work uses them.
pub trait PointSource<'a, E: Entry> {
    /// The tuner gave up the inputs built for `point`: drop what the source
    /// kept of them. A later [`PointSource::build`] builds them again.
    fn release(&mut self, _point: usize) {}
    /// Every point, in ascending cost.
    fn points(&self) -> Vec<PointSpec>;
    /// The dimensions of calls the unit's configuration serves at shapes its
    /// points' inputs do not take, each naming every dimension its launches
    /// read: a configuration whose launches at any exceed the formed
    /// functions' or the device's limits is excluded as it is formed.
    fn served(&self) -> Vec<std::collections::BTreeMap<String, u64>> {
        Vec::new()
    }
    /// Build point `point`'s inputs, or say why they cannot be built: an
    /// allocation the device refused is [`BuildError::Refused`] (what a
    /// [`TensorError`] or [`CallError`] of allocation capacity converts to),
    /// which the tuner prices and offers again rather than failing the unit.
    fn build(&mut self, point: usize) -> Result<PointInputs<'a>, BuildError>;
    /// A survey ([`Strategy::Survey`]) came this far: reported once its
    /// configurations are counted, before it forms any, and as each batch
    /// of them is formed and measured.
    fn surveyed(&mut self, _progress: SurveyProgress) {}
}

impl<'a, E: Entry, S: PointSource<'a, E> + ?Sized> PointSource<'a, E> for &mut S {
    fn release(&mut self, point: usize) {
        (**self).release(point);
    }
    fn points(&self) -> Vec<PointSpec> {
        (**self).points()
    }

    fn served(&self) -> Vec<std::collections::BTreeMap<String, u64>> {
        (**self).served()
    }

    fn build(&mut self, point: usize) -> Result<PointInputs<'a>, BuildError> {
        (**self).build(point)
    }

    fn surveyed(&mut self, progress: SurveyProgress) {
        (**self).surveyed(progress);
    }
}

/// A typed point source as Seismic's runtime asks it.
struct TypedPoints<'s, 'a, E: Entry>(&'s mut dyn PointSource<'a, E>);

impl<'a, E: Entry> seismic_runtime::native::tune::PointSource<'a> for TypedPoints<'_, 'a, E> {
    fn release(&mut self, point: usize) {
        self.0.release(point);
    }
    fn points(&self) -> Vec<PointSpec> {
        self.0.points()
    }

    fn served(&self) -> Vec<std::collections::BTreeMap<String, u64>> {
        self.0.served()
    }

    fn build(&mut self, point: usize) -> Result<PointInputs<'a>, BuildError> {
        self.0.build(point)
    }

    fn surveyed(&mut self, progress: SurveyProgress) {
        self.0.surveyed(progress);
    }
}

/// The tuning of one native unit of entry `E`, advanced one piece of work at
/// a time. From [`NativeSearch::prepare_defaults`] on the unit holds a legal
/// served choice with a complete record ([`NativeSearch::terminal`]);
/// everything after is optional and can only replace that choice with a
/// better one. A consumer tuning several units decides which unit's next
/// piece ([`NativeSearch::next`]) is worth starting, and whether its whole
/// price fits the budget left ([`Quote::fits`]); a piece begun
/// ([`NativeSearch::perform`]) runs to its end.
pub struct NativeSearch<'a, E: Entry> {
    inner: seismic_runtime::native::tune::UnitSearch<'static, 'a>,
    entry: std::marker::PhantomData<E>,
}

/// The compile of a unit's next piece of work ([`NativeSearch::forming`]).
pub type Forming = seismic_runtime::native::tune::Forming<'static>;
/// The compile of a unit's canonical program ([`NativeSearch::grounding`]).
pub type Grounding = seismic_runtime::native::tune::Grounding<'static>;
impl<'a, E: Entry> NativeSearch<'a, E> {
    /// The unit's mandatory work: form the declared canonical program and
    /// hand it to serving, without building point inputs or observations.
    pub fn prepare_defaults(
        &mut self,
        points: &mut dyn PointSource<'a, E>,
    ) -> Result<(), TuneError> {
        self.inner.prepare_defaults(&mut TypedPoints(points))
    }

    /// The compile of the mandatory work, to run on any thread; none once
    /// the canonical program is formed. [`NativeSearch::grounded`] takes
    /// what it formed and ends the mandatory work.
    pub fn grounding(
        &mut self,
        points: &mut dyn PointSource<'a, E>,
    ) -> Result<Option<Grounding>, TuneError> {
        self.inner.grounding(&TypedPoints(points))
    }

    pub fn grounded(&mut self, grounded: Grounded) -> Result<(), TuneError> {
        self.inner.grounded(grounded)
    }

    /// The compile of the next piece of work ([`Work::forming`]), to run
    /// ahead of it on any thread; none when the piece compiles nothing
    /// ahead.
    pub fn forming(&self) -> Option<Forming> {
        self.inner.forming()
    }

    /// Give up what only running work needs: every point's inputs
    /// (`points` drops what it built), the default's observations of them
    /// and result storage; with no work left, the held programs too. The
    /// served choice and the record stay, and work that follows builds the
    /// inputs again, priced with that.
    pub fn release(&mut self, points: &mut dyn PointSource<'a, E>) {
        self.inner.release(&mut TypedPoints(points))
    }

    /// Offer the next piece with one challenger fewer; false when it has
    /// no fewer to offer.
    pub fn narrow(&mut self) -> bool {
        self.inner.narrow()
    }

    /// Whether any of the unit's points has its inputs resident.
    pub fn holds_inputs(&self) -> bool {
        self.inner.holds_inputs()
    }

    /// The allocations the unit's resident inputs bind, each once: its
    /// identity and its backed bytes. None when the unit holds no inputs.
    pub fn bound_allocations(&self) -> Option<Vec<(u64, u64)>> {
        self.inner.bound_allocations()
    }

    /// How many programs the unit's search holds formed.
    pub fn programs_held(&self) -> usize {
        self.inner.programs_held()
    }

    /// What the unit's held programs hold, as forming them was observed to
    /// add.
    pub fn held_programs(&self) -> u64 {
        self.inner.held_programs()
    }

    /// Give up the programs the unit's search holds; a piece that follows
    /// forms what it runs again, priced with that.
    pub fn give_up_programs(&mut self) {
        self.inner.give_up_programs()
    }

    /// The most a single piece of the unit's work has added to the bytes
    /// held beyond the inputs it kept resident: what an unpriced footprint
    /// is held to; none until a piece of it has been measured.
    pub fn largest_footprint(&self) -> Option<u64> {
        self.inner.largest_footprint()
    }

    /// Hold what the next piece's compile formed. Returns what the
    /// compile took.
    pub fn formed(&mut self, formed: Formed) -> Result<std::time::Duration, TuneError> {
        self.inner.formed(formed)
    }

    /// How the unit's optional work measures and searches.
    pub fn plan(&mut self, plan: UnitPlan) {
        self.inner.plan(plan)
    }

    /// The unit's complete record as it stands.
    pub fn terminal(&self) -> Result<TuningResult, TuneError> {
        self.inner.terminal()
    }

    /// The unit's next piece of work with its whole price; none when it
    /// has none left.
    pub fn next(&self) -> Option<Work> {
        self.inner.next()
    }

    /// Run optional work before the consumer's cutoff. None is for an
    /// explicitly unbounded caller, such as a kernel survey or test.
    pub fn perform(
        &mut self,
        points: &mut dyn PointSource<'a, E>,
        before: Option<std::time::Instant>,
    ) -> Result<Performed, TuneError> {
        self.inner.perform(&mut TypedPoints(points), before)
    }

    /// The longest single operation the unit has completed.
    pub fn longest_operation(&self) -> std::time::Duration {
        self.inner.longest_operation()
    }

    /// What building the inputs the unit holds took: what giving them up
    /// is held to before the load has observed inputs given up.
    pub fn inputs_built(&self) -> Quote {
        self.inner.inputs_built()
    }

    /// The defaults' measurements at the points discovered so far.
    pub fn service_measurements(&self) -> Vec<PointMeasurement> {
        self.inner.service_measurements()
    }

    /// Weigh the unit's points by `weights`, in the order its source gave
    /// them, before anything of its search is measured against the weights
    /// it opened with ([`UnitSearch::reweigh`]). False once it has.
    pub fn reweigh(&mut self, weights: &[f64]) -> bool {
        self.inner.reweigh(weights)
    }

    /// Whether the unit's defaults are measured at every discovery point.
    pub fn discovered(&self) -> bool {
        self.inner.discovered()
    }

    /// Where the unit's tuning stands.
    pub fn standing(&self) -> Standing {
        self.inner.standing()
    }
}

/// Complete numerical `rotation`, its declared performance `workload`, and
/// state initialization/observation as a point's inputs.
pub fn point_inputs<'a, E: Entry>(
    rotation: Vec<E::Args<'_>>,
    workload: WorkloadRotation,
    initialize: Option<TuningInitializer<'a>>,
    written: std::collections::BTreeMap<String, WrittenRows>,
) -> PointInputs<'a> {
    PointInputs {
        rotation: rotation.into_iter().map(E::encode).collect(),
        workload,
        initialize,
        written,
    }
}

/// One workload for native tuning with its inputs built up front: argument
/// sets forming its full numerical contract, an explicit performance workload,
/// and the point's invocation demand. Built up front, its inputs are handed
/// over once: a tuner that gave them up cannot have them built again.
pub struct TuningPoint<'a, E: Entry> {
    pub label: String,
    pub weight: f64,
    /// Served-region identity for keyed reuse and discovery; it does not
    /// independently normalize the expected elapsed-service objective.
    pub class: Option<String>,
    /// The point's cost relative to the unit's other points: points come in
    /// ascending cost, and the tuner predicts a point's time from the
    /// previous one's by the ratio of their costs.
    pub cost: f64,
    pub rotation: Vec<E::Args<'a>>,
    /// Performance invocations selected from the full numerical witness vector.
    pub workload: WorkloadRotation,
    /// Required when the entry has `&mut` parameters: restores every writable
    /// tensor in every rotation. The caller owns the pristine bytes; the tuner
    /// invokes this initializer before reference and validation executions.
    /// Timed passes after a candidate's validated invocation run without it.
    pub initialize: Option<TuningInitializer<'a>>,
    /// The leading-axis rows each named `&mut` parameter's entry writes. A
    /// `&mut` parameter absent here is observed whole, as written
    /// everywhere.
    pub written: std::collections::BTreeMap<String, WrittenRows>,
}

impl<'a, E: Entry> PointSource<'a, E> for Vec<TuningPoint<'a, E>> {
    fn points(&self) -> Vec<PointSpec> {
        self.iter()
            .map(|point| PointSpec {
                label: point.label.clone(),
                weight: point.weight,
                class: point.class.clone(),
                cost: point.cost,
                census: point.weight > 0.0,
            })
            .collect()
    }

    fn build(&mut self, point: usize) -> Result<PointInputs<'a>, BuildError> {
        let point = &mut self[point];
        if point.rotation.is_empty() {
            return Err(BuildError::Failed(format!(
                "the inputs of `{}` were built up front and handed over already",
                point.label
            )));
        }
        Ok(point_inputs::<E>(
            std::mem::take(&mut point.rotation),
            point.workload.clone(),
            point.initialize.take(),
            std::mem::take(&mut point.written),
        ))
    }
}

/// A direct-native graph is assembled from generated entry arguments and
/// symbolic result edges. Seismic derives every intermediate tensor from the
/// checked entry contracts and owns its storage plan.
pub struct NativeGraph {
    inner: seismic_runtime::native::graph::NativeGraphDraft,
}

/// Backend-free storage projection of a native graph. The caller follows the
/// same checked topology as a prepared graph. Native scratch uses the largest
/// checked charge across the declaration's finite tuning choices.
pub struct NativeGraphMetadata {
    backend: BackendName,
    inner: seismic_runtime::native::graph::NativeGraphMetadataDraft,
    template: Option<MetadataTemplateRecorder>,
    class_scope: Option<&'static str>,
}

/// The process's checked entries, keyed by the backend, the entry (its module
/// and name) and its element bindings: what each binds to is a pure function
/// of those, derived once, which every layer of a graph and every graph of
/// every model, form and load binds again. Only successes are held, so every
/// failure is the binding's own. A node's call at its dimensions is checked
/// once per process too ([`checked_call`]).
type CheckedEntries = std::collections::HashMap<EntryKey, generated::NativeGraphCheckedEntry>;

type EntryKey = (BackendName, usize, &'static str, Vec<(String, Element)>);

/// [`CheckedEntries`], one per process.
fn checked_entries() -> &'static std::sync::RwLock<CheckedEntries> {
    static CHECKED: std::sync::OnceLock<std::sync::RwLock<CheckedEntries>> = std::sync::OnceLock::new();
    CHECKED.get_or_init(Default::default)
}

/// A node's checked shapes and scratch, and the entry they were checked by.
struct CheckedNode {
    checked: generated::NativeGraphCheckedEntry,
    parameters: Vec<Option<(RepresentationId, Vec<u64>)>>,
    results: Vec<Option<(RepresentationId, Vec<u64>)>>,
    scratch: Vec<seismic_runtime::native::ScratchNeed>,
}

struct MetadataTemplateRecorder {
    ports: Vec<MetadataPortSource>,
    nodes: Vec<MetadataNodeSource>,
}

/// A template port: its extents, or its entry parameter's shape at its
/// dimensions, each a number `N`: an expression over the recording's facts
/// as the template keeps it ([`MetadataPortSource`], [`trace`]), or its value
/// at a model's facts ([`PortPoint`]).
enum TemplatePort<N> {
    Direct {
        element: Element,
        extents: Vec<N>,
        class_extent: Option<(usize, &'static str)>,
    },
    Checked {
        port: Arc<CheckedPortSource>,
        dimensions: Vec<(String, N)>,
    },
}

/// A template port as its recording keeps it.
type MetadataPortSource = TemplatePort<trace::Expr>;

/// A template port at a model's facts.
type PortPoint = TemplatePort<u64>;

/// What a checked port is apart from its dimensions.
struct CheckedPortSource {
    checked: generated::NativeGraphCheckedEntry,
    entry_name: &'static str,
    class_scope: Option<&'static str>,
    parameter: String,
}

impl MetadataPortSource {
    /// It at `facts`, `None` where a number does not evaluate there.
    fn at(&self, facts: &[u64]) -> Option<PortPoint> {
        Some(match self {
            Self::Direct { element, extents, class_extent } => TemplatePort::Direct {
                element: *element,
                extents: extents.iter().map(|extent| extent.evaluate(facts)).collect::<Option<_>>()?,
                class_extent: *class_extent,
            },
            Self::Checked { port, dimensions } => {
                TemplatePort::Checked { port: Arc::clone(port), dimensions: evaluated(dimensions, facts)? }
            }
        })
    }
}

struct MetadataNodeSource {
    checked: generated::NativeGraphCheckedEntry,
    entry_name: &'static str,
    class_scope: Option<&'static str>,
    dimensions: Vec<(String, trace::Expr)>,
}

/// A template's ports and nodes with every dimension evaluated: at its
/// recording, or at a trace's facts ([`NativeGraphResourceTemplate::points`]).
struct TemplatePoints {
    /// Each port at the facts.
    ports: Vec<PortPoint>,
    /// Each node's dimensions.
    nodes: Vec<Vec<(String, u64)>>,
}

fn evaluated(dimensions: &[(String, trace::Expr)], facts: &[u64]) -> Option<Vec<(String, u64)>> {
    dimensions
        .iter()
        .map(|(name, expr)| Some((name.clone(), expr.evaluate(facts)?)))
        .collect()
}

fn borrowed(dimensions: &[(String, u64)]) -> Vec<(&str, u64)> {
    dimensions.iter().map(|(name, value)| (name.as_str(), *value)).collect()
}

/// One checked graph topology reused across the numeric shape classes of a
/// structural regime. Certification asks the generated entry contracts for
/// exact shapes and scratch before Seismic places the graph's original edges
/// and lifetimes.
pub struct NativeGraphResourceTemplate {
    backend: BackendName,
    inner: seismic_runtime::native::graph::ValidatedMetadataDraft,
    recorder: MetadataTemplateRecorder,
}

/// One slice of the admitted classes of a structural graph regime: every
/// combination of its listed dimension values. A regime's classes are the
/// union of its slices, so correlated dimensions (slots bounded by rows) are
/// fixed together per slice. Every slice of a regime names the same
/// dimensions. A dimension's values are a set, held ascending and distinct:
/// a value listed twice adds no class, and every combination is formed
/// once.
#[derive(Clone, Debug, Default)]
pub struct NativeGraphClassSlice {
    dimensions: Vec<(&'static str, Vec<u64>)>,
    scoped: Vec<(&'static str, &'static str, Vec<u64>)>,
}

/// `values` ascending, each once.
fn distinct(values: impl IntoIterator<Item = u64>) -> Vec<u64> {
    values.into_iter().collect::<std::collections::BTreeSet<_>>().into_iter().collect()
}

impl NativeGraphClassSlice {
    pub fn new() -> Self {
        Self::default()
    }

    /// The values of a class dimension for every node that declares it.
    pub fn dimension(mut self, name: &'static str, values: impl IntoIterator<Item = u64>) -> Self {
        self.dimensions.push((name, distinct(values)));
        self
    }

    /// The values of a dimension for nodes of one class scope or entry name.
    /// A class scope takes precedence over an entry name, and both over an
    /// unscoped dimension of the same name.
    pub fn scoped(
        mut self,
        scope: &'static str,
        name: &'static str,
        values: impl IntoIterator<Item = u64>,
    ) -> Self {
        self.scoped.push((scope, name, distinct(values)));
        self
    }

    fn names(&self) -> Vec<(Option<&'static str>, &'static str)> {
        let mut names = self
            .dimensions
            .iter()
            .map(|(name, _)| (None, *name))
            .chain(
                self.scoped
                    .iter()
                    .map(|(scope, name, _)| (Some(*scope), *name)),
            )
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// The values of class dimension `name` of scope `scope`, or unscoped.
    fn class_values(&self, scope: Option<&str>, name: &str) -> Option<&[u64]> {
        match scope {
            None => self.global(name),
            Some(scope) => self
                .scoped
                .iter()
                .find(|(candidate, axis, _)| *candidate == scope && *axis == name)
                .map(|(_, _, values)| values.as_slice()),
        }
    }

    fn global(&self, name: &str) -> Option<&[u64]> {
        self.dimensions
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, values)| values.as_slice())
    }

    fn values(&self, entry_name: &str, class_scope: Option<&str>, name: &str) -> Option<&[u64]> {
        let scoped = |scope: &str| {
            self.scoped
                .iter()
                .find(|(candidate, axis, _)| *candidate == scope && *axis == name)
                .map(|(_, _, values)| values.as_slice())
        };
        class_scope
            .and_then(scoped)
            .or_else(|| scoped(entry_name))
            .or_else(|| self.global(name))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeGraphMetadataError {
    Bundle(CheckedBundleError),
    Unsupported(String),
    /// A call at statics outside its implementation's domain.
    Inadmissible(KernelDomainViolation),
    Tensor(TensorError),
    Call(CallError),
    Workflow(WorkflowError),
}

impl fmt::Display for NativeGraphMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(error) => write!(f, "{error}"),
            Self::Unsupported(reason) => write!(f, "{reason}"),
            Self::Inadmissible(violation) => write!(f, "{violation}"),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::Call(error) => write!(f, "{error}"),
            Self::Workflow(error) => write!(f, "{error}"),
        }
    }
}

/// An entry call whose statics lie outside its implementation's domain on
/// a backend: no configuration satisfies the implementation's `where`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelDomainViolation {
    pub entry: &'static str,
    pub backend: BackendName,
    /// The call's static dimensions, by name.
    pub statics: Vec<(String, u64)>,
}

impl fmt::Display for KernelDomainViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let statics = self
            .statics
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "`{}` has no admissible {} configuration at {statics}",
            self.entry,
            self.backend.as_str()
        )
    }
}

impl std::error::Error for NativeGraphMetadataError {}

pub use generated::forget_evaluated_entry_shapes;
pub use seismic_runtime::native::graph::{GraphFamilyCensus, HostHeap, NativeGraphStorageBytes};

#[derive(Clone)]
pub struct NativeGraphLayout {
    inner: Arc<seismic_runtime::native::graph::NativeGraphLayout>,
}

impl NativeGraphLayout {
    /// The host heap of each plan sealed into this layout, known from the
    /// layout's topology before any plan is sealed
    /// (`NativeGraphPlan::host_heap` is what a sealed plan holds).
    pub fn plan_heap(&self) -> HostHeap {
        self.inner.plan_heap()
    }

    /// The host heap of one static binding of a plan sealed into this
    /// layout, known before any plan is sealed or bound
    /// (`BoundNativeGraphPlan::host_heap` is what a binding holds).
    pub fn bound_heap(&self) -> HostHeap {
        self.inner.bound_heap()
    }

    pub fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.inner.storage_bytes()
    }
}

fn owned_dimensions(dimensions: &[(&str, u64)]) -> Vec<(String, u64)> {
    dimensions
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

/// Named dimensions as a graph takes them, each a [`Dim`] or a `u64`.
fn dims<'n>(dimensions: &[(&'n str, impl Clone + Into<Dim>)]) -> Vec<(&'n str, Dim)> {
    dimensions
        .iter()
        .map(|(name, dimension)| (*name, dimension.clone().into()))
        .collect()
}

/// Each dimension at the recording's facts: what the graph records at.
fn recorded_dimensions<'n>(dimensions: &[(&'n str, Dim)]) -> Vec<(&'n str, u64)> {
    dimensions
        .iter()
        .map(|(name, dimension)| (*name, dimension.recorded()))
        .collect()
}

/// Each dimension's expression over the recording's facts: what a template
/// keeps.
fn dimension_exprs(dimensions: &[(&str, Dim)]) -> Vec<(String, trace::Expr)> {
    dimensions
        .iter()
        .map(|(name, dimension)| ((*name).to_owned(), dimension.expression()))
        .collect()
}


/// What every admitted class of a structural regime holds
/// ([`NativeGraphResourceTemplate::outline`]): its storage, census and host
/// heaps exactly as the regime's layout
/// ([`NativeGraphResourceTemplate::certify`]) holds them, without the
/// layout a load seals into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeGraphOutline {
    storage: NativeGraphStorageBytes,
    plan_heap: HostHeap,
    bound_heap: HostHeap,
}

impl NativeGraphOutline {
    pub fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage
    }
    pub fn plan_heap(&self) -> HostHeap {
        self.plan_heap
    }
    pub fn bound_heap(&self) -> HostHeap {
        self.bound_heap
    }
}

/// Every buffer of a template at its capacity over a regime's classes.
struct Capacities {
    ports: Vec<u64>,
    results: Vec<Vec<u64>>,
    scratch: Vec<Vec<seismic_runtime::native::ScratchNeed>>,
}

impl NativeGraphResourceTemplate {
    /// Place one layout for every admitted class of this structural regime,
    /// at its buffers' capacities ([`Self::capacities`]). Seismic places
    /// those capacities once; production seals each exact class into the
    /// same offsets and rejects any class that does not fit.
    pub fn certify(
        &self,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphLayout, NativeGraphMetadataError> {
        self.certify_of(&self.recorded(), slices)
    }

    /// [`Self::certify`] at `points`.
    fn certify_of(
        &self,
        points: &TemplatePoints,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphLayout, NativeGraphMetadataError> {
        let Some(first) = slices.first() else {
            return Err(NativeGraphMetadataError::Unsupported("graph regime has no admitted classes".into()));
        };
        self.certify_with(&self.capacity_program(points, first)?, slices)
    }

    /// [`Self::certify`] with its buffers compiled into `program`
    /// ([`Self::capacity_program`]).
    fn certify_with(
        &self,
        program: &CapacityProgram,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphLayout, NativeGraphMetadataError> {
        let Capacities { ports, results, scratch } = program.at(slices)?;
        self.inner
            .capacity_layout(&ports, &results, scratch)
            .map(|inner| NativeGraphLayout {
                inner: Arc::new(inner),
            })
            .map_err(NativeGraphMetadataError::Workflow)
    }

    /// What [`Self::certify`] at the same classes holds, exactly, from the
    /// same capacities and the same storage plan, without building the
    /// layout.
    pub fn outline(
        &self,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphOutline, NativeGraphMetadataError> {
        self.outline_of(&self.recorded(), slices)
    }

    /// Its points at its recording: a template recorded outside a trace has
    /// constant dimensions alone.
    fn recorded(&self) -> TemplatePoints {
        self.points(&[])
            .expect("a template's constant dimensions are its recording's values")
    }

    /// Every port's and node's dimensions at `facts`, `None` where one does
    /// not evaluate there.
    fn points(&self, facts: &[u64]) -> Option<TemplatePoints> {
        let ports = self.recorder.ports.iter().map(|source| source.at(facts)).collect::<Option<_>>()?;
        let nodes = self
            .recorder
            .nodes
            .iter()
            .map(|source| evaluated(&source.dimensions, facts))
            .collect::<Option<_>>()?;
        Some(TemplatePoints { ports, nodes })
    }

    /// Whether every check the recording made of a shape holds at `points`:
    /// each checked port's parameter and each node's call admitted at its
    /// dimensions ([`checked_call`]: contract, domain, scratch), and each
    /// argument, stored and shared result matching where it is wired
    /// (`NativeGraphMetadataDraft::validate_at`). Topology is the
    /// recording's. A refusal is a trace that does not answer: the family
    /// records again at the model's facts, which reports the refusal itself.
    fn admitted_at(&self, facts: &[u64], points: &TemplatePoints) -> bool {
        let ports = points
            .ports
            .iter()
            .map(|point| match point {
                TemplatePort::Direct { extents, .. } => Some(extents.clone()),
                TemplatePort::Checked { port, dimensions } => port
                    .checked
                    .parameter(&port.parameter, &borrowed(dimensions))
                    .ok()
                    .map(|metadata| metadata.extents),
            })
            .collect::<Option<Vec<_>>>();
        let Some(ports) = ports else {
            return false;
        };
        let nodes = self
            .recorder
            .nodes
            .iter()
            .zip(&points.nodes)
            .map(|(source, dimensions)| {
                checked_call(&source.checked, source.entry_name, self.backend, dimensions).ok()
            })
            .collect::<Option<Vec<_>>>();
        let Some(nodes) = nodes else {
            return false;
        };
        let (parameters, results): (Vec<_>, Vec<_>) =
            nodes.iter().map(|node| (node.parameters.clone(), node.results.clone())).unzip();
        self.inner.validate_at(facts, &ports, &parameters, &results).is_ok()
    }

    /// What the template holds over `slices` at `points`
    /// ([`Self::outline`]).
    fn outline_of(
        &self,
        points: &TemplatePoints,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphOutline, NativeGraphMetadataError> {
        let Some(first) = slices.first() else {
            return Err(NativeGraphMetadataError::Unsupported("graph regime has no admitted classes".into()));
        };
        self.outline_with(&self.capacity_program(points, first)?, slices)
    }

    /// What the template holds over `slices` with its buffers compiled into
    /// `program` ([`Self::capacity_program`]).
    fn outline_with(
        &self,
        program: &CapacityProgram,
        slices: &[NativeGraphClassSlice],
    ) -> Result<NativeGraphOutline, NativeGraphMetadataError> {
        let Capacities { ports, results, scratch } = program.at(slices)?;
        let storage = self
            .inner
            .capacity_storage(&ports, &results, scratch)
            .map_err(NativeGraphMetadataError::Workflow)?;
        Ok(NativeGraphOutline {
            storage,
            plan_heap: self.inner.plan_heap(),
            bound_heap: self.inner.bound_heap(),
        })
    }

    /// Its buffers' sizes at `points` as functions of the class dimensions
    /// the slices named like `namer` vary: a buffer no class dimension
    /// reaches is evaluated here, once; any other is bound to its contract's
    /// compiled shape (or its scratch bound) over the class dimensions it
    /// reads alone, so a regime's slices evaluate it at their class points
    /// with no lookup by name and no contract walk ([`CapacityProgram::at`]).
    fn capacity_program(
        &self,
        points: &TemplatePoints,
        namer: &NativeGraphClassSlice,
    ) -> Result<CapacityProgram, NativeGraphMetadataError> {
        let unsupported = NativeGraphMetadataError::Unsupported;
        let ports = points
            .ports
            .iter()
            .map(|point| match point {
                TemplatePort::Direct { element, extents, class_extent } => {
                    let element = *element;
                    let Some((axis, name)) = *class_extent else {
                        return element
                            .canonical_byte_len(extents)
                            .map(Capacity::Fixed)
                            .map_err(NativeGraphMetadataError::Tensor);
                    };
                    let extents = extents.clone();
                    Ok(Capacity::Varying {
                        reads: vec![ClassRead::Global(name)],
                        size: Box::new(move |values: &[u64]| {
                            let mut extents = extents.clone();
                            extents[axis] = values[0];
                            element.canonical_byte_len(&extents).map_err(|error| error.to_string())
                        }),
                    })
                }
                TemplatePort::Checked { port, dimensions } => {
                    let CheckedPortSource { checked, entry_name, class_scope, parameter } = &**port;
                    let reads = checked.parameter_dimensions(parameter).map_err(unsupported)?;
                    bound_capacity(
                        checked,
                        seismic_lang::entry::TensorSlot::Parameter(parameter),
                        namer,
                        entry_name,
                        *class_scope,
                        dimensions,
                        &reads,
                    )
                    .map_err(|error| unsupported(format!("`{entry_name}.{parameter}`: {error}")))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        // A graph repeats the same call in every layer: the same contract
        // and implementation at the same class scope and dimensions have
        // the same buffers, so each distinct call is compiled once and
        // evaluated once per slice set ([`CapacityProgram::at`]).
        let mut distinct = std::collections::HashMap::<
            (usize, usize, Option<&'static str>, &[(String, u64)]),
            usize,
        >::new();
        let mut programs = Vec::new();
        let mut nodes = Vec::with_capacity(self.recorder.nodes.len());
        for (source, dimensions) in self.recorder.nodes.iter().zip(&points.nodes) {
            let implementation = source.checked.implementation();
            let key = (
                source.checked.identity(),
                std::ptr::from_ref(implementation) as usize,
                source.class_scope,
                dimensions.as_slice(),
            );
            if let Some(&index) = distinct.get(&key) {
                nodes.push(index);
                continue;
            }
            let entry = source.entry_name;
            // Every class calls the node's entry at the statics it was
            // enqueued and domain-checked at: classes vary only dynamic
            // dimensions.
            if let Some(name) = implementation
                .statics
                .iter()
                .find(|name| namer.values(entry, source.class_scope, name).is_some())
            {
                return Err(unsupported(format!(
                    "graph regime classes vary `{entry}`'s static dimension `{name}`"
                )));
            }
            let results = (0..source.checked.result_count())
                .map(|ordinal| {
                    let reads = source.checked.result_dimensions(ordinal).map_err(unsupported)?;
                    bound_capacity(
                        &source.checked,
                        seismic_lang::entry::TensorSlot::Result(ordinal),
                        namer,
                        entry,
                        source.class_scope,
                        dimensions,
                        &reads,
                    )
                    .map_err(|error| unsupported(format!("`{entry}`: {error}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Every buffer's bound at a class point is one walk of the
            // declaration's admitted configurations
            // (`NativeImplementation::scratch_bound`): one walk per point
            // for all of them, over every class dimension any of them reads.
            let class_reads = implementation
                .scratch
                .iter()
                .flat_map(|buffer| buffer.dimensions())
                .filter(|name| namer.values(entry, source.class_scope, name).is_some())
                .fold(Vec::<String>::new(), |mut reads, name| {
                    if !reads.contains(&name) {
                        reads.push(name);
                    }
                    reads
                });
            let (reads, fixed) = (class_reads.clone(), dimensions.clone());
            let bound = move |values: &[u64]| {
                let dimension = |name: &str| {
                    reads
                        .iter()
                        .position(|read| read == name)
                        .map(|index| values[index])
                        .or_else(|| fixed.iter().find(|(candidate, _)| candidate == name).map(|(_, value)| *value))
                };
                scratch_bound(implementation, &dimension)
            };
            let scratch = match class_reads.is_empty() {
                true => Capacity::Fixed(bound(&[]).map_err(|error| unsupported(format!("`{entry}`: {error}")))?),
                false => Capacity::Varying {
                    reads: class_reads
                        .into_iter()
                        .map(|name| ClassRead::Node { entry, scope: source.class_scope, name })
                        .collect(),
                    size: Box::new(bound),
                },
            };
            distinct.insert(key, programs.len());
            nodes.push(programs.len());
            programs.push(NodeProgram {
                entry,
                results,
                scratch,
                sync: implementation.scratch.iter().map(|buffer| buffer.sync).collect(),
            });
        }
        Ok(CapacityProgram { names: namer.names(), ports, programs, nodes })
    }
}

/// Where a class dimension's values are read in a slice: unscoped (a
/// port's class extent), or as a node of an entry and class scope reads it
/// ([`NativeGraphClassSlice::values`]).
enum ClassRead {
    Global(&'static str),
    Node { entry: &'static str, scope: Option<&'static str>, name: String },
}

impl ClassRead {
    fn values<'s>(&self, slice: &'s NativeGraphClassSlice) -> Option<&'s [u64]> {
        match self {
            Self::Global(name) => slice.global(name),
            Self::Node { entry, scope, name } => slice.values(entry, *scope, name),
        }
    }
}

/// A buffer's size over a regime's classes: a constant where no class
/// dimension reaches it, else a function of the class dimensions it reads,
/// in `reads`' order.
enum Capacity<T> {
    Fixed(T),
    Varying {
        reads: Vec<ClassRead>,
        size: Box<dyn Fn(&[u64]) -> Result<T, String> + Send + Sync>,
    },
}

/// A size that grows to cover another: a byte count, or each buffer's of a
/// node's scratch.
trait Covers: Clone {
    fn cover(self, other: Self) -> Self;
}

impl Covers for u64 {
    fn cover(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Covers for Vec<u64> {
    fn cover(self, other: Self) -> Self {
        self.into_iter().zip(other).map(|(left, right)| left.max(right)).collect()
    }
}

impl<T: Covers> Capacity<T> {
    /// Its largest size over every class of `slices`: at each combination
    /// of the values each slice gives the dimensions it reads.
    fn over(&self, slices: &[NativeGraphClassSlice]) -> Result<T, String> {
        let (reads, size) = match self {
            Self::Fixed(value) => return Ok(value.clone()),
            Self::Varying { reads, size } => (reads, size),
        };
        let mut widest: Option<T> = None;
        let mut point = vec![0; reads.len()];
        // Two slices may share class points: each is evaluated once.
        let mut seen = std::collections::HashSet::<Vec<u64>>::new();
        for slice in slices {
            let lists = reads
                .iter()
                .map(|read| read.values(slice).ok_or("class slice omits a dimension its buffer reads"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut digits = vec![0; reads.len()];
            'combinations: loop {
                for ((value, values), &digit) in point.iter_mut().zip(&lists).zip(&digits) {
                    *value = values[digit];
                }
                if slices.len() == 1 || seen.insert(point.clone()) {
                    let value = size(&point)?;
                    widest = Some(match widest {
                        None => value,
                        Some(widest) => widest.cover(value),
                    });
                }
                for (digit, values) in digits.iter_mut().zip(&lists) {
                    *digit += 1;
                    if *digit < values.len() {
                        continue 'combinations;
                    }
                    *digit = 0;
                }
                break;
            }
        }
        widest.ok_or_else(|| "graph regime has no admitted classes".to_owned())
    }
}

/// One node's buffers over a regime's classes.
struct NodeProgram {
    entry: &'static str,
    results: Vec<Capacity<u64>>,
    scratch: Capacity<Vec<u64>>,
    sync: Vec<bool>,
}

/// A template's buffers at a model's points, compiled for the slices of a
/// regime that name `names` ([`NativeGraphResourceTemplate::capacity_program`]).
pub(crate) struct CapacityProgram {
    names: Vec<(Option<&'static str>, &'static str)>,
    ports: Vec<Capacity<u64>>,
    /// Each distinct call's buffers, once however many nodes make it.
    programs: Vec<NodeProgram>,
    /// Each node's call, an index into `programs`.
    nodes: Vec<usize>,
}

impl CapacityProgram {
    /// Whether it serves slices naming `names`.
    pub(crate) fn names(&self) -> &[(Option<&'static str>, &'static str)] {
        &self.names
    }

    /// Every buffer at its capacity over the classes of `slices`, which
    /// name its dimensions.
    fn at(&self, slices: &[NativeGraphClassSlice]) -> Result<Capacities, NativeGraphMetadataError> {
        let unsupported = NativeGraphMetadataError::Unsupported;
        let valid = !slices.is_empty()
            && self.names.windows(2).all(|pair| pair[0] != pair[1])
            && slices.iter().all(|slice| {
                slice.names() == self.names
                    && slice.dimensions.iter().all(|(_, values)| !values.is_empty())
                    && slice.scoped.iter().all(|(_, _, values)| !values.is_empty())
            });
        if !valid {
            return Err(unsupported(
                "graph regime slices must name the same dimensions once, each with values".into(),
            ));
        }
        let ports = self
            .ports
            .iter()
            .map(|port| port.over(slices).map_err(unsupported))
            .collect::<Result<Vec<_>, _>>()?;
        let programs = self
            .programs
            .iter()
            .map(|program| {
                let entry = program.entry;
                let results = program
                    .results
                    .iter()
                    .map(|result| result.over(slices).map_err(|error| unsupported(format!("`{entry}`: {error}"))))
                    .collect::<Result<Vec<_>, _>>()?;
                let bytes = program.scratch.over(slices).map_err(|error| unsupported(format!("`{entry}`: {error}")))?;
                let scratch = bytes
                    .into_iter()
                    .zip(&program.sync)
                    .map(|(bytes, &sync)| seismic_runtime::native::ScratchNeed { bytes, sync })
                    .collect::<Vec<_>>();
                Ok((results, scratch))
            })
            .collect::<Result<Vec<_>, NativeGraphMetadataError>>()?;
        let (results, scratch) = self
            .nodes
            .iter()
            .map(|&index| programs[index].clone())
            .unzip();
        Ok(Capacities { ports, results, scratch })
    }
}

/// A checked tensor's canonical bytes over a regime's classes: fixed where
/// none of the dimensions its shape reads (`reads`) is a class dimension of
/// `namer`, else bound over those.
fn bound_capacity(
    checked: &generated::NativeGraphCheckedEntry,
    tensor: seismic_lang::entry::TensorSlot<'_>,
    namer: &NativeGraphClassSlice,
    entry: &'static str,
    scope: Option<&'static str>,
    dimensions: &[(String, u64)],
    reads: &[&str],
) -> Result<Capacity<u64>, String> {
    let varying = reads
        .iter()
        .copied()
        .filter(|name| namer.values(entry, scope, name).is_some())
        .collect::<Vec<_>>();
    let size = checked.bound_bytes(tensor, &borrowed(dimensions), &varying)?;
    Ok(match varying.is_empty() {
        true => Capacity::Fixed(size(&[])?),
        false => Capacity::Varying {
            reads: varying
                .into_iter()
                .map(|name| ClassRead::Node { entry, scope, name: name.to_owned() })
                .collect(),
            size: Box::new(size),
        },
    })
}

/// A node's call: its contract and implementation, its entry, backend and
/// dimensions, of which what it checks to is a pure function
/// ([`check_node`]).
#[derive(Clone, PartialEq, Eq, Hash)]
struct CallKey {
    contract: usize,
    implementation: usize,
    entry: &'static str,
    backend: BackendName,
    dimensions: Vec<(String, u64)>,
}

/// What a call of `checked` (entry `entry` on `backend`) at `dimensions`
/// checks to ([`check_node`]), refusal included, checked once per process:
/// every layer of a graph, every recording and every trace's admission at
/// another model's dimensions repeat the same calls.
fn checked_call(
    checked: &generated::NativeGraphCheckedEntry,
    entry: &'static str,
    backend: BackendName,
    dimensions: &[(String, u64)],
) -> Result<Arc<CheckedNode>, NativeGraphMetadataError> {
    type Calls = std::collections::HashMap<CallKey, Result<Arc<CheckedNode>, NativeGraphMetadataError>>;
    static CALLS: std::sync::OnceLock<std::sync::RwLock<Calls>> = std::sync::OnceLock::new();
    let key = CallKey {
        contract: checked.identity(),
        implementation: std::ptr::from_ref(checked.implementation()) as usize,
        entry,
        backend,
        dimensions: dimensions.to_vec(),
    };
    let calls = CALLS.get_or_init(Default::default);
    if let Some(call) = calls.read().expect("checked calls are never poisoned").get(&key) {
        return call.clone();
    }
    let call = check_node(checked.clone(), entry, backend, &borrowed(dimensions)).map(Arc::new);
    calls.write().expect("checked calls are never poisoned").insert(key, call.clone());
    call
}

/// A native kernel a program needs on a backend: an entry at element
/// bindings and static values. Every specialization of it (its tuning
/// choices) shares the request.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelRequest {
    pub backend: BackendName,
    pub entry: &'static str,
    pub elements: std::collections::BTreeMap<String, Element>,
    pub statics: std::collections::BTreeMap<String, u64>,
}

impl KernelRequest {
    fn new(
        backend: BackendName,
        entry: &'static str,
        elements: &[(&str, Element)],
        statics: &NativeSpecialization,
    ) -> Self {
        Self {
            backend,
            entry,
            elements: elements
                .iter()
                .map(|(name, element)| ((*name).to_owned(), *element))
                .collect(),
            statics: statics.statics().clone(),
        }
    }
}

impl fmt::Display for KernelRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements = self
            .elements
            .iter()
            .map(|(name, element)| format!("{name}={}", element.name()))
            .collect::<Vec<_>>()
            .join(",");
        let statics = self
            .statics
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        write!(
            f,
            "{} `{}` [{elements}] {{{statics}}}",
            self.backend.as_str(),
            self.entry
        )
    }
}

/// An entry bound to its element bindings: prepared on a device, or named,
/// without one, as the request preparing it makes.
pub struct BoundEntry<E> {
    elements: Vec<(&'static str, Element)>,
    cpu: Option<&'static native_cpu::CpuNativeKernels>,
    entry: std::marker::PhantomData<fn() -> E>,
}

impl<E> Clone for BoundEntry<E> {
    fn clone(&self) -> Self {
        Self {
            elements: self.elements.clone(),
            cpu: self.cpu,
            entry: std::marker::PhantomData,
        }
    }
}

impl<E: Entry> BoundEntry<E> {
    /// Every element parameter of the entry with the element it is bound
    /// to: with the entry and its static values, what the entry's native
    /// implementation on a device is a function of.
    pub fn elements(&self) -> &[(&'static str, Element)] {
        &self.elements
    }

    /// The element the entry's element parameter `name` is bound to.
    pub fn element(&self, name: &str) -> Option<Element> {
        self.elements
            .iter()
            .find(|(bound, _)| *bound == name)
            .map(|(_, element)| *element)
    }

    /// The default specialization at `statics` as `device` offers this entry.
    pub fn default_specialization(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
    ) -> Result<Option<NativeSpecialization>, CheckedBundleError> {
        generated::native_default_specialization::<E>(device, statics, &self.elements)
    }

    /// The entry's native implementation as `device` offers it at this
    /// entry's elements ([`generated::native_offered`]).
    pub fn offered(&self, device: &Device) -> Result<Option<OnDevice>, CheckedBundleError> {
        generated::native_offered::<E>(device, &self.elements)
    }

    /// Validate a stored choice for this entry's actual element bindings.
    pub fn specialization_valid(
        &self,
        device: &Device,
        specialization: &NativeSpecialization,
    ) -> Result<bool, CheckedBundleError> {
        generated::native_specialization_valid::<E>(device, specialization, &self.elements)
    }

    /// The identity of this entry's implementation at `statics` on `device`.
    pub fn digest(
        &self,
        device: &Device,
        statics: &NativeSpecialization,
    ) -> Result<String, TuneError> {
        generated::digest_native::<E>(device, statics, &self.elements, self.cpu)
    }

    pub fn prepare(
        &self,
        device: &Device,
        specialization: &NativeSpecialization,
    ) -> Result<NativeKernel<E>, LoadError> {
        generated::prepare_native::<E>(device, specialization, &self.elements, self.cpu)
    }

    pub fn request(
        &self,
        backend: BackendName,
        specialization: &NativeSpecialization,
    ) -> KernelRequest {
        KernelRequest::new(backend, E::NAME, &self.elements, specialization)
    }
}

thread_local! {
    static RECORDING: std::cell::RefCell<Option<Vec<KernelRequest>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `run`, recording the request of every kernel this thread prepares
/// meanwhile. Recordings do not nest.
pub fn record_kernel_requests<R>(run: impl FnOnce() -> R) -> (R, Vec<KernelRequest>) {
    /// Ends the recording however `run` exits.
    struct Recording;
    impl Drop for Recording {
        fn drop(&mut self) {
            RECORDING.with(|recording| recording.borrow_mut().take());
        }
    }
    RECORDING.with(|recording| {
        let mut recording = recording.borrow_mut();
        assert!(recording.is_none(), "kernel request recordings do not nest");
        *recording = Some(Vec::new());
    });
    let guard = Recording;
    let result = run();
    let requests = RECORDING
        .with(|recording| recording.borrow_mut().take())
        .expect("the recording is active until its guard drops");
    drop(guard);
    (result, requests)
}

fn record(request: impl FnOnce() -> KernelRequest) {
    RECORDING.with(|recording| {
        if let Some(requests) = recording.borrow_mut().as_mut() {
            requests.push(request());
        }
    });
}

impl NativeGraphMetadata {
    pub fn new(backend: BackendName) -> Self {
        Self {
            backend,
            inner: seismic_runtime::native::graph::NativeGraphMetadataDraft::new(backend),
            template: None,
            class_scope: None,
        }
    }

    pub fn new_template(backend: BackendName) -> Self {
        Self {
            backend,
            inner: seismic_runtime::native::graph::NativeGraphMetadataDraft::new(backend),
            template: Some(MetadataTemplateRecorder {
                ports: Vec::new(),
                nodes: Vec::new(),
            }),
            class_scope: None,
        }
    }

    /// The backend whose native declarations the graph checks.
    pub fn backend(&self) -> BackendName {
        self.backend
    }

    /// A timeline node after every node enqueued so far
    /// (`NativeGraph::timeline`).
    pub fn timeline<N: timeline::TimelineNodeKind>(&mut self, node: N) -> NativeTimelinePort<N::Role> {
        self.inner.timeline(node)
    }

    /// A span mark after every node enqueued so far, its ordinal returned
    /// (`NativeGraph::span_mark`).
    pub fn span_mark(&mut self) -> usize {
        self.inner.span_mark()
    }

    /// Tag following checked ports and nodes with a semantic class extent
    /// scope. A scoped override distinguishes repeated calls to one entry.
    pub fn set_class_scope(&mut self, scope: Option<&'static str>) {
        self.class_scope = scope;
    }

    pub fn port<D: Clone>(
        &mut self,
        element: Element,
        extents: &[D],
    ) -> Result<NativePort, NativeGraphMetadataError>
    where
        Dim: From<D>,
    {
        let extents = extents.iter().cloned().map(Dim::from).collect::<Vec<_>>();
        let values = extents.iter().map(Dim::recorded).collect::<Vec<_>>();
        let inner = self
            .inner
            .port(element.id(), &values, false, false)
            .map_err(NativeGraphMetadataError::Tensor)?;
        if let Some(template) = &mut self.template {
            template.ports.push(MetadataPortSource::Direct {
                element,
                extents: extents.iter().map(Dim::expression).collect(),
                class_extent: None,
            });
        }
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Mark one direct port extent as a graph-class dimension. Checked entry
    /// ports already carry named dimensions in their declarations.
    pub fn port_with_class_extent<D: Clone>(
        &mut self,
        element: Element,
        extents: &[D],
        extent_axis: usize,
        class_dimension: &'static str,
    ) -> Result<NativePort, NativeGraphMetadataError>
    where
        Dim: From<D>,
    {
        let port = self.port(element, extents)?;
        if let Some(template) = &mut self.template {
            let Some(MetadataPortSource::Direct { class_extent, .. }) = template.ports.last_mut()
            else {
                unreachable!("direct port was just recorded")
            };
            if extent_axis >= extents.len() {
                return Err(NativeGraphMetadataError::Unsupported(format!(
                    "graph port extent axis {extent_axis} is absent"
                )));
            }
            *class_extent = Some((extent_axis, class_dimension));
        }
        Ok(port)
    }

    fn entry_key<E: Entry>(
        &self,
        elements: &[(&str, Element)],
    ) -> Result<EntryKey, NativeGraphMetadataError> {
        let module = E::module().map_err(NativeGraphMetadataError::Bundle)?;
        Ok((
            self.backend,
            std::ptr::from_ref(module) as usize,
            E::NAME,
            elements
                .iter()
                .map(|(name, element)| ((*name).to_owned(), *element))
                .collect(),
        ))
    }

    fn bind_checked<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
    ) -> Result<generated::NativeGraphCheckedEntry, NativeGraphMetadataError> {
        let key = self.entry_key::<E>(elements)?;
        if let Some(checked) = checked_entries().read().expect("checked entries are never poisoned").get(&key) {
            return Ok(checked.clone());
        }
        let checked = generated::NativeGraphCheckedEntry::bind::<E>(self.backend, elements)
            .map_err(NativeGraphMetadataError::Bundle)?
            .map_err(NativeGraphMetadataError::Unsupported)?;
        checked_entries().write().expect("checked entries are never poisoned").insert(key, checked.clone());
        Ok(checked)
    }

    fn checked_port<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, Dim)],
        local: bool,
        owned_input: bool,
    ) -> Result<NativePort, NativeGraphMetadataError> {
        let checked = self.bind_checked::<E>(elements)?;
        let metadata = checked
            .parameter(parameter, &recorded_dimensions(dimensions))
            .map_err(NativeGraphMetadataError::Unsupported)?;
        let inner = self
            .inner
            .port(metadata.element.id(), &metadata.extents, local, owned_input)
            .map_err(NativeGraphMetadataError::Tensor)?;
        if let Some(template) = &mut self.template {
            template.ports.push(MetadataPortSource::Checked {
                port: Arc::new(CheckedPortSource {
                    checked,
                    entry_name: E::NAME,
                    class_scope: self.class_scope,
                    parameter: parameter.to_owned(),
                }),
                dimensions: dimension_exprs(dimensions),
            });
        }
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    pub fn input_for<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        self.checked_port::<E>(elements, parameter, &dims(dimensions), true, true)
    }

    pub fn local_for<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        self.checked_port::<E>(elements, parameter, &dims(dimensions), true, false)
    }

    /// An external port from a checked parameter (`NativeGraph::port_for`).
    pub fn port_for<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, NativeGraphMetadataError> {
        self.checked_port::<E>(elements, parameter, &dims(dimensions), false, false)
    }

    pub fn prewrite(&mut self, port: &NativePort) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .prewrite(port.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn enqueue<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, NativeGraphMetadataError> {
        self.enqueue_node::<E>(elements, &dims(dimensions), args, None)
    }

    /// [`Self::enqueue`] under a condition (`NativeGraph::enqueue_under`):
    /// planning holds the node as any other, its condition kept for the
    /// proof that results may share storage ([`Self::share`]).
    pub fn enqueue_under<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
        port: &NativePort,
        index: u64,
        encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, NativeGraphMetadataError> {
        let condition = seismic_runtime::native::graph::NodeCondition {
            port: port.inner,
            index,
            encoded,
        };
        self.enqueue_node::<E>(elements, &dims(dimensions), args, Some(condition))
    }

    /// `NativeGraph::share`.
    pub fn share(
        &mut self,
        other: &WorkflowTensor,
        primary: &WorkflowTensor,
    ) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .share(other.inner, primary.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    fn enqueue_node<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, Dim)],
        args: E::WorkflowArgs<'_>,
        condition: Option<seismic_runtime::native::graph::NodeCondition>,
    ) -> Result<E::WorkflowResults, NativeGraphMetadataError> {
        let node = self.checked_node::<E>(elements, &recorded_dimensions(dimensions))?;
        let encoded = E::encode_workflow(args);
        let (parameters, results, scratch, sealed) = (
            node.parameters.clone(),
            node.results.clone(),
            node.scratch.clone(),
            node.checked.sealed_node().clone(),
        );
        let pending = match condition {
            Some(condition) => self
                .inner
                .enqueue_under(encoded, parameters, results, scratch, sealed, condition),
            None => self
                .inner
                .enqueue(encoded, parameters, results, scratch, sealed),
        }
        .map_err(NativeGraphMetadataError::Call)?;
        if let Some(template) = &mut self.template {
            template.nodes.push(MetadataNodeSource {
                checked: node.checked.clone(),
                entry_name: E::NAME,
                class_scope: self.class_scope,
                dimensions: dimension_exprs(dimensions),
            });
        }
        Ok(E::decode_workflow(pending))
    }

    /// What a call of `E` at `dimensions` checks to: its shapes, its domain
    /// (a call at statics outside it is refused) and its scratch.
    fn checked_node<E: Entry>(
        &mut self,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<Arc<CheckedNode>, NativeGraphMetadataError> {
        let checked = self.bind_checked::<E>(elements)?;
        checked_call(&checked, E::NAME, self.backend, &owned_dimensions(dimensions))
    }

    pub fn export(&mut self, result: &WorkflowTensor) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .export(result.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    /// `NativeGraph::store`.
    pub fn store(
        &mut self,
        result: &WorkflowTensor,
        port: &NativePort,
    ) -> Result<(), NativeGraphMetadataError> {
        self.inner
            .store(result.inner, port.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    /// The host heap of the plan this exact topology seals to, from the
    /// lengths its entries' schemas and declarations fix.
    pub fn plan_heap(&self) -> HostHeap {
        self.inner.plan_heap()
    }

    pub fn seal(self) -> Result<NativeGraphStorageBytes, NativeGraphMetadataError> {
        self.inner
            .seal()
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn seal_with_layout(
        self,
        layout: &NativeGraphLayout,
    ) -> Result<NativeGraphStorageBytes, NativeGraphMetadataError> {
        self.inner
            .seal_with_layout(&layout.inner)
            .map_err(NativeGraphMetadataError::Workflow)
    }

    pub fn seal_template(self) -> Result<NativeGraphResourceTemplate, NativeGraphMetadataError> {
        let inner = self.inner.validated().map_err(NativeGraphMetadataError::Workflow)?;
        let recorder = self.template.ok_or_else(|| {
            NativeGraphMetadataError::Unsupported(
                "graph was not built as a resource template".into(),
            )
        })?;
        Ok(NativeGraphResourceTemplate {
            backend: self.backend,
            inner,
            recorder,
        })
    }
}


/// Each scratch buffer's bound at `dimension`
/// ([`NativeImplementation::scratch_bound`]), the declaration's admitted
/// configurations walked once per process for each value of its statics
/// ([`NativeImplementation::scratch_configurations`]): every point at
/// statics already walked is each configuration's charges, arithmetic.
fn scratch_bound(
    implementation: &'static NativeImplementation,
    dimension: &impl Fn(&str) -> Option<u64>,
) -> Result<Vec<u64>, String> {
    type Walked = std::collections::HashMap<(usize, Vec<u64>), Arc<Vec<NativeSpecialization>>>;
    static WALKED: std::sync::OnceLock<std::sync::RwLock<Walked>> = std::sync::OnceLock::new();
    let statics = implementation
        .statics
        .iter()
        .map(|name| dimension(name).ok_or_else(|| NativeSpecializationError::MissingStatic(name.clone()).to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let key = (std::ptr::from_ref(implementation) as usize, statics);
    let walked = WALKED.get_or_init(Default::default);
    let known = walked.read().expect("walked configurations are never poisoned").get(&key).cloned();
    let configurations = match known {
        Some(configurations) => configurations,
        None => {
            let statics = implementation
                .statics
                .iter()
                .zip(&key.1)
                .fold(NativeSpecialization::new(), |statics, (name, &value)| statics.with_static(name.clone(), value));
            let configurations =
                Arc::new(implementation.scratch_configurations(&statics).map_err(|error| error.to_string())?);
            walked
                .write()
                .expect("walked configurations are never poisoned")
                .insert(key, Arc::clone(&configurations));
            configurations
        }
    };
    implementation
        .scratch_bound_over(&configurations, dimension)
        .map_err(|error| error.to_string())
}

/// What a call of `checked` (entry `entry` on `backend`) at `dimensions`
/// checks to: its shapes, its domain (a call at statics outside it is
/// refused) and its scratch at those dimensions. What a recording checks of
/// every node, and a trace again at another model's dimensions
/// ([`NativeGraphResourceTemplate::admitted_at`]).
fn check_node(
    checked: generated::NativeGraphCheckedEntry,
    entry: &'static str,
    backend: BackendName,
    dimensions: &[(&str, u64)],
) -> Result<CheckedNode, NativeGraphMetadataError> {
    let (parameters, results) = checked
        .shapes(dimensions)
        .map_err(NativeGraphMetadataError::Unsupported)?;
    let implementation = checked.implementation();
    // A graph can only call an entry at statics its implementation
    // admits, so every checked call can be prepared.
    let mut statics = NativeSpecialization::new();
    for name in &implementation.statics {
        let value = dimensions
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| *value)
            .ok_or_else(|| {
                NativeGraphMetadataError::Unsupported(format!(
                    "`{}` declares `{name}` static, but the call supplies no value",
                    entry
                ))
            })?;
        statics = statics.with_static(name.clone(), value);
    }
    match implementation.admits(&statics) {
        Ok(()) => {}
        Err(NativeSpecializationError::Inadmissible) => {
            return Err(NativeGraphMetadataError::Inadmissible(
                KernelDomainViolation {
                    entry,
                    backend,
                    statics: statics
                        .statics()
                        .iter()
                        .map(|(name, value)| (name.clone(), *value))
                        .collect(),
                },
            ));
        }
        Err(error) => {
            return Err(NativeGraphMetadataError::Unsupported(format!(
                "`{}`: {error}",
                entry
            )));
        }
    }
    let scratch = scratch_bound(implementation, &|name| {
            dimensions
                .iter()
                .find(|(candidate, _)| *candidate == name)
                .map(|(_, value)| *value)
        })
        .map_err(|reason| {
            NativeGraphMetadataError::Unsupported(format!(
                "`{}` scratch bound at {dimensions:?}: {reason}",
                entry
            ))
        })?
        .into_iter()
        .zip(&implementation.scratch)
        .map(|(bytes, buffer)| seismic_runtime::native::ScratchNeed {
            bytes,
            sync: buffer.sync,
        })
        .collect();
    Ok(CheckedNode {
        parameters: parameters
            .into_iter()
            .map(|parameter| {
                parameter.map(|metadata| (metadata.element.id(), metadata.extents))
            })
            .collect(),
        results: results
            .into_iter()
            .map(|result| result.map(|metadata| (metadata.element.id(), metadata.extents)))
            .collect(),
        scratch,
        checked,
    })
}

#[derive(Clone)]
pub struct NativePort {
    inner: seismic_runtime::native::graph::NativePort,
    tensor: WorkflowTensor,
}

impl NativePort {
    pub fn tensor(&self) -> &WorkflowTensor {
        &self.tensor
    }
    pub fn tensor_mut(&mut self) -> &mut WorkflowTensor {
        &mut self.tensor
    }
}

impl NativeGraph {
    /// A timeline node after every node enqueued so far: a host gate the
    /// device waits at, or a signal the host waits for (`Timeline`). Each
    /// run binds it (`NativeGraphBindings::set_timeline`; a park,
    /// `set_park` or `set_answered`).
    pub fn timeline<N: timeline::TimelineNodeKind>(&mut self, node: N) -> NativeTimelinePort<N::Role> {
        self.inner.timeline(node)
    }

    /// A span mark after every node enqueued so far, its ordinal returned: a
    /// boundary of the plan's calls where a sampled run
    /// ([`ReadyNativeGraphRun::queue_sampled`]) stamps the device's time, so
    /// the device time between two marks is the calls' between them. It
    /// divides no graph. Every plan of one layout marks alike.
    pub fn span_mark(&mut self) -> usize {
        self.inner.span_mark()
    }

    /// Declare an external tensor descriptor from model metadata. Sealing
    /// checks each use against its generated Seismic entry contract.
    pub fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, TensorError> {
        let inner = self.inner.port(element.id(), extents)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Declare graph-owned mutable storage using a checked tensor parameter.
    /// The entry contract derives its representation and extents from the
    /// supplied dimension values; callers provide no independent shape.
    pub fn local_for<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let inner = self.inner.local_for(&kernel.inner, parameter, dimensions)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Declare a Seismic-owned host upload tensor from a checked input
    /// parameter. Its storage remains live from upload through its last use.
    pub fn input_for<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let inner = self.inner.input_for(&kernel.inner, parameter, dimensions)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Declare an external tensor from a checked parameter: each run binds
    /// it, in the layout the entry contract derives from the dimension
    /// values.
    pub fn port_for<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, CallError> {
        let inner = self.inner.port_for(&kernel.inner, parameter, dimensions)?;
        Ok(NativePort {
            tensor: WorkflowTensor {
                inner: inner.reference(),
            },
            inner,
        })
    }

    /// Mark a checked local as filled before graph execution. Seismic keeps
    /// its storage live from the start, including across earlier nodes.
    pub fn prewrite(&mut self, port: &NativePort) -> Result<(), WorkflowError> {
        self.inner.prewrite(port.inner)
    }

    pub fn enqueue<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, WorkflowError> {
        let pending = self
            .inner
            .enqueue(&kernel.inner, E::encode_workflow(args))?;
        Ok(E::decode_workflow(pending))
    }

    /// [`Self::enqueue`] under a condition on word `index` of the i32 tensor
    /// each run binds to external port `port`: the node's launches are
    /// encoded only while the word is nonzero (`LaunchEncoding::WhileSet`)
    /// or zero (`WhileClear`), as the host reads it when the run is queued,
    /// and a park before it waits only where they are.
    pub fn enqueue_under<E: Entry>(
        &mut self,
        kernel: &NativeKernel<E>,
        args: E::WorkflowArgs<'_>,
        port: &NativePort,
        index: u64,
        encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, WorkflowError> {
        let condition = seismic_runtime::native::graph::NodeCondition {
            port: port.inner,
            index,
            encoded,
        };
        let pending =
            self.inner
                .enqueue_under(&kernel.inner, E::encode_workflow(args), condition)?;
        Ok(E::decode_workflow(pending))
    }

    /// Keep a result beyond the graph run. Exported tensors receive separate
    /// storage so releasing a scratch slot does not reclaim a retained output.
    pub fn export(&mut self, result: &WorkflowTensor) -> Result<(), WorkflowError> {
        self.inner.export(result.inner)
    }

    /// Place node result `other` where node result `primary` lies, one
    /// storage for both: their nodes' conditions are on one word, under
    /// opposite encodings (`enqueue_under`), so each run has exactly one
    /// writer, and every reader reads `primary`. Sealing proves it.
    pub fn share(
        &mut self,
        other: &WorkflowTensor,
        primary: &WorkflowTensor,
    ) -> Result<(), WorkflowError> {
        self.inner.share(other.inner, primary.inner)
    }

    /// Write the node result `result` into the tensor each run binds to the
    /// external port `port`, which describes exactly that result, instead of
    /// into graph storage.
    pub fn store(
        &mut self,
        result: &WorkflowTensor,
        port: &NativePort,
    ) -> Result<(), WorkflowError> {
        self.inner.store(result.inner, port.inner)
    }

    pub fn seal(self) -> Result<NativeGraphPlan, CallError> {
        self.inner.seal().map(|inner| NativeGraphPlan {
            inner: Arc::new(inner),
        })
    }

    /// Seal per-use work whose shapes no load can enumerate ahead (rows a
    /// request brings): no slot of it instantiates a graph, so one may be
    /// made with work in flight (`seismic_runtime`'s `seal_launched`).
    pub fn seal_launched(self) -> Result<NativeGraphPlan, CallError> {
        self.inner.seal_launched().map(|inner| NativeGraphPlan {
            inner: Arc::new(inner),
        })
    }
}

#[derive(Clone)]
pub struct NativeGraphPlan {
    inner: Arc<seismic_runtime::native::graph::NativeGraphPlan>,
}

impl NativeGraphPlan {
    /// The host heap this sealed plan holds, at its vectors' capacities.
    pub fn host_heap(&self) -> HostHeap {
        self.inner.host_heap()
    }
    /// The bytes of `host_heap`.
    pub fn host_bytes(&self) -> u64 {
        self.inner.host_bytes()
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }
    pub fn output_bytes(&self) -> u64 {
        self.inner.output_bytes()
    }
    pub fn slot_storage_bytes(&self) -> u64 {
        self.inner.slot_storage_bytes()
    }
    /// Bytes of one submission's host-written inputs.
    pub fn upload_bytes(&self) -> u64 {
        self.inner.upload_bytes()
    }
    /// What recording one of its runs takes: its graphs, which every run of
    /// it replays.
    pub fn recorded(&self) -> GraphCensus {
        self.inner.recorded()
    }
    /// What its runs' graphs cost serving so far: graphs recorded and
    /// replays that updated them ([`Device::graph_replays`]).
    pub fn graph_replays(&self) -> GraphReplays {
        self.inner.graph_replays()
    }
    pub fn new_slot(&self) -> Result<NativeGraphSlot, TensorError> {
        self.inner.new_slot().map(|inner| NativeGraphSlot { inner })
    }
    /// Runs of the plan with `bindings`, on a slot of their own, as
    /// [`measure`] times them, each pass one run timed as `timing` says. A
    /// plan with timeline nodes is refused.
    pub fn point(
        &self,
        bindings: NativeGraphBindings,
        timing: GraphTiming,
    ) -> Result<GraphPoint, CallError> {
        self.inner.point(bindings.inner, timing)
    }
    pub fn new_outputs(&self) -> Result<NativeGraphOutputs, TensorError> {
        self.inner
            .new_outputs()
            .map(|inner| NativeGraphOutputs { inner })
    }
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            inner: self.inner.bindings(),
        }
    }
    pub fn bind_static(
        &self,
        fixed: &[(&NativePort, &Tensor)],
    ) -> Result<BoundNativeGraphPlan, CallError> {
        let fixed = fixed
            .iter()
            .map(|(port, tensor)| (port.inner, tensor.inner.clone()))
            .collect::<Vec<_>>();
        self.inner
            .bind_static(&fixed)
            .map(|inner| BoundNativeGraphPlan { inner })
    }
}

pub struct NativeGraphFamilyOutputSlot {
    inner: seismic_runtime::native::graph::NativeGraphFamilyOutputSlot,
}

impl NativeGraphFamilyOutputSlot {
    /// Release the slot's storage: the bytes the device is no longer
    /// charged, those of each allocation the slot alone held; one another
    /// holder shares stays charged until it releases it.
    pub fn release(self) -> u64 {
        self.inner.release()
    }

    pub fn activate(self, plan: &NativeGraphPlan) -> Result<NativeGraphOutputs, WorkflowError> {
        self.inner
            .activate(&plan.inner)
            .map(|inner| NativeGraphOutputs { inner })
    }
}

pub struct BoundNativeGraphPlan {
    inner: seismic_runtime::native::graph::BoundNativeGraphPlan,
}

impl BoundNativeGraphPlan {
    pub fn bindings(&self) -> NativeGraphBindings {
        NativeGraphBindings {
            inner: self.inner.bindings(),
        }
    }

    /// The host heap this binding holds, at its vectors' capacities.
    pub fn host_heap(&self) -> HostHeap {
        self.inner.host_heap()
    }
}

#[derive(Clone)]
pub struct NativeGraphFamily {
    inner: Arc<seismic_runtime::native::graph::NativeGraphFamily>,
}

impl NativeGraphFamily {
    /// A family of `plans` whose launch is one run of one of them, the
    /// first `decodes` those a decode step may make, kept live for each
    /// launch in flight.
    pub fn new(plans: &[NativeGraphPlan], decodes: usize) -> Result<Self, WorkflowError> {
        let plans = plans.iter().map(|plan| plan.inner.clone()).collect::<Vec<_>>();
        seismic_runtime::native::graph::NativeGraphFamily::new(&plans, decodes)
            .map(|inner| Self { inner: Arc::new(inner) })
    }
    /// A family of `plans` whose launches each make the runs of one of
    /// `launches`, a plan once for each of its runs, the first `decodes`
    /// those a decode step may make, kept live for each launch in flight:
    /// its plans of one layout
    /// share one graph, recorded as a run
    /// first needs it, within what its live slots hold ([`Self::graphs`]).
    /// A chain of `k + 2` launches in one submission further runs
    /// `chains[k]`, each plan its count of runs, beyond each launch's own.
    pub fn with_launches(
        plans: &[NativeGraphPlan],
        launches: &[Vec<NativeGraphPlan>],
        decodes: usize,
        chains: &[Vec<(NativeGraphPlan, u64)>],
    ) -> Result<Self, WorkflowError> {
        let inner = |plans: &[NativeGraphPlan]| {
            plans
                .iter()
                .map(|plan| plan.inner.clone())
                .collect::<Vec<_>>()
        };
        let launches = launches.iter().map(|runs| inner(runs)).collect::<Vec<_>>();
        let chains = chains
            .iter()
            .map(|runs| runs.iter().map(|(plan, count)| (plan.inner.clone(), *count)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        seismic_runtime::native::graph::NativeGraphFamily::with_launches(&inner(plans), &launches, decodes, &chains)
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
    }
    /// What it pins for each launch in flight and the union of its
    /// launches' graphs, which its live slots, each a
    /// launch in flight, hold of its device's graph budget
    /// ([`LaunchGraphs::held`]).
    pub fn graphs(&self) -> LaunchGraphs {
        self.inner.graphs()
    }
    /// Whether [`Self::record_decode`] records ahead (CUDA's only): elsewhere
    /// a graph is recorded at its shape's first use, a bank shrink's step
    /// included.
    pub fn records_decode_ahead(&self) -> bool {
        self.inner.records_decode_ahead()
    }
    /// Record, ahead of it, every graph its decode launch replays that is
    /// not kept, once the device's queued work has drained (CUDA's; other
    /// backends record at a shape's first use, with no wait): a step of it
    /// then records none, even the first after the bank shrank.
    pub fn record_decode(&self) -> Result<(), TensorError> {
        self.inner.record_decode()
    }
    /// What its graphs are made of: its layouts and what each launch's runs
    /// replay.
    pub fn census(&self) -> GraphFamilyCensus {
        self.inner.census()
    }
    /// What its plans' graphs cost serving so far: graphs recorded and
    /// replays that updated them, this family's alone of what its device
    /// counts ([`Device::graph_replays`]).
    pub fn graph_replays(&self) -> GraphReplays {
        self.inner.graph_replays()
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }
    /// The workspace no placement of the family's plans fits in less than.
    pub fn workspace_floor_bytes(&self) -> u64 {
        self.inner.workspace_floor_bytes()
    }
    pub fn output_bytes(&self) -> u64 {
        self.inner.output_bytes()
    }
    /// The most input bytes one of the family's runs writes.
    pub fn upload_bytes(&self) -> u64 {
        self.inner.upload_bytes()
    }
    pub fn new_output_slot(&self) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.inner
            .new_output_slot()
            .map(|inner| NativeGraphFamilyOutputSlot { inner })
    }
    /// [`Self::new_output_slot`] of outputs the host reads as they complete
    /// (`Contract::Shared`), with no copy queued behind later runs.
    pub fn new_host_output_slot(&self) -> Result<NativeGraphFamilyOutputSlot, TensorError> {
        self.inner
            .new_host_output_slot()
            .map(|inner| NativeGraphFamilyOutputSlot { inner })
    }
    /// A slot whose upload regions, of `regions` bytes each (one per run it
    /// keeps in flight at once, holding that run's input), are allocated
    /// here; activation never allocates.
    pub fn new_slot(&self, regions: &[u64]) -> Result<NativeGraphFamilySlot, TensorError> {
        self.inner
            .new_slot(regions)
            .map(|inner| NativeGraphFamilySlot { inner })
    }
    /// A slot whose workspace is the shared `arena`, with upload regions of
    /// `regions` bytes each of its own.
    pub fn new_slot_in(
        &self,
        arena: &NativeExecutionArena,
        regions: &[u64],
    ) -> Result<NativeGraphFamilySlot, WorkflowError> {
        self.inner
            .new_slot_in(&arena.inner, regions)
            .map(|inner| NativeGraphFamilySlot { inner })
    }
}

/// One device's graph workspace, shared by the family slots created in it.
#[derive(Clone)]
pub struct NativeExecutionArena {
    inner: seismic_runtime::native::graph::NativeExecutionArena,
}

impl NativeExecutionArena {
    pub fn bytes(&self) -> u64 {
        self.inner.bytes()
    }

    /// What the arena is charged: its bytes, and the record tables graph
    /// slots formed over it while they live.
    pub fn charged_bytes(&self) -> u64 {
        self.inner.charged_bytes()
    }
}

pub struct NativeGraphFamilySlot {
    inner: seismic_runtime::native::graph::NativeGraphFamilySlot,
}

impl NativeGraphFamilySlot {
    /// Release the slot's storage: the bytes the device is no longer
    /// charged, those of each allocation the slot alone held; one another
    /// holder shares stays charged until it releases it.
    pub fn release(self) -> u64 {
        self.inner.release()
    }

    pub fn activate<'a>(
        &'a mut self,
        plan: &NativeGraphPlan,
    ) -> Result<NativeGraphFamilyActive<'a>, WorkflowError> {
        self.inner
            .activate(&plan.inner)
            .map(|inner| NativeGraphFamilyActive { inner })
    }

    /// Tensors of `tensors`' elements and extents, each holding its bytes,
    /// written once into one of the slot's idle upload regions for several
    /// runs to bind as external ports. The region stays taken while any of
    /// them or a view of one lives and until the device's last use of it
    /// completes. It holds [`upload_region_bytes`] of them.
    pub fn upload(
        &mut self,
        tensors: &[(Element, &[u64], &[u8])],
    ) -> Result<Vec<Tensor>, WorkflowError> {
        let tensors = tensors
            .iter()
            .map(|&(element, extents, bytes)| (element.id(), extents, bytes))
            .collect::<Vec<_>>();
        Ok(self
            .inner
            .upload(&tensors)?
            .into_iter()
            .map(|inner| Tensor { inner })
            .collect())
    }
}

/// The bytes of the upload region [`NativeGraphFamilySlot::upload`] takes
/// for tensors of `tensors`' elements and extents.
pub fn upload_region_bytes(tensors: &[(Element, &[u64])]) -> Result<u64, WorkflowError> {
    let tensors = tensors
        .iter()
        .map(|&(element, extents)| (element.id(), extents))
        .collect::<Vec<_>>();
    seismic_runtime::native::graph::upload_region_bytes(&tensors)
}

pub struct NativeGraphFamilyActive<'a> {
    inner: seismic_runtime::native::graph::NativeGraphFamilyActive<'a>,
}

impl NativeGraphFamilyActive<'_> {
    pub fn local(&mut self, port: &NativePort) -> Option<Tensor> {
        self.inner.local(port.inner).map(|inner| Tensor { inner })
    }

    pub fn write_input(&mut self, port: &NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_input(port.inner, bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        self.inner
            .attach(bindings.inner, outputs.inner)
            .map(|inner| ReadyNativeGraphRun { inner })
    }
}

pub struct NativeGraphBindings {
    inner: seismic_runtime::native::graph::NativeGraphBindings,
}

impl NativeGraphBindings {
    pub fn set(&mut self, port: &NativePort, tensor: &Tensor) -> Result<(), WorkflowError> {
        self.inner.set(port.inner, tensor.inner.clone())
    }

    /// Bind timeline node `port` to `timeline` at `value` for this run: a
    /// gate waits until the timeline reaches the value, a signal raises it
    /// to the value.
    pub fn set_timeline<R: timeline::ValuedRole>(
        &mut self,
        port: NativeTimelinePort<R>,
        timeline: &Timeline<R>,
        value: u64,
    ) -> Result<(), WorkflowError> {
        self.inner.set_timeline(port, &timeline.host, value)
    }

    /// Bind park node `port` to `timeline`: where the device's kernels do
    /// not wait for the host themselves, a queued run that encodes the
    /// park's call takes the timeline's next park value `v`, reported by
    /// its outputs (`NativeGraphOutputs::park`): the device signals `v`
    /// once the call waits, every write before visible to the host, and
    /// goes on once the host answers (`Timeline::answer`). The timeline's
    /// values are the runtime's: no node binds it at a value.
    pub fn set_park(
        &mut self,
        port: NativeTimelinePort<timeline::Parks>,
        timeline: &Timeline<timeline::Parks>,
    ) -> Result<(), WorkflowError> {
        self.inner.set_park(port, &timeline.host)
    }

    /// Bind park node `port` as answered before the run is submitted (the
    /// words its call awaits already hold the answer): it binds no
    /// operation, and a run of only such parks is the same work each time.
    pub fn set_answered(&mut self, port: NativeTimelinePort<timeline::Parks>) -> Result<(), WorkflowError> {
        self.inner.set_answered(port)
    }

    /// Bind an unsubmitted exported output lease as a graph destination.
    /// The tensor is only exposed to the checked graph attachment.
    pub fn set_reserved_export(
        &mut self,
        port: &NativePort,
        outputs: &NativeGraphOutputs,
        result: &WorkflowTensor,
    ) -> Result<(), WorkflowError> {
        self.inner
            .set_reserved_export(port.inner, &outputs.inner, result.inner)
    }
}

pub struct NativeGraphSlot {
    inner: seismic_runtime::native::graph::NativeGraphSlot,
}

impl NativeGraphSlot {
    pub fn write_input(&mut self, port: &NativePort, bytes: &[u8]) -> Result<(), TensorError> {
        self.inner.write_input(port.inner, bytes)
    }
    pub fn attach(
        &mut self,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
    ) -> Result<ReadyNativeGraphRun<'_>, CallError> {
        self.inner
            .attach(bindings.inner, outputs.inner)
            .map(|inner| ReadyNativeGraphRun { inner })
    }
}

pub struct NativeGraphOutputs {
    inner: seismic_runtime::native::graph::NativeGraphOutputs,
}

impl NativeGraphOutputs {
    pub fn recycle(self) -> Result<NativeGraphFamilyOutputSlot, WorkflowError> {
        self.inner
            .recycle()
            .map(|inner| NativeGraphFamilyOutputSlot { inner })
    }
    pub fn exported(&self, result: &WorkflowTensor) -> Option<Tensor> {
        self.inner
            .exported_tensor(result.inner)
            .map(|inner| Tensor { inner })
    }

    /// The value park node `port` of the queued run took on its timeline
    /// (`NativeGraphBindings::set_park`); `None` where it binds nothing:
    /// the device's kernels park themselves, it was bound answered, or the
    /// run encodes none of its call's launches.
    pub fn park(&self, port: NativeTimelinePort<timeline::Parks>) -> Option<u64> {
        self.inner.park(port)
    }
}

pub struct ReadyNativeGraphRun<'a> {
    inner: seismic_runtime::native::graph::ReadyNativeGraphRun<'a>,
}

impl ReadyNativeGraphRun<'_> {
    /// Submit without waiting. The outputs can be bound into later runs at
    /// once; host reads of exported tensors wait for this run. The
    /// completion reports its outcome.
    pub fn submit(self) -> Result<(NativeGraphOutputs, NativeGraphCompletion), CallError> {
        self.inner.submit().map(|(inner, completion)| {
            (
                NativeGraphOutputs { inner },
                NativeGraphCompletion { inner: completion },
            )
        })
    }

    /// Append this run to `sequence` instead of submitting it: every run
    /// queued on a sequence is submitted as one unit of device work, in
    /// queue order. The outputs may be bound into later queued runs at
    /// once; host reads of them are valid only after the sequence is
    /// submitted.
    pub fn queue(
        self,
        sequence: &mut NativeGraphSequence,
    ) -> Result<NativeGraphOutputs, CallError> {
        self.inner
            .queue(&mut sequence.inner)
            .map(|inner| NativeGraphOutputs { inner })
    }

    /// [`Self::queue`], the run sampled for its spans' device time: each
    /// mark of its plan ([`NativeGraph::span_mark`]) stamped as the device
    /// passes it, read through the returned spans once the run's work
    /// completes ([`NativeSpans::seconds`]). Why the device's backend times
    /// no span (Metal, the CPU).
    pub fn queue_sampled(
        self,
        sequence: &mut NativeGraphSequence,
    ) -> Result<(NativeGraphOutputs, NativeSpans), CallError> {
        self.inner
            .queue_sampled(&mut sequence.inner)
            .map(|(inner, spans)| (NativeGraphOutputs { inner }, spans))
    }
}

/// Graph runs queued for one submission (one Metal command buffer, one
/// CUDA graph launch), executed in queue order as if submitted one by one.
pub struct NativeGraphSequence {
    inner: seismic_runtime::native::graph::NativeGraphSequence,
}

impl NativeGraphSequence {
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Submit every queued run without waiting; the completion reports all
    /// of them.
    pub fn submit(self) -> Result<NativeGraphCompletion, CallError> {
        self.inner
            .submit()
            .map(|inner| NativeGraphCompletion { inner })
    }
}

/// The outcome of one submitted native graph run.
#[must_use = "a native graph completion reports whether the run succeeded"]
pub struct NativeGraphCompletion {
    inner: seismic_runtime::native::graph::NativeGraphCompletion,
}

impl NativeGraphCompletion {
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }
    pub fn wait(self) -> Result<(), CallError> {
        self.inner.wait()
    }
}

/// The only workflow phase that accepts nodes. It owns symbolic producer edges
/// and no bound policy or live device resources.
pub struct WorkflowDraft {
    inner: seismic_runtime::api::kernel::WorkflowDraftAny,
}

/// A completely bound workflow. Selection, concrete output descriptors,
/// hazards, lifetimes, and allocation requirements are fixed; no capacity or
/// native resources have been acquired.
pub struct BoundWorkflow {
    inner: seismic_runtime::api::kernel::BoundWorkflowAny,
}

/// A graph-wide admitted workflow. It owns all reservations, allocations,
/// persistent leases, and access permits required for submission.
pub struct AdmittedWorkflow {
    inner: seismic_runtime::api::kernel::AdmittedWorkflowAny,
}

/// Completion owner for a submitted workflow. It retains the native execution
/// and all admitted resources until the first typed resolution completes it;
/// later result groups resolve from the completed table without resubmission.
pub struct WorkflowCompletion {
    inner: seismic_runtime::api::kernel::WorkflowCompletionAny,
}

impl WorkflowDraft {
    pub fn enqueue<E: Entry>(
        &mut self,
        kernel: &Kernel<E>,
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, WorkflowError> {
        let encoded = E::encode_workflow(args);
        let pending =
            seismic_runtime::api::kernel::enqueue(&mut self.inner, &kernel.inner, encoded)?;
        Ok(E::decode_workflow(pending))
    }

    pub fn bind(self) -> Result<BoundWorkflow, CallError> {
        seismic_runtime::api::kernel::bind_workflow(self.inner).map(|inner| BoundWorkflow { inner })
    }
}

impl BoundWorkflow {
    pub fn admit(self) -> Result<AdmittedWorkflow, CallError> {
        seismic_runtime::api::kernel::admit_workflow(self.inner)
            .map(|inner| AdmittedWorkflow { inner })
    }
}

impl AdmittedWorkflow {
    pub fn submit(self) -> Result<WorkflowCompletion, CallError> {
        seismic_runtime::api::kernel::submit_workflow(self.inner)
            .map(|inner| WorkflowCompletion { inner })
    }
}

impl WorkflowCompletion {
    pub fn resolve<E: Entry>(&self, outputs: E::WorkflowResults) -> Result<E::Results, CallError> {
        let outputs = E::workflow_outputs(outputs);
        let decoded = self.inner.resolve(outputs)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> Kernel<E> {
    pub fn feedback_report(&self) -> Option<&FeedbackReport> {
        self.inner.feedback_report()
    }

    fn prepare(
        device: &Device,
        options: PreparationOptions,
        bindings: seismic_lang::entry::ElementBindings,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        seismic_runtime::api::kernel::prepare(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            options,
        )
        .map(|inner| Self {
            inner: Arc::new(inner),
            marker: std::marker::PhantomData,
        })
        .map_err(LoadError::from_prepare)
    }

    pub fn call(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call(&self.inner, encoded)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> NativeKernel<E> {
    /// The device storage this prepared specialization holds: its
    /// scalar-result slots ([`Entry::NATIVE_SCALAR_RESULT_BYTES`]).
    pub fn scalar_result_bytes(&self) -> u64 {
        self.inner.scalar_result_bytes()
    }

    fn prepare(
        device: &Device,
        specialization: NativeSpecialization,
        bindings: seismic_lang::entry::ElementBindings,
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<Self, LoadError> {
        let module = E::module().map_err(LoadError::Bundle)?;
        let entry = E::resolve(module).map_err(LoadError::Bundle)?;
        seismic_runtime::api::kernel::prepare_native(
            module.checked(),
            entry.id(),
            bindings,
            device.inner(),
            specialization,
            cpu,
        )
        .map(|inner| Self {
            inner: Arc::new(inner),
            marker: std::marker::PhantomData,
        })
        .map_err(LoadError::from_prepare)
    }

    /// The specialization this kernel was formed under.
    pub fn specialization(&self) -> &NativeSpecialization {
        self.inner.specialization()
    }

    /// Check the launches of a call whose dimensions are `dimensions`
    /// against the formed functions and the device limits, as a call's are
    /// checked from its arguments: whether the kernel serves a shape no
    /// argument set at hand takes.
    pub fn admits(
        &self,
        dimensions: &std::collections::BTreeMap<String, u64>,
    ) -> Result<(), CallError> {
        self.inner.admits(dimensions)
    }

    /// What was formed: backend, bindings, specialization, source digest
    /// and toolchain.
    pub fn artifact(&self) -> &NativeArtifactIdentity {
        self.inner.artifact()
    }

    /// Calls cycling through `rotation`, each argument set validated and
    /// placed once with results of its own, as [`measure::measure`] times
    /// them.
    pub fn point(&self, rotation: Vec<E::Args<'_>>) -> Result<KernelPoint, CallError> {
        self.inner
            .point(rotation.into_iter().map(E::encode).collect())
    }

    pub fn call(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call_native(&self.inner, encoded)?;
        Ok(E::decode(decoded))
    }

    /// [`Self::call`] of a call whose every awaited (`&await`) word already
    /// holds its answer, written by the host before the call: it waits for
    /// nothing, so it runs on every backend, where kernels park and where a
    /// graph's park is a submission boundary alike (a graph binds such a
    /// park answered: `NativeGraphBindings::set_answered`).
    pub fn call_answered(&self, args: E::Args<'_>) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let decoded = seismic_runtime::api::kernel::call_native_answered(&self.inner, encoded)?;
        Ok(E::decode(decoded))
    }

    /// Execute into caller-owned tensor results after validating their checked
    /// representation, device, and extents. Scalar results remain returned.
    pub fn call_into(
        &self,
        args: E::Args<'_>,
        outputs: E::OutputArgs<'_>,
    ) -> Result<E::Results, CallError> {
        let encoded = E::encode(args);
        let encoded_outputs = E::encode_outputs(outputs);
        let decoded =
            seismic_runtime::api::kernel::call_native_into(&self.inner, encoded, encoded_outputs)?;
        Ok(E::decode(decoded))
    }
}

impl<E: Entry> Clone for NativeKernel<E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<E: Entry> fmt::Debug for NativeKernel<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernel")
            .field("entry", &E::NAME)
            .finish()
    }
}

impl<E: Entry> Clone for Kernel<E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<E: Entry> fmt::Debug for Kernel<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel").field("entry", &E::NAME).finish()
    }
}

/// Helpers generated code composes. Not for hand-written use.
#[doc(hidden)]
pub mod generated {
    use super::*;

    pub use std::sync::OnceLock;

    use seismic_runtime::api::kernel::EncodedScalar;
    pub use seismic_runtime::api::kernel::{
        DecodedResults, EncodedArgs, EncodedOutputs, EncodedWorkflowArgs, PendingWorkflowResults,
        WorkflowResultRef,
    };

    /// Opaque checked module token used only by generated bindings. Consumers
    /// can name the type because Rust trait implementations must, but cannot
    /// inspect or construct the compiler artifact it owns.
    pub struct Module {
        checked: seismic_lang::checked::CheckedModule,
        /// The contract of every entry and element binding graph metadata
        /// has bound in this process (`EntryContract`), each formed once:
        /// graphs bind the same few over and over, and forming one per graph
        /// makes assessment many times slower. After assessing every
        /// assessment fixture on Metal, these and their evaluated shapes
        /// are at most 42 MB of what the process holds beyond the decoded
        /// module (3 MB of it evaluated shapes), and assessing them all
        /// again adds nothing.
        entry_contracts: std::sync::Mutex<Vec<EntryContract>>,
    }

    /// What graph metadata reads of one checked entry at one element
    /// binding: its call schema and its compiled tensor shapes. The
    /// monomorphized entry they are derived from (`LogicalEntry`, its
    /// arena and semantic program) is formed to derive them and dropped:
    /// nothing else of it is read, and it is most of its size.
    struct EntryContract {
        entry: seismic_lang::ids::EntryId,
        bindings: seismic_lang::entry::ElementBindings,
        schema: Arc<seismic_lang::entry::CallSchema>,
        shapes: Arc<seismic_lang::entry::CompiledEntryShapes>,
    }

    /// The compiled shapes every module of the process retains in its
    /// cache: each memoizes the shapes it has evaluated
    /// (`CompiledEntryShapes::forget_evaluated`).
    static RETAINED_ENTRY_SHAPES: std::sync::Mutex<
        Vec<std::sync::Weak<seismic_lang::entry::CompiledEntryShapes>>,
    > = std::sync::Mutex::new(Vec::new());

    /// Forget the shapes the process's checked entries have evaluated for
    /// graph metadata, and return how many there were. Checking a graph
    /// family evaluates each entry's shapes at every class and keeps them,
    /// so that its classes and the families after it do not evaluate them
    /// again; once those are built the kept shapes serve nothing until
    /// another family is checked, which evaluates what it needs anew.
    pub fn forget_evaluated_entry_shapes() -> usize {
        let mut retained = RETAINED_ENTRY_SHAPES
            .lock()
            .expect("retained entry shapes poisoned");
        retained.retain(|shapes| shapes.strong_count() > 0);
        retained
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .map(|shapes| shapes.forget_evaluated())
            .sum()
    }

    /// Opaque proof that a generated entry name resolved in its own checked
    /// module. Generated code can pass it back to Seismic, but consumers
    /// never observe or fabricate compiler entry identifiers.
    pub struct EntryToken(seismic_lang::ids::EntryId);

    pub use seismic_compiler::feedback::InvocationParameter;
    pub use seismic_lang::expr::SymbolValue;

    pub fn invocation_scope<E: Entry>() -> Result<InvocationScope, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        let stable = module
            .checked()
            .entries()
            .iter()
            .find(|candidate| candidate.id == entry.id())
            .expect("resolved entry belongs to module")
            .stable;
        Ok(InvocationScope::for_entry(stable))
    }

    impl EntryToken {
        pub(super) fn id(&self) -> seismic_lang::ids::EntryId {
            self.0
        }
    }

    impl Module {
        fn new(checked: seismic_lang::checked::CheckedModule) -> Self {
            Self {
                checked,
                entry_contracts: std::sync::Mutex::new(Vec::new()),
            }
        }

        pub(crate) fn checked(&self) -> &seismic_lang::checked::CheckedModule {
            &self.checked
        }
        #[doc(hidden)]
        pub fn entry_named(&self, name: &str) -> Option<EntryToken> {
            self.checked.entry_named(name).map(EntryToken)
        }

        /// The call schema and compiled tensor shapes of `entry` at
        /// `bindings` (`EntryContract`), formed on first use.
        fn entry_contract(
            &self,
            entry: seismic_lang::ids::EntryId,
            bindings: &seismic_lang::entry::ElementBindings,
        ) -> Result<
            (
                Arc<seismic_lang::entry::CallSchema>,
                Arc<seismic_lang::entry::CompiledEntryShapes>,
            ),
            seismic_lang::checked::SourceError,
        > {
            let held = |contracts: &[EntryContract]| {
                contracts
                    .iter()
                    .find(|contract| contract.entry == entry && contract.bindings == *bindings)
                    .map(|contract| (contract.schema.clone(), contract.shapes.clone()))
            };
            if let Some(contract) = held(
                &self
                    .entry_contracts
                    .lock()
                    .expect("entry contract mutex poisoned"),
            ) {
                return Ok(contract);
            }
            let logical = self.checked.entry(entry, bindings)?;
            let schema = logical.shared_schema();
            let shapes = Arc::new(logical.compile_tensor_shapes());
            drop(logical);
            let mut contracts = self
                .entry_contracts
                .lock()
                .expect("entry contract mutex poisoned");
            if let Some(contract) = held(&contracts) {
                return Ok(contract);
            }
            RETAINED_ENTRY_SHAPES
                .lock()
                .expect("retained entry shapes poisoned")
                .push(Arc::downgrade(&shapes));
            contracts.push(EntryContract {
                entry,
                bindings: bindings.clone(),
                schema: schema.clone(),
                shapes: shapes.clone(),
            });
            Ok((schema, shapes))
        }
    }

    #[cfg(test)]
    mod entry_contract_tests {
        use super::*;
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};

        #[test]
        fn forms_one_contract_per_element_binding() {
            let source = SourceSet::new(vec![SourceFile {
                path: "cache.seismic".into(),
                text:
                    "fn copy[N](input: &tensor[N] A) -> tensor[N] A:\n    return to_owned(input)\n"
                        .into(),
            }]);
            let module = Module::new(check_source(source).unwrap());
            let entry = module.entry_named("copy").unwrap().id();
            let f32_binding =
                seismic_lang::entry::ElementBindings::new().bind("A", Element::f32().id());
            let (first_schema, first_shapes) = module.entry_contract(entry, &f32_binding).unwrap();
            let (schema, shapes) = module.entry_contract(entry, &f32_binding).unwrap();
            assert!(Arc::ptr_eq(&first_schema, &schema));
            assert!(Arc::ptr_eq(&first_shapes, &shapes));
            assert_eq!(module.entry_contracts.lock().unwrap().len(), 1);
        }
    }

    /// Opaque generated-code encoder. It preserves parameter order without
    /// exposing compiler argument enums or mutable ABI vectors to consumers.
    pub struct ArgsEncoder {
        inner: EncodedArgs,
    }

    impl ArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedArgs::new(),
            }
        }
        pub fn tensor(&mut self, tensor: &Tensor) {
            self.inner.push_tensor(tensor.inner().clone());
        }
        pub fn f32(&mut self, value: f32) {
            self.inner.push_scalar(EncodedScalar::from_f32(value));
        }
        pub fn f16(&mut self, value: F16) {
            self.inner.push_scalar(EncodedScalar::F16(value.to_bits()));
        }
        pub fn bf16(&mut self, value: BF16) {
            self.inner.push_scalar(EncodedScalar::BF16(value.to_bits()));
        }
        pub fn i32(&mut self, value: i32) {
            self.inner.push_scalar(EncodedScalar::I32(value));
        }
        pub fn u32(&mut self, value: u32) {
            self.inner.push_scalar(EncodedScalar::U32(value));
        }
        pub fn u64(&mut self, value: u64) {
            self.inner.push_scalar(EncodedScalar::U64(value));
        }
        pub fn bool(&mut self, value: bool) {
            self.inner.push_scalar(EncodedScalar::Bool(value));
        }
        pub fn index(&mut self, value: BigUint) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (BigUint, BigUint)) {
            self.inner.push_scalar(EncodedScalar::Range {
                start: value.0,
                end: value.1,
            });
        }
        pub fn finish(self) -> EncodedArgs {
            self.inner
        }
    }

    /// Generated-code encoder for caller-supplied native result tensors.
    pub struct OutputArgsEncoder {
        inner: EncodedOutputs,
    }

    impl OutputArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedOutputs::new(),
            }
        }
        pub fn tensor(&mut self, tensor: &mut Tensor) {
            self.inner.push_tensor(tensor.inner().clone());
        }
        pub fn finish(self) -> EncodedOutputs {
            self.inner
        }
    }

    pub struct WorkflowArgsEncoder {
        inner: EncodedWorkflowArgs,
    }

    impl WorkflowArgsEncoder {
        pub fn new() -> Self {
            Self {
                inner: EncodedWorkflowArgs::new(),
            }
        }
        pub fn shared_tensor(&mut self, tensor: WorkflowTensorRef<'_>) {
            match tensor {
                WorkflowTensorRef::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorRef::Result(result) => self.inner.push_result_tensor(result.inner),
                WorkflowTensorRef::View(view) => self
                    .inner
                    .push_result_tensor_view(view.inner, view.operations.clone(), view.expressions.clone()),
            }
        }
        pub fn mutable_tensor(&mut self, tensor: WorkflowTensorMut<'_>) {
            match tensor {
                WorkflowTensorMut::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorMut::Result(result) => self.inner.push_result_tensor(result.inner),
                WorkflowTensorMut::View(view) => self
                    .inner
                    .push_result_tensor_view(view.inner, view.operations.clone(), view.expressions.clone()),
            }
        }
        pub fn owned_tensor(&mut self, tensor: WorkflowTensorOwned<'_>) {
            match tensor.value {
                WorkflowTensorOwnedValue::External(tensor) => {
                    self.inner.push_external_tensor(tensor.inner().clone())
                }
                WorkflowTensorOwnedValue::Result(result) => {
                    self.inner.push_result_tensor(result.inner)
                }
            }
        }
        pub fn f32(&mut self, value: crate::Argument<f32>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::from_f32(value));
        }
        pub fn f16(&mut self, value: crate::Argument<F16>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::F16(value.to_bits()));
        }
        pub fn bf16(&mut self, value: crate::Argument<BF16>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::BF16(value.to_bits()));
        }
        pub fn i32(&mut self, value: crate::Argument<i32>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::I32(value));
        }
        pub fn u32(&mut self, value: crate::Argument<u32>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::U32(value));
        }
        pub fn u64(&mut self, value: crate::Argument<u64>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::U64(value));
        }
        pub fn bool(&mut self, value: crate::Argument<bool>) {
            let value = value.value();
            self.inner.push_scalar(EncodedScalar::Bool(value));
        }
        pub fn index(&mut self, value: BigUint) {
            self.inner.push_scalar(EncodedScalar::Index(value));
        }
        pub fn range(&mut self, value: (BigUint, BigUint)) {
            self.inner.push_scalar(EncodedScalar::Range {
                start: value.0,
                end: value.1,
            });
        }
        pub fn scalar_result<T>(&mut self, value: &WorkflowScalar<T>) {
            self.inner.push_result_scalar(value.inner)
        }
        pub fn finish(self) -> EncodedWorkflowArgs {
            self.inner
        }
    }

    pub fn take_workflow_tensor(results: &mut PendingWorkflowResults) -> WorkflowTensor {
        WorkflowTensor {
            inner: results.take(),
        }
    }

    pub fn take_workflow_scalar<T>(results: &mut PendingWorkflowResults) -> WorkflowScalar<T> {
        WorkflowScalar {
            inner: results.take(),
            marker: std::marker::PhantomData,
        }
    }

    pub fn workflow_tensor_ref(value: WorkflowTensor) -> WorkflowResultRef {
        value.inner
    }
    pub fn workflow_scalar_ref<T>(value: WorkflowScalar<T>) -> WorkflowResultRef {
        value.inner
    }

    pub fn prepare<E: Entry>(
        device: &Device,
        options: PreparationOptions,
        elements: &[(&str, Element)],
    ) -> Result<Kernel<E>, LoadError> {
        let bindings = elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        Kernel::prepare(device, options, bindings)
    }

    pub fn start_feedback<'device, E: Entry>(
        device: &'device Device,
        precision: PrecisionPolicy,
        options: FeedbackOptions,
        elements: &[(&str, Element)],
    ) -> Result<(FeedbackPreparation<'device, E>, Kernel<E>), LoadError> {
        let bindings = elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        );
        FeedbackPreparation::start(device, precision, options, bindings)
    }

    fn element_bindings(elements: &[(&str, Element)]) -> seismic_lang::entry::ElementBindings {
        elements.iter().fold(
            seismic_lang::entry::ElementBindings::new(),
            |bindings, (name, element)| bindings.bind(name, element.id()),
        )
    }

    pub fn bound_entry<E: Entry>(
        elements: &[(&'static str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> super::BoundEntry<E> {
        super::BoundEntry {
            elements: elements.to_vec(),
            cpu,
            entry: std::marker::PhantomData,
        }
    }

    pub fn prepare_native<E: Entry>(
        device: &Device,
        specialization: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<NativeKernel<E>, LoadError> {
        super::record(|| {
            super::KernelRequest::new(device.backend(), E::NAME, elements, specialization)
        });
        NativeKernel::prepare(
            device,
            specialization.clone(),
            element_bindings(elements),
            cpu,
        )
    }

    /// Whether preparation on a `backend` device takes `specialization` at
    /// `elements`, decided without a device by the checks preparation makes
    /// of the declaration (`seismic_runtime::native::admission`).
    pub fn admit_native<E: Entry>(
        backend: BackendName,
        specialization: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<NativeAdmission, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(seismic_runtime::native::admission(
            module.checked(),
            entry.id(),
            &element_bindings(elements),
            backend,
            specialization.clone(),
            cpu,
        ))
    }

    /// The checked native implementation of an entry for a backend, without
    /// opening a device: its declaration-level facts (statics, parameters,
    /// constraint, whether one is declared). Its expressions may read the
    /// bound elements (`operands(W)`, `form(W, NAME)`), so evaluating them,
    /// and enumerating or validating configurations where `where` reads
    /// them, takes [`native_implementation_at`].
    pub fn native_implementation_for_backend<E: Entry>(
        backend: BackendName,
    ) -> Result<Option<NativeImplementation>, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(module
            .checked()
            .native_implementation(entry.id(), backend)
            .cloned())
    }

    /// The checked native implementation of an entry for a backend at
    /// element bindings `elements`, every element read substituted: the
    /// declaration a request of these bindings prepares, evaluable without a
    /// device. Metadata-only assessment inspects the same declaration that
    /// execution later prepares.
    pub fn native_implementation_at<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
    ) -> Result<Option<&'static NativeImplementation>, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(module.checked().native_implementation_at(
            entry.id(),
            backend,
            &element_bindings(elements),
        ))
    }

    /// Resolve one metadata graph call from one checked entry and one set of
    /// element bindings. The graph still validates every argument edge and
    /// Seismic still places its storage when the graph seals.
    pub enum CheckedNativeGraphCall {
        Checked {
            implementation: NativeImplementation,
            parameters: Vec<Option<NativeTensorMetadata>>,
            results: Vec<Option<NativeTensorMetadata>>,
        },
        Unsupported(String),
    }

    /// A checked entry and its backend declaration resolved once for a graph
    /// resource template. Numeric dimensions can then be evaluated without
    /// repeating module resolution or element monomorphization.
    #[derive(Clone)]
    pub struct NativeGraphCheckedEntry {
        shapes: Arc<seismic_lang::entry::CompiledEntryShapes>,
        /// The module's declaration at these element bindings (its element
        /// reads substituted, `native_implementation_at`): bound for
        /// every node and port a graph makes, so it is borrowed, never
        /// copied.
        implementation: &'static NativeImplementation,
        /// The lengths of the vectors a sealed node calling the entry holds.
        sealed_node: seismic_runtime::native::graph::SealedNodeShape,
    }

    impl NativeGraphCheckedEntry {
        pub fn bind<E: Entry>(
            backend: BackendName,
            elements: &[(&str, Element)],
        ) -> Result<Result<Self, String>, CheckedBundleError> {
            let module = E::module()?;
            let entry = E::resolve(module)?;
            let bindings = element_bindings(elements);
            let Some(implementation) =
                module
                    .checked()
                    .native_implementation_at(entry.id(), backend, &bindings)
            else {
                return Ok(Err(format!(
                    "`{}` has no native implementation for `{}`",
                    E::NAME,
                    backend.as_str()
                )));
            };
            let (schema, shapes) = match module.entry_contract(entry.id(), &bindings) {
                Ok(contract) => contract,
                Err(error) => return Ok(Err(error.to_string())),
            };
            Ok(Ok(Self {
                shapes,
                implementation,
                sealed_node: seismic_runtime::native::graph::SealedNodeShape::of(
                    &schema,
                    implementation,
                ),
            }))
        }

        pub fn implementation(&self) -> &'static NativeImplementation {
            self.implementation
        }

        pub(crate) fn sealed_node(&self) -> &seismic_runtime::native::graph::SealedNodeShape {
            &self.sealed_node
        }

        /// The identity of the entry and element bindings this contract
        /// checks: contracts share the module's one shape table for them.
        pub(crate) fn identity(&self) -> usize {
            Arc::as_ptr(&self.shapes) as usize
        }

        /// The entry dimensions a tensor parameter's checked shape reads.
        pub fn parameter_dimensions(&self, name: &str) -> Result<Vec<&str>, String> {
            self.shapes
                .parameter_dimensions(name)
                .map_err(|error| error.to_string())
        }

        /// The entry dimensions a tensor result's checked shape reads.
        pub fn result_dimensions(&self, ordinal: usize) -> Result<Vec<&str>, String> {
            self.shapes
                .result_dimensions(ordinal)
                .map_err(|error| error.to_string())
        }

        pub fn result_count(&self) -> usize {
            self.shapes.result_count()
        }

        /// `tensor`'s canonical bytes as a function of the dimensions
        /// `varying` (in their order), every other dimension fixed at
        /// `dimensions` (`CompiledEntryShapes::bound`).
        pub(crate) fn bound_bytes(
            &self,
            tensor: seismic_lang::entry::TensorSlot<'_>,
            dimensions: &[(&str, u64)],
            varying: &[&str],
        ) -> Result<impl Fn(&[u64]) -> Result<u64, String> + Send + Sync + 'static, String> {
            let bound = self
                .shapes
                .bound(tensor, dimensions, varying)
                .map_err(|error| error.to_string())?;
            Ok(move |values: &[u64]| {
                bound
                    .at(values)
                    .map_err(|error| error.to_string())
                    .and_then(tensor_metadata)
                    .map(|metadata| metadata.canonical_bytes)
            })
        }

        pub fn result(
            &self,
            ordinal: usize,
            dimensions: &[(&str, u64)],
        ) -> Result<NativeTensorMetadata, String> {
            self.shapes
                .result_shape(ordinal, dimensions)
                .map_err(|error| error.to_string())
                .and_then(tensor_metadata)
        }

        pub fn parameter(
            &self,
            name: &str,
            dimensions: &[(&str, u64)],
        ) -> Result<NativeTensorMetadata, String> {
            self.shapes
                .parameter_shape(name, dimensions)
                .map_err(|error| error.to_string())
                .and_then(tensor_metadata)
        }

        pub fn shapes(
            &self,
            dimensions: &[(&str, u64)],
        ) -> Result<
            (
                Vec<Option<NativeTensorMetadata>>,
                Vec<Option<NativeTensorMetadata>>,
            ),
            String,
        > {
            let (parameters, results) = self
                .shapes
                .all_shapes(dimensions)
                .map_err(|error| error.to_string())?;
            let parameters = parameters
                .into_iter()
                .map(|shape| shape.map(tensor_metadata).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            let results = results
                .into_iter()
                .map(|shape| shape.map(tensor_metadata).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            Ok((parameters, results))
        }
    }

    fn tensor_metadata(
        shape: seismic_lang::entry::CheckedTensorShape,
    ) -> Result<NativeTensorMetadata, String> {
        let element = Element(shape.representation);
        let canonical_bytes = element
            .canonical_byte_len(&shape.extents)
            .map_err(|error| error.to_string())?;
        Ok(NativeTensorMetadata {
            element,
            extents: shape.extents,
            canonical_bytes,
        })
    }

    pub fn checked_native_graph_call<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<CheckedNativeGraphCall, CheckedBundleError> {
        let checked = match NativeGraphCheckedEntry::bind::<E>(backend, elements)? {
            Ok(checked) => checked,
            Err(reason) => return Ok(CheckedNativeGraphCall::Unsupported(reason)),
        };
        let (parameters, results) = match checked.shapes(dimensions) {
            Ok(shapes) => shapes,
            Err(reason) => return Ok(CheckedNativeGraphCall::Unsupported(reason)),
        };
        Ok(CheckedNativeGraphCall::Checked {
            implementation: checked.implementation.clone(),
            parameters,
            results,
        })
    }

    /// Check an entry's exact element bindings with the same checked-module
    /// operation used by native preparation, without opening a device or
    /// forming backend code. A successful check proves the semantic binding,
    /// not that a backend compiler can form the native implementation.
    pub fn checked_native_binding<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
    ) -> Result<NativeBindingCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeBindingCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        Ok(module
            .checked()
            .entry(entry.id(), &element_bindings(elements))
            .map(|_| NativeBindingCheck::AcceptedByCheckedEntry)
            .unwrap_or_else(|error| NativeBindingCheck::Unsupported(error.to_string())))
    }

    /// Resolve one native graph port from the checked entry and registry,
    /// without opening a device or preparing a native kernel. This checks a
    /// port's semantic shape and canonical storage, not graph liveness or
    /// backend formation.
    pub fn checked_native_tensor_parameter<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorParameterCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorParameterCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module
            .checked()
            .entry(entry.id(), &element_bindings(elements))
        {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        let shape = match logical.tensor_parameter_shape(parameter, dimensions) {
            Ok(shape) => shape,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        let element = Element(shape.representation);
        let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
            Ok(bytes) => bytes,
            Err(error) => return Ok(NativeTensorParameterCheck::Unsupported(error.to_string())),
        };
        Ok(NativeTensorParameterCheck::Checked(NativeTensorMetadata {
            element,
            extents: shape.extents,
            canonical_bytes,
        }))
    }

    /// Resolve checked result storage for a native entry at exact dimensions.
    /// This does not account for the result's graph lifetime or placement.
    pub fn checked_native_tensor_results<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorResultsCheck, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorResultsCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module
            .checked()
            .entry(entry.id(), &element_bindings(elements))
        {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let shapes = match logical.tensor_result_shapes(dimensions) {
            Ok(shapes) => shapes,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let mut results = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let Some(shape) = shape else {
                results.push(None);
                continue;
            };
            let element = Element(shape.representation);
            let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            results.push(Some(NativeTensorMetadata {
                element,
                extents: shape.extents,
                canonical_bytes,
            }));
        }
        Ok(NativeTensorResultsCheck::Checked(results))
    }

    /// Parameter shapes in checked schema order, for a metadata graph node.
    /// Scalar parameters have no tensor descriptor.
    pub fn checked_native_tensor_parameters<E: Entry>(
        backend: BackendName,
        elements: &[(&str, Element)],
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorResultsCheck, CheckedBundleError> {
        use seismic_lang::entry::ParameterKind;

        let module = E::module()?;
        let entry = E::resolve(module)?;
        if module
            .checked()
            .native_implementation(entry.id(), backend)
            .is_none()
        {
            return Ok(NativeTensorResultsCheck::Unsupported(format!(
                "`{}` has no native implementation for `{}`",
                E::NAME,
                backend.as_str()
            )));
        }
        let logical = match module
            .checked()
            .entry(entry.id(), &element_bindings(elements))
        {
            Ok(logical) => logical,
            Err(error) => return Ok(NativeTensorResultsCheck::Unsupported(error.to_string())),
        };
        let mut parameters = Vec::with_capacity(logical.schema().parameters().len());
        for parameter in logical.schema().parameters() {
            if !matches!(&parameter.kind, ParameterKind::Tensor { .. }) {
                parameters.push(None);
                continue;
            }
            let shape = match logical.tensor_parameter_shape(&parameter.name, dimensions) {
                Ok(shape) => shape,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            let element = Element(shape.representation);
            let canonical_bytes = match element.canonical_byte_len(&shape.extents) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(NativeTensorResultsCheck::Unsupported(error.to_string()));
                }
            };
            parameters.push(Some(NativeTensorMetadata {
                element,
                extents: shape.extents,
                canonical_bytes,
            }));
        }
        Ok(NativeTensorResultsCheck::Checked(parameters))
    }

    /// The checked native implementation of an entry for a device's
    /// backend, bound on the device as preparation binds it
    /// (`seismic_runtime::native::on_device`): every `device.NAME` it reads
    /// is the device's value, and on a CPU it holds Seismic's worker and
    /// tier parameters.
    pub fn native_implementation<E: Entry>(
        device: &Device,
    ) -> Result<Option<OnDevice>, CheckedBundleError> {
        Ok(native_implementation_for_backend::<E>(device.backend())?
            .map(|implementation| seismic_runtime::native::on_device(device.inner(), implementation)))
    }

    /// The entry's native implementation at element bindings `elements` as
    /// the opened device offers it (`seismic_runtime::native::offered`): the
    /// declaration whose configurations preparation forms, its defaults the
    /// ones preparation takes.
    pub fn native_offered<E: Entry>(
        device: &Device,
        elements: &[(&str, Element)],
    ) -> Result<Option<OnDevice>, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(seismic_runtime::native::offered(
            device.inner(),
            module.checked(),
            entry.id(),
            &element_bindings(elements),
        ))
    }

    /// The entry's default specialization at `statics` as the opened device
    /// offers its implementation, including Seismic-owned CPU workers and ISA
    /// tier. `None` when the backend has no implementation or nothing is
    /// admissible at `statics`.
    pub fn native_default_specialization<E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
    ) -> Result<Option<NativeSpecialization>, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(seismic_runtime::native::default_specialization(
            device.inner(),
            module.checked(),
            entry.id(),
            &element_bindings(elements),
            statics,
        ))
    }

    /// Validate a native choice against the opened device's complete
    /// parameter domain at element bindings `elements`, as preparation takes
    /// it: Seismic-owned CPU workers and ISA tier as the choice gives them,
    /// at their defaults where it leaves them unset.
    pub fn native_specialization_valid<E: Entry>(
        device: &Device,
        specialization: &NativeSpecialization,
        elements: &[(&str, Element)],
    ) -> Result<bool, CheckedBundleError> {
        let module = E::module()?;
        let entry = E::resolve(module)?;
        Ok(seismic_runtime::native::specialization_valid(
            device.inner(),
            module.checked(),
            entry.id(),
            &element_bindings(elements),
            specialization,
        ))
    }

    /// Digest of the entry's implementation for `device`'s backend at these
    /// bindings and static values, for keying stored tuning results.
    pub fn digest_native<E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
    ) -> Result<String, TuneError> {
        let module = E::module().map_err(|error| TuneError::Declaration(error.to_string()))?;
        let entry =
            E::resolve(module).map_err(|error| TuneError::Declaration(error.to_string()))?;
        seismic_runtime::native::tune::implementation_digest(
            device.inner(),
            module.checked(),
            entry.id(),
            &element_bindings(elements),
            statics,
            cpu,
        )
    }

    pub fn tune_native<E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
        points: &mut dyn PointSource<'_, E>,
        validation: NumericalRequirement,
        strategy: Strategy,
    ) -> Result<TuningResult, TuneError> {
        let module = E::module().map_err(|error| TuneError::Declaration(error.to_string()))?;
        let entry =
            E::resolve(module).map_err(|error| TuneError::Declaration(error.to_string()))?;
        seismic_runtime::native::tune::tune(seismic_runtime::native::tune::TuneRequest {
            device: device.inner(),
            module: module.checked(),
            entry: entry.id(),
            bindings: element_bindings(elements),
            statics: statics.clone(),
            cpu,
            points: &mut TypedPoints(points),
            validation,
            strategy,
        })
    }

    /// Open the search of the tuning unit of `E` at `statics` on `device`.
    pub fn search_native<'a, E: Entry>(
        device: &Device,
        statics: &NativeSpecialization,
        elements: &[(&str, Element)],
        cpu: Option<&'static native_cpu::CpuNativeKernels>,
        validation: NumericalRequirement,
    ) -> Result<NativeSearch<'a, E>, TuneError> {
        let module = E::module().map_err(|error| TuneError::Declaration(error.to_string()))?;
        let entry =
            E::resolve(module).map_err(|error| TuneError::Declaration(error.to_string()))?;
        Ok(NativeSearch {
            inner: seismic_runtime::native::tune::UnitSearch::open(
                seismic_runtime::native::tune::UnitRequest {
                    device: device.inner(),
                    module: module.checked(),
                    entry: entry.id(),
                    bindings: element_bindings(elements),
                    statics: statics.clone(),
                    cpu,
                    validation,
                },
            )?,
            entry: std::marker::PhantomData,
        })
    }

    fn tensor_result(inner: Arc<seismic_runtime::api::tensor::TensorInner>) -> Tensor {
        Tensor { inner }
    }

    pub fn take_tensor(results: &mut DecodedResults) -> Tensor {
        let inner = results.take_tensor();
        tensor_result(inner)
    }

    macro_rules! scalar_result {
        ($name:ident, $variant:ident, $ty:ty, $map:expr) => {
            pub fn $name(results: &mut DecodedResults) -> $ty {
                match results.take_scalar() {
                    ArgumentValue::$variant(value) => ($map)(value),
                    _ => panic!("generated result schema disagrees with prepared kernel"),
                }
            }
        };
    }
    scalar_result!(take_f32, F32, f32, |value| value);
    scalar_result!(take_f16, F16, F16, F16::from_bits);
    scalar_result!(take_bf16, BF16, BF16, BF16::from_bits);
    scalar_result!(take_i32, I32, i32, |value| value);
    scalar_result!(take_u32, U32, u32, |value| value);
    scalar_result!(take_u64, U64, u64, |value| value);
    scalar_result!(take_bool, Bool, bool, |value| value);
    scalar_result!(take_index, Index, BigUint, |value| value);

    pub fn take_range(results: &mut DecodedResults) -> (BigUint, BigUint) {
        match results.take_scalar() {
            ArgumentValue::Range { start, end } => (start, end),
            _ => panic!("generated result schema disagrees with prepared kernel"),
        }
    }

    pub fn module_from_bundle(
        cell: &'static OnceLock<Result<Module, CheckedBundleError>>,
        bytes: &'static [u8],
    ) -> Result<&'static Module, CheckedBundleError> {
        match cell
            .get_or_init(|| seismic_lang::bundle::decode_checked_bundle(bytes).map(Module::new))
        {
            Ok(module) => Ok(module),
            Err(error) => Err(error.clone()),
        }
    }
}

/// Checked dynamic integration for runtime-discovered functions.
pub mod dynamic;

/// Numerical policy values shared by every host language.
pub mod precision {
    pub use seismic_compiler::numerics::PolicyIdentity;
    pub use seismic_lang::precision::*;
}

/// The scope of [`Device::lift_table_seal`].
/// It belongs to the thread that took it.
pub struct TableSealLifted(Device, std::marker::PhantomData<*const ()>);

impl Drop for TableSealLifted {
    fn drop(&mut self) {
        self.0.inner.restore_table_seal();
    }
}
