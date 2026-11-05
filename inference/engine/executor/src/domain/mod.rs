//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Demand, Draw, DrawKind, Row, RowHistory, Select, Shaping as RowShaping, Slot,
    ValidatedTargetBatch,
};
use crate::memory::{ClaimId, HoldingClass, HoldingId, MemoryHeap};
use crate::platform::{DomainReading, DomainRole, MemoryBand};
use crate::programs::{Chain, ProgramSubmission};
use crate::programs::graph::readout;
use crate::{
    AllocatedResources, AttestedPrograms, CapacityError, Completion, ComponentLoader, ExecutionPlan,
    FeatureReader, FeatureRef, FeatureRows, FeatureSpan, GroupKey, HeadConditioning,
    HeadLaunchInputs, ImageRef, NativeGraphOutputLease, NativeGraphWorkspaceLease, Operation,
    Outcome, PoolClass, Priming, ProgramIdentity, RequestId, ResidentHead, ResidentVision,
    ResourceDomain, ResourceDomainId, ResourceKind, RowResult, SelectSpec, Selected,
    StateLaunchInputs, StateWork, TargetLaunchInputs, TargetTokens, TokenId, ValidatedHeadLaunch,
    ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchInputs,
    WorkKind,
};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::{
    Holder, InFlightState, OwnedAdvanceResolution, OwnedChainAdvance, OwnedStateAdvance,
    OwnedTailRelocation, RowDemand, SequenceState, StateCheckpoint, StateStore, StoreBindings,
    TentativeAdvance,
};
use seismic::Tensor;
mod draft;
mod family;
mod features;
mod head;
mod heap;
mod in_flight;
mod input;
mod ownership;
mod queue;
mod reconcile;
mod state;
mod target;
mod vision;

#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
#[cfg(test)]
mod domain_tests;

/// What a residency cache of `cached` bytes holds beyond the `inherited`
/// bytes of imports it shares with another component: the component's own.
/// A cache holding less than what it inherits omits one of its allocations.
fn beyond_inherited(cached: u64, inherited: u64) -> Result<u64, &'static str> {
    cached
        .checked_sub(inherited)
        .ok_or("head residency cache omits a target allocation")
}

pub use family::{
    BankHolding, BankResize, BankResized, BankStep, ElasticBank, ExpertResidency, Growth, GrowthDecided, GrowthOffer, HostResidency,
    NativeFamily,
    ProgramFamily, ResizeTimes, TableResidency, ShrinkCause, TierFill, TrafficHold, Trip,
};
pub use heap::{ClaimRefusal, DeviceHeap, Reclaimed, StatusReading, require_normal};
pub use in_flight::{
    HeadFlight, PrimingCompletionFlight, TargetContinuation, TargetFlight, VisionFlight,
};
use in_flight::{PrimingFlight, TargetWork, decode_selected};
pub use ownership::OpenRequirements;
pub use state::{MeasurementRoom, MemoryChargeReconciliation, MemoryMoment, StoreAllocation};
pub use target::TargetHostTiming;
pub(crate) use target::{
    AppendedRuns, accepted_history, cached_causal_domain, indexer_domain, indexer_history,
};

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::{Duration, Instant},
};

/// The logical owner chooses this only after inspecting every completed row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalDecision {
    pub accepted_rows: usize,
}

#[derive(Clone, Debug)]
pub enum DomainError {
    Capacity(crate::CapacityError),
    /// A required device-memory observation failed. This is not a demand
    /// deficit: reclaiming another request cannot make the reading valid.
    Blind(String),
    /// A domain the device uses is in the Reclaim band (the reading it was
    /// judged on); growth waits while reclaimable holdings are released.
    Reclaim(DomainReading),
    Input(String),
    State(magnitude_state::Error),
    Submit(crate::SubmitError),
    Device(crate::DeviceError),
    Invariant(crate::InvariantError),
    /// One sequence's weights (table rows or experts) could not be read:
    /// that sequence fails, and execution continues.
    WeightRead(crate::WeightReadError),
    /// The state growth a group needs would take the stores' committed
    /// state (`committed` with `growth` more) past the ceiling its caller
    /// set; nothing grew. The caller releases retained state and retries.
    StateCeiling {
        committed: u64,
        growth: u64,
        ceiling: u64,
    },
}

impl DomainError {
    pub(crate) fn invariant(detail: impl Into<String>) -> Self {
        Self::Invariant(crate::InvariantError {
            context: "executor domain",
            detail: detail.into(),
        })
    }
}

impl std::fmt::Display for DomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity(error) => error.fmt(f),
            Self::Blind(message) => write!(f, "device memory observation unavailable: {message}"),
            Self::Reclaim(reading) => Reclaimed(reading).fmt(f),
            Self::Input(message) => f.write_str(message),
            Self::State(error) => error.fmt(f),
            Self::Submit(error) => error.fmt(f),
            Self::Device(error) => error.fmt(f),
            Self::Invariant(error) => error.fmt(f),
            Self::WeightRead(error) => error.fmt(f),
            Self::StateCeiling {
                committed,
                growth,
                ceiling,
            } => write!(
                f,
                "state growth of {growth} bytes beside {committed} committed exceeds the {ceiling}-byte state ceiling"
            ),
        }
    }
}

impl std::error::Error for DomainError {}
impl From<String> for DomainError {
    fn from(message: String) -> Self {
        Self::Input(message)
    }
}
impl From<&str> for DomainError {
    fn from(message: &str) -> Self {
        Self::Input(message.into())
    }
}
impl From<crate::CapacityError> for DomainError {
    fn from(error: crate::CapacityError) -> Self {
        Self::Capacity(error)
    }
}
impl From<crate::SubmitError> for DomainError {
    fn from(error: crate::SubmitError) -> Self {
        match error {
            crate::SubmitError::Device(device) => Self::Device(device),
            crate::SubmitError::Invariant(invariant) => Self::Invariant(invariant),
        }
    }
}
impl From<ClaimRefusal> for DomainError {
    fn from(refusal: ClaimRefusal) -> Self {
        match refusal {
            ClaimRefusal::Blind(error) => Self::Blind(error.to_string()),
            ClaimRefusal::Reclaim(reading) => Self::Reclaim(reading),
            ClaimRefusal::Deficit {
                role,
                required,
                available,
                ..
            } => Self::Capacity(CapacityError {
                resource: match role {
                    DomainRole::Allocation => ResourceKind::DeviceMemory,
                    DomainRole::Staging => ResourceKind::HostStaging,
                },
                required,
                available,
            }),
            ClaimRefusal::Accounting(error) => {
                Self::invariant(format!("memory accounting: {error:?}"))
            }
        }
    }
}

impl From<magnitude_state::Error> for DomainError {
    fn from(error: magnitude_state::Error) -> Self {
        match error {
            magnitude_state::Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity {
                    required,
                    available,
                },
            )) => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: u64::try_from(&required).unwrap_or(u64::MAX),
                available,
            }),
            magnitude_state::Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationFailed(_),
            )) => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: 1,
                available: 0,
            }),
            magnitude_state::Error::Capacity {
                required,
                available_bytes,
            } => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::StateRows,
                required,
                available: available_bytes,
            }),
            magnitude_state::Error::BanksExhausted { .. } => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::RecurrentBanks,
                required: 1,
                available: 0,
            }),
            other => Self::State(other),
        }
    }
}

/// The right to change the stores' slab bindings: growth, residency, shrink,
/// compaction, tail relocation and idle release. The domain hands out exactly
/// one with itself; it is never cloned. Submitting a group moves it into the
/// flight and finishing the flight returns it, so no binding changes while
/// work is on the device.
pub struct StateBindings {
    target: StoreBindings,
    head: Option<StoreBindings>,
}

impl StateBindings {
    /// The binding right of the target store, or of the head store.
    fn store(&mut self, head: bool) -> Result<&mut StoreBindings, DomainError> {
        if head {
            self.head
                .as_mut()
                .ok_or_else(|| DomainError::invariant("head state growth without a head store"))
        } else {
            Ok(&mut self.target)
        }
    }
}

/// A group that did not become a flight.
pub enum SubmitFailure {
    /// Refused before anything reached the device; accepted state is
    /// unchanged and the bindings return.
    Refused(DomainError, StateBindings),
    /// The device or a domain invariant failed. The bindings are gone, so no
    /// binding can change again.
    Failed(DomainError),
}

impl SubmitFailure {
    pub fn error(&self) -> &DomainError {
        match self {
            Self::Refused(error, _) | Self::Failed(error) => error,
        }
    }
}

/// Why a submission produced no flight, before the bindings are attached.
enum Unsubmitted {
    Refused(DomainError),
    Failed(DomainError),
}

impl Unsubmitted {
    fn with(self, bindings: StateBindings) -> SubmitFailure {
        match self {
            Self::Refused(error) => SubmitFailure::Refused(error, bindings),
            Self::Failed(error) => SubmitFailure::Failed(error),
        }
    }
}

/// Whether `store`'s free pages hold `demands` without growth: in every
/// stored domain, the page-rounded rows the advances append beyond their last
/// pages.
fn holds(store: &StateStore, demands: &[RowDemand]) -> Result<(), CapacityError> {
    for domain in store.history_domains() {
        let required = demands
            .iter()
            .filter(|demand| demand.domain == domain)
            .map(|demand| demand.rows)
            .sum::<usize>();
        let available = store.free_rows(domain);
        if available < required {
            return Err(CapacityError {
                resource: ResourceKind::StateRows,
                required: required as u64,
                available: available as u64,
            });
        }
    }
    Ok(())
}

/// One request's immutable numerical result and its still-owned physical
/// transaction. Cloning the view never clones the transaction.
pub struct PendingOperationOutcome {
    request: RequestId,
    outcome: Outcome,
    advance: Option<Advance>,
    /// A prompt chunk's drafter entry, committed whole with the chunk.
    primed: Option<OwnedStateAdvance>,
    rows: usize,
    committed_rows: usize,
    kind: WorkKind,
    physical_duration: Duration,
    /// Whether its duration is its launches' alone ([`LaunchEpoch`]): no
    /// graph was recorded, nor any expert bank resize ran, within it.
    timed: bool,
    image: Option<ImageRef>,
}

/// What a launch's duration is its own across: the device's recorded graphs
/// (a recording is paid on the host within the duration) and the expert
/// bank's resizes begun (a resize holds the thread at the boundaries it runs
/// across, and a growth's new slots or a shrink's evictions change what the
/// launch misses). A launch is timed only where the epoch at its finish is
/// the one at its submission, none while a resize runs: a resize changes no
/// launch cost the draft width is chosen by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaunchEpoch {
    graphs: u64,
    resizes: u64,
}

/// The state an outcome publishes when reconciled.
enum Advance {
    /// Its step's.
    Step(OwnedStateAdvance),
    /// None of its own: a chunk of a prompt group's chain before its last,
    /// whose rows the chain publishes with the last chunk's.
    Chained,
    /// The last chunk's: the whole chain.
    Chain(OwnedChainAdvance),
}

impl PendingOperationOutcome {
    pub fn request(&self) -> RequestId {
        self.request
    }
    pub fn outcome(&self) -> &Outcome {
        &self.outcome
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn kind(&self) -> WorkKind {
        self.kind
    }
    pub fn physical_duration(&self) -> Duration {
        self.physical_duration
    }
    /// What the launches took, where the duration is theirs alone
    /// ([`LaunchEpoch`]).
    pub fn launch_duration(&self) -> Option<Duration> {
        self.timed.then_some(self.physical_duration)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationLane {
    Target,
    Head,
    Vision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainRequirements {
    lane: ReservationLane,
    pool: PoolClass,
    /// Each advance's page demand in its store (see
    /// `SequenceState::demands`): rows a history appends in place need none.
    state_demand: Vec<RowDemand>,
    successor_banks: usize,
    /// Head page demand (and one successor bank each) of the drafter entries
    /// the target group's prompt chunks prime.
    priming_demand: Vec<RowDemand>,
    priming_banks: usize,
    /// The logits rows a target launch projects (`projected_rows`): the
    /// readout slot class it takes.
    projected: usize,
    /// The operations claim the front queued step: nothing is reserved.
    claim: bool,
}

impl DomainRequirements {
    pub fn lane(&self) -> ReservationLane {
        self.lane
    }
    pub fn pool(&self) -> PoolClass {
        self.pool
    }
    pub fn successor_banks(&self) -> usize {
        self.successor_banks
    }
}

/// Capacity reserved for one operation group. The caller keeps the
/// operations and submits them with these resources.
pub struct DomainReservation {
    requirements: DomainRequirements,
    resources: ReservedResources,
}

/// The logits rows a target launch of `operations` projects: a forward's
/// rows that demand logits (decode and verification rows carry the demand,
/// a prompt chunk's only its final row), else the rows it selects. Its
/// readout slot is of the class that holds them
/// (`AllocatedResources::target_readout_graph`).
pub(super) fn projected_rows(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| match operation {
            Operation::Forward {
                kind,
                tokens,
                demand,
                select,
                ..
            } => match demand.contains(Demand::LOGITS) {
                true => match kind {
                    WorkKind::Decode | WorkKind::Verify => tokens.len(),
                    WorkKind::Prefill | WorkKind::Replay => 1,
                },
                false => select.len(),
            },
            Operation::Head { .. } | Operation::Encode { .. } => 0,
        })
        .sum()
}

/// The patch rows of the class a vision reservation encodes in.
fn image_patch_rows(pool: PoolClass) -> Result<u64, CapacityError> {
    match pool {
        PoolClass::Vision { patch_rows } => Ok(patch_rows as u64),
        _ => Err(CapacityError {
            resource: ResourceKind::Output,
            required: 1,
            available: 0,
        }),
    }
}

pub enum ReservedResources {
    Target(TargetGraphReservation),
    Head(
        NativeGraphWorkspaceLease,
        NativeGraphOutputLease,
        HeadReservation,
    ),
    Vision(NativeGraphWorkspaceLease, NativeGraphOutputLease),
}

/// The numerical state a head reservation moves into its flight, one entry
/// per operation.
pub enum HeadReservation {
    /// Advances of the head's own store.
    Advances(Vec<OwnedStateAdvance>),
    /// For a head that keeps no history: each request's accepted target
    /// state. The flight reads its history and returns it unchanged at
    /// physical completion; while it is held the request has no other target
    /// work, checkpoint or close, exactly as for an in-flight advance.
    TargetReads(Vec<SequenceState>),
}

/// Complete target capacity is owned before launch construction. Decoder
/// graphs include checked recurrent state copies; readout has its own lease.
pub enum TargetGraphReservation {
    Launch(TargetLaunchReservation),
    /// The operations claim the front queued step's slots, in operation
    /// order.
    Claim(Vec<usize>),
}

pub struct TargetLaunchReservation {
    boundary: Option<FeatureRef>,
    advances: Vec<OwnedStateAdvance>,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: NativeGraphOutputLease,
    /// The spare output slot of the chain the launch heads
    /// (`TargetProgram::submit`).
    spare_output: NativeGraphOutputLease,
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: NativeGraphOutputLease,
    priming: Option<PrimingReservation>,
}

/// A lone prompt chunk's drafter entry: its head advance and draft launch
/// leases.
pub struct PrimingReservation {
    advance: OwnedStateAdvance,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_output: NativeGraphOutputLease,
}

impl DomainReservation {
    pub fn requirements(&self) -> &DomainRequirements {
        &self.requirements
    }
    pub fn into_resources(self) -> ReservedResources {
        self.resources
    }
}

struct InputImage {
    image: ImageRef,
    features: Option<FeatureRef>,
}

/// A request's prepared input, owned from admission until the request
/// closes. Residency (numerical state in this domain) comes and goes within
/// that lifetime; encoded features are released with it and re-encoded.
struct RequestInput {
    input: PreparedModelInput,
    images: BTreeMap<String, InputImage>,
    resident: bool,
    boundary: Option<FeatureRef>,
    /// The tokens of the resident sequence.
    tokens: input::TokenHistory,
}

/// Everything needed to make a request resident at a position without
/// recomputing the rows before it: both numerical sequence lanes and the
/// features of the media spans that straddle the position. It holds no
/// request's input: which path it represents is the prefix cache's record.
pub struct ResumeState {
    target: StateCheckpoint,
    head: Option<StateCheckpoint>,
    features: BTreeMap<String, FeatureRef>,
    boundary: Option<FeatureRef>,
    /// The sequence's tokens before its position.
    tokens: Rc<[crate::TokenId]>,
}

impl ResumeState {
    pub fn position(&self) -> usize {
        self.target.position()
    }
}

/// Native executor with no erased stage or executor type parameters. The
/// accepted map is vacant while that request's advance is in flight.
pub struct ExecutorDomain<F: ProgramFamily = NativeFamily> {
    execution: Rc<ExecutionPlan>,
    definition: Rc<ModelDefinition>,
    resources: AllocatedResources,
    domain: ResourceDomain,
    /// The device's heap, carried over from the startup claims.
    memory: RefCell<DeviceHeap>,
    static_holding: Option<HoldingId>,
    optional_holding: Option<HoldingId>,
    target_store: Rc<StateStore>,
    head_store: Option<Rc<StateStore>>,
    family: F,
    head_loader: Option<ComponentLoader<ResidentHead>>,
    vision_loader: Option<ComponentLoader<ResidentVision>>,
    target: BTreeMap<RequestId, SequenceState>,
    /// The steps queued behind the last flight: the rest of a chain, with
    /// the chain under way, or one successor, with their drafter primings
    /// (`queue`). They are the only work that holds bindings between
    /// flights, so every binding change resolves them first.
    queue: queue::Queue<F>,
    head: BTreeMap<RequestId, SequenceState>,
    input: BTreeMap<RequestId, RequestInput>,
    /// Group identities of the target, head and encoder executables.
    lane_identities: [ProgramIdentity; 3],
    /// When the last target selection was read to the host.
    selection_read: Option<Instant>,
    target_timing: Option<TargetHostTiming>,
    /// Log each finished target step's host timing (read once at start).
    trace_host_steps: bool,
    /// The owner's planned successors of the group it submits next: the
    /// operations known to follow it before it finishes (a prompt's next
    /// prefill chunks). Consumed by that group's provisioning and the steps
    /// queued behind it (`behind`).
    planned: Vec<Operation>,
    /// The decode seconds owed each second the planned prompt chunks run,
    /// while decode waits behind them (`plan_prompt`).
    decode_owed: Option<f64>,
    /// The places a prompt's chunks are worth beyond what memory gave its
    /// group, and the slots the bank shrinks to for them (`grant_group`).
    places_want: Option<state::PlacesWant>,
    /// The service's measurement job's claim beside the model, which the
    /// bank's shrink makes room for (`ExecutorDomain::measurement_room`).
    measurement_claim: Option<state::MeasurementClaim>,
    /// The storage kind of the expert bank's last growth step: the other's
    /// step comes first next (`ExecutorDomain::expert_bank_growths`).
    grown_last: Option<crate::planning::BankTier>,
    /// Identity of the next target flight.
    next_flight: u64,
    host_turn: Option<in_flight::HostTurn>,
    /// Sequences a step of which could not read its weights: every later
    /// outcome of theirs fails (`Outcome::WeightReadFailed`) until they
    /// close.
    poisoned: BTreeMap<RequestId, crate::WeightReadError>,
    /// What the load measures of its experts once its warm-up ended
    /// ([`Self::measure_experts`]); none before, so a warm-up's steps are
    /// not counted.
    experts: Option<ExpertMeasures>,
    /// The heap claim of the expert bank's running growth.
    bank_claim: Option<ClaimId>,
    /// What expert bank resizes held the submitting thread.
    resize_held: ResizeHeld,
    /// What the rest of the host does to memory, read by read, and what
    /// sizes the expert bank's resizes from it.
    pace: RefCell<MemoryPace>,
}

/// The resident decode launch of a number of rows at a depth,
/// `(rows, depth)`, as the load's assessment estimates it on this executor
/// (`assessment::step_seconds`, a row a sequence).
pub type ResidentStep = dyn Fn(u32, u32) -> Result<f64, crate::assessment::AssessmentError>;

/// Where a streamed load stores each miss record as it is due: in place of
/// what the load stored last, within what every load stored
/// (`MissRecord::restored`), under the store's lock.
pub type MissStore = dyn Fn(&crate::assessment::MissRecord) -> Result<(), MissStoreError>;

/// What every streamed load prices its experts' decisions on, each input in
/// its state (`assessment::Priced`): a decision whose input is pending runs
/// its policy that needs no cost until the measurement job stores it and the
/// load prices again ([`ExecutorDomain::measure_experts`]). Each record is
/// the load's own as it is due (it prices the bank's growth), and is stored
/// merged onto earlier loads' where the host keeps them.
pub struct ExpertMeasures {
    /// Each streamed layer's lines on this executor, the host's the bank
    /// holds and the GPU's its kernels' costs price
    /// (`assessment::ExpertLines::priced`): σ's residuals and the records
    /// are taken along them, and the route's policy words are written from
    /// them. Until measured, every miss is served on the GPU's path, no
    /// record is due, and nothing grows.
    pub lines: crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>,
    /// Where the records are stored; none where the host keeps no store or
    /// the model has no key to keep them under (reported where the store is
    /// built), and they live for the load alone.
    pub store: Option<Box<MissStore>>,
    /// What the bank's growth is priced on: until measured, nothing grows.
    pub resident: crate::assessment::Priced<Box<ResidentStep>>,
    /// What a routed pass over each streamed layer's experts costs on this
    /// executor, which prices a prompt group's compute by its size
    /// (`GroupReads::grant`): until measured, places weigh reads alone.
    pub pass: crate::assessment::Priced<crate::assessment::ExpertPass>,
    /// Each bank group's shrink move line on this executor, at its own
    /// record form and moves (`MoveCosts`), which pace a shrink to a decode
    /// step: until measured, a shrink takes what memory is short of, its
    /// deficit, in one resize.
    pub moves: crate::assessment::Priced<crate::assessment::MoveLines>,
    /// What a host tier's copy costs (`assessment::TierCopyCosts`): the
    /// blocks it takes over the link, the link's rate, and the device time
    /// it takes from work beside it; none without a host tier. Until
    /// measured, a copy takes a list's whole grid and a promotion's copy is
    /// priced at nothing.
    pub tier_copy: crate::assessment::Priced<Option<crate::assessment::TierCopyCosts>>,
    /// Each bank group's convert line in a decode barrier
    /// (`assessment::ConvertCosts`), none beside a host tier: until
    /// measured, the host lands every miss.
    pub convert: crate::assessment::Priced<Option<std::sync::Arc<[crate::assessment::RecordLine]>>>,
}

/// Why a streamed load's miss record was not stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MissStoreError {
    /// The store refused the record.
    Store(seismic::measure::StoreError),
    /// The load's record does not merge onto the stored one.
    Merge(crate::assessment::MissMergeError),
}

impl std::fmt::Display for MissStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "the miss record could not be stored: {error}"),
            Self::Merge(error) => write!(formatter, "the miss record does not merge onto the stored one: {error}"),
        }
    }
}

impl std::error::Error for MissStoreError {}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Each streamed layer's lines as the bank's load measured them
    /// (`ElasticBank::lines`); `None` of a load without a bank.
    pub fn expert_bank_lines(&self) -> Option<std::sync::Arc<[crate::assessment::BankLines]>> {
        self.family.expert_bank().map(ElasticBank::lines)
    }

    /// What the load's graphs cost serving so far: graphs recorded and
    /// replays that updated a graph's parameters to the storage they ran on,
    /// of its own plans beside other loads on the device
    /// (`seismic::NativeGraphFamily::graph_replays`).
    pub fn graph_replays(&self) -> seismic::GraphReplays {
        self.family.graph_replays()
    }

    /// The expert bank's resizes so far ([`ExpertResizes`]).
    pub fn expert_resizes(&self) -> ExpertResizes {
        ExpertResizes { begun: self.resize_held.begun, running: self.resize_held.runs() }
    }

    /// The device's graph budget (`seismic::Device::graph_budget`) and what
    /// it costs in memory: what the families' live slots hold recorded.
    pub fn graph_budget(&self) -> (seismic::GraphCensus, seismic::GraphBytes) {
        let device = self.domain.device();
        let budget = device.graph_budget();
        let bytes = device.graph_cost().bytes(budget).expect("a device's graph budget's memory fits u64");
        (budget, bytes)
    }

    /// The load's expert bank, as planned; `None` of a load whose experts
    /// are held whole.
    pub fn expert_bank_plan(&self) -> Option<&crate::ExpertBankPlan> {
        self.execution.load().expert_bank()
    }

    /// What the load's weight cap bounds, by the owner of each byte
    /// (`ModelLoadPlan::weight_owners`).
    pub fn weight_owners(&self) -> Result<Vec<(&'static str, u64)>, String> {
        self.execution.load().weight_owners()
    }

    /// From now on, measure what the load's decode does of its experts by
    /// `measures`, in place of what it measured by before: its warm-up has
    /// ended, or the executor's stored costs have changed since.
    pub fn measure_experts(&mut self, measures: ExpertMeasures) -> Result<(), DomainError> {
        tracing::info!(
            records = match measures.store {
                None => "kept for the load alone",
                Some(_) => "stored",
            },
            lines = measures.lines.why().unwrap_or("measured"),
            resident = measures.resident.why().unwrap_or("measured"),
            pass = measures.pass.why().unwrap_or("measured"),
            moves = measures.moves.why().unwrap_or("measured"),
            tier_copy = measures.tier_copy.why().unwrap_or("measured"),
            convert = measures.convert.why().unwrap_or("measured"),
            "the experts are priced"
        );
        if let Some(bank) = self.family.expert_bank() {
            bank.price(measures.lines.measured().cloned(), measures.pass.measured().cloned());
            if let Some(Some(costs)) = measures.tier_copy.measured() {
                bank.tier_copy(costs)?;
            }
            if let Some(Some(lines)) = measures.convert.measured() {
                bank.device_convert(lines);
            }
        }
        self.experts = Some(measures);
        Ok(())
    }

    /// Price every miss to land on the device from now on.
    #[cfg(test)]
    pub(crate) fn prefer_device_landing(&self) {
        if let Some(bank) = self.family.expert_bank() {
            bank.prefer_device_landing();
        }
    }

    /// The landings the device has converted.
    #[cfg(test)]
    pub(crate) fn device_landed(&self) -> u64 {
        self.family.expert_bank().map_or(0, |bank| bank.device_landed())
    }

    /// Each streamed layer's lines in their state; none before
    /// [`Self::measure_experts`].
    pub fn expert_lines(&self) -> Option<&crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>> {
        self.experts.as_ref().map(|measures| &measures.lines)
    }

    /// What a routed pass over each streamed layer's experts costs, in its
    /// state; none before [`Self::measure_experts`].
    pub fn expert_pass(&self) -> Option<&crate::assessment::Priced<crate::assessment::ExpertPass>> {
        self.experts.as_ref().map(|measures| &measures.pass)
    }

    /// The epoch a launch submitted now is timed across ([`LaunchEpoch`]);
    /// none while an expert bank resize runs.
    fn launch_epoch(&self) -> Option<LaunchEpoch> {
        (!self.resize_held.runs()).then(|| LaunchEpoch {
            graphs: self.family.graph_replays().recorded,
            resizes: self.resize_held.begun,
        })
    }

    /// Whether a launch submitted at `epoch` finishing now was timed alone.
    fn timed(&self, epoch: Option<LaunchEpoch>) -> bool {
        epoch.is_some() && self.launch_epoch() == epoch
    }
}

/// How a completed step's misses are recorded by a load measuring its
/// experts by `experts`: not before [`ExecutorDomain::measure_experts`]; then
/// by each layer's lines where the load has them, which σ's residuals need.
fn recording(experts: Option<&ExpertMeasures>) -> Option<crate::experts::Recording> {
    experts.map(|measures| crate::experts::Recording {
        lines: measures.lines.measured().cloned(),
    })
}

/// Store a record due where a load measuring its experts by `experts` keeps
/// them. A record the store refuses is reported, as a cache write that fails
/// is: the load prices on its own records all the same, and only later loads
/// lose it.
fn publish_misses(experts: Option<&ExpertMeasures>, record: Option<std::sync::Arc<crate::assessment::MissRecord>>) {
    let (Some(record), Some(ExpertMeasures { store: Some(store), .. })) = (record, experts) else {
        return;
    };
    if let Err(error) = store(&record) {
        tracing::warn!("{error}");
    }
}

/// The expert bank's resizes: how many began so far, and whether one runs
/// now. A reading equal across a span with none running at either end saw
/// no resize in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertResizes {
    pub begun: u64,
    pub running: bool,
}

/// The time expert bank resizes held the submitting thread at flight
/// boundaries: the running resize's so far (none while none runs), and each
/// holding's last finished one's, which is what its growth step costs the
/// decode (none before one of it finished).
#[derive(Default)]
struct ResizeHeld {
    running: Option<f64>,
    last: std::collections::HashMap<BankHolding, f64>,
    /// The resizes begun.
    begun: u64,
}

impl ResizeHeld {
    /// A resize began, holding the thread since `started`.
    fn begin(&mut self, started: Instant) {
        self.running = Some(started.elapsed().as_secs_f64());
        self.begun += 1;
    }

    /// The running resize, if one runs, held the thread since `started`.
    fn hold(&mut self, started: Instant) {
        if let Some(running) = &mut self.running {
            *running += started.elapsed().as_secs_f64();
        }
    }

    /// The running resize, of `holding`, finished, holding the thread since
    /// `started`: the seconds it held the thread in all.
    fn finish(&mut self, started: Instant, holding: BankHolding) -> f64 {
        self.hold(started);
        let held = self.running.take().expect("a finished resize began");
        self.last.insert(holding, held);
        held
    }

    /// Whether a resize runs.
    fn runs(&self) -> bool {
        self.running.is_some()
    }
}

/// What the rest of the host does to each memory domain, measured from the
/// owner's memory probes (`ExecutorDomain::probe_memory`, one every reading
/// interval and one after each release while memory is short), and what
/// sizes the expert bank's resizes from them.
///
/// Memory available to the process is its headroom and its own charge. A
/// release drops the charge at once; the host shows it in headroom only later
/// (a released Metal buffer's pages stay wired until the GPU driver unwires
/// them). The process knows its own releases exactly, and the readings say
/// when the host shows them: available memory, counting what is unseen
/// ([`DomainPace::unseen`]), rises only as a release shows, and falls only as
/// the host takes. So no lag makes a release answer the same pressure twice,
/// on any OS: a shrink covers what the reading is below the planning line,
/// plus what the host takes until the next shrink takes effect, less what is
/// unseen and what the process released since the reading. What that covers
/// is awaited, not released again ([`Shortfall::Covered`]). A span across a
/// fall of the charge measures no taking.
///
/// What the host takes is the rate of available memory's fall over the latest
/// span of at least one reading interval (a probe right after a release spans
/// too little to tell a rate from the reading's own noise).
/// The next shrink takes effect after this one's latency, the wait for the
/// reading after it (at most the reading interval) and its own latency, each
/// latency the last finished release's, from its decision to its effect.
/// Before the first, the reading interval alone: a release drops only the
/// chunks it releases, 1–2 ms for a 38 MB slot of Qwen3.6-35B-A3B Q4 (M4
/// Pro), against the 100 ms interval.
///
/// A growth needs room above the planning line for one growth step plus the
/// reading's fluctuation in steady state ([`Fluctuation`]). A step of `s`
/// bytes decided at a reading of headroom `h` leaves `h − s`; nothing checks
/// it again before the next reading, and in steady state available memory
/// falls between consecutive readings by at most the fluctuation `f`. Reclaim
/// is headroom at or below the planning line `p`, so the step never takes the
/// domain there iff `h − s − f > p`: `s + f` below the ceiling `h − p`. Under
/// held pressure growth so stops short of the line by the reading's own
/// fluctuation and does not re-enter Reclaim; once the pressure lifts, the
/// ceiling rises by what it freed and the bank grows while a step and the
/// fluctuation fit, to every expert where memory allows.
struct MemoryPace {
    domains: BTreeMap<seismic::StoreId, DomainPace>,
    /// The interval the owner reads memory at, as its probes say.
    reading_interval: Duration,
    /// The last finished release's latency; zero before the first.
    latency: Duration,
}

/// One domain's pace, as of its latest probe.
struct DomainPace {
    role: DomainRole,
    headroom: u64,
    planning: u64,
    /// The emergency reserve (`DomainThresholds::emergency_bytes`): the
    /// headroom a load must not run into.
    emergency: u64,
    ceiling: u64,
    charged: u64,
    /// The reading's band: Reclaim at or below the planning line, or while
    /// the host is in distress.
    band: MemoryBand,
    /// What the process released that the readings have yet to show. None
    /// is owed from a reading above the line.
    unseen: u64,
    /// The memory available at this reading: headroom, charge and `unseen`.
    available: u64,
    /// The probe the rate is measured from, and the memory available then.
    base: (Instant, u64),
    /// The rest of the host's taking over the latest span from `base`, in
    /// bytes a second.
    rate: f64,
    /// The fastest `rate` measured since the load's first probe.
    fastest: f64,
    /// Available memory's fluctuation over its readings since the domain
    /// last left Reclaim.
    fluctuation: Fluctuation,
}

/// Available memory's fluctuation in steady state: the largest fall between
/// consecutive readings while the domain's demand is unchanged (the
/// process's charge equal at both, and no bank resize running at either,
/// whose backed pages and charge a reading may see apart), over the readings
/// since the domain last left Reclaim.
///
/// A pair across a change of the process's demand measures the process,
/// not the host; a fall that reached the line is a change of the host's
/// demand, which Reclaim answers with shrinks, so the steady state starts
/// again once it is left (and growth waits for memory to stay free first,
/// `RECOVERY_STABLE_TIME`, so its first step has the readings of that wait).
/// A rise never takes room from a growth: a lifted pressure is not a
/// fluctuation.
#[derive(Clone, Copy, Default)]
struct Fluctuation {
    /// The latest reading's available bytes and the process's charge then,
    /// if no bank resize ran.
    settled: Option<(u64, u64)>,
    largest: u64,
}

impl Fluctuation {
    /// Fold in a reading of `available` bytes with the process charging
    /// `charged`, `settled` if no bank resize ran.
    fn read(&mut self, available: u64, charged: u64, settled: bool) {
        let reading = settled.then_some((available, charged));
        if let (Some((before, was)), Some(_)) = (self.settled, reading) {
            if was == charged {
                self.largest = self.largest.max(before.saturating_sub(available));
            }
        }
        self.settled = reading;
    }
}

/// What a domain's latest reading leaves a shrink to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shortfall {
    /// At or above the planning line.
    Above,
    /// Below it by no more than what the process released, which the host
    /// has yet to show; or in distress above it: nothing to release.
    Covered,
    /// These bytes short beyond what the process released.
    Short(u64),
}

/// The target each memory reading of the owner is traced to: what an
/// instrument samples a load's charge against its room from.
pub const MEMORY_TRACE: &str = "magnitude_memory_reading";

impl DomainPace {
    /// An allocation domain's wait bound (`MemoryPace::wait_bound`): none
    /// in a domain that grows nothing, or before the host took memory.
    fn wait_bound(&self) -> Option<f64> {
        (self.role == DomainRole::Allocation && self.fastest > 0.0)
            .then(|| self.planning.saturating_sub(self.emergency) as f64 / self.fastest)
    }
}

impl MemoryPace {
    /// The pace before any probe or release.
    fn new() -> Self {
        Self {
            domains: BTreeMap::new(),
            reading_interval: Duration::ZERO,
            latency: Duration::ZERO,
        }
    }

    /// Fold in a probe of every domain, the process charging each store as
    /// `charges` say, `settled` if no bank resize runs, and memory read
    /// every `reading_interval`.
    fn observe(
        &mut self,
        readings: &[DomainReading],
        charges: &[(seismic::StoreId, u64)],
        settled: bool,
        reading_interval: Duration,
    ) {
        let at = Instant::now();
        self.reading_interval = reading_interval;
        for reading in readings {
            let charged = charges
                .iter()
                .filter(|(store, _)| *store == reading.domain)
                .map(|(_, bytes)| bytes)
                .sum::<u64>();
            tracing::debug!(
                target: MEMORY_TRACE,
                store = ?reading.identity,
                headroom = reading.headroom_bytes,
                planning = reading.thresholds.planning_bytes,
                charged,
                "memory read"
            );
            let before = self.domains.remove(&reading.domain);
            let shown = reading.headroom_bytes.saturating_add(charged);
            let unseen = match (reading.band, &before) {
                (MemoryBand::Reclaim, Some(before)) => {
                    let owed = before.unseen + before.charged.saturating_sub(charged);
                    // A rise of what is available is the host showing a
                    // release, up to what it owes.
                    owed - (shown + owed).saturating_sub(before.available).min(owed)
                }
                _ => 0,
            };
            let available = shown + unseen;
            let (base, rate) = match &before {
                // A fall of the charge is a release the host may not show
                // yet: the fall of available memory it makes is no taking,
                // and what the host took across it is unknown until a span
                // from here measures it. A rate from before it, carried over
                // the release's latency, would project a taking long over.
                Some(before) if charged < before.charged => ((at, available), 0.0),
                Some(before) if at.duration_since(before.base.0) >= reading_interval => {
                    let (then, was) = before.base;
                    ((at, available), was.saturating_sub(available) as f64 / at.duration_since(then).as_secs_f64())
                }
                Some(before) => (before.base, before.rate),
                None => ((at, available), 0.0),
            };
            let fastest = before.as_ref().map_or(rate, |before| before.fastest.max(rate));
            let fluctuation = match reading.band {
                MemoryBand::Reclaim => Fluctuation::default(),
                MemoryBand::Normal => {
                    let mut fluctuation = before.map(|before| before.fluctuation).unwrap_or_default();
                    fluctuation.read(available, charged, settled);
                    fluctuation
                }
            };
            self.domains.insert(
                reading.domain,
                DomainPace {
                    role: reading.role,
                    headroom: reading.headroom_bytes,
                    planning: reading.thresholds.planning_bytes,
                    emergency: reading.thresholds.emergency_bytes,
                    ceiling: reading.ceiling_bytes,
                    charged,
                    band: reading.band,
                    unseen,
                    available,
                    base,
                    rate,
                    fastest,
                    fluctuation,
                },
            );
        }
    }

    /// What a shrink of what is charged in `store` has to do, decided at the
    /// latest probe with the process charging it `charged` now: that domain's
    /// reading below the planning line, plus what the host takes until the
    /// next shrink takes effect, less what the readings have yet to show and
    /// the charge dropped since this one. Only a release charged in a domain
    /// lifts its deficit, so each holding shrinks by its own domain's; a
    /// domain no probe observed is above.
    fn shortfall(&self, store: seismic::StoreId, charged: u64) -> Shortfall {
        let Some(pace) = self.domains.get(&store) else {
            return Shortfall::Above;
        };
        let below = pace.planning.saturating_sub(pace.headroom);
        if below == 0 {
            // In distress above the line, the room the load reads already
            // leaves out what the host displaced since it began serving
            // (`HostRoom::Served`): nothing it holds is short, and a release
            // would only buy what no measure says is owed.
            return match pace.band {
                MemoryBand::Reclaim => Shortfall::Covered,
                MemoryBand::Normal => Shortfall::Above,
            };
        }
        let released = pace.unseen + pace.charged.saturating_sub(charged);
        match projected(below, pace.rate, self.horizon()).saturating_sub(released) {
            0 => Shortfall::Covered,
            short => Shortfall::Short(short),
        }
    }

    /// Reclaim outlasted the unload clock on what is unseen: everything
    /// released is taken as shown, and the next decision stands on its
    /// reading alone. A span in which the host took as it showed a release
    /// reads as a lag; without this that would wait forever.
    fn lapse_credit(&mut self) {
        for pace in self.domains.values_mut() {
            pace.available -= pace.unseen;
            pace.unseen = 0;
        }
    }

    /// What a growth step in `store` must stay below at the latest probe,
    /// and the margin it leaves: its domain's ceiling less the reading's
    /// fluctuation; `None` before a probe of it. Each store's room is its
    /// own: a step in host memory takes nothing of a device's.
    fn growth_room(&self, store: seismic::StoreId) -> Option<(u64, u64)> {
        self.domains
            .get(&store)
            .map(|pace| (pace.ceiling.saturating_sub(pace.fluctuation.largest), pace.fluctuation.largest))
    }

    /// `store`'s available memory's fluctuation at the latest probe, which a
    /// growth step leaves free; none before a probe of it.
    fn fluctuation(&self, store: seismic::StoreId) -> u64 {
        self.domains.get(&store).map_or(0, |pace| pace.fluctuation.largest)
    }

    /// The wait bound: how long pressure that arrives as a prompt group
    /// starts may wait for a release, in seconds. Nothing is released while
    /// a group runs (the owner defers releases while a chain is held), so
    /// from the planning line `P` the rest of the host may take what lies
    /// above the domain's emergency reserve `E`
    /// (`DomainThresholds::emergency_bytes`), at the fastest rate `r`
    /// measured of it since the load's first probe: `W = (P − E) / r`, the
    /// least of any allocation domain. The reserve is the region a load must
    /// not run into, whatever lies past it on the host: the worker's kill
    /// where the OS reports no pressure level, critical pressure and
    /// compression on macOS, failed allocations in a device's own memory.
    /// None while no allocation domain has seen the host take memory.
    fn wait_bound(&self) -> Option<f64> {
        self.domains
            .values()
            .filter_map(DomainPace::wait_bound)
            .min_by(f64::total_cmp)
    }

    /// From a shrink's decision to the next one's effect.
    fn horizon(&self) -> Duration {
        self.reading_interval + 2 * self.latency
    }

    /// A release finished, as `times` say.
    fn released(&mut self, times: &ResizeTimes) {
        self.latency = times.released - times.decided;
    }
}

/// `deficit` bytes now, and what the host takes at `rate` bytes a second
/// until the next shrink takes effect `horizon` from now.
fn projected(deficit: u64, rate: f64, horizon: Duration) -> u64 {
    deficit.saturating_add((rate * horizon.as_secs_f64()) as u64)
}

impl ExecutorDomain<NativeFamily> {
    /// What steps cost each streamed routed layer so far (requests, misses,
    /// missed bytes, time parked and reading), and what the last step
    /// completed cost; `None` of a load whose experts are held whole.
    pub fn expert_costs(&self) -> Option<crate::experts::ExpertCosts> {
        self.family.expert_costs()
    }

    /// What streamed layer `layer`'s CPU site answered of step `step`
    /// (`Compute::answered`).
    #[cfg(test)]
    pub(crate) fn site_answered(&self, layer: usize, step: u64) -> Option<crate::experts::Answered> {
        self.family.site_answered(layer, step)
    }

    /// The experts each streamed routed layer holds now; `None` of a load
    /// whose experts are held whole.
    pub fn experts_held(&self) -> Option<Vec<Vec<usize>>> {
        self.family.experts_held()
    }

    /// Classify the allocations that are already physically committed before
    /// any request reservation begins. The byte total comes from the actual
    /// graph pools and state stores; Seismic remains the charge authority.
    /// The holding is classified as every observation classifies it
    /// ([`Self::refresh_memory`]).
    pub fn register_allocated_holdings(&mut self) -> Result<HoldingId, String> {
        let holding = self
            .memory
            .borrow_mut()
            .heap_mut()
            .insert(0, HoldingClass::Model)
            .map_err(|error| format!("static memory holding rejected: {error:?}"))?;
        self.static_holding = Some(holding);
        self.refresh_memory().map_err(|error| error.to_string())?;
        Ok(holding)
    }

    pub fn new(
        execution: Rc<ExecutionPlan>,
        definition: Rc<ModelDefinition>,
        memory: DeviceHeap,
        programs: Rc<AttestedPrograms>,
        resources: AllocatedResources,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        resident: crate::ResidentTarget,
        target_state: StoreBindings,
        head_state: Option<StoreBindings>,
    ) -> Result<(Self, StateBindings), String> {
        // A model's drafter (its draft head or separate draft) is enabled
        // only when its method drafts.
        if (head_loader.is_some() && definition.head.is_none() && definition.draft.is_none())
            || vision_loader.is_some() != definition.vision.is_some()
        {
            return Err("component loaders differ from enabled model components".into());
        }
        // A draft head reading the target's history keeps no state arena.
        let keeps_history = definition.draft.is_some()
            || definition
                .head
                .as_ref()
                .is_some_and(|head| head.history == magnitude_family_contracts::HeadHistory::Own);
        if head_state.is_some() != (head_loader.is_some() && keeps_history) {
            return Err("head state arena differs from enabled head component".into());
        }
        let family = NativeFamily::new(programs, resident, definition.decoder.clone())?;
        Ok(Self::with_family(
            execution,
            definition,
            resources,
            memory,
            target_state,
            head_state,
            head_loader,
            vision_loader,
            family,
        ))
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// The target launches graphs were sealed for: the launch row bound and
    /// the target store's span limit.
    fn target_class_limits(&self) -> crate::batching::ClassLimits {
        crate::batching::ClassLimits {
            rows: self.execution.policy().limits().max_launch_rows,
            segments: self.target_store.max_span_limit(),
        }
    }

    /// The head (drafter) launches graphs were sealed for: the launch row
    /// bound and the span limit of the store its rows read (the head's own,
    /// or the target's for a head that keeps none).
    fn head_class_limits(&self) -> Result<crate::batching::ClassLimits, DomainError> {
        let store = match &self.head_store {
            Some(store) => store,
            None if self.head_reads_target() => &self.target_store,
            None => return Err(DomainError::invariant("a head launch without a head store")),
        };
        Ok(crate::batching::ClassLimits {
            rows: self.execution.policy().limits().max_launch_rows,
            segments: store.max_span_limit(),
        })
    }

    /// Whether the load's drafter is a head that keeps no history and reads
    /// each request's accepted target history instead (`HeadHistory::Target`).
    pub(super) fn head_reads_target(&self) -> bool {
        self.head_store.is_none()
            && self
                .definition
                .head
                .as_ref()
                .is_some_and(|head| head.history == magnitude_family_contracts::HeadHistory::Target)
    }

    /// Return every byte that remains resident as part of the model's static
    /// footprint.  This is the single accounting path used both when the
    /// holding is first registered and whenever lazy state/program binding
    /// changes the footprint.
    fn static_holding_bytes(&self) -> Result<u64, String> {
        let mut bytes = self
            .resources
            .committed_bytes()
            .map_err(|error| error.to_owned())?;
        bytes = bytes
            .checked_add(self.target_store.committed_bytes())
            .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        if let Some(store) = &self.head_store {
            bytes = bytes
                .checked_add(store.committed_bytes())
                .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        }
        bytes = bytes
            .checked_add(
                self.target_store
                    .external_pinned_bytes()
                    .map_err(|error| error.to_string())?,
            )
            .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        if let Some(store) = &self.head_store {
            bytes = bytes
                .checked_add(
                    store
                        .external_pinned_bytes()
                        .map_err(|error| error.to_string())?,
                )
                .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        }
        // What the device's graph budget charges, which its ledgers count.
        let device = self.domain.device();
        let graphs = device.graph_cost().bytes(device.graph_budget()).ok_or("graph budget charge overflows")?;
        for charge in [
            self.family.target_weight_bytes()?,
            self.family.prepared_program_bytes(),
            self.family.target_constant_bytes()?,
            graphs.device,
            graphs.upload,
        ] {
            bytes = bytes
                .checked_add(charge)
                .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        }
        Ok(bytes)
    }

    /// The head's own resident weights: what the residency cache its loader
    /// inherits from the target holds beyond the target's imports
    /// (`target_cached_bytes`; the target's expert bank is none of them).
    /// None without a head.
    pub(super) fn head_weight_bytes(&self) -> Result<u64, String> {
        match &self.head_loader {
            None => Ok(0),
            Some(loader) => Ok(beyond_inherited(loader.resident_bytes()?, self.family.target_cached_bytes()?)?),
        }
    }

    /// Classify the optional components' storage, against a charge read now
    /// with every holding whose charge can fall off this thread reconciled
    /// after it ([`Self::refresh_memory`]): a shrink's release lowers the
    /// bank's charge whenever its device work completes, and an observation
    /// any claim took meanwhile would otherwise leave the static holding
    /// classified above the charge it reads.
    pub(super) fn sync_optional_holding(&mut self) -> Result<(), String> {
        self.refresh_memory().map_err(|error| error.to_string())?;
        let head_weights = self.head_weight_bytes()?;
        let vision_weights = self
            .vision_loader
            .as_ref()
            .map(|loader| loader.resident_bytes())
            .transpose()?
            .unwrap_or(0);
        let bound_constants = self.family.optional_constant_bytes()?;
        let bytes = head_weights
            .checked_add(vision_weights)
            .and_then(|weights| weights.checked_add(bound_constants))
            .ok_or_else(|| "optional holding byte count overflow".to_owned())?;
        let mut memory = self.memory.borrow_mut();
        let heap = memory.heap_mut();
        match (self.optional_holding, bytes) {
            (Some(id), 0) => {
                heap.remove(id)
                    .map_err(|error| format!("optional holding release rejected: {error:?}"))?;
                self.optional_holding = None;
            }
            (Some(id), bytes) => heap
                .resize(id, bytes)
                .map_err(|error| format!("optional holding update rejected: {error:?}"))?,
            (None, 0) => {}
            (None, bytes) => {
                let id = heap
                    .insert(bytes, HoldingClass::Dormant)
                    .map_err(|error| format!("optional holding rejected: {error:?}"))?;
                self.optional_holding = Some(id);
            }
        }
        Ok(())
    }

    /// Observe every domain the device uses, set Seismic's allocation
    /// ceiling from the readings, and refresh the heap's view from the same
    /// readings. Seismic remains the physical charge authority; the heap only
    /// classifies ownership and issues claims. A failed observation revokes
    /// unspent grants and leaves the heap Blind.
    ///
    /// The static holding is classified only here, against the observation
    /// just made, its live charges read after the charge while the expert
    /// bank holds every rise of its charge off this thread
    /// (`ElasticBank::hold_rises`): what remains off this thread is a
    /// release (an expert bank shrink's, once the device work before it
    /// completes; a released slab an outside view retained), which only
    /// ever lowers its charge, so the holding is never classified above
    /// the charge it is read against.
    pub fn refresh_memory(&self) -> Result<Vec<DomainReading>, DomainError> {
        let _rises = self.family.expert_bank().map(|bank| bank.hold_rises());
        let readings = self
            .memory
            .borrow_mut()
            .refresh()
            .map_err(|error| DomainError::Blind(error.to_string()))?;
        if let Some(holding) = self.static_holding {
            let bytes = self.static_holding_bytes().map_err(DomainError::Input)?;
            self.memory
                .borrow_mut()
                .heap_mut()
                .resize(holding, bytes)
                .map_err(|error| {
                    DomainError::Input(format!("static memory holding update rejected: {error:?}"))
                })?;
        }
        Ok(readings)
    }

    pub fn memory(&self) -> std::cell::Ref<'_, MemoryHeap> {
        std::cell::Ref::map(self.memory.borrow(), DeviceHeap::heap)
    }

    /// Bytes of the model's host-resident tables held in host RAM.
    pub fn host_table_bytes(&self) -> u64 {
        self.memory.borrow().host_table_bytes()
    }

    /// Refuse a launch whose vocabulary work the load did not plan: more
    /// selected target rows or drafting requests than the selection bound,
    /// logits a served load does not export, more exported rows than a
    /// diagnostic load exports, or shaping in a launch whose logits are
    /// exported (shaping rewrites the logits in place).
    fn admit_selection(
        &self,
        lane: ReservationLane,
        operations: &[Operation],
    ) -> Result<(), DomainError> {
        let limits = self.execution.policy().limits();
        match lane {
            ReservationLane::Target => {
                let forwards = operations.iter().filter_map(|operation| match operation {
                    Operation::Forward {
                        kind,
                        tokens,
                        demand,
                        select,
                        ..
                    } => Some((kind, tokens, demand, select)),
                    Operation::Head { .. } | Operation::Encode { .. } => None,
                });
                let selected = forwards
                    .clone()
                    .map(|(_, _, _, select)| select.len())
                    .sum::<usize>();
                if selected > limits.max_selected_rows {
                    return Err(DomainError::Input(format!(
                        "launch selects {selected} rows; the selection bound is {}",
                        limits.max_selected_rows
                    )));
                }
                if !readout::exports_logits(limits) {
                    if forwards
                        .clone()
                        .any(|(_, _, demand, _)| demand.contains(Demand::LOGITS))
                    {
                        return Err(DomainError::Input(
                            "launch demands logits from a load that exports none".into(),
                        ));
                    }
                    return Ok(());
                }
                let projected = projected_rows(operations);
                for (_, _, _, select) in forwards {
                    for spec in select {
                        let params = target::select_row(spec)
                            .shaping
                            .params()
                            .map_err(|error| DomainError::Input(error.to_string()))?;
                        if readout::shapes(&params) {
                            return Err(DomainError::Input(
                                "a load that exports logits selects unshaped".into(),
                            ));
                        }
                    }
                }
                if projected > limits.exported_logits_rows {
                    return Err(DomainError::Input(format!(
                        "launch projects {projected} logits rows; the load exports {}",
                        limits.exported_logits_rows
                    )));
                }
            }
            ReservationLane::Head => {
                let drafting = operations
                    .iter()
                    .filter(|operation| {
                        matches!(operation, Operation::Head { proposals, .. } if !proposals.is_empty())
                    })
                    .count();
                if drafting > limits.max_drafting_slots {
                    return Err(DomainError::Input(format!(
                        "launch drafts for {drafting} requests; the selection bound is {}",
                        limits.max_drafting_slots
                    )));
                }
            }
            ReservationLane::Vision => {}
        }
        Ok(())
    }

    pub fn requirements(&self, operations: &[Operation]) -> Result<DomainRequirements, DomainError> {
        if let Some(class) = self.claim_class(operations) {
            return Ok(DomainRequirements {
                lane: ReservationLane::Target,
                pool: PoolClass::Target(class),
                state_demand: Vec::new(),
                successor_banks: 0,
                priming_demand: Vec::new(),
                priming_banks: 0,
                projected: projected_rows(operations),
                claim: true,
            });
        }
        let first = operations
            .first()
            .ok_or_else(|| DomainError::invariant("empty reservation"))?;
        let lane = match first {
            Operation::Forward { .. } => ReservationLane::Target,
            Operation::Head { .. } => ReservationLane::Head,
            Operation::Encode { .. } => ReservationLane::Vision,
        };
        if operations.iter().any(|operation| {
            !matches!(
                (lane, operation),
                (ReservationLane::Target, Operation::Forward { .. })
                    | (ReservationLane::Head, Operation::Head { .. })
                    | (ReservationLane::Vision, Operation::Encode { .. })
            )
        }) {
            return Err(DomainError::invariant(
                "reservation crosses numerical lanes",
            ));
        }
        let mut priming_demand = Vec::new();
        let mut priming_banks = 0usize;
        for operation in operations {
            if let Operation::Forward {
                request,
                prime: Some(prime),
                ..
            } = operation
            {
                if self.head_reads_target() {
                    return Err(DomainError::invariant(
                        "a head reading target history keeps no history to prime",
                    ));
                }
                if operations.len() != 1 {
                    return Err(DomainError::Input(
                        "a primed prompt chunk launches alone".into(),
                    ));
                }
                let state = self.head.get(request).ok_or_else(|| {
                    DomainError::Input("a primed prompt chunk's drafter is not idle".into())
                })?;
                if state.position() != prime.position {
                    return Err(DomainError::Input(
                        "drafter entry position differs from accepted drafter state".into(),
                    ));
                }
                if prime.boundary {
                    let Operation::Forward {
                        position, tokens, ..
                    } = operation
                    else {
                        unreachable!()
                    };
                    if prime.position != position.saturating_sub(1)
                        || prime.tokens.as_slice()
                            != if *position == 0 {
                                &tokens[1..]
                            } else {
                                &tokens[..]
                            }
                        || self
                            .input
                            .get(request)
                            .and_then(|input| input.boundary.as_ref())
                            .is_some()
                            != (*position > 0)
                    {
                        return Err(DomainError::Input(
                            "boundary priming differs from the accepted target prefix".into(),
                        ));
                    }
                }
                priming_demand.extend(state.demands(prime.tokens.len()));
                priming_banks += usize::from(!prime.tokens.is_empty());
            }
        }
        // A head that keeps no history reads each request's accepted target
        // state, which its flight holds unchanged: it reserves no rows or
        // banks.
        let reads_target = lane == ReservationLane::Head && self.head_reads_target();
        let (pool, state_demand, successor_banks) = match lane {
            ReservationLane::Target | ReservationLane::Head => {
                let mut rows = 0usize;
                let mut state_demand = Vec::new();
                let mut segments = 1usize;
                let mut demand = Demand::NONE;
                let mut seen = BTreeSet::new();
                for operation in operations {
                    operation.validate().map_err(|error| error.to_string())?;
                    if let Operation::Head {
                        form: crate::DraftForm::Block,
                        proposals,
                        ..
                    } = operation
                    {
                        if !proposals.is_empty()
                            && !self
                                .execution
                                .policy()
                                .method()
                                .block_widths()
                                .iter()
                                .any(|width| usize::from(*width) == proposals.len())
                        {
                            return Err(DomainError::Input(
                                "draft width was not prepared by this load".into(),
                            ));
                        }
                    }
                    if !seen.insert(operation.request()) {
                        return Err(DomainError::Input("reservation repeats a request".into()));
                    }
                    rows = rows
                        .checked_add(operation.row_count())
                        .ok_or_else(|| DomainError::invariant("reservation row count overflow"))?;
                    demand |= operation.demand();
                    let state = match lane {
                        ReservationLane::Target => self.target.get(&operation.request()),
                        ReservationLane::Head if reads_target => {
                            self.target.get(&operation.request())
                        }
                        ReservationLane::Head => self.head.get(&operation.request()),
                        _ => unreachable!(),
                    }
                    .ok_or_else(|| DomainError::Input("reservation request is not idle".into()))?;
                    if reads_target
                        && !matches!(
                            operation,
                            Operation::Head {
                                phase: crate::HeadPhase::Generation,
                                tokens,
                                proposals,
                                form: crate::DraftForm::Chained,
                                ..
                            } if tokens.len() == 1 && !proposals.is_empty()
                        )
                    {
                        return Err(DomainError::invariant(
                            "a head reading target history drafts proposals from one anchor row",
                        ));
                    }
                    let position = match operation {
                        Operation::Forward { position, .. } | Operation::Head { position, .. } => {
                            *position
                        }
                        _ => unreachable!(),
                    };
                    if state.position() != position {
                        return Err(DomainError::Input(
                            "reservation position differs from accepted state".into(),
                        ));
                    }
                    if !reads_target {
                        state_demand.extend(state.demands(operation.row_count()));
                    }
                    // Speculative target rows are recorded on the recurrent
                    // tape, which holds the planned draft rows.
                    if lane == ReservationLane::Target
                        && operation.row_count() - operation.committed_rows()
                            > self.execution.policy().method().draft_rows()
                    {
                        return Err(DomainError::Input(
                            "speculative target rows exceed the planned draft rows".into(),
                        ));
                    }
                    match operation {
                        Operation::Forward {
                            tokens,
                            conditioning,
                            ..
                        } if conditioning.as_ref().is_some_and(|lease| {
                            lease.domain() != self.domain.id()
                                || lease.allocation().rows() != tokens.len()
                        }) =>
                        {
                            return Err(DomainError::Input(
                                "target conditioning differs from physical rows or resource domain"
                                    .into(),
                            ));
                        }
                        Operation::Head { conditioning, .. }
                            if conditioning.row_bytes() != self.head_conditioning_bytes() =>
                        {
                            return Err(DomainError::Input(
                                "head conditioning rows differ from the activation width".into(),
                            ));
                        }
                        _ => {}
                    }
                    segments = segments.max(if lane == ReservationLane::Target {
                        target::reserved_segments(
                            &self.target_store,
                            state.span_count(),
                            operation.row_count(),
                        )
                    } else {
                        state.span_count()
                    });
                }
                self.admit_selection(lane, operations)?;
                let class_limits = if lane == ReservationLane::Target {
                    self.target_class_limits()
                } else {
                    self.head_class_limits()?
                };
                let class = crate::LaunchClass::covering(rows, segments, demand, class_limits)
                    .map_err(|error| error.to_string())?;
                (
                    if lane == ReservationLane::Target {
                        PoolClass::Target(class)
                    } else {
                        PoolClass::Head(class)
                    },
                    state_demand,
                    if reads_target { 0 } else { operations.len() },
                )
            }
            ReservationLane::Vision => {
                let [Operation::Encode { image, .. }] = operations else {
                    return Err(DomainError::Input(
                        "vision reservation must contain one image".into(),
                    ));
                };
                let input = self.input.get(&first.request()).ok_or_else(|| {
                    DomainError::Input("vision request has no admitted input".into())
                })?;
                let slot = input
                    .images
                    .values()
                    .find(|slot| slot.image == *image)
                    .ok_or_else(|| {
                        DomainError::Input("image is not part of admitted input".into())
                    })?;
                if slot.features.is_some() {
                    return Err(DomainError::Input("image is already encoded".into()));
                }
                (
                    PoolClass::Vision {
                        patch_rows: self
                            .image_class(image.patches())
                            .map_err(DomainError::Input)?,
                    },
                    Vec::new(),
                    0,
                )
            }
        };
        Ok(DomainRequirements {
            lane,
            pool,
            state_demand,
            successor_banks,
            priming_demand,
            priming_banks,
            projected: projected_rows(operations),
            claim: false,
        })
    }

    pub fn can_reserve(&self, requirements: &DomainRequirements) -> Result<(), CapacityError> {
        if requirements.claim {
            return Ok(());
        }
        let available = |workspace: Option<usize>, output: Option<usize>| {
            if workspace.unwrap_or(0) == 0 {
                return Err(CapacityError {
                    resource: ResourceKind::Workspace,
                    required: 1,
                    available: 0,
                });
            }
            if output.is_some_and(|count| count == 0) {
                return Err(CapacityError {
                    resource: ResourceKind::Output,
                    required: 1,
                    available: 0,
                });
            }
            Ok(())
        };
        match requirements.lane {
            ReservationLane::Target => {
                let graph = self.resources.target_graph();
                if graph.available_workspace() == 0 || graph.available_output() < 2 {
                    return Err(CapacityError {
                        resource: if graph.available_workspace() == 0 {
                            ResourceKind::Workspace
                        } else {
                            ResourceKind::Output
                        },
                        required: if graph.available_workspace() == 0 {
                            1
                        } else {
                            2
                        },
                        available: if graph.available_workspace() == 0 {
                            0
                        } else {
                            graph.available_output() as u64
                        },
                    });
                }
                let readout = self.resources.target_readout_graph(requirements.projected);
                available(
                    Some(readout.available_workspace()),
                    Some(readout.available_output()),
                )?;
            }
            ReservationLane::Head => {
                let graph = self.resources.head_graph().ok_or(CapacityError {
                    resource: ResourceKind::Workspace,
                    required: 1,
                    available: 0,
                })?;
                available(
                    Some(graph.available_workspace()),
                    Some(graph.available_output()),
                )?;
            }
            ReservationLane::Vision => {
                let graph = self.resources.vision_graph().ok_or(CapacityError {
                    resource: ResourceKind::Workspace,
                    required: 1,
                    available: 0,
                })?;
                let class = image_patch_rows(requirements.pool)?;
                available(
                    Some(graph.available_workspace(class)),
                    Some(graph.available_output(class)),
                )?;
            }
        }
        if requirements.priming_banks != 0 {
            let graph = self.resources.head_graph().ok_or(CapacityError {
                resource: ResourceKind::Workspace,
                required: 1,
                available: 0,
            })?;
            available(
                Some(graph.available_workspace()),
                Some(graph.available_output()),
            )?;
            let store = self.head_store.as_ref().ok_or(CapacityError {
                resource: ResourceKind::StateRows,
                required: requirements
                    .priming_demand
                    .iter()
                    .map(|demand| demand.rows as u64)
                    .sum(),
                available: 0,
            })?;
            holds(store, &requirements.priming_demand)?;
            if store.available_banks() < requirements.priming_banks {
                return Err(CapacityError {
                    resource: ResourceKind::RecurrentBanks,
                    required: requirements.priming_banks as u64,
                    available: store.available_banks() as u64,
                });
            }
        }
        let store = if requirements.lane == ReservationLane::Head {
            self.head_store.as_ref()
        } else {
            Some(&self.target_store)
        };
        // A head reading target history reserved no rows or banks.
        let reads_target = requirements.lane == ReservationLane::Head && self.head_reads_target();
        if requirements.lane != ReservationLane::Vision && !reads_target {
            let store = store.expect("head requirements require a head store");
            holds(store, &requirements.state_demand)?;
            let banks = store.available_banks();
            if banks < requirements.successor_banks {
                return Err(CapacityError {
                    resource: ResourceKind::RecurrentBanks,
                    required: requirements.successor_banks as u64,
                    available: banks as u64,
                });
            }
        }
        Ok(())
    }

    /// Import and bind the optional component `lane` needs when it is unbound.
    fn bind_component(&mut self, lane: ReservationLane) -> Result<(), DomainError> {
        match lane {
            ReservationLane::Head if !self.family.head_is_bound() => {
                let resident = self
                    .head_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("head component is disabled".into()))?
                    .load()
                    .map_err(|error| DomainError::Input(error.to_string()))?;
                self.family.bind_head(resident, &self.definition)?;
                self.hold_bound_head()?;
            }
            ReservationLane::Vision if !self.family.vision_is_bound() => {
                let resident = self
                    .vision_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("vision component is disabled".into()))?
                    .load()
                    .map_err(|error| DomainError::Input(error.to_string()))?;
                self.family.bind_vision(resident, &self.definition)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Reserve capacity for one group: the front queued step's slots when
    /// the group claims it, otherwise its provisioned state, advances and
    /// launch leases.
    pub fn reserve(
        &mut self,
        bindings: &mut StateBindings,
        operations: &[Operation],
    ) -> Result<DomainReservation, DomainError> {
        self.refresh_memory()?;
        let requirements = self.requirements(operations)?;
        if requirements.claim {
            let slots = self
                .claim_slots(operations)
                .ok_or_else(|| DomainError::invariant("a claimable group lost its queued step"))?;
            return Ok(DomainReservation {
                requirements,
                resources: ReservedResources::Target(TargetGraphReservation::Claim(slots)),
            });
        }
        // Commit the selected group's numerical backing before a lazy
        // component import. Provisioning grants the physical growth peak
        // through the heap before any backing allocation. The owner applied
        // any state ceiling when it selected the group.
        self.provision(bindings, operations, None)?;
        // Provisioning may relocate a shared tail into a fresh page. Recompute
        // row demand against that layout: the next append can now fit in place.
        let requirements = self.requirements(operations)?;
        self.refresh_memory()?;
        // The import and binding of an unbound component run under one heap
        // claim for their peak, released once Seismic's charge reflects them.
        let claim = self
            .component_growth(operations)?
            .map(|(required, staged)| {
                self.claim_device_growth(required, staged, HoldingClass::Dormant)
            })
            .transpose()?;
        let bound = self.bind_component(requirements.lane).and_then(|()| {
            if requirements.priming_banks != 0 {
                self.bind_component(ReservationLane::Head)
            } else {
                Ok(())
            }
        });
        if let Some(claim) = claim {
            self.release_claim(claim);
        }
        bound?;
        self.refresh_memory()?;
        self.sync_optional_holding().map_err(DomainError::Input)?;
        self.can_reserve(&requirements)
            .map_err(DomainError::Capacity)?;
        let reads_target = requirements.lane == ReservationLane::Head && self.head_reads_target();
        let invariant = |detail: &str, error: CapacityError| {
            DomainError::invariant(format!(
                "{detail} changed after availability check: {error}"
            ))
        };
        let boundary = if operations.iter().any(|op| {
            matches!(
                op,
                Operation::Forward {
                    prime: Some(Priming { boundary: true, .. }),
                    ..
                }
            )
        }) {
            Some(self.allocate_boundary()?)
        } else {
            None
        };
        let resources = match requirements.lane {
            ReservationLane::Target => {
                let graph = self.resources.target_graph();
                let graph_workspace = graph
                    .acquire_workspace()
                    .map_err(|error| invariant("target graph workspace", error))?;
                let output = || {
                    graph
                        .acquire_output()
                        .map_err(|error| invariant("target graph output", error))
                };
                let (graph_output, spare_output) = (output()?, output()?);
                let readout = self.resources.target_readout_graph(requirements.projected);
                let readout_workspace = readout
                    .acquire_workspace()
                    .map_err(|error| invariant("target readout workspace", error))?;
                let readout_output = readout
                    .acquire_output()
                    .map_err(|error| invariant("target readout output", error))?;
                ReservedResources::Target(TargetGraphReservation::Launch(TargetLaunchReservation {
                    advances: Vec::with_capacity(operations.len()),
                    graph_workspace,
                    graph_output,
                    spare_output,
                    readout_workspace,
                    readout_output,
                    priming: None,
                    boundary,
                }))
            }
            ReservationLane::Head => {
                let graph = self.resources.head_graph().expect("checked head graph");
                ReservedResources::Head(
                    graph
                        .acquire_workspace()
                        .map_err(|error| invariant("head graph workspace", error))?,
                    graph
                        .acquire_output()
                        .map_err(|error| invariant("head graph output", error))?,
                    if reads_target {
                        HeadReservation::TargetReads(Vec::with_capacity(operations.len()))
                    } else {
                        HeadReservation::Advances(Vec::with_capacity(operations.len()))
                    },
                )
            }
            ReservationLane::Vision => {
                let graph = self.resources.vision_graph().expect("checked vision graph");
                let class = image_patch_rows(requirements.pool).map_err(DomainError::Capacity)?;
                ReservedResources::Vision(
                    graph
                        .acquire_workspace(class)
                        .map_err(|error| invariant("vision graph workspace", error))?,
                    graph
                        .acquire_output(class)
                        .map_err(|error| invariant("vision graph output", error))?,
                )
            }
        };
        let mut resources = resources;
        match &mut resources {
            ReservedResources::Target(TargetGraphReservation::Launch(
                TargetLaunchReservation {
                    advances, priming, ..
                },
            )) => {
                if let [
                    Operation::Forward {
                        request,
                        prime: Some(prime),
                        ..
                    },
                ] = operations
                {
                    if !prime.tokens.is_empty() {
                        let graph = self.resources.head_graph().expect("checked head graph");
                        let graph_workspace = graph
                            .acquire_workspace()
                            .map_err(|error| invariant("drafter entry workspace", error))?;
                        let graph_output = graph
                            .acquire_output()
                            .map_err(|error| invariant("drafter entry output", error))?;
                        let state = self
                            .head
                            .remove(request)
                            .expect("reservation preflight established drafter ownership");
                        let rows = prime.tokens.len();
                        match OwnedStateAdvance::begin_speculative(state, rows, rows) {
                            Ok(advance) => {
                                *priming = Some(PrimingReservation {
                                    advance,
                                    graph_workspace,
                                    graph_output,
                                })
                            }
                            Err((state, error)) => {
                                self.head.insert(*request, state);
                                return Err(DomainError::invariant(format!(
                                    "drafter state capacity changed after availability check: {error}"
                                )));
                            }
                        }
                    }
                }
                for operation in operations {
                    let request = operation.request();
                    let state = self
                        .target
                        .remove(&request)
                        .expect("reservation preflight established target ownership");
                    match OwnedStateAdvance::begin_speculative(
                        state,
                        operation.row_count(),
                        operation.committed_rows(),
                    ) {
                        Ok(advance) => advances.push(advance),
                        Err((state, error)) => {
                            self.target.insert(request, state);
                            for (operation, advance) in operations.iter().zip(advances.drain(..)) {
                                self.target.insert(operation.request(), advance.abort());
                            }
                            if let Some(priming) = priming.take() {
                                self.head.insert(request, priming.advance.abort());
                            }
                            return Err(DomainError::invariant(format!(
                                "target state capacity changed after availability check: {error}"
                            )));
                        }
                    }
                }
            }
            ReservedResources::Head(_, _, HeadReservation::TargetReads(states)) => {
                for operation in operations {
                    states.push(
                        self.target
                            .remove(&operation.request())
                            .expect("reservation preflight established target ownership"),
                    );
                }
            }
            ReservedResources::Head(_, _, HeadReservation::Advances(advances)) => {
                for operation in operations {
                    let request = operation.request();
                    let state = self
                        .head
                        .remove(&request)
                        .expect("reservation preflight established head ownership");
                    match OwnedStateAdvance::begin_speculative(
                        state,
                        operation.row_count(),
                        operation.committed_rows(),
                    ) {
                        Ok(advance) => advances.push(advance),
                        Err((state, error)) => {
                            self.head.insert(request, state);
                            for (operation, advance) in operations.iter().zip(advances.drain(..)) {
                                self.head.insert(operation.request(), advance.abort());
                            }
                            return Err(DomainError::invariant(format!(
                                "head state capacity changed after availability check: {error}"
                            )));
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(DomainReservation {
            requirements,
            resources,
        })
    }
    /// The domain over `family`, and the one [`StateBindings`] of its stores.
    pub fn with_family(
        execution: Rc<ExecutionPlan>,
        definition: Rc<ModelDefinition>,
        resources: AllocatedResources,
        memory: DeviceHeap,
        target_state: StoreBindings,
        head_state: Option<StoreBindings>,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        family: F,
    ) -> (Self, StateBindings) {
        let id = resources.domain().clone();
        let lane_identities = ["target", "head", "vision"].map(|lane| {
            ProgramIdentity::new(format!("{id}:{lane}"))
                .expect("a lane identity names its lane and is never empty")
        });
        let domain = Self {
            execution,
            definition,
            resources,
            domain: ResourceDomain::new(id, memory.device().clone()),
            memory: RefCell::new(memory),
            static_holding: None,
            optional_holding: None,
            target_store: Rc::clone(&target_state),
            head_store: head_state.as_ref().map(|state| Rc::clone(state)),
            family,
            head_loader,
            vision_loader,
            target: BTreeMap::new(),
            queue: queue::Queue::Empty,
            head: BTreeMap::new(),
            input: BTreeMap::new(),
            lane_identities,
            selection_read: None,
            target_timing: None,
            trace_host_steps: std::env::var_os("MAGNITUDE_TRACE_HOST_STEP").is_some(),
            planned: Vec::new(),
            decode_owed: None,
            places_want: None,
            measurement_claim: None,
            grown_last: None,
            next_flight: 0,
            host_turn: None,
            poisoned: BTreeMap::new(),
            experts: None,
            bank_claim: None,
            resize_held: ResizeHeld::default(),
            pace: RefCell::new(MemoryPace::new()),
        };
        let bindings = StateBindings {
            target: target_state,
            head: head_state,
        };
        (domain, bindings)
    }

    pub fn resource_identity(&self) -> &ResourceDomainId {
        self.domain.id()
    }
    pub fn execution_path(&self) -> crate::ExecutionPath {
        self.execution.policy().path()
    }
    /// The backend of the device this domain executes on.
    pub fn execution_backend(&self) -> seismic::BackendName {
        self.execution.device().backend()
    }
    pub fn resources(&self) -> &ResourceDomain {
        &self.domain
    }

    /// The compatibility key of an operation. Keying does not validate;
    /// reservation validates every operation at the domain boundary.
    pub fn group_key(&self, operation: &Operation) -> GroupKey {
        let executable = operation.executable();
        let lane = match executable {
            crate::ExecutableKind::Target => 0,
            crate::ExecutableKind::Head => 1,
            crate::ExecutableKind::Encoder => 2,
        };
        GroupKey {
            program_identity: self.lane_identities[lane].clone(),
            executable,
            commitment: operation.commitment(),
            draft_width: match operation {
                Operation::Head {
                    form: crate::DraftForm::Block,
                    proposals,
                    ..
                } => Some(proposals.len()),
                _ => None,
            },
        }
    }

    /// Host timing of the most recently finished target step.
    pub fn target_timing(&self) -> Option<TargetHostTiming> {
        self.target_timing
    }
}
