//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Demand, Draw, DrawKind, Row, RowHistory, Select, Shaping as RowShaping, Slot,
    ValidatedTargetBatch,
};
use crate::memory::{ClaimId, HoldingClass, MemoryHeap};
use crate::platform::{
    DomainReading, DomainRole, DomainThresholds, MemoryBand, READING_INTERVAL, ReleaseRate, Releasable, StoreMotion,
};
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
mod borrowing;
pub(crate) mod value;
mod draft;
mod family;
mod features;
mod head;
mod heap;
mod holdings;
mod serving_heap;
pub use serving_heap::{BankGrant, BankRelease, OwnerHoldings, ServingSettlement};
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

pub use family::{
    BankHolding, BankResize, BankResized, BankStep, ElasticBank, ExpertResidency, BankCurve, Reach, Reaches, SizeStep, HostResidency,
    NativeFamily,
    ProgramFamily, ResizeTimes, RunningBegun, RunningPressure, TableResidency, OptionalComponent, ShrinkCause, TierFill, TrafficHold, Trip,
};
pub use heap::{ClaimRefusal, DeviceHeap, MachineReading, Reclaimed, StatusReading, require_normal};
pub use in_flight::{
    HeadFlight, PrimingCompletionFlight, TargetContinuation, TargetFlight, VisionFlight,
};
use in_flight::{PrimingFlight, TargetWork, decode_selected};
pub use ownership::OpenRequirements;
pub use state::{MeasurementRoom, MemoryChargeReconciliation, MemoryMoment, SlotRelease, StateShrink, StoreAllocation};
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
        let refusal = match &error {
            magnitude_state::Error::Tensor(seismic::TensorError::Execution(execution)) => execution.capacity_refusal(),
            _ => None,
        };
        if let Some((required, available)) = refusal {
            return Self::Capacity(crate::CapacityError { resource: crate::ResourceKind::DeviceMemory, required, available });
        }
        match error {
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
    /// graph was recorded, nor any expert bank resize ran or fill began or
    /// ended, within it; and whether the bank's fill ran beside it.
    timed: Timed,
    image: Option<ImageRef>,
}

/// What a launch's duration is its own across: the graphs recorded on the
/// device, by any load (a recording is paid on the host within the
/// duration), read from the device's own count at each submission and
/// finish, the expert bank's resizes begun (a resize holds the thread at
/// the boundaries it runs across, and a growth's new slots or a shrink's
/// evictions change what the launch misses), and its fill's starts and
/// finishes (`ElasticBank::fill_epoch`). A launch is timed only where the
/// epoch at its finish is the one at its submission, none while a resize
/// runs: a resize changes no launch cost the draft width is chosen by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaunchEpoch {
    graphs: u64,
    resizes: u64,
    /// None without an elastic bank.
    fills: Option<u64>,
}

impl LaunchEpoch {
    /// Whether the bank's fill ran beside a launch timed across this epoch:
    /// its count is even while the fill lands experts.
    fn filling(self) -> bool {
        self.fills.is_some_and(|fills| fills % 2 == 0)
    }
}

/// Whether a launch's duration was its own ([`LaunchEpoch`]), and whether
/// the expert bank's fill ran beside it then.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Timed {
    Untimed,
    Settled,
    Filling,
}

/// A launch's duration where it was its own ([`LaunchEpoch`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchDuration {
    pub duration: Duration,
    /// Whether the expert bank's fill landed experts beside the launch: it
    /// missed and read what the fill left, not what the bank settles to, so
    /// its duration prices the load only until the fill finishes.
    pub filling: bool,
}

impl LaunchDuration {
    /// `duration`, measured with no fill beside it.
    pub fn settled(duration: Duration) -> Self {
        Self { duration, filling: false }
    }
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
    pub fn launch_duration(&self) -> Option<LaunchDuration> {
        let filling = match self.timed {
            Timed::Untimed => return None,
            Timed::Settled => false,
            Timed::Filling => true,
        };
        Some(LaunchDuration { duration: self.physical_duration, filling })
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
    /// The load's serving rates as it serves (`WorkloadMeter`), from its
    /// priced decode's prior on ([`Self::serving_rates`]); none before the
    /// bank is priced.
    workload: Option<crate::workload::WorkloadMeter>,
    /// The service's measurement job's claim beside the model, which the
    /// bank's shrink makes room for (`ExecutorDomain::measurement_room`).
    measurement_claim: Option<state::MeasurementClaim>,
    /// What of the prompt-group class the plan reserves room for
    /// (`ModelLoadPlan::group_class_pass`: the class of the places its pass
    /// merges)
    /// no group holds yet, in device storage: no growth of the bank takes
    /// it, so a group's places find it however the bank grew before them.
    /// Set as the domain is made, falling as groups hold places
    /// (`grant_group`) and whole again once Reclaim releases the class to
    /// its floor; none where the plan reserves none.
    class_reserve: u64,
    /// Identity of the next target flight.
    next_flight: u64,
    host_turn: Option<in_flight::HostTurn>,
    /// Requests whose drafter proposed since their last target launch: that
    /// launch is their round's verification, however many proposals it
    /// verifies, and queues no decode behind it (`queue::continuations`).
    proposed: BTreeSet<RequestId>,
    /// Sequences a step of which could not read its weights: every later
    /// outcome of theirs fails (`Outcome::WeightReadFailed`) until they
    /// close.
    poisoned: BTreeMap<RequestId, crate::WeightReadError>,
    /// What the load is priced on once its warm-up ended
    /// ([`Self::measure_load`]); none before, so a warm-up's steps are not
    /// counted.
    measures: Option<LoadMeasures>,
    /// The heap claim of the expert bank's running growth.
    bank_claim: Option<GrowthClaim>,
    /// What expert bank resizes held the submitting thread.
    resize_held: ResizeHeld,
    /// A bank shrink began and its step, the next decode launch, has not
    /// launched: that launch records no graph, as every one it replays was
    /// recorded and kept as its slots were made.
    shrink_step: bool,
    /// What the rest of the host does to memory, read by read, and what
    /// sizes the expert bank's resizes from it.
    pace: RefCell<MemoryPace>,
    /// The readings of the heap's charge taken ([`Self::refresh_memory`]).
    #[cfg(test)]
    refreshes: std::cell::Cell<usize>,
    /// A cut of the expert bank begun beside a running chain, until the
    /// owner takes it ([`Self::take_running_release`]): the storage it
    /// resizes and what the process held as it was decided.
    running_release: Option<(crate::planning::BankTier, MemoryMoment)>,
    /// One prompt group's seconds by size as last priced
    /// (`serving_heap::PlacesPrompt`); cleared as the load's measures
    /// change.
    places_prompt: RefCell<Option<serving_heap::PlacesPrompt>>,
    /// The outcome of the newest periodic reading observed that the owner
    /// has not taken yet ([`Self::take_published_memory`]): one observed
    /// between a chain's submissions is the owner's reading too.
    unseen_reading: RefCell<Option<Result<Vec<DomainReading>, DomainError>>>,
    /// The last serving settle's inputs and the room it stands in
    /// (`serving_heap::SettleStands`); none before one, after a decision the
    /// owner could not carry out, or as the load's measures change.
    settled: Option<serving_heap::SettleStands>,
    /// Requests served so far (`served_request`): an input of the settle.
    requests: u64,
    /// The serving settles run: a test's count of them.
    #[cfg(test)]
    settles: std::cell::Cell<u64>,
}

impl<F: ProgramFamily> Drop for ExecutorDomain<F> {
    /// The queued steps are the one device work the domain owns: each is
    /// waited for and dropped before any field, so its waits on a streamed
    /// load's host are answered by a watcher that still serves (the
    /// family's bank, whose watcher answers nothing once it stops, drops
    /// with the fields). A queued step that faulted is nobody's outcome:
    /// the device is past it either way.
    fn drop(&mut self) {
        match self.drain_queue() {
            Ok(()) => {}
            Err(DomainError::Device(error)) => {
                tracing::warn!("a queued step the dropped load never claimed faulted: {error}");
            }
            Err(error) => panic!("a queued step the dropped load never claimed: {error}"),
        }
    }
}

/// A load's launch of a number of rows at a depth, `(rows, depth)`, in
/// each form it decodes in, as the load's assessment estimates it on this
/// executor (`assessment::decode_step`): a decode step of a row a sequence
/// within the decode row classes, a prefill launch beyond them
/// (`estimate::launch_seconds`). A load without a bank decodes in one form,
/// both the same.
pub type DecodeStep = dyn Fn(u32, u32) -> Result<crate::assessment::FormSeconds, crate::assessment::AssessmentError>;

/// What every load prices its memory's decisions on
/// ([`ExecutorDomain::measure_load`]).
pub struct LoadMeasures {
    /// The load's costs on its executor's: every load waits for the
    /// executor's required figures before it serves, so they are never
    /// pending here; unavailable, and why, where nothing keeps them (a load
    /// without a kernel cache), when nothing the load would grow is worth
    /// anything known.
    pub costs: Result<LoadCosts, String>,
    /// A banked load's measures of its experts; none of a load without a
    /// bank, and only of one with.
    pub experts: Option<ExpertMeasures>,
}

/// What a load's own decisions are priced on of its executor's costs.
pub struct LoadCosts {
    /// The launch in each form ([`DecodeStep`]): what the load's workload
    /// prior is priced on (`ExecutorDomain::serving_prior`), and a bank's
    /// growth and the drafter's worth.
    pub step: Box<DecodeStep>,
    /// What a prompt group's submission costs the host beyond a launch
    /// chained behind another (`ExecutorCosts::group_submission_seconds`):
    /// what each group fewer saves a prompt (`ExecutorDomain::group_seconds`).
    pub submission: f64,
}

impl LoadCosts {
    /// The costs of a load of `definition` planned as `planned`, as
    /// `resident` before any bank, on `costs`: its launch in each form
    /// (`assessment::decode_step`) and its groups' submission. The one
    /// derivation every load is priced by.
    pub fn of(
        definition: &magnitude_family_contracts::ModelDefinition,
        resident: &crate::ExecutionPlanDraft,
        planned: &crate::ExecutionPlanDraft,
        costs: &crate::assessment::ExecutorCosts,
    ) -> Result<Self, crate::assessment::AssessmentError> {
        Ok(Self {
            step: crate::assessment::decode_step(definition, resident, planned, costs.clone())?,
            submission: costs.group_submission_seconds(),
        })
    }
}

/// What every streamed load prices its experts' decisions on, each input in
/// its state (`assessment::Priced`): a decision whose input is pending runs
/// its policy that needs no cost until the measurement job stores it and the
/// load prices again ([`ExecutorDomain::measure_load`]). Its decode is
/// priced on the generic miss model beside what the load's own decode read
/// ([`ExecutorDomain::decode_price`]).
pub struct ExpertMeasures {
    /// Each streamed layer's lines on this executor, the host's the bank
    /// holds and the GPU's its kernels' costs price
    /// (`assessment::ExpertLines::priced`): the decode is priced along them,
    /// and the route's policy words are written from them. Until measured,
    /// every miss is served on the GPU's path, and nothing grows.
    pub lines: crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>,
    /// The model store's read line the decode's file misses take, the
    /// measurement job's (`assessment::StreamedFigures`), which also prices
    /// a stored table's records.
    pub read: crate::assessment::Priced<crate::assessment::ReadLine>,
    /// What a routed pass over each streamed layer's experts costs on this
    /// executor, which prices a prompt group's compute by its size
    /// (`GroupReads::grant`), and before the load measured a group, the
    /// routed seconds its places save (`ExecutorDomain::group_seconds`):
    /// until measured, a measured group's places weigh its reads alone, and
    /// before a group is measured its places are not priced.
    pub pass: crate::assessment::Priced<crate::assessment::ExpertPass>,
    /// Each bank group's shrink move line on this executor, at its own
    /// record form and moves (`MoveCosts`): what a bank step's exchange is
    /// charged for its moves before a resize of the bank was timed
    /// (`bank_heap`).
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

    /// The storage `holding` of the load's expert bank lies in, which its
    /// resizes are of (`ElasticBank::holding_storage`).
    pub fn expert_holding_storage(&self, holding: BankHolding) -> Result<crate::planning::BankTier, DomainError> {
        self.family
            .expert_bank()
            .ok_or_else(|| DomainError::invariant(format!("{holding:?} of a load without an expert bank")))?
            .holding_storage(holding)
    }

    /// What the load's weight cap bounds, by the owner of each byte
    /// (`ModelLoadPlan::weight_owners`).
    pub fn weight_owners(&self) -> Result<Vec<(&'static str, u64)>, String> {
        self.execution.load().weight_owners()
    }

    /// From now on, price the load by `measures`, and measure what its
    /// decode does of its experts by them, in place of what it was priced
    /// by before: its warm-up has ended, or the executor's stored costs have
    /// changed since. Between flights (the bindings prove none is on the
    /// device), so every step is recorded and priced by the measures it ran
    /// under, and the bank's device-side pricing changes under no step in
    /// flight. Its one fallible part, the tier copy's grid, is written
    /// before anything is priced: a refusal is the caller's error and
    /// changes nothing. Measures of experts a load without a bank holds, or
    /// none of a bank's, are an invariant violation.
    ///
    /// The load's serving rates start at the prior the first measures with
    /// a step price (`serving_rates`): a load priced while its costs were
    /// pending takes no room for what they would price, and takes it once
    /// it is priced again on them.
    pub fn measure_load(
        &mut self,
        measures: LoadMeasures,
        _between_flights: &mut StateBindings,
    ) -> Result<(), DomainError> {
        match (&measures.experts, self.family.expert_bank()) {
            (None, None) => {}
            (Some(experts), Some(bank)) => {
                tracing::info!(
                    read = experts.read.why().unwrap_or("measured"),
                    lines = experts.lines.why().unwrap_or("measured"),
                    pass = experts.pass.why().unwrap_or("measured"),
                    moves = experts.moves.why().unwrap_or("measured"),
                    tier_copy = experts.tier_copy.why().unwrap_or("measured"),
                    convert = experts.convert.why().unwrap_or("measured"),
                    "the experts are priced"
                );
                if let Some(Some(costs)) = experts.tier_copy.measured() {
                    bank.tier_copy(costs)?;
                }
                bank.price(experts.lines.measured().cloned(), experts.pass.measured().cloned());
                if let Some(Some(lines)) = experts.convert.measured() {
                    bank.device_convert(lines);
                }
            }
            (None, Some(_)) => return Err(DomainError::invariant("a banked load priced without its experts' measures")),
            (Some(_), None) => return Err(DomainError::invariant("expert measures of a load without a bank")),
        }
        tracing::info!(costs = measures.costs.as_ref().err().map_or("measured", String::as_str), "the load is priced");
        self.measures = Some(measures);
        *self.places_prompt.get_mut() = None;
        self.settled = None;
        Ok(())
    }

    /// The load's costs; none before [`Self::measure_load`], or where
    /// nothing keeps them.
    fn load_costs(&self) -> Option<&LoadCosts> {
        self.measures.as_ref().and_then(|measures| measures.costs.as_ref().ok())
    }

    /// The measures of the load's experts; none before
    /// [`Self::measure_load`], or without a bank.
    fn expert_measures(&self) -> Option<&ExpertMeasures> {
        self.measures.as_ref().and_then(|measures| measures.experts.as_ref())
    }

    /// Measure the host's landings slow from now on
    /// (`Shared::slow_host_landing`).
    #[cfg(test)]
    pub(crate) fn slow_host_landing(&self) {
        if let Some(bank) = self.family.expert_bank() {
            bank.slow_host_landing();
        }
    }

    /// The readings of the heap's charge taken so far.
    #[cfg(test)]
    pub(crate) fn memory_refreshes(&self) -> usize {
        self.refreshes.get()
    }

    /// The device the domain charges.
    #[cfg(test)]
    pub(crate) fn device(&self) -> &seismic::Device {
        self.domain.device()
    }

    /// The landings the device has converted.
    #[cfg(test)]
    pub(crate) fn device_landed(&self) -> u64 {
        self.family.expert_bank().map_or(0, |bank| bank.device_landed())
    }

    /// The decode steps begun while the bank was whole, which ran its
    /// resident forms.
    #[cfg(test)]
    pub(crate) fn resident_decodes(&self) -> u64 {
        self.family.expert_bank().map_or(0, |bank| bank.resident_decodes())
    }

    /// Each streamed layer's lines in their state; none before
    /// [`Self::measure_load`].
    pub fn expert_lines(&self) -> Option<&crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>> {
        self.expert_measures().map(|measures| &measures.lines)
    }

    /// The load's decode step as its experts' decisions price it now
    /// ([`crate::assessment::DecodePrice`]): its variant's routing
    /// calibrated by the decode it counted, at that decode's mean rows and
    /// context, on the step of those rows there in each form; before it
    /// counted a step, the generic model's routing alone, at one row at the
    /// start of its context (`DecodePrice::prior`), so every decision is
    /// priced from the load's first boundary. None before its lines, read
    /// line and step are measured.
    pub(crate) fn decode_price(&self) -> Result<Option<crate::assessment::DecodePrice>, DomainError> {
        let (Some(measures), Some(bank)) = (self.measures.as_ref(), self.family.expert_bank()) else {
            return Ok(None);
        };
        let Some(experts) = measures.experts.as_ref() else {
            return Ok(None);
        };
        let measured = (experts.lines.measured(), experts.read.measured(), measures.costs.as_ref().ok().map(|costs| &costs.step));
        let (Some(lines), Some(&read), Some(step)) = measured else {
            return Ok(None);
        };
        let step = |rows: u32, depth: u32| {
            step(rows, depth).map_err(|error| DomainError::invariant(format!("the decode step: {error}")))
        };
        let layers = bank.streamed_layers(lines.clone());
        Ok(Some(match bank.live() {
            Some(live) => {
                let compute = step(live.rows.round() as u32, live.depth)?;
                crate::assessment::DecodePrice::new(layers, live, read, compute)
            }
            None => crate::assessment::DecodePrice::prior(layers, 1., read, step(1, 0)?),
        }))
    }

    /// What a routed pass over each streamed layer's experts costs, in its
    /// state; none before [`Self::measure_load`], or without a bank.
    pub fn expert_pass(&self) -> Option<&crate::assessment::Priced<crate::assessment::ExpertPass>> {
        self.expert_measures().map(|measures| &measures.pass)
    }

    /// The epoch a launch submitted now is timed across ([`LaunchEpoch`]);
    /// none while an expert bank resize runs.
    fn launch_epoch(&self) -> Option<LaunchEpoch> {
        (!self.resize_held.runs()).then(|| LaunchEpoch {
            graphs: self.domain.device().graph_replays().recorded,
            resizes: self.resize_held.begun,
            fills: self.family.expert_bank().map(|bank| bank.fill_epoch()),
        })
    }

    /// Whether a launch submitted at `epoch` finishing now was timed alone,
    /// and whether the bank's fill ran beside it.
    fn timed(&self, epoch: Option<LaunchEpoch>) -> Timed {
        match epoch {
            Some(epoch) if self.launch_epoch() == Some(epoch) && epoch.filling() => Timed::Filling,
            Some(epoch) if self.launch_epoch() == Some(epoch) => Timed::Settled,
            _ => Timed::Untimed,
        }
    }
}

/// Whether a completed step of `work` has its decode counted by a load
/// measuring its experts by `experts`: not before
/// [`ExecutorDomain::measure_load`], only a banked load's, and only generation's (a prompt's
/// rows, even in the decode form, are no decode step its decisions price).
fn recording(experts: Option<&ExpertMeasures>, work: crate::operation::LaunchWork) -> Option<crate::experts::Recording> {
    experts.filter(|_| work == crate::operation::LaunchWork::Generation).map(|_| crate::experts::Recording)
}

/// The expert bank's resizes: how many began so far, and whether one runs
/// now. A reading equal across a span with none running at either end saw
/// no resize in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertResizes {
    pub begun: u64,
    pub running: bool,
}

/// The heap's claim of a running expert bank growth: the bytes it claimed of `kind` storage,
/// of which the claim holds what the growth has not backed yet (`refresh_memory`).
#[derive(Clone, Copy, Debug)]
struct GrowthClaim {
    id: ClaimId,
    kind: crate::planning::BankTier,
    bytes: u64,
}

/// What reads memory for the domain: the device's stores and ledgers, its
/// heap, the memory pace, the heap claim of the bank's running growth, and
/// whether a bank resize runs. None of the program family's, which each
/// reading is handed its expert bank from: a chain's encoding reads memory
/// through it while it holds the family
/// ([`ExecutorDomain::submit_chain`]).
struct MemoryProbe<'d> {
    domain: &'d ResourceDomain,
    memory: &'d RefCell<DeviceHeap>,
    pace: &'d RefCell<MemoryPace>,
    /// [`ExecutorDomain::unseen_reading`].
    unseen: &'d RefCell<Option<Result<Vec<DomainReading>, DomainError>>>,
    bank_claim: Option<&'d GrowthClaim>,
    resizing: bool,
    /// [`ExecutorDomain::planned_unheld`], read when the probe was taken.
    planned: u64,
}

impl<'d> MemoryProbe<'d> {
    /// The domain's probe, of its parts apart from the program family.
    fn of(
        domain: &'d ResourceDomain,
        memory: &'d RefCell<DeviceHeap>,
        pace: &'d RefCell<MemoryPace>,
        unseen: &'d RefCell<Option<Result<Vec<DomainReading>, DomainError>>>,
        bank_claim: Option<&'d GrowthClaim>,
        held: &ResizeHeld,
        planned: u64,
    ) -> Self {
        Self { domain, memory, pace, unseen, bank_claim, resizing: held.runs(), planned }
    }

    /// [`ExecutorDomain::observe_memory`], beside `bank`.
    fn observe(&self, bank: Option<&dyn ElasticBank>) -> Result<Vec<DomainReading>, DomainError> {
        let _rises = bank.map(|bank| bank.hold_rises());
        let readings = self
            .memory
            .borrow_mut()
            .refresh(self.planned)
            .map_err(|error| DomainError::Blind(error.to_string()))?;
        self.settle_growth_claim(bank)?;
        Ok(readings)
    }

    /// [`ExecutorDomain::refresh_memory`], beside `bank`.
    fn account(&self, bank: Option<&dyn ElasticBank>) -> Result<(), DomainError> {
        let _rises = bank.map(|bank| bank.hold_rises());
        self.memory.borrow_mut().account();
        self.settle_growth_claim(bank)
    }

    /// Take what `bank`'s running growth backed out of its claim: the
    /// charge just read holds those bytes, so the claim holds only the rest,
    /// read under the same hold of the bank's rises
    /// ([`ElasticBank::hold_rises`]), so a growth's bytes are counted once.
    /// The load's holding itself is the device's charge, which Seismic
    /// keeps; nothing here counts it again.
    fn settle_growth_claim(&self, bank: Option<&dyn ElasticBank>) -> Result<(), DomainError> {
        if let (Some(claim), Some(bank)) = (self.bank_claim, bank) {
            let rest = claim.bytes.saturating_sub(bank.growth_backed(claim.kind)?);
            self.memory
                .borrow_mut()
                .reduce_reserved(claim.id, claim.kind, rest)
                .map_err(|error| DomainError::invariant(format!("a growth's claim: {error:?}")))?;
        }
        Ok(())
    }

    /// [`ExecutorDomain::probe_memory`], beside `bank`: a reading taken now.
    fn probe(&self, bank: Option<&dyn ElasticBank>) -> Result<(), DomainError> {
        let readings = self.observe(bank)?;
        self.observed(bank, readings, || self.memory.borrow().footprint(), Instant::now())
    }

    /// [`Self::probe`] of the periodic reading published since the last
    /// (`DeviceHeap::refresh_published`), read off this thread, its outcome
    /// held for the owner until it takes it
    /// ([`ExecutorDomain::take_published_memory`]): none where none was
    /// published since.
    fn probe_published(&self, bank: Option<&dyn ElasticBank>) -> Option<Result<(), DomainError>> {
        let _rises = bank.map(|bank| bank.hold_rises());
        let (readings, footprint, at) = self.memory.borrow_mut().refresh_published(self.planned)?;
        let observed = readings
            .map_err(|error| DomainError::Blind(error.to_string()))
            .and_then(|readings| {
                self.settle_growth_claim(bank)?;
                self.observed(bank, readings.clone(), || footprint, at).map(|()| readings)
            });
        let outcome = observed.as_ref().map(drop).map_err(Clone::clone);
        *self.unseen.borrow_mut() = Some(observed);
        Some(outcome)
    }

    /// The readings `readings` taken at `at`, the process's footprint then
    /// `footprint`: the memory pace is measured from them, those taken while
    /// no bank resize runs settled ones.
    fn observed(&self, bank: Option<&dyn ElasticBank>, readings: Vec<DomainReading>, footprint: impl FnOnce() -> Option<u64>, at: Instant) -> Result<(), DomainError> {
        let host = self.host_witness(&readings, footprint);
        let releasable = bank.map(releasable).transpose()?.unwrap_or_default();
        let release = self.memory.borrow().release_rate();
        self.pace.borrow_mut().observe(
            &readings,
            &self.domain.device().ledgers().own_claims(),
            host,
            !self.resizing,
            releasable,
            release,
            at,
        );
        // Every fit from the next reading on holds the line at the motion
        // and release latency measured now.
        let motions = self.pace.borrow().motions();
        self.memory.borrow_mut().measure(motions.host, motions.device, motions.latency, motions.release, releasable);
        match readings.into_iter().find(|reading| reading.band == MemoryBand::Reclaim) {
            Some(reading) => Err(DomainError::Reclaim(reading)),
            None => Ok(()),
        }
    }

    /// Host RAM's witness of what the process holds resident at a probe that read
    /// `readings` (`MemoryPace`): what the system counts the process holding
    /// out of free memory, which falls by a release as free memory rises by
    /// it. A Metal device's allocations are wired through its residency set,
    /// so the host's wired pages (`seismic::HostHeld`): a released buffer
    /// leaves the footprint at once, but its pages are free only once the
    /// driver unwires them at its next tick (on an M4 Pro, 0.47 s after a
    /// 138 MB slot's release free and wired memory moved together; the
    /// first slot a reserved buffer released was not unwired within 3 s).
    /// The count is the host's, the process's own wired size counting none
    /// of the residency set's: another program wiring memory in the span
    /// holds a release unseen, credited until the host takes more than it,
    /// and one unwiring shows it early. Any other device's allocations are
    /// pages of the process, so its footprint (`seismic::ProcessMemory`), which the
    /// system frees as they leave it (a GB10's fell by each bank slot within
    /// its release). None where the system reports neither.
    fn host_witness(&self, readings: &[DomainReading], footprint: impl FnOnce() -> Option<u64>) -> Option<u64> {
        match self.domain.device().backend() {
            seismic::BackendName::Metal => readings
                .iter()
                .find(|reading| reading.identity == seismic::StoreIdentity::HostRam)
                .and_then(|reading| reading.held)
                .map(|held| held.wired_bytes),
            _ => footprint(),
        }
    }

    /// The store `kind` storage lies in.
    fn store_of(&self, kind: seismic::Contract) -> Option<seismic::StoreId> {
        let device = self.domain.device();
        state::storage_store(device.topology(), device.info(), kind)
    }

    /// What the process charges `store` now: its own claims there.
    fn charged_in(&self, store: seismic::StoreId) -> u64 {
        self.domain
            .device()
            .ledgers()
            .own_claims()
            .into_iter()
            .filter(|(of, _)| *of == store)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// What `store` is short of now: its level's shortfall
    /// (`MemoryPace::shortfall`), against the process's charge of it now.
    /// While the one store's bank holds borrowed room, host RAM's level
    /// reads what the OS can obtain with it counted (`platform::Serving`):
    /// what an ask takes of it is short like any.
    fn shortfall(&self, store: seismic::StoreId) -> Shortfall {
        self.pace.borrow().shortfall(store, self.charged_in(store))
    }

    /// What the process holds now (`MemoryMoment`).
    fn moment(&self) -> MemoryMoment {
        let ledger = self.domain.device().device_ledger();
        MemoryMoment {
            charged: ledger.charged,
            claimed: ledger.claim,
            footprint: seismic::ProcessMemory::read().footprint_bytes,
        }
    }

    /// What the expert bank's storage is short of now, read between a
    /// chain's submissions (`ExpertBank::release_running`): the periodic
    /// reading published since the last ([`Self::probe_published`]), taken
    /// without waiting, and the shortfall it leaves the bank's store
    /// ([`Self::shortfall`]: the level's, or the borrowed room the bank owes
    /// back on displacement now or for the load's claims), with what the
    /// process held then. None
    /// while a bank resize runs (its advance is a boundary's, and the bank
    /// begins no cut beside it), with no reading published since, where it is
    /// blind (the heap holds it so, and the owner's next probe reads again),
    /// or where nothing is short.
    fn running_pressure(&self, bank: &dyn ElasticBank) -> Option<RunningPressure> {
        if self.resizing {
            return None;
        }
        if let None | Some(Err(DomainError::Blind(_))) = self.probe_published(Some(bank)) {
            return None;
        }
        let kind = bank.holding_storage(BankHolding::Experts).expect("an expert bank holds its experts");
        let store = self.store_of(kind.storage())?;
        match self.shortfall(store) {
            Shortfall::Short(bytes) => Some(RunningPressure { bytes, held: self.moment() }),
            Shortfall::Above | Shortfall::Covered => None,
        }
    }
}

/// The time expert bank resizes held the submitting thread at flight
/// boundaries: each store's running resize's so far (none while none of it
/// runs), and each holding's last finished one's in all, which is what its
/// growth step costs the decode (none before one of it finished).
#[derive(Default)]
struct ResizeHeld {
    running: [Option<ThreadHeld>; crate::planning::BankTier::COUNT],
    last: std::collections::HashMap<BankHolding, f64>,
    /// The resizes begun.
    begun: u64,
}

/// What one resize held the submitting thread, by source: the call that
/// began it, each flight boundary's advance of the bank while it ran, and
/// each decision taken while it ran that began nothing (a shrink found its
/// store resizing, deferred or at its floor), each with its count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreadHeld {
    /// When the call that began it started: `ResizeTimes::decided` less
    /// this is what deciding it held the thread, before the bank was asked.
    pub started: Instant,
    /// Seconds the call that began it held the thread.
    pub begun: f64,
    /// Seconds and count of the boundaries' advances while it ran, the one
    /// that finished it among them.
    pub advances: HeldBy,
    /// Seconds and count of the decisions that began nothing while it ran.
    pub decisions: HeldBy,
}

/// Seconds a source held the thread over `count` calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeldBy {
    /// The calls.
    pub count: u64,
    /// The seconds they held the thread in all.
    pub seconds: f64,
}

impl HeldBy {
    fn add(&mut self, seconds: f64) {
        self.count += 1;
        self.seconds += seconds;
    }
}

impl ThreadHeld {
    /// The seconds it held the thread in all.
    pub fn total(&self) -> f64 {
        self.begun + self.advances.seconds + self.decisions.seconds
    }
}

/// What held the thread beside a running resize ([`ResizeHeld::hold`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeldSource {
    Advance,
    Decision,
}

impl ResizeHeld {
    /// A resize of `tier` storage began, holding the thread since `started`.
    fn begin(&mut self, tier: crate::planning::BankTier, started: Instant) {
        self.begin_held(tier, started, started.elapsed());
    }

    /// A resize of `tier` storage began, its call starting at `started` and
    /// holding the thread `held`: a cut begun beside a running chain, whose
    /// asking held it only that while the chain was encoded.
    fn begin_held(&mut self, tier: crate::planning::BankTier, started: Instant, held: Duration) {
        assert!(self.running[tier.index()].is_none(), "a {tier:?} resize began beside a running one");
        let (advances, decisions) = Default::default();
        self.running[tier.index()] = Some(ThreadHeld { started, begun: held.as_secs_f64(), advances, decisions });
        self.begun += 1;
    }

    /// Each running resize held the thread since `started` by `source`: the
    /// boundary's work is every running one's.
    fn hold(&mut self, started: Instant, source: HeldSource) {
        let held = started.elapsed().as_secs_f64();
        for running in self.running.iter_mut().flatten() {
            match source {
                HeldSource::Advance => running.advances.add(held),
                HeldSource::Decision => running.decisions.add(held),
            }
        }
    }

    /// The running resize of `tier` storage, of `holding`, finished at an
    /// advance holding the thread since `started`: what it held the thread.
    fn finish(&mut self, started: Instant, tier: crate::planning::BankTier, holding: BankHolding) -> ThreadHeld {
        self.hold(started, HeldSource::Advance);
        let held = self.running[tier.index()].take().expect("a finished resize began");
        self.last.insert(holding, held.total());
        held
    }

    /// Whether a resize runs.
    fn runs(&self) -> bool {
        self.running.iter().any(Option::is_some)
    }

    /// Whether a resize of `tier` storage runs.
    fn runs_in(&self, tier: crate::planning::BankTier) -> bool {
        self.running[tier.index()].is_some()
    }
}

/// What the rest of the host does to each memory domain, measured from the
/// owner's memory probes (`ExecutorDomain::probe_memory`, one every reading
/// interval and one after each release while memory is short), and what
/// sizes the expert bank's resizes from them.
///
/// Memory available to the process is its headroom and its own charge. A
/// release drops the charge at once; the host may show it in headroom only
/// later. The process knows its own releases exactly, and a witness says
/// when the host shows them: a count that falls by a release as headroom
/// rises by it, and that the rest of the host's demand for free memory moves
/// none of, but where it wires memory itself (host RAM's,
/// `ExecutorDomain::host_witness`; a device's own memory shows a release
/// with its charge). What the process released that
/// the witness has yet to drop is unseen ([`DomainPace::unseen`]), and
/// available memory, counting it, rises only as a release shows and falls
/// only as the host takes. So no lag makes a release answer the same
/// pressure twice, and a release that lands in a span in which the host
/// takes as much is seen, not credited: a shrink
/// covers what the reading is below the planning line, plus what the host
/// takes until the next shrink takes effect, less what is unseen and what the
/// process released since the reading. What that covers is awaited, not
/// released again ([`Shortfall::Covered`]), however long the host takes to
/// show it: the pace alone decides when a release has answered its deficit,
/// and no clock releases for it a second time. A release the witness never
/// drops stays credited while memory is short; more taken than is credited
/// is short past it, so pressure still escalates down the ladder to the
/// model's unload. Where no witness is read, a
/// release is taken as shown at the reading after it. A span across a fall
/// of the charge measures no taking.
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
/// fluctuation fit, to every expert where memory allows. A step's room is
/// the free headroom's ceiling (`DomainThresholds::ceiling`), at
/// most the headroom's: growth takes nothing it would have to displace
/// another program's data for, and the band it must not re-enter is the
/// headroom's.
struct MemoryPace {
    domains: BTreeMap<seismic::StoreId, DomainPace>,
    /// When memory was last read; none before the first probe.
    read: Option<Instant>,
    /// The last finished release's latency; zero before the first.
    latency: Duration,
    /// The slowest finished release's rate: its bytes over its span from
    /// decision to effect; none before one released any.
    release: Option<ReleaseRate>,
}

/// What a load measured of the rest of the host and of its own releases
/// ([`MemoryPace::motions`]).
#[derive(Debug, PartialEq)]
struct PaceMotions {
    host: Option<StoreMotion>,
    device: Option<StoreMotion>,
    latency: Option<Duration>,
    release: Option<ReleaseRate>,
}

/// One domain's pace, as of its latest probe.
struct DomainPace {
    role: DomainRole,
    headroom: u64,
    /// The planning and emergency reserves: the line a load reads Reclaim
    /// at, and the headroom it must not run into.
    thresholds: DomainThresholds,
    charged: u64,
    /// The reading's band: Reclaim at or below the planning line, or while
    /// the host is in distress.
    band: MemoryBand,
    /// What the process released that its witness has yet to drop. None
    /// is owed from a reading above the line.
    unseen: u64,
    /// What the process holds resident in the store at this reading, as its
    /// witness reads it: what a release lowers and what backing its own
    /// charge raises; none in a device's own memory, or where none is read.
    witness: Option<u64>,
    /// What the rest of the host had not taken at each witnessed probe a
    /// span apart (`MemoryPace::observe`'s `untaken`), oldest first, back
    /// to the first at or before the rate's window
    /// ([`DomainPace::sustained`]); without a witness, since the charge
    /// last changed.
    untaken: std::collections::VecDeque<(Instant, u64)>,
    /// The rate the rest of the host's taking sustains over the window a
    /// margin holds a take across, in bytes a second
    /// ([`DomainPace::sustained`]).
    rate: f64,
    /// Available memory's fluctuation over its readings since the domain
    /// last left Reclaim.
    fluctuation: Fluctuation,
    /// Whether the domain is host RAM: its margin is host RAM's
    /// ([`MemoryPace::margins`]).
    host: bool,
    /// What the reading's hard limit leaves a claim where a driver's grant
    /// bounds the domain (Metal's working set); none where none does.
    grant_left: Option<u64>,
    /// The line room borrowed beyond the hold line is held within, where
    /// the bank borrows (`platform::StoreFit::borrowable_bytes`).
    borrowable: Option<u64>,
    /// The spans in which the rest of the host took more than the steady
    /// fluctuation, since `first`: another program's asks; none before
    /// the first.
    takes: Option<Takes>,
    /// The domain's first probe.
    first: Instant,
}

/// Another program's asks a domain's pace has read ([`DomainPace::takes`]):
/// how many, and the sum of each one's rate across its span, whose mean is
/// the rate the host's asks take memory at, which borrowed room is priced
/// by.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Takes {
    count: std::num::NonZeroU32,
    rate_sum: f64,
}

impl Takes {
    /// These asks and one more, at `rate` across its span; the first where
    /// none was read.
    fn with(takes: Option<Self>, rate: f64) -> Self {
        takes.map_or(Self { count: std::num::NonZeroU32::MIN, rate_sum: rate }, |takes| Self {
            count: takes.count.checked_add(1).expect("a load reads fewer than 2^32 asks"),
            rate_sum: takes.rate_sum + rate,
        })
    }

    /// The asks' mean rate across their spans.
    fn mean(&self) -> f64 {
        self.rate_sum / f64::from(self.count.get())
    }
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
    /// The rate the rest of the host sustains taking at `at`, where it has
    /// not taken `now` bytes: the fall from the last of the earlier probes
    /// `untaken` at or before `at − window` (its oldest while it reaches no
    /// further back) to `now`, over the span since that probe or the whole
    /// `window` where that is longer, never negative. The window is
    /// the span a margin holds a take across (`MemoryPace::observe`): a
    /// take that stops within it threatens what a release lands by no more
    /// than its share of the window, and one that sustains it reads as
    /// itself. So a transient fall, however steep across one reading span
    /// (an import's wiring, a compiler service's burst), sets no rate of
    /// its own past the window, while a steady take reads at its rate once
    /// the window holds it.
    fn sustained(untaken: &std::collections::VecDeque<(Instant, u64)>, at: Instant, now: u64, window: Duration) -> f64 {
        let start = at.checked_sub(window);
        let Some(&(then, was)) =
            untaken.iter().rev().find(|(then, _)| start.is_some_and(|start| *then <= start)).or(untaken.front())
        else {
            return 0.0;
        };
        let span = window.max(at.duration_since(then));
        was.saturating_sub(now) as f64 / span.as_secs_f64()
    }

    /// An allocation domain's wait bound (`MemoryPace::wait_bound`): none
    /// in a domain that grows nothing, or while the host takes none of it.
    fn wait_bound(&self) -> Option<f64> {
        (self.role == DomainRole::Allocation && self.rate > 0.0)
            .then(|| self.thresholds.planning_bytes.saturating_sub(self.thresholds.emergency_bytes) as f64 / self.rate)
    }
}

impl MemoryPace {
    /// The pace before any probe or release.
    fn new() -> Self {
        Self {
            domains: BTreeMap::new(),
            read: None,
            latency: Duration::ZERO,
            release: None,
        }
    }


    /// Fold in a probe of every domain, the process charging each store as
    /// `charges` say and host RAM's witness of what it holds resident
    /// reading `host`, `settled` if no bank resize runs, memory read every
    /// [`READING_INTERVAL`], the load able to release `releasable` now at
    /// `release`, read `at`.
    ///
    /// Each domain's take rate is the one sustained over the span its
    /// margin holds a take across (`platform::StoreMotion::margin`): from
    /// noticing it within a reading interval, through the first release's
    /// latency, to the last of the bytes the load can release there
    /// landing at the release rate.
    fn observe(
        &mut self,
        readings: &[DomainReading],
        charges: &[(seismic::StoreId, u64)],
        host: Option<u64>,
        settled: bool,
        releasable: Releasable,
        release: ReleaseRate,
        at: Instant,
    ) {
        self.read = Some(at);
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
            let mut before = self.domains.remove(&reading.domain);
            let shown = reading.headroom_bytes.saturating_add(charged);
            let witness = host.filter(|_| reading.identity == seismic::StoreIdentity::HostRam);
            let unseen = match (reading.band, &before) {
                (MemoryBand::Reclaim, Some(before)) => {
                    let owed = before.unseen + before.charged.saturating_sub(charged);
                    // The witness's fall is what of it shows; all of it
                    // without one.
                    let shown = before.witness.zip(witness).map_or(owed, |(was, now)| was.saturating_sub(now));
                    owed - shown.min(owed)
                }
                _ => 0,
            };
            // What the rest of the host has not taken, for its motion: the
            // headroom and what the process holds there as its witness
            // shows. A reservation is charged before its pages are backed,
            // so the headroom falls as the process backs its own charge
            // (an import's wiring, a bank's first fill); read against the
            // charge, that fall is another program's take, at the rate the
            // process backs pages. A release lowers the witness as the host
            // frees its pages, so it moves nothing here and the history
            // stands across it. With no witness (a device's own memory, a
            // host that reports neither wired bytes nor a footprint) the
            // headroom and the charge, measured as `Fluctuation::read`
            // measures: only across spans of an unchanged charge. Such a
            // span holds no release the host has yet to show begin, and one
            // it shows late within it raises the headroom, so it can read a
            // take low, never invent one; a span across which the charge
            // changed restarts the history.
            let host_ram = reading.identity == seismic::StoreIdentity::HostRam;
            let untaken = witness.map_or(shown, |held| reading.headroom_bytes.saturating_add(held));
            let restarts = witness.is_none() && before.as_ref().is_some_and(|before| before.charged != charged);
            let releasable = match reading.identity {
                seismic::StoreIdentity::HostRam => releasable.host,
                _ => releasable.device,
            };
            let window =
                READING_INTERVAL + self.latency + Duration::from_secs_f64(releasable as f64 / release.bytes_per_second());
            let mut took = None;
            let (untaken_at, rate) = match before.as_mut().map(|before| (before.untaken.back().copied(), before)) {
                _ if restarts => ([(at, untaken)].into(), 0.0),
                Some((Some((then, was)), before)) if at.duration_since(then) >= READING_INTERVAL => {
                    let fall = was.saturating_sub(untaken);
                    took = (fall > before.fluctuation.largest)
                        .then(|| fall as f64 / at.duration_since(then).as_secs_f64());
                    let rate = DomainPace::sustained(&before.untaken, at, untaken, window);
                    let mut history = std::mem::take(&mut before.untaken);
                    history.push_back((at, untaken));
                    // Keep the last at or before the window's start, which
                    // anchors its fall, and everything after it.
                    let start = at.checked_sub(window);
                    while history.len() > 1 && start.is_some_and(|start| history[1].0 <= start) {
                        history.pop_front();
                    }
                    (history, rate)
                }
                Some((Some(_), before)) => (std::mem::take(&mut before.untaken), before.rate),
                Some((None, _)) | None => ([(at, untaken)].into(), 0.0),
            };
            let before_takes = before.as_ref().and_then(|before| before.takes);
            let takes = took.map_or(before_takes, |rate| Some(Takes::with(before_takes, rate)));
            let first = before.as_ref().map_or(at, |before| before.first);
            let fluctuation = match reading.band {
                MemoryBand::Reclaim => Fluctuation::default(),
                MemoryBand::Normal => {
                    let mut fluctuation = before.map(|before| before.fluctuation).unwrap_or_default();
                    fluctuation.read(untaken, charged, settled);
                    fluctuation
                }
            };
            self.domains.insert(
                reading.domain,
                DomainPace {
                    role: reading.role,
                    headroom: reading.headroom_bytes,
                    thresholds: reading.thresholds,
                    charged,
                    band: reading.band,
                    unseen,
                    witness,
                    untaken: untaken_at,
                    rate,
                    fluctuation,
                    host: host_ram,
                    grant_left: (reading.constraint == crate::platform::MemoryConstraint::Grant).then_some(reading.ceiling_bytes),
                    borrowable: reading.borrowable_bytes,
                    takes,
                    first,
                },
            );
        }
    }

    /// What a shrink of what is charged in `store` has to do, decided at the
    /// latest probe with the process charging it `charged` now: that domain's
    /// reading below the planning line, plus what the host takes until the
    /// next shrink takes effect, less what the witness has yet to drop and
    /// the charge dropped since this one. Only a release charged in a domain
    /// lifts its deficit, so each holding shrinks by its own domain's; a
    /// domain no probe observed is above. In distress above the line nothing
    /// it holds is short: a release would only buy what no measure says is
    /// owed, its reading already leaving out what was displaced since the
    /// load began to serve (`HostRoom::Load`).
    fn shortfall(&self, store: seismic::StoreId, charged: u64) -> Shortfall {
        let Some(pace) = self.domains.get(&store) else {
            return Shortfall::Above;
        };
        let below = match (pace.thresholds.planning_bytes.saturating_sub(pace.headroom), pace.band) {
            (0, MemoryBand::Normal) => return Shortfall::Above,
            (0, MemoryBand::Reclaim) => return Shortfall::Covered,
            (below, _) => below,
        };
        let released = pace.unseen + pace.charged.saturating_sub(charged);
        match projected(below, pace.rate, self.horizon()).saturating_sub(released) {
            0 => Shortfall::Covered,
            short => Shortfall::Short(short),
        }
    }

    /// What a growth step in `store` must stay below at the latest probe,
    /// and the margin it leaves: its domain's ceiling, whose planning line
    /// holds the reading's fluctuation ([`Self::margin`]), and that
    /// fluctuation; `None` before a probe of it. Each store's room is its
    /// own: a step in host memory takes nothing of a device's.
    fn growth_room(&self, store: seismic::StoreId) -> Option<(u64, u64)> {
        self.domains
            .get(&store)
            .map(|pace| (pace.thresholds.ceiling(pace.headroom), pace.fluctuation.largest))
    }

    /// What the load measured of the rest of the host's motion
    /// (`platform::StoreMotion`), which its reserves hold their margin at:
    /// host RAM's, a device's own memory's (the largest of its allocation
    /// domains'), each the steady fluctuation of available memory and the
    /// rate the rest of the host sustains taking it at now
    /// ([`DomainPace::sustained`]), none where no domain of it reads a take; and the last finished release's latency, none
    /// before one. A measured motion only ever raises what was known
    /// (`platform::MemoryReserves::measured`), whose take rate starts at
    /// the machine's page-fill rate; and the slowest finished release's
    /// rate, none before one, which only ever lowers what was known (the
    /// machine's page-release rate).
    fn motions(&self) -> PaceMotions {
        let motion = |host: bool| {
            self.domains
                .values()
                .filter(|pace| pace.host == host && (host || pace.role == DomainRole::Allocation))
                .filter(|pace| pace.rate > 0.0)
                .map(|pace| StoreMotion::new(pace.fluctuation.largest, pace.rate))
                .reduce(StoreMotion::with)
        };
        PaceMotions {
            host: motion(true),
            device: motion(false),
            latency: (self.latency > Duration::ZERO).then_some(self.latency),
            release: self.release,
        }
    }

    /// The room a settle of the heap grants into in `store`, the process
    /// charging it `charged` now ([`value::Supply`]); none before a probe
    /// of it.
    ///
    /// Free room is the headroom above the planning line, which holds the
    /// measured margin; what is owed is the shortfall below it
    /// ([`Self::shortfall`]). Borrowed room is what the host gives beyond
    /// that by compressing other programs' data, up to the borrowing line.
    /// Its price, in serving seconds saved per second per byte, is what
    /// holding a byte of it costs another program: each ask it makes waits
    /// for the byte about `1 / t` seconds, `t` the mean rate its asks took
    /// memory at (a release hands it back at the rate the host takes it),
    /// and asks come `takes / elapsed` times a second, so
    /// `takes / (elapsed × t)`; zero while no ask was seen,
    /// where borrowing harms no one. The device's hard limit is the grant
    /// left where a grant bounds the domain. An exchange's gain is earned
    /// over `horizon`, how long the settle's decision stands.
    fn supply(&self, store: seismic::StoreId, charged: u64, horizon: Duration) -> Option<value::Supply> {
        let pace = self.domains.get(&store)?;
        let free = pace.thresholds.ceiling(pace.headroom);
        let owed = match self.shortfall(store, charged) {
            Shortfall::Short(bytes) => bytes,
            Shortfall::Above | Shortfall::Covered => 0,
        };
        let borrowable = pace.borrowable.map_or(0, |line| pace.thresholds.ceiling(line).saturating_sub(free));
        let elapsed = self.read.map_or(0.0, |read| read.duration_since(pace.first).as_secs_f64());
        let borrow_price = pace.takes.map_or(0.0, |takes| {
            f64::from(takes.count.get()) / (elapsed.max(READING_INTERVAL.as_secs_f64()) * takes.mean())
        });
        Some(value::Supply {
            free,
            owed,
            borrowable,
            borrow_price,
            ceiling: pace.grant_left.unwrap_or(u64::MAX),
            horizon,
        })
    }

    /// `store`'s available memory's fluctuation at the latest probe, which
    /// the reserve's margin holds; none before a probe of it.
    fn fluctuation(&self, store: seismic::StoreId) -> u64 {
        self.domains.get(&store).map_or(0, |pace| pace.fluctuation.largest)
    }

    /// The wait bound: how long pressure that arrives as a prompt group
    /// starts may wait for a release, in seconds. The owner's ladder
    /// releases nothing while a group runs (it defers releases while a
    /// chain is held), and only the one store's bank is cut beside it
    /// (`ExpertBank::release_running`), down to its floor at most, so from
    /// the planning line `P` the rest of the host may take what lies
    /// above the domain's emergency reserve `E`
    /// (`DomainThresholds::emergency_bytes`), at the rate `r` the host
    /// sustains taking it at now ([`DomainPace::sustained`]): `W = (P − E) / r`, the
    /// least of any allocation domain. The reserve is the region a load must
    /// not run into, whatever lies past it on the host: the worker's kill
    /// where the OS reports no pressure level, critical pressure and
    /// compression on macOS, failed allocations in a device's own memory.
    /// None while the host takes memory from no allocation domain.
    fn wait_bound(&self) -> Option<f64> {
        self.domains
            .values()
            .filter_map(DomainPace::wait_bound)
            .min_by(f64::total_cmp)
    }

    /// From a shrink's decision to the next one's effect.
    fn horizon(&self) -> Duration {
        READING_INTERVAL + 2 * self.latency
    }

    /// A release of `bytes` finished, as `times` say: its latency, and its
    /// rate over the span its bytes took (`ResizeTimes::releasing`) where it
    /// released any. The margin grows its deficit for the `B / r` runs a
    /// release of all `B` releasable bytes takes; a rate read over a wait
    /// any release takes alike would multiply that wait by every byte of
    /// the bank (a 20 GB bank's margin past a terabyte).
    fn released(&mut self, bytes: u64, times: &ResizeTimes) {
        self.latency = times.latency();
        let seconds = times.releasing().as_secs_f64();
        if bytes > 0 && seconds > 0.0 {
            let rate = ReleaseRate::new(bytes as f64 / seconds);
            self.release = Some(self.release.map_or(rate, |slowest| slowest.with(rate)));
        }
    }
}

/// What `bank` can release now: its slots beyond its floor, each an equal
/// share of its records' charge in every store, host RAM's apart from the
/// device's own memory's.
fn releasable(bank: &dyn ElasticBank) -> Result<Releasable, DomainError> {
    let residency = bank.residency()?;
    let above = u128::from(residency.slots.saturating_sub(residency.floor));
    let slots = u128::from(residency.slots.max(1));
    let share = |bytes: u64| u64::try_from(u128::from(bytes) * above / slots).expect("a share of a charge fits its charge");
    Ok(residency.stores.iter().fold(Releasable::default(), |releasable, (store, bytes)| match store {
        seismic::StoreIdentity::HostRam => Releasable { host: releasable.host + share(*bytes), ..releasable },
        _ => Releasable { device: releasable.device + share(*bytes), ..releasable },
    }))
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

    /// The head's own resident weights: the imports its loader brought into
    /// the residency cache it inherits from the target, as imported
    /// (`ComponentLoader::own_bytes`), none of the target's it found there.
    /// None without a head.
    pub(super) fn head_weight_bytes(&self) -> Result<u64, String> {
        match &self.head_loader {
            None => Ok(0),
            Some(loader) => Ok(loader.own_bytes()?),
        }
    }

    /// The owner's periodic observation: the newest reading published off
    /// its thread observed (`MemoryProbe::probe_published`) where one was
    /// since, and the outcome of the newest one observed, here or between a
    /// chain's submissions, that it has not taken yet; none where there is
    /// none: its readings where it is in no Reclaim band. It waits for no
    /// reading.
    pub fn take_published_memory(&mut self) -> Option<Result<Vec<DomainReading>, DomainError>> {
        self.memory_probe().probe_published(self.family.expert_bank());
        self.unseen_reading.get_mut().take()
    }

    /// Observe every domain the device uses from a reading taken now, set
    /// Seismic's allocation ceiling from the readings, and refresh the
    /// heap's view of the charge ([`Self::refresh_memory`]): a memory
    /// probe's, and what waits on memory another process frees. Seismic
    /// remains the physical charge authority; the heap only keeps the
    /// claims. A failed observation revokes unspent
    /// grants and leaves the heap Blind.
    pub fn observe_memory(&self) -> Result<Vec<DomainReading>, DomainError> {
        self.memory_probe().observe(self.family.expert_bank())
    }

    /// Refresh the heap's view of Seismic's charge after the process's own
    /// claims and releases, reading no store: the stores and Seismic's
    /// ceiling stand as the last observation ([`Self::observe_memory`]) or
    /// claim read them, so the decode's own steps read nothing of the OS.
    pub fn refresh_memory(&self) -> Result<(), DomainError> {
        #[cfg(test)]
        self.refreshes.set(self.refreshes.get() + 1);
        self.memory_probe().account(self.family.expert_bank())
    }

    /// What reads memory for the domain ([`MemoryProbe`]).
    fn memory_probe(&self) -> MemoryProbe<'_> {
        MemoryProbe::of(&self.domain, &self.memory, &self.pace, &self.unseen_reading, self.bank_claim.as_ref(), &self.resize_held, self.planned_unheld())
    }

    /// Submit a chain of target launches and its spare output slot
    /// (`ProgramFamily::submit_target`). Its encoding reads memory between
    /// its submissions ([`MemoryProbe::running_pressure`]), so memory
    /// another program takes while a long chain runs is answered by a cut
    /// of the expert bank's one store beside it
    /// (`ExpertBank::release_running`), not at the chain's end; a cut it
    /// began is a resize begun ([`Self::note_running_cut`]).
    fn submit_chain(
        &mut self,
        chain: Chain<crate::ValidatedTargetLaunch>,
        spare: crate::TargetGraphOutputLease,
    ) -> Result<Chain<F::TargetSubmission>, crate::programs::ChainRefusal> {
        // The family is lent to the chain, so the probe borrows the rest,
        // the plan's unheld holdings read before it is.
        let planned = self.planned_unheld();
        let probe = MemoryProbe::of(&self.domain, &self.memory, &self.pace, &self.unseen_reading, self.bank_claim.as_ref(), &self.resize_held, planned);
        let submitted = self.family.submit_target(chain, spare, &|bank| probe.running_pressure(bank));
        self.note_running_cut();
        submitted
    }

    /// Take a cut of the expert bank begun beside a running chain since
    /// (`ElasticBank::running_begun`): a resize of the experts' storage
    /// begun, which held the submitting thread what its asking did, and
    /// which the owner takes ([`Self::take_running_release`]) before the
    /// advance that finishes it. None begins while another resize runs:
    /// the probe that begins one reads no memory then.
    fn note_running_cut(&mut self) {
        let Some(bank) = self.family.expert_bank() else {
            return;
        };
        let Some(RunningBegun { decided, held, moment }) = bank.running_begun() else {
            return;
        };
        let tier = bank.holding_storage(BankHolding::Experts).expect("an expert bank holds its experts");
        self.resize_held.begin_held(tier, decided, held);
        assert!(self.running_release.replace((tier, moment)).is_none(), "the owner takes a running cut before the next begins");
    }

    /// The cut of the expert bank begun beside a running chain since the
    /// owner last asked, taken: the storage it resizes and what the process
    /// held as it was decided. The owner holds it as a Reclaim shrink it
    /// began, which a later advance finishes
    /// ([`Self::advance_expert_bank`]).
    pub fn take_running_release(&mut self) -> Option<(crate::planning::BankTier, MemoryMoment)> {
        self.running_release.take()
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

    /// Import and bind the optional component `lane` needs when it is
    /// unbound, and whether it did.
    fn bind_component(&mut self, lane: ReservationLane) -> Result<bool, DomainError> {
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
                Ok(true)
            }
            ReservationLane::Vision if !self.family.vision_is_bound() => {
                let resident = self
                    .vision_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("vision component is disabled".into()))?
                    .load()
                    .map_err(|error| DomainError::Input(error.to_string()))?;
                self.family.bind_vision(resident, &self.definition)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Reserve capacity for one group: the front queued step's slots when
    /// the group claims it, otherwise its provisioned state, advances and
    /// launch leases.
    ///
    /// It reads the heap's charge once, on entry. What changes the charge
    /// later on this thread reads it again itself: each growth provisioning
    /// grants (`grant_state_growth`, `grant_group`, `provision_output`,
    /// `grow_vision`) once its allocation ran, and the reservation once an
    /// import ran. Off this thread the charge only rises under the expert
    /// bank's claim, which the heap already holds.
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
        // The import and binding of an unbound component run under one heap
        // claim for their peak, released once Seismic's charge reflects them.
        let claim = self
            .component_growth(operations)?
            .map(|(required, staged)| {
                self.claim_device_growth(required, staged, HoldingClass::Dormant)
            })
            .transpose()?;
        let imported = self.bind_component(requirements.lane).and_then(|lane| {
            Ok(lane | (requirements.priming_banks != 0 && self.bind_component(ReservationLane::Head)?))
        });
        if let Some(claim) = claim {
            self.release_claim(claim);
        }
        if imported? {
            self.refresh_memory()?;
        }
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
            workload: None,
            measurement_claim: None,
            class_reserve: 0,
            next_flight: 0,
            host_turn: None,
            proposed: BTreeSet::new(),
            poisoned: BTreeMap::new(),
            measures: None,
            bank_claim: None,
            resize_held: ResizeHeld::default(),
            shrink_step: false,
            pace: RefCell::new(MemoryPace::new()),
            #[cfg(test)]
            refreshes: std::cell::Cell::new(0),
            running_release: None,
            places_prompt: RefCell::new(None),
            unseen_reading: RefCell::new(None),
            settled: None,
            requests: 0,
            #[cfg(test)]
            settles: std::cell::Cell::new(0),
        };
        let mut domain = domain;
        domain.class_reserve = domain.planned_class_reserve();
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
