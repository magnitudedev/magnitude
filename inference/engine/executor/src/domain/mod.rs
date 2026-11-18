//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Demand, Draw, DrawKind, Row, RowHistory, Select, Shaping as RowShaping, Slot,
    ValidatedTargetBatch,
};
use crate::platform::DomainReading;
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
    InFlightState, OwnedAdvanceResolution, OwnedChainAdvance, OwnedStateAdvance,
    OwnedTailRelocation, RowDemand, SequenceState, StateCheckpoint, StateStore, StoreBindings,
    TentativeAdvance,
};
use seismic::Tensor;
mod draft;
mod family;
mod features;
mod head;
mod reader;
mod serving_heap;
pub use serving_heap::{Followed, Yielded};
mod in_flight;
mod input;
mod ownership;
mod queue;
pub use queue::Unchained;
mod reconcile;
mod state;
pub(crate) use state::claimed_through;
pub use state::request_slot_bytes;
mod target;
mod vision;

#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
#[cfg(test)]
mod domain_tests;

pub use family::{
    BankHolding, BankResize, BankResized, BankStep, ElasticBank, ExpertResidency, HostResidency,
    NativeFamily,
    ProgramFamily, ResizeTimes, RunningBegun, RunningPressure, TableResidency, OptionalComponent, TrafficHold, Trip,
};
pub use reader::{DeviceReader, LoadRoom, MachineReading, PlanningRoom, StatusReading};
pub use in_flight::{
    HeadFlight, PrimingCompletionFlight, TargetContinuation, TargetFlight, VisionFlight,
};
use in_flight::{PrimingFlight, TargetWork, decode_selected};
pub use ownership::OpenRequirements;
pub use state::{MemoryChargeReconciliation, StoreAllocation};
#[cfg(any(test, feature = "fixtures"))]
pub use state::{SlotRelease, StateShrink, StateYield};
pub use target::TargetHostTiming;
pub(crate) use target::{
    AppendedRuns, accepted_history, cached_causal_domain, indexer_history,
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
    /// A store's ledger refused the claim (`memory::ledger::Short`): it
    /// waits for a change of the process's that frees memory, or fails
    /// where none can come.
    Capacity(crate::CapacityError),
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
    /// A group of a chain's holder that is not the chain's queued chunk
    /// (its round was lowered at other rows than the chain planned): the
    /// chain was released, the holder's state with it, as an eviction's
    /// (`ExecutorDomain::release_chain`). The holder replays from its
    /// committed history; nothing else changed.
    ChainReleased(RequestId),
}

impl DomainError {
    pub(crate) fn invariant(detail: impl Into<String>) -> Self {
        Self::Invariant(crate::InvariantError {
            context: "executor domain",
            detail: detail.into(),
        })
    }

    /// A request's reservation on the device's ledger refused as `short`,
    /// as it opens or grows: a deficit of the memory its state draws on
    /// ([`ResourceKind::RequestState`]).
    pub(crate) fn request_short(short: crate::memory::ledger::Short) -> Self {
        let crate::memory::ledger::Short { required, over, .. } = short;
        Self::Capacity(CapacityError { resource: ResourceKind::RequestState, required, available: required - over })
    }

    /// A claim on `ledgers` refused as `short`: a deficit of that store's
    /// memory.
    pub(crate) fn short(short: crate::memory::ledger::Short, ledgers: &crate::memory::ledger::DeviceLedgers) -> Self {
        let crate::memory::ledger::Short { store, required, over } = short;
        Self::Capacity(CapacityError {
            resource: match store == ledgers.device().store() {
                true => ResourceKind::DeviceMemory,
                false => ResourceKind::HostStaging,
            },
            required,
            available: required - over,
        })
    }
}

impl std::fmt::Display for DomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity(error) => error.fmt(f),
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
            Self::ChainReleased(holder) => write!(f, "request {}'s group is not its chain's queued chunk: the chain was released", holder.0),
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
    /// graph was recorded, nor any expert bank resize ran or cover read
    /// began or ended, within it; and whether the bank's cover read beside
    /// it.
    timed: Timed,
    image: Option<ImageRef>,
}

/// What a launch's duration is its own across: the graphs recorded on the
/// device, by any load (a recording is paid on the host within the
/// duration), read from the device's own count at each submission and
/// finish, the expert bank's resizes begun (a resize holds the thread at
/// the boundaries it runs across, and a growth's new slots or a shrink's
/// evictions change what the launch misses), and its cover's reads
/// (`ElasticBank::cover`). A launch is timed only where the epoch at its
/// finish is the one at its submission, none while a resize runs: a resize
/// changes no launch cost the draft width is chosen by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaunchEpoch {
    graphs: u64,
    resizes: u64,
    /// None without an elastic bank.
    cover: Option<ExpertCover>,
}

/// Whether a launch's duration was its own ([`LaunchEpoch`]), and whether
/// the expert bank's cover read beside it then.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Timed {
    Untimed,
    Settled,
    Covering,
}

/// A launch's duration where it was its own ([`LaunchEpoch`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchDuration {
    pub duration: Duration,
    /// Whether the expert bank's cover read beside the launch: its misses
    /// shared the drive with the cover's reads and found what the cover had
    /// not yet landed, not what the bank settles to, so its duration prices
    /// the load only until the cover reads nothing.
    pub covering: bool,
}

impl LaunchDuration {
    /// `duration`, measured with no cover read beside it.
    pub fn settled(duration: Duration) -> Self {
        Self { duration, covering: false }
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
        let covering = match self.timed {
            Timed::Untimed => return None,
            Timed::Settled => false,
            Timed::Covering => true,
        };
        Some(LaunchDuration { duration: self.physical_duration, covering })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationLane {
    Target,
    Head,
    Vision,
}

impl ReservationLane {
    /// The lane `operation` runs on.
    fn of(operation: &Operation) -> Self {
        match operation {
            Operation::Forward { .. } => Self::Target,
            Operation::Head { .. } => Self::Head,
            Operation::Encode { .. } => Self::Vision,
        }
    }
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
    TargetReads(Vec<InFlightState>),
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

    /// The target store's checkpoint of the sequence.
    pub fn target(&self) -> &magnitude_state::StateCheckpoint {
        &self.target
    }
}

/// Native executor with no erased stage or executor type parameters. The
/// accepted map is vacant while that request's advance is in flight.
pub struct ExecutorDomain<F: ProgramFamily = NativeFamily> {
    execution: Rc<ExecutionPlan>,
    definition: Rc<ModelDefinition>,
    resources: AllocatedResources,
    /// The arena's committed claim, taken as the domain is made and held
    /// for the load: its graph scratch and the target graphs' bound
    /// constants, with groups of one place. A group's places beyond it are
    /// its own class's ([`Self::open_group`]).
    arena: crate::memory::ledger::ArenaClaim,
    /// What the load allocated as it started that no holding names (its
    /// weights, tables, state and programs' storage): held for the load.
    _recorded: Vec<crate::memory::ledger::Claim>,
    /// Each resident request's state on the device's ledger, reserved as
    /// it opens ([`Self::open_state`]), extended ahead of its steps and its
    /// rows drawn within it as it advances ([`Self::provision`]), and
    /// dropped as it closes or is evicted.
    state_claims: BTreeMap<RequestId, crate::memory::ledger::Reservation>,
    /// The priming boundary of the group provisioned last
    /// ([`Self::provision`]), which reserving it takes ([`Self::reserve`]).
    boundary: Option<FeatureRef>,
    /// The load's request slot: one request's opening reservation
    /// ([`request_slot_bytes`]), held for the domain's life. Reservations
    /// draw on it first (`memory::ledger::Ledger::reserve`).
    request_slot: crate::memory::ledger::Claim,
    /// The state slabs and the holdings over them on the device's ledger.
    state_slabs: state::StateSlabs,
    domain: ResourceDomain,
    /// The device's memory reading, carried over from the load's startup.
    memory: RefCell<DeviceReader>,
    target_store: Rc<StateStore>,
    head_store: Option<Rc<StateStore>>,
    family: F,
    head_loader: Option<ComponentLoader<ResidentHead>>,
    vision_loader: Option<ComponentLoader<ResidentVision>>,
    /// The claims of what the bound head allocates beside its imports: its
    /// graph constants and its stagings beyond the floor, released as it
    /// unbinds.
    head_binding: Vec<crate::memory::ledger::Claim>,
    /// The claims of the bound vision component's graph constants, released
    /// as it unbinds.
    vision_binding: Vec<crate::memory::ledger::Claim>,
    /// The bound optional components held as cache while no request is
    /// resident (`ExecutorDomain::release_yielded`), committed again as one
    /// opens (`ExecutorDomain::activate_components`).
    idle_components: Vec<serving_heap::IdleComponent>,
    /// The latest reading's answer ([`MemoryProbe::answer`]): each store's
    /// generation at it, and the response the owner owes.
    answered: RefCell<Answered>,
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
    /// What the load served since it opened, which its cache holdings are
    /// valued over (`ExecutorDomain::value_cache`).
    served: serving_heap::Served,
    /// The service's measurement job's committed claim beside the model
    /// (`ExecutorDomain::claim_measurement_room`).
    measurement_claim: Option<state::MeasurementClaim>,
    /// The committed claims of the prompt-group class held beyond one
    /// place (`AllocatedResources::group_places`), from the group's open to
    /// its close ([`Self::open_group`], [`Self::close_group`]): its
    /// device-side bytes on the device's ledger and its host bytes on host
    /// RAM's. None while groups hold one place.
    group_class: Option<state::GroupClass>,
    /// A closed group's class's bytes, the idle graph slots its pools hold
    /// beyond one place until they release them: cache
    /// (`memory::ledger::CacheKind::GraphSlots`), turned from its claims as
    /// it closed ([`Self::close_group`]): handed to the slots' release as
    /// it begins, and freed with them as it ends
    /// (`AllocatedResources::release_idle_slots`), or let go as a group's
    /// class takes the slots again ([`Self::open_group`]). What claims took
    /// of it in process releases until then ([`Self::idle_slots_taken`]).
    idle_class: Vec<crate::memory::ledger::Cached>,

    /// Identity of the next target flight.
    next_flight: u64,
    host_turn: Option<in_flight::HostTurn>,
    /// Requests whose drafter proposed since their last target launch: that
    /// launch is their round's verification, however many proposals it
    /// verifies, and queues no decode behind it (`queue::continuations`).
    proposed: BTreeSet<RequestId>,
    /// Requests whose next round prepares to draft from a separate draft's
    /// fused taps: their next target launch's decode continuation publishes
    /// its feature (`queue::continuations`, `Self::publish_behind`).
    publishing: BTreeSet<RequestId>,
    /// Sequences a step of which could not read its weights: every later
    /// outcome of theirs fails (`Outcome::WeightReadFailed`) until they
    /// close.
    poisoned: BTreeMap<RequestId, crate::WeightReadError>,
    /// What the load is priced on once its warm-up ended
    /// ([`Self::measure_load`]); none before, so a warm-up's steps are not
    /// counted.
    measures: Option<LoadMeasures>,
    /// What expert bank resizes held the submitting thread.
    resize_held: ResizeHeld,
    /// A bank shrink began and its step, the next decode launch, has not
    /// launched: where decode graphs are recorded ahead (CUDA's), that
    /// launch records no graph, as every one it replays was recorded and
    /// kept as its slots were made; elsewhere it records its graphs at
    /// first use, as any new shape's.
    shrink_step: bool,
    /// A cut of the expert bank begun beside a running chain, until the
    /// owner takes it ([`Self::take_running_release`]): the storage it
    /// resizes.
    running_release: Option<crate::planning::BankTier>,
    /// The thread it is made on owns its memory for its life: an owner's
    /// change of the scope begun on another panics.
    _owner: crate::memory::ledger::Owner,
    /// The domain's teardown, one change from its drop until this, its
    /// last field, drops: every free and record its fields make dropping is
    /// part of it.
    teardown: crate::memory::ledger::OpenChange,
}

/// A gap between flights the domain saw: no step of the expert bank runs
/// (`ElasticBank::in_gap`), so a group's places may be handed to it
/// (`ProgramFamily::hold_group`). Only the domain makes one
/// ([`ExecutorDomain::quiet`]); steps begin only on its thread, so one it
/// holds stays true until it begins a step.
#[derive(Clone, Copy, Debug)]
pub struct Quiet(());

impl<F: ProgramFamily> Drop for ExecutorDomain<F> {
    /// The queued steps are the one device work the domain owns: each is
    /// waited for and dropped before any field, so its waits on a streamed
    /// load's host are answered by a watcher that still serves (the
    /// family's bank, whose watcher answers nothing once it stops, drops
    /// with the fields). A queued step that faulted is nobody's outcome:
    /// the device is past it either way.
    fn drop(&mut self) {
        self.teardown = crate::memory::ledger::OpenChange::begin(self.resources.ledgers().changes());
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
}

impl LoadCosts {
    /// The costs of a load of `definition` planned as `planned`, as
    /// `resident` before any bank, on `costs`: its launch in each form
    /// (`assessment::decode_step`). The one derivation every load is
    /// priced by.
    pub fn of(
        definition: &magnitude_family_contracts::ModelDefinition,
        resident: &crate::ExecutionPlanDraft,
        planned: &crate::ExecutionPlanDraft,
        costs: &crate::assessment::ExecutorCosts,
    ) -> Result<Self, crate::assessment::AssessmentError> {
        Ok(Self {
            step: crate::assessment::decode_step(definition, resident, planned, costs.clone())?,
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
    /// every miss is served on the GPU's path, and the experts' slots target
    /// their ceiling (`ElasticBank::targets`).
    pub lines: crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>,
    /// The model store's read line the decode's file misses take, the
    /// measurement job's (`assessment::StreamedFigures`), which also prices
    /// a stored table's records.
    pub read: crate::assessment::Priced<crate::assessment::ReadLine>,
    /// What a routed pass over each streamed layer's experts costs on this
    /// executor, which prices a prompt group's compute by its size and the
    /// passes its launches merge (`Stages::merge_sets`): until measured, a
    /// group merges by its reads alone.
    pub pass: crate::assessment::Priced<crate::assessment::ExpertPass>,
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

    /// [`Self::graph_replays`] by family.
    pub fn family_graph_replays(&self) -> Vec<(&'static str, seismic::GraphReplays)> {
        self.family.family_graph_replays()
    }

    /// The expert bank's cover ([`ExpertCover`]) as now, which its
    /// residency carries too (`ExpertResidency::cover`); none without an
    /// elastic bank. It moves as the bank's watcher reads, with no holding
    /// moving: the census publishes it on its own change.
    pub fn expert_cover(&self) -> Option<ExpertCover> {
        self.family.expert_bank().map(|bank| bank.cover())
    }

    /// Wait until the expert bank's cover reads nothing
    /// (`ElasticBank::wait_covered`); at once without a bank.
    pub fn wait_expert_cover(&self) -> Result<(), DomainError> {
        self.family.expert_bank().map_or(Ok(()), |bank| bank.wait_covered())
    }

    /// The expert bank's resizes so far ([`ExpertResizes`]).
    pub fn expert_resizes(&self) -> ExpertResizes {
        let held = &self.resize_held;
        ExpertResizes { begun: held.begun, running: held.runs(), advances: held.advances }
    }

    /// The expert bank's targets now ([`ExpertTargets`]); none without a
    /// bank.
    pub fn expert_targets(&self) -> Result<Option<ExpertTargets>, DomainError> {
        let Some(bank) = self.family.expert_bank() else {
            return Ok(None);
        };
        let targets = self.derived_targets(bank)?;
        let of = |holding: BankHolding| {
            targets.iter().find(|target| target.holding == holding).map(|target| target.target)
        };
        let slots = of(BankHolding::Experts).expect("invariant violated: an expert bank without its experts' target");
        Ok(Some(ExpertTargets { slots, host: of(BankHolding::Host) }))
    }

    /// `read` of each of `bank`'s holdings beside its derived target, at
    /// the decode's price and the ledgers' room now
    /// (`experts::derived::derive`): the one derivation the bank follows
    /// (`Self::next_bank_move`) and the domain reports
    /// ([`Self::derived_targets`]).
    fn derive_bank<R>(
        &self,
        bank: &dyn ElasticBank,
        read: impl FnOnce(&[(&crate::experts::derived::HoldingTarget<'_>, u64)]) -> R,
    ) -> Result<R, DomainError> {
        let holdings = bank.targets(self.decode_price()?.as_ref())?;
        Ok(read(&crate::experts::derived::derive(self.resources.ledgers(), &holdings)))
    }

    /// Each of `bank`'s holdings beside its derived target, as the bank
    /// follows it ([`Self::derive_bank`]).
    fn derived_targets(
        &self,
        bank: &dyn ElasticBank,
    ) -> Result<Vec<crate::experts::derived::DerivedTarget>, DomainError> {
        let ledgers = self.resources.ledgers();
        self.derive_bank(bank, |derived| {
            derived
                .iter()
                .map(|(target, slots)| crate::experts::derived::DerivedTarget {
                    holding: target.holding,
                    store: ledgers.of(target.tier).store(),
                    held: target.at,
                    target: *slots,
                    block: target.shape.block_bytes(),
                })
                .collect()
        })
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

    /// The device the domain charges.
    #[cfg(test)]
    pub(crate) fn device(&self) -> &seismic::Device {
        self.domain.device()
    }

    /// The arena's committed claim beside its terms: the claim's bytes,
    /// what its allocations charged and the target graphs' bound constants
    /// held now.
    #[cfg(test)]
    pub(crate) fn arena_terms(&self) -> (u64, u64, u64) {
        let constants = self.family.target_constant_bytes().expect("the target's constants fit u64");
        (self.arena.bytes(), self.arena.allocated(), constants)
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

    /// A gap between flights ([`Quiet`]): none while a step of the expert
    /// bank runs; always without a bank.
    fn quiet(&self) -> Option<Quiet> {
        self.family.expert_bank().is_none_or(|bank| bank.in_gap()).then_some(Quiet(()))
    }

    /// The epoch a launch submitted now is timed across ([`LaunchEpoch`]);
    /// none while an expert bank resize runs.
    fn launch_epoch(&self) -> Option<LaunchEpoch> {
        (!self.resize_held.runs()).then(|| LaunchEpoch {
            graphs: self.domain.device().graph_replays().recorded,
            resizes: self.resize_held.begun,
            cover: self.expert_cover(),
        })
    }

    /// Whether a launch submitted at `epoch` finishing now was timed alone,
    /// and whether the bank's cover read beside it.
    fn timed(&self, epoch: Option<LaunchEpoch>) -> Timed {
        match epoch {
            Some(epoch) if self.launch_epoch() == Some(epoch) => match epoch.cover {
                Some(ExpertCover { reading: true, .. }) => Timed::Covering,
                _ => Timed::Settled,
            },
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

/// The expert bank's resizes: how many began so far, whether one runs now,
/// and the flight boundaries that advanced a running one so far. A reading
/// equal across a span with none running at either end saw no resize in
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertResizes {
    pub begun: u64,
    pub running: bool,
    pub advances: u64,
}

impl ExpertResizes {
    /// Whether one resize ran from this reading through `after` and a
    /// flight boundary advanced it between them: none began between them
    /// and one runs at both ends. A resize ends only by its advances, so
    /// one advanced across a span is on its way to its end, though the
    /// slots it changes change only as it installs or joins them.
    pub fn advanced_to(self, after: Self) -> bool {
        self.running && after.running && self.begun == after.begun && after.advances > self.advances
    }
}

/// The slots the expert bank's experts and its host tier (none without
/// one) follow now: each holding's derived target, at the decode's price
/// and its store's room (`experts::derived::derive`). The bank resizes
/// toward them, so a bank whose targets move may turn back on a resize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertTargets {
    pub slots: u64,
    pub host: Option<u64>,
}

/// The expert bank's cover (a host tier with a record for every expert,
/// read in behind every step's reads): how many times its reads began so
/// far, counting the bank's open, and whether they run now. A reading
/// equal across a span with none running at either end saw no cover read
/// in it. A bank with no tier that covers its experts reads none once its
/// first turn publishes so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExpertCover {
    pub begun: u64,
    pub reading: bool,
}

impl ExpertCover {
    /// The cover after `changes` starts and ends of its reads, reading as
    /// the bank opens.
    pub(crate) fn after(changes: u64) -> Self {
        Self { begun: changes / 2 + 1, reading: changes % 2 == 0 }
    }
}

/// What reads memory for the domain: the device's reader and ledgers, and
/// whether a bank resize runs. None of the program family's, which each
/// reading is handed its expert bank from: a chain's encoding reads memory
/// through it while it holds the family ([`ExecutorDomain::submit_chain`]).
struct MemoryProbe<'d> {
    memory: &'d RefCell<DeviceReader>,
    ledgers: &'d crate::memory::ledger::DeviceLedgers,
    answered: &'d RefCell<Answered>,
    resizing: bool,
}

/// The stores `ledgers` reach, each once: the device's own, and host RAM's
/// where it is a store apart.
fn stores(ledgers: &crate::memory::ledger::DeviceLedgers) -> impl Iterator<Item = &crate::memory::ledger::Ledger> {
    let (device, host) = (ledgers.device(), ledgers.host());
    std::iter::once(device).chain((host.store() != device.store()).then_some(host))
}

/// What the latest reading's answer left ([`MemoryProbe::answer`]): each
/// store's generation at it, before its response moved the ledger (what a
/// resize begun on it names, [`ExecutorDomain::sizing_terms`]), and the
/// response the owner has yet to carry out, every store's merged, cache
/// yielding already taken.
#[derive(Default)]
struct Answered {
    generations: Vec<(seismic::StoreIdentity, u64)>,
    owed: Option<MemoryResponse>,
}

/// One store's answer to its reading ([`MemoryProbe::answer`]).
struct StoreAnswer<'d> {
    ledger: &'d crate::memory::ledger::Ledger,
    generation: u64,
    shortage: Option<crate::memory::shortage::Shortage>,
    /// What its shortage's relief left the owner: the cache it shed, which
    /// each holder frees at the next boundary, and the requests to preempt.
    owed: crate::memory::ledger::Owed,
}

impl<'d> MemoryProbe<'d> {
    /// The domain's probe, of its parts apart from the program family.
    fn of(memory: &'d RefCell<DeviceReader>, ledgers: &'d crate::memory::ledger::DeviceLedgers, answered: &'d RefCell<Answered>, held: &ResizeHeld) -> Self {
        Self { memory, ledgers, answered, resizing: held.runs() }
    }

    /// The one answer to a reading (design §3.3): each store's real
    /// shortage at it judged once (`memory::ledger::Ledger::shortage`), its
    /// relief plan carried out whole at once (`Ledger::execute`: its cache
    /// yields, each holder freeing it at the next boundary), and the rest
    /// of every store's response, its preemptions among it, merged into
    /// what the owner
    /// owes, which it carries out in order at its next reading
    /// ([`ExecutorDomain::read_memory`]). Each store's generation at the
    /// reading is kept for the resizes begun on it. Whichever event read,
    /// a flight boundary or a pressure event beside a running chain, this
    /// is the one consumer of the shortage's answer. Its plans preempt none
    /// but `evictable`, the requests the owner can evict now, so the owner
    /// carries out every plan it is owed.
    fn answer(&self, evictable: &std::collections::BTreeSet<RequestId>) -> Vec<StoreAnswer<'d>> {
        let answers = stores(self.ledgers)
            .map(|ledger| {
                // The reading's generation, before its response moves the
                // ledger: what a resize that follows its target names.
                let generation = ledger.generation();
                let mut shortage = ledger.shortage(evictable);
                // A plan of cache alone is carried out at once; one naming
                // requests, each evictable now, is the owner's.
                let owed = match shortage.as_mut().and_then(|shortage| shortage.response.relief.take_if(|plan| plan.preempt().is_empty())) {
                    Some(plan) => ledger.execute(plan),
                    None => crate::memory::ledger::Owed::default(),
                };
                StoreAnswer { ledger, generation, shortage, owed }
            })
            .collect::<Vec<_>>();
        let mut answered = self.answered.borrow_mut();
        answered.generations = answers.iter().map(|answer| (answer.ledger.store(), answer.generation)).collect();
        for StoreAnswer { ledger, generation, shortage, owed: relieved } in &answers {
            let Some(shortage) = shortage else { continue };
            tracing::info!(
                store = ?ledger.store(),
                generation,
                cause = ?shortage.cause,
                yielded = relieved.yielded,
                admission_waits = shortage.response.admission_waits,
                preempt = shortage.response.relief.as_ref().map_or(0, |plan| plan.preempt().len()),
                unload = shortage.response.unload,
                "a real memory shortage"
            );
            let owed = answered.owed.get_or_insert_with(MemoryResponse::default);
            owed.admission_waits |= shortage.response.admission_waits;
            owed.relief.extend(shortage.response.relief.clone());
            owed.unload |= shortage.response.unload;
        }
        answers
    }

    /// [`ExecutorDomain::observe_memory`], beside `bank`.
    ///
    /// # Panics
    /// Where the device's reading fails: no decision stands without it.
    fn observe(&self, bank: Option<&dyn ElasticBank>) -> Vec<DomainReading> {
        let _rises = bank.map(|bank| bank.hold_rises());
        self.memory
            .borrow_mut()
            .refresh()
            .unwrap_or_else(|error| panic!("invariant violated: the device's memory reading failed: {error}"))
    }

    /// The device's ledgers read from a sample taken now, beside `bank`
    /// (`DeviceReader::read_at_event_after`): whether one stood, and so was read.
    fn read(&self, bank: Option<&dyn ElasticBank>, after_idle: bool) -> bool {
        let _rises = bank.map(|bank| bank.hold_rises());
        self.memory.borrow().read_at_event_after(self.ledgers, after_idle).is_some()
    }

    /// What the expert bank's store yields between a chain's submissions
    /// (`ExpertBank::release_running`): where an OS pressure event came
    /// since the last reading (`DeviceReader::pressured`), the ledgers read
    /// now and the reading answered ([`Self::answer`]), and the bank's
    /// share of the cache the answer yielded. Cache of every kind yields in
    /// rank order; only the bank's can be cut beside the chain, and only its
    /// share is: what the shortage took of the other kinds their holders
    /// free at the next boundary (`ExecutorDomain::release_yielded`), so
    /// nothing is released twice, and the rest of the response is owed to
    /// the owner's next reading. Each submission is an own change the next
    /// boundary reads; between them only another program's pressure is
    /// news. None while a bank resize runs (its advance is a boundary's, and
    /// the bank begins no cut beside it), with no pressure event since,
    /// where a change beside the sample deferred the reading (the pressure
    /// stays owed to the next), or where the answer took none of the bank's
    /// cache.
    fn running_pressure(&self, bank: &dyn ElasticBank) -> Option<RunningPressure> {
        let reader = self.memory.borrow();
        if self.resizing || !reader.pressured() {
            return None;
        }
        drop(reader);
        if !self.read(Some(bank), false) {
            self.memory.borrow().repress();
            return None;
        }
        let tier = bank.holding_storage(BankHolding::Experts).expect("an expert bank holds its experts");
        let ledger = self.ledgers.of(tier);
        let bank_cache = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank_cache();
        // Beside a running chain the owner evicts nothing: the answer's
        // relief preempts no request.
        drop(self.answer(&std::collections::BTreeSet::new()));
        match before - bank_cache() {
            0 => None,
            bytes => Some(RunningPressure { bytes }),
        }
    }
}

/// A reading of memory at an owner's event ([`ExecutorDomain::read_memory`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryReading {
    /// A change of ours off the owner's thread stood beside the sample
    /// (`memory::ledger::Sample::try_now`): nothing was read nor answered.
    /// Its completion wakes the owner, whose step reads then.
    Beside,
    Read(MemoryRead),
}

/// A reading taken and answered: what reading took, the ledgers'
/// generation right after it (`ExecutorDomain::memory_generation`), before
/// any response, and the response left for the owner, none where no store
/// is short anew.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRead {
    pub cost: Duration,
    pub generation: u64,
    pub response: Option<MemoryResponse>,
}

/// The reading a store's bank target is derived from
/// ([`ExecutorDomain::sizing_terms`]): its store, its generation as its
/// `memory read` record states it (none before the owner answered a
/// reading of the store), its instant on the owner's clock, and its
/// figures as the estimate stands now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizingTerms {
    pub store: seismic::StoreIdentity,
    pub reading: Option<u64>,
    pub read_at: Instant,
    pub figures: SizingFigures,
}

/// The figures of a store's estimate its bank target is derived from
/// (`memory::ledger::Estimate::target`):
/// `floor + (budget − committed − other cache − guarantee) / slot`, the
/// budget `min(grant, held + supply)` within the boundary `cap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizingFigures {
    pub budget: u64,
    pub supply: i64,
    pub others: u64,
    pub committed: u64,
    pub other_cache: u64,
    pub guarantee: u64,
    pub cap: Option<u64>,
}

/// One term of [`SizingFigures`] that moved between two readings, from
/// and to, named as the `memory read` record names its field
/// ([`Self::field`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingTerm {
    Budget { from: u64, to: u64 },
    Supply { from: i64, to: i64 },
    Others { from: u64, to: u64 },
    Committed { from: u64, to: u64 },
    OtherCache { from: u64, to: u64 },
    Guarantee { from: u64, to: u64 },
    Cap { from: Option<u64>, to: Option<u64> },
}

impl SizingTerm {
    /// The field of the `memory read` record it is.
    pub fn field(&self) -> &'static str {
        match self {
            Self::Budget { .. } => "budget",
            Self::Supply { .. } => "supply",
            Self::Others { .. } => "others",
            Self::Committed { .. } => "committed",
            Self::OtherCache { .. } => "other_cache",
            Self::Guarantee { .. } => "guarantee",
            Self::Cap { .. } => "cap",
        }
    }
}

impl SizingTerms {
    /// The term whose change since `before` set the target these give
    /// (`floor + (budget − committed − other cache − guarantee) / slot`,
    /// within the boundary): of the target's own terms, the one that
    /// moved most, named by what moved it, supply or others within the
    /// budget; the boundary
    /// where only it moved. None where no term moved: the target moved
    /// with the bank's own price, not with memory.
    pub fn term(&self, before: &Self) -> Option<&'static str> {
        let moved = self.moved(before);
        let has = |field: &str| moved.iter().any(|term| term.field() == field);
        let delta = |from: u64, to: u64| from.abs_diff(to);
        let (from, to) = (before.figures, self.figures);
        let largest = [
            ("budget", delta(from.budget, to.budget)),
            ("committed", delta(from.committed, to.committed)),
            ("other_cache", delta(from.other_cache, to.other_cache)),
            ("guarantee", delta(from.guarantee, to.guarantee)),
        ]
        .into_iter()
        .filter(|(_, by)| *by > 0)
        .max_by_key(|(_, by)| *by)
        .map(|(field, _)| field);
        Some(match largest {
            Some("budget") if has("supply") => "supply",
            Some("budget") if has("others") => "others",
            Some(field) => field,
            None if has("cap") => "cap",
            None => return None,
        })
    }

    /// The terms that moved from `before` to these: what traces a resize
    /// begun on these to the event that moved them.
    pub fn moved(&self, before: &Self) -> Vec<SizingTerm> {
        let (from, to) = (before.figures, self.figures);
        let budget = (from.budget != to.budget).then_some(SizingTerm::Budget { from: from.budget, to: to.budget });
        let supply = (from.supply != to.supply).then_some(SizingTerm::Supply { from: from.supply, to: to.supply });
        let terms = [
            (from.others, to.others, SizingTerm::Others { from: from.others, to: to.others }),
            (from.committed, to.committed, SizingTerm::Committed { from: from.committed, to: to.committed }),
            (from.other_cache, to.other_cache, SizingTerm::OtherCache { from: from.other_cache, to: to.other_cache }),
            (from.guarantee, to.guarantee, SizingTerm::Guarantee { from: from.guarantee, to: to.guarantee }),
        ];
        let mut moved = budget.into_iter().chain(supply).chain(terms.into_iter().filter(|(from, to, _)| from != to).map(|(.., term)| term)).collect::<Vec<_>>();
        if from.cap != to.cap {
            moved.push(SizingTerm::Cap { from: from.cap, to: to.cap });
        }
        moved
    }
}

/// The ordered response to the real shortages of a load's stores
/// (`memory::shortage`), with the first step, cache yielding, already
/// taken ([`ExecutorDomain::read_memory`]): what the owner carries out of
/// the rest, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryResponse {
    /// New work waits: committed memory or others' rise is short, or no
    /// relief covers the shortage.
    pub admission_waits: bool,
    /// Relief naming requests to preempt, the lowest value first: the
    /// owner carries each out whole, its cache shed and its requests
    /// evicted, only where it can evict every one of them now
    /// (`ExecutorDomain::execute_relief`); otherwise none of it.
    pub relief: Vec<crate::memory::ledger::ReliefPlan>,
    /// A floor shortfall persists with nothing to give and nothing of ours
    /// pending: the model unloads.
    pub unload: bool,
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
    /// The flight boundaries' advances of the bank while a resize ran
    /// ([`ExpertResizes::advances`]).
    advances: u64,
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
        if source == HeldSource::Advance && self.runs() {
            self.advances += 1;
        }
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

/// The target each memory reading of the owner is traced to: what an
/// instrument samples a load's charge against its room from.
pub const MEMORY_TRACE: &str = "magnitude_memory_reading";

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

    /// The domain over the load's native family. Its holdings of what the
    /// load allocated as it started (`recorded`, and `constants`, the
    /// target's binding constants it binds now, claimed as
    /// [`crate::memory::ledger::CommittedKind::Arena`]) are carved from that
    /// record; a store short of what it commits beyond it refuses it as
    /// [`DomainError::Capacity`], a state the load matches.
    pub fn new(
        execution: Rc<ExecutionPlan>,
        definition: Rc<ModelDefinition>,
        memory: DeviceReader,
        programs: Rc<AttestedPrograms>,
        resources: AllocatedResources,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        resident: crate::ResidentTarget,
        target_state: StoreBindings,
        head_state: Option<StoreBindings>,
        mut recorded: crate::memory::ledger::Recorded,
        mut constants: crate::memory::ledger::Held,
        request_slot: crate::memory::ledger::Claim,
    ) -> Result<(Self, StateBindings), DomainError> {
        // A model's drafter (its draft head or separate draft) is enabled
        // only when its method drafts.
        if (head_loader.is_some() && definition.head.is_none() && definition.draft.is_none())
            || vision_loader.is_some() != definition.vision.is_some()
        {
            return Err(DomainError::invariant("component loaders differ from enabled model components"));
        }
        // A draft head reading the target's history keeps no state arena.
        let keeps_history = definition.draft.is_some()
            || definition
                .head
                .as_ref()
                .is_some_and(|head| head.history == magnitude_family_contracts::HeadHistory::Own);
        if head_state.is_some() != (head_loader.is_some() && keeps_history) {
            return Err(DomainError::invariant("head state arena differs from enabled head component"));
        }
        // The family binds the target's constants, the arena's: one change
        // with their record.
        let family = constants.allocate(|| NativeFamily::new(programs, resident, definition.decoder.clone(), resources.ledgers(), &mut recorded))?;
        recorded.record(constants);
        Self::with_family(
            execution,
            definition,
            resources,
            memory,
            target_state,
            head_state,
            head_loader,
            vision_loader,
            family,
            recorded,
            request_slot,
        )
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

    /// A reading at an event (a flight boundary, an own change, an OS
    /// pressure event, a reading owed): the device's ledgers read a sample
    /// taken now, and each store's shortage judged on it
    /// (`memory::ledger::Ledger::shortage`), once per reading: a shortage
    /// already answered is not answered again. The first step of each
    /// store's response is taken at once: its cache yields in rank order
    /// (`Ledger::execute`, the relief plan carried out whole), which each holder frees at this boundary.
    /// What is left of the response, every store's in one, is returned for
    /// the owner to carry out in order; none where no store is short anew.
    /// Where a change off the owner's thread stood beside the sample,
    /// nothing is read and nothing answered ([`MemoryReading::Beside`]).
    /// The response preempts none but `evictable`, the requests the owner
    /// can evict now.
    /// Each store's reading is traced as one `memory read` event at
    /// [`MEMORY_TRACE`], of the one estimate the judgment reads, and each
    /// bank holding in the store with what it holds and its derived target
    /// (`experts::derived::derive`, as the bank follows it).
    pub fn read_memory(&self, after_idle: bool, evictable: &std::collections::BTreeSet<RequestId>) -> Result<MemoryReading, DomainError> {
        let began = Instant::now();
        let probe = self.memory_probe();
        if !probe.read(self.family.expert_bank(), after_idle) {
            tracing::debug!(target: MEMORY_TRACE, "memory read deferred: a change of ours stood beside the sample");
            return Ok(MemoryReading::Beside);
        }
        let cost = began.elapsed();
        let generation = self.memory_generation();
        let traced = tracing::enabled!(target: MEMORY_TRACE, tracing::Level::DEBUG);
        let targets = match (traced, self.family.expert_bank()) {
            (true, Some(bank)) => self.derived_targets(bank)?,
            _ => Vec::new(),
        };
        for StoreAnswer { ledger, generation, shortage, owed } in probe.answer(evictable) {
            if !traced {
                continue;
            }
            let estimate = ledger.estimate();
            let held = |holding| targets.iter().find(|target| target.holding == holding && target.store == ledger.store());
            let (experts, host) = (held(BankHolding::Experts), held(BankHolding::Host));
            tracing::debug!(
                target: MEMORY_TRACE,
                store = ?ledger.store(),
                generation,
                budget = estimate.budget(),
                supply = estimate.supply(),
                committed = estimate.committed(),
                cache = estimate.cache(),
                bank_cache = estimate.cache_of(crate::memory::ledger::CacheKind::Bank),
                other_cache = estimate.cache() - estimate.cache_of(crate::memory::ledger::CacheKind::Bank),
                // Zero where no weight cap bounds the store.
                cap = estimate.cap().unwrap_or(0),
                free = estimate.free(),
                others = estimate.others(),
                nonresident = estimate.nonresident(),
                unobserved_resident = estimate.unobserved_resident(),
                unobserved_readings = estimate.unobserved_readings(),
                accounting_violations = estimate.accounting_violations(),
                pressure_lost = ledger.pressure_lost(),
                unwired_shortfall = estimate.unwired_shortfall(),
                unwired_readings = estimate.unwired_readings(),
                // Zero where no wired ceiling bounds the store.
                wired_limit = estimate.wired_limit().unwrap_or(0),
                wire_count = estimate.wire_count().unwrap_or(0),
                allocated = estimate.allocated(),
                guarantee = estimate.guarantee(),
                pending = estimate.pending(),
                short = shortage.is_some(),
                yielded = owed.yielded,
                bank_slots = experts.map_or(0, |target| target.held),
                bank_target = experts.map_or(0, |target| target.target),
                // The least the bank shrinks by; zero where no bank holds
                // experts in the store.
                bank_block = experts.map_or(0, |target| target.block),
                host_slots = host.map_or(0, |target| target.held),
                host_target = host.map_or(0, |target| target.target),
                "memory read"
            );
        }
        let response = self.answered.borrow_mut().owed.take();
        Ok(MemoryReading::Read(MemoryRead { cost, generation, response }))
    }

    /// The load's stores, each once ([`stores`]).
    fn stores(&self) -> impl Iterator<Item = &crate::memory::ledger::Ledger> {
        stores(self.resources.ledgers())
    }

    /// The load's ledgers' generation: the sum of its stores'
    /// (`memory::ledger::Ledger::generation`), which moves on every change
    /// and reading of either and on nothing else.
    pub fn memory_generation(&self) -> u64 {
        self.stores().map(crate::memory::ledger::Ledger::generation).sum()
    }

    /// The generation of what the load's ledgers hold: the sum of its
    /// stores' (`memory::ledger::Ledger::holdings_generation`), which moves
    /// on every change of what either holds and never on a reading.
    pub fn holdings_generation(&self) -> u64 {
        self.stores().map(crate::memory::ledger::Ledger::holdings_generation).sum()
    }

    /// Whether anything stands that only a later reading can answer: a
    /// store's own standing (`memory::ledger::Ledger::standing`: a real
    /// shortage, a refused claim waiting, a growth the boundary holds
    /// back), or the expert bank short of its saturation in any holding
    /// (`experts::derived::BankShape::saturation_slots`), which only memory
    /// another program frees can grow, and which the ledger, holding no
    /// miss curve, cannot see.
    pub fn memory_stands(&self) -> Result<bool, DomainError> {
        if self.stores().any(|ledger| {
            matches!(
                ledger.standing(),
                Some(
                    crate::memory::ledger::Standing::Shortage(_)
                        | crate::memory::ledger::Standing::Wait
                        | crate::memory::ledger::Standing::Step
                )
            )
        }) {
            return Ok(true);
        }
        let Some(bank) = self.family.expert_bank() else {
            return Ok(false);
        };
        self.derive_bank(bank, |derived| derived.iter().any(|(target, _)| target.at < target.shape.saturation_slots()))
    }

    /// The sum of readings folded into the load's stores so far: unchanged
    /// across an idle sleep iff nothing read any of its stores while asleep.
    pub fn readings_generation(&self) -> u64 {
        self.stores().map(crate::memory::ledger::Ledger::readings).sum()
    }

    /// The shortest interval between two readings across which another
    /// program's memory was seen to move, over the load's stores
    /// (`memory::ledger::Ledger::others_interval`); none until one was.
    pub fn others_interval(&self) -> Option<Duration> {
        self.stores().filter_map(crate::memory::ledger::Ledger::others_interval).min()
    }

    /// Whether an OS pressure event came since this was last asked
    /// (`DeviceReader::pressured`).
    pub fn take_pressure(&self) -> bool {
        self.memory.borrow().pressured()
    }

    /// The reading `tier`'s store's bank target is derived from now: the
    /// latest reading's instant and its estimate's terms, the bank's own
    /// cache apart from the rest.
    pub fn sizing_terms(&self, tier: crate::planning::BankTier) -> SizingTerms {
        let ledger = self.resources.ledgers().of(tier);
        let estimate = ledger.estimate();
        let bank = estimate.cache_of(crate::memory::ledger::CacheKind::Bank);
        let store = ledger.store();
        let reading = self.answered.borrow().generations.iter().find(|(read, _)| *read == store).map(|(_, generation)| *generation);
        SizingTerms {
            store,
            reading,
            read_at: estimate.at().at(),
            figures: SizingFigures {
                budget: estimate.budget(),
                supply: estimate.supply(),
                others: estimate.others(),
                committed: estimate.committed(),
                other_cache: estimate.cache() - bank,
                guarantee: estimate.guarantee(),
                cap: estimate.cap(),
            },
        }
    }

    /// What the process holds in the device's store now, by its ledger
    /// (`Estimate::allocated`): what a release or growth is recorded
    /// beside.
    pub fn held_bytes(&self) -> u64 {
        self.resources.ledgers().device().estimate().allocated()
    }

    /// Observe every domain the device uses from a reading taken now: the
    /// worker's report of them.
    pub fn observe_memory(&self) -> Vec<DomainReading> {
        self.memory_probe().observe(self.family.expert_bank())
    }

    /// What reads memory for the domain ([`MemoryProbe`]).
    fn memory_probe(&self) -> MemoryProbe<'_> {
        MemoryProbe::of(&self.memory, self.resources.ledgers(), &self.answered, &self.resize_held)
    }

    /// Watch the OS's memory pressure events for the domain's life
    /// (`DeviceReader::watch_pressure`): each wakes the owner through
    /// `wake`, whose run reads memory, and a chain's encoding reads it
    /// between submissions ([`MemoryProbe::running_pressure`]).
    pub fn watch_pressure(&self, wake: impl Fn() + Send + Sync + 'static) {
        self.memory.borrow_mut().watch_pressure(self.resources.ledgers().host(), wake);
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
        // The family is lent to the chain, so the probe borrows the rest.
        let probe = MemoryProbe::of(&self.memory, self.resources.ledgers(), &self.answered, &self.resize_held);
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
        let Some(RunningBegun { decided, held }) = bank.running_begun() else {
            return;
        };
        let tier = bank.holding_storage(BankHolding::Experts).expect("an expert bank holds its experts");
        self.resize_held.begin_held(tier, decided, held);
        assert!(self.running_release.replace(tier).is_none(), "the owner takes a running cut before the next begins");
    }

    /// The cut of the expert bank begun beside a running chain since the
    /// owner last asked, taken: the storage it resizes. The owner holds it
    /// as a shrink it began, which a later advance finishes
    /// ([`Self::advance_expert_bank`]).
    pub fn take_running_release(&mut self) -> Option<crate::planning::BankTier> {
        self.running_release.take()
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
        let lane = ReservationLane::of(first);
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
    /// unbound, each allocation under its claim (`Self::claim_growth`): the
    /// import's resident weights held by its loader for the import's life,
    /// its upload peak only while it runs, and its graph constants for its
    /// binding's life.
    fn bind_component(&mut self, lane: ReservationLane) -> Result<(), DomainError> {
        // Its import and binding allocate under the claims below, tables included.
        let _lifted = self.domain.device().lift_table_seal();
        match lane {
            ReservationLane::Head if !self.family.head_is_bound() => {
                let loader = self
                    .head_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("head component is disabled".into()))?;
                let constants = loader.binding_constant_bytes().map_err(DomainError::Input)?;
                let resident = self.import_component(
                    loader,
                    self.execution.resources().bytes().head_weights,
                    self.execution.load().head_upload_peak_bytes(),
                )?;
                let mut binding = self.claim_growth(
                    crate::memory::ledger::CommittedKind::Component,
                    crate::memory::ClaimDemand { device: constants, ..Default::default() },
                )?;
                binding.allocate(|| self.family.bind_head(resident, &self.definition))?;
                self.head_binding = binding.into_claims();
                self.hold_bound_head()?;
                Ok(())
            }
            ReservationLane::Vision if !self.family.vision_is_bound() => {
                let loader = self
                    .vision_loader
                    .as_ref()
                    .ok_or_else(|| DomainError::Input("vision component is disabled".into()))?;
                let constants = loader.binding_constant_bytes().map_err(DomainError::Input)?;
                let resident = self.import_component(
                    loader,
                    self.execution.resources().bytes().vision_weights,
                    self.execution.load().vision_upload_peak_bytes(),
                )?;
                let mut binding = self.claim_growth(
                    crate::memory::ledger::CommittedKind::Component,
                    crate::memory::ClaimDemand { device: constants, ..Default::default() },
                )?;
                binding.allocate(|| self.family.bind_vision(resident, &self.definition))?;
                self.vision_binding = binding.into_claims();
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// `loader`'s component, imported where it is not yet: its `resident`
    /// bytes claimed for the import's life, kept by the loader, and its
    /// upload `peak` (staged through host RAM where that is a store apart)
    /// claimed only while the import runs.
    fn import_component<T: Clone>(&self, loader: &ComponentLoader<T>, resident: u64, peak: u64) -> Result<T, DomainError> {
        let imported = match loader.imported() {
            true => loader.load(),
            false => {
                use crate::memory::{ClaimDemand, ledger::CommittedKind};
                let weights = self.claim_growth(CommittedKind::Component, ClaimDemand { device: resident, ..Default::default() })?;
                let _upload = self.claim_growth(CommittedKind::Upload, ClaimDemand { device: peak, staged: peak, ..Default::default() })?;
                loader.import(weights)
            }
        };
        imported.map_err(|error| DomainError::Input(error.to_string()))
    }

    /// Reserve capacity for one group: the front queued step's slots when
    /// the group claims it, otherwise its provisioned state, advances and
    /// launch leases.
    ///
    /// It allocates nothing of its own: every allocation the group needs is
    /// provisioning's ([`Self::provision`]), so after the group was
    /// provisioned, provisioning it again here grows nothing and no memory
    /// refusal reaches a reservation.
    pub fn reserve(
        &mut self,
        bindings: &mut StateBindings,
        operations: &[Operation],
    ) -> Result<DomainReservation, DomainError> {
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
        // The owner applied any state ceiling when it provisioned the group.
        self.provision(bindings, operations, None)?;
        // Provisioning may relocate a shared tail into a fresh page. Recompute
        // row demand against that layout: the next append can now fit in place.
        let requirements = self.requirements(operations)?;
        self.can_reserve(&requirements)
            .map_err(DomainError::Capacity)?;
        let reads_target = requirements.lane == ReservationLane::Head && self.head_reads_target();
        let invariant = |detail: &str, error: CapacityError| {
            DomainError::invariant(format!(
                "{detail} changed after availability check: {error}"
            ))
        };
        let boundary = self.boundary.take();
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
                    states.push(InFlightState::new(
                        self.target
                            .remove(&operation.request())
                            .expect("reservation preflight established target ownership"),
                    ));
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
        memory: DeviceReader,
        target_state: StoreBindings,
        head_state: Option<StoreBindings>,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        family: F,
        mut recorded: crate::memory::ledger::Recorded,
        request_slot: crate::memory::ledger::Claim,
    ) -> Result<(Self, StateBindings), DomainError> {
        assert_eq!(
            request_slot.kind(),
            crate::memory::ledger::CommittedKind::RequestSlot,
            "invariant violated: a domain's request slot is a {:?} claim",
            request_slot.kind()
        );
        let owner = resources.ledgers().changes().own();
        // The load's last allocation is behind it, and the OS has since
        // refilled the free pages its allocations took from the file cache:
        // the arena's claim, the first serving claims and the bank's first
        // target are judged by a reading taken now, not by the one the
        // load's last claim was made on. Where a change of ours off this
        // thread stands beside it, they are judged by the last reading with
        // every claim since counted; the owner's first step owes a reading
        // and takes it.
        if memory.read_at_event(resources.ledgers()).is_none() {
            tracing::debug!(target: MEMORY_TRACE, "the domain's opening memory read deferred: a change of ours stood beside the sample");
        }
        // The arena: the load's graph scratch and the target graphs' bound
        // constants (the floor's group buffers and the held pass's, among
        // them) as their allocations charged them, with groups of one
        // place; a group's class beyond it is claimed as the group opens
        // (`Self::open_group`).
        let arena = recorded.claim_arena(resources.ledgers().device());
        let id = resources.domain().clone();
        let lane_identities = ["target", "head", "vision"].map(|lane| {
            ProgramIdentity::new(format!("{id}:{lane}"))
                .expect("a lane identity names its lane and is never empty")
        });
        let domain = Self {
            execution,
            definition,
            resources,
            arena,
            _recorded: recorded.into_claims(),
            state_claims: BTreeMap::new(),
            boundary: None,
            request_slot,
            state_slabs: state::StateSlabs::default(),
            domain: ResourceDomain::new(id, memory.device().clone()),
            memory: RefCell::new(memory),
            target_store: Rc::clone(&target_state),
            head_store: head_state.as_ref().map(|state| Rc::clone(state)),
            family,
            head_loader,
            vision_loader,
            head_binding: Vec::new(),
            vision_binding: Vec::new(),
            target: BTreeMap::new(),
            queue: queue::Queue::Empty,
            head: BTreeMap::new(),
            input: BTreeMap::new(),
            lane_identities,
            selection_read: None,
            target_timing: None,
            trace_host_steps: std::env::var_os("MAGNITUDE_TRACE_HOST_STEP").is_some(),
            planned: Vec::new(),
            served: Default::default(),
            measurement_claim: None,
            group_class: None,
            idle_class: Vec::new(),
            idle_components: Vec::new(),
            answered: RefCell::default(),
            next_flight: 0,
            host_turn: None,
            proposed: BTreeSet::new(),
            publishing: BTreeSet::new(),
            poisoned: BTreeMap::new(),
            measures: None,
            resize_held: ResizeHeld::default(),
            shrink_step: false,
            running_release: None,
            _owner: owner,
            teardown: Default::default(),
        };
        let bindings = StateBindings {
            target: target_state,
            head: head_state,
        };
        Ok((domain, bindings))
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
            draft: operation.draft_shape(),
        }
    }

    /// Host timing of the most recently finished target step.
    pub fn target_timing(&self) -> Option<TargetHostTiming> {
        self.target_timing
    }
}
