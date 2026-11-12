//! Serialized ownership of logical generation and one executor resource domain.
//!
//! The owner is the one schedule. Admission is logical: it records a request
//! and replies. Residency, rounds, reclamation and eviction all happen in
//! [`Owner::run`], between flights, where the owner holds the
//! [`StateBindings`] that every binding change requires. The pipeline waits
//! only on its own flight; a request that cannot proceed waits alone, in
//! exactly one [`State`].
use super::{
    domain::{
        DomainFlight, ExecutorDomain, FlightOutcomes, OperationGroup, ResumeState, finish, group,
        requirements, submit_group,
    },
    policy::{
        AvailabilityEpoch, Operation as ScheduledOperation, Phase, Scheduler, Selection,
        ServiceLimits, Victim, order_victims,
    },
    prefix_cache::{
        CachePoint, MIN_BRANCH_GAIN, MIN_PREFIX_HIT, PrefixCache, PrefixCacheCapacity, PrefixPath,
        PrefixRetention,
    },
    protocol::{RequestSnapshot, RoomGrant, RoomRefusal, WorkerReply},
    publication::{
        CapacityResource, ModelUnloadCause, PhysicalTimings, PublicationPermit, PublicationQueue,
        PublicationReceiver, PublicationSender, PublicationWake, PublicationWakeKind, PublishError,
        RequestError, ServiceCapacityError,
    },
    release::{self, Release, Releases, Rung},
    round_driver::{RoundError, lower_round, reconcile_forward},
    serving::ServingCensus,
    worker::Wakes,
};
use magnitude_executor::{
    BankResize, BankResized, BankStep, BankTier, DomainError, Followed, LoadMeasures, InvariantError,
    MemoryChargeReconciliation, MemoryMoment, NativeFamily,
    OpenRequirements, Operation, Outcome, PendingOperationOutcome, PhysicalDecision,
    ProgramFamily, RequestId, ResourceKind, ResourcePlan, SlotRelease, StateBindings, SubmitError,
    SubmitFailure, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{
    FinishReason, Generation, OutputToken, RoundStart, StartedRound, TimedLaunch,
};
use magnitude_state::ShrinkPolicy;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    num::NonZeroU64,
    sync::Arc,
    time::Duration,
};

/// Where a newly admitted request stops its prefill to retain branch
/// points, and the shared prefix it waits for a live peer to retain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BranchPlan {
    branches: BTreeSet<usize>,
    awaited: Option<usize>,
}

/// Whether retaining a state at `point` is worth a split prefill chunk and a
/// retained state: a hit, inside the prompt, and at least
/// [`MIN_BRANCH_GAIN`] rows beyond `from`, where the request resumes.
fn worthwhile(point: usize, from: usize, prompt: usize) -> bool {
    point >= MIN_PREFIX_HIT && point >= from + MIN_BRANCH_GAIN && point < prompt
}

/// Plan a branch point at `point` when it is worthwhile over both `from`
/// and the deepest branch point already planned below it.
fn plan_branch(branches: &mut BTreeSet<usize>, point: usize, from: usize, prompt: usize) {
    let below = branches.range(..point).next_back().copied().unwrap_or(0);
    if !branches.contains(&point) && worthwhile(point, from.max(below), prompt) {
        branches.insert(point);
    }
}

/// Plan a declared cache point at `point` when it is a hit inside the prompt
/// beyond `from`, where the request resumes. Unlike a discovered divergence,
/// a declared point is not held to [`MIN_BRANCH_GAIN`]: the host declares it
/// where later requests are expected to diverge, and each turn of a
/// conversation resumes at the point its previous turn declared. A turn
/// shorter than the gain would otherwise retain nothing beyond where it
/// resumed, so every following turn would resume there too and recompute
/// all the turns since.
fn plan_declared(branches: &mut BTreeSet<usize>, point: usize, from: usize, prompt: usize) {
    if point >= MIN_PREFIX_HIT && point > from && point < prompt {
        branches.insert(point);
    }
}

/// Where a peer's prefill stops within the prefix it shares with a new
/// request, up to `shared` rows: its deepest planned point there. That is
/// `shared` itself, or a point fewer than [`MIN_BRANCH_GAIN`] rows below it
/// (such as a declared cache point) that kept `shared` from being planned.
fn peer_stop(branches: &BTreeSet<usize>, shared: usize) -> Option<usize> {
    branches.range(..=shared).next_back().copied()
}

/// The prefix a new request waits for: `in_flight`, the deepest stop a peer
/// is still prefilling toward, when it is worth resuming from over the
/// deepest cached entry (`resumed`) and within [`MIN_BRANCH_GAIN`] rows of
/// the deepest prefix the request shares with a cached path or a peer
/// (`deepest`); otherwise the request plans its own branch. Only an entry is
/// a resume point: a cached path sharing the prefix without an entry ending
/// in it saves the request nothing, so it never stands in for the peer.
fn awaited_prefix(in_flight: usize, deepest: usize, resumed: usize, prompt: usize) -> Option<usize> {
    (worthwhile(in_flight, resumed, prompt) && in_flight + MIN_BRANCH_GAIN > deepest)
        .then_some(in_flight)
}

/// Whether a request outside a round may take the device before a member's
/// queued successor chunk, which would then be discarded: it is unfinished
/// (or still owes its reconciliation) and is not waiting on a prefix that a
/// member computes.
fn competes(request: &Request, status: Status) -> bool {
    !matches!(status, Status::Terminal(_))
        && !matches!(request.state, State::Between(_, Stage::AwaitingPrefix { .. }))
}

/// Where a request's prompt is retained: one row before the prompt end, the
/// deepest point a request with the same prompt can resume from (it must
/// compute the row it samples from). None when that is not an exact input
/// boundary, too short to be a hit, or already retained.
fn prompt_boundary(record: &Record, generation: &Generation) -> Option<usize> {
    let boundary = generation.prompt().len().checked_sub(1)?;
    (record.prefix_cache
        && !record.prefill_retained
        && boundary >= MIN_PREFIX_HIT
        && generation.layout().boundary(boundary))
    .then_some(boundary)
}

/// The state every holder occupies, cached prefixes included: what the
/// prefix cache holds within the state the memory plan charges.
fn occupied_state<F: ProgramFamily>(
    domain: &ExecutorDomain<F>,
    cache: &PrefixCache<ResumeState>,
) -> Result<u64, String> {
    let cached = cache.states().collect::<Vec<_>>();
    let (target, head) = domain.state_holding_census(&cached)?;
    target
        .occupied()
        .checked_add(head.map_or(0, |head| head.occupied()))
        .ok_or_else(|| "occupied state overflows".to_owned())
}

/// The first position after `position` where the request's prefill stops
/// for the prefix cache: a planned branch point or its prompt boundary.
fn next_stop(record: &Record, generation: &Generation, position: usize) -> Option<usize> {
    let branch = record.branches.range(position + 1..).next().copied();
    [branch, prompt_boundary(record, generation)]
        .into_iter()
        .flatten()
        .filter(|&stop| stop > position)
        .min()
}

/// The prompt chunks, at most `chunks` of at most `chunk_rows` rows, a lone
/// prefill group's request submits next, when they are known now, so the
/// domain queues them behind this group (a chain, or a pipelined successor;
/// `ExecutorDomain::plan_prompt`). Planning stops at a planned branch or
/// retention point: the chunk ending there is followed by that retention,
/// not planned past.
fn planned_prefill(
    round: &Round,
    operations: &[Operation],
    chunk_rows: usize,
    chunks: usize,
) -> Result<Vec<Operation>, RequestError> {
    let [Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens,
        position,
        ..
    }] = operations
    else {
        return Ok(Vec::new());
    };
    let Some(Member {
        record,
        state: Membership::Started(started),
    }) = round.members.get(request)
    else {
        return Ok(Vec::new());
    };
    let end = position + tokens.len();
    let stop = next_stop(record, started.generation(), *position);
    if stop.is_some_and(|stop| stop <= end) {
        return Ok(Vec::new());
    }
    started
        .planned_prefill(*request, chunk_rows, stop, chunks)
        .map_err(invariant("planned prefill"))
}

/// Whether `operations` are one primed prompt chunk that closes its request's
/// prompt, so its target result publishes before its drafter entry finishes.
/// The prompt's end says so, never the entry's shape: a drafter that enters
/// each row with its successor enters every row of a chunk but a prompt's
/// first.
fn closes_prompt(round: &Round, operations: &[Operation]) -> bool {
    let [Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens,
        position,
        prime: Some(_),
        ..
    }] = operations
    else {
        return false;
    };
    let member = round
        .members
        .get(request)
        .unwrap_or_else(|| panic!("a primed prefill of request {request:?} is not a member of its round"));
    position + tokens.len() == member.state.generation().prompt().len()
}

/// The selection bound `operations` spend: every selected target row, and one
/// per drafting request.
fn selection_charge(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| match operation {
            Operation::Forward { select, .. } => select.len(),
            Operation::Head { proposals, .. } => usize::from(!proposals.is_empty()),
            Operation::Encode { .. } => 0,
        })
        .sum()
}

/// Whether `generation` is resident, unfinished, and not yet past `position`.
fn prefilling_toward(generation: &Generation, position: usize) -> bool {
    generation.is_resident()
        && generation.finish_reason().is_none()
        && generation.resident_position() < position
}

/// Whether a request outside the pipeline will compute its prompt up to
/// `position`: it prefills toward it now, or it is admitted and unfinished
/// but not yet resident, and its first round prefills from where the cache
/// resumes it (`resumes`), below `position`. A request admitted while an
/// earlier one has not yet started otherwise prefills the shared prefix
/// beside it; one the cache resumes beyond `position` never computes it. A
/// request awaiting a prefix computes nothing until a peer caches it, so
/// waiters never wait on each other.
fn computes_toward(state: &State, position: usize, resumes: impl FnOnce(&Generation) -> usize) -> bool {
    match state {
        State::Between(generation, Stage::NonResident(_)) => {
            generation.finish_reason().is_none() && resumes(generation) < position
        }
        State::Between(_, Stage::AwaitingPrefix { .. }) => false,
        _ => prefilling_toward(state.generation(), position),
    }
}

/// Whether a finished request still owes the prefix cache its reconciliation
/// round: numerical state trails its accepted path.
fn owes_reconciliation(record: &Record, generation: &Generation) -> bool {
    record.prefix_cache
        && !record.terminal_retained
        && matches!(
            generation.finish_reason(),
            Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
        )
        && generation.pending_reconciliation().is_some()
}

fn ended(generation: &Generation) -> bool {
    matches!(
        generation.finish_reason(),
        Some(FinishReason::Cancelled | FinishReason::Failed)
    )
}

/// What stays with a request wherever its state is.
struct Record {
    waiting_since: u64,
    service_ns: u64,
    physical_timings: PhysicalTimings,
    /// Purpose of the current operation chain, retained across pending/parked work.
    timing_phase: TimingPhase,
    error: Option<RequestError>,
    preemption_debt: u32,
    protected_until: Option<usize>,
    /// Whether the request resumes from and contributes to the prefix cache.
    prefix_cache: bool,
    /// The cached prefix the request's residency forked, protected while
    /// work derived from it is submitted.
    prefix_source: Option<u64>,
    prefill_retained: bool,
    terminal_retained: bool,
    /// Planned branch points, in prompt order: where this request diverges
    /// from a path the engine holds, and the cache points its admission
    /// declared. Prefill stops at each so the cache can retain a state every
    /// later request diverging there resumes from.
    branches: BTreeSet<usize>,
    /// The host's stream; `None` once the host released the request.
    publication: Option<PublicationSender>,
    publication_batch_limit: usize,
    pending_publication: Option<Vec<OutputToken>>,
    publication_permits: VecDeque<PublicationPermit>,
}

/// Request service purpose, independent of target/head lane and replay geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingPhase {
    Prompt,
    Generation,
    Retention,
}

impl TimingPhase {
    fn add(self, timings: &mut PhysicalTimings, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        let bucket = match self {
            Self::Prompt => &mut timings.prompt_ns,
            Self::Generation => &mut timings.predicted_ns,
            Self::Retention => return,
        };
        *bucket = bucket.saturating_add(nanos);
    }
}

impl Record {
    fn begin_service(&mut self, candidate: &Candidate) {
        // A newly started chain owns its accounting purpose. Follow-up priming
        // inherits it even after the target publishes its first token. Resuming
        // pending/parked operations must not reclassify that existing chain.
        if candidate.composed().is_none() {
            let generation = candidate.generation();
            self.timing_phase = if owes_reconciliation(self, generation) {
                TimingPhase::Retention
            } else if generation.generated().is_empty() {
                TimingPhase::Prompt
            } else {
                TimingPhase::Generation
            };
        }
    }

    fn add_physical_duration(&mut self, duration: Duration) {
        self.timing_phase.add(&mut self.physical_timings, duration);
    }

    /// Drop everything the host would still receive.
    fn release_publication(&mut self) {
        self.pending_publication = None;
        self.publication_permits.clear();
        self.publication.take();
    }
}

/// Lowered operations a request submits before anything new. Never empty.
struct Work(Vec<Operation>);

impl Work {
    fn new(operations: Vec<Operation>) -> Option<Self> {
        (!operations.is_empty()).then_some(Self(operations))
    }
}

/// Why residency or provisioning was refused, after every release was tried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wait {
    Capacity {
        at: AvailabilityEpoch,
        resource: ResourceKind,
        required: u64,
        available: u64,
    },
    /// The ledger named sequences to preempt that cannot be evicted now,
    /// or the reading is blind: it retries once the epoch moves (an
    /// eviction, a release, a request ending).
    Memory { at: AvailabilityEpoch },
}

enum Stage {
    /// Not resident; a live peer is computing the prefix through this
    /// position, and this request will resume from it.
    AwaitingPrefix { through: usize },
    /// Not resident. Selection makes it resident.
    NonResident(Option<Wait>),
    /// Resident; selection starts a round. A finished request waits here for
    /// its output to drain, or for its reconciliation round.
    Ready,
    /// Resident; its host must drain output before the next round's
    /// publication permits fit.
    AwaitingCredit,
    /// Resident with work outside a target round to submit first: image
    /// encodes after residency, or method (drafter) operations.
    Pending(Work, Option<Wait>),
}

/// A request outside the pipeline.
enum State {
    /// No round started.
    Between(Generation, Stage),
    /// A started target round whose next operation couldn't be provisioned.
    /// It is submitted unchanged when selected.
    Parked(StartedRound, Work, Option<Wait>),
}

impl State {
    /// All composed work has completed. Only this state may be retained or
    /// retired successfully; a logical terminal decision can precede priming.
    fn settled(&self) -> Option<&Generation> {
        match self {
            Self::Between(
                generation,
                Stage::Ready
                | Stage::NonResident(_)
                | Stage::AwaitingCredit
                | Stage::AwaitingPrefix { .. },
            ) => Some(generation),
            _ => None,
        }
    }

    fn generation(&self) -> &Generation {
        match self {
            Self::Between(generation, _) => generation,
            Self::Parked(round, ..) => round.generation(),
        }
    }

    fn wait(&self) -> Option<Wait> {
        match self {
            Self::Between(_, Stage::NonResident(wait) | Stage::Pending(_, wait))
            | Self::Parked(_, _, wait) => *wait,
            Self::Between(_, _) => None,
        }
    }

    /// Terminate the request's generation, discarding any started round.
    fn terminate(self, cancel: bool) -> Generation {
        match self {
            Self::Between(generation, _) => terminated(generation, cancel),
            Self::Parked(round, ..) if cancel => round.cancel(),
            Self::Parked(round, ..) => round.fail(),
        }
    }
}

struct Request {
    record: Record,
    state: State,
}

/// A request whose work is in the current round.
enum Membership {
    /// Encodes or drafter work: no target round started.
    Generation(Generation),
    Started(StartedRound),
}

impl Membership {
    fn generation(&self) -> &Generation {
        match self {
            Self::Generation(generation) => generation,
            Self::Started(round) => round.generation(),
        }
    }

    fn terminate(self, cancel: bool) -> Generation {
        match self {
            Self::Generation(generation) => terminated(generation, cancel),
            Self::Started(round) if cancel => round.cancel(),
            Self::Started(round) => round.fail(),
        }
    }
}

/// A terminated request's finish: cancelled when withdrawn by its client, failed otherwise, so a
/// failure always publishes its classified error rather than an empty completion.
fn terminated(mut generation: Generation, cancel: bool) -> Generation {
    if cancel {
        generation.cancel();
    } else {
        generation.fail();
    }
    generation
}

struct Member {
    record: Record,
    state: Membership,
}

/// A request selection can take, moved out of the map while a round forms.
enum Candidate {
    NonResident(Generation),
    Ready(Generation),
    Pending(Generation, Work),
    Parked(StartedRound, Work),
}

impl Candidate {
    /// The operations already composed for this candidate, which join a
    /// round as they are.
    fn composed(&self) -> Option<&[Operation]> {
        match self {
            Self::Pending(_, work) | Self::Parked(_, work) => Some(&work.0),
            Self::NonResident(_) | Self::Ready(_) => None,
        }
    }

    fn generation(&self) -> &Generation {
        match self {
            Self::NonResident(generation) | Self::Ready(generation) | Self::Pending(generation, _) => {
                generation
            }
            Self::Parked(round, _) => round.generation(),
        }
    }

    fn into_state(self) -> State {
        match self {
            Self::NonResident(generation) => State::Between(generation, Stage::NonResident(None)),
            Self::Ready(generation) => State::Between(generation, Stage::Ready),
            Self::Pending(generation, work) => State::Between(generation, Stage::Pending(work, None)),
            Self::Parked(round, work) => State::Parked(round, work, None),
        }
    }
}

/// How starting a selected request ended.
enum Start {
    /// Its work joins the round.
    Joined(Member, Vec<Operation>),
    /// It waits, or ended, outside the round.
    Returned(Request),
}

fn classify_domain_error(error: DomainError) -> RequestError {
    match error {
        DomainError::Capacity(error) => RequestError::Capacity(ServiceCapacityError {
            resource: CapacityResource::Execution(error.resource),
            required: error.required,
            available: error.available,
        }),
        DomainError::Blind(message) => RequestError::Invariant(InvariantError {
            context: "device memory observation",
            detail: message,
        }),
        DomainError::Input(error) => RequestError::Input(error),
        DomainError::State(error) => RequestError::State(error),
        DomainError::Device(error) | DomainError::Submit(SubmitError::Device(error)) => {
            RequestError::Device(error)
        }
        DomainError::Invariant(error) | DomainError::Submit(SubmitError::Invariant(error)) => {
            RequestError::Invariant(error)
        }
        DomainError::WeightRead(error) => RequestError::WeightRead(error),
        error @ DomainError::StateCeiling { .. } => RequestError::Invariant(InvariantError {
            context: "state ceiling",
            detail: error.to_string(),
        }),
        error @ DomainError::Preempt(_) => RequestError::Invariant(InvariantError {
            context: "state preemption",
            detail: error.to_string(),
        }),
    }
}

/// Whether a domain error stops execution rather than one request.
fn is_fatal(error: &DomainError) -> bool {
    matches!(
        error,
        DomainError::Device(_)
            | DomainError::Invariant(_)
            | DomainError::State(_)
            | DomainError::Submit(SubmitError::Device(_) | SubmitError::Invariant(_))
    )
}

fn invariant(context: &'static str) -> impl FnOnce(String) -> RequestError {
    move |detail| RequestError::Invariant(InvariantError { context, detail })
}

/// A release that failed an engine invariant, as the domain's own releases
/// fail ([`Releasing`]).
fn release_invariant(context: &'static str) -> impl FnOnce(String) -> DomainError {
    move |detail| DomainError::Invariant(InvariantError { context, detail })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    /// It never ran, and waits on memory: the sequences the ledger named
    /// for preemption to be evicted, or a reading that is not blind.
    MemoryBlocked,
    CapacityBlocked { required: u64, available: u64 },
    Terminal(FinishReason),
}

/// A request refused before it acquires accepted numerical state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// A required device-memory observation failed. The caller may retry
    /// after the platform can report availability again.
    MemoryObservationUnavailable(String),
    ModelUnloaded {
        cause: ModelUnloadCause,
    },
    /// The request cannot be admitted: the typed failure it would otherwise
    /// have reported (invalid input or a failed engine invariant).
    Refused(RequestError),
}

impl AdmissionError {
    /// An engine invariant failed while admitting.
    fn invariant(context: &'static str) -> impl FnOnce(String) -> Self {
        move |detail| Self::Refused(RequestError::Invariant(InvariantError { context, detail }))
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryObservationUnavailable(message) => {
                write!(
                    formatter,
                    "device memory observation unavailable: {message}"
                )
            }
            Self::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => formatter.write_str("model unloaded due to memory pressure"),
            Self::Refused(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AdmissionError {}

impl From<DomainError> for AdmissionError {
    fn from(error: DomainError) -> Self {
        match error {
            DomainError::Blind(message) => Self::MemoryObservationUnavailable(message),
            other => Self::Refused(classify_domain_error(other)),
        }
    }
}

/// What one [`Owner::run`] left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ran {
    /// When the owner must run again without an event: its next memory
    /// observation.
    pub due: u64,
    /// Whether a group is on the device.
    pub in_flight: bool,
    /// Memory escalation decided to unload the model. Every request has been
    /// terminated with memory pressure; the caller drains the flight.
    pub unload: bool,
}

/// How often the owner reads memory without another reason to run, in
/// nanoseconds: the interval the machine's reader publishes readings at
/// ([`magnitude_executor::platform::READING_INTERVAL`]).
const OBSERVATION_INTERVAL_NS: u64 = magnitude_executor::platform::READING_INTERVAL.as_nanos() as u64;

/// A running expert bank resize: what the process held when it began, and
/// why it began.
#[derive(Clone, Copy, Debug)]
struct BankBegun {
    held: MemoryMoment,
    cause: BankBegunFor,
}

/// Why an expert bank resize began: a shrink to the bank's derived target,
/// or a growth into free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankBegunFor {
    Shrink,
    Growth,
}

impl BankBegunFor {
    /// The resize from `from` to `to` that finished, at `at_unix_ms`, as the
    /// census states it: the shrink or growth it began as, by at least one.
    fn finished(self, from: u64, to: u64, at_unix_ms: u64) -> Result<BankResize, String> {
        let by = |less: u64, more: u64| more.checked_sub(less).and_then(NonZeroU64::new);
        match self {
            Self::Shrink => by(to, from).map(|by| BankResize::Shrank { to, by, at_unix_ms }),
            Self::Growth => by(from, to).map(|by| BankResize::Grew { from, by, at_unix_ms }),
        }
        .ok_or_else(|| format!("a resize begun for {self:?} went from {from} to {to}"))
    }
}

struct ActiveGroup<F: ProgramFamily> {
    flight: DomainFlight<F>,
    operations: Vec<Operation>,
    prefix_uses: Vec<u64>,
}

struct Round {
    selection: Selection,
    started: u64,
    /// The rows its forward and head operations carry: a prefill round's
    /// tokens.
    rows: usize,
    /// The requests whose work is in this round, moved out of the map.
    members: BTreeMap<RequestId, Member>,
    /// Groups not yet submitted, in order; reconciliation appends follow-ups.
    groups: VecDeque<OperationGroup>,
}

enum Pipeline<F: ProgramFamily> {
    /// Between flights: the right to change store bindings is here.
    Idle(StateBindings),
    /// One group on the device. Its flight holds the bindings until
    /// completion returns them.
    InFlight(Round, ActiveGroup<F>),
}

/// How provisioning a group ended.
enum Provisioned {
    /// The group fits: submit it.
    Fits(OperationGroup),
    /// Capacity fits the leading operations: submit them first.
    Split(OperationGroup, OperationGroup),
    /// It cannot proceed now; its requests wait.
    Short(OperationGroup, Wait),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemoryDeficit {
    resource: ResourceKind,
    required: u64,
}

/// Everything the owner holds except its pipeline.
struct Service<F: ProgramFamily> {
    domain: ExecutorDomain<F>,
    scheduler: Scheduler,
    /// Requests outside the pipeline.
    requests: BTreeMap<RequestId, Request>,
    epoch: AvailabilityEpoch,
    next_id: u64,
    now: u64,
    prefix_cache: PrefixCache<ResumeState>,
    /// What the owner holds of memory between flights ([`between_flights`]).
    memory: BetweenFlights,
    wakes: Arc<dyn Wakes>,
}

/// What a loaded model's owner holds of its memory between flights
/// ([`between_flights`]): the deficits capacity-blocked requests wait on,
/// the expert bank's running resizes, the load's newly stored measures, and
/// when it reads memory next.
pub struct BetweenFlights {
    /// Per store's memory, the least a capacity-blocked request still needs
    /// there.
    deficits: Vec<MemoryDeficit>,
    /// Each store's running expert bank resize this owner began: the stores
    /// resize on their own (`BankStep::Begun`).
    bank_began: [Option<BankBegun>; BankTier::COUNT],
    /// Costs the executor stored since the load was priced, which price it
    /// from the next flight boundary on ([`Owner::measure_load`]).
    measures: Option<LoadMeasures>,
    next_observation: u64,
}

impl BetweenFlights {
    /// A resize of `tier` storage began, as `begun`: the bank begins none of
    /// a store while one of it runs.
    fn bank_begun(&mut self, tier: BankTier, begun: BankBegun) -> Result<(), String> {
        match &mut self.bank_began[tier.index()] {
            Some(running) => Err(format!("a {tier:?} resize began beside one running since {:?}", running.cause)),
            slot => {
                *slot = Some(begun);
                Ok(())
            }
        }
    }

    /// A load's memory as it opens: nothing in motion, read at its first
    /// boundary.
    pub fn new() -> Self {
        Self { deficits: Vec::new(), bank_began: [None; BankTier::COUNT], measures: None, next_observation: 0 }
    }
}

/// What [`between_flights`] runs on: the loaded model's domain and the
/// owner's memory, the service clock and its completion wakes, and the
/// model's unload (below the scheduler, an instrument has no requests): the
/// one owner of the between-flight memory step is that function, and a host
/// gives it only these.
pub trait MemoryHost {
    type Family: ProgramFamily;
    fn domain(&mut self) -> &mut ExecutorDomain<Self::Family>;
    fn memory(&mut self) -> &mut BetweenFlights;
    /// The service clock, in nanoseconds.
    fn now(&self) -> u64;
    fn completion(&self) -> magnitude_executor::CompletionWake;
    /// An availability change: what waits on one retries.
    fn advance_epoch(&mut self);
    /// Unload the model for memory: every request is terminated.
    fn unload(&mut self);
}

pub struct Owner<F: ProgramFamily = NativeFamily> {
    service: Service<F>,
    pipeline: Pipeline<F>,
}

impl<F: ProgramFamily> Owner<F> {
    pub fn new(
        domain: ExecutorDomain<F>,
        bindings: StateBindings,
        limits: ServiceLimits,
        wakes: Arc<dyn Wakes>,
    ) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity::disabled(),
            wakes,
        )
    }

    pub fn with_prefix_cache_capacity(
        domain: ExecutorDomain<F>,
        bindings: StateBindings,
        limits: ServiceLimits,
        capacity: PrefixCacheCapacity,
        wakes: Arc<dyn Wakes>,
    ) -> Result<Self, String> {
        Ok(Self {
            service: Service {
                domain,
                scheduler: Scheduler::new(limits)?,
                requests: BTreeMap::new(),
                epoch: AvailabilityEpoch::default(),
                next_id: 1,
                now: 0,
                prefix_cache: PrefixCache::new(capacity),
                memory: BetweenFlights::new(),
                wakes,
            },
            pipeline: Pipeline::Idle(bindings),
        })
    }

    pub fn with_resource_plan(
        domain: ExecutorDomain<F>,
        bindings: StateBindings,
        limits: ServiceLimits,
        plan: &ResourcePlan,
        wakes: Arc<dyn Wakes>,
    ) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(
            domain,
            bindings,
            limits,
            PrefixCacheCapacity::from_resource_plan(plan),
            wakes,
        )
    }

    /// Logical admission: record the request and hand the host its stream.
    /// It opens no state and grows nothing; it becomes resident when a round
    /// selects it.
    pub fn admit(
        &mut self,
        generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
        now: u64,
    ) -> Result<(RequestId, PublicationReceiver), AdmissionError> {
        let members = match &mut self.pipeline {
            Pipeline::InFlight(round, _) => Some(&mut round.members),
            Pipeline::Idle(_) => None,
        };
        self.service
            .admit(generation, input, retention, output_capacity, now, members)
    }

    /// Stop a live request through its ordered stream. Work in flight still
    /// reconciles before the request retires.
    pub fn stop(&mut self, request: RequestId) {
        self.terminate_request(request, false);
    }

    /// Release a request the host abandoned: no terminal outcome reaches it.
    pub fn cancel(&mut self, request: RequestId) {
        self.terminate_request(request, true);
    }

    fn terminate_request(&mut self, id: RequestId, release: bool) {
        if let Pipeline::InFlight(round, _) = &mut self.pipeline {
            if let Some(Member { mut record, state }) = round.members.remove(&id) {
                // Work in flight reconciles as aborted; queued work is dropped.
                let mut generation = state.terminate(true);
                generation.discard_output();
                if release {
                    record.release_publication();
                } else {
                    record.pending_publication = None;
                    record.publication_permits.clear();
                }
                round.members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
                return;
            }
        }
        self.service.terminate_request(id, release);
    }

    pub fn snapshot(&self, request: RequestId) -> Option<RequestSnapshot> {
        if let Pipeline::InFlight(round, _) = &self.pipeline {
            if let Some(member) = round.members.get(&request) {
                return Some(snapshot(
                    member.state.generation(),
                    match member.state.generation().finish_reason() {
                        Some(finish) if ended(member.state.generation()) => {
                            Status::Terminal(finish)
                        }
                        _ => Status::AwaitingCompletion,
                    },
                ));
            }
        }
        let request = self.service.requests.get(&request)?;
        Some(snapshot(
            request.state.generation(),
            self.service.status(request),
        ))
    }

    /// A fresh reading of every memory domain, then Seismic's charge
    /// classified by every holder the owner keeps.
    pub fn observe(&self) -> Result<WorkerReply, String> {
        Ok(WorkerReply::Observed {
            readings: self
                .service
                .domain
                .observe_memory()
                .map_err(|error| error.to_string())?,
            reconciliation: self.reconcile_memory_charge()?,
            serving: self.serving_census()?,
        })
    }

    /// How the model serves now ([`ServingCensus`]).
    pub fn serving_census(&self) -> Result<ServingCensus, String> {
        let service = &self.service;
        let experts = service.domain.expert_residency().map_err(|error| error.to_string())?;
        Ok(ServingCensus {
            group_places: service.domain.group_places(),
            group_capacity: service.domain.chain_capacity(),
            experts,
        })
    }

    /// Seismic's charge reconciled against every state holder the owner
    /// keeps: resident requests and cached prefixes; submitted work is
    /// counted through its stores' claims.
    pub fn reconcile_memory_charge(&self) -> Result<MemoryChargeReconciliation, String> {
        let cached = self.service.prefix_cache.states().collect::<Vec<_>>();
        self.service.domain.reconcile_memory_charge(&cached)
    }

    pub fn completed_service_ns(&self) -> u128 {
        self.service.scheduler.completed_service_ns()
    }

    pub fn cached_prefixes(&self) -> usize {
        self.service.prefix_cache.len()
    }

    /// Claim `bytes` of memory beside the model for the service's
    /// measurement job, in place of any claim before, which is released
    /// (`ExecutorDomain::claim_measurement_room`): a committed claim on the
    /// device's ledger, which cache yields to at once, so it is answered
    /// now, refused only where the model's committed memory leaves it short
    /// or the device's memory reading failed.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> RoomGrant {
        let (granter, grant) = RoomGrant::new();
        let claimed = self.service.domain.claim_measurement_room(bytes);
        granter.answer(claimed.map_err(|short| match short {
            magnitude_executor::memory::ledger::Short::Over { over, .. } => RoomRefusal::Short { bytes: over },
            magnitude_executor::memory::ledger::Short::Blind { store, reason } => {
                tracing::warn!(?store, %reason, "a measurement room is refused: the memory reading failed");
                RoomRefusal::Blind
            }
        }));
        grant
    }

    /// The measurement job ended: its claim is released.
    pub fn release_measurement_room(&mut self) {
        self.service.domain.release_measurement_room();
    }

    /// Price the load by `measures` from the next flight boundary on
    /// ([`between_flights`]), a later call's replacing an earlier one's not
    /// yet applied: no step in flight is recorded or priced by measures it
    /// did not run under.
    pub fn measure_load(&mut self, measures: LoadMeasures) {
        self.service.memory.measures = Some(measures);
    }

    pub fn inspect_domain<R>(
        &self,
        inspect: impl FnOnce(&ExecutorDomain<F>) -> Result<R, String>,
    ) -> Result<R, String> {
        inspect(&self.service.domain)
    }

    /// Consume a reserved wake for this request's exact queue. Output credit
    /// lets a request waiting on it start its next round; receiver closure
    /// releases the request.
    pub fn publication_wake(&mut self, request: RequestId, wake: PublicationWake) {
        let cancelled = match wake.kind() {
            PublicationWakeKind::OutputCredit => {
                self.service.output_credit(request, wake, &mut self.pipeline);
                false
            }
            PublicationWakeKind::Cancelled => self
                .service
                .publication_sender(request, &mut self.pipeline)
                .is_some_and(|sender| sender.process_cancelled(wake)),
        };
        if cancelled {
            self.cancel(request);
        }
    }

    /// Terminate every live request with `error`: the lifecycle is draining.
    pub fn terminate_all(&mut self, error: RequestError) {
        let members = match &mut self.pipeline {
            Pipeline::InFlight(round, _) => Some(&mut round.members),
            Pipeline::Idle(_) => None,
        };
        self.service.terminalize_all(error, members);
    }

    /// Run every transition that is possible now. An error is fatal: every
    /// request has been terminated with it and execution must stop.
    pub fn run(self, now: u64) -> Result<(Self, Ran), RequestError> {
        let Owner {
            mut service,
            pipeline,
        } = self;
        match service.drive(pipeline, now) {
            Ok((pipeline, ran)) => Ok((Owner { service, pipeline }, ran)),
            Err((mut members, error)) => {
                // Members are dropped with the owner; terminating them first
                // delivers their outcome.
                service.terminalize_all(error.clone(), members.as_mut());
                Err(error)
            }
        }
    }
}

fn snapshot(generation: &Generation, status: Status) -> RequestSnapshot {
    let usage = generation.detailed_usage();
    RequestSnapshot {
        status,
        prompt_tokens: usage.prompt_tokens,
        cached_tokens: usage.cached_tokens,
        resident_position: generation.resident_position(),
        output_tokens: generation.output_len(),
    }
}

type Fatal = (Option<BTreeMap<RequestId, Member>>, RequestError);

impl<F: ProgramFamily> Service<F> {
    fn time(&mut self, now: u64) -> Result<(), RequestError> {
        if now < self.now {
            return Err(invariant("service clock")("service clock moved backwards".into()));
        }
        self.now = now;
        Ok(())
    }

    fn admit(
        &mut self,
        mut generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
        now: u64,
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) -> Result<(RequestId, PublicationReceiver), AdmissionError> {
        self.time(now).map_err(AdmissionError::Refused)?;
        self.domain.served_request(!input.vision().is_empty());
        if generation.is_resident() || generation.usage().completion_tokens != 0 {
            return Err(AdmissionError::invariant("admission")(
                "only a fresh generation can be admitted".into(),
            ));
        }
        let (prefix_cache, cache_points) = match retention {
            PrefixRetention::Transient => (false, Vec::new()),
            PrefixRetention::Retain { cache_points } => (true, cache_points),
        };
        generation
            .set_prefix_reuse(prefix_cache)
            .map_err(AdmissionError::invariant("admission"))?;
        let path = PrefixPath::of(&generation);
        if cache_points
            .iter()
            .any(|&point| point >= generation.prompt().len() || !path.exact_boundary(point))
        {
            return Err(AdmissionError::invariant("admission")(
                "a cache point must be an exact boundary inside the prompt".into(),
            ));
        }
        let id = RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            AdmissionError::invariant("service identity")("identity exhausted".into())
        })?;
        let wakes = self.wakes.clone();
        let (sender, receiver) = PublicationQueue::bounded(output_capacity, move |wake| {
            wakes.publication(id, wake);
        })
        .map_err(|error| AdmissionError::invariant("publication")(error.to_owned()))?;
        self.domain
            .install_input(id, input)
            .map_err(|error| AdmissionError::Refused(RequestError::Input(error)))?;
        let plan = if prefix_cache {
            self.plan_branches(path, generation.resume_bound(), &cache_points, members)
        } else {
            BranchPlan::default()
        };
        let stage = match plan.awaited {
            Some(through) => Stage::AwaitingPrefix { through },
            None => Stage::NonResident(None),
        };
        self.requests.insert(
            id,
            Request {
                record: Record {
                    waiting_since: now,
                    service_ns: 0,
                    physical_timings: PhysicalTimings::default(),
                    timing_phase: TimingPhase::Prompt,
                    error: None,
                    preemption_debt: 0,
                    protected_until: None,
                    prefix_cache,
                    prefix_source: None,
                    prefill_retained: false,
                    terminal_retained: false,
                    branches: plan.branches,
                    publication: Some(sender),
                    publication_batch_limit: output_capacity,
                    pending_publication: None,
                    publication_permits: VecDeque::new(),
                },
                state: State::Between(generation, stage),
            },
        );
        Ok((id, receiver))
    }

    /// Branch points for a new request's `path`: each declared cache point
    /// no entry holds yet, and the deepest position where it diverges from a
    /// cached path or a live request's prompt. A live request that has not
    /// yet prefilled to its divergence from this one stops there too, so the
    /// shared prefix is cached by whichever request reaches it first and
    /// every later request resumes at it instead of recomputing.
    fn plan_branches(
        &mut self,
        path: PrefixPath,
        bound: usize,
        cache_points: &[usize],
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) -> BranchPlan {
        let resumed = self.prefix_cache.deepest(path, bound);
        let mut branches = BTreeSet::new();
        for &point in cache_points {
            if !self.prefix_cache.holds(path, point) {
                plan_declared(&mut branches, point, resumed, path.len());
            }
        }
        let mut deepest = self.prefix_cache.shared_boundary(path);
        // The deepest shared prefix a live peer is still prefilling toward
        // and will cache at its branch point.
        let mut in_flight = 0;
        let live = self
            .requests
            .values_mut()
            .map(|request| {
                (&mut request.record, request.state.generation(), Some(&request.state))
            })
            .chain(
                members
                    .into_iter()
                    .flat_map(|members| members.values_mut())
                    .map(|member| (&mut member.record, member.state.generation(), None)),
            )
            .filter(|(record, ..)| record.prefix_cache);
        for (record, generation, outside) in live {
            let shared = path.shared_with_prompt_of(PrefixPath::of(generation));
            deepest = deepest.max(shared);
            plan_branch(
                &mut record.branches,
                shared,
                generation.resident_position(),
                generation.prompt().len(),
            );
            if let Some(stop) = peer_stop(&record.branches, shared) {
                // A round member is resident; a request outside the
                // pipeline may not be yet.
                let computes = match outside {
                    Some(state) => computes_toward(state, stop, |peer| {
                        self.prefix_cache
                            .deepest(PrefixPath::of(peer), peer.resume_bound())
                    }),
                    None => prefilling_toward(generation, stop),
                };
                if computes {
                    in_flight = in_flight.max(stop);
                }
            }
        }
        if !worthwhile(deepest, resumed, path.len()) {
            return BranchPlan {
                branches,
                awaited: None,
            };
        }
        if let Some(through) = awaited_prefix(in_flight, deepest, resumed, path.len()) {
            // A peer computes this prefix now: wait for it to be cached.
            return BranchPlan {
                branches,
                awaited: Some(through),
            };
        }
        plan_branch(&mut branches, deepest, resumed, path.len());
        BranchPlan {
            branches,
            awaited: None,
        }
    }

    /// The cached entries live requests hold, among the service's requests
    /// and the in-flight round's `members`: the entry each resumed from and
    /// the deepest within its resident rows (see [`PrefixCache::held`]).
    /// Each costs one bank while its holder lives: the cache releases it
    /// last.
    fn held_entries(&self, members: Option<&BTreeMap<RequestId, Member>>) -> BTreeSet<u64> {
        self.requests
            .values()
            .map(|request| (&request.record, request.state.generation()))
            .chain(
                members
                    .into_iter()
                    .flat_map(BTreeMap::values)
                    .map(|member| (&member.record, member.state.generation())),
            )
            .filter(|(_, generation)| generation.is_resident() && generation.finish_reason().is_none())
            .flat_map(|(record, generation)| {
                self.prefix_cache.held(
                    record.prefix_source,
                    PrefixPath::of(generation),
                    generation.resident_position(),
                )
            })
            .collect()
    }

    /// Whether a live request other than `except` is still prefilling toward
    /// a planned branch point at `position`.
    fn prefix_pending(&self, position: usize, except: RequestId) -> bool {
        self.requests.iter().any(|(&id, request)| {
            id != except
                && request.record.prefix_cache
                && request.record.branches.contains(&position)
                && computes_toward(&request.state, position, |generation| {
                    self.prefix_cache
                        .deepest(PrefixPath::of(generation), generation.resume_bound())
                })
        })
    }

    fn status(&self, request: &Request) -> Status {
        let generation = request.state.generation();
        if let Some(finish) = generation.finish_reason() {
            if !owes_reconciliation(&request.record, generation) {
                return Status::Terminal(finish);
            }
        }
        match request.state.wait() {
            Some(Wait::Capacity {
                required,
                available,
                ..
            }) => {
                return Status::CapacityBlocked {
                    required,
                    available,
                };
            }
            Some(Wait::Memory { .. }) if self.waits_on_memory(request) => return Status::MemoryBlocked,
            Some(Wait::Memory { .. }) => return Status::Preempted,
            None => {}
        }
        match &request.state {
            State::Between(_, Stage::AwaitingCredit) => Status::OutputBlocked,
            State::Between(_, Stage::NonResident(_)) if request.record.preemption_debt > 0 => {
                Status::Preempted
            }
            _ => Status::Runnable,
        }
    }

    fn publication_sender<'a>(
        &'a mut self,
        request: RequestId,
        pipeline: &'a mut Pipeline<F>,
    ) -> Option<&'a mut PublicationSender> {
        if let Pipeline::InFlight(round, _) = pipeline {
            if let Some(member) = round.members.get_mut(&request) {
                return member.record.publication.as_mut();
            }
        }
        self.requests
            .get_mut(&request)
            .and_then(|request| request.record.publication.as_mut())
    }

    /// A credit wake sends a request waiting on credit back to selection,
    /// whatever the ring holds now: its next reservation either fits or,
    /// refused, re-arms the queue's next wake.
    fn output_credit(&mut self, id: RequestId, wake: PublicationWake, pipeline: &mut Pipeline<F>) {
        if let Some(sender) = self.publication_sender(id, pipeline) {
            sender.process_output_credit(wake);
        }
        if let Some(request) = self.requests.get_mut(&id) {
            if let State::Between(_, stage @ Stage::AwaitingCredit) = &mut request.state {
                *stage = Stage::Ready;
            }
        }
    }

    /// Terminate a request outside the pipeline, and retire it once nothing
    /// remains for its host.
    fn terminate_request(&mut self, id: RequestId, release: bool) {
        let Some(Request { mut record, state }) = self.requests.remove(&id) else {
            return;
        };
        let mut generation = state.terminate(true);
        generation.discard_output();
        if release {
            record.release_publication();
        } else {
            record.pending_publication = None;
            record.publication_permits.clear();
        }
        self.requests.insert(
            id,
            Request {
                record,
                state: State::Between(generation, Stage::Ready),
            },
        );
        self.epoch.advance();
    }

    /// Terminate every live request with `error`, delivering each host its
    /// outcome now.
    fn terminalize_all(
        &mut self,
        error: RequestError,
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) {
        let terminate = |record: &mut Record, generation: &mut Generation| {
            if generation.finish_reason().is_none() {
                generation.fail();
                record.error.get_or_insert_with(|| error.clone());
            }
            // Only output already accepted by the queue precedes a terminal
            // outcome. Unpublished generation output is not promoted by a
            // second, unbounded failure delivery channel.
            if let Some(sender) = record.publication.take() {
                generation.discard_output();
                record.pending_publication = None;
                record.publication_permits.clear();
                match generation.finish_reason() {
                    Some(FinishReason::Failed) | None => {
                        sender.fail(record.error.clone().unwrap_or_else(|| error.clone()));
                    }
                    Some(finish) => sender.complete(
                        finish,
                        generation.detailed_usage(),
                        generation.method_identity().to_owned(),
                        record.physical_timings,
                    ),
                }
            }
        };
        let ids = self.requests.keys().copied().collect::<Vec<_>>();
        for id in ids {
            let Request { mut record, state } = self.requests.remove(&id).expect("listed request");
            let mut generation = match state {
                State::Between(generation, _) => generation,
                State::Parked(round, ..) => round.fail(),
            };
            terminate(&mut record, &mut generation);
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::Ready),
                },
            );
        }
        if let Some(members) = members {
            let ids = members.keys().copied().collect::<Vec<_>>();
            for id in ids {
                let Member { mut record, state } = members.remove(&id).expect("listed member");
                let mut generation = match state {
                    Membership::Generation(generation) => generation,
                    Membership::Started(round) => round.fail(),
                };
                terminate(&mut record, &mut generation);
                members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
            }
        }
    }

    fn fail_generation(record: &mut Record, generation: &mut Generation, error: RequestError) {
        // Accepted output still precedes the terminal failure on this
        // request's stream. Its reserved permits must survive until that
        // output has drained.
        if generation.finish_reason().is_none() {
            generation.fail();
        }
        record.error = Some(error);
    }

    fn drive(&mut self, mut pipeline: Pipeline<F>, now: u64) -> Result<(Pipeline<F>, Ran), Fatal> {
        self.time(now).map_err(|error| (None, error))?;
        let mut unload = false;
        // A flight submitted by this run is reconciled by a later one, even
        // when the device completed it during submission (CPU): its
        // completion event returns the worker here only after it has applied
        // the controls and events that arrived during the flight.
        let mut submitted = false;
        loop {
            pipeline = match pipeline {
                Pipeline::InFlight(round, mut active) => {
                    if submitted || !active.flight.completion().is_complete() {
                        // The round's members end with a fatal error, not
                        // dropped with the round.
                        unload |= match self.observe(None, round.rows as u64) {
                            Ok(unload) => unload,
                            Err(error) => return Err((Some(round.members), error)),
                        };
                        return Ok((
                            Pipeline::InFlight(round, active),
                            self.ran(true, unload),
                        ));
                    }
                    // Reconciling submits the round's next flight.
                    submitted = true;
                    self.reconcile(round, active)?
                }
                Pipeline::Idle(mut bindings) => {
                    unload |= self
                        .observe(Some(&mut bindings), 0)
                        .map_err(|error| (None, error))?;
                    // Unloading terminated every request: nothing is left to
                    // select, and observing again would decide it again.
                    if unload {
                        self.publish_ready().map_err(|error| (None, error))?;
                        return Ok((Pipeline::Idle(bindings), self.ran(false, true)));
                    }
                    self.publish_ready().map_err(|error| (None, error))?;
                    match self.select(&mut bindings)? {
                        Selected::Round(round) => {
                            submitted = true;
                            self.advance(round, bindings)?
                        }
                        Selected::Retry => Pipeline::Idle(bindings),
                        Selected::Empty => {
                            // Waiting for work is no launch's service.
                            self.domain.host_idle();
                            if self.fail_capacity_waiters() {
                                Pipeline::Idle(bindings)
                            } else {
                                return Ok((Pipeline::Idle(bindings), self.ran(false, unload)));
                            }
                        }
                    }
                }
            };
        }
    }

    fn ran(&self, in_flight: bool, unload: bool) -> Ran {
        Ran {
            due: self.memory.next_observation,
            in_flight,
            unload,
        }
    }
}

/// The owner's memory between flights, on what a host gives it
/// ([`MemoryHost`]): every host takes these, and none has its own.
trait MemoryStep: MemoryHost + Sized {
    /// The memory step at a flight boundary, or while a flight holds the
    /// bindings (none here); `chunk_rows` the rows of the batch in flight,
    /// whose state growth the guarantee covers. Returns whether the model
    /// unloaded, after terminating every request. This is the owner's own
    /// work, which serves no request: no launch is charged the time it takes
    /// (`ExecutorDomain::owner_worked`).
    fn observe(&mut self, bindings: Option<&mut StateBindings>, chunk_rows: u64) -> Result<bool, RequestError> {
        let began = std::time::Instant::now();
        let observed = self.observe_memory(bindings, chunk_rows);
        self.domain().owner_worked(began.elapsed());
        observed
    }

    fn observe_memory(&mut self, bindings: Option<&mut StateBindings>, chunk_rows: u64) -> Result<bool, RequestError> {
        let now = self.now();
        // A reading is taken off this thread and published, and recorded in
        // the device's ledgers as it is taken
        // (`ExecutorDomain::take_published_memory`): one is looked for once
        // an interval has passed, and none waited for, so no reading sits
        // between a step and the next however long either takes.
        let read = match now >= self.memory().next_observation {
            true => {
                self.memory().next_observation = now.saturating_add(OBSERVATION_INTERVAL_NS);
                self.domain().take_published_memory()
            }
            false => None,
        };
        let readings = match read {
            None => None,
            Some(Ok(readings)) => Some(readings),
            // Blind: the ledger's snapshot unloads below.
            Some(Err(DomainError::Blind(reason))) => {
                tracing::warn!(%reason, "the memory reading failed");
                Some(Vec::new())
            }
            Some(Err(error)) => return Err(classify_domain_error(error)),
        };
        if readings.is_some() {
            self.reopen_memory_deficits();
        }
        // A cut of the expert bank a chain's encoding began beside its
        // running steps, the guarantee found short between its submissions
        // (`ExecutorDomain::take_running_release`), is a shrink this owner
        // holds as begun: the advance that finishes it reports it as any.
        if let Some((tier, held)) = self.domain().take_running_release() {
            let begun = BankBegun { held, cause: BankBegunFor::Shrink };
            self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))?;
        }
        // A flight boundary: newly stored costs price the load first, so
        // every decision past it is made on them; the expert bank's and the
        // graph slots' releases advance, and the bank follows its target,
        // derived from the ledgers afresh.
        if let Some(bindings) = bindings {
            // A refusal (the device would not take the tier copy's grid) is a device
            // fault the load ends on, the experts still priced as before.
            if let Some(measures) = self.memory().measures.take() {
                self.domain().measure_load(measures, bindings).map_err(classify_domain_error)?;
            }
            self.advance_expert_bank()?;
            self.advance_slot_release()?;
            self.follow_cache_targets(bindings)?;
        }
        // The model unloads at the reading where the guarantee cannot be
        // kept, or the reading failed (`ExecutorDomain::memory_reading`),
        // each store's reading traced as it is judged.
        if readings.is_some() && self.domain().memory_reading(chunk_rows).map_err(classify_domain_error)? {
            self.unload();
            return Ok(true);
        }
        Ok(false)
    }

    /// A request blocked on a memory deficit waits for an availability
    /// change. Memory another process frees is one: advance the epoch once
    /// the deficit's store, by its ledger's reading, holds what it needs
    /// (`ExecutorDomain::memory_covers`), so blocked work retries.
    fn reopen_memory_deficits(&mut self) {
        let mut deficits = std::mem::take(&mut self.memory().deficits);
        let before = deficits.len();
        deficits.retain(|deficit| !self.domain().memory_covers(deficit.resource, deficit.required));
        let reopened = deficits.len() != before;
        self.memory().deficits = deficits;
        if reopened {
            self.advance_epoch();
        }
    }

    /// Follow the expert bank's derived target
    /// (`ExecutorDomain::follow_cache_targets`): a resize it begins is held as
    /// begun, a growth as one and a shrink as the target's. One deferred or
    /// running is followed again at the next boundary, which derives afresh.
    fn follow_cache_targets(&mut self, bindings: &mut StateBindings) -> Result<(), RequestError> {
        let (before, wake) = (self.domain().memory_moment(), self.completion());
        let followed = self.domain().follow_cache_targets(bindings, wake).map_err(classify_domain_error)?;
        let (tier, cause) = match followed {
            Followed::Grew(BankStep::Begun(tier)) => (tier, BankBegunFor::Growth),
            Followed::Shrank(BankStep::Begun(tier)) => (tier, BankBegunFor::Shrink),
            Followed::Holds | Followed::ReleasingSlots | Followed::Grew(_) | Followed::Shrank(_) => return Ok(()),
        };
        self.memory().bank_begun(tier, BankBegun { held: before, cause }).map_err(invariant("expert bank"))
    }

    /// At a flight boundary, finish the idle graph slots' release off the
    /// submitting thread (`ExecutorDomain::advance_slot_release`), recorded
    /// with the slots and how long the release thread took. It freed
    /// memory, so what waited on memory may retry.
    fn advance_slot_release(&mut self) -> Result<(), RequestError> {
        let Some(released) = self.domain().advance_slot_release().map_err(classify_domain_error)? else {
            return Ok(());
        };
        let after = self.domain().memory_moment();
        tracing::info!(
            slots = released.slots,
            released = released.bytes,
            release_micros = released.released.duration_since(released.began).as_micros() as u64,
            charged_after = after.charged,
            claimed_after = after.claimed,
            footprint_after = after.footprint,
            "graph slots released"
        );
        self.advance_epoch();
        Ok(())
    }

    /// At a flight boundary, advance the expert bank's resize
    /// (`ExecutorDomain::advance_expert_bank`). A finished one is recorded as
    /// a release is (a shrink as memory released, a growth as memory
    /// grown): how long it held this thread, the bank's slots after it, and
    /// what the process held when it began and now. A finished shrink freed
    /// memory: what waited on memory may retry.
    fn advance_expert_bank(&mut self) -> Result<(), RequestError> {
        let wake = self.completion();
        let Some((finished, held)) = self.domain().advance_expert_bank(wake).map_err(classify_domain_error)? else {
            return Ok(());
        };
        let after = self.domain().memory_moment();
        let tier = self.domain().expert_holding_storage(finished.holding()).map_err(classify_domain_error)?;
        let BankBegun { held: before, cause } = self.memory().bank_began[tier.index()]
            .take()
            .ok_or_else(|| invariant("expert bank")(format!("a {tier:?} resize finished that never began")))?;
        let at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| invariant("expert bank")(format!("the clock is before the Unix epoch: {error}")))?
            .as_millis() as u64;
        let resized = |from, to| cause.finished(from, to, at_unix_ms).map_err(invariant("expert bank"));
        // A finished resize has held this thread for a measured time, by
        // source: the call that began it, the boundaries' advances while it
        // ran, and the decisions that began nothing meanwhile.
        let micros = (held.total() * 1e6) as u64;
        let begun_micros = (held.begun * 1e6) as u64;
        let (advances, advance_micros) = (held.advances.count, (held.advances.seconds * 1e6) as u64);
        let (decisions, decision_micros) = (held.decisions.count, (held.decisions.seconds * 1e6) as u64);
        // Each step of the resize, in microseconds: those on this thread, and
        // how long before now the others happened, so a decode stall around
        // it is placed on the decode's timeline.
        let now = std::time::Instant::now();
        let before_now = |at: std::time::Instant| now.duration_since(at).as_micros() as u64;
        match finished {
            BankResized::Shrunk { holding, from, to, released, times } => {
                self.domain().record_expert_resize(holding, resized(from, to)?);
                tracing::info!(
                    micros,
                    begun_micros,
                    advances,
                    advance_micros,
                    decisions,
                    decision_micros,
                    bank_slots = self.domain().expert_bank_slots(),
                    ?holding,
                    from,
                    to,
                    released,
                    charged_before = before.charged,
                    charged_after = after.charged,
                    claimed_before = before.claimed,
                    claimed_after = after.claimed,
                    footprint_before = before.footprint,
                    footprint_after = after.footprint,
                    deciding_micros = times.decided.duration_since(held.started).as_micros() as u64,
                    queued_micros = times.queued.as_micros() as u64,
                    planned_micros = times.planned.as_micros() as u64,
                    unpinned_micros = times.unpinned.as_micros() as u64,
                    plan_picked_micros = times.trips[0].picked.as_micros() as u64,
                    plan_served_micros = times.trips[0].served.as_micros() as u64,
                    commit_picked_micros = times.trips[1].picked.as_micros() as u64,
                    commit_served_micros = times.trips[1].served.as_micros() as u64,
                    moves_micros = times.moves.as_micros() as u64,
                    decided_before = before_now(times.decided),
                    prepared_before = before_now(times.prepared),
                    installed_before = before_now(times.installed),
                    released_before = before_now(times.released),
                    "memory released"
                );
                self.advance_epoch();
            }
            // A growth's `prepared` is when its first slot joined and
            // `installed` its last (`ResizeTimes`).
            BankResized::Grown { holding, from, to, grown, refused, times } => {
                self.domain().record_expert_resize(holding, resized(from, to)?);
                tracing::info!(
                    micros,
                    begun_micros,
                    advances,
                    advance_micros,
                    decisions,
                    decision_micros,
                    bank_slots = self.domain().expert_bank_slots(),
                    ?holding,
                    from,
                    to,
                    grown,
                    refused = refused.as_ref().map(ToString::to_string),
                    charged_before = before.charged,
                    charged_after = after.charged,
                    claimed_before = before.claimed,
                    claimed_after = after.claimed,
                    footprint_before = before.footprint,
                    footprint_after = after.footprint,
                    decided_before = before_now(times.decided),
                    prepared_before = before_now(times.prepared),
                    installed_before = before_now(times.installed),
                    released_before = before_now(times.released),
                    "memory grown"
                )
            }
            // Nothing changed: no resize of the holding is recorded.
            BankResized::Stopped { holding, at } => {
                tracing::info!(?holding, at, "the expert bank's growth stopped before its first slot")
            }
            // Only a growth needs memory: a shrink's refusal broke the bank,
            // which reports it as an invariant violation instead.
            BankResized::GrowthRefused { holding, at, refusal } => {
                if cause != BankBegunFor::Growth {
                    return Err(invariant("expert bank")(format!("a shrink of {holding:?} reported a refusal: {refusal}")));
                }
                tracing::warn!(?holding, at, %refusal, "the expert bank's storage refused a growth")
            }
        }
        Ok(())
    }

}

impl<H: MemoryHost> MemoryStep for H {}

/// The owner's memory step at a flight boundary, `bindings` there where no
/// flight holds them, `chunk_rows` the rows of the batch in flight: memory
/// read when due, the expert bank's running resize advanced and its derived
/// target followed, and the model unloaded at the reading where the
/// guarantee cannot be kept. Whether it unloaded. The one step every host
/// runs: the owner at each of its boundaries, and an instrument that drives
/// the domain below the scheduler at each of its own.
pub fn between_flights<H: MemoryHost>(
    host: &mut H,
    bindings: Option<&mut StateBindings>,
    chunk_rows: u64,
) -> Result<bool, RequestError> {
    host.observe(bindings, chunk_rows)
}

impl<F: ProgramFamily> Service<F> {
    /// Record that blocked work needs `required` bytes of `resource`, a
    /// store's memory, keeping the least requirement per store so the
    /// earliest retry is kept.
    fn record_memory_deficit(&mut self, resource: ResourceKind, required: u64) {
        if !matches!(resource, ResourceKind::DeviceMemory | ResourceKind::HostStaging) {
            return;
        }
        match self
            .memory
            .deficits
            .iter_mut()
            .find(|deficit| deficit.resource == resource)
        {
            Some(deficit) => deficit.required = deficit.required.min(required),
            None => self.memory.deficits.push(MemoryDeficit { resource, required }),
        }
    }

    /// The wait for a capacity deficit, recorded at the current epoch: after
    /// every release has been tried.
    fn capacity_wait(&mut self, resource: ResourceKind, required: u64, available: u64) -> Wait {
        self.record_memory_deficit(resource, required);
        Wait::Capacity {
            at: self.epoch,
            resource,
            required,
            available,
        }
    }

    /// Whether `request` waits on memory to become resident: it never was,
    /// and memory is being reclaimed or its last attempt was refused for
    /// memory. Preemption is the only eviction and it counts itself
    /// (`preemption_debt`), so a non-resident request without that debt never
    /// was resident; a preempted one is not refused, mid-prefill included.
    /// Such a request is the one refused once only the model's unload could
    /// make room for it.
    fn waits_on_memory(&self, request: &Request) -> bool {
        request.record.preemption_debt == 0
            && matches!(&request.state, State::Between(_, Stage::NonResident(Some(Wait::Memory { .. }))))
    }

    /// When nothing can move the epoch (no flight: this runs in `Idle`, and
    /// no request waits on its host or on memory), every capacity wait ends
    /// in its typed error. Returns whether any did.
    fn fail_capacity_waiters(&mut self) -> bool {
        let independent = self.requests.values().any(|request| {
            request.state.generation().finish_reason().is_some()
                || matches!(request.state, State::Between(_, Stage::AwaitingCredit))
                || matches!(request.state.wait(), Some(Wait::Memory { .. }))
        });
        if independent {
            return false;
        }
        let waiting = self
            .requests
            .iter()
            .filter_map(|(&id, request)| match request.state.wait() {
                Some(Wait::Capacity {
                    resource,
                    required,
                    available,
                    ..
                }) => Some((id, resource, required, available)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if waiting.is_empty() {
            return false;
        }
        for (id, resource, required, available) in waiting {
            let Request { mut record, state } = self.requests.remove(&id).expect("waiting request");
            let mut generation = state.terminate(false);
            Self::fail_generation(
                &mut record,
                &mut generation,
                RequestError::Capacity(ServiceCapacityError {
                    resource: CapacityResource::Execution(resource),
                    required,
                    available,
                }),
            );
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::Ready),
                },
            );
        }
        self.epoch.advance();
        true
    }

    /// End the waits that have ended, then take the requests selection can
    /// start out of the map.
    fn candidates(&mut self) -> BTreeMap<RequestId, (Record, Candidate)> {
        let epoch = self.epoch;
        let ended_wait = |wait: Option<Wait>| match wait {
            Some(Wait::Capacity { at, .. } | Wait::Memory { at }) => epoch.changed_since(at),
            None => true,
        };
        let awaiting = self
            .requests
            .iter()
            .filter_map(|(&id, request)| match request.state {
                State::Between(_, Stage::AwaitingPrefix { through }) => Some((id, through)),
                _ => None,
            })
            .collect::<Vec<_>>();
        for (id, through) in awaiting {
            let generation = self.requests[&id].state.generation();
            let cached = self
                .prefix_cache
                .deepest(PrefixPath::of(generation), generation.resume_bound())
                >= through;
            if cached || !self.prefix_pending(through, id) {
                if let State::Between(_, stage) = &mut self.requests.get_mut(&id).unwrap().state {
                    *stage = Stage::NonResident(None);
                }
            }
        }
        let selectable = self
            .requests
            .iter()
            .filter(|(_, request)| match &request.state {
                State::Between(generation, Stage::Ready) => {
                    generation.finish_reason().is_none()
                        || owes_reconciliation(&request.record, generation)
                }
                State::Between(_, Stage::NonResident(wait) | Stage::Pending(_, wait))
                | State::Parked(_, _, wait) => ended_wait(*wait),
                State::Between(_, Stage::AwaitingPrefix { .. } | Stage::AwaitingCredit) => false,
            })
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        selectable
            .into_iter()
            .map(|id| {
                let Request { record, state } = self.requests.remove(&id).expect("selectable");
                let candidate = match state {
                    State::Between(generation, Stage::Ready) => Candidate::Ready(generation),
                    State::Between(generation, Stage::NonResident(_)) => {
                        Candidate::NonResident(generation)
                    }
                    State::Between(generation, Stage::Pending(work, _)) => {
                        Candidate::Pending(generation, work)
                    }
                    State::Parked(round, work, _) => Candidate::Parked(round, work),
                    State::Between(_, Stage::AwaitingPrefix { .. } | Stage::AwaitingCredit) => {
                        unreachable!("filtered above")
                    }
                };
                (id, (record, candidate))
            })
            .collect()
    }

    fn select(&mut self, bindings: &mut StateBindings) -> Result<Selected, Fatal> {
        let mut candidates = self.candidates();
        let mut holder = self.domain.chain_holder();
        // A holder that is no longer scheduled ended mid-chain: its chain is
        // released, and its state with it.
        if holder.is_some_and(|holder| !candidates.contains_key(&holder)) {
            self.domain
                .release_chain()
                .map_err(|error| (None, classify_domain_error(error)))?;
            holder = None;
        }
        let operations = candidates
            .iter()
            .map(|(&id, (record, candidate))| {
                let generation = candidate.generation();
                let reconciliation = owes_reconciliation(record, generation);
                let prefill = !reconciliation
                    && (!generation.is_resident()
                        || generation.resident_position() < generation.usage().prompt_tokens);
                ScheduledOperation {
                    identity: id,
                    phase: if prefill { Phase::Prefill } else { Phase::Decode },
                    active: record.service_ns > 0,
                    resident: generation.is_resident(),
                    waiting_since_ns: record.waiting_since,
                    service_ns: record.service_ns,
                    preemption_debt: record.preemption_debt,
                    // Its next chunk claims the chain's queued launch before
                    // any other round could orphan it.
                    holds_chain: holder == Some(id),
                }
            })
            .collect::<Vec<_>>();
        let selection = self
            .scheduler
            .select(&operations, self.now)
            .map_err(|error| (None, invariant("scheduler")(error)))?;
        let Some(mut selection) = selection else {
            // Idle (no request is live, not merely none schedulable now):
            // release state backing well beyond what the stores need.
            let live = self.requests.values().any(|request| {
                request.state.generation().is_resident()
                    && request.state.generation().finish_reason().is_none()
            });
            if !live {
                let wake = self.wakes.completion();
                self.domain
                    .shrink_state(bindings, ShrinkPolicy::Idle, wake)
                    .map_err(|error| (None, classify_domain_error(error)))?;
            }
            self.return_candidates(candidates);
            return Ok(Selected::Empty);
        };
        // Requests take rows from one phase-wide token budget in selection
        // order. Additional ready requests wait for the next round.
        let mut remaining_rows = match selection.phase() {
            Phase::Prefill => self.scheduler.limits().prefill_tokens,
            Phase::Decode => self.scheduler.limits().decode_tokens,
        };
        // The round's vocabulary work: selected target rows and drafting
        // requests together stay within the selection bound. Every such
        // charge also takes a row from the decode budget, so only a prefill
        // round finishing more prompts than the bound meets it.
        let mut remaining_selections = self.scheduler.limits().selection_bound();
        let mut attempted = 0;
        let mut members = BTreeMap::new();
        let mut operations = Vec::new();
        // Requests whose drafted blocks await verification share the launch
        // that verifies them: its rows go to the proposals most often right,
        // whichever request drafted them.
        let drafted = selection
            .requests()
            .iter()
            .filter_map(|id| Some((*id, candidates[id].1.generation().draft_outlook()?)))
            .collect::<Vec<_>>();
        let outlooks = drafted
            .iter()
            .map(|(_, outlook)| outlook.clone())
            .collect::<Vec<_>>();
        let shares = magnitude_generation::share_rows(
            &outlooks,
            remaining_rows * drafted.len() / selection.requests().len().max(1),
        )
        .map(|shares| {
            drafted
                .iter()
                .map(|(id, _)| *id)
                .zip(shares)
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
        for &id in selection.requests() {
            if remaining_rows == 0 {
                break;
            }
            // Share the remaining rows across requests still eligible in this
            // round. A lone request can use the whole budget, while peers get
            // compatible chunks instead of the first request consuming it.
            let allowance = shares
                .get(&id)
                .copied()
                .unwrap_or_else(|| remaining_rows.div_ceil(selection.requests().len() - attempted))
                .min(remaining_rows);
            attempted += 1;
            let (record, candidate) = candidates.remove(&id).expect("selected candidate");
            // Work composed before this round joins as it is, or waits.
            if candidate
                .composed()
                .is_some_and(|work| selection_charge(work) > remaining_selections)
            {
                self.requests.insert(
                    id,
                    Request {
                        record,
                        state: candidate.into_state(),
                    },
                );
                continue;
            }
            // On a fatal error every selected request, started or not, goes
            // back where the owner terminates it with that error: one held
            // here would be dropped without its outcome.
            let started = match self.start(
                id,
                record,
                candidate,
                allowance,
                remaining_selections,
                bindings,
            ) {
                Ok(started) => started,
                Err((_, error)) => {
                    self.return_candidates(candidates);
                    return Err((Some(members), error));
                }
            };
            match started {
                Start::Joined(member, mut next) => {
                    let rows = next
                        .iter()
                        .filter(|operation| {
                            matches!(operation, Operation::Forward { .. } | Operation::Head { .. })
                        })
                        .map(Operation::row_count)
                        .sum::<usize>();
                    remaining_rows = remaining_rows.saturating_sub(rows);
                    let charge = selection_charge(&next);
                    members.insert(id, member);
                    let Some(left) = remaining_selections.checked_sub(charge) else {
                        self.return_candidates(candidates);
                        let error = invariant("scheduler")(
                            "a started member exceeds the round's selection bound".into(),
                        );
                        return Err((Some(members), error));
                    };
                    remaining_selections = left;
                    operations.append(&mut next);
                }
                Start::Returned(request) => {
                    self.requests.insert(id, request);
                }
            }
        }
        self.return_candidates(candidates);
        selection
            .truncate(attempted)
            .map_err(|error| (None, invariant("scheduler")(error)))?;
        if operations.is_empty() {
            self.scheduler.completed(selection, 0);
            return Ok(Selected::Retry);
        }
        let rows = operations
            .iter()
            .filter(|operation| matches!(operation, Operation::Forward { .. } | Operation::Head { .. }))
            .map(Operation::row_count)
            .sum();
        Ok(Selected::Round(Round {
            selection,
            started: self.now,
            rows,
            members,
            groups: group(&self.domain, operations).into_iter().collect(),
        }))
    }

    fn return_candidates(&mut self, candidates: BTreeMap<RequestId, (Record, Candidate)>) {
        for (id, (record, candidate)) in candidates {
            self.requests.insert(
                id,
                Request {
                    record,
                    state: candidate.into_state(),
                },
            );
        }
    }

    /// Start a selected request: its pending work, residency, or a round.
    fn start(
        &mut self,
        id: RequestId,
        mut record: Record,
        candidate: Candidate,
        allowance: usize,
        selections: usize,
        bindings: &mut StateBindings,
    ) -> Result<Start, Fatal> {
        record.begin_service(&candidate);
        let generation = match candidate {
            Candidate::Parked(round, work) => {
                return Ok(Start::Joined(
                    Member {
                        record,
                        state: Membership::Started(round),
                    },
                    work.0,
                ));
            }
            Candidate::Pending(generation, work) => {
                return Ok(Start::Joined(
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                    work.0,
                ));
            }
            Candidate::Ready(generation) => generation,
            Candidate::NonResident(generation) => {
                let residency = match self.make_resident(id, &mut record, generation, bindings) {
                    Ok(residency) => residency,
                    // No map holds `record` now: it takes the error itself.
                    Err((members, error)) => {
                        if let Some(sender) = record.publication.take() {
                            sender.fail(error.clone());
                        }
                        return Err((members, error));
                    }
                };
                match residency {
                    Residency::Resident(generation, encodes) => match Work::new(encodes) {
                        Some(work) => {
                            return Ok(Start::Joined(
                                Member {
                                    record,
                                    state: Membership::Generation(generation),
                                },
                                work.0,
                            ));
                        }
                        None => generation,
                    },
                    Residency::Waiting(generation, wait) => {
                        return Ok(Start::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::NonResident(Some(wait))),
                        }));
                    }
                }
            }
        };
        self.start_round(id, record, generation, allowance, selections)
            .map_err(|error| (None, error))
    }

    fn start_round(
        &mut self,
        id: RequestId,
        mut record: Record,
        generation: Generation,
        allowance: usize,
        selections: usize,
    ) -> Result<Start, RequestError> {
        let fail = |mut record: Record, mut generation: Generation, detail: String| {
            Self::fail_generation(
                &mut record,
                &mut generation,
                invariant("generation round start")(detail),
            );
            Start::Returned(Request {
                record,
                state: State::Between(generation, Stage::Ready),
            })
        };
        if owes_reconciliation(&record, &generation) {
            return Ok(match generation.start_reconciliation(allowance) {
                Ok(round) => match lower_round(&round, id, round.generation().resident_position(), None) {
                    Ok(operation) => Start::Joined(
                        Member {
                            record,
                            state: Membership::Started(round),
                        },
                        vec![operation],
                    ),
                    Err(detail) => fail(record, round.fail(), detail),
                },
                Err((generation, detail)) => fail(record, generation, detail),
            });
        }
        // A prefill chunk ends exactly at each planned branch point, and at
        // the prompt's retained boundary one row before its end.
        let resident = generation.resident_position();
        let allowance = next_stop(&record, &generation, resident)
            .map_or(allowance, |stop| allowance.min(stop - resident));
        // With the round's selection bound spent, a chunk that would reach
        // the end of the prompt (and select there) stops one row short and
        // finishes in a later round; with no row left before the end, the
        // request waits. Decode never gets here with the bound spent.
        let allowance = if selections == 0 {
            allowance.min(
                generation
                    .resume_bound()
                    .saturating_sub(1)
                    .saturating_sub(resident),
            )
        } else {
            allowance
        };
        if allowance == 0 {
            return Ok(Start::Returned(Request {
                record,
                state: State::Between(generation, Stage::Ready),
            }));
        }
        let bound = match generation.publication_bound(allowance) {
            Ok(bound) => bound,
            Err(detail) => return Ok(fail(record, generation, detail)),
        };
        if bound != 0 {
            if let Some(sender) = &record.publication {
                let slots = bound.div_ceil(record.publication_batch_limit);
                match sender.reserve_outputs(slots) {
                    Ok(permits) => record.publication_permits.extend(permits),
                    Err(PublishError::Full) => {
                        return Ok(Start::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::AwaitingCredit),
                        }));
                    }
                    Err(PublishError::ReceiverClosed) => {
                        let mut generation = generation;
                        generation.cancel();
                        generation.discard_output();
                        record.release_publication();
                        self.epoch.advance();
                        return Ok(Start::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::Ready),
                        }));
                    }
                }
            }
        }
        Ok(match generation.start_round(id, allowance) {
            Ok(RoundStart::Target(round)) => {
                // Where the next round prepares to draft, the decode queued
                // behind this one publishes its feature as that round.
                self.domain.publish_behind(id, round.publishes_behind());
                match lower_round(&round, id, round.generation().resident_position(), None) {
                    Ok(operation) => Start::Joined(
                        Member {
                            record,
                            state: Membership::Started(round),
                        },
                        vec![operation],
                    ),
                    Err(detail) => fail(record, round.fail(), detail),
                }
            }
            Ok(RoundStart::Method(generation, operations)) => {
                if operations.iter().any(|operation| operation.request() != id) {
                    fail(
                        record,
                        generation,
                        "method returned an operation for another request".into(),
                    )
                } else {
                    Start::Joined(
                        Member {
                            record,
                            state: Membership::Generation(generation),
                        },
                        operations,
                    )
                }
            }
            Err((generation, detail)) => fail(record, generation, detail),
        })
    }

    /// The single transition that makes a request resident: from the deepest
    /// cached prefix of its path below its resume bound, or from fresh state.
    fn make_resident(
        &mut self,
        id: RequestId,
        record: &mut Record,
        mut generation: Generation,
        bindings: &mut StateBindings,
    ) -> Result<Residency, Fatal> {
        let fatal = |error: RequestError| (None, error);
        let hit = if record.prefix_cache {
            self.prefix_cache
                .lookup(PrefixPath::of(&generation), generation.resume_bound())
                .map_err(|error| fatal(invariant("prefix cache")(error)))?
        } else {
            None
        };
        let opened = match hit {
            Some(hit) => {
                let cached = self
                    .prefix_cache
                    .entry(hit)
                    .map_err(|error| fatal(invariant("prefix cache")(error)))?;
                self.domain.open_state(bindings, id, Some(cached.state()))
            }
            None => self
                .ensure_open_capacity(bindings)
                .and_then(|()| self.domain.open_state(bindings, id, None)),
        };
        let encodes = match opened {
            Ok(encodes) => encodes,
            Err(DomainError::Capacity(error)) => {
                let wait = self.capacity_wait(error.resource, error.required, error.available);
                return Ok(Residency::Waiting(generation, wait));
            }
            Err(DomainError::Blind(_)) => {
                return Ok(Residency::Waiting(generation, Wait::Memory { at: self.epoch }));
            }
            Err(error) if is_fatal(&error) => return Err(fatal(classify_domain_error(error))),
            Err(error) => {
                Self::fail_generation(record, &mut generation, classify_domain_error(error));
                return Ok(Residency::Resident(generation, Vec::new()));
            }
        };
        let resumed = match hit {
            Some(hit) => self
                .prefix_cache
                .entry(hit)
                .and_then(|cached| generation.resume_at(Some((hit.position(), cached.method())))),
            None => generation.resume_at(None),
        };
        if let Err(error) = resumed {
            return match self.domain.release_state(&[id]) {
                Ok(_) => {
                    Self::fail_generation(record, &mut generation, invariant("residency")(error));
                    Ok(Residency::Resident(generation, Vec::new()))
                }
                Err(rollback) => Err(fatal(invariant("residency")(format!(
                    "{error}; executor rollback failed ({rollback})"
                )))),
            };
        }
        // The table rows of the tokens the host knows it will feed, its path
        // from where its state resumes, are prefetched ahead of their steps.
        let position = generation.resident_position();
        let path = generation.prompt().iter().chain(generation.generated()).skip(position).copied().collect::<Vec<_>>();
        if let Err(error) = self.domain.prefetch_prompt(id, position, &path) {
            return Err(fatal(invariant("stored table prefetch")(error)));
        }
        record.prefix_source = hit.map(|hit| hit.id());
        // Its host answers it admitted only now: one refused before this
        // never was.
        if let Some(sender) = &record.publication {
            sender.began();
        }
        Ok(Residency::Resident(generation, encodes))
    }

    /// Release down the whole ladder until a new request's state opens.
    fn ensure_open_capacity(&mut self, bindings: &mut StateBindings) -> Result<(), DomainError> {
        let requirements = self.domain.open_requirements();
        let mut current = self.open_capacity(bindings, requirements);
        let held = self.held_entries(None);
        while matches!(current, Err(DomainError::Capacity(_)))
            && self.release_once(bindings, true, &held, &[], true)?
        {
            current = self.open_capacity(bindings, requirements);
        }
        current
    }

    fn open_capacity(
        &mut self,
        bindings: &mut StateBindings,
        requirements: OpenRequirements,
    ) -> Result<(), DomainError> {
        self.domain.provision_open(bindings)?;
        self.domain
            .can_open(requirements)
            .map_err(DomainError::from)
    }

    /// Submit the round's next groups until one is on the device or the
    /// round is exhausted.
    fn advance(&mut self, mut round: Round, mut bindings: StateBindings) -> Result<Pipeline<F>, Fatal> {
        loop {
            let Some(group_) = round.groups.pop_front() else {
                self.finish_round(round)
                    .map_err(|error| (None, error))?;
                return Ok(Pipeline::Idle(bindings));
            };
            // Work of a request that ended while the round ran is dropped.
            let live = group_
                .operations()
                .iter()
                .all(|operation| {
                    round
                        .members
                        .get(&operation.request())
                        .is_some_and(|member| !ended(member.state.generation()))
                });
            if !live {
                let live = group_
                    .into_operations()
                    .into_iter()
                    .filter(|operation| {
                        round
                            .members
                            .get(&operation.request())
                            .is_some_and(|member| !ended(member.state.generation()))
                    })
                    .collect::<Vec<_>>();
                for group_ in group(&self.domain, live).into_iter().rev() {
                    round.groups.push_front(group_);
                }
                continue;
            }
            let chunk_rows = self.scheduler.limits().prefill_tokens.max(1);
            let decode_owed = round.selection.contended().then(|| self.scheduler.limits().decode_owed());
            let alone = round.members.len() == 1
                && !self
                    .requests
                    .values()
                    .any(|other| competes(other, self.status(other)));
            // A chain opens only behind the round's last group: a later one would orphan it.
            let last = round.groups.is_empty();
            let planned = self.domain.plan_prompt(decode_owed, alone, last, |chunks| {
                planned_prefill(&round, group_.operations(), chunk_rows, chunks)
            });
            if let Err(error) = planned {
                return Err((Some(round.members), error));
            }
            let group_ = match self.provision(&mut bindings, &round, group_) {
                Ok(Provisioned::Fits(group_)) => group_,
                Ok(Provisioned::Split(leading, trailing)) => {
                    round.groups.push_front(trailing);
                    round.groups.push_front(leading);
                    continue;
                }
                Ok(Provisioned::Short(group_, wait)) => {
                    self.park(&mut round, group_, wait);
                    continue;
                }
                Err(error) => return Err((Some(round.members), error)),
            };
            match submit_group(&mut self.domain, bindings, &group_, self.wakes.completion()) {
                Ok(flight) => {
                    let mut prefix_uses = Vec::new();
                    let requests = group_
                        .operations()
                        .iter()
                        .map(Operation::request)
                        .collect::<BTreeSet<_>>();
                    for request in requests {
                        let source = round
                            .members
                            .get(&request)
                            .and_then(|member| member.record.prefix_source);
                        if let Some(source) = source {
                            match self.prefix_cache.begin_submitted_if_cached(source) {
                                Ok(true) => prefix_uses.push(source),
                                Ok(false) => {}
                                Err(error) => {
                                    return Err((Some(round.members), invariant("prefix cache")(error)));
                                }
                            }
                        }
                    }
                    return Ok(Pipeline::InFlight(
                        round,
                        ActiveGroup {
                            flight,
                            operations: group_.into_operations(),
                            prefix_uses,
                        },
                    ));
                }
                Err(SubmitFailure::Refused(error, returned)) => {
                    bindings = returned;
                    match error {
                        DomainError::Blind(_) => {
                            let wait = Wait::Memory { at: self.epoch };
                            self.park(&mut round, group_, wait);
                        }
                        DomainError::Input(error) => {
                            let requests = group_
                                .operations()
                                .iter()
                                .map(Operation::request)
                                .collect::<BTreeSet<_>>();
                            for request in requests {
                                Self::fail_member(
                                    &mut round,
                                    request,
                                    RequestError::Input(error.clone()),
                                );
                            }
                            self.epoch.advance();
                        }
                        other => return Err((Some(round.members), classify_domain_error(other))),
                    }
                }
                Err(SubmitFailure::Failed(error)) => {
                    return Err((Some(round.members), classify_domain_error(error)));
                }
            }
        }
    }

    /// Provision a group's state and check its reservation. A requirement
    /// and a growth claim are one capacity question: recheck both after each
    /// release. Selection is provisional until every exact requirement is
    /// available: a deficit releases down the ladder short of preemption,
    /// then sheds the group's trailing operations, which loses no work, and
    /// only then preempts eligible live victims outside the round.
    fn provision(
        &mut self,
        bindings: &mut StateBindings,
        round: &Round,
        group_: OperationGroup,
    ) -> Result<Provisioned, RequestError> {
        let short = |current: &Result<(), DomainError>| matches!(current, Err(DomainError::Capacity(_)));
        let mut group_ = group_;
        let mut current = self.fits(bindings, round, &group_);
        // Retained prefixes never make the stores grow past the workload
        // state the plan charges: the group takes the rows and banks of the
        // least recently used ones instead. Once none is left to release,
        // `fits` sets no ceiling.
        let held = self.held_entries(Some(&round.members));
        while matches!(current, Err(DomainError::StateCeiling { .. })) {
            self.prefix_cache.release_unspared(&held);
            current = self.fits(bindings, round, &group_);
        }
        // The ledger named the sequences whose state the group's rows
        // preempt: evicted, the group provisions again, which completes its
        // claim; any it cannot evict now leaves the group waiting on memory.
        if let Err(DomainError::Preempt(sequences)) = &current {
            let sequences = sequences.clone();
            if self.evict_named(round, &sequences).map_err(classify_domain_error)? {
                current = self.fits(bindings, round, &group_);
            }
        }
        while short(&current)
            && self
                .release_once(bindings, false, &held, &[], false)
                .map_err(classify_domain_error)?
        {
            current = self.fits(bindings, round, &group_);
        }
        if short(&current) {
            group_ = match group_.split_last() {
                Ok((leading, trailing)) => return Ok(Provisioned::Split(leading, trailing)),
                Err(whole) => whole,
            };
            let selected = round.members.keys().copied().collect::<Vec<_>>();
            while short(&current)
                && self
                    .release_once(bindings, true, &held, &selected, false)
                    .map_err(classify_domain_error)?
            {
                current = self.fits(bindings, round, &group_);
            }
        }
        match current {
            Ok(()) => Ok(Provisioned::Fits(group_)),
            Err(DomainError::Blind(_) | DomainError::Preempt(_)) => Ok(Provisioned::Short(group_, Wait::Memory { at: self.epoch })),
            Err(DomainError::Capacity(deficit)) => {
                let wait = self.capacity_wait(deficit.resource, deficit.required, deficit.available);
                Ok(Provisioned::Short(group_, wait))
            }
            Err(error) => Err(classify_domain_error(error)),
        }
    }

    /// Whether the group fits now: provision it, then check the reservation
    /// its requirement, derived from the layout it was just provisioned
    /// into, would take. While a retained prefix could be released instead,
    /// state growth stops at the workload state the plan charges.
    fn fits(
        &mut self,
        bindings: &mut StateBindings,
        round: &Round,
        group_: &OperationGroup,
    ) -> Result<(), DomainError> {
        let held = self.held_entries(Some(&round.members));
        let ceiling = self
            .prefix_cache
            .holds_unspared(&held)
            .then(|| self.prefix_cache.workload_state_bytes());
        self.domain.provision(bindings, group_.operations(), ceiling)?;
        let requirement = requirements(&self.domain, group_)?;
        self.domain
            .can_reserve(&requirement)
            .map_err(DomainError::from)
    }

    /// Move a group that cannot proceed out of the round: each of its
    /// requests takes all of its remaining work with it and waits alone.
    fn park(&mut self, round: &mut Round, group_: OperationGroup, wait: Wait) {
        let requests = group_
            .operations()
            .iter()
            .map(Operation::request)
            .collect::<BTreeSet<_>>();
        if matches!(wait, Wait::Memory { .. }) {
            tracing::info!(
                requests = ?requests,
                rows = group_.operations().iter().map(Operation::row_count).sum::<usize>(),
                "a group waits on memory"
            );
        }
        let mut operations = group_.into_operations();
        let groups = std::mem::take(&mut round.groups);
        let mut kept = Vec::new();
        for queued in groups {
            let (theirs, others): (Vec<_>, Vec<_>) = queued
                .into_operations()
                .into_iter()
                .partition(|operation| requests.contains(&operation.request()));
            operations.extend(theirs);
            kept.extend(others);
        }
        round.groups = group(&self.domain, kept).into_iter().collect();
        for request in requests {
            let Some(Member { record, state }) = round.members.remove(&request) else {
                continue;
            };
            let theirs = operations
                .iter()
                .filter(|operation| operation.request() == request)
                .cloned()
                .collect::<Vec<_>>();
            let work = Work::new(theirs).expect("a parked request has work in the group");
            let state = match state {
                Membership::Started(started) => State::Parked(started, work, Some(wait)),
                Membership::Generation(generation) => {
                    State::Between(generation, Stage::Pending(work, Some(wait)))
                }
            };
            self.requests.insert(request, Request { record, state });
        }
    }

    /// Reconcile the completed group and take back the bindings.
    fn reconcile(&mut self, mut round: Round, active: ActiveGroup<F>) -> Result<Pipeline<F>, Fatal> {
        let ActiveGroup {
            flight,
            operations,
            prefix_uses,
        } = active;
        for source in prefix_uses {
            if let Err(error) = self.prefix_cache.end_submitted(source) {
                return Err((Some(round.members), invariant("prefix cache")(error)));
            }
        }
        let (outcomes, continuation) = match finish(&mut self.domain, flight, closes_prompt(&round, &operations)) {
            Ok(finished) => finished,
            Err(error) => return Err((Some(round.members), classify_domain_error(error))),
        };
        let reconciled = match outcomes {
            FlightOutcomes::Target(pending) => self.reconcile_target(&mut round, pending, &operations),
            FlightOutcomes::Head(pending) => self.reconcile_method(&mut round, pending, &operations),
            FlightOutcomes::Vision(item) => self.reconcile_vision(&mut round, item, &operations),
            FlightOutcomes::Priming(request, duration) => {
                if let Some(member) = round.members.get_mut(&request) {
                    member.record.add_physical_duration(duration);
                }
                Ok(())
            }
        };
        if let Err(error) = reconciled {
            return Err((Some(round.members), error));
        }
        let mut bindings = match continuation {
            magnitude_executor::TargetContinuation::Ready(bindings) => bindings,
            magnitude_executor::TargetContinuation::Priming(priming) => {
                // Publish the accepted target token while the entry still owns
                // the binding right. Its own completion wakes the event pump;
                // controls remain responsive and no checkpoint can observe a
                // half-resolved pair of numerical lanes.
                self.publish_members(&mut round);
                let mut flight = DomainFlight::Priming(priming);
                flight.completion().notify(self.wakes.completion());
                return Ok(Pipeline::InFlight(round, ActiveGroup {
                    flight, operations: Vec::new(), prefix_uses: Vec::new(),
                }));
            }
        };
        self.epoch.advance();
        if let Err(error) = self.observe(Some(&mut bindings), round.rows as u64) {
            return Err((Some(round.members), error));
        }
        self.publish_members(&mut round);
        self.advance(round, bindings)
    }

    fn reconcile_target(
        &mut self,
        round: &mut Round,
        pending: Vec<PendingOperationOutcome>,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let expected = operations
            .iter()
            .map(|operation| (operation.request(), operation))
            .collect::<BTreeMap<_, _>>();
        let actual = pending
            .iter()
            .map(PendingOperationOutcome::request)
            .collect::<BTreeSet<_>>();
        if pending.len() != operations.len()
            || actual.len() != pending.len()
            || actual != expected.keys().copied().collect()
        {
            return Err(invariant("service target flight")(
                "executor outcomes do not match submitted request identities".into(),
            ));
        }
        let mut followups = Vec::new();
        let sequences = pending.len();
        let rows = operations.iter().map(Operation::row_count).sum::<usize>();
        for item in pending {
            let request = item.request();
            if !matches!(expected[&request], Operation::Forward { .. }) {
                return Err(invariant("service target flight")(
                    "target flight contains a non-forward operation".into(),
                ));
            }
            let Some(Member { mut record, state }) = round.members.remove(&request) else {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            record.add_physical_duration(item.physical_duration());
            // The member returns to the round before any fatal error is
            // returned: the owner terminates the round's members with it, and
            // a member held here would be dropped without its outcome.
            let (state, fatal) = match state {
                Membership::Started(mut started) if !ended(started.generation()) => {
                    started.timed(TimedLaunch::Target, item.launch_duration(), sequences, rows);
                    match reconcile_forward(started, &mut self.domain, request, item) {
                        Ok((generation, effects)) => {
                            followups.extend(effects.operations);
                            (Membership::Generation(generation), None)
                        }
                        // Interrupted: the generation runs the round again
                        // (or re-prefills), unfailed.
                        Err((generation, RoundError::Interrupted(_))) => (Membership::Generation(generation), None),
                        Err((mut generation, RoundError::Logical(detail))) => {
                            Self::fail_generation(
                                &mut record,
                                &mut generation,
                                invariant("generation round")(detail),
                            );
                            (Membership::Generation(generation), None)
                        }
                        Err((generation, RoundError::Physical(error))) if is_fatal(&error) => {
                            (Membership::Generation(generation), Some(classify_domain_error(error)))
                        }
                        Err((mut generation, RoundError::Physical(error))) => {
                            Self::fail_generation(
                                &mut record,
                                &mut generation,
                                classify_domain_error(error),
                            );
                            (Membership::Generation(generation), None)
                        }
                    }
                }
                state => {
                    let fatal = self.domain.abort(item).map_err(classify_domain_error).err();
                    (state, fatal)
                }
            };
            round.members.insert(request, Member { record, state });
            if let Some(error) = fatal {
                return Err(error);
            }
        }
        if !followups.is_empty() {
            round
                .groups
                .extend(group(&self.domain, followups));
        }
        Ok(())
    }

    fn reconcile_method(
        &mut self,
        round: &mut Round,
        pending: Vec<PendingOperationOutcome>,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let expected = operations
            .iter()
            .map(|operation| (operation.request(), operation))
            .collect::<BTreeMap<_, _>>();
        let actual = pending
            .iter()
            .map(PendingOperationOutcome::request)
            .collect::<BTreeSet<_>>();
        if pending.len() != operations.len()
            || actual.len() != pending.len()
            || actual != expected.keys().copied().collect()
        {
            return Err(invariant("service method flight")(
                "method outcomes do not match submitted requests".into(),
            ));
        }
        let sequences = pending.len();
        let rows = operations.iter().map(Operation::row_count).sum::<usize>();
        for item in pending {
            let request = item.request();
            let operation = expected[&request];
            let correct = matches!(operation, Operation::Head { .. })
                && matches!(item.outcome(), Outcome::Head { .. });
            let Some(member) = round.members.get_mut(&request) else {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            member.record.add_physical_duration(item.physical_duration());
            let Membership::Generation(generation) = &mut member.state else {
                return Err(invariant("service method flight")(
                    "drafter work reached a request with a started target round".into(),
                ));
            };
            if ended(generation) {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            }
            if !correct {
                self.domain.abort(item).map_err(classify_domain_error)?;
                Self::fail_generation(
                    &mut member.record,
                    generation,
                    invariant("service method flight")(
                        "method outcome differs from submitted operation".into(),
                    ),
                );
                continue;
            }
            if let Operation::Head { proposals, .. } = operation {
                let launch = if proposals.is_empty() {
                    TimedLaunch::Entry
                } else {
                    TimedLaunch::Draft
                };
                generation.timed(launch, item.launch_duration(), sequences, rows);
            }
            let transition = match generation.prepare_method_transition(operation, item.outcome()) {
                Ok(transition) => transition,
                Err(detail) => {
                    self.domain.abort(item).map_err(classify_domain_error)?;
                    Self::fail_generation(
                        &mut member.record,
                        generation,
                        invariant("generation method transition")(detail),
                    );
                    continue;
                }
            };
            match self.domain.reconcile(
                item,
                PhysicalDecision {
                    accepted_rows: transition.decision().accepted_rows,
                },
            ) {
                Ok(()) => generation.commit_method_transition(transition),
                Err(error) if is_fatal(&error) => return Err(classify_domain_error(error)),
                Err(error) => Self::fail_generation(
                    &mut member.record,
                    generation,
                    classify_domain_error(error),
                ),
            }
        }
        Ok(())
    }

    fn reconcile_vision(
        &mut self,
        round: &mut Round,
        item: PendingOperationOutcome,
        operations: &[Operation],
    ) -> Result<(), RequestError> {
        let request = item.request();
        let valid = matches!(operations,
            [Operation::Encode { request: expected, .. }] if *expected == request)
            && matches!(item.outcome(), Outcome::Encode { .. });
        if !valid {
            return Err(invariant("service vision flight")(
                "vision outcome differs from submitted request".into(),
            ));
        }
        let Some(member) = round.members.get_mut(&request) else {
            return self.domain.abort(item).map_err(classify_domain_error);
        };
        member
            .record
            .add_physical_duration(item.physical_duration());
        if ended(member.state.generation()) {
            return self.domain.abort(item).map_err(classify_domain_error);
        }
        match self.domain.reconcile(item, PhysicalDecision { accepted_rows: 0 }) {
            Ok(()) => Ok(()),
            Err(error) if is_fatal(&error) => Err(classify_domain_error(error)),
            Err(error) => {
                Self::fail_member(round, request, classify_domain_error(error));
                Ok(())
            }
        }
    }

    /// Fail a round member with `error`; its started round, if any, ends.
    fn fail_member(round: &mut Round, request: RequestId, error: RequestError) {
        if let Some(Member { mut record, state }) = round.members.remove(&request) {
            let mut generation = state.terminate(false);
            Self::fail_generation(&mut record, &mut generation, error);
            round.members.insert(
                request,
                Member {
                    record,
                    state: Membership::Generation(generation),
                },
            );
        }
    }

    /// The round is exhausted: charge its service and return its members.
    fn finish_round(&mut self, round: Round) -> Result<(), RequestError> {
        let Round {
            selection,
            started,
            rows,
            members,
            ..
        } = round;
        let elapsed = self.now.saturating_sub(started);
        self.domain.served_round(selection.phase() == Phase::Decode, rows, elapsed as f64 / 1e9);
        let count = selection.requests().len() as u64;
        let base = elapsed / count;
        let remainder = elapsed % count;
        let mut members = members;
        for (index, request) in selection.requests().iter().enumerate() {
            let share = base + u64::from((index as u64) < remainder);
            let (record, generation) = match members.get_mut(request) {
                Some(member) => (&mut member.record, member.state.generation()),
                None => match self.requests.get_mut(request) {
                    Some(request) => (&mut request.record, request.state.generation()),
                    None => continue,
                },
            };
            record.service_ns = record.service_ns.saturating_add(share);
            record.waiting_since = self.now;
            if record
                .protected_until
                .is_some_and(|boundary| generation.resident_position() > boundary)
            {
                record.protected_until = None;
            }
        }
        self.scheduler.completed(selection, elapsed);
        let ids = members.keys().copied().collect::<Vec<_>>();
        for (id, Member { mut record, state }) in members {
            let generation = match state {
                Membership::Generation(generation) => generation,
                // Every submitted target operation reconciled and every
                // unsubmitted one parked or ended, so a started round left
                // here lost its work: it fails.
                Membership::Started(started) => {
                    let mut generation = started.fail();
                    Self::fail_generation(
                        &mut record,
                        &mut generation,
                        invariant("service round")(
                            "a started round ended without its operation".into(),
                        ),
                    );
                    generation
                }
            };
            let stage = if generation.is_resident() {
                Stage::Ready
            } else {
                Stage::NonResident(None)
            };
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, stage),
                },
            );
        }
        for id in ids {
            self.retain_branch_point(id)?;
            self.retain_prefill_boundary(id)?;
            self.retain_terminal(id)?;
        }
        self.publish_ready()
    }

    /// A request whose composed work is complete: its record and generation.
    /// Retention must wait for pending method work even after target replay
    /// reaches the accepted boundary.
    fn between(&self, id: RequestId) -> Option<(&Record, &Generation)> {
        let request = self.requests.get(&id)?;
        Some((&request.record, request.state.settled()?))
    }

    /// Retain the request's reconciled state at a planned branch point it
    /// has reached: the recurrent bank and method state there, sharing every
    /// history row before it with the request itself. Points it has passed
    /// (it resumed beyond them) or can no longer reach are dropped.
    fn retain_branch_point(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        if record.branches.is_empty() {
            return Ok(());
        }
        let resident = generation.resident_position();
        let live = generation.finish_reason().is_none() && generation.is_resident();
        let reached = live && record.branches.contains(&resident);
        let branches = &mut self.requests.get_mut(&id).unwrap().record.branches;
        if live {
            branches.retain(|&branch| branch > resident);
        } else {
            branches.clear();
        }
        if reached {
            self.cache_prefix(id, resident, CachePoint::Branch)?;
        }
        Ok(())
    }

    /// Cache the request's reconciled state as the prefix of its path ending
    /// at `position`, where it is resident.
    fn cache_prefix(
        &mut self,
        id: RequestId,
        position: usize,
        point: CachePoint,
    ) -> Result<(), RequestError> {
        let state = self
            .domain
            .resume_state(id)
            .map_err(invariant("prefix retention"))?;
        if state.position() != position {
            return Err(invariant("prefix retention")(
                "cached prefix state differs from its path position".into(),
            ));
        }
        let Some(Request {
            state: State::Between(generation, _),
            ..
        }) = self.requests.get(&id)
        else {
            return Ok(());
        };
        let method = generation
            .method_checkpoint()
            .map_err(invariant("prefix retention"))?;
        let retained = self
            .prefix_cache
            .retain(PrefixPath::of(generation), position, point, state, method)
            .map_err(invariant("prefix retention"))?;
        if retained {
            let held = self.held_entries(None);
            let domain = &self.domain;
            self.prefix_cache
                .release_beyond_workload(&held, |cache| occupied_state(domain, cache))
                .map_err(invariant("prefix retention"))?;
        }
        Ok(())
    }

    /// Retain the request's prompt one row before its end (see
    /// [`prompt_boundary`]): the deepest point a request with the same prompt
    /// can resume from, since it must compute the row it samples from.
    fn retain_prefill_boundary(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        let Some(boundary) = prompt_boundary(record, generation) else {
            return Ok(());
        };
        if ended(generation)
            || generation.resident_position() != boundary
            || !generation.is_resident()
        {
            return Ok(());
        }
        self.cache_prefix(id, boundary, CachePoint::PromptBoundary)?;
        self.requests.get_mut(&id).unwrap().record.prefill_retained = true;
        Ok(())
    }

    /// Cache a finished request's whole path, prompt and reply: the next turn
    /// of a conversation extends it.
    fn retain_terminal(&mut self, id: RequestId) -> Result<(), RequestError> {
        let Some((record, generation)) = self.between(id) else {
            return Ok(());
        };
        if record.terminal_retained
            || !record.prefix_cache
            || !generation.is_resident()
            || !matches!(
                generation.finish_reason(),
                Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
            )
        {
            return Ok(());
        }
        let end = generation.reconciled_position();
        if generation.resident_position() != end {
            if generation.pending_reconciliation().is_some() {
                return Ok(());
            }
            return Err(invariant("prefix retention")(
                "terminal numerical state is not reconciled at the accepted boundary".into(),
            ));
        }
        self.cache_prefix(id, end, CachePoint::Terminal)?;
        self.requests.get_mut(&id).unwrap().record.terminal_retained = true;
        Ok(())
    }

    /// Drain round members' accepted output into their reserved permits, so
    /// publication overlaps the next submission.
    fn publish_members(&mut self, round: &mut Round) {
        let mut closed = Vec::new();
        for (&id, member) in round.members.iter_mut() {
            if let Member {
                record,
                state: Membership::Generation(generation),
            } = member
            {
                if Self::drain(record, generation) {
                    closed.push(id);
                }
            }
        }
        // A closed receiver releases its request.
        for id in closed {
            if let Some(Member { mut record, state }) = round.members.remove(&id) {
                let mut generation = state.terminate(true);
                generation.discard_output();
                record.release_publication();
                round.members.insert(
                    id,
                    Member {
                        record,
                        state: Membership::Generation(generation),
                    },
                );
            }
        }
    }

    /// Drain accepted output into reserved permits. Returns whether the
    /// receiver has closed.
    fn drain(record: &mut Record, generation: &mut Generation) -> bool {
        let Some(sender) = record.publication.as_mut() else {
            return false;
        };
        if !sender.receiver_open() {
            return true;
        }
        loop {
            let tokens = match record.pending_publication.take() {
                Some(tokens) => tokens,
                None if generation.output_len() > 0 => {
                    match generation.take(record.publication_batch_limit) {
                        Ok(tokens) => tokens,
                        Err(_) => break,
                    }
                }
                None => break,
            };
            let Some(permit) = record.publication_permits.pop_front() else {
                // Output beyond the reserved bound waits for the next round's
                // reservation.
                record.pending_publication = Some(tokens);
                break;
            };
            match sender.output_with_permit(tokens, permit) {
                Ok(()) => {}
                Err((PublishError::Full, _)) => {
                    unreachable!("a publication permit owns an exact output slot")
                }
                Err((PublishError::ReceiverClosed, _)) => return true,
            }
        }
        if record.pending_publication.is_none() && generation.output_len() == 0 {
            record.publication_permits.clear();
        }
        false
    }

    /// Publish every request outside the pipeline, and retire the ones with
    /// nothing left for their host.
    fn publish_ready(&mut self) -> Result<(), RequestError> {
        let ids = self.requests.keys().copied().collect::<Vec<_>>();
        for id in ids {
            let request = self.requests.get_mut(&id).expect("listed request");
            let State::Between(generation, _) = &mut request.state else {
                continue;
            };
            if Self::drain(&mut request.record, generation) {
                self.terminate_request(id, true);
            }
            let request = &self.requests[&id];
            let Some(generation) = request.state.settled() else {
                continue;
            };
            let record = &request.record;
            let finished = generation.finish_reason().is_some()
                && record.pending_publication.is_none()
                && generation.output_len() == 0
                && !owes_reconciliation(record, generation);
            if !finished {
                continue;
            }
            self.retain_terminal(id)?;
            let request = self.requests.get_mut(&id).expect("listed request");
            let State::Between(generation, _) = &request.state else {
                continue;
            };
            let record = &mut request.record;
            if let Some(sender) = record.publication.take() {
                match generation.finish_reason().expect("terminal generation") {
                    FinishReason::Failed => sender.fail(record.error.clone().unwrap_or_else(|| {
                        RequestError::Invariant(InvariantError {
                            context: "service request",
                            detail: "failed without a classified cause".into(),
                        })
                    })),
                    finish => sender.complete(
                        finish,
                        generation.detailed_usage(),
                        generation.method_identity().to_owned(),
                        record.physical_timings,
                    ),
                }
            }
            self.retire(id)?;
        }
        Ok(())
    }

    /// Close a terminal request's executor state and forget it.
    fn retire(&mut self, id: RequestId) -> Result<(), RequestError> {
        self.domain.close(id).map_err(invariant("request retirement"))?;
        self.requests.remove(&id);
        self.epoch.advance();
        Ok(())
    }

    fn evict_victims(
        &mut self,
        selected: &[RequestId],
        release_resident_slot: bool,
    ) -> Result<bool, DomainError> {
        let mut victims = Vec::new();
        for (&id, request) in &self.requests {
            let State::Between(generation, stage @ (Stage::Ready | Stage::AwaitingCredit)) =
                &request.state
            else {
                continue;
            };
            if selected.contains(&id)
                || !generation.is_resident()
                || generation.finish_reason().is_some()
                || request.record.protected_until.is_some()
            {
                continue;
            }
            let executor_bytes = self
                .domain
                .reclaimable(&[id])
                .map_err(release_invariant("preemption"))?;
            victims.push(Victim {
                identity: id,
                output_blocked: matches!(stage, Stage::AwaitingCredit),
                preemption_debt: request.record.preemption_debt,
                exclusive_bytes: executor_bytes
                    .checked_add(generation.method_reclaimable())
                    .ok_or_else(|| release_invariant("preemption")("request reclaim byte count overflow".into()))?,
                replay_tokens: generation.accepted_position() as u64,
                service_ns: request.record.service_ns,
            });
        }
        order_victims(&mut victims);
        let mut set = Vec::new();
        for victim in victims {
            set.push(victim.identity);
            let method_bytes = set.iter().try_fold(0u64, |total, request| {
                total
                    .checked_add(self.requests[request].state.generation().method_reclaimable())
                    .ok_or_else(|| release_invariant("preemption")("method reclaim byte count overflow".into()))
            })?;
            let reclaimable = self
                .domain
                .reclaimable(&set)
                .map_err(release_invariant("preemption"))?;
            if !release_resident_slot && reclaimable == 0 && method_bytes == 0 {
                continue;
            }
            let released = self
                .domain
                .release_state(&set)
                .map_err(release_invariant("preemption"))?;
            if !release_resident_slot && released == 0 && method_bytes == 0 {
                continue;
            }
            self.evicted(set)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Evict exactly `sequences`, the ledger's preemption of them
    /// (`DomainError::Preempt`): only where each is resident between rounds
    /// and outside `round`, else none of them. Whether they were evicted.
    fn evict_named(&mut self, round: &Round, sequences: &[RequestId]) -> Result<bool, DomainError> {
        let evictable = sequences.iter().all(|id| {
            !round.members.contains_key(id)
                && self.requests.get(id).is_some_and(|request| {
                    matches!(&request.state, State::Between(generation, Stage::Ready | Stage::AwaitingCredit)
                        if generation.is_resident())
                })
        });
        if !evictable {
            return Ok(false);
        }
        self.domain.release_state(sequences).map_err(release_invariant("preemption"))?;
        self.evicted(sequences.to_vec())?;
        Ok(true)
    }

    /// Mark `set`, whose state was just released, evicted: each waits
    /// non-resident, protected up to what it accepted, a preemption owed.
    fn evicted(&mut self, set: Vec<RequestId>) -> Result<(), DomainError> {
        for id in set {
            let Request { mut record, state } = self.requests.remove(&id).expect("victim");
            let State::Between(mut generation, _) = state else {
                unreachable!("victims are between rounds")
            };
            record.protected_until = Some(generation.accepted_position());
            record.preemption_debt = record.preemption_debt.saturating_add(1);
            generation.evicted().map_err(release_invariant("preemption"))?;
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::NonResident(None)),
                },
            );
        }
        self.epoch.advance();
        Ok(())
    }
}

/// The owner's releases for a live claim, between flights: the bindings
/// every one of them changes are here.
struct Releasing<'a, F: ProgramFamily> {
    service: &'a mut Service<F>,
    bindings: &'a mut StateBindings,
    held: &'a BTreeSet<u64>,
    spared: &'a [RequestId],
    resident_slot: bool,
    /// This release began releasing idle graph slots off the submitting
    /// thread: it is recorded as one that released now.
    slots_began: bool,
}

impl<F: ProgramFamily> Releases for Releasing<'_, F> {
    type Error = DomainError;

    /// Each release is recorded: its rung, how long it took, and what the
    /// process held on each side of it.
    fn release(&mut self, rung: Rung) -> Result<Release, DomainError> {
        let before = self.service.domain.memory_moment();
        let started = std::time::Instant::now();
        let released = self.release_at(rung)?;
        if released == Release::Released || std::mem::take(&mut self.slots_began) {
            let domain = &self.service.domain;
            let after = domain.memory_moment();
            tracing::info!(
                rung = ?rung,
                micros = started.elapsed().as_micros() as u64,
                bank_slots = domain.expert_bank_slots(),
                group_places = domain.group_places(),
                charged_before = before.charged,
                charged_after = after.charged,
                claimed_before = before.claimed,
                claimed_after = after.claimed,
                footprint_before = before.footprint,
                footprint_after = after.footprint,
                "memory released"
            );
        }
        Ok(released)
    }
}

impl<F: ProgramFamily> Releasing<'_, F> {
    fn release_at(&mut self, rung: Rung) -> Result<Release, DomainError> {
        let released = |released: bool| match released {
            true => Release::Released,
            false => Release::Nothing,
        };
        let service = &mut *self.service;
        Ok(match rung {
            // Idle graph slots release off this thread: a later boundary
            // finishes them, and nothing costlier runs before it.
            Rung::Surplus => {
                let wake = service.wakes.completion();
                let shrunk = service.domain.shrink_state(self.bindings, ShrinkPolicy::Reclaim, wake)?;
                self.slots_began = shrunk.slots == SlotRelease::Began;
                match shrunk.slots {
                    SlotRelease::Began | SlotRelease::Running => Release::Pending,
                    SlotRelease::None => released(shrunk.released != 0),
                }
            }
            Rung::Idle => released(
                service.domain.reclaim_idle(self.bindings).map_err(release_invariant("idle state release"))? != 0,
            ),
            Rung::Prefix => released(service.prefix_cache.evict_one(self.held)),
            Rung::Optional => match service.domain.lowest_optional()? {
                Some((component, _)) => released(
                    service
                        .domain
                        .release_idle_optional_components(self.bindings, &[component])
                        .map_err(release_invariant("optional component release"))?
                        != 0,
                ),
                None => Release::Nothing,
            },
            Rung::Preemption => released(service.evict_victims(self.spared, self.resident_slot)?),
        })
    }
}

impl<F: ProgramFamily> Service<F> {
    /// Release once at the first rung that releases anything, preemption
    /// only where `preempt` ([`release::release_once`]): whether it
    /// released now; preemption spares `spared`, and with `resident_slot`
    /// takes requests that free no bytes, since what is short is a resident
    /// slot. A pending release (idle graph slots) releases nothing now: what
    /// waits on it retries once a boundary finishes it.
    fn release_once(
        &mut self,
        bindings: &mut StateBindings,
        preempt: bool,
        held: &BTreeSet<u64>,
        spared: &[RequestId],
        resident_slot: bool,
    ) -> Result<bool, DomainError> {
        let mut releasing = Releasing { service: self, bindings, held, spared, resident_slot, slots_began: false };
        Ok(release::release_once(&mut releasing, preempt)? == Release::Released)
    }
}

impl<F: ProgramFamily> MemoryHost for Service<F> {
    type Family = F;

    fn domain(&mut self) -> &mut ExecutorDomain<F> {
        &mut self.domain
    }

    fn memory(&mut self) -> &mut BetweenFlights {
        &mut self.memory
    }

    fn now(&self) -> u64 {
        self.now
    }

    fn completion(&self) -> magnitude_executor::CompletionWake {
        self.wakes.completion()
    }

    fn advance_epoch(&mut self) {
        self.epoch.advance();
    }

    fn unload(&mut self) {
        tracing::error!(requests = self.requests.len(), "unloading the model: the memory guarantee cannot be kept");
        self.terminalize_all(
            RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            },
            None,
        );
    }
}

enum Selected {
    Round(Round),
    /// Selection started no work but changed some request's state.
    Retry,
    Empty,
}

enum Residency {
    Resident(Generation, Vec<Operation>),
    Waiting(Generation, Wait),
}

#[cfg(test)]
mod branch_plan_tests {
    use super::{
        MIN_BRANCH_GAIN, MIN_PREFIX_HIT, awaited_prefix, peer_stop, plan_branch, plan_declared,
    };
    use std::collections::BTreeSet;

    /// Chat requests sharing a system prompt diverge a few rows into the
    /// last user message, fewer than the gain past the peer's declared cache
    /// point at the system prompt's end, so the divergence is never planned
    /// on the peer. The new request still waits for the peer's declared
    /// point instead of prefilling the whole prompt alongside it.
    #[test]
    fn a_request_waits_for_a_peer_stop_just_below_their_divergence() {
        let (declared, shared, prompt) = (14864, 14891, 14908);
        let mut peer = BTreeSet::from([declared, prompt - 6]);
        plan_branch(&mut peer, shared, 0, prompt);
        assert!(!peer.contains(&shared));
        assert_eq!(peer_stop(&peer, shared), Some(declared));
        assert_eq!(awaited_prefix(declared, shared, 0, prompt), Some(declared));
        // The divergence itself, when planned, is awaited as before.
        assert_eq!(awaited_prefix(shared, shared, 0, prompt), Some(shared));
        // Not a stop too far below the divergence, one not worth resuming
        // from (an entry already ends at or near it), or none at all.
        assert_eq!(awaited_prefix(shared - MIN_BRANCH_GAIN, shared, 0, prompt), None);
        assert_eq!(awaited_prefix(declared, shared, declared, prompt), None);
        assert_eq!(awaited_prefix(declared, shared, declared - 1, prompt), None);
        assert_eq!(awaited_prefix(0, shared, 0, prompt), None);
        assert_eq!(peer_stop(&BTreeSet::from([prompt - 6]), shared), None);
    }

    /// cc237 round 2: a conversation that shares the system prompt and the
    /// opening of the user message is cached, but its entry at the system
    /// prompt's end is gone, so nothing on the new requests' path ends below
    /// their divergence (they resume at 0). The cached path still shares
    /// rows up to just past the peer's stop. Those rows are no resume point,
    /// so the new request waits for the peer instead of prefilling the whole
    /// prompt beside it, as all eight requests did.
    #[test]
    fn a_cached_path_without_an_entry_in_the_prefix_never_replaces_the_peer() {
        let (declared, shared, prompt) = (14869, 14894, 14917);
        let cached_path = shared;
        let resumed = 0;
        let deepest = cached_path.max(shared);
        assert_eq!(awaited_prefix(declared, deepest, resumed, prompt), Some(declared));
    }

    /// A short turn of a conversation: it resumes where the previous turn
    /// declared the next one leaves the prompt, and declares the opening of
    /// its last message and where its own next turn leaves the prompt, all
    /// within the gain. Both are planned, so the next turn resumes beyond
    /// this one instead of where this one resumed.
    #[test]
    fn a_short_turn_plans_its_declared_points() {
        let (resumed, prompt) = (459, 490);
        let mut branches = BTreeSet::new();
        for point in [466, 483] {
            assert!(point < resumed + MIN_BRANCH_GAIN);
            plan_declared(&mut branches, point, resumed, prompt);
        }
        assert_eq!(branches, BTreeSet::from([466, 483]));
        // Never at or below the resume point, below a hit, or past the prompt.
        let mut branches = BTreeSet::new();
        for point in [resumed - 1, resumed, prompt] {
            plan_declared(&mut branches, point, resumed, prompt);
        }
        plan_declared(&mut branches, MIN_PREFIX_HIT - 1, 0, prompt);
        assert!(branches.is_empty());
    }

    #[test]
    fn a_branch_point_saves_enough_over_the_resume_point_and_lower_branches() {
        let prompt = 6000;
        let mut branches = BTreeSet::new();
        // Below a hit, at or past the prompt's end, or too near the resume
        // point: not worth a split chunk and a retained state.
        plan_branch(&mut branches, MIN_PREFIX_HIT - 1, 0, prompt);
        plan_branch(&mut branches, prompt, 0, prompt);
        plan_branch(&mut branches, 1000 + MIN_BRANCH_GAIN - 1, 1000, prompt);
        assert!(branches.is_empty());
        // A branch point, then a divergence just beyond it and one far
        // beyond it.
        plan_branch(&mut branches, 5800, 0, prompt);
        plan_branch(&mut branches, 5800 + MIN_BRANCH_GAIN - 1, 0, prompt);
        plan_branch(&mut branches, 5800 + MIN_BRANCH_GAIN, 0, prompt);
        // A shallower point is planned below the deeper ones.
        plan_branch(&mut branches, 3000, 0, prompt);
        assert_eq!(
            branches,
            BTreeSet::from([3000, 5800, 5800 + MIN_BRANCH_GAIN])
        );
    }
}

#[cfg(test)]
mod termination_tests {
    use super::Membership;
    use magnitude_generation::{
        EndOfGeneration, FinishReason, Generation, InputLayout, MethodChoice, Options, RoundStart,
        Sampling, Shaping, StartedRound, TokenId,
    };
    use std::collections::BTreeSet;

    pub(super) fn generation() -> Generation {
        generation_with_limit(8)
    }

    pub(super) fn generation_with_limit(max_tokens: usize) -> Generation {
        let mut generation = Generation::new(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, vec![]).unwrap(),
            Options {
                max_tokens,
                output_capacity: 4,
                context_limit: 32,
                vocabulary: 100,
                stop_tokens: BTreeSet::from([TokenId(99)]),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping {
                    temperature: 0.0,
                    ..Default::default()
                },
                seed: 42,
                forced_quantum: 4,
                method: MethodChoice::Plain,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
        )
        .unwrap();
        generation.resume_at(None).unwrap();
        generation
    }

    fn started() -> StartedRound {
        match generation().start_round(magnitude_generation::RequestId(1), 4) {
            Ok(RoundStart::Target(round)) => round,
            _ => panic!("expected a target round"),
        }
    }

    #[test]
    fn a_failed_started_round_finishes_failed_not_cancelled() {
        let failed = Membership::Started(started()).terminate(false);
        assert_eq!(failed.finish_reason(), Some(FinishReason::Failed));
        let cancelled = Membership::Started(started()).terminate(true);
        assert_eq!(cancelled.finish_reason(), Some(FinishReason::Cancelled));
    }

    /// An interrupted round (`RoundError::Interrupted`) is abandoned: its
    /// generation stays unfinished and starts the same round again.
    #[test]
    fn an_abandoned_round_runs_again() {
        let round = started();
        let forward = round.round_forward().clone();
        let generation = round.abandon();
        assert_eq!(generation.finish_reason(), None, "an abandoned round finishes its generation");
        match generation.start_round(magnitude_generation::RequestId(1), 4) {
            Ok(RoundStart::Target(again)) => assert_eq!(again.round_forward(), &forward, "the round starts again as it was"),
            _ => panic!("expected the target round again"),
        }
    }

    #[test]
    fn a_failed_waiting_request_finishes_failed() {
        let failed = Membership::Generation(generation()).terminate(false);
        assert_eq!(failed.finish_reason(), Some(FinishReason::Failed));
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;
    use magnitude_executor::{FeatureReader, FeatureRows, FeatureSpan};
    use magnitude_generation::{RoundStart, TokenId};
    use std::time::Duration;

    struct NoFeatures;
    impl FeatureReader for NoFeatures {
        fn read(&mut self, _: &FeatureSpan) -> Result<FeatureRows, String> {
            panic!("plain prompt needs no features")
        }
    }

    /// A lone prefill group queues its next chunk only when nothing outside
    /// the round competes for the device: another live request's work would
    /// orphan the queued chunk, discarding a whole chunk of device work.
    #[test]
    fn only_live_work_outside_the_round_competes_with_a_queued_chunk() {
        let record = || Record {
            waiting_since: 0, service_ns: 0, physical_timings: PhysicalTimings::default(),
            timing_phase: TimingPhase::Prompt, error: None, preemption_debt: 0,
            protected_until: None, prefix_cache: true, prefix_source: None,
            prefill_retained: false, terminal_retained: false, branches: BTreeSet::new(),
            publication: None, publication_batch_limit: 4, pending_publication: None,
            publication_permits: VecDeque::new(),
        };
        let request = |stage| Request {
            record: record(),
            state: State::Between(super::termination_tests::generation(), stage),
        };
        assert!(competes(&request(Stage::NonResident(None)), Status::Runnable));
        assert!(competes(&request(Stage::Ready), Status::Runnable));
        assert!(!competes(
            &request(Stage::AwaitingPrefix { through: 1 }),
            Status::Runnable
        ));
        assert!(!competes(
            &request(Stage::Ready),
            Status::Terminal(FinishReason::Length)
        ));
    }

    /// An admitted peer that has not yet become resident still computes the
    /// shared prefix in its first round, so a request admitted right after it
    /// waits for it instead of prefilling the prefix beside it. A peer the
    /// cache resumes at or beyond the prefix never computes it (a
    /// conversation's next turn resuming from its own deeper entry), so no
    /// request waits on it. A peer awaiting a prefix itself never counts, so
    /// two waiters cannot hold each other.
    #[test]
    fn an_admitted_peer_not_yet_resident_computes_its_prefix() {
        let generation = super::termination_tests::generation;
        let admitted = || State::Between(generation(), Stage::NonResident(None));
        assert!(computes_toward(&admitted(), 1, |_| 0));
        assert!(!computes_toward(&admitted(), 1, |_| 1));
        assert!(!computes_toward(&admitted(), 1, |_| 2));
        assert!(!computes_toward(
            &State::Between(generation(), Stage::AwaitingPrefix { through: 1 }),
            1,
            |_| 0
        ));
        let cancelled = terminated(generation(), true);
        assert!(!computes_toward(&State::Between(cancelled, Stage::NonResident(None)), 1, |_| 0));
    }

    #[test]
    fn published_prompt_token_does_not_reclassify_parked_or_pending_work() {
        let mut record = Record {
            waiting_since: 0, service_ns: 0, physical_timings: PhysicalTimings::default(),
            timing_phase: TimingPhase::Prompt, error: None, preemption_debt: 0,
            protected_until: None, prefix_cache: true, prefix_source: None,
            prefill_retained: false, terminal_retained: false, branches: BTreeSet::new(),
            publication: None, publication_batch_limit: 4, pending_publication: None,
            publication_permits: VecDeque::new(),
        };
        let candidate = Candidate::Ready(super::termination_tests::generation());
        record.begin_service(&candidate);
        let Candidate::Ready(generation) = candidate else { unreachable!() };
        let Ok(RoundStart::Target(round)) = generation.start_round(RequestId(1), 4) else {
            panic!("prompt round")
        };
        let work = lower_round(&round, RequestId(1), 0, None).unwrap();
        let candidate = Candidate::Parked(round, Work(vec![work.clone()]));
        record.begin_service(&candidate);
        assert_eq!(record.timing_phase, TimingPhase::Prompt);
        let Candidate::Parked(round, _) = candidate else { unreachable!() };
        let transition = round.prepare_round_transition(
            RequestId(1), &[TokenId(3)], None, &mut NoFeatures,
        ).unwrap();
        let (generation, _) = round.commit(transition);
        assert_eq!(generation.generated(), &[TokenId(3)]);
        // Pending operations retain the origin even though publication changed
        // generation state. The operation's lane/kind is irrelevant here.
        let candidate = Candidate::Pending(generation, Work(vec![work]));
        record.begin_service(&candidate);
        assert_eq!(record.timing_phase, TimingPhase::Prompt);
        let Candidate::Pending(mut generation, _) = candidate else { unreachable!() };
        generation.evicted().unwrap();
        let candidate = Candidate::NonResident(generation);
        record.begin_service(&candidate);
        assert_eq!(record.timing_phase, TimingPhase::Generation);

        let generation = super::termination_tests::generation_with_limit(1);
        let Ok(RoundStart::Target(round)) = generation.start_round(RequestId(1), 4) else {
            panic!("terminal prompt round")
        };
        let transition = round.prepare_round_transition(
            RequestId(1), &[TokenId(3)], None, &mut NoFeatures,
        ).unwrap();
        let (generation, _) = round.commit(transition);
        assert!(owes_reconciliation(&record, &generation));
        let candidate = Candidate::Ready(generation);
        record.begin_service(&candidate);
        assert_eq!(record.timing_phase, TimingPhase::Retention);
        let Candidate::Ready(generation) = candidate else { unreachable!() };
        let round = generation.start_reconciliation(4).unwrap_or_else(|(_, e)| panic!("{e}"));
        let work = lower_round(&round, RequestId(1), round.generation().resident_position(), None).unwrap();
        let candidate = Candidate::Parked(round, Work(vec![work]));
        record.begin_service(&candidate);
        assert_eq!(record.timing_phase, TimingPhase::Retention);
    }

    #[test]
    fn history_work_follows_its_origin_and_retention_is_excluded() {
        let mut timings = PhysicalTimings::default();
        // The final target prompt publishes a token before its head priming
        // completes. Both completions keep the prompt chain's phase.
        let prompt_chain = TimingPhase::Prompt;
        prompt_chain.add(&mut timings, Duration::from_millis(100));
        prompt_chain.add(&mut timings, Duration::from_millis(3));
        // Recovery can replay prompt rows after output has begun; it is still
        // generation service, including its deferred head preparation.
        let recovery_chain = TimingPhase::Generation;
        recovery_chain.add(&mut timings, Duration::from_millis(7));
        recovery_chain.add(&mut timings, Duration::from_millis(2));
        // Retaining finished output does not change either throughput field.
        TimingPhase::Retention.add(&mut timings, Duration::from_millis(20));
        TimingPhase::Retention.add(&mut timings, Duration::from_millis(1));
        assert_eq!(timings.prompt_ns, 103_000_000);
        assert_eq!(timings.predicted_ns, 9_000_000);
    }

    #[test]
    fn measured_duration_accumulation_saturates() {
        let mut timings = PhysicalTimings { prompt_ns: u64::MAX - 1, predicted_ns: 0 };
        TimingPhase::Prompt.add(&mut timings, Duration::from_nanos(2));
        assert_eq!(timings.prompt_ns, u64::MAX);
        assert_eq!(timings.predicted_ns, 0);
    }
}

#[cfg(test)]
mod pending_retention_tests {
    use super::*;
    use magnitude_executor::{DraftForm, FeatureReader, FeatureRows, FeatureSpan, HeadPhase};

    struct NoFeatures;
    impl FeatureReader for NoFeatures {
        fn read(&mut self, _: &FeatureSpan) -> Result<FeatureRows, String> {
            panic!("plain logical fixture needs no features")
        }
    }

    fn priming() -> Work {
        let operation = Operation::Head {
            request: RequestId(1),
            phase: HeadPhase::Priming { draft_from: 3 },
            tokens: vec![magnitude_generation::TokenId(3)],
            conditioning: FeatureRows::new(vec![0u8; 4].into(), 1).unwrap(),
            position: 2,
            proposals: Vec::new(),
            form: DraftForm::Chained,
        };
        operation.validate().unwrap();
        Work::new(vec![operation]).unwrap()
    }

    #[test]
    fn terminal_replay_with_parked_priming_cannot_retain_or_retire() {
        let generation = super::termination_tests::generation_with_limit(1);
        let Ok(RoundStart::Target(round)) = generation.start_round(RequestId(1), 4) else {
            panic!("prompt round")
        };
        let transition = round.prepare_round_transition(
            RequestId(1), &[magnitude_generation::TokenId(3)], None, &mut NoFeatures,
        ).unwrap();
        let (mut generation, _) = round.commit(transition);
        generation.discard_output();
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        let round = generation.start_reconciliation(4).unwrap_or_else(|(_, e)| panic!("{e}"));
        let work = lower_round(&round, RequestId(1), round.generation().resident_position(), None).unwrap();
        let parked = State::Parked(round, Work::new(vec![work]).unwrap(), Some(Wait::Memory { at: AvailabilityEpoch::default() }));
        assert!(parked.settled().is_none());
        let State::Parked(round, _, _) = parked else { unreachable!() };
        let transition = round.prepare_round_transition(RequestId(1), &[], None, &mut NoFeatures).unwrap();
        let (generation, _) = round.commit(transition);
        // These are exactly the old publication predicates: all are satisfied
        // before the priming continuation has become eligible to run.
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        assert_eq!(generation.output_len(), 0);
        assert!(generation.pending_reconciliation().is_none());
        assert_eq!(generation.resident_position(), generation.accepted_position());
        let pending = State::Between(generation, Stage::Pending(priming(), Some(Wait::Memory { at: AvailabilityEpoch::default() })));
        assert!(pending.settled().is_none(), "pending priming must block retention and terminal publication");
        let State::Between(generation, _) = pending else { unreachable!() };
        // finish_round returns completed method work as Ready.
        let completed = State::Between(generation, Stage::Ready);
        assert_eq!(completed.settled().unwrap().finish_reason(), Some(FinishReason::Length));
    }

    #[test]
    fn abandoned_pending_work_can_still_finish_cancelled_or_failed() {
        for (cancel, expected) in [(true, FinishReason::Cancelled), (false, FinishReason::Failed)] {
            let pending = State::Between(
                super::termination_tests::generation(),
                Stage::Pending(priming(), Some(Wait::Memory { at: AvailabilityEpoch::default() })),
            );
            assert!(pending.settled().is_none());
            // terminate_request / fail_capacity_waiters discard composed work
            // and normalize the terminated generation to Ready.
            let terminal = State::Between(pending.terminate(cancel), Stage::Ready);
            assert_eq!(terminal.settled().unwrap().finish_reason(), Some(expected));
        }
    }

    #[test]
    fn finished_without_residency_does_not_require_model_work_to_publish() {
        let mut generation = super::termination_tests::generation_with_limit(0);
        generation.evicted().unwrap();
        assert!(!generation.is_resident());
        assert_eq!(generation.finish_reason(), Some(FinishReason::Length));
        let state = State::Between(generation, Stage::NonResident(Some(Wait::Memory { at: AvailabilityEpoch::default() })));
        assert!(state.settled().is_some());
        let State::Between(generation, _) = state else { unreachable!() };
        let state = State::Between(generation, Stage::AwaitingPrefix { through: 1 });
        assert!(state.settled().is_some());
    }
}
