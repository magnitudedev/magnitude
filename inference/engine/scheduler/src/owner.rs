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
        PrefixRetention, PrefixStep,
    },
    protocol::{RequestSnapshot, RoomGrant, RoomGranter, RoomRefusal, WorkerReply},
    publication::{
        CapacityResource, ModelUnloadCause, PhysicalTimings, PublicationPermit, PublicationQueue,
        PublicationReceiver, PublicationSender, PublicationWake, PublicationWakeKind, PublishError,
        RequestError, ServiceCapacityError,
    },
    release::{self, Pending, Release, Releases, Rung},
    round_driver::{RoundError, lower_round, reconcile_forward},
    serving::ServingCensus,
    worker::Wakes,
    workload::WorkloadRates,
};
use magnitude_executor::{
    BankResize, BankResized, BankStep, BankTier, DomainError, OwnerHoldings, LoadMeasures, InvariantError, MeasurementRoom,
    MemoryChargeReconciliation, MemoryMoment, NativeFamily,
    OpenRequirements, Operation, Outcome, PendingOperationOutcome, PhysicalDecision,
    ProgramFamily, RequestId, ResourceKind, ResourcePlan, ShrinkCause, SlotRelease, StateBindings, SubmitError,
    SubmitFailure, WorkKind, platform::DomainRole,
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
    /// Blind or Reclaim.
    Memory,
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
        error @ DomainError::Reclaim(_) => RequestError::Invariant(InvariantError {
            context: "memory reclaim",
            detail: error.to_string(),
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
    /// It never ran, and waits for the release ladder to make room in
    /// memory; it is refused (`RequestError::MemoryReclaim`) once only the
    /// model's unload could.
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

/// How long Reclaim may persist after releases are exhausted, and how long
/// observations may fail continuously, before the model unloads.
const MEMORY_ESCALATION_NS: u64 = 1_000_000_000;

/// How often the owner observes memory without another reason to run, in
/// nanoseconds: the interval a load's reserves hold the host's motion over
/// ([`magnitude_executor::platform::READING_INTERVAL`]).
const OBSERVATION_INTERVAL_NS: u64 = magnitude_executor::platform::READING_INTERVAL.as_nanos() as u64;

/// One probe of every domain the loaded model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryObservation {
    /// Every domain's headroom is above its planning reserve.
    Normal,
    /// Some domain's headroom is at or below its planning reserve.
    Reclaim,
    /// A required observation failed.
    Blind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryCondition {
    Normal,
    /// Admission and growth pause while releases run. `since` starts the
    /// unload clock: when the band was entered, restarted whenever a release
    /// frees memory, so the model unloads once Reclaim persists for
    /// [`MEMORY_ESCALATION_NS`] after releases are exhausted.
    Reclaim {
        since: u64,
    },
    /// Observations have failed continuously since `since`.
    Blind {
        since: u64,
    },
}

impl MemoryCondition {
    fn observe(self, observation: MemoryObservation, now: u64) -> Self {
        match observation {
            MemoryObservation::Normal => Self::Normal,
            MemoryObservation::Reclaim => match self {
                Self::Reclaim { since } => Self::Reclaim { since },
                _ => Self::Reclaim { since: now },
            },
            MemoryObservation::Blind => match self {
                // One continuous escalation interval of Blind is Reclaim
                // whose unload clock has already expired.
                Self::Blind { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS => {
                    Self::Reclaim { since }
                }
                Self::Blind { since } => Self::Blind { since },
                Self::Reclaim { since } => Self::Reclaim { since },
                Self::Normal => Self::Blind { since: now },
            },
        }
    }

    /// A release freed memory at `now`: releases are not exhausted yet.
    fn released(self, now: u64) -> Self {
        match self {
            Self::Reclaim { .. } => Self::Reclaim { since: now },
            other => other,
        }
    }

    fn should_unload(self, now: u64) -> bool {
        matches!(self, Self::Reclaim { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS)
    }
}

/// What of the release ladder is still in motion between flight
/// boundaries. Only once nothing is are requests waiting on memory refused,
/// or the model unloaded, for memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ladder {
    /// Reclaim was observed while a flight held the bindings; its releases
    /// run once they return.
    owed: bool,
    /// The releases running that a later flight boundary finishes: the
    /// ladder resumes once none is.
    pending: Pendings,
    /// A release waits for a gap between flights or for the records in
    /// flight to land (a bank shrink deferred, `BankStep::Deferred`): the
    /// next boundary tries it again.
    deferred: bool,
    /// The last release found what is short released already, the host yet
    /// to show it (`Release::Covered`): the next reading decides again.
    covered: bool,
}

/// Which releases the ladder waits for a later flight boundary to finish,
/// each kind's apart ([`Pending`]): each finish clears its own alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pendings {
    slots: bool,
    bank: bool,
}

impl Pendings {
    fn of(&mut self, pending: Pending) -> &mut bool {
        match pending {
            Pending::Slots => &mut self.slots,
            Pending::Bank => &mut self.bank,
        }
    }

    fn any(self) -> bool {
        self.slots || self.bank
    }
}

impl Ladder {
    /// Take what one release at a rung came to.
    fn released(&mut self, release: Release) {
        if let Release::Pending(pending) = release {
            *self.pending.of(pending) = true;
        }
        self.deferred |= release == Release::Deferred;
        self.covered |= release == Release::Covered;
    }

    /// Whether nothing of the ladder is in motion.
    fn settled(&self) -> bool {
        !self.owed && !self.pending.any() && !self.deferred && !self.covered
    }

    /// A walk down the rungs begins: what it finds covered is its own.
    fn walk(&mut self) {
        self.covered = false;
    }

    /// Whether the model unloads: Reclaim in `condition` outlasted the clock
    /// at `now` with nothing of the ladder in motion.
    fn unload_due(&self, condition: MemoryCondition, now: u64) -> bool {
        self.settled() && condition.should_unload(now)
    }

    /// At a flight boundary: whether a release was deferred to it, which is
    /// tried again now.
    fn retry_deferred(&mut self) -> bool {
        std::mem::take(&mut self.deferred)
    }
}

/// Whether the measurement room's claim is owed a decision: memory was
/// probed with no resize of the expert bank under way, and none has
/// finished since. A probe while a flight holds the bindings stays owed
/// until they return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RoomOwed(bool);

impl RoomOwed {
    /// Memory was probed, a resize of the bank `resizing` or not.
    fn probed(&mut self, resizing: bool) {
        self.0 = !resizing;
    }

    /// A resize finished: the probe before it no longer reads the room.
    fn resized(&mut self) {
        self.0 = false;
    }

    /// Whether the decision is taken now, where the claim can be decided
    /// (`ready`): it is owed no longer.
    fn take(&mut self, ready: bool) -> bool {
        ready && std::mem::take(&mut self.0)
    }
}

/// The measurement job's claim of room beside the model, until it is
/// answered (`Owner::claim_measurement_room`).
struct RoomClaim(RoomGranter);

impl RoomClaim {
    /// Decide the claim at a probe no resize ran across, where it stands at
    /// `standing`: made, it is answered; short beyond what the bank holds
    /// above its floor, it is answered with why before anything shrinks;
    /// short within it, the bank shrinks by the rest through `shrink`, and
    /// the claim waits for a later probe to read what that freed, unless the
    /// shrink finds the bank at its floor, which answers it with why. The
    /// claim, while it waits.
    fn decide<E>(
        self,
        standing: MeasurementRoom,
        shrink: impl FnOnce(u64) -> Result<BankStep, E>,
    ) -> Result<Option<Self>, E> {
        let answer = match standing {
            MeasurementRoom::Made => Ok(()),
            // Unread, or made with the bank's own traffic still running:
            // the claim waits for the next observation.
            MeasurementRoom::Unread | MeasurementRoom::Draining => return Ok(Some(self)),
            MeasurementRoom::Beyond { short, releasable } => Err(RoomRefusal::BeyondFloor { short, releasable }),
            MeasurementRoom::Short(short) => match shrink(short)? {
                BankStep::Begun(_) | BankStep::Running | BankStep::Deferred => return Ok(Some(self)),
                BankStep::AtLimit => Err(RoomRefusal::AtFloor { short }),
            },
        };
        self.0.answer(answer);
        Ok(None)
    }
}

/// A running expert bank resize: what the process held when it began, and
/// why it began.
#[derive(Clone, Copy, Debug)]
struct BankBegun {
    held: MemoryMoment,
    cause: BankBegunFor,
}

/// Why an expert bank resize began: a shrink, for its cause, or a growth
/// into free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankBegunFor {
    Shrink(ShrinkCause),
    Growth,
}

impl BankBegunFor {
    /// The resize from `from` to `to` that finished, at `at_unix_ms`, as the
    /// census states it: the shrink or growth it began as, by at least one.
    fn finished(self, from: u64, to: u64, at_unix_ms: u64) -> Result<BankResize, String> {
        let by = |less: u64, more: u64| more.checked_sub(less).and_then(NonZeroU64::new);
        match self {
            Self::Shrink(cause) => by(to, from).map(|by| BankResize::Shrank { cause, to, by, at_unix_ms }),
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
    role: DomainRole,
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
/// ([`between_flights`]): its condition, the release ladder's motion, the
/// expert bank's running and finished resizes, and when it observes next.
pub struct BetweenFlights {
    condition: MemoryCondition,
    /// Per domain role, the least memory a capacity-blocked request still
    /// needs from that domain's ceiling after every release was exhausted.
    deficits: Vec<MemoryDeficit>,
    /// What of the release ladder is still in motion.
    ladder: Ladder,
    /// Memory was observed while a flight held the bindings: the heap is
    /// settled once they return.
    settle_owed: bool,
    /// Whether the measurement room's claim is owed a decision.
    room_owed: RoomOwed,
    /// Each store's running expert bank resize this owner began: the stores
    /// resize on their own (`BankStep::Begun`).
    bank_began: [Option<BankBegun>; BankTier::COUNT],
    /// The measurement job's room claimed of the domain and not yet
    /// answered ([`Owner::claim_measurement_room`]).
    measurement_room: Option<RoomClaim>,
    /// Costs the executor stored since the load was priced, which price it
    /// from the next flight boundary on ([`Owner::measure_load`]).
    measures: Option<LoadMeasures>,
    next_observation: u64,
}

impl BetweenFlights {
    /// Whether an expert bank resize this owner began runs in any store.
    fn resizing(&self) -> bool {
        self.bank_began.iter().any(Option::is_some)
    }

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

    /// A load's memory as it opens: Normal, nothing in motion, observed at
    /// its first boundary.
    pub fn new() -> Self {
        Self {
            condition: MemoryCondition::Normal,
            deficits: Vec::new(),
            ladder: Ladder::default(),
            settle_owed: false,
            room_owed: RoomOwed::default(),
            bank_began: [None; BankTier::COUNT],
            measurement_room: None,
            measures: None,
            next_observation: 0,
        }
    }
}

/// What [`between_flights`] runs on: the loaded model's domain and the
/// owner's memory, the service clock and its completion wakes, and what it
/// does to the requests its owner holds (none, below the scheduler): the
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
    /// Evict the least recently used retained prefix: whether one was.
    /// The cached entries live requests hold, which the prefix rung
    /// releases last.
    fn held_prefixes(&self) -> BTreeSet<u64>;
    fn evict_prefix(&mut self, held: &BTreeSet<u64>) -> bool;
    /// The cached prefixes the heap may release, with their bytes and what
    /// their hits save a second at `rates`, the least valuable first
    /// (`PrefixCache::held_steps`).
    fn prefix_steps(&self, rates: &WorkloadRates) -> Vec<PrefixStep>;
    /// Evict the cached prefix `id` a settle released: whether it was still
    /// cached and unused.
    fn evict_cached_prefix(&mut self, id: u64) -> bool;
    /// Preempt eligible live requests, sparing `spared`, and with
    /// `resident_slot` those that free no bytes: whether any was.
    fn evict_victims(&mut self, spared: &[RequestId], resident_slot: bool) -> Result<bool, DomainError>;
    /// Refuse every request waiting on memory to become resident.
    fn refuse_memory_waiters(&mut self);
    /// Unload the model for memory pressure: every request is terminated.
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

    /// Claim `bytes` of device memory beside the model for the service's
    /// measurement job, in place of any claim before, which is released
    /// (`ExecutorDomain::claim_measurement_room`): at flight boundaries the
    /// bank shrinks toward its floor until the room is made, and the grant
    /// is answered then, or once the floor leaves it short.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> RoomGrant {
        self.service.domain.claim_measurement_room(bytes);
        let (granter, grant) = RoomGrant::new();
        self.service.memory.measurement_room = Some(RoomClaim(granter));
        grant
    }

    /// The measurement job ended: its claim, answered or not, is released.
    pub fn release_measurement_room(&mut self) {
        self.service.domain.release_measurement_room();
        self.service.memory.measurement_room = None;
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
        // A request arriving in Reclaim waits while the ladder gives memory
        // back: it is refused only once the ladder's next release would be
        // the model's unload (`MemoryStep::release_for_reclaim`).
        match self.domain.probe_memory() {
            Err(error @ DomainError::Reclaim(_)) => self.refused_for_memory(&error),
            observed => {
                observed.map_err(AdmissionError::from)?;
            }
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
            Some(Wait::Memory) if self.waits_on_memory(request) => return Status::MemoryBlocked,
            Some(Wait::Memory) => return Status::Preempted,
            None if self.waits_on_memory(request) => return Status::MemoryBlocked,
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
                        unload |= match self.observe(None) {
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
                        .observe(Some(&mut bindings))
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
    /// Observe memory when due. Reclaim releases need the bindings; while a
    /// flight holds them they are owed until it completes. Returns whether
    /// memory escalation decided to unload, after terminating every request.
    /// This is the owner's own work, which serves no request: no launch is
    /// charged the time it takes (`ExecutorDomain::owner_worked`).
    fn observe(&mut self, bindings: Option<&mut StateBindings>) -> Result<bool, RequestError> {
        let began = std::time::Instant::now();
        let observed = self.observe_memory(bindings);
        self.domain().owner_worked(began.elapsed());
        observed
    }

    fn observe_memory(&mut self, bindings: Option<&mut StateBindings>) -> Result<bool, RequestError> {
        let memory = |error: String| invariant("memory observation")(error);
        let now = self.now();
        // A reading is taken off this thread and published
        // (`ExecutorDomain::take_published_memory`): one is looked for once
        // an interval has passed, and none waited for, so no reading sits
        // between a step and the next however long either takes.
        let published = match now >= self.memory().next_observation {
            true => {
                self.memory().next_observation = now.saturating_add(OBSERVATION_INTERVAL_NS);
                self.domain().take_published_memory()
            }
            false => None,
        };
        if let Some(published) = published {
            let resizing = self.memory().resizing();
            self.memory().room_owed.probed(resizing);
            let readings = published.as_ref().ok().cloned();
            let observation = memory_observation(published.map(drop)).map_err(memory)?;
            let next = self.memory().condition.observe(observation, now);
            self.enter_memory_condition(next);
            // Each reading may settle the heap afresh: its supply holds the
            // reserve derived from the host's measured motion, so what it
            // grants is room no wait would make safer. The settle runs only
            // where the reading moved its inputs (`ExecutorDomain::settle_memory`).
            self.memory().settle_owed = observation == MemoryObservation::Normal;
            if !matches!(self.memory().condition, MemoryCondition::Reclaim { .. }) {
                self.memory().ladder.owed = false;
                self.memory().ladder.covered = false;
                if let (MemoryObservation::Normal, Some(readings)) = (observation, readings) {
                    self.reopen_memory_deficits(&readings);
                }
            } else if observation == MemoryObservation::Reclaim {
                // Releases are measured against a fresh reading; a Blind
                // escalation has none, so it goes straight to the unload
                // decision.
                self.memory().ladder.owed = true;
            }
        }
        // A cut of the expert bank a chain's encoding began beside its
        // running steps, memory found short between its submissions
        // (`ExecutorDomain::take_running_release`), is a Reclaim shrink this
        // owner holds as begun: the advance that finishes it reports it as
        // any.
        if let Some((tier, held)) = self.domain().take_running_release() {
            let begun = BankBegun { held, cause: BankBegunFor::Shrink(ShrinkCause::Reclaim) };
            self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))?;
        }
        let mut bindings = bindings;
        // A flight boundary: newly stored costs price the load first, so
        // every decision past it is made on them; the expert bank's resize
        // advances, and a finished one frees what waited on it.
        if let Some(bindings) = bindings.as_deref_mut() {
            // A refusal (the device would not take the tier copy's grid) is a device
            // fault the load ends on, the experts still priced as before.
            if let Some(measures) = self.memory().measures.take() {
                self.domain().measure_load(measures, bindings).map_err(classify_domain_error)?;
            }
            self.advance_expert_bank()?;
            self.advance_slot_release()?;
            // A release deferred for want of a gap is tried again at this
            // boundary: what waited on it retries, and a Reclaim release is
            // owed again.
            if self.memory().ladder.retry_deferred() {
                self.advance_epoch();
                let reclaim = matches!(self.memory().condition, MemoryCondition::Reclaim { .. });
                self.memory().ladder.owed |= reclaim;
            }
        }
        // The measurement job's claim takes its room through the bank's
        // shrink toward its floor, read at a probe no resize ran across
        // (`room_owed`), so each shrink is seen before the next is decided:
        // answered once the room is made, or once the floor leaves it short.
        // A shrink deferred while a chain is under way waits for the next
        // probe.
        let ready = !self.memory().resizing() && bindings.is_some();
        let deciding = match self.memory().room_owed.take(ready) {
            true => self.domain().measurement_room().map_err(classify_domain_error)?,
            false => None,
        };
        if let (Some(standing), Some(bindings), Some(claim)) =
            (deciding, bindings.as_deref_mut(), deciding.and_then(|_| self.memory().measurement_room.take()))
        {
            let mut began = None;
            let wake = self.completion();
            let domain = self.domain();
            let waiting = claim.decide(standing, |bytes| {
                let before = domain.memory_moment();
                let step = domain.shrink_expert_bank_by(bindings, bytes, wake);
                let step = step.map_err(classify_domain_error)?;
                if let BankStep::Begun(tier) = step {
                    began = Some((tier, before));
                }
                Ok(step)
            })?;
            self.memory().measurement_room = waiting;
            if let Some((tier, held)) = began {
                let begun = BankBegun { held, cause: BankBegunFor::Shrink(ShrinkCause::Measurement) };
                self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))?;
            }
        }
        // Every release changes bindings, so it waits for a chain under way
        // to end: the chain's device work is done when its first chunk
        // finishes, and its holder claims the rest first (`chain_holder`).
        let releasable = self.memory().ladder.owed && self.domain().chain_holder().is_none();
        if let (true, Some(bindings)) = (releasable, bindings.as_deref_mut()) {
            self.memory().ladder.owed = false;
            if self.release_for_reclaim(bindings)? {
                let now = self.now();
                self.memory().condition = self.memory().condition.released(now);
                let after = memory_observation(self.domain().probe_memory()).map_err(memory)?;
                let next = self.memory().condition.observe(after, now);
                self.enter_memory_condition(next);
            }
        }
        if let (true, Some(bindings)) = (self.memory().settle_owed, bindings) {
            self.memory().settle_owed = false;
            self.settle_memory(bindings)?;
        }
        // Unloading is the last step of the release order, past the ladder's
        // last rung (`release::Rung`): while a flight holds the bindings, the
        // releases owed to it have not run yet, while one is pending it has
        // not finished, and one deferred for a gap is tried again at the next
        // boundary, so the decision waits for them. What the host has yet to
        // show released is awaited, never released again: only a reading,
        // the witness's fall or more taken than is credited, moves the
        // ladder past it (`MemoryPace`).
        let now = self.now();
        let condition = self.memory().condition;
        if self.memory().ladder.unload_due(condition, now) {
            self.unload();
            return Ok(true);
        }
        Ok(false)
    }

    /// A group refused because memory is in the Reclaim band, or could not
    /// be observed, is a memory observation: enter its condition now. A
    /// memory wait ends only when memory is Normal again, so without this the
    /// refused request would be selected again at once, within the same run,
    /// before the next timed observation could see the band.
    fn refused_for_memory(&mut self, error: &DomainError) {
        let observation = match error {
            DomainError::Reclaim(_) => MemoryObservation::Reclaim,
            DomainError::Blind(_) => MemoryObservation::Blind,
            _ => return,
        };
        let next = self.memory().condition.observe(observation, self.now());
        self.enter_memory_condition(next);
        if matches!(self.memory().condition, MemoryCondition::Reclaim { .. }) {
            self.memory().ladder.owed = true;
        }
    }

    /// Enter `next`. Memory returning to Normal is an availability change,
    /// and it ends every memory wait.
    fn enter_memory_condition(&mut self, next: MemoryCondition) {
        let memory = self.memory();
        let recovered = memory.condition != MemoryCondition::Normal && next == MemoryCondition::Normal;
        if std::mem::discriminant(&memory.condition) != std::mem::discriminant(&next) {
            tracing::warn!(from = ?memory.condition, to = ?next, "memory condition changed");
        }
        memory.condition = next;
        if recovered {
            self.advance_epoch();
        }
    }

    /// A request blocked on a memory deficit waits for an availability
    /// change. Memory another process frees is one: advance the epoch once a
    /// deficit's domain ceiling covers it, so blocked work retries.
    fn reopen_memory_deficits(&mut self, readings: &[magnitude_executor::platform::DomainReading]) {
        if self.memory().deficits.is_empty() {
            return;
        }
        let deficits = &mut self.memory().deficits;
        let before = deficits.len();
        deficits.retain(|deficit| {
            !readings.iter().any(|reading| {
                reading.role == deficit.role && reading.ceiling_bytes >= deficit.required
            })
        });
        if deficits.len() != before {
            self.advance_epoch();
        }
    }

    /// Settle the memory heap (`ExecutorDomain::settle_memory`) on
    /// `bindings` and apply it: the prefixes it released evicted, the idle
    /// optional components it released unbound, then the prompt-group
    /// places held beyond the target it settled, else one bank resize: its
    /// least valuable release first (what it gives back is what the
    /// settle's growths and places are paid with), else its most valuable
    /// growth, each to its settled size; the rest settle again at the next
    /// reading. While memory is Normal and no blocked request waits on
    /// memory (what is free is theirs first), and no resize runs (one begun
    /// decides the next). A grant's claim refused in Reclaim or blind is
    /// the memory observation it is.
    fn settle_memory(&mut self, bindings: &mut StateBindings) -> Result<(), RequestError> {
        if self.memory().condition != MemoryCondition::Normal
            || !self.memory().deficits.is_empty()
            || self.memory().resizing()
        {
            return Ok(());
        }
        let rates = self.domain().serving_rates().map_err(classify_domain_error)?;
        let prefixes = rates.map_or_else(Vec::new, |rates| self.prefix_steps(&rates));
        let owner = OwnerHoldings { prefixes: prefixes.iter().map(|step| (step.bytes, step.seconds_per_second)).collect() };
        let settled = self.domain().settle_memory(&owner).map_err(classify_domain_error)?;
        // The settle releases the least valuable first, in the order given.
        for step in &prefixes[..settled.prefixes] {
            self.evict_cached_prefix(step.id);
        }
        if !settled.optional.is_empty() {
            self.domain()
                .release_idle_optional_components(bindings, &settled.optional)
                .map_err(invariant("optional component release"))?;
        }
        if settled.prefixes > 0 || !settled.optional.is_empty() {
            self.advance_epoch();
        }
        // Places the settle released are given back first, between groups:
        // their room is what its growths may be paid with.
        if settled.places_released {
            let wake = self.completion();
            let (released, slots) = self.domain().release_group_places(wake).map_err(classify_domain_error)?;
            tracing::info!(released, ?slots, "prompt places release settled");
            // A release deferred (a chain under way, a step of the bank
            // running) is settled again at the next reading, as a deferred
            // shrink is.
            if self.domain().group_places() > self.domain().places_target() {
                self.domain().settle_again();
            }
            return Ok(());
        }
        // What the process holds as a resize begins, read only for one.
        if let Some(release) = settled.releases.first().copied() {
            let magnitude_executor::BankRelease { holding, to, per_byte } = release;
            let (before, wake) = (self.domain().memory_moment(), self.completion());
            // A shrink deferred (a chain under way, a step still running)
            // is settled again at the next reading.
            let step = self.domain().shrink_expert_bank_to(bindings, holding, to, wake).map_err(classify_domain_error)?;
            tracing::info!(?holding, to, per_byte, ?step, "expert bank release settled");
            if matches!(step, BankStep::Deferred | BankStep::Running) {
                self.domain().settle_again();
            }
            // A holding worth more per byte won the bank's memory.
            if let BankStep::Begun(tier) = step {
                let begun = BankBegun { held: before, cause: BankBegunFor::Shrink(ShrinkCause::Rebalance) };
                self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))?;
            }
            return Ok(());
        }
        let Some(grant) = settled.grants.first().cloned() else {
            return Ok(());
        };
        let (before, wake) = (self.domain().memory_moment(), self.completion());
        let decided = |outcome: &dyn std::fmt::Display| {
            let magnitude_executor::BankGrant { holding, kind, from, to, bytes, borrowed, per_byte, .. } = &grant;
            tracing::info!(?holding, ?kind, from, to, bytes, borrowed, per_byte, %outcome, "expert bank growth settled");
        };
        match self.domain().grow_expert_bank(&grant, wake) {
            Ok(BankStep::Begun(tier)) => {
                decided(&"begun");
                let begun = BankBegun { held: before, cause: BankBegunFor::Growth };
                self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))?;
            }
            // A growth not begun is never final: the next reading settles
            // afresh, whatever it reads.
            Ok(step) => {
                decided(&format_args!("{step:?}"));
                self.domain().settle_again();
            }
            // Sized by the claim's own test, a refusal is the machine's
            // change since the settle's reading: not even the grant's first
            // step fits now.
            Err(DomainError::Capacity(refusal)) => {
                decided(&format_args!("refused: {} bytes required, {} available", refusal.required, refusal.available));
                self.domain().settle_again();
            }
            Err(error @ (DomainError::Reclaim(_) | DomainError::Blind(_))) => {
                decided(&format_args!("refused: {error}"));
                self.refused_for_memory(&error);
                self.domain().settle_again();
            }
            Err(error) => return Err(classify_domain_error(error)),
        }
        Ok(())
    }

    /// At a flight boundary, finish the idle graph slots' release off the
    /// submitting thread (`ExecutorDomain::advance_slot_release`), recorded
    /// with the slots and how long the release thread took. It freed
    /// memory, so the reclaim clock restarts and what waited on memory may
    /// retry. One the ladder waited for (its surplus rung began it, or found
    /// the idle policy's running) is the surplus rung's: the ladder resumes
    /// while memory is short. Any other is the idle policy's alone.
    fn advance_slot_release(&mut self) -> Result<(), RequestError> {
        let Some(released) = self.domain().advance_slot_release().map_err(classify_domain_error)? else {
            return Ok(());
        };
        let laddered = std::mem::take(&mut self.memory().ladder.pending.slots);
        let after = self.domain().memory_moment();
        tracing::info!(
            by = if laddered { "the surplus rung" } else { "the idle policy" },
            slots = released.slots,
            released = released.bytes,
            release_micros = released.released.duration_since(released.began).as_micros() as u64,
            charged_after = after.charged,
            claimed_after = after.claimed,
            footprint_after = after.footprint,
            "graph slots released"
        );
        let now = self.now();
        self.memory().condition = self.memory().condition.released(now);
        self.advance_epoch();
        if laddered && matches!(self.memory().condition, MemoryCondition::Reclaim { .. }) {
            self.memory().ladder.owed = true;
        }
        Ok(())
    }

    /// At a flight boundary, advance the expert bank's resize
    /// (`ExecutorDomain::advance_expert_bank`). A finished one is recorded as
    /// a release is (a shrink as the bank rung's, a growth as memory grown):
    /// how long it held this thread, the bank's slots after it, and what the
    /// process held when it began and now. A finished shrink freed memory:
    /// the reclaim clock restarts, what waited on memory may retry, and the
    /// ladder resumes while memory is short.
    fn advance_expert_bank(&mut self) -> Result<(), RequestError> {
        let wake = self.completion();
        let Some((finished, held)) = self.domain().advance_expert_bank(wake).map_err(classify_domain_error)? else {
            return Ok(());
        };
        self.memory().ladder.pending.bank = false;
        self.memory().room_owed.resized();
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
                    rung = ?Rung::Bank,
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
                let now = self.now();
                self.memory().condition = self.memory().condition.released(now);
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
        if matches!(self.memory().condition, MemoryCondition::Reclaim { .. }) {
            self.memory().ladder.owed = true;
        }
        Ok(())
    }

    /// Whether a fresh reading is still in the Reclaim band. A failed reading
    /// cannot measure a release, so it stops releasing.
    fn reclaim_needed(&mut self) -> Result<bool, RequestError> {
        Ok(memory_observation(self.domain().probe_memory())
            .map_err(invariant("memory observation"))?
            == MemoryObservation::Reclaim)
    }

    /// Release once at the cheapest rung that releases anything, preemption
    /// only where `preempt` ([`release::release_once`]): whether it released now;
    /// preemption spares `spared`, and with `resident_slot` takes requests
    /// that free no bytes, since what is short is a resident slot. A pending
    /// release (a bank resize) is recorded: no costlier rung runs, and the
    /// model is not unloaded, before the boundary that finishes it.
    fn release_once(
        &mut self,
        bindings: &mut StateBindings,
        cause: ReleaseCause,
        preempt: bool,
        held: &BTreeSet<u64>,
        spared: &[RequestId],
        resident_slot: bool,
    ) -> Result<bool, DomainError> {
        let release = release::release_once(
            &mut Releasing {
                host: self,
                bindings,
                cause,
                held,
                spared,
                resident_slot,
                began: None,
                slots_began: false,
            },
            preempt,
        )?;
        self.memory().ladder.released(release);
        Ok(release == Release::Released)
    }

    /// Release down the whole ladder until every domain is back above its
    /// planning reserve, or a release is pending. Preempted requests replay
    /// only after the band is Normal again, so no victim is chosen twice.
    /// Returns whether anything was released. With the ladder exhausted and
    /// memory still short, the next release would be the model's unload:
    /// every request waiting on memory to become resident is refused.
    fn release_for_reclaim(&mut self, bindings: &mut StateBindings) -> Result<bool, RequestError> {
        let mut released = false;
        self.memory().ladder.walk();
        let held = self.held_prefixes();
        while self.reclaim_needed()?
            && self
                .release_once(bindings, ReleaseCause::Reclaim, true, &held, &[], false)
                .map_err(classify_domain_error)?
        {
            released = true;
        }
        if released {
            self.advance_epoch();
        }
        if self.memory().ladder.settled() && self.reclaim_needed()? {
            self.refuse_memory_waiters();
        }
        Ok(released)
    }
}

impl<H: MemoryHost> MemoryStep for H {}

/// The owner's memory step at a flight boundary, `bindings` there where no
/// flight holds them: memory observed when due (and growth owed once it has
/// stayed free), the expert bank's running resize advanced, a prompt's
/// places and the measurement job's claim taken through the bank's shrink,
/// the release ladder run while memory is short, a growth step of the bank
/// begun where owed, and the model unloaded once Reclaim outlasts every
/// release. Whether it unloaded. The one step every host runs: the owner at
/// each of its boundaries, and an instrument that drives the domain below
/// the scheduler at each of its own.
pub fn between_flights<H: MemoryHost>(
    host: &mut H,
    bindings: Option<&mut StateBindings>,
) -> Result<bool, RequestError> {
    host.observe(bindings)
}

impl<F: ProgramFamily> Service<F> {
    /// Record that blocked work needs `required` bytes from `role`'s domain,
    /// keeping the least requirement per role so the earliest retry is kept.
    fn record_memory_deficit(&mut self, resource: ResourceKind, required: u64) {
        let role = match resource {
            ResourceKind::DeviceMemory => DomainRole::Allocation,
            ResourceKind::HostStaging => DomainRole::Staging,
            _ => return,
        };
        match self
            .memory
            .deficits
            .iter_mut()
            .find(|deficit| deficit.role == role)
        {
            Some(deficit) => deficit.required = deficit.required.min(required),
            None => self.memory.deficits.push(MemoryDeficit { role, required }),
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
            && matches!(&request.state, State::Between(_, Stage::NonResident(wait @ (None | Some(Wait::Memory))))
                if wait.is_some() || matches!(self.memory.condition, MemoryCondition::Reclaim { .. }))
    }

    /// Refuse every request waiting on memory to become resident: it never
    /// was, and only the model's unload could make room for it.
    fn refuse_memory_waiters(&mut self) {
        let waiting = self
            .requests
            .iter()
            .filter(|(_, request)| self.waits_on_memory(request))
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        for id in waiting {
            let Request { mut record, state } = self.requests.remove(&id).expect("waiting request");
            let mut generation = state.terminate(false);
            Self::fail_generation(&mut record, &mut generation, RequestError::MemoryReclaim);
            self.requests.insert(
                id,
                Request {
                    record,
                    state: State::Between(generation, Stage::Ready),
                },
            );
        }
    }

    /// When nothing can move the epoch (no flight: this runs in `Idle`, no
    /// request waits on its host or on memory, and memory is Normal), every
    /// capacity wait ends in its typed error. Returns whether any did.
    fn fail_capacity_waiters(&mut self) -> bool {
        if self.memory.condition != MemoryCondition::Normal {
            return false;
        }
        let independent = self.requests.values().any(|request| {
            request.state.generation().finish_reason().is_some()
                || matches!(request.state, State::Between(_, Stage::AwaitingCredit))
                || request.state.wait() == Some(Wait::Memory)
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
        let memory_normal = self.memory.condition == MemoryCondition::Normal;
        let epoch = self.epoch;
        let ended_wait = |wait: Option<Wait>| match wait {
            Some(Wait::Capacity { at, .. }) => epoch.changed_since(at),
            Some(Wait::Memory) => memory_normal,
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
                State::Between(_, Stage::NonResident(wait)) => memory_normal && ended_wait(*wait),
                State::Between(_, Stage::Pending(_, wait)) | State::Parked(_, _, wait) => {
                    ended_wait(*wait)
                }
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
            Err(DomainError::Reclaim(_) | DomainError::Blind(_)) => {
                return Ok(Residency::Waiting(generation, Wait::Memory));
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
            && self.release_once(bindings, ReleaseCause::LiveClaim, true, &held, &[], true)?
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
                        DomainError::Blind(_) | DomainError::Reclaim(_) => {
                            self.refused_for_memory(&error);
                            self.park(&mut round, group_, Wait::Memory);
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
        while short(&current)
            && self
                .release_once(bindings, ReleaseCause::LiveClaim, false, &held, &[], false)
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
                    .release_once(bindings, ReleaseCause::LiveClaim, true, &held, &selected, false)
                    .map_err(classify_domain_error)?
            {
                current = self.fits(bindings, round, &group_);
            }
        }
        match current {
            Ok(()) => Ok(Provisioned::Fits(group_)),
            Err(error @ (DomainError::Blind(_) | DomainError::Reclaim(_))) => {
                self.refused_for_memory(&error);
                Ok(Provisioned::Short(group_, Wait::Memory))
            }
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
        if wait == Wait::Memory {
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
        if let Err(error) = self.observe(Some(&mut bindings)) {
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
            return Ok(true);
        }
        Ok(false)
    }
}

/// The owner's releases, between flights: the bindings every one of them
/// changes are here.
struct Releasing<'a, H: MemoryHost> {
    host: &'a mut H,
    bindings: &'a mut StateBindings,
    cause: ReleaseCause,
    held: &'a BTreeSet<u64>,
    spared: &'a [RequestId],
    resident_slot: bool,
    /// The store a bank shrink this release began is of.
    began: Option<BankTier>,
    /// This release began releasing idle graph slots off the submitting
    /// thread: it is recorded as one that released now.
    slots_began: bool,
}

/// Why the owner releases memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseCause {
    /// Memory is in the Reclaim band.
    Reclaim,
    /// A live request's claim is short.
    LiveClaim,
}

impl ReleaseCause {
    /// The cause of an expert bank shrink this release begins.
    fn shrink(self) -> BankBegunFor {
        BankBegunFor::Shrink(match self {
            Self::Reclaim => ShrinkCause::Reclaim,
            Self::LiveClaim => ShrinkCause::LiveClaim,
        })
    }
}

impl<H: MemoryHost> Releases for Releasing<'_, H> {
    type Error = DomainError;

    /// Each release is recorded: its rung, how long it took, the bank's
    /// slots after it, and what the process held on each side of it (a bank
    /// step's when it finishes, `MemoryStep::advance_expert_bank`).
    fn release(&mut self, rung: Rung) -> Result<Release, DomainError> {
        let before = self.host.domain().memory_moment();
        // The bank decision's deficits, before this release can change
        // memory: what the release is attributed to.
        let mut deficit = |holding| {
            let domain = self.host.domain();
            domain
                .expert_bank_plan()
                .and_then(|bank| bank.holding_storage(holding))
                .map_or(0, |kind| domain.expert_reclaim_deficit(kind.storage()))
        };
        let device_deficit = deficit(magnitude_executor::BankHolding::Experts);
        let host_deficit = deficit(magnitude_executor::BankHolding::Host);
        let started = std::time::Instant::now();
        let released = self.release_at(rung)?;
        if let Some(tier) = self.began.take() {
            let begun = BankBegun { held: before, cause: self.cause.shrink() };
            self.host.memory().bank_begun(tier, begun).map_err(release_invariant("expert bank"))?;
        }
        if released == Release::Released || std::mem::take(&mut self.slots_began) {
            let domain = self.host.domain();
            let after = domain.memory_moment();
            // Unread values are absent fields.
            tracing::info!(
                rung = ?rung,
                cause = ?self.cause,
                reclaim_device_deficit_bytes = device_deficit,
                reclaim_host_deficit_bytes = host_deficit,
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

    /// Idle state and graph slots save nothing; the bank's, the cached
    /// prefixes' and the bound optional components' lowest held rate per
    /// byte at the load's serving rates (`ExecutorDomain::bank_release_rate`,
    /// `PrefixCache::held_steps`, `ExecutorDomain::lowest_optional`). An
    /// optional component whose worth is unknown holds nothing priced: the
    /// rung goes last of the priced ones.
    fn rate(&mut self, rung: Rung) -> Result<Option<f64>, DomainError> {
        match rung {
            Rung::Surplus | Rung::Idle => Ok(Some(0.)),
            Rung::Bank => self.host.domain().bank_release_rate(),
            Rung::Optional => Ok(self.host.domain().lowest_optional()?.and_then(|(_, per_byte)| per_byte)),
            Rung::Prefix => Ok(self.host.domain().serving_rates()?.and_then(|rates| {
                self.host
                    .prefix_steps(&rates)
                    .first()
                    .map(|step| step.seconds_per_second / step.bytes.max(1) as f64)
            })),
            Rung::Preemption => unreachable!("the ladder rates only its priced rungs"),
        }
    }
}

impl<H: MemoryHost> Releasing<'_, H> {
    fn release_at(&mut self, rung: Rung) -> Result<Release, DomainError> {
        let released = |released: bool| match released {
            true => Release::Released,
            false => Release::Nothing,
        };
        Ok(match rung {
            // Idle graph slots release off this thread: a later boundary
            // finishes them, and nothing costlier runs before it.
            Rung::Surplus => {
                let wake = self.host.completion();
                let shrunk = self.host.domain().shrink_state(self.bindings, ShrinkPolicy::Reclaim, wake)?;
                self.slots_began = shrunk.slots == SlotRelease::Began;
                match shrunk.slots {
                    SlotRelease::Began | SlotRelease::Running => Release::Pending(Pending::Slots),
                    SlotRelease::None => released(shrunk.released != 0),
                }
            }
            Rung::Idle => released(
                self.host
                    .domain()
                    .reclaim_idle(self.bindings)
                    .map_err(release_invariant("idle state release"))?
                    != 0,
            ),
            Rung::Bank if self.host.domain().expert_reclaim_covered() => Release::Covered,
            // A step begun, or a resize running, releases at a later flight
            // boundary; the bank's floor releases nothing.
            Rung::Bank => {
                let wake = self.host.completion();
                match self.host.domain().shrink_expert_bank(self.bindings, wake)? {
                    BankStep::Begun(tier) => {
                        self.began = Some(tier);
                        Release::Pending(Pending::Bank)
                    }
                    BankStep::Running => Release::Pending(Pending::Bank),
                    BankStep::Deferred => Release::Deferred,
                    BankStep::AtLimit => Release::Nothing,
                }
            }
            Rung::Prefix => released(self.host.evict_prefix(self.held)),
            Rung::Optional => match self.host.domain().lowest_optional()? {
                Some((component, _)) => released(
                    self.host
                        .domain()
                        .release_idle_optional_components(self.bindings, &[component])
                        .map_err(release_invariant("optional component release"))?
                        != 0,
                ),
                None => Release::Nothing,
            },
            Rung::Preemption => released(self.host.evict_victims(self.spared, self.resident_slot)?),
        })
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

    fn held_prefixes(&self) -> BTreeSet<u64> {
        self.held_entries(None)
    }

    fn evict_prefix(&mut self, held: &BTreeSet<u64>) -> bool {
        self.prefix_cache.evict_one(held)
    }

    fn prefix_steps(&self, rates: &WorkloadRates) -> Vec<PrefixStep> {
        self.prefix_cache.held_steps(self.domain.retained_entry_bytes(), rates, &self.held_entries(None))
    }

    fn evict_cached_prefix(&mut self, id: u64) -> bool {
        self.prefix_cache.evict(id)
    }

    fn evict_victims(&mut self, spared: &[RequestId], resident_slot: bool) -> Result<bool, DomainError> {
        Service::evict_victims(self, spared, resident_slot)
    }

    fn refuse_memory_waiters(&mut self) {
        Service::refuse_memory_waiters(self)
    }

    fn unload(&mut self) {
        tracing::error!(requests = self.requests.len(), "unloading the model: memory stayed below the reclaim reserve");
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

fn memory_observation(result: Result<(), DomainError>) -> Result<MemoryObservation, String> {
    match result {
        Ok(()) => Ok(MemoryObservation::Normal),
        Err(DomainError::Reclaim(_)) => Ok(MemoryObservation::Reclaim),
        Err(DomainError::Blind(_)) => Ok(MemoryObservation::Blind),
        Err(error) => Err(error.to_string()),
    }
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
mod memory_condition_tests {
    use super::{MEMORY_ESCALATION_NS, MemoryCondition, MemoryObservation};

    #[test]
    fn continuous_blind_escalates_to_immediately_eligible_reclaim() {
        let blind = MemoryCondition::Normal.observe(MemoryObservation::Blind, 7);
        assert_eq!(blind, MemoryCondition::Blind { since: 7 });
        let before = blind.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS - 1);
        assert_eq!(before, blind);
        assert!(!before.should_unload(7 + MEMORY_ESCALATION_NS - 1));
        let escalated = before.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS);
        assert_eq!(escalated, MemoryCondition::Reclaim { since: 7 });
        assert!(escalated.should_unload(7 + MEMORY_ESCALATION_NS));
    }

    #[test]
    fn interrupted_blind_restarts_its_interval() {
        let blind = MemoryCondition::Normal.observe(MemoryObservation::Blind, 7);
        let normal = blind.observe(MemoryObservation::Normal, 8);
        assert_eq!(normal, MemoryCondition::Normal);
        let renewed = normal.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS);
        assert_eq!(
            renewed,
            MemoryCondition::Blind {
                since: 7 + MEMORY_ESCALATION_NS
            }
        );
        assert!(!renewed.should_unload(7 + MEMORY_ESCALATION_NS));
    }

    #[test]
    fn reclaim_unloads_one_interval_after_releases_are_exhausted() {
        let reclaim = MemoryCondition::Normal.observe(MemoryObservation::Reclaim, 5);
        assert_eq!(reclaim, MemoryCondition::Reclaim { since: 5 });
        // Continued Reclaim keeps the clock; a release that frees memory
        // restarts it.
        assert_eq!(reclaim.observe(MemoryObservation::Reclaim, 9), reclaim);
        let released = reclaim.released(100);
        assert_eq!(released, MemoryCondition::Reclaim { since: 100 });
        assert!(!released.should_unload(100 + MEMORY_ESCALATION_NS - 1));
        assert!(released.should_unload(100 + MEMORY_ESCALATION_NS));
        // A failed reading during Reclaim neither resets nor extends it.
        assert_eq!(released.observe(MemoryObservation::Blind, 200), released);
    }

    #[test]
    fn recovery_clears_reclaim_and_a_new_episode_starts_fresh() {
        let reclaim = MemoryCondition::Normal.observe(MemoryObservation::Reclaim, 5);
        let normal = reclaim.observe(MemoryObservation::Normal, 10);
        assert_eq!(normal, MemoryCondition::Normal);
        assert!(!normal.should_unload(10 + MEMORY_ESCALATION_NS));
        assert_eq!(normal.released(11), MemoryCondition::Normal);
        let renewed = normal.observe(MemoryObservation::Reclaim, 11);
        assert_eq!(renewed, MemoryCondition::Reclaim { since: 11 });
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
        let parked = State::Parked(round, Work::new(vec![work]).unwrap(), Some(Wait::Memory));
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
        let pending = State::Between(generation, Stage::Pending(priming(), Some(Wait::Memory)));
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
                Stage::Pending(priming(), Some(Wait::Memory)),
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
        let state = State::Between(generation, Stage::NonResident(Some(Wait::Memory)));
        assert!(state.settled().is_some());
        let State::Between(generation, _) = state else { unreachable!() };
        let state = State::Between(generation, Stage::AwaitingPrefix { through: 1 });
        assert!(state.settled().is_some());
    }
}

#[cfg(test)]
mod ladder_tests {
    use super::{Ladder, MEMORY_ESCALATION_NS, MemoryCondition, Release, RoomOwed};

    /// A probe with no resize under way owes the claim one decision, taken
    /// at the first boundary that can decide it, however many observations
    /// without the bindings pass first; a resize finishing since, or under
    /// way at the probe, leaves nothing owed.
    #[test]
    fn a_probe_owes_the_room_one_decision_until_the_bindings_return() {
        let mut owed = RoomOwed::default();
        owed.probed(false);
        assert!(!owed.take(false), "a flight holds the bindings");
        assert!(!owed.take(false));
        assert!(owed.take(true));
        assert!(!owed.take(true), "decided once");
        owed.probed(false);
        owed.resized();
        assert!(!owed.take(true), "a resize finished since the probe");
        owed.probed(true);
        assert!(!owed.take(true), "a resize ran across the probe");
    }

    /// A release deferred for a gap keeps the ladder in motion, so nothing
    /// is refused or unloaded for memory meanwhile, until the next boundary
    /// takes it to try again; one pending keeps it in motion until its
    /// resize finishes.
    #[test]
    fn a_deferred_release_holds_refusal_and_unload_until_its_boundary() {
        let mut ladder = Ladder::default();
        assert!(ladder.settled());
        ladder.released(Release::Deferred);
        assert!(!ladder.settled(), "a deferred shrink is not an exhausted ladder");
        assert!(ladder.retry_deferred(), "the next boundary tries it again");
        assert!(ladder.settled() && !ladder.retry_deferred());
        ladder.released(Release::Pending(crate::release::Pending::Bank));
        assert!(!ladder.settled());
        ladder.released(Release::Nothing);
        assert!(!ladder.settled(), "nothing more at a rung leaves the pending release");
    }

    /// A slot release and a bank resize pending at once each hold the
    /// ladder until its own finish: either finishing first leaves the other
    /// pending.
    #[test]
    fn a_slot_release_and_a_bank_resize_pending_at_once_each_finish_alone() {
        use crate::release::Pending;
        for (first, second) in [(Pending::Slots, Pending::Bank), (Pending::Bank, Pending::Slots)] {
            let mut ladder = Ladder::default();
            ladder.released(Release::Pending(Pending::Slots));
            ladder.released(Release::Pending(Pending::Bank));
            *ladder.pending.of(first) = false;
            assert!(!ladder.settled(), "{second:?} still runs after {first:?} finished");
            assert!(*ladder.pending.of(second));
            *ladder.pending.of(second) = false;
            assert!(ladder.settled());
        }
    }

    /// What the host has yet to show released holds refusal and unload, as a
    /// running release does.
    #[test]
    fn a_covered_release_holds_refusal_and_unload() {
        let mut ladder = Ladder::default();
        ladder.released(Release::Covered);
        assert!(!ladder.settled());
    }

    /// A release the host has yet to show is waited on however long
    /// Reclaim lasts: the clock neither unloads the model nor asks the bank
    /// again for what it released (Metal unwires a released slab 0.47 s to
    /// over 3 s after its release, past the 1 s clock), so one deficit is
    /// released once.
    #[test]
    fn a_release_the_host_has_yet_to_show_is_never_released_again() {
        let mut ladder = Ladder::default();
        ladder.released(Release::Covered);
        let condition = MemoryCondition::Reclaim { since: 0 };
        for now in [0, MEMORY_ESCALATION_NS, 3 * MEMORY_ESCALATION_NS] {
            assert!(!ladder.unload_due(condition, now), "at {now} ns");
        }
    }

    /// More taken than is credited is real pressure however the credited
    /// release fares: the next walk finds it short and, every rung spent,
    /// the model unloads once Reclaim outlasts the clock.
    #[test]
    fn pressure_past_a_release_the_host_never_shows_unloads() {
        let mut ladder = Ladder::default();
        ladder.released(Release::Covered);
        let condition = MemoryCondition::Reclaim { since: 0 };
        assert!(!ladder.unload_due(condition, MEMORY_ESCALATION_NS));
        // A reading short past the credit walks the rungs again, to the last.
        ladder.walk();
        ladder.released(Release::Nothing);
        assert!(!ladder.unload_due(condition, MEMORY_ESCALATION_NS - 1));
        assert!(ladder.unload_due(condition, MEMORY_ESCALATION_NS));
    }
}

#[cfg(test)]
mod measurement_room_tests {
    use super::{BankStep, MeasurementRoom, RoomClaim};
    use crate::protocol::{RoomAnswer, RoomGrant, RoomRefusal};
    use std::task::{Context, Poll, Waker};

    /// A claim, and the grant its job waits on.
    fn claimed() -> (RoomClaim, RoomGrant) {
        let (granter, grant) = RoomGrant::new();
        (RoomClaim(granter), grant)
    }

    fn answer(grant: &mut RoomGrant) -> Option<RoomAnswer> {
        match grant.poll_answer(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(answer) => Some(answer),
            Poll::Pending => None,
        }
    }

    /// A claim short of its room shrinks the bank by the rest and waits; it
    /// is answered only at a later probe that reads the room made.
    #[test]
    fn a_claim_is_answered_after_its_shrink_makes_the_room() {
        let (claim, mut grant) = claimed();
        let mut asked = None;
        let claim = claim
            .decide::<()>(MeasurementRoom::Short(4096), |bytes| {
                asked = Some(bytes);
                Ok(BankStep::Begun(magnitude_executor::BankTier::Bank))
            })
            .unwrap()
            .expect("the claim waits for its shrink");
        assert_eq!(asked, Some(4096), "the bank shrinks by what the room is short");
        assert_eq!(answer(&mut grant), None, "not answered before the shrink is read");
        let unshrunk = |_| -> Result<BankStep, ()> { panic!("a made room shrinks nothing") };
        assert!(claim.decide(MeasurementRoom::Made, unshrunk).unwrap().is_none());
        assert_eq!(answer(&mut grant), Some(RoomAnswer::Made));
    }

    /// A shrink that cannot run yet (no gap, or another resize) leaves the
    /// claim waiting for the next probe, which shrinks again; one at the
    /// bank's floor refuses it.
    #[test]
    fn a_claim_at_the_floor_is_refused_and_one_without_a_gap_waits() {
        let (claim, mut grant) = claimed();
        let claim = claim
            .decide::<()>(MeasurementRoom::Short(4096), |_| Ok(BankStep::Deferred))
            .unwrap()
            .expect("a deferred shrink waits");
        let claim = claim
            .decide::<()>(MeasurementRoom::Unread, |_| panic!("an unread room shrinks nothing"))
            .unwrap()
            .expect("an unread room waits");
        assert_eq!(answer(&mut grant), None);
        let claim = claim
            .decide::<()>(MeasurementRoom::Draining, |_| panic!("a draining room shrinks nothing"))
            .unwrap()
            .expect("a room whose bank's traffic still runs waits");
        assert_eq!(answer(&mut grant), None);
        assert!(claim
            .decide::<()>(MeasurementRoom::Short(4096), |_| Ok(BankStep::AtLimit))
            .unwrap()
            .is_none());
        assert_eq!(answer(&mut grant), Some(RoomAnswer::Refused(RoomRefusal::AtFloor { short: 4096 })));
    }

    /// A claim short by more than the bank holds above its floor is refused
    /// at once, before anything shrinks.
    #[test]
    fn a_claim_beyond_the_banks_floor_is_refused_without_a_shrink() {
        let (claim, mut grant) = claimed();
        let beyond = MeasurementRoom::Beyond { short: 8192, releasable: 4096 };
        assert!(claim
            .decide::<()>(beyond, |_| panic!("a claim no shrink can meet shrinks nothing"))
            .unwrap()
            .is_none());
        let refused = RoomRefusal::BeyondFloor { short: 8192, releasable: 4096 };
        assert_eq!(answer(&mut grant), Some(RoomAnswer::Refused(refused)));
    }
}
