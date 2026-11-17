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
        ServiceLimits,
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
    round_driver::{RoundError, lower_round, reconcile_forward},
    serving::ServingCensus,
    worker::{Reread, Wakes},
};
use magnitude_executor::{
    BankResize, BankResized, BankStep, BankTier, DomainError, Followed, LoadMeasures, InvariantError,
    MemoryChargeReconciliation, MemoryReading, NativeFamily, SizingTerm, SizingTerms,
    Operation, Outcome, PendingOperationOutcome, PhysicalDecision,
    ProgramFamily, RequestId, ResourcePlan, StateBindings, SubmitError,
    SubmitFailure, WorkKind, Yielded,
};
use magnitude_executor::memory::ledger::Relief;
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{
    FinishReason, Generation, OutputToken, RoundStart, StartedRound, TimedLaunch,
};
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

/// A residency or provisioning that waits on memory once the cache the
/// ledger took was freed: a request standing beside it holds what it
/// lacks, or the cache it took frees off this thread, or a real shortage
/// holds new work back (`MemoryResponse::admission_waits`). It retries at
/// the next change of ours past `at` (an eviction, a release, a request
/// ending, a reading that ends the shortage); one no change of ours can
/// let in fails at once with the typed capacity error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wait {
    at: AvailabilityEpoch,
    on: WaitsOn,
}

/// What a wait is for: memory, or the end of another request's prompt
/// chain, whose queued chunks an open would orphan (a flight boundary,
/// which advances the epoch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WaitsOn {
    Memory,
    Chain,
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
        DomainError::Input(error) => RequestError::Input(error),
        DomainError::State(error) => RequestError::State(error),
        DomainError::Device(error) | DomainError::Submit(SubmitError::Device(error)) => {
            RequestError::Device(error)
        }
        DomainError::Invariant(error) | DomainError::Submit(SubmitError::Invariant(error)) => {
            RequestError::Invariant(error)
        }
        DomainError::WeightRead(error) => RequestError::WeightRead(error),
        error @ (DomainError::StateCeiling { .. } | DomainError::ChainReleased(_)) => RequestError::Invariant(InvariantError {
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

/// An eviction that failed an engine invariant, as the domain's own
/// releases fail.
fn release_invariant(context: &'static str) -> impl FnOnce(String) -> DomainError {
    move |detail| DomainError::Invariant(InvariantError { context, detail })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    /// It never ran, and waits on memory: a reading that is not blind, a
    /// standing request's end, or the cache the ledger took to be freed.
    MemoryBlocked,
    Terminal(FinishReason),
}

/// A request refused before it acquires accepted numerical state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
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
        Self::Refused(classify_domain_error(error))
    }
}

/// What one [`Owner::run`] left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ran {
    /// Whether a group is on the device.
    pub in_flight: bool,
    /// Memory escalation decided to unload the model. Every request has been
    /// terminated with memory pressure; the caller drains the flight.
    pub unload: bool,
    /// The owner's next reading of memory that no arrival brings.
    pub reread: Reread,
}

/// A running expert bank resize: what the process held in the device's
/// store when it began (`ExecutorDomain::held_bytes`), and why it began:
/// the reading its store's target was derived from, the generation of the
/// reading the store's previous resize began on (`before`), and the terms
/// that moved between the two, with the one whose change set the target
/// (`SizingTerms::term`), which trace the resize to the event that moved
/// them. A store's first resize follows the load's own reading: its term
/// is `load`, nothing moved before it, and it names no `before`.
#[derive(Clone, Debug)]
struct BankBegun {
    held: u64,
    cause: BankBegunFor,
    reading: SizingTerms,
    before: Option<u64>,
    moved: Vec<SizingTerm>,
    term: Option<&'static str>,
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
    /// It waits on memory: its requests wait.
    Short(OperationGroup, Wait),
    /// Over budget once every release was tried and it holds one operation:
    /// its requests fail at once with the typed capacity error.
    Refused(OperationGroup, RequestError),
    /// Its chain's holder's group was not the chain's queued chunk: the
    /// chain was released with the holder's state, which replays
    /// (`DomainError::ChainReleased`).
    Replay(OperationGroup, RequestId),
}

/// Everything the owner holds except its pipeline.
struct Service<F: ProgramFamily> {
    domain: ExecutorDomain<F>,
    scheduler: Scheduler,
    /// Requests outside the pipeline.
    requests: BTreeMap<RequestId, Request>,
    epoch: AvailabilityEpoch,
    now: u64,
    prefix_cache: PrefixCache<ResumeState>,
    /// What the owner holds of memory between flights ([`between_flights`]).
    memory: BetweenFlights,
    /// The model unloaded for memory since the run last asked
    /// ([`MemoryHost::unload`]).
    unloaded: bool,
    /// The timed reading the run's last step that read memory left
    /// ([`Self::ran`]).
    reread: Reread,
    wakes: Arc<dyn Wakes>,
}

/// What a loaded model's owner holds of its memory between flights
/// ([`between_flights`]): the expert bank's running resizes, the load's
/// newly stored measures, and whether new work waits on a real shortage.
pub struct BetweenFlights {
    /// Each store's running expert bank resize this owner began: the stores
    /// resize on their own (`BankStep::Begun`).
    bank_began: [Option<BankBegun>; BankTier::COUNT],
    /// Costs the executor stored since the load was priced, which price it
    /// from the next flight boundary on ([`Owner::measure_load`]).
    measures: Option<LoadMeasures>,
    /// The last reading's shortage holds the bank at its floor: no request
    /// becomes resident anew until a reading finds none (the ordered
    /// response's second step, `MemoryResponse::admission_waits`).
    admission_waits: bool,
    /// A reading is owed at the next step whatever brings it: the load's
    /// first, the worker's timed one ([`Owner::reread`]), or one a change
    /// beside it deferred (`MemoryReading::Beside`).
    owed: bool,
    /// Readings generation when the worker went idle with [`Reread::Settled`]
    /// (no flight and no timed reading running): the next admitted request
    /// owes a fresh reading before selection/provisioning, and an unpressured
    /// wake whose stores were unread while asleep opens a fresh baseline.
    settled: Option<u64>,
    /// Each store's reading its last resize began on: what the next one's
    /// moved terms are read against.
    sized: [Option<SizingTerms>; BankTier::COUNT],
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
    /// step.
    pub fn new() -> Self {
        Self {
            bank_began: [const { None }; BankTier::COUNT],
            measures: None,
            admission_waits: false,
            owed: true,
            settled: None,
            sized: [const { None }; BankTier::COUNT],
        }
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
    /// Evict the retained prefixes worth least while the ledger owes their
    /// cache (`ExecutorDomain::prefix_owed`), settling the slabs after
    /// each: whether any was. A host without a prefix cache retains none.
    fn evict_owed_prefixes(&mut self) -> bool;
    /// The retained prefix worth least: the prefill tokens it saves and its
    /// hits since it was retained, which value the prefixes' cache
    /// (`ExecutorDomain::value_cache`); none without one.
    fn prefix_value(&self) -> Option<(u64, u64)>;
    /// The requests the host can evict now: the only ones a real
    /// shortage's relief names ([`Self::preempt`]).
    fn evictable(&self) -> BTreeSet<RequestId>;
    /// Evict `sequences`, the requests a real shortage's relief named, each
    /// of them evictable since the reading that named them.
    fn preempt(&mut self, sequences: &[RequestId]) -> Result<(), RequestError>;
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
        // The OS's pressure events wake the owner, whose run reads memory.
        domain.watch_pressure(wakes.pressure());
        Ok(Self {
            service: Service {
                domain,
                scheduler: Scheduler::new(limits)?,
                requests: BTreeMap::new(),
                epoch: AvailabilityEpoch::default(),
                now: 0,
                prefix_cache: PrefixCache::new(capacity),
                memory: BetweenFlights::new(),
                unloaded: false,
                reread: Reread::Unread,
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
            readings: self.service.domain.observe_memory(),
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
    /// now, refused only where the model's committed memory leaves it short.
    /// A measurement waits for no request's end: short is short.
    pub fn claim_measurement_room(&mut self, bytes: u64) -> RoomGrant {
        let (granter, grant) = RoomGrant::new();
        let claimed = self.service.domain.claim_measurement_room(bytes);
        granter.answer(claimed.map_err(|short| RoomRefusal::Short { bytes: short.over }));
        grant
    }

    /// The measurement job holding the room was started as `process`
    /// (`ExecutorDomain::attach_measurement_process`).
    pub fn attach_measurement_process(&mut self, process: magnitude_executor::ProcessId) {
        self.service.domain.attach_measurement_process(process);
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

    /// The worker's timed reading of memory is due ([`Reread`]): the next
    /// run reads memory.
    pub fn reread(&mut self) {
        self.service.memory.owed = true;
    }

    /// What the census counts moves only with: the generation of what the
    /// load's ledgers hold (`ExecutorDomain::holdings_generation`), and the
    /// expert bank's cover (`ExecutorDomain::expert_cover`), which moves
    /// through decode with no holding moving. Equal across two runs,
    /// nothing the census counts changed.
    pub fn census_key(&self) -> (u64, Option<magnitude_executor::ExpertCover>) {
        (self.service.domain.holdings_generation(), self.service.domain.expert_cover())
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
        let id = RequestId::admitted();
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
        if self.memory.settled.is_some() {
            self.memory.owed = true;
        }
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
            Some(_) if self.waits_on_memory(request) => return Status::MemoryBlocked,
            // Behind another request's chain it waits on no memory: it runs
            // once the chain ends.
            Some(Wait { on: WaitsOn::Chain, .. }) => {}
            Some(_) => return Status::Preempted,
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
                        if let Err(error) = self.step(None, false) {
                            return Err((Some(round.members), error));
                        }
                        unload |= std::mem::take(&mut self.unloaded);
                        return Ok((Pipeline::InFlight(round, active), self.ran(true, unload)));
                    }
                    // Reconciling reads memory at the flight's boundary and
                    // submits the round's next flight.
                    submitted = true;
                    self.reconcile(round, active)?
                }
                Pipeline::Idle(mut bindings) => {
                    self.step(Some(&mut bindings), false).map_err(|error| (None, error))?;
                    unload |= std::mem::take(&mut self.unloaded);
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
                            let ran = self.ran(false, unload);
                            if ran.reread == Reread::Settled {
                                self.memory.settled = Some(self.domain.readings_generation());
                            }
                            return Ok((Pipeline::Idle(bindings), ran));
                        }
                    }
                }
            };
        }
    }

    /// The memory step at an event ([`MemoryStep::observe`]), `boundary`
    /// where it is a flight's: the timed reading it leaves kept for the
    /// run's end ([`Self::ran`]); an unload is the host's
    /// ([`MemoryHost::unload`]).
    fn step(&mut self, bindings: Option<&mut StateBindings>, boundary: bool) -> Result<(), RequestError> {
        let observed = self.observe(bindings, boundary)?;
        if observed.reread != Reread::Unread {
            self.reread = observed.reread;
        }
        Ok(())
    }

    /// What the run came to. A flight in progress, or an unload, owes no
    /// timed reading (the flight's boundary reads next); otherwise the
    /// last step that read memory this run sets it, and a run that read
    /// none leaves it as it was.
    fn ran(&mut self, in_flight: bool, unload: bool) -> Ran {
        let reread = std::mem::replace(&mut self.reread, Reread::Unread);
        let reread = match in_flight || unload {
            true => Reread::Settled,
            false => reread,
        };
        Ran { in_flight, unload, reread }
    }
}

/// What one memory step came to: whether the model unloaded, and the timed
/// reading it leaves ([`Reread`]).
struct Observed {
    unload: bool,
    reread: Reread,
}

/// The owner's memory between flights, on what a host gives it
/// ([`MemoryHost`]): every host takes these, and none has its own.
trait MemoryStep: MemoryHost + Sized {
    /// The memory step at an event: `boundary` where it is a flight's
    /// boundary, `bindings` there where no flight holds them. Returns
    /// whether the model unloaded, after terminating every request, and
    /// the timed reading the step leaves. This is the owner's own work,
    /// which serves no request: no launch is charged the time it takes
    /// (`ExecutorDomain::owner_worked`).
    fn observe(&mut self, bindings: Option<&mut StateBindings>, boundary: bool) -> Result<Observed, RequestError> {
        let began = std::time::Instant::now();
        let observed = self.observe_memory(bindings, boundary);
        self.domain().owner_worked(began.elapsed());
        observed
    }

    /// The OS is read only at events (design §4.8), never at every run of
    /// the owner: a flight's boundary, the completion of a change of ours
    /// (a measured load priced, a bank resize or a slot release finished),
    /// an OS pressure event, and a reading owed (the load's first, the
    /// worker's timed one, one a change beside deferred). The reading
    /// (`ExecutorDomain::read_memory`) never reads our own change as another
    /// program's, and a real shortage is answered once per reading, in its
    /// order (spec §4.5), each step only where the one before did not meet
    /// it:
    /// 1. cache yields in rank order at once (taken by the reading); each
    ///    holder frees what was taken of it here, and the bank follows its
    ///    target, derived from the ledgers afresh;
    /// 2. new work waits while the bank sits at its floor;
    /// 3. the named sequences are preempted, the lowest value first;
    /// 4. the model unloads, only on a shortage that holds across readings
    ///    spanning the store's lag with no change of ours pending.
    ///
    /// Where no flight holds the bindings, what claims took of the cache is
    /// freed and the bank follows its target at every step: neither reads
    /// the OS.
    fn observe_memory(&mut self, mut bindings: Option<&mut StateBindings>, boundary: bool) -> Result<Observed, RequestError> {
        // A cut of the expert bank a chain's encoding began beside its
        // running steps (`ExecutorDomain::take_running_release`) is a shrink
        // this owner holds as begun: the advance that finishes it reports
        // it as any.
        if let Some(tier) = self.domain().take_running_release() {
            let held = self.domain().held_bytes();
            let reading = self.domain().sizing_terms(tier);
            self.begin_resize(tier, BankBegunFor::Shrink, held, reading)?;
        }
        // Own changes that complete here: newly stored costs price the load
        // first, so every decision past it is made on them; the expert
        // bank's and the graph slots' releases advance. The cache and each
        // sequence's state are valued afresh.
        let mut completed = false;
        if let Some(bindings) = bindings.as_deref_mut() {
            // A refusal (the device would not take the tier copy's grid) is a device
            // fault the load ends on, the experts still priced as before.
            if let Some(measures) = self.memory().measures.take() {
                self.domain().measure_load(measures, bindings).map_err(classify_domain_error)?;
                completed = true;
            }
            completed |= self.advance_expert_bank()?;
            completed |= self.advance_slot_release()?;
            let prefix = self.prefix_value();
            self.domain().value_cache(prefix).map_err(classify_domain_error)?;
            self.domain().value_state().map_err(classify_domain_error)?;
        }
        let owed = std::mem::take(&mut self.memory().owed);
        let pressured = self.domain().take_pressure();
        if pressured {
            self.memory().settled = None;
        }
        let settled = self.memory().settled;
        let after_idle = settled == Some(self.domain().readings_generation());
        let read = match boundary || completed || owed || pressured {
            false => None,
            true => match {
                let evictable = self.evictable();
                self.domain().read_memory(after_idle, &evictable).map_err(classify_domain_error)?
            } {
                MemoryReading::Read(read) => {
                    self.memory().settled = None;
                    Some(read)
                }
                // A change off this thread stood beside the sample: its
                // completion wakes the owner, whose step reads then.
                MemoryReading::Beside => {
                    self.memory().owed = true;
                    None
                }
            },
        };
        if let Some(bindings) = bindings {
            self.follow_yields(bindings)?;
        }
        let Some(read) = read else {
            return Ok(Observed { unload: false, reread: Reread::Unread });
        };
        let response = read.response.unwrap_or_default();
        if response.unload {
            self.unload();
            return Ok(Observed { unload: true, reread: Reread::Settled });
        }
        // Each store's plan is carried out whole; a request two stores'
        // plans name is evicted once.
        let mut preempt = BTreeSet::new();
        for plan in response.relief.iter().cloned() {
            preempt.extend(self.domain().execute_relief(plan).preempt);
        }
        let preempted = !preempt.is_empty();
        if preempted {
            let preempt = preempt.into_iter().collect::<Vec<_>>();
            tracing::info!(requests = ?preempt, "a real memory shortage preempts the requests worth least");
            self.preempt(&preempt)?;
        }
        // New work waits while a shortage holds the bank at its floor; the
        // reading that finds none lets it retry.
        let waited = std::mem::replace(&mut self.memory().admission_waits, response.admission_waits);
        if waited && !response.admission_waits {
            self.advance_epoch();
        }
        // The reading changed something where its response moved what we
        // hold (the ledgers' generation past the reading's own), or
        // preempted, or let new work in or held it back.
        let changed = preempted
            || waited != response.admission_waits
            || self.domain().memory_generation() != read.generation;
        let reread = match self.domain().memory_stands().map_err(classify_domain_error)? {
            true => Reread::Stands { cost: read.cost, changed, ceiling: self.domain().others_interval() },
            false => Reread::Settled,
        };
        Ok(Observed { unload: false, reread })
    }

    /// A resize of `tier`'s storage begun for `cause`, `held` the bytes the
    /// process held as it began: held as begun with the reading its target
    /// was derived from and the terms that moved since the store's last.
    fn begin_resize(
        &mut self,
        tier: BankTier,
        cause: BankBegunFor,
        held: u64,
        reading: magnitude_executor::SizingTerms,
    ) -> Result<(), RequestError> {
        let (before, moved, term) = match self.memory().sized[tier.index()].replace(reading.clone()) {
            Some(before) => (before.reading, reading.moved(&before), reading.term(&before)),
            None => (None, Vec::new(), Some("load")),
        };
        let begun = BankBegun { held, cause, reading, before, moved, term };
        self.memory().bank_begun(tier, begun).map_err(invariant("expert bank"))
    }

    /// Follow the expert bank's derived target
    /// (`ExecutorDomain::follow_cache_targets`): a resize it begins is held as
    /// begun, a growth as one and a shrink as the target's. One deferred or
    /// running is followed again at the next boundary, which derives afresh.
    /// Whether the bank owes a shrink: one begun, running, or deferred to a
    /// boundary the flight that defers it brings; its finish advances the
    /// epoch, so what waits on memory retries then.
    fn follow_cache_targets(
        &mut self,
        bindings: &mut StateBindings,
        sizing: [magnitude_executor::SizingTerms; 2],
    ) -> Result<bool, RequestError> {
        let (before, wake) = (self.domain().held_bytes(), self.completion());
        let followed = self.domain().follow_cache_targets(bindings, wake).map_err(classify_domain_error)?;
        let (tier, cause) = match followed {
            Followed::Grew(BankStep::Begun(tier)) => (tier, BankBegunFor::Growth),
            Followed::Shrank(BankStep::Begun(tier)) => (tier, BankBegunFor::Shrink),
            Followed::Shrank(BankStep::Running | BankStep::Deferred) => return Ok(true),
            Followed::Holds | Followed::Grew(_) | Followed::Shrank(BankStep::AtLimit) => return Ok(false),
        };
        let [bank, host] = sizing;
        let reading = match tier {
            BankTier::Bank => bank,
            BankTier::Host => host,
        };
        self.begin_resize(tier, cause, before, reading)?;
        Ok(cause == BankBegunFor::Shrink)
    }

    /// Free what the ledger took of the load's cache, as its holders do:
    /// the domain's own kinds at once (`ExecutorDomain::release_yielded`),
    /// the retained prefixes worth least while their cache is owed
    /// ([`MemoryHost::evict_owed_prefixes`]), and the expert bank, which
    /// follows its target derived afresh ([`Self::follow_cache_targets`]):
    /// a claim that took the bank's cache allocates it only once the bank's
    /// shrink frees it, so a shrink owed is pending. The one release order
    /// every boundary and every claim short of memory follow. Each release
    /// is recorded with what the process held on each side of it, and what
    /// waits on memory retries once one freed any.
    fn follow_yields(&mut self, bindings: &mut StateBindings) -> Result<Yielded, RequestError> {
        let sizing = [BankTier::Bank, BankTier::Host].map(|tier| self.domain().sizing_terms(tier));
        let (before, started, wake) = (self.domain().held_bytes(), std::time::Instant::now(), self.completion());
        let yielded = self.domain().release_yielded(bindings, wake).map_err(classify_domain_error)?;
        let yielded = match (self.evict_owed_prefixes(), yielded) {
            (true, Yielded::Pending | Yielded::Nothing) => Yielded::Released { released: 0, returned: 0 },
            (_, yielded) => yielded,
        };
        if let Yielded::Released { released, returned } = yielded {
            tracing::info!(
                micros = started.elapsed().as_micros() as u64,
                held_before = before,
                held_after = self.domain().held_bytes(),
                state_released = released,
                state_returned = returned,
                "memory released: the cache the ledger took"
            );
            self.advance_epoch();
        }
        let shrinking = self.follow_cache_targets(bindings, sizing)?;
        Ok(match (yielded, shrinking) {
            (Yielded::Nothing, true) => Yielded::Pending,
            (yielded, _) => yielded,
        })
    }

    /// At a flight boundary, finish the idle graph slots' release off the
    /// submitting thread (`ExecutorDomain::advance_slot_release`), recorded
    /// with the slots and how long the release thread took. It freed
    /// memory, so what waited on memory may retry. Whether one finished.
    fn advance_slot_release(&mut self) -> Result<bool, RequestError> {
        let Some(released) = self.domain().advance_slot_release().map_err(classify_domain_error)? else {
            return Ok(false);
        };
        tracing::info!(
            slots = released.slots,
            released = released.bytes,
            release_micros = released.released.duration_since(released.began).as_micros() as u64,
            held_after = self.domain().held_bytes(),
            "graph slots released"
        );
        self.advance_epoch();
        Ok(true)
    }

    /// At a flight boundary, advance the expert bank's resize
    /// (`ExecutorDomain::advance_expert_bank`). A finished one is recorded as
    /// a release is (a shrink as memory released, a growth as memory
    /// grown): how long it held this thread, the bank's slots after it,
    /// what the process held when it began and now, and the reading it
    /// began on with the terms of it that moved. A finished shrink freed
    /// memory: what waited on memory may retry. Whether one finished.
    fn advance_expert_bank(&mut self) -> Result<bool, RequestError> {
        let wake = self.completion();
        let Some((finished, held)) = self.domain().advance_expert_bank(wake).map_err(classify_domain_error)? else {
            return Ok(false);
        };
        let after = self.domain().held_bytes();
        let tier = self.domain().expert_holding_storage(finished.holding()).map_err(classify_domain_error)?;
        let BankBegun { held: before, cause, reading, before: before_generation, moved, term } = self.memory().bank_began[tier.index()]
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
        // The reading the resize began on: its store, its generation (as
        // its `memory read` record states it), how long before now it was
        // taken, its terms, the generation of the reading the store's last
        // resize began on (none for its first, whose term is `load`), the
        // terms that moved between the two, and the one whose change set
        // the target.
        let read_before = before_now(reading.read_at);
        let (store, figures) = (reading.store, reading.figures);
        let reading = reading.reading;
        match finished {
            BankResized::Shrunk { holding, from, to, released, returned, times } => {
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
                    returned,
                    held_before = before,
                    held_after = after,
                    store = ?store,
                    reading,
                    before_generation,
                    term,
                    read_before,
                    figures = ?figures,
                    moved = ?moved,
                    deciding_micros = times.decided.duration_since(held.started).as_micros() as u64,
                    queued_micros = times.queued.as_micros() as u64,
                    planned_micros = times.planned.as_micros() as u64,
                    plan_picked_micros = times.trips[0].picked.as_micros() as u64,
                    plan_served_micros = times.trips[0].served.as_micros() as u64,
                    commit_picked_micros = times.trips[1].picked.as_micros() as u64,
                    commit_served_micros = times.trips[1].served.as_micros() as u64,
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
                    held_before = before,
                    held_after = after,
                    store = ?store,
                    reading,
                    before_generation,
                    term,
                    read_before,
                    figures = ?figures,
                    moved = ?moved,
                    decided_before = before_now(times.decided),
                    prepared_before = before_now(times.prepared),
                    installed_before = before_now(times.installed),
                    released_before = before_now(times.released),
                    "memory grown"
                )
            }
            // Nothing changed: no resize of the holding is recorded.
            BankResized::Stopped { holding, at } => {
                tracing::info!(?holding, at, store = ?store, reading, term, read_before, figures = ?figures, moved = ?moved, "the expert bank's growth stopped before its first slot")
            }
            // Only a growth needs memory: a shrink's refusal broke the bank,
            // which reports it as an invariant violation instead.
            BankResized::GrowthRefused { holding, at, refusal } => {
                if cause != BankBegunFor::Growth {
                    return Err(invariant("expert bank")(format!("a shrink of {holding:?} reported a refusal: {refusal}")));
                }
                tracing::warn!(?holding, at, %refusal, store = ?store, reading, term, read_before, figures = ?figures, moved = ?moved, "the expert bank's storage refused a growth")
            }
        }
        Ok(true)
    }

}

impl<H: MemoryHost> MemoryStep for H {}

/// The owner's memory step at an event, `bindings` there where no flight
/// holds them: memory read now, the expert bank's running resize advanced
/// and its derived target followed, and a real shortage answered in order,
/// the model unloaded only where it persists. Whether it unloaded. The one
/// step every host runs: the owner at each of its events, and an instrument
/// that drives the domain below the scheduler at each of its own, each of
/// which is a flight's boundary.
pub fn between_flights<H: MemoryHost>(host: &mut H, bindings: Option<&mut StateBindings>) -> Result<bool, RequestError> {
    host.observe(bindings, true).map(|observed| observed.unload)
}

impl<F: ProgramFamily> Service<F> {
    /// Whether `request` waits on memory to become resident: it never was,
    /// and memory is being reclaimed or its last attempt was refused for
    /// memory. Preemption is the only eviction and it counts itself
    /// (`preemption_debt`), so a non-resident request without that debt never
    /// was resident; a preempted one is not refused, mid-prefill included.
    /// Such a request is the one refused once only the model's unload could
    /// make room for it.
    fn waits_on_memory(&self, request: &Request) -> bool {
        request.record.preemption_debt == 0
            && matches!(&request.state, State::Between(_, Stage::NonResident(Some(Wait { on: WaitsOn::Memory, .. }))))
    }

    /// End the waits that have ended, then take the requests selection can
    /// start out of the map.
    fn candidates(&mut self) -> BTreeMap<RequestId, (Record, Candidate)> {
        let epoch = self.epoch;
        let ended_wait = |wait: Option<Wait>| wait.is_none_or(|wait| epoch.changed_since(wait.at));
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
        // Idle, the stores' spare state and idle graph slots stay the
        // ledger's cache: they are freed only as it takes them
        // (`ExecutorDomain::release_yielded`), never apart from it.
        let Some(mut selection) = selection else {
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
        Ok(Selected::Round(Round {
            selection,
            started: self.now,
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
                    Residency::Failed(generation) => {
                        return Ok(Start::Returned(Request {
                            record,
                            state: State::Between(generation, Stage::Ready),
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
        // A real shortage holds new work back while the bank sits at its
        // floor (`MemoryResponse::admission_waits`).
        if self.memory.admission_waits {
            return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Memory }));
        }
        // Its whole state is reserved as it opens: every row its prompt and
        // its generation may take (`ExecutorDomain::open_state`), so none of
        // its rows ever waits on memory.
        let rows = generation.request_rows() as u64;
        // Short of memory, it is relieved as any claim is ([`Self::relieve`]),
        // taking cache alone, its prefix looked up afresh at each try, as an
        // eviction may drop it. While another request's chain is under way
        // the open changes no bindings: it waits for the chain's end, a
        // flight boundary.
        let prefix = record.prefix_cache;
        let open = |this: &mut Self, bindings: &mut StateBindings| {
                let Some(unchained) = this.domain.unchained(bindings) else {
                    return Ok(Ok(None));
                };
                let hit = match prefix {
                    true => this
                        .prefix_cache
                        .lookup(PrefixPath::of(&generation), generation.resume_bound())
                        .map_err(invariant("prefix cache"))?,
                    false => None,
                };
                let from = match hit {
                    Some(hit) => Some(this.prefix_cache.entry(hit).map_err(invariant("prefix cache"))?.state()),
                    None => None,
                };
                Ok(this.domain.open_state(unchained, id, from, rows).map(|encodes| Some((hit, encodes))))
            };
        let first = open(self, bindings).map_err(fatal)?;
        let relieved = self.relieve(bindings, first, false, open).map_err(fatal)?;
        let (hit, encodes) = match relieved.current {
            Ok(Some(opened)) => opened,
            Ok(None) => return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Chain })),
            // New work waits for memory a change of ours will free.
            Err(DomainError::Capacity(_))
                if relieved.pending || self.requests.values().any(|request| request.state.generation().is_resident()) =>
            {
                return Ok(Residency::Waiting(generation, Wait { at: self.epoch, on: WaitsOn::Memory }));
            }
            Err(error) if is_fatal(&error) => return Err(fatal(classify_domain_error(error))),
            // Nothing of ours can make room: it fails at once.
            Err(error) => {
                Self::fail_generation(record, &mut generation, classify_domain_error(error));
                return Ok(Residency::Failed(generation));
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
                Ok(()) => {
                    Self::fail_generation(record, &mut generation, invariant("residency")(error));
                    Ok(Residency::Failed(generation))
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
                Ok(Provisioned::Refused(group_, error)) => {
                    self.refuse(&mut round, &group_, error);
                    continue;
                }
                Ok(Provisioned::Replay(group_, holder)) => {
                    self.replay(&mut round, &group_, holder).map_err(|error| (None, classify_domain_error(error)))?;
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
                        DomainError::Input(error) => self.refuse(&mut round, &group_, RequestError::Input(error)),
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
    /// available: the sequences the device's ledger names for the group's
    /// rows are evicted (the ledger took every cheaper byte, cache worth
    /// least first, before it named them), the cache it took is freed where
    /// the device would not give the memory yet, then the group sheds its
    /// trailing operations, which loses no work. What is still short waits
    /// for a change of ours that frees memory (a request standing beside it
    /// ending, a named sequence leaving its round, the cache taken freeing
    /// off this thread); where none can come, its requests fail.
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
        let Relieved { current, pending } = self.relieve(bindings, current, true, |this, bindings| Ok(this.fits(bindings, round, &group_)))?;
        if short(&current) {
            group_ = match group_.split_last() {
                Ok((leading, trailing)) => return Ok(Provisioned::Split(leading, trailing)),
                Err(whole) => whole,
            };
        }
        match current {
            Ok(()) => Ok(Provisioned::Fits(group_)),
            Err(DomainError::Capacity(_)) if pending || self.others_stand(round, &group_) => Ok(Provisioned::Short(group_, Wait { at: self.epoch, on: WaitsOn::Memory })),
            Err(error @ (DomainError::Capacity(_) | DomainError::Input(_))) => Ok(Provisioned::Refused(group_, classify_domain_error(error))),
            Err(DomainError::ChainReleased(holder)) => Ok(Provisioned::Replay(group_, holder)),
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

    /// Fail every request of a group the domain refused with `error`: what
    /// it held frees as each retires.
    fn refuse(&mut self, round: &mut Round, group_: &OperationGroup, error: RequestError) {
        let requests = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        for request in requests {
            Self::fail_member(round, request, error.clone());
        }
        self.epoch.advance();
    }

    /// Move a group that cannot proceed out of the round: each of its
    /// requests takes all of its remaining work with it and waits alone.
    /// `group_`'s requests leave the round between rounds, and `holder`,
    /// whose state its chain's release took, replays from its committed
    /// history, as an evicted request does.
    fn replay(&mut self, round: &mut Round, group_: &OperationGroup, holder: RequestId) -> Result<(), DomainError> {
        tracing::info!(?holder, "a chain holder's group is not its queued chunk: the chain is released and the holder replays");
        let requests = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        // Their work queued in the round's later groups leaves with them.
        let kept = std::mem::take(&mut round.groups)
            .into_iter()
            .flat_map(OperationGroup::into_operations)
            .filter(|operation| !requests.contains(&operation.request()))
            .collect();
        round.groups = group(&self.domain, kept).into_iter().collect();
        for request in requests {
            let Some(Member { record, state }) = round.members.remove(&request) else { continue };
            let generation = match state {
                Membership::Started(started) => started.abandon(),
                Membership::Generation(generation) => generation,
            };
            self.requests.insert(request, Request { record, state: State::Between(generation, Stage::Ready) });
        }
        self.evicted(&[holder])
    }

    fn park(&mut self, round: &mut Round, group_: OperationGroup, wait: Wait) {
        let requests = group_
            .operations()
            .iter()
            .map(Operation::request)
            .collect::<BTreeSet<_>>();
        tracing::info!(
            requests = ?requests,
            rows = group_.operations().iter().map(Operation::row_count).sum::<usize>(),
            "a group waits on memory"
        );
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
        // The flight's boundary: memory is read once here.
        if let Err(error) = self.step(Some(&mut bindings), true) {
            return Err((Some(round.members), error));
        }
        if self.unloaded {
            // The unload terminated every request outside the round; the
            // round's members end with them, and the round with its flight.
            let error = RequestError::ModelUnloaded { cause: ModelUnloadCause::MemoryPressure };
            self.terminalize_all(error, Some(&mut round.members));
            self.finish_round(round).map_err(|error| (None, error))?;
            return Ok(Pipeline::Idle(bindings));
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
            members,
            ..
        } = round;
        let elapsed = self.now.saturating_sub(started);
        self.domain.served_round(selection.phase() == Phase::Decode);
        let count = selection.requests().len() as u64;
        let base = elapsed / count;
        let remainder = elapsed % count;
        let mut members = members;
        for (index, request) in selection.requests().iter().enumerate() {
            let share = base + u64::from((index as u64) < remainder);
            let record = match members.get_mut(request) {
                Some(member) => &mut member.record,
                None => match self.requests.get_mut(request) {
                    Some(request) => &mut request.record,
                    None => continue,
                },
            };
            record.service_ns = record.service_ns.saturating_add(share);
            record.waiting_since = self.now;
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

    /// Whether `request` can be evicted now: live and resident between
    /// rounds (`Stage::Ready` or `Stage::AwaitingCredit`). A round's
    /// members are out of the map, pending or parked work is mid-operation,
    /// and a finished request keeps its state until it retires.
    fn can_evict(request: &Request) -> bool {
        matches!(&request.state, State::Between(generation, Stage::Ready | Stage::AwaitingCredit)
            if generation.is_resident() && generation.finish_reason().is_none())
    }

    /// The one path a claim short of memory takes, from its `first` try,
    /// trying again by `attempt`: first free the cache the ledger took for
    /// claims already admitted ([`MemoryStep::follow_yields`]); where that
    /// frees nothing, ask the device's ledger for relief that covers what
    /// the claim lacks, taking cache, and where `preempting`, the requests
    /// that can be evicted now (`Ledger::plan_relief`), and carry it out
    /// whole: its cache yields and its requests are evicted
    /// ([`Self::evict`]). Where no relief covers it, nothing yields. Each
    /// that freed anything tries again; one that
    /// frees memory only later (a round's end, slots releasing off this
    /// thread, the expert bank's shrink) leaves the claim pending, and the
    /// caller decides from there whether it waits or fails.
    fn relieve<T>(
        &mut self,
        bindings: &mut StateBindings,
        first: Result<T, DomainError>,
        preempting: bool,
        mut attempt: impl FnMut(&mut Self, &mut StateBindings) -> Result<Result<T, DomainError>, RequestError>,
    ) -> Result<Relieved<T>, RequestError> {
        let mut current = first;
        loop {
            let pending = match &current {
                Err(DomainError::Capacity(capacity)) => match self.follow_yields(bindings)? {
                    Yielded::Released { .. } => None,
                    Yielded::Pending => Some(true),
                    Yielded::Nothing => {
                        let lacks = capacity.required.saturating_sub(capacity.available);
                        let evictable = self.evictable();
                        let allow = match preempting {
                            true => Relief::CacheAndRequests(&evictable),
                            false => Relief::CacheOnly,
                        };
                        match self.domain.plan_relief(capacity.resource, lacks, allow) {
                            None => Some(false),
                            Some(plan) => {
                                let owed = self.domain.execute_relief(plan);
                                if !owed.preempt.is_empty() {
                                    self.evict(&owed.preempt).map_err(classify_domain_error)?;
                                }
                                match self.follow_yields(bindings)? {
                                    Yielded::Released { .. } => None,
                                    Yielded::Pending => Some(true),
                                    Yielded::Nothing => match owed.preempt.is_empty() {
                                        true => Some(owed.yielded > 0),
                                        false => None,
                                    },
                                }
                            }
                        }
                    }
                },
                _ => Some(false),
            };
            match pending {
                Some(pending) => return Ok(Relieved { current, pending }),
                None => current = attempt(self, bindings)?,
            }
        }
    }

    /// The one eviction: `sequences`, each evictable now, have their state
    /// released and wait non-resident, a preemption owed.
    fn evict(&mut self, sequences: &[RequestId]) -> Result<(), DomainError> {
        self.domain.release_state(sequences).map_err(release_invariant("preemption"))?;
        self.evicted(sequences)
    }

    /// Whether a request outside `group_` holds memory whose release would
    /// let it in: one resident, or in this round. A claim short beside one
    /// waits for it to end; with none, nothing of ours can free memory for
    /// it, and it fails.
    fn others_stand(&self, round: &Round, group_: &OperationGroup) -> bool {
        let own = group_.operations().iter().map(Operation::request).collect::<BTreeSet<_>>();
        round.members.keys().any(|id| !own.contains(id))
            || self.requests.iter().any(|(id, request)| !own.contains(id) && request.state.generation().is_resident())
    }

    /// Mark `set`, whose state was just released, evicted: each waits
    /// non-resident, a preemption owed.
    fn evicted(&mut self, set: &[RequestId]) -> Result<(), DomainError> {
        for &id in set {
            let Request { mut record, state } = self.requests.remove(&id).expect("an evicted request");
            let State::Between(mut generation, _) = state else {
                unreachable!("an evicted request is between rounds")
            };
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
        tracing::error!(requests = self.requests.len(), "unloading the model: a real memory shortage persists with nothing of ours pending");
        self.unloaded = true;
        self.terminalize_all(
            RequestError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            },
            None,
        );
    }

    fn evict_owed_prefixes(&mut self) -> bool {
        let held = self.held_entries(None);
        let mut evicted = false;
        while self.domain.prefix_owed() > 0 && self.prefix_cache.evict_one(&held) {
            self.domain.settle_state();
            evicted = true;
        }
        evicted
    }

    fn prefix_value(&self) -> Option<(u64, u64)> {
        self.prefix_cache.least_value()
    }

    /// Those that can be evicted now ([`Self::can_evict`]); a chain's
    /// holder between its chunks is resident and ready, but its state is
    /// the chain's until the chain ends: never evicted.
    fn evictable(&self) -> BTreeSet<RequestId> {
        let holder = self.domain.chain_holder();
        self.requests.iter().filter(|(id, request)| Some(**id) != holder && Self::can_evict(request)).map(|(id, _)| *id).collect()
    }

    fn preempt(&mut self, sequences: &[RequestId]) -> Result<(), RequestError> {
        let evictable = self.evictable();
        assert!(sequences.iter().all(|id| evictable.contains(id)), "invariant violated: a relief names a request that cannot be evicted: {sequences:?}");
        self.evict(sequences).map_err(classify_domain_error)
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
    Failed(Generation),
}

/// What relieving a claim short of memory came to ([`Service::relieve`]):
/// its last try, and whether a change of ours under way will free memory
/// for it.
struct Relieved<T> {
    current: Result<T, DomainError>,
    pending: bool,
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
            prefix_cache: true, prefix_source: None,
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
            prefix_cache: true, prefix_source: None,
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
        let parked = State::Parked(round, Work::new(vec![work]).unwrap(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory }));
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
        let pending = State::Between(generation, Stage::Pending(priming(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })));
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
                Stage::Pending(priming(), Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })),
            );
            assert!(pending.settled().is_none());
            // terminate_request discards composed work
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
        let state = State::Between(generation, Stage::NonResident(Some(Wait { at: AvailabilityEpoch::default(), on: WaitsOn::Memory })));
        assert!(state.settled().is_some());
        let State::Between(generation, _) = state else { unreachable!() };
        let state = State::Between(generation, Stage::AwaitingPrefix { through: 1 });
        assert!(state.settled().is_some());
    }
}

/// The owner over the controlled domain (`magnitude_executor::fixtures`):
/// opens refused for memory, relief's evictions, and request identities
/// across loads.
#[cfg(test)]
mod claim_tests {
    use super::*;
    use crate::publication::Publication;
    use magnitude_executor::fixtures;
    use magnitude_generation::{EndOfGeneration, InputLayout, MethodChoice, Options, Sampling, Shaping, TokenId};

    struct NoWakes;
    impl Wakes for NoWakes {
        fn completion(&self) -> magnitude_executor::CompletionWake {
            magnitude_executor::CompletionWake::new(|| {})
        }
        fn publication(&self, _: RequestId, _: PublicationWake) {}
        fn pressure(&self) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
    }

    /// Rounds of two rows, as the fixture's launches carry.
    const ROWS: usize = 2;

    fn limits() -> ServiceLimits {
        ServiceLimits {
            prefill_tokens: ROWS,
            decode_tokens: ROWS,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        }
    }

    /// A fresh generation of four prompt tokens within the fixture's
    /// eight-token context.
    fn fresh(vocabulary: usize) -> Generation {
        fresh_by(vocabulary, MethodChoice::Plain)
    }

    /// [`fresh`], generating by `method`: plain, or MTP by its own method.
    fn fresh_by(vocabulary: usize, method: MethodChoice) -> Generation {
        let factory: Arc<dyn magnitude_generation::Method> = match method {
            MethodChoice::Plain => Arc::new(magnitude_generation::Plain),
            MethodChoice::Mtp { proposals } => Arc::new(magnitude_generation::Mtp::new("fixture", proposals.into()).unwrap()),
            other => unreachable!("the owner's tests generate by no {other:?}"),
        };
        Generation::new_with_method(
            vec![TokenId(1), TokenId(2), TokenId(3), TokenId(4)],
            InputLayout::new(4, vec![]).unwrap(),
            Options {
                max_tokens: 2,
                output_capacity: 4,
                context_limit: 8,
                vocabulary,
                stop_tokens: BTreeSet::new(),
                suppressed_tokens: BTreeSet::new(),
                sampling: Sampling::Greedy,
                shaping: Shaping { temperature: 0.0, ..Default::default() },
                seed: 0,
                forced_quantum: 0,
                method,
                end_of_generation: EndOfGeneration::Stop,
                reasoning_budget: None,
            },
            None,
            factory,
        )
        .unwrap()
    }

    /// Make `id` resident between rounds, as selection does: whether it is
    /// (`Some(true)`) or waits (`Some(false)`); none where it failed.
    fn residency(service: &mut Service<fixtures::TestFamily>, bindings: &mut StateBindings, id: RequestId) -> Option<bool> {
        let Request { mut record, state } = service.requests.remove(&id).expect("an admitted request");
        let State::Between(generation, Stage::NonResident(None)) = state else { panic!("{id:?} is not waiting to be resident") };
let Ok(residency) = service.make_resident(id, &mut record, generation, bindings) else { panic!("{id:?}'s open is fatal") };
        let (state, resident) = match residency {
            Residency::Resident(generation, _) => (State::Between(generation, Stage::Ready), Some(true)),
            Residency::Waiting(generation, _) => (State::Between(generation, Stage::NonResident(None)), Some(false)),
            Residency::Failed(generation) => (State::Between(generation, Stage::NonResident(None)), None),
        };
        service.requests.insert(id, Request { record, state });
        resident
    }

    /// Take every byte the device's ledger would give new work now (its
    /// free room and the cache worth nothing) as a measurement's room.
    fn take_room<F: ProgramFamily>(service: &mut Service<F>) {
        const PAST: u64 = u64::MAX / 4;
        service.domain.release_measurement_room();
        let room = match service.domain.claim_measurement_room(PAST) {
            Err(magnitude_executor::memory::ledger::Short { over, .. }) => PAST - over,
            other => panic!("a claim past the budget is refused over it, not {other:?}"),
        };
        service.domain.claim_measurement_room(room).expect("the room the ledger gives is held");
    }

    /// Admit `count` fresh requests in turn.
    fn admit(owner: &mut Owner<fixtures::TestFamily>, count: usize) -> Vec<RequestId> {
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        (0..count)
            .map(|_| owner.admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0).unwrap().0)
            .collect()
    }

    /// The first request draws the load's request slot; with the room
    /// beside it taken, a second request's open is refused for memory and
    /// waits while the first holds the slot: it holds no state, and no
    /// prefix entry is touched.
    #[test]
    fn an_open_refused_while_the_slot_is_lent_waits_and_touches_no_prefix() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let [first, second] = admit(&mut owner, 2)[..] else { unreachable!() };
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        assert_eq!(residency(service, bindings, first), Some(true), "the first draws the slot");
        take_room(service);
        let entries = service.prefix_cache.len();
        assert_eq!(residency(service, bindings, second), Some(false), "the second waits while the slot is lent");
        assert!(matches!(service.requests[&second].state, State::Between(_, Stage::NonResident(None))), "it holds no state");
        assert_eq!(service.prefix_cache.len(), entries, "no prefix entry is touched");
    }

    /// Relief carried out for a claim short of memory evicts exactly the
    /// requests its plan names, the latest admitted first and just enough,
    /// never the first.
    #[test]
    fn relief_evicts_exactly_the_requests_its_plan_names() {
        let (domain, bindings) = fixtures::fixture(None);
        let mut owner = Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap();
        let ids = admit(&mut owner, 3);
        let Pipeline::Idle(bindings) = &mut owner.pipeline else { panic!("an idle owner") };
        let service = &mut owner.service;
        for id in &ids {
            assert_eq!(residency(service, bindings, *id), Some(true), "{id:?} opens");
        }
        take_room(service);
        let resource = magnitude_executor::ResourceKind::DeviceMemory;
        let evictable = MemoryHost::evictable(service);
        let lacks = std::iter::successors(Some(1u64), |bytes| bytes.checked_mul(2))
            .find(|bytes| service.domain.plan_relief(resource, *bytes, Relief::CacheAndRequests(&evictable)).is_some_and(|plan| !plan.preempt().is_empty()))
            .expect("a need some request covers");
        let named = service.domain.plan_relief(resource, lacks, Relief::CacheAndRequests(&evictable)).expect("the plan").preempt().to_vec();
        assert_eq!(named, [ids[2]], "the latest admitted, just enough");
        let short = Err(DomainError::Capacity(magnitude_executor::CapacityError { resource, required: lacks, available: 0 }));
        let relieved = service.relieve(bindings, short, true, |_, _| Ok(Ok(()))).unwrap();
        assert!(relieved.current.is_ok(), "the claim fits once relieved");
        let evicted = |service: &Service<fixtures::TestFamily>, id: RequestId| {
            let request = &service.requests[&id];
            matches!(request.state, State::Between(_, Stage::NonResident(None))) && request.record.preemption_debt == 1
        };
        assert!(evicted(service, ids[2]), "the named request is evicted");
        for id in &ids[..2] {
            assert!(!evicted(service, *id) && service.requests[id].record.preemption_debt == 0, "{id:?} stands");
        }
    }

    /// A drafting load whose drafter was released with nothing resident
    /// serves a drafting request with every byte of room taken: the
    /// drafter's reload is refused for memory while that request's group
    /// provisions, so the request fails on its own, and the owner, never
    /// stopping, serves the next request once the room is free again. Each
    /// request draws the load's request slot in turn, so none of them needs
    /// room to open.
    #[test]
    fn a_drafter_reload_refused_for_memory_fails_its_request_alone() {
        use std::task::{Context, Poll, Waker};
        let (directory, definition, domain, bindings) = fixtures::loaded::drafted("refused-drafter-reload");
        let vocabulary = definition.decoder.vocabulary as usize;
        let mut owner = Some(Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap());
        // The service clock, a millisecond a run.
        let now = std::cell::Cell::new(0u64);
        // Run the owner until `receiver` ends: its terminal publication.
        let serve = |owner: &mut Option<Owner<magnitude_executor::NativeFamily>>, receiver: &mut PublicationReceiver| loop {
            let (ran, _) = owner.take().expect("the owner is between runs").run(now.get()).unwrap_or_else(|error| panic!("the owner stopped: {error}"));
            *owner = Some(ran);
            now.set(now.get() + 1_000_000);
            while let Poll::Ready(publication) = receiver.poll_next(&mut Context::from_waker(Waker::noop())) {
                match publication.expect("a request ends with a terminal publication") {
                    Publication::Output(_) => {}
                    terminal => return terminal,
                }
            }
        };
        let admit = |owner: &mut Option<Owner<magnitude_executor::NativeFamily>>, method| {
            owner.as_mut().unwrap().admit(fresh_by(vocabulary, method), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, now.get()).unwrap().1
        };
        let mut plain = admit(&mut owner, MethodChoice::Plain);
        assert!(matches!(serve(&mut owner, &mut plain), Publication::Completed { .. }), "the first request is served");
        {
            let Owner { service, pipeline: Pipeline::Idle(bindings) } = owner.as_mut().unwrap() else { panic!("an idle owner") };
            // With nothing resident, the drafter is released, as the ledger takes it.
            service.domain.release_idle_optional_components(bindings, &[magnitude_executor::OptionalComponent::Drafter]).unwrap();
            take_room(service);
        }
        let mut drafting = admit(&mut owner, MethodChoice::Mtp { proposals: 2 });
        assert!(
            matches!(serve(&mut owner, &mut drafting), Publication::Failed(RequestError::Capacity(_))),
            "the drafting request fails for memory on its own"
        );
        owner.as_mut().unwrap().service.domain.release_measurement_room();
        let mut next = admit(&mut owner, MethodChoice::Plain);
        assert!(matches!(serve(&mut owner, &mut next), Publication::Completed { .. }), "the owner serves the next request");
        drop(owner);
        std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
    }

    /// Two loads in one process, each with its own owner, admit in turn:
    /// no two of their requests share an identity, and the identities
    /// order their admissions across both, as the store's shared ledger
    /// keys and orders sequences by them.
    #[test]
    fn two_loads_never_share_a_request_identity() {
        let vocabulary = fixtures::tiny_definition().decoder.vocabulary as usize;
        let mut owners = [fixtures::fixture(None), fixtures::fixture(None)]
            .map(|(domain, bindings)| Owner::new(domain, bindings, limits(), Arc::new(NoWakes)).unwrap());
        let mut admitted = Vec::new();
        for owner in [0, 1, 0, 1] {
            let (id, receiver) = owners[owner]
                .admit(fresh(vocabulary), PreparedModelInput::continuation_only(), PrefixRetention::Transient, 4, 0)
                .unwrap();
            admitted.push((id, receiver));
        }
        let ids = admitted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "unique and in admission order across loads: {ids:?}");
    }
}
