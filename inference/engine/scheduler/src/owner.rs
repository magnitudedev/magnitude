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
        MIN_BRANCH_GAIN, MIN_PREFIX_HIT, PrefixCache, PrefixCacheCapacity, PrefixPath,
        PrefixRetention,
    },
    protocol::{RequestSnapshot, WorkerReply},
    publication::{
        CapacityResource, ModelUnloadCause, PhysicalTimings, PublicationPermit, PublicationQueue,
        PublicationReceiver, PublicationSender, PublicationWake, PublicationWakeKind, PublishError,
        RequestError, ServiceCapacityError,
    },
    round_driver::{RoundError, lower_round, reconcile_forward},
    worker::Wakes,
};
use magnitude_executor::{
    DomainError, HeadPhase, InvariantError, MemoryChargeReconciliation,
    NativeFamily,
    OpenRequirements, Operation, Outcome, PendingOperationOutcome, PhysicalDecision,
    ProgramFamily, RequestId, ResourceKind, ResourcePlan, StateBindings, SubmitError,
    SubmitFailure, WorkKind, platform::DomainRole,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{FinishReason, Generation, OutputToken, RoundStart, StartedRound};
use magnitude_state::ShrinkPolicy;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
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

/// Whether `generation` is resident, unfinished, and not yet past `position`.
fn prefilling_toward(generation: &Generation, position: usize) -> bool {
    generation.is_resident()
        && generation.finish_reason().is_none()
        && generation.resident_position() < position
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

impl Record {
    fn add_physical_duration(&mut self, kind: WorkKind, duration: Duration) {
        let nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        let bucket = match kind {
            WorkKind::Prefill | WorkKind::Replay => &mut self.physical_timings.prompt_ns,
            WorkKind::Decode | WorkKind::Verify => &mut self.physical_timings.predicted_ns,
        };
        *bucket = bucket.saturating_add(nanos);
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
        let generation = match self {
            Self::Between(generation, _) => generation,
            Self::Parked(round, ..) => round.cancel(),
        };
        let mut generation = generation;
        if cancel {
            generation.cancel();
        } else {
            generation.fail();
        }
        generation
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
        let mut generation = match self {
            Self::Generation(generation) => generation,
            Self::Started(round) => round.cancel(),
        };
        if cancel {
            generation.cancel();
        } else {
            generation.fail();
        }
        generation
    }
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
        DomainError::Reclaim => RequestError::Invariant(InvariantError {
            context: "memory reclaim",
            detail: DomainError::Reclaim.to_string(),
        }),
        DomainError::Input(error) => RequestError::Input(error),
        DomainError::State(error) => RequestError::State(error),
        DomainError::Device(error) | DomainError::Submit(SubmitError::Device(error)) => {
            RequestError::Device(error)
        }
        DomainError::Invariant(error) | DomainError::Submit(SubmitError::Invariant(error)) => {
            RequestError::Invariant(error)
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Runnable,
    AwaitingCompletion,
    OutputBlocked,
    Preempted,
    CapacityBlocked { required: u64, available: u64 },
    Terminal(FinishReason),
}

/// A request refused before it acquires accepted numerical state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// A required device-memory observation failed. The caller may retry
    /// after the platform can report availability again.
    MemoryObservationUnavailable(String),
    /// A domain the model uses has headroom at or below its planning
    /// reserve; admission pauses while memory is reclaimed. Retryable.
    MemoryReclaim,
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
            Self::MemoryReclaim => formatter.write_str(
                "memory headroom is at or below the planning reserve; reclaiming memory",
            ),
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
            DomainError::Reclaim => Self::MemoryReclaim,
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

/// How often the owner observes memory without another reason to run.
const OBSERVATION_INTERVAL_NS: u64 = 100_000_000;

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
    Idle(StateBindings<F>),
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
    memory_condition: MemoryCondition,
    /// Per domain role, the least memory a capacity-blocked request still
    /// needs from that domain's ceiling after every release was exhausted.
    memory_deficits: Vec<MemoryDeficit>,
    /// Reclaim was observed while a flight held the bindings; its releases
    /// run once they return.
    release_owed: bool,
    next_observation: u64,
    wakes: Arc<dyn Wakes>,
}

pub struct Owner<F: ProgramFamily = NativeFamily> {
    service: Service<F>,
    pipeline: Pipeline<F>,
}

impl<F: ProgramFamily> Owner<F> {
    pub fn new(
        domain: ExecutorDomain<F>,
        bindings: StateBindings<F>,
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
        bindings: StateBindings<F>,
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
                memory_condition: MemoryCondition::Normal,
                memory_deficits: Vec::new(),
                release_owed: false,
                next_observation: 0,
                wakes,
            },
            pipeline: Pipeline::Idle(bindings),
        })
    }

    pub fn with_resource_plan(
        domain: ExecutorDomain<F>,
        bindings: StateBindings<F>,
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
                .refresh_memory()
                .map_err(|error| error.to_string())?,
            reconciliation: self.reconcile_memory_charge()?,
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
        generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
        now: u64,
        members: Option<&mut BTreeMap<RequestId, Member>>,
    ) -> Result<(RequestId, PublicationReceiver), AdmissionError> {
        self.time(now).map_err(AdmissionError::Refused)?;
        if matches!(self.memory_condition, MemoryCondition::Reclaim { .. }) {
            return Err(AdmissionError::MemoryReclaim);
        }
        if generation.is_resident() || generation.usage().completion_tokens != 0 {
            return Err(AdmissionError::invariant("admission")(
                "only a fresh generation can be admitted".into(),
            ));
        }
        let path = PrefixPath::of(&generation);
        let (prefix_cache, cache_points) = match retention {
            PrefixRetention::Transient => (false, Vec::new()),
            PrefixRetention::Retain { cache_points } => (true, cache_points),
        };
        if cache_points
            .iter()
            .any(|&point| point >= generation.prompt().len() || !path.exact_boundary(point))
        {
            return Err(AdmissionError::invariant("admission")(
                "a cache point must be an exact boundary inside the prompt".into(),
            ));
        }
        self.domain.probe_memory().map_err(AdmissionError::from)?;
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
                plan_branch(&mut branches, point, resumed, path.len());
            }
        }
        let cached = self.prefix_cache.shared_boundary(path);
        let mut deepest = cached;
        // The deepest shared prefix a live peer is still prefilling toward
        // and will cache at its branch point.
        let mut in_flight = 0;
        let live = self
            .requests
            .values_mut()
            .map(|request| (&mut request.record, request.state.generation()))
            .chain(
                members
                    .into_iter()
                    .flat_map(|members| members.values_mut())
                    .map(|member| (&mut member.record, member.state.generation())),
            )
            .filter(|(record, _)| record.prefix_cache);
        for (record, generation) in live {
            let shared = path.shared_with_prompt_of(PrefixPath::of(generation));
            deepest = deepest.max(shared);
            plan_branch(
                &mut record.branches,
                shared,
                generation.resident_position(),
                generation.prompt().len(),
            );
            if record.branches.contains(&shared) && prefilling_toward(generation, shared) {
                in_flight = in_flight.max(shared);
            }
        }
        if !worthwhile(deepest, resumed, path.len()) {
            return BranchPlan {
                branches,
                awaited: None,
            };
        }
        if in_flight == deepest && deepest > cached {
            // A peer computes this prefix now: wait for it to be cached.
            return BranchPlan {
                branches,
                awaited: Some(deepest),
            };
        }
        plan_branch(&mut branches, deepest, resumed, path.len());
        BranchPlan {
            branches,
            awaited: None,
        }
    }

    /// Whether a live request other than `except` is still prefilling toward
    /// a planned branch point at `position`.
    fn prefix_pending(&self, position: usize, except: RequestId) -> bool {
        self.requests.iter().any(|(&id, request)| {
            id != except
                && request.record.prefix_cache
                && request.record.branches.contains(&position)
                && prefilling_toward(request.state.generation(), position)
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
            Some(Wait::Memory) => return Status::Preempted,
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
        loop {
            pipeline = match pipeline {
                Pipeline::InFlight(round, mut active) => {
                    if !active.flight.completion().is_complete() {
                        unload |= self.observe(None).map_err(|error| (None, error))?;
                        return Ok((
                            Pipeline::InFlight(round, active),
                            self.ran(true, unload),
                        ));
                    }
                    let (round, bindings) = self.reconcile(round, active)?;
                    self.advance(round, bindings)?
                }
                Pipeline::Idle(mut bindings) => {
                    unload |= self
                        .observe(Some(&mut bindings))
                        .map_err(|error| (None, error))?;
                    self.publish_ready().map_err(|error| (None, error))?;
                    match self.select(&mut bindings)? {
                        Selected::Round(round) => self.advance(round, bindings)?,
                        Selected::Retry => Pipeline::Idle(bindings),
                        Selected::Empty => {
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
            due: self.next_observation,
            in_flight,
            unload,
        }
    }

    /// Observe memory when due. Reclaim releases need the bindings; while a
    /// flight holds them they are owed until it completes. Returns whether
    /// memory escalation decided to unload, after terminating every request.
    fn observe(&mut self, bindings: Option<&mut StateBindings<F>>) -> Result<bool, RequestError> {
        let memory = |error: String| invariant("memory observation")(error);
        if self.now >= self.next_observation {
            self.next_observation = self.now.saturating_add(OBSERVATION_INTERVAL_NS);
            let observation = memory_observation(self.domain.probe_memory()).map_err(memory)?;
            self.enter_memory_condition(self.memory_condition.observe(observation, self.now));
            if !matches!(self.memory_condition, MemoryCondition::Reclaim { .. }) {
                self.release_owed = false;
                if observation == MemoryObservation::Normal {
                    self.reopen_memory_deficits().map_err(memory)?;
                }
            } else if observation == MemoryObservation::Reclaim {
                // Releases are measured against a fresh reading; a Blind
                // escalation has none, so it goes straight to the unload
                // decision.
                self.release_owed = true;
            }
        }
        if let (true, Some(bindings)) = (self.release_owed, bindings) {
            self.release_owed = false;
            if self.release_for_reclaim(bindings)? {
                self.memory_condition = self.memory_condition.released(self.now);
                let after = memory_observation(self.domain.probe_memory()).map_err(memory)?;
                self.enter_memory_condition(self.memory_condition.observe(after, self.now));
            }
        }
        if self.memory_condition.should_unload(self.now) {
            self.terminalize_all(
                RequestError::ModelUnloaded {
                    cause: ModelUnloadCause::MemoryPressure,
                },
                None,
            );
            return Ok(true);
        }
        Ok(false)
    }

    /// Enter `next`. Memory returning to Normal is an availability change,
    /// and it ends every memory wait.
    fn enter_memory_condition(&mut self, next: MemoryCondition) {
        let recovered =
            self.memory_condition != MemoryCondition::Normal && next == MemoryCondition::Normal;
        self.memory_condition = next;
        if recovered {
            self.epoch.advance();
        }
    }

    /// A request blocked on a memory deficit waits for an availability
    /// change. Memory another process frees is one: advance the epoch once a
    /// deficit's domain ceiling covers it, so blocked work retries.
    fn reopen_memory_deficits(&mut self) -> Result<(), String> {
        if self.memory_deficits.is_empty() {
            return Ok(());
        }
        let readings = match self.domain.refresh_memory() {
            Ok(readings) => readings,
            // A failed reading says nothing about the deficit; the next
            // periodic observation looks again.
            Err(DomainError::Blind(_)) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        let before = self.memory_deficits.len();
        self.memory_deficits.retain(|deficit| {
            !readings.iter().any(|reading| {
                reading.role == deficit.role && reading.ceiling_bytes >= deficit.required
            })
        });
        if self.memory_deficits.len() != before {
            self.epoch.advance();
        }
        Ok(())
    }

    /// Record that blocked work needs `required` bytes from `role`'s domain,
    /// keeping the least requirement per role so the earliest retry is kept.
    fn record_memory_deficit(&mut self, resource: ResourceKind, required: u64) {
        let role = match resource {
            ResourceKind::DeviceMemory => DomainRole::Allocation,
            ResourceKind::HostStaging => DomainRole::Staging,
            _ => return,
        };
        match self
            .memory_deficits
            .iter_mut()
            .find(|deficit| deficit.role == role)
        {
            Some(deficit) => deficit.required = deficit.required.min(required),
            None => self.memory_deficits.push(MemoryDeficit { role, required }),
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

    /// Whether a fresh reading is still in the Reclaim band. A failed reading
    /// cannot measure a release, so it stops releasing.
    fn reclaim_needed(&mut self) -> Result<bool, RequestError> {
        Ok(memory_observation(self.domain.probe_memory())
            .map_err(invariant("memory observation"))?
            == MemoryObservation::Reclaim)
    }

    /// Release in the Reclaim order until every domain is back above its
    /// planning reserve: surplus state backing, retained prefixes (least
    /// recently used first), dormant optional components, then preemption of
    /// each eligible victim once. Preempted requests replay only after the
    /// band is Normal again, so no victim is chosen twice. Returns whether
    /// anything was released.
    fn release_for_reclaim(&mut self, bindings: &mut StateBindings<F>) -> Result<bool, RequestError> {
        let mut released = self.release_surplus(bindings)?;
        while self.reclaim_needed()? {
            if !self.prefix_cache.evict_one() {
                break;
            }
            // Eviction drops claims, but committed rows and banks remain
            // charged until the stores shrink.
            self.release_surplus(bindings)?;
            released = true;
        }
        if self.reclaim_needed()? {
            released |= self
                .domain
                .release_idle_optional_components(bindings)
                .map_err(invariant("optional component release"))?
                != 0;
        }
        while self.reclaim_needed()? {
            if !self.evict_victims(&[], false)? {
                break;
            }
            self.release_surplus(bindings)?;
            released = true;
        }
        // Preemption can leave the optional components dormant.
        if self.reclaim_needed()? {
            released |= self
                .domain
                .release_idle_optional_components(bindings)
                .map_err(invariant("optional component release"))?
                != 0;
        }
        if released {
            self.epoch.advance();
        }
        Ok(released)
    }

    /// Release state backing beyond what the stores need, including idle
    /// state. Returns whether any bytes were released.
    fn release_surplus(&mut self, bindings: &mut StateBindings<F>) -> Result<bool, RequestError> {
        let shrunk = self
            .domain
            .shrink_state(bindings, ShrinkPolicy::Reclaim)
            .map_err(classify_domain_error)?;
        let idle = self
            .domain
            .reclaim_idle(bindings)
            .map_err(invariant("idle state release"))?;
        Ok(shrunk != 0 || idle != 0)
    }

    /// When nothing can move the epoch (no flight: this runs in `Idle`, no
    /// request waits on its host or on memory, and memory is Normal), every
    /// capacity wait ends in its typed error. Returns whether any did.
    fn fail_capacity_waiters(&mut self) -> bool {
        if self.memory_condition != MemoryCondition::Normal {
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
        let memory_normal = self.memory_condition == MemoryCondition::Normal;
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

    fn select(&mut self, bindings: &mut StateBindings<F>) -> Result<Selected, Fatal> {
        let mut candidates = self.candidates();
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
                self.domain
                    .shrink_state(bindings, ShrinkPolicy::Idle)
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
        let mut attempted = 0;
        let mut members = BTreeMap::new();
        let mut operations = Vec::new();
        for &id in selection.requests() {
            if remaining_rows == 0 {
                break;
            }
            // Share the remaining rows across requests still eligible in this
            // round. A lone request can use the whole budget, while peers get
            // compatible chunks instead of the first request consuming it.
            let allowance = remaining_rows.div_ceil(selection.requests().len() - attempted);
            attempted += 1;
            let (record, candidate) = candidates.remove(&id).expect("selected candidate");
            match self.start(id, record, candidate, allowance, bindings)? {
                Start::Joined(member, mut next) => {
                    let rows = next
                        .iter()
                        .filter(|operation| {
                            matches!(operation, Operation::Forward { .. } | Operation::Head { .. })
                        })
                        .map(Operation::row_count)
                        .sum::<usize>();
                    remaining_rows = remaining_rows.saturating_sub(rows);
                    operations.append(&mut next);
                    members.insert(id, member);
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
        bindings: &mut StateBindings<F>,
    ) -> Result<Start, Fatal> {
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
                match self.make_resident(id, &mut record, generation, bindings)? {
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
        self.start_round(id, record, generation, allowance)
            .map_err(|error| (None, error))
    }

    fn start_round(
        &mut self,
        id: RequestId,
        mut record: Record,
        generation: Generation,
        allowance: usize,
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
        bindings: &mut StateBindings<F>,
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
            Err(DomainError::Reclaim | DomainError::Blind(_)) => {
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
        record.prefix_source = hit.map(|hit| hit.id());
        Ok(Residency::Resident(generation, encodes))
    }

    fn ensure_open_capacity(&mut self, bindings: &mut StateBindings<F>) -> Result<(), DomainError> {
        let release = |context: &'static str| {
            move |detail: String| DomainError::Invariant(InvariantError { context, detail })
        };
        let requirements = self.domain.open_requirements();
        let mut current = self.open_capacity(bindings, requirements);
        if !matches!(current, Err(DomainError::Capacity(_))) {
            return current;
        }
        self.domain.shrink_state(bindings, ShrinkPolicy::Reclaim)?;
        current = self.open_capacity(bindings, requirements);
        if matches!(current, Err(DomainError::Capacity(_))) {
            self.domain
                .reclaim_idle(bindings)
                .map_err(release("idle state release"))?;
            current = self.open_capacity(bindings, requirements);
        }
        while matches!(current, Err(DomainError::Capacity(_))) {
            if !self.prefix_cache.evict_one() {
                break;
            }
            self.domain.shrink_state(bindings, ShrinkPolicy::Reclaim)?;
            current = self.open_capacity(bindings, requirements);
        }
        if matches!(current, Err(DomainError::Capacity(_))) {
            self.domain
                .release_idle_optional_components(bindings)
                .map_err(release("optional component release"))?;
            current = self.open_capacity(bindings, requirements);
        }
        while matches!(current, Err(DomainError::Capacity(_))) {
            if !self
                .evict_victims(&[], true)
                .map_err(|error| release("preemption")(error.to_string()))?
            {
                break;
            }
            self.domain.shrink_state(bindings, ShrinkPolicy::Reclaim)?;
            current = self.open_capacity(bindings, requirements);
        }
        current
    }

    fn open_capacity(
        &mut self,
        bindings: &mut StateBindings<F>,
        requirements: OpenRequirements,
    ) -> Result<(), DomainError> {
        self.domain.provision_open(bindings)?;
        self.domain
            .can_open(requirements)
            .map_err(DomainError::from)
    }

    /// Submit the round's next groups until one is on the device or the
    /// round is exhausted.
    fn advance(&mut self, mut round: Round, mut bindings: StateBindings<F>) -> Result<Pipeline<F>, Fatal> {
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
            let planned = match self.planned_prefill(&round, group_.operations()) {
                Ok(planned) => planned,
                Err(error) => return Err((Some(round.members), error)),
            };
            self.domain.plan_successors(planned);
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
                        DomainError::Blind(_) | DomainError::Reclaim => {
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

    /// The prompt chunk a lone prefill group's request submits next, when it
    /// is known now, so the domain queues it behind this group. A chunk
    /// ending at a planned branch or retention point is followed by that
    /// retention, not planned past.
    fn planned_prefill(&self, round: &Round, operations: &[Operation]) -> Result<Vec<Operation>, RequestError> {
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
        let allowance = stop
            .map(|stop| stop - end)
            .into_iter()
            .fold(self.scheduler.limits().prefill_tokens.max(1), usize::min);
        Ok(started
            .planned_prefill(*request, allowance)
            .map_err(invariant("planned prefill"))?
            .into_iter()
            .collect())
    }

    /// Provision a group's state and check its reservation. A requirement
    /// and a growth claim are one capacity question: recheck both after each
    /// release. Selection is provisional until every exact requirement is
    /// available: a deficit first releases state backing beyond what the
    /// stores need, then least-recently-used retention until the
    /// requirement fits, then idle residency, then the group's trailing
    /// operations, then eligible live victims outside the round.
    fn provision(
        &mut self,
        bindings: &mut StateBindings<F>,
        round: &Round,
        group_: OperationGroup,
    ) -> Result<Provisioned, RequestError> {
        let short = |current: &Result<(), DomainError>| matches!(current, Err(DomainError::Capacity(_)));
        let mut group_ = group_;
        let mut current = self.fits(bindings, &group_);
        if short(&current) {
            self.domain
                .shrink_state(bindings, ShrinkPolicy::Reclaim)
                .map_err(classify_domain_error)?;
            current = self.fits(bindings, &group_);
        }
        if short(&current) {
            self.domain
                .reclaim_idle(bindings)
                .map_err(invariant("idle state release"))?;
            current = self.fits(bindings, &group_);
        }
        while short(&current) && self.prefix_cache.evict_one() {
            self.domain
                .shrink_state(bindings, ShrinkPolicy::Reclaim)
                .map_err(classify_domain_error)?;
            current = self.fits(bindings, &group_);
        }
        if short(&current) {
            self.domain
                .release_idle_optional_components(bindings)
                .map_err(invariant("optional component release"))?;
            current = self.fits(bindings, &group_);
        }
        if short(&current) {
            group_ = match group_.split_last() {
                Ok((leading, trailing)) => return Ok(Provisioned::Split(leading, trailing)),
                Err(whole) => whole,
            };
            let selected = round.members.keys().copied().collect::<Vec<_>>();
            while short(&current) && self.evict_victims(&selected, false)? {
                self.domain
                    .shrink_state(bindings, ShrinkPolicy::Reclaim)
                    .map_err(classify_domain_error)?;
                current = self.fits(bindings, &group_);
            }
        }
        match current {
            Ok(()) => Ok(Provisioned::Fits(group_)),
            Err(DomainError::Blind(_) | DomainError::Reclaim) => {
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
    /// into, would take.
    fn fits(&mut self, bindings: &mut StateBindings<F>, group_: &OperationGroup) -> Result<(), DomainError> {
        self.domain.provision(bindings, group_.operations())?;
        let requirement = requirements(&self.domain, bindings, group_)?;
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
    fn reconcile(&mut self, mut round: Round, active: ActiveGroup<F>) -> Result<(Round, StateBindings<F>), Fatal> {
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
        let (outcomes, mut bindings) = match finish(&mut self.domain, flight) {
            Ok(finished) => finished,
            Err(error) => return Err((Some(round.members), classify_domain_error(error))),
        };
        let reconciled = match outcomes {
            FlightOutcomes::Target(pending) => self.reconcile_target(&mut round, pending, &operations),
            FlightOutcomes::Head(pending) => self.reconcile_method(&mut round, pending, &operations),
            FlightOutcomes::Vision(item) => self.reconcile_vision(&mut round, item, &operations),
        };
        if let Err(error) = reconciled {
            return Err((Some(round.members), error));
        }
        self.epoch.advance();
        if let Err(error) = self.observe(Some(&mut bindings)) {
            return Err((Some(round.members), error));
        }
        self.publish_members(&mut round);
        Ok((round, bindings))
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
            record.add_physical_duration(item.kind(), item.physical_duration());
            let state = match state {
                Membership::Started(started) if !ended(started.generation()) => {
                    match reconcile_forward(started, &mut self.domain, request, item) {
                        Ok((generation, effects)) => {
                            followups.extend(effects.operations);
                            Membership::Generation(generation)
                        }
                        Err((mut generation, RoundError::Logical(detail))) => {
                            Self::fail_generation(
                                &mut record,
                                &mut generation,
                                invariant("generation round")(detail),
                            );
                            Membership::Generation(generation)
                        }
                        Err((_, RoundError::Physical(error))) if is_fatal(&error) => {
                            return Err(classify_domain_error(error));
                        }
                        Err((mut generation, RoundError::Physical(error))) => {
                            Self::fail_generation(
                                &mut record,
                                &mut generation,
                                classify_domain_error(error),
                            );
                            Membership::Generation(generation)
                        }
                    }
                }
                state => {
                    self.domain.abort(item).map_err(classify_domain_error)?;
                    state
                }
            };
            round.members.insert(request, Member { record, state });
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
        for item in pending {
            let request = item.request();
            let operation = expected[&request];
            let correct = matches!(operation, Operation::Head { .. })
                && matches!(item.outcome(), Outcome::Head { .. });
            let Some(member) = round.members.get_mut(&request) else {
                self.domain.abort(item).map_err(classify_domain_error)?;
                continue;
            };
            let kind = match operation {
                Operation::Head {
                    phase: HeadPhase::Priming { .. },
                    ..
                } => WorkKind::Prefill,
                _ => item.kind(),
            };
            member.record.add_physical_duration(kind, item.physical_duration());
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
            .add_physical_duration(item.kind(), item.physical_duration());
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

    /// A request outside the pipeline between rounds: its record and
    /// generation.
    fn between(&self, id: RequestId) -> Option<(&Record, &Generation)> {
        match self.requests.get(&id) {
            Some(Request {
                record,
                state: State::Between(generation, _),
            }) => Some((record, generation)),
            _ => None,
        }
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
            self.cache_prefix(id, resident)?;
        }
        Ok(())
    }

    /// Cache the request's reconciled state as the prefix of its path ending
    /// at `position`, where it is resident.
    fn cache_prefix(&mut self, id: RequestId, position: usize) -> Result<(), RequestError> {
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
        self.prefix_cache
            .retain(PrefixPath::of(generation), position, state, method)
            .map_err(invariant("prefix retention"))?;
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
        self.cache_prefix(id, boundary)?;
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
        let end = PrefixPath::of(generation).len();
        if generation.resident_position() != end {
            if generation.pending_reconciliation().is_some() {
                return Ok(());
            }
            return Err(invariant("prefix retention")(
                "terminal numerical state is not reconciled at the accepted boundary".into(),
            ));
        }
        self.cache_prefix(id, end)?;
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
            let State::Between(generation, _) = &request.state else {
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
    ) -> Result<bool, RequestError> {
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
                .map_err(invariant("preemption"))?;
            victims.push(Victim {
                identity: id,
                output_blocked: matches!(stage, Stage::AwaitingCredit),
                preemption_debt: request.record.preemption_debt,
                exclusive_bytes: executor_bytes
                    .checked_add(generation.method_reclaimable())
                    .ok_or_else(|| invariant("preemption")("request reclaim byte count overflow".into()))?,
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
                    .ok_or_else(|| invariant("preemption")("method reclaim byte count overflow".into()))
            })?;
            let reclaimable = self
                .domain
                .reclaimable(&set)
                .map_err(invariant("preemption"))?;
            if !release_resident_slot && reclaimable == 0 && method_bytes == 0 {
                continue;
            }
            let released = self
                .domain
                .release_state(&set)
                .map_err(invariant("preemption"))?;
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
                generation.evicted().map_err(invariant("preemption"))?;
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
        Err(DomainError::Reclaim) => Ok(MemoryObservation::Reclaim),
        Err(DomainError::Blind(_)) => Ok(MemoryObservation::Blind),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod branch_plan_tests {
    use super::{MIN_BRANCH_GAIN, MIN_PREFIX_HIT, plan_branch};
    use std::collections::BTreeSet;

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
        // The declared start of the last message, then a divergence just
        // beyond it and one far beyond it.
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
