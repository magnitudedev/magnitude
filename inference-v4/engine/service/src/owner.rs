//! Serialized ownership of logical generation and one executor resource domain.
use super::{
    domain::{
        group, requirements, submit_group, DomainCheckpoint, DomainFlight, ExecutorDomain,
        OperationGroup,
    },
    policy::{
        order_victims, AvailabilityEpoch, Operation as ScheduledOperation, Phase, Scheduler,
        Selection, ServiceLimits, Victim,
    },
    publication::{
        CapacityResource, ModelUnloadCause, PhysicalTimings, PublicationPermit, PublicationSender,
        PublicationWake, PublicationWakeKind, PublishError, RequestError, ServiceCapacityError,
    },
    retention::{
        Retention, RetentionCapacity, RetentionKey, RetentionRequest, MIN_BRANCH_GAIN,
        MIN_RETENTION_HIT,
    },
    round_driver::{lower_round, reconcile_forward, RoundError},
};
use magnitude_generation::{
    DetailedUsage, FinishReason, Generation, OutputToken, RoundStart, WaitReason,
};
use magnitude_model_executor::{
    DomainError, DomainRequirements, InvariantError, NativeFamily, OpenRequirements, Operation,
    Outcome, PhysicalDecision, PressureLevel, ProgramFamily, RequestId, ResourceKind, ResourcePlan,
    SubmitError, WorkKind,
};
use magnitude_model_state::ShrinkPolicy;
use std::{
    collections::{BTreeMap, VecDeque},
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointId(u64);

/// Where a newly admitted request stops its prefill to retain a branch
/// point, or the shared prefix it waits for a live peer to retain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BranchPlan {
    branch: Option<usize>,
    awaited: Option<usize>,
}

/// Where a request's prompt is retained: one row before the prompt end, the
/// deepest point a request with the same prompt can resume from (it must
/// compute the row it samples from). None when that is not an exact input
/// boundary, too short to be a hit, or already retained.
fn prompt_boundary(record: &Record) -> Option<usize> {
    let input = record.retention.as_ref()?;
    let boundary = record.generation.prompt().len().checked_sub(1)?;
    (!record.prefill_retained && boundary >= MIN_RETENTION_HIT && input.exact_boundary(boundary))
        .then_some(boundary)
}

/// Whether `generation` is resident, unfinished, and not yet past `position`.
fn prefilling_toward(generation: &Generation, position: usize) -> bool {
    generation.is_resident()
        && generation.finish_reason().is_none()
        && generation.resident_position() < position
}

struct Checkpoint {
    generation: Generation,
    numerical: DomainCheckpoint,
    error: Option<RequestError>,
    protected_until: Option<usize>,
}

struct Record {
    generation: Generation,
    admission: VecDeque<Operation>,
    waiting_since: u64,
    service_ns: u64,
    physical_timings: PhysicalTimings,
    error: Option<RequestError>,
    capacity: Option<(AvailabilityEpoch, u64, u64)>,
    preemption_debt: u32,
    protected_until: Option<usize>,
    retire_on_completion: bool,
    retention: Option<RetentionRequest>,
    retention_source: Option<u64>,
    prefill_retained: bool,
    terminal_retained: bool,
    /// A planned branch point: the prompt position where this request
    /// diverges from a path the engine holds. Prefill stops there so the
    /// index can retain a checkpoint every later divergent request resumes
    /// from.
    branch: Option<usize>,
    /// A prompt position a live peer is prefilling toward and will retain:
    /// this request is not scheduled until it attaches to that checkpoint
    /// (or no peer computes it any more), so the prefix is computed once.
    awaited_prefix: Option<usize>,
    publication: Option<PublicationSender>,
    publication_batch_limit: usize,
    pending_publication: Option<Vec<OutputToken>>,
    publication_permits: VecDeque<PublicationPermit>,
    publication_blocked: bool,
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
        DomainError::Pressure(level) => RequestError::Invariant(InvariantError {
            context: "platform memory pressure",
            detail: format!("{level:?}"),
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
    MemoryPressure(PressureLevel),
    ModelUnloaded {
        cause: ModelUnloadCause,
    },
    Other(String),
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
            Self::MemoryPressure(level) => {
                write!(formatter, "platform memory pressure is {level:?}")
            }
            Self::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            } => formatter.write_str("model unloaded due to memory pressure"),
            Self::Other(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for AdmissionError {}

impl From<String> for AdmissionError {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

impl From<&str> for AdmissionError {
    fn from(message: &str) -> Self {
        Self::Other(message.into())
    }
}

impl From<DomainError> for AdmissionError {
    fn from(error: DomainError) -> Self {
        match error {
            DomainError::Blind(message) => Self::MemoryObservationUnavailable(message),
            DomainError::Pressure(level) => Self::MemoryPressure(level),
            other => Self::Other(other.to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Submitted,
    Reconciled,
    Waiting,
    Idle,
    Progress,
}

const MEMORY_ESCALATION_NS: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryObservation {
    Normal,
    Pressure,
    Emergency,
    Blind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryCondition {
    Normal,
    Pressure,
    Blind { since: u64 },
    Emergency { since: u64 },
    Unloading,
}

impl MemoryCondition {
    fn observe(self, observation: MemoryObservation, now: u64) -> Self {
        if self == Self::Unloading {
            return self;
        }
        match observation {
            MemoryObservation::Normal => Self::Normal,
            MemoryObservation::Pressure => Self::Pressure,
            MemoryObservation::Emergency => match self {
                Self::Emergency { since } => Self::Emergency { since },
                _ => Self::Emergency { since: now },
            },
            MemoryObservation::Blind => match self {
                Self::Blind { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS => {
                    Self::Emergency { since: now }
                }
                Self::Blind { since } => Self::Blind { since },
                Self::Emergency { since } => Self::Emergency { since },
                _ => Self::Blind { since: now },
            },
        }
    }

    fn should_unload(self, now: u64) -> bool {
        matches!(self, Self::Emergency { since } if now.saturating_sub(since) >= MEMORY_ESCALATION_NS)
    }
}

struct ActiveGroup<F: ProgramFamily> {
    flight: DomainFlight<F>,
    operations: Vec<Operation>,
    retention_uses: Vec<u64>,
    publication_permits: BTreeMap<RequestId, Vec<PublicationPermit>>,
}

struct Batch<F: ProgramFamily> {
    selection: Selection,
    started: u64,
    queued: VecDeque<OperationGroup>,
    active: Option<ActiveGroup<F>>,
    blocked: Option<AvailabilityEpoch>,
}

pub struct Owner<F: ProgramFamily = NativeFamily> {
    checkpoints: BTreeMap<CheckpointId, Checkpoint>,
    checkpoint_capacity: usize,
    resident_capacity: usize,
    domain: ExecutorDomain<F>,
    scheduler: Scheduler,
    records: BTreeMap<RequestId, Record>,
    batch: Option<Batch<F>>,
    epoch: AvailabilityEpoch,
    next_id: u64,
    next_checkpoint: u64,
    now: u64,
    fatal: Option<String>,
    retention: Retention<DomainCheckpoint>,
    pressure_release_pending: bool,
    memory_condition: MemoryCondition,
}

impl<F: ProgramFamily> Owner<F> {
    pub fn new(domain: ExecutorDomain<F>, limits: ServiceLimits) -> Result<Self, String> {
        Self::with_retention_capacity(domain, limits, RetentionCapacity::disabled())
    }

    pub fn with_retention_capacity(
        domain: ExecutorDomain<F>,
        limits: ServiceLimits,
        retention_capacity: RetentionCapacity,
    ) -> Result<Self, String> {
        let checkpoint_capacity = limits.max_requests;
        let resident_capacity = limits.max_requests;
        Self::with_capacities(
            domain,
            limits,
            retention_capacity,
            checkpoint_capacity,
            resident_capacity,
        )
    }

    fn with_capacities(
        domain: ExecutorDomain<F>,
        limits: ServiceLimits,
        retention_capacity: RetentionCapacity,
        checkpoint_capacity: usize,
        resident_capacity: usize,
    ) -> Result<Self, String> {
        if resident_capacity == 0 {
            return Err("service resident capacity must be positive".into());
        }
        Ok(Self {
            checkpoints: BTreeMap::new(),
            checkpoint_capacity,
            resident_capacity,
            domain,
            scheduler: Scheduler::new(limits)?,
            records: BTreeMap::new(),
            batch: None,
            epoch: AvailabilityEpoch::default(),
            next_id: 1,
            next_checkpoint: 1,
            now: 0,
            fatal: None,
            retention: Retention::new(retention_capacity),
            pressure_release_pending: false,
            memory_condition: MemoryCondition::Normal,
        })
    }

    pub fn with_resource_plan(
        domain: ExecutorDomain<F>,
        limits: ServiceLimits,
        plan: &ResourcePlan,
    ) -> Result<Self, String> {
        if limits.max_batch != plan.capacity().active
            || limits.max_batch != plan.capacity().in_flight
        {
            return Err("service numerical concurrency differs from the resource plan".into());
        }
        let checkpoint_capacity = plan
            .capacity()
            .retained
            .checked_sub(plan.capacity().retention_entries)
            .ok_or("resource plan retained capacity is inconsistent")?;
        Self::with_capacities(
            domain,
            limits,
            RetentionCapacity::from_resource_plan(plan),
            checkpoint_capacity,
            plan.capacity().active,
        )
    }

    pub fn request_capacity(&self) -> (usize, usize) {
        (self.records.len(), self.scheduler.limits().max_requests)
    }

    /// Bind the sole host-visible stream before the worker next advances this
    /// request. The queue is created after admission so its reserved wake can
    /// carry the assigned request identity.
    pub fn attach_publication(
        &mut self,
        request: RequestId,
        sender: PublicationSender,
        token_batch_limit: usize,
    ) -> Result<(), String> {
        if token_batch_limit == 0 {
            return Err("publication token batch limit must be positive".into());
        }
        let record = self.records.get_mut(&request).ok_or("unknown request")?;
        if record.publication.is_some() || record.publication_batch_limit != 0 {
            return Err("request already has a publication stream".into());
        }
        record.publication = Some(sender);
        record.publication_batch_limit = token_batch_limit;
        Ok(())
    }

    /// Consume only a reserved wake for this request's exact queue. A stale
    /// credit cannot make a full request schedulable; receiver closure cancels
    /// through the existing physical-completion-aware release path.
    pub fn publication_wake(
        &mut self,
        request: RequestId,
        wake: PublicationWake,
    ) -> Result<(), String> {
        let (credit, cancelled) = {
            let Some(record) = self.records.get_mut(&request) else {
                return Ok(());
            };
            let Some(sender) = record.publication.as_mut() else {
                return Ok(());
            };
            match wake.kind() {
                PublicationWakeKind::OutputCredit => {
                    let credit = sender.process_output_credit(wake);
                    if credit {
                        record.publication_blocked = false;
                    }
                    (credit, false)
                }
                PublicationWakeKind::Cancelled => (false, sender.process_cancelled(wake)),
            }
        };
        if credit {
            self.epoch.advance()?;
        }
        if cancelled {
            self.release(request)?;
        }
        Ok(())
    }

    pub fn inspect_domain<R>(
        &self,
        inspect: impl FnOnce(&ExecutorDomain<F>) -> Result<R, String>,
    ) -> Result<R, String> {
        inspect(&self.domain)
    }

    fn time(&mut self, now: u64) -> Result<(), String> {
        if now < self.now {
            return Err("service clock moved backwards".into());
        }
        self.now = now;
        Ok(())
    }

    fn observe_periodic_memory(&mut self, now: u64) -> Result<(), String> {
        self.time(now)?;
        if self.memory_condition == MemoryCondition::Unloading {
            return Ok(());
        }
        let observation = Self::memory_observation(self.domain.probe_memory())?;
        self.memory_condition = self.memory_condition.observe(observation, now);
        if matches!(
            observation,
            MemoryObservation::Normal | MemoryObservation::Blind
        ) && !matches!(self.memory_condition, MemoryCondition::Emergency { .. })
        {
            self.pressure_release_pending = false;
            return Ok(());
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_some())
        {
            self.pressure_release_pending = true;
            return Ok(());
        }
        self.release_pressure_holdings()?;
        let after = Self::memory_observation(self.domain.probe_memory())?;
        self.memory_condition = self.memory_condition.observe(after, now);
        if self.memory_condition.should_unload(now) {
            self.begin_memory_unload();
        }
        Ok(())
    }

    fn memory_observation(result: Result<(), DomainError>) -> Result<MemoryObservation, String> {
        match result {
            Ok(()) => Ok(MemoryObservation::Normal),
            Err(DomainError::Pressure(PressureLevel::Normal)) => Ok(MemoryObservation::Normal),
            Err(DomainError::Pressure(PressureLevel::Pressure)) => Ok(MemoryObservation::Pressure),
            Err(DomainError::Pressure(PressureLevel::Emergency)) => {
                Ok(MemoryObservation::Emergency)
            }
            Err(DomainError::Blind(_)) => Ok(MemoryObservation::Blind),
            Err(error) => Err(error.to_string()),
        }
    }

    fn release_pressure_holdings(&mut self) -> Result<(), String> {
        self.relieve_platform_pressure()?;
        while matches!(self.memory_condition, MemoryCondition::Emergency { .. }) {
            if matches!(self.domain.probe_memory(), Ok(())) {
                break;
            }
            if !self.evict_victims(&[], false)? {
                break;
            }
            self.domain
                .shrink_state(ShrinkPolicy::Pressure)
                .map_err(|error| error.to_string())?;
        }
        self.epoch.advance()?;
        Ok(())
    }

    fn release_deferred_pressure(&mut self) -> Result<(), String> {
        if !self.pressure_release_pending {
            return Ok(());
        }
        self.pressure_release_pending = false;
        self.observe_periodic_memory(self.now)
    }

    pub fn admit(&mut self, generation: Generation, now: u64) -> Result<RequestId, AdmissionError> {
        self.admit_with(generation, now, |_, _, _| Ok(Vec::new()))
    }

    pub fn admit_with(
        &mut self,
        generation: Generation,
        now: u64,
        prepare: impl FnOnce(
            &mut ExecutorDomain<F>,
            RequestId,
            &Generation,
        ) -> Result<Vec<Operation>, String>,
    ) -> Result<RequestId, AdmissionError> {
        self.admit_inner(generation, None, now, |domain, request, generation, _| {
            prepare(domain, request, generation)
        })
    }

    pub fn admit_retained_with(
        &mut self,
        generation: Generation,
        retention: RetentionRequest,
        now: u64,
        prepare: impl FnOnce(
            &mut ExecutorDomain<F>,
            RequestId,
            &Generation,
            Option<usize>,
        ) -> Result<Vec<Operation>, String>,
    ) -> Result<RequestId, AdmissionError> {
        self.admit_inner(generation, Some(retention), now, prepare)
    }

    fn admit_inner(
        &mut self,
        mut generation: Generation,
        retention_request: Option<RetentionRequest>,
        now: u64,
        prepare: impl FnOnce(
            &mut ExecutorDomain<F>,
            RequestId,
            &Generation,
            Option<usize>,
        ) -> Result<Vec<Operation>, String>,
    ) -> Result<RequestId, AdmissionError> {
        self.time(now)?;
        if self.memory_condition == MemoryCondition::Unloading {
            return Err(AdmissionError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            });
        }
        if matches!(self.memory_condition, MemoryCondition::Emergency { .. }) {
            return Err(AdmissionError::MemoryPressure(PressureLevel::Emergency));
        }
        if let Some(error) = &self.fatal {
            return Err(error.clone().into());
        }
        if self.records.len() >= self.scheduler.limits().max_requests {
            return Err("service request limit reached".into());
        }
        if generation.awaiting_completion()
            || generation.resident_position() != 0
            || generation.usage().completion_tokens != 0
            || !generation.is_resident()
        {
            return Err("continued generation requires checkpoint admission".into());
        }
        // Observe before any resident-slot eviction. A Blind refusal must
        // leave accepted peers and their numerical holdings intact.
        self.domain.probe_memory().map_err(AdmissionError::from)?;
        self.ensure_resident_slot()?;
        let id = RequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("service identity exhausted")?;
        let hit = retention_request
            .as_ref()
            .map(|request| self.retention.lookup(request, request.tokens.len()))
            .transpose()?
            .flatten();
        let opened = match hit {
            Some(hit) => {
                {
                    let retained = self.retention.entry(hit)?;
                    generation.restore_prefix(hit.position(), retained.method())?;
                }
                let retained = self.retention.entry(hit)?;
                self.domain.open_checkpoint(id, retained.checkpoint())
            }
            None => {
                self.ensure_open_capacity()?;
                self.domain
                    .reserve_open(id)
                    .and_then(|reservation| self.domain.open_reserved(reservation))
            }
        };
        if let Err(error) = opened {
            let error = AdmissionError::from(error);
            if let Some(fatal) = self.domain.fatal_error().cloned() {
                self.fail_domain_error(fatal);
            }
            return Err(error);
        }
        let admission = match prepare(
            &mut self.domain,
            id,
            &generation,
            hit.map(|hit| hit.position()),
        ) {
            Ok(operations) => operations,
            Err(error) => {
                let rollback = self.domain.close(id);
                return Err(match rollback {
                    Ok(()) => error,
                    Err(rollback) => {
                        let fatal = format!(
                            "admission failed ({error}); executor rollback failed ({rollback})"
                        );
                        self.fail_all(fatal.clone());
                        fatal
                    }
                }
                .into());
            }
        };
        if admission.iter().any(|operation| {
            operation.request() != id || !matches!(operation, Operation::Encode { .. })
        }) {
            let rollback = self.domain.close(id);
            return Err(match rollback {
                Ok(()) => "admission returned invalid initial operations".into(),
                Err(rollback) => format!(
                    "admission returned invalid initial operations; rollback failed: {rollback}"
                ),
            }
            .into());
        }
        let mut plan = retention_request
            .as_ref()
            .map(|request| self.plan_branches(request, hit.map_or(0, |hit| hit.position())))
            .unwrap_or_default();
        // Only a text prefix waits; a request with media encodes up front and
        // computes the prefix itself, branching there.
        if !admission.is_empty() {
            if let Some(point) = plan.awaited.take() {
                plan.branch = Some(point);
            }
        }
        self.records.insert(
            id,
            Record {
                generation,
                admission: admission.into(),
                waiting_since: now,
                service_ns: 0,
                physical_timings: PhysicalTimings::default(),
                error: None,
                capacity: None,
                preemption_debt: 0,
                protected_until: None,
                retire_on_completion: false,
                retention: retention_request,
                retention_source: hit.map(|hit| hit.id()),
                prefill_retained: false,
                terminal_retained: false,
                branch: plan.branch,
                awaited_prefix: plan.awaited,
                publication: None,
                publication_batch_limit: 0,
                pending_publication: None,
                publication_permits: VecDeque::new(),
                publication_blocked: false,
            },
        );
        self.epoch.advance()?;
        Ok(id)
    }

    /// Branch points for a request resuming at `resumed` (zero or its
    /// retention hit): the deepest position where it diverges from a
    /// retained path or a live request's prompt. A live request that has not
    /// yet prefilled to its divergence from this one stops there too, so the
    /// shared prefix is retained by whichever request reaches it first and
    /// every later request resumes at it instead of recomputing.
    fn plan_branches(&mut self, request: &RetentionRequest, resumed: usize) -> BranchPlan {
        let worthwhile = |point: usize, from: usize, prompt: usize| {
            point >= MIN_RETENTION_HIT && point >= from + MIN_BRANCH_GAIN && point < prompt
        };
        let retained = self.retention.shared_boundary(request);
        let mut deepest = retained;
        // The deepest shared prefix a live peer is still prefilling toward
        // and will retain at its branch point.
        let mut in_flight = 0;
        for record in self.records.values_mut() {
            let Some(live) = record.retention.as_ref() else {
                continue;
            };
            if live.key != request.key {
                continue;
            }
            let shared = request.shared_boundary(&live.tokens, &live.conditioning);
            deepest = deepest.max(shared);
            if record.branch.is_none()
                && worthwhile(
                    shared,
                    record.generation.resident_position(),
                    live.tokens.len(),
                )
            {
                record.branch = Some(shared);
            }
            if record.branch == Some(shared) && prefilling_toward(&record.generation, shared) {
                in_flight = in_flight.max(shared);
            }
        }
        if !worthwhile(deepest, resumed, request.tokens.len()) {
            return BranchPlan::default();
        }
        if in_flight == deepest && deepest > retained {
            // A peer computes this prefix now: wait for its checkpoint.
            return BranchPlan {
                branch: None,
                awaited: Some(deepest),
            };
        }
        BranchPlan {
            branch: Some(deepest),
            awaited: None,
        }
    }

    /// Whether a live request other than `except` is still prefilling toward
    /// a planned branch point at `position` on `key`'s paths.
    fn prefix_pending(&self, key: &RetentionKey, position: usize, except: RequestId) -> bool {
        self.records.iter().any(|(&id, record)| {
            id != except
                && record.branch == Some(position)
                && record
                    .retention
                    .as_ref()
                    .is_some_and(|live| &live.key == key)
                && prefilling_toward(&record.generation, position)
        })
    }

    /// Requests waiting for a peer's prefix attach to its retained
    /// checkpoint once it exists, and compute the prefix themselves once no
    /// peer is still computing it.
    fn attach_awaited(&mut self) -> Result<(), String> {
        let waiting = self
            .records
            .iter()
            .filter_map(|(&id, record)| record.awaited_prefix.map(|position| (id, position)))
            .collect::<Vec<_>>();
        for (id, awaited) in waiting {
            let input = self.records[&id]
                .retention
                .clone()
                .ok_or("an awaited prefix requires a retention request")?;
            let hit = self
                .retention
                .lookup(&input, input.tokens.len())?
                .filter(|hit| hit.position() >= awaited);
            let Some(hit) = hit else {
                if !self.prefix_pending(&input.key, awaited, id) {
                    let record = self.records.get_mut(&id).expect("waiting request");
                    record.awaited_prefix = None;
                    self.epoch.advance()?;
                }
                continue;
            };
            let retained = self.retention.entry(hit)?;
            if self
                .domain
                .resume_from_checkpoint(id, retained.checkpoint())
                .is_err()
            {
                if let Some(fatal) = self.domain.fatal_error().cloned() {
                    self.fail_domain_error(fatal);
                    return Ok(());
                }
                // The request computes the prefix itself instead.
                self.records
                    .get_mut(&id)
                    .expect("waiting request")
                    .awaited_prefix = None;
                self.epoch.advance()?;
                continue;
            }
            let record = self.records.get_mut(&id).expect("waiting request");
            record
                .generation
                .restore_prefix(hit.position(), retained.method())?;
            record.retention_source = Some(hit.id());
            record.awaited_prefix = None;
            self.epoch.advance()?;
        }
        Ok(())
    }

    fn ensure_resident_slot(&mut self) -> Result<(), String> {
        let resident = self
            .records
            .values()
            .filter(|record| record.generation.is_resident())
            .count();
        if resident < self.resident_capacity {
            return Ok(());
        }
        self.evict_victims(&[], true)?;
        let resident = self
            .records
            .values()
            .filter(|record| record.generation.is_resident())
            .count();
        if resident >= self.resident_capacity {
            return Err("resident numerical request limit reached".into());
        }
        Ok(())
    }

    fn ensure_open_capacity(&mut self) -> Result<(), AdmissionError> {
        let requirements = self.domain.open_requirements();
        let mut current = self.open_capacity(requirements);
        if matches!(current, Err(DomainError::Capacity(_))) {
            self.domain
                .shrink_state(ShrinkPolicy::Pressure)
                .map_err(AdmissionError::from)?;
            current = self.open_capacity(requirements);
            if matches!(current, Err(DomainError::Capacity(_))) {
                self.domain.reclaim_idle()?;
                current = self.open_capacity(requirements);
            }
            while matches!(current, Err(DomainError::Capacity(_))) {
                if self
                    .retention
                    .evict_one(DomainCheckpoint::exclusive_bytes)?
                    .is_none()
                {
                    break;
                }
                self.domain
                    .shrink_state(ShrinkPolicy::Pressure)
                    .map_err(AdmissionError::from)?;
                current = self.open_capacity(requirements);
            }
            if matches!(current, Err(DomainError::Capacity(_))) {
                self.domain.release_idle_optional_components()?;
                current = self.open_capacity(requirements);
            }
            while matches!(current, Err(DomainError::Capacity(_))) {
                if !self.evict_victims(&[], true)? {
                    break;
                }
                self.domain
                    .shrink_state(ShrinkPolicy::Pressure)
                    .map_err(AdmissionError::from)?;
                current = self.open_capacity(requirements);
            }
        }
        current.map_err(AdmissionError::from)
    }

    fn open_capacity(&mut self, requirements: OpenRequirements) -> Result<(), DomainError> {
        self.domain.provision_open()?;
        self.domain
            .can_open(requirements)
            .map_err(DomainError::from)
    }

    pub fn checkpoint(&mut self, request: RequestId) -> Result<CheckpointId, String> {
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.checkpoints.len() >= self.checkpoint_capacity {
            return Err("service checkpoint limit reached".into());
        }
        let record = self.records.get(&request).ok_or("unknown request")?;
        if record.retire_on_completion
            || !record.admission.is_empty()
            || self.batch_contains(request)
        {
            return Err("cannot checkpoint submitted or releasing work".into());
        }
        let numerical = self.domain.checkpoint(request)?;
        let generation = record.generation.fork_at(numerical.position())?;
        let id = CheckpointId(self.next_checkpoint);
        self.next_checkpoint = self
            .next_checkpoint
            .checked_add(1)
            .ok_or("checkpoint identity exhausted")?;
        self.checkpoints.insert(
            id,
            Checkpoint {
                generation,
                numerical,
                error: record.error.clone(),
                protected_until: record.protected_until,
            },
        );
        Ok(id)
    }

    pub fn fork_checkpoint(
        &mut self,
        checkpoint: CheckpointId,
        now: u64,
    ) -> Result<RequestId, String> {
        self.time(now)?;
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.records.len() >= self.scheduler.limits().max_requests {
            return Err("service request limit reached".into());
        }
        self.ensure_resident_slot()?;
        let generation = {
            let checkpoint = self
                .checkpoints
                .get(&checkpoint)
                .ok_or("unknown checkpoint")?;
            checkpoint
                .generation
                .fork_at(checkpoint.numerical.position())?
        };
        let id = RequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("service identity exhausted")?;
        let opened = {
            let checkpoint = self
                .checkpoints
                .get(&checkpoint)
                .ok_or("unknown checkpoint")?;
            self.domain.open_checkpoint(id, &checkpoint.numerical)
        };
        if let Err(error) = opened {
            let error = error.to_string();
            if let Some(fatal) = self.domain.fatal_error().cloned() {
                self.fail_domain_error(fatal);
            }
            return Err(error);
        }
        let checkpoint = self
            .checkpoints
            .get(&checkpoint)
            .ok_or("unknown checkpoint")?;
        self.records.insert(
            id,
            Record {
                generation,
                admission: VecDeque::new(),
                waiting_since: now,
                service_ns: 0,
                physical_timings: PhysicalTimings::default(),
                error: checkpoint.error.clone(),
                capacity: None,
                preemption_debt: 0,
                protected_until: checkpoint.protected_until,
                retire_on_completion: false,
                retention: None,
                retention_source: None,
                prefill_retained: false,
                terminal_retained: false,
                branch: None,
                awaited_prefix: None,
                publication: None,
                publication_batch_limit: 0,
                pending_publication: None,
                publication_permits: VecDeque::new(),
                publication_blocked: false,
            },
        );
        self.epoch.advance()?;
        Ok(id)
    }

    pub fn release_checkpoint(&mut self, checkpoint: CheckpointId) -> Result<(), String> {
        if self.checkpoints.remove(&checkpoint).is_some() {
            self.epoch.advance()?;
        }
        Ok(())
    }

    pub fn error(&self, id: RequestId) -> Option<&RequestError> {
        self.records
            .get(&id)
            .and_then(|record| record.error.as_ref())
    }
    pub fn output_len(&self, id: RequestId) -> Result<usize, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .output_len())
    }
    pub fn usage(&self, id: RequestId) -> Result<DetailedUsage, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .detailed_usage())
    }
    pub fn resident_position(&self, id: RequestId) -> Result<usize, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .resident_position())
    }
    pub fn method_identity(&self, id: RequestId) -> Result<&str, String> {
        Ok(self
            .records
            .get(&id)
            .ok_or("unknown request")?
            .generation
            .method_identity())
    }
    pub fn fatal_error(&self) -> Option<&str> {
        self.fatal.as_deref()
    }
    pub fn completed_service_ns(&self) -> u128 {
        self.scheduler.completed_service_ns()
    }
    pub fn retained_entries(&self) -> usize {
        self.retention.len()
    }
    pub const fn retention_entry_capacity(&self) -> usize {
        self.retention.max_entries()
    }
    pub const fn retained_bytes(&self) -> u64 {
        self.retention.charged_bytes()
    }

    pub fn status(&self, id: RequestId) -> Result<Status, String> {
        let record = self.records.get(&id).ok_or("unknown request")?;
        if record.generation.awaiting_completion() || self.batch_contains(id) {
            return Ok(Status::AwaitingCompletion);
        }
        if let Some(reason) = record.generation.finish_reason() {
            return Ok(Status::Terminal(reason));
        }
        if record.publication_blocked {
            return Ok(Status::OutputBlocked);
        }
        if let Some((epoch, required, available)) = record.capacity {
            if !self.epoch.changed_since(epoch) {
                return Ok(Status::CapacityBlocked {
                    required,
                    available,
                });
            }
        }
        Ok(match record.generation.wait_reason() {
            None => Status::Runnable,
            Some(WaitReason::Output) => Status::OutputBlocked,
            Some(WaitReason::Residency) => Status::Preempted,
            Some(WaitReason::Completion) => Status::AwaitingCompletion,
            Some(WaitReason::Finished) => {
                Status::Terminal(record.generation.finish_reason().unwrap())
            }
        })
    }

    pub fn take(&mut self, id: RequestId, count: usize) -> Result<Vec<OutputToken>, String> {
        let output = self
            .records
            .get_mut(&id)
            .ok_or("unknown request")?
            .generation
            .take(count)?;
        if !output.is_empty() {
            self.epoch.advance()?;
        }
        Ok(output)
    }

    pub fn cancel(&mut self, id: RequestId, discard_output: bool) -> Result<(), String> {
        let record = self.records.get_mut(&id).ok_or("unknown request")?;
        let changed = record.generation.finish_reason().is_none()
            || (discard_output && record.generation.output_len() > 0);
        record.generation.cancel();
        if discard_output {
            record.generation.discard_output();
        }
        if changed {
            self.epoch.advance()?;
        }
        Ok(())
    }

    pub fn retire(&mut self, id: RequestId) -> Result<(), String> {
        if self.batch_contains(id) {
            return Err("retirement must wait for shared batch reconciliation".into());
        }
        let record = self.records.get(&id).ok_or("unknown request")?;
        if record.generation.awaiting_completion()
            || record.generation.finish_reason().is_none()
            || record.generation.output_len() != 0
            || record.pending_publication.is_some()
            || record.publication.is_some()
        {
            return Err("retirement requires terminal reconciled work and drained output".into());
        }
        self.retain_terminal(id)?;
        if let Err(error) = self.domain.close(id) {
            if let Some(fatal) = self.domain.fatal_error().cloned() {
                self.fail_domain_error(fatal);
            }
            return Err(error);
        }
        self.records.remove(&id);
        // Rows the closed request shared with retained checkpoints are now
        // held by retention alone.
        self.retention
            .refresh_charge(DomainCheckpoint::exclusive_bytes)?;
        self.epoch.advance()
    }

    pub fn release(&mut self, id: RequestId) -> Result<(), String> {
        if !self.records.contains_key(&id) {
            return Ok(());
        }
        self.cancel(id, true)?;
        if let Some(record) = self.records.get_mut(&id) {
            record.pending_publication = None;
            record.publication_permits.clear();
            // Release owns abandonment, including an admission reply that was
            // dropped before the host received its endpoint. An open receiver
            // observes WorkerClosed; a closed receiver needs no terminal.
            record.publication.take();
        }
        if self.batch_contains(id) {
            self.records.get_mut(&id).unwrap().retire_on_completion = true;
            Ok(())
        } else {
            self.retire(id)
        }
    }

    /// Stop a live host request through the same ordered stream. In-flight
    /// numerical work still reconciles before its owner is retired.
    pub fn stop(&mut self, id: RequestId) -> Result<(), String> {
        self.cancel(id, true)?;
        let record = self.records.get_mut(&id).ok_or("unknown request")?;
        record.pending_publication = None;
        record.publication_permits.clear();
        self.publish_ready().map(|_| ())
    }

    fn publish_ready(&mut self) -> Result<bool, String> {
        let ids = self.records.keys().copied().collect::<Vec<_>>();
        let mut progressed = false;
        for id in ids {
            let in_batch = self.batch_contains(id);
            let mut receiver_closed = false;
            let mut drained = false;
            let mut terminal = false;
            {
                let record = self
                    .records
                    .get_mut(&id)
                    .expect("known publication request");
                let Some(sender) = record.publication.as_mut() else {
                    continue;
                };
                if !sender.receiver_open() {
                    receiver_closed = true;
                } else {
                    if !record.publication_blocked {
                        loop {
                            let tokens = match record.pending_publication.take() {
                                Some(tokens) => tokens,
                                None if record.generation.output_len() > 0 => {
                                    drained = true;
                                    record.generation.take(record.publication_batch_limit)?
                                }
                                None => break,
                            };
                            let Some(permit) = record.publication_permits.pop_front() else {
                                return Err(
                                    "accepted output has no reserved publication permit".into()
                                );
                            };
                            match sender.output_with_permit(tokens, permit) {
                                Ok(()) => {
                                    progressed = true;
                                }
                                Err((PublishError::Full, _)) => {
                                    unreachable!("a publication permit owns an exact output slot")
                                }
                                Err((PublishError::ReceiverClosed, _)) => {
                                    receiver_closed = true;
                                    break;
                                }
                            }
                        }
                        if record.pending_publication.is_none()
                            && record.generation.output_len() == 0
                        {
                            record.publication_permits.clear();
                        }
                    }
                    let needs_reconciliation = record.retention.is_some()
                        && record.generation.pending_reconciliation().is_some()
                        && matches!(
                            record.generation.finish_reason(),
                            Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
                        );
                    terminal = !receiver_closed
                        && record.pending_publication.is_none()
                        && record.generation.output_len() == 0
                        && record.generation.finish_reason().is_some()
                        && !record.generation.awaiting_completion()
                        && !in_batch
                        && !needs_reconciliation;
                }
            }
            if drained {
                self.epoch.advance()?;
            }
            if receiver_closed {
                self.release(id)?;
                progressed = true;
                continue;
            }
            if terminal {
                self.retain_terminal(id)?;
                let record = self
                    .records
                    .get_mut(&id)
                    .ok_or("unknown terminal request")?;
                let sender = record
                    .publication
                    .take()
                    .expect("terminal publication sender");
                match record
                    .generation
                    .finish_reason()
                    .expect("terminal generation")
                {
                    FinishReason::Failed => {
                        sender.fail(record.error.clone().unwrap_or_else(|| {
                            RequestError::Invariant(InvariantError {
                                context: "service request",
                                detail: "failed without a classified cause".into(),
                            })
                        }))
                    }
                    finish => sender.complete(
                        finish,
                        record.generation.detailed_usage(),
                        record.generation.method_identity().to_owned(),
                        record.physical_timings,
                    ),
                }
                self.retire(id)?;
                progressed = true;
            }
        }
        Ok(progressed)
    }

    fn batch_contains(&self, request: RequestId) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|batch| batch.selection.requests().contains(&request))
    }

    fn fail_all(&mut self, error: String) {
        self.fail_all_classified(RequestError::Invariant(InvariantError {
            context: "service worker",
            detail: error,
        }));
    }

    fn fail_domain_error(&mut self, error: DomainError) {
        self.fail_all_classified(classify_domain_error(error));
    }

    fn fail_domain_outcome(&mut self, request: RequestId, error: DomainError) {
        match error {
            DomainError::Device(_)
            | DomainError::Invariant(_)
            | DomainError::Submit(SubmitError::Device(_) | SubmitError::Invariant(_)) => {
                self.fail_domain_error(error);
            }
            other => self.fail_request(request, classify_domain_error(other)),
        }
    }

    fn fail_all_classified(&mut self, error: RequestError) {
        let first = self.fatal.is_none();
        self.fatal.get_or_insert_with(|| format!("{error:?}"));
        if first {
            self.terminalize_all(error);
        }
    }

    fn begin_memory_unload(&mut self) {
        if self.memory_condition == MemoryCondition::Unloading {
            return;
        }
        self.memory_condition = MemoryCondition::Unloading;
        self.terminalize_all(RequestError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure,
        });
    }

    pub fn memory_unload_ready(&mut self) -> bool {
        self.memory_condition == MemoryCondition::Unloading
            && self.batch.as_mut().is_none_or(|batch| {
                batch
                    .active
                    .as_mut()
                    .is_none_or(|active| active.flight.completion().is_complete())
            })
    }

    pub fn memory_unloading(&self) -> bool {
        self.memory_condition == MemoryCondition::Unloading
    }

    fn terminalize_all(&mut self, error: RequestError) {
        for record in self.records.values_mut() {
            if record.generation.finish_reason().is_none()
                || record.generation.awaiting_completion()
            {
                record.generation.fail();
                record.error.get_or_insert_with(|| error.clone());
            }
        }
        for record in self.records.values_mut() {
            // Only output already accepted by the queue precedes a terminal
            // outcome. Unpublished generation output is not promoted by
            // a second, unbounded failure delivery channel.
            if let Some(sender) = record.publication.take() {
                record.generation.discard_output();
                record.pending_publication = None;
                record.publication_permits.clear();
                match record.generation.finish_reason() {
                    Some(FinishReason::Failed) | None => {
                        sender.fail(record.error.clone().unwrap_or_else(|| error.clone()));
                    }
                    Some(finish) => sender.complete(
                        finish,
                        record.generation.detailed_usage(),
                        record.generation.method_identity().to_owned(),
                        record.physical_timings,
                    ),
                }
            }
        }
    }

    fn fail_request(&mut self, id: RequestId, error: RequestError) {
        if let Some(record) = self.records.get_mut(&id) {
            record.publication_permits.clear();
            record.generation.fail();
            record.error = Some(error);
            record.capacity = None;
        }
    }

    pub fn step(&mut self, now: u64) -> Result<Step, String> {
        self.time(now)?;
        if let Some(error) = self.domain.fatal_error().cloned() {
            self.fail_domain_error(error);
            return Ok(Step::Idle);
        }
        let before = self.publish_ready()?;
        let mut step = self.step_inner(now)?;
        if step == Step::Reconciled {
            // Submit the next round before publishing the reconciled batch's
            // tokens, so publication overlaps device execution. The round
            // never depends on publication; it reads only reconciled state.
            step = self.step_inner(now)?;
        }
        let after = self.publish_ready()?;
        if (before || after) && matches!(step, Step::Idle | Step::Waiting) {
            Ok(Step::Progress)
        } else {
            Ok(step)
        }
    }

    fn step_inner(&mut self, now: u64) -> Result<Step, String> {
        if self.memory_condition == MemoryCondition::Unloading {
            return Ok(Step::Idle);
        }
        if self.batch.is_some() {
            return self.drive_batch(now);
        }
        if self.fatal.is_some() {
            return Ok(Step::Idle);
        }
        self.attach_awaited()?;
        let mut candidates = Vec::new();
        for (&id, record) in &mut self.records {
            if record.awaited_prefix.is_some() {
                continue;
            }
            if record
                .capacity
                .is_some_and(|(epoch, _, _)| !self.epoch.changed_since(epoch))
            {
                continue;
            }
            record.capacity = None;
            let reconciliation = record.retention.is_some()
                && !record.terminal_retained
                && matches!(
                    record.generation.finish_reason(),
                    Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
                )
                && record.generation.pending_reconciliation().is_some();
            if record.generation.awaiting_completion()
                || record.publication_blocked
                || (!reconciliation
                    && (record.generation.finish_reason().is_some()
                        || matches!(record.generation.wait_reason(), Some(WaitReason::Output))))
            {
                continue;
            }
            let prefill = !reconciliation
                && (!record.generation.is_resident()
                    || record.generation.resident_position()
                        < record.generation.usage().prompt_tokens);
            candidates.push(ScheduledOperation {
                identity: id,
                phase: if prefill {
                    Phase::Prefill
                } else {
                    Phase::Decode
                },
                active: record.service_ns > 0,
                resident: record.generation.is_resident(),
                waiting_since_ns: record.waiting_since,
                service_ns: record.service_ns,
                preemption_debt: record.preemption_debt,
            });
        }
        let Some(mut selection) = self.scheduler.select(&candidates, now)? else {
            // Idle (no request is live, not merely none schedulable now):
            // release state backing well beyond what the stores need.
            let live = self.records.values().any(|record| {
                record.generation.is_resident() && record.generation.finish_reason().is_none()
            });
            if !live {
                if let Err(error) = self.domain.shrink_state(ShrinkPolicy::Idle) {
                    self.fail_domain_error(error);
                }
            }
            return Ok(Step::Idle);
        };
        // Prefill rows are one budget for the whole launch: requests take
        // their chunks from it in selection order, and requests left without
        // rows wait for the next round. Decode allowances are per request.
        let mut prefill_rows = self.scheduler.limits().prefill_tokens;
        let mut started = 0;
        let mut operations = Vec::new();
        for &request in selection.requests() {
            let allowance = match selection.phase() {
                Phase::Prefill if prefill_rows == 0 => break,
                Phase::Prefill => prefill_rows,
                Phase::Decode => self.scheduler.limits().decode_tokens,
            };
            started += 1;
            match self.start_request(request, allowance) {
                Ok(mut next) => {
                    let rows = next
                        .iter()
                        .filter(|operation| matches!(operation, Operation::Forward { .. }))
                        .map(Operation::row_count)
                        .sum::<usize>();
                    prefill_rows = prefill_rows.saturating_sub(rows);
                    operations.append(&mut next);
                }
                Err(error) => {
                    self.fail_request(request, RequestError::Input(error));
                    self.epoch.advance()?;
                }
            }
        }
        selection.truncate(started)?;
        if operations.is_empty() {
            self.scheduler.completed(selection, 0);
            return Ok(Step::Progress);
        }
        let groups = group(&self.domain, operations);
        self.batch = Some(Batch {
            selection,
            started: now,
            queued: groups.into_iter().collect(),
            active: None,
            blocked: None,
        });
        self.drive_batch(now)
    }

    /// Reopen an evicted request. Replay resumes from the deepest retained
    /// prefix of its input (its own prompt checkpoint, or a shared prefix
    /// another request left in the index), or from the start without one.
    fn restore(&mut self, request: RequestId) -> Result<(), String> {
        let record = self.records.get(&request).ok_or("unknown request")?;
        let accepted = record.generation.accepted_position();
        let hit = match record.retention.as_ref() {
            Some(input) => self.retention.lookup(input, accepted)?,
            None => None,
        };
        let Some(hit) = hit else {
            self.ensure_open_capacity()
                .map_err(|error| error.to_string())?;
            let reservation = self
                .domain
                .reserve_open(request)
                .map_err(|error| error.to_string())?;
            self.domain
                .open_reserved(reservation)
                .map_err(|error| error.to_string())?;
            return self
                .records
                .get_mut(&request)
                .expect("known request")
                .generation
                .restored();
        };
        let retained = self.retention.entry(hit)?;
        self.domain
            .open_checkpoint_state(request, retained.checkpoint())
            .map_err(|error| error.to_string())?;
        let record = self.records.get_mut(&request).expect("known request");
        if let Err(error) = record
            .generation
            .restored_at(hit.position(), retained.method())
        {
            return Err(match self.domain.close(request) {
                Ok(()) => error,
                Err(rollback) => format!("{error}; executor rollback failed ({rollback})"),
            });
        }
        record.retention_source = Some(hit.id());
        Ok(())
    }

    fn start_request(
        &mut self,
        request: RequestId,
        allowance: usize,
    ) -> Result<Vec<Operation>, String> {
        let needs_open = !self
            .records
            .get(&request)
            .ok_or("unknown request")?
            .generation
            .is_resident();
        if needs_open {
            self.restore(request)?;
        }
        let record = self.records.get_mut(&request).expect("known request");
        if !record.admission.is_empty() {
            return Ok(record.admission.drain(..).collect());
        }
        if matches!(
            record.generation.finish_reason(),
            Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
        ) && record.generation.pending_reconciliation().is_some()
        {
            record.generation.start_reconciliation(allowance)?;
            return Ok(vec![lower_round(
                &record.generation,
                request,
                record.generation.resident_position(),
                None,
            )?]);
        }
        // A prefill chunk ends exactly at a planned branch point, and at the
        // prompt's retained boundary one row before its end.
        let resident = record.generation.resident_position();
        let stop = [record.branch, prompt_boundary(record)]
            .into_iter()
            .flatten()
            .filter(|&stop| stop > resident)
            .min();
        let allowance = match stop {
            Some(stop) => allowance.min(stop - resident),
            None => allowance,
        };
        match record.generation.start_round(request, allowance)? {
            RoundStart::Target => Ok(vec![lower_round(
                &record.generation,
                request,
                record.generation.resident_position(),
                None,
            )?]),
            RoundStart::Method(operations) => {
                if operations
                    .iter()
                    .any(|operation| operation.request() != request)
                {
                    return Err("method returned an operation for another request".into());
                }
                Ok(operations)
            }
        }
    }

    fn drive_batch(&mut self, now: u64) -> Result<Step, String> {
        if self
            .batch
            .as_ref()
            .and_then(|batch| batch.blocked)
            .is_some_and(|blocked| !self.epoch.changed_since(blocked))
        {
            return Ok(Step::Waiting);
        }
        if let Some(batch) = &mut self.batch {
            batch.blocked = None;
        }
        let complete = self
            .batch
            .as_mut()
            .and_then(|batch| batch.active.as_mut())
            .is_some_and(|active| active.flight.completion().is_complete());
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_some())
            && !complete
        {
            return Ok(Step::Waiting);
        }
        if complete {
            self.reconcile_active()?;
            self.release_deferred_pressure()?;
            if self.memory_condition == MemoryCondition::Unloading {
                return Ok(Step::Idle);
            }
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_none() && !batch.queued.is_empty())
        {
            return self.submit_next();
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_none() && batch.queued.is_empty())
        {
            self.finish_batch(now)?;
            return Ok(Step::Reconciled);
        }
        Ok(Step::Waiting)
    }

    fn submit_next(&mut self) -> Result<Step, String> {
        let queued = self.batch.as_mut().unwrap().queued.pop_front().unwrap();
        let ended = queued
            .operations()
            .iter()
            .map(Operation::request)
            .filter(|request| {
                self.records.get(request).is_none_or(|record| {
                    matches!(
                        record.generation.finish_reason(),
                        Some(FinishReason::Cancelled | FinishReason::Failed)
                    )
                })
            })
            .collect::<Vec<_>>();
        if !ended.is_empty() {
            let live = queued
                .into_operations()
                .into_iter()
                .filter(|operation| !ended.contains(&operation.request()))
                .collect::<Vec<_>>();
            if !live.is_empty() {
                let mut groups = group(&self.domain, live);
                if groups.len() != 1 {
                    return Err("filtered operation group changed its numerical lane".into());
                }
                self.batch
                    .as_mut()
                    .unwrap()
                    .queued
                    .push_front(groups.pop().unwrap());
            }
            self.epoch.advance()?;
            return Ok(Step::Progress);
        }
        let first_provision = match self.domain.provision(queued.operations()) {
            Ok(()) => Ok(()),
            Err(
                error @ (DomainError::Capacity(_)
                | DomainError::Blind(_)
                | DomainError::Pressure(_)),
            ) => Err(error),
            Err(error) => {
                self.fail_domain_error(error);
                return Ok(Step::Progress);
            }
        };
        let requirement = match requirements(&self.domain, &queued) {
            Ok(requirement) => requirement,
            Err(error) => {
                self.fail_domain_error(error);
                return Ok(Step::Progress);
            }
        };
        // A growth claim and the structural reservation are one capacity
        // question. Recheck both after each release: free device bytes do not
        // become committed state rows until provisioning succeeds.
        let mut current = first_provision.and_then(|_| {
            self.domain
                .can_reserve(&requirement)
                .map_err(DomainError::from)
        });
        if matches!(&current, Err(DomainError::Capacity(_))) {
            // Selection is provisional until every exact requirement is
            // available. Memory pressure first releases state backing beyond
            // what the stores need (another store may hold the device memory
            // this one needs to grow into); policy then releases
            // least-recently-used retention only until the requirement fits,
            // then idle residency, then eligible live victims. No launch or
            // state transaction exists while these scheduling decisions run.
            match self.domain.shrink_state(ShrinkPolicy::Pressure) {
                Ok(_) => current = self.provisioned_capacity(queued.operations(), &requirement),
                Err(error) => {
                    self.fail_domain_error(error);
                    return Ok(Step::Progress);
                }
            }
            if matches!(&current, Err(DomainError::Capacity(_))) {
                self.domain.reclaim_idle()?;
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
            while matches!(&current, Err(DomainError::Capacity(_))) {
                if self
                    .retention
                    .evict_one(DomainCheckpoint::exclusive_bytes)?
                    .is_none()
                {
                    break;
                }
                if let Err(error) = self.domain.shrink_state(ShrinkPolicy::Pressure) {
                    self.fail_domain_error(error);
                    return Ok(Step::Progress);
                }
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
            if matches!(&current, Err(DomainError::Capacity(_))) {
                self.domain.release_idle_optional_components()?;
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
            let selected = self.batch.as_ref().unwrap().selection.requests().to_vec();
            while matches!(&current, Err(DomainError::Capacity(_))) {
                if !self.evict_victims(&selected, false)? {
                    break;
                }
                if let Err(error) = self.domain.shrink_state(ShrinkPolicy::Pressure) {
                    self.fail_domain_error(error);
                    return Ok(Step::Progress);
                }
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
        }
        if matches!(&current, Err(DomainError::Pressure(_))) {
            self.relieve_platform_pressure()?;
            self.requeue_group(queued);
            return Ok(Step::Waiting);
        }
        match current {
            Ok(()) => {}
            Err(DomainError::Blind(_)) => {
                // Keep the accepted batch intact. A missing observation
                // revokes the allocation grant but says nothing about which
                // existing holdings could satisfy this request.
                self.requeue_group(queued);
                return Ok(Step::Waiting);
            }
            Err(DomainError::Capacity(deficit)) => {
                return self.capacity_unavailable(
                    queued,
                    deficit.resource,
                    deficit.required,
                    deficit.available,
                )
            }
            Err(DomainError::Pressure(_)) => unreachable!("pressure handled above"),
            Err(error) => {
                self.fail_domain_error(error);
                return Ok(Step::Progress);
            }
        }
        let Some(publication_permits) = self.reserve_group_publications(queued.operations())?
        else {
            let batch = self.batch.as_mut().unwrap();
            batch.queued.push_front(queued);
            batch.blocked = Some(self.epoch);
            return Ok(Step::Waiting);
        };
        let submitted = submit_group(&mut self.domain, &queued);
        match submitted {
            Ok(flight) => {
                let operations = queued.into_operations();
                let mut retention_uses = Vec::new();
                for request in operations
                    .iter()
                    .map(Operation::request)
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    if let Some(source) = self
                        .records
                        .get(&request)
                        .and_then(|record| record.retention_source)
                    {
                        if self.retention.begin_submitted_if_resident(source)? {
                            retention_uses.push(source);
                        }
                    }
                }
                self.batch.as_mut().unwrap().active = Some(ActiveGroup {
                    flight,
                    operations,
                    retention_uses,
                    publication_permits,
                });
                Ok(Step::Submitted)
            }
            Err(DomainError::Capacity(error)) => {
                self.fail_all_classified(RequestError::Invariant(InvariantError {
                    context: "reserved service submission",
                    detail: format!("capacity changed after reservation: {error}"),
                }));
                Ok(Step::Progress)
            }
            Err(DomainError::Blind(_)) => {
                // The domain can re-observe during reserve after the owner's
                // preflight. No work was submitted. Give unused publication
                // permits back, then rebuild groups in case provisioning
                // changed the domain's grouping state.
                drop(publication_permits);
                self.requeue_group(queued);
                Ok(Step::Waiting)
            }
            Err(DomainError::Pressure(_)) => {
                drop(publication_permits);
                self.requeue_group(queued);
                Ok(Step::Waiting)
            }
            Err(DomainError::Input(error)) => {
                for request in queued
                    .operations()
                    .iter()
                    .map(Operation::request)
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    self.fail_request(request, RequestError::Input(error.clone()));
                }
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
            Err(DomainError::State(error)) => {
                self.fail_all_classified(RequestError::State(error));
                self.batch = None;
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
            Err(DomainError::Submit(SubmitError::Device(error)))
            | Err(DomainError::Device(error)) => {
                self.fail_all_classified(RequestError::Device(error));
                self.batch = None;
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
            Err(DomainError::Submit(SubmitError::Invariant(error)))
            | Err(DomainError::Invariant(error)) => {
                self.fail_all_classified(RequestError::Invariant(error));
                self.batch = None;
                self.epoch.advance()?;
                Ok(Step::Progress)
            }
        }
    }

    fn requeue_group(&mut self, queued: OperationGroup) {
        let regrouped = group(&self.domain, queued.into_operations());
        let batch = self.batch.as_mut().expect("queued group has a batch");
        for group in regrouped.into_iter().rev() {
            batch.queued.push_front(group);
        }
    }

    /// Pressure may discard surplus and retained reuse state, never accepted
    /// live work. Stop as soon as the platform reports Normal or Blind.
    /// Emergency preemption and model unload require the separate owner
    /// lifecycle transition; this routine performs only safe release rungs.
    fn relieve_platform_pressure(&mut self) -> Result<(), String> {
        self.domain
            .shrink_state(ShrinkPolicy::Pressure)
            .map_err(|error| error.to_string())?;
        if !self.pressure_release_needed() {
            return Ok(());
        }
        self.domain.reclaim_idle()?;
        while self.pressure_release_needed() {
            if self
                .retention
                .evict_one(DomainCheckpoint::exclusive_bytes)?
                .is_none()
            {
                break;
            }
            // Eviction drops claims, but committed rows and banks remain
            // charged until the stores shrink. Probe the device only after
            // that physical release has had a chance to complete.
            self.domain
                .shrink_state(ShrinkPolicy::Pressure)
                .map_err(|error| error.to_string())?;
            self.domain.reclaim_idle()?;
        }
        // Optional weights are dormant only after every request has left the
        // owner. Use the device's actual charge decrease as release credit.
        if self.records.is_empty()
            && self.pressure_release_needed()
            && matches!(
                self.memory_condition,
                MemoryCondition::Pressure | MemoryCondition::Emergency { .. }
            )
        {
            self.domain.release_idle_optional_components()?;
        }
        Ok(())
    }

    fn pressure_release_needed(&mut self) -> bool {
        match self.domain.probe_memory() {
            Err(DomainError::Pressure(_)) => true,
            Err(DomainError::Blind(_))
                if matches!(self.memory_condition, MemoryCondition::Emergency { .. }) =>
            {
                true
            }
            _ => false,
        }
    }

    fn reserve_group_publications(
        &mut self,
        operations: &[Operation],
    ) -> Result<Option<BTreeMap<RequestId, Vec<PublicationPermit>>>, String> {
        let mut tokens = BTreeMap::<RequestId, usize>::new();
        for operation in operations {
            let count = match operation {
                Operation::Forward { select, .. } => select.len(),
                Operation::Head { .. } | Operation::Encode { .. } => 0,
            };
            if count != 0 {
                *tokens.entry(operation.request()).or_default() += count;
            }
        }
        let mut reserved = BTreeMap::new();
        for (request, tokens) in tokens {
            let record = self
                .records
                .get_mut(&request)
                .ok_or("unknown publication request")?;
            let Some(sender) = record.publication.as_ref() else {
                continue;
            };
            let batch = record.publication_batch_limit;
            if batch == 0 {
                return Err("publication stream has no token batch limit".into());
            }
            let slots = tokens.div_ceil(batch);
            match sender.reserve_outputs(slots) {
                Ok(permits) => {
                    reserved.insert(request, permits);
                }
                Err(PublishError::Full | PublishError::ReceiverClosed) => {
                    record.publication_blocked = true;
                    return Ok(None);
                }
            }
        }
        Ok(Some(reserved))
    }

    fn provisioned_capacity(
        &mut self,
        operations: &[Operation],
        requirement: &DomainRequirements,
    ) -> Result<(), DomainError> {
        self.domain.provision(operations)?;
        self.domain
            .can_reserve(requirement)
            .map_err(DomainError::from)
    }

    fn capacity_unavailable(
        &mut self,
        queued: OperationGroup,
        resource: ResourceKind,
        required: u64,
        available: u64,
    ) -> Result<Step, String> {
        let operations = queued.operations();
        if std::env::var_os("MAGNITUDE_TRACE_TARGET").is_some() {
            eprintln!(
                "target admission deficit {:?} required={} available={} operations={} rows={:?}",
                resource,
                required,
                available,
                operations.len(),
                operations
                    .iter()
                    .map(Operation::row_count)
                    .collect::<Vec<_>>()
            );
        }
        let requests = operations
            .iter()
            .map(Operation::request)
            .collect::<Vec<_>>();
        let selected = self.batch.as_ref().unwrap().selection.requests().to_vec();

        let can_change = self.records.iter().any(|(id, record)| {
            !selected.contains(id)
                && (record.generation.finish_reason().is_some()
                    || record.generation.output_len() > 0
                    || record.capacity.is_none())
        });
        if can_change {
            for request in &requests {
                if let Some(record) = self.records.get_mut(request) {
                    record.capacity = Some((self.epoch, required, available));
                }
            }
            let batch = self.batch.as_mut().unwrap();
            batch.queued.push_front(queued);
            batch.blocked = Some(self.epoch);
            Ok(Step::Waiting)
        } else {
            for request in requests {
                self.fail_request(
                    request,
                    RequestError::Capacity(ServiceCapacityError {
                        resource: CapacityResource::Execution(resource),
                        required,
                        available,
                    }),
                );
            }
            self.epoch.advance()?;
            Ok(Step::Progress)
        }
    }

    fn abort_pending(
        &mut self,
        pending: magnitude_model_executor::PendingOperationOutcome,
    ) -> Result<(), String> {
        match self.domain.abort(pending) {
            Ok(()) => Ok(()),
            Err(error) => {
                let detail = error.to_string();
                self.fail_domain_error(error);
                Err(detail)
            }
        }
    }

    fn reconcile_active(&mut self) -> Result<(), String> {
        let active = self.batch.as_mut().unwrap().active.take().unwrap();
        for source in active.retention_uses {
            self.retention.end_submitted(source)?;
        }
        for (request, permits) in active.publication_permits {
            if let Some(record) = self.records.get_mut(&request) {
                record.publication_permits.extend(permits);
            }
        }
        match active.flight {
            DomainFlight::Head(flight) => {
                let pending = match self.domain.finish_head(flight) {
                    Ok(pending) => pending,
                    Err(error) => {
                        self.fail_domain_error(error);
                        return Ok(());
                    }
                };
                self.reconcile_method_outcomes(pending, &active.operations)?;
            }
            DomainFlight::Vision(flight) => {
                let item = match self.domain.finish_vision(flight) {
                    Ok(item) => item,
                    Err(error) => {
                        self.fail_domain_error(error);
                        return Ok(());
                    }
                };
                let request = item.request();
                let valid = matches!(active.operations.as_slice(),
                    [Operation::Encode { request: expected, .. }] if *expected == request)
                    && matches!(item.outcome(), Outcome::Encode { .. });
                if !valid || !self.records.contains_key(&request) {
                    self.abort_pending(item)?;
                    self.fail_all("vision outcome differs from submitted request".into());
                    return Ok(());
                }
                let record = self.records.get_mut(&request).unwrap();
                record.add_physical_duration(item.kind(), item.physical_duration());
                if matches!(
                    record.generation.finish_reason(),
                    Some(FinishReason::Cancelled | FinishReason::Failed)
                ) {
                    self.abort_pending(item)?;
                } else if let Err(error) = self
                    .domain
                    .reconcile(item, PhysicalDecision { accepted_rows: 0 })
                {
                    self.fail_domain_outcome(request, error);
                }
            }
            DomainFlight::Target(flight) => {
                let pending = match self.domain.finish_target(flight) {
                    Ok(pending) => pending,
                    Err(error) => {
                        self.fail_domain_error(error);
                        return Ok(());
                    }
                };
                let expected = active
                    .operations
                    .iter()
                    .map(|operation| (operation.request(), operation))
                    .collect::<BTreeMap<_, _>>();
                let actual = pending
                    .iter()
                    .map(|item| item.request())
                    .collect::<std::collections::BTreeSet<_>>();
                if pending.len() != active.operations.len()
                    || actual.len() != pending.len()
                    || actual != expected.keys().copied().collect()
                {
                    for item in pending {
                        self.abort_pending(item)?;
                    }
                    self.fail_all(
                        "executor outcomes do not match submitted request identities".into(),
                    );
                    return Ok(());
                }
                let mut followups = Vec::new();
                for item in pending {
                    let request = item.request();
                    let kind = item.kind();
                    let duration = item.physical_duration();
                    let operation = expected[&request];
                    let Some(record) = self.records.get_mut(&request) else {
                        self.abort_pending(item)?;
                        continue;
                    };
                    record.add_physical_duration(kind, duration);
                    if matches!(
                        record.generation.finish_reason(),
                        Some(FinishReason::Cancelled | FinishReason::Failed)
                    ) {
                        self.abort_pending(item)?;
                        continue;
                    }
                    if !matches!(operation, Operation::Forward { .. }) {
                        self.abort_pending(item)?;
                        self.fail_request(
                            request,
                            RequestError::Invariant(InvariantError {
                                context: "service target flight",
                                detail: "target flight contains a non-forward operation".into(),
                            }),
                        );
                        continue;
                    }
                    match reconcile_forward(&mut record.generation, &mut self.domain, request, item)
                    {
                        Ok(effects) => {
                            followups.extend(effects.operations);
                        }
                        Err(RoundError::Logical(error)) => {
                            self.fail_request(request, RequestError::Input(error))
                        }
                        Err(RoundError::Physical(error)) => {
                            self.fail_domain_outcome(request, error)
                        }
                    }
                }
                if !followups.is_empty() {
                    self.queue_operations(followups);
                }
            }
        }
        Ok(())
    }

    fn reconcile_method_outcomes(
        &mut self,
        pending: Vec<magnitude_model_executor::PendingOperationOutcome>,
        operations: &[Operation],
    ) -> Result<(), String> {
        let expected = operations
            .iter()
            .map(|operation| (operation.request(), operation))
            .collect::<BTreeMap<_, _>>();
        let actual = pending
            .iter()
            .map(|item| item.request())
            .collect::<std::collections::BTreeSet<_>>();
        if pending.len() != operations.len()
            || actual.len() != pending.len()
            || actual != expected.keys().copied().collect()
        {
            for item in pending {
                self.abort_pending(item)?;
            }
            self.fail_all("method outcomes do not match submitted requests".into());
            return Ok(());
        }
        for item in pending {
            let request = item.request();
            let duration = item.physical_duration();
            let kind = item.kind();
            let operation = expected[&request];
            let correct = matches!(operation, Operation::Head { .. })
                && matches!(item.outcome(), Outcome::Head { .. });
            let Some(record) = self.records.get_mut(&request) else {
                self.abort_pending(item)?;
                continue;
            };
            record.add_physical_duration(kind, duration);
            if matches!(
                record.generation.finish_reason(),
                Some(FinishReason::Cancelled | FinishReason::Failed)
            ) {
                self.abort_pending(item)?;
                continue;
            }
            if !correct {
                self.abort_pending(item)?;
                self.fail_request(
                    request,
                    RequestError::Invariant(InvariantError {
                        context: "service method flight",
                        detail: "method outcome differs from submitted operation".into(),
                    }),
                );
                continue;
            }
            let transition = match record
                .generation
                .prepare_method_transition(operation, item.outcome())
            {
                Ok(transition) => transition,
                Err(error) => {
                    self.abort_pending(item)?;
                    self.fail_request(request, RequestError::Input(error));
                    continue;
                }
            };
            match self.domain.reconcile(
                item,
                PhysicalDecision {
                    accepted_rows: transition.decision().accepted_rows,
                },
            ) {
                Ok(_) => record.generation.commit_method_transition(transition),
                Err(error) => self.fail_domain_outcome(request, error),
            }
        }
        Ok(())
    }

    fn queue_operations(&mut self, operations: Vec<Operation>) {
        let groups = group(&self.domain, operations);
        self.batch.as_mut().unwrap().queued.extend(groups);
    }

    fn finish_batch(&mut self, now: u64) -> Result<(), String> {
        let batch = self.batch.take().unwrap();
        let elapsed = now.saturating_sub(batch.started);
        let count = batch.selection.requests().len() as u64;
        let base = elapsed / count;
        let remainder = elapsed % count;
        for (index, request) in batch.selection.requests().iter().enumerate() {
            if let Some(record) = self.records.get_mut(request) {
                let share = base + u64::from((index as u64) < remainder);
                record.service_ns = record.service_ns.saturating_add(share);
                record.waiting_since = now;
                if record
                    .protected_until
                    .is_some_and(|boundary| record.generation.resident_position() > boundary)
                {
                    record.protected_until = None;
                }
            }
        }
        let completed_requests = batch.selection.requests().to_vec();
        self.scheduler.completed(batch.selection, elapsed);
        for request in completed_requests {
            self.retain_branch_point(request)?;
            self.retain_prefill_boundary(request)?;
            self.retain_terminal(request)?;
        }
        self.epoch.advance()?;
        let released = self
            .records
            .iter()
            .filter_map(|(&id, record)| record.retire_on_completion.then_some(id))
            .collect::<Vec<_>>();
        for id in released {
            self.retire(id)?;
        }
        Ok(())
    }

    /// Retain the request's reconciled state at its planned branch point: the
    /// recurrent bank and method state there, sharing every history row
    /// before it with the request itself.
    fn retain_branch_point(&mut self, request: RequestId) -> Result<(), String> {
        let Some(record) = self.records.get_mut(&request) else {
            return Ok(());
        };
        let Some(branch) = record.branch else {
            return Ok(());
        };
        let resident = record.generation.resident_position();
        if record.generation.finish_reason().is_some()
            || !record.generation.is_resident()
            || resident > branch
        {
            record.branch = None;
            return Ok(());
        }
        if resident < branch || record.generation.awaiting_completion() {
            return Ok(());
        }
        record.branch = None;
        let tokens = record.generation.prompt()[..branch].to_vec();
        let numerical = self.domain.checkpoint(request)?;
        let method = self.records[&request].generation.method_checkpoint()?;
        let retained = self.retention.retain(
            self.records[&request]
                .retention
                .as_ref()
                .ok_or("a branch point requires a retention request")?,
            tokens,
            numerical,
            method,
            DomainCheckpoint::exclusive_bytes,
        )?;
        if retained {
            self.epoch.advance()?;
        }
        Ok(())
    }

    /// Retain the request's prompt one row before its end (see
    /// [`prompt_boundary`]): the deepest point a request with the same prompt
    /// can resume from, since it must compute the row it samples from.
    fn retain_prefill_boundary(&mut self, request: RequestId) -> Result<(), String> {
        let Some(record) = self.records.get(&request) else {
            return Ok(());
        };
        let Some(boundary) = prompt_boundary(record) else {
            return Ok(());
        };
        if record.prefill_retained
            || matches!(
                record.generation.finish_reason(),
                Some(FinishReason::Cancelled | FinishReason::Failed)
            )
            || record.generation.resident_position() != boundary
            || record.generation.awaiting_completion()
            || !record.generation.is_resident()
        {
            return Ok(());
        }
        let prompt = record.generation.prompt()[..boundary].to_vec();
        let numerical = self.domain.checkpoint(request)?;
        if numerical.position() != boundary {
            return Err("prefill retention checkpoint differs from the prompt boundary".into());
        }
        let method = self
            .records
            .get(&request)
            .unwrap()
            .generation
            .method_checkpoint()?;
        let retained = self.retention.retain(
            self.records
                .get(&request)
                .unwrap()
                .retention
                .as_ref()
                .unwrap(),
            prompt,
            numerical,
            method,
            DomainCheckpoint::exclusive_bytes,
        )?;
        self.records.get_mut(&request).unwrap().prefill_retained = true;
        if retained {
            self.epoch.advance()?;
        }
        Ok(())
    }

    fn retain_terminal(&mut self, request: RequestId) -> Result<(), String> {
        let record = self.records.get(&request).ok_or("unknown request")?;
        if record.terminal_retained
            || record.retention.is_none()
            || !record.generation.is_resident()
            || !matches!(
                record.generation.finish_reason(),
                Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
            )
        {
            return Ok(());
        }
        let tokens = record
            .generation
            .prompt()
            .iter()
            .chain(record.generation.generated())
            .copied()
            .collect::<Vec<_>>();
        let numerical = self.domain.checkpoint(request)?;
        if numerical.position() != tokens.len() {
            if record.generation.pending_reconciliation().is_some() {
                return Ok(());
            }
            return Err(
                "terminal numerical state is not reconciled at the accepted boundary".into(),
            );
        }
        let method = self
            .records
            .get(&request)
            .unwrap()
            .generation
            .method_checkpoint()?;
        let retained = self.retention.retain(
            self.records
                .get(&request)
                .unwrap()
                .retention
                .as_ref()
                .unwrap(),
            tokens,
            numerical,
            method,
            DomainCheckpoint::exclusive_bytes,
        )?;
        if retained {
            self.epoch.advance()?;
        }
        self.records.get_mut(&request).unwrap().terminal_retained = true;
        Ok(())
    }

    fn evict_victims(
        &mut self,
        selected: &[RequestId],
        release_resident_slot: bool,
    ) -> Result<bool, String> {
        let mut victims = Vec::new();
        for (&id, record) in &self.records {
            if selected.contains(&id)
                || !record.generation.is_resident()
                || record.generation.awaiting_completion()
                || record.generation.finish_reason().is_some()
                || !record.admission.is_empty()
                || self.batch_contains(id)
                || record.protected_until.is_some()
            {
                continue;
            }
            let executor_bytes = self.domain.reclaimable(&[id])?;
            victims.push(Victim {
                identity: id,
                output_blocked: record.publication_blocked
                    || matches!(record.generation.wait_reason(), Some(WaitReason::Output)),
                preemption_debt: record.preemption_debt,
                exclusive_bytes: executor_bytes
                    .checked_add(record.generation.method_reclaimable())
                    .ok_or("request reclaim byte count overflow")?,
                replay_tokens: record.generation.accepted_position() as u64,
                service_ns: record.service_ns,
            });
        }
        order_victims(&mut victims);
        let mut set = Vec::new();
        for victim in victims {
            set.push(victim.identity);
            let method_bytes = set.iter().try_fold(0u64, |total, request| {
                total
                    .checked_add(
                        self.records
                            .get(request)
                            .ok_or("unknown eviction request")?
                            .generation
                            .method_reclaimable(),
                    )
                    .ok_or_else(|| "method reclaim byte count overflow".to_owned())
            })?;
            if !release_resident_slot && self.domain.reclaimable(&set)? == 0 && method_bytes == 0 {
                continue;
            }
            let executor_released = match self.domain.evict(&set) {
                Ok(bytes) => bytes,
                Err(error) => {
                    if let Some(fatal) = self.domain.fatal_error().cloned() {
                        self.fail_domain_error(fatal);
                    }
                    return Err(error);
                }
            };
            if !release_resident_slot && executor_released == 0 && method_bytes == 0 {
                continue;
            }
            for id in set {
                let record = self.records.get_mut(&id).unwrap();
                record.protected_until = Some(record.generation.accepted_position());
                record.generation.evicted()?;
                record.preemption_debt = record.preemption_debt.saturating_add(1);
                record.capacity = None;
            }
            self.epoch.advance()?;
            return Ok(true);
        }
        Ok(false)
    }
}

impl<F: ProgramFamily> super::worker::Driven for Owner<F> {
    fn periodic(&mut self, now: u64) -> Result<(), String> {
        self.observe_periodic_memory(now)
    }

    fn publication_wake(
        &mut self,
        request: RequestId,
        wake: PublicationWake,
        _now: u64,
    ) -> Result<(), String> {
        Owner::publication_wake(self, request, wake)
    }

    fn failure(&self) -> Option<&str> {
        self.fatal_error()
    }

    fn advance(
        &mut self,
        now: u64,
        wake: super::worker::CompletionWake,
    ) -> Result<super::worker::Drive, String> {
        use super::worker::Drive;
        match self.step(now)? {
            Step::Submitted => {
                self.batch
                    .as_mut()
                    .and_then(|batch| batch.active.as_mut())
                    .expect("submitted domain flight")
                    .flight
                    .completion()
                    .notify(wake);
                Ok(Drive::AwaitingCompletion)
            }
            Step::Reconciled | Step::Progress => Ok(Drive::Progress),
            Step::Idle => Ok(Drive::Idle),
            Step::Waiting
                if self
                    .batch
                    .as_ref()
                    .is_none_or(|batch| batch.active.is_none()) =>
            {
                Ok(Drive::Idle)
            }
            Step::Waiting => Err("completion notification preceded executor completion".into()),
        }
    }

    fn failed(&mut self, error: &str) {
        self.fail_all(error.into());
    }

    fn shutdown(&mut self) -> Result<bool, String> {
        self.checkpoints.clear();
        let ids = self.records.keys().copied().collect::<Vec<_>>();
        for &id in &ids {
            self.cancel(id, true)?;
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_some())
        {
            return Ok(false);
        }
        self.batch = None;
        for id in ids {
            self.domain.close(id)?;
            self.records.remove(&id);
        }
        self.retention
            .evict_all(DomainCheckpoint::exclusive_bytes)?;
        Ok(true)
    }
}

#[cfg(test)]
mod memory_condition_tests {
    use super::{MemoryCondition, MemoryObservation, MEMORY_ESCALATION_NS};

    #[test]
    fn continuous_blind_escalates_then_unloads_after_emergency_interval() {
        let blind = MemoryCondition::Normal.observe(MemoryObservation::Blind, 7);
        assert_eq!(blind, MemoryCondition::Blind { since: 7 });
        let before = blind.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS - 1);
        assert_eq!(before, blind);
        let emergency = before.observe(MemoryObservation::Blind, 7 + MEMORY_ESCALATION_NS);
        assert_eq!(
            emergency,
            MemoryCondition::Emergency {
                since: 7 + MEMORY_ESCALATION_NS
            }
        );
        assert!(!emergency.should_unload(7 + 2 * MEMORY_ESCALATION_NS - 1));
        assert!(emergency.should_unload(7 + 2 * MEMORY_ESCALATION_NS));
        assert_eq!(
            emergency.observe(MemoryObservation::Normal, 7 + 2 * MEMORY_ESCALATION_NS),
            MemoryCondition::Normal
        );
    }

    #[test]
    fn emergency_timer_resets_after_pressure_recovery() {
        let emergency = MemoryCondition::Normal.observe(MemoryObservation::Emergency, 5);
        assert_eq!(
            emergency.observe(MemoryObservation::Emergency, 9),
            emergency
        );
        let pressure = emergency.observe(MemoryObservation::Pressure, 10);
        assert_eq!(pressure, MemoryCondition::Pressure);
        let renewed = pressure.observe(MemoryObservation::Emergency, 11);
        assert_eq!(renewed, MemoryCondition::Emergency { since: 11 });
    }
}
