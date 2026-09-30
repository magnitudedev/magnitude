//! Serialized ownership of logical generation and one executor resource domain.
use super::{
    domain::{
        DomainFlight, ExecutorDomain, OperationGroup, ResumeState, group, requirements,
        submit_group,
    },
    policy::{
        AvailabilityEpoch, Operation as ScheduledOperation, Phase, Scheduler, Selection,
        ServiceLimits, Victim, order_victims,
    },
    prefix_cache::{MIN_BRANCH_GAIN, MIN_PREFIX_HIT, PrefixCache, PrefixCacheCapacity, PrefixPath},
    publication::{
        CapacityResource, ModelUnloadCause, PhysicalTimings, PublicationPermit, PublicationSender,
        PublicationWake, PublicationWakeKind, PublishError, RequestError, ServiceCapacityError,
    },
    round_driver::{RoundError, lower_round, reconcile_forward},
};
use magnitude_executor::{
    DomainError, DomainRequirements, HeadPhase, InvariantError, MemoryChargeReconciliation,
    NativeFamily, OpenRequirements, Operation, Outcome, PhysicalDecision, ProgramFamily, RequestId,
    ResourceKind, ResourcePlan, SubmitError, WorkKind, platform::DomainRole,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{
    DetailedUsage, FinishReason, Generation, OutputToken, RoundStart, WaitReason,
};
use magnitude_state::ShrinkPolicy;
use std::{
    collections::{BTreeMap, VecDeque},
    time::Duration,
};

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
    let boundary = record.generation.prompt().len().checked_sub(1)?;
    (record.prefix_cache
        && !record.prefill_retained
        && boundary >= MIN_PREFIX_HIT
        && record.generation.layout().boundary(boundary))
    .then_some(boundary)
}

/// Whether `generation` is resident, unfinished, and not yet past `position`.
fn prefilling_toward(generation: &Generation, position: usize) -> bool {
    generation.is_resident()
        && generation.finish_reason().is_none()
        && generation.resident_position() < position
}

struct Record {
    generation: Generation,
    /// Image encodes the request must complete before its next round, from
    /// its latest residency.
    encodes: VecDeque<Operation>,
    waiting_since: u64,
    service_ns: u64,
    physical_timings: PhysicalTimings,
    error: Option<RequestError>,
    capacity: Option<(AvailabilityEpoch, u64, u64)>,
    preemption_debt: u32,
    protected_until: Option<usize>,
    retire_on_completion: bool,
    /// Whether the request resumes from and contributes to the prefix cache.
    prefix_cache: bool,
    /// The cached prefix the request's residency forked, protected while
    /// work derived from it is submitted.
    prefix_source: Option<u64>,
    prefill_retained: bool,
    terminal_retained: bool,
    /// A planned branch point: the prompt position where this request
    /// diverges from a path the engine holds. Prefill stops there so the
    /// cache can retain a state every later divergent request resumes from.
    branch: Option<usize>,
    /// A prompt position a live peer is prefilling toward and will retain:
    /// this request stays non-resident until that prefix is cached (or no
    /// peer computes it any more), so the prefix is computed once.
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
    /// The request cannot open: the typed failure it would otherwise have
    /// reported (invalid input, capacity after the release order, device
    /// failure, or a failed engine invariant).
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Submitted,
    Reconciled,
    Waiting,
    Idle,
    Progress,
}

/// How long Reclaim may persist after releases are exhausted, and how long
/// observations may fail continuously, before the model unloads.
const MEMORY_ESCALATION_NS: u64 = 1_000_000_000;

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
    Unloading,
}

impl MemoryCondition {
    fn observe(self, observation: MemoryObservation, now: u64) -> Self {
        if self == Self::Unloading {
            return self;
        }
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
                _ => Self::Blind { since: now },
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
    domain: ExecutorDomain<F>,
    scheduler: Scheduler,
    records: BTreeMap<RequestId, Record>,
    batch: Option<Batch<F>>,
    epoch: AvailabilityEpoch,
    next_id: u64,
    now: u64,
    /// The classified failure that stopped the owner; every live request
    /// terminated with it and every later admission is refused with it.
    fatal: Option<RequestError>,
    prefix_cache: PrefixCache<ResumeState>,
    reclaim_release_pending: bool,
    memory_condition: MemoryCondition,
    /// Per domain role, the least memory a capacity-blocked request still
    /// needs from that domain's ceiling after every release was exhausted.
    memory_deficits: Vec<MemoryDeficit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemoryDeficit {
    role: DomainRole,
    required: u64,
}

impl<F: ProgramFamily> Owner<F> {
    pub fn new(domain: ExecutorDomain<F>, limits: ServiceLimits) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(domain, limits, PrefixCacheCapacity::disabled())
    }

    pub fn with_prefix_cache_capacity(
        domain: ExecutorDomain<F>,
        limits: ServiceLimits,
        capacity: PrefixCacheCapacity,
    ) -> Result<Self, String> {
        Ok(Self {
            domain,
            scheduler: Scheduler::new(limits)?,
            records: BTreeMap::new(),
            batch: None,
            epoch: AvailabilityEpoch::default(),
            next_id: 1,
            now: 0,
            fatal: None,
            prefix_cache: PrefixCache::new(capacity),
            reclaim_release_pending: false,
            memory_condition: MemoryCondition::Normal,
            memory_deficits: Vec::new(),
        })
    }

    pub fn with_resource_plan(
        domain: ExecutorDomain<F>,
        limits: ServiceLimits,
        plan: &ResourcePlan,
    ) -> Result<Self, String> {
        Self::with_prefix_cache_capacity(
            domain,
            limits,
            PrefixCacheCapacity::from_resource_plan(plan),
        )
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

    /// Observe every domain the model uses. In Reclaim, release in order
    /// until the band is Normal; unload once Reclaim persists for the
    /// escalation interval after releases are exhausted, or once
    /// observations have failed for that interval. Releases and unload never
    /// run under a flight: they wait for its completion.
    fn observe_periodic_memory(&mut self, now: u64) -> Result<(), String> {
        self.time(now)?;
        if self.memory_condition == MemoryCondition::Unloading {
            return Ok(());
        }
        let observation = Self::memory_observation(self.domain.probe_memory())?;
        self.enter_memory_condition(self.memory_condition.observe(observation, now))?;
        if !matches!(self.memory_condition, MemoryCondition::Reclaim { .. }) {
            self.reclaim_release_pending = false;
            if observation == MemoryObservation::Normal {
                self.reopen_memory_deficits()?;
            }
            return Ok(());
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.active.is_some())
        {
            self.reclaim_release_pending = true;
            return Ok(());
        }
        // Releases are measured against a fresh reading; a Blind escalation
        // has none, so it goes straight to the unload decision.
        if observation == MemoryObservation::Reclaim && self.release_for_reclaim()? {
            self.memory_condition = self.memory_condition.released(now);
            let after = Self::memory_observation(self.domain.probe_memory())?;
            self.enter_memory_condition(self.memory_condition.observe(after, now))?;
        }
        if self.memory_condition.should_unload(now) {
            self.begin_memory_unload();
        }
        Ok(())
    }

    /// Enter `next`. Memory returning to Normal is an availability change:
    /// every request waiting on memory retries.
    fn enter_memory_condition(&mut self, next: MemoryCondition) -> Result<(), String> {
        let recovered =
            self.memory_condition != MemoryCondition::Normal && next == MemoryCondition::Normal;
        self.memory_condition = next;
        if recovered {
            self.epoch.advance()?;
        }
        Ok(())
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
            self.epoch.advance()?;
        }
        Ok(())
    }

    /// Record that blocked work needs `required` bytes from `role`'s domain,
    /// keeping the least requirement per role so the earliest retry is kept.
    fn record_memory_deficit(&mut self, role: DomainRole, required: u64) {
        match self
            .memory_deficits
            .iter_mut()
            .find(|deficit| deficit.role == role)
        {
            Some(deficit) => deficit.required = deficit.required.min(required),
            None => self.memory_deficits.push(MemoryDeficit { role, required }),
        }
    }

    /// Block `requests` on a capacity deficit until availability changes,
    /// when a live request outside `excluded` can still finish, drain or
    /// release; a memory deficit also reopens once another process frees the
    /// memory. With nothing that could change, fail them with the typed
    /// deficit. Returns whether they are blocked.
    fn block_on_capacity(
        &mut self,
        requests: &[RequestId],
        excluded: &[RequestId],
        resource: ResourceKind,
        required: u64,
        available: u64,
    ) -> Result<bool, String> {
        let can_change = self.records.iter().any(|(id, record)| {
            !excluded.contains(id)
                && (record.generation.finish_reason().is_some()
                    || record.generation.output_len() > 0
                    || record.capacity.is_none())
        });
        if !can_change {
            for &request in requests {
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
            return Ok(false);
        }
        for request in requests {
            if let Some(record) = self.records.get_mut(request) {
                record.capacity = Some((self.epoch, required, available));
            }
        }
        match resource {
            ResourceKind::DeviceMemory => {
                self.record_memory_deficit(DomainRole::Allocation, required)
            }
            ResourceKind::HostStaging => self.record_memory_deficit(DomainRole::Staging, required),
            _ => {}
        }
        Ok(true)
    }

    fn memory_observation(result: Result<(), DomainError>) -> Result<MemoryObservation, String> {
        match result {
            Ok(()) => Ok(MemoryObservation::Normal),
            Err(DomainError::Reclaim) => Ok(MemoryObservation::Reclaim),
            Err(DomainError::Blind(_)) => Ok(MemoryObservation::Blind),
            Err(error) => Err(error.to_string()),
        }
    }

    /// Whether a fresh reading is still in the Reclaim band. A failed reading
    /// cannot measure a release, so it stops releasing.
    fn reclaim_needed(&mut self) -> Result<bool, String> {
        Ok(Self::memory_observation(self.domain.probe_memory())? == MemoryObservation::Reclaim)
    }

    /// Release in the Reclaim order until every domain is back above its
    /// planning reserve: surplus state backing, retained prefixes (least
    /// recently used first), dormant optional components, then preemption of
    /// each eligible victim once. Preempted requests replay only after the
    /// band is Normal again, so no victim is chosen twice. Returns whether
    /// anything was released.
    fn release_for_reclaim(&mut self) -> Result<bool, String> {
        let mut released = self.release_surplus()?;
        while self.reclaim_needed()? {
            if !self.prefix_cache.evict_one() {
                break;
            }
            // Eviction drops claims, but committed rows and banks remain
            // charged until the stores shrink.
            self.release_surplus()?;
            released = true;
        }
        if self.reclaim_needed()? {
            released |= self.domain.release_idle_optional_components()? != 0;
        }
        while self.reclaim_needed()? {
            if !self.evict_victims(&[], false)? {
                break;
            }
            self.release_surplus()?;
            released = true;
        }
        // Preemption can leave the optional components dormant.
        if self.reclaim_needed()? {
            released |= self.domain.release_idle_optional_components()? != 0;
        }
        self.epoch.advance()?;
        Ok(released)
    }

    /// Release state backing beyond what the stores need, including idle
    /// state. Returns whether any bytes were released.
    fn release_surplus(&mut self) -> Result<bool, String> {
        let shrunk = self
            .domain
            .shrink_state(ShrinkPolicy::Reclaim)
            .map_err(|error| error.to_string())?;
        let idle = self.domain.reclaim_idle()?;
        Ok(shrunk != 0 || idle != 0)
    }

    fn release_deferred_reclaim(&mut self) -> Result<(), String> {
        if !self.reclaim_release_pending {
            return Ok(());
        }
        self.reclaim_release_pending = false;
        self.observe_periodic_memory(self.now)
    }

    /// Admit a fresh generation with its prepared input. Unless it waits for
    /// a live peer to compute a shared prefix, it becomes resident now, from
    /// the deepest cached prefix of its path when `prefix_cache` allows.
    pub fn admit(
        &mut self,
        generation: Generation,
        input: PreparedModelInput,
        prefix_cache: bool,
        now: u64,
    ) -> Result<RequestId, AdmissionError> {
        self.time(now)
            .map_err(AdmissionError::invariant("service clock"))?;
        if self.memory_condition == MemoryCondition::Unloading {
            return Err(AdmissionError::ModelUnloaded {
                cause: ModelUnloadCause::MemoryPressure,
            });
        }
        if matches!(self.memory_condition, MemoryCondition::Reclaim { .. }) {
            return Err(AdmissionError::MemoryReclaim);
        }
        if let Some(error) = &self.fatal {
            return Err(AdmissionError::Refused(error.clone()));
        }
        if generation.is_resident() || generation.usage().completion_tokens != 0 {
            return Err(AdmissionError::invariant("admission")(
                "only a fresh generation can be admitted".into(),
            ));
        }
        // A Blind refusal must leave accepted peers and their numerical
        // holdings intact.
        self.domain.probe_memory().map_err(AdmissionError::from)?;
        let id = RequestId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            AdmissionError::invariant("service identity")("identity exhausted".into())
        })?;
        self.domain
            .install_input(id, input)
            .map_err(|error| AdmissionError::Refused(RequestError::Input(error)))?;
        let plan = if prefix_cache {
            self.plan_branches(PrefixPath::of(&generation), generation.resume_bound())
        } else {
            BranchPlan::default()
        };
        self.records.insert(
            id,
            Record {
                generation,
                encodes: VecDeque::new(),
                waiting_since: now,
                service_ns: 0,
                physical_timings: PhysicalTimings::default(),
                error: None,
                capacity: None,
                preemption_debt: 0,
                protected_until: None,
                retire_on_completion: false,
                prefix_cache,
                prefix_source: None,
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
        if plan.awaited.is_none() {
            if let Err(error) = self.make_resident(id) {
                self.records.remove(&id);
                if let Err(rollback) = self.domain.close(id) {
                    let fatal = format!("admission failed ({error}); rollback failed ({rollback})");
                    self.fail_all(fatal.clone());
                    return Err(AdmissionError::invariant("admission rollback")(fatal));
                }
                return Err(error);
            }
        }
        self.epoch
            .advance()
            .map_err(AdmissionError::invariant("availability epoch"))?;
        Ok(id)
    }

    /// The single transition that makes a request resident: from the deepest
    /// cached prefix of its path below its resume bound, or from fresh state.
    /// Admission, a waiting request whose prefix is ready, and an evicted
    /// request all come through here.
    fn make_resident(&mut self, id: RequestId) -> Result<(), AdmissionError> {
        let invariant = |detail: String| AdmissionError::invariant("residency")(detail);
        let record = self
            .records
            .get(&id)
            .ok_or_else(|| invariant("unknown request".into()))?;
        let hit = if record.prefix_cache {
            self.prefix_cache
                .lookup(
                    PrefixPath::of(&record.generation),
                    record.generation.resume_bound(),
                )
                .map_err(AdmissionError::invariant("prefix cache"))?
        } else {
            None
        };
        let opened = match hit {
            Some(hit) => {
                let cached = self
                    .prefix_cache
                    .entry(hit)
                    .map_err(AdmissionError::invariant("prefix cache"))?;
                self.domain.open_state(id, Some(cached.state()))
            }
            None => {
                self.ensure_open_capacity()?;
                self.domain.open_state(id, None)
            }
        };
        let encodes = match opened {
            Ok(encodes) => encodes,
            Err(error) => {
                if let Some(fatal) = self.domain.fatal_error().cloned() {
                    self.fail_domain_error(fatal);
                }
                return Err(AdmissionError::from(error));
            }
        };
        let record = self.records.get_mut(&id).expect("known request");
        let resumed = match hit {
            Some(hit) => self.prefix_cache.entry(hit).and_then(|cached| {
                record
                    .generation
                    .resume_at(Some((hit.position(), cached.method())))
            }),
            None => record.generation.resume_at(None),
        };
        if let Err(error) = resumed {
            return Err(match self.domain.release_state(&[id]) {
                Ok(_) => invariant(error),
                Err(rollback) => {
                    let fatal = format!("{error}; executor rollback failed ({rollback})");
                    self.fail_all(fatal.clone());
                    invariant(fatal)
                }
            });
        }
        record.encodes.extend(encodes);
        record.prefix_source = hit.map(|hit| hit.id());
        Ok(())
    }

    /// Branch points for a new request's `path`: the deepest position where
    /// it diverges from a cached path or a live request's prompt. A live
    /// request that has not yet prefilled to its divergence from this one
    /// stops there too, so the shared prefix is cached by whichever request
    /// reaches it first and every later request resumes at it instead of
    /// recomputing.
    fn plan_branches(&mut self, path: PrefixPath, bound: usize) -> BranchPlan {
        let worthwhile = |point: usize, from: usize, prompt: usize| {
            point >= MIN_PREFIX_HIT && point >= from + MIN_BRANCH_GAIN && point < prompt
        };
        let resumed = self.prefix_cache.deepest(path, bound);
        let cached = self.prefix_cache.shared_boundary(path);
        let mut deepest = cached;
        // The deepest shared prefix a live peer is still prefilling toward
        // and will cache at its branch point.
        let mut in_flight = 0;
        for record in self
            .records
            .values_mut()
            .filter(|record| record.prefix_cache)
        {
            let live = PrefixPath::of(&record.generation);
            let shared = path.shared_with_prompt_of(live);
            deepest = deepest.max(shared);
            if record.branch.is_none()
                && worthwhile(
                    shared,
                    record.generation.resident_position(),
                    record.generation.prompt().len(),
                )
            {
                record.branch = Some(shared);
            }
            if record.branch == Some(shared) && prefilling_toward(&record.generation, shared) {
                in_flight = in_flight.max(shared);
            }
        }
        if !worthwhile(deepest, resumed, path.len()) {
            return BranchPlan::default();
        }
        if in_flight == deepest && deepest > cached {
            // A peer computes this prefix now: wait for it to be cached.
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
    /// a planned branch point at `position`.
    fn prefix_pending(&self, position: usize, except: RequestId) -> bool {
        self.records.iter().any(|(&id, record)| {
            id != except
                && record.prefix_cache
                && record.branch == Some(position)
                && prefilling_toward(&record.generation, position)
        })
    }

    /// A waiting request stops waiting once the prefix it awaits is cached,
    /// or once no peer computes it any more. It then becomes resident
    /// through the ordinary residency path, from the deepest cached prefix.
    fn release_awaited(&mut self) -> Result<(), String> {
        let waiting = self
            .records
            .iter()
            .filter_map(|(&id, record)| record.awaited_prefix.map(|position| (id, position)))
            .collect::<Vec<_>>();
        let mut released = false;
        for (id, awaited) in waiting {
            let generation = &self.records[&id].generation;
            let cached = self
                .prefix_cache
                .deepest(PrefixPath::of(generation), generation.resume_bound())
                >= awaited;
            if cached || !self.prefix_pending(awaited, id) {
                self.records
                    .get_mut(&id)
                    .expect("waiting request")
                    .awaited_prefix = None;
                released = true;
            }
        }
        if released {
            self.epoch.advance()?;
        }
        Ok(())
    }

    fn ensure_open_capacity(&mut self) -> Result<(), AdmissionError> {
        let requirements = self.domain.open_requirements();
        let mut current = self.open_capacity(requirements);
        if matches!(current, Err(DomainError::Capacity(_))) {
            self.domain
                .shrink_state(ShrinkPolicy::Reclaim)
                .map_err(AdmissionError::from)?;
            current = self.open_capacity(requirements);
            if matches!(current, Err(DomainError::Capacity(_))) {
                self.domain
                    .reclaim_idle()
                    .map_err(AdmissionError::invariant("idle state release"))?;
                current = self.open_capacity(requirements);
            }
            while matches!(current, Err(DomainError::Capacity(_))) {
                if !self.prefix_cache.evict_one() {
                    break;
                }
                self.domain
                    .shrink_state(ShrinkPolicy::Reclaim)
                    .map_err(AdmissionError::from)?;
                current = self.open_capacity(requirements);
            }
            if matches!(current, Err(DomainError::Capacity(_))) {
                self.domain
                    .release_idle_optional_components()
                    .map_err(AdmissionError::invariant("optional component release"))?;
                current = self.open_capacity(requirements);
            }
            while matches!(current, Err(DomainError::Capacity(_))) {
                if !self
                    .evict_victims(&[], true)
                    .map_err(AdmissionError::invariant("preemption"))?
                {
                    break;
                }
                self.domain
                    .shrink_state(ShrinkPolicy::Reclaim)
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
    pub fn fatal_error(&self) -> Option<&RequestError> {
        self.fatal.as_ref()
    }
    pub fn completed_service_ns(&self) -> u128 {
        self.scheduler.completed_service_ns()
    }
    pub fn cached_prefixes(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Seismic's charge reconciled against every state holder the owner
    /// keeps: resident requests and cached prefixes; submitted work is
    /// counted through its stores' transactions.
    pub fn reconcile_memory_charge(&self) -> Result<MemoryChargeReconciliation, String> {
        let cached = self.prefix_cache.states().collect::<Vec<_>>();
        self.domain.reconcile_memory_charge(&cached)
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
            // Waiting for a peer to compute a shared prefix is not preemption.
            Some(WaitReason::Residency) if record.awaited_prefix.is_some() => Status::Runnable,
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
                    let needs_reconciliation = record.prefix_cache
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
        if self.fatal.is_none() {
            self.fatal = Some(error.clone());
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
            // Accepted output still precedes the terminal failure on this
            // request's stream. Its reserved permits must survive until that
            // output has drained.
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
        self.release_awaited()?;
        let mut candidates = Vec::new();
        for (&id, record) in &mut self.records {
            if record.awaited_prefix.is_some() {
                continue;
            }
            // Becoming resident is growth: it waits for memory to be Normal,
            // whose return advances the epoch.
            if !record.generation.is_resident() && self.memory_condition != MemoryCondition::Normal
            {
                continue;
            }
            if record
                .capacity
                .is_some_and(|(epoch, _, _)| !self.epoch.changed_since(epoch))
            {
                continue;
            }
            record.capacity = None;
            let reconciliation = record.prefix_cache
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
        // Requests take rows from one phase-wide token budget in selection
        // order. Additional ready requests wait for the next round.
        let mut remaining_rows = match selection.phase() {
            Phase::Prefill => self.scheduler.limits().prefill_tokens,
            Phase::Decode => self.scheduler.limits().decode_tokens,
        };
        let mut started = 0;
        let mut operations = Vec::new();
        for &request in selection.requests() {
            if remaining_rows == 0 {
                break;
            }
            // Share the remaining rows across requests still eligible in this
            // round. A lone request can use the whole budget, while peers get
            // compatible chunks instead of the first request consuming it.
            let waiting = selection.requests().len() - started;
            let allowance = remaining_rows.div_ceil(waiting);
            started += 1;
            match self.start_request(request, allowance) {
                Ok(mut next) => {
                    let rows = next
                        .iter()
                        .filter(|operation| {
                            matches!(
                                operation,
                                Operation::Forward { .. } | Operation::Head { .. }
                            )
                        })
                        .map(Operation::row_count)
                        .sum::<usize>();
                    remaining_rows = remaining_rows.saturating_sub(rows);
                    operations.append(&mut next);
                }
                Err(detail) => {
                    // The request was admitted; failing to start its next
                    // round is an engine failure, not a client error.
                    self.fail_request(
                        request,
                        RequestError::Invariant(InvariantError {
                            context: "generation round start",
                            detail,
                        }),
                    );
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

    /// Make a non-resident request (evicted, or done waiting for a peer's
    /// prefix) resident. Returns false when it cannot be now: short of
    /// capacity it is blocked (or failed) like any capacity-blocked work; in
    /// Reclaim or without a memory reading, the owner records that reading
    /// and residency waits until memory is Normal again.
    fn restore(&mut self, request: RequestId) -> Result<bool, String> {
        let observation = match self.make_resident(request) {
            Ok(()) => return Ok(true),
            Err(AdmissionError::Refused(RequestError::Capacity(ServiceCapacityError {
                resource: CapacityResource::Execution(resource),
                required,
                available,
            }))) => {
                self.block_on_capacity(&[request], &[request], resource, required, available)?;
                return Ok(false);
            }
            Err(AdmissionError::MemoryReclaim) => MemoryObservation::Reclaim,
            Err(AdmissionError::MemoryObservationUnavailable(_)) => MemoryObservation::Blind,
            Err(error) => return Err(error.to_string()),
        };
        self.enter_memory_condition(self.memory_condition.observe(observation, self.now))?;
        Ok(false)
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
        if needs_open && !self.restore(request)? {
            return Ok(Vec::new());
        }
        let record = self.records.get_mut(&request).expect("known request");
        if !record.encodes.is_empty() {
            return Ok(record.encodes.drain(..).collect());
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
            self.release_deferred_reclaim()?;
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

    /// The prompt chunk a lone prefill group's request submits next, when it
    /// is known now, so the domain queues it behind this group. A chunk
    /// ending at a planned branch or retention point is followed by that
    /// retention, not planned past.
    fn planned_prefill(&self, operations: &[Operation]) -> Result<Vec<Operation>, String> {
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
        let Some(record) = self.records.get(request) else {
            return Ok(Vec::new());
        };
        if !record.encodes.is_empty() {
            return Ok(Vec::new());
        }
        let end = position + tokens.len();
        let stops = [record.branch, prompt_boundary(record)]
            .into_iter()
            .flatten()
            .filter(|&stop| stop > *position)
            .collect::<Vec<_>>();
        if stops.iter().any(|&stop| stop <= end) {
            return Ok(Vec::new());
        }
        let allowance = stops
            .iter()
            .map(|stop| stop - end)
            .fold(self.scheduler.limits().prefill_tokens.max(1), usize::min);
        Ok(record
            .generation
            .planned_prefill(*request, allowance)?
            .into_iter()
            .collect())
    }

    fn submit_next(&mut self) -> Result<Step, String> {
        let mut queued = self.batch.as_mut().unwrap().queued.pop_front().unwrap();
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
        let planned = self.planned_prefill(queued.operations())?;
        self.domain.plan_successors(planned);
        let first_provision = match self.domain.provision(queued.operations()) {
            Ok(()) => Ok(()),
            Err(
                error @ (DomainError::Capacity(_) | DomainError::Blind(_) | DomainError::Reclaim),
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
            // available. A demand deficit first releases state backing beyond
            // what the stores need (another store may hold the device memory
            // this one needs to grow into); policy then releases
            // least-recently-used retention only until the requirement fits,
            // then idle residency, then eligible live victims. No launch or
            // state transaction exists while these scheduling decisions run.
            match self.domain.shrink_state(ShrinkPolicy::Reclaim) {
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
                if !self.prefix_cache.evict_one() {
                    break;
                }
                if let Err(error) = self.domain.shrink_state(ShrinkPolicy::Reclaim) {
                    self.fail_domain_error(error);
                    return Ok(Step::Progress);
                }
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
            if matches!(&current, Err(DomainError::Capacity(_))) {
                self.domain.release_idle_optional_components()?;
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
            if matches!(&current, Err(DomainError::Capacity(_))) {
                queued = match queued.split_last() {
                    Ok((leading, trailing)) => {
                        let batch = self.batch.as_mut().unwrap();
                        batch.queued.push_front(trailing);
                        batch.queued.push_front(leading);
                        return Ok(Step::Progress);
                    }
                    Err(queued) => queued,
                };
            }
            let selected = self.batch.as_ref().unwrap().selection.requests().to_vec();
            while matches!(&current, Err(DomainError::Capacity(_))) {
                if !self.evict_victims(&selected, false)? {
                    break;
                }
                if let Err(error) = self.domain.shrink_state(ShrinkPolicy::Reclaim) {
                    self.fail_domain_error(error);
                    return Ok(Step::Progress);
                }
                current = self.provisioned_capacity(queued.operations(), &requirement);
            }
        }
        if matches!(&current, Err(DomainError::Reclaim)) {
            // Growth pauses in the Reclaim band. Keep the group, then run
            // the Reclaim releases now rather than at the next periodic
            // observation; no flight is active here.
            self.requeue_group(queued);
            self.observe_periodic_memory(self.now)?;
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
                );
            }
            Err(DomainError::Reclaim) => unreachable!("reclaim handled above"),
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
                let mut prefix_uses = Vec::new();
                for request in operations
                    .iter()
                    .map(Operation::request)
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    if let Some(source) = self
                        .records
                        .get(&request)
                        .and_then(|record| record.prefix_source)
                    {
                        if self.prefix_cache.begin_submitted_if_cached(source)? {
                            prefix_uses.push(source);
                        }
                    }
                }
                self.batch.as_mut().unwrap().active = Some(ActiveGroup {
                    flight,
                    operations,
                    prefix_uses,
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
            Err(DomainError::Reclaim) => {
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

    fn reserve_group_publications(
        &mut self,
        operations: &[Operation],
    ) -> Result<Option<BTreeMap<RequestId, Vec<PublicationPermit>>>, String> {
        let mut tokens = BTreeMap::<RequestId, usize>::new();
        for operation in operations {
            let count = match operation {
                Operation::Forward { .. } => self
                    .records
                    .get(&operation.request())
                    .ok_or("unknown publication request")?
                    .generation
                    .round_publication_bound()
                    .ok_or("forward has no suspended generation round")?,
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
        if self.block_on_capacity(&requests, &selected, resource, required, available)? {
            let batch = self.batch.as_mut().unwrap();
            batch.queued.push_front(queued);
            batch.blocked = Some(self.epoch);
            Ok(Step::Waiting)
        } else {
            Ok(Step::Progress)
        }
    }

    fn abort_pending(
        &mut self,
        pending: magnitude_executor::PendingOperationOutcome,
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
        for source in active.prefix_uses {
            self.prefix_cache.end_submitted(source)?;
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
                        Err(RoundError::Logical(detail)) => self.fail_request(
                            request,
                            RequestError::Invariant(InvariantError {
                                context: "generation round",
                                detail,
                            }),
                        ),
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
        pending: Vec<magnitude_executor::PendingOperationOutcome>,
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
            let operation = expected[&request];
            let correct = matches!(operation, Operation::Head { .. })
                && matches!(item.outcome(), Outcome::Head { .. });
            let Some(record) = self.records.get_mut(&request) else {
                self.abort_pending(item)?;
                continue;
            };
            let kind = match operation {
                Operation::Head {
                    phase: HeadPhase::Priming { .. },
                    ..
                } => WorkKind::Prefill,
                _ => item.kind(),
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
                Err(detail) => {
                    self.abort_pending(item)?;
                    self.fail_request(
                        request,
                        RequestError::Invariant(InvariantError {
                            context: "generation method transition",
                            detail,
                        }),
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
        self.cache_prefix(request, branch)
    }

    /// Cache the request's reconciled state as the prefix of its path ending
    /// at `position`, where it is resident.
    fn cache_prefix(&mut self, request: RequestId, position: usize) -> Result<(), String> {
        let state = self.domain.resume_state(request)?;
        if state.position() != position {
            return Err("cached prefix state differs from its path position".into());
        }
        let generation = &self
            .records
            .get(&request)
            .ok_or("unknown request")?
            .generation;
        let method = generation.method_checkpoint()?;
        if self
            .prefix_cache
            .retain(PrefixPath::of(generation), position, state, method)?
        {
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
        self.cache_prefix(request, boundary)?;
        self.records
            .get_mut(&request)
            .expect("known request")
            .prefill_retained = true;
        Ok(())
    }

    /// Cache a finished request's whole path, prompt and reply: the next turn
    /// of a conversation extends it.
    fn retain_terminal(&mut self, request: RequestId) -> Result<(), String> {
        let record = self.records.get(&request).ok_or("unknown request")?;
        if record.terminal_retained
            || !record.prefix_cache
            || !record.generation.is_resident()
            || !matches!(
                record.generation.finish_reason(),
                Some(FinishReason::Stop | FinishReason::Length | FinishReason::Context)
            )
        {
            return Ok(());
        }
        let end = PrefixPath::of(&record.generation).len();
        if record.generation.resident_position() != end {
            if record.generation.pending_reconciliation().is_some() {
                return Ok(());
            }
            return Err(
                "terminal numerical state is not reconciled at the accepted boundary".into(),
            );
        }
        self.cache_prefix(request, end)?;
        self.records
            .get_mut(&request)
            .expect("known request")
            .terminal_retained = true;
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
                || !record.encodes.is_empty()
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
            let executor_released = match self.domain.release_state(&set) {
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
    fn admission_ready(&self) -> bool {
        self.domain.open_growth_unblocked()
    }

    fn settle_completion(&mut self, now: u64) -> Result<(), String> {
        self.time(now)?;
        let complete = self
            .batch
            .as_mut()
            .and_then(|batch| batch.active.as_mut())
            .is_some_and(|active| active.flight.completion().is_complete());
        if !complete {
            return Err("completion notification preceded executor completion".into());
        }
        self.reconcile_active()?;
        self.release_deferred_reclaim()?;
        if self.memory_condition != MemoryCondition::Unloading
            && self
                .batch
                .as_ref()
                .is_some_and(|batch| batch.queued.is_empty())
        {
            self.finish_batch(now)?;
        }
        Ok(())
    }

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

    fn failure(&self) -> Option<&RequestError> {
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
        self.prefix_cache.evict_all();
        Ok(true)
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

    #[test]
    fn unloading_is_terminal() {
        for observation in [
            MemoryObservation::Normal,
            MemoryObservation::Reclaim,
            MemoryObservation::Blind,
        ] {
            assert_eq!(
                MemoryCondition::Unloading.observe(observation, 1),
                MemoryCondition::Unloading
            );
        }
        assert!(!MemoryCondition::Unloading.should_unload(u64::MAX));
    }
}
