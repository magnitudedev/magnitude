//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Draw, DrawKind, Row, RowHistory, Select, Shaping as RowShaping, Slot, ValidatedTargetBatch,
};
use crate::memory::{HoldingClass, HoldingId, MemoryHeap};
use crate::platform::{DomainReading, DomainRole};
use crate::programs::ProgramSubmission;
use crate::{
    AllocatedResources, AttestedPrograms, CapacityError, Completion, ComponentLoader,
    ExecutionPlan, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, GraphOutputTensor,
    GroupKey, HeadConditioning, HeadLaunchInputs, ImageRef, NativeGraphOutputLease,
    NativeGraphWorkspaceLease, Operation, Outcome, PoolClass, Priming, ProgramIdentity, RequestId,
    ResidentHead, ResidentVision, ResourceDomain, ResourceDomainId, ResourceKind, RowResult,
    SelectSpec, Selected, StateLaunchInputs, StateWork, TargetLaunchInputs, TokenId,
    TargetTokens, ValidatedHeadLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchInputs, WorkKind,
};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::{
    Holder, InFlightState, OwnedAdvanceResolution, OwnedStateAdvance, OwnedTailRelocation,
    RowDemand, SequenceState, StateCheckpoint, StateStore, StoreBindings, TentativeAdvance,
};
use seismic::Tensor;
mod draft;
mod family;
mod features;
mod head;
mod heap;
mod in_flight;
mod input;
mod lookahead;
mod ownership;
mod reconcile;
mod state;
mod target;
mod vision;

#[cfg(test)]
mod domain_tests;

pub use family::{NativeFamily, ProgramFamily};
pub use heap::{ClaimRefusal, DeviceHeap};
use in_flight::{decode_selected, PrimingFlight, TargetWork};
pub use in_flight::{HeadFlight, TargetFlight, VisionFlight};
pub use ownership::OpenRequirements;
pub use state::MemoryChargeReconciliation;
pub(crate) use target::accepted_history;
pub use target::TargetHostTiming;

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
    /// A domain the device uses has headroom at or below its planning
    /// reserve; growth waits while reclaimable holdings are released.
    Reclaim,
    Input(String),
    State(magnitude_state::Error),
    Submit(crate::SubmitError),
    Device(crate::DeviceError),
    Invariant(crate::InvariantError),
}

impl DomainError {
    fn invariant(detail: impl Into<String>) -> Self {
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
            Self::Reclaim => f.write_str("memory headroom is at or below the planning reserve"),
            Self::Input(message) => f.write_str(message),
            Self::State(error) => error.fmt(f),
            Self::Submit(error) => error.fmt(f),
            Self::Device(error) => error.fmt(f),
            Self::Invariant(error) => error.fmt(f),
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
            ClaimRefusal::Reclaim { .. } => Self::Reclaim,
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
pub struct StateBindings<F: ProgramFamily = NativeFamily> {
    target: StoreBindings,
    head: Option<StoreBindings>,
    /// The step queued behind the last flight, with its drafter priming. It
    /// is the only work that holds bindings between flights, so it lives
    /// here and is resolved before bindings change.
    lookahead: Option<lookahead::Lookahead<F>>,
}

impl<F: ProgramFamily> StateBindings<F> {
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
pub enum SubmitFailure<F: ProgramFamily = NativeFamily> {
    /// Refused before anything reached the device; accepted state is
    /// unchanged and the bindings return.
    Refused(DomainError, StateBindings<F>),
    /// The device or a domain invariant failed. The bindings are gone, so no
    /// binding can change again.
    Failed(DomainError),
}

impl<F: ProgramFamily> SubmitFailure<F> {
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
    fn with<F: ProgramFamily>(self, bindings: StateBindings<F>) -> SubmitFailure<F> {
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
    advance: Option<OwnedStateAdvance>,
    /// A prompt chunk's drafter entry, committed whole with the chunk.
    primed: Option<OwnedStateAdvance>,
    rows: usize,
    committed_rows: usize,
    kind: WorkKind,
    physical_duration: Duration,
    image: Option<ImageRef>,
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
    /// The operations claim the queued lookahead step: nothing is reserved.
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

pub enum ReservedResources {
    Target(TargetGraphReservation),
    Head(
        NativeGraphWorkspaceLease,
        NativeGraphOutputLease,
        Vec<OwnedStateAdvance>,
    ),
    Vision(NativeGraphWorkspaceLease, NativeGraphOutputLease),
}

/// Complete target capacity is owned before launch construction. Decoder
/// graphs include checked recurrent state copies; readout has its own lease.
pub enum TargetGraphReservation {
    Launch(TargetLaunchReservation),
    /// The operations claim the queued lookahead step's slots, in operation
    /// order.
    Claim(Vec<usize>),
}

pub struct TargetLaunchReservation {
    advances: Vec<OwnedStateAdvance>,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_outputs: [NativeGraphOutputLease; 2],
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
}

/// Everything needed to make a request resident at a position without
/// recomputing the rows before it: both numerical sequence lanes and the
/// features of the media spans that straddle the position. It holds no
/// request's input: which path it represents is the prefix cache's record.
pub struct ResumeState {
    target: StateCheckpoint,
    head: Option<StateCheckpoint>,
    features: BTreeMap<String, FeatureRef>,
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
    /// prefill chunk). Consumed by that group's provisioning and lookahead.
    planned: Vec<Operation>,
    /// Identity of the next target flight.
    next_flight: u64,
    /// Log lookahead queues, claims and orphans (read once at start).
    trace_lookahead: bool,
}

impl ExecutorDomain<NativeFamily> {
    /// Classify the allocations that are already physically committed before
    /// any request reservation begins. The byte total comes from the actual
    /// graph pools and state stores; Seismic remains the charge authority.
    pub fn register_allocated_holdings(&mut self) -> Result<HoldingId, String> {
        self.refresh_memory().map_err(|error| error.to_string())?;
        let static_bytes = self.static_holding_bytes()?;
        let holding = self
            .memory
            .borrow_mut()
            .heap_mut()
            .insert(static_bytes, HoldingClass::Model)
            .map_err(|error| format!("static memory holding rejected: {error:?}"))?;
        self.static_holding = Some(holding);
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
        if head_loader.is_some() != head_state.is_some() {
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
    /// bound and the head store's span limit.
    fn head_class_limits(&self) -> Result<crate::batching::ClassLimits, DomainError> {
        let store = self
            .head_store
            .as_ref()
            .ok_or_else(|| DomainError::invariant("a head launch without a head store"))?;
        Ok(crate::batching::ClassLimits {
            rows: self.execution.policy().limits().max_launch_rows,
            segments: store.max_span_limit(),
        })
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
        for charge in [
            self.family.target_weight_bytes()?,
            self.family.prepared_program_bytes()?,
            self.family.target_constant_bytes()?,
        ] {
            bytes = bytes
                .checked_add(charge)
                .ok_or_else(|| "static holding byte count overflow".to_owned())?;
        }
        Ok(bytes)
    }

    pub(super) fn sync_optional_holding(&mut self) -> Result<(), String> {
        let target_weights = self.family.target_weight_bytes()?;
        let head_weights = self
            .head_loader
            .as_ref()
            .map(|loader| {
                loader
                    .resident_bytes()?
                    .checked_sub(target_weights)
                    .ok_or("head residency cache omits a target allocation")
            })
            .transpose()?
            .unwrap_or(0);
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

    pub(super) fn sync_static_holding(&mut self) -> Result<(), String> {
        let Some(holding) = self.static_holding else {
            return Ok(());
        };
        let bytes = self.static_holding_bytes()?;
        self.memory
            .borrow_mut()
            .heap_mut()
            .resize(holding, bytes)
            .map_err(|error| format!("static memory holding update rejected: {error:?}"))
    }

    /// Observe every domain the device uses, set Seismic's allocation
    /// ceiling from the readings, and refresh the heap's view from the same
    /// readings. Seismic remains the physical charge authority; the heap only
    /// classifies ownership and issues claims. A failed observation revokes
    /// unspent grants and leaves the heap Blind.
    pub fn refresh_memory(&self) -> Result<Vec<DomainReading>, DomainError> {
        let readings = self
            .memory
            .borrow_mut()
            .refresh()
            .map_err(|error| DomainError::Blind(error.to_string()))?;
        // A released slab can remain charged while an outside tensor view
        // retains it. Its final release happens when that view is dropped, so
        // reconcile this holding on every fresh charge observation.
        let retired_storage = self.target_store.has_retired_storage()
            || self
                .head_store
                .as_ref()
                .is_some_and(|store| store.has_retired_storage());
        if let (Some(holding), true) = (self.static_holding, retired_storage) {
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

    pub fn requirements(
        &self,
        bindings: &StateBindings<F>,
        operations: &[Operation],
    ) -> Result<DomainRequirements, DomainError> {
        if let Some(class) = self.claim_class(bindings, operations) {
            return Ok(DomainRequirements {
                lane: ReservationLane::Target,
                pool: PoolClass::Target(class),
                state_demand: Vec::new(),
                successor_banks: 0,
                priming_demand: Vec::new(),
                priming_banks: 0,
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
                priming_demand.extend(state.demands(prime.tokens.len()));
                priming_banks += 1;
            }
        }
        let (pool, state_demand, successor_banks) = match lane {
            ReservationLane::Target | ReservationLane::Head => {
                let mut rows = 0usize;
                let mut state_demand = Vec::new();
                let mut segments = 1usize;
                let mut demand = crate::batching::Demand::NONE;
                let mut seen = BTreeSet::new();
                for operation in operations {
                    operation.validate().map_err(|error| error.to_string())?;
                    if !seen.insert(operation.request()) {
                        return Err(DomainError::Input("reservation repeats a request".into()));
                    }
                    rows = rows
                        .checked_add(operation.row_count())
                        .ok_or_else(|| DomainError::invariant("reservation row count overflow"))?;
                    demand |= operation.demand();
                    let state = match lane {
                        ReservationLane::Target => self.target.get(&operation.request()),
                        ReservationLane::Head => self.head.get(&operation.request()),
                        _ => unreachable!(),
                    }
                    .ok_or_else(|| DomainError::Input("reservation request is not idle".into()))?;
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
                    state_demand.extend(state.demands(operation.row_count()));
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
                    operations.len(),
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
                        patch_rows: image.patches(),
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
                let readout = self.resources.target_readout_graph();
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
                available(
                    Some(graph.available_workspace()),
                    Some(graph.available_output()),
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
        if requirements.lane != ReservationLane::Vision {
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

    /// Reserve capacity for one group: the queued lookahead's slots when the
    /// group claims it, otherwise its provisioned state, advances and launch
    /// leases.
    pub fn reserve(
        &mut self,
        bindings: &mut StateBindings<F>,
        operations: &[Operation],
    ) -> Result<DomainReservation, DomainError> {
        self.refresh_memory()?;
        let requirements = self.requirements(bindings, operations)?;
        if requirements.claim {
            let slots = self
                .claim_slots(bindings, operations)
                .ok_or_else(|| DomainError::invariant("a claimable group lost its lookahead"))?;
            return Ok(DomainReservation {
                requirements,
                resources: ReservedResources::Target(TargetGraphReservation::Claim(slots)),
            });
        }
        // Commit the selected group's numerical backing before a lazy
        // component import. Provisioning grants the physical growth peak
        // through the heap before any backing allocation.
        self.provision(bindings, operations)?;
        self.sync_static_holding().map_err(DomainError::Input)?;
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
        let invariant = |detail: &str, error: CapacityError| {
            DomainError::invariant(format!(
                "{detail} changed after availability check: {error}"
            ))
        };
        let resources = match requirements.lane {
            ReservationLane::Target => {
                let graph = self.resources.target_graph();
                let graph_workspace = graph
                    .acquire_workspace()
                    .map_err(|error| invariant("target graph workspace", error))?;
                let graph_outputs = [
                    graph
                        .acquire_output()
                        .map_err(|error| invariant("target graph output", error))?,
                    graph
                        .acquire_output()
                        .map_err(|error| invariant("target graph output", error))?,
                ];
                let readout = self.resources.target_readout_graph();
                let readout_workspace = readout
                    .acquire_workspace()
                    .map_err(|error| invariant("target readout workspace", error))?;
                let readout_output = readout
                    .acquire_output()
                    .map_err(|error| invariant("target readout output", error))?;
                ReservedResources::Target(TargetGraphReservation::Launch(TargetLaunchReservation {
                    advances: Vec::with_capacity(operations.len()),
                    graph_workspace,
                    graph_outputs,
                    readout_workspace,
                    readout_output,
                    priming: None,
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
                    Vec::with_capacity(operations.len()),
                )
            }
            ReservationLane::Vision => {
                let graph = self.resources.vision_graph().expect("checked vision graph");
                ReservedResources::Vision(
                    graph
                        .acquire_workspace()
                        .map_err(|error| invariant("vision graph workspace", error))?,
                    graph
                        .acquire_output()
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
                if let [Operation::Forward {
                    request,
                    prime: Some(prime),
                    ..
                }] = operations
                {
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
            ReservedResources::Head(_, _, advances) => {
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
    ) -> (Self, StateBindings<F>) {
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
            head: BTreeMap::new(),
            input: BTreeMap::new(),
            lane_identities,
            selection_read: None,
            target_timing: None,
            trace_host_steps: std::env::var_os("MAGNITUDE_TRACE_HOST_STEP").is_some(),
            planned: Vec::new(),
            next_flight: 0,
            trace_lookahead: std::env::var_os("MAGNITUDE_TRACE_LOOKAHEAD").is_some(),
        };
        let bindings = StateBindings {
            target: target_state,
            head: head_state,
            lookahead: None,
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
        }
    }

    /// Host timing of the most recently finished target step.
    pub fn target_timing(&self) -> Option<TargetHostTiming> {
        self.target_timing
    }
}
