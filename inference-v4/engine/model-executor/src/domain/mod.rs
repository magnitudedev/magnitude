//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Draw, DrawKind, Row, Select, Shaping as RowShaping, Slot, ValidatedTargetBatch,
};
use crate::memory::{HoldingClass, HoldingId, MemoryBand, MemoryHeap, MemoryObservation};
use crate::platform::{DomainReading, DomainRole, MemoryReserves};
use crate::programs::ProgramSubmission;
use crate::{
    AllocatedResources, AttestedPrograms, CapacityError, Completion, ComponentLoader,
    ExecutionPlan, FeatureReader, FeatureRef, FeatureRows, FeatureSpan, GroupKey, HeadLaunchInputs,
    ImageRef, NativeGraphOutputLease, NativeGraphWorkspaceLease, Operation, Outcome, PoolClass,
    ProgramIdentity, RequestId, ResidentHead, ResidentVision, ResourceDomain, ResourceDomainId,
    ResourceKind, RowResult, Selected, StateLaunchInputs, StateWork, TargetLaunchInputs,
    TargetTokens, ValidatedHeadLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchInputs, WorkKind,
};
use magnitude_model_contracts::{ModelDefinition, PreparedModelInput, TextCoordinateSemantics};
use magnitude_model_state::{
    Holder, InFlightState, OwnedAdvanceResolution, OwnedCompaction, OwnedCompactionPreparation,
    OwnedStateAdvance, SequenceState, StateCheckpoint, StateStore, TentativeAdvance,
};
use seismic::Tensor;
mod family;
mod features;
mod head;
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
use in_flight::decode_selected;
pub use in_flight::{HeadFlight, TargetFlight, VisionFlight};
pub use ownership::{OpenRequirements, OpenReservation};
pub use state::MemoryChargeReconciliation;
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
    State(magnitude_model_state::Error),
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
impl From<magnitude_model_state::Error> for DomainError {
    fn from(error: magnitude_model_state::Error) -> Self {
        match error {
            magnitude_model_state::Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationCapacity {
                    required,
                    available,
                },
            )) => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: u64::try_from(&required).unwrap_or(u64::MAX),
                available,
            }),
            magnitude_model_state::Error::Tensor(seismic::TensorError::Execution(
                seismic::ExecutionError::AllocationFailed(_),
            )) => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::DeviceMemory,
                required: 1,
                available: 0,
            }),
            magnitude_model_state::Error::Capacity {
                required,
                available_bytes,
            } => Self::Capacity(crate::CapacityError {
                resource: crate::ResourceKind::StateRows,
                required,
                available: available_bytes,
            }),
            magnitude_model_state::Error::BanksExhausted { .. } => {
                Self::Capacity(crate::CapacityError {
                    resource: crate::ResourceKind::RecurrentBanks,
                    required: 1,
                    available: 0,
                })
            }
            other => Self::State(other),
        }
    }
}

/// One request's immutable numerical result and its still-owned physical
/// transaction. Cloning the view never clones the transaction.
pub struct PendingOperationOutcome {
    request: RequestId,
    outcome: Outcome,
    advance: Option<OwnedStateAdvance>,
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
    state_rows: usize,
    successor_banks: usize,
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
    pub fn state_rows(&self) -> usize {
        self.state_rows
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
}

impl DomainReservation {
    pub fn requirements(&self) -> &DomainRequirements {
        &self.requirements
    }
    pub fn into_resources(self) -> ReservedResources {
        self.resources
    }
}

#[derive(Clone)]
struct InputImage {
    image: ImageRef,
    features: Option<FeatureRef>,
}

#[derive(Clone)]
struct RequestInput {
    input: PreparedModelInput,
    images: BTreeMap<String, InputImage>,
}

/// Retains both numerical sequence lanes and admitted media claims for a
/// fork, restoration, or cross-request retention entry.
pub struct DomainCheckpoint {
    target: StateCheckpoint,
    head: Option<StateCheckpoint>,
    input: Option<RequestInput>,
}

impl DomainCheckpoint {
    pub fn position(&self) -> usize {
        self.target.position()
    }

    /// Bytes released if exactly this set of checkpoints were dropped: target
    /// and head history rows and recurrent banks that nothing outside the set
    /// references (a shared prefix counts once, and not at all while a live
    /// request or another checkpoint shares it), plus the set's distinct
    /// encoded media.
    pub fn exclusive_bytes(set: &[&DomainCheckpoint]) -> Result<u64, String> {
        let Some(first) = set.first() else {
            return Ok(0);
        };
        let lane = |holders: Vec<Holder<'_>>, store: &Rc<StateStore>| {
            store
                .exclusive_bytes(&holders)
                .map_err(|error| error.to_string())
        };
        let mut total = lane(
            set.iter()
                .map(|checkpoint| Holder::Checkpoint(&checkpoint.target))
                .collect(),
            first.target.store(),
        )?;
        if let Some(head) = &first.head {
            let heads = set
                .iter()
                .map(|checkpoint| {
                    checkpoint
                        .head
                        .as_ref()
                        .map(Holder::Checkpoint)
                        .ok_or("checkpoint set mixes head and headless state")
                })
                .collect::<Result<Vec<_>, _>>()?;
            total = total
                .checked_add(lane(heads, head.store())?)
                .ok_or("checkpoint byte count overflow")?;
        }
        let mut media: Vec<&Tensor> = Vec::new();
        for input in set
            .iter()
            .filter_map(|checkpoint| checkpoint.input.as_ref())
        {
            for feature in input
                .images
                .values()
                .filter_map(|image| image.features.as_ref())
            {
                let tensor = feature
                    .allocation()
                    .tensor()
                    .map_err(|error| error.to_string())?;
                if media.iter().all(|seen| !seen.shares_allocation(tensor)) {
                    media.push(tensor);
                }
            }
        }
        media.iter().try_fold(total, |total, tensor| {
            total
                .checked_add(tensor.storage_bytes())
                .ok_or_else(|| "checkpoint media byte count overflow".to_owned())
        })
    }
}

/// The live memory authority of one domain's device: the worker's Seismic
/// catalog, the host's threshold policy and the allocation domain's stable
/// capacity.
struct DomainMemoryPolicy {
    catalog: seismic::DeviceCatalog,
    reserves: MemoryReserves,
    capacity_bytes: u64,
}

/// Native executor with no erased stage or executor type parameters. The
/// accepted map is vacant while that request's advance is in flight.
pub struct ExecutorDomain<F: ProgramFamily = NativeFamily> {
    execution: Rc<ExecutionPlan>,
    definition: Rc<ModelDefinition>,
    resources: AllocatedResources,
    domain: ResourceDomain,
    memory_policy: Option<DomainMemoryPolicy>,
    memory: Rc<RefCell<MemoryHeap>>,
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
    fatal: Option<DomainError>,
    /// Group identities of the target, head and encoder executables.
    lane_identities: [ProgramIdentity; 3],
    /// When the last target selection was read to the host.
    selection_read: Option<Instant>,
    target_timing: Option<TargetHostTiming>,
    /// Log each finished target step's host timing (read once at start).
    trace_host_steps: bool,
    /// The step queued behind the last submitted target step, until claimed
    /// or orphaned (see `lookahead`).
    lookahead: Option<lookahead::Lookahead<F::TargetSubmission>>,
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
            .insert(static_bytes, HoldingClass::Model)
            .map_err(|error| format!("static memory holding rejected: {error:?}"))?;
        self.static_holding = Some(holding);
        Ok(holding)
    }

    pub fn new(
        execution: Rc<ExecutionPlan>,
        definition: Rc<ModelDefinition>,
        device: Rc<seismic::Device>,
        programs: Rc<AttestedPrograms>,
        resources: AllocatedResources,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        resident: crate::ResidentTarget,
        target_store: Rc<StateStore>,
        head_store: Option<Rc<StateStore>>,
    ) -> Result<Self, String> {
        // A model's draft head is enabled only when its method drafts.
        if (head_loader.is_some() && definition.head.is_none())
            || vision_loader.is_some() != definition.vision.is_some()
        {
            return Err("component loaders differ from enabled model components".into());
        }
        if head_loader.is_some() != head_store.is_some() {
            return Err("head state arena differs from enabled head component".into());
        }
        let family = NativeFamily::new(programs, resident, definition.geometry.clone())?;
        Ok(Self::with_family(
            execution,
            definition,
            resources,
            device,
            target_store,
            head_store,
            head_loader,
            vision_loader,
            family,
        ))
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
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
        Ok(bytes)
    }

    pub(super) fn sync_optional_holding(&mut self) -> Result<(), String> {
        if !self.memory.borrow().is_observed() {
            return Ok(());
        }
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
        let bound_constants = self.family.bound_constant_bytes()?;
        let bytes = head_weights
            .checked_add(vision_weights)
            .and_then(|weights| weights.checked_add(bound_constants))
            .ok_or_else(|| "optional holding byte count overflow".to_owned())?;
        match (self.optional_holding, bytes) {
            (Some(id), 0) => {
                self.memory
                    .borrow_mut()
                    .remove(id)
                    .map_err(|error| format!("optional holding release rejected: {error:?}"))?;
                self.optional_holding = None;
            }
            (Some(id), bytes) => self
                .memory
                .borrow_mut()
                .resize(id, bytes)
                .map_err(|error| format!("optional holding update rejected: {error:?}"))?,
            (None, 0) => {}
            (None, bytes) => {
                let id = self
                    .memory
                    .borrow_mut()
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
            .resize(holding, bytes)
            .map_err(|error| format!("static memory holding update rejected: {error:?}"))
    }

    /// Install the worker's selected-device catalog and the host's threshold
    /// policy for every later live memory observation. The allocation
    /// domain's stable capacity is fixed here.
    pub fn install_memory_policy(
        &mut self,
        catalog: seismic::DeviceCatalog,
        reserves: MemoryReserves,
    ) -> Result<(), String> {
        let host = catalog
            .host_memory_status()
            .map_err(|error| error.to_string())?;
        let capacity_bytes = crate::platform::fit_capacities(
            &catalog.topology(),
            self.domain.device().info(),
            &host,
            &reserves,
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|(role, _)| *role == DomainRole::Allocation)
        .expect("fit capacities include the allocation domain")
        .1
        .capacity_bytes;
        self.memory_policy = Some(DomainMemoryPolicy {
            catalog,
            reserves,
            capacity_bytes,
        });
        Ok(())
    }

    /// Observe every domain the device uses, set Seismic's allocation
    /// ceiling from the readings, and refresh the heap's view from the same
    /// readings. Seismic remains the physical charge authority; the heap only
    /// classifies ownership and issues claims. A failed observation revokes
    /// unspent grants and leaves the heap Blind.
    pub fn refresh_memory(&self) -> Result<Vec<DomainReading>, DomainError> {
        let policy = self
            .memory_policy
            .as_ref()
            .ok_or_else(|| DomainError::invariant("memory policy has not been installed"))?;
        let device = self.domain.device();
        let readings =
            crate::platform::refresh_device_ceiling(&policy.catalog, device, &policy.reserves);
        let (available_bytes, band) = match &readings {
            Ok(readings) => (
                readings
                    .iter()
                    .find(|reading| reading.role == DomainRole::Allocation)
                    .expect("readings include the allocation domain")
                    .ceiling_bytes,
                MemoryBand::from(crate::platform::band_of(readings)),
            ),
            Err(_) => (0, MemoryBand::Blind),
        };
        self.memory
            .borrow_mut()
            .observe(MemoryObservation {
                capacity_bytes: policy.capacity_bytes,
                available_bytes,
                charged_bytes: device.memory_usage().charged,
                band,
            })
            .map_err(|error| {
                DomainError::invariant(format!("memory observation rejected: {error:?}"))
            })?;
        readings.map_err(|error| DomainError::Blind(error.to_string()))
    }

    pub fn memory(&self) -> std::cell::Ref<'_, MemoryHeap> {
        self.memory.borrow()
    }

    pub fn requirements(
        &self,
        operations: &[Operation],
    ) -> Result<DomainRequirements, DomainError> {
        self.healthy()?;
        if let Some(class) = self.claim_class(operations) {
            return Ok(DomainRequirements {
                lane: ReservationLane::Target,
                pool: PoolClass::Target(class),
                state_rows: 0,
                successor_banks: 0,
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
        let limits = self.execution.policy().limits();
        let (pool, state_rows, successor_banks) = match lane {
            ReservationLane::Target | ReservationLane::Head => {
                let mut rows = 0usize;
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
                    segments = segments.max(state.history_ranges().len());
                }
                let class =
                    crate::LaunchClass::covering(rows, segments, demand, limits.max_batch_rows)
                        .map_err(|error| error.to_string())?;
                (
                    if lane == ReservationLane::Target {
                        PoolClass::Target(class)
                    } else {
                        PoolClass::Head(class)
                    },
                    rows,
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
                    0,
                    0,
                )
            }
        };
        Ok(DomainRequirements {
            lane,
            pool,
            state_rows,
            successor_banks,
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
        let store = if requirements.lane == ReservationLane::Head {
            self.head_store.as_ref()
        } else {
            Some(&self.target_store)
        };
        if requirements.state_rows != 0 {
            let store = store.expect("head requirements require a head store");
            let rows = store.available_rows();
            if rows < requirements.state_rows {
                return Err(CapacityError {
                    resource: ResourceKind::StateRows,
                    required: requirements.state_rows as u64,
                    available: rows as u64,
                });
            }
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

    pub fn reserve(&mut self, operations: &[Operation]) -> Result<DomainReservation, DomainError> {
        self.refresh_memory()?;
        let requirements = self.requirements(operations)?;
        if requirements.claim {
            let slots = self
                .claim_slots(operations)
                .ok_or_else(|| DomainError::invariant("a claimable group lost its lookahead"))?;
            return Ok(DomainReservation {
                requirements,
                resources: ReservedResources::Target(TargetGraphReservation::Claim(slots)),
            });
        }
        // Commit the selected group's numerical backing before a lazy
        // component import. Provisioning grants the physical growth peak
        // through the heap before any backing allocation.
        self.provision(operations)?;
        self.sync_static_holding().map_err(DomainError::Input)?;
        match requirements.lane {
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
        self.sync_static_holding().map_err(DomainError::Input)?;
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
                TargetLaunchReservation { advances, .. },
            )) => {
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
    pub fn with_family(
        execution: Rc<ExecutionPlan>,
        definition: Rc<ModelDefinition>,
        resources: AllocatedResources,
        device: Rc<seismic::Device>,
        target_store: Rc<StateStore>,
        head_store: Option<Rc<StateStore>>,
        head_loader: Option<ComponentLoader<ResidentHead>>,
        vision_loader: Option<ComponentLoader<ResidentVision>>,
        family: F,
    ) -> Self {
        let id = resources.domain().clone();
        let lane_identities = ["target", "head", "vision"].map(|lane| {
            ProgramIdentity::new(format!("{id}:{lane}"))
                .expect("a lane identity names its lane and is never empty")
        });
        Self {
            execution,
            definition,
            resources,
            domain: ResourceDomain::new(id, device),
            memory_policy: None,
            memory: Rc::new(RefCell::new(MemoryHeap::new())),
            static_holding: None,
            optional_holding: None,
            target_store,
            head_store,
            family,
            head_loader,
            vision_loader,
            target: BTreeMap::new(),
            head: BTreeMap::new(),
            input: BTreeMap::new(),
            fatal: None,
            lane_identities,
            selection_read: None,
            target_timing: None,
            trace_host_steps: std::env::var_os("MAGNITUDE_TRACE_HOST_STEP").is_some(),
            lookahead: None,
            next_flight: 0,
            trace_lookahead: std::env::var_os("MAGNITUDE_TRACE_LOOKAHEAD").is_some(),
        }
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
    pub fn fatal_error(&self) -> Option<&DomainError> {
        self.fatal.as_ref()
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

    fn healthy(&self) -> Result<(), DomainError> {
        self.fatal
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }

    /// Record a broken executor-owned relationship. Request validation errors
    /// never use this path; once ownership has moved into a submission, a
    /// mismatch means the numerical domain can no longer continue safely.
    fn fatal_invariant(&mut self, detail: impl Into<String>) -> DomainError {
        let error = DomainError::invariant(detail);
        self.fatal = Some(error.clone());
        error
    }

    /// State reconciliation failures occur after request preflight and retain
    /// their state/capacity category while poisoning this physical domain.
    fn fatal_state(&mut self, error: magnitude_model_state::Error) -> DomainError {
        let error = DomainError::from(error);
        self.fatal = Some(error.clone());
        error
    }
}
