//! The physical execution domain owns accepted sequence state and the native
//! programs bound to its one attested device. A completed result remains a
//! proposal until its owner supplies an exact reconciliation decision.

use crate::batching::{
    Draw, DrawKind, Row, Select, Shaping as RowShaping, Slot, ValidatedTargetBatch,
};
use crate::programs::ProgramSubmission;
use crate::{
    AllocatedResources, AttestedPrograms, CapacityError, Completion, ComponentLoader,
    ExecutionPlan, FeatureRef, FeatureRetainer, FeatureSpan, GroupKey, HeadLaunchInputs, ImageRef,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, Operation, Outcome, PoolClass,
    ProgramIdentity, ProjectionLaunchInputs, ProjectionRequest, RequestId, ResidentHead,
    ResidentVision, ResourceDomain, ResourceDomainId, ResourceKind, RetainedFeatureSpan, RowResult,
    Selected, StateLaunchInputs, StateWork, TargetGraphOutputLease, TargetGraphWorkspaceLease,
    TargetLaunchInputs, ValidatedHeadLaunch, ValidatedProjectionLaunch, ValidatedStateLaunch,
    ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchInputs, WorkKind,
};
use magnitude_model_contracts::{ModelDefinition, PreparedModelInput, TextCoordinateSemantics};
use magnitude_model_state::{
    OwnedAdvanceResolution, OwnedRepairAdvance, OwnedStateAdvance, SequenceState, StateCheckpoint,
    StateStore,
};
mod family;
mod head;
mod in_flight;
mod input;
mod ownership;
mod project;
mod reconcile;
mod retention;
mod state;
mod target;
mod vision;

#[cfg(test)]
mod domain_tests;

pub use family::{NativeFamily, ProgramFamily};
use in_flight::decode_selected;
pub use in_flight::{HeadFlight, ProjectFlight, StateFlight, TargetFlight, VisionFlight};
pub use ownership::{OpenRequirements, OpenReservation};

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::{Duration, Instant},
};

/// The logical owner chooses this only after inspecting every completed row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalDecision {
    pub accepted_rows: usize,
    pub head_prefix: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum DomainError {
    Capacity(crate::CapacityError),
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
    slot: Option<Slot>,
    conditioning: Option<crate::ConditioningRef>,
    conditioning_slices: Vec<crate::ConditioningSlice>,
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

pub enum PhysicalResolution {
    Committed,
    Repair { request: RequestId, rows: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationLane {
    Target,
    Head,
    Project,
    Vision,
    Repair,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainRequirements {
    lane: ReservationLane,
    pool: PoolClass,
    state_rows: usize,
    successor_banks: usize,
    secondary_pool: Option<PoolClass>,
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

pub struct DomainReservation {
    requirements: DomainRequirements,
    operations: Vec<Operation>,
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
    Repair(ReservedRepair),
}

/// Complete target capacity is owned before launch construction. Decoder
/// graphs include checked recurrent state copies; readout has its own lease.
pub struct TargetGraphReservation {
    advances: Vec<OwnedStateAdvance>,
    graph_workspace: NativeGraphWorkspaceLease,
    graph_outputs: [NativeGraphOutputLease; 2],
    readout_workspace: NativeGraphWorkspaceLease,
    readout_output: NativeGraphOutputLease,
}

pub struct ReservedRepair {
    class_rows: usize,
    state_graph_workspace: NativeGraphWorkspaceLease,
    graph_workspace: TargetGraphWorkspaceLease,
    graph_outputs: [TargetGraphOutputLease; 2],
    pending: PendingRepair,
}

impl DomainReservation {
    pub fn requirements(&self) -> &DomainRequirements {
        &self.requirements
    }
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
    pub fn into_parts(self) -> (Vec<Operation>, ReservedResources) {
        (self.operations, self.resources)
    }
}

struct PendingRepair {
    advance: OwnedRepairAdvance,
    slot: Slot,
    conditioning: Option<crate::ConditioningRef>,
    conditioning_slices: Vec<crate::ConditioningSlice>,
    head_prefix: Option<usize>,
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

    pub fn retained_bytes(&self) -> Result<u64, String> {
        let mut total = self.target.retained_bytes()?;
        if let Some(head) = &self.head {
            total = total
                .checked_add(head.retained_bytes()?)
                .ok_or("checkpoint byte count overflow")?;
        }
        if let Some(input) = &self.input {
            for image in input.images.values() {
                if let Some(feature) = &image.features {
                    total = total
                        .checked_add(
                            feature
                                .allocation()
                                .tensor()
                                .map_err(|error| error.to_string())?
                                .storage_bytes(),
                        )
                        .ok_or("checkpoint media byte count overflow")?;
                }
            }
        }
        Ok(total)
    }
}

/// Native executor with no erased stage or executor type parameters. The
/// accepted map is vacant while that request's advance is in flight.
pub struct ExecutorDomain<F: ProgramFamily = NativeFamily> {
    execution: Rc<ExecutionPlan>,
    definition: Rc<ModelDefinition>,
    resources: AllocatedResources,
    domain: ResourceDomain,
    target_store: Rc<StateStore>,
    head_store: Option<Rc<StateStore>>,
    family: F,
    head_loader: Option<ComponentLoader<ResidentHead>>,
    vision_loader: Option<ComponentLoader<ResidentVision>>,
    target: BTreeMap<RequestId, SequenceState>,
    head: BTreeMap<RequestId, SequenceState>,
    head_pending: BTreeMap<RequestId, OwnedStateAdvance>,
    input: BTreeMap<RequestId, RequestInput>,
    repairs: BTreeMap<RequestId, PendingRepair>,
    retained_used: Rc<Cell<u64>>,
    fatal: Option<DomainError>,
}

impl ExecutorDomain<NativeFamily> {
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
        if head_loader.is_some() != definition.head.is_some()
            || vision_loader.is_some() != definition.vision.is_some()
        {
            return Err("component loaders differ from enabled model components".into());
        }
        if head_loader.is_some() != head_store.is_some() {
            return Err("head state arena differs from enabled head component".into());
        }
        let family = NativeFamily::new(
            programs,
            resident,
            definition.geometry.clone(),
            target_store.clone(),
        )?;
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
    pub fn requirements(
        &self,
        operations: &[Operation],
    ) -> Result<DomainRequirements, DomainError> {
        self.healthy()?;
        let first = operations
            .first()
            .ok_or_else(|| DomainError::invariant("empty reservation"))?;
        let lane = match first {
            Operation::Forward { .. } => ReservationLane::Target,
            Operation::Head { .. } => ReservationLane::Head,
            Operation::Project { .. } => ReservationLane::Project,
            Operation::Encode { .. } => ReservationLane::Vision,
            Operation::Repair { .. } => ReservationLane::Repair,
        };
        if operations.iter().any(|operation| {
            !matches!(
                (lane, operation),
                (ReservationLane::Target, Operation::Forward { .. })
                    | (ReservationLane::Head, Operation::Head { .. })
                    | (ReservationLane::Project, Operation::Project { .. })
                    | (ReservationLane::Vision, Operation::Encode { .. })
                    | (ReservationLane::Repair, Operation::Repair { .. })
            )
        }) {
            return Err(DomainError::invariant(
                "reservation crosses numerical lanes",
            ));
        }
        let limits = self.execution.policy().limits();
        let (pool, secondary_pool, state_rows, successor_banks) = match lane {
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
                        Operation::Head {
                            tokens,
                            conditioning,
                            ..
                        } if conditioning.count != tokens.len()
                            || conditioning.features.domain() != self.domain.id() =>
                        {
                            return Err(DomainError::Input(
                                "head conditioning differs from physical rows or resource domain"
                                    .into(),
                            ));
                        }
                        _ => {}
                    }
                    if lane == ReservationLane::Head
                        && self.head_pending.contains_key(&operation.request())
                    {
                        return Err(DomainError::Input(
                            "head reservation has a suspended advance".into(),
                        ));
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
                    None,
                    rows,
                    operations.len(),
                )
            }
            ReservationLane::Project => {
                let mut seen = BTreeSet::new();
                for operation in operations {
                    let request = operation.request();
                    if !seen.insert(request)
                        || !self.head.contains_key(&request)
                            && !self.head_pending.contains_key(&request)
                    {
                        return Err(DomainError::Input(
                            "projection request is repeated or not open".into(),
                        ));
                    }
                }
                let rows = operations
                    .iter()
                    .try_fold(0usize, |total, operation| match operation {
                        Operation::Project { features, .. } => total
                            .checked_add(features.allocation().rows())
                            .ok_or("projection reservation rows overflow"),
                        _ => unreachable!(),
                    })
                    .map_err(DomainError::from)?;
                let class = crate::LaunchClass::covering(
                    rows,
                    operations.len(),
                    crate::batching::Demand::SELECT,
                    limits.max_batch_rows,
                )
                .map_err(|error| error.to_string())?;
                (PoolClass::Head(class), None, 0, 0)
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
                    None,
                    0,
                    0,
                )
            }
            ReservationLane::Repair => {
                let [Operation::Repair { rows, .. }] = operations else {
                    return Err(DomainError::Input(
                        "repair reservation must contain one request".into(),
                    ));
                };
                let [Operation::Repair { request, .. }] = operations else {
                    unreachable!()
                };
                let pending = self.repairs.get(request).ok_or_else(|| {
                    DomainError::Input("request has no pending recurrent repair".into())
                })?;
                let segments = pending.advance.history_ranges().len().max(1);
                let target = crate::LaunchClass::covering(
                    *rows,
                    segments,
                    crate::batching::Demand::NONE,
                    limits.max_batch_rows,
                )
                .map_err(|error| error.to_string())?;
                (
                    PoolClass::State { rows: *rows },
                    Some(PoolClass::Target(target)),
                    0,
                    0,
                )
            }
        };
        Ok(DomainRequirements {
            lane,
            pool,
            secondary_pool,
            state_rows,
            successor_banks,
        })
    }

    pub fn can_reserve(&self, requirements: &DomainRequirements) -> Result<(), CapacityError> {
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
            ReservationLane::Head | ReservationLane::Project => {
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
            ReservationLane::Repair => {
                let graph = self.resources.target_graph();
                if graph.available_workspace() == 0 {
                    return Err(CapacityError {
                        resource: ResourceKind::Workspace,
                        required: 1,
                        available: 0,
                    });
                }
                if graph.available_output() < 2 {
                    return Err(CapacityError {
                        resource: ResourceKind::Output,
                        required: 2,
                        available: graph.available_output() as u64,
                    });
                }
                if self.resources.state_graph().available_workspace() == 0 {
                    return Err(CapacityError {
                        resource: ResourceKind::Workspace,
                        required: 1,
                        available: 0,
                    });
                }
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

    pub fn reserve(
        &mut self,
        operations: Vec<Operation>,
    ) -> Result<DomainReservation, DomainError> {
        let requirements = self.requirements(&operations)?;
        match requirements.lane {
            ReservationLane::Head | ReservationLane::Project if !self.family.head_is_bound() => {
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
                ReservedResources::Target(TargetGraphReservation {
                    advances: Vec::with_capacity(operations.len()),
                    graph_workspace,
                    graph_outputs,
                    readout_workspace,
                    readout_output,
                })
            }
            ReservationLane::Head | ReservationLane::Project => {
                let graph = self.resources.head_graph().expect("checked head graph");
                ReservedResources::Head(
                    graph
                        .acquire_workspace()
                        .map_err(|error| invariant("head graph workspace", error))?,
                    graph
                        .acquire_output()
                        .map_err(|error| invariant("head graph output", error))?,
                    Vec::with_capacity(if requirements.lane == ReservationLane::Head {
                        operations.len()
                    } else {
                        0
                    }),
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
            ReservationLane::Repair => {
                let PoolClass::State { rows: class_rows } = requirements.pool else {
                    unreachable!()
                };
                let graph_workspace = self
                    .resources
                    .target_graph()
                    .acquire_workspace()
                    .map_err(|error| invariant("repair graph workspace", error))?;
                let state_graph_workspace = self
                    .resources
                    .state_graph()
                    .acquire_workspace()
                    .map_err(|error| invariant("state graph workspace", error))?;
                let graph_output_0 = self
                    .resources
                    .target_graph()
                    .acquire_output()
                    .map_err(|error| invariant("repair graph output", error))?;
                let graph_output_1 = self
                    .resources
                    .target_graph()
                    .acquire_output()
                    .map_err(|error| invariant("repair graph output", error))?;
                let pending = self
                    .repairs
                    .remove(&operations[0].request())
                    .expect("reservation preflight established repair ownership");
                ReservedResources::Repair(ReservedRepair {
                    class_rows,
                    state_graph_workspace,
                    graph_workspace,
                    graph_outputs: [graph_output_0, graph_output_1],
                    pending,
                })
            }
        };
        let mut resources = resources;
        match &mut resources {
            ReservedResources::Target(TargetGraphReservation { advances, .. }) => {
                for operation in &operations {
                    let request = operation.request();
                    let state = self
                        .target
                        .remove(&request)
                        .expect("reservation preflight established target ownership");
                    match OwnedStateAdvance::begin(state, operation.row_count()) {
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
            ReservedResources::Head(_, _, advances)
                if requirements.lane == ReservationLane::Head =>
            {
                for operation in &operations {
                    let request = operation.request();
                    let state = self
                        .head
                        .remove(&request)
                        .expect("reservation preflight established head ownership");
                    match OwnedStateAdvance::begin(state, operation.row_count()) {
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
            operations,
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
        Self {
            execution,
            definition,
            resources,
            domain: ResourceDomain::new(id, device),
            target_store,
            head_store,
            family,
            head_loader,
            vision_loader,
            target: BTreeMap::new(),
            head: BTreeMap::new(),
            head_pending: BTreeMap::new(),
            input: BTreeMap::new(),
            repairs: BTreeMap::new(),
            retained_used: Rc::new(Cell::new(0)),
            fatal: None,
        }
    }

    pub fn resource_identity(&self) -> &ResourceDomainId {
        self.domain.id()
    }
    pub fn execution_path(&self) -> crate::ExecutionPath {
        self.execution.policy().path()
    }
    pub fn resources(&self) -> &ResourceDomain {
        &self.domain
    }
    pub fn fatal_error(&self) -> Option<&DomainError> {
        self.fatal.as_ref()
    }

    pub fn group_key(&self, operation: &Operation) -> Result<GroupKey, String> {
        operation.validate().map_err(|error| error.to_string())?;
        let lane = match operation.executable() {
            crate::ExecutableKind::Target => "target",
            crate::ExecutableKind::Head => "head",
            crate::ExecutableKind::Encoder => "vision",
        };
        Ok(GroupKey {
            program_identity: ProgramIdentity::new(format!("{}:{lane}", self.domain.id()))?,
            executable: operation.executable(),
            commitment: operation.commitment(),
        })
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
