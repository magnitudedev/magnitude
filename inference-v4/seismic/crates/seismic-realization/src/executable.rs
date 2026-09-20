//! Executable physical planning contract.
//!
//! Unlike the legacy flat physical graph, a plan alternative can only be
//! constructed by consuming every task, call, dependency, and result obligation
//! of one scheduling-normal logical task graph. Resolved execution is nested by
//! phase and launch; native encoders receive one complete kernel at a time.

use seismic_lang::{
    logical::{
        AxisOrder, CallBoundaryId, ChoiceId, DependencyId, LocalViewId, LogicalCompilationIdentity,
        LogicalDependency, LogicalDependencyKind, LogicalEndpoint, LogicalProgram,
        LogicalTaskGraph, OperandId, StorageRef, TaskGraphId, TaskId, Type,
    },
    precision::{NumericalAssessment, NumericalEffect},
    sym::Sym,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
};

macro_rules! ids {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);
    )+};
}

ids!(
    PhaseId,
    LaunchId,
    StorageId,
    BindingId,
    BindingGroupId,
    KernelValueId
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedStorageId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedPhaseId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedLaunchId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedBindingId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedBindingGroupId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResolvedKernelValueId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonEmpty<T> {
    head: T,
    tail: Vec<T>,
}

impl<T> NonEmpty<T> {
    pub fn new(head: T) -> Self {
        Self { head, tail: vec![] }
    }

    pub fn push(&mut self, value: T) {
        self.tail.push(value);
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.head).chain(self.tail.iter())
    }

    pub fn into_vec(self) -> Vec<T> {
        std::iter::once(self.head).chain(self.tail).collect()
    }

    pub fn len(&self) -> usize {
        1 + self.tail.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalAccess {
    pub storage: StorageId,
    pub mode: AccessMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InstructionResources {
    pub registers: u32,
    pub private_bytes: u64,
    pub workgroup_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstructionConsequences<C> {
    pub accesses: Vec<PhysicalAccess>,
    pub capability: Option<C>,
    pub numerical: Vec<NumericalEffect>,
    pub resources: InstructionResources,
}

/// A dialect instruction is already physical. Its consequences are derived
/// from the instruction value itself rather than attached in parallel vectors.
pub trait ExecutableDialect: Clone + Debug + PartialEq + Eq + 'static {
    type TemplateInstruction: Clone + Debug + PartialEq;
    type ResolvedInstruction: Clone + Debug + PartialEq;
    type TemplateLayout: Clone + Debug + PartialEq + Eq;
    type ResolvedLayout: Clone + Debug + PartialEq + Eq;
    type Capability: Clone + Debug + PartialEq + Eq + PartialOrd + Ord;

    fn consequences(
        instruction: &Self::TemplateInstruction,
    ) -> InstructionConsequences<Self::Capability>;
    fn resolve_instruction(
        instruction: &Self::TemplateInstruction,
        symbols: &BTreeMap<String, i64>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
    ) -> Result<Self::ResolvedInstruction, String>;
    fn resolve_layout(
        layout: &Self::TemplateLayout,
        symbols: &BTreeMap<String, i64>,
    ) -> Result<Self::ResolvedLayout, String>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueTransportTemplate {
    /// A semantic `Void` value has no physical channel.
    Void,
    Kernel(KernelValueId),
    /// Ordered representation planes for one logical value. Dense values have
    /// exactly one member; packed values preserve representation plane order.
    Storage(NonEmpty<StorageId>),
    Tuple(NonEmpty<Box<ValueTransportTemplate>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedValueTransport {
    Void,
    Kernel(ResolvedKernelValueId),
    Storage(NonEmpty<ResolvedStorageId>),
    Tuple(NonEmpty<Box<ResolvedValueTransport>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandTransportTemplate {
    pub operand: OperandId,
    pub transport: ValueTransportTemplate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedOperandTransport {
    pub operand: OperandId,
    pub transport: ResolvedValueTransport,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AxisMap {
    Grid {
        logical_axis: u32,
        workgroup_axis: u8,
        participant_axis: u8,
        mode: GridMapping,
    },
    SubgroupLane {
        logical_axis: u32,
    },
    Serial {
        logical_axis: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GridMapping {
    OnePass,
    GridStride { stride: Sym },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantMap {
    pub task: TaskId,
    pub axes: Vec<AxisMap>,
    pub logical_extents: Vec<Sym>,
    pub masks_inactive_participants: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierScope {
    Subgroup,
    Workgroup,
}

#[derive(Clone, Debug, PartialEq)]
enum KernelStepKind<D: ExecutableDialect> {
    MappedTask {
        task: TaskId,
        mapping: ParticipantMap,
        bindings: Vec<OperandTransportTemplate>,
        instructions: NonEmpty<D::TemplateInstruction>,
    },
    Barrier {
        dependency: DependencyId,
        scope: BarrierScope,
        transport: DependencyTransport,
    },
    Publish {
        output: u32,
        operand: OperandId,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelStep<D: ExecutableDialect> {
    kind: KernelStepKind<D>,
}

pub enum KernelStepView<'a, D: ExecutableDialect> {
    MappedTask {
        task: TaskId,
        mapping: &'a ParticipantMap,
        bindings: &'a [OperandTransportTemplate],
        instructions: &'a NonEmpty<D::TemplateInstruction>,
    },
    Barrier {
        dependency: DependencyId,
        scope: BarrierScope,
        transport: &'a DependencyTransport,
    },
    Publish {
        output: u32,
        operand: OperandId,
    },
}

impl<D: ExecutableDialect> KernelStep<D> {
    pub fn view(&self) -> KernelStepView<'_, D> {
        match &self.kind {
            KernelStepKind::MappedTask {
                task,
                mapping,
                bindings,
                instructions,
            } => KernelStepView::MappedTask {
                task: *task,
                mapping,
                bindings,
                instructions,
            },
            KernelStepKind::Barrier {
                dependency,
                scope,
                transport,
            } => KernelStepView::Barrier {
                dependency: *dependency,
                scope: *scope,
                transport,
            },
            KernelStepKind::Publish { output, operand } => KernelStepView::Publish {
                output: *output,
                operand: *operand,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replication {
    Once,
    PerSubgroup,
    PerWorkgroup,
    PerParticipant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageScope {
    External,
    Device,
    Workgroup,
    Participant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalAddressTemplate {
    DenseAffine {
        byte_offset: Sym,
        byte_strides: Vec<Sym>,
    },
    Representation {
        representation: String,
        plane: String,
        logical_strides: Vec<Sym>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedPhysicalAddress {
    DenseAffine {
        byte_offset: u64,
        byte_strides: Vec<u64>,
    },
    Representation {
        representation: String,
        plane: String,
        logical_strides: Vec<u64>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbiRole {
    Parameter {
        ordinal: u32,
        path: Vec<u32>,
        representation_plane: Option<String>,
    },
    Result {
        ordinal: u32,
        path: Vec<u32>,
        representation_plane: Option<String>,
    },
    InvocationResource {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalStorageProvenance {
    /// Tensor/effect storage identity. Scalar spill/ABI slots intentionally
    /// have no logical storage and are instead tied to `operand`.
    pub logical_storage: Option<StorageRef>,
    pub operand: Option<OperandId>,
    pub view: Option<LocalViewId>,
    pub subrange: PhysicalSubrangeTemplate,
    pub address: PhysicalAddressTemplate,
    pub abi: Option<AbiRole>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPhysicalStorageProvenance {
    pub logical_storage: Option<StorageRef>,
    pub operand: Option<OperandId>,
    pub view: Option<LocalViewId>,
    pub subrange: ResolvedPhysicalSubrange,
    pub address: ResolvedPhysicalAddress,
    pub abi: Option<AbiRole>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalSubrangeTemplate {
    pub byte_offset: Sym,
    pub bytes: Sym,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPhysicalSubrange {
    pub byte_offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageTemplate<D: ExecutableDialect> {
    pub id: StorageId,
    pub scope: StorageScope,
    pub replication: Replication,
    pub bytes: Sym,
    pub alignment: u64,
    pub layout: D::TemplateLayout,
    pub provenance: PhysicalStorageProvenance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingTemplate {
    pub id: BindingId,
    pub storage: StorageId,
    pub access: AccessMode,
    pub operand: Option<OperandId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingGroupKind {
    Direct,
    ArgumentTable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingGroupTemplate {
    pub id: BindingGroupId,
    pub kind: BindingGroupKind,
    pub slot: u32,
    pub members: NonEmpty<BindingTemplate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchTemplate {
    pub workgroups: [Sym; 3],
    pub participants_per_workgroup: [Sym; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelTemplate<D: ExecutableDialect> {
    pub participant_storage: Vec<StorageTemplate<D>>,
    pub workgroup_storage: Vec<StorageTemplate<D>>,
    pub steps: NonEmpty<KernelStep<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchTemplate<D: ExecutableDialect> {
    pub id: LaunchId,
    pub geometry: DispatchTemplate,
    pub binding_groups: Vec<BindingGroupTemplate>,
    pub kernel: KernelTemplate<D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhaseTemplate<D: ExecutableDialect> {
    pub id: PhaseId,
    pub predecessors: Vec<PhaseId>,
    pub launches: NonEmpty<LaunchTemplate<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceStoragePlan<D: ExecutableDialect> {
    pub allocations: Vec<StorageTemplate<D>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedStorage<D: ExecutableDialect> {
    pub id: ResolvedStorageId,
    pub scope: StorageScope,
    pub replication: Replication,
    pub bytes: u64,
    pub alignment: u64,
    pub layout: D::ResolvedLayout,
    pub provenance: ResolvedPhysicalStorageProvenance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedBinding {
    pub id: ResolvedBindingId,
    pub storage: ResolvedStorageId,
    pub access: AccessMode,
    pub operand: Option<OperandId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedBindingGroup {
    pub id: ResolvedBindingGroupId,
    pub kind: BindingGroupKind,
    pub slot: u32,
    pub members: NonEmpty<ResolvedBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedParticipantMap {
    pub task: TaskId,
    pub axes: Vec<ResolvedAxisMap>,
    pub logical_extents: Vec<u64>,
    pub masks_inactive_participants: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedAxisMap {
    Grid {
        logical_axis: u32,
        workgroup_axis: u8,
        participant_axis: u8,
        mode: ResolvedGridMapping,
    },
    SubgroupLane {
        logical_axis: u32,
    },
    Serial {
        logical_axis: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedGridMapping {
    OnePass,
    GridStride { stride: u64 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedKernelStep<D: ExecutableDialect> {
    MappedTask {
        task: TaskId,
        mapping: ResolvedParticipantMap,
        bindings: Vec<ResolvedOperandTransport>,
        instructions: NonEmpty<D::ResolvedInstruction>,
    },
    Barrier {
        dependency: DependencyId,
        scope: BarrierScope,
        transport: ResolvedDependencyTransport,
    },
    Publish {
        output: u32,
        operand: OperandId,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedKernel<D: ExecutableDialect> {
    pub participant_storage: Vec<ResolvedStorage<D>>,
    pub workgroup_storage: Vec<ResolvedStorage<D>>,
    pub steps: NonEmpty<ResolvedKernelStep<D>>,
    pub resources: ResolvedKernelResources<D::Capability>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedKernelResources<C> {
    pub registers: u64,
    pub private_bytes: u64,
    pub workgroup_bytes: u64,
    pub capabilities: BTreeSet<C>,
    pub numerical: Vec<NumericalEffect>,
    pub accesses: Vec<ResolvedPhysicalAccess>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPhysicalAccess {
    pub storage: ResolvedStorageId,
    pub mode: AccessMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDispatchGeometry {
    pub workgroups: [u64; 3],
    pub participants_per_workgroup: [u64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedLaunch<D: ExecutableDialect> {
    pub id: ResolvedLaunchId,
    pub geometry: ResolvedDispatchGeometry,
    /// Complete template-to-resolved storage identity map for this launch,
    /// including plan, workgroup, and participant storage.
    pub storage: BTreeMap<StorageId, ResolvedStorage<D>>,
    pub value_map: BTreeMap<KernelValueId, ResolvedKernelValueId>,
    pub binding_groups: Vec<ResolvedBindingGroup>,
    pub kernel: ResolvedKernel<D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPhase<D: ExecutableDialect> {
    pub id: ResolvedPhaseId,
    pub predecessors: Vec<ResolvedPhaseId>,
    pub launches: NonEmpty<ResolvedLaunch<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedDeviceStoragePlan<D: ExecutableDialect> {
    pub allocations: Vec<ResolvedStorage<D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedScheduleItem<D: ExecutableDialect> {
    Phase(ResolvedPhase<D>),
    Subplan(ResolvedSubplan<D>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSubplan<D: ExecutableDialect> {
    pub call: CallBoundaryId,
    pub choice: ChoiceId,
    pub inputs: Vec<ResolvedCallOperandBinding>,
    pub results: Vec<ResolvedCallOperandBinding>,
    pub plan: Box<ResolvedPlan<D>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallOperandBindingTemplate {
    pub operand: OperandId,
    pub boundary_port: Option<u32>,
    pub path: Vec<u32>,
    pub caller: ValueTransportTemplate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCallOperandBinding {
    pub operand: OperandId,
    pub boundary_port: Option<u32>,
    pub path: Vec<u32>,
    pub caller: ResolvedValueTransport,
    pub callee: ResolvedValueTransport,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutableTargetLimits {
    pub max_allocation_bytes: u64,
    pub max_device_bytes: u64,
    pub max_workgroup_bytes: u64,
    pub max_private_bytes_per_participant: u64,
    pub max_bindings_per_launch: u64,
    pub max_registers_per_kernel: u64,
    pub max_workgroups: [u64; 3],
    pub max_participants_per_workgroup: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutableTargetProfile<C> {
    pub target: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub capabilities: BTreeSet<C>,
    pub limits: ExecutableTargetLimits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPlan<D: ExecutableDialect> {
    identity: ExecutableResolutionIdentity,
    task_graph: TaskGraphId,
    device_storage: ResolvedDeviceStoragePlan<D>,
    items: NonEmpty<ResolvedScheduleItem<D>>,
    dependencies: Vec<ResolvedDependencyPlacement>,
    boundary_inputs: BTreeMap<u32, ResolvedValueTransport>,
    boundary_results: BTreeMap<u32, ResolvedValueTransport>,
    numerical: Vec<NumericalEffect>,
    numerical_assessment: NumericalAssessment,
    estimated_cost: i64,
    optimal: bool,
}

impl<D: ExecutableDialect> ResolvedPlan<D> {
    pub fn identity(&self) -> &ExecutableResolutionIdentity {
        &self.identity
    }

    pub fn task_graph(&self) -> TaskGraphId {
        self.task_graph
    }

    pub fn device_storage(&self) -> &ResolvedDeviceStoragePlan<D> {
        &self.device_storage
    }

    pub fn items(&self) -> &NonEmpty<ResolvedScheduleItem<D>> {
        &self.items
    }

    pub fn dependencies(&self) -> &[ResolvedDependencyPlacement] {
        &self.dependencies
    }

    pub fn boundary_inputs(&self) -> &BTreeMap<u32, ResolvedValueTransport> {
        &self.boundary_inputs
    }

    pub fn boundary_results(&self) -> &BTreeMap<u32, ResolvedValueTransport> {
        &self.boundary_results
    }

    pub fn numerical_effects(&self) -> &[NumericalEffect] {
        &self.numerical
    }

    pub fn numerical_assessment(&self) -> &NumericalAssessment {
        &self.numerical_assessment
    }

    pub fn estimated_cost(&self) -> i64 {
        self.estimated_cost
    }

    pub fn optimal(&self) -> bool {
        self.optimal
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecisionResolutionIdentity {
    pub method_revision: String,
    pub evidence_domain: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutableResolutionIdentity {
    pub logical: LogicalCompilationIdentity,
    pub entry: String,
    pub target: String,
    pub capability_fingerprint: String,
    pub toolchain_fingerprint: String,
    pub assignment: PlanAssignment,
    pub precision: PrecisionResolutionIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanSelection {
    pub logical_alternative: u32,
    pub physical_alternative: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanAssignment {
    selections: BTreeMap<ChoiceId, PlanSelection>,
    symbols: BTreeMap<String, i64>,
}

impl PlanAssignment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select(&mut self, choice: ChoiceId, selection: PlanSelection) -> Result<(), String> {
        if self.selections.insert(choice, selection).is_some() {
            return Err(format!("choice#{} is selected more than once", choice.0));
        }
        Ok(())
    }

    pub fn bind_symbol(&mut self, name: impl Into<String>, value: i64) -> Result<(), String> {
        let name = name.into();
        if value < 0 {
            return Err(format!("physical symbol `{name}` is negative"));
        }
        if self.symbols.insert(name.clone(), value).is_some() {
            return Err(format!("physical symbol `{name}` is bound more than once"));
        }
        Ok(())
    }

    pub fn selections(&self) -> &BTreeMap<ChoiceId, PlanSelection> {
        &self.selections
    }

    pub fn symbols(&self) -> &BTreeMap<String, i64> {
        &self.symbols
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleTemplate<D: ExecutableDialect> {
    device_storage: DeviceStoragePlan<D>,
    items: NonEmpty<ScheduleItem<D>>,
    dependency_placements: Vec<DependencyPlacement>,
    boundary_inputs: BTreeMap<u32, ValueTransportTemplate>,
    boundary_results: BTreeMap<u32, ValueTransportTemplate>,
}

impl<D: ExecutableDialect> ScheduleTemplate<D> {
    pub fn device_storage(&self) -> &DeviceStoragePlan<D> {
        &self.device_storage
    }

    pub fn items(&self) -> &NonEmpty<ScheduleItem<D>> {
        &self.items
    }

    pub fn dependency_placements(&self) -> &[DependencyPlacement] {
        &self.dependency_placements
    }

    pub fn boundary_inputs(&self) -> &BTreeMap<u32, ValueTransportTemplate> {
        &self.boundary_inputs
    }

    pub fn boundary_results(&self) -> &BTreeMap<u32, ValueTransportTemplate> {
        &self.boundary_results
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScheduleItem<D: ExecutableDialect> {
    Phase(PhaseTemplate<D>),
    Subplan(SubplanInvocation),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubplanInvocation {
    pub call: CallBoundaryId,
    pub choice: ChoiceId,
    pub inputs: Vec<CallOperandBindingTemplate>,
    pub results: Vec<CallOperandBindingTemplate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependencyTransport {
    Control,
    Value {
        operand: OperandId,
        transport: ValueTransportTemplate,
    },
    Effect {
        logical_storage: StorageRef,
        storage: NonEmpty<StorageId>,
    },
    Ownership {
        logical_storage: StorageRef,
        storage: NonEmpty<StorageId>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyOrder {
    ProgramOrder,
    Barrier,
    LaunchBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyPlacement {
    pub dependency: DependencyId,
    pub order: DependencyOrder,
    pub transport: DependencyTransport,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedDependencyTransport {
    Control,
    Value {
        operand: OperandId,
        transport: ResolvedValueTransport,
    },
    Effect {
        logical_storage: StorageRef,
        storage: NonEmpty<ResolvedStorageId>,
    },
    Ownership {
        logical_storage: StorageRef,
        storage: NonEmpty<ResolvedStorageId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedDependencyPlacement {
    pub dependency: DependencyId,
    pub order: DependencyOrder,
    pub transport: ResolvedDependencyTransport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanAlternative<D: ExecutableDialect> {
    physical_alternative: u32,
    logical_choice: ChoiceId,
    logical_alternative: u32,
    task_graph: TaskGraphId,
    schedule: ScheduleTemplate<D>,
    cost: Sym,
}

impl<D: ExecutableDialect> PlanAlternative<D> {
    pub fn physical_alternative(&self) -> u32 {
        self.physical_alternative
    }
    pub fn task_graph(&self) -> TaskGraphId {
        self.task_graph
    }

    pub fn logical_choice(&self) -> ChoiceId {
        self.logical_choice
    }

    pub fn logical_alternative(&self) -> u32 {
        self.logical_alternative
    }

    pub fn schedule(&self) -> &ScheduleTemplate<D> {
        &self.schedule
    }

    pub fn cost(&self) -> &Sym {
        &self.cost
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanChoice<D: ExecutableDialect> {
    pub logical_choice: ChoiceId,
    /// Empty only for the entry choice. Multiple entries are disjunctive
    /// activation paths to the same occurrence-qualified choice.
    pub active_when: Vec<ChoiceActivation>,
    pub alternatives: NonEmpty<PlanAlternative<D>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChoiceActivation {
    pub parent: ChoiceId,
    pub alternative: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanFamily<D: ExecutableDialect> {
    logical_identity: LogicalCompilationIdentity,
    entry_name: String,
    target: String,
    capability_fingerprint: String,
    entry: ChoiceId,
    choices: Vec<PlanChoice<D>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlternativeStorageFact {
    pub storage: StorageId,
    pub launch: Option<LaunchId>,
    pub scope: StorageScope,
    pub bytes: Sym,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlternativeLaunchFacts<C> {
    pub launch: LaunchId,
    pub workgroups: [Sym; 3],
    pub participants_per_workgroup: Sym,
    pub workgroup_bytes: Sym,
    pub private_bytes_per_participant: Sym,
    pub registers: Sym,
    pub bindings: u64,
    pub capabilities: BTreeSet<C>,
    pub numerical: Vec<NumericalEffect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanAlternativeFacts<C> {
    pub choice: ChoiceId,
    pub logical_alternative: u32,
    pub physical_alternative: u32,
    pub allocations: Vec<AlternativeStorageFact>,
    pub aggregate_device_bytes: Sym,
    pub launches: Vec<AlternativeLaunchFacts<C>>,
}

impl<D: ExecutableDialect> PlanFamily<D> {
    pub fn logical_identity(&self) -> &LogicalCompilationIdentity {
        &self.logical_identity
    }

    pub fn entry(&self) -> ChoiceId {
        self.entry
    }

    pub fn choices(&self) -> &[PlanChoice<D>] {
        &self.choices
    }

    /// Solver-facing hard facts derived from the exact same storage and
    /// instruction objects consumed by resolution.
    pub fn alternative_facts(&self) -> Result<Vec<PlanAlternativeFacts<D::Capability>>, String> {
        let mut facts = Vec::new();
        for choice in &self.choices {
            for alternative in choice.alternatives.iter() {
                let mut allocations = alternative
                    .schedule
                    .device_storage
                    .allocations
                    .iter()
                    .map(|storage| AlternativeStorageFact {
                        storage: storage.id,
                        launch: None,
                        scope: storage.scope,
                        bytes: storage.bytes.clone(),
                    })
                    .collect::<Vec<_>>();
                let aggregate_device_bytes = alternative
                    .schedule
                    .device_storage
                    .allocations
                    .iter()
                    .filter(|storage| storage.scope == StorageScope::Device)
                    .fold(Sym::constant(0), |total, storage| total.add(&storage.bytes));
                let mut launches = Vec::new();
                for item in alternative.schedule.items.iter() {
                    let ScheduleItem::Phase(phase) = item else {
                        continue;
                    };
                    for launch in phase.launches.iter() {
                        allocations.extend(
                            launch
                                .kernel
                                .workgroup_storage
                                .iter()
                                .chain(launch.kernel.participant_storage.iter())
                                .map(|storage| AlternativeStorageFact {
                                    storage: storage.id,
                                    launch: Some(launch.id),
                                    scope: storage.scope,
                                    bytes: storage.bytes.clone(),
                                }),
                        );
                        let mut workgroup_bytes = launch
                            .kernel
                            .workgroup_storage
                            .iter()
                            .fold(Sym::constant(0), |total, storage| total.add(&storage.bytes));
                        let mut private_bytes = launch
                            .kernel
                            .participant_storage
                            .iter()
                            .fold(Sym::constant(0), |total, storage| total.add(&storage.bytes));
                        let mut registers = Sym::constant(0);
                        let mut capabilities = BTreeSet::new();
                        let mut numerical = Vec::new();
                        for step in launch.kernel.steps.iter() {
                            let KernelStepKind::MappedTask { instructions, .. } = &step.kind else {
                                continue;
                            };
                            for instruction in instructions.iter() {
                                let consequence = D::consequences(instruction);
                                registers = registers.add(&Sym::constant(i64_resource(
                                    u64::from(consequence.resources.registers),
                                    "register resource",
                                )?));
                                private_bytes = private_bytes.add(&Sym::constant(i64_resource(
                                    consequence.resources.private_bytes,
                                    "private resource",
                                )?));
                                workgroup_bytes =
                                    workgroup_bytes.add(&Sym::constant(i64_resource(
                                        consequence.resources.workgroup_bytes,
                                        "workgroup resource",
                                    )?));
                                if let Some(capability) = consequence.capability {
                                    capabilities.insert(capability);
                                }
                                numerical.extend(consequence.numerical);
                            }
                        }
                        let participants_per_workgroup = launch
                            .geometry
                            .participants_per_workgroup
                            .iter()
                            .fold(Sym::constant(1), |product, value| product.mul(value));
                        launches.push(AlternativeLaunchFacts {
                            launch: launch.id,
                            workgroups: launch.geometry.workgroups.clone(),
                            participants_per_workgroup,
                            workgroup_bytes,
                            private_bytes_per_participant: private_bytes,
                            registers,
                            bindings: launch.binding_groups.len() as u64,
                            capabilities,
                            numerical,
                        });
                    }
                }
                facts.push(PlanAlternativeFacts {
                    choice: choice.logical_choice,
                    logical_alternative: alternative.logical_alternative,
                    physical_alternative: alternative.physical_alternative,
                    allocations,
                    aggregate_device_bytes,
                    launches,
                });
            }
        }
        Ok(facts)
    }

    /// Resolve the complete occurrence-qualified choice tree exactly once.
    /// Calls remain nested subplans; only phase items contain native kernels.
    pub fn resolve(
        &self,
        assignment: &PlanAssignment,
        target: &ExecutableTargetProfile<D::Capability>,
        precision: PrecisionResolutionIdentity,
        numerical_assessment: Option<NumericalAssessment>,
        optimal: bool,
    ) -> Result<ResolvedPlan<D>, String> {
        if target.target != self.target
            || target.capability_fingerprint != self.capability_fingerprint
        {
            return Err("executable target identity differs from logical specialization".into());
        }
        let identity = ExecutableResolutionIdentity {
            logical: self.logical_identity.clone(),
            entry: self.entry_name.clone(),
            target: target.target.clone(),
            capability_fingerprint: target.capability_fingerprint.clone(),
            toolchain_fingerprint: target.toolchain_fingerprint.clone(),
            assignment: assignment.clone(),
            precision,
        };
        let mut resolver = FamilyResolver {
            family: self,
            assignment,
            target,
            identity,
            numerical_assessment,
            optimal,
            next_storage: 0,
            next_phase: 0,
            next_launch: 0,
            next_binding_group: 0,
            next_binding: 0,
            next_kernel_value: 0,
            visiting: BTreeSet::new(),
            visited: BTreeSet::new(),
        };
        let plan = resolver.resolve_choice(self.entry, None)?;
        let selected = assignment
            .selections
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        if selected != resolver.visited {
            let inactive = selected
                .difference(&resolver.visited)
                .map(|choice| choice.0.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let missing = resolver
                .visited
                .difference(&selected)
                .map(|choice| choice.0.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "plan assignment is not the complete active choice tree (inactive [{inactive}], missing [{missing}])"
            ));
        }
        Ok(plan)
    }
}

struct FamilyResolver<'a, D: ExecutableDialect> {
    family: &'a PlanFamily<D>,
    assignment: &'a PlanAssignment,
    target: &'a ExecutableTargetProfile<D::Capability>,
    identity: ExecutableResolutionIdentity,
    numerical_assessment: Option<NumericalAssessment>,
    optimal: bool,
    next_storage: u64,
    next_phase: u64,
    next_launch: u64,
    next_binding_group: u64,
    next_binding: u64,
    next_kernel_value: u64,
    visiting: BTreeSet<ChoiceId>,
    visited: BTreeSet<ChoiceId>,
}

impl<D: ExecutableDialect> FamilyResolver<'_, D> {
    fn resolve_choice(
        &mut self,
        choice_id: ChoiceId,
        activated_by: Option<ChoiceActivation>,
    ) -> Result<ResolvedPlan<D>, String> {
        let choice = self
            .family
            .choices
            .iter()
            .find(|choice| choice.logical_choice == choice_id)
            .ok_or_else(|| format!("assignment names absent choice#{}", choice_id.0))?;
        if let Some(activation) = activated_by {
            if !choice.active_when.contains(&activation) {
                return Err(format!(
                    "choice#{} is not active under choice#{} alternative#{}",
                    choice_id.0, activation.parent.0, activation.alternative
                ));
            }
        } else if choice_id != self.family.entry {
            return Err(format!(
                "non-entry choice#{} has no activation",
                choice_id.0
            ));
        }
        if !self.visiting.insert(choice_id) {
            return Err(format!(
                "choice topology contains a cycle at choice#{}",
                choice_id.0
            ));
        }
        if self.visited.contains(&choice_id) {
            return Err(format!(
                "occurrence-qualified choice#{} is invoked more than once",
                choice_id.0
            ));
        }
        let selection = self
            .assignment
            .selections
            .get(&choice_id)
            .copied()
            .ok_or_else(|| format!("active choice#{} has no selection", choice_id.0))?;
        let alternative = choice
            .alternatives
            .iter()
            .find(|alternative| {
                alternative.logical_alternative == selection.logical_alternative
                    && alternative.physical_alternative == selection.physical_alternative
            })
            .ok_or_else(|| {
                format!(
                    "choice#{} has no logical alternative#{} physical alternative#{}",
                    choice_id.0, selection.logical_alternative, selection.physical_alternative
                )
            })?;
        let result = self.resolve_alternative(alternative)?;
        self.visiting.remove(&choice_id);
        self.visited.insert(choice_id);
        Ok(result)
    }

    fn resolve_alternative(
        &mut self,
        alternative: &PlanAlternative<D>,
    ) -> Result<ResolvedPlan<D>, String> {
        let mut device_ids = BTreeMap::new();
        let mut device_descriptors = BTreeMap::new();
        let mut device = Vec::new();
        let mut device_bytes = 0u64;
        for storage in &alternative.schedule.device_storage.allocations {
            let resolved = self.resolve_storage(storage)?;
            if storage.scope == StorageScope::Device {
                device_bytes = device_bytes
                    .checked_add(resolved.bytes)
                    .ok_or("device storage byte total overflows u64")?;
            }
            device_ids.insert(storage.id, resolved.id);
            device_descriptors.insert(storage.id, resolved.clone());
            device.push(resolved);
        }
        if device_bytes > self.target.limits.max_device_bytes {
            return Err(format!(
                "plan needs {device_bytes} device bytes, limit {}",
                self.target.limits.max_device_bytes
            ));
        }
        let mut phase_ids = BTreeMap::new();
        for item in alternative.schedule.items.iter() {
            if let ScheduleItem::Phase(phase) = item {
                phase_ids.insert(phase.id, self.fresh_phase());
            }
        }
        let mut items = Vec::new();
        let mut numerical = Vec::new();
        for item in alternative.schedule.items.iter() {
            match item {
                ScheduleItem::Phase(phase) => {
                    let resolved = self.resolve_phase(phase, &phase_ids, &device_descriptors)?;
                    for launch in resolved.launches.iter() {
                        numerical.extend(launch.kernel.resources.numerical.iter().cloned());
                    }
                    items.push(ResolvedScheduleItem::Phase(resolved));
                }
                ScheduleItem::Subplan(invocation) => {
                    let plan = self.resolve_choice(
                        invocation.choice,
                        Some(ChoiceActivation {
                            parent: alternative.logical_choice,
                            alternative: alternative.logical_alternative,
                        }),
                    )?;
                    let inputs = self.resolve_call_bindings(
                        &invocation.inputs,
                        &device_ids,
                        plan.boundary_inputs(),
                        "input",
                    )?;
                    let results = self.resolve_call_bindings(
                        &invocation.results,
                        &device_ids,
                        plan.boundary_results(),
                        "result",
                    )?;
                    numerical.extend(plan.numerical.iter().cloned());
                    items.push(ResolvedScheduleItem::Subplan(ResolvedSubplan {
                        call: invocation.call,
                        choice: invocation.choice,
                        inputs,
                        results,
                        plan: Box::new(plan),
                    }));
                }
            }
        }
        let items = nonempty(items, "resolved plan has no executable item")?;
        let cost = eval_i64(&alternative.cost, &self.assignment.symbols, "plan cost")?;
        if cost < 0 {
            return Err("resolved plan cost is negative".into());
        }
        let boundary_inputs = alternative
            .schedule
            .boundary_inputs
            .iter()
            .map(|(port, transport)| Ok((*port, resolve_plan_transport(transport, &device_ids)?)))
            .collect::<Result<_, String>>()?;
        let boundary_results = alternative
            .schedule
            .boundary_results
            .iter()
            .map(|(port, transport)| Ok((*port, resolve_plan_transport(transport, &device_ids)?)))
            .collect::<Result<_, String>>()?;
        let mut kernel_values = BTreeMap::new();
        for item in items.iter() {
            if let ResolvedScheduleItem::Phase(phase) = item {
                for launch in phase.launches.iter() {
                    for (template, resolved) in &launch.value_map {
                        if kernel_values.insert(*template, *resolved).is_some() {
                            return Err(format!(
                                "kernel value#{} is reused across launches",
                                template.0
                            ));
                        }
                    }
                }
            }
        }
        let dependencies = alternative
            .schedule
            .dependency_placements
            .iter()
            .map(|placement| resolve_dependency(placement, &device_ids, &kernel_values))
            .collect::<Result<_, _>>()?;
        Ok(ResolvedPlan {
            identity: self.identity.clone(),
            task_graph: alternative.task_graph,
            device_storage: ResolvedDeviceStoragePlan {
                allocations: device,
            },
            items,
            dependencies,
            boundary_inputs,
            boundary_results,
            numerical_assessment: self.numerical_assessment.clone().unwrap_or_else(|| {
                if numerical.is_empty() {
                    NumericalAssessment::exact()
                } else {
                    NumericalAssessment::unknown(
                        "numerical effects admitted by an unconstrained compilation",
                    )
                }
            }),
            numerical,
            estimated_cost: cost,
            optimal: self.optimal,
        })
    }

    fn resolve_phase(
        &mut self,
        phase: &PhaseTemplate<D>,
        phase_ids: &BTreeMap<PhaseId, ResolvedPhaseId>,
        device: &BTreeMap<StorageId, ResolvedStorage<D>>,
    ) -> Result<ResolvedPhase<D>, String> {
        let predecessors = phase
            .predecessors
            .iter()
            .map(|predecessor| {
                phase_ids
                    .get(predecessor)
                    .copied()
                    .ok_or_else(|| format!("phase#{} predecessor is absent", phase.id.0))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut launches = Vec::new();
        for launch in phase.launches.iter() {
            launches.push(self.resolve_launch(launch, device)?);
        }
        Ok(ResolvedPhase {
            id: phase_ids[&phase.id],
            predecessors,
            launches: nonempty(launches, "resolved phase has no launch")?,
        })
    }

    fn resolve_launch(
        &mut self,
        launch: &LaunchTemplate<D>,
        device: &BTreeMap<StorageId, ResolvedStorage<D>>,
    ) -> Result<ResolvedLaunch<D>, String> {
        let workgroups = resolve_array(&launch.geometry.workgroups, &self.assignment.symbols)?;
        let participants = resolve_array(
            &launch.geometry.participants_per_workgroup,
            &self.assignment.symbols,
        )?;
        for axis in 0..3 {
            if workgroups[axis] == 0
                || workgroups[axis] > self.target.limits.max_workgroups[axis]
                || participants[axis] == 0
            {
                return Err(format!(
                    "launch#{} has illegal dispatch axis {axis}",
                    launch.id.0
                ));
            }
        }
        let participant_count = participants.iter().try_fold(1u64, |value, extent| {
            value
                .checked_mul(*extent)
                .ok_or("participant count overflows u64")
        })?;
        if participant_count > self.target.limits.max_participants_per_workgroup {
            return Err(format!(
                "launch#{} needs {participant_count} participants, limit {}",
                launch.id.0, self.target.limits.max_participants_per_workgroup
            ));
        }
        let mut storage_descriptors = device.clone();
        let mut participant_storage = Vec::new();
        let mut workgroup_storage = Vec::new();
        for storage in &launch.kernel.participant_storage {
            let resolved = self.resolve_storage(storage)?;
            storage_descriptors.insert(storage.id, resolved.clone());
            participant_storage.push(resolved);
        }
        for storage in &launch.kernel.workgroup_storage {
            let resolved = self.resolve_storage(storage)?;
            storage_descriptors.insert(storage.id, resolved.clone());
            workgroup_storage.push(resolved);
        }
        let mut kernel_values = BTreeMap::new();
        for step in launch.kernel.steps.iter() {
            if let KernelStepKind::MappedTask { bindings, .. } = &step.kind {
                for binding in bindings.iter() {
                    if let ValueTransportTemplate::Kernel(value) = &binding.transport {
                        if !kernel_values.contains_key(value) {
                            let resolved = self.fresh_kernel_value();
                            kernel_values.insert(*value, resolved);
                        }
                    }
                }
            }
        }
        let mut resources = ResolvedKernelResources {
            registers: 0,
            private_bytes: participant_storage
                .iter()
                .try_fold(0u64, |value, storage| {
                    value
                        .checked_add(storage.bytes)
                        .ok_or("private storage overflows u64")
                })?,
            workgroup_bytes: workgroup_storage.iter().try_fold(0u64, |value, storage| {
                value
                    .checked_add(storage.bytes)
                    .ok_or("workgroup storage overflows u64")
            })?,
            capabilities: BTreeSet::new(),
            numerical: Vec::new(),
            accesses: Vec::new(),
        };
        let storage_ids = storage_descriptors
            .iter()
            .map(|(template, resolved)| (*template, resolved.id))
            .collect::<BTreeMap<_, _>>();
        let mut steps = Vec::new();
        for step in launch.kernel.steps.iter() {
            steps.push(self.resolve_step(step, &storage_ids, &kernel_values, &mut resources)?);
        }
        for step in &steps {
            let ResolvedKernelStep::MappedTask { mapping, .. } = step else {
                continue;
            };
            for axis in &mapping.axes {
                let ResolvedAxisMap::Grid {
                    logical_axis,
                    workgroup_axis,
                    participant_axis,
                    mode,
                } = axis
                else {
                    continue;
                };
                let capacity = workgroups[*workgroup_axis as usize]
                    .checked_mul(participants[*participant_axis as usize])
                    .ok_or("grid-axis capacity overflows u64")?;
                let extent = mapping.logical_extents[*logical_axis as usize];
                match mode {
                    ResolvedGridMapping::OnePass => {
                        if extent > capacity
                            || (extent < capacity && !mapping.masks_inactive_participants)
                        {
                            return Err(format!(
                                "launch#{} one-pass grid mapping does not exactly cover task#{} axis#{}",
                                launch.id.0, mapping.task.0, logical_axis
                            ));
                        }
                    }
                    ResolvedGridMapping::GridStride { stride } => {
                        if *stride != capacity
                            || (extent % stride != 0 && !mapping.masks_inactive_participants)
                        {
                            return Err(format!(
                                "launch#{} grid-stride mapping is inconsistent for task#{} axis#{}",
                                launch.id.0, mapping.task.0, logical_axis
                            ));
                        }
                    }
                }
            }
        }
        if resources.registers > self.target.limits.max_registers_per_kernel
            || resources.private_bytes > self.target.limits.max_private_bytes_per_participant
            || resources.workgroup_bytes > self.target.limits.max_workgroup_bytes
        {
            return Err(format!(
                "launch#{} exceeds kernel resource limits",
                launch.id.0
            ));
        }
        if !resources.capabilities.is_subset(&self.target.capabilities) {
            return Err(format!(
                "launch#{} requires unsupported capabilities",
                launch.id.0
            ));
        }
        if launch.binding_groups.len() as u64 > self.target.limits.max_bindings_per_launch {
            return Err(format!(
                "launch#{} exceeds native binding limit",
                launch.id.0
            ));
        }
        let binding_groups = launch
            .binding_groups
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|binding| {
                        let storage = device
                            .get(&binding.storage)
                            .map(|value| value.id)
                            .ok_or_else(|| {
                                format!("binding#{} does not name plan-level storage", binding.id.0)
                            })?;
                        Ok(ResolvedBinding {
                            id: self.fresh_binding(),
                            storage,
                            access: binding.access,
                            operand: binding.operand,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(ResolvedBindingGroup {
                    id: self.fresh_binding_group(),
                    kind: group.kind,
                    slot: group.slot,
                    members: nonempty(members, "resolved binding group has no member")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(ResolvedLaunch {
            id: self.fresh_launch(),
            geometry: ResolvedDispatchGeometry {
                workgroups,
                participants_per_workgroup: participants,
            },
            storage: storage_descriptors,
            value_map: kernel_values,
            binding_groups,
            kernel: ResolvedKernel {
                participant_storage,
                workgroup_storage,
                steps: nonempty(steps, "resolved kernel has no step")?,
                resources,
            },
        })
    }

    fn resolve_step(
        &self,
        step: &KernelStep<D>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
        kernel_values: &BTreeMap<KernelValueId, ResolvedKernelValueId>,
        resources: &mut ResolvedKernelResources<D::Capability>,
    ) -> Result<ResolvedKernelStep<D>, String> {
        Ok(match &step.kind {
            KernelStepKind::MappedTask {
                task,
                mapping,
                bindings,
                instructions,
            } => {
                let mut resolved_instructions = Vec::new();
                for instruction in instructions.iter() {
                    let consequence = D::consequences(instruction);
                    resources.registers = resources
                        .registers
                        .checked_add(u64::from(consequence.resources.registers))
                        .ok_or("register resource total overflows u64")?;
                    resources.private_bytes = resources
                        .private_bytes
                        .checked_add(consequence.resources.private_bytes)
                        .ok_or("private resource total overflows u64")?;
                    resources.workgroup_bytes = resources
                        .workgroup_bytes
                        .checked_add(consequence.resources.workgroup_bytes)
                        .ok_or("workgroup resource total overflows u64")?;
                    if let Some(capability) = consequence.capability {
                        resources.capabilities.insert(capability);
                    }
                    resources.numerical.extend(consequence.numerical);
                    for access in consequence.accesses {
                        resources.accesses.push(ResolvedPhysicalAccess {
                            storage: storage.get(&access.storage).copied().ok_or_else(|| {
                                format!("instruction accesses absent storage#{}", access.storage.0)
                            })?,
                            mode: access.mode,
                        });
                    }
                    resolved_instructions.push(D::resolve_instruction(
                        instruction,
                        &self.assignment.symbols,
                        storage,
                    )?);
                }
                ResolvedKernelStep::MappedTask {
                    task: *task,
                    mapping: ResolvedParticipantMap {
                        task: *task,
                        axes: mapping
                            .axes
                            .iter()
                            .map(|axis| match axis {
                                AxisMap::Grid {
                                    logical_axis,
                                    workgroup_axis,
                                    participant_axis,
                                    mode,
                                } => Ok(ResolvedAxisMap::Grid {
                                    logical_axis: *logical_axis,
                                    workgroup_axis: *workgroup_axis,
                                    participant_axis: *participant_axis,
                                    mode: match mode {
                                        GridMapping::OnePass => ResolvedGridMapping::OnePass,
                                        GridMapping::GridStride { stride } => {
                                            ResolvedGridMapping::GridStride {
                                                stride: eval_u64(
                                                    stride,
                                                    &self.assignment.symbols,
                                                    "grid stride",
                                                )?,
                                            }
                                        }
                                    },
                                }),
                                AxisMap::SubgroupLane { logical_axis } => {
                                    Ok(ResolvedAxisMap::SubgroupLane {
                                        logical_axis: *logical_axis,
                                    })
                                }
                                AxisMap::Serial { logical_axis } => Ok(ResolvedAxisMap::Serial {
                                    logical_axis: *logical_axis,
                                }),
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        logical_extents: mapping
                            .logical_extents
                            .iter()
                            .map(|value| {
                                eval_u64(value, &self.assignment.symbols, "logical extent")
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        masks_inactive_participants: mapping.masks_inactive_participants,
                    },
                    bindings: bindings
                        .iter()
                        .map(|binding| {
                            Ok(ResolvedOperandTransport {
                                operand: binding.operand,
                                transport: resolve_launch_transport(
                                    &binding.transport,
                                    storage,
                                    kernel_values,
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                    instructions: nonempty(
                        resolved_instructions,
                        "mapped task has no physical instruction",
                    )?,
                }
            }
            KernelStepKind::Barrier {
                dependency,
                scope,
                transport,
            } => ResolvedKernelStep::Barrier {
                dependency: *dependency,
                scope: *scope,
                transport: resolve_dependency_transport(transport, storage, kernel_values)?,
            },
            KernelStepKind::Publish { output, operand } => ResolvedKernelStep::Publish {
                output: *output,
                operand: *operand,
            },
        })
    }

    fn resolve_storage(
        &mut self,
        storage: &StorageTemplate<D>,
    ) -> Result<ResolvedStorage<D>, String> {
        let bytes = eval_u64(&storage.bytes, &self.assignment.symbols, "storage bytes")?;
        if bytes == 0 || bytes > self.target.limits.max_allocation_bytes {
            return Err(format!(
                "storage#{} needs {bytes} bytes, allocation limit {}",
                storage.id.0, self.target.limits.max_allocation_bytes
            ));
        }
        let subrange_offset = eval_u64(
            &storage.provenance.subrange.byte_offset,
            &self.assignment.symbols,
            "storage subrange offset",
        )?;
        let subrange_bytes = eval_u64(
            &storage.provenance.subrange.bytes,
            &self.assignment.symbols,
            "storage subrange bytes",
        )?;
        if subrange_bytes == 0
            || subrange_offset
                .checked_add(subrange_bytes)
                .is_none_or(|end| end > bytes)
        {
            return Err(format!(
                "storage#{} logical subrange exceeds its physical allocation",
                storage.id.0
            ));
        }
        Ok(ResolvedStorage {
            id: self.fresh_storage(),
            scope: storage.scope,
            replication: storage.replication,
            bytes,
            alignment: storage.alignment,
            layout: D::resolve_layout(&storage.layout, &self.assignment.symbols)?,
            provenance: ResolvedPhysicalStorageProvenance {
                logical_storage: storage.provenance.logical_storage.clone(),
                operand: storage.provenance.operand,
                view: storage.provenance.view,
                subrange: ResolvedPhysicalSubrange {
                    byte_offset: subrange_offset,
                    bytes: subrange_bytes,
                },
                address: resolve_address(&storage.provenance.address, &self.assignment.symbols)?,
                abi: storage.provenance.abi.clone(),
            },
        })
    }

    fn fresh_storage(&mut self) -> ResolvedStorageId {
        let id = ResolvedStorageId(self.next_storage);
        self.next_storage += 1;
        id
    }
    fn fresh_phase(&mut self) -> ResolvedPhaseId {
        let id = ResolvedPhaseId(self.next_phase);
        self.next_phase += 1;
        id
    }
    fn fresh_launch(&mut self) -> ResolvedLaunchId {
        let id = ResolvedLaunchId(self.next_launch);
        self.next_launch += 1;
        id
    }
    fn fresh_binding_group(&mut self) -> ResolvedBindingGroupId {
        let id = ResolvedBindingGroupId(self.next_binding_group);
        self.next_binding_group += 1;
        id
    }
    fn fresh_binding(&mut self) -> ResolvedBindingId {
        let id = ResolvedBindingId(self.next_binding);
        self.next_binding += 1;
        id
    }
    fn fresh_kernel_value(&mut self) -> ResolvedKernelValueId {
        let id = ResolvedKernelValueId(self.next_kernel_value);
        self.next_kernel_value += 1;
        id
    }

    fn resolve_call_bindings(
        &self,
        bindings: &[CallOperandBindingTemplate],
        caller_storage: &BTreeMap<StorageId, ResolvedStorageId>,
        callee_boundary: &BTreeMap<u32, ResolvedValueTransport>,
        direction: &str,
    ) -> Result<Vec<ResolvedCallOperandBinding>, String> {
        let mut resolved = Vec::with_capacity(bindings.len());
        let mut leaves = BTreeMap::new();
        for binding in bindings {
            let caller = resolve_plan_transport(&binding.caller, caller_storage)?;
            let Some(port) = binding.boundary_port else {
                resolved.push((binding, caller, None));
                continue;
            };
            let callee = callee_boundary.get(&port).cloned().ok_or_else(|| {
                format!(
                    "call {direction} operand#{} names absent child boundary port#{port}",
                    binding.operand.0
                )
            })?;
            if !transport_shape_matches(&caller, &callee) {
                return Err(format!(
                    "call {direction} operand#{} transport shape disagrees with child port#{port}",
                    binding.operand.0
                ));
            }
            leaves.insert(binding.path.clone(), callee.clone());
            resolved.push((binding, caller, Some(callee)));
        }
        resolved
            .into_iter()
            .map(|(binding, caller, callee)| {
                let callee = match callee {
                    Some(callee) => callee,
                    None => rebuild_transport(&caller, &mut Vec::new(), &leaves)?,
                };
                Ok(ResolvedCallOperandBinding {
                    operand: binding.operand,
                    boundary_port: binding.boundary_port,
                    path: binding.path.clone(),
                    caller,
                    callee,
                })
            })
            .collect()
    }
}

fn transport_shape_matches(left: &ResolvedValueTransport, right: &ResolvedValueTransport) -> bool {
    match (left, right) {
        (ResolvedValueTransport::Void, ResolvedValueTransport::Void) => true,
        (ResolvedValueTransport::Kernel(_), ResolvedValueTransport::Kernel(_)) => true,
        (ResolvedValueTransport::Storage(left), ResolvedValueTransport::Storage(right)) => {
            left.len() == right.len()
        }
        (ResolvedValueTransport::Tuple(left), ResolvedValueTransport::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(left, right)| transport_shape_matches(left, right))
        }
        _ => false,
    }
}

fn rebuild_transport(
    shape: &ResolvedValueTransport,
    path: &mut Vec<u32>,
    leaves: &BTreeMap<Vec<u32>, ResolvedValueTransport>,
) -> Result<ResolvedValueTransport, String> {
    match shape {
        ResolvedValueTransport::Void => Ok(ResolvedValueTransport::Void),
        ResolvedValueTransport::Tuple(fields) => {
            let values = fields
                .iter()
                .enumerate()
                .map(|(field, value)| {
                    path.push(field as u32);
                    let result = rebuild_transport(value, path, leaves);
                    path.pop();
                    result.map(Box::new)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ResolvedValueTransport::Tuple(nonempty(
                values,
                "call aggregate transport has an empty tuple",
            )?))
        }
        _ => leaves
            .get(path)
            .cloned()
            .ok_or_else(|| format!("call aggregate omits result path {path:?}")),
    }
}

fn resolve_plan_transport(
    transport: &ValueTransportTemplate,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
) -> Result<ResolvedValueTransport, String> {
    match transport {
        ValueTransportTemplate::Void => Ok(ResolvedValueTransport::Void),
        ValueTransportTemplate::Storage(ids) => Ok(ResolvedValueTransport::Storage(
            resolve_storage_bundle(ids, storage)?,
        )),
        ValueTransportTemplate::Tuple(fields) => Ok(ResolvedValueTransport::Tuple(nonempty(
            fields
                .iter()
                .map(|field| resolve_plan_transport(field, storage).map(Box::new))
                .collect::<Result<Vec<_>, _>>()?,
            "tuple transport has no field",
        )?)),
        ValueTransportTemplate::Kernel(_) => {
            Err("kernel values cannot cross a plan or call boundary".into())
        }
    }
}

fn resolve_launch_transport(
    transport: &ValueTransportTemplate,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
    kernel_values: &BTreeMap<KernelValueId, ResolvedKernelValueId>,
) -> Result<ResolvedValueTransport, String> {
    match transport {
        ValueTransportTemplate::Void => Ok(ResolvedValueTransport::Void),
        ValueTransportTemplate::Storage(ids) => Ok(ResolvedValueTransport::Storage(
            resolve_storage_bundle(ids, storage)?,
        )),
        ValueTransportTemplate::Tuple(fields) => Ok(ResolvedValueTransport::Tuple(nonempty(
            fields
                .iter()
                .map(|field| resolve_launch_transport(field, storage, kernel_values).map(Box::new))
                .collect::<Result<Vec<_>, _>>()?,
            "tuple transport has no field",
        )?)),
        ValueTransportTemplate::Kernel(id) => kernel_values
            .get(&id)
            .copied()
            .map(ResolvedValueTransport::Kernel)
            .ok_or_else(|| format!("value transport names absent kernel value#{}", id.0)),
    }
}

fn resolve_storage_bundle(
    ids: &NonEmpty<StorageId>,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
) -> Result<NonEmpty<ResolvedStorageId>, String> {
    nonempty(
        ids.iter()
            .map(|id| {
                storage
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("value transport names absent storage#{}", id.0))
            })
            .collect::<Result<Vec<_>, _>>()?,
        "storage transport has no representation plane",
    )
}

fn resolve_dependency_transport(
    transport: &DependencyTransport,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
    kernel_values: &BTreeMap<KernelValueId, ResolvedKernelValueId>,
) -> Result<ResolvedDependencyTransport, String> {
    Ok(match transport {
        DependencyTransport::Control => ResolvedDependencyTransport::Control,
        DependencyTransport::Value { operand, transport } => ResolvedDependencyTransport::Value {
            operand: *operand,
            transport: resolve_launch_transport(transport, storage, kernel_values)?,
        },
        DependencyTransport::Effect {
            logical_storage,
            storage: id,
        } => ResolvedDependencyTransport::Effect {
            logical_storage: logical_storage.clone(),
            storage: resolve_storage_bundle(id, storage)?,
        },
        DependencyTransport::Ownership {
            logical_storage,
            storage: id,
        } => ResolvedDependencyTransport::Ownership {
            logical_storage: logical_storage.clone(),
            storage: resolve_storage_bundle(id, storage)?,
        },
    })
}

fn resolve_dependency(
    placement: &DependencyPlacement,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
    kernel_values: &BTreeMap<KernelValueId, ResolvedKernelValueId>,
) -> Result<ResolvedDependencyPlacement, String> {
    let transport = match &placement.transport {
        DependencyTransport::Control => ResolvedDependencyTransport::Control,
        DependencyTransport::Value { operand, transport } => ResolvedDependencyTransport::Value {
            operand: *operand,
            transport: resolve_launch_transport(transport, storage, kernel_values)?,
        },
        DependencyTransport::Effect {
            logical_storage,
            storage: id,
        } => ResolvedDependencyTransport::Effect {
            logical_storage: logical_storage.clone(),
            storage: resolve_storage_bundle(id, storage)?,
        },
        DependencyTransport::Ownership {
            logical_storage,
            storage: id,
        } => ResolvedDependencyTransport::Ownership {
            logical_storage: logical_storage.clone(),
            storage: resolve_storage_bundle(id, storage)?,
        },
    };
    Ok(ResolvedDependencyPlacement {
        dependency: placement.dependency,
        order: placement.order,
        transport,
    })
}

fn nonempty<T>(mut values: Vec<T>, message: &str) -> Result<NonEmpty<T>, String> {
    if values.is_empty() {
        return Err(message.into());
    }
    let mut result = NonEmpty::new(values.remove(0));
    for value in values {
        result.push(value);
    }
    Ok(result)
}

fn i64_resource(value: u64, label: &str) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| format!("{label} exceeds symbolic i64 range"))
}

fn eval_i64(value: &Sym, symbols: &BTreeMap<String, i64>, label: &str) -> Result<i64, String> {
    value
        .eval(&|name| symbols.get(name).copied())
        .ok_or_else(|| format!("{label} contains an unresolved or invalid symbol"))
}

fn eval_u64(value: &Sym, symbols: &BTreeMap<String, i64>, label: &str) -> Result<u64, String> {
    u64::try_from(eval_i64(value, symbols, label)?)
        .map_err(|_| format!("{label} is negative or exceeds u64"))
}

fn resolve_array(values: &[Sym; 3], symbols: &BTreeMap<String, i64>) -> Result<[u64; 3], String> {
    Ok([
        eval_u64(&values[0], symbols, "dispatch extent")?,
        eval_u64(&values[1], symbols, "dispatch extent")?,
        eval_u64(&values[2], symbols, "dispatch extent")?,
    ])
}

fn resolve_address(
    address: &PhysicalAddressTemplate,
    symbols: &BTreeMap<String, i64>,
) -> Result<ResolvedPhysicalAddress, String> {
    Ok(match address {
        PhysicalAddressTemplate::DenseAffine {
            byte_offset,
            byte_strides,
        } => ResolvedPhysicalAddress::DenseAffine {
            byte_offset: eval_u64(byte_offset, symbols, "byte offset")?,
            byte_strides: byte_strides
                .iter()
                .map(|stride| eval_u64(stride, symbols, "byte stride"))
                .collect::<Result<_, _>>()?,
        },
        PhysicalAddressTemplate::Representation {
            representation,
            plane,
            logical_strides,
        } => ResolvedPhysicalAddress::Representation {
            representation: representation.clone(),
            plane: plane.clone(),
            logical_strides: logical_strides
                .iter()
                .map(|stride| eval_u64(stride, symbols, "logical stride"))
                .collect::<Result<_, _>>()?,
        },
    })
}

pub struct PlanFamilyBuilder<D: ExecutableDialect> {
    logical_identity: LogicalCompilationIdentity,
    entry_name: String,
    target: String,
    capability_fingerprint: String,
    entry: ChoiceId,
    expected: BTreeMap<ChoiceId, BTreeMap<u32, TaskGraphId>>,
    active_when: BTreeMap<ChoiceId, BTreeSet<ChoiceActivation>>,
    alternatives: BTreeMap<ChoiceId, Vec<PlanAlternative<D>>>,
}

impl<D: ExecutableDialect> PlanFamilyBuilder<D> {
    pub fn from_logical(logical: &LogicalProgram) -> Result<Self, String> {
        if logical.entry_choice.0 as usize >= logical.choices.len() {
            return Err("logical entry choice is absent".into());
        }
        let mut expected = BTreeMap::new();
        let mut active_when = BTreeMap::<ChoiceId, BTreeSet<ChoiceActivation>>::new();
        for (ordinal, choice) in logical.choices.iter().enumerate() {
            let choice_id = ChoiceId(ordinal as u32);
            let mut alternatives = BTreeMap::new();
            for (alternative, value) in choice.alternatives.iter().enumerate() {
                if alternatives
                    .insert(alternative as u32, value.task_graph)
                    .is_some()
                {
                    return Err(format!(
                        "choice#{} repeats logical alternative#{alternative}",
                        choice_id.0
                    ));
                }
                let graph = logical
                    .task_graphs
                    .get(value.task_graph.0 as usize)
                    .ok_or_else(|| {
                        format!(
                            "choice#{} alternative#{alternative} names absent task graph#{}",
                            choice_id.0, value.task_graph.0
                        )
                    })?;
                if graph.id != value.task_graph
                    || graph.choice != choice_id
                    || graph.alternative != alternative as u32
                {
                    return Err(format!(
                        "task graph#{} identity disagrees with choice#{} alternative#{alternative}",
                        value.task_graph.0, choice_id.0
                    ));
                }
                for call in &graph.calls {
                    if call.choice.0 as usize >= logical.choices.len() {
                        return Err(format!(
                            "call#{} in task graph#{} targets absent choice#{}",
                            call.id.0, graph.id.0, call.choice.0
                        ));
                    }
                    active_when
                        .entry(call.choice)
                        .or_default()
                        .insert(ChoiceActivation {
                            parent: choice_id,
                            alternative: alternative as u32,
                        });
                }
            }
            expected.insert(choice_id, alternatives);
        }
        for choice in expected.keys() {
            if *choice != logical.entry_choice && !active_when.contains_key(choice) {
                return Err(format!(
                    "non-entry choice#{} has no call activation path",
                    choice.0
                ));
            }
        }
        Ok(Self {
            logical_identity: logical.identity.clone(),
            entry_name: logical.entry.clone(),
            target: logical.target.clone(),
            capability_fingerprint: logical.capability_fingerprint.clone(),
            entry: logical.entry_choice,
            expected,
            active_when,
            alternatives: BTreeMap::new(),
        })
    }

    pub fn add_alternative(&mut self, mut alternative: PlanAlternative<D>) -> Result<(), String> {
        let expected = self
            .expected
            .get(&alternative.logical_choice)
            .and_then(|values| values.get(&alternative.logical_alternative))
            .ok_or_else(|| "schedule names an absent logical alternative".to_string())?;
        if *expected != alternative.task_graph {
            return Err("schedule task graph disagrees with its logical alternative".into());
        }
        let values = self
            .alternatives
            .entry(alternative.logical_choice)
            .or_default();
        alternative.physical_alternative = values.len() as u32;
        values.push(alternative);
        Ok(())
    }

    pub fn finish(self) -> Result<PlanFamily<D>, String> {
        let mut choices = Vec::with_capacity(self.expected.len());
        for (choice, logical_alternatives) in self.expected {
            let values = self.alternatives.get(&choice).cloned().unwrap_or_default();
            for logical_alternative in logical_alternatives.keys() {
                if !values
                    .iter()
                    .any(|value| value.logical_alternative == *logical_alternative)
                {
                    return Err(format!(
                        "choice#{} logical alternative#{} has no complete physical schedule",
                        choice.0, logical_alternative
                    ));
                }
            }
            let mut values = values.into_iter();
            let first = values
                .next()
                .ok_or_else(|| format!("choice#{} has no physical schedule", choice.0))?;
            let mut alternatives = NonEmpty::new(first);
            for value in values {
                alternatives.push(value);
            }
            choices.push(PlanChoice {
                logical_choice: choice,
                active_when: self
                    .active_when
                    .get(&choice)
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect(),
                alternatives,
            });
        }
        Ok(PlanFamily {
            logical_identity: self.logical_identity,
            entry_name: self.entry_name,
            target: self.target,
            capability_fingerprint: self.capability_fingerprint,
            entry: self.entry,
            choices,
        })
    }
}

#[derive(Debug)]
pub struct TaskObligation {
    graph: TaskGraphId,
    task: TaskId,
}

#[derive(Debug)]
pub struct CallObligation {
    graph: TaskGraphId,
    call: CallBoundaryId,
    choice: ChoiceId,
}

#[derive(Debug)]
pub struct DependencyObligation {
    graph: TaskGraphId,
    dependency: DependencyId,
}

#[derive(Debug)]
pub struct OutputObligation {
    graph: TaskGraphId,
    output: u32,
}

#[derive(Debug)]
pub struct InputObligation {
    graph: TaskGraphId,
    input: u32,
}

/// The sole constructor of a complete schedule alternative. Obligation tokens
/// are not cloneable and all commits re-check pending ownership, so duplicated,
/// foreign, and already-consumed work cannot enter the schedule.
pub struct ScheduleBuilder<D: ExecutableDialect> {
    graph: TaskGraphId,
    logical_choice: ChoiceId,
    logical_alternative: u32,
    input_types: Vec<Type>,
    result_types: Vec<Type>,
    input_paths: Vec<Vec<u32>>,
    result_paths: Vec<Vec<u32>>,
    result_count: usize,
    pending_tasks: BTreeSet<TaskId>,
    task_axes: BTreeMap<TaskId, Vec<AxisOrder>>,
    task_operands: BTreeMap<TaskId, (BTreeSet<OperandId>, BTreeSet<OperandId>)>,
    pending_calls: BTreeSet<CallBoundaryId>,
    call_choices: BTreeMap<CallBoundaryId, ChoiceId>,
    call_operands: BTreeMap<CallBoundaryId, (Vec<OperandId>, Vec<OperandId>)>,
    call_results: BTreeMap<CallBoundaryId, OperandId>,
    call_result_paths: BTreeMap<CallBoundaryId, BTreeMap<OperandId, Vec<u32>>>,
    pending_dependencies: BTreeSet<DependencyId>,
    pending_inputs: BTreeSet<u32>,
    pending_outputs: BTreeSet<u32>,
    operands: BTreeSet<OperandId>,
    operand_types: BTreeMap<OperandId, Type>,
    operand_storage: BTreeMap<OperandId, StorageRef>,
    effect_storage: Vec<StorageRef>,
    boundary_storage: BTreeSet<StorageRef>,
    dependencies: BTreeMap<DependencyId, LogicalDependency>,
    device_storage: DeviceStoragePlan<D>,
    phases: Vec<PhaseTemplate<D>>,
    items: Vec<ScheduleItem<D>>,
    dependency_placements: Vec<DependencyPlacement>,
    task_bindings: BTreeMap<TaskId, BTreeMap<OperandId, ValueTransportTemplate>>,
    input_bindings: BTreeMap<u32, ValueTransportTemplate>,
    output_bindings: BTreeMap<u32, ValueTransportTemplate>,
}

impl<D: ExecutableDialect> ScheduleBuilder<D> {
    pub fn new(graph: &LogicalTaskGraph) -> Result<Self, String> {
        validate_task_graph(graph)?;
        Ok(Self {
            graph: graph.id,
            logical_choice: graph.choice,
            logical_alternative: graph.alternative,
            input_types: graph.inputs.iter().map(|port| port.ty.clone()).collect(),
            result_types: graph.results.iter().map(|port| port.ty.clone()).collect(),
            input_paths: graph.inputs.iter().map(|port| port.path.clone()).collect(),
            result_paths: graph.results.iter().map(|port| port.path.clone()).collect(),
            result_count: graph.results.len(),
            pending_tasks: graph.tasks.iter().map(|task| task.id).collect(),
            task_axes: graph
                .tasks
                .iter()
                .map(|task| {
                    (
                        task.id,
                        task.domain.axes.iter().map(|axis| axis.order).collect(),
                    )
                })
                .collect(),
            task_operands: graph
                .tasks
                .iter()
                .map(|task| {
                    (
                        task.id,
                        (
                            task.inputs.iter().copied().collect(),
                            task.outputs.iter().copied().collect(),
                        ),
                    )
                })
                .collect(),
            pending_calls: graph.calls.iter().map(|call| call.id).collect(),
            call_choices: graph
                .calls
                .iter()
                .map(|call| (call.id, call.choice))
                .collect(),
            call_operands: graph
                .calls
                .iter()
                .map(|call| (call.id, (call.inputs.clone(), call.outputs.clone())))
                .collect(),
            call_results: graph
                .calls
                .iter()
                .map(|call| (call.id, call.result))
                .collect(),
            call_result_paths: graph
                .calls
                .iter()
                .map(|call| {
                    let mut paths = Vec::new();
                    flatten_type_paths(
                        &graph.operands[call.result.0 as usize].ty,
                        &mut Vec::new(),
                        &mut paths,
                    );
                    (call.id, call.outputs.iter().copied().zip(paths).collect())
                })
                .collect(),
            pending_dependencies: graph
                .dependencies
                .iter()
                .map(|dependency| dependency.id)
                .collect(),
            pending_inputs: (0..graph.inputs.len() as u32).collect(),
            pending_outputs: (0..graph.results.len() as u32).collect(),
            operands: graph.operands.iter().map(|operand| operand.id).collect(),
            operand_types: graph
                .operands
                .iter()
                .map(|operand| (operand.id, operand.ty.clone()))
                .collect(),
            operand_storage: graph
                .operands
                .iter()
                .filter_map(|operand| operand.storage.clone().map(|storage| (operand.id, storage)))
                .collect(),
            effect_storage: graph
                .tasks
                .iter()
                .flat_map(|task| task.effects.accesses.iter())
                .chain(
                    graph
                        .calls
                        .iter()
                        .flat_map(|call| call.effects.accesses.iter()),
                )
                .map(|access| access.storage.clone())
                .collect(),
            boundary_storage: graph
                .inputs
                .iter()
                .enumerate()
                .map(|(port, _)| StorageRef::Input {
                    port: port as u32,
                    path: Vec::new(),
                })
                .chain(graph.results.iter().enumerate().map(|(port, _)| {
                    StorageRef::Result {
                        port: port as u32,
                        path: Vec::new(),
                    }
                }))
                .collect(),
            dependencies: graph
                .dependencies
                .iter()
                .cloned()
                .map(|dependency| (dependency.id, dependency))
                .collect(),
            device_storage: DeviceStoragePlan {
                allocations: vec![],
            },
            phases: vec![],
            items: vec![],
            dependency_placements: vec![],
            task_bindings: BTreeMap::new(),
            input_bindings: BTreeMap::new(),
            output_bindings: BTreeMap::new(),
        })
    }

    pub fn task(&self, task: TaskId) -> Result<TaskObligation, String> {
        self.pending_tasks
            .contains(&task)
            .then_some(TaskObligation {
                graph: self.graph,
                task,
            })
            .ok_or_else(|| format!("task#{} is absent or already consumed", task.0))
    }

    pub fn call(&self, call: CallBoundaryId) -> Result<CallObligation, String> {
        self.pending_calls
            .contains(&call)
            .then_some(CallObligation {
                graph: self.graph,
                call,
                choice: self.call_choices[&call],
            })
            .ok_or_else(|| format!("call#{} is absent or already consumed", call.0))
    }

    pub fn dependency(&self, dependency: DependencyId) -> Result<DependencyObligation, String> {
        self.pending_dependencies
            .contains(&dependency)
            .then_some(DependencyObligation {
                graph: self.graph,
                dependency,
            })
            .ok_or_else(|| format!("dependency#{} is absent or already consumed", dependency.0))
    }

    pub fn output(&self, output: u32) -> Result<OutputObligation, String> {
        self.pending_outputs
            .contains(&output)
            .then_some(OutputObligation {
                graph: self.graph,
                output,
            })
            .ok_or_else(|| format!("output#{output} is absent or already consumed"))
    }

    pub fn input(&self, input: u32) -> Result<InputObligation, String> {
        self.pending_inputs
            .contains(&input)
            .then_some(InputObligation {
                graph: self.graph,
                input,
            })
            .ok_or_else(|| format!("input#{input} is absent or already consumed"))
    }

    pub fn add_operand_storage(
        &mut self,
        operand: OperandId,
        storage: StorageTemplate<D>,
    ) -> Result<(), String> {
        if storage.provenance.operand != Some(operand) {
            return Err("operand storage provenance does not name its operand".into());
        }
        self.insert_device_storage(storage)
    }

    pub fn add_input_storage(
        &mut self,
        input: u32,
        storage: StorageTemplate<D>,
    ) -> Result<(), String> {
        let path = self
            .input_paths
            .get(input as usize)
            .ok_or_else(|| format!("input#{input} is absent"))?;
        if !matches!(
            &storage.provenance.abi,
            Some(AbiRole::Parameter { ordinal, path: abi_path, .. })
                if *ordinal == input && abi_path == path
        ) {
            return Err(format!("input#{input} storage has the wrong parameter ABI role"));
        }
        let logical = StorageRef::Input {
            port: input,
            path: Vec::new(),
        };
        if storage.provenance.operand.is_some()
            || storage.provenance.logical_storage.as_ref() != Some(&logical)
        {
            return Err(format!("input#{input} storage has the wrong logical provenance"));
        }
        self.insert_device_storage(storage)
    }

    pub fn add_result_storage(
        &mut self,
        result: u32,
        storage: StorageTemplate<D>,
    ) -> Result<(), String> {
        let path = self
            .result_paths
            .get(result as usize)
            .ok_or_else(|| format!("result#{result} is absent"))?;
        if !matches!(
            &storage.provenance.abi,
            Some(AbiRole::Result { ordinal, path: abi_path, .. })
                if *ordinal == result && abi_path == path
        ) {
            return Err(format!("result#{result} storage has the wrong result ABI role"));
        }
        let logical = StorageRef::Result {
            port: result,
            path: Vec::new(),
        };
        if storage.provenance.operand.is_some()
            || storage.provenance.logical_storage.as_ref() != Some(&logical)
        {
            return Err(format!("result#{result} storage has the wrong logical provenance"));
        }
        self.insert_device_storage(storage)
    }

    pub fn add_effect_storage(
        &mut self,
        logical_storage: StorageRef,
        storage: StorageTemplate<D>,
    ) -> Result<(), String> {
        if storage.provenance.logical_storage.as_ref() != Some(&logical_storage)
            || !self.effect_storage.contains(&logical_storage)
        {
            return Err("effect storage provenance is not an effect of this task graph".into());
        }
        self.insert_device_storage(storage)
    }

    pub fn add_invocation_storage(&mut self, storage: StorageTemplate<D>) -> Result<(), String> {
        if !matches!(
            storage.provenance.abi,
            Some(AbiRole::InvocationResource { .. })
        ) {
            return Err("invocation storage requires an invocation-resource ABI role".into());
        }
        self.insert_device_storage(storage)
    }

    fn insert_device_storage(&mut self, storage: StorageTemplate<D>) -> Result<(), String> {
        if !matches!(storage.scope, StorageScope::External | StorageScope::Device) {
            return Err("plan-level storage must have external or device scope".into());
        }
        if self
            .device_storage
            .allocations
            .iter()
            .any(|existing| existing.id == storage.id)
        {
            return Err(format!("storage#{} is duplicated", storage.id.0));
        }
        self.validate_storage(&storage)?;
        self.device_storage.allocations.push(storage);
        Ok(())
    }

    pub fn add_phase(&mut self, phase: PhaseTemplate<D>) -> Result<(), String> {
        if self.phases.iter().any(|existing| existing.id == phase.id) {
            return Err(format!("phase#{} is duplicated", phase.id.0));
        }
        for predecessor in &phase.predecessors {
            if !self
                .phases
                .iter()
                .any(|existing| existing.id == *predecessor)
            {
                return Err(format!(
                    "phase#{} names absent or later predecessor#{}",
                    phase.id.0, predecessor.0
                ));
            }
        }
        self.consume_phase(&phase)?;
        self.phases.push(phase.clone());
        self.items.push(ScheduleItem::Phase(phase));
        Ok(())
    }

    pub fn mapped_task_step(
        &self,
        obligation: TaskObligation,
        mapping: ParticipantMap,
        bindings: Vec<OperandTransportTemplate>,
        instructions: NonEmpty<D::TemplateInstruction>,
    ) -> Result<KernelStep<D>, String> {
        self.require_graph(obligation.graph)?;
        if mapping.task != obligation.task {
            return Err("participant map belongs to another logical task".into());
        }
        let axes = &self.task_axes[&obligation.task];
        let rank = axes.len();
        if mapping.logical_extents.len() != rank {
            return Err(format!(
                "participant map for task#{} disagrees with logical rank {rank}",
                obligation.task.0
            ));
        }
        let mut logical_axes = BTreeSet::new();
        let mut participant_axes = BTreeSet::new();
        let mut workgroup_axes = BTreeSet::new();
        for axis in &mapping.axes {
            let (logical, grid, serial) = match axis {
                AxisMap::Grid {
                    logical_axis,
                    workgroup_axis,
                    participant_axis,
                    ..
                } => (
                    *logical_axis,
                    Some((*workgroup_axis, *participant_axis)),
                    false,
                ),
                AxisMap::SubgroupLane { logical_axis } => (*logical_axis, None, false),
                AxisMap::Serial { logical_axis } => (*logical_axis, None, true),
            };
            if logical as usize >= rank || !logical_axes.insert(logical) {
                return Err(format!(
                    "task#{} omits or duplicates logical axis#{logical}",
                    obligation.task.0
                ));
            }
            if axes[logical as usize] == AxisOrder::Ordered && !serial {
                return Err(format!(
                    "task#{} ordered axis#{logical} must be serial",
                    obligation.task.0
                ));
            }
            if let Some((workgroup, participant)) = grid {
                if workgroup >= 3
                    || participant >= 3
                    || !workgroup_axes.insert(workgroup)
                    || !participant_axes.insert(participant)
                {
                    return Err(format!(
                        "task#{} has an invalid or multiply claimed participant axis",
                        obligation.task.0
                    ));
                }
            }
        }
        if logical_axes.len() != rank {
            return Err(format!(
                "participant map for task#{} does not represent every logical axis",
                obligation.task.0
            ));
        }
        let expected = &self.task_operands[&obligation.task];
        let expected = expected
            .0
            .union(&expected.1)
            .copied()
            .collect::<BTreeSet<_>>();
        let mut actual = BTreeSet::new();
        for binding in bindings.iter() {
            if !actual.insert(binding.operand) {
                return Err(format!(
                    "task#{} repeats operand#{} binding",
                    obligation.task.0, binding.operand.0
                ));
            }
            if !transport_matches_type(&self.operand_types[&binding.operand], &binding.transport) {
                return Err(format!(
                    "task#{} operand#{} transport disagrees with its logical type",
                    obligation.task.0, binding.operand.0
                ));
            }
        }
        if actual != expected {
            return Err(format!(
                "task#{} operand bindings are incomplete or extraneous",
                obligation.task.0
            ));
        }
        Ok(KernelStep {
            kind: KernelStepKind::MappedTask {
                task: obligation.task,
                mapping,
                bindings,
                instructions,
            },
        })
    }

    pub fn add_subplan(
        &mut self,
        obligation: CallObligation,
        inputs: Vec<OperandTransportTemplate>,
        results: Vec<OperandTransportTemplate>,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        let expected = &self.call_operands[&obligation.call];
        let validate_inputs = |expected: &[OperandId],
                               bindings: &[OperandTransportTemplate]|
         -> Result<Vec<CallOperandBindingTemplate>, String> {
            if bindings.len() != expected.len() {
                return Err("call input bindings are incomplete".into());
            }
            expected
                .iter()
                .enumerate()
                .map(|(port, operand)| {
                    let binding = bindings
                        .iter()
                        .find(|binding| binding.operand == *operand)
                        .ok_or_else(|| format!("call input omits operand#{}", operand.0))?;
                    self.validate_plan_value_transport(*operand, &binding.transport)?;
                    Ok(CallOperandBindingTemplate {
                        operand: *operand,
                        boundary_port: Some(port as u32),
                        path: Vec::new(),
                        caller: binding.transport.clone(),
                    })
                })
                .collect()
        };
        let inputs = validate_inputs(&expected.0, &inputs)?;
        let aggregate = self.call_results[&obligation.call];
        let aggregate_is_leaf = expected.1.contains(&aggregate);
        let expected_result_count = expected.1.len() + usize::from(!aggregate_is_leaf);
        if results.len() != expected_result_count {
            return Err("call result bindings are incomplete".into());
        }
        let paths = &self.call_result_paths[&obligation.call];
        let mut resolved_results = Vec::with_capacity(expected_result_count);
        for (port, operand) in expected.1.iter().enumerate() {
            let binding = results
                .iter()
                .find(|binding| binding.operand == *operand)
                .ok_or_else(|| format!("call result omits operand#{}", operand.0))?;
            self.validate_plan_value_transport(*operand, &binding.transport)?;
            resolved_results.push(CallOperandBindingTemplate {
                operand: *operand,
                boundary_port: Some(port as u32),
                path: paths.get(operand).cloned().unwrap_or_default(),
                caller: binding.transport.clone(),
            });
        }
        if !aggregate_is_leaf {
            let binding = results
                .iter()
                .find(|binding| binding.operand == aggregate)
                .ok_or_else(|| format!("call result omits aggregate operand#{}", aggregate.0))?;
            self.validate_plan_value_transport(aggregate, &binding.transport)?;
            resolved_results.push(CallOperandBindingTemplate {
                operand: aggregate,
                boundary_port: None,
                path: Vec::new(),
                caller: binding.transport.clone(),
            });
        }
        if !self.pending_calls.remove(&obligation.call) {
            return Err(format!(
                "call#{} is scheduled more than once",
                obligation.call.0
            ));
        }
        self.items.push(ScheduleItem::Subplan(SubplanInvocation {
            call: obligation.call,
            choice: obligation.choice,
            inputs,
            results: resolved_results,
        }));
        Ok(())
    }

    pub fn barrier_step(
        &self,
        obligation: DependencyObligation,
        scope: BarrierScope,
        transport: DependencyTransport,
    ) -> Result<KernelStep<D>, String> {
        self.require_graph(obligation.graph)?;
        Ok(KernelStep {
            kind: KernelStepKind::Barrier {
                dependency: obligation.dependency,
                scope,
                transport,
            },
        })
    }

    pub fn publish_step(
        &self,
        obligation: OutputObligation,
        operand: OperandId,
    ) -> Result<KernelStep<D>, String> {
        self.require_graph(obligation.graph)?;
        if !self.operands.contains(&operand) {
            return Err(format!("publication names absent operand#{}", operand.0));
        }
        Ok(KernelStep {
            kind: KernelStepKind::Publish {
                output: obligation.output,
                operand,
            },
        })
    }

    /// Publish an already-retained task or call result without manufacturing
    /// a native kernel step. The resolved plan boundary preserves the exact
    /// transport, and dependency order places the logical output at plan end.
    pub fn publish_output(
        &mut self,
        obligation: OutputObligation,
        operand: OperandId,
        transport: ValueTransportTemplate,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        if !self.operands.contains(&operand) {
            return Err(format!("publication names absent operand#{}", operand.0));
        }
        if self.operand_transport(operand) != Some(transport.clone()) {
            return Err(format!(
                "publication transport disagrees with operand#{} producer",
                operand.0
            ));
        }
        if !self.pending_outputs.remove(&obligation.output) {
            return Err(format!(
                "output#{} is published more than once",
                obligation.output
            ));
        }
        self.output_bindings.insert(obligation.output, transport);
        Ok(())
    }

    /// Bind a boundary result that is produced through effects/storage rather
    /// than a logical value operand (including zero-channel `Void`).
    pub fn bind_output(
        &mut self,
        obligation: OutputObligation,
        transport: ValueTransportTemplate,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        let ty = self
            .result_types
            .get(obligation.output as usize)
            .ok_or_else(|| format!("output#{} is absent", obligation.output))?;
        if !transport_matches_type(ty, &transport) {
            return Err(format!(
                "output#{} transport disagrees with its logical type",
                obligation.output
            ));
        }
        let ids = transport_storage_ids(&transport)
            .ok_or("plan output requires storage-backed or void transport")?;
        for id in ids {
            let storage = self
                .device_storage
                .allocations
                .iter()
                .find(|storage| storage.id == *id)
                .ok_or_else(|| {
                    format!("output#{} names absent storage#{}", obligation.output, id.0)
                })?;
            if storage.scope != StorageScope::External {
                return Err(format!(
                    "output#{} transport storage#{} is not external",
                    obligation.output, id.0
                ));
            }
        }
        if !self.pending_outputs.remove(&obligation.output) {
            return Err(format!("output#{} is bound more than once", obligation.output));
        }
        self.output_bindings.insert(obligation.output, transport);
        Ok(())
    }

    /// Bind one declared plan input explicitly. Inputs are obligations even
    /// when the selected body does not read them; they are never inferred from
    /// whichever task operands happen to survive normalization.
    pub fn bind_input(
        &mut self,
        obligation: InputObligation,
        transport: ValueTransportTemplate,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        let ty = self
            .input_types
            .get(obligation.input as usize)
            .ok_or_else(|| format!("input#{} is absent", obligation.input))?;
        if !transport_matches_type(ty, &transport) {
            return Err(format!(
                "input#{} transport disagrees with its logical type",
                obligation.input
            ));
        }
        let ids = transport_storage_ids(&transport)
            .ok_or("plan input requires storage-backed or void transport")?;
        for id in ids {
            let storage = self
                .device_storage
                .allocations
                .iter()
                .find(|storage| storage.id == *id)
                .ok_or_else(|| format!("input#{} names absent storage#{}", obligation.input, id.0))?;
            if storage.scope != StorageScope::External {
                return Err(format!(
                    "input#{} transport storage#{} is not external",
                    obligation.input, id.0
                ));
            }
        }
        if !self.pending_inputs.remove(&obligation.input) {
            return Err(format!("input#{} is bound more than once", obligation.input));
        }
        self.input_bindings.insert(obligation.input, transport);
        Ok(())
    }

    fn consume_phase(&mut self, phase: &PhaseTemplate<D>) -> Result<(), String> {
        for launch in phase.launches.iter() {
            self.validate_launch(launch)?;
            for step in launch.kernel.steps.iter() {
                match &step.kind {
                    KernelStepKind::MappedTask { task, bindings, .. } => {
                        if !self.pending_tasks.remove(task) {
                            return Err(format!("task#{} is scheduled more than once", task.0));
                        }
                        self.task_bindings.insert(
                            *task,
                            bindings
                                .iter()
                                .map(|binding| (binding.operand, binding.transport.clone()))
                                .collect(),
                        );
                    }
                    KernelStepKind::Barrier {
                        dependency,
                        transport,
                        ..
                    } => {
                        self.validate_dependency_transport(*dependency, transport, false)?;
                        if !self.pending_dependencies.remove(dependency) {
                            return Err(format!(
                                "dependency#{} is placed more than once",
                                dependency.0
                            ));
                        }
                        self.dependency_placements.push(DependencyPlacement {
                            dependency: *dependency,
                            order: DependencyOrder::Barrier,
                            transport: transport.clone(),
                        });
                    }
                    KernelStepKind::Publish { output, operand } => {
                        if !self.pending_outputs.remove(output) {
                            return Err(format!("output#{output} is published more than once"));
                        }
                        let transport = self
                            .task_bindings
                            .values()
                            .find_map(|bindings| bindings.get(operand).cloned())
                            .ok_or_else(|| {
                                format!("published operand#{} has no physical transport", operand.0)
                            })?;
                        self.output_bindings.insert(*output, transport);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn place_program_order(
        &mut self,
        obligation: DependencyObligation,
        transport: DependencyTransport,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        self.validate_dependency_transport(obligation.dependency, &transport, false)?;
        if !self.pending_dependencies.remove(&obligation.dependency) {
            return Err(format!(
                "dependency#{} is placed more than once",
                obligation.dependency.0
            ));
        }
        self.dependency_placements.push(DependencyPlacement {
            dependency: obligation.dependency,
            order: DependencyOrder::ProgramOrder,
            transport,
        });
        Ok(())
    }

    pub fn place_launch_boundary(
        &mut self,
        obligation: DependencyObligation,
        transport: DependencyTransport,
    ) -> Result<(), String> {
        self.require_graph(obligation.graph)?;
        self.validate_dependency_transport(obligation.dependency, &transport, true)?;
        if !self.pending_dependencies.remove(&obligation.dependency) {
            return Err(format!(
                "dependency#{} is placed more than once",
                obligation.dependency.0
            ));
        }
        self.dependency_placements.push(DependencyPlacement {
            dependency: obligation.dependency,
            order: DependencyOrder::LaunchBoundary,
            transport,
        });
        Ok(())
    }

    pub fn finish(self, cost: Sym) -> Result<PlanAlternative<D>, String> {
        if !self.pending_tasks.is_empty()
            || !self.pending_calls.is_empty()
            || !self.pending_dependencies.is_empty()
            || !self.pending_inputs.is_empty()
            || !self.pending_outputs.is_empty()
        {
            return Err(format!(
                "schedule is incomplete: {} tasks, {} calls, {} dependencies, {} inputs, {} outputs remain",
                self.pending_tasks.len(),
                self.pending_calls.len(),
                self.pending_dependencies.len(),
                self.pending_inputs.len(),
                self.pending_outputs.len()
            ));
        }
        self.validate_dependency_order()?;
        let boundary_inputs = self.input_bindings.clone();
        let boundary_results = self.output_bindings.clone();
        if boundary_results.len() != self.result_count {
            return Err("result boundary transports are incomplete".into());
        }
        let mut items = self.items.into_iter();
        let first = items
            .next()
            .ok_or_else(|| "schedule has no executable item".to_string())?;
        let mut nonempty = NonEmpty::new(first);
        for item in items {
            nonempty.push(item);
        }
        Ok(PlanAlternative {
            physical_alternative: 0,
            logical_choice: self.logical_choice,
            logical_alternative: self.logical_alternative,
            task_graph: self.graph,
            schedule: ScheduleTemplate {
                device_storage: self.device_storage,
                items: nonempty,
                dependency_placements: self.dependency_placements,
                boundary_inputs,
                boundary_results,
            },
            cost,
        })
    }

    fn require_graph(&self, graph: TaskGraphId) -> Result<(), String> {
        (graph == self.graph)
            .then_some(())
            .ok_or_else(|| "obligation belongs to another task graph".into())
    }

    fn operand_transport(&self, operand: OperandId) -> Option<ValueTransportTemplate> {
        let mut found = None;
        for transport in self
            .task_bindings
            .values()
            .filter_map(|bindings| bindings.get(&operand).cloned())
            .chain(self.items.iter().filter_map(|item| {
                match item {
                    ScheduleItem::Subplan(call) => call
                        .inputs
                        .iter()
                        .chain(&call.results)
                        .find(|binding| binding.operand == operand)
                        .map(|binding| binding.caller.clone()),
                    ScheduleItem::Phase(_) => None,
                }
            }))
        {
            if found.is_some_and(|old| old != transport) {
                return None;
            }
            found = Some(transport);
        }
        found
    }

    fn endpoint_transport(
        &self,
        endpoint: LogicalEndpoint,
        operand: OperandId,
    ) -> Option<ValueTransportTemplate> {
        match endpoint {
            LogicalEndpoint::Input(_) => self.operand_transport(operand),
            LogicalEndpoint::Task(task) => self
                .task_bindings
                .get(&task)
                .and_then(|bindings| bindings.get(&operand))
                .cloned(),
            LogicalEndpoint::Call(call) => self.items.iter().find_map(|item| match item {
                ScheduleItem::Subplan(invocation) if invocation.call == call => invocation
                    .inputs
                    .iter()
                    .chain(&invocation.results)
                    .find(|binding| binding.operand == operand)
                    .map(|binding| binding.caller.clone()),
                _ => None,
            }),
            LogicalEndpoint::Output(output) => self.output_bindings.get(&output).cloned(),
        }
    }

    fn validate_dependency_transport(
        &self,
        dependency: DependencyId,
        transport: &DependencyTransport,
        crosses_launch: bool,
    ) -> Result<(), String> {
        let logical = self
            .dependencies
            .get(&dependency)
            .ok_or_else(|| format!("dependency#{} is absent", dependency.0))?;
        let storage = |id: StorageId| {
            self.device_storage
                .allocations
                .iter()
                .find(|storage| storage.id == id)
                .ok_or_else(|| format!("dependency names absent plan storage#{}", id.0))
        };
        match (&logical.kind, transport) {
            (LogicalDependencyKind::Control, DependencyTransport::Control) => {}
            (
                LogicalDependencyKind::Value(expected),
                DependencyTransport::Value { operand, transport },
            ) if expected == operand => {
                if !transport_matches_type(&self.operand_types[operand], transport) {
                    return Err("value dependency transport disagrees with operand type".into());
                }
                if crosses_launch {
                    let Some(ids) = transport_storage_ids(transport) else {
                        return Err("cross-launch value requires retained storage".into());
                    };
                    if ids.into_iter().any(|id| {
                        storage(*id)
                            .map(|value| {
                                !matches!(
                                    value.scope,
                                    StorageScope::External | StorageScope::Device
                                )
                            })
                            .unwrap_or(true)
                    }) {
                        return Err("cross-launch value storage is not retained".into());
                    }
                }
            }
            (
                LogicalDependencyKind::Effect(expected),
                DependencyTransport::Effect {
                    logical_storage,
                    storage: ids,
                },
            )
            | (
                LogicalDependencyKind::Ownership(expected),
                DependencyTransport::Ownership {
                    logical_storage,
                    storage: ids,
                },
            ) if expected == logical_storage => {
                for id in ids.iter() {
                    match storage(*id) {
                        Ok(value)
                            if value.provenance.logical_storage.as_ref() == Some(expected)
                                && (!crosses_launch
                                    || matches!(
                                        value.scope,
                                        StorageScope::External | StorageScope::Device
                                    )) => {}
                        Ok(_) | Err(_) => {
                            return Err(format!(
                                "dependency#{} storage plane#{} lacks exact retained provenance",
                                dependency.0, id.0
                            ));
                        }
                    }
                }
            }
            _ => {
                return Err(format!(
                    "dependency#{} transport disagrees with its logical kind",
                    dependency.0
                ));
            }
        }
        Ok(())
    }

    fn validate_plan_value_transport(
        &self,
        operand: OperandId,
        transport: &ValueTransportTemplate,
    ) -> Result<(), String> {
        if !transport_matches_type(&self.operand_types[&operand], transport) {
            return Err(format!(
                "operand#{} transport disagrees with its logical type",
                operand.0
            ));
        }
        let Some(ids) = transport_storage_ids(transport) else {
            return Err("plan boundary requires storage-backed value transport".into());
        };
        let mut unique = BTreeSet::new();
        for id in ids {
            if !unique.insert(*id) {
                return Err(format!(
                    "operand#{} repeats storage plane#{}",
                    operand.0, id.0
                ));
            }
            let allocation = self
                .device_storage
                .allocations
                .iter()
                .find(|allocation| allocation.id == *id)
                .ok_or_else(|| format!("operand#{} names absent storage#{}", operand.0, id.0))?;
            if !matches!(
                allocation.scope,
                StorageScope::External | StorageScope::Device
            ) {
                return Err("plan-boundary storage transport is not retained".into());
            }
            let exact_operand = allocation.provenance.operand == Some(operand);
            let exact_storage = self.operand_storage.get(&operand)
                == allocation.provenance.logical_storage.as_ref();
            if !matches!(transport, ValueTransportTemplate::Tuple(_))
                && !exact_operand
                && !exact_storage
            {
                return Err(format!(
                    "operand#{} storage plane#{} lacks exact provenance",
                    operand.0, id.0
                ));
            }
        }
        Ok(())
    }

    fn validate_storage(&self, storage: &StorageTemplate<D>) -> Result<(), String> {
        if storage.alignment == 0 || !storage.alignment.is_power_of_two() {
            return Err(format!(
                "storage#{} alignment is not a nonzero power of two",
                storage.id.0
            ));
        }
        let replication_ok = match storage.scope {
            StorageScope::External | StorageScope::Device => {
                storage.replication == Replication::Once
            }
            StorageScope::Workgroup => storage.replication == Replication::PerWorkgroup,
            StorageScope::Participant => matches!(
                storage.replication,
                Replication::PerParticipant | Replication::PerSubgroup
            ),
        };
        if !replication_ok {
            return Err(format!(
                "storage#{} has incompatible scope/replication",
                storage.id.0
            ));
        }
        if let Some(operand) = storage.provenance.operand {
            if !self.operands.contains(&operand) {
                return Err(format!(
                    "storage#{} provenance names absent operand#{}",
                    storage.id.0, operand.0
                ));
            }
            if self.operand_storage.get(&operand) != storage.provenance.logical_storage.as_ref() {
                return Err(format!(
                    "storage#{} provenance disagrees with operand#{} logical storage",
                    storage.id.0, operand.0
                ));
            }
        } else if !storage
            .provenance
            .logical_storage
            .as_ref()
            .is_some_and(|logical| self.effect_storage.contains(logical))
            && !storage
                .provenance
                .logical_storage
                .as_ref()
                .is_some_and(|logical| self.boundary_storage.contains(logical))
            && !matches!(
                storage.provenance.abi,
                Some(AbiRole::InvocationResource { .. })
            )
        {
            return Err(format!(
                "storage#{} has no logical operand, effect, or invocation provenance",
                storage.id.0
            ));
        }
        if storage.scope == StorageScope::External && storage.provenance.abi.is_none() {
            return Err(format!("external storage#{} has no ABI role", storage.id.0));
        }
        if storage.scope != StorageScope::External
            && matches!(
                storage.provenance.abi,
                Some(AbiRole::Parameter { .. } | AbiRole::Result { .. })
            )
        {
            return Err(format!(
                "non-external storage#{} claims a parameter/result ABI role",
                storage.id.0
            ));
        }
        Ok(())
    }

    fn validate_launch(&self, launch: &LaunchTemplate<D>) -> Result<(), String> {
        let mut storage = BTreeMap::<StorageId, &StorageTemplate<D>>::new();
        for value in self
            .device_storage
            .allocations
            .iter()
            .chain(launch.kernel.workgroup_storage.iter())
            .chain(launch.kernel.participant_storage.iter())
        {
            self.validate_storage(value)?;
            if storage.insert(value.id, value).is_some() {
                return Err(format!(
                    "launch#{} has duplicate storage#{}",
                    launch.id.0, value.id.0
                ));
            }
        }
        for value in &launch.kernel.workgroup_storage {
            if value.scope != StorageScope::Workgroup {
                return Err("workgroup storage has non-workgroup scope".into());
            }
        }
        for value in &launch.kernel.participant_storage {
            if value.scope != StorageScope::Participant {
                return Err("participant storage has non-participant scope".into());
            }
        }
        let mut group_ids = BTreeSet::new();
        let mut binding_ids = BTreeSet::new();
        let mut slots = BTreeSet::new();
        let mut bound_storage = BTreeMap::new();
        for group in &launch.binding_groups {
            if !group_ids.insert(group.id) || !slots.insert(group.slot) {
                return Err(format!(
                    "launch#{} repeats a binding-group id or native slot",
                    launch.id.0
                ));
            }
            if group.kind == BindingGroupKind::Direct && group.members.len() != 1 {
                return Err(format!(
                    "direct binding group#{} must contain exactly one member",
                    group.id.0
                ));
            }
            for binding in group.members.iter() {
                if !binding_ids.insert(binding.id) {
                    return Err(format!(
                        "launch#{} repeats binding#{}",
                        launch.id.0, binding.id.0
                    ));
                }
                let allocation = storage.get(&binding.storage).ok_or_else(|| {
                    format!(
                        "binding#{} names absent storage#{}",
                        binding.id.0, binding.storage.0
                    )
                })?;
                if !matches!(
                    allocation.scope,
                    StorageScope::External | StorageScope::Device
                ) {
                    return Err("native bindings may only name external/device storage".into());
                }
                if binding.operand.is_some() && binding.operand != allocation.provenance.operand {
                    return Err(format!(
                        "binding#{} operand disagrees with storage provenance",
                        binding.id.0
                    ));
                }
                if bound_storage
                    .insert(binding.storage, binding.access)
                    .is_some()
                {
                    return Err(format!(
                        "launch#{} binds storage#{} more than once",
                        launch.id.0, binding.storage.0
                    ));
                }
            }
        }
        for step in launch.kernel.steps.iter() {
            match &step.kind {
                KernelStepKind::MappedTask {
                    bindings,
                    instructions,
                    ..
                } => {
                    for binding in bindings.iter() {
                        if let Some(ids) = transport_storage_ids(&binding.transport) {
                            for id in ids {
                                let allocation = storage.get(id).ok_or_else(|| {
                                    format!(
                                        "operand#{} names absent storage#{}",
                                        binding.operand.0, id.0
                                    )
                                })?;
                                let exact_operand =
                                    allocation.provenance.operand == Some(binding.operand);
                                let exact_storage = self.operand_storage.get(&binding.operand)
                                    == allocation.provenance.logical_storage.as_ref();
                                if !exact_operand && !exact_storage {
                                    return Err(format!(
                                        "operand#{} storage transport lacks matching provenance",
                                        binding.operand.0
                                    ));
                                }
                            }
                        }
                    }
                    for instruction in instructions.iter() {
                        for access in D::consequences(instruction).accesses {
                            let allocation = storage.get(&access.storage).ok_or_else(|| {
                                format!("instruction accesses absent storage#{}", access.storage.0)
                            })?;
                            if matches!(
                                allocation.scope,
                                StorageScope::External | StorageScope::Device
                            ) && !bound_storage
                                .get(&access.storage)
                                .is_some_and(|bound| access_permits(*bound, access.mode))
                            {
                                return Err(format!(
                                    "instruction access to storage#{} is not covered by a binding",
                                    access.storage.0
                                ));
                            }
                        }
                    }
                }
                KernelStepKind::Barrier { transport, .. } => match transport {
                    DependencyTransport::Effect {
                        logical_storage,
                        storage: ids,
                    }
                    | DependencyTransport::Ownership {
                        logical_storage,
                        storage: ids,
                    } => {
                        if ids.iter().any(|id| {
                            storage.get(id).and_then(|allocation| {
                                allocation.provenance.logical_storage.as_ref()
                            }) != Some(logical_storage)
                        }) {
                            return Err("barrier storage transport lacks exact provenance".into());
                        }
                    }
                    DependencyTransport::Value { transport, .. }
                        if transport_storage_ids(transport).is_some_and(|ids| {
                            ids.into_iter().any(|id| !storage.contains_key(id))
                        }) =>
                    {
                        return Err("barrier value transport names absent storage".into());
                    }
                    _ => {}
                },
                KernelStepKind::Publish { .. } => {}
            }
        }
        Ok(())
    }

    fn validate_dependency_order(&self) -> Result<(), String> {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        struct Position {
            item: i128,
            launch: usize,
            step: usize,
        }
        let start = Position {
            item: -1,
            launch: 0,
            step: 0,
        };
        let end = Position {
            item: i128::MAX,
            launch: usize::MAX,
            step: usize::MAX,
        };
        let mut endpoints = BTreeMap::<LogicalEndpoint, Position>::new();
        let mut barriers = BTreeMap::<DependencyId, Position>::new();
        for (item_index, item) in self.items.iter().enumerate() {
            match item {
                ScheduleItem::Subplan(call) => {
                    endpoints.insert(
                        LogicalEndpoint::Call(call.call),
                        Position {
                            item: item_index as i128,
                            launch: 0,
                            step: 0,
                        },
                    );
                }
                ScheduleItem::Phase(phase) => {
                    for (launch_index, launch) in phase.launches.iter().enumerate() {
                        for (step_index, step) in launch.kernel.steps.iter().enumerate() {
                            let position = Position {
                                item: item_index as i128,
                                launch: launch_index,
                                step: step_index,
                            };
                            match &step.kind {
                                KernelStepKind::MappedTask { task, .. } => {
                                    endpoints.insert(LogicalEndpoint::Task(*task), position);
                                }
                                KernelStepKind::Barrier { dependency, .. } => {
                                    barriers.insert(*dependency, position);
                                }
                                KernelStepKind::Publish { output, .. } => {
                                    endpoints.insert(LogicalEndpoint::Output(*output), position);
                                }
                            }
                        }
                    }
                }
            }
        }
        let placement = self
            .dependency_placements
            .iter()
            .map(|placement| (placement.dependency, placement))
            .collect::<BTreeMap<_, _>>();
        for dependency in self.dependencies.values() {
            let from = match dependency.from {
                LogicalEndpoint::Input(_) => start,
                endpoint => *endpoints.get(&endpoint).ok_or_else(|| {
                    format!("dependency#{} source was not scheduled", dependency.id.0)
                })?,
            };
            let to = match dependency.to {
                LogicalEndpoint::Output(_) => *endpoints.get(&dependency.to).unwrap_or(&end),
                endpoint => *endpoints.get(&endpoint).ok_or_else(|| {
                    format!(
                        "dependency#{} destination was not scheduled",
                        dependency.id.0
                    )
                })?,
            };
            if from >= to {
                return Err(format!(
                    "dependency#{} is placed before its source or after its destination",
                    dependency.id.0
                ));
            }
            let placed = placement.get(&dependency.id).ok_or_else(|| {
                format!("dependency#{} has no physical placement", dependency.id.0)
            })?;
            if let (
                LogicalDependencyKind::Value(operand),
                DependencyTransport::Value {
                    operand: physical_operand,
                    transport,
                },
            ) = (&dependency.kind, &placed.transport)
            {
                if operand != physical_operand
                    || self.endpoint_transport(dependency.from, *operand) != Some(transport.clone())
                    || self.endpoint_transport(dependency.to, *operand) != Some(transport.clone())
                {
                    return Err(format!(
                        "dependency#{} value transport is not the exact producer/consumer channel",
                        dependency.id.0
                    ));
                }
                let crosses_launch = from.item != to.item || from.launch != to.launch;
                if crosses_launch {
                    let Some(storages) = transport_storage_ids(transport) else {
                        return Err(format!(
                            "dependency#{} carries a kernel value across a launch boundary",
                            dependency.id.0
                        ));
                    };
                    if storages.into_iter().any(|storage| {
                        !self.device_storage.allocations.iter().any(|allocation| {
                            allocation.id == *storage
                                && matches!(
                                    allocation.scope,
                                    StorageScope::External | StorageScope::Device
                                )
                        })
                    }) {
                        return Err(format!(
                            "dependency#{} cross-launch storage is not retained",
                            dependency.id.0
                        ));
                    }
                }
            }
            if let Some(barrier) = barriers.get(&dependency.id) {
                if placed.order != DependencyOrder::Barrier {
                    return Err(format!(
                        "dependency#{} barrier disagrees with its placement order",
                        dependency.id.0
                    ));
                }
                if from.item != barrier.item
                    || from.launch != barrier.launch
                    || to.item != barrier.item
                    || to.launch != barrier.launch
                    || !(from < *barrier && *barrier < to)
                {
                    return Err(format!(
                        "dependency#{} barrier is outside its producer/consumer launch scope",
                        dependency.id.0
                    ));
                }
            } else {
                if placed.order == DependencyOrder::Barrier {
                    return Err(format!(
                        "dependency#{} has barrier placement but no barrier step",
                        dependency.id.0
                    ));
                }
                if placed.order == DependencyOrder::LaunchBoundary
                    && from.item == to.item
                    && from.launch == to.launch
                {
                    return Err(format!(
                        "dependency#{} launch boundary does not cross launches",
                        dependency.id.0
                    ));
                }
            }
        }
        Ok(())
    }
}

fn access_permits(binding: AccessMode, requested: AccessMode) -> bool {
    match binding {
        AccessMode::Read => requested == AccessMode::Read,
        AccessMode::Write => requested == AccessMode::Write,
        AccessMode::ReadWrite => matches!(
            requested,
            AccessMode::Read | AccessMode::Write | AccessMode::ReadWrite
        ),
        // An atomic read-modify-write place is also read while its address and
        // prior value are formed. It must not, however, admit an independent
        // non-atomic write through the same binding.
        AccessMode::Atomic => matches!(requested, AccessMode::Read | AccessMode::Atomic),
    }
}

fn transport_matches_type(ty: &Type, transport: &ValueTransportTemplate) -> bool {
    match (ty, transport) {
        (Type::Void, ValueTransportTemplate::Void) => true,
        (Type::Void, _) | (_, ValueTransportTemplate::Void) => false,
        (Type::Tuple(types), ValueTransportTemplate::Tuple(values)) => {
            types.len() == values.len()
                && types
                    .iter()
                    .zip(values.iter())
                    .all(|(ty, value)| transport_matches_type(ty, value))
        }
        (Type::Tuple(_), _) | (_, ValueTransportTemplate::Tuple(_)) => false,
        (_, ValueTransportTemplate::Kernel(_)) => true,
        (_, ValueTransportTemplate::Storage(_)) => true,
    }
}

fn transport_storage_ids(transport: &ValueTransportTemplate) -> Option<Vec<&StorageId>> {
    match transport {
        ValueTransportTemplate::Void => Some(Vec::new()),
        ValueTransportTemplate::Kernel(_) => None,
        ValueTransportTemplate::Storage(ids) => Some(ids.iter().collect()),
        ValueTransportTemplate::Tuple(fields) => {
            let mut result = Vec::new();
            for field in fields.iter() {
                result.extend(transport_storage_ids(field)?);
            }
            Some(result)
        }
    }
}

fn flatten_type_paths(ty: &Type, path: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
    match ty {
        Type::Void => {}
        Type::Tuple(fields) => {
            for (field, ty) in fields.iter().enumerate() {
                path.push(field as u32);
                flatten_type_paths(ty, path, out);
                path.pop();
            }
        }
        _ => out.push(path.clone()),
    }
}

fn validate_task_graph(graph: &LogicalTaskGraph) -> Result<(), String> {
    for (ordinal, task) in graph.tasks.iter().enumerate() {
        if task.id.0 as usize != ordinal {
            return Err(format!(
                "task graph#{} has noncanonical task ids",
                graph.id.0
            ));
        }
    }
    for (ordinal, call) in graph.calls.iter().enumerate() {
        if call.id.0 as usize != ordinal {
            return Err(format!(
                "task graph#{} has noncanonical call ids",
                graph.id.0
            ));
        }
    }
    for (ordinal, operand) in graph.operands.iter().enumerate() {
        if operand.id.0 as usize != ordinal {
            return Err(format!(
                "task graph#{} has noncanonical operand ids",
                graph.id.0
            ));
        }
    }
    let endpoint_exists = |endpoint: LogicalEndpoint| match endpoint {
        LogicalEndpoint::Input(port) => (port as usize) < graph.inputs.len(),
        LogicalEndpoint::Task(task) => graph.task(task).is_some(),
        LogicalEndpoint::Call(call) => graph.call(call).is_some(),
        LogicalEndpoint::Output(port) => (port as usize) < graph.results.len(),
    };
    for (ordinal, dependency) in graph.dependencies.iter().enumerate() {
        if dependency.id.0 as usize != ordinal
            || !endpoint_exists(dependency.from)
            || !endpoint_exists(dependency.to)
            || dependency.from == dependency.to
        {
            return Err(format!(
                "task graph#{} has invalid dependency#{}",
                graph.id.0, dependency.id.0
            ));
        }
        if let LogicalDependencyKind::Value(operand) = dependency.kind {
            if graph.operand(operand).is_none() {
                return Err(format!(
                    "dependency#{} names absent operand#{}",
                    dependency.id.0, operand.0
                ));
            }
        }
    }
    Ok(())
}

impl TaskObligation {
    pub fn id(&self) -> TaskId {
        self.task
    }

    pub fn graph(&self) -> TaskGraphId {
        self.graph
    }
}

impl CallObligation {
    pub fn id(&self) -> CallBoundaryId {
        self.call
    }

    pub fn graph(&self) -> TaskGraphId {
        self.graph
    }
}

impl OutputObligation {
    pub fn id(&self) -> u32 {
        self.output
    }

    pub fn graph(&self) -> TaskGraphId {
        self.graph
    }
}
