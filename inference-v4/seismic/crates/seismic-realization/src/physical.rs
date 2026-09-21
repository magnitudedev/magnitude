//! Sealed physical plans (package P1).
//!
//! `PhysicalPlan` fields are not publicly constructible. Resolution creates
//! an unsealed internal value, checks totality once, converts all sparse ids
//! to dense typed indices, proves expression bounds under the workload
//! envelope, and returns `PhysicalPlan` only after sealing. Every reference
//! is dense and in-bounds by construction.

use crate::ids::{
    BranchIx, BufferSlot, CallIx, DenseMap, GuardIx, InvocationValueId, LaunchIx, NativeFactIx,
    ObligationRef, RepeatIx, ResultFieldIx, ScalarSlot, ScalarSlotIx, StatusFieldIx, StorageIx,
};
use crate::invocation::{GuardedExecutionExpr, InvocationContract};
use crate::kernel::{ClosedKernelBlock, ExecutableDialect};
use crate::residence::{Replication, StoragePlane};
use crate::routes::ViewTransformTemplate;
use seismic_lang::logical::Access;
use seismic_lang::precision::NumericalAssessment;
use seismic_lang::sir::LoopKind;
use seismic_lang::types::{DType, NonEmpty, TensorType};

/// Where one physical storage lives after resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoragePlacement {
    Abi { slot: BufferSlot },
    Arena { offset: u64 },
    Workgroup,
    Participant,
}

/// The address-space fact of one launch binding, in binding-slot order:
/// caller or device memory, or one launch-local allocation whose bytes and
/// alignment the encoder/backend declares natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageFact {
    Global,
    Workgroup { bytes: u64, alignment: u64 },
    Participant { bytes: u64, alignment: u64 },
}

/// One resolved physical storage plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalStorage<D: ExecutableDialect> {
    pub placement: StoragePlacement,
    pub replication: Replication,
    /// Bytes at capacity.
    pub bytes: u64,
    pub alignment: u64,
    pub plane: StoragePlane,
    pub layout: D::ResolvedLayout,
}

/// One storage view: storage, access, the transform in storage coordinates
/// with dense scalar endpoints, and the view-side tensor type the encoder
/// addresses (the routed value's type — including any trailing `Reshape` —
/// not the residence's storage type).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalStorageView {
    pub storage: StorageIx,
    pub access: Access,
    pub transform: ViewTransformTemplate,
    pub ty: TensorType,
}

/// A retained execution expression: invocation-known, guarded executor
/// arithmetic, a reflected native fact, or one field of the compiler-owned
/// result block (written before every use by schedule order). No unchecked
/// variant exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionExpr {
    Invocation(InvocationValueId),
    Guarded(GuardedExecutionExpr),
    NativeFact(NativeFactIx),
    ResultField(ResultFieldIx),
}

/// One executed launch, as observed by its executor: the geometry it
/// dispatched and the wall time of its execution (encoding, submission, and
/// completion included). Executors record one per launch they run; the
/// runtime folds them into `ExecutionObservation` so every consumer of an
/// executed invocation sees exactly what ran, at what size, for how long.
#[derive(Clone, Debug)]
pub struct LaunchExecution {
    pub launch: crate::ids::LaunchIx,
    pub work_items: u64,
    pub participants: u64,
    pub workgroups: [u64; 3],
    pub seconds: f64,
}

/// One launch of the sealed schedule: everything an encoder needs, with
/// nothing to look up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedLaunch<D: ExecutableDialect> {
    pub id: LaunchIx,
    pub work_items: ExecutionExpr,
    pub participants: ExecutionExpr,
    pub workgroups: [ExecutionExpr; 3],
    pub bindings: Vec<LaunchBinding>,
    /// One entry per `bindings` entry, in binding-slot order: the
    /// address-space fact of the bound storage.
    pub storage_facts: Vec<StorageFact>,
    pub workgroup_storage: Vec<StorageIx>,
    pub participant_storage: Vec<StorageIx>,
    pub kernel: ClosedKernelBlock<D::Intrinsic>,
    /// One entry per `KernelInputId` of the kernel interface, in id order.
    pub inputs: Vec<LaunchInput>,
    /// One entry per `KernelOutputId`, in id order.
    pub outputs: Vec<LaunchOutput>,
    /// One entry per `KernelLocalId`, in id order.
    pub locals: Vec<NonEmpty<PhysicalStorageView>>,
    /// One entry per status field template of the kernel, in kernel order.
    pub status_fields: Vec<StatusFieldIx>,
    /// The pull-counter storage of a `DynamicPull` block (zeroed by the
    /// `Fill` step that precedes this launch and claimed from by the
    /// encoded pull loop); `None` for every other policy.
    pub pull_counter: Option<StorageIx>,
    pub resources: LaunchResources,
    pub native_facts: Vec<NativeFactDeclaration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchBinding {
    pub slot: u32,
    pub storage: StorageIx,
    pub access: AccessMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

/// The physical source of one kernel input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchInput {
    Storage(NonEmpty<PhysicalStorageView>),
    Scalar { slot: u32, source: ScalarSource, dtype: DType },
}

/// The physical destination of one kernel output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchOutput {
    Storage(NonEmpty<PhysicalStorageView>),
    ExecutorSlot { slot: ScalarSlotIx, dtype: DType },
    ResultField { field: ResultFieldIx, dtype: DType },
}

/// Where a by-value kernel scalar comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarSource {
    Abi(ScalarSlot),
    Executor(ScalarSlotIx),
    Invocation(InvocationValueId),
    /// One field of the compiler-owned result scalar block (written before
    /// every use by schedule order).
    Result(ResultFieldIx),
}

/// The exact selected resource contract of one launch, validated identically
/// against reflected native facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchResources {
    pub workgroup_bytes: u64,
    pub private_bytes_per_participant: u64,
    pub direct_bindings: u32,
    pub static_code_units: u64,
    pub required_subgroup_width: Option<u32>,
    pub barriers: u32,
}

/// One native fact the assembler must reflect for this launch, with its
/// admissible domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFactDeclaration {
    pub index: NativeFactIx,
    pub kind: crate::strategy::NativeFactKind,
    pub min: u64,
    pub max: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedGuard {
    pub id: GuardIx,
    pub obligation: ObligationRef,
    pub status: StatusFieldIx,
    pub predicate: GuardPredicate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardPredicate {
    ProductFits { factors: Vec<ExecutionExpr>, bits: u8 },
    ExtentPositive { extent: ExecutionExpr },
    RangeOrdered { start: ExecutionExpr, end: ExecutionExpr, bound: ExecutionExpr },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedCall<D: ExecutableDialect> {
    pub id: CallIx,
    pub body: PhysicalSchedule<D>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedBranch<D: ExecutableDialect> {
    pub id: BranchIx,
    pub condition: ScalarSource,
    pub then_schedule: PhysicalSchedule<D>,
    pub else_schedule: PhysicalSchedule<D>,
    /// Value joins across the branch: `joined` holds the taken side's value
    /// after the branch. State joins are absent: joined state threads through
    /// its storage by route equality.
    pub joins: Vec<SealedJoin>,
}

/// One value crossing a schedule boundary: a kernel scalar or a tensor view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SealedValue {
    Scalar(ScalarSource),
    Tensor(NonEmpty<PhysicalStorageView>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedJoin {
    pub then_value: SealedValue,
    pub else_value: SealedValue,
    pub joined: SealedValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedRepeat<D: ExecutableDialect> {
    pub id: RepeatIx,
    pub kind: LoopKind,
    pub start: ExecutionExpr,
    pub end: ExecutionExpr,
    pub bound: ExecutionExpr,
    pub binder: ScalarSlotIx,
    pub body: PhysicalSchedule<D>,
    /// Value carries threaded by the executor: `current` is rebound from
    /// `initial` before the first visit and from `update` after each visit;
    /// `result` holds the final value. State carries are absent: carried
    /// state threads through its storage.
    pub carries: Vec<SealedCarry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedCarry {
    pub initial: SealedValue,
    pub current: SealedValue,
    pub update: SealedValue,
    pub result: SealedValue,
}

/// One host-side fill: zeroes `bytes` bytes of one storage. The pull-counter
/// reset step lowers to this; it is executed by the host before the launch
/// that follows it and is never backend-encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SealedFill {
    pub storage: StorageIx,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalStep<D: ExecutableDialect> {
    Launch(SealedLaunch<D>),
    Guard(SealedGuard),
    Call(SealedCall<D>),
    If(SealedBranch<D>),
    Repeat(SealedRepeat<D>),
    Fill(SealedFill),
}

/// The structured execution tree. Empty is the valid identity schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalSchedule<D: ExecutableDialect> {
    pub steps: Vec<PhysicalStep<D>>,
}

/// One field of the compiler-owned result scalar block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultField {
    pub index: ResultFieldIx,
    pub path: seismic_lang::types::ValuePath,
    pub endpoint: Option<seismic_lang::abi::RangeEndpoint>,
    pub dtype: DType,
}

/// One field of the root status block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusField {
    pub index: StatusFieldIx,
    pub obligation: ObligationRef,
    pub kind: crate::failure::SafetyKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanResources {
    pub arena_bytes: u64,
    pub max_workgroup_bytes: u64,
    pub max_private_bytes_per_participant: u64,
    pub direct_bindings: u32,
    pub scalar_slots: u32,
    pub result_bytes: u64,
    pub status_bytes: u64,
}

/// Identity of one resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionIdentity {
    pub logical: seismic_lang::logical::LogicalIdentity,
    pub entry: String,
    pub target: seismic_lang::logical::EffectiveTargetIdentity,
    pub toolchain_fingerprint: String,
    pub selections: Vec<(crate::ids::OccurrenceId, Option<crate::ids::StrategyId>)>,
}

/// The sealed physical plan. Construction is private to the seal.
pub struct PhysicalPlan<D: ExecutableDialect> {
    identity: ResolutionIdentity,
    contract: InvocationContract,
    storages: DenseMap<StorageIx, PhysicalStorage<D>>,
    schedule: PhysicalSchedule<D>,
    result_fields: DenseMap<ResultFieldIx, ResultField>,
    status_fields: DenseMap<StatusFieldIx, StatusField>,
    /// The actual-value execution expression of every runtime extent of the
    /// program: what a `CoreKernelOp::RuntimeExtent` read evaluates to at
    /// execution (invocation-known or guarded executor arithmetic).
    runtime_extents: std::collections::BTreeMap<seismic_lang::types::RuntimeExtentId, ExecutionExpr>,
    resources: PlanResources,
    numerical: NumericalAssessment,
    estimated_cost: u64,
    optimal: bool,
}

impl<D: ExecutableDialect> PhysicalPlan<D> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seal(
        identity: ResolutionIdentity,
        contract: InvocationContract,
        storages: DenseMap<StorageIx, PhysicalStorage<D>>,
        schedule: PhysicalSchedule<D>,
        result_fields: DenseMap<ResultFieldIx, ResultField>,
        status_fields: DenseMap<StatusFieldIx, StatusField>,
        runtime_extents: std::collections::BTreeMap<
            seismic_lang::types::RuntimeExtentId,
            ExecutionExpr,
        >,
        resources: PlanResources,
        numerical: NumericalAssessment,
        estimated_cost: u64,
        optimal: bool,
    ) -> PhysicalPlan<D> {
        PhysicalPlan {
            identity,
            contract,
            storages,
            schedule,
            result_fields,
            status_fields,
            runtime_extents,
            resources,
            numerical,
            estimated_cost,
            optimal,
        }
    }

    pub fn identity(&self) -> &ResolutionIdentity {
        &self.identity
    }
    pub fn contract(&self) -> &InvocationContract {
        &self.contract
    }
    pub fn storages(&self) -> &DenseMap<StorageIx, PhysicalStorage<D>> {
        &self.storages
    }
    pub fn schedule(&self) -> &PhysicalSchedule<D> {
        &self.schedule
    }
    pub fn result_fields(&self) -> &DenseMap<ResultFieldIx, ResultField> {
        &self.result_fields
    }
    pub fn status_fields(&self) -> &DenseMap<StatusFieldIx, StatusField> {
        &self.status_fields
    }
    /// The actual-value execution expression of one runtime extent read by
    /// the sealed plan (a schedule-level bound or a kernel's
    /// `CoreKernelOp::RuntimeExtent`); total over those ids by the seal's
    /// own table construction. An extent no sealed expression reads has no
    /// entry and is never asked for.
    pub fn runtime_extent(
        &self,
        extent: seismic_lang::types::RuntimeExtentId,
    ) -> &ExecutionExpr {
        &self.runtime_extents[&extent]
    }
    pub fn resources(&self) -> &PlanResources {
        &self.resources
    }
    pub fn numerical(&self) -> &NumericalAssessment {
        &self.numerical
    }
    pub fn estimated_cost(&self) -> u64 {
        self.estimated_cost
    }
    pub fn optimal(&self) -> bool {
        self.optimal
    }
    /// Every launch of the schedule in tree order, for exhaustive encoding.
    pub fn launches(&self) -> Vec<&SealedLaunch<D>> {
        fn walk<'p, D: ExecutableDialect>(
            steps: &'p [PhysicalStep<D>],
            out: &mut Vec<&'p SealedLaunch<D>>,
        ) {
            for step in steps {
                match step {
                    PhysicalStep::Launch(launch) => out.push(launch),
                    PhysicalStep::Guard(_) | PhysicalStep::Fill(_) => {}
                    PhysicalStep::Call(call) => walk(&call.body.steps, out),
                    PhysicalStep::If(branch) => {
                        walk(&branch.then_schedule.steps, out);
                        walk(&branch.else_schedule.steps, out);
                    }
                    PhysicalStep::Repeat(repeat) => walk(&repeat.body.steps, out),
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.schedule.steps, &mut out);
        out
    }
}
