//! Executable representation and the runtime service contracts (spec §7.5,
//! §11.3, §12.3).
//!
//! `ExecutableStep<C, P, R>` is the one structured control type used by the
//! frozen plan (over physical commands) and by every backend's executable
//! variant (over native commands). CPU, Metal, and CUDA do not define
//! schedule mirrors.
//!
//! An `ExecutableVariant<T, H>` contains exactly: identity, exact guard
//! evaluator, allocation/layout evaluators, one structured
//! native schedule, the call-schema binding table, numerical assessment,
//! and provenance. It retains no frozen plan.
//!
//! Runtime executes; it does not prove. The executor trait receives typed
//! commands and an environment of already-validated buffers and slot
//! values; it may report only real external failures and data-check
//! failures.

use crate::errors::ExecutionError;
use crate::frozen::{FrozenPlan, VariantIdentity};
use crate::numerics::NumericalAssessment;
use seismic_ir::schedule::{AnyScalarSlot, HostQuantitySlot};
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::compiled::{Compiled, CompiledInt, CompiledNat, CompiledPredicate, InvocationValues};
use seismic_lang::expr::LoopBinderId;
use seismic_lang::expr::SymbolValue;
use std::fmt;

/// Structured control over commands `C`, predicates `P`, ranges `R`.
#[derive(Debug)]
pub enum ExecutableStep<C, P, R> {
    BindArgumentTensor { allocation: ExecutableAllocationId, representation: seismic_lang::ids::RepresentationId, result: u32, strides: Vec<seismic_lang::expr::SymbolId> },
    BeginAllocationInstance { allocation: ExecutableAllocationId, banks: Vec<ExecutableAllocationId>, bytes: CompiledNat, alignment: u64, geometry: CompiledBufferView, result: CompiledRegionDestination },
    PublishTensor { path: Vec<u32>, view: CompiledBufferView, bytes: CompiledNat, declared_axes: Vec<CompiledNat> },
    Command(C),
    EvaluateHost {
        value: CompiledHostValue,
        to: seismic_ir::schedule::HostValueDestination,
        failure: Option<seismic_ir::kernel::ops::CheckSite>,
    },
    /// A data-dependent semantic check evaluated by the core schedule
    /// driver. This is deliberately not a backend command: a native
    /// executor cannot receive or reject it.
    Check {
        condition: AnyScalarSlot,
        expectation: seismic_ir::schedule::ScalarCheckExpectation,
        site: seismic_ir::kernel::ops::CheckSite,
    },
    If {
        condition: P,
        then_steps: Box<[ExecutableStep<C, P, R>]>,
        else_steps: Box<[ExecutableStep<C, P, R>]>,
        results: seismic_ir::region::Product<CompiledBranchResult>,
    },
    Repeat {
        range: R,
        binder: LoopBinder,
        body: Box<[ExecutableStep<C, P, R>]>,
        carries: seismic_ir::region::Product<CompiledRepeatCarry>,
    },
}

#[derive(Debug)]
pub enum CompiledHostValue {
    Integer(CompiledInt),
    Natural(CompiledNat),
    Bool(CompiledPredicate),
    Word { dtype: seismic_lang::types::DType, value: CompiledInt },
}
impl CompiledHostValue {
    fn retained_metadata_bytes(&self) -> usize {
        match self {
            Self::Integer(value) | Self::Word { value, .. } => value.retained_metadata_bytes(),
            Self::Natural(value) => value.retained_metadata_bytes(),
            Self::Bool(value) => value.retained_metadata_bytes(),
        }
    }
}

#[derive(Debug)]
pub struct CompiledBranchResult {
    then_value: CompiledRegionOperand,
    else_value: CompiledRegionOperand,
    result: CompiledRegionDestination,
}

#[derive(Debug)]
pub struct CompiledRepeatCarry {
    initial: CompiledRegionOperand,
    header: CompiledRegionDestination,
    backedge: CompiledRegionOperand,
    result: CompiledRegionDestination,
}
#[derive(Debug)]
enum CompiledRegionOperand {
    Natural(CompiledNat),
    Integer(CompiledInt),
    Word(seismic_lang::expr::SymbolId),
    Tensor(CompiledBufferView),
}
#[derive(Debug)]
enum CompiledTensorAxisDestination {
    Bound(seismic_lang::expr::SymbolId),
    Invariant(CompiledNat),
}
#[derive(Debug)]
enum CompiledRegionDestination {
    Scalar(AnyScalarSlot),
    Quantity(HostQuantitySlot),
    Tensor { id: u32, extents: Vec<CompiledTensorAxisDestination>, strides: Vec<seismic_lang::expr::SymbolId> },
}
#[derive(Clone, Debug)]
enum RegionValue {
    Scalar(SymbolValue),
    Tensor(TensorDescriptor),
}

/// A lexical loop binder and the symbol its value is bound to during each
/// visit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopBinder {
    pub binder: LoopBinderId,
    pub symbol: seismic_lang::expr::SymbolId,
}

/// A runtime predicate: compiled over invocation symbols and schedule
/// slots.
pub type RuntimePredicate = CompiledPredicate;

/// A runtime half-open range.
#[derive(Debug)]
pub struct RuntimeRange {
    pub start: CompiledNat,
    pub end: CompiledNat,
}

pub type NativeStep<B> = ExecutableStep<ExecutableCommand<B>, RuntimePredicate, RuntimeRange>;

/// Closed ordinal of one native kernel in this executable. Only core can
/// construct it; executors resolve it through the invocation environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutableKernelId(u32);

/// Closed ordinal of one physical allocation in this executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutableAllocationId(u32);
impl ExecutableAllocationId {
    pub fn ordinal(self) -> usize { self.0 as usize }
}


/// One evaluated view binding. Allocation identity and byte base offset are
/// compiled together; native code never assumes every view begins at zero.
#[derive(Debug)]
pub struct CompiledBufferView {
    base: CompiledViewBase,
    pub representation: seismic_lang::ids::RepresentationId,
    pub byte_offset: CompiledNat,
    pub extents: Vec<CompiledNat>,
    pub strides: Vec<CompiledNat>,
}

#[derive(Clone, Copy, Debug)]
enum CompiledViewBase {
    Allocation(ExecutableAllocationId),
    TensorValue(u32),
}

#[derive(Clone, Debug)]
struct TensorDescriptor {
    allocation: ExecutableAllocationId,
    representation: seismic_lang::ids::RepresentationId,
    byte_offset: seismic_lang::expr::BigUint,
    extents: Vec<seismic_lang::expr::BigUint>,
    strides: Vec<seismic_lang::expr::BigUint>,
}

impl CompiledBufferView {
    pub fn allocation_index(&self) -> usize {
        match self.base {
            CompiledViewBase::Allocation(id) => id.0 as usize,
            CompiledViewBase::TensorValue(_) => panic!("root publication must name its ABI allocation"),
        }
    }
}

/// One compiler-derived launch-local layout. Native executors consume these
/// offsets and strides verbatim; they never reconstruct local packing.
#[derive(Debug)]
pub struct CompiledLocalLayout {
    pub kind: seismic_ir::storage::LaunchLocalKind,
    pub representation: seismic_lang::ids::RepresentationId,
    pub byte_offset: CompiledNat,
    pub extents: Vec<CompiledNat>,
    pub strides: Vec<CompiledNat>,
    pub bytes: CompiledNat,
    pub alignment: u64,
}

#[derive(Debug)]
pub struct CompiledAddressableResource {
    pub offset_units: CompiledNat,
    pub units: CompiledNat,
}

/// Exact aligned totals for the three distinct launch-local resource
/// classes. These are compiled from the same expressions constrained by the
/// solver.
#[derive(Debug)]
pub struct CompiledLocalClassTotals {
    pub workgroup_bytes: CompiledNat,
    pub participant_bytes: CompiledNat,
    pub register_bytes: CompiledNat,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaunchScratchBindings {
    workgroup: Option<ExecutableAllocationId>,
    participant: Option<ExecutableAllocationId>,
    register: Option<ExecutableAllocationId>,
}

impl LaunchScratchBindings {
    pub fn allocation(
        self,
        class: seismic_ir::storage::LaunchLocalKind,
    ) -> Option<ExecutableAllocationId> {
        match class {
            seismic_ir::storage::LaunchLocalKind::Workgroup => self.workgroup,
            seismic_ir::storage::LaunchLocalKind::Participant => self.participant,
            seismic_ir::storage::LaunchLocalKind::Register => self.register,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KernelAbiBindings {
    allocations: Box<
        [(
            seismic_ir::target::KernelAbiAllocationRole,
            ExecutableAllocationId,
        )],
    >,
}

impl KernelAbiBindings {
    pub fn allocation(
        &self,
        role: seismic_ir::target::KernelAbiAllocationRole,
    ) -> ExecutableAllocationId {
        self.allocations
            .iter()
            .find_map(|(candidate, allocation)| (*candidate == role).then_some(*allocation))
            .unwrap_or_else(|| panic!("native executor requested undeclared ABI role {role:?}"))
    }
}

/// Core-owned executable command vocabulary. Backends compile kernels and
/// execute these commands; they cannot invent or reinterpret schedule facts.
#[derive(Debug)]
pub enum ExecutableCommand<B: seismic_target::TargetFamily> {
    Launch {
        kernel: ExecutableKernelId,
        descriptor: B::LaunchDescriptor,
        grid: [CompiledNat; 3],
        workgroup: [CompiledNat; 3],
        empty: CompiledPredicate,
        bindings: Vec<CompiledBufferView>,
        nat_args: Vec<CompiledNat>,
        scalar_args: Vec<seismic_lang::expr::SymbolId>,
        /// Semantic result destinations belong to this launch, never cached native code.
        result_slots: Vec<AnyScalarSlot>,
        locals: Vec<CompiledLocalLayout>,
        addressable_resources: Vec<CompiledAddressableResource>,
        local_totals: CompiledLocalClassTotals,
        scratch: LaunchScratchBindings,
        abi: KernelAbiBindings,
    },
    Copy {
        source: CompiledBufferView,
        destination: CompiledBufferView,
        bytes: CompiledNat,
    },
    Fill {
        destination: CompiledBufferView,
        value: seismic_ir::schedule::FillValue,
        bytes: CompiledNat,
    },
    ScalarMove {
        from: AnyScalarSlot,
        to: AnyScalarSlot,
    },
    ScalarRead {
        source: CompiledBufferView,
        bounds: Vec<CompiledPredicate>,
        byte_offset: CompiledNat,
        to: AnyScalarSlot,
    },
}

/// One global allocation the runtime performs before execution.
#[derive(Debug)]
pub struct AllocationPlan {
    pub acquisition: AllocationAcquisition,
    pub kind: ExecutableAllocationKind,
    /// Runtime byte expressions of all logical allocations coalesced into
    /// this physical slot. The driver allocates their maximum.
    byte_candidates: AllocationByteCandidates,
    pub alignment: u64,
}

pub use seismic_ir::storage::AllocationAcquisition;

/// A constructionally non-empty set of byte requirements sharing one
/// physical allocation.  The runtime cannot be handed an allocation whose
/// required size has no definition.
#[derive(Debug)]
pub struct AllocationByteCandidates {
    first: CompiledNat,
    rest: Box<[CompiledNat]>,
}

impl AllocationByteCandidates {
    fn one(value: CompiledNat) -> Self {
        Self {
            first: value,
            rest: Box::new([]),
        }
    }

    fn with_rest(first: CompiledNat, rest: Vec<CompiledNat>) -> Self {
        Self {
            first,
            rest: rest.into_boxed_slice(),
        }
    }

    pub fn first(&self) -> &CompiledNat {
        &self.first
    }

    pub fn rest(&self) -> &[CompiledNat] {
        &self.rest
    }
}

impl AllocationPlan {
    pub fn byte_candidates(&self) -> &AllocationByteCandidates {
        &self.byte_candidates
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutableAllocationKind {
    Global(ExecutableGlobalAllocationKind),
    LaunchScratch {
        launch: u32,
        class: seismic_ir::storage::LaunchLocalKind,
    },
    KernelAbi {
        launch: u32,
        role: seismic_ir::target::KernelAbiAllocationRole,
    },
}

/// Root-executable global storage kinds. Construction-only Imported storage
/// and non-ABI arguments have no variant and therefore cannot reach runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutableGlobalAllocationKind {
    Argument(seismic_lang::ids::ParameterId),
    Result,
    Arena,
    Persistent,
}

#[derive(Debug)]
pub enum ExecutableResultBinding {
    Buffer {
        representation: seismic_lang::ids::RepresentationId,
        rank: usize,
    },
    Scalar {
        slot: AnyScalarSlot,
    },
    Quantity {
        slot: HostQuantitySlot,
    },
    Range {
        start: HostQuantitySlot,
        end: HostQuantitySlot,
    },
}

/// Actual tensor result captured by the selected source publication operation.
/// The allocation names this execution's physical bank, not a nominal output.
#[derive(Clone, Debug)]
pub struct ExecutedTensorPublication {
    pub path: Vec<u32>,
    pub allocation: ExecutableAllocationId,
    pub representation: seismic_lang::ids::RepresentationId,
    pub byte_offset: u64,
    pub bytes: u64,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
}

/// The call-schema binding table: which allocation each argument and result
/// occupies.
#[derive(Debug)]
pub struct CallBindingTable {
    /// Compiled allocation and byte base per parameter, `None` for scalars.
    pub arguments: Vec<Option<CompiledBufferView>>,
    /// Schema result leaves in exact schema order, including aliases and
    /// scalar publications.
    pub results: Vec<(Vec<u32>, ExecutableResultBinding)>,
}

/// Handle-free executable content produced by planning.
struct CompiledVariantBody<B: seismic_target::TargetFamily> {
    invocation: std::sync::Arc<crate::prepared::InvocationContract>,
    device: seismic_target::DeviceDescriptionIdentity,
    native_index_bits: u32,
    identity: VariantIdentity,
    allocations: Vec<AllocationPlan>,
    slots: Vec<AnyScalarSlot>,
    schedule: Box<[NativeStep<B>]>,
    bindings: CallBindingTable,
    provenance: crate::refinement::ImplementationProvenance,
}

#[derive(Clone)]
struct NumericalAcceptance {
    guard: RuntimePredicate,
    assessment: NumericalAssessment,
}

impl NumericalAcceptance {
    fn retained_metadata_bytes(&self) -> usize {
        let regions = &self.assessment.regions;
        let bytes = regions
            .capacity()
            .saturating_mul(std::mem::size_of::<crate::numerics::NumericalRegion>())
            .saturating_add(
                regions
                    .iter()
                    .map(|region| region.scope.retained_metadata_bytes())
                    .sum::<usize>(),
            );
        std::mem::size_of_val(self)
            .saturating_add(self.guard.retained_metadata_bytes())
            .saturating_add(bytes)
    }
}

/// Execution-safe compilation output. It is not a selectable candidate.
pub(crate) struct CompiledCandidate<B: seismic_target::TargetFamily> {
    body: std::sync::Arc<CompiledVariantBody<B>>,
    native: std::sync::Arc<crate::realization::RealizedNativeSet<B>>,
    safety: seismic_lang::expr::BoolExpr,
    implementation: std::sync::Arc<crate::implementation::Implementation<B>>,
    fixed: seismic_lang::expr::PartialAssignment,
}

mod numerical_acceptance;

/// One selected, fully lowered variant before native handles are attached.
pub struct RetainedCandidate<B: seismic_target::TargetFamily> {
    body: std::sync::Arc<CompiledVariantBody<B>>,
    admission: std::sync::Arc<NumericalAcceptance>,
    native: std::sync::Arc<crate::realization::RealizedNativeSet<B>>,
}

/// One native executable variant.
pub struct ExecutableVariant<T: seismic_target::TargetFamily, H> {
    candidate_id: crate::evaluation_session::PreparedCandidateId,
    body: std::sync::Arc<CompiledVariantBody<T>>,
    guard: std::sync::Arc<RuntimePredicate>,
    numerical: std::sync::Arc<NumericalAssessment>,
    kernels: Vec<std::sync::Arc<seismic_target::NativeKernel<T, H>>>,
}

impl<T: seismic_target::TargetFamily> Clone for RetainedCandidate<T> {
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            admission: self.admission.clone(),
            native: self.native.clone(),
        }
    }
}

impl<T: seismic_target::TargetFamily, H> Clone for ExecutableVariant<T, H> {
    fn clone(&self) -> Self {
        Self {
            candidate_id: self.candidate_id,
            body: self.body.clone(),
            guard: self.guard.clone(),
            numerical: self.numerical.clone(),
            kernels: self.kernels.clone(),
        }
    }
}

impl<T: seismic_target::TargetFamily, H> ExecutableVariant<T, H> {
    /// Exact formation outcome selected in this preparation, across evaluation
    /// and publication. Semantic digests may label several such outcomes.
    pub fn prepared_candidate_id(&self) -> crate::evaluation_session::PreparedCandidateId {
        self.candidate_id
    }

    pub fn invocation_contract(&self) -> &crate::prepared::InvocationContract {
        &self.body.invocation
    }

    pub fn device_identity(&self) -> &seismic_target::DeviceDescriptionIdentity {
        &self.body.device
    }
    pub fn identity(&self) -> &VariantIdentity {
        &self.body.identity
    }
    pub fn guard(&self) -> &RuntimePredicate {
        &self.guard
    }
    pub fn allocations(&self) -> &[AllocationPlan] {
        &self.body.allocations
    }
    pub fn slots(&self) -> &[AnyScalarSlot] {
        &self.body.slots
    }
    pub fn bindings(&self) -> &CallBindingTable {
        &self.body.bindings
    }
    pub fn numerical(&self) -> &NumericalAssessment {
        &self.numerical
    }
    pub fn provenance(&self) -> &crate::refinement::ImplementationProvenance {
        &self.body.provenance
    }
}

impl<T: seismic_target::TargetFamily, H> fmt::Debug for ExecutableVariant<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutableVariant")
            .field("identity", self.identity())
            .field("kernels", &self.kernels.len())
            .finish()
    }
}

/// The only FrozenPlan -> executable transition. Core consumes every closed
/// fact and compiles evaluators and bindings around the already-reflected
/// native kernels owned by the implementation.
pub(crate) fn compile_variant<B: seismic_target::TargetFamily>(
    plan: FrozenPlan<'_, B>,
) -> CompiledCandidate<B> {
    let parts = plan.into_exact_parts();
    let arena = &*parts.arena;
    let implementation = &*parts.implementation;
    let topology = implementation.global_allocations();
    let kernel_arena = implementation.kernels();
    let native = implementation.native();
    let (allocation_remap, mut allocations) = compile_allocations(arena, &parts.fixed, topology);
    let launch_bindings = compile_launch_resources(
        arena,
        &parts.fixed,
        implementation.launch_resources(),
        &mut allocations,
    );
    let view = |value: seismic_ir::storage::AnyBufferView| {
        compile_view(arena, &parts.fixed, topology, &allocation_remap, value)
    };
    let schedule = compile_steps(
        native,
        arena,
        &parts.fixed,
        topology,
        &allocation_remap,
        kernel_arena,
        implementation.schedule(),
        implementation.launch_resources(),
        &launch_bindings,
        implementation.schedule().steps(),
    )
    .into_boxed_slice();
    let mut arguments = Vec::with_capacity(parts.invocation.schema().parameters().len());
    for parameter in parts.invocation.schema().parameters() {
        let binding = topology.allocations().iter().enumerate().find_map(|(allocation, value)| {
            matches!(&value.kind, GlobalBufferKind::Argument { abi: Some(id), .. } if id == &parameter.id)
                .then(|| primary_view(topology, allocation as u32))
        }).map(view);
        arguments.push(binding);
    }
    let results = implementation
        .result_publications()
        .iter()
        .map(|publication| {
            let binding = match publication.binding {
                crate::refinement::PublishedResult::Buffer {
                    representation, rank,
                } => ExecutableResultBinding::Buffer {
                    representation, rank,
                },
                crate::refinement::PublishedResult::Scalar { slot } => {
                    ExecutableResultBinding::Scalar { slot }
                }
                crate::refinement::PublishedResult::Quantity { slot } => {
                    ExecutableResultBinding::Quantity { slot }
                }
                crate::refinement::PublishedResult::Range { start, end } => {
                    ExecutableResultBinding::Range { start, end }
                }
            };
            (publication.path.clone(), binding)
        })
        .collect();
    let bindings = CallBindingTable { arguments, results };
    CompiledCandidate {
        native: native.clone(),
        safety: parts.guard.node(),
        fixed: parts.fixed.clone(),
        implementation: parts.implementation.clone(),
        body: std::sync::Arc::new(CompiledVariantBody {
            invocation: parts.invocation,
            device: parts.device,
            native_index_bits: implementation.native_index_bits(),
            identity: parts.identity,
            allocations,
            slots: implementation.schedule().slots().to_vec(),
            schedule,
            bindings,
            provenance: implementation.provenance().clone(),
        }),
    }
}

impl<B: seismic_target::TargetFamily> RetainedCandidate<B> {
    pub(crate) fn native_instances(&self) -> impl Iterator<Item = crate::realization::NativeArtifactInstanceId> + '_ {
        self.native.artifact_instances()
    }

    pub fn identity(&self) -> &VariantIdentity {
        &self.body.identity
    }
    pub fn guard(&self) -> &RuntimePredicate {
        &self.admission.guard
    }

    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
        // The final variant resolves each native instance to one Arc. The
        // planning budget therefore accounts the projected handle table
        // size without requiring the handles themselves.
        let handle_bytes = self.native.artifact_instances().len()
            * std::mem::size_of::<std::sync::Arc<()>>();
        let temporary = std::mem::size_of_val(self)
            .saturating_add(self.native.retained_metadata_bytes());
        let materialized = std::mem::size_of::<ExecutableVariant<B, ()>>()
            .saturating_add(handle_bytes)
            .saturating_add(self.body.retained_metadata_bytes())
            .saturating_add(self.admission.retained_metadata_bytes());
        u64::try_from(materialized.max(temporary)).unwrap_or(u64::MAX)
    }
}

pub(crate) trait EvaluatedVariant {
    fn selection_guard(&self) -> &RuntimePredicate;
}

impl<B: seismic_target::TargetFamily> EvaluatedVariant for RetainedCandidate<B> {
    fn selection_guard(&self) -> &RuntimePredicate {
        self.guard()
    }
}

impl<T: seismic_target::TargetFamily, H> EvaluatedVariant for ExecutableVariant<T, H> {
    fn selection_guard(&self) -> &RuntimePredicate {
        self.guard()
    }
}

/// The single invocation-selection policy shared before and after exact
/// native materialization.
pub(crate) fn select_candidate_index<V: EvaluatedVariant>(
    selection: &crate::prepared::SelectionFunction,
    variants: &[V],
    values: &InvocationValues,
) -> usize {
    let universal = variants
        .first()
        .expect("a planned portfolio always contains its universal variant");
    assert!(
        universal
            .selection_guard()
            .evaluate(values)
            .unwrap_or_else(|error| {
                panic!("validated invocation could not evaluate the universal guard: {error:?}")
            }),
        "validated invocation escaped the constructionally total universal variant"
    );
    let selected = selection.apply(values).as_usize();
    let candidate = variants
        .get(selected)
        .unwrap_or_else(|| panic!("selection function names an absent retained candidate"));
    assert!(
        candidate
            .selection_guard()
            .evaluate(values)
            .unwrap_or_else(|error| {
                panic!(
                    "validated invocation could not evaluate selected candidate guard: {error:?}"
                )
            }),
        "selection function chose a candidate outside its applicability guard"
    );
    selected
}

impl<B: seismic_target::TargetFamily> CompiledVariantBody<B> {
    fn retained_metadata_bytes(&self) -> usize {
        fn vec_storage<T>(value: &Vec<T>) -> usize {
            value.capacity().saturating_mul(std::mem::size_of::<T>())
        }
        fn view(value: &CompiledBufferView) -> usize {
            vec_storage(&value.extents).saturating_add(vec_storage(&value.strides))
        }
        fn command<B: seismic_target::TargetFamily>(value: &ExecutableCommand<B>) -> usize {
            match value {
                ExecutableCommand::Launch {
                    bindings,
                    nat_args,
                    scalar_args,
                    result_slots,
                    locals,
                    addressable_resources,
                    abi,
                    ..
                } => vec_storage(bindings)
                    .saturating_add(bindings.iter().map(view).sum::<usize>())
                    .saturating_add(vec_storage(nat_args))
                    .saturating_add(vec_storage(scalar_args))
                    .saturating_add(vec_storage(result_slots))
                    .saturating_add(vec_storage(locals))
                    .saturating_add(
                        locals
                            .iter()
                            .map(|local| {
                                vec_storage(&local.extents)
                                    .saturating_add(vec_storage(&local.strides))
                            })
                            .sum::<usize>(),
                    )
                    .saturating_add(vec_storage(addressable_resources))
                    .saturating_add(abi.allocations.len().saturating_mul(std::mem::size_of::<(
                        seismic_ir::target::KernelAbiAllocationRole,
                        ExecutableAllocationId,
                    )>())),
                ExecutableCommand::Copy {
                    source,
                    destination,
                    ..
                } => view(source).saturating_add(view(destination)),
                ExecutableCommand::Fill { destination, .. } => view(destination),
                ExecutableCommand::ScalarRead { source, bounds, .. } => {
                    view(source).saturating_add(vec_storage(bounds))
                }
                ExecutableCommand::ScalarMove { .. } => 0,
            }
        }
        fn operand_bytes(value: &CompiledRegionOperand) -> usize {
            match value {
                CompiledRegionOperand::Natural(value)=>value.retained_metadata_bytes(),
                CompiledRegionOperand::Integer(value)=>value.retained_metadata_bytes(),
                CompiledRegionOperand::Word(_)=>0,
                CompiledRegionOperand::Tensor(value)=>view(value),
            }
        }
        fn destination_bytes(value: &CompiledRegionDestination) -> usize {
            match value {
                CompiledRegionDestination::Tensor { extents,strides,.. } => extents.capacity()*std::mem::size_of::<CompiledTensorAxisDestination>() + strides.capacity()*std::mem::size_of::<seismic_lang::expr::SymbolId>() + extents.iter().map(|value|match value {CompiledTensorAxisDestination::Invariant(value)=>value.retained_metadata_bytes(), _=>0}).sum::<usize>(),
                _=>0,
            }
        }
        fn steps<B: seismic_target::TargetFamily>(values: &[NativeStep<B>]) -> usize {
            values
                .len()
                .saturating_mul(std::mem::size_of::<NativeStep<B>>())
                .saturating_add(
                    values
                        .iter()
                        .map(|step| match step {
                            ExecutableStep::BindArgumentTensor { .. } => 0,
                            ExecutableStep::BeginAllocationInstance { banks, geometry, .. } => banks.capacity() * std::mem::size_of::<ExecutableAllocationId>() + view(geometry),
                            ExecutableStep::PublishTensor { path, view: buffer, declared_axes, .. } => path.capacity() * std::mem::size_of::<u32>() + view(buffer) + vec_storage(declared_axes),
                            ExecutableStep::Command(value) => command(value),
                            ExecutableStep::EvaluateHost { value, failure, .. } => value.retained_metadata_bytes()
                                .saturating_add(failure.as_ref().map_or(0, |site| site.path.capacity())),
                            ExecutableStep::If {
                                then_steps,
                                else_steps,
                                results,
                                ..
                            } => steps(then_steps).saturating_add(steps(else_steps)).saturating_add(results.retained_heap_bytes(&|result|operand_bytes(&result.then_value)+operand_bytes(&result.else_value)+destination_bytes(&result.result))),
                            ExecutableStep::Repeat { body, carries, .. } => steps(body).saturating_add(carries.retained_heap_bytes(&|carry|operand_bytes(&carry.initial)+operand_bytes(&carry.backedge)+destination_bytes(&carry.header)+destination_bytes(&carry.result))),
                            ExecutableStep::Check { .. } => 0,
                        })
                        .sum::<usize>(),
                )
        }
        fn result(value: &ExecutableResultBinding) -> usize {
            match value {
                ExecutableResultBinding::Buffer { .. } => 0,
                ExecutableResultBinding::Scalar { .. } | ExecutableResultBinding::Quantity { .. } | ExecutableResultBinding::Range { .. } => 0,
            }
        }
        vec_storage(&self.allocations)
            .saturating_add(
                self.allocations
                    .iter()
                    .map(|allocation| {
                        allocation.byte_candidates.rest.len() * std::mem::size_of::<CompiledNat>()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(vec_storage(&self.slots))
            .saturating_add(steps(&self.schedule))
            .saturating_add(vec_storage(&self.bindings.arguments))
            .saturating_add(
                self.bindings
                    .arguments
                    .iter()
                    .flatten()
                    .map(view)
                    .sum::<usize>(),
            )
            .saturating_add(vec_storage(&self.bindings.results))
            .saturating_add(
                self.bindings
                    .results
                    .iter()
                    .map(|(path, binding)| vec_storage(path).saturating_add(result(binding)))
                    .sum::<usize>(),
            )
            .saturating_add(vec_storage(&self.provenance.callees))
    }
}

pub(crate) fn materialize_variant<T: seismic_target::TargetFamily, H>(
    planned: &RetainedCandidate<T>,
    registry: &crate::realization::RealizationRegistry<T, H>,
    candidate_id: crate::evaluation_session::PreparedCandidateId,
) -> ExecutableVariant<T, H> {
    let kernels = registry.resolve(
        &planned.body.device,
        &planned.native,
    );
    ExecutableVariant {
        candidate_id,
        body: planned.body.clone(),
        guard: std::sync::Arc::new(planned.admission.guard.clone()),
        numerical: std::sync::Arc::new(planned.admission.assessment.clone()),
        kernels,
    }
}

impl<T: seismic_target::TargetFamily> CompiledCandidate<T> {
    pub(crate) fn fixed(&self) -> &seismic_lang::expr::PartialAssignment {
        &self.fixed
    }
}

fn compile_allocations(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    topology: &seismic_ir::storage::GlobalAllocationTopology,
) -> (Vec<u32>, Vec<AllocationPlan>) {
    use std::collections::BTreeMap;
    let mut arena_groups: BTreeMap<i64, (usize, Vec<usize>)> = BTreeMap::new();
    let mut dedicated = Vec::new();
    let mut next_universal_slot = -1i64;
    for (index, allocation) in topology.allocations().iter().enumerate() {
        if matches!(allocation.kind, GlobalBufferKind::Imported { .. }) {
            panic!("caller-owned imported storage escaped into an executable root implementation");
        }
        if matches!(allocation.kind, GlobalBufferKind::Arena) {
            let slot = match allocation.slot {
                Some(decision) => match fixed.get(arena.decision_symbol(decision)) {
                    Some(SymbolValue::Int(value)) => i64::try_from(value).expect("closed finite decision exceeds its integer domain"),
                    _ => panic!("arena reuse decision is absent from exact assignment"),
                },
                None => {
                    let value = next_universal_slot;
                    next_universal_slot -= 1;
                    value
                }
            };
            arena_groups
                .entry(slot)
                .and_modify(|(_, rest)| rest.push(index))
                .or_insert((index, Vec::new()));
        } else {
            dedicated.push(index);
        }
    }
    let mut remap = vec![u32::MAX; topology.allocations().len()];
    let mut plans = Vec::new();
    for index in dedicated {
        let allocation = &topology.allocations()[index];
        let kind = match &allocation.kind {
            GlobalBufferKind::Argument {
                abi: Some(parameter),
                ..
            } => ExecutableGlobalAllocationKind::Argument(*parameter),
            GlobalBufferKind::Result { .. } => ExecutableGlobalAllocationKind::Result,
            GlobalBufferKind::Arena => ExecutableGlobalAllocationKind::Arena,
            GlobalBufferKind::Persistent => ExecutableGlobalAllocationKind::Persistent,
            GlobalBufferKind::Argument { abi: None, .. } | GlobalBufferKind::Imported { .. } => {
                panic!("construction-only storage escaped into an executable implementation")
            }
        };
        remap[index] = plans.len() as u32;
        plans.push(AllocationPlan {
            acquisition: AllocationAcquisition::Invocation,
            kind: ExecutableAllocationKind::Global(kind),
            byte_candidates: AllocationByteCandidates::one(
                arena.compile_nat_with(allocation.reserved_bytes, fixed),
            ),
            alignment: allocation.alignment,
        });
    }
    for (first, rest) in arena_groups.into_values() {
        let acquisition = topology.allocations()[first].acquisition;
        assert!(rest.iter().all(|index| topology.allocations()[*index].acquisition == acquisition),
            "physical reuse group mixes allocation timing");
        let physical = plans.len() as u32;
        let alignment = rest.iter().fold(
            topology.allocations()[first].alignment,
            |alignment, index| alignment.max(topology.allocations()[*index].alignment),
        );
        remap[first] = physical;
        let first_bytes = arena.compile_nat_with(topology.allocations()[first].reserved_bytes, fixed);
        let byte_candidates = rest
            .iter()
            .map(|index| {
                remap[*index] = physical;
                arena.compile_nat_with(topology.allocations()[*index].reserved_bytes, fixed)
            })
            .collect::<Vec<_>>();
        plans.push(AllocationPlan {
            acquisition,
            kind: ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Arena),
            byte_candidates: AllocationByteCandidates::with_rest(first_bytes, byte_candidates),
            alignment,
        });
        let instances=topology.allocations()[first].instances;
        assert!(instances==1 || rest.is_empty(),"banked allocations cannot share a physical slot");
        for _ in 1..instances {
            plans.push(AllocationPlan {
            acquisition,
                kind: ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Arena),
                byte_candidates: AllocationByteCandidates::one(arena.compile_nat_with(topology.allocations()[first].reserved_bytes,fixed)),
                alignment,
            });
        }
    }
    assert!(
        remap.iter().all(|value| *value != u32::MAX),
        "every logical allocation must map to one physical allocation"
    );
    (remap, plans)
}

struct LaunchResourceBindings {
    scratch: LaunchScratchBindings,
    abi: KernelAbiBindings,
}

fn compile_launch_resources(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    requirements: &[seismic_ir::execution::LaunchResources],
    allocations: &mut Vec<AllocationPlan>,
) -> Vec<LaunchResourceBindings> {
    requirements
        .iter()
        .enumerate()
        .map(|(launch, resources)| {
            let requirement = resources.scratch();
            let mut add = |class, requirement: &Option<seismic_ir::storage::ScratchRequirement>| {
                requirement.as_ref().map(|requirement| {
                    let allocation = u32::try_from(allocations.len())
                        .expect("executable allocation ordinal space exhausted");
                    allocations.push(AllocationPlan {
                        acquisition: AllocationAcquisition::Invocation,
                        kind: ExecutableAllocationKind::LaunchScratch {
                            launch: u32::try_from(launch).expect("launch ordinal space exhausted"),
                            class,
                        },
                        byte_candidates: AllocationByteCandidates::one(
                            arena.compile_nat_with(requirement.bytes, fixed),
                        ),
                        alignment: requirement.alignment,
                    });
                    ExecutableAllocationId(allocation)
                })
            };
            let scratch = LaunchScratchBindings {
                workgroup: add(
                    seismic_ir::storage::LaunchLocalKind::Workgroup,
                    &requirement.workgroup,
                ),
                participant: add(
                    seismic_ir::storage::LaunchLocalKind::Participant,
                    &requirement.participant,
                ),
                register: add(
                    seismic_ir::storage::LaunchLocalKind::Register,
                    &requirement.register,
                ),
            };
            let requirements = resources.abi();
            let mut bindings = Vec::with_capacity(requirements.len());
            for requirement in requirements {
                let allocation = u32::try_from(allocations.len())
                    .expect("executable allocation ordinal space exhausted");
                allocations.push(AllocationPlan {
                        acquisition: AllocationAcquisition::Invocation,
                    kind: ExecutableAllocationKind::KernelAbi {
                        launch: u32::try_from(launch).expect("launch ordinal space exhausted"),
                        role: requirement.role,
                    },
                    byte_candidates: AllocationByteCandidates::one(
                        arena.compile_nat_with(requirement.bytes, fixed),
                    ),
                    alignment: requirement.alignment,
                });
                bindings.push((requirement.role, ExecutableAllocationId(allocation)));
            }
            let abi = KernelAbiBindings {
                allocations: bindings.into_boxed_slice(),
            };
            LaunchResourceBindings { scratch, abi }
        })
        .collect()
}

fn primary_view(
    topology: &seismic_ir::storage::GlobalAllocationTopology,
    allocation: u32,
) -> seismic_ir::storage::AnyBufferView {
    let id = topology
        .allocation_ids()
        .nth(allocation as usize)
        .unwrap_or_else(|| panic!("binding allocation is outside the closed topology"));
    topology
        .views()
        .iter()
        .enumerate()
        .find_map(|(index, layout)| {
            (layout.base == seismic_ir::storage::ViewBase::Allocation(id)).then(|| topology.view_id(index as u32))
        })
        .unwrap_or_else(|| panic!("ABI allocation has no published buffer view"))
}

fn compile_view(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    topology: &seismic_ir::storage::GlobalAllocationTopology,
    allocation_remap: &[u32],
    view: seismic_ir::storage::AnyBufferView,
) -> CompiledBufferView {
    let layout = topology.view(view);
    CompiledBufferView {
        base: match layout.base {
            seismic_ir::storage::ViewBase::Allocation(id) => CompiledViewBase::Allocation(ExecutableAllocationId(allocation_remap[id.index() as usize])),
            seismic_ir::storage::ViewBase::TensorValue(id) => CompiledViewBase::TensorValue(id.index()),
        },
        representation: layout.representation,
        byte_offset: arena.compile_nat_with(layout.offset, fixed),
        extents: layout
            .extents
            .iter()
            .map(|value| arena.compile_nat_with(*value, fixed))
            .collect(),
        strides: layout
            .strides
            .iter()
            .map(|value| arena.compile_nat_with(*value, fixed))
            .collect(),
    }
}

fn compile_steps<B: seismic_target::TargetFamily>(
    native_set: &crate::realization::RealizedNativeSet<B>,
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    topology: &seismic_ir::storage::GlobalAllocationTopology,
    allocation_remap: &[u32],
    kernels: &seismic_ir::kernel::KernelArena<B>,
    schedule: &seismic_ir::schedule::ParametricSchedule<B>,
    launch_resources: &[seismic_ir::execution::LaunchResources],
    launch_bindings: &[LaunchResourceBindings],
    steps: &[seismic_ir::schedule::ScheduleStep],
) -> Vec<NativeStep<B>> {
    let view = |value| compile_view(arena, fixed, topology, allocation_remap, value);
                    use seismic_ir::region::{ValueOperand, ScalarOperand, QuantityOperand, ValueDestination};
                    let operand = |value| match value {
                        ValueOperand::Scalar(ScalarOperand::Natural(value)) => CompiledRegionOperand::Natural(arena.compile_nat_with(value, fixed)),
                        ValueOperand::Scalar(ScalarOperand::Word { symbol, .. }) => CompiledRegionOperand::Word(symbol),
                        ValueOperand::Quantity(QuantityOperand::Integer(value)) => CompiledRegionOperand::Integer(arena.compile_int_with(value, fixed)),
                        ValueOperand::Quantity(QuantityOperand::Natural(value)) => CompiledRegionOperand::Natural(arena.compile_nat_with(value, fixed)),
                        ValueOperand::Tensor(value) => CompiledRegionOperand::Tensor(view(value)),
                    };
                    let destination = |value| match value {
                        ValueDestination::Scalar(slot) => CompiledRegionDestination::Scalar(slot),
                        ValueDestination::Quantity(slot) => CompiledRegionDestination::Quantity(slot),
                        ValueDestination::Tensor(value) => {
                            let layout = topology.view(value);
                            let seismic_ir::storage::ViewBase::TensorValue(id) = layout.base else { panic!("region tensor destination is not a region view") };
                            let symbols = |values: &[seismic_lang::expr::NatExpr]| values.iter().map(|value| match arena.view((*value).into()) {
                                seismic_lang::expr::NodeView::Symbol(symbol) => symbol,
                                _ => panic!("tensor geometry must be its constructor-owned transport"),
                            }).collect::<Vec<_>>();
                            CompiledRegionDestination::Tensor { id: id.index(), extents: symbols(&layout.extents).into_iter().map(CompiledTensorAxisDestination::Bound).collect(), strides: symbols(&layout.strides) }
                        }
                    };
    let mut out = Vec::new();
    for step in steps {
        let compiled = match step {
            seismic_ir::schedule::ScheduleStep::Imported { body, .. } => {
                out.extend(compile_steps(native_set, arena, fixed, topology, allocation_remap,
                    kernels, schedule, launch_resources, launch_bindings, body));
                continue;
            }
            seismic_ir::schedule::ScheduleStep::BindArgumentTensor { source, result } => {
                let seismic_ir::storage::ViewBase::Allocation(root) = topology.view(*source).base else { panic!("argument tensor has no backing binding") };
                let layout = topology.view(*result);
                let seismic_ir::storage::ViewBase::TensorValue(id) = layout.base else { panic!("argument must define a tensor value") };
                let symbols = |values: &[seismic_lang::expr::NatExpr]| values.iter().map(|value| match arena.view((*value).into()) {
                    seismic_lang::expr::NodeView::Symbol(symbol) => symbol,
                    _ => panic!("argument geometry destination must be constructor-owned"),
                }).collect();
                ExecutableStep::BindArgumentTensor {
                    allocation: ExecutableAllocationId(allocation_remap[root.index() as usize]), representation: layout.representation,
                    result: id.index(), strides: symbols(&layout.strides),
                }
            }
            seismic_ir::schedule::ScheduleStep::BeginAllocationInstance { source, result } => {
                let seismic_ir::storage::ViewBase::Allocation(root) = topology.view(*source).base else { panic!("allocation occurrence has no allocation") };
                assert!(topology.allocation(root).geometry.is_some(), "Begin requires tensor geometry");
                let allocation = ExecutableAllocationId(allocation_remap[root.index() as usize]);
                let banks = (0..topology.allocation(root).instances).map(|offset| ExecutableAllocationId(allocation.0.checked_add(offset).expect("bank ordinal overflow"))).collect();
                let layout = topology.view(*result);
                let seismic_ir::storage::ViewBase::TensorValue(id) = layout.base else { panic!("Begin must define a tensor value") };
                let symbols = |values: &[seismic_lang::expr::NatExpr]| values.iter().map(|value| match arena.view((*value).into()) {
                    seismic_lang::expr::NodeView::Symbol(symbol) => symbol,
                    _ => panic!("tensor geometry destination must be a constructor-owned slot"),
                }).collect();
                ExecutableStep::BeginAllocationInstance {
                    allocation, banks, bytes: arena.compile_nat_with(topology.allocation(root).bytes, fixed),
                    alignment: topology.allocation(root).alignment, geometry: view(*source),
                    result: CompiledRegionDestination::Tensor {
                        id: id.index(), extents: layout.extents.iter().map(|axis| match arena.view((*axis).into()) {
                            seismic_lang::expr::NodeView::Symbol(symbol) if matches!(arena.symbol_kind(symbol), seismic_lang::expr::SymbolKind::ScheduleSlot(_)) => CompiledTensorAxisDestination::Bound(symbol),
                            _ => CompiledTensorAxisDestination::Invariant(arena.compile_nat_with(*axis, fixed)),
                        }).collect(), strides: symbols(&layout.strides),
                    },
                }
            }
            seismic_ir::schedule::ScheduleStep::PublishTensor { view, path, bytes, declared_axes } => ExecutableStep::PublishTensor {
                path: path.clone(),
                view: compile_view(arena, fixed, topology, allocation_remap, *view),
                bytes: arena.compile_nat_with(*bytes, fixed),
                declared_axes: declared_axes.iter().map(|axis| arena.compile_nat_with(*axis, fixed)).collect(),
            },
            seismic_ir::schedule::ScheduleStep::Launch(id) => {
                let launch = schedule.launch(*id);
                let kernel = kernels.kernel(launch.kernel);
                let layout = launch_resources[id.index() as usize].layout();
                if let Some(logical) = launch.logical_base {
                    assert!(
                        (logical.argument() as usize) < kernel.interface().nat_args.len(),
                        "logical launch base names an absent natural kernel argument"
                    );
                }
                ExecutableStep::Command(ExecutableCommand::Launch {
                    kernel: ExecutableKernelId(
                        // Coordinate-exact reconciliation supplies a dense
                        // table in which every launched kernel is realized.
                        native_set.native_kernel_index(launch.kernel).unwrap_or_else(
                            || panic!("launched kernel has no realized native kernel"),
                        ),
                    ),
                    descriptor: launch.descriptor.clone(),
                    grid: launch
                        .grid
                        .map(|value| arena.compile_nat_with(value, fixed)),
                    workgroup: launch
                        .workgroup
                        .map(|value| arena.compile_nat_with(value, fixed)),
                    empty: arena.compile_bool_with(launch.empty, fixed),
                    bindings: kernel
                        .interface()
                        .bindings
                        .iter()
                        .map(|binding| view(binding.view))
                        .collect(),
                    nat_args: kernel
                        .interface()
                        .nat_args
                        .iter()
                        .enumerate()
                        .map(|(argument, value)| {
                            let value = launch
                                .logical_base
                                .filter(|logical| logical.argument() as usize == argument)
                                .map_or(*value, |logical| logical.value());
                            arena.compile_nat_with(value, fixed)
                        })
                        .collect(),
                    scalar_args: kernel
                        .interface()
                        .scalar_args
                        .iter()
                        .map(|(symbol, _)| *symbol)
                        .collect(),
                    result_slots: kernel.interface().result_slots.iter().copied().collect(),
                    locals: layout
                        .locals
                        .iter()
                        .map(|local| CompiledLocalLayout {
                            kind: local.kind,
                            representation: local.representation,
                            byte_offset: arena.compile_nat_with(local.offset, fixed),
                            extents: local
                                .extents
                                .iter()
                                .map(|value| arena.compile_nat_with(*value, fixed))
                                .collect(),
                            strides: local
                                .strides
                                .iter()
                                .map(|value| arena.compile_nat_with(*value, fixed))
                                .collect(),
                            bytes: arena.compile_nat_with(local.bytes, fixed),
                            alignment: local.alignment,
                        })
                        .collect(),
                    addressable_resources: kernel
                        .addressable_resources()
                        .iter()
                        .map(|lease| CompiledAddressableResource {
                            offset_units: arena.compile_nat_with(lease.offset_units, fixed),
                            units: arena.compile_nat_with(lease.units, fixed),
                        })
                        .collect(),
                    local_totals: CompiledLocalClassTotals {
                        workgroup_bytes: arena.compile_nat_with(layout.workgroup_bytes, fixed),
                        participant_bytes: arena.compile_nat_with(layout.participant_bytes, fixed),
                        register_bytes: arena.compile_nat_with(layout.register_bytes, fixed),
                    },
                    scratch: launch_bindings[id.index() as usize].scratch,
                    abi: launch_bindings[id.index() as usize].abi.clone(),
                })
            }
            seismic_ir::schedule::ScheduleStep::Copy(copy) => {
                ExecutableStep::Command(ExecutableCommand::Copy {
                    source: view(copy.source),
                    destination: view(copy.destination),
                    bytes: arena.compile_nat_with(copy.bytes, fixed),
                })
            }
            seismic_ir::schedule::ScheduleStep::Fill(fill) => {
                ExecutableStep::Command(ExecutableCommand::Fill {
                    destination: view(fill.destination),
                    value: fill.value,
                    bytes: arena.compile_nat_with(fill.bytes, fixed),
                })
            }
            seismic_ir::schedule::ScheduleStep::ScalarMove(value) => {
                ExecutableStep::Command(ExecutableCommand::ScalarMove {
                    from: value.from,
                    to: value.to,
                })
            }
            seismic_ir::schedule::ScheduleStep::EvaluateHost(evaluation) => {
                use seismic_ir::schedule::HostValueExpr;
                let value = match evaluation.value {
                    HostValueExpr::Integer(expr) => CompiledHostValue::Integer(arena.compile_int_with(expr, fixed)),
                    HostValueExpr::Natural(expr) => CompiledHostValue::Natural(arena.compile_nat_with(expr, fixed)),
                    HostValueExpr::Bool(expr) => CompiledHostValue::Bool(arena.compile_bool_with(expr, fixed)),
                    HostValueExpr::Word { dtype, value } => CompiledHostValue::Word { dtype, value: arena.compile_int_with(value, fixed) },
                };
                ExecutableStep::EvaluateHost { value, to: evaluation.to, failure: evaluation.failure.clone() }
            }
            seismic_ir::schedule::ScheduleStep::ScalarRead(read) => {
                ExecutableStep::Command(ExecutableCommand::ScalarRead {
                    source: view(read.source),
                    bounds: read
                        .bounds
                        .iter()
                        .map(|bound| arena.compile_bool_with(*bound, fixed))
                        .collect(),
                    byte_offset: arena.compile_nat_with(read.byte_offset, fixed),
                    to: read.to,
                })
            }
            seismic_ir::schedule::ScheduleStep::Check(check) => ExecutableStep::Check {
                condition: check.condition,
                expectation: check.expectation,
                site: check.site.clone(),
            },
            seismic_ir::schedule::ScheduleStep::If {
                condition,
                then_steps,
                else_steps,
                results,
            } => ExecutableStep::If {
                results: results.map(&mut |result| CompiledBranchResult { then_value: operand(result.then_value()), else_value: operand(result.else_value()), result: destination(result.result()) }),
                condition: arena.compile_bool_with(*condition, fixed),
                then_steps: compile_steps(
                    native_set,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_resources,
                    launch_bindings,
                    then_steps,
                )
                .into_boxed_slice(),
                else_steps: compile_steps(
                    native_set,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_resources,
                    launch_bindings,
                    else_steps,
                )
                .into_boxed_slice(),
            },
            seismic_ir::schedule::ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                body,
                carries,
            } => ExecutableStep::Repeat {
                carries: carries.map(&mut |carry| {
                    CompiledRepeatCarry { initial: operand(carry.initial()), header: destination(carry.header()), backedge: operand(carry.backedge()), result: destination(carry.result()) }
                }),
                range: RuntimeRange {
                    start: arena.compile_nat_with(*start, fixed),
                    end: arena.compile_nat_with(*end, fixed),
                },
                binder: LoopBinder {
                    binder: *binder,
                    symbol: *symbol,
                },
                body: compile_steps(
                    native_set,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_resources,
                    launch_bindings,
                    body,
                )
                .into_boxed_slice(),
            },
        };
        out.push(compiled);
    }
    out
}

/// Device services a backend provides to the generic runtime.
pub trait DeviceService<T: seismic_target::TargetFamily>: Send + Sync + 'static {
    type Buffer: Clone + Send + Sync + 'static;
    fn allocate(&self, bytes: u64, alignment: u64) -> Result<Self::Buffer, ExecutionError>;
    fn write(&self, buffer: &Self::Buffer, offset: u64, bytes: &[u8])
        -> Result<(), ExecutionError>;
    fn read(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError>;
    fn buffer_len(&self, buffer: &Self::Buffer) -> u64;
}

/// One invocation-bound physical allocation. `base_offset` is the caller's
/// byte base inside the device buffer; compiled view offsets are relative to
/// it. `accessible_bytes` has already been validated by invocation binding.
#[derive(Clone, Debug)]
pub struct RuntimeTensorGeometry {
    pub representation: seismic_lang::ids::RepresentationId,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
}

pub struct RuntimeBuffer<T> {
    pub tensor: Option<RuntimeTensorGeometry>,
    pub buffer: T,
    pub base_offset: u64,
    pub accessible_bytes: u64,
}

/// Runtime-owned backing for the executable's declared allocation slots.
/// Schedule execution requests an already-planned instance; this interface
/// cannot change the selected storage plan or acquire existing external access.
pub trait ExecutionResources<B> {
    fn buffer(&self, allocation: ExecutableAllocationId) -> &RuntimeBuffer<B>;
    fn acquire_instance(&mut self, allocation: ExecutableAllocationId, bytes: u64, alignment: u64)
        -> Result<(), ExecutionError>;
    /// Retire private backing only after the schedule has completed its native
    /// uses and excluded all retained products. Eager bindings remain owned.
    fn retire_completed_instances(&mut self, allocations: &[ExecutableAllocationId]);
}

pub struct ResolvedBufferView<'a, T> {
    pub buffer: &'a T,
    pub byte_offset: u64,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
    pub byte_span: u64,
}

/// Values available to a command during execution: invocation symbols and
/// the current schedule slot values (bound as symbols).
pub struct ExecutionEnvironment<'a, T: seismic_target::TargetFamily, H, D: DeviceService<T>> {
    device: &'a D,
    native_index_bits: u32,
    /// Allocation index -> buffer.
    resources: &'a mut dyn ExecutionResources<D::Buffer>,
    /// Invocation and slot symbol values; slots are rebound by commands.
    values: &'a mut InvocationValues,
    tensor_values: std::collections::BTreeMap<u32, TensorDescriptor>,
    issued_banks: std::collections::BTreeSet<ExecutableAllocationId>,
    published: std::collections::BTreeMap<Vec<u32>, ExecutedTensorPublication>,
    kernels: &'a [std::sync::Arc<seismic_target::NativeKernel<T, H>>],
}

impl<'a, T: seismic_target::TargetFamily, H, D: DeviceService<T>>
    ExecutionEnvironment<'a, T, H, D>
{
    fn region_operand(&self, operand: &CompiledRegionOperand) -> Result<RegionValue, ExecutionError> {
        Ok(match operand {
            CompiledRegionOperand::Natural(value) => RegionValue::Scalar(SymbolValue::Nat(value.evaluate(self.values).expect("closed region quantity is defined"))),
            CompiledRegionOperand::Integer(value) => RegionValue::Scalar(SymbolValue::Int(value.evaluate(self.values).expect("closed region quantity is defined"))),
            CompiledRegionOperand::Word(symbol) => RegionValue::Scalar(self.symbol(*symbol)),
            CompiledRegionOperand::Tensor(view) => RegionValue::Tensor(self.capture_view(view)?),
        })
    }

    fn bind_region_value(&mut self, destination: &CompiledRegionDestination, value: &RegionValue) {
        match (destination, value) {
            (CompiledRegionDestination::Scalar(slot), RegionValue::Scalar(value)) => self.set_slot(*slot,value.clone()),
            (CompiledRegionDestination::Quantity(slot), RegionValue::Scalar(value)) => {
                assert!(matches!((slot.kind(), value),
                    (seismic_ir::schedule::HostQuantityKind::Integer, SymbolValue::Int(_))
                    | (seismic_ir::schedule::HostQuantityKind::Natural, SymbolValue::Nat(_))), "closed region quantity kind changed");
                self.values.bind(slot.symbol(), value.clone());
            }
            (CompiledRegionDestination::Tensor { id, extents, strides }, RegionValue::Tensor(value)) => {
                assert_eq!(extents.len(),value.extents.len(),"closed tensor rank changed");
                for (destination, extent) in extents.iter().zip(&value.extents) {
                    match destination {
                        CompiledTensorAxisDestination::Bound(symbol) => self.values.bind(*symbol, SymbolValue::Nat(extent.clone())),
                        CompiledTensorAxisDestination::Invariant(expression) => assert_eq!(expression.evaluate(self.values).expect("constructor-owned invariant axis must be defined after acquisition"), *extent, "acquired geometry disagrees with its defining expression"),
                    }
                }
                assert_eq!(strides.len(),value.strides.len(),"closed region tensor rank changed");
                for (symbol,stride) in strides.iter().zip(&value.strides) { self.values.bind(*symbol,SymbolValue::Nat(stride.clone())); }
                self.tensor_values.insert(*id,value.clone());
            }
            _ => panic!("closed region product changed leaf kind"),
        }
    }

    fn bind_region_product(
        &mut self, carries: &seismic_ir::region::Product<CompiledRepeatCarry>,
        values: &seismic_ir::region::Product<RegionValue>, result: bool,
    ) {
        use seismic_ir::region::Product;
        match (carries,values) {
            (Product::Unit,Product::Unit) => (),
            (Product::Leaf(carry),Product::Leaf(value)) => self.bind_region_value(if result { &carry.result } else { &carry.header }, value),
            (Product::Range(a,b),Product::Range(c,d)) => { self.bind_region_product(a,c,result); self.bind_region_product(b,d,result); }
            (Product::Tuple(a),Product::Tuple(b)) => {
                assert_eq!(a.len(),b.len());
                for (carry,value) in a.iter().zip(b) { self.bind_region_product(carry,value,result); }
            }
            _ => panic!("closed region product changed shape"),
        }
    }

    pub fn set_slot(&mut self, slot: AnyScalarSlot, value: SymbolValue) {
        self.values.bind(slot.symbol(), value);
    }

    /// Returns the native handle selected for one closed executable launch.
    ///
    /// Only a `NativeSubmission` receives an execution environment. Runtime,
    /// planning, and evaluation cannot construct one or enumerate its handles.
    pub fn kernel_handle(&self, id: ExecutableKernelId) -> &H {
        self.kernels
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("closed executable kernel handle is not bound"))
            .handle()
    }

    pub fn values(&self) -> &InvocationValues {
        self.values
    }

    pub fn buffer(&self, id: ExecutableAllocationId) -> &RuntimeBuffer<D::Buffer> {
        self.resources.buffer(id)
    }

    /// Evaluates a guarded Nat expression. Frozen-plan side conditions and
    /// the executable guard make failure a private construction contradiction, not
    /// a backend/runtime error channel.
    pub fn nat(&self, value: &CompiledNat) -> u64 {
        value
            .evaluate_u64(self.values)
            .unwrap_or_else(|error| panic!("guarded executable Nat failed: {error:?}"))
    }

    pub fn predicate(&self, value: &CompiledPredicate) -> bool {
        value
            .evaluate(self.values)
            .unwrap_or_else(|error| panic!("guarded executable predicate failed: {error:?}"))
    }

    pub fn symbol(&self, symbol: seismic_lang::expr::SymbolId) -> SymbolValue {
        self.values
            .get(symbol)
            .unwrap_or_else(|| panic!("closed executable reads an unbound scalar symbol"))
    }

    fn physical_natural(&self, value: &CompiledNat, purpose: &str) -> Result<u64, ExecutionError> {
        let exact = value.evaluate(self.values).map_err(|error| eval_failure(purpose, error))?;
        self.physical_value(exact)
    }

    fn physical_value(&self, exact: seismic_lang::expr::BigUint) -> Result<u64, ExecutionError> {
        let available = if self.native_index_bits >= 64 { u64::MAX }
            else { (1u64 << self.native_index_bits) - 1 };
        let finite = u64::try_from(&exact).ok().filter(|value| *value <= available)
            .ok_or_else(|| ExecutionError::AllocationCapacity { required: exact, available })?;
        Ok(finite)
    }

    /// Capture a complete logical value without narrowing metadata to a native ABI.
    fn capture_view(&self, view: &CompiledBufferView) -> Result<TensorDescriptor, ExecutionError> {
        let contradiction = |detail: &str| ExecutionError::ConstructionContradiction(detail.into());
        let (allocation, offset) = match view.base {
            CompiledViewBase::Allocation(id) => (id, seismic_lang::expr::BigUint::default()),
            CompiledViewBase::TensorValue(id) => {
                let descriptor = self.tensor_values.get(&id)
                    .ok_or_else(|| contradiction("tensor used outside its product scope"))?;
                (descriptor.allocation, descriptor.byte_offset.clone())
            }
        };
        let evaluate = |value: &CompiledNat| value.evaluate(self.values)
            .map_err(|error| eval_failure("tensor geometry capture", error));
        Ok(TensorDescriptor {
            allocation, representation: view.representation,
            byte_offset: offset + evaluate(&view.byte_offset)?,
            extents: view.extents.iter().map(evaluate).collect::<Result<_, _>>()?,
            strides: view.strides.iter().map(evaluate).collect::<Result<_, _>>()?,
        })
    }

    /// Project the complete value only at a native access.
    pub fn resolve_view(&self, view: &CompiledBufferView) -> Result<ResolvedBufferView<'_, D::Buffer>, ExecutionError> {
        let contradiction = |detail: &str| ExecutionError::ConstructionContradiction(detail.into());
        let descriptor = self.capture_view(view)?;
        let allocation = self.resources.buffer(descriptor.allocation);
        let relative = self.physical_value(descriptor.byte_offset)?;
        let extents = descriptor.extents.into_iter().map(|value| self.physical_value(value)).collect::<Result<Vec<_>, _>>()?;
        let strides = descriptor.strides.into_iter().map(|value| self.physical_value(value)).collect::<Result<Vec<_>, _>>()?;
        if extents.len() != strides.len() { return Err(contradiction("view rank differs from stride count")); }
        let mut addressed_extents = extents.clone();
        let unit_bytes = match &seismic_lang::registry::representation_info(view.representation).kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => u64::from(dtype.bytes()),
            seismic_lang::registry::RepresentationKind::Packed(layout) => {
                if let Some(last) = addressed_extents.last_mut() { *last = last.div_ceil(u64::from(layout.group)); }
                u64::from(layout.packet_size)
            }
            seismic_lang::registry::RepresentationKind::External(layout) => {
                if let Some(last) = addressed_extents.last_mut() { *last = last.div_ceil(u64::from(layout.logical_group)); }
                u64::from(layout.packet_size)
            }
        };
        let exact_span = if addressed_extents.contains(&0) { seismic_lang::expr::BigUint::default() }
            else {
                let units = addressed_extents.iter().zip(&strides).fold(seismic_lang::expr::BigUint::from(1u8),
                    |sum, (extent, stride)| sum + seismic_lang::expr::BigUint::from(extent - 1) * *stride);
                units * unit_bytes
            };
        let byte_span = u64::try_from(&exact_span).map_err(|_| ExecutionError::AllocationCapacity {
            required: exact_span, available: allocation.accessible_bytes,
        })?;
        if !relative.checked_add(byte_span).is_some_and(|end| end <= allocation.accessible_bytes) {
            return Err(contradiction("proved view exceeds its actual invocation backing"));
        }
        if !allocation.base_offset.checked_add(allocation.accessible_bytes)
            .is_some_and(|end| end <= self.device.buffer_len(&allocation.buffer)) {
            return Err(contradiction("invocation backing exceeds physical buffer"));
        }
        let byte_offset = allocation.base_offset.checked_add(relative)
            .ok_or_else(|| contradiction("physical view base overflow"))?;
        let alignment = seismic_ir::storage::representation_alignment(view.representation);
        if byte_offset % alignment != 0 { return Err(contradiction("proved view alignment differs from actual backing")); }
        Ok(ResolvedBufferView { buffer: &allocation.buffer, byte_offset, extents, strides, byte_span })
    }

    fn retain_command_backings(&mut self, command: &ExecutableCommand<T>) -> Result<(), ExecutionError> {
        let mut views = Vec::new();
        match command {
            ExecutableCommand::Launch { empty, bindings, .. } => {
                if empty.evaluate(self.values).map_err(|error| eval_failure("launch emptiness", error))? { return Ok(()); }
                views.extend(bindings.iter());
            }
            ExecutableCommand::Copy { source, destination, .. } => { views.push(source); views.push(destination); }
            ExecutableCommand::Fill { destination, .. } => views.push(destination),
            ExecutableCommand::ScalarRead { source, .. } => views.push(source),
            _ => {},
        }
        for view in views { self.issued_banks.insert(self.capture_view(view)?.allocation); }
        Ok(())
    }

    fn retire_tensor_definitions(&mut self, steps: &[NativeStep<T>]) {
        for step in steps {
            match step {
                ExecutableStep::BeginAllocationInstance { result: CompiledRegionDestination::Tensor { id, .. }, .. } => { self.tensor_values.remove(id); }
                ExecutableStep::If { then_steps, else_steps, results, .. } => {
                    self.retire_tensor_definitions(then_steps); self.retire_tensor_definitions(else_steps);
                    results.visit(&mut |result| if let CompiledRegionDestination::Tensor { id, .. } = result.result { self.tensor_values.remove(&id); });
                }
                ExecutableStep::Repeat { body, carries, .. } => {
                    self.retire_tensor_definitions(body);
                    carries.visit(&mut |carry| for destination in [&carry.header, &carry.result] {
                        if let CompiledRegionDestination::Tensor { id, .. } = destination { self.tensor_values.remove(id); }
                    });
                }
                _ => {},
            }
        }
    }

    fn validate_transfer(&self, view: &CompiledBufferView, bytes: &CompiledNat) -> Result<(), ExecutionError> {
        let resolved = self.resolve_view(view)?;
        let bytes = self.physical_natural(bytes, "reached transfer bytes")?;
        if bytes > resolved.byte_span {
            return Err(ExecutionError::ConstructionContradiction("proved transfer exceeds its reached view".into()));
        }
        Ok(())
    }

}

/// Concurrency-safe factory for invocation-owned native submissions. The
/// executor itself is never mutably borrowed across execution or completion;
/// every admitted run owns a distinct submission value.
pub trait NativeExecutor<T: seismic_target::TargetFamily>: Send + Sync + 'static {
    /// The execution-side type of an already formed native artifact. Formation
    /// remains owned by the independently injected `NativeCompiler` service.
    type Handle: Send + Sync + 'static;
    type Device: DeviceService<T>;
    type Submission: NativeSubmission<T, Handle = Self::Handle, Device = Self::Device>;

    fn begin_submission(&self) -> Result<Self::Submission, ExecutionError>;
}

/// Backend submission state owned by exactly one admitted run. Its infallible
/// transfer into an execution owner preserves the lifetime boundary after
/// partially issued work.
pub trait NativeSubmission<T: seismic_target::TargetFamily>: Send + 'static {
    type Handle: Send + Sync + 'static;
    type Device: DeviceService<T>;
    type Execution: NativeExecution;
    fn execute(
        &mut self,
        command: &ExecutableCommand<T>,
        env: &mut ExecutionEnvironment<'_, T, Self::Handle, Self::Device>,
    ) -> Result<(), ExecutionError>;
    /// Reach the terminal boundary of all work issued so far without ending
    /// this submission. Only `Ok` permits backing reuse. On error the driver
    /// stops issuing and retains every admission lease through final completion.
    fn complete_prefix(&mut self) -> Result<(), ExecutionError>;
    /// Transfers every issued operation into one execution owner.
    ///
    /// This transition is infallible so an admitted run can never lose the
    /// only handle capable of reaching the backend's terminal boundary. Any
    /// failure while issuing work is returned by `execute`; any failure while
    /// reaching the terminal boundary is returned by `NativeExecution::complete`.
    fn submit(self) -> Self::Execution;
}

/// Submitted backend work. Completion borrows the handle so the runtime can
/// retain or deliberately leak both it and the admission-owned resources if a
/// backend bug panics before proving the terminal boundary. Returning `Err`
/// still means that all work is terminal; the error reports how completion
/// failed, not that work remains in flight.
pub trait NativeExecution: Send + 'static {
    fn complete(&mut self) -> Result<(), ExecutionError>;
}

/// Executes one already-selected variant. This is the sole entry that binds
/// opaque native handles and runtime storage to the compiler-owned schedule.
pub fn execute_variant<T, S>(
    variant: &ExecutableVariant<T, S::Handle>,
    submission: &mut S,
    device: &S::Device,
    resources: &mut dyn ExecutionResources<<S::Device as DeviceService<T>>::Buffer>,
    values: &mut InvocationValues,
) -> Result<Vec<ExecutedTensorPublication>, ExecutionError>
where
    T: seismic_target::TargetFamily,
    S: NativeSubmission<T>,
{
    let mut environment = ExecutionEnvironment {
        native_index_bits: variant.body.native_index_bits,
        device,
        resources,
        values,
        tensor_values: std::collections::BTreeMap::new(),
        issued_banks: std::collections::BTreeSet::new(),
        published: std::collections::BTreeMap::new(),
        kernels: &variant.kernels,
    };
    execute_schedule(&variant.body.schedule, submission, &mut environment)?;
    for (path, binding) in &variant.body.bindings.results {
        if let ExecutableResultBinding::Buffer { representation, rank } = binding {
            let publication = environment.published.get(path).ok_or_else(|| ExecutionError::ConstructionContradiction(
                format!("successful execution did not publish tensor result {path:?}")))?;
            if publication.representation != *representation || publication.extents.len() != *rank {
                return Err(ExecutionError::ConstructionContradiction("published result violates its declared tensor type".into()));
            }
        }
    }
    Ok(environment.published.into_values().collect())
}

fn reached_allocation_bytes(
    allocation: ExecutableAllocationId,
    bytes: &CompiledNat,
    values: &InvocationValues,
) -> Result<u64, ExecutionError> {
    let required = bytes.evaluate(values).map_err(|error| ExecutionError::ConstructionContradiction(
        format!("reached allocation {} size is undefined: {error:?}", allocation.ordinal())))?;
    u64::try_from(&required).map_err(|_| ExecutionError::AllocationCapacity { required, available: u64::MAX })
}

fn available_instance_banks(
    banks: &[ExecutableAllocationId],
    retained: &std::collections::BTreeMap<u32, TensorDescriptor>,
) -> Vec<ExecutableAllocationId> {
    banks.iter().copied().filter(|bank| !retained.values().any(|value| value.allocation == *bank)).collect()
}

/// A resource retry does not repeat source execution. The current Begin
/// supersedes the old instance of this allocation; only its other unretained
/// banks may be reclaimed, and only after the whole issued prefix is terminal.
fn acquire_with_completed_reclamation<B>(
    resources: &mut dyn ExecutionResources<B>,
    bank: ExecutableAllocationId,
    bytes: u64,
    alignment: u64,
    available: &[ExecutableAllocationId],
    complete: impl FnOnce() -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    match resources.acquire_instance(bank, bytes, alignment) {
        Err(ExecutionError::AllocationCapacity { .. }) if available.iter().any(|other| *other != bank) => {
            complete()?;
            let retired = available.iter().copied().filter(|other| *other != bank).collect::<Vec<_>>();
            resources.retire_completed_instances(&retired);
            resources.acquire_instance(bank, bytes, alignment)
        }
        result => result,
    }
}

/// Internal structured-schedule driver. It evaluates predicates and ranges,
/// binds loop values, and dispatches only already-closed native commands.
fn execute_schedule<T, S>(
    steps: &[NativeStep<T>],
    submission: &mut S,
    env: &mut ExecutionEnvironment<'_, T, S::Handle, S::Device>,
) -> Result<(), ExecutionError>
where
    T: seismic_target::TargetFamily,
    S: NativeSubmission<T>,
{
    for step in steps {
        match step {
            ExecutableStep::BindArgumentTensor { allocation, representation, result, strides } => {
                let geometry = env.resources.buffer(*allocation).tensor.as_ref().ok_or_else(|| ExecutionError::ConstructionContradiction("argument tensor lost its bound geometry".into()))?;
                if geometry.representation != *representation { return Err(ExecutionError::ConstructionContradiction("argument representation changed after binding".into())); }
                let value = TensorDescriptor {
                    allocation: *allocation, representation: *representation, byte_offset: 0u64.into(),
                    extents: geometry.extents.iter().map(|value| (*value).into()).collect(),
                    strides: geometry.strides.iter().map(|value| (*value).into()).collect(),
                };
                assert_eq!(strides.len(), value.strides.len(), "bound argument rank changed");
                for (symbol, stride) in strides.iter().zip(&value.strides) {
                    env.values.bind(*symbol, SymbolValue::Nat(stride.clone()));
                }
                env.tensor_values.insert(*result, value);
            }
            ExecutableStep::BeginAllocationInstance { allocation, banks, bytes, alignment, geometry, result } => {
                let available = available_instance_banks(banks, &env.tensor_values).into_iter()
                    .filter(|bank| !env.published.values().any(|value| value.allocation == *bank)).collect::<Vec<_>>();
                let bank = available.iter().copied().find(|bank| !env.issued_banks.contains(bank)).or_else(|| available.first().copied()).expect("closed backing pool cannot satisfy its retained product bound");
                if env.issued_banks.contains(&bank) {
                    submission.complete_prefix()?;
                    env.issued_banks.clear();
                }
                let bytes = reached_allocation_bytes(*allocation, bytes, env.values)?;
                acquire_with_completed_reclamation(env.resources, bank, bytes, *alignment, &available, || {
                    submission.complete_prefix()?;
                    env.issued_banks.clear();
                    Ok(())
                })?;
                let mut descriptor = env.capture_view(geometry)?;
                descriptor.allocation = bank;
                env.bind_region_value(result, &RegionValue::Tensor(descriptor));
            }
            ExecutableStep::PublishTensor { path, view, bytes, declared_axes } => {
                if env.published.contains_key(path) {
                    return Err(ExecutionError::ConstructionContradiction(format!("result {path:?} was published twice on one execution path")));
                }
                let allocation = match view.base {
                    CompiledViewBase::Allocation(id) => id,
                    CompiledViewBase::TensorValue(id) => env.tensor_values.get(&id).expect("unbound published region tensor").allocation,
                };
                let bytes = reached_allocation_bytes(allocation, bytes, env.values)?;
                let resolved = env.resolve_view(view)?;
                let declared = declared_axes.iter().map(|axis| axis.evaluate(env.values)
                    .map_err(|error| eval_failure("declared result axis", error)))
                    .collect::<Result<Vec<_>, _>>()?;
                if resolved.extents.len() != declared.len() || declared.iter().zip(&resolved.extents).any(|(expected, actual)| expected != &(*actual).into()) {
                    return Err(ExecutionError::ConstructionContradiction(format!("published result {path:?} differs from its declared shape")));
                }
                let publication = ExecutedTensorPublication {
                    path: path.clone(), allocation, representation: view.representation,
                    byte_offset: resolved.byte_offset, bytes, extents: resolved.extents, strides: resolved.strides,
                };
                env.published.insert(path.clone(), publication);
            }
            ExecutableStep::EvaluateHost { value, to, failure } => {
                use seismic_ir::schedule::HostValueDestination;
                let evaluated = match value {
                    CompiledHostValue::Integer(expr) => expr.evaluate(env.values).map(SymbolValue::Int),
                    CompiledHostValue::Natural(expr) => expr.evaluate(env.values).map(SymbolValue::Nat),
                    CompiledHostValue::Bool(expr) => expr.evaluate(env.values).map(SymbolValue::Bool),
                    CompiledHostValue::Word { dtype: seismic_lang::types::DType::I32, value } => value.evaluate(env.values).and_then(|word| i32::try_from(word).map(SymbolValue::I32).map_err(|_| seismic_lang::expr::EvalError::Unrepresentable)),
                    CompiledHostValue::Word { dtype: seismic_lang::types::DType::U32, value } => value.evaluate(env.values).and_then(|word| u32::try_from(word).map(SymbolValue::U32).map_err(|_| seismic_lang::expr::EvalError::Unrepresentable)),
                    CompiledHostValue::Word { .. } => panic!("host word operation has no integer scalar dtype"),
                };
                let evaluated = evaluated.map_err(|error| {
                    if matches!(error, seismic_lang::expr::EvalError::DivisionByZero | seismic_lang::expr::EvalError::ScalarFailure(seismic_lang::reference_math::ScalarFailure::IntegerDivisionByZero)) {
                        if let Some(site) = failure {
                            return ExecutionError::DataCheckFailed(crate::errors::CheckFailure {
                                failure: site.failure.clone(), path: site.path.clone(), line: site.line,
                            });
                        }
                    }
                    eval_failure("reached host value", error)
                })?;
                match to {
                    HostValueDestination::Quantity(slot) => env.values.bind(slot.symbol(), evaluated),
                    HostValueDestination::Native(slot) => env.set_slot(*slot, evaluated),
                }
            }
            ExecutableStep::Check {
                condition,
                expectation,
                site,
            } => {
                let value = env.values.get(condition.symbol()).unwrap_or_else(|| {
                    panic!("prepared schedule check reads an unbound condition slot")
                });
                let passed = match (expectation, value) {
                    (
                        seismic_ir::schedule::ScalarCheckExpectation::BoolTrue,
                        SymbolValue::Bool(value),
                    ) => value,
                    (
                        seismic_ir::schedule::ScalarCheckExpectation::U32Zero,
                        SymbolValue::U32(value),
                    ) => value == 0,
                    _ => {
                        panic!("prepared schedule check value differs from its closed expectation")
                    }
                };
                if !passed {
                    return Err(ExecutionError::DataCheckFailed(
                        crate::errors::CheckFailure {
                            failure: site.failure.clone(),
                            path: site.path.clone(),
                            line: site.line,
                        },
                    ));
                }
            }
            ExecutableStep::Command(command @ ExecutableCommand::ScalarRead { bounds, source, byte_offset, to }) => {
                let in_bounds = bounds
                    .iter()
                    .try_fold(true, |all, bound| {
                        bound.evaluate(env.values).map(|value| all && value)
                    })
                    .map_err(|error| eval_failure("scalar-read bounds", error))?;
                if !in_bounds {
                    return Err(ExecutionError::ConstructionContradiction("scalar read exceeds its established source bounds".into()));
                }
                let resolved = env.resolve_view(source)?;
                let offset = env.physical_natural(byte_offset, "reached scalar-read offset")?;
                if !offset.checked_add(u64::from(to.kind().bytes())).is_some_and(|end| end <= resolved.byte_span) {
                    return Err(ExecutionError::ConstructionContradiction("scalar read exceeds reached descriptor".into()));
                }
                env.retain_command_backings(command)?;
                submission.execute(command, env)?;
            }
            ExecutableStep::Command(command) => {
                match command {
                    ExecutableCommand::Launch { empty, grid, workgroup, nat_args, bindings, locals, addressable_resources, local_totals, .. } => {
                        let empty = empty.evaluate(env.values).map_err(|error| eval_failure("reached launch emptiness", error))?;
                        if !empty {
                            for value in grid.iter().chain(workgroup).chain(nat_args) {
                                env.physical_natural(value, "reached native natural argument")?;
                            }
                            for binding in bindings { env.resolve_view(binding)?; }
                            for local in locals {
                                for value in std::iter::once(&local.byte_offset).chain(&local.extents)
                                    .chain(&local.strides).chain(std::iter::once(&local.bytes)) {
                                    env.physical_natural(value, "reached local geometry")?;
                                }
                            }
                            for value in [&local_totals.workgroup_bytes, &local_totals.participant_bytes, &local_totals.register_bytes] {
                                env.physical_natural(value, "reached local bytes")?;
                            }
                            for resource in addressable_resources {
                                env.physical_natural(&resource.offset_units, "reached addressable offset")?;
                                env.physical_natural(&resource.units, "reached addressable size")?;
                            }
                        }
                    }
                    ExecutableCommand::Copy { source, destination, bytes } => {
                        env.validate_transfer(source, bytes)?; env.validate_transfer(destination, bytes)?;
                    }
                    ExecutableCommand::Fill { destination, bytes, .. } => env.validate_transfer(destination, bytes)?,
                    _ => {}
                }
                env.retain_command_backings(command)?;
                submission.execute(command, env)?;
            }
            ExecutableStep::If {
                condition,
                then_steps,
                else_steps,
                results,
            } => {
                let taken = condition
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("branch condition", error))?;
                let branch = if taken { then_steps } else { else_steps };
                execute_schedule(branch, submission, env)?;
                let values = results.try_map(&mut |result| env.region_operand(if taken { &result.then_value } else { &result.else_value }))?;
                // Capture the entire selected product before retiring its scope.
                // Pending native operations retain their separate backing pins.
                // Source If products own their escaping values. Selected-operation
                // construction still exports bindings directly until its result
                // transport is closed; keep those definitions alive meanwhile.
                if !matches!(results, seismic_ir::region::Product::Unit) {
                    env.retire_tensor_definitions(branch);
                }
                fn install<B: seismic_target::TargetFamily, H, D: DeviceService<B>>(env: &mut ExecutionEnvironment<'_,B,H,D>, results: &seismic_ir::region::Product<CompiledBranchResult>, values: &seismic_ir::region::Product<RegionValue>) {
                    use seismic_ir::region::Product;
                    match (results, values) {
                        (Product::Unit,Product::Unit)=>(),
                        (Product::Leaf(result),Product::Leaf(value))=>env.bind_region_value(&result.result,value),
                        (Product::Range(a,b),Product::Range(c,d))=>{install(env,a,c);install(env,b,d);},
                        (Product::Tuple(a),Product::Tuple(b))=>{assert_eq!(a.len(),b.len());for(a,b)in a.iter().zip(b){install(env,a,b);}},
                        _=>unreachable!("closed branch product shape"),
                    }
                }
                install(env, results, &values);
            }
            ExecutableStep::Repeat {
                range,
                binder,
                body,
                carries,
            } => {
                let start = range
                    .start
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("repeat start", error))?;
                let end = range
                    .end
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("repeat end", error))?;
                let initial = carries.try_map(&mut |carry| env.region_operand(&carry.initial))?;
                env.bind_region_product(carries, &initial, false);
                let mut current = initial;
                let mut i = start;
                while i < end {
                    env.values.bind(binder.symbol, SymbolValue::Nat(i.clone()));
                    execute_schedule(body, submission, env)?;
                    current = carries.try_map(&mut |carry| env.region_operand(&carry.backedge))?;
                    env.retire_tensor_definitions(body);
                    env.bind_region_product(carries, &current, false);
                    i += 1u8;
                }
                env.bind_region_product(carries, &current, true);
                carries.visit(&mut |carry| if let CompiledRegionDestination::Tensor { id, .. } = &carry.header { env.tensor_values.remove(id); });
            }
        }
    }
    Ok(())
}

/// An evaluation failure at runtime can only be a domain violation of an
/// invocation that passed validation, which the private `PreparedKernel`
/// constructor excludes: it is a panic (§13.3.6).
fn eval_failure(context: &str, error: seismic_lang::expr::EvalError) -> ExecutionError {
    panic!(
        "PreparedKernel coverage invariant violated: guard-admitted invocation failed {context} evaluation: {error:?}"
    )
}

#[allow(dead_code)]
fn _assert_compiled_send<T>()
where
    Compiled<T>: Send + Sync,
{
}

#[cfg(test)]
mod allocation_size_tests {
    use super::*;
    use crate::realization::demand_driven_tests::FakeTarget;
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

    struct NoCommands;
    struct NoDevice;
    impl DeviceService<FakeTarget> for NoDevice {
        type Buffer = ();
        fn allocate(&self, _: u64, _: u64) -> Result<(), ExecutionError> { unreachable!() }
        fn write(&self, _: &(), _: u64, _: &[u8]) -> Result<(), ExecutionError> { unreachable!() }
        fn read(&self, _: &(), _: u64, _: &mut [u8]) -> Result<(), ExecutionError> { unreachable!() }
        fn buffer_len(&self, _: &()) -> u64 { u64::MAX }
    }
    struct NoExecution;
    impl NativeExecution for NoExecution {
        fn complete(&mut self) -> Result<(), ExecutionError> { Ok(()) }
    }
    impl NativeSubmission<FakeTarget> for NoCommands {
        type Handle = ();
        type Device = NoDevice;
        type Execution = NoExecution;
        fn execute(&mut self, _: &ExecutableCommand<FakeTarget>, _: &mut ExecutionEnvironment<'_, FakeTarget, (), NoDevice>) -> Result<(), ExecutionError> { unreachable!() }
        fn complete_prefix(&mut self) -> Result<(), ExecutionError> { Ok(()) }
        fn submit(self) -> NoExecution { NoExecution }
    }

    struct LimitedResources {
        bytes: [u64; 3],
        limit: u64,
        completed: Arc<AtomicBool>,
        attempts: usize,
        retired: Vec<ExecutableAllocationId>,
    }

    impl ExecutionResources<()> for LimitedResources {
        fn buffer(&self, _: ExecutableAllocationId) -> &RuntimeBuffer<()> { unreachable!() }
        fn acquire_instance(&mut self, id: ExecutableAllocationId, bytes: u64, _: u64) -> Result<(), ExecutionError> {
            self.attempts += 1;
            let other = self.bytes.iter().sum::<u64>() - self.bytes[id.ordinal()];
            let available = self.limit.saturating_sub(other);
            if bytes > available { return Err(ExecutionError::AllocationCapacity { required: bytes.into(), available }); }
            self.bytes[id.ordinal()] = bytes;
            Ok(())
        }
        fn retire_completed_instances(&mut self, allocations: &[ExecutableAllocationId]) {
            assert!(self.completed.load(Ordering::SeqCst), "native references must complete before retirement");
            for allocation in allocations {
                self.retired.push(*allocation);
                self.bytes[allocation.ordinal()] = 0;
            }
        }
    }

    fn limited_resources() -> LimitedResources {
        LimitedResources { bytes: [0, 16, 8], limit: 24, completed: Arc::new(AtomicBool::new(false)), attempts: 0, retired: vec![] }
    }

    fn available_with_retained_product() -> Vec<ExecutableAllocationId> {
        let banks = [ExecutableAllocationId(0), ExecutableAllocationId(1), ExecutableAllocationId(2)];
        let retained = std::collections::BTreeMap::from([(7, TensorDescriptor {
            allocation: banks[2], representation: seismic_lang::registry::dense(seismic_lang::types::DType::I32), byte_offset: 0u64.into(), extents: vec![2u64.into()], strides: vec![4u64.into()],
        })]);
        available_instance_banks(&banks, &retained)
    }

    #[test]
    fn repeat_transports_actual_axes_through_header_backedge_and_zero_trip_result() {
        use seismic_ir::construction::Construction;
        use seismic_ir::schedule::{HostQuantityKind, HostValueDestination};
        for trips in [0u64, 3] {
            let mut arena = seismic_lang::expr::ExprArena::default();
            let mut construction = Construction::<FakeTarget>::new(&mut arena, vec![], false, 0);
            let mut slot = || construction.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Natural);
            let header_axis = slot(); let header_stride = slot();
            let result_axis = slot(); let result_stride = slot(); let observed = slot();
            let (binder, symbol, index) = arena.nat_loop_binder();
            let zero = arena.nat(0); let one = arena.nat(1); let three = arena.nat(3);
            let initial_axis = arena.nat_exact(seismic_lang::expr::BigUint::from(1u8) << 80usize);
            let next_axis = arena.nat_add(index, three);
            let end = arena.nat(trips);
            let header_value = arena.nat_symbol(header_axis.symbol());
            let representation = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            let view = |axis| CompiledBufferView {
                base: CompiledViewBase::TensorValue(0), representation,
                byte_offset: arena.compile_nat(zero), extents: vec![arena.compile_nat(axis)],
                strides: vec![arena.compile_nat(one)],
            };
            let steps = [ExecutableStep::Repeat {
                range: RuntimeRange { start: arena.compile_nat(zero), end: arena.compile_nat(end) },
                binder: LoopBinder { binder, symbol },
                body: vec![ExecutableStep::EvaluateHost {
                    value: CompiledHostValue::Natural(arena.compile_nat(header_value)),
                    to: HostValueDestination::Quantity(observed), failure: None,
                }].into_boxed_slice(),
                carries: seismic_ir::region::Product::Leaf(CompiledRepeatCarry {
                    initial: CompiledRegionOperand::Tensor(view(initial_axis)),
                    header: CompiledRegionDestination::Tensor { id: 1, extents: vec![CompiledTensorAxisDestination::Bound(header_axis.symbol())], strides: vec![header_stride.symbol()] },
                    backedge: CompiledRegionOperand::Tensor(view(next_axis)),
                    result: CompiledRegionDestination::Tensor { id: 2, extents: vec![CompiledTensorAxisDestination::Bound(result_axis.symbol())], strides: vec![result_stride.symbol()] },
                }),
            }];
            let mut resources = limited_resources(); let mut values = InvocationValues::new();
            let initial = TensorDescriptor { allocation: ExecutableAllocationId(2), representation,
                byte_offset: 0u64.into(), extents: vec![seismic_lang::expr::BigUint::from(1u8) << 80usize], strides: vec![1u64.into()] };
            let mut env = ExecutionEnvironment {
                native_index_bits: 64, device: &NoDevice, resources: &mut resources, values: &mut values,
                tensor_values: std::collections::BTreeMap::from([(0, initial.clone())]),
                issued_banks: Default::default(), published: Default::default(), kernels: &[],
            };
            execute_schedule::<FakeTarget, NoCommands>(&steps, &mut NoCommands, &mut env).unwrap();
            let expected = if trips == 0 { initial.extents[0].clone() } else { 5u64.into() };
            assert_eq!(env.tensor_values[&2].extents, [expected.clone()]);
            assert_eq!(env.tensor_values[&2].allocation, initial.allocation);
            assert_eq!(env.values.get(result_axis.symbol()), Some(SymbolValue::Nat(expected)));
            if trips == 0 { assert!(env.values.get(observed.symbol()).is_none()); }
            else { assert_eq!(env.values.get(observed.symbol()), Some(SymbolValue::Nat(4u64.into()))); }
        }
    }

    #[test]
    fn branch_captures_selected_geometry_before_retiring_arm_instances() {
        struct Resources { buffer: RuntimeBuffer<()>, acquired: Vec<ExecutableAllocationId> }
        impl ExecutionResources<()> for Resources {
            fn buffer(&self, _: ExecutableAllocationId) -> &RuntimeBuffer<()> { &self.buffer }
            fn acquire_instance(&mut self, id: ExecutableAllocationId, _: u64, _: u64) -> Result<(), ExecutionError> { self.acquired.push(id); Ok(()) }
            fn retire_completed_instances(&mut self, _: &[ExecutableAllocationId]) {}
        }
        for taken in [false,true] {
            let mut arena=seismic_lang::expr::ExprArena::default();
            let zero=arena.nat(0); let one=arena.nat(1); let two=arena.nat(2); let five=arena.nat(5);
            let eight=arena.nat(8); let twenty=arena.nat(20); let condition=arena.bool(taken);
            let symbols=(0..6).map(|i|arena.schedule_slot(i,seismic_lang::expr::SymbolSort::Nat)).collect::<Vec<_>>();
            let representation=seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            let view=|base,axis|CompiledBufferView {base,representation,byte_offset:arena.compile_nat(zero),extents:vec![arena.compile_nat(axis)],strides:vec![arena.compile_nat(one)]};
            let destination=|id,index:usize|CompiledRegionDestination::Tensor {id,extents:vec![CompiledTensorAxisDestination::Bound(symbols[index])],strides:vec![symbols[index+1]]};
            let begin=|bank,id,axis,bytes,index|ExecutableStep::BeginAllocationInstance {
                allocation:ExecutableAllocationId(bank),banks:vec![ExecutableAllocationId(bank)],bytes:arena.compile_nat(bytes),alignment:4,
                geometry:view(CompiledViewBase::Allocation(ExecutableAllocationId(bank)),axis),result:destination(id,index),
            };
            let branch=ExecutableStep::If {
                condition:arena.compile_bool(condition),then_steps:vec![begin(0,1,two,eight,0)].into_boxed_slice(),else_steps:vec![begin(1,2,five,twenty,2)].into_boxed_slice(),
                results:seismic_ir::region::Product::Leaf(CompiledBranchResult {
                    then_value:CompiledRegionOperand::Tensor(view(CompiledViewBase::TensorValue(1),two)),
                    else_value:CompiledRegionOperand::Tensor(view(CompiledViewBase::TensorValue(2),five)), result:destination(3,4),
                }),
            };
            let mut resources=Resources {buffer:RuntimeBuffer {tensor:None,buffer:(),base_offset:0,accessible_bytes:64},acquired:Vec::new()};
            let mut values=InvocationValues::new();
            {
                let mut env=ExecutionEnvironment {native_index_bits:64,device:&NoDevice,resources:&mut resources,values:&mut values,tensor_values:Default::default(),issued_banks:Default::default(),published:Default::default(),kernels:&[]};
                env.issued_banks.insert(ExecutableAllocationId(9));
                execute_schedule::<FakeTarget,NoCommands>(&[branch],&mut NoCommands,&mut env).unwrap();
                assert_eq!(env.tensor_values.len(),1,"arm-local definitions have retired");
                let result=&env.tensor_values[&3];
                assert_eq!(result.allocation,ExecutableAllocationId(if taken {0}else{1}));
                assert_eq!(result.extents,vec![seismic_lang::expr::BigUint::from(if taken {2u64}else{5u64})]);
                assert!(env.issued_banks.contains(&ExecutableAllocationId(9)),"value retirement cannot release pending commands");
            }
            assert_eq!(resources.acquired,vec![ExecutableAllocationId(if taken {0}else{1})],"untaken arm acquires nothing");
        }
    }

    #[test]
    fn argument_binding_preserves_validated_axes_and_actual_strided_backing() {
        struct Resources(RuntimeBuffer<()>);
        impl ExecutionResources<()> for Resources {
            fn buffer(&self, _: ExecutableAllocationId) -> &RuntimeBuffer<()> { &self.0 }
            fn acquire_instance(&mut self, _: ExecutableAllocationId, _: u64, _: u64) -> Result<(), ExecutionError> { unreachable!() }
            fn retire_completed_instances(&mut self, _: &[ExecutableAllocationId]) {}
        }
        let mut arena = seismic_lang::expr::ExprArena::default();
        let stride = arena.schedule_slot(0, seismic_lang::expr::SymbolSort::Nat);
        let actual_stride = arena.nat_symbol(stride);
        let zero = arena.nat(0);
        let axis = arena.nat(3);
        let representation = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let mut resources = Resources(RuntimeBuffer {
            tensor: Some(RuntimeTensorGeometry { representation, extents: vec![3], strides: vec![2] }),
            buffer: (), base_offset: 16, accessible_bytes: 20,
        });
        let mut values = InvocationValues::new();
        let mut env = ExecutionEnvironment {
            native_index_bits: 64, device: &NoDevice, resources: &mut resources, values: &mut values,
            tensor_values: Default::default(), issued_banks: Default::default(),
            published: Default::default(), kernels: &[],
        };
        let step = ExecutableStep::BindArgumentTensor {
            allocation: ExecutableAllocationId(0), representation, result: 1, strides: vec![stride],
        };
        execute_schedule::<FakeTarget, NoCommands>(&[step], &mut NoCommands, &mut env).unwrap();
        let view = CompiledBufferView {
            base: CompiledViewBase::TensorValue(1), representation,
            byte_offset: arena.compile_nat(zero), extents: vec![arena.compile_nat(axis)],
            strides: vec![arena.compile_nat(actual_stride)],
        };
        let resolved = env.resolve_view(&view).unwrap();
        assert_eq!(resolved.extents, [3]);
        assert_eq!(resolved.strides, [2]);
        assert_eq!(resolved.byte_offset, 16);
        assert_eq!(resolved.byte_span, 20);
    }

    #[test]
    fn begin_freezes_whole_tensor_geometry_for_the_actual_instance() {
        struct Resources(RuntimeBuffer<()>);
        impl ExecutionResources<()> for Resources {
            fn buffer(&self, _: ExecutableAllocationId) -> &RuntimeBuffer<()> { &self.0 }
            fn acquire_instance(&mut self, _: ExecutableAllocationId, bytes: u64, _: u64) -> Result<(), ExecutionError> {
                self.0.accessible_bytes = bytes; Ok(())
            }
            fn retire_completed_instances(&mut self, _: &[ExecutableAllocationId]) {}
        }
        let mut arena = seismic_lang::expr::ExprArena::default();
        let symbol = arena.schedule_slot(0, seismic_lang::expr::SymbolSort::Nat);
        let extent = arena.nat_symbol(symbol);
        let zero = arena.nat(0);
        let one = arena.nat(1);
        let four = arena.nat(4);
        let bytes = arena.nat_mul(extent, four);
        let axis_symbol = arena.schedule_slot(1, seismic_lang::expr::SymbolSort::Nat);
        let stride_symbol = arena.schedule_slot(2, seismic_lang::expr::SymbolSort::Nat);
        let actual_axis = arena.nat_symbol(axis_symbol);
        let actual_stride = arena.nat_symbol(stride_symbol);
        let view = |base, axis, stride| CompiledBufferView {
            base, representation: seismic_lang::registry::dense(seismic_lang::types::DType::I32),
            byte_offset: arena.compile_nat(zero), extents: vec![arena.compile_nat(axis)],
            strides: vec![arena.compile_nat(stride)],
        };
        let whole = view(CompiledViewBase::TensorValue(1), actual_axis, actual_stride);
        let begin = ExecutableStep::BeginAllocationInstance {
            allocation: ExecutableAllocationId(0), banks: vec![ExecutableAllocationId(0), ExecutableAllocationId(1)],
            bytes: arena.compile_nat(bytes), alignment: 4,
            geometry: view(CompiledViewBase::Allocation(ExecutableAllocationId(0)), extent, one),
            result: CompiledRegionDestination::Tensor { id: 1, extents: vec![CompiledTensorAxisDestination::Bound(axis_symbol)], strides: vec![stride_symbol] },
        };
        let mut resources = Resources(RuntimeBuffer { tensor: None, buffer: (), base_offset: 0, accessible_bytes: 0 });
        let mut values = InvocationValues::new();
        values.bind(symbol, SymbolValue::Nat(3u64.into()));
        let mut env = ExecutionEnvironment {
            native_index_bits: 64, device: &NoDevice, resources: &mut resources, values: &mut values,
            tensor_values: Default::default(),
            issued_banks: Default::default(), published: Default::default(), kernels: &[],
        };
        assert!(matches!(env.resolve_view(&whole), Err(ExecutionError::ConstructionContradiction(_))));
        execute_schedule::<FakeTarget, NoCommands>(std::slice::from_ref(&begin), &mut NoCommands, &mut env).unwrap();
        env.values.bind(symbol, SymbolValue::Nat(99u64.into()));
        let resolved = env.resolve_view(&whole).unwrap();
        assert_eq!(resolved.extents, [3]);
        assert_eq!(resolved.byte_span, 12);
        assert!(matches!(env.resolve_view(&view(CompiledViewBase::Allocation(ExecutableAllocationId(0)), extent, one)), Err(ExecutionError::ConstructionContradiction(_))));
        // Capturing an alias keeps the old instance even after this exact
        // Begin definition executes again with identical geometry.
        let old = env.capture_view(&whole).unwrap();
        env.tensor_values.insert(2, old.clone());
        env.values.bind(symbol, SymbolValue::Nat(3u64.into()));
        execute_schedule::<FakeTarget, NoCommands>(std::slice::from_ref(&begin), &mut NoCommands, &mut env).unwrap();
        let current = env.capture_view(&whole).unwrap();
        assert_ne!(old.allocation, current.allocation);
        assert_eq!(env.tensor_values[&2].allocation, old.allocation);
        assert_eq!(env.tensor_values[&2].extents, current.extents);

    }

    #[test]
    fn reached_capacity_reclaims_completed_banks_but_keeps_live_products_charged() {
        let mut resources = limited_resources();
        let completed = resources.completed.clone();
        let available = available_with_retained_product();
        acquire_with_completed_reclamation(&mut resources, ExecutableAllocationId(0), 16, 4, &available, || {
            completed.store(true, Ordering::SeqCst);
            Ok(())
        }).unwrap();
        assert_eq!(resources.bytes, [16, 0, 8]);
        assert_eq!(resources.attempts, 2);
        assert_eq!(resources.retired, [ExecutableAllocationId(1)]);
        // The same live product must still prevent a genuinely excessive request.
        assert_eq!(resources.acquire_instance(ExecutableAllocationId(0), 17, 4),
            Err(ExecutionError::AllocationCapacity { required: 17u64.into(), available: 16 }));
    }

    #[test]
    fn failed_prefix_completion_keeps_pending_banks_charged_and_does_not_retry() {
        let mut resources = limited_resources();
        let result = acquire_with_completed_reclamation(&mut resources, ExecutableAllocationId(0), 16, 4,
            &available_with_retained_product(), || Err(ExecutionError::ConstructionContradiction("completion failed".into())));
        assert!(matches!(result, Err(ExecutionError::ConstructionContradiction(_))));
        assert_eq!(resources.bytes, [0, 16, 8]);
        assert_eq!(resources.attempts, 1);
        assert!(resources.retired.is_empty());
    }

    #[test]
    fn exact_reached_demand_beyond_host_width_is_a_capacity_refusal() {
        let mut arena = seismic_lang::expr::ExprArena::default();
        let required = seismic_lang::expr::BigUint::from(1u32) << 90usize;
        let bytes = arena.nat_exact(required.clone());
        let bytes = arena.compile_nat(bytes);
        assert_eq!(reached_allocation_bytes(ExecutableAllocationId(0), &bytes, &InvocationValues::new()),
            Err(ExecutionError::AllocationCapacity { required, available: u64::MAX }));
    }

    #[test]
    fn undefined_reached_size_is_a_construction_error_not_a_source_or_capacity_failure() {
        let mut arena = seismic_lang::expr::ExprArena::default();
        let negative = arena.int(-1);
        let invalid = arena.nat_from_int(negative);
        let bytes = arena.compile_nat_with(invalid, &seismic_lang::expr::PartialAssignment::new());
        let error = reached_allocation_bytes(ExecutableAllocationId(0), &bytes, &InvocationValues::new()).unwrap_err();
        assert!(matches!(error, ExecutionError::ConstructionContradiction(ref detail) if detail.contains("NegativeNat")));
    }

    #[test]
    fn exact_repeat_visits_the_successor_of_u64_max() {
        use seismic_ir::construction::Construction;
        use seismic_ir::schedule::{HostQuantityKind, HostValueDestination};
        let mut arena = seismic_lang::expr::ExprArena::default();
        let mut construction = Construction::<FakeTarget>::new(&mut arena, vec![], false, 0);
        let slot = construction.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Natural);
        let comparison = construction.schedule(&mut arena, 0).slot_any(seismic_ir::repr::ScalarKind::Scalar(seismic_lang::types::DType::Bool));
        let (binder, symbol, index) = arena.nat_loop_binder();
        let start = arena.nat(u64::MAX);
        let one = arena.nat(1);
        let end = arena.nat_add(start, one);
        let observed = arena.nat_symbol(slot.symbol());
        let equal = arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, observed, start);
        let steps = [ExecutableStep::Repeat {
            range: RuntimeRange { start: arena.compile_nat(start), end: arena.compile_nat(end) },
            binder: LoopBinder { binder, symbol },
            body: vec![ExecutableStep::EvaluateHost {
                value: CompiledHostValue::Natural(arena.compile_nat(index)),
                to: HostValueDestination::Quantity(slot), failure: None,
            }].into_boxed_slice(),
            carries: seismic_ir::region::Product::Unit,
        }, ExecutableStep::EvaluateHost {
            value: CompiledHostValue::Bool(arena.compile_bool(equal)),
            to: HostValueDestination::Native(comparison), failure: None,
        }];
        let mut resources = limited_resources();
        let mut values = InvocationValues::new();
        let device = NoDevice;
        let mut env = ExecutionEnvironment {
            native_index_bits: 64,
            device: &device, resources: &mut resources, values: &mut values,
            tensor_values: Default::default(),
            issued_banks: Default::default(), published: Default::default(), kernels: &[],
        };
        execute_schedule::<FakeTarget, NoCommands>(&steps, &mut NoCommands, &mut env).unwrap();
        assert_eq!(env.values.get(slot.symbol()), Some(SymbolValue::Nat(u64::MAX.into())));
        assert_eq!(env.values.get(comparison.symbol()), Some(SymbolValue::Bool(true)));
        // Exact source execution above remains valid. Finite representation is
        // required only when this value is consumed by a native argument.
        assert_eq!(env.physical_natural(&arena.compile_nat(start), "native argument").unwrap(), u64::MAX);
        assert!(matches!(env.physical_natural(&arena.compile_nat(end), "native argument"),
            Err(ExecutionError::AllocationCapacity { required, available: u64::MAX })
                if required == seismic_lang::expr::BigUint::from(u64::MAX) + 1u8));
        env.native_index_bits = 8;
        assert!(matches!(env.physical_natural(&arena.compile_nat(start), "native argument"),
            Err(ExecutionError::AllocationCapacity { available: 255, .. })));
    }
}
