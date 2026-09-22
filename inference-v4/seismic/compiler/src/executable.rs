//! Executable representation and the runtime service contracts (spec §7.5,
//! §11.3, §12.3).
//!
//! `ExecutableStep<C, P, R>` is the one structured control type used by the
//! frozen plan (over physical commands) and by every backend's executable
//! variant (over native commands). CPU, Metal, and CUDA do not define
//! schedule mirrors.
//!
//! An `ExecutableVariant<T, H>` contains exactly: identity, exact guard
//! evaluator, duration evaluator, allocation/layout evaluators, one structured
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
use seismic_ir::schedule::AnyScalarSlot;
use seismic_ir::storage::GlobalBufferKind;
use seismic_lang::expr::compiled::{
    Compiled, CompiledDuration, CompiledNat, CompiledPredicate, InvocationValues,
};
use seismic_lang::expr::SymbolValue;
use seismic_lang::expr::{AnyExpr, ExprArena, LoopBinderId, PartialAssignment, SymbolKind};
use std::fmt;

/// Structured control over commands `C`, predicates `P`, ranges `R`.
#[derive(Debug)]
pub enum ExecutableStep<C, P, R> {
    Command(C),
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
    },
    Repeat {
        range: R,
        binder: LoopBinder,
        body: Box<[ExecutableStep<C, P, R>]>,
    },
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecutableAllocationId(u32);

/// One evaluated view binding. Allocation identity and byte base offset are
/// compiled together; native code never assumes every view begins at zero.
#[derive(Debug)]
pub struct CompiledBufferView {
    allocation: ExecutableAllocationId,
    pub representation: seismic_lang::ids::RepresentationId,
    pub byte_offset: CompiledNat,
    pub extents: Vec<CompiledNat>,
    pub strides: Vec<CompiledNat>,
}

impl CompiledBufferView {
    pub fn allocation_index(&self) -> usize {
        self.allocation.0 as usize
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
        mode: B::NativeLaunchMode,
        grid: [CompiledNat; 3],
        workgroup: [CompiledNat; 3],
        empty: CompiledPredicate,
        bindings: Vec<CompiledBufferView>,
        nat_args: Vec<CompiledNat>,
        scalar_args: Vec<seismic_lang::expr::SymbolId>,
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
    pub kind: ExecutableAllocationKind,
    /// Runtime byte expressions of all logical allocations coalesced into
    /// this physical slot. The driver allocates their maximum.
    byte_candidates: AllocationByteCandidates,
    pub alignment: u64,
}

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
        view: CompiledBufferView,
        bytes: CompiledNat,
    },
    Scalar {
        slot: AnyScalarSlot,
        kind: ExecutableScalarResultKind,
    },
    Range {
        start: AnyScalarSlot,
        end: AnyScalarSlot,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutableScalarResultKind {
    Value(seismic_lang::types::DType),
    Index,
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
    device: seismic_target::DeviceDescriptionIdentity,
    evaluation: crate::evaluation::EvaluationIdentity,
    identity: VariantIdentity,
    guard: RuntimePredicate,
    duration: CompiledDuration,
    allocations: Vec<AllocationPlan>,
    slots: Vec<AnyScalarSlot>,
    schedule: Box<[NativeStep<B>]>,
    bindings: CallBindingTable,
    numerical: NumericalAssessment,
    provenance: crate::refinement::ImplementationProvenance,
}

/// One selected, fully lowered variant before native handles are attached.
pub struct PlannedVariant<B: seismic_target::TargetFamily> {
    body: CompiledVariantBody<B>,
    native_artifacts: Vec<seismic_target::NativeKernelIdentity>,
}

/// One native executable variant.
pub struct ExecutableVariant<T: seismic_target::TargetFamily, H> {
    body: CompiledVariantBody<T>,
    kernels: Vec<std::sync::Arc<seismic_target::NativeKernel<T, H>>>,
}

impl<T: seismic_target::TargetFamily, H> ExecutableVariant<T, H> {
    pub fn device_identity(&self) -> &seismic_target::DeviceDescriptionIdentity {
        &self.body.device
    }
    pub fn evaluation_identity(&self) -> &crate::evaluation::EvaluationIdentity {
        &self.body.evaluation
    }
    pub fn identity(&self) -> &VariantIdentity {
        &self.body.identity
    }
    pub fn guard(&self) -> &RuntimePredicate {
        &self.body.guard
    }
    pub fn duration(&self) -> &CompiledDuration {
        &self.body.duration
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
        &self.body.numerical
    }
    pub fn provenance(&self) -> &crate::refinement::ImplementationProvenance {
        &self.body.provenance
    }
    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
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
                    locals,
                    addressable_resources,
                    abi,
                    ..
                } => {
                    let binding_nested = bindings.iter().map(view).sum::<usize>();
                    let local_nested = locals
                        .iter()
                        .map(|local| {
                            vec_storage(&local.extents).saturating_add(vec_storage(&local.strides))
                        })
                        .sum::<usize>();
                    vec_storage(bindings)
                        .saturating_add(binding_nested)
                        .saturating_add(vec_storage(nat_args))
                        .saturating_add(vec_storage(scalar_args))
                        .saturating_add(vec_storage(locals))
                        .saturating_add(local_nested)
                        .saturating_add(vec_storage(addressable_resources))
                        .saturating_add(abi.allocations.len().saturating_mul(std::mem::size_of::<(
                            seismic_ir::target::KernelAbiAllocationRole,
                            ExecutableAllocationId,
                        )>(
                        )))
                }
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
        fn steps<B: seismic_target::TargetFamily>(values: &[NativeStep<B>]) -> usize {
            values
                .len()
                .saturating_mul(std::mem::size_of::<NativeStep<B>>())
                .saturating_add(values.iter().fold(0usize, |bytes, step| {
                    bytes.saturating_add(match step {
                        ExecutableStep::Command(command_value) => command(command_value),
                        ExecutableStep::If {
                            then_steps,
                            else_steps,
                            ..
                        } => steps(then_steps).saturating_add(steps(else_steps)),
                        ExecutableStep::Repeat { body, .. } => steps(body),
                        ExecutableStep::Check { .. } => 0,
                    })
                }))
        }
        fn result(value: &ExecutableResultBinding) -> usize {
            match value {
                ExecutableResultBinding::Buffer { view: buffer, .. } => view(buffer),
                ExecutableResultBinding::Scalar { .. } | ExecutableResultBinding::Range { .. } => 0,
            }
        }
        let body = &self.body;
        let bytes =
            std::mem::size_of_val(self)
                .saturating_add(vec_storage(&body.allocations))
                .saturating_add(body.allocations.iter().fold(0usize, |bytes, allocation| {
                    bytes.saturating_add(
                        allocation.byte_candidates.rest.len() * std::mem::size_of::<CompiledNat>(),
                    )
                }))
                .saturating_add(vec_storage(&body.slots))
                .saturating_add(vec_storage(&self.kernels))
                .saturating_add(steps(&body.schedule))
                .saturating_add(vec_storage(&body.bindings.arguments))
                .saturating_add(
                    body.bindings
                        .arguments
                        .iter()
                        .flatten()
                        .map(view)
                        .sum::<usize>(),
                )
                .saturating_add(vec_storage(&body.bindings.results))
                .saturating_add(body.bindings.results.iter().fold(
                    0usize,
                    |bytes, (path, binding)| {
                        bytes
                            .saturating_add(vec_storage(path))
                            .saturating_add(result(binding))
                    },
                ))
                .saturating_add(vec_storage(&body.provenance.callees));
        u64::try_from(bytes).unwrap_or(u64::MAX)
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
    plan: FrozenPlan<B>,
) -> PlannedVariant<B> {
    let parts = plan.into_exact_parts();
    let arena = &*parts.arena;
    let implementation = &*parts.implementation;
    let topology = implementation.global_allocations();
    let kernel_arena = implementation.kernels();
    let native_descriptions = implementation.native_descriptions();
    let native_artifacts = native_descriptions
        .iter()
        .map(|description| description.identity.clone())
        .collect::<Vec<_>>();
    let (allocation_remap, mut allocations) =
        compile_allocations(arena, &parts.fixed, topology, &parts.assignment);
    let scratch_bindings = compile_launch_scratch(
        arena,
        &parts.fixed,
        implementation.launch_scratch(),
        &mut allocations,
    );
    let abi_bindings = compile_launch_abi(
        arena,
        &parts.fixed,
        implementation.launch_abi(),
        &mut allocations,
    );
    let view = |value: seismic_ir::storage::AnyBufferView| {
        compile_view(arena, &parts.fixed, topology, &allocation_remap, value)
    };
    let schedule = compile_steps(
        implementation.native_launch_modes(),
        arena,
        &parts.fixed,
        topology,
        &allocation_remap,
        kernel_arena,
        implementation.schedule(),
        implementation.launch_layouts(),
        &scratch_bindings,
        &abi_bindings,
        implementation.schedule().steps(),
        &parts.assignment,
    )
    .into_boxed_slice();
    let mut arguments = Vec::with_capacity(parts.schema.parameters().len());
    for parameter in parts.schema.parameters() {
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
                    view: output,
                    bytes,
                } => ExecutableResultBinding::Buffer {
                    view: view(output),
                    bytes: arena.compile_nat_with(bytes, &parts.fixed),
                },
                crate::refinement::PublishedResult::Scalar { slot, kind } => {
                    let kind = match kind {
                        crate::refinement::PublishedScalarKind::Value(dtype) => {
                            ExecutableScalarResultKind::Value(dtype)
                        }
                        crate::refinement::PublishedScalarKind::Index => {
                            ExecutableScalarResultKind::Index
                        }
                    };
                    ExecutableResultBinding::Scalar { slot, kind }
                }
                crate::refinement::PublishedResult::Range { start, end } => {
                    ExecutableResultBinding::Range { start, end }
                }
            };
            (publication.path.clone(), binding)
        })
        .collect();
    let bindings = CallBindingTable { arguments, results };
    let duration = parts.performance.estimate();
    let mut duration_symbols = arena.free_symbols(AnyExpr::Duration(duration));
    duration_symbols.sort();
    duration_symbols.dedup();
    let duration_invocation_evaluable =
        invocation_evaluable(arena, AnyExpr::Duration(duration), &parts.fixed);
    assert!(
        duration_invocation_evaluable,
        "closed implementation duration retained a non-invocation symbol: {}",
        duration_symbols
            .iter()
            .filter(|symbol| {
                parts.fixed.get(**symbol).is_none()
                    && !matches!(
                        arena.symbol_kind(**symbol),
                        SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_)
                    )
            })
            .map(|symbol| format!("{symbol:?}:{:?}", arena.symbol_kind(*symbol)))
            .collect::<Vec<_>>()
            .join(",")
    );
    PlannedVariant {
        native_artifacts,
        body: CompiledVariantBody {
            device: parts.device,
            evaluation: parts.evaluation,
            identity: parts.identity,
            guard: arena.compile_bool_with(parts.guard.node(), parts.guard.fixed()),
            duration: arena.compile_duration_with(duration, &parts.fixed),
            allocations,
            slots: implementation.schedule().slots().to_vec(),
            schedule,
            bindings,
            numerical: parts.numerical,
            provenance: implementation.provenance().clone(),
        },
    }
}

impl<B: seismic_target::TargetFamily> PlannedVariant<B> {
    pub fn identity(&self) -> &VariantIdentity {
        &self.body.identity
    }
    pub fn guard(&self) -> &RuntimePredicate {
        &self.body.guard
    }
    pub fn duration(&self) -> &CompiledDuration {
        &self.body.duration
    }

    pub(crate) fn retained_metadata_bytes(&self) -> u64 {
        // The final variant replaces each artifact identity with one Arc. The
        // planning budget therefore accounts the exact projected handle table
        // size without requiring the handles themselves.
        let identity_bytes = self.native_artifacts.capacity()
            * std::mem::size_of::<seismic_target::NativeKernelIdentity>();
        let handle_bytes =
            self.native_artifacts.capacity() * std::mem::size_of::<std::sync::Arc<()>>();
        let temporary = std::mem::size_of_val(self).saturating_add(identity_bytes);
        let materialized = std::mem::size_of_val(&self.body)
            .saturating_add(std::mem::size_of::<Vec<()>>())
            .saturating_add(handle_bytes)
            .saturating_add(self.body.retained_metadata_bytes());
        u64::try_from(materialized.max(temporary)).unwrap_or(u64::MAX)
    }
}

pub(crate) trait EvaluatedVariant {
    fn selection_identity(&self) -> &VariantIdentity;
    fn selection_guard(&self) -> &RuntimePredicate;
    fn selection_duration(&self) -> &CompiledDuration;
}

impl<B: seismic_target::TargetFamily> EvaluatedVariant for PlannedVariant<B> {
    fn selection_identity(&self) -> &VariantIdentity {
        self.identity()
    }
    fn selection_guard(&self) -> &RuntimePredicate {
        self.guard()
    }
    fn selection_duration(&self) -> &CompiledDuration {
        self.duration()
    }
}

impl<T: seismic_target::TargetFamily, H> EvaluatedVariant for ExecutableVariant<T, H> {
    fn selection_identity(&self) -> &VariantIdentity {
        self.identity()
    }
    fn selection_guard(&self) -> &RuntimePredicate {
        self.guard()
    }
    fn selection_duration(&self) -> &CompiledDuration {
        self.duration()
    }
}

/// The single invocation-selection policy shared before and after exact
/// native materialization.
pub(crate) fn select_variant_index<V: EvaluatedVariant>(
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
    variants
        .iter()
        .enumerate()
        .filter_map(|(index, variant)| {
            variant
                .selection_guard()
                .evaluate(values)
                .unwrap_or_else(|error| {
                    panic!("validated invocation could not evaluate a variant guard: {error:?}")
                })
                .then(|| {
                    let duration = variant
                        .selection_duration()
                        .evaluate(values)
                        .unwrap_or_else(|error| {
                            panic!(
                                "validated invocation could not evaluate a variant duration: {error:?}"
                            )
                        });
                    let identity = variant.selection_identity();
                    (
                        index,
                        duration.upper(),
                        (
                            identity.implementation.factory.name,
                            identity.implementation.factory.revision,
                            &identity.implementation.structure,
                            &identity.assignment,
                        ),
                    )
                })
        })
        .min_by(|left, right| left.1.cmp(&right.1).then(left.2.cmp(&right.2)))
        .map(|selected| selected.0)
        .expect("validated invocation has no applicable evaluated variant")
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
                    locals,
                    addressable_resources,
                    abi,
                    ..
                } => vec_storage(bindings)
                    .saturating_add(bindings.iter().map(view).sum::<usize>())
                    .saturating_add(vec_storage(nat_args))
                    .saturating_add(vec_storage(scalar_args))
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
        fn steps<B: seismic_target::TargetFamily>(values: &[NativeStep<B>]) -> usize {
            values
                .len()
                .saturating_mul(std::mem::size_of::<NativeStep<B>>())
                .saturating_add(
                    values
                        .iter()
                        .map(|step| match step {
                            ExecutableStep::Command(value) => command(value),
                            ExecutableStep::If {
                                then_steps,
                                else_steps,
                                ..
                            } => steps(then_steps).saturating_add(steps(else_steps)),
                            ExecutableStep::Repeat { body, .. } => steps(body),
                            ExecutableStep::Check { .. } => 0,
                        })
                        .sum::<usize>(),
                )
        }
        fn result(value: &ExecutableResultBinding) -> usize {
            match value {
                ExecutableResultBinding::Buffer { view: buffer, .. } => view(buffer),
                ExecutableResultBinding::Scalar { .. } | ExecutableResultBinding::Range { .. } => 0,
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
    planned: PlannedVariant<T>,
    registry: &crate::realization::RealizationRegistry<T, H>,
) -> ExecutableVariant<T, H> {
    let kernels = registry.resolve(
        &planned.body.device,
        &planned.body.identity.implementation,
        &planned.native_artifacts,
    );
    ExecutableVariant {
        body: planned.body,
        kernels,
    }
}

fn invocation_evaluable(arena: &ExprArena, expression: AnyExpr, fixed: &PartialAssignment) -> bool {
    arena.free_symbols(expression).iter().all(|symbol| {
        fixed.get(*symbol).is_some()
            || matches!(
                arena.symbol_kind(*symbol),
                SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_)
            )
    })
}

fn compile_allocations(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    topology: &seismic_ir::storage::GlobalAllocationTopology,
    assignment: &crate::solve::FeasibleAssignment,
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
                Some(decision) => assignment.value(decision).unwrap_or_else(|| {
                    panic!("arena reuse decision is absent from exact assignment")
                }),
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
            kind: ExecutableAllocationKind::Global(kind),
            byte_candidates: AllocationByteCandidates::one(
                arena.compile_nat_with(allocation.bytes, fixed),
            ),
            alignment: allocation.alignment,
        });
    }
    for (first, rest) in arena_groups.into_values() {
        let physical = plans.len() as u32;
        let alignment = rest.iter().fold(
            topology.allocations()[first].alignment,
            |alignment, index| alignment.max(topology.allocations()[*index].alignment),
        );
        remap[first] = physical;
        let first_bytes = arena.compile_nat_with(topology.allocations()[first].bytes, fixed);
        let byte_candidates = rest
            .iter()
            .map(|index| {
                remap[*index] = physical;
                arena.compile_nat_with(topology.allocations()[*index].bytes, fixed)
            })
            .collect::<Vec<_>>();
        plans.push(AllocationPlan {
            kind: ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Arena),
            byte_candidates: AllocationByteCandidates::with_rest(first_bytes, byte_candidates),
            alignment,
        });
    }
    assert!(
        remap.iter().all(|value| *value != u32::MAX),
        "every logical allocation must map to one physical allocation"
    );
    (remap, plans)
}

fn compile_launch_scratch(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    requirements: &[seismic_ir::storage::LaunchScratchRequirements],
    allocations: &mut Vec<AllocationPlan>,
) -> Vec<LaunchScratchBindings> {
    requirements
        .iter()
        .enumerate()
        .map(|(launch, requirement)| {
            let mut add = |class, requirement: &Option<seismic_ir::storage::ScratchRequirement>| {
                requirement.as_ref().map(|requirement| {
                    let allocation = u32::try_from(allocations.len())
                        .expect("executable allocation ordinal space exhausted");
                    allocations.push(AllocationPlan {
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
            LaunchScratchBindings {
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
            }
        })
        .collect()
}

fn compile_launch_abi(
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    requirements: &[Vec<seismic_ir::storage::LaunchAbiRequirement>],
    allocations: &mut Vec<AllocationPlan>,
) -> Vec<KernelAbiBindings> {
    requirements
        .iter()
        .enumerate()
        .map(|(launch, requirements)| {
            let mut bindings = Vec::with_capacity(requirements.len());
            for requirement in requirements {
                let allocation = u32::try_from(allocations.len())
                    .expect("executable allocation ordinal space exhausted");
                allocations.push(AllocationPlan {
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
            KernelAbiBindings {
                allocations: bindings.into_boxed_slice(),
            }
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
            (layout.allocation == id).then(|| topology.view_id(index as u32))
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
        allocation: ExecutableAllocationId(allocation_remap[layout.allocation.index() as usize]),
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
    native_launch_modes: &[B::NativeLaunchMode],
    arena: &seismic_lang::expr::ExprArena,
    fixed: &seismic_lang::expr::PartialAssignment,
    topology: &seismic_ir::storage::GlobalAllocationTopology,
    allocation_remap: &[u32],
    kernels: &seismic_ir::kernel::KernelArena<B>,
    schedule: &seismic_ir::schedule::ParametricSchedule,
    launch_layouts: &[seismic_ir::storage::LaunchLocalLayout],
    scratch_bindings: &[LaunchScratchBindings],
    abi_bindings: &[KernelAbiBindings],
    steps: &[seismic_ir::schedule::ScheduleStep],
    assignment: &crate::solve::FeasibleAssignment,
) -> Vec<NativeStep<B>> {
    let view = |value| compile_view(arena, fixed, topology, allocation_remap, value);
    let mut out = Vec::new();
    for step in steps {
        let compiled = match step {
            seismic_ir::schedule::ScheduleStep::Launch(id) => {
                let launch = schedule.launch(*id);
                let kernel = kernels.kernel(launch.kernel);
                let layout = launch_layouts.get(id.index() as usize).unwrap_or_else(|| {
                    panic!(
                        "closed implementation has no local layout for launch {}",
                        id.index()
                    )
                });
                assert_eq!(
                    layout.locals.len(),
                    kernel.locals().len(),
                    "closed launch-local layout does not match its kernel locals"
                );
                if let Some(logical) = launch.logical_base {
                    assert!(
                        (logical.argument as usize) < kernel.interface().nat_args.len(),
                        "logical launch base names an absent natural kernel argument"
                    );
                }
                ExecutableStep::Command(ExecutableCommand::Launch {
                    kernel: ExecutableKernelId(launch.kernel.ordinal()),
                    mode: native_launch_modes[id.index() as usize].clone(),
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
                                .filter(|logical| logical.argument as usize == argument)
                                .map_or(*value, |logical| logical.value);
                            arena.compile_nat_with(value, fixed)
                        })
                        .collect(),
                    scalar_args: kernel
                        .interface()
                        .scalar_args
                        .iter()
                        .map(|(symbol, _)| *symbol)
                        .collect(),
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
                    scratch: *scratch_bindings
                        .get(id.index() as usize)
                        .unwrap_or_else(|| {
                            panic!(
                                "closed implementation has no scratch realization for launch {}",
                                id.index()
                            )
                        }),
                    abi: abi_bindings
                        .get(id.index() as usize)
                        .cloned()
                        .unwrap_or_else(|| {
                            panic!(
                                "closed implementation has no ABI realization for launch {}",
                                id.index()
                            )
                        }),
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
            } => ExecutableStep::If {
                condition: arena.compile_bool_with(*condition, fixed),
                then_steps: compile_steps(
                    native_launch_modes,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_layouts,
                    scratch_bindings,
                    abi_bindings,
                    then_steps,
                    assignment,
                )
                .into_boxed_slice(),
                else_steps: compile_steps(
                    native_launch_modes,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_layouts,
                    scratch_bindings,
                    abi_bindings,
                    else_steps,
                    assignment,
                )
                .into_boxed_slice(),
            },
            seismic_ir::schedule::ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                body,
            } => ExecutableStep::Repeat {
                range: RuntimeRange {
                    start: arena.compile_nat_with(*start, fixed),
                    end: arena.compile_nat_with(*end, fixed),
                },
                binder: LoopBinder {
                    binder: *binder,
                    symbol: *symbol,
                },
                body: compile_steps(
                    native_launch_modes,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_layouts,
                    scratch_bindings,
                    abi_bindings,
                    body,
                    assignment,
                )
                .into_boxed_slice(),
            },
            seismic_ir::schedule::ScheduleStep::Choose { decision, options } => {
                let selected = assignment
                    .value(*decision)
                    .unwrap_or_else(|| panic!("schedule choice is absent from exact assignment"));
                let body = options
                    .iter()
                    .find_map(|(value, body)| (*value == selected).then_some(body))
                    .unwrap_or_else(|| {
                        panic!("exact assignment selected an absent schedule option")
                    });
                out.extend(compile_steps(
                    native_launch_modes,
                    arena,
                    fixed,
                    topology,
                    allocation_remap,
                    kernels,
                    schedule,
                    launch_layouts,
                    scratch_bindings,
                    abi_bindings,
                    body,
                    assignment,
                ));
                continue;
            }
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
pub struct RuntimeBuffer<T> {
    pub buffer: T,
    pub base_offset: u64,
    pub accessible_bytes: u64,
}

pub struct ResolvedBufferView<'a, T> {
    pub buffer: &'a T,
    pub byte_offset: u64,
    pub extents: Vec<u64>,
    pub strides: Vec<u64>,
}

/// Values available to a command during execution: invocation symbols and
/// the current schedule slot values (bound as symbols).
pub struct ExecutionEnvironment<'a, T: seismic_target::TargetFamily, H, D: DeviceService<T>> {
    device: &'a D,
    /// Allocation index -> buffer.
    buffers: &'a [RuntimeBuffer<D::Buffer>],
    /// Invocation and slot symbol values; slots are rebound by commands.
    values: &'a mut InvocationValues,
    kernels: &'a [std::sync::Arc<seismic_target::NativeKernel<T, H>>],
}

impl<'a, T: seismic_target::TargetFamily, H, D: DeviceService<T>>
    ExecutionEnvironment<'a, T, H, D>
{
    pub fn set_slot(&mut self, slot: AnyScalarSlot, value: SymbolValue) {
        let value = match (slot.sort(), value) {
            (seismic_lang::expr::SymbolSort::Nat, SymbolValue::U32(value)) => {
                SymbolValue::Nat(u64::from(value))
            }
            (seismic_lang::expr::SymbolSort::Int, SymbolValue::I32(value)) => {
                SymbolValue::Int(i64::from(value))
            }
            (_, value) => value,
        };
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
        self.buffers
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("closed executable allocation handle is not bound"))
    }

    /// Evaluates a guarded Nat expression. Frozen-plan side conditions and
    /// the executable guard make failure a private construction contradiction, not
    /// a backend/runtime error channel.
    pub fn nat(&self, value: &CompiledNat) -> u64 {
        value
            .evaluate(self.values)
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

    /// Resolves a compiler-owned relative view against the invocation's
    /// physical buffer base. Failure here is an invariant violation: entry
    /// validation and the executable guard have already proved these bounds.
    pub fn resolve_view(&self, view: &CompiledBufferView) -> ResolvedBufferView<'_, D::Buffer> {
        let allocation = self
            .buffers
            .get(view.allocation.0 as usize)
            .unwrap_or_else(|| panic!("executable view references an unbound physical allocation"));
        let relative = view
            .byte_offset
            .evaluate(self.values)
            .unwrap_or_else(|error| panic!("guarded view offset failed evaluation: {error:?}"));
        let extents = view
            .extents
            .iter()
            .map(|value| {
                value.evaluate(self.values).unwrap_or_else(|error| {
                    panic!("guarded view extent failed evaluation: {error:?}")
                })
            })
            .collect::<Vec<_>>();
        let strides = view
            .strides
            .iter()
            .map(|value| {
                value.evaluate(self.values).unwrap_or_else(|error| {
                    panic!("guarded view stride failed evaluation: {error:?}")
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            extents.len(),
            strides.len(),
            "compiled view rank and stride count differ"
        );
        let mut address_extents = extents.clone();
        let unit_bytes = match &seismic_lang::registry::representation_info(view.representation)
            .kind
        {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => u64::from(dtype.bytes()),
            seismic_lang::registry::RepresentationKind::Packed(layout) => {
                if let Some(last) = address_extents.last_mut() {
                    *last = last
                        .checked_add(u64::from(layout.group) - 1)
                        .and_then(|value| value.checked_div(u64::from(layout.group)))
                        .expect("guarded packed extent is representable");
                }
                u64::from(layout.packet_size)
            }
            seismic_lang::registry::RepresentationKind::External(layout) => {
                if let Some(last) = address_extents.last_mut() {
                    *last = last
                        .checked_add(u64::from(layout.logical_group) - 1)
                        .and_then(|value| value.checked_div(u64::from(layout.logical_group)))
                        .expect("guarded external packet extent is representable");
                }
                u64::from(layout.packet_size)
            }
        };
        let relative_end = if address_extents.iter().any(|extent| *extent == 0) {
            relative
        } else {
            let units = address_extents
                .iter()
                .zip(&strides)
                .try_fold(1u64, |span, (extent, stride)| {
                    let axis = extent.checked_sub(1)?.checked_mul(*stride)?;
                    span.checked_add(axis)
                })
                .expect("guarded view address span is representable");
            relative
                .checked_add(
                    units
                        .checked_mul(unit_bytes)
                        .expect("guarded view byte span is representable"),
                )
                .expect("guarded view byte end is representable")
        };
        assert!(
            relative_end <= allocation.accessible_bytes,
            "compiled view exceeds its invocation binding"
        );
        let accessible_end = allocation
            .base_offset
            .checked_add(allocation.accessible_bytes)
            .expect("invocation buffer binding byte range overflows");
        assert!(
            accessible_end <= self.device.buffer_len(&allocation.buffer),
            "invocation buffer binding exceeds the physical buffer"
        );
        let byte_offset = allocation
            .base_offset
            .checked_add(relative)
            .expect("invocation buffer base plus compiled view offset overflows");
        ResolvedBufferView {
            buffer: &allocation.buffer,
            byte_offset,
            extents,
            strides,
        }
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
    buffers: &[RuntimeBuffer<<S::Device as DeviceService<T>>::Buffer>],
    values: &mut InvocationValues,
) -> Result<(), ExecutionError>
where
    T: seismic_target::TargetFamily,
    S: NativeSubmission<T>,
{
    let mut environment = ExecutionEnvironment {
        device,
        buffers,
        values,
        kernels: &variant.kernels,
    };
    execute_schedule(&variant.body.schedule, submission, &mut environment)
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
                            reason: site.reason.clone(),
                            path: site.path.clone(),
                            line: site.line,
                        },
                    ));
                }
            }
            ExecutableStep::Command(command @ ExecutableCommand::ScalarRead { bounds, .. }) => {
                let in_bounds = bounds
                    .iter()
                    .try_fold(true, |all, bound| {
                        bound.evaluate(env.values).map(|value| all && value)
                    })
                    .map_err(|error| eval_failure("scalar-read bounds", error))?;
                if !in_bounds {
                    return Err(ExecutionError::DataCheckFailed(
                        crate::errors::CheckFailure {
                            reason: "scalar read index out of bounds".into(),
                            path: "schedule scalar read".into(),
                            line: 0,
                        },
                    ));
                }
                submission.execute(command, env)?;
            }
            ExecutableStep::Command(command) => submission.execute(command, env)?,
            ExecutableStep::If {
                condition,
                then_steps,
                else_steps,
            } => {
                let taken = condition
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("branch condition", error))?;
                let branch = if taken { then_steps } else { else_steps };
                execute_schedule(branch, submission, env)?;
            }
            ExecutableStep::Repeat {
                range,
                binder,
                body,
            } => {
                let start = range
                    .start
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("repeat start", error))?;
                let end = range
                    .end
                    .evaluate(env.values)
                    .map_err(|error| eval_failure("repeat end", error))?;
                for i in start..end {
                    env.values.bind(binder.symbol, SymbolValue::Nat(i));
                    execute_schedule(body, submission, env)?;
                }
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
