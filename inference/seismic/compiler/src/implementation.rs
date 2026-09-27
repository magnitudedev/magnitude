//! Candidate-family construction and reconciled, handle-free implementations.
//!
//! An `Implementation<B>` owns, together: identity, semantic coverage,
//! parametric schedule, kernels, global and local allocation topology, finite
//! decisions, hard constraints, numerical transfer, and provenance.
//! All fields are private. Source construction closes the actual checked
//! computation into pure `ConstructedCandidate` data. Native realization is
//! coordinated by [`crate::realization`]; a realized implementation retains
//! immutable reflected descriptions, never executable handles.
//!
//! The source owner constructs calls and regions through their complete bound
//! products. Owned continuations may pause this construction, but do not expose
//! arbitrary IR authorship or replay a requested source value out of order.

mod call;
pub(crate) mod candidate;
pub(crate) mod native;
pub(crate) use call::CallConstruction;
pub(crate) use internals::BuilderState;

use crate::numerics::NumericalApplicability;
use crate::target::{CompilerRegistry, TargetConstants};
use candidate::{
    ChoiceDeclaration, ConstructedCandidate, ConstructedCandidateIdentity,
    ConstructedCandidateParts, ImplementationProvenance, PublishedResult, ResultPublication,
};
use seismic_ir::construction::Construction;
use seismic_ir::kernel::KernelArena;
use seismic_ir::repr::ScalarKind;
use seismic_ir::schedule::{
    AnyScalarSlot, ClosedSchedule, HostQuantityKind, HostQuantitySlot, ParametricSchedule,
    ScheduleBuilder,
};
use seismic_ir::storage::{
    AnyBufferView, GlobalAllocationId, GlobalAllocationTopology, LocalAllocationTopology,
};
use seismic_lang::entry::{
    AliasRule, CallSchema, CandidateKind, ParameterAccess, ParameterKind, SemanticEventKind,
    SemanticFunction, SemanticNodeView, SemanticProgram, SemanticType,
};
use seismic_lang::expr::{
    AnyExpr, BoolExpr, DecisionId, ExprArena, FiniteDomain, NatExpr, NodeView, TargetPredicate,
};
use seismic_lang::ids::{FamilyId, NodeId, RegionId, SemanticValueId, StableFunctionId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry::BackendName;
use seismic_lang::types::DType;
use seismic_native_target::DeviceDescription;
use std::fmt;
use std::sync::Arc;

/// Stable identity of one natively realized implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImplementationIdentity {
    /// Digest over the candidate structure and reconciled native artifacts.
    pub structure: [u8; 32],
}

/// Consumes one pure family into a natively reconciled implementation.
///
/// Universal launch specialization uses only reflected native legality.
/// Performance evaluation is owned by `evaluation` after the whole candidate
/// domain has been sealed and cannot change executable structure.
pub(crate) fn reconcile_candidate_with_descriptions<B: seismic_native_target::TargetFamily>(
    family: Arc<ConstructedCandidate<B>>,
    assignment: seismic_lang::expr::PartialAssignment,
    arena: &mut ExprArena,
    target: &DeviceDescription<B>,
    registry: &CompilerRegistry<B>,
    native: Arc<crate::realization::RealizedNativeSet<B>>,
) -> Result<Implementation<B>, crate::errors::PreparationError> {
    let reflected = native_hard_constraints(arena, target, registry, &family, &native);
    let combined = arena.and(family.hard_constraints, reflected);
    let side_conditions = arena.side_conditions(AnyExpr::Bool(combined));
    let hard_constraints = arena.and(side_conditions, combined);

    let mut digest = seismic_ir::identity::StructureDigest::new("seismic-implementation-native-v1");
    digest.bytes(&family.identity.structure);
    digest.bytes(&native.assignment_identity());
    for description in native.descriptions() {
        digest.bytes(&description.identity.compatibility.fingerprint);
        digest.bytes(&description.identity.artifact_digest);
        digest.bytes(&description.numerical_identity.fingerprint);
    }
    let identity = ImplementationIdentity {
        structure: digest.finish(),
    };
    Ok(Implementation {
        family,
        identity,
        assignment_identity: native.assignment_identity(),
        assignment,
        hard_constraints,
        native,
    })
}

#[cfg(test)]
mod implementation_invariant_tests {
    use super::*;
    use seismic_lang::expr::{CmpOp, SymbolSort};

    #[test]
    fn total_conjunction_implies_any_exact_conjunct_subset() {
        let mut arena = ExprArena::default();
        let (_, x_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, y_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, z_symbol) = arena.target_constant(SymbolSort::Nat);
        let ten = arena.nat(10);
        let x = arena.nat_symbol(x_symbol);
        let y = arena.nat_symbol(y_symbol);
        let z = arena.nat_symbol(z_symbol);
        let x_bound = arena.nat_cmp(CmpOp::Le, x, ten);
        let y_bound = arena.nat_cmp(CmpOp::Le, y, ten);
        let z_bound = arena.nat_cmp(CmpOp::Le, z, ten);
        let required = arena.and(x_bound, y_bound);
        let target_tail = arena.and(y_bound, z_bound);
        let target = arena.and(x_bound, target_tail);
        let implication = arena.implies(target, required);
        assert!(matches!(
            arena.view(AnyExpr::Bool(implication)),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn wider_same_shape_allocation_certifies_narrower_storage_bound() {
        let mut arena = ExprArena::default();
        let (_, rows_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, columns_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, memory_symbol) = arena.target_constant(SymbolSort::Nat);
        let rows = arena.nat_symbol(rows_symbol);
        let columns = arena.nat_symbol(columns_symbol);
        let memory = arena.nat_symbol(memory_symbol);
        let elements = arena.nat_mul(rows, columns);
        let four = arena.nat(4);
        let two = arena.nat(2);
        let f32_bytes = arena.nat_mul(elements, four);
        let bf16_bytes = arena.nat_mul(two, elements);
        let abi_bound = arena.nat_cmp(CmpOp::Le, f32_bytes, memory);
        let internal_bound = arena.nat_cmp(CmpOp::Le, bf16_bytes, memory);
        let implication = arena.implies(abi_bound, internal_bound);
        assert!(matches!(
            arena.view(AnyExpr::Bool(implication)),
            NodeView::BoolConst(true)
        ));

        let reverse = arena.implies(internal_bound, abi_bound);
        assert!(!matches!(
            arena.view(AnyExpr::Bool(reverse)),
            NodeView::BoolConst(true)
        ));
    }

    #[test]
    fn zero_stride_address_axis_does_not_evaluate_a_partial_extent_span() {
        let mut arena = ExprArena::default();
        let (_, start_symbol) = arena.target_constant(SymbolSort::Nat);
        let (_, end_symbol) = arena.target_constant(SymbolSort::Nat);
        let start = arena.nat_symbol(start_symbol);
        let end = arena.nat_symbol(end_symbol);
        let partial_extent = arena.nat_sub(end, start);
        let zero = arena.nat(0);

        let span = seismic_ir::storage::addressed_axis_span(&mut arena, partial_extent, zero);
        assert!(matches!(
            arena.view(AnyExpr::Nat(span)),
            NodeView::NatConst(0)
        ));
        let side_conditions = arena.side_conditions(AnyExpr::Nat(span));
        assert!(matches!(
            arena.view(AnyExpr::Bool(side_conditions)),
            NodeView::BoolConst(true)
        ));
    }
}

fn native_hard_constraints<B: seismic_native_target::TargetFamily>(
    arena: &mut ExprArena,
    target: &DeviceDescription<B>,
    registry: &CompilerRegistry<B>,
    family: &ConstructedCandidate<B>,
    native: &crate::realization::RealizedNativeSet<B>,
) -> BoolExpr {
    let mut launch_requirements = vec![arena.bool(true); family.schedule().launches().len()];
    let schedule = family.schedule();
    for (index, launch) in schedule.launches().iter().enumerate() {
        let mut constraints = Vec::new();
        let local_layout = family.launch_resources()[index].layout();
        let description = native
            .description(launch.kernel)
            .unwrap_or_else(|| panic!("launch has no realized native kernel"));
        let kernel = family.kernels().kernel(launch.kernel);
        let domain = &description.launch;
        // Reflection restricts where the selected request is applicable; it
        // never substitutes a different launch descriptor. The schedule scopes
        // this requirement to the paths on which this launch actually runs.
        constraints.push(arena.bool(domain.modes.contains(&launch.descriptor)));
        let threads = arena.nat_product(&launch.workgroup);
        if kernel.interface().uses_subgroup {
            let width =
                arena.nat(u64::from(domain.subgroup_width.expect(
                    "subgroup-using native kernel closed without a subgroup width",
                )));
            constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Ge, threads, width));
            let remainder = arena.nat_rem(threads, width);
            let zero = arena.nat(0);
            constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, remainder, zero));
            if let Some(extent) = launch.parallel_extent {
                let remainder = arena.nat_rem(extent, width);
                constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, remainder, zero));
            }
        }
        let max_threads = arena.nat(domain.max_workgroup_threads);
        constraints.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Le, threads, max_threads));
        for axis in 0..3 {
            let workgroup_max = arena.nat(domain.max_workgroup_size[axis]);
            constraints.push(arena.nat_cmp(
                seismic_lang::expr::CmpOp::Le,
                launch.workgroup[axis],
                workgroup_max,
            ));
            let grid_max = arena.nat(domain.max_grid[axis]);
            constraints.push(arena.nat_cmp(
                seismic_lang::expr::CmpOp::Le,
                launch.grid[axis],
                grid_max,
            ));
        }
        let local_max = arena.nat(domain.max_dynamic_local_bytes);
        constraints.push(arena.nat_cmp(
            seismic_lang::expr::CmpOp::Le,
            local_layout.workgroup_bytes,
            local_max,
        ));
        constraints.extend(registry.native_launch_constraints(
            target,
            arena,
            launch,
            local_layout,
            kernel,
            description,
        ));
        let predicate = arena.all(&constraints);
        let defined = arena.side_conditions(predicate.into());
        launch_requirements[index] = arena.and(defined, predicate);
    }
    use seismic_ir::schedule::{RequirementPoint, ScheduleStep};
    let storage = family.global_allocations();
    schedule.scoped_requirements(
        arena,
        storage.views(),
        &mut |arena, point| match point {
            RequirementPoint::Step(ScheduleStep::Launch(id)) => {
                launch_requirements[id.index() as usize]
            }
            _ => arena.bool(true),
        },
        &mut |arena, step| acquisition_established(arena, storage, step),
    )
}

/// A successful acquisition establishes that its allocation's geometry is
/// defined (every side condition of its byte size).
fn acquisition_established(
    arena: &mut ExprArena,
    storage: &seismic_ir::storage::GlobalAllocationTopology,
    step: &seismic_ir::schedule::ScheduleStep,
) -> BoolExpr {
    if let seismic_ir::schedule::ScheduleStep::BeginAllocationInstance { source, .. } = step {
        if let seismic_ir::storage::ViewBase::Allocation(id) =
            storage.views()[source.index() as usize].base
        {
            return arena.side_conditions(storage.allocation(id).bytes.into());
        }
    }
    arena.bool(true)
}

pub(crate) fn expression_detail(arena: &ExprArena, expression: AnyExpr, depth: usize) -> String {
    if depth == 0 {
        return format!("{expression:?}");
    }
    match arena.view(expression) {
        NodeView::NatConst(value) => value.to_string(),
        NodeView::IntConst(value) => value.to_string(),
        NodeView::BoolConst(value) => value.to_string(),
        NodeView::ScalarConst { dtype, bits } => format!("{dtype:?}(0x{bits:08x})"),
        NodeView::ScalarInteger {
            operation,
            operands,
        } => format!(
            "{operation:?}({})",
            operands
                .iter()
                .map(|(dtype, value)| format!(
                    "{dtype:?}:{}",
                    expression_detail(arena, (*value).into(), depth - 1)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        NodeView::Symbol(symbol) => format!("{symbol:?}:{:?}", arena.symbol_kind(symbol)),
        NodeView::Unary { op, operand } => {
            format!("{op:?}({})", expression_detail(arena, operand, depth - 1))
        }
        NodeView::Binary { op, lhs, rhs } => format!(
            "({} {op:?} {})",
            expression_detail(arena, lhs, depth - 1),
            expression_detail(arena, rhs, depth - 1)
        ),
        NodeView::Nary { op, operands } => format!(
            "{op:?}({})",
            operands
                .iter()
                .map(|operand| expression_detail(arena, *operand, depth - 1))
                .collect::<Vec<_>>()
                .join(",")
        ),
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => format!(
            "select({},{},{})",
            expression_detail(arena, AnyExpr::Bool(cond), depth - 1),
            expression_detail(arena, then, depth - 1),
            expression_detail(arena, otherwise, depth - 1)
        ),
        NodeView::Cmp { op, lhs, rhs } => format!(
            "({} {op:?} {})",
            expression_detail(arena, lhs, depth - 1),
            expression_detail(arena, rhs, depth - 1)
        ),
        NodeView::In { operand, values } => format!(
            "{} in {values:?}",
            expression_detail(arena, operand, depth - 1)
        ),
        NodeView::Fold {
            op,
            binder,
            start,
            extent,
            body,
        } => format!(
            "fold({op:?},{binder:?},start={},extent={},body={})",
            expression_detail(arena, AnyExpr::Nat(start), depth - 1),
            expression_detail(arena, AnyExpr::Nat(extent), depth - 1),
            expression_detail(arena, AnyExpr::Nat(body), depth - 1)
        ),
        NodeView::Duration(terms) => format!("duration({} terms)", terms.len()),
        NodeView::DurationScale { duration, by } => format!(
            "duration_scale({},{})",
            expression_detail(arena, AnyExpr::Duration(duration), depth - 1),
            expression_detail(arena, AnyExpr::Nat(by), depth - 1)
        ),
    }
}

/// The only implementation state accepted by planning and freezing. Every
/// kernel has a reconciled native artifact and all reflected limits have
/// already been folded into `hard_constraints`.
#[derive(Debug)]
pub struct Implementation<B: seismic_native_target::TargetFamily> {
    family: Arc<ConstructedCandidate<B>>,
    identity: ImplementationIdentity,
    assignment_identity: [u8; 32],
    assignment: seismic_lang::expr::PartialAssignment,
    hard_constraints: BoolExpr,
    native: Arc<crate::realization::RealizedNativeSet<B>>,
}

impl<B: seismic_native_target::TargetFamily> Implementation<B> {
    pub fn identity(&self) -> &ImplementationIdentity {
        &self.identity
    }
    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.family.semantic_coverage()
    }
    pub fn schedule(&self) -> &ParametricSchedule<B> {
        self.family.schedule()
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        self.family.kernels()
    }
    pub(crate) fn native(&self) -> &Arc<crate::realization::RealizedNativeSet<B>> {
        &self.native
    }

    pub(crate) fn assignment(&self) -> &seismic_lang::expr::PartialAssignment {
        &self.assignment
    }
    pub(crate) fn assignment_identity(&self) -> [u8; 32] {
        self.assignment_identity
    }
    pub(crate) fn native_index_bits(&self) -> u32 {
        self.family.native_index_bits
    }

    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        self.family.global_allocations()
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.family.local_allocations()
    }
    pub fn launch_resources(&self) -> &[seismic_ir::execution::LaunchResources] {
        self.family.launch_resources()
    }

    pub fn decisions(&self) -> Vec<(DecisionId, &'static str)> {
        self.family
            .choices()
            .iter()
            .map(|choice| (choice.decision(), choice.meaning()))
            .collect()
    }
    pub fn hard_constraints(&self) -> BoolExpr {
        self.hard_constraints
    }
    pub fn numerical_applicability(&self) -> &NumericalApplicability {
        self.family.numerical_applicability()
    }
    pub fn provenance(&self) -> &ImplementationProvenance {
        self.family.provenance()
    }
    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        self.family.result_publications()
    }
}

pub(crate) fn validate_universal_implementation<B: seismic_native_target::TargetFamily>(
    implementation: &Implementation<B>,
    arena: &mut ExprArena,
    target_domain: BoolExpr,
    constants: &TargetConstants,
) -> Result<(), crate::errors::PreparationError> {
    let mut fixed = implementation.assignment().clone();
    for (symbol, value) in constants.bindings() {
        fixed.bind(*symbol, value.clone());
    }
    let mut close = |expression: BoolExpr| {
        let expression = arena.partial(expression, &fixed);
        if arena.free_symbols(AnyExpr::Bool(expression)).is_empty() {
            if let Ok(value) = arena.eval_bool(expression, &seismic_lang::expr::Assignment::new()) {
                return arena.bool(value);
            }
        }
        expression
    };
    let target_domain = close(target_domain);
    let semantic_coverage = close(implementation.semantic_coverage().node());
    let hard_constraints = close(implementation.hard_constraints());
    let coverage = arena.all(&[semantic_coverage, hard_constraints]);
    let total = arena.implies(target_domain, coverage);
    if !matches!(arena.view(AnyExpr::Bool(total)), NodeView::BoolConst(true)) {
        let symbol_summary = |expression: BoolExpr| {
            arena
                .free_symbols(AnyExpr::Bool(expression))
                .into_iter()
                .map(|symbol| format!("{symbol:?}:{:?}", arena.symbol_kind(symbol)))
                .collect::<Vec<_>>()
                .join(",")
        };
        return Err(crate::errors::PreparationError::UniversalClosure(format!(
                "post-native legality is not constructionally total over TargetDomain (target={:?} target_symbols=[{}], semantic={:?}, hard={:?} hard_detail={} hard_symbols=[{}], implication={:?})",
                arena.view(AnyExpr::Bool(target_domain)),
                symbol_summary(target_domain),
                arena.view(AnyExpr::Bool(semantic_coverage)),
                arena.view(AnyExpr::Bool(hard_constraints)),
                expression_detail(arena, AnyExpr::Bool(hard_constraints), 8),
                symbol_summary(hard_constraints),
                arena.view(AnyExpr::Bool(total)),
            )));
    }
    Ok(())
}

/// The site an implementation is constructed for.
#[derive(Clone, Copy)]
pub(crate) enum CallSite<'a> {
    Root,
    Spliced {
        /// The call node in the parent function.
        call: NodeId,
        /// Parent values bound to the callee's parameters, in order.
        arguments: &'a [ValueBinding],
        bindings: &'a crate::portable::BindingArena,
        contents: &'a crate::portable::initialization::StorageContents,
        binders: &'a [seismic_lang::expr::SymbolId],
        storage: &'a seismic_ir::storage::TopologyBuilder,
        selections: &'a crate::portable::BindingSelections,
    },
}

impl fmt::Debug for CallSite<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => formatter.write_str("Root"),
            Self::Spliced {
                call, arguments, ..
            } => formatter
                .debug_struct("Spliced")
                .field("call", call)
                .field("arguments", arguments)
                .finish_non_exhaustive(),
        }
    }
}

/// A scalar's actual realization, shared by lowering and call binding.
/// Index expressions never require a fabricated backing symbol. Published
/// values derive their type and symbol from the one owning schedule slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarBinding {
    Value {
        symbol: seismic_lang::expr::SymbolId,
        dtype: seismic_lang::types::DType,
    },
    /// Immutable mathematical integer expression; no reached slot is needed.
    Integer(seismic_lang::expr::IntExpr),
    Index(seismic_lang::expr::NatExpr),
    Quantity(HostQuantitySlot),
    Published(AnyScalarSlot),
}

/// A value in the construction's common binding owner. A physical adapter is
/// derived from this handle only when a consumer needs its current residence.
pub use crate::portable::BindingId as ValueBinding;

/// The physical adapter derived from a construction binding.
#[derive(Clone, Debug)]
pub enum PhysicalBinding {
    /// A global view of the parent (argument, result, or arena).
    View {
        view: AnyBufferView,
        layout: seismic_ir::storage::BufferViewLayout,
    },
    /// A scalar slot or symbol of the parent.
    Scalar(ScalarBinding),
    Range {
        start: ScalarBinding,
        end: ScalarBinding,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ScalarPublication {
    Scalar(AnyScalarSlot),
    Quantity(HostQuantitySlot),
    Range {
        start: HostQuantitySlot,
        end: HostQuantitySlot,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PortableSourceStatus {
    pub(crate) view: AnyBufferView,
    pub(crate) slot: AnyScalarSlot,
    pub(crate) index: u64,
}

/// Contract of a semantic function. Result value IDs name actual producers;
/// at the root, result types describe the checked public declaration. Nested
/// functions derive their result types from their semantic values.
#[derive(Clone, Debug)]
pub struct FunctionContract {
    parameters: Vec<ContractParameter>,
    results: Vec<ContractResult>,
    effects: ContractEffects,
}

#[derive(Clone, Debug, Default)]
pub struct ContractEffects {
    pub reads: Vec<SemanticValueId>,
    pub writes: Vec<SemanticValueId>,
}

#[derive(Clone, Debug)]
pub struct ContractParameter {
    pub value: SemanticValueId,
    pub access: ParameterAccess,
    pub ty: SemanticType,
}

#[derive(Clone, Debug)]
pub struct ContractResult {
    pub value: SemanticValueId,
    /// Declared public type at root; semantic result type for a nested call.
    pub ty: SemanticType,
    pub paths: Vec<Vec<u32>>,
}

impl FunctionContract {
    pub fn parameters(&self) -> &[ContractParameter] {
        &self.parameters
    }
    pub fn results(&self) -> &[ContractResult] {
        &self.results
    }
    pub fn effects(&self) -> &ContractEffects {
        &self.effects
    }

    pub(crate) fn derive(function: &SemanticFunction) -> Self {
        let parameters = function
            .parameters()
            .iter()
            .map(|parameter| ContractParameter {
                value: parameter.value,
                access: parameter.access,
                ty: function.value(parameter.value).ty.clone(),
            })
            .collect();
        let results = function
            .results()
            .iter()
            .enumerate()
            .map(|(ordinal, value)| ContractResult {
                value: *value,
                ty: function.value(*value).ty.clone(),
                paths: vec![vec![ordinal as u32]],
            })
            .collect();
        let mut effects = ContractEffects::default();
        fn visit(function: &SemanticFunction, region: RegionId, effects: &mut ContractEffects) {
            for (_, node) in function.nodes(region) {
                for event in node.events() {
                    let Some(value) = event.place() else { continue };
                    let collection = match event.kind() {
                        SemanticEventKind::Read => &mut effects.reads,
                        SemanticEventKind::Write(_) | SemanticEventKind::AtomicRmw { .. } => {
                            &mut effects.writes
                        }
                        SemanticEventKind::Barrier(_) | SemanticEventKind::MayFail => continue,
                    };
                    if !collection.contains(&value) {
                        collection.push(value);
                    }
                }
                match node.view() {
                    SemanticNodeView::If {
                        then, otherwise, ..
                    } => {
                        visit(function, then, effects);
                        visit(function, otherwise, effects);
                    }
                    SemanticNodeView::Loop { body, .. } => visit(function, body, effects),
                    _ => {}
                }
            }
        }
        visit(function, function.root(), &mut effects);
        Self {
            parameters,
            results,
            effects,
        }
    }
}

/// A lexical schedule branch while its source bodies are being constructed.
/// Its physical environment is restored at each arm boundary by this owner.
pub(crate) struct ScheduleBranch {
    parent: u32,
    first_region: u32,
    remaining: std::vec::IntoIter<(i64, u32)>,
    inherited: internals::PhysicalEnvironment,
}

/// The existing construction owns the complete loop transport and its lexical region.
pub(crate) struct ScheduleRepeat {
    parent: u32,
    product: seismic_ir::construction::RepeatConstruction,
}
impl ScheduleRepeat {
    pub(crate) fn binding(&self) -> seismic_ir::schedule::LoopBinding {
        self.product.binding()
    }
    pub(crate) fn header(
        &self,
    ) -> &seismic_ir::region::Product<seismic_ir::region::ValueDestination> {
        self.product.header()
    }
}

/// The one way to build an implementation. Created while closing a candidate
/// domain.
pub(crate) struct ImplementationBuilder<'a, B: seismic_native_target::TargetFamily> {
    inner: internals::Builder<'a, B>,
}

/// A temporary borrow of the immutable construction inputs and the domain's
/// expression owner. Suspended physical state contains none of these borrows.
pub(crate) struct ConstructionContext<'a, B: seismic_native_target::TargetFamily> {
    arena: &'a mut ExprArena,
    program: &'a SemanticProgram,
    target: &'a DeviceDescription<B>,
    registry: &'a CompilerRegistry<B>,
    constants: &'a TargetConstants,
    precision: &'a PrecisionPolicy,
}

impl<'a, B: seismic_native_target::TargetFamily> ConstructionContext<'a, B> {
    /// Resume only against the checked entry retained by the candidate domain.
    pub(crate) fn for_resume(
        entry: crate::candidate_domain::SourceEntryBorrow<'a>,
        target: &'a DeviceDescription<B>,
        registry: &'a CompilerRegistry<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
    ) -> Self {
        let (arena, program, _) = entry.into_parts();
        Self {
            arena,
            program,
            target,
            registry,
            constants,
            precision,
        }
    }

    pub(crate) fn program(&self) -> &'a SemanticProgram {
        self.program
    }
}

/// The candidate domain can start source construction, but cannot acquire an
/// unrestricted physical builder. Only the source cursor owns that builder.
pub(crate) fn begin_source_construction<'a, B: seismic_native_target::TargetFamily>(
    entry: crate::candidate_domain::SourceEntryBorrow<'a>,
    selection: crate::candidate_domain::BodySelection,
    target: &'a DeviceDescription<B>,
    registry: &'a CompilerRegistry<B>,
    constants: &'a TargetConstants,
    precision: &'a PrecisionPolicy,
) -> (
    crate::portable::construction::SourceConstruction<B>,
    ConstructionContext<'a, B>,
) {
    let (arena, program, root_schema) = entry.into_parts();
    let candidate = program
        .family(program.root())
        .candidates()
        .iter()
        .find(|candidate| {
            let function = program.function(candidate.function);
            function.stable() == selection.body
                && function.source_definition() == selection.source_definition
                && program.subject() == &selection.subject
        })
        .expect("root source selection names a checked body");
    let mode = selection.mode();
    match candidate.kind {
        CandidateKind::Portable => assert!(
            mode != crate::portable::SemanticMode::AuthoredBackend,
            "portable body requires a source-owned mapping"
        ),
        CandidateKind::Lowering { backend } | CandidateKind::Helper { backend } => {
            assert_eq!(
                backend,
                B::NAME,
                "authored body targets a different backend"
            );
            assert_eq!(
                mode,
                crate::portable::SemanticMode::AuthoredBackend,
                "authored body requires its checked backend mapping"
            );
        }
    }
    assert!(
        candidate
            .requires
            .iter()
            .all(|capability| target.supports_capability(*capability)),
        "selected body requires an unavailable target capability"
    );
    let builder = ImplementationBuilder::new(
        arena,
        program,
        program.function(candidate.function),
        target,
        registry,
        constants,
        precision,
        candidate.applicability,
        CallSite::Root,
        Some(root_schema),
    );
    crate::portable::construction::SourceConstruction::begin(builder, mode)
}

impl<'a, B: seismic_native_target::TargetFamily> ImplementationBuilder<'a, B> {
    pub(crate) fn suspend(self) -> BuilderState<B> {
        self.inner.state
    }

    pub(crate) fn into_parts(self) -> (BuilderState<B>, ConstructionContext<'a, B>) {
        self.inner.into_parts()
    }

    pub(crate) fn resume(
        context: &'a mut ConstructionContext<'_, B>,
        state: BuilderState<B>,
    ) -> Self {
        Self {
            inner: internals::Builder::resume(
                context.arena,
                context.program,
                context.target,
                context.registry,
                context.constants,
                context.precision,
                state,
            ),
        }
    }

    fn new(
        arena: &'a mut ExprArena,
        program: &'a SemanticProgram,
        function: &'a SemanticFunction,
        target: &'a DeviceDescription<B>,
        registry: &'a CompilerRegistry<B>,
        constants: &'a TargetConstants,
        precision: &'a PrecisionPolicy,
        semantic_coverage: TargetPredicate,
        site: CallSite<'_>,
        root_schema: Option<&CallSchema>,
    ) -> Self {
        Self {
            inner: internals::Builder::new(
                arena,
                program,
                function,
                target,
                registry,
                constants,
                precision,
                semantic_coverage,
                site,
                root_schema,
            ),
        }
    }
    pub fn arena(&mut self) -> &mut ExprArena {
        self.inner.arena()
    }
    pub(crate) fn bindings(&self) -> &crate::portable::BindingArena {
        &self.inner.state.bindings
    }
    pub(crate) fn bindings_mut(&mut self) -> &mut crate::portable::BindingArena {
        &mut self.inner.state.bindings
    }
    pub(crate) fn expressions_and_bindings(
        &mut self,
    ) -> (&mut ExprArena, &crate::portable::BindingArena) {
        (self.inner.arena, &self.inner.state.bindings)
    }
    pub fn target(&self) -> &DeviceDescription<B> {
        self.inner.target()
    }
    pub(crate) fn portable_target_ref(&self) -> &'a DeviceDescription<B> {
        self.inner.target
    }
    pub(crate) fn portable_registry_ref(&self) -> &'a CompilerRegistry<B> {
        self.inner.registry
    }
    /// Resolve a call whose values remain within one participant segment.
    /// Portable families use their sealed reference body; backend-only helpers
    /// require a unique capability-supported body. Neither acquires a schedule
    /// ABI for participant-local values. Alternative bodies remain explicit
    /// implementation choices at ordinary call-splicing boundaries.
    pub(crate) fn segment_callee(
        &self,
        family: FamilyId,
        backend: BackendName,
    ) -> &'a SemanticFunction {
        let semantic_family = self.inner.program.family(family);
        if semantic_family
            .candidates()
            .iter()
            .any(|candidate| candidate.kind == CandidateKind::Portable)
        {
            let reference = semantic_family.reference().candidate();
            assert!(
                matches!(
                    self.inner
                        .arena
                        .view(AnyExpr::Bool(reference.applicability.node())),
                    NodeView::BoolConst(true)
                ),
                "participant-local portable call lacks checked applicability"
            );
            return self.inner.program.function(reference.function);
        }
        let mut matching = self
            .inner
            .program
            .family(family)
            .candidates()
            .iter()
            .filter(|candidate| {
                candidate.kind == CandidateKind::Helper { backend }
                    && candidate
                        .requires
                        .iter()
                        .all(|capability| self.inner.target.supports_capability(*capability))
                    && matches!(
                        self.inner
                            .arena
                            .view(AnyExpr::Bool(candidate.applicability.node())),
                        NodeView::BoolConst(true)
                    )
            });
        let candidate = matching.next().unwrap_or_else(|| {
            panic!(
                "checked authored helper `{}` has no unconditionally applicable {backend:?} body",
                self.inner.program.family(family).name()
            )
        });
        assert!(
            matching.next().is_none(),
            "checked authored helper resolution is ambiguous"
        );
        self.inner.program.function(candidate.function)
    }
    pub(crate) fn call_can_write(&self, family: FamilyId) -> bool {
        let reference = self.inner.program.family(family).reference().candidate();
        self.inner
            .program
            .function(reference.function)
            .parameters()
            .iter()
            .any(|parameter| {
                matches!(
                    parameter.access,
                    ParameterAccess::Mutable | ParameterAccess::Owned
                )
            })
    }

    /// Declares a finite decision. The domain must be non-empty.
    pub fn decision(&mut self, name: &'static str, domain: FiniteDomain) -> DecisionId {
        self.inner.decision(name, domain)
    }

    /// Adds a hard constraint (target limits are added by the builder from
    /// the topology; factories add only structural legality).
    pub fn constrain(&mut self, predicate: BoolExpr) {
        self.inner.constrain(predicate)
    }

    // ----- storage -------------------------------------------------------------

    // ----- kernels and schedule ----------------------------------------------

    pub fn schedule(&mut self) -> ScheduleBuilder<'_, B> {
        self.inner.schedule()
    }

    pub(crate) fn portable_kernel(
        &mut self,
    ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
        self.inner.portable_kernel()
    }
    pub(crate) fn resume_portable_kernel(
        &mut self,
        cursor: seismic_ir::kernel::dynamic::PortableCursor,
    ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
        self.inner.resume_portable_kernel(cursor)
    }
    pub(crate) fn portable_kernel_with_bindings(
        &mut self,
    ) -> (
        seismic_ir::kernel::dynamic::PortableBuilder<'_, B>,
        &crate::portable::BindingArena,
    ) {
        self.inner.portable_kernel_with_bindings()
    }
    pub(crate) fn portable_is_root(&self) -> bool {
        self.inner.state.root
    }

    /// The checked function reference has the builder's construction
    /// lifetime rather than the borrow of `self`, so a portable lowerer may
    /// inspect it while mutating the builder without an aliasing workaround.
    pub(crate) fn portable_program_ref(&self) -> &'a SemanticProgram {
        self.inner.program
    }
    pub(crate) fn portable_function_ref(&self) -> &'a SemanticFunction {
        self.inner.function
    }

    pub(crate) fn portable_allocate_tensor_axes(
        &mut self,
        value: SemanticValueId,
        axes: Vec<NatExpr>,
    ) -> AnyBufferView {
        self.inner.portable_allocate_tensor_axes(value, axes)
    }
    pub(crate) fn portable_relocate_tensor(
        &mut self,
        source: &crate::portable::initialization::StoredView,
        destination: AnyBufferView,
    ) {
        self.inner.portable_relocate_tensor(source, destination)
    }
    pub(crate) fn portable_publish_tensor(&mut self, value: SemanticValueId, view: AnyBufferView) {
        self.inner.portable_publish_tensor(value, view)
    }
    pub(crate) fn portable_views_may_overlap(
        &self,
        left: AnyBufferView,
        right: AnyBufferView,
    ) -> bool {
        self.inner.portable_views_may_overlap(left, right)
    }
    pub(crate) fn allocate_tensor_product(
        &mut self,
        tensor: &seismic_lang::entry::TensorSemantics,
    ) -> AnyBufferView {
        self.inner.allocate_tensor_product(tensor)
    }
    pub(crate) fn portable_binding(&self, value: SemanticValueId) -> PhysicalBinding {
        self.inner.portable_binding(value)
    }
    pub(crate) fn portable_publish(&mut self, value: SemanticValueId) -> ScalarPublication {
        self.inner.portable_publish(value)
    }
    pub(crate) fn portable_affine_view(
        &mut self,
        view: &crate::portable::initialization::StoredView,
    ) -> Option<AnyBufferView> {
        self.inner.portable_affine_view(view)
    }

    pub(crate) fn portable_layout(
        &self,
        view: AnyBufferView,
    ) -> seismic_ir::storage::BufferViewLayout {
        self.inner.portable_layout(view)
    }
    pub(crate) fn portable_source_statuses(&mut self, count: usize) -> Vec<PortableSourceStatus> {
        self.inner.portable_source_statuses(count)
    }
    pub(crate) fn portable_finish_source_check(
        &mut self,
        status: PortableSourceStatus,
        site: seismic_ir::kernel::ops::CheckSite,
    ) {
        self.inner.portable_finish_source_check(status, site)
    }
    pub(crate) fn begin_source_selection(
        &mut self,
        selector: crate::portable::BindingSelector,
    ) -> (i64, ScheduleBranch) {
        let parent = self.inner.state.schedule_region;
        let inherited = self.inner.physical_environment();
        let (then, otherwise) = self.inner.portable_begin_branch(selector);
        let mut regions = vec![(1, then), (0, otherwise)].into_iter();
        let (selected, region) = regions.next().expect("a selection has at least one arm");
        self.inner.state.schedule_region = region;
        (
            selected,
            ScheduleBranch {
                parent,
                first_region: then,
                remaining: regions,
                inherited,
            },
        )
    }

    pub(crate) fn begin_source_branch(&mut self, condition: BoolExpr) -> ScheduleBranch {
        self.begin_source_selection(condition).1
    }

    pub(crate) fn next_source_branch(&mut self, branch: &mut ScheduleBranch) -> Option<i64> {
        let (selected, region) = branch.remaining.next()?;
        self.inner
            .restore_physical_environment(branch.inherited.clone());
        self.inner.state.schedule_region = region;
        Some(selected)
    }

    pub(crate) fn finish_source_branch(&mut self, branch: ScheduleBranch) {
        self.inner.state.schedule_region = branch.parent;
        self.inner.restore_physical_environment(branch.inherited);
    }

    pub(crate) fn finish_value_branch(
        &mut self,
        branch: ScheduleBranch,
        then_values: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
        else_values: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
    ) -> seismic_ir::region::Product<seismic_ir::region::ValueDestination> {
        let destination = self.inner.finish_value_branch(
            branch.parent,
            branch.first_region,
            then_values,
            else_values,
        );
        self.finish_source_branch(branch);
        destination
    }

    pub(crate) fn begin_value_repeat(
        &mut self,
        start: NatExpr,
        end: NatExpr,
        visits: seismic_ir::schedule::RepeatVisits,
        initial: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
    ) -> ScheduleRepeat {
        self.inner.begin_value_repeat(start, end, visits, initial)
    }
    pub(crate) fn finish_value_repeat(
        &mut self,
        repeat: ScheduleRepeat,
        backedge: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
    ) -> seismic_ir::region::Product<seismic_ir::region::ValueDestination> {
        self.inner.finish_value_repeat(repeat, backedge)
    }
    /// Closes the source construction's own schedule and implementation in
    /// one consuming operation. No caller supplies an execution body.
    pub(crate) fn close(mut self, mode: crate::portable::SemanticMode) -> ConstructedCandidate<B> {
        let schedule = self.inner.schedule().close();
        self.inner.close(schedule, mode)
    }
}

mod internals {
    use super::*;
    use seismic_ir::schedule::ScheduleStep;
    use seismic_ir::storage::GlobalBufferKind;
    use seismic_lang::expr::{AnyExpr, CmpOp, RootName, SymbolSort};
    use seismic_lang::registry;
    use std::collections::HashMap;

    /// Each authored body owns distinct semantic values. The checked source
    /// parameter coordinates, not those local IDs, bind its common entry ABI.
    fn entry_parameter<'a>(
        schema: &'a CallSchema,
        function: &SemanticFunction,
        parameter: &ContractParameter,
    ) -> &'a seismic_lang::entry::Parameter {
        let formal = function
            .parameters()
            .iter()
            .find(|formal| formal.value == parameter.value)
            .expect("contract parameter belongs to the selected body");
        let seismic_lang::entry::FunctionParameterOrigin::Source { ordinal, path } = &formal.origin
        else {
            panic!("public entry schema contains a captured helper dimension")
        };
        let actual = schema
            .parameters()
            .iter()
            .find(|actual| actual.source == *ordinal && actual.path == *path)
            .expect("checked body parameter has its entry ABI coordinate");
        use seismic_lang::entry::TensorAccess;
        match (&parameter.ty, &actual.kind) {
            (
                SemanticType::Tensor(tensor),
                ParameterKind::Tensor {
                    access,
                    representation,
                    axes,
                },
            ) => {
                assert_eq!(tensor.representation, *representation);
                assert_eq!(&tensor.axes, axes);
                assert!(matches!(
                    (parameter.access, access),
                    (ParameterAccess::Owned, TensorAccess::Owned)
                        | (ParameterAccess::Shared, TensorAccess::Shared)
                        | (ParameterAccess::Mutable, TensorAccess::Mutable)
                ));
            }
            (SemanticType::Scalar(expected), ParameterKind::Scalar { dtype, .. }) => {
                assert_eq!(expected, dtype)
            }
            (SemanticType::Index { bound: expected }, ParameterKind::Index { bound, .. })
            | (SemanticType::Range { bound: expected }, ParameterKind::Range { bound, .. }) => {
                assert_eq!(expected, bound)
            }
            _ => panic!("checked body parameter differs from the entry ABI type"),
        }
        actual
    }

    /// Private move slot for the open construction. Builder closure consumes
    /// the IR owner before the remaining compiler metadata has been finalized;
    /// deref keeps the open-phase implementation uncluttered without exposing
    /// an optional lifecycle in the public IR API.
    pub(super) struct OpenConstruction<B: seismic_native_target::TargetFamily>(
        Option<Construction<B>>,
    );
    impl<B: seismic_native_target::TargetFamily> OpenConstruction<B> {
        fn new(construction: Construction<B>) -> Self {
            Self(Some(construction))
        }
        fn take(&mut self) -> Construction<B> {
            self.0
                .take()
                .expect("builder construction is consumed exactly once at close")
        }
    }
    impl<B: seismic_native_target::TargetFamily> std::ops::Deref for OpenConstruction<B> {
        type Target = Construction<B>;
        fn deref(&self) -> &Self::Target {
            self.0
                .as_ref()
                .expect("builder construction is available before close")
        }
    }
    impl<B: seismic_native_target::TargetFamily> std::ops::DerefMut for OpenConstruction<B> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.0
                .as_mut()
                .expect("builder construction is available before close")
        }
    }

    struct CaptureBindings<'a, B: seismic_native_target::TargetFamily> {
        construction: &'a mut Construction<B>,
        arena: &'a mut ExprArena,
        storage: &'a seismic_ir::storage::TopologyBuilder,
        parameter: Option<SemanticValueId>,
        views: HashMap<AnyBufferView, AnyBufferView>,
        slots: HashMap<AnyScalarSlot, AnyScalarSlot>,
        quantities: HashMap<HostQuantitySlot, HostQuantitySlot>,
    }
    impl<B: seismic_native_target::TargetFamily> crate::portable::BindingPhysicalImport
        for CaptureBindings<'_, B>
    {
        fn remap_tensor(
            &mut self,
            value: &crate::portable::initialization::StoredTensor,
            _: &crate::portable::BindingPath,
        ) -> crate::portable::initialization::StoredTensor {
            let view = value
                .view
                .map(|view| self.remap_view(*view), |value| *value);
            crate::portable::initialization::StoredTensor {
                view,
                ..value.clone()
            }
        }
        fn remap_slot(
            &mut self,
            source: AnyScalarSlot,
            _: &crate::portable::BindingPath,
        ) -> AnyScalarSlot {
            *self
                .slots
                .entry(source)
                .or_insert_with(|| self.construction.schedule_state().capture_slot(source))
        }
        fn remap_quantity_slot(
            &mut self,
            source: HostQuantitySlot,
            _: &crate::portable::BindingPath,
        ) -> HostQuantitySlot {
            *self.quantities.entry(source).or_insert_with(|| {
                self.construction
                    .schedule_state()
                    .capture_quantity_slot(source)
            })
        }
        fn remap_prepared(
            &mut self,
            value: crate::portable::PreparedArg,
            _: &crate::portable::BindingPath,
        ) -> crate::portable::PreparedArg {
            value
        }
        fn remap_axis(&mut self, value: NatExpr, _: &crate::portable::BindingPath) -> NatExpr {
            value
        }
        fn remap_selector(
            &mut self,
            value: crate::portable::BindingSelector,
            _: &crate::portable::BindingPath,
        ) -> crate::portable::BindingSelector {
            value
        }
    }
    impl<B: seismic_native_target::TargetFamily> CaptureBindings<'_, B> {
        fn remap_view(&mut self, source: AnyBufferView) -> AnyBufferView {
            if let Some(view) = self.views.get(&source) {
                return *view;
            }
            let view = self.construction.storage_mut().capture_view(
                self.arena,
                self.storage,
                source,
                self.parameter.expect("captured storage has a parameter"),
            );
            self.views.insert(source, view);
            view
        }
    }

    #[derive(Clone)]
    pub(super) struct PhysicalEnvironment {
        value_views: HashMap<SemanticValueId, AnyBufferView>,
        scalar_symbols: HashMap<SemanticValueId, PhysicalBinding>,
        result_slots: HashMap<SemanticValueId, ScalarPublication>,
    }

    /// All mutable physical construction state. No borrowed arena, source body
    /// or target context survives when construction is suspended.
    pub(crate) struct BuilderState<B: seismic_native_target::TargetFamily> {
        function: seismic_lang::ids::FunctionId,
        pub(super) construction: OpenConstruction<B>,
        contract: FunctionContract,
        semantic_coverage: TargetPredicate,
        pub(super) bindings: crate::portable::BindingArena,
        pub(super) root: bool,
        physical: PhysicalEnvironment,
        pending_result_paths: HashMap<SemanticValueId, Vec<(Vec<u32>, Vec<NatExpr>)>>,
        pub(super) schedule_region: u32,
        pub(super) choices: Vec<ChoiceDeclaration>,
        pub(super) constraints: Vec<BoolExpr>,
        pub(super) numerical_children: NumericalApplicability,
        pub(super) callees: Vec<StableFunctionId>,
        pub(super) call_occurrences: HashMap<NodeId, u32>,
    }

    impl<B: seismic_native_target::TargetFamily> BuilderState<B> {
        pub(crate) fn function(&self) -> seismic_lang::ids::FunctionId {
            self.function
        }
    }

    pub(super) struct Builder<'a, B: seismic_native_target::TargetFamily> {
        pub(super) state: BuilderState<B>,
        pub(super) arena: &'a mut ExprArena,
        pub(super) program: &'a SemanticProgram,
        pub(super) function: &'a SemanticFunction,
        pub(super) target: &'a DeviceDescription<B>,
        pub(super) registry: &'a CompilerRegistry<B>,
        pub(super) constants: &'a TargetConstants,
        pub(super) precision: &'a PrecisionPolicy,
    }

    impl<'a, B: seismic_native_target::TargetFamily> Builder<'a, B> {
        pub(super) fn into_parts(self) -> (BuilderState<B>, ConstructionContext<'a, B>) {
            (
                self.state,
                ConstructionContext {
                    arena: self.arena,
                    program: self.program,
                    target: self.target,
                    registry: self.registry,
                    constants: self.constants,
                    precision: self.precision,
                },
            )
        }

        pub(super) fn resume(
            arena: &'a mut ExprArena,
            program: &'a SemanticProgram,
            target: &'a DeviceDescription<B>,
            registry: &'a CompilerRegistry<B>,
            constants: &'a TargetConstants,
            precision: &'a PrecisionPolicy,
            state: BuilderState<B>,
        ) -> Self {
            let function = program.function(state.function);
            Self {
                state,
                arena,
                program,
                function,
                target,
                registry,
                constants,
                precision,
            }
        }

        pub(super) fn new(
            arena: &'a mut ExprArena,
            program: &'a SemanticProgram,
            function: &'a SemanticFunction,
            target: &'a DeviceDescription<B>,
            registry: &'a CompilerRegistry<B>,
            constants: &'a TargetConstants,
            precision: &'a PrecisionPolicy,
            semantic_coverage: TargetPredicate,
            site: CallSite<'_>,
            root_schema: Option<&CallSchema>,
        ) -> Self {
            let mut contract = FunctionContract::derive(function);
            if let Some(schema) = root_schema {
                assert_eq!(
                    contract.parameters.len(),
                    schema.parameters().len(),
                    "checked root parameter arity"
                );
                assert_eq!(
                    contract.results.len(),
                    schema.results().len(),
                    "checked root result arity"
                );
                for (result, leaf) in contract.results.iter_mut().zip(schema.results()) {
                    use seismic_lang::entry::ResultKind;
                    match (&result.ty, &leaf.kind) {
                        (
                            SemanticType::Tensor(tensor),
                            ResultKind::Tensor {
                                representation,
                                axes,
                            },
                        ) => {
                            assert_eq!(tensor.representation, *representation);
                            assert_eq!(tensor.axes.len(), axes.len());
                        }
                        (SemanticType::Scalar(expected), ResultKind::Scalar(actual)) => {
                            assert_eq!(expected, actual)
                        }
                        (SemanticType::Index { .. }, ResultKind::Index { .. })
                        | (SemanticType::Range { .. }, ResultKind::Range { .. }) => {}
                        _ => panic!("checked body result differs from the entry ABI type"),
                    }
                    match (&mut result.ty, &leaf.kind) {
                        (SemanticType::Tensor(tensor), ResultKind::Tensor { axes, .. }) => {
                            tensor.axes.clone_from(axes)
                        }
                        (SemanticType::Index { bound: actual }, ResultKind::Index { bound })
                        | (SemanticType::Range { bound: actual }, ResultKind::Range { bound }) => {
                            *actual = *bound
                        }
                        _ => {}
                    }
                    result.paths = vec![leaf.path.clone()];
                }
            }
            let disjoint = root_schema
                .map(|schema| {
                    schema
                        .aliases()
                        .iter()
                        .filter_map(|rule| match rule {
                            AliasRule::Disjoint(a, b) => Some((*a, *b)),
                            AliasRule::MayOverlap(_, _) => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut construction = Construction::new(
                arena,
                disjoint,
                matches!(site, CallSite::Spliced { .. }),
                target.addressable_resources().len(),
            );
            let mut value_views = HashMap::new();
            let mut scalar_symbols = HashMap::new();
            let mut pending_result_paths: HashMap<SemanticValueId, Vec<(Vec<u32>, Vec<NatExpr>)>> =
                HashMap::new();
            for parameter in &contract.parameters {
                if matches!(site, CallSite::Spliced { .. }) {
                    continue;
                }
                match &parameter.ty {
                    SemanticType::Tensor(tensor) => {
                        let abi = root_schema
                            .map(|schema| entry_parameter(schema, function, parameter).id);
                        let kind = GlobalBufferKind::Argument {
                            value: parameter.value,
                            abi,
                        };
                        let bytes = seismic_ir::storage::tensor_bytes(
                            arena,
                            tensor.representation,
                            &tensor.axes,
                        );
                        let allocation = construction.storage_mut().allocate(
                            kind,
                            bytes,
                            seismic_ir::storage::representation_alignment(tensor.representation),
                        );
                        let zero = arena.nat(0);
                        let view = construction.storage_mut().dense_view(
                            arena,
                            allocation,
                            tensor.representation,
                            zero,
                            tensor.axes.clone(),
                        );
                        let source = construction.view(view, tensor.representation);
                        let actual = construction.bind_argument_tensor(arena, source);
                        value_views.insert(parameter.value, actual);
                    }
                    SemanticType::Scalar(_) | SemanticType::Index { .. } => {
                        let symbol = match site {
                            CallSite::Root => {
                                let schema =
                                    root_schema.expect("root builder requires its CallSchema");
                                let p = entry_parameter(schema, function, parameter);
                                match &p.kind {
                                    ParameterKind::Scalar { symbol, dtype } => {
                                        ScalarBinding::Value {
                                            symbol: *symbol,
                                            dtype: *dtype,
                                        }
                                    }
                                    ParameterKind::Index { symbol, .. } => {
                                        ScalarBinding::Index(arena.nat_symbol(*symbol))
                                    }
                                    _ => panic!(
                                        "scalar semantic parameter has a non-scalar ABI kind"
                                    ),
                                }
                            }
                            CallSite::Spliced { .. } => {
                                unreachable!("complete call bindings are imported below")
                            }
                        };
                        scalar_symbols.insert(parameter.value, PhysicalBinding::Scalar(symbol));
                    }
                    SemanticType::Range { .. } => {
                        let binding = match site {
                            CallSite::Root => {
                                let schema =
                                    root_schema.expect("root builder requires its CallSchema");
                                let parameter = entry_parameter(schema, function, parameter);
                                match &parameter.kind {
                                    ParameterKind::Range { start, end, .. } => {
                                        PhysicalBinding::Range {
                                            start: ScalarBinding::Index(arena.nat_symbol(*start)),
                                            end: ScalarBinding::Index(arena.nat_symbol(*end)),
                                        }
                                    }
                                    _ => {
                                        panic!("range semantic parameter has a non-range ABI kind")
                                    }
                                }
                            }
                            CallSite::Spliced { .. } => {
                                unreachable!("complete call bindings are imported below")
                            }
                        };
                        scalar_symbols.insert(parameter.value, binding);
                    }
                    SemanticType::Integer => {
                        panic!("root parameter ABI does not expose unbounded signed Integer")
                    }
                    SemanticType::Tuple(_) | SemanticType::Opaque { .. } | SemanticType::Void => {}
                }
            }
            let mut bindings = crate::portable::BindingArena::default();
            if let CallSite::Spliced {
                arguments,
                bindings: caller,
                contents,
                binders,
                storage,
                selections,
                ..
            } = site
            {
                let mut capture = CaptureBindings {
                    construction: &mut construction,
                    arena,
                    storage,
                    parameter: contract.parameters.first().map(|parameter| parameter.value),
                    views: HashMap::new(),
                    slots: HashMap::new(),
                    quantities: HashMap::new(),
                };
                bindings.import_parameters(
                    caller,
                    arguments,
                    selections,
                    contents,
                    binders,
                    &mut capture,
                );
                for (parameter, binding) in
                    contract.parameters.iter().zip(bindings.parameter_handles())
                {
                    if let Some(adapter) = bindings.physical_binding(*binding, |view| {
                        construction.storage().view_layout(view).clone()
                    }) {
                        match adapter {
                            PhysicalBinding::View { view, .. } => {
                                value_views.insert(parameter.value, view);
                            }
                            other => {
                                scalar_symbols.insert(parameter.value, other);
                            }
                        }
                    }
                }
            }
            let result_slots = HashMap::new();
            if matches!(site, CallSite::Root) {
                for result in &contract.results {
                    if let SemanticType::Tensor(tensor) = &result.ty {
                        pending_result_paths
                            .entry(result.value)
                            .or_default()
                            .extend(
                                result
                                    .paths
                                    .iter()
                                    .cloned()
                                    .map(|path| (path, tensor.axes.clone())),
                            );
                    }
                }
            }
            Self {
                arena,
                program,
                function,
                target,
                registry,
                constants,
                precision,
                state: BuilderState {
                    function: function.id(),
                    construction: OpenConstruction::new(construction),
                    contract,
                    semantic_coverage,
                    bindings,
                    root: matches!(site, CallSite::Root),
                    physical: PhysicalEnvironment {
                        value_views,
                        scalar_symbols,
                        result_slots,
                    },
                    pending_result_paths,
                    schedule_region: 0,
                    choices: Vec::new(),
                    constraints: Vec::new(),
                    numerical_children: NumericalApplicability::required_source(),
                    callees: Vec::new(),
                    call_occurrences: HashMap::new(),
                },
            }
        }

        pub(super) fn arena(&mut self) -> &mut ExprArena {
            self.arena
        }
        pub(super) fn target(&self) -> &DeviceDescription<B> {
            self.target
        }

        pub(super) fn decision(&mut self, name: &'static str, domain: FiniteDomain) -> DecisionId {
            let decision = self.arena.decision(domain);
            let active_when = self.arena.bool(true);
            self.state.choices.push(ChoiceDeclaration {
                kind: crate::refinement::ChoiceKind::WorkgroupSize,
                decision,
                meaning: name,
                active_when,
            });
            decision
        }
        pub(super) fn constrain(&mut self, predicate: BoolExpr) {
            self.state.constraints.push(predicate)
        }

        pub(super) fn bind_view(&self, view: AnyBufferView) -> PhysicalBinding {
            PhysicalBinding::View {
                view,
                layout: self.state.construction.storage().view_layout(view).clone(),
            }
        }

        pub(super) fn schedule(&mut self) -> ScheduleBuilder<'_, B> {
            self.state
                .construction
                .schedule(self.arena, self.state.schedule_region)
        }
        pub(crate) fn portable_kernel(
            &mut self,
        ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
            self.state.construction.portable_kernel(
                self.arena,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
            )
        }
        pub(super) fn resume_portable_kernel(
            &mut self,
            cursor: seismic_ir::kernel::dynamic::PortableCursor,
        ) -> seismic_ir::kernel::dynamic::PortableBuilder<'_, B> {
            self.state.construction.resume_portable_kernel(
                self.arena,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
                cursor,
            )
        }
        pub(crate) fn portable_kernel_with_bindings(
            &mut self,
        ) -> (
            seismic_ir::kernel::dynamic::PortableBuilder<'_, B>,
            &crate::portable::BindingArena,
        ) {
            let kernel = self.state.construction.portable_kernel(
                self.arena,
                self.target.facts(),
                self.target.addressable_resources(),
                self.target.vectors(),
            );
            (kernel, &self.state.bindings)
        }
        pub(crate) fn portable_allocate_tensor_axes(
            &mut self,
            value: SemanticValueId,
            axes: Vec<NatExpr>,
        ) -> AnyBufferView {
            if let Some(view) = self.state.physical.value_views.get(&value).copied() {
                return view;
            }
            let SemanticType::Tensor(mut tensor) = self.function.value(value).ty.clone() else {
                panic!("tensor allocation has scalar type")
            };
            tensor.axes = axes;
            let view = self.allocate_tensor_product(&tensor);
            self.state.physical.value_views.insert(value, view);
            view
        }
        /// Relocate dense storage without interpreting unspecified elements as
        /// source values. The same logical map used by kernel accesses derives
        /// byte addresses; the reached Copy owns their physical validity.
        pub(crate) fn portable_relocate_tensor(
            &mut self,
            source: &crate::portable::initialization::StoredView,
            destination: AnyBufferView,
        ) {
            use seismic_ir::{region::Product, tensor_view::CoordinateOp};
            let backing = *source.backing();
            let source_layout = self
                .state
                .construction
                .storage()
                .view_layout(backing)
                .clone();
            let destination_layout = self
                .state
                .construction
                .storage()
                .view_layout(destination)
                .clone();
            let seismic_lang::registry::RepresentationKind::Dense(dtype) =
                seismic_lang::registry::representation_info(backing.representation()).kind
            else {
                panic!("raw logical relocation requires dense storage")
            };
            assert_eq!(backing.representation(), destination.representation());
            assert!(destination_layout.contiguous);
            assert_eq!(source.extents(), destination_layout.extents);
            let zero = self.arena.nat(0);
            let count = self.arena.nat_product(source.extents());
            // No coordinate exists for an empty tensor. Avoid constructing
            // division by a statically zero axis even inside an unvisited body.
            if count == zero {
                return;
            }
            let repeat = self.begin_value_repeat(
                zero,
                count,
                seismic_ir::schedule::RepeatVisits::Ordered,
                Product::Unit,
            );
            let linear = repeat.binding().index;
            let mut remaining = linear;
            let mut indices = vec![zero; source.extents().len()];
            for axis in (0..indices.len()).rev() {
                indices[axis] = self.arena.nat_rem(remaining, source.extents()[axis]);
                remaining = self.arena.nat_div(remaining, source.extents()[axis]);
            }
            let indices = source
                .dense_coordinates(&indices, zero, |op, a, b| match op {
                    CoordinateOp::Add => self.arena.nat_add(a, b),
                    CoordinateOp::Mul => self.arena.nat_mul(a, b),
                    CoordinateOp::Div => self.arena.nat_div(a, b),
                    CoordinateOp::Rem => self.arena.nat_rem(a, b),
                })
                .expect("dense relocation cannot contain plane projection");
            let mut source_element = zero;
            for (index, stride) in indices.iter().zip(&source_layout.strides) {
                let offset = self.arena.nat_mul(*index, *stride);
                source_element = self.arena.nat_add(source_element, offset);
            }
            let width = self.arena.nat(u64::from(dtype.bytes()));
            let source_offset = self.arena.nat_mul(source_element, width);
            let destination_offset = self.arena.nat_mul(linear, width);
            let source_index = self.state.construction.storage_mut().subview(
                self.arena,
                backing.index(),
                source_offset,
                vec![],
                vec![],
            );
            let destination_index = self.state.construction.storage_mut().subview(
                self.arena,
                destination.index(),
                destination_offset,
                vec![],
                vec![],
            );
            let source_element = self
                .state
                .construction
                .view(source_index, backing.representation());
            let destination_element = self
                .state
                .construction
                .view(destination_index, destination.representation());
            self.schedule()
                .copy_any(source_element, destination_element);
            self.finish_value_repeat(repeat, Product::Unit);
        }
        pub(crate) fn portable_publish_tensor(
            &mut self,
            value: SemanticValueId,
            view: AnyBufferView,
        ) {
            let paths = self
                .state
                .pending_result_paths
                .get(&value)
                .expect("root tensor publication has no declared result path")
                .clone();
            for (path, declared_axes) in paths {
                self.state
                    .construction
                    .schedule(self.arena, self.state.schedule_region)
                    .publish_tensor(view, path, declared_axes);
            }
        }
        pub(crate) fn portable_views_may_overlap(
            &self,
            left: AnyBufferView,
            right: AnyBufferView,
        ) -> bool {
            self.state.construction.may_overlap_views(left, right)
        }
        pub(crate) fn allocate_tensor_product(
            &mut self,
            tensor: &seismic_lang::entry::TensorSemantics,
        ) -> AnyBufferView {
            let (_, index) = self.state.construction.storage_mut().tensor(
                self.arena,
                GlobalBufferKind::Arena,
                tensor.representation,
                tensor.axes.clone(),
            );
            let view = self.state.construction.view(index, tensor.representation);
            self.state.construction.begin_tensor_instance(
                self.arena,
                self.state.schedule_region,
                view,
            )
        }
        pub(crate) fn portable_binding(&self, value: SemanticValueId) -> PhysicalBinding {
            self.portable_existing_binding(value)
                .unwrap_or_else(|| panic!("semantic value has no portable binding"))
        }
        pub(crate) fn portable_existing_binding(
            &self,
            value: SemanticValueId,
        ) -> Option<PhysicalBinding> {
            if let Some(view) = self.state.physical.value_views.get(&value).copied() {
                return Some(self.bind_view(view));
            }
            if let Some(binding) = self.state.physical.scalar_symbols.get(&value) {
                return Some(binding.clone());
            }
            match self.state.physical.result_slots.get(&value).copied() {
                Some(ScalarPublication::Scalar(slot)) => {
                    Some(PhysicalBinding::Scalar(ScalarBinding::Published(slot)))
                }
                Some(ScalarPublication::Quantity(slot)) => {
                    Some(PhysicalBinding::Scalar(ScalarBinding::Quantity(slot)))
                }
                Some(ScalarPublication::Range { start, end }) => Some(PhysicalBinding::Range {
                    start: ScalarBinding::Quantity(start),
                    end: ScalarBinding::Quantity(end),
                }),
                None => None,
            }
        }
        pub(crate) fn portable_publish(&mut self, value: SemanticValueId) -> ScalarPublication {
            if let Some(publication) = self.state.physical.result_slots.get(&value).copied() {
                return publication;
            }
            let publication = match self.function.value(value).ty {
                SemanticType::Scalar(dtype) => ScalarPublication::Scalar(
                    self.state
                        .construction
                        .schedule_state()
                        .slot_any(self.arena, ScalarKind::Scalar(dtype)),
                ),
                SemanticType::Index { .. } => ScalarPublication::Quantity(
                    self.state
                        .construction
                        .schedule_state()
                        .quantity_slot(self.arena, HostQuantityKind::Natural),
                ),
                SemanticType::Integer => ScalarPublication::Quantity(
                    self.state
                        .construction
                        .schedule_state()
                        .quantity_slot(self.arena, HostQuantityKind::Integer),
                ),
                SemanticType::Range { .. } => ScalarPublication::Range {
                    start: self
                        .state
                        .construction
                        .schedule_state()
                        .quantity_slot(self.arena, HostQuantityKind::Natural),
                    end: self
                        .state
                        .construction
                        .schedule_state()
                        .quantity_slot(self.arena, HostQuantityKind::Natural),
                },
                _ => panic!("portable scalar publication requested for a non-scalar value"),
            };
            self.state.physical.result_slots.insert(value, publication);
            publication
        }
        pub(crate) fn portable_affine_view(
            &mut self,
            view: &crate::portable::initialization::StoredView,
        ) -> Option<AnyBufferView> {
            self.state
                .construction
                .storage_mut()
                .affine_view(self.arena, view)
        }
        pub(crate) fn portable_layout(
            &self,
            view: AnyBufferView,
        ) -> seismic_ir::storage::BufferViewLayout {
            self.state.construction.storage().view_layout(view).clone()
        }
        pub(crate) fn portable_source_statuses(
            &mut self,
            count: usize,
        ) -> Vec<PortableSourceStatus> {
            if count == 0 {
                return Vec::new();
            }
            let count = u64::try_from(count).expect("source check count exceeds address domain");
            let representation = registry::dense(DType::U32);
            let bytes = self.arena.nat(
                count
                    .checked_mul(u64::from(DType::U32.bytes()))
                    .expect("source status byte count overflow"),
            );
            let allocation = self.state.construction.storage_mut().allocate(
                GlobalBufferKind::Arena,
                bytes,
                seismic_ir::storage::representation_alignment(representation),
            );
            let zero = self.arena.nat(0);
            let extent = self.arena.nat(count);
            let index = self.state.construction.storage_mut().dense_view(
                self.arena,
                allocation,
                representation,
                zero,
                vec![extent],
            );
            let view = self.state.construction.view(index, representation);
            let mut schedule = self.schedule();
            schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
            (0..count)
                .map(|index| PortableSourceStatus {
                    view,
                    slot: schedule.slot_any(ScalarKind::Scalar(DType::U32)),
                    index,
                })
                .collect()
        }
        pub(crate) fn portable_finish_source_check(
            &mut self,
            status: PortableSourceStatus,
            site: seismic_ir::kernel::ops::CheckSite,
        ) {
            self.state.construction.assert_view(status.view);
            self.state.construction.assert_slot(status.slot);
            let index = self.arena.nat(status.index);
            let mut schedule = self.schedule();
            schedule.scalar_read_any(status.view, vec![index], status.slot);
            schedule.check_zero_any(status.slot, site);
        }
        pub(super) fn physical_environment(&self) -> PhysicalEnvironment {
            self.state.physical.clone()
        }
        pub(super) fn restore_physical_environment(&mut self, environment: PhysicalEnvironment) {
            self.state.physical = environment;
        }
        pub(crate) fn portable_begin_branch(&mut self, condition: BoolExpr) -> (u32, u32) {
            self.state
                .construction
                .schedule_state()
                .begin_branch(self.state.schedule_region, condition)
        }
        pub(crate) fn finish_value_branch(
            &mut self,
            parent: u32,
            then_region: u32,
            then_values: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
            else_values: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
        ) -> seismic_ir::region::Product<seismic_ir::region::ValueDestination> {
            self.state.construction.finish_value_branch(
                self.arena,
                parent,
                then_region,
                then_values,
                else_values,
            )
        }
        pub(crate) fn begin_value_repeat(
            &mut self,
            start: NatExpr,
            end: NatExpr,
            visits: seismic_ir::schedule::RepeatVisits,
            initial: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
        ) -> ScheduleRepeat {
            let parent = self.state.schedule_region;
            let product = self
                .state
                .construction
                .begin_value_repeat(self.arena, parent, start, end, visits, initial);
            self.state.schedule_region = product.body();
            ScheduleRepeat { parent, product }
        }
        pub(crate) fn finish_value_repeat(
            &mut self,
            repeat: ScheduleRepeat,
            backedge: seismic_ir::region::Product<seismic_ir::region::ValueOperand>,
        ) -> seismic_ir::region::Product<seismic_ir::region::ValueDestination> {
            self.state.schedule_region = repeat.parent;
            self.state
                .construction
                .finish_value_repeat(repeat.product, backedge)
        }
        pub(super) fn close(
            mut self,
            closed: ClosedSchedule,
            mode: crate::portable::SemanticMode,
        ) -> ConstructedCandidate<B> {
            let coverage = self.state.semantic_coverage;
            let allocation_ids = (0..self.state.construction.storage().allocation_count())
                .map(|ordinal| self.state.construction.allocation(ordinal))
                .collect::<Vec<_>>();
            let construction = self.state.construction.take();
            let analyzed = construction
                .close(closed)
                .normalize_launches(
                    self.arena,
                    self.target.limits().max_grid[0],
                    self.target.limits().max_index_bits,
                )
                .expect("one implementation and its imported children share an immutable target")
                .analyze_allocations();
            self.validate_schedule(analyzed.schedule(), analyzed.storage());
            let reuse_policy = crate::refinement::AllocationReusePolicy::Explore;
            let (planned, reuse_decisions, reuse_constraints) =
                crate::refinement::refine_allocation_choices(self.arena, analyzed, reuse_policy)
                    .into_parts();
            for (decision, meaning) in reuse_decisions {
                let active_when = self.arena.bool(true);
                self.state.choices.push(ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::AllocationSlot,
                    decision,
                    meaning,
                    active_when,
                });
            }
            self.state.constraints.extend(reuse_constraints);
            crate::refinement::validate_choice_declarations(self.arena, &self.state.choices);
            let executable = planned.finish().close_execution(
                self.arena,
                self.target.local_realization(),
                self.target.kernel_abi(),
            );
            let kernels = executable
                .kernels()
                .kernels()
                .map(|(_, kernel)| kernel)
                .collect::<Vec<_>>();
            self.derive_constraints(
                executable.storage(),
                &allocation_ids,
                executable.schedule(),
                &kernels,
                executable.launch_resources(),
            );
            let raw_constraints = self.arena.all(&self.state.constraints);
            let side_conditions = self.arena.side_conditions(AnyExpr::Bool(raw_constraints));
            let hard_constraints = self.arena.and(side_conditions, raw_constraints);
            let required = self
                .program
                .families()
                .any(|(_, family)| family.reference().function() == self.function.id());
            let numerical_applicability =
                if required && mode == crate::portable::SemanticMode::Portable {
                    self.state.numerical_children.clone()
                } else {
                    NumericalApplicability::selected_replacement(
                        self.arena,
                        self.program,
                        self.function,
                        &self.state.bindings,
                        &executable,
                        &self.state.numerical_children,
                    )
                };
            let roots = self.register_roots(
                executable.storage(),
                &allocation_ids,
                &kernels,
                executable.schedule(),
                executable.launch_resources(),
                hard_constraints,
                coverage,
            );
            let expression_digest = self.arena.canonical_digest(&roots).bytes();
            let mut digest =
                seismic_ir::identity::StructureDigest::new("seismic-implementation-v7");
            let (reference_math_version, reference_math_digest) =
                seismic_ir::kernel::reference_math_identity();
            digest.bytes(reference_math_version.as_bytes());
            digest.bytes(&reference_math_digest);
            digest.bytes(&expression_digest);
            digest_structure(
                &mut digest,
                &self,
                executable.storage(),
                &allocation_ids,
                &kernels,
                executable.schedule(),
            );
            let identity = ConstructedCandidateIdentity {
                structure: digest.finish(),
            };
            let mut result_publications = Vec::new();
            for result in &self.state.contract.results {
                if !self.state.root {
                    continue;
                }
                for path in &result.paths {
                    let binding = match &result.ty {
                        SemanticType::Tensor(tensor) => {
                            let publications = executable
                                .storage()
                                .result_views()
                                .iter()
                                .filter(|publication| &publication.path == path)
                                .collect::<Vec<_>>();
                            assert!(
                                !publications.is_empty(),
                                "closed result tensor path has no publication operation"
                            );
                            assert!(
                                publications.iter().all(|publication| {
                                    let view = executable.storage().view(publication.view);
                                    view.representation == tensor.representation
                                        && view.extents.len() == tensor.axes.len()
                                }),
                                "source result publication changed its declared tensor type"
                            );
                            PublishedResult::Buffer {
                                representation: tensor.representation,
                                rank: tensor.axes.len(),
                            }
                        }
                        SemanticType::Scalar(_) => {
                            let ScalarPublication::Scalar(slot) = self
                                .state
                                .physical
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed scalar result has no publication")
                            else {
                                panic!("closed scalar result has a range publication")
                            };
                            PublishedResult::Scalar { slot }
                        }
                        SemanticType::Index { .. } => {
                            let ScalarPublication::Quantity(slot) = self
                                .state
                                .physical
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed index result has no publication")
                            else {
                                panic!("closed index result has a non-natural publication")
                            };
                            PublishedResult::Quantity { slot }
                        }
                        SemanticType::Integer => {
                            panic!("root result ABI does not expose unbounded signed Integer")
                        }
                        SemanticType::Range { .. } => {
                            let ScalarPublication::Range { start, end } = self
                                .state
                                .physical
                                .result_slots
                                .get(&result.value)
                                .copied()
                                .expect("closed range result has no publication")
                            else {
                                panic!("closed range result has a scalar publication")
                            };
                            PublishedResult::Range { start, end }
                        }
                        SemanticType::Void => continue,
                        SemanticType::Tuple(_) => {
                            panic!("tuple result survived semantic leaf normalization")
                        }
                        SemanticType::Opaque { .. } => {
                            panic!("opaque result crossed the implementation boundary")
                        }
                    };
                    result_publications.push(ResultPublication {
                        path: path.clone(),
                        binding,
                    });
                }
            }
            ConstructedCandidate::from_parts(ConstructedCandidateParts {
                identity,
                semantic_coverage: coverage,
                native_index_bits: self.target.limits().max_index_bits,
                executable,
                choices: self.state.choices,
                hard_constraints,
                numerical_applicability,
                provenance: ImplementationProvenance {
                    root: self.function.stable(),
                    callees: self.state.callees,
                },
                result_publications,
                bindings: self.state.bindings.freeze(),
            })
        }

        fn derive_constraints(
            &mut self,
            storage: &seismic_ir::storage::GlobalAllocationTopology,
            allocation_ids: &[GlobalAllocationId],
            schedule: &ParametricSchedule<B>,
            kernels: &[&seismic_ir::kernel::Kernel<B>],
            resources: &[seismic_ir::execution::LaunchResources],
        ) {
            let max_allocation = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_allocation_bytes),
            );
            // Natural expressions are unbounded, so even a 64-bit native
            // index limit is a real target constraint. Form the exact bound
            // for the target's declared width rather than truncating to u64.
            let index_bits = self.target.limits().max_index_bits;
            let max_index = if index_bits <= 64 {
                self.arena.nat(u64::MAX >> (64 - index_bits))
            } else {
                self.arena.nat_exact(
                    (seismic_lang::expr::BigUint::from(1u8) << index_bits)
                        - seismic_lang::expr::BigUint::from(1u8),
                )
            };
            assert_eq!(allocation_ids.len(), storage.allocations().len());
            for &id in allocation_ids {
                let allocation = storage.allocation(id);
                let bytes = allocation.reserved_bytes;
                // An imported allocation is a proxy for a caller-owned view.
                // The caller established that view's representability where
                // it acquired it; restating its span here puts the caller's
                // reached geometry slots into this callee's guard.
                let imported = matches!(
                    allocation.kind,
                    seismic_ir::storage::GlobalBufferKind::Imported { .. }
                );
                if allocation.acquisition == seismic_ir::storage::AllocationAcquisition::Invocation
                    && !imported
                {
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        bytes,
                        max_allocation,
                    ));
                    self.state
                        .constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, bytes, max_index));
                }
                let required_alignment = storage.allocation(id).alignment;
                self.state.constraints.push(
                    self.arena
                        .bool(required_alignment <= self.target.limits().max_allocation_alignment),
                );
            }
            let mut view_requirements = Vec::new();
            for (view_ordinal, layout) in storage.views().iter().enumerate() {
                let constraint_start = self.state.constraints.len();
                // The whole-tensor constructor gives this view its allocation's
                // exact geometry and byte demand. Rechecking its own span as
                // `bytes <= bytes` introduces reached tensor slots into an
                // invocation guard without establishing any new property.
                if matches!(
                    layout.mapping,
                    seismic_ir::storage::ViewMapping::WholeAllocation
                ) {
                    view_requirements.push(self.arena.bool(true));
                    continue;
                }
                if let seismic_ir::storage::ViewMapping::Transpose { source, .. } = &layout.mapping
                {
                    view_requirements.push(view_requirements[source.index() as usize]);
                    continue;
                }
                if matches!(layout.base, seismic_ir::storage::ViewBase::TensorValue(id) if id.index() as usize == view_ordinal)
                {
                    view_requirements.push(self.arena.bool(true));
                    continue;
                }
                // Whole canonical views must use the same tensor-byte
                // expression as allocation and TargetDomain construction.
                // Re-expanding them as a generic strided range introduces an
                // equivalent max/sub/select tree (and subtraction side
                // conditions) that structural universal closure cannot and
                // should not have to rediscover. Nonzero-offset and strided
                // views retain the full addressed-range proof below.
                let addressed = if layout.contiguous
                    && matches!(
                        self.arena.view(AnyExpr::Nat(layout.offset)),
                        NodeView::NatConst(0)
                    ) {
                    seismic_ir::storage::tensor_bytes(
                        self.arena,
                        layout.representation,
                        &layout.extents,
                    )
                } else {
                    addressed_bytes(self.arena, layout)
                };
                let allocation_bytes = match layout.base {
                    seismic_ir::storage::ViewBase::Allocation(id) => storage.allocation(id).bytes,
                    seismic_ir::storage::ViewBase::TensorValue(id) => {
                        let source = &storage.views()[id.index() as usize];
                        addressed_bytes(self.arena, source)
                    }
                };
                self.state.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    addressed,
                    allocation_bytes,
                ));
                let view_alignment =
                    seismic_ir::storage::representation_alignment(layout.representation);
                let alignment = self.arena.nat(view_alignment);
                let remainder = self.arena.nat_rem(layout.offset, alignment);
                let zero = self.arena.nat(0);
                self.state
                    .constraints
                    .push(self.arena.nat_cmp(CmpOp::Eq, remainder, zero));
                let terms = self.state.constraints.split_off(constraint_start);
                view_requirements.push(self.arena.all(&terms));
            }
            let max_bindings = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_bindings),
            );
            let max_argument_bytes = self.arena.nat_symbol(
                self.arena
                    .target_constant_symbol(self.constants.max_argument_bytes),
            );
            let mut launch_constraints = Vec::new();
            for (launch, resources) in schedule.launches().iter().zip(resources) {
                let local_layout = resources.layout();
                let scratch = resources.scratch();
                let abi_allocations = resources.abi();
                let constraint_start = self.state.constraints.len();
                let threads = self.arena.nat_product(&launch.workgroup);
                let max_threads = self.arena.nat(self.target.limits().max_workgroup_threads);
                self.state
                    .constraints
                    .push(self.arena.nat_cmp(CmpOp::Le, threads, max_threads));
                let grid_envelope = schedule.launch_grid_envelope(self.arena, launch);
                for axis in 0..3 {
                    let max_workgroup_axis = self
                        .arena
                        .nat(self.target.limits().max_workgroup_size[axis]);
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        launch.workgroup[axis],
                        max_workgroup_axis,
                    ));
                    let max_grid = self.arena.nat(self.target.limits().max_grid[axis]);
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        grid_envelope[axis],
                        max_grid,
                    ));
                    let physical_participants = self
                        .arena
                        .nat_mul(grid_envelope[axis], launch.workgroup[axis]);
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        physical_participants,
                        max_index,
                    ));
                }

                let kernel = kernels
                    .get(launch.kernel.index() as usize)
                    .expect("launch kernel belongs to this implementation");
                for local in &local_layout.locals {
                    for value in std::iter::once(local.offset)
                        .chain(local.extents.iter().copied())
                        .chain(local.strides.iter().copied())
                        .chain(std::iter::once(local.bytes))
                    {
                        self.state.constraints.push(self.arena.nat_cmp(
                            CmpOp::Le,
                            value,
                            max_index,
                        ));
                    }
                }
                for total in [
                    local_layout.workgroup_bytes,
                    local_layout.participant_bytes,
                    local_layout.register_bytes,
                ] {
                    self.state
                        .constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, total, max_index));
                }
                for requirement in [&scratch.workgroup, &scratch.participant, &scratch.register]
                    .into_iter()
                    .flatten()
                {
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        requirement.bytes,
                        max_allocation,
                    ));
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        requirement.bytes,
                        max_index,
                    ));
                    self.state.constraints.push(self.arena.bool(
                        requirement.alignment.is_power_of_two()
                            && requirement.alignment
                                <= self.target.limits().max_allocation_alignment,
                    ));
                }
                let data = kernel;
                for lease in data.addressable_resources() {
                    let capacity = self.arena.nat_symbol(self.arena.target_constant_symbol(
                        self.constants.addressable_resource_capacity(lease.class_id),
                    ));
                    let end = self.arena.nat_add(lease.offset_units, lease.units);
                    self.state
                        .constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, end, capacity));
                    self.state
                        .constraints
                        .push(self.arena.nat_cmp(CmpOp::Le, end, max_index));
                }
                let binding_count = self.arena.nat(data.interface().bindings.len() as u64);
                self.state.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    binding_count,
                    max_bindings,
                ));
                // Core ABI metadata size, not bytes reachable through bound
                // tensors. Buffer/result slots are encoded as machine-width
                // addresses, Nat arguments as u64, and scalar arguments at
                // their declared width.
                let footprint = self.target.kernel_abi_layout(kernel).footprint;
                assert!(
                    footprint.alignment.is_power_of_two(),
                    "backend kernel ABI footprint alignment is a nonzero power of two"
                );
                let argument_bytes = self.arena.nat(footprint.bytes);
                self.state.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    argument_bytes,
                    max_argument_bytes,
                ));
                for allocation in abi_allocations {
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        allocation.bytes,
                        max_allocation,
                    ));
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        allocation.bytes,
                        max_index,
                    ));
                    self.state.constraints.push(self.arena.bool(
                        allocation.alignment.is_power_of_two()
                            && allocation.alignment
                                <= self.target.limits().max_allocation_alignment,
                    ));
                }

                for resource in data.intrinsic_resources() {
                    if resource.requires_subgroup {
                        assert!(
                            data.interface().uses_subgroup,
                            "intrinsic subgroup requirement was not propagated to the kernel interface"
                        );
                    }
                }
                let dynamic_local_limit = self.arena.nat(self.target.limits().max_workgroup_bytes);
                self.state.constraints.push(self.arena.nat_cmp(
                    CmpOp::Le,
                    local_layout.workgroup_bytes,
                    dynamic_local_limit,
                ));
                if let Some(participant_limit) = self.constants.participant_local_bytes {
                    let participant_limit = self
                        .arena
                        .nat_symbol(self.arena.target_constant_symbol(participant_limit));
                    self.state.constraints.push(self.arena.nat_cmp(
                        CmpOp::Le,
                        local_layout.participant_bytes,
                        participant_limit,
                    ));
                }
                let terms = self.state.constraints.split_off(constraint_start);
                let predicate = self.arena.all(&terms);
                let defined = self.arena.side_conditions(predicate.into());
                launch_constraints.push(self.arena.and(defined, predicate));
            }
            let scoped = schedule.scoped_requirements(
                self.arena,
                storage.views(),
                &mut |arena, point| {
                    use seismic_ir::schedule::RequirementPoint;
                    let mut terms = Vec::new();
                    match point {
                        RequirementPoint::Step(ScheduleStep::Launch(id)) => {
                            terms.push(launch_constraints[id.index() as usize]);
                            let mut views = Vec::new();
                            for binding in &kernels[schedule.launch(*id).kernel.index() as usize]
                                .interface()
                                .bindings
                            {
                                views.push(view_requirements[binding.view.index() as usize]);
                            }
                            let views = arena.all(&views);
                            let reached = arena.not(schedule.launch(*id).empty);
                            terms.push(arena.implies(reached, views));
                        }
                        RequirementPoint::Step(ScheduleStep::Copy(copy)) => {
                            terms.push(view_requirements[copy.source.index() as usize]);
                            terms.push(view_requirements[copy.destination.index() as usize]);
                        }
                        RequirementPoint::Step(ScheduleStep::Fill(fill)) => {
                            terms.push(view_requirements[fill.destination.index() as usize])
                        }
                        RequirementPoint::Step(ScheduleStep::ScalarRead(read)) => {
                            terms.push(view_requirements[read.source.index() as usize])
                        }
                        RequirementPoint::Step(ScheduleStep::PublishTensor { view, .. }) => {
                            terms.push(view_requirements[view.index() as usize])
                        }
                        RequirementPoint::BranchResult(results, then) => {
                            results.visit(&mut |result| {
                                if let seismic_ir::region::ValueOperand::Tensor(view) = if then {
                                    result.then_value()
                                } else {
                                    result.else_value()
                                } {
                                    terms.push(view_requirements[view.index() as usize]);
                                }
                            })
                        }
                        RequirementPoint::RepeatInitial(carries) => carries.visit(&mut |carry| {
                            if let seismic_ir::region::ValueOperand::Tensor(view) = carry.initial()
                            {
                                terms.push(view_requirements[view.index() as usize]);
                            }
                        }),
                        RequirementPoint::RepeatBackedge(carries) => carries.visit(&mut |carry| {
                            if let seismic_ir::region::ValueOperand::Tensor(view) = carry.backedge()
                            {
                                terms.push(view_requirements[view.index() as usize]);
                            }
                        }),
                        _ => {}
                    }
                    arena.all(&terms)
                },
                &mut |arena, step| acquisition_established(arena, storage, step),
            );
            self.state.constraints.push(scoped);
        }

        fn validate_schedule(
            &mut self,
            schedule: &ParametricSchedule<B>,
            storage: &seismic_ir::storage::TopologyBuilder,
        ) {
            let scoped = schedule.scoped_requirements(
                self.arena,
                storage.views(),
                &mut |arena, point| {
                    use seismic_ir::schedule::RequirementPoint;
                    let mut terms = Vec::new();
                    let (views, bytes) = match point {
                        RequirementPoint::Step(ScheduleStep::Copy(copy)) => {
                            let source = storage.view_layout(copy.source);
                            let destination = storage.view_layout(copy.destination);
                            assert_eq!(
                                source.representation, destination.representation,
                                "copy representation mismatch"
                            );
                            assert_eq!(
                                source.extents.len(),
                                destination.extents.len(),
                                "copy rank mismatch"
                            );
                            for (left, right) in source.extents.iter().zip(&destination.extents) {
                                terms.push(arena.nat_cmp(CmpOp::Eq, *left, *right));
                            }
                            (vec![copy.source, copy.destination], copy.bytes)
                        }
                        RequirementPoint::Step(ScheduleStep::Fill(fill)) => {
                            (vec![fill.destination], fill.bytes)
                        }
                        RequirementPoint::Step(ScheduleStep::PublishTensor {
                            view,
                            declared_axes,
                            ..
                        }) => {
                            let actual = storage.view_layout(*view);
                            assert_eq!(
                                actual.extents.len(),
                                declared_axes.len(),
                                "published result rank changed"
                            );
                            for (reached, declared) in actual.extents.iter().zip(declared_axes) {
                                terms.push(arena.nat_cmp(CmpOp::Eq, *reached, *declared));
                            }
                            return arena.all(&terms);
                        }
                        _ => return arena.bool(true),
                    };
                    terms.extend(
                        views
                            .into_iter()
                            .map(|view| {
                                let layout = storage.view_layout(view);
                                let payload = seismic_ir::storage::tensor_bytes(
                                    arena,
                                    layout.representation,
                                    &layout.extents,
                                );
                                arena.nat_cmp(CmpOp::Le, bytes, payload)
                            })
                            .collect::<Vec<_>>(),
                    );
                    arena.all(&terms)
                },
                &mut |arena, step| {
                    if let ScheduleStep::BeginAllocationInstance { source: view, .. } = step {
                        if let seismic_ir::storage::ViewBase::Allocation(id) =
                            storage.view_layout(*view).base
                        {
                            return arena.side_conditions(storage.allocation_bytes(id).into());
                        }
                    }
                    arena.bool(true)
                },
            );
            self.state.constraints.push(scoped);
        }

        fn register_roots(
            &mut self,
            storage: &seismic_ir::storage::GlobalAllocationTopology,
            allocation_ids: &[GlobalAllocationId],
            kernels: &[&seismic_ir::kernel::Kernel<B>],
            schedule: &ParametricSchedule<B>,
            resources: &[seismic_ir::execution::LaunchResources],
            hard_constraints: BoolExpr,
            coverage: TargetPredicate,
        ) -> Vec<seismic_lang::expr::RootId> {
            let mut roots = Vec::new();
            assert_eq!(allocation_ids.len(), storage.allocations().len());
            for (allocation, &id) in allocation_ids.iter().enumerate() {
                roots.push(self.arena.root(
                    RootName::AllocationBytes {
                        allocation: allocation as u32,
                    },
                    AnyExpr::Nat(storage.allocation(id).bytes),
                ));
            }
            for (view, layout) in storage.views().iter().enumerate() {
                roots.push(self.arena.root(
                    RootName::ViewOffset { view: view as u32 },
                    AnyExpr::Nat(layout.offset),
                ));
                for (axis, extent) in layout.extents.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::ViewExtent {
                            view: view as u32,
                            axis: axis as u32,
                        },
                        AnyExpr::Nat(*extent),
                    ));
                }
                for (axis, stride) in layout.strides.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::ViewStride {
                            view: view as u32,
                            axis: axis as u32,
                        },
                        AnyExpr::Nat(*stride),
                    ));
                }
            }
            for (launch_id, launch) in schedule.launches().iter().enumerate() {
                for axis in 0..3 {
                    roots.push(self.arena.root(
                        RootName::LaunchGrid {
                            launch: launch_id as u32,
                            axis: axis as u8,
                        },
                        AnyExpr::Nat(launch.grid[axis]),
                    ));
                }
                for axis in 0..3 {
                    roots.push(self.arena.root(
                        RootName::Workgroup {
                            launch: launch_id as u32,
                            axis: axis as u8,
                        },
                        AnyExpr::Nat(launch.workgroup[axis]),
                    ));
                }
                roots.push(self.arena.root(
                    RootName::LaunchEmpty {
                        launch: launch_id as u32,
                    },
                    AnyExpr::Bool(launch.empty),
                ));
                let layout = resources[launch_id].layout();
                for (local, value) in layout.locals.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::LocalOffset {
                            launch: launch_id as u32,
                            local: local as u32,
                        },
                        AnyExpr::Nat(value.offset),
                    ));
                    for (axis, stride) in value.strides.iter().enumerate() {
                        roots.push(self.arena.root(
                            RootName::LocalStride {
                                launch: launch_id as u32,
                                local: local as u32,
                                axis: axis as u32,
                            },
                            AnyExpr::Nat(*stride),
                        ));
                    }
                }
                for (class, bytes) in [
                    layout.workgroup_bytes,
                    layout.participant_bytes,
                    layout.register_bytes,
                ]
                .into_iter()
                .enumerate()
                {
                    roots.push(self.arena.root(
                        RootName::LocalClassBytes {
                            launch: launch_id as u32,
                            class: class as u8,
                        },
                        AnyExpr::Nat(bytes),
                    ));
                }
                let scratch = resources[launch_id].scratch();
                for (class, requirement) in
                    [&scratch.workgroup, &scratch.participant, &scratch.register]
                        .into_iter()
                        .enumerate()
                {
                    if let Some(requirement) = requirement {
                        roots.push(self.arena.root(
                            RootName::LaunchScratchBytes {
                                launch: launch_id as u32,
                                class: class as u8,
                            },
                            AnyExpr::Nat(requirement.bytes),
                        ));
                    }
                }
                for (allocation, requirement) in resources[launch_id].abi().iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::LaunchAbiBytes {
                            launch: launch_id as u32,
                            allocation: allocation as u32,
                        },
                        AnyExpr::Nat(requirement.bytes),
                    ));
                }
            }
            for (kernel_index, kernel) in kernels.iter().enumerate() {
                let data = kernel;
                for (argument, value) in data.interface().nat_args.iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::KernelNatArgument {
                            kernel: kernel_index as u32,
                            argument: argument as u32,
                        },
                        AnyExpr::Nat(*value),
                    ));
                }
                for (argument, (symbol, _)) in data.interface().scalar_args.iter().enumerate() {
                    let expression = match self.arena.symbol_sort(*symbol) {
                        SymbolSort::Nat => AnyExpr::Nat(self.arena.nat_symbol(*symbol)),
                        SymbolSort::Int => AnyExpr::Int(self.arena.int_symbol(*symbol)),
                        SymbolSort::Scalar(DType::F32) => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::F32>(*symbol),
                        ),
                        SymbolSort::Scalar(DType::F16) => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::F16>(*symbol),
                        ),
                        SymbolSort::Scalar(DType::BF16) => AnyExpr::from(
                            self.arena
                                .scalar_symbol::<seismic_lang::expr::BF16>(*symbol),
                        ),
                        SymbolSort::Scalar(DType::I32) => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::I32>(*symbol),
                        ),
                        SymbolSort::Scalar(DType::U32) => AnyExpr::from(
                            self.arena.scalar_symbol::<seismic_lang::expr::U32>(*symbol),
                        ),
                        SymbolSort::Scalar(DType::Bool) => AnyExpr::from(
                            self.arena
                                .scalar_symbol::<seismic_lang::expr::BoolScalar>(*symbol),
                        ),
                    };
                    roots.push(self.arena.root(
                        RootName::KernelScalarArgument {
                            kernel: kernel_index as u32,
                            argument: argument as u32,
                        },
                        expression,
                    ));
                }
                for (local, allocation) in data.locals().iter().enumerate() {
                    for (axis, extent) in allocation.extents.iter().enumerate() {
                        roots.push(self.arena.root(
                            RootName::LocalExtent {
                                kernel: kernel_index as u32,
                                local: local as u32,
                                axis: axis as u32,
                            },
                            AnyExpr::Nat(*extent),
                        ));
                    }
                }
                for (lease, resource) in data.addressable_resources().iter().enumerate() {
                    roots.push(self.arena.root(
                        RootName::AddressableResourceOffset {
                            kernel: kernel_index as u32,
                            lease: lease as u32,
                        },
                        AnyExpr::Nat(resource.offset_units),
                    ));
                    roots.push(self.arena.root(
                        RootName::AddressableResourceUnits {
                            kernel: kernel_index as u32,
                            lease: lease as u32,
                        },
                        AnyExpr::Nat(resource.units),
                    ));
                }
                for (resource, values) in data.intrinsic_resources().iter().enumerate() {
                    if let Some(value) = values.workgroup_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicWorkgroupBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                    if let Some(value) = values.participant_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicParticipantBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                    if let Some(value) = values.register_bytes {
                        roots.push(self.arena.root(
                            RootName::IntrinsicRegisterBytes {
                                kernel: kernel_index as u32,
                                resource: resource as u32,
                            },
                            AnyExpr::Nat(value),
                        ));
                    }
                }
            }
            fn schedule_roots(
                arena: &mut ExprArena,
                steps: &[ScheduleStep],
                roots: &mut Vec<seismic_lang::expr::RootId>,
                control: &mut u32,
                repeat: &mut u32,
                scalar_read: &mut u32,
                host_eval: &mut u32,
                publication: &mut u32,
            ) {
                fn operand_root(
                    arena: &mut ExprArena,
                    value: seismic_ir::region::ValueOperand,
                    roots: &mut Vec<seismic_lang::expr::RootId>,
                    ordinal: &mut u32,
                ) {
                    use seismic_ir::region::{QuantityOperand, ScalarOperand, ValueOperand};
                    let expression = match value {
                        ValueOperand::Tensor(_) => return,
                        ValueOperand::Scalar(ScalarOperand::Natural(value))
                        | ValueOperand::Quantity(QuantityOperand::Natural(value)) => {
                            AnyExpr::Nat(value)
                        }
                        ValueOperand::Quantity(QuantityOperand::Integer(value)) => {
                            AnyExpr::Int(value)
                        }
                        ValueOperand::Scalar(ScalarOperand::Word { symbol, dtype }) => {
                            match dtype {
                                DType::F32 => arena
                                    .scalar_symbol::<seismic_lang::expr::F32>(symbol)
                                    .into(),
                                DType::F16 => arena
                                    .scalar_symbol::<seismic_lang::expr::F16>(symbol)
                                    .into(),
                                DType::BF16 => arena
                                    .scalar_symbol::<seismic_lang::expr::BF16>(symbol)
                                    .into(),
                                DType::I32 => arena
                                    .scalar_symbol::<seismic_lang::expr::I32>(symbol)
                                    .into(),
                                DType::U32 => arena
                                    .scalar_symbol::<seismic_lang::expr::U32>(symbol)
                                    .into(),
                                DType::Bool => arena
                                    .scalar_symbol::<seismic_lang::expr::BoolScalar>(symbol)
                                    .into(),
                            }
                        }
                    };
                    roots.push(
                        arena.root(RootName::RegionOperand { operand: *ordinal }, expression),
                    );
                    *ordinal = ordinal
                        .checked_add(1)
                        .expect("region operand identity space exhausted");
                }
                for step in steps {
                    match step {
                        ScheduleStep::BeginAllocationInstance { .. }
                        | ScheduleStep::BindArgumentTensor { .. }
                        | ScheduleStep::Launch(_)
                        | ScheduleStep::Copy(_)
                        | ScheduleStep::Fill(_)
                        | ScheduleStep::ScalarMove(_)
                        | ScheduleStep::Check(_) => {}
                        ScheduleStep::PublishTensor { declared_axes, .. } => {
                            let step = *publication;
                            *publication = publication
                                .checked_add(1)
                                .expect("publication root ordinal space exhausted");
                            for (axis, value) in declared_axes.iter().enumerate() {
                                roots.push(arena.root(
                                    RootName::PublishedExtent {
                                        step,
                                        axis: axis as u32,
                                    },
                                    AnyExpr::Nat(*value),
                                ));
                            }
                        }
                        ScheduleStep::Imported { body, .. } => schedule_roots(
                            arena,
                            body,
                            roots,
                            control,
                            repeat,
                            scalar_read,
                            host_eval,
                            publication,
                        ),
                        ScheduleStep::EvaluateHost(evaluation) => {
                            let step = *host_eval;
                            *host_eval = (*host_eval)
                                .checked_add(1)
                                .expect("host-evaluation root ordinal space exhausted");
                            let value = match evaluation.value {
                                seismic_ir::schedule::HostValueExpr::Integer(value)
                                | seismic_ir::schedule::HostValueExpr::Word { value, .. }
                                | seismic_ir::schedule::HostValueExpr::Float { value, .. } => {
                                    AnyExpr::Int(value)
                                }
                                seismic_ir::schedule::HostValueExpr::Natural(value) => {
                                    AnyExpr::Nat(value)
                                }
                                seismic_ir::schedule::HostValueExpr::Bool(value) => {
                                    AnyExpr::Bool(value)
                                }
                            };
                            roots.push(arena.root(RootName::HostEvaluation { step }, value));
                        }
                        ScheduleStep::ScalarRead(read) => {
                            let step = *scalar_read;
                            *scalar_read = (*scalar_read)
                                .checked_add(1)
                                .expect("scalar-read root ordinal space exhausted");
                            for (axis, index) in read.index.iter().enumerate() {
                                roots.push(arena.root(
                                    RootName::ScalarReadIndex {
                                        step,
                                        axis: axis as u32,
                                    },
                                    AnyExpr::Nat(*index),
                                ));
                            }
                        }
                        ScheduleStep::If {
                            condition,
                            then_steps,
                            else_steps,
                            results,
                        } => {
                            results.visit(&mut |result| {
                                for operand in [result.then_value(), result.else_value()] {
                                    operand_root(arena, operand, roots, host_eval);
                                }
                            });
                            let id = *control;
                            *control = (*control)
                                .checked_add(1)
                                .expect("schedule-control root ordinal space exhausted");
                            roots.push(arena.root(
                                RootName::ScheduleCondition { control: id },
                                AnyExpr::Bool(*condition),
                            ));
                            schedule_roots(
                                arena,
                                then_steps,
                                roots,
                                control,
                                repeat,
                                scalar_read,
                                host_eval,
                                publication,
                            );
                            schedule_roots(
                                arena,
                                else_steps,
                                roots,
                                control,
                                repeat,
                                scalar_read,
                                host_eval,
                                publication,
                            );
                        }
                        ScheduleStep::Repeat {
                            start,
                            end,
                            body,
                            carries,
                            ..
                        } => {
                            carries.visit(&mut |carry| {
                                for operand in [carry.initial(), carry.backedge()] {
                                    operand_root(arena, operand, roots, host_eval);
                                }
                            });
                            let id = *repeat;
                            *repeat = (*repeat)
                                .checked_add(1)
                                .expect("schedule-repeat root ordinal space exhausted");
                            roots.push(
                                arena.root(
                                    RootName::RepeatStart { repeat: id },
                                    AnyExpr::Nat(*start),
                                ),
                            );
                            roots.push(
                                arena.root(RootName::RepeatEnd { repeat: id }, AnyExpr::Nat(*end)),
                            );
                            schedule_roots(
                                arena,
                                body,
                                roots,
                                control,
                                repeat,
                                scalar_read,
                                host_eval,
                                publication,
                            );
                        }
                    }
                }
            }
            let (mut control, mut repeat, mut scalar_read, mut host_eval, mut publication) =
                (0, 0, 0, 0, 0);
            schedule_roots(
                self.arena,
                schedule.steps(),
                &mut roots,
                &mut control,
                &mut repeat,
                &mut scalar_read,
                &mut host_eval,
                &mut publication,
            );
            roots.push(
                self.arena
                    .root(RootName::HardConstraints, AnyExpr::Bool(hard_constraints)),
            );
            roots.push(
                self.arena
                    .root(RootName::Guard, AnyExpr::Bool(coverage.node())),
            );
            roots
        }
    }

    /// Contribution of one view axis to its furthest addressed element.
    ///
    /// A zero stride means the axis never participates in address formation.
    /// Do not construct or evaluate `(max(extent, 1) - 1)` in that case: the
    /// extent can itself be a partial range subtraction, but its definedness
    /// is irrelevant to this axis's address contribution.
    pub(super) use seismic_ir::storage::addressed_bytes;

    fn digest_structure<B: seismic_native_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        builder: &Builder<'_, B>,
        storage: &seismic_ir::storage::GlobalAllocationTopology,
        allocation_ids: &[GlobalAllocationId],
        kernels: &[&seismic_ir::kernel::Kernel<B>],
        schedule: &ParametricSchedule<B>,
    ) {
        digest.bytes(builder.function.stable().digest());
        digest.hashed(&builder.state.choices.len());
        for choice in &builder.state.choices {
            digest.bytes(choice.meaning.as_bytes());
            digest.hashed(builder.arena.decision_domain(choice.decision).values());
        }

        assert_eq!(allocation_ids.len(), storage.allocations().len());
        digest.hashed(&(storage.allocations().len() as u32));
        for &id in allocation_ids {
            match &storage.allocation(id).kind {
                GlobalBufferKind::Argument { value, abi } => {
                    digest.bytes(b"argument");
                    digest.hashed(
                        &builder
                            .state
                            .contract
                            .parameters
                            .iter()
                            .position(|parameter| parameter.value == *value),
                    );
                    digest.hashed(
                        &abi.as_ref()
                            .and_then(|id| {
                                builder
                                    .state
                                    .contract
                                    .parameters
                                    .iter()
                                    .position(|parameter| parameter.value == *value)
                                    .map(|_| id)
                            })
                            .is_some(),
                    );
                }
                GlobalBufferKind::Result { value } => {
                    digest.bytes(b"result");
                    digest.hashed(
                        &builder
                            .state
                            .contract
                            .results
                            .iter()
                            .position(|result| result.value == *value),
                    );
                }
                GlobalBufferKind::Imported { value, source } => {
                    digest.bytes(b"imported");
                    digest.hashed(
                        &builder
                            .state
                            .contract
                            .parameters
                            .iter()
                            .position(|parameter| parameter.value == *value),
                    );
                    digest.hashed(
                        &builder
                            .state
                            .contract
                            .results
                            .iter()
                            .position(|result| result.value == *value),
                    );
                    digest.bytes(
                        seismic_lang::registry::representation_info(source.representation())
                            .name
                            .as_bytes(),
                    );
                }
                GlobalBufferKind::Arena => digest.bytes(b"arena"),
                GlobalBufferKind::Persistent => digest.bytes(b"persistent"),
            }
            digest.hashed(&storage.allocation(id).alignment);
            digest.hashed(&storage.allocation(id).instances);
            digest.hashed(&storage.allocation(id).liveness.uses().len());
            for at in storage.allocation(id).liveness.uses() {
                digest.hashed(at.region());
                digest.hashed(&at.ordinal());
            }
            digest.hashed(&storage.allocation(id).slot.and_then(|decision| {
                builder
                    .state
                    .choices
                    .iter()
                    .position(|choice| choice.decision == decision)
            }));
        }
        digest.hashed(&storage.views().len());
        for view in storage.views() {
            match view.base {
                seismic_ir::storage::ViewBase::Allocation(id) => {
                    digest.hashed(&0u8);
                    digest.hashed(&id.index());
                }
                seismic_ir::storage::ViewBase::TensorValue(id) => {
                    digest.hashed(&1u8);
                    digest.hashed(&id.index());
                }
            }
            digest.bytes(
                seismic_lang::registry::representation_info(view.representation)
                    .name
                    .as_bytes(),
            );
            digest.hashed(&view.extents.len());
            digest.hashed(&view.contiguous);
            digest.hashed(&matches!(
                view.mapping,
                seismic_ir::storage::ViewMapping::WholeAllocation
            ));
        }
        digest.hashed(&storage.result_views().len());
        for publication in storage.result_views() {
            digest.hashed(&publication.path);
            digest.hashed(&publication.view.index());
        }

        digest.hashed(&kernels.len());
        for kernel in kernels {
            digest_kernel::<B>(digest, kernel);
        }
        digest.hashed(&schedule.launches().len());
        for launch in schedule.launches() {
            digest.bytes(b"launch-definition");
            digest.hashed(&launch.kernel.index());
            digest.hashed(&launch.descriptor);
        }
        digest_schedule(digest, builder, schedule.steps());

        for result in &builder.state.contract.results {
            if let Some(publication) = builder.state.physical.result_slots.get(&result.value) {
                digest.bytes(b"published-scalar");
                digest.hashed(
                    &builder
                        .state
                        .contract
                        .results
                        .iter()
                        .position(|candidate| candidate.value == result.value),
                );
                match publication {
                    ScalarPublication::Scalar(slot) => {
                        digest.bytes(b"one");
                        digest.hashed(&slot.index());
                        digest.hashed(&slot.kind());
                    }
                    ScalarPublication::Quantity(slot) => {
                        digest.bytes(b"quantity");
                        digest.hashed(&slot.index());
                        digest.bytes(match slot.kind() {
                            HostQuantityKind::Integer => b"integer",
                            HostQuantityKind::Natural => b"natural",
                        });
                    }
                    ScalarPublication::Range { start, end } => {
                        digest.bytes(b"range");
                        digest.hashed(&start.index());
                        digest.hashed(&end.index());
                    }
                }
            }
        }
        for callee in &builder.state.callees {
            digest.bytes(callee.digest());
        }
    }

    fn digest_kernel<B: seismic_native_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        kernel: &seismic_ir::kernel::Kernel<B>,
    ) {
        use seismic_ir::kernel::ops::{Op, PlaceRef};
        fn value(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: seismic_ir::kernel::ops::ErasedValue,
        ) {
            digest.hashed(&value.ordinal());
        }
        fn values(
            digest: &mut seismic_ir::identity::StructureDigest,
            values: &[seismic_ir::kernel::ops::ErasedValue],
        ) {
            digest.hashed(&values.len());
            for item in values {
                value(digest, *item);
            }
        }
        fn place(
            digest: &mut seismic_ir::identity::StructureDigest,
            place: seismic_ir::kernel::ops::PlaceRef,
        ) {
            match place {
                seismic_ir::kernel::ops::PlaceRef::Global { slot } => {
                    digest.bytes(b"global");
                    digest.hashed(&slot.ordinal());
                }
                seismic_ir::kernel::ops::PlaceRef::Local { index } => {
                    digest.bytes(b"local");
                    digest.hashed(&index);
                }
            }
        }
        let data = kernel;
        let interface = data.interface();
        digest.hashed(&interface.bindings.len());
        for binding in &interface.bindings {
            digest.hashed(&binding.slot.index());
            digest.hashed(&binding.view.index());
            digest.hashed(&binding.access);
            digest.hashed(&binding.rank);
        }
        digest.hashed(&interface.nat_args.len());
        digest.hashed(
            &interface
                .scalar_args
                .iter()
                .map(|(_, dtype)| *dtype)
                .collect::<Vec<_>>(),
        );
        for slot in &interface.result_slots {
            digest.hashed(&slot.index());
            digest.hashed(&slot.kind());
        }
        digest.hashed(&interface.uses_subgroup);
        digest.hashed(&data.value_types().copied().collect::<Vec<_>>());
        for local in data.locals() {
            digest.hashed(&local.kind);
            digest.bytes(
                seismic_lang::registry::representation_info(local.representation)
                    .name
                    .as_bytes(),
            );
            digest.hashed(&local.extents.len());
            digest.hashed(&local.alignment);
        }
        digest.hashed(&data.addressable_resources().len());
        for lease in data.addressable_resources() {
            digest.bytes(lease.class.stable_name.as_bytes());
            digest.bytes(lease.class.unit_name.as_bytes());
            digest.hashed(&lease.class.ownership);
            digest.hashed(&lease.class.realization);
            digest.hashed(&lease.alignment_units);
            digest.hashed(&lease.lifetime);
        }
        digest.hashed(&data.blocks().len());
        for block in data.blocks() {
            digest.hashed(&block.ops.len());
            for op in &block.ops {
                match op {
                    Op::Constant {
                        out,
                        value: constant,
                    } => {
                        digest.bytes(b"constant");
                        value(digest, *out);
                        match constant {
                            seismic_ir::kernel::ops::ConstantValue::F32(value) => {
                                digest.bytes(b"f32");
                                digest.hashed(&value.to_bits())
                            }
                            seismic_ir::kernel::ops::ConstantValue::F16(value) => {
                                digest.bytes(b"f16");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::BF16(value) => {
                                digest.bytes(b"bf16");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::I32(value) => {
                                digest.bytes(b"i32");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::U32(value) => {
                                digest.bytes(b"u32");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::Bool(value) => {
                                digest.bytes(b"bool");
                                digest.hashed(value)
                            }
                            seismic_ir::kernel::ops::ConstantValue::Index(value) => {
                                digest.bytes(b"index");
                                digest.hashed(value)
                            }
                        }
                    }
                    Op::Binary { op, out, a, b } => {
                        digest.bytes(b"binary");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Unary { op, out, a } => {
                        digest.bytes(b"unary");
                        digest.hashed(op);
                        values(digest, &[*out, *a]);
                    }
                    Op::Bit { op, out, a, b } => {
                        digest.bytes(b"bit");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Fma { out, a, b, c } => {
                        digest.bytes(b"fma");
                        values(digest, &[*out, *a, *b, *c]);
                    }
                    Op::VectorFromLanes { out, lanes } => {
                        digest.bytes(b"vector-from-lanes");
                        value(digest, *out);
                        values(digest, lanes);
                    }
                    Op::VectorSplat { out, value: scalar } => {
                        digest.bytes(b"vector-splat");
                        values(digest, &[*out, *scalar]);
                    }
                    Op::VectorBinary { op, out, a, b } => {
                        digest.bytes(b"vector-binary");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::VectorUnary { op, out, a } => {
                        digest.bytes(b"vector-unary");
                        digest.hashed(op);
                        values(digest, &[*out, *a]);
                    }
                    Op::VectorBit { op, out, a, b } => {
                        digest.bytes(b"vector-bit");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::VectorFma { out, a, b, c } => {
                        digest.bytes(b"vector-fma");
                        values(digest, &[*out, *a, *b, *c]);
                    }
                    Op::VectorCast { out, a, to } => {
                        digest.bytes(b"vector-cast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::VectorLane { out, vector, lane } => {
                        digest.bytes(b"vector-lane");
                        digest.hashed(lane);
                        values(digest, &[*out, *vector]);
                    }
                    Op::VectorReduceAdd { out, vector } => {
                        digest.bytes(b"vector-reduce-add");
                        values(digest, &[*out, *vector]);
                    }
                    Op::Math {
                        op,
                        precision,
                        out,
                        a,
                    } => {
                        digest.bytes(b"math");
                        digest.hashed(op);
                        digest.hashed(precision);
                        values(digest, &[*out, *a]);
                    }
                    Op::Cast { out, a, to } => {
                        digest.bytes(b"cast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::Bitcast { out, a, to } => {
                        digest.bytes(b"bitcast");
                        digest.hashed(to);
                        values(digest, &[*out, *a]);
                    }
                    Op::ScalarBits { out, a } => {
                        digest.bytes(b"scalar-bits");
                        values(digest, &[*out, *a]);
                    }
                    Op::ScalarFromBits { out, a } => {
                        digest.bytes(b"scalar-from-bits");
                        values(digest, &[*out, *a]);
                    }
                    Op::Cmp { op, out, a, b } => {
                        digest.bytes(b"cmp");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Select { out, cond, a, b } => {
                        digest.bytes(b"select");
                        values(digest, &[*out, *cond, *a, *b]);
                    }
                    Op::Logic { op, out, a, b } => {
                        digest.bytes(b"logic");
                        digest.hashed(op);
                        values(digest, &[*out, *a, *b]);
                    }
                    Op::Not { out, a } => {
                        digest.bytes(b"not");
                        values(digest, &[*out, *a]);
                    }
                    Op::Geometry { out, kind } => {
                        digest.bytes(b"geometry");
                        digest.hashed(kind);
                        value(digest, *out);
                    }
                    Op::NatArg { out, index } => {
                        digest.bytes(b"nat-arg");
                        digest.hashed(index);
                        value(digest, *out);
                    }
                    Op::ScalarArg { out, index } => {
                        digest.bytes(b"scalar-arg");
                        digest.hashed(index);
                        value(digest, *out);
                    }
                    Op::Read {
                        out,
                        place: source,
                        representation,
                        index,
                    } => {
                        digest.bytes(b"read");
                        value(digest, *out);
                        place(digest, *source);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                    }
                    Op::VectorRead {
                        out,
                        place: source,
                        representation,
                        index,
                        axis,
                        active,
                    } => {
                        digest.bytes(b"vector-read");
                        value(digest, *out);
                        place(digest, *source);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        digest.hashed(axis);
                        value(digest, *active);
                    }
                    Op::VectorWrite {
                        place: destination,
                        representation,
                        index,
                        axis,
                        active,
                        value: stored,
                    } => {
                        digest.bytes(b"vector-write");
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        digest.hashed(axis);
                        values(digest, &[*active, *stored]);
                    }
                    Op::ReadPlaneField {
                        out,
                        place: source,
                        plane,
                        field,
                        index,
                    } => {
                        digest.bytes(b"read-plane-field");
                        value(digest, *out);
                        place(digest, *source);
                        digest.hashed(plane);
                        digest.hashed(field);
                        values(digest, index);
                    }
                    Op::ReadPlane {
                        out,
                        place: source,
                        plane,
                        element,
                        index,
                    } => {
                        digest.bytes(b"read-plane");
                        value(digest, *out);
                        place(digest, *source);
                        digest.hashed(plane);
                        value(digest, *element);
                        values(digest, index);
                    }
                    Op::RepresentationConvertPacket {
                        source,
                        destination,
                        conversion,
                        packet,
                    } => {
                        digest.bytes(b"representation-convert-packet");
                        place(digest, PlaceRef::Global { slot: *source });
                        place(digest, PlaceRef::Global { slot: *destination });
                        let conversion =
                            seismic_lang::registry::representation_conversion_info(*conversion);
                        digest.bytes(
                            seismic_lang::registry::representation_info(conversion.source)
                                .name
                                .as_bytes(),
                        );
                        digest.bytes(
                            seismic_lang::registry::representation_info(conversion.destination)
                                .name
                                .as_bytes(),
                        );
                        value(digest, *packet);
                    }
                    Op::Write {
                        place: destination,
                        representation,
                        index,
                        value: stored,
                    } => {
                        digest.bytes(b"write");
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        value(digest, *stored);
                    }
                    Op::Extent {
                        out,
                        place: source,
                        axis,
                    } => {
                        digest.bytes(b"extent");
                        value(digest, *out);
                        place(digest, *source);
                        digest.hashed(axis);
                    }
                    Op::Atomic {
                        op,
                        place: destination,
                        representation,
                        index,
                        value: stored,
                    } => {
                        digest.bytes(b"atomic");
                        digest.hashed(op);
                        place(digest, *destination);
                        digest.bytes(
                            seismic_lang::registry::representation_info(*representation)
                                .name
                                .as_bytes(),
                        );
                        values(digest, index);
                        value(digest, *stored);
                    }
                    Op::StoreSlot {
                        slot,
                        value: stored,
                        election,
                    } => {
                        digest.bytes(b"store-slot");
                        digest.hashed(slot);
                        digest.hashed(election);
                        value(digest, *stored);
                    }
                    Op::Barrier(scope) => {
                        digest.bytes(b"barrier");
                        digest.hashed(scope);
                    }
                    Op::Intrinsic {
                        intrinsic,
                        op,
                        outs,
                        args,
                        mapping_dependencies,
                    } => {
                        digest.bytes(b"intrinsic");
                        let signature = seismic_lang::registry::intrinsic_signature(*intrinsic);
                        digest.bytes(signature.name.as_bytes());
                        let mut identity = seismic_ir::identity::IntrinsicIdentityBuilder::new();
                        B::write_intrinsic_identity(op, &mut identity);
                        digest.bytes(&identity.finish());
                        values(digest, outs);
                        values(digest, args);
                        digest.bytes(b"mapping-dependencies");
                        values(digest, mapping_dependencies);
                    }
                    Op::Branch {
                        cond,
                        then,
                        otherwise,
                        outs,
                    } => {
                        digest.bytes(b"branch");
                        value(digest, *cond);
                        digest.hashed(&then.ordinal());
                        digest.hashed(&otherwise.ordinal());
                        values(digest, outs);
                    }
                    Op::Repeat {
                        start,
                        end,
                        binder,
                        carries_in,
                        carry_params,
                        body,
                        outs,
                    } => {
                        digest.bytes(b"repeat");
                        digest.hashed(&binder.ordinal());
                        values(digest, &[*start, *end]);
                        values(digest, carries_in);
                        values(digest, carry_params);
                        digest.hashed(&body.ordinal());
                        values(digest, outs);
                    }
                    Op::Yield { values: yielded } => {
                        digest.bytes(b"yield");
                        values(digest, yielded);
                    }
                }
            }
        }
        for intrinsic in data.intrinsics_used() {
            let signature = seismic_lang::registry::intrinsic_signature(*intrinsic);
            let capability = seismic_lang::registry::capability_info(signature.capability);
            digest.hashed(&capability.backend);
            digest.bytes(capability.name.as_bytes());
            digest.bytes(signature.name.as_bytes());
        }
        digest.hashed(data.resource_facts());
    }

    fn digest_schedule<B: seismic_native_target::TargetFamily>(
        digest: &mut seismic_ir::identity::StructureDigest,
        builder: &Builder<'_, B>,
        steps: &[ScheduleStep],
    ) {
        fn product<T>(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: &seismic_ir::region::Product<T>,
            leaf: &mut impl FnMut(&mut seismic_ir::identity::StructureDigest, &T),
        ) {
            use seismic_ir::region::Product;
            match value {
                Product::Unit => digest.bytes(b"unit"),
                Product::Leaf(value) => {
                    digest.bytes(b"leaf");
                    leaf(digest, value);
                }
                Product::Range(a, b) => {
                    digest.bytes(b"range");
                    product(digest, a, leaf);
                    product(digest, b, leaf);
                }
                Product::Tuple(values) => {
                    digest.bytes(b"tuple");
                    digest.hashed(&values.len());
                    for value in values {
                        product(digest, value, leaf);
                    }
                }
            }
        }
        fn operand(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: seismic_ir::region::ValueOperand,
        ) {
            use seismic_ir::region::ValueOperand;
            match value {
                ValueOperand::Tensor(view) => {
                    digest.bytes(b"tensor");
                    digest.hashed(&view.index());
                }
                ValueOperand::Scalar(value) => {
                    digest.bytes(b"scalar");
                    digest.hashed(&value.kind());
                }
                ValueOperand::Quantity(value) => {
                    digest.bytes(match value.kind() {
                        seismic_ir::schedule::HostQuantityKind::Natural => b"natural",
                        seismic_ir::schedule::HostQuantityKind::Integer => b"integer",
                    });
                }
            }
        }
        fn destination(
            digest: &mut seismic_ir::identity::StructureDigest,
            value: seismic_ir::region::ValueDestination,
        ) {
            use seismic_ir::region::ValueDestination;
            match value {
                ValueDestination::Tensor(view) => {
                    digest.bytes(b"tensor");
                    digest.hashed(&view.index());
                }
                ValueDestination::Scalar(slot) => {
                    digest.bytes(b"scalar");
                    digest.hashed(&slot.index());
                }
                ValueDestination::Quantity(slot) => {
                    digest.bytes(b"quantity");
                    digest.hashed(&slot.index());
                }
            }
        }
        digest.hashed(&steps.len());
        for step in steps {
            match step {
                ScheduleStep::Imported { body, .. } => {
                    digest.bytes(b"imported");
                    digest_schedule(digest, builder, body);
                }
                ScheduleStep::BeginAllocationInstance { source, result }
                | ScheduleStep::BindArgumentTensor { source, result } => {
                    digest.bytes(
                        if matches!(step, ScheduleStep::BeginAllocationInstance { .. }) {
                            b"begin-tensor"
                        } else {
                            b"argument-tensor"
                        },
                    );
                    digest.hashed(&source.index());
                    digest.hashed(&result.index());
                }
                ScheduleStep::PublishTensor { view, path, .. } => {
                    digest.bytes(b"publish-tensor");
                    digest.hashed(&view.index());
                    digest.hashed(path);
                }
                ScheduleStep::Launch(id) => {
                    digest.bytes(b"launch");
                    digest.hashed(&id.index());
                }
                ScheduleStep::Copy(copy) => {
                    digest.bytes(b"copy");
                    digest.hashed(&(copy.source.index(), copy.destination.index()));
                }
                ScheduleStep::Fill(fill) => {
                    digest.bytes(b"fill");
                    digest.hashed(&fill.destination.index());
                    digest.hashed(&fill.value);
                }
                ScheduleStep::ScalarMove(value) => {
                    digest.bytes(b"scalar-move");
                    digest.hashed(&(value.from.index(), value.to.index()));
                }
                ScheduleStep::EvaluateHost(evaluation) => {
                    digest.bytes(b"evaluate-host");
                    match evaluation.value {
                        seismic_ir::schedule::HostValueExpr::Integer(_) => digest.bytes(b"integer"),
                        seismic_ir::schedule::HostValueExpr::Natural(_) => digest.bytes(b"natural"),
                        seismic_ir::schedule::HostValueExpr::Bool(_) => digest.bytes(b"bool"),
                        seismic_ir::schedule::HostValueExpr::Word { dtype, .. } => {
                            digest.bytes(b"word");
                            digest.hashed(&dtype);
                        }
                        seismic_ir::schedule::HostValueExpr::Float { dtype, .. } => {
                            digest.bytes(b"float");
                            digest.hashed(&dtype);
                        }
                    }
                    match evaluation.to {
                        seismic_ir::schedule::HostValueDestination::Quantity(slot) => {
                            digest.bytes(b"quantity-slot");
                            digest.hashed(&slot.index());
                            digest.bytes(match slot.kind() {
                                HostQuantityKind::Integer => b"integer",
                                HostQuantityKind::Natural => b"natural",
                            });
                        }
                        seismic_ir::schedule::HostValueDestination::Native(slot) => {
                            digest.bytes(b"native-slot");
                            digest.hashed(&slot.index());
                            digest.hashed(&slot.kind());
                        }
                    }
                    if let Some(site) = &evaluation.failure {
                        digest.bytes(b"source-failure");
                        digest.hashed(&site.failure);
                        digest.bytes(site.path.as_bytes());
                        digest.hashed(&site.line);
                    }
                }
                ScheduleStep::ScalarRead(value) => {
                    digest.bytes(b"scalar-read");
                    digest.hashed(&(value.source.index(), value.index.len(), value.to.index()));
                    digest.hashed(&value.bounds.len());
                }
                ScheduleStep::Check(value) => {
                    digest.bytes(b"check");
                    digest.hashed(&value.condition.index());
                    digest.bytes(match value.expectation {
                        seismic_ir::schedule::ScalarCheckExpectation::BoolTrue => b"bool-true",
                        seismic_ir::schedule::ScalarCheckExpectation::U32Zero => b"u32-zero",
                    });
                    digest.hashed(&value.site.failure);
                    digest.bytes(value.site.path.as_bytes());
                    digest.hashed(&value.site.line);
                }
                ScheduleStep::If {
                    then_steps,
                    else_steps,
                    results,
                    ..
                } => {
                    digest.bytes(b"if");
                    product(digest, results, &mut |digest, result| {
                        operand(digest, result.then_value());
                        operand(digest, result.else_value());
                        destination(digest, result.result());
                    });
                    digest_schedule(digest, builder, then_steps);
                    digest_schedule(digest, builder, else_steps);
                }
                ScheduleStep::Repeat { body, carries, .. } => {
                    digest.bytes(b"repeat");
                    product(digest, carries, &mut |digest, carry| {
                        operand(digest, carry.initial());
                        destination(digest, carry.header());
                        operand(digest, carry.backedge());
                        destination(digest, carry.result());
                    });
                    digest_schedule(digest, builder, body);
                }
            }
        }
    }
}
