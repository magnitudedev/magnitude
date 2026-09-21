//! `PlanSpace<B>`: closed executable alternatives plus one exact constraint
//! system (spec §2.1, §6.1, §10).
//!
//! `plan_space` consumes the logical entry, binds the opened machine into
//! the entry's arena, asks every registered factory (and the universal
//! portable factory) for closed implementations of the root function, and
//! builds one solver model over all of them together with the target domain
//! and numerical admissibility. Every member already owns reconciled native
//! kernels; it contains no partial proposals, labels, or alternate route.
//!
//! W6 owns the internals.

use crate::errors::PreparationError;
use crate::expression::{InvocationExpr, PlanningExpr};
use crate::implementation::{OptimizedImplementation, UniversalImplementation};
use crate::numerics::EvidenceCatalog;
use crate::preparation_budget::{PreparationBudget, PreparationBudgetTracker};
use crate::solve::SolverModel;
use crate::target::{
    Backend, DeviceContractIdentity, ExecutionProfileIdentity, NumericalEnvironmentIdentity,
    PlanningMachine, TargetConstants,
};
use seismic_lang::entry::{CallSchema, LogicalEntry, SemanticEventManifest};
use seismic_lang::expr::{BoolExpr, EntryPredicate, ExprArena};
use seismic_lang::ids::{ModuleHash, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use std::sync::Arc;

/// The target domain (§10.1): entry domain intersected with target integer
/// and address representability and backend call-schema representability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetDomain {
    predicate: EntryPredicate,
    identity: [u8; 32],
}

impl TargetDomain {
    pub(crate) fn new(predicate: EntryPredicate, identity: [u8; 32]) -> Self {
        Self {
            predicate,
            identity,
        }
    }
    pub fn predicate(&self) -> EntryPredicate {
        self.predicate
    }
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Non-empty vector with a private constructor.
#[derive(Debug)]
pub struct NonEmpty<T> {
    items: Vec<T>,
}

impl<T> NonEmpty<T> {
    pub(crate) fn new(items: Vec<T>) -> Option<Self> {
        (!items.is_empty()).then_some(Self { items })
    }
    pub fn first(&self) -> &T {
        &self.items[0]
    }
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }
}

/// The plan space of one entry on one target under one precision policy.
#[derive(Debug)]
pub struct PlanSpace<B: Backend> {
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    semantic_events: Arc<SemanticEventManifest>,
    target_domain: TargetDomain,
    constants: TargetConstants,
    device: DeviceContractIdentity,
    execution: ExecutionProfileIdentity,
    target: NumericalEnvironmentIdentity,
    evidence: Arc<EvidenceCatalog>,
    arena: ExprArena,
    universal: UniversalImplementation<B>,
    optimized: Vec<PlanCandidate<B>>,
    model: SolverModel,
    precision: PrecisionPolicy,
    budget: PreparationBudgetTracker,
    optimization_exhausted: bool,
}

impl<B: Backend> PlanSpace<B> {
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &Arc<CallSchema> {
        &self.schema
    }
    pub fn semantic_event_manifest(&self) -> &Arc<SemanticEventManifest> {
        &self.semantic_events
    }
    pub fn target_domain(&self) -> TargetDomain {
        self.target_domain
    }
    pub fn constants(&self) -> &TargetConstants {
        &self.constants
    }
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }
    pub fn model(&self) -> &SolverModel {
        &self.model
    }
    pub fn precision(&self) -> &PrecisionPolicy {
        &self.precision
    }

    pub(crate) fn from_parts(parts: PlanSpaceParts<B>) -> Self {
        Self {
            entry: parts.entry,
            module: parts.module,
            schema: parts.schema,
            semantic_events: parts.semantic_events,
            target_domain: parts.target_domain,
            constants: parts.constants,
            device: parts.device,
            execution: parts.execution,
            target: parts.target,
            evidence: parts.evidence,
            arena: parts.arena,
            universal: parts.universal,
            optimized: parts.optimized,
            model: parts.model,
            precision: parts.precision,
            budget: parts.budget,
            optimization_exhausted: parts.optimization_exhausted,
        }
    }

    /// Consumes the space for freezing: the arena becomes shared and
    /// immutable.
    pub(crate) fn into_frozen_parts(self) -> FrozenSpace<B> {
        FrozenSpace {
            entry: self.entry,
            module: self.module,
            schema: self.schema,
            semantic_events: self.semantic_events,
            target_domain: self.target_domain,
            constants: self.constants,
            device: self.device,
            execution: self.execution,
            target: self.target,
            evidence: self.evidence,
            arena: Arc::new(self.arena),
            universal: self.universal,
            optimized: self.optimized,
            model: self.model,
            precision: self.precision,
            budget: self.budget,
            optimization_exhausted: self.optimization_exhausted,
        }
    }
}

pub(crate) struct PlanSpaceParts<B: Backend> {
    pub entry: StableEntryId,
    pub module: ModuleHash,
    pub schema: Arc<CallSchema>,
    pub semantic_events: Arc<SemanticEventManifest>,
    pub target_domain: TargetDomain,
    pub constants: TargetConstants,
    pub device: DeviceContractIdentity,
    pub execution: ExecutionProfileIdentity,
    pub target: NumericalEnvironmentIdentity,
    pub evidence: Arc<EvidenceCatalog>,
    pub arena: ExprArena,
    pub universal: UniversalImplementation<B>,
    pub optimized: Vec<PlanCandidate<B>>,
    pub model: SolverModel,
    pub precision: PrecisionPolicy,
    pub budget: PreparationBudgetTracker,
    pub optimization_exhausted: bool,
}

/// A plan space after its arena is frozen for the coverage loop.
#[derive(Debug)]
pub struct FrozenSpace<B: Backend> {
    pub(crate) entry: StableEntryId,
    pub(crate) module: ModuleHash,
    pub(crate) schema: Arc<CallSchema>,
    pub(crate) semantic_events: Arc<SemanticEventManifest>,
    pub(crate) target_domain: TargetDomain,
    pub(crate) constants: TargetConstants,
    pub(crate) device: DeviceContractIdentity,
    pub(crate) execution: ExecutionProfileIdentity,
    pub(crate) target: NumericalEnvironmentIdentity,
    pub(crate) evidence: Arc<EvidenceCatalog>,
    pub(crate) arena: Arc<ExprArena>,
    pub(crate) universal: UniversalImplementation<B>,
    pub(crate) optimized: Vec<PlanCandidate<B>>,
    pub(crate) model: SolverModel,
    pub(crate) precision: PrecisionPolicy,
    pub(crate) budget: PreparationBudgetTracker,
    pub(crate) optimization_exhausted: bool,
}

/// One closed implementation together with the complete planning predicate
/// that admits it.  Keeping these facts in one object prevents the solver,
/// freezer and coverage builder from joining parallel tables by index.
#[derive(Debug)]
pub(crate) struct PlanCandidate<B: Backend> {
    pub(crate) implementation: OptimizedImplementation<B>,
    /// `TargetDomain && semantic coverage && hard constraints && numerical
    /// admissibility`, before target/decision binding.
    pub(crate) guard: InvocationExpr<BoolExpr>,
}

/// The consuming transition `LogicalEntry -> PlanSpace<B>` (§2.2).
pub fn plan_space<B: Backend>(
    entry: LogicalEntry,
    machine: PlanningMachine<'_, B>,
    precision: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    budget: &PreparationBudget,
) -> Result<PlanSpace<B>, PreparationError> {
    internals::plan_space(entry, machine, precision, evidence, budget)
}

mod internals {
    use super::*;
    use crate::implementation::{construct_root_implementations, ConstructionBudget};
    use crate::numerics;
    use crate::solve::SolverModelBuilder;
    use seismic_lang::entry::{ParameterKind, ResultKind};
    use seismic_lang::expr::{AnyExpr, CmpOp, NodeView, RootName};

    pub(super) fn plan_space<B: Backend>(
        entry: LogicalEntry,
        machine: PlanningMachine<'_, B>,
        precision: &PrecisionPolicy,
        evidence: &EvidenceCatalog,
        budget: &PreparationBudget,
    ) -> Result<PlanSpace<B>, PreparationError> {
        let target = machine.device();
        let semantic_events = Arc::new(entry.semantic_event_manifest());
        let seismic_lang::entry::LogicalEntryParts {
            identity,
            module,
            schema,
            domain,
            mut arena,
            program,
        } = entry.into_parts();

        // Target facts enter the arena exactly once, before any
        // implementation is constructed. Every later physical constraint
        // therefore refers to this single set of symbols and bindings.
        let constants = target.bind_constants(&mut arena);
        let target_domain = self::target_domain(&mut arena, &schema, domain, target)?;
        let construction_budget: ConstructionBudget = std::rc::Rc::new(std::cell::RefCell::new(
            PreparationBudgetTracker::new(budget.clone()),
        ));
        let optimization_exhausted = std::cell::Cell::new(false);
        let mut closed_universal = None;
        let mut closed_optimized = Vec::new();
        construct_root_implementations(
            &mut arena,
            &program,
            &schema,
            machine,
            &constants,
            precision,
            construction_budget.clone(),
            |draft, arena| {
                let templates = draft.native_template_identities(target);
                let within_templates = construction_budget
                    .borrow_mut()
                    .record_required_native_templates(&templates)?;
                let (implementation, metrics) =
                    draft.close_native(arena, target, machine.execution(), &constants)?;
                let retained_bytes = implementation.retained_metadata_bytes();
                let mut tracker = construction_budget.borrow_mut();
                let within_artifact = tracker.record_required_native_artifact(metrics)?;
                let within_metadata = tracker.record_required_metadata(retained_bytes)?;
                optimization_exhausted
                    .set(!within_templates || !within_artifact || !within_metadata);
                drop(tracker);
                closed_universal = Some(implementation);
                Ok(())
            },
            |draft, arena| {
                if optimization_exhausted.get() {
                    return Ok(false);
                }
                let templates = draft.native_template_identities(target);
                let within_templates = construction_budget
                    .borrow_mut()
                    .record_native_templates(&templates);
                let (implementation, metrics) =
                    draft.close_native(arena, target, machine.execution(), &constants)?;
                let retained_bytes = implementation.retained_metadata_bytes();
                let mut tracker = construction_budget.borrow_mut();
                let within_artifact = tracker.record_native_artifact(metrics);
                let within_metadata = tracker.charge_metadata(retained_bytes);
                drop(tracker);
                // Exact charges happen immediately after close. A candidate
                // whose exact charge reaches the ceiling is retained, then
                // enumeration stops; there is no build-then-drop path.
                closed_optimized.push(OptimizedImplementation::from_closed(implementation));
                if !within_templates || !within_artifact || !within_metadata {
                    optimization_exhausted.set(true);
                }
                Ok(!optimization_exhausted.get())
            },
        )?;
        let tracker = construction_budget.borrow().clone();
        let universal = closed_universal
            .unwrap_or_else(|| panic!("root enumeration did not retain its typed universal"));
        let universal = UniversalImplementation::from_closed_reference(
            universal,
            &mut arena,
            target_domain.predicate().node(),
            &constants,
        )?;

        let target_node = target_domain.predicate().node();
        let mut optimized = Vec::with_capacity(closed_optimized.len());
        let mut selections = Vec::with_capacity(closed_optimized.len());
        for implementation in closed_optimized {
            let numerical = numerics::admissibility(
                &mut arena,
                implementation.as_inner().numerical_transfer(),
                precision,
                evidence,
                implementation.as_inner().identity(),
                implementation.as_inner().decisions(),
                machine.device().numerical_identity(),
                implementation.as_inner().native_numerical_identity(),
                target_domain.identity(),
            );
            let semantic = implementation.as_inner().semantic_coverage().node();
            // TargetDomain may contain scalar preconditions. Those are
            // invocation facts, not compile-time decisions, and are
            // validated before selection. Keep them in the executable
            // guard while exporting only target/decision/shape facts to
            // the finite solver.
            // The solver must choose only a physically admissible
            // implementation. Keeping hard constraints solely in the
            // runtime guard recreates the forbidden split-brain path in
            // which search selects a plan that execution must reject.
            let semantic_and_numerical = arena.and(semantic, numerical);
            let qualified = arena.all(implementation.as_inner().duration_qualification());
            let legal = arena.and(implementation.as_inner().hard_constraints(), qualified);
            let selection = arena.and(semantic_and_numerical, legal);
            let guard = arena.all(&[target_node, selection]);
            selections.push(selection);
            optimized.push(PlanCandidate {
                implementation,
                guard: invocation_projection(&mut arena, guard),
            });
        }

        let planning_selections = selections
            .into_iter()
            .map(|selection| planning_projection(&mut arena, selection))
            .collect::<Vec<_>>();
        let mut builder = SolverModelBuilder::new(&mut arena);
        for (symbol, value) in constants.bindings() {
            builder.bind_target(*symbol, *value);
        }
        for (index, (candidate, planning)) in optimized.iter().zip(planning_selections).enumerate()
        {
            builder.implementation(
                (index + 1) as u32,
                candidate.implementation.as_inner(),
                planning,
            );
        }
        let model = builder.build();

        Ok(PlanSpace::from_parts(PlanSpaceParts {
            entry: identity,
            module,
            schema: Arc::new(schema),
            semantic_events,
            target_domain,
            constants,
            device: machine.device().identity().clone(),
            execution: machine.execution().identity().clone(),
            target: machine.device().numerical_identity().clone(),
            evidence: Arc::new(evidence.clone()),
            arena,
            universal,
            optimized,
            model,
            precision: precision.clone(),
            budget: tracker,
            optimization_exhausted: optimization_exhausted.get(),
        }))
    }

    fn planning_projection(arena: &mut ExprArena, predicate: BoolExpr) -> PlanningExpr<BoolExpr> {
        let mut terms = Vec::new();
        collect_planning_terms(arena, predicate, &mut terms);
        let node = arena.all(&terms);
        PlanningExpr::new(arena, node)
            .unwrap_or_else(|| panic!("planning projection retained an invocation expression"))
    }

    fn invocation_projection(
        arena: &mut ExprArena,
        predicate: BoolExpr,
    ) -> InvocationExpr<BoolExpr> {
        let mut terms = Vec::new();
        collect_invocation_terms(arena, predicate, &mut terms);
        let node = arena.all(&terms);
        InvocationExpr::new(arena, node)
            .unwrap_or_else(|| panic!("invocation projection retained a schedule expression"))
    }

    fn collect_invocation_terms(
        arena: &ExprArena,
        predicate: BoolExpr,
        output: &mut Vec<BoolExpr>,
    ) {
        match arena.view(AnyExpr::Bool(predicate)) {
            NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::And,
                lhs: AnyExpr::Bool(left),
                rhs: AnyExpr::Bool(right),
            } => {
                collect_invocation_terms(arena, left, output);
                collect_invocation_terms(arena, right, output);
            }
            NodeView::Nary {
                op: seismic_lang::expr::NaryOp::All,
                operands,
            } => {
                for operand in operands {
                    let AnyExpr::Bool(term) = *operand else {
                        panic!("Boolean conjunction contains a non-Boolean operand")
                    };
                    collect_invocation_terms(arena, term, output);
                }
            }
            _ if InvocationExpr::new(arena, predicate).is_some() => output.push(predicate),
            _ => {}
        }
    }

    fn collect_planning_terms(arena: &ExprArena, predicate: BoolExpr, output: &mut Vec<BoolExpr>) {
        match arena.view(AnyExpr::Bool(predicate)) {
            NodeView::Binary {
                op: seismic_lang::expr::BinaryOp::And,
                lhs: AnyExpr::Bool(left),
                rhs: AnyExpr::Bool(right),
            } => {
                collect_planning_terms(arena, left, output);
                collect_planning_terms(arena, right, output);
            }
            NodeView::Nary {
                op: seismic_lang::expr::NaryOp::All,
                operands,
            } => {
                for operand in operands {
                    let AnyExpr::Bool(term) = *operand else {
                        panic!("Boolean conjunction contains a non-Boolean operand")
                    };
                    collect_planning_terms(arena, term, output);
                }
            }
            _ if PlanningExpr::new(arena, predicate).is_some() => output.push(predicate),
            _ => {}
        }
    }

    fn target_domain<B: Backend>(
        arena: &mut ExprArena,
        schema: &CallSchema,
        entry: seismic_lang::entry::EntryDomain,
        target: &crate::target::DeviceContract<B>,
    ) -> Result<TargetDomain, PreparationError> {
        let max_index = if target.limits().max_index_bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << target.limits().max_index_bits) - 1
        };
        let max_index = arena.nat(max_index);
        let max_allocation = arena.nat(target.limits().max_allocation_bytes);
        let mut terms = vec![entry.predicate().node()];

        let mut tensor = |representation,
                          axes: &[seismic_lang::expr::NatExpr]|
         -> Result<(), PreparationError> {
            if !target.dtypes().representations.contains(&representation) {
                return Err(PreparationError::TargetDomainUnrepresentable(format!(
                    "target {:?} does not support representation `{}`",
                    B::NAME,
                    seismic_lang::registry::representation_info(representation).name,
                )));
            }
            for axis in axes {
                terms.push(arena.nat_cmp(CmpOp::Le, *axis, max_index));
                terms.push(arena.side_conditions(AnyExpr::Nat(*axis)));
            }
            let bytes = crate::storage::tensor_bytes(arena, representation, axes);
            terms.push(arena.side_conditions(AnyExpr::Nat(bytes)));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_allocation));
            terms.push(arena.nat_cmp(CmpOp::Le, bytes, max_index));
            Ok(())
        };
        for parameter in schema.parameters() {
            if let ParameterKind::Tensor {
                representation,
                axes,
                ..
            } = &parameter.kind
            {
                tensor(*representation, axes)?;
            }
        }
        for result in schema.results() {
            if let ResultKind::Tensor {
                representation,
                axes,
            } = &result.kind
            {
                tensor(*representation, axes)?;
            }
        }
        let predicate = arena.all(&terms);
        let predicate = seismic_lang::expr::EntryPredicate::new(arena, predicate)
            .unwrap_or_else(|_| panic!("target-domain construction retained a compiler symbol"));
        let root = arena.root(RootName::Guard, AnyExpr::Bool(predicate.node()));
        let identity = arena.canonical_digest(&[root]).bytes();
        Ok(TargetDomain::new(predicate, identity))
    }
}
