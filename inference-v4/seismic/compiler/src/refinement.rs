//! Semantic-to-executable refinement without prediction or native realization.
//!
//! Refinement owns the finite candidate-family domain.  Its result is a pure,
//! inspectable value: one universal family, zero or more optimized families,
//! their shared expression arena, and an explicit completeness report.

mod candidate;
mod enumerate;

pub(crate) use candidate::{
    activate_choices, validate_choice_declarations, CandidateFamilyParts, ConstructionAuthority,
    PublishedResult, PublishedScalarKind, ResultPublication,
};
pub use candidate::{
    CandidateFamily, CandidateFamilyIdentity, ChoiceDeclaration, FactoryIdentity,
    ImplementationProvenance,
};

use crate::{
    errors::PreparationError,
    target::{CompilerRegistry, TargetConstants},
    PreparationBudget,
};
use seismic_ir::construction::{
    AllocationPlan, AllocationSlotChoice, AnalyzedConstruction, StoragePlannedConstruction,
};
use seismic_ir::target::KernelDialect;
use seismic_lang::{
    entry::{CallSchema, SemanticProgram},
    expr::{BoolExpr, CmpOp, DecisionId, ExprArena, FiniteDomain},
    precision::PrecisionPolicy,
};
use seismic_target::DeviceDescription;
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

/// Whether refinement should expose physical-slot reuse as a finite candidate
/// choice. Universal construction remains choice-free; optimized construction
/// explores every canonical reuse partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AllocationReusePolicy {
    Distinct,
    Explore,
}

/// The result of compiler-owned allocation-choice refinement. IR supplies the
/// structural facts and enforces the resulting plan; the compiler owns which
/// finite alternatives exist and how they are named. `constraints` is the
/// complete validity relation for these axes: canonical-label constraints plus
/// the mandatory incompatibility constraints returned by IR.
pub(crate) struct RefinedAllocationChoices<B: KernelDialect> {
    construction: StoragePlannedConstruction<B>,
    choices: Vec<(DecisionId, &'static str)>,
    constraints: Vec<BoolExpr>,
}
impl<B: KernelDialect> RefinedAllocationChoices<B> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        StoragePlannedConstruction<B>,
        Vec<(DecisionId, &'static str)>,
        Vec<BoolExpr>,
    ) {
        (self.construction, self.choices, self.constraints)
    }
}

/// Generates the finite allocation-reuse axes for one analyzed construction,
/// then submits those choices to IR's policy-neutral storage transition.
///
/// Allocation `n` chooses a slot in `0..=n`. Slot `n` is fresh; lower slots
/// refer to slots introduced by preceding allocations. By induction this
/// represents every partition of the ordered arena allocations exactly once:
/// adding allocation `n` either joins one existing block or starts one new
/// block. IR independently contributes constraints forbidding structurally
/// incompatible allocations from selecting the same slot.
pub(crate) fn refine_allocation_choices<B: KernelDialect>(
    arena: &mut ExprArena,
    analyzed: AnalyzedConstruction<B>,
    policy: AllocationReusePolicy,
) -> RefinedAllocationChoices<B> {
    if policy == AllocationReusePolicy::Distinct {
        return RefinedAllocationChoices {
            construction: analyzed.apply_allocation_plan(arena, AllocationPlan::distinct()),
            choices: Vec::new(),
            constraints: Vec::new(),
        };
    }

    let allocations = analyzed.arena_allocations().to_vec();
    let mut choices = Vec::with_capacity(allocations.len());
    let mut assignments = Vec::with_capacity(allocations.len());
    for (ordinal, allocation) in allocations.into_iter().enumerate() {
        let decision = arena.decision(canonical_slot_domain(ordinal));
        choices.push((decision, "arena physical slot"));
        assignments.push(AllocationSlotChoice::new(allocation, decision));
    }
    let mut constraints = canonical_slot_constraints(arena, &choices);
    let construction = analyzed.apply_allocation_plan(arena, AllocationPlan::new(assignments));
    constraints.extend(construction.mandatory_constraints().iter().copied());
    RefinedAllocationChoices {
        construction,
        choices,
        constraints,
    }
}

fn canonical_slot_domain(ordinal: usize) -> FiniteDomain {
    let ordinal = i64::try_from(ordinal).expect("allocation ordinal exceeds decision domain");
    FiniteDomain::new((0..=ordinal).collect())
        .expect("an arena allocation always has one canonical physical slot")
}

fn canonical_slot_constraints(
    arena: &mut ExprArena,
    choices: &[(DecisionId, &'static str)],
) -> Vec<BoolExpr> {
    let Some(&(first, _)) = choices.first() else {
        return Vec::new();
    };
    let one = arena.int(1);
    let mut maximum_previous_slot = arena.decision_value(first);
    let mut constraints = Vec::with_capacity(choices.len().saturating_sub(1));
    for &(choice, _) in &choices[1..] {
        let value = arena.decision_value(choice);
        let next_canonical_slot = arena.int_add(maximum_previous_slot, one);
        constraints.push(arena.int_cmp(CmpOp::Le, value, next_canonical_slot));
        maximum_previous_slot = arena.int_max(maximum_previous_slot, value);
    }
    constraints
}

/// Limits that belong exclusively to structural refinement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefinementLimits {
    /// Maximum number of non-universal implementation alternatives that may
    /// be constructed across the whole refinement tree, including nested
    /// call alternatives. This is deliberately not a root-family count.
    pub constructed_alternatives: u64,
    pub construction_wall_time: Duration,
}

impl From<&PreparationBudget> for RefinementLimits {
    fn from(budget: &PreparationBudget) -> Self {
        Self {
            constructed_alternatives: budget.refinement_constructed_alternatives,
            construction_wall_time: budget.refinement_construction_wall_time,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementLimit {
    ConstructedAlternativeCount,
    ConstructionWallTime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementCompletion {
    Complete,
    BudgetLimited(RefinementLimit),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefinementReport {
    pub completion: RefinementCompletion,
    /// Root optimized families actually present in the returned domain.
    pub optimized_families: u64,
    /// All non-universal alternatives constructed, including nested ones.
    pub constructed_alternatives: u64,
    /// Monotonic elapsed time for the entire refinement session. Nested work
    /// is included once because this is measured from one session start.
    pub construction_wall_time: Duration,
    /// True only when every registered factory reachable from every eligible
    /// semantic candidate was considered.  Nested refinement shares the same
    /// budget, so a nested limit makes this false even if the root loop exits
    /// normally.
    pub registered_factory_traversal_complete: bool,
}

pub struct RefinementRequest<'a, B: seismic_target::TargetFamily> {
    pub program: &'a SemanticProgram,
    pub schema: &'a CallSchema,
    pub target: &'a DeviceDescription<B>,
    pub registry: &'a CompilerRegistry<B>,
    pub constants: &'a TargetConstants,
    pub precision: &'a PrecisionPolicy,
}

/// The complete structural output of one refinement pass.  This is the only
/// candidate-domain value; `PlanSpace` consumes it rather than mirroring it.
pub struct RefinedCandidateFamilies<B: KernelDialect> {
    arena: ExprArena,
    universal: CandidateFamily<B>,
    optimized: Vec<CandidateFamily<B>>,
    report: RefinementReport,
}

impl<B: KernelDialect> RefinedCandidateFamilies<B> {
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }

    pub fn universal(&self) -> &CandidateFamily<B> {
        &self.universal
    }

    pub fn optimized(&self) -> &[CandidateFamily<B>] {
        &self.optimized
    }

    pub fn report(&self) -> &RefinementReport {
        &self.report
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ExprArena,
        CandidateFamily<B>,
        Vec<CandidateFamily<B>>,
        RefinementReport,
    ) {
        (self.arena, self.universal, self.optimized, self.report)
    }
}

pub struct RefinementSession {
    limits: RefinementLimits,
}

impl RefinementSession {
    pub fn new(limits: RefinementLimits) -> Self {
        Self { limits }
    }

    pub fn refine<B: seismic_target::TargetFamily>(
        self,
        mut arena: ExprArena,
        request: RefinementRequest<'_, B>,
    ) -> Result<RefinedCandidateFamilies<B>, PreparationError> {
        let budget = Rc::new(RefCell::new(RefinementBudgetState::new(self.limits)));
        let enumerated =
            enumerate::enumerate_candidate_families(&mut arena, request, budget.clone())?;
        let report = budget.borrow().report(
            enumerated.registered_factory_traversal_complete,
            enumerated.optimized.len() as u64,
        );
        Ok(RefinedCandidateFamilies {
            arena,
            universal: enumerated.universal,
            optimized: enumerated.optimized,
            report,
        })
    }
}

pub(crate) type RefinementBudget = Rc<RefCell<RefinementBudgetState>>;

/// Mutable state shared by nested family construction.  It intentionally has
/// no solver, native-artifact, executable, or metadata accounting.
pub(crate) struct RefinementBudgetState {
    limits: RefinementLimits,
    constructed_alternatives: u64,
    started: Instant,
    limit: Option<RefinementLimit>,
}

impl RefinementBudgetState {
    fn new(limits: RefinementLimits) -> Self {
        Self {
            limits,
            constructed_alternatives: 0,
            started: Instant::now(),
            limit: None,
        }
    }

    /// Reserves one non-universal alternative before its builder exists. Both
    /// root families and nested call alternatives use the same work-unit
    /// ceiling.
    pub(crate) fn admit_optional_implementation(&mut self) -> bool {
        if self.constructed_alternatives >= self.limits.constructed_alternatives {
            self.limit
                .get_or_insert(RefinementLimit::ConstructedAlternativeCount);
            return false;
        }
        if self.started.elapsed() >= self.limits.construction_wall_time {
            self.limit
                .get_or_insert(RefinementLimit::ConstructionWallTime);
            return false;
        }
        self.constructed_alternatives += 1;
        true
    }

    /// Checks the session deadline after an alternative completes. The
    /// parameter remains for construction call sites but is deliberately not
    /// accumulated: nested intervals overlap their parents. An alternative
    /// that crosses the deadline remains in the domain; the deadline only
    /// prevents later construction.
    pub(crate) fn record_implementation_construction(&mut self, _nested_elapsed: Duration) -> bool {
        if self.started.elapsed() >= self.limits.construction_wall_time {
            self.limit
                .get_or_insert(RefinementLimit::ConstructionWallTime);
            return false;
        }
        self.limit.is_none()
    }

    fn report(&self, root_traversal_complete: bool, optimized_families: u64) -> RefinementReport {
        RefinementReport {
            completion: self.limit.map_or(
                RefinementCompletion::Complete,
                RefinementCompletion::BudgetLimited,
            ),
            optimized_families,
            constructed_alternatives: self.constructed_alternatives,
            construction_wall_time: self.started.elapsed(),
            registered_factory_traversal_complete: root_traversal_complete && self.limit.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_slot_constraints, canonical_slot_domain, refine_allocation_choices,
        AllocationReusePolicy, RefinementBudgetState, RefinementCompletion, RefinementLimit,
        RefinementLimits,
    };
    use seismic_ir::{
        construction::Construction,
        kernel::ops::AddressableResourceHandle,
        repr::{DenseF32, Representation},
        storage::GlobalBufferKind,
        target::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelDialect},
    };
    use seismic_lang::expr::{Assignment, ExprArena, SymbolValue};
    use std::{collections::BTreeSet, time::Duration};

    type Partition = Vec<Vec<usize>>;

    fn independent_partitions(elements: usize) -> BTreeSet<Partition> {
        let mut partitions = vec![Vec::<Vec<usize>>::new()];
        for element in 0..elements {
            let mut next = Vec::new();
            for partition in partitions {
                for block in 0..partition.len() {
                    let mut extended = partition.clone();
                    extended[block].push(element);
                    next.push(extended);
                }
                let mut extended = partition;
                extended.push(vec![element]);
                next.push(extended);
            }
            partitions = next;
        }
        partitions.into_iter().collect()
    }

    fn assignments(domains: &[Vec<i64>]) -> Vec<Vec<i64>> {
        fn extend(domains: &[Vec<i64>], at: usize, values: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
            if at == domains.len() {
                out.push(values.clone());
                return;
            }
            for &value in &domains[at] {
                values.push(value);
                extend(domains, at + 1, values, out);
                values.pop();
            }
        }
        let mut out = Vec::new();
        extend(domains, 0, &mut Vec::new(), &mut out);
        out
    }

    fn partition_from_labels(labels: &[i64]) -> Partition {
        let Some(maximum) = labels.iter().copied().max() else {
            return Vec::new();
        };
        let mut partition = vec![Vec::new(); usize::try_from(maximum + 1).unwrap()];
        for (element, &label) in labels.iter().enumerate() {
            partition[usize::try_from(label).unwrap()].push(element);
        }
        assert!(partition.iter().all(|block| !block.is_empty()));
        partition
    }

    fn represented_partitions(
        arena: &ExprArena,
        choices: &[(seismic_lang::expr::DecisionId, &'static str)],
        constraints: &[seismic_lang::expr::BoolExpr],
    ) -> BTreeSet<Partition> {
        let domains = choices
            .iter()
            .map(|(choice, _)| arena.decision_domain(*choice).values().to_vec())
            .collect::<Vec<_>>();
        let mut represented = BTreeSet::new();
        for values in assignments(&domains) {
            let mut assignment = Assignment::new();
            for ((choice, _), &value) in choices.iter().zip(&values) {
                assignment.bind(arena.decision_symbol(*choice), SymbolValue::Int(value));
            }
            if constraints
                .iter()
                .all(|constraint| arena.eval_bool(*constraint, &assignment).unwrap())
            {
                assert!(
                    represented.insert(partition_from_labels(&values)),
                    "two assignments represented the same physical partition"
                );
            }
        }
        represented
    }

    #[test]
    fn canonical_slot_axes_offer_every_existing_slot_and_one_fresh_slot() {
        assert_eq!(canonical_slot_domain(0).values(), &[0]);
        assert_eq!(canonical_slot_domain(1).values(), &[0, 1]);
        assert_eq!(canonical_slot_domain(4).values(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn canonical_slot_constraints_remove_duplicate_partition_labels() {
        let mut arena = ExprArena::new();
        let first = arena.decision(canonical_slot_domain(0));
        let second = arena.decision(canonical_slot_domain(1));
        let third = arena.decision(canonical_slot_domain(2));
        let choices = [(first, "slot"), (second, "slot"), (third, "slot")];
        let constraints = canonical_slot_constraints(&mut arena, &choices);

        let assignment = |second_value, third_value| {
            let mut assignment = Assignment::new();
            assignment.bind(arena.decision_symbol(first), SymbolValue::Int(0));
            assignment.bind(
                arena.decision_symbol(second),
                SymbolValue::Int(second_value),
            );
            assignment.bind(arena.decision_symbol(third), SymbolValue::Int(third_value));
            assignment
        };
        assert!(constraints
            .iter()
            .all(|constraint| arena.eval_bool(*constraint, &assignment(1, 2)).unwrap()));
        assert!(!constraints
            .iter()
            .all(|constraint| arena.eval_bool(*constraint, &assignment(0, 2)).unwrap()));
    }

    #[test]
    fn canonical_slot_encoding_matches_every_partition_exactly_once_through_six_allocations() {
        const BELL: [usize; 7] = [1, 1, 2, 5, 15, 52, 203];
        for allocations in 0..=6 {
            let mut arena = ExprArena::new();
            let choices = (0..allocations)
                .map(|ordinal| {
                    (
                        arena.decision(canonical_slot_domain(ordinal)),
                        "physical slot",
                    )
                })
                .collect::<Vec<_>>();
            let constraints = canonical_slot_constraints(&mut arena, &choices);
            let represented = represented_partitions(&arena, &choices, &constraints);
            let expected = independent_partitions(allocations);
            assert_eq!(represented, expected, "allocation count {allocations}");
            assert_eq!(represented.len(), BELL[allocations]);
        }
    }

    #[derive(Debug)]
    struct TestDialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    impl KernelDialect for TestDialect {
        const NAME: seismic_lang::registry::BackendName = seismic_lang::registry::BackendName::Cpu;
        type Facts = ();
        type Intrinsic = NoIntrinsic;

        fn write_intrinsic_identity(op: &Self::Intrinsic, _: &mut IntrinsicIdentityBuilder) {
            match *op {}
        }
        fn intrinsic_numerics(
            _: &Self::Facts,
            _: &seismic_lang::registry::IntrinsicSignature,
            op: &Self::Intrinsic,
        ) -> IntrinsicNumericalSemantics {
            match *op {}
        }
        fn intrinsic_addressable_resources(op: &Self::Intrinsic) -> Vec<AddressableResourceHandle> {
            match *op {}
        }
    }

    #[test]
    fn ir_incompatibility_constraints_retain_exactly_the_compatible_partitions() {
        let mut arena = ExprArena::new();
        let mut construction = Construction::<TestDialect>::new(&mut arena, vec![], false, 0);
        let length = arena.nat(8);
        let views = (0..3)
            .map(|_| {
                let (_, index) = construction.storage_mut().tensor(
                    &mut arena,
                    GlobalBufferKind::Arena,
                    DenseF32::id(),
                    vec![length],
                );
                let view = construction.view(index, DenseF32::id());
                construction.typed_view::<DenseF32>(view)
            })
            .collect::<Vec<_>>();
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_zero(views[0]);
        schedule.fill_zero(views[1]);
        schedule.fill_zero(views[0]);
        schedule.fill_zero(views[2]);
        let closed = schedule.close();
        let analyzed = construction.close(closed).analyze_allocations();
        assert_eq!(
            analyzed
                .allocation_relations()
                .iter()
                .map(|relation| relation.may_share_slot())
                .collect::<Vec<_>>(),
            vec![false, true, true]
        );

        let (_planned, choices, constraints) =
            refine_allocation_choices(&mut arena, analyzed, AllocationReusePolicy::Explore)
                .into_parts();
        let represented = represented_partitions(&arena, &choices, &constraints);
        let expected = independent_partitions(3)
            .into_iter()
            .filter(|partition| {
                !partition
                    .iter()
                    .any(|block| block.contains(&0) && block.contains(&1))
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(represented, expected);
        assert_eq!(represented.len(), 3);
    }

    #[test]
    fn refinement_budget_reports_incomplete_factory_traversal() {
        let mut budget = RefinementBudgetState::new(RefinementLimits {
            constructed_alternatives: 0,
            construction_wall_time: Duration::from_secs(1),
        });
        assert!(!budget.admit_optional_implementation());
        let report = budget.report(true, 0);
        assert_eq!(
            report.completion,
            RefinementCompletion::BudgetLimited(RefinementLimit::ConstructedAlternativeCount)
        );
        assert_eq!(report.optimized_families, 0);
        assert_eq!(report.constructed_alternatives, 0);
        assert!(report.construction_wall_time < Duration::from_secs(1));
        assert!(!report.registered_factory_traversal_complete);
    }

    #[test]
    fn completed_alternative_is_retained_when_session_deadline_is_reached() {
        let mut budget = RefinementBudgetState::new(RefinementLimits {
            constructed_alternatives: 2,
            construction_wall_time: Duration::ZERO,
        });
        // Simulate an alternative that was admitted just before its deadline.
        budget.constructed_alternatives = 1;
        assert!(!budget.record_implementation_construction(Duration::from_secs(99)));
        let report = budget.report(true, 1);
        assert_eq!(
            report.completion,
            RefinementCompletion::BudgetLimited(RefinementLimit::ConstructionWallTime)
        );
        assert_eq!(report.optimized_families, 1);
        assert_eq!(report.constructed_alternatives, 1);
        assert!(report.construction_wall_time >= Duration::ZERO);
        assert!(!report.registered_factory_traversal_complete);
        assert!(!budget.admit_optional_implementation());
    }
}
