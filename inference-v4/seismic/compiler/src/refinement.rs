//! Closed physical candidate data and allocation-choice refinement.
//!
//! Candidate definitions and resumable construction belong to CandidateDomain.
//! This module supplies the physical storage transition and immutable payload.

pub(crate) use crate::implementation::candidate::{
    validate_choice_declarations, PublishedResult,
};
#[cfg(test)]
pub(crate) use crate::implementation::candidate::ConstructedCandidateParts;
pub use crate::implementation::candidate::{
    ChoiceDeclaration, ChoiceKind, ConstructedCandidate, ConstructedCandidateIdentity,
    ImplementationProvenance, PhysicalChoice,
};

use seismic_ir::construction::{
    AllocationPlan, AllocationSlotChoice, AnalyzedConstruction, StoragePlannedConstruction,
};
use seismic_ir::physical_target::PhysicalDialect;
use seismic_lang::expr::{BoolExpr, CmpOp, DecisionId, ExprArena, FiniteDomain};

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
pub(crate) struct RefinedAllocationChoices<B: PhysicalDialect> {
    construction: StoragePlannedConstruction<B>,
    choices: Vec<(DecisionId, &'static str)>,
    constraints: Vec<BoolExpr>,
}
impl<B: PhysicalDialect> RefinedAllocationChoices<B> {
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
pub(crate) fn refine_allocation_choices<B: PhysicalDialect>(
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

#[cfg(test)]
mod tests {
    use super::{
        canonical_slot_constraints, canonical_slot_domain, refine_allocation_choices,
        AllocationReusePolicy,
    };
    use seismic_ir::{
        construction::Construction,
        kernel::ops::AddressableResourceHandle,
        repr::{DenseF32, Representation},
        storage::GlobalBufferKind,
        physical_target::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, PhysicalDialect},
    };
    use seismic_lang::expr::{Assignment, ExprArena, SymbolValue};
    use std::collections::BTreeSet;

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
                assignment.bind(arena.decision_symbol(*choice), SymbolValue::Int((value).into()));
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

        let assignment = |second_value: i64, third_value: i64| {
            let mut assignment = Assignment::new();
            assignment.bind(arena.decision_symbol(first), SymbolValue::Int((0).into()));
            assignment.bind(
                arena.decision_symbol(second),
                SymbolValue::Int((second_value).into()),
            );
            assignment.bind(arena.decision_symbol(third), SymbolValue::Int((third_value).into()));
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
    impl PhysicalDialect for TestDialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() -> Self::LaunchDescriptor {
            ()
        }

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
                construction.view(index, DenseF32::id())
            })
            .collect::<Vec<_>>();
        let mut schedule = construction.schedule(&mut arena, 0);
        for view in [views[0], views[1], views[0], views[2]] {
            schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
        }
        let closed = schedule.close();
        let analyzed = construction
            .close(closed)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
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
}
