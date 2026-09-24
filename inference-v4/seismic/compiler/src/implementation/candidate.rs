//! Pure, structurally closed physical data owned by the candidate domain.
//!
//! This is materialized data, not an authored-root family definition. A
//! parameterized physical payload can serve several selected coordinates; it
//! contains no native artifacts, measured results, or evaluator output.

use crate::numerics::NumericalApplicability;
use seismic_ir::{
    kernel::KernelArena,
    schedule::{AnyScalarSlot, ParametricSchedule},
    storage::{GlobalAllocationTopology, LocalAllocationTopology},
    target::PhysicalDialect,
};
use seismic_lang::{
    expr::{AnyExpr, BoolExpr, DecisionId, SymbolKind, TargetPredicate},
    ids::StableFunctionId,
};
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MATERIALIZED_FAMILY: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChoiceKind {
    WorkgroupSize,
    AllocationSlot,
}

/// One physical axis in deterministic construction order. Arena decisions are
/// private parameters of a materialization, never part of this stable name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalChoice {
    pub ordinal: u32,
    pub kind: ChoiceKind,
}

/// One finite structural choice and the exact condition under which it is
/// semantically active. Declarations are stored in dependency order:
/// `active_when` may reference only decisions declared earlier in the family.
#[derive(Clone, Debug)]
pub struct ChoiceDeclaration {
    pub(crate) decision: DecisionId,
    pub(crate) kind: ChoiceKind,
    pub(crate) meaning: &'static str,
    pub(crate) active_when: BoolExpr,
}

impl ChoiceDeclaration {
    pub fn kind(&self) -> ChoiceKind {
        self.kind
    }
    pub fn decision(&self) -> DecisionId {
        self.decision
    }
    pub fn meaning(&self) -> &'static str {
        self.meaning
    }
    pub fn active_when(&self) -> BoolExpr {
        self.active_when
    }
}

pub(crate) fn validate_choice_declarations(
    arena: &seismic_lang::expr::ExprArena,
    choices: &[ChoiceDeclaration],
) {
    let mut earlier = HashSet::new();
    for choice in choices {
        assert!(
            !earlier.contains(&choice.decision),
            "candidate family declares one decision more than once"
        );
        arena.decision_domain(choice.decision);
        for symbol in arena.free_symbols(AnyExpr::Bool(choice.active_when)) {
            match arena.symbol_kind(symbol) {
                SymbolKind::Decision(decision) if earlier.contains(&decision) => {}
                SymbolKind::Decision(_) => {
                    panic!("choice activation references its own or a later decision")
                }
                _ => panic!("choice activation references a non-decision symbol"),
            }
        }
        earlier.insert(choice.decision);
    }
}

/// Identity of materialized physical data before native realization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstructedCandidateIdentity {
    /// Digest over the refined structure (schedule, kernels, topology, and
    /// decisions), independent of any solver assignment or native artifact.
    pub structure: [u8; 32],
}

/// Where a family's commands came from, for diagnostics and telemetry only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplementationProvenance {
    pub root: StableFunctionId,
    /// Spliced callees, in splice order.
    pub callees: Vec<StableFunctionId>,
}

/// One root result leaf in exact semantic-contract order.  Refinement seals
/// this mapping so executable translation never reconstructs result meaning.
#[derive(Clone, Debug)]
pub(crate) struct ResultPublication {
    pub path: Vec<u32>,
    pub binding: PublishedResult,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PublishedResult {
    Buffer {
        representation: seismic_lang::ids::RepresentationId,
        rank: usize,
    },
    Scalar {
        slot: AnyScalarSlot,
    },
    Quantity {
        slot: seismic_ir::schedule::HostQuantitySlot,
    },
    Range {
        start: seismic_ir::schedule::HostQuantitySlot,
        end: seismic_ir::schedule::HostQuantitySlot,
    },
}

/// A physically closed construction payload.
///
/// This owns a closed executable IR and its symbolic planning facts.  It has
/// deliberately no operation that can compile, measure, estimate, or solve.
pub struct ConstructedCandidate<B: PhysicalDialect> {
    pub(super) materialization_id: u64,
    pub(super) identity: ConstructedCandidateIdentity,
    pub(super) semantic_coverage: TargetPredicate,
    pub(super) executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub(super) native_index_bits: u32,
    pub(super) choices: Vec<ChoiceDeclaration>,
    pub(super) hard_constraints: BoolExpr,
    pub(super) numerical_applicability: NumericalApplicability,
    pub(super) provenance: ImplementationProvenance,
    pub(super) result_publications: Vec<ResultPublication>,
    pub(super) bindings: crate::portable::FrozenBindings,
}

impl<B: PhysicalDialect> ConstructedCandidate<B> {
    pub(crate) fn materialization_id(&self) -> u64 {
        self.materialization_id
    }

    pub fn identity(&self) -> &ConstructedCandidateIdentity {
        &self.identity
    }

    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.semantic_coverage
    }

    pub fn schedule(&self) -> &ParametricSchedule<B> {
        self.executable.schedule()
    }

    pub fn kernels(&self) -> &KernelArena<B> {
        self.executable.kernels()
    }

    pub fn global_allocations(&self) -> &GlobalAllocationTopology {
        self.executable.storage()
    }

    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.executable.local_allocations()
    }

    pub fn launch_resources(&self) -> &[seismic_ir::execution::LaunchResources] {
        self.executable.launch_resources()
    }

    pub fn choices(&self) -> &[ChoiceDeclaration] {
        &self.choices
    }

    pub fn hard_constraints(&self) -> BoolExpr {
        self.hard_constraints
    }

    pub fn numerical_applicability(&self) -> &NumericalApplicability {
        &self.numerical_applicability
    }

    pub fn provenance(&self) -> &ImplementationProvenance {
        &self.provenance
    }

    pub(crate) fn executable(&self) -> &seismic_ir::execution::ClosedExecutableIr<B> {
        &self.executable
    }

    #[cfg(test)]
    pub(crate) fn bindings(&self) -> &crate::portable::FrozenBindings {
        &self.bindings
    }

    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        &self.result_publications
    }

    pub(super) fn from_parts(parts: ConstructedCandidateParts<B>) -> Self {
        let materialization_id = NEXT_MATERIALIZED_FAMILY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("materialized family identity space exhausted");
        Self {
            materialization_id,
            identity: parts.identity,
            semantic_coverage: parts.semantic_coverage,
            executable: parts.executable,
            native_index_bits: parts.native_index_bits,
            choices: parts.choices,
            hard_constraints: parts.hard_constraints,
            numerical_applicability: parts.numerical_applicability,
            provenance: parts.provenance,
            result_publications: parts.result_publications,
            bindings: parts.bindings,
        }
    }

    pub(super) fn into_parts(self) -> ConstructedCandidateParts<B> {
        ConstructedCandidateParts {
            identity: self.identity,
            semantic_coverage: self.semantic_coverage,
            executable: self.executable,
            native_index_bits: self.native_index_bits,
            choices: self.choices,
            hard_constraints: self.hard_constraints,
            numerical_applicability: self.numerical_applicability,
            provenance: self.provenance,
            result_publications: self.result_publications,
            bindings: self.bindings,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_from_parts(parts: ConstructedCandidateParts<B>) -> Self {
        Self::from_parts(parts)
    }

    #[cfg(test)]
    pub(crate) fn test_into_parts(self) -> ConstructedCandidateParts<B> {
        self.into_parts()
    }
}

impl<B: PhysicalDialect> fmt::Debug for ConstructedCandidate<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConstructedCandidate")
            .field("identity", &self.identity)
            .field("launches", &self.executable.schedule().launches().len())
            .field("kernels", &self.executable.kernels().kernels().count())
            .finish_non_exhaustive()
    }
}

/// Crate-private carrier used only while a builder seals a family.
pub(crate) struct ConstructedCandidateParts<B: PhysicalDialect> {
    pub identity: ConstructedCandidateIdentity,
    pub semantic_coverage: TargetPredicate,
    pub executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub native_index_bits: u32,
    pub choices: Vec<ChoiceDeclaration>,
    pub hard_constraints: BoolExpr,
    pub numerical_applicability: NumericalApplicability,
    pub provenance: ImplementationProvenance,
    pub result_publications: Vec<ResultPublication>,
    pub bindings: crate::portable::FrozenBindings,
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::expr::{Assignment, CmpOp, ExprArena, FiniteDomain, SymbolSort, SymbolValue};

    #[test]
    fn nested_choice_is_inactive_outside_its_parent_alternative() {
        let mut arena = ExprArena::new();
        let parent = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let child_guard = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let child_leaf = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let child_guard_active = arena.bool(true);
        let child_leaf_active = arena.decision_is(child_guard, 1);
        let selected = arena.decision_is(parent, 1);
        let nested = vec![
            ChoiceDeclaration {
                kind: ChoiceKind::WorkgroupSize,
                decision: child_guard,
                meaning: "child guard",
                active_when: arena.and(selected, child_guard_active),
            },
            ChoiceDeclaration {
                kind: ChoiceKind::WorkgroupSize,
                decision: child_leaf,
                meaning: "child leaf",
                active_when: arena.and(selected, child_leaf_active),
            },
        ];
        let parent_active = arena.bool(true);
        let mut declarations = vec![ChoiceDeclaration {
            kind: crate::refinement::ChoiceKind::WorkgroupSize,
            decision: parent,
            meaning: "parent",
            active_when: parent_active,
        }];
        declarations.extend(nested);
        validate_choice_declarations(&arena, &declarations);

        let active = declarations[2].active_when;
        for (parent_value, guard_value, expected) in
            [(0, 0, false), (0, 1, false), (1, 0, false), (1, 1, true)]
        {
            let mut assignment = Assignment::new();
            assignment.bind(
                arena.decision_symbol(parent),
                SymbolValue::Int((parent_value).into()),
            );
            assignment.bind(
                arena.decision_symbol(child_guard),
                SymbolValue::Int((guard_value).into()),
            );
            assert_eq!(arena.eval_bool(active, &assignment).unwrap(), expected);
        }
    }

    #[test]
    #[should_panic(expected = "choice activation references its own or a later decision")]
    fn choice_activation_cannot_reference_a_later_decision() {
        let mut arena = ExprArena::new();
        let first = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let later = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let first_active = arena.decision_is(later, 1);
        let always = arena.bool(true);
        validate_choice_declarations(
            &arena,
            &[
                ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    decision: first,
                    meaning: "first",
                    active_when: first_active,
                },
                ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    decision: later,
                    meaning: "later",
                    active_when: always,
                },
            ],
        );
    }

    #[test]
    #[should_panic(expected = "choice activation references a non-decision symbol")]
    fn choice_activation_cannot_reference_target_truth() {
        let mut arena = ExprArena::new();
        let decision = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let (_, target_symbol) = arena.target_constant(SymbolSort::Int);
        let target_value = arena.int_symbol(target_symbol);
        let zero = arena.int(0);
        let active_when = arena.int_cmp(CmpOp::Eq, target_value, zero);
        validate_choice_declarations(
            &arena,
            &[ChoiceDeclaration {
                kind: crate::refinement::ChoiceKind::WorkgroupSize,
                decision,
                meaning: "invalid",
                active_when,
            }],
        );
    }
}
