//! Pure, structurally closed candidate families produced by refinement.
//!
//! A candidate family contains every symbolic choice and constraint needed by
//! later planning, but it contains no native artifacts, measured results, or
//! evaluator output.  Refinement is therefore inspectable without invoking a
//! backend toolchain.

use crate::numerics::NumericalTransfer;
use seismic_ir::{
    kernel::KernelArena,
    schedule::{AnyScalarSlot, ParametricSchedule},
    storage::{AnyBufferView, GlobalAllocationTopology, LocalAllocationTopology},
    target::KernelDialect,
};
use seismic_lang::{
    expr::{AnyExpr, BoolExpr, DecisionId, NatExpr, SymbolKind, TargetPredicate},
    ids::StableFunctionId,
    types::DType,
};
use std::collections::HashSet;
use std::fmt;

/// One finite structural choice and the exact condition under which it is
/// semantically active. Declarations are stored in dependency order:
/// `active_when` may reference only decisions declared earlier in the family.
#[derive(Clone, Debug)]
pub struct ChoiceDeclaration {
    pub(crate) decision: DecisionId,
    pub(crate) meaning: &'static str,
    pub(crate) active_when: BoolExpr,
}

impl ChoiceDeclaration {
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

pub(crate) fn activate_choices(
    arena: &mut seismic_lang::expr::ExprArena,
    parent: BoolExpr,
    choices: Vec<ChoiceDeclaration>,
) -> Vec<ChoiceDeclaration> {
    choices
        .into_iter()
        .map(|choice| ChoiceDeclaration {
            decision: choice.decision,
            meaning: choice.meaning,
            active_when: arena.and(parent, choice.active_when),
        })
        .collect()
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

/// Stable identity of one candidate family before native realization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CandidateFamilyIdentity {
    pub factory: FactoryIdentity,
    /// Digest over the refined structure (schedule, kernels, topology, and
    /// decisions), independent of any solver assignment or native artifact.
    pub structure: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FactoryIdentity {
    pub name: &'static str,
    pub revision: &'static str,
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
        view: AnyBufferView,
        bytes: NatExpr,
    },
    Scalar {
        slot: AnyScalarSlot,
        kind: PublishedScalarKind,
    },
    Range {
        start: AnyScalarSlot,
        end: AnyScalarSlot,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PublishedScalarKind {
    Value(DType),
    Index,
}

/// Constructional coverage class carried explicitly into realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ConstructionAuthority {
    UniversalPortable,
    Optimized,
}

/// The pure output unit of refinement.
///
/// This owns a closed executable IR and its symbolic planning facts.  It has
/// deliberately no operation that can compile, measure, estimate, or solve.
pub struct CandidateFamily<B: KernelDialect> {
    pub(crate) authority: ConstructionAuthority,
    pub(crate) numerical_role: seismic_lang::entry::NumericalRole,
    pub(crate) identity: CandidateFamilyIdentity,
    pub(crate) semantic_coverage: TargetPredicate,
    pub(crate) executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub(crate) launch_scratch: Vec<seismic_ir::storage::LaunchScratchRequirements>,
    pub(crate) launch_abi: Vec<Vec<seismic_ir::storage::LaunchAbiRequirement>>,
    pub(crate) choices: Vec<ChoiceDeclaration>,
    pub(crate) hard_constraints: BoolExpr,
    pub(crate) numerical_transfer: NumericalTransfer,
    pub(crate) provenance: ImplementationProvenance,
    pub(crate) result_publications: Vec<ResultPublication>,
}

impl<B: KernelDialect> CandidateFamily<B> {
    pub fn identity(&self) -> &CandidateFamilyIdentity {
        &self.identity
    }

    pub fn semantic_coverage(&self) -> TargetPredicate {
        self.semantic_coverage
    }

    pub fn schedule(&self) -> &ParametricSchedule {
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

    pub fn launch_layouts(&self) -> &[seismic_ir::storage::LaunchLocalLayout] {
        self.executable.launch_layouts()
    }

    pub fn launch_scratch(&self) -> &[seismic_ir::storage::LaunchScratchRequirements] {
        &self.launch_scratch
    }

    pub fn launch_abi(&self) -> &[Vec<seismic_ir::storage::LaunchAbiRequirement>] {
        &self.launch_abi
    }

    pub fn choices(&self) -> &[ChoiceDeclaration] {
        &self.choices
    }

    pub fn hard_constraints(&self) -> BoolExpr {
        self.hard_constraints
    }

    pub fn numerical_transfer(&self) -> &NumericalTransfer {
        &self.numerical_transfer
    }

    pub fn provenance(&self) -> &ImplementationProvenance {
        &self.provenance
    }

    pub(crate) fn result_publications(&self) -> &[ResultPublication] {
        &self.result_publications
    }

    pub(crate) fn from_parts(parts: CandidateFamilyParts<B>) -> Self {
        Self {
            authority: parts.authority,
            numerical_role: parts.numerical_role,
            identity: parts.identity,
            semantic_coverage: parts.semantic_coverage,
            executable: parts.executable,
            launch_scratch: parts.launch_scratch,
            launch_abi: parts.launch_abi,
            choices: parts.choices,
            hard_constraints: parts.hard_constraints,
            numerical_transfer: parts.numerical_transfer,
            provenance: parts.provenance,
            result_publications: parts.result_publications,
        }
    }

    pub(crate) fn into_parts(self) -> CandidateFamilyParts<B> {
        CandidateFamilyParts {
            authority: self.authority,
            numerical_role: self.numerical_role,
            identity: self.identity,
            semantic_coverage: self.semantic_coverage,
            executable: self.executable,
            launch_scratch: self.launch_scratch,
            launch_abi: self.launch_abi,
            choices: self.choices,
            hard_constraints: self.hard_constraints,
            numerical_transfer: self.numerical_transfer,
            provenance: self.provenance,
            result_publications: self.result_publications,
        }
    }
}

impl<B: KernelDialect> fmt::Debug for CandidateFamily<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandidateFamily")
            .field("identity", &self.identity)
            .field("authority", &self.authority)
            .field("launches", &self.executable.schedule().launches().len())
            .field("kernels", &self.executable.kernels().kernels().count())
            .finish_non_exhaustive()
    }
}

/// Crate-private carrier used only while a builder seals a family.
pub(crate) struct CandidateFamilyParts<B: KernelDialect> {
    pub authority: ConstructionAuthority,
    pub numerical_role: seismic_lang::entry::NumericalRole,
    pub identity: CandidateFamilyIdentity,
    pub semantic_coverage: TargetPredicate,
    pub executable: seismic_ir::execution::ClosedExecutableIr<B>,
    pub launch_scratch: Vec<seismic_ir::storage::LaunchScratchRequirements>,
    pub launch_abi: Vec<Vec<seismic_ir::storage::LaunchAbiRequirement>>,
    pub choices: Vec<ChoiceDeclaration>,
    pub hard_constraints: BoolExpr,
    pub numerical_transfer: NumericalTransfer,
    pub provenance: ImplementationProvenance,
    pub result_publications: Vec<ResultPublication>,
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
        let nested = activate_choices(
            &mut arena,
            selected,
            vec![
                ChoiceDeclaration {
                    decision: child_guard,
                    meaning: "child guard",
                    active_when: child_guard_active,
                },
                ChoiceDeclaration {
                    decision: child_leaf,
                    meaning: "child leaf",
                    active_when: child_leaf_active,
                },
            ],
        );
        let parent_active = arena.bool(true);
        let mut declarations = vec![ChoiceDeclaration {
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
                SymbolValue::Int(parent_value),
            );
            assignment.bind(
                arena.decision_symbol(child_guard),
                SymbolValue::Int(guard_value),
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
                    decision: first,
                    meaning: "first",
                    active_when: first_active,
                },
                ChoiceDeclaration {
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
                decision,
                meaning: "invalid",
                active_when,
            }],
        );
    }
}
