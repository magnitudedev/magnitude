//! One exact, transient view of a closed implementation (spec §11.1).
//!
//! A `FrozenPlan` does not copy or move the parametric implementation. It
//! owns an `Arc` to that immutable object and fixes every compile-time symbol
//! with one exact assignment. Native formation and reflection have already
//! completed before the implementation entered `CandidateDomain`; freezing only
//! resolves every `Choose` and arena-reuse decision and compiles evaluators
//! with these fixed bindings. The executable retains neither object.
//!
//! Raw solver witnesses cannot cross this boundary. Both the raw assignment
//! representation and the freezing transition are crate-private:
//!
//! ```compile_fail
//! use seismic_compiler::frozen::freeze;
//! use seismic_compiler::solve::RawAssignment;
//! ```

use crate::evaluation::CandidatePerformanceModel;
use crate::implementation::{Implementation, ImplementationIdentity};
use crate::numerics::NumericalAssessment;
use crate::planning::{AssessedAssignment, PlanningState};
use crate::solve::FeasibleAssignment;
use seismic_lang::entry::CallSchema;
use seismic_lang::expr::{BoolExpr, PartialAssignment, SymbolKind, SymbolValue};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantIdentity {
    pub implementation: ImplementationIdentity,
    pub assignment: [u8; 32],
}

/// A complete executable guard whose target and decision symbols are fixed.
#[derive(Clone, Debug)]
pub(crate) struct FrozenGuard {
    node: BoolExpr,
    fixed: PartialAssignment,
}

impl FrozenGuard {
    fn new(
        arena: &seismic_lang::expr::ExprArena,
        node: BoolExpr,
        fixed: PartialAssignment,
    ) -> Self {
        for symbol in arena.free_symbols(node.into()) {
            match arena.symbol_kind(symbol) {
                SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_) => {}
                SymbolKind::TargetConstant(_) | SymbolKind::Decision(_)
                    if fixed.get(symbol).is_some() => {}
                kind => panic!("frozen guard retained an unfixed planning symbol: {kind:?}"),
            }
        }
        Self { node, fixed }
    }

    pub(crate) fn node(&self) -> BoolExpr {
        self.node
    }
    pub(crate) fn fixed(&self) -> &PartialAssignment {
        &self.fixed
    }
}

#[derive(Debug)]
pub(crate) struct FrozenPlan<B: seismic_target::TargetFamily> {
    schema: Arc<CallSchema>,
    device: seismic_target::DeviceDescriptionIdentity,
    evaluation: crate::evaluation::EvaluationIdentity,
    identity: VariantIdentity,
    arena: Arc<seismic_lang::expr::ExprArena>,
    implementation: Arc<Implementation<B>>,
    assignment: FeasibleAssignment,
    fixed: PartialAssignment,
    guard: FrozenGuard,
    numerical: NumericalAssessment,
    performance: CandidatePerformanceModel,
}

impl<B: seismic_target::TargetFamily> FrozenPlan<B> {
    pub(crate) fn into_exact_parts(self) -> FrozenPlanParts<B> {
        FrozenPlanParts {
            schema: self.schema,
            device: self.device,
            evaluation: self.evaluation,
            identity: self.identity,
            arena: self.arena,
            implementation: self.implementation,
            assignment: self.assignment,
            fixed: self.fixed,
            guard: self.guard,
            numerical: self.numerical,
            performance: self.performance,
        }
    }
}

pub(crate) struct FrozenPlanParts<B: seismic_target::TargetFamily> {
    pub schema: Arc<CallSchema>,
    pub device: seismic_target::DeviceDescriptionIdentity,
    pub evaluation: crate::evaluation::EvaluationIdentity,
    pub identity: VariantIdentity,
    pub arena: Arc<seismic_lang::expr::ExprArena>,
    pub implementation: Arc<Implementation<B>>,
    pub assignment: FeasibleAssignment,
    pub fixed: PartialAssignment,
    pub guard: FrozenGuard,
    pub numerical: NumericalAssessment,
    pub performance: CandidatePerformanceModel,
}

/// Infallible for a solver-produced assignment. Any failure below is a
/// private compiler-construction bug.
pub(crate) fn freeze<B: seismic_target::TargetFamily>(
    space: &PlanningState<B>,
    planned: AssessedAssignment,
) -> FrozenPlan<B> {
    let AssessedAssignment {
        assignment,
        numerical,
    } = planned;
    let index = assignment.implementation() as usize;
    let (implementation, guard_node, performance) = if index == 0 {
        (
            space.universal.implementation.shared(),
            space.target_domain.predicate().node(),
            space.universal.performance.clone(),
        )
    } else {
        let candidate = space
            .optimized
            .get(index - 1)
            .unwrap_or_else(|| panic!("solver selected a family outside its planning state"));
        (
            candidate.candidate.implementation.shared(),
            candidate.candidate.constraints.predicate(),
            candidate.performance.clone(),
        )
    };
    let declared = implementation.decisions();
    if declared.len() != assignment.decisions().len()
        || declared
            .iter()
            .zip(assignment.decisions())
            .any(|((expected, _), (actual, _))| expected != actual)
    {
        panic!("solver assignment does not exactly bind the selected implementation decisions");
    }

    let mut fixed = PartialAssignment::new();
    for (symbol, value) in space.constants.bindings() {
        fixed.bind(*symbol, *value);
    }
    for (decision, value) in assignment.decisions() {
        fixed.bind(
            space.arena.decision_symbol(*decision),
            SymbolValue::Int(*value),
        );
    }
    let guard = FrozenGuard::new(&space.arena, guard_node, fixed.clone());

    let mut digest = Sha256::new();
    digest.update(b"seismic-variant-assignment-v1");
    for (ordinal, (_, value)) in assignment.decisions().iter().enumerate() {
        digest.update((ordinal as u64).to_le_bytes());
        digest.update(value.to_le_bytes());
    }
    let identity = VariantIdentity {
        implementation: implementation.identity().clone(),
        assignment: digest.finalize().into(),
    };
    FrozenPlan {
        schema: space.schema.clone(),
        device: space.device.clone(),
        evaluation: space.evaluation.clone(),
        identity,
        arena: space.arena.clone(),
        implementation,
        assignment,
        fixed,
        guard,
        numerical,
        performance,
    }
}
