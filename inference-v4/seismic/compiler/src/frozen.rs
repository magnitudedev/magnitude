//! Coordinate-exact lowering input.
//!
//! A `FrozenPlan` is created only after a structural coordinate has been
//! natively realized and numerically admitted. It fixes target constants and
//! active choices, but makes no search or selection decision. Both analytical
//! and feedback evaluators use this same lowering boundary.

use crate::implementation::{Implementation, ImplementationIdentity};
use crate::numerics::NumericalAssessment;
use seismic_lang::entry::CallSchema;
use seismic_lang::expr::{BoolExpr, ExprArena, PartialAssignment, SymbolKind};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantIdentity {
    pub implementation: ImplementationIdentity,
    pub assignment: [u8; 32],
}

#[derive(Clone, Debug)]
pub(crate) struct FrozenGuard {
    node: BoolExpr,
    fixed: PartialAssignment,
}

impl FrozenGuard {
    fn new(arena: &ExprArena, node: BoolExpr, fixed: PartialAssignment) -> Self {
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

pub(crate) struct FinalizationContext {
    pub(crate) schema: Arc<CallSchema>,
    pub(crate) device: seismic_target::DeviceDescriptionIdentity,
    pub(crate) evaluation: crate::evaluation::EvaluationIdentity,
    pub(crate) arena: Arc<ExprArena>,
    pub(crate) constants: crate::target::TargetConstants,
}

#[derive(Debug)]
pub(crate) struct FrozenPlan<B: seismic_target::TargetFamily> {
    schema: Arc<CallSchema>,
    device: seismic_target::DeviceDescriptionIdentity,
    evaluation: crate::evaluation::EvaluationIdentity,
    identity: VariantIdentity,
    arena: Arc<ExprArena>,
    implementation: Arc<Implementation<B>>,
    fixed: PartialAssignment,
    guard: FrozenGuard,
    numerical: NumericalAssessment,
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
            fixed: self.fixed,
            guard: self.guard,
            numerical: self.numerical,
        }
    }
}

pub(crate) struct FrozenPlanParts<B: seismic_target::TargetFamily> {
    pub schema: Arc<CallSchema>,
    pub device: seismic_target::DeviceDescriptionIdentity,
    pub evaluation: crate::evaluation::EvaluationIdentity,
    pub identity: VariantIdentity,
    pub arena: Arc<ExprArena>,
    pub implementation: Arc<Implementation<B>>,
    pub fixed: PartialAssignment,
    pub guard: FrozenGuard,
    pub numerical: NumericalAssessment,
}

pub(crate) fn freeze<B: seismic_target::TargetFamily>(
    context: &FinalizationContext,
    implementation: Arc<Implementation<B>>,
    guard_node: BoolExpr,
    numerical: NumericalAssessment,
) -> FrozenPlan<B> {
    let mut fixed = PartialAssignment::new();
    for (symbol, value) in context.constants.bindings() {
        fixed.bind(*symbol, *value);
    }
    for (decision, _) in implementation.decisions() {
        let symbol = context.arena.decision_symbol(decision);
        if let Some(value) = implementation.assignment().get(symbol) {
            fixed.bind(symbol, value);
        }
    }
    let guard = FrozenGuard::new(&context.arena, guard_node, fixed.clone());
    let identity = VariantIdentity {
        implementation: implementation.identity().clone(),
        assignment: implementation.assignment_identity(),
    };
    FrozenPlan {
        schema: context.schema.clone(),
        device: context.device.clone(),
        evaluation: context.evaluation.clone(),
        identity,
        arena: context.arena.clone(),
        implementation,
        fixed,
        guard,
        numerical,
    }
}
