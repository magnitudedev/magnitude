//! Coordinate-exact lowering input.
//!
//! A `FrozenPlan` is created only after a structural coordinate has been
//! natively realized and execution-safe. Numerical acceptance follows compilation.
//! It fixes target constants and
//! active choices, but makes no search or selection decision. Both analytical
//! and feedback evaluators use this same lowering boundary.

use crate::implementation::{Implementation, ImplementationIdentity};
use crate::prepared::InvocationContract;
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
}

impl FrozenGuard {
    fn new(arena: &ExprArena, node: BoolExpr, fixed: &PartialAssignment) -> Self {
        for symbol in arena.free_symbols(node.into()) {
            match arena.symbol_kind(symbol) {
                SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_) => {}
                SymbolKind::TargetConstant(_) | SymbolKind::Decision(_)
                    if fixed.get(symbol).is_some() => {}
                kind => panic!("frozen guard retained an unfixed planning symbol: {kind:?}"),
            }
        }
        Self { node }
    }

    pub(crate) fn node(&self) -> BoolExpr {
        self.node
    }
}

pub(crate) struct CandidateContext<'a> {
    pub(crate) invocation: Arc<InvocationContract>,
    pub(crate) device: seismic_target::DeviceDescriptionIdentity,
    pub(crate) arena: &'a ExprArena,
    pub(crate) constants: crate::target::TargetConstants,
}

#[derive(Debug)]
pub(crate) struct FrozenPlan<'a, B: seismic_target::TargetFamily> {
    invocation: Arc<InvocationContract>,
    device: seismic_target::DeviceDescriptionIdentity,
    identity: VariantIdentity,
    arena: &'a ExprArena,
    implementation: Arc<Implementation<B>>,
    fixed: PartialAssignment,
    guard: FrozenGuard,
}

impl<'a, B: seismic_target::TargetFamily> FrozenPlan<'a, B> {
    pub(crate) fn into_exact_parts(self) -> FrozenPlanParts<'a, B> {
        FrozenPlanParts {
            invocation: self.invocation,
            device: self.device,
            identity: self.identity,
            arena: self.arena,
            implementation: self.implementation,
            fixed: self.fixed,
            guard: self.guard,
        }
    }
}

pub(crate) struct FrozenPlanParts<'a, B: seismic_target::TargetFamily> {
    pub invocation: Arc<InvocationContract>,
    pub device: seismic_target::DeviceDescriptionIdentity,
    pub identity: VariantIdentity,
    pub arena: &'a ExprArena,
    pub implementation: Arc<Implementation<B>>,
    pub fixed: PartialAssignment,
    pub guard: FrozenGuard,
}

/// The values a frozen plan fixes: target constants and the implementation's
/// active choices.
pub(crate) fn plan_assignment<B: seismic_target::TargetFamily>(
    arena: &ExprArena,
    constants: &crate::target::TargetConstants,
    implementation: &Implementation<B>,
) -> PartialAssignment {
    let mut fixed = PartialAssignment::new();
    for (symbol, value) in constants.bindings() {
        fixed.bind(*symbol, value.clone());
    }
    for (decision, _) in implementation.decisions() {
        let symbol = arena.decision_symbol(decision);
        if let Some(value) = implementation.assignment().get(symbol) {
            fixed.bind(symbol, value);
        }
    }
    fixed
}

/// `guard_node` is already partially evaluated under `fixed`
/// (`plan_assignment`): a requirement that holds for every fixed target
/// value no longer mentions the schedule values it was scoped under.
pub(crate) fn freeze<'a, B: seismic_target::TargetFamily>(
    context: &CandidateContext<'a>,
    implementation: Arc<Implementation<B>>,
    fixed: PartialAssignment,
    guard_node: BoolExpr,
) -> FrozenPlan<'a, B> {
    let guard = FrozenGuard::new(&context.arena, guard_node, &fixed);
    let identity = VariantIdentity {
        implementation: implementation.identity().clone(),
        assignment: implementation.assignment_identity(),
    };
    FrozenPlan {
        invocation: context.invocation.clone(),
        device: context.device.clone(),
        identity,
        arena: context.arena,
        implementation,
        fixed,
        guard,
    }
}
