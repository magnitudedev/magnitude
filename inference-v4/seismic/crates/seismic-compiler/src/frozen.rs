//! One exact, transient view of a closed implementation (spec §11.1).
//!
//! A `FrozenPlan` does not copy or move the parametric implementation. It
//! owns an `Arc` to that immutable object and fixes every compile-time symbol
//! with one exact assignment. Native formation and reflection have already
//! completed before the implementation entered `PlanSpace`; freezing only
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

use crate::implementation::{Implementation, ImplementationIdentity};
use crate::numerics::{self, NumericalAssessment};
use crate::plan_space::{FrozenSpace, TargetDomain};
use crate::solve::FeasibleAssignment;
use crate::target::Backend;
use seismic_lang::entry::CallSchema;
use seismic_lang::expr::{BoolExpr, PartialAssignment, SymbolKind, SymbolValue};
use seismic_lang::ids::{ModuleHash, StableEntryId};
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
pub struct FrozenPlan<B: Backend> {
    entry: StableEntryId,
    module: ModuleHash,
    schema: Arc<CallSchema>,
    target_domain: TargetDomain,
    device: crate::target::DeviceContractIdentity,
    execution: crate::target::ExecutionProfileIdentity,
    identity: VariantIdentity,
    arena: Arc<seismic_lang::expr::ExprArena>,
    implementation: Arc<Implementation<B>>,
    assignment: FeasibleAssignment,
    fixed: PartialAssignment,
    guard: FrozenGuard,
    numerical: NumericalAssessment,
}

impl<B: Backend> FrozenPlan<B> {
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn module(&self) -> ModuleHash {
        self.module
    }
    pub fn schema(&self) -> &Arc<CallSchema> {
        &self.schema
    }
    pub fn target_domain(&self) -> TargetDomain {
        self.target_domain
    }
    pub fn device_identity(&self) -> &crate::target::DeviceContractIdentity {
        &self.device
    }
    pub fn execution_profile_identity(&self) -> &crate::target::ExecutionProfileIdentity {
        &self.execution
    }
    pub fn identity(&self) -> &VariantIdentity {
        &self.identity
    }
    pub fn numerical(&self) -> &NumericalAssessment {
        &self.numerical
    }

    pub(crate) fn arena(&self) -> &seismic_lang::expr::ExprArena {
        &self.arena
    }
    pub(crate) fn implementation(&self) -> &Implementation<B> {
        &self.implementation
    }
    pub(crate) fn assignment(&self) -> &FeasibleAssignment {
        &self.assignment
    }
    pub(crate) fn fixed(&self) -> &PartialAssignment {
        &self.fixed
    }
    pub(crate) fn guard(&self) -> &FrozenGuard {
        &self.guard
    }

    pub(crate) fn into_exact_parts(self) -> FrozenPlanParts<B> {
        FrozenPlanParts {
            entry: self.entry,
            module: self.module,
            schema: self.schema,
            target_domain: self.target_domain,
            device: self.device,
            execution: self.execution,
            identity: self.identity,
            arena: self.arena,
            implementation: self.implementation,
            assignment: self.assignment,
            fixed: self.fixed,
            guard: self.guard,
            numerical: self.numerical,
        }
    }
}

pub(crate) struct FrozenPlanParts<B: Backend> {
    pub entry: StableEntryId,
    pub module: ModuleHash,
    pub schema: Arc<CallSchema>,
    pub target_domain: TargetDomain,
    pub device: crate::target::DeviceContractIdentity,
    pub execution: crate::target::ExecutionProfileIdentity,
    pub identity: VariantIdentity,
    pub arena: Arc<seismic_lang::expr::ExprArena>,
    pub implementation: Arc<Implementation<B>>,
    pub assignment: FeasibleAssignment,
    pub fixed: PartialAssignment,
    pub guard: FrozenGuard,
    pub numerical: NumericalAssessment,
}

/// Infallible for a solver-produced assignment. Any failure below is a
/// private compiler-construction bug.
pub(crate) fn freeze<B: Backend>(
    space: &FrozenSpace<B>,
    assignment: FeasibleAssignment,
) -> FrozenPlan<B> {
    let index = assignment.implementation() as usize;
    let (implementation, guard_node) = if index == 0 {
        (
            space.universal.shared(),
            space.target_domain.predicate().node(),
        )
    } else {
        let candidate = space
            .optimized
            .get(index - 1)
            .unwrap_or_else(|| panic!("solver selected an implementation outside its plan space"));
        (candidate.implementation.shared(), candidate.guard.node())
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
    let numerical = numerics::assess(
        &space.arena,
        implementation.numerical_transfer(),
        &space.precision,
        &space.evidence,
        implementation.identity(),
        &space.target,
        implementation.native_numerical_identity(),
        space.target_domain.identity(),
        assignment.decisions(),
    );

    FrozenPlan {
        entry: space.entry,
        module: space.module,
        schema: space.schema.clone(),
        target_domain: space.target_domain,
        device: space.device.clone(),
        execution: space.execution.clone(),
        identity,
        arena: space.arena.clone(),
        implementation,
        assignment,
        fixed,
        guard,
        numerical,
    }
}
