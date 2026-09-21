//! Automatic portfolio construction with exact coverage (spec §10.3).
//!
//! 1. Freeze and compile the sealed, decision-free universal implementation.
//!    Its construction proves complete `TargetDomain` coverage before this
//!    module is entered.
//! 2. Enumerate finite optimized assignments, freeze each exact residual
//!    guard, and compile it.
//! 3. Stop optimized enumeration when complete or when its optional budget is
//!    exhausted. The universal variant remains the total final partition, so
//!    an optimization budget can affect optimality but never coverage.
//!
//! Coverage is mandatory and constructional. A prepared portfolio may be
//! fully covered but not optimal; it may never be partially covered. There is
//! no retry, uncovered-domain search, or executable fallback path here.
//!
//! W7 owns the loop; the interface here is frozen.

use crate::errors::PreparationError;
use crate::plan_space::PlanSpace;
use crate::prepared::PreparedKernel;
use crate::target::{Backend, PlanningMachine};

/// Builds a fully covered prepared kernel from a plan space.
pub fn prepare_kernel<B: Backend>(
    space: PlanSpace<B>,
    machine: PlanningMachine<'_, B>,
) -> Result<PreparedKernel<B>, PreparationError> {
    internals::prepare_kernel(space, machine)
}

mod internals {
    use super::*;
    use crate::solve::AssignmentStep;
    use seismic_lang::expr::PartialAssignment;

    pub(super) fn prepare_kernel<B: Backend>(
        space: PlanSpace<B>,
        machine: PlanningMachine<'_, B>,
    ) -> Result<PreparedKernel<B>, PreparationError> {
        let entry = space.entry();
        let module = space.module();
        let schema = space.schema().clone();
        let target_domain = space.target_domain();
        let frozen = space.into_frozen_parts();
        assert_eq!(
            &frozen.device,
            machine.device().identity(),
            "PlanSpace prepared with a different device contract"
        );
        assert_eq!(
            &frozen.execution,
            machine.execution().identity(),
            "PlanSpace prepared with a different execution profile"
        );
        let semantic_events = frozen.semantic_events.clone();
        let device = frozen.device.clone();
        let execution = frozen.execution.clone();
        let mut target_bindings = PartialAssignment::new();
        for (symbol, value) in frozen.constants.bindings() {
            target_bindings.bind(*symbol, *value);
        }
        let invocation = crate::prepared::InvocationContract::compile(
            &frozen.arena,
            &schema,
            target_domain,
            &target_bindings,
        );

        // Coverage is constructional: exactly one sealed universal portable
        // implementation covers TargetDomain and has no finite decisions.
        // Optional optimized assignments can improve selection, but coverage
        // never depends on their search budget.
        let universal_assignment = crate::solve::FeasibleAssignment::universal();
        let universal_index = universal_assignment.implementation();
        let mut tracker = frozen.budget.clone();
        let plan = crate::frozen::freeze(&frozen, universal_assignment);
        let executable = crate::executable::compile_variant(plan);
        let universal_within_budget =
            tracker.record_required_variant(executable.retained_metadata_bytes())?;
        let mut optimal = !frozen.optimization_exhausted && universal_within_budget;
        let allow_optional_variants = universal_within_budget;
        let mut variants = vec![executable];

        if allow_optional_variants {
            let mut cursor = frozen.model.assignments(tracker.solver_allowance());
            loop {
                match cursor.next() {
                    AssignmentStep::Assignment(assignment) => {
                        if assignment.implementation() == universal_index {
                            continue;
                        }
                        if !tracker.charge_optimized_assignment() || !tracker.charge_variant() {
                            optimal = false;
                            break;
                        }
                        let plan = crate::frozen::freeze(&frozen, assignment);
                        let executable = crate::executable::compile_variant(plan);
                        let within_metadata =
                            tracker.charge_metadata(executable.retained_metadata_bytes());
                        variants.push(executable);
                        if !within_metadata {
                            optimal = false;
                            break;
                        }
                    }
                    AssignmentStep::Complete => break,
                    AssignmentStep::BudgetExhausted(_) => {
                        optimal = false;
                        break;
                    }
                }
            }
        }
        let variants = crate::plan_space::NonEmpty::new(variants)
            .unwrap_or_else(|| panic!("universal executable was not retained"));
        Ok(PreparedKernel::prepare(
            entry,
            module,
            schema,
            semantic_events,
            device,
            execution,
            invocation,
            variants,
            optimal,
        ))
    }
}
