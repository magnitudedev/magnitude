//! Planning over an already evaluated candidate domain.
//!
//! This module solely owns solver adaptation, planning budgets and portfolio
//! materialization. Its input is passive [`EvaluatedCandidateDomain`] data
//! plus a distinct [`PlanningBudget`]. Device, evaluator and native compiler
//! services are absent from this boundary.

use crate::candidate_domain::{NonEmpty, TargetDomain};
use crate::evaluation::{
    EvaluatedCandidate, EvaluatedCandidateDomain, EvaluatedCandidateDomainParts, EvaluatedUniversal,
};
use crate::numerics::{self, NumericalAssessment};
use crate::preparation_budget::PlanningBudget;
use crate::solve::{
    AssignmentStep, CursorBudgetReport, FeasibleAssignment, SolverModel, SolverModelBuilder,
};
use crate::target::TargetConstants;
use seismic_lang::entry::{CallSchema, SemanticEventManifest};
use seismic_lang::expr::{ExprArena, PartialAssignment};
use seismic_lang::ids::{ModuleHash, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::DeviceDescriptionIdentity;
use seismic_target::NumericalEnvironmentIdentity;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetCoverage {
    Exhaustive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanningLimit {
    CandidateConstruction,
    Solver(CursorBudgetReport),
    OptimizedAssignments,
    ExecutableVariants,
    RetainedMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptimizationCompletion {
    Complete,
    Limited(PlanningLimit),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanningBudgetReport {
    pub solver_work_units: u64,
    pub solver_elapsed_ms: u64,
    pub solver_memory_bytes: u64,
    pub optimized_assignments: u64,
    pub executable_variants: u64,
    pub retained_metadata_bytes: u64,
}

/// Target-domain coverage and optional optimization completion are separate
/// facts. A returned kernel is always exhaustively covered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningCoverage {
    pub target: TargetCoverage,
    pub optimization: OptimizationCompletion,
    pub budget: PlanningBudgetReport,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanningBudgetResource {
    RequiredRetainedMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningInfeasibleReport {
    pub entry: StableEntryId,
    pub reason: &'static str,
}

/// Mandatory infeasibility/failure is distinct from optional search limits,
/// which are returned in [`PlanningCoverage`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanningError {
    Infeasible(PlanningInfeasibleReport),
    BudgetExceeded {
        resource: PlanningBudgetResource,
        required: u64,
        limit: u64,
    },
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible(report) => write!(
                f,
                "planning is infeasible for {:?}: {}",
                report.entry, report.reason
            ),
            Self::BudgetExceeded {
                resource,
                required,
                limit,
            } => write!(
                f,
                "mandatory planning resource {resource:?} requires {required}, limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for PlanningError {}

pub fn plan<B: seismic_target::TargetFamily>(
    domain: EvaluatedCandidateDomain<B>,
    budget: &PlanningBudget,
) -> Result<PlannedPolicy<B>, PlanningError> {
    internals::plan(domain, budget)
}

/// The complete, handle-free result of planning. It is independently
/// selectable and can be materialized only by exact registry attachment.
pub struct PlannedPolicy<B: seismic_target::TargetFamily> {
    pub(crate) entry: StableEntryId,
    pub(crate) module: ModuleHash,
    pub(crate) schema: Arc<CallSchema>,
    pub(crate) semantic_events: Arc<SemanticEventManifest>,
    pub(crate) device: DeviceDescriptionIdentity,
    pub(crate) evaluation: crate::evaluation::EvaluationIdentity,
    pub(crate) invocation: crate::prepared::InvocationContract,
    pub(crate) variants: NonEmpty<crate::executable::PlannedVariant<B>>,
    pub(crate) coverage: PlanningCoverage,
}

impl<B: seismic_target::TargetFamily> PlannedPolicy<B> {
    pub fn entry(&self) -> StableEntryId {
        self.entry
    }
    pub fn device_identity(&self) -> &DeviceDescriptionIdentity {
        &self.device
    }
    pub fn coverage(&self) -> &PlanningCoverage {
        &self.coverage
    }
    pub fn variant_identities(&self) -> impl Iterator<Item = &crate::frozen::VariantIdentity> {
        self.variants
            .iter()
            .map(crate::executable::PlannedVariant::identity)
    }

    /// Pure invocation-time choice over the planned portfolio. Native handles
    /// and services are unnecessary to verify the policy's decision.
    pub fn select(
        &self,
        values: &seismic_lang::expr::compiled::InvocationValues,
    ) -> SelectedPlannedVariant<'_, B> {
        let index = crate::executable::select_variant_index(self.variants.as_slice(), values);
        SelectedPlannedVariant {
            index,
            variant: &self.variants.as_slice()[index],
        }
    }
}

#[derive(Clone, Copy)]
pub struct SelectedPlannedVariant<'a, B: seismic_target::TargetFamily> {
    pub index: usize,
    pub variant: &'a crate::executable::PlannedVariant<B>,
}

/// Transient planning-owned state. It is derived after evaluation and is not
/// a second candidate-membership authority.
#[derive(Debug)]
pub(crate) struct PlanningState<B: seismic_target::TargetFamily> {
    pub(crate) entry: StableEntryId,
    pub(crate) module: ModuleHash,
    pub(crate) schema: Arc<CallSchema>,
    pub(crate) semantic_events: Arc<SemanticEventManifest>,
    pub(crate) target_domain: TargetDomain,
    pub(crate) constants: TargetConstants,
    pub(crate) device: DeviceDescriptionIdentity,
    pub(crate) evaluation: crate::evaluation::EvaluationIdentity,
    pub(crate) target: NumericalEnvironmentIdentity,
    pub(crate) evidence: Arc<crate::numerics::EvidenceCatalog>,
    pub(crate) arena: Arc<ExprArena>,
    pub(crate) universal: EvaluatedUniversal<B>,
    pub(crate) optimized: Vec<EvaluatedCandidate<B>>,
    pub(crate) model: SolverModel,
    pub(crate) precision: PrecisionPolicy,
    pub(crate) construction_limited: bool,
}

/// Exact selection produced by planning. The numerical report is established
/// once here and carried through freezing.
pub(crate) struct AssessedAssignment {
    pub(crate) assignment: FeasibleAssignment,
    pub(crate) numerical: NumericalAssessment,
}

mod internals {
    use super::*;

    pub(super) fn plan<B: seismic_target::TargetFamily>(
        domain: EvaluatedCandidateDomain<B>,
        budget: &PlanningBudget,
    ) -> Result<PlannedPolicy<B>, PlanningError> {
        let state = planning_state(domain);
        let mut target_bindings = PartialAssignment::new();
        for (symbol, value) in state.constants.bindings() {
            target_bindings.bind(*symbol, *value);
        }
        let invocation = crate::prepared::InvocationContract::compile(
            &state.arena,
            &state.schema,
            state.target_domain,
            &target_bindings,
        );
        let mut accounting = PlanningAccounting::new(budget);

        let universal = planned_variant(&state, FeasibleAssignment::universal());
        let executable =
            crate::executable::compile_variant(crate::frozen::freeze(&state, universal));
        accounting.record_required_variant(executable.retained_metadata_bytes())?;
        let mut variants = vec![executable];

        let mut completion = state
            .construction_limited
            .then_some(PlanningLimit::CandidateConstruction)
            .or_else(|| accounting.optional_limit());
        if completion.is_none() {
            completion = enumerate_optional(&state.model, &mut accounting, |assignment| {
                let planned = planned_variant(&state, assignment);
                let executable =
                    crate::executable::compile_variant(crate::frozen::freeze(&state, planned));
                let bytes = executable.retained_metadata_bytes();
                variants.push(executable);
                bytes
            });
        }

        let variants =
            NonEmpty::new(variants).ok_or(PlanningError::Infeasible(PlanningInfeasibleReport {
                entry: state.entry,
                reason: "evaluated domain omitted its mandatory universal member",
            }))?;
        let coverage = PlanningCoverage {
            target: TargetCoverage::Exhaustive,
            optimization: completion.map_or(
                OptimizationCompletion::Complete,
                OptimizationCompletion::Limited,
            ),
            budget: accounting.report(),
        };
        Ok(PlannedPolicy {
            entry: state.entry,
            module: state.module,
            schema: state.schema,
            semantic_events: state.semantic_events,
            device: state.device,
            evaluation: state.evaluation,
            invocation,
            variants,
            coverage,
        })
    }

    fn planning_state<B: seismic_target::TargetFamily>(
        domain: EvaluatedCandidateDomain<B>,
    ) -> PlanningState<B> {
        let EvaluatedCandidateDomainParts {
            entry,
            module,
            schema,
            semantic_events,
            target_domain,
            constants,
            device,
            evaluation,
            target,
            evidence,
            mut arena,
            universal,
            optimized,
            precision,
            optimization_exhausted,
        } = domain.into_parts();
        let predicates = optimized
            .iter()
            .map(|candidate| {
                crate::candidate_domain::internals::planning_projection(
                    &mut arena,
                    candidate.candidate.constraints.predicate(),
                )
            })
            .collect::<Vec<_>>();
        let mut builder = SolverModelBuilder::new(&mut arena);
        for (symbol, value) in constants.bindings() {
            builder.bind_target(*symbol, *value);
        }
        for (index, (candidate, predicate)) in optimized.iter().zip(predicates).enumerate() {
            let implementation = candidate.candidate.implementation.as_inner();
            let decisions = implementation
                .decisions()
                .iter()
                .map(|(decision, _)| *decision)
                .collect::<Vec<_>>();
            builder.implementation(
                u32::try_from(index + 1).expect("candidate family count exceeds u32"),
                implementation.identity().clone(),
                &decisions,
                predicate,
            );
        }
        let model = builder.build();
        PlanningState {
            entry,
            module,
            schema,
            semantic_events,
            target_domain,
            constants,
            device,
            evaluation,
            target,
            evidence,
            arena: Arc::new(arena),
            universal,
            optimized,
            model,
            precision,
            construction_limited: optimization_exhausted,
        }
    }

    fn planned_variant<B: seismic_target::TargetFamily>(
        state: &PlanningState<B>,
        assignment: FeasibleAssignment,
    ) -> AssessedAssignment {
        let implementation = if assignment.implementation() == 0 {
            state.universal.implementation.as_inner()
        } else {
            state
                .optimized
                .get(assignment.implementation() as usize - 1)
                .unwrap_or_else(|| panic!("solver selected a family outside the evaluated domain"))
                .candidate
                .implementation
                .as_inner()
        };
        let numerical = numerics::assess(
            &state.arena,
            implementation.numerical_transfer(),
            &state.precision,
            &state.evidence,
            implementation.identity(),
            &state.target,
            implementation.native_numerical_identity(),
            state.target_domain.identity(),
            assignment.decisions(),
        );
        AssessedAssignment {
            assignment,
            numerical,
        }
    }

    pub(super) fn enumerate_optional(
        model: &SolverModel,
        accounting: &mut PlanningAccounting<'_>,
        mut materialize: impl FnMut(FeasibleAssignment) -> u64,
    ) -> Option<PlanningLimit> {
        let mut cursor = model.assignments(accounting.solver_allowance());
        loop {
            match cursor.next() {
                AssignmentStep::Assignment(assignment) => {
                    if assignment.implementation() == 0 {
                        continue;
                    }
                    if let Some(limit) = accounting.before_optional_variant() {
                        accounting.solver = cursor.report();
                        return Some(limit);
                    }
                    let bytes = materialize(assignment);
                    if let Some(limit) = accounting.record_optional_variant(bytes) {
                        accounting.solver = cursor.report();
                        return Some(limit);
                    }
                }
                AssignmentStep::Complete(report) => {
                    accounting.solver = report;
                    return None;
                }
                AssignmentStep::BudgetExhausted(report) => {
                    accounting.solver = report.clone();
                    return Some(PlanningLimit::Solver(report));
                }
            }
        }
    }

    pub(super) struct PlanningAccounting<'a> {
        limit: &'a PlanningBudget,
        solver: CursorBudgetReport,
        optimized_assignments: u64,
        executable_variants: u64,
        retained_metadata_bytes: u64,
    }

    impl<'a> PlanningAccounting<'a> {
        pub(super) fn new(limit: &'a PlanningBudget) -> Self {
            Self {
                limit,
                solver: CursorBudgetReport {
                    work_units: 0,
                    elapsed_ms: 0,
                    memory_bytes: 0,
                },
                optimized_assignments: 0,
                executable_variants: 0,
                retained_metadata_bytes: 0,
            }
        }

        fn solver_allowance(&self) -> crate::solve::SolverAllowance {
            crate::solve::SolverAllowance {
                work: self.limit.solver_work,
                memory_bytes: Some(self.limit.solver_memory_bytes),
            }
        }

        pub(super) fn record_required_variant(&mut self, bytes: u64) -> Result<(), PlanningError> {
            if bytes > self.limit.required_retained_metadata_bytes {
                return Err(PlanningError::BudgetExceeded {
                    resource: PlanningBudgetResource::RequiredRetainedMetadata,
                    required: bytes,
                    limit: self.limit.required_retained_metadata_bytes,
                });
            }
            self.executable_variants = 1;
            self.retained_metadata_bytes = bytes;
            Ok(())
        }

        fn optional_limit(&self) -> Option<PlanningLimit> {
            if self.executable_variants >= self.limit.executable_variants {
                Some(PlanningLimit::ExecutableVariants)
            } else if self.retained_metadata_bytes >= self.limit.retained_metadata_bytes {
                Some(PlanningLimit::RetainedMetadata)
            } else {
                None
            }
        }

        fn before_optional_variant(&mut self) -> Option<PlanningLimit> {
            if self.optimized_assignments >= self.limit.optimized_assignments {
                return Some(PlanningLimit::OptimizedAssignments);
            }
            if self.executable_variants >= self.limit.executable_variants {
                return Some(PlanningLimit::ExecutableVariants);
            }
            self.optimized_assignments += 1;
            self.executable_variants += 1;
            None
        }

        fn record_optional_variant(&mut self, bytes: u64) -> Option<PlanningLimit> {
            self.retained_metadata_bytes = self.retained_metadata_bytes.saturating_add(bytes);
            (self.retained_metadata_bytes > self.limit.retained_metadata_bytes)
                .then_some(PlanningLimit::RetainedMetadata)
        }

        pub(super) fn report(&self) -> PlanningBudgetReport {
            PlanningBudgetReport {
                solver_work_units: self.solver.work_units,
                solver_elapsed_ms: self.solver.elapsed_ms,
                solver_memory_bytes: self.solver.memory_bytes,
                optimized_assignments: self.optimized_assignments,
                executable_variants: self.executable_variants,
                retained_metadata_bytes: self.retained_metadata_bytes,
            }
        }
    }
}

#[cfg(test)]
mod planning_oracle_tests {
    use super::*;
    use crate::expression::PlanningExpr;
    use crate::implementation::ImplementationIdentity;
    use crate::refinement::FactoryIdentity;
    use seismic_lang::expr::{CmpOp, FiniteDomain};
    use std::collections::BTreeSet;

    #[test]
    fn complete_planning_enumeration_matches_cartesian_oracle_and_reports_coverage() {
        for bound in 0..10 {
            let mut arena = ExprArena::default();
            let a = arena.decision(FiniteDomain::new(vec![0, 1, 2, 3]).unwrap());
            let b = arena.decision(FiniteDomain::new(vec![0, 1, 2, 3]).unwrap());
            let av = arena.decision_value(a);
            let bv = arena.decision_value(b);
            let product = arena.int_mul(av, bv);
            let limit = arena.int(bound);
            let predicate = arena.int_cmp(CmpOp::Le, product, limit);
            let planning = PlanningExpr::new(&arena, predicate).unwrap();
            let mut builder = SolverModelBuilder::new(&mut arena);
            builder.implementation(
                1,
                ImplementationIdentity {
                    factory: FactoryIdentity {
                        name: "planning-oracle",
                        revision: "1",
                    },
                    structure: [0; 32],
                },
                &[a, b],
                planning,
            );
            let model = builder.build();
            let budget = PlanningBudget {
                solver_work: 1_000_000,
                solver_memory_bytes: u64::MAX,
                optimized_assignments: 32,
                executable_variants: 33,
                retained_metadata_bytes: u64::MAX,
                required_retained_metadata_bytes: u64::MAX,
            };
            let mut accounting = internals::PlanningAccounting::new(&budget);
            accounting.record_required_variant(0).unwrap();
            let mut actual = BTreeSet::new();
            let completion = internals::enumerate_optional(&model, &mut accounting, |assignment| {
                assert!(
                    actual.insert((assignment.value(a).unwrap(), assignment.value(b).unwrap(),))
                );
                0
            });
            let expected = (0..4)
                .flat_map(|a| (0..4).filter_map(move |b| (a * b <= bound).then_some((a, b))))
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected);
            assert_eq!(completion, None);
            let report = accounting.report();
            assert_eq!(report.optimized_assignments, expected.len() as u64);
            assert_eq!(report.executable_variants, expected.len() as u64 + 1);
            let coverage = PlanningCoverage {
                target: TargetCoverage::Exhaustive,
                optimization: OptimizationCompletion::Complete,
                budget: report,
            };
            assert_eq!(coverage.target, TargetCoverage::Exhaustive);
            assert_eq!(coverage.optimization, OptimizationCompletion::Complete);
        }
    }

    #[test]
    fn planning_limit_is_explicit_and_never_weakens_target_coverage() {
        let budget = PlanningBudget {
            optimized_assignments: 0,
            ..PlanningBudget::default()
        };
        let mut arena = ExprArena::default();
        let decision = arena.decision(FiniteDomain::new(vec![1]).unwrap());
        let predicate = arena.bool(true);
        let planning = PlanningExpr::new(&arena, predicate).unwrap();
        let mut builder = SolverModelBuilder::new(&mut arena);
        builder.implementation(
            1,
            ImplementationIdentity {
                factory: FactoryIdentity {
                    name: "planning-limit",
                    revision: "1",
                },
                structure: [1; 32],
            },
            &[decision],
            planning,
        );
        let model = builder.build();
        let mut accounting = internals::PlanningAccounting::new(&budget);
        accounting.record_required_variant(0).unwrap();
        let limit = internals::enumerate_optional(&model, &mut accounting, |_| 0);
        assert_eq!(limit, Some(PlanningLimit::OptimizedAssignments));
        let coverage = PlanningCoverage {
            target: TargetCoverage::Exhaustive,
            optimization: OptimizationCompletion::Limited(limit.unwrap()),
            budget: accounting.report(),
        };
        assert_eq!(coverage.target, TargetCoverage::Exhaustive);
    }
}
