//! Private analytical search over a totally modeled candidate domain.
//!
//! This module solely owns solver adaptation, planning budgets and portfolio
//! selection. Its input is passive [`AnalyticalDomainModel`] data
//! plus a distinct [`PlanningBudget`]. Device, evaluator and native compiler
//! services are absent from this boundary.

use crate::candidate_domain::{canonical_coordinate, CandidateCoordinate, NonEmpty};
use crate::evaluation::{AnalyticalDomainModel, AnalyticalDomainModelParts};
use crate::numerics::StructuralNumericalObligation;
use crate::preparation_budget::PlanningBudget;
use crate::solve::{
    AssignmentStep, CursorBudgetReport, FeasibleAssignment, SolverModel, SolverModelBuilder,
};
use crate::target::TargetConstants;
use seismic_lang::expr::ExprArena;
use seismic_lang::ids::StableEntryId;

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
    NativeArtifacts,
    RetainedMetadata,
    UnsupportedObjective(crate::solve::UnsupportedObjective),
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

pub(crate) fn plan<B: seismic_target::TargetFamily>(
    domain: AnalyticalDomainModel<B>,
    budget: &PlanningBudget,
) -> Result<AnalyticalSearchResult<B>, PlanningError> {
    internals::plan(domain, budget)
}

/// One checked structural selection and the analytical evidence that caused
/// this strategy to retain it. Neither value is an executable variant.
#[derive(Clone, Debug)]
pub(crate) struct AnalyticalCandidate {
    coordinate: CandidateCoordinate,
    performance: crate::evaluation::CandidatePerformanceModel,
    numerical: StructuralNumericalObligation,
}

impl AnalyticalCandidate {
    pub fn coordinate(&self) -> &CandidateCoordinate {
        &self.coordinate
    }
    pub fn performance(&self) -> &crate::evaluation::CandidatePerformanceModel {
        &self.performance
    }
}

/// Pre-realization analytical planning result. The evaluated structural
/// domain remains owned here so later realization can consume selected
/// coordinates without reconstructing family or arena ownership. Symbolic
/// costs may vary over the invocation domain, so this stage prunes only when
/// fixed applicability and cost expressions prove one coordinate redundant.
/// It retains every incomparable coordinate admitted by the explicit search
/// budget and performs no representative-shape ranking.
pub(crate) struct AnalyticalSearchResult<B: seismic_target::TargetFamily> {
    domain: AnalyticalDomainModel<B>,
    selections: NonEmpty<AnalyticalCandidate>,
    coverage: PlanningCoverage,
}

impl<B: seismic_target::TargetFamily> AnalyticalSearchResult<B> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::candidate_domain::CandidateDomain<B>,
        crate::evaluation::EvaluationIdentity,
        NonEmpty<AnalyticalCandidate>,
        PlanningCoverage,
    ) {
        let (domain, evaluation) = self.domain.into_candidate_domain();
        (domain, evaluation, self.selections, self.coverage)
    }
}

/// The method-independent result of candidate evaluation: the retained
/// candidates and a total deterministic function selecting one of them for
/// every valid invocation. Preparation context, diagnostics, and native
/// artifacts are deliberately absent.
pub struct SelectionPolicy<B: seismic_target::TargetFamily> {
    pub(crate) candidates: NonEmpty<crate::executable::RetainedCandidate<B>>,
    pub(crate) selection_function: crate::prepared::SelectionFunction,
}

impl<B: seismic_target::TargetFamily> SelectionPolicy<B> {
    pub fn candidates(&self) -> &NonEmpty<crate::executable::RetainedCandidate<B>> {
        &self.candidates
    }

    pub fn selection_function(&self) -> &crate::prepared::SelectionFunction {
        &self.selection_function
    }

    pub fn candidate_identities(&self) -> impl Iterator<Item = &crate::frozen::VariantIdentity> {
        self.candidates
            .iter()
            .map(crate::executable::RetainedCandidate::identity)
    }

    /// Applies the evaluator-produced function without native handles or
    /// evaluator-specific state.
    pub fn select(
        &self,
        values: &seismic_lang::expr::compiled::InvocationValues,
    ) -> SelectedCandidate<'_, B> {
        let index = crate::executable::select_candidate_index(
            &self.selection_function,
            self.candidates.as_slice(),
            values,
        );
        SelectedCandidate {
            index: crate::prepared::CandidateIndex::from_usize(index),
            candidate: &self.candidates.as_slice()[index],
        }
    }
}

#[derive(Clone, Copy)]
pub struct SelectedCandidate<'a, B: seismic_target::TargetFamily> {
    pub index: crate::prepared::CandidateIndex,
    pub candidate: &'a crate::executable::RetainedCandidate<B>,
}

mod internals {
    use super::*;

    pub(super) fn plan<B: seismic_target::TargetFamily>(
        domain: AnalyticalDomainModel<B>,
        budget: &PlanningBudget,
    ) -> Result<AnalyticalSearchResult<B>, PlanningError> {
        let AnalyticalDomainModelParts {
            domain_token,
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
        let objective_model = {
            let mut builder = SolverModelBuilder::new(&mut arena);
            for (symbol, value) in constants.bindings() {
                builder.bind_target(*symbol, *value);
            }
            let mut unsupported = None;
            for (index, (candidate, predicate)) in optimized.iter().zip(&predicates).enumerate() {
                let family = &candidate.candidate.family;
                let decisions = family
                    .choices()
                    .iter()
                    .map(crate::refinement::ChoiceDeclaration::decision)
                    .collect::<Vec<_>>();
                if let Err(error) = builder.implementation_with_objective(
                    u32::try_from(index + 1).expect("candidate family count exceeds u32"),
                    family.identity().clone(),
                    &decisions,
                    *predicate,
                    candidate.performance.estimate(),
                ) {
                    unsupported = Some(error);
                    break;
                }
            }
            unsupported.map_or_else(|| Ok(builder.build()), Err)
        };
        let (model, unsupported_objective) = match objective_model {
            Ok(model) => (model, None),
            Err(error) => {
                let mut builder = SolverModelBuilder::new(&mut arena);
                for (symbol, value) in constants.bindings() {
                    builder.bind_target(*symbol, *value);
                }
                for (index, (candidate, predicate)) in optimized.iter().zip(predicates).enumerate()
                {
                    let family = &candidate.candidate.family;
                    let decisions = family
                        .choices()
                        .iter()
                        .map(crate::refinement::ChoiceDeclaration::decision)
                        .collect::<Vec<_>>();
                    builder.implementation(
                        u32::try_from(index + 1).expect("candidate family count exceeds u32"),
                        family.identity().clone(),
                        &decisions,
                        predicate,
                    );
                }
                (builder.build(), Some(error))
            }
        };
        let mut accounting = PlanningAccounting::new(budget);
        accounting.record_required_variant(0)?;

        let universal_coordinate =
            canonical_coordinate(domain_token, &arena, &universal.candidate.family, &[])
                .unwrap_or_else(|error| panic!("universal coordinate is not closed: {error:?}"));
        let mut seen = std::collections::HashSet::new();
        seen.insert(universal_coordinate.clone());
        let mut selections = vec![AnalyticalCandidate {
            coordinate: universal_coordinate,
            performance: universal.performance.clone(),
            numerical: universal.candidate.numerical,
        }];

        let mut completion = optimization_exhausted
            .then_some(PlanningLimit::CandidateConstruction)
            .or_else(|| unsupported_objective.map(PlanningLimit::UnsupportedObjective))
            .or_else(|| accounting.optional_limit());
        if !optimization_exhausted {
            let prior_limit = completion.take();
            completion = enumerate_optional(&model, &mut accounting, |assignment| {
                let candidate = optimized
                    .get(assignment.implementation() as usize - 1)
                    .unwrap_or_else(|| panic!("solver selected a family outside the domain"));
                let coordinate = canonical_coordinate(
                    domain_token,
                    &arena,
                    &candidate.candidate.family,
                    assignment.decisions(),
                )
                .unwrap_or_else(|error| {
                    panic!("solver produced a non-canonical coordinate: {error:?}")
                });
                if seen.insert(coordinate.clone()) {
                    selections.push(AnalyticalCandidate {
                        coordinate,
                        performance: candidate.performance.clone(),
                        numerical: candidate.candidate.numerical,
                    });
                }
                0
            });
            completion = completion.or(prior_limit);
        }
        selections =
            exact_prune_and_order(&mut arena, &constants, &universal, &optimized, selections);
        if selections.len() as u64 > budget.executable_variants {
            let retained = usize::try_from(budget.executable_variants)
                .unwrap_or(usize::MAX)
                .max(1);
            selections.truncate(retained);
            completion = Some(PlanningLimit::ExecutableVariants);
        }
        accounting.executable_variants = selections.len() as u64;
        let selections = NonEmpty::new(selections).ok_or(PlanningError::Infeasible(
            PlanningInfeasibleReport {
                entry,
                reason: "evaluated domain omitted its mandatory universal member",
            },
        ))?;
        let coverage = PlanningCoverage {
            target: TargetCoverage::Exhaustive,
            optimization: completion.map_or(
                OptimizationCompletion::Complete,
                OptimizationCompletion::Limited,
            ),
            budget: accounting.report(),
        };
        let domain = AnalyticalDomainModel::from_parts(AnalyticalDomainModelParts {
            domain_token,
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
            arena,
            universal,
            optimized,
            precision,
            optimization_exhausted,
        });
        Ok(AnalyticalSearchResult {
            domain,
            selections,
            coverage,
        })
    }

    #[derive(Clone, Copy)]
    pub(super) struct ExactSelectionFacts {
        pub(super) guard: seismic_lang::expr::BoolExpr,
        pub(super) duration: seismic_lang::expr::DurationExpr,
        pub(super) constant_upper: Option<seismic_lang::expr::RationalDuration>,
        pub(super) numerical_analytic: seismic_lang::expr::BoolExpr,
        pub(super) qualification_allowed: bool,
    }

    fn exact_prune_and_order<B: seismic_target::TargetFamily>(
        arena: &mut ExprArena,
        constants: &TargetConstants,
        universal: &crate::evaluation::EvaluatedUniversal<B>,
        optimized: &[crate::evaluation::EvaluatedCandidate<B>],
        selections: Vec<AnalyticalCandidate>,
    ) -> Vec<AnalyticalCandidate> {
        let facts = selections
            .iter()
            .map(|selection| {
                let candidate = if selection.coordinate.family()
                    == universal.candidate.family.identity()
                {
                    &universal.candidate
                } else {
                    &optimized
                        .iter()
                        .find(|candidate| {
                            candidate.candidate.family.identity() == selection.coordinate.family()
                        })
                        .unwrap_or_else(|| {
                            panic!("analytical selection references a foreign family")
                        })
                        .candidate
                };
                let mut fixed = seismic_lang::expr::PartialAssignment::new();
                for (symbol, value) in constants.bindings() {
                    fixed.bind(*symbol, *value);
                }
                for (decision, value) in selection.coordinate.choices() {
                    fixed.bind(
                        arena.decision_symbol(*decision),
                        seismic_lang::expr::SymbolValue::Int(*value),
                    );
                }
                let guard = arena.partial(candidate.constraints.predicate(), &fixed);
                let duration = arena.partial(selection.performance.estimate(), &fixed);
                let constant_upper = arena
                    .free_symbols(seismic_lang::expr::AnyExpr::Duration(duration))
                    .is_empty()
                    .then(|| {
                        arena
                            .eval_duration(duration, &seismic_lang::expr::Assignment::new())
                            .map(|estimate| estimate.upper())
                            .ok()
                    })
                    .flatten();
                ExactSelectionFacts {
                    guard,
                    duration,
                    constant_upper,
                    numerical_analytic: selection.numerical.analytic_predicate(),
                    qualification_allowed: selection.numerical.qualification_allowed(),
                }
            })
            .collect::<Vec<_>>();
        let stable = |left: usize, right: usize| {
            stable_coordinate_cmp(
                selections[left].coordinate(),
                selections[right].coordinate(),
            )
        };
        let retained = exact_retained_indices(&facts, stable);
        let mut selections = selections.into_iter().map(Some).collect::<Vec<_>>();
        retained
            .into_iter()
            .map(|index| selections[index].take().unwrap())
            .collect()
    }

    pub(super) fn exact_retained_indices(
        facts: &[ExactSelectionFacts],
        stable: impl Fn(usize, usize) -> std::cmp::Ordering + Copy,
    ) -> Vec<usize> {
        let mut retained = (0..facts.len())
            .filter(|candidate| {
                *candidate == 0
                    || !(0..facts.len()).any(|other| {
                        other != *candidate
                            && exactly_dominates(
                                facts[other],
                                facts[*candidate],
                                stable(other, *candidate),
                            )
                    })
            })
            .collect::<Vec<_>>();
        retained
            .get_mut(1..)
            .unwrap_or_default()
            .sort_by(|left, right| {
                let left_facts = facts[*left];
                let right_facts = facts[*right];
                if left_facts.guard == right_facts.guard {
                    if let (Some(left_cost), Some(right_cost)) =
                        (left_facts.constant_upper, right_facts.constant_upper)
                    {
                        return left_cost
                            .cmp(&right_cost)
                            .then_with(|| stable(*left, *right));
                    }
                }
                stable(*left, *right)
            });
        retained
    }

    fn exactly_dominates(
        left: ExactSelectionFacts,
        right: ExactSelectionFacts,
        stable_order: std::cmp::Ordering,
    ) -> bool {
        if left.guard != right.guard
            || left.numerical_analytic != right.numerical_analytic
            || left.qualification_allowed != right.qualification_allowed
        {
            return false;
        }
        if left.duration == right.duration {
            return stable_order.is_lt();
        }
        matches!(
            (left.constant_upper, right.constant_upper),
            (Some(left), Some(right)) if left < right
        )
    }

    fn stable_coordinate_cmp(
        left: &CandidateCoordinate,
        right: &CandidateCoordinate,
    ) -> std::cmp::Ordering {
        let left_family = left.family();
        let right_family = right.family();
        left_family
            .factory
            .name
            .cmp(right_family.factory.name)
            .then_with(|| {
                left_family
                    .factory
                    .revision
                    .cmp(right_family.factory.revision)
            })
            .then_with(|| left_family.structure.cmp(&right_family.structure))
            .then_with(|| {
                left.choices()
                    .iter()
                    .map(|(_, value)| value)
                    .cmp(right.choices().iter().map(|(_, value)| value))
            })
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
            } else {
                None
            }
        }

        fn before_optional_variant(&mut self) -> Option<PlanningLimit> {
            if self.optimized_assignments >= self.limit.optimized_assignments {
                return Some(PlanningLimit::OptimizedAssignments);
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
    use crate::refinement::CandidateFamilyIdentity;
    use crate::refinement::FactoryIdentity;
    use seismic_lang::expr::{AnyExpr, Assignment, CmpOp, DurationTerm, FiniteDomain, SymbolSort};
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
                CandidateFamilyIdentity {
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
            CandidateFamilyIdentity {
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

    #[test]
    fn exact_fixed_cost_dominance_removes_the_slower_coordinate() {
        let mut arena = ExprArena::default();
        let always = arena.bool(true);
        let never = arena.bool(false);
        let numerical = arena.bool(true);
        let one = arena.nat(1);
        let fast = arena.duration(&[DurationTerm {
            demand: one,
            lower_numerator: 3,
            upper_numerator: 3,
            denominator: 1,
        }]);
        let slow = arena.duration(&[DurationTerm {
            demand: one,
            lower_numerator: 7,
            upper_numerator: 7,
            denominator: 1,
        }]);
        let value = |duration| {
            arena
                .eval_duration(duration, &Assignment::new())
                .unwrap()
                .upper()
        };
        let facts = [
            internals::ExactSelectionFacts {
                guard: never,
                duration: fast,
                constant_upper: Some(value(fast)),
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
            internals::ExactSelectionFacts {
                guard: always,
                duration: fast,
                constant_upper: Some(value(fast)),
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
            internals::ExactSelectionFacts {
                guard: always,
                duration: slow,
                constant_upper: Some(value(slow)),
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
        ];

        assert_eq!(
            internals::exact_retained_indices(&facts, |left, right| left.cmp(&right)),
            vec![0, 1]
        );
    }

    #[test]
    fn invocation_dependent_incomparable_costs_are_both_retained() {
        let mut arena = ExprArena::default();
        let always = arena.bool(true);
        let never = arena.bool(false);
        let numerical = arena.bool(true);
        // After coordinate and target binding, an invocation-dependent
        // duration has this same proof state: a residual expression with no
        // scalar constant upper bound. The retention proof never samples it.
        let (_, invocation_like_symbol) = arena.target_constant(SymbolSort::Nat);
        let invocation_like_value = arena.nat_symbol(invocation_like_symbol);
        let scaled = arena.duration(&[DurationTerm {
            demand: invocation_like_value,
            lower_numerator: 1,
            upper_numerator: 1,
            denominator: 1,
        }]);
        let one = arena.nat(1);
        let fixed = arena.duration(&[DurationTerm {
            demand: one,
            lower_numerator: 4,
            upper_numerator: 4,
            denominator: 1,
        }]);
        assert!(!arena.free_symbols(AnyExpr::Duration(scaled)).is_empty());
        let facts = [
            internals::ExactSelectionFacts {
                guard: never,
                duration: fixed,
                constant_upper: None,
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
            internals::ExactSelectionFacts {
                guard: always,
                duration: scaled,
                constant_upper: None,
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
            internals::ExactSelectionFacts {
                guard: always,
                duration: fixed,
                constant_upper: None,
                numerical_analytic: numerical,
                qualification_allowed: false,
            },
        ];

        assert_eq!(
            internals::exact_retained_indices(&facts, |left, right| left.cmp(&right)),
            vec![0, 1, 2]
        );
    }
}
