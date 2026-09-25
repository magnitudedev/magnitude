//! Preparation composition. The evaluator owns analytical estimation, search,
//! retention requests, and its invocation decision. `EvaluationSession` owns
//! every compiler-facing and resource-owning operation.

use crate::candidate_domain::NonEmpty;
use crate::errors::PreparationError;
use crate::evaluation::{AnalyticalEvaluationContext, AnalyticalEvaluator, CandidateEvaluator};
use crate::evaluation_session::{EvaluationSession, RealizationAdmission, RealizationPending};
use crate::planning::{plan, SelectionPolicy};
use crate::preparation_budget::{PlanningBudget, PreparationBudget};
use crate::prepared::{PreparedKernel, SelectionFunction};
use crate::target::CompilerRegistry;
use seismic_lang::entry::LogicalEntry;
use seismic_lang::precision::PrecisionPolicy;
use seismic_native_target::NativeCompiler;

impl<T, C> CandidateEvaluator<T, C> for AnalyticalEvaluator<'_, T>
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
{
    fn evaluate(
        &mut self,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<
        (
            SelectionPolicy,
            crate::evaluation::EvaluationIdentity,
            crate::planning::PlanningReport,
        ),
        PreparationError,
    > {
        let mut construction =
            crate::evaluation::construction::ConstructionTraversal::new(session.domain());
        let mut allowance = session.construction_allowance();
        construction.advance(session.domain_mut(), &mut allowance);
        let mut evaluated = self
            .evaluate_domain(session.domain_mut())
            .map_err(PreparationError::Evaluation)?;
        evaluated.construction_complete = construction.complete();
        let budget = session.search_budget().clone();
        let structural =
            plan(session.domain_mut(), evaluated, &budget).map_err(PreparationError::Planning)?;
        let (evaluation, selections, mut report) = structural.into_parts();
        let mut retained = Vec::new();
        for selection in selections.into_vec() {
            match session.realize_checked(selection.coordinate())? {
                RealizationAdmission::Ready(candidate) => {
                    retained.push((candidate, selection.performance().clone()))
                }
                RealizationAdmission::Rejected(_) => {}
                RealizationAdmission::Pending(RealizationPending::NumericalAnalysis) => {
                    report.optimization = crate::planning::OptimizationCompletion::Limited(
                        crate::planning::PlanningLimit::NumericalAnalysis,
                    );
                }
                RealizationAdmission::Pending(RealizationPending::NativeBudget) => {
                    report.optimization = crate::planning::OptimizationCompletion::Limited(
                        crate::planning::PlanningLimit::NativeArtifacts,
                    );
                    break;
                }
            }
        }
        // The analytical strategy owns this retention choice. Finalization never trims.
        loop {
            let candidates = NonEmpty::new(retained.clone()).ok_or_else(|| {
                PreparationError::UniversalClosure(
                    "analytical preparation did not admit its universal candidate".into(),
                )
            })?;
            match session.policy(candidates, |arena, selections| {
                SelectionFunction::analytical_minimum(selections.map_payload(
                    |performance, fixed| arena.compile_duration_with(performance.estimate(), fixed),
                ))
            }) {
                Ok(policy) => {
                    return Ok((policy, evaluation, report));
                }
                Err(PreparationError::Planning(
                    crate::planning::PlanningError::BudgetExceeded { .. },
                )) if retained.len() > 1 => {
                    retained.pop();
                    report.optimization = crate::planning::OptimizationCompletion::Limited(
                        crate::planning::PlanningLimit::RetainedMetadata,
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }
}

pub fn prepare_analytically<T, C>(
    entry: LogicalEntry,
    analytical: &AnalyticalEvaluationContext<T>,
    registry: &CompilerRegistry<T>,
    compiler: &C,
    native_context: &C::Context,
    precision: &PrecisionPolicy,
    preparation_budget: &PreparationBudget,
    planning_budget: &PlanningBudget,
) -> Result<PreparedKernel<T, C::Handle>, PreparationError>
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
{
    let mut session = EvaluationSession::new(
        entry,
        analytical.device(),
        registry,
        compiler,
        native_context,
        precision,
        preparation_budget,
        planning_budget,
    )?;
    prepare_with_evaluator(&mut session, &mut AnalyticalEvaluator::new(analytical))
}

/// Run one strategy turn and publish exactly its checked retained policy.
/// The session remains available for further evaluation; each published kernel
/// independently owns the admitted artifacts it retains.
pub fn prepare_with_evaluator<T, C, E>(
    session: &mut EvaluationSession<'_, T, C>,
    evaluator: &mut E,
) -> Result<PreparedKernel<T, C::Handle>, PreparationError>
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
    E: CandidateEvaluator<T, C>,
{
    evaluate_and_finalize(session, evaluator).0
}

/// The same orchestration also returns publication time to a continuing search
/// campaign. Timing is observation only; it cannot change the returned policy.
pub(crate) fn evaluate_and_finalize<T, C, E>(
    session: &mut EvaluationSession<'_, T, C>,
    evaluator: &mut E,
) -> (
    Result<PreparedKernel<T, C::Handle>, PreparationError>,
    std::time::Duration,
)
where
    T: seismic_native_target::TargetFamily,
    C: NativeCompiler<T>,
    E: CandidateEvaluator<T, C>,
{
    let (policy, identity, report) = match evaluator.evaluate(session) {
        Ok(result) => result,
        Err(error) => return (Err(error), std::time::Duration::ZERO),
    };
    let started = std::time::Instant::now();
    let result = session.finalize(policy, identity, report);
    (result, started.elapsed())
}
