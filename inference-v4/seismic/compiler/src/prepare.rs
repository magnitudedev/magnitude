//! Preparation composition. The evaluator owns analytical estimation, search,
//! retention requests, and its invocation decision. `EvaluationSession` owns
//! every compiler-facing and resource-owning operation.

use crate::candidate_domain::{construct_candidate_domain, NonEmpty};
use crate::errors::PreparationError;
use crate::evaluation::{AnalyticalEvaluationContext, AnalyticalEvaluator, CandidateEvaluator};
use crate::evaluation_session::{
    EvaluationCompletion, EvaluationRequest, EvaluationSession, RealizationAdmission,
};
use crate::numerics::EvidenceCatalog;
use crate::planning::{plan, SelectionPolicy};
use crate::preparation_budget::{PlanningBudget, PreparationBudget};
use crate::prepared::{PreparedKernel, SelectionFunction};
use crate::target::CompilerRegistry;
use seismic_lang::entry::LogicalEntry;
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::NativeCompiler;

impl<T, C> CandidateEvaluator<T, C> for AnalyticalEvaluator<'_, T>
where
    T: seismic_target::TargetFamily,
    C: NativeCompiler<T>,
{
    fn evaluate(
        &self,
        domain: crate::candidate_domain::CandidateDomain<T>,
        session: &mut EvaluationSession<'_, T, C>,
    ) -> Result<SelectionPolicy<T>, PreparationError> {
        let evaluated = self
            .evaluate_domain(domain)
            .map_err(PreparationError::Evaluation)?;
        let structural =
            plan(evaluated, session.search_budget()).map_err(PreparationError::Planning)?;
        let (domain, evaluation, selections, coverage) = structural.into_parts();
        session.begin(domain, evaluation, coverage)?;
        let mut admitted = Vec::new();
        for selection in selections.into_vec() {
            let request = EvaluationRequest::new(
                selection.coordinate().clone(),
                selection.performance().clone(),
            );
            match session.realize_checked(request)? {
                RealizationAdmission::Admitted(candidate) => admitted.push(candidate),
                RealizationAdmission::Rejected => {}
                RealizationAdmission::BudgetClosed => break,
            }
        }
        let admitted =
            NonEmpty::new(admitted).expect("analytical search admits its universal coordinate");
        session.publish(admitted, |arena, selections| {
            let scores = selections.map_payload(|performance, fixed| {
                arena.compile_duration_with(performance.estimate(), fixed)
            });
            SelectionFunction::analytical_minimum(scores)
        })
    }
}

pub fn prepare_analytically<T, C>(
    entry: LogicalEntry,
    analytical: &AnalyticalEvaluationContext<T>,
    registry: &CompilerRegistry<T>,
    compiler: &C,
    native_context: &C::Context,
    precision: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    preparation_budget: &PreparationBudget,
    planning_budget: &PlanningBudget,
) -> Result<PreparedKernel<T, C::Handle>, PreparationError>
where
    T: seismic_target::TargetFamily,
    C: NativeCompiler<T>,
{
    let domain = construct_candidate_domain(
        entry,
        analytical.device(),
        registry,
        precision,
        evidence,
        preparation_budget,
    )?;
    let mut session = EvaluationSession::new(
        registry,
        compiler,
        native_context,
        analytical.device(),
        preparation_budget,
        planning_budget,
    );
    let policy = <AnalyticalEvaluator<'_, T> as CandidateEvaluator<T, C>>::evaluate(
        &AnalyticalEvaluator::new(analytical),
        domain,
        &mut session,
    )?;
    let completion = session.take_completion().ok_or_else(|| {
        PreparationError::InvalidCandidateDomain(
            "candidate evaluator returned without completing preparation context".into(),
        )
    })?;
    Ok(materialize(policy, completion))
}

pub(crate) fn materialize<T: seismic_target::TargetFamily, H>(
    policy: SelectionPolicy<T>,
    completion: EvaluationCompletion<T, H>,
) -> PreparedKernel<T, H> {
    let SelectionPolicy {
        candidates,
        selection_function,
    } = policy;
    let EvaluationCompletion {
        entry,
        module,
        schema,
        semantic_events,
        device,
        evaluation,
        invocation,
        coverage,
        realizations,
    } = completion;
    let variants = candidates
        .into_vec()
        .into_iter()
        .map(|variant| crate::executable::materialize_variant(variant, &realizations))
        .collect::<Vec<_>>();
    let variants = NonEmpty::new(variants).expect("a policy always contains its general variant");
    PreparedKernel::prepare(
        entry,
        module,
        schema,
        semantic_events,
        device,
        evaluation,
        invocation,
        selection_function,
        variants,
        coverage,
    )
}
