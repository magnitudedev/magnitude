//! Preparation composition across construction, evaluation, planning and
//! exact native materialization.
//!
//! This is the only owner that retains the realization registry while the
//! handle-free domain crosses evaluator and planner boundaries.

use crate::candidate_domain::{construct_candidate_domain, NonEmpty};
use crate::errors::PreparationError;
use crate::evaluation::{AnalyticalEvaluationContext, AnalyticalEvaluator, CandidateEvaluator};
use crate::numerics::EvidenceCatalog;
use crate::planning::{plan, PlannedPolicy};
use crate::preparation_budget::{PlanningBudget, PreparationBudget};
use crate::prepared::PreparedKernel;
use crate::realization::RealizationRegistry;
use crate::target::CompilerRegistry;
use seismic_lang::entry::LogicalEntry;
use seismic_lang::precision::PrecisionPolicy;

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
    C: seismic_target::NativeCompiler<T>,
{
    let (domain, realizations) = construct_candidate_domain(
        entry,
        analytical.device(),
        registry,
        compiler,
        native_context,
        precision,
        evidence,
        preparation_budget,
    )?;
    let evaluated = AnalyticalEvaluator::new(analytical)
        .evaluate(domain)
        .map_err(PreparationError::Evaluation)?;
    let policy = plan(evaluated, planning_budget).map_err(PreparationError::Planning)?;
    Ok(materialize(policy, realizations))
}

pub(crate) fn materialize<T: seismic_target::TargetFamily, H>(
    policy: PlannedPolicy<T>,
    realizations: RealizationRegistry<T, H>,
) -> PreparedKernel<T, H> {
    let PlannedPolicy {
        entry,
        module,
        schema,
        semantic_events,
        device,
        evaluation,
        invocation,
        variants,
        coverage,
    } = policy;
    let variants = variants
        .into_vec()
        .into_iter()
        .map(|variant| {
            let planned_bytes = variant.retained_metadata_bytes();
            let executable = crate::executable::materialize_variant(variant, &realizations);
            debug_assert!(
                executable.retained_metadata_bytes() <= planned_bytes,
                "materialization exceeded the metadata charged by planning"
            );
            executable
        })
        .collect::<Vec<_>>();
    let variants =
        NonEmpty::new(variants).expect("a planned policy always contains its universal variant");
    PreparedKernel::prepare(
        entry,
        module,
        schema,
        semantic_events,
        device,
        evaluation,
        invocation,
        variants,
        coverage,
    )
}
