//! The executor's side of a plan's bank prices
//! ([`magnitude_cost_model::BankPrices`]): the executor's costs, the
//! machine's streamed figures and its decode step in each form, from which
//! a bank's prices are built for the plan that holds it ([`PlanPrices`]).

use super::{ExecutionPlanDraft, ResourceLimits};
use crate::assessment::{AssessmentError, ExecutorCosts, Priced, StreamedFigures};
use crate::error::PlanError;
use magnitude_cost_model::{BankPrices, FormSeconds, PassLayers, Prior, PriorOverflow};
use magnitude_family_contracts::ModelDefinition;

/// A plan's launch of some rows at some depth in each form, as its
/// assessment prices it on this executor (`LaunchPrices::launch`).
pub type PlanForms = dyn Fn(u32, u32) -> Result<FormSeconds, AssessmentError>;

/// What prices a plan's decode step in each form: of `definition` planned
/// as the plan that banks its experts, its step `C` (a launch of some rows
/// at some depth, the plan before any bank's and the banked plan's
/// streamed form's).
pub type PlanSteps = dyn Fn(&ModelDefinition, &ExecutionPlanDraft) -> Result<Box<PlanForms>, AssessmentError>;

/// The prices of a plan's holdings: the executor's costs, the machine's
/// streamed figures, and its decode step in each form.
pub struct PlanPrices {
    costs: ExecutorCosts,
    figures: StreamedFigures,
    steps: Box<PlanSteps>,
}

impl std::fmt::Debug for PlanPrices {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PlanPrices").field("figures", &self.figures).finish_non_exhaustive()
    }
}

impl From<PriorOverflow> for PlanError {
    fn from(_: PriorOverflow) -> Self {
        PlanError::Arithmetic("a prior's figure exceeds u32")
    }
}

/// The workload prior of a load planning `context` rows of context with
/// `limits` ([`Prior`]).
pub(crate) fn prior(context: u64, limits: ResourceLimits) -> Result<Prior, PlanError> {
    let region_rows = magnitude_batching::row_classes(limits.max_launch_rows)
        .last()
        .copied()
        .expect("a launch's row bound has a row class") as u64;
    Ok(Prior::of(context, limits.max_launch_rows as u64, region_rows)?)
}

impl PlanPrices {
    pub fn new(costs: ExecutorCosts, figures: StreamedFigures, steps: Box<PlanSteps>) -> Self {
        Self { costs, figures, steps }
    }

    /// The prices of the holdings of `banked`'s bank, `definition` planned
    /// as `banked`, under its workload prior, its decode steps the step `C`
    /// in each form and its rates on the resident form's: the misses a step
    /// adds are what the settle sizes, so the prior cannot read them first.
    /// Serving starts its meter at the step the bank it holds then prices
    /// (`ExecutorDomain::serving_prior`).
    /// Pending or unavailable as its layers' lines are (`streamed_layers`).
    pub(crate) fn of_bank(&self, banked: &ExecutionPlanDraft, definition: &ModelDefinition) -> Result<Priced<BankPrices>, PlanError> {
        let priced = |error: String| PlanError::ResourcePlanning(format!("the plan's prices: {error}"));
        let bank = banked.load().expert_bank().ok_or_else(|| priced("the plan holds no bank".into()))?;
        let limits = banked.policy().limits();
        let prior = prior(definition.decoder.context_limit, limits)?;
        let step = (self.steps)(definition, banked).map_err(|error| priced(error.to_string()))?;
        let step = |rows: u32, depth: u32| step(rows, depth).map_err(|error| priced(error.to_string()));
        let compute = step(1, prior.decode_depth)?;
        let rates = prior.rates(compute.resident, |rows, depth| step(rows, depth).map(|step| step.resident))?;
        let layers = match crate::assessment::streamed_layers(bank, &self.costs, &self.figures) {
            Priced::Measured(layers) => layers,
            Priced::Pending(reason) => return Ok(Priced::Pending(reason)),
            Priced::Unavailable(reason) => return Ok(Priced::Unavailable(reason)),
        };
        let pass = crate::assessment::expert_pass(bank, &self.costs).map_err(|error| priced(error.to_string()))?;
        let routed = PassLayers::of(&pass, bank.layers().iter().map(|layer| layer.experts), bank.selected());
        Ok(Priced::Measured(BankPrices::new(
            layers,
            pass,
            routed,
            bank.windows(),
            self.figures.read,
            compute,
            rates,
            prior.region_rows,
            // A pass is sized for the prompts a load's chain takes, every
            // chunk of a whole-context one (`ExecutionPlanDraft::prompt_places`):
            // a pass the prior's half-context prompt fills would split
            // longer prompts into more passes, each reading its misses again.
            definition.decoder.context_limit,
        )))
    }
}

/// The prices a load's settle takes of `priced`: a bank waits for prices
/// that are not measured yet (`PlanError::Pending`), or fails where they
/// cannot be obtained (`PlanError::Unavailable`).
pub(crate) fn measured<T>(priced: Priced<T>) -> Result<T, PlanError> {
    match priced {
        Priced::Measured(prices) => Ok(prices),
        Priced::Pending(pending) => Err(PlanError::Pending(pending)),
        Priced::Unavailable(reason) => Err(PlanError::Unavailable(reason)),
    }
}

