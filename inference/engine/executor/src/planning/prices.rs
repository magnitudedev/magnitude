//! What the plan's settle prices its holdings with (`domain::value`): the
//! generic prior assessment estimates with, from the model's headers and
//! the machine's standing figures alone (`ExecutorCosts`,
//! `StreamedFigures`), never live data or recorded data of a model. A bank
//! slot is worth the decode seconds it saves (`StreamedLayers::step_seconds`
//! on the step `C` of the form its slots decode in) times decode steps a
//! second; a prompt pass
//! size the routed prefill seconds its fewer passes save
//! (`routed_prompt_seconds`) times requests a second; both rates the
//! workload prior's ([`WorkloadRates::prior`]).

use super::{ExecutionPlanDraft, PassLaunches, ResourceLimits};
use crate::assessment::{AssessmentError, ExecutorCosts, ExpertPass, FormSeconds, HeldSlots, StreamedFigures, StreamedLayers};
use crate::error::PlanError;
use crate::workload::WorkloadRates;
use magnitude_family_contracts::ModelDefinition;

/// What prices a plan's decode step in each form: of `definition` planned
/// as the plan that banks its experts, its step `C` (`crate::DecodeStep`: a
/// launch of some rows at some depth, the plan before any bank's and the
/// banked plan's streamed form's).
pub type PlanSteps = dyn Fn(&ModelDefinition, &ExecutionPlanDraft) -> Result<Box<crate::DecodeStep>, AssessmentError>;

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
    pub(crate) fn of_bank(&self, banked: &ExecutionPlanDraft, definition: &ModelDefinition) -> Result<BankPrices, PlanError> {
        let priced = |error: String| PlanError::ResourcePlanning(format!("the plan's prices: {error}"));
        let bank = banked.load().expert_bank().ok_or_else(|| priced("the plan holds no bank".into()))?;
        let limits = banked.policy().limits();
        let prior = Prior::of(definition.decoder.context_limit, limits)?;
        let step = (self.steps)(definition, banked).map_err(|error| priced(error.to_string()))?;
        let step = |rows: u32, depth: u32| step(rows, depth).map_err(|error| priced(error.to_string()));
        let compute = step(1, prior.decode_depth)?;
        let rates = prior.rates(compute.resident, |rows, depth| step(rows, depth).map(|step| step.resident))?;
        Ok(BankPrices {
            layers: StreamedLayers::of_bank(bank, &self.costs, &self.figures).map_err(|error| priced(error.to_string()))?,
            pass: ExpertPass::of_bank(bank, self.costs.clone()).map_err(|error| priced(error.to_string()))?,
            routed: bank.layers().iter().map(|layer| (layer.experts, bank.selected())).collect(),
            read: self.figures.read,
            compute,
            rates,
            region_rows: prior.region_rows,
            prompt_rows: prior.prefill_tokens,
        })
    }
}

/// The workload prior before a load serves, with no request history to
/// read: one stream that prefills half the planned context, then decodes
/// the other half, a row at a time. The plan prices its holdings with it,
/// and serving starts every load's meter from it, banked or not
/// (`ExecutorDomain::serving_rates`).
pub(crate) struct Prior {
    prefill_tokens: u64,
    decode_steps: u64,
    /// The depth a decode step is priced at: the middle of the decode.
    pub(crate) decode_depth: u32,
    /// The prefill's launches of `launch_rows` rows each, priced at the
    /// middle of the prompt.
    launch_rows: u32,
    prefill_launches: u64,
    prefill_depth: u32,
    /// The rows of one launch's region of a prompt pass
    /// (`pass_buffer_bytes`'s widest row class).
    pub(crate) region_rows: u64,
}

impl Prior {
    /// The prior of a load planning `context` rows of context with
    /// `limits`.
    pub(crate) fn of(context: u64, limits: ResourceLimits) -> Result<Self, PlanError> {
        let narrow = |figure: u64| u32::try_from(figure).map_err(|_| PlanError::Arithmetic("a prior's figure exceeds u32"));
        let context = context.max(2);
        let prefill_tokens = context / 2;
        let decode_steps = context - prefill_tokens;
        let launch_rows = limits.max_launch_rows as u64;
        let region_rows = magnitude_batching::row_classes(limits.max_launch_rows)
            .last()
            .copied()
            .expect("a launch's row bound has a row class") as u64;
        Ok(Self {
            prefill_tokens,
            decode_steps,
            decode_depth: narrow(prefill_tokens + decode_steps / 2)?,
            launch_rows: narrow(launch_rows)?,
            prefill_launches: prefill_tokens.div_ceil(launch_rows),
            prefill_depth: narrow(prefill_tokens / 2)?,
            region_rows,
        })
    }

    /// The prior's rates, its decode steps `step_seconds` each and its
    /// prefill's launches priced on the resident step `resident(rows,
    /// depth)`.
    pub(crate) fn rates<E>(&self, step_seconds: f64, resident: impl Fn(u32, u32) -> Result<f64, E>) -> Result<WorkloadRates, E> {
        let launch = resident(self.launch_rows, self.prefill_depth)?;
        let prefill_seconds = self.prefill_launches as f64 * launch;
        Ok(WorkloadRates::prior(step_seconds, self.decode_steps as f64, self.prefill_tokens as f64, prefill_seconds))
    }
}

/// The prices of one bank's holdings under the workload prior.
pub(crate) struct BankPrices {
    layers: StreamedLayers,
    pass: ExpertPass,
    /// Each routed layer's `(experts, selected)`.
    routed: Vec<(u64, u64)>,
    read: crate::assessment::ReadLine,
    /// `C`: the step at the prior's row and depth in each form.
    compute: FormSeconds,
    rates: WorkloadRates,
    region_rows: u64,
    prompt_rows: u64,
}

impl BankPrices {
    /// A decode step's seconds holding `held` of each layer's experts.
    pub(crate) fn step_seconds(&self, held: HeldSlots) -> f64 {
        self.layers.step_seconds(None, self.read, held, 1.0, self.compute).total_seconds
    }

    /// The prior's prompt's routed seconds in passes of `pass` launches
    /// holding `held`: each pass's compute and the reads of its misses
    /// ([`StreamedLayers::pass_reads`]).
    pub(crate) fn prompt_seconds(&self, pass: PassLaunches, held: HeldSlots) -> f64 {
        let reads = |rows: u64| self.layers.pass_reads(self.read, held, rows as f64);
        super::routed_prompt_seconds(&self.pass, &self.routed, self.region_rows, self.prompt_rows, pass, &reads)
    }

    pub(crate) fn rates(&self) -> WorkloadRates {
        self.rates
    }
}
