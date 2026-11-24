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
            bank.decode_rows(),
        )))
    }

    /// Derived minimum expected per-row miss-DMA bytes for 3-stage `DecodeHits`
    /// splitting: `overhead_s / transfer_s_per_byte`, where `overhead_s` is the
    /// measured extra host-sync submission plus the 2 split kernel launches
    /// (`routed_gate_up_part` + `routed_down_project`), and `transfer_s_per_byte`
    /// is the measured H2D `transfer_copy` bandwidth (falling back to the link's
    /// weight read line if `TierCopy` has not been recorded yet).
    ///
    /// Returns `u64::MAX` when measured H2D DMA + resident VRAM read is not
    /// faster per byte than SM zero-copy over the link (consistent with
    /// [`crate::experts::transport::SplitDmaCost`]), so 3-stage `DecodeHits`
    /// graphs are not compiled on machines where `Whole` always wins.
    pub(crate) fn decode_hits_min_bytes(&self, bank: &super::ExpertBankPlan) -> u64 {
        let overhead_s = self.costs.step_round_trip_seconds.value + 2.0 * self.costs.launch();
        if !overhead_s.is_finite() || overhead_s <= 0.0 {
            return 0;
        }
        let least_bytes = bank.least_move_bytes().ok().map(|b| b as f64);
        let layer0_gpu = crate::assessment::expert_pass(bank, &self.costs)
            .ok()
            .and_then(|pass| pass.lines(0)?.ok());
        if let (Some(transfer), Some(bytes), Some(gpu)) = (self.figures.transfer.as_ref(), least_bytes, layer0_gpu) {
            if bytes > 0.0 {
                let link_s = gpu.link.weights.get() * 1e-9;
                let resident_s = gpu.resident.weights.get() * 1e-9;
                let cost = crate::experts::transport::SplitDmaCost::from_measured(
                    transfer.seconds_per_byte() + resident_s / bytes,
                    link_s / bytes,
                    transfer.latency_seconds() + self.costs.step_round_trip_seconds.value,
                    bytes as u64,
                );
                if !cost.can_ever_win() {
                    return u64::MAX;
                }
            }
        }
        let transfer_s_per_byte: Option<f64> = self
            .figures
            .transfer
            .as_ref()
            .map(magnitude_cost_model::ReadLine::seconds_per_byte)
            .or_else(|| {
                let bytes = least_bytes?;
                let gpu = layer0_gpu?;
                let link_s = gpu.link.weights.get() * 1e-9;
                (bytes > 0.0 && link_s > 0.0).then_some(link_s / bytes)
            });
        let Some(s_per_byte) = transfer_s_per_byte.filter(|s| s.is_finite() && *s > 0.0) else {
            return 0;
        };
        let min_bytes = (overhead_s / s_per_byte).ceil();
        if !min_bytes.is_finite() || min_bytes >= u64::MAX as f64 {
            u64::MAX
        } else {
            min_bytes as u64
        }
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

