//! Explicit preparation ownership keeps continuation out of immutable kernels.
use super::controller::FeedbackEvaluator;
use super::{ControlledObserver, FeedbackOptions, FeedbackReport};
use crate::errors::PreparationError;
use crate::evaluation_session::EvaluationSession;
use crate::preparation_budget::{PlanningBudget, PreparationBudget};
use crate::prepared::PreparedKernel;
use crate::target::CompilerRegistry;
use seismic_lang::{entry::LogicalEntry, precision::PrecisionPolicy};
use seismic_target::{DeviceDescription, NativeCompiler, TargetFamily};
use std::time::{Duration, Instant};

/// An in-process search campaign. Returned kernels own only their executable
/// resources and selection function; dropping this campaign cannot invalidate
/// them. Continuing the campaign never mutates an earlier result.
pub struct FeedbackPreparation<'a, T: TargetFamily, C: NativeCompiler<T>, O> {
    session: EvaluationSession<'a, T, C>,
    evaluator: FeedbackEvaluator<O>,
}

impl<'a, T: TargetFamily, C: NativeCompiler<T>, O: ControlledObserver<T, C::Handle>>
    FeedbackPreparation<'a, T, C, O>
{
    pub fn start(
        entry: LogicalEntry,
        device: &'a DeviceDescription<T>,
        registry: &'a CompilerRegistry<T>,
        compiler: &'a C,
        native_context: &'a C::Context,
        precision: &PrecisionPolicy,
        preparation_budget: &PreparationBudget,
        planning_budget: &PlanningBudget,
        observer: O,
        options: FeedbackOptions,
    ) -> Result<(Self, PreparedKernel<T, C::Handle>), PreparationError> {
        let started = Instant::now();
        let session = EvaluationSession::new(
            entry,
            device,
            registry,
            compiler,
            native_context,
            precision,
            preparation_budget,
            planning_budget,
        )?;
        let evaluator = FeedbackEvaluator::new(&session, observer, options, started)?;
        let mut campaign = Self { session, evaluator };
        let kernel = campaign.run()?;
        Ok((campaign, kernel))
    }

    pub fn continue_for(
        &mut self,
        additional: Duration,
    ) -> Result<PreparedKernel<T, C::Handle>, PreparationError> {
        self.evaluator.resume(additional);
        self.session.extend_native_time(additional);
        self.run()
    }

    pub fn report(&self) -> &FeedbackReport {
        self.evaluator.report()
    }

    fn run(&mut self) -> Result<PreparedKernel<T, C::Handle>, PreparationError> {
        let (result, publication_time) =
            crate::prepare::evaluate_and_finalize(&mut self.session, &mut self.evaluator);
        self.evaluator.finalized(
            publication_time,
            result.as_ref().ok().map(|kernel| kernel.planning_report()),
        );
        self.evaluator.suspend();
        result
    }
}
