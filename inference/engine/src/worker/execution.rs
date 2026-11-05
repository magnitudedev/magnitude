//! The worker's thread-confined execution owner over the family-neutral
//! scheduler. Only the closed scheduler command vocabulary reaches it.

use crate::error::UnloadCause;
use crate::execution::{ExpertPricing, expert_measures};
use magnitude_executor::ProgramFamily;
use magnitude_generation::Method;
use magnitude_scheduler::{
    owner::{AdmissionError, Owner, Ran},
    protocol::{AdmitRequest, RoomGrant, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, RequestError},
    worker::{Driven, Event},
};
use std::sync::Arc;

/// The execution lifecycle. Every transition moves the owner by value.
enum Execution<F: ProgramFamily> {
    Serving(Owner<F>),
    /// Every request has been terminated with the cause; the in-flight group
    /// finishes, then `Stopped`.
    Draining(Owner<F>, UnloadCause),
    Stopped(UnloadCause),
}

pub(crate) struct ExecutionOwner<F: ProgramFamily> {
    execution: Execution<F>,
    /// The owner whose execution stopped, what it owed settled: it drops
    /// after the stop is announced, so a worker process's end, which needs
    /// none of its teardown, can answer first and cut that teardown short.
    ended: Option<Owner<F>>,
    method: Arc<dyn Method>,
    /// What prices the bank's experts, which prices them again, measuring
    /// nothing, once the service stores new measurement records; none
    /// without a bank.
    pricing: Option<ExpertPricing>,
}

impl<F: ProgramFamily> ExecutionOwner<F> {
    pub fn new(owner: Owner<F>, method: Arc<dyn Method>, pricing: Option<ExpertPricing>) -> Self {
        Self {
            execution: Execution::Serving(owner),
            ended: None,
            method,
            pricing,
        }
    }

    /// Move the execution out for a transition. The placeholder is truthful:
    /// a transition that panics has stopped execution.
    fn take(&mut self) -> Execution<F> {
        std::mem::replace(
            &mut self.execution,
            Execution::Stopped(UnloadCause::Internal {
                reason: "execution transition interrupted".into(),
            }),
        )
    }
}

/// The admission refusal of an execution that is draining or stopped.
fn unloaded(cause: &UnloadCause) -> AdmissionError {
    match cause {
        UnloadCause::MemoryPressure => AdmissionError::ModelUnloaded {
            cause: ModelUnloadCause::MemoryPressure,
        },
        UnloadCause::Shutdown | UnloadCause::DeviceLost { .. } | UnloadCause::Internal { .. } => {
            AdmissionError::Refused(RequestError::WorkerClosed)
        }
    }
}

impl<F: ProgramFamily> Driven for ExecutionOwner<F> {
    type Stopped = UnloadCause;

    fn command(&mut self, command: WorkerCommand, now: u64) -> Result<WorkerReply, String> {
        match command {
            WorkerCommand::Admit(AdmitRequest {
                seed,
                input,
                retention,
                output_capacity,
            }) => match &mut self.execution {
                Execution::Serving(owner) => {
                    let generation = seed.into_generation(self.method.clone())?;
                    Ok(
                        match owner.admit(generation, input, retention, output_capacity, now) {
                            Ok((request, receiver)) => WorkerReply::Admitted { request, receiver },
                            Err(error) => WorkerReply::AdmissionRefused(error),
                        },
                    )
                }
                Execution::Draining(_, cause) | Execution::Stopped(cause) => {
                    Ok(WorkerReply::AdmissionRefused(unloaded(cause)))
                }
            },
            WorkerCommand::Stop { request } => {
                if let Execution::Serving(owner) | Execution::Draining(owner, _) =
                    &mut self.execution
                {
                    owner.stop(request);
                }
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Status { request } => Ok(WorkerReply::Status(match &self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => {
                    owner.snapshot(request)
                }
                Execution::Stopped(_) => None,
            })),
            WorkerCommand::Observe => match &self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => owner.observe(),
                Execution::Stopped(cause) => Err(format!("execution stopped: {cause}")),
            },
            WorkerCommand::ClaimMeasurementRoom { bytes } => Ok(WorkerReply::MeasurementRoom(match &mut self.execution {
                Execution::Serving(owner) => owner.claim_measurement_room(bytes),
                // An ending model holds no claim; its end answers the host.
                Execution::Draining(..) | Execution::Stopped(_) => RoomGrant::released(),
            })),
            WorkerCommand::ReleaseMeasurementRoom => {
                if let Execution::Serving(owner) | Execution::Draining(owner, _) = &mut self.execution {
                    owner.release_measurement_room();
                }
                Ok(WorkerReply::Acknowledged)
            }
            WorkerCommand::Reprice => {
                match (&mut self.execution, &mut self.pricing) {
                    (Execution::Serving(owner), Some(pricing)) => owner.measure_experts(expert_measures(pricing)),
                    // A model without a bank prices nothing of its experts, and an ending one
                    // has nothing left to price.
                    (Execution::Serving(_), None) | (Execution::Draining(..) | Execution::Stopped(_), _) => {}
                }
                Ok(WorkerReply::Acknowledged)
            }
        }
    }

    fn event(&mut self, event: Event, _: u64) {
        match event {
            // The owner reconciles its flight when it next runs.
            Event::Completion => {}
            Event::Publication(request, wake) => match &mut self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => {
                    owner.publication_wake(request, wake)
                }
                Execution::Stopped(_) => {}
            },
            Event::Cancel(request) => match &mut self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => owner.cancel(request),
                Execution::Stopped(_) => {}
            },
            Event::Close => {
                self.execution = match self.take() {
                    Execution::Serving(mut owner) => {
                        owner.terminate_all(RequestError::WorkerClosed);
                        Execution::Draining(owner, UnloadCause::Shutdown)
                    }
                    draining @ Execution::Draining(..) => draining,
                    stopped @ Execution::Stopped(_) => stopped,
                }
            }
        }
    }

    fn run(&mut self, now: u64) -> Result<u64, UnloadCause> {
        loop {
            match self.take() {
                Execution::Serving(owner) => match owner.run(now) {
                    Ok((owner, Ran { unload: true, .. })) => {
                        self.execution = Execution::Draining(owner, UnloadCause::MemoryPressure)
                    }
                    Ok((owner, Ran { due, .. })) => {
                        self.execution = Execution::Serving(owner);
                        return Ok(due);
                    }
                    Err(error) => self.execution = Execution::Stopped(UnloadCause::from(&error)),
                },
                Execution::Draining(owner, cause) => match owner.run(now) {
                    Ok((owner, Ran { in_flight: false, .. })) => {
                        self.ended = Some(owner);
                        self.execution = Execution::Stopped(cause)
                    }
                    Ok((owner, Ran { due, .. })) => {
                        self.execution = Execution::Draining(owner, cause);
                        return Ok(due);
                    }
                    Err(error) => self.execution = Execution::Stopped(UnloadCause::from(&error)),
                },
                Execution::Stopped(cause) => {
                    // Before the stop is announced, every durable effect of
                    // the load is made (`magnitude_executor::settle`).
                    magnitude_executor::settle::settle();
                    self.execution = Execution::Stopped(cause.clone());
                    return Err(cause);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{
        fixtures, platform, CompletionWake, ComponentSelection, ExecutionPath,
        ExecutionPlanner, KernelCache, PlannedMethod, RequestId,
    };
    use magnitude_generation::Plain;
    use magnitude_executor::BankResize;
    use magnitude_scheduler::{publication::PublicationWake, worker::Wakes, ServiceLimits};
    use seismic::measure::{Attempt, Reason};

    /// The owner's wakes, which a test that runs it itself needs none of.
    struct Unwoken;

    impl Wakes for Unwoken {
        fn completion(&self) -> CompletionWake {
            CompletionWake::new(|| {})
        }

        fn publication(&self, _: RequestId, _: PublicationWake) {}
    }

    /// The bank's slots and its last resize, as an observation reads them.
    fn slots<F: ProgramFamily>(execution: &mut ExecutionOwner<F>) -> (u64, Option<BankResize>) {
        let Ok(WorkerReply::Observed { serving, .. }) = execution.command(WorkerCommand::Observe, 0) else {
            panic!("an observation of a serving owner")
        };
        let experts = serving.experts.expect("a banked load");
        (experts.slots, experts.last_resize)
    }

    /// Run the owner over `observations` of its memory observations, as the
    /// worker wakes for each, from `now`; the clock after.
    fn observe<F: ProgramFamily>(execution: &mut ExecutionOwner<F>, mut now: u64, observations: usize) -> u64 {
        for _ in 0..observations {
            now = execution.run(now).unwrap().max(now + 1);
        }
        now
    }

    /// A live load priced before its executor's costs were stored grows
    /// nothing, however long memory stays free; once the measurement job
    /// stores them (here its probe, which gives the executor's costs) and
    /// the host asks it to reprice, the execution owner prices the bank
    /// again from what is stored, and the bank grows. On Metal, which every
    /// macOS host has: a device that does not open fails the test.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the required figures alone give the costs where the bank moves natively (Metal)")]
    fn a_live_load_reprices_on_newly_stored_costs_and_grows() {
        let backend = seismic::BackendName::Metal;
        let (mut domain, bindings) = fixtures::banked(backend, 4, 2);
        fixtures::record_misses(&mut domain, fixtures::misses_at(3, 100, 4));
        // The pricing: the streamed fixture model planned on this device,
        // its bank and resident plan, its costs kept in an empty cache.
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let reserves = platform::MemoryReserves::standard();
        let selected =
            platform::select_device(&catalog, ExecutionPath::Native, platform::DeviceRequest::Backend(backend), &reserves)
                .unwrap();
        let device = catalog.open(catalog.resolve(selected.info.selector()).unwrap()).unwrap();
        let definition = fixtures::qwen4exp_fixture_definition();
        let manifest = fixtures::streamed_experts_manifest(&definition);
        let resident = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            magnitude_state::KvCodec::Dense,
            fixtures::limits(2, 1, false),
        )
        .unwrap();
        let bank = resident.clone().with_bank(&definition, None, None).unwrap().load().expert_bank().cloned();
        let bank = bank.expect("the streamed fixture plans an expert bank");
        let cache = tempfile::tempdir().unwrap();
        let pricing = ExpertPricing::unmeasured(cache.path().to_owned(), definition, resident, device.clone(), bank);
        let limits = ServiceLimits {
            prefill_tokens: 2,
            decode_tokens: 2,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let mut owner = Owner::new(domain, bindings, limits, Arc::new(Unwoken)).unwrap();
        // Priced as the load is once its warm-up ends: nothing is stored.
        owner.measure_experts(expert_measures(&pricing));
        let mut execution = ExecutionOwner::new(owner, Arc::new(Plain), Some(pricing));
        // Memory stays free well past the growth's recovery wait: an
        // unpriced bank grows nothing.
        let now = observe(&mut execution, 0, 100);
        assert_eq!(slots(&mut execution), (2, None), "an unpriced load grows nothing");
        // The measurement job stores the executor's costs: its probe and the
        // figures no other prices (the launch graph's times and the
        // composition chain, the points of `calls`), the rest skipped.
        let mut measurer =
            seismic::Measurer::new(magnitude_executor::SEARCH_SETTINGS.error_rate, magnitude_executor::MIN_SAMPLE_SECONDS);
        let mut probe = true;
        magnitude_executor::assessment::requests::measure(
            &mut measurer,
            &device,
            &KernelCache::measurement_records(cache.path()),
            &[magnitude_executor::assessment::PointRequest::Costs],
            std::time::Instant::now() + std::time::Duration::from_secs(60),
            &mut |key| match std::mem::take(&mut probe) || key.configuration.contains_key("calls") {
                true => Attempt::Proceed,
                false => Attempt::Skip(Reason::new("not this test's point".to_owned())),
            },
            &mut |_, _| {},
        )
        .unwrap();
        assert!(matches!(execution.command(WorkerCommand::Reprice, now), Ok(WorkerReply::Acknowledged)));
        observe(&mut execution, now, 100);
        let (grown, resize) = slots(&mut execution);
        assert!(grown > 2, "the repriced bank grew: {grown} slots");
        assert!(matches!(resize, Some(BankResize::Grew { .. })), "{resize:?}");
    }
}
