//! The worker's thread-confined execution owner over the family-neutral
//! scheduler. Only the closed scheduler command vocabulary reaches it.

use crate::error::UnloadCause;
use crate::execution::{LoadPricing, load_measures};
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
    /// What prices the load, which prices it again, measuring nothing,
    /// once the service stores new measurement records.
    pricing: LoadPricing,
}

impl<F: ProgramFamily> ExecutionOwner<F> {
    pub fn new(owner: Owner<F>, method: Arc<dyn Method>, pricing: LoadPricing) -> Self {
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
                match &mut self.execution {
                    // The costs the load planned on, found pending again, end it.
                    Execution::Serving(owner) => owner.measure_load(load_measures(&self.pricing)?),
                    // An ending model has nothing left to price.
                    Execution::Draining(..) | Execution::Stopped(_) => {}
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
    use magnitude_executor::{fixtures, CompletionWake, KernelCache, RequestId};
    use magnitude_generation::Plain;
    use magnitude_executor::BankResize;
    use magnitude_scheduler::{publication::PublicationWake, worker::Wakes, ServiceLimits};
    use seismic::measure::Attempt;

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

    /// Store the read line of the volume `cache` lies on as the
    /// measurement job measures it.
    fn store_read_line(device: &seismic::Device, cache: &std::path::Path) {
        let mut measurer =
            seismic::Measurer::new(magnitude_executor::SEARCH_SETTINGS.error_rate, magnitude_executor::MIN_SAMPLE_SECONDS);
        magnitude_executor::assessment::requests::measure(
            &mut measurer,
            device,
            &KernelCache::measurement_records(cache),
            &magnitude_executor::assessment::Outstanding {
                standing: vec![magnitude_executor::assessment::PointRequest::Storage { directory: cache.to_owned() }],
                loads: Vec::new(),
            },
            magnitude_executor::assessment::requests::JobDeadlines::new(std::time::Instant::now(), std::time::Duration::from_secs(60)),
            &mut |_| Attempt::Proceed,
            &mut |_, _| {},
        )
        .unwrap();
    }

    /// A live load priced on its executor's costs, which every load waits
    /// for, before the job stored the read line its streamed decode is
    /// priced along: its bank is cache, and grows into free memory to its
    /// ceiling before its decode is priced (`ElasticBank::targets`). Once
    /// the measurement job stores the read line and the host asks it to
    /// reprice, the execution owner prices the load again from what is
    /// stored, and the bank shrinks to where its decode's misses stop
    /// falling. On Metal, which every macOS host has: a device that does
    /// not open fails the test.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "the required figures alone give the costs where the bank moves natively (Metal)")]
    fn a_live_load_grows_its_bank_unpriced_and_reprices_on_newly_stored_figures() {
        let backend = seismic::BackendName::Metal;
        let (mut domain, bindings) = fixtures::banked(backend, 4, 2);
        // Every step's reads lie 2 deep: a bank of 3 of the 4 rows hits them.
        let decode = fixtures::misses_at(2, 100);
        fixtures::count_decode(&mut domain, decode);
        // The pricing: the streamed fixture model planned on this device,
        // its bank and resident plan, its costs kept in a cache that holds
        // the executor's required figures, as a load's has once it plans.
        let cache = tempfile::tempdir().unwrap();
        let (pricing, device) = LoadPricing::fixture(magnitude_executor::platform::DeviceRequest::Backend(backend), true, Some(cache.path().to_owned()));
        crate::execution::store_fixed_costs(device.topology(), device.info(), cache.path());
        let limits = ServiceLimits {
            prefill_tokens: 2,
            decode_tokens: 2,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        };
        let mut owner = Owner::new(domain, bindings, limits, Arc::new(Unwoken)).unwrap();
        // Priced as the load is once its warm-up ends: the costs, no read line.
        owner.measure_load(load_measures(&pricing).unwrap());
        let mut execution = ExecutionOwner::new(owner, Arc::new(Plain), pricing);
        // Memory stays free well past the growth's recovery wait: the bank
        // grows to its 4 rows, its decode unpriced.
        let now = observe(&mut execution, 0, 100);
        let (grown, resize) = slots(&mut execution);
        assert_eq!(grown, 4, "the unpriced bank grew to its ceiling");
        assert!(matches!(resize, Some(BankResize::Grew { from: 2, .. })), "{resize:?}");
        // The measurement job stores the store's read line.
        store_read_line(&device, cache.path());
        assert!(matches!(execution.command(WorkerCommand::Reprice, now), Ok(WorkerReply::Acknowledged)));
        observe(&mut execution, now, 100);
        let (repriced, resize) = slots(&mut execution);
        assert_eq!(repriced, 3, "the repriced bank shrank to where its misses stop falling");
        assert!(matches!(resize, Some(BankResize::Shrank { to: 3, .. })), "{resize:?}");
    }

    /// A load without a bank chains its prompt as a base load does, priced
    /// from the executor's costs as stored (`execution::store_fixed_costs`), through the
    /// path a load's worker prices it (`load_measures`): its prompt group
    /// takes its chunks' places, the plan's chain.
    #[test]
    fn a_dense_load_chains_its_prompt_priced_from_stored_costs() {
        use magnitude_executor::platform::DeviceRequest;
        let (mut domain, mut bindings) = fixtures::chained(DeviceRequest::Automatic);
        let cache = tempfile::tempdir().unwrap();
        let (pricing, device) = LoadPricing::fixture(DeviceRequest::Automatic, false, Some(cache.path().to_owned()));
        crate::execution::store_fixed_costs(device.topology(), device.info(), cache.path());
        let measures = load_measures(&pricing).unwrap();
        assert!(measures.experts.is_none(), "a load without a bank prices no experts");
        domain.measure_load(measures, &mut bindings).unwrap();
        assert_eq!(fixtures::group_places_opened((domain, bindings)), 3, "the group holds its chunks' places");
    }
}
