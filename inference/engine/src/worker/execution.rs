//! The worker's thread-confined execution owner over the family-neutral
//! scheduler. Only the closed scheduler command vocabulary reaches it.

use crate::error::UnloadCause;
use crate::execution::{LoadPricing, load_measures};
use magnitude_executor::{MemoryChargeReconciliation, ProgramFamily};
use magnitude_generation::Method;
use magnitude_scheduler::{
    owner::{AdmissionError, Owner, Ran},
    protocol::{AdmitRequest, RoomGrant, WorkerCommand, WorkerReply},
    publication::{ModelUnloadCause, RequestError},
    serving::ServingCensus,
    worker::{Driven, Event, Reread},
};
use std::sync::{Arc, Mutex};
use std::task::Waker;

/// What the owner counts of its memory: the census's inputs.
type Counted = (MemoryChargeReconciliation, ServingCensus);

/// The census the owner counted last and the session has not sent yet:
/// published by the owner as its count changes after a run, taken by the
/// session, which sends it to the host. Pushed on change, never sampled.
#[derive(Default)]
pub(crate) struct CensusWatch {
    state: Mutex<(Option<Counted>, Option<Waker>)>,
}

impl CensusWatch {
    fn publish(&self, counted: Counted) {
        let mut state = self.state.lock().unwrap();
        state.0 = Some(counted);
        if let Some(waker) = state.1.take() {
            waker.wake();
        }
    }

    /// The census counted since the last take; otherwise `waker` is woken
    /// once one is.
    pub(crate) fn take(&self, waker: &Waker) -> Option<Counted> {
        let mut state = self.state.lock().unwrap();
        let counted = state.0.take();
        if counted.is_none() {
            state.1 = Some(waker.clone());
        }
        counted
    }
}

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
    /// Where the census goes as it changes, and what it was last.
    census: Arc<CensusWatch>,
    counted: Option<Counted>,
    /// What the census was last counted at (`Owner::census_key`): nothing
    /// it counts changes while it stands, and readings never move it.
    counted_at: Option<(u64, Option<magnitude_executor::ExpertCover>)>,
}

impl<F: ProgramFamily> ExecutionOwner<F> {
    pub fn new(owner: Owner<F>, method: Arc<dyn Method>, pricing: LoadPricing, census: Arc<CensusWatch>) -> Self {
        Self {
            execution: Execution::Serving(owner),
            ended: None,
            method,
            pricing,
            census,
            counted: None,
            counted_at: None,
        }
    }

    /// Publish `owner`'s census where its count changed since the last,
    /// counted only once what the ledgers hold, or the expert bank's cover,
    /// moved since then (`Owner::census_key`).
    fn count(&mut self, owner: &Owner<F>) -> Result<(), String> {
        let key = owner.census_key();
        if self.counted_at == Some(key) {
            return Ok(());
        }
        self.counted_at = Some(key);
        let counted = (owner.reconcile_memory_charge()?, owner.serving_census()?);
        if self.counted.as_ref() != Some(&counted) {
            self.counted = Some(counted.clone());
            self.census.publish(counted);
        }
        Ok(())
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
            WorkerCommand::AttachMeasurementProcess { process } => {
                if let Execution::Serving(owner) | Execution::Draining(owner, _) = &mut self.execution {
                    owner.attach_measurement_process(process);
                }
                Ok(WorkerReply::Acknowledged)
            }
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
            // The owner reads memory when it next runs: the reader marks the
            // pressure event it was woken for.
            Event::MemoryPressure => {}
            // The worker's timed reading is due: the owner reads memory when
            // it next runs.
            Event::Reread => match &mut self.execution {
                Execution::Serving(owner) | Execution::Draining(owner, _) => owner.reread(),
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

    fn run(&mut self, now: u64) -> Result<Reread, UnloadCause> {
        loop {
            match self.take() {
                Execution::Serving(owner) => match owner.run(now) {
                    Ok((owner, Ran { unload: true, .. })) => {
                        self.execution = Execution::Draining(owner, UnloadCause::MemoryPressure)
                    }
                    Ok((owner, Ran { reread, .. })) => match self.count(&owner) {
                        Ok(()) => {
                            self.execution = Execution::Serving(owner);
                            return Ok(reread);
                        }
                        Err(reason) => self.execution = Execution::Stopped(UnloadCause::Internal { reason }),
                    },
                    Err(error) => self.execution = Execution::Stopped(UnloadCause::from(&error)),
                },
                Execution::Draining(owner, cause) => match owner.run(now) {
                    Ok((owner, Ran { in_flight: false, .. })) => {
                        self.ended = Some(owner);
                        self.execution = Execution::Stopped(cause)
                    }
                    // A draining owner answers no shortage: its flight's end
                    // is all it waits for.
                    Ok((owner, Ran { .. })) => {
                        self.execution = Execution::Draining(owner, cause);
                        return Ok(Reread::Settled);
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

        fn pressure(&self) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
    }

    /// The bank's slots and its last resize, as an observation reads them.
    fn slots<F: ProgramFamily>(execution: &mut ExecutionOwner<F>) -> (u64, Option<BankResize>) {
        let Ok(WorkerReply::Observed { serving, .. }) = execution.command(WorkerCommand::Observe, 0) else {
            panic!("an observation of a serving owner")
        };
        let experts = serving.experts.expect("a banked load");
        (experts.slots, experts.last_resize)
    }

    /// Run the owner at `events` events (each a flight boundary or an
    /// arrival, which reads memory), from `now`; the clock after.
    fn observe<F: ProgramFamily>(execution: &mut ExecutionOwner<F>, mut now: u64, events: usize) -> u64 {
        for _ in 0..events {
            execution.run(now).unwrap();
            now += 1;
        }
        now
    }

    /// Store the read line of the volume `cache` lies on as the
    /// measurement job measures it: on a device of `backend`'s of its own,
    /// opened outside any ledger as the job's process opens its own, and
    /// charged to its tallies.
    fn store_read_line(backend: seismic::BackendName, cache: &std::path::Path) {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let (device, _) = seismic::testing::open_backend_tallied(&catalog, backend, Default::default()).unwrap();
        let mut measurer =
            seismic::Measurer::new(magnitude_executor::SEARCH_SETTINGS.error_rate, magnitude_executor::MIN_SAMPLE_SECONDS);
        magnitude_executor::assessment::requests::measure(
            &mut measurer,
            &seismic::Charges::tally(&device),
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
        let mut execution = ExecutionOwner::new(owner, Arc::new(Plain), pricing, Arc::default());
        // Memory stays free across the events: the bank grows to its 4
        // rows, its decode unpriced.
        let now = observe(&mut execution, 0, 100);
        let (grown, resize) = slots(&mut execution);
        assert_eq!(grown, 4, "the unpriced bank grew to its ceiling");
        assert!(matches!(resize, Some(BankResize::Grew { from: 2, .. })), "{resize:?}");
        // The measurement job stores the store's read line.
        store_read_line(backend, cache.path());
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
