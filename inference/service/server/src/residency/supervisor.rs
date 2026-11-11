//! The service's independent memory guard and resident-worker supervision (memory-reserves spec
//! §3.3, §5.5; integration spec §10).
//!
//! The engine prevents and gracefully answers memory pressure; the service only guards. It samples
//! system-RAM headroom through Seismic in the service process, every 100 ms while a worker is
//! resident and every second otherwise:
//!
//! - one continuous second of failed observation while resident fails the worker;
//! - an engine unload for memory pressure releases the instance as `memory_pressure` and closes
//!   load admission until headroom has stayed above the planning reserve `P` for five seconds
//!   (any failed sample restarts that wait). Nothing reloads.
//!
//! Unloading for memory is the engine's decision alone; the service reads no host emergency or
//! distress.
//!
//! It also watches the resident worker for exit and unload and publishes its census.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::future::BoxFuture;
use magnitude_engine::error::{RequestError, UnloadCause};
use magnitude_engine::worker::EngineClient;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_executor::platform::{DomainThresholds, MemoryReserves, MotionMeter, Recovery, StoreMotion};
use magnitude_service_contracts::models::{ModelInstanceId, ModelReleaseReason};
use seismic::Devices;
use tokio::time::Instant;

use super::worker::worker_exit_code;
use super::{ModelOperationFailure, ResidencyNotification, ResidencyNotifier, ResidencyWorker};
use crate::memory_domains::instance_allocation;

const RESIDENT_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const IDLE_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const MONITOR_LOSS_DEADLINE: Duration = Duration::from_secs(1);
const CENSUS_INTERVAL: Duration = Duration::from_secs(1);

pub trait HostMemoryObserver: Send + Sync + 'static {
    /// Host RAM's headroom: what the OS gives a load, within every process limit
    /// (`HostStatus::allocatable_bytes` of `Reclaim::CleanFileCache`): what load admission
    /// reopens on.
    fn headroom(&self) -> Result<u64, String>;
    /// Fold in a resident load's reading of host RAM at `at`: `available` bytes beside the
    /// `charged` bytes its Seismic charge holds, both read by the worker in one observation
    /// ([`load_reading`]), so the load's own imports are never taken for another program's.
    fn observed_load(&self, at: std::time::Instant, available: u64, charged: u64);
    /// How the rest of the host moves memory, as the observer measured it from the loads'
    /// readings: what a load's reserves hold their margin at (`MemoryReserves::measured`).
    fn motion(&self) -> StoreMotion;
}

/// System RAM observed through the service's own device catalog.
pub struct SeismicHostMemory {
    catalog: Arc<Devices>,
    /// Every resident load's reading of host RAM beside its charge: the host's motion, which the
    /// next load starts from.
    meter: std::sync::Mutex<MotionMeter>,
}

impl SeismicHostMemory {
    /// Host RAM of `catalog`'s inventory, its motion measured from `reserves`' own (the
    /// machine's figure, `MemoryReserves::of_machine`).
    pub fn new(catalog: Arc<Devices>, reserves: MemoryReserves) -> Self {
        Self { catalog, meter: std::sync::Mutex::new(MotionMeter::new(reserves.host_motion())) }
    }
}

impl HostMemoryObserver for SeismicHostMemory {
    /// One host reading's headroom ([`seismic::Inventory::host_status`]): nothing of volumes,
    /// grants or opened executors is read.
    fn headroom(&self) -> Result<u64, String> {
        let reading = self.catalog.inventory().host_status().map_err(|error| error.to_string())?;
        // What admits a load: a load's own action takes the clean file cache once.
        Ok(reading.allocatable_bytes(seismic::Reclaim::CleanFileCache))
    }

    fn observed_load(&self, at: std::time::Instant, available: u64, charged: u64) {
        self.meter.lock().expect("motion meter lock poisoned").read(at, available, Some(charged));
    }

    fn motion(&self) -> StoreMotion {
        self.meter.lock().expect("motion meter lock poisoned").motion()
    }
}

/// What the supervisor needs of a resident engine.
pub trait SupervisedEngine: Send + Sync + 'static {
    /// `Ok` while the worker serves; otherwise why it stopped.
    fn check(&self) -> Result<(), RequestError>;
    /// The engine heap's current standing and a fresh reading of its memory domains.
    fn observe(&self) -> BoxFuture<'_, Result<MemoryObservation, RequestError>>;
}

impl SupervisedEngine for EngineClient {
    fn check(&self) -> Result<(), RequestError> {
        EngineClient::check(self)
    }

    fn observe(&self) -> BoxFuture<'_, Result<MemoryObservation, RequestError>> {
        Box::pin(EngineClient::observe(self))
    }
}

struct Supervised {
    instance_id: ModelInstanceId,
    worker: Arc<dyn ResidencyWorker>,
    engine: Arc<dyn SupervisedEngine>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    Open,
    /// Closed by memory pressure; open again once memory has recovered: headroom has stayed
    /// above `P` ([`Recovery`]).
    Recovering(Recovery<Instant>),
}

struct State {
    supervised: Option<Supervised>,
    admission: Admission,
    observation_failed_since: Option<Instant>,
}

struct Inner {
    observer: Arc<dyn HostMemoryObserver>,
    thresholds: DomainThresholds,
    notifier: ResidencyNotifier,
    /// The device's jobs, held off while memory recovers.
    device: crate::device_lock::DeviceLock,
    state: Mutex<State>,
    admission: tokio::sync::watch::Sender<bool>,
    supervised_changed: tokio::sync::Notify,
}

/// The memory guard and resident-worker supervisor. Cloning shares one guard.
#[derive(Clone)]
pub struct MemorySupervisor {
    inner: Arc<Inner>,
}

impl MemorySupervisor {
    /// Start guarding with `thresholds`, the system-RAM domain's reserves, reporting to the
    /// residency actor through `notifier`, and holding `device`'s jobs off under pressure.
    pub fn start(
        observer: Arc<dyn HostMemoryObserver>,
        thresholds: DomainThresholds,
        notifier: ResidencyNotifier,
        device: crate::device_lock::DeviceLock,
    ) -> Self {
        let (admission, _) = tokio::sync::watch::channel(true);
        let supervisor = Self {
            inner: Arc::new(Inner {
                observer,
                thresholds,
                notifier,
                device,
                state: Mutex::new(State {
                    supervised: None,
                    admission: Admission::Open,
                    observation_failed_since: None,
                }),
                admission,
                supervised_changed: tokio::sync::Notify::new(),
            }),
        };
        magnitude_service_utils::owner::spawn("the memory supervisor", Arc::clone(&supervisor.inner).run());
        supervisor
    }

    /// Wait until loads may be admitted: immediately, unless memory pressure released the last
    /// instance and headroom has not yet stayed above `P` for five seconds.
    pub async fn admission(&self) {
        let mut admission = self.inner.admission.subscribe();
        // The sender lives in `inner`, which `self` holds.
        let _ = admission.wait_for(|open| *open).await;
    }

    pub fn admission_open(&self) -> bool {
        *self.inner.admission.borrow()
    }

    /// A fresh memory observation from the resident worker; `None` when no worker is resident or
    /// it stopped serving before answering.
    pub async fn observe_resident(&self) -> Option<MemoryObservation> {
        let engine = Arc::clone(&self.inner.lock().supervised.as_ref()?.engine);
        engine.observe().await.ok()
    }

    /// Supervise a ready worker until it exits or is released.
    pub fn supervise(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        engine: Arc<dyn SupervisedEngine>,
        context_window_tokens: u32,
    ) {
        self.inner.state.lock().expect("supervisor state lock").supervised = Some(Supervised {
            instance_id: instance_id.clone(),
            worker,
            engine: Arc::clone(&engine),
        });
        self.inner.supervised_changed.notify_one();
        magnitude_service_utils::owner::spawn("a resident's memory census", Arc::clone(&self.inner).publish_census(
            instance_id,
            engine,
            context_window_tokens,
        ));
    }
}

impl Inner {
    async fn run(self: Arc<Self>) {
        loop {
            let interval = if self.lock().supervised.is_some() {
                RESIDENT_SAMPLE_INTERVAL
            } else {
                IDLE_SAMPLE_INTERVAL
            };
            tokio::select! {
                () = tokio::time::sleep(interval) => {}
                () = self.supervised_changed.notified() => {}
            }
            self.tick(Instant::now());
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("supervisor state lock")
    }

    /// One supervision step at `now`: worker outcome first, then one host sample.
    fn tick(&self, now: Instant) {
        self.check_worker();
        let sample = self.observer.headroom();
        let mut state = self.lock();
        match sample {
            Ok(headroom) => {
                state.observation_failed_since = None;
                if let Admission::Recovering(recovery) = &mut state.admission {
                    let free = headroom > self.thresholds.planning_bytes;
                    if recovery.observe(free, now, |since, now| now.duration_since(since)) {
                        state.admission = Admission::Open;
                    }
                }
            }
            Err(error) => {
                let since = *state.observation_failed_since.get_or_insert(now);
                // A failed reading is not a free one.
                if let Admission::Recovering(recovery) = &mut state.admission {
                    recovery.observe(false, now, |since, now| now.duration_since(since));
                }
                if now.duration_since(since) >= MONITOR_LOSS_DEADLINE
                    && let Some(supervised) = state.supervised.take()
                {
                    self.notifier.send(ResidencyNotification::WorkerFailed {
                        instance_id: supervised.instance_id,
                        failure: ModelOperationFailure::new(
                            "memory_monitor_unavailable",
                            format!("system memory supervision unavailable: {error}"),
                            true,
                        ),
                    });
                    state.admission = Admission::Recovering(Recovery::new());
                }
            }
        }
        let open = state.admission == Admission::Open;
        drop(state);
        // Jobs start again only once memory has recovered.
        self.device.hold_off_jobs(!open);
        self.admission.send_if_modified(|current| {
            let changed = *current != open;
            *current = open;
            changed
        });
    }

    /// Release the resident instance when its worker unloaded or exited.
    fn check_worker(&self) {
        let mut state = self.lock();
        let Some(supervised) = &state.supervised else {
            return;
        };
        let unloaded = supervised.engine.check().err();
        let exited = match supervised.worker.try_wait() {
            Ok(status) => status,
            Err(error) => {
                let supervised = state.supervised.take().expect("supervised worker");
                drop(state);
                self.notifier.send(ResidencyNotification::WorkerFailed {
                    instance_id: supervised.instance_id,
                    failure: ModelOperationFailure::new(
                        "worker_monitor_failed",
                        format!("failed to observe inference worker: {error}"),
                        true,
                    ),
                });
                return;
            }
        };
        let memory_pressure = matches!(
            unloaded,
            Some(RequestError::ModelUnloaded {
                cause: UnloadCause::MemoryPressure
            })
        ) || exited.and_then(|status| status.code())
            == Some(worker_exit_code(&magnitude_engine::worker::WorkerExit::Unloaded(
                UnloadCause::MemoryPressure,
            )));
        if memory_pressure {
            let supervised = state.supervised.take().expect("supervised worker");
            state.admission = Admission::Recovering(Recovery::new());
            drop(state);
            tracing::warn!(
                worker.pid = supervised.worker.pid(),
                "the engine unloaded its model under memory pressure"
            );
            self.notifier.send(ResidencyNotification::ReleaseRequested {
                instance_id: supervised.instance_id,
                reason: ModelReleaseReason::MemoryPressure,
            });
            return;
        }
        let failure = match (unloaded, exited) {
            (None, None) => return,
            (Some(error), _) => ModelOperationFailure::new(
                request_failure_code(&error),
                format!("inference worker stopped serving: {error}"),
                true,
            ),
            (None, Some(status)) => ModelOperationFailure::new(
                "worker_exited",
                format!("inference worker exited unexpectedly: {status}"),
                true,
            ),
        };
        let supervised = state.supervised.take().expect("supervised worker");
        drop(state);
        self.notifier.send(ResidencyNotification::WorkerFailed {
            instance_id: supervised.instance_id,
            failure,
        });
    }

    /// Publish the resident engine's census while it stays the supervised instance.
    async fn publish_census(
        self: Arc<Self>,
        instance_id: ModelInstanceId,
        engine: Arc<dyn SupervisedEngine>,
        context_window_tokens: u32,
    ) {
        let mut published = None;
        loop {
            tokio::time::sleep(CENSUS_INTERVAL).await;
            let current = self
                .lock()
                .supervised
                .as_ref()
                .is_some_and(|supervised| supervised.instance_id == instance_id);
            if !current {
                return;
            }
            let observation = match engine.observe().await {
                Ok(observation) => observation,
                // One failed reading; the next interval observes again.
                Err(RequestError::MemoryObservationUnavailable { .. }) => continue,
                // A closed engine is the worker check's to report.
                Err(_) => return,
            };
            if let Some((available, charged)) = load_reading(&observation) {
                self.observer.observed_load(std::time::Instant::now(), available, charged);
            }
            let allocation = match instance_allocation(context_window_tokens, &observation.census) {
                Ok(allocation) => allocation,
                // A census outside the contract's bounds is the engine's failure.
                Err(error) => {
                    let mut state = self.lock();
                    if let Some(supervised) = state.supervised.take_if(|supervised| supervised.instance_id == instance_id) {
                        drop(state);
                        self.notifier.send(ResidencyNotification::WorkerFailed {
                            instance_id: supervised.instance_id,
                            failure: ModelOperationFailure::new("worker_census_invalid", error.to_string(), false),
                        });
                    }
                    return;
                }
            };
            if published.as_ref() != Some(&allocation) {
                published = Some(allocation.clone());
                self.notifier.send(ResidencyNotification::AllocationObserved {
                    instance_id: instance_id.clone(),
                    allocation,
                });
            }
        }
    }
}

/// A worker's reading of host RAM in one observation: its available bytes and what the load's
/// census counts charged there, every class; none where the observation read no host RAM.
fn load_reading(observation: &MemoryObservation) -> Option<(u64, u64)> {
    let available = observation
        .stores
        .iter()
        .find(|store| store.store == seismic::StoreIdentity::HostRam)?
        .headroom_bytes;
    let charged = observation
        .census
        .stores
        .iter()
        .filter(|store| store.store == seismic::StoreIdentity::HostRam)
        .map(|store| store.model_bytes + store.context_bytes + store.compute_bytes + store.auxiliary_bytes)
        .sum();
    Some((available, charged))
}

/// The residency failure code of an engine that stopped serving.
fn request_failure_code(error: &RequestError) -> &'static str {
    match error {
        RequestError::ModelUnloaded {
            cause: UnloadCause::DeviceLost { .. },
        }
        | RequestError::DeviceLost { .. } => "device_lost",
        RequestError::ModelUnloaded { .. } => "model_unloaded",
        RequestError::WorkerLost { .. } => "worker_exited",
        _ => "worker_failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Scripted headroom (`u64::MAX` fails the observation).
    struct ScriptedMemory(AtomicU64);

    impl HostMemoryObserver for ScriptedMemory {
        fn headroom(&self) -> Result<u64, String> {
            match self.0.load(Ordering::Acquire) {
                u64::MAX => Err("observation failed".to_owned()),
                headroom_bytes => Ok(headroom_bytes),
            }
        }

        fn observed_load(&self, _at: std::time::Instant, _available: u64, _charged: u64) {}

        fn motion(&self) -> StoreMotion {
            StoreMotion::new(0, 1e9)
        }
    }

    /// The meter reads a load's import from the one observation that holds both its fall in
    /// host RAM and its charge: an import between two censuses moves both together at the next,
    /// and no host sample between them reaches the meter, so it measures no take.
    #[test]
    fn a_loads_import_between_censuses_measures_no_take() {
        const GB: u64 = 1_000_000_000;
        let observation = |available: u64, charged: u64| MemoryObservation {
            census: magnitude_engine::census::AllocationCensus {
                stores: vec![magnitude_executor::StoreAllocation {
                    store: seismic::StoreIdentity::HostRam,
                    model_bytes: charged,
                    context_bytes: 0,
                    compute_bytes: 0,
                    auxiliary_bytes: 0,
                }],
                serving: magnitude_engine::census::ServingCensus {
                    group_places: std::num::NonZeroUsize::MIN,
                    group_capacity: std::num::NonZeroUsize::MIN,
                    experts: None,
                },
            },
            stores: vec![magnitude_engine::worker::protocol::StoreHeadroom {
                store: seismic::StoreIdentity::HostRam,
                headroom_bytes: available,
            }],
        };
        let prior = StoreMotion::new(0, GB as f64);
        let mut meter = MotionMeter::new(prior);
        let start = std::time::Instant::now();
        for (second, (available, charged)) in [(20 * GB, 0), (15 * GB, 5 * GB), (12 * GB, 8 * GB)].into_iter().enumerate() {
            let (available, charged) = load_reading(&observation(available, charged)).expect("host RAM read");
            meter.read(start + CENSUS_INTERVAL * second as u32, available, Some(charged));
        }
        assert_eq!(meter.motion(), prior);
    }

    #[derive(Default)]
    struct ScriptedEngine(Mutex<Option<RequestError>>);

    impl SupervisedEngine for ScriptedEngine {
        fn check(&self) -> Result<(), RequestError> {
            self.0.lock().unwrap().clone().map_or(Ok(()), Err)
        }

        fn observe(&self) -> BoxFuture<'_, Result<MemoryObservation, RequestError>> {
            Box::pin(async {
                Ok(MemoryObservation {
                    census: magnitude_engine::census::AllocationCensus {
                        stores: Vec::new(),
                        serving: magnitude_engine::census::ServingCensus {
                            group_places: std::num::NonZeroUsize::MIN,
                            group_capacity: std::num::NonZeroUsize::MIN,
                            experts: None,
                        },
                    },
                    stores: Vec::new(),
                })
            })
        }
    }

    #[derive(Default)]
    struct ScriptedWorker {
        terminated: std::sync::atomic::AtomicBool,
    }

    impl ResidencyWorker for ScriptedWorker {
        fn pid(&self) -> Option<u32> {
            None
        }
        fn try_wait(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
            #[cfg(unix)]
            use std::os::unix::process::ExitStatusExt;
            #[cfg(windows)]
            use std::os::windows::process::ExitStatusExt;
            Ok(self
                .terminated
                .load(Ordering::Acquire)
                .then(|| std::process::ExitStatus::from_raw(9)))
        }
        fn terminate(&self, _code: &str, _reason: &str) -> std::io::Result<()> {
            self.terminated.store(true, Ordering::Release);
            Ok(())
        }
        fn shutdown(&self) {}
        fn answered_shutdown(&self) -> bool {
            false
        }
    }

    const GIB: u64 = 1 << 30;

    struct Fixture {
        inner: Arc<Inner>,
        memory: Arc<ScriptedMemory>,
        notifications: tokio::sync::mpsc::UnboundedReceiver<ResidencyNotification>,
    }

    fn fixture() -> Fixture {
        let memory = Arc::new(ScriptedMemory(AtomicU64::new(10 * GIB)));
        let (sender, notifications) = tokio::sync::mpsc::unbounded_channel();
        let notifier = ResidencyNotifier {
            send: Arc::new(move |notification| {
                let _ = sender.send(notification);
            }),
        };
        let (admission, _) = tokio::sync::watch::channel(true);
        let inner = Arc::new(Inner {
            observer: memory.clone(),
            thresholds: DomainThresholds { planning_bytes: 2 * GIB },
            notifier,
            device: crate::device_lock::DeviceLock::default(),
            state: Mutex::new(State {
                supervised: None,
                admission: Admission::Open,
                observation_failed_since: None,
            }),
            admission,
            supervised_changed: tokio::sync::Notify::new(),
        });
        Fixture {
            inner,
            memory,
            notifications,
        }
    }

    fn supervise(fixture: &Fixture, engine: Arc<ScriptedEngine>) -> Arc<ScriptedWorker> {
        let worker = Arc::new(ScriptedWorker::default());
        fixture.inner.lock().supervised = Some(Supervised {
            instance_id: ModelInstanceId("instance".to_owned()),
            worker: worker.clone(),
            engine,
        });
        worker
    }

    fn released_for_memory_pressure(notification: ResidencyNotification) -> bool {
        matches!(
            notification,
            ResidencyNotification::ReleaseRequested {
                reason: ModelReleaseReason::MemoryPressure,
                ..
            }
        )
    }

    /// An engine unload for memory pressure releases the instance as such and closes admission
    /// until headroom has stayed above `P` for five seconds; a failed sample restarts the wait,
    /// and nothing reloads.
    #[test]
    fn engine_unload_for_memory_pressure_gates_admission_until_headroom_stays_above_p() {
        let mut fixture = fixture();
        let engine = Arc::new(ScriptedEngine::default());
        let worker = supervise(&fixture, engine.clone());
        *engine.0.lock().unwrap() = Some(RequestError::ModelUnloaded {
            cause: UnloadCause::MemoryPressure,
        });
        fixture.memory.0.store(2 * GIB, Ordering::Release);
        let start = Instant::now();
        fixture.inner.tick(start);
        assert!(!worker.terminated.load(Ordering::Acquire));
        assert!(released_for_memory_pressure(fixture.notifications.try_recv().unwrap()));
        assert!(!*fixture.inner.admission.borrow());
        assert!(fixture.inner.lock().supervised.is_none());

        // Not above P: still recovering.
        fixture.inner.tick(start + Duration::from_secs(10));
        assert!(!*fixture.inner.admission.borrow());
        // Above P, then a failed sample restarts the five seconds.
        fixture.memory.0.store(3 * GIB, Ordering::Release);
        fixture.inner.tick(start + Duration::from_secs(11));
        fixture.memory.0.store(u64::MAX, Ordering::Release);
        fixture.inner.tick(start + Duration::from_secs(12));
        fixture.memory.0.store(3 * GIB, Ordering::Release);
        fixture.inner.tick(start + Duration::from_secs(16));
        assert!(!*fixture.inner.admission.borrow());
        fixture.inner.tick(start + Duration::from_secs(20));
        assert!(!*fixture.inner.admission.borrow());
        fixture.inner.tick(start + Duration::from_secs(21));
        assert!(*fixture.inner.admission.borrow());
        assert!(fixture.notifications.try_recv().is_err(), "nothing reloads");
    }

    #[test]
    fn worker_loss_and_monitor_loss_fail_the_instance() {
        let mut fixture = fixture();
        let engine = Arc::new(ScriptedEngine::default());
        supervise(&fixture, engine.clone());
        *engine.0.lock().unwrap() = Some(RequestError::WorkerLost {
            reason: "gone".to_owned(),
        });
        fixture.inner.tick(Instant::now());
        assert!(matches!(
            fixture.notifications.try_recv().unwrap(),
            ResidencyNotification::WorkerFailed { failure, .. } if failure.code() == "worker_exited"
        ));
        assert!(*fixture.inner.admission.borrow());

        supervise(&fixture, Arc::default());
        fixture.memory.0.store(u64::MAX, Ordering::Release);
        let start = Instant::now();
        fixture.inner.tick(start);
        assert!(fixture.notifications.try_recv().is_err());
        fixture.inner.tick(start + MONITOR_LOSS_DEADLINE);
        assert!(matches!(
            fixture.notifications.try_recv().unwrap(),
            ResidencyNotification::WorkerFailed { failure, .. }
                if failure.code() == "memory_monitor_unavailable"
        ));
    }
}
