//! The service's independent memory guard and resident-worker supervision (memory-reserves spec
//! §3.3, §5.5; integration spec §10).
//!
//! The engine prevents and gracefully answers memory pressure; the service only guards. It samples
//! system-RAM headroom through Seismic in the service process, every 100 ms while a worker is
//! resident and every second otherwise:
//!
//! - the first sample at or below the emergency reserve `E`, or at critical kernel memory
//!   pressure, kills the resident worker;
//! - one continuous second of failed observation while resident fails the worker;
//! - an engine unload for memory pressure and a kill both release the instance as
//!   `memory_pressure` and close load admission until the host has stayed out of distress with
//!   headroom above the planning reserve `P` for five seconds (any failed sample restarts that
//!   wait). Nothing reloads.
//!
//! It also watches the resident worker for exit and unload and publishes its census.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::future::BoxFuture;
use magnitude_engine::error::{RequestError, UnloadCause};
use magnitude_engine::worker::EngineClient;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_executor::platform::{DomainThresholds, HostDistress, host_distress};
use magnitude_service_contracts::models::{ModelInstanceId, ModelReleaseReason};
use seismic::{DeviceCatalog, DeviceTopology};
use tokio::time::Instant;

use super::worker::worker_exit_code;
use super::{ModelOperationFailure, ResidencyNotification, ResidencyNotifier, ResidencyWorker};
use crate::memory_domains::instance_allocation;

const RESIDENT_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const IDLE_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const MONITOR_LOSS_DEADLINE: Duration = Duration::from_secs(1);
const RECOVERY_STABLE_TIME: Duration = Duration::from_secs(5);
const CENSUS_INTERVAL: Duration = Duration::from_secs(1);

/// The failure code of a load refused while memory has not recovered.
pub const MEMORY_PRESSURE_FAILURE_CODE: &str = "memory_pressure";

/// One sample of system RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemorySample {
    /// Seismic's platform estimate bounded by the room left under every process limit.
    pub headroom_bytes: u64,
    /// The host's distress, on a platform that reports displacement.
    pub distress: Option<HostDistress>,
}

pub trait HostMemoryObserver: Send + Sync + 'static {
    fn sample(&self) -> Result<HostMemorySample, String>;
}

/// System RAM observed through the service's own device catalog.
pub struct SeismicHostMemory {
    catalog: Arc<DeviceCatalog>,
}

impl SeismicHostMemory {
    pub fn new(catalog: Arc<DeviceCatalog>) -> Self {
        Self { catalog }
    }
}

impl HostMemoryObserver for SeismicHostMemory {
    fn sample(&self) -> Result<HostMemorySample, String> {
        let status = self
            .catalog
            .host_memory_status()
            .map_err(|error| error.to_string())?;
        Ok(HostMemorySample {
            headroom_bytes: status
                .limits
                .iter()
                .map(|limit| limit.remaining_bytes())
                .fold(status.headroom.bytes, u64::min),
            distress: host_distress(&status),
        })
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
    /// Closed by memory pressure; open again once the host has stayed out of distress with
    /// headroom above `P` since `since`.
    Recovering { above_planning_since: Option<Instant> },
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
    /// residency actor through `notifier`.
    pub fn start(
        observer: Arc<dyn HostMemoryObserver>,
        thresholds: DomainThresholds,
        notifier: ResidencyNotifier,
    ) -> Self {
        let (admission, _) = tokio::sync::watch::channel(true);
        let supervisor = Self {
            inner: Arc::new(Inner {
                observer,
                thresholds,
                notifier,
                state: Mutex::new(State {
                    supervised: None,
                    admission: Admission::Open,
                    observation_failed_since: None,
                }),
                admission,
                supervised_changed: tokio::sync::Notify::new(),
            }),
        };
        tokio::spawn(Arc::clone(&supervisor.inner).run());
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
        topology: Arc<DeviceTopology>,
    ) {
        self.inner.state.lock().expect("supervisor state lock").supervised = Some(Supervised {
            instance_id: instance_id.clone(),
            worker,
            engine: Arc::clone(&engine),
        });
        self.inner.supervised_changed.notify_one();
        tokio::spawn(Arc::clone(&self.inner).publish_census(
            instance_id,
            engine,
            context_window_tokens,
            topology,
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
        let sample = self.observer.sample();
        let mut state = self.lock();
        match sample {
            Ok(HostMemorySample {
                headroom_bytes: headroom,
                distress,
            }) => {
                state.observation_failed_since = None;
                let emergency = if headroom <= self.thresholds.emergency_bytes {
                    Some("system memory headroom reached the emergency reserve")
                } else if distress == Some(HostDistress::KernelCritical) {
                    Some("the kernel reports critical memory pressure")
                } else {
                    None
                };
                if let Some(reason) = emergency
                    && let Some(supervised) = state.supervised.take()
                {
                    tracing::warn!(
                        memory.headroom_bytes = headroom,
                        memory.emergency_bytes = self.thresholds.emergency_bytes,
                        worker.pid = supervised.worker.pid(),
                        "killing the inference worker: {reason}"
                    );
                    supervised
                        .worker
                        .terminate(MEMORY_PRESSURE_FAILURE_CODE, reason);
                    self.notifier.send(ResidencyNotification::ReleaseRequested {
                        instance_id: supervised.instance_id,
                        reason: ModelReleaseReason::MemoryPressure,
                    });
                    state.admission = Admission::Recovering {
                        above_planning_since: None,
                    };
                }
                if let Admission::Recovering {
                    above_planning_since,
                } = &mut state.admission
                {
                    if headroom > self.thresholds.planning_bytes && distress.is_none() {
                        let since = *above_planning_since.get_or_insert(now);
                        if now.duration_since(since) >= RECOVERY_STABLE_TIME {
                            state.admission = Admission::Open;
                        }
                    } else {
                        *above_planning_since = None;
                    }
                }
            }
            Err(error) => {
                let since = *state.observation_failed_since.get_or_insert(now);
                if let Admission::Recovering {
                    above_planning_since,
                } = &mut state.admission
                {
                    *above_planning_since = None;
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
                    state.admission = Admission::Recovering {
                        above_planning_since: None,
                    };
                }
            }
        }
        let open = state.admission == Admission::Open;
        drop(state);
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
            state.admission = Admission::Recovering {
                above_planning_since: None,
            };
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
        topology: Arc<DeviceTopology>,
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
            let allocation = instance_allocation(&topology, context_window_tokens, &observation.census);
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

    /// Scripted headroom (`u64::MAX` fails the observation) and distress.
    struct ScriptedMemory(AtomicU64, Mutex<Option<HostDistress>>);

    impl HostMemoryObserver for ScriptedMemory {
        fn sample(&self) -> Result<HostMemorySample, String> {
            match self.0.load(Ordering::Acquire) {
                u64::MAX => Err("observation failed".to_owned()),
                headroom_bytes => Ok(HostMemorySample {
                    headroom_bytes,
                    distress: *self.1.lock().unwrap(),
                }),
            }
        }
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
                    census: magnitude_engine::census::AllocationCensus { domains: Vec::new() },
                    domains: Vec::new(),
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
        fn terminate(&self, _code: &str, _reason: &str) {
            self.terminated.store(true, Ordering::Release);
        }
        fn shutdown(&self) {}
    }

    const GIB: u64 = 1 << 30;

    struct Fixture {
        inner: Arc<Inner>,
        memory: Arc<ScriptedMemory>,
        notifications: tokio::sync::mpsc::UnboundedReceiver<ResidencyNotification>,
    }

    fn fixture() -> Fixture {
        let memory = Arc::new(ScriptedMemory(AtomicU64::new(10 * GIB), Mutex::new(None)));
        let (sender, notifications) = tokio::sync::mpsc::unbounded_channel();
        let notifier = ResidencyNotifier {
            send: Arc::new(move |notification| {
                let _ = sender.send(notification);
            }),
        };
        let (admission, _) = tokio::sync::watch::channel(true);
        let inner = Arc::new(Inner {
            observer: memory.clone(),
            thresholds: DomainThresholds {
                planning_bytes: 2 * GIB,
                emergency_bytes: GIB,
            },
            notifier,
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

    #[test]
    fn first_sample_at_the_emergency_reserve_kills_and_gates_admission_for_five_seconds() {
        let mut fixture = fixture();
        let worker = supervise(&fixture, Arc::default());
        let start = Instant::now();
        fixture.inner.tick(start);
        assert!(fixture.notifications.try_recv().is_err());

        fixture.memory.0.store(GIB, Ordering::Release);
        fixture.inner.tick(start);
        assert!(worker.terminated.load(Ordering::Acquire));
        assert!(released_for_memory_pressure(fixture.notifications.try_recv().unwrap()));
        assert!(!*fixture.inner.admission.borrow());

        // Above E but not above P: still recovering.
        fixture.memory.0.store(2 * GIB, Ordering::Release);
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
    fn critical_kernel_pressure_kills_and_only_a_host_out_of_distress_recovers() {
        let mut fixture = fixture();
        let worker = supervise(&fixture, Arc::default());
        let start = Instant::now();

        // Swapping and thrashing are the engine's to answer; the guard waits.
        *fixture.memory.1.lock().unwrap() = Some(HostDistress::Swapping);
        fixture.inner.tick(start);
        *fixture.memory.1.lock().unwrap() = Some(HostDistress::Thrashing);
        fixture.inner.tick(start);
        assert!(!worker.terminated.load(Ordering::Acquire));
        assert!(*fixture.inner.admission.borrow());

        // Headroom is far above the emergency reserve throughout.
        *fixture.memory.1.lock().unwrap() = Some(HostDistress::KernelCritical);
        fixture.inner.tick(start);
        assert!(worker.terminated.load(Ordering::Acquire));
        assert!(released_for_memory_pressure(fixture.notifications.try_recv().unwrap()));
        assert!(!*fixture.inner.admission.borrow());

        // Distress of any kind holds admission closed however long it lasts.
        *fixture.memory.1.lock().unwrap() = Some(HostDistress::Swapping);
        fixture.inner.tick(start + Duration::from_secs(10));
        fixture.inner.tick(start + Duration::from_secs(20));
        assert!(!*fixture.inner.admission.borrow());
        *fixture.memory.1.lock().unwrap() = None;
        fixture.inner.tick(start + Duration::from_secs(21));
        assert!(!*fixture.inner.admission.borrow());
        fixture.inner.tick(start + Duration::from_secs(26));
        assert!(*fixture.inner.admission.borrow());
    }

    #[test]
    fn engine_unload_for_memory_pressure_is_a_memory_pressure_release() {
        let mut fixture = fixture();
        let engine = Arc::new(ScriptedEngine::default());
        let worker = supervise(&fixture, engine.clone());
        *engine.0.lock().unwrap() = Some(RequestError::ModelUnloaded {
            cause: UnloadCause::MemoryPressure,
        });
        fixture.inner.tick(Instant::now());
        assert!(!worker.terminated.load(Ordering::Acquire));
        assert!(released_for_memory_pressure(fixture.notifications.try_recv().unwrap()));
        assert!(!*fixture.inner.admission.borrow());
        assert!(fixture.inner.lock().supervised.is_none());
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
