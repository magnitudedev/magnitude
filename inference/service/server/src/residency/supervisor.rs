//! Resident-worker supervision (integration spec §10).
//!
//! The engine owns its memory: it reads the machine on events, answers a real shortage in order,
//! and unloads only on one that persists. The service holds no memory figure of its own and
//! samples nothing. It waits on the resident engine's connection ending, which is the one event
//! of both the worker's unload and its exit (the worker's stream ends only as its process does):
//!
//! - an engine unload for memory pressure releases the instance as `memory_pressure`. Nothing
//!   reloads, and the next load's plan decides in the room the host then has;
//! - any other end fails the instance.
//!
//! It also publishes the resident engine's census as the engine pushes each change, which the
//! instance's status shows.

use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use magnitude_engine::census::AllocationCensus;
use magnitude_engine::error::{RequestError, UnloadCause};
use magnitude_engine::worker::EngineClient;
use magnitude_engine::worker::protocol::MemoryObservation;
use magnitude_service_contracts::models::{ModelInstanceId, ModelReleaseReason};

use super::{ModelOperationFailure, ResidencyNotification, ResidencyNotifier, ResidencyWorker};
use crate::memory_domains::instance_allocation;

/// What the supervisor needs of a resident engine.
pub trait SupervisedEngine: Send + Sync + 'static {
    /// Why the engine stopped serving, once it stops: its own unload, or its worker's exit or
    /// loss.
    fn closed(&self) -> BoxFuture<'_, RequestError>;
    /// The engine heap's current standing and a fresh reading of its memory domains.
    fn observe(&self) -> BoxFuture<'_, Result<MemoryObservation, RequestError>>;
    /// The census once the engine pushed more than `seen` changes of it, with how many it
    /// pushed ([`EngineClient::census`]).
    fn census(&self, seen: u64) -> BoxFuture<'_, Result<(u64, AllocationCensus), RequestError>>;
}

impl SupervisedEngine for EngineClient {
    fn closed(&self) -> BoxFuture<'_, RequestError> {
        Box::pin(EngineClient::closed(self))
    }

    fn observe(&self) -> BoxFuture<'_, Result<MemoryObservation, RequestError>> {
        Box::pin(EngineClient::observe(self))
    }

    fn census(&self, seen: u64) -> BoxFuture<'_, Result<(u64, AllocationCensus), RequestError>> {
        Box::pin(EngineClient::census(self, seen))
    }
}

struct Supervised {
    instance_id: ModelInstanceId,
    worker: Arc<dyn ResidencyWorker>,
    engine: Arc<dyn SupervisedEngine>,
}

struct Inner {
    notifier: ResidencyNotifier,
    supervised: Mutex<Option<Supervised>>,
}

/// The resident-worker supervisor. Cloning shares one supervisor.
#[derive(Clone)]
pub struct MemorySupervisor {
    inner: Arc<Inner>,
}

impl MemorySupervisor {
    /// Supervise resident workers, reporting to the residency actor through `notifier`.
    pub fn new(notifier: ResidencyNotifier) -> Self {
        Self {
            inner: Arc::new(Inner {
                notifier,
                supervised: Mutex::new(None),
            }),
        }
    }

    /// A fresh memory observation from the resident worker; `None` when no worker is resident or
    /// it stopped serving before answering.
    pub async fn observe_resident(&self) -> Option<MemoryObservation> {
        let engine = Arc::clone(&self.inner.lock().as_ref()?.engine);
        engine.observe().await.ok()
    }

    /// Supervise a ready worker until its engine stops serving or another is supervised.
    pub fn supervise(
        &self,
        instance_id: ModelInstanceId,
        worker: Arc<dyn ResidencyWorker>,
        engine: Arc<dyn SupervisedEngine>,
        context_window_tokens: u32,
    ) {
        *self.inner.lock() = Some(Supervised {
            instance_id: instance_id.clone(),
            worker,
            engine: Arc::clone(&engine),
        });
        magnitude_service_utils::owner::spawn(
            "a resident's end",
            Arc::clone(&self.inner).watch(instance_id.clone(), Arc::clone(&engine)),
        );
        magnitude_service_utils::owner::spawn(
            "a resident's memory census",
            Arc::clone(&self.inner).publish_census(instance_id, engine, context_window_tokens),
        );
    }
}

impl Inner {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Supervised>> {
        self.supervised.lock().expect("supervisor state lock")
    }

    /// Wait for the engine of `instance_id` to stop serving, and answer its end.
    async fn watch(self: Arc<Self>, instance_id: ModelInstanceId, engine: Arc<dyn SupervisedEngine>) {
        let error = engine.closed().await;
        self.ended(&instance_id, error);
    }

    /// The engine of `instance_id` stopped serving with `error`: release the instance as one
    /// unloaded under memory pressure, or fail it. An instance no longer supervised is past it.
    fn ended(&self, instance_id: &ModelInstanceId, error: RequestError) {
        let Some(supervised) = self.lock().take_if(|supervised| supervised.instance_id == *instance_id) else {
            return;
        };
        if let RequestError::ModelUnloaded {
            cause: UnloadCause::MemoryPressure,
        } = error
        {
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
        self.notifier.send(ResidencyNotification::WorkerFailed {
            instance_id: supervised.instance_id,
            failure: ModelOperationFailure::new(
                request_failure_code(&error),
                format!("inference worker stopped serving: {error}"),
                true,
            ),
        });
    }

    /// Publish the resident engine's census as the engine pushes each change, while it stays the
    /// supervised instance.
    async fn publish_census(
        self: Arc<Self>,
        instance_id: ModelInstanceId,
        engine: Arc<dyn SupervisedEngine>,
        context_window_tokens: u32,
    ) {
        let (mut seen, mut published) = (0, None);
        loop {
            // A closed engine is its watch's to report.
            let Ok((pushed, census)) = engine.census(seen).await else { return };
            seen = pushed;
            let current = self
                .lock()
                .as_ref()
                .is_some_and(|supervised| supervised.instance_id == instance_id);
            if !current {
                return;
            }
            let allocation = match instance_allocation(context_window_tokens, &census) {
                Ok(allocation) => allocation,
                // A census outside the contract's bounds is the engine's failure.
                Err(error) => {
                    if let Some(supervised) = self.lock().take_if(|supervised| supervised.instance_id == instance_id) {
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
    use std::sync::atomic::{AtomicBool, Ordering};

    /// An engine that never ends on its own: the tests hand its end to the supervisor.
    struct ScriptedEngine;

    impl SupervisedEngine for ScriptedEngine {
        fn closed(&self) -> BoxFuture<'_, RequestError> {
            Box::pin(std::future::pending())
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

        fn census(&self, _: u64) -> BoxFuture<'_, Result<(u64, AllocationCensus), RequestError>> {
            Box::pin(std::future::pending())
        }
    }

    #[derive(Default)]
    struct ScriptedWorker {
        terminated: AtomicBool,
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

    struct Fixture {
        inner: Arc<Inner>,
        notifications: tokio::sync::mpsc::UnboundedReceiver<ResidencyNotification>,
    }

    fn fixture() -> Fixture {
        let (sender, notifications) = tokio::sync::mpsc::unbounded_channel();
        let notifier = ResidencyNotifier {
            send: Arc::new(move |notification| {
                let _ = sender.send(notification);
            }),
        };
        Fixture {
            inner: MemorySupervisor::new(notifier).inner,
            notifications,
        }
    }

    fn instance() -> ModelInstanceId {
        ModelInstanceId("instance".to_owned())
    }

    fn supervise(fixture: &Fixture) -> Arc<ScriptedWorker> {
        let worker = Arc::new(ScriptedWorker::default());
        *fixture.inner.lock() = Some(Supervised {
            instance_id: instance(),
            worker: worker.clone(),
            engine: Arc::new(ScriptedEngine),
        });
        worker
    }

    /// An engine unload for memory pressure releases the instance as such, without terminating
    /// its worker; nothing reloads, and the instance is supervised no more.
    #[test]
    fn engine_unload_for_memory_pressure_releases_the_instance() {
        let mut fixture = fixture();
        let worker = supervise(&fixture);
        fixture.inner.ended(
            &instance(),
            RequestError::ModelUnloaded {
                cause: UnloadCause::MemoryPressure,
            },
        );
        assert!(!worker.terminated.load(Ordering::Acquire));
        assert!(matches!(
            fixture.notifications.try_recv().unwrap(),
            ResidencyNotification::ReleaseRequested {
                reason: ModelReleaseReason::MemoryPressure,
                ..
            }
        ));
        assert!(fixture.inner.lock().is_none());
        assert!(fixture.notifications.try_recv().is_err(), "nothing reloads");
    }

    /// A worker lost fails the instance; the end of an instance no longer supervised reports
    /// nothing.
    #[test]
    fn worker_loss_fails_the_instance_once() {
        let mut fixture = fixture();
        supervise(&fixture);
        let lost = || RequestError::WorkerLost {
            reason: "gone".to_owned(),
        };
        fixture.inner.ended(&instance(), lost());
        assert!(matches!(
            fixture.notifications.try_recv().unwrap(),
            ResidencyNotification::WorkerFailed { failure, .. } if failure.code() == "worker_exited"
        ));
        fixture.inner.ended(&instance(), lost());
        assert!(fixture.notifications.try_recv().is_err());
    }
}
