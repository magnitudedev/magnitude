//! Post-install kernel tuning: after a catalog download, one inference worker runs the model's
//! Prepare on the serving device, as a load does, so the first load finds every tuning record in the
//! kernel cache and is one worker, as any later load. The job resolves and previews exactly as a
//! load does and never imports the model. It runs only with the device to itself
//! ([`crate::device_lock`]): a search beside a resident or loading model would take the memory
//! that model's verdict promised its requests. While residency or another job holds the device
//! it waits, reported as no optimization, and runs once the device is free. Any load preempts
//! it, and it waits again; the records it published stand. A load of its own model ends it: that
//! load's Prepare searched every unit the job would have.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use magnitude_engine::worker::Preparation as WorkerPreparation;
use magnitude_engine::worker::protocol::LoadProgress;
use magnitude_service_contracts::models::{
    CatalogOptimizationProgress, CatalogOptimizationStage, ModelId, ModelLoadDevice,
};
use magnitude_service_models::{ServableModelBundleKey, servable_model_bundle_key_for_bundle};

use super::controller::{ResidencyEnvironment, load_plan, preview};
use super::worker::EngineWorker;
use super::{ModelOperationFailure, ResidencyWorker as _};
use crate::device_lock::{DeviceHold, DeviceJob, DeviceLock};

/// The preparation jobs, one per bundle, running or waiting for the device.
#[derive(Clone, Default)]
pub(crate) struct PreparationJobs(Arc<Mutex<HashMap<ServableModelBundleKey, PreparationJob>>>);

/// A job's worker while one runs, and whether a load of its bundle was requested since.
#[derive(Default)]
struct PreparationJob {
    worker: Option<Arc<EngineWorker>>,
    loaded: bool,
}

/// A registered job; dropping it unregisters the job.
struct PreparationGuard {
    jobs: PreparationJobs,
    key: ServableModelBundleKey,
}

impl PreparationJobs {
    /// Register a job for `key`, unless one already runs or waits for it.
    fn register(&self, key: ServableModelBundleKey) -> Option<PreparationGuard> {
        let mut jobs = self.0.lock().expect("preparation job lock");
        if jobs.contains_key(&key) {
            return None;
        }
        jobs.insert(key.clone(), PreparationJob::default());
        Some(PreparationGuard {
            jobs: self.clone(),
            key,
        })
    }

    /// A load of `key` was requested: its Prepare searches what a job of that bundle would, so a
    /// job waiting for the device ends once it gets it.
    pub(crate) fn loading(&self, key: &ServableModelBundleKey) {
        if let Some(job) = self.0.lock().expect("preparation job lock").get_mut(key) {
            job.loaded = true;
        }
    }
}

impl PreparationGuard {
    /// Hand the job its worker, which ends with the job.
    fn attach(&self, worker: Arc<EngineWorker>) {
        let mut jobs = self.jobs.0.lock().expect("preparation job lock");
        jobs.get_mut(&self.key)
            .expect("a guarded job is registered")
            .worker = Some(worker);
    }

    /// Whether a load of the job's bundle was requested since the job was registered.
    fn loaded(&self) -> bool {
        self.jobs
            .0
            .lock()
            .expect("preparation job lock")
            .get(&self.key)
            .is_some_and(|job| job.loaded)
    }
}

/// A job ends with its worker: the blocking connection holds the worker too, so a stopped job
/// (cancel, remove) must terminate it rather than wait for the last reference to drop.
impl Drop for PreparationGuard {
    fn drop(&mut self) {
        let worker = self
            .jobs
            .0
            .lock()
            .expect("preparation job lock")
            .remove(&self.key)
            .and_then(|job| job.worker);
        if let Some(worker) = worker {
            worker.terminate("preparation_ended", "the installation's optimization ended");
        }
    }
}

/// Prepare `model_id`'s programs on the device a load would select. Every outcome leaves the
/// model installed; one that did not finish is logged and left to the next load.
pub(crate) async fn optimize(
    environment: &ResidencyEnvironment,
    jobs: &PreparationJobs,
    model_id: &ModelId,
    progress: Arc<dyn Fn(Option<CatalogOptimizationProgress>) + Send + Sync>,
) {
    let outcome = prepare(environment, jobs, model_id, progress).await;
    match outcome {
        Ok(Preparation::Prepared) => tracing::info!(%model_id, "installed model optimized"),
        Ok(Preparation::Skipped(reason)) => {
            tracing::info!(%model_id, reason, "installed model optimization skipped");
        }
        Err(failure) => tracing::warn!(
            %model_id,
            code = failure.code(),
            message = failure.message(),
            "installed model optimization did not finish; its next load's preparation continues it"
        ),
    }
}

enum Preparation {
    Prepared,
    Skipped(&'static str),
}

async fn prepare(
    environment: &ResidencyEnvironment,
    jobs: &PreparationJobs,
    model_id: &ModelId,
    progress: Arc<dyn Fn(Option<CatalogOptimizationProgress>) + Send + Sync>,
) -> Result<Preparation, ModelOperationFailure> {
    let configuration = environment.model_variants.serving_configuration(model_id)?;
    let Some(job) = jobs.register(servable_model_bundle_key_for_bundle(&configuration.bundle)) else {
        return Ok(Preparation::Skipped("the bundle is already being prepared"));
    };
    let resolved = environment.configurations.resolve(&configuration).await?;
    // A model this device cannot load has nothing to tune for.
    let Ok(preview) = preview(&environment.catalog, &resolved).await else {
        return Ok(Preparation::Skipped("the load preview failed"));
    };
    let device = load_plan(&preview)?.device;
    let manifest = resolved.manifest.clone();
    loop {
        let waiting = || {
            tracing::info!(%model_id, "installed model optimization waits for the device");
            progress(None);
        };
        let Some(hold) = take_device(&environment.device, &job, waiting).await else {
            return Ok(Preparation::Skipped("its model was loaded, and the load prepared it"));
        };
        progress(Some(CatalogOptimizationProgress::preparing(Some(device.clone()))));
        let mut preemption = hold.preemption();
        let spawned = EngineWorker::spawn(&environment.launcher, DeviceHold::Job(hold)).map_err(
            |error| ModelOperationFailure::new("worker_spawn_failed", format!("{error:#}"), true),
        )?;
        let worker = Arc::clone(&spawned.worker);
        job.attach(Arc::clone(&worker));
        let mut preparing = crate::spawn_blocking_traced({
            let manifest = manifest.clone();
            let progress = Arc::clone(&progress);
            let mut progression = OptimizationProgression::new(device.clone());
            move || {
                magnitude_engine::worker::prepare(
                    &manifest,
                    || Ok(spawned.into_process()),
                    move |update| {
                        if let Some(update) = progression.advance(update) {
                            progress(Some(update));
                        }
                    },
                )
            }
        });
        // A load wanting the device preempts the job: its worker is killed, and the load proceeds
        // once it is proven retired.
        let (prepared, preempted) = tokio::select! {
            prepared = &mut preparing => (prepared, false),
            () = preemption.preempted() => {
                worker.terminate("preparation_preempted", "a load takes the device");
                (preparing.await, true)
            }
        };
        let prepared = prepared.map_err(|error| {
            ModelOperationFailure::new(
                "worker_exited",
                format!("worker preparation task failed: {error}"),
                true,
            )
        })?;
        match prepared {
            Ok(WorkerPreparation::Prepared | WorkerPreparation::Unneeded) => {
                return Ok(Preparation::Prepared);
            }
            Ok(WorkerPreparation::Unshared) => {
                return Ok(Preparation::Skipped("the configuration names no kernel cache"));
            }
            Ok(WorkerPreparation::Incomplete(_)) | Err(_) if preempted => {
                tracing::info!(%model_id, "installed model optimization preempted; it waits for the device again");
            }
            Ok(WorkerPreparation::Incomplete(error)) | Err(error) => {
                return Err(ModelOperationFailure::new(
                    "model_optimization_failed",
                    match worker.diagnostics() {
                        diagnostics if diagnostics.is_empty() => error.to_string(),
                        diagnostics => format!("{error}; worker diagnostics: {diagnostics}"),
                    },
                    error.retryable(),
                ));
            }
        }
    }
}

/// The device to itself for `job`, at once or once nothing else holds it: while residency or
/// another job holds it, `waiting` reports that nothing optimizes, and the job waits for each
/// release, as the measurement job does. `None` when a load of the job's own model was
/// requested meanwhile: that load's Prepare searches what the job would.
async fn take_device(
    device: &DeviceLock,
    job: &PreparationGuard,
    waiting: impl FnOnce(),
) -> Option<DeviceJob> {
    let hold = match device.try_job() {
        Some(hold) => hold,
        None => {
            waiting();
            device.job().await
        }
    };
    (!job.loaded()).then_some(hold)
}

/// The least share of the tuning units reported again after, so a job publishes a bounded
/// number of snapshots however many units it tunes.
const REPORT_STEPS: u64 = 200;

/// A job's reported progress from its worker's: preparation, then tuning by work units.
struct OptimizationProgression {
    device: ModelLoadDevice,
    reported: Option<(CatalogOptimizationStage, u64)>,
}

impl OptimizationProgression {
    fn new(device: ModelLoadDevice) -> Self {
        Self {
            device,
            reported: None,
        }
    }

    fn advance(&mut self, progress: LoadProgress) -> Option<CatalogOptimizationProgress> {
        let (stage, completed, total) = match progress {
            LoadProgress::Preparing => (CatalogOptimizationStage::Preparing, 0, 0),
            LoadProgress::Tuning { completed, total } => {
                (CatalogOptimizationStage::Tuning, completed.min(total), total)
            }
            LoadProgress::ImportingWeights { .. } | LoadProgress::Finalizing => return None,
        };
        let step = completed
            .saturating_mul(REPORT_STEPS)
            .checked_div(total)
            .unwrap_or(0);
        if self.reported == Some((stage, step)) {
            return None;
        }
        self.reported = Some((stage, step));
        Some(CatalogOptimizationProgress {
            stage,
            completed,
            total,
            device: Some(self.device.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_service_contracts::{ExecutionBackend, HardwareDeviceId};

    fn bundle(name: &str) -> ServableModelBundleKey {
        ServableModelBundleKey(name.to_owned())
    }

    /// QA optimization-memory: an optimization deferred while a model is resident is kept, not
    /// dropped. It reports that nothing optimizes while it waits, and takes the device once the
    /// resident model releases it.
    #[tokio::test]
    async fn an_optimization_waits_for_a_resident_model_and_runs_once_the_device_is_free() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let job = jobs.register(bundle("downloaded")).unwrap();
        let resident = device.try_share().unwrap();
        let waited = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let taking = tokio::spawn({
            let device = device.clone();
            let waited = Arc::clone(&waited);
            async move {
                take_device(&device, &job, || {
                    waited.store(true, std::sync::atomic::Ordering::SeqCst);
                })
                .await
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!taking.is_finished(), "the job waits while a model is resident");
        assert!(waited.load(std::sync::atomic::Ordering::SeqCst));
        jobs.loading(&bundle("resident"));
        drop(resident);
        let hold = taking.await.unwrap();
        assert!(hold.is_some(), "the job takes the device once it is free");
        assert!(device.try_share().is_none());
    }

    /// A load of the job's own model while the job waits prepares the model: the job then ends
    /// without searching.
    #[tokio::test]
    async fn a_waiting_optimization_ends_once_its_model_is_loaded() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let job = jobs.register(bundle("downloaded")).unwrap();
        let load = device.try_share().unwrap();
        jobs.loading(&bundle("downloaded"));
        let taking = tokio::spawn({
            let device = device.clone();
            async move { take_device(&device, &job, || {}).await.is_some() }
        });
        drop(load);
        assert!(!taking.await.unwrap(), "the loaded model's job takes nothing");
        assert!(device.try_job().is_some(), "the job released the device");
    }

    /// With the device free, the job takes it at once and reports no wait.
    #[tokio::test]
    async fn an_optimization_takes_a_free_device_at_once() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let job = jobs.register(bundle("downloaded")).unwrap();
        let hold = take_device(&device, &job, || panic!("nothing holds the device")).await;
        assert!(hold.is_some());
    }

    #[test]
    fn tuning_progress_is_reported_in_bounded_steps() {
        let mut progression = OptimizationProgression::new(ModelLoadDevice {
            id: HardwareDeviceId::new("metal:0".to_owned()),
            backend: ExecutionBackend::Metal,
        });
        let preparing = progression.advance(LoadProgress::Preparing).unwrap();
        assert_eq!(preparing.stage, CatalogOptimizationStage::Preparing);
        assert!(progression.advance(LoadProgress::Preparing).is_none());
        let reported = (0..=10_000)
            .filter_map(|completed| {
                progression.advance(LoadProgress::Tuning {
                    completed,
                    total: 10_000,
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(reported.len(), REPORT_STEPS as usize + 1);
        let last = reported.last().unwrap();
        assert_eq!((last.completed, last.total), (10_000, 10_000));
        assert!(
            progression
                .advance(LoadProgress::ImportingWeights {
                    completed_bytes: 1,
                    total_bytes: 2,
                })
                .is_none()
        );
    }
}
