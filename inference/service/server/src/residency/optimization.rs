//! Post-install kernel tuning: after a catalog download, one inference worker prepares the
//! model's programs on the device a load would select, so the first load finds its tuning in the
//! kernel cache. The job resolves and previews exactly as a load does, holds the residency side
//! of the device exclusion, and never imports the model. A load of the same bundle stops it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use magnitude_engine::worker::protocol::LoadProgress;
use magnitude_executor::platform::DeviceRequest;
use magnitude_service_contracts::models::{
    CatalogOptimizationProgress, CatalogOptimizationStage, ModelId, ModelLoadDevice,
};
use magnitude_service_models::{ServableModelBundleKey, servable_model_bundle_key_for_bundle};

use super::controller::{ResidencyEnvironment, load_plan, preview};
use super::worker::EngineWorker;
use super::{ModelOperationFailure, ResidencyWorker as _};

/// The running preparation jobs, one per bundle.
#[derive(Clone, Default)]
pub(crate) struct PreparationJobs(Arc<Mutex<HashMap<ServableModelBundleKey, PreparationJob>>>);

#[derive(Default)]
struct PreparationJob {
    stopped: bool,
    worker: Option<Arc<EngineWorker>>,
}

/// A registered job; dropping it unregisters the job.
struct PreparationGuard {
    jobs: PreparationJobs,
    key: ServableModelBundleKey,
}

impl PreparationJobs {
    /// Register a job for `key`, unless one already runs for it.
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

    /// Stop the job preparing `key`, if any, and prove its worker has exited.
    pub(crate) fn stop(&self, key: &ServableModelBundleKey) {
        let worker = {
            let mut jobs = self.0.lock().expect("preparation job lock");
            let Some(job) = jobs.get_mut(key) else {
                return;
            };
            job.stopped = true;
            job.worker.take()
        };
        if let Some(worker) = worker {
            worker.terminate("preparation_stopped", "a load of the model takes over its tuning");
        }
    }
}

impl PreparationGuard {
    /// Hand the job its worker; `false` when the job was already stopped.
    fn attach(&self, worker: Arc<EngineWorker>) -> bool {
        let mut jobs = self.jobs.0.lock().expect("preparation job lock");
        let job = jobs.get_mut(&self.key).expect("a guarded job is registered");
        if job.stopped {
            return false;
        }
        job.worker = Some(worker);
        true
    }

    fn stopped(&self) -> bool {
        self.jobs
            .0
            .lock()
            .expect("preparation job lock")
            .get(&self.key)
            .is_some_and(|job| job.stopped)
    }
}

/// A job ends with its worker: the blocking connection holds the worker too, so a stopped job
/// (cancel, remove) must terminate it rather than wait for the last reference to drop.
impl Drop for PreparationGuard {
    fn drop(&mut self) {
        let job = self
            .jobs
            .0
            .lock()
            .expect("preparation job lock")
            .remove(&self.key);
        if let Some(worker) = job.and_then(|job| job.worker) {
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
    progress: Arc<dyn Fn(CatalogOptimizationProgress) + Send + Sync>,
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
            "installed model optimization did not finish; its next load tunes"
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
    progress: Arc<dyn Fn(CatalogOptimizationProgress) + Send + Sync>,
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
    progress(CatalogOptimizationProgress::preparing(Some(device.clone())));
    let spawned = EngineWorker::spawn(&environment.launcher).map_err(|error| {
        ModelOperationFailure::new("worker_spawn_failed", format!("{error:#}"), true)
    })?;
    let worker = Arc::clone(&spawned.worker);
    if !job.attach(Arc::clone(&worker)) {
        return Ok(Preparation::Skipped("a load of the model stopped it"));
    }
    let mut manifest = resolved.manifest.clone();
    manifest.device = DeviceRequest::Selector(preview.device);
    let prepared = crate::spawn_blocking_traced(move || {
        let mut progression = OptimizationProgression::new(device);
        spawned.prepare(manifest, move |update| {
            if let Some(update) = progression.advance(update) {
                progress(update);
            }
        })
    })
    .await
    .map_err(|error| {
        ModelOperationFailure::new(
            "worker_exited",
            format!("worker preparation task failed: {error}"),
            true,
        )
    })?;
    match prepared {
        Ok(()) => Ok(Preparation::Prepared),
        Err(_) if job.stopped() => Ok(Preparation::Skipped("a load of the model stopped it")),
        Err(error) => Err(ModelOperationFailure::new(
            "model_optimization_failed",
            match worker.diagnostics() {
                diagnostics if diagnostics.is_empty() => error.to_string(),
                diagnostics => format!("{error}; worker diagnostics: {diagnostics}"),
            },
            error.retryable(),
        )),
    }
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
