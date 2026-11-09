//! Post-install kernel tuning: after a catalog download, one inference worker runs the model's
//! Prepare on the serving device, as a load does, so the first load finds every tuning record in the
//! kernel cache and is one worker, as any later load. The job resolves and previews exactly as a
//! load does and never imports the model. It runs only with the device to itself
//! ([`crate::device_lock`]): a search beside a resident or loading model would take the memory
//! that model's verdict promised its requests. While residency or another job holds the device
//! it waits, reported as `Waiting`, and runs once the device is free. Any load preempts it, and
//! it waits again; the units it settled stand, and those it left mid-search have only provisional
//! records, which no Prepare takes for a choice. A load of its own model searches what the job
//! would: while that load's Prepare runs the job reports its progress, and once the load has
//! prepared the model the job ends; a load whose Prepare did not finish leaves the job waiting.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use magnitude_engine::worker::Preparation as WorkerPreparation;
use magnitude_engine::worker::protocol::LoadProgress;
use magnitude_service_contracts::models::{
    CatalogOptimizationProgress, CatalogOptimizationStage, ModelId, ModelLoadDevice,
};
use magnitude_service_models::{ServableModelBundleKey, servable_model_bundle_key_for_bundle};
use tokio::sync::watch;

use super::controller::{ResidencyEnvironment, load_plan, preview, priced, request_prices};
use super::worker::EngineWorker;
use super::{ModelOperationFailure, ResidencyWorker as _};
use crate::device_lock::{DeviceHold, DeviceJob, DeviceLock};

type Report = Arc<dyn Fn(CatalogOptimizationProgress) + Send + Sync>;

/// The preparation jobs, one per bundle, running or waiting for the device, and the loads of
/// each bundle requested and not yet ended.
#[derive(Clone, Default)]
pub(crate) struct PreparationJobs(Arc<Mutex<Preparations>>);

#[derive(Default)]
struct Preparations {
    jobs: HashMap<ServableModelBundleKey, PreparationJob>,
    loads: HashMap<ServableModelBundleKey, usize>,
}

/// A job's progress report, its worker while one runs, and what loads of its bundle did:
/// whether one's Prepare reports for the waiting job, and whether one prepared the bundle.
struct PreparationJob {
    report: Report,
    device: Option<ModelLoadDevice>,
    worker: Option<Arc<EngineWorker>>,
    running: bool,
    load_reporting: bool,
    prepared: bool,
    changed: watch::Sender<()>,
}

/// A registered job; dropping it unregisters the job.
struct PreparationGuard {
    jobs: PreparationJobs,
    key: ServableModelBundleKey,
    changed: watch::Receiver<()>,
}

/// A requested load of a bundle, until the request ends: a job of that bundle leaves the device
/// to it meanwhile.
pub(crate) struct LoadRequest {
    jobs: PreparationJobs,
    key: ServableModelBundleKey,
}

/// A load's Prepare sequence for a bundle, reported as its job's progress while the job waits.
/// Dropped without [`LoadPreparation::prepared`], the job waits again.
pub(crate) struct LoadPreparation {
    jobs: PreparationJobs,
    key: ServableModelBundleKey,
    progression: OptimizationProgression,
    prepared: bool,
}

impl PreparationJobs {
    /// Register a job for `key`, reporting through `report`, unless one already runs or waits
    /// for it.
    fn register(&self, key: ServableModelBundleKey, report: Report) -> Option<PreparationGuard> {
        let mut preparations = self.0.lock().expect("preparation job lock");
        if preparations.jobs.contains_key(&key) {
            return None;
        }
        let (changed, receiver) = watch::channel(());
        preparations.jobs.insert(
            key.clone(),
            PreparationJob {
                report,
                device: None,
                worker: None,
                running: false,
                load_reporting: false,
                prepared: false,
                changed,
            },
        );
        Some(PreparationGuard {
            jobs: self.clone(),
            key,
            changed: receiver,
        })
    }

    /// A load of `key` was requested: a job of that bundle leaves the device to it until the
    /// request ends. Taken before the residency releases another model, whose release frees the
    /// device.
    pub(crate) fn request(&self, key: ServableModelBundleKey) -> LoadRequest {
        let mut preparations = self.0.lock().expect("preparation job lock");
        *preparations.loads.entry(key.clone()).or_default() += 1;
        if let Some(job) = preparations.jobs.get(&key) {
            job.changed.send_replace(());
        }
        LoadRequest {
            jobs: self.clone(),
            key,
        }
    }

    /// A load of `key` on `device` starts its Prepare sequence.
    pub(crate) fn load(&self, key: ServableModelBundleKey, device: ModelLoadDevice) -> LoadPreparation {
        LoadPreparation {
            jobs: self.clone(),
            key,
            progression: OptimizationProgression::new(device),
            prepared: false,
        }
    }

    /// `f` on `key`'s job, if one is registered, with whether a load of `key` is requested.
    fn with_job<T>(
        &self,
        key: &ServableModelBundleKey,
        f: impl FnOnce(&mut PreparationJob, bool) -> T,
    ) -> Option<T> {
        let mut preparations = self.0.lock().expect("preparation job lock");
        let loading = preparations.loads.contains_key(key);
        preparations.jobs.get_mut(key).map(|job| f(job, loading))
    }
}

impl Drop for LoadRequest {
    fn drop(&mut self) {
        let mut preparations = self.jobs.0.lock().expect("preparation job lock");
        if let Some(loads) = preparations.loads.get_mut(&self.key) {
            *loads -= 1;
            if *loads == 0 {
                preparations.loads.remove(&self.key);
            }
        }
        if let Some(job) = preparations.jobs.get(&self.key) {
            job.changed.send_replace(());
        }
    }
}

impl LoadPreparation {
    /// The load's Prepare progressed: a waiting job of its bundle reports it as its own.
    pub(crate) fn report(&mut self, progress: LoadProgress) {
        let Some(update) = self.progression.advance(progress) else {
            return;
        };
        self.jobs.with_job(&self.key, |job, _| {
            if !job.running {
                job.load_reporting = true;
                (job.report)(update);
            }
        });
    }

    /// The load found its bundle's tuning finished, or finished it: its job has nothing to do.
    pub(crate) fn prepared(mut self) {
        self.prepared = true;
    }
}

impl Drop for LoadPreparation {
    fn drop(&mut self) {
        let prepared = self.prepared;
        self.jobs.with_job(&self.key, |job, _| {
            if prepared {
                job.prepared = true;
                job.changed.send_replace(());
            } else if std::mem::take(&mut job.load_reporting) && !job.running {
                (job.report)(CatalogOptimizationProgress::waiting(job.device.clone()));
            }
        });
    }
}

/// What a job that wants the device finds.
enum Admission {
    Run(DeviceJob),
    Wait { loading: bool },
    Prepared,
}

impl PreparationGuard {
    fn job<T>(&self, f: impl FnOnce(&mut PreparationJob, bool) -> T) -> T {
        self.jobs
            .with_job(&self.key, f)
            .expect("a guarded job is registered")
    }

    /// The device a load of the job's model selects, once previewed.
    fn select(&self, device: ModelLoadDevice) {
        self.job(|job, _| job.device = Some(device));
    }

    /// Report the job's own progress.
    fn report(&self, progress: CatalogOptimizationProgress) {
        self.job(|job, _| (job.report)(progress));
    }

    /// Hand the job its worker, which ends with the job.
    fn attach(&self, worker: Arc<EngineWorker>) {
        self.job(|job, _| job.worker = Some(worker));
    }

    /// The device for the job, `taken` or from `device`, unless a load of its bundle has
    /// prepared it or wants the device.
    fn admit(&self, device: &DeviceLock, taken: Option<DeviceJob>) -> Admission {
        self.job(|job, loading| {
            if job.prepared {
                return Admission::Prepared;
            }
            if loading {
                return Admission::Wait { loading };
            }
            match taken.or_else(|| device.try_job()) {
                Some(hold) => {
                    job.running = true;
                    job.load_reporting = false;
                    Admission::Run(hold)
                }
                None => Admission::Wait { loading },
            }
        })
    }

    /// The job waits for the device: reported `Waiting`, unless a load's Prepare of its bundle
    /// reports for it.
    fn wait(&self) {
        self.job(|job, _| {
            job.running = false;
            job.worker = None;
            if !job.load_reporting {
                (job.report)(CatalogOptimizationProgress::waiting(job.device.clone()));
            }
        });
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
            .jobs
            .remove(&self.key)
            .and_then(|job| job.worker);
        if let Some(worker) = worker {
            worker.end("preparation_ended", "the installation's optimization ended");
        }
    }
}

/// Prepare `model_id`'s programs on the device a load would select. Every outcome leaves the
/// model installed; one that did not finish is logged and left to the next load.
pub(crate) async fn optimize(
    environment: &ResidencyEnvironment,
    jobs: &PreparationJobs,
    model_id: &ModelId,
    progress: Report,
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
    progress: Report,
) -> Result<Preparation, ModelOperationFailure> {
    let configuration = environment.model_variants.serving_configuration(model_id)?;
    let key = servable_model_bundle_key_for_bundle(&configuration.bundle);
    let Some(mut job) = jobs.register(key, Arc::clone(&progress)) else {
        return Ok(Preparation::Skipped("the bundle is already being prepared"));
    };
    let resolved = environment.configurations.resolve(&configuration).await?;
    // A model this device cannot load has nothing to tune for.
    // Its plan waits for its prices as a load's does, holding nothing of the device.
    let previewed = priced(
        &environment.device,
        || preview(&environment.catalog, &resolved),
        || request_prices(&environment.catalog, &resolved),
    );
    let Ok(preview) = previewed.await else {
        return Ok(Preparation::Skipped("the load preview failed"));
    };
    let device = load_plan(&preview)?.device;
    job.select(device.clone());
    let manifest = resolved.manifest.clone();
    loop {
        let Some(hold) = take_device(&environment.device, &mut job).await else {
            return Ok(Preparation::Skipped("its model was loaded, and the load prepared it"));
        };
        job.report(CatalogOptimizationProgress::preparing(Some(device.clone())));
        let mut preemption = hold.preemption();
        let spawned = EngineWorker::spawn(&environment.launcher, DeviceHold::Job(hold)).map_err(
            |error| ModelOperationFailure::new("worker_spawn_failed", format!("{error:#}"), true),
        )?;
        let worker = Arc::clone(&spawned.worker);
        job.attach(Arc::clone(&worker));
        let preparing = magnitude_service_utils::blocking::run({
            let manifest = manifest.clone();
            let progress = Arc::clone(&progress);
            let mut progression = OptimizationProgression::new(device.clone());
            move || {
                magnitude_engine::worker::prepare(
                    &manifest,
                    || Ok(spawned.into_process()),
                    move |update| {
                        if let Some(update) = progression.advance(update) {
                            progress(update);
                        }
                    },
                )
            }
        });
        tokio::pin!(preparing);
        // A load wanting the device preempts the job: its worker is killed, and the load proceeds
        // once it is proven retired.
        let (prepared, preempted) = tokio::select! {
            prepared = &mut preparing => (prepared, false),
            () = preemption.preempted() => {
                worker.end("preparation_preempted", "a load takes the device");
                (preparing.as_mut().await, true)
            }
        };
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

/// The device to itself for `job`, at once or once nothing else holds it and no load of its
/// bundle is requested. While it waits the job reports `Waiting`, or a load's Prepare of its
/// bundle reports for it, and it tries again after each release of the device and each change
/// to its bundle's loads, as the measurement job does. `None` once a load of its bundle has
/// prepared it.
async fn take_device(device: &DeviceLock, job: &mut PreparationGuard) -> Option<DeviceJob> {
    let mut taken = None;
    loop {
        // A change after this is seen by `changed` below; the device's releases are counted.
        job.changed.borrow_and_update();
        let loading = match job.admit(device, taken.take()) {
            Admission::Run(hold) => return Some(hold),
            Admission::Prepared => return None,
            Admission::Wait { loading } => loading,
        };
        job.wait();
        // While a load of its bundle is requested the job does not contend for the device.
        taken = tokio::select! {
            hold = device.job(), if !loading => Some(hold),
            _ = job.changed.changed() => None,
        };
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
            LoadProgress::Tuning { completed, total, .. } => {
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

    fn metal() -> ModelLoadDevice {
        ModelLoadDevice {
            id: HardwareDeviceId::new("metal:0".to_owned()),
            backend: ExecutionBackend::Metal,
        }
    }

    /// A job's reports, in order.
    #[derive(Clone, Default)]
    struct Reports(Arc<Mutex<Vec<CatalogOptimizationProgress>>>);

    impl Reports {
        fn report(&self) -> Report {
            let reports = self.clone();
            Arc::new(move |progress| reports.0.lock().unwrap().push(progress))
        }

        fn last(&self) -> Option<CatalogOptimizationProgress> {
            self.0.lock().unwrap().last().cloned()
        }
    }

    fn register(jobs: &PreparationJobs, reports: &Reports) -> PreparationGuard {
        let job = jobs.register(bundle("downloaded"), reports.report()).unwrap();
        job.select(metal());
        job
    }

    async fn settle() {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    fn tuning(completed: u64, total: u64) -> LoadProgress {
        LoadProgress::Tuning {
            completed,
            total,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
        }
    }

    /// QA optimization-memory: an optimization deferred while a model is resident is kept, not
    /// dropped. It reports `Waiting` while it waits, never done, and takes the device once the
    /// resident model releases it.
    #[tokio::test]
    async fn an_optimization_waits_for_a_resident_model_and_runs_once_the_device_is_free() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let reports = Reports::default();
        let mut job = register(&jobs, &reports);
        let resident = device.try_share().unwrap();
        let taking = tokio::spawn({
            let device = device.clone();
            async move { take_device(&device, &mut job).await.map(|hold| (hold, job)) }
        });
        settle().await;
        assert!(!taking.is_finished(), "the job waits while a model is resident");
        assert_eq!(reports.last(), Some(CatalogOptimizationProgress::waiting(Some(metal()))));
        let other = jobs.request(bundle("resident"));
        drop(other);
        drop(resident);
        let taken = taking.await.unwrap();
        assert!(taken.is_some(), "the job takes the device once it is free");
        assert!(device.try_share().is_none());
    }

    /// A load of the job's own model reports its Prepare as the job's progress, and once it has
    /// prepared the model the job ends without searching.
    #[tokio::test]
    async fn a_load_that_prepares_the_model_reports_its_tuning_and_ends_the_job() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let reports = Reports::default();
        let mut job = register(&jobs, &reports);
        let request = jobs.request(bundle("downloaded"));
        let taking = tokio::spawn({
            let device = device.clone();
            async move { take_device(&device, &mut job).await.is_some() }
        });
        settle().await;
        assert!(!taking.is_finished(), "the job leaves a free device to the requested load");
        assert_eq!(reports.last(), Some(CatalogOptimizationProgress::waiting(Some(metal()))));
        let share = device.try_share().unwrap();
        let mut load = jobs.load(bundle("downloaded"), metal());
        load.report(tuning(3, 10));
        assert_eq!(
            reports.last(),
            Some(CatalogOptimizationProgress {
                stage: CatalogOptimizationStage::Tuning,
                completed: 3,
                total: 10,
                device: Some(metal()),
            }),
            "the job shows the load's tuning"
        );
        load.report(tuning(10, 10));
        load.prepared();
        assert!(!taking.await.unwrap(), "the prepared model's job takes nothing");
        assert_eq!(reports.last().map(|last| last.stage), Some(CatalogOptimizationStage::Tuning));
        drop(share);
        drop(request);
        assert!(device.try_job().is_some(), "the job released the device");
    }

    /// A load of the job's own model whose Prepare did not finish leaves the job pending: it
    /// reports `Waiting` again and runs once the load's request ends and the device is free.
    #[tokio::test]
    async fn a_load_that_did_not_prepare_the_model_leaves_the_job_waiting() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let reports = Reports::default();
        let mut job = register(&jobs, &reports);
        let request = jobs.request(bundle("downloaded"));
        let share = device.try_share().unwrap();
        let taking = tokio::spawn({
            let device = device.clone();
            async move { take_device(&device, &mut job).await.map(|hold| (hold, job)) }
        });
        settle().await;
        let mut load = jobs.load(bundle("downloaded"), metal());
        load.report(tuning(3, 10));
        drop(load);
        assert_eq!(reports.last(), Some(CatalogOptimizationProgress::waiting(Some(metal()))));
        drop(share);
        settle().await;
        assert!(!taking.is_finished(), "the job waits while the load is requested");
        drop(request);
        assert!(taking.await.unwrap().is_some(), "the job runs once the device is free");
    }

    /// With the device free, the job takes it at once and reports no wait.
    #[tokio::test]
    async fn an_optimization_takes_a_free_device_at_once() {
        let device = DeviceLock::default();
        let jobs = PreparationJobs::default();
        let reports = Reports::default();
        let mut job = register(&jobs, &reports);
        let hold = take_device(&device, &mut job).await;
        assert!(hold.is_some());
        assert_eq!(reports.last(), None);
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
                    work: None,
                    measured_configurations: 0,
                    formed_programs: 0,
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
