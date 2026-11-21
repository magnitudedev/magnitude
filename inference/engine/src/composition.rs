//! The engine facade: `EngineConfiguration` → `resolve()` (device-free host
//! resolution) → `ResolvedEngineConfiguration` → `preview()` (read-only
//! planning) or a load (a numerical worker, in-process or in a worker
//! process) → `ReadyEngine`.

use crate::error::{ArtifactError, LoadError, ResolveError, UnsupportedModel};
use crate::host::HostArtifacts;
use crate::options::{ExecutionManifest, InputModalities, ModelPolicy, PackageOptions, ReadyInfo};
use crate::execution::{request_load_points, LoadAttempt};
use crate::preview::now;
use crate::worker::{
    connect_worker, load_prepared,
    measure::{InProcessJobs, JobsStopped},
    prepare, protocol::LoadProgress, run_priced, run_worker, start_worker_process,
    transport::channel_pair, AcquiredDevice, EngineClient, Preparation, PricedError,
    PricedFailure, WorkerConnection,
};
use magnitude_executor::UnpreparedUnits;
use magnitude_artifacts::{Package, PackageIdentity};
use magnitude_executor::{
    image_cell_limit, max_image_cells,
    platform::{DeviceRequest, ReleaseRate},
    ExecutionPath,
};
use magnitude_scheduler::ServiceLimits;
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

/// Host-owned inputs to engine construction. Resolving this value performs all
/// filesystem and family interpretation needed before the numerical worker is
/// started, but it never opens a device or imports a tensor.
#[derive(Clone, Debug)]
pub struct EngineConfiguration {
    pub package: PackageOptions,
    pub model: ModelPolicy,
    /// Host-selected serving context. The executor sizes all context-bound
    /// state to this limit. `None` serves the artifact's declared maximum.
    pub context_tokens: Option<usize>,
    pub service: ServiceLimits,
    pub path: ExecutionPath,
    /// The device the numerical worker opens (selection rule §5.5). A host
    /// that previewed the load passes the previewed exact selector.
    pub device: DeviceRequest,
    /// The directory the host keeps formed kernels and tuning results in
    /// (`--cache-dir`; ACN names `<dataDir>/cache/kernels`). `None` caches
    /// nothing: no tuning record reaches a load, which forms every entry
    /// at its canonical default.
    pub kernel_cache: Option<PathBuf>,
    /// The machine's page-release rate (`ReleaseRate::of_machine(None)` for the
    /// service and CLI). The engine never defaults it.
    pub release: ReleaseRate,
}

/// The two authorities produced by host resolution: host artifacts (chat
/// semantics) stay on the host, while the owned, serializable manifest is the
/// only value a numerical worker needs. The host artifacts are shared from
/// resolution on: every engine and configuration that serves them holds them
/// so, and a resolution moved through its callers moves only the manifest.
pub struct ResolvedEngineConfiguration {
    pub host: Arc<HostArtifacts>,
    pub manifest: ExecutionManifest,
}

impl EngineConfiguration {
    pub fn resolve(self) -> Result<ResolvedEngineConfiguration, ResolveError> {
        let package = self.package.open().map_err(|error| {
            ResolveError::Artifact(ArtifactError::from_artifacts(error, self.package.target()))
        })?;
        let host = HostArtifacts::interpret(
            package,
            self.context_tokens,
            Some(self.service.launch_rows()),
        )?;
        let definition = host.definition();
        definition.validate().map_err(|error| {
            ResolveError::Unsupported(UnsupportedModel::unreadable(error.to_string()))
        })?;
        let invalid = |reason: String| ResolveError::InvalidConfiguration { reason };
        let model = self
            .model
            .resolve_admitted(definition, host.draft_unavailable())
            .map_err(invalid)?;
        let manifest = ExecutionManifest::new(
            host.package().manifest(),
            definition.clone(),
            model,
            self.service,
            self.path,
            self.device,
            self.kernel_cache,
            Some(host.package().store().to_path_buf()),
            self.release,
        )
        .map_err(invalid)?;
        Ok(ResolvedEngineConfiguration { host: Arc::new(host), manifest })
    }
}

/// A loaded engine: the host's chat semantics plus a client of its worker.
pub struct ReadyEngine {
    host: Arc<HostArtifacts>,
    client: EngineClient,
    ready: ReadyInfo,
}

/// A ready worker whose loaded model differs from the host's resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadinessMismatch {
    Package {
        host: PackageIdentity,
        worker: PackageIdentity,
    },
    TemplateFingerprint {
        host: String,
        worker: String,
    },
    Modalities {
        host: InputModalities,
        worker: InputModalities,
    },
    /// The host cannot bound its image resize to the cells the worker's
    /// device encodes per image.
    ImageCells { reason: String },
}

impl fmt::Display for ReadinessMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package { host, worker } => write!(
                formatter,
                "the worker loaded package {worker} but the host resolved {host}"
            ),
            Self::TemplateFingerprint { host, worker } => write!(
                formatter,
                "the worker's chat template fingerprint {worker} differs from the host's {host}"
            ),
            Self::Modalities { host, worker } => write!(
                formatter,
                "the worker accepts {worker:?} but the host resolved {host:?}"
            ),
            Self::ImageCells { reason } => write!(
                formatter,
                "the host cannot bound its images to the cells the worker's device encodes: {reason}"
            ),
        }
    }
}

impl std::error::Error for ReadinessMismatch {}

impl ReadyEngine {
    /// Bind host artifacts to a connected worker. The worker must have
    /// loaded exactly the package the host resolved and read the same chat
    /// templates and input modalities from it. The engine's host resizes
    /// images to the cells the worker's device encodes.
    pub fn new(
        host: Arc<HostArtifacts>,
        connection: WorkerConnection,
    ) -> Result<Self, ReadinessMismatch> {
        let ready = &connection.ready;
        if ready.package != host.package().identity() {
            return Err(ReadinessMismatch::Package {
                host: host.package().identity(),
                worker: ready.package,
            });
        }
        let fingerprint = &host.template_inspection().fingerprint;
        if &ready.template_fingerprint != fingerprint {
            return Err(ReadinessMismatch::TemplateFingerprint {
                host: fingerprint.clone(),
                worker: ready.template_fingerprint.clone(),
            });
        }
        if ready.modalities != host.modalities() {
            return Err(ReadinessMismatch::Modalities {
                host: host.modalities(),
                worker: ready.modalities,
            });
        }
        let host = image_cell_limit(
            host.definition(),
            ready.service.launch_rows(),
            max_image_cells(ready.backend),
        )
        .map_err(|error| error.to_string())
        .and_then(|cells| host.within_image_cells(cells))
        .map_err(|reason| ReadinessMismatch::ImageCells { reason })?;
        Ok(Self {
            host,
            client: connection.client,
            ready: connection.ready,
        })
    }

    pub fn ready_info(&self) -> &ReadyInfo {
        &self.ready
    }

    pub fn host(&self) -> &Arc<HostArtifacts> {
        &self.host
    }

    pub fn client(&self) -> &EngineClient {
        &self.client
    }

    pub fn into_parts(self) -> (Arc<HostArtifacts>, EngineClient, ReadyInfo) {
        (self.host, self.client, self.ready)
    }
}

/// Run an in-process operation (`load_prepared`, `prepare`, or a standalone
/// price settlement) through [`run_priced`], measuring pending machine prices
/// via [`InProcessJobs`].
///
/// No device of this process is open while settling, so each job opens its
/// executor on tallies of its own and charges them (`worker::measure::measure_job`).
pub fn in_process_priced<T>(
    manifest: &ExecutionManifest,
    mut run: impl FnMut() -> Result<T, LoadError>,
) -> Result<T, LoadError> {
    let internal = |reason: String| LoadError::Internal { reason };
    let paths = vec![manifest.store().to_path_buf()];
    let catalog = seismic::Devices::discover(&paths).map_err(|error| internal(error.to_string()))?;
    let mut jobs = InProcessJobs::of_load(&catalog, paths, manifest).map_err(internal)?;
    let has_jobs = jobs.is_some();
    match now(run_priced(
        || {
            std::future::ready(if has_jobs {
                manifest.preview(&catalog).map(|_| ())
            } else {
                Ok(())
            })
        },
        || {
            std::future::ready((|| {
                let Some(jobs) = jobs.as_mut() else {
                    return Ok(());
                };
                manifest.request_prices(&catalog).map_err(internal)?;
                match jobs.run() {
                    Ok(_) | Err(JobsStopped::Stalled { .. }) => Ok(()),
                    Err(stopped) => {
                        Err(internal(format!("the plan's prices are pending, and {stopped}")))
                    }
                }
            })())
        },
        |&()| std::future::ready(Ok(AcquiredDevice::immediate(()))),
        |()| {},
        (),
        |&(), &(), ()| std::future::ready(run().map_err(|error| PricedFailure::Load(error, (), ()))),
    )) {
        Ok((value, (), ())) => Ok(value),
        Err(PricedError::Preview(error)) => Err(error.into()),
        Err(PricedError::Load(error, ()) | PricedError::Host(error)) => Err(error),
    }
}

/// Settle the machine's standing measurements and the load-scoped measurements
/// (`PointRequest::Site`, `TierCopy`, `Volume`, etc.) that a load of `manifest`
/// and `package` waits on, running [`InProcessJobs`] in this process until
/// nothing remains outstanding or a job stalls.
///
/// Nothing to do without a kernel cache, where no measurement reaches a load.
/// Standing figures required for planning settle first through
/// [`in_process_priced`], then [`request_load_points`] registers the load's
/// own points, and [`InProcessJobs`] loops until all standing and load
/// requests are recorded or a job makes no progress ([`JobsStopped::Stalled`]).
pub fn settle_load_measurements(
    manifest: &ExecutionManifest,
    package: &Package,
) -> Result<(), LoadError> {
    if manifest.kernel_cache.is_none() {
        return Ok(());
    }
    let internal = |reason: String| LoadError::Internal { reason };
    in_process_priced(manifest, || Ok(()))?;
    let paths = vec![manifest.store().to_path_buf()];
    let catalog = Rc::new(
        seismic::Devices::discover(&paths).map_err(|error| internal(error.to_string()))?,
    );
    request_load_points(&catalog, manifest, package)?;
    let mut jobs = InProcessJobs::of_load(&catalog, paths, manifest)
        .map_err(internal)?
        .expect("a load with a kernel cache has jobs");
    let mut left = jobs.outstanding().map_err(internal)?;
    while !left.is_empty() {
        match jobs.run() {
            Ok(run) => left = run.outstanding,
            Err(stopped @ JobsStopped::Stalled { .. }) => {
                tracing::warn!("in-process measurements stalled: {stopped}");
                break;
            }
            Err(stopped) => return Err(internal(stopped.to_string())),
        }
    }
    Ok(())
}

/// The model's Prepare, as a host that loads in its own process runs it
/// for no load ([`prepare`]), settling pending prices through [`run_priced`].
pub fn prepare_in_process(
    manifest: &ExecutionManifest,
    mut worker: std::process::Command,
    mut progress: impl FnMut(LoadProgress),
) -> Result<Preparation, LoadError> {
    in_process_priced(manifest, || {
        prepare(manifest, || start_worker_process(&mut worker), &mut progress)
    })
}

/// [`load_prepared`] for a host that loads in its own process: `load` is its
/// load, and `worker` starts the host's executable in its worker role
/// ([`crate::worker::serve_worker`] over its standard streams) for the
/// model's Prepare when the load finds tuning units without a stored record.
/// Standing and load-scoped measurements settle first through
/// [`settle_load_measurements`] when a kernel cache is configured.
/// A Prepare that does not finish is reported on the diagnostic stream; the
/// load then serves canonical defaults for the units without a record. The
/// stream also carries `kernel preparation began` when the Prepare's process
/// is started and `kernel preparation ended in S s` once it is gone.
pub fn load_in_process<L>(
    manifest: &ExecutionManifest,
    package: &Package,
    mut load: impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<L>, LoadError>,
    mut worker: std::process::Command,
    mut prepare_progress: impl FnMut(LoadProgress),
) -> Result<L, LoadError> {
    settle_load_measurements(manifest, package)?;
    // The diagnostic stream says when the Prepare's process began and that it
    // is gone, so a log's reader can tell preparation from the load itself.
    let preparing = std::cell::Cell::new(None);
    let (loaded, preparation) = in_process_priced(manifest, || {
        load_prepared(
            manifest,
            |unprepared| {
                if let Some(began) = preparing.take() {
                    let began: std::time::Instant = began;
                    tracing::info!(
                        "kernel preparation ended in {:.3} s",
                        began.elapsed().as_secs_f64()
                    );
                }
                load(unprepared)
            },
            || {
                tracing::info!("kernel preparation began");
                preparing.set(Some(std::time::Instant::now()));
                start_worker_process(&mut worker)
            },
            &mut prepare_progress,
        )
    })?;
    if let Preparation::Incomplete(error) = preparation {
        tracing::warn!(
            "kernel preparation did not finish ({error}); the load serves \
             canonical defaults for the units without a stored tuning record"
        );
    }
    Ok(loaded)
}

/// Load the model on a worker thread of this process over the in-process
/// transport: the same worker and protocol a worker process runs. Like every
/// load it searches nothing: `worker` starts this executable in its worker
/// role for the model's Prepare when records are missing
/// ([`load_in_process`]).
pub fn start_in_process(
    resolved: ResolvedEngineConfiguration,
    worker: std::process::Command,
    progress: impl FnMut(LoadProgress),
) -> Result<ReadyEngine, LoadError> {
    let ResolvedEngineConfiguration { host, manifest } = resolved;
    // The load's attempts and the Prepare report to one observer, in turn.
    let progress = std::cell::RefCell::new(progress);
    let attempt = |unprepared| -> Result<LoadAttempt<ReadyEngine>, LoadError> {
        let (host_end, worker_end) = channel_pair();
        let worker_manifest = manifest.clone();
        let worker = seismic::threads::spawn("magnitude-worker", move || {
            let exit = run_worker(worker_manifest, unprepared, worker_end);
            tracing::info!("in-process worker exited: {exit:?}");
        })
        .map_err(|error| LoadError::Internal {
            reason: error.to_string(),
        })?;
        // An attempt that did not load has released everything it opened
        // once its thread ends, which it does as its transport closes: its
        // device among them, so the jobs that settle its pending prices
        // open the device anew ([`in_process_priced`]).
        let unloaded = |outcome: Result<LoadAttempt<ReadyEngine>, LoadError>| {
            worker.join().expect("the in-process worker's panic ends the process");
            outcome
        };
        match connect_worker(host_end, None, |update| (progress.borrow_mut())(update)) {
            Ok(LoadAttempt::Loaded(connection)) => {
                match ReadyEngine::new(Arc::clone(&host), *connection) {
                    Ok(ready) => Ok(LoadAttempt::Loaded(Box::new(ready))),
                    Err(mismatch) => unloaded(Err(LoadError::Internal {
                        reason: mismatch.to_string(),
                    })),
                }
            }
            Ok(LoadAttempt::Unprepared) => unloaded(Ok(LoadAttempt::Unprepared)),
            Err(error) => unloaded(Err(error)),
        }
    };
    load_in_process(&manifest, host.package(), attempt, worker, |update| {
        (progress.borrow_mut())(update)
    })
}
