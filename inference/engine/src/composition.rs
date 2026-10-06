//! The engine facade: `EngineConfiguration` → `resolve()` (device-free host
//! resolution) → `ResolvedEngineConfiguration` → `preview()` (read-only
//! planning) or a load (a numerical worker, in-process or in a worker
//! process) → `ReadyEngine`.

use crate::error::{ArtifactError, LoadError, ResolveError, UnsupportedModel};
use crate::host::HostArtifacts;
use crate::options::{ExecutionManifest, InputModalities, ModelPolicy, PackageOptions, ReadyInfo};
use magnitude_artifacts::PackageIdentity;
use std::fmt;
use crate::worker::{
    connect_worker, protocol::LoadProgress, run_worker, transport::channel_pair, EngineClient,
    WorkerConnection,
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath,
};
use magnitude_scheduler::ServiceLimits;
use std::path::PathBuf;
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
    /// nothing: every load forms and tunes every entry.
    pub kernel_cache: Option<PathBuf>,
    /// The host's threshold policy (`MemoryReserves::standard()` for the
    /// service and CLI). The engine never defaults it.
    pub reserves: MemoryReserves,
}

/// The two authorities produced by host resolution: host artifacts (chat
/// semantics) stay on the host, while the owned, serializable manifest is the
/// only value a numerical worker needs.
pub struct ResolvedEngineConfiguration {
    pub host: HostArtifacts,
    pub manifest: ExecutionManifest,
}

impl EngineConfiguration {
    pub fn resolve(self) -> Result<ResolvedEngineConfiguration, ResolveError> {
        let package = self.package.open().map_err(|error| {
            ResolveError::Artifact(ArtifactError::from_artifacts(error, &self.package.target))
        })?;
        let host = HostArtifacts::interpret(
            package,
            self.context_tokens,
            Some(self.service.launch_rows()),
        )?;
        let definition = host.definition();
        definition.validate().map_err(|error| {
            ResolveError::Unsupported(UnsupportedModel::Representation {
                reason: error.to_string(),
            })
        })?;
        let invalid = |reason: String| ResolveError::InvalidConfiguration { reason };
        let model = self.model.resolve(definition).map_err(invalid)?;
        let manifest = ExecutionManifest::new(
            host.package().manifest(),
            definition.clone(),
            model,
            self.service,
            self.path,
            self.device,
            self.kernel_cache,
            self.reserves,
        )
        .map_err(invalid)?;
        Ok(ResolvedEngineConfiguration { host, manifest })
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
        }
    }
}

impl std::error::Error for ReadinessMismatch {}

impl ReadyEngine {
    /// Bind host artifacts to a connected worker. The worker must have
    /// loaded exactly the package the host resolved and read the same chat
    /// templates and input modalities from it.
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

/// Load the model on a worker thread of this process over the in-process
/// transport: the same worker and protocol a worker process runs.
pub fn start_in_process(
    resolved: ResolvedEngineConfiguration,
    progress: impl FnMut(LoadProgress),
) -> Result<ReadyEngine, LoadError> {
    let ResolvedEngineConfiguration { host, manifest } = resolved;
    let (host_end, worker_end) = channel_pair();
    std::thread::Builder::new()
        .name("magnitude-worker".into())
        .spawn(move || {
            let exit = run_worker(manifest, worker_end);
            eprintln!("magnitude-engine: in-process worker exited: {exit:?}");
        })
        .map_err(|error| LoadError::Internal {
            reason: error.to_string(),
        })?;
    let connection = connect_worker(host_end, None, progress)?;
    ReadyEngine::new(Arc::new(host), connection).map_err(|mismatch| LoadError::Internal {
        reason: mismatch.to_string(),
    })
}
