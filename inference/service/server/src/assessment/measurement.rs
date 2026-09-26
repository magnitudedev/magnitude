//! The measurement job: the environment's fixed measurement basis, measured once in one contained
//! child process on the selected device and cached in the service's assessment cache.
//!
//! The child opens only the device it is given, loads the basis stored for that device's exact
//! measurement identity or measures and stores it, and reports the identity. The service then
//! reads the basis from the shared cache directory. Opening a device, forming kernels and timing
//! them never happens in the service process.

use std::fmt;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use magnitude_engine::worker::protocol::EngineBuild;
use magnitude_executor::assessment::{
    BasisIdentity, MeasurementBasis, MeasurementError, load_basis, measure_basis, store_basis,
};
use magnitude_executor::platform::{self, MemoryReserves, PlatformConfig, PlatformError};
use magnitude_executor::{DEFAULT_KERNEL_CACHE_BYTES, ExecutionPath, KernelCache};
use seismic::{DeviceCatalog, DeviceSelector};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt as _;

use crate::worker_process::{WorkerLauncher, WorkerRole};

/// Bounds the complete job, including a cold kernel cache on the slowest backend.
const MEASUREMENT_DEADLINE: Duration = Duration::from_secs(15 * 60);
const RETIREMENT_DEADLINE: Duration = Duration::from_secs(5);
const MAX_REPORT_BYTES: u64 = 64 * 1024;
const MAX_DIAGNOSTIC_BYTES: u64 = 64 * 1024;

/// The arguments of the `measurement-worker` subcommand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasurementWorkerArgs {
    /// The exact device to measure: the device assessment and loads select.
    pub device: DeviceSelector,
    pub basis_directory: PathBuf,
    pub kernel_directory: PathBuf,
}

impl MeasurementWorkerArgs {
    fn append_to(&self, command: &mut std::process::Command) {
        command
            .arg("--device")
            .arg(self.device.to_string())
            .arg("--basis-dir")
            .arg(&self.basis_directory)
            .arg("--kernel-dir")
            .arg(&self.kernel_directory);
    }
}

/// How the child obtained the basis.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BasisSource {
    /// The basis stored for this identity was reused.
    Cached,
    /// The basis was measured now.
    Measured { seconds: f64 },
}

/// The child's single stdout record.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasurementReport {
    engine_build: String,
    backend: String,
    device: String,
    protocol_version: u32,
    source: BasisSource,
}

impl MeasurementReport {
    fn identity(&self) -> BasisIdentity {
        BasisIdentity {
            engine_build: self.engine_build.clone(),
            backend: self.backend.clone(),
            device: self.device.clone(),
            protocol_version: self.protocol_version,
        }
    }
}

#[derive(Debug)]
pub enum MeasurementWorkerError {
    Discovery(String),
    KernelCache(String),
    Platform(PlatformError),
    Measurement(MeasurementError),
    Store(std::io::Error),
    Report(std::io::Error),
}

impl fmt::Display for MeasurementWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(error) => write!(formatter, "device discovery: {error}"),
            Self::KernelCache(error) => write!(formatter, "kernel cache: {error}"),
            Self::Platform(error) => write!(formatter, "device: {error}"),
            Self::Measurement(error) => write!(formatter, "measurement: {error}"),
            Self::Store(error) => write!(formatter, "storing the basis: {error}"),
            Self::Report(error) => write!(formatter, "reporting the basis: {error}"),
        }
    }
}

impl std::error::Error for MeasurementWorkerError {}

/// The `measurement-worker` child: establish the basis of `args.device` and report its identity
/// on stdout.
pub fn run_measurement_worker(args: MeasurementWorkerArgs) -> Result<(), MeasurementWorkerError> {
    let catalog = DeviceCatalog::discover()
        .map_err(|error| MeasurementWorkerError::Discovery(error.to_string()))?;
    let reserves = MemoryReserves::standard();
    let kernels = KernelCache::open(args.kernel_directory, DEFAULT_KERNEL_CACHE_BYTES)
        .map_err(|error| MeasurementWorkerError::KernelCache(error.to_string()))?;
    let opened = platform::open_selected(
        &catalog,
        args.device,
        PlatformConfig {
            path: ExecutionPath::Native,
            artifacts: Some(Arc::new(kernels) as Arc<dyn seismic::ArtifactStore>),
            reserves,
        },
    )
    .map_err(MeasurementWorkerError::Platform)?;
    let identity = BasisIdentity::for_device(opened.device(), &EngineBuild::current().0);
    let source = if load_basis(&args.basis_directory, &identity).is_some() {
        BasisSource::Cached
    } else {
        let started = Instant::now();
        let basis = measure_basis(&catalog, opened.device(), reserves, identity.clone())
            .map_err(MeasurementWorkerError::Measurement)?;
        let seconds = started.elapsed().as_secs_f64();
        store_basis(&args.basis_directory, &basis).map_err(MeasurementWorkerError::Store)?;
        BasisSource::Measured { seconds }
    };
    let report = MeasurementReport {
        engine_build: identity.engine_build,
        backend: identity.backend,
        device: identity.device,
        protocol_version: identity.protocol_version,
        source,
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report)
        .map_err(|error| MeasurementWorkerError::Report(error.into()))?;
    stdout
        .write_all(b"\n")
        .and_then(|()| stdout.flush())
        .map_err(MeasurementWorkerError::Report)
}

/// A basis established by the measurement job.
pub struct EstablishedBasis {
    pub basis: MeasurementBasis,
    pub source: BasisSource,
    /// Wall time of the whole job, process start to basis read.
    pub elapsed: Duration,
}

#[derive(Debug)]
pub enum MeasurementJobError {
    Spawn(String),
    Deadline,
    Io(std::io::Error),
    /// The child failed; its bounded diagnostic tail is kept.
    Failed {
        status: std::process::ExitStatus,
        diagnostics: String,
    },
    MalformedReport(String),
    /// The child reported a basis the cache does not hold.
    MissingBasis(String),
}

impl fmt::Display for MeasurementJobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(error) => {
                write!(formatter, "failed to start the measurement worker: {error}")
            }
            Self::Deadline => write!(formatter, "the measurement worker exceeded its deadline"),
            Self::Io(error) => write!(formatter, "measurement worker I/O: {error}"),
            Self::Failed {
                status,
                diagnostics,
            } => write!(
                formatter,
                "the measurement worker failed ({status}): {diagnostics}"
            ),
            Self::MalformedReport(error) => {
                write!(
                    formatter,
                    "the measurement worker report is malformed: {error}"
                )
            }
            Self::MissingBasis(device) => write!(
                formatter,
                "the measurement worker reported a basis for {device} that the cache does not hold"
            ),
        }
    }
}

impl std::error::Error for MeasurementJobError {}

/// Runs the measurement job. Measurement and model residency exclude each other on the device:
/// the job holds [`DeviceExclusion::measurement`] for its whole duration.
#[derive(Clone)]
pub struct MeasurementJob {
    launcher: WorkerLauncher,
    basis_directory: PathBuf,
    kernel_directory: PathBuf,
    exclusion: DeviceExclusion,
}

impl MeasurementJob {
    pub fn new(
        launcher: WorkerLauncher,
        basis_directory: PathBuf,
        kernel_directory: PathBuf,
        exclusion: DeviceExclusion,
    ) -> Self {
        Self {
            launcher,
            basis_directory,
            kernel_directory,
            exclusion,
        }
    }

    pub async fn establish(
        &self,
        device: DeviceSelector,
    ) -> Result<EstablishedBasis, MeasurementJobError> {
        let _exclusive = self.exclusion.measurement().await;
        let started = Instant::now();
        let args = MeasurementWorkerArgs {
            device,
            basis_directory: self.basis_directory.clone(),
            kernel_directory: self.kernel_directory.clone(),
        };
        let mut command = self
            .launcher
            .command(WorkerRole::Measurement)
            .map_err(|error| MeasurementJobError::Spawn(format!("{error:#}")))?;
        args.append_to(&mut command);
        let mut child = self
            .launcher
            .spawn(command)
            .map_err(|error| MeasurementJobError::Spawn(format!("{error:#}")))?;
        drop(child.stdin.take());
        let stdout = child.stdout.take().ok_or_else(|| {
            MeasurementJobError::Spawn("measurement worker stdout is unavailable".to_owned())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            MeasurementJobError::Spawn("measurement worker stderr is unavailable".to_owned())
        })?;
        let outcome = tokio::time::timeout(MEASUREMENT_DEADLINE, async {
            let (report, diagnostics, status) = tokio::join!(
                read_bounded(stdout, MAX_REPORT_BYTES),
                read_bounded(stderr, MAX_DIAGNOSTIC_BYTES),
                child.wait(),
            );
            Ok::<_, std::io::Error>((report?, diagnostics?, status?))
        })
        .await;
        let (report, diagnostics, status) = match outcome {
            Ok(result) => result.map_err(MeasurementJobError::Io)?,
            Err(_) => {
                retire(&mut child).await;
                return Err(MeasurementJobError::Deadline);
            }
        };
        if !status.success() {
            return Err(MeasurementJobError::Failed {
                status,
                diagnostics: String::from_utf8_lossy(&diagnostics).trim().to_owned(),
            });
        }
        let report: MeasurementReport = serde_json::from_slice(&report)
            .map_err(|error| MeasurementJobError::MalformedReport(error.to_string()))?;
        let identity = report.identity();
        let directory = self.basis_directory.clone();
        let basis = crate::spawn_blocking_traced(move || load_basis(&directory, &identity))
            .await
            .map_err(|error| MeasurementJobError::Io(std::io::Error::other(error)))?
            .ok_or_else(|| MeasurementJobError::MissingBasis(report.device.clone()))?;
        Ok(EstablishedBasis {
            basis,
            source: report.source,
            elapsed: started.elapsed(),
        })
    }
}

async fn read_bounded(
    reader: impl tokio::io::AsyncRead + Unpin,
    maximum: u64,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.take(maximum).read_to_end(&mut bytes).await?;
    Ok(bytes)
}

async fn retire(child: &mut crate::worker_process::ContainedChild) {
    if let Err(error) = child.start_kill() {
        tracing::error!(%error, "measurement worker kill failed");
        return;
    }
    if tokio::time::timeout(RETIREMENT_DEADLINE, child.wait())
        .await
        .is_err()
    {
        tracing::error!("measurement worker retirement remains unproven");
    }
}

/// Measurement and a resident model never share the device: measurement holds the exclusion
/// exclusively; every loading or resident instance holds it shared.
#[derive(Clone, Default)]
pub struct DeviceExclusion(Arc<tokio::sync::RwLock<()>>);

impl DeviceExclusion {
    pub async fn measurement(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        Arc::clone(&self.0).write_owned().await
    }

    pub async fn residency(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        Arc::clone(&self.0).read_owned().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_arguments_name_the_exact_device_and_directories() {
        let args = MeasurementWorkerArgs {
            device: DeviceSelector::HostCpu,
            basis_directory: PathBuf::from("/cache/bases"),
            kernel_directory: PathBuf::from("/cache/kernels"),
        };
        let mut command = std::process::Command::new("service");
        args.append_to(&mut command);
        assert_eq!(
            command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            [
                "--device".to_owned(),
                DeviceSelector::HostCpu.to_string(),
                "--basis-dir".to_owned(),
                "/cache/bases".to_owned(),
                "--kernel-dir".to_owned(),
                "/cache/kernels".to_owned(),
            ]
        );
    }

    #[test]
    fn the_report_round_trips_the_basis_identity() {
        let report = MeasurementReport {
            engine_build: "engine".to_owned(),
            backend: "metal".to_owned(),
            device: "device".to_owned(),
            protocol_version: 7,
            source: BasisSource::Measured { seconds: 1.5 },
        };
        let decoded: MeasurementReport =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(decoded.identity(), report.identity());
        assert_eq!(decoded.source, BasisSource::Measured { seconds: 1.5 });
    }

    #[tokio::test]
    async fn measurement_waits_for_residency_to_end() {
        let exclusion = DeviceExclusion::default();
        let resident = exclusion.residency().await;
        let waiting = exclusion.clone();
        let measurement = tokio::spawn(async move {
            let _guard = waiting.measurement().await;
        });
        tokio::task::yield_now().await;
        assert!(!measurement.is_finished());
        drop(resident);
        measurement.await.unwrap();
    }
}
