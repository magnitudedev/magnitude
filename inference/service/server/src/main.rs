use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use magnitude_executor::platform::MemoryReserves;
use magnitude_service_api::{AppState, ServerIdentity, app};
use magnitude_service_contracts::bootstrap_protocol::{IcnStartupRecord, IcnStartupRecordType};
use magnitude_service_models::{
    HuggingFaceDiscovery, InventoryConfig, ManagedModelDownloads, ManagedModelStore, ModelDomainResolver,
    managed_model_services,
};
use magnitude_service_server::assessment::ManagedModelAssessments;
use magnitude_service_server::assessment::assessor::ModelAssessor;
use magnitude_service_server::assessment::environment::AssessmentEnvironment;
use magnitude_service_server::build_identity;
use magnitude_service_server::configurations::ResolvedConfigurations;
use magnitude_service_server::hardware::HardwareInventory;
use magnitude_service_server::residency::controller::{
    MODEL_IDLE_TIMEOUT, ModelInstances, ResidencyEnvironment,
};
use magnitude_service_server::residency::supervisor::SeismicHostMemory;
use magnitude_service_server::residency::worker::run_inference_worker;
use magnitude_service_server::serving::ServiceModels;
use magnitude_service_server::worker_process::{self, WorkerLauncher};
use tower_http::trace::{DefaultOnResponse, TraceLayer};

mod installation;
mod parent_control;
mod startup;
mod telemetry;

use startup::open_installation_catalog;

#[derive(Debug, Parser)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    version,
    about = "Magnitude inference service"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        /// Opaque owner-provided identity echoed by the startup and health protocols.
        #[arg(long, default_value = "standalone")]
        instance_id: String,
        /// Exit when the private owning process closes stdin.
        #[arg(long)]
        exit_on_stdin_eof: bool,
        /// Private owner capability. Prefer the environment-backed form used by managed launch.
        #[arg(long, env = "MAGNITUDE_ICN_AUTH_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
        /// Magnitude-owned model inventory and Hugging Face cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// Magnitude-owned root for all disposable derived cache data, including formed kernels.
        #[arg(long)]
        cache_root: Option<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        /// Verified release or prepared development installation.
        #[arg(long)]
        installation: PathBuf,
    },
    Doctor,
    Version {
        #[arg(long)]
        json: bool,
    },
    /// The contained engine worker: load one model and serve it over standard streams.
    #[command(hide = true)]
    InferenceWorker,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        // The engine worker owns standard output for its protocol: it installs no telemetry
        // subscriber and runs no async runtime of the service's.
        Command::InferenceWorker => {
            worker_process::install_parent_watchdog()?;
            std::process::exit(run_inference_worker());
        }
        command => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("failed to start the service runtime")?
            .block_on(run(command)),
    }
}

async fn run(command: Command) -> anyhow::Result<()> {
    let mut parent_shutdown = if matches!(
        &command,
        Command::Serve {
            exit_on_stdin_eof: true,
            ..
        }
    ) {
        Some(parent_control::install()?)
    } else {
        None
    };
    let _telemetry = telemetry::init(matches!(&command, Command::Serve { .. }))?;
    match command {
        Command::Serve {
            bind,
            instance_id,
            exit_on_stdin_eof: _,
            auth_token,
            model_store,
            cache_root,
            hf_caches,
            installation,
        } => {
            let installation = installation::Installation::load(&installation)
                .context("invalid ICN installation")?;
            let release_catalog = Arc::new(open_installation_catalog(&installation)?);
            let worker_launcher = WorkerLauncher::current()?;
            let reserves = MemoryReserves::standard();
            let catalog = Arc::new(
                tokio::task::spawn_blocking(seismic::DeviceCatalog::discover)
                    .await
                    .context("device discovery task failed")?
                    .context("device discovery failed")?,
            );
            // Every model is assessed on the device a load selects, established once from the
            // discovered devices: selection, host memory and bandwidth.
            let assessment_environment = {
                let catalog = Arc::clone(&catalog);
                Arc::new(
                    tokio::task::spawn_blocking(move || AssessmentEnvironment::establish(&catalog))
                        .await
                        .context("assessment environment task failed")?
                        .context("failed to establish the assessment environment")?,
                )
            };
            let inventory_root = match model_store {
                Some(root) => root,
                None => InventoryConfig::default_root()
                    .context("failed to determine default model store")?,
            };
            let cache_root = match cache_root {
                Some(root) => root,
                None => InventoryConfig::default_cache_root()
                    .context("failed to determine default cache root")?,
            };
            let mut inventory_config = InventoryConfig::with_roots(inventory_root, cache_root)
                .context("invalid model inventory configuration")?;
            inventory_config.hf_cache_dirs.extend(hf_caches);
            inventory_config.catalog_models = release_catalog.catalog().models.clone();
            let inventory = Arc::new(
                ManagedModelStore::open(inventory_config)
                    .await
                    .context("failed to initialize model inventory")?,
            );
            let model_downloads = Arc::new(
                ManagedModelDownloads::open(inventory.clone())
                    .await
                    .context("failed to initialize model downloads")?,
            );
            let native_build = build_identity::native_build();
            let model_variants =
                ModelDomainResolver::new(inventory.clone(), release_catalog.catalog().clone());
            let kernel_directory = inventory.derived_cache().kernel_directory();
            let assessor = Arc::new(ModelAssessor::new(
                inventory.clone(),
                model_variants.clone(),
                release_catalog.clone(),
                assessment_environment,
            ));
            let instances = ModelInstances::start(ResidencyEnvironment {
                models: inventory.clone(),
                model_variants: model_variants.clone(),
                configurations: Arc::new(ResolvedConfigurations::new(
                    inventory.clone(),
                    kernel_directory,
                    reserves,
                )),
                host_memory: Arc::new(SeismicHostMemory::new(catalog.clone())),
                catalog: catalog.clone(),
                reserves,
                idle_timeout: MODEL_IDLE_TIMEOUT,
                launcher: worker_launcher,
                instance_id_namespace: instance_id.clone(),
            });
            let model_services = managed_model_services(
                model_variants,
                model_downloads.clone(),
                Arc::new(instances.clone()),
                Arc::new(instances.clone()),
            )
            .context("failed to initialize model domains")?;
            let assessments = ManagedModelAssessments::start(
                assessor,
                model_services.catalog.clone(),
                model_services.discovered.clone(),
            );
            let mut state = AppState::new()
                .with_model_downloads(model_downloads)
                .with_hugging_face_catalog(Arc::new(HuggingFaceDiscovery::new(inventory.clone())))
                .with_identity(ServerIdentity {
                    instance_id: instance_id.clone(),
                    api_version: 1,
                    native_build: native_build.clone(),
                })
                .with_model_domains(
                    model_services.catalog.clone(),
                    model_services.discovered.clone(),
                    model_services.installations.clone(),
                )
                .with_model_assessments(assessments)
                .with_hardware(Arc::new(HardwareInventory::new(
                    catalog,
                    reserves,
                    Arc::new(instances.clone()),
                )))
                .with_model_controller(Arc::new(instances.clone()));
            if let Some(auth_token) = auth_token {
                state = state.with_authorization(auth_token);
            }
            let serving = magnitude_serving::Serving::new(Arc::new(ServiceModels::new(instances)));
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            let address = listener
                .local_addr()
                .context("failed to read bound address")?;
            let startup = IcnStartupRecord {
                record_type: IcnStartupRecordType::IcnReady,
                protocol_version: 1,
                origin: format!("http://{address}"),
                instance_id: instance_id.clone(),
                pid: std::process::id(),
                api_version: 1,
                native_build,
            };
            println!("MAGNITUDE_ICN_READY {}", serde_json::to_string(&startup)?);
            tracing::info!(
                service.name = telemetry::SERVICE_NAME,
                server.address = %address,
                "inference service ready"
            );
            let app = app(state, serving).layer(
                TraceLayer::new_for_http()
                    .make_span_with(telemetry::http_request_span)
                    .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
            );
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    match parent_shutdown.as_mut() {
                        Some(receiver) => tokio::select! {
                            _ = interrupt_signal() => {},
                            _ = receiver.wait_for(|shutdown| *shutdown) => {},
                        },
                        None => interrupt_signal().await,
                    }
                })
                .await?;
            tracing::info!("inference service stopped");
        }
        Command::Doctor => println!("inference engine loaded successfully"),
        Command::Version { json } => {
            if json {
                println!("{}", serde_json::to_string(&build_identity::identity())?);
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
        Command::InferenceWorker => unreachable!("the inference worker runs without the service runtime"),
    }
    Ok(())
}

#[cfg(unix)]
async fn interrupt_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM handler must install");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = terminate.recv() => {},
    }
}

#[cfg(not(unix))]
async fn interrupt_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_flag_aliases_parse() {
        let aliases = Cli::try_parse_from([
            env!("CARGO_BIN_NAME"),
            "serve",
            "--installation",
            "/tmp/installation.json",
            "--models-dir",
            "/tmp/models",
            "--hf-cache-dir",
            "/tmp/hf",
        ])
        .expect("documented inventory flag aliases should parse");
        let Command::Serve {
            model_store,
            hf_caches,
            ..
        } = aliases.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(model_store, Some(PathBuf::from("/tmp/models")));
        assert_eq!(hf_caches, vec![PathBuf::from("/tmp/hf")]);
    }

    #[test]
    fn managed_parent_pipe_flag_parses() {
        let managed = Cli::try_parse_from([
            env!("CARGO_BIN_NAME"),
            "serve",
            "--installation",
            "/tmp/installation.json",
            "--exit-on-stdin-eof",
        ])
        .expect("managed parent-pipe flag should parse");
        let Command::Serve {
            exit_on_stdin_eof, ..
        } = managed.command
        else {
            panic!("expected serve command")
        };
        assert!(exit_on_stdin_eof);
    }

    #[test]
    fn version_json_reports_native_and_build_provenance() {
        let value = build_identity::identity();
        assert_eq!(value.native_build, build_identity::native_build());
        assert_eq!(value.target, build_identity::TARGET);
        assert_eq!(value.profile, build_identity::PROFILE);
        assert!(!value.backends.is_empty());
    }
}
