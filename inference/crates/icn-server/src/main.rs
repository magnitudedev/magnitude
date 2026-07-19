use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use futures_util::future::BoxFuture;
use icn_api::{AppState, FakeBackend, app};
use icn_contracts::{
    CacheType, CompletionBackend, ComponentRole, ExecutionConfig, FlashAttention, GpuLayers,
    HardwareAssessment, InventoryError, InventoryHardwareAssessor, LoadStage, ModelId,
    ModelInventory, ModelStatus, ProjectorConfig, RadixCacheConfig, ResolvedExecutionPlan,
    ResolvedModel, SplitMode,
};
use icn_engine::{LlamaCompletionBackend, resolve_execution_plan};
use icn_hardware::{CapacityPolicy, assess as assess_hardware, assess_with_backend};
use icn_models::{InventoryConfig, ModelManager};
use icn_reasoning::NativeTemplateAssessor;

mod build_identity;

#[derive(Debug, Parser)]
#[command(
    name = "magnitude-icn",
    version,
    about = "Magnitude inference control node"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FlashAttentionArg {
    Auto,
    Off,
    On,
}

#[derive(Debug, Subcommand)]
// Clap's flat `serve` command intentionally keeps its complete execution profile visible in
// `--help`; boxing individual flags would only optimize the one-time CLI parse allocation.
#[allow(clippy::large_enum_variant)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        #[arg(long, conflicts_with_all = ["model", "model_id"])]
        fake: bool,
        #[arg(long, conflicts_with = "model_id")]
        model: Option<PathBuf>,
        /// Load a ready inventory model by its stable ID.
        #[arg(long, conflicts_with = "model")]
        model_id: Option<String>,
        /// Multimodal projector GGUF paired with the loaded text model.
        #[arg(long, requires = "model")]
        mmproj: Option<PathBuf>,
        /// Explicit separate MTP GGUF. Bundled MTP is selected natively before this override.
        #[arg(long, requires = "model")]
        mtp_model: Option<PathBuf>,
        #[arg(long, requires = "mmproj")]
        no_mmproj_offload: bool,
        #[arg(long, requires = "mmproj")]
        no_mmproj_warmup: bool,
        #[arg(long, requires = "mmproj")]
        image_min_tokens: Option<NonZeroU32>,
        #[arg(long, requires = "mmproj")]
        image_max_tokens: Option<NonZeroU32>,
        #[arg(long)]
        model_alias: Option<String>,
        /// Magnitude-owned model inventory and Hugging Face cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// One-time read-only import source for the TypeScript v1 managed store.
        #[arg(long)]
        legacy_store: Option<PathBuf>,
        /// Additional read-only directories containing GGUF models.
        #[arg(long = "model-source")]
        model_sources: Vec<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        #[arg(long, default_value_t = 4096)]
        context_size: u32,
        #[arg(long, default_value_t = 512)]
        batch_size: u32,
        #[arg(long, default_value_t = 512)]
        ubatch_size: u32,
        #[arg(long, default_value_t = 1)]
        max_sequences: u32,
        #[arg(long)]
        prefill_quantum: Option<u32>,
        /// GPU layers: `auto` runs pinned common/fit, `all` fully offloads, or use a count.
        #[arg(long, default_value = "auto")]
        gpu_layers: GpuLayers,
        /// Disable model memory mapping (enabled by default, matching llama-server).
        #[arg(long)]
        no_mmap: bool,
        /// Keep mapped model pages resident in RAM.
        #[arg(long)]
        mlock: bool,
        #[arg(long, default_value = "layer")]
        split_mode: SplitMode,
        /// Comma-separated per-device model placement proportions.
        #[arg(long)]
        tensor_split: Option<TensorSplitArg>,
        #[arg(long, default_value = "f16")]
        cache_type_k: CacheType,
        #[arg(long, default_value = "f16")]
        cache_type_v: CacheType,
        #[arg(long)]
        no_kv_offload: bool,
        #[arg(long)]
        no_op_offload: bool,
        #[arg(long)]
        swa_full: bool,
        #[arg(long)]
        kv_unified: bool,
        /// Disable automatic radix-prefix KV reuse even when unified KV supports native pages.
        #[arg(long)]
        no_radix_cache: bool,
        /// Strict lazy-allocation budget for serialized KV pages retained in system RAM.
        #[arg(long, default_value_t = 0)]
        radix_cache_host_bytes: u64,
        /// Strict budget for serialized KV pages retained on local storage.
        #[arg(long, default_value_t = 0)]
        radix_cache_disk_bytes: u64,
        /// Storage directory for the disk KV tier (required when its budget is non-zero).
        #[arg(long)]
        radix_cache_dir: Option<PathBuf>,
        #[arg(long)]
        threads: Option<NonZeroU32>,
        #[arg(long)]
        threads_batch: Option<NonZeroU32>,
        #[arg(long, value_enum, default_value_t = FlashAttentionArg::Auto)]
        flash_attention: FlashAttentionArg,
    },
    Doctor,
    Version {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone)]
struct TensorSplitArg(Vec<f32>);

#[derive(Debug, Clone, serde::Serialize)]
struct RuntimePlanDefaults {
    context_size: u32,
    batch_size: u32,
    ubatch_size: u32,
    max_sequences: u32,
    prefill_quantum: u32,
    radix_cache: RadixCacheConfig,
    execution: ExecutionConfig,
    projector_use_gpu: bool,
    projector_warmup: bool,
    image_min_tokens: Option<NonZeroU32>,
    image_max_tokens: Option<NonZeroU32>,
}

#[derive(Debug)]
enum MtpSelection {
    Automatic(Vec<PathBuf>),
    Explicit(PathBuf),
}

fn resolved_plan(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    mtp_selection: MtpSelection,
    defaults: &RuntimePlanDefaults,
) -> anyhow::Result<ResolvedExecutionPlan> {
    let mut plan = base_plan(model_path, projector_path, defaults)?;
    let candidates = match &mtp_selection {
        MtpSelection::Automatic(paths) => icn_mtp::CandidatePolicy::Automatic(paths),
        MtpSelection::Explicit(path) => icn_mtp::CandidatePolicy::Explicit(path),
    };
    plan.mtp = icn_mtp::select_mtp(&plan, candidates)
        .context("failed to select a native MTP configuration")?;
    resolve_execution_plan(plan).context("failed to resolve MTP execution defaults")
}

fn base_plan(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    defaults: &RuntimePlanDefaults,
) -> anyhow::Result<ResolvedExecutionPlan> {
    resolve_execution_plan(ResolvedExecutionPlan {
        model_path,
        context_size: defaults.context_size,
        batch_size: defaults.batch_size,
        ubatch_size: defaults.ubatch_size,
        max_sequences: defaults.max_sequences,
        prefill_quantum: defaults.prefill_quantum,
        radix_cache: defaults.radix_cache.clone(),
        execution: defaults.execution.clone(),
        projector: projector_path.map(|path| {
            let mut projector = ProjectorConfig::new(path);
            projector.use_gpu = defaults.projector_use_gpu;
            projector.warmup = defaults.projector_warmup;
            projector.image_min_tokens = defaults.image_min_tokens;
            projector.image_max_tokens = defaults.image_max_tokens;
            projector
        }),
        mtp: icn_contracts::MtpConfig::default(),
    })
    .context("failed to resolve execution defaults")
}

struct NativeHardwareAssessor {
    defaults: RuntimePlanDefaults,
    native_executor: Arc<RwLock<Option<Arc<LlamaCompletionBackend>>>>,
    gate: tokio::sync::Mutex<()>,
}

impl InventoryHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(&self) -> BoxFuture<'_, Result<String, InventoryError>> {
        Box::pin(async move {
            let defaults = serde_json::to_string(&self.defaults)
                .map_err(|error| InventoryError::Internal(error.to_string()))?;
            let parallelism = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1);
            let topology = stable_hardware_topology();
            Ok(format!(
                "inventory-hardware-v1:{}:{}:{}:{parallelism}:{topology}:{defaults}",
                build_identity::BINDINGS_REVISION,
                build_identity::LLAMA_CPP_REVISION,
                build_identity::enabled_backends().join(","),
            ))
        })
    }

    fn assess(
        &self,
        resolved: ResolvedModel,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(async move {
            let _guard = self.gate.lock().await;
            let id = resolved.model.id.clone();
            let primary = resolved
                .components
                .iter()
                .filter(|component| {
                    matches!(
                        component.role,
                        ComponentRole::Weights | ComponentRole::Shard
                    )
                })
                .min_by_key(|component| component.shard_index.unwrap_or(0))
                .map(|component| component.path.clone())
                .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".into()))?;
            let projector = resolved
                .components
                .iter()
                .find(|component| component.role == ComponentRole::Projector)
                .map(|component| component.path.clone());
            let mtp: Vec<PathBuf> = resolved
                .components
                .iter()
                .filter(|component| {
                    matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft)
                })
                .map(|component| component.path.clone())
                .collect();
            let native_executor = self
                .native_executor
                .read()
                .map_err(|_| InventoryError::Internal("native executor lock poisoned".to_owned()))?
                .clone();
            let defaults = self.defaults.clone();
            match tokio::task::spawn_blocking(move || match native_executor {
                Some(executor) => executor
                    .run_exclusive_native(move |backend| {
                        let mut plan = base_plan(primary, projector, &defaults)?;
                        plan.mtp = icn_mtp::select_mtp_with_backend(
                            backend,
                            &plan,
                            icn_mtp::CandidatePolicy::Automatic(&mtp),
                        )
                        .context("failed to select a native MTP configuration")?;
                        let plan = resolve_execution_plan(plan)
                            .context("failed to resolve MTP execution defaults")?;
                        assess_with_backend(backend, &plan, CapacityPolicy::default())
                            .map(|value| value.assessment)
                            .map_err(anyhow::Error::from)
                    })
                    .map_err(|error| anyhow::anyhow!(error))?,
                None => {
                    let plan =
                        resolved_plan(primary, projector, MtpSelection::Automatic(mtp), &defaults)?;
                    assess_hardware(&plan, CapacityPolicy::default())
                        .map(|value| value.assessment)
                        .map_err(anyhow::Error::from)
                }
            })
            .await
            {
                Ok(Ok(assessment)) => Ok(assessment),
                Ok(Err(error)) => Err(InventoryError::Internal(format!(
                    "hardware assessment failed for {}: {error:#}",
                    id.0
                ))),
                Err(error) => Err(InventoryError::Internal(format!(
                    "hardware assessment task failed for {}: {error}",
                    id.0
                ))),
            }
        })
    }
}

fn stable_hardware_topology() -> String {
    #[cfg(target_os = "macos")]
    {
        let mut values = Vec::new();
        for key in ["hw.model", "hw.memsize"] {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", key])
                .output()
                && output.status.success()
            {
                values.push(String::from_utf8_lossy(&output.stdout).trim().to_owned());
            }
        }
        if !values.is_empty() {
            return values.join(":");
        }
    }
    #[cfg(target_os = "linux")]
    {
        let mut values = Vec::new();
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(line) = contents.lines().find(|line| line.starts_with("MemTotal:")) {
                values.push(line.to_owned());
            }
        }
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            && output.status.success()
        {
            values.push(String::from_utf8_lossy(&output.stdout).trim().to_owned());
        }
        if !values.is_empty() {
            return values.join(":");
        }
    }
    "unknown".to_owned()
}

impl std::str::FromStr for TensorSplitArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() {
            return Err("tensor split must contain at least one weight".to_owned());
        }
        let weights = value
            .split([',', '/'])
            .enumerate()
            .map(|(index, weight)| {
                weight.parse::<f32>().map_err(|_| {
                    format!("tensor split weight {index} is not a valid number: {weight:?}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some((index, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "tensor split weight {index} must be finite and non-negative, received {weight}"
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err("tensor split must assign a positive weight to at least one device".into());
        }
        Ok(Self(weights))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Serve {
            bind,
            fake,
            model,
            model_id,
            mmproj,
            mtp_model,
            no_mmproj_offload,
            no_mmproj_warmup,
            image_min_tokens,
            image_max_tokens,
            model_alias,
            model_store,
            legacy_store: configured_legacy_store,
            model_sources,
            hf_caches,
            context_size,
            batch_size,
            ubatch_size,
            max_sequences,
            prefill_quantum,
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            cache_type_v,
            no_kv_offload,
            no_op_offload,
            swa_full,
            kv_unified,
            no_radix_cache,
            radix_cache_host_bytes,
            radix_cache_disk_bytes,
            radix_cache_dir,
            threads,
            threads_batch,
            flash_attention,
        } => {
            let (inventory_root, legacy_store) = match model_store {
                Some(root) => (root, configured_legacy_store),
                None => {
                    let root = InventoryConfig::default_root()
                        .context("failed to determine default model store")?;
                    let legacy = root
                        .parent()
                        .map(|parent| parent.join("local-inference/huggingface"));
                    (root, configured_legacy_store.or(legacy))
                }
            };
            let mut inventory_config = InventoryConfig::with_root(inventory_root)
                .context("invalid model inventory configuration")?;
            inventory_config.legacy_store = legacy_store;
            inventory_config.model_sources.extend(model_sources);
            inventory_config.hf_cache_dirs.extend(hf_caches);
            let plan_defaults = RuntimePlanDefaults {
                context_size,
                batch_size,
                ubatch_size,
                max_sequences,
                prefill_quantum: prefill_quantum.unwrap_or(batch_size),
                radix_cache: RadixCacheConfig {
                    enabled: !no_radix_cache,
                    page_tokens: NonZeroU32::new(16).expect("16 is non-zero"),
                    host_bytes: radix_cache_host_bytes,
                    disk_bytes: radix_cache_disk_bytes,
                    disk_path: radix_cache_dir,
                },
                execution: ExecutionConfig {
                    gpu_layers,
                    use_mmap: !no_mmap,
                    use_mlock: mlock,
                    split_mode,
                    tensor_split: tensor_split.map(|value| value.0),
                    cache_type_k,
                    cache_type_v,
                    offload_kqv: !no_kv_offload,
                    operation_offload: !no_op_offload,
                    swa_full,
                    kv_unified,
                    threads,
                    threads_batch,
                    flash_attention: match flash_attention {
                        FlashAttentionArg::Auto => FlashAttention::Auto,
                        FlashAttentionArg::Off => FlashAttention::Disabled,
                        FlashAttentionArg::On => FlashAttention::Enabled,
                    },
                },
                projector_use_gpu: !no_mmproj_offload,
                projector_warmup: !no_mmproj_warmup,
                image_min_tokens,
                image_max_tokens,
            };
            let inventory = Arc::new(
                ModelManager::open_with_template_assessor(
                    inventory_config,
                    Some(Arc::new(NativeTemplateAssessor)),
                )
                .await
                .context("failed to initialize model inventory")?,
            );
            let native_executor_slot = Arc::new(RwLock::new(None));
            let inventory_hardware_assessor = Arc::new(NativeHardwareAssessor {
                defaults: plan_defaults.clone(),
                native_executor: Arc::clone(&native_executor_slot),
                gate: tokio::sync::Mutex::new(()),
            });
            inventory
                .set_hardware_assessor(inventory_hardware_assessor)
                .context("failed to configure inventory hardware assessment")?;
            let (model, mmproj, mtp_model, model_alias, selected_inventory_id) = if let Some(
                raw_id,
            ) = model_id
            {
                if mmproj.is_some() || mtp_model.is_some() {
                    anyhow::bail!(
                        "--model-id resolves projector and MTP components from inventory; explicit --mmproj/--mtp-model overrides are not allowed"
                    );
                }
                let id = ModelId::parse(raw_id).context("invalid inventory model ID")?;
                inventory
                    .ensure_model_inventory()
                    .await
                    .context("failed to reconcile inventory for model selection")?;
                let resolved = inventory
                    .resolve_ready(&id)
                    .await
                    .context("failed to resolve inventory model")?;
                let primary = resolved
                    .components
                    .iter()
                    .filter(|component| {
                        matches!(
                            component.role,
                            ComponentRole::Weights | ComponentRole::Shard
                        )
                    })
                    .min_by_key(|component| component.shard_index.unwrap_or(0))
                    .map(|component| component.path.clone())
                    .context("inventory model has no runnable weight component")?;
                let projector = resolved
                    .components
                    .iter()
                    .find(|component| component.role == ComponentRole::Projector)
                    .map(|component| component.path.clone());
                let mtp = resolved
                    .components
                    .iter()
                    .filter(|component| {
                        matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft)
                    })
                    .map(|component| component.path.clone())
                    .collect();
                (
                    Some(primary),
                    projector,
                    MtpSelection::Automatic(mtp),
                    model_alias.or(Some(resolved.model.name)),
                    Some(id),
                )
            } else {
                (
                    model,
                    mmproj,
                    mtp_model.map_or_else(
                        || MtpSelection::Automatic(Vec::new()),
                        MtpSelection::Explicit,
                    ),
                    model_alias,
                    None,
                )
            };
            let (state, native_executor) = if fake || model.is_none() {
                (
                    AppState::new(FakeBackend::new(
                        model_alias.unwrap_or_else(|| "icn-fake".into()),
                        "Hello from ICN.",
                    )),
                    None,
                )
            } else {
                let path = model.expect("model is present");
                let alias = model_alias.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("local-model")
                        .to_owned()
                });
                let inventory_id = match selected_inventory_id {
                    Some(id) => id,
                    None => inventory
                        .register_active_model(&path, Some(&alias))
                        .await
                        .context("failed to register the active model")?,
                };
                let requested_plan = resolved_plan(path, mmproj, mtp_model, &plan_defaults)?;
                let assessed = assess_hardware(&requested_plan, CapacityPolicy::default())
                    .context("failed to assess the resolved execution plan")?;
                if let icn_contracts::HardwareAssessment::DoesNotFit { memory, .. } =
                    &assessed.assessment
                {
                    anyhow::bail!(
                        "resolved execution plan requires {} bytes but stable capacity is {} bytes",
                        memory.required_bytes,
                        memory.available_bytes
                    );
                }
                let load_id = format!("load-{}", unix_timestamp());
                inventory
                    .update_status(
                        &inventory_id,
                        ModelStatus::Loading {
                            load_id,
                            stage: LoadStage::Opening,
                            started_at: unix_timestamp(),
                        },
                    )
                    .await
                    .context("failed to project model loading state")?;
                let backend = match LlamaCompletionBackend::load(
                    inventory_id.0.clone(),
                    assessed.plan.clone(),
                ) {
                    Ok(backend) => backend,
                    Err(error) => {
                        let _ = inventory
                            .update_status(
                                &inventory_id,
                                ModelStatus::LoadFailed {
                                    attempted_at: unix_timestamp(),
                                    stage: LoadStage::Opening,
                                    code: "backend_load_failed".to_owned(),
                                    retryable: true,
                                },
                            )
                            .await;
                        return Err(error).context("failed to load llama.cpp backend");
                    }
                };
                let properties = backend
                    .properties()
                    .context("failed to read loaded model properties")?;
                let mut execution: std::collections::BTreeMap<_, _> =
                    serde_json::to_value(&properties.execution.resolved)
                        .ok()
                        .and_then(|value| value.as_object().cloned())
                        .unwrap_or_default()
                        .into_iter()
                        .collect();
                if let Ok(radix_cache) = serde_json::to_value(&assessed.plan.radix_cache) {
                    execution.insert("radix_cache".to_owned(), radix_cache);
                }
                inventory
                    .update_status(
                        &inventory_id,
                        ModelStatus::Loaded {
                            loaded_at: unix_timestamp(),
                            backend: "llama_cpp".to_owned(),
                            context_length: properties.context_tokens,
                            execution,
                        },
                    )
                    .await
                    .context("failed to project loaded model state")?;
                let backend = Arc::new(backend);
                (
                    AppState::from_shared_backend(backend.clone()).with_model_alias(alias),
                    Some(backend),
                )
            };
            *native_executor_slot
                .write()
                .map_err(|_| anyhow::anyhow!("native executor lock poisoned"))? = native_executor;
            let state = state.with_inventory(inventory);
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            println!("magnitude-icn listening on http://{bind}");
            axum::serve(listener, app(state))
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        Command::Doctor => println!("ICN runtime and native backend loaded successfully"),
        Command::Version { json } => {
            if json {
                println!("{}", build_identity::json());
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_cli_defaults_and_explicit_values_are_typed() {
        let defaults = Cli::try_parse_from(["magnitude-icn", "serve", "--fake"]).unwrap();
        let Command::Serve {
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            cache_type_v,
            no_kv_offload,
            no_op_offload,
            swa_full,
            kv_unified,
            no_radix_cache,
            radix_cache_host_bytes,
            radix_cache_disk_bytes,
            radix_cache_dir,
            threads,
            threads_batch,
            ..
        } = defaults.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(gpu_layers, GpuLayers::Auto);
        assert!(!no_mmap && !mlock);
        assert_eq!(split_mode, SplitMode::Layer);
        assert!(tensor_split.is_none());
        assert_eq!(cache_type_k, CacheType::F16);
        assert_eq!(cache_type_v, CacheType::F16);
        assert!(!no_kv_offload && !no_op_offload && !swa_full && !kv_unified);
        assert!(!no_radix_cache);
        assert_eq!(radix_cache_host_bytes, 0);
        assert_eq!(radix_cache_disk_bytes, 0);
        assert!(radix_cache_dir.is_none());
        assert!(threads.is_none() && threads_batch.is_none());

        let explicit = Cli::try_parse_from([
            "magnitude-icn",
            "serve",
            "--fake",
            "--gpu-layers",
            "all",
            "--no-mmap",
            "--mlock",
            "--split-mode",
            "row",
            "--tensor-split",
            "3,1",
            "--cache-type-k",
            "q8_0",
            "--threads",
            "6",
            "--threads-batch",
            "8",
            "--radix-cache-host-bytes",
            "50000000000",
            "--radix-cache-disk-bytes",
            "100000000000",
            "--radix-cache-dir",
            "/tmp/icn-cache",
        ])
        .unwrap();
        let Command::Serve {
            gpu_layers,
            no_mmap,
            mlock,
            split_mode,
            tensor_split,
            cache_type_k,
            threads,
            threads_batch,
            radix_cache_host_bytes,
            radix_cache_disk_bytes,
            radix_cache_dir,
            ..
        } = explicit.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(gpu_layers, GpuLayers::All);
        assert!(no_mmap && mlock);
        assert_eq!(split_mode, SplitMode::Row);
        assert_eq!(tensor_split.unwrap().0, vec![3.0, 1.0]);
        assert_eq!(cache_type_k, CacheType::Q8_0);
        assert_eq!(threads, NonZeroU32::new(6));
        assert_eq!(threads_batch, NonZeroU32::new(8));
        assert_eq!(radix_cache_host_bytes, 50_000_000_000);
        assert_eq!(radix_cache_disk_bytes, 100_000_000_000);
        assert_eq!(radix_cache_dir, Some(PathBuf::from("/tmp/icn-cache")));
    }

    #[test]
    fn tensor_split_cli_rejects_unsafe_weights() {
        assert!("0,0".parse::<TensorSplitArg>().is_err());
        assert!("1,-1".parse::<TensorSplitArg>().is_err());
        assert!("NaN,1".parse::<TensorSplitArg>().is_err());
    }

    #[test]
    fn inventory_model_id_is_mutually_exclusive_with_paths_and_fake_mode() {
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef",
                "--model",
                "/tmp/model.gguf"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "magnitude-icn",
                "serve",
                "--model-id",
                "mdl_0123456789abcdef",
                "--fake"
            ])
            .is_err()
        );

        let aliases = Cli::try_parse_from([
            "magnitude-icn",
            "serve",
            "--fake",
            "--models-dir",
            "/tmp/models",
            "--legacy-store",
            "/tmp/legacy",
            "--hf-cache-dir",
            "/tmp/hf",
        ])
        .expect("documented inventory flag aliases should parse");
        let Command::Serve {
            model_store,
            legacy_store,
            hf_caches,
            ..
        } = aliases.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(model_store, Some(PathBuf::from("/tmp/models")));
        assert_eq!(legacy_store, Some(PathBuf::from("/tmp/legacy")));
        assert_eq!(hf_caches, vec![PathBuf::from("/tmp/hf")]);
    }

    #[test]
    fn version_json_reports_native_and_build_provenance() {
        let value = build_identity::json();
        assert_eq!(
            value["bindings_revision"],
            build_identity::BINDINGS_REVISION
        );
        assert_eq!(
            value["llama_cpp_revision"],
            build_identity::LLAMA_CPP_REVISION
        );
        assert_eq!(value["target"], build_identity::TARGET);
        assert_eq!(value["profile"], build_identity::PROFILE);
        assert!(
            value["backends"]
                .as_array()
                .is_some_and(|values| !values.is_empty())
        );
    }
}
