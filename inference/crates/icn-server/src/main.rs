use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use icn_api::{AppState, FakeBackend, app};
use icn_core::{
    CacheType, ExecutionConfig, FlashAttention, GpuLayers, ModelConfig, ProjectorConfig, SplitMode,
};
use icn_llamacpp::LlamaCompletionBackend;

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
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        #[arg(long, conflicts_with = "model")]
        fake: bool,
        #[arg(long)]
        model: Option<PathBuf>,
        /// Multimodal projector GGUF paired with the loaded text model.
        #[arg(long, requires = "model")]
        mmproj: Option<PathBuf>,
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
            mmproj,
            no_mmproj_offload,
            no_mmproj_warmup,
            image_min_tokens,
            image_max_tokens,
            model_alias,
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
            threads,
            threads_batch,
            flash_attention,
        } => {
            let state = if fake || model.is_none() {
                AppState::new(FakeBackend::new(
                    model_alias.unwrap_or_else(|| "icn-fake".into()),
                    "Hello from ICN.",
                ))
            } else {
                let path = model.expect("model is present");
                let alias = model_alias.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("local-model")
                        .to_owned()
                });
                let backend = LlamaCompletionBackend::load(
                    alias,
                    ModelConfig {
                        model_path: path,
                        context_size,
                        batch_size,
                        ubatch_size,
                        max_sequences,
                        prefill_quantum: prefill_quantum.unwrap_or(batch_size),
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
                        projector: mmproj.map(|path| {
                            let mut projector = ProjectorConfig::new(path);
                            projector.use_gpu = !no_mmproj_offload;
                            projector.warmup = !no_mmproj_warmup;
                            projector.image_min_tokens = image_min_tokens;
                            projector.image_max_tokens = image_max_tokens;
                            projector
                        }),
                    },
                )
                .context("failed to load llama.cpp backend")?;
                AppState::new(backend)
            };
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
    }

    #[test]
    fn tensor_split_cli_rejects_unsafe_weights() {
        assert!("0,0".parse::<TensorSplitArg>().is_err());
        assert!("1,-1".parse::<TensorSplitArg>().is_err());
        assert!("NaN,1".parse::<TensorSplitArg>().is_err());
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
