use std::num::NonZeroU32;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use icn_hardware::{
    CacheType, FitContextType, FitFlashAttention, FitOptions, FitRequest, GpuLayers, SplitMode,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "icn-fit",
    about = "Estimate and fit a GGUF using pinned llama.cpp common/fit without loading model tensors"
)]
struct Args {
    /// GGUF model file to inspect.
    #[arg(long)]
    model: PathBuf,
    /// Context tokens, or `auto` for the model's trained context.
    #[arg(long, default_value = "auto")]
    context: ContextArg,
    /// Minimum context during fitting, or `full` to forbid context reduction.
    #[arg(long, default_value = "4096", value_parser = parse_minimum_context)]
    minimum_context: u32,
    /// Margin per device in MiB. One value broadcasts; comma-separated values are per device.
    #[arg(long, value_delimiter = ',', default_value = "1024")]
    margin_mib: Vec<u64>,
    /// Logical prompt batch size.
    #[arg(long, default_value_t = 2_048)]
    batch_size: u32,
    /// Physical prompt micro-batch size.
    #[arg(long, default_value_t = 512)]
    ubatch_size: u32,
    /// Maximum parallel sequences sharing the context.
    #[arg(long, default_value_t = 1)]
    sequences: u32,
    /// GPU layers, or `auto` for common/fit selection.
    #[arg(long, default_value = "auto")]
    gpu_layers: GpuLayersArg,
    /// K-cache type (same values accepted by the pinned llama.cpp CLI).
    #[arg(long, default_value = "f16")]
    cache_type_k: CacheType,
    /// V-cache type (same values accepted by the pinned llama.cpp CLI).
    #[arg(long, default_value = "f16")]
    cache_type_v: CacheType,
    /// Flash Attention planning policy.
    #[arg(long, value_enum, default_value_t = FlashArg::Auto)]
    flash_attention: FlashArg,
    /// Keep KV and K/Q/V operations on the host.
    #[arg(long, default_value_t = false)]
    no_kv_offload: bool,
    /// Disable host-operation offload.
    #[arg(long, default_value_t = false)]
    no_op_offload: bool,
    /// Allocate the full sliding-window cache.
    #[arg(long, default_value_t = false)]
    swa_full: bool,
    /// Use one unified KV cache for all sequences.
    #[arg(long, default_value_t = false)]
    kv_unified: bool,
    /// Emit compact JSON instead of pretty-printed JSON.
    #[arg(long, default_value_t = false)]
    compact: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FlashArg {
    Auto,
    Off,
    On,
}

#[derive(Clone, Copy, Debug)]
enum ContextArg {
    Auto,
    Tokens(NonZeroU32),
}

impl std::str::FromStr for ContextArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") || value.eq_ignore_ascii_case("full") {
            return Ok(Self::Auto);
        }
        value
            .parse::<u32>()
            .map_err(|error| error.to_string())
            .and_then(|value| {
                NonZeroU32::new(value)
                    .map(Self::Tokens)
                    .ok_or_else(|| "context must be greater than zero".to_owned())
            })
    }
}

#[derive(Clone, Copy, Debug)]
enum GpuLayersArg {
    Auto,
    All,
    Count(u32),
}

impl std::str::FromStr for GpuLayersArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if value.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        value
            .parse::<u32>()
            .map(Self::Count)
            .map_err(|error| error.to_string())
    }
}

impl From<FlashArg> for FitFlashAttention {
    fn from(value: FlashArg) -> Self {
        match value {
            FlashArg::Auto => Self::Auto,
            FlashArg::Off => Self::Disabled,
            FlashArg::On => Self::Enabled,
        }
    }
}

#[derive(Serialize)]
struct Output<'a> {
    implementation: &'static str,
    estimator: &'static str,
    allocates_model_tensors: bool,
    model: &'a std::path::Path,
    options: &'a FitOptions,
    report: &'a llama_cpp_2::model::params::fit::FitReport,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Keep stdout machine-readable and avoid thousands of no-allocation graph
    // planning lines on stderr. Library callers retain control of global logs.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let options = FitOptions {
        context_tokens: match args.context {
            ContextArg::Auto => None,
            ContextArg::Tokens(value) => Some(value),
        },
        minimum_context_tokens: args.minimum_context,
        margins_bytes: args
            .margin_mib
            .iter()
            .copied()
            .map(|value| value.saturating_mul(1024 * 1024))
            .collect(),
        batch_tokens: args.batch_size,
        micro_batch_tokens: args.ubatch_size,
        sequence_count: args.sequences,
        gpu_layers: match args.gpu_layers {
            GpuLayersArg::Auto => GpuLayers::Auto,
            GpuLayersArg::All => GpuLayers::All,
            GpuLayersArg::Count(value) => GpuLayers::Count(value),
        },
        split_mode: SplitMode::Layer,
        tensor_split: None,
        use_mmap: true,
        use_mlock: false,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        flash_attention: args.flash_attention.into(),
        offload_kqv: !args.no_kv_offload,
        operation_offload: !args.no_op_offload,
        swa_full: args.swa_full,
        kv_unified: args.kv_unified,
        context_type: FitContextType::Target,
        recurrent_snapshots: 0,
        maximum_outputs: None,
    };
    let request = FitRequest {
        model: args.model,
        options,
    };
    let report = icn_hardware::estimate(&request).context("model fit failed")?;
    let output = Output {
        implementation: "magnitude-icn",
        estimator: "pinned_llama_cpp_common_fit",
        allocates_model_tensors: false,
        model: &request.model,
        options: &request.options,
        report: &report,
    };
    if args.compact {
        serde_json::to_writer(std::io::stdout().lock(), &output)?;
    } else {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &output)?;
    }
    println!();
    if report.is_success() {
        Ok(())
    } else {
        anyhow::bail!("llama.cpp common/fit returned {:?}", report.status)
    }
}

fn parse_minimum_context(value: &str) -> Result<u32, String> {
    if value.eq_ignore_ascii_case("full") {
        return Ok(u32::MAX);
    }
    value
        .parse::<u32>()
        .map_err(|error| error.to_string())
        .and_then(|value| {
            (value > 0)
                .then_some(value)
                .ok_or_else(|| "minimum context must be greater than zero".to_owned())
        })
}
