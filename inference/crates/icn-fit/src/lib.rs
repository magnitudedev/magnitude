//! Model-free memory fitting over the exact pinned llama.cpp `common/fit` path.

use std::ffi::{CString, NulError};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

pub use icn_core::{CacheType, FlashAttention as FitFlashAttention};
use llama_cpp_2::context::params::{FlashAttentionPolicy, KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::fit::{FitReport, FitReportError};

fn cache_type_into_native(cache_type: CacheType) -> KvCacheType {
    match cache_type {
        CacheType::F32 => KvCacheType::F32,
        CacheType::F16 => KvCacheType::F16,
        CacheType::Bf16 => KvCacheType::BF16,
        CacheType::Q8_0 => KvCacheType::Q8_0,
        CacheType::Q4_0 => KvCacheType::Q4_0,
        CacheType::Q4_1 => KvCacheType::Q4_1,
        CacheType::Iq4Nl => KvCacheType::IQ4_NL,
        CacheType::Q5_0 => KvCacheType::Q5_0,
        CacheType::Q5_1 => KvCacheType::Q5_1,
    }
}

fn flash_attention_into_native(policy: FitFlashAttention) -> FlashAttentionPolicy {
    match policy {
        FitFlashAttention::Auto => FlashAttentionPolicy::Auto,
        FitFlashAttention::Disabled => FlashAttentionPolicy::Disabled,
        FitFlashAttention::Enabled => FlashAttentionPolicy::Enabled,
    }
}

/// Inputs that affect llama.cpp's model, context, and compute estimates.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FitOptions {
    /// Context length. `None` requests the model's trained context length.
    pub context_tokens: Option<NonZeroU32>,
    /// Minimum context length allowed when fitting. `u32::MAX` preserves the full context.
    pub minimum_context_tokens: u32,
    /// One margin to broadcast or one value per `llama_max_devices()`, in bytes.
    pub margins_bytes: Vec<u64>,
    /// Logical prompt batch size.
    pub batch_tokens: u32,
    /// Physical prompt micro-batch size.
    pub micro_batch_tokens: u32,
    /// Maximum parallel sequences sharing the context.
    pub sequence_count: u32,
    /// `None` leaves GPU layers in auto mode; `Some` pins an explicit count.
    pub gpu_layers: Option<u32>,
    /// K-cache data type.
    pub cache_type_k: CacheType,
    /// V-cache data type.
    pub cache_type_v: CacheType,
    /// Flash Attention policy.
    pub flash_attention: FitFlashAttention,
    /// Whether K/Q/V operations and KV memory may be offloaded.
    pub offload_kqv: bool,
    /// Whether host tensor operations may be offloaded.
    pub operation_offload: bool,
    /// Whether to allocate the full sliding-window cache.
    pub swa_full: bool,
    /// Whether sequences share a unified KV cache.
    pub kv_unified: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            context_tokens: None,
            minimum_context_tokens: 4_096,
            margins_bytes: vec![1024 * 1024 * 1024],
            batch_tokens: 2_048,
            micro_batch_tokens: 512,
            sequence_count: 1,
            gpu_layers: None,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
            flash_attention: FitFlashAttention::Auto,
            offload_kqv: true,
            operation_offload: true,
            swa_full: false,
            kv_unified: false,
        }
    }
}

/// Request for a no-allocation model fit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FitRequest {
    /// GGUF file to inspect.
    pub model: PathBuf,
    /// Planning parameters.
    pub options: FitOptions,
}

/// Validation, backend, or native bridge failure.
#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    /// The model path is not a regular file.
    #[error("model does not exist or is not a file: {0}")]
    InvalidModel(PathBuf),
    /// The model path contains an interior NUL byte.
    #[error("model path contains an interior NUL byte: {0}")]
    ModelPathNul(#[from] NulError),
    /// Invalid fit option.
    #[error("invalid fit options: {0}")]
    InvalidOptions(String),
    /// llama.cpp backend initialization failed.
    #[error("failed to initialize llama.cpp: {0}")]
    Backend(#[source] llama_cpp_2::LlamaCppError),
    /// Structured fit bridge failed.
    #[error(transparent)]
    Fit(#[from] FitReportError),
}

/// Initialize llama.cpp and estimate a model without allocating its tensor data.
///
/// # Errors
///
/// Returns [`EstimateError`] if validation, backend initialization, or native
/// diagnostics fail. A native fit `Failure`/`Error` is still returned as a
/// typed [`FitReport`] so callers can inspect its diagnostics.
pub fn estimate(request: &FitRequest) -> Result<FitReport, EstimateError> {
    let backend = LlamaBackend::init().map_err(EstimateError::Backend)?;
    estimate_with_backend(&backend, request)
}

/// Estimate a model using an already initialized llama.cpp backend.
///
/// The backend reference is an explicit lifetime proof for callers such as ICN;
/// the pinned C fitting function itself uses global backend registration.
///
/// # Errors
///
/// Returns [`EstimateError`] for invalid options or bridge/report failures.
pub fn estimate_with_backend(
    _backend: &LlamaBackend,
    request: &FitRequest,
) -> Result<FitReport, EstimateError> {
    validate(request)?;
    let model_path = path_c_string(&request.model)?;
    let max_devices = llama_cpp_2::max_devices();
    let mut margins = expand_margins(&request.options.margins_bytes, max_devices)?;

    let model_params = match request.options.gpu_layers {
        Some(layers) => LlamaModelParams::default().with_n_gpu_layers(layers),
        None => LlamaModelParams::default(),
    };
    let mut model_params = std::pin::pin!(model_params);
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(request.options.context_tokens)
        .with_n_batch(request.options.batch_tokens)
        .with_n_ubatch(request.options.micro_batch_tokens)
        .with_n_seq_max(request.options.sequence_count)
        .with_type_k(cache_type_into_native(request.options.cache_type_k))
        .with_type_v(cache_type_into_native(request.options.cache_type_v))
        .with_flash_attention(flash_attention_into_native(request.options.flash_attention))
        .with_offload_kqv(request.options.offload_kqv)
        .with_op_offload(request.options.operation_offload)
        .with_swa_full(request.options.swa_full)
        .with_kv_unified(request.options.kv_unified);

    model_params
        .as_mut()
        .fit_params_report(
            &model_path,
            &mut context_params,
            &mut margins,
            request.options.minimum_context_tokens,
        )
        .map_err(EstimateError::Fit)
}

fn validate(request: &FitRequest) -> Result<(), EstimateError> {
    if !request.model.is_file() {
        return Err(EstimateError::InvalidModel(request.model.clone()));
    }
    let options = &request.options;
    if options.minimum_context_tokens == 0 {
        return Err(EstimateError::InvalidOptions(
            "minimum context must be greater than zero".to_owned(),
        ));
    }
    if options.batch_tokens == 0 || options.micro_batch_tokens == 0 {
        return Err(EstimateError::InvalidOptions(
            "batch and micro-batch sizes must be greater than zero".to_owned(),
        ));
    }
    if options.micro_batch_tokens > options.batch_tokens {
        return Err(EstimateError::InvalidOptions(
            "micro-batch size must not exceed batch size".to_owned(),
        ));
    }
    if options.sequence_count == 0 {
        return Err(EstimateError::InvalidOptions(
            "sequence count must be greater than zero".to_owned(),
        ));
    }
    if options.margins_bytes.is_empty() {
        return Err(EstimateError::InvalidOptions(
            "at least one memory margin is required".to_owned(),
        ));
    }
    Ok(())
}

fn expand_margins(values: &[u64], count: usize) -> Result<Vec<usize>, EstimateError> {
    if values.len() != 1 && values.len() != count {
        return Err(EstimateError::InvalidOptions(format!(
            "provide one memory margin to broadcast or exactly {count}; received {}",
            values.len()
        )));
    }
    let convert = |value: u64| {
        usize::try_from(value).map_err(|_| {
            EstimateError::InvalidOptions(format!(
                "memory margin {value} does not fit this target's usize"
            ))
        })
    };
    if values.len() == 1 {
        return Ok(vec![convert(values[0])?; count]);
    }
    values.iter().copied().map(convert).collect()
}

fn path_c_string(path: &Path) -> Result<CString, EstimateError> {
    Ok(CString::new(path.to_string_lossy().as_bytes())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_margin_is_broadcast() {
        assert_eq!(expand_margins(&[512], 3).expect("margins"), vec![512; 3]);
    }

    #[test]
    fn per_device_margin_count_is_exact() {
        assert!(expand_margins(&[1, 2], 3).is_err());
        assert_eq!(
            expand_margins(&[1, 2, 3], 3).expect("margins"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn cache_types_match_upstream_spelling() {
        assert_eq!("iq4_nl".parse::<CacheType>(), Ok(CacheType::Iq4Nl));
        assert!("q6_k".parse::<CacheType>().is_err());
    }
}
