//! Native MTP discovery, compatibility validation, and automatic selection policy.
//!
//! This crate owns ICN's MTP policy. Architecture-specific capability remains in the pinned
//! llama.cpp implementation, and all native access goes through safe `llama-cpp-2` APIs.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use icn_contracts::{
    CacheType, ExecutionConfig, FlashAttention, GpuLayers, MtpConfig, MtpSource,
    ResolvedExecutionPlan, SplitMode,
};
use llama_cpp_2::context::params::{FlashAttentionPolicy, KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::{LlamaGpuLayers, LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::mtp::{
    MtpModelKind, MtpPreflightError, MtpPreflightParams, inspect_mtp_model, preflight_mtp,
};

/// Pinned llama.cpp's default speculative policy. Capability selection is exact; these values are
/// runtime tuning policy and can be calibrated independently.
const DEFAULT_N_MAX: u32 = 3;
const DEFAULT_N_MIN: u32 = 0;
const DEFAULT_P_MIN: f32 = 0.0;

/// How separately packaged MTP artifacts are supplied to automatic selection.
#[derive(Clone, Copy, Debug)]
pub enum CandidatePolicy<'a> {
    /// Consider every locally discovered candidate and enable only a unique compatible artifact.
    Automatic(&'a [PathBuf]),
    /// Require this exact user-selected artifact to be compatible.
    Explicit(&'a Path),
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("failed to initialize llama.cpp for MTP inspection: {0}")]
    Backend(String),
    #[error("invalid native execution parameters: {0}")]
    InvalidExecution(String),
    #[error("failed to inspect MTP artifact {path}: {source}")]
    Inspection {
        path: PathBuf,
        #[source]
        source: MtpPreflightError,
    },
    #[error("target artifact is an MTP draft and cannot be served as the target model")]
    DraftUsedAsTarget,
    #[error("bundled MTP failed native compatibility preflight: {0}")]
    BundledPreflight(#[source] MtpPreflightError),
    #[error("selected MTP artifact is incompatible: {0}")]
    ExplicitCandidate(#[source] MtpPreflightError),
    #[error("multiple separate MTP artifacts are natively compatible: {0:?}")]
    AmbiguousCandidates(Vec<PathBuf>),
}

/// Select MTP for the exact execution plan without loading model tensors.
///
/// Selection is intentionally small and deterministic:
///
/// 1. Use executable MTP bundled in the target when native preflight succeeds.
/// 2. Otherwise use the only separate candidate that native linked-context preflight accepts.
/// 3. Disable MTP when none work, and reject ambiguity rather than guessing.
pub fn select_mtp(
    plan: &ResolvedExecutionPlan,
    candidates: CandidatePolicy<'_>,
) -> Result<MtpConfig, SelectionError> {
    let backend =
        LlamaBackend::init().map_err(|error| SelectionError::Backend(error.to_string()))?;
    select_mtp_with_backend(&backend, plan, candidates)
}

/// Select MTP using an already initialized llama.cpp backend.
///
/// Serving processes call this from their exclusive native executor. The backend reference is a
/// lifetime proof for the process-global device state used by file inspection and linked graph
/// preflight; selection itself still loads no model tensors.
pub fn select_mtp_with_backend(
    _backend: &LlamaBackend,
    plan: &ResolvedExecutionPlan,
    candidates: CandidatePolicy<'_>,
) -> Result<MtpConfig, SelectionError> {
    let model_params = native_model_params(&plan.execution)?;
    let target_context = native_context_params(plan, &plan.execution, DEFAULT_N_MAX);
    let target_info = inspect_mtp_model(&plan.model_path, &model_params).map_err(|source| {
        SelectionError::Inspection {
            path: plan.model_path.clone(),
            source,
        }
    })?;

    match target_info.map(|info| info.kind) {
        Some(MtpModelKind::Bundled) => {
            preflight_mtp(
                &plan.model_path,
                None,
                &MtpPreflightParams {
                    target_model: &model_params,
                    target_context: &target_context,
                    draft_model: None,
                    draft_context: None,
                },
            )
            .map_err(SelectionError::BundledPreflight)?;
            return Ok(enabled(MtpSource::Bundled));
        }
        Some(MtpModelKind::Draft) => return Err(SelectionError::DraftUsedAsTarget),
        None => {}
    }

    let paths: Vec<&Path> = match candidates {
        CandidatePolicy::Automatic(paths) => paths.iter().map(PathBuf::as_path).collect(),
        CandidatePolicy::Explicit(path) => vec![path],
    };
    let draft_context = native_context_params(plan, &plan.execution, 0);
    let mut compatible = Vec::new();
    for path in paths {
        let info = inspect_mtp_model(path, &model_params).map_err(|source| {
            SelectionError::Inspection {
                path: path.to_path_buf(),
                source,
            }
        })?;
        if !matches!(info.map(|info| info.kind), Some(MtpModelKind::Draft)) {
            if matches!(candidates, CandidatePolicy::Explicit(_)) {
                return Err(SelectionError::ExplicitCandidate(
                    MtpPreflightError::DraftNotSupported,
                ));
            }
            continue;
        }
        let result = preflight_mtp(
            &plan.model_path,
            Some(path),
            &MtpPreflightParams {
                target_model: &model_params,
                target_context: &target_context,
                draft_model: Some(&model_params),
                draft_context: Some(&draft_context),
            },
        );
        match result {
            Ok(_) => compatible.push(path.to_path_buf()),
            Err(error) if matches!(candidates, CandidatePolicy::Explicit(_)) => {
                return Err(SelectionError::ExplicitCandidate(error));
            }
            Err(
                MtpPreflightError::VocabularyMismatch
                | MtpPreflightError::EmbeddingMismatch
                | MtpPreflightError::ContextUnsupported
                | MtpPreflightError::DraftNotSupported,
            ) => {}
            Err(error) => {
                return Err(SelectionError::Inspection {
                    path: path.to_path_buf(),
                    source: error,
                });
            }
        }
    }

    resolve_compatible_candidates(compatible)
}

fn resolve_compatible_candidates(compatible: Vec<PathBuf>) -> Result<MtpConfig, SelectionError> {
    match compatible.as_slice() {
        [] => Ok(MtpConfig::Disabled {
            reason: "native_mtp_unavailable".to_owned(),
        }),
        [path] => Ok(enabled(MtpSource::Separate {
            model_path: path.clone(),
        })),
        _ => Err(SelectionError::AmbiguousCandidates(compatible)),
    }
}

fn enabled(source: MtpSource) -> MtpConfig {
    MtpConfig::Enabled {
        source,
        n_max: DEFAULT_N_MAX,
        n_min: DEFAULT_N_MIN,
        p_min: DEFAULT_P_MIN,
        cache_type_k: CacheType::F16,
        cache_type_v: CacheType::F16,
    }
}

fn native_model_params(execution: &ExecutionConfig) -> Result<LlamaModelParams, SelectionError> {
    let params = LlamaModelParams::default()
        .with_gpu_layers(match execution.gpu_layers {
            GpuLayers::Auto => LlamaGpuLayers::Auto,
            GpuLayers::All => LlamaGpuLayers::All,
            GpuLayers::Count(value) => LlamaGpuLayers::Count(value),
        })
        .with_use_mmap(execution.use_mmap)
        .with_use_mlock(execution.use_mlock)
        .with_split_mode(match execution.split_mode {
            SplitMode::None => LlamaSplitMode::None,
            SplitMode::Layer => LlamaSplitMode::Layer,
            SplitMode::Row => LlamaSplitMode::Row,
            SplitMode::Tensor => LlamaSplitMode::Tensor,
        });
    match &execution.tensor_split {
        Some(weights) => params
            .with_tensor_split(weights)
            .map_err(|error| SelectionError::InvalidExecution(error.to_string())),
        None => Ok(params),
    }
}

fn native_context_params(
    plan: &ResolvedExecutionPlan,
    execution: &ExecutionConfig,
    recurrent_snapshots: u32,
) -> LlamaContextParams {
    let threads = execution
        .threads
        .map(NonZeroU32::get)
        .unwrap_or_else(default_threads);
    let threads_batch = execution.threads_batch.map_or(threads, NonZeroU32::get);
    let outputs = plan
        .max_sequences
        .saturating_mul(recurrent_snapshots.saturating_add(1))
        .min(plan.batch_size);
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(plan.context_size))
        .with_n_batch(plan.batch_size)
        .with_n_ubatch(plan.ubatch_size)
        .with_n_seq_max(plan.max_sequences)
        .with_n_outputs_max(NonZeroU32::new(outputs))
        .with_n_threads(threads.min(i32::MAX as u32) as i32)
        .with_n_threads_batch(threads_batch.min(i32::MAX as u32) as i32)
        .with_type_k(native_cache_type(execution.cache_type_k))
        .with_type_v(native_cache_type(execution.cache_type_v))
        .with_offload_kqv(execution.offload_kqv)
        .with_op_offload(execution.operation_offload)
        .with_swa_full(execution.swa_full)
        .with_kv_unified(execution.kv_unified)
        .with_n_rs_seq(recurrent_snapshots)
        .with_flash_attention(match execution.flash_attention {
            FlashAttention::Auto => FlashAttentionPolicy::Auto,
            FlashAttention::Disabled => FlashAttentionPolicy::Disabled,
            FlashAttention::Enabled => FlashAttentionPolicy::Enabled,
        })
}

fn default_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|value| u32::try_from(value.get()).unwrap_or(u32::MAX))
        .unwrap_or(1)
}

fn native_cache_type(cache_type: CacheType) -> KvCacheType {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disables_mtp_when_native_preflight_accepts_no_candidate() {
        assert!(matches!(
            resolve_compatible_candidates(Vec::new()).unwrap(),
            MtpConfig::Disabled { reason } if reason == "native_mtp_unavailable"
        ));
    }

    #[test]
    fn selects_the_only_natively_compatible_candidate() {
        let path = PathBuf::from("draft.gguf");
        assert!(matches!(
            resolve_compatible_candidates(vec![path.clone()]).unwrap(),
            MtpConfig::Enabled {
                source: MtpSource::Separate { model_path },
                n_max: DEFAULT_N_MAX,
                n_min: DEFAULT_N_MIN,
                p_min: DEFAULT_P_MIN,
                ..
            } if model_path == path
        ));
    }

    #[test]
    fn rejects_multiple_natively_compatible_candidates_without_guessing() {
        let paths = vec![PathBuf::from("first.gguf"), PathBuf::from("second.gguf")];
        assert!(matches!(
            resolve_compatible_candidates(paths.clone()),
            Err(SelectionError::AmbiguousCandidates(candidates)) if candidates == paths
        ));
    }
}
