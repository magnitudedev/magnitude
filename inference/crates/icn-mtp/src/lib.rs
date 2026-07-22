//! Native MTP discovery, compatibility validation, and automatic selection policy.
//!
//! This crate owns ICN's MTP policy. Construction of the linked contexts and native speculative
//! controller is the capability test, and all native access goes through safe `llama-cpp-2` APIs.

use std::path::{Path, PathBuf};

use icn_contracts::{CacheType, ExecutionIntent, MtpConfig, MtpSource};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::mtp::{MtpPreflightError, MtpPreflightParams, preflight_mtp};

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
    #[error("invalid native execution parameters: {0}")]
    InvalidExecution(String),
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
/// 2. Otherwise use the only separate candidate that native execution preflight accepts.
/// 3. Disable MTP when none work, and reject ambiguity rather than guessing.
///
/// Serving processes call this from their exclusive native executor. The backend reference is a
/// lifetime proof for the process-global device state used by file inspection and linked graph
/// preflight; selection itself still loads no model tensors.
pub fn select_mtp_with_backend(
    _backend: &LlamaBackend,
    plan: &ExecutionIntent,
    candidates: CandidatePolicy<'_>,
) -> Result<MtpConfig, SelectionError> {
    let native = icn_hardware::mtp_preflight_parameters(plan, DEFAULT_N_MAX)
        .map_err(|error| SelectionError::InvalidExecution(error.to_string()))?;
    let model_params = native.model_params.as_ref().get_ref();
    let bundled = preflight_mtp(
        &plan.model_path,
        None,
        &MtpPreflightParams {
            target_model: model_params,
            target_context: &native.target_context,
            draft_model: None,
            draft_context: None,
        },
    );
    match bundled {
        Ok(_) => return Ok(enabled(MtpSource::Bundled)),
        Err(MtpPreflightError::ContextUnsupported) => {}
        Err(error) => return Err(SelectionError::BundledPreflight(error)),
    }

    let paths: Vec<&Path> = match candidates {
        CandidatePolicy::Automatic(paths) => paths.iter().map(PathBuf::as_path).collect(),
        CandidatePolicy::Explicit(path) => vec![path],
    };
    let mut compatible = Vec::new();
    for path in paths {
        let result = preflight_mtp(
            &plan.model_path,
            Some(path),
            &MtpPreflightParams {
                target_model: model_params,
                target_context: &native.target_context,
                draft_model: Some(model_params),
                draft_context: Some(&native.draft_context),
            },
        );
        match result {
            Ok(_) => compatible.push(path.to_path_buf()),
            Err(error) if matches!(candidates, CandidatePolicy::Explicit(_)) => {
                return Err(SelectionError::ExplicitCandidate(error));
            }
            Err(MtpPreflightError::ContextUnsupported) => {}
            Err(error) => return Err(SelectionError::ExplicitCandidate(error)),
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
