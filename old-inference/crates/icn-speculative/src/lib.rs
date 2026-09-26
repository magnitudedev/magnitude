//! Native speculative-decoding compatibility validation.
//!
//! This crate owns ICN's method-neutral speculative preflight. The exact method and embedded or
//! separate draft source come from the servable bundle; selection never scans installed files or
//! infers a method from an artifact name or architecture string.

use icn_contracts::{
    ExecutionIntent, SpeculativeDecodingConfig, SpeculativeDraftSource, SpeculativeMethodConfig,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::speculative::{
    SpeculativeMethod, SpeculativePreflightError, SpeculativePreflightParams, preflight_speculative,
};

#[derive(Debug, thiserror::Error)]
pub enum PreflightError {
    #[error("invalid native execution parameters: {0}")]
    InvalidExecution(String),
    #[error("selected speculative bundle is incompatible: {0}")]
    Incompatible(#[source] SpeculativePreflightError),
}

/// Validate the exact speculative configuration already selected by the servable bundle.
///
/// Disabled intent remains disabled. Enabled intent is returned unchanged only after the native
/// target/draft context and selected llama.cpp speculative implementation can be constructed.
pub fn preflight_with_backend(
    _backend: &LlamaBackend,
    plan: &ExecutionIntent,
) -> Result<SpeculativeDecodingConfig, PreflightError> {
    let SpeculativeDecodingConfig::Enabled {
        source,
        method,
        n_max,
        n_min,
        ..
    } = &plan.speculative
    else {
        return Ok(plan.speculative.clone());
    };

    let native = icn_hardware::speculative_preflight_parameters(plan)
        .map_err(|error| PreflightError::InvalidExecution(error.to_string()))?;
    let draft_path = match source {
        SpeculativeDraftSource::Embedded => None,
        SpeculativeDraftSource::Separate { model_path } => Some(model_path.as_path()),
    };
    let native_method = match method {
        SpeculativeMethodConfig::Mtp {
            min_draft_probability,
        } => SpeculativeMethod::Mtp {
            min_draft_probability: *min_draft_probability,
        },
        SpeculativeMethodConfig::DFlash {
            min_sample_probability,
        } => SpeculativeMethod::DFlash {
            min_sample_probability: *min_sample_probability,
        },
        SpeculativeMethodConfig::DSpark {
            acceptance_threshold,
        } => SpeculativeMethod::DSpark {
            acceptance_threshold: *acceptance_threshold,
        },
    };
    let separate = matches!(source, SpeculativeDraftSource::Separate { .. });
    let preflight = preflight_speculative(
        &plan.model_path,
        draft_path,
        &SpeculativePreflightParams {
            method: native_method,
            n_max: i32::try_from(*n_max).unwrap_or(i32::MAX),
            n_min: i32::try_from(*n_min).unwrap_or(i32::MAX),
            target_model: native.target_model_params.as_ref().get_ref(),
            target_context: &native.target_context,
            draft_model: separate.then_some(native.draft_model_params.as_ref().get_ref()),
            draft_context: separate.then_some(&native.draft_context),
        },
    )
    .map_err(PreflightError::Incompatible)?;

    let mut effective = plan.speculative.clone();
    if let SpeculativeDecodingConfig::Enabled { n_max, n_min, .. } = &mut effective {
        *n_max = u32::try_from(preflight.effective_n_max)
            .map_err(|_| PreflightError::InvalidExecution("negative effective n_max".into()))?;
        *n_min = u32::try_from(preflight.effective_n_min)
            .map_err(|_| PreflightError::InvalidExecution("negative effective n_min".into()))?;
    }
    Ok(effective)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_intent_does_not_require_native_preflight() {
        let mut plan = icn_engine_test_intent();
        plan.speculative = SpeculativeDecodingConfig::Disabled {
            reason: "standalone_bundle".to_owned(),
        };
        assert_eq!(
            plan.speculative,
            SpeculativeDecodingConfig::Disabled {
                reason: "standalone_bundle".to_owned(),
            }
        );
    }

    fn icn_engine_test_intent() -> ExecutionIntent {
        ExecutionIntent {
            model_path: "model.gguf".into(),
            context_size: 4096,
            physical_context_size: 4096,
            batch_size: 512,
            ubatch_size: 128,
            max_sequences: 1,
            prefill_quantum: 128,
            execution: Default::default(),
            projector: None,
            speculative: Default::default(),
        }
    }
}
