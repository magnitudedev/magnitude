//! The one derivation of a model's execution plan draft from a resolved
//! execution manifest and a selected device. The production load and
//! metadata-only assessment both plan through here, so assessment sees
//! exactly the components, method, codec and resource limits a load uses.
//!
//! Only the manifest's device-free facts are read: the package manifest's
//! tensor directory (a header-only open is sufficient), the model
//! definition, the resolved model policy and the service limits.

use crate::options::{ExecutionManifest, ResolvedMethod};
use magnitude_batching::MAX_CLASS_ROWS;
use magnitude_executor::{
    platform::SelectedDevice, ComponentSelection, ExecutionPlanDraft, ExecutionPlanner, PlanError,
    PlannedMethod, ResourceLimits,
};
use magnitude_scheduler::ServiceLimits;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionPlanningError {
    /// The service's token allowance needs more rows than one launch class admits.
    LaunchRows { required: usize, admitted: usize },
    /// The execution planner rejected the model on this device.
    Plan(PlanError),
}

impl fmt::Display for ExecutionPlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaunchRows { required, admitted } => write!(
                formatter,
                "service policy requires {required} rows but the execution contract admits at most {admitted}"
            ),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExecutionPlanningError {}

/// Plan the manifest's model on the selected device: component selection,
/// planned method, KV codec and resource limits resolved from the manifest,
/// then `ExecutionPlanner::prepare`. Opens no device and reads no payload.
pub fn plan_execution(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ExecutionPlanDraft, ExecutionPlanningError> {
    let selection = ComponentSelection {
        head: !matches!(manifest.model.method, ResolvedMethod::Plain),
        vision: manifest.definition.vision.is_some(),
    };
    let method = match manifest.model.method {
        ResolvedMethod::Plain => PlannedMethod::Plain,
        ResolvedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
        } => PlannedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
        },
        ResolvedMethod::DFlash { proposals } => PlannedMethod::DFlash { proposals },
    };
    let limits = resource_limits(
        &manifest.service,
        manifest.model.lookahead,
        selected.info.backend,
    )?;
    ExecutionPlanner::prepare(
        selected,
        &manifest.package,
        &manifest.definition,
        selection,
        manifest.path,
        method,
        manifest.model.kv_codec,
        limits,
    )
    .map_err(ExecutionPlanningError::Plan)
}

/// The resource limits a load of this manifest plans for.
fn resource_limits(
    service: &ServiceLimits,
    lookahead: bool,
    backend: seismic::BackendName,
) -> Result<ResourceLimits, ExecutionPlanningError> {
    let max_launch_rows = service.prefill_tokens.max(service.decode_tokens);
    // Larger prefill classes have a served gain and memory gate on CUDA.
    // Other backends retain their measured 512-row admission bound.
    let admitted = if backend == seismic::BackendName::Cuda {
        MAX_CLASS_ROWS
    } else {
        512
    };
    if max_launch_rows > admitted {
        return Err(ExecutionPlanningError::LaunchRows {
            required: max_launch_rows,
            admitted,
        });
    }
    Ok(ResourceLimits {
        max_launch_rows,
        // Every request consumes at least one token in either phase. A
        // prefill round can therefore have as many request slots and logits
        // rows as its larger token allowance admits.
        max_launch_slots: max_launch_rows,
        max_projected_rows: max_launch_rows,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        lookahead,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_budget_admits_more_request_slots_than_decode_budget() {
        let limits = resource_limits(
            &ServiceLimits {
                prefill_tokens: 64,
                decode_tokens: 16,
                decode_share: 0.5,
                locality_seconds: 1.0,
            },
            false,
            seismic::BackendName::Metal,
        )
        .unwrap();
        assert_eq!(limits.max_launch_rows, 64);
        assert_eq!(limits.max_launch_slots, 64);
        assert_eq!(limits.max_projected_rows, 64);
    }

    #[test]
    fn large_prefill_classes_require_cuda() {
        let service = ServiceLimits {
            prefill_tokens: 1024,
            decode_tokens: 32,
            decode_share: 0.5,
            locality_seconds: 1.0,
        };
        assert_eq!(
            resource_limits(&service, false, seismic::BackendName::Cuda)
                .unwrap()
                .max_launch_rows,
            1024
        );
        assert_eq!(
            resource_limits(&service, false, seismic::BackendName::Metal),
            Err(ExecutionPlanningError::LaunchRows {
                required: 1024,
                admitted: 512,
            })
        );
    }
}
