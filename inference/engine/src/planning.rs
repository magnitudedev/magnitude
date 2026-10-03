//! The one derivation of a model's execution plan draft from a resolved
//! execution manifest and a selected device. The production load and
//! metadata-only assessment both plan through here, so assessment sees
//! exactly the components, method, codec and resource limits a load uses.
//!
//! Only the manifest's device-free facts are read: the package manifest's
//! tensor directory (a header-only open is sufficient), the model
//! definition, the resolved model policy and the service limits.

use crate::options::{ExecutionManifest, ResolvedMethod, ResolvedModelPolicy};
use magnitude_batching::MAX_CLASS_ROWS;
use magnitude_executor::{
    platform::SelectedDevice, BackendPlan, ComponentSelection, ExecutionPlanDraft,
    ExecutionPlanner, PlanError, PlannedMethod, ResourceLimits,
};
use magnitude_scheduler::ServiceLimits;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionPlanningError {
    /// The service's token allowance needs more rows than one launch class admits.
    LaunchRows { required: usize, admitted: usize },
    /// The policy exports logits for more rows than one launch carries.
    LogitsRows { required: usize, admitted: usize },
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
            Self::LogitsRows { required, admitted } => write!(
                formatter,
                "model policy exports logits for {required} rows but a launch carries at most {admitted}"
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
    let (selection, method, limits) = planner_inputs(manifest, selected.info.backend)?;
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

/// Plan the manifest's model on `backend`, which is all planning reads of a
/// device: the plan a load on any device of that backend executes.
pub fn plan_backend(
    manifest: &ExecutionManifest,
    backend: seismic::BackendName,
) -> Result<BackendPlan, ExecutionPlanningError> {
    let (selection, method, limits) = planner_inputs(manifest, backend)?;
    ExecutionPlanner::backend_plan(
        backend,
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

/// The component selection, planned method and resource limits the
/// manifest resolves on `backend`.
fn planner_inputs(
    manifest: &ExecutionManifest,
    backend: seismic::BackendName,
) -> Result<(ComponentSelection, PlannedMethod, ResourceLimits), ExecutionPlanningError> {
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
    let limits = resource_limits(&manifest.service, &manifest.model, backend)?;
    Ok((selection, method, limits))
}

/// The resource limits a load of this manifest plans for.
fn resource_limits(
    service: &ServiceLimits,
    model: &ResolvedModelPolicy,
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
    if model.exported_logits_rows > max_launch_rows {
        return Err(ExecutionPlanningError::LogitsRows {
            required: model.exported_logits_rows,
            admitted: max_launch_rows,
        });
    }
    Ok(ResourceLimits {
        max_launch_rows,
        // Every request consumes at least one token in either phase. A
        // prefill round can therefore have as many request slots as its
        // larger token allowance admits.
        max_launch_slots: max_launch_rows,
        max_selected_rows: service.selection_bound(),
        max_drafting_slots: service.selection_bound(),
        exported_logits_rows: model.exported_logits_rows,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        lookahead: model.lookahead,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::KvCodec;

    fn model(exported_logits_rows: usize) -> ResolvedModelPolicy {
        ResolvedModelPolicy {
            method: ResolvedMethod::Plain,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows,
        }
    }

    fn service(prefill_tokens: usize, decode_tokens: usize) -> ServiceLimits {
        ServiceLimits {
            prefill_tokens,
            decode_tokens,
            decode_share: 0.5,
            locality_seconds: 1.0,
        }
    }

    #[test]
    fn prefill_budget_admits_more_request_slots_than_decode_budget() {
        let limits =
            resource_limits(&service(64, 16), &model(0), seismic::BackendName::Metal).unwrap();
        assert_eq!(limits.max_launch_rows, 64);
        assert_eq!(limits.max_launch_slots, 64);
    }

    #[test]
    fn the_selection_bound_is_the_decode_allowance() {
        let limits =
            resource_limits(&service(512, 32), &model(0), seismic::BackendName::Metal).unwrap();
        assert_eq!(limits.max_selected_rows, 32);
        assert_eq!(limits.max_drafting_slots, 32);
        assert_eq!(limits.exported_logits_rows, 0);
    }

    #[test]
    fn logits_export_is_bounded_by_the_launch() {
        let limits =
            resource_limits(&service(512, 32), &model(512), seismic::BackendName::Metal).unwrap();
        assert_eq!(limits.exported_logits_rows, 512);
        assert_eq!(
            resource_limits(&service(512, 32), &model(513), seismic::BackendName::Metal),
            Err(ExecutionPlanningError::LogitsRows {
                required: 513,
                admitted: 512,
            })
        );
    }

    #[test]
    fn large_prefill_classes_require_cuda() {
        let service = service(1024, 32);
        assert_eq!(
            resource_limits(&service, &model(0), seismic::BackendName::Cuda)
                .unwrap()
                .max_launch_rows,
            1024
        );
        assert_eq!(
            resource_limits(&service, &model(0), seismic::BackendName::Metal),
            Err(ExecutionPlanningError::LaunchRows {
                required: 1024,
                admitted: 512,
            })
        );
    }
}
