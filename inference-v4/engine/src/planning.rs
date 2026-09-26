//! The one derivation of a model's execution plan draft from a resolved
//! execution manifest and a selected device. The production load and
//! metadata-only assessment both plan through here, so assessment sees
//! exactly the components, method, codec and resource limits a load uses.
//!
//! Only the manifest's device-free facts are read: the package manifest's
//! tensor directory (a header-only open is sufficient), the model
//! definition, the resolved model policy and the service limits.

use crate::options::{ExecutionManifest, ResolvedMethod};
use magnitude_model_batching::MAX_CLASS_ROWS;
use magnitude_model_executor::{
    platform::SelectedDevice, ComponentSelection, ExecutionPlanDraft, ExecutionPlanner, PlanError,
    PlannedMethod, ResourceLimits,
};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionPlanningError {
    /// The model's context limit does not fit the host's address domain.
    ContextExceedsHost(u64),
    /// The service's batch policy needs more rows than one launch class admits.
    BatchRows { required: usize, admitted: usize },
    /// The execution planner rejected the model on this device.
    Plan(PlanError),
}

impl fmt::Display for ExecutionPlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextExceedsHost(context) => {
                write!(formatter, "model context limit {context} exceeds host domain")
            }
            Self::BatchRows { required, admitted } => write!(
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
        head: matches!(manifest.model.method, ResolvedMethod::Mtp { .. }),
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
    };
    let limits = resource_limits(manifest)?;
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
fn resource_limits(manifest: &ExecutionManifest) -> Result<ResourceLimits, ExecutionPlanningError> {
    // A batch holds at most `max_batch` requests of at most the context each,
    // so no batch exceeds their product whatever the service's token budgets.
    let context_limit = manifest.definition.geometry.context_limit;
    let context = usize::try_from(context_limit)
        .map_err(|_| ExecutionPlanningError::ContextExceedsHost(context_limit))?;
    let service = &manifest.service;
    let max_batch_rows = service
        .prefill_tokens
        .max(service.decode_tokens)
        .min(service.max_batch.saturating_mul(context));
    if max_batch_rows > MAX_CLASS_ROWS {
        return Err(ExecutionPlanningError::BatchRows {
            required: max_batch_rows,
            admitted: MAX_CLASS_ROWS,
        });
    }
    Ok(ResourceLimits {
        max_retained_entries: service.max_requests,
        active_requests: service.max_batch,
        in_flight_requests: service.max_batch,
        // The service exposes checkpoint/fork for plain as well as MTP
        // requests. Reserve the bounded live-request checkpoint class.
        branch_checkpoints: service.max_batch,
        max_batch_rows,
        max_projected_rows: service
            .decode_tokens
            .max(service.max_batch)
            .min(max_batch_rows),
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        lookahead: manifest.model.lookahead,
    })
}
