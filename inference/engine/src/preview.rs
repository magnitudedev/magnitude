//! Read-only load planning (integration spec §4.2, §7.3): which device a load
//! of a resolved configuration would use and how much of that device's
//! allocation domain it claims at startup. Opens no device, reads no weight
//! payload and allocates nothing; a load repeats device resolution and makes
//! live claims.

use crate::error::{classify_graph, classify_plan, classify_platform, PlanOutcome, PreviewError};
use crate::options::ExecutionManifest;
use crate::planning::{plan_execution, ExecutionPlanningError};
use magnitude_executor::{
    platform,
    AssessmentGraphResourceBounds, AssessmentHeaderBounds, AssessmentMemoryTerms,
    ResourceCapacity, ResourcePlanner,
};
use seismic::{BackendName, DeviceCatalog, DeviceSelector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadPreview {
    /// The served context ceiling.
    pub context_tokens: u64,
    /// The device a load would open, by its cross-process identity.
    pub device: DeviceSelector,
    pub backend: BackendName,
    /// The load's startup peak claim in the device's allocation domain:
    /// resident weights, initial state, prepared resources and the startup
    /// transient. History growth is elastic and claimed as it grows, so it is
    /// not included.
    pub required_bytes: u64,
}

fn internal(reason: impl Into<String>) -> PreviewError {
    PreviewError::Internal {
        reason: reason.into(),
    }
}

impl ExecutionManifest {
    /// Preview a load of this manifest on `catalog`. Only the manifest is read, so a host that
    /// shares its host artifacts elsewhere can still preview.
    pub fn preview(&self, catalog: &DeviceCatalog) -> Result<LoadPreview, PreviewError> {
        let manifest = self;
        let selected = platform::select_device(
            catalog,
            manifest.path,
            manifest.device,
            &manifest.reserves,
        )
        .map_err(|error| PreviewError::from(classify_platform(error)))?;
        let backend = selected.info.backend;
        let draft = plan_execution(manifest, &selected).map_err(|error| match error {
            ExecutionPlanningError::Plan(error) => classify_plan(error, backend).into(),
            error => PreviewError::from(PlanOutcome::Internal(error.to_string())),
        })?;
        let definition = &manifest.definition;
        let policy = draft.policy();
        let terms = AssessmentMemoryTerms::derive(
            definition,
            draft.load(),
            policy.selection(),
            policy.codec(),
            policy.method(),
            policy.limits(),
        )
        .map_err(internal)?;
        let state = ResourcePlanner::state_plan(
            definition,
            draft.load(),
            policy.method(),
            policy.codec(),
            policy.limits(),
            ResourceCapacity {
                domain_bytes: selected.assessment_capacity_bytes,
            },
        )
        .map_err(|error| internal(error.to_string()))?;
        let graph = AssessmentGraphResourceBounds::derive(
            definition,
            draft.load(),
            &state,
            policy.method(),
            policy.codec(),
            policy.limits(),
            backend,
        )
        .map_err(|error| PreviewError::from(classify_graph(error)))?;
        let bounds =
            AssessmentHeaderBounds::derive(definition, draft.load(), policy.codec(), backend)
                .and_then(|header| header.with_graph_resource_bound(&graph))
                .map_err(internal)?;
        let initial_state = state
            .target_state()
            .initial_committed_bytes()
            .and_then(|target| {
                let head = state
                    .head_state()
                    .map(|head| head.initial_committed_bytes())
                    .transpose()?
                    .unwrap_or(0);
                target
                    .checked_add(head)
                    .ok_or_else(|| "initial state byte count overflow".to_owned())
            })
            .map_err(internal)?;
        let required_bytes = [
            terms.target_weights,
            terms.head_weights,
            terms.vision_weights,
            initial_state,
            bounds.prepared_resource_bytes,
            bounds.startup_additional_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| internal("startup byte count overflow"))?;
        Ok(LoadPreview {
            context_tokens: definition.decoder.context_limit,
            device: selected.info.selector,
            backend,
            required_bytes,
        })
    }
}
