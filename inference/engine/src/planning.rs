//! The one derivation of a model's execution plan draft from a resolved
//! execution manifest and a selected device. The production load and
//! metadata-only assessment both plan through here, so assessment sees
//! exactly the components, method, codec and resource limits a load uses.
//!
//! Only the manifest's device-free facts are read: the package manifest's
//! tensor directory (a header-only open is sufficient), the model
//! definition, the resolved model policy and the service limits.

use crate::options::{DraftPolicy, ExecutionManifest, ResolvedMethod, ResolvedModelPolicy};
use magnitude_batching::MAX_CLASS_ROWS;
use magnitude_executor::{
    image_cell_limit, max_image_cells,
    platform::{self, PlatformError, SelectedDevice},
    BackendPlan, ComponentSelection,
    DraftWidths,
    ChargedDraft, ExecutionPlanDraft, ExecutionPlanner, HeldCharge, PlanError, PlannedMethod, ResourceLimits,
};
use magnitude_scheduler::ServiceLimits;
use seismic::Devices;
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

/// Select the manifest's device as a load and its preview do, measuring
/// nothing: what its graphs hold is the measurement job's point, read from
/// its kernel cache's records, its backend's bound until stored there
/// (`platform::select_recorded`).
pub(crate) fn select_load_device(
    catalog: &Devices,
    manifest: &ExecutionManifest,
) -> Result<SelectedDevice, PlatformError> {
    platform::select_recorded(catalog, manifest.path, manifest.device, &manifest.reserves, manifest.kernel_cache.as_deref())
}

/// Plan the manifest's model on the selected device: component selection,
/// planned method, KV codec and resource limits resolved from the manifest,
/// then `ExecutionPlanner::prepare`, then its expert bank under the model
/// policy's weight cap, if any, as the plan's settle grants it on the
/// machine's prices (`ExecutionPlanDraft::with_bank`, [`plan_banked`]).
/// Opens no device and reads no payload. The draft comes with the memory
/// charge its planning confirmed, each way of holding its experts charged
/// as a `C` (`HeldCharge`).
pub fn plan_execution<C: HeldCharge>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ChargedDraft<C>, ExecutionPlanningError> {
    plan_resident(manifest, selected).and_then(|resident| plan_banked(manifest, selected, resident))
}

/// [`plan_execution`]'s structure, whatever room the machine has: its
/// experts held as its largest plan holds them under the model policy's
/// weight cap (`ExecutionPlanDraft::with_ceiling_bank`), what metadata-only
/// assessment prepares from the structure alone.
pub fn plan_ceiling<C: HeldCharge>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ChargedDraft<C>, ExecutionPlanningError> {
    plan_resident(manifest, selected)?
        .with_ceiling_bank(&manifest.definition, manifest.model.weight_cap)
        .map_err(ExecutionPlanningError::Plan)
}

/// [`plan_execution`]'s plan before its expert bank: every expert resident.
pub fn plan_resident(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ExecutionPlanDraft, ExecutionPlanningError> {
    let (selection, method, limits) = planner_inputs(manifest, selected.info.backend())?;
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

/// [`plan_execution`]'s plan from its `resident` plan: its expert bank,
/// under the model policy's weight cap, if any, with its charge, settled on
/// the machine's prices (`assessment::plan_prices`): the records of its
/// kernel cache, its model store's figures.
///
/// Every load, banked or not, is priced on its executor's costs as it
/// serves, so none plans before the measurement job stored the costs'
/// required figures: while they are pending the plan's prices are
/// (`PlanError::Unpriced`), and the load waits for them holding nothing of
/// the device, once per machine. Without a kernel cache nothing is measured
/// for it, and its prices are unavailable.
pub fn plan_banked<C: HeldCharge>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
    resident: ExecutionPlanDraft,
) -> Result<ChargedDraft<C>, ExecutionPlanningError> {
    use crate::assessment::Measurements;
    let measurements = match manifest.kernel_cache.as_deref() {
        None => Measurements::Unavailable { reason: "the load keeps no kernel cache, which holds the machine's figures".into() },
        Some(kernels) => crate::assessment::stored_measurements(
            &selected.reading.topology,
            &selected.info,
            kernels,
            manifest.store(),
        ),
    };
    if let Measurements::Pending = measurements {
        return Err(ExecutionPlanningError::Plan(PlanError::Unpriced {
            pending: true,
            reason: "the executor's costs are not measured yet".into(),
        }));
    }
    let prices = crate::assessment::plan_prices(&measurements, &resident);
    resident
        .with_bank(&manifest.definition, manifest.model.weight_cap, &prices)
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
    // A fixed budget drafts its generated width, or a narrower one at a
    // request's end; an adaptive load any width its controller chooses.
    let widths = match manifest.model.draft_policy {
        Some(DraftPolicy::Fixed(budget)) => DraftWidths::Fixed(budget.generated as u8),
        Some(DraftPolicy::Adaptive) | None => DraftWidths::Adaptive,
    };
    let method = match manifest.model.method {
        ResolvedMethod::Plain => PlannedMethod::Plain,
        ResolvedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
            ..
        } => PlannedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
            widths,
        },
        ResolvedMethod::DFlash { proposals } => PlannedMethod::DFlash { proposals, widths },
    };
    let image_cells = image_cell_limit(
        &manifest.definition,
        manifest.service.launch_rows(),
        max_image_cells(backend),
    )
    .map_err(ExecutionPlanningError::Plan)?;
    let limits = resource_limits(&manifest.service, &manifest.model, image_cells, backend)?;
    Ok((selection, method, limits))
}

/// The resource limits a load of this manifest plans for, encoding images of
/// at most `image_cells` cells.
fn resource_limits(
    service: &ServiceLimits,
    model: &ResolvedModelPolicy,
    image_cells: usize,
    backend: seismic::BackendName,
) -> Result<ResourceLimits, ExecutionPlanningError> {
    let max_launch_rows = service.launch_rows();
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
    // A CPU launch completes during its submission, so a queued successor
    // would run before the owner could apply a cancellation, stop or
    // admission, and it overlaps nothing.
    let lookahead = model.lookahead && backend != seismic::BackendName::Cpu;
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
        max_image_cells: image_cells,
        lookahead,
        // One launch until the load's chain is planned beside its weights
        // (`ExecutionPlanDraft::with_bank`).
        chain_capacity: std::num::NonZeroUsize::MIN,
        conversations: service.conversations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::KvCodec;

    fn model(exported_logits_rows: usize) -> ResolvedModelPolicy {
        ResolvedModelPolicy {
            draft_policy: None,
            method: ResolvedMethod::Plain,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows,
            numerical_profile: None,
            weight_cap: None,
        }
    }

    fn service(prefill_tokens: usize, decode_tokens: usize) -> ServiceLimits {
        ServiceLimits {
            prefill_tokens,
            decode_tokens,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        }
    }

    #[test]
    fn prefill_budget_admits_more_request_slots_than_decode_budget() {
        let limits =
            resource_limits(&service(64, 16), &model(0), 0, seismic::BackendName::Metal).unwrap();
        assert_eq!(limits.max_launch_rows, 64);
        assert_eq!(limits.max_launch_slots, 64);
    }

    #[test]
    fn the_selection_bound_is_the_decode_allowance() {
        let limits =
            resource_limits(&service(512, 32), &model(0), 0, seismic::BackendName::Metal).unwrap();
        assert_eq!(limits.max_selected_rows, 32);
        assert_eq!(limits.max_drafting_slots, 32);
        assert_eq!(limits.exported_logits_rows, 0);
    }

    #[test]
    fn logits_export_is_bounded_by_the_launch() {
        let limits = resource_limits(
            &service(512, 32),
            &model(512),
            0,
            seismic::BackendName::Metal,
        )
        .unwrap();
        assert_eq!(limits.exported_logits_rows, 512);
        assert_eq!(
            resource_limits(
                &service(512, 32),
                &model(513),
                0,
                seismic::BackendName::Metal
            ),
            Err(ExecutionPlanningError::LogitsRows {
                required: 513,
                admitted: 512,
            })
        );
    }

    #[test]
    fn synchronous_cpu_launches_queue_no_lookahead() {
        let model = ResolvedModelPolicy {
            lookahead: true,
            ..model(0)
        };
        for (backend, expected) in [
            (seismic::BackendName::Cpu, false),
            (seismic::BackendName::Metal, true),
            (seismic::BackendName::Cuda, true),
            (seismic::BackendName::Vulkan, true),
        ] {
            assert_eq!(
                resource_limits(&service(512, 32), &model, 0, backend)
                    .unwrap()
                    .lookahead,
                expected,
                "{backend:?}"
            );
        }
    }

    #[test]
    fn large_prefill_classes_require_cuda() {
        let service = service(1024, 32);
        assert_eq!(
            resource_limits(&service, &model(0), 0, seismic::BackendName::Cuda)
                .unwrap()
                .max_launch_rows,
            1024
        );
        assert_eq!(
            resource_limits(&service, &model(0), 0, seismic::BackendName::Metal),
            Err(ExecutionPlanningError::LaunchRows {
                required: 1024,
                admitted: 512,
            })
        );
    }
}
