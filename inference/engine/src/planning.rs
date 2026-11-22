//! The one derivation of a model's execution plan draft from a resolved
//! execution manifest and a selected device. The production load and
//! metadata-only assessment both plan through here, so assessment sees
//! exactly the components, method, codec and resource limits a load uses.
//!
//! Only the manifest's device-free facts are read: the package manifest's
//! tensor directory (a header-only open is sufficient), the model
//! definition, the resolved model policy and the service limits.

use crate::assessment::Measurements;
use crate::options::{DraftPolicy, ExecutionManifest, ResolvedMethod, ResolvedModelPolicy};
use magnitude_batching::MAX_CLASS_ROWS;
use magnitude_executor::{
    image_cell_limit, max_image_cells,
    platform::{self, PlatformError, SelectedDevice},
    BackendPlan, ComponentSelection,
    DraftWidths,
    ChargedDraft, ExecutionPlanDraft, ExecutionPlanner, HeldCharge, MemoryBudget, PlanError, PlanForm, PlanPrices, Placed, PlannedMethod, ResourceLimits,
    assessment::{launch_prices, ExecutorCosts, Priced},
};
use magnitude_scheduler::ServiceLimits;
use seismic::Devices;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
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
    platform::select_recorded(catalog, manifest.path, manifest.device, &manifest.release, manifest.kernel_cache.as_deref())
}

/// Plan the manifest's model on the selected device using the static stable
/// capacity budget ([`MemoryBudget::of_selected_stable`]).
pub fn plan_execution<C: HeldCharge>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ChargedDraft<C>, ExecutionPlanningError> {
    let budget = MemoryBudget::of_selected_stable(selected).map_err(ExecutionPlanningError::Plan)?;
    plan_resident(manifest, selected).and_then(|resident| {
        plan_banked(manifest, selected, resident, manifest.plan_form, &budget)
            .map(|(draft, _)| draft)
    })
}

/// [`plan_execution`]'s plan before its expert bank: every expert resident.
pub fn plan_resident(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
) -> Result<ExecutionPlanDraft, ExecutionPlanningError> {
    let (selection, method, limits) = planner_inputs(manifest, selected.info.backend())?;
    let resident = ExecutionPlanner::prepare(
        selected,
        &manifest.package,
        &manifest.definition,
        selection,
        manifest.path,
        method,
        manifest.model.kv_codec,
        limits,
    )
    .map_err(ExecutionPlanningError::Plan)?;
    Ok(match manifest.model.record_routes {
        true => resident.recording_routes(),
        false => resident,
    })
}

/// Plan `resident` against `budget`, honoring `form` when decided:
/// - `None`: decide the form against `budget` (whole if it fits, else streamed)
///   and return `(draft, form)`.
/// - `Some(PlanForm::Whole)`: require a whole plan fitting `budget`; fail
///   immediately with `CapacityError` if not.
/// - `Some(PlanForm::Streamed)`: plan a streamed bank sized to `budget`.
pub fn plan_banked<C: HeldCharge>(
    manifest: &ExecutionManifest,
    selected: &SelectedDevice,
    resident: ExecutionPlanDraft,
    form: Option<PlanForm>,
    budget: &MemoryBudget,
) -> Result<(ChargedDraft<C>, PlanForm), ExecutionPlanningError> {
    let measurements = match manifest.kernel_cache.as_deref() {
        None => Measurements::Unavailable {
            reason: "the load keeps no kernel cache, which holds the machine's figures".into(),
        },
        Some(kernels) => crate::assessment::stored_measurements(
            &selected.reading.topology,
            &selected.info,
            kernels,
            manifest.store(),
        ),
    };
    if let Measurements::Pending(pending) = measurements {
        return Err(ExecutionPlanningError::Plan(PlanError::Pending(pending)));
    }
    let prices = plan_prices(&measurements, &resident);
    let check_recording = |whole: &ChargedDraft<C>| -> Result<(), ExecutionPlanningError> {
        // A recording load reads its routing from its bank's routing
        // records, which a load of no bank keeps none of.
        if manifest.model.record_routes && whole.draft().load().expert_bank().is_none() {
            return Err(ExecutionPlanningError::Plan(PlanError::Topology(
                "a load recording its routes holds its experts in a bank: it reads them from the bank's routing records",
            )));
        }
        Ok(())
    };
    let short = match form {
        None => match resident
            .placed(&manifest.definition, manifest.model.weight_bound, &prices, budget)
            .map_err(ExecutionPlanningError::Plan)?
        {
            Placed::Whole(whole) => {
                check_recording(&whole)?;
                return Ok((whole, PlanForm::Whole));
            }
            Placed::Short(short) => short,
        },
        Some(PlanForm::Whole) => match resident
            .placed(&manifest.definition, manifest.model.weight_bound, &prices, budget)
            .map_err(ExecutionPlanningError::Plan)?
        {
            Placed::Whole(whole) => {
                check_recording(&whole)?;
                return Ok((whole, PlanForm::Whole));
            }
            Placed::Short(short) => {
                return Err(ExecutionPlanningError::Plan(short.whole_short_error()));
            }
        },
        Some(PlanForm::Streamed) => resident
            .streaming(&manifest.definition, manifest.model.weight_bound, budget)
            .map_err(ExecutionPlanningError::Plan)?,
    };
    let streamed = short
        .streamed(&manifest.definition, &prices, budget)
        .map_err(ExecutionPlanningError::Plan)?;
    Ok((streamed, PlanForm::Streamed))
}

/// The prices a plan of `definition` as `plan` (its plan before any bank)
/// settles its held pass on (`PlanPrices`): the executor's costs and the
/// streamed figures `measurements` hold, and its decode on those costs in
/// each form, `plan`'s and the banked plan's streamed form's
/// (`assessment::launch_prices`, `LaunchPrices::launch`). Pending or unavailable as the measurements
/// or their streamed figures are: a plan that holds a bank waits for them.
pub fn plan_prices(measurements: &Measurements, plan: &ExecutionPlanDraft) -> Priced<PlanPrices> {
    match measurements {
        Measurements::Pending(pending) => Priced::Pending(pending.clone()),
        Measurements::Unavailable { reason } => Priced::Unavailable(reason.clone()),
        Measurements::Measured { costs, streamed, .. } => streamed.as_ref().and_then(|figures| {
            let (resident, steps) = (plan.clone(), ExecutorCosts::clone(costs));
            Priced::Measured(PlanPrices::new(
                ExecutorCosts::clone(costs),
                *figures,
                Box::new(move |definition, banked| {
                    let prices = launch_prices(definition, &resident, banked, &steps)?;
                    Ok(Box::new(move |rows, depth| Ok(prices.launch(rows, depth)?)))
                }),
            ))
        }),
    }
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
        max_launch_slots: service.steps().launch_slots(),
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
            weight_bound: None,
            record_routes: false,
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

    /// `plan_banked` honors an already-decided `PlanForm`: `Some(PlanForm::Whole)`
    /// refuses with `PlanError::Resource` when the live budget cannot hold the
    /// whole model instead of silently toggling to streamed, and
    /// `Some(PlanForm::Streamed)` stays streamed even when whole would fit.
    #[test]
    fn plan_banked_honors_decided_plan_form_without_toggling() {
        use magnitude_executor::{
            AssessmentMemoryCharge, ExecutionPath, MemoryBudget, PlanForm,
            platform::{DeviceRequest, ReleaseRate},
        };
        let cache = tempfile::tempdir().unwrap();
        let definition = magnitude_executor::fixtures::tiny_definition();
        let model = crate::options::ModelPolicy::default().resolve(&definition).unwrap();
        let manifest = ExecutionManifest::new(
            magnitude_executor::fixtures::tiny_manifest(&definition),
            definition,
            model,
            crate::options::standard_service_limits(),
            ExecutionPath::Native,
            DeviceRequest::Backend(seismic::BackendName::Cpu),
            Some(cache.path().to_owned()),
            Some(cache.path().to_owned()),
            ReleaseRate::of_machine(None),
        )
        .unwrap();
        let catalog = Devices::discover(&[]).unwrap();
        let selected = select_load_device(&catalog, &manifest).unwrap();
        crate::execution::store_fixed_costs(&catalog.topology(), &selected.info, cache.path());
        let resident = plan_resident(&manifest, &selected).unwrap();

        // With generous budget, Whole succeeds and returns PlanForm::Whole.
        let generous = MemoryBudget::fixed(16 << 30, None);
        let (_, form) = plan_banked::<AssessmentMemoryCharge>(
            &manifest,
            &selected,
            resident.clone(),
            Some(PlanForm::Whole),
            &generous,
        )
        .unwrap();
        assert_eq!(form, PlanForm::Whole);

        // With a tiny live budget, Whole fails immediately with PlanError::Resource.
        let tight = MemoryBudget::fixed(1024, None);
        let err = plan_banked::<AssessmentMemoryCharge>(
            &manifest,
            &selected,
            resident,
            Some(PlanForm::Whole),
            &tight,
        )
        .err()
        .expect("PlanForm::Whole must fail when tight budget cannot hold it");
        assert!(
            matches!(err, ExecutionPlanningError::Plan(PlanError::Resource(ref c)) if c.available == 1024 && c.required > 1024),
            "expected PlanError::Resource with available=1024, got {err:?}"
        );
    }
}

