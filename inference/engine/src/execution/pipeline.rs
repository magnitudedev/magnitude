//! Explicit paired construction, using ordinary device-local preparation.
use super::*;
use crate::error::UnsupportedModel;
use crate::options::ExplicitPipeline;
use crate::options::{ResolvedMethod, ResolvedModelPolicy};
use magnitude_executor::{
    pipeline::{PipelineModel, PipelineRefusal, StageResources, TwoStageCudaPipeline},
    ExecutionPath, PipelineNativeFamily,
};
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;

/// One ordinary domain owns both unique allocation authorities. Local plans
/// and identities stay separate so readiness never reports a pooled budget.
pub(crate) struct PipelineDomain {
    pub domain: ExecutorDomain<PipelineNativeFamily>,
    pub bindings: StateBindings<PipelineNativeFamily>,
    pub plans: [ResourcePlan; 2],
    pub devices: [DeviceSelector; 2],
    pub pools: [MemoryPoolKind; 2],
    pub allocation_pools: [seismic::MemoryPoolId; 2],
}

fn unsupported(reason: impl Into<String>) -> LoadError {
    LoadError::Unsupported(UnsupportedModel::Representation {
        reason: reason.into(),
    })
}
fn profile(
    path: ExecutionPath,
    model: &ResolvedModelPolicy,
    service: &ServiceLimits,
) -> Result<(), LoadError> {
    service.validate().map_err(unsupported)?;
    if path != ExecutionPath::Native
        || model.method != ResolvedMethod::Plain
        || model.kv_codec != KvCodec::Dense
        || model.lookahead
        || model.exported_logits_rows > 1
        || !(1..=2).contains(&service.prefill_tokens)
        || service.decode_tokens != 1
    {
        return Err(unsupported("explicit pipeline requires native plain dense-KV execution, one/two-row prefill, one-row decode and no lookahead"));
    }
    Ok(())
}
fn classify_refusal(error: PipelineRefusal, domain: MemoryDomain) -> LoadError {
    match error {
        PipelineRefusal::Plan(magnitude_executor::PlanError::Resource(capacity))
            if matches!(
                capacity.resource,
                magnitude_executor::ResourceKind::DeviceMemory
                    | magnitude_executor::ResourceKind::HostStaging
            ) =>
        {
            LoadError::InsufficientMemory {
                purpose: "pipeline stage plan".into(),
                domain: if capacity.resource == magnitude_executor::ResourceKind::HostStaging {
                    MemoryDomain::HostRam
                } else {
                    domain
                },
                memory: InsufficientMemory {
                    required: capacity.required,
                    available: capacity.available,
                },
            }
        }
        PipelineRefusal::Plan(error) => classify_plan(error, BackendName::Cuda).into(),
        PipelineRefusal::Memory {
            refusal,
            allocation,
            staged,
        } => classify_claim(
            domain,
            "pipeline stage startup",
            refusal,
            |role| match role {
                DomainRole::Allocation => allocation,
                DomainRole::Staging => staged,
            },
        ),
        PipelineRefusal::Preparation(reason) => internal(reason),
        error => unsupported(error.to_string()),
    }
}

pub(crate) fn build(
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    placement: ExplicitPipeline,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<PipelineDomain, LoadError> {
    // Metadata admission happens before opening either device. No weights are
    // imported by prepare: each stage below imports only its original roles.
    profile(manifest.path, &manifest.model, &manifest.service)?;
    let intent = magnitude_executor::placement::ModelPlacement::pipeline(
        placement.devices.into_iter().zip([
            0..placement.split,
            placement.split..manifest.definition.decoder.blocks.len(),
        ]),
    );
    let (model, devices) =
        PipelineModel::from_placement(Rc::new(manifest.definition.clone()), &intent)
            .map_err(|error| classify_refusal(error, MemoryDomain::HostRam))?;
    TwoStageCudaPipeline::qualify(&model, &devices)
        .map_err(|error| classify_refusal(error, MemoryDomain::HostRam))?;
    let mut stages = Vec::with_capacity(2);
    let mut pools = Vec::with_capacity(2);
    let mut allocation_pools = Vec::with_capacity(2);
    for (assignment, selector) in model.stages().zip(devices) {
        let mut local = manifest.clone();
        local.device = platform::DeviceRequest::Selector(selector);
        let catalog = DeviceCatalog::discover().map_err(|error| internal(error.to_string()))?;
        let PreparedPrograms {
            draft,
            opened,
            programs,
            pool,
            capacity,
            ..
        } = prepare_with_plan(
            &catalog,
            &local,
            &package,
            progress.clone(),
            crate::planning::plan_pipeline_execution,
        )?;
        let device = Rc::new(opened.into_device());
        let DeviceMemory::Established(memory) = &device.info().memory else {
            return Err(internal("pipeline device memory identity is unavailable"));
        };
        allocation_pools.push(memory.allocation_pool);
        let heap = DeviceHeap::open(catalog, manifest.reserves, device)
            .map_err(|error| platform_error(PlatformError::Memory(error)))?;
        let state = ResourcePlanner::stage_state_plan(
            &manifest.definition,
            draft.load(),
            manifest.model.kv_codec,
            draft.policy().limits(),
            capacity,
            assignment.view().global_range(),
        )
        .map_err(internal)?;
        let domain = MemoryDomain::of(selector, pool);
        stages.push(
            StageResources::prepare(
                assignment,
                Rc::new(programs),
                draft,
                heap,
                state,
                ResourceDomainId::new(format!("{}:{selector}", manifest.package.identity))
                    .map_err(internal)?,
                &package,
            )
            .map_err(|error| classify_refusal(error, domain))?,
        );
        pools.push(pool);
    }
    let stages = stages
        .try_into()
        .unwrap_or_else(|_| unreachable!("qualified two stages"));
    let (executor, allocations) = TwoStageCudaPipeline::new(stages)
        .map_err(|error| classify_refusal(error, MemoryDomain::HostRam))?;
    let plans = allocations.each_ref().map(|a| a.resource_plan().clone());
    let (mut domain, bindings) = PipelineNativeFamily::new(executor)
        .into_domain(allocations)
        .map_err(|error| classify_refusal(error, MemoryDomain::HostRam))?;
    let bindings = warm_up(&mut domain, bindings)?;
    Ok(PipelineDomain {
        domain,
        bindings,
        plans,
        devices: placement.devices,
        allocation_pools: allocation_pools
            .try_into()
            .unwrap_or_else(|_| unreachable!("two allocation pools")),
        pools: pools
            .try_into()
            .unwrap_or_else(|_| unreachable!("qualified two pools")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy() -> ResolvedModelPolicy {
        ResolvedModelPolicy {
            method: ResolvedMethod::Plain,
            kv_codec: KvCodec::Dense,
            lookahead: false,
            exported_logits_rows: 1,
        }
    }
    fn limits() -> ServiceLimits {
        ServiceLimits {
            prefill_tokens: 2,
            decode_tokens: 1,
            decode_share: 0.5,
            locality_seconds: 1.0,
        }
    }
    #[test]
    fn bounded_profile_is_explicit_and_refuses_unsupported_policies() {
        assert!(profile(ExecutionPath::Native, &policy(), &limits()).is_ok());
        assert!(profile(ExecutionPath::Planned, &policy(), &limits()).is_err());
        for change in 0..7 {
            let mut model = policy();
            let mut service = limits();
            match change {
                0 => model.lookahead = true,
                1 => model.kv_codec = KvCodec::AffineK8V4,
                2 => model.exported_logits_rows = 2,
                3 => service.prefill_tokens = 3,
                4 => service.decode_tokens = 2,
                5 => {
                    model.method = ResolvedMethod::Mtp {
                        greedy_proposals: 1,
                        sampled_proposals: 1,
                    }
                }
                _ => service.prefill_tokens = 0,
            }
            assert!(matches!(
                profile(ExecutionPath::Native, &model, &service),
                Err(LoadError::Unsupported(_))
            ));
        }
    }
    #[test]
    fn stage_plan_capacity_is_not_misclassified_as_internal_or_unsupported() {
        let domain = MemoryDomain::DeviceLocal {
            device: DeviceSelector::Cuda { uuid: [2; 16] },
        };
        let error = classify_refusal(
            PipelineRefusal::Plan(magnitude_executor::PlanError::Resource(
                magnitude_executor::CapacityError {
                    resource: magnitude_executor::ResourceKind::DeviceMemory,
                    required: 100,
                    available: 90,
                },
            )),
            domain,
        );
        assert!(
            matches!(error, LoadError::InsufficientMemory { domain: actual,
            memory: InsufficientMemory { required: 100, available: 90 }, .. } if actual == domain)
        );
    }
    #[test]
    fn local_memory_refusal_keeps_its_device_and_requested_charge() {
        let domain = MemoryDomain::DeviceLocal {
            device: DeviceSelector::Cuda { uuid: [1; 16] },
        };
        let error = classify_refusal(
            PipelineRefusal::Memory {
                refusal: ClaimRefusal::Reclaim {
                    role: DomainRole::Allocation,
                },
                allocation: 123,
                staged: 456,
            },
            domain,
        );
        assert!(
            matches!(error, LoadError::InsufficientMemory { domain: actual,
            memory: InsufficientMemory { required: 123, available: 0 }, .. } if actual == domain)
        );
        let error = classify_refusal(
            PipelineRefusal::Memory {
                refusal: ClaimRefusal::Reclaim {
                    role: DomainRole::Staging,
                },
                allocation: 123,
                staged: 456,
            },
            domain,
        );
        assert!(matches!(
            error,
            LoadError::InsufficientMemory {
                domain: MemoryDomain::HostRam,
                memory: InsufficientMemory {
                    required: 456,
                    available: 0
                },
                ..
            }
        ));
    }
}

#[cfg(test)]
mod qualification;
