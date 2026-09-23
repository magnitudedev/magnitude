//! One production composition root from an admitted manifest to a ready
//! numerical worker. Every device-bound value is created inside that worker.

use crate::composition::{ReadyEngine, ResolvedEngineConfiguration};
use crate::service::EngineService;
use magnitude_artifacts::Package;
use magnitude_model_batching::MAX_CLASS_ROWS;
use magnitude_model_executor::{
    platform::{self, PlatformConfig},
    AttestedPrograms, ComponentLoader, ComponentSelection, ExecutionPlanner, ExecutorDomain,
    PlannedMethod, ResidencyStore, ResourceAllocator, ResourceBudget, ResourceDomainId,
    ResourceLimits, ResourcePlan, ResourcePlanner,
};
use magnitude_model_state::CodecIdentity;
use magnitude_service::retention::{RetentionKey, TokenizerIdentity};
use seismic::BackendName;
use std::rc::Rc;

pub(crate) fn start(configuration: ResolvedEngineConfiguration) -> Result<ReadyEngine, String> {
    let ResolvedEngineConfiguration {
        artifacts,
        manifest,
        control_capacity,
    } = configuration;
    let package_identity = manifest.package.identity.to_string();
    let method = manifest.model.method.factory(&package_identity)?;
    let retention = (manifest.storage.retention_bytes != 0)
        .then(|| {
            Ok::<_, String>(RetentionKey::new(
                manifest.package.identity,
                TokenizerIdentity::new(artifacts.tokenizer().identity())?,
                CodecIdentity::new(manifest.model.kv_codec.identity())?,
            ))
        })
        .transpose()?;
    let (service, ready) = EngineService::spawn_planned_domain(
        build_native_domain,
        manifest,
        method,
        control_capacity,
        retention,
    )?;
    Ok(ReadyEngine::new(artifacts, service, ready))
}

fn build_native_domain(
    manifest: &crate::options::ExecutionManifest,
) -> Result<(ExecutorDomain, ResourcePlan), String> {
    let package = Rc::new(Package::reopen(&manifest.package).map_err(|error| error.to_string())?);
    let head_enabled = matches!(
        manifest.model.method,
        crate::options::ResolvedMethod::Mtp { .. }
    );
    let vision_enabled = manifest.definition.vision.is_some();
    let selection = ComponentSelection {
        head: head_enabled,
        vision: vision_enabled,
    };
    let max_batch_rows = manifest
        .service
        .prefill_tokens
        .max(manifest.service.decode_tokens);
    if max_batch_rows > MAX_CLASS_ROWS {
        return Err(format!(
            "service policy requires {max_batch_rows} rows but the execution contract admits at most {MAX_CLASS_ROWS}"
        ));
    }
    let discovery = platform::discover().map_err(|error| error.to_string())?;
    let endpoint = platform::select(discovery.topology(), Some(BackendName::Metal))
        .map_err(|error| error.to_string())?;
    let method = match manifest.model.method {
        crate::options::ResolvedMethod::Plain => PlannedMethod::Plain,
        crate::options::ResolvedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
            ..
        } => PlannedMethod::Mtp {
            greedy_proposals,
            sampled_proposals,
        },
    };
    let limits = ResourceLimits {
        active_requests: manifest.service.max_batch,
        in_flight_requests: manifest.service.max_batch,
        // The service exposes checkpoint/fork for plain as well as MTP
        // requests. Reserve the bounded live-request checkpoint class.
        branch_checkpoints: manifest.service.max_batch,
        max_batch_rows,
        max_projected_rows: manifest.service.decode_tokens.max(manifest.service.max_batch),
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
    };
    let budget = ResourceBudget {
        storage_bytes: manifest.storage.storage_bytes,
        retention_bytes: manifest.storage.retention_bytes,
        safety_reserve_bytes: manifest.storage.safety_reserve_bytes,
    };
    let draft = ExecutionPlanner::prepare(
        &endpoint,
        &manifest.package,
        &manifest.definition,
        selection,
        manifest.path,
        method,
        manifest.model.kv_codec,
        limits,
        budget,
    )
    .map_err(|error| error.to_string())?;
    let state = ResourcePlanner::state_plan(
        &manifest.definition,
        draft.load(),
        manifest.model.kv_codec,
        limits,
        budget,
    )?;
    let opened = platform::open_selected(
        &discovery,
        endpoint,
        PlatformConfig {
            path: manifest.path,
            requested_backend: Some(BackendName::Metal),
            storage_bytes: Some(manifest.storage.storage_bytes),
            requirements: platform::MemoryRequirements::default(),
        },
    )
    .map_err(|error| error.to_string())?;
    let mut programs = AttestedPrograms::prepare_draft(&draft, opened.device())
        .map_err(|error| error.to_string())?;
    let target_graphs = programs.prepare_target_graphs(
        opened.device(), draft.load(), &manifest.definition.geometry, &state, limits,
    )?;
    let target_readout_graphs = programs.prepare_target_readout_graphs(
        opened.device(), draft.load(), &manifest.definition.geometry, limits,
    )?;
    programs.prepare_auxiliary_graphs(
        opened.device(), draft.load(), &manifest.definition,
        state.target_state(), state.head_state(), limits,
    )?;
    let resources = ResourcePlanner::plan_with_state(
        state, &target_graphs, &target_readout_graphs,
        programs.head_graphs().map(|graphs| graphs.as_ref()),
        programs.vision_graphs().map(|graphs| graphs.as_ref()),
        programs.state_graphs().ok_or("prepared program set has no state graphs")?.as_ref(),
    )?;
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let execution_plan = draft.admit(resources).map_err(|error| error.to_string())?;
    let plan = execution_plan.resources().clone();
    let qualified = opened
        .admit(&discovery, plan.memory_requirements(), programs)
        .map_err(|error| error.to_string())?;
    if qualified.plan.endpoint.backend != execution_plan.device().backend()
        || qualified.plan.endpoint.ordinal != execution_plan.device().ordinal()
        || qualified.plan.endpoint.name != execution_plan.device().name()
    {
        return Err("qualified device differs from the selected execution plan".into());
    }
    let resource_identity = ResourceDomainId::new(format!(
        "{}:{}:{}",
        manifest.package.identity, qualified.plan.endpoint.name, qualified.plan.endpoint.ordinal,
    ))?;
    let device = Rc::new(qualified.device);
    let programs = Rc::new(qualified.programs);
    let target_graphs = programs
        .target_graphs()
        .ok_or("qualified program set has no target graphs")?;
    let target_readout_graphs = programs
        .target_readout_graphs()
        .ok_or("qualified program set has no target readout graphs")?;
    let state_graphs = programs
        .state_graphs()
        .ok_or("qualified program set has no state graphs")?;
    let resources = ResourceAllocator::allocate(
        &execution_plan,
        &device,
        resource_identity.clone(),
        target_graphs,
        target_readout_graphs,
        programs.head_graphs().map(|graphs| graphs.as_ref()),
        programs.vision_graphs().map(|graphs| graphs.as_ref()),
        state_graphs.as_ref(),
    )
    .map_err(|error| error.to_string())?;
    let mut residency = ResidencyStore::new(
        device.clone(),
        programs.clone(),
        execution_plan.clone(),
        resource_identity.clone(),
    )
    .map_err(|error| error.to_string())?;
    let target = residency
        .load_target(&manifest.definition, &package)
        .map_err(|error| error.to_string())?;
    let definition = Rc::new(manifest.definition.clone());
    let head_loader = definition
        .head
        .as_ref()
        .map(|_| ComponentLoader::head(residency, definition.clone(), package.clone()))
        .transpose()
        .map_err(|error| error.to_string())?;
    let vision_loader = definition
        .vision
        .as_ref()
        .map(|_| {
            let store = ResidencyStore::new(
                device.clone(),
                programs.clone(),
                execution_plan.clone(),
                resource_identity.clone(),
            )?;
            ComponentLoader::vision(store, definition.clone(), package.clone())
        })
        .transpose()
        .map_err(|error| error.to_string())?;
    let target_state = plan.allocate_target_state(device.clone())?;
    let head_state = plan.allocate_head_state(device.clone())?;
    let domain = ExecutorDomain::new(
        Rc::new(execution_plan),
        definition,
        device,
        programs,
        resources,
        head_loader,
        vision_loader,
        target,
        target_state,
        head_state,
    )?;
    Ok((domain, plan))
}
