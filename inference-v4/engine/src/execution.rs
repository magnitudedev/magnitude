//! One production composition root from an admitted manifest to a ready
//! numerical worker. Every device-bound value is created inside that worker.

use crate::composition::{ReadyEngine, ResolvedEngineConfiguration};
use crate::service::EngineService;
use magnitude_artifacts::Package;
use magnitude_chat::PreparedVocabulary;
use magnitude_model_batching::Demand;
use magnitude_model_executor::{
    platform::{self, PlatformConfig},
    AttestedPrograms, ComponentLoader, ExecutorDomain, KernelCache, Operation, RequestId,
    ReservedResources, ResidencyStore, ResourceAllocator, ResourceCapacity, ResourceDomainId,
    ResourcePlan, ResourcePlanner, TokenId, TuningContext, TuningEvent, TuningObserver,
    TuningOrigin, WorkKind, DEFAULT_KERNEL_CACHE_BYTES,
};
use magnitude_model_state::CodecIdentity;
use magnitude_service::retention::{RetentionKey, TokenizerIdentity};
use seismic::DeviceCatalog;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn start(configuration: ResolvedEngineConfiguration) -> Result<ReadyEngine, String> {
    let ResolvedEngineConfiguration {
        artifacts,
        manifest,
        control_capacity,
    } = configuration;
    let package_identity = manifest.package.identity.to_string();
    let package = artifacts.shared_package();
    let method = manifest.model.method.factory(&package_identity)?;
    let retention_identity = manifest.package.identity;
    let retention_codec = manifest.model.kv_codec;
    let (service, ready, (artifacts, vocabulary)) = EngineService::spawn_planned_domain(
        move |manifest| build_native_domain(manifest, package),
        manifest,
        method,
        control_capacity,
        move || {
            let artifacts = artifacts.finish()?;
            let vocabulary = PreparedVocabulary::new(
                artifacts.shared_tokenizer(),
                usize::try_from(artifacts.definition().geometry.vocabulary)
                    .map_err(|_| "model vocabulary exceeds host domain")?,
            )?;
            let retention = RetentionKey::new(
                retention_identity,
                TokenizerIdentity::new(artifacts.tokenizer().identity())?,
                CodecIdentity::new(retention_codec.identity())?,
            );
            Ok(((artifacts, vocabulary), Some(retention)))
        },
    )?;
    Ok(ReadyEngine::new(artifacts, vocabulary, service, ready))
}

/// Construct the numerical worker's executor domain from an admitted
/// manifest. The service calls this inside its worker; measurement tools call
/// it directly to drive the domain below the service.
pub fn build_native_domain(
    manifest: &crate::options::ExecutionManifest,
    package: Arc<Package>,
) -> Result<(ExecutorDomain, ResourcePlan), String> {
    let mut phase_started = Instant::now();
    if package.manifest() != manifest.package {
        return Err("opened package differs from the execution manifest".into());
    }
    // This runs inside the numerical worker: its own Seismic catalog and its
    // own process-scoped observations decide selection and admission.
    let catalog = DeviceCatalog::discover().map_err(|error| error.to_string())?;
    let reserves = manifest.reserves;
    let selected = platform::select_device(&catalog, manifest.path, manifest.device, &reserves)
        .map_err(|error| error.to_string())?;
    let capacity_bytes = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes,
    };
    // The same derivation metadata-only assessment plans through.
    let draft = crate::planning::plan_execution(manifest, &selected)
        .map_err(|error| error.to_string())?;
    let limits = draft.policy().limits();
    let head_enabled = draft.policy().selection().head;
    let state = ResourcePlanner::state_plan(
        &manifest.definition,
        draft.load(),
        draft.policy().method(),
        manifest.model.kv_codec,
        limits,
        capacity_bytes,
    )?;
    let kernel_cache = manifest
        .kernel_cache
        .clone()
        .map(|root| KernelCache::open(root, DEFAULT_KERNEL_CACHE_BYTES).map(Arc::new))
        .transpose()
        .map_err(|error| error.to_string())?;
    report_load_phase("device selection and planning", &mut phase_started);
    let opened = platform::open_selected(
        &catalog,
        draft.device().selector(),
        PlatformConfig {
            path: manifest.path,
            artifacts: kernel_cache
                .clone()
                .map(|cache| cache as Arc<dyn seismic::ArtifactStore>),
            reserves,
        },
    )
    .map_err(|error| error.to_string())?;
    report_load_phase("device open", &mut phase_started);
    let preparing = Instant::now();
    let mut programs = AttestedPrograms::prepare_draft(
        &draft,
        opened.device(),
        TuningContext {
            definition: &manifest.definition,
            weights: package.as_ref(),
            observer: &LoadProgress,
            cache: kernel_cache.as_deref(),
        },
    )
    .map_err(|error| error.to_string())?;
    let tuned = programs.tuned();
    eprintln!(
        "magnitude-engine: prepared programs in {:.2} s, {:.2} s of it tuning {} entries \
         ({} searched, {} stored; forming {:.2} s, measuring {:.2} s, validating {:.2} s)",
        preparing.elapsed().as_secs_f64(),
        tuned.iter().map(|tuned| tuned.seconds).sum::<f64>(),
        tuned.len(),
        tuned
            .iter()
            .filter(|tuned| tuned.origin == TuningOrigin::Searched)
            .count(),
        tuned
            .iter()
            .filter(|tuned| tuned.origin == TuningOrigin::Stored)
            .count(),
        tuned
            .iter()
            .map(|tuned| tuned.time.forming_seconds)
            .sum::<f64>(),
        tuned
            .iter()
            .map(|tuned| tuned.time.measuring_seconds)
            .sum::<f64>(),
        tuned
            .iter()
            .map(|tuned| tuned.time.validating_seconds)
            .sum::<f64>(),
    );
    phase_started = Instant::now();
    let target_graphs = programs.prepare_target_graphs(
        opened.device(),
        draft.load(),
        &manifest.definition.geometry,
        &state,
        limits,
    )?;
    let target_readout_graphs = programs.prepare_target_readout_graphs(
        opened.device(),
        draft.load(),
        &manifest.definition.geometry,
        limits,
    )?;
    programs.prepare_auxiliary_graphs(
        opened.device(),
        draft.load(),
        &manifest.definition,
        state.target_state(),
        state.head_state(),
        limits,
        manifest.model.method.proposals(),
    )?;
    let resources = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        programs.head_graphs().map(|graphs| graphs.as_ref()),
        programs.vision_graphs().map(|graphs| graphs.as_ref()),
        programs
            .state_graphs()
            .ok_or("prepared program set has no state graphs")?
            .as_ref(),
    )?;
    let seal = target_graphs.seal_report();
    eprintln!(
        "magnitude-engine: sealed {} target graph classes ({} graphs) in {:.2} s",
        seal.classes, seal.sealed_graphs, seal.seconds
    );
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let execution_plan = draft.admit(resources).map_err(|error| error.to_string())?;
    let plan = execution_plan.resources().clone();
    report_load_phase("graph and resource planning", &mut phase_started);
    if opened.selector() != execution_plan.device().selector() {
        return Err("opened device differs from the selected execution plan".into());
    }
    let resource_identity = ResourceDomainId::new(format!(
        "{}:{}",
        manifest.package.identity,
        opened.selector(),
    ))?;
    let device = Rc::new(opened.into_device());
    let programs = Rc::new(programs);
    let target_graphs = programs
        .target_graphs()
        .ok_or("qualified program set has no target graphs")?;
    let target_binding_constants = target_graphs.binding_constant_bytes()?;
    let target_readout_graphs = programs
        .target_readout_graphs()
        .ok_or("qualified program set has no target readout graphs")?;
    let state_graphs = programs
        .state_graphs()
        .ok_or("qualified program set has no state graphs")?;
    let startup = StartupClaims {
        catalog: &catalog,
        device: &device,
        reserves: &reserves,
    };
    startup.claim("graph scratch", plan.bytes().scratch, 0)?;
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
    report_load_phase("resource allocation", &mut phase_started);
    let mut residency = ResidencyStore::new(
        device.clone(),
        programs.clone(),
        execution_plan.clone(),
        resource_identity.clone(),
    )
    .map_err(|error| error.to_string())?;
    let target_upload = execution_plan.load().target_upload_peak_bytes()?;
    let target_peak = plan
        .bytes()
        .target_weights
        .checked_add(target_upload)
        .ok_or("target import peak byte count overflow")?;
    startup.claim("target import", target_peak, target_upload)?;
    let target = residency
        .load_target(&manifest.definition, &package)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "magnitude-engine: resident target imported in {:.2} s ({} distinct weights)",
        phase_started.elapsed().as_secs_f64(),
        residency.len(),
    );
    let import = residency.mapped_import_report();
    if import.windows != 0 {
        eprintln!(
            "magnitude-engine: mapped import {} weights in {} windows: map {:.3} s, prepare {:.3} s, submit {:.3} s, wait {:.3} s, publish {:.3} s",
            import.weights,
            import.windows,
            import.mapping.as_secs_f64(),
            import.preparing.as_secs_f64(),
            import.submitting.as_secs_f64(),
            import.waiting.as_secs_f64(),
            import.publishing.as_secs_f64(),
        );
    }
    phase_started = Instant::now();
    let definition = Rc::new(manifest.definition.clone());
    let head_loader = head_enabled
        .then(|| ComponentLoader::head(residency, definition.clone(), package.clone()))
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
    startup.claim(
        "initial target state",
        plan.target_state().initial_committed_bytes()?,
        0,
    )?;
    let target_state = plan.allocate_target_state(device.clone())?;
    let head_state = if let Some(state) = plan.head_state() {
        startup.claim("initial head state", state.initial_committed_bytes()?, 0)?;
        Some(state.allocate(device.clone())?)
    } else {
        None
    };
    startup.claim("target binding constants", target_binding_constants, 0)?;
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
    let mut domain = domain;
    domain.install_memory_policy(catalog, reserves)?;
    domain
        .register_allocated_holdings()
        .map_err(|error| format!("register allocated memory holdings: {error}"))?;
    report_load_phase("state and domain allocation", &mut phase_started);
    warm_up(&mut domain)?;
    Ok((domain, plan))
}

fn report_load_phase(name: &str, started: &mut Instant) {
    eprintln!(
        "magnitude-engine: {name} in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    *started = Instant::now();
}

/// Startup allocations claimed against fresh readings of every domain the
/// device uses, each keeping headroom above its planning reserve.
struct StartupClaims<'a> {
    catalog: &'a DeviceCatalog,
    device: &'a seismic::Device,
    reserves: &'a platform::MemoryReserves,
}

impl StartupClaims<'_> {
    /// Claim `allocation` additional bytes of the device's allocation domain,
    /// `staged` of which a dedicated device also stages through host RAM. A
    /// host-backed device has no staging domain: its staged bytes are part
    /// of `allocation`. Refreshes Seismic's enforced ceiling.
    fn claim(&self, purpose: &str, allocation: u64, staged: u64) -> Result<(), String> {
        let readings = platform::refresh_device_ceiling(self.catalog, self.device, self.reserves)
            .map_err(|error| error.to_string())?;
        for reading in readings {
            let required = match reading.role {
                platform::DomainRole::Allocation => allocation,
                platform::DomainRole::Staging => staged,
            };
            if required > reading.ceiling_bytes {
                return Err(format!(
                    "{purpose} requires {required} bytes of {}; {} are available above the \
                     {}-byte planning reserve ({} bytes of headroom)",
                    reading.constraint,
                    reading.ceiling_bytes,
                    reading.thresholds.planning_bytes,
                    reading.headroom_bytes,
                ));
            }
        }
        Ok(())
    }
}

/// Run one throwaway single-row forward before readiness, so a broken device
/// path fails the load instead of the first request, and the process's
/// one-time first-forward cost is paid here. Tuning has already executed every
/// kernel, and no row class carries its own first-use cost, so one row
/// suffices. The request's state advance is aborted, so no state survives it.
fn warm_up(domain: &mut ExecutorDomain) -> Result<(), String> {
    let began = std::time::Instant::now();
    let request = RequestId(u64::MAX);
    let failed = |error: String| format!("load warm-up forward: {error}");
    domain
        .open(request)
        .map_err(|error| failed(error.to_string()))?;
    let operations = [Operation::Forward {
        request,
        kind: WorkKind::Replay,
        tokens: vec![TokenId(0)],
        position: 0,
        conditioning: None,
        demand: Demand::NONE,
        select: Vec::new(),
        committed: 1,
    }];
    let resources = domain
        .reserve(&operations)
        .map_err(|error| failed(error.to_string()))?
        .into_resources();
    let ReservedResources::Target(reservation) = resources else {
        return Err(failed("reserved a non-target lane".into()));
    };
    let flight = domain
        .submit_target(&operations, reservation)
        .map_err(|error| failed(error.to_string()))?;
    for pending in domain
        .finish_target(flight)
        .map_err(|error| failed(error.to_string()))?
    {
        domain
            .abort(pending)
            .map_err(|error| failed(error.to_string()))?;
    }
    domain.close(request)?;
    eprintln!(
        "magnitude-engine: warm-up forward in {:.0} ms",
        began.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Reports tuning at load on the engine's diagnostic stream while the worker
/// is not yet ready.
struct LoadProgress;

impl TuningObserver for LoadProgress {
    fn event(&self, event: &TuningEvent) {
        match event {
            TuningEvent::Started {
                entry,
                bindings,
                configurations,
                points,
            } => eprintln!(
                "magnitude-engine: tuning {entry} [{bindings}]: {configurations} configurations at {points} points"
            ),
            TuningEvent::Finished(tuned) => {
                let origin = match (tuned.origin, tuned.search) {
                    (TuningOrigin::Stored, _) => "stored".to_owned(),
                    (TuningOrigin::Searched, Some((budget, stop))) => {
                        format!("budget {budget}, stop {stop:?}")
                    }
                    (TuningOrigin::Searched, None) => "surveyed".to_owned(),
                };
                eprintln!(
                    "magnitude-engine: tuned {} [{}] in {:.2} s ({origin}): {:?} ({} measured, {} excluded, {} defects)",
                    tuned.entry,
                    tuned.bindings,
                    tuned.seconds,
                    tuned.overall.params,
                    tuned.measured,
                    tuned.excluded,
                    tuned.defects
                );
                if tuned.search.is_some_and(|(_, stop)| stop == seismic::SearchStop::Expired) {
                    eprintln!(
                        "magnitude-engine: warning: tuning reached its safety stop; {} [{}] keeps the best configuration found and is not stored",
                        tuned.entry, tuned.bindings
                    );
                }
                if let Some(defect) = &tuned.first_defect {
                    eprintln!("magnitude-engine:   first defect of {}: {defect}", tuned.entry);
                }
            }
        }
    }
}
