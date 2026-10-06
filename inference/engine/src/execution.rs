//! One production composition root from an admitted manifest to a ready
//! executor domain. Every device-bound value is created inside the numerical
//! worker that runs this.

use crate::census::MemoryDomain;
use crate::error::{
    classify_catalog, classify_plan, classify_platform, ArtifactError, InsufficientMemory,
    LoadError,
};
use crate::options::ExecutionManifest;
use crate::worker::protocol::LoadProgress;
use magnitude_artifacts::Package;
use magnitude_batching::Demand;
use magnitude_executor::{
    memory::{ClaimId, HoldingClass},
    platform::{self, DomainRole, OpenedPlatform, PlatformConfig, PlatformError},
    require_normal,
    AttestedPrograms, ExecutionPlanDraft, ClaimRefusal, ComponentLoader, DeviceHeap, ExecutorDomain, KernelCache,
    Operation, RequestId, ReservedResources, ResidencyError, ResidencyStore, ResourceAllocator,
    ResourceCapacity, ResourceDomainId, ResourcePlan, ResourcePlanner, StateBindings, TokenId,
    TuningContext, TuningEvent,
    TuningObserver, TuningOrigin, WeightImportError, WorkKind, DEFAULT_KERNEL_CACHE_BYTES,
};
use magnitude_family_contracts::{InputLayout, PreparedModelInput, TokenPlan};
use seismic::{BackendName, DeviceCatalog, DeviceMemory, DeviceSelector, MemoryPoolKind};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How often a weight import observes its memory domains' band. The import
/// runs under one claim for as long as the weights take to become resident.
const IMPORT_OBSERVATION_INTERVAL: Duration = Duration::from_millis(100);

/// A constructed executor domain with the facts readiness reports.
pub(crate) struct NativeDomain {
    pub domain: ExecutorDomain,
    /// The domain's one right to change its stores' bindings.
    pub bindings: StateBindings,
    pub plan: ResourcePlan,
    pub device: DeviceSelector,
    pub pool: MemoryPoolKind,
}

fn internal(reason: impl Into<String>) -> LoadError {
    LoadError::Internal {
        reason: reason.into(),
    }
}

fn platform_error(error: PlatformError) -> LoadError {
    classify_platform(error).into()
}

/// Construct the executor domain from an admitted manifest, opening and
/// charging nothing a load would not. Measurement and memory tools call this
/// to drive the domain below the scheduler.
pub fn build_native_domain(
    manifest: &ExecutionManifest,
    package: Arc<Package>,
) -> Result<(ExecutorDomain, StateBindings, ResourcePlan), LoadError> {
    let catalog = DeviceCatalog::discover().map_err(|error| internal(error.to_string()))?;
    build(catalog, manifest, package, Rc::new(|_| {}))
        .map(|built| (built.domain, built.bindings, built.plan))
}

/// A load's front half on its opened device: the selected device, the plan
/// draft and the attested programs, tuned against the kernel cache.
pub(crate) struct PreparedPrograms {
    pub draft: ExecutionPlanDraft,
    pub opened: OpenedPlatform,
    pub programs: AttestedPrograms,
    pub backend: BackendName,
    pub pool: MemoryPoolKind,
    pub capacity: ResourceCapacity,
}

/// Select and open the manifest's device, plan the model on it and prepare
/// its programs, tuning whatever the kernel cache lacks. A load continues
/// from here; the post-install prepare job ends here, so both produce and
/// look up the same tuning keys.
pub(crate) fn prepare(
    catalog: &DeviceCatalog,
    manifest: &ExecutionManifest,
    package: &Package,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<PreparedPrograms, LoadError> {
    let mut phase_started = Instant::now();
    if package.manifest() != manifest.package {
        return Err(LoadError::Artifact(ArtifactError::Invalid {
            reason: "opened package differs from the execution manifest".into(),
        }));
    }
    progress(LoadProgress::Preparing);
    // This runs inside the numerical worker: its own Seismic catalog and its
    // own process-scoped observations decide selection and admission. An
    // exact selector from the host is re-resolved here; a missing or
    // ambiguous match is refused, never substituted.
    let reserves = manifest.reserves;
    let selected = platform::select_device(catalog, manifest.path, manifest.device, &reserves)
        .map_err(platform_error)?;
    let backend = selected.info.backend;
    let topology = catalog.topology();
    let pool = match &selected.info.memory {
        DeviceMemory::Established(memory) => {
            topology
                .pool(memory.allocation_pool)
                .ok_or_else(|| internal("the selected device's pool is not in its topology"))?
                .kind
        }
        DeviceMemory::Unsupported { reason } => return Err(internal(reason.clone())),
    };
    let capacity = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes,
        tensor_operations: magnitude_executor::TensorOperations::of(selected.tensor_operations),
    };
    // The same derivation metadata-only assessment and preview plan through.
    let draft =
        crate::planning::plan_execution(manifest, &selected).map_err(|error| match error {
            crate::planning::ExecutionPlanningError::Plan(error) => {
                classify_plan(error, backend).into()
            }
            error => internal(error.to_string()),
        })?;
    let kernel_cache = manifest
        .kernel_cache
        .clone()
        .map(|root| KernelCache::open(root, DEFAULT_KERNEL_CACHE_BYTES).map(Arc::new))
        .transpose()
        .map_err(|error| internal(error.to_string()))?;
    report_load_phase("device selection and planning", &mut phase_started);
    let opened = platform::open_selected(
        catalog,
        draft.device().selector(),
        PlatformConfig {
            path: manifest.path,
            artifacts: kernel_cache
                .clone()
                .map(|cache| cache as Arc<dyn seismic::ArtifactStore>),
            reserves,
        },
    )
    .map_err(platform_error)?;
    report_load_phase("device open", &mut phase_started);
    let preparing = Instant::now();
    let error_classes = magnitude_executor::AdmittedErrorClasses::of(&manifest.model.error_classes)
        .map_err(internal)?;
    let programs = AttestedPrograms::prepare_draft(
        &draft,
        opened.device(),
        TuningContext {
            definition: &manifest.definition,
            weights: package,
            observer: &TuningReport { progress },
            cache: kernel_cache.as_deref(),
            error_classes: &error_classes,
        },
    )
    .map_err(|error| LoadError::from(classify_catalog(error)))?;
    let tuned = programs.tuned();
    eprintln!(
        "magnitude-engine: prepared programs in {:.2} s, {:.2} s of it tuning {} entries \
         ({} searched, {} stored; building inputs {:.2} s, forming {:.2} s, measuring {:.2} s, \
         validating {:.2} s)",
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
            .map(|tuned| tuned.time.building_seconds)
            .sum::<f64>(),
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
    Ok(PreparedPrograms {
        draft,
        opened,
        programs,
        backend,
        pool,
        capacity,
    })
}

pub(crate) fn build(
    catalog: DeviceCatalog,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<NativeDomain, LoadError> {
    let PreparedPrograms {
        draft,
        opened,
        mut programs,
        backend,
        pool,
        capacity,
    } = prepare(&catalog, manifest, &package, progress.clone())?;
    let reserves = manifest.reserves;
    let limits = draft.policy().limits();
    let head_enabled = draft.policy().selection().head;
    let state = ResourcePlanner::state_plan(
        &manifest.definition,
        draft.load(),
        draft.policy().method(),
        manifest.model.kv_codec,
        limits,
        capacity,
    )
    .map_err(internal)?;
    let mut phase_started = Instant::now();
    let target_graphs = programs
        .prepare_target_graphs(
            opened.device(),
            draft.load(),
            &manifest.definition.decoder,
            &state,
            draft.programs().target(),
            limits,
        )
        .map_err(internal)?;
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(
            opened.device(),
            draft.load(),
            &manifest.definition.decoder,
            limits,
        )
        .map_err(internal)?;
    programs
        .prepare_auxiliary_graphs(
            opened.device(),
            draft.load(),
            &manifest.definition,
            draft.programs().vision(),
            state.target_state(),
            state.head_state(),
            limits,
            manifest.model.method.proposals(),
        )
        .map_err(internal)?;
    let resources = ResourcePlanner::plan_with_state(
        opened.device().backend(),
        state,
        &target_graphs,
        &target_readout_graphs,
        programs.drafter_graphs(),
        programs.vision_graphs().map(|graphs| graphs.as_ref()),
        programs
            .state_graphs()
            .ok_or_else(|| internal("prepared program set has no state graphs"))?
            .as_ref(),
    )
    .map_err(internal)?;
    let seal = target_graphs.seal_report();
    eprintln!(
        "magnitude-engine: sealed {} target graph classes ({} graphs) in {:.2} s",
        seal.classes, seal.sealed_graphs, seal.seconds
    );
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let execution_plan = draft
        .admit(resources)
        .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    let plan = execution_plan.resources().clone();
    report_load_phase("graph and resource planning", &mut phase_started);
    if opened.selector() != execution_plan.device().selector() {
        return Err(internal(
            "opened device differs from the selected execution plan",
        ));
    }
    let device_selector = opened.selector();
    let resource_identity = ResourceDomainId::new(format!(
        "{}:{}",
        manifest.package.identity,
        opened.selector(),
    ))
    .map_err(internal)?;
    let device = Rc::new(opened.into_device());
    let programs = Rc::new(programs);
    let target_graphs = programs
        .target_graphs()
        .ok_or_else(|| internal("qualified program set has no target graphs"))?;
    let target_binding_constants = target_graphs.binding_constant_bytes().map_err(internal)?;
    let target_readout_graphs = programs
        .target_readout_graphs()
        .ok_or_else(|| internal("qualified program set has no target readout graphs"))?;
    let state_graphs = programs
        .state_graphs()
        .ok_or_else(|| internal("qualified program set has no state graphs"))?;
    let catalog = Rc::new(catalog);
    let mut startup = StartupClaims {
        heap: DeviceHeap::open(catalog.clone(), reserves, device.clone())
            .map_err(|error| platform_error(PlatformError::Memory(error)))?,
        allocation_domain: MemoryDomain::of(device_selector, pool),
        held: None,
    };
    startup.claim("graph scratch", plan.bytes().scratch, 0)?;
    let resources = ResourceAllocator::allocate(
        &execution_plan,
        &device,
        resource_identity.clone(),
        target_graphs,
        target_readout_graphs,
        programs.drafter_graphs(),
        programs.vision_graphs().map(|graphs| graphs.as_ref()),
        state_graphs.as_ref(),
    )
    .map_err(|error| internal(error.to_string()))?;
    report_load_phase("resource allocation", &mut phase_started);
    let mut residency = ResidencyStore::new(
        device.clone(),
        programs.clone(),
        execution_plan.clone(),
        resource_identity.clone(),
    )
    .map_err(|error| internal(error.to_string()))?;
    let target_upload = execution_plan
        .load()
        .target_upload_peak_bytes()
        .map_err(|error| internal(error.to_string()))?;
    let target_peak = plan
        .bytes()
        .target_weights
        .checked_add(target_upload)
        .ok_or_else(|| internal("target import peak byte count overflow"))?;
    startup.hold_host_tables(
        execution_plan
            .load()
            .host_table_bytes()
            .map_err(|error| internal(error.to_string()))?,
    )?;
    startup.claim("target import", target_peak, target_upload)?;
    let import_progress = progress.clone();
    let import_device = device.clone();
    let mut band_observed = Instant::now();
    let target = residency
        .load_target(
            &manifest.definition,
            &package,
            Box::new(move |completed_bytes, total_bytes| {
                import_progress(LoadProgress::ImportingWeights {
                    completed_bytes,
                    total_bytes,
                });
                if band_observed.elapsed() >= IMPORT_OBSERVATION_INTERVAL {
                    band_observed = Instant::now();
                    require_normal(&catalog, &import_device, &reserves)?;
                }
                Ok(())
            }),
        )
        .map_err(|error| match error {
            ResidencyError::Import(WeightImportError::Refused(refusal)) => {
                startup.refusal("target import", refusal)
            }
            error => internal(error.to_string()),
        })?;
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
    progress(LoadProgress::Finalizing);
    let definition = Rc::new(manifest.definition.clone());
    let head_loader = head_enabled
        .then(|| ComponentLoader::head(residency, definition.clone(), package.clone()))
        .transpose()
        .map_err(|error| internal(error.to_string()))?;
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
        .map_err(|error| internal(error.to_string()))?;
    startup.claim(
        "initial target state",
        plan.target_state()
            .initial_committed_bytes()
            .map_err(internal)?,
        0,
    )?;
    let target_state = plan
        .allocate_target_state(device.clone())
        .map_err(internal)?;
    let head_state = if let Some(state) = plan.head_state() {
        startup.claim(
            "initial head state",
            state.initial_committed_bytes().map_err(internal)?,
            0,
        )?;
        Some(state.allocate(device.clone()).map_err(internal)?)
    } else {
        None
    };
    startup.claim("target binding constants", target_binding_constants, 0)?;
    let (mut domain, bindings) = ExecutorDomain::new(
        Rc::new(execution_plan),
        definition,
        startup.into_heap(),
        programs,
        resources,
        head_loader,
        vision_loader,
        target,
        target_state,
        head_state,
    )
    .map_err(internal)?;
    domain
        .register_allocated_holdings()
        .map_err(|error| internal(format!("register allocated memory holdings: {error}")))?;
    report_load_phase("state and domain allocation", &mut phase_started);
    let bindings = warm_up(&mut domain, bindings)?;
    Ok(NativeDomain {
        domain,
        bindings,
        plan,
        device: device_selector,
        pool,
    })
}

fn report_load_phase(name: &str, started: &mut Instant) {
    eprintln!(
        "magnitude-engine: {name} in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    *started = Instant::now();
}

/// Startup allocations claimed from the device's heap, the same heap the
/// domain claims from once loaded. Each claim keeps every domain's headroom
/// above its planning reserve and is held through its allocation, until the
/// next claim follows it.
struct StartupClaims {
    heap: DeviceHeap,
    /// The device's allocation domain; a dedicated device stages through
    /// host RAM.
    allocation_domain: MemoryDomain,
    held: Option<ClaimId>,
}

impl StartupClaims {
    /// Claim `allocation` additional bytes of the device's allocation domain,
    /// `staged` of which a dedicated device also stages through host RAM. A
    /// host-backed device has no staging domain: its staged bytes are part
    /// of `allocation`. Releases the previous startup claim, whose
    /// allocation Seismic now charges.
    fn claim(&mut self, purpose: &str, allocation: u64, staged: u64) -> Result<(), LoadError> {
        if let Some(held) = self.held.take() {
            self.heap.release(held);
        }
        let claim = self
            .heap
            .claim(allocation, staged, HoldingClass::Model)
            .map_err(|refusal| self.refusal(purpose, refusal))?;
        self.held = Some(claim);
        Ok(())
    }

    /// Hold the model's host-resident tables (`bytes`) in the host-RAM
    /// domain for the model's lifetime.
    fn hold_host_tables(&mut self, bytes: u64) -> Result<(), LoadError> {
        if bytes == 0 {
            return Ok(());
        }
        self.heap
            .hold_host_table(bytes)
            .map_err(|refusal| self.refusal("host tables", refusal))
    }

    fn refusal(&self, purpose: &str, refusal: ClaimRefusal) -> LoadError {
        let domain = |role: DomainRole| match role {
            DomainRole::Allocation => self.allocation_domain,
            DomainRole::Staging => MemoryDomain::HostRam,
        };
        match refusal {
            ClaimRefusal::Blind(error) => platform_error(PlatformError::Memory(error)),
            ClaimRefusal::Reclaim { role, distress } => LoadError::MemoryPressure {
                purpose: match distress {
                    Some(distress) => format!("{purpose} ({distress})"),
                    None => format!("{purpose} (memory at or below the planning reserve)"),
                },
                domain: domain(role),
            },
            ClaimRefusal::Deficit {
                role,
                constraint,
                required,
                available,
            } => LoadError::InsufficientMemory {
                purpose: format!("{purpose} ({constraint})"),
                domain: domain(role),
                memory: InsufficientMemory {
                    required,
                    available,
                },
            },
            ClaimRefusal::Accounting(error) => {
                internal(format!("{purpose}: memory accounting: {error:?}"))
            }
        }
    }

    /// The heap with every startup claim released, for the loaded domain.
    fn into_heap(mut self) -> DeviceHeap {
        if let Some(held) = self.held.take() {
            self.heap.release(held);
        }
        self.heap
    }
}

/// Run one throwaway single-row forward before readiness, so a broken device
/// path fails the load instead of the first request, and the process's
/// one-time first-forward cost is paid here. Tuning has already executed every
/// kernel, and no row class carries its own first-use cost, so one row
/// suffices. The request's state advance is aborted, so no state survives it.
/// The bindings travel with the forward and return with it.
fn warm_up(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
) -> Result<StateBindings, LoadError> {
    let began = std::time::Instant::now();
    let request = RequestId(u64::MAX);
    let failed = |error: String| internal(format!("load warm-up forward: {error}"));
    let input = PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            vec![TokenId(0)],
            InputLayout::new(1, Vec::new()).map_err(|error| failed(error.to_string()))?,
        )
        .map_err(|error| failed(error.to_string()))?,
        vec![[0; 3]],
    )
    .map_err(|error| failed(error.to_string()))?;
    domain.install_input(request, input).map_err(failed)?;
    domain
        .open_state(&mut bindings, request, None)
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
        prime: None,
    }];
    let resources = domain
        .reserve(&mut bindings, &operations)
        .map_err(|error| failed(error.to_string()))?
        .into_resources();
    let ReservedResources::Target(reservation) = resources else {
        return Err(failed("reserved a non-target lane".into()));
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .map_err(|failure| failed(failure.error().to_string()))?;
    let (pending, bindings) = domain
        .finish_target(flight)
        .map_err(|error| failed(error.to_string()))?;
    for pending in pending {
        domain
            .abort(pending)
            .map_err(|error| failed(error.to_string()))?;
    }
    domain.close(request).map_err(failed)?;
    eprintln!(
        "magnitude-engine: warm-up forward in {:.0} ms",
        began.elapsed().as_secs_f64() * 1000.0
    );
    Ok(bindings)
}

/// Reports tuning at load: its progress to the host, and each unit on the
/// engine's diagnostic stream while the worker is not yet ready.
struct TuningReport {
    progress: Rc<dyn Fn(LoadProgress)>,
}

impl TuningObserver for TuningReport {
    fn event(&self, event: &TuningEvent) {
        match event {
            // A load whose every unit has a stored result does not tune.
            TuningEvent::Progress { completed, total } => {
                if *total > 0 {
                    (self.progress)(LoadProgress::Tuning {
                        completed: *completed as u64,
                        total: *total as u64,
                    });
                }
            }
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
                    (TuningOrigin::Searched, Some((allowance, stop))) => {
                        format!("budget {:.2} s, stop {stop:?}", allowance.as_secs_f64())
                    }
                    (TuningOrigin::Searched, None) => "surveyed".to_owned(),
                };
                eprintln!(
                    "magnitude-engine: tuned {} [{}] in {:.2} s ({origin}): {:?} ({} measured, {} excluded, {} qualification rejections)",
                    tuned.entry,
                    tuned.bindings,
                    tuned.seconds,
                    tuned.overall.params,
                    tuned.measured,
                    tuned.excluded,
                    tuned.rejections
                );
                if let Some(rejection) = &tuned.first_rejection {
                    eprintln!("magnitude-engine:   first qualification rejection of {}: {rejection}", tuned.entry);
                }
            }
        }
    }
}
