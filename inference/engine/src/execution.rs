//! One production composition root from an admitted manifest to a ready
//! executor domain. Every device-bound value is created inside the numerical
//! worker that runs this.

use crate::census::MemoryDomain;
use crate::error::{
    ArtifactError, InsufficientMemory, LoadError, classify_catalog, classify_plan,
    classify_platform,
};
use crate::options::{ExecutionManifest, ResolvedMethod};
use crate::worker::protocol::LoadProgress;
use magnitude_artifacts::{ArtifactIdentity, Package, PackageIdentity};
use magnitude_batching::Demand;
use magnitude_executor::{
    AttestedPrograms, ClaimRefusal, ComponentLoader, DEFAULT_KERNEL_CACHE_BYTES, DeviceHeap,
    ExecutionPlanDraft, ExecutorDomain, KernelCache, Operation, RequestId, ReservedResources,
    ResidencyError, ResidencyStore, ResourceAllocator, ResourceCapacity, ResourceDomainId,
    CatalogError, CatalogFailure, ResourcePlan, ResourcePlanner, SEARCH_VERSION, StateBindings,
    TokenId, TracedWork, TunedEntry, TuningCacheKey, TuningContext, TuningEvent, TuningObserver,
    TuningOrigin, TuningSettings, TuningTrace, UnpreparedUnits, WeightImportError, WorkKind,
    memory::{ClaimId, HoldingClass},
    platform::{self, DeviceRequest, DomainRole, OpenedPlatform, PlatformConfig, PlatformError},
    require_normal,
};
use magnitude_family_contracts::{HeadHistory, InputLayout, PreparedModelInput, TokenPlan};
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
/// charging nothing a load would not: this is a load, so it reads stored
/// tuning records and searches nothing. Measurement and memory tools call
/// this to drive the domain below the scheduler; with `unprepared` as
/// [`UnpreparedUnits::Report`] it is the load step of
/// [`crate::worker::load_prepared`].
pub fn load_native_domain(
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
) -> Result<LoadAttempt<(ExecutorDomain, StateBindings, ResourcePlan)>, LoadError> {
    let catalog = DeviceCatalog::discover().map_err(|error| internal(error.to_string()))?;
    Ok(
        match build(catalog, manifest, package, unprepared, Rc::new(|_| {}))? {
            LoadAttempt::Loaded(built) => {
                LoadAttempt::Loaded((built.domain, built.bindings, built.plan))
            }
            LoadAttempt::Unprepared => LoadAttempt::Unprepared,
        },
    )
}

/// [`load_native_domain`] serving canonical defaults for every tuning unit
/// without a stored record: the load of a host that has no search run.
pub fn build_native_domain(
    manifest: &ExecutionManifest,
    package: Arc<Package>,
) -> Result<(ExecutorDomain, StateBindings, ResourcePlan), LoadError> {
    match load_native_domain(manifest, package, UnpreparedUnits::ServeDefaults)? {
        LoadAttempt::Loaded(built) => Ok(built),
        LoadAttempt::Unprepared => Err(internal(
            "a load serving canonical defaults stopped unprepared",
        )),
    }
}

/// A load's front half on its opened device: the selected device, the plan
/// draft and the attested programs, prepared from the kernel cache's stored
/// tuning records.
pub(crate) struct PreparedPrograms {
    pub draft: ExecutionPlanDraft,
    pub opened: OpenedPlatform,
    pub programs: AttestedPrograms,
    pub backend: BackendName,
    pub pool: MemoryPoolKind,
    pub capacity: ResourceCapacity,
    /// The load's one set of worker threads: everything its preparation
    /// does off the calling thread runs on these and no others. It is lent
    /// to the opened device for the programs the load forms, and seals the
    /// load's graphs.
    pub workers: Arc<seismic::Workers>,
}

/// A manifest's model planned on its selected and opened device: where a
/// search and a load both start, so both derive the same tuning keys.
struct PlannedModel {
    draft: ExecutionPlanDraft,
    opened: OpenedPlatform,
    backend: BackendName,
    pool: MemoryPoolKind,
    capacity: ResourceCapacity,
    kernel_cache: Option<Arc<KernelCache>>,
    numerical_profile: magnitude_executor::ResolvedNumericalProfile,
}

/// Select and open the manifest's device and plan the model on it.
fn plan_on_device(
    catalog: &DeviceCatalog,
    manifest: &ExecutionManifest,
    package: &Package,
    progress: &dyn Fn(LoadProgress),
) -> Result<PlannedModel, LoadError> {
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
    report_load_memory("device opened", opened.device());
    let numerical_profile = manifest
        .model
        .numerical_profile
        .clone()
        .unwrap_or_else(magnitude_executor::NumericalProfile::standard)
        .resolve()
        .map_err(internal)?;
    Ok(PlannedModel {
        draft,
        opened,
        backend,
        pool,
        capacity,
        kernel_cache,
        numerical_profile,
    })
}

/// The tuning report of `work` on `manifest`: progress to the host, units to
/// the diagnostic stream, every event to the trace the host asked for.
fn tuning_report(
    manifest: &ExecutionManifest,
    work: TracedWork,
    numerical_profile: &magnitude_executor::ResolvedNumericalProfile,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<TuningReport, LoadError> {
    Ok(TuningReport {
        progress,
        trace: manifest
            .tuning_trace
            .as_deref()
            .map(|path| TuningTrace::begin(path, work, manifest.tuning, numerical_profile))
            .transpose()
            .map_err(internal)?,
    })
}

/// The Prepare request's work, and the engine's only search: on the device a
/// load of `manifest` selects, search every tuning unit the kernel cache has
/// no record for, within the manifest's tuning budget, publishing each
/// unit's record as it stands. Nothing is prepared for serving and the model
/// is not imported: the process ends here, and a load reads the records.
pub(crate) fn search(
    catalog: &DeviceCatalog,
    manifest: &ExecutionManifest,
    package: &Package,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<(), LoadError> {
    let PlannedModel {
        draft,
        opened,
        kernel_cache,
        numerical_profile,
        ..
    } = plan_on_device(catalog, manifest, package, progress.as_ref())?;
    let searching = Instant::now();
    let report = tuning_report(manifest, TracedWork::Search, &numerical_profile, progress)?;
    let tuned = AttestedPrograms::search_draft(
        &draft,
        opened.device(),
        TuningContext {
            definition: &manifest.definition,
            weights: package,
            observer: &report,
            cache: kernel_cache.as_deref(),
            numerical_profile: &numerical_profile,
            settings: manifest.tuning,
        },
        Some(opened.tuning_room()),
    )
    .map_err(|error| LoadError::from(classify_catalog(error)))?;
    if let Some(trace) = &report.trace {
        trace.conclude(&tuned).map_err(internal)?;
    }
    let sum = |part: fn(&TunedEntry) -> f64| tuned.iter().map(part).sum::<f64>();
    eprintln!(
        "magnitude-engine: searched in {:.2} s, {:.2} s of it tuning {} entries \
         ({} searched, {} stored; building inputs {:.2} s, forming {:.2} s, measuring {:.2} s, \
         validating {:.2} s)",
        searching.elapsed().as_secs_f64(),
        sum(|tuned| tuned.seconds),
        tuned.len(),
        tuned
            .iter()
            .filter(|tuned| tuned.origin == TuningOrigin::Searched)
            .count(),
        tuned
            .iter()
            .filter(|tuned| tuned.origin == TuningOrigin::Stored)
            .count(),
        sum(|tuned| tuned.time.building_seconds),
        sum(|tuned| tuned.time.forming_seconds),
        sum(|tuned| tuned.time.measuring_seconds),
        sum(|tuned| tuned.time.validating_seconds),
    );
    report_load_memory("searched", opened.device());
    if let Some(cache) = &kernel_cache {
        cache.store_prepared(&prepared_key(manifest)?);
    }
    Ok(())
}

/// The kernel cache key under which a finished Prepare of `manifest` is
/// recorded ([`KernelCache::prepared`]): everything of the manifest that
/// decides what a Prepare searches, and the search's version. The key
/// outlives the process that resolved the manifest, so nothing the process
/// or the boot names is part of it: not the package's artifact identities,
/// which are process-local ([`ArtifactIdentity`]; the components' files and
/// tensors stand for them), and not the device request, whose selector
/// names the device as this boot enumerated it. The device's part is its
/// tuning identity, which is in every unit's record key and is known only
/// once the device is opened: a load on a device whose records are missing
/// meets that unit and reports it unprepared all the same. Where the host
/// keeps its cache and trace is not part of the key either, nor whether the
/// Prepare searched afresh.
fn prepared_key(manifest: &ExecutionManifest) -> Result<TuningCacheKey, LoadError> {
    let mut load = ExecutionManifest {
        device: DeviceRequest::Automatic,
        kernel_cache: None,
        tuning_trace: None,
        tuning: TuningSettings {
            fresh: false,
            ..manifest.tuning
        },
        ..manifest.clone()
    };
    let unnamed = ArtifactIdentity([0; 32]);
    let package = &mut load.package;
    package.identity = PackageIdentity {
        target: unnamed,
        projector: package.identity.projector.map(|_| unnamed),
    };
    for component in [
        Some(&mut package.target),
        package.projector.as_mut(),
        package.draft.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        component.identity = unnamed;
    }
    load.definition.artifact_identity = package.identity;
    let load = serde_json::to_string(&load).map_err(|error| internal(error.to_string()))?;
    Ok(TuningCacheKey::of(&format!(
        "prepared {SEARCH_VERSION}\n{load}"
    )))
}

/// How an attempt to load ended, short of failing.
#[derive(Debug)]
pub enum LoadAttempt<L> {
    Loaded(L),
    /// A load asked to report tuning units without a stored record
    /// ([`UnpreparedUnits::Report`]) met one. Nothing of the model was
    /// placed, and nothing of the attempt remains: the host has the model's
    /// Prepare run and loads again ([`crate::worker::load_prepared`]).
    Unprepared,
}

/// A load's front half: select and open the manifest's device, plan the
/// model on it and prepare its programs from the kernel cache's stored
/// tuning records. A load never searches: at a unit without a record it
/// stops unprepared or prepares the unit at its canonical default, as
/// `unprepared` says.
fn prepare(
    catalog: &DeviceCatalog,
    manifest: &ExecutionManifest,
    package: &Package,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<LoadAttempt<PreparedPrograms>, LoadError> {
    // A load asked to report stops here when no Prepare of it has finished,
    // or when its Prepare is to search afresh: before the device is opened,
    // so a host that loads in its own process has formed no program when
    // its Prepare runs ([`KernelCache::prepared`]).
    if unprepared == UnpreparedUnits::Report {
        if let Some(root) = &manifest.kernel_cache {
            let cache = KernelCache::open(root.clone(), DEFAULT_KERNEL_CACHE_BYTES)
                .map_err(|error| internal(error.to_string()))?;
            if manifest.tuning.fresh || !cache.prepared(&prepared_key(manifest)?) {
                return Ok(LoadAttempt::Unprepared);
            }
        }
    }
    let PlannedModel {
        draft,
        opened,
        backend,
        pool,
        capacity,
        kernel_cache,
        numerical_profile,
    } = plan_on_device(catalog, manifest, package, progress.as_ref())?;
    let preparing = Instant::now();
    let workers = Arc::new(seismic::Workers::of_the_host());
    opened.device().staff(Some(workers.clone()));
    let report = tuning_report(manifest, TracedWork::Load, &numerical_profile, progress)?;
    let mut programs = match AttestedPrograms::prepare_draft(
        &draft,
        opened.device(),
        TuningContext {
            definition: &manifest.definition,
            weights: package,
            observer: &report,
            cache: kernel_cache.as_deref(),
            numerical_profile: &numerical_profile,
            settings: manifest.tuning,
        },
        unprepared,
    ) {
        Ok(programs) => programs,
        Err(CatalogError {
            failure: CatalogFailure::Unprepared { entry, bindings },
            ..
        }) => {
            eprintln!(
                "magnitude-engine: no stored tuning record for {entry} [{bindings}]: the load \
                 stops unprepared, before placing anything"
            );
            // The load that follows the Prepare writes the load's trace.
            if let Some(trace) = report.trace {
                trace.abandon().map_err(internal)?;
            }
            return Ok(LoadAttempt::Unprepared);
        }
        Err(error) => return Err(LoadError::from(classify_catalog(error))),
    };
    let draft = draft
        .finalize_readout(&manifest.definition, programs.readout_choices())
        .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    if draft.load().readout_choices() != programs.readout_choices() {
        programs
            .discard_readout_choices(draft.programs())
            .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    }
    let tuned = programs.tuned();
    if let Some(trace) = &report.trace {
        trace.conclude(tuned).map_err(internal)?;
    }
    eprintln!(
        "magnitude-engine: prepared programs in {:.2} s: {} tuned entries from stored records, \
         {} at canonical defaults for want of one",
        preparing.elapsed().as_secs_f64(),
        tuned.len(),
        programs.unprepared(),
    );
    report_load_memory("programs prepared", opened.device());
    Ok(LoadAttempt::Loaded(PreparedPrograms {
        draft,
        opened,
        programs,
        backend,
        pool,
        capacity,
        workers,
    }))
}

/// A load: one code path whose `unprepared` decides only what happens at a
/// tuning unit without a stored record, which it meets while its programs
/// are prepared, before anything of the model is placed.
pub(crate) fn build(
    catalog: DeviceCatalog,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<LoadAttempt<NativeDomain>, LoadError> {
    let PreparedPrograms {
        draft,
        opened,
        mut programs,
        backend,
        pool,
        capacity,
        workers,
    } = match prepare(&catalog, manifest, &package, unprepared, progress.clone())? {
        LoadAttempt::Loaded(prepared) => prepared,
        LoadAttempt::Unprepared => return Ok(LoadAttempt::Unprepared),
    };
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
            &workers,
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
            draft.programs(),
            &state,
            limits,
            draft.policy().method(),
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
    // What the sealed graphs hold on the host against what the plan held
    // them to before any was sealed, from both sides: one line of JSON,
    // beside the memory moments around it (`programs prepared`, `graphs
    // planned`). Each heap is requested bytes, its allocations, and the
    // bound of the allocator's rounding of them, which no plan counts.
    let heap = |heap: seismic::HostHeap| {
        serde_json::json!({
            "bytes": heap.bytes,
            "allocations": heap.allocations,
            "rounding_bound": heap.rounding,
        })
    };
    let graphs = |graphs: &magnitude_executor::SealedGraphsHeap| {
        serde_json::json!({
            "graphs": graphs.graphs,
            "plans": heap(graphs.plans),
            "storage": heap(graphs.storage),
            "owned": heap(graphs.owned),
        })
    };
    let held = seal.held();
    let total = |graphs: &magnitude_executor::SealedGraphsHeap, constants: seismic::HostHeap| {
        graphs
            .heap()
            .and_then(|graphs| graphs.plus(constants))
            .map(heap)
    };
    eprintln!(
        "magnitude-engine: sealed graph host heap {}",
        serde_json::json!({
            "planned_exact": seal.planned.exact(),
            "planned_at_most": graphs(&seal.planned.at_most),
            "planned_at_least": graphs(&seal.planned.at_least),
            "planned_constants": heap(seal.planned.constants),
            "planned_at_most_total": total(&seal.planned.at_most, seal.planned.constants),
            "planned_at_least_total": total(&seal.planned.at_least, seal.planned.constants),
            "held": graphs(&held),
            "held_constants": heap(seal.constants),
            "held_total": total(&held, seal.constants),
            "mismatched_graphs": seal.mismatched_graphs,
        })
    );
    if let Some(state_graphs) = programs.state_graphs() {
        eprintln!(
            "magnitude-engine: state copy graph host heap {}",
            serde_json::json!({
                "planned": heap(state_graphs.planned_host_heap()),
                "held": heap(state_graphs.host_heap()),
            })
        );
    }
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let execution_plan = draft
        .admit(resources)
        .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    let plan = execution_plan.resources().clone();
    // The load's preparation off the calling thread is over: its threads
    // are taken back from the device and joined.
    opened.device().staff(None);
    drop(workers);
    // Every graph family of the load is checked and sealed: the shapes its
    // checked entries evaluated for them, class by class, serve nothing
    // further and are released. The moments on either side say what they
    // held.
    report_load_memory("graphs sealed", opened.device());
    eprintln!(
        "magnitude-engine: released {} evaluated entry shapes",
        seismic::forget_evaluated_entry_shapes()
    );
    report_load_phase("graph and resource planning", &mut phase_started);
    report_load_memory("graphs planned", opened.device());
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
    report_load_memory("resources allocated", &device);
    let mut residency = ResidencyStore::new(
        device.clone(),
        programs.clone(),
        execution_plan.clone(),
        resource_identity.clone(),
    )
    .map_err(|error| internal(error.to_string()))?;
    // The rows steps name of a streamed table are kept beside the kernels,
    // for the next load of the model.
    if let Some(cache) = &manifest.kernel_cache {
        residency.keep_table_rows(cache.join("table-rows"));
    }
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
    // The stored tables' slots and directories, part of the target's
    // weights, are allocated where the device's host-shared storage lives:
    // on a dedicated device whose shared storage is host RAM, they are
    // claimed there, beside the staged upload, not in its own memory.
    let shared_in_host = match magnitude_executor::platform::shared_role(device.info())
        .map_err(|error| platform_error(PlatformError::Memory(error)))?
    {
        DomainRole::Allocation => 0,
        DomainRole::Staging => execution_plan
            .load()
            .shared_table_bytes()
            .map_err(|error| internal(error.to_string()))?,
    };
    startup.claim(
        "target import",
        target_peak
            .checked_sub(shared_in_host)
            .ok_or_else(|| internal("stored tables exceed the target import peak"))?,
        target_upload
            .checked_add(shared_in_host)
            .ok_or_else(|| internal("target import staging byte count overflow"))?,
    )?;
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
    report_load_memory("target imported", &device);
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
    report_load_memory("domain allocated", &device);
    // A drafter that keeps its own history is primed by the warm-up; a draft
    // head reading the target's history keeps none to prime.
    let primes_drafter = head_enabled
        && !matches!(
            manifest.model.method,
            ResolvedMethod::Mtp {
                history: HeadHistory::Target,
                ..
            }
        );
    let bindings = warm_up(&mut domain, bindings, primes_drafter)?;
    report_load_memory("loaded", &device);
    Ok(LoadAttempt::Loaded(NativeDomain {
        domain,
        bindings,
        plan,
        device: device_selector,
        pool,
    }))
}

fn report_load_phase(name: &str, started: &mut Instant) {
    eprintln!(
        "magnitude-engine: {name} in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    *started = Instant::now();
}

/// Report what the process holds at one moment of a load, by what holds it
/// ([`seismic::ProcessMemory`]), with Seismic's charges on the opened device
/// and the most the footprint has been so far: one line of JSON on the
/// engine's diagnostic stream. Every load reports the same moments, so two
/// loads compare line by line, part by part; a search reports the device's
/// opening and its own end (`searched`), whose peak is its process's.
fn report_load_memory(moment: &str, device: &seismic::Device) {
    let memory = seismic::ProcessMemory::read();
    let usage = device.memory_usage();
    eprintln!(
        "magnitude-engine: memory {}",
        serde_json::json!({
            "moment": moment,
            "footprint_bytes": memory.footprint_bytes,
            "malloc_live_bytes": memory.malloc_live_bytes,
            "graphics_bytes": memory.graphics_bytes,
            "mapped_bytes": memory.mapped_bytes,
            "rest_bytes": memory.rest_bytes(),
            "device_charged_bytes": usage.charged,
            "device_held_bytes": usage.held,
            "device_unallocated_bytes": usage.unallocated,
            "peak_footprint_bytes": seismic::ProcessMemory::peak_footprint(),
        })
    );
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
/// one-time first-forward cost is paid here. No row class carries its own
/// first-use cost, so one row suffices. An enabled drafter that keeps its own history also receives one
/// state-only priming row (`primes_drafter`), so its component import and
/// binding complete before readiness; a draft head reading the target's
/// history keeps none, and its component was imported and bound at load. The
/// request's state advances are aborted, so no state survives it.
/// The bindings travel with the forward and return with it.
fn warm_up(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    primes_drafter: bool,
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
        kind: if primes_drafter {
            WorkKind::Prefill
        } else {
            WorkKind::Replay
        },
        tokens: vec![TokenId(0)],
        position: 0,
        conditioning: None,
        demand: if primes_drafter {
            Demand::FEATURES
        } else {
            Demand::NONE
        },
        select: Vec::new(),
        committed: 1,
        prime: primes_drafter.then(|| magnitude_executor::Priming {
            boundary: false,
            tokens: vec![TokenId(0)],
            position: 0,
            draft_from: 1,
        }),
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

/// Reports a search's tuning, or the records a load read: a search's
/// progress to the host, each unit on the engine's diagnostic stream, and
/// every event to the tuning trace the host asked for.
struct TuningReport {
    progress: Rc<dyn Fn(LoadProgress)>,
    trace: Option<TuningTrace>,
}

impl TuningObserver for TuningReport {
    fn event(&self, event: &TuningEvent) {
        if let Some(trace) = &self.trace {
            trace.event(event);
        }
        match event {
            // Only a search that finds a unit without a record reports
            // progress; a load never does.
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
                points,
            } => eprintln!("magnitude-engine: tuning {entry} [{bindings}] at {points} points"),
            // Each unit's mandatory work, each piece of optional work and
            // each record published when the optional work stops are in
            // the tuning trace, the latter two with their forecasts.
            TuningEvent::Mandatory { .. }
            | TuningEvent::Stage(_)
            | TuningEvent::Published { .. } => {}
            TuningEvent::Untuned {
                programs,
                forming_seconds,
            } => eprintln!(
                "magnitude-engine: formed {programs} untuned programs for the load in {forming_seconds:.2} s"
            ),
            TuningEvent::Budget(budget) => {
                eprintln!(
                    "magnitude-engine: tuning budget {:.2} s: {} pieces of work used {:.2} s, {:.2} s unused, {:.2} s over; {} offered pieces did not fit",
                    budget.budget_seconds,
                    budget.stages,
                    budget.used_seconds,
                    budget.unused_seconds,
                    budget.overrun_seconds,
                    budget.refused.len()
                );
            }
            TuningEvent::Unprepared {
                entry,
                bindings,
                statics,
            } => eprintln!(
                "magnitude-engine: no stored tuning record for {entry} [{bindings}] {statics:?}: \
                 canonical default"
            ),
            TuningEvent::Pinned {
                entry,
                bindings,
                statics,
                choice,
                ..
            } => eprintln!(
                "magnitude-engine: pinned {entry} [{bindings}] {statics:?}: {choice:?} (not tuned)"
            ),
            TuningEvent::Finished(tuned) => {
                let origin = match (tuned.origin, tuned.stop) {
                    (TuningOrigin::Stored, _) => "stored".to_owned(),
                    (TuningOrigin::Searched, Some(stop)) => format!("stop {stop:?}"),
                    (TuningOrigin::Searched, None) if tuned.progress.is_some() => {
                        "not searched".to_owned()
                    }
                    (TuningOrigin::Searched, None) => "surveyed".to_owned(),
                };
                // How far the unit's search got: reported here, not stored.
                if let Some(progress) = &tuned.progress {
                    let standing = progress.standing;
                    eprintln!(
                        "magnitude-engine: searched {} [{}]: share {:.1}% of step time; {} pieces of work: {} of the {} configurations asked for measured, {} programs formed, serving {:.3} of the defaults; building {:.2} s, references {:.2} s, forming {:.2} s, measuring {:.2} s, validating {:.2} s",
                        tuned.entry,
                        tuned.bindings,
                        progress.share * 100.0,
                        progress.stages,
                        standing.measured,
                        standing.generated,
                        standing.programs,
                        standing.cost,
                        tuned.time.building_seconds,
                        tuned.time.reference_seconds,
                        tuned.time.forming_seconds,
                        tuned.time.measuring_seconds,
                        tuned.time.validating_seconds
                    );
                }
                eprintln!(
                    "magnitude-engine: tuned {} [{}] in {:.2} s ({origin}, {}): {:?} ({} measured, {} excluded, {} qualification rejections)",
                    tuned.entry,
                    tuned.bindings,
                    tuned.seconds,
                    if tuned.qualified {
                        "qualified"
                    } else {
                        "canonical default, unqualified"
                    },
                    tuned.overall.params,
                    tuned.measured,
                    tuned.excluded,
                    tuned.rejections
                );
                if let Some(rejection) = &tuned.first_rejection {
                    eprintln!(
                        "magnitude-engine:   first qualification rejection of {}: {rejection}",
                        tuned.entry
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::tests::manifest_with_kernel_cache;
    use std::path::PathBuf;

    /// Another process, after a reboot, resolves the same model under other
    /// artifact identities (they count the packages each process opened)
    /// and names the same device by another selector (its registry id is
    /// assigned at boot); its load finds the Prepare this one recorded.
    /// Another model does not.
    #[test]
    fn a_prepare_is_found_by_another_process_resolving_the_same_load() {
        let prepared = manifest_with_kernel_cache(Some(PathBuf::from("/cache/kernels")));
        let mut elsewhere = manifest_with_kernel_cache(Some(PathBuf::from("/other/kernels")));
        let identity = ArtifactIdentity([7; 32]);
        elsewhere.package.identity.target = identity;
        elsewhere.package.target.identity = identity;
        elsewhere.definition.artifact_identity.target = identity;
        elsewhere.device = "metal:000000010000050f".parse().unwrap();
        assert_eq!(
            prepared_key(&prepared).unwrap(),
            prepared_key(&elsewhere).unwrap()
        );
        let mut other = prepared.clone();
        other.package.target.files[0].size += 1;
        assert_ne!(
            prepared_key(&prepared).unwrap(),
            prepared_key(&other).unwrap()
        );
    }
}
