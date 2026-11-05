//! One production composition root from an admitted manifest to a ready
//! executor domain. Every device-bound value is created inside the numerical
//! worker that runs this.

use crate::error::{
    ArtifactError, InsufficientMemory, LoadError, classify_catalog, classify_plan,
    classify_platform,
};
use crate::options::{ExecutionManifest, ResolvedMethod};
use crate::worker::protocol::{LoadProgress, TuningOperation, TuningWork};
use std::cell::RefCell;
use std::collections::BTreeMap;
use magnitude_artifacts::{ArtifactIdentity, Package, PackageIdentity};
use magnitude_batching::Demand;
use magnitude_executor::{
    AttestedPrograms, ClaimRefusal, ComponentLoader, DEFAULT_KERNEL_CACHE_BYTES, DeviceHeap,
    ExecutionPlanDraft, ExecutorDomain, KernelCache, Operation, RequestId, ReservedResources,
    ResidencyError, ResidencyStore, ResourceAllocator, ResourceCapacity, ResourceDomainId,
    CatalogError, CatalogFailure, ResourcePlan, ResourcePlanner, SEARCH_VERSION, StateBindings,
    TokenId, TracedWork, TunedEntry, TuningCacheKey, TuningContext, TuningEvent, TuningObserver,
    TuningOrigin, TuningSettings, TuningTrace, UnpreparedUnits, WeightImportError, WorkKind,
    memory::{ClaimDemand, ClaimId, HoldingClass},
    platform::{self, DeviceRequest, OpenedPlatform, PlatformConfig, PlatformError},
    require_normal,
};
use magnitude_family_contracts::{HeadHistory, InputLayout, PreparedModelInput, TokenPlan};
use seismic::{BackendName, Devices, ExecutorSelector};
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
    pub device: ExecutorSelector,
    /// What prices a streamed load's experts again; none without a bank.
    pub pricing: Option<ExpertPricing>,
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
    let catalog = Devices::discover(&[]).map_err(|error| internal(error.to_string()))?;
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
    startup: StartupClaims,
    pub draft: ExecutionPlanDraft,
    /// `draft` before the weight cap: every expert resident.
    pub resident: ExecutionPlanDraft,
    pub opened: OpenedPlatform,
    pub programs: AttestedPrograms,
    /// Each streamed layer's request of its CPU site's verdict
    /// (`ExecutionPlanDraft::site_requests`).
    site_requests: Vec<Option<magnitude_executor::assessment::SiteRequest>>,
    pub backend: BackendName,
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
    startup: StartupClaims,
    draft: ExecutionPlanDraft,
    /// `draft` before the weight cap: every expert resident.
    resident: ExecutionPlanDraft,
    opened: OpenedPlatform,
    backend: BackendName,
    capacity: ResourceCapacity,
    kernel_cache: Option<Arc<KernelCache>>,
    numerical_profile: magnitude_executor::ResolvedNumericalProfile,
}

/// The numerical profile `manifest`'s load is held to.
fn numerical_profile(manifest: &ExecutionManifest) -> Result<magnitude_executor::ResolvedNumericalProfile, LoadError> {
    manifest
        .model
        .numerical_profile
        .clone()
        .unwrap_or_else(magnitude_executor::NumericalProfile::standard)
        .resolve()
        .map_err(internal)
}

/// What a load of a manifest plans before it opens a device.
struct LoadPlans {
    /// The plan before the weight cap: every expert resident, what a streamed
    /// load's growth is priced on.
    resident: ExecutionPlanDraft,
    /// The plan the load runs.
    draft: ExecutionPlanDraft,
    kernel_cache: Option<Arc<KernelCache>>,
}

/// Plan `manifest`'s load on `selected`: the same derivation metadata-only
/// assessment and preview plan through (`planning::plan_execution`), its
/// resident plan kept.
fn plan_load(manifest: &ExecutionManifest, selected: &platform::SelectedDevice) -> Result<LoadPlans, LoadError> {
    let backend = selected.info.backend();
    let planned = |error: crate::planning::ExecutionPlanningError| -> LoadError {
        match error {
            crate::planning::ExecutionPlanningError::Plan(error) => classify_plan(error, backend).into(),
            error => internal(error.to_string()),
        }
    };
    let resident = crate::planning::plan_resident(manifest, selected).map_err(planned)?;
    let kernel_cache = manifest
        .kernel_cache
        .clone()
        .map(|root| KernelCache::open(root, DEFAULT_KERNEL_CACHE_BYTES).map(Arc::new))
        .transpose()
        .map_err(|error| internal(error.to_string()))?;
    let draft = crate::planning::plan_banked(manifest, resident.clone(), kernel_cache.as_deref()).map_err(planned)?;
    Ok(LoadPlans { resident, draft, kernel_cache })
}

/// Select and open the manifest's device and plan the model on it.
fn plan_on_device(
    catalog: Rc<Devices>,
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
    let selected = crate::planning::select_load_device(&catalog, manifest).map_err(platform_error)?;
    if let (Some(root), platform::GraphPricing::Bound(_)) = (&manifest.kernel_cache, selected.graph_cost) {
        touch(root, &selected.identity, &[magnitude_executor::assessment::PointRequest::GraphCost]);
    }
    let backend = selected.info.backend();
    let capacity = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes,
        tensor_operations: magnitude_executor::TensorOperations::of(
            selected.info.forms_tensor_operations(),
        ),
    };
    let LoadPlans { resident, draft, kernel_cache } = plan_load(manifest, &selected)?;
    if let Some(bank) = draft.load().expert_bank() {
        let weights = draft.load().weight_bytes().map_err(internal)?;
        let slot = bank.slot_bytes().map_err(internal)?;
        tracing::info!(
            "expert bank: {} of the widest routed layer's {} experts in {} groups; \
             weights {weights} bytes, {slot} more a slot",
            bank.slots,
            bank.widest(),
            bank.groups.len(),
        );
    }
    tracing::info!(
        "prompt groups chain up to {} places",
        draft.policy().limits().chain_capacity,
    );
    report_load_phase("device selection and planning", &mut phase_started);
    let opened = platform::open_selected(
        &catalog,
        draft.device().selector(),
        PlatformConfig {
            path: manifest.path,
            artifacts: kernel_cache
                .clone()
                .map(|cache| cache as Arc<dyn seismic::ArtifactStore>),
            graph_cost: Some(draft.device().graph_cost()),
            reserves,
        },
    )
    .map_err(platform_error)?;
    report_load_phase("device open", &mut phase_started);
    report_load_memory("device opened", &catalog, opened.device());
    let numerical_profile = numerical_profile(manifest)?;
    let mut startup = StartupClaims {
        heap: DeviceHeap::open(catalog.clone(), reserves, Rc::new(opened.device().clone()))
            .map_err(|error| platform_error(PlatformError::Memory(error)))?,
        held: None,
    };
    // Preparation holds the programs' device storage and runs qualification
    // fixtures before any model weights are imported. It is part of startup,
    // so it must claim its peak through the same heap as later allocations.
    let qualification = AttestedPrograms::qualification_peak_bytes(draft.load()).map_err(internal)?;
    let preparation_peak = AttestedPrograms::DEVICE_BYTES
        .checked_add(qualification)
        .ok_or_else(|| internal("program preparation peak byte count overflows"))?;
    startup.claim("program preparation", preparation_peak, 0)?;
    Ok(PlannedModel {
        startup,
        draft,
        resident,
        opened,
        backend,
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
    device: &seismic::Device,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<TuningReport, LoadError> {
    Ok(TuningReport {
        progress,
        live: RefCell::new(LiveTuning::default()),
        trace: manifest
            .tuning_trace
            .as_deref()
            .map(|path| TuningTrace::begin(path, work, manifest.tuning, numerical_profile, device.properties()))
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
    catalog: Rc<Devices>,
    manifest: &ExecutionManifest,
    package: &Package,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<(), LoadError> {
    let PlannedModel {
        startup: _startup,
        draft,
        opened,
        kernel_cache,
        numerical_profile,
        ..
    } = plan_on_device(catalog.clone(), manifest, package, progress.as_ref())?;
    let searching = Instant::now();
    let report = tuning_report(manifest, TracedWork::Search, &numerical_profile, opened.device(), progress)?;
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
    tracing::info!(
        "searched in {:.2} s, {:.2} s of it tuning {} entries \
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
    report_load_memory("searched", &catalog, opened.device());
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
    catalog: Rc<Devices>,
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
        startup,
        draft,
        resident,
        opened,
        backend,
        capacity,
        kernel_cache,
        numerical_profile,
    } = plan_on_device(catalog.clone(), manifest, package, progress.as_ref())?;
    let preparing = Instant::now();
    let workers = Arc::new(seismic::Workers::of_the_host());
    opened.device().staff(Some(workers.clone()));
    let report = tuning_report(manifest, TracedWork::Load, &numerical_profile, opened.device(), progress)?;
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
            tracing::info!(
                "no stored tuning record for {entry} [{bindings}]: the load \
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
        .finalize_readout(&manifest.definition, programs.readout_choices(), kernel_cache.as_deref())
        .and_then(|draft| draft.finalize_sites(|bank| programs.sited_layers(bank)))
        .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    if draft.load().readout_choices() != programs.readout_choices() {
        programs
            .discard_readout_choices()
            .map_err(|error| LoadError::from(classify_plan(error, backend)))?;
    }
    let site_requests = draft.site_requests(&numerical_profile).map_err(internal)?;
    let tuned = programs.tuned();
    if let Some(trace) = &report.trace {
        trace.conclude(tuned).map_err(internal)?;
    }
    let formed = opened.device().programs_formed();
    tracing::info!(
        "prepared programs in {:.2} s: {} tuned entries from stored records, \
         {} at canonical defaults for want of one; this process compiled {} programs and loaded \
         {} through the platform's cache in {:.2} s",
        preparing.elapsed().as_secs_f64(),
        tuned.len(),
        programs.unprepared(),
        formed.compiles,
        formed.platform_loads,
        formed.platform_load_seconds,
    );
    report_load_memory("programs prepared", &catalog, opened.device());
    Ok(LoadAttempt::Loaded(PreparedPrograms {
        startup,
        draft,
        resident,
        opened,
        programs,
        site_requests,
        backend,
        capacity,
        workers,
    }))
}

/// A load: one code path whose `unprepared` decides only what happens at a
/// tuning unit without a stored record, which it meets while its programs
/// are prepared, before anything of the model is placed.
pub(crate) fn build(
    catalog: Devices,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<LoadAttempt<NativeDomain>, LoadError> {
    let catalog = Rc::new(catalog);
    let PreparedPrograms {
        mut startup,
        draft,
        resident,
        opened,
        mut programs,
        site_requests,
        backend,
        capacity,
        workers,
    } = match prepare(catalog.clone(), manifest, &package, unprepared, progress.clone())? {
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
    tracing::info!(
        "sealed {} target graph classes ({} graphs) in {:.2} s; a group's pass covers at most {} rows; workspace arena {} bytes",
        seal.classes,
        seal.sealed_graphs,
        seal.seconds,
        target_graphs.pass_rows(),
        resources.arena_bytes()
    );
    // What the target's graphs are made of, beside what they cost on this
    // device: what its decode launch's runs replay, as the family is
    // charged for it (`LaunchGraphs::decode`), and its largest launch
    // class's, which the family's live slots hold of its device's graph
    // budget.
    let family = target_graphs.family();
    let census = family.census();
    let graph_cost = opened.device().graph_cost();
    // The family was charged at this cost, so its every census prices.
    let cost = |graphs: seismic::GraphCensus| {
        let bytes = graph_cost.bytes(graphs).expect("a charged family's graphs price at its device's cost");
        bytes.device + bytes.upload
    };
    let decode = family.graphs().decode;
    // Every plan of a family runs in one of its launches, so it has one.
    let largest = census.launches.iter().copied().max_by_key(|graphs| cost(*graphs)).expect("a target family has a launch class");
    tracing::info!(
        "target graph census: {} layouts; of {} launch classes, the decode class replays {} graphs of {} launches ({} bytes charged), the largest {} of {} ({} bytes charged)",
        census.layouts,
        census.launches.len(),
        decode.graphs,
        decode.launches,
        cost(decode),
        largest.graphs,
        largest.launches,
        cost(largest),
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
    tracing::info!(
        "sealed graph host heap {}",
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
        })
    );
    if let Some(state_graphs) = programs.state_graphs() {
        tracing::info!(
            "state copy graph host heap {}",
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
    report_load_memory("graphs sealed", &catalog, opened.device());
    let released = seismic::forget_evaluated_entry_shapes();
    tracing::info!("released {released} evaluated entry shapes");
    report_load_phase("graph and resource planning", &mut phase_started);
    report_load_memory("graphs planned", &catalog, opened.device());
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
    report_load_memory("resources allocated", &catalog, &device);
    // The served file's volume as the job probed it; with no record it reads
    // at its bounds until a later load's.
    if let Some(root) = &manifest.kernel_cache {
        adopt_volumes(root, &device, package.target(), execution_plan.load().expert_bank()).map_err(internal)?;
    }
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
        residency.keep_cpu_site(stored_cpu_site(cache, &device, &site_requests).map_err(internal)?);
    }
    let target_upload = execution_plan
        .load()
        .target_upload_peak_bytes()
        .map_err(|error| internal(error.to_string()))?;
    startup.hold_host_tables(
        execution_plan
            .load()
            .host_table_bytes()
            .map_err(|error| internal(error.to_string()))?,
    )?;
    // Each weight allocation in the storage it lies in, and the upload
    // window, device storage staged through host RAM.
    let target_demand = execution_plan.load().weight_demands().map_err(internal)?[0];
    startup.claim_storage(
        "target import",
        target_demand
            .checked_plus(ClaimDemand { device: target_upload, staged: target_upload, ..ClaimDemand::default() })
            .ok_or_else(|| internal("target import demand overflows"))?,
    )?;
    let import_progress = progress.clone();
    let import_device = device.clone();
    let import_catalog = catalog.clone();
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
                    require_normal(&import_catalog, &import_device, &reserves)?;
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
    tracing::info!(
        "resident target imported in {:.2} s ({} distinct weights)",
        phase_started.elapsed().as_secs_f64(),
        residency.len(),
    );
    let import = residency.mapped_import_report();
    if import.windows != 0 {
        tracing::info!(
            "mapped import {} weights in {} windows: map {:.3} s, \
             prepare {:.3} s, submit {:.3} s, wait {:.3} s, publish {:.3} s",
            import.weights,
            import.windows,
            import.mapping.as_secs_f64(),
            import.preparing.as_secs_f64(),
            import.submitting.as_secs_f64(),
            import.waiting.as_secs_f64(),
            import.publishing.as_secs_f64(),
        );
    }
    report_load_memory("target imported", &catalog, &device);
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
    // The bank's groups, whose host tier lines the load takes.
    let bank = execution_plan.load().expert_bank().cloned();
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
    report_load_memory("domain allocated", &catalog, &device);
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
    // What the load is doing from its warm-up until it serves, in the
    // process's report on demand (`magnitude_executor::diagnostics`).
    let phase = magnitude_executor::diagnostics::Phase::new("load", "warm-up forward");
    let bindings = warm_up(&mut domain, bindings, primes_drafter)?;
    phase.enter("loaded memory report");
    report_loaded_memory(&domain, &catalog, &device).map_err(internal)?;
    phase.enter("pricing the experts: shrink moves and host storage's routes");
    let pricing = bank
        .map(|bank| -> Result<ExpertPricing, LoadError> {
            let lines = domain.expert_bank_lines().ok_or_else(|| internal("a streamed load opened no expert bank"))?;
            let pricing = ExpertPricing::new(manifest, &package, resident, (*device).clone(), bank, lines, site_requests)?;
            domain.measure_experts(expert_measures(&pricing)).map_err(|error| internal(error.to_string()))?;
            Ok(pricing)
        })
        .transpose()?;
    phase.enter("serving");
    domain
        .serve()
        .map_err(|error| internal(format!("the loaded model's memory reading: {error}")))?;
    Ok(LoadAttempt::Loaded(NativeDomain {
        domain,
        bindings,
        plan,
        device: device_selector,
        pricing,
    }))
}

/// What a streamed load unpriced without a kernel cache lacks: where the
/// measurement job's records are.
const UNPRICED: &str = "the load keeps no kernel cache, where the executor's costs are";

/// What prices a streamed load's experts ([`expert_measures`]):
/// held by the load's worker, which prices them once its warm-up ends and
/// again each time the service's measurement job stores new records. No
/// load measures: what is not stored yet it requests of the job
/// ([`Self::new`]).
pub(crate) struct ExpertPricing {
    kernel_cache: Option<std::path::PathBuf>,
    definition: magnitude_family_contracts::ModelDefinition,
    /// The model's fingerprint, its miss records' key, and the most misses
    /// a record of it holds; none where the model has no key, reported once
    /// as the pricing is made, and its records live for the load alone.
    keyed: Option<(crate::fingerprint::ModelFingerprint, u64)>,
    /// The plan before the weight cap: every expert resident.
    resident: ExecutionPlanDraft,
    /// The load's executor's identity, its miss records' other key.
    executor: seismic::Identity,
    device: seismic::Device,
    bank: magnitude_executor::ExpertBankPlan,
    /// The miss record the load stored last, which every store of the
    /// load's records replaces within what every load stored, across
    /// pricings (`MissRecord::restored`).
    previous: Rc<RefCell<Option<magnitude_executor::assessment::MissRecord>>>,
    /// Each layer's lines the bank holds.
    lines: Arc<[magnitude_executor::assessment::BankLines]>,
    /// Each layer's request of its CPU site's verdict, where it holds a
    /// site: its CPU lines wait on it.
    sites: Vec<Option<magnitude_executor::assessment::SiteRequest>>,
    /// The threads that read host memory at once beside the load (its
    /// process's CPUs), which its memory line is keyed by.
    readers: u64,
    /// The points the load's decisions wait on, which it requests of the
    /// measurement job as it loads and touches each time it prices again
    /// while their inputs are pending ([`expert_measures`]), so the job
    /// measures them first.
    requests: Vec<magnitude_executor::assessment::PointRequest>,
}

impl ExpertPricing {
    /// What prices the load's experts, measuring nothing: each layer's lines
    /// the bank holds (`lines`, `ExpertBank::lines`), beside what the
    /// measurement job stores, read at each pricing ([`expert_measures`]):
    /// the executor's costs, a host tier's link line and host memory line,
    /// and each group's shrink move line. Those the load waits on it
    /// requests of the job as it loads, so the job measures them first.
    fn new(
        manifest: &ExecutionManifest,
        package: &Package,
        resident: ExecutionPlanDraft,
        device: seismic::Device,
        bank: magnitude_executor::ExpertBankPlan,
        lines: Arc<[magnitude_executor::assessment::BankLines]>,
        sites: Vec<Option<magnitude_executor::assessment::SiteRequest>>,
    ) -> Result<Self, LoadError> {
        let executor = device
            .topology()
            .executor_identity(device.info().id())
            .ok_or_else(|| internal("the load's executor is not of its topology's revision"))?;
        let (requests, readers) = expert_requests(&bank, &sites)?;
        let keyed = crate::fingerprint::ModelFingerprint::from_directory(package.target().directory())
            .map_err(|error| error.to_string())
            .and_then(|model| {
                let largest = crate::assessment::largest_misses(&executor, &model)?.expect("a streamed load's model streams experts");
                Ok((model, largest))
            });
        let keyed = match keyed {
            Ok(keyed) => Some(keyed),
            Err(reason) => {
                tracing::warn!("the load's miss records are not stored and live for the load alone: {reason}");
                None
            }
        };
        // Requested as the load loads, before its model is resident: the
        // service's job reads them once it is ([`expert_measures`] touches
        // those still pending with every pricing).
        if let Some(root) = &manifest.kernel_cache {
            touch(root, &executor, &requests);
        }
        tracing::info!(
            "the expert host lines of {} layers, {} with CPU lines",
            lines.len(),
            lines.iter().filter(|line| line.cpu.is_some()).count(),
        );
        Ok(Self {
            kernel_cache: manifest.kernel_cache.clone(),
            definition: manifest.definition.clone(),
            keyed,
            resident,
            executor,
            device,
            bank,
            previous: Rc::default(),
            lines,
            sites,
            readers,
            requests,
        })
    }

    /// A pricing of `bank` on `device`, planned as `resident` of
    /// `definition`, its executor's costs and miss records kept in
    /// `kernel_cache`, measured nothing of its own: its layers take no
    /// lines, its shrink no move line, and its model has no key, so
    /// no record is stored. What a test prices a live load by as costs are
    /// stored.
    #[cfg(test)]
    pub(crate) fn unmeasured(
        kernel_cache: std::path::PathBuf,
        definition: magnitude_family_contracts::ModelDefinition,
        resident: ExecutionPlanDraft,
        device: seismic::Device,
        bank: magnitude_executor::ExpertBankPlan,
    ) -> Self {
        let executor = device
            .topology()
            .executor_identity(device.info().id())
            .expect("a device is of its topology's revision");
        Self {
            kernel_cache: Some(kernel_cache),
            definition,
            keyed: None,
            resident,
            executor,
            device,
            bank,
            previous: Rc::default(),
            lines: Arc::from([]),
            sites: Vec::new(),
            readers: 1,
            requests: Vec::new(),
        }
    }
}

/// What a streamed load's experts are priced on now, by `pricing`,
/// measuring nothing, each input in its state (`assessment::Priced`): the
/// bank's lines beside the host memory line, the link line and the move
/// lines the job stored; the records stored where assessment reads them, under the model's
/// fingerprint, where the cache is; and, on the executor's stored costs,
/// the resident step the growth is priced on, `resident` the plan before
/// the weight cap, and the routed pass each streamed layer's prompt groups
/// and GPU lines are priced on (`ExpertLines::priced`, beside the load's
/// host lines). Each call reads what is stored then, so a
/// pricing after the measurement job stored more is priced on it.
pub(crate) fn expert_measures(pricing: &ExpertPricing) -> magnitude_executor::ExpertMeasures {
    use magnitude_executor::assessment::{
        BankLines, CeilingError, ConvertCosts, ExpertLines, ExpertPass, HostLines, MemoryCosts, MemoryError, MoveCosts, MoveError, PointRequest, PricingError,
        Priced, StoredMisses, TierCopyCosts, TierCopyError,
    };
    use magnitude_executor::MissStoreError;
    let (device, bank) = (&pricing.device, &pricing.bank);
    let Some(root) = &pricing.kernel_cache else {
        return magnitude_executor::ExpertMeasures {
            lines: Priced::Unavailable(UNPRICED.into()),
            store: None,
            resident: Priced::Unavailable(UNPRICED.into()),
            pass: Priced::Unavailable(UNPRICED.into()),
            moves: Priced::Unavailable(UNPRICED.into()),
            tier_copy: Priced::Unavailable(UNPRICED.into()),
            convert: Priced::Unavailable(UNPRICED.into()),
        };
    };
    let (topology, info) = (device.topology(), device.info());
    let records = KernelCache::measurement_records(root);
    let executor = pricing.executor.clone();
    // Each record the load stores replaces what it stored last in the
    // stored record, under the store's lock, so concurrent loads of the
    // model each keep their steps (`MissRecord::restored`).
    let store = pricing.keyed.clone().map(|(model, largest)| {
        let store = KernelCache::miss_records(root);
        let previous = pricing.previous.clone();
        Box::new(move |record: &magnitude_executor::assessment::MissRecord| {
            let key = StoredMisses::key(&executor, &model);
            let updated = store.update(&key, largest, |stored: Option<StoredMisses<_>>| {
                let restored = magnitude_executor::assessment::MissRecord::restored(
                    stored.as_ref().map(|stored| &stored.record),
                    previous.borrow().as_ref(),
                    record,
                )?;
                Ok(StoredMisses { executor: executor.clone(), model: model.clone(), record: restored })
            });
            updated.map_err(MissStoreError::Store)?.map_err(MissStoreError::Merge)?;
            *previous.borrow_mut() = Some(record.clone());
            Ok(())
        }) as Box<magnitude_executor::MissStore>
    });
    // The executor's costs walked once: what the load prices on, and
    // whether the job has points of them left, which its request asks.
    let walked = magnitude_executor::assessment::ExecutorCosts::stored(topology, info, &records);
    let costs_outstanding = walked.as_ref().is_ok_and(|walked| walked.outstanding());
    let costs = crate::assessment::priced_costs(walked);
    let pass = costs.clone().derive(|costs| ExpertPass::of_bank(bank, costs));
    // A host tier's record read from host memory, at its memory line.
    let memory = match bank.tier_reservation() {
        None => Priced::Measured(None),
        Some(_) => match MemoryCosts::stored(topology, info, pricing.readers, &records) {
            Ok(memory) => Priced::Measured(Some(memory.byte_seconds)),
            Err(error @ MemoryError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
            Err(error) => Priced::Unavailable(error.to_string()),
        },
    };
    // A CPU site's lines as the job stored them, and whose verdict admits it:
    // with either missing the site serves nothing, and a pricing reads them
    // once they are here.
    let bank_lines = pricing
        .lines
        .iter()
        .zip(&pricing.sites)
        .map(|(lines, site)| {
            let Some(site) = site else { return Ok(*lines) };
            let costs = site.costs(topology, info, &records).map_err(|error| error.to_string())?;
            let admitted = site.verdict(topology, info, &records).measured().is_some();
            Ok(BankLines { cpu: costs.filter(|_| admitted).map(|costs| costs.lines), ..*lines })
        })
        .collect::<Result<Vec<_>, String>>();
    let host = match bank_lines {
        Ok(bank_lines) => memory.derive(|byte_seconds| HostLines::of_bank(bank, &bank_lines, byte_seconds)),
        Err(reason) => Priced::Unavailable(reason),
    };
    // A host tier's link line is host storage's routes of the costs, which
    // the load requested: pending until the job reads them.
    let lines = host.and_then(|host| {
        pass.as_ref().and_then(|pass| match ExpertLines::priced(&host, pass) {
            Ok(lines) => Priced::Measured(lines),
            Err(PricingError::UnreadLink) => Priced::Pending(PricingError::UnreadLink.to_string()),
            Err(error) => Priced::Unavailable(error.to_string()),
        })
    });
    let moves = match MoveCosts::of_bank(topology, info, bank, &records) {
        Ok(moves) => Priced::Measured(moves),
        Err(error @ MoveError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
        Err(error) => Priced::Unavailable(error.to_string()),
    };
    // A host tier's copy on the blocks the job measured: until then a list's
    // whole grid, as the bank opened.
    let tier_copy = match bank.tier_reservation() {
        None => Priced::Measured(None),
        Some(_) => match TierCopyCosts::stored(topology, info, &records) {
            Ok(costs) => Priced::Measured(Some(costs)),
            Err(error @ TierCopyError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
            Err(error) => Priced::Unavailable(error.to_string()),
        },
    };
    // The device's convert on unified memory on the line the job measured:
    // until then the host lands every miss.
    let convert = match bank.tier_reservation() {
        Some(_) => Priced::Measured(None),
        None => match ConvertCosts::lines(topology, info, bank, &records) {
            Ok(lines) => Priced::Measured(Some(lines.into())),
            Err(error @ CeilingError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
            Err(error) => Priced::Unavailable(error.to_string()),
        },
    };
    // The load waits now on its requests the job has points of left to
    // measure: the job measures them first.
    let waiting = pricing
        .requests
        .iter()
        .filter(|request| match request {
            PointRequest::Costs => costs_outstanding,
            request => request.outstanding(topology, info, &records),
        })
        .cloned()
        .collect::<Vec<_>>();
    touch(root, &pricing.executor, &waiting);
    magnitude_executor::ExpertMeasures {
        lines,
        store,
        resident: costs.derive(|costs| crate::assessment::resident_step(&pricing.definition, &pricing.resident, costs)),
        pass,
        moves,
        tier_copy,
        convert,
    }
}

/// What `bank` waits on of the measurement job, in the order the job
/// measures it ([`touch`]), and the host memory readers its pricing asks
/// for: a host tier's link line is host storage's routes of the executor's
/// costs (first, as its share of the walk), its record read from host
/// memory the host memory's line, and its copies' blocks the tier copy's
/// points; then the executor's costs, which every line, pass and resident
/// step of it reads; each moving group's shrink is paced by its move line;
/// each of its `sites` (by streamed layer) serves once admitted with its
/// lines stored. A load serves without them until the job has measured
/// them.
fn expert_requests(
    bank: &magnitude_executor::ExpertBankPlan,
    sites: &[Option<magnitude_executor::assessment::SiteRequest>],
) -> Result<(Vec<magnitude_executor::assessment::PointRequest>, u64), LoadError> {
    use magnitude_executor::assessment::{ConvertCosts, MoveCosts, PointRequest};
    let readers = std::thread::available_parallelism().map_err(|error| internal(error.to_string()))?.get() as u64;
    let tier = bank
        .tier_reservation()
        .map(|_| [PointRequest::HostRoutes, PointRequest::HostMemory { readers }, PointRequest::TierCopy])
        .into_iter()
        .flatten();
    let moves = MoveCosts::forms(bank).map(|(_, form)| PointRequest::Moves(form));
    let sites = sites.iter().flatten().cloned().map(PointRequest::Site);
    let converts = bank.tier_reservation().is_none().then(|| ConvertCosts::forms(bank).map(PointRequest::Convert)).into_iter().flatten();
    Ok((tier.chain([PointRequest::Costs]).chain(moves).chain(sites).chain(converts).collect(), readers))
}

/// The volumes a load of `artifact` reads for: each source for a reader no
/// step waits on and for `bank`'s urgent request.
fn volumes(
    artifact: &magnitude_artifacts::gguf::GgufArtifact,
    bank: Option<&magnitude_executor::ExpertBankPlan>,
) -> Result<Vec<(Arc<magnitude_artifacts::FileSource>, Option<magnitude_artifacts::Urgent>)>, String> {
    let urgent = bank.map(magnitude_executor::ExpertBankPlan::urgent).transpose()?;
    Ok(artifact
        .sources()
        .flat_map(|source| std::iter::once(None).chain(urgent.clone().map(Some)).map(|urgent| (source.clone(), urgent)))
        .collect())
}

/// The request of `source`'s volume for `urgent`.
fn volume_request(
    source: &magnitude_artifacts::FileSource,
    urgent: Option<&magnitude_artifacts::Urgent>,
) -> magnitude_executor::assessment::PointRequest {
    magnitude_executor::assessment::PointRequest::Volume {
        file: source.path().to_owned(),
        urgent: urgent.map(|urgent| (urgent.parts.clone(), urgent.records)),
    }
}

/// Request every point a load of `manifest` waits on of the measurement job,
/// loading nothing: what a load requests as it loads, for a tool that runs
/// the job before it times a load, so the load finds what a served load
/// finds once its job has run. Nothing to do without a kernel cache.
pub fn request_load_points(catalog: &Devices, manifest: &ExecutionManifest, package: &Package) -> Result<(), LoadError> {
    let Some(root) = &manifest.kernel_cache else {
        return Ok(());
    };
    let selected = crate::planning::select_load_device(catalog, manifest).map_err(platform_error)?;
    let LoadPlans { draft, .. } = plan_load(manifest, &selected)?;
    let bank = draft.load().expert_bank();
    let sites = draft.site_requests(&numerical_profile(manifest)?).map_err(internal)?;
    let mut requests = vec![magnitude_executor::assessment::PointRequest::GraphCost];
    if let Some(bank) = bank {
        requests.extend(expert_requests(bank, &sites)?.0);
    }
    let volumes = volumes(package.target(), bank).map_err(internal)?;
    requests.extend(volumes.iter().map(|(source, urgent)| volume_request(source, urgent.as_ref())));
    touch(root, &selected.identity, &requests);
    Ok(())
}

/// What the job stored of the load's CPU sites (`sites`, by streamed layer):
/// their pool's workers, where one is stored, and each layer's lines.
fn stored_cpu_site(
    root: &std::path::Path,
    device: &seismic::Device,
    sites: &[Option<magnitude_executor::assessment::SiteRequest>],
) -> Result<magnitude_executor::assessment::StoredCpuSite, String> {
    use magnitude_executor::assessment::StoredCpuSite;
    let records = KernelCache::measurement_records(root);
    let mut stored = StoredCpuSite::default();
    for (layer, site) in sites.iter().enumerate() {
        let Some(site) = site else { continue };
        if let Some(costs) = site.costs(device.topology(), device.info(), &records).map_err(|error| error.to_string())? {
            stored.workers = stored.workers.max(Some(costs.workers));
            stored.lines.insert(layer, costs.lines);
        }
    }
    Ok(stored)
}

/// Read `artifact`'s volumes as the job's records say, for a reader no step
/// waits on and for `bank`'s urgent request; request what they lack.
fn adopt_volumes(
    root: &std::path::Path,
    device: &seismic::Device,
    artifact: &magnitude_artifacts::gguf::GgufArtifact,
    bank: Option<&magnitude_executor::ExpertBankPlan>,
) -> Result<(), String> {
    use magnitude_executor::assessment::VolumeCosts;
    let records = KernelCache::measurement_records(root);
    let mut requests = Vec::new();
    for (source, urgent) in volumes(artifact, bank)? {
        let urgent = urgent.as_ref();
        match VolumeCosts::stored(device.topology(), device.info(), source.path(), urgent, &records).map_err(|error| error.to_string())? {
            Some(choice) => magnitude_artifacts::adopt_volume(&source, urgent, choice).map_err(|error| error.to_string())?,
            None => requests.push(volume_request(&source, urgent)),
        }
    }
    let executor = device
        .topology()
        .executor_identity(device.info().id())
        .ok_or("the load's executor is not of its topology's revision")?;
    touch(root, &executor, &requests);
    Ok(())
}

/// Request `requests` of the measurement job in the kernel cache at `root`, on `executor`, a
/// load waiting on them now (`assessment::PointRequests`), in the order the job measures them:
/// each at its place in `requests`, which orders a touch's requests (`Waiter::rank`).
/// One that cannot be stored is reported: the load serves without it as it does until the job
/// measures it.
fn touch(root: &std::path::Path, executor: &seismic::Identity, requests: &[magnitude_executor::assessment::PointRequest]) {
    use magnitude_executor::assessment::Waiter;
    let requested = magnitude_executor::assessment::PointRequests::new(root);
    let now = Waiter::now(0);
    for (rank, request) in (0u64..).zip(requests) {
        if let Err(error) = requested.request(executor, request.clone(), Waiter { rank, ..now }) {
            tracing::warn!("the load's request of {request:?} could not be stored: {error}");
        }
    }
}

fn report_load_phase(name: &str, started: &mut Instant) {
    tracing::info!(
        "{name} in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    *started = Instant::now();
}

/// Report what the process holds at one moment of a load, by what holds it
/// ([`seismic::ProcessMemory`]), with Seismic's charges on the opened device
/// and the most the footprint has been so far: one event (`load memory`)
/// whose `record` is a JSON object. Every load reports the same moments, so
/// two loads compare line by line, part by part; a search reports the
/// device's opening and its own end (`searched`), whose peak is its
/// process's.
fn report_load_memory(moment: &str, catalog: &Devices, device: &seismic::Device) {
    tracing::info!(record = %load_memory_record(moment, catalog, device), "load memory");
}

/// The load's end (`loaded`): its memory record with Seismic's charge
/// split by the domain's holders (`MemoryChargeReconciliation`), each in
/// the plan's term that charges it, so the plan's terms read beside what
/// the load holds of each.
fn report_loaded_memory(
    domain: &ExecutorDomain,
    catalog: &Devices,
    device: &seismic::Device,
) -> Result<(), String> {
    let charge = domain.reconcile_memory_charge(&[])?;
    let sum = |terms: &[u64]| {
        terms
            .iter()
            .try_fold(0u64, |sum, &term| sum.checked_add(term))
            .ok_or_else(|| format!("the load's charge terms {terms:?} overflow"))
    };
    let state = sum(&[
        charge.target_state.total(),
        charge.head_state.map_or(0, |state| state.total()),
    ])?;
    // What the device's backend holds beside every charge (its blocks' free
    // space and its recorded graphs' driver memory), which the plan bounds
    // as its bytes beyond allocations.
    let beyond = device.backend_held(seismic::Contract::DeviceOnly).beyond;
    let mut record = load_memory_record("loaded", catalog, device);
    record["charge"] = serde_json::json!({
        "charged": charge.charged,
        "weights": sum(&[charge.target_weights, charge.optional_weights])?,
        "state": state,
        "graphs": sum(&[charge.graph_pools, charge.bound_constants])?,
        "programs": charge.prepared_programs,
        "owned_features": charge.owned_features,
        "external_pins": charge.external_pins,
        "unattributed": charge.unattributed,
        "beyond_allocations": beyond,
    });
    tracing::info!(%record, "load memory");
    Ok(())
}

fn load_memory_record(moment: &str, catalog: &Devices, device: &seismic::Device) -> serde_json::Value {
    let memory = seismic::ProcessMemory::read();
    let ledger = device.device_ledger();
    let held = device.backend_held(seismic::Contract::DeviceOnly);
    let large = seismic::large_page_backing();
    // The host the load runs beside: what the memory bands judge, and what
    // the OS holds outside it.
    let host = match catalog.inventory().host_status() {
        Ok(host) => serde_json::json!({
            "allocatable_bytes": host.allocatable.value,
            "obtainable_bytes": host.obtainable.value,
            "wired_bytes": host.held.map(|held| held.wired_bytes),
            "compressed_bytes": host.held.map(|held| held.compressed_bytes),
        }),
        Err(error) => serde_json::json!({ "unobserved": error.to_string() }),
    };
    serde_json::json!({
        "host": host,
        "moment": moment,
        "footprint_bytes": memory.footprint_bytes,
        "malloc_live_bytes": memory.malloc_live_bytes,
        "graphics_bytes": memory.graphics_bytes,
        "mapped_bytes": memory.mapped_bytes,
        "rest_bytes": memory.rest_bytes(),
        // The device's own ledger on its device store: on a device whose
        // memory is the host's, every charge it makes in that memory.
        "device_charged_bytes": ledger.charged,
        "device_held_bytes": ledger.charged + held.beyond,
        "device_unallocated_bytes": held.unallocated,
        "device_bound_bytes": ledger.bound,
        "device_refused_demand_bytes": ledger.refused,
        // Host pages devices read over their link (CUDA's registered and
        // Vulkan's imported host storage, `seismic_reservation`'s
        // `PageSize::Large`) where the host has large pages, by what backs
        // them: through a translating IOMMU small-page granules read at a
        // fraction of the line, which the transfer line (measured over
        // small-page mapped memory) prices, never the large pages' rate.
        "host_large_page_bytes": large.large_bytes,
        "host_small_page_bytes": large.small_bytes,
        "peak_footprint_bytes": seismic::ProcessMemory::peak_footprint(),
    })
}

/// Startup allocations claimed from the device's heap, the same heap the
/// domain claims from once loaded. Each claim keeps every domain's headroom
/// above its planning reserve and is held through its allocation, until the
/// next claim follows it.
struct StartupClaims {
    heap: DeviceHeap,
    held: Option<ClaimId>,
}

impl StartupClaims {
    /// Claim `allocation` additional bytes of the device's storage, `staged`
    /// of which the backend stages through host RAM; fit charges the staged
    /// bytes to host RAM as well only where host RAM is not a store of the
    /// device's placement (`Fit::host_staging`). Releases the previous
    /// startup claim, whose allocation Seismic now charges.
    fn claim(&mut self, purpose: &str, allocation: u64, staged: u64) -> Result<(), LoadError> {
        self.claim_storage(purpose, ClaimDemand { device: allocation, staged, ..ClaimDemand::default() })
    }

    /// [`Self::claim`] with `upload` bytes of upload storage beside the
    /// `allocation` device bytes.
    fn claim_storage(&mut self, purpose: &str, demand: ClaimDemand) -> Result<(), LoadError> {
        if let Some(held) = self.held.take() {
            self.heap.release(held);
        }
        let claim = self
            .heap
            .claim_storage(demand, HoldingClass::Model)
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
        let identity = |store| {
            let topology = self.heap.device().topology();
            topology
                .store_identity(store)
                .expect("refused store belongs to the opened topology")
        };
        match refusal {
            ClaimRefusal::Blind(error) => platform_error(PlatformError::Memory(error)),
            ClaimRefusal::Reclaim(reading) => LoadError::MemoryPressure {
                purpose: format!("{purpose} ({})", magnitude_executor::Reclaimed(&reading)),
                store: reading.identity,
            },
            ClaimRefusal::Deficit {
                store,
                constraint,
                required,
                available,
                ..
            } => LoadError::InsufficientMemory {
                purpose: format!("{purpose} ({constraint})"),
                store: identity(store),
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
/// first-use cost, so one row suffices. An enabled drafter that keeps its
/// own history also receives one state-only priming row (`primes_drafter`),
/// so its component import and binding complete before readiness; a draft
/// head reading the target's history keeps none, and its component was
/// imported and bound at load. The request's state advances are aborted, so
/// no state survives it.
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
    // A forward whose experts or table rows the step could not obtain (a
    // read that failed, or a park the device gave up) fails the load with
    // why, as it would fail a request.
    let unread = pending.iter().find_map(|pending| match pending.outcome() {
        magnitude_executor::Outcome::WeightReadFailed(error) => Some(error.to_string()),
        // Interrupted at the load: nothing is served yet, so nothing re-runs
        // it; the load fails with why.
        magnitude_executor::Outcome::Interrupted { reason, .. } => Some(reason.clone()),
        _ => None,
    });
    for pending in pending {
        domain
            .abort(pending)
            .map_err(|error| failed(error.to_string()))?;
    }
    if let Some(unread) = unread {
        return Err(failed(unread));
    }
    domain.close(request).map_err(failed)?;
    tracing::info!(
        "warm-up forward in {:.0} ms",
        began.elapsed().as_secs_f64() * 1000.0
    );
    Ok(bindings)
}

/// Reports a search's tuning, or the records a load read: a search's
/// progress to the host, each unit on the engine's diagnostic stream, and
/// every event to the tuning trace the host asked for.
#[derive(Default)]
struct LiveTuning {
    completed: u64,
    total: u64,
    work: Option<TuningWork>,
    units: BTreeMap<(String, String, BTreeMap<String, u64>), (u64, u64)>,
}

struct TuningReport {
    live: RefCell<LiveTuning>,
    progress: Rc<dyn Fn(LoadProgress)>,
    trace: Option<TuningTrace>,
}

impl TuningReport {
    fn report_live(&self, live: &LiveTuning) {
        if live.total == 0 {
            return;
        }
        let (measured_configurations, formed_programs) = live.units.values().fold(
            (0u64, 0u64),
            |(measured, programs), (next_measured, next_programs)| (
                measured.saturating_add(*next_measured),
                programs.saturating_add(*next_programs),
            ),
        );
        (self.progress)(LoadProgress::Tuning {
            completed: live.completed,
            total: live.total,
            work: live.work.clone(),
            measured_configurations,
            formed_programs,
        });
    }
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
                let mut live = self.live.borrow_mut();
                live.completed = *completed as u64;
                live.total = *total as u64;
                self.report_live(&live);
            }
            TuningEvent::WorkStarted { entry, stage } => {
                let mut live = self.live.borrow_mut();
                live.work = Some(TuningWork {
                    entry: (*entry).to_owned(),
                    operation: match stage {
                        seismic::Stage::Discovery => TuningOperation::Discovery,
                        seismic::Stage::Form => TuningOperation::Form,
                        seismic::Stage::Measure => TuningOperation::Measure,
                        seismic::Stage::Promote => TuningOperation::Promote,
                        seismic::Stage::Confirm => TuningOperation::Confirm,
                    },
                });
                self.report_live(&live);
            }
            TuningEvent::Stage(stage) => {
                let mut live = self.live.borrow_mut();
                live.work = None;
                if let Some(counts) = &stage.counts {
                    live.units.insert(
                        (stage.entry.to_owned(), stage.bindings.clone(), stage.statics.clone()),
                        (counts.measured as u64, counts.programs as u64),
                    );
                }
                self.report_live(&live);
            }
            TuningEvent::Started {
                entry,
                bindings,
                points,
            } => tracing::info!("tuning {entry} [{bindings}] at {points} points"),
            // Each unit's mandatory work, each piece of optional work and
            // each record published when the optional work stops are in
            // the tuning trace, the latter two with their forecasts.
            TuningEvent::Mandatory { .. }
            | TuningEvent::Published { .. } => {}
            TuningEvent::Untuned {
                programs,
                forming_seconds,
            } => tracing::info!(
                "formed {programs} untuned programs for the load in \
                 {forming_seconds:.2} s"
            ),
            TuningEvent::Budget(budget) => {
                let mut live = self.live.borrow_mut();
                live.work = None;
                self.report_live(&live);
                // Each refused piece by its unit, so a void load says which
                // units it left unfinished.
                let refused = budget
                    .refused
                    .iter()
                    .map(|refused| {
                        format!(
                            "{} ({:?} piece, {:.2} s priced, refused for its {})",
                            refused.unit, refused.stage, refused.priced_seconds, refused.reason
                        )
                    })
                    .collect::<Vec<_>>();
                tracing::info!(
                    "tuning budget {:.2} s: {} pieces of work used {:.2} s, \
                     {:.2} s unused, {:.2} s over; {} offered pieces did not fit{}{}",
                    budget.budget_seconds,
                    budget.stages,
                    budget.used_seconds,
                    budget.unused_seconds,
                    budget.overrun_seconds,
                    budget.refused.len(),
                    if refused.is_empty() { "" } else { ": " },
                    refused.join("; ")
                );
            }
            TuningEvent::Unprepared {
                entry,
                bindings,
                statics,
            } => tracing::info!(
                "no stored tuning record for {entry} [{bindings}] {statics:?}: \
                 canonical default"
            ),
            TuningEvent::Pinned {
                entry,
                bindings,
                statics,
                choice,
                ..
            } => tracing::info!(
                "pinned {entry} [{bindings}] {statics:?}: {choice:?} (not tuned)"
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
                    tracing::info!(
                        "searched {} [{}]: share {:.1}% of step time; {} \
                         pieces of work: {} of the {} configurations asked for measured, {} \
                         programs formed, serving {:.3} of the defaults; building {:.2} s, \
                         references {:.2} s, forming {:.2} s, measuring {:.2} s, validating \
                         {:.2} s",
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
                tracing::info!(
                    "tuned {} [{}] in {:.2} s ({origin}, {}): {:?} ({} \
                     measured, {} excluded, {} qualification rejections)",
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
                    tracing::info!(
                        "first qualification rejection of {}: {rejection}",
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

    /// The job measures a load's requests in the order the load lists
    /// them: host storage's routes ahead of the executor's whole costs.
    #[test]
    fn a_load_s_requests_are_measured_in_its_order() {
        use magnitude_executor::assessment::{PointRequest, PointRequests};
        let root = tempfile::tempdir().unwrap();
        let topology = Devices::discover(&[]).unwrap().topology();
        let executor = topology.executors().iter().find(|executor| executor.backend() == BackendName::Cpu).unwrap().clone();
        let identity = topology.executor_identity(executor.id()).unwrap();
        let requests = [PointRequest::HostRoutes, PointRequest::Costs];
        touch(root.path(), &identity, &requests);
        let records = seismic::measure::HeldStore::default();
        assert_eq!(PointRequests::new(root.path()).outstanding(&topology, &executor, &records).unwrap(), requests);
    }
}
