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
use crate::planning::Holding;
use magnitude_executor::{
    AttestedPrograms, ComponentLoader, DEFAULT_KERNEL_CACHE_BYTES, DeviceReader, MachineReading,
    ExecutionPlanDraft, ExecutorDomain, KernelCache, ModelLoadPlan, Operation, RequestId, ReservedResources,
    ResidencyStore, ResourceAllocator, ResourceCapacity, ResourceDomainId,
    CatalogError, CatalogFailure, PlanError, ResourcePlan, ResourcePlanner, SEARCH_VERSION, StateBindings,
    TokenId, TracedWork, TunedEntry, TuningCacheKey, TuningContext, TuningEvent, TuningObserver,
    TuningOrigin, TuningSettings, TuningTrace, UnpreparedUnits, WorkKind,
    memory::{ClaimDemand, ledger::{CommittedKind, DeviceLedgers, Held, Ledgers, LoadShape, OpenChange, Recorded, Short}},
    platform::{self, DeviceRequest, OpenedPlatform, PlatformConfig, PlatformError},
};
use magnitude_family_contracts::{HeadHistory, InputLayout, PreparedModelInput, TokenPlan};
use seismic::{BackendName, Devices, ExecutorSelector};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

/// A constructed executor domain with the facts readiness reports.
pub(crate) struct NativeDomain {
    pub domain: ExecutorDomain,
    /// The domain's one right to change its stores' bindings.
    pub bindings: StateBindings,
    pub plan: ResourcePlan,
    pub device: ExecutorSelector,
    /// What prices the load again.
    pub pricing: LoadPricing,
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
    startup: Startup,
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
    startup: Startup,
    /// The programs' device storage, claimed as preparation begins.
    programs_claim: Held,
    /// The qualification fixtures' room beside it, while preparation runs.
    qualification_claim: Held,
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
fn plan_load(manifest: &ExecutionManifest, selected: &platform::SelectedDevice, holding: Holding) -> Result<LoadPlans, LoadError> {
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
    let draft = crate::planning::plan_banked::<magnitude_executor::AssessmentMemoryCharge>(manifest, selected, resident.clone(), holding)
        .map_err(planned)?
        .into_draft();
    Ok(LoadPlans { resident, draft, kernel_cache })
}

/// Select and open the manifest's device and plan the model on it.
fn plan_on_device(
    catalog: Rc<Devices>,
    manifest: &ExecutionManifest,
    package: &Package,
    progress: &dyn Fn(LoadProgress),
    holding: Holding,
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
    let backend = selected.info.backend();
    let capacity = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes,
        tensor_operations: magnitude_executor::TensorOperations::of(
            selected.info.forms_tensor_operations(),
        ),
    };
    let LoadPlans { resident, draft, kernel_cache } = plan_load(manifest, &selected, holding)?;
    if let Some(bank) = draft.load().expert_bank() {
        let weights = draft.load().weight_bytes().map_err(internal)?;
        let slot = bank.slot_bytes().map_err(internal)?;
        tracing::info!(
            "expert bank: {} of the widest routed layer's {} experts in {} groups; \
             weights {weights} bytes, {slot} more a slot",
            bank.slots,
            bank.widest(),
            bank.groups().len(),
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
        },
    )
    .map_err(platform_error)?;
    report_load_phase("device open", &mut phase_started);
    report_load_memory("device opened", &catalog, opened.device());
    let numerical_profile = numerical_profile(manifest)?;
    // The load's ledgers size no state before it is planned: no allocation
    // admitted before then is a sequence's.
    let ledgers = DeviceLedgers::open(opened.device(), Arc::new(LoadShape { stores: Vec::new() }));
    let startup = Startup {
        reader: DeviceReader::open(Rc::new(MachineReading::new(catalog.clone())), reserves, Rc::new(opened.device().clone()))
            .map_err(|error| platform_error(PlatformError::Memory(error)))?,
        recorded: Recorded::new(&ledgers),
        ledgers,
    };
    // Preparation holds the programs' device storage and runs qualification
    // fixtures beside it before any model weights are imported: the first
    // allocations the load's ledgers claim.
    let programs_claim = startup.claim_device("program preparation", AttestedPrograms::DEVICE_BYTES)?;
    let qualification = AttestedPrograms::qualification_peak_bytes(draft.load()).map_err(internal)?;
    let qualification_claim = startup.claim_transient(
        "program qualification",
        CommittedKind::Qualification,
        ClaimDemand { device: qualification, ..ClaimDemand::default() },
    )?;
    Ok(PlannedModel {
        startup,
        programs_claim,
        qualification_claim,
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
        startup,
        // Held while the search prepares and qualifies its programs.
        programs_claim: _programs_claim,
        qualification_claim: _qualification_claim,
        draft,
        opened,
        kernel_cache,
        numerical_profile,
        ..
    } = plan_on_device(catalog.clone(), manifest, package, progress.as_ref(), Holding::Placed)?;
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
        // The search reads its room from the load's ledgers.
        Some(&magnitude_executor::LoadRoom { reader: &startup.reader, ledgers: &startup.ledgers }),
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
        model_store: None,
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
    holding: Holding,
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
        mut startup,
        mut programs_claim,
        qualification_claim,
        draft,
        resident,
        opened,
        backend,
        capacity,
        kernel_cache,
        numerical_profile,
    } = plan_on_device(catalog.clone(), manifest, package, progress.as_ref(), holding)?;
    let preparing = Instant::now();
    let workers = Arc::new(seismic::Workers::of_the_host());
    opened.device().staff(Some(workers.clone()));
    let report = tuning_report(manifest, TracedWork::Load, &numerical_profile, opened.device(), progress)?;
    // The programs' storage is made under its claim, one change with its
    // record.
    let mut programs = match programs_claim.allocate(|| AttestedPrograms::prepare_draft(
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
    )) {
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
    // Qualification ran, its fixtures freed; the programs' storage stays.
    drop(qualification_claim);
    startup.recorded.record(programs_claim);
    let draft = draft
        .finalize_readout(&manifest.definition, programs.readout_choices())
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
/// are prepared, before anything of the model is placed. A plan holding
/// its experts whole that its target import's claim cannot hold is loaded
/// again streaming them, its bank fitted by that claim.
pub(crate) fn build(
    catalog: Devices,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<LoadAttempt<NativeDomain>, LoadError> {
    let catalog = Rc::new(catalog);
    match build_held(catalog.clone(), manifest, package.clone(), unprepared, progress.clone(), Holding::Placed)? {
        Built::Attempt(attempt) => Ok(attempt),
        Built::Unheld(refused) => {
            tracing::info!(
                "{refused}: the plan holding the experts whole does not fit the memory free now, \
                 so the load streams them, its bank fitted to what the import admits"
            );
            // A plan placed whole has its whole form's tuning records, not
            // necessarily its streamed form's; units without one serve
            // their canonical defaults rather than stop unprepared, whose
            // Prepare would search the whole form again.
            match build_held(catalog, manifest, package, UnpreparedUnits::ServeDefaults, progress, Holding::Streamed)? {
                Built::Attempt(attempt) => Ok(attempt),
                // Experts the host cannot stream beside their tier are held
                // whole by the streamed plan too: nothing holds them now.
                Built::Unheld(refused) => Err(refused),
            }
        }
    }
}

/// What a load attempt holding its experts as `holding` says built
/// ([`build_held`]).
enum Built {
    Attempt(LoadAttempt<NativeDomain>),
    /// The target import's claim refused a plan holding the experts whole
    /// that could stream them: the load streams them instead.
    Unheld(LoadError),
}

/// [`build`] of a plan holding its experts as `holding` says.
fn build_held(
    catalog: Rc<Devices>,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
    holding: Holding,
) -> Result<Built, LoadError> {
    // The load is one change of the process's memory, until everything it
    // leaves behind has dropped: the owner reads between its allocations,
    // each recorded as it is made, and a failed load's frees are part of it.
    let _load = OpenChange::begin(Ledgers::process().changes());
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
    } = match prepare(catalog.clone(), manifest, &package, unprepared, progress.clone(), holding)? {
        LoadAttempt::Loaded(prepared) => prepared,
        LoadAttempt::Unprepared => return Ok(Built::Attempt(LoadAttempt::Unprepared)),
    };
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
    // device: what each launch in flight beside another holds, its
    // decode launches' pinned graphs, as the family is charged for it
    // (`LaunchGraphs::pinned`), and its largest launch
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
    let pinned = family.graphs().pinned;
    // Every plan of a family runs in one of its launches, so it has one.
    let largest = census.launches.iter().copied().max_by_key(|graphs| cost(*graphs)).expect("a target family has a launch class");
    tracing::info!(
        "target graph census: {} layouts; of {} launch classes, each launch in flight beside another replays {} graphs of {} launches ({} bytes charged), the largest {} of {} ({} bytes charged)",
        census.layouts,
        census.launches.len(),
        pinned.graphs,
        pinned.launches,
        cost(pinned),
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
    // The plan the load runs, which its pricing derives its step from
    // ([`LoadPricing`]).
    let planned = draft.clone();
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
    // The device's store ledgers as the planned load sizes its state; every
    // allocation from here is claimed on them.
    startup.ledgers = DeviceLedgers::open(&device, Arc::new(LoadShape::of(&execution_plan)));
    let mut scratch = startup.claim_device("graph scratch", plan.bytes().scratch)?;
    let resources = scratch
        .allocate(|| {
            ResourceAllocator::allocate(
                &execution_plan,
                &device,
                resource_identity.clone(),
                target_graphs,
                target_readout_graphs,
                programs.drafter_graphs(),
                programs.vision_graphs().map(|graphs| graphs.as_ref()),
                state_graphs.as_ref(),
                &startup.ledgers,
            )
        })
        .map_err(|error| internal(error.to_string()))?;
    startup.recorded.record(scratch);
    // The load's release thread, which the pools' slots and the expert
    // bank's storage release on.
    let releases = resources.releases();
    report_load_phase("resource allocation", &mut phase_started);
    report_load_memory("resources allocated", &catalog, &device);
    let host_tables = execution_plan.load().host_table_bytes().map_err(|error| internal(error.to_string()))?;
    let mut host_tables = startup.claim("host tables", ClaimDemand { host: host_tables, ..ClaimDemand::default() })?;
    // The allocations after the target import are claimed before it, so the
    // import's bank holds what they leave.
    let mut target_state_claim = startup.claim_device(
        "initial target state",
        plan.target_state().initial_committed_bytes().map_err(internal)?,
    )?;
    let head_state_claim = plan
        .head_state()
        .map(|state| Ok::<_, LoadError>((state, startup.claim_device("initial head state", state.initial_committed_bytes().map_err(internal)?)?)))
        .transpose()?;
    let constants_claim = startup.claim_device("target binding constants", target_binding_constants)?;
    // What the load needs to serve beside its weights and its bank's floor is
    // one request's minimum state: claimed with them, under the load's
    // policy, and held for the domain's life, the first request's state
    // drawing from it ([`CommittedKind::StateFloor`]).
    let state_floor = startup.claim_transient(
        "minimum request state",
        CommittedKind::StateFloor,
        ClaimDemand { device: LoadShape::minimum_state(&execution_plan), ..ClaimDemand::default() },
    )?;
    startup.recorded.record_state_floor(state_floor);
    let upload_window = startup.claim_transient(
        "target import",
        CommittedKind::Upload,
        execution_plan.load().target_upload_demand().map_err(internal)?,
    )?;
    // The target import holds the bank at its derived target now, beside
    // the allocations claimed and the speed guarantee, at most its planned
    // slots; the bank grows toward its ceiling as it serves.
    let in_process = magnitude_executor::in_process_growth(&execution_plan, resources.most_places());
    let (execution_plan, planned, mut weights) = match startup.claim_target_import(execution_plan.load(), in_process)? {
        Imported::Planned(weights) => (execution_plan, planned, weights),
        Imported::Fitted(slots, weights) => (
            execution_plan.with_load_slots(slots).map_err(|error| internal(error.to_string()))?,
            planned.with_load_slots(slots).map_err(|error| internal(error.to_string()))?,
            weights,
        ),
        // A plan holding its experts whole that the memory free now does
        // not hold streams them, where they can stream.
        Imported::Unheld(refused) => {
            return match planned.streaming::<magnitude_executor::AssessmentMemoryCharge>(
                &manifest.definition,
                manifest.model.weight_bound,
            ) {
                Ok(_) => Ok(Built::Unheld(refused)),
                Err(PlanError::Unstreamed(_) | PlanError::WeightBound { .. } | PlanError::Resource(_)) => Err(refused),
                Err(error) => Err(internal(error.to_string())),
            };
        }
    };
    // The served file's volume as the job probed it; with no record it reads
    // at its bounds until a later load's.
    if let Some(root) = &manifest.kernel_cache {
        adopt_volumes(root, &device, package.target(), execution_plan.load().expert_bank()).map_err(internal)?;
    }
    // A linked host tier backs to its load-time slots from the bank's open
    // (`ExpertBank::back_host`): its claim is split from the target import's
    // as the store is made, allocated as each slot backs, never before it
    // exists.
    let mut residency = ResidencyStore::for_target(
        device.clone(),
        programs.clone(),
        execution_plan.clone(),
        resource_identity.clone(),
        releases.clone(),
        &startup.ledgers,
        &mut weights,
    )
    .map_err(|error| internal(error.to_string()))?;
    // The rows steps name of a streamed table are kept beside the kernels,
    // for the next load of the model.
    if let Some(cache) = &manifest.kernel_cache {
        residency.keep_table_rows(cache.join("table-rows"));
        residency.keep_cpu_site(stored_cpu_site(cache, &device, &site_requests).map_err(internal)?);
    }
    let import_progress = progress.clone();
    // The import and its record, its host tables' with it, are one change.
    let target = weights.allocate(|| {
        host_tables.allocate(|| {
            residency
                .load_target(
                    &manifest.definition,
                    &package,
                    Box::new(move |completed_bytes, total_bytes| {
                        import_progress(LoadProgress::ImportingWeights {
                            completed_bytes,
                            total_bytes,
                        });
                    }),
                )
                .map_err(|error| import_refused(error, &device, &startup.ledgers))
        })
    })?;
    drop(upload_window);
    startup.recorded.record(weights);
    startup.recorded.record(host_tables);
    tracing::info!(
        "resident target imported in {:.2} s ({} distinct weights)",
        phase_started.elapsed().as_secs_f64(),
        residency.len(),
    );
    let import = residency.import_report();
    if import.batches != 0 {
        tracing::info!(
            "import {} weights in {} batches: prepare {:.3} s, submit {:.3} s, \
             wait {:.3} s, publish {:.3} s",
            import.weights,
            import.batches,
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
            let store = ResidencyStore::for_component(
                device.clone(),
                programs.clone(),
                execution_plan.clone(),
                resource_identity.clone(),
                releases.clone(),
                startup.ledgers.changes().clone(),
            )?;
            ComponentLoader::vision(store, definition.clone(), package.clone())
        })
        .transpose()
        .map_err(|error| internal(error.to_string()))?;
    let target_state = target_state_claim.allocate(|| plan.allocate_target_state(device.clone()).map_err(internal))?;
    startup.recorded.record(target_state_claim);
    let head_state = head_state_claim
        .map(|(state, mut claim)| {
            let allocated = claim.allocate(|| state.allocate(device.clone()).map_err(internal))?;
            startup.recorded.record(claim);
            Ok::<_, LoadError>(allocated)
        })
        .transpose()?;
    let (mut domain, bindings) = ExecutorDomain::new(
        Rc::new(execution_plan),
        definition,
        startup.reader,
        programs,
        resources,
        head_loader,
        vision_loader,
        target,
        target_state,
        head_state,
        startup.recorded,
        constants_claim,
    )
    .map_err(|error| domain_refused(error, &startup.ledgers))?;
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
    let mut bindings = warm_up(&mut domain, bindings, primes_drafter)?;
    // Everything the load allocates is charged: tables formed from here on are
    // outside it (`seismic::Device::seal_record_tables`).
    device.seal_record_tables();
    phase.enter("loaded memory report");
    report_loaded_memory(&domain, &catalog, &device).map_err(internal)?;
    phase.enter("pricing the load: its step, shrink moves and host storage's routes");
    let bank = planned
        .load()
        .expert_bank()
        .map(|bank| -> Result<BankPricing, LoadError> {
            let lines = domain.expert_bank_lines().ok_or_else(|| internal("a streamed load opened no expert bank"))?;
            BankPricing::new(manifest, bank, lines, site_requests)
        })
        .transpose()?;
    let pricing = LoadPricing::new(manifest, resident, planned, (*device).clone(), bank)?;
    domain
        .measure_load(load_measures(&pricing).map_err(internal)?, &mut bindings)
        .map_err(|error| internal(error.to_string()))?;
    phase.enter("serving");
    Ok(Built::Attempt(LoadAttempt::Loaded(NativeDomain {
        domain,
        bindings,
        plan,
        device: device_selector,
        pricing,
    })))
}

/// What a load unpriced without a kernel cache lacks: where the executor's
/// costs are.
const UNPRICED: &str = "the load keeps no kernel cache, where the executor's costs are";

/// What prices a load ([`load_measures`]): held by the load's worker,
/// which prices it once its warm-up ends and again each time the service's
/// measurement job stores new records. No load measures: what is not stored
/// yet a streamed load requests of the job ([`BankPricing::new`]).
pub(crate) struct LoadPricing {
    kernel_cache: Option<std::path::PathBuf>,
    definition: magnitude_family_contracts::ModelDefinition,
    /// The plan before the weight cap: every expert resident.
    resident: ExecutionPlanDraft,
    /// The plan the load runs.
    planned: ExecutionPlanDraft,
    /// The load's executor's identity.
    executor: seismic::Identity,
    device: seismic::Device,
    /// What prices its bank's experts; none without a bank.
    bank: Option<BankPricing>,
}

/// What prices a streamed load's experts beside its executor's costs.
struct BankPricing {
    /// The model's target file: its volume's read line is what its file
    /// misses take.
    storage: std::path::PathBuf,
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
    /// while their inputs are pending ([`load_measures`]), so the job
    /// measures them first.
    requests: Vec<magnitude_executor::assessment::PointRequest>,
}

impl LoadPricing {
    /// What prices the load of `manifest` planned as `planned` (`resident`
    /// before the weight cap) on `device`, measuring nothing: its
    /// executor's costs as the measurement job stores them, read at each
    /// pricing ([`load_measures`]), and a streamed load's experts by `bank`.
    fn new(
        manifest: &ExecutionManifest,
        resident: ExecutionPlanDraft,
        planned: ExecutionPlanDraft,
        device: seismic::Device,
        bank: Option<BankPricing>,
    ) -> Result<Self, LoadError> {
        let executor = device
            .topology()
            .executor_identity(device.info().id())
            .ok_or_else(|| internal("the load's executor is not of its topology's revision"))?;
        if bank.is_some() != planned.load().expert_bank().is_some() {
            return Err(internal("a load's experts are priced where, and only where, its plan banks them"));
        }
        // Requested as the load loads, before its model is resident: the
        // service's job reads them once it is ([`load_measures`] touches
        // those still pending with every pricing).
        if let (Some(root), Some(bank)) = (&manifest.kernel_cache, &bank) {
            touch(root, &executor, &bank.requests);
        }
        Ok(Self {
            kernel_cache: manifest.kernel_cache.clone(),
            definition: manifest.definition.clone(),
            resident,
            planned,
            executor,
            device,
            bank,
        })
    }

    /// The load's bank, which a streamed load's plan holds.
    fn bank_plan(&self) -> Option<&magnitude_executor::ExpertBankPlan> {
        self.planned.load().expert_bank()
    }

    /// A pricing of the streamed fixture model planned on the device
    /// `device_request` selects, `banked` as its largest bank there, if it
    /// plans one, or unbanked, its executor's costs kept in `kernel_cache`
    /// where it keeps one, measured nothing of its own: a bank's layers take
    /// no lines and its shrink no move line. What a test prices a live load
    /// by as costs are stored, through the path a load's worker prices it;
    /// and the device it plans on.
    #[cfg(test)]
    pub(crate) fn fixture(
        device_request: magnitude_executor::platform::DeviceRequest,
        banked: bool,
        kernel_cache: Option<std::path::PathBuf>,
    ) -> (Self, seismic::Device) {
        use magnitude_executor::{fixtures, platform, ComponentSelection, ExecutionPath, ExecutionPlanner, PlannedMethod};
        let catalog = Devices::discover(&[]).unwrap();
        let reserves = platform::MemoryReserves::of_machine(None);
        let selected = platform::select_device(&catalog, ExecutionPath::Native, device_request, &reserves).unwrap();
        let device = catalog.open(catalog.resolve(selected.info.selector()).unwrap()).unwrap();
        let definition = fixtures::qwen4exp_fixture_definition();
        let manifest = fixtures::streamed_experts_manifest(&definition);
        let resident = ExecutionPlanner::prepare(
            &selected,
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: false },
            ExecutionPath::Native,
            PlannedMethod::Plain,
            magnitude_state::KvCodec::Dense,
            fixtures::limits(2, 1, false),
        )
        .unwrap();
        let planned = match banked {
            true => {
                let structure = |draft: &magnitude_executor::ExecutionPlanDraft| {
                    magnitude_executor::assessment::StructuralCharge::of(&definition, draft)
                        .map(|charge| (charge, ()))
                        .map_err(|error| -> magnitude_executor::PlanError { panic!("the fixture outlines: {error}") })
                };
                match resident.floor_forms(&definition, None, structure).unwrap() {
                    magnitude_executor::FloorForms::Banked { bank, .. } => bank.form.draft,
                    // A backend that streams no experts (CPU) holds them whole.
                    magnitude_executor::FloorForms::Resident(held) => held.draft,
                }
            }
            false => resident.clone(),
        };
        let executor = device.topology().executor_identity(device.info().id()).expect("a device is of its topology's revision");
        let bank = planned.load().expert_bank().map(|_| BankPricing {
            storage: kernel_cache.clone().unwrap_or_default(),
            lines: Arc::from([]),
            sites: Vec::new(),
            readers: 1,
            requests: Vec::new(),
        });
        let pricing = Self { kernel_cache, definition, resident, planned, executor, device: device.clone(), bank };
        (pricing, device)
    }
}

/// Store the required figures of `executor` of `topology` in the kernel
/// cache at `cache` as a machine's measurements, fixed rather than timed,
/// so what a test asserts of them holds on any host however loaded: each
/// required point the walk's retrace finds unrecorded stored at 1 ms, the
/// launch graph's host times 0.15 ms (each run waited for) and 0.05 ms
/// (queued) past its device time.
#[cfg(test)]
pub(crate) fn store_fixed_costs(topology: &seismic::DeviceTopology, executor: &seismic::Executor, cache: &std::path::Path) {
    use magnitude_executor::assessment::{ExecutorCosts, StoredCosts};
    use seismic::measure::{Record, RecordStore, Timing};
    let records = KernelCache::measurement_records(cache);
    loop {
        let key = match ExecutorCosts::stored(topology, executor, &records).unwrap() {
            StoredCosts::Missing(key) => key,
            StoredCosts::Complete { .. } => return,
        };
        let median = match key.point.get("timing") {
            Some(&1) => 1.15e-3,
            Some(&2) => 1.05e-3,
            _ => 1e-3,
        };
        RecordStore::store(&records, &key, &Record::Measured(Timing { median, interval: None, samples: 1 })).unwrap();
    }
}

impl BankPricing {
    /// What prices `bank`'s experts, measuring nothing: each layer's lines
    /// the bank holds (`lines`, `ExpertBank::lines`), beside what the
    /// measurement job stores: a host tier's link line and host memory
    /// line, and each group's shrink move line. Those the load waits on it
    /// requests of the job as it loads ([`LoadPricing::new`]), so the job
    /// measures them first.
    fn new(
        manifest: &ExecutionManifest,
        bank: &magnitude_executor::ExpertBankPlan,
        lines: Arc<[magnitude_executor::assessment::BankLines]>,
        sites: Vec<Option<magnitude_executor::assessment::SiteRequest>>,
    ) -> Result<Self, LoadError> {
        let (requests, readers) = expert_requests(bank, &sites)?;
        tracing::info!(
            "the expert host lines of {} layers, {} with CPU lines",
            lines.len(),
            lines.iter().filter(|line| line.cpu.is_some()).count(),
        );
        Ok(Self {
            storage: manifest.store().to_path_buf(),
            lines,
            sites,
            readers,
            requests,
        })
    }
}

/// What the load is priced on now, by `pricing`, measuring nothing: on the
/// executor's stored costs, its own (`LoadCosts::of`, `resident` the plan
/// before the weight cap and `planned` the plan the load runs) and a
/// streamed load's experts', each input in its state
/// ([`expert_measures`]). Each call reads what is stored then, so a pricing
/// after the measurement job stored more is priced on it.
///
/// A load plans only once the costs' required figures are stored
/// (`planning::plan_banked`), so they are never pending here: costs found
/// pending are an invariant violation, the error. Without a kernel cache
/// they are unavailable.
pub(crate) fn load_measures(pricing: &LoadPricing) -> Result<magnitude_executor::LoadMeasures, String> {
    use magnitude_executor::assessment::Priced;
    let Some(root) = &pricing.kernel_cache else {
        fn unpriced<T>() -> Priced<T> {
            Priced::Unavailable(UNPRICED.into())
        }
        return Ok(magnitude_executor::LoadMeasures {
            costs: Err(UNPRICED.into()),
            experts: pricing.bank.as_ref().map(|_| magnitude_executor::ExpertMeasures {
                lines: unpriced(),
                read: unpriced(),
                pass: unpriced(),
                moves: unpriced(),
                tier_copy: unpriced(),
                convert: unpriced(),
            }),
        });
    };
    let records = KernelCache::measurement_records(root);
    // The executor's costs walked once: what the load prices on, and the
    // part of them the job has points of left, which its requests ask.
    let walked = magnitude_executor::assessment::ExecutorCosts::stored(pricing.device.topology(), pricing.device.info(), &records);
    let costs_outstanding = walked.as_ref().ok().and_then(magnitude_executor::assessment::StoredCosts::outstanding);
    let costs = crate::assessment::priced_costs(walked);
    if let Priced::Pending(reason) = &costs {
        return Err(format!("a load priced before its executor's costs were stored, which it plans on: {reason}"));
    }
    let experts = pricing.bank.as_ref().zip(pricing.bank_plan()).map(|(bank, plan)| expert_measures(pricing, bank, plan, root, &costs, costs_outstanding));
    let own = match costs {
        Priced::Measured(costs) => magnitude_executor::LoadCosts::of(&pricing.definition, &pricing.resident, &pricing.planned, &costs).map_err(|error| error.to_string()),
        Priced::Unavailable(reason) => Err(reason),
        Priced::Pending(_) => unreachable!("pending costs returned above"),
    };
    Ok(magnitude_executor::LoadMeasures { costs: own, experts })
}

/// What a streamed load's experts are priced on now, by `pricing` of
/// `load`, its bank `bank`, measuring nothing, each input in its state:
/// the bank's lines beside the host memory line, the link line and the move
/// lines the job stored under `root`; its store's read line; and, on the
/// executor's stored `costs`, the routed pass each streamed layer's prompt
/// groups and GPU lines are priced on (`ExpertLines::priced`, beside the
/// load's host lines). It touches the points the load still waits on, so
/// the job measures them first, the costs' as `costs_outstanding` reads
/// them (`PointRequest::outstanding`).
fn expert_measures(
    load: &LoadPricing,
    pricing: &BankPricing,
    bank: &magnitude_executor::ExpertBankPlan,
    root: &std::path::Path,
    costs: &magnitude_executor::assessment::Priced<magnitude_executor::assessment::ExecutorCosts>,
    costs_outstanding: Option<magnitude_executor::assessment::CostWalk>,
) -> magnitude_executor::ExpertMeasures {
    use magnitude_executor::assessment::{
        BankLines, CeilingError, ConvertCosts, ExpertLines, ExpertPass, HostLines, MemoryCosts, MemoryError, MoveCosts, MoveError,
        Priced, StorageCosts, StorageError, TierCopyCosts, TierCopyError,
    };
    let (topology, info) = (load.device.topology(), load.device.info());
    let records = KernelCache::measurement_records(root);
    // The model store's read line, a standing request of the job.
    let read = match StorageCosts::stored(topology, info, &pricing.storage, &records) {
        Ok(line) => Priced::Measured(line),
        Err(error @ StorageError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
        Err(error) => Priced::Unavailable(error.to_string()),
    };
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
    let lines = host.and_then(|host| pass.as_ref().and_then(|pass| ExpertLines::priced(&host, pass)));
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
    // measure: the job measures them after assessment's standing requests.
    let waiting = pricing
        .requests
        .iter()
        .filter(|request| request.outstanding(topology, info, &records, costs_outstanding))
        .cloned()
        .collect::<Vec<_>>();
    touch(root, &load.executor, &waiting);
    magnitude_executor::ExpertMeasures {
        lines,
        read,
        pass,
        moves,
        tier_copy,
        convert,
    }
}

/// What `bank` waits on of the measurement job at runtime, in the order
/// the job measures it after assessment's standing requests ([`touch`]),
/// and the host memory readers its pricing reads the standing line of (the
/// executor's costs and the host memory's line are standing requests, of
/// no load): a host tier's copies' blocks are the tier copy's points; each
/// moving group's move line prices a bank step's moves; each of its `sites` (by
/// streamed layer) serves once admitted with its lines stored. A load
/// serves without them until the job has measured them.
fn expert_requests(
    bank: &magnitude_executor::ExpertBankPlan,
    sites: &[Option<magnitude_executor::assessment::SiteRequest>],
) -> Result<(Vec<magnitude_executor::assessment::PointRequest>, u64), LoadError> {
    use magnitude_executor::assessment::{ConvertCosts, MoveCosts, PointRequest};
    let readers = std::thread::available_parallelism().map_err(|error| internal(error.to_string()))?.get() as u64;
    let tier = bank
        .tier_reservation()
        .map(|_| PointRequest::TierCopy)
        .into_iter();
    let moves = MoveCosts::forms(bank).map(|(_, form)| PointRequest::Moves(form));
    let sites = sites.iter().flatten().cloned().map(PointRequest::Site);
    let converts = bank.tier_reservation().is_none().then(|| ConvertCosts::forms(bank).map(PointRequest::Convert)).into_iter().flatten();
    Ok((tier.chain(moves).chain(sites).chain(converts).collect(), readers))
}

/// The volumes a load of `artifact` reads for: each source for a reader no
/// step waits on and for `bank`'s urgent request.
fn volumes(
    artifact: &magnitude_artifacts::gguf::GgufArtifact,
    bank: Option<&magnitude_executor::ExpertBankPlan>,
) -> Result<Vec<(Arc<magnitude_artifacts::FileSource>, Option<seismic_io::Urgent>)>, String> {
    let urgent = bank.map(magnitude_executor::ExpertBankPlan::urgent).transpose()?;
    Ok(artifact
        .sources()
        .flat_map(|source| std::iter::once(None).chain(urgent.clone().map(Some)).map(|urgent| (source.clone(), urgent)))
        .collect())
}

/// The request of `source`'s volume for `urgent`.
fn volume_request(
    source: &magnitude_artifacts::FileSource,
    urgent: Option<&seismic_io::Urgent>,
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
    let LoadPlans { draft, .. } = plan_load(manifest, &selected, Holding::Placed)?;
    let bank = draft.load().expert_bank();
    let sites = draft.site_requests(&numerical_profile(manifest)?).map_err(internal)?;
    let mut requests = Vec::new();
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
    use magnitude_executor::assessment::{StoredVolume, VolumeCosts};
    let records = KernelCache::measurement_records(root);
    let mut requests = Vec::new();
    for (source, urgent) in volumes(artifact, bank)? {
        let urgent = urgent.as_ref();
        let stored = VolumeCosts::stored(device.topology(), device.info(), source.path(), urgent, &records).map_err(|error| error.to_string())?;
        let reads = if urgent.is_some() { "decode" } else { "loads" };
        match &stored {
            Some(StoredVolume::Current(curve)) => {
                tracing::info!("volume of {} for {reads}: its curve, {} points", source.path().display(), curve.points.len());
                seismic_io::adopt_volume(&*source, urgent, curve).map_err(|error| error.to_string())?;
            }
            Some(StoredVolume::Earlier(choice)) => {
                tracing::info!(
                    "volume of {} for {reads}: earlier rule's choice, window {} KiB, demand {} KiB, read {} KiB, demand read {} KiB",
                    source.path().display(),
                    choice.window >> 10,
                    choice.demand >> 10,
                    choice.read >> 10,
                    choice.demand_read >> 10,
                );
                seismic_io::adopt_earlier(&*source, urgent, *choice).map_err(|error| error.to_string())?;
            }
            None => tracing::info!("volume of {} for {reads}: unprobed, read at the probe's bounds", source.path().display()),
        }
        let shape = seismic_io::link_shape(&*source, urgent).map_err(|error| error.to_string())?;
        tracing::info!("volume of {} for {reads}: link {shape}", source.path().display());
        // An earlier rule's choice stands until the job probes the current one.
        if !matches!(stored, Some(StoredVolume::Current(_))) {
            requests.push(volume_request(&source, urgent));
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
/// Requests that cannot be stored are reported: the load serves without them as it does until
/// the job measures them.
fn touch(root: &std::path::Path, executor: &seismic::Identity, requests: &[magnitude_executor::assessment::PointRequest]) {
    if let Err(error) = request(root, executor, requests) {
        tracing::warn!("{error}");
    }
}

/// [`touch`]'s requests, the first that cannot be stored failing them.
fn request(
    root: &std::path::Path,
    executor: &seismic::Identity,
    requests: &[magnitude_executor::assessment::PointRequest],
) -> Result<(), String> {
    use magnitude_executor::assessment::Waiter;
    let requested = magnitude_executor::assessment::PointRequests::new(root);
    let now = Waiter::now(0);
    for (rank, request) in (0u64..).zip(requests) {
        requested
            .request(executor, request.clone(), Waiter { rank, ..now })
            .map_err(|error| format!("the load's request of {request:?} could not be stored: {error}"))?;
    }
    Ok(())
}

impl ExecutionManifest {
    /// Ask the measurement job for the standing figures a plan of this manifest on `catalog`
    /// prices its holdings on (`planning::plan_banked`): assessment's standing requests
    /// (`requests::standing`) of the model's own store, in the load's kernel cache, a load
    /// waiting on them now. Its plan's prices pending ([`crate::error::PreviewError::PricesPending`]),
    /// a load asks before it waits for them without the device. Opens no device and measures
    /// nothing.
    pub fn request_prices(&self, catalog: &Devices) -> Result<(), String> {
        let root = self
            .kernel_cache
            .as_deref()
            .ok_or("the load keeps no kernel cache, which holds the machine's figures")?;
        let selected = crate::planning::select_load_device(catalog, self).map_err(|error| error.to_string())?;
        let requests = magnitude_executor::assessment::requests::standing(
            &selected.reading.topology,
            &selected.info,
            self.store(),
        )?;
        request(root, &selected.identity, &requests)
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
    // The host the load runs beside: what it can take, and what the OS
    // holds outside it.
    let host = match catalog.inventory().host_status() {
        Ok(host) => serde_json::json!({
            "allocatable_bytes": host.allocatable.value,
            "reclaimable_bytes": host.reclaimable,
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

/// What the target import's claim took ([`Startup::claim_target_import`]).
enum Imported {
    /// The import as planned, its weights claimed.
    Planned(Held),
    /// The import with its expert bank at these slots, fewer than planned,
    /// its weights claimed.
    Fitted(u64, Held),
    /// Nothing: the plan holds its experts whole, and the memory free now
    /// does not hold them so (the refusal).
    Unheld(LoadError),
}

/// A load's memory as it starts: the device's reading, which reads the
/// device's store ledgers and bounds Seismic's by them alone, and reads a
/// search's room; those ledgers, which claim each allocation the load
/// makes before it is made
/// ([`magnitude_executor::memory::ledger::DeviceLedgers::claim`]), each
/// from a reading taken for it; and the record of what it allocated
/// ([`Recorded`]), which the domain's holdings carve their bytes out of, so
/// no allocation is ever admitted twice nor read as others' memory.
struct Startup {
    reader: DeviceReader,
    ledgers: DeviceLedgers,
    recorded: Recorded,
}

impl Startup {
    /// The device's ledgers read now, Seismic's bounded by them
    /// ([`DeviceReader::read_ledgers`]).
    fn read(&self) {
        self.reader.read_ledgers_now(&self.ledgers);
    }

    /// Claim `demand`, an allocation the load makes and keeps, from a
    /// reading taken now; refused as insufficient memory for `purpose`.
    fn claim(&self, purpose: &str, demand: ClaimDemand) -> Result<Held, LoadError> {
        self.claim_transient(purpose, CommittedKind::Load, demand)
    }

    /// [`Self::claim`] of an allocation of `kind` the load frees as it
    /// starts (its qualification fixtures, its upload window): held while
    /// it lives, never recorded.
    fn claim_transient(&self, purpose: &str, kind: CommittedKind, demand: ClaimDemand) -> Result<Held, LoadError> {
        self.read();
        self.ledgers.claim(self.reader.device(), kind, demand).map_err(|short| refused(purpose, short))
    }

    /// [`Self::claim`] of `bytes` of device storage.
    fn claim_device(&self, purpose: &str, bytes: u64) -> Result<Held, LoadError> {
        self.claim(purpose, ClaimDemand { device: bytes, ..ClaimDemand::default() })
    }

    /// Claim the target import of `load`: each weight allocation in the
    /// storage it lies in, beside the upload window and every allocation
    /// claimed before it. A streamed load's bank imports at its derived
    /// target at this reading ([`magnitude_executor::import_slots`], beside
    /// the guarantee of `in_process`), the most of those slots, down to its
    /// floor, every store holds: what its first reading keeps. Experts held
    /// whole that do not fit are handed back ([`Imported::Unheld`]).
    fn claim_target_import(&self, load: &ModelLoadPlan, in_process: u64) -> Result<Imported, LoadError> {
        const PURPOSE: &str = "target import";
        let bank = load.expert_bank();
        self.read();
        let demand = |slots: u64| -> Result<ClaimDemand, LoadError> {
            match bank {
                Some(_) => load.clone().with_load_slots(slots).map_err(|error| internal(error.to_string()))?.target_weight_demand(),
                None => load.target_weight_demand(),
            }
            .map_err(internal)
        };
        let (floor, planned, most) = match bank {
            None => (0, 0, 0),
            Some(bank) => {
                let device = self.reader.device();
                let most = magnitude_executor::import_slots(&self.ledgers, device, bank, demand(bank.floor())?, in_process);
                (bank.floor(), bank.slots, most)
            }
        };
        match self.ledgers.admit_most(self.reader.device(), CommittedKind::Load, floor, most, demand)? {
            Ok((slots, held)) if slots == planned => Ok(Imported::Planned(held)),
            Ok((slots, held)) => {
                tracing::info!(
                    "the target import holds {slots} of the expert bank's {planned} planned slots: \
                     its derived target now, beside the allocations claimed before it and the speed guarantee",
                );
                Ok(Imported::Fitted(slots, held))
            }
            // Short of room for experts held whole: streamed, they may fit.
            Err(short) if bank.is_none() => Ok(Imported::Unheld(refused(PURPOSE, short))),
            Err(short) => Err(refused(PURPOSE, short)),
        }
    }
}

/// The executor domain's refusal to be made: a store short of what the
/// domain commits there (its arena, its expert bank's floor, landing rows
/// or cache) is the store's [`Short`], refused as a startup allocation is
/// ([`refused`]), a store whose reading failed included; anything else is
/// internal.
fn domain_refused(error: magnitude_executor::DomainError, ledgers: &DeviceLedgers) -> LoadError {
    use magnitude_executor::{CapacityError, DomainError, ResourceKind};
    let store = |resource| match resource {
        ResourceKind::DeviceMemory => Some(ledgers.device().store()),
        ResourceKind::HostStaging => Some(ledgers.host().store()),
        _ => None,
    };
    match error {
        DomainError::Capacity(CapacityError { resource, required, available }) => match store(resource) {
            Some(store) => refused("the executor domain", Short::Over { store, required, over: required.saturating_sub(available) }),
            None => internal(format!("the executor domain: {resource:?} requires {required} units; {available} available")),
        },
        DomainError::Blind(reason) => LoadError::MemoryObservationUnavailable { reason: format!("the executor domain: {reason}") },
        error => internal(error.to_string()),
    }
}

/// The target import's failure: an allocation its store had no room for
/// is that store's [`Short`], refused as a startup allocation is
/// ([`refused`]); anything else is internal.
fn import_refused(error: magnitude_executor::ResidencyError, device: &seismic::Device, ledgers: &DeviceLedgers) -> LoadError {
    use magnitude_executor::{ResidencyError, WeightImportError};
    match error {
        ResidencyError::Import(WeightImportError::Refused { contract, required, available }) => {
            // The ledger's own figures at the refusal: what Seismic's bound
            // lacked of the allocation, against what the ledger holds.
            let ledger = ledgers.of_contract(device, contract);
            let snapshot = ledger.snapshot();
            tracing::warn!(
                required,
                available,
                committed = snapshot.committed(),
                cache = snapshot.cache(),
                allocated = snapshot.allocated(),
                releasing = snapshot.releasing(),
                budget = snapshot.budget(),
                load_budget = snapshot.budget_for(seismic::Reclaim::CleanFileCache),
                others = snapshot.others(),
                allocatable = snapshot.allocatable(),
                "the target import was refused: the ledger's figures at the refusal",
            );
            refused(
                "target import",
                Short::Over { store: ledger.store(), required, over: required.saturating_sub(available) },
            )
        }
        error => internal(error.to_string()),
    }
}

/// A startup allocation `short` refused, for `purpose`: a store whose
/// reading failed is the reading's failure, never a shortage.
fn refused(purpose: &str, short: Short) -> LoadError {
    match short {
        Short::Blind { store, reason } => LoadError::MemoryObservationUnavailable {
            reason: format!("{purpose}: the reading of {store:?} failed: {reason}"),
        },
        Short::Over { store, required, over } => LoadError::InsufficientMemory {
            purpose: purpose.to_owned(),
            store,
            memory: InsufficientMemory { required, available: required - over },
        },
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
    // Its token is host-known, as an admitted prompt's are: its table rows
    // are prefetched before its step is queued, so it waits on none.
    for prefetched in domain
        .prefetch_prompt(request, 0, &[TokenId(0)])
        .map_err(failed)?
    {
        prefetched.wait();
    }
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
            TuningEvent::Surveyed {
                entry,
                bindings,
                progress,
            } => match progress.formed {
                0 => tracing::info!(
                    "surveying {entry} [{bindings}]: forming and measuring {} configurations",
                    progress.configurations
                ),
                formed => tracing::info!(
                    "surveying {entry} [{bindings}]: {formed} of {} configurations formed, {} measured",
                    progress.configurations,
                    progress.measured
                ),
            },
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
    /// them, after assessment's standing requests, which no load names.
    #[test]
    fn a_load_s_requests_are_measured_in_its_order() {
        use magnitude_executor::assessment::{PointRequest, PointRequests};
        let root = tempfile::tempdir().unwrap();
        let topology = Devices::discover(&[]).unwrap().topology();
        let executor = topology.executors().iter().find(|executor| executor.backend() == BackendName::Cpu).unwrap().clone();
        let identity = topology.executor_identity(executor.id()).unwrap();
        let requests = [PointRequest::HostMemory { readers: 2 }, PointRequest::HostMemory { readers: 1 }];
        touch(root.path(), &identity, &requests);
        let records = seismic::measure::HeldStore::default();
        let costs = magnitude_executor::assessment::ExecutorCosts::stored(&topology, &executor, &records).ok().and_then(|walked| walked.outstanding());
        let storage = root.path();
        let mut standing = magnitude_executor::assessment::requests::standing(&topology, &executor, storage).unwrap();
        // A CPU executor's graphs hold no memory: its graph cost is never outstanding.
        standing.retain(|request| request.outstanding(&topology, &executor, &records, costs));
        assert_eq!(
            PointRequests::new(root.path()).outstanding(&topology, &executor, storage, &records, costs).unwrap(),
            magnitude_executor::assessment::Outstanding { standing, loads: requests.to_vec() }
        );
    }

    /// A target import an allocation's store had no room for reaches the
    /// client as that store's insufficient memory, with its figures, never
    /// as an internal failure: a 929,955,840-byte bank allocation with
    /// 60,887,171 bytes left in device storage's store.
    #[test]
    fn an_import_short_of_room_is_its_stores_insufficient_memory() {
        use magnitude_executor::{ResidencyError, WeightImportError, memory::ledger::Ledgers};
        let catalog = Devices::discover(&[]).unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, Arc::new(LoadShape { stores: Vec::new() }));
        let short = ResidencyError::Import(WeightImportError::Refused {
            contract: seismic::Contract::DeviceOnly,
            required: 929_955_840,
            available: 60_887_171,
        });
        match import_refused(short, &device, &ledgers) {
            LoadError::InsufficientMemory { store, memory, .. } => {
                assert_eq!(store, ledgers.device().store());
                assert_eq!((memory.required, memory.available), (929_955_840, 60_887_171));
            }
            other => panic!("an import short of room is refused for memory: {other:?}"),
        }
    }
}
