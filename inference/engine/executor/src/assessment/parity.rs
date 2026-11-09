//! Placement parity with the resident-only planner this one replaced: every
//! served catalog target, on every executor of every recorded machine, at
//! the standard service's limits and at the KL harness's, is placed whole
//! (and fits) wherever the resident-only planner's charge fits. Plans only:
//! no device is opened and no payload read.
//!
//! The resident-only planner's charge is ported from its source at
//! `c15abaf8a`, `inference/engine/executor/src/planning/assessment.rs`
//! (cited below as `main:<line>`). Its composition and the terms it held
//! differently are its own exact expressions; every other term is this
//! planner's own derivation of the same quantity for the same load, a
//! resident load of one prompt-group place and one conversation, which the
//! resident-only planner derived the same way. Each such reuse is named
//! where it is made.

use super::catalog::served_targets;
use crate::platform::{DomainRole, GraphPricing, MemoryReserves};
use crate::{
    AssessmentFit, AssessmentFitVerdict, AssessmentMemoryCharge, AttestedPrograms, ComponentSelection, ExecutionPath,
    ExecutionPlanDraft, ExecutionPlanner, Placed, PlannedMethod, ResourceLimits, StableRoom,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use seismic::fixtures::recorded;
use seismic::{BackendName, DeviceTopology, Executor, Status};
use std::num::NonZeroUsize;

/// The service's KV codec (`ModelPolicy::default`), on both sides.
const CODEC: KvCodec = KvCodec::AffineK8V4;

/// The KL harness's fixtures decoded together and scored rows per forward
/// (`harness logits`, `--batch 8 --verify-width 1`).
const KL_BATCH: usize = 8;
const KL_VERIFY: usize = 1;

/// Contexts the KL harness is measured at (`mrig kl --context`).
const KL_CONTEXTS: [u64; 2] = [4096, 16_384];

/// The smallest context a model is served at (`SERVED_CONTEXT_FLOOR`).
const SERVED_FLOOR: u64 = 32_768;

/// One workload a row plans: its label, its context, and its limits.
#[derive(Clone, Copy)]
struct Workload {
    label: &'static str,
    context: u64,
    limits: ResourceLimits,
}

/// The standard service's limits (`standard_service_limits`: 512 prefill
/// tokens, 32 decode tokens, one conversation) as the engine resolves them
/// (`planning::resource_limits`): lookahead where the backend is not the
/// CPU's and the entry gathers no per-layer inputs (`ModelPolicy::resolve`).
fn service(backend: BackendName, definition: &ModelDefinition) -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: 512,
        max_launch_slots: 512,
        max_selected_rows: 32,
        max_drafting_slots: 32,
        exported_logits_rows: 0,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead: backend != BackendName::Cpu && definition.decoder.entry.per_layer.is_none(),
        chain_capacity: NonZeroUsize::MIN,
        conversations: NonZeroUsize::MIN,
    }
}

/// The KL harness's load (`harness::logits::run`): 512 prefill rows,
/// `batch × verify` decode rows, logits exported for the larger of those and
/// 512, no lookahead, `batch` sequences.
fn kl(conversations: usize) -> ResourceLimits {
    let rows = KL_BATCH * KL_VERIFY;
    ResourceLimits {
        max_launch_rows: 512,
        max_launch_slots: 512,
        max_selected_rows: rows,
        max_drafting_slots: rows,
        exported_logits_rows: rows.max(512),
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: NonZeroUsize::MIN,
        conversations: NonZeroUsize::new(conversations).unwrap(),
    }
}

/// The workloads of `definition` on `backend`: the service at its largest
/// served tier (its declared context) and at the served floor below it, and
/// the KL harness at each of its contexts within the declared one (a
/// fixture of `context` tokens loads `context + verify`).
fn workloads(backend: BackendName, definition: &ModelDefinition) -> Vec<Workload> {
    let declared = definition.decoder.context_limit;
    let mut workloads = vec![Workload { label: "service", context: declared, limits: service(backend, definition) }];
    if declared > SERVED_FLOOR {
        workloads.push(Workload { label: "service", context: SERVED_FLOOR, limits: service(backend, definition) });
    }
    for context in KL_CONTEXTS.map(|context| context + KL_VERIFY as u64) {
        if context <= declared {
            workloads.push(Workload { label: "kl", context, limits: kl(KL_BATCH) });
        }
    }
    workloads
}

/// What the RTX 3090's graphs hold as each backend measured them, as the
/// engine's recorded setups price them (`tests/recorded`); every other
/// recorded device is priced at its backend's bound, as an unmeasured load
/// is (`platform::graph_pricing`). A macOS host builds no Vulkan backend, so
/// it knows no Vulkan bound.
fn pricing(machine: &str, backend: BackendName) -> Option<GraphPricing> {
    Some(match (machine, backend) {
        ("rtx3090", BackendName::Cuda) => {
            GraphPricing::Measured(seismic::GraphCost::Pooled { graph: 92_663, launch: 8_102, granule: 2 << 20 })
        }
        ("rtx3090", BackendName::Vulkan) => GraphPricing::Measured(seismic::GraphCost::Chunked { launch: 0, chunk: 256 << 10 }),
        (_, BackendName::Vulkan) if cfg!(target_os = "macos") => return None,
        _ => GraphPricing::Bound(seismic::unmeasured_graph_cost(backend)),
    })
}

/// A recorded machine's reading before any load: stable fit reads no
/// headroom (the engine's recorded setups, `tests/recorded`).
fn unopened(topology: &DeviceTopology) -> Status {
    let none = seismic::Reported::new(0, seismic::ReportedBy::MachAllocatablePages);
    Status {
        sample: seismic::Sample {
            taken: std::time::SystemTime::now(),
            host_ram: Ok(seismic::HostStatus {
                allocatable: none,
                obtainable: seismic::Reported::new(0, seismic::ReportedBy::MachMovablePages),
                limits: Vec::new(),
                limit_visibility: seismic::LimitVisibility::Complete,
                displacement: None,
                kernel_pressure: None,
                free_target: None,
                held: None,
                displaced_bytes: None,
            }),
            volumes: Vec::new(),
            executors: topology.executors().iter().map(|executor| executor.id()).collect(),
            grants: Vec::new(),
        },
        runtime: Default::default(),
    }
}

/// The resident-only planner's charge of a resident load whose terms this
/// planner charges as `base` (a load of one prompt-group place and one
/// conversation, `main_basis`), on `backend`. Its composition, `main:862`
/// (`AssessmentMemoryTerms::charge`), term by term:
/// - prepared = weights + `AttestedPrograms::DEVICE_BYTES` + graph pools +
///   recorded graphs + the worker's device bytes (`main:719`);
/// - recorded graphs = `seismic::recorded_graph_device_bytes(backend)`, a
///   fixed 192 MiB on CUDA and Vulkan and none elsewhere (`c15abaf8a`
///   `seismic/runtime/src/native/mod.rs:47`, `graph_replays.rs:24`), in
///   device storage, never the census of the graphs a load records;
/// - loading = startup state + the startup import peak (`main:869`), with no
///   loaded-state phase;
/// - serving peak = state + the component import peak (`main:879`);
/// - allocation = prepared + max(serving peak, loading) (`main:882`);
/// - serving = prepared + state;
/// - no prompt group's class, which it had no term for.
///
/// Reused from `base`, as this planner derives them for the same load: the
/// weights (each in the storage it lies in: `main:966` placed all but the
/// shared tables on the device and those in upload storage, which an
/// unbanked load's demand is, asserted), the state at the fit depth and at
/// startup (one conversation, `1 + lookahead` successor banks at one place,
/// as `main:771` counts), the graph pools at one place, the worker's device
/// bytes, the import peaks and every host term (`main:906`).
fn main_charge(base: &AssessmentMemoryCharge, backend: BackendName) -> AssessmentMemoryCharge {
    let demand = base.weight_demand;
    assert_eq!(
        (demand.host, demand.reserved),
        (0, 0),
        "a resident load holds weights in host or reserved storage: {demand:?}"
    );
    let weights = demand.checked_bytes().unwrap();
    let recorded_graph_bytes = match backend {
        BackendName::Cuda | BackendName::Vulkan => 192 << 20,
        BackendName::Metal | BackendName::Cpu => 0,
    };
    let bounds = crate::AssessmentMemoryBounds { recorded_graph_bytes, recorded_graph_upload_bytes: 0, ..base.bounds };
    let prepared = weights
        + AttestedPrograms::DEVICE_BYTES
        + bounds.graph_bytes
        + bounds.recorded_graph_bytes
        + bounds.device_process.bytes().unwrap();
    let loading = base.startup_state_bytes + bounds.startup_additional_bytes;
    let serving_peak = base.state_bytes + bounds.component_upload_bytes;
    AssessmentMemoryCharge {
        exact_resident_bytes: weights + base.state_bytes,
        loaded_state_bytes: 0,
        bounds,
        serving_bytes: prepared + base.state_bytes,
        allocation_bytes: prepared + serving_peak.max(loading),
        staging_bytes: bounds.staging_upload_bytes.max(bounds.component_upload_bytes),
        ..base.clone()
    }
}

/// The allocation domain's room and requirement in `fit`, and its verdict.
fn need(fit: &AssessmentFit) -> (u64, u64, bool) {
    let allocation = fit
        .domains
        .iter()
        .find(|domain| domain.role == DomainRole::Allocation)
        .expect("a fit reaches its allocation domain");
    (
        allocation.capacity_bytes - allocation.reserve_bytes,
        allocation.required_bytes,
        matches!(fit.verdict, AssessmentFitVerdict::Fits),
    )
}

/// A charge's terms, as the table prints them, in GB.
fn terms(charge: &AssessmentMemoryCharge, chain: usize, successors: usize) -> String {
    let gb = |bytes: u64| bytes as f64 / 1e9;
    let bounds = &charge.bounds;
    format!(
        "weights {:.2} state {:.2} startup {:.2} loaded {:.2} succ {successors} pools {:.3}@{chain} recorded {:.3}+{:.3}up \
         process {:.3} import {:.2} component {:.2} allocation {:.2}",
        gb(charge.weight_demand.checked_bytes().unwrap()),
        gb(charge.state_bytes),
        gb(charge.startup_state_bytes),
        gb(charge.loaded_state_bytes),
        gb(bounds.graph_bytes),
        gb(bounds.recorded_graph_bytes - bounds.recorded_graph_upload_bytes),
        gb(bounds.recorded_graph_upload_bytes),
        gb(bounds.device_process.bytes().unwrap()),
        gb(bounds.startup_additional_bytes),
        gb(bounds.component_upload_bytes),
        gb(charge.allocation_bytes),
    )
}

/// One row: a target on one executor of one machine under one workload.
struct Row {
    line: String,
    /// Main fits resident and this planner does not place it whole and
    /// fitting.
    regression: bool,
    /// The row's planning broke an invariant.
    panicked: bool,
}

fn plan(definition: &ModelDefinition, manifest: &magnitude_artifacts::PackageManifest, device: &crate::platform::SelectedDevice, limits: ResourceLimits) -> Result<ExecutionPlanDraft, String> {
    ExecutionPlanner::prepare(
        device,
        manifest,
        definition,
        ComponentSelection { head: false, vision: false },
        ExecutionPath::Native,
        PlannedMethod::Plain,
        CODEC,
        limits,
    )
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn row(
    name: &str,
    definition: &ModelDefinition,
    manifest: &magnitude_artifacts::PackageManifest,
    machine: &str,
    topology: &DeviceTopology,
    status: &Status,
    executor: &Executor,
    workload: Workload,
) -> Row {
    let reserves = MemoryReserves::of_machine(None);
    let backend = executor.backend();
    let head = format!("{name} | {machine}:{} | {}@{}", backend.as_str(), workload.label, workload.context);
    let failed = |stage: &str, error: String| Row { line: format!("{head} | {stage} fails: {error}"), regression: false, panicked: false };
    let mut definition = definition.clone();
    definition.decoder.context_limit = workload.context;
    let Some(pricing) = pricing(machine, backend) else {
        return failed("pricing", "this host knows no Vulkan graph bound".to_owned());
    };
    let device = match crate::platform::selected(topology, status, &reserves, executor.clone(), pricing) {
        Ok(device) => device,
        Err(error) => return failed("selection", error.to_string()),
    };
    let room = StableRoom { topology, executor, status, reserves: &reserves };
    // The resident-only planner's load: this load at one conversation
    // (`main` charged one, whatever the harness decoded) and one place.
    let main_limits = ResourceLimits { conversations: NonZeroUsize::MIN, ..workload.limits };
    let main_basis = match plan(&definition, manifest, &device, main_limits) {
        Ok(draft) => draft,
        Err(error) => return failed("planning", error),
    };
    let main = match crate::assessment::execution_memory_charge(&definition, &main_basis) {
        Ok(base) => main_charge(&base, backend),
        Err(error) => return failed("charging", format!("{error:?}")),
    };
    let (main_room, main_need, main_fits) = match main.assess_fit(topology, executor, status, &reserves) {
        Ok(fit) => need(&fit),
        Err(error) => return failed("main's fit", error),
    };
    // This planner's load and its placement, as `plan_banked` places it.
    let draft = match plan(&definition, manifest, &device, workload.limits) {
        Ok(draft) => draft,
        Err(error) => return failed("planning", error),
    };
    let whole = match draft.held_whole(&definition) {
        Ok(whole) => whole,
        Err(error) => return failed("holding whole", error.to_string()),
    };
    let whole_charge = match crate::assessment::execution_memory_charge(&definition, &whole) {
        Ok(charge) => charge,
        Err(error) => return failed("charging whole", format!("{error:?}")),
    };
    let (_, whole_need, whole_fits) = match room.fit(&whole, &whole_charge) {
        Ok(fit) => need(&fit),
        Err(error) => return failed("whole's fit", error.to_string()),
    };
    let verdict = match draft.placed::<AssessmentMemoryCharge>(&definition, None) {
        Ok(Placed::Whole(placed)) => {
            let fits = match room.fit(placed.draft(), placed.charge()) {
                Ok(fit) => need(&fit).2,
                Err(error) => return failed("placed fit", error.to_string()),
            };
            let form = if placed.draft().load().expert_bank().is_some() { "whole bank" } else { "resident" };
            (format!("{form}, {}", if fits { "fits" } else { "short" }), fits)
        }
        Ok(Placed::Short(_)) => ("streams".to_owned(), false),
        Err(error) => (format!("refused: {error}"), false),
    };
    let gb = |bytes: u64| bytes as f64 / 1e9;
    let whole_limits = whole.policy().limits();
    let line = format!(
        "{head} | room {:.2} | main {:.2} {} | ours whole {:.2} {} | placed {} | main: {} | ours: {}",
        gb(main_room),
        gb(main_need),
        if main_fits { "fits" } else { "short" },
        gb(whole_need),
        if whole_fits { "fits" } else { "short" },
        verdict.0,
        terms(&main, 1, 1 + usize::from(main_limits.lookahead)),
        terms(&whole_charge, whole_limits.chain_capacity.get(), whole_limits.target_successor_banks()),
    );
    Row { line, regression: main_fits && !verdict.1, panicked: false }
}

/// Every served catalog target is placed whole, fitting, on every recorded
/// machine's every executor under every workload where the resident-only
/// planner's charge of it fits; the table of both charges, term by term,
/// is printed for every row.
#[test]
fn every_target_is_placed_whole_wherever_the_resident_planner_fits_it() {
    let machines: [(&str, DeviceTopology); 9] = [
        ("m4_pro", recorded::m4_pro()),
        ("m4_pro_without_unified_memory", recorded::m4_pro_without_unified_memory()),
        ("rtx3090", recorded::rtx3090()),
        ("gb10", recorded::gb10()),
        ("strix_halo_linux", recorded::strix_halo_linux()),
        ("strix_halo_windows", recorded::strix_halo_windows()),
        ("lunar_lake", recorded::lunar_lake()),
        ("ryzen_igpu_and_rtx4070", recorded::ryzen_igpu_and_rtx4070()),
        ("a10_8q", recorded::a10_8q()),
    ];
    let targets = served_targets();
    let jobs = targets
        .iter()
        .flat_map(|target| machines.iter().map(move |machine| (target, machine)))
        .collect::<Vec<_>>();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, NonZeroUsize::get);
    let mut rows: Vec<Row> = std::thread::scope(|scope| {
        let workers = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut rows = Vec::new();
                    while let Some(&(target, machine)) = jobs.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                        let (name, definition, manifest) = (&target.0, &target.1, &target.2);
                        let (machine, topology) = (machine.0, &machine.1);
                        let status = unopened(topology);
                        for executor in topology.executors() {
                            for workload in workloads(executor.backend(), definition) {
                                // A row that panics is reported as its own failure, so
                                // the rest of the sweep still prints.
                                let computed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    row(name, definition, manifest, machine, topology, &status, executor, workload)
                                }))
                                .unwrap_or_else(|panic| Row {
                                    line: format!(
                                        "{name} | {machine}:{} | {}@{} | panics: {}",
                                        executor.backend().as_str(),
                                        workload.label,
                                        workload.context,
                                        panic
                                            .downcast_ref::<String>()
                                            .map(String::as_str)
                                            .or_else(|| panic.downcast_ref::<&str>().copied())
                                            .unwrap_or("a panic without a message")
                                    ),
                                    regression: false,
                                    panicked: true,
                                });
                                // Each row as it is computed, so a partial sweep still
                                // yields its rows.
                                println!("{}{}", if computed.regression { "REGRESSION " } else { "" }, computed.line);
                                rows.push(computed);
                            }
                        }
                    }
                    rows
                })
            })
            .collect::<Vec<_>>();
        workers.into_iter().flat_map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
    });
    rows.sort_by(|a, b| a.line.cmp(&b.line));
    let panicked = rows.iter().filter(|row| row.panicked).map(|row| row.line.as_str()).collect::<Vec<_>>();
    let regressions = rows.iter().filter(|row| row.regression).map(|row| row.line.as_str()).collect::<Vec<_>>();
    assert!(
        panicked.is_empty() && regressions.is_empty(),
        "of {} rows, {} broke an invariant and {} fit the resident-only planner and are not placed whole and fitting:\n{}\n{}",
        rows.len(),
        panicked.len(),
        regressions.len(),
        panicked.join("\n"),
        regressions.join("\n")
    );
}
