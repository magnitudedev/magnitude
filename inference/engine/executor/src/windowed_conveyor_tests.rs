//! End-to-end execution and ledger verification for the bounded-window prompt
//! expert-streaming conveyor (`ExpertBankPlan::window_slots() < widest()`).

use crate::planning::fixtures::{qwen4exp_fixture_definition_of, streamed_experts_manifest};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain,
    Operation, Outcome, PhysicalDecision, PlannedMethod, RequestId, ReservedResources,
    ResidencyStore, ResourceAllocator, ResourceCapacity, ResourceDomainId, ResourceLimits,
    ResourcePlanner, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
    domain::TargetGraphReservation,
    platform,
    test_support::available,
};
use magnitude_artifacts::Package;
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::KvCodec;
use std::rc::Rc;
use std::sync::Arc;

const STREAMING_BACKENDS: [seismic::BackendName; 3] = [
    seismic::BackendName::Metal,
    seismic::BackendName::Cuda,
    seismic::BackendName::Vulkan,
];

const CHUNK_ROWS: usize = 512;
const CHUNKS: usize = 4;
const CONTEXT_LIMIT: u64 = 4096;

#[derive(Clone, Copy)]
enum Mode {
    Resident,
    /// Unwindowed (`W = widest`, `P = 1`) bank at its floor (`WeightBound::Cap` at `W = widest` floor).
    UnwindowedFloor,
    /// Windowed (`W = round_records`, `P >= 4`) bank at its floor (`WeightBound::Cap` at `W = round_records` floor).
    WindowedFloor,
}

type StepBits = Vec<(Option<Vec<u32>>, Option<Vec<u32>>, Option<(u32, u8)>)>;

fn fixture(test: &str, mut definition: ModelDefinition) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let manifest = streamed_experts_manifest(&definition);
    let directory = std::env::temp_dir().join(format!(
        "magnitude-{test}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("qwen4exp-windowed.gguf");
    crate::domain::fixtures::loaded::write_gguf(&manifest, &path, &|_| None);
    let package = Package::open_without_projector(&path).unwrap();
    definition.artifact_identity = package.manifest().identity;
    (directory, Rc::new(definition), Arc::new(package))
}

fn limits(rows: usize, chain_capacity: usize) -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: rows,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: rows,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead: false,
        chain_capacity: std::num::NonZeroUsize::new(chain_capacity).unwrap(),
        conversations: std::num::NonZeroUsize::new(2).unwrap(),
    }
}

fn open_domain(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    mode: Mode,
) -> (
    Box<ExecutorDomain>,
    crate::StateBindings,
    crate::memory::ledger::DeviceLedgers,
) {
    crate::diagnostics::install();
    let release = platform::ReleaseRate::of_machine(None);
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &release,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: device selected: {error}"));
    let device = Rc::new(
        crate::test_support::open(catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("{backend:?}: device opens: {error}")),
    );
    let capacity = ResourceCapacity {
        domain_bytes: crate::StableRoom::of_selected(&selected).capacity().unwrap().min(1024 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
        allocations: seismic::NativeAllocations::EXACT,
    };
    let draft = ExecutionPlanner::prepare(
        &selected,
        &package.manifest(),
        definition,
        ComponentSelection { head: false, vision: false },
        ExecutionPath::Native,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let resident = draft.clone();
    let prices = crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices());
    let draft = match mode {
        Mode::Resident => draft,
        Mode::UnwindowedFloor => {
            let whole = draft
                .clone()
                .with_bank::<crate::AssessmentMemoryCharge>(definition, None, &prices)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                .into_draft();
            let bank = whole.load().expert_bank().expect("whole bank");
            let floor_load = whole
                .load()
                .clone()
                .with_bank_slots(bank.floor(), bank.floor())
                .unwrap();
            let cap = floor_load.weight_bytes().unwrap();
            draft
                .with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(cap)), &prices)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                .into_draft()
        }
        Mode::WindowedFloor => {
            let whole = draft
                .clone()
                .with_bank::<crate::AssessmentMemoryCharge>(definition, None, &prices)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                .into_draft();
            let bank = whole.load().expert_bank().expect("whole bank");
            let windowed_load = whole.load().clone().banked_as(definition, bank, bank.round_records()).unwrap();
            let floor = windowed_load.expert_bank().unwrap().floor();
            let floor_load = windowed_load.with_bank_slots(floor, floor).unwrap();
            let cap = floor_load.weight_bytes().unwrap();
            draft
                .with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(cap)), &prices)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                .into_draft()
        }
    };
    let state = ResourcePlanner::state_plan(
        definition,
        draft.load(),
        draft.policy().method(),
        KvCodec::Dense,
        limits,
        capacity,
    )
    .unwrap();
    let mut programs = AttestedPrograms::prepare_draft(
        &draft,
        &seismic::Charges::tally(&device),
        crate::TuningContext {
            definition,
            weights: &crate::ZeroTuningWeights,
            observer: &crate::UnreportedTuning,
            cache: None,
            numerical_profile: crate::ResolvedNumericalProfile::standard(),
            settings: crate::TuningSettings::STANDARD,
        },
        crate::UnpreparedUnits::ServeDefaults,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let draft = draft
        .finalize_readout(definition, programs.readout_choices(), &crate::StableRoom::of_selected(&selected))
        .and_then(|draft| draft.finalize_sites(|bank| programs.sited_layers(bank)))
        .unwrap();
    let target_graphs = programs
        .prepare_target_graphs(
            &device,
            draft.load(),
            &definition.decoder,
            &state,
            draft.programs().target(),
            limits,
            draft.load().pass(limits),
            &seismic::Workers::new(1),
        )
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits)
        .unwrap();
    programs
        .prepare_auxiliary_graphs(
            &device,
            draft.load(),
            definition,
            draft.programs(),
            &state,
            limits,
            PlannedMethod::Plain,
        )
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let resources_plan = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        programs.drafter_graphs(),
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let planned = draft.clone();
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    let identity = ResourceDomainId::new("windowed-conveyor").unwrap();
    let programs = Rc::new(programs);
    let ledgers = crate::domain::fixtures::fixture_ledgers(&device, &execution, &seismic::Devices::discover(&[]).unwrap());
    let mut recorded = crate::memory::ledger::Recorded::default();
    let scratch = crate::memory::ClaimDemand { device: execution.resources().bytes().scratch, ..Default::default() };
    let resources = crate::domain::fixtures::fixture_charged(
        &device,
        &ledgers,
        &mut recorded,
        crate::memory::ledger::CommittedKind::Arena,
        scratch,
        |charges| {
            ResourceAllocator::allocate(
                &execution,
                charges,
                identity.clone(),
                programs.target_graphs().unwrap(),
                programs.target_readout_graphs().unwrap(),
                programs.drafter_graphs(),
                None,
                programs.state_graphs().unwrap(),
                &ledgers,
            )
        },
    )
    .unwrap();
    let load = execution.load();
    let demand = match load.expert_bank() {
        Some(bank) => load.clone().with_load_slots(bank.slots, bank.host_slots).unwrap().target_weight_demand(),
        None => load.target_weight_demand(),
    }
    .unwrap();
    let mut weights = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, demand)).unwrap();
    let host_tables = crate::memory::ClaimDemand { host: load.host_table_bytes().unwrap(), ..Default::default() };
    let host_tables = ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, host_tables)).unwrap();
    let pools = crate::LoadPools::open(&device, &ledgers, &execution).unwrap();
    let mut residency = ResidencyStore::for_target(
        device.clone(),
        programs.clone(),
        (*execution).clone(),
        identity,
        &ledgers,
        &mut weights,
        pools.clone(),
    )
    .unwrap();
    let target = weights
        .allocate(&device, |charges| residency.load_target(charges, definition, package, &host_tables, Box::new(|_, _| ())))
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    recorded.record(weights);
    recorded.record(host_tables);
    let plan = execution.resources().clone();
    let target_state = plan.allocate_target_state(&seismic::Charges::tally(&device), pools.state.clone()).unwrap();
    let constants = crate::memory::ClaimDemand {
        device: programs.target_graphs().unwrap().binding_constant_bytes().unwrap(),
        ..Default::default()
    };
    let constants = resources.ledgers().changes().change(|keeping| resources.ledgers().claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, constants)).unwrap();
    let request_slot = crate::domain::fixtures::request_slot(resources.ledgers(), &device, &execution, 0);
    let reader = crate::DeviceReader::open(
        Rc::new(seismic::Devices::discover(&[]).unwrap()),
        release,
        device,
    )
    .unwrap();
    let (mut domain, mut bindings) = ExecutorDomain::new(
        execution,
        definition.clone(),
        reader,
        programs,
        resources,
        None,
        None,
        target,
        target_state,
        None,
        recorded,
        constants,
        request_slot,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let costs = crate::assessment::synthetic::costs();
    fn pending<T>() -> crate::assessment::Priced<T> {
        crate::assessment::Priced::Pending("not measured by the test".to_owned())
    }
    let experts = planned.load().expert_bank().map(|bank| crate::ExpertMeasures {
        lines: pending(),
        read: pending(),
        pass: crate::assessment::Priced::Measured(
            crate::assessment::expert_pass(bank, &costs).expect("synthetic costs price pass"),
        ),
        tier_copy: pending(),
        convert: pending(),
    });
    let load_costs = crate::LoadCosts::of(definition, &resident, &planned, &costs).map_err(|error| error.to_string());
    domain
        .measure_load(crate::LoadMeasures { costs: load_costs, experts }, &mut bindings)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    domain.observe_memory();
    (domain, bindings, ledgers)
}

fn prompt_chunk(request: RequestId, chunk: usize, chunks: usize, rows: usize) -> Operation {
    let position = chunk * rows;
    let last = chunk + 1 == chunks;
    let demand = crate::batching::Demand::FEATURES | crate::batching::Demand::LOGITS;
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: (position..position + rows)
            .map(|row| TokenId((row * 37 + 11 + request.0 as usize * 5) as u32 % 256))
            .collect(),
        position,
        conditioning: None,
        demand: if last { demand | crate::batching::Demand::SELECT } else { demand },
        select: last
            .then(|| SelectSpec {
                sampling: Sampling::Greedy,
                seed: 0,
                position: position + rows - 1,
                domain: 0,
                mask: None,
                shaping: Shaping::default(),
                history: None,
            })
            .into_iter()
            .collect(),
        committed: rows,
        prime: None,
        front: Vec::new(),
        chain: None,
    }
}

fn decode_step(request: RequestId, token: u32, position: usize) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Decode,
        tokens: vec![TokenId(token)],
        position,
        conditioning: None,
        demand: crate::batching::Demand::SELECT | crate::batching::Demand::LOGITS,
        select: vec![SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        }],
        committed: 1,
        prime: None,
        front: Vec::new(),
        chain: None,
    }
}

fn step_bits(outcome: &Outcome) -> StepBits {
    let Outcome::Forward { rows, .. } = outcome else {
        panic!("expected forward outcome, got {outcome:?}");
    };
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    rows.iter()
        .map(|row| {
            (
                row.features.as_ref().map(|features| bits(features.read_to_host().unwrap())),
                row.logits.as_ref().map(|logits| bits(logits.read_to_host().unwrap())),
                row.selected.map(|selected| (selected.token.0, selected.status)),
            )
        })
        .collect()
}

fn run_step(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operation: Operation,
    claims: bool,
    backend: seismic::BackendName,
) -> (Outcome, crate::StateBindings) {
    let operations = [operation];
    let reserved = domain
        .reserve(&mut bindings, &operations)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        .into_resources();
    let ReservedResources::Target(reservation) = reserved else {
        panic!("{backend:?}: forward reserves target lane");
    };
    assert_eq!(matches!(reservation, TargetGraphReservation::Claim(_)), claims, "{backend:?}: chain claim");
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
    let (mut pending, bindings) = domain
        .finish_target(flight)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let pending = pending.pop().expect("one operation");
    let outcome = pending.outcome().clone();
    assert!(!matches!(outcome, Outcome::WeightReadFailed(_)), "{backend:?}: {outcome:?}");
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: operations[0].row_count(), ahead: crate::AheadVerdict::default() })
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (outcome, bindings)
}

fn run_workload(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    mode: Mode,
) -> (Vec<StepBits>, u64, Option<(u64, u64, u64)>) {
    let (mut domain, mut bindings, ledgers) =
        open_domain(catalog, definition, package, backend, limits(CHUNK_ROWS, CHUNKS), mode);
    let (claim, allocated, constants) = domain.arena_terms();
    assert!(constants > 0 && claim >= constants && claim == allocated, "{backend:?}: arena terms ({claim}, {allocated}, {constants})");
    assert_eq!(ledgers.device().estimate().accounting_violations(), 0, "{backend:?}: zero device accounting violations at open");
    assert_eq!(ledgers.host().estimate().accounting_violations(), 0, "{backend:?}: zero host accounting violations at open");
    let committed_at_open = ledgers.device().estimate().committed();
    let bank_shape = domain
        .expert_bank_plan()
        .map(|bank| (bank.window_slots(), bank.widest().get(), bank.windows()));

    let requests = [RequestId(1), RequestId(2)];
    for request in requests {
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
    }
    let mut steps = Vec::new();
    let mut selected_tokens = Vec::new();
    for request in requests {
        domain.plan_successors(
            (1..CHUNKS)
                .map(|chunk| prompt_chunk(request, chunk, CHUNKS, CHUNK_ROWS))
                .collect(),
        );
        for chunk in 0..CHUNKS {
            let (outcome, returned) = run_step(
                &mut domain,
                bindings,
                prompt_chunk(request, chunk, CHUNKS, CHUNK_ROWS),
                chunk > 0,
                backend,
            );
            bindings = returned;
            steps.push(step_bits(&outcome));
        }
        let token = steps
            .last()
            .and_then(|bits| bits.last())
            .and_then(|(_, _, selected)| *selected)
            .expect("last prompt chunk selects token")
            .0;
        selected_tokens.push(token);
    }
    for (request, mut token) in requests.into_iter().zip(selected_tokens) {
        for offset in 0..2 {
            let (outcome, returned) =
                run_step(&mut domain, bindings, decode_step(request, token, CHUNKS * CHUNK_ROWS + offset), false, backend);
            bindings = returned;
            let bits = step_bits(&outcome);
            token = bits.last().and_then(|(_, _, selected)| *selected).expect("decode selects token").0;
            steps.push(bits);
        }
    }
    domain.observe_memory();
    assert_eq!(ledgers.device().estimate().accounting_violations(), 0, "{backend:?}: zero device accounting violations after run");
    assert_eq!(ledgers.host().estimate().accounting_violations(), 0, "{backend:?}: zero host accounting violations after run");
    drop(bindings);
    (steps, committed_at_open, bank_shape)
}

/// On every streaming backend this host runs, a 128-expert decoder capped to
/// its minimum prompt scratch window (`window_slots == round_records`, `P >= 4`,
/// `slots == floor`, `host_slots == 0`) streams chained 2048-row prompt groups
/// where cold experts exceed `window_slots` by >4x without deadlock, scores
/// bit-for-bit identically to both the resident load and the unwindowed floor
/// bank across every prompt row and decode step, and reduces the device
/// ledger's committed floor bytes with zero accounting violations.
#[test]
fn a_windowed_conveyor_streams_cold_layers_bit_identically_within_its_bounded_ledger() {
    let mut definition = qwen4exp_fixture_definition_of(3, 128);
    definition.decoder.context_limit = CONTEXT_LIMIT;
    let (directory, definition, package) = fixture("windowed-conveyor", definition);
    {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
            let (resident, _, _) = run_workload(&catalog, &definition, &package, backend, Mode::Resident);
            let (unwindowed, unwindowed_committed, unwindowed_shape) =
                run_workload(&catalog, &definition, &package, backend, Mode::UnwindowedFloor);
            let (windowed, windowed_committed, windowed_shape) =
                run_workload(&catalog, &definition, &package, backend, Mode::WindowedFloor);

            let (uw_slots, uw_widest, uw_windows) = unwindowed_shape.expect("unwindowed bank");
            assert_eq!((uw_slots, uw_widest, uw_windows), (128, 128, 1), "{backend:?}: unwindowed floor holds all 128 scratch slots");

            let (w_slots, w_widest, w_windows) = windowed_shape.expect("windowed bank");
            assert!(
                w_slots < w_widest && w_windows >= 4,
                "{backend:?}: windowed floor stepped down to window_slots={w_slots} of {w_widest} (windows={w_windows})"
            );
            assert!(
                windowed_committed < unwindowed_committed,
                "{backend:?}: windowed floor commits fewer device ledger bytes ({windowed_committed} < {unwindowed_committed})"
            );
            assert_eq!(unwindowed, resident, "{backend:?}: unwindowed floor bank matches resident bit for bit");
            assert_eq!(windowed, resident, "{backend:?}: windowed conveyor matches resident bit for bit across prompt groups and decode");
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}
