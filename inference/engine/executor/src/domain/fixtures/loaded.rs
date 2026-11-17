//! Loads of a written GGUF as the engine's load makes them: weights
//! imported, graphs prepared, sealed and bound, a drafter bound on its
//! first work. Test-only: what tests of real steps, and an owner over a
//! real domain, run on.
// Most of it is the executor's own tests'; the `fixtures` feature serves
// only [`drafted`] and [`plain`] above.
#![cfg_attr(not(test), allow(dead_code))]

use crate::planning::fixtures::{qwen4exp_fixture_definition, stored_table_manifest};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain, PlannedMethod, ResidencyStore,
    ResourceAllocator, ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner, domain::StatusReading, platform,
};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::KvCodec;
use std::rc::Rc;
use std::sync::Arc;

/// A drafting load (an MTP head over the planning fixture's decoder) on
/// the CPU, its memory read as the fixture's alone
/// (`Machine::Still`), its drafter unbound until its first work: the
/// directory its GGUF was written to, for the test to remove once the
/// domain is dropped, the definition, the domain and its bindings.
pub fn drafted(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, ExecutorDomain, crate::StateBindings) {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with(test, mtp, crate::planning::fixtures::fixture_manifest);
    let method = PlannedMethod::Mtp { greedy_proposals: 2, sampled_proposals: 2, widths: crate::DraftWidths::Adaptive };
    let (domain, bindings, _, _) =
        load_read(&catalog, &definition, &package, seismic::BackendName::Cpu, limits(4, false, 1, 0), Banked::Resident, method, KvCodec::Dense, Machine::Still);
    (directory, definition, domain, bindings)
}

/// A plain load of the planning fixture's decoder at `context` rows on the
/// CPU, its memory read as the fixture's alone (`Machine::Still`): what an
/// owner's tests of a request writing toward a long context decode on. The
/// directory its GGUF was written to, the definition, the domain and its
/// bindings, as [`drafted`] gives them.
pub fn plain(test: &str, context: u64) -> (std::path::PathBuf, Rc<ModelDefinition>, ExecutorDomain, crate::StateBindings) {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let mut definition = crate::planning::fixtures::fixture_definition();
    definition.decoder.context_limit = context;
    let (directory, definition, package) = fixture_with(test, definition, crate::planning::fixtures::fixture_manifest);
    let (domain, bindings, _, _) =
        load_read(&catalog, &definition, &package, seismic::BackendName::Cpu, limits(4, false, 1, 0), Banked::Resident, PlannedMethod::Plain, KvCodec::Dense, Machine::Still);
    (directory, definition, domain, bindings)
}

/// A GGUF of every tensor `manifest` names (its n-gram tables IQ4_NL
/// blocks, a streamed load's expert stacks Q8_0 or Q4_0 blocks, everything else
/// F16, BF16 or F32 as it lays them out), with small deterministic values:
/// each float tensor `constant` names holds its value throughout.
pub(crate) fn write_gguf(
    manifest: &magnitude_artifacts::PackageManifest,
    path: &std::path::Path,
    constant: &dyn Fn(&str) -> Option<f32>,
) {
    let mut header = Vec::new();
    let string = |bytes: &mut Vec<u8>, value: &str| {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    };
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(manifest.target.tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    let mut payload = Vec::new();
    let mut state = 0x5eed_u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for tensor in &manifest.target.tensors {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for extent in tensor.shape.iter().rev() {
            header.extend_from_slice(&extent.to_le_bytes());
        }
        header.extend_from_slice(&(tensor.encoding as u32).to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let values = tensor.shape.iter().product::<u64>() as usize;
        match tensor.encoding {
            // Both 32 values a block: d, then 16 bytes of 4-bit codes.
            Encoding::Iq4Nl | Encoding::Q4_0 => {
                for _ in 0..values / 32 {
                    // d = 1/64, then 32 random 4-bit codes.
                    payload.extend_from_slice(&0x2400u16.to_le_bytes());
                    for _ in 0..16 {
                        payload.push(next() as u8);
                    }
                }
            }
            Encoding::Q8_0 => {
                for _ in 0..values / 32 {
                    // d = 1/1024, then 32 random 8-bit values.
                    payload.extend_from_slice(&0x1400u16.to_le_bytes());
                    for _ in 0..32 {
                        payload.push(next() as u8);
                    }
                }
            }
            Encoding::F16 | Encoding::BF16 | Encoding::F32 => {
                // A gated delta decay rate (`ssm_a`, -exp(A_log)) is negative,
                // as every model's is, so the recurrent state decays: a
                // positive one grows it exponentially along the prompt.
                let decay_rate = tensor.name.ends_with(".ssm_a");
                let constant = constant(&tensor.name);
                for _ in 0..values {
                    // Within ±1/16: a small value of the tensor's element.
                    let value = constant.unwrap_or_else(|| {
                        let value = ((next() % 2001) as f32 - 1000.0) / 16_000.0;
                        if decay_rate { -value.abs() } else { value }
                    });
                    match tensor.encoding {
                        Encoding::F16 => payload.extend_from_slice(&half_bits(value).to_le_bytes()),
                        // Its F32 bits' upper half.
                        Encoding::BF16 => payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()),
                        _ => payload.extend_from_slice(&value.to_le_bytes()),
                    }
                }
            }
            // E4M3 codes of a sign, an exponent of one to eight and a mantissa: finite, none of the NaN codes.
            Encoding::F8E4M3 => {
                for _ in 0..values {
                    let code = next();
                    payload.push((((code >> 7) & 1) << 7 | (1 + (code >> 3) % 8) << 3 | (code & 7)) as u8);
                }
            }
            other => panic!("the fixture stores no {other:?} tensor"),
        }
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend_from_slice(&payload);
    std::fs::write(path, header).unwrap();
}

/// The F16 bits nearest a small normal `value`.
pub(crate) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    let mantissa = (bits >> 13) & 0x3ff;
    sign | ((exponent as u16) << 10) | mantissa as u16
}

/// What a test load is priced on, as a load's worker prices it once its
/// warm-up ends (`LoadCosts::of`): the executor's costs, which every load
/// waits for before it serves (the synthetic costs here,
/// `assessment::synthetic::costs`), and the load's plans.
pub(crate) struct Pricing {
    pub(crate) definition: Rc<ModelDefinition>,
    /// The plan before any bank.
    pub(crate) resident: crate::ExecutionPlanDraft,
    /// The plan the load runs.
    pub(crate) planned: crate::ExecutionPlanDraft,
    pub(crate) costs: crate::assessment::ExecutorCosts,
}

impl Pricing {
    /// The load's measures: its costs; a banked load's experts by `lines`
    /// and `convert`, the routed pass the costs price, and every other
    /// input pending, as the job has not stored it.
    pub(crate) fn measures(
        &self,
        lines: crate::assessment::Priced<std::sync::Arc<[crate::assessment::ExpertLines]>>,
        convert: crate::assessment::Priced<Option<std::sync::Arc<[crate::assessment::RecordLine]>>>,
    ) -> crate::LoadMeasures {
        use crate::assessment::{ExpertPass, Priced};
        fn pending<T>() -> Priced<T> {
            Priced::Pending("not measured by the test".to_owned())
        }
        let experts = self.planned.load().expert_bank().map(|bank| crate::ExpertMeasures {
            lines,
            read: pending(),
            pass: Priced::Measured(ExpertPass::of_bank(bank, self.costs.clone()).expect("the synthetic costs price the bank's pass")),
            tier_copy: pending(),
            convert,
        });
        let costs = crate::LoadCosts::of(&self.definition, &self.resident, &self.planned, &self.costs).map_err(|error| error.to_string());
        crate::LoadMeasures { costs, experts }
    }
}

/// The fixture decoder written as a GGUF in a fresh directory named for
/// `test`, and its definition naming the written artifact.
pub(crate) fn fixture(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_of(test, stored_table_manifest)
}

/// [`fixture`] of the tensors `manifest` lays out.
pub(crate) fn fixture_of(
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_with(test, qwen4exp_fixture_definition(), manifest)
}

/// [`fixture_of`] the decoder `definition`.
pub(crate) fn fixture_with(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_valued(test, definition, manifest, &|_| None)
}

/// [`fixture_with`], each tensor `constant` names holding its value.
pub(crate) fn fixture_valued(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
    constant: &dyn Fn(&str) -> Option<f32>,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let manifest = manifest(&definition);
    fixture_stored(test, definition, &manifest, constant)
}

/// [`fixture_valued`] of the tensors `manifest` names.
pub(crate) fn fixture_stored(
    test: &str,
    mut definition: ModelDefinition,
    manifest: &magnitude_artifacts::PackageManifest,
    constant: &dyn Fn(&str) -> Option<f32>,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let directory = std::env::temp_dir().join(format!(
        "magnitude-{test}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("qwen4exp-fixture.gguf");
    write_gguf(manifest, &path, constant);
    let package = Package::open_without_projector(&path).unwrap();
    definition.artifact_identity = package.manifest().identity;
    (directory, Rc::new(definition), Arc::new(package))
}

/// Launches of `rows` rows, `lookahead` or not, in chains of up to
/// `chain_capacity`, exporting up to `logits` rows of logits.
pub(crate) fn limits(rows: usize, lookahead: bool, chain_capacity: usize, logits: usize) -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: rows,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: logits,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead,
        chain_capacity: std::num::NonZeroUsize::new(chain_capacity).unwrap(),
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// How a load holds its routed experts: resident, streamed through an
/// expert bank whose weights stay within a share of the resident load's,
/// in a bank as an uncapped load holds them, or in a bank of about half
/// the slots.
#[derive(Clone, Copy)]
pub(crate) enum Banked {
    Resident,
    /// The bank's cap as a fraction of the resident load's allocation.
    Streamed { numerator: u64, denominator: u64 },
    /// No cap.
    Whole,
    /// Capped at the whole bank's weights less half its slots'.
    Half,
}

/// Load `definition` from `package` on `backend`, an available one, as the
/// engine's load does, within `limits`, with the most rows a group's pass
/// covers (`group_pass_rows`).
pub(crate) fn load(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
) -> (ExecutorDomain, crate::StateBindings, u64, Pricing) {
    load_with(catalog, definition, package, backend, limits, banked, PlannedMethod::Plain)
}

/// [`load`] serving `method`: one that drafts binds the definition's drafter
/// as the engine's load does, its graphs sealed beside the target's and its
/// state allocated.
pub(crate) fn load_with(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
    method: PlannedMethod,
) -> (ExecutorDomain, crate::StateBindings, u64, Pricing) {
    load_coded(catalog, definition, package, backend, limits, banked, method, KvCodec::Dense)
}

/// [`load_with`] keeping its history in `codec`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_coded(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
    method: PlannedMethod,
    codec: KvCodec,
) -> (ExecutorDomain, crate::StateBindings, u64, Pricing) {
    load_read(catalog, definition, package, backend, limits, banked, method, codec, Machine::Live)
}

/// What a loaded fixture's memory readings read.
#[derive(Clone, Copy)]
pub(crate) enum Machine {
    /// The live machine.
    Live,
    /// The machine as first read, moved only by the fixture's own
    /// allocations (`StillHost`): what a test decides by memory is the
    /// fixture's alone.
    Still,
}

/// [`load_coded`], its memory read as `machine`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_read(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
    method: PlannedMethod,
    codec: KvCodec,
    machine: Machine,
) -> (ExecutorDomain, crate::StateBindings, u64, Pricing) {
    // A hung step reports every owner and every thread's stack on SIGUSR1,
    // as the rig asks a job past its estimate.
    crate::diagnostics::install();
    let release = platform::ReleaseRate::of_machine(None);
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &release,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: an available device is selected: {error}"));
    let device = Rc::new(
        crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("{backend:?}: the selected device opens: {error}")),
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
        ComponentSelection {
            head: method.drafts(),
            vision: false,
        },
        ExecutionPath::Native,
        method,
        codec,
        limits,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let resident = draft.clone();
    let draft = match banked {
        Banked::Resident => draft,
        Banked::Streamed { numerator, denominator } => {
            let resident = draft.allocation_bytes(definition).unwrap();
            draft
                .with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(resident * numerator / denominator)), &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices()))
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
                .into_draft()
        }
        Banked::Whole => draft.with_bank::<crate::AssessmentMemoryCharge>(definition, None, &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft(),
        Banked::Half => {
            let whole = draft.clone().with_bank::<crate::AssessmentMemoryCharge>(definition, None, &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft();
            let bank = whole.load().expert_bank().unwrap_or_else(|| panic!("{backend:?}: the uncapped load banks its experts"));
            let cap = whole.load().weight_bytes().unwrap() - bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
            draft.with_bank::<crate::AssessmentMemoryCharge>(definition, Some(crate::WeightBound::Cap(cap)), &crate::assessment::Priced::Measured(crate::assessment::synthetic::plan_prices())).unwrap_or_else(|error| panic!("{backend:?}: {error}")).into_draft()
        }
    };
    let state = ResourcePlanner::state_plan(
        definition,
        draft.load(),
        draft.policy().method(),
        codec,
        limits,
        capacity,
    )
    .unwrap();
    let mut programs = AttestedPrograms::prepare_draft(
        &draft,
        &device,
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
    // As a load finalizes it (`execution`): its readout, and the CPU sites
    // its qualification admitted, which its graphs encode.
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
            method,
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
    let pass_rows = target_graphs.pass_rows();
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let planned = draft.clone();
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    let identity = ResourceDomainId::new("ngram-steps").unwrap();
    let programs = Rc::new(programs);
    let (reading, ledgers): (Rc<dyn StatusReading>, _) = match machine {
        Machine::Live => {
            let live = Rc::new(seismic::Devices::discover(&[]).unwrap());
            let ledgers = super::fixture_ledgers(&device, &execution, &*live);
            (live, ledgers)
        }
        Machine::Still => {
            let clock = super::TestClock::new();
            let still = Rc::new(super::StillHost::of(seismic::Devices::discover(&[]).unwrap(), &device, super::Pressed::default(), clock.clone()));
            let ledgers = super::fixture_ledgers_at(&device, &execution, &*still, &clock);
            assert!(still.ledgers.set(ledgers.clone()).is_ok(), "a fixture opens its ledgers once");
            (still, ledgers)
        }
    };
    // As the engine's load records them: the arena it allocated, the
    // target's weights, each charged to its claim as it was made.
    let mut recorded = crate::memory::ledger::Recorded::default();
    let scratch = crate::memory::ClaimDemand { device: execution.resources().bytes().scratch, ..Default::default() };
    let resources = crate::domain::fixtures::fixture_charged(
        &device,
        &ledgers,
        &mut recorded,
        crate::memory::ledger::CommittedKind::Arena,
        scratch,
        || {
            ResourceAllocator::allocate(
                &execution,
                &device,
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
    // As the engine's load claims the target import, whose claim the host
    // tier of a capped bank backs its load-time slots in.
    let load = execution.load();
    let demand = match load.expert_bank() {
        Some(bank) => load.clone().with_load_slots(bank.slots).unwrap().target_weight_demand(),
        None => load.target_weight_demand(),
    }
    .unwrap();
    let mut weights = ledgers.claim(&device, crate::memory::ledger::CommittedKind::Load, demand).unwrap();
    // As the engine's load claims them: its host tables' row caches.
    let host_tables = crate::memory::ClaimDemand { host: load.host_table_bytes().unwrap(), ..Default::default() };
    let host_tables = ledgers.claim(&device, crate::memory::ledger::CommittedKind::Load, host_tables).unwrap();
    // As the engine's load opens them: the state's pool, and the bank's
    // records pool where it has a bank.
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
        .allocate(|| residency.load_target(definition, package, &host_tables, Box::new(|_, _| ())))
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    recorded.record(weights);
    recorded.record(host_tables);
    // As the engine's load does: the drafter is imported from the store that
    // imported the target, when its first work binds it.
    let head_loader = match method.drafts() {
        true => Some(
            crate::ComponentLoader::head(residency, definition.clone(), package.clone())
                .unwrap_or_else(|error| panic!("{backend:?}: {error}")),
        ),
        false => None,
    };
    let plan = execution.resources().clone();
    let target_state = plan.allocate_target_state(device.clone(), pools.state.clone()).unwrap();
    let head_state = plan.head_state().map(|state| state.allocate(device.clone(), pools.state.clone()).unwrap());
    // The constants its domain binds, as the engine's load claims them.
    let constants = crate::memory::ClaimDemand {
        device: programs.target_graphs().unwrap().binding_constant_bytes().unwrap(),
        ..Default::default()
    };
    let constants = resources.ledgers().claim(&device, crate::memory::ledger::CommittedKind::Load, constants).unwrap();
    let request_slot = crate::domain::fixtures::request_slot(resources.ledgers(), &device, &execution, 0);
    let reader = crate::DeviceReader::open(reading.clone(), release, device).unwrap();
    let (mut domain, mut bindings) = ExecutorDomain::new(
        execution,
        definition.clone(),
        reader,
        programs,
        resources,
        head_loader,
        None,
        target,
        target_state,
        head_state,
        recorded,
        constants,
        request_slot,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    if let Machine::Still = machine {
        super::settle_ledgers(domain.domain.device(), domain.resources.ledgers(), &*reading);
    }
    // As a load ends: priced on the costs it waited for, its memory read as
    // the owner reads it before its first round.
    let pricing = Pricing { definition: definition.clone(), resident, planned, costs: crate::assessment::synthetic::costs() };
    let pending = "not measured by the test";
    domain
        .measure_load(pricing.measures(crate::assessment::Priced::Pending(pending.into()), crate::assessment::Priced::Pending(pending.into())), &mut bindings)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    domain.observe_memory();
    (domain, bindings, pass_rows, pricing)
}
