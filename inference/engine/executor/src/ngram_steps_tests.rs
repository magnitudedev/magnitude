//! A decoder with an n-gram input steps on the device with real
//! submissions: its weights imported from a written GGUF, its graphs
//! prepared, sealed and bound as a load binds them, and target steps of the
//! host's tokens and of the device's selection run through its n-gram block.

use crate::planning::tests::{qwen4exp_fixture_definition, stored_table_manifest};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain,
    Operation, Outcome, PhysicalDecision, PlannedMethod, ReservedResources, ResidencyStore,
    ResourceAllocator, ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner,
    RequestId, SelectSpec, Sampling, Shaping, TokenId, WorkKind, domain::TargetGraphReservation,
    platform,
};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::KvCodec;
use std::rc::Rc;

/// A GGUF of every tensor `definition` names (its n-gram tables IQ4_NL
/// blocks, everything else F16), with small deterministic values: what the
/// steps compute is not checked, only that they run.
fn write_gguf(definition: &ModelDefinition, path: &std::path::Path) {
    let manifest = stored_table_manifest(definition);
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
            Encoding::Iq4Nl => {
                for _ in 0..values / 32 {
                    // d = 1/64, then 32 random 4-bit codes.
                    payload.extend_from_slice(&0x2400u16.to_le_bytes());
                    for _ in 0..16 {
                        payload.push(next() as u8);
                    }
                }
            }
            Encoding::F16 => {
                for _ in 0..values {
                    // Within ±1/16: a small F16.
                    let value = ((next() % 2001) as f32 - 1000.0) / 16_000.0;
                    payload.extend_from_slice(&half_bits(value).to_le_bytes());
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
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    let mantissa = (bits >> 13) & 0x3ff;
    sign | ((exponent as u16) << 10) | mantissa as u16
}

fn forward(request: RequestId, kind: WorkKind, tokens: &[u32], position: usize, select: bool) -> Operation {
    Operation::Forward {
        request,
        kind,
        tokens: tokens.iter().map(|token| TokenId(*token)).collect(),
        position,
        conditioning: None,
        demand: if select {
            crate::batching::Demand::SELECT
        } else {
            crate::batching::Demand::NONE
        },
        select: select
            .then(|| SelectSpec {
                sampling: Sampling::Greedy,
                seed: 0,
                position: position + tokens.len() - 1,
                domain: 0,
                mask: None,
                shaping: Shaping::default(),
                history: None,
            })
            .into_iter()
            .collect(),
        committed: tokens.len(),
        prime: None,
    }
}

/// On Metal and on the CPU: a prompt chunk and a decode step of the host's
/// tokens, then the decode step of the token the device selected, which
/// claims the queued lookahead and reads its token rows from the selection.
#[test]
fn an_ngram_decoder_steps_on_host_and_selected_tokens() {
    let Ok(catalog) = seismic::Devices::discover(&[]) else {
        return;
    };
    let directory = std::env::temp_dir().join(format!(
        "magnitude-ngram-steps-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("qwen4exp-fixture.gguf");
    let mut definition = qwen4exp_fixture_definition();
    write_gguf(&definition, &path);
    let package = Package::open_without_projector(&path).unwrap();
    definition.artifact_identity = package.manifest().identity;
    let definition = Rc::new(definition);
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Cpu] {
        let Some((mut domain, mut bindings)) = load(&catalog, &definition, &package, backend)
        else {
            continue;
        };
        let request = RequestId(1);
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        // A prompt chunk and a decode step of the host's tokens.
        let (_, returned) = run(
            &mut domain,
            bindings,
            forward(request, WorkKind::Prefill, &[5, 1], 0, false),
            false,
            backend,
        );
        let (outcome, returned) = run(
            &mut domain,
            returned,
            forward(request, WorkKind::Decode, &[9], 2, true),
            false,
            backend,
        );
        let Outcome::Forward { rows } = outcome else {
            panic!("{backend:?}: a decode step has a forward outcome");
        };
        let selected = rows
            .last()
            .and_then(|row| row.selected)
            .unwrap_or_else(|| panic!("{backend:?}: the decode step selects a token"));
        assert_eq!(selected.status, 0, "{backend:?}");
        // The device's selection is the next step's token: it claims the
        // lookahead queued behind the decode step.
        let (_, returned) = run(
            &mut domain,
            returned,
            forward(request, WorkKind::Decode, &[selected.token.0], 3, true),
            true,
            backend,
        );
        drop(returned);
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Reserve, submit, finish and accept one target step; with `claims`, the
/// step must claim the queued lookahead.
fn run(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operation: Operation,
    claims: bool,
    backend: seismic::BackendName,
) -> (Outcome, crate::StateBindings) {
    let operations = [operation];
    let rows = operations[0].row_count();
    let reserved = domain
        .reserve(&mut bindings, &operations)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        .into_resources();
    let ReservedResources::Target(reservation) = reserved else {
        panic!("{backend:?}: a forward reserves the target lane");
    };
    assert_eq!(
        matches!(reservation, TargetGraphReservation::Claim(_)),
        claims,
        "{backend:?}: whether the step claims the queued lookahead"
    );
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
    let (mut pending, bindings) = domain
        .finish_target(flight)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let pending = pending.pop().expect("one operation");
    let outcome = pending.outcome().clone();
    assert!(
        !matches!(outcome, Outcome::TableReadFailed(_)),
        "{backend:?}: {outcome:?}"
    );
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: rows })
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (outcome, bindings)
}

/// Load `definition` from `package` on `backend` as the engine's load
/// does, lookahead on; `None` without such a device.
fn load(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Package,
    backend: seismic::BackendName,
) -> Option<(ExecutorDomain, crate::StateBindings)> {
    let reserves = platform::MemoryReserves::standard();
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &reserves,
    )
    .ok()?;
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap(),
    );
    let limits = ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 2,
        max_selected_rows: 2,
        max_drafting_slots: 2,
        exported_logits_rows: 0,
        max_images_per_request: magnitude_artifacts::MAX_IMAGES_PER_REQUEST,
        max_image_cells: 0,
        lookahead: true,
    };
    let capacity = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes.min(1024 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
    };
    let draft = ExecutionPlanner::prepare(
        &selected,
        &package.manifest(),
        definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        ExecutionPath::Native,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
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
    let draft = draft
        .finalize_readout(definition, programs.readout_choices())
        .unwrap();
    let target_graphs = programs
        .prepare_target_graphs(
            &device,
            draft.load(),
            &definition.decoder,
            &state,
            draft.programs().target(),
            limits,
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
        .unwrap();
    let resources_plan = ResourcePlanner::plan_with_state(
        state,
        &target_graphs,
        &target_readout_graphs,
        None,
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let execution = Rc::new(draft.admit(resources_plan).unwrap());
    let identity = ResourceDomainId::new("ngram-steps").unwrap();
    let programs = Rc::new(programs);
    let resources = ResourceAllocator::allocate(
        &execution,
        &device,
        identity.clone(),
        programs.target_graphs().unwrap(),
        programs.target_readout_graphs().unwrap(),
        None,
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    let mut residency = ResidencyStore::new(
        device.clone(),
        programs.clone(),
        (*execution).clone(),
        identity,
    )
    .unwrap();
    let target = residency
        .load_target(definition, package, Box::new(|_, _| Ok(())))
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let plan = execution.resources().clone();
    let target_state = plan.allocate_target_state(device.clone()).unwrap();
    let heap = crate::DeviceHeap::open(
        Rc::new(seismic::Devices::discover(&[]).unwrap()),
        reserves,
        device,
    )
    .unwrap();
    Some(
        ExecutorDomain::new(
            execution,
            definition.clone(),
            heap,
            programs,
            resources,
            None,
            None,
            target,
            target_state,
            None,
        )
        .unwrap_or_else(|error| panic!("{backend:?}: {error}")),
    )
}
