//! Manual complete-model qualification against the ordinary unsliced path.
//! No artifacts, physical identities or automatic placement policy in CI.
use super::*;
use crate::{
    platform, memory::HoldingClass, AttestedPrograms, ComponentSelection, DeviceHeap,
    ExecutionPath, ExecutionPlanner, KernelCache, NativeGraphPool, PlannedMethod, ResidencyStore,
    ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner, TargetLaunchInputs,
    TargetTokens, TuningContext, UnreportedTuning, ValidatedTargetLaunch,
};
use crate::programs::{ProgramSubmission, TargetProgram};
use crate::domain::{PipelineNativeFamily, ProgramFamily};
use crate::programs::native_target::{NativeTargetProgram, TargetOutput};
use magnitude_artifacts::Package;
use magnitude_batching::{
    ClassLimits, Demand, Draw, DrawKind, Row, RowHistory, Select, Shaping, Slot,
    ValidatedTargetBatch,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{
    GrowthChoice, KvCodec, OwnedStateAdvance, SequenceState, StoreBindings, TentativeAdvance,
};
use seismic::{BackendName, Device, DeviceCatalog, DeviceSelector};
use serde::Deserialize;
use std::rc::Rc;

#[derive(Deserialize)]
struct Fixture {
    vocabulary: usize,
    stop_tokens: Vec<i32>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    text: String,
    tokens: Vec<i32>,
    teacher_tokens: Vec<i32>,
}
fn limits() -> ResourceLimits {
    ResourceLimits {
        max_launch_rows: 2,
        max_launch_slots: 1,
        max_selected_rows: 1,
        max_drafting_slots: 1,
        exported_logits_rows: 1,
        max_images_per_request: 1,
        lookahead: false,
    }
}
fn prepare(
    package: &Package,
    definition: &ModelDefinition,
    selector: DeviceSelector,
    cache: &KernelCache,
) -> (
    Rc<Device>,
    DeviceHeap,
    crate::ExecutionPlanDraft,
    AttestedPrograms,
    ResourceCapacity,
) {
    let catalog = DeviceCatalog::discover().unwrap();
    let reserves = platform::MemoryReserves::standard();
    let selected = platform::select_device(
        &catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Selector(selector),
        &reserves,
    )
    .unwrap();
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector).unwrap())
            .unwrap(),
    );
    let mut heap = DeviceHeap::open(catalog, reserves, device.clone()).unwrap();
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
        limits(),
    )
    .unwrap();
    let claim = heap
        .claim(
            AttestedPrograms::qualification_peak_bytes(draft.load()).unwrap(),
            draft.load().target_upload_peak_bytes().unwrap(),
            HoldingClass::InFlight,
        )
        .unwrap();
    let programs = AttestedPrograms::prepare_draft(
        &draft,
        &device,
        TuningContext {
            definition,
            weights: package,
            observer: &UnreportedTuning,
            cache: Some(cache),
        },
    )
    .unwrap();
    heap.release(claim);
    let capacity = ResourceCapacity {
        domain_bytes: selected.assessment_capacity_bytes,
    };
    eprintln!(
        "qualification prepared selector={} capacity={} tuned={}",
        selector,
        capacity.domain_bytes,
        programs.tuned().len()
    );
    (device, heap, draft, programs, capacity)
}
struct Control {
    device: Rc<Device>,
    heap: DeviceHeap,
    claim: crate::memory::ClaimId,
    program: NativeTargetProgram,
    store: StoreBindings,
    target: NativeGraphPool,
    readout: NativeGraphPool,
    hidden: usize,
    vocabulary: usize,
}
impl Drop for Control {
    fn drop(&mut self) {
        self.heap.release(self.claim);
    }
}
impl Control {
    fn new(
        package: &Package,
        definition: &ModelDefinition,
        selector: DeviceSelector,
        cache: &KernelCache,
    ) -> Self {
        let (device, mut heap, draft, mut programs, capacity) =
            prepare(package, definition, selector, cache);
        let state = ResourcePlanner::state_plan(
            definition,
            draft.load(),
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits(),
            capacity,
        )
        .unwrap();
        let graphs = programs
            .prepare_target_graphs(
                &device,
                draft.load(),
                &definition.decoder,
                &state,
                draft.programs().target(),
                limits(),
            )
            .unwrap();
        let readout = programs
            .prepare_target_readout_graphs(&device, draft.load(), &definition.decoder, limits())
            .unwrap();
        programs
            .prepare_auxiliary_graphs(
                &device,
                draft.load(),
                definition,
                draft.programs().vision(),
                state.target_state(),
                None,
                limits(),
                0,
            )
            .unwrap();
        let plan = ResourcePlanner::plan_with_state(
            state.clone(),
            &graphs,
            &readout,
            None,
            None,
            programs.state_graphs().unwrap(),
        )
        .unwrap();
        // Test owner retains a conservative local peak across bounded state growth.
        let peak = plan
            .startup_peak_bytes()
            .checked_add(512 * 1024 * 1024)
            .unwrap();
        let claim = heap
            .claim(
                peak,
                draft.load().target_upload_peak_bytes().unwrap(),
                HoldingClass::InFlight,
            )
            .unwrap();
        programs.install_target_graphs(graphs);
        programs.install_target_readout_graphs(readout);
        let programs = Rc::new(programs);
        let domain = ResourceDomainId::new("ordinary-qualification").unwrap();
        let mut residency = ResidencyStore::new(
            device.clone(),
            programs.clone(),
            draft.admit(plan.clone()).unwrap(),
            domain.clone(),
        )
        .unwrap();
        let resident = residency
            .load_target(definition, package, Box::new(|_, _| {}))
            .unwrap();
        let program = programs
            .bind_target(resident, definition.decoder.clone())
            .unwrap();
        let arena = device.execution_arena(plan.arena_bytes()).unwrap();
        let target = NativeGraphPool::new(
            domain.clone(),
            programs.target_graphs().unwrap().family(),
            plan.target_graph(),
            &arena,
        )
        .unwrap();
        let readout = NativeGraphPool::new(
            domain,
            programs.target_readout_graphs().unwrap().family(),
            plan.target_readout_graph(),
            &arena,
        )
        .unwrap();
        let store = state.target_state().allocate(device.clone()).unwrap();
        Self {
            device,
            heap,
            claim,
            program,
            store,
            target,
            readout,
            hidden: definition.decoder.hidden as usize,
            vocabulary: definition.decoder.vocabulary as usize,
        }
    }
    fn step(&mut self, source: SequenceState, tokens: &[i32]) -> (SequenceState, Record) {
        self.store
            .provision_with_growth(&source.demands(tokens.len()), 1, GrowthChoice::Minimum)
            .unwrap();
        let advance =
            OwnedStateAdvance::begin(source, tokens.len()).unwrap_or_else(|(_, e)| panic!("{e}"));
        let launch = launch(
            &self.store,
            advance,
            tokens,
            self.hidden,
            self.vocabulary,
            true,
            &self.target,
            &self.readout,
        );
        let submission = self
            .program
            .submit(launch)
            .unwrap_or_else(|(e, _)| panic!("{e}"));
        let (core, output) = submission.finish().unwrap().into_parts();
        let record = observe(&output, &[&core]);
        let (_, mut advances, _, _) = core.into_parts();
        let TentativeAdvance::Accepted(advance) = advances.pop().unwrap() else {
            panic!("nonordinary advance")
        };
        let source = advance
            .commit(tokens.len())
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let magnitude_state::OwnedAdvanceResolution::Committed(source) = source else {
            panic!("no acceptance")
        };
        assert_eq!(source.position(), record.position);
        (source, record)
    }
}
fn launch(
    store: &StoreBindings,
    advance: OwnedStateAdvance,
    tokens: &[i32],
    hidden: usize,
    vocabulary: usize,
    final_stage: bool,
    target: &NativeGraphPool,
    readout: &NativeGraphPool,
) -> ValidatedTargetLaunch {
    let binding = advance.bindings();
    let rows = tokens
        .iter()
        .enumerate()
        .map(|(row, &token)| {
            let position = advance.position() + row;
            let read = final_stage && row + 1 == tokens.len();
            Row {
                token,
                coordinates: [position as i32, position as i32, position as i32, 0],
                histories: store
                    .history_domains()
                    .map(|domain| RowHistory {
                        visible: advance
                            .visible_ranges(
                                domain,
                                store.history_domain_kind(domain).visible_from(position),
                            )
                            .into_iter()
                            .map(|(s, n)| [s as i32, (s + n) as i32])
                            .collect(),
                        fresh_start: 0,
                        bidirectional_end: None,
                        destination: binding.destinations[domain.0][row] as i32,
                    })
                    .collect(),
                demand: if read {
                    Demand::LOGITS | Demand::FEATURES | Demand::SELECT
                } else if final_stage {
                    // Match ordinary prefill: every requested feature row is
                    // exported, while logits/selection belong to the final row.
                    Demand::FEATURES
                } else {
                    Demand::NONE
                },
                select: read.then_some(Select {
                    draw: Draw {
                        kind: DrawKind::Greedy,
                        seed: 0,
                        position: position as u64,
                        domain: 0,
                    },
                    mask: None,
                    shaping: Shaping {
                        temperature: 0.,
                        ..Shaping::default()
                    },
                    history: vec![],
                }),
            }
        })
        .collect();
    let batch = ValidatedTargetBatch::from_slots(
        &[Slot {
            rows,
            bank: binding.previous_bank as i32,
            previous_tape: binding.previous_tape as i32,
            following_bank: binding.following_bank as i32,
            stop: tokens.len() as i32,
        }],
        vocabulary,
        ClassLimits {
            rows: 2,
            segments: 1,
        },
    )
    .unwrap();
    let inputs = TargetLaunchInputs::new(
        batch,
        TargetTokens::Host,
        vec![TentativeAdvance::Accepted(advance)],
        vec![None],
        vec![vec![]],
        target.acquire_workspace().unwrap(),
        [
            target.acquire_output().unwrap(),
            target.acquire_output().unwrap(),
        ],
        readout.acquire_workspace().unwrap(),
        readout.acquire_output().unwrap(),
    );
    ValidatedTargetLaunch::new(inputs, store, target.domain(), hidden)
        .unwrap_or_else(|(_, e)| panic!("{e}"))
}
#[derive(Debug, PartialEq, Eq)]
struct Record {
    position: usize,
    features: Vec<u8>,
    logits: Vec<u8>,
    selected: i32,
    history: Vec<(String, Vec<u8>)>,
    recurrent: Vec<Vec<u8>>,
}
fn observe(output: &TargetOutput, cores: &[&crate::TargetLaunchCore]) -> Record {
    observe_advances(
        output,
        &cores
            .iter()
            .map(|core| &core.advances()[0])
            .collect::<Vec<_>>(),
    )
}
fn observe_advances(output: &TargetOutput, advances: &[&TentativeAdvance]) -> Record {
    let readout = output.readout.as_ref().unwrap();
    let selection = readout
        .selected
        .as_ref()
        .unwrap()
        .tensor()
        .read_to_host()
        .unwrap();
    assert_eq!(i32::from_le_bytes(selection[4..8].try_into().unwrap()), 0);
    let owned = advances
        .iter()
        .map(|advance| match advance {
            TentativeAdvance::Accepted(advance) => advance,
            _ => panic!("qualification requires ordinary accepted advances"),
        })
        .collect::<Vec<_>>();
    let (history, recurrent) = observe_state(&owned);
    Record {
        position: advances[0].position() + advances[0].rows(),
        features: readout.features.tensor().read_to_host().unwrap(),
        logits: readout
            .logits
            .as_ref()
            .unwrap()
            .tensor()
            .read_to_host()
            .unwrap(),
        selected: i32::from_le_bytes(selection[..4].try_into().unwrap()),
        history,
        recurrent,
    }
}
fn observe_state(advances: &[&OwnedStateAdvance]) -> (Vec<(String, Vec<u8>)>, Vec<Vec<u8>>) {
    let mut history = Vec::new();
    let mut recurrent = Vec::new();
    for &advance in advances {
        let binding = advance.bindings();
        for plane in binding.history {
            let mut bytes = Vec::new();
            for (start, count) in advance.history_ranges(plane.domain) {
                bytes.extend(
                    plane
                        .buffer
                        .slice_leading(start as u64, (start + count) as u64)
                        .unwrap()
                        .read_to_host()
                        .unwrap(),
                );
            }
            for &row in &binding.destinations[plane.domain.0] {
                bytes.extend(
                    plane
                        .buffer
                        .slice_leading(row as u64, row as u64 + 1)
                        .unwrap()
                        .read_to_host()
                        .unwrap(),
                );
            }
            history.push((
                format!("{:?}/{:?}/{:?}", plane.layer, plane.vector, plane.name),
                bytes,
            ));
        }
        for arena in binding.recurrent {
            recurrent.push(
                arena
                    .slice_leading(
                        binding.following_bank as u64,
                        binding.following_bank as u64 + 1,
                    )
                    .unwrap()
                    .read_to_host()
                    .unwrap(),
            );
        }
    }
    history.sort_by(|a, b| a.0.cmp(&b.0));
    (history, recurrent)
}
fn numeric(label: &str, reference: &[u8], actual: &[u8]) {
    assert_eq!(reference.len(), actual.len());
    let mut absolute = 0f64;
    let mut relative = 0f64;
    let mut nonfinite = 0usize;
    for (a, b) in reference.chunks_exact(4).zip(actual.chunks_exact(4)) {
        let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
        let b = f32::from_le_bytes(b.try_into().unwrap()) as f64;
        nonfinite += usize::from(!a.is_finite() || !b.is_finite());
        absolute = absolute.max((a - b).abs());
        relative = relative.max((a - b).abs() / a.abs().max(1e-6));
    }
    let first = reference.iter().zip(actual).position(|(a, b)| a != b);
    eprintln!("{label} max_abs={absolute} max_rel={relative} relative_floor=1e-6 nonfinite={nonfinite} first_mismatch={first:?}");
    assert_eq!(nonfinite, 0);
    assert!(first.is_none(), "bit-exact comparison failed");
}
fn compare(reference: &Record, actual: &Record) {
    numeric("full_logits", &reference.logits, &actual.logits);
    assert_eq!(
        (reference.position, reference.selected),
        (actual.position, actual.selected)
    );
    assert_eq!(reference.features, actual.features, "normalized features");
    assert_eq!(reference.history, actual.history, "global logical KV");
    assert_eq!(
        reference.recurrent, actual.recurrent,
        "logical recurrent state"
    );
}
fn inputs(
    case: &Case,
    mut step: impl FnMut(&[i32]) -> Record,
    stops: &[i32],
    teacher: bool,
) -> Vec<Record> {
    let mut records = Vec::new();
    for chunk in case.tokens.chunks(2) {
        records.push(step(chunk));
    }
    let mut generated = Vec::new();
    if teacher {
        for &token in &case.teacher_tokens {
            records.push(step(&[token]));
        }
    } else {
        for _ in 0..8 {
            let token = records.last().unwrap().selected;
            generated.push(token);
            if stops.contains(&token) {
                break;
            }
            records.push(step(&[token]));
        }
        assert_eq!(generated.len(), 8, "EOS may not be bypassed");
        eprintln!("generated={generated:?}");
    }
    records
}
fn domain_step(
    domain: &mut crate::domain::ExecutorDomain<PipelineNativeFamily>,
    bindings: crate::domain::StateBindings<PipelineNativeFamily>,
    request: crate::RequestId,
    position: usize,
    kind: crate::WorkKind,
    tokens: &[i32],
) -> (Record, crate::domain::StateBindings<PipelineNativeFamily>) {
    let operation = domain_operation(request, position, kind, tokens);
    let (pending, bindings) = domain_pending(domain, bindings, operation);
    let crate::Outcome::Forward { rows } = pending.outcome() else {
        panic!("forward outcome")
    };
    let row = rows.last().unwrap();
    let selected = row.selected.unwrap();
    assert_eq!(selected.status, 0);
    let features = row
        .features
        .as_ref()
        .unwrap()
        .allocation()
        .tensor()
        .unwrap()
        .read_to_host()
        .unwrap();
    let logits = row
        .logits
        .as_ref()
        .unwrap()
        .read_to_host()
        .unwrap()
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let (history, recurrent) = observe_state(&pending.pipeline_advances());
    let record = Record {
        position: position + tokens.len(),
        features,
        logits,
        selected: selected.token.0 as i32,
        history,
        recurrent,
    };
    domain
        .reconcile(
            pending,
            crate::domain::PhysicalDecision {
                accepted_rows: tokens.len(),
            },
        )
        .unwrap();
    assert_eq!(
        domain.pipeline_positions(request),
        [Some(position + tokens.len()); 2]
    );
    (record, bindings)
}
fn domain_operation(
    request: crate::RequestId,
    position: usize,
    kind: crate::WorkKind,
    tokens: &[i32],
) -> crate::Operation {
    crate::Operation::Forward {
        request,
        kind,
        tokens: tokens.iter().map(|t| crate::TokenId(*t as u32)).collect(),
        position,
        conditioning: None,
        demand: Demand::LOGITS | Demand::FEATURES | Demand::SELECT,
        select: vec![crate::SelectSpec {
            sampling: crate::Sampling::Greedy,
            seed: 0,
            position: position + tokens.len() - 1,
            domain: 0,
            mask: None,
            shaping: crate::Shaping {
                temperature: 0.,
                ..crate::Shaping::default()
            },
            history: None,
        }],
        committed: tokens.len(),
        prime: None,
    }
}
fn domain_pending(
    domain: &mut crate::domain::ExecutorDomain<PipelineNativeFamily>,
    mut bindings: crate::domain::StateBindings<PipelineNativeFamily>,
    operation: crate::Operation,
) -> (
    crate::PendingOperationOutcome,
    crate::domain::StateBindings<PipelineNativeFamily>,
) {
    let request = operation.request();
    let position = match &operation {
        crate::Operation::Forward { position, .. } => *position,
        _ => unreachable!(),
    };
    assert_eq!(domain.pipeline_positions(request), [Some(position); 2]);
    let reservation = domain.reserve(&mut bindings, &[operation.clone()]).unwrap();
    let crate::ReservedResources::Target(reservation) = reservation.into_resources() else {
        panic!("target reservation")
    };
    assert_eq!(domain.pipeline_positions(request), [None; 2]);
    let flight = domain
        .submit_target(bindings, &[operation], reservation)
        .unwrap_or_else(|e| panic!("normal paired submit: {}", e.error()));
    let (mut pending, bindings) = domain.finish_target(flight).unwrap();
    assert_eq!(pending.len(), 1);
    let pending = pending.pop().unwrap();
    assert_eq!(domain.pipeline_positions(request), [None; 2]);
    (pending, bindings)
}
#[test]
#[ignore = "requires complete cached model, tokenizer fixture, exactly two CUDA GPUs and NVRTC"]
fn complete_model_pipeline_matches_ordinary_control() {
    let package =
        Package::open_without_projector(std::env::var("MAGNITUDE_PIPELINE_MODEL").unwrap())
            .unwrap();
    let definition = Rc::new(magnitude_family_qwen35::inspect_package(&package).unwrap());
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(std::env::var("MAGNITUDE_PIPELINE_TOKEN_FIXTURE").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.vocabulary, definition.decoder.vocabulary as usize);
    let ordinals = std::env::var("MAGNITUDE_PIPELINE_CUDA_ORDINALS")
        .unwrap()
        .split(',')
        .map(|i| i.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ordinals.len(), 2);
    let catalog = DeviceCatalog::discover().unwrap();
    let cuda = catalog
        .topology()
        .devices()
        .iter()
        .filter(|d| d.backend == BackendName::Cuda)
        .map(|d| d.selector)
        .collect::<Vec<_>>();
    let devices = [cuda[ordinals[0]], cuda[ordinals[1]]];
    assert_ne!(devices[0], devices[1]);
    let cache = KernelCache::open(
        std::env::var("MAGNITUDE_PIPELINE_CACHE").unwrap().into(),
        crate::DEFAULT_KERNEL_CACHE_BYTES,
    )
    .unwrap();
    let cuts = std::env::var("MAGNITUDE_PIPELINE_CUTS")
        .unwrap()
        .split(',')
        .map(|i| i.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    eprintln!(
        "complete_model blocks={} hidden={} dtype={:?} devices={devices:?}",
        definition.decoder.blocks.len(),
        definition.decoder.hidden,
        definition.decoder.activation_dtype
    );
    let mut controls = Vec::new();
    {
        let mut control = Control::new(&package, &definition, devices[0], &cache);
        eprintln!(
            "ordinary_resident_device_bytes={}",
            control.device.memory_usage().charged
        );
        for case in &fixture.cases {
            eprintln!("prompt={:?} ids={:?}", case.text, case.tokens);
            for teacher in [true, false] {
                let mut source = Some(control.store.create().unwrap());
                let a = inputs(
                    case,
                    |tokens| {
                        let (next, r) = control.step(source.take().unwrap(), tokens);
                        source = Some(next);
                        r
                    },
                    &fixture.stop_tokens,
                    teacher,
                );
                drop(source);
                let mut source = Some(control.store.create().unwrap());
                let b = inputs(
                    case,
                    |tokens| {
                        let (next, r) = control.step(source.take().unwrap(), tokens);
                        source = Some(next);
                        r
                    },
                    &fixture.stop_tokens,
                    teacher,
                );
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(&b) {
                    compare(a, b);
                }
                controls.push(a);
            }
        }
    }
    for cut in cuts {
        let placement = crate::placement::ModelPlacement::pipeline(
            devices
                .into_iter()
                .zip([0..cut, cut..definition.decoder.blocks.len()]),
        );
        let (model, assigned_devices) =
            PipelineModel::from_placement(definition.clone(), &placement).unwrap();
        TwoStageCudaPipeline::qualify(&model, &assigned_devices).unwrap();
        let mut stages = Vec::new();
        for (assignment, selector) in model.stages().zip(assigned_devices) {
            let (device, heap, draft, programs, capacity) =
                prepare(&package, &definition, selector, &cache);
            let state = ResourcePlanner::stage_state_plan(
                &definition,
                draft.load(),
                KvCodec::Dense,
                limits(),
                capacity,
                assignment.view().global_range(),
            )
            .unwrap();
            stages.push(
                StageResources::prepare(
                    assignment,
                    Rc::new(programs),
                    draft,
                    heap,
                    state,
                    ResourceDomainId::new(format!("pipeline-{selector}")).unwrap(),
                    &package,
                )
                .unwrap(),
            );
            assert_eq!(
                device.memory_usage().charged,
                stages
                    .last()
                    .unwrap()
                    .allocation
                    .execution
                    .resources()
                    .steady_committed_bytes(),
                "local plan must classify every ready physical allocation"
            );
            eprintln!(
                "pipeline_stage resident_weights={} local_plan={} device_charged={}",
                stages
                    .last()
                    .unwrap()
                    .executable
                    .residency
                    .resident_bytes()
                    .unwrap(),
                stages
                    .last()
                    .unwrap()
                    .allocation
                    .execution
                    .resources()
                    .steady_committed_bytes(),
                device.memory_usage().charged
            );
        }
        let (pipeline, allocations) =
            TwoStageCudaPipeline::new(stages.try_into().unwrap_or_else(|_| panic!("two stages")))
                .unwrap();
        let pipeline = PipelineNativeFamily::new(pipeline);
        assert_eq!(pipeline.executor().stages[0].activation_transfer_bytes(), 0);
        assert_eq!(
            pipeline.activation_transfer_bytes(),
            allocations[1]
                .execution
                .resources()
                .bytes()
                .activation_transfer
        );
        let devices = pipeline
            .executor()
            .stages
            .each_ref()
            .map(|s| s.device.clone());
        let before = devices.each_ref().map(|d| d.memory_usage().charged);
        let (mut domain, mut bindings) = pipeline.into_domain(allocations).unwrap();
        assert_eq!(devices.each_ref().map(|d| d.memory_usage().charged), before);
        let mut reference = controls.iter();
        for case in &fixture.cases {
            for teacher in [true, false] {
                let request = crate::RequestId(800);
                domain
                    .install_input(
                        request,
                        magnitude_family_contracts::PreparedModelInput::continuation_only(),
                    )
                    .unwrap();
                domain.open_state(&mut bindings, request, None).unwrap();
                let mut position = 0;
                let mut right = Some(bindings);
                let actual = inputs(
                    case,
                    |tokens| {
                        let (record, returned) = domain_step(
                            &mut domain,
                            right.take().unwrap(),
                            request,
                            position,
                            if position < case.tokens.len() {
                                crate::WorkKind::Prefill
                            } else {
                                crate::WorkKind::Decode
                            },
                            tokens,
                        );
                        position += tokens.len();
                        right = Some(returned);
                        record
                    },
                    &fixture.stop_tokens,
                    teacher,
                );
                bindings = right.take().unwrap();
                let expected = reference.next().unwrap();
                assert_eq!(actual.len(), expected.len());
                for (a, b) in expected.iter().zip(&actual) {
                    compare(a, b);
                }
                domain.close(request).unwrap();
            }
        }
        // Refusal before physical submission returns both original sources.
        let request = crate::RequestId(899);
        domain
            .install_input(
                request,
                magnitude_family_contracts::PreparedModelInput::continuation_only(),
            )
            .unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let mut invalid = domain_operation(
            request,
            0,
            crate::WorkKind::Prefill,
            &fixture.cases[0].tokens[..2],
        );
        if let crate::Operation::Forward { select, .. } = &mut invalid {
            select[0].mask = Some(std::sync::Arc::from([0u32]));
        }
        let reservation = domain.reserve(&mut bindings, &[invalid.clone()]).unwrap();
        let crate::ReservedResources::Target(reservation) = reservation.into_resources() else {
            unreachable!()
        };
        bindings = match domain.submit_target(bindings, &[invalid.clone()], reservation) {
            Err(crate::domain::SubmitFailure::Refused(_, bindings)) => bindings,
            _ => panic!("invalid suffix mask must refuse before physical work"),
        };
        assert_eq!(domain.pipeline_positions(request), [Some(0); 2]);
        // A reservation is tied to its original request and extent. Refuse
        // altered submission metadata and restore the original pair, not the
        // request identity supplied by a later caller.
        for changed in [
            vec![domain_operation(
                crate::RequestId(898),
                0,
                crate::WorkKind::Prefill,
                &fixture.cases[0].tokens[..2],
            )],
            vec![domain_operation(
                request,
                0,
                crate::WorkKind::Prefill,
                &fixture.cases[0].tokens[..1],
            )],
            vec![invalid.clone(), invalid.clone()],
        ] {
            let original = domain_operation(
                request,
                0,
                crate::WorkKind::Prefill,
                &fixture.cases[0].tokens[..2],
            );
            let reservation = domain.reserve(&mut bindings, &[original]).unwrap();
            let crate::ReservedResources::Target(reservation) = reservation.into_resources() else {
                unreachable!()
            };
            bindings = match domain.submit_target(bindings, &changed, reservation) {
                Err(crate::domain::SubmitFailure::Refused(_, bindings)) => bindings,
                _ => panic!("altered paired submission must refuse before physical work"),
            };
            assert_eq!(domain.pipeline_positions(request), [Some(0); 2]);
            assert_eq!(domain.pipeline_positions(crate::RequestId(898)), [None; 2]);
        }
        // Completed cancellation aborts both without claiming physical rollback.
        let operation = domain_operation(
            request,
            0,
            crate::WorkKind::Prefill,
            &fixture.cases[0].tokens[..2],
        );
        let (pending, returned) = domain_pending(&mut domain, bindings, operation);
        bindings = returned;
        domain.abort(pending).unwrap();
        assert_eq!(domain.pipeline_positions(request), [Some(0); 2]);
        // Ordinary prompt chunks can carry no readout demand. Both stages still
        // finish and jointly accept rows; no successful token is fabricated.
        let mut operation = domain_operation(
            request,
            0,
            crate::WorkKind::Prefill,
            &fixture.cases[0].tokens[..2],
        );
        if let crate::Operation::Forward { demand, select, .. } = &mut operation {
            *demand = Demand::NONE;
            select.clear();
        }
        let (pending, returned) = domain_pending(&mut domain, bindings, operation);
        bindings = returned;
        let crate::Outcome::Forward { rows } = pending.outcome() else {
            unreachable!()
        };
        assert!(rows
            .iter()
            .all(|row| row.selected.is_none() && row.features.is_none() && row.logits.is_none()));
        domain
            .reconcile(
                pending,
                crate::domain::PhysicalDecision { accepted_rows: 2 },
            )
            .unwrap();
        assert_eq!(domain.pipeline_positions(request), [Some(2); 2]);
        let (next, returned) = domain_step(
            &mut domain,
            bindings,
            request,
            2,
            crate::WorkKind::Prefill,
            &fixture.cases[0].tokens[2..4],
        );
        bindings = returned;
        compare(&controls[0][1], &next);
        domain.close(request).unwrap();
        eprintln!("normal_domain_prephysical_suffix_refusal=true reserved_submission_refusal=true completed_abort=true no_readout_prefill=true joint_publication=true");
        // Valid state growth changes charge, but every allocation stays classified.
        let prefix = domain.pipeline_prefix_memory_charge().unwrap().unwrap();
        let suffix = domain.reconcile_memory_charge(&[]).unwrap();
        assert!(prefix.complete(), "{prefix:?}");
        assert!(suffix.complete(), "{suffix:?}");
        assert_eq!(
            [prefix.charged, suffix.charged],
            devices.each_ref().map(|d| d.memory_usage().charged)
        );
        assert_eq!(prefix.activation_transfer, 0);
        assert_eq!(suffix.activation_transfer, 20480);
        eprintln!("normal_domain_adoption prefix={prefix:?} suffix={suffix:?} serving=false");
        domain.refresh_memory().unwrap();
        let first = crate::RequestId(901);
        let second = crate::RequestId(902);
        domain
            .install_input(
                first,
                magnitude_family_contracts::PreparedModelInput::continuation_only(),
            )
            .unwrap();
        assert!(domain
            .open_state(&mut bindings, first, None)
            .unwrap()
            .is_empty());
        assert!(domain
            .install_input(
                second,
                magnitude_family_contracts::PreparedModelInput::continuation_only()
            )
            .is_err());
        assert!(domain.resume_state(first).is_err());
        // Reclamation checks both idle sources. Releasing resident state must
        // drop both, while leaving installed input until ordinary close.
        domain.reclaimable(&[first]).unwrap();
        domain.release_state(&[first]).unwrap();
        domain.close(first).unwrap();
        domain
            .install_input(
                second,
                magnitude_family_contracts::PreparedModelInput::continuation_only(),
            )
            .unwrap();
        assert!(domain
            .open_state(&mut bindings, second, None)
            .unwrap()
            .is_empty());
        domain.reclaimable(&[second]).unwrap();
        domain.close(second).unwrap();
        domain.reclaim_idle(&mut bindings).unwrap();
        domain.refresh_memory().unwrap();
        assert!(domain
            .pipeline_prefix_memory_charge()
            .unwrap()
            .unwrap()
            .complete());
        assert!(domain.reconcile_memory_charge(&[]).unwrap().complete());
        eprintln!("normal_domain_fresh_requests=2 concurrency_refused=true resume_refused=true paired_release_close=true normal_domain_generation=true owner_worker_serving=false");
        drop((domain, bindings));
        eprintln!("PASS complete two-CUDA cut={cut} bit_exact_features_logits_KV_recurrent_positions_feedback=true program_family=true normal_domain=true serving=false");
    }
}
