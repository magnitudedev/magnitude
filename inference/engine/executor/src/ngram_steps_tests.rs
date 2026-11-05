//! A decoder with an n-gram input steps on the device with real
//! submissions: its weights imported from a written GGUF, its graphs
//! prepared, sealed and bound as a load binds them, and target steps of the
//! host's tokens and of the device's selection run through its n-gram block.
//! A chain of its prompt chunks computes exactly what the chunks compute one
//! step at a time.

use crate::planning::fixtures::{
    expert_stacks, qwen4exp_fixture_definition, qwen4exp_fixture_definition_of, relaid_manifest,
    stored_table_manifest, streamed_experts_manifest,
};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain,
    Operation, Outcome, PhysicalDecision, PlannedMethod, ReservedResources, ResidencyStore,
    ResourceAllocator, ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner,
    RequestId, SelectSpec, Sampling, Shaping, TokenId, WorkKind, domain::TargetGraphReservation,
    platform,
    test_support::available,
};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::KvCodec;
use std::rc::Rc;

/// A GGUF of every tensor `manifest` names (its n-gram tables IQ4_NL
/// blocks, a streamed load's expert stacks Q8_0 or Q4_0 blocks, everything else
/// F16), with small deterministic values.
fn write_gguf(manifest: &magnitude_artifacts::PackageManifest, path: &std::path::Path) {
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

/// The value of finite F16 bits `bits` ([`half_bits`]' inverse).
fn half_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let mantissa = f32::from(bits & 0x3ff);
    sign * match i32::from((bits >> 10) & 0x1f) {
        0 => mantissa * 2f32.powi(-24),
        31 => panic!("an F16 of {bits:#06x} is not finite"),
        exponent => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
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
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture("ngram-steps");
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _) =
            load(&catalog, &definition, &package, backend, limits(2, true, 1, 0), Banked::Resident);
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

/// On Metal and on the CPU, a load streaming its experts through a bank
/// (some layers' experts partly held, some held whole as the bank fills):
/// a prompt chunk, then two decode steps of the device's selected tokens,
/// each step's routing record written at every streamed layer the watcher
/// reads (its step completes, or the watcher's invariant fails it), and
/// the selections the resident load makes.
#[test]
fn a_streamed_decoder_steps_twice_through_its_bank() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture_of("streamed-steps", streamed_experts_manifest);
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let selections = |banked| {
            let (mut domain, mut bindings, _) = load(&catalog, &definition, &package, backend, limits(2, false, 1, 0), banked);
            let request = RequestId(1);
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_state(&mut bindings, request, None).unwrap();
            let (outcome, mut returned) =
                run(&mut domain, bindings, forward(request, WorkKind::Prefill, &[5, 1], 0, true), false, backend);
            let mut tokens = vec![selected_token(&outcome, backend)];
            for position in [2, 3] {
                let token = *tokens.last().unwrap();
                let (outcome, next) =
                    run(&mut domain, returned, forward(request, WorkKind::Decode, &[token], position, true), false, backend);
                tokens.push(selected_token(&outcome, backend));
                returned = next;
            }
            drop(returned);
            tokens
        };
        let resident = selections(Banked::Resident);
        let streamed = selections(Banked::Streamed { numerator: 1, denominator: 2 });
        assert_eq!(streamed, resident, "{backend:?}: a streamed load selects as the resident one");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On every backend experts stream on, a load holding every expert in its
/// bank (a whole bank, as an uncapped load holds them, imported at load
/// into the bank's records however its storage is reserved) over a decoder
/// whose first two attention blocks share one sealed graph, their streamed
/// layers at different homes of one group
/// (`streamed_layers_at_different_homes_share_a_graph`): each layer reads
/// its own experts, through the resident load's arithmetic, so a prompt
/// chunk and its decode steps score bit for bit as the resident load does.
#[test]
fn a_whole_banks_shared_graphs_read_each_layers_own_experts() {
    let definition = qwen4exp_fixture_definition_of(4, 32);
    banked_shared_graphs_score_as_resident("whole-bank-shared", definition, &STREAMING_BACKENDS, Banked::Whole);
}

/// The backends experts stream on.
const STREAMING_BACKENDS: [seismic::BackendName; 3] =
    [seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan];

/// On every backend experts stream on, a bank of about half of every layer's experts
/// (a cap only a partial bank meets: at 128 experts a layer the fixture's
/// floor bank weighs less than its whole one, as a real model's does) over
/// the decoder of `a_whole_banks_shared_graphs_read_each_layers_own_experts`:
/// its layers hit, miss and promote, each at its own map words, so the
/// steps score as the resident load does ([`Scores::assert_as_resident`]).
/// A layer resolving at another's words would serve that layer's experts as
/// hits.
#[test]
fn a_partial_banks_shared_graphs_read_each_layers_own_experts() {
    let definition = qwen4exp_fixture_definition_of(4, 128);
    banked_shared_graphs_score_as_resident("partial-bank-shared", definition, &STREAMING_BACKENDS, Banked::Half);
}

/// On each of `backends` this host has, `definition` (its attention blocks
/// sharing a graph) loaded resident and `banked` score [`scores`]'s steps
/// alike ([`Scores::assert_as_resident`]), each step's last row's logits and
/// selection.
fn banked_shared_graphs_score_as_resident(
    test: &str,
    definition: ModelDefinition,
    backends: &[seismic::BackendName],
    banked: Banked,
) {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture_with(test, definition, streamed_experts_manifest);
    for &backend in backends.iter().filter(|&&backend| available(&catalog, backend)) {
        scores(&catalog, &definition, &package, backend, banked, Filled::Unwaited)
            .assert_as_resident(backend, "the banked load's logits and selection");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On CUDA, a partial bank of the fixture of 128 experts a layer beside a
/// host tier the plan sizes at load to every expert: the tier fills from
/// the bank's open on, so once it keeps every expert (nothing left only
/// on disk) the steps read nothing from the file, every miss the tier's,
/// and they score as the resident load does ([`Scores::assert_as_resident`]).
#[test]
fn a_partial_banks_tier_fills_at_load_and_serves_every_miss() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let backend = seismic::BackendName::Cuda;
    // Only a host with CUDA runs it.
    if !available(&catalog, backend) {
        return;
    }
    let definition = qwen4exp_fixture_definition_of(3, 128);
    let (directory, definition, package) = fixture_with("tier-filled", definition, streamed_experts_manifest);
    scores(&catalog, &definition, &package, backend, Banked::Half, Filled::Waited)
        .assert_as_resident(backend, "the filled tier's load scores as the resident one");
    std::fs::remove_dir_all(&directory).ok();
}

/// On every unified device, a partial bank whose misses are priced to land on
/// the device: each is converted past its barrier, ordered after the wait
/// whose answer writes its list on a backend that runs independent launches
/// concurrently too, so its steps score as the resident load does. Each read
/// is held back long enough that a convert run beside the wait finds no list,
/// and its step reads a record nothing converted.
#[test]
fn misses_the_device_lands_score_as_resident() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let definition = qwen4exp_fixture_definition_of(4, 128);
    let (directory, definition, package) = fixture_with("device-landed", definition, streamed_experts_manifest);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        scores(&catalog, &definition, &package, backend, Banked::Half, Filled::DeviceLanded)
            .assert_as_resident(backend, "the misses the device landed score as the resident load's");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Whether [`scores`] steps a banked load at once, once its host tier keeps
/// every expert, then asserting no step read the file, or with its misses
/// priced to land on the device, where the host places.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Filled {
    Unwaited,
    Waited,
    DeviceLanded,
}

/// The decode steps [`scores`] runs: at 2 choices a step of 128 experts,
/// enough that a later layer routes experts an earlier one holds.
const DECODE_STEPS: usize = 32;

/// `operation` with its last row's logits demanded too, to compare.
fn scored(operation: Operation) -> Operation {
    match operation {
        Operation::Forward { request, kind, tokens, position, conditioning, demand, select, committed, prime } => {
            Operation::Forward {
                request,
                kind,
                tokens,
                position,
                conditioning,
                demand: demand | crate::batching::Demand::LOGITS,
                select,
                committed,
                prime,
            }
        }
        other => other,
    }
}

/// The outputs [`scores`] ran: the resident load's steps, the banked
/// load's, every answer the banked load's CPU site gave (its step, its
/// layer and the misses it served there, in step order), each checked as it
/// was given ([`site_answers`]), and whether the banked load has a site
/// that may serve a miss (a partial bank's, a site held at a layer, priced
/// to serve: [`test_measures`]).
struct Scores {
    resident: Vec<StepBits>,
    banked: Vec<StepBits>,
    sited: Vec<(usize, usize, u64)>,
    site: bool,
}

impl Scores {
    /// Every step scores its last row; the banked load's steps before its
    /// CPU site first served a miss score bit for bit as the resident
    /// load's, and every later one's logits are finite; a load with a site
    /// served a miss on it. A miss the site computes runs the CPU's
    /// arithmetic (q8 activations), so neither that step nor any after it, whose state it wrote,
    /// can score bit for bit as the resident load does: each of the site's
    /// answers was instead checked against its layer's own experts within
    /// the bound that arithmetic admits ([`site_answers`]).
    fn assert_as_resident(&self, backend: seismic::BackendName, what: &str) {
        let unsited = self.sited.first().map_or(self.banked.len(), |&(step, ..)| step);
        for (step, (resident, banked)) in self.resident.iter().zip(&self.banked).enumerate() {
            for (load, bits) in [("resident", resident), ("banked", banked)] {
                assert!(bits.last().is_some_and(|(_, logits, _)| logits.is_some()), "{backend:?} step {step}: the {load} load scores its last row");
            }
            if step < unsited {
                assert_eq!(banked, resident, "{backend:?} step {step} (the site served none of the first {unsited}): {what}");
            } else {
                let finite = banked.last().and_then(|(_, logits, _)| logits.as_ref()).is_some_and(|logits| logits.iter().all(|&bits| f32::from_bits(bits).is_finite()));
                assert!(finite, "{backend:?} step {step}, the CPU site serving misses from step {unsited}: the banked load's logits are finite");
            }
        }
        let misses = self.sited.iter().map(|&(.., misses)| misses).sum::<u64>();
        assert!(!self.site || misses > 0, "{backend:?}: the banked load's CPU site served no miss, so its path never ran");
        if misses > 0 {
            let layers = self.sited.iter().map(|&(_, layer, _)| layer).collect::<std::collections::BTreeSet<_>>();
            println!(
                "{backend:?}: bit for bit through step {unsited}; the CPU site served {misses} misses from step {unsited} at layers {layers:?}, \
                 {} answers within their layers' own experts' q8 bound",
                self.sited.len()
            );
        }
    }
}

/// One routed layer's stored expert stack `[E, rows, columns]`: its Q8_0
/// bytes as [`write_gguf`] writes a streamed load's experts, each expert
/// decoded only as a check reads it.
struct Stack {
    bytes: Vec<u8>,
    shape: [u64; 3],
}

impl Stack {
    fn columns(&self) -> usize {
        self.shape[2] as usize
    }

    /// Expert `expert`'s `[rows, columns]` values, exactly: each Q8_0
    /// block's F16 scale times each of its 32 codes. (The registry's
    /// general decode, `import_transforms::dequantize`, takes seconds an
    /// expert in a debug build, which the site's every answer would pay.)
    fn expert(&self, expert: usize) -> Vec<f32> {
        let per = self.bytes.len() / self.shape[0] as usize;
        self.bytes[expert * per..(expert + 1) * per]
            .chunks_exact(34)
            .flat_map(|block| {
                let scale = half_value(u16::from_le_bytes([block[0], block[1]]));
                block[2..].iter().map(move |&code| scale * f32::from(code as i8))
            })
            .collect()
    }
}

/// Each routed layer's gate, up and down stacks as `package` stores them,
/// in decoder order (the streamed layers' order: the fixture streams every
/// routed layer).
fn stored_experts(package: &Package, definition: &ModelDefinition) -> Vec<[Stack; 3]> {
    expert_stacks(definition)
        .into_iter()
        .map(|names| {
            names.map(|name| {
                let tensor = package.target().tensor(&name).unwrap();
                assert_eq!(tensor.encoding, Encoding::Q8_0, "{name}: the fixture stores a streamed load's experts Q8_0");
                Stack { bytes: tensor.read().unwrap(), shape: tensor.shape.clone().try_into().unwrap() }
            })
        })
        .collect()
}

/// The CPU site's answers of `domain`'s last completed step (step `step`
/// of the test's): each layer's, where its site served a miss, checked
/// against the reference of the layer's own experts' stored values within
/// the bound the site's arithmetic admits (`experts::site_reference`, its
/// one owner); each such layer and the misses its site served.
fn site_answers(
    domain: &ExecutorDomain,
    stored: &[[Stack; 3]],
    function: i32,
    backend: seismic::BackendName,
    step: usize,
) -> Vec<(usize, usize, u64)> {
    let Some((sequence, cost)) = domain.expert_costs().and_then(|costs| costs.completed) else {
        return Vec::new();
    };
    let mut answers = Vec::new();
    for (layer, experts) in cost.experts.iter().enumerate().filter(|(_, experts)| experts.sited > 0) {
        let answered = domain
            .site_answered(layer, sequence)
            .unwrap_or_else(|| panic!("{backend:?} step {step}: layer {layer}'s site served {} misses and holds no answer", experts.sited));
        let [gate, up, down] = &stored[layer];
        let hidden = up.columns();
        for (row, (x, output)) in answered.rows.chunks_exact(hidden).zip(answered.output.chunks_exact(hidden)).enumerate() {
            let named = answered
                .entries
                .iter()
                .filter(|entry| entry.row == row)
                .map(|entry| (entry.weight, entry.scales, [gate, up, down].map(|stack| stack.expert(entry.expert))))
                .collect::<Vec<_>>();
            let named = named
                .iter()
                .map(|(weight, scales, [gate, up, down])| {
                    (*weight, crate::experts::ExpertValues { gate: Some(gate.as_slice()), up, down, scales: *scales })
                })
                .collect::<Vec<_>>();
            let (expected, bound) = crate::experts::site_reference(function, x, &named);
            for (channel, ((got, want), bound)) in output.iter().zip(&expected).zip(&bound).enumerate() {
                assert!(
                    (got - want).abs() <= *bound,
                    "{backend:?} step {step} layer {layer} row {row} channel {channel}: the site's {got} against its layer's own experts' {want} ±{bound}"
                );
            }
        }
        answers.push((step, layer, experts.sited));
    }
    answers
}

/// On `backend`, the outputs of the resident load and then of `banked`:
/// a prompt chunk and [`DECODE_STEPS`] decode steps of distinct tokens
/// (37 is prime to the vocabulary's 256), so each layer routes many
/// experts, each step's last row scored, and the banked load's CPU site's
/// answers checked ([`site_answers`]).
fn scores(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Package,
    backend: seismic::BackendName,
    banked: Banked,
    filled: Filled,
) -> Scores {
    let stored = stored_experts(package, definition);
    let function = definition
        .decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            magnitude_family_contracts::Operator::RoutedFfn(routed) => match &routed.expert_up {
                magnitude_family_contracts::FeedForwardUp::Gated { activation, .. } => Some(*activation),
                _ => None,
            },
            _ => None,
        })
        .map(crate::operators::dense_ffn::activation_code)
        .expect("the fixture's routed layers are gated");
    let mut scores = Vec::new();
    for banked in [Banked::Resident, banked] {
        let (mut domain, mut bindings, _) = load(catalog, definition, package, backend, limits(2, false, 1, 1), banked);
        let plan = domain.expert_bank_plan();
        let whole = plan.map(crate::ExpertBankPlan::whole);
        match banked {
            Banked::Resident => {}
            Banked::Whole => assert_eq!(whole, Some(true), "{backend:?}: the uncapped load holds every expert in its bank"),
            Banked::Streamed { .. } | Banked::Half => {
                assert_eq!(whole, Some(false), "{backend:?}: the capped load banks part of its experts")
            }
        }
        let site = plan.is_some_and(|plan| !plan.whole() && (0..plan.layers.len()).any(|layer| plan.sited(layer)));
        let waited = filled == Filled::Waited && !matches!(banked, Banked::Resident);
        if waited {
            // The tier fills from the bank's open on: nothing stays only on
            // disk once it keeps every expert.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while domain.expert_residency().unwrap().unwrap().disk_bytes > 0 {
                assert!(std::time::Instant::now() < deadline, "{backend:?}: the host tier never kept every expert");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        // A site serves only once its load is priced: until then every miss
        // is served on the GPU's path. A miss lands on the device once its
        // convert is priced under the host's landing.
        let device_landed = filled == Filled::DeviceLanded && !matches!(banked, Banked::Resident);
        if device_landed {
            domain.slow_host_landing();
        }
        if site || device_landed {
            domain
                .measure_experts(test_measures(&domain, site, device_landed), &mut bindings)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        }
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (outcome, mut returned) =
            run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, backend);
        let mut steps = vec![step_bits(&outcome)];
        let mut sited = Vec::new();
        for position in 2..2 + DECODE_STEPS {
            let token = (position * 37 % 256) as u32;
            let (outcome, next) =
                run(&mut domain, returned, scored(forward(request, WorkKind::Decode, &[token], position, true)), false, backend);
            sited.extend(site_answers(&domain, &stored, function, backend, steps.len()));
            steps.push(step_bits(&outcome));
            returned = next;
        }
        drop(returned);
        if filled == Filled::DeviceLanded && domain.expert_bank_plan().is_some_and(|plan| plan.placer() == crate::planning::Placer::Bank) {
            assert!(domain.device_landed() > 0, "{backend:?}: the device landed no miss, so its path never ran");
        }
        if waited {
            let costs = domain.expert_costs().expect("a banked load's costs");
            let (misses, read) = costs.layers.iter().fold((0, 0), |(misses, read), layer| {
                (misses + layer.misses, read + layer.staged + layer.reading + layer.unanticipated)
            });
            assert!(misses > 0, "{backend:?}: a partial bank misses");
            assert_eq!(read, 0, "{backend:?}: every miss is the tier's, none read from the file ({misses} misses)");
        }
        scores.push((steps, sited, site));
    }
    let [(resident, ..), (banked, sited, site)] = <[_; 2]>::try_from(scores).expect("a resident and a banked load");
    Scores { resident, banked, sited, site }
}

/// What a test's banked load is priced on, every other input pending, as a
/// first load's are. A `sited` load's lines are each streamed layer's the
/// reference host's at its own record (`ExpertLines::reference`), at the
/// layer's record bytes, where a one-row miss ends on the CPU (40 µs and a
/// 4 µs handshake) before the link's 86 µs, so the route assigns misses to
/// the site, as a real model's on that host; the fixture's experts are too
/// small for a link at their own bytes to lose to a handshake. A
/// `device_landed` load's bank groups each convert at a line far under the
/// host's landing (`Shared::slow_host_landing`).
fn test_measures(domain: &ExecutorDomain, sited: bool, device_landed: bool) -> crate::ExpertMeasures {
    use crate::assessment::{ExpertLines, HostLines, Priced, RecordLine};
    let pending = || "not measured by the test".to_owned();
    let lines = match sited {
        true => {
            let reference = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
            Priced::Measured(
                domain
                    .expert_bank_lines()
                    .expect("a sited load has a bank")
                    .iter()
                    .map(|bank| ExpertLines { host: HostLines { record_bytes: bank.record_bytes, ..reference.host }, ..reference })
                    .collect(),
            )
        }
        false => Priced::Pending(pending()),
    };
    let convert = match device_landed {
        true => {
            let groups = domain.expert_bank_plan().expect("a device-landed load has a bank").groups.len();
            Priced::Measured(Some(vec![RecordLine::new(1e-9, 1e-9).expect("a line"); groups].into()))
        }
        false => Priced::Pending(pending()),
    };
    crate::ExpertMeasures {
        lines,
        store: None,
        resident: Priced::Pending(pending()),
        pass: Priced::Pending(pending()),
        moves: Priced::Pending(pending()),
        tier_copy: Priced::Pending(pending()),
        convert,
    }
}

/// The token a step's last row selected.
fn selected_token(outcome: &Outcome, backend: seismic::BackendName) -> u32 {
    let Outcome::Forward { rows } = outcome else {
        panic!("{backend:?}: a forward step has a forward outcome");
    };
    let selected = rows
        .last()
        .and_then(|row| row.selected)
        .unwrap_or_else(|| panic!("{backend:?}: the step selects a token"));
    assert_eq!(selected.status, 0, "{backend:?}");
    selected.token.0
}

/// On Metal and CPU, a streamed load's prompt groups at the kl's shape (a
/// 4096-token context, groups of 7 chunks of 512 rows, two prompts in
/// turn, the second's experts held in the host tier the first filled) end,
/// then decode, and select the resident load's tokens.
#[test]
fn a_streamed_load_runs_its_prompt_groups_then_decodes() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture_of("streamed-groups", streamed_experts_manifest);
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let chunks = (SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS;
    let prompts = Prompts { requests: 2, chunks, rows: SPLIT_CHUNK_ROWS };
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let selected = |banked| {
            let (steps, _) = chained_prompts(&catalog, &definition, &package, backend, prompts, chunks, banked);
            steps.iter().filter_map(|(bits, _)| bits.last().and_then(|(_, _, selected)| *selected)).collect::<Vec<_>>()
        };
        let resident = selected(Banked::Resident);
        let streamed = selected(Banked::Streamed { numerator: 1, denominator: 2 });
        assert_eq!(streamed, resident, "{backend:?}: a streamed load's prompt groups select as the resident load's");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On Metal (no CPU site: a banked load runs the resident arithmetic), the
/// 128-expert fixture's prompt groups at the kl's shape (two prompts of 7
/// chunks of 512 rows in turn) and their decode steps, on a whole bank and
/// on a partial one, score bit for bit as the resident load does: every
/// row's features, each step's logits and its selection.
#[test]
fn a_banked_loads_prompt_groups_score_as_resident() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let metal = seismic::BackendName::Metal;
    // Only a host with Metal runs it.
    if !available(&catalog, metal) {
        return;
    }
    let (directory, definition, package) =
        fixture_with("banked-groups", qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let chunks = (SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS;
    let prompts = Prompts { requests: 2, chunks, rows: SPLIT_CHUNK_ROWS };
    let steps = |banked| chained_prompts(&catalog, &definition, &package, metal, prompts, chunks, banked).0;
    let resident = steps(Banked::Resident);
    for (label, banked) in [("whole bank", Banked::Whole), ("partial bank", Banked::Half)] {
        assert_steps_equal(&steps(banked), &resident, prompts, label);
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On Metal, a whole bank of the 128-expert fixture shrunk plateau by
/// plateau between decode steps, then to its floor (as the release ladder
/// shrinks it under pressure; on unified memory the kept experts' bytes
/// move within the one store), decodes on bit for bit as the resident load
/// does: each expert it dropped is a file miss from then on, each one it
/// kept is read where the shrink moved it, from the first step after it.
/// With every stack of one form (one bank group) and with the last routed
/// layer's in another (two groups, each shrinking its own records), and
/// without and with lookahead. After each shrink the miss record counts
/// misses at the slots the bank commits (`Telemetry::due`).
#[test]
fn a_whole_bank_shrunk_between_decode_steps_scores_as_resident() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    // Only a host with Metal runs it.
    if !available(&catalog, seismic::BackendName::Metal) {
        return;
    }
    let manifests: [(&str, fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest, usize); 2] =
        [("whole-bank-shrunk", streamed_experts_manifest, 1), ("two-groups-shrunk", two_group_experts_manifest, 2)];
    for (test, manifest, groups) in manifests {
        shrunk_bank_scores_as_resident(&catalog, test, manifest, groups);
    }
}

/// A whole bank shrunk between decode steps on CUDA, where graphs are
/// recorded as a run first needs one: the decode launch's graphs were
/// recorded as its slots were made (`seismic::NativeGraphFamily::record_decode`),
/// so a shrink with no memory left to allocate, every ledger of the device
/// bound to what it charges, begins and completes, and its step records
/// none.
#[test]
#[ignore = "needs a CUDA device: run with --ignored on one"]
fn a_whole_banks_shrink_step_records_no_graph() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let cuda = seismic::BackendName::Cuda;
    assert!(available(&catalog, cuda), "the test runs on a host with a CUDA device");
    let test = "whole-bank-shrink-records";
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let (mut domain, mut bindings, _) = load(&catalog, &definition, &package, cuda, limits(2, false, 1, 1), Banked::Whole);
    let request = RequestId(1);
    domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
    domain.open_state(&mut bindings, request, None).unwrap();
    let (_, mut bindings) = run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, cuda);
    let mut position = 2;
    let mut decode = |domain: &mut ExecutorDomain, bindings: crate::StateBindings| {
        let token = (position * 37 % 256) as u32;
        let (_, next, _) = step(domain, bindings, scored(forward(request, WorkKind::Decode, &[token], position, true)), cuda);
        position += 1;
        next
    };
    bindings = decode(&mut domain, bindings);
    let whole = domain.expert_bank_slots();
    let device = domain.device().clone();
    let own = device.ledgers().own;
    for (store, view) in &own {
        device.bound(seismic::Ledger::Own(*store), Some(view.charged)).unwrap();
    }
    let step = domain.shrink_expert_bank_by(&mut bindings, u64::MAX / 4, crate::CompletionWake::new(|| {})).unwrap();
    assert!(matches!(step, crate::BankStep::Begun(_)), "a whole bank shrinks: {step:?}");
    while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
    assert!(domain.expert_bank_slots() < whole, "the bank shrank from {whole:?}");
    for (store, view) in &own {
        device.bound(seismic::Ledger::Own(*store), view.bound).unwrap();
    }
    let before = domain.graph_replays().recorded;
    bindings = decode(&mut domain, bindings);
    assert_eq!(domain.graph_replays().recorded, before, "the shrink's step records no graph");
    drop(bindings);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// The last routed layer's expert stacks stored as Q4_0, the rest as
/// [`streamed_experts_manifest`] stores them: two record forms, so two bank
/// groups.
fn two_group_experts_manifest(definition: &ModelDefinition) -> magnitude_artifacts::PackageManifest {
    let last = expert_stacks(definition).pop().expect("the fixture routes");
    relaid_manifest(streamed_experts_manifest(definition), &last, Encoding::Q4_0, 18)
}

/// [`a_whole_bank_shrunk_between_decode_steps_scores_as_resident`] over
/// the 128-expert fixture's experts as `manifest` stores them, in `groups`
/// bank groups.
fn shrunk_bank_scores_as_resident(
    catalog: &seismic::Devices,
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
    groups: usize,
) {
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(3, 128), manifest);
    let metal = seismic::BackendName::Metal;
    // Without lookahead each step's token is the host's; with it, the
    // device's selection, each step claiming the successor queued behind
    // the one before (unless a shrink orphaned it), as served decode runs.
    let decode = |banked: Banked, lookahead: bool| {
        let (mut domain, mut bindings, _) = load(catalog, &definition, &package, metal, limits(2, lookahead, 1, 1), banked);
        if matches!(banked, Banked::Whole) {
            assert_eq!(domain.expert_bank_plan().map(|bank| bank.groups.len()), Some(groups), "{test}: the bank's groups");
        }
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (outcome, mut bindings) = run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, metal);
        let mut steps = vec![step_bits(&outcome)];
        let mut shrinks = 0;
        for position in 2..2 + DECODE_STEPS {
            // From a quarter of the decode, one slot's worth a step, as the
            // release ladder shrinks it, each shrink beside the misses the
            // last ones made land; then to the floor at once.
            let shrinking = 2 + DECODE_STEPS / 4..2 + DECODE_STEPS / 2;
            if matches!(banked, Banked::Whole) && (shrinking.contains(&position) || position == shrinking.end) {
                let deficit = if position == shrinking.end { u64::MAX / 4 } else { 1 };
                let wake = crate::CompletionWake::new(|| {});
                // Each shrink reaches the next physical plateau; at the
                // floor the bank is at its limit.
                // A step still running (a queued successor) defers it to a
                // later boundary.
                match domain.shrink_expert_bank_by(&mut bindings, deficit, wake).unwrap() {
                    crate::BankStep::Begun(_) => {
                        shrinks += 1;
                        while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
                        let priced = domain.expert_costs().map(|costs| costs.slots as u64);
                        assert_eq!(priced, domain.expert_bank_slots(), "{test}: the record counts misses at the bank's slots");
                    }
                    step => assert!(matches!(step, crate::BankStep::AtLimit | crate::BankStep::Deferred), "{step:?}"),
                }
            }
            let token = match lookahead {
                true => steps.last().and_then(|bits| bits.last()).and_then(|(_, _, selected)| *selected).expect("a step selects").0,
                false => (position * 37 % 256) as u32,
            };
            let (outcome, next, _) = step(&mut domain, bindings, scored(forward(request, WorkKind::Decode, &[token], position, true)), metal);
            steps.push(step_bits(&outcome));
            bindings = next;
        }
        (steps, shrinks)
    };
    for lookahead in [false, true] {
        let (resident, _) = decode(Banked::Resident, lookahead);
        let (shrunk, shrinks) = decode(Banked::Whole, lookahead);
        assert!(shrinks >= 2, "{test}, lookahead {lookahead}: the bank shrank in steps ({shrinks})");
        for (step, (shrunk, resident)) in shrunk.iter().zip(&resident).enumerate() {
            assert_eq!(shrunk, resident, "{test}, lookahead {lookahead}, step {step}: the shrunk bank's logits and selection");
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// The fixture decoder written as a GGUF in a fresh directory named for
/// `test`, and its definition naming the written artifact.
fn fixture(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, Package) {
    fixture_of(test, stored_table_manifest)
}

/// [`fixture`] of the tensors `manifest` lays out.
fn fixture_of(
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Package) {
    fixture_with(test, qwen4exp_fixture_definition(), manifest)
}

/// [`fixture_of`] the decoder `definition`.
fn fixture_with(
    test: &str,
    mut definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Package) {
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
    write_gguf(&manifest(&definition), &path);
    let package = Package::open_without_projector(&path).unwrap();
    definition.artifact_identity = package.manifest().identity;
    (directory, Rc::new(definition), package)
}

/// Rows of one prompt chunk, the launch row bound of the chain test: a
/// prompt class, above the decode form's rows (`DECODE_ROWS`), so every
/// routed layer splits into its front and back around the group's pass.
const CHUNK_ROWS: usize = 2 * crate::operators::routed::fused_graph::DECODE_ROWS as usize;

/// Prompt chunk `chunk` of `chunks` of `rows` rows of `request`'s own
/// prompt, each row's features and its last row's logits demanded; the last
/// chunk selects the first token.
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
        demand: if last {
            demand | crate::batching::Demand::SELECT
        } else {
            demand
        },
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
    }
}

/// A step's outputs as bits: per row, its features, its logits and its
/// selection.
type StepBits = Vec<(Option<Vec<u32>>, Option<Vec<u32>>, Option<(u32, u8)>)>;

fn step_bits(outcome: &Outcome) -> StepBits {
    let Outcome::Forward { rows } = outcome else {
        panic!("a forward step has a forward outcome: {outcome:?}");
    };
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    rows.iter()
        .map(|row| {
            (
                row.features
                    .as_ref()
                    .map(|features| bits(features.read_to_host().unwrap())),
                row.logits
                    .as_ref()
                    .map(|logits| bits(logits.read_to_host().unwrap())),
                row.selected.map(|selected| (selected.token.0, selected.status)),
            )
        })
        .collect()
}

/// The prompts of a chain test: `requests` sequences of `chunks` prompt
/// chunks of `rows` rows each.
#[derive(Clone, Copy)]
struct Prompts {
    requests: u64,
    chunks: usize,
    rows: usize,
}

/// On `backend`: `prompts`' sequences open together, as a batch of prompts
/// is scored; each one's prompt in turn, run in chains of up to
/// `chain_capacity` launches while the earlier ones hold their histories;
/// then two decode steps of each request's selected tokens. Returns every
/// step's outputs and its sequence's position after it, and the most rows a
/// group's pass covers (`group_pass_rows`).
fn chained_prompts(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Package,
    backend: seismic::BackendName,
    Prompts { requests, chunks, rows }: Prompts,
    chain_capacity: usize,
    banked: Banked,
) -> (Vec<(StepBits, Option<usize>)>, u64) {
    let (mut domain, mut bindings, pass_rows) = load(
        catalog,
        definition,
        package,
        backend,
        limits(rows, false, chain_capacity, rows),
        banked,
    );
    let requests = (1..=requests).map(RequestId).collect::<Vec<_>>();
    for &request in &requests {
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
    }
    let mut steps = Vec::new();
    let mut selected = Vec::new();
    for &request in &requests {
        for chunk in 0..chunks {
            // A chain opens at every capacity-th chunk, of the chunks
            // planned behind it; its holder claims the rest.
            let opens = chunk % chain_capacity == 0;
            if opens {
                domain.plan_successors(
                    (chunk + 1..chunks.min(chunk + chain_capacity))
                        .map(|next| prompt_chunk(request, next, chunks, rows))
                        .collect(),
                );
            }
            if chain_capacity > 1 {
                assert_eq!(domain.chain_holder().is_some(), !opens);
            }
            let (outcome, returned) = run(
                &mut domain,
                bindings,
                prompt_chunk(request, chunk, chunks, rows),
                !opens,
                backend,
            );
            bindings = returned;
            // A chain holds its holder's state until its last chunk.
            let published = domain.resume_state(request).ok().map(|state| state.position());
            assert_eq!(published.is_none(), domain.chain_holder().is_some());
            steps.push((step_bits(&outcome), published));
        }
        assert!(domain.chain_holder().is_none(), "the chains ended");
        selected.push(steps.last().and_then(|(bits, _)| bits.last()).and_then(|(_, _, selected)| *selected));
    }
    for (&request, token) in requests.iter().zip(selected) {
        let mut token = token.expect("the prompt's last chunk selects the decoded token").0;
        for offset in 0..2 {
            let mut decode = forward(
                request,
                WorkKind::Decode,
                &[token],
                chunks * rows + offset,
                true,
            );
            if let Operation::Forward { demand, .. } = &mut decode {
                *demand |= crate::batching::Demand::LOGITS;
            }
            let (outcome, returned) = run(&mut domain, bindings, decode, false, backend);
            bindings = returned;
            let bits = step_bits(&outcome);
            token = bits
                .last()
                .and_then(|(_, _, selected)| *selected)
                .expect("a decode step selects the next token")
                .0;
            steps.push((bits, Some(domain.resume_state(request).unwrap().position())));
        }
    }
    (steps, pass_rows)
}

/// Two runs of `prompts` in chains of different lengths (`label` names the
/// pair) are equal: in every output byte (each row's features and each
/// chunk's last logits), in each committed position both publish (a chain
/// publishes its last chunk's; one of chunks alone, every chunk's), and in
/// the two decode steps of each request after the prompts, which read every
/// committed history row and recurrent state: their logits bytes and
/// selected tokens.
fn assert_steps_equal(
    one: &[(StepBits, Option<usize>)],
    other: &[(StepBits, Option<usize>)],
    prompts: Prompts,
    label: &str,
) {
    let chunks = prompts.requests as usize * prompts.chunks;
    // Every prompt row reads back its features, and every step's last row
    // its logits.
    assert!(one
        .iter()
        .all(|(bits, _)| bits.last().is_some_and(|(_, logits, _)| logits.is_some())));
    assert!(one[..chunks].iter().all(|(bits, _)| {
        bits.len() == prompts.rows && bits.iter().all(|(features, ..)| features.is_some())
    }));
    assert_eq!(one.len(), other.len());
    for (step, ((one, at), (other, published))) in one.iter().zip(other).enumerate() {
        assert_eq!(one, other, "{label}: step {step} differs");
        if let (Some(at), Some(published)) = (at, published) {
            assert_eq!(at, published, "{label}: step {step}'s position");
        }
    }
}

/// On Metal: chains of 2 and of 4 prompt chunks equal the same chunks
/// submitted one step at a time, for one request and for three whose
/// prompts run in turn while the earlier ones hold their histories
/// (`assert_steps_equal`). Only a host with Metal runs it.
#[test]
fn a_chain_of_prompt_chunks_equals_its_chunks_one_step_at_a_time() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let metal = seismic::BackendName::Metal;
    if !available(&catalog, metal) {
        return;
    }
    let (directory, definition, package) = fixture("chain-steps");
    for (requests, chunks) in [(1, 2), (1, 4), (3, 4)] {
        let prompts = Prompts {
            requests,
            chunks,
            rows: CHUNK_ROWS,
        };
        let (alone, _) = chained_prompts(&catalog, &definition, &package, metal, prompts, 1, Banked::Resident);
        let (chained, _) = chained_prompts(&catalog, &definition, &package, metal, prompts, chunks, Banked::Resident);
        assert_steps_equal(&alone, &chained, prompts, &format!("{requests} requests of {chunks} chunks"));
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Rows of one prompt chunk of the split-pass test: Metal's admitted launch
/// bound, a prompt class, so every routed layer's backs merge into the
/// group's pass.
const SPLIT_CHUNK_ROWS: usize = 512;

/// The split-pass test's context: its prompt is a group of 7 chunks, whose
/// pass on the CPU needs more workspace than the arena the chunks' own
/// graphs need (`group_pass_rows`): it holds 3 chunks, so the group runs
/// passes of 3, 3 and 1, the last over the chain's last rows.
const SPLIT_CONTEXT: u64 = 4096;

/// On the CPU: a group of more chunks than its pass holds (`group_pass_rows`,
/// from the model's shapes) runs as several passes, and equals the same
/// chunks submitted one step at a time and in groups of the most chunks one
/// pass holds (`assert_steps_equal`), for one request and for two whose
/// prompts run in turn while the first holds its history. The CPU's routed
/// kernels compute each row from that row alone (`routed_experts`), so a
/// row's bits do not depend on the rows its pass covers. On Metal this
/// model's launch graphs need an arena that holds a pass of its whole
/// context (at 512 to 16,384 tokens), so no group of it splits.
#[test]
fn a_group_longer_than_its_pass_equals_its_chunks_in_one_pass() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture("split-pass");
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    // The longest prompt the context holds beside its two decode rows.
    let chunks = (SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS;
    let cpu = seismic::BackendName::Cpu;
    for requests in [1, 2] {
        let prompts = Prompts {
            requests,
            chunks,
            rows: SPLIT_CHUNK_ROWS,
        };
        let (split, pass_rows) = chained_prompts(&catalog, &definition, &package, cpu, prompts, chunks, Banked::Resident);
        // A pass of several chunks, fewer than the group's.
        let pass = usize::try_from(pass_rows).unwrap() / SPLIT_CHUNK_ROWS;
        assert!((2..chunks).contains(&pass), "a pass of {pass} of the group's {chunks} chunks");
        let (one_pass, rows) = chained_prompts(&catalog, &definition, &package, cpu, prompts, pass, Banked::Resident);
        assert_eq!(rows, pass_rows, "the pass's rows are the shapes', not the chain's");
        let (alone, _) = chained_prompts(&catalog, &definition, &package, cpu, prompts, 1, Banked::Resident);
        let label = |of: &str| format!("{requests} requests: {of}");
        assert_steps_equal(&one_pass, &split, prompts, &label("split against one pass"));
        assert_steps_equal(&alone, &split, prompts, &label("split against alone"));
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Reserve, submit, finish and accept one target step; with `claims`, the
/// step must claim the queued step.
fn run(
    domain: &mut ExecutorDomain,
    bindings: crate::StateBindings,
    operation: Operation,
    claims: bool,
    backend: seismic::BackendName,
) -> (Outcome, crate::StateBindings) {
    let (outcome, bindings, claimed) = step(domain, bindings, operation, backend);
    assert_eq!(claimed, claims, "{backend:?}: whether the step claims the queued step");
    (outcome, bindings)
}

/// [`run`] whichever way the step reserves: whether it claimed the queued
/// step.
fn step(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operation: Operation,
    backend: seismic::BackendName,
) -> (Outcome, crate::StateBindings, bool) {
    let operations = [operation];
    let rows = operations[0].row_count();
    let reserved = domain
        .reserve(&mut bindings, &operations)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        .into_resources();
    let ReservedResources::Target(reservation) = reserved else {
        panic!("{backend:?}: a forward reserves the target lane");
    };
    let claimed = matches!(reservation, TargetGraphReservation::Claim(_));
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
    let (mut pending, bindings) = domain
        .finish_target(flight)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    let pending = pending.pop().expect("one operation");
    let outcome = pending.outcome().clone();
    assert!(
        !matches!(outcome, Outcome::WeightReadFailed(_)),
        "{backend:?}: {outcome:?}"
    );
    domain
        .reconcile(pending, PhysicalDecision { accepted_rows: rows })
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (outcome, bindings, claimed)
}

/// Launches of `rows` rows, `lookahead` or not, in chains of up to
/// `chain_capacity`, exporting up to `logits` rows of logits.
fn limits(rows: usize, lookahead: bool, chain_capacity: usize, logits: usize) -> ResourceLimits {
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
enum Banked {
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
fn load(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Package,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
) -> (ExecutorDomain, crate::StateBindings, u64) {
    // A hung step reports every owner and every thread's stack on SIGUSR1,
    // as the rig asks a job past its estimate.
    crate::diagnostics::install();
    let reserves = platform::MemoryReserves::standard();
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &reserves,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: an available device is selected: {error}"));
    let device = Rc::new(
        catalog
            .open(catalog.resolve(selected.info.selector()).unwrap())
            .unwrap_or_else(|error| panic!("{backend:?}: the selected device opens: {error}")),
    );
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
    let draft = match banked {
        Banked::Resident => draft,
        Banked::Streamed { numerator, denominator } => {
            let resident = draft.allocation_bytes(definition).unwrap();
            draft
                .with_bank(definition, Some(resident * numerator / denominator), None)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        }
        Banked::Whole => draft.with_bank(definition, None, None).unwrap_or_else(|error| panic!("{backend:?}: {error}")),
        Banked::Half => {
            let whole = draft.clone().with_bank(definition, None, None).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let bank = whole.load().expert_bank().unwrap_or_else(|| panic!("{backend:?}: the uncapped load banks its experts"));
            let cap = whole.load().weight_bytes().unwrap() - bank.slot_bytes().unwrap() * (bank.widest().get() / 2);
            draft.with_bank(definition, Some(cap), None).unwrap_or_else(|error| panic!("{backend:?}: {error}"))
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
        .finalize_readout(definition, programs.readout_choices(), None)
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
    let pass_rows = target_graphs.pass_rows();
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
    let (domain, bindings) = ExecutorDomain::new(
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
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (domain, bindings, pass_rows)
}

/// A streamed load's step whose host starves past every wait's patience
/// while it runs (no beat of its pulse, no answer written: its waits end
/// before their answers) is interrupted, poisoning nothing: aborting it
/// returns the sequence's accepted state, and the step run again from there
/// scores bit for bit as the same step of a sequence of the same history
/// that was never starved. A stall no wait outlasts interrupts nothing.
/// Where the round driver re-runs an interrupted round
/// (`StartedRound::abandon`), this is the physical half it relies on. The
/// starved step is a prompt of a grouped class on a partial bank, whose
/// first round every streamed layer waits on whatever the bank holds; a
/// decode step waits only on an absent expert, which no load guarantees.
#[test]
fn a_starved_step_is_interrupted_and_its_rerun_scores_as_unstarved() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) =
        fixture_with("starved-step", qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let prompt = (0..CHUNK_ROWS as u32).map(|row| (row * 37 + 11) % 256).collect::<Vec<_>>();
    let backends = [seismic::BackendName::Cuda, seismic::BackendName::Metal];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let limits = ResourceLimits { conversations: std::num::NonZeroUsize::new(2).unwrap(), ..limits(CHUNK_ROWS, false, 1, CHUNK_ROWS) };
        let (mut domain, mut bindings, _) = load(&catalog, &definition, &package, backend, limits, Banked::Half);
        let (unstarved, starved) = (RequestId(1), RequestId(2));
        for request in [unstarved, starved] {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_state(&mut bindings, request, None).unwrap();
        }
        let operation = |request| scored(forward(request, WorkKind::Prefill, &prompt, 0, true));
        let (reference, returned) = run(&mut domain, bindings, operation(unstarved), false, backend);
        // Past the slowest wait's patience (under a second).
        let (pending, returned) =
            starved_step(&mut domain, returned, operation(starved), std::time::Duration::from_millis(1200), backend);
        assert!(
            matches!(pending.outcome(), Outcome::Interrupted { resident: true, .. }),
            "{backend:?}: a starved step is interrupted: {:?}",
            pending.outcome()
        );
        domain.abort(pending).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        let (rerun, returned) = run(&mut domain, returned, operation(starved), false, backend);
        assert_eq!(step_bits(&rerun), step_bits(&reference), "{backend:?}: the re-run scores as unstarved");
        // A stall of a tenth of the least patience.
        let operation = scored(forward(starved, WorkKind::Decode, &[148], prompt.len(), true));
        let (pending, returned) = starved_step(&mut domain, returned, operation, std::time::Duration::from_millis(10), backend);
        assert!(matches!(pending.outcome(), Outcome::Forward { .. }), "{backend:?}: a short stall interrupts: {:?}", pending.outcome());
        domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        drop(returned);
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// `operation` submitted alone, its host starved for `starve` from before
/// its submission (`ExecutorDomain::starve_expert_host`), so whatever
/// answers its waits is held however soon the host has it, then finished:
/// its pending outcome, unreconciled.
fn starved_step(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operation: Operation,
    starve: std::time::Duration,
    backend: seismic::BackendName,
) -> (crate::PendingOperationOutcome, crate::StateBindings) {
    let operations = [operation];
    domain.starve_expert_host(starve);
    let reserved = domain
        .reserve(&mut bindings, &operations)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        .into_resources();
    let ReservedResources::Target(reservation) = reserved else {
        panic!("{backend:?}: a forward reserves the target lane");
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
    let (mut pending, bindings) = domain.finish_target(flight).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (pending.pop().expect("one operation"), bindings)
}

/// A prompt chain whose host starves past every wait's patience while its
/// chunks run is interrupted at the chunk that launched it, the chain
/// released with its holder's state (`Outcome::Interrupted` not resident),
/// so the sequence re-prefills, as an evicted one. The chain's later chunks
/// run in its launch (layer-major), so the first chunk's flight is the one
/// their waits fall in. Re-opened and prefilled again, it decodes bit for
/// bit as a sequence whose prompt was never starved.
#[test]
fn a_starved_chunk_of_a_chain_releases_it_and_the_rerun_scores_as_unstarved() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    // At the kl's shape, where a banked load's prompt runs as one chain
    // (`a_banked_loads_prompt_groups_score_as_resident`).
    let (directory, definition, package) =
        fixture_with("starved-chain", qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let (chunks, rows) = ((SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS, SPLIT_CHUNK_ROWS);
    let capacity = chunks;
    let request = RequestId(1);
    let backends = [seismic::BackendName::Cuda, seismic::BackendName::Metal];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let open = || load(&catalog, &definition, &package, backend, limits(rows, false, capacity, rows), Banked::Half);
        let (mut domain, mut bindings, _) = open();
        // The unstarved sequence: its prompt one chunk at a time, then two
        // decode steps.
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (reference, returned) = prompt_then_decode(&mut domain, bindings, request, chunks, rows, backend);
        drop(returned);
        drop(domain);

        let (mut domain, mut bindings, _) = open();
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        // The chain opens with its first chunk, which launches the rest,
        // and is starved.
        domain.plan_successors((1..chunks).map(|next| prompt_chunk(request, next, chunks, rows)).collect());
        let (pending, returned) =
            starved_step(&mut domain, bindings, prompt_chunk(request, 0, chunks, rows), std::time::Duration::from_millis(1200), backend);
        assert!(
            matches!(pending.outcome(), Outcome::Interrupted { resident: false, .. }),
            "{backend:?}: a starved chunk of a chain is interrupted, its state released: {:?}",
            pending.outcome()
        );
        domain.abort(pending).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        assert_eq!(domain.chain_holder(), None, "{backend:?}: the chain was released");
        assert!(domain.resume_state(request).is_err(), "{backend:?}: the holder's state was released");
        // Re-opened, it re-prefills and decodes as the unstarved sequence.
        let mut bindings = returned;
        domain.open_state(&mut bindings, request, None).unwrap();
        let (rerun, returned) = prompt_then_decode(&mut domain, bindings, request, chunks, rows, backend);
        assert!(rerun == reference, "{backend:?}: the re-prefilled sequence decodes as unstarved: {}", first_difference(&reference, &rerun));
        drop(returned);
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// A fresh load's prompt run one chunk at a time with no chain planned, each
/// chunk a prompt group of one step, at a bank half the experts: every step's
/// features and logits are finite on every backend this host runs. Each
/// group's park answers its request with the layer's map final (its hits,
/// the lent records' experts unmapped), so the pass's map is read after the
/// park on a backend that runs independent launches concurrently, as on one
/// that keeps submission order (the park declares the map host-written).
#[test]
fn a_lone_prompt_chunk_at_a_half_bank_is_finite() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) =
        fixture_with("lone-chunks", qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let (chunks, rows) = ((SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS, SPLIT_CHUNK_ROWS);
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _) = load(&catalog, &definition, &package, backend, limits(rows, false, chunks, rows), Banked::Half);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (steps, _) = prompt_then_decode(&mut domain, bindings, request, chunks, rows, backend);
        for (index, step) in steps.iter().enumerate() {
            let finite = step.iter().all(|(features, logits, _)| features.iter().chain(logits).flatten().all(|&bits| f32::from_bits(bits).is_finite()));
            assert!(finite, "{backend:?}: step {index} of a lone-chunk prompt at a half bank is not finite");
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Where `rerun` first differs from `reference`, steps by step and row: the
/// step, the row, which output (features, logits or the selected token),
/// the first differing value's index and both values; or that their step
/// or row counts differ.
fn first_difference(reference: &[StepBits], rerun: &[StepBits]) -> String {
    if reference.len() != rerun.len() {
        return format!("{} steps against {}", rerun.len(), reference.len());
    }
    for (step, (want, got)) in reference.iter().zip(rerun).enumerate() {
        if want.len() != got.len() {
            return format!("step {step}: {} rows against {}", got.len(), want.len());
        }
        for (row, (want, got)) in want.iter().zip(got).enumerate() {
            let values = |name: &str, want: &Option<Vec<u32>>, got: &Option<Vec<u32>>| match (want, got) {
                (Some(want), Some(got)) => want.iter().zip(got).position(|(a, b)| a != b).map(|at| {
                    let differing = want.iter().zip(got).filter(|(a, b)| a != b).count();
                    format!("step {step} row {row} {name}[{at}]: {} against {} ({differing} of {} differ)", f32::from_bits(got[at]), f32::from_bits(want[at]), want.len())
                }),
                (want, got) if want.is_some() != got.is_some() => Some(format!("step {step} row {row} {name}: present {} against {}", got.is_some(), want.is_some())),
                _ => None,
            };
            if let Some(found) = values("features", &want.0, &got.0).or_else(|| values("logits", &want.1, &got.1)) {
                return found;
            }
            if want.2 != got.2 {
                return format!("step {step} row {row} selected {:?} against {:?}", got.2, want.2);
            }
        }
    }
    "no difference".to_owned()
}

/// `request`'s prompt of `chunks` chunks of `rows` one chunk at a time,
/// then two decode steps of its selected tokens: every step's outputs.
fn prompt_then_decode(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    request: RequestId,
    chunks: usize,
    rows: usize,
    backend: seismic::BackendName,
) -> (Vec<StepBits>, crate::StateBindings) {
    let mut steps = Vec::new();
    for chunk in 0..chunks {
        let (outcome, returned) = run(domain, bindings, prompt_chunk(request, chunk, chunks, rows), false, backend);
        bindings = returned;
        steps.push(step_bits(&outcome));
    }
    let mut token = selected_token_of(&steps, backend);
    for offset in 0..2 {
        let (outcome, returned) =
            run(domain, bindings, scored(forward(request, WorkKind::Decode, &[token], chunks * rows + offset, true)), false, backend);
        bindings = returned;
        steps.push(step_bits(&outcome));
        token = selected_token_of(&steps, backend);
    }
    (steps, bindings)
}

/// The token the last of `steps` selected.
fn selected_token_of(steps: &[StepBits], backend: seismic::BackendName) -> u32 {
    steps
        .last()
        .and_then(|bits| bits.last())
        .and_then(|(_, _, selected)| *selected)
        .unwrap_or_else(|| panic!("{backend:?}: the step selects a token"))
        .0
}
