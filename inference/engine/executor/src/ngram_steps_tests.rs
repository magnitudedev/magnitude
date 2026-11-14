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
use crate::operators::attention::tests::selected::{self, CompressedForm};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::KvCodec;
use std::rc::Rc;
use std::sync::Arc;

/// A GGUF of every tensor `manifest` names (its n-gram tables IQ4_NL
/// blocks, a streamed load's expert stacks Q8_0 or Q4_0 blocks, everything else
/// F16, BF16 or F32 as it lays them out), with small deterministic values:
/// each float tensor `constant` names holds its value throughout.
fn write_gguf(
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

/// A native load's arena claim is exactly its graph scratch and its target
/// graphs' bound constants (the floor's group buffers and the held pass's
/// buffers among them), with no class beyond the floor's at one place: one
/// charge of each, none missing and none counted twice.
#[test]
fn a_native_loads_arena_claim_is_its_scratch_and_bound_constants() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fixture("arena-claim");
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Cpu].into_iter().filter(|&backend| available(&catalog, backend)) {
        let (domain, _bindings, _, _) = load(&catalog, &definition, &package, backend, limits(2, true, 1, 0), Banked::Resident);
        let (claim, scratch, constants) = domain.arena_terms();
        assert!(constants > 0, "{backend:?}: the fixture binds constants");
        assert_eq!(claim, scratch + constants, "{backend:?}: the arena claim of {scratch} scratch and {constants} constant bytes");
    }
    std::fs::remove_dir_all(&directory).ok();
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
        let (mut domain, mut bindings, _, _) =
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
            let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits(2, false, 1, 0), banked);
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
/// (`streamed_layers_at_different_homes_share_a_graph`): every decode step
/// runs the resident forms, each block's bound to its own layer's home
/// records, and each layer reads its own experts, through the resident
/// load's arithmetic, so a prompt chunk and its decode steps score bit for
/// bit as the resident load does.
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
        scores(&catalog, &definition, &package, backend, banked, Start::AtOnce)
            .assert_as_resident(backend, "the banked load's logits and selection");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On a discrete CUDA device, a partial bank of the fixture of 128 experts
/// a layer under a weight cap of half its experts, beside a host tier the
/// cap bounds (`ExpertBankPlan::host_cap`, fewer host slots than every
/// expert): the tier grows from the bank's open on to its planned size
/// (`ExpertBankPlan::host_slots`) and never holds more than its cap, and
/// the steps, every miss served from the tier or the file, score as the
/// resident load does ([`Scores::assert_as_resident`]).
#[test]
fn a_capped_partial_banks_tier_grows_within_its_cap_and_serves_every_miss() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let backend = seismic::BackendName::Cuda;
    // Only a CUDA device across a link has a host tier: on unified memory
    // (GB10's) the bank is the one store, and there is no tier to grow.
    let topology = catalog.topology();
    let linked = seismic::testing::selected(&topology, backend).is_some_and(|executor| {
        matches!(crate::platform::host_tier(&topology, executor), Ok(Some(crate::planning::HostTier::Linked(_))))
    });
    if !linked {
        return;
    }
    let definition = qwen4exp_fixture_definition_of(3, 128);
    let (directory, definition, package) = fixture_with("tier-grown", definition, streamed_experts_manifest);
    scores(&catalog, &definition, &package, backend, Banked::Half, Start::TierGrown)
        .assert_as_resident(backend, "the grown tier's load scores as the resident one");
    std::fs::remove_dir_all(&directory).ok();
}

/// On every unified device, a partial bank's misses land on the device,
/// unpriced: each record converted by its layer's landing launch as its read
/// lands, the experts after it reading it in submission order, so its steps
/// score as the resident load does ([`scores`] asserts the device landed
/// some).
#[test]
fn misses_the_device_lands_score_as_resident() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let definition = qwen4exp_fixture_definition_of(4, 128);
    let (directory, definition, package) = fixture_with("device-landed", definition, streamed_experts_manifest);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        scores(&catalog, &definition, &package, backend, Banked::Half, Start::AtOnce)
            .assert_as_resident(backend, "the misses the device landed score as the resident load's");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Whether [`scores`] steps a banked load at once, or once its host tier
/// has grown to its planned size within its weight cap.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Start {
    AtOnce,
    TierGrown,
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
    expansion: crate::operators::routed::Expansion,
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
            let (expected, bound) = crate::experts::site_reference(expansion.activation, expansion.limit, x, &named);
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
    package: &Arc<Package>,
    backend: seismic::BackendName,
    banked: Banked,
    start: Start,
) -> Scores {
    let stored = stored_experts(package, definition);
    let expansion = definition
        .decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            magnitude_family_contracts::Operator::RoutedFfn(routed) => {
                Some(crate::operators::routed::Expansion::of(&routed.expert_up)).filter(|expansion| expansion.gated)
            }
            _ => None,
        })
        .expect("the fixture's routed layers are gated");
    let mut scores = Vec::new();
    for banked in [Banked::Resident, banked] {
        let (mut domain, mut bindings, _, pricing) = load(catalog, definition, package, backend, limits(2, false, 1, 1), banked);
        let plan = domain.expert_bank_plan();
        let whole = plan.map(crate::ExpertBankPlan::whole);
        match banked {
            Banked::Resident => {}
            Banked::Whole => assert_eq!(whole, Some(true), "{backend:?}: the uncapped load holds every expert in its bank"),
            Banked::Streamed { .. } | Banked::Half => {
                assert_eq!(whole, Some(false), "{backend:?}: the capped load banks part of its experts")
            }
        }
        let site = plan.is_some_and(|plan| !plan.whole() && (0..plan.layers().len()).any(|layer| plan.sited(layer)));
        let waited = start == Start::TierGrown && !matches!(banked, Banked::Resident);
        if waited {
            // The cap bounds the tier below every expert; the tier grows
            // from the bank's open on to its planned size, within that bound.
            let (cap, planned, widest) = plan
                .and_then(|plan| plan.host_cap.map(|cap| (cap, plan.host_slots, plan.widest().get())))
                .unwrap_or_else(|| panic!("{backend:?}: a capped load's host tier has a cap"));
            assert!(cap < widest, "{backend:?}: the cap leaves the host tier {cap} host slots, fewer than every expert ({widest})");
            assert!(planned <= cap, "{backend:?}: the tier is planned at {planned} host slots, past its cap of {cap}");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            loop {
                let residency = domain.expert_residency().unwrap().unwrap();
                let host = residency.host.unwrap_or_else(|| panic!("{backend:?}: a linked load has a host tier"));
                assert_eq!(host.ceiling, cap, "{backend:?}: the host tier's ceiling is its cap");
                assert!(host.slots <= cap, "{backend:?}: the host tier holds {} host slots, past its cap of {cap}", host.slots);
                if host.slots == planned {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "{backend:?}: the host tier never grew to its planned {planned} host slots ({host:?})");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        // A site serves only once its load is priced: until then every miss
        // is served on the GPU's path.
        if site {
            domain
                .measure_load(test_measures(&domain, &pricing), &mut bindings)
                .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        }
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (outcome, mut returned) =
            run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, backend);
        // The prompt step's misses reach the site as a decode step's do.
        let mut sited = site_answers(&domain, &stored, expansion, backend, 0);
        let mut steps = vec![step_bits(&outcome)];
        for position in 2..2 + DECODE_STEPS {
            let token = (position * 37 % 256) as u32;
            let (outcome, next) =
                run(&mut domain, returned, scored(forward(request, WorkKind::Decode, &[token], position, true)), false, backend);
            sited.extend(site_answers(&domain, &stored, expansion, backend, steps.len()));
            steps.push(step_bits(&outcome));
            returned = next;
        }
        drop(returned);
        // A whole bank's every launch of decode rows (its two-row prompt
        // chunk's too) runs the resident forms; no partial one's does.
        let resident = match banked {
            Banked::Whole => 1 + DECODE_STEPS as u64,
            _ => 0,
        };
        assert_eq!(domain.resident_decodes(), resident, "{backend:?}: the decode steps the resident forms ran");
        // Where the bank places, a half bank's misses all land on the device.
        if matches!(banked, Banked::Half) && domain.expert_bank_plan().is_some_and(|plan| plan.placer() == crate::planning::Placer::Bank) {
            assert!(domain.device_landed() > 0, "{backend:?}: the device landed no miss, so its path never ran");
        }
        if waited {
            let costs = domain.expert_costs().expect("a banked load's costs");
            let misses = costs.layers.iter().map(|layer| layer.misses).sum::<u64>();
            assert!(misses > 0, "{backend:?}: a partial bank misses");
        }
        scores.push((steps, sited, site));
    }
    let [(resident, ..), (banked, sited, site)] = <[_; 2]>::try_from(scores).expect("a resident and a banked load");
    Scores { resident, banked, sited, site }
}

/// What a test's banked load is priced on: its costs and the pass they
/// price (`Pricing::measures`), every other input pending, as a first
/// load's are. A `sited` load's lines are each streamed layer's the
/// reference host's at its own record (`ExpertLines::reference`), at the
/// layer's record bytes, where a one-row miss ends on the CPU (40 µs and a
/// 4 µs handshake) before the link's 86 µs, so the route assigns misses to
/// the site, as a real model's on that host; the fixture's experts are too
/// small for a link at their own bytes to lose to a handshake.
fn test_measures(domain: &ExecutorDomain, pricing: &Pricing) -> crate::LoadMeasures {
    use crate::assessment::{ExpertLines, HostLines, Priced};
    let reference = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
    let lines = Priced::Measured(
        domain
            .expert_bank_lines()
            .expect("a sited load has a bank")
            .iter()
            .map(|bank| ExpertLines { host: HostLines { record_bytes: bank.record_bytes, ..reference.host }, ..reference })
            .collect(),
    );
    pricing.measures(lines, Priced::Pending("not measured by the test".to_owned()))
}

/// What a test load is priced on, as a load's worker prices it once its
/// warm-up ends (`LoadCosts::of`): the executor's costs, which every load
/// waits for before it serves (the synthetic costs here,
/// `assessment::synthetic::costs`), and the load's plans.
struct Pricing {
    definition: Rc<ModelDefinition>,
    /// The plan before any bank.
    resident: crate::ExecutionPlanDraft,
    /// The plan the load runs.
    planned: crate::ExecutionPlanDraft,
    costs: crate::assessment::ExecutorCosts,
}

impl Pricing {
    /// The load's measures: its costs; a banked load's experts by `lines`
    /// and `convert`, the routed pass the costs price, and every other
    /// input pending, as the job has not stored it.
    fn measures(
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
            moves: pending(),
            tier_copy: pending(),
            convert,
        });
        let costs = crate::LoadCosts::of(&self.definition, &self.resident, &self.planned, &self.costs).map_err(|error| error.to_string());
        crate::LoadMeasures { costs, experts }
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

/// On Metal and Vulkan, a whole bank of the 128-expert fixture shrunk
/// plateau by plateau between decode steps, then to its floor (as the
/// release ladder shrinks it under pressure; on unified memory the kept
/// experts' bytes move within the one store), decodes on bit for bit as the
/// resident load does, losing no step: where the backend runs a whole
/// bank's resident forms (Vulkan) every step before the first shrink runs
/// them and none after it does, each expert it dropped is a file miss from
/// then on, each one it kept is read where the shrink moved it, from the
/// first step after it.
/// With every stack of one form (one bank group) and with the last routed
/// layer's in another (two groups, each shrinking its own records), and
/// without and with lookahead.
#[test]
fn a_whole_bank_shrunk_between_decode_steps_scores_as_resident() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let manifests: [(&str, fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest, usize); 2] =
        [("whole-bank-shrunk", streamed_experts_manifest, 1), ("two-groups-shrunk", two_group_experts_manifest, 2)];
    // Only a host with Metal or Vulkan runs it, on each it has.
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Vulkan] {
        if !available(&catalog, backend) {
            continue;
        }
        for (test, manifest, groups) in manifests {
            shrunk_bank_scores_as_resident(&catalog, backend, test, manifest, groups);
        }
    }
}

/// A whole bank shrunk between decode steps on CUDA, where graphs are
/// recorded as a run first needs one: the decode launch's graphs were
/// recorded as its slots were made (`seismic::NativeGraphFamily::record_decode`),
/// so a shrink with no memory left to allocate, every ledger of the device
/// bound to what it charges, begins and completes, and its step records
/// none. The steps before it ran the whole bank's resident forms; none after
/// it does.
#[test]
#[ignore = "needs a CUDA device: run with --ignored on one"]
fn a_whole_banks_shrink_step_records_no_graph() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let cuda = seismic::BackendName::Cuda;
    assert!(available(&catalog, cuda), "the test runs on a host with a CUDA device");
    let test = "whole-bank-shrink-records";
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, cuda, limits(2, false, 1, 1), Banked::Whole);
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
        domain.bound(seismic::Ledger::Own(*store), Some(view.charged)).unwrap();
    }
    let ran = domain.resident_decodes();
    assert!(ran > 0, "the whole bank's steps before the shrink ran its resident forms");
    let floor = domain.expert_residency().unwrap().expect("a streamed load holds a bank").floor;
    let step = domain.shrink_expert_bank_to(&mut bindings, crate::BankHolding::Experts, floor, crate::CompletionWake::new(|| {})).unwrap();
    assert!(matches!(step, crate::BankStep::Begun(_)), "a whole bank shrinks: {step:?}");
    while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
    assert!(domain.expert_bank_slots() < whole, "the bank shrank from {whole:?}");
    for (store, view) in &own {
        domain.bound(seismic::Ledger::Own(*store), view.bound).unwrap();
    }
    let before = domain.graph_replays().recorded;
    bindings = decode(&mut domain, bindings);
    assert_eq!(domain.graph_replays().recorded, before, "the shrink's step records no graph");
    assert_eq!(domain.resident_decodes(), ran, "no step after the shrink runs the resident forms");
    drop(bindings);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// The live-graph owner's contract: every graph a decode step replays is
/// kept live from the load on, whichever decode launch class the step takes
/// (its count of requests), so on CUDA no decode step records a graph: not
/// the first of a class, not after Reclaim's surplus rung releases a prompt
/// group's class, and not on a bank shrink's step. Each step's recordings,
/// by family, are reported before the verdict.
#[test]
#[ignore = "needs a CUDA device: run with --ignored on one"]
fn no_decode_step_records_a_graph() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let cuda = seismic::BackendName::Cuda;
    assert!(available(&catalog, cuda), "the test runs on a host with a CUDA device");
    decode_steps_record_no_graph(&catalog, cuda);
}

/// [`no_decode_step_records_a_graph`] on Metal, run by default on every
/// Mac: a Metal schedule holds no device memory (`GraphCost::None`), so
/// every family's graph charge is none, and `record_decode` pins nothing
/// there, so no decode step may record a graph either: of one request,
/// eight and four, across Reclaim's surplus release and a bank shrink's
/// step. It fails once Metal charges or records graphs without the pins.
#[cfg(target_os = "macos")]
#[test]
fn no_decode_step_records_a_graph_on_metal() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let metal = seismic::BackendName::Metal;
    assert!(available(&catalog, metal), "a Mac has a Metal device");
    decode_steps_record_no_graph(&catalog, metal);
}

/// Decode steps' recordings, by family, of open requests at their positions.
struct DecodeSteps {
    requests: Vec<RequestId>,
    positions: Vec<usize>,
    recorded: Vec<(&'static str, Vec<(&'static str, u64)>)>,
}

impl DecodeSteps {
    const REQUESTS: usize = 8;

    /// Limits of launches of up to [`Self::REQUESTS`] requests, in chains of
    /// up to `chain_capacity`.
    fn limits(chain_capacity: usize) -> ResourceLimits {
        ResourceLimits {
            max_launch_slots: Self::REQUESTS,
            max_selected_rows: Self::REQUESTS,
            ..limits(Self::REQUESTS, false, chain_capacity, Self::REQUESTS)
        }
    }

    /// [`Self::REQUESTS`] requests opened, each prompted with two tokens, the
    /// first through a chain of `chunks` chunks.
    fn open(domain: &mut ExecutorDomain, mut bindings: crate::StateBindings, chunks: usize, backend: seismic::BackendName) -> (Self, crate::StateBindings) {
        let requests = (1..=Self::REQUESTS as u64).map(RequestId).collect::<Vec<_>>();
        let mut positions = Vec::new();
        for (index, &request) in requests.iter().enumerate() {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_state(&mut bindings, request, None).unwrap();
            let chunks = if index == 0 { chunks } else { 1 };
            domain.plan_successors((1..chunks).map(|next| prompt_chunk(request, next, chunks, Self::REQUESTS)).collect());
            for chunk in 0..chunks {
                let (_, returned) = run(domain, bindings, prompt_chunk(request, chunk, chunks, Self::REQUESTS), chunk > 0, backend);
                bindings = returned;
            }
            positions.push(chunks * Self::REQUESTS);
        }
        (Self { requests, positions, recorded: Vec::new() }, bindings)
    }

    /// A decode step of the first `count` requests, its recordings kept as
    /// `step`'s.
    fn step(&mut self, domain: &mut ExecutorDomain, mut bindings: crate::StateBindings, count: usize, step: &'static str, backend: seismic::BackendName) -> crate::StateBindings {
        let operations = self.requests[..count]
            .iter()
            .zip(&self.positions)
            .map(|(&request, &position)| scored(forward(request, WorkKind::Decode, &[(position * 37 % 256) as u32], position, true)))
            .collect::<Vec<_>>();
        let before = domain.family_graph_replays();
        let reserved = domain.reserve(&mut bindings, &operations).unwrap().into_resources();
        let ReservedResources::Target(reservation) = reserved else {
            panic!("{backend:?}: a decode step reserves the target lane");
        };
        let flight = domain.submit_target(bindings, &operations, reservation).unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
        let (pendings, bindings) = domain.finish_target(flight).unwrap();
        for pending in pendings {
            domain.reconcile(pending, PhysicalDecision { accepted_rows: 1 }).unwrap();
        }
        for position in &mut self.positions[..count] {
            *position += 1;
        }
        self.recorded.push((step, recorded_since(domain, before)));
        bindings
    }

    /// The verdict: no step recorded a graph.
    fn assert_none(&self, backend: seismic::BackendName, context: &str) {
        eprintln!("{backend:?}, {context}: graphs recorded by decode steps (step, graphs by family): {:?}", self.recorded);
        assert!(self.recorded.iter().all(|(_, graphs)| graphs.is_empty()), "{backend:?}, {context}: a decode step recorded graphs: {:?}", self.recorded);
    }
}

/// The graphs each of `domain`'s families recorded since `before`, its
/// replays then: the families that recorded any.
fn recorded_since(domain: &ExecutorDomain, before: Vec<(&'static str, seismic::GraphReplays)>) -> Vec<(&'static str, u64)> {
    domain
        .family_graph_replays()
        .into_iter()
        .zip(before)
        .map(|((family, after), (_, before))| (family, after.since(before).recorded))
        .filter(|&(_, graphs)| graphs > 0)
        .collect()
}

fn decode_steps_record_no_graph(catalog: &seismic::Devices, backend: seismic::BackendName) {
    let test = "decode-records-no-graph";
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    const FOUR: usize = DecodeSteps::REQUESTS / 2;
    // Reclaim: a load whose experts are resident chains any prompt, its
    // group claiming a class of two places as it opens and freeing it as
    // its chain closes, before Reclaim's surplus rung releases the slots.
    {
        let (mut domain, bindings, _, _) = load(catalog, &definition, &package, backend, DecodeSteps::limits(2), Banked::Resident);
        if backend == seismic::BackendName::Metal {
            assert_eq!(domain.device().graph_cost(), seismic::GraphCost::None, "Metal charges no family for its graphs");
        }
        let (mut steps, mut bindings) = DecodeSteps::open(&mut domain, bindings, 2, backend);
        assert_eq!(domain.group_places().get(), 1, "{backend:?}: the prompt group's class is freed as its chain closes");
        bindings = steps.step(&mut domain, bindings, 1, "the first of one request", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "the first of eight", backend);
        let released = crate::domain::fixtures::shrink_settled(&mut domain, &mut bindings, magnitude_state::ShrinkPolicy::Reclaim);
        assert_eq!(domain.group_places().get(), 1, "{backend:?}: Reclaim holds no class");
        bindings = steps.step(&mut domain, bindings, FOUR, "the first of four, after the surplus release", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "of eight, after the surplus release", backend);
        steps.assert_none(backend, &format!("a Reclaim release of {released} bytes"));
        drop(bindings);
    }
    // A bank shrink: experts streamed through a whole bank, shrunk to its
    // floor between decode steps.
    if STREAMING_BACKENDS.contains(&backend) {
        let (mut domain, bindings, _, _) = load(catalog, &definition, &package, backend, DecodeSteps::limits(1), Banked::Whole);
        let (mut steps, mut bindings) = DecodeSteps::open(&mut domain, bindings, 1, backend);
        bindings = steps.step(&mut domain, bindings, 1, "the first of one request", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "the first of eight", backend);
        let whole = domain.expert_bank_slots();
        let device = domain.device().clone();
        let own = device.ledgers().own;
        for (store, view) in &own {
            domain.bound(seismic::Ledger::Own(*store), Some(view.charged)).unwrap();
        }
        let floor = domain.expert_residency().unwrap().expect("a streamed load holds a bank").floor;
        let step = domain.shrink_expert_bank_to(&mut bindings, crate::BankHolding::Experts, floor, crate::CompletionWake::new(|| {})).unwrap();
        assert!(matches!(step, crate::BankStep::Begun(_)), "{backend:?}: a whole bank shrinks: {step:?}");
        while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
        assert!(domain.expert_bank_slots() < whole, "{backend:?}: the bank shrank from {whole:?}");
        for (store, view) in &own {
            domain.bound(seismic::Ledger::Own(*store), view.bound).unwrap();
        }
        bindings = steps.step(&mut domain, bindings, FOUR, "the first of four, the bank shrink's step", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "of eight, after the bank shrink", backend);
        steps.assert_none(backend, "a bank shrink");
        drop(bindings);
    }
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// The drafter's graph families on CUDA: a load of an MTP head seals each
/// family within what the planner charged for it
/// (`programs::graph::serial_family` asserts it as the head, readout and
/// state families seal), and its priming and drafting steps record no graph
/// beyond those charges. Two passes run the same work, each over two fresh
/// requests: each request primes its head through a prompt chunk and the
/// chunk's pipelined successor, two primings in flight, then drafts one
/// and two proposals, alone and beside the other request. Every graph a
/// drafter or state-copy step replays is recorded at its shape's first use
/// and kept within its family's charge, so the second pass records none: a
/// graph beyond its charge would be released and recorded again. Each
/// step's recordings, by family, are reported before the verdict.
#[test]
#[ignore = "needs a CUDA device: run with --ignored on one"]
fn drafter_steps_record_no_graph_beyond_their_charge() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let cuda = seismic::BackendName::Cuda;
    assert!(available(&catalog, cuda), "the test runs on a host with a CUDA device");
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with("drafter-graphs", mtp, crate::planning::fixtures::fixture_manifest);
    let method = PlannedMethod::Mtp {
        greedy_proposals: DrafterSteps::PROPOSALS as u8,
        sampled_proposals: DrafterSteps::PROPOSALS as u8,
        widths: crate::DraftWidths::Adaptive,
    };
    // Lookahead: a prompt chunk's successor runs behind it, its own drafter
    // entry behind it.
    let limits = limits(DrafterSteps::ROWS, true, 1, 0);
    let (mut domain, mut bindings, _, _) = load_with(&catalog, &definition, &package, cuda, limits, Banked::Resident, method);
    let row_bytes = definition.decoder.hidden as usize * definition.decoder.activation_dtype.bytes();
    let mut steps = DrafterSteps { backend: cuda, row_bytes, recorded: Vec::new(), drafted: Vec::new() };
    for pass in [1, 2] {
        let first = steps.recorded.len();
        let requests = [RequestId(2 * pass - 1), RequestId(2 * pass)];
        let mut positions = Vec::new();
        for request in requests {
            let (position, returned) = steps.prime(&mut domain, bindings, request);
            bindings = returned;
            positions.push(position);
        }
        // Drafts of one and two proposals, of one request and of both.
        for (count, proposals) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
            let drafting = requests[..count].iter().copied().zip(positions.iter_mut());
            bindings = steps.draft(&mut domain, bindings, drafting, proposals);
        }
        for request in requests {
            domain.close(request).unwrap_or_else(|error| panic!("{error}"));
        }
        if pass == 2 {
            steps.assert_none_since(first);
        }
    }
    drop(bindings);
    drop(domain);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// A head that does not select reads each entry row's accepted history
/// alone, whatever the entry pass's rows: its launch's earlier rows are fresh
/// to the launch, not stored history. A draft from a two-row entry pass runs
/// (the lane's validation of each row's visible spans agrees with the
/// producer's) on a head of the plain fixture, on the CPU backend.
#[test]
fn a_head_without_selection_drafts_from_a_multi_row_entry_pass() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let backend = seismic::BackendName::Cpu;
    assert!(available(&catalog, backend), "the CPU backend is always available");
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with("head-multi-row-entry", mtp, crate::planning::fixtures::fixture_manifest);
    let method = PlannedMethod::Mtp {
        greedy_proposals: DrafterSteps::PROPOSALS as u8,
        sampled_proposals: DrafterSteps::PROPOSALS as u8,
        widths: crate::DraftWidths::Adaptive,
    };
    let (mut domain, bindings, _, _) =
        load_with(&catalog, &definition, &package, backend, limits(DrafterSteps::ROWS, true, 1, 0), Banked::Resident, method);
    let row_bytes = definition.decoder.hidden as usize * definition.decoder.activation_dtype.bytes();
    let mut steps = DrafterSteps { backend, row_bytes, recorded: Vec::new(), drafted: Vec::new() };
    let request = RequestId(1);
    let (mut position, mut bindings) = steps.prime(&mut domain, bindings, request);
    for _ in 0..2 {
        bindings = steps.draft_from(&mut domain, bindings, std::iter::once((request, &mut position)), 2, DrafterSteps::PROPOSALS);
    }
    assert_eq!(steps.drafted.len(), 2);
    domain.close(request).unwrap_or_else(|error| panic!("{error}"));
    drop(bindings);
    drop(domain);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// The limits the harness's drafting load resolves from its step allowances
/// (`StepAllowances`, which `resource_limits` and the harness read too).
fn harness_drafting_limits() -> ResourceLimits {
    let steps = crate::StepAllowances::EVALUATION_DRAFTING;
    ResourceLimits {
        max_launch_slots: steps.launch_slots(),
        max_selected_rows: steps.selection_bound(),
        max_drafting_slots: steps.selection_bound(),
        ..limits(steps.launch_rows(), true, 1, 0)
    }
}

/// A load of the GLM MTP head (a general routed feed-forward over one residual) binds its graphs on every backend
/// this host has, at a launch bound past the routed decode form's rows: its head classes certify, seal and bind the constants and
/// weights its prepared layouts hold. The load runs no step.
#[test]
fn glm_mtp_head_loads_and_binds_its_graphs_on_every_available_backend() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let method = PlannedMethod::Mtp {
        greedy_proposals: DrafterSteps::PROPOSALS as u8,
        sampled_proposals: DrafterSteps::PROPOSALS as u8,
        widths: crate::DraftWidths::Adaptive,
    };
    for backend in [seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan] {
        if !available(&catalog, backend) {
            continue;
        }
        let (definition, manifest) = crate::planning::fixtures::glm_slice_b_mtp_fixture();
        let (directory, definition, package) = fixture_stored("glm-mtp-head", definition, &manifest, &|_| None);
        // The harness's drafting load: the head's grouped and many-slot classes certify and bind too.
        let limits = harness_drafting_limits();
        let (domain, bindings, _, _) = load_with(&catalog, &definition, &package, backend, limits, Banked::Resident, method);
        drop(bindings);
        drop(domain);
        std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
    }
}

/// The GLM MTP head reads the rows its own pooled indexer picks, on every
/// backend this host has. Its prompt rows enter its state, then drafts of two
/// proposals each accept their entry row alone, so every round rolls the
/// chained row's pooled keys back. The same drafts run over two loads that
/// differ only in the head's indexer weights, its own and the body's tied
/// ones: until a row of the head sees more complete units than the indexer
/// picks, the head reads its whole history and the loads draft bit for bit
/// alike; from the first round that scores, the picks decide what the head
/// reads, so the drafts differ. They would not were the head to read the
/// body's picks or never to score.
#[test]
fn glm_mtp_head_drafts_from_the_rows_its_own_indexer_picks_on_every_available_backend() {
    use crate::planning::fixtures::{GLM_SLICE_B_RATE, GLM_SLICE_B_TOP};
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let method = PlannedMethod::Mtp {
        greedy_proposals: DrafterSteps::PROPOSALS as u8,
        sampled_proposals: DrafterSteps::PROPOSALS as u8,
        widths: crate::DraftWidths::Adaptive,
    };
    // A round's last row sits at position `PRIMED + round + PROPOSALS - 1`
    // and sees the units complete through it, `(position + 1) / RATE`.
    let scoring = ((GLM_SLICE_B_TOP + 1) * GLM_SLICE_B_RATE) as usize - DrafterSteps::PRIMED - DrafterSteps::PROPOSALS;
    let rounds = scoring + 4;
    for backend in [seismic::BackendName::Cpu, seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan] {
        if !available(&catalog, backend) {
            continue;
        }
        let drafts = |own_indexer: bool| {
            let (definition, manifest) = crate::planning::fixtures::glm_slice_b_mtp_fixture_of(own_indexer);
            let (directory, definition, package) = fixture_stored("glm-mtp-head-drafts", definition, &manifest, &|_| None);
            let (mut domain, bindings, _, _) =
                load_with(&catalog, &definition, &package, backend, limits(DrafterSteps::ROWS, true, 1, 0), Banked::Resident, method);
            let row_bytes = definition.decoder.hidden as usize * definition.decoder.activation_dtype.bytes();
            let mut steps = DrafterSteps { backend, row_bytes, recorded: Vec::new(), drafted: Vec::new() };
            let request = RequestId(1);
            let (mut position, mut bindings) = steps.prime(&mut domain, bindings, request);
            assert_eq!(position, DrafterSteps::PRIMED);
            for _ in 0..rounds {
                bindings = steps.draft(&mut domain, bindings, std::iter::once((request, &mut position)), DrafterSteps::PROPOSALS);
            }
            domain.close(request).unwrap_or_else(|error| panic!("{error}"));
            drop(bindings);
            drop(domain);
            std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
            steps.drafted
        };
        let own = drafts(true);
        let tied = drafts(false);
        assert_eq!(own.len(), rounds);
        assert_eq!(own[..scoring], tied[..scoring], "{backend:?}: drafts before any row scores read the whole history");
        assert!(
            own[scoring..] != tied[scoring..],
            "{backend:?}: drafts from the first scoring round do not depend on the head's indexer"
        );
    }
}

/// Drafter steps on CUDA, each step's recordings kept by family.
struct DrafterSteps {
    backend: seismic::BackendName,
    /// Bytes of a drafter entry row's conditioning feature.
    row_bytes: usize,
    recorded: Vec<(String, Vec<(&'static str, u64)>)>,
    /// What each draft proposed, in order.
    drafted: Vec<Outcome>,
}

impl DrafterSteps {
    /// Rows of a prompt chunk, the launch row bound.
    const ROWS: usize = 4;
    /// The head rows a primed prompt leaves (`prime`).
    const PRIMED: usize = 2 * Self::ROWS - 1;
    /// The most proposals a draft makes.
    const PROPOSALS: usize = 2;
    /// The families a drafter step runs graphs of.
    const DRAFTING: [&'static str; 2] = ["drafter", "state"];

    /// Keep `step`'s recordings, each family's since `before`.
    fn keep(&mut self, domain: &ExecutorDomain, before: Vec<(&'static str, seismic::GraphReplays)>, step: String) {
        self.recorded.push((step, recorded_since(domain, before)));
    }

    /// Open `request` and prompt it with two chunks, each entering its rows
    /// to the head behind it: the second is the first's pipelined
    /// successor, so both entries are in flight at once. Returns the head's
    /// position after them.
    fn prime(&mut self, domain: &mut ExecutorDomain, mut bindings: crate::StateBindings, request: RequestId) -> (usize, crate::StateBindings) {
        let backend = self.backend;
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let before = domain.family_graph_replays();
        domain.plan_successors(vec![Self::primed_chunk(request, 1)]);
        for chunk in 0..2 {
            let (_, returned) = run(domain, bindings, Self::primed_chunk(request, chunk), chunk > 0, backend);
            bindings = returned;
        }
        self.keep(domain, before, format!("{request:?}'s two primed chunks"));
        // An embedded head's entry pairs each token with the feature of the
        // row before it, so it trails the target by a row.
        (Self::PRIMED, bindings)
    }

    /// Prompt chunk `chunk` of two of `request`'s prompt, its features
    /// demanded and its rows entered to the head behind it as an embedded
    /// head enters them (`Priming::boundary`); the last selects the first
    /// token.
    fn primed_chunk(request: RequestId, chunk: usize) -> Operation {
        let position = chunk * Self::ROWS;
        let last = chunk == 1;
        let tokens = (position..position + Self::ROWS)
            .map(|row| TokenId((row * 37 + 11 + request.0 as usize * 5) as u32 % 256))
            .collect::<Vec<_>>();
        let demand = crate::batching::Demand::FEATURES;
        Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            prime: Some(crate::Priming {
                boundary: true,
                tokens: if position == 0 { tokens[1..].to_vec() } else { tokens.clone() },
                position: position.saturating_sub(1),
                draft_from: 2 * Self::ROWS,
            }),
            tokens,
            position,
            conditioning: None,
            demand: if last { demand | crate::batching::Demand::SELECT } else { demand },
            select: last.then(|| Self::greedy(position + Self::ROWS - 1)).into_iter().collect(),
            committed: Self::ROWS,
        }
    }

    fn greedy(position: usize) -> SelectSpec {
        SelectSpec {
            sampling: Sampling::Greedy,
            seed: 0,
            position,
            domain: 0,
            mask: None,
            shaping: Shaping::default(),
            history: None,
        }
    }

    /// A draft of `proposals` proposals by each of `drafting`, its request
    /// and its head's position, which the draft's entry row advances.
    fn draft<'p>(
        &mut self,
        domain: &mut ExecutorDomain,
        bindings: crate::StateBindings,
        drafting: impl Iterator<Item = (RequestId, &'p mut usize)>,
        proposals: usize,
    ) -> crate::StateBindings {
        self.draft_from(domain, bindings, drafting, 1, proposals)
    }

    /// [`Self::draft`] from `entry_rows` entry rows, all of which the round
    /// accepts.
    fn draft_from<'p>(
        &mut self,
        domain: &mut ExecutorDomain,
        mut bindings: crate::StateBindings,
        drafting: impl Iterator<Item = (RequestId, &'p mut usize)>,
        entry_rows: usize,
        proposals: usize,
    ) -> crate::StateBindings {
        assert!(proposals <= Self::PROPOSALS, "a draft makes at most the method's proposals");
        let mut drafted = Vec::new();
        let operations = drafting
            .map(|(request, position)| {
                let at = *position;
                *position += entry_rows;
                drafted.push(request);
                Operation::Head {
                    request,
                    phase: crate::HeadPhase::Generation,
                    tokens: (0..entry_rows).map(|row| TokenId(((at + row) * 13 % 256) as u32)).collect(),
                    conditioning: crate::FeatureRows::new(vec![0; self.row_bytes * entry_rows].into(), entry_rows).unwrap(),
                    position: at,
                    proposals: (1..=proposals).map(|step| Self::greedy(at + step)).collect(),
                    form: crate::DraftForm::Chained,
                }
            })
            .collect::<Vec<_>>();
        let before = domain.family_graph_replays();
        let reserved = domain.reserve(&mut bindings, &operations).unwrap_or_else(|error| panic!("{error}")).into_resources();
        let ReservedResources::Head(workspace, output, advances) = reserved else {
            panic!("a draft reserves the head lane");
        };
        let flight = domain
            .submit_head(bindings, &operations, workspace, output, advances)
            .unwrap_or_else(|failure| panic!("{}", failure.error()));
        let (pendings, bindings) = domain.finish_head(flight).unwrap_or_else(|error| panic!("{error}"));
        for pending in pendings {
            self.drafted.push(pending.outcome().clone());
            domain.reconcile(pending, PhysicalDecision { accepted_rows: entry_rows }).unwrap_or_else(|error| panic!("{error}"));
        }
        self.keep(domain, before, format!("a draft of {proposals} proposals by {drafted:?}"));
        bindings
    }

    /// The verdict: no drafting family recorded a graph in the steps from
    /// `first` on.
    fn assert_none_since(&self, first: usize) {
        eprintln!("graphs recorded by drafter steps (step, graphs by family): {:?}", self.recorded);
        let beyond = self.recorded[first..]
            .iter()
            .filter(|(_, graphs)| graphs.iter().any(|(family, _)| Self::DRAFTING.contains(family)))
            .collect::<Vec<_>>();
        assert!(beyond.is_empty(), "drafter steps recorded graphs again, beyond their families' charges: {beyond:?}");
    }
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
/// bank groups, on `backend`.
fn shrunk_bank_scores_as_resident(
    catalog: &seismic::Devices,
    backend: seismic::BackendName,
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
    groups: usize,
) {
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(3, 128), manifest);
    // Without lookahead each step's token is the host's; with it, the
    // device's selection, each step claiming the successor queued behind
    // the one before (unless a shrink orphaned it), as served decode runs.
    let decode = |banked: Banked, lookahead: bool| {
        let (mut domain, mut bindings, _, _) = load(catalog, &definition, &package, backend, limits(2, lookahead, 1, 1), banked);
        if matches!(banked, Banked::Whole) {
            assert_eq!(domain.expert_bank_plan().map(|bank| bank.groups().len()), Some(groups), "{test}: the bank's groups");
        }
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (outcome, mut bindings) = run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, backend);
        let mut steps = vec![step_bits(&outcome)];
        let mut shrinks = 0;
        // The launches of decode rows run before the first shrink (its
        // two-row prompt chunk's among them), and the resident forms' runs
        // as it began.
        let mut first_shrink = None;
        for position in 2..2 + DECODE_STEPS {
            // From a quarter of the decode, one slot a step, each shrink
            // beside the misses the last ones made land; then to the floor
            // at once.
            let shrinking = 2 + DECODE_STEPS / 4..2 + DECODE_STEPS / 2;
            if matches!(banked, Banked::Whole) && (shrinking.contains(&position) || position == shrinking.end) {
                let residency = domain.expert_residency().unwrap().expect("a whole bank");
                let to = match position == shrinking.end {
                    true => residency.floor,
                    false => residency.slots.saturating_sub(1).max(residency.floor),
                };
                let wake = crate::CompletionWake::new(|| {});
                // Each shrink reaches the next physical plateau; at the
                // floor the bank is at its limit.
                // A step still running (a queued successor) defers it to a
                // later boundary.
                match domain.shrink_expert_bank_to(&mut bindings, crate::BankHolding::Experts, to, wake).unwrap() {
                    crate::BankStep::Begun(_) => {
                        shrinks += 1;
                        first_shrink.get_or_insert((steps.len() as u64, domain.resident_decodes()));
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
            let (outcome, next, _) = step(&mut domain, bindings, scored(forward(request, WorkKind::Decode, &[token], position, true)), backend);
            steps.push(step_bits(&outcome));
            bindings = next;
        }
        // Every step before the first shrink ran the resident forms (with
        // lookahead, a successor begun then too), and none from it on: a
        // shrink begins only once every begun step has ended.
        if matches!(banked, Banked::Whole) {
            let (decoded, ran) = first_shrink.expect("the whole bank shrank");
            assert!(ran >= decoded, "{test}, lookahead {lookahead}: {ran} resident runs before the first shrink, of {decoded} steps");
            assert!(lookahead || ran == decoded, "{test}: {ran} resident runs before the first shrink, of {decoded} steps");
            assert_eq!(domain.resident_decodes(), ran, "{test}, lookahead {lookahead}: a decode step after the first shrink ran the resident forms");
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
fn fixture(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_of(test, stored_table_manifest)
}

/// [`fixture`] of the tensors `manifest` lays out.
fn fixture_of(
    test: &str,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_with(test, qwen4exp_fixture_definition(), manifest)
}

/// [`fixture_of`] the decoder `definition`.
fn fixture_with(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    fixture_valued(test, definition, manifest, &|_| None)
}

/// [`fixture_with`], each tensor `constant` names holding its value.
fn fixture_valued(
    test: &str,
    definition: ModelDefinition,
    manifest: fn(&ModelDefinition) -> magnitude_artifacts::PackageManifest,
    constant: &dyn Fn(&str) -> Option<f32>,
) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let manifest = manifest(&definition);
    fixture_stored(test, definition, &manifest, constant)
}

/// [`fixture_valued`] of the tensors `manifest` names.
fn fixture_stored(
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
    package: &Arc<Package>,
    backend: seismic::BackendName,
    Prompts { requests, chunks, rows }: Prompts,
    chain_capacity: usize,
    banked: Banked,
) -> (Vec<(StepBits, Option<usize>)>, u64) {
    let (mut domain, mut bindings, pass_rows, _) = load(
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

/// On CUDA, Metal and the CPU: an Engram input (DeepSeek V4.1's form: mapped
/// ids, a fill and no boundary, the stored product gate, no convolution)
/// ending an attention block over Sinkhorn streams adds `gate · value`, so
/// at a zero value projection it adds exactly nothing. Two prompt chunks,
/// whose second reads the first's token window, then equal bit for bit
/// those of the decoder without it: the Sinkhorn `pre` the block carries
/// passes through the input unchanged, and the input's bank binds beside
/// the attention history under its own slot tables.
#[test]
fn a_zero_valued_engram_input_leaves_its_attention_decoder_unchanged() {
    use crate::planning::fixtures::{ENGRAM_VALUE, engram_fixture_definition};
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let zero = |name: &str| (name == ENGRAM_VALUE).then_some(0.0);
    let (directory, with, package) =
        fixture_valued("engram-zero", engram_fixture_definition(), stored_table_manifest, &zero);
    // The same file, its Engram tensors unread.
    let mut without = (*with).clone();
    without.decoder.blocks[0].sublayers.truncate(2);
    let without = Rc::new(without);
    let prompts = Prompts {
        requests: 1,
        chunks: 2,
        rows: CHUNK_ROWS,
    };
    let backends = [seismic::BackendName::Cuda, seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (added, _) = chained_prompts(&catalog, &with, &package, backend, prompts, 1, Banked::Resident);
        let (plain, _) = chained_prompts(&catalog, &without, &package, backend, prompts, 1, Banked::Resident);
        assert_steps_equal(&plain, &added, prompts, &format!("{backend:?}: with a zero-valued Engram input"));
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On Metal and on the CPU: two requests decode through an Engram input
/// ending an attention block in one launch of two slots, as each decodes
/// alone. The input's bank keeps a token window per request slot under
/// exact slot tables, so the block's graphs are classed by request slots
/// as well as history segments (`BlockKeeps::HistoryAndBank`): the
/// launch's two slots' tables bind a graph sealed for two. Each request's
/// two decode steps hash its own prompt's token window, so its logits stay
/// within float rounding of its decode alone (`TOGETHER_KL`: the two
/// launches' row classes differ).
#[test]
fn engram_requests_decode_together_as_each_alone() {
    use crate::planning::fixtures::engram_fixture_definition;
    const TOGETHER_KL: f64 = 1e-4;
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) =
        fixture_valued("engram-slots", engram_fixture_definition(), stored_table_manifest, &|_: &str| None);
    let requests = [RequestId(1), RequestId(2)];
    // Each request's logits at its two decode steps, the requests decoding
    // `together` in one launch or each alone.
    let decoded = |backend: seismic::BackendName, together: bool| {
        let limits = limits(CHUNK_ROWS, false, 1, CHUNK_ROWS);
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits, Banked::Resident);
        for &request in &requests {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_state(&mut bindings, request, None).unwrap();
            bindings = run(&mut domain, bindings, prompt_chunk(request, 0, 1, CHUNK_ROWS), false, backend).1;
        }
        let mut logits = Vec::new();
        for offset in 0..2 {
            let decode = |request: RequestId| {
                let token = (offset * 13 + request.0 as usize * 29) as u32 % 256;
                scored(forward(request, WorkKind::Decode, &[token], CHUNK_ROWS + offset, false))
            };
            let launches = match together {
                true => vec![requests.map(decode).to_vec()],
                false => requests.map(|request| vec![decode(request)]).to_vec(),
            };
            for operations in launches {
                let (outcomes, returned, _) = steps(&mut domain, bindings, operations, backend);
                bindings = returned;
                logits.extend(outcomes.iter().map(|outcome| {
                    let Outcome::Forward { rows } = outcome else {
                        panic!("{backend:?}: a forward has a forward outcome: {outcome:?}");
                    };
                    rows[0].logits.as_ref().expect("a scored row has logits").read_to_host().unwrap()
                }));
            }
        }
        logits
    };
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (together, alone) = (decoded(backend, true), decoded(backend, false));
        assert_eq!(together.len(), 4);
        for (at, (together, alone)) in together.iter().zip(&alone).enumerate() {
            let kl = kl_of(alone, together);
            assert!(kl <= TOGETHER_KL, "{backend:?}: decode {} of request {}: KL {kl:.3e}", at / 2, at % 2 + 1);
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// A Reindexed form whose pool (32 blocks of 2 units) and picks (64) hold
/// every row of `compressed_layers_decode_equals_prefill`'s contexts (at
/// most 46), so no selection there hinges on a near-tie of its scores.
const WIDE_POOL: CompressedForm = CompressedForm::Reindexed { blocks: 32, block_units: 2, top: 64 };

/// Decode equals prefill, on Metal, CUDA, Vulkan and the CPU: slice C's layer forms
/// (a window, a pooled source of rate 2 with its own indexer, and a layer
/// reusing that source and its picks) give each of the 8 rows after a
/// prompt the same distribution whether they run one row a forward after
/// the prompt, or in one forward with the prompt's rows after its last whole
/// 16-row chunk, as slice C's verdict scores a prompt and its rows together.
/// - Prompt 15 runs whole in that forward, as the verdict's short fixture
///   does (prompt 15, rows 15–22).
/// - Prompts 5, 6, 37 and 38 end on either side of a group boundary; 37
///   and 38's forward opens at 32, a member of the group its second row
///   completes, as slice C's 613 split's verify opens at 612.
///
/// It holds in two numerics: the test's own (unrounded pooled and index
/// rows, two picks), and slice C's (`slice_c_numerics`: its trained row
/// roundings, and a top that picks every visible block, as below 1,024
/// tokens). It holds too for `WIDE_POOL` (a rate-1 source whose indexer
/// produces a candidate pool, and a Reindex layer restricted to it), and for
/// two Full layers at one rate sharing its domains: of one form, which share
/// one block plan (each binding its own pooled planes), and, as the control,
/// of two forms, a gated and a mean pool
/// (`two_full_layers_of_one_form_share_one_plan`). The pair differ only by
/// summation order, so each row's KL stays at float rounding (`DECODE_KL`).
/// Norm weights are 1, so the rows' attention moves the logits.
#[test]
fn compressed_layers_decode_equals_prefill() {
    const DECODE_KL: f64 = 1e-4;
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (definition, manifest) = selected::compressed(CompressedForm::FullAndReuse);
    let (wide, wide_manifest) = selected::compressed(WIDE_POOL);
    let (two_gated, two_gated_manifest) = selected::compressed(CompressedForm::TwoFull { second_gated: true });
    let (two_mixed, two_mixed_manifest) = selected::compressed(CompressedForm::TwoFull { second_gated: false });
    let unit = |name: &str| name.ends_with("norm").then_some(1.0);
    let limits = ResourceLimits { conversations: std::num::NonZeroUsize::new(2).unwrap(), ..limits(32, false, 1, 32) };
    let mut differing = Vec::new();
    let backends = [
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ];
    let forms = [
        ("unrounded", definition.clone(), &manifest),
        ("slice-c", slice_c_numerics(definition), &manifest),
        ("wide-pool", wide, &wide_manifest),
        ("two-full-gated", two_gated, &two_gated_manifest),
        ("two-full-mixed", two_mixed, &two_mixed_manifest),
    ]
    .map(|(form, definition, manifest)| (form, fixture_stored("decode-equals-prefill", definition, manifest, &unit)));
    // Each backend whole before the next, so a backend that fails leaves the
    // ones before it reported.
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        for (form, (_, definition, package)) in &forms {
            for prompt in [15, 5, 6, 37, 38] {
                let case = format!("{form} {backend:?} prompt {prompt}");
                eprintln!("decode against prefill: {case}: loading");
                let (mut domain, bindings, _, _) = load(&catalog, definition, package, backend, limits, Banked::Resident);
                let (stepped, bindings) = decode_logits(&mut domain, bindings, RequestId(1), prompt, true, backend);
                let (batched, _) = decode_logits(&mut domain, bindings, RequestId(2), prompt, false, backend);
                for (position, (stepped, batched)) in (prompt..).zip(stepped.iter().zip(&batched)) {
                    let kl = kl_of(stepped, batched);
                    let row = format!("{case} position {position}: KL {kl:.3e}");
                    // As it goes, so a later failure still shows what passed.
                    eprintln!("decode against prefill: {row}");
                    if kl > DECODE_KL {
                        differing.push(row);
                    }
                }
            }
        }
    }
    for (_, (directory, ..)) in &forms {
        std::fs::remove_dir_all(directory).ok();
    }
    assert!(differing.is_empty(), "decode differs from prefill:\n{}", differing.join("\n"));
}

/// `definition` (`compressed`'s full-and-reuse forms) in slice C's numerics: its window
/// rows E4M3 per 32 (UE8M0, floor 1e-4), its pooled rows E2M1 per 16 (an
/// E4M3 scale, floor 6·2⁻⁹), its index query and keys E2M1 per 32 (UE8M0,
/// floor 6·2⁻¹²⁶), as V4.1's family states them, and a top of 512, which
/// picks every block of the test's contexts.
fn slice_c_numerics(mut definition: ModelDefinition) -> ModelDefinition {
    use magnitude_family_contracts::{
        BlockElement, BlockRounding, BlockScale, HistoryReads, HistoryRows, IndexerKeys, Operator, RowProducer,
        RowRounding, Selection,
    };
    let rounding = |element, block, scale, amax_floor| RowRounding::Block(BlockRounding { element, block, scale, amax_floor });
    let index = rounding(BlockElement::E2M1, 32, BlockScale::Ue8m0, 6.0 * f32::MIN_POSITIVE as f64);
    for block in &mut definition.decoder.blocks {
        let Operator::Attention(attention) = &mut block.sublayers[0].op else {
            unreachable!("each block opens with attention")
        };
        for source in &mut attention.history {
            match &mut source.rows {
                HistoryRows::Own(RowProducer::Projected(rows)) => {
                    rows.rounding = rounding(BlockElement::E4M3, 32, BlockScale::Ue8m0, 1e-4);
                }
                HistoryRows::Own(RowProducer::Pooled(rows)) => {
                    rows.rounding = rounding(BlockElement::E2M1, 16, BlockScale::E4M3, 6.0 / 512.0);
                }
                HistoryRows::Shared { .. } => {}
            }
            if let HistoryReads::Selected(Selection::Own(indexer)) = &mut source.reads {
                indexer.query.rounding = index;
                let IndexerKeys::Latent { rounding, .. } = &mut indexer.keys else {
                    unreachable!("full-and-reuse indexer keys are latent")
                };
                *rounding = index;
                indexer.top = 512;
            }
        }
    }
    definition.validate().unwrap();
    definition
}

/// The logits of the 8 rows after `request`'s `prompt`-row prompt.
/// `stepped`: the prompt in 16-row chunks, then one row a forward.
/// Otherwise the prompt's whole 16-row chunks, then one forward of its
/// remaining rows and the 8, each row's logits demanded.
fn decode_logits(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    request: RequestId,
    prompt: usize,
    stepped: bool,
    backend: seismic::BackendName,
) -> (Vec<Vec<f32>>, crate::StateBindings) {
    let token = |position: usize| (position * 37 + 11) as u32 % 256;
    domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
    domain.open_state(&mut bindings, request, None).unwrap();
    let chunked = if stepped { prompt } else { prompt / 16 * 16 };
    for start in (0..chunked).step_by(16) {
        let tokens = (start..chunked.min(start + 16)).map(token).collect::<Vec<_>>();
        bindings = run(domain, bindings, forward(request, WorkKind::Prefill, &tokens, start, false), false, backend).1;
    }
    let forwards = if stepped { (prompt..prompt + 8).map(|start| start..start + 1).collect() } else { vec![chunked..prompt + 8] };
    let mut logits = Vec::new();
    for rows in forwards {
        let tokens = rows.clone().map(token).collect::<Vec<_>>();
        // Every row's logits; no selection.
        let operation = scored(forward(request, WorkKind::Decode, &tokens, rows.start, false));
        let (outcome, returned) = run(domain, bindings, operation, false, backend);
        bindings = returned;
        let Outcome::Forward { rows: results } = outcome else {
            panic!("{backend:?}: a forward has a forward outcome: {outcome:?}");
        };
        // The scored rows: those from the prompt's end.
        let kept = results.iter().zip(rows).filter(|(_, position)| *position >= prompt);
        logits.extend(kept.map(|(row, _)| {
            row.logits
                .as_ref()
                .unwrap_or_else(|| panic!("{backend:?}: a row demanding logits has them"))
                .read_to_host()
                .unwrap()
        }));
    }
    (logits, bindings)
}

/// KL(p || q) of the softmax distributions of logits `p` and `q`.
fn kl_of(p: &[f32], q: &[f32]) -> f64 {
    let log_softmax = |logits: &[f32]| {
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let total = logits.iter().map(|&value| (value as f64 - top).exp()).sum::<f64>().ln();
        logits.iter().map(|&value| value as f64 - top - total).collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(p), log_softmax(q));
    p.iter().zip(&q).map(|(p, q)| p.exp() * (p - q)).sum()
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
    bindings: crate::StateBindings,
    operation: Operation,
    backend: seismic::BackendName,
) -> (Outcome, crate::StateBindings, bool) {
    let (mut outcomes, bindings, claimed) = steps(domain, bindings, vec![operation], backend);
    (outcomes.pop().expect("one operation"), bindings, claimed)
}

/// One launch of `operations`, one request's each, whichever way it
/// reserves: their outcomes in order, and whether it claimed the queued
/// step.
fn steps(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operations: Vec<Operation>,
    backend: seismic::BackendName,
) -> (Vec<Outcome>, crate::StateBindings, bool) {
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
    let (pending, bindings) = domain
        .finish_target(flight)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    assert_eq!(pending.len(), operations.len(), "{backend:?}: an outcome per operation");
    let mut outcomes = Vec::new();
    for (pending, operation) in pending.into_iter().zip(&operations) {
        let outcome = pending.outcome().clone();
        assert!(
            !matches!(outcome, Outcome::WeightReadFailed(_)),
            "{backend:?}: {outcome:?}"
        );
        domain
            .reconcile(pending, PhysicalDecision { accepted_rows: operation.row_count() })
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        outcomes.push(outcome);
    }
    (outcomes, bindings, claimed)
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
fn load_with(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    limits: ResourceLimits,
    banked: Banked,
    method: PlannedMethod,
) -> (ExecutorDomain, crate::StateBindings, u64, Pricing) {
    // A hung step reports every owner and every thread's stack on SIGUSR1,
    // as the rig asks a job past its estimate.
    crate::diagnostics::install();
    let reserves = platform::MemoryReserves::of_machine(None);
    let selected = platform::select_device(
        catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &reserves,
    )
    .unwrap_or_else(|error| panic!("{backend:?}: an available device is selected: {error}"));
    let device = Rc::new(
        crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap())
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
            head: method.drafts(),
            vision: false,
        },
        ExecutionPath::Native,
        method,
        KvCodec::Dense,
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
        .finalize_readout(definition, programs.readout_choices())
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
    let ledgers = crate::domain::fixtures::fixture_ledgers(&device, &execution, &seismic::Devices::discover(&[]).unwrap());
    let resources = ResourceAllocator::allocate(
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
    let mut residency = ResidencyStore::for_target(
        device.clone(),
        programs.clone(),
        (*execution).clone(),
        identity,
        resources.releases(),
        &ledgers,
        &mut weights,
    )
    .unwrap();
    let target = residency
        .load_target(definition, package, Box::new(|_, _| ()))
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
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
    let target_state = plan.allocate_target_state(device.clone()).unwrap();
    let head_state = plan.head_state().map(|state| state.allocate(device.clone()).unwrap());
    // As the engine's load records them: the scratch and target weights it
    // allocated, and the constants its domain binds.
    let allocated = execution
        .load()
        .target_weight_demand()
        .unwrap()
        .checked_plus(crate::memory::ClaimDemand { device: plan.bytes().scratch, ..Default::default() })
        .unwrap();
    let recorded = crate::domain::fixtures::fixture_recorded(&device, resources.ledgers(), allocated);
    let constants = crate::memory::ClaimDemand {
        device: programs.target_graphs().unwrap().binding_constant_bytes().unwrap(),
        ..Default::default()
    };
    let constants = resources.ledgers().claim(&device, crate::memory::ledger::CommittedKind::Load, constants).unwrap();
    let reader = crate::DeviceReader::open(
        Rc::new(seismic::Devices::discover(&[]).unwrap()),
        reserves,
        device,
    )
    .unwrap();
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
    )
    .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    // As a load ends: priced on the costs it waited for, its memory read as
    // the owner reads it before its first round.
    let pricing = Pricing { definition: definition.clone(), resident, planned, costs: crate::assessment::synthetic::costs() };
    let pending = "not measured by the test";
    domain
        .measure_load(pricing.measures(crate::assessment::Priced::Pending(pending.into()), crate::assessment::Priced::Pending(pending.into())), &mut bindings)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    domain.observe_memory().unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    (domain, bindings, pass_rows, pricing)
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
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits, Banked::Half);
        let (unstarved, starved) = (RequestId(1), RequestId(2));
        for request in [unstarved, starved] {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_state(&mut bindings, request, None).unwrap();
        }
        let operation = |request| scored(forward(request, WorkKind::Prefill, &prompt, 0, true));
        let (reference, returned) = run(&mut domain, bindings, operation(unstarved), false, backend);
        let (pending, returned) = starved_step(&mut domain, returned, operation(starved), PATIENCE_BOUND, backend);
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

/// The longest a starved test host waits for a wait of the device to
/// outlast its still pulse: its patience is PULSE_SPINS spins, a spin's
/// time the device's own (70 ns on an idle M4 Pro; across a discrete
/// device's bus no 3090's or 4070's wait expired within 1.2 s), so the
/// starvation lasts until the device records the expiry (`Pulse::starve`),
/// and a device none of whose waits expires within this fails the test
/// instead of passing it uninterrupted.
const PATIENCE_BOUND: std::time::Duration = std::time::Duration::from_secs(30);

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
        let (mut domain, mut bindings, _, _) = open();
        // The unstarved sequence: its prompt one chunk at a time, then two
        // decode steps.
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let (reference, returned) = prompt_then_decode(&mut domain, bindings, request, chunks, rows, backend);
        drop(returned);
        drop(domain);

        let (mut domain, mut bindings, _, _) = open();
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        // The chain opens with its first chunk, which launches the rest,
        // and is starved.
        domain.plan_successors((1..chunks).map(|next| prompt_chunk(request, next, chunks, rows)).collect());
        let (pending, returned) =
            starved_step(&mut domain, bindings, prompt_chunk(request, 0, chunks, rows), PATIENCE_BOUND, backend);
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

/// A streamed load dropped with a pipelined successor still running tears
/// down: the successor, a prompt chunk of a grouped class on a partial bank,
/// whose first round every streamed layer waits on the host for, is waited
/// for and dropped while the watcher still answers it. (Regression: the
/// bank's watcher stopped serving first, beside a pulse the load's live
/// tables kept beating, so the successor's waits never ended and every
/// release behind it, residency's records and a growth's recommit, waited
/// for it forever.) A teardown past [`TEARDOWN_BOUND`] ends the process.
#[test]
fn a_streamed_load_dropped_with_a_running_successor_tears_down() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) =
        fixture_with("queued-teardown", qwen4exp_fixture_definition_of(3, 128), streamed_experts_manifest);
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _, _) =
            load(&catalog, &definition, &package, backend, limits(CHUNK_ROWS, true, 1, CHUNK_ROWS), Banked::Half);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        domain.plan_successors(vec![prompt_chunk(request, 1, 2, CHUNK_ROWS)]);
        let (_, returned) = run(&mut domain, bindings, prompt_chunk(request, 0, 2, CHUNK_ROWS), false, backend);
        assert_eq!(domain.queued_steps(), 1, "{backend:?}: the next chunk runs queued behind the first");
        let (dropped, watched) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = watched.recv_timeout(TEARDOWN_BOUND) {
                eprintln!("{backend:?}: the load's teardown hung past {TEARDOWN_BOUND:?} behind its queued successor");
                std::process::abort();
            }
        });
        drop(domain);
        drop(returned);
        dropped.send(()).unwrap();
        watchdog.join().unwrap();
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// The longest a load's teardown may take behind one queued fixture step
/// that its host answers: a few of the step's reads and launches, which take
/// milliseconds.
const TEARDOWN_BOUND: std::time::Duration = std::time::Duration::from_secs(60);

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
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits(rows, false, chunks, rows), Banked::Half);
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

/// The Sinkhorn fixture's streams, hidden width and numerics
/// (`sinkhorn_fixture_definition`).
const SINKHORN: (usize, usize, f32, u32) = (4, 128, 1e-6, 20);

/// The BF16 rounding stages a reordered F32 sum can tip on the deepest path
/// from the embedding to the features, per block (kernels `routed.seismic`,
/// `functions.seismic`):
/// 1. the routed input under its norm, published in A (`routed_select`,
///    986-1016: `normalized` 992, 995);
/// 2. the gate (and, beside it, the up) projection in A (decode
///    `routed_gate_up` 1139-1181: 1163-1177; prompt `routed_experts`
///    625-689: 634-669);
/// 3. `glu`'s activation in A (`functions.seismic:40`);
/// 4. `glu`'s product with up in A (`functions.seismic:48`);
/// 5. the down projection in A (decode `decode_down` 1224-1240: 1239;
///    prompt `routed_experts` 679).
///
/// None else counts. The slot sum is F32 (`decode_selected` 1280-1290,
/// `routed_scatter`); the Sinkhorn site, its collapse and the joins are F32
/// throughout (`streams.seismic` 104-171, 187-191; `functions.seismic`
/// 60-76); the attention adds an exact zero (its output projection is zero
/// in this fixture, so no tip reaches past it); the embedding's BF16 import
/// has no arithmetic before it, so nothing reorders it (`resident_weights.rs`
/// 301). Two blocks make 10 stages.
const STAGES: u8 = 10;

/// The most a device feature may be from the reference, in units of 2^-8 of
/// the reference row's RMS. The reference rounds where the device rounds
/// ([`STAGES`], the resident BF16 weights), so what differs is F32 order
/// (the natives' reductions, fma or not, `exp`, `rsqrt`): a few F32 ulps,
/// which tip a BF16 rounding across a midpoint by at most one ulp. Each
/// stage's tip is counted as one BF16 ulp at row scale: a BF16 ulp is at
/// most 2^-7 of its value, so at most 2^-7 · RMS = 2 units. The features'
/// own rounding (`readout_features_rows`, `readout.seismic:14-18`) adds one
/// ulp of the feature itself: 2 units for a feature no larger than the
/// RMS, and the check takes each feature's own ulp. So 2 + 2 · 10 = 22 at
/// row scale.
const BOUND: f32 = 2.0 + TIPS;

/// [`STAGES`]' tips at row scale, in [`BOUND`]'s units.
const TIPS: f32 = 2.0 * STAGES as f32;

/// The forms of a Sinkhorn fixture: whether each site mixes by its own pre
/// (GLM-5.3) or by the previous site's (V4.1), and whether the exit takes
/// the streams' mean or collapses them by the last site's pre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SinkhornForm {
    own: bool,
    mean: bool,
}

/// The host reference of the Sinkhorn fixture as written, and the faults it
/// tells apart from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SinkhornHost {
    Faithful,
    /// Every site mixes by the one-hot pre.
    FirstPre,
    /// The routed branch joins without the site's post.
    UnscaledJoin,
    /// Every site mixes by the other site's pre (its own for the previous
    /// site's, and the reverse).
    OtherInput,
    /// The exit takes the other weights (the last pre for the mean, and the
    /// reverse).
    OtherExit,
}

impl SinkhornHost {
    const FAULTS: [Self; 4] = [Self::FirstPre, Self::UnscaledJoin, Self::OtherInput, Self::OtherExit];
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

/// `dot` as the projections accumulate it: one F32 fma a term, in order.
fn fused_dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).fold(0.0, |sum, (left, right)| left.mul_add(*right, sum))
}

/// `value` published in BF16: rounded to nearest, ties to even.
fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// One BF16 ulp at `value`, a normal number's.
fn bf16_ulp(value: f32) -> f32 {
    f32::from_bits(value.abs().to_bits() & 0x7f80_0000) * 2f32.powi(-7)
}

/// `values` under an RMS norm of unit weight.
fn rms_normalized(values: &[f32], epsilon: f32) -> Vec<f32> {
    let inverse = 1.0 / (dot(values, values) / values.len() as f32 + epsilon).sqrt();
    values.iter().map(|value| value * inverse).collect()
}

/// A Sinkhorn site over streams `x` `[S·H]`, in F32 by its formulas (pre
/// `sigmoid + ε`, post `2·sigmoid`, the comb a row softmax plus ε, then
/// alternating column and row normalization): the site's pre, post and comb
/// `[S, S]` (source major).
fn sinkhorn_site(x: &[f32], weights: [&[f32]; 3]) -> [Vec<f32>; 3] {
    let (s, h, epsilon, iterations) = SINKHORN;
    let [function, base, scale] = weights;
    let inverse = 1.0 / (dot(x, x) / (s * h) as f32 + epsilon).sqrt();
    let mix = |at: usize, scale: f32| dot(&function[at * s * h..(at + 1) * s * h], x) * inverse * scale + base[at];
    let pre = (0..s).map(|at| sigmoid(mix(at, scale[0])) + epsilon).collect();
    let post = (0..s).map(|at| 2.0 * sigmoid(mix(s + at, scale[1]))).collect();
    let mut comb = (0..s * s).map(|at| mix(2 * s + at, scale[2])).collect::<Vec<_>>();
    for row in comb.chunks_mut(s) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.iter_mut().for_each(|value| *value = (*value - maximum).exp());
        let total = row.iter().sum::<f32>();
        row.iter_mut().for_each(|value| *value = *value / total + epsilon);
    }
    let columns = |comb: &mut Vec<f32>| {
        for target in 0..s {
            let sum = (0..s).map(|source| comb[source * s + target]).sum::<f32>();
            (0..s).for_each(|source| comb[source * s + target] /= sum + epsilon);
        }
    };
    for sweep in 0..iterations {
        if sweep > 0 {
            for row in comb.chunks_mut(s) {
                let sum = row.iter().sum::<f32>();
                row.iter_mut().for_each(|value| *value /= sum + epsilon);
            }
        }
        columns(&mut comb);
    }
    [pre, post, comb]
}

/// Streams `x` `[S·H]` collapsed by `weights` `[S]` in F32, in stream order.
fn collapse_streams(x: &[f32], weights: &[f32]) -> Vec<f32> {
    let (s, h, ..) = SINKHORN;
    (0..h).map(|column| (0..s).map(|stream| weights[stream] * x[stream * h + column]).sum()).collect()
}

/// The Sinkhorn fixture's readout features of `token`'s row under `form`,
/// from the resident weights `tensor` reads, rounded where the device
/// rounds ([`BOUND`]): per block its attention (whose zero output
/// projection adds nothing, so its join is the comb's mix) and its routed
/// branch over every expert, each reading its site's input (the streams
/// mixed by the site's own pre or the previous site's), joined by post and
/// comb in F32; then the streams collapsed by the last pre or into their
/// mean under the unit output norm, published in BF16.
fn sinkhorn_reference(
    tensor: &dyn Fn(&str) -> Rc<Vec<f32>>,
    blocks: usize,
    token: usize,
    form: SinkhornForm,
    host: SinkhornHost,
) -> Vec<f32> {
    let (s, h, epsilon, _) = SINKHORN;
    let own = form.own != (host == SinkhornHost::OtherInput);
    let mean = form.mean != (host == SinkhornHost::OtherExit);
    let embedding = tensor("embedding");
    let mut x = embedding[token * h..(token + 1) * h].repeat(s);
    let first = (0..s).map(|stream| f32::from(u8::from(stream == 0))).collect::<Vec<_>>();
    let mut previous = first.clone();
    for block in 0..blocks {
        for sublayer in ["attn", "ffn"] {
            let name = |part: &str| format!("blk.{block}.{sublayer}_{part}");
            let [function, base, scale] = ["hc_fn", "hc_base", "hc_scale"].map(|part| tensor(&name(part)));
            let [pre, post, comb] = sinkhorn_site(&x, [&function, &base, &scale]);
            let mixing = match host {
                SinkhornHost::FirstPre => &first,
                _ if own => &pre,
                _ => &previous,
            };
            let input = collapse_streams(&x, mixing);
            previous = pre;
            let y = match sublayer {
                "attn" => vec![0.0; h],
                _ => routed_reference(tensor, block, &input),
            };
            let post = if host == SinkhornHost::UnscaledJoin { vec![1.0; s] } else { post };
            x = (0..s * h)
                .map(|at| {
                    let (stream, column) = (at / h, at % h);
                    (0..s).map(|from| comb[from * s + stream] * x[from * h + column]).sum::<f32>() + y[column] * post[stream]
                })
                .collect();
        }
    }
    let weights = if mean { vec![1.0 / s as f32; s] } else { previous };
    rms_normalized(&collapse_streams(&x, &weights), epsilon).into_iter().map(bf16).collect()
}

/// Block `block`'s routed branch over its sublayer input `input`, as the
/// general routed entries' bodies round it (`routed.seismic`): the input
/// under its unit norm in BF16 (`routed_select`); every expert selected,
/// in slot order of ascending score (`rank_experts`), weighted by its
/// softmax score over their sum; per expert the BF16 gate and up
/// projections, `glu`'s BF16 activation and product, the BF16 down
/// projection; their F32 fma sum in slot order onto zeros (`routed_down`,
/// `routed_scatter`).
fn routed_reference(tensor: &dyn Fn(&str) -> Rc<Vec<f32>>, block: usize, input: &[f32]) -> Vec<f32> {
    let (h, epsilon) = (input.len(), SINKHORN.2);
    let name = |part: &str| format!("blk.{block}.{part}");
    let (router, gate, up, down) =
        (tensor(&name("ffn_gate_inp")), tensor(&name("ffn_gate_exps")), tensor(&name("ffn_up_exps")), tensor(&name("ffn_down_exps")));
    let normalized = rms_normalized(input, epsilon).into_iter().map(bf16).collect::<Vec<_>>();
    let experts = router.len() / h;
    let features = gate.len() / experts / h;
    let logits = router.chunks(h).map(|row| fused_dot(&normalized, row)).collect::<Vec<_>>();
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = logits.iter().map(|logit| (logit - maximum).exp()).collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f32>();
    let scores = exponentials.iter().map(|exponential| exponential / total).collect::<Vec<_>>();
    let mut slots = (0..experts).collect::<Vec<_>>();
    slots.sort_by(|&one, &other| scores[one].total_cmp(&scores[other]).then(one.cmp(&other)));
    let denominator = slots.iter().map(|&expert| scores[expert]).sum::<f32>();
    let mut y = vec![0.0f32; h];
    for expert in slots {
        let rows = |stack: &[f32], at: usize| stack[(expert * features + at) * h..(expert * features + at + 1) * h].to_vec();
        let product = (0..features)
            .map(|at| {
                let gated = bf16(fused_dot(&normalized, &rows(&gate, at)));
                let up = bf16(fused_dot(&normalized, &rows(&up, at)));
                bf16(bf16(gated * sigmoid(gated)) * up)
            })
            .collect::<Vec<_>>();
        let weight = scores[expert] / denominator;
        for (column, value) in y.iter_mut().enumerate() {
            let row = &down[(expert * h + column) * features..(expert * h + column + 1) * features];
            *value = weight.mul_add(bf16(fused_dot(&product, row)), *value);
        }
    }
    y
}

/// On CUDA, Metal and the CPU, the Sinkhorn fixture (two blocks of attention
/// and a general routed feed-forward over four Sinkhorn streams, V4.1's
/// form: each site mixing by the previous site's pre, the exit by the last)
/// equals its host reference (`sinkhorn_reference`, rounding where the
/// device rounds) at every row of a prompt
/// group, whose routed layers split into fronts and backs around the
/// group's pass, and at decode steps of the same tokens; and a chain of its
/// prompt chunks equals its chunks one step at a time. The output
/// projections are zero
/// (rows stay apart), the norms one and the site scales 16. The streams
/// enter as copies and differ only by their joins' `post · y`, so a fault in
/// whose pre mixes them or in the exit's weights shows only as far as the
/// sites' pre and post differ across streams: at scale 16 a site's mixes
/// (about ±0.8 each) put its pre and post near 0 or 1 (2 for post) and its
/// comb near a permutation, so the streams differ by up to `2·y` and two
/// sites' pre select different streams. Every row is within [`BOUND`] of the
/// reference, and each fault the reference can play (`SinkhornHost`) lies
/// past twice it at some prompt row and at some decode row.
#[test]
fn a_sinkhorn_decoder_matches_its_host_reference() {
    assert_sinkhorn_decoder_matches_its_host_reference(SinkhornForm { own: false, mean: false });
}

/// [`a_sinkhorn_decoder_matches_its_host_reference`] in GLM-5.3's form: each
/// site mixing by its own pre, carrying none, and the exit the streams'
/// mean.
#[test]
fn an_own_site_mean_exit_sinkhorn_decoder_matches_its_host_reference() {
    assert_sinkhorn_decoder_matches_its_host_reference(SinkhornForm { own: true, mean: true });
}

/// A Kimi Delta Attention layer (GLM-5.3-Flash slice A's block) prepares
/// and qualifies on every backend this host has: qualification reads the
/// layer's gate and decay ranks from its binding, whichever statics the
/// backend's `attention_project` keeps (the CPU keeps neither).
#[test]
fn a_kimi_delta_layer_qualifies_on_every_backend() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (_directory, definition, package) = fixture_valued(
        "kimi-delta-qualifies",
        crate::planning::fixtures::glm_slice_fixture_definition(),
        crate::planning::fixtures::fixture_manifest,
        &|_| None,
    );
    let backends = [
        seismic::BackendName::Cuda,
        seismic::BackendName::Metal,
        seismic::BackendName::Vulkan,
        seismic::BackendName::Cpu,
    ];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        load(&catalog, &definition, &package, backend, limits(2, false, 1, 0), Banked::Resident);
    }
}

fn assert_sinkhorn_decoder_matches_its_host_reference(form: SinkhornForm) {
    use magnitude_family_contracts::{ExitSite, SinkhornInput};
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let constant = |name: &str| match name {
        "o" => Some(0.0),
        _ if name.ends_with("_hc_scale") => Some(16.0),
        _ if name.ends_with("norm") => Some(1.0),
        _ => None,
    };
    let blocks = 2;
    let (directory, definition, package) = fixture_valued(
        &format!("sinkhorn-steps-own-{}-mean-{}", form.own, form.mean),
        crate::planning::fixtures::sinkhorn_fixture_definition_of(
            blocks as u64,
            if form.own { SinkhornInput::OwnSite } else { SinkhornInput::PreviousSite },
            if form.mean { ExitSite::Mean } else { ExitSite::PreviousSite },
        ),
        crate::planning::fixtures::fixture_manifest,
        &constant,
    );
    let tensors = std::cell::RefCell::new(std::collections::HashMap::<String, Rc<Vec<f32>>>::new());
    let tensor = |name: &str| -> Rc<Vec<f32>> {
        tensors
            .borrow_mut()
            .entry(name.to_owned())
            .or_insert_with(|| {
                let bytes = package.target().tensor(name).unwrap().read().unwrap();
                // Resident as BF16, apart from the sites' F32 roles.
                let resident = if name.contains("_hc_") { |value: f32| value } else { bf16 };
                Rc::new(bytes.chunks_exact(2).map(|bits| resident(half_value(u16::from_le_bytes([bits[0], bits[1]])))).collect())
            })
            .clone()
    };
    let references = std::cell::RefCell::new(std::collections::HashMap::new());
    let check = |features: &Option<Vec<u32>>, row: usize, token: usize, label: &str| {
        // Each row's features read back as its step's BF16 feature rows, two
        // values to each word `step_bits` reads.
        let words = features.as_ref().unwrap_or_else(|| panic!("{label}: features"));
        let h = SINKHORN.1;
        let device = words
            .iter()
            .flat_map(|bits| [bits << 16, bits & 0xffff_0000].map(f32::from_bits))
            .skip(row * h)
            .take(h)
            .collect::<Vec<_>>();
        assert_eq!(device.len(), h, "{label}: the step's BF16 features hold the row's");
        let reference = |host| {
            references.borrow_mut().entry((token, host)).or_insert_with(|| sinkhorn_reference(&tensor, blocks, token, form, host)).clone()
        };
        // In units of 2^-8 of the faithful row's RMS.
        let faithful = reference(SinkhornHost::Faithful);
        let unit = 2f32.powi(-8) * (dot(&faithful, &faithful) / faithful.len() as f32).sqrt();
        let error = |host: Vec<f32>| device.iter().zip(&host).map(|(device, host)| (device - host).abs() / unit).fold(0.0, f32::max);
        for (device, host) in device.iter().zip(&faithful) {
            // The stages' tips at row scale, and the feature's own rounding.
            let allowed = TIPS + bf16_ulp(*host) / unit;
            let units = (device - host).abs() / unit;
            assert!(units <= allowed, "{label}: {device} is {units} units from {host}, past {allowed}");
        }
        let mut errors = [error(faithful.clone()); 5];
        for (at, fault) in SinkhornHost::FAULTS.into_iter().enumerate() {
            errors[at + 1] = error(reference(fault));
        }
        errors
    };
    let backends = [seismic::BackendName::Cuda, seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let prompts = Prompts { requests: 1, chunks: 2, rows: CHUNK_ROWS };
        let (alone, _) = chained_prompts(&catalog, &definition, &package, backend, prompts, 1, Banked::Resident);
        let (chained, _) = chained_prompts(&catalog, &definition, &package, backend, prompts, 2, Banked::Resident);
        assert_steps_equal(&alone, &chained, prompts, &format!("{backend:?} chain of 2"));
        // The most units any prompt row (the chain's rows, being equal) is
        // from the reference, and from each fault.
        let worse = |worst: [f32; 5], units: [f32; 5]| std::array::from_fn(|at| worst[at].max(units[at]));
        // The prompt's token at `position` (`prompt_chunk`, request 1).
        let token = |position: usize| (position * 37 + 11 + 5) % 256;
        let mut prompt = [0.0f32; 5];
        for (chunk, (bits, _)) in alone[..prompts.chunks].iter().enumerate() {
            for (row, (features, ..)) in bits.iter().enumerate() {
                let position = chunk * CHUNK_ROWS + row;
                prompt = worse(prompt, check(features, row, token(position), &format!("{backend:?} prompt row {position}")));
            }
        }
        // The same tokens decoded one step at a time, which carries every
        // site's pre across the decode graphs: each fault is judged over all
        // of them, as over the prompt's rows (a row's divergence, and so a
        // fault's reach, follows its token; the zero output projections
        // keep rows apart).
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits(2, false, 1, 0), Banked::Resident);
        let request = RequestId(1);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_state(&mut bindings, request, None).unwrap();
        let mut decoded = [0.0f32; 5];
        for position in 0..prompts.chunks * CHUNK_ROWS {
            let mut decode = forward(request, WorkKind::Decode, &[token(position) as u32], position, false);
            if let Operation::Forward { demand, .. } = &mut decode {
                *demand |= crate::batching::Demand::FEATURES;
            }
            let (outcome, returned) = run(&mut domain, bindings, decode, false, backend);
            bindings = returned;
            let label = format!("{backend:?} decode row {position}");
            decoded = worse(decoded, check(&step_bits(&outcome)[0].0, 0, token(position), &label));
        }
        drop(bindings);
        for (mode, [within, faults @ ..]) in [("prompt", prompt), ("decode", decoded)] {
            eprintln!("{backend:?} {form:?} {mode}: within {within} units (bound {BOUND}); faults {:?} at {faults:?}", SinkhornHost::FAULTS);
        }
        for (mode, [_, faults @ ..]) in [("prompt", prompt), ("decode", decoded)] {
            assert!(
                faults.iter().all(|units| *units > 2.0 * BOUND),
                "{backend:?} {form:?} {mode}: a fault within twice the bound: {faults:?}"
            );
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}
