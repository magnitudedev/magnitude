//! A decoder with an n-gram input steps on the device with real
//! submissions: its weights imported from a written GGUF, its graphs
//! prepared, sealed and bound as a load binds them, and target steps of the
//! host's tokens and of the device's selection run through its n-gram block.
//! A chain of its prompt chunks computes exactly what the chunks compute one
//! step at a time.

use crate::planning::fixtures::{
    RoutedShape, expert_stacks, qwen4exp_fixture_definition_of, relaid_manifest, routed_fixture_definition,
    stored_table_manifest, streamed_experts_manifest,
};
use crate::{
    AttestedPrograms, ComponentSelection, ExecutionPath, ExecutionPlanner, ExecutorDomain,
    Operation, Outcome, PhysicalDecision, PlannedMethod, ReservedResources, ResidencyStore,
    ResourceCapacity, ResourceDomainId, ResourceLimits, ResourcePlanner,
    RequestId, SelectSpec, Sampling, Shaping, TokenId, WorkKind, domain::TargetGraphReservation,
    platform,
    test_support::available,
};
use crate::operators::attention::tests::selected::{self, CompressedForm};
use magnitude_artifacts::{Package, gguf::Encoding};
use magnitude_family_contracts::{ModelDefinition, PreparedModelInput};
use magnitude_state::KvCodec;
use crate::domain::fixtures::loaded::{Banked, LoadRequest, Machine, Pricing, fixture, fixture_of, fixture_stored, fixture_valued, fixture_with, limits, load, load_read, load_with};
use std::rc::Rc;
use std::sync::Arc;

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
        front: Vec::new(),
        chain: None,
    }
}

/// A native load's arena claim holds exactly what its graph scratch and its
/// target graphs' bound constants (the floor's group buffers and the held
/// pass's buffers among them) charged, with no class beyond the floor's at
/// one place: the constants among it, and no byte held that none of them
/// allocated.
#[test]
fn a_native_loads_arena_claim_is_its_scratch_and_bound_constants() {
    let (directory, definition, package) = fixture("arena-claim");
    {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        for backend in [seismic::BackendName::Metal, seismic::BackendName::Cpu].into_iter().filter(|&backend| available(&catalog, backend)) {
            let (domain, _bindings, _, _) = load(&catalog, &definition, &package, backend, limits(2, true, 1, 0), Banked::Resident);
            let (claim, allocated, constants) = domain.arena_terms();
            assert!(constants > 0, "{backend:?}: the fixture binds constants");
            assert!(claim >= constants, "{backend:?}: the arena claim of {claim} bytes holds its {constants} constant bytes");
            assert_eq!(claim, allocated, "{backend:?}: the arena claim is what its allocations charged");
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On Metal and on the CPU: a prompt chunk and a decode step of the host's
/// tokens, then the decode step of the token the device selected, which
/// claims the queued lookahead and reads its token rows from the selection.
#[test]
fn an_ngram_decoder_steps_on_host_and_selected_tokens() {
    let (directory, definition, package) = fixture("ngram-steps");
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _, _) =
            load(&catalog, &definition, &package, backend, limits(2, true, 1, 0), Banked::Resident);
        let request = RequestId(1);
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
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
        let Outcome::Forward { rows, .. } = outcome else {
            panic!("{backend:?}: a decode step has a forward outcome");
        };
        let crate::Selected::Token(selected) = rows
            .last()
            .and_then(|row| row.selected)
            .unwrap_or_else(|| panic!("{backend:?}: the decode step selects a token"))
        else {
            panic!("{backend:?}: the decode step selects a token");
        };
        // The device's selection is the next step's token: it claims the
        // lookahead queued behind the decode step.
        let (_, returned) = run(
            &mut domain,
            returned,
            forward(request, WorkKind::Decode, &[selected.0], 3, true),
            true,
            backend,
        );
        drop(returned);
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Logits of one history from launches of different row classes agree within
/// float rounding: the classes sum in their own orders, which precision
/// admission, not bit identity, governs.
const ROW_CLASS_KL: f64 = 1e-4;

/// A verification row scores its history as plain decode does, within
/// [`ROW_CLASS_KL`]. One prompt, then its continuation decoded a token a step;
/// then, from the same prompt, each verification width up to the decode row
/// bound feeds that continuation as one forward, and each of its rows'
/// logits are the decode step's at its position. The prompt (37 tokens) puts
/// the rows' histories across attention's 32-key tiles and partitions, so
/// every width splits a row's keys between history and the pass's fresh rows
/// at another place. Dense and K8/V4 history, on every backend present, of a
/// plain load, an MTP load and a load streaming its routed experts over a
/// bank of about half the slots: the rows a drafted round verifies. A DFlash
/// load needs a package of two components, which this harness's one-file package cannot hold; the
/// serving pair compares it end to end.
#[test]
fn a_verification_row_scores_its_history_as_plain_decode_does() {
    const PROMPT: usize = 37;
    const ROWS: usize = magnitude_kernels::constants::VERIFY_ROWS as usize;
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let proposals = (ROWS - 1) as u8;
    let loads = [
        ("plain", fixture("verify-rows"), PlannedMethod::Plain, Banked::Resident),
        (
            "mtp",
            fixture_with("verify-rows-mtp", mtp, crate::planning::fixtures::fixture_manifest),
            PlannedMethod::Mtp {
                greedy_proposals: proposals,
                sampled_proposals: proposals,
                widths: crate::DraftWidths::Adaptive,
            },
            Banked::Resident,
        ),
        // Routed layers over a bank of about half the slots where the backend
        // streams (resident on the CPU): ten layers of 256 experts span more
        // than one CUDA pool block, so the half-slot cap forms a partial bank.
        (
            "streamed",
            fixture_with("verify-rows-streamed", qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest),
            PlannedMethod::Plain,
            Banked::Half,
        ),
    ];
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let token = |position: usize| (position * 37 + 11) as u32 % 256;
    // Every row selects greedily, as a served round does, and exports its
    // logits.
    let scoring = |request: RequestId, kind: WorkKind, tokens: &[u32], position: usize| {
        let Operation::Forward { request, kind, tokens, position, conditioning, committed, prime, .. } =
            forward(request, kind, tokens, position, false)
        else {
            unreachable!("a forward")
        };
        Operation::Forward {
            select: (0..tokens.len())
                .map(|row| SelectSpec {
                    sampling: Sampling::Greedy,
                    seed: 0,
                    position: position + row,
                    domain: 0,
                    mask: None,
                    shaping: Shaping::default(),
                    history: None,
                })
                .collect(),
            demand: crate::batching::Demand::SELECT | crate::batching::Demand::LOGITS,
            request,
            kind,
            tokens,
            position,
            conditioning,
            committed,
            prime,
            front: Vec::new(),
            chain: None,
        }
    };
    let scores = |outcome: &Outcome| -> Vec<Vec<f32>> {
        let Outcome::Forward { rows, .. } = outcome else {
            panic!("a forward has a forward outcome: {outcome:?}");
        };
        rows.iter()
            .map(|row| row.logits.as_ref().expect("every row exports its logits").read_to_host().unwrap())
            .collect()
    };
    let backends = [
        seismic::BackendName::Cpu,
        seismic::BackendName::Metal,
        seismic::BackendName::Cuda,
        seismic::BackendName::Vulkan,
    ];
    let mut failures = Vec::new();
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        for (name, (_, definition, package), method, banked) in &loads {
            let banked = if STREAMING_BACKENDS.contains(&backend) { *banked } else { Banked::Resident };
            for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
                let (mut domain, mut bindings, _, _) = load_read(&LoadRequest {
                    catalog: &catalog,
                    definition,
                    package,
                    backend,
                    limits: ResourceLimits { max_selected_rows: ROWS, ..limits(ROWS, false, 1, ROWS) },
                    banked,
                    method: *method,
                    codec,
                    machine: Machine::Live,
                    pass: None,
                });
                // `request`'s prompt, in launches of up to ROWS rows.
                let prompted = |domain: &mut ExecutorDomain, mut bindings: crate::StateBindings, request: RequestId| {
                    domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
                    domain.open_served(&mut bindings, request, None).unwrap();
                    for start in (0..PROMPT).step_by(ROWS) {
                        let tokens = (start..PROMPT.min(start + ROWS)).map(token).collect::<Vec<_>>();
                        let (_, returned) = run(domain, bindings, forward(request, WorkKind::Prefill, &tokens, start, false), false, backend);
                        bindings = returned;
                    }
                    bindings
                };
                let plain = RequestId(1);
                bindings = prompted(&mut domain, bindings, plain);
                let mut decoded = Vec::new();
                for position in PROMPT..PROMPT + ROWS {
                    let (outcome, returned) =
                        run(&mut domain, bindings, scoring(plain, WorkKind::Decode, &[token(position)], position), false, backend);
                    bindings = returned;
                    decoded.extend(scores(&outcome));
                }
                domain.close(plain).unwrap_or_else(|error| panic!("{error}"));
                for width in 2..=ROWS {
                    let request = RequestId(1 + width as u64);
                    bindings = prompted(&mut domain, bindings, request);
                    let tokens = (PROMPT..PROMPT + width).map(token).collect::<Vec<_>>();
                    let (outcome, returned) =
                        run(&mut domain, bindings, scoring(request, WorkKind::Verify, &tokens, PROMPT), false, backend);
                    bindings = returned;
                    for (row, logits) in scores(&outcome).iter().enumerate() {
                        let kl = kl_of(&decoded[row], logits);
                        if !(kl <= ROW_CLASS_KL) {
                            failures.push(format!(
                                "{backend:?} {name} {codec:?}: width {width} row {row}: KL {kl:.3e} from plain decode's logits"
                            ));
                        }
                    }
                    domain.close(request).unwrap_or_else(|error| panic!("{error}"));
                }
            }
        }
    }
    for (_, (directory, _, _), _, _) in &loads {
        std::fs::remove_dir_all(directory).ok();
    }
    assert!(failures.is_empty(), "verification rows that do not score as plain decode:\n{}", failures.join("\n"));
}

/// On Metal and on the CPU, a load streaming its experts through a bank
/// (some layers' experts partly held, some held whole as the bank fills):
/// a prompt chunk, then two decode steps of the device's selected tokens,
/// each step's routing record written at every streamed layer the watcher
/// reads (its step completes, or the watcher's invariant fails it), and
/// the selections the resident load makes.
#[test]
fn a_streamed_decoder_steps_twice_through_its_bank() {
    let (directory, definition, package) = fixture_of("streamed-steps", streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let selections = |banked| {
            let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits(2, false, 1, 0), banked);
            let request = RequestId(1);
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_served(&mut bindings, request, None).unwrap();
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
        let streamed = selections(Banked::Half);
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
/// hits. Its streamed decode launches meet every residency
/// ([`Scores::assert_every_residency`]): a layer cold (every choice
/// absent, all computed after its tier wait), partly warm (its held
/// choices computed before the wait, its absent ones after it) and warm
/// (every choice held, all before it), each step bit for bit the resident
/// load's, whose arithmetic holds no bank: decode depends on nothing the
/// bank held.
#[test]
fn a_partial_banks_shared_graphs_read_each_layers_own_experts() {
    let definition = qwen4exp_fixture_definition_of(9, 256);
    banked_shared_graphs_score_as_resident("partial-bank-shared", definition, &STREAMING_BACKENDS, Banked::Half);
}

/// A partial bank of a general routed model (no shared expert: 128 experts
/// choosing 8), whose gate/up runs in two passes around the tier wait (the
/// up-only wait after its latent projection included), scores as the resident
/// load does, each streamed decode launch meeting every residency
/// ([`Scores::assert_every_residency`]).
#[test]
fn a_partial_banks_general_routed_layers_score_as_resident() {
    let mut definition = routed_fixture_definition(RoutedShape { layers: 12, experts: 128, selected: 8, hidden: 192, intermediate: 64 });
    // `routed_fixture_definition` defaults to 16K context for planning tests;
    // this test runs a 2-token prompt and 32 decode steps, so 128 matches
    // `fixture_definition` and keeps the bank's floor below the model's weight.
    definition.decoder.context_limit = 128;
    banked_shared_graphs_score_as_resident("partial-bank-general", definition, &STREAMING_BACKENDS, Banked::Half);
}

/// The fronting fixture written for `test`: the decoder of
/// `two_row_exact_rows_score_alike_across_guesses_and_banks` (four
/// attention blocks of 128 experts a layer after the gated delta block)
/// with an MTP head of its own history (`with_mtp_head`), its experts'
/// stacks stored as a bank streams them. A load of it serving
/// [`FRONTING`] with its experts in a bank fronts
/// (`ExecutorDomain::fronts`): its decode passes run front rows on the
/// chain the head launch behind each pass makes (`crate::TwoRowDecode`).
fn fronting_fixture(test: &str) -> (std::path::PathBuf, Rc<ModelDefinition>, Arc<Package>) {
    let definition = crate::planning::fixtures::with_mtp_head(qwen4exp_fixture_definition_of(4, 128));
    fixture_with(test, definition, streamed_experts_manifest)
}

/// The method a fronting load serves: the MTP head of its own history,
/// which on a load that fronts makes each pass's chain and never proposes.
const FRONTING: PlannedMethod =
    PlannedMethod::Mtp { greedy_proposals: 2, sampled_proposals: 2, widths: crate::DraftWidths::Adaptive };

/// [`fronting_fixture`]'s load on `backend` serving [`FRONTING`], its
/// experts `banked` (in a bank, so it fronts), `lookahead` or not,
/// exporting up to `logits` rows of logits: its launches and selections as
/// wide as a chained launch's rows ([`crate::CHAINED_ROWS`]), so a front
/// pass of any depth a chain reaches launches.
#[allow(clippy::too_many_arguments)]
fn load_fronting(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    lookahead: bool,
    logits: usize,
    banked: Banked,
) -> (Box<ExecutorDomain>, crate::StateBindings) {
    let limits = ResourceLimits { max_selected_rows: crate::CHAINED_ROWS, ..limits(crate::CHAINED_ROWS, lookahead, 1, logits) };
    let (domain, bindings, ..) = load_with(catalog, definition, package, backend, limits, banked, FRONTING);
    assert!(domain.fronts(), "{backend:?}: a banked load of a head of its own history fronts");
    (domain, bindings)
}

/// A greedy selection at `position`.
fn greedy(position: usize) -> SelectSpec {
    SelectSpec { sampling: Sampling::Greedy, seed: 0, position, domain: 0, mask: None, shaping: Shaping::default(), history: None }
}

/// The prompt `tokens` of a fronting load's `request` from its first row,
/// in one chunk that closes it, selecting the first token: its drafter
/// entry primes the head through it, every pair inside the chunk as the
/// first chunk's (`Priming::pairs_previous`), and the head launch behind
/// it enters its selection with its boundary feature right past that entry
/// (`ChainEntry`), so the head stands level with the target's rows.
fn fronting_prompt(request: RequestId, tokens: &[u32]) -> Operation {
    let tokens = tokens.iter().map(|&token| TokenId(token)).collect::<Vec<_>>();
    let prime = crate::Priming { boundary: true, tokens: tokens[1..].to_vec(), position: 0, draft_from: tokens.len() };
    Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        select: vec![greedy(tokens.len() - 1)],
        committed: tokens.len(),
        chain: Some(crate::ChainEntry { position: prime.position + prime.tokens.len() }),
        prime: Some(prime),
        tokens,
        position: 0,
        conditioning: None,
        demand: crate::batching::Demand::SELECT | crate::batching::Demand::FEATURES,
        front: Vec::new(),
    }
}

/// `request`'s front pass of `rows` from its `pending` token at
/// `position`, which enters the head right behind it at the same position
/// (its closing chunk left the head level with the target's rows), every
/// row selecting greedily: demanding the selection, the features its head
/// entry reads, and `demand`.
fn front_pass(request: RequestId, rows: &crate::Rows, pending: TokenId, position: usize, demand: crate::batching::Demand) -> Operation {
    Operation::Forward {
        request,
        kind: WorkKind::Decode,
        tokens: rows.tokens(pending),
        position,
        conditioning: None,
        demand: crate::batching::Demand::SELECT | crate::batching::Demand::FEATURES | demand,
        select: vec![greedy(position)],
        committed: 1,
        prime: None,
        front: rows
            .front_kinds()
            .into_iter()
            .enumerate()
            .map(|(depth, kind)| crate::FrontRow { select: greedy(position + 1 + depth), kind })
            .collect(),
        chain: Some(crate::ChainEntry { position }),
    }
}

/// A fronting load's prompt (`fronting_prompt`) run, its outcome's
/// selection and the chain its head launch made: the head's guess alone,
/// of its selection's successor.
fn run_fronting_prompt(
    domain: &mut ExecutorDomain,
    bindings: crate::StateBindings,
    request: RequestId,
    backend: seismic::BackendName,
) -> (TokenId, Vec<crate::Guess>, crate::StateBindings) {
    let (outcome, bindings) = run(domain, bindings, fronting_prompt(request, &[5, 1]), false, backend);
    let Outcome::Forward { rows, guesses } = outcome else { panic!("{backend:?}: a prompt's outcome") };
    let crate::Selected::Token(selected) = rows
        .last()
        .and_then(|row| row.selected)
        .expect("the closing chunk selects")
    else {
        panic!("{backend:?}: the closing chunk selects a token");
    };
    assert!(
        matches!(guesses.as_slice(), [crate::Guess { kind: crate::GuessKind::Head, .. }]),
        "{backend:?}: the closing chunk's chain is the head's guess alone: {guesses:?}"
    );
    (selected, guesses, bindings)
}

/// On every backend experts stream on, front passes score their exact rows
/// alike whatever their front rows guess and whatever the bank holds: a
/// partial bank (the load of
/// `a_partial_banks_shared_graphs_read_each_layers_own_experts`, under an
/// MTP head of its own history: [`fronting_fixture`]) decoding in chained
/// passes of one front row, which guesses the next token or not, of each
/// kind in turn, scores each pass's exact row bit for bit as a whole bank
/// (which scores as the resident load does) through the same passes on one
/// fixed guess. Each prompt primes the head and its closing chunk enters
/// it, and each pass enters it behind its step. The front row, which
/// computes only the experts its step holds, reaches neither its exact row
/// nor any later one, nor the head's committed entry.
/// The bank's record counts each pass with front rows that completed.
#[test]
fn two_row_exact_rows_score_alike_across_guesses_and_banks() {
    let (directory, definition, package) = fronting_fixture("two-row-exact");
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        // Each pass's exact row's bits and its selection, of the passes of
        // `banked` whose front row guesses as `guess` makes it of the pass's
        // index and its exact row's position.
        let steps = |banked: Banked, guess: &dyn Fn(usize, usize) -> crate::Guess| {
            // Two logits rows: a pass's front row's besides its exact row's.
            let (mut domain, mut bindings) = load_fronting(&catalog, &definition, &package, backend, false, 2, banked);
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_served(&mut bindings, request, None).unwrap();
            let (mut pending, _, mut returned) = run_fronting_prompt(&mut domain, bindings, request, backend);
            let mut steps = Vec::new();
            let mut selections = Vec::new();
            for (index, position) in (2..2 + DECODE_STEPS).enumerate() {
                let rows = crate::Rows::front(vec![guess(index, position)], 0);
                let operation = front_pass(request, &rows, pending, position, crate::batching::Demand::LOGITS);
                let (mut outcomes, next, claimed) = launched(&mut domain, returned, &[operation], backend);
                assert!(!claimed, "{backend:?} pass {index}: no pass is queued without lookahead");
                let outcome = outcomes.pop().expect("one pass");
                steps.push(step_bits(outcome.outcome()));
                pending = TokenId(selected_token(outcome.outcome(), backend));
                selections.push(pending);
                let decision = PhysicalDecision { accepted_rows: 1, ahead: crate::AheadVerdict::new(&rows, &[pending]) };
                domain.reconcile(outcome, decision).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
                returned = next;
            }
            if matches!(banked, Banked::Half) {
                let passes = domain.expert_costs().map(|costs| costs.front.passes);
                assert_eq!(passes, Some(DECODE_STEPS as u64), "{backend:?}: the passes that ran");
            }
            (steps, selections)
        };
        // One fixed guess.
        let (whole, truths) = steps(Banked::Whole, &|_, _| crate::Guess { token: TokenId(0), kind: crate::GuessKind::Chained });
        // Every other pass's front row guesses the token its exact row
        // selects (as the whole bank's did); the rest guess another. The
        // kinds take turns, every other pair of passes.
        let (banked, _) = steps(Banked::Half, &|index, position| crate::Guess {
            token: TokenId((truths[index].0 + (position % 2) as u32) % 256),
            kind: crate::GuessKind::ALL[position / 2 % crate::GuessKind::ALL.len()],
        });
        for (step, (whole, banked)) in whole.iter().zip(&banked).enumerate() {
            assert!(whole.last().is_some_and(|(_, logits, _)| logits.is_some()), "{backend:?} step {step}: the whole bank's pass scores its exact row");
            assert_eq!(banked, whole, "{backend:?} step {step}: the partial bank's pass's exact row is the whole bank's");
        }
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// Front passes pipeline, and emit the tokens the same passes do without
/// lookahead. With lookahead, each pass claims the pass queued behind the
/// one before where it is that pass's continuation: its exact token read
/// on the device from the selection behind it, its rows as deep as the
/// queue ran them. On [`fronting_fixture`]'s load, which measures no
/// expert lines, no regime is priced (`ExecutorDomain::two_row_pricing`),
/// so each continuation is queued with its exact row alone
/// (`at_chain_depths`): a pass of one row claims it, and a pass the host
/// builds deeper, on the chain the head launch behind the pass before made
/// (`guess_chain`), orphans it, running as submitted. Each pass, its rows
/// and their guesses, the token its exact row selects and the chain its
/// head launch makes are those of the same pass submitted without
/// lookahead, at the same history: a partial bank on every backend experts
/// stream on.
///
/// The chain behind each pass is the head's: its consistent front rows'
/// selections, chained, then the head's guess one past them; one deeper
/// than the pass's guessed rows at most, and the head's guess alone where
/// the pass's first guess fell or it guessed nothing. Each pass's verdict
/// is told by its step: the orphaned passes are discarded, no verdict
/// judging them, and every later verdict judges its own pass (a verdict
/// matched to another pass is a broken invariant, which fails the load).
/// So every guessed row whose shallower rows stood and whose truth a later
/// pass learned is judged, by its kind and depth, and those that stood are
/// counted as the passes saw them.
#[test]
fn pipelined_two_row_passes_claim_their_successors_and_emit_their_unpipelined_tokens() {
    /// The front rows each pass the host builds runs at most, by index: a
    /// pass of none claims the queued one.
    const DEPTHS: [usize; 12] = [0, 1, 0, 2, 2, 0, 0, 1, 2, 0, 1, 0];
    let (directory, definition, package) = fronting_fixture("two-row-pipelined");
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        // Each pass, `lookahead` or not: its rows, its exact row's token and
        // the chain made behind it.
        let passes = |lookahead: bool| {
            let (mut domain, mut bindings) = load_fronting(&catalog, &definition, &package, backend, lookahead, 0, Banked::Half);
            assert!(
                domain.two_row_pricing().unwrap_or_else(|error| panic!("{backend:?}: {error}")).is_none(),
                "{backend:?}: the fixture's load prices no regime"
            );
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_served(&mut bindings, request, None).unwrap();
            let (mut exact, chain, returned) = run_fronting_prompt(&mut domain, bindings, request, backend);
            bindings = returned;
            let mut two_row = crate::TwoRowDecode::default();
            two_row.guessed(chain);
            let mut passes = Vec::new();
            // The front rows' record of the load.
            let front = |domain: &ExecutorDomain| domain.expert_costs().expect("a banked load's record").front;
            let mut queued_passes = 0usize;
            let mut claimed_passes = 0usize;
            for (index, depth) in DEPTHS.into_iter().enumerate() {
                let rows = two_row.pass(crate::FrontGrant::UpTo(depth));
                let guesses = rows.guesses().to_vec();
                let before = front(&domain);
                let queued = domain.queued_steps();
                if lookahead && index > 0 {
                    // The bank's watcher may stage a policy update (e.g. the
                    // hint gate's first flip after timing) and break the
                    // lookahead pipeline for one boundary via
                    // `ElasticBank::wants_gap()` so `write_staged()` can update
                    // device-visible policy words while no step is in flight.
                    assert!(queued <= 1, "{backend:?} pass {index}: at most one pass queued behind the one before");
                    queued_passes += queued;
                } else {
                    assert_eq!(queued, 0, "{backend:?} pass {index}: no pass is queued");
                }
                let operation = front_pass(request, &rows, exact, 2 + index, crate::batching::Demand::NONE);
                let (mut pending, next, claimed) = launched(&mut domain, bindings, &[operation], backend);
                let pending = pending.pop().expect("one pass");
                assert_eq!(
                    claimed,
                    queued == 1 && guesses.is_empty(),
                    "{backend:?} pass {index}: whether it claims the queued pass"
                );
                claimed_passes += usize::from(claimed);
                let Outcome::Forward { rows: results, guesses: chain } = pending.outcome().clone() else {
                    panic!("{backend:?}: a pass's outcome")
                };
                assert_eq!(results.len(), 1, "{backend:?} pass {index}: its exact row");
                let crate::Selected::Token(token) = results[0].selected.expect("an exact row selects") else {
                    panic!("{backend:?} pass {index}: an exact row selects a token");
                };
                exact = token;
                let kinds = chain.iter().map(|guess| guess.kind).collect::<Vec<_>>();
                let (head, chained) = kinds.split_last().unwrap_or_else(|| panic!("{backend:?} pass {index}: the head guesses behind every pass"));
                assert!(
                    *head == crate::GuessKind::Head && chained.iter().all(|&kind| kind == crate::GuessKind::Chained),
                    "{backend:?} pass {index}: its chain's consistent selections, then the head's guess: {kinds:?}"
                );
                assert!(chain.len() <= guesses.len() + 1, "{backend:?} pass {index}: one deeper than its guesses at most");
                if guesses.first().is_none_or(|guess| guess.token != exact) {
                    assert_eq!(chain.len(), 1, "{backend:?} pass {index}: the head's guess alone where no guess stood");
                }
                let decision = PhysicalDecision { accepted_rows: 1, ahead: crate::AheadVerdict::new(&rows, &[exact]) };
                domain.reconcile(pending, decision).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
                // The orphaned queued pass ran its exact row alone.
                let after = front(&domain);
                assert_eq!(after.passes - before.passes, u64::from(!guesses.is_empty()), "{backend:?} pass {index}: the passes with front rows that completed");
                passes.push((rows, exact, chain.clone()));
                two_row.guessed(chain);
                bindings = next;
            }
            if lookahead {
                assert!(
                    queued_passes >= DEPTHS.len() - 3,
                    "{backend:?}: lookahead queued most passes ({queued_passes}/{})",
                    DEPTHS.len() - 1
                );
                assert!(claimed_passes > 0, "{backend:?}: at least one exact-only pass claimed its queued successor");
            }
            // The guesses the verdicts judged and those that stood, by kind
            // and by depth.
            let verdicts = |domain: &ExecutorDomain| {
                let front = front(domain);
                (front.judged, front.stood, front.judged_at, front.stood_at)
            };
            // Each guessed row is judged where every shallower row of its
            // pass stood and a pass learned its truth: the exact selection
            // of the pass `depth - 1` past its own.
            let (mut judged, mut stood, mut judged_at, mut stood_at) = verdicts(&domain);
            judged.fill(0);
            stood.fill(0);
            judged_at.fill(0);
            stood_at.fill(0);
            for (index, (rows, ..)) in passes.iter().enumerate() {
                for (depth, guess) in rows.guesses().iter().enumerate() {
                    let Some((_, truth, _)) = passes.get(index + depth) else { break };
                    let held = guess.token == *truth;
                    judged[guess.kind.index()] += 1;
                    stood[guess.kind.index()] += u64::from(held);
                    judged_at[depth] += 1;
                    stood_at[depth] += u64::from(held);
                    if !held {
                        break;
                    }
                }
            }
            // Judged on the service's thread.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while verdicts(&domain).0 != judged || verdicts(&domain).2 != judged_at {
                assert!(std::time::Instant::now() < deadline, "{backend:?}: the verdicts were never judged: {:?}", front(&domain));
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(verdicts(&domain), (judged, stood, judged_at, stood_at), "{backend:?}: each verdict judged its own pass");
            passes
        };
        assert_eq!(passes(true), passes(false), "{backend:?}: pipelined passes run and emit as the same passes unpipelined");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On each of `backends` this host has, `definition` (its attention blocks
/// sharing a graph) loaded resident and `banked` score [`scores`]'s steps
/// alike ([`Scores::assert_as_resident`]), each step's last row's logits and
/// selection; a partial bank's streamed launches having met every residency
/// ([`Scores::assert_every_residency`]).
fn banked_shared_graphs_score_as_resident(
    test: &str,
    definition: ModelDefinition,
    backends: &[seismic::BackendName],
    banked: Banked,
) {
    let (directory, definition, package) = fixture_with(test, definition, streamed_experts_manifest);
    let stored = stored_experts(&package, &definition);
    {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        for &backend in backends.iter().filter(|&&backend| available(&catalog, backend)) {
            let scores = scores(&catalog, &definition, &package, &stored, backend, banked, Start::AtOnce);
            if matches!(banked, Banked::Half) {
                scores.assert_every_residency(backend);
            }
            scores.assert_as_resident(backend, "the banked load's logits and selection");
        }
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
    let backend = seismic::BackendName::Cuda;
    let linked = {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        // Only a CUDA device across a link has a host tier: on unified memory
        // (GB10's) the bank is the one store, and there is no tier to grow.
        let topology = catalog.topology();
        seismic::testing::selected(&topology, backend).is_some_and(|executor| {
            matches!(crate::platform::host_tier(&topology, executor), Some(crate::planning::HostTier::Linked))
        })
    };
    if !linked {
        return;
    }
    let definition = qwen4exp_fixture_definition_of(9, 256);
    let (directory, definition, package) = fixture_with("tier-grown", definition, streamed_experts_manifest);
    let stored = stored_experts(&package, &definition);
    {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        scores(&catalog, &definition, &package, &stored, backend, Banked::Half, Start::TierGrown)
            .assert_as_resident(backend, "the grown tier's load scores as the resident one");
    }
    std::fs::remove_dir_all(&directory).ok();
}

/// On every unified device, a partial bank's misses land on the device,
/// unpriced: each record converted by its layer's landing launch as its read
/// lands, the experts after it reading it in submission order, so its steps
/// score as the resident load does ([`scores`] asserts the device landed
/// some).
#[test]
fn misses_the_device_lands_score_as_resident() {
    let definition = qwen4exp_fixture_definition_of(9, 256);
    let (directory, definition, package) = fixture_with("device-landed", definition, streamed_experts_manifest);
    let stored = stored_experts(&package, &definition);
    {
        let _placement = crate::test_support::process_placement();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
            scores(&catalog, &definition, &package, &stored, backend, Banked::Half, Start::AtOnce)
                .assert_as_resident(backend, "the misses the device landed score as the resident load's");
        }
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
        Operation::Forward { request, kind, tokens, position, conditioning, demand, select, committed, prime, front, chain } => {
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
                front,
                chain,
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
    /// The banked load's streamed decode launches by what their layer's
    /// bank held of the experts they routed to ([`residencies`]).
    held: [u64; 3],
}

impl Scores {
    /// The banked load's streamed decode launches met every residency:
    /// some layer held none of the experts its launch routed to, some held
    /// part of them, and some held every one.
    fn assert_every_residency(&self, backend: seismic::BackendName) {
        let [cold, partial, warm] = self.held;
        assert!(
            cold > 0 && partial > 0 && warm > 0,
            "{backend:?}: the streamed launches held none, part and all of their experts {cold}, {partial} and {warm} times"
        );
    }

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

/// How many streamed layers of `domain`'s last completed step held none,
/// part and every one of the experts its decode launch routed to (a grouped
/// step routes none).
fn residencies(domain: &ExecutorDomain) -> [u64; 3] {
    let mut held = [0; 3];
    let Some((_, cost)) = domain.expert_costs().and_then(|costs| costs.completed) else {
        return held;
    };
    for layer in &cost.experts {
        let Some(routed) = &layer.routed else {
            continue;
        };
        let absent = routed.experts.iter().filter(|&expert| layer.missed.contains(expert)).count();
        let residency = match absent {
            0 => 2,
            absent if absent == routed.experts.len() => 0,
            _ => 1,
        };
        held[residency] += 1;
    }
    held
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
    stored: &[[Stack; 3]],
    backend: seismic::BackendName,
    banked: Banked,
    start: Start,
) -> Scores {
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
            Banked::Half => {
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
        domain.open_served(&mut bindings, request, None).unwrap();
        let (outcome, mut returned) =
            run(&mut domain, bindings, scored(forward(request, WorkKind::Prefill, &[5, 1], 0, true)), false, backend);
        // The prompt step's misses reach the site as a decode step's do.
        let mut sited = site_answers(&domain, &stored, expansion, backend, 0);
        let mut held = residencies(&domain);
        let mut steps = vec![step_bits(&outcome)];
        for position in 2..2 + DECODE_STEPS {
            let token = (position * 37 % 256) as u32;
            let (outcome, next) =
                run(&mut domain, returned, scored(forward(request, WorkKind::Decode, &[token], position, true)), false, backend);
            sited.extend(site_answers(&domain, &stored, expansion, backend, steps.len()));
            for (total, step) in held.iter_mut().zip(residencies(&domain)) {
                *total += step;
            }
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
        scores.push((steps, sited, site, held));
    }
    let [(resident, ..), (banked, sited, site, held)] = <[_; 2]>::try_from(scores).expect("a resident and a banked load");
    Scores { resident, banked, sited, site, held }
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
    use crate::assessment::Priced;
    use magnitude_cost_model::{ExpertLines, HostLines};
    let reference = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
    let lines = Priced::Measured(
        domain
            .expert_bank_lines()
            .expect("a sited load has a bank")
            .iter()
            .map(|bank| ExpertLines { host: HostLines { record_bytes: bank.record_bytes, ..reference.host }, ..reference })
            .collect(),
    );
    pricing.measures(lines, Priced::Pending(crate::assessment::PendingPrices::standing(crate::assessment::PointRequest::Costs(crate::assessment::CostWalk::Required))))
}

/// The token a step's last row selected.
fn selected_token(outcome: &Outcome, backend: seismic::BackendName) -> u32 {
    let Outcome::Forward { rows, .. } = outcome else {
        panic!("{backend:?}: a forward step has a forward outcome");
    };
    let crate::Selected::Token(selected) = rows
        .last()
        .and_then(|row| row.selected)
        .unwrap_or_else(|| panic!("{backend:?}: the step selects a token"))
    else {
        panic!("{backend:?}: the step selects a token");
    };
    selected.0
}

/// On Metal and CPU, a streamed load's prompt groups at the kl's shape (a
/// 4096-token context, groups of 7 chunks of 512 rows, two prompts in
/// turn, the second's experts held in the host tier the first filled) end,
/// then decode, and select the resident load's tokens.
#[test]
fn a_streamed_load_runs_its_prompt_groups_then_decodes() {
    let (directory, definition, package) = fixture_of("streamed-groups", streamed_experts_manifest);
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let chunks = (SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS;
    let prompts = Prompts { requests: 2, chunks, rows: SPLIT_CHUNK_ROWS };
    let backends = [seismic::BackendName::Metal, seismic::BackendName::Cpu];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let selected = |banked| {
            let (steps, _) = chained_prompts(&catalog, &definition, &package, backend, prompts, chunks, banked);
            steps.iter().filter_map(|(bits, _)| bits.last().and_then(|(_, _, selected)| *selected)).collect::<Vec<_>>()
        };
        let resident = selected(Banked::Resident);
        let streamed = selected(Banked::Half);
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
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, cuda, limits(2, false, 1, 1), Banked::Whole);
    let request = RequestId(1);
    domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
    domain.open_served(&mut bindings, request, None).unwrap();
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
    // The shrink takes nothing more than the bank holds: the device store's
    // count never rises above what it is now.
    let held = device.allocated();
    device.restart_high_water();
    let ran = domain.resident_decodes();
    assert!(ran > 0, "the whole bank's steps before the shrink ran its resident forms");
    let floor = domain.expert_residency().unwrap().expect("a streamed load holds a bank").floor;
    let step = domain.shrink_expert_bank_to(&mut bindings, crate::BankHolding::Experts, floor, crate::CompletionWake::new(|| {})).unwrap();
    assert!(matches!(step, crate::BankStep::Begun(_)), "a whole bank shrinks: {step:?}");
    while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
    assert!(domain.expert_bank_slots() < whole, "the bank shrank from {whole:?}");
    assert!(device.high_water() <= held, "the shrink held {} bytes, more than the {held} the bank held", device.high_water());
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
/// the first of a class, not after a shrink of the state's surplus releases
/// a prompt group's class, and not on a bank shrink's step. Each step's recordings,
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
/// eight and four, across a shrink of the state's surplus and a bank shrink's
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
            domain.open_served(&mut bindings, request, None).unwrap();
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
            domain.reconcile(pending, PhysicalDecision::rows(1)).unwrap();
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
    let (directory, definition, package) = fixture_with(test, qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    const FOUR: usize = DecodeSteps::REQUESTS / 2;
    // A shrink: a load whose experts are resident chains any prompt, its
    // group claiming a class of two places as it opens and freeing it as
    // its chain closes, before a shrink of the state's surplus releases the
    // slots.
    {
        let (mut domain, bindings, _, _) = load(catalog, &definition, &package, backend, DecodeSteps::limits(2), Banked::Resident);
        if backend == seismic::BackendName::Metal {
            assert_eq!(domain.device().graph_cost(), seismic::GraphCost::None, "Metal charges no family for its graphs");
        }
        let (mut steps, mut bindings) = DecodeSteps::open(&mut domain, bindings, 2, backend);
        assert_eq!(domain.group_places().get(), 1, "{backend:?}: the prompt group's class is freed as its chain closes");
        bindings = steps.step(&mut domain, bindings, 1, "the first of one request", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "the first of eight", backend);
        let released = crate::domain::fixtures::shrink_settled(&mut domain, &mut bindings);
        assert_eq!(domain.group_places().get(), 1, "{backend:?}: a shrink holds no class");
        bindings = steps.step(&mut domain, bindings, FOUR, "the first of four, after the surplus release", backend);
        bindings = steps.step(&mut domain, bindings, DecodeSteps::REQUESTS, "of eight, after the surplus release", backend);
        steps.assert_none(backend, &format!("a shrink of {released} bytes"));
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
        // The shrink takes nothing more than the bank holds.
        let held = device.allocated();
        device.restart_high_water();
        let floor = domain.expert_residency().unwrap().expect("a streamed load holds a bank").floor;
        let step = domain.shrink_expert_bank_to(&mut bindings, crate::BankHolding::Experts, floor, crate::CompletionWake::new(|| {})).unwrap();
        assert!(matches!(step, crate::BankStep::Begun(_)), "{backend:?}: a whole bank shrinks: {step:?}");
        while domain.advance_expert_bank(crate::CompletionWake::new(|| {})).unwrap().is_none() {}
        assert!(domain.expert_bank_slots() < whole, "{backend:?}: the bank shrank from {whole:?}");
        assert!(device.high_water() <= held, "{backend:?}: the shrink held {} bytes, more than the {held} the bank held", device.high_water());
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
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with("drafter-graphs", mtp, crate::planning::fixtures::fixture_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let cuda = seismic::BackendName::Cuda;
    assert!(available(&catalog, cuda), "the test runs on a host with a CUDA device");
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
/// producer's) on a head of the plain fixture, on every backend the run
/// selects.
#[test]
fn a_head_without_selection_drafts_from_a_multi_row_entry_pass() {
    let mtp = crate::planning::fixtures::with_mtp_head(crate::planning::fixtures::fixture_definition());
    let (directory, definition, package) = fixture_with("head-multi-row-entry", mtp, crate::planning::fixtures::fixture_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let method = PlannedMethod::Mtp {
        greedy_proposals: DrafterSteps::PROPOSALS as u8,
        sampled_proposals: DrafterSteps::PROPOSALS as u8,
        widths: crate::DraftWidths::Adaptive,
    };
    let backends = [seismic::BackendName::Cpu, seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, bindings, _, _) =
            load_with(&catalog, &definition, &package, backend, limits(DrafterSteps::ROWS, true, 1, 0), Banked::Resident, method);
        let row_bytes = definition.decoder.hidden as usize * definition.decoder.activation_dtype.bytes();
        let mut steps = DrafterSteps { backend, row_bytes, recorded: Vec::new(), drafted: Vec::new() };
        let request = RequestId(1);
        let (mut position, mut bindings) = steps.prime(&mut domain, bindings, request);
        for _ in 0..2 {
            bindings = steps.draft_from(&mut domain, bindings, std::iter::once((request, &mut position)), 2, DrafterSteps::PROPOSALS);
        }
        assert_eq!(steps.drafted.len(), 2, "{backend:?}");
        domain.close(request).unwrap_or_else(|error| panic!("{error}"));
        drop(bindings);
        drop(domain);
    }
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
        domain.open_served(&mut bindings, request, None).unwrap();
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
            front: Vec::new(),
            chain: None,
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
            domain.reconcile(pending, PhysicalDecision::rows(entry_rows)).unwrap_or_else(|error| panic!("{error}"));
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
        domain.open_served(&mut bindings, request, None).unwrap();
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
                true => match steps.last().and_then(|bits| bits.last()).and_then(|(_, _, selected)| *selected) {
                    Some(crate::Selected::Token(token)) => token.0,
                    other => panic!("a step selects a token, got {other:?}"),
                },
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

/// Rows of one prompt chunk, the launch row bound of the chain test: a
/// prompt class, above the decode form's rows (`VERIFY_ROWS`), so every
/// routed layer splits into its front and back around the group's pass.
const CHUNK_ROWS: usize = 2 * magnitude_kernels::constants::VERIFY_ROWS as usize;

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
        front: Vec::new(),
        chain: None,
    }
}

/// A step's outputs as bits: per row, its features, its logits and its
/// selection.
type StepBits = Vec<(Option<Vec<u32>>, Option<Vec<u32>>, Option<crate::Selected>)>;

fn step_bits(outcome: &Outcome) -> StepBits {
    let Outcome::Forward { rows, .. } = outcome else {
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
                row.selected,
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
    prompts: Prompts,
    chain_capacity: usize,
    banked: Banked,
) -> (Vec<(StepBits, Option<usize>)>, u64) {
    chained_prompts_at_pass(catalog, definition, package, backend, prompts, chain_capacity, banked, None)
}

fn chained_prompts_at_pass(
    catalog: &seismic::Devices,
    definition: &Rc<ModelDefinition>,
    package: &Arc<Package>,
    backend: seismic::BackendName,
    Prompts { requests, chunks, rows }: Prompts,
    chain_capacity: usize,
    banked: Banked,
    pass: Option<crate::PassLaunches>,
) -> (Vec<(StepBits, Option<usize>)>, u64) {
    let (mut domain, mut bindings, pass_rows, _) = load_read(&LoadRequest {
        catalog,
        definition,
        package,
        backend,
        limits: limits(rows, false, chain_capacity, rows),
        banked,
        method: PlannedMethod::Plain,
        codec: KvCodec::Dense,
        machine: Machine::Live,
        pass,
    });
    let requests = (1..=requests).map(RequestId).collect::<Vec<_>>();
    for &request in &requests {
        domain
            .install_input(request, PreparedModelInput::continuation_only())
            .unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
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
        let mut token = match token {
            Some(crate::Selected::Token(token)) => token.0,
            other => panic!("the prompt's last chunk selects the decoded token, got {other:?}"),
        };
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
            token = match bits.last().and_then(|(_, _, selected)| *selected) {
                Some(crate::Selected::Token(token)) => token.0,
                other => panic!("a decode step selects the next token, got {other:?}"),
            };
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
    let zero = |name: &str| (name == ENGRAM_VALUE).then_some(0.0);
    let (directory, with, package) =
        fixture_valued("engram-zero", engram_fixture_definition(), stored_table_manifest, &zero);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
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
/// within float rounding of its decode alone ([`ROW_CLASS_KL`]: the two
/// launches' row classes differ).
#[test]
fn engram_requests_decode_together_as_each_alone() {
    use crate::planning::fixtures::engram_fixture_definition;
    let (directory, definition, package) =
        fixture_valued("engram-slots", engram_fixture_definition(), stored_table_manifest, &|_: &str| None);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let requests = [RequestId(1), RequestId(2)];
    // Each request's logits at its two decode steps, the requests decoding
    // `together` in one launch or each alone.
    let decoded = |backend: seismic::BackendName, together: bool| {
        let limits = limits(CHUNK_ROWS, false, 1, CHUNK_ROWS);
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits, Banked::Resident);
        for &request in &requests {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_served(&mut bindings, request, None).unwrap();
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
                    let Outcome::Forward { rows, .. } = outcome else {
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
            assert!(kl <= ROW_CLASS_KL, "{backend:?}: decode {} of request {}: KL {kl:.3e}", at / 2, at % 2 + 1);
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
    domain.open_served(&mut bindings, request, None).unwrap();
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
        let Outcome::Forward { rows: results, .. } = outcome else {
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

/// The split-pass test's context: its prompt is a group of 7 chunks; with a
/// 3-chunk pass budget, the group runs passes of 3, 3 and 1, the last over
/// the chain's last rows.
const SPLIT_CONTEXT: u64 = 4096;

/// On the CPU: a group of more chunks than its pass holds runs as several
/// passes, and equals the same chunks submitted one step at a time, in
/// groups of the most chunks one pass holds, and in one full merged pass
/// (`assert_steps_equal`), for one request and for two whose prompts run in
/// turn while the first holds its history. The CPU's routed kernels compute
/// each row from that row alone (`routed_experts`), so a row's bits do not
/// depend on the rows its pass covers.
#[test]
fn a_group_longer_than_its_pass_equals_its_chunks_in_one_pass() {
    let (directory, definition, package) = fixture("split-pass");
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    // The longest prompt the context holds beside its two decode rows.
    let chunks = (SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS;
    let cpu = seismic::BackendName::Cpu;
    let pass = 3;
    let pass_budget = crate::PassLaunches::new(std::num::NonZeroUsize::new(pass).unwrap());
    for requests in [1, 2] {
        let prompts = Prompts {
            requests,
            chunks,
            rows: SPLIT_CHUNK_ROWS,
        };
        // Default resident load honors the full merged pass (`chunks` launches).
        let (full, full_pass_rows) = chained_prompts(&catalog, &definition, &package, cpu, prompts, chunks, Banked::Resident);
        assert_eq!(full_pass_rows, (chunks * SPLIT_CHUNK_ROWS) as u64, "resident load must honor full merged pass");
        // With an explicit pass smaller than `chunks`, the group splits into passes of 3, 3, and 1.
        let (split, pass_rows) = chained_prompts_at_pass(&catalog, &definition, &package, cpu, prompts, chunks, Banked::Resident, Some(pass_budget));
        assert_eq!(pass_rows, (pass * SPLIT_CHUNK_ROWS) as u64, "explicit pass budget of {pass} chunks");
        let (one_pass, rows) = chained_prompts(&catalog, &definition, &package, cpu, prompts, pass, Banked::Resident);
        assert_eq!(rows, pass_rows);
        let (alone, _) = chained_prompts(&catalog, &definition, &package, cpu, prompts, 1, Banked::Resident);
        let label = |of: &str| format!("{requests} requests: {of}");
        assert_steps_equal(&full, &split, prompts, &label("split against full pass"));
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
    bindings: crate::StateBindings,
    operations: Vec<Operation>,
    backend: seismic::BackendName,
) -> (Vec<Outcome>, crate::StateBindings, bool) {
    let pairs = operations
        .into_iter()
        .map(|operation| {
            let accepted = operation.row_count() - operation.front_rows();
            (operation, PhysicalDecision::rows(accepted))
        })
        .collect();
    steps_decided(domain, bindings, pairs, backend)
}

fn steps_decided(
    domain: &mut ExecutorDomain,
    bindings: crate::StateBindings,
    pairs: Vec<(Operation, PhysicalDecision)>,
    backend: seismic::BackendName,
) -> (Vec<Outcome>, crate::StateBindings, bool) {
    let (operations, decisions): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let (pending, bindings, claimed) = launched(domain, bindings, &operations, backend);
    let mut outcomes = Vec::new();
    for (pending, decision) in pending.into_iter().zip(decisions) {
        outcomes.push(pending.outcome().clone());
        domain
            .reconcile(pending, decision)
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    }
    (outcomes, bindings, claimed)
}

/// One launch of `operations`, one request's each, whichever way it
/// reserves, finished and not yet reconciled: their outcomes in order, none
/// a failed read's, and whether it claimed the queued step.
fn launched(
    domain: &mut ExecutorDomain,
    mut bindings: crate::StateBindings,
    operations: &[Operation],
    backend: seismic::BackendName,
) -> (Vec<crate::domain::PendingOperationOutcome>, crate::StateBindings, bool) {
    let reserved = domain
        .reserve(&mut bindings, operations)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"))
        .into_resources();
    let ReservedResources::Target(reservation) = reserved else {
        panic!("{backend:?}: a forward reserves the target lane");
    };
    let claimed = matches!(reservation, TargetGraphReservation::Claim(_));
    let flight = domain
        .submit_target(bindings, operations, reservation)
        .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error()));
    let (pending, bindings) = domain
        .finish_target(flight)
        .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
    assert_eq!(pending.len(), operations.len(), "{backend:?}: an outcome per operation");
    for pending in &pending {
        assert!(
            !matches!(pending.outcome(), Outcome::WeightReadFailed(_)),
            "{backend:?}: {:?}",
            pending.outcome()
        );
    }
    (pending, bindings, claimed)
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
    let (directory, definition, package) =
        fixture_with("starved-step", qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let prompt = (0..CHUNK_ROWS as u32).map(|row| (row * 37 + 11) % 256).collect::<Vec<_>>();
    let backends = [seismic::BackendName::Cuda, seismic::BackendName::Metal];
    for backend in backends.into_iter().filter(|&backend| available(&catalog, backend)) {
        let limits = ResourceLimits { conversations: std::num::NonZeroUsize::new(2).unwrap(), ..limits(CHUNK_ROWS, false, 1, CHUNK_ROWS) };
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits, Banked::Half);
        let (unstarved, starved) = (RequestId(1), RequestId(2));
        for request in [unstarved, starved] {
            domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
            domain.open_served(&mut bindings, request, None).unwrap();
        }
        let operation = |request| scored(forward(request, WorkKind::Prefill, &prompt, 0, true));
        let (reference, returned) = run(&mut domain, bindings, operation(unstarved), false, backend);
        let (pending, returned) = starved_step(&mut domain, returned, operation(starved), PATIENCE_BOUND, backend);
        assert!(
            matches!(pending.outcome(), Outcome::Interrupted { .. }),
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
        domain.reconcile(pending, PhysicalDecision::rows(1)).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
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
/// chunks run is interrupted at the chunk that launched it: aborted, the
/// chain concludes with no chunk reconciled, and its holder's state is the
/// one it opened with. The chain's later chunks run in its launch
/// (layer-major), so the first chunk's flight is the one their waits fall
/// in. Prefilled again from that state, it decodes bit for bit as a
/// sequence whose prompt was never starved.
#[test]
fn a_starved_chunk_of_a_chain_concludes_it_and_the_rerun_scores_as_unstarved() {
    // At the kl's shape, where a banked load's prompt runs as one chain
    // (`a_banked_loads_prompt_groups_score_as_resident`).
    let (directory, definition, package) =
        fixture_with("starved-chain", qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
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
        domain.open_served(&mut bindings, request, None).unwrap();
        let (reference, returned) = prompt_then_decode(&mut domain, bindings, request, chunks, rows, backend);
        drop(returned);
        drop(domain);

        let (mut domain, mut bindings, _, _) = open();
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
        // The chain opens with its first chunk, which launches the rest,
        // and is starved.
        domain.plan_successors((1..chunks).map(|next| prompt_chunk(request, next, chunks, rows)).collect());
        let (pending, returned) =
            starved_step(&mut domain, bindings, prompt_chunk(request, 0, chunks, rows), PATIENCE_BOUND, backend);
        assert!(
            matches!(pending.outcome(), Outcome::Interrupted { .. }),
            "{backend:?}: a starved chunk of a chain is interrupted: {:?}",
            pending.outcome()
        );
        domain.abort(pending).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        assert_eq!(domain.chain_holder(), None, "{backend:?}: the chain concluded");
        assert_eq!(domain.resume_state(request).unwrap().position(), 0, "{backend:?}: at the state it opened with");
        // It prefills again and decodes as the unstarved sequence.
        let (rerun, returned) = prompt_then_decode(&mut domain, returned, request, chunks, rows, backend);
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
    let (directory, definition, package) =
        fixture_with("queued-teardown", qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _, _) =
            load(&catalog, &definition, &package, backend, limits(CHUNK_ROWS, true, 1, CHUNK_ROWS), Banked::Half);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
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
    let (directory, definition, package) =
        fixture_with("lone-chunks", qwen4exp_fixture_definition_of(9, 256), streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let mut long = (*definition).clone();
    long.decoder.context_limit = SPLIT_CONTEXT;
    let definition = Rc::new(long);
    let (chunks, rows) = ((SPLIT_CONTEXT as usize - 2) / SPLIT_CHUNK_ROWS, SPLIT_CHUNK_ROWS);
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings, _, _) = load(&catalog, &definition, &package, backend, limits(rows, false, chunks, rows), Banked::Half);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
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
    match steps
        .last()
        .and_then(|bits| bits.last())
        .and_then(|(_, _, selected)| *selected)
    {
        Some(crate::Selected::Token(token)) => token.0,
        other => panic!("{backend:?}: the step selects a token, got {other:?}"),
    }
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
    let (_directory, definition, package) = fixture_valued(
        "kimi-delta-qualifies",
        crate::planning::fixtures::glm_slice_fixture_definition(),
        crate::planning::fixtures::fixture_manifest,
        &|_| None,
    );
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
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
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
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
        domain.open_served(&mut bindings, request, None).unwrap();
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


/// On the CPU backend, without a GPU, loading both a whole expert bank
/// (`HostTier::Unified`, multiple routed layers sharing one bank group at
/// distinct `home` offsets, imported record-by-record and exposed through
/// multi-slab `SlabView` sub-views) and a tier-placed expert bank
/// (`HostTier::Linked`, exercising both an empty 0-slot host tier slice and
/// a grown multi-slab host tier) succeeds within strict memory-ledger claims,
/// and every layer's whole-bank home sub-view holds bit-for-bit identical
/// weights to the unbanked CPU resident load.
#[test]
fn cpu_loads_whole_and_tier_placed_expert_banks() {
    let backend = seismic::BackendName::Cpu;
    let (directory, definition, package) =
        fixture_with("cpu-whole-and-tier-banks", qwen4exp_fixture_definition_of(2, 16), streamed_experts_manifest);
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");

    let limits = limits(2, false, 1, 1);
    let release = platform::ReleaseRate::of_machine(None);
    let selected = platform::select_device(
        &catalog,
        ExecutionPath::Native,
        platform::DeviceRequest::Backend(backend),
        &release,
    )
    .unwrap();
    let device = Rc::new(
        crate::test_support::open(&catalog, catalog.resolve(selected.info.selector()).unwrap()).unwrap(),
    );
    let capacity = ResourceCapacity {
        domain_bytes: crate::MemoryBudget::of_selected_stable(&selected).unwrap().capacity()
            .min(1024 * 1024 * 1024),
        tensor_operations: crate::TensorOperations::of(selected.info.forms_tensor_operations()),
        allocations: seismic::NativeAllocations::EXACT,
    };
    let base_draft = ExecutionPlanner::prepare(
        &selected,
        &package.manifest(),
        &definition,
        ComponentSelection {
            head: false,
            vision: false,
        },
        ExecutionPath::Native,
        PlannedMethod::Plain,
        KvCodec::Dense,
        limits,
    )
    .unwrap();

    // Attest target weight-import kernels once on CPU using `base_draft`, and
    // seal graphs once to obtain an admitted `ResourcePlan` shape.
    let base_state = ResourcePlanner::state_plan(
        &definition,
        base_draft.load(),
        base_draft.policy().method(),
        KvCodec::Dense,
        limits,
        capacity,
    )
    .unwrap();
    let mut programs = AttestedPrograms::prepare_draft(
        &base_draft,
        &seismic::Charges::tally(&device),
        crate::TuningContext {
            definition: &definition,
            weights: &crate::ZeroTuningWeights,
            observer: &crate::UnreportedTuning,
            cache: None,
            numerical_profile: crate::ResolvedNumericalProfile::standard(),
            settings: crate::TuningSettings::STANDARD,
        },
        crate::UnpreparedUnits::ServeDefaults,
    )
    .unwrap();
    let base_draft = base_draft
        .finalize_readout(&definition, programs.readout_choices(), &crate::MemoryBudget::of_selected_stable(&selected).unwrap())
        .unwrap();
    let target_graphs = programs
        .prepare_target_graphs(
            &device,
            base_draft.load(),
            &definition.decoder,
            &base_state,
            base_draft.programs().target(),
            limits,
            base_draft.load().pass(limits),
            &seismic::Workers::new(1),
        )
        .unwrap();
    let target_readout_graphs = programs
        .prepare_target_readout_graphs(&device, base_draft.load(), &definition.decoder, limits)
        .unwrap();
    programs
        .prepare_auxiliary_graphs(
            &device,
            base_draft.load(),
            &definition,
            base_draft.programs(),
            &base_state,
            limits,
            PlannedMethod::Plain,
        )
        .unwrap();
    let base_resources = ResourcePlanner::plan_with_state(
        base_state,
        &target_graphs,
        &target_readout_graphs,
        programs.drafter_graphs(),
        None,
        programs.state_graphs().unwrap(),
    )
    .unwrap();
    programs.install_target_graphs(target_graphs);
    programs.install_target_readout_graphs(target_readout_graphs);
    let programs = Rc::new(programs);

    let load_resident_target = |draft: crate::ExecutionPlanDraft| {
        let execution = Rc::new(draft.admit(base_resources.clone()).unwrap());
        let identity = ResourceDomainId::new("cpu-bank-test").unwrap();
        let ledgers = crate::domain::fixtures::fixture_ledgers(
            &device,
            &execution,
            &seismic::Devices::discover(&[]).unwrap(),
        );
        let mut recorded = crate::memory::ledger::Recorded::default();
        let load = execution.load();
        let demand = match load.expert_bank() {
            Some(bank) => load.clone().with_load_slots(bank.slots, bank.host_slots).unwrap().target_weight_demand(),
            None => load.target_weight_demand(),
        }
        .unwrap();
        let mut weights = ledgers
            .changes()
            .change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, demand))
            .unwrap();
        let host_tables = crate::memory::ClaimDemand {
            host: load.host_table_bytes().unwrap(),
            ..Default::default()
        };
        let host_tables = ledgers
            .changes()
            .change(|keeping| ledgers.claim(&keeping.claims(), &device, crate::memory::ledger::CommittedKind::Load, host_tables))
            .unwrap();
        // On CPU both `DeviceOnly` and `Shared` lie in `HostRam` at the bank's
        // one uniform granule, so a tier-placed CPU fixture shares that single
        // `Records` pool across `records` and `host`.
        let pools = match execution.load().expert_bank().filter(|bank| bank.host_granule().is_some()) {
            Some(bank) => {
                let state = ledgers
                    .open_pool(
                        &device,
                        seismic::Contract::DeviceOnly,
                        crate::memory::ledger::PoolClass::State,
                        execution.resources().target_state().granule,
                    )
                    .unwrap();
                let records = ledgers
                    .open_pool(
                        &device,
                        seismic::Contract::DeviceOnly,
                        crate::memory::ledger::PoolClass::Records,
                        bank.granule(),
                    )
                    .unwrap();
                crate::LoadPools {
                    state,
                    records: Some(records.clone()),
                    host: Some(records),
                }
            }
            None => crate::LoadPools::open(&device, &ledgers, &execution).unwrap(),
        };
        let mut residency = ResidencyStore::for_target(
            device.clone(),
            programs.clone(),
            (*execution).clone(),
            identity,
            &ledgers,
            &mut weights,
            pools,
        )
        .unwrap();
        let target = weights
            .allocate(&device, |charges| residency.load_target(charges, &definition, &package, &host_tables, Box::new(|_, _| ())))
            .unwrap();
        recorded.record(weights);
        recorded.record(host_tables);
        (target, ledgers, recorded)
    };

    // 1. Unbanked resident CPU baseline vs whole-bank CPU load: every layer's
    // home view is a 32-expert multi-slab `SlabView` sub-view of group 0,
    // imported on CPU via `import_records` and carved cleanly in the ledger.
    let (resident_target, _, _) = load_resident_target(base_draft.clone());

    let unified = crate::ExecutionPlanDraft::test_bank_substrate(
        backend,
        crate::planning::HostTier::Unified(crate::planning::Parking::InKernels),
    );
    let whole_draft = base_draft
        .with_test_bank(&definition, unified, 16, 16, 0, None)
        .unwrap();
    assert_eq!(whole_draft.load().expert_bank().map(|b| b.whole()), Some(true));
    let (whole_target, whole_ledgers, mut whole_recorded) = load_resident_target(whole_draft);
    let whole_bank = whole_target.bank.as_ref().expect("whole bank opened on CPU");
    assert!(whole_bank.plan().layers().len() >= 2);
    for (index, layer) in whole_bank.plan().layers().iter().enumerate() {
        assert_eq!(whole_bank.plan().home(index), (index as u64) * layer.experts);
        for kind in [
            magnitude_family_contracts::WeightKind::ExpertGate,
            magnitude_family_contracts::WeightKind::ExpertUp,
            magnitude_family_contracts::WeightKind::ExpertDown,
        ] {
            let role = magnitude_family_contracts::WeightRole {
                scope: layer.scope,
                kind,
            };
            let home = whole_bank.home(role).unwrap();
            let resident = resident_target.sublayers.get(role).unwrap();
            assert_eq!(home.extents(), resident.tensor().extents());
            assert_eq!(
                home.read_to_host().unwrap(),
                resident.tensor().read_to_host().unwrap(),
                "CPU whole-bank home sub-view for {role:?} matches unbanked resident weights bit for bit"
            );
        }
    }
    let whole_served = crate::experts::ServedBank::new(
        whole_bank.clone(),
        Vec::new(),
        &whole_ledgers,
        &mut whole_recorded,
    )
    .expect("whole CPU bank carves its record holding cleanly");
    drop(whole_served);
    drop(whole_target);

    // 2. Tier-placed (`HostTier::Linked`) bank on CPU: exercise both `host_slots = 0`
    // (empty host tier slice) and `host_slots = 4` (grown multi-slab host tier)
    // within strict memory-ledger claims.
    let linked = crate::ExecutionPlanDraft::test_bank_substrate(
        backend,
        crate::planning::HostTier::Linked,
    );
    let floor = base_draft
        .load()
        .bank_floor(&definition, 1, limits.flights(), linked)
        .unwrap();
    for host_slots in [0u64, 4u64] {
        let tier_draft = base_draft
            .with_test_bank(&definition, linked, floor, 16, host_slots, Some(8))
            .unwrap();
        assert_eq!(tier_draft.load().expert_bank().map(|b| b.whole()), Some(false));
        assert_eq!(tier_draft.load().expert_bank().map(|b| b.tiered()), Some(true));
        let (tier_target, tier_ledgers, mut tier_recorded) = load_resident_target(tier_draft);
        let bank = tier_target.bank.as_ref().expect("tier-placed bank opened on CPU");
        let first_layer = &bank.plan().layers()[0];
        let tier_up = bank
            .part(
                first_layer.scope,
                crate::planning::BankPart::Tier {
                    kind: crate::planning::RoutedTensor::Up,
                },
            )
            .unwrap();
        let empty_tier = tier_up.slice_leading(0, 0).unwrap();
        assert_eq!(empty_tier.extents()[0], 0);
        let sub_tier = tier_up.slice_leading(0, 2).unwrap();
        assert_eq!(sub_tier.extents()[0], 2);
        assert_eq!(bank.host_tier(), Some((bank.host_slots(), 8)));
        let _served = crate::experts::ServedBank::new(
            bank.clone(),
            Vec::new(),
            &tier_ledgers,
            &mut tier_recorded,
        )
        .expect("tier-placed bank carves its Bank and Host record holdings cleanly");
    }

    std::fs::remove_dir_all(&directory).ok();
}

/// A prompt chunk resumed level with its head (a turn continuing a prefix
/// decoded in front passes, whose last pass's head launch entered its
/// selection: the prefix cache's resumed state, or a request continuing in
/// place) primes the head with its pairs past its first token, at its own
/// position (`Priming::pairs_previous` false). It reads no boundary
/// feature, and launches though the decoded prefix holds none: a decode
/// pass keeps no boundary feature. On every backend experts stream on, the
/// closing chunk of such a turn runs and its head launch makes the head's
/// guess.
#[test]
fn a_prompt_chunk_resumed_level_with_its_head_primes_without_a_boundary_feature() {
    let _placement = crate::test_support::process_placement();
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let (directory, definition, package) = fronting_fixture("resumed-priming");
    let request = RequestId(1);
    for backend in STREAMING_BACKENDS.into_iter().filter(|&backend| available(&catalog, backend)) {
        let (mut domain, mut bindings) = load_fronting(&catalog, &definition, &package, backend, false, 1, Banked::Whole);
        domain.install_input(request, PreparedModelInput::continuation_only()).unwrap();
        domain.open_served(&mut bindings, request, None).unwrap();
        // The first turn: its prompt at 0..2, then one front pass at 2 that
        // leaves the head level with the target at 3.
        let (pending, _, returned) = run_fronting_prompt(&mut domain, bindings, request, backend);
        let rows = crate::Rows::front(Vec::new(), 0);
        let operation = front_pass(request, &rows, pending, 2, crate::batching::Demand::NONE);
        let (mut outcomes, returned, _) = launched(&mut domain, returned, &[operation], backend);
        let outcome = outcomes.pop().expect("one pass");
        let selected = TokenId(selected_token(outcome.outcome(), backend));
        let decision = PhysicalDecision { accepted_rows: 1, ahead: crate::AheadVerdict::new(&rows, &[selected]) };
        domain.reconcile(outcome, decision).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        // The next turn's closing chunk at 3, its first token the pass's
        // selection, which the head entered already.
        let tokens = vec![selected, TokenId(7), TokenId(9)];
        let prime = crate::Priming { boundary: true, tokens: tokens[1..].to_vec(), position: 3, draft_from: 6 };
        assert!(!prime.pairs_previous(tokens.len()));
        let chunk = Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            select: vec![greedy(5)],
            committed: tokens.len(),
            chain: Some(crate::ChainEntry { position: prime.position + prime.tokens.len() }),
            prime: Some(prime),
            tokens,
            position: 3,
            conditioning: None,
            demand: crate::batching::Demand::SELECT | crate::batching::Demand::FEATURES,
            front: Vec::new(),
        };
        domain.requirements(std::slice::from_ref(&chunk)).unwrap_or_else(|error| panic!("{backend:?}: {error}"));
        let (outcome, _) = run(&mut domain, returned, chunk, false, backend);
        let Outcome::Forward { guesses, .. } = outcome else { panic!("{backend:?}: a prompt's outcome") };
        assert!(
            matches!(guesses.as_slice(), [crate::Guess { kind: crate::GuessKind::Head, .. }]),
            "{backend:?}: the closing chunk's chain is the head's guess alone: {guesses:?}"
        );
    }
    std::fs::remove_dir_all(&directory).ok();
}
