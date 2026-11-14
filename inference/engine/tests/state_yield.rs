//! The production state yield: what a claim takes of the state's spare
//! slabs, the load frees (`ExecutorDomain::release_yielded`). Its stores
//! release the slabs beyond the live requests' next steps, moving no row or
//! bank, and the pool returns what they freed to the OS, emptying sparse
//! blocks where whole ones fall short. Every live history row and claimed
//! bank stays at its offset and reads back its bytes, and a continuation is
//! unchanged. Set `MAGNITUDE_TEST_DEVICE` to `metal`, `cuda`, `vulkan`, or
//! `cpu` to qualify a specific backend. `MAGNITUDE_TEST_KV_CODEC=dense`
//! reduces the number of rows needed to span a slab on a slower backend.
//! The bank test needs a model with multiple banks per slab; the
//! Qwen3.5-0.8B Q4_K_M and BF16 GGUFs have three with the 64 MiB slab
//! target.
//! Run with `MAGNITUDE_TEST_MTP_GGUF=<model.gguf> cargo test --release
//! -p magnitude-engine --test state_yield -- --ignored --nocapture`.

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, ReleaseRate},
    CompletionWake, Demand, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision,
    RequestId, ReservedResources, StateBindings, TokenId, WorkKind, Yielded,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use std::path::PathBuf;

fn domain() -> (ExecutorDomain, StateBindings, usize, usize, usize) {
    let model = PathBuf::from(
        std::env::var_os("MAGNITUDE_TEST_MTP_GGUF").expect("set MAGNITUDE_TEST_MTP_GGUF"),
    );
    let resolved = EngineConfiguration {
        package: PackageOptions {
            target: model,
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            draft_policy: None,
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: std::env::var("MAGNITUDE_TEST_KV_CODEC")
                .ok()
                .map(|codec| codec.parse::<KvCodec>().unwrap()),
            lookahead: false,
            exported_logits_rows: 64,
            numerical_profile: None,
            weight_bound: None,
        },
        context_tokens: Some(8192),
        service: ServiceLimits {
            prefill_tokens: 64,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
            conversations: std::num::NonZeroUsize::MIN,
        },
        path: ExecutionPath::Native,
        device: std::env::var("MAGNITUDE_TEST_DEVICE")
            .map(|device| device.parse::<DeviceRequest>().unwrap())
            .unwrap_or(DeviceRequest::Automatic),
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        release: ReleaseRate::of_machine(None),
    }
    .resolve()
    .unwrap();
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let package = resolved.host.shared_package();
    let (domain, bindings, plan) = build_native_domain(&resolved.manifest, package).unwrap();
    (
        domain,
        bindings,
        vocabulary,
        // Qwen's target store has one Token history domain.
        plan.target_state().sole_history().unwrap().slab_rows as usize,
        plan.target_state().bank_slab_banks().unwrap() as usize,
    )
}

fn tokens(vocabulary: usize, start: usize, count: usize) -> Vec<TokenId> {
    let span = (vocabulary - 1024) as u64;
    (start..start + count)
        .map(|i| TokenId(((i as u64 * 2_654_435_761 + 12_345) % span + 512) as u32))
        .collect()
}

/// Run one forward, accept every row, and return its last row's logits when
/// `logits`. The flight carries the bindings and returns them.
fn forward(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    position: usize,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    logits: bool,
) -> (Option<Vec<u32>>, StateBindings) {
    let count = tokens.len();
    let operation = Operation::Forward {
        request,
        kind,
        tokens,
        position,
        conditioning: None,
        demand: if logits { Demand::LOGITS } else { Demand::NONE },
        select: Vec::new(),
        committed: count,
        prime: None,
    };
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a forward runs on the target lane")
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (mut pending, bindings) = domain.finish_target(flight).unwrap();
    let pending = pending.pop().unwrap();
    let Outcome::Forward { rows } = pending.outcome().clone() else {
        panic!("a forward returns rows")
    };
    let result = logits.then(|| {
        rows.last()
            .unwrap()
            .logits
            .as_ref()
            .unwrap()
            .read_to_host()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    });
    domain
        .reconcile(
            pending,
            PhysicalDecision {
                accepted_rows: count,
            },
        )
        .unwrap();
    (result, bindings)
}

/// Make `request` resident on fresh state with no prompt rows.
fn open(domain: &mut ExecutorDomain, bindings: &mut StateBindings, request: RequestId) {
    domain
        .install_input(request, PreparedModelInput::continuation_only())
        .unwrap();
    domain.open_state(bindings, request, None, 0).unwrap();
}

fn prefill(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    vocabulary: usize,
    count: usize,
) -> StateBindings {
    for start in (0..count).step_by(64) {
        let rows = (count - start).min(64);
        bindings = forward(
            domain,
            bindings,
            request,
            start,
            WorkKind::Prefill,
            tokens(vocabulary, start, rows),
            false,
        )
        .1;
    }
    bindings
}

fn assert_accounted(domain: &ExecutorDomain) {
    domain.reconcile_memory_charge(&[]).unwrap();
}

/// Where one request's state lies and what it holds: each history domain's
/// ranges and the bytes of every plane over them, its bank's slot and the
/// bytes of every recurrent component there.
#[derive(Debug, PartialEq, Eq)]
struct Placed {
    ranges: Vec<Vec<(usize, usize)>>,
    rows: Vec<Vec<u8>>,
    bank: usize,
    bank_bytes: Vec<Vec<u8>>,
}

fn placed(domain: &ExecutorDomain, request: RequestId) -> Placed {
    let resume = domain.resume_state(request).unwrap();
    let state = resume.target();
    let store = state.store();
    let planes = store.history_planes().unwrap();
    let (mut ranges, mut rows) = (Vec::new(), Vec::new());
    for history in store.history_domains() {
        let held = state.history_ranges(history);
        for plane in planes.iter().filter(|plane| plane.domain == history) {
            for &(start, count) in &held {
                rows.push(
                    plane
                        .buffer
                        .slice_leading(start as u64, (start + count) as u64)
                        .unwrap()
                        .read_to_host()
                        .unwrap(),
                );
            }
        }
        ranges.push(held);
    }
    let bank = state.bank_index();
    let bank_bytes = match store.has_recurrent_components() {
        true => store
            .recurrent_arenas()
            .iter()
            .map(|arena| arena.slice_leading(bank as u64, bank as u64 + 1).unwrap().read_to_host().unwrap())
            .collect(),
        false => Vec::new(),
    };
    Placed { ranges, rows, bank, bank_bytes }
}

/// The bytes the state's pool holds, read through `request`'s store.
fn state_pool_held(domain: &ExecutorDomain, request: RequestId) -> u64 {
    domain.resume_state(request).unwrap().target().store().pool().held_bytes()
}

/// Have a claim take every byte of cache the device's ledger yields, the
/// state's spare slabs among it, and let the load free what it took: the
/// pool's bytes returned to the OS. The allocated charge falls by exactly
/// them, and every `live` request's rows and bank stay where they were with
/// their bytes.
fn yield_spare_state(domain: &mut ExecutorDomain, bindings: &mut StateBindings, live: &[RequestId]) -> u64 {
    let before = live.iter().map(|&request| placed(domain, request)).collect::<Vec<_>>();
    let held = state_pool_held(domain, live[0]);
    let charge_before = domain.reconcile_memory_charge(&[]).unwrap();
    let claim = magnitude_executor::fixtures::take_all_cache(domain);
    let yielded = domain.release_yielded(bindings, CompletionWake::new(|| {})).unwrap();
    assert!(matches!(yielded, Yielded::Released { .. }), "{yielded:?}");
    drop(claim);
    let returned = held - state_pool_held(domain, live[0]);
    let charge_after = domain.reconcile_memory_charge(&[]).unwrap();
    eprintln!("returned={returned} charge={}=>{}", charge_before.allocated, charge_after.allocated);
    assert!(returned > 0, "the yield returned nothing to the OS");
    assert_eq!(charge_before.allocated - charge_after.allocated, returned);
    for (&request, before) in live.iter().zip(&before) {
        assert_eq!(&placed(domain, request), before, "{request:?} moved or changed");
    }
    returned
}

#[test]
#[ignore = "requires a model with more than one bank per slab"]
fn a_yield_returns_bank_slabs_with_every_live_bank_in_place() {
    let (mut domain, mut bindings, vocabulary, _, slab_banks) = domain();
    assert!(slab_banks > 1, "model needs multiple banks per slab");

    // Closing every other peer leaves live banks scattered over several
    // slabs, every one sparse, and the slabs past them empty.
    let peers = slab_banks * 4;
    let reference = RequestId(peers as u64 + 1);
    let kept = (0..peers)
        .filter(|index| index % 2 == 1)
        .map(|index| RequestId(index as u64 + 1))
        .collect::<Vec<_>>();
    open(&mut domain, &mut bindings, reference);
    bindings = prefill(&mut domain, bindings, reference, vocabulary, 1);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, &mut bindings, request);
        bindings = prefill(&mut domain, bindings, request, vocabulary, 1);
    }
    for index in (0..peers).filter(|index| index % 2 == 0) {
        domain.close(RequestId(index as u64 + 1)).unwrap();
    }

    let (expected, mut bindings) = forward(
        &mut domain,
        bindings,
        reference,
        1,
        WorkKind::Decode,
        tokens(vocabulary, 1, 1),
        true,
    );
    let expected = expected.unwrap();
    let live = std::iter::once(reference).chain(kept.iter().copied()).collect::<Vec<_>>();
    yield_spare_state(&mut domain, &mut bindings, &live);

    for &request in &kept {
        let actual;
        (actual, bindings) = forward(
            &mut domain,
            bindings,
            request,
            1,
            WorkKind::Decode,
            tokens(vocabulary, 1, 1),
            true,
        );
        assert_eq!(actual.unwrap(), expected, "continuation changed after the yield");
    }
    assert_accounted(&domain);
    domain.close(reference).unwrap();
    for request in kept {
        domain.close(request).unwrap();
    }
}

#[test]
#[ignore = "requires a model device and MAGNITUDE_TEST_MTP_GGUF"]
fn a_yield_returns_history_and_bank_slabs_with_every_live_row_in_place() {
    let (mut domain, mut bindings, vocabulary, slab_rows, slab_banks) = domain();
    assert!(slab_rows > 0 && slab_banks > 0);
    eprintln!("history_slab_rows={slab_rows} bank_slab_banks={slab_banks}");

    // The reference opens in the low slab; peers fill the rest of it and
    // the whole slab above before the kept request opens past them, so
    // closing the peers leaves the low and high slabs sparse and the one
    // between empty, and their banks' slabs empty past the live ones.
    let peers = slab_banks.max(10);
    let peer_rows = (2 * slab_rows).div_ceil(peers) + 1;
    assert!(peer_rows < 8192, "test context cannot span two history slabs");
    let reference = RequestId(peers as u64 + 1);
    let kept = RequestId(peers as u64 + 2);
    open(&mut domain, &mut bindings, reference);
    bindings = prefill(&mut domain, bindings, reference, vocabulary, 32);
    assert_accounted(&domain);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, &mut bindings, request);
        bindings = prefill(&mut domain, bindings, request, vocabulary, peer_rows);
        assert_accounted(&domain);
    }
    open(&mut domain, &mut bindings, kept);
    bindings = prefill(&mut domain, bindings, kept, vocabulary, 32);
    assert_accounted(&domain);
    for index in 0..peers {
        domain.close(RequestId(index as u64 + 1)).unwrap();
    }
    let (expected, mut bindings) = forward(
        &mut domain,
        bindings,
        reference,
        32,
        WorkKind::Decode,
        tokens(vocabulary, 32, 1),
        true,
    );
    let expected = expected.unwrap();
    yield_spare_state(&mut domain, &mut bindings, &[reference, kept]);
    let (actual, _) = forward(
        &mut domain,
        bindings,
        kept,
        32,
        WorkKind::Decode,
        tokens(vocabulary, 32, 1),
        true,
    );
    assert_eq!(actual.unwrap(), expected, "continuation changed after the yield");
    assert_accounted(&domain);
    domain.close(reference).unwrap();
    domain.close(kept).unwrap();
}
