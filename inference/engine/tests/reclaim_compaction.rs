//! Production executor Reclaim compaction with live history and, when a slab
//! holds multiple banks, recurrent banks. Set `MAGNITUDE_TEST_DEVICE` to
//! `metal`, `cuda`, `vulkan`, or `cpu` to qualify a specific backend.
//! `MAGNITUDE_TEST_KV_CODEC=dense` reduces the number of rows needed to span
//! a slab on a slower backend.
//! The bank-only test needs a model with multiple banks per slab; the
//! Qwen3.5-0.8B Q4_K_M GGUF has three with the 64 MiB slab target.
//! Run with `MAGNITUDE_TEST_MTP_GGUF=<model.gguf> cargo test --release
//! -p magnitude-engine --test reclaim_compaction -- --ignored --nocapture`.

use magnitude_engine::{
    build_native_domain,
    composition::EngineConfiguration,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    Demand, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision, RequestId,
    ReservedResources, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::ServiceLimits;
use magnitude_state::{KvCodec, ShrinkPolicy};
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
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: std::env::var("MAGNITUDE_TEST_KV_CODEC")
                .map(|codec| codec.parse::<KvCodec>().unwrap())
                .unwrap_or(KvCodec::AffineK8V4),
            lookahead: false,
            exported_logits_rows: 64,
        },
        context_tokens: Some(8192),
        service: ServiceLimits {
            prefill_tokens: 64,
            decode_tokens: 16,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: std::env::var("MAGNITUDE_TEST_DEVICE")
            .map(|device| device.parse::<DeviceRequest>().unwrap())
            .unwrap_or(DeviceRequest::Automatic),
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
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
    domain.open_state(bindings, request, None).unwrap();
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
    let charge = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge.unattributed, 0, "{charge:?}");
}

#[test]
#[ignore = "requires a model with more than one bank per slab"]
fn reclaim_relocates_live_banks_without_changing_continuation() {
    let (mut domain, mut bindings, vocabulary, _, slab_banks) = domain();
    assert!(slab_banks > 1, "model needs multiple banks per slab");

    let peers = slab_banks * 2;
    let reference = RequestId(peers as u64 + 1);
    let relocated = RequestId(peers as u64 + 2);
    open(&mut domain, &mut bindings, reference);
    bindings = prefill(&mut domain, bindings, reference, vocabulary, 1);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, &mut bindings, request);
        bindings = prefill(&mut domain, bindings, request, vocabulary, 1);
    }
    open(&mut domain, &mut bindings, relocated);
    bindings = prefill(&mut domain, bindings, relocated, vocabulary, 1);
    for index in 0..peers {
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
    let before = domain.state_compactions().0;
    let charge_before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge_before.unattributed, 0, "{charge_before:?}");
    let released = domain
        .shrink_state(&mut bindings, ShrinkPolicy::Reclaim)
        .unwrap();
    let after = domain.state_compactions().0;
    let charge_after = domain.reconcile_memory_charge(&[]).unwrap();
    assert!(after.banks > before.banks, "banks were not relocated");
    assert!(released > 0 && charge_after.charged < charge_before.charged);
    assert_eq!(released, charge_before.charged - charge_after.charged);
    assert_eq!(charge_after.unattributed, 0, "{charge_after:?}");

    let (actual, _) = forward(
        &mut domain,
        bindings,
        relocated,
        1,
        WorkKind::Decode,
        tokens(vocabulary, 1, 1),
        true,
    );
    assert_eq!(
        actual.unwrap(),
        expected,
        "continuation changed after bank relocation"
    );
    domain.close(reference).unwrap();
    domain.close(relocated).unwrap();
}

#[test]
#[ignore = "requires a model device and MAGNITUDE_TEST_MTP_GGUF"]
fn reclaim_relocates_live_history_and_banks_without_changing_continuation() {
    let (mut domain, mut bindings, vocabulary, slab_rows, slab_banks) = domain();
    assert!(slab_rows > 0 && slab_banks > 0);
    eprintln!("history_slab_rows={slab_rows} bank_slab_banks={slab_banks}");

    // Keep the reference in the low slab. Fill the rest of that slab with
    // peers before opening the relocated request in a higher slab. Closing
    // the peers creates holes below live rows in the higher slab.
    let peers = slab_banks.max(10);
    let peer_rows = slab_rows.div_ceil(peers) + 1;
    assert!(peer_rows < 8192, "test context cannot span a history slab");
    let reference = RequestId(peers as u64 + 1);
    let relocated = RequestId(peers as u64 + 2);
    open(&mut domain, &mut bindings, reference);
    bindings = prefill(&mut domain, bindings, reference, vocabulary, 32);
    assert_accounted(&domain);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, &mut bindings, request);
        bindings = prefill(&mut domain, bindings, request, vocabulary, peer_rows);
        assert_accounted(&domain);
    }
    open(&mut domain, &mut bindings, relocated);
    bindings = prefill(&mut domain, bindings, relocated, vocabulary, 32);
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
    let before = domain.state_compactions().0;
    let charge_before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge_before.unattributed, 0, "{charge_before:?}");
    let released = domain
        .shrink_state(&mut bindings, ShrinkPolicy::Reclaim)
        .unwrap();
    let after = domain.state_compactions().0;
    let charge_after = domain.reconcile_memory_charge(&[]).unwrap();
    eprintln!(
        "released={released} compactions={after:?} charge={}=>{}",
        charge_before.charged, charge_after.charged
    );
    assert!(
        after.history_rows > before.history_rows,
        "history was not relocated"
    );
    if slab_banks > 1 {
        assert!(after.banks > before.banks, "banks were not relocated");
    }
    assert!(released > 0 && charge_after.charged < charge_before.charged);
    assert_eq!(released, charge_before.charged - charge_after.charged);
    assert_eq!(charge_after.unattributed, 0, "{charge_after:?}");
    let (actual, _) = forward(
        &mut domain,
        bindings,
        relocated,
        32,
        WorkKind::Decode,
        tokens(vocabulary, 32, 1),
        true,
    );
    assert_eq!(
        actual.unwrap(),
        expected,
        "continuation changed after Reclaim"
    );
    assert_accounted(&domain);
    domain.close(reference).unwrap();
    domain.close(relocated).unwrap();
}
