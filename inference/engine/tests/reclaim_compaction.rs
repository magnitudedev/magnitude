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
    TokenId, WorkKind,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_scheduler::{
    domain::{self as service_domain, DomainFlight},
    ServiceLimits,
};
use magnitude_state::{KvCodec, ShrinkPolicy};
use std::path::PathBuf;

fn domain() -> (ExecutorDomain, usize, usize, usize) {
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
    let (domain, plan) = build_native_domain(&resolved.manifest, package).unwrap();
    (
        domain,
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

fn forward(
    domain: &mut ExecutorDomain,
    request: RequestId,
    position: usize,
    kind: WorkKind,
    tokens: Vec<TokenId>,
    logits: bool,
) -> Option<Vec<u32>> {
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
    let groups = service_domain::group(domain, vec![operation]);
    let [group] = groups.as_slice() else {
        panic!("one forward forms one group")
    };
    let DomainFlight::Target(flight) = service_domain::submit_group(domain, group).unwrap() else {
        panic!("a forward runs on the target lane")
    };
    let pending = domain.finish_target(flight).unwrap().pop().unwrap();
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
    result
}

/// Make `request` resident on fresh state with no prompt rows.
fn open(domain: &mut ExecutorDomain, request: RequestId) {
    domain
        .install_input(request, PreparedModelInput::continuation_only())
        .unwrap();
    domain.open_state(request, None).unwrap();
}

fn prefill(domain: &mut ExecutorDomain, request: RequestId, vocabulary: usize, count: usize) {
    for start in (0..count).step_by(64) {
        let rows = (count - start).min(64);
        forward(
            domain,
            request,
            start,
            WorkKind::Prefill,
            tokens(vocabulary, start, rows),
            false,
        );
    }
}

fn assert_accounted(domain: &ExecutorDomain) {
    let charge = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge.unattributed, 0, "{charge:?}");
}

#[test]
#[ignore = "requires a model with more than one bank per slab"]
fn reclaim_relocates_live_banks_without_changing_continuation() {
    let (mut domain, vocabulary, _, slab_banks) = domain();
    assert!(slab_banks > 1, "model needs multiple banks per slab");

    let peers = slab_banks * 2;
    let reference = RequestId(peers as u64 + 1);
    let relocated = RequestId(peers as u64 + 2);
    open(&mut domain, reference);
    prefill(&mut domain, reference, vocabulary, 1);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, request);
        prefill(&mut domain, request, vocabulary, 1);
    }
    open(&mut domain, relocated);
    prefill(&mut domain, relocated, vocabulary, 1);
    for index in 0..peers {
        domain.close(RequestId(index as u64 + 1)).unwrap();
    }

    let expected = forward(
        &mut domain,
        reference,
        1,
        WorkKind::Decode,
        tokens(vocabulary, 1, 1),
        true,
    )
    .unwrap();
    let before = domain.state_compactions().0;
    let charge_before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge_before.unattributed, 0, "{charge_before:?}");
    let released = domain.shrink_state(ShrinkPolicy::Reclaim).unwrap();
    let after = domain.state_compactions().0;
    let charge_after = domain.reconcile_memory_charge(&[]).unwrap();
    assert!(after.banks > before.banks, "banks were not relocated");
    assert!(released > 0 && charge_after.charged < charge_before.charged);
    assert_eq!(released, charge_before.charged - charge_after.charged);
    assert_eq!(charge_after.unattributed, 0, "{charge_after:?}");

    let actual = forward(
        &mut domain,
        relocated,
        1,
        WorkKind::Decode,
        tokens(vocabulary, 1, 1),
        true,
    )
    .unwrap();
    assert_eq!(
        actual, expected,
        "continuation changed after bank relocation"
    );
    domain.close(reference).unwrap();
    domain.close(relocated).unwrap();
}

#[test]
#[ignore = "requires a model device and MAGNITUDE_TEST_MTP_GGUF"]
fn reclaim_relocates_live_history_and_banks_without_changing_continuation() {
    let (mut domain, vocabulary, slab_rows, slab_banks) = domain();
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
    open(&mut domain, reference);
    prefill(&mut domain, reference, vocabulary, 32);
    assert_accounted(&domain);
    for index in 0..peers {
        let request = RequestId(index as u64 + 1);
        open(&mut domain, request);
        prefill(&mut domain, request, vocabulary, peer_rows);
        assert_accounted(&domain);
    }
    open(&mut domain, relocated);
    prefill(&mut domain, relocated, vocabulary, 32);
    assert_accounted(&domain);
    for index in 0..peers {
        domain.close(RequestId(index as u64 + 1)).unwrap();
    }
    let expected = forward(
        &mut domain,
        reference,
        32,
        WorkKind::Decode,
        tokens(vocabulary, 32, 1),
        true,
    )
    .unwrap();
    let before = domain.state_compactions().0;
    let charge_before = domain.reconcile_memory_charge(&[]).unwrap();
    assert_eq!(charge_before.unattributed, 0, "{charge_before:?}");
    let released = domain.shrink_state(ShrinkPolicy::Reclaim).unwrap();
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
    let actual = forward(
        &mut domain,
        relocated,
        32,
        WorkKind::Decode,
        tokens(vocabulary, 32, 1),
        true,
    )
    .unwrap();
    assert_eq!(actual, expected, "continuation changed after Reclaim");
    assert_accounted(&domain);
    domain.close(reference).unwrap();
    domain.close(relocated).unwrap();
}
