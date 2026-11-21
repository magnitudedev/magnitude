//! The bank's derived target on a reservation-backed bank, read from its
//! device's ledger at each boundary.

use super::Followed;
use crate::domain::fixtures::{bank_backends, bank_row, banked, NO_LAG};
use crate::domain::{BankResized, BankStep};
use crate::memory::ledger::{Ledger, Relief};
use crate::memory::supply::Reading;
use std::time::{Duration, Instant};

/// The store's capacity.
const TOTAL: u64 = 1 << 40;

/// What other programs hold, unchanged at every reading: their rise
/// between readings is no part of the guarantee here.
const OTHERS: u64 = 1 << 38;

fn wake() -> crate::CompletionWake {
    crate::CompletionWake::new(|| {})
}

/// A reading at `at` showing free what neither other programs nor the
/// process's allocations hold.
fn read(ledger: &Ledger, at: Instant) {
    let available = TOTAL - OTHERS - ledger.estimate().allocated();
    ledger.read(ledger.sampled(at), Reading::device(TOTAL, available, NO_LAG));
}

/// Item 9, the laptop's failure, on the ledger: a bank at its full target,
/// a prompt's claim taking room so the target falls and the bank shrinks,
/// then the claim freed. At the next boundary the target is the full bank
/// again and the bank grows back to it. On main a settle released the bank
/// for the prompt and nothing grew it back.
#[test]
fn the_bank_regrows_to_its_target_once_a_prompts_claim_is_freed() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        let at = |reading: u32| start + Duration::from_millis(100) * reading;
        // Two readings of the same others: no rise between them, so the
        // guarantee holds only the load's own growth.
        read(&ledger, at(1));
        read(&ledger, at(2));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        // The prompt's claim leaves the bank's cache a slot and a half of
        // room above its floor: its target falls to two slots.
        let room = ledger.estimate().target();
        let prompt = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, room - row * 3 / 2)).expect("the prompt's claim fits");
        read(&ledger, at(3));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let shrunk = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(shrunk, BankResized::Shrunk { from: 4, to: 2, .. }), "{backend:?}: {shrunk:?}");
        // The prompt ends: its claim is freed, and the next boundary grows
        // the bank back to its full target.
        drop(prompt);
        read(&ledger, at(4));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Grew(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let grown = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(grown, BankResized::Grown { from: 2, to: 4, .. }), "{backend:?}: {grown:?}");
        read(&ledger, at(5));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the full bank again");
    }
}

/// The cache worth less than the bank's, which yields before it: spare
/// granules and idle graph slots.
fn cheaper_than_the_bank(ledger: &Ledger) -> u64 {
    use crate::memory::ledger::CacheKind;
    let estimate = ledger.estimate();
    estimate.cache_of(CacheKind::Spare) + estimate.cache_of(CacheKind::GraphSlots)
}

/// A claim that takes less than one granule of the bank's cache (its
/// granule a slot here) is the ledger's fact, not the room's jitter: the
/// claim allocates those bytes only once the bank frees them, so the bank
/// sheds a whole slot though its room moved by less than a block, and the
/// claim's allocation, refused while the slot stands, is admitted once the
/// shrink returns it. A first claim holds the free room, allocated, and the
/// cache that yields before the bank's (CUDA's fixture holds some), so the
/// second takes the bank's alone.
#[test]
fn a_claim_taking_under_a_granule_of_the_bank_sheds_a_slot_and_then_allocates() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        let at = |reading: u32| start + Duration::from_millis(100) * reading;
        read(&ledger, at(1));
        read(&ledger, at(2));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        let estimate = ledger.estimate();
        assert_eq!(estimate.guarantee(), 0, "{backend:?}: no rise: every free byte is a claim's");
        let free = estimate.free();
        let room = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, free + cheaper_than_the_bank(&ledger))).expect("free room and the cheaper cache");
        ledger.change(|_| { let entry = room.entry(); entry.take(&entry.change(), free, seismic::memory::Wiring::Pageable) }).unwrap_or_else(|refused| panic!("{backend:?}: free room allocates: {refused}"));
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let bytes = row / 2;
        let claim = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, bytes)).expect("the bank's cache yields to the claim");
        assert_eq!(before - bank(), bytes, "{backend:?}: half a slot of the bank's cache, taken for the claim");
        let allocate = || ledger.change(|_| { let entry = claim.entry(); entry.take(&entry.change(), bytes, seismic::memory::Wiring::Pageable) });
        assert!(allocate().is_err(), "{backend:?}: the taken half slot is the bank's still");
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let shrunk = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(shrunk, BankResized::Shrunk { from: 4, to: 3, .. }), "{backend:?}: {shrunk:?}");
        allocate().unwrap_or_else(|refused| panic!("{backend:?}: the shed slot admits the claim: {refused}"));
    }
}

/// Cache a shortage takes to fit the budget as others move owes no claim:
/// a yield of half a slot (under the bank's block) to the room's own
/// movement keeps the bank as it is, its threshold judging it as it judges
/// the room, as the Metal gate's steady decode needs.
#[test]
fn a_shortages_yield_under_a_block_keeps_the_bank() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        let at = |reading: u32| start + Duration::from_millis(100) * reading;
        read(&ledger, at(1));
        read(&ledger, at(2));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        // Cache worth no more than the bank's yields first; past it, half a
        // slot of the bank's.
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let plan = ledger.plan_relief(cheaper_than_the_bank(&ledger) + row / 2, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("the cache covers the shortage");
        ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
        let fell = before - bank();
        assert!(fell > 0 && fell < row, "{backend:?}: the shortage took {fell} bytes of the bank's cache, under a block");
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: within a block, the bank keeps its slots");
    }
}

/// The load's own commit never spends the guarantee, the room the bank's
/// target leaves for others' rise: a bank at its target, then bytes a
/// driver took for the load past the free room beside the guarantee (the
/// Vulkan gate's growth sized to the room, then the load's own commit, 0.9
/// MB over budget less the guarantee and held there by the block
/// threshold), leave the free room at the guarantee, the bank's cache
/// yielding for them, and the bank sheds what they took though it is under
/// a block, then holds.
#[test]
fn the_loads_own_commit_never_spends_the_guarantee() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        let at = |reading: u32| start + Duration::from_millis(100) * reading;
        read(&ledger, at(1));
        read(&ledger, at(2));
        read(&ledger, at(3));
        // The bank follows its target beside the guarantee.
        let mut reading = 4;
        loop {
            match domain.follow_cache_targets(&mut bindings, wake()).unwrap() {
                Followed::Holds => break,
                Followed::Shrank(BankStep::Begun(_)) | Followed::Grew(BankStep::Begun(_)) => while domain.advance_expert_bank(wake()).unwrap().is_none() {},
                other => panic!("{backend:?}: the bank follows its target at once: {other:?}"),
            }
            read(&ledger, at(reading));
            reading += 1;
            assert!(reading < 16, "{backend:?}: the bank settles on its target");
        }
        let estimate = ledger.estimate();
        assert!(estimate.free() >= estimate.guarantee(), "{backend:?}: the bank leaves the guarantee free");
        // The room beside the guarantee and the cache worth less than the
        // bank's, held; then a driver takes half a slot for the load,
        // never refused.
        let _room = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, estimate.free() - estimate.guarantee() + cheaper_than_the_bank(&ledger))).expect("the room beside the guarantee");
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let commit = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, 1)).expect("a byte");
        ledger.change(|_| { let entry = commit.entry(); entry.took(&entry.change(), row / 2, seismic::memory::Wiring::Pageable) });
        assert!(before - bank() > 0 && before - bank() < row, "{backend:?}: the bank's cache gave the commit under a block");
        let estimate = ledger.estimate();
        assert!(estimate.free() >= estimate.guarantee(), "{backend:?}: the bank's cache gave way, the guarantee unspent");
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}: the bank sheds what the load took of it");
        while domain.advance_expert_bank(wake()).unwrap().is_none() {}
        read(&ledger, at(reading));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: and holds");
        let estimate = ledger.estimate();
        assert!(estimate.free() >= estimate.guarantee(), "{backend:?}: within budget less the guarantee");
    }
}

/// A drafted domain with an idle drafter of 4 KiB held as cache, and
/// optionally a resident `holder` (`RequestId(1)`) occupying the request slot.
fn holding_an_idle_drafter(
    test: &str,
    open_holder: bool,
) -> (
    std::path::PathBuf,
    Box<crate::domain::ExecutorDomain>,
    crate::domain::StateBindings,
    Ledger,
) {
    use crate::memory::ledger::CacheKind;
    let (directory, _, mut domain, mut bindings) = crate::domain::fixtures::loaded::drafted(test);
    for request in [crate::RequestId(1), crate::RequestId(2)] {
        domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
    }
    if open_holder {
        assert!(domain.open_served(&mut bindings, crate::RequestId(1), None).unwrap().is_empty());
    }
    let ledger = domain.resources.ledgers().device().clone();
    let weights = ledger
        .cached_with(CacheKind::Drafter, 4096, crate::memory::ledger::Reclaim::Deferred)
        .expect("the drafter fits");
    let loader = match std::mem::replace(&mut domain.head_component, super::OptionalSlot::Absent) {
        super::OptionalSlot::Unloaded(loader) => loader,
        _ => panic!("drafted fixture has an unloaded head loader"),
    };
    domain.head_component = super::OptionalSlot::Idle {
        loader,
        cache: super::IdleCache { weights: vec![weights], binding: Vec::new() },
    };
    (directory, domain, bindings, ledger)
}

/// An open keeps the idle optional components in its change's keep phase,
/// before its reservation claims; refused short of room, they idle again
/// as they were, whole, and the open holds nothing.
#[test]
fn an_open_refused_short_idles_the_components_it_kept() {
    use crate::memory::ledger::{CacheKind, CommittedKind};
    let (directory, mut domain, mut bindings, ledger) = holding_an_idle_drafter("refused-open-idles-kept", true);
    let (room, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::MIN, u64::MAX));
    assert!(
        matches!(domain.open_served(&mut bindings, crate::RequestId(2), None), Err(crate::domain::DomainError::Capacity(_))),
        "nothing is left for a second request"
    );
    assert!(domain.resident(crate::RequestId(2)).is_none(), "a refused open holds nothing");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Drafter), 4096, "the drafter idles whole as it was");
    assert!(domain.head_component.idle_cache().is_some_and(|idle| !idle.taken()));
    drop(room);
    drop(domain);
    let _ = std::fs::remove_dir_all(directory);
}

/// An exogenous supply drop shrinking the bank suppresses immediate small
/// bounce-back regrows during the 3 s cooldown (stopping 10 ms ping-pong),
/// while letting the bank regrow after the cooldown expires or immediately
/// when released memory jumps past the rebound margin.
#[test]
fn exogenous_shrink_damps_ping_pong_until_cooldown_or_large_release() {
    use crate::domain::ProgramFamily as _;
    let read_host = |ledger: &Ledger, free_beyond_target: i64, at: Instant| {
        let sample = seismic::HostSample {
            at,
            page_bytes: 16 * 1024,
            total: TOTAL,
            supply: seismic::Supply { free_beyond_target, clean_cache: 0, cold_reclaimable: 0 },
            counted: seismic::Counted { at, lag: Duration::ZERO },
            cold: seismic::ColdPages::default(),
            own_reclaimed: Some(seismic::OwnReclaimed::NONE),
            wired: None,
            harm: Some(seismic::Harm { machine: 0, own: 0 }),
            pressure: Some(seismic::PressureLevel::Normal),
            lag: NO_LAG,
        };
        ledger.read(ledger.sampled(at), Reading::host(&sample));
    };
    let pending_move = |d: &crate::domain::ExecutorDomain<crate::domain::fixtures::TestFamily>| {
        let bank = d.family.expert_bank().expect("bank");
        d.next_bank_move(bank).unwrap()
    };
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        // Set baseline headroom leaving bank comfortable at 4 slots (1 slot free), then establish calm baseline.
        let mut t = Instant::now();
        read_host(&ledger, row as i64, t);
        t += Duration::from_millis(100);
        read_host(&ledger, row as i64, t);
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}");

        // Exogenous deficit (-2.5 rows) drops hold_room -> bank shrinks below 4 slots.
        t += Duration::from_millis(100);
        read_host(&ledger, -((row * 5 / 2) as i64), t);
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        while domain.advance_expert_bank(wake()).unwrap().is_none() {}
        let shrunk_to = domain.family.expert_bank().expect("bank").slots();
        assert!(shrunk_to < 4, "{backend:?}: shrunk_to={shrunk_to}");

        // Within 100ms, exogenous pressure recovers modestly (enough to regrow,
        // still below the rebound margin): 3s cooldown suppresses ping-pong.
        t += Duration::from_millis(100);
        read_host(&ledger, (row * 2) as i64, t);
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Holds, "{backend:?}: ping-pong suppressed during cooldown");
        assert_eq!(pending_move(&domain), None, "{backend:?}");

        // After > 3s calm availability at the same room, cooldown expires cleanly.
        t += Duration::from_secs(4);
        read_host(&ledger, (row * 2) as i64, t);
        assert!(
            matches!(
                pending_move(&domain),
                Some((crate::BankHolding::Experts, crate::experts::derived::BankMove::Grow { to })) if to > shrunk_to
            ),
            "{backend:?}: regrows past shrunk_to={shrunk_to} after cooldown"
        );
    }

    // Large exogenous release (>= rebound margin) bypasses the 3s cooldown immediately.
    let backend = bank_backends()[0];
    let (mut domain, mut bindings) = banked(backend, 4, 4);
    let row = bank_row(&domain);
    let ledger = domain.resources.ledgers().device().clone();
    let mut t = Instant::now();
    read_host(&ledger, row as i64, t);
    t += Duration::from_millis(100);
    read_host(&ledger, -((row * 5 / 2) as i64), t);
    assert_eq!(domain.follow_cache_targets(&mut bindings, wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)));
    while domain.advance_expert_bank(wake()).unwrap().is_none() {}
    t += Duration::from_millis(50);
    read_host(&ledger, (row * 6) as i64, t);
    assert_eq!(
        pending_move(&domain),
        Some((crate::BankHolding::Experts, crate::experts::derived::BankMove::Grow { to: 4 })),
        "large exogenous release bypasses cooldown immediately"
    );
}

/// A component an earlier claim took part of is no longer whole: the next
/// open keeps none of it and unloads it whole.
#[test]
fn an_open_unloads_a_component_an_earlier_claim_took() {
    use crate::memory::ledger::{CacheKind, Frees, Relief};
    let (directory, mut domain, mut bindings, ledger) = holding_an_idle_drafter("open-unloads-taken", false);
    let plan = ledger.plan_relief(1, Relief::CacheOnly, Frees::Room).expect("the drafter yields");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, 1);
    assert!(domain.head_component.idle_cache().is_some_and(super::IdleCache::taken), "relief took the drafter's cache");
    assert!(domain.open_served(&mut bindings, crate::RequestId(2), None).unwrap().is_empty(), "it opens");
    assert!(domain.head_component.idle_cache().is_none(), "the drafter is unloaded whole");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Drafter), 0);
    drop(domain);
    let _ = std::fs::remove_dir_all(directory);
}

/// A pass the host builds runs up to the regime's depth: no request
/// fronts before a pass is timed, whether the bank is partial or whole.
#[test]
fn a_built_pass_fronts_nothing_before_a_pass_is_timed() {
    use crate::RequestId;
    let requests = [RequestId(1), RequestId(2), RequestId(3)];
    for whole in [false, true] {
        let pricing = super::TwoRowPricing {
            measured: None,
            prices: crate::domain::fixtures::load_costs(None).prices,
            depth: 1,
            rows: crate::CHAINED_ROWS,
            verifies: 3,
            whole,
            overhead: None,
        };
        assert_eq!(
            pricing.built(&requests, &requests).unwrap(),
            crate::FrontPass::Built { form: crate::BuiltForm::Front, depths: std::collections::BTreeMap::new() },
            "whole={whole}"
        );
    }
}

/// Whole-bank and zero-miss partial-bank loads price verify passes against
/// plain decode on resident compute, verify stand rates, and verify turn
/// overhead, and transition seamlessly to and from streamed front passes as
/// the bank shrinks or grows.
#[test]
fn whole_bank_and_zero_miss_loads_price_verify_passes_and_transition_cleanly() {
    use crate::RequestId;
    use magnitude_cost_model::{Form, Needed, PathTerm, Terms, Waste};
    let requests = [RequestId(1)];
    let prices = crate::domain::fixtures::load_costs(None).prices;
    let compute = |seconds: f64| Terms { compute: seconds, drive: 0., link: 0., cpu: 0., memory: 0. };
    let mut plain_terms = [None; crate::experts::DEPTHS];
    plain_terms[0] = Some(compute(0.032));
    let zero_miss = crate::experts::Measured {
        depth: 0,
        verifying: false,
        terms: plain_terms,
        verify_terms: [None; crate::experts::DEPTHS],
        paths: None,
        stall: None,
        needed: [[None; crate::experts::DEPTHS]; 2],
        stand: None,
        accepts: [None; crate::experts::DEPTHS - 1],
        fresh: [None; crate::experts::DEPTHS - 1],
        held: None,
    };
    let pricing = |measured: crate::experts::Measured, whole: bool, verifies: usize, overhead: Option<f64>| super::TwoRowPricing {
        measured: Some(measured),
        prices: prices.clone(),
        depth: 1,
        rows: crate::CHAINED_ROWS,
        verifies,
        whole,
        overhead,
    };

    // 1. Whole bank after the first plain pass: no misses, no shadows yet ->
    //    regime chooses Verify (never Front), built pass clamps to depth 1
    //    (running + 1) because the previous pass was a front/plain pass.
    let whole_initial = pricing(zero_miss.clone(), true, 4, None);
    let regime = whole_initial.regime(1, 1).unwrap().expect("whole bank prices after first pass");
    assert_eq!(regime.depth, 0, "whole bank never runs front rows");
    assert!(matches!(regime.pass, Form::Verify(d) if d >= 1), "whole bank chooses Verify: {:?}", regime.pass);
    assert_eq!(
        whole_initial.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Verify, depths: std::collections::BTreeMap::from([(RequestId(1), 1)]) }
    );

    // 2. Whole bank after a verify pass with high acceptance and measured turn
    //    overhead: unclamped verify depth > 1.
    let mut high_accept = zero_miss.clone();
    high_accept.depth = 1;
    high_accept.verifying = true;
    high_accept.verify_terms[1] = Some(compute(0.034));
    high_accept.accepts = [Some(0.9); crate::experts::DEPTHS - 1];
    let whole_verifying = pricing(high_accept.clone(), true, 4, Some(0.005));
    let built = whole_verifying.built(&requests, &requests).unwrap();
    match built {
        crate::FrontPass::Built { form: crate::BuiltForm::Verify, depths } => {
            let depth = depths.get(&RequestId(1)).copied().unwrap_or(0);
            assert!(depth > 1 && depth <= 4, "expected verify depth in 2..=4, got {depth}");
        }
        other => panic!("expected BuiltForm::Verify, got {other:?}"),
    }

    // 3. Whole bank with low acceptance and turn overhead falls back to plain
    //    decode (Front(0), empty depths).
    let mut low_accept = high_accept.clone();
    low_accept.accepts = [Some(0.05); crate::experts::DEPTHS - 1];
    let whole_low = pricing(low_accept, true, 4, Some(0.020));
    let regime = whole_low.regime(1, 1).unwrap().expect("whole bank prices at low acceptance");
    assert_eq!(regime.pass, Form::Front(0));
    assert_eq!(
        whole_low.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Front, depths: std::collections::BTreeMap::new() }
    );

    // 4. Whole bank with verifies == 0 produces plain decode.
    let whole_no_verify = pricing(zero_miss.clone(), true, 0, None);
    assert_eq!(
        whole_no_verify.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Front, depths: std::collections::BTreeMap::new() }
    );

    // 5. Partial bank with zero misses over the window also prices Verify
    //    instead of returning None.
    let partial_zero_miss = pricing(high_accept.clone(), false, 4, Some(0.005));
    let regime = partial_zero_miss.regime(1, 1).unwrap().expect("zero-miss partial bank prices");
    assert_eq!(regime.depth, 0);
    assert!(matches!(regime.pass, Form::Verify(_)), "{:?}", regime.pass);

    // 6. Streamed <-> whole transition: drive-bound partial bank chooses
    //    Front(k), and once whole == true it immediately ignores stale
    //    streaming misses and chooses Verify.
    let drive = PathTerm {
        seconds: 1e-3,
        wasted: Waste { front: vec![Some(0.); crate::experts::DEPTHS], verify: vec![Some(0.); crate::experts::DEPTHS] },
    };
    let needed_row: [Option<Needed>; crate::experts::DEPTHS] =
        std::array::from_fn(|k| Some(Needed::Executed(245. - 4. * k as f64)));
    let mut streamed = high_accept;
    streamed.paths = Some(vec![drive]);
    streamed.needed = [needed_row, needed_row];
    streamed.stand = Some(0.95);
    let partial_drive = pricing(streamed.clone(), false, 4, Some(0.005));
    let regime = partial_drive.regime(1, 1).unwrap().expect("drive-bound partial bank prices");
    assert!(matches!(regime.pass, Form::Front(d) if d >= 1), "drive-bound partial bank fronts: {:?}", regime.pass);
    let grown_whole = pricing(streamed, true, 4, Some(0.005));
    let regime = grown_whole.regime(1, 1).unwrap().expect("grown whole bank prices");
    assert_eq!(regime.depth, 0, "whole bank drops stale streaming misses");
    assert!(regime.regime.paths.is_empty(), "whole bank clears path terms");
    assert!(matches!(regime.pass, Form::Verify(d) if d >= 1), "whole bank verifies: {:?}", regime.pass);
}

/// A whole banked domain still returns `TwoRowPricing` so verify passes are
/// priced instead of trapping the load in 0-draft plain decode.
#[test]
fn a_whole_banked_domain_prices_two_row_passes() {
    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        crate::domain::fixtures::measure_load(&mut domain, &mut bindings);
        let pricing = domain
            .two_row_pricing()
            .unwrap()
            .expect("a whole banked domain prices two-row passes");
        assert!(pricing.whole, "{backend:?}: 4/4 bank is whole");
    }
}

/// Distinct device (VRAM) and host (RAM) ledgers for multi-store bank move tests.
fn discrete_ledgers(at: Instant) -> (Ledger, Ledger) {
    use crate::memory::ledger::{Ledgers, LoadShape};
    let scope = Ledgers::new();
    let shape = std::sync::Arc::new(LoadShape { stores: Vec::new() });
    let device = scope.ledger(seismic::StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([1; 16])), shape.clone());
    let host = scope.ledger(seismic::StoreIdentity::HostRam, shape);
    device.read(device.sampled(at), Reading::device(TOTAL, TOTAL - OTHERS, NO_LAG));
    host.read(host.sampled(at), Reading::device(TOTAL, TOTAL - OTHERS, NO_LAG));
    (device, host)
}

/// A sub-noise partial grow on `BankHolding::Experts` (`gained < min_partial`)
/// must not shadow an eligible grow on `BankHolding::Host`, and shrink-before-grow
/// priority across holdings is preserved.
#[test]
fn sub_noise_experts_grow_does_not_starve_host_grow() {
    use crate::domain::BankHolding;
    use crate::experts::derived::{BankMove, BankShape, HoldingTarget};
    use crate::planning::BankTier;

    let slot_bytes: u64 = 16 << 20;
    let block_bytes: u64 = 2 << 20;
    let (device, host) = discrete_ledgers(Instant::now());
    let ledger_of = |tier| match tier {
        BankTier::Bank => &device,
        BankTier::Host => &host,
    };

    // sat_charge = 512 * 16 MiB = 8 GiB -> sat_charge / 64 = 128 MiB = min_partial.
    // Growing Experts 100 -> 101 gains 16 MiB (< 128 MiB) while still below saturation (512).
    let experts = HoldingTarget {
        holding: BankHolding::Experts,
        tier: BankTier::Bank,
        at: 100,
        owed: 0,
        shape: BankShape::new(block_bytes, 8, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };
    let host_target = HoldingTarget {
        holding: BankHolding::Host,
        tier: BankTier::Host,
        at: 146,
        owed: 0,
        shape: BankShape::new(block_bytes, 0, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };

    // Sanity check: Experts alone is filtered out by min_partial.
    assert_eq!(super::select_bank_move(&[(&experts, 101)], ledger_of, None), None);
    // With Host eligible to grow, Host is selected instead of returning None.
    assert_eq!(
        super::select_bank_move(&[(&experts, 101), (&host_target, 512)], ledger_of, None),
        Some((BankHolding::Host, BankMove::Grow { to: 512 })),
    );
    // Shrink on a later holding still beats an eligible grow on an earlier holding.
    assert_eq!(
        super::select_bank_move(&[(&experts, 512), (&host_target, 100)], ledger_of, None),
        Some((BankHolding::Host, BankMove::Shrink { to: 100 })),
    );
}

/// An `ExogenousShrink` cooldown on `DeviceMemory` suppresses `BankHolding::Experts`
/// regrow without starving `BankHolding::Host` on `HostRam`.
#[test]
fn exogenous_shrink_cooldown_on_device_does_not_starve_host_grow() {
    use crate::domain::BankHolding;
    use crate::experts::derived::{BankMove, BankShape, HoldingTarget};
    use crate::planning::BankTier;

    let slot_bytes: u64 = 16 << 20;
    let block_bytes: u64 = 2 << 20;
    let t0 = Instant::now();
    let (device, host) = discrete_ledgers(t0);
    let est = device.estimate();
    let shrink = super::ExogenousShrink {
        store: device.store(),
        at: est.at().at(),
        committed: est.committed(),
        hold_room: est.hold_room(),
        block: block_bytes,
    };
    // Advance within the 3 s cooldown with unchanged headroom/committed bytes.
    read(&device, t0 + Duration::from_millis(100));

    let ledger_of = |tier| match tier {
        BankTier::Bank => &device,
        BankTier::Host => &host,
    };
    let experts = HoldingTarget {
        holding: BankHolding::Experts,
        tier: BankTier::Bank,
        at: 100,
        owed: 0,
        shape: BankShape::new(block_bytes, 8, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };
    let host_target = HoldingTarget {
        holding: BankHolding::Host,
        tier: BankTier::Host,
        at: 146,
        owed: 0,
        shape: BankShape::new(block_bytes, 0, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };

    // Sanity check: saturation grow on Experts is suppressed by DeviceMemory cooldown.
    assert_eq!(super::select_bank_move(&[(&experts, 512)], ledger_of, Some(&shrink)), None);
    // Host on HostRam is unaffected by DeviceMemory's cooldown and is selected to grow.
    assert_eq!(
        super::select_bank_move(&[(&experts, 512), (&host_target, 512)], ledger_of, Some(&shrink)),
        Some((BankHolding::Host, BankMove::Grow { to: 512 })),
    );
}
