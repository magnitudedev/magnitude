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
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        // The prompt's claim takes room from the bank's cache: synchronous bank
        // reclaim shrinks the bank below its 4-slot target before admitting the claim.
        let room = ledger.estimate().target();
        let prompt = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, room - row * 3 / 2)).expect("the prompt's claim fits");
        let shrunk = crate::domain::ElasticBank::slots(crate::domain::fixtures::fake_bank_mut(&mut domain));
        assert!(shrunk < 4, "{backend:?}: synchronous reclaim shrank the bank below 4 (got {shrunk})");
        // The prompt ends: its claim is freed, and the next boundary grows
        // the bank back to its full 4-slot target.
        drop(prompt);
        read(&ledger, at(4));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Grew(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        let grown = loop {
            if let Some((resized, _)) = domain.advance_expert_bank(wake()).unwrap() {
                break resized;
            }
        };
        assert!(matches!(grown, BankResized::Grown { from, to: 4, .. } if from == shrunk), "{backend:?}: {grown:?}");
        read(&ledger, at(5));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: the full bank again");
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
/// granule a slot here) synchronously reclaims one whole slot before
/// returning the claim, so the claim's allocation is admitted immediately
/// without waiting for a separate async shrink.
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
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        let estimate = ledger.estimate();
        assert_eq!(estimate.guarantee(), 0, "{backend:?}: no rise: every free byte is a claim's");
        let free = estimate.free();
        let spare = estimate.cache_of(crate::memory::ledger::CacheKind::Spare);
        let room = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, free + spare)).expect("free room and spare cache");
        ledger.change(|_| { let entry = room.entry(); entry.take(&entry.change(), free, seismic::memory::Wiring::Pageable) }).unwrap_or_else(|refused| panic!("{backend:?}: free room allocates: {refused}"));
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let slots_before = crate::domain::ElasticBank::slots(crate::domain::fixtures::fake_bank_mut(&mut domain));
        let bytes = row / 2;
        let claim = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, bytes)).expect("the bank's cache synchronously reclaims a slot for the claim");
        assert_eq!(before - bank(), row, "{backend:?}: one whole bank slot reclaimed for the sub-slot claim");
        assert_eq!(crate::domain::ElasticBank::slots(crate::domain::fixtures::fake_bank_mut(&mut domain)), slots_before - 1, "{backend:?}: bank shrank by 1 slot");
        let allocate = || ledger.change(|_| { let entry = claim.entry(); entry.take(&entry.change(), bytes, seismic::memory::Wiring::Pageable) });
        allocate().unwrap_or_else(|refused| panic!("{backend:?}: the reclaimed slot admits the claim immediately: {refused}"));
    }
}

/// Cache a shortage takes in-flight to fit the budget as others move owes no
/// claim: a logical yield of half a slot (under the bank's block) to the
/// room's own movement keeps the bank as it is, its threshold judging it as it
/// judges the room, as the Metal gate's steady decode needs.
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
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: the bank holds its full target");
        // Cache worth no more than the bank's yields first; past it, half a
        // slot of the bank's (in-flight logical yield, below one block).
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let plan = ledger.plan_relief(cheaper_than_the_bank(&ledger) + row / 2, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("the cache covers the shortage");
        ledger.change(|keeping| ledger.execute(&keeping.claims(), plan, crate::memory::ledger::ReliefBoundary::InFlight));
        let fell = before - bank();
        assert!(fell > 0 && fell < row, "{backend:?}: the shortage took {fell} bytes of the bank's cache, under a block");
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: within a block, the bank keeps its slots");
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
            match domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap() {
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
        // Drain cheaper-than-bank cache (`Spare` + `GraphSlots`) first so the
        // bank cannot regrow into it after shedding a slot for the load's commit.
        let cheaper = cheaper_than_the_bank(&ledger);
        if cheaper > 0 {
            if let Some(plan) = ledger.plan_relief(cheaper, Relief::CacheOnly, crate::memory::ledger::Frees::Room) {
                ledger.change(|keeping| ledger.execute(&keeping.claims(), plan, crate::memory::ledger::ReliefBoundary::InFlight));
            }
        }
        let estimate = ledger.estimate();
        // The room beside the guarantee (leaving 1 byte for the load's claim)
        // held; then a driver takes half a slot for the load, never refused.
        let _room = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, (estimate.free() - estimate.guarantee()).saturating_sub(1))).expect("the room beside the guarantee");
        let bank = || ledger.estimate().cache_of(crate::memory::ledger::CacheKind::Bank);
        let before = bank();
        let commit = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, crate::memory::ledger::CommittedKind::Load, 1)).expect("a byte");
        ledger.change(|_| { let entry = commit.entry(); entry.took(&entry.change(), row * 3 / 4, seismic::memory::Wiring::Pageable) });
        assert!(before - bank() > 0 && before - bank() < row, "{backend:?}: the bank's cache gave the commit under a block");
        let estimate = ledger.estimate();
        assert!(estimate.free() >= estimate.guarantee(), "{backend:?}: the bank's cache gave way, the guarantee unspent");
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}: the bank sheds what the load took of it");
        while domain.advance_expert_bank(wake()).unwrap().is_none() {}
        read(&ledger, at(reading));
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}: and holds");
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

/// After an exogenous supply drop shrinks the bank, subsequent room recovery
/// regrows immediately from the ledger's valuation alone (zero cooldown
/// hysteresis).
#[test]
fn bank_regrows_immediately_from_ledger_valuation_after_shrink() {
    use crate::domain::ProgramFamily as _;
    let read_host = |ledger: &Ledger, free_beyond_target: i64, at: Instant| {
        let sample = seismic::HostSample {
            at,
            page_bytes: 16 * 1024,
            total: TOTAL,
            supply: seismic::Supply { free_beyond_target, clean_cache: 0, cold_reclaimable: 0 },
            reclaim_band: 0,
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
        let mut t = Instant::now();
        read_host(&ledger, row as i64, t);
        // Settle the initial device-to-host fixture reading's pulse envelope
        // before injecting the exogenous deficit.
        t += Duration::from_secs(600);
        read_host(&ledger, row as i64, t);
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Holds, "{backend:?}");

        // Exogenous deficit (-2.5 rows) drops hold_room -> bank shrinks below 4 slots.
        t += Duration::from_millis(100);
        read_host(&ledger, -((row * 5 / 2) as i64), t);
        assert_eq!(domain.follow_cache_targets(&mut bindings, wake(), wake()).unwrap(), Followed::Shrank(BankStep::Begun(crate::planning::BankTier::Bank)), "{backend:?}");
        while domain.advance_expert_bank(wake()).unwrap().is_none() {}
        let shrunk_to = domain.family.expert_bank().expect("bank").slots();
        assert!(shrunk_to < 4, "{backend:?}: shrunk_to={shrunk_to}");

        // As soon as room recovers past the 3.5-row pulse envelope (100 ms later),
        // the bank regrows immediately from the ledger's valuation without cooldown hysteresis.
        t += Duration::from_millis(100);
        read_host(&ledger, (row * 6) as i64, t);
        assert!(
            matches!(
                pending_move(&domain),
                Some((crate::BankHolding::Experts, crate::experts::derived::BankMove::Grow { to })) if to > shrunk_to
            ),
            "{backend:?}: regrows immediately past shrunk_to={shrunk_to} without hysteresis"
        );
    }
}

/// A component an earlier claim took part of is no longer whole: the next
/// open keeps none of it and unloads it whole.
#[test]
fn an_open_unloads_a_component_an_earlier_claim_took() {
    use crate::memory::ledger::{CacheKind, Frees, Relief};
    let (directory, mut domain, mut bindings, ledger) = holding_an_idle_drafter("open-unloads-taken", false);
    let plan = ledger.plan_relief(1, Relief::CacheOnly, Frees::Room).expect("the drafter yields");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan, crate::memory::ledger::ReliefBoundary::Idle)).yielded, 1);
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
        layers: 48,
        depth: 0,
        verifying: false,
        terms: plain_terms,
        verify_terms: [None; crate::experts::DEPTHS],
        verify_samples: [0; crate::experts::DEPTHS],
        plain_streak: 0,
        paths: None,
        stall: None,
        front_stall: [None; crate::experts::DEPTHS - 1],
        verify_stall: [None; crate::experts::DEPTHS - 1],
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
        overhead: overhead.map(magnitude_cost_model::VerifyOverhead::fixed),
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
    high_accept.verify_samples[1] = crate::experts::MIN_VERIFY_SAMPLES;
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
    //    decode (Front(0), empty depths -> BuiltForm::Plain).
    let mut low_accept = high_accept.clone();
    low_accept.accepts = [Some(0.05); crate::experts::DEPTHS - 1];
    let whole_low = pricing(low_accept, true, 4, Some(0.020));
    let regime = whole_low.regime(1, 1).unwrap().expect("whole bank prices at low acceptance");
    assert_eq!(regime.pass, Form::Front(0));
    assert_eq!(
        whole_low.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Plain, depths: std::collections::BTreeMap::new() }
    );

    // 4. Whole bank with verifies == 0 produces plain decode.
    let whole_no_verify = pricing(zero_miss.clone(), true, 0, None);
    assert_eq!(
        whole_no_verify.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Plain, depths: std::collections::BTreeMap::new() }
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

    let (device, host) = discrete_ledgers(Instant::now());
    let ledger_of = |tier| match tier {
        BankTier::Bank => &device,
        BankTier::Host => &host,
    };

    // Sanity check: Experts alone is filtered out by min_partial.
    assert_eq!(super::select_bank_move(&[(&experts, 101)], ledger_of, &[None; BankTier::COUNT]), None);
    // With Host eligible to grow, Host is selected instead of returning None.
    assert_eq!(
        super::select_bank_move(&[(&experts, 101), (&host_target, 512)], ledger_of, &[None; BankTier::COUNT]),
        Some((BankHolding::Host, BankMove::Grow { to: 512 })),
    );
    // Shrink on a later holding still beats an eligible grow on an earlier holding.
    assert_eq!(
        super::select_bank_move(&[(&experts, 512), (&host_target, 100)], ledger_of, &[None; BankTier::COUNT]),
        Some((BankHolding::Host, BankMove::Shrink { to: 100 })),
    );
}

/// A `GrowthRefusal` latch on `BankTier::Bank` suppresses repeated allocator
/// growth retries until memory on that store is released or `free`/`budget`
/// genuinely increases by at least one block unit, while never starving
/// `BankTier::Host` growth.
#[test]
fn growth_refusal_latches_store_until_release_or_room_grows_without_starving_other_tier() {
    use crate::domain::{BankHolding, GrowthRefusal};
    use crate::experts::derived::{BankMove, BankShape, HoldingTarget};
    use crate::memory::ledger::CommittedKind;
    use crate::planning::BankTier;

    let slot_bytes: u64 = 16 << 20;
    let block_bytes: u64 = 8 << 20;
    let t0 = Instant::now();
    let (device, host) = discrete_ledgers(t0);
    // Hold one committed claim and allocate its bytes so we can test releasing allocated bytes later.
    let held = device
        .change(|keeping| device.claim_at(&keeping.claims(), None, CommittedKind::Arena, block_bytes))
        .expect("committed claim fits");
    device
        .change(|_| held.entry().take(&held.entry().change(), block_bytes, seismic::memory::Wiring::Pageable))
        .expect("take committed bytes");
    // Refresh device reading with `free = TOTAL - OTHERS - block_bytes`.
    let t1 = t0 + Duration::from_millis(50);
    device.read(device.sampled(t1), Reading::device(TOTAL, TOTAL - OTHERS - block_bytes, NO_LAG));
    let est = device.estimate();
    let mut refused = [None; BankTier::COUNT];
    refused[BankTier::Bank.index()] = Some(GrowthRefusal {
        allocated: est.allocated(),
        free: est.free(),
        budget: est.budget(),
    });

    let ledger_of = |tier| match tier {
        BankTier::Bank => &device,
        BankTier::Host => &host,
    };
    let experts = HoldingTarget {
        holding: BankHolding::Experts,
        tier: BankTier::Bank,
        at: 57,
        owed: 0,
        shape: BankShape::new(block_bytes, 8, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };
    let host_target = HoldingTarget {
        holding: BankHolding::Host,
        tier: BankTier::Host,
        at: 98,
        owed: 0,
        shape: BankShape::new(block_bytes, 0, 512, 512),
        charge: Box::new(move |slots| slots * slot_bytes),
    };

    // Unchanged store: Experts growth is latched (no retry spam).
    assert_eq!(super::select_bank_move(&[(&experts, 512)], ledger_of, &refused), None);
    // Sub-unit jitter (< block_bytes) still stays latched.
    let t2 = t1 + Duration::from_millis(50);
    device.read(device.sampled(t2), Reading::device(TOTAL, TOTAL - OTHERS - block_bytes + (block_bytes - 1), NO_LAG));
    assert_eq!(super::select_bank_move(&[(&experts, 512)], ledger_of, &refused), None);
    // Host on BankTier::Host is NOT starved while BankTier::Bank is latched.
    assert_eq!(
        super::select_bank_move(&[(&experts, 512), (&host_target, 512)], ledger_of, &refused),
        Some((BankHolding::Host, BankMove::Grow { to: 512 })),
    );

    // Budget/free rising by >= one block unit opens the latch.
    let t3 = t2 + Duration::from_millis(50);
    device.read(device.sampled(t3), Reading::device(TOTAL, TOTAL - OTHERS, NO_LAG));
    assert_eq!(
        super::select_bank_move(&[(&experts, 512)], ledger_of, &refused),
        Some((BankHolding::Experts, BankMove::Grow { to: 512 })),
    );

    // Reset reading back to refusal baseline, then release >= block_bytes of own allocation.
    let t4 = t3 + Duration::from_millis(50);
    device.read(device.sampled(t4), Reading::device(TOTAL, TOTAL - OTHERS - block_bytes, NO_LAG));
    assert_eq!(super::select_bank_move(&[(&experts, 512)], ledger_of, &refused), None);
    device.change(|_| {
        held.entry().freed(&held.entry().change(), block_bytes, seismic::memory::Wiring::Pageable);
        drop(held);
    });
    assert_eq!(
        super::select_bank_move(&[(&experts, 512)], ledger_of, &refused),
        Some((BankHolding::Experts, BankMove::Grow { to: 512 })),
    );
}

/// Post-domain settled baseline (`read_after(..., true)`, as called before
/// `family.begin_host_backing` in `ExecutorDomain::with_family`) absorbs one-time uncharged load/warm-up
/// host overhead into `others` and `pinned_budget()` without poisoning
/// `guarantee()` (`read.rise`), so `pinned_import_room()` and
/// `live_pinned_headroom()` do not double-subtract warm-up overhead.
/// Subsequent serving-time rises (`after_idle = false`) still set `guarantee()`.
#[test]
fn post_warmup_settled_read_absorbs_load_overhead_without_poisoning_pinned_headroom() {
    use crate::memory::ledger::{Ledgers, LoadShape, PoolClass};
    use std::num::NonZeroU64;
    use std::sync::Arc;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    const TOTAL_64G: u64 = 62 * GIB;
    const RECLAIM_BAND: u64 = 64 * MIB;
    const BLOCK: u64 = 130 * MIB;
    const RESERVE: u64 = RECLAIM_BAND + BLOCK;
    const WARMUP_UNCHANGED_RSS: u64 = 5_583_413_248; // 5.20 GiB observed in 7aa835

    let sample_at = |at: Instant, free_beyond_target: i64, clean_cache: u64| seismic::HostSample {
        at,
        page_bytes: 4096,
        total: TOTAL_64G,
        supply: seismic::Supply { free_beyond_target, clean_cache, cold_reclaimable: 0 },
        reclaim_band: RECLAIM_BAND,
        counted: seismic::Counted { at, lag: Duration::ZERO },
        cold: seismic::ColdPages::default(),
        own_reclaimed: None,
        wired: None,
        harm: Some(seismic::Harm { machine: 0, own: 0 }),
        pressure: Some(seismic::PressureLevel::Normal),
        lag: NO_LAG,
    };

    let scope = Ledgers::new();
    let host = scope.ledger(seismic::StoreIdentity::HostRam, Arc::new(LoadShape { stores: Vec::new() }));
    let _pool = host.pool_block(PoolClass::Records, NonZeroU64::new(BLOCK).unwrap());

    // Opening read before model load / warm-up.
    let t0 = Instant::now();
    let s0 = sample_at(t0, (48 * GIB) as i64, 9 * GIB);
    host.read(host.sampled(t0), Reading::host(&s0));

    // Model load + warm-up adds 5.20 GiB of uncharged host/driver RSS.
    // Settled post-warm-up baseline read (`after_idle = true`) absorbs it into
    // `others` and `pinned_budget()` with `guarantee() == 0`.
    let t1 = t0 + Duration::from_secs(5);
    let s1 = sample_at(t1, (48 * GIB) as i64 - WARMUP_UNCHANGED_RSS as i64, 9 * GIB);
    let post_warmup = host.read_after(host.sampled(t1), Reading::host(&s1), true);
    assert_eq!(post_warmup.guarantee(), 0, "one-time warm-up RSS must not poison guarantee");
    assert_eq!(post_warmup.others(), 5 * GIB + WARMUP_UNCHANGED_RSS);
    let expected_post_warmup_imm = (48 * GIB - WARMUP_UNCHANGED_RSS) + 9 * GIB;
    assert_eq!(post_warmup.pinned_import_room(), expected_post_warmup_imm - RESERVE);

    // Late in host-tier backing: free_beyond_target = -3.02 GiB (cgroup_floor
    // already subtracted), clean_cache = +8.55 GiB -> immediate_supply = +5.53 GiB.
    let t2 = t1 + Duration::from_secs(5);
    let late = sample_at(t2, -3_245_690_880, 9_185_488_896);
    let expected_late_imm = (9_185_488_896_i64 - 3_245_690_880_i64) as u64;
    assert_eq!(
        post_warmup.live_pinned_headroom(&late),
        expected_late_imm - RESERVE,
        "live_pinned_headroom must expose immediate_supply - pinned_reserve without double-subtracting warm-up RSS"
    );

    // During live serving (`after_idle = false`), a genuine 1 GiB competing-process
    // rise still sets `guarantee() == 1 GiB` and is preserved by `live_pinned_headroom`.
    let t3 = t2 + Duration::from_secs(2);
    let s3 = sample_at(t3, (47 * GIB) as i64 - WARMUP_UNCHANGED_RSS as i64, 9 * GIB);
    let serving = host.read(host.sampled(t3), Reading::host(&s3));
    assert_eq!(serving.guarantee(), GIB);
    assert_eq!(serving.live_pinned_headroom(&late), expected_late_imm - RESERVE - GIB);
}

#[test]
fn release_yielded_never_returns_pending_without_active_async_release() {
    let (mut domain, mut bindings) = crate::domain::fixtures::fixture(None);
    let yielded = domain
        .release_yielded(&mut bindings, crate::CompletionWake::new(|| {}))
        .unwrap();
    if matches!(yielded, crate::domain::Yielded::Pending) {
        assert!(
            domain.releasing_slots(),
            "Yielded::Pending requires an active async release that will wake the owner"
        );
    } else {
        assert!(!domain.releasing_slots());
    }
}

/// A slow `Verify(15)` turn on a banked draft head (`2 ms + 15 * 15 ms = 227 ms`)
/// must NOT poison `TwoRowPricing` into collapsing to `Front(0)`: depth-scaled
/// `VerifyOverhead` prices `Verify(2..=5)` at `2 ms + k * 15 ms` and chooses
/// `BuiltForm::Verify` at `k in 2..=5`.
#[test]
fn two_row_pricing_scales_verify_overhead_with_depth_instead_of_collapsing_to_front0() {
    use crate::RequestId;
    use magnitude_cost_model::{Form, Terms, VerifyOverhead};
    let requests = [RequestId(1)];
    let prices = crate::domain::fixtures::load_costs(None).prices;
    let compute = |seconds: f64| Terms { compute: seconds, drive: 0., link: 0., cpu: 0., memory: 0. };
    let mut terms = [None; crate::experts::DEPTHS];
    let mut verify_terms = [None; crate::experts::DEPTHS];
    let mut verify_samples = [0; crate::experts::DEPTHS];
    for k in 0..crate::experts::DEPTHS {
        let secs = 0.060 + 0.005 * k as f64;
        terms[k] = Some(compute(secs));
        if k > 0 {
            verify_terms[k] = Some(compute(secs));
            verify_samples[k] = crate::experts::MIN_VERIFY_SAMPLES;
        }
    }
    let accepts_vals = [0.85, 0.80, 0.75, 0.70, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05];
    let mut accepts = [None; crate::experts::DEPTHS - 1];
    for (slot, &val) in accepts.iter_mut().zip(&accepts_vals) {
        *slot = Some(val);
    }
    let measured = crate::experts::Measured {
        layers: 48,
        depth: 15.min(crate::experts::DEPTHS - 1),
        verifying: true,
        terms,
        verify_terms,
        verify_samples,
        plain_streak: 0,
        paths: None,
        stall: None,
        front_stall: [None; crate::experts::DEPTHS - 1],
        verify_stall: [None; crate::experts::DEPTHS - 1],
        needed: [[None; crate::experts::DEPTHS]; 2],
        stand: Some(0.85),
        accepts,
        fresh: [None; crate::experts::DEPTHS - 1],
        held: None,
    };
    let pricing = |overhead: VerifyOverhead| super::TwoRowPricing {
        measured: Some(measured.clone()),
        prices: prices.clone(),
        depth: 1,
        rows: crate::CHAINED_ROWS,
        verifies: crate::experts::DEPTHS - 1,
        whole: true,
        overhead: Some(overhead),
    };

    // Flat 227 ms scalar collapses to Front(0).
    let flat = pricing(VerifyOverhead::fixed(0.227));
    assert_eq!(flat.regime(1, 1).unwrap().unwrap().pass, Form::Front(0));

    // Depth-scaled `2 ms + k * 15 ms` selects Verify(2..=5).
    let scaled = pricing(VerifyOverhead { fixed: 0.002, per_step: 0.015 });
    let front_regime = scaled.regime(1, 1).unwrap().unwrap();
    assert!(
        matches!(front_regime.pass, Form::Verify(k) if (2..=5).contains(&k)),
        "expected Verify(2..=5), got {:?}",
        front_regime.pass
    );
    match scaled.built(&requests, &requests).unwrap() {
        crate::FrontPass::Built { form: crate::BuiltForm::Verify, depths } => {
            let depth = depths.get(&RequestId(1)).copied().unwrap_or(0);
            assert!((2..=5).contains(&depth), "expected depth in 2..=5, got {depth}");
        }
        other => panic!("expected BuiltForm::Verify, got {other:?}"),
    }
}

/// A freshly retained KV prefix (`hits == 0`) has already paid its prefill cost
/// and must receive a non-zero recompute value (`hits.max(1)`), so memory relief
/// evicts lower-value `Bank` cache before touching the active conversation's
/// prefix cache.
#[test]
fn freshly_retained_zero_hit_prefix_is_valued_ahead_of_lower_value_bank_slot() {
    use crate::domain::fixtures::{misses_at, price_growth};
    use crate::memory::ledger::{CacheKind, Frees, Relief, ReliefBoundary};

    for backend in bank_backends() {
        let (mut domain, mut bindings) = banked(backend, 4, 4);
        let row = bank_row(&domain);
        let ledger = domain.resources.ledgers().device().clone();
        let start = Instant::now();
        read(&ledger, start + Duration::from_millis(100));
        read(&ledger, start + Duration::from_millis(200));

        // Price both the bank's decode savings and the load's prefill rate,
        // and record one decode step so the bank has a positive (non-zero) value.
        price_growth(&mut domain, &mut bindings, misses_at(2, 10));
        domain.served_round(true);

        // Hold a freshly retained prefix in `state_slabs` (`CacheKind::Prefix`).
        let prefix_bytes = domain.retained_entry_bytes();
        assert!(prefix_bytes > 0, "{backend:?}: retained prefix entry has non-zero bytes");
        ledger.change(|_| domain.state_slabs.settle(&ledger, prefix_bytes, prefix_bytes, 0, 0));
        assert_eq!(ledger.estimate().cache_of(CacheKind::Prefix), prefix_bytes, "{backend:?}");

        // Value caches with a freshly retained prefix (`hits == 0`).
        domain.value_cache(Some((10_000, 0))).unwrap();

        // Relief past cheaper cache (Spare/GraphSlots) by half a bank slot must
        // take from the lower-value Bank and leave the zero-hit Prefix untouched.
        let bank_before = ledger.estimate().cache_of(CacheKind::Bank);
        assert!(bank_before >= row, "{backend:?}: bank holds cache above floor");
        let need = cheaper_than_the_bank(&ledger) + row / 2;
        let plan = ledger
            .plan_relief(need, Relief::CacheOnly, Frees::Room)
            .expect("cache covers the relief");
        ledger.change(|keeping| ledger.execute(&keeping.claims(), plan, ReliefBoundary::Idle));

        assert_eq!(
            ledger.estimate().cache_of(CacheKind::Prefix),
            prefix_bytes,
            "{backend:?}: freshly retained zero-hit prefix must not be evicted before lower-value bank"
        );
        assert_eq!(domain.prefix_owed(), 0, "{backend:?}: prefix owes nothing");
        assert!(
            ledger.estimate().cache_of(CacheKind::Bank) < bank_before,
            "{backend:?}: lower-value bank yields before freshly retained prefix"
        );
    }
}

/// A single cold `200 ms` `Verify(1)` sample must NOT permanently lock
/// `TwoRowPricing` into `BuiltForm::Plain`: until `MIN_VERIFY_SAMPLES` (`2`)
/// verify passes are recorded, `build_resident(true)` extrapolates from the
/// plain prior (`terms[0]`). And even when `>= 2` slow verify samples do push
/// `Regime::pass()` to `Form::Front(0)`, `PLAIN_REPROBE_INTERVAL` consecutive
/// plain passes trigger a `Verify(1)` re-probe so a transient spike can never
/// permanently disable speculative drafting.
#[test]
fn single_cold_verify_sample_does_not_lock_out_drafting_and_plain_streak_reprobes() {
    use crate::RequestId;
    use magnitude_cost_model::{Form, Terms};
    let requests = [RequestId(1)];
    let prices = crate::domain::fixtures::load_costs(None).prices;
    let compute = |seconds: f64| Terms { compute: seconds, drive: 0., link: 0., cpu: 0., memory: 0. };
    let mut terms = [None; crate::experts::DEPTHS];
    terms[0] = Some(compute(0.030));
    let mut verify_terms = [None; crate::experts::DEPTHS];
    verify_terms[1] = Some(compute(0.200));
    let mut verify_samples = [0; crate::experts::DEPTHS];
    verify_samples[1] = 1;
    let mut measured = crate::experts::Measured {
        layers: 48,
        depth: 1,
        verifying: true,
        terms,
        verify_terms,
        verify_samples,
        plain_streak: 0,
        paths: None,
        stall: None,
        front_stall: [None; crate::experts::DEPTHS - 1],
        verify_stall: [None; crate::experts::DEPTHS - 1],
        needed: [[None; crate::experts::DEPTHS]; 2],
        stand: None,
        accepts: [Some(0.85); crate::experts::DEPTHS - 1],
        fresh: [None; crate::experts::DEPTHS - 1],
        held: None,
    };
    let make_pricing = |measured: crate::experts::Measured| super::TwoRowPricing {
        measured: Some(measured),
        prices: prices.clone(),
        depth: 1,
        rows: crate::CHAINED_ROWS,
        verifies: 4,
        whole: true,
        overhead: None,
    };

    // 1. Single cold 200 ms Verify(1) sample (`verify_samples[1] == 1`) is
    //    ignored in favor of the LaunchPrices-scaled `terms[0]` prior -> Verify.
    let single_cold = make_pricing(measured.clone());
    let front = single_cold.regime(1, 1).unwrap().unwrap();
    assert!(
        matches!(front.pass, Form::Verify(d) if d >= 1),
        "single cold 200 ms Verify(1) must not lock out drafting: {:?}",
        front.pass
    );
    assert!(
        matches!(
            single_cold.built(&requests, &requests).unwrap(),
            crate::FrontPass::Built { form: crate::BuiltForm::Verify, ref depths }
                if depths.get(&RequestId(1)).copied().unwrap_or(0) >= 1
        ),
        "built pass must continue verifying after single cold sample"
    );

    // 2. Two slow 200 ms Verify(1) samples (`verify_samples[1] == 2`) are
    //    proven -> regime chooses Plain (`Form::Front(0)`) while `plain_streak`
    //    is below `PLAIN_REPROBE_INTERVAL`.
    measured.verify_samples[1] = crate::experts::MIN_VERIFY_SAMPLES;
    measured.depth = 0;
    measured.verifying = false;
    measured.plain_streak = crate::experts::PLAIN_REPROBE_INTERVAL - 1;
    let locked = make_pricing(measured.clone());
    assert_eq!(locked.regime(1, 1).unwrap().unwrap().pass, Form::Front(0));
    assert_eq!(
        locked.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Plain, depths: std::collections::BTreeMap::new() }
    );

    // 3. After `PLAIN_REPROBE_INTERVAL` consecutive plain passes, `built()`
    //    re-probes `Verify(1)` so a transient spike cannot permanently lock out
    //    drafting.
    measured.plain_streak = crate::experts::PLAIN_REPROBE_INTERVAL;
    let reprobing = make_pricing(measured);
    assert_eq!(
        reprobing.built(&requests, &requests).unwrap(),
        crate::FrontPass::Built { form: crate::BuiltForm::Verify, depths: std::collections::BTreeMap::from([(RequestId(1), 1)]) }
    );
}
