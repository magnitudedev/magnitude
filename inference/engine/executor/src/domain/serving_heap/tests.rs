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

/// A tiny domain whose `holder` opened on the request slot, with `idle`, an
/// idle drafter of 4 KiB, held as cache beside it.
fn holding_an_idle_drafter() -> (crate::domain::ExecutorDomain<crate::domain::fixtures::TestFamily>, crate::domain::StateBindings, Ledger) {
    use crate::domain::fixtures::{fixture_over, limits, tiny_definition, tiny_manifest};
    use crate::memory::ledger::CacheKind;
    let (mut domain, mut bindings) = fixture_over(
        tiny_definition(),
        tiny_manifest(&tiny_definition()),
        limits(2, 1, false),
        crate::PlannedMethod::Plain,
        None,
        false,
        crate::platform::DeviceRequest::Automatic,
    );
    for request in [crate::RequestId(1), crate::RequestId(2)] {
        domain.install_input(request, magnitude_family_contracts::PreparedModelInput::continuation_only()).unwrap();
    }
    assert!(domain.open_served(&mut bindings, crate::RequestId(1), None).unwrap().is_empty());
    let ledger = domain.resources.ledgers().device().clone();
    let weights = ledger.cached(CacheKind::Drafter, 4096).expect("the drafter fits");
    domain.idle_components.push(super::IdleComponent { component: super::OptionalComponent::Drafter, weights: vec![weights], binding: Vec::new() });
    (domain, bindings, ledger)
}

/// An open keeps the idle optional components in its change's keep phase,
/// before its reservation claims; refused short of room, they idle again
/// as they were, whole, and the open holds nothing.
#[test]
fn an_open_refused_short_idles_the_components_it_kept() {
    use crate::memory::ledger::{CacheKind, CommittedKind};
    let (mut domain, mut bindings, ledger) = holding_an_idle_drafter();
    let (room, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::MIN, u64::MAX));
    assert!(
        matches!(domain.open_served(&mut bindings, crate::RequestId(2), None), Err(crate::domain::DomainError::Capacity(_))),
        "nothing is left for a second request"
    );
    assert!(!domain.state_claims.contains_key(&crate::RequestId(2)), "a refused open holds nothing");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Drafter), 4096, "the drafter idles whole as it was");
    assert!(matches!(&domain.idle_components[..], [idle] if idle.component == super::OptionalComponent::Drafter && !idle.taken()));
    drop(room);
}

/// A component an earlier claim took part of is no longer whole: the next
/// open keeps none of it and unloads it whole.
#[test]
fn an_open_unloads_a_component_an_earlier_claim_took() {
    use crate::memory::ledger::{CacheKind, CommittedKind};
    let (mut domain, mut bindings, ledger) = holding_an_idle_drafter();
    domain.close(crate::RequestId(1)).unwrap();
    let (room, _) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, std::num::NonZeroU64::MIN, u64::MAX));
    let taken = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, 1)).expect("the drafter yields");
    assert!(domain.idle_components[0].taken(), "the claim took the drafter's cache");
    drop(room);
    assert!(domain.open_served(&mut bindings, crate::RequestId(2), None).unwrap().is_empty(), "it opens");
    assert!(domain.idle_components.is_empty(), "the drafter is unloaded whole");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Drafter), 0);
    drop(taken);
}
