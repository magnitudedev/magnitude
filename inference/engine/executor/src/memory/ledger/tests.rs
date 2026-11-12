//! The ledger's catalog failures, each a state the ledger has no
//! representation for. No OS: readings are handed in.

use super::*;
use std::num::NonZeroU64;
use std::time::Duration;

const GB: u64 = 1 << 30;

/// A load's shape: one store of one per-token history domain of 1024 f32
/// values a row, at 1024 tokens of context.
fn shape() -> Arc<LoadShape> {
    let component = magnitude_state::ComponentDescriptor::new(
        magnitude_state::LayerRef::Target(0),
        magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
        1,
    )
    .expect("a component");
    let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
    Arc::new(LoadShape { stores: vec![StoreShape { domains, context: 1024, max_advance: 16 }] })
}

/// `sequence`'s state growing from `from` whole pages of [`shape`]'s
/// history to `to`: a sequence's state grows a page at a time.
fn pages(sequence: u64, from: u64, to: u64) -> Need {
    let shape = shape();
    let store = &shape.stores[0];
    let page = store.domains[0].geometry(&store.domains, store.context, store.max_advance).expect("a geometry").page_rows as u64;
    Need::StateRows { sequence: RequestId(sequence), from: from * page, to: to * page }
}

/// A scope of its own and host RAM's ledger in it.
fn host() -> (Ledgers, Ledger) {
    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    (scope, ledger)
}

/// Readings a reading interval apart, from one start.
struct Clock(Instant, u32);

impl Clock {
    fn new() -> Self {
        Self(Instant::now(), 0)
    }

    fn next(&mut self) -> Instant {
        self.1 += 1;
        self.0 + Duration::from_millis(100) * self.1
    }
}

/// Read `ledger` at `total` and `available`, a driver's grant letting the
/// process allocate `grant` in all.
fn read(ledger: &Ledger, clock: &mut Clock, total: u64, available: u64, grant: Option<u64>) {
    let at = clock.next();
    let snapshot = ledger.snapshot();
    let left = grant.map(|grant| grant.saturating_sub(snapshot.allocated()));
    ledger.read(|| Ok(Available::new(total, available, left, at).expect("a reading")));
}

/// A committed claim of exactly `bytes`, sized without any formula, and
/// allocated: memory the process made.
fn commit(ledger: &Ledger, bytes: u64) -> Claim {
    let mut claim = ledger.claim(Need::Measurement { bytes }).expect("the claim fits");
    claim.allocated(bytes);
    claim
}

/// Items 1, 31, 62: our own free is never other programs' memory: it
/// moves nothing until the next reading, which shows it, so it never
/// shrinks anything twice and never unloads.
#[test]
fn an_own_free_is_never_others_memory() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let weights = commit(&ledger, 40 * GB);
    let bank = ledger.cache(CacheKind::Bank, 20 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    let before = ledger.snapshot();
    assert_eq!(before.others(), Some(0));
    drop(bank);
    let freed = ledger.snapshot();
    assert_eq!((freed.others(), freed.budget(), freed.guarantee(0)), (Some(0), before.budget(), 0));
    assert!(!freed.unload(0));
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let shown = ledger.snapshot();
    assert_eq!((shown.others(), shown.budget(), shown.guarantee(0)), (Some(0), before.budget(), 0));
    drop(weights);
}

/// Bytes a GPU fence still reads stay held in their own entry until the
/// fence completes and the free returns: what the ledger holds covers them,
/// so no claim takes them meanwhile.
#[test]
fn a_fenced_free_stays_held_until_its_fence_completes() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 16 * GB, 16 * GB, None);
    let retired = ledger.cache(CacheKind::Bank, 10 * GB).expect("free room");
    // The fence is in flight: the holder keeps its handle.
    let fenced = retired;
    assert_eq!(ledger.snapshot().cache(), 10 * GB);
    assert!(ledger.cache(CacheKind::Prefix, 7 * GB).is_err(), "the fenced bytes are not free");
    // The fence completes; the free returns.
    drop(fenced);
    assert_eq!(ledger.snapshot().cache(), 0);
    assert!(ledger.cache(CacheKind::Prefix, 7 * GB).is_ok());
}

/// Items 2, 25: memory the process claims and backs is never read as other
/// programs taking it, so it sets no guarantee and moves no budget.
#[test]
fn own_claims_are_never_others_taking_memory() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    let idle = ledger.snapshot();
    assert_eq!(idle.others(), Some(24 * GB));
    let mut held = Vec::new();
    for step in 1..=10 {
        held.push(commit(&ledger, 2 * GB));
        read(&ledger, &mut clock, 64 * GB, 40 * GB - step * 2 * GB, None);
        let backed = ledger.snapshot();
        assert_eq!(backed.others(), Some(24 * GB));
        assert_eq!(backed.guarantee(0), 0);
        assert_eq!(backed.budget(), idle.budget());
    }
}

/// Item 23: the guarantee is the chunk's mandatory growth and others' rise
/// between the last two readings, never a forecast scaled by our own
/// release latency.
#[test]
fn the_guarantee_is_others_rise_and_the_chunk_only() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let bank = ledger.cache(CacheKind::Bank, 20 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    drop(bank);
    // Our free, shown: no term of it.
    read(&ledger, &mut clock, 64 * GB, 34 * GB, None);
    assert_eq!(ledger.snapshot().guarantee(3 * GB), 3 * GB);
    // Another program takes 2 GB in one interval.
    read(&ledger, &mut clock, 64 * GB, 32 * GB, None);
    assert_eq!(ledger.snapshot().guarantee(3 * GB), 5 * GB);
}

/// Items 35, 43: the budget is what the process holds plus what the OS
/// has available, bounded by a grant: never others' pages the OS would
/// compress or swap, never a figure counted twice.
#[test]
fn the_budget_is_held_and_available() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 48 * GB, 48 * GB, None);
    let _weights = commit(&ledger, 12 * GB);
    let _prefix = ledger.cache(CacheKind::Prefix, 2 * GB).expect("free room");
    read(&ledger, &mut clock, 48 * GB, 10 * GB, None);
    let snapshot = ledger.snapshot();
    assert_eq!((snapshot.committed(), snapshot.cache()), (12 * GB, 2 * GB));
    assert_eq!(snapshot.budget(), 12 * GB + 2 * GB + 10 * GB);
    assert_eq!(snapshot.others(), Some(48 * GB - 10 * GB - 14 * GB));
    read(&ledger, &mut clock, 48 * GB, 10 * GB, Some(20 * GB));
    assert_eq!(ledger.snapshot().budget(), 20 * GB);
}

/// Items 53, 61: other programs' memory lowers the budget as they take it
/// and raises it as they let it go, by the same reading; nothing of ours
/// moves it.
#[test]
fn others_memory_moves_the_budget_both_ways() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    read(&ledger, &mut clock, 64 * GB, 44 * GB, None);
    let alone = ledger.snapshot();
    assert_eq!(alone.others(), Some(0));
    read(&ledger, &mut clock, 64 * GB, 19 * GB, None);
    let beside = ledger.snapshot();
    assert_eq!(beside.others(), Some(25 * GB));
    assert_eq!(beside.budget(), alone.budget() - 25 * GB);
    assert_eq!(beside.guarantee(0), 25 * GB);
    read(&ledger, &mut clock, 64 * GB, 19 * GB, None);
    assert_eq!(ledger.snapshot().guarantee(0), 0);
    read(&ledger, &mut clock, 64 * GB, 44 * GB, None);
    let after = ledger.snapshot();
    assert_eq!(after.budget(), alone.budget());
    assert_eq!(after.guarantee(0), 0);
}

/// Items 10, 19: a store has one ledger in a scope, whoever asks for it,
/// and readings from several holders never mix.
#[test]
fn a_store_has_one_ledger() {
    let scope = Ledgers::new();
    let one = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    let two = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    let mut clock = Clock::new();
    read(&two, &mut clock, 64 * GB, 64 * GB, None);
    let _claim = commit(&one, 5 * GB);
    assert_eq!(two.snapshot().committed(), 5 * GB);
    read(&two, &mut clock, 64 * GB, 30 * GB, None);
    let stale = clock.0 + Duration::from_millis(50);
    one.read(|| Ok(Available::new(64 * GB, 1 * GB, None, stale).expect("a reading")));
    assert_eq!(one.snapshot().others(), Some(64 * GB - 30 * GB - 5 * GB));
    let other = Ledgers::new().ledger(seismic::StoreIdentity::HostRam, shape());
    assert_eq!(other.snapshot().committed(), 0);
}

/// Item 20: every figure of a decision comes from one reading.
#[test]
fn a_snapshot_is_one_reading() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(50 * GB));
    let _claim = commit(&ledger, 8 * GB);
    read(&ledger, &mut clock, 64 * GB, 40 * GB, Some(50 * GB));
    let snapshot = ledger.snapshot();
    let _more = commit(&ledger, 4 * GB);
    read(&ledger, &mut clock, 64 * GB, 10 * GB, Some(50 * GB));
    let others = snapshot.others().expect("not blind");
    assert_eq!(snapshot.committed(), 8 * GB);
    assert_eq!(others, 64 * GB - 40 * GB - 8 * GB);
    assert_eq!(snapshot.budget(), (64 * GB - others).min(50 * GB));
}

/// Item 46: the process's own memory never unloads it. The budget is
/// what it holds plus what the OS has available, under a grant whose
/// ceiling is what it holds plus what the grant leaves, so committed memory
/// never exceeds it; a claim past it is refused instead. A blind store
/// unloads.
#[test]
fn a_full_store_refuses_and_only_blindness_unloads() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(32 * GB));
    let _weights = commit(&ledger, 30 * GB);
    // Other programs take everything else, and the grant leaves nothing.
    read(&ledger, &mut clock, 64 * GB, 34 * GB, Some(30 * GB));
    read(&ledger, &mut clock, 64 * GB, 34 * GB, Some(30 * GB));
    let full = ledger.snapshot();
    assert_eq!(full.budget(), 30 * GB);
    assert!(!full.unload(0));
    assert!(matches!(ledger.claim(Need::Measurement { bytes: 1 }), Err(Refused::Short(Short::Over { .. }))));
    ledger.read(|| Err(Blind("no reading".into())));
    assert!(ledger.snapshot().unload(0));
}

/// Item 46: others squeezing the load unload it at the reading where the
/// guarantee cannot be kept: free room below others' rise, the cache at its
/// floor and nothing preemptible.
#[test]
fn a_squeeze_past_the_guarantee_unloads() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 50 * GB);
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    assert!(!ledger.snapshot().unload(0));
    // Others take 12 GB in one interval: 2 GB stay free.
    read(&ledger, &mut clock, 64 * GB, 2 * GB, None);
    let squeezed = ledger.snapshot();
    assert_eq!(squeezed.guarantee(0), 12 * GB);
    assert!(squeezed.unload(0));
    // Others stop rising: the guarantee falls to zero, and the load stays.
    read(&ledger, &mut clock, 64 * GB, 2 * GB, None);
    assert!(!ledger.snapshot().unload(0));
}

/// The same squeeze beside a later sequence: preemption has something to
/// take, so a claim preempts it and nothing unloads.
#[test]
fn a_squeeze_with_preemptible_state_preempts_instead() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 40 * GB);
    let mut first = ledger.claim(pages(1, 0, 1)).expect("room");
    let page = first.bytes();
    let _second = ledger.claim(pages(2, 0, (8 * GB).div_ceil(page))).expect("room");
    let held = ledger.snapshot().committed();
    read(&ledger, &mut clock, 64 * GB, 64 * GB - held, None);
    read(&ledger, &mut clock, 64 * GB, 1 * GB, None);
    let squeezed = ledger.snapshot();
    assert!(squeezed.guarantee(0) > 1 * GB);
    assert!(!squeezed.unload(0), "the second sequence is preemptible");
    let Err(Refused::Preempt(preemption)) = first.grow(pages(1, 1, 1 + (2 * GB).div_ceil(page))) else {
        panic!("the growth preempts the second sequence");
    };
    assert_eq!(preemption.sequences(), [RequestId(2)]);
}

/// Four sequences of one page each beside a bank filling the rest of a
/// 40 GB grant, and the page's bytes.
fn four_sequences(ledger: &Ledger) -> ([Claim; 4], Cached, u64) {
    let mut clock = Clock::new();
    read(ledger, &mut clock, 64 * GB, 64 * GB, Some(40 * GB));
    let claims = [1, 2, 3, 4].map(|sequence| ledger.claim(pages(sequence, 0, 1)).expect("room"));
    let page = claims[0].bytes();
    let bank = ledger.cache(CacheKind::Bank, ledger.snapshot().budget() - 4 * page).expect("the rest");
    (claims, bank, page)
}

/// A sequence's growth takes free room, then cache, then the state of the
/// sequences admitted after it, the latest first and just enough, once
/// they are evicted; never an earlier sequence's.
#[test]
fn a_claim_preempts_only_later_sequences_after_the_cache() {
    let (_scope, ledger) = host();
    let ([mut first, second, third, fourth], _bank, page) = four_sequences(&ledger);
    let bank = ledger.snapshot().cache();
    let grown = bank / page + 2;
    let Err(Refused::Preempt(preemption)) = first.grow(pages(1, 1, 1 + grown)) else {
        panic!("the growth needs preemption");
    };
    assert_eq!(preemption.sequences(), [RequestId(4), RequestId(3)]);
    assert_eq!(ledger.snapshot().cache(), 0, "the cache went to the growth");
    drop((fourth, third));
    first.complete(preemption);
    assert_eq!(first.bytes(), (1 + grown) * page);
    assert_eq!(ledger.snapshot().cache_of(CacheKind::Spare), bank + 2 * page - grown * page);
    assert_eq!(second.bytes(), page, "an earlier sequence is kept");
    // The second sequence can never take the first's.
    assert!(ledger.admit(&pages(2, 1, 1 + grown + 8)).is_err());
}

/// An invariant broken under the ledger's lock panics as itself: the
/// handles dropped while unwinding never touch the ledger it poisoned, so
/// the test binary goes on.
#[test]
#[should_panic(expected = "a preemption completes only once every sequence it named is evicted")]
fn an_invariant_broken_under_the_lock_panics_as_itself() {
    let (_scope, ledger) = host();
    let ([mut first, _second, _third, _fourth], _bank, page) = four_sequences(&ledger);
    let grown = ledger.snapshot().cache() / page + 1;
    let Err(Refused::Preempt(preemption)) = first.grow(pages(1, 1, 1 + grown)) else {
        panic!("the growth needs preemption");
    };
    // The fourth sequence is still held.
    first.complete(preemption);
}

/// Two claims race for one eviction's bytes: the preempting claim holds
/// them from the moment it names the sequence, so it wins with exactly one
/// eviction, and the other claim never takes them.
#[test]
fn a_preemption_holds_its_evictions_bytes_against_a_racing_claim() {
    let (_scope, ledger) = host();
    let ([mut first, mut second, third, fourth], _bank, page) = four_sequences(&ledger);
    let grown = ledger.snapshot().cache() / page + 1;
    let Err(Refused::Preempt(preemption)) = first.grow(pages(1, 1, 1 + grown)) else {
        panic!("the growth needs preemption");
    };
    assert_eq!(preemption.sequences(), [RequestId(4)]);
    // The second sequence races for the fourth's bytes: they are reserved,
    // so it can only name the third.
    let Err(Refused::Preempt(racing)) = second.grow(pages(2, 1, 2)) else {
        panic!("the race finds no free room");
    };
    assert_eq!(racing.sequences(), [RequestId(3)]);
    drop(racing);
    drop(fourth);
    first.complete(preemption);
    assert_eq!(first.bytes(), (1 + grown) * page);
    assert_eq!(third.bytes(), page, "one eviction only");
}

/// A claim shrinks by bytes its holder freed and claims no more: its bytes
/// and what it allocated fall together, the store's free room rising by
/// them.
#[test]
fn a_claim_shrinks_by_what_its_holder_freed() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 60 * GB, None);
    let (mut claim, units) = ledger.claim_units(CommittedKind::GroupClass, NonZeroU64::new(GB).unwrap(), 3);
    assert_eq!(units, 3);
    claim.allocated(3 * GB);
    let free = ledger.snapshot().free();
    claim.shrink(GB);
    assert_eq!((claim.bytes(), claim.allocated_bytes()), (2 * GB, 2 * GB));
    assert_eq!(ledger.snapshot().free(), free + GB);
}

/// A claim takes cache synchronously and is refused only where committed
/// memory with it exceeds the budget; admission is the same check.
#[test]
fn a_claim_takes_cache_and_is_refused_only_past_the_budget() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let _bank = ledger.cache(CacheKind::Bank, 20 * GB).expect("free room");
    // Another program holds 20 GB: 4 GB are free.
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    let need = Need::Measurement { bytes: 10 * GB };
    assert_eq!(ledger.admit(&need), Ok(()));
    let _taken = ledger.claim(need).expect("cache yields to it");
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.committed(), 30 * GB);
    // The free 4 GB first, then 6 GB of the bank's.
    assert_eq!(snapshot.cache(), 14 * GB);
    let past = Need::Measurement { bytes: 15 * GB };
    let short = Short::Over { store: ledger.store(), required: 15 * GB, over: GB };
    assert_eq!(ledger.admit(&past), Err(short.clone()));
    assert!(matches!(ledger.claim(past), Err(Refused::Short(refused)) if refused == short));
}

/// A load's allocation is the user's own action and counts the clean file
/// cache beside the free memory above the OS's floor, once: the claim holds
/// it, so the next claim finds it held. A growth (a measurement's room, a
/// backing) counts only the free memory.
#[test]
fn a_loads_claim_takes_clean_file_cache_once_and_a_growth_never() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    let at = clock.next();
    // Other programs hold 54 GB: 4 GB are free above the floor, and 6 GB of
    // clean file cache lie beside them.
    ledger.read(|| Ok(Available::new(64 * GB, 4 * GB, None, at).expect("a reading").with_reclaimable(6 * GB)));
    assert_eq!(ledger.snapshot().free(), 4 * GB);
    assert_eq!(ledger.snapshot().free_for(seismic::Reclaim::CleanFileCache), 10 * GB);
    // A growth takes no file cache.
    assert!(ledger.admit(&Need::Measurement { bytes: 5 * GB }).is_err());
    assert!(ledger.backing(5 * GB).is_err());
    assert!(ledger.covers(5 * GB, seismic::Reclaim::Nothing).is_err());
    assert_eq!(ledger.covers(10 * GB, seismic::Reclaim::CleanFileCache), Ok(()));
    // A load's claim counts it, up to the file cache and no further.
    assert!(ledger.admit(&Need::BankFloor { bytes: 11 * GB }).is_err());
    let floor = ledger.claim(Need::BankFloor { bytes: 10 * GB }).expect("a load counts the clean file cache");
    // Once: the claim holds all of it.
    assert!(ledger.admit(&Need::BankFloor { bytes: 1 }).is_err());
    assert_eq!(ledger.snapshot().free_for(seismic::Reclaim::CleanFileCache), 0);
    drop(floor);
    assert_eq!(ledger.snapshot().free_for(seismic::Reclaim::CleanFileCache), 10 * GB);
}

/// A reading's clean file cache is never more than the room it leaves in
/// the total.
#[test]
fn a_readings_file_cache_never_exceeds_what_it_leaves() {
    let at = Instant::now();
    assert_eq!(Available::new(10 * GB, 8 * GB, None, at).expect("a reading").with_reclaimable(5 * GB).reclaimable(), 2 * GB);
}

/// A unit claim takes the most whole units free room, spare granules and
/// bank cache cover, up to its most, yielding spare before the bank; it
/// never yields a prefix, a drafter or vision, never preempts a later
/// sequence, holds nothing with nothing to take, and frees what it held
/// as it drops.
#[test]
fn a_unit_claim_takes_whole_units_from_free_spare_and_bank_alone() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(40 * GB));
    let unit = NonZeroU64::new(GB).unwrap();
    let state = ledger.claim(pages(2, 0, 1)).expect("room");
    let page = state.bytes();
    let held = |claim: &Claim| claim.bytes();
    // Free room alone, capped by `most`.
    let (few, units) = ledger.claim_units(CommittedKind::Scratch, unit, 3);
    assert_eq!((units, held(&few)), (3, 3 * GB));
    drop(few);
    // Fill the store: some free room, spare, bank and a prefix.
    let free = ledger.snapshot().budget() - ledger.snapshot().committed();
    let prefix = ledger.cache(CacheKind::Prefix, 10 * GB).expect("room");
    let spare = ledger.cache(CacheKind::Spare, 3 * GB).expect("room");
    let bank = ledger.cache(CacheKind::Bank, free - 10 * GB - 3 * GB - GB / 2).expect("room");
    let bank_bytes = bank.bytes();
    // Free room is half a unit: whole units come from spare, then the bank.
    let (scratch, taken) = ledger.claim_units(CommittedKind::Scratch, unit, u64::MAX);
    assert_eq!(taken, (GB / 2 + 3 * GB + bank_bytes) / GB);
    assert_eq!(held(&scratch), taken * GB);
    assert_eq!(spare.bytes(), 0, "spare yields first");
    assert_eq!(bank.bytes(), bank_bytes + 3 * GB + GB / 2 - taken * GB, "the bank yields the rest");
    assert_eq!(prefix.bytes(), 10 * GB, "a prefix never yields to it");
    assert_eq!(held(&state), page, "no sequence is preempted");
    // Nothing left to take: it holds none.
    let (none, units) = ledger.claim_units(CommittedKind::Scratch, unit, 5);
    assert_eq!((units, held(&none)), (0, 0));
    drop(none);
    let committed = ledger.snapshot().committed();
    drop(scratch);
    assert_eq!(ledger.snapshot().committed(), committed - taken * GB, "dropping it frees its bytes");
}

/// A store whose reading failed admits a load's allocation as blind, never
/// as a shortage of zero available bytes; once it is read, the same
/// allocation is admitted or short by its figures.
#[test]
fn a_blind_store_refuses_an_allocation_as_blind_not_short() {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
    let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, shape());
    let demand = crate::memory::ClaimDemand { device: GB, ..crate::memory::ClaimDemand::default() };
    let store = ledgers.device().store();
    for ledger in [ledgers.device(), ledgers.host()] {
        ledger.read(|| Err(Blind("the OS refused the reading".into())));
    }
    let claim = |demand| ledgers.claim(&device, CommittedKind::Load, demand);
    assert_eq!(claim(demand).err(), Some(Short::Blind { store: store.clone(), reason: "the OS refused the reading".into() }));
    let mut clock = Clock::new();
    for ledger in [ledgers.device(), ledgers.host()] {
        read(ledger, &mut clock, 8 * GB, 4 * GB, None);
    }
    assert!(claim(demand).is_ok());
    let more = crate::memory::ClaimDemand { device: 5 * GB, ..demand };
    assert_eq!(claim(more).err(), Some(Short::Over { store, required: 5 * GB, over: GB }));
}

/// The laptop's load: the target import allocated the bank 16.3 GB past
/// its floor, leaving 2.4 GB available. The domain's holdings of those
/// bytes are carved from what the load recorded, never claimed again
/// against a reading that already shows them allocated: nothing is
/// refused, free room and others' memory stand, and no guarantee rises.
#[test]
fn a_loads_allocation_is_never_claimed_twice() {
    const ABOVE: u64 = 16_331_243_520;
    const AVAILABLE: u64 = 2_444_917_344;
    const FLOOR: u64 = 2 * GB;
    const TOTAL: u64 = 36 * GB;
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
    let ledgers = DeviceLedgers::open_in(&Ledgers::new(), &device, shape());
    let ledger = ledgers.device();
    let store = ledger.store();
    let mut clock = Clock::new();
    let others = TOTAL - FLOOR - ABOVE - AVAILABLE;
    read(ledger, &mut clock, TOTAL, TOTAL - others, None);
    let mut recorded = Recorded::new(&ledgers);
    let import = crate::memory::ClaimDemand { device: FLOOR + ABOVE, ..Default::default() };
    let mut held = ledgers.claim(&device, CommittedKind::Load, import).expect("the import fits");
    // The import allocates; the next reading shows it.
    held.allocated();
    recorded.record(held);
    read(ledger, &mut clock, TOTAL, AVAILABLE, None);
    let imported = ledger.snapshot();
    assert_eq!((imported.others(), imported.free(), imported.guarantee(0)), (Some(others), AVAILABLE, 0));
    let _floor = recorded.carve(&store, CommittedKind::BankFloor, FLOOR);
    let bank = recorded.carve(&store, CommittedKind::Load, ABOVE).into_cache(CacheKind::Bank);
    assert_eq!(bank.bytes(), ABOVE);
    read(ledger, &mut clock, TOTAL, AVAILABLE, None);
    let served = ledger.snapshot();
    assert_eq!((served.others(), served.free(), served.guarantee(0)), (Some(others), AVAILABLE, 0));
    assert_eq!(served.cache_of(CacheKind::Bank), ABOVE);
}

/// A claim taken but not yet allocated, such as a measurement job's 4 GiB
/// room, stays out of free room across a reading: the reading shows none
/// of it, and it never reads as free.
#[test]
fn a_claim_not_yet_allocated_stays_out_of_free_room_across_a_reading() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let room = ledger.claim(Need::Measurement { bytes: 4 * GB }).expect("room");
    assert_eq!(ledger.snapshot().free(), 20 * GB);
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let after = ledger.snapshot();
    assert_eq!((after.others(), after.free(), after.guarantee(0)), (Some(40 * GB), 20 * GB, 0));
    assert!(ledger.cache(CacheKind::Bank, 21 * GB).is_err(), "the bank never grows into the room");
    drop(room);
}

/// A closed group's class turns cache in place: one entry, its bytes and
/// their allocation unchanged, so no reading moves; a claim then takes it
/// just before the bank.
#[test]
fn a_claim_turns_cache_in_place_and_yields_before_the_bank() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let class = commit(&ledger, 4 * GB);
    let bank = ledger.cache(CacheKind::Bank, 60 * GB).expect("the rest");
    read(&ledger, &mut clock, 64 * GB, 0, None);
    let before = ledger.snapshot();
    let entries = ledger.entries().len();
    let slots = class.into_cache(CacheKind::GraphSlots);
    let after = ledger.snapshot();
    assert_eq!(ledger.entries().len(), entries, "the same entry");
    assert_eq!((after.allocated(), after.free(), after.others()), (before.allocated(), before.free(), before.others()));
    assert_eq!((after.committed(), after.cache_of(CacheKind::GraphSlots)), (before.committed() - 4 * GB, 4 * GB));
    let _taken = ledger.claim(Need::Measurement { bytes: 5 * GB }).expect("cache yields to it");
    assert_eq!((slots.bytes(), bank.bytes()), (0, 59 * GB), "the idle slots yield first, then the bank");
}

/// A blind store refuses a need, its growth and a cache holding as blind,
/// never as a shortage of the zero budget a failed reading leaves; once
/// read, the same need is held.
#[test]
fn a_blind_store_refuses_a_claim_as_blind_not_short() {
    let (_scope, ledger) = host();
    let blind = Short::Blind { store: ledger.store(), reason: "the OS refused the reading".into() };
    ledger.read(|| Err(Blind("the OS refused the reading".into())));
    let need = Need::Measurement { bytes: GB };
    assert_eq!(ledger.admit(&need), Err(blind.clone()));
    assert!(matches!(ledger.claim(need.clone()), Err(Refused::Short(refused)) if refused == blind));
    assert!(matches!(ledger.cache(CacheKind::Prefix, GB), Err(refused) if refused == blind));
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 8 * GB, 4 * GB, None);
    let mut state = ledger.claim(pages(1, 0, 1)).expect("room");
    ledger.read(|| Err(Blind("the OS refused the reading".into())));
    assert!(matches!(state.grow(pages(1, 1, 2)), Err(Refused::Short(refused)) if refused == blind));
    read(&ledger, &mut clock, 8 * GB, 4 * GB, None);
    let _held = ledger.claim(need).expect("a read store holds it");
}

/// A claim that takes cache in process allocates those bytes only as the
/// cache's holder frees them: until then they stay the process's
/// allocation, releasing, so a reading counts them neither others' nor
/// free, and what the process may allocate leaves them out.
#[test]
fn a_claim_allocates_taken_cache_only_as_its_holder_frees_it() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let mut bank = ledger.cache(CacheKind::Bank, 40 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    let _taken = ledger.claim(Need::Measurement { bytes: 10 * GB }).expect("cache yields to it");
    assert_eq!((bank.bytes(), bank.releasing()), (34 * GB, 6 * GB), "the bank yields 6 GB, still allocated");
    let taken = ledger.snapshot();
    assert_eq!((taken.allocated(), taken.releasing(), taken.allocatable()), (60 * GB, 6 * GB, 4 * GB), "only the free 4 GB are allocatable");
    // The bank has not freed them: the next reading counts them the process's.
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert_eq!(ledger.snapshot().others(), Some(0), "the yielded bytes read as others' memory");
    bank.released(6 * GB);
    let freed = ledger.snapshot();
    assert_eq!((freed.allocated(), freed.releasing(), freed.allocatable()), (54 * GB, 0, 10 * GB), "the claim allocates them once freed");
    read(&ledger, &mut clock, 64 * GB, 10 * GB, None);
    assert_eq!(ledger.snapshot().others(), Some(0));
}

/// A cache holding dropped frees what it yielded and had not freed yet.
#[test]
fn a_cache_holding_dropped_frees_what_it_yielded() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let bank = ledger.cache(CacheKind::Bank, 64 * GB).expect("free room");
    let _taken = ledger.claim(Need::Measurement { bytes: 8 * GB }).expect("cache yields to it");
    drop(bank);
    let snapshot = ledger.snapshot();
    assert_eq!((snapshot.allocated(), snapshot.releasing(), snapshot.allocatable()), (0, 0, 64 * GB));
}

/// A backing's claim holds its bytes from free room at once and is
/// allocated only as they back; ended, what it backed is cache and the
/// rest is free.
#[test]
fn a_backing_is_allocated_as_it_backs_and_ends_as_cache_of_what_it_backed() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let mut backing = ledger.backing(8 * GB).expect("free room");
    assert!(ledger.backing(57 * GB).is_err(), "its bytes are held from the start");
    assert_eq!((ledger.snapshot().allocated(), ledger.snapshot().free()), (0, 56 * GB));
    backing.backed(3 * GB);
    assert_eq!(ledger.snapshot().allocated(), 3 * GB);
    let cache = backing.into_backed_cache(5 * GB, CacheKind::Bank);
    let snapshot = ledger.snapshot();
    assert_eq!((cache.bytes(), snapshot.allocated(), snapshot.cache_of(CacheKind::Bank), snapshot.free()), (5 * GB, 5 * GB, 5 * GB, 59 * GB));
}

/// A sequence's first rows draw from the state floor the load holds, with
/// no room beside it and no window in which the floor is lost: with every
/// byte of the budget held, a sequence claims what the floor holds and
/// holds nothing more than the floor did; its next page is refused.
#[test]
fn a_sequences_first_rows_draw_from_the_state_floor() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 8 * GB, None);
    let Need::StateRows { to: page, .. } = pages(1, 0, 1) else { unreachable!() };
    let _floor = ledger.claim(Need::StateFloor { rows: page }).expect("the floor fits");
    let room = ledger.snapshot().free();
    let _rest = commit(&ledger, room);
    assert_eq!(ledger.snapshot().free(), 0, "the floor and the rest hold the whole budget");
    let held = ledger.snapshot().committed();
    let mut state = ledger.claim(pages(1, 0, 1)).expect("the first page draws from the floor");
    assert_eq!(ledger.snapshot().committed(), held, "the floor's bytes are held once");
    assert!(state.grow(pages(1, 1, 2)).is_err(), "the page past the floor has no room");
}
