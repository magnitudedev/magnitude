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
    ledger.read(ledger.sampled(at), Ok(Available::new(total, available, left).expect("a reading")));
}

/// A committed claim of exactly `bytes`, sized without any formula, and
/// allocated: memory the process made.
fn commit(ledger: &Ledger, bytes: u64) -> Claim {
    let mut claim = ledger.claim(Need::Measurement { bytes }).expect("the claim fits");
    ledger.change(|changing| claim.allocated(bytes, changing));
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
    let bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
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
    let retired = ledger.cached(CacheKind::Bank, 10 * GB).expect("free room");
    // The fence is in flight: the holder keeps its handle.
    let fenced = retired;
    assert_eq!(ledger.snapshot().cache(), 10 * GB);
    assert!(ledger.cached(CacheKind::Prefix, 7 * GB).is_err(), "the fenced bytes are not free");
    // The fence completes; the free returns.
    drop(fenced);
    assert_eq!(ledger.snapshot().cache(), 0);
    assert!(ledger.cached(CacheKind::Prefix, 7 * GB).is_ok());
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
    let bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
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
    let _prefix = ledger.cached(CacheKind::Prefix, 2 * GB).expect("free room");
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
    one.read(one.sampled(stale), Ok(Available::new(64 * GB, 1 * GB, None).expect("a reading")));
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
    ledger.read(ledger.sampled(Instant::now()), Err(Blind("no reading".into())));
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
    let bank = ledger.cached(CacheKind::Bank, ledger.snapshot().budget() - 4 * page).expect("the rest");
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
    ledger.change(|changing| claim.allocated(3 * GB, changing));
    let free = ledger.snapshot().free();
    ledger.change(|changing| claim.shrink(GB, changing));
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
    let _bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
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
    ledger.read(ledger.sampled(at), Ok(Available::new(64 * GB, 4 * GB, None).expect("a reading").with_reclaimable(6 * GB)));
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

/// A load's claim that took the clean file cache is judged, for the serving
/// claims after it, on a reading taken after its allocation: the OS then
/// refills free pages from the inactive cache, which only a new reading shows.
/// On the reading before it (1.40 GB allocated, 7.03 GB available, 39.47 GB
/// of clean file cache, others 43.10 GB) the 13.7 GB import leaves no room
/// for a serving claim past the load's floor; on one after it (about 7.27 GB
/// available) the same claim is admitted.
#[test]
fn a_serving_claim_after_a_load_is_judged_on_a_reading_taken_after_the_load() {
    const MB: u64 = 1 << 20;
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    // Total memory: others 43.10 + available 7.03 + allocated 1.40 GB.
    let total = (44_134 + 7_199 + 1_434) * MB;
    read(&ledger, &mut clock, total, total, None);
    let _before = commit(&ledger, 1_434 * MB);
    let at = clock.next();
    ledger.read(ledger.sampled(at), Ok(Available::new(total, 7_199 * MB, None).expect("a reading").with_reclaimable(40_417 * MB)));
    assert_eq!(ledger.snapshot().others(), Some(44_134 * MB));
    let mut import = ledger.claim(Need::BankFloor { bytes: 14_029 * MB }).expect("a load counts the clean file cache");
    ledger.change(|changing| import.allocated(14_029 * MB, changing));
    // The reading predates the import: its room is the free memory of then,
    // all held.
    let serving = Need::Measurement { bytes: 1_187_840 };
    assert_eq!(ledger.snapshot().free(), 0);
    assert!(ledger.admit(&serving).is_err(), "the stale reading leaves a serving claim no room");
    // The OS refilled free pages from the cache the import displaced.
    let at = clock.next();
    ledger.read(ledger.sampled(at), Ok(Available::new(total, 7_444 * MB, None).expect("a reading").with_reclaimable(32_000 * MB)));
    assert_eq!(ledger.admit(&serving), Ok(()), "a reading taken after the import shows the room");
}

/// A reading's clean file cache is never more than the room it leaves in
/// the total.
#[test]
fn a_readings_file_cache_never_exceeds_what_it_leaves() {
    assert_eq!(Available::new(10 * GB, 8 * GB, None).expect("a reading").with_reclaimable(5 * GB).reclaimable(), 2 * GB);
}

/// The user's shrink cascades (55→20 and 76→20 bank slots ~100 ms apart):
/// a reading sampled before a shrink reached the ledger after it, read the
/// freed bytes as others' and set the guarantee by them, and the next
/// shrink was larger. A shrink never raises others or the guarantee,
/// whichever side of it the reading was sampled on.
#[test]
fn a_shrink_never_raises_others_or_the_guarantee() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let mut bank: Vec<Cached> = (0..4).map(|_| ledger.cached(CacheKind::Bank, 5 * GB).expect("free room")).collect();
    // Other programs hold 4 GB.
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 50 * GB, None);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 50 * GB, None);
    let before = ledger.snapshot();
    assert_eq!((before.others(), before.guarantee(0)), (Some(4 * GB), 0));
    while let Some(slice) = bank.pop() {
        // Sampled with the slice held, paired after its free.
        let held = ledger.snapshot().allocated();
        let at = ledger.sampled(clock.next());
        let sampled = Available::new(64 * GB, 64 * GB - 4 * GB - held, None).expect("a reading");
        drop(slice);
        ledger.read(at, Ok(sampled));
        let after = ledger.snapshot();
        assert_eq!((after.others(), after.guarantee(0), after.budget()), (Some(4 * GB), 0, before.budget()));
        // The next reading shows the free.
        read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - ledger.snapshot().allocated(), None);
        let shown = ledger.snapshot();
        assert_eq!((shown.others(), shown.guarantee(0), shown.budget()), (Some(4 * GB), 0, before.budget()));
        assert!(!shown.unload(0));
    }
}

/// A change that lands while a sample reads the OS (one bank slot backed
/// beside it) could show in one figure and not the other, or in the figure
/// and not the sample's place among the changes: such a sample does not
/// stand, and the one that does pairs exactly.
#[test]
fn a_sample_a_change_falls_within_is_taken_again() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    let slot = std::cell::RefCell::new(None);
    let reads = std::cell::Cell::new(0);
    let at = clock.next();
    let figure = || {
        reads.set(reads.get() + 1);
        let figure = 64 * GB - 4 * GB - ledger.snapshot().allocated();
        // The first read's figure is taken, then a slot lands beside it.
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(ledger.cached(CacheKind::Bank, GB).expect("free room"));
        }
        figure
    };
    assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "the sample the slot fell within does not stand");
    let (sampled, figure) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("the next stands");
    assert_eq!(reads.get(), 2);
    ledger.read(sampled, Ok(Available::new(64 * GB, figure, None).expect("a reading")));
    assert_eq!((ledger.snapshot().others(), ledger.snapshot().guarantee(0)), (Some(4 * GB), 0));
}

/// A shrink's memory is freed before its ledger record is written (the
/// release thread unmaps, the record follows): within the change that
/// spans both, no sample stands, so none reads the freed memory as others'
/// while the ledger still counts it; after it, others are exact. Others
/// never move.
#[test]
fn a_free_before_its_record_never_moves_others() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let slice = ledger.cached(CacheKind::Bank, 8 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 38 * GB, None);
    let used = std::cell::Cell::new(4 * GB + 38 * GB);
    let at = clock.next();
    let figure = || 64 * GB - used.get();
    ledger.store.changes.back(|_| {
        // The OS frees the slice; the ledger has not recorded it yet.
        used.set(4 * GB + 30 * GB);
        assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "no sample stands between the free and its record");
        drop(slice);
        assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "nor before the change ends");
    });
    let (sampled, available) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("a sample stands once the change ended");
    ledger.read(sampled, Ok(Available::new(64 * GB, available, None).expect("a reading")));
    let after = ledger.snapshot();
    assert_eq!((after.others(), after.guarantee(0)), (Some(4 * GB), 0));
}

/// A host tier's slot backed at load, as the bank opens and before any
/// served bank exists, is allocated in the tier's claim within one change
/// with its backing: between the slot's memory landing and its record no
/// sample stands, and after both others are exact. Others never move.
#[test]
fn a_slot_backed_at_load_before_its_record_never_moves_others() {
    let (scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    // The tier's claim, split from the import's, held by the bank from its open.
    let cache = crate::experts::cache::BankCache::new(scope.changes().clone());
    cache.grow(crate::domain::BankHolding::Host, 0, ledger.backing(8 * GB).expect("room"));
    let used = std::cell::Cell::new(4 * GB + 30 * GB);
    let at = clock.next();
    let figure = || 64 * GB - used.get();
    cache.changes().back(|changing| {
        // The slot's memory lands; its record has not.
        used.set(4 * GB + 32 * GB);
        assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "no sample stands between the backing and its record");
        cache.backed(2 * GB, changing);
    });
    let (sampled, available) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("a sample stands once the change ended");
    ledger.read(sampled, Ok(Available::new(64 * GB, available, None).expect("a reading")));
    let after = ledger.snapshot();
    assert_eq!((after.others(), after.guarantee(0)), (Some(4 * GB), 0));
}

/// Blind readings still let go of every change no later sample can miss:
/// the journal is bounded while the store cannot be read.
#[test]
fn blind_readings_bound_the_journal() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let mut held: Vec<Claim> = (0..32).map(|_| commit(&ledger, GB)).collect();
    while let Some(claim) = held.pop() {
        drop(claim);
        ledger.read(ledger.sampled(clock.next()), Err(Blind("the OS refused the reading".into())));
        assert!(ledger.state().journal.is_empty(), "a blind reading lets go of what every later sample shows");
    }
    read(&ledger, &mut clock, 64 * GB, 60 * GB, None);
    assert_eq!(ledger.snapshot().others(), Some(4 * GB));
}

/// The user's rebounds (45→79, 52→114 within a second): a reading sampled
/// before a growth reached the ledger after it, read others as having let
/// go of the growth's bytes, and the next true reading as their rise. A
/// growth never lowers others, and the reading after it raises nothing.
#[test]
fn a_growth_never_lowers_others() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    let before = ledger.snapshot();
    let mut bank = Vec::new();
    for _ in 0..4 {
        let held = ledger.snapshot().allocated();
        let at = ledger.sampled(clock.next());
        let sampled = Available::new(64 * GB, 64 * GB - 4 * GB - held, None).expect("a reading");
        bank.push(ledger.cached(CacheKind::Bank, 5 * GB).expect("free room"));
        ledger.read(at, Ok(sampled));
        assert_eq!((ledger.snapshot().others(), ledger.snapshot().budget()), (Some(4 * GB), before.budget()));
        read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - ledger.snapshot().allocated(), None);
        let shown = ledger.snapshot();
        assert_eq!((shown.others(), shown.guarantee(0), shown.budget()), (Some(4 * GB), 0, before.budget()));
    }
}

/// Every store's figure shows a release of the process's at once: its
/// bytes are free at the first sample after it, and others' memory stays
/// exact across it.
#[test]
fn a_release_is_free_at_the_first_sample_after_it() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let scratch = commit(&ledger, 4 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 6 * GB - 24 * GB, None);
    let before = ledger.snapshot();
    drop(scratch);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 6 * GB - 20 * GB, None);
    let shown = ledger.snapshot();
    assert_eq!((shown.others(), shown.free()), (Some(6 * GB), before.free() + 4 * GB));
}

/// Other programs' allocation landing while a sample reads: the sample
/// stands (no change of the process's is beside it) and shows their rise
/// at once.
#[test]
fn an_outside_allocation_racing_a_sample_shows_at_once() {
    let (_scope, ledger) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    let others = std::cell::Cell::new(4 * GB);
    let at = clock.next();
    let figure = || {
        let figure = 64 * GB - others.get() - 30 * GB;
        others.set(others.get() + 3 * GB);
        figure
    };
    let (sampled, available) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("others' change never stops a sample");
    ledger.read(sampled, Ok(Available::new(64 * GB, available, None).expect("a reading")));
    assert_eq!((ledger.snapshot().others(), ledger.snapshot().guarantee(0)), (Some(4 * GB), 0));
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 7 * GB - 30 * GB, None);
    assert_eq!((ledger.snapshot().others(), ledger.snapshot().guarantee(0)), (Some(7 * GB), 3 * GB));
}

/// An allocation change recorded outside any change of its memory (a
/// holder dropped where no sample is held off) is the invariant's
/// violation, in the process's scope.
#[test]
#[should_panic(expected = "an allocation change recorded outside a change of its memory")]
fn a_record_outside_a_change_panics() {
    let scope = Ledgers::strict();
    let _owner = scope.changes().own();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    read(&ledger, &mut Clock::new(), 64 * GB, 64 * GB, None);
    let claim = ledger.change(|changing| {
        let mut claim = ledger.claim(Need::Measurement { bytes: GB }).expect("the claim fits");
        claim.allocated(GB, changing);
        claim
    });
    drop(claim);
}

/// A sample a change lands beside at every try never stands: past a reading
/// interval with no change under way, that is the invariant's violation,
/// and the sample's wait is let go however it ends, so a step off the
/// owner's thread waits for it no longer.
#[test]
fn a_sample_that_never_stands_panics_and_lets_its_wait_go() {
    let (_scope, ledger) = host();
    let changes = ledger.store.changes.clone();
    let beside = || changes.change(|_| ());
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Sampled::steady(&changes, Instant::now, beside, || false)));
    let payload = panicked.expect_err("no sample stands");
    let message = payload.downcast_ref::<&str>().copied().unwrap_or_default();
    assert!(message.contains("no sample stood within a reading interval"), "{message}");
    assert_eq!(changes.meet().wanted, 0, "the panicked sample waits no more");
    changes.back(|_| ());
}

/// A growth off the owner's thread longer than a reading interval, slot by
/// slot, leaves every interval its sample: each slot begins only once no
/// sample waits, and each sample stands within the interval.
#[test]
fn a_growth_longer_than_an_interval_leaves_every_interval_its_sample() {
    let (_scope, ledger) = host();
    let changes = ledger.store.changes.clone();
    let growing = {
        let changes = changes.clone();
        std::thread::spawn(move || {
            for _ in 0..30 {
                changes.back(|_| std::thread::sleep(Duration::from_millis(15)));
            }
        })
    };
    for _ in 0..4 {
        let began = Instant::now();
        Sampled::steady(&changes, Instant::now, || (), || false).expect("never stopped");
        assert!(began.elapsed() < crate::platform::READING_INTERVAL, "a sample stands within its interval");
        std::thread::sleep(Duration::from_millis(50));
    }
    growing.join().expect("the growth ends");
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
    let prefix = ledger.cached(CacheKind::Prefix, 10 * GB).expect("room");
    let spare = ledger.cached(CacheKind::Spare, 3 * GB).expect("room");
    let bank = ledger.cached(CacheKind::Bank, free - 10 * GB - 3 * GB - GB / 2).expect("room");
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
        ledger.read(ledger.sampled(Instant::now()), Err(Blind("the OS refused the reading".into())));
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
    held.made();
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
    assert!(ledger.cached(CacheKind::Bank, 21 * GB).is_err(), "the bank never grows into the room");
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
    let bank = ledger.cached(CacheKind::Bank, 60 * GB).expect("the rest");
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
    ledger.read(ledger.sampled(Instant::now()), Err(Blind("the OS refused the reading".into())));
    let need = Need::Measurement { bytes: GB };
    assert_eq!(ledger.admit(&need), Err(blind.clone()));
    assert!(matches!(ledger.claim(need.clone()), Err(Refused::Short(refused)) if refused == blind));
    assert!(matches!(ledger.cached(CacheKind::Prefix, GB), Err(refused) if refused == blind));
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 8 * GB, 4 * GB, None);
    let mut state = ledger.claim(pages(1, 0, 1)).expect("room");
    ledger.read(ledger.sampled(Instant::now()), Err(Blind("the OS refused the reading".into())));
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
    let mut bank = ledger.cached(CacheKind::Bank, 40 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    let _taken = ledger.claim(Need::Measurement { bytes: 10 * GB }).expect("cache yields to it");
    assert_eq!((bank.bytes(), bank.releasing()), (34 * GB, 6 * GB), "the bank yields 6 GB, still allocated");
    let taken = ledger.snapshot();
    assert_eq!((taken.allocated(), taken.releasing(), taken.allocatable()), (60 * GB, 6 * GB, 4 * GB), "only the free 4 GB are allocatable");
    // The bank has not freed them: the next reading counts them the process's.
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert_eq!(ledger.snapshot().others(), Some(0), "the yielded bytes read as others' memory");
    ledger.change(|changing| bank.released(6 * GB, changing));
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
    let bank = ledger.cached(CacheKind::Bank, 64 * GB).expect("free room");
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
    ledger.change(|changing| backing.backed(3 * GB, changing));
    assert_eq!(ledger.snapshot().allocated(), 3 * GB);
    let cache = ledger.change(|changing| backing.into_backed_cache(5 * GB, CacheKind::Bank, changing));
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
