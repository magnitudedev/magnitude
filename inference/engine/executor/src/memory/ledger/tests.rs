//! The ledger's catalog failures, each a state the ledger has no
//! representation for. No OS: readings are handed in.

use super::*;
use seismic::memory::Wiring::Pageable;
use std::num::NonZeroU64;
use std::time::{Duration, Instant};

const GB: u64 = 1 << 30;

/// A load's shape: one store of one per-token history domain of 1024 f32
/// values a row, at 1024 tokens of context.
pub(crate) fn shape() -> Arc<LoadShape> {
    let component = magnitude_state::ComponentDescriptor::new(
        magnitude_state::LayerRef::Target(0),
        magnitude_state::CodecSpec::dense(seismic::DType::F32, 1024, 1024),
        1,
    )
    .expect("a component");
    let domains = vec![magnitude_state::HistoryDomainLayout::Token { components: vec![component] }];
    Arc::new(LoadShape { stores: vec![StoreShape::paged(domains, 1024, 16)] })
}

/// The rows of one page of [`shape`]'s history: a request's state is laid
/// out a page at a time.
pub(crate) fn page_rows() -> u64 {
    let shape = shape();
    let store = &shape.stores[0];
    store.domains[0].pages(&store.domains, store.context, store.max_advance).expect("its pages").page_rows as u64
}

/// A load's request slot of `bytes`.
fn slot(ledger: &Ledger, bytes: u64) -> Claim {
    ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::RequestSlot { bytes })).expect("the slot fits")
}

/// `sequence`'s reservation of `pages` pages, drawn from `slot` first and
/// the rest beside it.
fn request(ledger: &Ledger, slot: &Claim, sequence: u64, pages: u64) -> Result<Reservation, Short> {
    ledger.change(|keeping| ledger.reserve(&keeping.claims(), slot, RequestId(sequence), pages * page_rows()))
}

/// A store's two pools' entries, each open while held: the records pool's,
/// which the bank's and backings' granules are charged to and which the
/// handle derefs to, and the state pool's.
struct Pools {
    records: Arc<dyn seismic::memory::Entry>,
    state: Arc<dyn seismic::memory::Entry>,
}

impl std::ops::Deref for Pools {
    type Target = Arc<dyn seismic::memory::Entry>;

    fn deref(&self) -> &Self::Target {
        &self.records
    }
}

/// A scope of its own, host RAM's ledger in it, and its granule pools'
/// entries ([`Pools`]), open while held.
fn host() -> (Ledgers, Ledger, Pools) {
    let scope = Ledgers::new();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    let records = ledger.pool_block(PoolClass::Records, NonZeroU64::new(GB).unwrap()).blocks;
    let state = ledger.pool_block(PoolClass::State, NonZeroU64::new(GB).unwrap()).blocks;
    (scope, ledger, Pools { records, state })
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

/// No lag: every change shows at once.
const NOW: seismic::Lag = seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO };

/// A store of `total` bytes whose grant leaves `available`.
fn device(total: u64, available: u64) -> Reading {
    Reading::device(total, available, NOW)
}

/// Host RAM at `total`, `available` free above the OS's target and nothing
/// else: no cache, no cold pages, no paging, its counters showing the
/// machine as it is read.
pub(crate) fn host_sample(total: u64, available: u64) -> seismic::HostSample {
    let at = Instant::now();
    seismic::HostSample {
        at,
        page_bytes: 4096,
        total,
        supply: seismic::Supply { free_beyond_target: i64::try_from(available).expect("bytes fit i64"), clean_cache: 0, cold_reclaimable: 0 },
        counted: seismic::Counted { at, lag: Duration::ZERO },
        cold: seismic::ColdPages::default(),
        own_reclaimed: Some(seismic::OwnReclaimed::NONE),
        wired: None,
        harm: Some(seismic::Harm { machine: 0, own: 0 }),
        pressure: Some(seismic::PressureLevel::Normal),
        lag: NOW,
    }
}

/// Read `ledger` at `total` and `available`; a wired ceiling letting the
/// process wire `grant` in all where given (unified Metal, nothing wired
/// but what the process charged as wired).
fn read(ledger: &Ledger, clock: &mut Clock, total: u64, available: u64, grant: Option<u64>) {
    let at = clock.next();
    let reading = match grant {
        None => device(total, available),
        Some(grant) => {
            let wired = seismic::WiredFigures { limit: seismic::WiredLimit::Set(grant), wire_count: ledger.state().wired };
            Reading::unified(&seismic::HostSample { wired: Some(wired), ..host_sample(total, available) }, u64::MAX)
        }
    };
    ledger.read(ledger.sampled(at), reading);
}

/// A committed claim of exactly `bytes`, sized without any formula, and
/// allocated: memory the process made.
fn commit(ledger: &Ledger, bytes: u64) -> Claim {
    let claim = held(ledger, bytes).expect("the claim fits");
    ledger.change(|_| claim.entry().take(&claim.entry().change(), bytes, Pageable)).expect("the claim's own bytes");
    claim
}

/// A committed claim of exactly `bytes`, sized without any formula, none
/// of it allocated yet.
fn held(ledger: &Ledger, bytes: u64) -> Result<Claim, Short> {
    ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, bytes))
}

/// The OS file cache is supply, never others' memory: the same 48 GB
/// unified machine whose clean file cache holds 30 GB or 38 GB of the memory
/// others leave (the rest free) gives one room, one target and so one bank.
/// Main read 14.8 GB allocatable beside 29.7 GB of cache and 7.0 GB beside
/// 37.8 GB on one M4 Pro, sizing a 35B q4 bank to 147 slots, then 83.
#[test]
fn file_cache_is_supply_whatever_it_holds() {
    let estimate = |cache: u64| {
        let (_scope, ledger, _pool) = host();
        let (total, others) = (48 * GB, 8 * GB);
        let sample = seismic::HostSample {
            supply: seismic::Supply { free_beyond_target: i64::try_from(total - others - cache).expect("bytes fit i64"), clean_cache: cache, cold_reclaimable: 0 },
            wired: Some(seismic::WiredFigures { limit: seismic::WiredLimit::Set(36 * GB), wire_count: GB }),
            ..host_sample(total, 0)
        };
        ledger.read(ledger.sampled(Clock::new().next()), Reading::unified(&sample, u64::MAX));
        ledger.estimate()
    };
    let (thirty, thirty_eight) = (estimate(30 * GB), estimate(38 * GB));
    assert_eq!(thirty.others(), 8 * GB, "cache is no one's memory");
    assert_eq!(
        (thirty.others(), thirty.budget(), thirty.room(), thirty.target()),
        (thirty_eight.others(), thirty_eight.budget(), thirty_eight.room(), thirty_eight.target())
    );
}

/// Items 1, 31, 62: our own free is never other programs' memory: it
/// moves nothing until the next reading, which shows it, so it never
/// shrinks anything twice and never unloads.
#[test]
fn an_own_free_is_never_others_memory() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let weights = commit(&ledger, 40 * GB);
    let bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    let before = ledger.estimate();
    assert_eq!(before.others(), 0);
    ledger.change(|_| {
        drop(bank);
        pool.freed(&pool.change(), 20 * GB, Pageable);
    });
    let freed = ledger.estimate();
    assert_eq!((freed.others(), freed.budget(), freed.guarantee()), (0, before.budget(), 0));
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let shown = ledger.estimate();
    assert_eq!((shown.others(), shown.budget(), shown.guarantee()), (0, before.budget(), 0));
    drop(weights);
}

/// A Vulkan driver that shows our free two readings late shows it late in
/// its usage and its budget's grant alike, both read in one query: each
/// reading settles the driver's own entry to its usage less our blocks, so
/// the freed bytes stay counted as the driver's, never as others', until
/// the driver shows them free. Nothing moves meanwhile; then the driver's
/// entry drops by them and room grows by them.
#[test]
fn a_free_the_driver_shows_late_stays_the_drivers_until_it_shows() {
    use seismic::memory::Store as _;
    const TOTAL: u64 = 16 * GB;
    const FREED: u64 = 2 * GB;
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, TOTAL, TOTAL, None);
    let driver = ledger.own();
    let weights = commit(&ledger, 6 * GB);
    let scratch = commit(&ledger, FREED);
    // The driver's usage of the device heaps, each reading's: our blocks
    // until the free shows, two readings after it.
    let shown = [8 * GB, 8 * GB, 8 * GB, 6 * GB];
    let mut charged: Option<seismic::memory::Charge> = None;
    // One reading as `Sample::try_now` takes it, by the one function it
    // takes it with: `seismic_vulkan::Device::reconcile_driver_memory`
    // settles the driver's entry to the usage less our blocks and returns
    // the budget less that same usage, then the sample is stamped, then
    // `DeviceLedgers::read` reads the reconcile's grant.
    let mut reading = |usage: u64, ours: u64, charged: &mut Option<seismic::memory::Charge>| {
        let at = clock.next();
        let reconcile = || {
            let unheld = usage - ours;
            if charged.as_ref().map_or(0, seismic::memory::Charge::bytes) != unheld {
                ledger.change(|_| {
                    *charged = None;
                    *charged = (unheld > 0).then(|| seismic::memory::Charge::took(&driver.change(), &driver, unheld, Pageable));
                });
            }
            seismic::memory::Grant { left: TOTAL - usage }
        };
        let (sampled, grant, ()) = Sampled::reconciled(&ledger.store.changes, &|| at, reconcile, &|| ()).expect("no change of another thread is beside it");
        ledger.read(sampled, device(TOTAL, grant.left));
        ledger.estimate()
    };
    let before = reading(shown[0], 8 * GB, &mut charged);
    assert_eq!((before.others(), before.guarantee()), (0, 0));
    ledger.change(|_| {
        scratch.entry().freed(&scratch.entry().change(), FREED, Pageable);
        drop(scratch);
    });
    for usage in &shown[1..3] {
        let lagging = reading(*usage, 6 * GB, &mut charged);
        assert_eq!(lagging.others(), before.others(), "the freed bytes are never others'");
        assert_eq!(lagging.guarantee(), 0, "others rose by nothing");
        assert_eq!(lagging.cap(), before.cap(), "no settle or harm step");
        assert_eq!(lagging.room(), before.room(), "the driver still counts them");
        assert_eq!(charged.as_ref().map(seismic::memory::Charge::bytes), Some(FREED), "the driver's entry holds them");
        assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "no shortage, so nothing yields, preempts or unloads");
    }
    let shown_free = reading(shown[3], 6 * GB, &mut charged);
    assert!(charged.is_none(), "the driver's entry drops by them");
    assert_eq!(shown_free.others(), before.others());
    assert_eq!(shown_free.room(), before.room() + FREED, "room grows by them");
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    assert_eq!(ledger.allocated(), 6 * GB);
    drop(weights);
}

/// Bytes a GPU fence still reads stay held in their own entry until the
/// fence completes and the free returns: what the ledger holds covers them,
/// so no claim takes them meanwhile.
#[test]
fn a_fenced_free_stays_held_until_its_fence_completes() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 16 * GB, 16 * GB, None);
    let retired = ledger.cached(CacheKind::Bank, 10 * GB).expect("free room");
    // The fence is in flight: the holder keeps its handle.
    let fenced = retired;
    assert_eq!(ledger.estimate().cache(), 10 * GB);
    assert!(ledger.cached(CacheKind::Prefix, 7 * GB).is_err(), "the fenced bytes are not free");
    // The fence completes; the free returns.
    ledger.change(|_| {
        drop(fenced);
        pool.freed(&pool.change(), 10 * GB, Pageable);
    });
    assert_eq!(ledger.estimate().cache(), 0);
    assert!(ledger.cached(CacheKind::Prefix, 7 * GB).is_ok());
}

/// Items 2, 25: memory the process claims and backs is never read as other
/// programs taking it, so it sets no guarantee and moves no budget.
#[test]
fn own_claims_are_never_others_taking_memory() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    let idle = ledger.estimate();
    assert_eq!(idle.others(), (24 * GB));
    let mut held = Vec::new();
    for step in 1..=10 {
        held.push(commit(&ledger, 2 * GB));
        read(&ledger, &mut clock, 64 * GB, 40 * GB - step * 2 * GB, None);
        let backed = ledger.estimate();
        assert_eq!(backed.others(), (24 * GB));
        assert_eq!(backed.guarantee(), 0);
        assert_eq!(backed.budget(), idle.budget());
    }
}

/// Item 23: the guarantee is others' rise between the last two readings
/// alone: never a forecast scaled by our own release latency, and nothing
/// of the process's own (a request's state is committed ahead of the
/// rows it writes).
#[test]
fn the_guarantee_is_others_rise_only() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    ledger.change(|_| {
        drop(bank);
        pool.freed(&pool.change(), 20 * GB, Pageable);
    });
    // Our free, shown: no term of it.
    read(&ledger, &mut clock, 64 * GB, 34 * GB, None);
    assert_eq!(ledger.estimate().guarantee(), 0);
    // Another program takes 2 GB in one interval.
    read(&ledger, &mut clock, 64 * GB, 32 * GB, None);
    assert_eq!(ledger.estimate().guarantee(), 2 * GB);
}

/// Others' rise stands across a reading that cannot judge it: the process's
/// own free, read before every figure shows it, leaves the guarantee the
/// rise it answered, so our own resize never reads as others' rise ending.
#[test]
fn our_own_change_out_of_step_keeps_others_rise() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    // Another program takes 2 GB in one interval.
    read(&ledger, &mut clock, 64 * GB, 12 * GB, None);
    assert_eq!(ledger.estimate().guarantee(), 2 * GB);
    // The bank's cache yields; a reading whose figures show it a second
    // late judges no rise of others', and keeps the one it had.
    ledger.change(|_| {
        drop(bank);
        pool.freed(&pool.change(), 20 * GB, Pageable);
    });
    let late = seismic::Lag { release: Duration::from_secs(1), growth: Duration::from_secs(1) };
    ledger.read(ledger.sampled(clock.next()), Reading::device(64 * GB, 12 * GB, late));
    assert!(ledger.estimate().pending(), "the free is not yet shown");
    assert_eq!(ledger.estimate().guarantee(), 2 * GB, "others' rise stands across our own change");
}

/// A growth freed before the figures show it, where a release shows at
/// once and a growth late (a Linux cgroup's 2 s statistics flush): the
/// release shows no sooner than the growth it returns, so the process reads
/// as holding what it held before both, never less, and others hold
/// nothing throughout.
#[test]
fn a_growth_freed_before_it_shows_reads_as_neither() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let cgroup = seismic::Lag { release: Duration::ZERO, growth: Duration::from_secs(2) };
    let staging = commit(&ledger, 16 << 20);
    ledger.read(ledger.sampled(clock.next()), Reading::device(64 * GB, 64 * GB, cgroup));
    drop(staging);
    ledger.read(ledger.sampled(clock.next()), Reading::device(64 * GB, 64 * GB, cgroup));
    let estimate = ledger.estimate();
    assert!(estimate.pending(), "the growth is not yet shown, nor the release after it");
    assert_eq!(estimate.others(), 0);
    for _ in 0..20 {
        clock.next();
    }
    ledger.read(ledger.sampled(clock.next()), Reading::device(64 * GB, 64 * GB, cgroup));
    let estimate = ledger.estimate();
    assert!(!estimate.pending(), "both show once the growth's lag has passed");
    assert_eq!(estimate.others(), 0);
}

/// Items 35, 43: the budget is what the process holds plus what the OS
/// has available, bounded by a grant: never others' pages the OS would
/// compress or swap, never a figure counted twice.
#[test]
fn the_budget_is_held_and_available() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 48 * GB, 48 * GB, None);
    let _weights = commit(&ledger, 12 * GB);
    let _prefix = ledger.cached(CacheKind::Prefix, 2 * GB).expect("free room");
    read(&ledger, &mut clock, 48 * GB, 10 * GB, None);
    let snapshot = ledger.estimate();
    assert_eq!((snapshot.committed(), snapshot.cache()), (12 * GB, 2 * GB));
    assert_eq!(snapshot.budget(), 12 * GB + 2 * GB + 10 * GB);
    assert_eq!(snapshot.others(), (48 * GB - 10 * GB - 14 * GB));
    read(&ledger, &mut clock, 48 * GB, 10 * GB, Some(20 * GB));
    assert_eq!(ledger.estimate().budget(), 20 * GB);
}

/// Items 53, 61: other programs' memory lowers the budget as they take it
/// and raises it as they let it go, by the same reading; nothing of ours
/// moves it.
#[test]
fn others_memory_moves_the_budget_both_ways() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    read(&ledger, &mut clock, 64 * GB, 44 * GB, None);
    let alone = ledger.estimate();
    assert_eq!(alone.others(), 0);
    read(&ledger, &mut clock, 64 * GB, 19 * GB, None);
    let beside = ledger.estimate();
    assert_eq!(beside.others(), (25 * GB));
    assert_eq!(beside.budget(), alone.budget() - 25 * GB);
    assert_eq!(beside.guarantee(), 25 * GB);
    read(&ledger, &mut clock, 64 * GB, 19 * GB, None);
    assert_eq!(ledger.estimate().guarantee(), 0);
    read(&ledger, &mut clock, 64 * GB, 44 * GB, None);
    let after = ledger.estimate();
    assert_eq!(after.budget(), alone.budget());
    assert_eq!(after.guarantee(), 0);
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
    assert_eq!(two.estimate().committed(), 5 * GB);
    read(&two, &mut clock, 64 * GB, 30 * GB, None);
    let stale = clock.0 + Duration::from_millis(50);
    one.read(one.sampled(stale), device(64 * GB, 1 * GB));
    assert_eq!(one.estimate().others(), (64 * GB - 30 * GB - 5 * GB));
    let other = Ledgers::new().ledger(seismic::StoreIdentity::HostRam, shape());
    assert!(other.entries().is_empty());
}

/// Item 20: every figure of a decision comes from one reading.
#[test]
fn a_snapshot_is_one_reading() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(50 * GB));
    let _claim = commit(&ledger, 8 * GB);
    read(&ledger, &mut clock, 64 * GB, 40 * GB, Some(50 * GB));
    let snapshot = ledger.estimate();
    let _more = commit(&ledger, 4 * GB);
    read(&ledger, &mut clock, 64 * GB, 10 * GB, Some(50 * GB));
    let others = snapshot.others();
    assert_eq!(snapshot.committed(), 8 * GB);
    assert_eq!(others, 64 * GB - 40 * GB - 8 * GB);
    assert_eq!(snapshot.budget(), (64 * GB - others).min(50 * GB));
}

/// Item 46: the process's own memory never unloads it. The budget is
/// what it holds plus the store's supply, under a grant whose ceiling is
/// what it holds plus what the grant leaves, so committed memory never
/// exceeds it; a claim past it is refused instead.
#[test]
fn a_full_store_refuses_and_never_unloads() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(32 * GB));
    let _weights = commit(&ledger, 30 * GB);
    // Other programs take everything else, and the grant leaves nothing.
    read(&ledger, &mut clock, 64 * GB, 34 * GB, Some(30 * GB));
    read(&ledger, &mut clock, 64 * GB, 34 * GB, Some(30 * GB));
    let full = ledger.estimate();
    assert_eq!(full.budget(), 30 * GB);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    assert!(matches!(held(&ledger, 1), Err(Short { over: 1, .. })));
}

/// Item 46: others squeezing the load at one reading is a real shortage,
/// but one reading decides no unload: others' rise stops at the next, the
/// guarantee falls to zero and the load stays.
#[test]
fn a_squeeze_at_one_reading_never_unloads() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 50 * GB);
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    // Others take 12 GB in one interval: 2 GB stay free.
    read(&ledger, &mut clock, 64 * GB, 2 * GB, None);
    let squeezed = ledger.estimate();
    assert_eq!(squeezed.guarantee(), 12 * GB);
    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("the guarantee cannot be kept");
    assert!(matches!(shortage.cause, shortage::Cause::RiseShort { over } if over == 10 * GB));
    assert!(!shortage.response.unload, "one reading decides nothing");
    // Others stop rising: the guarantee falls to zero, and the load stays.
    read(&ledger, &mut clock, 64 * GB, 2 * GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
}

/// Others rising past what the budget holds beside committed memory is a
/// rise shortfall: relieved where it can be, new work waiting, never an
/// unload, however many readings it spans. A request refused for room
/// waits for the next reading.
#[test]
fn a_rise_shortfall_never_unloads() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 50 * GB);
    let slot = slot(&ledger, 0);
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    for available in [4, 1, 0] {
        read(&ledger, &mut clock, 64 * GB, available * GB, None);
        let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("others' rise is a real shortage");
        assert!(matches!(shortage.cause, shortage::Cause::RiseShort { .. }), "{:?}", shortage.cause);
        assert!(shortage.response.admission_waits && !shortage.response.unload, "a rise shortfall never unloads");
    }
    read(&ledger, &mut clock, 64 * GB, 0, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "others stopped rising");
    assert!(request(&ledger, &slot, 1, 1).is_err(), "no room for a request");
    assert_eq!(ledger.standing(), Some(Standing::Wait), "the refused request waits for a reading");
    read(&ledger, &mut clock, 64 * GB, 0, None);
    assert_eq!(ledger.standing(), None, "a reading answers the wait");
}

/// Free memory below the OS's target reads whole: others' growth into the
/// OS's reserve is their rise, and a budget it takes below what is
/// committed is a floor shortfall.
#[test]
fn free_below_the_target_reads_as_others_rise_and_a_shortfall() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    let read_free = |ledger: &Ledger, clock: &mut Clock, free: i64| {
        let sample = seismic::HostSample { supply: seismic::Supply { free_beyond_target: free, clean_cache: 0, cold_reclaimable: 0 }, ..host_sample(64 * GB, 0) };
        ledger.read(ledger.sampled(clock.next()), Reading::host(&sample));
    };
    read_free(&ledger, &mut clock, 64 * GB as i64);
    let _weights = commit(&ledger, 50 * GB);
    read_free(&ledger, &mut clock, 14 * GB as i64);
    read_free(&ledger, &mut clock, 0);
    read_free(&ledger, &mut clock, -2 * GB as i64);
    let below = ledger.estimate();
    assert_eq!((below.supply(), below.guarantee(), below.budget()), (-2 * GB as i64, 2 * GB, 48 * GB), "the dip is others' rise, the budget below what is committed");
    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("committed past the budget");
    assert_eq!(shortage.cause, shortage::Cause::FloorShort { over: 2 * GB });
}

/// A floor shortfall following a rise shortfall that stood past λ is
/// judged from its own first reading: one reading of it decides nothing.
#[test]
fn a_floor_shortfall_after_a_standing_rise_waits_its_own_lambda() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = held(&ledger, 50 * GB).expect("room");
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    for available in [55, 52] {
        read(&ledger, &mut clock, 64 * GB, available * GB, None);
        let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("others' rise is a real shortage");
        assert!(matches!(shortage.cause, shortage::Cause::RiseShort { .. }), "{:?}", shortage.cause);
    }
    read(&ledger, &mut clock, 64 * GB, 45 * GB, None);
    let floor = ledger.shortage(&std::collections::BTreeSet::new()).expect("the budget is short of what is committed");
    assert!(matches!(floor.cause, shortage::Cause::FloorShort { .. }), "{:?}", floor.cause);
    assert!(!floor.response.unload, "the rise's span is no part of the floor's");
    read(&ledger, &mut clock, 64 * GB, 45 * GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).expect("still short").response.unload, "past its own λ: unload");
}

/// A floor shortfall (the budget short of what is committed) that holds
/// on readings spanning the store's λ, with nothing to relieve it and no
/// change of the process's pending, unloads; one reading decides nothing.
#[test]
fn a_floor_shortfall_past_lambda_unloads() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = held(&ledger, 50 * GB).expect("room");
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    let first = ledger.shortage(&std::collections::BTreeSet::new()).expect("the budget is short of what is committed");
    assert!(matches!(first.cause, shortage::Cause::FloorShort { over } if over == 10 * GB), "{:?}", first.cause);
    assert!(!first.response.unload, "one reading decides nothing");
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).expect("still short").response.unload, "past λ, nothing to give: unload");
}

/// Each reading is answered once: asking again at a later event with no
/// new reading yields nothing more, however long the shortage stands, and
/// the shortage still stands until a reading clears it.
#[test]
fn a_shortage_is_answered_once_per_reading() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 56 * GB);
    let _bank = ledger.cached(CacheKind::Bank, 2 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 6 * GB, None);
    read(&ledger, &mut clock, 64 * GB, GB, None);
    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("others rose by 5 GB: the budget holds 59, the weights and the rise 61");
    assert_eq!(shortage.response.relief.as_ref().map(ReliefPlan::covers), Some(2 * GB), "the whole cache");
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "the reading was answered");
    assert!(matches!(ledger.standing(), Some(Standing::Shortage(_))), "and still stands");
    read(&ledger, &mut clock, 64 * GB, GB, None);
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "others stopped rising: the next reading clears it");
    assert_eq!(ledger.standing(), None);
}

/// A rise shortfall covered by cache with cache remaining above floor yields
/// only the shortfall and never holds admission back (§4.5).
#[test]
fn a_rise_shortfall_covered_by_cache_above_floor_does_not_wait_admission() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 4 * GB);
    let bank = ledger.cached(CacheKind::Bank, 8 * GB).expect("free room");
    // Baseline: others hold 38 GB, supply 14 GB (budget 26 GB).
    read(&ledger, &mut clock, 64 * GB, 14 * GB, None);
    // Others rise by 13 GB -> supply 1 GB, budget 13 GB, guarantee 13 GB.
    // committed (4 GB) + guarantee (13 GB) - budget (13 GB) = 4 GB < bank (8 GB).
    read(&ledger, &mut clock, 64 * GB, GB, None);
    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("rise shortfall");
    assert_eq!(shortage.cause, shortage::Cause::RiseShort { over: 4 * GB });
    assert!(!shortage.response.admission_waits, "cache covers the shortfall with 4 GB remaining above floor");
    assert!(!shortage.response.unload);
    let plan = shortage.response.relief.expect("cache covers the rise shortfall");
    assert!(plan.preempt().is_empty());
    assert_eq!(plan.covers(), 4 * GB);
    let owed = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
    assert_eq!(owed.yielded, 4 * GB);
    assert_eq!(bank.bytes(), 4 * GB, "bank remains above floor after covering the shortfall");
}

/// A rise shortfall exceeding cache with no evictable requests still drains
/// all remaining cache to floor (Step 1 of §4.5) alongside waiting admission.
#[test]
fn a_rise_shortfall_past_cache_drains_cache_before_waiting_admission() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 4 * GB);
    let bank = ledger.cached(CacheKind::Bank, 8 * GB).expect("free room");
    // Baseline: others hold 32 GB, supply 20 GB (budget 32 GB).
    read(&ledger, &mut clock, 64 * GB, 20 * GB, None);
    // Others rise by 19 GB -> supply 1 GB, budget 13 GB, guarantee 19 GB.
    // committed (4 GB) + guarantee (19 GB) - budget (13 GB) = 10 GB > bank (8 GB).
    read(&ledger, &mut clock, 64 * GB, GB, None);
    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("rise shortfall past cache");
    assert_eq!(shortage.cause, shortage::Cause::RiseShort { over: 10 * GB });
    assert!(shortage.response.admission_waits, "bank drains to floor and cannot cover the full rise");
    assert!(!shortage.response.unload);
    let plan = shortage.response.relief.expect("remaining cache drains to floor");
    assert!(plan.preempt().is_empty());
    assert_eq!(plan.covers(), 8 * GB);
    let owed = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
    assert_eq!(owed.yielded, 8 * GB);
    assert_eq!(bank.bytes(), 0, "bank reached its floor");
}

/// A floor shortfall larger than remaining cache drains cache to floor first
/// (never unloading on a reading that yielded cache), and unloads only on a
/// subsequent reading past λ once cache is empty.
#[test]
fn a_floor_shortfall_with_cache_drains_cache_before_unloading() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = held(&ledger, 50 * GB).expect("room");
    let bank = ledger.cached(CacheKind::Bank, 4 * GB).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    // Drop supply to 40 GB -> budget = 4 GB (cached) + 40 GB = 44 GB (FloorShort over 6 GB > bank 4 GB).
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    let first = ledger.shortage(&std::collections::BTreeSet::new()).expect("floor shortfall");
    assert_eq!(first.cause, shortage::Cause::FloorShort { over: 6 * GB });
    assert!(first.response.admission_waits);
    assert!(!first.response.unload, "must drain cache before unloading");
    let plan = first.response.relief.expect("remaining cache drains to floor");
    assert_eq!(plan.covers(), 4 * GB);
    ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
    assert_eq!(bank.bytes(), 0);
    // Next reading past λ with cache empty: now unloads.
    read(&ledger, &mut clock, 64 * GB, 40 * GB, None);
    let second = ledger.shortage(&std::collections::BTreeSet::new()).expect("floor shortfall persists");
    assert!(second.response.relief.is_none());
    assert!(second.response.unload, "past λ with cache at floor: unload");
}

/// The same squeeze beside a later request: relief has its reservation to
/// take, so the plan preempts it and nothing unloads.
#[test]
fn a_squeeze_with_a_later_request_preempts_it_instead() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 40 * GB);
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    let slot = slot(&ledger, page);
    let _first = request(&ledger, &slot, 1, 1).expect("the slot");
    let _second = request(&ledger, &slot, 2, (8 * GB).div_ceil(page)).expect("room");
    let held = ledger.estimate().committed();
    read(&ledger, &mut clock, 64 * GB, 64 * GB - held, None);
    read(&ledger, &mut clock, 64 * GB, 1 * GB, None);
    let shortage = ledger.shortage(&[1, 2].map(RequestId).into()).expect("the guarantee cannot be kept");
    assert_eq!(shortage.response.relief.as_ref().map(ReliefPlan::preempt), Some(&[RequestId(2)][..]));
    assert!(!shortage.response.unload, "the second request is preemptible");
}

/// Four requests of one page each, the first drawn from the slot, beside a
/// retained prefix of a page and a bank filling the rest of a 40 GB grant;
/// the slot and the page's bytes.
fn four_requests(ledger: &Ledger) -> ([Reservation; 4], (Cached, Cached), Claim, u64) {
    let mut clock = Clock::new();
    read(ledger, &mut clock, 64 * GB, 64 * GB, Some(40 * GB));
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    let slot = slot(ledger, page);
    let requests = [1, 2, 3, 4].map(|sequence| request(ledger, &slot, sequence, 1).expect("room"));
    let prefix = ledger.cached(CacheKind::Prefix, page).expect("room");
    let bank = ledger.cached(CacheKind::Bank, ledger.estimate().budget() - 5 * page).expect("the rest");
    (requests, (bank, prefix), slot, page)
}

/// A relief plan takes cache, then the requests admitted after the first,
/// the latest first and just enough, never the first nor one its holder
/// cannot evict now; it exists whole or not at all, and none takes
/// nothing: a need past what yields leaves the bank as it was.
#[test]
fn a_relief_plan_covers_its_need_whole_or_takes_nothing() {
    let (_scope, ledger, _pool) = host();
    let (_requests, (bank, prefix), _slot, page) = four_requests(&ledger);
    let cached = bank.bytes() + prefix.bytes();
    let every = [1, 2, 3, 4].map(RequestId).into();
    let plan = ledger.plan_relief(cached + 2 * page, Relief::CacheAndRequests(&every), crate::memory::ledger::Frees::Room).expect("the cache and two requests cover it");
    assert_eq!(plan.preempt(), [RequestId(4), RequestId(3)]);
    let held = [1, 2, 3].map(RequestId).into();
    let around = ledger.plan_relief(cached + 2 * page, Relief::CacheAndRequests(&held), crate::memory::ledger::Frees::Room).expect("the cache and two evictable requests cover it");
    assert_eq!(around.preempt(), [RequestId(3), RequestId(2)], "the request its holder cannot evict is never named");
    assert_eq!(ledger.plan_relief(cached + 4 * page, Relief::CacheAndRequests(&every), crate::memory::ledger::Frees::Room), None, "the first request is never preempted");
    assert_eq!(ledger.plan_relief(cached + 1, Relief::CacheOnly, crate::memory::ledger::Frees::Room), None, "cache alone cannot cover it");
    let after = ledger.estimate();
    assert_eq!((after.cache_of(CacheKind::Bank), after.cache_of(CacheKind::Prefix)), (cached - page, page), "no plan took the bank or the prefix");
    let owed = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
    assert_eq!((owed.yielded, owed.preempt), (cached, vec![RequestId(4), RequestId(3)]));
    assert_eq!((bank.bytes(), prefix.bytes()), (0, 0), "the plan carried out whole");
}

/// An invariant broken under the ledger's lock panics as itself: the
/// handles dropped while unwinding never touch the ledger it poisoned, so
/// the test binary goes on.
#[test]
#[should_panic(expected = "a request's state is reserved by its own path, never in units")]
fn an_invariant_broken_under_the_lock_panics_as_itself() {
    let (_scope, ledger, _pool) = host();
    let (_requests, _cache, _slot, _page) = four_requests(&ledger);
    let _ = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Request(RequestId(5)), NonZeroU64::new(1).unwrap(), 1));
}

/// A claim shrinks by bytes its holder freed and claims no more: its bytes
/// and what it allocated fall together, the store's free room rising by
/// them.
#[test]
fn a_claim_shrinks_by_what_its_holder_freed() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 60 * GB, None);
    let (claim, units) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::GroupClass, NonZeroU64::new(GB).unwrap(), 3));
    assert_eq!(units, 3);
    ledger.change(|_| claim.entry().take(&claim.entry().change(), 3 * GB, Pageable)).expect("the claim's own bytes");
    let free = ledger.estimate().free();
    ledger.change(|_| claim.entry().freed(&claim.entry().change(), GB, Pageable));
    assert_eq!((claim.bytes(), claim.allocated_bytes()), (2 * GB, 2 * GB));
    assert_eq!(ledger.estimate().free(), free + GB);
}

/// A claim takes cache synchronously and is refused only where committed
/// memory with it exceeds the budget; admission is the same check. Free
/// room inside the guarantee (here others' rise holds it) is the claim's only
/// once cache is gone: the guarantee is the state's growth's room.
#[test]
fn a_claim_takes_cache_and_is_refused_only_past_the_budget() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let _bank = ledger.cached(CacheKind::Bank, 20 * GB).expect("free room");
    // Another program holds 20 GB: 4 GB are free.
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert_eq!(ledger.covers(None, 10 * GB), Ok(()));
    let _taken = held(&ledger, 10 * GB).expect("cache yields to it");
    let snapshot = ledger.estimate();
    assert_eq!(snapshot.committed(), 30 * GB);
    // The free 4 GB lie in the guarantee: 10 GB of the bank's.
    assert!(snapshot.guarantee() >= 4 * GB, "others' rise holds the free room: {}", snapshot.guarantee());
    assert_eq!(snapshot.cache(), 10 * GB);
    let short = Short { store: ledger.store(), required: 15 * GB, over: GB };
    assert_eq!(ledger.covers(None, 15 * GB), Err(short.clone()));
    assert!(matches!(held(&ledger, 15 * GB), Err(refused) if refused == short));
}

/// Every claim counts the store's supply from the first reading: free memory
/// above the OS's target, clean file cache, and cold reclaimable pages (minus
/// any unwired idle cache when the OS gives no own cold share).
#[test]
fn every_claim_counts_the_supply() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    let sample = host_sample(64 * GB, 4 * GB);
    let sample = seismic::HostSample {
        supply: seismic::Supply { clean_cache: 6 * GB, cold_reclaimable: 3 * GB, ..sample.supply },
        cold: seismic::ColdPages { machine: 6 * GB, own: None },
        ..sample
    };
    ledger.read(ledger.sampled(clock.next()), Reading::host(&sample));
    assert_eq!(ledger.estimate().free(), 13 * GB, "first reading counts cold reclaimable pages so loads never starve");
    let floor = held(&ledger, 13 * GB).expect("the supply covers it");
    assert!(ledger.covers(None, 1).is_err(), "what it took is held");
    drop(floor);
    let drafter_claim = commit(&ledger, 2 * GB);
    let idle_drafter = ledger.change(|_| drafter_claim.into_cache(CacheKind::Drafter, Reclaim::Deferred));
    ledger.read(ledger.sampled(clock.next()), Reading::host(&sample));
    assert_eq!(ledger.estimate().supply(), 11 * GB as i64, "idle cache reduces unknown-owner cold supply conservatively");
    assert_eq!(ledger.estimate().free(), 11 * GB);
    let entry = idle_drafter.entry();
    ledger.change(|_| {
        entry.freed(&entry.change(), 2 * GB, Pageable);
        drop(idle_drafter);
    });
}

/// A reading whose supply exceeds its total is no reading.
#[test]
#[should_panic(expected = "a device's grant leaves")]
fn a_supply_past_the_total_is_an_invariant_violation() {
    let _ = device(8 * GB, 9 * GB);
}

/// The user's shrink cascades (55→20 and 76→20 bank slots ~100 ms apart):
/// a reading sampled before a shrink reached the ledger after it, read the
/// freed bytes as others' and set the guarantee by them, and the next
/// shrink was larger. A shrink never raises others or the guarantee,
/// whichever side of it the reading was sampled on.
#[test]
fn a_shrink_never_raises_others_or_the_guarantee() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    let mut bank: Vec<Cached> = (0..4).map(|_| ledger.cached(CacheKind::Bank, 5 * GB).expect("free room")).collect();
    // Other programs hold 4 GB.
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 50 * GB, None);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 50 * GB, None);
    let before = ledger.estimate();
    assert_eq!((before.others(), before.guarantee()), (4 * GB, 0));
    while let Some(slice) = bank.pop() {
        // Sampled with the slice held, paired after its free.
        let held = ledger.estimate().allocated();
        let at = ledger.sampled(clock.next());
        let sampled = device(64 * GB, 64 * GB - 4 * GB - held);
        ledger.change(|_| {
            drop(slice);
            pool.freed(&pool.change(), 5 * GB, Pageable);
        });
        ledger.read(at, sampled);
        let after = ledger.estimate();
        assert_eq!((after.others(), after.guarantee(), after.budget()), (4 * GB, 0, before.budget()));
        // The next reading shows the free.
        read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - ledger.estimate().allocated(), None);
        let shown = ledger.estimate();
        assert_eq!((shown.others(), shown.guarantee(), shown.budget()), (4 * GB, 0, before.budget()));
        assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none());
    }
}

/// A change that lands while a sample reads the OS (one bank slot backed
/// beside it) could show in one figure and not the other, or in the figure
/// and not the sample's place among the changes: such a sample does not
/// stand, and the one that does pairs exactly.
#[test]
fn a_sample_a_change_falls_within_is_taken_again() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    let slot = std::cell::RefCell::new(None);
    let reads = std::cell::Cell::new(0);
    let at = clock.next();
    let figure = || {
        reads.set(reads.get() + 1);
        let figure = 64 * GB - 4 * GB - ledger.estimate().allocated();
        // The first read's figure is taken, then a slot lands beside it.
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(ledger.cached(CacheKind::Bank, GB).expect("free room"));
        }
        figure
    };
    assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "the sample the slot fell within does not stand");
    let (sampled, figure) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("the next stands");
    assert_eq!(reads.get(), 2);
    ledger.read(sampled, device(64 * GB, figure));
    assert_eq!((ledger.estimate().others(), ledger.estimate().guarantee()), (4 * GB, 0));
}

/// A shrink's memory is freed before its ledger record is written (the
/// release thread unmaps, the record follows): within the change that
/// spans both, no sample stands, so none reads the freed memory as others'
/// while the ledger still counts it; after it, others are exact. Others
/// never move.
#[test]
fn a_free_before_its_record_never_moves_others() {
    let (_scope, ledger, pool) = host();
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
        pool.freed(&pool.change(), 8 * GB, Pageable);
        assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "nor before the change ends");
    });
    let (sampled, available) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("a sample stands once the change ended");
    ledger.read(sampled, device(64 * GB, available));
    let after = ledger.estimate();
    assert_eq!((after.others(), after.guarantee()), (4 * GB, 0));
}

/// A host tier's slot backed at load, as the bank opens and before any
/// served bank exists, is allocated by the pool's block within one change
/// with its backing: between the slot's memory landing and its record no
/// sample stands, and after both others are exact. Others never move.
#[test]
fn a_slot_backed_at_load_before_its_record_never_moves_others() {
    let (scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    // The tier's claim, held by the bank from its open.
    let backing = ledger.change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, 8 * GB)).expect("room");
    let used = std::cell::Cell::new(4 * GB + 30 * GB);
    let at = clock.next();
    let figure = || 64 * GB - used.get();
    scope.changes().back(|_| {
        // The slot's memory lands; its record has not.
        used.set(4 * GB + 32 * GB);
        assert!(Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).is_none(), "no sample stands between the backing and its record");
        pool.take(&pool.change(), 2 * GB, Pageable).expect("the backing holds the block's granules");
    });
    drop(backing);
    let (sampled, available) = Sampled::attempt(&ledger.store.changes, &|| at, &figure, 0).expect("a sample stands once the change ended");
    ledger.read(sampled, device(64 * GB, available));
    let after = ledger.estimate();
    assert_eq!((after.others(), after.guarantee()), (4 * GB, 0));
}

/// Every reading lets go of every change it shows: the journal holds only
/// what samples to come may not show.
#[test]
fn readings_bound_the_journal() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let mut held: Vec<Claim> = (0..32).map(|_| commit(&ledger, GB)).collect();
    while let Some(claim) = held.pop() {
        drop(claim);
        assert!(ledger.estimate().pending());
        let allocated = ledger.state().allocated;
        read(&ledger, &mut clock, 64 * GB, 64 * GB - allocated, None);
        assert!(!ledger.estimate().pending(), "a reading lets go of what every later sample shows");
    }
    read(&ledger, &mut clock, 64 * GB, 60 * GB, None);
    assert_eq!(ledger.estimate().others(), 4 * GB);
}

/// The user's rebounds (45→79, 52→114 within a second): a reading sampled
/// before a growth reached the ledger after it, read others as having let
/// go of the growth's bytes, and the next true reading as their rise. A
/// growth never lowers others, and the reading after it raises nothing:
/// the guarantee stays what others' rise before the growth left it.
#[test]
fn a_growth_never_lowers_others() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - 30 * GB, None);
    let before = ledger.estimate();
    let mut bank = Vec::new();
    for _ in 0..4 {
        let held = ledger.estimate().allocated();
        let at = ledger.sampled(clock.next());
        let sampled = device(64 * GB, 64 * GB - 4 * GB - held);
        bank.push(ledger.cached(CacheKind::Bank, 5 * GB).expect("free room"));
        ledger.read(at, sampled);
        assert_eq!((ledger.estimate().others(), ledger.estimate().budget()), (4 * GB, before.budget()));
        read(&ledger, &mut clock, 64 * GB, 64 * GB - 4 * GB - ledger.estimate().allocated(), None);
        let shown = ledger.estimate();
        assert_eq!((shown.others(), shown.guarantee(), shown.budget()), (4 * GB, before.guarantee(), before.budget()));
    }
}

/// Every store's figure shows a release of the process's at once: its
/// bytes are free at the first sample after it, and others' memory stays
/// exact across it.
#[test]
fn a_release_is_free_at_the_first_sample_after_it() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let scratch = commit(&ledger, 4 * GB);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 6 * GB - 24 * GB, None);
    let before = ledger.estimate();
    drop(scratch);
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 6 * GB - 20 * GB, None);
    let shown = ledger.estimate();
    assert_eq!((shown.others(), shown.free()), (6 * GB, before.free() + 4 * GB));
}

/// Other programs' allocation landing while a sample reads: the sample
/// stands (no change of the process's is beside it) and shows their rise
/// at once.
#[test]
fn an_outside_allocation_racing_a_sample_shows_at_once() {
    let (_scope, ledger, _pool) = host();
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
    ledger.read(sampled, device(64 * GB, available));
    assert_eq!((ledger.estimate().others(), ledger.estimate().guarantee()), (4 * GB, 0));
    read(&ledger, &mut clock, 64 * GB, 64 * GB - 7 * GB - 30 * GB, None);
    assert_eq!((ledger.estimate().others(), ledger.estimate().guarantee()), (7 * GB, 3 * GB));
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
    let claim = ledger.change(|_| {
        let claim = held(&ledger, GB).expect("the claim fits");
        claim.entry().take(&claim.entry().change(), GB, Pageable).expect("the claim's own bytes");
        claim
    });
    drop(claim);
}

/// A unit claim takes the most whole units free room, spare granules and
/// bank cache cover, up to its most, yielding spare before the bank; it
/// never yields a prefix, a drafter or vision, never preempts a later
/// sequence, holds nothing with nothing to take, and frees what it held
/// as it drops.
#[test]
fn a_unit_claim_takes_whole_units_from_free_spare_and_bank_alone() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, Some(40 * GB));
    let unit = NonZeroU64::new(GB).unwrap();
    let slot = slot(&ledger, 0);
    let state = request(&ledger, &slot, 2, 1).expect("room");
    let page = state.bytes();
    let held = |claim: &Claim| claim.bytes();
    // Free room alone, capped by `most`.
    let (few, units) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, unit, 3));
    assert_eq!((units, held(&few)), (3, 3 * GB));
    drop(few);
    // Fill the store: some free room, spare, bank and a prefix.
    let free = ledger.estimate().budget() - ledger.estimate().committed();
    let prefix = ledger.cached(CacheKind::Prefix, 10 * GB).expect("room");
    let spare = ledger.cached(CacheKind::Spare, 3 * GB).expect("room");
    let bank = ledger.cached(CacheKind::Bank, free - 10 * GB - 3 * GB - GB / 2).expect("room");
    let bank_bytes = bank.bytes();
    // Free room is half a unit: whole units come from spare, then the bank.
    let (scratch, taken) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, unit, u64::MAX));
    assert_eq!(taken, (GB / 2 + 3 * GB + bank_bytes) / GB);
    assert_eq!(held(&scratch), taken * GB);
    assert_eq!(spare.bytes(), 0, "spare yields first");
    assert_eq!(bank.bytes(), bank_bytes + 3 * GB + GB / 2 - taken * GB, "the bank yields the rest");
    assert_eq!(prefix.bytes(), 10 * GB, "a prefix never yields to it");
    assert_eq!(state.bytes(), page, "no request is preempted");
    // Nothing left to take: it holds none.
    let (none, units) = ledger.change(|keeping| ledger.claim_units(&keeping.claims(), CommittedKind::Scratch, unit, 5));
    assert_eq!((units, held(&none)), (0, 0));
    drop(none);
    let committed = ledger.estimate().committed();
    drop(scratch);
    assert_eq!(ledger.estimate().committed(), committed - taken * GB, "dropping it frees its bytes");
}

/// A load's allocation is admitted or short by its store's figures.
#[test]
fn a_loads_allocation_is_admitted_or_short_by_its_figures() {
    let figures = device(8 * GB, 4 * GB);
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    let device = crate::test_support::open_backend(&catalog, seismic::BackendName::Cpu).expect("the CPU backend opens");
    let (ledgers, _) = DeviceLedgers::open_in(&Ledgers::new(), &device, shape()).read_scripted(figures);
    let demand = crate::memory::ClaimDemand { device: GB, ..crate::memory::ClaimDemand::default() };
    let store = ledgers.device().store();
    let claim = |demand| ledgers.changes().change(|keeping| ledgers.claim(&keeping.claims(), &device, CommittedKind::Load, demand));
    assert!(claim(demand).is_ok());
    let more = crate::memory::ClaimDemand { device: 5 * GB, ..demand };
    assert_eq!(claim(more).err(), Some(Short { store, required: 5 * GB, over: GB }));
}

/// A claim in a store never read is an invariant violation: a load's
/// ledgers claim only once read ([`UnreadLedgers`]); a test's own ledger
/// handle reaches the violation.
#[test]
#[should_panic(expected = "admits before its first reading")]
fn a_claim_in_a_store_never_read_panics() {
    let (_scope, ledger, _pool) = host();
    let _ = held(&ledger, GB);
}

/// A device's own memory is charged as it opens, before its store's first
/// reading: the reading needs the opened device (its grant is the driver's),
/// so the order buffer Metal opens with, its probe's buffer and CUDA's host
/// word are charged through Seismic's view of the store ([`Ledgers::stores`])
/// with no reading to judge them by, never refused by the ledger. On unified
/// memory host RAM is the device's store; the first reading counts what
/// opening charged, and gives the ledgers every claim after it is judged in.
#[test]
fn a_devices_own_memory_is_charged_as_it_opens_before_its_store_is_read() {
    use seismic::memory::{Backed, Refused};
    const OPENING: u64 = 16 << 10;
    let scope = Ledgers::new();
    let [opening] = <[_; 1]>::try_from(scope.stores(vec![seismic::StoreIdentity::HostRam])).ok().expect("one store");
    let order = Backed::take(&opening.own(), OPENING, Pageable, |_| Ok::<_, Refused>(())).expect("opening's own bytes are admitted by no reading");
    let probe = Backed::take(&opening.own(), OPENING, Pageable, |_| Ok::<_, Refused>(())).expect("a second own entry's too");
    assert_eq!(opening.allocated(), 2 * OPENING, "the device's own memory is charged and counted");
    let sample = Sample::scripted(scope.changes(), seismic::host_samples::m4_pro());
    let (ledgers, reading) = DeviceLedgers::unified_in(&scope, shape()).read_granted(&sample, DeviceGrant::Wired { working_set: seismic::fixtures::M4_PRO_WORKING_SET });
    assert_eq!(reading.device.allocated(), 2 * OPENING, "the first reading counts what opening charged");
    assert!(ledgers.device().change(|keeping| ledgers.device().claim_at(&keeping.claims(), Some(&reading.device), CommittedKind::Load, GB)).is_ok(), "a load claims after the first reading, judged by it");
    drop((order, probe));
    assert_eq!(opening.allocated(), 0, "freed with the device's objects");
}

/// Unified Metal on a host that reads no wired figures (an Intel Mac's
/// integrated GPU, whose macOS defines no wired limit) is bounded by the
/// device's working set alone, a static ceiling, where the reading needed
/// a wired ceiling before.
#[test]
fn unified_metal_without_wired_figures_is_bounded_by_its_working_set() {
    let sample = seismic::HostSample { wired: None, ..seismic::host_samples::m4_pro() };
    let read = |grant| {
        let scope = Ledgers::new();
        let (_ledgers, reading) = DeviceLedgers::unified_in(&scope, shape()).read_granted(&Sample::scripted(scope.changes(), sample), grant);
        reading.device
    };
    let room = read(DeviceGrant::Host).budget();
    let working_set = room / 2;
    let unified = read(DeviceGrant::Wired { working_set });
    assert_eq!(unified.wired_limit(), None, "no wired ceiling is read");
    assert_eq!(unified.budget(), working_set, "the working set bounds the room");
}

/// What opening charges is the device's own entries': any other entry
/// allocating in a store never read is an invariant violation.
#[test]
#[should_panic(expected = "is charged before the store's first reading")]
fn an_entry_other_than_a_devices_own_allocating_in_a_store_never_read_panics() {
    let (_scope, ledger, pool) = host();
    let _ = ledger.change(|_| pool.take(&pool.change(), GB, Pageable));
}

/// The laptop's load: the target import allocated the bank 16.3 GB past
/// its floor, leaving 2.4 GB available. The bank's slots are granules of
/// the pool, charged once to its entry as the import allocates its blocks;
/// the floor's claim and the slots above it hold them, never claimed again
/// against a reading that already shows them allocated: nothing is
/// refused, free room and others' memory stand, and no guarantee rises.
#[test]
fn a_loads_bank_is_charged_once_to_the_pool() {
    const ABOVE: u64 = 16_331_243_520;
    const AVAILABLE: u64 = 2_444_917_344;
    const FLOOR: u64 = 2 * GB;
    const TOTAL: u64 = 36 * GB;
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    let others = TOTAL - FLOOR - ABOVE - AVAILABLE;
    read(&ledger, &mut clock, TOTAL, TOTAL - others, None);
    let floor = ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::BankFloor { bytes: FLOOR })).expect("the floor fits");
    let bank = ledger.cache(CacheKind::Bank, ABOVE, Reclaim::Deferred).expect("the slots above it fit");
    // The import allocates the blocks; the next reading shows them.
    ledger.change(|_| pool.take(&pool.change(), FLOOR + ABOVE, Pageable)).expect("the holdings hold the blocks' granules");
    read(&ledger, &mut clock, TOTAL, AVAILABLE, None);
    let served = ledger.estimate();
    assert_eq!((served.others(), served.free(), served.guarantee()), (others, AVAILABLE, 0));
    assert_eq!((served.allocated(), served.cache_of(CacheKind::Bank), served.cache_of(CacheKind::Spare)), (FLOOR + ABOVE, ABOVE, 0));
    assert_eq!((floor.allocated_bytes(), bank.bytes()), (0, ABOVE), "the holdings allocate nothing of their own");
}

/// The bank's room counts the records pool's spare granules, which its
/// slots grow into, and never the state pool's: those lie in granules of
/// another class, beside the bank, as every KV slab does.
#[test]
fn the_banks_room_holds_no_spare_of_the_state_pool() {
    const ABOVE: u64 = 12 * GB;
    const AVAILABLE: u64 = 2 * GB;
    const FLOOR: u64 = 2 * GB;
    const STATE_SPARE: u64 = GB;
    const TOTAL: u64 = 36 * GB;
    let (_scope, ledger, pools) = host();
    let mut clock = Clock::new();
    let others = TOTAL - FLOOR - ABOVE - STATE_SPARE - AVAILABLE;
    read(&ledger, &mut clock, TOTAL, TOTAL - others, None);
    let _floor = ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::BankFloor { bytes: FLOOR })).expect("the floor fits");
    let _bank = ledger.cache(CacheKind::Bank, ABOVE, Reclaim::Deferred).expect("the slots above it fit");
    ledger.change(|_| pools.records.take(&pools.records.change(), FLOOR + ABOVE, Pageable)).expect("the records pool holds the slots");
    ledger.change(|_| pools.state.take(&pools.state.change(), STATE_SPARE, Pageable)).expect("the state pool holds a spare block");
    read(&ledger, &mut clock, TOTAL, AVAILABLE, None);
    let estimate = ledger.estimate();
    assert_eq!(estimate.cache_of(CacheKind::Spare), STATE_SPARE, "the state pool's block is spare");
    assert_eq!(estimate.import_room(), ABOVE + AVAILABLE, "the bank's room: its slots and the supply, not the state's spare");
}

/// A claim taken but not yet allocated, such as a measurement job's 4 GiB
/// room, stays out of free room across a reading: the reading shows none
/// of it, and it never reads as free.
#[test]
fn a_claim_not_yet_allocated_stays_out_of_free_room_across_a_reading() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let room = held(&ledger, 4 * GB).expect("room");
    assert_eq!(ledger.estimate().free(), 20 * GB);
    read(&ledger, &mut clock, 64 * GB, 24 * GB, None);
    let after = ledger.estimate();
    assert_eq!((after.others(), after.free(), after.guarantee()), (40 * GB, 20 * GB, 0));
    assert!(ledger.cached(CacheKind::Bank, 21 * GB).is_err(), "the bank never grows into the room");
    drop(room);
}

/// A closed group's class turns cache in place: one entry, its bytes and
/// their allocation unchanged, so no reading moves; a claim then takes it
/// just before the bank.
#[test]
fn a_claim_turns_cache_in_place_and_yields_before_the_bank() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let class = commit(&ledger, 4 * GB);
    let bank = ledger.cached(CacheKind::Bank, 60 * GB).expect("the rest");
    read(&ledger, &mut clock, 64 * GB, 0, None);
    let before = ledger.estimate();
    let entries = ledger.entries().len();
    let slots = class.into_test_cache(CacheKind::GraphSlots);
    let after = ledger.estimate();
    assert_eq!(ledger.entries().len(), entries, "the same entry");
    assert_eq!((after.allocated(), after.free(), after.others()), (before.allocated(), before.free(), before.others()));
    assert_eq!((after.committed(), after.cache_of(CacheKind::GraphSlots)), (before.committed() - 4 * GB, 4 * GB));
    let _taken = held(&ledger, 5 * GB).expect("cache yields to it");
    assert_eq!((slots.bytes(), bank.bytes()), (0, 59 * GB), "the idle slots yield first, then the bank");
}

/// Deferred cache (no synchronous reclaimer) is never taken directly by
/// admission: it must be shed through a relief plan and physically freed by
/// its holder before a claim can admit or allocate those bytes. Until freed,
/// the yielded bytes stay releasing so readings count them neither others'
/// nor free.
#[test]
fn a_claim_allocates_taken_cache_only_as_its_holder_frees_it() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 20 * GB);
    let bank = ledger.cached_with(CacheKind::Bank, 40 * GB, Reclaim::Deferred).expect("free room");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert!(held(&ledger, 10 * GB).is_err(), "deferred cache is not admitted without physical free");
    let plan = ledger.plan_relief(6 * GB, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("bank covers 6 GB");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, 6 * GB);
    assert_eq!((bank.bytes(), bank.releasing()), (34 * GB, 0), "the bank yields 6 GB, the pool's still");
    let held_est = ledger.estimate();
    assert_eq!((held_est.allocated(), held_est.releasing(), held_est.allocatable()), (60 * GB, 6 * GB, 4 * GB), "only the free 4 GB are allocatable");
    assert!(held(&ledger, 10 * GB).is_err(), "still unfreed: admission waits for physical free");
    // The bank has not freed them: the next reading counts them the process's.
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert_eq!(ledger.estimate().others(), 0, "the yielded bytes read as the process's");
    ledger.change(|_| pool.freed(&pool.change(), 6 * GB, Pageable));
    let freed = ledger.estimate();
    assert_eq!((freed.allocated(), freed.releasing(), freed.allocatable()), (54 * GB, 0, 10 * GB), "freed, all 10 GB are allocatable");
    read(&ledger, &mut clock, 64 * GB, 10 * GB, None);
    assert_eq!(ledger.estimate().others(), 0);
    let taken = held(&ledger, 10 * GB).expect("freed, the claim is admitted");
    ledger.change(|_| taken.entry().take(&taken.entry().change(), 10 * GB, Pageable)).expect("freed, the claim allocates its bytes");
}

/// Cache shed to a shortage is held by no entry and allocated still, until
/// its holder frees it: it is never free room meanwhile, so no new holding
/// takes memory the process still has.
#[test]
fn a_shortages_yield_is_free_only_once_freed() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 40 * GB);
    let bank = ledger.cached_with(CacheKind::Bank, 20 * GB, Reclaim::Deferred).expect("free room");
    assert_eq!(ledger.estimate().free(), 4 * GB);
    let plan = ledger.plan_relief(8 * GB, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("the bank covers it");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, 8 * GB);
    assert_eq!((bank.bytes(), ledger.estimate().free()), (12 * GB, 4 * GB), "shed, still allocated: not free");
    assert!(ledger.cached(CacheKind::Prefix, 5 * GB).is_err());
    ledger.change(|_| pool.freed(&pool.change(), 8 * GB, Pageable));
    assert_eq!(ledger.estimate().free(), 12 * GB, "freed, it is free");
}

/// A shortage preempts the sequences of least value first, whatever their
/// admission, and never the earliest admitted: here the second, then the
/// fourth, just enough.
#[test]
fn a_shortage_preempts_the_lowest_value_first() {
    let (_scope, ledger, _pool) = host();
    read(&ledger, &mut Clock::new(), 64 * GB, 64 * GB, Some(40 * GB));
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    let slot = slot(&ledger, page);
    // No cache: the shortage's relief is the requests'.
    let [first, second, third, fourth] = [1, 2, 3, 4].map(|sequence| request(&ledger, &slot, sequence, 1).expect("room"));
    let per_page = NonZeroU64::new(page).unwrap();
    for (request, seconds) in [(&first, 0.0), (&second, 0.1), (&third, 0.9), (&fourth, 0.5)] {
        request.value(Value::per_byte(seconds, per_page));
    }
    // The grant falls to two and a half pages: the four pages are a page
    // and a half over it.
    read(&ledger, &mut Clock::new(), 64 * GB, 64 * GB, Some(2 * page + page / 2));
    let shortage = ledger.shortage(&[1, 2, 3, 4].map(RequestId).into()).expect("committed past the budget");
    assert_eq!(shortage.response.relief.as_ref().map(ReliefPlan::preempt), Some(&[RequestId(2), RequestId(4)][..]));
}

/// A bank holding dropped leaves its granules spare in the pool, still
/// allocated: the pool's frees let go of what the bank yielded first, then
/// of the spare.
#[test]
fn a_dropped_bank_leaves_spare_granules_until_the_pool_frees_them() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let bank = ledger.cached_with(CacheKind::Bank, 64 * GB, Reclaim::Deferred).expect("free room");
    let plan = ledger.plan_relief(8 * GB, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("the bank covers 8 GB");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, 8 * GB);
    drop(bank);
    let dropped = ledger.estimate();
    assert_eq!((dropped.allocated(), dropped.releasing(), dropped.cache_of(CacheKind::Spare)), (64 * GB, 8 * GB, 56 * GB));
    ledger.change(|_| pool.freed(&pool.change(), 64 * GB, Pageable));
    let freed = ledger.estimate();
    assert_eq!((freed.allocated(), freed.releasing(), freed.cache_of(CacheKind::Spare), freed.allocatable()), (0, 0, 0, 64 * GB));
}

/// A backing's claim holds its bytes from free room at once; the pool
/// allocates its blocks as the slots back, charged to the pool's entry and
/// never to the backing's; ended, the backing is cache whole.
#[test]
fn a_backing_holds_its_bytes_and_the_pool_allocates_them() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let backing = ledger.change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, 8 * GB)).expect("free room");
    assert!(ledger.change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, 57 * GB)).is_err(), "its bytes are held from the start");
    assert_eq!((ledger.estimate().allocated(), ledger.estimate().free()), (0, 56 * GB));
    ledger.change(|_| pool.take(&pool.change(), 3 * GB, Pageable)).expect("the backing holds the block's granules");
    assert_eq!((ledger.estimate().allocated(), ledger.estimate().free(), backing.allocated_bytes()), (3 * GB, 56 * GB, 0));
    ledger.change(|_| pool.take(&pool.change(), 2 * GB, Pageable)).expect("the backing holds the block's granules");
    let cache = backing.into_backed_cache(CacheKind::Bank, Reclaim::Deferred);
    let snapshot = ledger.estimate();
    assert_eq!((cache.bytes(), snapshot.allocated(), snapshot.cache_of(CacheKind::Bank), snapshot.free()), (8 * GB, 5 * GB, 8 * GB, 56 * GB));
}

/// The request slot is the first request's: it reserves what the slot
/// holds, holding nothing more than the slot did, and a second request
/// beside it draws free room. With every byte of the budget held, a third
/// has no room; the first's close gives the slot back, and the third
/// draws it then.
#[test]
fn a_reservation_draws_the_slot_and_gives_it_back() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 8 * GB, None);
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    let slot = slot(&ledger, page);
    let committed = ledger.estimate().committed();
    let first = request(&ledger, &slot, 1, 1).expect("the first request draws the slot");
    assert_eq!((ledger.estimate().committed(), slot.bytes()), (committed, 0), "the slot's bytes are held once");
    let free = ledger.estimate().free();
    let _second = request(&ledger, &slot, 2, 1).expect("the second draws free room beside the slot in use");
    assert_eq!((ledger.estimate().committed(), ledger.estimate().free()), (committed + page, free - page), "the second's page is free room's");
    let _rest = commit(&ledger, ledger.estimate().free());
    assert!(request(&ledger, &slot, 3, 1).is_err(), "a third request has no room");
    drop(first);
    assert_eq!(slot.bytes(), page, "the close gives the slot back");
    let _third = request(&ledger, &slot, 3, 1).expect("the third draws it now");
}

/// A grant's process's footprint is the process's own, within the grant:
/// a measurement job that grows by exactly its grant leaves others, their
/// rise and the guarantee as they were, and causes nothing; one byte past
/// it is others'.
#[test]
fn a_grants_footprint_is_never_others_memory() {
    struct Scripted(std::sync::Mutex<u64>);
    impl Footprints for Scripted {
        fn footprint(&self, _: seismic::ProcessId, _: &seismic::StoreIdentity) -> seismic::ProcessFootprint {
            seismic::ProcessFootprint::Observed(*self.0.lock().unwrap())
        }
    }
    let footprint = Arc::new(Scripted(std::sync::Mutex::new(0)));
    let scope = Ledgers::footprinted(footprint.clone());
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 30 * GB);
    read(&ledger, &mut clock, 64 * GB, 34 * GB, None);
    let grant = ledger.change(|keeping| ledger.grant(&keeping.claims(), GrantRole::Measurement, 4 * GB)).expect("room");
    grant.attach(seismic::ProcessId::of(1));
    let before = ledger.estimate();
    *footprint.0.lock().unwrap() = 4 * GB;
    read(&ledger, &mut clock, 64 * GB, 30 * GB, None);
    let grown = ledger.estimate();
    let figures = |estimate: &Estimate| (estimate.others(), estimate.guarantee(), estimate.budget(), estimate.free());
    assert_eq!(figures(&grown), figures(&before), "the job's growth is its grant's");
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "and causes nothing");
    *footprint.0.lock().unwrap() = 4 * GB + 1;
    read(&ledger, &mut clock, 64 * GB, 30 * GB - 1, None);
    assert_eq!(ledger.estimate().others(), before.others() + 1, "a byte past the grant is others'");
}

/// The bank's target as the catalog's failures met it (taken over from the
/// bank's own tests, in bytes): one store of the laptop's 64 GB, others'
/// memory handed in, slots of 0.4 GB.
mod target {
    use super::*;

    const TOTAL: u64 = 64 << 30;
    const SLOT: u64 = 400 << 20;
    /// The laptop catalog's bank: the slots a 64 GB store with 10 GB of
    /// committed weights holds (10 GB + 132 * 400 MiB < 64 GiB). The tests
    /// prove the target moves by claimed bytes, whatever the count.
    const SLOTS: u64 = 132;

    /// A reading with other programs holding `others`: the OS shows free
    /// what neither they nor the process's allocations hold.
    fn read(ledger: &Ledger, clock: &mut Clock, others: u64) {
        let allocated = ledger.state().allocated;
        let available = TOTAL.checked_sub(others + allocated).expect("others and the process fit the store");
        super::read(ledger, clock, TOTAL, available, None);
    }

    /// Item 9: the laptop's sequence. A bank of 132 slots past its floor in
    /// blocks of one slot, a load serving one sequence; others take a third
    /// of a slot for a reading and give it back, while the sequence's
    /// decode claims its state a page at a time: the bank resizes at no
    /// reading. The pages are written within the request's reservation,
    /// so the target never moves with them; others' jitter
    /// moves it by less than a block, which the bank keeps: a rise lowers
    /// the budget by its bytes and raises the guarantee by them.
    #[test]
    fn the_laptops_bank_holds_through_a_prompt_and_its_decode() {
        use crate::experts::derived::{bank_move, held_target, BankShape};
        let (_scope, ledger, _pool) = host();
        let mut clock = Clock::new();
        let committed = 10 * GB;
        let pages = 41;
        let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
        let whole = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: pages * page_rows() });
        // A measurement job's room beside the load, none of it allocated
        // yet: what others' jitter takes of the store's free memory.
        let job = SLOT;
        let base = TOTAL - committed - whole - job - SLOTS * SLOT;
        read(&ledger, &mut clock, base);
        let _weights = commit(&ledger, committed);
        let _job = ledger.change(|keeping| ledger.grant(&keeping.claims(), GrantRole::Measurement, job)).expect("the job's room");
        let slot = slot(&ledger, whole);
        let mut sequence = request(&ledger, &slot, 1, pages).expect("the slot");
        let _bank = ledger.cached(CacheKind::Bank, SLOTS * SLOT).expect("the bank's room");
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), SLOTS * SLOT, "the bank as it stood");
        let shape = BankShape::new(SLOT, 0, 226, 226);
        let charge = |slots: u64| slots * SLOT;
        let resizes = |target: u64| bank_move(held_target(target, 132, 0, &shape, &charge), 132);
        let mut held = 1;
        for reading in 0..40 {
            // The decode's next page, claimed as it grows.
            sequence.grow((held + 1) * page_rows());
            held += 1;
            let jitter = if reading % 2 == 0 { SLOT / 3 } else { 0 };
            read(&ledger, &mut clock, base + jitter);
            let target = ledger.estimate().target();
            assert_eq!(target, SLOTS * SLOT - 2 * jitter, "reading {reading}: only others' jitter moves the target, in the budget and the guarantee");
            assert_eq!(resizes(target), None, "reading {reading}: no resize");
        }
        assert!(held * page <= sequence.bytes(), "the decode stayed within its reservation");
    }

    /// Items 4, 5, 12, 13, 15, 44: growth is never dropped. As another
    /// program gives its memory back over readings, each reading's target
    /// follows it up by every byte the freed room holds.
    #[test]
    fn the_target_follows_freed_memory_up_at_every_reading() {
        let (_scope, ledger, _pool) = host();
        let mut clock = Clock::new();
        let committed = 10 * GB;
        read(&ledger, &mut clock, TOTAL - committed - SLOT);
        let _weights = commit(&ledger, committed);
        let mut before = 0;
        for slots in (20..=SLOTS).step_by(20) {
            read(&ledger, &mut clock, TOTAL - committed - slots * SLOT);
            let target = ledger.estimate().target();
            assert_eq!(target, slots * SLOT, "a fall of others' memory is no rise");
            assert!(target >= before);
            before = target;
        }
    }

    /// Items 3, 42: a committed claim lowers the target by its bytes at the
    /// reading that pairs it with the OS's figure, and its release raises
    /// it back.
    #[test]
    fn a_committed_claim_moves_the_target_by_its_bytes() {
        let (_scope, ledger, _pool) = host();
        let mut clock = Clock::new();
        let base = TOTAL - 10 * GB - SLOTS * SLOT;
        read(&ledger, &mut clock, base);
        let _weights = commit(&ledger, 10 * GB);
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), SLOTS * SLOT);
        let job = commit(&ledger, 20 * SLOT);
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), (SLOTS - 20) * SLOT, "a measurement job's room");
        drop(job);
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), SLOTS * SLOT, "given back");
    }

    /// The guarantee is the only margin: a request's reservation lowers the
    /// target by its bytes as it opens, and committed memory filling the
    /// budget leaves none.
    #[test]
    fn the_guarantee_is_the_only_margin() {
        let (_scope, ledger, _pool) = host();
        let mut clock = Clock::new();
        let base = TOTAL - 10 * GB - SLOTS * SLOT;
        read(&ledger, &mut clock, base);
        let _weights = commit(&ledger, 10 * GB);
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), SLOTS * SLOT, "idle: nothing reserved");
        let slot = slot(&ledger, 0);
        let serving = request(&ledger, &slot, 1, 3).expect("room");
        assert_eq!(ledger.estimate().target(), SLOTS * SLOT - serving.bytes(), "serving: the request's reservation");
        drop((serving, slot));
        let _rest = commit(&ledger, SLOTS * SLOT);
        read(&ledger, &mut clock, base);
        assert_eq!(ledger.estimate().target(), 0, "committed fills the budget");
    }

    /// A load plans by the estimate it will hold: what the plan prices at
    /// load (`Ledger::room_after` the load sample, nothing claimed yet) is
    /// the target its first reading derives once the load holds it.
    #[test]
    fn a_load_holds_what_its_plan_priced() {
        let (_scope, ledger, _pool) = host();
        let mut clock = Clock::new();
        let others = 40 * GB;
        read(&ledger, &mut clock, others);
        let planned = ledger.room_after(&ledger.estimate()).room() - 7 * GB;
        let _held = commit(&ledger, 7 * GB);
        let _cache = ledger.cached(CacheKind::Bank, planned).expect("the plan's room holds its cache");
        read(&ledger, &mut clock, others);
        assert_eq!(ledger.estimate().target(), planned);
    }
}

/// Cache yields by value per byte, least first: a used prefix outlasts a
/// bank whose miss curve values it less, spare goes first.
#[test]
fn cache_yields_by_value_per_byte() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _weights = commit(&ledger, 40 * GB);
    let bank = ledger.cached(CacheKind::Bank, 8 * GB).expect("room");
    let prefix = ledger.cached(CacheKind::Prefix, 8 * GB).expect("room");
    let spare = ledger.cached(CacheKind::Spare, 8 * GB).expect("room");
    let gb = NonZeroU64::new(8 * GB).unwrap();
    bank.value(Value::per_byte(1.0, gb));
    prefix.value(Value::recompute(0.5, 4, gb));
    let plan = ledger.plan_relief(12 * GB, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("the cache covers it");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, 12 * GB);
    assert_eq!((spare.bytes(), bank.bytes(), prefix.bytes()), (0, 4 * GB, 8 * GB), "spare, then the bank worth less");
}

/// Seismic allocates through an entry: what an entry holds unallocated
/// serves first, what it lacks is admitted, what a driver took is never
/// refused, and what it freed is neither allocated nor held.
#[test]
fn seismic_allocates_through_its_entry() {
    use seismic::memory::Store as _;
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 8 * GB, 8 * GB, None);
    let own = ledger.own();
    assert_eq!(own.take(&own.change(), 2 * GB, Pageable).err().map(|refused| refused.required), None);
    own.took(&own.change(), GB, Pageable);
    let estimate = ledger.estimate();
    assert_eq!((estimate.committed(), estimate.allocated()), (3 * GB, 3 * GB));
    assert!(own.take(&own.change(), 6 * GB, Pageable).is_err(), "past the budget: refused, nothing allocated");
    assert_eq!(ledger.estimate().allocated(), 3 * GB);
    own.freed(&own.change(), 2 * GB, Pageable);
    let estimate = ledger.estimate();
    assert_eq!((estimate.committed(), estimate.allocated()), (GB, GB));
}

/// A change keeps the cache it takes up again (an idle component made
/// active again) committed in place, whole, in its keep phase; a holding an
/// earlier claim took any of comes back as cache, every entry as it was,
/// its value kept.
#[test]
fn a_change_keeps_cache_whole_or_not_at_all() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let weights = ledger.cached_with(CacheKind::Drafter, 4 * GB, Reclaim::Deferred).expect("room");
    let active = ledger.change(|keeping| keeping.keep(vec![weights], CommittedKind::Component)).ok().expect("whole");
    assert_eq!((ledger.estimate().committed(), ledger.estimate().cache()), (4 * GB, 0));
    let weights = active.into_iter().next().expect("one claim").into_cache(CacheKind::Drafter, Reclaim::Deferred);
    weights.value(Value::per_byte(2.0, NonZeroU64::new(4 * GB).unwrap()));
    let binding = ledger.cached_with(CacheKind::Drafter, GB, Reclaim::Deferred).expect("room");
    let rest = ledger.cached(CacheKind::Bank, 59 * GB).expect("room");
    rest.value(Value::per_byte(1.0, NonZeroU64::new(59 * GB).unwrap()));
    // The binding, worth least, yields first to relief.
    let plan = ledger.plan_relief(GB / 2, Relief::CacheOnly, crate::memory::ledger::Frees::Room).expect("binding covers 0.5 GB");
    assert_eq!(ledger.change(|keeping| ledger.execute(&keeping.claims(), plan)).yielded, GB / 2);
    let before = ledger.estimate();
    let kept = ledger.change(|keeping| keeping.keep(vec![weights, binding], CommittedKind::Component));
    let cached = kept.err().expect("the binding was taken in part");
    assert_eq!(cached.iter().map(Cached::kind).collect::<Vec<_>>(), [CacheKind::Drafter; 2]);
    assert_eq!((ledger.estimate().committed(), ledger.estimate().cache()), (before.committed(), before.cache()));
    // Its value kept, the weights yield after the bank.
    let _more = held(&ledger, GB).expect("cache yields");
    assert_eq!(cached[0].bytes(), 4 * GB, "the weights kept their value as they came back");
}

/// A change keeps nothing once it claims: a change joined to one that
/// claims already panics at a keep, so no claim of a change takes the cache
/// it keeps.
#[test]
#[should_panic(expected = "a change keeps cache after it claimed")]
fn a_change_keeps_nothing_once_it_claims() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let drafter = ledger.cached(CacheKind::Drafter, 4 * GB).expect("room");
    ledger.change(|keeping| {
        let _claim = ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, GB).expect("room");
        let _kept = ledger.change(|joined| joined.keep(vec![drafter], CommittedKind::Component));
    });
}

/// One charged allocation counts once: recording a claim or a cache holding
/// moves only what is held, and Seismic's allocation through the entry is
/// the one record of it allocated.
#[test]
fn one_charged_allocation_counts_once() {
    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 64 * GB, None);
    let _bank = ledger.cache(CacheKind::Bank, 4 * GB, Reclaim::Deferred).expect("room");
    let claim = held(&ledger, 2 * GB).expect("room");
    let recorded = ledger.estimate();
    assert_eq!((recorded.allocated(), recorded.cache() + recorded.committed()), (0, 6 * GB), "recorded, held, none allocated");
    ledger.change(|_| {
        pool.take(&pool.change(), 4 * GB, Pageable).expect("the bank holds the block's granules");
        claim.entry().take(&claim.entry().change(), 2 * GB, Pageable).expect("the claim's own bytes");
    });
    assert_eq!(ledger.estimate().allocated(), 6 * GB, "each byte once");
    read(&ledger, &mut clock, 64 * GB, 58 * GB, None);
    assert_eq!(ledger.estimate().others(), 0, "the reading shows them once");
}

/// Two loads watching host RAM's store share its one watch: the second
/// starts none of its own, each event the watch delivers is counted once
/// for the store's next reading and wakes both loads, and the watch stops
/// only as the last load's drops. Where the OS gives no unprivileged
/// events, neither load watches.
#[test]
fn two_loads_share_one_pressure_watch_and_count_each_event_once() {
    use std::sync::atomic::AtomicUsize;
    let scope = Ledgers::new();
    let (first, second) = (scope.ledger(seismic::StoreIdentity::HostRam, shape()), scope.ledger(seismic::StoreIdentity::HostRam, shape()));
    let woken = |count: &Arc<AtomicUsize>| {
        let count = count.clone();
        move || {
            count.fetch_add(1, Ordering::SeqCst);
        }
    };
    let (a, b) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (watch_a, watch_b) = (first.watch_pressure(woken(&a)), second.watch_pressure(woken(&b)));
    assert_eq!(watch_a.is_some(), watch_b.is_some(), "the OS gives both loads events or neither");
    if watch_a.is_none() {
        return;
    }
    let started = |ledger: &Ledger| matches!(ledger.store.pressure.watchers.lock().unwrap().watch, pressure::Watch::Events(_));
    assert!(started(&first) && Arc::ptr_eq(&first.store.pressure, &second.store.pressure), "one watch for the store");
    first.pressure_event();
    assert_eq!((a.load(Ordering::SeqCst), b.load(Ordering::SeqCst)), (1, 1), "the event wakes both loads");
    assert_eq!(first.store.pressure.take(), 1, "and is counted once");
    drop(watch_a);
    assert!(started(&second), "the watch stays while a load watches");
    second.pressure_event();
    assert_eq!((a.load(Ordering::SeqCst), b.load(Ordering::SeqCst)), (1, 2), "a dropped watch is woken no more");
    drop(watch_b);
    assert!(!started(&second), "the last watch's drop stops it");
}

/// A load's claims find the room its search priced, with no device: a load
/// prices the reading it holds, and its claims find that same room with
/// nothing held since (`Ledger::room_after`, the one each store's room is
/// read by), then less exactly what they took (`Ledger::claim_at`, the one
/// each store's claim is taken by). A later reading of the store, others
/// taking 36 GB of the 40 GB the load priced, moves neither: 64 GB, 40 GB
/// available, so the search's room is 40 GB, 34 GB after a 6 GB claim,
/// and a 30 GB claim still fits at the held reading. It does
/// not prove the plan's size is that room: `plan_load` sizes the plan from
/// the assessment capacity, not from the reading.
#[test]
fn a_loads_claims_find_the_room_its_search_priced() {
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    let held = ledger.read(ledger.sampled(clock.next()), device(64 * GB, 40 * GB));
    let searched = ledger.room_after(&held).room();
    assert_eq!(searched, 40 * GB, "the supply");
    let _claim = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), Some(&held), CommittedKind::Load, 6 * GB)).expect("the search's room holds it");
    assert_eq!(ledger.room_after(&held).room(), searched - 6 * GB, "less exactly the claim");
    read(&ledger, &mut clock, 64 * GB, 4 * GB, None);
    assert_eq!(ledger.estimate().others(), 60 * GB, "the store's latest reading moved");
    assert_eq!(ledger.room_after(&held).room(), searched - 6 * GB, "the load's reading stands");
    assert!(ledger.change(|keeping| ledger.claim_at(&keeping.claims(), Some(&held), CommittedKind::Load, 30 * GB)).is_ok(), "claims at the held reading");
}

/// The 48 GB M4 Pro's recorded idle sample, read by the path a load on
/// unified memory takes (`Sample::settled`, then `DeviceLedgers::read`'s
/// fold of `Reading::unified` into the store's ledger), prices a load's
/// room at the wired ceiling's grant: the 38,654,705,664-byte working set
/// less the 1,795,801,088 bytes others wire, 36,858,904,576 bytes, beside
/// 47,779,758,080 of supply in the 51,539,607,552-byte store. A load
/// claims the whole room it prices, and one byte past it is refused.
///
/// It proves the room the ledgers grant from a recorded sample: the one
/// room a load's plan settles into (`PlanningRoom`) and its claims read, so
/// a plan that fits the store is granted by it.
#[test]
fn the_m4_pros_recorded_sample_prices_the_wired_grant() {
    const STORE: u64 = 51_539_607_552;
    const GRANT: u64 = 36_858_904_576;
    let scope = Ledgers::new();
    let recorded = seismic::host_samples::m4_pro();
    assert_eq!(recorded.total, STORE);
    let sample = Sample::scripted(scope.changes(), recorded);
    let (ledgers, reading) = DeviceLedgers::unified_in(&scope, shape()).read_granted(&sample, DeviceGrant::Wired { working_set: seismic::fixtures::M4_PRO_WORKING_SET });
    let ledger = ledgers.device();
    let held = &reading.device;
    assert_eq!((held.supply(), held.others()), (47_779_758_080, 3_759_849_472), "free above the target and clean cache; others' the rest");
    assert_eq!(held.budget(), GRANT, "the working set less others' wired bytes");
    let room = reading.room_after(&ledgers).device.room();
    assert_eq!(room, GRANT, "the grant");
    let _plan = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), Some(held), CommittedKind::Load, room)).expect("the room the load prices is admitted");
    assert_eq!(reading.room_after(&ledgers).device.room(), 0, "the claim takes the whole room");
    let past = 1;
    assert_eq!(
        ledger.change(|keeping| ledger.claim_at(&keeping.claims(), Some(held), CommittedKind::Load, past)).err(),
        Some(Short { store: ledger.store(), required: past, over: 1 }),
        "one byte past it is refused"
    );
}

/// `PlanningRoom::capacity()` returns the store's structural capacity
/// (`total.min(grant_limit)`), never momentary free room (`net()`), so a load
/// starting under memory pressure plans a bank ceiling that can grow into
/// freed RAM later.
#[test]
fn planning_room_capacity_is_structural_not_momentary_free() {
    use crate::domain::PlanningRoom;
    use crate::SettleRoom;

    // 64 GiB Mac with 48 GiB working set, but only 20 GiB currently free because
    // another app holds ~44 GiB.
    const TOTAL_64G: u64 = 64 * GB;
    const WORKING_SET_48G: u64 = 48 * GB;
    const FREE_20G: u64 = 20 * GB;
    let scope = Ledgers::new();
    let mut sample_data = seismic::host_samples::m4_pro();
    sample_data.total = TOTAL_64G;
    sample_data.supply = seismic::Supply {
        free_beyond_target: i64::try_from(FREE_20G).unwrap(),
        clean_cache: 0,
        cold_reclaimable: 0,
    };
    sample_data.wired = Some(seismic::WiredFigures {
        limit: seismic::WiredLimit::OsDefault,
        wire_count: 8 * GB,
    });
    let sample = Sample::scripted(scope.changes(), sample_data);
    let (ledgers, reading) = DeviceLedgers::unified_in(&scope, shape())
        .read_granted(&sample, DeviceGrant::Wired { working_set: WORKING_SET_48G });
    let room = PlanningRoom::new(&reading, &ledgers);
    assert_eq!(room.net(), FREE_20G, "momentary free room at load is 20 GiB");
    assert_eq!(
        room.capacity().unwrap(),
        WORKING_SET_48G,
        "structural capacity is 48 GiB working set, not momentary 20 GiB free"
    );

    // Unified Linux CUDA (128 GiB host, 120 GiB CUDA limit) with 30 GiB free at load.
    let mut linux_sample = seismic::host_samples::m4_pro();
    linux_sample.total = 128 * GB;
    linux_sample.wired = None;
    linux_sample.supply = seismic::Supply {
        free_beyond_target: i64::try_from(30 * GB).unwrap(),
        clean_cache: 0,
        cold_reclaimable: 0,
    };
    let l_scope = Ledgers::new();
    let l_sample = Sample::scripted(l_scope.changes(), linux_sample);
    let (l_ledgers, l_reading) = DeviceLedgers::unified_in(&l_scope, shape())
        .read_granted(&l_sample, DeviceGrant::HostLimit { limit: 120 * GB });
    let l_room = PlanningRoom::new(&l_reading, &l_ledgers);
    assert_eq!(l_room.net(), 30 * GB, "momentary free room at load is 30 GiB");
    assert_eq!(l_room.capacity().unwrap(), 120 * GB, "structural capacity is 120 GiB CUDA limit");

    // Discrete GPU (24 GiB VRAM) with 10 GiB free at load.
    let discrete = Reading::device(24 * GB, 10 * GB, seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO });
    assert_eq!(discrete.capacity(), 24 * GB, "structural VRAM capacity is 24 GiB");
}

/// A Linux unified-memory host (NVIDIA GB10: CUDA and Vulkan sharing host
/// RAM with `wired == None`) prices CPU, CUDA total-memory, and Vulkan
/// heap-budget grants on the one host-RAM store without requiring macOS
/// wired-memory counters.
#[test]
fn linux_unified_host_prices_cuda_and_vulkan_grants_without_wired_figures() {
    let at = Instant::now();
    let linux_gb10 = seismic::HostSample {
        at,
        page_bytes: 4_096,
        total: seismic::fixtures::GB10_HOST,
        supply: seismic::Supply {
            free_beyond_target: i64::try_from(112 * GB).expect("bytes fit i64"),
            clean_cache: 8 * GB,
            cold_reclaimable: 0,
        },
        counted: seismic::Counted { at, lag: Duration::ZERO },
        cold: seismic::ColdPages::default(),
        own_reclaimed: None,
        wired: None,
        harm: None,
        pressure: None,
        lag: seismic::Lag { release: Duration::ZERO, growth: Duration::ZERO },
    };

    // Unified CUDA on GB10: `path.grant()` is `CudaTotalMemory` (`GB10_CUDA_TOTAL`), `wired` is `None`.
    let cuda_scope = Ledgers::new();
    let cuda_sample = Sample::scripted(cuda_scope.changes(), linux_gb10.clone());
    let (cuda_ledgers, cuda_reading) = DeviceLedgers::unified_in(&cuda_scope, shape())
        .read_granted(&cuda_sample, DeviceGrant::HostLimit { limit: seismic::fixtures::GB10_CUDA_TOTAL });
    assert!(Arc::ptr_eq(&cuda_ledgers.device.store, &cuda_ledgers.host.store));
    assert!(cuda_reading.host.is_none());
    assert_eq!(cuda_reading.device.wired_limit(), None);
    assert_eq!(cuda_reading.device.wire_count(), None);
    assert_eq!(
        cuda_reading.device.budget(),
        112 * GB,
        "without PSI harm monitoring, clean page cache is withheld from Linux host supply"
    );

    // With PSI harm monitoring (`harm: Some(...)`, as on GB10 Linux), clean cache is usable (`120 GB`).
    let mut monitored_gb10 = linux_gb10.clone();
    monitored_gb10.harm = Some(seismic::Harm { machine: 0, own: 0 });
    let monitored_scope = Ledgers::new();
    let monitored_sample = Sample::scripted(monitored_scope.changes(), monitored_gb10);
    let (_, monitored_reading) = DeviceLedgers::unified_in(&monitored_scope, shape())
        .read_granted(&monitored_sample, DeviceGrant::HostLimit { limit: seismic::fixtures::GB10_CUDA_TOTAL });
    assert_eq!(monitored_reading.device.wired_limit(), None);
    assert_eq!(monitored_reading.device.budget(), 120 * GB, "monitored Linux host supply includes clean cache");

    // Unified Vulkan on GB10: `path.grant()` is `VulkanHeaps` (`GB10_VULKAN_HEAP` < `112 GB` supply).
    let vulkan_scope = Ledgers::new();
    let vulkan_sample = Sample::scripted(vulkan_scope.changes(), linux_gb10.clone());
    let vulkan_left = seismic::fixtures::GB10_VULKAN_HEAP - 4 * GB;
    let (_, vulkan_reading) = DeviceLedgers::unified_in(&vulkan_scope, shape())
        .read_granted(&vulkan_sample, DeviceGrant::HostBudget { capacity: seismic::fixtures::GB10_VULKAN_HEAP, left: vulkan_left });
    assert_eq!(vulkan_reading.device.wired_limit(), None);
    assert_eq!(vulkan_reading.device.budget(), vulkan_left, "bounded by the Vulkan heap budget left");

    // CPU on the same unmonitored host: `path.grant()` is `None`.
    let cpu_scope = Ledgers::new();
    let cpu_sample = Sample::scripted(cpu_scope.changes(), linux_gb10);
    let (_, cpu_reading) = DeviceLedgers::unified_in(&cpu_scope, shape())
        .read_granted(&cpu_sample, DeviceGrant::Host);
    assert_eq!(cpu_reading.device.wired_limit(), None);
    assert_eq!(cpu_reading.device.budget(), 112 * GB);
}

/// A native object whose allocation and free are observed against the
/// scope's changes.
struct Observed(Arc<Changes>);

impl Drop for Observed {
    fn drop(&mut self) {
        assert!(self.0.open_here().is_some(), "a native free made outside a change");
    }
}

/// Our own effect and its record are one change, on a thread that holds
/// none: a reading never stands beside the allocation, its record, or the
/// free and the record of it, so neither shows as others' memory.
#[test]
fn an_effect_and_its_record_are_one_change_no_reading_stands_across() {
    use seismic::memory::{Backed, Store as _};
    let scope = Ledgers::strict();
    let _owner = scope.changes().own();
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, shape());
    read(&ledger, &mut Clock::new(), 64 * GB, 64 * GB, None);
    let claim = ledger.change(|_| held(&ledger, GB).expect("the claim fits"));
    let changes = scope.changes().clone();
    let reading = |changes: &Changes| Sampled::attempt(changes, &std::time::Instant::now, &|| (), 0).is_some();
    assert!(reading(&changes), "no change is under way");
    let backed = Backed::take(&claim.entry(), GB, Pageable, |_| {
        assert!(!reading(&changes), "no reading stands beside the allocation");
        Ok::<_, seismic::memory::Refused>(Observed(changes.clone()))
    })
    .expect("the claim's own bytes are admitted");
    assert!(reading(&changes), "the change ended with the record");
    assert_eq!(ledger.allocated(), GB);
    drop(backed);
    assert!(reading(&changes));
    assert_eq!(ledger.allocated(), 0);
    ledger.change(|_| drop(claim));
}

/// An entry's holder dropping is its close: what its allocations keep idle
/// returns then, and an entry already closed runs what is asked at once.
#[test]
fn a_holder_dropping_closes_its_entry() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (_scope, ledger, _pool) = host();
    read(&ledger, &mut Clock::new(), 64 * GB, 64 * GB, None);
    let claim = held(&ledger, GB).expect("the claim fits");
    let entry = claim.entry();
    let ran = Arc::new(AtomicBool::new(false));
    entry.on_close(Box::new({
        let ran = ran.clone();
        move || ran.store(true, Ordering::SeqCst)
    }));
    assert!(!entry.closed() && !ran.load(Ordering::SeqCst));
    ledger.change(|_| drop(claim));
    assert!(entry.closed() && ran.load(Ordering::SeqCst), "the holder's drop ran what was waiting");
    let late = Arc::new(AtomicBool::new(false));
    entry.on_close(Box::new({
        let late = late.clone();
        move || late.store(true, Ordering::SeqCst)
    }));
    assert!(late.load(Ordering::SeqCst), "a closed entry runs it at once");
    ledger.change(|_| drop(entry));
}

/// Settling before an unmonitored idle sleep opens a fresh baseline at the
/// next reading rather than attributing everything that moved while asleep to
/// one step's rise, while active readings after waking chain again.
#[test]
fn a_settled_idle_sleep_opens_a_fresh_baseline_at_the_next_reading() {
    const LAG: Duration = Duration::from_millis(50);
    let (_scope, ledger, _pool) = host();
    let mut clock = Clock::new();
    for _ in 0..3 {
        let at = clock.next();
        let sample = seismic::HostSample {
            at,
            counted: seismic::Counted { at, lag: LAG },
            wired: Some(seismic::WiredFigures { limit: seismic::WiredLimit::Set(22 * GB), wire_count: ledger.state().wired }),
            ..host_sample(48 * GB, 40 * GB)
        };
        ledger.read(ledger.sampled(at), Reading::unified(&sample, u64::MAX));
    }
    assert_eq!(ledger.estimate().guarantee(), 0);
    assert_eq!(ledger.others_interval(), None);

    let wake = clock.next() + Duration::from_secs(15);
    let woken_sample = seismic::HostSample {
        at: wake,
        counted: seismic::Counted { at: wake, lag: LAG },
        harm: Some(seismic::Harm { machine: 1024, own: 0 }),
        wired: Some(seismic::WiredFigures { limit: seismic::WiredLimit::Set(22 * GB), wire_count: 4 * GB }),
        ..host_sample(48 * GB, 20 * GB)
    };
    let woken = ledger.read_after(ledger.sampled(wake), Reading::unified(&woken_sample, u64::MAX), true);
    assert_eq!(woken.guarantee(), 0, "an unmonitored sleep is not one step's supply rise");
    assert_eq!(woken.budget(), 18 * GB, "wired ceiling takes the new wired baseline without a wake wired_rise");
    assert_eq!(woken.room(), 18 * GB);
    assert_eq!(woken.cap(), None, "paging while asleep sets no active boundary");
    assert!(ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "paging while asleep raises no active harm shortage");
    assert_eq!(ledger.standing(), None, "waking into ample free supply raises no false standing");
    assert_eq!(ledger.others_interval(), None, "an unmonitored idle gap is not an active-serving others interval");

    let active = wake + Duration::from_millis(100);
    let active_sample = seismic::HostSample {
        at: active,
        counted: seismic::Counted { at: active, lag: LAG },
        harm: Some(seismic::Harm { machine: 1024, own: 0 }),
        wired: Some(seismic::WiredFigures { limit: seismic::WiredLimit::Set(22 * GB), wire_count: 5 * GB }),
        ..host_sample(48 * GB, 18 * GB)
    };
    let chained = ledger.read(ledger.sampled(active), Reading::unified(&active_sample, u64::MAX));
    assert_eq!(chained.guarantee(), 2 * GB, "active readings after waking chain supply rise");
    assert_eq!(chained.budget(), 16 * GB, "active readings after waking chain wired_rise (22 - 5 - 1 = 16 GB)");
    assert_eq!(chained.room(), 14 * GB);
    assert_eq!(ledger.others_interval(), Some(Duration::from_millis(100)));

    // A pressure event arriving during an idle sleep is never suppressed as an unmonitored wake.
    ledger.pressure_event();
    let pressured_wake = active + Duration::from_secs(10);
    let pressured_sample = seismic::HostSample {
        at: pressured_wake,
        counted: seismic::Counted { at: pressured_wake, lag: LAG },
        harm: Some(seismic::Harm { machine: 1024, own: 0 }),
        wired: Some(seismic::WiredFigures { limit: seismic::WiredLimit::Set(22 * GB), wire_count: 6 * GB }),
        ..host_sample(48 * GB, 15 * GB)
    };
    let pressured = ledger.read_after(ledger.sampled(pressured_wake), Reading::unified(&pressured_sample, u64::MAX), true);
    assert_eq!(pressured.guarantee(), 3 * GB, "a pressured wake preserves supply rise");
}

/// A reservation extends ahead of its rows from what the request slot
/// holds first, then from free room beside it; short, it holds what it
/// held and the extension waits; dropped, it gives the slot back exactly
/// what it drew of it, never the bytes it took beside it.
#[test]
fn a_reservation_extends_from_the_slot_then_free_room_and_gives_back_what_it_drew() {
    let (_scope, ledger, _pools) = host();
    let mut clock = Clock::new();
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    read(&ledger, &mut clock, 64 * GB, 4 * page, Some(64 * GB));
    let slot = slot(&ledger, 2 * page);
    let mut sequence = request(&ledger, &slot, 1, 1).expect("a page of the slot");
    assert_eq!((sequence.bytes(), slot.bytes()), (page, page), "it drew its page of the slot");
    ledger.change(|keeping| ledger.extend(&keeping.claims(), &slot, &mut sequence, 3 * page_rows())).expect("the slot's page, then a page of free room");
    assert_eq!((sequence.rows(), sequence.bytes(), slot.bytes()), (3 * page_rows(), 3 * page, 0), "the slot first, the rest beside it");
    let free = ledger.estimate().free();
    assert!(ledger.change(|keeping| ledger.extend(&keeping.claims(), &slot, &mut sequence, 64 * page_rows())).is_err(), "an extension past free room is short");
    assert_eq!((sequence.rows(), sequence.bytes(), ledger.estimate().free()), (3 * page_rows(), 3 * page, free), "short, it holds what it held");
    drop(sequence);
    assert_eq!(slot.bytes(), 2 * page, "dropped, the slot has back the two pages drawn of it");
}

/// A growth's relief counts all a preempted request holds, since a
/// request's state draws on the slot first; any other claim's counts only
/// what it holds beside the slot, and a request that drew only the slot
/// frees nothing for it.
#[test]
fn a_preempted_request_frees_its_whole_reservation_for_state_and_beside_the_slot_for_room() {
    let (_scope, ledger, _pools) = host();
    let mut clock = Clock::new();
    let page = ledger.bytes(&Need::Request { sequence: RequestId(1), rows: page_rows() });
    read(&ledger, &mut clock, 64 * GB, 4 * page, Some(64 * GB));
    let slot = slot(&ledger, page);
    // The second drew beside the slot the first held; the first gone, the
    // third drew the slot, and the second is now the earliest admitted.
    let first = request(&ledger, &slot, 1, 1).expect("the slot's page");
    let _second = request(&ledger, &slot, 2, 1).expect("a page beside the slot");
    drop(first);
    let _third = request(&ledger, &slot, 3, 1).expect("the slot's page again");
    let every = [2, 3].map(RequestId).into();
    let state = ledger.plan_relief(page, Relief::CacheAndRequests(&every), Frees::RequestState).expect("the third's page covers a growth");
    assert_eq!(state.preempt(), [RequestId(3)], "the slot's page it frees is a growth's");
    assert_eq!(ledger.plan_relief(page, Relief::CacheAndRequests(&every), Frees::Room), None, "it frees nothing beside the slot");
}

/// `Ledger::admit_or_reclaim` invokes a registered [`CacheReclaimer`] outside
/// the ledger mutex so physical memory is freed (`allocatable >= bytes`)
/// before `claim_at` or `backing(PoolClass::State)` hands out a [`Claim`] —
/// and `entry_take` succeeds immediately against it.
#[test]
fn admit_or_reclaim_frees_bank_cache_before_handing_out_claim_and_backing() {
    struct FakeBankReclaimer {
        cached: Mutex<Option<Cached>>,
        pool: Arc<dyn seismic::memory::Entry>,
        calls: AtomicU64,
    }
    impl CacheReclaimer for FakeBankReclaimer {
        fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut guard = self.cached.lock().unwrap();
            let Some(cached) = guard.as_mut() else {
                return Err(ReclaimError::AtFloor);
            };
            let has = cached.bytes();
            let free = has.min(need);
            if free == 0 {
                return Err(ReclaimError::AtFloor);
            }
            if free == has {
                drop(guard.take());
            } else {
                drop(cached.split(free, CacheKind::Bank));
            }
            self.pool.freed(&self.pool.change(), free, seismic::memory::Wiring::Pageable);
            Ok(free)
        }
    }

    let (_scope, ledger, pools) = host();
    let records_pool = pools.records.clone();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 8 * GB, Some(64 * GB));

    // Fill all 8 GB with Bank cache backed by the records pool and its reclaimer at creation.
    let reclaimer = Arc::new(FakeBankReclaimer {
        cached: Mutex::new(None),
        pool: records_pool.clone(),
        calls: AtomicU64::new(0),
    });
    let bank_cache = ledger
        .cached_with(CacheKind::Bank, 8 * GB, Reclaim::Backed(Arc::downgrade(&reclaimer) as Weak<dyn CacheReclaimer>))
        .expect("bank cache fits");
    *reclaimer.cached.lock().unwrap() = Some(bank_cache);
    assert_eq!(ledger.estimate().allocatable(), 0, "all 8 GB allocated in records pool");

    // 1. Non-pooled claim (`claim_at`) synchronously reclaims 2 GB before handing out Claim.
    let claim = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Component, 2 * GB))
        .expect("claim_at reclaims bank cache");
    assert!(reclaimer.calls.load(Ordering::Relaxed) >= 1, "reclaimer was invoked");
    assert!(ledger.estimate().allocatable() >= 2 * GB, "allocatable >= bytes before entry_take");
    claim.entry().take(&claim.entry().change(), 2 * GB, seismic::memory::Wiring::Pageable).expect("entry_take succeeds immediately");

    // 2. Pooled state backing (`backing(PoolClass::State)`) synchronously reclaims bank cache.
    let backing = ledger
        .change(|keeping| ledger.backing(&keeping.claims(), PoolClass::State, 1 * GB))
        .expect("backing(PoolClass::State) reclaims bank cache");
    assert!(ledger.estimate().allocatable() >= 1 * GB, "state pool block is physically allocatable");
    drop((claim, backing));
}

/// When a registered `CacheReclaimer` cannot free any more bytes (`AtFloor`),
/// `admit_or_reclaim` terminates and returns `Err(Short)` rather than looping.
#[test]
fn admit_or_reclaim_returns_short_when_reclaimer_hits_floor() {
    struct FloorReclaimer(AtomicU64);
    impl CacheReclaimer for FloorReclaimer {
        fn reclaim(&self, _need: u64) -> Result<u64, ReclaimError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Err(ReclaimError::AtFloor)
        }
    }

    let (_scope, ledger, _pools) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 4 * GB, Some(64 * GB));
    let reclaimer = Arc::new(FloorReclaimer(AtomicU64::new(0)));
    let _bank_cache = ledger
        .cached_with(CacheKind::Bank, 4 * GB, Reclaim::Backed(Arc::downgrade(&reclaimer) as Weak<dyn CacheReclaimer>))
        .expect("bank cache fits");

    // Asking for 5 GB exceeds total budget (4 GB) -> immediate Short without even calling reclaimer.
    let short = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Component, 5 * GB));
    assert!(short.is_err(), "shortfall past total room returns Err(Short)");
    assert_eq!(reclaimer.0.load(Ordering::Relaxed), 0);

    // Asking for 2 GB invokes the reclaimer once, sees AtFloor, and returns Err(Short).
    let short_at_floor = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Component, 2 * GB));
    assert!(short_at_floor.is_err(), "AtFloor returns Err(Short)");
    assert_eq!(reclaimer.0.load(Ordering::Relaxed), 1);
}

/// A `CacheReclaimer` invariant violation panics immediately rather than
/// logging a warning and continuing with an inconsistent ledger.
#[test]
#[should_panic(expected = "invariant violated: cache reclaim failed")]
fn admit_or_reclaim_panics_on_reclaimer_invariant_violation() {
    struct BrokenReclaimer;
    impl CacheReclaimer for BrokenReclaimer {
        fn reclaim(&self, _need: u64) -> Result<u64, ReclaimError> {
            Err(ReclaimError::Invariant("synthetic device shrink failure".to_string()))
        }
    }

    let (_scope, ledger, _pools) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 4 * GB, Some(64 * GB));
    let reclaimer = Arc::new(BrokenReclaimer);
    let _bank_cache = ledger
        .cached_with(CacheKind::Bank, 4 * GB, Reclaim::Backed(Arc::downgrade(&reclaimer) as Weak<dyn CacheReclaimer>))
        .expect("bank cache fits");

    let _ = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Component, 2 * GB));
}

/// Unreturned `Holding::Pool` spare bytes cannot mint a non-pooled `Claim`
/// (via `claim_at`, `claim_units`, or `entry_take`) unless a registered
/// `CacheReclaimer` physically returns the pool's blocks first.
#[test]
fn unreturned_pool_bytes_cannot_mint_claim_without_physical_return() {
    use seismic::memory::Store as _;

    struct FakePoolReclaimer {
        pool: Arc<dyn seismic::memory::Entry>,
        freed: AtomicU64,
    }
    impl CacheReclaimer for FakePoolReclaimer {
        fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
            self.pool.freed(&self.pool.change(), need, Pageable);
            self.freed.fetch_add(need, Ordering::Relaxed);
            Ok(need)
        }
    }

    let (_scope, ledger, pool) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 64 * GB, 8 * GB, Some(64 * GB));

    // Allocate 8 GB of spare granules into the records pool (no reclaimer yet).
    ledger.change(|_| pool.take(&pool.change(), 8 * GB, Pageable)).expect("pool block fits");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Spare), 8 * GB);
    assert_eq!(ledger.estimate().allocatable(), 0);

    // Without a live pool reclaimer, non-pooled claims and entry_take cannot
    // take unreturned pool bytes.
    let short = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::GroupClass, 2 * GB));
    assert!(short.is_err(), "claim_at must refuse unreturned pool bytes without a reclaimer");
    let (_, units) = ledger.change(|keeping| {
        ledger.claim_units(&keeping.claims(), CommittedKind::GroupClass, NonZeroU64::new(2 * GB).unwrap(), 1)
    });
    assert_eq!(units, 0, "claim_units must yield 0 units from unreturned pool bytes");
    let own = ledger.own();
    assert!(
        own.take(&own.change(), 2 * GB, Pageable).is_err(),
        "entry_take must refuse unreturned pool bytes"
    );

    // Once a live pool reclaimer physically frees blocks, claims succeed.
    let reclaimer = Arc::new(FakePoolReclaimer { pool: pool.clone(), freed: AtomicU64::new(0) });
    ledger.register_pool_reclaimer(PoolClass::Records, reclaimer.clone());
    let claim = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::GroupClass, 2 * GB))
        .expect("reclaimer physically freed 2 GB");
    assert_eq!(reclaimer.freed.load(Ordering::Relaxed), 2 * GB);
    assert_eq!(ledger.estimate().allocatable(), 2 * GB);
    drop(claim);
}

/// A non-pooled committed entry (`CommittedKind::Device`, e.g. `self.own`)
/// asking for bytes when only unyieldable `Holding::Pool` spare bytes remain
/// (partial free room + pool spare >= request) must be refused cleanly rather
/// than taking a partial entry and panicking on `allocated > entry.bytes`.
#[test]
fn non_pooled_entry_take_refuses_when_only_pool_spare_remains() {
    let (_scope, ledger, pools) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 4 * GB, 4 * GB, None);
    // Allocate 3 GB of spare granules in the records pool, leaving 1 GB free.
    ledger.change(|_| pools.records.take(&pools.records.change(), 3 * GB, Pageable)).expect("pool spare fits");
    // A non-pooled Device entry asking for 2 GB cannot take the records pool's
    // spare bytes: it must return Err(Refused), never panic.
    let own = seismic::memory::Store::own(&ledger);
    let refused = ledger
        .change(|_| own.take(&own.change(), 2 * GB, Pageable))
        .expect_err("non-pooled entry_take must refuse when only pool spare covers the shortfall");
    assert_eq!(refused, seismic::memory::Refused { required: 2 * GB, available: 1 * GB });
    // A pool entry refusing when free room < lacks must also report available < required,
    // never adding unrelated cache bytes onto `available`.
    let pool_refused = ledger
        .change(|_| pools.state.take(&pools.state.change(), 2 * GB, Pageable))
        .expect_err("pool entry_take must refuse when free room < lacks");
    assert_eq!(pool_refused, seismic::memory::Refused { required: 2 * GB, available: 1 * GB });
    ledger.change(|_| pools.records.freed(&pools.records.change(), 3 * GB, Pageable));
}

#[test]
fn relief_skips_at_floor_pool_and_reclaims_bank_synchronously() {
    let (_scope, ledger, pools) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 4 * GB, 4 * GB, None);

    // State pool has 1 GB of unfreeable spare bytes (reclaimer returns AtFloor).
    ledger.change(|_| pools.state.take(&pools.state.change(), 1 * GB, Pageable)).expect("state pool fits");
    struct FloorPool;
    impl CacheReclaimer for FloorPool {
        fn reclaim(&self, _need: u64) -> Result<u64, ReclaimError> {
            Err(ReclaimError::AtFloor)
        }
    }
    ledger.register_pool_reclaimer(PoolClass::State, Arc::new(FloorPool));

    // Bank holds 3 GB of reclaimable cache (worth > 0 so it ranks after Pool spare).
    struct BankReclaim {
        ledger: Ledger,
        pool: Arc<dyn seismic::memory::Entry>,
        piece: std::sync::Mutex<Option<Cached>>,
    }
    impl CacheReclaimer for BankReclaim {
        fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
            let mut guard = self.piece.lock().unwrap();
            let Some(cached) = guard.take() else {
                return Err(ReclaimError::AtFloor);
            };
            let bytes = cached.bytes();
            let cut = bytes.min(need);
            self.ledger.change(|_| {
                drop(cached);
                self.pool.freed(&self.pool.change(), bytes, Pageable);
            });
            Ok(cut)
        }
    }
    let bank_claim = ledger
        .change(|keeping| ledger.backing(&keeping.claims(), PoolClass::Records, 3 * GB))
        .expect("bank backing fits");
    ledger.change(|_| pools.records.take(&pools.records.change(), 3 * GB, Pageable)).expect("records pool allocates");
    let reclaimer = Arc::new(BankReclaim {
        ledger: ledger.clone(),
        pool: pools.records.clone(),
        piece: std::sync::Mutex::new(None),
    });
    let weak = Arc::downgrade(&reclaimer) as std::sync::Weak<dyn CacheReclaimer>;
    let bank_cache = ledger.change(|_| bank_claim.into_backed_cache(CacheKind::Bank, Reclaim::Backed(weak)));
    bank_cache.value(Value::per_byte(1.0, NonZeroU64::new(GB).unwrap()));
    *reclaimer.piece.lock().unwrap() = Some(bank_cache);

    // First relief pass picks Pool(State) spare (worth 0) -> AtFloor -> recorded in `exhausted`.
    let mut exhausted = std::collections::BTreeSet::new();
    let plan1 = ledger
        .plan_relief_excluding(1 * GB, Relief::CacheOnly, Frees::Room, &exhausted)
        .expect("first plan covers 1 GB from state pool spare");
    let owed1 = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan1));
    assert_eq!(owed1.reclaimed, 0);
    assert_eq!(owed1.yielded, 0);
    assert!(!owed1.exhausted.is_empty());
    exhausted.extend(owed1.exhausted);

    // Second relief pass excludes exhausted Pool(State) and synchronously reclaims Bank.
    let plan2 = ledger
        .plan_relief_excluding(1 * GB, Relief::CacheOnly, Frees::Room, &exhausted)
        .expect("second plan covers 1 GB from bank cache");
    let owed2 = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan2));
    assert_eq!(owed2.reclaimed, 1 * GB);
    assert!(owed2.exhausted.is_empty());

    ledger.change(|_| pools.state.freed(&pools.state.change(), 1 * GB, Pageable));
}

/// `Ledger::grant(GrantRole::Measurement, ...)` synchronously reclaims backed
/// `CacheKind::Bank` cache before admitting the grant, leaves `Reclaim::Deferred`
/// cache untouched, and refuses shortfalls (including a `PAST` room probe)
/// without spuriously draining the bank.
#[test]
fn measurement_grant_reclaims_bank_cache_and_refuses_shortfall_without_draining() {
    use std::sync::{Mutex, Weak};

    struct FakeBankReclaimer {
        cached: Mutex<Option<Cached>>,
        pool: Arc<dyn seismic::memory::Entry>,
        calls: AtomicU64,
    }
    impl CacheReclaimer for FakeBankReclaimer {
        fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut guard = self.cached.lock().unwrap();
            let Some(cached) = guard.as_mut() else {
                return Err(ReclaimError::AtFloor);
            };
            let has = cached.bytes();
            let free = has.min(need);
            if free == 0 {
                return Err(ReclaimError::AtFloor);
            }
            if free == has {
                drop(guard.take());
            } else {
                drop(cached.split(free, CacheKind::Bank));
            }
            self.pool.freed(&self.pool.change(), free, Pageable);
            Ok(free)
        }
    }

    let (_scope, ledger, pools) = host();
    let mut clock = Clock::new();
    // 12 GB available: 2 GB committed weights, 2 GB deferred Prefix cache,
    // 8 GB backed Bank cache (0 free, 0 allocatable).
    read(&ledger, &mut clock, 64 * GB, 12 * GB, Some(64 * GB));
    let _weights = commit(&ledger, 2 * GB);
    let _deferred_prefix = ledger.cached_with(CacheKind::Prefix, 2 * GB, Reclaim::Deferred).expect("prefix fits");

    let reclaimer = Arc::new(FakeBankReclaimer {
        cached: Mutex::new(None),
        pool: pools.records.clone(),
        calls: AtomicU64::new(0),
    });
    let bank_cache = ledger
        .cached_with(CacheKind::Bank, 8 * GB, Reclaim::Backed(Arc::downgrade(&reclaimer) as Weak<dyn CacheReclaimer>))
        .expect("bank cache fits");
    *reclaimer.cached.lock().unwrap() = Some(bank_cache);
    assert_eq!(ledger.estimate().allocatable(), 0, "all 12 GB allocated");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Bank), 8 * GB);

    // 1. `PAST` room probe exceeds reclaimable room (8 GB) -> Err(Short) without
    //    invoking the reclaimer or draining the bank; `PAST - over == 8 GB`.
    const PAST: u64 = u64::MAX / 4;
    let probe = ledger
        .change(|keeping| ledger.grant(&keeping.claims(), GrantRole::Measurement, PAST))
        .err()
        .expect("PAST probe must be refused");
    assert_eq!(PAST - probe.over, 8 * GB, "room equals reclaimable bank cache (deferred prefix excluded)");
    assert_eq!(reclaimer.calls.load(Ordering::Relaxed), 0, "PAST probe must not drain the bank");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Bank), 8 * GB);

    // 2. Requesting 9 GB (exceeds 8 GB reclaimable bank; deferred prefix is not
    //    reclaimable without freeing backing) -> Err(Short { over: 1 GB }), 0 calls.
    let short = ledger
        .change(|keeping| ledger.grant(&keeping.claims(), GrantRole::Measurement, 9 * GB))
        .err()
        .expect("exceeds reclaimable room");
    assert_eq!(short.over, 1 * GB);
    assert_eq!(reclaimer.calls.load(Ordering::Relaxed), 0, "shortfall must not drain the bank");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Prefix), 2 * GB, "deferred prefix untouched");

    // 3. Requesting 5 GB synchronously reclaims 5 GB of Bank cache before admitting.
    let grant = ledger
        .change(|keeping| ledger.grant(&keeping.claims(), GrantRole::Measurement, 5 * GB))
        .expect("measurement grant reclaims bank cache");
    assert!(reclaimer.calls.load(Ordering::Relaxed) >= 1, "reclaimer was invoked");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Bank), 3 * GB, "bank shrunk by 5 GB");
    assert_eq!(ledger.estimate().cache_of(CacheKind::Prefix), 2 * GB, "deferred prefix untouched");
    assert_eq!(ledger.estimate().committed(), 7 * GB, "2 GB weights + 5 GB measurement grant");
    assert!(ledger.estimate().allocatable() >= 5 * GB, "5 GB physically freed for measurement process");
    drop(grant);
}

/// A `CommittedKind::Measurement` claim synchronously reclaims backed bank
/// cache when `allocatable` is tight, keeps its peak reservation across
/// sequential measurement points (`entry_freed` does not shrink it), and
/// releases the whole reservation when dropped.
#[test]
fn measurement_claim_reclaims_bank_and_reuses_peak_reservation_across_points() {
    use std::sync::atomic::AtomicU64;
    use std::sync::{Mutex, Weak};

    const MIB: u64 = 1024 * 1024;
    const PROBE: u64 = 32 * MIB;
    const PEAK: u64 = 3 * PROBE;

    struct FakeBankReclaimer {
        pool: Arc<dyn seismic::memory::Entry>,
        cached: Mutex<Option<Cached>>,
        reclaimed: AtomicU64,
    }
    impl CacheReclaimer for FakeBankReclaimer {
        fn reclaim(&self, need: u64) -> Result<u64, ReclaimError> {
            let mut guard = self.cached.lock().unwrap();
            let Some(cached) = guard.as_mut() else {
                return Err(ReclaimError::AtFloor);
            };
            let current = cached.bytes();
            let drop_bytes = current.min(need);
            if drop_bytes == 0 {
                return Err(ReclaimError::AtFloor);
            }
            if drop_bytes == current {
                drop(guard.take());
            } else {
                drop(cached.split(drop_bytes, CacheKind::Bank));
            }
            self.pool.freed(&self.pool.change(), drop_bytes, Pageable);
            self.reclaimed.fetch_add(drop_bytes, Ordering::Relaxed);
            Ok(drop_bytes)
        }
    }

    let (_scope, ledger, pools) = host();
    let mut clock = Clock::new();
    read(&ledger, &mut clock, 12 * GB, 12 * GB, None);

    // Fill the entire 12 GB budget: 4 GB committed + 8 GB backed bank cache.
    let _weights = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, 4 * GB))
        .expect("weights claim fits");
    ledger
        .change(|_| _weights.entry().take(&_weights.entry().change(), 4 * GB, Pageable))
        .expect("weights allocate");
    let reclaimer = Arc::new(FakeBankReclaimer {
        pool: pools.records.clone(),
        cached: Mutex::new(None),
        reclaimed: AtomicU64::new(0),
    });
    let bank_cache = ledger
        .cached_with(CacheKind::Bank, 8 * GB, Reclaim::Backed(Arc::downgrade(&reclaimer) as Weak<dyn CacheReclaimer>))
        .expect("bank cache fits");
    *reclaimer.cached.lock().unwrap() = Some(bank_cache);
    assert_eq!(ledger.estimate().allocatable(), 0, "VRAM completely full before measurement claim");

    // Pre-claim PEAK (96 MiB) for an in-process measurement: reclaims 96 MiB from Bank.
    let measurement = ledger
        .change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Measurement, PEAK))
        .expect("measurement claim synchronously reclaims bank cache");
    assert_eq!(reclaimer.reclaimed.load(Ordering::Relaxed), PEAK);
    assert_eq!(measurement.bytes(), PEAK);
    assert!(ledger.estimate().allocatable() >= PEAK);

    let entry = measurement.entry();
    // Point 1 & 2 (`tier_copy` blocks=1 and 512): allocate 32 MiB then free;
    // peak reservation must NOT shrink on `freed`.
    for _ in 0..2 {
        ledger
            .change(|_| entry.take(&entry.change(), PROBE, Pageable))
            .expect("tier_copy point fits in peak reservation");
        ledger.change(|_| entry.freed(&entry.change(), PROBE, Pageable));
        assert_eq!(measurement.bytes(), PEAK, "measurement peak reservation survives point drop");
        assert_eq!(measurement.allocated_bytes(), 0);
    }

    // Point 3 (stall probe beside `into`): 3 x 32 MiB live simultaneously.
    for _ in 0..3 {
        ledger
            .change(|_| entry.take(&entry.change(), PROBE, Pageable))
            .expect("stall probe buffers fit in peak reservation");
    }
    assert_eq!(measurement.allocated_bytes(), PEAK);
    ledger.change(|_| entry.freed(&entry.change(), PEAK, Pageable));
    assert_eq!(measurement.bytes(), PEAK);

    // Dropping the measurement claim releases the entire peak reservation.
    drop(entry);
    drop(measurement);
    assert_eq!(ledger.estimate().committed(), 4 * GB, "measurement peak reservation released whole");
}
