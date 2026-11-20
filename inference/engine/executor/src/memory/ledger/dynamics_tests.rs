//! The ledger's dynamics against a scripted OS (design §8, 11-11 §8): a
//! store whose figures show the process's release after its λ, whose free
//! memory sits at the OS's target and which displaces clean cache and then
//! compresses others' cold pages under the process's growth, with harm
//! counters, the swap of the process's own pages and the lapse of its
//! wiring the script drives. A bank follows the ledger's target at every
//! reading by the domain's own rule (`experts::derived::held_target`), a
//! granule at a time.

use super::*;
use seismic::memory::Wiring;
use std::sync::Mutex as Lock;
use std::time::{Duration, Instant};

const GB: u64 = 1 << 30;
const MB: u64 = 1 << 20;
const PAGE: u64 = 16 * 1024;
/// The bank's granule: its only threshold.
const GRANULE: u64 = 256 * MB;

/// A release shows 50 ms after it; growth at once.
const LAG: seismic::Lag = seismic::Lag { release: Duration::from_millis(50), growth: Duration::ZERO };

/// The counters' snapshot (cold, wire count, harm) is served again until
/// it is this old, as `host_statistics64` serves one: a reading within it
/// shows the machine as the snapshot did.
const SNAPSHOT: Duration = Duration::from_millis(200);

/// One snapshot of the counters: when it was taken, others' cold pages,
/// others' wired bytes, the process's wired bytes the OS had let lapse,
/// and the machine's and the process's own pages brought back. The
/// process's own wired bytes in its wire count are those the snapshot's
/// instant shows ([`Os::wired_at`]), less those lapsed.
#[derive(Clone, Copy)]
struct Snapshot {
    at: Instant,
    cold: u64,
    others_wired: Option<u64>,
    lapsed: u64,
    machine: u64,
    own: u64,
}

/// The scripted OS: one store of `total` bytes. Where it has a wired
/// ceiling it is unified Metal's macOS, which counts none of the process's
/// cold pages; otherwise a Linux host whose cgroup is the process's alone,
/// which counts them (none of the process's pages is cold in the script).
struct Os {
    total: u64,
    /// Free pages the OS keeps free by reclaiming, of which its hard
    /// reserve ([`Self::hard_reserve`]) no allocation takes.
    target: u64,
    /// Other programs' working set, which they touch.
    hot: u64,
    /// Other programs' cold anonymous pages still resident, and what the
    /// compressor stores of those it compressed (half).
    cold: u64,
    compressed: u64,
    /// Clean file cache.
    cache: u64,
    /// The process's memory as the OS shows it, and its releases not shown
    /// yet (shown at the instant given).
    ours: u64,
    releasing: Vec<(Instant, u64)>,
    /// Of `ours`, pageable bytes the OS swapped out (others' burst came and
    /// went): in no figure of what the store uses.
    swapped: u64,
    /// Whether the OS reports the process's bytes out of RAM
    /// (`seismic::HostSample::own_reclaimed`), whether its count is exact,
    /// and of `swapped` the bytes it leaves out of the count.
    reports_reclaimed: bool,
    exact: bool,
    unreported: u64,
    /// Of `swapped`, the RAM the OS's compressed store holds of them, in
    /// what the machine uses, and whether the OS reports it.
    own_stored: u64,
    reports_stored: bool,
    /// Of `ours`, wired bytes the OS let lapse from its wire count (a
    /// residency set not yet wired again).
    lapsed: u64,
    /// The process's growth the supply does not show yet, each until the
    /// instant given: until then the supply still counts it as the clean
    /// cache it displaced (a cgroup's `memory.stat`, read within memcg's
    /// statistics flush).
    growing: Vec<(Instant, u64)>,
    /// The moves of what the process allocated wired
    /// (`seismic::memory::Wiring::Wired`), each from the instant the wire
    /// count shows it: the OS wires only those bytes of the process's, and
    /// keeps them wired while the process holds them.
    wiring: Vec<(Instant, i128)>,
    /// Cumulative pages others brought back, and the process's own.
    harm: u64,
    own_paging: u64,
    /// Whether the OS counts the process's share of the machine's harm:
    /// none where it does not (a cgroup shared with other processes), so no
    /// harm is read at all.
    counts_own: bool,
    pressure: seismic::PressureLevel,
    /// Whether the OS reports a pressure level.
    reports_pressure: bool,
    /// GPU wired figures: the ceiling, and others' wired bytes.
    wired: Option<(u64, u64)>,
    lag: seismic::Lag,
    /// The counters' snapshot served now ([`SNAPSHOT`]).
    served: Option<Snapshot>,
}

impl Os {
    /// 64 GB: others' 24 GB working set, 8 GB of cold pages, 4 GB of clean
    /// cache, 1 GB kept free. Beside a load's 20 GB the supply is 15 GB: 7
    /// free above the target, 4 of cache and 4 the compressor frees of the
    /// cold pages.
    fn new() -> Self {
        Self {
            total: 64 * GB,
            target: GB,
            hot: 24 * GB,
            cold: 8 * GB,
            compressed: 0,
            cache: 4 * GB,
            ours: 0,
            releasing: Vec::new(),
            swapped: 0,
            reports_reclaimed: true,
            exact: true,
            unreported: 0,
            own_stored: 0,
            reports_stored: true,
            lapsed: 0,
            growing: Vec::new(),
            wiring: Vec::new(),
            harm: 0,
            own_paging: 0,
            counts_own: true,
            pressure: seismic::PressureLevel::Normal,
            reports_pressure: true,
            wired: None,
            lag: LAG,
            served: None,
        }
    }

    fn used(&self) -> u64 {
        self.ours - self.swapped + self.own_stored + self.releasing.iter().map(|(_, bytes)| bytes).sum::<u64>() + self.hot + self.cold + self.compressed + self.cache
    }

    /// The free pages within the target no allocation takes: Linux's min
    /// watermark, two thirds of the high one kswapd reclaims to, as macOS's
    /// `vm_page_free_min` is within its `vm_page_free_target`.
    fn hard_reserve(&self) -> u64 {
        self.target * 2 / 3
    }

    /// Keep free at the target: drop clean cache, then compress cold pages
    /// (a compressed page stores half of itself); what neither gives comes
    /// out of the free pages below the target down to the hard reserve, the
    /// soft reserve a growth within one reading dips into. Out of memory
    /// past it.
    fn reclaim(&mut self) {
        let mut need = (self.used() + self.target).saturating_sub(self.total);
        let dropped = need.min(self.cache);
        self.cache -= dropped;
        need -= dropped;
        let compressed = (2 * need).min(self.cold);
        self.cold -= compressed;
        self.compressed += compressed / 2;
        assert!(self.used() + self.hard_reserve() <= self.total, "the scripted OS ran out of memory: an allocation took its hard reserve");
    }

    /// The process allocated `bytes` of `wiring` at `at`: shown after the
    /// growth lag, in the supply and, wired, in the wire count.
    fn grow(&mut self, at: Instant, bytes: u64, wiring: Wiring) {
        let shown = at + self.lag.growth;
        self.ours += bytes;
        self.growing.push((shown, bytes));
        if wiring == Wiring::Wired {
            self.wiring.push((shown, i128::from(bytes)));
        }
        self.reclaim();
    }

    /// The process freed `bytes` of `wiring` at `at`: shown after the
    /// release lag.
    fn release(&mut self, at: Instant, bytes: u64, wiring: Wiring) {
        let shown = at + self.lag.release;
        self.ours -= bytes;
        self.releasing.push((shown, bytes));
        if wiring == Wiring::Wired {
            self.wiring.push((shown, -i128::from(bytes)));
        }
    }

    /// The process's wired bytes the wire count shows at `at`.
    fn wired_at(&self, at: Instant) -> u64 {
        let wired = self.wiring.iter().filter(|(shown, _)| *shown <= at).map(|(_, moved)| moved).sum::<i128>();
        u64::try_from(wired).expect("the process never frees more wired bytes than it allocated")
    }

    /// Host RAM at `at`: the live figures now, the counters from the
    /// snapshot served, taken again once it is [`SNAPSHOT`] old.
    fn sample(&mut self, at: Instant) -> seismic::HostSample {
        self.releasing.retain(|(shown, _)| *shown > at);
        self.growing.retain(|(shown, _)| *shown > at);
        self.reclaim();
        let free = self.total - self.used();
        let growing = self.growing.iter().map(|(_, bytes)| bytes).sum::<u64>();
        let snapshot = match self.served {
            Some(served) if at < served.at + SNAPSHOT => served,
            _ => Snapshot {
                at,
                cold: self.cold,
                others_wired: self.wired.map(|(_, others)| others),
                lapsed: self.lapsed,
                machine: self.harm + self.own_paging,
                own: self.own_paging,
            },
        };
        self.served = Some(snapshot);
        let wire_count = snapshot.others_wired.map(|others| others + self.wired_at(snapshot.at) - snapshot.lapsed);
        seismic::HostSample {
            at,
            page_bytes: PAGE,
            total: self.total,
            supply: seismic::Supply {
                free_beyond_target: i64::try_from(free).expect("bytes fit i64") - i64::try_from(self.target).expect("bytes fit i64"),
                clean_cache: self.cache + growing,
                cold_reclaimable: self.cold / 2,
            },
            counted: seismic::Counted { at: snapshot.at, lag: Duration::ZERO },
            cold: seismic::ColdPages {
                machine: snapshot.cold,
                own: match self.wired {
                    Some(_) => None,
                    None => Some(0),
                },
            },
            own_reclaimed: self.reports_reclaimed.then_some(seismic::OwnReclaimed {
                bytes: self.swapped - self.unreported,
                stored: if self.reports_stored { self.own_stored } else { 0 },
                exact: self.exact,
            }),
            wired: self.wired.map(|(limit, _)| seismic::WiredFigures {
                limit: seismic::WiredLimit::Set(limit),
                wire_count: wire_count.expect("a wired store's snapshot counts its wire count"),
            }),
            harm: self.counts_own.then_some(seismic::Harm { machine: snapshot.machine, own: snapshot.own }),
            pressure: self.reports_pressure.then_some(self.pressure),
            lag: self.lag,
        }
    }
}

/// A scripted machine: the OS, the scripted clock, one store's ledger, a
/// load's weights and in-process figure, and the bank that follows the
/// target.
struct Machine {
    os: Os,
    now: Arc<Lock<Instant>>,
    _scope: Ledgers,
    ledger: Ledger,
    /// The store's records pool's entry: the bank's blocks are charged to
    /// it as they are allocated and freed from it as they are returned.
    pool: Arc<dyn seismic::memory::Entry>,
    /// The store's state pool's entry: the sequence's pages are charged to
    /// it as decode grows them.
    state: Arc<dyn seismic::memory::Entry>,
    /// How the OS holds the load's memory: wired on unified Metal, where
    /// the weights and every pool block lie in the residency set; pageable
    /// host memory otherwise.
    wiring: Wiring,
    _load: Claim,
    /// The load's request slot, which its first request draws its
    /// reservation from.
    slot: Claim,
    /// The load's one request and the pages it wrote, where it serves: its
    /// reservation, drawn from the slot, which decode writes a page at a
    /// time.
    serving: Option<(Reservation, u64)>,
    bank: Vec<Cached>,
    /// Readings at which the bank resized.
    resizes: u32,
}

impl Machine {
    /// A load of `weights` bytes and a request slot of `request` bytes in
    /// `os` ([`Self::start`]), serving one request where the slot holds
    /// any.
    fn load(os: Os, weights: u64, request: u64) -> Self {
        Self::start(os, weights, request, request > 0)
    }

    /// A load of `weights` bytes and a request slot of `request` bytes in
    /// `os`, read once before and once after it, one request reserving the
    /// slot's whole pages where `serving`.
    fn start(mut os: Os, weights: u64, request: u64, serving: bool) -> Self {
        let now = Arc::new(Lock::new(Instant::now()));
        let clock = now.clone();
        let scope = Ledgers::scripted(move || *clock.lock().expect("the clock"));
        let ledger = scope.ledger(seismic::StoreIdentity::HostRam, super::tests::shape());
        let pool = ledger.pool_block(PoolClass::Records, std::num::NonZeroU64::new(GRANULE).unwrap()).blocks;
        let state = ledger.pool_block(PoolClass::State, std::num::NonZeroU64::new(GRANULE).unwrap()).blocks;
        let wiring = match os.wired {
            Some(_) => Wiring::Wired,
            None => Wiring::Pageable,
        };
        let at = *now.lock().expect("the clock");
        read(&ledger, &mut os, at);
        let claim = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, weights)).expect("the weights fit");
        ledger.change(|_| { let entry = claim.entry(); entry.take(&entry.change(), weights, wiring) }).expect("the weights' own bytes");
        os.grow(at, weights, wiring);
        let slot = ledger.change(|keeping| ledger.claim(&keeping.claims(), Need::RequestSlot { bytes: request })).expect("the slot fits");
        let serving = serving.then(|| {
            let rows = request / page_bytes(&ledger) * super::tests::page_rows();
            (ledger.change(|keeping| ledger.reserve(&keeping.claims(), &slot, RequestId(1), rows)).expect("the request draws the slot"), 0)
        });
        if let Some((request, _)) = &serving {
            // Its slabs are allocated as it opens: the OS holds its whole
            // state from then (`ExecutorDomain::back_reservation`).
            let bytes = request.bytes();
            ledger.change(|_| state.take(&state.change(), bytes, wiring)).expect("the reservation holds the slabs' granules");
            os.grow(at, bytes, wiring);
        }
        let mut machine = Self { os, now, _scope: scope, ledger, pool, state, wiring, _load: claim, slot, serving, bank: Vec::new(), resizes: 0 };
        machine.tick(100);
        machine.read();
        machine
    }

    fn at(&self) -> Instant {
        *self.now.lock().expect("the clock")
    }

    /// `ms` pass.
    fn tick(&mut self, ms: u64) {
        *self.now.lock().expect("the clock") += Duration::from_millis(ms);
    }

    /// One reading.
    fn read(&mut self) {
        let at = self.at();
        read(&self.ledger, &mut self.os, at);
    }

    fn bank_bytes(&self) -> u64 {
        self.bank.len() as u64 * GRANULE
    }

    /// The bank follows the target by the domain's rule, a granule at a
    /// time: it grows once the target holds a granule more, and shrinks once
    /// it holds a granule more than the target, to what the target fits.
    fn follow(&mut self) {
        let shape = crate::experts::derived::BankShape::new(GRANULE, 0, u64::MAX / GRANULE, u64::MAX / GRANULE);
        let held = self.bank.len() as u64;
        let est = self.ledger.estimate();
        let slots = crate::experts::derived::held_target_in(est.hold_room(), est.target(), held, 0, &shape, &|slots| slots * GRANULE);
        let at = self.at();
        if slots > held {
            for _ in held..slots {
                let granule = self.ledger.cache(CacheKind::Bank, GRANULE, super::Reclaim::Unbacked).expect("the target is free room");
                self.ledger.change(|_| self.pool.take(&self.pool.change(), GRANULE, self.wiring)).expect("the pool's block for the granule is admitted");
                self.os.grow(at, GRANULE, self.wiring);
                self.bank.push(granule);
            }
            self.resizes += 1;
        } else if slots < held {
            while self.bank.len() as u64 > slots {
                let granule = self.bank.pop().expect("a granule");
                self.ledger.change(|_| {
                    drop(granule);
                    self.pool.freed(&self.pool.change(), GRANULE, self.wiring);
                });
                self.os.release(at, GRANULE, self.wiring);
            }
            self.resizes += 1;
        }
    }

    /// The decode's next `pages` pages of the request's state, written
    /// into slabs its open allocated: nothing allocates.
    fn decode(&mut self, pages: u64) {
        let (request, written) = self.serving.as_mut().expect("the load serves a request");
        *written += pages;
        request.grow(*written * super::tests::page_rows());
    }

    /// A reading `ms` after the last, then the bank following it.
    fn step(&mut self, ms: u64) {
        self.tick(ms);
        self.read();
        self.follow();
    }
}

/// The bytes of one page of a request's state.
fn page_bytes(ledger: &Ledger) -> u64 {
    ledger.bytes(&Need::Request { sequence: RequestId(1), rows: super::tests::page_rows() })
}

/// `ledger` reads `os` at `at`, as the store's kind reads it.
fn read(ledger: &Ledger, os: &mut Os, at: Instant) {
    let sample = os.sample(at);
    let reading = match os.wired {
        Some(_) => Reading::unified(&sample, u64::MAX),
        None => Reading::host(&sample),
    };
    ledger.read(ledger.sampled(at), reading);
}

/// One reading a counter snapshot later: its counters are new.
const SNAPSHOT_MS: u64 = SNAPSHOT.as_millis() as u64;

/// Load 20 GB, and let the bank fill: the load's step closes, a baseline
/// is learned, and the bank grows, at the reading that returns: the next
/// reading a snapshot later judges the bank's growth step.
fn filled(os: Os, request: u64) -> Machine {
    let mut machine = Machine::load(os, 20 * GB, request);
    for _ in 0..20 {
        if machine.bank_bytes() > 0 {
            break;
        }
        machine.step(SNAPSHOT_MS);
    }
    assert!(machine.bank_bytes() > 0, "the bank fills");
    machine
}

/// [`filled`], settled cleanly after the bank claims the full 15 GB supply
/// (7 GB free beyond target, 4 GB clean cache, 4 GB cold reclaimable):
/// cold pages count from reading #1 so the bank fills whole once the load
/// step closes, and subsequent readings resize nothing.
fn settled(os: Os, request: u64) -> Machine {
    let mut machine = filled(os, request);
    let resizes = machine.resizes;
    machine.step(SNAPSHOT_MS);
    assert_eq!(machine.resizes, resizes, "settled: the next reading resizes nothing");
    machine.step(SNAPSHOT_MS);
    assert_eq!(machine.resizes, resizes, "settled: closing the growth step resizes nothing");
    assert_eq!(machine.ledger.standing(), None, "settled: nothing stands");
    machine
}

/// Steady decode makes no resize: once the bank has settled, the request
/// writes its state a page a reading, within the reservation it holds from
/// its open, and others take a third of a granule at every other reading
/// and give it back, which moves the target by less than a block.
#[test]
fn steady_decode_makes_zero_resizes() {
    let mut machine = settled(Os::new(), GB);
    let settled = machine.resizes;
    let (_, first) = *machine.serving.as_ref().expect("the load serves a request");
    for reading in 0..200 {
        machine.decode(1);
        match reading % 2 {
            0 => machine.os.hot += GRANULE / 3,
            _ => machine.os.hot -= GRANULE / 3,
        }
        machine.step(100);
    }
    let page = page_bytes(&machine.ledger);
    let (request, pages) = machine.serving.as_ref().expect("the load serves a request");
    assert!(*pages - first == 200 && *pages * page <= request.bytes(), "the decode's pages, within the reservation");
    assert_eq!(machine.resizes, settled, "steady decode resizes nothing");
}

/// A request reserved at its open writes every row of its context while
/// others rise inside the guarantee at every reading: its slabs were
/// allocated as it opened, so nothing it writes allocates or waits on
/// memory, no relief preempts it, and nothing unloads.
#[test]
fn a_context_length_request_completes_beside_rising_others() {
    let mut machine = settled(Os::new(), GB);
    let pages = machine.serving.as_ref().expect("the load serves a request").0.rows() / super::tests::page_rows();
    let per_reading = pages.div_ceil(40);
    while machine.serving.as_ref().expect("the load serves a request").1 < pages {
        let written = machine.serving.as_ref().expect("the load serves a request").1;
        machine.decode(per_reading.min(pages - written));
        machine.os.hot += GRANULE / 16;
        machine.step(100);
        if let Some(shortage) = machine.ledger.shortage(&std::collections::BTreeSet::from([RequestId(1)])) {
            assert!(!shortage.response.unload, "others' rise inside the guarantee unloads nothing");
            assert!(shortage.response.relief.is_none_or(|plan| plan.preempt().is_empty()), "the request is never preempted");
        }
    }
    let (request, written) = machine.serving.as_ref().expect("the load serves a request");
    assert_eq!(*written * super::tests::page_rows(), request.rows(), "every row of its context written");
}

/// A shrink never raises others' memory: a release the store does not show
/// yet is the process's until it shows, and it raises no guarantee.
#[test]
fn a_shrink_never_raises_others() {
    let mut machine = settled(Os::new(), 0);
    let before = machine.ledger.estimate();
    let at = machine.at();
    for _ in 0..8 {
        let granule = machine.bank.pop().expect("a granule");
        machine.ledger.change(|_| {
            drop(granule);
            machine.pool.freed(&machine.pool.change(), GRANULE, machine.wiring);
        });
        machine.os.release(at, GRANULE, machine.wiring);
    }
    // Within the release lag, and after it.
    for ms in [10, 20, 30] {
        machine.tick(ms);
        machine.read();
        let estimate = machine.ledger.estimate();
        assert_eq!((estimate.others(), estimate.guarantee(), estimate.budget()), (before.others(), 0, before.budget()));
    }
}

/// Growth into a compressing OS stops at the supply: the clean cache it
/// drops and the cold pages it compresses leave the supply as it takes
/// them, so the room never rises with the process's own growth.
#[test]
fn growth_into_a_compressing_os_stops_at_supply() {
    let mut machine = settled(Os::new(), 0);
    // The supply beside the load ([`Os::new`]): 7 GB free above the
    // target, 4 GB of cache and 4 GB the compressor frees of the cold pages.
    let supply = 15 * GB;
    let held = machine.bank_bytes();
    assert!(held <= supply && supply - held < GRANULE, "the bank takes the supply, {held} of {supply}, and no more");
    assert!(machine.ledger.estimate().supply() < GRANULE as i64, "what it took left the supply");
    assert!(machine.os.compressed > 0, "the OS compressed others' cold pages under it");
    let room = machine.ledger.estimate().budget();
    for _ in 0..20 {
        machine.step(300);
    }
    assert_eq!(machine.bank_bytes(), held, "displacement never reads as room");
    assert_eq!(machine.ledger.estimate().budget(), room);
}

/// Sustained active thrashing under scarcity sets B cleanly and sheds cache
/// smoothly, without over-reacting to transient or sub-threshold noise.
#[test]
fn harm_on_a_step_sets_the_boundary_and_settles() {
    let mut machine = filled(Os::new(), 0);
    assert_eq!(machine.bank_bytes(), 15 * GB);
    assert!(machine.ledger.estimate().cap().is_none());

    // Simulate sustained 256 MB/s thrashing over two 1s counter spans while scarce (free == 0).
    let thrash_step_pages = (256 * MB) / PAGE;
    for _ in 0..2 {
        machine.os.harm += thrash_step_pages;
        machine.tick(1000);
        machine.read();
        machine.follow();
    }
    let cap = machine.ledger.estimate().cap().expect("sustained thrashing sets B");
    assert!(cap >= 20 * GB && machine.bank_bytes() < 15 * GB, "sustained thrashing caps ({cap}) and sheds cache ({}) smoothly", machine.bank_bytes());
}

/// Regrowth past B needs an exogenous release: the process's own shrink
/// never admits it; others quitting does.
#[test]
fn regrowth_needs_exogenous_evidence() {
    let mut machine = filled(Os::new(), 0);
    let step = machine.bank_bytes();
    // Sustained 350 MB/s thrashing under scarcity over > 1s confirms contention and sheds >= 1 granule.
    let pages_per_step = (70 * MB) / PAGE; // 350 MB/s at 200ms ticks
    for _ in 0..8 {
        machine.os.harm += pages_per_step;
        machine.step(SNAPSHOT_MS);
    }
    let cap = machine.ledger.estimate().cap().expect("sustained thrashing sets B");
    let shrunk_bank = machine.bank_bytes();
    assert!(shrunk_bank < step, "bank yielded to sustained thrashing ({shrunk_bank} < {step})");
    for _ in 0..10 {
        machine.step(300);
    }
    assert_eq!(machine.ledger.estimate().cap(), Some(cap), "own release never self-lifts B");
    assert_eq!(machine.bank_bytes(), shrunk_bank);
    // Another program quits, freeing more working set than the cap step.
    machine.os.hot -= step + GB;
    for _ in 0..5 {
        machine.step(300);
    }
    assert!(machine.ledger.estimate().cap().is_none(), "others' release lifts B");
    assert!(machine.bank_bytes() > shrunk_bank, "the bank grows again");
}

/// Pressure names no bytes: B falls one block per pressure event the OS
/// delivers, never per reading; a level that stands with no new event
/// drains nothing more; each reading is answered once, yielding what the
/// process holds above B; the bank never regrows while the level lasts, nor
/// after it ends until others change.
#[test]
fn pressure_yields_a_block_per_event_and_regrows_only_on_evidence() {
    let mut machine = filled(Os::new(), 0);
    // The level set with no event: no pressure step.
    machine.os.pressure = seismic::PressureLevel::Warn;
    for _ in 0..3 {
        machine.step(100);
        assert!(machine.ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "a level with no event drains nothing");
        assert!(machine.ledger.estimate().cap().is_none(), "nor sets B");
    }
    let bank = machine.bank_bytes();
    let mut yielded = 0;
    for event in 0..4 {
        // One event, taken by the store's next reading, then readings with
        // the level standing and none new.
        machine.ledger.pressure_event();
        for reading in 0..3 {
            machine.tick(100);
            machine.read();
            let shortage = machine.ledger.shortage(&std::collections::BTreeSet::new());
            if reading == 0 {
                let shortage = shortage.expect("a pressure event is a real shortage");
                assert!(matches!(shortage.cause, shortage::Cause::Pressure(seismic::PressureLevel::Warn)));
                assert_eq!(shortage.response.relief.as_ref().map(ReliefPlan::covers), Some(GRANULE), "event {event}: one block");
                assert!(!shortage.response.unload);
                assert!(machine.ledger.shortage(&std::collections::BTreeSet::new()).is_none(), "the reading was answered");
                yielded += GRANULE;
            } else {
                assert!(shortage.is_none(), "event {event}: the standing level drains nothing more");
            }
            machine.follow();
            assert_eq!(machine.bank_bytes(), bank - yielded, "the bank follows B down a block an event");
        }
    }
    // The pressure ends: B stands until others change.
    machine.os.pressure = seismic::PressureLevel::Normal;
    for _ in 0..10 {
        machine.step(300);
    }
    assert_eq!(machine.bank_bytes(), bank - yielded, "no regrowth on the pressure's end alone");
    machine.os.hot -= yielded + GB;
    for _ in 0..5 {
        machine.step(300);
    }
    assert!(machine.bank_bytes() > bank - yielded, "others' release admits regrowth");
}

/// The wired ceiling: on unified Metal what the process holds stays under
/// the system's wired limit less others' wired memory, and others' wired
/// rise lowers the grant at once.
#[test]
fn wired_stays_under_the_ceiling() {
    let mut os = Os::new();
    os.wired = Some((40 * GB, 6 * GB));
    os.hot = 4 * GB;
    let mut machine = filled(os, 0);
    for _ in 0..10 {
        machine.step(300);
        let held = machine.ledger.estimate().allocated();
        assert!(held + 6 * GB <= 40 * GB, "wired {held} past the ceiling");
    }
    assert!(machine.ledger.estimate().allocated() + 6 * GB + GRANULE > 40 * GB, "the ceiling binds");
    machine.os.wired = Some((40 * GB, 10 * GB));
    for _ in 0..5 {
        machine.step(300);
        let held = machine.ledger.estimate().allocated();
        assert!(held + 10 * GB <= 40 * GB, "wired {held} past the ceiling");
    }
}

/// A discrete device's memory as its driver's grant shows it: other
/// programs' bytes, the process's, and its frees, shown only once the
/// driver's lag has passed.
struct Driver {
    capacity: u64,
    others: u64,
    ours: u64,
    releasing: Vec<(Instant, u64)>,
    lag: seismic::Lag,
}

impl Driver {
    /// The process freed `bytes` at `at`: shown after the release lag.
    fn release(&mut self, at: Instant, bytes: u64) {
        self.ours -= bytes;
        self.releasing.push((at + self.lag.release, bytes));
    }

    /// The store's reading at `at`: what the grant leaves of its capacity.
    fn reading(&mut self, at: Instant) -> Reading {
        self.releasing.retain(|(shown, _)| *shown > at);
        let used = self.others + self.ours + self.releasing.iter().map(|(_, bytes)| bytes).sum::<u64>();
        Reading::device(self.capacity, self.capacity - used, self.lag)
    }
}

/// A device store beside host RAM keeps its own λ: each store's journal
/// holds the process's change for that store's lag alone. A 24 GB device
/// whose driver shows a free 300 ms late, others holding 4 GB of it, the
/// load's 12 GB of weights and 4 GB of scratch there; host RAM (50 ms) with
/// 2 GB of the load's staging. Both frees made at once: 100 ms later host
/// RAM shows its free and the device does not, and others, the budget and
/// the guarantee stand in both; 400 ms after them the device shows its free
/// too, others still 4 GB and its free room 4 GB more.
#[test]
fn a_device_store_shows_the_process_after_its_own_lag() {
    let now = Arc::new(Lock::new(Instant::now()));
    let clock = now.clone();
    let scope = Ledgers::scripted(move || *clock.lock().expect("the clock"));
    let at = || *now.lock().expect("the clock");
    let tick = |ms: u64| *now.lock().expect("the clock") += Duration::from_millis(ms);
    let host = scope.ledger(seismic::StoreIdentity::HostRam, super::tests::shape());
    let device = scope.ledger(seismic::StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([3; 16])), super::tests::shape());
    let mut os = Os::new();
    os.cold = 0;
    let lag = seismic::Lag { release: Duration::from_millis(300), growth: Duration::ZERO };
    let mut driver = Driver { capacity: 24 * GB, others: 4 * GB, ours: 0, releasing: Vec::new(), lag };
    let read_both = |os: &mut Os, driver: &mut Driver| {
        let at = at();
        read(&host, os, at);
        device.read(device.sampled(at), driver.reading(at));
    };
    read_both(&mut os, &mut driver);
    let charged = |ledger: &Ledger, bytes: u64| {
        let claim = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, bytes)).expect("room");
        ledger.change(|_| { let entry = claim.entry(); entry.take(&entry.change(), bytes, Wiring::Pageable) }).expect("the claim's own bytes");
        claim
    };
    let _weights = charged(&device, 12 * GB);
    let scratch = charged(&device, 4 * GB);
    let staging = charged(&host, 2 * GB);
    driver.ours += 16 * GB;
    os.grow(at(), 2 * GB, Wiring::Pageable);
    tick(100);
    read_both(&mut os, &mut driver);
    let (host_before, device_before) = (host.estimate(), device.estimate());
    assert_eq!((host_before.others(), device_before.others()), (25 * GB, 4 * GB));
    assert_eq!((host_before.lag(), device_before.lag()), (Duration::from_millis(50), Duration::from_millis(300)), "each store's own λ");
    let freed = at();
    for (ledger, claim, bytes) in [(&device, scratch, 4 * GB), (&host, staging, 2 * GB)] {
        ledger.change(|_| {
            let entry = claim.entry();
            entry.freed(&entry.change(), bytes, Wiring::Pageable);
            drop(claim);
        });
    }
    driver.release(freed, 4 * GB);
    os.release(freed, 2 * GB, Wiring::Pageable);
    tick(100);
    read_both(&mut os, &mut driver);
    let (host_now, device_now) = (host.estimate(), device.estimate());
    assert!(!host_now.pending(), "host RAM shows its free within its 50 ms");
    assert!(device_now.pending(), "the device does not, within its 300 ms");
    assert_eq!((host_now.others(), host_now.budget(), host_now.guarantee()), (host_before.others(), host_before.budget(), 0));
    assert_eq!(
        (device_now.others(), device_now.budget(), device_now.guarantee()),
        (device_before.others(), device_before.budget(), 0),
        "the freed bytes stay the process's until the driver shows them"
    );
    tick(300);
    read_both(&mut os, &mut driver);
    let device_shown = device.estimate();
    assert!(!device_shown.pending(), "the device shows it past its lag");
    assert_eq!((device_shown.others(), device_shown.budget(), device_shown.guarantee()), (device_before.others(), device_before.budget(), 0));
    assert_eq!(device_shown.free(), device_before.free() + 4 * GB, "and its bytes are free");
}

/// A Linux cgroup's supply shows the process's growth only within memcg's
/// statistics flush (growth λ = 2 s, `FLUSH_TIME`): until then the cache it
/// displaced still reads as supply, the ledger holds the growth in its
/// journal, and others stand where they were. 20 GB of weights beside
/// others' 25 GB (their 24 GB working set and the OS's 1 GB free target),
/// read every 100 ms for 2.5 s: others stay 25 GB and the budget 39 GB, and
/// the growth leaves the journal at the first reading 2 s after it.
#[test]
fn a_cgroups_growth_shows_after_its_flush() {
    let mut os = Os::new();
    os.cold = 0;
    os.lag = seismic::Lag { release: Duration::ZERO, growth: Duration::from_secs(2) };
    let mut machine = Machine::load(os, 20 * GB, 0);
    let grown = machine.at() - Duration::from_millis(100);
    let before = machine.ledger.estimate();
    assert_eq!((before.others(), before.budget(), before.allocated()), (25 * GB, 39 * GB, 20 * GB));
    assert!(before.pending(), "the growth is not shown yet");
    for _ in 0..24 {
        machine.tick(100);
        machine.read();
        let estimate = machine.ledger.estimate();
        let since = machine.at().duration_since(grown);
        assert_eq!((estimate.others(), estimate.budget(), estimate.guarantee()), (25 * GB, 39 * GB, 0), "{since:?} after the growth");
        assert_eq!(estimate.pending(), since < Duration::from_secs(2), "{since:?} after the growth");
    }
}

/// Where the OS counts no share of the process's paging (a cgroup shared
/// with other processes: no harm read), neither harm nor others' cold
/// pages can be judged: no growth step or boundary is ever set, the cold
/// pages are never supply, and growth is bounded by the supply alone. The OS
/// still reports its pressure, so clean cache is supply. Beside the load's 20 GB the bank takes the 7 GB free above the target
/// and the 4 GB of clean cache, 11 GB, at the first reading after the load,
/// and holds them; the OS compresses none of others' cold pages, and others
/// bringing pages back moves nothing.
#[test]
fn without_harm_growth_is_bounded_by_the_supply_alone() {
    let mut os = Os::new();
    os.counts_own = false;
    let mut machine = Machine::load(os, 20 * GB, 0);
    machine.step(100);
    assert_eq!(machine.ledger.estimate().supply(), 11 * GB as i64, "free above the target and clean cache, no cold pages");
    assert_eq!(machine.bank_bytes(), 11 * GB, "the bank takes the supply at once");
    let grown = machine.resizes;
    for _ in 0..10 {
        machine.step(300);
    }
    machine.os.harm += 4096;
    for _ in 0..5 {
        machine.step(300);
    }
    let estimate = machine.ledger.estimate();
    assert_eq!((machine.bank_bytes(), machine.resizes), (11 * GB, grown), "the bank holds");
    assert_eq!((machine.os.compressed, estimate.cap()), (0, None), "no cold page compressed, no boundary");
    assert!(machine.ledger.shortage(&std::collections::BTreeSet::new()).is_none() && machine.ledger.standing().is_none(), "nothing stands");
}

/// Where the host gives no feedback of any kind (no harm and no pressure
/// level: Linux without PSI, in a cgroup shared with other processes), clean
/// cache is no supply: growth into it would go unobserved. Beside the load's
/// 20 GB the bank takes the 7 GB free above the target and holds it, the 4 GB
/// of clean cache left to others.
#[test]
fn without_harm_and_pressure_clean_cache_is_no_supply() {
    let mut os = Os::new();
    (os.counts_own, os.reports_pressure) = (false, false);
    let mut machine = Machine::load(os, 20 * GB, 0);
    machine.step(100);
    assert_eq!(machine.ledger.estimate().supply(), 7 * GB as i64, "free above the target alone");
    assert_eq!(machine.bank_bytes(), 7 * GB, "the bank takes the supply at once");
    let grown = machine.resizes;
    for _ in 0..10 {
        machine.step(300);
    }
    assert_eq!((machine.bank_bytes(), machine.resizes), (7 * GB, grown), "the bank holds");
}

/// Others' wired bytes and the wired ceiling's grant at `machine`'s latest
/// reading.
fn wired(machine: &Machine) -> (Option<u64>, Option<u64>) {
    let others = machine.ledger.state().wired_others;
    (others, machine.ledger.estimate().read.grant)
}

/// On unified Metal the wire count holds only what the OS wires (§4.3):
/// the process's pageable memory (anonymous host memory, a buffer outside
/// the residency set) is not in it, so it never lowers others' wired bytes
/// nor raises the grant, while the process's wired memory is in it and is
/// subtracted from it as the process's own. A 40 GB ceiling, others wiring
/// 6 GB: beside 20 GB of wired weights, 4 GB of pageable memory and a wired
/// bank granule leave others' wired bytes at 6 GB and the grant at 34 GB;
/// others wiring 2 GB more lower it to 30 GB, their 8 GB and their 2 GB
/// rise.
#[test]
fn only_wired_memory_is_the_processs_in_the_wire_count() {
    let mut os = Os::new();
    os.wired = Some((40 * GB, 6 * GB));
    os.hot = 4 * GB;
    let mut machine = Machine::load(os, 20 * GB, 0);
    assert_eq!(wired(&machine), (Some(6 * GB), Some(34 * GB)));
    machine.tick(100);
    let anonymous = machine.ledger.change(|keeping| machine.ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, 4 * GB)).expect("room");
    machine.ledger.change(|_| { let entry = anonymous.entry(); entry.take(&entry.change(), 4 * GB, Wiring::Pageable) }).expect("the claim's own bytes");
    let at = machine.at();
    machine.os.grow(at, 4 * GB, Wiring::Pageable);
    machine.read();
    let wired_bytes = machine.ledger.state().wired;
    assert_eq!((wired_bytes, machine.ledger.estimate().allocated()), (20 * GB, 24 * GB));
    assert_eq!(wired(&machine), (Some(6 * GB), Some(34 * GB)), "pageable memory is not in the wire count");
    machine.tick(100);
    let _granule = machine.ledger.cache(CacheKind::Bank, GRANULE, super::Reclaim::Unbacked).expect("room");
    machine.ledger.change(|_| { let entry = &machine.pool; entry.take(&entry.change(), GRANULE, Wiring::Wired) }).expect("the pool's block for the granule is admitted");
    let at = machine.at();
    machine.os.grow(at, GRANULE, Wiring::Wired);
    for _ in 0..2 {
        machine.tick(SNAPSHOT_MS);
        machine.read();
        assert_eq!(wired(&machine), (Some(6 * GB), Some(34 * GB)), "wired memory is the process's in the count");
    }
    machine.os.wired = Some((40 * GB, 8 * GB));
    machine.tick(SNAPSHOT_MS);
    machine.read();
    assert_eq!(wired(&machine), (Some(8 * GB), Some(30 * GB)), "others' wired rise lowers the grant");
}

/// The OS swaps the process's own pages out (others' burst came and went)
/// on a dedicated host, where others hold nothing: what the store uses
/// falls below what the process allocated. The swapped bytes are the
/// process's out of RAM, never others' and never a failed reading: where
/// the OS reports them they are `nonresident`, where it does not they are
/// `unobserved_resident` and the reading is counted; others stay none and
/// the budget stands, as the swapped bytes come back only into the supply.
/// 20 GB of pageable weights in 64 GB, 8 GB of them swapped.
#[test]
fn the_os_swapping_the_processs_pages_reads_as_neither_others_nor_room() {
    for reports in [true, false] {
        let mut os = Os::new();
        (os.hot, os.cold, os.target, os.reports_reclaimed) = (0, 0, 0, reports);
        let mut machine = Machine::load(os, 20 * GB, 0);
        let before = machine.ledger.estimate();
        assert_eq!((before.others(), before.nonresident(), before.unobserved_resident()), (0, 0, 0));
        machine.os.swapped = 8 * GB;
        machine.tick(100);
        machine.read();
        let estimate = machine.ledger.estimate();
        assert_eq!((estimate.others(), estimate.budget(), estimate.allocated()), (0, before.budget(), 20 * GB), "reported: {reports}");
        let (nonresident, unobserved, readings) = match reports {
            true => (8 * GB, 0, 0),
            false => (0, 8 * GB, 1),
        };
        assert_eq!((estimate.nonresident(), estimate.unobserved_resident(), estimate.unobserved_readings()), (nonresident, unobserved, readings), "reported: {reports}");
        assert_eq!(estimate.free(), before.free(), "the RAM the swapped bytes left is no new room");
    }
}

/// The OS's compressed store holds some of the process's swapped pages in
/// RAM, at their compressed size: where the OS reports that figure they
/// are the process's and others stay none; where it reports none they read
/// as others', by at most that size. 20 GB of weights, 8 GB of them
/// out of RAM, 3 GB of the OS's compressed store holding them.
#[test]
fn the_oss_compressed_store_of_the_processs_pages_is_not_others() {
    for reports in [true, false] {
        let mut os = Os::new();
        (os.hot, os.cold, os.target, os.reports_stored) = (0, 0, 0, reports);
        let mut machine = Machine::load(os, 20 * GB, 0);
        machine.os.swapped = 8 * GB;
        machine.os.own_stored = 3 * GB;
        machine.tick(100);
        machine.read();
        let estimate = machine.ledger.estimate();
        let others = if reports { 0 } else { 3 * GB };
        assert_eq!((estimate.others(), estimate.nonresident(), estimate.unobserved_resident()), (others, 8 * GB, 0), "reported: {reports}");
    }
}

/// Where the OS counts the process's bytes out of RAM exactly, bytes the
/// store's use does not hold beyond them are an accounting error, counted
/// apart and never as a missing figure; where its count is approximate
/// they are the unobserved term. 20 GB of weights, 8 GB swapped out and
/// 2 GB of those left out of the OS's count.
#[test]
fn an_exact_count_leaves_no_bytes_unobserved_but_an_accounting_error() {
    for exact in [true, false] {
        let mut os = Os::new();
        (os.hot, os.cold, os.target, os.exact) = (0, 0, 0, exact);
        let mut machine = Machine::load(os, 20 * GB, 0);
        machine.os.swapped = 8 * GB;
        machine.os.unreported = 2 * GB;
        machine.tick(100);
        machine.read();
        let estimate = machine.ledger.estimate();
        let (violations, unobserved) = if exact { (1, 0) } else { (0, 1) };
        assert_eq!((estimate.unobserved_resident(), estimate.accounting_violations(), estimate.unobserved_readings()), (2 * GB, violations, unobserved), "exact: {exact}");
        assert_eq!(estimate.others(), 0, "never others'");
    }
}

/// A wire count short of what the process wires (a residency set the OS
/// let lapse, not yet wired again when the counters were read) is no
/// failed reading: the shortfall is its own term, its reading counted, and
/// others' wired bytes and the grant stand at the last count that showed
/// them, with no rise once the set is wired again. A 40 GB ceiling, others
/// wiring nothing, the load's 20 GB wired, 4 GB of it lapsed for one
/// snapshot.
#[test]
fn a_wire_count_short_of_the_processs_is_a_shortfall() {
    let mut os = Os::new();
    os.wired = Some((40 * GB, 0));
    let mut machine = Machine::load(os, 20 * GB, 0);
    assert_eq!(wired(&machine), (Some(0), Some(40 * GB)));
    machine.os.lapsed = 4 * GB;
    machine.tick(SNAPSHOT_MS);
    machine.read();
    let estimate = machine.ledger.estimate();
    assert_eq!((estimate.unwired_shortfall(), estimate.unwired_readings()), (4 * GB, 1), "the shortfall and its reading");
    assert_eq!(wired(&machine), (Some(0), Some(40 * GB)), "others' wired bytes stand");
    machine.os.lapsed = 0;
    machine.tick(SNAPSHOT_MS);
    machine.read();
    let estimate = machine.ledger.estimate();
    assert_eq!((estimate.unwired_shortfall(), estimate.unwired_readings()), (0, 1), "wired again");
    assert_eq!(wired(&machine), (Some(0), Some(40 * GB)), "and no rise as it is");
}

/// An idle span longer than Metal's residency lapse (macOS un-wires a
/// residency set about 1 s after its last GPU work): the device requests
/// residency on the process's own events, so the OS's wire count keeps the
/// process's wired bytes across the span, and neither others' wired bytes
/// nor the grant moves, at the span's end or the reading after it. A 40 GB
/// ceiling, others wiring 6 GB, the load's 20 GB and its 14 GB bank wired,
/// idle for 3 s.
#[test]
fn an_idle_span_past_the_residency_lapse_moves_nothing() {
    let mut os = Os::new();
    os.wired = Some((40 * GB, 6 * GB));
    os.hot = 4 * GB;
    let mut machine = filled(os, 0);
    for _ in 0..3 {
        machine.step(300);
    }
    let before = wired(&machine);
    assert_eq!(before, (Some(6 * GB), Some(34 * GB)));
    let (bank, resizes) = (machine.bank_bytes(), machine.resizes);
    assert_eq!(bank, 14 * GB, "the grant less the weights");
    machine.step(3000);
    assert_eq!(wired(&machine), before, "across the idle span");
    machine.step(100);
    assert_eq!(wired(&machine), before, "and the reading after it");
    assert_eq!((machine.bank_bytes(), machine.resizes), (bank, resizes), "the bank never moves");
}

/// An idle load: 20 GB of weights and a request slot of 1 GB, no request
/// admitted and no bank above its floor, so nothing to yield or preempt.
fn idle(os: Os) -> Machine {
    Machine::start(os, 20 * GB, GB, false)
}

/// `reading` (a reading of the scripted OS with its cause standing) four
/// times: each answered with the shortage standing, nothing to yield or
/// preempt and no change of the process's pending, until it spans more than
/// the store's λ; a load short by anything but its floor never unloads
/// (§4.5). The cause at the last reading.
fn stands_idle(machine: &mut Machine, mut reading: impl FnMut(&mut Machine)) -> shortage::Cause {
    let mut first = None;
    let mut cause = None;
    for _ in 0..4 {
        reading(machine);
        let shortage = machine.ledger.shortage(&std::collections::BTreeSet::new()).expect("the cause stands");
        let estimate = machine.ledger.estimate();
        let covers = shortage.response.relief.as_ref().map_or(0, ReliefPlan::covers);
        assert_eq!((covers, estimate.pending()), (0, false), "nothing to give, nothing pending");
        assert!(!shortage.response.unload, "an idle load short of no floor never unloads");
        first = first.or(Some(estimate.at().at()));
        cause = Some(shortage.cause);
    }
    let first = first.expect("four readings");
    assert!(machine.at().duration_since(first) > machine.ledger.estimate().lag(), "the shortage spans more than λ");
    cause.expect("four readings")
}

/// An idle load whose committed memory, its weights and request slot, lies
/// over a budget others squeezed: others take 15 GB, the OS drops its 4 GB
/// of clean cache and compresses every cold page, and the budget is the
/// load's own 20 GB, the slot's 1 GB short of it at every reading after the
/// first. What is committed cannot be held: past λ, it unloads.
#[test]
fn an_idle_load_over_its_floor_unloads_past_lambda() {
    let mut machine = idle(Os::new());
    machine.os.hot += 15 * GB;
    let mut unloads = Vec::new();
    for _ in 0..4 {
        machine.tick(100);
        machine.read();
        if let Some(shortage) = machine.ledger.shortage(&std::collections::BTreeSet::new()) {
            assert_eq!(shortage.cause, shortage::Cause::FloorShort { over: GB });
            unloads.push(shortage.response.unload);
        }
    }
    assert_eq!(unloads.first(), Some(&false), "one reading decides nothing");
    assert_eq!(unloads.last(), Some(&true), "past λ, the floor short: unload");
}

/// An idle load while others' working sets actively thrash under scarcity
/// across consecutive readings: short by sustained thrashing for more than λ,
/// it never unloads, and its admission waits.
#[test]
fn an_idle_load_under_harm_never_unloads() {
    let mut os = Os::new();
    // Consume free+cache so host sits under scarcity while committed (21 GB) fits budget (25 GB).
    os.hot += 11 * GB;
    let mut machine = idle(os);
    for _ in 0..3 {
        machine.tick(SNAPSHOT_MS);
        machine.read();
    }
    // Arm sustained thrashing streak (> 64 MiB/s over >= 1s) first.
    let pages_per_step = (32 * MB) / PAGE; // 160 MB/s over 200ms
    for _ in 0..5 {
        machine.os.harm += pages_per_step;
        machine.tick(SNAPSHOT_MS);
        machine.read();
    }
    let cause = stands_idle(&mut machine, |machine| {
        for _ in 0..5 {
            machine.os.harm += pages_per_step;
            machine.tick(SNAPSHOT_MS);
            machine.read();
        }
    });
    assert!(matches!(cause, shortage::Cause::Harm { .. }));
}

/// An idle load under the OS's pressure, a new event at every reading and
/// no cache to shed: B never falls below what is committed, so it is short
/// by the pressure for more than λ, never unloads, and its request slot
/// stands; a request past the slot is refused under B.
#[test]
fn an_idle_load_under_pressure_never_unloads() {
    let mut machine = idle(Os::new());
    machine.os.pressure = seismic::PressureLevel::Warn;
    let cause = stands_idle(&mut machine, |machine| {
        machine.ledger.pressure_event();
        machine.tick(100);
        machine.read();
    });
    assert_eq!(cause, shortage::Cause::Pressure(seismic::PressureLevel::Warn));
    let estimate = machine.ledger.estimate();
    assert!(estimate.cap().is_some_and(|cap| cap >= estimate.committed()), "B never falls below what is committed");
    let rows = super::tests::page_rows();
    let _first = machine.ledger.change(|keeping| machine.ledger.reserve(&keeping.claims(), &machine.slot, RequestId(1), rows)).expect("the slot holds the first request");
    // The slot's room left, and a request needing more than it.
    let left = machine.slot.bytes();
    let past = (1..).map(|pages| pages * rows).find(|rows| machine.ledger.bytes(&Need::Request { sequence: RequestId(2), rows: *rows }) > left).expect("a request past the slot");
    assert!(machine.ledger.change(|keeping| machine.ledger.reserve(&keeping.claims(), &machine.slot, RequestId(2), past)).is_err(), "a request past the slot is refused under B");
}

/// An idle load whose bank holds cache above its floor yields just the
/// confirmed sustained thrash excess at Step 1 (`admission_waits == false`)
/// and admits its first request instead of refusing it as if at floor.
#[test]
fn an_idle_load_with_cache_above_its_floor_admits_new_work_when_cache_meets_harm() {
    let mut machine = idle(Os::new());
    for _ in 0..4 {
        machine.step(SNAPSHOT_MS);
    }
    assert!(machine.bank_bytes() > GB, "the idle load's bank holds cache above its floor (free == 0, scarce)");
    for _ in 0..3 {
        machine.tick(SNAPSHOT_MS);
        machine.read();
    }
    // 5 spans * 200ms = 1.0s at 128 MB/s (exceeds 64 MiB/s thrash threshold, confirming on 5th step).
    let pages_per_step = (25_600 * 1024) / PAGE;
    for _ in 0..5 {
        machine.os.harm += pages_per_step;
        machine.tick(SNAPSHOT_MS);
        machine.read();
    }
    let shortage = machine.ledger.shortage(&std::collections::BTreeSet::new()).expect("confirmed sustained thrashing");
    let excess = match shortage.cause {
        shortage::Cause::Harm { bytes } => bytes,
        other => panic!("expected Harm shortage, got {other:?}"),
    };
    assert!(excess > 0);
    assert!(!shortage.response.admission_waits, "cache meets sustained thrash excess above floor");
    let plan = shortage.response.relief.expect("the bank covers the harm");
    assert_eq!(machine.ledger.change(|keeping| machine.ledger.execute(&keeping.claims(), plan)).yielded, excess);
    let admitted = machine.ledger.change(|keeping| machine.ledger.reserve(&keeping.claims(), &machine.slot, RequestId(1), 2 * super::tests::page_rows()));
    assert!(admitted.is_ok(), "the first request is admitted once cache meets the shortage: {:?}", admitted.err());
}

/// Real 64 GiB macOS desktop dynamics (reproducing PID 90382 cold-start load
/// and PID 1441 multi-app churn):
/// 1. Reading #1 (model load open) counts `cold_reclaimable` immediately so a
///    model requiring 13 GB admits even when `free + clean` is only 12 GB and
///    9 GB sits in cold reclaimable pages.
/// 2. Lazy background OS decompressions (20 MB/s), single tab-switch spikes,
///    and a 4 GB wakeup/exit release (`others` dropping, `free` rising) neither
///    tighten `cap` nor trigger panic collapses.
/// 3. Small noisy app GC allocations (`±80 MB`) cause zero bank ping-pong.
#[test]
fn macos_desktop_load_and_background_app_dynamics() {
    let mut os = Os::new();
    // Mirror real 64 GiB Mac snapshot at model open:
    // ~8 GB free beyond target, ~4 GB clean cache, ~18 GB cold -> 9 GB cold_reclaimable (~21 GB total supply).
    os.hot = 33 * GB;
    os.cache = 4 * GB;
    os.cold = 18 * GB;
    os.wired = Some((48 * GB, 4 * GB));
    let mut machine = Machine::start(os, 0, 0, false);
    // Immediately on reading #1 (no quiet baseline yet!), room covers 13 GB > 12 GB free+clean.
    let first_est = machine.ledger.estimate();
    assert!(first_est.free() >= 19 * GB, "reading #1 exposes cold_reclaimable ({})", first_est.free());
    let floor_weights = machine.ledger.change(|keeping| machine.ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, 13 * GB))
        .expect("13 GB model load admits on reading #1 alongside cold_reclaimable");
    machine.ledger.change(|_| floor_weights.entry().take(&floor_weights.entry().change(), 13 * GB, Wiring::Wired)).unwrap();
    let at = machine.at();
    machine.os.grow(at, 13 * GB, Wiring::Wired);

    // Fill bank smoothly into remaining headroom including cold.
    for _ in 0..5 {
        if machine.bank_bytes() >= 6 * GB {
            break;
        }
        machine.step(SNAPSHOT_MS);
    }
    let bank_before = machine.bank_bytes();
    assert!(bank_before >= 6 * GB, "bank grows cleanly into cold headroom ({bank_before})");

    // Background Electron/Safari lazy decompressions (20 MB/s) + single 160 MB spike + 4 GB exit wakeup!
    for _ in 0..5 {
        machine.os.harm += (20 * MB) / PAGE;
        machine.step(1000);
        assert_eq!(machine.ledger.estimate().cap(), None, "normal decompression noise never sets B");
    }
    machine.os.harm += (160 * MB) / PAGE;
    machine.step(1000);
    machine.os.harm += (10 * MB) / PAGE;
    machine.step(1000);
    assert_eq!(machine.ledger.estimate().cap(), None, "one-shot wakeup never sets B");

    // App frees 4 GB while triggering 4 GB of OS decompressions (PID 1441 10:25:04 scenario).
    machine.os.harm += (4 * GB) / PAGE;
    machine.os.hot -= 4 * GB;
    machine.step(1000);
    assert_eq!(machine.ledger.estimate().cap(), None, "exogenous release with decompressions never panics");

    // Re-fill bank to new room and confirm ±80 MB GC jitter produces zero resize ping-pong.
    machine.step(SNAPSHOT_MS);
    let settled_resizes = machine.resizes;
    let settled_bank = machine.bank_bytes();
    for tick in 0..10 {
        match tick % 2 {
            0 => machine.os.hot += 80 * MB,
            _ => machine.os.hot -= 80 * MB,
        }
        machine.step(SNAPSHOT_MS);
    }
    assert_eq!((machine.resizes, machine.bank_bytes()), (settled_resizes, settled_bank), "80 MB app jitter causes zero bank oscillation");
    drop(floor_weights);
}

/// Real Linux desktop dynamics (both shared cgroup `harm == None` with PSI and
/// post-load pageable CPU weights with `own_cold == None`):
/// 1. Retaining CPU weights (`Pageable`, `wired == 0`) keeps remaining machine
///    `cold_reclaimable` intact rather than triggering a false post-load
///    `FloorShort` cliff.
/// 2. With `harm == None` + `pressure = Warn` + initial PSI event, standing
///    `Warn` yields past 1 s dwell even when PSI rate-limiting emits no second event.
#[test]
fn linux_desktop_post_load_cold_supply_and_standing_psi() {
    // Part 1: dedicated Linux cgroup (`counts_own = true`) retains cold_reclaimable
    // immediately on load and after 20 GB pageable CPU weights are allocated.
    let mut os = Os::new();
    os.counts_own = true;
    os.hot = 20 * GB;
    os.cold = 16 * GB; // 8 GB cold_reclaimable
    os.cache = 4 * GB;
    let machine = Machine::start(os, 20 * GB, GB, false);
    assert!(machine.ledger.estimate().free() >= 10 * GB, "post-load CPU model keeps cold supply ({})", machine.ledger.estimate().free());

    // Part 2: shared cgroup (`counts_own = false`, harm == None) with PSI Warn
    // fires on initial PSI event and steps down smoothly past 1 s dwell even
    // when PSI rate-limiting emits no second edge event.
    let mut shared_os = Os::new();
    shared_os.counts_own = false;
    let mut machine = filled(shared_os, 0);
    machine.os.pressure = seismic::PressureLevel::Warn;
    machine.ledger.pressure_event();
    machine.tick(100);
    machine.read();
    machine.follow();
    let after_event = machine.bank_bytes();
    assert_eq!(after_event, 11 * GB - GRANULE);

    // Standing Warn with 0 new events steps down 1 more block past 1s dwell.
    machine.tick(1100);
    machine.read();
    let shortage = machine.ledger.shortage(&std::collections::BTreeSet::new()).expect("standing Warn yields past 1s dwell");
    assert_eq!(shortage.cause, shortage::Cause::Pressure(seismic::PressureLevel::Warn));
    machine.follow();
    assert_eq!(machine.bank_bytes(), after_event - GRANULE);
}

/// Real Windows desktop dynamics (`harm == None`, `cold_reclaimable == 0`,
/// level-triggered `LowMemoryResourceNotification` firing once on edge then
/// staying `Warn`/`Critical` with zero subsequent events): yields smoothly block
/// by block until pressure lifts, then regrows only on exogenous release.
#[test]
fn windows_level_triggered_pressure_yields_smoothly_until_relief() {
    let mut os = Os::new();
    os.counts_own = false;
    os.cold = 0; // Windows trims to standby (`cache`), cold_reclaimable == 0
    let mut machine = filled(os, 0);
    let full_bank = machine.bank_bytes();
    assert_eq!(full_bank, 19 * GB);

    // Windows LowMemoryResourceNotification fires once on transition into Warn.
    machine.os.pressure = seismic::PressureLevel::Warn;
    machine.ledger.pressure_event();
    machine.step(100);
    assert_eq!(machine.bank_bytes(), full_bank - GRANULE);

    // Pressure remains `Warn` with zero further OS edge events (level-triggered).
    // Stepping past 1 s dwell continues yielding cleanly without freezing.
    machine.tick(1100);
    machine.read();
    assert!(machine.ledger.shortage(&std::collections::BTreeSet::new()).is_some());
    machine.follow();
    assert_eq!(machine.bank_bytes(), full_bank - 2 * GRANULE);

    // Escalation to Critical (commit headroom exhausted) yields every 250 ms without events.
    machine.os.pressure = seismic::PressureLevel::Critical;
    machine.tick(300);
    machine.read();
    assert!(machine.ledger.shortage(&std::collections::BTreeSet::new()).is_some());
    machine.follow();
    assert_eq!(machine.bank_bytes(), full_bank - 3 * GRANULE);

    // Pressure clears, bank holds under cap until exogenous app release lifts cap.
    machine.os.pressure = seismic::PressureLevel::Normal;
    for _ in 0..3 {
        machine.step(300);
    }
    assert_eq!(machine.bank_bytes(), full_bank - 3 * GRANULE);
    machine.os.hot -= 4 * GB;
    machine.step(300);
    assert!(machine.bank_bytes() > full_bank - 3 * GRANULE, "exogenous release lifts cap cleanly");
}

/// A store with paging (such as `HostRam` on a dedicated GPU) may hold
/// non-pooled cache (`GraphSlots`, `Drafter`, `Vision`) without ever opening
/// a granule pool (`PoolClass::State` or `PoolClass::Records`). Readings must
/// not panic in `State::unit()`, and OS pressure events must lower `cap` by
/// the largest non-pooled cache entry and relieve it cleanly.
#[test]
fn non_pooled_cache_without_open_pool_reads_and_relieves_under_pressure() {
    let mut os = Os::new();
    let now = Arc::new(Lock::new(Instant::now()));
    let clock = now.clone();
    let scope = Ledgers::scripted(move || *clock.lock().expect("the clock"));
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, super::tests::shape());

    let at = *now.lock().expect("the clock");
    read(&ledger, &mut os, at);

    let weights = 2 * GB;
    let load = ledger.change(|keeping| ledger.claim_at(&keeping.claims(), None, CommittedKind::Load, weights)).expect("load fits");
    ledger.change(|_| { let entry = load.entry(); entry.take(&entry.change(), weights, Wiring::Pageable) }).expect("load allocated");
    os.grow(at, weights, Wiring::Pageable);

    let graph_bytes = 32 * MB;
    let drafter_bytes = 64 * MB;
    let graph_slots = ledger.cached(CacheKind::GraphSlots, graph_bytes).expect("graph slots fit");
    os.grow(at, graph_bytes, Wiring::Pageable);
    let drafter = ledger.cached(CacheKind::Drafter, drafter_bytes).expect("drafter fits");
    os.grow(at, drafter_bytes, Wiring::Pageable);

    // Paging reading with non-pooled cache and no opened granule pools succeeds.
    *now.lock().expect("the clock") += Duration::from_millis(100);
    read(&ledger, &mut os, *now.lock().expect("the clock"));
    assert_eq!(ledger.estimate().cache(), graph_bytes + drafter_bytes);
    assert!(ledger.estimate().cap().is_none());

    // Pressure lowers B by the largest non-pooled cache entry (64 MB) and
    // plans cache-only relief for it.
    os.pressure = seismic::PressureLevel::Warn;
    ledger.pressure_event();
    *now.lock().expect("the clock") += Duration::from_millis(100);
    read(&ledger, &mut os, *now.lock().expect("the clock"));
    let est = ledger.estimate();
    let held = weights + graph_bytes + drafter_bytes;
    assert_eq!(est.cap(), Some(held - drafter_bytes), "pressure lowers B by largest non-pooled cache entry");

    let shortage = ledger.shortage(&std::collections::BTreeSet::new()).expect("pressure is a real shortage");
    assert!(matches!(shortage.cause, shortage::Cause::Pressure(seismic::PressureLevel::Warn)));
    let plan = shortage.response.relief.expect("cache covers pressure relief");
    assert_eq!(plan.covers(), drafter_bytes);
    let owed = ledger.change(|keeping| ledger.execute(&keeping.claims(), plan));
    assert_eq!(owed.yielded, drafter_bytes);
    os.release(*now.lock().expect("the clock"), drafter_bytes, Wiring::Pageable);

    // Pressure clears and others release memory -> once the growth step closes
    // and a calm baseline is established, B lifts cleanly.
    os.pressure = seismic::PressureLevel::Normal;
    os.hot = os.hot.saturating_sub(GB);
    for _ in 0..4 {
        *now.lock().expect("the clock") += Duration::from_millis(SNAPSHOT_MS);
        read(&ledger, &mut os, *now.lock().expect("the clock"));
    }
    assert_eq!(ledger.estimate().cap(), None, "restored supply lifts B");

    drop((graph_slots, drafter, load));
}

/// Pooled cache (`CacheKind::Bank`, `Prefix`, `Spare`) held in a store whose
/// pool of that class never opened is still an invariant violation.
#[test]
#[should_panic(expected = "pool never opened")]
fn pooled_cache_without_open_pool_still_violates_invariant() {
    let mut os = Os::new();
    let now = Arc::new(Lock::new(Instant::now()));
    let clock = now.clone();
    let scope = Ledgers::scripted(move || *clock.lock().expect("the clock"));
    let ledger = scope.ledger(seismic::StoreIdentity::HostRam, super::tests::shape());

    let at = *now.lock().expect("the clock");
    read(&ledger, &mut os, at);
    let _bank = ledger.cache(CacheKind::Bank, 32 * MB, super::Reclaim::Unbacked).expect("unbacked bank cache fits");
    *now.lock().expect("the clock") += Duration::from_millis(100);
    read(&ledger, &mut os, *now.lock().expect("the clock"));
}
