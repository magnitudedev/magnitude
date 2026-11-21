//! A reading as the ledger estimated it, and the estimate every decision reads.

use std::time::Duration;

use crate::memory::shortage;
use crate::memory::supply::{Ceiling, HostGrant, Observed, Reading};

use super::changes::Sampled;
use super::state::State;
use super::vocabulary::CacheKind;

/// A reading as the ledger estimated it (§4.1-4.3): every figure of one
/// sample's instant, the process's own as the sample shows it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Read {
    pub(super) at: Sampled,
    /// The store's structural capacity (`Reading::capacity`): the most bytes
    /// the store could grant the process beside nothing else.
    capacity: u64,
    /// The store's supply, others' cold pages in it only once harm is
    /// judged ([`Boundary::judges`]).
    supply: i64,
    /// What the process had allocated as the sample shows it (including any
    /// own growth within `lag.growth` that the OS already flushed into
    /// `supply`).
    visible: u64,
    /// Of `visible`, the pageable bytes the OS reports holding outside RAM,
    /// compressed or swapped ([`Reading::own_reclaimed`]): the process's
    /// still, and in no figure of what the store uses.
    nonresident: u64,
    /// Of `visible` beyond `nonresident`, what the store's use does not hold
    /// and the OS did not report: [`Estimate::unobserved_resident`].
    unobserved: u64,
    /// What the grants' processes hold of their grants
    /// (`State::credited`): the process's own in the store's use, held by
    /// the grants' committed bytes, never allocatable by this process.
    credited: u64,
    /// Others' memory: what the store uses beyond the supply and the
    /// process's own bytes in it as the sample shows them.
    others: u64,
    /// Others' rise since the reading before.
    rise: u64,
    /// The wired ceiling's grant: the limit less others' wired bytes and
    /// their rise; none where no ceiling bounds the store.
    pub(super) grant: Option<u64>,
    /// The system's wired ceiling and wire count at the reading, as the
    /// sample gave them; none where no ceiling bounds the store.
    ceiling: Option<Ceiling>,
    /// What the process wires beyond the OS's wire count:
    /// [`Estimate::unwired_shortfall`].
    unwired: u64,
    lag: seismic::Lag,
    /// Whether the reading shows every change of the process's: others'
    /// rise is judged only between two such readings.
    shown: bool,
    /// The boundary B the reading left standing (`memory::supply`).
    cap: Option<u64>,
    /// Growth is admissible at the reading: the store reads no paging
    /// signals, or a baseline judges harm and no growth step's window λ_h
    /// is open.
    grows: bool,
}

impl Read {
    /// What the store's use held of the process's bytes at the reading:
    /// those the sample shows, less those out of RAM, reported or not, and
    /// what its grants' processes hold of their grants.
    fn in_use(&self) -> u64 {
        self.visible - self.nonresident - self.unobserved + self.credited
    }
}

/// The ledger's estimate of its store at the latest reading's instant
/// (§4.1-4.4): every figure one decision reads, the entries as they stand
/// and the reading as the journal attributes it.
#[derive(Clone, Debug)]
pub struct Estimate {
    pub(super) read: Read,
    committed: u64,
    cache: u64,
    /// What entries' holders allocated of them, and cache still releasing.
    allocated: u64,
    /// What cache yielded and its holders have not freed yet.
    releasing: u64,
    /// Of it, what was shed to a shortage: held by no claim.
    shed: u64,
    cache_of: [u64; CacheKind::ALL.len()],
    /// Of the spare cache, the records pool's free granules: the only spare
    /// the bank's slots grow into, each pool holding its own class
    /// (`PoolClass`).
    bank_spare: u64,
    /// A change of the process's no reading shows yet.
    pending: bool,
    /// The store's readings so far with some of the process's bytes
    /// unobserved ([`Self::unobserved_resident`]).
    unobserved_readings: u64,
    /// The store's readings so far with bytes of the process's that its
    /// use does not hold, where the OS counts those out of RAM exactly:
    /// the ledger charged what the OS does not show
    /// ([`Self::accounting_violations`]).
    accounting_violations: u64,
    /// The store's readings so far with a wire count short of the
    /// process's wired bytes ([`Self::unwired_shortfall`]).
    unwired_readings: u64,
}

impl Estimate {
    /// The instant of the reading it estimates.
    pub fn at(&self) -> Sampled {
        self.read.at
    }

    /// The store's lag: the longest any change of the process's takes to
    /// show in its live figures ([`seismic::Lag`]). A shortage persists
    /// across readings spanning more than it (`memory::shortage`).
    pub fn lag(&self) -> Duration {
        let lag = self.read.lag;
        lag.release.max(lag.growth)
    }

    /// Committed bytes held.
    pub fn committed(&self) -> u64 {
        self.committed
    }

    /// Cache bytes held.
    pub fn cache(&self) -> u64 {
        self.cache
    }

    /// Cache bytes of `kind` held.
    pub fn cache_of(&self, kind: CacheKind) -> u64 {
        self.cache_of[kind.index()]
    }

    /// What entries hold, committed and cache (yielded releasing cache excluded).
    pub(crate) fn held(&self) -> u64 {
        self.committed + self.cache
    }

    /// What the process holds now in the store's room: its entries, and
    /// what cache shed to a shortage still releases.
    pub(crate) fn held_now(&self) -> u64 {
        self.held().saturating_add(self.shed)
    }

    /// Bytes the process allocated: of what it holds, and what cache
    /// yielded that its holders have not freed yet.
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// What cache yielded that its holders have not freed yet
    /// ([`Cached::released`]).
    pub fn releasing(&self) -> u64 {
        self.releasing
    }

    /// Of releasing cache, what was shed to a shortage (held by no claim).
    pub(super) fn shed(&self) -> u64 {
        self.shed
    }

    /// What grants' processes hold of their grants in this store's use.
    pub(super) fn credited(&self) -> u64 {
        self.read.credited
    }

    /// The store's supply at the reading (§4.2): free memory less the OS's
    /// target, clean cache and, once harm is judged, cold reclaimable pages;
    /// below zero where the OS has fallen below its target by more than the
    /// rest gives.
    pub fn supply(&self) -> i64 {
        self.read.supply
    }

    /// Other programs' memory at the reading: what the store uses beyond
    /// its supply and the process's own as the sample shows it. No claim or
    /// free of the process's moves it.
    pub fn others(&self) -> u64 {
        self.read.others
    }

    /// The process's pageable bytes the OS reported holding outside RAM
    /// at the reading, compressed or swapped
    /// (`seismic::HostSample::own_reclaimed`): the process's still, held by
    /// its entries and never others', and out of the budget, as they come
    /// back into RAM only from the supply ([`Self::budget`]).
    pub fn nonresident(&self) -> u64 {
        self.read.nonresident
    }

    /// The process's bytes at the reading that neither what the store used
    /// held nor the OS reported outside RAM: out of RAM unreported (no OS
    /// figure, or an approximate one), or a device driver's charge counted
    /// at its bound. Never others', and out of the budget as
    /// [`Self::nonresident`] is; none where every byte was observed.
    pub fn unobserved_resident(&self) -> u64 {
        self.read.unobserved
    }

    /// The store's readings so far at which some of the process's bytes
    /// were unobserved ([`Self::unobserved_resident`]) where the OS gives no
    /// exact count of those out of RAM.
    pub fn unobserved_readings(&self) -> u64 {
        self.unobserved_readings
    }

    /// The store's readings so far at which bytes were unobserved where the
    /// OS counts those out of RAM exactly (macOS): the ledger charged bytes
    /// the OS does not show. An error, never a missing figure; none in a
    /// correct run, so a run's gate fails on any.
    pub fn accounting_violations(&self) -> u64 {
        self.accounting_violations
    }

    /// The wired ceiling's limit at the reading, `min(iogpu.wired_limit_mb
    /// or its default, recommendedMaxWorkingSetSize)`; none where no
    /// ceiling bounds the store.
    pub fn wired_limit(&self) -> Option<u64> {
        self.read.ceiling.map(|ceiling| ceiling.limit)
    }

    /// The bytes the OS wires machine-wide at the reading, the process's
    /// included; none where no ceiling bounds the store.
    pub fn wire_count(&self) -> Option<u64> {
        self.read.ceiling.map(|ceiling| ceiling.wired)
    }

    /// What the process wires beyond the OS's wire count at the reading:
    /// a residency set the OS let lapse and has not wired again. Others'
    /// wired bytes stand at the last count that showed them meanwhile.
    pub fn unwired_shortfall(&self) -> u64 {
        self.read.unwired
    }

    /// The store's readings so far whose wire count was short of the
    /// process's wired bytes ([`Self::unwired_shortfall`]).
    pub fn unwired_readings(&self) -> u64 {
        self.unwired_readings
    }

    /// The store's structural capacity at the reading (`Reading::capacity`):
    /// the most bytes it could grant the process beside nothing else.
    pub fn capacity(&self) -> u64 {
        self.read.capacity
    }

    /// What the process may hold in the store: `min(grant, in_use +
    /// supply)`, `in_use` what the store's use holds of the process's
    /// allocated bytes as the sample shows them (those out of RAM come
    /// back only into the supply), the grant the wired ceiling's where one
    /// bounds the store (§4.3).
    pub fn budget(&self) -> u64 {
        let room = u64::try_from((i128::from(self.read.in_use()) + i128::from(self.read.supply)).max(0)).expect("a store's bytes fit u64");
        self.read.grant.map_or(room, |grant| grant.min(room))
    }

    /// Room a new holding may take: the budget less everything held and
    /// what cache shed to a shortage still releases, allocated by no entry
    /// that holds it. What a claim took in process is held by the claim.
    pub fn free(&self) -> u64 {
        self.budget().saturating_sub(self.held().saturating_add(self.shed))
    }

    /// The one bound on an allocation (Seismic's, through an entry): the
    /// bytes the budget leaves beside what the process has allocated,
    /// releasing bytes included. A claim that took cache in process
    /// allocates those bytes only once their holder freed them.
    pub fn allocatable(&self) -> u64 {
        self.budget().saturating_sub(self.allocated.saturating_add(self.read.credited))
    }

    /// The speed guarantee: free room the bank's target leaves for growth
    /// that arrives within one reading, others' rise since the reading
    /// before. Nothing of the process's own grows unreserved: a request's
    /// rows draw within its reservation ([`Reservation`]), and a grant's
    /// process's growth is credited against the grant.
    pub fn guarantee(&self) -> u64 {
        self.read.rise
    }

    /// What the process holds past the budget less the guarantee: the
    /// guarantee's room its holdings spend, exact where free room
    /// ([`Self::free`]) reads none.
    pub fn over_guarantee(&self) -> u64 {
        self.held().saturating_add(self.shed).saturating_add(self.guarantee()).saturating_sub(self.budget())
    }

    /// What a claim may take beside what is committed and the guarantee:
    /// free room and every cache, which yields to it.
    pub fn room(&self) -> u64 {
        self.budget().saturating_sub(self.committed.saturating_add(self.guarantee()))
    }

    /// The boundary B: the most the process may hold, where it stands.
    pub fn cap(&self) -> Option<u64> {
        self.read.cap
    }

    /// Whether a change of the process's is not yet shown by the reading.
    pub fn pending(&self) -> bool {
        self.pending
    }

    /// The room the store leaves the bank and its records pool's spare
    /// granules before growth's harm/step gate (`Self::target`): `budget` less
    /// committed memory, other caches, unreleased shed cache, and the guarantee,
    /// bounded by `cap`. Load import sizes what the store's room holds at open
    /// (`experts::derived::import_target`).
    pub fn import_room(&self) -> u64 {
        self.room_less(self.guarantee())
    }

    /// The room the store leaves the bank and its records pool's spare
    /// granules for holding already-resident slots without subtracting the
    /// transient `guarantee` derivative a second time (`budget` already
    /// reflects others' growth).
    pub fn hold_room(&self) -> u64 {
        self.room_less(0)
    }

    fn room_less(&self, reserve: u64) -> u64 {
        let bank = self.cache_of(CacheKind::Bank) + self.bank_spare;
        let beside = self.committed.saturating_add(self.cache - bank).saturating_add(self.shed);
        let room = self.budget().saturating_sub(beside.saturating_add(reserve));
        self.read.cap.map_or(room, |cap| room.min(cap.saturating_sub(beside)))
    }

    pub fn target(&self) -> u64 {
        let bank = self.cache_of(CacheKind::Bank) + self.bank_spare;
        let room = self.import_room();
        match self.read.grows {
            true => room,
            false => room.min(bank),
        }
    }
}

impl State {
    /// The estimate of the store now; none before its first reading.
    pub(super) fn estimate(&self) -> Option<Estimate> {
        self.read.map(|read| self.estimate_at(read))
    }

    /// The estimate of the store at `read`'s OS figures with what the
    /// process holds now: the latest reading's ([`Self::estimate`]), or
    /// one a load holds ([`DeviceReading::room_after`]), every holding
    /// since counted.
    pub(super) fn estimate_at(&self, read: Read) -> Estimate {
        Estimate {
            read,
            committed: self.committed,
            cache: self.cache,
            allocated: self.allocated,
            releasing: self.releasing,
            shed: self.releasing - self.taken,
            cache_of: self.cache_of,
            bank_spare: self.spare_of(super::vocabulary::PoolClass::Records),
            pending: self.journal.pending(),
            unobserved_readings: self.unobserved_readings,
            accounting_violations: self.accounting_violations,
            unwired_readings: self.unwired_readings,
        }
    }

    /// The reading a decision is judged at: `at`, a reading a load holds
    /// ([`DeviceLedgers::claim_at`]), or else the latest. The one way a
    /// judgment obtains its reading: what admits, covers or sizes takes a
    /// [`Read`], so none can be made of a store with none. A claim is made
    /// only in ledgers read before they claim ([`super::UnreadLedgers`]),
    /// and Seismic's charge of a device's own entry in a store never read
    /// takes no judgment (`State::entry_take`).
    ///
    /// # Panics
    /// Where the store was never read: no ledger that judges exists then.
    pub(super) fn reading(&self, store: &seismic::StoreIdentity, at: Option<Read>) -> Read {
        at.or(self.read).unwrap_or_else(|| panic!("invariant violated: {store:?} admits before its first reading"))
    }

    /// The estimate at the latest reading ([`Self::reading`]).
    pub(super) fn judged(&self, store: &seismic::StoreIdentity) -> Estimate {
        self.estimate_at(self.reading(store, None))
    }

    /// Fold in `reading`, sampled at `at` (§4.1-4.3), `events`, the OS's
    /// pressure events in the store no reading took yet, and `after_idle`,
    /// whether the store had no readings across an unmonitored idle sleep;
    /// each grant's process's footprint read from `footprints`.
    pub(super) fn read(&mut self, store: &seismic::StoreIdentity, at: Sampled, reading: Reading, events: u64, after_idle: bool, footprints: &super::grant::Footprinted) {
        self.readings += 1;
        self.waiting = false;
        let settled = after_idle && events == 0;
        let lag = reading.lag();
        // The live figures show the machine at the sample's instant; the
        // counters (cold, harm, wire count) at their own, which may be
        // earlier (`seismic::Counted`): what they show of the process is
        // judged there, and a change leaves the journal only once every
        // figure shows it.
        let counted_at = reading.paging().map_or(at.at(), |paging| paging.counted.at);
        let own = self.own();
        let visible = self.journal.visible(own, at, at.at(), lag);
        let counted = self.journal.visible(own, at, counted_at, lag);
        self.journal.settle(at, counted_at.min(at.at()), lag);
        // Every change of the process's shows in this reading.
        let shown = !self.journal.pending();
        let figures = reading.supply();
        // None of the process's pages is supply (its cold pages are left
        // out of it below), but those the OS compressed or swapped out are
        // in no figure of what the store uses either. The OS reports those
        // where it can ([`Reading::own_reclaimed`]), counting anonymous
        // memory the ledger does not hold beside the process's pageable
        // bytes, so they are bounded by those; wired bytes are never
        // reclaimed. What the store's use holds of the process's is the
        // rest, which with the supply is what only others' changes move.
        let pageable = visible.allocated.saturating_sub(visible.wired);
        let reclaimed = reading.own_reclaimed();
        let nonresident = reclaimed.map_or(0, |own| own.bytes.min(pageable));
        let resident = visible.allocated - nonresident;
        // Others' cold pages are supply by the judgment of the readings
        // before this one: the boundary folds this reading below, with the
        // room the process's bytes in use leave, which the supply sets.
        let rest = i64::try_from(reading.clean_cache_supply() + self.cold_supply(&reading, figures)).expect("a store's bytes fit i64");
        let supply = figures.free_beyond_target + rest;
        // What the store uses beyond its supply is others' and the
        // process's own bytes in RAM. What its use does not hold of
        // `resident` is the process's bytes out of RAM that the OS did not
        // report: where it reports none (a cgroup shared with other
        // processes, Windows before 1809), where its count is approximate
        // (Linux's batched swap count), or where the store is a device's,
        // whose driver objects are charged at a bound (CUDA's module and
        // stack charge). Never others', and no failed reading: a term of
        // its own, its readings counted. Where the OS's count is exact
        // (macOS) nothing is unreported: the shortfall is a ledger charge
        // the OS does not show, an accounting error, counted apart and
        // logged. It stays a count, never a panic: a driver's charge or a
        // page the OS has not yet faulted in may produce it, and a budget
        // lowered by it is the safe side.
        // A supply below zero is the OS below its target: others' (or the
        // process's) growth it has not yet reclaimed for, used all the same.
        let used = u64::try_from(i128::from(reading.total()) - i128::from(supply)).expect("a store's use is no less than none");
        let unobserved = resident.saturating_sub(used);
        if unobserved > 0 {
            match reclaimed.is_some_and(|own| own.exact) {
                true => {
                    self.accounting_violations += 1;
                    tracing::error!(
                        target: crate::MEMORY_TRACE,
                        unobserved,
                        resident,
                        used,
                        "the ledger holds bytes in RAM that the store's use does not, where the OS reports its own pages out of RAM exactly"
                    );
                }
                false => self.unobserved_readings += 1,
            }
        }
        // A grant's process's memory is the grant's, within its bytes: its
        // growth is the process's own in the store's use, never others'
        // (none where its footprint is unobserved).
        let credited = self.credited(store, footprints).min(used.saturating_sub(resident - unobserved));
        let in_use = resident - unobserved + credited;
        // The OS's compressed store holds the process's compressed pages in
        // RAM, in what the store uses and in no byte of `in_use`: its
        // figure is an estimate at the compressor's machine-wide ratio, so
        // others' are no less than none.
        let stored = reclaimed.map_or(0, |own| own.stored.min(nonresident));
        let raw_others = (used - in_use).saturating_sub(stored);
        // Under a positive `lag.growth` (e.g. a Linux cgroup's 2 s memcg
        // flush), the OS may flush an own growth into `supply` earlier than
        // `lag.growth` (Linux flushes `memory.stat` past a few MiB of updates).
        // Any rise in `raw_others` matched by an unexplained drop in `in_use +
        // supply` up to `unshown_growth` is our own growth already deducted
        // from `supply`: credit it to `in_use` rather than `others` so
        // `budget()`, `free()`, `allocatable()`, `hold_room()`, and
        // `import_room()` never double-subtract our own growth while staying
        // exact (zero double-spending) when the OS has not flushed yet.
        let unshown_growth = own.allocated.saturating_sub(visible.allocated);
        let flushed = match self.read.filter(|_| !settled && unshown_growth > 0) {
            None => 0,
            Some(before) => {
                let risen = raw_others.saturating_sub(before.others);
                let fall = i128::from(before.supply) - i128::from(supply) - (i128::from(in_use) - i128::from(before.in_use()));
                let drop = u64::try_from(fall.max(0)).expect("a supply's fall fits u64");
                unshown_growth.min(risen).min(drop)
            }
        };
        let visible_allocated = visible.allocated + flushed;
        let in_use = in_use + flushed;
        let others = raw_others - flushed;
        if settled {
            self.boundary.settle_idle();
        }
        if let Some(paging) = reading.paging() {
            let room = in_use.saturating_add(figures.total());
            let free = u64::try_from(figures.free_beyond_target.max(0)).unwrap_or(0).saturating_add(reading.clean_cache_supply());
            let held = self.committed.saturating_add(self.cache).saturating_add(self.releasing - self.taken);
            self.boundary.observe(Observed {
                paging,
                total: reading.total(),
                allocated: counted.allocated,
                held,
                room,
                free,
                unit: self.unit(),
                events,
                committed: self.committed,
            });
        }
        // Others' rise counts only where it is attributed to them, between
        // readings that show every change of the process's: their harm, or a
        // fall of the supply that the process's own bytes in use do not
        // explain. A total that moves (a binding scope's) or the process's
        // own change shown out of step raises `others` with no such fall,
        // and raises nothing. A reading that cannot judge it, its own change
        // not yet shown in either side, keeps the rise last judged: our own
        // resize must not erase the rise it answered (macOS's counters, a
        // second stale, left 285 of 1,184 readings unshown on an M4 Pro,
        // and each one regrew a slot just shed into that rise). A reading
        // after an idle sleep judges none: the rise is none.
        let rise = match self.read.filter(|_| !settled) {
            None => 0,
            Some(before) if !(before.shown && shown) => before.rise,
            Some(before) => {
                let risen = others.saturating_sub(before.others);
                match self.boundary.harmed() {
                    Some(_) => risen,
                    None => {
                        let fall = i128::from(before.supply) - i128::from(supply) - (i128::from(in_use) - i128::from(before.in_use()));
                        risen.min(u64::try_from(fall.max(0)).expect("a supply's fall fits u64"))
                    }
                }
            }
        };
        let mut unwired = 0;
        let grant = match reading.grant() {
            Some(HostGrant::Wired(ceiling)) => {
                // Others' wired bytes: what the OS wires beyond the process's
                // own wired bytes as the counters show them. The process's
                // pageable memory is never in the count.
                let (wired_others, wired_rise) = match ceiling.wired.checked_sub(counted.wired) {
                    Some(wired_others) => {
                        let judged = !settled && self.read.is_some_and(|before| before.shown) && shown;
                        let wired_rise = self.wired_others.filter(|_| judged).map_or(0, |before| wired_others.saturating_sub(before));
                        self.wired_others = Some(wired_others);
                        (wired_others, wired_rise)
                    }
                    // The count shows less than the process wires: a residency
                    // set the OS let lapse, not yet wired again when the
                    // counters were read (whether Metal's `requestResidency`
                    // returns only once the set is wired is not measured), so
                    // the count shows neither side whole. The shortfall is its
                    // own term, its readings counted; others' wired bytes stand
                    // at the last count that showed them (the whole count, the
                    // most they can be, before one), and rise by nothing.
                    None => {
                        unwired = counted.wired - ceiling.wired;
                        self.unwired_readings += 1;
                        (self.wired_others.unwrap_or(ceiling.wired), 0)
                    }
                };
                Some(ceiling.limit.saturating_sub(wired_others).saturating_sub(wired_rise))
            }
            Some(HostGrant::Limit(limit)) => Some(limit),
            Some(HostGrant::Budget { capacity, left }) => Some(visible_allocated.saturating_add(left).min(capacity)),
            None => None,
        };
        if let Some(before) = self.read.filter(|before| !settled && before.others != others) {
            let interval = at.at().duration_since(before.at.at());
            self.others_interval = Some(self.others_interval.map_or(interval, |shortest| shortest.min(interval)));
        }
        let paged = reading.paging().is_some();
        let (cap, grows) = (self.boundary.cap(), !paged || self.boundary.admits_growth());
        self.read = Some(Read { at, capacity: reading.capacity(), supply, visible: visible_allocated, nonresident, unobserved, credited, others, rise, grant, ceiling: reading.ceiling(), unwired, lag, shown, cap, grows });
        let estimate = self.estimate().expect("a store just read is estimated");
        let cause = shortage::cause(&estimate, self.boundary.harmed(), self.boundary.pressed());
        // Only a floor shortfall unloads: its persistence alone is kept,
        // so a rise, harm or pressure that stood before it never makes it
        // unload at once.
        self.persistence.observe(at, cause.filter(|cause| matches!(cause, shortage::Cause::FloorShort { .. })));
    }

    /// The cold pages of `reading` that are supply (§4.2): available from
    /// reading #1 whenever host paging is monitored (`harm.is_some()`, e.g.
    /// macOS and dedicated Linux cgroups) so model opens never fail while
    /// gigabytes of cold app pages can be compressed/swapped. Left out (`0`)
    /// when `harm.is_none()` (shared multi-tenant Linux cgroup where other
    /// processes' inactive anon cannot be isolated). Where the OS isolates the
    /// process's own cold pages (`own_cold = Some`), others' alone; where it
    /// does not (`own_cold = None`), subtract only our own unwired idle caches
    /// (`Prefix`, idle `Drafter`, idle `Vision`, `Spare`) bounded by our pageable
    /// allocation, never active `Load`/`State`/`Bank` weights or KV reservations.
    fn cold_supply(&self, reading: &Reading, figures: seismic::Supply) -> u64 {
        let Some(paging) = reading.paging() else {
            return figures.cold_reclaimable;
        };
        if paging.harm.is_none() {
            return 0;
        }
        match paging.own_cold {
            Some(_) => figures.cold_reclaimable.min(paging.cold),
            None => {
                let own_idle = self.cache_of[CacheKind::Prefix.index()]
                    .saturating_add(self.cache_of[CacheKind::Drafter.index()])
                    .saturating_add(self.cache_of[CacheKind::Vision.index()])
                    .saturating_add(self.cache_of[CacheKind::Spare.index()])
                    .min(self.allocated.saturating_sub(self.wired));
                figures.cold_reclaimable.saturating_sub(own_idle)
            }
        }
    }
}
