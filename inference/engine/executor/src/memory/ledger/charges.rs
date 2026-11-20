//! Seismic's allocations: every byte it takes is an entry's, charged through
//! the entry's [`Record`].

use std::num::NonZeroU64;
use std::sync::Arc;

use super::state::{EntryId, Pool, State};
use super::view::Ledger;
use super::vocabulary::{CommittedKind, Holding, PoolClass};
#[cfg(test)] use super::admission::Short;
#[cfg(test)] use super::claims::Cached;
#[cfg(test)] use super::store::Held;
#[cfg(test)] use super::vocabulary::CacheKind;

/// One entry of a ledger, as Seismic charges it ([`seismic::memory::Entry`]):
/// shared by `Arc` between its holder ([`Claim`], [`Cached`]) and the
/// allocations Seismic charges to it. The entry lives until the last of
/// them drops, so no allocation outlives the entry it is charged to, and
/// its bytes are free then.
pub(super) struct Record {
    pub(super) ledger: Ledger,
    pub(super) id: EntryId,
    /// What runs as the holder is gone ([`Holder`]).
    closing: std::sync::Mutex<Closing>,
}

enum Closing {
    Open(Vec<Box<dyn FnOnce() + Send>>),
    Closed,
}

/// The holder's handle on its entry: dropped, the entry is closed, and
/// whatever allocations keep idle for reuse is returned there and then
/// ([`seismic::memory::Entry::on_close`]), inside a change of its own.
/// Allocations charged to the entry hold the [`Record`] itself, so they
/// outlive the holder, and the holder's drop is the one place that knows
/// it is gone. Not `Clone`: one holder.
pub(super) struct Holder(Arc<Record>);

impl Holder {
    pub(super) fn new(ledger: Ledger, id: EntryId) -> Self {
        Self(Arc::new(Record::new(ledger, id)))
    }

    /// The entry, as Seismic charges it.
    pub(super) fn entry(&self) -> Arc<dyn seismic::memory::Entry> {
        self.0.clone()
    }
}

impl std::ops::Deref for Holder {
    type Target = Record;

    fn deref(&self) -> &Record {
        &self.0
    }
}

impl Drop for Holder {
    fn drop(&mut self) {
        let closed = match std::mem::replace(&mut *self.0.closing.lock().expect("an entry's closing lock is never poisoned"), Closing::Closed) {
            Closing::Open(closed) => closed,
            Closing::Closed => unreachable!("an entry's holder drops once"),
        };
        if closed.is_empty() {
            return;
        }
        let _change = seismic::memory::Entry::change(&*self.0);
        closed.into_iter().for_each(|closed| closed());
    }
}

impl Drop for Record {
    /// The entry's bytes are free ([`Ledger::unpoisoned`]).
    fn drop(&mut self) {
        if let Some(mut state) = self.ledger.unpoisoned() {
            state.release(self.id);
        }
    }
}

impl Record {
    pub(super) fn new(ledger: Ledger, id: EntryId) -> Self {
        Self { ledger, id, closing: std::sync::Mutex::new(Closing::Open(Vec::new())) }
    }

    /// `charge`, a record of an OS effect of the process's memory, made
    /// inside the change its effect is made in: every record of an entry
    /// takes that change ([`seismic::memory::Change`], begun only by
    /// [`seismic::memory::Entry::change`]), so none is written outside one.
    fn charge<R>(&self, charge: impl FnOnce(&mut State) -> R) -> R {
        let mut state = self.ledger.state();
        charge(&mut *state)
    }
}

impl seismic::memory::Entry for Record {
    fn store(&self) -> &seismic::StoreIdentity {
        &self.ledger.store.identity
    }

    /// The change its thread has under way, or one begun now, which never
    /// waits on a sample: Seismic may charge holding a lock the sample's
    /// reading takes.
    fn change(&self) -> seismic::memory::Change {
        seismic::memory::Change::held(self.ledger.store.changes.begin(false))
    }

    fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
        let mut closing = self.closing.lock().expect("an entry's closing lock is never poisoned");
        match &mut *closing {
            Closing::Open(waiting) => waiting.push(closed),
            Closing::Closed => {
                drop(closing);
                closed();
            }
        }
    }

    fn closed(&self) -> bool {
        matches!(*self.closing.lock().expect("an entry's closing lock is never poisoned"), Closing::Closed)
    }

    fn take(&self, _change: &seismic::memory::Change, bytes: u64, wiring: seismic::memory::Wiring) -> Result<(), seismic::memory::Refused> {
        self.charge(|state| state.entry_take(&self.ledger.store.identity, self.id, bytes, wiring))
    }

    fn took(&self, _change: &seismic::memory::Change, bytes: u64, wiring: seismic::memory::Wiring) {
        self.charge(|state| state.entry_took(self.id, bytes, wiring));
    }

    /// Recorded as every release is ([`Ledger::unpoisoned`]): a free an
    /// unwind makes (a charge or a carved range dropping) is of bytes a take
    /// recorded, so it never frees more than the entry holds.
    fn freed(&self, _change: &seismic::memory::Change, bytes: u64, wiring: seismic::memory::Wiring) {
        if let Some(mut state) = self.ledger.unpoisoned() {
            state.entry_freed(self.id, bytes, wiring);
        }
    }
}

impl State {
    /// Entry `id`'s holder (Seismic, through [`seismic::memory::Entry::take`]) asks
    /// to allocate `bytes` more: what the entry holds unallocated serves
    /// first, and what it lacks is admitted as a claim of its kind is, cache
    /// yielding to it, or as cache from free room alone. Refused, nothing is
    /// allocated.
    ///
    /// The pool's entry allocates a block: what it adds beyond the granules
    /// pooled holdings hold is spare, held from free room alone.
    ///
    /// Judged at the store's latest reading. Before its first, the store's
    /// device is opening: a device's own memory is read through the opened
    /// device (its grant is the driver's: CUDA's free figure, Metal's
    /// working set), so what it allocates as it opens (Metal's order
    /// buffer, its probe's buffer, CUDA's host word) has no reading to be
    /// judged by, and is recorded as taken ([`Self::entry_took`]), never
    /// refused: the device does not exist without it. Only a device's own
    /// entries exist then: every other entry is a claim of ledgers read
    /// before they claim ([`super::UnreadLedgers`]).
    ///
    /// # Panics
    /// For a pooled holding: its granules are the pool's to allocate. For
    /// an entry other than a device's own in a store never read.
    fn entry_take(&mut self, store: &seismic::StoreIdentity, id: EntryId, bytes: u64, wiring: seismic::memory::Wiring) -> Result<(), seismic::memory::Refused> {
        let entry = *self.entry(id);
        assert!(!entry.holding.pooled(), "invariant violated: {:?} is charged; the pool's entry allocates its granules", entry.holding);
        assert!(
            !matches!(self.reclaim.get(&id), Some(super::admission::Reclaim::Unbacked)),
            "invariant violated: Reclaim::Unbacked cache entry {id} allocates physical memory"
        );
        let Some(at) = self.read else {
            assert!(
                entry.holding == Holding::Committed(CommittedKind::Device),
                "invariant violated: {:?} of {store:?} is charged before the store's first reading; only a device's own entries are",
                entry.holding
            );
            self.entry_took(id, bytes, wiring);
            return Ok(());
        };
        let lacks = match entry.holding {
            Holding::Pool(class) => self.spare_at(class, entry.allocated.checked_add(bytes).expect("a store's bytes fit u64")) - entry.bytes,
            _ => bytes.saturating_sub(entry.bytes - entry.allocated),
        };
        let refused = |state: &Self| seismic::memory::Refused {
            required: bytes,
            available: state.free_for(at, entry.holding, lacks) + state.cache,
        };
        // The one bound on an allocation: the bytes the budget leaves beside
        // what the process has allocated. A claim holding cache it took in
        // process allocates it only once its holder freed it.
        let allocatable = self.estimate_at(at).allocatable();
        if bytes > allocatable {
            return Err(seismic::memory::Refused { required: bytes, available: allocatable });
        }
        match entry.holding {
            _ if lacks == 0 => {}
            Holding::Pool(_) => {
                if self.free_for(at, entry.holding, lacks) < lacks {
                    return Err(refused(self));
                }
            }
            Holding::Committed(kind) => {
                let taken = self.take(at, kind, lacks).map_err(|_| refused(self))?;
                self.merge(taken, id);
            }
            Holding::Cache(_) => {
                if self.free_for(at, entry.holding, lacks) < lacks {
                    return Err(refused(self));
                }
                self.resize(id, entry.bytes + lacks);
            }
        }
        self.allocate(id, entry.allocated + bytes);
        self.wire(id, bytes, wiring);
        Ok(())
    }

    /// `bytes` entry `id`'s holder allocated more are wired, where
    /// `wiring` says so: counted in the entry and the store.
    fn wire(&mut self, id: EntryId, bytes: u64, wiring: seismic::memory::Wiring) {
        if wiring == seismic::memory::Wiring::Wired {
            self.entry(id).wired += bytes;
            self.wired += bytes;
        }
    }

    /// Entry `id`'s holder allocated `bytes` a driver already took (driver
    /// objects, graphs, block slack): never refused, the entry grows by what
    /// it did not hold.
    ///
    /// # Panics
    /// For a pooled holding, as [`Self::entry_take`].
    fn entry_took(&mut self, id: EntryId, bytes: u64, wiring: seismic::memory::Wiring) {
        let entry = *self.entry(id);
        assert!(!entry.holding.pooled(), "invariant violated: {:?} is charged; the pool's entry allocates its granules", entry.holding);
        let allocated = entry.allocated.checked_add(bytes).expect("a store's bytes fit u64");
        if !matches!(entry.holding, Holding::Pool(_)) && allocated > entry.bytes {
            self.resize(id, allocated);
        }
        self.allocate(id, allocated);
        self.wire(id, bytes, wiring);
        self.keep_guarantee(entry.holding, allocated.saturating_sub(entry.bytes), |_| true);
    }

    /// Entry `id`'s holder freed `bytes`: they are neither allocated nor
    /// held any more. What it yielded and still released goes first: a
    /// cache entry frees what a claim or a shortage took of it, then what
    /// it holds. The pool's entry holds only its spare granules, which
    /// follow what it allocated.
    ///
    /// # Panics
    /// For a pooled holding, as [`Self::entry_take`].
    fn entry_freed(&mut self, id: EntryId, bytes: u64, wiring: seismic::memory::Wiring) {
        let entry = *self.entry(id);
        assert!(!entry.holding.pooled(), "invariant violated: {:?} is freed; the pool's entry frees its granules", entry.holding);
        if wiring == seismic::memory::Wiring::Wired {
            let wired = entry.wired.checked_sub(bytes).unwrap_or_else(|| panic!("invariant violated: an entry frees {bytes} wired bytes of its {} wired", entry.wired));
            self.entry(id).wired = wired;
            self.wired -= bytes;
        }
        let released = bytes.min(entry.releasing);
        let held = bytes - released;
        let allocated = entry
            .allocated
            .checked_sub(held)
            .unwrap_or_else(|| panic!("invariant violated: an entry frees {bytes} of its {} allocated and {} releasing bytes", entry.allocated, entry.releasing));
        self.entry(id).releasing -= released;
        self.released(released);
        self.allocated -= released;
        self.allocate(id, allocated);
        // A pool's entry holds spare granules that follow `allocated`; an
        // import's upload window (`CommittedKind::Upload`) is a reusable peak
        // reservation across batches, released whole when its holder drops.
        if !matches!(entry.holding, Holding::Pool(_) | Holding::Committed(CommittedKind::Upload)) {
            self.resize(id, entry.bytes - held);
        }
    }
}

impl Ledger {
    /// A committed entry of `kind` opened at none, which grows as Seismic
    /// allocates through it, each allocation its own change
    /// ([`Record`]'s), and is released as it drops: a device's own, a
    /// measurement job's.
    pub(super) fn growing(&self, kind: CommittedKind) -> Record {
        let id = self.state().insert(Holding::Committed(kind), 0);
        Record::new(self.clone(), id)
    }

    /// The store's granule pool of `class` opened with blocks of `block`
    /// bytes, the unit a shrink returns to the OS, which a pressure event
    /// yields: its entries. The blocks' entry, which the backend charges
    /// each block it allocates for the pool, once ([`Holding::Pool`]), so
    /// what it holds beyond its class's pooled holdings is free granules a
    /// claim takes (and their blocks' rounding); and the absent granule's,
    /// committed ([`CommittedKind::Absent`]), which the pool allocates as
    /// it opens and no claim takes. The pool closes as the last of them
    /// drops.
    ///
    /// # Panics
    /// Where the store's pool of `class` is open already: one pool of each
    /// class per store.
    pub fn pool_block(&self, class: PoolClass, block: NonZeroU64) -> super::store::PoolEntries {
        let mut state = self.state();
        assert!(state.pools[class.index()].is_none(), "invariant violated: {:?} opens a second {class:?} pool", self.store.identity);
        let entry = state.insert(Holding::Pool(class), 0);
        state.pools[class.index()] = Some(Pool { entry, block, reclaimer: None });
        let absent = state.insert(Holding::Committed(CommittedKind::Absent(class)), 0);
        drop(state);
        super::store::PoolEntries { blocks: Arc::new(Record::new(self.clone(), entry)), absent: Arc::new(Record::new(self.clone(), absent)) }
    }

    /// Like [`Self::pool_block`], but using an already-claimed `absent`
    /// entry (`CommittedKind::Absent(class)`) so opening the pool allocates
    /// into its pre-claimed bytes at the load's reading.
    pub(super) fn pool_block_with_absent(
        &self,
        class: PoolClass,
        block: NonZeroU64,
        absent: super::claims::Claim,
    ) -> super::store::PoolEntries {
        assert_eq!(absent.kind(), CommittedKind::Absent(class), "invariant violated: pool absent claim has wrong kind");
        let mut state = self.state();
        assert!(state.pools[class.index()].is_none(), "invariant violated: {:?} opens a second {class:?} pool", self.store.identity);
        let entry = state.insert(Holding::Pool(class), 0);
        state.pools[class.index()] = Some(Pool { entry, block, reclaimer: None });
        drop(state);
        super::store::PoolEntries { blocks: Arc::new(Record::new(self.clone(), entry)), absent: absent.entry() }
    }
}

impl seismic::memory::Store for Ledger {
    fn identity(&self) -> &seismic::StoreIdentity {
        &self.store.identity
    }

    /// A device's own memory in the store (staging, contexts, streams): a
    /// committed [`CommittedKind::Device`] entry, which grows as Seismic
    /// allocates through it and lives while the device holds it.
    fn own(&self) -> Arc<dyn seismic::memory::Entry> {
        Arc::new(self.growing(CommittedKind::Device))
    }

    fn allocated(&self) -> u64 {
        self.state().allocated
    }

    /// What a claim may take now ([`Estimate::room`]).
    fn room(&self) -> u64 {
        self.estimate().room()
    }

    fn high_water(&self) -> u64 {
        self.state().high_water
    }

    fn restart_high_water(&self) {
        let mut state = self.state();
        state.high_water = state.allocated;
    }
}

#[cfg(test)]
struct TestReclaimer {
    store: std::sync::Weak<super::state::Store>,
    id: std::sync::Mutex<Option<EntryId>>,
}

#[cfg(test)]
impl super::CacheReclaimer for TestReclaimer {
    fn reclaim(&self, need: u64) -> Result<u64, super::ReclaimError> {
        let Some(store) = self.store.upgrade() else {
            return Err(super::ReclaimError::AtFloor);
        };
        let Some(id) = *self.id.lock().expect("test reclaimer id") else {
            return Err(super::ReclaimError::AtFloor);
        };
        let mut state = store.state.lock().expect("ledger lock poisoned");
        let Some(&entry) = state.held.get(&id) else {
            return Err(super::ReclaimError::AtFloor);
        };
        let free = entry.bytes.min(need);
        if free == 0 {
            return Err(super::ReclaimError::AtFloor);
        }
        if entry.holding.pooled() {
            let class = entry.holding.pool_class().expect("pooled cache has a pool class");
            let pool_id = state.pools[class.index()].as_ref().expect("pooled cache has open pool").entry;
            state.resize(id, entry.bytes - free);
            state.entry_freed(pool_id, free, seismic::memory::Wiring::Pageable);
        } else {
            state.entry_freed(id, free, seismic::memory::Wiring::Pageable);
        }
        Ok(free)
    }
}

/// A test's allocations, each made as a change of its own.
#[cfg(test)]
impl Ledger {
    /// [`Self::cache`] of memory a test made with an explicit [`super::Reclaim`]
    /// policy, charged as Seismic charges it.
    pub(crate) fn cached_with(&self, kind: CacheKind, bytes: u64, reclaim: super::Reclaim) -> Result<Cached, Short> {
        let cached = self.cache(kind, bytes, reclaim)?;
        let charged = self.change(|_| match Holding::Cache(kind).pooled() {
            true => {
                let mut state = self.state();
                let class = Holding::Cache(kind).pool_class().expect("pooled cache has a pool class");
                let pool = state.pools[class.index()].as_ref().expect("a test's pooled cache lies in its class's open pool").entry;
                state.entry_take(&self.store.identity, pool, bytes, seismic::memory::Wiring::Pageable)
            }
            false => cached.entry().take(&cached.entry().change(), bytes, seismic::memory::Wiring::Pageable),
        });
        charged.expect("a cache holding's own bytes are admitted");
        Ok(cached)
    }

    /// [`Self::cache`] of memory a test made, backed by a synchronous test
    /// [`super::CacheReclaimer`] that physically frees the entry's bytes on
    /// reclaim, charged as Seismic charges it.
    pub(crate) fn cached(&self, kind: CacheKind, bytes: u64) -> Result<Cached, Short> {
        let reclaimer = Arc::new(TestReclaimer {
            store: Arc::downgrade(&self.store),
            id: std::sync::Mutex::new(None),
        });
        let mut cached = self.cached_with(kind, bytes, super::Reclaim::Backed(Arc::downgrade(&reclaimer) as std::sync::Weak<dyn super::CacheReclaimer>))?;
        *reclaimer.id.lock().expect("test reclaimer id") = Some(cached.id());
        cached._reclaimer = Some(reclaimer);
        Ok(cached)
    }
}

#[cfg(test)]
impl super::claims::Claim {
    /// Turns this claim into a test [`Cached`] entry backed by a synchronous
    /// [`TestReclaimer`] that physically frees the entry's bytes on reclaim.
    pub(crate) fn into_test_cache(self, kind: CacheKind) -> Cached {
        let reclaimer = Arc::new(TestReclaimer {
            store: Arc::downgrade(&self.ledger().store),
            id: std::sync::Mutex::new(None),
        });
        let mut cached = self.into_cache(kind, super::Reclaim::Backed(Arc::downgrade(&reclaimer) as std::sync::Weak<dyn super::CacheReclaimer>));
        *reclaimer.id.lock().expect("test reclaimer id") = Some(cached.id());
        cached._reclaimer = Some(reclaimer);
        cached
    }
}

#[cfg(test)]
impl Held {
    /// An allocation a test made of every byte claimed, charged to the
    /// claims as Seismic charges one, in one change: pageable, as host
    /// memory no device wires is.
    pub(crate) fn made(&self) {
        let Some(changes) = self.claims().first().map(|claim| claim.ledger().store.changes.clone()) else {
            return;
        };
        changes.change(|_| {
            for claim in self.claims() {
                claim.entry().take(&claim.entry().change(), claim.bytes(), seismic::memory::Wiring::Pageable).expect("a claim's own bytes are admitted");
            }
        });
    }
}
