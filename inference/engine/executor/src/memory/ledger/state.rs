//! A store's record: its entries and their totals, and the hold a view takes of
//! it.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::memory::journal::{Journal, Own};
use crate::memory::shortage::Persistence;
use crate::memory::supply::Boundary;
use crate::operation::RequestId;

use super::changes::{Changes, Sampled};
use super::estimate::Read;
use super::pressure;
use super::view::Ledger;
use super::vocabulary::{CacheKind, CommittedKind, Entry, Holding, PoolClass, Value};

/// One store's record, and the changes of its scope.
pub(super) struct Store {
    pub(super) identity: seismic::StoreIdentity,
    pub(super) state: Mutex<State>,
    pub(super) changes: Arc<Changes>,
    /// The OS's pressure events in the store ([`Ledger::watch_pressure`]).
    pub(super) pressure: Arc<pressure::Pressure>,
    /// Where the footprints of grants' processes are read
    /// ([`super::Footprints`]).
    pub(super) footprints: Arc<dyn super::Footprints>,
}

/// Which entry of a store.
pub(super) type EntryId = u64;

/// A store's entries and its latest reading. The totals are the entries'
/// sums, kept with them.
#[derive(Default)]
pub(super) struct State {
    pub(super) held: BTreeMap<EntryId, Entry>,
    next: EntryId,
    pub(super) committed: u64,
    pub(super) cache: u64,
    /// What entries' holders allocated of them, and what cache that
    /// yielded still releases: the process's memory a reading shows.
    pub(super) allocated: u64,
    /// Of `allocated`, what the OS wires while the process holds it
    /// (`seismic::memory::Wiring::Wired`): what a wired ceiling's count
    /// shows of the process (§4.3).
    pub(super) wired: u64,
    /// The most `allocated` has been since the last restart
    /// ([`seismic::memory::Store::restart_high_water`]).
    pub(super) high_water: u64,
    /// What cache yielded to claims in process and its holders have not
    /// freed yet ([`Entry::releasing`]).
    pub(super) releasing: u64,
    /// Cache by kind ([`CacheKind::ALL`]'s order).
    pub(super) cache_of: [u64; CacheKind::ALL.len()],
    /// Each request's reservation, by admission.
    pub(super) state_of: BTreeMap<RequestId, u64>,
    /// The reservations that drew on a request slot, each with the slot's
    /// entry and the bytes it drew ([`Ledger::reserve`], [`Ledger::extend`]),
    /// which go back to the slot as it drops.
    pub(super) lent: HashMap<EntryId, (EntryId, u64)>,
    /// The grants, and the process each one's growth is credited from
    /// ([`Grant`]).
    pub(super) grants: BTreeMap<EntryId, super::grant::Granted>,
    /// The process's own changes no sample shows yet.
    pub(super) journal: Journal,
    /// The latest reading, as the ledger estimated it; none before the
    /// first.
    pub(super) read: Option<Read>,
    /// Others' wired bytes at the latest reading, where the store has a
    /// wired ceiling.
    pub(super) wired_others: Option<u64>,
    /// Where others' working set lies, from the store's paging signals.
    pub(super) boundary: Boundary,
    /// The store's shortage across readings.
    pub(super) persistence: Persistence,
    /// The store's granule pools, each once it opens, by class
    /// ([`PoolClass::index`], [`Ledger::pool_block`]).
    pub(super) pools: [Option<Pool>; 2],
    /// What pooled holdings of each class hold ([`Holding::pool_class`]),
    /// summed.
    pooled: [u64; 2],
    /// Moves on every change of the entries or a value
    /// ([`Ledger::holdings_generation`]).
    pub(super) holdings: u64,
    /// The readings folded in ([`Ledger::generation`]).
    pub(super) readings: u64,
    /// A claim was refused since the latest reading: it waits for the next
    /// ([`Standing::Wait`]).
    pub(super) waiting: bool,
    /// Of what is releasing, what claims took in process
    /// ([`Self::yields`]): held by them already. The rest was shed to a
    /// shortage, held by no entry and allocated still, so it is not free
    /// ([`Estimate::free`]).
    pub(super) taken: u64,
    /// The latest reading a shortage was answered at
    /// ([`Ledger::shortage`]): each reading is answered once.
    pub(super) answered: Option<Sampled>,
    /// The shortest interval between two readings across which others'
    /// memory moved ([`Ledger::others_interval`]).
    pub(super) others_interval: Option<Duration>,
    /// The readings at which some of the process's bytes were neither in
    /// what the store used nor reported outside RAM by the OS
    /// ([`Estimate::unobserved_resident`]).
    pub(super) unobserved_readings: u64,
    /// The readings at which the process's bytes the store's use does not
    /// hold exceeded what an exact OS count of its bytes out of RAM allows:
    /// an accounting error ([`Estimate::accounting_violations`]).
    pub(super) accounting_violations: u64,
    /// The readings at which the wire count showed less than the process
    /// wires ([`Estimate::unwired_shortfall`]).
    pub(super) unwired_readings: u64,
    /// Reclamation strategy for every cache and pool entry, keyed by `EntryId`,
    /// set at entry creation and removed on [`Self::release`].
    pub(super) reclaim: HashMap<EntryId, super::admission::Reclaim>,
}

/// One of a store's granule pools: its entry, which every block is charged
/// to, its block, the unit a shrink returns to the OS, and its optional
/// reclaimer owned for the pool's lifetime (holding only a weak pool handle).
pub(super) struct Pool {
    pub(super) entry: EntryId,
    pub(super) block: NonZeroU64,
    pub(super) reclaimer: Option<Arc<dyn super::admission::CacheReclaimer>>,
}

/// A view's hold of its store's state: every change of what the process
/// allocated that one hold makes is journaled, net, as it ends
/// (`memory::journal`), so no path that allocates or frees escapes the
/// journal: in the change its thread has under way, or, a holder's drop
/// letting its bytes go, in one of its own.
pub(super) struct Locked<'a> {
    /// Held until the hold drops, which lets it go before any panic.
    state: Option<MutexGuard<'a, State>>,
    ledger: &'a Ledger,
    /// The process's own figures as the hold began.
    own: Own,
}

impl std::ops::Deref for Locked<'_> {
    type Target = State;

    fn deref(&self) -> &State {
        self.state.as_ref().expect("a hold holds its state until it drops")
    }
}

impl std::ops::DerefMut for Locked<'_> {
    fn deref_mut(&mut self) -> &mut State {
        self.state.as_mut().expect("a hold holds its state until it drops")
    }
}

impl Drop for Locked<'_> {
    /// A record outside any change is journaled as one of its own, the
    /// state let go, and then panics: the ledger stays whole and
    /// unpoisoned, so the violation reports itself. Where a panic under
    /// this hold poisoned the ledger as the state was let go, that panic is
    /// the report.
    fn drop(&mut self) {
        let mut state = self.state.take().expect("a hold holds its state until it drops");
        let (before, now) = (self.own, state.own());
        if before == now {
            return;
        }
        let changes = &self.ledger.store.changes;
        let open = changes.open_here();
        let number = open.unwrap_or_else(|| changes.begin(false).number);
        state.journal.record(number, changes.now(), before, now);
        drop(state);
        if self.ledger.store.state.is_poisoned() {
            return;
        }
        assert!(
            open.is_some() || changes.alone(),
            "invariant violated: an allocation change recorded outside a change of its memory"
        );
    }
}

impl Ledger {
    pub(super) fn state(&self) -> Locked<'_> {
        self.hold(self.store.state.lock().expect("ledger lock poisoned"))
    }

    /// The state a release records in: every release is recorded, an
    /// unwind's included, so a domain that panics leaves none of its bytes
    /// held in the process's ledger. None where a panic under the lock
    /// poisoned it: that panic is the report, and the release is not
    /// recorded in a state the panic left half made. An invariant a release
    /// finds violated panics, unwinding or not: during an unwind that
    /// aborts the process, the one end for it, since the ledger outlives
    /// the domain whose release found it wrong and would admit every later
    /// load by it.
    pub(super) fn unpoisoned(&self) -> Option<Locked<'_>> {
        self.store.state.lock().ok().map(|state| self.hold(state))
    }

    fn hold<'a>(&'a self, state: MutexGuard<'a, State>) -> Locked<'a> {
        let own = state.own();
        Locked { state: Some(state), ledger: self, own }
    }
}

impl State {
    /// The process's own figures in the store now.
    pub(super) fn own(&self) -> Own {
        Own { allocated: self.allocated, wired: self.wired }
    }

    /// What one pressure event lowers the boundary by: the largest shrink unit
    /// of the cache held in this store — the largest open pool's block where a
    /// pool is open, or the largest non-pooled cache entry otherwise — capped
    /// at the cache held.
    ///
    /// # Panics
    /// Where pooled cache is held in a store whose pool of that class is not
    /// open.
    pub(super) fn unit(&self) -> u64 {
        for entry in self.held.values() {
            if entry.bytes > 0 {
                if let Some(class) = entry.holding.pool_class().filter(|_| entry.holding.pooled()) {
                    assert!(
                        self.pools[class.index()].is_some(),
                        "invariant violated: {:?} ({} bytes) held in a store whose {class:?} pool never opened",
                        entry.holding,
                        entry.bytes
                    );
                }
            }
        }
        let unit = self
            .pools
            .iter()
            .flatten()
            .map(|pool| pool.block.get())
            .max()
            .or_else(|| {
                self.held
                    .values()
                    .filter(|entry| matches!(entry.holding, Holding::Cache(_)) && !entry.holding.pooled())
                    .map(|entry| entry.bytes)
                    .max()
            })
            .unwrap_or(0);
        unit.min(self.cache)
    }

    /// What pooled holdings of `class` hold: the granules the class's
    /// pool's allocation serves.
    fn pooled_held(&self, class: PoolClass) -> u64 {
        self.pooled[class.index()]
    }

    /// The spare granules `class`'s pool allocating `allocated` holds: what
    /// no pooled holding of its class holds of it.
    pub(super) fn spare_at(&self, class: PoolClass, allocated: u64) -> u64 {
        allocated.saturating_sub(self.pooled_held(class))
    }

    /// The free granules of the pool of `class`: its entry's bytes, none
    /// before it opens.
    pub(super) fn spare_of(&self, class: PoolClass) -> u64 {
        self.pools[class.index()].as_ref().map_or(0, |pool| self.held[&pool.entry].bytes)
    }

    /// Keep `class`'s pool's entry holding its spare granules: what it
    /// allocated that no pooled holding of its class holds. Called on every
    /// change of either.
    fn repool(&mut self, class: PoolClass) {
        let Some(entry_id) = self.pools[class.index()].as_ref().map(|pool| pool.entry) else {
            return;
        };
        let entry = self.held[&entry_id];
        let spare = self.spare_at(class, entry.allocated);
        if spare != entry.bytes {
            self.count(Holding::Pool(class), entry.bytes, false);
            self.count(Holding::Pool(class), spare, true);
            self.entry(entry_id).bytes = spare;
        }
    }

    /// `bytes` of the pool's granules a yield let go of: allocated still,
    /// held no more, until the pool frees them (`entry_freed`), taken by a
    /// claim in process or shed to a shortage. Only what the pool allocated
    /// releases; none without a pool of `class`.
    fn release_pooled(&mut self, class: PoolClass, bytes: u64, taken: bool) {
        let Some(entry_id) = self.pools[class.index()].as_ref().map(|pool| pool.entry) else {
            return;
        };
        let entry = self.entry(entry_id);
        let released = bytes.min(entry.allocated);
        entry.allocated -= released;
        entry.releasing += released;
        self.releasing += released;
        if taken {
            self.taken += released;
        }
        self.repool(class);
    }

    /// `bytes` that were releasing are freed or let go: what claims took is
    /// freed first, as they allocate it only once freed.
    pub(super) fn released(&mut self, bytes: u64) {
        self.releasing -= bytes;
        self.taken -= bytes.min(self.taken);
    }

    /// What `sequence`'s state is worth per byte: its reservation's
    /// ([`Reservation::value`]), none where its holder never valued it.
    pub(super) fn value_of(&self, sequence: RequestId) -> Value {
        self.held
            .values()
            .filter(|entry| entry.holding == Holding::Committed(CommittedKind::Request(sequence)))
            .map(|entry| entry.value)
            .min()
            .unwrap_or(Value::NONE)
    }

    /// Add `bytes` to the totals of `holding`, or take them away.
    fn count(&mut self, holding: Holding, bytes: u64, add: bool) {
        self.holdings += 1;
        let apply = |total: &mut u64| {
            *total = if add {
                total.checked_add(bytes).expect("a store's bytes fit u64")
            } else {
                total.checked_sub(bytes).expect("a total holds its entries")
            }
        };
        match holding {
            Holding::Committed(kind) => {
                apply(&mut self.committed);
                if let CommittedKind::Request(sequence) = kind {
                    let held = self.state_of.entry(sequence).or_default();
                    apply(held);
                    if *held == 0 {
                        self.state_of.remove(&sequence);
                    }
                }
            }
            Holding::Cache(kind) => {
                apply(&mut self.cache);
                apply(&mut self.cache_of[kind.index()]);
            }
            Holding::Pool(_) => {
                apply(&mut self.cache);
                apply(&mut self.cache_of[CacheKind::Spare.index()]);
            }
        }
        if let Some(class) = holding.pool_class().filter(|_| holding.pooled()) {
            apply(&mut self.pooled[class.index()]);
            self.repool(class);
        }
    }

    /// A new entry of `bytes`, none of them allocated yet.
    pub(super) fn insert(&mut self, holding: Holding, bytes: u64) -> EntryId {
        let id = self.next;
        self.next += 1;
        self.held.insert(id, Entry { holding, bytes, allocated: 0, wired: 0, releasing: 0, claimed: 0, value: Value::NONE });
        self.count(holding, bytes, true);
        id
    }

    /// Entry `id`'s holder has `allocated` of its bytes allocated now, the
    /// store's total with it.
    pub(super) fn allocate(&mut self, id: EntryId, allocated: u64) {
        self.holdings += 1;
        let entry = *self.entry(id);
        match entry.holding {
            Holding::Pool(_) => {}
            holding => {
                assert!(
                    !holding.pooled() || allocated == 0,
                    "invariant violated: {holding:?} allocates its own bytes; the pool's entry allocates its granules"
                );
                assert!(allocated <= entry.bytes, "invariant violated: an entry's holder allocates {allocated} of its {} bytes", entry.bytes);
            }
        }
        assert!(
            entry.wired <= allocated + entry.releasing,
            "invariant violated: an entry allocates {allocated} bytes beside {} releasing, short of its {} wired",
            entry.releasing,
            entry.wired
        );
        self.allocated = (self.allocated - entry.allocated).checked_add(allocated).expect("a store's bytes fit u64");
        self.high_water = self.high_water.max(self.allocated);
        self.entry(id).allocated = allocated;
        if let Holding::Pool(class) = entry.holding {
            self.repool(class);
        }
    }

    pub(super) fn entry(&mut self, id: EntryId) -> &mut Entry {
        self.held.get_mut(&id).expect("a holder's entry is in its ledger")
    }

    /// Cache entry `id` yields `bytes` to a claim in process, or to a
    /// shortage. Pooled cache yielding to a pooled claim (`within`) moves
    /// only what is held: the granules stay the pool's allocation, and the
    /// claim holds them next. Otherwise the bytes are held no more and
    /// stay allocated, releasing, until Seismic frees them through the
    /// entry that allocated them (`entry_freed`): the pool's, for pooled
    /// cache. So the reading counts them the process's, and nothing
    /// allocates them twice: a claim allocates its bytes only as these are
    /// freed ([`Estimate::allocatable`]).
    pub(super) fn yields(&mut self, id: EntryId, bytes: u64, within: bool, taken: bool) {
        let holding = self.entry(id).holding;
        match holding {
            // Spare granules are what no holding holds: the claim's own
            // bytes take them as it is held.
            Holding::Pool(_) if within => return,
            Holding::Pool(class) => return self.release_pooled(class, bytes, taken),
            Holding::Cache(_) if holding.pooled() => {
                let has = self.entry(id).bytes;
                self.resize(id, has - bytes);
                if taken && !within {
                    self.entry(id).claimed += bytes;
                }
                if let (false, Some(class)) = (within, holding.pool_class()) {
                    self.release_pooled(class, bytes, taken);
                }
                return;
            }
            Holding::Cache(_) => {}
            Holding::Committed(_) => unreachable!("only cache yields to a claim"),
        }
        let entry = self.entry(id);
        entry.bytes -= bytes;
        entry.allocated -= bytes;
        entry.releasing += bytes;
        let holding = entry.holding;
        self.count(holding, bytes, false);
        self.releasing += bytes;
        if taken {
            self.taken += bytes;
        }
    }

    /// Resize entry `id` to `bytes`, its totals with it, what its holder
    /// allocated past them freed.
    pub(super) fn resize(&mut self, id: EntryId, bytes: u64) {
        let entry = *self.entry(id);
        if entry.allocated > bytes {
            self.allocate(id, bytes);
        }
        self.count(entry.holding, entry.bytes, false);
        self.count(entry.holding, bytes, true);
        self.entry(id).bytes = bytes;
    }

    /// Entry `from`'s bytes, and what of them is allocated, join committed
    /// entry `into` of the same kind.
    pub(super) fn merge(&mut self, from: EntryId, into: EntryId) {
        let from = self.held.remove(&from).expect("a taken entry is in its ledger");
        let into = self.entry(into);
        into.bytes += from.bytes;
        into.allocated += from.allocated;
        into.wired += from.wired;
    }

    /// Entry `id` turns `holding`, its bytes kept.
    pub(super) fn turn(&mut self, id: EntryId, holding: Holding) {
        let entry = *self.entry(id);
        assert_eq!(
            (entry.holding.pooled(), entry.holding.pool_class()),
            (holding.pooled(), holding.pool_class()),
            "invariant violated: {:?} turns {holding:?}, in or out of its pool",
            entry.holding
        );
        self.count(entry.holding, entry.bytes, false);
        self.count(holding, entry.bytes, true);
        self.entry(id).holding = holding;
    }

    /// Entry `id` was let go: its bytes are free.
    pub(super) fn release(&mut self, id: EntryId) {
        for pool in &mut self.pools {
            if pool.as_ref().is_some_and(|pool| pool.entry == id) {
                *pool = None;
            }
        }
        self.reclaim.remove(&id);
        let entry = self.held.remove(&id).expect("a holder's entry is in its ledger");
        self.allocated -= entry.allocated + entry.releasing;
        self.wired -= entry.wired;
        self.released(entry.releasing);
        self.grants.remove(&id);
        self.count(entry.holding, entry.bytes, false);
        // A reservation gives what it drew of a request slot back to it;
        // the granules stay the pool's, so nothing allocated moves.
        if let Some((slot, drawn)) = self.lent.remove(&id).filter(|(slot, _)| self.held.contains_key(slot)) {
            let has = self.held[&slot].bytes;
            self.resize(slot, has + drawn);
        }
    }

    /// What `holding` may hold anew beside what is held: free room, and for
    /// a pooled holding the spare granules of its class's pool too, which it
    /// holds in place.
    pub(super) fn room_for(&self, store: &seismic::StoreIdentity, holding: Holding, bytes: u64) -> u64 {
        let pool = holding.pool_class().filter(|_| holding.pooled()).and_then(|class| self.pools[class.index()].as_ref());
        let spare = pool.map_or(0, |pool| self.held[&pool.entry].bytes);
        self.free_for(self.reading(store, None), holding, bytes).saturating_add(spare)
    }
}
