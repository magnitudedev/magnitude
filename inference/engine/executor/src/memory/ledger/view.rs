//! One load's view of one store's ledger: its readings, standing, shortage and
//! admission queries.

use std::sync::Arc;
use std::time::Duration;

use crate::memory::shortage::{self, Shortage};
use crate::memory::supply::Reading;

use super::admission::{Admit, Short};
use super::changes::{Keeping, Sampled};
use super::estimate::Estimate;
use super::state::Store;
use super::vocabulary::{CacheKind, CommittedKind, Entry, Holding, LoadShape, Need, PoolClass, size};

/// One load's view of one store's ledger: the store's one record, and the
/// load's shape its needs are sized from ([`size`]). Clones share both.
#[derive(Clone)]
pub struct Ledger {
    pub(super) store: Arc<Store>,
    pub(super) shape: Arc<LoadShape>,
}

/// What a store's ledger holds allocated, by the class of its holdings
/// ([`Ledger::charge_classes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChargeClasses {
    /// The load's weights and the expert bank's records, their pool's
    /// spare included.
    pub model: u64,
    /// Sequences' state, the state floor and retained prefixes: the state's
    /// pool, its spare included, and the state stores' address tables.
    pub context: u64,
    /// Everything no other class holds: the arena, graph slots, group
    /// classes, scratch, outputs and the device's own memory.
    pub compute: u64,
    /// Optional components (a draft head, a vision encoder), resident or
    /// idle.
    pub auxiliary: u64,
}

impl ChargeClasses {
    /// What the classes hold together.
    pub fn allocated(&self) -> u64 {
        [self.model, self.context, self.compute, self.auxiliary]
            .into_iter()
            .try_fold(0u64, u64::checked_add)
            .expect("a store's bytes fit u64")
    }
}

/// What stands in a store that only a later reading can answer
/// ([`Ledger::standing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Standing {
    /// A real shortage (`memory::shortage`): whether it persists past λ,
    /// or clears, only a later reading shows.
    Shortage(shortage::Cause),
    /// A claim was refused since the latest reading: it waits for memory a
    /// holder frees, which only a later reading shows.
    Wait,
    /// A growth step's window λ_h is open (`memory::supply`): whether the
    /// growth harmed others, or closes, only a later reading shows.
    Step,
    /// The boundary B stands (`memory::supply`): only others' release lifts
    /// it. The bank's deficit against its own saturation is the domain's to
    /// judge, from its derived targets: the ledger holds no miss curve.
    Deficit,
}

impl Ledger {
    /// `change`, an OS change of the process's memory and its records in
    /// the ledgers, run as one change ([`Changes::change`]), handed in its
    /// keep phase.
    pub fn change<R>(&self, change: impl FnOnce(Keeping<'_>) -> R) -> R {
        self.store.changes.change(change)
    }

    pub fn store(&self) -> seismic::StoreIdentity {
        self.store.identity.clone()
    }

    /// Fold in the store's `reading`, sampled at `at` (§4.1-4.3): paired
    /// with what the process had allocated as the sample shows it (the
    /// journal's `allocated_visible`), so others' memory is what the store
    /// uses beyond that and its supply, and no change of the process's moves
    /// it. A reading sampled before the latest is not folded in, so
    /// readings never mix. Every change the sample shows leaves the journal.
    /// The estimate the store stands at once it is folded in, taken under
    /// the same hold: this reading's, or where a later one was folded in
    /// first (another load's, a pressure event's), that one's, which is
    /// newer still.
    pub fn read(&self, at: Sampled, reading: Reading) -> Estimate {
        self.read_after(at, reading, false)
    }

    /// [`Self::read`], `after_idle` marking the first unpressured reading
    /// after an unmonitored idle sleep so what moved while asleep opens a
    /// fresh baseline rather than counting as one active step's rise.
    pub fn read_after(&self, at: Sampled, reading: Reading, after_idle: bool) -> Estimate {
        let footprints = self.footprints();
        let mut state = self.state();
        if !state.read.is_some_and(|latest| !latest.at.precedes(&at)) {
            state.read(&self.store.identity, at, reading, self.store.pressure.take(), after_idle, &footprints);
        }
        state.judged(&self.store.identity)
    }

    /// The store's estimate now.
    ///
    /// # Panics
    /// Before the store's first reading: every store is read before a load
    /// claims in it.
    pub fn estimate(&self) -> Estimate {
        self.state().judged(&self.store.identity)
    }

    /// What stands in the store at its latest reading that only a later
    /// reading can answer (design §4.8); none where nothing does, or before
    /// the first reading. While one stands, the owner reads the store again
    /// on its doubling interval; once none stands it reads only at events.
    /// In order: a real shortage, a claim refused since the reading, then a
    /// growth the boundary holds back.
    pub fn standing(&self) -> Option<Standing> {
        let state = self.state();
        let estimate = state.estimate()?;
        if let Some(cause) = shortage::cause(&estimate, state.boundary.harmed(), state.boundary.pressed()) {
            return Some(Standing::Shortage(cause));
        }
        if state.waiting {
            return Some(Standing::Wait);
        }
        if state.boundary.step_open() {
            return Some(Standing::Step);
        }
        state.boundary.cap().is_some().then_some(Standing::Deficit)
    }

    /// One pressure event the OS delivered (`seismic::PressureEvents`: the
    /// macOS memorypressure source, a Linux PSI trigger, Windows' low-memory
    /// notification), as the store's one watch counts it
    /// ([`Self::watch_pressure`]): taken once, by the store's next reading,
    /// it lowers the boundary by one block where the level is set
    /// (`memory::supply`), and every watching load is woken.
    #[cfg(test)]
    pub(crate) fn pressure_event(&self) {
        self.store.pressure.event_now();
    }

    /// The store's generation: it moves on every change of what the
    /// process holds ([`Self::holdings_generation`]) and on every reading,
    /// and on nothing else. Equal generations read equal estimates.
    pub fn generation(&self) -> u64 {
        let state = self.state();
        state.holdings + state.readings
    }

    /// The generation of what the process holds in the store: it moves on
    /// every change of an entry (a claim, a cache holding, an allocation, a
    /// free, a hand-over) or of a value, and
    /// never on a reading. Equal holdings generations hold the same entries,
    /// so a census of them recomputed only when it moves misses nothing, and
    /// steady decode's readings never recompute it.
    pub fn holdings_generation(&self) -> u64 {
        self.state().holdings
    }

    /// Readings folded into the store so far.
    pub fn readings(&self) -> u64 {
        self.state().readings
    }

    /// Bytes the process currently wires in the store.
    pub fn wired(&self) -> u64 {
        self.state().wired
    }

    /// The shortest interval measured so far between two readings of the
    /// store across which others' memory moved, kept per store from its
    /// readings alone: how soon another program's change has been seen to
    /// follow the reading before it. None until one has been seen.
    pub fn others_interval(&self) -> Option<Duration> {
        self.state().others_interval
    }

    /// Every entry, for telemetry.
    pub fn entries(&self) -> Vec<Entry> {
        self.state().held.values().copied().collect()
    }

    /// What the store's ledger holds allocated, by class, read in one hold
    /// of its state: each entry's allocated and releasing bytes in its
    /// holding's class. A pooled holding allocates none of its own; each
    /// pool's entry allocates every granule of its class, its spare (free
    /// granules, its blocks' rounding) included, and its absent granule's
    /// entry the absent granule.
    ///
    /// # Panics
    /// Where a pooled holding allocates, or the classes sum to other than
    /// what the store has allocated: the ledger's invariants.
    pub fn charge_classes(&self) -> ChargeClasses {
        let state = self.state();
        let mut classes = ChargeClasses::default();
        for entry in state.held.values() {
            let bytes = entry.allocated.checked_add(entry.releasing).expect("a store's bytes fit u64");
            let class = match entry.holding {
                holding if holding.pooled() => {
                    assert_eq!(bytes, 0, "invariant violated: pooled {holding:?} allocates its own bytes");
                    continue;
                }
                Holding::Pool(PoolClass::State) | Holding::Committed(CommittedKind::Absent(PoolClass::State)) => &mut classes.context,
                Holding::Pool(PoolClass::Records) | Holding::Committed(CommittedKind::Absent(PoolClass::Records)) => &mut classes.model,
                Holding::Committed(CommittedKind::Load) => &mut classes.model,
                Holding::Committed(CommittedKind::StateTables) => &mut classes.context,
                Holding::Committed(CommittedKind::Component) | Holding::Cache(CacheKind::Drafter | CacheKind::Vision) => &mut classes.auxiliary,
                Holding::Committed(_) | Holding::Cache(_) => &mut classes.compute,
            };
            *class = class.checked_add(bytes).expect("a store's bytes fit u64");
        }
        assert_eq!(
            classes.allocated(),
            state.allocated,
            "invariant violated: {:?}'s entries allocate other than its total",
            self.store.identity
        );
        classes
    }

    /// The store's real shortage at its latest reading and the response to
    /// it (§4.5, `memory::shortage`), once per reading: none where it has
    /// none, was never read, or this reading was answered already. Every
    /// response is to new evidence, so asking at every event never repeats
    /// one; whether a shortage still stands between readings is
    /// [`Self::standing`]'s. Its relief preempts none but `evictable`, the
    /// requests the owner can evict now.
    pub fn shortage(&self, evictable: &std::collections::BTreeSet<crate::operation::RequestId>) -> Option<Shortage> {
        let mut state = self.state();
        let estimate = state.estimate()?;
        if state.answered == Some(estimate.at()) {
            return None;
        }
        state.answered = Some(estimate.at());
        let cause = shortage::cause(&estimate, state.boundary.harmed(), state.boundary.pressed())?;
        let relief = state.plan_relief(&self.store.identity, shortage::lacks(&estimate, cause), shortage::allows(cause, evictable), super::admission::Frees::Room);
        Some(shortage::respond(&estimate, cause, relief, &state.persistence))
    }

    /// The bytes `need` takes, by the formulas a claim of it is sized by.
    pub fn bytes(&self, need: &Need) -> u64 {
        size(&self.shape, need)
    }

    /// Whether `bytes` more could be held beside
    /// every committed claim at `held`, a reading of the store a load
    /// holds, or else at the latest: refused as [`Self::claim_at`] refuses
    /// them.
    pub fn covers(&self, held: Option<&Estimate>, bytes: u64) -> Result<(), Short> {
        let state = self.state();
        match state.covers(state.reading(&self.store.identity, held.map(|held| held.read)), bytes) {
            Admit::Admit => Ok(()),
            Admit::Refuse(over) => Err(over.of(&self.store.identity, bytes)),
        }
    }

    /// The store's estimate a load prices at `held`, a reading of the
    /// store it holds from its start: the reading's OS figures fixed, and
    /// every holding since it counted, the load's own claims and other
    /// loads' alike (design §5). Reads no OS figure. The one room a load's
    /// plan, search and claims read in a store: with nothing held since,
    /// [`Estimate::room`] of it is the reading's own room, and each claim at
    /// the reading ([`Self::claim_at`]) lowers it by exactly what it took.
    pub fn room_after(&self, held: &Estimate) -> Estimate {
        self.state().estimate_at(held.read)
    }
}
