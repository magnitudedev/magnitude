//! The embedder's ledger as Seismic allocates through it: Seismic keeps no
//! count of the memory it holds. Every byte it takes from the OS or a driver
//! is charged to one entry of the embedder's ledger ([`Entry`]) as it is
//! taken, and returned to it as it is freed, so the embedder's count of the
//! process's memory is the only one and nothing Seismic holds lies outside
//! it: an allocation's rounding, a pool block's free granules and a driver's
//! objects are bytes of the entry that caused them.
//!
//! An OS effect (a native allocation or free, a driver call that takes or
//! returns memory) and the record of its bytes are one change of the
//! process's memory ([`Change`]): a reading of the machine never stands
//! between them, so it never shows the process's own effect as other
//! programs' memory. The effect is therefore never made outside the change
//! that records it, by construction:
//!
//! - **A record needs the change.** Every record of an entry ([`Entry::take`],
//!   [`Entry::took`], [`Entry::freed`]) and every operation of a [`Charge`]
//!   that makes one (its constructors, [`Charge::grow`],
//!   [`Charge::took_more`], [`Charge::shrink`]) takes the `&Change` it is
//!   made in. A `Change` is made only by [`Entry::change`] (the embedder's
//!   guard), so no record can be written without one. The two records that
//!   cannot be passed one, a charge's and a carved range's own end as they
//!   drop, begin the change themselves ([`Entry::change`] joins the one
//!   under way), so they too lie inside one.
//! - **An effect is made inside the change that records it.** [`Backed`]
//!   owns a native object with its charge and makes the allocation and the
//!   free inside one change. An object that cannot be one `Backed` (its own
//!   drop makes effects beside its fields' frees, or it holds charges of
//!   many effects) keeps its change open across its whole drop with a
//!   [`DropScope`] as its last field, or holds a `&Change` across the
//!   effect and its records. The objects that use a `DropScope` are named
//!   at it.

use seismic_topology::StoreIdentity;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// An entry's refusal of bytes it was asked to admit: `required` more, of
/// which it had room for `available`. Nothing was allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refused {
    pub required: u64,
    pub available: u64,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the ledger admits {} of {} bytes", self.available, self.required)
    }
}

/// Whether the OS wires charged bytes while the process holds them: what a
/// wired ceiling (unified Metal's `iogpu.wired_limit_mb`) counts as the
/// process's own wired memory, apart from the rest of its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wiring {
    /// Wired by the OS for as long as the process holds them: a Metal
    /// buffer in its device's residency set, and so every pool block there.
    Wired,
    /// Everything else: anonymous host memory, a Metal buffer outside the
    /// residency set, device memory a discrete device's driver holds.
    Pageable,
}

/// One change of the process's memory under way on this thread, from before
/// its OS effect until its records are made: the guard of the embedder's
/// change ([`Entry::change`]; backends take theirs only from there), ended as
/// it drops. A change begun where this
/// thread has one under way joins it. It ends on the thread that began it:
/// the embedder panics where it drops elsewhere.
#[must_use = "a change ends as it drops"]
pub struct Change {
    /// The embedder's guard, held only for its drop.
    _guard: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl Change {
    /// A change the embedder holds open by `guard` until it drops.
    pub fn held(guard: impl std::any::Any + Send + Sync) -> Self {
        Self { _guard: Some(Box::new(guard)) }
    }

    /// No change: what [`Tally`], the entry with nothing beside its OS
    /// effects to hold off, returns from [`Entry::change`]. Crate-private,
    /// so a change comes only from an entry.
    pub(crate) fn none() -> Self {
        Self { _guard: None }
    }
}

/// One entry of the embedder's ledger: one holder's bytes in one store.
pub trait Entry: Send + Sync + 'static {
    /// The store the entry's bytes lie in.
    fn store(&self) -> &StoreIdentity;
    /// Begin a change of the process's memory on this thread, or join the one
    /// it has under way: every record of this entry is made inside one, and
    /// the OS effect it records is made inside the same, so no reading of
    /// the machine stands between the two ([`Backed`]). It never waits.
    fn change(&self) -> Change;
    /// Run `closed` once the entry's holder is gone and the entry stays
    /// alive only for the allocations charged to it: at once where it is.
    /// What an allocation keeps idle for reuse is returned then, inside its
    /// change, not left until something else asks.
    fn on_close(&self, closed: Box<dyn FnOnce() + Send>);
    /// Whether the entry's holder is gone ([`Self::on_close`]).
    fn closed(&self) -> bool;
    /// Admit `bytes` more of `wiring`, before a native allocation of that
    /// size is made, inside `change` (the one of [`Self::change`] its effect
    /// is made in). Refused, nothing is allocated; admitted, the bytes are
    /// the entry's until [`Self::freed`] with the same wiring.
    fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused>;
    /// `bytes` of `wiring` a driver already took for the entry, known only
    /// after the native call made them (an allocation's rounding, driver
    /// objects, recorded graphs): never refused. Recorded inside `change`.
    fn took(&self, change: &Change, bytes: u64, wiring: Wiring);
    /// `bytes` of the entry, taken as `wiring`, returned to the OS or the
    /// driver, as they are freed, inside `change`.
    fn freed(&self, change: &Change, bytes: u64, wiring: Wiring);
}

/// Bytes charged to an entry, held with what they back and freed from it
/// when dropped: a native allocation's exact bytes ([`Self::take`]), or
/// memory a driver took ([`Self::took`]). Every operation that records into
/// the ledger takes the `&Change` it is made in; the charge's own end, which
/// cannot be passed one, begins the change itself, so it too lies inside
/// one. The free of what a charge backs is its owner's: a [`Backed`], or an
/// object with a [`DropScope`], so the two are one change.
///
/// What this makes unwritable (each fails to compile for want of a
/// `&Change`): `Charge::take(&entry, bytes, wiring)`, `charge.shrink(bytes)`,
/// `entry.take(bytes, wiring)`: a record outside the change of its effect.
#[must_use]
pub struct Charge {
    entry: Arc<dyn Entry>,
    bytes: u64,
    wiring: Wiring,
}

impl Charge {
    /// Admit `bytes` of `wiring` on `entry` before the native allocation
    /// they back, inside `change`.
    pub fn take(change: &Change, entry: &Arc<dyn Entry>, bytes: u64, wiring: Wiring) -> Result<Self, Refused> {
        entry.take(change, bytes, wiring)?;
        Ok(Self { entry: entry.clone(), bytes, wiring })
    }

    /// `bytes` of `wiring` a driver already took for `entry`, recorded
    /// inside `change`, which the driver call was made in.
    pub fn took(change: &Change, entry: &Arc<dyn Entry>, bytes: u64, wiring: Wiring) -> Self {
        entry.took(change, bytes, wiring);
        Self { entry: entry.clone(), bytes, wiring }
    }

    /// `bytes` of `from`'s charge, already held (a block's free space),
    /// carved for `entry` (a range of the block), of the block's wiring:
    /// `from` frees them and `entry` takes them, never refused, since the
    /// process holds nothing more. Dropped, they return to `from`. It makes
    /// no OS effect, so it is one change of its own.
    pub fn carve(from: &Charge, entry: &Arc<dyn Entry>, bytes: u64) -> Carved {
        assert_eq!(from.entry.store(), entry.store(), "invariant violated: a range carved for an entry of another store");
        // One change: the bytes are never the block's and no one's between.
        let change = entry.change();
        from.entry.freed(&change, bytes, from.wiring);
        entry.took(&change, bytes, from.wiring);
        Carved { from: from.entry.clone(), charge: Self { entry: entry.clone(), bytes, wiring: from.wiring } }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn wiring(&self) -> Wiring {
        self.wiring
    }

    pub fn entry(&self) -> &Arc<dyn Entry> {
        &self.entry
    }

    /// Charge `more` bytes of its wiring on the same entry, admitted first,
    /// inside `change`.
    pub fn grow(&mut self, change: &Change, more: u64) -> Result<(), Refused> {
        self.entry.take(change, more, self.wiring)?;
        self.bytes += more;
        Ok(())
    }

    /// `more` bytes of its wiring on the same entry that the backing took
    /// beyond what was admitted, known only after the native call: never
    /// refused, recorded inside `change`.
    pub fn took_more(&mut self, change: &Change, more: u64) {
        self.entry.took(change, more, self.wiring);
        self.bytes += more;
    }

    /// Free `less` of the bytes, which the backing returned, inside
    /// `change`.
    pub fn shrink(&mut self, change: &Change, less: u64) {
        assert!(less <= self.bytes, "invariant violated: {less} bytes freed of a charge of {}", self.bytes);
        self.entry.freed(change, less, self.wiring);
        self.bytes -= less;
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        if self.bytes != 0 {
            // Joins the change its owner has under way, where it has one.
            let change = self.entry.change();
            self.entry.freed(&change, self.bytes, self.wiring);
        }
    }
}

/// A charge being settled by the allocation it backs, inside the change that
/// allocation runs in ([`Backed::take`]): the charge's operations with that
/// change held.
pub struct Charging<'a> {
    change: &'a Change,
    charge: &'a mut Charge,
}

impl Charging<'_> {
    pub fn bytes(&self) -> u64 {
        self.charge.bytes
    }

    /// [`Charge::grow`] in the allocation's change.
    pub fn grow(&mut self, more: u64) -> Result<(), Refused> {
        self.charge.grow(self.change, more)
    }

    /// [`Charge::took_more`] in the allocation's change.
    pub fn took_more(&mut self, more: u64) {
        self.charge.took_more(self.change, more);
    }

    /// [`Charge::shrink`] in the allocation's change.
    pub fn shrink(&mut self, less: u64) {
        self.charge.shrink(self.change, less);
    }
}

/// The change an object's own `Drop` runs in, for an object that cannot be
/// one [`Backed`]: its own drop makes OS effects beside its fields' frees (a
/// Metal buffer's pages discarded as it is released, `WiredAllocation`), it
/// is a context or module whose release and its charges are many effects
/// (CUDA's `Context` and `Module`), or the charges of its effects are pooled
/// (CUDA's `DirectGraph`, its graphs' driver memory). Declared as the
/// object's last field, begun first in its `Drop::drop`
/// ([`Self::begin`]), it ends after every field has dropped, so the frees
/// and the record of them are one change. A clone is another object: it
/// begins its own change as it drops.
#[derive(Default)]
pub struct DropScope(Option<Change>);

impl Clone for DropScope {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl DropScope {
    /// Begin the change of `entry` that the object's drop runs in, and
    /// return it, for the records the drop makes of its own.
    pub fn begin(&mut self, entry: &Arc<dyn Entry>) -> &Change {
        self.0.insert(entry.change())
    }
}

/// A native object with the bytes charged for it: the one owner of an OS
/// effect and its record. The allocation is made inside the change that
/// admits its bytes ([`Self::take`]) or records the bytes a driver took
/// ([`Self::took`]), and the object is freed, then its bytes, inside one as
/// it drops, so no reading of the machine stands between an effect and its
/// record. The change is begun on the thread that makes the effect, as
/// [`Entry::change`] says; where that thread's free waits (a backend waiting
/// for the device to finish with a block's storage), the wait is inside the
/// change and holds readings off for its length.
pub struct Backed<T> {
    /// Both until it drops, which takes them out.
    parts: Option<(T, Charge)>,
}

impl<T> Backed<T> {
    /// `bytes` of `wiring` admitted on `entry`, then `allocate` run to make
    /// the native object they back, with the charge for it to settle (the
    /// bytes the backing rounded, [`Charging::took_more`], or returned,
    /// [`Charging::shrink`]), all in one change. Refused or failed, nothing
    /// is allocated and nothing stays charged.
    pub fn take<E: From<Refused>>(
        entry: &Arc<dyn Entry>,
        bytes: u64,
        wiring: Wiring,
        allocate: impl FnOnce(&mut Charging<'_>) -> Result<T, E>,
    ) -> Result<Self, E> {
        let change = entry.change();
        Self::allocate(&change, Charge::take(&change, entry, bytes, wiring)?, allocate)
    }

    /// `allocate` run to make the native object `charge`, admitted already
    /// inside `change`, backs, in that change; failed, the charge is freed
    /// there.
    pub fn allocate<E>(change: &Change, mut charge: Charge, allocate: impl FnOnce(&mut Charging<'_>) -> Result<T, E>) -> Result<Self, E> {
        let native = allocate(&mut Charging { change, charge: &mut charge })?;
        Ok(Self { parts: Some((native, charge)) })
    }

    /// `allocate` run to make a native object and the bytes a driver took
    /// for it, which are then charged to `entry`, all in one change.
    pub fn took<E>(entry: &Arc<dyn Entry>, wiring: Wiring, allocate: impl FnOnce() -> Result<(T, u64), E>) -> Result<Self, E> {
        let change = entry.change();
        let (native, bytes) = allocate()?;
        Ok(Self { parts: Some((native, Charge::took(&change, entry, bytes, wiring))) })
    }

    pub fn charge(&self) -> &Charge {
        &self.parts.as_ref().expect("a backed object holds its parts until it drops").1
    }

    /// The entry its bytes are charged to.
    pub fn entry(&self) -> &Arc<dyn Entry> {
        self.charge().entry()
    }
}

impl<T> std::ops::Deref for Backed<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.parts.as_ref().expect("a backed object holds its parts until it drops").0
    }
}

impl<T> std::ops::DerefMut for Backed<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.parts.as_mut().expect("a backed object holds its parts until it drops").0
    }
}

impl<T> Drop for Backed<T> {
    /// The native object freed, then its bytes, inside one change.
    fn drop(&mut self) {
        let Some((native, charge)) = self.parts.take() else {
            return;
        };
        let _change = charge.entry.change();
        drop(native);
        drop(charge);
    }
}

/// A range carved from a block's free space ([`Charge::carve`]): its bytes
/// are its entry's while it lives, and the block's again after.
#[must_use]
pub struct Carved {
    from: Arc<dyn Entry>,
    charge: Charge,
}

impl Carved {
    pub fn bytes(&self) -> u64 {
        self.charge.bytes
    }
}

impl Drop for Carved {
    fn drop(&mut self) {
        let change = self.charge.entry.change();
        let bytes = std::mem::take(&mut self.charge.bytes);
        self.charge.entry.freed(&change, bytes, self.charge.wiring);
        self.from.took(&change, bytes, self.charge.wiring);
    }
}

/// The embedder's ledger of one store, which a device opens with, one for
/// each store it reaches.
pub trait Store: Send + Sync + 'static {
    fn identity(&self) -> &StoreIdentity;
    /// The bytes the ledger counts the process holding in the store now.
    fn allocated(&self) -> u64;
    /// The most a [`Entry::take`] in the store admits now.
    fn room(&self) -> u64;
    /// The most [`Self::allocated`] has been since the last
    /// [`Self::restart_high_water`] (or since the store's ledger began).
    fn high_water(&self) -> u64;
    /// Start [`Self::high_water`] again from what is allocated now.
    fn restart_high_water(&self);
    /// The device's own entry in the store: what it holds that no caller
    /// names (transfer staging, its contexts and queues, driver objects),
    /// which its backend charges by name. Seismic alone asks for it: a
    /// device's charges are minted from each store's identity, never the
    /// store, so no holder's charges name it.
    fn own(&self) -> Arc<dyn Entry>;
    /// Whether the store is a [`Tally`], its one entry counting every byte
    /// charged to it, rather than an embedder's ledger minting an entry for
    /// each holder. Only a tally is one.
    fn tally(&self) -> bool {
        false
    }
}

/// A store and its one entry counting every byte charged to it, admitting
/// up to a limit where it has one: the ledger of a device opened outside any
/// embedder's (a backend's own tests, a probe process, a frontend with no
/// ledger of its own), and the bound a test refuses allocations with. Clones
/// are the same tally.
#[derive(Clone)]
pub struct Tally(Arc<Count>);

struct Count {
    identity: StoreIdentity,
    limit: AtomicU64,
    held: AtomicU64,
    high_water: AtomicU64,
}

impl Tally {
    /// A tally of `identity` admitting everything.
    pub fn unbounded(identity: StoreIdentity) -> Self {
        Self::limited(identity, u64::MAX)
    }

    /// A tally of `identity` admitting up to `limit` bytes held.
    pub fn limited(identity: StoreIdentity, limit: u64) -> Self {
        Self(Arc::new(Count { identity, limit: AtomicU64::new(limit), held: AtomicU64::new(0), high_water: AtomicU64::new(0) }))
    }

    /// The bytes charged now.
    pub fn held(&self) -> u64 {
        self.0.held.load(Ordering::Acquire)
    }

    /// Admit up to `limit` bytes held from now (`u64::MAX`: everything).
    /// Bytes already held stay held.
    pub fn bound(&self, limit: u64) {
        self.0.limit.store(limit, Ordering::Release);
    }
}

impl Entry for Count {
    fn store(&self) -> &StoreIdentity {
        &self.identity
    }

    fn change(&self) -> Change {
        Change::none()
    }

    /// A tally has no holder: it is closed from the start, so what an
    /// allocation keeps idle for reuse returns as its last buffer drops
    /// ([`Entry::on_close`]).
    fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
        closed();
    }

    fn closed(&self) -> bool {
        true
    }

    fn take(&self, _change: &Change, bytes: u64, _wiring: Wiring) -> Result<(), Refused> {
        let limit = self.limit.load(Ordering::Acquire);
        self.held
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| held.checked_add(bytes).filter(|&next| next <= limit))
            .map(|held| self.rose_to(held + bytes))
            .map_err(|held| Refused { required: bytes, available: limit.saturating_sub(held) })
    }

    fn took(&self, _change: &Change, bytes: u64, _wiring: Wiring) {
        let held = self.held.fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.rose_to(held);
    }

    /// Recorded unwinding or not: every free is of bytes a take recorded,
    /// so a panic's unwind frees only what the tally holds. A free of more
    /// is an invariant violation, which leaves the count as it was and
    /// panics, aborting the process where it unwinds already.
    fn freed(&self, _change: &Change, bytes: u64, _wiring: Wiring) {
        self.held
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| held.checked_sub(bytes))
            .unwrap_or_else(|held| panic!("invariant violated: {bytes} bytes freed of a tally holding {held}"));
    }
}

impl Count {
    fn rose_to(&self, held: u64) {
        self.high_water.fetch_max(held, Ordering::AcqRel);
    }
}

impl Entry for Tally {
    fn store(&self) -> &StoreIdentity {
        self.0.store()
    }

    fn change(&self) -> Change {
        self.0.change()
    }

    fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
        self.0.on_close(closed);
    }

    fn closed(&self) -> bool {
        self.0.closed()
    }

    fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused> {
        self.0.take(change, bytes, wiring)
    }

    fn took(&self, change: &Change, bytes: u64, wiring: Wiring) {
        self.0.took(change, bytes, wiring)
    }

    fn freed(&self, change: &Change, bytes: u64, wiring: Wiring) {
        self.0.freed(change, bytes, wiring)
    }
}

impl Store for Tally {
    fn identity(&self) -> &StoreIdentity {
        &self.0.identity
    }

    fn allocated(&self) -> u64 {
        self.held()
    }

    fn room(&self) -> u64 {
        self.0.limit.load(Ordering::Acquire).saturating_sub(self.held())
    }

    fn high_water(&self) -> u64 {
        self.0.high_water.load(Ordering::Acquire)
    }

    fn restart_high_water(&self) {
        self.0.high_water.store(self.held(), Ordering::Release);
    }

    /// The tally itself: a tally is its store's one entry, the same entry
    /// on every call.
    fn own(&self) -> Arc<dyn Entry> {
        self.0.clone()
    }

    fn tally(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A store's tally whose changes are counted: an effect or a record made
    /// with none under way is the test's failure.
    struct Counted {
        tally: Tally,
        open: Arc<AtomicUsize>,
    }

    struct Open(Arc<AtomicUsize>);

    impl Drop for Open {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Counted {
        fn new() -> (Arc<dyn Entry>, Tally, Arc<AtomicUsize>) {
            let tally = Tally::unbounded(StoreIdentity::HostRam);
            let open = Arc::new(AtomicUsize::new(0));
            (Arc::new(Self { tally: tally.clone(), open: open.clone() }), tally, open)
        }

        fn inside(&self) {
            assert!(self.open.load(Ordering::SeqCst) > 0, "a record made outside a change");
        }
    }

    impl Entry for Counted {
        fn store(&self) -> &StoreIdentity {
            self.tally.store()
        }

        fn change(&self) -> Change {
            self.open.fetch_add(1, Ordering::SeqCst);
            Change::held(Open(self.open.clone()))
        }

        fn on_close(&self, _closed: Box<dyn FnOnce() + Send>) {}

        fn closed(&self) -> bool {
            false
        }

        fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused> {
            self.inside();
            self.tally.take(change, bytes, wiring)
        }

        fn took(&self, change: &Change, bytes: u64, wiring: Wiring) {
            self.inside();
            self.tally.took(change, bytes, wiring);
        }

        fn freed(&self, change: &Change, bytes: u64, wiring: Wiring) {
            self.inside();
            self.tally.freed(change, bytes, wiring);
        }
    }

    /// A native object whose free is observed.
    struct Native(Arc<AtomicUsize>, Arc<AtomicUsize>);

    impl Drop for Native {
        fn drop(&mut self) {
            assert!(self.0.load(Ordering::SeqCst) > 0, "a native free made outside a change");
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A native allocation, its admission and its record are one change,
    /// and so is its free with the free of its bytes: neither effect is made
    /// where no change is under way.
    #[test]
    fn a_backed_object_allocates_and_frees_inside_its_change() {
        let (entry, tally, open) = Counted::new();
        let frees = Arc::new(AtomicUsize::new(0));
        let backed = Backed::take(&entry, 64, Wiring::Pageable, |_| {
            assert_eq!(open.load(Ordering::SeqCst), 1, "the allocation runs in the change that admits it");
            Ok::<_, Refused>(Native(open.clone(), frees.clone()))
        })
        .expect("admitted");
        assert_eq!((open.load(Ordering::SeqCst), tally.held()), (0, 64));
        drop(backed);
        assert_eq!((open.load(Ordering::SeqCst), tally.held(), frees.load(Ordering::SeqCst)), (0, 0, 1));
    }

    /// Memory a driver took is recorded in the change its call ran in.
    #[test]
    fn a_driver_allocation_is_recorded_in_the_change_of_its_call() {
        let (entry, tally, open) = Counted::new();
        let frees = Arc::new(AtomicUsize::new(0));
        let backed = Backed::took(&entry, Wiring::Pageable, || {
            assert_eq!(open.load(Ordering::SeqCst), 1);
            Ok::<_, Refused>((Native(open.clone(), frees.clone()), 96))
        })
        .expect("never refused");
        assert_eq!(tally.held(), 96);
        drop(backed);
        assert_eq!((tally.held(), frees.load(Ordering::SeqCst)), (0, 1));
    }

    /// A charge dropped bare, with no change under way, is freed inside one
    /// it begins: no record lies outside a change.
    #[test]
    fn a_charge_dropped_bare_records_its_free_inside_a_change() {
        let (entry, tally, open) = Counted::new();
        let change = entry.change();
        let charge = Charge::took(&change, &entry, 8, Wiring::Pageable);
        drop(change);
        assert_eq!((open.load(Ordering::SeqCst), tally.held()), (0, 8));
        drop(charge);
        assert_eq!((open.load(Ordering::SeqCst), tally.held()), (0, 0));
    }

    /// A refused admission and a failed allocation allocate nothing and
    /// leave nothing charged.
    #[test]
    fn a_refused_or_failed_allocation_charges_nothing() {
        let (entry, tally, open) = Counted::new();
        tally.bound(10);
        let refused = Backed::take(&entry, 64, Wiring::Pageable, |_| -> Result<Native, Refused> { unreachable!("nothing is allocated") });
        assert!(refused.is_err());
        tally.bound(u64::MAX);
        let failed = Backed::take(&entry, 64, Wiring::Pageable, |_| Err::<Native, _>(Refused { required: 64, available: 0 }));
        assert!(failed.is_err());
        assert_eq!((tally.held(), open.load(Ordering::SeqCst)), (0, 0));
    }
}
