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
//!   many effects) wraps its fields in [`ScopedDrop`], which opens the change
//!   on the dropping thread before running [`DropInChange::before_drop`] and
//!   dropping the fields, or holds a `&Change` across the effect and its
//!   records.

use seismic_topology::StoreIdentity;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
/// it drops. A change begun where this thread has one under way joins it.
///
/// `!Send + !Sync`: a change ends on the thread that began it, enforced at
/// compile time.
#[must_use = "a change ends as it drops"]
pub struct Change {
    /// The embedder's guard, held only for its drop.
    _guard: Option<Box<dyn std::any::Any>>,
    /// Enforces `!Send + !Sync` so a `Change` cannot cross threads.
    _not_send: PhantomData<*const ()>,
}

impl Change {
    /// A change the embedder holds open by `guard` until it drops.
    pub fn held(guard: impl std::any::Any) -> Self {
        Self {
            _guard: Some(Box::new(guard)),
            _not_send: PhantomData,
        }
    }

    /// No change: what [`Tally`], the entry with nothing beside its OS
    /// effects to hold off, returns from [`Entry::change`]. Crate-private,
    /// so a change comes only from an entry.
    pub(crate) fn none() -> Self {
        Self {
            _guard: None,
            _not_send: PhantomData,
        }
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
/// object wrapped in a [`ScopedDrop`], so the two are one change.
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

    /// Convert this charge into a [`CarvableCharge`] whose uncarved bytes can
    /// be safely carved for child entries without over-carving, stale counts,
    /// or drop-order double-frees.
    pub fn into_carvable(mut self) -> CarvableCharge {
        let bytes = std::mem::take(&mut self.bytes);
        CarvableCharge {
            pool: Arc::new(CarvePool {
                entry: self.entry.clone(),
                wiring: self.wiring,
                total: bytes,
                uncarved: Mutex::new(Some(bytes)),
            }),
        }
    }

    /// Convert this uncarved block remainder charge paired with `first`
    /// (already charged to the caller's entry) into a [`CarvableCharge`] of
    /// `self.bytes() + first.bytes()` total bytes and the pre-carved [`Carved`]
    /// handle for `first`.
    pub fn into_carvable_with_first(mut self, mut first: Charge) -> (CarvableCharge, Carved) {
        assert_eq!(self.entry.store(), first.entry.store(), "invariant violated: first range belongs to another store");
        assert_eq!(self.wiring, first.wiring, "invariant violated: first range has different wiring");
        let uncarved = std::mem::take(&mut self.bytes);
        let first_bytes = std::mem::take(&mut first.bytes);
        let total = uncarved.checked_add(first_bytes).expect("carvable block bytes overflow");
        let pool = Arc::new(CarvePool {
            entry: self.entry.clone(),
            wiring: self.wiring,
            total,
            uncarved: Mutex::new(Some(uncarved)),
        });
        let carved = Carved {
            pool: pool.clone(),
            entry: first.entry.clone(),
            bytes: first_bytes,
        };
        (CarvableCharge { pool }, carved)
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

/// A value whose `Drop` (an optional OS/driver effect followed by dropping
/// all of its fields) runs inside one [`Change`] opened on the dropping
/// thread ([`ScopedDrop`]).
pub trait DropInChange {
    /// The ledger entry whose [`Change`] spans this value's drop, if any.
    fn drop_entry(&self) -> Option<&Arc<dyn Entry>>;

    /// Optional OS/driver effect to run before fields drop, while the
    /// [`Change`] returned by [`Self::drop_entry`] is held open.
    fn before_drop(&mut self) {}
}

/// Owns a `T: DropInChange` and guarantees by construction that
/// [`DropInChange::before_drop`] and every field drop of `T` execute inside
/// one stack-local [`Change`] on the dropping thread.
///
/// Unlike a field-ordered drop guard, `ScopedDrop<T>` is `Send`/`Sync`
/// whenever `T` is, never stores a [`Change`] in a struct, and cannot be
/// broken by reordering fields.
pub struct ScopedDrop<T: DropInChange> {
    inner: Option<T>,
}

impl<T: DropInChange> ScopedDrop<T> {
    pub fn new(inner: T) -> Self {
        Self { inner: Some(inner) }
    }
}

impl<T: DropInChange> std::ops::Deref for ScopedDrop<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.inner
            .as_ref()
            .expect("a scoped drop holds its value until it drops")
    }
}

impl<T: DropInChange> std::ops::DerefMut for ScopedDrop<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.inner
            .as_mut()
            .expect("a scoped drop holds its value until it drops")
    }
}

impl<T: DropInChange> Drop for ScopedDrop<T> {
    fn drop(&mut self) {
        let Some(mut inner) = self.inner.take() else {
            return;
        };
        let _change = inner.drop_entry().map(|entry| entry.change());
        inner.before_drop();
        drop(inner);
    }
}

/// A charge type that [`Backed`] can pair with a native allocation.
pub trait BackedCharge {
    fn entry(&self) -> &Arc<dyn Entry>;
}

impl BackedCharge for Charge {
    fn entry(&self) -> &Arc<dyn Entry> {
        Charge::entry(self)
    }
}

impl BackedCharge for CarvableCharge {
    fn entry(&self) -> &Arc<dyn Entry> {
        CarvableCharge::entry(self)
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
pub struct Backed<T, C: BackedCharge = Charge> {
    /// Both until it drops, which takes them out.
    parts: Option<(T, C)>,
}

impl<T> Backed<T, Charge> {
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

    /// Convert this backed object's charge into a [`CarvableCharge`] so
    /// sub-ranges can be safely carved from it.
    pub fn into_carvable(mut self) -> Backed<T, CarvableCharge> {
        let (native, charge) = self
            .parts
            .take()
            .expect("a backed object holds its parts until it drops");
        Backed {
            parts: Some((native, charge.into_carvable())),
        }
    }

    /// Allocate a native carvable block whose first range is already charged
    /// to `first` (the caller's entry) and whose uncarved remainder is charged
    /// to `own` (the block pool's own entry). If `allocate` fails, both
    /// charges drop and free cleanly.
    pub fn allocate_carvable_with_first<E>(
        own: Charge,
        first: Charge,
        allocate: impl FnOnce(u64) -> Result<T, E>,
    ) -> Result<(Backed<T, CarvableCharge>, Carved), E> {
        let total = own.bytes().checked_add(first.bytes()).expect("carvable block bytes overflow");
        let native = allocate(total)?;
        let (carvable, carved) = own.into_carvable_with_first(first);
        Ok((Backed { parts: Some((native, carvable)) }, carved))
    }
}

impl<T, C: BackedCharge> Backed<T, C> {
    pub fn charge(&self) -> &C {
        &self.parts.as_ref().expect("a backed object holds its parts until it drops").1
    }

    /// The entry its bytes are charged to.
    pub fn entry(&self) -> &Arc<dyn Entry> {
        self.charge().entry()
    }
}

impl<T, C: BackedCharge> std::ops::Deref for Backed<T, C> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.parts.as_ref().expect("a backed object holds its parts until it drops").0
    }
}

impl<T, C: BackedCharge> std::ops::DerefMut for Backed<T, C> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.parts.as_mut().expect("a backed object holds its parts until it drops").0
    }
}

impl<T, C: BackedCharge> Drop for Backed<T, C> {
    /// The native object freed, then its bytes, inside one change.
    fn drop(&mut self) {
        let Some((native, charge)) = self.parts.take() else {
            return;
        };
        let _change = charge.entry().change();
        drop(native);
        drop(charge);
    }
}

/// Shared state between a [`CarvableCharge`] (a block's charge) and every
/// live [`Carved`] range carved from it.
struct CarvePool {
    entry: Arc<dyn Entry>,
    wiring: Wiring,
    total: u64,
    /// `Some(uncarved)` while the parent `CarvableCharge` is live; `None`
    /// once the parent has released its uncarved bytes.
    uncarved: Mutex<Option<u64>>,
}

/// A block's charge whose uncarved bytes can be carved into [`Carved`]
/// sub-ranges for other entries of the same store.
///
/// Tracks live uncarved bytes atomically with every [`Carved`] handle so
/// over-carving is rejected, `uncarved_bytes()` is always exact, and the
/// ledger never double-frees or leaks regardless of drop order.
#[must_use]
pub struct CarvableCharge {
    pool: Arc<CarvePool>,
}

impl CarvableCharge {
    /// Total bytes originally charged to this block.
    pub fn bytes(&self) -> u64 {
        self.pool.total
    }

    /// Bytes currently held on the parent entry (not carved out and not yet
    /// released).
    pub fn uncarved_bytes(&self) -> u64 {
        self.pool
            .uncarved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .unwrap_or(0)
    }

    pub fn wiring(&self) -> Wiring {
        self.pool.wiring
    }

    pub fn entry(&self) -> &Arc<dyn Entry> {
        &self.pool.entry
    }

    /// Carve `bytes` from this block's uncarved space for `entry`, of the
    /// block's wiring: the parent frees them and `entry` takes them inside
    /// one change. Dropped, they return to the parent if it is still live.
    pub fn carve(&self, entry: &Arc<dyn Entry>, bytes: u64) -> Carved {
        assert_eq!(
            self.pool.entry.store(),
            entry.store(),
            "invariant violated: a range carved for an entry of another store"
        );
        let mut guard = self
            .pool
            .uncarved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(rem) = guard.as_mut() else {
            drop(guard);
            panic!("invariant violated: carving from a released block charge");
        };
        if bytes > *rem {
            let available = *rem;
            drop(guard);
            panic!(
                "invariant violated: carving {bytes} bytes from a block with only {available} uncarved bytes"
            );
        }
        // Hold both entries' change guards so neither a Tally nor an embedder
        // entry ever records outside a change.
        let _parent_change = self.pool.entry.change();
        let change = entry.change();
        self.pool.entry.freed(&change, bytes, self.pool.wiring);
        entry.took(&change, bytes, self.pool.wiring);
        *rem -= bytes;
        Carved {
            pool: self.pool.clone(),
            entry: entry.clone(),
            bytes,
        }
    }

    /// Explicitly release all remaining uncarved bytes inside `change`.
    /// Subsequent [`Drop`] is a no-op; any still-live [`Carved`] handles will
    /// only free their own bytes on drop without returning them to the parent.
    pub fn release(&self, change: &Change) {
        let mut guard = self
            .pool
            .uncarved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(rem) = guard.take() {
            if rem != 0 {
                self.pool.entry.freed(change, rem, self.pool.wiring);
            }
        }
    }
}

impl Drop for CarvableCharge {
    fn drop(&mut self) {
        let mut guard = self
            .pool
            .uncarved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(rem) = guard.take() {
            if rem != 0 {
                let change = self.pool.entry.change();
                self.pool.entry.freed(&change, rem, self.pool.wiring);
            }
        }
    }
}

/// A range carved from a block's free space ([`CarvableCharge::carve`]): its
/// bytes are its entry's while it lives, and return to the block's parent
/// entry on drop if the block is still live.
#[must_use]
pub struct Carved {
    pool: Arc<CarvePool>,
    entry: Arc<dyn Entry>,
    bytes: u64,
}

impl Carved {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Carved {
    fn drop(&mut self) {
        let mut guard = self
            .pool
            .uncarved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _parent_change = self.pool.entry.change();
        let change = self.entry.change();
        self.entry.freed(&change, self.bytes, self.pool.wiring);
        if let Some(rem) = guard.as_mut() {
            self.pool.entry.took(&change, self.bytes, self.pool.wiring);
            *rem += self.bytes;
        }
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

    /// Carving tracks uncarved bytes accurately and frees exact total bytes
    /// even if the parent `CarvableCharge` drops before a `Carved` handle
    /// (zero double-free, zero leak).
    #[test]
    fn carvable_charge_is_exact_regardless_of_drop_order() {
        let (parent_entry, parent_tally, parent_open) = Counted::new();
        let (child_entry, child_tally, child_open) = Counted::new();
        let change = parent_entry.change();
        let block = Charge::took(&change, &parent_entry, 100, Wiring::Pageable).into_carvable();
        drop(change);
        assert_eq!((block.bytes(), block.uncarved_bytes(), parent_tally.held()), (100, 100, 100));

        let c1 = block.carve(&child_entry, 40);
        let c2 = block.carve(&child_entry, 25);
        assert_eq!((block.uncarved_bytes(), parent_tally.held(), child_tally.held()), (35, 35, 65));

        // Normal order: c1 drops while block is still live -> returns 40 to parent.
        drop(c1);
        assert_eq!((block.uncarved_bytes(), parent_tally.held(), child_tally.held()), (75, 75, 25));

        // Reverse order: parent block drops while c2 is still live -> parent frees only its 75 uncarved bytes.
        drop(block);
        assert_eq!((parent_tally.held(), child_tally.held()), (0, 25));

        // c2 drops after parent -> frees its 25 bytes without returning them to parent.
        drop(c2);
        assert_eq!(
            (parent_tally.held(), child_tally.held(), parent_open.load(Ordering::SeqCst), child_open.load(Ordering::SeqCst)),
            (0, 0, 0, 0)
        );
    }

    /// Carving more than the remaining uncarved bytes panics immediately.
    #[test]
    #[should_panic(expected = "carving 60 bytes from a block with only 40 uncarved bytes")]
    fn carvable_charge_rejects_over_carving() {
        let (parent_entry, _, _) = Counted::new();
        let (child_entry, _, _) = Counted::new();
        let change = parent_entry.change();
        let block = Charge::took(&change, &parent_entry, 100, Wiring::Pageable).into_carvable();
        drop(change);
        let _c1 = block.carve(&child_entry, 60);
        let _c2 = block.carve(&child_entry, 60);
    }

    /// `ScopedDrop` runs `before_drop` and every field's `Drop` inside one
    /// open `Change`.
    #[test]
    fn scoped_drop_keeps_change_open_across_effect_and_fields() {
        let (entry, tally, open) = Counted::new();
        let before_ran = Arc::new(AtomicUsize::new(0));
        let frees = Arc::new(AtomicUsize::new(0));
        struct Resource {
            entry: Arc<dyn Entry>,
            open: Arc<AtomicUsize>,
            before_ran: Arc<AtomicUsize>,
            _native: Native,
            _charge: Charge,
        }
        impl DropInChange for Resource {
            fn drop_entry(&self) -> Option<&Arc<dyn Entry>> {
                Some(&self.entry)
            }
            fn before_drop(&mut self) {
                assert!(self.open.load(Ordering::SeqCst) > 0, "before_drop outside a change");
                self.before_ran.fetch_add(1, Ordering::SeqCst);
            }
        }
        let change = entry.change();
        let charge = Charge::took(&change, &entry, 32, Wiring::Pageable);
        drop(change);
        let scoped = ScopedDrop::new(Resource {
            entry: entry.clone(),
            open: open.clone(),
            before_ran: before_ran.clone(),
            _native: Native(open.clone(), frees.clone()),
            _charge: charge,
        });
        assert_eq!((tally.held(), open.load(Ordering::SeqCst)), (32, 0));
        drop(scoped);
        assert_eq!(
            (tally.held(), open.load(Ordering::SeqCst), before_ran.load(Ordering::SeqCst), frees.load(Ordering::SeqCst)),
            (0, 0, 1, 1)
        );
    }
}
