//! A device's granule pools (`seismic_memory::pool`) and the slab
//! stores over them: the storage the expert bank's records and the KV slabs
//! live in.
//!
//! - A load opens one pool per store a contract places in, once
//!   ([`Pool::open`]); fixtures open their own beside it. Its blocks are the backend's plain allocations
//!   (`OpenedKind::allocate_block`), each charged to the pool's ledger entry
//!   by the backend that makes it, through the device's charges
//!   ([`PoolCharges`]) as every allocation is.
//! - A slab is one granule. A store's address table holds each slot's region
//!   addresses, in the device's own memory, and each entry is written in
//!   place as one 8-byte store: by the host as a slab is added or freed
//!   where the host addresses the table, else as device work ordered after
//!   the work submitted before it, as it is when a slab moves
//!   ([`Pool::return_freed`]) and while a move's write of the table is
//!   pending; never by a host write through staging, which waits for the
//!   device's queue. So a
//!   slab is added, freed or moved beside device work that reads the table,
//!   never waiting on it: device work in flight reads the old address or
//!   the new one, each a whole slab's bytes held until that work completes.
//! - A free slot names the pool's absent granule, zero and held by no slab,
//!   so a masked load of a free slot reads device memory of a whole slab.
//! - The pool's blocks, its absent granule and its stores' tables share one
//!   hazard domain: device work bound to a store is a use of its pool, host
//!   access to a slab orders after it, and a granule a slab released is
//!   claimable once the pool's work submitted by then completes. A released
//!   granule leaves its slab once no view of it remains ([`SlabHold`]).
//! - A block is returned to the OS once every granule of it is free: no
//!   work in flight then reaches it, but its storage's release waits for
//!   every use of the domain, so a block leaves the pool at once and is
//!   released as the work in flight when it left ends, from that work's
//!   completion, never by a wait on the thread that returned it
//!   ([`Pool::return_freed`]).
//! - A block is emptied for outside demand by moving its live slabs into
//!   free granules of other blocks: each slab's copy and its store's
//!   repoint are device work ordered after the work submitted before them
//!   and before the work submitted after (`native::moves`), each moved
//!   slab's old granule released behind it, so the yield that asks for the
//!   bytes spends only their enqueue. A slab a view holds never moves: work
//!   no native queue orders (a transfer) holds the views of what it reads
//!   and writes until it completes, so no move copies bytes still landing.
//! - A store's leading slabs may be lent: records of an allocation its owner
//!   made, viewed for the store's life and never freed.

use crate::api::device::DeviceInner;
use crate::api::tensor::TensorInner;
use crate::api::TensorError;
use crate::completion::DeviceCompletion;
use crate::driver::{AccessGroup, Allocation, DeviceUses};
use crate::native::moves::{self, MoveStep};
use crate::memory::{Charges, Entry, Wiring};
use seismic_compiler::errors::ExecutionError;
use seismic_lang::ids::RepresentationId;
use seismic_memory::pool::{BlockId, Blocks, Claimed, Fence, GranuleId, GranulePool, Move};
use seismic_topology::Contract;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// How a pool's blocks are sized, which its opener decides
/// ([`Pool::open`]'s `block_bytes`).
pub use seismic_memory::pool::{block_bytes, BlockSize, BlockSizing, NativeAllocations, SizingError};

/// The alignment every native allocation a block is starts at, at least:
/// what a slab's alignment divides (`seismic::SlabLayout::slab_alignment`
/// starts at it).
///
/// A layout contract the runtime imposes on every backend, not a device
/// limit it reads, so it is one constant and not a per-backend query: a
/// slab's offsets inside a granule, and so the layout the kernels are
/// compiled against, are the same on every device. Every block's address is
/// checked against it as the block is allocated ([`Blocks::allocate`]): a
/// device whose allocations are weaker fails that check, loudly, rather
/// than shaping the layout to it.
pub const BLOCK_ALIGNMENT: u64 = 256;

/// The work that read a released granule: the pool's newest uses when it
/// was released.
impl Fence for DeviceUses {
    fn completed(&self) -> bool {
        self.ended()
    }

    fn wait(&self) {
        self.retire();
    }
}

/// A block's view of its pool's entry: every charge the backend makes for
/// the block passes through it to the entry, which it counts, so the bytes
/// the block holds in its store are exactly what was charged for it,
/// whatever the backend rounded or took beside it.
struct BlockEntry {
    entry: Arc<dyn Entry>,
    charged: AtomicU64,
}

impl Entry for BlockEntry {
    fn store(&self) -> &seismic_topology::StoreIdentity {
        self.entry.store()
    }

    fn change(&self) -> crate::memory::Change {
        self.entry.change()
    }

    fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
        self.entry.on_close(closed);
    }

    fn closed(&self) -> bool {
        self.entry.closed()
    }

    fn take(&self, change: &crate::memory::Change, bytes: u64, wiring: Wiring) -> Result<(), crate::memory::Refused> {
        self.entry.take(change, bytes, wiring)?;
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        Ok(())
    }

    fn took(&self, change: &crate::memory::Change, bytes: u64, wiring: Wiring) {
        self.entry.took(change, bytes, wiring);
        self.charged.fetch_add(bytes, Ordering::AcqRel);
    }

    fn freed(&self, change: &crate::memory::Change, bytes: u64, wiring: Wiring) {
        let before = self.charged.fetch_sub(bytes, Ordering::AcqRel);
        assert!(before >= bytes, "invariant violated: a block frees {bytes} of the {before} bytes charged for it");
        self.entry.freed(change, bytes, wiring);
    }
}

/// One block of a pool: its allocation, and what was charged for it.
struct PoolBlock {
    allocation: Arc<Allocation>,
    charged: Arc<BlockEntry>,
}

impl PoolBlock {
    /// The bytes it holds in its store, as charged to the pool's entry.
    fn held(&self) -> u64 {
        self.charged.charged.load(Ordering::Acquire)
    }
}

/// A pool's blocks: allocations of its store from the backend's plain
/// allocator, in the pool's hazard domain.
#[derive(Clone)]
struct PoolBlocks {
    device: Arc<DeviceInner>,
    /// The kinds of the pool's contract that lie in its store, in placement
    /// order: a block lies along the path of the one it lands in.
    kinds: Vec<u32>,
    access: Arc<AccessGroup>,
}

impl Blocks for PoolBlocks {
    type Block = PoolBlock;
    type Fence = DeviceUses;
    type Error = ExecutionError;

    fn allocate(&self, bytes: u64, entry: &Arc<dyn Entry>, fill: SlabFill) -> Result<PoolBlock, ExecutionError> {
        let charged = Arc::new(BlockEntry { entry: entry.clone(), charged: AtomicU64::new(0) });
        let through: Arc<dyn Entry> = charged.clone();
        let allocation = self.device.allocate_block(&self.kinds, bytes, &through, self.access.clone(), fill)?;
        let address = self.device.allocation_address(&allocation, 0);
        assert!(
            address % BLOCK_ALIGNMENT == 0,
            "invariant violated: a native allocation at {address:#x} is not {BLOCK_ALIGNMENT}-byte aligned"
        );
        Ok(PoolBlock { allocation, charged })
    }

    fn held(&self, block: &PoolBlock) -> u64 {
        block.held()
    }

    fn address(&self, block: &PoolBlock) -> u64 {
        self.device.allocation_address(&block.allocation, 0)
    }
}

/// Who holds a granule: a store's slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Holder {
    store: u64,
    slot: usize,
}

/// The charges a pool's memory is made through, both of one device: its
/// blocks, whose granules its slabs claim and release, through `blocks`;
/// its absent granule, which no slab holds and the pool holds for its life,
/// through `absent`. A ledger that counts what the blocks' entry holds
/// beyond its slabs' granules as granules a claim can take holds the absent
/// granule apart, so it is never counted as one.
#[derive(Clone)]
pub struct PoolCharges {
    blocks: Charges,
    absent: Charges,
}

impl PoolCharges {
    /// Every byte of the pool made through `charges`: a tally's or a
    /// fixture's, which counts what the pool holds without telling its
    /// granules apart.
    pub fn one(charges: &Charges) -> Self {
        Self { blocks: charges.clone(), absent: charges.clone() }
    }

    /// The pool's blocks made through `blocks`, its absent granule through
    /// `absent`.
    ///
    /// # Panics
    /// Where they are charges of two devices.
    pub fn apart(blocks: Charges, absent: Charges) -> Self {
        absent.of_device(blocks.device());
        Self { blocks, absent }
    }
}

/// One store's granule pool on a device. Clones name the same pool.
#[derive(Clone)]
pub struct Pool(Arc<PoolShared>);

/// A weak handle to a granule pool ([`Pool::downgrade`]), used by reclaimers
/// so the store's ledger holds no strong reference to the pool.
#[derive(Clone)]
pub struct WeakPool(Weak<PoolShared>);

impl WeakPool {
    pub fn upgrade(&self) -> Option<Pool> {
        self.0.upgrade().map(Pool)
    }
}

impl Pool {
    pub fn downgrade(&self) -> WeakPool {
        WeakPool(Arc::downgrade(&self.0))
    }
}

struct PoolShared {
    device: Arc<DeviceInner>,
    contract: Contract,
    blocks: PoolBlocks,
    entry: Arc<dyn Entry>,
    granule: u64,
    block_bytes: u64,
    /// One zero granule no slab holds, which every free slot of the pool's
    /// stores names, in the pool's hazard domain (as every block and table
    /// of the pool is).
    absent: PoolBlock,
    state: Mutex<PoolState>,
    grow_gate: seismic_memory::pool::GrowGate,
}

struct PoolState {
    granules: GranulePool<PoolBlocks, Holder>,
    /// The stores claiming from the pool, by identity: whose tables a
    /// vacate repoints.
    stores: HashMap<u64, Weak<Slots>>,
    /// Blocks out of the pool, every granule free, each with the pool's
    /// uses when it left: released as those end ([`Pool::return_empty`]).
    releasing: Vec<(PoolBlock, Arc<DeviceUses>)>,
    /// Number of [`Pool::return_empty`] calls currently dropping released
    /// blocks outside the pool lock.
    dropping: usize,
    /// Whether a thread waits on released granules' and releasing blocks'
    /// flights ([`Pool::await_owed`]): one at a time.
    waiting: bool,
    /// Callbacks registered via [`Pool::wake_on_release`], drained and fired
    /// outside the pool lock once `releasing` is empty, `dropping == 0`, and
    /// `waiting` is false.
    wakers: Vec<Box<dyn FnOnce() + Send>>,
    #[cfg(test)]
    waiters: Vec<std::thread::JoinHandle<()>>,
}

impl PoolState {
    fn is_releasing(&self) -> bool {
        !self.releasing.is_empty() || self.waiting || self.dropping > 0
    }
}

impl PoolShared {
    fn state(&self) -> MutexGuard<'_, PoolState> {
        seismic_memory::assert_blocking_allocation_allowed("PoolShared::state");
        self.state.lock().expect("granule pool lock poisoned")
    }

    /// Allocate one pool block outside all pool and slab locks.
    fn allocate_block(&self, fill: SlabFill) -> Result<PoolBlock, ExecutionError> {
        seismic_memory::assert_blocking_allocation_allowed("PoolShared::allocate_block");
        self.blocks.allocate(self.block_bytes, &self.entry, fill)
    }

    /// Perform one single-flight growth step outside all pool and slab locks.
    /// Returns `Ok(true)` if this thread allocated and installed a block, or
    /// `Ok(false)` if it waited for another thread's growth to finish.
    fn grow_once(&self, fill: SlabFill) -> Result<bool, ExecutionError> {
        match self.grow_gate.enter_or_wait() {
            seismic_memory::pool::GrowTurn::Waited => Ok(false),
            seismic_memory::pool::GrowTurn::Grow(_guard) => {
                let block = self.allocate_block(fill)?;
                self.state().granules.install(block, fill);
                Ok(true)
            }
        }
    }

    /// The newest uses of the pool's hazard domain: what a granule released
    /// now waits for.
    fn uses(&self) -> DeviceUses {
        self.absent.allocation.last_device_uses()
    }
}

impl Pool {
    /// A new pool of the store `contract` places in on `charges`' device,
    /// of `granule`-byte granules in `block_bytes`-byte blocks
    /// (`seismic_memory::pool::block_bytes`), made through `charges` (its
    /// blocks through one, its absent granule, allocated now, through the
    /// other), each charged to their entry of that store, in the kind they
    /// are confined to where they are, holding no block until it grows. Its
    /// owner decides who shares it, by its clones: a load opens each of its
    /// stores' pools once and hands it to every store that claims from it,
    /// while each load, tuning or qualification fixture on the device holds
    /// pools of its own, at its own granule, beside them. Refused for a
    /// block of no whole granules, or charges confined to a kind of none of
    /// the store's.
    ///
    /// # Panics
    /// Through charges that allocate nothing in that store
    /// ([`Charges::none`]): a pool's granules are bytes.
    pub fn open(charges: &PoolCharges, contract: Contract, granule: u64, block_bytes: u64) -> Result<Self, TensorError> {
        let refused = |why: String| TensorError::SlabLayout(why);
        let device = charges.blocks.device();
        let topology = device.topology();
        let store = topology
            .first_store(device.info(), contract)
            .ok_or_else(|| refused(format!("{contract:?} places in no store of the device")))?;
        let made = charges.blocks.of(contract, granule)?;
        let absent = charges.absent.of(contract, granule)?;
        assert_eq!(made.kinds, absent.kinds, "invariant violated: a pool's absent granule made in other kinds than its blocks");
        let (entry, absent) = (made.entry.clone(), absent.entry.clone());
        if granule == 0
            || granule % BLOCK_ALIGNMENT != 0
            || block_bytes < granule
            || block_bytes % granule != 0
            || block_bytes / granule > u64::from(u32::MAX)
        {
            return Err(refused(format!(
                "a pool block of {block_bytes} bytes is no whole number of {granule}-byte granules, \
                 each a multiple of {BLOCK_ALIGNMENT} bytes"
            )));
        }
        let kinds = device
            .info()
            .placement(contract)
            .iter()
            .map(|&id| topology.kind(id).expect("a placement names kinds of its revision"))
            .filter(|kind| {
                kind.paths()
                    .first()
                    .and_then(|&path| topology.access_path(path))
                    .is_some_and(|path| path.store() == store)
            })
            .map(|kind| kind.native().id)
            .filter(|kind| made.kinds.contains(kind))
            .collect::<Vec<_>>();
        if kinds.is_empty() {
            return Err(refused(format!("a pool of {contract:?} made through charges confined to none of its store's kinds {:?}", made.kinds)));
        }
        let access = Arc::new(AccessGroup::default());
        let blocks = PoolBlocks { device: device.clone(), kinds, access: access.clone() };
        let absent = blocks.allocate(granule, &absent, SlabFill::Zeroed)?;
        let shared = Arc::new(PoolShared {
            device: device.clone(),
            contract,
            blocks: blocks.clone(),
            entry: entry.clone(),
            granule,
            block_bytes,
            absent,
            state: Mutex::new(PoolState {
                granules: GranulePool::new(blocks, granule, block_bytes),
                stores: HashMap::new(),
                releasing: Vec::new(),
                dropping: 0,
                waiting: false,
                wakers: Vec::new(),
                #[cfg(test)]
                waiters: Vec::new(),
            }),
            grow_gate: seismic_memory::pool::GrowGate::new(),
        });
        Ok(Self(shared))
    }

    /// How `device`'s backend makes its native allocations, as discovery
    /// reports them (`seismic_topology::Limits::allocations`): what a
    /// pool's blocks are sized by (`seismic_memory::pool::block_bytes`).
    pub fn native_allocations(device: &Arc<DeviceInner>) -> NativeAllocations {
        device.info.limits().allocations()
    }

    pub fn granule(&self) -> u64 {
        self.0.granule
    }

    pub fn block_bytes(&self) -> u64 {
        self.0.block_bytes
    }

    /// The bytes the pool holds in its store: its blocks, those out of it
    /// not yet released among them, and its absent granule, as charged to
    /// its entries.
    pub fn held_bytes(&self) -> u64 {
        let state = self.0.state();
        state.granules.held_bytes() + state.releasing.iter().map(|(block, _)| block.held()).sum::<u64>() + self.0.absent.held()
    }

    /// Granules claimable now, in process.
    pub fn free_granules(&self) -> usize {
        self.0.state().granules.free_granules()
    }

    /// Bytes freed for outside demand not yet returned to the OS or driver.
    pub fn owed(&self) -> u64 {
        self.0.state().granules.owed()
    }

    /// Whether the pool has blocks awaiting flight completion before dropping
    /// to the OS/driver, blocks currently dropping outside the pool lock, or
    /// an `await_owed` waiter in flight.
    pub fn releasing(&self) -> bool {
        self.0.state().is_releasing()
    }

    /// Register a one-shot callback to fire once every block currently
    /// releasing, dropping, or awaited by [`Self::await_owed`] has dropped to
    /// the OS/driver. Fires immediately outside the pool lock if nothing is
    /// releasing, dropping, or awaited.
    pub fn wake_on_release(&self, wake: impl FnOnce() + Send + 'static) {
        let immediate = {
            let mut state = self.0.state();
            if !state.is_releasing() {
                Some(wake)
            } else {
                state.wakers.push(Box::new(wake));
                None
            }
        };
        if let Some(wake) = immediate {
            wake();
        }
    }

    /// Drain and fire registered release wakers outside the pool lock when no
    /// blocks remain in `releasing`, none are dropping off-lock, and no
    /// `await_owed` waiter is running.
    fn fire_release_wakers(&self) {
        let wakers = {
            let mut state = self.0.state();
            if !state.is_releasing() {
                std::mem::take(&mut state.wakers)
            } else {
                Vec::new()
            }
        };
        for wake in wakers {
            wake();
        }
    }

    /// Block until `releasing` is empty, `dropping == 0`, and `waiting` is false.
    fn wait_release_idle(&self) {
        let done = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let signal = done.clone();
        self.wake_on_release(move || {
            let (lock, cond) = &*signal;
            *lock.lock().expect("release wait lock poisoned") = true;
            cond.notify_all();
        });
        let (lock, cond) = &*done;
        let mut finished = lock.lock().expect("release wait lock poisoned");
        while !*finished {
            finished = cond.wait(finished).expect("release wait lock poisoned");
        }
    }

    /// Allocate one block, admitted by the pool's entry first: the pool's
    /// grow step, allocated outside the pool's lock. Refused
    /// (`ExecutionError::AllocationCapacity` where the entry refuses), nothing
    /// changes.
    pub fn grow(&self) -> Result<(), TensorError> {
        while !self.0.grow_once(SlabFill::Zeroed).map_err(TensorError::Execution)? {}
        Ok(())
    }

    /// Take every block whose granules are all free out of the pool: the
    /// bytes they hold, returned to the OS or driver. No work in flight
    /// reaches such a block, but its storage's release waits for every use
    /// of the pool's hazard domain, which it shares: a block that left with
    /// none in flight is released now, the others once the work in flight
    /// when they left ends, from its completion ([`Self::await_owed`]). So
    /// this never waits for device work; claims and releases of other
    /// granules go on meanwhile.
    fn return_empty(&self) -> u64 {
        // The uses before the blocks leave: work submitted after reaches
        // no granule of theirs, every one free.
        let uses = Arc::new(self.0.uses());
        let (bytes, released) = {
            let mut state = self.0.state();
            let returned = state.granules.return_empty();
            let bytes: u64 = returned.iter().map(|(_, held)| held).sum();
            state.releasing.extend(returned.into_iter().map(|(block, _)| (block, uses.clone())));
            let (released, releasing): (Vec<_>, Vec<_>) =
                std::mem::take(&mut state.releasing).into_iter().partition(|(_, uses)| uses.ended());
            state.releasing = releasing;
            if !released.is_empty() {
                state.dropping += 1;
            }
            (bytes, released)
        };
        if !released.is_empty() {
            // Released outside the pool's lock, behind work that has ended.
            // Keep `dropping > 0` until `released` has physically dropped so
            // `releasing()` and `wake_on_release()` never observe idle early.
            struct DropGuard<'a>(&'a Pool);
            impl Drop for DropGuard<'_> {
                fn drop(&mut self) {
                    let wakers = {
                        let mut state = self.0 .0.state();
                        state.dropping -= 1;
                        if !state.is_releasing() {
                            std::mem::take(&mut state.wakers)
                        } else {
                            Vec::new()
                        }
                    };
                    for wake in wakers {
                        wake();
                    }
                }
            }
            let _guard = DropGuard(self);
            drop(released);
        } else {
            self.fire_release_wakers();
        }
        bytes
    }

    fn return_freed_inner(&self, freed: u64, spawn_waiter: bool) -> Result<u64, TensorError> {
        self.0.state().granules.owe(freed);
        let mut returned = self.return_empty();
        let mut skip = std::collections::BTreeSet::new();
        loop {
            let owed = self.0.state().granules.owed() > 0;
            if !owed {
                break;
            }
            match self.vacate_emptiest(&mut skip)? {
                VacateStep::Vacated(bytes) => returned += bytes,
                VacateStep::Skipped => continue,
                VacateStep::None => break,
            }
        }
        // What emptied meanwhile: a vacated block whose copies completed as
        // they were submitted, its bytes counted already, and any other.
        returned += self.return_empty();
        if spawn_waiter {
            self.await_owed();
        }
        Ok(returned)
    }

    /// Return `freed` bytes to the OS or driver, what a shrink of the
    /// pool's holders freed for outside demand, with whatever the pool
    /// still owes: every empty block first ([`Self::return_empty`]), then,
    /// while what is owed exceeds the granules still awaiting the flight
    /// that read them, the emptiest block emptied ([`Self::vacate_emptiest`])
    /// until nothing is owed past them or the emptiest block cannot be
    /// emptied, then whatever emptied meanwhile. A granule awaiting its flight is no shortfall: it returns as
    /// that flight ends, from its completion ([`Self::await_owed`]), so no
    /// copy, and no wait for the flight, is spent on it. Only this call
    /// vacates: the yield that asks for the bytes is the one that enqueues
    /// the copies. It never waits for device work (but on Metal, whose
    /// copies the host makes once the pool's work has ended,
    /// `native::moves`).
    ///
    /// The bytes returned: of the blocks out of the pool, each released to
    /// the OS now or as the work in flight when it left ends, and of the
    /// blocks emptied, each released as its slabs' copies complete. No later
    /// call returns them again; the pool's entries are charged for each until
    /// it is released, so its ledger shows the return then.
    pub fn return_freed(&self, freed: u64) -> Result<u64, TensorError> {
        self.return_freed_inner(freed, true)
    }

    /// Synchronously return `freed` bytes to the OS or driver, waiting on any
    /// in-flight fences that pin freed granules or vacated blocks so every
    /// returned block is physically dropped and its ledger entry freed before
    /// this call returns.
    pub fn return_freed_sync(&self, freed: u64) -> Result<u64, TensorError> {
        let mut returned = match self.return_freed_inner(freed, false) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.await_owed();
                return Err(error);
            }
        };
        loop {
            let fences = {
                let mut state = self.0.state();
                let mut fences = state.granules.awaited().unwrap_or_default();
                fences.extend(state.releasing.iter().map(|(_, uses)| uses.clone()));
                fences
            };
            if fences.is_empty() {
                break;
            }
            for fence in &fences {
                fence.wait();
            }
            returned += self.return_empty();
            let more = self.0.state().granules.owed() > 0;
            if more {
                returned += match self.return_freed_inner(0, false) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        self.await_owed();
                        return Err(error);
                    }
                };
            }
        }
        self.wait_release_idle();
        Ok(returned)
    }

    /// Where blocks wait on flights to return, return them as those flights
    /// end: a thread of its own blocks on them
    /// ([`seismic_memory::pool::after`]), woken by their completion, and
    /// returns what emptied and what may be released ([`Self::return_empty`]),
    /// then waits again on what is still in flight. It waits on the
    /// granules released behind a flight where the pool owes the OS bytes,
    /// on those of a block emptied for it whatever is owed, and on the work
    /// a block out of the pool waits for. It never vacates: a copy out runs
    /// only inside the yield that asked for the bytes
    /// ([`Self::return_freed`]), never in the background mid-decode, so a
    /// debt that emptied blocks do not pay waits for the next yield. One
    /// such thread at a time; with nothing in flight to wait on, none runs.
    fn await_owed(&self) {
        let fences = {
            let mut state = self.0.state();
            if state.waiting {
                return;
            }
            let mut fences = state.granules.awaited().unwrap_or_default();
            fences.extend(state.releasing.iter().map(|(_, uses)| uses.clone()));
            if fences.is_empty() {
                return;
            }
            state.waiting = true;
            fences
        };
        self.spawn_await_owed(fences);
    }

    fn spawn_await_owed(&self, fences: Vec<Arc<DeviceUses>>) {
        let pool = self.clone();
        let handle = seismic_memory::pool::after(fences, move || {
            // Keep `waiting = true` across `return_empty()` and the re-check
            // so `wake_on_release()` / `releasing()` never observe a gap before
            // newly emptied blocks have physically dropped.
            pool.return_empty();
            let next = {
                let mut state = pool.0.state();
                let mut fences = state.granules.awaited().unwrap_or_default();
                fences.extend(state.releasing.iter().map(|(_, uses)| uses.clone()));
                if fences.is_empty() {
                    state.waiting = false;
                    let wakers = if !state.is_releasing() {
                        std::mem::take(&mut state.wakers)
                    } else {
                        Vec::new()
                    };
                    Err(wakers)
                } else {
                    Ok(fences)
                }
            };
            match next {
                Ok(fences) => pool.spawn_await_owed(fences),
                Err(wakers) => {
                    for wake in wakers {
                        wake();
                    }
                }
            }
        });
        #[cfg(test)]
        self.0.state().waiters.push(handle);
        #[cfg(not(test))]
        drop(handle);
    }

    #[cfg(test)]
    fn join_waiters(&self) {
        loop {
            let waiters = std::mem::take(&mut self.0.state().waiters);
            if waiters.is_empty() {
                break;
            }
            for handle in waiters {
                handle.join().unwrap();
            }
        }
    }

    /// Empty the block with the fewest live granules for outside demand:
    /// each live slab is moved into a free granule of another block, its
    /// copy and its store's repoint device work ordered after the work
    /// submitted before and before the work submitted after
    /// ([`Slots::adopt`], `native::moves`), one submission a store, and its
    /// old granule released behind them. The block is then returning
    /// (`GranulePool::vacated`): it returns as those copies complete
    /// ([`Self::await_owed`]), and its bytes, returned, pay what is owed now.
    /// None where no block can be emptied (too few granules free
    /// elsewhere), or the emptiest did not empty: where a slab stays (a view
    /// holds it, or its store's moves were refused) the block serves again,
    /// the slabs moved out staying at their new granules, the rest where
    /// they were; a refusal's error returns then.
    fn vacate_emptiest(&self, skip: &mut std::collections::BTreeSet<BlockId>) -> Result<VacateStep, TensorError> {
        let pool = &self.0;
        let vacating = {
            let mut state = pool.state();
            let Some(block) = state.granules.emptiest_excluding(skip) else {
                return Ok(VacateStep::None);
            };
            let Some(vacating) = state.granules.vacate(block) else {
                return Ok(VacateStep::None);
            };
            vacating
        };
        let block: BlockId = vacating.block();
        let mut by_store = std::collections::BTreeMap::<u64, Vec<Move<Holder>>>::new();
        for moved in vacating.moves() {
            by_store.entry(moved.holder.store).or_default().push(*moved);
        }
        let (mut failure, mut stays) = (None, false);
        for (store, moved) in by_store {
            let slots = pool.state().stores.get(&store).and_then(Weak::upgrade);
            let adopted = match &slots {
                Some(slots) => slots.adopt(&moved),
                None => Ok(vec![Adopted::Left; moved.len()]),
            };
            match adopted {
                Ok(adopted) => {
                    for (moved, adopted) in moved.iter().zip(adopted) {
                        match adopted {
                            Adopted::Moved => {}
                            // Its new granule goes back free.
                            Adopted::Left => pool.state().granules.release(moved.to, None),
                            Adopted::Viewed => {
                                pool.state().granules.release(moved.to, None);
                                stays = true;
                            }
                        }
                    }
                }
                // Nothing of the store's moves ran: their destinations go
                // back free, their slabs where they were.
                Err(error) => {
                    let mut state = pool.state();
                    for moved in &moved {
                        state.granules.release(moved.to, None);
                    }
                    failure = Some(error);
                    stays = true;
                }
            }
        }
        if stays {
            // A slab stayed in the block: it serves again, the slabs moved
            // out staying at their new granules. Skip this block in this
            // return pass so another block can still vacate.
            pool.state().granules.unvacate(vacating);
            skip.insert(block);
            match failure {
                Some(error) => Err(error),
                None => Ok(VacateStep::Skipped),
            }
        } else {
            debug_assert!(failure.is_none());
            // A concurrent `region_view` during `Slots::adopt`'s unlocked
            // submission phase (or a detached view on a freed slot) may still
            // hold a source granule's `SlabHold`, keeping that granule
            // `Granule::Held` until the view drops.
            let mut state = pool.state();
            match state.granules.try_vacated(vacating) {
                Ok(bytes) => Ok(VacateStep::Vacated(bytes)),
                Err(vacating) => {
                    state.granules.unvacate(vacating);
                    skip.insert(block);
                    Ok(VacateStep::Skipped)
                }
            }
        }
    }
}

/// Outcome of one [`Pool::vacate_emptiest`] attempt.
enum VacateStep {
    /// A block was vacated and marked returning; its held bytes.
    Vacated(u64),
    /// The emptiest block had a viewed slab and was skipped; try another.
    Skipped,
    /// No block can be vacated now.
    None,
}

/// A granule claimed for a slab, held while its slab or any view of it
/// lives ([`SlabHold`]): as the last drops it leaves the slab, free to claim
/// once the pool's work submitted by then completes. So a freed slab's bytes
/// outlive every view that could still reach them.
struct GranuleHold {
    pool: Arc<PoolShared>,
    granule: GranuleId,
}

/// A slab's place, held by the slab and by every view of its regions
/// (`TensorInner::in_slab`): its store's slot, and its granule where it is
/// a pool's. A view's bytes are that slot's slab's, which no other store
/// holds, so binding's no-alias check finds the slot a view lies in without
/// a search (`api::tensor::Reaching`).
pub(crate) struct SlabHold {
    /// The store's identity ([`Slots::id`]).
    store: u64,
    slot: usize,
    granule: Option<GranuleHold>,
}

impl SlabHold {
    /// The identity of the store whose slot holds the slab.
    pub(crate) fn store(&self) -> u64 {
        self.store
    }

    /// The slot that holds the slab, as long as it is not freed.
    pub(crate) fn slot(&self) -> usize {
        self.slot
    }
}

impl Drop for GranuleHold {
    /// Released behind the pool's newest uses; where the pool owes the OS
    /// bytes, they return as those uses end ([`Pool::await_owed`]).
    fn drop(&mut self) {
        let uses = Arc::new(self.pool.uses());
        self.pool.state().granules.release(self.granule, Some(uses));
        Pool(self.pool.clone()).await_owed();
    }
}

pub use seismic_memory::SlabFill;

/// How a store's slabs are laid out.
#[derive(Clone, Debug)]
pub struct SlabGeometry {
    pub rows_per_slab: u64,
    pub logical_rows: u64,
    pub capacity: usize,
    /// The bytes of one slab, its regions laid out within it.
    pub slab_bytes: u64,
    /// The least multiple of every region's alignment.
    pub slab_alignment: u64,
    /// Each region: its representation, the shape of one of its rows, and
    /// its byte offset in a slab.
    pub regions: Vec<(RepresentationId, Vec<u64>, u64)>,
}

/// One slot's slab.
struct Slab {
    /// Each region's allocation and byte offset in it.
    regions: Vec<(Arc<Allocation>, u64)>,
    /// Its place in the store, and its granule where it is a pool's: not
    /// for a lent slab, nor a copy's ([`Slots::copy`]).
    hold: Arc<SlabHold>,
    /// Monotonic count of [`Slots::region_view`] calls on this slab, so
    /// [`Slots::adopt`] detects any view created (and possibly written and
    /// dropped) during its unlocked copy phase.
    views: u64,
    /// True during the brief table-repoint phase of [`Slots::adopt`] for
    /// confirmed slots only, preventing a new [`Slots::region_view`] from
    /// observing `from` while `table[slot]` is being repointed to `to`.
    adopting: bool,
    /// Completion of the `MoveStep::Copy` that populated an adopted slab,
    /// waited on outside `self.slabs()` by the first [`Slots::region_view`] so
    /// an unordered transfer queue cannot write to `to` while the compute
    /// queue's `Copy(from -> to)` is still in flight.
    copy_done: Option<Arc<dyn DeviceCompletion>>,
}

impl Slab {
    fn new(regions: Vec<(Arc<Allocation>, u64)>, hold: Arc<SlabHold>) -> Self {
        Self { regions, hold, views: 0, adopting: false, copy_done: None }
    }

    fn granule(&self) -> Option<GranuleId> {
        self.hold.granule.as_ref().map(|hold| hold.granule)
    }
}

/// What a slot did with a granule moved out of a vacating block
/// ([`Slots::adopt`]).
#[derive(Clone, Copy)]
enum Adopted {
    /// Its copy and repoint enqueued: the old granule is released behind
    /// them.
    Moved,
    /// The slot holds another granule now, or none: the old one left it.
    Left,
    /// A view of the slab holds the old granule, which stays where it is.
    Viewed,
}

/// A store's slots and its address table, shared by the store, the views of
/// its regions and its pool.
pub(crate) struct Slots {
    id: u64,
    device: Arc<DeviceInner>,
    /// The pool its slabs are granules of; none for a copy.
    pool: Option<Arc<PoolShared>>,
    /// A record table of `regions × capacity` entries, region-major
    /// ([`Self::entry`]): entry `(region, slot)` the device address of that
    /// region of that slot's slab, or of the absent slab. Kernels bind a
    /// region's entries as `[capacity, 2]` u32 ([`Self::region_bytes`]).
    table: Arc<Allocation>,
    capacity: usize,
    slab_bytes: u64,
    /// Each region's byte offset in a granule.
    offsets: Vec<u64>,
    /// Per region, a globally unique synthetic allocation identity
    /// (`fresh_allocation_identity`), disjoint from every physical
    /// `Allocation::identity()` and from every other `(Slots, region)`.
    #[cfg(any(test, target_os = "macos"))]
    region_ids: Vec<u64>,
    /// The device address every free slot names: a whole slab's bytes.
    absent: u64,
    /// The leading slots whose slabs are lent, never freed.
    lent: usize,
    /// Per region, where its lent slabs lie: the allocation they are lent
    /// from, each in its own hazard domain, and the byte offset of slot 0's
    /// region in it, slot `s`'s `s` of the region's slab bytes after; empty
    /// for a store that lends none.
    lent_regions: Vec<(Arc<Allocation>, u64)>,
    /// The table's generation: bumped inside `self.slabs()` after every
    /// mutation (`add_slabs`, `free_slab`, `adopt`) so `SlabView::reach()`
    /// never caches stale slabs under a new generation.
    generation: AtomicU64,
    /// Serializes mutations (`add_slabs`, `free_slab`, `adopt`) so `slabs`
    /// stays a leaf lock for `region_view` / `backed` / `backed_count` / `reach`
    /// and is never held across `pool.state()`, `granules.grow()`,
    /// `repointed()`, `moves::submit()`, or `point_many()`.
    mutate: Mutex<()>,
    slabs: Mutex<Vec<Option<Slab>>>,
    /// Notifies [`Self::region_view`] callers waiting for a confirmed slot's
    /// brief table-repoint phase in [`Self::adopt`] to finish.
    adopt_done: std::sync::Condvar,
    /// The device work that last wrote the table ([`Self::adopt`]), while
    /// it may still run: a later repoint is ordered after it, never a host
    /// write it could land over. Changed under the mutate lock.
    repointed: Mutex<Option<Arc<dyn DeviceCompletion>>>,
}

impl Slots {
    fn mutate(&self) -> MutexGuard<'_, ()> {
        seismic_memory::assert_blocking_allocation_allowed("Slots::mutate");
        self.mutate.lock().expect("slab store mutate lock poisoned")
    }

    fn slabs(&self) -> MutexGuard<'_, Vec<Option<Slab>>> {
        self.slabs.lock().expect("slab store lock poisoned")
    }

    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }

    /// The store's identity, which its slabs' holds name
    /// ([`SlabHold::store`]).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The unique logical allocation identity for region `region` of this store.
    #[cfg(any(test, target_os = "macos"))]
    pub(crate) fn region_identity(&self, region: usize) -> u64 {
        self.region_ids[region]
    }

    /// The leading slots whose slabs are lent: `0..lent`.
    pub(crate) fn lent(&self) -> usize {
        self.lent
    }

    /// Where region `region`'s lent slabs lie: the allocation they are lent
    /// from and slot 0's byte offset in it, each next slot's a slab of the
    /// region after; none where the store lends none.
    pub(crate) fn lent_region(&self, region: usize) -> Option<(&Arc<Allocation>, u64)> {
        self.lent_regions.get(region).map(|(lender, offset)| (lender, *offset))
    }

    pub(crate) fn table(&self) -> &Arc<Allocation> {
        &self.table
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// A table of `regions × capacity` entries in `access`, of `contract`
    /// through `charges`: record-table storage, aligned to its entry, as every
    /// reader of an entry (a native kernel's, [`Self::point_many`]'s atomic
    /// store) reads it whole.
    fn allocate_table(
        charges: &Charges,
        contract: Contract,
        regions: usize,
        capacity: usize,
        access: Option<Arc<AccessGroup>>,
    ) -> Result<Arc<Allocation>, TensorError> {
        let bytes = (regions as u64)
            .checked_mul(capacity as u64)
            .and_then(seismic_lang::registry::record_table_bytes)
            .ok_or_else(|| TensorError::SlabLayout("a slab table's bytes overflow".into()))?;
        Ok(charges.allocate_contract(contract, bytes, seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES, access)?)
    }

    /// Where entry `(region, slot)` lies in the table, in bytes.
    pub(crate) fn entry(&self, region: usize, slot: usize) -> u64 {
        seismic_lang::registry::record_table_entry((region * self.capacity + slot) as u64)
    }

    /// The bytes of one region's entries.
    pub(crate) fn region_bytes(&self) -> u64 {
        self.entry(1, 0)
    }

    /// Where region `region`'s entries start in the table, in bytes.
    pub(crate) fn region_offset(&self, region: usize) -> u64 {
        self.entry(region, 0)
    }

    /// The allocations device work bound to a view of the store reaches
    /// besides the table's hazard domain, which every pooled slab shares:
    /// the lenders', each once.
    pub(crate) fn lenders(&self) -> impl Iterator<Item = &Arc<Allocation>> {
        let lent = &self.lent_regions;
        lent.iter()
            .enumerate()
            .filter(move |(region, (lender, _))| !lent[..*region].iter().any(|(earlier, _)| Arc::ptr_eq(earlier, lender)))
            .map(|(_, (lender, _))| lender)
    }

    /// The table's generation now: a view's reach read after it shows
    /// every slab change before it ([`Self::point`]).
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Point each `(slot, base_address)` pair's entries, region by region, at
    /// `base_address + self.offsets[region]`: one 8-byte store each, which
    /// device work in flight reads whole, old or new, and which never waits for
    /// device work. A host store where the host addresses the table and no
    /// device write of it may still run ([`Self::adopt`]), which the host store
    /// could land before; else one batched submission of ordered words
    /// (`native::moves`), after the work submitted before and before the work
    /// submitted after. Never a host write through staging (`Storage::write` of
    /// storage the host does not address): on a backend whose staging copy is a
    /// submission to the device's one queue (Vulkan), the host would wait
    /// behind every flight in flight, and behind a flight parked on a gate, for
    /// the gate, which the writing thread may be the one to answer.
    fn point_many(&self, updates: &[(usize, u64)]) -> Result<(), ExecutionError> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut repointed = self.repointed();
        let pending = repointed.as_ref().is_some_and(|written| matches!(written.completed(), Ok(false)));
        match self.table.storage().host_address().filter(|_| !pending) {
            Some(base) => {
                for &(slot, base_address) in updates {
                    for (region, &region_offset) in self.offsets.iter().enumerate() {
                        let offset = self.entry(region, slot);
                        // SAFETY: an 8-byte-aligned entry inside the table,
                        // which lives as `self` does; device work only reads it.
                        unsafe {
                            AtomicU64::from_ptr(base.as_ptr().add(offset as usize).cast())
                                .store(base_address + region_offset, Ordering::Release);
                        }
                    }
                }
            }
            None => {
                let mut words = Vec::with_capacity(updates.len() * self.offsets.len());
                for &(slot, base_address) in updates {
                    for (region, &region_offset) in self.offsets.iter().enumerate() {
                        words.push(MoveStep::Word {
                            destination: &self.table,
                            offset: self.entry(region, slot),
                            value: base_address + region_offset,
                        });
                    }
                }
                *repointed = moves::submit(&self.device, &self.table, &words)?;
            }
        }
        Ok(())
    }

    fn repointed(&self) -> MutexGuard<'_, Option<Arc<dyn DeviceCompletion>>> {
        self.repointed.lock().expect("slab table repoint lock poisoned")
    }

    /// A view of region `region` of slot `slot`'s slab as `representation`
    /// over `extents` (its rows), holding the slab's granule; none for a
    /// free slot.
    pub(crate) fn region_view(
        &self,
        slot: usize,
        region: usize,
        representation: RepresentationId,
        extents: &[u64],
    ) -> Result<Option<TensorInner>, TensorError> {
        let (allocation, offset, hold, copy_done) = {
            let mut slabs = self.slabs();
            loop {
                let Some(Some(slab)) = slabs.get_mut(slot) else {
                    return Ok(None);
                };
                if !slab.adopting {
                    slab.views = slab.views.wrapping_add(1);
                    let (allocation, offset) = slab.regions[region].clone();
                    let hold = slab.hold.clone();
                    let copy_done = match &slab.copy_done {
                        Some(done) if matches!(done.completed(), Ok(true)) => {
                            slab.copy_done = None;
                            None
                        }
                        Some(done) => Some(done.clone()),
                        None => None,
                    };
                    break (allocation, offset, hold, copy_done);
                }
                slabs = self.adopt_done.wait(slabs).expect("slab store lock poisoned");
            }
        };
        if let Some(done) = copy_done {
            done.ended()?;
        }
        TensorInner::in_slab(&self.device, allocation, offset, representation, extents, Some(hold)).map(Some)
    }

    /// The views of region `region` of `count` slots from `first`, as
    /// [`Self::region_view`].
    pub(crate) fn region_views(
        &self,
        region: usize,
        first: usize,
        count: usize,
        representation: RepresentationId,
        extents: &[u64],
    ) -> Result<Vec<Option<Arc<TensorInner>>>, TensorError> {
        (first..first + count)
            .map(|slot| Ok(self.region_view(slot, region, representation, extents)?.map(Arc::new)))
            .collect()
    }

    /// Where region `region` of each slot in `first..first + count` lies now
    /// (`Allocation::identity`, byte offset, `bytes`), `None` a free slot's:
    /// what device work bound to a view of those slots reaches, without
    /// keeping a removed slab's pool block alive inside a cached reach.
    pub(crate) fn reach(
        &self,
        region: usize,
        first: usize,
        count: usize,
        bytes: u64,
    ) -> Vec<Option<(u64, u64, u64)>> {
        self.slabs()[first..first + count]
            .iter()
            .map(|slab| {
                slab.as_ref().map(|slab| {
                    let (allocation, offset) = &slab.regions[region];
                    (allocation.identity(), *offset, bytes)
                })
            })
            .collect()
    }

    /// Take each of `moved`, this store's moves out of a vacating block,
    /// where its slot's slab is still the granule it moves from and nothing
    /// but the slab holds it: copy the slab into its new granule and repoint
    /// the slot's entries, as one submission of device work ordered after
    /// the work submitted before it and before the work submitted after
    /// (`native::moves`), so this never waits for device work (but on
    /// Metal); each old granule is then released behind it, and work in
    /// flight keeps reading and writing the old granules it names, whose
    /// bytes the copy follows. A view of a slab would read and write the old
    /// granule after the copy, so a viewed slab stays where it is. What each
    /// slot did, in order; refused, nothing run and every slab where it was,
    /// where the submission was.
    fn adopt(&self, moved: &[Move<Holder>]) -> Result<Vec<Adopted>, TensorError> {
        let pool = self.pool.as_ref().expect("a store whose granules move is a pool's");
        let _mutate = self.mutate();
        // Phase 1: classify candidate moves under a brief `slabs` leaf lock
        // and snapshot each candidate's `views` counter, then drop `slabs`
        // before touching `pool.state()`, `repointed()`, or `moves::submit()`
        // so concurrent `region_view` calls (for example `expert-requests`
        // answering a parked GPU kernel) never block behind device submission.
        let (mut adopted, recorded_views): (Vec<Adopted>, Vec<u64>) = {
            let slabs = self.slabs();
            moved
                .iter()
                .map(|moved| match slabs.get(moved.holder.slot) {
                    Some(Some(slab)) if slab.granule() == Some(moved.from) => {
                        match Arc::strong_count(&slab.hold) > 1 {
                            true => (Adopted::Viewed, 0),
                            false => (Adopted::Moved, slab.views),
                        }
                    }
                    Some(_) | None => (Adopted::Left, 0),
                })
                .unzip()
        };
        let moving: Vec<usize> = adopted
            .iter()
            .enumerate()
            .filter_map(|(i, status)| matches!(status, Adopted::Moved).then_some(i))
            .collect();
        if moving.is_empty() {
            return Ok(adopted);
        }
        let blocks: Vec<(Arc<Allocation>, Arc<Allocation>)> = {
            let state = pool.state();
            moving
                .iter()
                .map(|&i| {
                    let m = &moved[i];
                    (state.granules.block(m.from.block).allocation.clone(), state.granules.block(m.to.block).allocation.clone())
                })
                .collect()
        };
        // Phase 2 (no `slabs` lock held): submit ONLY `MoveStep::Copy`
        // (`from -> to`), leaving `self.table` and `self.slabs` pointing to
        // `from`. Any concurrent `region_view` during Phase 2 succeeds
        // immediately on `from` (no parked-kernel deadlock) and increments
        // `slab.views`, which Phase 3 detects so no write to `from` can ever
        // be orphaned behind a repointed table entry.
        let move_permit = moves::prepare(&self.device, &self.table)?;
        let at = |granule: GranuleId| u64::from(granule.index) * pool.granule;
        let copies: Vec<MoveStep<'_>> = moving
            .iter()
            .zip(&blocks)
            .map(|(&i, (from, to))| MoveStep::Copy {
                source: from,
                source_offset: at(moved[i].from),
                destination: to,
                destination_offset: at(moved[i].to),
                bytes: self.slab_bytes,
            })
            .collect();
        let copy_written = {
            let _repointed = self.repointed();
            moves::submit_prepared(&self.device, &self.table, &copies, move_permit)?
        };
        // Phase 3 (brief `slabs` leaf lock): confirm which slots were NOT
        // viewed during Phase 2. Any slot viewed during Phase 2 stays at
        // `from` (`Adopted::Viewed`, table untouched); confirmed slots are
        // marked `adopting = true` across the brief Phase 4 table repoint so
        // no `region_view` can slip in and write to `from` while `table[slot]`
        // is being repointed to `to`.
        let mut confirmed: Vec<(usize, Arc<Allocation>)> = Vec::with_capacity(moving.len());
        {
            let mut slabs = self.slabs();
            for (&i, (_, to)) in moving.iter().zip(blocks) {
                let m = &moved[i];
                let untouched = matches!(
                    slabs.get(m.holder.slot),
                    Some(Some(slab))
                        if slab.granule() == Some(m.from)
                            && slab.views == recorded_views[i]
                            && Arc::strong_count(&slab.hold) == 1
                );
                if untouched {
                    slabs[m.holder.slot].as_mut().expect("checked above").adopting = true;
                    confirmed.push((i, to));
                } else {
                    adopted[i] = Adopted::Viewed;
                }
            }
        }
        if confirmed.is_empty() {
            return Ok(adopted);
        }
        // Clear `adopting` and wake waiters if Phase 4 fails or unwinds.
        struct AdoptingGuard<'a> {
            slots: &'a Slots,
            indices: Vec<usize>,
            armed: bool,
        }
        impl Drop for AdoptingGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    let mut slabs = self.slots.slabs();
                    for &slot in &self.indices {
                        if let Some(Some(slab)) = slabs.get_mut(slot) {
                            slab.adopting = false;
                        }
                    }
                    self.slots.adopt_done.notify_all();
                }
            }
        }
        let mut guard = AdoptingGuard {
            slots: self,
            indices: confirmed.iter().map(|&(i, _)| moved[i].holder.slot).collect(),
            armed: true,
        };
        // Phase 4 (no `slabs` lock held): repoint `self.table` for confirmed
        // slots only, ordered after `copy_written` on the same queue.
        let words: Vec<MoveStep<'_>> = confirmed
            .iter()
            .flat_map(|&(i, _)| {
                let m = &moved[i];
                self.offsets.iter().enumerate().map(move |(region, offset)| MoveStep::Word {
                    destination: &self.table,
                    offset: self.entry(region, m.holder.slot),
                    value: m.to_address + offset,
                })
            })
            .collect();
        let words_written = {
            let mut repointed = self.repointed();
            let written = moves::submit(&self.device, &self.table, &words)?;
            *repointed = written.clone().or_else(|| copy_written.clone());
            written
        };
        let _ = words_written;
        // Phase 5 (brief `slabs` leaf lock): install the new slabs with
        // `copy_done = copy_written` (so the first `region_view` waits for
        // `Copy(from -> to)` before an unordered transfer can write to `to`),
        // bump `generation`, clear `adopting`, and wake waiters.
        let previous = {
            let mut slabs = self.slabs();
            let previous = confirmed
                .into_iter()
                .map(|(i, to)| {
                    let m = &moved[i];
                    let adopted = Slab {
                        regions: self.offsets.iter().map(|offset| (to.clone(), at(m.to) + offset)).collect(),
                        hold: Arc::new(SlabHold {
                            store: self.id,
                            slot: m.holder.slot,
                            granule: Some(GranuleHold { pool: pool.clone(), granule: m.to }),
                        }),
                        views: 0,
                        adopting: false,
                        copy_done: copy_written.clone(),
                    };
                    std::mem::replace(slabs[m.holder.slot].as_mut().expect("a confirmed slot is backed"), adopted)
                })
                .collect::<Vec<_>>();
            self.generation.fetch_add(1, Ordering::AcqRel);
            guard.armed = false;
            self.adopt_done.notify_all();
            previous
        };
        // Phase 6 (outside `slabs`): drop old `SlabHold`s. Each old granule
        // releases behind the pool's newest uses (the copy and repoint among
        // them).
        drop(previous);
        Ok(adopted)
    }

    /// A store of fixed slabs, one region of `representation` over
    /// `extents` (a slab's rows) each, holding `images` (`None` a free
    /// slot), in storage of their own: a slab view's copy
    /// (`TensorInner::copy_of`). Its free slots name its first slab.
    pub(crate) fn copy(
        charges: &Charges,
        representation: RepresentationId,
        extents: &[u64],
        images: &[Option<Vec<u8>>],
    ) -> Result<Arc<Self>, TensorError> {
        let device = charges.device();
        let id = crate::driver::fresh_allocation_identity();
        let capacity = images.len();
        let table = Self::allocate_table(charges, Contract::DeviceOnly, 1, capacity, None)?;
        let slabs = images
            .iter()
            .enumerate()
            .map(|(slot, image)| {
                image
                    .as_ref()
                    .map(|bytes| {
                        let slab = TensorInner::from_host_in(charges, representation, extents, bytes, Some(table.access_group().clone()))?;
                        Ok::<_, TensorError>(Slab::new(
                            vec![(slab.allocation().clone(), 0)],
                            Arc::new(SlabHold { store: id, slot, granule: None }),
                        ))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let address = |slab: &Slab| device.allocation_address(&slab.regions[0].0, 0);
        let absent = slabs.iter().flatten().next().map_or(0, address);
        let words = seismic_lang::registry::record_table(slabs.iter().map(|slab| slab.as_ref().map_or(absent, address)));
        // Fresh: nothing uses it yet.
        table.storage().write(0, &words)?;
        Ok(Arc::new(Self {
            id,
            device: device.clone(),
            pool: None,
            table,
            capacity,
            slab_bytes: crate::layout::canonical(representation, extents)?.byte_len,
            offsets: vec![0],
            #[cfg(any(test, target_os = "macos"))]
            region_ids: vec![crate::driver::fresh_allocation_identity()],
            absent,
            lent: 0,
            lent_regions: Vec::new(),
            generation: AtomicU64::new(0),
            mutate: Mutex::new(()),
            slabs: Mutex::new(slabs),
            adopt_done: std::sync::Condvar::new(),
            repointed: Mutex::new(None),
        }))
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.state().stores.remove(&self.id);
        }
        // Each slab's granule leaves once no view of it remains.
    }
}

/// A slab store over a pool: each slab one of its granules, claimed in
/// process, its region addresses in the store's table.
pub struct SlabStore {
    slots: Arc<Slots>,
    geometry: SlabGeometry,
    /// Each region's logical tensor's alias identity.
    aliases: Vec<u64>,
}

impl SlabStore {
    /// A store of `geometry` over `pool`, its leading slabs lent: `lent`
    /// holds one tensor per region (none for a store with no lent slabs),
    /// each a contiguous canonical view on the pool's device of a whole
    /// number of the region's slab rows, all as many slabs, which the store
    /// views for its life and never frees. Every other slot is free. Refused
    /// where a slab is larger than a granule and some slot is not lent, its
    /// alignment is not one every granule has, a region does not lie in the
    /// slab apart from every other, or the lent tensors do not match the
    /// regions or lie in a slab's bytes. Its address table is allocated
    /// through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the pool's.
    pub fn in_pool(charges: &Charges, pool: &Pool, geometry: SlabGeometry, lent: &[Arc<TensorInner>]) -> Result<Self, TensorError> {
        let shared = &pool.0;
        let refused = |why: String| TensorError::SlabLayout(why);
        if geometry.rows_per_slab == 0 {
            return Err(refused("a slab holds at least one row".into()));
        }
        if geometry.slab_alignment == 0 || BLOCK_ALIGNMENT % geometry.slab_alignment != 0 {
            return Err(refused(format!(
                "a slab aligned to {} bytes lies in no granule, each aligned to {BLOCK_ALIGNMENT}",
                geometry.slab_alignment
            )));
        }
        // Each region's bytes of a slab, which lie in it apart from every
        // other's: binding's no-alias check takes two regions of one slab
        // as sharing none (`api::tensor::Reaching`).
        let spans = geometry
            .regions
            .iter()
            .map(|(representation, row_shape, offset)| -> Result<std::ops::Range<u64>, TensorError> {
                let slab_extents = std::iter::once(geometry.rows_per_slab).chain(row_shape.iter().copied()).collect::<Vec<_>>();
                let bytes = crate::layout::canonical(*representation, &slab_extents)?.byte_len;
                offset
                    .checked_add(bytes)
                    .filter(|&end| end <= geometry.slab_bytes)
                    .map(|end| *offset..end)
                    .ok_or_else(|| refused(format!("a slab region of {bytes} bytes at {offset} exceeds its {}-byte slab", geometry.slab_bytes)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if spans.iter().enumerate().any(|(index, span)| spans[..index].iter().any(|earlier| crate::api::tensor::meet(earlier, span))) {
            return Err(refused("two regions of a slab share bytes".into()));
        }
        let device = &shared.device;
        charges.of_device(device);
        let id = crate::driver::fresh_allocation_identity();
        let capacity = geometry.capacity;
        let mut slabs = (0..capacity).map(|_| None).collect::<Vec<Option<Slab>>>();
        let lent_slabs = match lent {
            [] => 0,
            lent if lent.len() != geometry.regions.len() => {
                return Err(refused("a store's lent slabs take one tensor per region".into()))
            }
            lent => {
                let mut count = None;
                for (tensor, (representation, row_shape, _)) in lent.iter().zip(&geometry.regions) {
                    let slab_extents = std::iter::once(geometry.rows_per_slab).chain(row_shape.iter().copied()).collect::<Vec<_>>();
                    let slab = crate::layout::canonical(*representation, &slab_extents)?;
                    let rows = tensor.extents().first().copied().unwrap_or(0);
                    let whole = rows % geometry.rows_per_slab == 0
                        && tensor.representation() == *representation
                        && tensor.extents()[1..] == row_shape[..]
                        && Arc::ptr_eq(tensor.device(), device)
                        && !tensor.is_slab_view()
                        && crate::layout::canonical(*representation, tensor.extents())?.strides == tensor.strides()
                        && tensor.byte_len() == rows / geometry.rows_per_slab * slab.byte_len
                        // A slab's bytes are its store's alone: a view of
                        // one is never lent to another.
                        && tensor.slab_hold().is_none();
                    let slabs_of = rows / geometry.rows_per_slab;
                    if !whole || count.is_some_and(|count| count != slabs_of) {
                        return Err(refused(
                            "a lent tensor is a contiguous canonical view of whole slab rows of its region, \
                             as many slabs as every other's, on the pool's device, of no slab's bytes"
                                .into(),
                        ));
                    }
                    count = Some(slabs_of);
                }
                // A store that lends lends at least one slab, so its lent
                // regions are exactly the bytes it reaches through them.
                let count = usize::try_from(count.unwrap_or(0)).ok().filter(|&count| (1..=capacity).contains(&count)).ok_or_else(|| {
                    refused(format!("a store lends from one slab up to its {capacity} slots"))
                })?;
                for (slot, slab) in slabs.iter_mut().enumerate().take(count) {
                    let regions = lent
                        .iter()
                        .zip(&geometry.regions)
                        .map(|(tensor, (representation, row_shape, _))| {
                            let slab_extents = std::iter::once(geometry.rows_per_slab).chain(row_shape.iter().copied()).collect::<Vec<_>>();
                            let bytes = crate::layout::canonical(*representation, &slab_extents)?.byte_len;
                            Ok::<_, TensorError>((tensor.allocation().clone(), tensor.byte_offset() + slot as u64 * bytes))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    *slab = Some(Slab::new(regions, Arc::new(SlabHold { store: id, slot, granule: None })));
                }
                count
            }
        };
        // A slot not lent is free, naming the pool's absent granule, and is
        // backed by one of its granules: its slab is one.
        if lent_slabs < capacity && geometry.slab_bytes > shared.granule {
            return Err(refused(format!(
                "a slab of {} bytes exceeds the pool's {}-byte granule",
                geometry.slab_bytes, shared.granule
            )));
        }
        let table_contract = if shared.absent.allocation.storage().host_address().is_some()
            && charges.entry(shared.contract).is_some()
            && charges.of(shared.contract, 0).is_ok()
        {
            shared.contract
        } else {
            Contract::DeviceOnly
        };
        let table = Slots::allocate_table(
            charges,
            table_contract,
            geometry.regions.len(),
            capacity,
            Some(shared.absent.allocation.access_group().clone()),
        )?;
        let offsets = geometry.regions.iter().map(|(_, _, offset)| *offset).collect::<Vec<_>>();
        let absent = device.allocation_address(&shared.absent.allocation, 0);
        let slots = Arc::new(Slots {
            id,
            device: device.clone(),
            pool: Some(shared.clone()),
            table,
            capacity,
            slab_bytes: geometry.slab_bytes,
            #[cfg(any(test, target_os = "macos"))]
            region_ids: (0..offsets.len())
                .map(|_| crate::driver::fresh_allocation_identity())
                .collect(),
            offsets,
            absent,
            lent: lent_slabs,
            lent_regions: lent.iter().map(|tensor| (tensor.allocation().clone(), tensor.byte_offset())).collect(),
            generation: AtomicU64::new(0),
            mutate: Mutex::new(()),
            slabs: Mutex::new(Vec::new()),
            adopt_done: std::sync::Condvar::new(),
            repointed: Mutex::new(None),
        });
        let words = seismic_lang::registry::record_table((0..slots.offsets.len()).flat_map(|region| {
            let slots = &slots;
            let slabs = &slabs;
            (0..capacity).map(move |slot| match &slabs[slot] {
                Some(slab) => {
                    let (allocation, offset) = &slab.regions[region];
                    slots.device.allocation_address(allocation, *offset)
                }
                None => slots.absent + slots.offsets[region],
            })
        }));
        // Fresh: nothing uses it yet.
        slots.table.storage().write(0, &words)?;
        *slots.slabs() = slabs;
        shared.state().stores.insert(slots.id, Arc::downgrade(&slots));
        let aliases = (0..geometry.regions.len()).map(|_| crate::driver::fresh_allocation_identity()).collect();
        Ok(Self { slots, geometry, aliases })
    }

    pub fn geometry(&self) -> &SlabGeometry {
        &self.geometry
    }

    /// The slots whose slabs are lent.
    pub fn lent(&self) -> usize {
        self.slots.lent
    }

    /// The backed slots, in order.
    pub fn backed(&self) -> Vec<usize> {
        self.slots.slabs().iter().enumerate().filter_map(|(slot, slab)| slab.as_ref().map(|_| slot)).collect()
    }

    /// The number of backed slots, without allocating.
    pub fn backed_count(&self) -> usize {
        self.slots.slabs().iter().filter(|slab| slab.is_some()).count()
    }

    /// Region `region`'s table: `[capacity, 2]` u32, one 64-bit address an
    /// entry, what a record port binds the region's records through.
    pub fn table(&self, region: usize) -> Result<TensorInner, TensorError> {
        if region >= self.slots.offsets.len() {
            return Err(TensorError::SlabLayout("unknown slab region".into()));
        }
        let u32 = seismic_lang::registry::dense(seismic_lang::types::DType::U32);
        let capacity = self.slots.capacity as u64;
        let layout = crate::layout::canonical(u32, &[capacity, 2])?;
        assert_eq!(layout.byte_len, self.slots.region_bytes(), "invariant violated: a region's table view is its entries");
        Ok(TensorInner::new_view(
            self.slots.device.clone(),
            self.slots.table.clone(),
            self.slots.region_offset(region),
            layout.byte_len,
            u32,
            vec![capacity, 2],
            layout.strides,
        ))
    }

    /// The rows of region `region` of slot `slot`'s slab, holding its
    /// granule; none for a free slot.
    pub fn slab_region(&self, slot: usize, region: usize) -> Result<Option<TensorInner>, TensorError> {
        let (representation, row_shape, _) = self
            .geometry
            .regions
            .get(region)
            .ok_or_else(|| TensorError::SlabLayout("unknown slab region".into()))?;
        let extents = std::iter::once(self.geometry.rows_per_slab).chain(row_shape.iter().copied()).collect::<Vec<_>>();
        self.slots.region_view(slot, region, *representation, &extents)
    }

    /// The logical tensor of region `region`: its rows over every slot,
    /// bound through the region's table, which stays valid as slabs are
    /// added and freed.
    pub fn logical_region(&self, region: usize) -> Result<TensorInner, TensorError> {
        let (representation, row_shape, _) = self
            .geometry
            .regions
            .get(region)
            .ok_or_else(|| TensorError::SlabLayout("unknown slab region".into()))?;
        TensorInner::slabbed(
            &self.slots,
            region,
            *representation,
            self.geometry.rows_per_slab,
            self.geometry.logical_rows,
            row_shape,
            self.aliases[region],
        )
    }

    /// Back the lowest free slot with a granule of the pool, in process
    /// where one is free and otherwise by growing the pool, admitted by its
    /// entry (refused with its capacity error only where the entry refuses
    /// the block), its bytes as `fill` says. Never waits on device work.
    pub fn add_slab(&self, fill: SlabFill) -> Result<usize, TensorError> {
        self.add_slabs(1, fill).map(|slots| slots[0])
    }

    /// Back the `count` lowest free slots with granules of the pool in one
    /// batch, growing the pool outside locks as needed (admitted by its entry),
    /// with bytes as `fill` says. Zeroing of non-fresh device granules and
    /// table repointing are batched into a single submission. On refusal (too
    /// few free slots, or pool growth/submission failure), no slot is backed
    /// and any already-claimed granules are returned cleanly. Never waits on
    /// device work.
    pub fn add_slabs(&self, count: usize, fill: SlabFill) -> Result<Vec<usize>, TensorError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let slots = &self.slots;
        let pool = slots.pool.as_ref().expect("a store that adds slabs is a pool's");
        // Serialize mutations under `mutate` while keeping `slabs` a leaf lock:
        // `region_view` (called by request service to answer parked GPU kernels)
        // only acquires `slabs`, so it never blocks behind `pool.grow_once()`,
        // `moves::submit()`, or `slots.point_many()` (which may wait on
        // `device.native.order` or CUDA driver allocation).
        let _mutate = slots.mutate();
        let free_slots: Vec<usize> = {
            let slabs = slots.slabs();
            slabs
                .iter()
                .enumerate()
                .filter_map(|(slot, slab)| slab.is_none().then_some(slot))
                .take(count)
                .collect()
        };
        if free_slots.len() < count {
            return Err(TensorError::SlabLayout("all slab slots are backed".into()));
        }
        struct ClaimedSlab {
            slot: usize,
            address: u64,
            base: u64,
            fresh: bool,
            block: Arc<Allocation>,
            hold: Arc<SlabHold>,
        }
        // Any already-claimed granules drop their `SlabHold` -> `GranuleHold`
        // if a subsequent grow, zero, or repoint fails, returning them cleanly.
        let mut claimed = Vec::with_capacity(count);
        let mut next = 0;
        while next < free_slots.len() {
            {
                let mut state = pool.state();
                while next < free_slots.len() {
                    let slot = free_slots[next];
                    let holder = Holder { store: slots.id, slot };
                    let Some(Claimed { granule, address, zero: fresh }) = state.granules.take(holder) else {
                        break;
                    };
                    let block = state.granules.block(granule.block).allocation.clone();
                    let hold = Arc::new(SlabHold {
                        store: slots.id,
                        slot,
                        granule: Some(GranuleHold { pool: pool.clone(), granule }),
                    });
                    let base = u64::from(granule.index) * pool.granule;
                    claimed.push(ClaimedSlab { slot, address, base, fresh, block, hold });
                    next += 1;
                }
            }
            if next < free_slots.len() {
                // Neither `slots.slabs()` nor `pool.state()` is held while growing,
                // so no driver call runs under a lock. Growth is single-flight
                // through `PoolShared::grow_once`: concurrent callers wait for the
                // active growth to install and then re-check for a free granule
                // rather than over-admitting a second block.
                pool.grow_once(fill).map_err(TensorError::Execution)?;
            }
        }
        let mut copies = Vec::new();
        if fill == SlabFill::Zeroed {
            // No work in flight reads a claimed granule: its last reader ended.
            // The host zeroes storage it addresses; else the pool's absent
            // granule, zero, is copied over all non-fresh granules in one
            // batched submission together with the table repoint, never a
            // waited staging write.
            for item in &claimed {
                if !item.fresh {
                    match item.block.storage().host_address() {
                        Some(_) => crate::driver::write_zeros(item.block.storage(), item.base, slots.slab_bytes)?,
                        None => copies.push(MoveStep::Copy {
                            source: &pool.absent.allocation,
                            source_offset: 0,
                            destination: &item.block,
                            destination_offset: item.base,
                            bytes: slots.slab_bytes,
                        }),
                    }
                }
            }
        }
        if !copies.is_empty() {
            // Batch the zeroing copies before the table word writes in one
            // ordered submission and record its completion in `repointed` so
            // later host table writes cannot overtake it.
            copies.reserve(claimed.len() * slots.offsets.len());
            for item in &claimed {
                for (region, &region_offset) in slots.offsets.iter().enumerate() {
                    copies.push(MoveStep::Word {
                        destination: &slots.table,
                        offset: slots.entry(region, item.slot),
                        value: item.address + region_offset,
                    });
                }
            }
            let mut repointed = slots.repointed();
            *repointed = moves::submit(&slots.device, &slots.table, &copies)?;
        } else {
            let updates: Vec<(usize, u64)> = claimed.iter().map(|item| (item.slot, item.address)).collect();
            slots.point_many(&updates)?;
        }
        {
            let mut slabs = slots.slabs();
            for item in claimed {
                slabs[item.slot] = Some(Slab::new(
                    slots.offsets.iter().map(|offset| (item.block.clone(), item.base + offset)).collect(),
                    item.hold,
                ));
            }
            slots.generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(free_slots)
    }

    /// Free slot `slot`: its entries name the absent slab from now, and its
    /// granule leaves the store once no view of it remains, claimable once
    /// the pool's work submitted by then completes. Never waits on device
    /// work; work in flight may still read the slab, which stays whole until
    /// it completes. False for a free slot; refused for a lent one.
    pub fn free_slab(&self, slot: usize) -> Result<bool, TensorError> {
        let slots = &self.slots;
        if slot < slots.lent {
            return Err(TensorError::SlabLayout(format!("slab {slot} is lent, never freed")));
        }
        let _mutate = slots.mutate();
        let present = slots.slabs().get(slot).is_some_and(Option::is_some);
        if !present {
            return Ok(false);
        }
        slots.point_many(&[(slot, slots.absent)])?;
        let slab = {
            let mut slabs = slots.slabs();
            let slab = slabs[slot].take();
            slots.generation.fetch_add(1, Ordering::AcqRel);
            slab
        };
        drop(slab);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::DeviceQueue;
    use std::sync::Condvar;

    /// Work in flight on a pool until the test ends it, recording each
    /// thread that waited on it.
    #[derive(Default)]
    struct Flight {
        done: Mutex<bool>,
        ended: Condvar,
        waiters: Mutex<Vec<std::thread::ThreadId>>,
    }

    impl Flight {
        fn complete(&self) {
            *self.done.lock().unwrap() = true;
            self.ended.notify_all();
        }
    }

    struct InFlight(Arc<Flight>);

    impl DeviceCompletion for InFlight {
        fn completed(&self) -> Result<bool, ExecutionError> {
            Ok(*self.0.done.lock().unwrap())
        }

        fn ended(&self) -> Result<(), ExecutionError> {
            self.0.waiters.lock().unwrap().push(std::thread::current().id());
            let mut done = self.0.done.lock().unwrap();
            while !*done {
                done = self.0.ended.wait(done).unwrap();
            }
            Ok(())
        }
    }

    /// A vacate spends only the enqueue of its moves beside the pool's work
    /// in flight: the shrink's thread never waits for that work, the moved
    /// slab's bytes survive at its new granule, its table names that granule
    /// for later work, and the emptied block, returned at once, is released
    /// to the OS once the work that may still read its old granule
    /// completes, from that completion.
    #[test]
    fn a_vacate_enqueues_its_moves_beside_work_in_flight_and_releases_the_block_behind_them() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..8 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        assert_eq!(tally.held(), absent + 2 * block, "slots 0..4 fill the older block, 4..8 the newer");
        let image = (0..8i32).flat_map(|value| (value + 1).to_le_bytes()).collect::<Vec<_>>();
        let moved = slabs.slab_region(7, 0).unwrap().unwrap();
        moved.write_from_host(&image).unwrap();
        let old_address = moved.device_address();
        drop(moved);
        // A shrink frees a slot of the older block and three of the newer:
        // neither empties, and the newer's last slab moves into the older.
        for slot in [1, 4, 5, 6] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let flight = Arc::new(Flight::default());
        pool.0.absent.allocation.record_device_use(DeviceQueue::Transfer, Arc::new(InFlight(flight.clone())), false);

        assert_eq!(pool.return_freed(4 * granule).unwrap(), block, "the emptied block is returned");
        assert!(
            !flight.waiters.lock().unwrap().contains(&std::thread::current().id()),
            "the shrink's thread never waits for the work in flight"
        );
        let view = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_eq!(view.read_to_host().unwrap(), image, "the slab's bytes survive the move");
        let new_address = view.device_address();
        assert_ne!(new_address, old_address, "the slab lies in another granule");
        let table = slabs.table(0).unwrap().read_to_host().unwrap();
        let entry = |slot: usize| {
            let at = (slabs.slots.entry(0, slot) - slabs.slots.region_offset(0)) as usize;
            u64::from_le_bytes(table[at..at + 8].try_into().unwrap())
        };
        assert_eq!(entry(7), new_address, "later work reads the slab at its new granule");
        drop(view);
        assert_eq!(tally.held(), absent + 2 * block, "the old granule may be read until the work in flight completes");
        assert_eq!(pool.held_bytes(), tally.held());

        flight.complete();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while tally.held() != absent + block {
            assert!(std::time::Instant::now() < deadline, "the emptied block is released once the work in flight completes");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(pool.held_bytes(), tally.held());
        assert_eq!(pool.return_freed(0).unwrap(), 0, "a returned block is never returned again");
    }

    /// A viewed slab in the emptiest block pins only that block: `return_freed`
    /// skips it and vacates another eligible block instead of returning 0.
    #[test]
    fn return_freed_skips_viewed_block_and_vacates_another() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 48,
            capacity: 12,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..12 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        assert_eq!(tally.held(), absent + 3 * block);
        // Free 2 slabs in each of the 3 blocks (6 free total, 2 live per block).
        for slot in [0, 1, 4, 5, 8, 9] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        // Pin one live slab in the newest block (slots 8..12).
        let pinned = slabs.slab_region(11, 0).unwrap().unwrap();
        assert_eq!(pool.return_freed(4 * granule).unwrap(), block, "skips the pinned block and vacates block 1");
        assert_eq!(tally.held(), absent + 2 * block);
        drop(pinned);
    }

    /// A block holding both live slabs and fenced (recently freed) slabs
    /// vacates when enough free granules exist elsewhere, returning its bytes
    /// immediately and dropping to the OS once the in-flight work completes.
    #[test]
    fn return_freed_vacates_block_with_fenced_granules() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..8 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        // Free 2 slots in block 0 while idle (immediately Free).
        for slot in [0, 1] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        // Now put work in flight and free 2 slots in block 1 (Fenced).
        let flight = Arc::new(Flight::default());
        pool.0.absent.allocation.record_device_use(DeviceQueue::Transfer, Arc::new(InFlight(flight.clone())), false);
        for slot in [4, 5] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        // Block 1 has 2 live + 2 fenced; block 0 has 2 free elsewhere.
        assert_eq!(pool.return_freed(2 * granule).unwrap(), block, "block 1 vacates its 2 live slabs into block 0");
        assert_eq!(tally.held(), absent + 2 * block, "block 1 waits for the in-flight work before dropping");
        flight.complete();
        pool.join_waiters();
        assert_eq!(tally.held(), absent + block, "block 1 releases once the flight completes");
    }

    /// `Pool::releasing()` stays true while blocks await an in-flight fence
    /// and `Pool::wake_on_release` fires once the flight completes and the
    /// block has physically dropped; when idle, `wake_on_release` fires
    /// immediately.
    #[test]
    fn wake_on_release_fires_after_flight_completes_and_releasing_clears() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..8 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        assert!(!pool.releasing(), "idle pool is not releasing");
        let immediate = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let immediate_flag = immediate.clone();
        pool.wake_on_release(move || immediate_flag.store(true, Ordering::Release));
        assert!(immediate.load(Ordering::Acquire), "wake_on_release fires immediately when idle");

        // Free all 4 granules of block 1 while a flight is active on the pool.
        struct CompleteOnDrop(Arc<Flight>);
        impl Drop for CompleteOnDrop {
            fn drop(&mut self) {
                self.0.complete();
            }
        }
        let flight = Arc::new(Flight::default());
        let guard = CompleteOnDrop(flight.clone());
        pool.0.absent.allocation.record_device_use(DeviceQueue::Transfer, Arc::new(InFlight(flight.clone())), false);
        for slot in 4..8 {
            assert!(slabs.free_slab(slot).unwrap());
        }
        assert_eq!(pool.return_freed(block).unwrap(), block, "block 1 marks returning and awaits flight");
        assert!(pool.releasing(), "pool reports releasing while await_owed waits on flight");

        let woken = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let woken_flag = woken.clone();
        pool.wake_on_release(move || woken_flag.store(true, Ordering::Release));
        assert!(!woken.load(Ordering::Acquire), "waker waits until flight completes and block drops");

        drop(guard);
        pool.join_waiters();
        assert!(woken.load(Ordering::Acquire), "waker fired once releasing cleared");
        assert!(!pool.releasing(), "pool is no longer releasing once block dropped");
        assert_eq!(tally.held(), absent + block, "block 1 physically dropped to the tally");
    }

    /// `add_slabs` backs the lowest free slots in a single batch (crossing
    /// block boundaries and reusing freed gaps), zeroes recycled granules when
    /// asked, updates the table entries for every region, and keeps
    /// `backed_count` in lockstep with `backed().len()`.
    #[test]
    fn add_slabs_batches_lowest_free_slots_and_tracks_backed_count() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18); // 4 granules per block
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: 2 * BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0), (i32s, vec![2], BLOCK_ALIGNMENT)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        assert_eq!(slabs.backed_count(), 0);
        assert!(slabs.add_slabs(0, SlabFill::Unwritten).unwrap().is_empty());
        assert_eq!(slabs.backed_count(), 0);

        // Claim 6 slabs across 2 blocks in one batch.
        let added = slabs.add_slabs(6, SlabFill::Zeroed).unwrap();
        assert_eq!(added, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(slabs.backed_count(), 6);
        assert_eq!(slabs.backed(), vec![0, 1, 2, 3, 4, 5]);

        // Dirty slots 1 and 4, then free them and reclaim with SlabFill::Zeroed.
        let dirty = vec![0x5au8; 32];
        for slot in [1, 4] {
            slabs.slab_region(slot, 0).unwrap().unwrap().write_from_host(&dirty).unwrap();
            assert!(slabs.free_slab(slot).unwrap());
        }
        assert_eq!(slabs.backed_count(), 4);

        let refilled = slabs.add_slabs(3, SlabFill::Zeroed).unwrap();
        assert_eq!(refilled, vec![1, 4, 6]);
        assert_eq!(slabs.backed_count(), 7);
        for &slot in &refilled {
            let bytes = slabs.slab_region(slot, 0).unwrap().unwrap().read_to_host().unwrap();
            assert!(bytes.iter().all(|&b| b == 0), "recycled slot {slot} is zeroed");
        }
        // Requesting more than remaining capacity fails without modifying state.
        assert!(slabs.add_slabs(2, SlabFill::Unwritten).is_err());
        assert_eq!(slabs.backed_count(), 7);
    }

    /// When a pool growth is refused partway through `add_slabs`, no slot is
    /// backed and every granule already claimed for the batch is returned to
    /// the pool without leaking.
    #[test]
    fn add_slabs_rolls_back_claimed_granules_on_refusal() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let (granule, block) = (1 << 16, 1 << 18); // 4 granules per block
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        // Allow exactly one block beyond the absent granule (4 granules total).
        tally.bound(absent + block);
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        // Claim 2 slabs (2 granules remain free in block 0).
        assert_eq!(slabs.add_slabs(2, SlabFill::Unwritten).unwrap(), vec![0, 1]);
        assert_eq!(slabs.backed_count(), 2);
        assert_eq!(pool.free_granules(), 2);

        // Requesting 4 more needs a second block, which the tally refuses.
        assert!(slabs.add_slabs(4, SlabFill::Unwritten).is_err());
        assert_eq!(slabs.backed_count(), 2, "no slot backed on refusal");
        assert_eq!(slabs.backed(), vec![0, 1]);
        assert_eq!(pool.free_granules(), 2, "already-claimed granules returned cleanly");

        // The 2 remaining granules are still claimable.
        assert_eq!(slabs.add_slabs(2, SlabFill::Unwritten).unwrap(), vec![2, 3]);
        assert_eq!(slabs.backed_count(), 4);
        assert_eq!(pool.free_granules(), 0);
    }

    /// A granule whose slot was freed or whose store was dropped while a
    /// detached `TensorInner` view still pins its `GranuleHold` causes
    /// `vacate_emptiest` to unvacate and skip that block cleanly instead of
    /// panicking in `GranulePool::vacated`.
    #[test]
    fn vacate_skips_block_when_freed_slot_or_dropped_store_still_pinned_by_view() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18); // 4 granules per block
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 48,
            capacity: 12,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&Charges::tally(&device), &pool, geometry.clone(), &[]).unwrap();
        slabs.add_slabs(12, SlabFill::Zeroed).unwrap();
        assert_eq!(tally.held(), absent + 3 * block);

        // Case 1: freed slot still pinned by a detached view in block 2 (slots 8..12).
        for slot in [0, 1, 4, 5, 8, 9, 10] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let detached_freed = slabs.slab_region(11, 0).unwrap().unwrap();
        assert!(slabs.free_slab(11).unwrap());
        // Block 2 has 1 held granule (slot 11's detached view) and 3 free;
        // `adopt` returns `Adopted::Left`, so `try_vacated` must unvacate and
        // skip block 2, then vacate block 1 instead.
        assert_eq!(pool.return_freed(4 * granule).unwrap(), block);
        assert_eq!(tally.held(), absent + 2 * block);
        drop(detached_freed);
        assert_eq!(pool.return_freed(granule).unwrap(), block);
        assert_eq!(tally.held(), absent + block);

        // Case 2: dropped store still pinned by a detached view in the newest block.
        let store2 = SlabStore::in_pool(&Charges::tally(&device), &pool, geometry, &[]).unwrap();
        store2.add_slabs(4, SlabFill::Zeroed).unwrap();
        assert_eq!(tally.held(), absent + 2 * block);
        // Free one slot in block 0 so block 3 (with 1 pinned granule) is
        // eligible to vacate into block 0.
        assert!(slabs.free_slab(2).unwrap());
        let detached_dropped = store2.slab_region(3, 0).unwrap().unwrap();
        drop(store2);
        // `store2` is gone from `pool.state().stores`, yet `detached_dropped`
        // still pins its granule in block 3: `vacate_emptiest` must unvacate
        // and skip block 3 cleanly instead of panicking, then vacate block 0
        // into block 3's 3 free granules.
        assert_eq!(pool.return_freed(4 * granule).unwrap(), block);
        assert_eq!(tally.held(), absent + block);
        drop(detached_dropped);
    }

    /// `SlabStore::in_pool` rejects zero `slab_alignment` and zero
    /// `rows_per_slab` with `TensorError::SlabLayout` instead of panicking.
    #[test]
    fn slab_store_in_pool_rejects_zero_alignment_and_zero_rows_per_slab() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let charges = Charges::tally(&device);
        let pool = Pool::open(&PoolCharges::one(&charges), Contract::DeviceOnly, 1 << 16, 1 << 18).unwrap();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);

        let zero_align = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 16,
            capacity: 4,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: 0,
            regions: vec![(i32s, vec![2], 0)],
        };
        assert!(matches!(SlabStore::in_pool(&charges, &pool, zero_align, &[]), Err(TensorError::SlabLayout(_))));

        let zero_rows = SlabGeometry {
            rows_per_slab: 0,
            logical_rows: 16,
            capacity: 4,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        assert!(matches!(SlabStore::in_pool(&charges, &pool, zero_rows, &[]), Err(TensorError::SlabLayout(_))));
    }

    /// `Pool::return_freed_sync` waits for in-flight fences without spawning a
    /// redundant `await_owed` waiter and does not return until every returned
    /// block has physically dropped and `releasing()` is false.
    #[test]
    fn return_freed_sync_waits_for_in_flight_and_clears_releasing() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&Charges::tally(&device), &pool, geometry, &[]).unwrap();
        slabs.add_slabs(8, SlabFill::Zeroed).unwrap();

        let flight = Arc::new(Flight::default());
        pool.0.absent.allocation.record_device_use(DeviceQueue::Transfer, Arc::new(InFlight(flight.clone())), false);
        for slot in 4..8 {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let completer = {
            let flight = flight.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(10));
                flight.complete();
            })
        };
        assert_eq!(pool.return_freed_sync(block).unwrap(), block);
        completer.join().unwrap();
        assert_eq!(tally.held(), absent + block, "block physically dropped before return_freed_sync returned");
        assert!(!pool.releasing(), "pool is not releasing after return_freed_sync");
        let immediate = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = immediate.clone();
        pool.wake_on_release(move || flag.store(true, Ordering::Release));
        assert!(immediate.load(Ordering::Acquire), "wake_on_release fires immediately after return_freed_sync");
    }

    /// `Slots::generation` advances on `add_slabs`, `free_slab`, and `adopt`
    /// while holding `slabs`, keeping `SlabView::reach()` in sync with the
    /// live slab table.
    #[test]
    fn generation_advances_on_add_free_and_adopt() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let charges = Charges::tally(&device);
        let (granule, block) = (1 << 16, 1 << 18);
        let pool = Pool::open(&PoolCharges::one(&charges), Contract::DeviceOnly, granule, block).unwrap();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&charges, &pool, geometry, &[]).unwrap();
        let g0 = slabs.slots.generation();
        slabs.add_slabs(8, SlabFill::Zeroed).unwrap();
        let g1 = slabs.slots.generation();
        assert!(g1 > g0, "add_slabs bumps generation");

        for slot in [1, 4, 5, 6] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let g2 = slabs.slots.generation();
        assert!(g2 > g1, "free_slab bumps generation");

        assert_eq!(pool.return_freed(4 * granule).unwrap(), block);
        let g3 = slabs.slots.generation();
        assert!(g3 > g2, "adopt bumps generation");
    }

    /// `Slots::adopt` keeps `Slots::slabs` a leaf lock and never holds it
    /// across `moves::submit`: a concurrent `region_view` succeeds while
    /// `moves::submit` is blocked, keeps the old granule alive with intact
    /// bytes, and `vacate_emptiest` skips the block cleanly via
    /// `GranulePool::try_vacated` instead of panicking.
    #[test]
    fn adopt_releases_slabs_lock_across_moves_submit_and_skips_mid_adopt_view() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18); // 4 granules per block
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..8 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        assert_eq!(tally.held(), absent + 2 * block);
        let image = (0..8i32).flat_map(|value| (value + 11).to_le_bytes()).collect::<Vec<_>>();
        let old_address = {
            let view = slabs.slab_region(7, 0).unwrap().unwrap();
            view.write_from_host(&image).unwrap();
            view.device_address()
        };
        // Free 1 slot in block 0 and 3 slots in block 1 so vacating block 1
        // moves slot 7 into block 0. Pin slot 0 in block 0 so skipping block 1
        // does not vacate block 0 into block 1's 3 free slots.
        for slot in [1, 4, 5, 6] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let _pin_block0 = slabs.slab_region(0, 0).unwrap().unwrap();

        // Hold a host write permit on the pool's hazard domain so `moves::submit`
        // blocks inside `domain.acquire_for_device(true)` while holding
        // `slabs.slots.repointed`.
        let host_permit = pool.0.absent.allocation.acquire(true).unwrap();
        let pool_bg = pool.clone();
        let handle = std::thread::spawn(move || pool_bg.return_freed(4 * granule));

        // Wait until `Slots::adopt` enters Phase 2 (`repointed` locked, blocked
        // inside `moves::submit`).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while slabs.slots.repointed.try_lock().is_ok() {
            assert!(std::time::Instant::now() < deadline, "adopt did not reach moves::submit in time");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // `region_view` must succeed immediately while `moves::submit` is
        // blocked (would deadlock if `Slots::adopt` held `Slots::slabs`).
        let mid_view = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_eq!(mid_view.device_address(), old_address, "mid-adopt view still sees the old granule");

        // Unblock `moves::submit` and let `return_freed` finish. Because
        // `mid_view` was created during Phase 2, `Slots::adopt` detects the
        // view increment in Phase 3 and leaves slot 7 at `old_address` without
        // repointing `self.table`, so any write through `mid_view` is preserved.
        drop(host_permit);
        assert_eq!(
            handle.join().unwrap().unwrap(),
            0,
            "block pinned by a mid-adopt view is skipped cleanly"
        );
        assert_eq!(tally.held(), absent + 2 * block, "pinned block stays held while mid_view lives");

        // Writing through `mid_view` after `adopt` finishes must be visible to
        // subsequent readers of slot 7 (previously lost because `adopt`
        // unconditionally repointed slot 7 to the stale copy).
        let updated = (0..8i32).flat_map(|value| (value + 99).to_le_bytes()).collect::<Vec<_>>();
        mid_view.write_from_host(&updated).unwrap();
        assert_eq!(mid_view.read_to_host().unwrap(), updated, "mid-adopt view reads updated bytes from old granule");

        let still_old = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_eq!(still_old.device_address(), old_address, "slot 7 stayed on old granule while mid-adopt view was created");
        assert_eq!(still_old.read_to_host().unwrap(), updated, "slot 7 observes mid-adopt write with no data loss");
        drop(still_old);

        // Dropping the mid-adopt view allows the next `return_freed(0)` to
        // move slot 7 (carrying `updated`) to a new granule and return the
        // emptied block.
        drop(mid_view);
        assert_eq!(pool.return_freed(0).unwrap(), block, "emptied block returns once the mid-adopt view drops");
        pool.join_waiters();
        assert_eq!(tally.held(), absent + block);
        let moved_view = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_ne!(moved_view.device_address(), old_address, "slot 7 moved to a new granule on the follow-up pass");
        assert_eq!(moved_view.read_to_host().unwrap(), updated, "moved slot preserves the updated bytes");
    }

    /// Even if a concurrent writer acquires a `region_view` during `Slots::adopt`
    /// Phase 2, writes new bytes, and drops the view BEFORE Phase 3 runs (so
    /// `Arc::strong_count` is back to 1), `Slab::views` detects the mid-adopt
    /// view and aborts moving that slot so the write is never overwritten by
    /// the stale Phase 2 copy.
    #[test]
    fn adopt_preserves_mid_adopt_write_even_after_view_drops() {
        let catalog = crate::devices::Devices::discover(&[]).unwrap();
        let id = catalog.resolve(seismic_topology::ExecutorSelector::HostCpu).unwrap();
        let device = catalog.open(id).unwrap();
        let topology = catalog.topology();
        let store = topology.first_store(topology.executor(id).unwrap(), Contract::DeviceOnly).unwrap();
        let tally = crate::memory::Tally::unbounded(topology.store_identity(store).unwrap());
        let (granule, block) = (1 << 16, 1 << 18);
        let charges = PoolCharges::one(&Charges::in_store(&device, Contract::DeviceOnly, Arc::new(tally.clone())));
        let pool = Pool::open(&charges, Contract::DeviceOnly, granule, block).unwrap();
        let absent = tally.held();
        let i32s = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
        let geometry = SlabGeometry {
            rows_per_slab: 4,
            logical_rows: 32,
            capacity: 8,
            slab_bytes: BLOCK_ALIGNMENT,
            slab_alignment: BLOCK_ALIGNMENT,
            regions: vec![(i32s, vec![2], 0)],
        };
        let slabs = SlabStore::in_pool(&crate::memory::Charges::tally(&device), &pool, geometry, &[]).unwrap();
        for _ in 0..8 {
            slabs.add_slab(SlabFill::Zeroed).unwrap();
        }
        let old_image = (0..8i32).flat_map(|value| (value + 1).to_le_bytes()).collect::<Vec<_>>();
        let old_address = {
            let view = slabs.slab_region(7, 0).unwrap().unwrap();
            view.write_from_host(&old_image).unwrap();
            view.device_address()
        };
        for slot in [1, 4, 5, 6] {
            assert!(slabs.free_slab(slot).unwrap());
        }
        let _pin_block0 = slabs.slab_region(0, 0).unwrap().unwrap();

        let host_permit = pool.0.absent.allocation.acquire(true).unwrap();
        let pool_bg = pool.clone();
        let handle = std::thread::spawn(move || pool_bg.return_freed(4 * granule));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while slabs.slots.repointed.try_lock().is_ok() {
            assert!(std::time::Instant::now() < deadline, "adopt did not reach moves::submit in time");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Acquire a view during Phase 2, then drop it BEFORE Phase 3 so
        // `strong_count` returns to 1; `Slab::views` must still catch it.
        let mid_view = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_eq!(mid_view.device_address(), old_address);
        drop(mid_view);

        drop(host_permit);
        assert_eq!(handle.join().unwrap().unwrap(), 0, "transient mid-adopt view aborts moving slot 7");
        let after = slabs.slab_region(7, 0).unwrap().unwrap();
        assert_eq!(after.device_address(), old_address, "slot 7 was not repointed after transient mid-adopt view");
        drop(after);

        assert_eq!(pool.return_freed(0).unwrap(), block);
        pool.join_waiters();
        assert_eq!(tally.held(), absent + block);
    }
}
