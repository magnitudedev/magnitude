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
//!   granule leaves its slab once no view of it remains ([`GranuleHold`]).
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

    fn allocate(&self, bytes: u64, entry: &Arc<dyn Entry>) -> Result<PoolBlock, ExecutionError> {
        let charged = Arc::new(BlockEntry { entry: entry.clone(), charged: AtomicU64::new(0) });
        let through: Arc<dyn Entry> = charged.clone();
        let allocation = self.device.allocate_block(&self.kinds, bytes, &through, self.access.clone())?;
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

struct PoolShared {
    device: Arc<DeviceInner>,
    granule: u64,
    block_bytes: u64,
    /// One zero granule no slab holds, which every free slot of the pool's
    /// stores names, in the pool's hazard domain (as every block and table
    /// of the pool is).
    absent: PoolBlock,
    state: Mutex<PoolState>,
}

struct PoolState {
    granules: GranulePool<PoolBlocks, Holder>,
    /// The stores claiming from the pool, by identity: whose tables a
    /// vacate repoints.
    stores: HashMap<u64, Weak<Slots>>,
    /// Blocks out of the pool, every granule free, each with the pool's
    /// uses when it left: released as those end ([`Pool::return_empty`]).
    releasing: Vec<(PoolBlock, Arc<DeviceUses>)>,
    /// Whether a thread waits on released granules' and releasing blocks'
    /// flights ([`Pool::await_owed`]): one at a time.
    waiting: bool,
}

impl PoolShared {
    fn state(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().expect("granule pool lock poisoned")
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
        let absent = blocks.allocate(granule, &absent)?;
        let shared = Arc::new(PoolShared {
            device: device.clone(),
            granule,
            block_bytes,
            absent,
            state: Mutex::new(PoolState {
                granules: GranulePool::new(blocks, granule, block_bytes, entry),
                stores: HashMap::new(),
                releasing: Vec::new(),
                waiting: false,
            }),
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

    /// Allocate one block, admitted by the pool's entry first: the pool's
    /// grow step. Refused (`ExecutionError::AllocationCapacity` where the
    /// entry refuses), nothing changes.
    pub fn grow(&self) -> Result<(), TensorError> {
        self.0.state().granules.grow().map(drop).map_err(TensorError::Execution)
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
            (bytes, released)
        };
        // Released outside the pool's lock, behind work that has ended.
        drop(released);
        bytes
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
        self.0.state().granules.owe(freed);
        let mut returned = self.return_empty();
        loop {
            let short = {
                let mut state = self.0.state();
                state.granules.owed() > state.granules.awaiting()
            };
            if !short {
                break;
            }
            match self.vacate_emptiest()? {
                Some(bytes) => returned += bytes,
                None => break,
            }
        }
        // What emptied meanwhile: a vacated block whose copies completed as
        // they were submitted, its bytes counted already, and any other.
        returned += self.return_empty();
        self.await_owed();
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
        let pool = self.clone();
        drop(seismic_memory::pool::after(fences, move || {
            pool.0.state().waiting = false;
            pool.return_empty();
            pool.await_owed();
        }));
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
    fn vacate_emptiest(&self) -> Result<Option<u64>, TensorError> {
        let pool = &self.0;
        let vacating = {
            let mut state = pool.state();
            let Some(block) = state.granules.emptiest() else {
                return Ok(None);
            };
            let Some(vacating) = state.granules.vacate(block) else {
                return Ok(None);
            };
            vacating
        };
        let block: BlockId = vacating.block;
        let mut by_store = std::collections::BTreeMap::<u64, Vec<Move<Holder>>>::new();
        for moved in &vacating.moves {
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
            // out staying at their new granules.
            pool.state().granules.unvacate(block);
        }
        match (failure, stays) {
            (Some(error), _) => Err(error),
            (None, true) => Ok(None),
            (None, false) => Ok(Some(pool.state().granules.vacated(block))),
        }
    }
}

/// A granule claimed for a slab, held while its slab or any view of it
/// lives: as the last drops it leaves the slab, free to claim once the
/// pool's work submitted by then completes. So a freed slab's bytes outlive
/// every view that could still reach them.
pub(crate) struct GranuleHold {
    pool: Arc<PoolShared>,
    granule: GranuleId,
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

/// What a new slab's bytes hold ([`SlabStore::add_slab`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabFill {
    /// Zero, as kernels that read a slab before writing it require.
    Zeroed,
    /// Whatever its granule held: for a holder that writes every byte
    /// before device work reads them.
    Unwritten,
}

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
    /// Its granule; none for a lent slab, and for a copy's slab
    /// ([`Slots::copy`]).
    hold: Option<Arc<GranuleHold>>,
}

impl Slab {
    fn granule(&self) -> Option<GranuleId> {
        self.hold.as_ref().map(|hold| hold.granule)
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
    /// The device address every free slot names: a whole slab's bytes.
    absent: u64,
    /// The leading slots whose slabs are lent, never freed.
    lent: usize,
    /// The allocations the lent slabs lie in, each in its own hazard domain.
    lenders: Vec<Arc<Allocation>>,
    /// The table's generation: moves each time a slot is repointed
    /// ([`Self::point`]), under the slab lock, with the slab it names.
    generation: AtomicU64,
    slabs: Mutex<Vec<Option<Slab>>>,
    /// The device work that last wrote the table ([`Self::adopt`]), while
    /// it may still run: a later repoint is ordered after it, never a host
    /// write it could land over. Changed under the slab lock.
    repointed: Mutex<Option<Arc<dyn DeviceCompletion>>>,
}

impl Slots {
    fn slabs(&self) -> MutexGuard<'_, Vec<Option<Slab>>> {
        self.slabs.lock().expect("slab store lock poisoned")
    }

    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }

    pub(crate) fn table(&self) -> &Arc<Allocation> {
        &self.table
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// A table of `regions × capacity` entries in `access`, through
    /// `charges`: record-table storage, aligned to its entry, as every
    /// reader of an entry (a native kernel's, [`Self::point`]'s atomic
    /// store) reads it whole.
    fn allocate_table(
        charges: &Charges,
        regions: usize,
        capacity: usize,
        access: Option<Arc<AccessGroup>>,
    ) -> Result<Arc<Allocation>, TensorError> {
        let bytes = (regions as u64)
            .checked_mul(capacity as u64)
            .and_then(seismic_lang::registry::record_table_bytes)
            .ok_or_else(|| TensorError::SlabLayout("a slab table's bytes overflow".into()))?;
        Ok(charges.allocate_in(bytes, seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES, access)?)
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
    /// the lenders'.
    pub(crate) fn lenders(&self) -> &[Arc<Allocation>] {
        &self.lenders
    }

    /// The table's generation now: a view's reach read after it shows
    /// every slab change before it ([`Self::point`]).
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Point `slot`'s entries, region by region, at `addresses`: one 8-byte
    /// store each, which device work in flight reads whole, old or new, and
    /// which never waits for device work. A host store where the host
    /// addresses the table and no device write of it may still run
    /// ([`Self::adopt`]), which the host store could land before; else
    /// ordered words (`native::moves`), after the work submitted before and
    /// before the work submitted after. Never a host write through staging
    /// (`Storage::write` of storage the host does not address): on a backend
    /// whose staging copy is a submission to the device's one queue
    /// (Vulkan), the host would wait behind every flight in flight, and
    /// behind a flight parked on a gate, for the gate, which the writing
    /// thread may be the one to answer. The caller holds the slab lock and
    /// changes the slot's slab under it, so the generation moves with the
    /// change.
    fn point(&self, slot: usize, addresses: impl Iterator<Item = u64>) -> Result<(), ExecutionError> {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut repointed = self.repointed();
        let pending = repointed.as_ref().is_some_and(|written| matches!(written.completed(), Ok(false)));
        match self.table.storage().host_address().filter(|_| !pending) {
            Some(base) => {
                for (region, address) in addresses.enumerate() {
                    let offset = self.entry(region, slot);
                    // SAFETY: an 8-byte-aligned entry inside the table,
                    // which lives as `self` does; device work only reads it.
                    unsafe { AtomicU64::from_ptr(base.as_ptr().add(offset as usize).cast()).store(address, Ordering::Release) }
                }
            }
            None => {
                let words = addresses
                    .enumerate()
                    .map(|(region, value)| MoveStep::Word { destination: &self.table, offset: self.entry(region, slot), value })
                    .collect::<Vec<_>>();
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
        let slabs = self.slabs();
        let Some(Some(slab)) = slabs.get(slot) else {
            return Ok(None);
        };
        let (allocation, offset) = slab.regions[region].clone();
        TensorInner::in_slab(&self.device, allocation, offset, representation, extents, slab.hold.clone()).map(Some)
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

    /// Where region `region` of each backed slot in `first..first + count`
    /// lies now (`Allocation::identity`, byte offset, `bytes`): what device
    /// work bound to a view of those slots reaches, without keeping a removed
    /// slab's pool block alive inside a cached reach.
    pub(crate) fn reach(
        &self,
        region: usize,
        first: usize,
        count: usize,
        bytes: u64,
    ) -> Vec<(u64, u64, u64)> {
        self.slabs()[first..first + count]
            .iter()
            .flatten()
            .map(|slab| {
                let (allocation, offset) = &slab.regions[region];
                (allocation.identity(), *offset, bytes)
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
        let mut slabs = self.slabs();
        let adopted = moved
            .iter()
            .map(|moved| match slabs.get(moved.holder.slot) {
                Some(Some(slab)) if slab.granule() == Some(moved.from) => {
                    match slab.hold.as_ref().is_some_and(|hold| Arc::strong_count(hold) > 1) {
                        true => Adopted::Viewed,
                        false => Adopted::Moved,
                    }
                }
                Some(_) | None => Adopted::Left,
            })
            .collect::<Vec<_>>();
        let moving = moved.iter().zip(&adopted).filter(|(_, adopted)| matches!(adopted, Adopted::Moved)).map(|(moved, _)| moved).collect::<Vec<_>>();
        if moving.is_empty() {
            return Ok(adopted);
        }
        let blocks = {
            let state = pool.state();
            moving
                .iter()
                .map(|moved| (state.granules.block(moved.from.block).allocation.clone(), state.granules.block(moved.to.block).allocation.clone()))
                .collect::<Vec<_>>()
        };
        let at = |granule: GranuleId| u64::from(granule.index) * pool.granule;
        let copies = moving.iter().zip(&blocks).map(|(moved, (from, to))| MoveStep::Copy {
            source: from,
            source_offset: at(moved.from),
            destination: to,
            destination_offset: at(moved.to),
            bytes: self.slab_bytes,
        });
        let words = moving.iter().flat_map(|moved| {
            self.offsets.iter().enumerate().map(move |(region, offset)| MoveStep::Word {
                destination: &self.table,
                offset: self.entry(region, moved.holder.slot),
                value: moved.to_address + offset,
            })
        });
        let steps = copies.chain(words).collect::<Vec<_>>();
        // Ordered after any earlier repoint of the table, which runs on the
        // same queue before it.
        let mut repointed = self.repointed();
        let written = moves::submit(&self.device, &self.table, &steps)?;
        *repointed = written;
        drop(repointed);
        self.generation.fetch_add(1, Ordering::AcqRel);
        let previous = moving
            .iter()
            .zip(&blocks)
            .map(|(moved, (_, to))| {
                let adopted = Slab {
                    regions: self.offsets.iter().map(|offset| (to.clone(), at(moved.to) + offset)).collect(),
                    hold: Some(Arc::new(GranuleHold { pool: pool.clone(), granule: moved.to })),
                };
                std::mem::replace(slabs[moved.holder.slot].as_mut().expect("a moving slot is backed"), adopted)
            })
            .collect::<Vec<_>>();
        drop(slabs);
        // No view holds them: each old granule leaves now, released behind
        // the pool's newest uses, the moves among them.
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
        let capacity = images.len();
        let table = Self::allocate_table(charges, 1, capacity, None)?;
        let slabs = images
            .iter()
            .map(|image| {
                image
                    .as_ref()
                    .map(|bytes| {
                        let slab = TensorInner::from_host_in(charges, representation, extents, bytes, Some(table.access_group().clone()))?;
                        Ok::<_, TensorError>(Slab { regions: vec![(slab.allocation().clone(), 0)], hold: None })
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
            id: crate::driver::fresh_allocation_identity(),
            device: device.clone(),
            pool: None,
            table,
            capacity,
            slab_bytes: crate::layout::canonical(representation, extents)?.byte_len,
            offsets: vec![0],
            absent,
            lent: 0,
            lenders: Vec::new(),
            generation: AtomicU64::new(0),
            slabs: Mutex::new(slabs),
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
    /// alignment is not one every granule has, or the lent tensors do not
    /// match the regions. Its address table is allocated through `charges`.
    ///
    /// # Panics
    /// Where `charges` are of another device than the pool's.
    pub fn in_pool(charges: &Charges, pool: &Pool, geometry: SlabGeometry, lent: &[Arc<TensorInner>]) -> Result<Self, TensorError> {
        let shared = &pool.0;
        let refused = |why: String| TensorError::SlabLayout(why);
        if BLOCK_ALIGNMENT % geometry.slab_alignment != 0 {
            return Err(refused(format!(
                "a slab aligned to {} bytes lies in no granule, each aligned to {BLOCK_ALIGNMENT}",
                geometry.slab_alignment
            )));
        }
        let device = &shared.device;
        charges.of_device(device);
        let capacity = geometry.capacity;
        let mut slabs = (0..capacity).map(|_| None).collect::<Vec<Option<Slab>>>();
        let mut lenders = Vec::<Arc<Allocation>>::new();
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
                        && tensor.byte_len() == rows / geometry.rows_per_slab * slab.byte_len;
                    let slabs_of = rows / geometry.rows_per_slab;
                    if !whole || count.is_some_and(|count| count != slabs_of) {
                        return Err(refused(
                            "a lent tensor is a contiguous canonical view of whole slab rows of its region, \
                             as many slabs as every other's, on the pool's device"
                                .into(),
                        ));
                    }
                    count = Some(slabs_of);
                }
                let count = usize::try_from(count.unwrap_or(0)).ok().filter(|&count| count <= capacity).ok_or_else(|| {
                    refused(format!("more lent slabs than the store's {capacity} slots"))
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
                    *slab = Some(Slab { regions, hold: None });
                }
                for tensor in lent {
                    if !lenders.iter().any(|held| Arc::ptr_eq(held, tensor.allocation())) {
                        lenders.push(tensor.allocation().clone());
                    }
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
        let table = Slots::allocate_table(charges, geometry.regions.len(), capacity, Some(shared.absent.allocation.access_group().clone()))?;
        let offsets = geometry.regions.iter().map(|(_, _, offset)| *offset).collect::<Vec<_>>();
        let absent = device.allocation_address(&shared.absent.allocation, 0);
        let slots = Arc::new(Slots {
            id: crate::driver::fresh_allocation_identity(),
            device: device.clone(),
            pool: Some(shared.clone()),
            table,
            capacity,
            slab_bytes: geometry.slab_bytes,
            offsets,
            absent,
            lent: lent_slabs,
            lenders,
            generation: AtomicU64::new(0),
            slabs: Mutex::new(Vec::new()),
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
        let slots = &self.slots;
        let pool = slots.pool.as_ref().expect("a store that adds slabs is a pool's");
        let mut slabs = slots.slabs();
        let slot = slabs
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| TensorError::SlabLayout("all slab slots are backed".into()))?;
        let holder = Holder { store: slots.id, slot };
        let (Claimed { granule, address, zero: fresh }, block) = {
            let mut state = pool.state();
            let claimed = match state.granules.take(holder) {
                Some(claimed) => claimed,
                None => {
                    state.granules.grow().map_err(TensorError::Execution)?;
                    state.granules.take(holder).expect("a grown block has free granules")
                }
            };
            let block = state.granules.block(claimed.granule.block).allocation.clone();
            (claimed, block)
        };
        // Its granule goes back free should the slab not form.
        let hold = Arc::new(GranuleHold { pool: pool.clone(), granule });
        let base = u64::from(granule.index) * pool.granule;
        if fill == SlabFill::Zeroed && !fresh {
            // No work in flight reads the granule: its last reader ended. The
            // host zeroes storage it addresses; else the pool's absent
            // granule, zero, is copied over it in device order, before any
            // work that reads it, never a waited staging write (`point`).
            match block.storage().host_address() {
                Some(_) => crate::driver::write_zeros(block.storage(), base, slots.slab_bytes)?,
                None => drop(moves::submit(
                    &slots.device,
                    &slots.table,
                    &[MoveStep::Copy {
                        source: &pool.absent.allocation,
                        source_offset: 0,
                        destination: &block,
                        destination_offset: base,
                        bytes: slots.slab_bytes,
                    }],
                )?),
            }
        }
        slots.point(slot, slots.offsets.iter().map(|offset| address + offset))?;
        slabs[slot] = Some(Slab {
            regions: slots.offsets.iter().map(|offset| (block.clone(), base + offset)).collect(),
            hold: Some(hold),
        });
        Ok(slot)
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
        let mut slabs = slots.slabs();
        let Some(slab) = slabs.get_mut(slot).and_then(Option::take) else {
            return Ok(false);
        };
        if let Err(error) = slots.point(slot, slots.offsets.iter().map(|offset| slots.absent + offset)) {
            slabs[slot] = Some(slab);
            return Err(error.into());
        }
        drop(slabs);
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
}
