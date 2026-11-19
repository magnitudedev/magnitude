//! One store's granule pool: the memory the expert bank's records and the KV
//! slabs both live in.
//!
//! - Blocks come from the backend's plain allocator ([`Blocks`]: a shared
//!   MTLBuffer, `cuMemAlloc`, `vkAllocateMemory`, host pages), one ordinary
//!   allocation each, sized by [`block_bytes`].
//! - Fixed-size granules are carved inside the blocks. Device work reaches
//!   each through its holder's record table, so a granule's place never
//!   matters: the pool allows holes, and freeing a granule moves nothing.
//! - A claim takes a free granule in process first ([`GranulePool::take`]);
//!   only [`GranulePool::grow`] allocates.
//! - A granule leaves its holder through the completion of the last flight
//!   that read it ([`GranulePool::release`]), claimable again once that
//!   completes: a claim waits at most one flight ([`GranulePool::pending`]).
//! - A block returns to the OS only when every granule in it is free
//!   ([`GranulePool::return_empty`] hands it to its caller to drop). To
//!   empty one for outside demand, its live granules are copied into free
//!   granules elsewhere ([`GranulePool::vacate`]): the pool's only copy.
//!   Once every copy is made, each source released behind the copy's
//!   flight, the block is returning ([`GranulePool::vacated`]): it pays what
//!   is owed then, and returns as those flights complete.
//! - Every block is charged to the pool's ledger entry ([`Entry`]) by the
//!   backend that allocates it, at its exact native bytes, and freed from it
//!   as the block drops: the pool's free granules are bytes the process
//!   holds, and the entry counts them. Which granules the bank or a sequence
//!   holds is the embedder's, who claims and releases them.

use crate::entry::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// What a new slab's or pool block's bytes hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabFill {
    /// Zero, as kernels that read a slab before writing it require.
    Zeroed,
    /// Whatever its backing storage held: skips synchronous device zeroing on
    /// block allocation and slab claim. Valid only when the holder writes
    /// every byte before device work reads the slab. Granules installed from
    /// an `Unwritten` block are tracked `zero: false`, so any later `Zeroed`
    /// claim still zeroes its slab before use.
    Unwritten,
}

/// Single-flight coordinator for out-of-lock pool growth: at most one thread
/// allocates a block at a time while concurrent callers wait for that growth
/// to install and then re-check for a free granule before deciding whether
/// another growth is needed.
#[derive(Default)]
pub struct GrowGate {
    growing: std::sync::Mutex<bool>,
    done: std::sync::Condvar,
}

/// Outcome of [`GrowGate::enter_or_wait`].
pub enum GrowTurn<'a> {
    /// This thread owns the single active growth until the guard drops.
    Grow(GrowGuard<'a>),
    /// Another thread finished a growth while this thread waited; re-check
    /// free granules before growing again.
    Waited,
}

/// RAII guard for the active grower: clears `growing` and wakes all waiters
/// on normal completion, error, or panic.
pub struct GrowGuard<'a> {
    gate: &'a GrowGate,
}

impl GrowGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Either become the sole grower ([`GrowTurn::Grow`]) or block until the
    /// current grower finishes ([`GrowTurn::Waited`]). Never called while
    /// holding the pool state lock.
    pub fn enter_or_wait(&self) -> GrowTurn<'_> {
        self.enter_or_wait_inner(|| {})
    }

    #[cfg(test)]
    pub(crate) fn enter_or_wait_with(&self, on_park: impl FnOnce()) -> GrowTurn<'_> {
        self.enter_or_wait_inner(on_park)
    }

    fn enter_or_wait_inner(&self, on_park: impl FnOnce()) -> GrowTurn<'_> {
        let mut growing = self.growing.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if *growing {
            let mut parked = Some(on_park);
            while *growing {
                if let Some(notify) = parked.take() {
                    notify();
                }
                growing = self.done.wait(growing).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            return GrowTurn::Waited;
        }
        *growing = true;
        GrowTurn::Grow(GrowGuard { gate: self })
    }
}

impl Drop for GrowGuard<'_> {
    fn drop(&mut self) {
        let mut growing = self.gate.growing.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *growing = false;
        self.gate.done.notify_all();
    }
}

/// A backend's plain allocator, which a pool's blocks come from.
pub trait Blocks: Send + Sync + 'static {
    /// One ordinary allocation, freed when dropped.
    type Block: Send + Sync;
    /// The completion of a flight that read granules (Metal's command-buffer
    /// completion, a CUDA event, a Vulkan fence, the CPU backend's join).
    type Fence: Fence;
    type Error: Send;
    /// Allocate one block of `bytes`, every page backed before it returns,
    /// never lazily: the ledger counts the block from the moment it is made,
    /// so a block backed later would read as others' memory falling and then
    /// rising. When `fill` is [`SlabFill::Zeroed`], every byte reads zero;
    /// when [`SlabFill::Unwritten`], synchronous device zeroing is skipped
    /// because the holder writes every byte before device work reads it. Its
    /// exact native bytes are charged to `entry`, admitted before the native
    /// allocation (refused, nothing is allocated), and held with the block:
    /// dropping it returns its memory to the OS and frees the charge at once.
    fn allocate(&self, bytes: u64, entry: &Arc<dyn Entry>, fill: SlabFill) -> Result<Self::Block, Self::Error>;
    /// The bytes `block` holds in its store, at least those asked for: its
    /// allocation's rounding and the driver objects it took included, as
    /// charged to its entry.
    fn held(&self, block: &Self::Block) -> u64;
    /// The device address of `block`'s first byte, which record tables hold.
    fn address(&self, block: &Self::Block) -> u64;
}

/// A flight's completion.
pub trait Fence: Send + Sync + 'static {
    /// Whether the flight has completed. Never waits.
    fn completed(&self) -> bool;
    /// Block until the flight ends, woken by its completion.
    fn wait(&self);
}

/// Run `settle` on a thread of its own once every one of `fences` has
/// ended, the thread blocked on each in turn ([`Fence::wait`]): woken by
/// the flights' completion, never polling and never on a timer. What a
/// pool owing the OS bytes its released granules hold runs at their
/// flights' end ([`GranulePool::awaited`]).
pub fn after<F: Fence>(fences: Vec<Arc<F>>, settle: impl FnOnce() + Send + 'static) -> std::thread::JoinHandle<()> {
    seismic_threads::spawn("pool-return", move || {
        for fence in &fences {
            fence.wait();
        }
        settle();
    })
    .expect("a pool's return thread starts")
}

/// A block of one pool, named for its life: never reused after it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(u64);

/// A granule: its block and its index in that block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GranuleId {
    pub block: BlockId,
    pub index: u32,
}

/// A claimed granule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claimed {
    pub granule: GranuleId,
    /// Its device address.
    pub address: u64,
    /// Whether it reads zero: never held since its block was allocated. A
    /// reused granule holds its last holder's bytes.
    pub zero: bool,
}

/// A live granule's copy out of a vacating block ([`GranulePool::vacate`]):
/// its holder copies `from` into `to` and repoints its table to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move<H> {
    pub holder: H,
    pub from: GranuleId,
    pub from_address: u64,
    pub to: GranuleId,
    pub to_address: u64,
}

/// A block being emptied: the copies its holders make, each holder then
/// releasing the granule it moved from ([`GranulePool::release`]), behind
/// the flight that copies it; once every one is released the block is
/// returning ([`GranulePool::vacated`]), and returns once all its granules
/// are free ([`GranulePool::return_empty`]). Where a copy is not made,
/// [`GranulePool::unvacate`] takes the block back into service.
#[must_use = "a vacating block takes no claims until it empties or is unvacated"]
#[derive(Debug, PartialEq, Eq)]
pub struct Vacating<H> {
    pub block: BlockId,
    pub moves: Vec<Move<H>>,
}

pub use seismic_topology::NativeAllocations;

/// The parameters a pool's block size follows from ([`block_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSizing {
    /// Bytes of one granule.
    pub granule: u64,
    pub allocations: NativeAllocations,
}

/// Why a pool's blocks could not be sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingError {
    /// A granule of no bytes, or allocations of no granularity.
    NoGranule,
    /// The block's bytes overflow a `u64`, or hold more than `u32::MAX`
    /// granules.
    Overflow,
}

/// A pool's block ([`block_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSize {
    /// The bytes of its whole granules: what the pool asks the backend for.
    pub bytes: u64,
    /// What the backend allocates, and so charges, for it: `bytes` rounded
    /// up to its granularity. A block is the unit a pool returns to the
    /// OS, so this is what each block costs and each return frees.
    pub held: u64,
}

impl BlockSize {
    /// The granules it holds.
    pub fn granules(&self, granule: u64) -> u64 {
        self.bytes / granule
    }

    /// What `granules` cost in a pool of these blocks: whole blocks.
    pub fn charge(&self, granule: u64, granules: u64) -> Option<u64> {
        granules.div_ceil(self.granules(granule)).checked_mul(self.held)
    }
}

/// The most a block spans, in units of the larger of the least allocation
/// and the granularity. A block's span costs twice: its rounding
/// (`held − bytes`, under one granularity unit, so under `1/units` of
/// `held` at the span's end) and its size, which is both the least a pool
/// charges and the unit a shrink returns. The two pull opposite ways, so
/// the span is set where both are small against what a pool holds: at 64
/// units the rounding is under 1/64 (1.6%) of a block, while the block is
/// at most 64 units, 128 MiB under CUDA's 2 MiB granularity, 0.5% of a
/// 24 GiB device. Only a granule that fills no whole unit within the span
/// reaches it: one sharing a large power of two with the granularity, as a
/// tensor record does, fills one far sooner (a Qwen record, 27 units in 32
/// granules).
pub const MAX_BLOCK_UNITS: u64 = 64;

/// A pool's block: whole granules, at least the backend's least
/// allocation, of the fewest granules whose bytes are a multiple of its
/// granularity (the least common multiple of the two), so the backend's
/// rounding wastes nothing. Where that block would span more than
/// [`MAX_BLOCK_UNITS`], it is the block within that span wasting least per
/// granule (the fewest granules among equals), its rounding charged. The
/// block is also the unit the pool returns to the OS, so no term of its own
/// sizes a return. Where the backend bounds no count and rounds nothing a
/// block is one granule.
pub fn block_bytes(sizing: BlockSizing) -> Result<BlockSize, SizingError> {
    let BlockSizing { granule, allocations: NativeAllocations { least, granularity } } = sizing;
    if granule == 0 || granularity == 0 {
        return Err(SizingError::NoGranule);
    }
    let held = |granules: u64| {
        granules.checked_mul(granule).and_then(|bytes| Some(BlockSize { bytes, held: bytes.checked_next_multiple_of(granularity)? }))
    };
    let fewest = least.max(granule).div_ceil(granule);
    let unit = least.max(granularity);
    let most = (unit.checked_mul(MAX_BLOCK_UNITS).ok_or(SizingError::Overflow)? / granule).max(fewest);
    let mut best: Option<BlockSize> = None;
    for granules in fewest..=most {
        let block = held(granules).ok_or(SizingError::Overflow)?;
        let waste = block.held - block.bytes;
        // Least waste per granule: `waste / granules` below the best's.
        let better = best.is_none_or(|best| {
            u128::from(waste) * u128::from(best.granules(granule)) < u128::from(best.held - best.bytes) * u128::from(granules)
        });
        if better {
            best = Some(block);
        }
        if waste == 0 {
            break;
        }
    }
    let block = best.expect("a block of the fewest granules is sized");
    if block.granules(granule) > u64::from(u32::MAX) {
        return Err(SizingError::Overflow);
    }
    Ok(block)
}

/// One granule's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Granule<H> {
    /// Claimable; `zero` while never held since its block was allocated.
    Free { zero: bool },
    Held(H),
    /// Released, awaiting the completion of the last flight that read it.
    Fenced,
}

struct Block<K, H> {
    block: K,
    /// The bytes it holds in its store, charged to the pool's entry while
    /// it lives.
    held: u64,
    address: u64,
    granules: Vec<Granule<H>>,
    /// Being emptied ([`GranulePool::vacate`]): its granules take no claims.
    vacating: bool,
    /// Emptied, every live granule copied out and released
    /// ([`GranulePool::vacated`]): it returns once its released granules'
    /// flights complete, and has paid what is owed by its bytes already.
    returning: bool,
}

/// One store's granule pool over `B`'s blocks, its granules held by `H`
/// (the embedder's name of a holder's table slot, which a vacate repoints).
pub struct GranulePool<B: Blocks, H> {
    blocks: B,
    granule: u64,
    block_bytes: u64,
    held: BTreeMap<BlockId, Block<B::Block, H>>,
    /// Claimable granules, oldest block first: claims fill the oldest
    /// blocks, leaving the newest to empty.
    free: BTreeSet<GranuleId>,
    /// Released granules and the completion each waits for.
    fenced: Vec<(GranuleId, Arc<B::Fence>)>,
    /// Granules a holder holds, in every block: the rest are free or
    /// fenced.
    holding: u64,
    /// Bytes its holders freed for outside demand that have not yet
    /// returned to the OS ([`Self::owe`]): each block returned or returning
    /// pays them, and so does each claim that takes a granule back into
    /// use. Never more than the bytes of the granules free or fenced outside
    /// returning blocks, but while a vacate's copies are made (a move holds
    /// both of its granules).
    owed: u64,
    next: u64,
}

impl<B: Blocks, H: Copy + std::fmt::Debug> GranulePool<B, H> {
    /// An empty pool of `granule`-byte granules in `block_bytes`-byte blocks
    /// ([`block_bytes`]).
    ///
    /// # Panics
    /// Where a block is not a whole number of granules, at least one, or
    /// holds more than `u32::MAX` of them.
    pub fn new(blocks: B, granule: u64, block_bytes: u64) -> Self {
        assert!(
            granule > 0 && block_bytes >= granule && block_bytes % granule == 0,
            "invariant violated: a block of {block_bytes} bytes is not whole {granule}-byte granules"
        );
        assert!(block_bytes / granule <= u64::from(u32::MAX), "invariant violated: a block holds at most u32::MAX granules");
        Self { blocks, granule, block_bytes, held: BTreeMap::new(), free: BTreeSet::new(), fenced: Vec::new(), holding: 0, owed: 0, next: 0 }
    }

    pub fn granule(&self) -> u64 {
        self.granule
    }

    pub fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    pub fn backend(&self) -> &B {
        &self.blocks
    }

    /// The bytes the pool's blocks hold in its store: what its entry is
    /// charged.
    pub fn held_bytes(&self) -> u64 {
        self.held.values().map(|block| block.held).sum()
    }

    /// The blocks the pool holds, oldest first.
    pub fn blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.held.keys().copied()
    }

    /// Granules claimable now, in process.
    pub fn free_granules(&mut self) -> usize {
        self.settle();
        self.free.len()
    }

    /// The device address of `granule`.
    ///
    /// # Panics
    /// Where its block is not held.
    pub fn address(&self, granule: GranuleId) -> u64 {
        self.held_block(granule.block).address + u64::from(granule.index) * self.granule
    }

    /// The allocation `block` is.
    ///
    /// # Panics
    /// Where it is not held.
    pub fn block(&self, block: BlockId) -> &B::Block {
        &self.held_block(block).block
    }

    /// Who holds `granule`, where one does.
    pub fn holder(&self, granule: GranuleId) -> Option<H> {
        match self.held_block(granule.block).granules[granule.index as usize] {
            Granule::Held(holder) => Some(holder),
            Granule::Free { .. } | Granule::Fenced => None,
        }
    }

    /// The completions released granules still wait for, which a claim that
    /// found none free waits on: at most one flight.
    pub fn pending(&self) -> impl Iterator<Item = &Arc<B::Fence>> + '_ {
        self.fenced.iter().map(|(_, fence)| fence)
    }

    /// Claim a free granule for `holder`, in process: no allocation and no
    /// OS return. None where no granule is free now. A granule taken back
    /// into use pays what is owed by its bytes, where the granules still
    /// free or fenced no longer cover it: those bytes serve the process
    /// again, and no block need return for them.
    pub fn take(&mut self, holder: H) -> Option<Claimed> {
        self.settle();
        let granule = self.free.pop_first()?;
        let zero = match std::mem::replace(self.slot(granule), Granule::Held(holder)) {
            Granule::Free { zero } => zero,
            state => panic!("invariant violated: free granule {granule:?} was {state:?}"),
        };
        self.holding += 1;
        self.cap_owed();
        Some(Claimed { granule, address: self.address(granule), zero })
    }

    /// Install an already-allocated `block` (allocated outside any lock via
    /// [`Blocks::allocate`]) into the pool, with its granules' zero state set
    /// from `fill` (`true` for [`SlabFill::Zeroed`], `false` for
    /// [`SlabFill::Unwritten`]).
    pub fn install(&mut self, block: B::Block, fill: SlabFill) -> BlockId {
        let held = self.blocks.held(&block);
        assert!(held >= self.block_bytes, "invariant violated: a block of {} bytes holds {held}", self.block_bytes);
        let id = BlockId(self.next);
        self.next += 1;
        let count = (self.block_bytes / self.granule) as u32;
        let address = self.blocks.address(&block);
        let zero = fill == SlabFill::Zeroed;
        self.held.insert(
            id,
            Block { block, held, address, granules: vec![Granule::Free { zero }; count as usize], vacating: false, returning: false },
        );
        self.free.extend((0..count).map(|index| GranuleId { block: id, index }));
        id
    }

    /// Release `granule` from its holder, which has repointed its table off
    /// it: free once `last_read`, the completion of the last flight that
    /// read it, completes (at once where none read it). Nothing moves.
    ///
    /// # Panics
    /// Where it is not held.
    pub fn release(&mut self, granule: GranuleId, last_read: Option<Arc<B::Fence>>) {
        assert!(self.holder(granule).is_some(), "invariant violated: releasing unheld granule {granule:?}");
        self.holding -= 1;
        match last_read {
            Some(fence) if !fence.completed() => {
                *self.slot(granule) = Granule::Fenced;
                self.fenced.push((granule, fence));
            }
            Some(_) | None => self.free(granule),
        }
    }

    /// The held block with the fewest live granules, where it has fewer than
    /// the free granules elsewhere: the cheapest to empty for outside
    /// demand ([`Self::vacate`]). None where no block can be emptied.
    pub fn emptiest(&mut self) -> Option<BlockId> {
        self.settle();
        let free = self.free.len();
        self.held
            .iter()
            .filter(|(_, block)| !block.vacating)
            .map(|(&id, block)| {
                let live = block.granules.iter().filter(|state| !matches!(state, Granule::Free { .. })).count();
                let own_free = block.granules.len() - live;
                (live, id, free - own_free)
            })
            .filter(|&(live, _, elsewhere)| live <= elsewhere)
            .min_by_key(|&(live, id, _)| (live, std::cmp::Reverse(id)))
            .map(|(_, id, _)| id)
    }

    /// Begin emptying `block` for outside demand: each live granule gets a
    /// free granule in another block to be copied into, held by its holder
    /// from now, and the block's granules take no claims. None where a
    /// granule of it awaits a flight or too few are free elsewhere; the
    /// block is then unchanged.
    ///
    /// # Panics
    /// Where `block` is not held or already vacating.
    pub fn vacate(&mut self, block: BlockId) -> Option<Vacating<H>> {
        self.settle();
        let of = self.held_block(block);
        assert!(!of.vacating, "invariant violated: block {block:?} is already vacating");
        if of.granules.iter().any(|state| matches!(state, Granule::Fenced)) {
            return None;
        }
        let live = of
            .granules
            .iter()
            .enumerate()
            .filter_map(|(index, state)| match state {
                Granule::Held(holder) => Some((GranuleId { block, index: index as u32 }, *holder)),
                Granule::Free { .. } | Granule::Fenced => None,
            })
            .collect::<Vec<_>>();
        let elsewhere = self.free.iter().filter(|granule| granule.block != block).take(live.len()).copied().collect::<Vec<_>>();
        if elsewhere.len() < live.len() {
            return None;
        }
        self.held.get_mut(&block).expect("a held block").vacating = true;
        self.free.retain(|granule| granule.block != block);
        let moves = live
            .into_iter()
            .zip(elsewhere)
            .map(|((from, holder), to)| {
                self.free.remove(&to);
                *self.slot(to) = Granule::Held(holder);
                // A move, not a claim: its source is released as the copy
                // completes, so what is owed stays.
                self.holding += 1;
                Move { holder, from, from_address: self.address(from), to, to_address: self.address(to) }
            })
            .collect();
        Some(Vacating { block, moves })
    }

    /// Give up emptying `block` (a copy out of it failed): its free
    /// granules take claims again, and the granules a holder still holds
    /// stay where they are. Granules already copied out stay at their new
    /// places.
    ///
    /// # Panics
    /// Where `block` is not held or not vacating.
    pub fn unvacate(&mut self, block: BlockId) {
        let of = self.held.get_mut(&block).unwrap_or_else(|| panic!("invariant violated: block {block:?} is not held"));
        assert!(of.vacating && !of.returning, "invariant violated: block {block:?} is not vacating, or already returning");
        of.vacating = false;
        let free = of
            .granules
            .iter()
            .enumerate()
            .filter(|(_, state)| matches!(state, Granule::Free { .. }))
            .map(|(index, _)| GranuleId { block, index: index as u32 })
            .collect::<Vec<_>>();
        self.free.extend(free);
    }

    /// Every live granule of vacating `block` was copied out and its source
    /// released ([`Self::release`], behind the flight that copies it): the
    /// block is returning. No claim takes it back; it returns as its
    /// released granules' flights complete ([`Self::return_empty`],
    /// [`Self::awaited`]), and its bytes pay what is owed now, as a block
    /// returned does. The bytes it holds.
    ///
    /// # Panics
    /// Where `block` is not held, not vacating, or a holder still holds a
    /// granule of it.
    pub fn vacated(&mut self, block: BlockId) -> u64 {
        let of = self.held.get_mut(&block).unwrap_or_else(|| panic!("invariant violated: block {block:?} is not held"));
        assert!(of.vacating && !of.returning, "invariant violated: block {block:?} is not vacating, or already returning");
        assert!(
            !of.granules.iter().any(|state| matches!(state, Granule::Held(_))),
            "invariant violated: a holder still holds a granule of vacated block {block:?}"
        );
        of.returning = true;
        let held = of.held;
        self.owed -= self.owed.min(held);
        held
    }

    /// `bytes` its holders freed for outside demand are owed to the OS
    /// until blocks returned pay them ([`Self::return_empty`]), or claims
    /// take them back into use ([`Self::take`]): what a claim since the
    /// shrink already reused is owed no more.
    pub fn owe(&mut self, bytes: u64) {
        self.owed = self.owed.checked_add(bytes).expect("a pool's owed bytes fit u64");
        self.cap_owed();
    }

    /// Bytes of the granules no holder holds, free or awaiting a flight,
    /// outside returning blocks: those have paid already. A returning block
    /// holds no held granule ([`Self::vacated`]).
    fn unheld_bytes(&self) -> u64 {
        let serving = self.held.values().filter(|block| !block.returning).count() as u64;
        (serving * (self.block_bytes / self.granule) - self.holding) * self.granule
    }

    /// What is owed is at most what could return: the granules still free
    /// or fenced outside returning blocks.
    fn cap_owed(&mut self) {
        self.owed = self.owed.min(self.unheld_bytes());
    }

    /// Bytes owed to the OS not yet returned.
    pub fn owed(&self) -> u64 {
        self.owed
    }

    /// Bytes of released granules still awaiting the flight that read
    /// them, outside returning blocks (which paid already): freed as it
    /// completes.
    pub fn awaiting(&mut self) -> u64 {
        self.settle();
        let held = &self.held;
        self.fenced.iter().filter(|(granule, _)| !held[&granule.block].returning).count() as u64 * self.granule
    }

    /// The completions a return waits for ([`after`]): those of a
    /// returning block's released granules, and where bytes are owed, of
    /// every released granule. None where none awaits a flight.
    pub fn awaited(&mut self) -> Option<Vec<Arc<B::Fence>>> {
        self.settle();
        let (held, owed) = (&self.held, self.owed > 0);
        let fences = self
            .fenced
            .iter()
            .filter(|(granule, _)| owed || held[&granule.block].returning)
            .map(|(_, fence)| fence.clone())
            .collect::<Vec<_>>();
        (!fences.is_empty()).then_some(fences)
    }

    /// Take every block whose granules are all free out of the pool, each
    /// with the bytes its return reports: those it holds, but none for a
    /// returning block, whose bytes its vacate reported
    /// ([`Self::vacated`]). Dropping one returns it to the OS and frees its
    /// charge from the pool's entry, which its caller does outside whatever
    /// guards the pool, since a backend may wait for the device to finish
    /// with a block's storage before releasing it. What they hold pays what
    /// is owed first, but a returning block's, which paid as it was
    /// vacated; a block returned beyond it owes nothing.
    #[must_use = "a returned block is released as it drops"]
    pub fn return_empty(&mut self) -> Vec<(B::Block, u64)> {
        self.settle();
        let empty = self
            .held
            .iter()
            .filter(|(_, block)| block.granules.iter().all(|state| matches!(state, Granule::Free { .. })))
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        empty
            .into_iter()
            .map(|id| {
                let block = self.held.remove(&id).expect("an empty block is held");
                self.free.retain(|granule| granule.block != id);
                if block.returning {
                    return (block.block, 0);
                }
                self.owed -= self.owed.min(block.held);
                (block.block, block.held)
            })
            .collect()
    }

    fn held_block(&self, block: BlockId) -> &Block<B::Block, H> {
        self.held.get(&block).unwrap_or_else(|| panic!("invariant violated: block {block:?} is not held"))
    }

    fn slot(&mut self, granule: GranuleId) -> &mut Granule<H> {
        &mut self
            .held
            .get_mut(&granule.block)
            .unwrap_or_else(|| panic!("invariant violated: block {:?} is not held", granule.block))
            .granules[granule.index as usize]
    }

    fn free(&mut self, granule: GranuleId) {
        *self.slot(granule) = Granule::Free { zero: false };
        if !self.held_block(granule.block).vacating {
            self.free.insert(granule);
        }
    }

    /// Free the released granules whose flights completed.
    fn settle(&mut self) {
        let mut index = 0;
        while index < self.fenced.len() {
            if self.fenced[index].1.completed() {
                let (granule, _) = self.fenced.swap_remove(index);
                self.free(granule);
            } else {
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests;
