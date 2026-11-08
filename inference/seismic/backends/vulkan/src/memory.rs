//! The owned block allocator (§8.2, §15.4) over the shared block core
//! (`seismic_compiler::blocks`).
//!
//! Device memory is allocated in blocks, each one `VkBuffer` with a device
//! address; an allocation is a range of a block, and its address is the
//! block's address plus the range's offset. A pool holds the blocks of one
//! ordered list of kinds, as the placement rule names them; its blocks grow
//! from 64 MiB to the block cap (min(`maxMemoryAllocationSize`,
//! `maxBufferSize`, 1 GiB)); a request of more than half the first block gets a
//! block of its own.
//! A block is freed once it holds no allocation, so released storage returns
//! to the device's memory budget: the engine reads the budget less usage as
//! its headroom, and a block kept empty would hide released storage from it.
//! A block's unallocated space counts against that budget too, so a new
//! block of a pool the caller's room bounds never exceeds that room beyond
//! the free space such blocks already hold, and when a doubled block does
//! not fit that room the block holds only its allocation: an allocation
//! that fits its grant never grows the device's usage past it. That free
//! space is memory this process holds beyond its charges
//! ([`Memory::room_bounded_slack`], and per pool [`Memory::free_bytes`]).
//! Host-visible blocks are persistently mapped.

use crate::device::{call, Inner};
use crate::instance::Instance;
use ash::vk;
use seismic_compiler::blocks::{
    block_bytes, first_fit, largest_shared, takes_own_block, FreeRanges, PackedBlock,
    BLOCK_CAP_BYTES,
};
use seismic_compiler::errors::ExecutionError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
/// Every allocation starts at least this aligned.
const MINIMUM_ALIGNMENT: u64 = 16;

/// The accounts a device's memory is charged to, which it is opened with:
/// each kind's blocks to its kind's, staging to host RAM's, and the
/// driver's memory beyond every block to `driver`.
pub struct MemoryAccounts {
    pub host: Arc<dyn seismic_compiler::executable::MemoryAccount>,
    pub kinds: Vec<AllocationAccount>,
    pub driver: Arc<dyn seismic_compiler::executable::MemoryAccount>,
}

impl MemoryAccounts {
    /// The account blocks of `kind` are charged to. The opener names an
    /// account for every kind its placements allocate in, the only kinds a
    /// pool or reservation of the device allocates in.
    fn kind(&self, kind: u32) -> &Arc<dyn seismic_compiler::executable::MemoryAccount> {
        &self
            .kinds
            .iter()
            .find(|account| account.kind == kind)
            .unwrap_or_else(|| panic!("the device was opened with no account for kind {kind}, which it allocates in"))
            .account
    }

    /// The account a new block of `pool` on `candidate` is charged to.
    fn pool(&self, pool: &Pool, candidate: &PoolType) -> &Arc<dyn seismic_compiler::executable::MemoryAccount> {
        match pool.charged {
            Charged::Staging => &self.host,
            Charged::Kind => self.kind(candidate.memory_type),
        }
    }
}

/// The account new blocks of `kind` are charged to.
pub struct AllocationAccount {
    /// The kind (`crate::topology`) whose blocks it is charged.
    pub kind: u32,
    /// The account itself.
    pub account: Arc<dyn seismic_compiler::executable::MemoryAccount>,
}

pub(crate) struct Block {
    heap: u32,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    address: u64,
    /// Usable buffer extent, distinct from the allocation's required padding.
    bytes: u64,
    allocated_bytes: u64,
    /// Base of the persistent mapping; null for unmapped memory.
    mapped: *mut u8,
    dedicated: bool,
    free: Mutex<FreeRanges>,
    /// The block's charge to its pool's account, released with the block.
    _charge: Box<dyn seismic_compiler::executable::MemoryHold>,
}

// The mapping is plain host memory; ranges are handed out exclusively.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl PackedBlock for Block {
    fn bytes(&self) -> u64 {
        self.bytes
    }
    fn own(&self) -> bool {
        self.dedicated
    }
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        self.ranges().take(bytes, alignment)
    }
}

impl Block {
    /// What the block takes of `heaps`: its whole allocation, padding
    /// included, where it lies on one of them.
    fn heap_bytes(&self, heaps: &[u32]) -> u64 {
        if heaps.contains(&self.heap) {
            self.allocated_bytes
        } else {
            0
        }
    }

    /// Return a range; true when the block is entirely free afterwards.
    fn give(&self, offset: u64, bytes: u64) -> bool {
        self.ranges().give(offset, bytes)
    }

    fn ranges(&self) -> std::sync::MutexGuard<'_, FreeRanges> {
        self.free
            .lock()
            .expect("block free list lock is never poisoned")
    }
}

/// The blocks of one ordered list of kinds.
pub(crate) struct Pool {
    /// The kinds it was asked for, which identify it.
    kinds: Vec<u32>,
    /// What its blocks are charged to.
    charged: Charged,
    /// The memory types new blocks are allocated from, in order: each of
    /// these is passed over when its heap cannot hold the block, and `last`
    /// is left to the driver.
    passable: Vec<PoolType>,
    last: PoolType,
    /// Whether the caller's room bounds the pool's blocks: its first type is
    /// device-local. Staging's never is.
    room_bounded: bool,
    blocks: Mutex<Vec<Arc<Block>>>,
    /// Bytes of the device's own ranges in the pool's blocks (argument
    /// chunks, staging), which live until the device is destroyed: held,
    /// charged to no allocation.
    internal: AtomicU64,
}

/// What a pool's blocks are charged to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Charged {
    /// Each block's kind's account.
    Kind,
    /// The host account: staging is host RAM's whatever heap it is on.
    Staging,
}

/// One memory type a pool allocates blocks from.
#[derive(Clone, Copy, Debug)]
struct PoolType {
    memory_type: u32,
    heap: u32,
    /// Host-visible and coherent: its blocks are persistently mapped.
    mapped: bool,
}

impl Pool {
    /// Return `range` to its block. A block left without an allocation
    /// leaves the pool and is returned for the caller to free. The pool
    /// lock is held across the return, so no allocation can take a range of
    /// a block being removed.
    fn give(&self, range: &Range) -> Option<Arc<Block>> {
        let mut blocks = self.blocks.lock().expect("pool lock is never poisoned");
        if !range.block.give(range.offset, range.bytes) {
            return None;
        }
        let index = blocks
            .iter()
            .position(|block| Arc::ptr_eq(block, &range.block))
            .expect("a live range's block belongs to its pool");
        Some(blocks.remove(index))
    }

    /// The free space of the pool's blocks.
    fn free_bytes(&self) -> u64 {
        self.blocks
            .lock()
            .expect("pool lock is never poisoned")
            .iter()
            .map(|block| block.ranges().free_bytes())
            .sum()
    }
}

pub(crate) struct Memory {
    /// The pools asked of it so far, one per ordered list of kinds.
    pools: Mutex<Vec<Arc<Pool>>>,
    /// Host-visible staging for transfers of unmapped device storage.
    staging: Arc<Pool>,
    /// The backend's own device and upload storage, as the placement rule
    /// places them.
    device: Arc<Pool>,
    upload: Arc<Pool>,
    /// The memory types pools are made of.
    layout: crate::MemoryLayout,
    /// The types host pages are imported as (`Allocation::host`).
    imports: Vec<u32>,
    /// The queue families that reach each buffer: the work queue's, and the
    /// second queue's where that is a family of its own.
    families: Vec<u32>,
    /// The heaps of the device's own memory, whose usage its memory budget
    /// reads.
    device_heaps: Vec<u32>,
    block_cap: u64,
    max_allocation: u64,
    /// Where heap budgets are read.
    instance: Arc<Instance>,
    physical: vk::PhysicalDevice,
    /// Bytes of every block on the heaps device storage is allocated from,
    /// pooled and chunks alike, counted within each block's transition.
    device_blocks: AtomicU64,
    /// Block transitions begun and ended: a block's creation from before
    /// its allocation until it is counted, its freeing from its uncount
    /// until after its memory is freed ([`Transition`]). A reading with as
    /// many ended as begun, and none begun across it, saw no block between
    /// the heap and its count.
    transitions_begun: AtomicU64,
    transitions_ended: AtomicU64,
    /// What a reading waiting for transitions to end parks on: each end
    /// is counted and notified under its lock, so no end is missed.
    quiet: Mutex<()>,
    ended: std::sync::Condvar,
    /// The driver's device memory beyond every block: see [`DriverMemory`].
    driver: DriverMemory,
}

/// Device memory the driver holds for the device outside every block: the
/// logical device itself, pipelines, command and query memory, shader
/// scratch, staging and second-queue objects.
///
/// I9 for driver memory: **it is never under-charged at an admission
/// decision against the device's store.** Every such decision is made by
/// the runtime's store ledger (`seismic_runtime` memory, `StoreLedger`),
/// which runs this device's reconciliation ([`Memory::reconcile_for_admission`],
/// through `Device::admission`) under its decision lock, then decides, so
/// the reading and the decision are one step: no other admission against
/// the store, from this executor, another executor on the store or a CPU
/// executor on unified memory, decides between them. That covers each
/// logical charge (an allocation's, a reservation's and its growth's) and
/// each native block's charge, which also sizes the block and picks its
/// memory type within that decision, by the reconciliation's own budgets
/// ([`Memory::budgets`], their one source), before its memory joins the
/// heap. Every device charging the store registers its reconciliation, and
/// all of them run. Readings of what the device holds beyond its
/// allocations reconcile too.
///
/// A reconciliation reads the heaps' usage less every block on them less
/// the usage before the device existed, with no block in transition (a
/// block counts within one transition from before its memory joins the
/// heap, and frees within one), so it is exactly what the driver holds at
/// that reading. The charge is the high-water of the readings, never
/// returned, as the driver keeps what it took for reuse. Not covered:
/// growth the driver takes between the reading and the decision (a queue
/// submission on another thread within that step), and a store admission
/// made while no ledger hook is set (before the device's backend is
/// opened). The one assumption is that `VK_EXT_memory_budget` reports
/// usage current when read: a stale reading under-charges by its lag.
struct DriverMemory {
    /// The process's usage of the heaps before the device was created.
    baseline: u64,
    /// The charge: the most the reconciled readings found.
    charged: AtomicU64,
    /// The heaps' budgets as the latest reconciliation read them: what a
    /// block's admission sizes it by, read within the same decision.
    budgets: Mutex<Vec<crate::HeapBudget>>,
    holds: Mutex<Vec<Box<dyn seismic_compiler::executable::MemoryHold>>>,
}

thread_local! {
    /// The block transitions this thread has open: none may be while it
    /// reconciles for an admission, which would wait on its own.
    static OPEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A block's transition ([`Memory::transition`]), ended when it drops, on
/// the thread that opened it (`OPEN` counts per thread), so it is not Send.
struct Transition<'a>(&'a Memory, std::marker::PhantomData<*const ()>);

impl Drop for Transition<'_> {
    fn drop(&mut self) {
        OPEN.with(|open| open.set(open.get() - 1));
        let _quiet = self.0.quiet.lock().expect("transitions lock is never poisoned");
        self.0.transitions_ended.fetch_add(1, Ordering::SeqCst);
        self.0.ended.notify_all();
    }
}

impl Memory {
    /// The backend's own device and upload storage, on the kinds the
    /// placement rule gives them, and staging on `facts.allocation`'s type.
    pub(crate) fn new(
        facts: &crate::Facts,
        instance: Arc<Instance>,
        physical: vk::PhysicalDevice,
        baseline: u64,
    ) -> Result<Self, String> {
        let layout = &facts.memory;
        let staging = facts
            .allocation
            .staging
            .ok_or("no host staging memory type")?;
        let kinds = crate::topology::kinds(facts);
        let placed = |contract| {
            let kinds = seismic_topology::place(&kinds, facts.device_type.memory_is_host(), seismic_topology::Placing::Fixed(contract));
            new_pool(layout, &kinds, Charged::Kind).map(Arc::new)
        };
        let (device, upload) = (placed(seismic_topology::Contract::DeviceOnly)?, placed(seismic_topology::Contract::Upload)?);
        Ok(Self {
            pools: Mutex::new(vec![device.clone(), upload.clone()]),
            staging: Arc::new(new_pool(layout, &[staging], Charged::Staging)?),
            device,
            upload,
            layout: layout.clone(),
            imports: facts.allocation.host.clone(),
            families: std::iter::once(facts.queue_family).chain(facts.side_family.filter(|&family| family != facts.queue_family)).collect(),
            device_heaps: facts.allocation.device_heaps.clone(),
            block_cap: facts.limits.max_allocation_bytes.min(BLOCK_CAP_BYTES),
            max_allocation: facts.limits.max_allocation_bytes,
            instance,
            physical,
            device_blocks: AtomicU64::new(0),
            transitions_begun: AtomicU64::new(0),
            transitions_ended: AtomicU64::new(0),
            quiet: Mutex::new(()),
            ended: std::sync::Condvar::new(),
            driver: DriverMemory {
                baseline,
                charged: AtomicU64::new(0),
                budgets: Mutex::new(Vec::new()),
                holds: Mutex::new(Vec::new()),
            },
        })
    }

    /// Begin a block's transition: until it drops, no reconciliation reads.
    fn transition(&self) -> Transition<'_> {
        OPEN.with(|open| open.set(open.get() + 1));
        self.transitions_begun.fetch_add(1, Ordering::SeqCst);
        Transition(self, std::marker::PhantomData)
    }

    /// Count `block` on the device heaps, within its creation's transition.
    fn count(&self, block: &Block) {
        self.device_blocks.fetch_add(block.heap_bytes(&self.device_heaps), Ordering::SeqCst);
    }

    /// Free `block` (no allocation is in it) and uncount it, within one
    /// transition.
    pub(crate) fn free(&self, device: &ash::Device, block: &Block) {
        let _transition = self.transition();
        self.device_blocks.fetch_sub(block.heap_bytes(&self.device_heaps), Ordering::SeqCst);
        destroy_block(device, block);
    }

    /// Reconcile the driver's memory ([`DriverMemory`]): read the blocks and
    /// the heaps' budgets with no block in transition ([`Self::unheld`]),
    /// keep the budgets ([`Self::budgets`]), and charge where the usage less
    /// the blocks and the baseline exceeds the charge. Takes no lock a
    /// block's creation or freeing holds, so it runs from anywhere but
    /// inside a transition. What it newly finds is charged to `account`,
    /// the driver account of the open that reconciles.
    pub(crate) fn reconcile(&self, account: &dyn seismic_compiler::executable::MemoryAccount) {
        let (budgets, unheld) = self.unheld();
        *self.driver.budgets.lock().expect("driver budgets lock poisoned") = budgets;
        let mut holds = self.driver.holds.lock().expect("driver holds lock poisoned");
        let charged = self.driver.charged.load(Ordering::SeqCst);
        if unheld > charged {
            self.driver.charged.store(unheld, Ordering::SeqCst);
            holds.push(account.record(unheld - charged));
        }
    }

    /// The heaps' budgets as the latest reconciliation read them: within an
    /// admission's decision, that decision's own reading.
    fn budgets(&self) -> Vec<crate::HeapBudget> {
        self.driver.budgets.lock().expect("driver budgets lock poisoned").clone()
    }

    /// The heaps' budgets, and what of the device heaps' usage is the
    /// driver's: the usage less the blocks and the baseline, read with no
    /// block in transition. A transition in flight, or one begun across the
    /// reading, parks it until transitions end and it reads again.
    pub(crate) fn unheld(&self) -> (Vec<crate::HeapBudget>, u64) {
        loop {
            {
                let mut quiet = self.quiet.lock().expect("transitions lock is never poisoned");
                while self.transitions_begun.load(Ordering::SeqCst) != self.transitions_ended.load(Ordering::SeqCst) {
                    quiet = self.ended.wait(quiet).expect("transitions lock is never poisoned");
                }
            }
            let begun = self.transitions_begun.load(Ordering::SeqCst);
            if begun != self.transitions_ended.load(Ordering::SeqCst) {
                continue;
            }
            let held = self.device_blocks.load(Ordering::SeqCst);
            let budgets = crate::device::heap_budgets(&self.instance, self.physical);
            if self.transitions_begun.load(Ordering::SeqCst) == begun {
                let usage = self.device_heaps.iter().map(|&heap| budgets[heap as usize].usage).sum::<u64>();
                return (budgets, usage.saturating_sub(held).saturating_sub(self.driver.baseline));
            }
        }
    }

    /// Reconcile for an admission against the device's store, which the
    /// runtime's store ledger runs under its decision lock before each
    /// (`Device::admission`). A block's own admission charges before its
    /// transition opens (`create_block`), so no transition is open here.
    pub(crate) fn reconcile_for_admission(&self, account: &dyn seismic_compiler::executable::MemoryAccount) {
        assert_eq!(OPEN.with(std::cell::Cell::get), 0, "an admission reconciles inside a block transition");
        self.reconcile(account);
    }

    /// The driver's memory as last reconciled.
    pub(crate) fn driver_bytes(&self) -> u64 {
        self.driver.charged.load(Ordering::SeqCst)
    }

    /// `VK_EXT_memory_budget` of the heaps device storage is allocated from.
    pub(crate) fn memory_budget(&self) -> crate::MemoryBudget {
        crate::device::memory_budget(&self.instance, self.physical, &self.device_heaps)
    }

    /// The pool of the ordered list of memory type `kinds`, where one has
    /// been asked for.
    pub(crate) fn pooled(&self, kinds: &[u32]) -> Option<Arc<Pool>> {
        self.pools.lock().expect("pools lock is never poisoned").iter().find(|pool| pool.kinds == kinds).cloned()
    }

    /// The pool of the ordered list of memory type `kinds`.
    pub(crate) fn pool(&self, kinds: &[u32]) -> Result<Arc<Pool>, ExecutionError> {
        let mut pools = self.pools.lock().expect("pools lock is never poisoned");
        if let Some(pool) = pools.iter().find(|pool| pool.kinds == kinds) {
            return Ok(pool.clone());
        }
        let pool = Arc::new(new_pool(&self.layout, kinds, Charged::Kind).map_err(ExecutionError::AllocationFailed)?);
        pools.push(pool.clone());
        Ok(pool)
    }

    /// The pool of the backend's own upload storage.
    pub(crate) fn upload(&self) -> &Arc<Pool> {
        &self.upload
    }

    /// The pool of the backend's own device storage.
    pub(crate) fn device(&self) -> &Arc<Pool> {
        &self.device
    }

    /// The pool of staging.
    pub(crate) fn staging(&self) -> &Arc<Pool> {
        &self.staging
    }

    /// Free space held by room-bounded blocks: device usage that no
    /// allocation charges.
    pub(crate) fn room_bounded_slack(&self) -> u64 {
        self.pools
            .lock()
            .expect("pools lock is never poisoned")
            .iter()
            .filter(|pool| pool.room_bounded)
            .map(|pool| pool.free_bytes())
            .sum()
    }

    /// The free space of `pool`'s blocks.
    pub(crate) fn free_bytes(&self, pool: &Pool) -> u64 {
        pool.free_bytes()
    }

    /// Bytes of the device's own ranges in `pool`'s blocks.
    pub(crate) fn internal_bytes(&self, pool: &Pool) -> u64 {
        pool.internal.load(Ordering::Relaxed)
    }

    /// A range of `bytes`. `room`, when given, is the device usage the
    /// caller may add: a new block of a room-bounded pool is sized within it
    /// less the free space room-bounded blocks already hold, and an
    /// allocation that needs a block larger than that is refused as a
    /// capacity shortfall.
    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
        accounts: &MemoryAccounts,
    ) -> Result<Range, ExecutionError> {
        if bytes > self.max_allocation {
            return Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: self.max_allocation,
            });
        }
        let bytes = bytes.max(1);
        let alignment = alignment.max(MINIMUM_ALIGNMENT);
        // Read before taking this pool's lock: the slack spans every pool.
        let room = room
            .filter(|_| pool.room_bounded)
            .map(|room| room.saturating_sub(self.room_bounded_slack()));
        let mut blocks = pool.blocks.lock().expect("pool lock is never poisoned");
        let found = first_fit(&blocks, bytes, alignment)
            .map(|(index, offset)| (blocks[index].clone(), offset));
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                // A new block, decided under the pool's lock: its charge is
                // the store's decision (`DriverMemory`), made before its
                // transition opens.
                let largest = largest_shared(&blocks);
                let dedicated = takes_own_block(bytes, self.block_cap);
                let block = self.create(device, pool, room, dedicated, Mapping::Pool, accounts, |within| {
                    block_bytes(largest, self.block_cap, bytes, within, 1)
                })?;
                let offset = block
                    .take(bytes, alignment)
                    .expect("a new block holds its first allocation");
                blocks.push(block.clone());
                (block, offset)
            }
        };
        Ok(Range {
            pool: Some(pool.clone()),
            block,
            offset,
            bytes,
        })
    }

    /// A new block of `pool`, of the size `size` gives within what
    /// the block may take (none bounds it when `None`). A type whose heap's
    /// budget cannot hold the block, or whose heap refuses it, is passed
    /// over for the next; the last type, with none after it, is left to the
    /// driver.
    fn create(
        &self,
        device: &ash::Device,
        pool: &Pool,
        room: Option<u64>,
        dedicated: bool,
        mapping: Mapping<'_>,
        accounts: &MemoryAccounts,
        size: impl Fn(Option<u64>) -> Result<u64, ExecutionError>,
    ) -> Result<Arc<Block>, ExecutionError> {
        // What a block of `candidate` may take, decided within its
        // admission: the caller's room, what its account has available and,
        // for a passable type, what its heap's budget leaves by the
        // decision's own reading; none bounds it when none is given.
        let within = |candidate: &PoolType, passable: bool, available: u64| {
            let heap_room = passable.then(|| {
                let heap = self.budgets()[candidate.heap as usize];
                heap.budget.saturating_sub(heap.usage)
            });
            [heap_room, Some(available), room].into_iter().flatten().min()
        };
        for candidate in &pool.passable {
            match create_block(self, device, *candidate, dedicated, accounts.pool(pool, candidate), mapping, |available| {
                size(within(candidate, true, available))
            }) {
                Ok(block) => return Ok(block),
                Err(error) if error.capacity_refusal().is_some() => {}
                Err(error) => return Err(error),
            }
        }
        create_block(self, device, pool.last, dedicated, accounts.pool(pool, &pool.last), mapping, |available| {
            size(within(&pool.last, false, available))
        })
    }

    /// A block of `pool` of exactly `bytes` for one chunk of a reservation
    /// (`crate::reservation`), its own memory object, charged to its kind's
    /// account like every block and freed when it drops ([`Chunk`]).
    /// Mapped as `mapping` says: as its pool's type, or whole at a placed
    /// host address (`Mapping::Placed`).
    pub(crate) fn chunk(
        &self,
        device: &ash::Device,
        pool: &Pool,
        bytes: u64,
        mapping: Mapping<'_>,
        accounts: &MemoryAccounts,
    ) -> Result<Arc<Block>, ExecutionError> {
        self.create(device, pool, None, true, mapping, accounts, |within| match within {
            Some(within) if within < bytes => Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: within,
            }),
            _ => Ok(bytes),
        })
    }

    /// A block over the `bytes` of host memory at `pointer`, imported
    /// (`VK_EXT_external_memory_host`): as the first of the import types
    /// that `importable` (the pointer's memory type bits) admits, charged to
    /// imported pages' account and freed when it drops. The caller keeps the
    /// host memory mapped while the block lives.
    pub(crate) fn import(
        &self,
        device: &ash::Device,
        pointer: std::ptr::NonNull<u8>,
        bytes: u64,
        importable: u32,
        accounts: &MemoryAccounts,
    ) -> Result<Arc<Block>, ExecutionError> {
        let memory_type = self
            .imports
            .iter()
            .copied()
            .find(|memory_type| importable & (1 << memory_type) != 0)
            .ok_or_else(|| {
                ExecutionError::AllocationFailed(format!(
                    "no memory type imports host memory (importable types {importable:#x})"
                ))
            })?;
        let account = accounts.kind(crate::topology::IMPORTED_KIND);
        create_block(self, device, pool_type(&self.layout, memory_type), true, account, Mapping::Imported(pointer), |_| Ok(bytes))
    }

    /// A range for the device's own use (argument chunks, staging), kept
    /// until the device is destroyed and counted as held by its pool.
    pub(crate) fn allocate_internal(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        accounts: &MemoryAccounts,
    ) -> Result<Range, ExecutionError> {
        let range = self.allocate(device, pool, bytes, alignment, None, accounts)?;
        pool.internal.fetch_add(range.bytes, Ordering::Relaxed);
        Ok(range)
    }

    /// Return `range` to its block, and free the block once it holds no
    /// allocation.
    pub(crate) fn release(&self, device: &ash::Device, range: &Range) {
        let pool = range.pool.as_ref().expect("only a pool's ranges are released");
        if let Some(block) = pool.give(range) {
            self.free(device, &block);
        }
    }

    /// Destroy every block. The device is idle and no allocation lives.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        let pools = self.pools.lock().expect("pools lock is never poisoned");
        for pool in pools.iter().chain([&self.staging]) {
            let blocks = std::mem::take(&mut *pool.blocks.lock().expect("pool lock is never poisoned"));
            for block in blocks {
                self.free(device, &block);
            }
        }
    }
}

/// The memory type `memory_type` of `layout`, as a pool allocates from it.
fn pool_type(layout: &crate::MemoryLayout, memory_type: u32) -> PoolType {
    let reported = layout.types[memory_type as usize];
    PoolType {
        memory_type,
        heap: reported.heap,
        mapped: reported.mappable(),
    }
}

/// An empty pool of memory types `kinds`, in order, charged as `charged`.
fn new_pool(layout: &crate::MemoryLayout, kinds: &[u32], charged: Charged) -> Result<Pool, String> {
    let (&last, passable) = kinds.split_last().ok_or("no memory type for the storage")?;
    Ok(Pool {
        kinds: kinds.to_vec(),
        charged,
        passable: passable.iter().map(|&memory_type| pool_type(layout, memory_type)).collect(),
        last: pool_type(layout, last),
        room_bounded: charged == Charged::Kind && layout.types[kinds[0] as usize].device_local,
        blocks: Mutex::new(Vec::new()),
        internal: AtomicU64::new(0),
    })
}

fn destroy_block(device: &ash::Device, block: &Block) {
    unsafe {
        device.destroy_buffer(block.buffer, None);
        device.free_memory(block.memory, None);
    }
}

/// Native allocation extent, bounded by the actual host region for imports.
fn allocation_extent(required: u64, imported: Option<u64>) -> Result<u64, ExecutionError> {
    match imported {
        Some(owned) if required > owned => Err(ExecutionError::AllocationFailed(format!(
            "Vulkan buffer requires {required} allocation bytes but its host import owns {owned} bytes"
        ))),
        Some(owned) => Ok(owned),
        None => Ok(required),
    }
}

/// How a block is mapped for the host ([`create_block`]).
#[derive(Clone, Copy)]
pub(crate) enum Mapping<'a> {
    /// As its pool's type: persistently, where the type is mappable.
    Pool,
    /// Host memory imported (`VK_EXT_external_memory_host`), which is its
    /// mapping.
    Imported(std::ptr::NonNull<u8>),
    /// Whole at a placed host address, which `map` maps it at
    /// (`Inner::map_placed`).
    Placed(std::ptr::NonNull<u8>, &'a dyn Fn(vk::DeviceMemory) -> Result<(), ExecutionError>),
}

/// A block of `bytes` on `pool`'s type, charged to `account`: memory the
/// driver allocates, or the host memory it imports, mapped as `mapping`
/// says.
fn create_block(
    memory_owner: &Memory,
    device: &ash::Device,
    pool: PoolType,
    dedicated: bool,
    account: &Arc<dyn seismic_compiler::executable::MemoryAccount>,
    mapping: Mapping<'_>,
    size: impl FnOnce(u64) -> Result<u64, ExecutionError>,
) -> Result<Arc<Block>, ExecutionError> {
    let import = match mapping {
        Mapping::Imported(pointer) => Some(pointer),
        Mapping::Pool | Mapping::Placed(..) => None,
    };
    let allocation = |error: ExecutionError| match error {
        ExecutionError::SubmissionFailed(message) => ExecutionError::AllocationFailed(message),
        other => other,
    };
    // The block's buffer of the size `size` decides from what the account
    // has available, and the bytes its memory takes.
    let mut size = Some(size);
    let mut form = |available: u64| -> Result<(vk::Buffer, u64, u64), ExecutionError> {
        let bytes = (size.take().expect("a block is sized once"))(available)?;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let families = memory_owner.families.as_slice();
        let mut info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(
                // Any buffer may hold a grid a launch wrote (`DirectLaunch::grid`).
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
            )
            .sharing_mode(if families.len() > 1 { vk::SharingMode::CONCURRENT } else { vk::SharingMode::EXCLUSIVE })
            .queue_family_indices(if families.len() > 1 { families } else { &[] });
        if import.is_some() {
            info = info.push_next(&mut external);
        }
        let buffer = call(unsafe { device.create_buffer(&info, None) }, "vkCreateBuffer").map_err(allocation)?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let refuse = |error: ExecutionError| -> Result<(vk::Buffer, u64, u64), ExecutionError> {
            unsafe { device.destroy_buffer(buffer, None) };
            Err(error)
        };
        let allocated_bytes = match allocation_extent(requirements.size, import.map(|_| bytes)) {
            Ok(bytes) => bytes,
            Err(error) => return refuse(error),
        };
        if requirements.memory_type_bits & (1 << pool.memory_type) == 0 {
            return refuse(ExecutionError::AllocationFailed(format!(
                "memory type {} cannot back a storage buffer",
                pool.memory_type
            )));
        }
        Ok((buffer, bytes, allocated_bytes))
    };
    // Sized and charged in one decision of the account's stores, by that
    // decision's reading (`DriverMemory`).
    let mut formed = None;
    let decided = account.reserve_decided(&mut |available| {
        let (buffer, bytes, allocated) = form(available)?;
        formed = Some((buffer, bytes));
        Ok(allocated)
    });
    let (buffer, bytes, allocated_bytes, charge) = match decided {
        Ok((charge, allocated)) => {
            let (buffer, bytes) = formed.expect("a decided charge formed its block's buffer");
            (buffer, bytes, allocated, charge)
        }
        Err(error) => {
            if let Some((buffer, _)) = formed {
                unsafe { device.destroy_buffer(buffer, None) };
            }
            return Err(error);
        }
    };
    // The charge decided, the block's memory joins the heap within one
    // transition, until it is counted.
    let _transition = memory_owner.transition();
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut imported = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
        .host_pointer(import.map_or(std::ptr::null_mut(), |pointer| pointer.as_ptr().cast()));
    let mut allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(allocated_bytes)
        .memory_type_index(pool.memory_type)
        .push_next(&mut flags);
    if import.is_some() {
        allocate = allocate.push_next(&mut imported);
    }
    let memory = match unsafe { device.allocate_memory(&allocate, None) } {
        Ok(memory) => memory,
        Err(result) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(match result {
                // The driver's own refusal, with its budget read now.
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    let budget = memory_owner.memory_budget();
                    ExecutionError::DeviceRefused {
                        required: bytes,
                        reading: seismic_compiler::errors::RefusalReading::HeapBudget {
                            budget: budget.heap_budget_bytes,
                            usage: budget.heap_usage_bytes,
                        },
                    }
                }
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => ExecutionError::DeviceRefused {
                    required: bytes,
                    reading: seismic_compiler::errors::RefusalReading::Host,
                },
                other => {
                    ExecutionError::AllocationFailed(format!("vkAllocateMemory failed: {other}"))
                }
            });
        }
    };
    let bound = call(
        unsafe { device.bind_buffer_memory(buffer, memory, 0) },
        "vkBindBufferMemory",
    );
    let mapped = bound.and_then(|()| match mapping {
        Mapping::Imported(pointer) => Ok(pointer.as_ptr()),
        Mapping::Placed(at, map) => map(memory).map(|()| at.as_ptr()),
        Mapping::Pool if pool.mapped => call(
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) },
            "vkMapMemory",
        )
        .map(|pointer| pointer.cast::<u8>()),
        Mapping::Pool => Ok(std::ptr::null_mut()),
    });
    let mapped = match mapped {
        Ok(mapped) => mapped,
        Err(error) => {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(allocation(error));
        }
    };
    let address = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };
    let block = Arc::new(Block {
        heap: pool.heap,
        memory,
        buffer,
        address,
        bytes,
        allocated_bytes,
        mapped,
        dedicated,
        free: Mutex::new(FreeRanges::new(bytes)),
        _charge: charge,
    });
    memory_owner.count(&block);
    Ok(block)
}

/// A range of a block. [`Buffer`] releases it on drop; the device's own
/// ranges (argument chunks, staging) live until the device is destroyed.
pub(crate) struct Range {
    /// The pool it returns to; none for a chunk, whose block is its own.
    pool: Option<Arc<Pool>>,
    block: Arc<Block>,
    offset: u64,
    bytes: u64,
}

impl Range {
    pub(crate) fn address(&self) -> u64 {
        self.block.address + self.offset
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn raw(&self) -> vk::Buffer {
        self.block.buffer
    }

    /// Offset of the range in its block's `VkBuffer`.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// The range's bytes through the persistent mapping, when mapped.
    pub(crate) fn mapped(&self) -> Option<*mut u8> {
        (!self.block.mapped.is_null())
            .then(|| unsafe { self.block.mapped.add(self.offset as usize) })
    }

    /// Host write of mapped storage: a plain memory write.
    pub(crate) fn write_mapped(&self, offset: u64, bytes: &[u8]) {
        let base = self
            .mapped()
            .expect("write_mapped precondition: the storage is mapped");
        assert!(
            offset + bytes.len() as u64 <= self.bytes,
            "mapped write exceeds its range"
        );
        // SAFETY: the bytes lie inside this range of the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(offset as usize), bytes.len())
        };
    }

    pub(crate) fn read_mapped(&self, offset: u64, into: &mut [u8]) {
        let base = self
            .mapped()
            .expect("read_mapped precondition: the storage is mapped");
        assert!(
            offset + into.len() as u64 <= self.bytes,
            "mapped read exceeds its range"
        );
        // SAFETY: the bytes lie inside this range of the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(base.add(offset as usize), into.as_mut_ptr(), into.len())
        };
    }
}

/// One chunk of a reservation (`crate::reservation`): a block of its own,
/// whole, freed when it drops; its holder drops it only after the device
/// work using it.
pub(crate) struct Chunk {
    device: Arc<Inner>,
    range: Range,
    /// Whether its memory is mapped at a placed host address, which it
    /// unmaps back to its reserved range before it is freed.
    placed: bool,
}

impl Chunk {
    /// The chunk over `block`, a block no pool holds ([`Memory::chunk`],
    /// [`Memory::import`]).
    pub(crate) fn new(device: Arc<Inner>, block: Arc<Block>, placed: bool) -> Self {
        let bytes = block.bytes;
        Self {
            device,
            range: Range {
                pool: None,
                block,
                offset: 0,
                bytes,
            },
            placed,
        }
    }

    pub(crate) fn range(&self) -> &Range {
        &self.range
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        if self.placed {
            self.device.unmap_placed(self.range.block.memory);
        }
        self.device.memory.free(&self.device.device, &self.range.block);
    }
}

/// One allocation of the opened device. Its storage returns to its block
/// when it drops; the runtime drops it only after the device work using it.
pub struct Buffer {
    device: Arc<Inner>,
    range: Range,
}

impl Buffer {
    pub(crate) fn new(device: Arc<Inner>, range: Range) -> Self {
        Self { device, range }
    }

    /// Device address of the allocation's first byte.
    pub fn address(&self) -> u64 {
        self.range.address()
    }

    pub fn bytes(&self) -> u64 {
        self.range.bytes()
    }

    /// The host address of the allocation's first byte through its
    /// persistent mapping, when its memory is host-visible.
    pub fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        self.range.mapped().and_then(std::ptr::NonNull::new)
    }

    pub(crate) fn range(&self) -> &Range {
        &self.range
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.memory.release(&self.device.device, &self.range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_extent_must_contain_queried_requirement() {
        assert_eq!(allocation_extent(8192, Some(4096)), Err(ExecutionError::AllocationFailed(
            "Vulkan buffer requires 8192 allocation bytes but its host import owns 4096 bytes".into())));
        assert_eq!(allocation_extent(4096, Some(8192)), Ok(8192));
        assert_eq!(allocation_extent(8192, None), Ok(8192));
    }

    #[test]
    fn owned_heap_observation_counts_padding_without_exposing_it() {
        let mut block = block(4096);
        Arc::get_mut(&mut block).unwrap().allocated_bytes = 8192;
        assert_eq!(block.heap_bytes(&[0]), 8192);
        assert_eq!(block.heap_bytes(&[1]), 0);
        assert_eq!(block.take(4096, 1), Some(0));
        assert_eq!(block.take(1, 1), None, "allocation padding is not usable buffer space");
    }

    fn block(bytes: u64) -> Arc<Block> {
        Arc::new(Block {
            heap: 0,
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            address: 0,
            bytes,
            allocated_bytes: bytes,
            mapped: std::ptr::null_mut(),
            dedicated: false,
            free: Mutex::new(FreeRanges::new(bytes)),
            _charge: Box::new(()),
        })
    }

    fn range(block: &Arc<Block>, bytes: u64) -> Range {
        Range {
            pool: None,
            block: block.clone(),
            offset: block
                .take(bytes, MINIMUM_ALIGNMENT)
                .expect("the block has room"),
            bytes,
        }
    }

    /// What `VK_EXT_memory_budget` reports as the pool's usage: every block
    /// it holds, whether or not any allocation remains in it.
    fn usage(pool: &Pool) -> u64 {
        pool.blocks
            .lock()
            .unwrap()
            .iter()
            .map(|block| block.bytes)
            .sum()
    }

    /// Retained state fills blocks; releasing it (the engine's release order
    /// evicting retained prefixes and shrinking their slabs) must return
    /// every block it emptied to the device, or the budget less usage the
    /// engine admits a claim against never grows and a claim larger than the
    /// headroom is refused although the released bytes are free.
    #[test]
    fn released_storage_returns_emptied_blocks_to_the_device() {
        const SLAB: u64 = 64 << 20;
        let pool = Pool {
            kinds: vec![0],
            charged: Charged::Kind,
            passable: Vec::new(),
            last: PoolType {
                memory_type: 0,
                heap: 0,
                mapped: false,
            },
            room_bounded: true,
            blocks: Mutex::new(Vec::new()),
            internal: AtomicU64::new(0),
        };
        let blocks = (0..4).map(|_| block(16 * SLAB)).collect::<Vec<_>>();
        pool.blocks.lock().unwrap().extend(blocks.iter().cloned());
        // The model's resident storage occupies the first block; retained
        // state fills the rest, one slab beside it in the first block.
        let model = range(&blocks[0], 15 * SLAB);
        let retained = blocks
            .iter()
            .enumerate()
            .flat_map(|(index, block)| {
                let slabs = if index == 0 { 1 } else { 16 };
                (0..slabs)
                    .map(|_| range(block, SLAB))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(usage(&pool), 64 * SLAB);

        let mut freed = Vec::new();
        for slab in &retained {
            freed.extend(pool.give(slab));
        }
        // Each emptied block left the pool once, for the caller to free; the
        // block still holding the model stays.
        assert_eq!(freed.len(), 3);
        assert!(freed
            .iter()
            .all(|block| blocks[1..].iter().any(|other| Arc::ptr_eq(block, other))));
        assert_eq!(usage(&pool), 16 * SLAB);
        assert!(Arc::ptr_eq(&pool.blocks.lock().unwrap()[0], &blocks[0]));

        // The block keeps its free space for later allocations until its
        // last allocation is released too.
        let reused = range(&blocks[0], SLAB);
        assert!(pool.give(&reused).is_none());
        assert!(pool.give(&model).is_some());
        assert_eq!(usage(&pool), 0);
    }
}
