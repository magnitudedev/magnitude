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
//! Every block is charged whole to its store's own ledger entry before its
//! memory is allocated, and each allocation's range is carved from it for
//! the entry the allocation names, so a block's free space is the device's
//! own bytes ([`seismic_compiler::blocks::charge_block`]); a pool block of
//! a passable type is no larger than its heap's budget leaves.
//! Host-visible blocks are persistently mapped.
//! Every block holds at least the heaps' bytes over
//! `maxMemoryAllocationCount`, so the device's memory objects stay within
//! the count by construction, whatever allocates them.

use crate::device::{call, Inner};
use crate::instance::Instance;
use ash::vk;
use seismic_compiler::blocks::{
    charge_block, first_fit, largest_shared, takes_own_block, FreeRanges, PackedBlock,
    BLOCK_CAP_BYTES,
};
use seismic_compiler::errors::ExecutionError;
use seismic_memory::entry::{Carved, Change, Charge, Entry, Store, Wiring};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
/// Every allocation starts at least this aligned.
const MINIMUM_ALIGNMENT: u64 = 16;

/// The stores of the embedder's ledger a device's memory lies in, which it
/// is opened with: each memory type's (kind's), and host RAM's, which
/// staging is charged to whatever heap it is on.
pub struct MemoryStores {
    pub kinds: Vec<(u32, Arc<dyn Store>)>,
    pub host: Arc<dyn Store>,
}

impl MemoryStores {
    /// The store of `kind`. The opener names a store for every kind its
    /// placements allocate in, the only kinds a pool of the device
    /// allocates in.
    fn kind(&self, kind: u32) -> &Arc<dyn Store> {
        &self
            .kinds
            .iter()
            .find(|(each, _)| *each == kind)
            .unwrap_or_else(|| panic!("the device was opened with no store for kind {kind}, which it allocates in"))
            .1
    }

    /// The store a new block of `pool` on `candidate` lies in.
    fn pool(&self, pool: &Pool, candidate: &PoolType) -> &Arc<dyn Store> {
        match pool.charged {
            Charged::Staging => &self.host,
            Charged::Kind => self.kind(candidate.memory_type),
        }
    }

    /// The own entry of the store of the device's own storage: what the
    /// driver's memory for the device ([`DriverMemory`]) is charged to, on
    /// whose heaps it lies, and what the backend's own probes allocate
    /// with.
    pub(crate) fn own(&self, memory: &Memory) -> Arc<dyn Entry> {
        self.kind(memory.device_kinds[0]).own()
    }
}

pub(crate) struct Block {
    /// The memory type its memory is of: the kind it lies in.
    memory_type: u32,
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
    /// The block's memory's exact bytes, charged before it was allocated:
    /// to its store's own entry for a pool's block, whose ranges are carved
    /// from it; to its allocation's entry for a block of its own
    /// ([`Memory::block`]). Freed as the block is ([`Memory::free`]), in the
    /// change that destroys it.
    charge: Mutex<Charge>,
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
    /// The entry its memory's bytes are charged to.
    fn entry(&self) -> Arc<dyn Entry> {
        self.charge.lock().expect("block charge lock poisoned").entry().clone()
    }

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
    /// Which store its blocks lie in.
    charged: Charged,
    /// The memory types new blocks are allocated from, in order: each of
    /// these is passed over when its heap cannot hold the block, and `last`
    /// is left to the driver.
    passable: Vec<PoolType>,
    last: PoolType,
    blocks: Mutex<Vec<Arc<Block>>>,
}

/// Which store a pool's blocks lie in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Charged {
    /// Each block's kind's.
    Kind,
    /// Host RAM's: staging is host RAM's whatever heap it is on.
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
    /// The memory types it allocates in, in order.
    #[cfg(test)]
    pub(crate) fn kinds(&self) -> &[u32] {
        &self.kinds
    }

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
}

/// The block pools of one [`crate::Device`] open, isolated to its own
/// [`MemoryStores`] so opens sharing a logical device never share blocks or
/// charge one open's blocks to another's store.
pub(crate) struct DevicePools {
    /// The pools asked of this open so far, one per ordered list of kinds.
    pools: Mutex<Vec<Arc<Pool>>>,
    /// Host-visible staging for transfers of unmapped device storage.
    staging: Arc<Pool>,
    /// The backend's own device and upload storage, as the placement rule
    /// places them.
    device: Arc<Pool>,
    upload: Arc<Pool>,
}

impl DevicePools {
    pub(crate) fn new(memory: &Memory) -> Result<Self, String> {
        let device = Arc::new(new_pool(&memory.layout, &memory.device_kinds, Charged::Kind)?);
        let upload = Arc::new(new_pool(&memory.layout, &memory.upload_kinds, Charged::Kind)?);
        let staging = Arc::new(new_pool(&memory.layout, &[memory.staging_kind], Charged::Staging)?);
        Ok(Self {
            pools: Mutex::new(vec![device.clone(), upload.clone()]),
            staging,
            device,
            upload,
        })
    }

    /// The pool of the ordered list of memory type `kinds` for this open.
    pub(crate) fn pool(&self, memory: &Memory, kinds: &[u32]) -> Result<Arc<Pool>, ExecutionError> {
        let mut pools = self.pools.lock().expect("pools lock is never poisoned");
        if let Some(pool) = pools.iter().find(|pool| pool.kinds == kinds) {
            return Ok(pool.clone());
        }
        let pool = Arc::new(new_pool(&memory.layout, kinds, Charged::Kind).map_err(ExecutionError::AllocationFailed)?);
        pools.push(pool.clone());
        Ok(pool)
    }

    pub(crate) fn upload(&self) -> &Arc<Pool> {
        &self.upload
    }

    pub(crate) fn device(&self) -> &Arc<Pool> {
        &self.device
    }

    pub(crate) fn staging(&self) -> &Arc<Pool> {
        &self.staging
    }
}

pub(crate) struct Memory {
    /// The kinds the placement rule gives the backend's own device, upload
    /// and staging storage, from which each open forms its [`DevicePools`].
    device_kinds: Vec<u32>,
    upload_kinds: Vec<u32>,
    staging_kind: u32,
    /// The memory types pools are made of.
    layout: crate::MemoryLayout,
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
    /// pooled and blocks of their own alike, counted within each block's transition.
    device_blocks: AtomicU64,
    /// The device memory objects that exist: every block, on every heap,
    /// pooled and of its own alike, counted as its memory is allocated and
    /// uncounted as it is freed. Every `vkAllocateMemory` and
    /// `vkFreeMemory` of the device is a block's ([`create_block`],
    /// [`destroy_block`]), so this is exactly the memory objects Seismic
    /// holds against `maxMemoryAllocationCount`.
    objects: AtomicU64,
    /// `maxMemoryAllocationCount`: "the maximum number of device memory
    /// allocations, as created by `vkAllocateMemory`, which can
    /// simultaneously exist" (`VkPhysicalDeviceLimits`). Only the
    /// application's allocations count: the driver's internal memory is
    /// outside it by definition.
    object_limit: u64,
    /// The least bytes of every memory object: the device's heaps together
    /// over `object_limit`, rounded up. Every block holds at least this
    /// ([`create_block`]), so while the blocks hold no more than the heaps
    /// they are never more than the limit, whoever allocates them and in whatever sizes. Beyond
    /// the heaps, the driver's refusal reads the count
    /// (`RefusalReading::MemoryObjects`).
    least_object: u64,
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
/// It is one charge on the own entry of the device's store
/// ([`MemoryStores::own`]), settled to what the driver holds at each
/// reconciliation ([`Memory::reconcile`]), which the embedder's ledger runs
/// before each of its readings (`Device::reconcile_driver_memory`): the
/// reading counts the driver's memory as it is then, its growth and what
/// it returned alike.
///
/// A reconciliation reads the heaps' usage less every block on them less
/// the process's usage outside the device, with no block in transition (a
/// block counts within one transition from before its memory joins the
/// heap, and frees within one), so it is what the driver holds at that
/// reading ([`settle`]). Not covered: what the driver takes or returns
/// between the reconciliation and the ledger's reading (a queue submission
/// on another thread within that step).
///
/// A driver may show our own changes late (`VK_EXT_memory_budget` usage is
/// the driver's count, which may lag): a free shown late keeps the freed
/// bytes in the usage, so they stay charged as the driver's until it shows
/// them; an allocation shown late leaves the usage below the blocks we
/// hold, and the reading is deferred until it shows them: the charge stays
/// as last settled, and the grant is read against the usage our own count
/// says it will show. Either way our own change never moves what the
/// ledger reads as others'.
struct DriverMemory {
    /// The process's usage of the heaps when the device was created, before
    /// any block of it (another device's, another API's): what of the usage
    /// beyond the blocks is outside the device. Fixed; each reading takes
    /// the least of it and what lies beyond the blocks then, so that memory
    /// leaving and coming back never reads as the driver's ([`settle`]).
    baseline: u64,
    /// The usage beyond the blocks at the last reading that showed every
    /// block: what a deferred reading takes as it ([`settle`]).
    beyond: AtomicU64,
}

/// One reading of the device heaps, settled ([`settle`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Settled {
    /// The usage beyond the blocks the reading stands for.
    beyond: u64,
    /// What of it is the driver's: beyond the process's usage outside the
    /// device.
    pub(crate) driver: u64,
    /// Bytes of our blocks the usage does not show yet (a driver showing an
    /// allocation late): the reading is deferred by them.
    unshown: u64,
}

/// Settle a reading of the device heaps' `usage` with `held` bytes of
/// blocks on them, against the process's usage outside the device at its
/// creation (`baseline`) and the usage beyond the blocks the last reading
/// that showed every block found (`last_beyond`).
///
/// - A usage that shows every block (`usage >= held`) stands: what lies
///   beyond the blocks is the outside usage, up to the baseline, and the
///   driver's past it. Outside usage that fell below the baseline is read
///   again at the next reading, so it can rise back to it without reading
///   as the driver's; past the baseline it reads as the driver's: charged
///   as ours, never as others'.
/// - A usage below the blocks is a driver that has not shown an allocation
///   of ours yet: the reading is deferred, standing for the last one that
///   showed every block, with the bytes it does not show yet.
fn settle(usage: u64, held: u64, baseline: u64, last_beyond: u64) -> Settled {
    match usage.checked_sub(held) {
        Some(beyond) => Settled { beyond, driver: beyond - baseline.min(beyond), unshown: 0 },
        None => Settled {
            beyond: last_beyond,
            driver: last_beyond - baseline.min(last_beyond),
            unshown: held + last_beyond - usage,
        },
    }
}

thread_local! {
    /// The block transitions this thread has open: none may be while it
    /// reconciles, which would wait on its own.
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

/// The bytes of `layout`'s heaps together.
fn heap_bytes(layout: &crate::MemoryLayout) -> Result<u64, String> {
    layout.heaps.iter().try_fold(0u64, |sum, heap| sum.checked_add(heap.bytes)).ok_or_else(|| "the device's heaps' bytes overflow".to_owned())
}

/// The least bytes of each memory object of the device `facts` describes:
/// its heaps over `maxMemoryAllocationCount`, so its memory objects stay
/// within the count by construction. What discovery reports
/// (`seismic_topology::VulkanReport::least_object`) and the device's blocks
/// are at least.
pub(crate) fn least_object(facts: &crate::Facts) -> Result<u64, String> {
    let object_limit = facts.limits.max_memory_allocations;
    if object_limit == 0 {
        return Err("the device allows no memory object".to_owned());
    }
    Ok(heap_bytes(&facts.memory)?.div_ceil(object_limit))
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
        let staging_kind = facts
            .allocation
            .staging
            .ok_or("no host staging memory type")?;
        let kinds = crate::topology::kinds(facts);
        let placed = |contract| {
            let kinds = seismic_topology::place(&kinds, facts.device_type.memory_is_host(), contract);
            let _ = new_pool(layout, &kinds, Charged::Kind)?;
            Ok::<Vec<u32>, String>(kinds)
        };
        let (device_kinds, upload_kinds) = (
            placed(seismic_topology::Contract::DeviceOnly)?,
            placed(seismic_topology::Contract::Upload)?,
        );
        let _ = new_pool(layout, &[staging_kind], Charged::Staging)?;
        let (heaps, object_limit) = (heap_bytes(layout)?, facts.limits.max_memory_allocations);
        let least_object = least_object(facts)?;
        let block_cap = facts.limits.max_allocation_bytes.min(BLOCK_CAP_BYTES);
        if least_object > block_cap {
            return Err(format!(
                "the device's {heaps} heap bytes over its {object_limit} memory objects need objects of {least_object} bytes, beyond its {block_cap}-byte blocks"
            ));
        }
        Ok(Self {
            device_kinds,
            upload_kinds,
            staging_kind,
            layout: layout.clone(),
            families: std::iter::once(facts.queue_family).chain(facts.side_family.filter(|&family| family != facts.queue_family)).collect(),
            device_heaps: facts.allocation.device_heaps.clone(),
            block_cap,
            max_allocation: facts.limits.max_allocation_bytes,
            instance,
            physical,
            device_blocks: AtomicU64::new(0),
            objects: AtomicU64::new(0),
            object_limit,
            least_object,
            transitions_begun: AtomicU64::new(0),
            transitions_ended: AtomicU64::new(0),
            quiet: Mutex::new(()),
            ended: std::sync::Condvar::new(),
            driver: DriverMemory {
                baseline,
                beyond: AtomicU64::new(baseline),
            },
        })
    }

    /// Begin a block's transition: until it drops, no reconciliation reads.
    fn transition(&self) -> Transition<'_> {
        OPEN.with(|open| open.set(open.get() + 1));
        self.transitions_begun.fetch_add(1, Ordering::SeqCst);
        Transition(self, std::marker::PhantomData)
    }

    /// Count `block`, its memory allocated: one memory object, and its
    /// bytes on the device heaps within its creation's transition.
    fn count(&self, block: &Block) {
        self.objects.fetch_add(1, Ordering::SeqCst);
        self.device_blocks.fetch_add(block.heap_bytes(&self.device_heaps), Ordering::SeqCst);
    }

    /// Free `block` (no allocation is in it) and uncount it, within one
    /// transition.
    pub(crate) fn free(&self, device: &ash::Device, block: &Block) {
        // The memory's destruction and the free of its bytes are one change.
        let change = block.entry().change();
        let _transition = self.transition();
        self.device_blocks.fetch_sub(block.heap_bytes(&self.device_heaps), Ordering::SeqCst);
        destroy_block(device, block);
        self.objects.fetch_sub(1, Ordering::SeqCst);
        let mut charge = block.charge.lock().expect("block charge lock poisoned");
        let bytes = charge.bytes();
        if bytes > 0 {
            charge.shrink(&change, bytes);
        }
    }

    /// The device memory objects Seismic holds now: every block.
    pub(crate) fn objects(&self) -> u64 {
        self.objects.load(Ordering::SeqCst)
    }

    /// Reconcile the driver's memory ([`DriverMemory`]): read the blocks and
    /// the heaps' usage with no block in transition ([`Self::unheld`]), and
    /// settle its charge on `driver` to the driver's part of the usage
    /// ([`settle`]), never refused (the driver already took it). The new
    /// charge is taken before the old one is freed, so the entry never
    /// shows less than the driver holds. The grant is the heaps' budget less
    /// their usage, less what of our blocks the usage does not show yet.
    /// Takes no lock a block's creation or freeing holds, so it runs from
    /// anywhere but inside a transition.
    pub(crate) fn reconcile(&self, driver: &Arc<dyn Entry>, driver_charge: &Mutex<Option<Charge>>) -> u64 {
        assert_eq!(OPEN.with(std::cell::Cell::get), 0, "a reconciliation inside a block transition");
        let (budgets, settled) = self.unheld();
        let mut charge = driver_charge.lock().expect("driver charge lock poisoned");
        if charge.as_ref().map_or(0, Charge::bytes) != settled.driver {
            // Memory the driver took that no call of ours made: a change of
            // its own, with no effect of ours beside it, ended before the
            // sample is stamped.
            let change = driver.change();
            *charge = (settled.driver > 0).then(|| Charge::took(&change, driver, settled.driver, Wiring::Pageable));
        }
        self.device_heaps
            .iter()
            .map(|&heap| budgets[heap as usize])
            .map(|heap| heap.budget.saturating_sub(heap.usage))
            .sum::<u64>()
            .saturating_sub(settled.unshown)
    }

    /// The heaps' budgets, and the reading of the device heaps' usage
    /// settled against the blocks ([`settle`]), read with no block in
    /// transition. A transition in flight, or one begun across the reading,
    /// parks it until transitions end and it reads again.
    pub(crate) fn unheld(&self) -> (Vec<crate::HeapBudget>, Settled) {
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
                let settled = settle(usage, held, self.driver.baseline, self.driver.beyond.load(Ordering::SeqCst));
                if settled.unshown == 0 {
                    self.driver.beyond.store(settled.beyond, Ordering::SeqCst);
                }
                return (budgets, settled);
            }
        }
    }

    /// `VK_EXT_memory_budget` of the heaps device storage is allocated from.
    pub(crate) fn memory_budget(&self) -> crate::MemoryBudget {
        crate::device::memory_budget(&self.instance, self.physical, &self.device_heaps)
    }

    /// A range of `bytes` of `pool`, on a block of its own where it takes
    /// one (`takes_own_block`); a new block is charged to its store's own
    /// entry ([`Self::create`]). The caller carves an allocation's bytes
    /// from that entry for its own ([`Range::carve`]); the device's own
    /// ranges (argument chunks, staging) stay the store's own, until the
    /// device is destroyed.
    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        stores: &MemoryStores,
    ) -> Result<Range, ExecutionError> {
        if bytes > self.max_allocation {
            return Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: self.max_allocation,
            });
        }
        let bytes = bytes.max(1);
        let alignment = alignment.max(MINIMUM_ALIGNMENT);
        let mut blocks = pool.blocks.lock().expect("pool lock is never poisoned");
        let found = first_fit(&blocks, bytes, alignment)
            .map(|(index, offset)| (blocks[index].clone(), offset));
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                let block = self.create(device, pool, largest_shared(&blocks), bytes, stores)?;
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

    /// A new block of `pool` for an allocation of `bytes`, its memory's
    /// bytes charged to its store's own entry
    /// ([`seismic_compiler::blocks::charge_block`]), sized by its heaps'
    /// budgets as the driver reports them now. A passable type is passed over when
    /// its heap's budget cannot hold the allocation or its heap refuses the
    /// block, and its block holds only the allocation where the heap's
    /// budget cannot hold the larger one; the last type, with none after
    /// it, is left to the driver.
    fn create(
        &self,
        device: &ash::Device,
        pool: &Pool,
        largest: Option<u64>,
        bytes: u64,
        stores: &MemoryStores,
    ) -> Result<Arc<Block>, ExecutionError> {
        // Each charge, the memory its block takes and the record of it are
        // one change.
        let change = stores.pool(pool, &pool.last).own().change();
        let budgets = crate::device::heap_budgets(&self.instance, self.physical);
        let dedicated = takes_own_block(bytes, self.block_cap);
        let charge = |candidate: &PoolType| charge_block(&change, &stores.pool(pool, candidate).own(), largest, self.block_cap, bytes, 1);
        // The least a new block of the allocation holds.
        let least = bytes.max(self.least_object);
        for candidate in &pool.passable {
            let heap = budgets[candidate.heap as usize];
            let room = heap.budget.saturating_sub(heap.usage);
            if least > room {
                continue;
            }
            let mut charged = match charge(candidate) {
                Ok(charged) => charged,
                Err(error) if error.capacity_refusal().is_some() => continue,
                Err(error) => return Err(error),
            };
            if charged.bytes() > room {
                charged.shrink(&change, charged.bytes().saturating_sub(least));
            }
            match create_block(self, device, &change, *candidate, dedicated, charged) {
                Ok(block) => return Ok(block),
                Err(error) if error.capacity_refusal().is_some() => {}
                Err(error) => return Err(error),
            }
        }
        create_block(self, device, &change, pool.last, dedicated, charge(&pool.last)?)
    }

    /// A block of its own of `bytes` of memory type `kind`, charged exactly
    /// to `entry` before its memory is allocated, and freed when its range
    /// drops (`Device::allocate_block`, which zeroes it).
    pub(crate) fn block(
        &self,
        device: &ash::Device,
        kind: u32,
        bytes: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Range, ExecutionError> {
        // Whole words, which a fill zeroes (`vkCmdFillBuffer`).
        let bytes = bytes.max(1).next_multiple_of(4);
        if bytes > self.max_allocation {
            return Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: self.max_allocation,
            });
        }
        if kind as usize >= self.layout.types.len() {
            return Err(ExecutionError::AllocationFailed(format!("Vulkan allocates no block of kind {kind}")));
        }
        // The charge, the memory and the record of it are one change.
        let change = entry.change();
        let charge = Charge::take(&change, entry, bytes, Wiring::Pageable)?;
        let block = create_block(self, device, &change, pool_type(&self.layout, kind), true, charge)?;
        Ok(Range { pool: None, block, offset: 0, bytes })
    }

    /// Return `range` to its block, and free the block once it holds no
    /// allocation: at once for a block of its own.
    pub(crate) fn release(&self, device: &ash::Device, range: &Range) {
        match &range.pool {
            Some(pool) => {
                if let Some(block) = pool.give(range) {
                    self.free(device, &block);
                }
            }
            None => self.free(device, &range.block),
        }
    }

    /// Assert that every block was released before the logical device is destroyed.
    pub(crate) fn destroy(&self, _device: &ash::Device) {
        debug_assert_eq!(self.objects(), 0, "every Vulkan block is freed before the logical device");
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
        blocks: Mutex::new(Vec::new()),
    })
}

fn destroy_block(device: &ash::Device, block: &Block) {
    unsafe {
        device.destroy_buffer(block.buffer, None);
        device.free_memory(block.memory, None);
    }
}

/// A block of `charge`'s bytes on `pool`'s type, mapped persistently where
/// the type is mappable. The memory the buffer requires beyond them (its
/// allocation's rounding) is charged to the same entry before the memory is
/// allocated, so the charge is the memory's exact bytes.
fn create_block(
    memory_owner: &Memory,
    device: &ash::Device,
    change: &Change,
    pool: PoolType,
    dedicated: bool,
    mut charge: Charge,
) -> Result<Arc<Block>, ExecutionError> {
    let allocation = |error: ExecutionError| match error {
        ExecutionError::SubmissionFailed(message) => ExecutionError::AllocationFailed(message),
        other => other,
    };
    // Every memory object holds at least the device's least
    // (`Memory::least_object`), its free space a pool block's to use and
    // a block of its own's to strand.
    if let Some(short) = memory_owner.least_object.checked_sub(charge.bytes()).filter(|&short| short > 0) {
        charge.grow(change, short).map_err(ExecutionError::from)?;
    }
    let bytes = charge.bytes();
    let families = memory_owner.families.as_slice();
    let info = vk::BufferCreateInfo::default()
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
    let buffer = call(unsafe { device.create_buffer(&info, None) }, "vkCreateBuffer").map_err(allocation)?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let refuse = |error: ExecutionError| -> Result<Arc<Block>, ExecutionError> {
        unsafe { device.destroy_buffer(buffer, None) };
        Err(error)
    };
    if requirements.memory_type_bits & (1 << pool.memory_type) == 0 {
        return refuse(ExecutionError::AllocationFailed(format!(
            "memory type {} cannot back a storage buffer",
            pool.memory_type
        )));
    }
    let allocated_bytes = requirements.size.max(bytes);
    if let Err(refused) = charge.grow(change, allocated_bytes - bytes) {
        return refuse(refused.into());
    }
    // The charge taken, the block's memory joins the heap within one
    // transition, until it is counted.
    let _transition = memory_owner.transition();
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(allocated_bytes)
        .memory_type_index(pool.memory_type)
        .push_next(&mut flags);
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
                // The count reached: only where the blocks hold more than
                // the heaps (`Memory::least_object`).
                vk::Result::ERROR_TOO_MANY_OBJECTS => ExecutionError::DeviceRefused {
                    required: bytes,
                    reading: seismic_compiler::errors::RefusalReading::MemoryObjects {
                        limit: memory_owner.object_limit,
                        held: memory_owner.objects(),
                    },
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
    let mapped = bound.and_then(|()| match pool.mapped {
        true => call(
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) },
            "vkMapMemory",
        )
        .map(|pointer| pointer.cast::<u8>()),
        false => Ok(std::ptr::null_mut()),
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
        memory_type: pool.memory_type,
        heap: pool.heap,
        memory,
        buffer,
        address,
        bytes,
        allocated_bytes,
        mapped,
        dedicated,
        free: Mutex::new(FreeRanges::new(bytes)),
        charge: Mutex::new(charge),
    });
    memory_owner.count(&block);
    Ok(block)
}

/// A range of a block. [`Buffer`] releases it on drop; the device's own
/// ranges (argument chunks, staging) live until the device is destroyed.
pub(crate) struct Range {
    /// The pool it returns to; none for a block of its own
    /// ([`Memory::block`]).
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

    /// The range's bytes carved for `entry` from its pool block's, which
    /// its store's own entry holds.
    pub(crate) fn carve(&self, entry: &Arc<dyn Entry>) -> Carved {
        Charge::carve(&self.block.charge.lock().expect("block charge lock poisoned"), entry, self.bytes)
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

/// One allocation of the opened device. Its storage returns to its block
/// when it drops, and its bytes leave its entry; the runtime drops it only
/// after the device work using it.
pub struct Buffer {
    device: Arc<Inner>,
    /// A pool range's bytes, carved from its block for the allocation's
    /// entry; none for a block of its own, whose charge is the entry's
    /// already. Returned to the block's entry first as the buffer drops,
    /// before the release that frees a block it empties with its charge.
    carved: Option<Carved>,
    range: Range,
}

impl Buffer {
    pub(crate) fn new(device: Arc<Inner>, range: Range, carved: Option<Carved>) -> Self {
        Self { device, carved, range }
    }

    /// Device address of the allocation's first byte.
    pub fn address(&self) -> u64 {
        self.range.address()
    }

    pub fn bytes(&self) -> u64 {
        self.range.bytes()
    }

    /// The kind the allocation lies in: its block's memory type, the one of
    /// the kinds it was asked for that it landed on.
    pub fn kind(&self) -> u32 {
        self.range.block.memory_type
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
    /// The carved bytes return to the block's charge, then the range to its
    /// pool: a drop body runs before the fields drop, so a block the release
    /// frees would otherwise free its whole charge while the range's bytes
    /// still lay outside it, and the carve's return would free them twice.
    fn drop(&mut self) {
        drop(self.carved.take());
        self.device.memory.release(&self.device.device, &self.range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A usage showing every block stands: beyond the blocks, the outside
    /// usage up to the baseline and the driver's past it.
    #[test]
    fn a_usage_showing_every_block_settles_the_driver_past_the_baseline() {
        assert_eq!(settle(1000, 600, 100, 0), Settled { beyond: 400, driver: 300, unshown: 0 });
        assert_eq!(settle(600, 600, 100, 400), Settled { beyond: 0, driver: 0, unshown: 0 });
    }

    /// A driver that shows an allocation of ours late leaves the usage
    /// below the blocks: the reading is deferred, standing for the last
    /// one that showed every block, with the bytes not shown yet, which the
    /// grant is read against. It never panics and never moves the driver's
    /// charge.
    #[test]
    fn a_usage_below_the_blocks_defers_the_reading() {
        let shown = settle(1000, 600, 100, 0);
        // A 500-byte block allocated, the usage not showing it yet.
        let lagging = settle(1000, 1100, 100, shown.beyond);
        assert_eq!(lagging, Settled { beyond: 400, driver: 300, unshown: 500 });
        // Shown at last: the same driver figure, nothing deferred.
        assert_eq!(settle(1500, 1100, 100, lagging.beyond), Settled { beyond: 400, driver: 300, unshown: 0 });
    }

    /// The outside usage is read again at each reading: falling below the
    /// baseline and rising back to it, it never reads as the driver's.
    #[test]
    fn outside_usage_falling_and_rising_back_is_never_the_drivers() {
        let baseline = 300;
        assert_eq!(settle(600 + 300, 600, baseline, 300).driver, 0);
        assert_eq!(settle(600 + 50, 600, baseline, 300).driver, 0, "the outside usage fell");
        assert_eq!(settle(600 + 300, 600, baseline, 50).driver, 0, "and rose back: still not the driver's");
        assert_eq!(settle(600 + 380, 600, baseline, 300).driver, 80, "past the baseline, the driver's");
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
            memory_type: 0,
            heap: 0,
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            address: 0,
            bytes,
            allocated_bytes: bytes,
            mapped: std::ptr::null_mut(),
            dedicated: false,
            free: Mutex::new(FreeRanges::new(bytes)),
            charge: Mutex::new({
                let entry = tally();
                Charge::took(&entry.change(), &entry, bytes, Wiring::Pageable)
            }),
        })
    }

    fn tally() -> Arc<dyn Entry> {
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam))
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
            blocks: Mutex::new(Vec::new()),
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
