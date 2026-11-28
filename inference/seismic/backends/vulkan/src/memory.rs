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

use crate::device::Inner;
use crate::instance::Instance;
use ash::vk;
use seismic_compiler::blocks::{
    charge_block, charge_block_for, first_fit, largest_shared, takes_own_block, FreeRanges,
    PackedBlock, BLOCK_CAP_BYTES,
};
use seismic_compiler::errors::ExecutionError;
use seismic_memory::entry::{CarvableCharge, Carved, Change, Charge, Entry, Store, Wiring};
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
            Charged::Kind => self.kind(candidate.kind),
        }
    }

    /// The own entry of the store of the device's own storage: what the
    /// driver's memory for the device ([`DriverMemory`]) is charged to, on
    /// whose heaps it lies, and what the backend's own probes allocate
    /// with.
    pub(crate) fn own(&self, memory: &Memory) -> Arc<dyn Entry> {
        self.kind(memory.device_kind()).own()
    }
}

pub(crate) struct Block {
    /// The topology kind this block serves (the store/path identity).
    kind: u32,
    /// The Vulkan memory type index backing this block (may differ from
    /// `kind` when `kind` cannot back a buffer and resolved to a compatible
    /// memory type on the same heap).
    memory_type: u32,
    heap: u32,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    /// Device address for shader-addressable blocks; `None` for staging
    /// blocks, which are only copied by `vkCmdCopyBuffer` and do not request
    /// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS`.
    address: Option<u64>,
    /// Usable buffer extent, distinct from the allocation's required padding.
    bytes: u64,
    allocated_bytes: u64,
    /// Base of the persistent mapping; null for unmapped memory.
    mapped: *mut u8,
    dedicated: bool,
    free: Mutex<FreeRanges>,
    held: Held,
}

/// What a block's memory is held as.
enum Held {
    /// Its exact bytes, charged before it was allocated: to its store's own
    /// entry for a pool's block, whose ranges are carved from it; to its
    /// allocation's entry for a block of its own ([`Memory::block`]). Freed
    /// as the block is ([`Memory::free`]), in the change that destroys it.
    Charged(CarvableCharge),
    /// The logical device's own ([`Memory::driver_block`]): uncounted among
    /// the blocks, so each reconciliation charges it as the driver's memory
    /// ([`DriverMemory`]) to every live open's own entry, like the logical
    /// device itself.
    Driver,
}

/// What a new block's memory is to be held as ([`create_block`]): charged
/// exactly in `Change`, admitted before its memory is allocated, or the
/// driver's, of these bytes, admitted by none.
enum Holding<'c> {
    Charged(&'c Change, Charge),
    /// A pool block whose first allocation's `bytes` are charged directly to
    /// the caller's entry (`first`, drawing on its already-committed claim)
    /// and whose uncarved remainder is charged to the store's `own` entry.
    ChargedWithFirst(&'c Change, Charge, Charge),
    Driver(u64),
}

impl Holding<'_> {
    fn bytes(&self) -> u64 {
        match self {
            Self::Charged(_, charge) => charge.bytes(),
            Self::ChargedWithFirst(_, remainder, first) => {
                remainder
                    .bytes()
                    .checked_add(first.bytes())
                    .expect("carvable block bytes overflow")
            }
            Self::Driver(bytes) => *bytes,
        }
    }

    /// Take `bytes` more.
    fn grow(&mut self, bytes: u64) -> Result<(), ExecutionError> {
        if bytes == 0 {
            return Ok(());
        }
        match self {
            Self::Charged(change, charge) => charge.grow(*change, bytes).map_err(ExecutionError::from),
            Self::ChargedWithFirst(change, remainder, _) => {
                remainder.grow(*change, bytes).map_err(ExecutionError::from)
            }
            Self::Driver(held) => {
                *held += bytes;
                Ok(())
            }
        }
    }

    /// Charge mandatory driver alignment padding (`VkMemoryRequirements::size`
    /// beyond usable buffer bytes). When the first allocation is already
    /// admitted (`ChargedWithFirst`), fall back to recording the padding on
    /// the block's `own` entry rather than refusing the admitted allocation
    /// (matching CUDA's `charge_block_for` granularity rounding).
    fn grow_padding(&mut self, bytes: u64) -> Result<(), ExecutionError> {
        if bytes == 0 {
            return Ok(());
        }
        match self {
            Self::Charged(change, charge) => charge.grow(*change, bytes).map_err(ExecutionError::from),
            Self::ChargedWithFirst(change, remainder, _) => {
                if remainder.grow(*change, bytes).is_err() {
                    remainder.took_more(*change, bytes);
                }
                Ok(())
            }
            Self::Driver(held) => {
                *held += bytes;
                Ok(())
            }
        }
    }

    fn held(self) -> (Held, Option<Carved>) {
        match self {
            Self::Charged(_, charge) => (Held::Charged(charge.into_carvable()), None),
            Self::ChargedWithFirst(_, remainder, first) => {
                let (carvable, carved) = remainder.into_carvable_with_first(first);
                (Held::Charged(carvable), Some(carved))
            }
            Self::Driver(_) => (Held::Driver, None),
        }
    }
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
    /// What the block adds to the blocks counted on `heaps`: what it takes
    /// of them ([`Self::heap_bytes`]) where it is charged; none where the
    /// driver's memory holds it ([`Held::Driver`]).
    fn counted_bytes(&self, heaps: &[u32]) -> u64 {
        match self.held {
            Held::Charged(_) => self.heap_bytes(heaps),
            Held::Driver => 0,
        }
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

    /// Carve `bytes` for `entry` from this pool block's charge, which its
    /// store's own entry holds.
    fn carve(&self, entry: &Arc<dyn Entry>, bytes: u64) -> Carved {
        let Held::Charged(charge) = &self.held else {
            panic!("a range carved from a block the driver holds: only a pool's ranges are carved");
        };
        charge.carve(entry, bytes)
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

impl Charged {
    /// Whether blocks of this pool need a shader device address. Staging
    /// blocks are only copied via `vkCmdCopyBuffer` and never addressed by
    /// shaders, so omitting BDA stops drivers such as RADV from pinning all
    /// staging BOs into their global BO list.
    fn shader_addressable(self) -> bool {
        match self {
            Self::Kind => true,
            Self::Staging => false,
        }
    }
}

/// One memory type a pool allocates blocks from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolType {
    /// The topology kind this entry serves (used for store/path accounting).
    kind: u32,
    /// The Vulkan memory type index used for `vkAllocateMemory`.
    memory_type: u32,
    heap: u32,
    /// Host-visible and coherent: its blocks are persistently mapped.
    mapped: bool,
}

impl Pool {
    /// The memory types it allocates in, in order.
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
        let device = Arc::new(new_pool(&memory.layout, &memory.device_kinds, Charged::Kind, memory.storage_memory_type_bits)?);
        let upload = Arc::new(new_pool(&memory.layout, &memory.upload_kinds, Charged::Kind, memory.storage_memory_type_bits)?);
        let staging = Arc::new(new_pool(&memory.layout, &[memory.staging_kind], Charged::Staging, memory.staging_memory_type_bits)?);
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
        let pool = Arc::new(
            new_pool(&memory.layout, kinds, Charged::Kind, memory.storage_memory_type_bits)
                .map_err(ExecutionError::AllocationFailed)?,
        );
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
    /// `VkMemoryRequirements::memoryTypeBits` for shader-addressable storage
    /// buffers and for host staging buffers, queried once at device open via
    /// `vkGetDeviceBufferMemoryRequirements` (Vulkan 1.3 `maintenance4`).
    storage_memory_type_bits: u32,
    staging_memory_type_bits: u32,
    /// The queue families that reach each buffer: the work queue's, and the
    /// second queue's where that is a family of its own.
    families: Vec<u32>,
    /// The heaps of the device's own memory, whose usage its memory budget
    /// reads.
    device_heaps: Vec<u32>,
    /// The non-device-local heap that discrete `Contract::Shared` host-tier
    /// pool blocks allocate from, when distinct from `device_heaps`.
    host_tier_heap: Option<u32>,
    block_cap: u64,
    max_allocation: u64,
    /// Where heap budgets are read.
    instance: Arc<Instance>,
    physical: vk::PhysicalDevice,
    /// Bytes of every block on the heaps device storage is allocated from,
    /// pooled and blocks of their own alike, counted within each block's transition.
    device_blocks: AtomicU64,
    /// Live `allocated_bytes` of every block on each Vulkan heap, indexed by
    /// heap number. Provides lag-free accounting alongside `VK_EXT_memory_budget`
    /// (for example when unwritten GTT blocks are not yet reflected in kernel
    /// `heapUsage`).
    heap_blocks: Vec<AtomicU64>,
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
/// scratch, staging and second-queue objects, and the blocks it holds
/// itself (`Memory::driver_block`: the tables buffer).
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

/// The least bytes of each memory object of the device `info` describes:
/// its heaps over `maxMemoryAllocationCount`, so its memory objects stay
/// within the count by construction. What discovery reports
/// (`seismic_topology::VulkanReport::least_object`) and the device's blocks
/// are at least.
pub(crate) fn least_object(info: &crate::DeviceInfo) -> Result<u64, String> {
    let object_limit = info.limits.max_memory_allocations;
    if object_limit == 0 {
        return Err("the device allows no memory object".to_owned());
    }
    Ok(heap_bytes(&info.memory)?.div_ceil(object_limit))
}

impl Memory {
    /// The first kind of the backend's own device storage: the kind of the
    /// store whose own entry the device's own memory is charged to.
    pub(crate) fn device_kind(&self) -> u32 {
        self.device_kinds[0]
    }

    /// The backend's own device and upload storage, on the kinds the
    /// placement rule gives them, and staging on `info.allocation`'s type.
    pub(crate) fn new(
        info: &crate::DeviceInfo,
        device: &ash::Device,
        instance: Arc<Instance>,
        physical: vk::PhysicalDevice,
        baseline: u64,
    ) -> Result<Self, String> {
        let layout = &info.memory;
        let families: Vec<u32> = std::iter::once(info.queue_family)
            .chain(info.side_family.filter(|&family| family != info.queue_family))
            .collect();
        let storage_memory_type_bits = buffer_memory_type_bits(device, &families, true);
        let staging_memory_type_bits = buffer_memory_type_bits(device, &families, false);
        let staging_kind = select_staging_kind(&info.allocation.host, staging_memory_type_bits)
            .ok_or("no host staging memory type can back a staging buffer")?;
        let kinds = crate::topology::kinds(info);
        let placed = |contract| {
            let kinds = seismic_topology::place(&kinds, info.device_type.memory_is_host(), contract);
            let _ = new_pool(layout, &kinds, Charged::Kind, storage_memory_type_bits)?;
            Ok::<Vec<u32>, String>(kinds)
        };
        let (device_kinds, upload_kinds) = (
            placed(seismic_topology::Contract::DeviceOnly)?,
            placed(seismic_topology::Contract::Upload)?,
        );
        let host_tier_heap = seismic_topology::place(
            &kinds,
            info.device_type.memory_is_host(),
            seismic_topology::Contract::Shared,
        )
        .first()
        .map(|&kind| layout.types[kind as usize].heap)
        .filter(|heap| {
            !info.allocation.device_heaps.contains(heap)
                && !layout.heaps[*heap as usize].device_local
        });
        let _ = new_pool(layout, &[staging_kind], Charged::Staging, staging_memory_type_bits)?;
        let (heaps, object_limit) = (heap_bytes(layout)?, info.limits.max_memory_allocations);
        let least_object = least_object(info)?;
        let block_cap = info.limits.max_allocation_bytes.min(BLOCK_CAP_BYTES);
        if least_object > block_cap {
            return Err(format!(
                "the device's {heaps} heap bytes over its {object_limit} memory objects need objects of {least_object} bytes, beyond its {block_cap}-byte blocks"
            ));
        }
        let heap_blocks = (0..layout.heaps.len()).map(|_| AtomicU64::new(0)).collect();
        Ok(Self {
            device_kinds,
            upload_kinds,
            staging_kind,
            layout: layout.clone(),
            storage_memory_type_bits,
            staging_memory_type_bits,
            families,
            device_heaps: info.allocation.device_heaps.clone(),
            host_tier_heap,
            block_cap,
            max_allocation: info.limits.max_allocation_bytes,
            instance,
            physical,
            device_blocks: AtomicU64::new(0),
            heap_blocks,
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
        self.device_blocks.fetch_add(block.counted_bytes(&self.device_heaps), Ordering::SeqCst);
        self.heap_blocks[block.heap as usize].fetch_add(block.allocated_bytes, Ordering::SeqCst);
    }

    /// Free `block` (no allocation is in it) and uncount it, within one
    /// transition.
    pub(crate) fn free(&self, device: &ash::Device, block: &Block) {
        // The memory's destruction and the free of its bytes are one change.
        let charged = match &block.held {
            Held::Charged(charge) => Some((charge.entry().change(), charge)),
            Held::Driver => None,
        };
        let _transition = self.transition();
        self.device_blocks.fetch_sub(block.counted_bytes(&self.device_heaps), Ordering::SeqCst);
        self.heap_blocks[block.heap as usize].fetch_sub(block.allocated_bytes, Ordering::SeqCst);
        destroy_block(device, block);
        self.objects.fetch_sub(1, Ordering::SeqCst);
        if let Some((change, charge)) = charged {
            charge.release(&change);
        }
    }

    /// The device memory objects Seismic holds now: every block.
    pub(crate) fn objects(&self) -> u64 {
        self.objects.load(Ordering::SeqCst)
    }

    /// The device's limit on simultaneous memory allocations.
    pub(crate) fn object_limit(&self) -> u64 {
        self.object_limit
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
    /// one (`takes_own_block`), kept on its store's own entry (argument
    /// chunks, staging) until the device is destroyed.
    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        stores: &MemoryStores,
    ) -> Result<Range, ExecutionError> {
        let (range, carved) = self.allocate_inner(device, pool, bytes, alignment, stores, None)?;
        debug_assert!(carved.is_none());
        Ok(range)
    }

    /// A range of `bytes` of `pool` charged to `entry`: carved from an
    /// existing shared block when one fits, or charged directly to `entry`
    /// (`charge_block_for`) when opening a new block so an already-committed
    /// claim is never double-charged against the store's own entry.
    pub(crate) fn allocate_for(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        stores: &MemoryStores,
        entry: &Arc<dyn Entry>,
    ) -> Result<(Range, Carved), ExecutionError> {
        let (range, carved) = self.allocate_inner(device, pool, bytes, alignment, stores, Some(entry))?;
        Ok((range, carved.expect("an entry-backed allocation always returns a carved charge")))
    }

    fn allocate_inner(
        &self,
        device: &ash::Device,
        pool: &Arc<Pool>,
        bytes: u64,
        alignment: u64,
        stores: &MemoryStores,
        entry: Option<&Arc<dyn Entry>>,
    ) -> Result<(Range, Option<Carved>), ExecutionError> {
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
        let (block, offset, carved) = match found {
            Some((block, offset)) => {
                let carved = entry.map(|entry| block.carve(entry, bytes));
                (block, offset, carved)
            }
            None => {
                let (block, carved) =
                    self.create(device, pool, largest_shared(&blocks), bytes, stores, entry)?;
                let offset = block
                    .take(bytes, alignment)
                    .expect("a new block holds its first allocation");
                blocks.push(block.clone());
                (block, offset, carved)
            }
        };
        Ok((
            Range {
                pool: Some(pool.clone()),
                block,
                offset,
                bytes,
            },
            carved,
        ))
    }

    /// A new block of `pool` for an allocation of `bytes`, sized by its heaps'
    /// budgets as the driver reports them now. When `entry` is `Some`, the
    /// first allocation's `bytes` are charged directly to `entry`
    /// ([`seismic_compiler::blocks::charge_block_for`]) and only the uncarved
    /// remainder is charged to the store's own entry; otherwise the whole block
    /// is charged to the store's own entry
    /// ([`seismic_compiler::blocks::charge_block`]). A passable type is passed
    /// over when its heap's budget cannot hold the allocation or its heap
    /// refuses the block, and a non-dedicated block holds only the allocation
    /// where its heap's budget cannot hold the grown slab (including on
    /// `pool.last`).
    fn create(
        &self,
        device: &ash::Device,
        pool: &Pool,
        largest: Option<u64>,
        bytes: u64,
        stores: &MemoryStores,
        entry: Option<&Arc<dyn Entry>>,
    ) -> Result<(Arc<Block>, Option<Carved>), ExecutionError> {
        // Each charge, the memory its block takes and the record of it are
        // one change.
        let change = stores.pool(pool, &pool.last).own().change();
        let budgets = crate::device::heap_budgets(&self.instance, self.physical);
        let dedicated = takes_own_block(bytes, self.block_cap);
        let shader_addressable = pool.charged.shader_addressable();
        // The least a new block of the allocation holds.
        let least = bytes.max(self.least_object);
        let charge = |candidate: &PoolType, room: u64| -> Result<Holding<'_>, ExecutionError> {
            let own = stores.pool(pool, candidate).own();
            match entry {
                Some(entry) => {
                    let (mut remainder, first) = charge_block_for(
                        &change,
                        &own,
                        entry,
                        largest,
                        self.block_cap,
                        bytes,
                        1,
                    )?;
                    let total = remainder.bytes() + first.bytes();
                    let trim = trim_for_room(total, least, room, dedicated);
                    if trim > 0 {
                        remainder.shrink(&change, trim);
                    }
                    Ok(Holding::ChargedWithFirst(&change, remainder, first))
                }
                None => {
                    let mut charged = charge_block(&change, &own, largest, self.block_cap, bytes, 1)?;
                    let trim = trim_for_room(charged.bytes(), least, room, dedicated);
                    if trim > 0 {
                        charged.shrink(&change, trim);
                    }
                    Ok(Holding::Charged(&change, charged))
                }
            }
        };
        for candidate in &pool.passable {
            let heap = budgets[candidate.heap as usize];
            let room = heap.budget.saturating_sub(heap.usage);
            if least > room {
                continue;
            }
            let holding = match charge(candidate, room) {
                Ok(holding) => holding,
                Err(error) if error.capacity_refusal().is_some() => continue,
                Err(error) => return Err(error),
            };
            match create_block(self, device, *candidate, dedicated, shader_addressable, holding) {
                Ok(created) => return Ok(created),
                Err(error) if error.capacity_refusal().is_some() => {}
                Err(error) => return Err(error),
            }
        }
        let heap = budgets[pool.last.heap as usize];
        let room = heap.budget.saturating_sub(heap.usage);
        let holding = charge(&pool.last, room)?;
        create_block(self, device, pool.last, dedicated, shader_addressable, holding)
    }

    /// Live headroom for additional dedicated pool blocks on a discrete
    /// device's non-device-local (GTT) heap, keeping command-submission
    /// headroom free. `None` for device-local heaps or unified devices whose
    /// heaps are all in `device_heaps`.
    fn non_device_local_block_room(&self, heap: u32) -> Option<u64> {
        let heap_index = heap as usize;
        if self.device_heaps.contains(&heap) || self.layout.heaps[heap_index].device_local {
            return None;
        }
        let budgets = crate::device::heap_budgets(&self.instance, self.physical);
        let reported = budgets[heap_index];
        let tracked = self.heap_blocks[heap_index].load(Ordering::SeqCst);
        Some(non_device_local_block_room(
            self.layout.heaps[heap_index].bytes,
            reported.budget,
            reported.usage,
            tracked,
        ))
    }

    /// Live headroom for additional host-tier pool blocks on a discrete
    /// device's non-device-local (GTT) heap, or `None` when the device has no
    /// separate non-device-local host-tier heap.
    pub(crate) fn host_tier_headroom(&self) -> Option<u64> {
        self.host_tier_heap
            .and_then(|heap| self.non_device_local_block_room(heap))
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
        let (bytes, pool) = self.own_block(kind, bytes)?;
        if let Some(available) = self.non_device_local_block_room(pool.heap) {
            let required = bytes.max(self.least_object);
            if required > available {
                return Err(ExecutionError::AllocationCapacity {
                    required: required.into(),
                    available,
                });
            }
        }
        // The charge, the memory and the record of it are one change.
        let change = entry.change();
        let charge = Charge::take(&change, entry, bytes, Wiring::Pageable)?;
        let (block, carved) =
            create_block(self, device, pool, true, true, Holding::Charged(&change, charge))?;
        debug_assert!(carved.is_none());
        Ok(Range { pool: None, block, offset: 0, bytes })
    }

    /// A block of its own of `bytes` of memory type `kind` that the logical
    /// device itself holds, as it holds its pipelines: charged to no entry
    /// and admitted by no ledger, its memory counted as the driver's
    /// ([`Held::Driver`]), freed when its range is released.
    pub(crate) fn driver_block(&self, device: &ash::Device, kind: u32, bytes: u64) -> Result<Range, ExecutionError> {
        let (bytes, pool) = self.own_block(kind, bytes)?;
        let (block, carved) = create_block(self, device, pool, true, true, Holding::Driver(bytes))?;
        debug_assert!(carved.is_none());
        Ok(Range { pool: None, block, offset: 0, bytes })
    }

    /// The bytes, as whole words (which a fill zeroes, `vkCmdFillBuffer`),
    /// and the type of a block of its own of `bytes` of memory type `kind`.
    fn own_block(&self, kind: u32, bytes: u64) -> Result<(u64, PoolType), ExecutionError> {
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
        let resolved = pool_type(&self.layout, kind, self.storage_memory_type_bits)
            .map_err(ExecutionError::AllocationFailed)?;
        Ok((bytes, resolved))
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

fn memory_type_allowed(allowed_bits: u32, memory_type: u32) -> bool {
    1u32.checked_shl(memory_type)
        .is_some_and(|bit| allowed_bits & bit != 0)
}

/// First host memory type in `candidates` allowed by `staging_memory_type_bits`.
fn select_staging_kind(candidates: &[u32], staging_memory_type_bits: u32) -> Option<u32> {
    candidates
        .iter()
        .copied()
        .find(|&kind| memory_type_allowed(staging_memory_type_bits, kind))
}

/// Resolve topology `kind` to a Vulkan `memoryTypeIndex` allowed by
/// `allowed_bits` on the same heap.
///
/// Some drivers (e.g. Mesa ANV on Intel Battlemage / Arc Pro B70) expose a
/// compressed `DEVICE_LOCAL` memory type at index 0 whose `VkMemoryPropertyFlags`
/// match uncompressed `DEVICE_LOCAL` memory on the same heap, but whose bit is
/// clear in `VkMemoryRequirements::memoryTypeBits` for storage buffers. Keeping
/// `kind` as the topology/ledger identity while resolving `memory_type` on the
/// same heap preserves exact store/budget accounting and avoids failing open.
fn resolve_memory_type(
    layout: &crate::MemoryLayout,
    kind: u32,
    allowed_bits: u32,
) -> Result<u32, String> {
    let Some(requested) = layout.types.get(kind as usize).copied() else {
        return Err(format!("memory type {kind} is outside the device's {} types", layout.types.len()));
    };
    if memory_type_allowed(allowed_bits, kind) {
        return Ok(kind);
    }
    // Exact property match on the same heap (e.g. uncompressed sibling of a
    // compressed DEVICE_LOCAL type).
    if let Some(index) = layout.types.iter().enumerate().find_map(|(i, ty)| {
        let mt = i as u32;
        (memory_type_allowed(allowed_bits, mt)
            && ty.heap == requested.heap
            && ty.device_local == requested.device_local
            && ty.mappable() == requested.mappable()
            && ty.host_cached == requested.host_cached)
            .then_some(mt)
    }) {
        return Ok(index);
    }
    // Compatible superset on the same heap: preserve locality and never
    // downgrade mappability or cached host reads.
    if let Some(index) = layout.types.iter().enumerate().find_map(|(i, ty)| {
        let mt = i as u32;
        let reach_ok = !requested.mappable()
            || (ty.mappable() && (!requested.host_cached || ty.host_cached));
        (memory_type_allowed(allowed_bits, mt)
            && ty.heap == requested.heap
            && ty.device_local == requested.device_local
            && reach_ok)
            .then_some(mt)
    }) {
        return Ok(index);
    }
    Err(format!(
        "memory type {kind} on heap {} cannot back a buffer (allowed bits {allowed_bits:#x})",
        requested.heap
    ))
}

/// The memory type `kind` of `layout`, resolved against `allowed_bits`, as a
/// pool allocates from it.
fn pool_type(layout: &crate::MemoryLayout, kind: u32, allowed_bits: u32) -> Result<PoolType, String> {
    let memory_type = resolve_memory_type(layout, kind, allowed_bits)?;
    let reported = layout.types[memory_type as usize];
    Ok(PoolType {
        kind,
        memory_type,
        heap: reported.heap,
        mapped: reported.mappable(),
    })
}

/// An empty pool of memory types `kinds`, in order, charged as `charged`.
fn new_pool(
    layout: &crate::MemoryLayout,
    kinds: &[u32],
    charged: Charged,
    allowed_bits: u32,
) -> Result<Pool, String> {
    let (&last, passable) = kinds.split_last().ok_or("no memory type for the storage")?;
    Ok(Pool {
        kinds: kinds.to_vec(),
        charged,
        passable: passable
            .iter()
            .map(|&kind| pool_type(layout, kind, allowed_bits))
            .collect::<Result<Vec<_>, _>>()?,
        last: pool_type(layout, last, allowed_bits)?,
        blocks: Mutex::new(Vec::new()),
    })
}

fn destroy_block(device: &ash::Device, block: &Block) {
    unsafe {
        device.destroy_buffer(block.buffer, None);
        device.free_memory(block.memory, None);
    }
}

/// Maximum GTT headroom reserved for command submission, staging, upload
/// buffers, and driver internals on a discrete device's non-device-local heap
/// so shader-addressable pool blocks never exhaust GTT during `vkQueueSubmit2`
/// (`radv/amdgpu: Not enough memory for command submission`).
const NON_DEVICE_LOCAL_SUBMIT_RESERVE_MAX: u64 = 2 << 30;

/// Available bytes for a new dedicated pool block on a discrete device's
/// non-device-local (GTT) heap, keeping submission headroom free and guarding
/// against lagging `VK_EXT_memory_budget` usage counters via `tracked_usage`.
fn non_device_local_block_room(
    heap_size: u64,
    heap_budget: u64,
    heap_usage: u64,
    tracked_usage: u64,
) -> u64 {
    let effective_budget = if heap_budget > 0 {
        heap_budget.min(heap_size)
    } else {
        heap_size
    };
    let reserve = (effective_budget / 8).min(NON_DEVICE_LOCAL_SUBMIT_RESERVE_MAX);
    let effective_usage = heap_usage.max(tracked_usage);
    effective_budget
        .saturating_sub(reserve)
        .saturating_sub(effective_usage)
}

/// Excess bytes to trim from a non-dedicated pool block charge when its heap's
/// current headroom (`room`) cannot hold the grown slab. Dedicated blocks
/// already match their allocation, and no block shrinks below `least`.
fn trim_for_room(charged_bytes: u64, least: u64, room: u64, dedicated: bool) -> u64 {
    if !dedicated && charged_bytes > room {
        charged_bytes.saturating_sub(least)
    } else {
        0
    }
}

/// Buffer usage flags for a block. Staging blocks omit
/// `SHADER_DEVICE_ADDRESS` because they are only used by `vkCmdCopyBuffer`.
fn block_buffer_usage(shader_addressable: bool) -> vk::BufferUsageFlags {
    // Any buffer may hold a grid a launch wrote (`DirectLaunch::grid`).
    let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::INDIRECT_BUFFER;
    if shader_addressable {
        usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    usage
}

fn block_buffer_create_info<'a>(
    bytes: u64,
    shader_addressable: bool,
    families: &'a [u32],
) -> vk::BufferCreateInfo<'a> {
    vk::BufferCreateInfo::default()
        .size(bytes)
        .usage(block_buffer_usage(shader_addressable))
        .sharing_mode(if families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        })
        .queue_family_indices(if families.len() > 1 { families } else { &[] })
}

/// Query `VkMemoryRequirements::memoryTypeBits` for a block buffer using
/// Vulkan 1.3 `maintenance4` (`vkGetDeviceBufferMemoryRequirements`), without
/// creating a `VkBuffer`.
fn buffer_memory_type_bits(device: &ash::Device, families: &[u32], shader_addressable: bool) -> u32 {
    let info = block_buffer_create_info(4, shader_addressable, families);
    let query = vk::DeviceBufferMemoryRequirements::default().create_info(&info);
    let mut requirements = vk::MemoryRequirements2::default();
    unsafe { device.get_device_buffer_memory_requirements(&query, &mut requirements) };
    requirements.memory_requirements.memory_type_bits
}

/// A block of `charge`'s bytes on `pool`'s type, mapped persistently where
/// the type is mappable. The memory the buffer requires beyond them (its
/// allocation's rounding) is charged to the same entry before the memory is
/// allocated, so the charge is the memory's exact bytes.
fn create_block(
    memory_owner: &Memory,
    device: &ash::Device,
    pool: PoolType,
    dedicated: bool,
    shader_addressable: bool,
    mut holding: Holding<'_>,
) -> Result<(Arc<Block>, Option<Carved>), ExecutionError> {
    // Every memory object holds at least the device's least
    // (`Memory::least_object`), its free space a pool block's to use and
    // a block of its own's to strand.
    if let Some(short) = memory_owner.least_object.checked_sub(holding.bytes()).filter(|&short| short > 0) {
        holding.grow(short)?;
    }
    let bytes = holding.bytes();
    let info = block_buffer_create_info(bytes, shader_addressable, memory_owner.families.as_slice());
    let buffer = crate::device::memory_call(
        unsafe { device.create_buffer(&info, None) },
        "vkCreateBuffer",
        bytes,
        memory_owner,
        ExecutionError::AllocationFailed,
    )?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let refuse = |error: ExecutionError| -> Result<(Arc<Block>, Option<Carved>), ExecutionError> {
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
    if let Err(refused) = holding.grow_padding(allocated_bytes - bytes) {
        return refuse(refused);
    }
    // The charge taken, the block's memory joins the heap within one
    // transition, until it is counted.
    let _transition = memory_owner.transition();
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(allocated_bytes)
        .memory_type_index(pool.memory_type);
    if shader_addressable {
        allocate = allocate.push_next(&mut flags);
    }
    let memory = match crate::device::memory_call(
        unsafe { device.allocate_memory(&allocate, None) },
        "vkAllocateMemory",
        bytes,
        memory_owner,
        ExecutionError::AllocationFailed,
    ) {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    let bound = crate::device::memory_call(
        unsafe { device.bind_buffer_memory(buffer, memory, 0) },
        "vkBindBufferMemory",
        bytes,
        memory_owner,
        ExecutionError::AllocationFailed,
    );
    let mapped = bound.and_then(|()| match pool.mapped {
        true => crate::device::memory_call(
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) },
            "vkMapMemory",
            bytes,
            memory_owner,
            ExecutionError::AllocationFailed,
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
            return Err(error);
        }
    };
    let address = shader_addressable.then(|| unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    });
    let (held, carved) = holding.held();
    let block = Arc::new(Block {
        kind: pool.kind,
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
        held,
    });
    memory_owner.count(&block);
    Ok((block, carved))
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
        self.block
            .address
            .expect("device address requested for a block without shader device address")
            + self.offset
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

    /// Verify that `[offset, offset + bytes)` lies within this range without
    /// overflowing `u64`.
    pub(crate) fn check_span(&self, offset: u64, bytes: u64) {
        assert!(
            offset.checked_add(bytes).is_some_and(|end| end <= self.bytes),
            "range access [{offset}..+{bytes}) exceeds range length {}",
            self.bytes,
        );
    }

    /// Host write of mapped storage: a plain memory write.
    pub(crate) fn write_mapped(&self, offset: u64, bytes: &[u8]) {
        let base = self
            .mapped()
            .expect("write_mapped precondition: the storage is mapped");
        self.check_span(offset, bytes.len() as u64);
        // SAFETY: the bytes lie inside this range of the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(offset as usize), bytes.len())
        };
    }

    pub(crate) fn read_mapped(&self, offset: u64, into: &mut [u8]) {
        let base = self
            .mapped()
            .expect("read_mapped precondition: the storage is mapped");
        self.check_span(offset, into.len() as u64);
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

    /// The kind the allocation lies in: the topology kind it was asked for
    /// (preserving store/path accounting even when resolved to a sibling
    /// Vulkan memory type on the same heap).
    pub fn kind(&self) -> u32 {
        self.range.block.kind
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
            kind: 0,
            memory_type: 0,
            heap: 0,
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            address: Some(0),
            bytes,
            allocated_bytes: bytes,
            mapped: std::ptr::null_mut(),
            dedicated: false,
            free: Mutex::new(FreeRanges::new(bytes)),
            held: Held::Charged({
                let entry = tally();
                Charge::took(&entry.change(), &entry, bytes, Wiring::Pageable).into_carvable()
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
                kind: 0,
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

    #[test]
    fn range_check_span_accepts_in_bounds_spans() {
        let blk = block(256);
        let sub = range(&blk, 64);
        sub.check_span(0, 0);
        sub.check_span(0, 64);
        sub.check_span(64, 0);
        sub.check_span(32, 32);
    }

    #[test]
    #[should_panic(expected = "exceeds range length")]
    fn range_check_span_rejects_past_end() {
        let blk = block(256);
        let sub = range(&blk, 64);
        sub.check_span(32, 33);
    }

    #[test]
    #[should_panic(expected = "exceeds range length")]
    fn range_check_span_rejects_offset_past_end() {
        let blk = block(256);
        let sub = range(&blk, 64);
        sub.check_span(65, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds range length")]
    fn range_check_span_rejects_u64_overflow() {
        let blk = block(256);
        let sub = range(&blk, 64);
        sub.check_span(u64::MAX - 8, 16);
    }

    #[test]
    fn staging_omits_shader_device_address_while_kind_keeps_it() {
        assert!(!Charged::Staging.shader_addressable());
        assert!(Charged::Kind.shader_addressable());

        let staging_usage = block_buffer_usage(Charged::Staging.shader_addressable());
        assert!(!staging_usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
        assert!(staging_usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert!(staging_usage.contains(vk::BufferUsageFlags::TRANSFER_DST));

        let kind_usage = block_buffer_usage(Charged::Kind.shader_addressable());
        assert!(kind_usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS));
    }

    #[test]
    #[should_panic(expected = "without shader device address")]
    fn range_address_panics_on_non_addressable_block() {
        let mut blk = block(256);
        Arc::get_mut(&mut blk).unwrap().address = None;
        let sub = range(&blk, 64);
        let _ = sub.address();
    }

    #[test]
    fn trim_for_room_caps_non_dedicated_slab_growth_when_headroom_is_tight() {
        const MIB: u64 = 1 << 20;
        // Ample headroom: keep the grown 512 MiB slab.
        assert_eq!(trim_for_room(512 * MIB, 4 * MIB, 1024 * MIB, false), 0);
        // Tight headroom: shrink a 512 MiB slab back to `least` (4 MiB).
        assert_eq!(trim_for_room(512 * MIB, 4 * MIB, 128 * MIB, false), 508 * MIB);
        // Zero headroom: still never shrinks below `least`.
        assert_eq!(trim_for_room(512 * MIB, 4 * MIB, 0, false), 508 * MIB);
        // Dedicated block: never trimmed.
        assert_eq!(trim_for_room(512 * MIB, 512 * MIB, 128 * MIB, true), 0);
    }

    #[test]
    fn non_device_local_block_room_reserves_submit_headroom_and_respects_tracked_usage() {
        const GIB: u64 = 1 << 30;
        // 32 GiB GTT heap (64 GiB Linux amdgpu host): reserves 2 GiB for submit/staging/driver.
        assert_eq!(
            non_device_local_block_room(32 * GIB, 32 * GIB, 1 * GIB, 0),
            29 * GIB
        );
        // Lagging kernel heapUsage (0) is overridden by exact tracked_usage.
        assert_eq!(
            non_device_local_block_room(32 * GIB, 32 * GIB, 0, 29 * GIB),
            1 * GIB
        );
        // Once tracked_usage reaches budget - reserve, no more blocks are admitted.
        assert_eq!(
            non_device_local_block_room(32 * GIB, 32 * GIB, 0, 30 * GIB),
            0
        );
        // Driver budget below raw heap size caps effective budget.
        assert_eq!(
            non_device_local_block_room(32 * GIB, 24 * GIB, 2 * GIB, 2 * GIB),
            20 * GIB
        );
        // Zero reported budget falls back to heap size; small heaps cap reserve at 1/8.
        assert_eq!(
            non_device_local_block_room(8 * GIB, 0, 1 * GIB, 0),
            6 * GIB
        );
    }

    #[test]
    fn resolve_memory_type_skips_non_storage_type_on_same_heap_and_preserves_kind() {
        use crate::device_info::{MemoryHeap, MemoryLayout, MemoryType};
        const GIB: u64 = 1 << 30;
        // Intel Arc Pro B70 (Mesa ANV BMG): type 0 is compressed DEVICE_LOCAL
        // on heap 0 (bit 0 clear in storage memoryTypeBits), type 1 is
        // uncompressed DEVICE_LOCAL on heap 0 (bit 1 set).
        let layout = MemoryLayout {
            heaps: vec![
                MemoryHeap { bytes: 32 * GIB, device_local: true },
                MemoryHeap { bytes: 64 * GIB, device_local: false },
            ],
            types: vec![
                MemoryType { heap: 0, device_local: true, host_visible: false, host_coherent: false, host_cached: false },
                MemoryType { heap: 0, device_local: true, host_visible: false, host_coherent: false, host_cached: false },
                MemoryType { heap: 0, device_local: true, host_visible: true, host_coherent: true, host_cached: false },
                MemoryType { heap: 1, device_local: false, host_visible: true, host_coherent: true, host_cached: false },
                MemoryType { heap: 1, device_local: false, host_visible: true, host_coherent: true, host_cached: true },
            ],
        };

        // ANV BMG: kind 0 resolves to memory_type 1 while keeping kind == 0 for store/path accounting.
        let resolved = pool_type(&layout, 0, 0b11110).unwrap();
        assert_eq!(resolved, PoolType { kind: 0, memory_type: 1, heap: 0, mapped: false });

        // Identity fast path when requested bit is set.
        let identity = pool_type(&layout, 1, 0b11110).unwrap();
        assert_eq!(identity, PoolType { kind: 1, memory_type: 1, heap: 0, mapped: false });

        // Superset fallback on same heap: unmappable DL falls back to mappable DL (type 2).
        let rebar = pool_type(&layout, 0, 0b11100).unwrap();
        assert_eq!(rebar, PoolType { kind: 0, memory_type: 2, heap: 0, mapped: true });

        // Upload (uncached mappable) falls back to cached mappable on same heap.
        let upload = pool_type(&layout, 3, 0b10000).unwrap();
        assert_eq!(upload, PoolType { kind: 3, memory_type: 4, heap: 1, mapped: true });

        // Cached mappable refuses to downgrade to uncached mappable.
        assert!(pool_type(&layout, 4, 0b01000).is_err());

        // Never crosses heaps when heap 0 has no allowed memory types.
        assert!(pool_type(&layout, 0, 0b11000).is_err());

        // Staging selection skips incompatible host[0] and picks next compatible host type.
        assert_eq!(select_staging_kind(&[3, 4], 0b10000), Some(4));
        assert_eq!(select_staging_kind(&[3, 4], 0b00111), None);

        // Out-of-range kind or zero mask returns Err cleanly.
        assert!(pool_type(&layout, 99, u32::MAX).is_err());
        assert!(pool_type(&layout, 0, 0).is_err());
    }

    #[test]
    fn charged_with_first_avoids_double_charging_own_and_tolerates_padding() {
        const MIB: u64 = 1 << 20;
        let identity = seismic_topology::StoreIdentity::HostRam;
        // Mirrors a tight 16 GB Vulkan store after bank growth: `own` has less
        // free room than the 300 MiB dedicated allocation, while `entry`
        // (`constants_claim`) already covers the full 300 MiB.
        let own_tally = seismic_memory::entry::Tally::limited(identity.clone(), 0);
        let entry_tally = seismic_memory::entry::Tally::limited(identity, 300 * MIB);
        let own: Arc<dyn Entry> = Arc::new(own_tally.clone());
        let entry: Arc<dyn Entry> = Arc::new(entry_tally.clone());
        let change = own.change();

        let (remainder, first) = charge_block_for(
            &change,
            &own,
            &entry,
            None,
            BLOCK_CAP_BYTES,
            300 * MIB,
            1,
        )
        .expect("dedicated block draws directly from entry even with 0 own room");
        assert_eq!(first.bytes(), 300 * MIB);
        assert_eq!(remainder.bytes(), 0);

        let mut holding = Holding::ChargedWithFirst(&change, remainder, first);
        assert_eq!(holding.bytes(), 300 * MIB);
        // Mandatory VkMemoryRequirements::size rounding (e.g. 64 KiB) is
        // recorded on `own` without refusing the already-admitted allocation.
        holding.grow_padding(65_536).unwrap();
        assert_eq!(holding.bytes(), 300 * MIB + 65_536);

        let (Held::Charged(carvable), Some(carved)) = holding.held() else {
            panic!("ChargedWithFirst must yield Held::Charged and Carved");
        };
        assert_eq!(carvable.bytes(), 300 * MIB + 65_536);
        assert_eq!(carvable.uncarved_bytes(), 65_536);
        assert_eq!(carved.bytes(), 300 * MIB);
        assert_eq!(entry_tally.held(), 300 * MIB);
        assert_eq!(own_tally.held(), 65_536);

        // Buffer::drop drops `carved` first, then Block::free releases `carvable`.
        drop(carved);
        assert_eq!(entry_tally.held(), 0);
        assert_eq!(own_tally.held(), 300 * MIB + 65_536);
        carvable.release(&change);
        assert_eq!(own_tally.held(), 0);
        assert_eq!(entry_tally.held(), 0);
    }

    #[test]
    fn charged_with_first_trim_for_room_only_shrinks_uncarved_remainder() {
        const MIB: u64 = 1 << 20;
        let identity = seismic_topology::StoreIdentity::HostRam;
        let own_tally = seismic_memory::entry::Tally::unbounded(identity.clone());
        let entry_tally = seismic_memory::entry::Tally::unbounded(identity);
        let own: Arc<dyn Entry> = Arc::new(own_tally.clone());
        let entry: Arc<dyn Entry> = Arc::new(entry_tally.clone());
        let change = own.change();

        let (mut remainder, first) = charge_block_for(
            &change,
            &own,
            &entry,
            Some(64 * MIB),
            BLOCK_CAP_BYTES,
            4 * MIB,
            1,
        )
        .unwrap();
        assert_eq!(first.bytes(), 4 * MIB);
        assert_eq!(remainder.bytes(), 124 * MIB);

        let total = remainder.bytes() + first.bytes();
        let trim = trim_for_room(total, 4 * MIB, 32 * MIB, false);
        assert_eq!(trim, 124 * MIB);
        remainder.shrink(&change, trim);
        assert_eq!(remainder.bytes(), 0);
        assert_eq!(first.bytes(), 4 * MIB);
        assert_eq!(own_tally.held(), 0);
        assert_eq!(entry_tally.held(), 4 * MIB);

        drop(remainder);
        drop(first);
        assert_eq!(own_tally.held(), 0);
        assert_eq!(entry_tally.held(), 0);
    }
}
