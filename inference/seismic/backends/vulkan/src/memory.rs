//! The owned block allocator (§8.2, §15.4) over the shared block core
//! (`seismic_compiler::blocks`).
//!
//! Device memory is allocated in blocks, each one `VkBuffer` with a device
//! address; an allocation is a range of a block, and its address is the
//! block's address plus the range's offset. Blocks of one storage kind grow
//! from 64 MiB to the block cap (min(`maxMemoryAllocationSize`,
//! `maxBufferSize`, 1 GiB)); a request beyond the cap gets a dedicated block.
//! A block is freed once it holds no allocation, so released storage returns
//! to the device's memory budget: the engine reads the budget less usage as
//! its headroom, and a block kept empty would hide released storage from it.
//! A block's unallocated space counts against that budget too, so a new
//! device-local block never exceeds the room the caller grants beyond the
//! free space device-local blocks already hold, and when a doubled block
//! does not fit that room the block holds only its allocation: an
//! allocation that fits its grant never grows the device's usage past it. That free space is device
//! memory this process holds beyond its charges
//! ([`Memory::device_local_slack`]). Host-visible blocks are persistently
//! mapped.

use crate::device::{call, Inner};
use ash::vk;
use seismic_compiler::blocks::{block_bytes, FreeRanges, BLOCK_CAP_BYTES};
use seismic_compiler::errors::ExecutionError;
use std::sync::{Arc, Mutex};
/// Every allocation starts at least this aligned.
const MINIMUM_ALIGNMENT: u64 = 16;

/// What an allocation is for; each kind has its own memory type and blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StorageKind {
    /// Device-local storage.
    Device,
    /// Inputs the host writes before each submission: host-visible,
    /// device-local when the device has a large such heap.
    Upload,
    /// Host-visible staging for transfers of unmapped device storage.
    Staging,
}

struct Block {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    address: u64,
    bytes: u64,
    /// Base of the persistent mapping; null for unmapped memory.
    mapped: *mut u8,
    dedicated: bool,
    free: Mutex<FreeRanges>,
}

// The mapping is plain host memory; ranges are handed out exclusively.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        self.ranges().take(bytes, alignment)
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

/// The blocks of one storage kind.
struct Pool {
    kind: StorageKind,
    memory_type: u32,
    mapped: bool,
    device_local: bool,
    blocks: Mutex<Vec<Arc<Block>>>,
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
}

pub(crate) struct Memory {
    pools: [Pool; 3],
    /// Device-local bytes of the device's own ranges (argument chunks,
    /// staging), which live until the device is destroyed: held, charged
    /// to no allocation.
    internal: std::sync::atomic::AtomicU64,
    block_cap: u64,
    max_allocation: u64,
}

impl Memory {
    pub(crate) fn new(
        properties: &vk::PhysicalDeviceMemoryProperties,
        facts: &crate::Facts,
    ) -> Result<Self, String> {
        let types = &properties.memory_types[..properties.memory_type_count as usize];
        let find = |required: vk::MemoryPropertyFlags, preferred: vk::MemoryPropertyFlags| {
            let candidates = || {
                types
                    .iter()
                    .enumerate()
                    .filter(move |(_, kind)| kind.property_flags.contains(required))
            };
            candidates()
                .find(|(_, kind)| kind.property_flags.contains(preferred))
                .or_else(|| candidates().next())
                .map(|(index, kind)| (index as u32, kind.property_flags))
        };
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        // Unified memory (integrated GPUs, lavapipe): the device-local heap
        // is host-visible, so device storage is mapped too.
        let unified = facts.host_visible_device_local_bytes == facts.device_local_bytes;
        let device = if unified {
            find(local | visible, local | visible)
        } else {
            find(local, local)
        }
        .ok_or("no device-local memory type")?;
        // A small BAR window (typically 256 MiB) is not spent on upload
        // regions; ReBAR/SAM and unified memory are. Every upload type is
        // HOST_COHERENT (`visible`): host writes need no flush.
        let upload = if facts.uploads_device_local() {
            find(local | visible, local | visible)
        } else {
            find(visible, visible)
        }
        .ok_or("no host-visible coherent memory type")?;
        let staging = find(visible, visible | vk::MemoryPropertyFlags::HOST_CACHED)
            .ok_or("no host-visible coherent memory type")?;
        let pool = |kind, (memory_type, flags): (u32, vk::MemoryPropertyFlags)| Pool {
            kind,
            memory_type,
            mapped: flags.contains(visible),
            device_local: flags.contains(local),
            blocks: Mutex::new(Vec::new()),
        };
        Ok(Self {
            pools: [
                pool(StorageKind::Device, device),
                pool(StorageKind::Upload, upload),
                pool(StorageKind::Staging, staging),
            ],
            internal: std::sync::atomic::AtomicU64::new(0),
            block_cap: facts.limits.max_allocation_bytes.min(BLOCK_CAP_BYTES),
            max_allocation: facts.limits.max_allocation_bytes,
        })
    }

    fn pool(&self, kind: StorageKind) -> &Pool {
        self.pools
            .iter()
            .find(|pool| pool.kind == kind)
            .expect("every storage kind has a pool")
    }

    /// Free space held by device-local blocks: device usage that no
    /// allocation charges.
    pub(crate) fn device_local_slack(&self) -> u64 {
        self.pools
            .iter()
            .filter(|pool| pool.device_local)
            .map(|pool| {
                pool.blocks
                    .lock()
                    .expect("pool lock is never poisoned")
                    .iter()
                    .map(|block| block.ranges().free_bytes())
                    .sum::<u64>()
            })
            .sum()
    }

    /// A range of `bytes`. `room`, when given, is the device usage the
    /// caller may add: a new device-local block is sized within it less the
    /// free space device-local blocks already hold, and an allocation that
    /// needs a block larger than that is refused as a capacity shortfall.
    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        kind: StorageKind,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Range, ExecutionError> {
        if bytes > self.max_allocation {
            return Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: self.max_allocation,
            });
        }
        let bytes = bytes.max(1);
        let alignment = alignment.max(MINIMUM_ALIGNMENT);
        let pool = self.pool(kind);
        // Read before taking this pool's lock: the slack spans every pool.
        let room = room
            .filter(|_| pool.device_local)
            .map(|room| room.saturating_sub(self.device_local_slack()));
        let mut blocks = pool.blocks.lock().expect("pool lock is never poisoned");
        let found = blocks
            .iter()
            .filter(|block| !block.dedicated)
            .find_map(|block| {
                block
                    .take(bytes, alignment)
                    .map(|offset| (block.clone(), offset))
            });
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                let largest = blocks
                    .iter()
                    .filter(|block| !block.dedicated)
                    .map(|block| block.bytes)
                    .max();
                let size = block_bytes(largest, self.block_cap, bytes, room, 1)?;
                let dedicated = bytes > self.block_cap;
                let block = create_block(device, pool, size, dedicated)?;
                let offset = block
                    .take(bytes, alignment)
                    .expect("a new block holds its first allocation");
                blocks.push(block.clone());
                (block, offset)
            }
        };
        Ok(Range {
            kind,
            block,
            offset,
            bytes,
        })
    }

    /// A range for the device's own use (argument chunks, staging), kept
    /// until the device is destroyed and counted as held when device-local.
    pub(crate) fn allocate_internal(
        &self,
        device: &ash::Device,
        kind: StorageKind,
        bytes: u64,
        alignment: u64,
    ) -> Result<Range, ExecutionError> {
        let range = self.allocate(device, kind, bytes, alignment, None)?;
        if self.pool(kind).device_local {
            self.internal
                .fetch_add(range.bytes, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(range)
    }

    /// Bytes of every device-local block.
    pub(crate) fn device_local_block_bytes(&self) -> u64 {
        self.pools
            .iter()
            .filter(|pool| pool.device_local)
            .map(|pool| {
                pool.blocks
                    .lock()
                    .expect("pool lock is never poisoned")
                    .iter()
                    .map(|block| block.bytes)
                    .sum::<u64>()
            })
            .sum()
    }

    /// Device-local bytes of the device's own ranges.
    pub(crate) fn internal_device_local(&self) -> u64 {
        self.internal.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Return `range` to its block, and free the block once it holds no
    /// allocation.
    pub(crate) fn release(&self, device: &ash::Device, range: &Range) {
        if let Some(block) = self.pool(range.kind).give(range) {
            destroy_block(device, &block);
        }
    }

    /// Destroy every block. The device is idle and no allocation lives.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        for pool in &self.pools {
            for block in pool
                .blocks
                .lock()
                .expect("pool lock is never poisoned")
                .drain(..)
            {
                destroy_block(device, &block);
            }
        }
    }
}

fn destroy_block(device: &ash::Device, block: &Block) {
    unsafe {
        device.destroy_buffer(block.buffer, None);
        device.free_memory(block.memory, None);
    }
}

fn create_block(
    device: &ash::Device,
    pool: &Pool,
    bytes: u64,
    dedicated: bool,
) -> Result<Arc<Block>, ExecutionError> {
    let allocation = |error: ExecutionError| match error {
        ExecutionError::SubmissionFailed(message) => ExecutionError::AllocationFailed(message),
        other => other,
    };
    let info = vk::BufferCreateInfo::default()
        .size(bytes)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = call(
        unsafe { device.create_buffer(&info, None) },
        "vkCreateBuffer",
    )
    .map_err(allocation)?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    if requirements.memory_type_bits & (1 << pool.memory_type) == 0 {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(ExecutionError::AllocationFailed(format!(
            "memory type {} cannot back a storage buffer",
            pool.memory_type
        )));
    }
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(pool.memory_type)
        .push_next(&mut flags);
    let memory = match unsafe { device.allocate_memory(&allocate, None) } {
        Ok(memory) => memory,
        Err(result) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(match result {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    ExecutionError::AllocationCapacity {
                        required: bytes.into(),
                        available: 0,
                    }
                }
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
    let mapped = bound.and_then(|()| {
        if pool.mapped {
            call(
                unsafe {
                    device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                },
                "vkMapMemory",
            )
            .map(|pointer| pointer.cast::<u8>())
        } else {
            Ok(std::ptr::null_mut())
        }
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
    Ok(Arc::new(Block {
        memory,
        buffer,
        address,
        bytes,
        mapped,
        dedicated,
        free: Mutex::new(FreeRanges::new(bytes)),
    }))
}

/// A range of a block. [`Buffer`] releases it on drop; the device's own
/// ranges (argument chunks, staging) live until the device is destroyed.
pub(crate) struct Range {
    kind: StorageKind,
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

    fn block(bytes: u64) -> Arc<Block> {
        Arc::new(Block {
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            address: 0,
            bytes,
            mapped: std::ptr::null_mut(),
            dedicated: false,
            free: Mutex::new(FreeRanges::new(bytes)),
        })
    }

    fn range(pool: &Pool, block: &Arc<Block>, bytes: u64) -> Range {
        Range {
            kind: pool.kind,
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
            kind: StorageKind::Device,
            memory_type: 0,
            mapped: false,
            device_local: true,
            blocks: Mutex::new(Vec::new()),
        };
        let blocks = (0..4).map(|_| block(16 * SLAB)).collect::<Vec<_>>();
        pool.blocks.lock().unwrap().extend(blocks.iter().cloned());
        // The model's resident storage occupies the first block; retained
        // state fills the rest, one slab beside it in the first block.
        let model = range(&pool, &blocks[0], 15 * SLAB);
        let retained = blocks
            .iter()
            .enumerate()
            .flat_map(|(index, block)| {
                let slabs = if index == 0 { 1 } else { 16 };
                (0..slabs)
                    .map(|_| range(&pool, block, SLAB))
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
        let reused = range(&pool, &blocks[0], SLAB);
        assert!(pool.give(&reused).is_none());
        assert!(pool.give(&model).is_some());
        assert_eq!(usage(&pool), 0);
    }
}
