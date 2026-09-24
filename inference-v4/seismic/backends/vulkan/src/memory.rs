//! The owned block allocator (§8.2, §15.4).
//!
//! Device memory is allocated in blocks, each one `VkBuffer` with a device
//! address; an allocation is a range of a block, and its address is the
//! block's address plus the range's offset. Blocks of one storage kind grow
//! from 64 MiB to the block cap (min(`maxMemoryAllocationSize`,
//! `maxBufferSize`, 1 GiB)); a request beyond the cap gets a dedicated block,
//! released with it. Host-visible blocks are persistently mapped.

use crate::device::{call, Inner};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use std::sync::{Arc, Mutex};

const FIRST_BLOCK_BYTES: u64 = 64 << 20;
const BLOCK_CAP_BYTES: u64 = 1 << 30;
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
    /// Free ranges `(offset, bytes)`, sorted and coalesced.
    free: Mutex<Vec<(u64, u64)>>,
}

// The mapping is plain host memory; ranges are handed out exclusively.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        let mut free = self.free.lock().expect("block free list lock is never poisoned");
        let (index, start) = free.iter().enumerate().find_map(|(index, (offset, size))| {
            let start = offset.next_multiple_of(alignment);
            (start + bytes <= offset + size).then_some((index, start))
        })?;
        let (offset, size) = free.remove(index);
        if start + bytes < offset + size {
            free.insert(index, (start + bytes, offset + size - start - bytes));
        }
        if start > offset {
            free.insert(index, (offset, start - offset));
        }
        Some(start)
    }

    /// Return a range; true when the block is entirely free afterwards.
    fn give(&self, offset: u64, bytes: u64) -> bool {
        let mut free = self.free.lock().expect("block free list lock is never poisoned");
        let index = free.partition_point(|(start, _)| *start < offset);
        free.insert(index, (offset, bytes));
        if index + 1 < free.len() && free[index].0 + free[index].1 == free[index + 1].0 {
            free[index].1 += free.remove(index + 1).1;
        }
        if index > 0 && free[index - 1].0 + free[index - 1].1 == free[index].0 {
            free[index - 1].1 += free.remove(index).1;
        }
        free.len() == 1 && free[0] == (0, self.bytes)
    }
}

/// The blocks of one storage kind.
struct Pool {
    kind: StorageKind,
    memory_type: u32,
    mapped: bool,
    blocks: Mutex<Vec<Arc<Block>>>,
}

pub(crate) struct Memory {
    pools: [Pool; 3],
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
            let candidates =
                || types.iter().enumerate().filter(move |(_, kind)| kind.property_flags.contains(required));
            candidates()
                .find(|(_, kind)| kind.property_flags.contains(preferred))
                .or_else(|| candidates().next())
                .map(|(index, kind)| (index as u32, kind.property_flags))
        };
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        // Unified memory (integrated GPUs, lavapipe): the device-local heap
        // is host-visible, so device storage is mapped too.
        let unified = facts.host_visible_device_local_bytes == facts.device_local_bytes;
        let device = if unified { find(local | visible, local | visible) } else { find(local, local) }
            .ok_or("no device-local memory type")?;
        // A small BAR window (typically 256 MiB) is not spent on upload
        // regions; ReBAR/SAM and unified memory are.
        let upload = if facts.host_visible_device_local_bytes > 256 << 20 {
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
            blocks: Mutex::new(Vec::new()),
        };
        Ok(Self {
            pools: [
                pool(StorageKind::Device, device),
                pool(StorageKind::Upload, upload),
                pool(StorageKind::Staging, staging),
            ],
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

    pub(crate) fn allocate(
        &self,
        device: &ash::Device,
        kind: StorageKind,
        bytes: u64,
        alignment: u64,
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
        let mut blocks = pool.blocks.lock().expect("pool lock is never poisoned");
        let found = blocks
            .iter()
            .filter(|block| !block.dedicated)
            .find_map(|block| block.take(bytes, alignment).map(|offset| (block.clone(), offset)));
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                let largest = blocks.iter().filter(|block| !block.dedicated).map(|block| block.bytes).max();
                let size = largest.map_or(FIRST_BLOCK_BYTES, |largest| largest * 2).min(self.block_cap);
                let dedicated = bytes > size;
                let block = create_block(device, pool, if dedicated { bytes } else { size }, dedicated)?;
                let offset = block.take(bytes, alignment).expect("a new block holds its first allocation");
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

    /// Return `range` to its block; a dedicated block is freed with it.
    pub(crate) fn release(&self, device: &ash::Device, range: &Range) {
        let empty = range.block.give(range.offset, range.bytes);
        if empty && range.block.dedicated {
            let mut blocks = self.pool(range.kind).blocks.lock().expect("pool lock is never poisoned");
            blocks.retain(|block| !Arc::ptr_eq(block, &range.block));
            destroy_block(device, &range.block);
        }
    }

    /// Destroy every block. The device is idle and no allocation lives.
    pub(crate) fn destroy(&self, device: &ash::Device) {
        for pool in &self.pools {
            for block in pool.blocks.lock().expect("pool lock is never poisoned").drain(..) {
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

fn create_block(device: &ash::Device, pool: &Pool, bytes: u64, dedicated: bool) -> Result<Arc<Block>, ExecutionError> {
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
    let buffer = call(unsafe { device.create_buffer(&info, None) }, "vkCreateBuffer").map_err(allocation)?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    if requirements.memory_type_bits & (1 << pool.memory_type) == 0 {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(ExecutionError::AllocationFailed(format!(
            "memory type {} cannot back a storage buffer",
            pool.memory_type
        )));
    }
    let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
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
                other => ExecutionError::AllocationFailed(format!("vkAllocateMemory failed: {other}")),
            });
        }
    };
    let bound = call(unsafe { device.bind_buffer_memory(buffer, memory, 0) }, "vkBindBufferMemory");
    let mapped = bound.and_then(|()| {
        if pool.mapped {
            call(
                unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) },
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
    let address =
        unsafe { device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };
    Ok(Arc::new(Block {
        memory,
        buffer,
        address,
        bytes,
        mapped,
        dedicated,
        free: Mutex::new(vec![(0, bytes)]),
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
        (!self.block.mapped.is_null()).then(|| unsafe { self.block.mapped.add(self.offset as usize) })
    }

    /// Host write of mapped storage: a plain memory write.
    pub(crate) fn write_mapped(&self, offset: u64, bytes: &[u8]) {
        let base = self.mapped().expect("write_mapped precondition: the storage is mapped");
        assert!(offset + bytes.len() as u64 <= self.bytes, "mapped write exceeds its range");
        // SAFETY: the bytes lie inside this range of the mapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(offset as usize), bytes.len()) };
    }

    pub(crate) fn read_mapped(&self, offset: u64, into: &mut [u8]) {
        let base = self.mapped().expect("read_mapped precondition: the storage is mapped");
        assert!(offset + into.len() as u64 <= self.bytes, "mapped read exceeds its range");
        // SAFETY: the bytes lie inside this range of the mapping.
        unsafe { std::ptr::copy_nonoverlapping(base.add(offset as usize), into.as_mut_ptr(), into.len()) };
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
