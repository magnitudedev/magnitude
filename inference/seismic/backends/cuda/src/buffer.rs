//! CUDA storage exposed through the generic device service: a range of a
//! device block, pinned host memory mapped into the device address space,
//! or a pool's block of its own, of device memory or of registered host
//! pages.

use crate::blocks::{DeviceBlock, Range};
use crate::driver::{DriverError, HostMapped, HostNodeBlock};
use crate::registered::RegisteredBlock;
use std::sync::Arc;

#[derive(Clone)]
pub struct Buffer {
    memory: Arc<Memory>,
}

enum Memory {
    /// A range of one of the device's blocks.
    Device(Range),
    /// Host-written inputs: filled by the host without a driver call.
    Mapped(HostMapped),
    /// A block of device memory of its own (`Device::allocate_block`).
    Block(DeviceBlock),
    /// A block of host-NUMA memory created by the driver for a unified-memory
    /// device (`Device::allocate_block`).
    HostNode(HostNodeBlock),
    /// A block of host pages the device reads in place over its link
    /// (`Device::allocate_block`).
    Registered(RegisteredBlock),
}

impl Buffer {
    pub(crate) fn device(range: Range) -> Self {
        Self { memory: Arc::new(Memory::Device(range)) }
    }
    pub(crate) fn mapped(mapped: HostMapped) -> Self {
        Self { memory: Arc::new(Memory::Mapped(mapped)) }
    }
    pub(crate) fn block(block: DeviceBlock) -> Self {
        Self { memory: Arc::new(Memory::Block(block)) }
    }
    pub(crate) fn host_node(block: HostNodeBlock) -> Self {
        Self { memory: Arc::new(Memory::HostNode(block)) }
    }
    pub(crate) fn registered(block: RegisteredBlock) -> Self {
        Self { memory: Arc::new(Memory::Registered(block)) }
    }
    pub fn len(&self) -> u64 {
        match &*self.memory {
            Memory::Device(range) => range.bytes(),
            Memory::Mapped(mapped) => mapped.bytes as u64,
            Memory::Block(block) => block.bytes(),
            Memory::HostNode(block) => block.bytes() as u64,
            Memory::Registered(block) => block.bytes() as u64,
        }
    }
    /// The host address of host memory's first byte (mapped, host-node or
    /// registered memory); device memory has none (the host reaches it by
    /// driver copies).
    pub(crate) fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        match &*self.memory {
            Memory::Mapped(mapped) => std::ptr::NonNull::new(mapped.host()),
            Memory::HostNode(block) => Some(block.host()),
            Memory::Registered(block) => Some(block.host()),
            Memory::Device(_) | Memory::Block(_) => None,
        }
    }
    /// The address kernels use.
    pub fn pointer(&self) -> u64 {
        match &*self.memory {
            Memory::Device(range) => range.pointer(),
            Memory::Mapped(mapped) => mapped.device,
            Memory::Block(block) => block.pointer(),
            Memory::HostNode(block) => block.device(),
            Memory::Registered(block) => block.device(),
        }
    }
    /// Synchronous host write. Device memory is written by a driver copy
    /// on the non-blocking transfer stream, ordered only by the caller's
    /// access fence; host memory is written in place.
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(range) => range.upload_at(offset, bytes),
            Memory::Block(block) => block.upload_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.write_at(offset, bytes);
                Ok(())
            }
            Memory::HostNode(block) => {
                assert_within(offset, bytes.len(), block.bytes(), "upload_at");
                // SAFETY: within the block, which the host maps.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), block.host().as_ptr().add(offset), bytes.len()) };
                Ok(())
            }
            Memory::Registered(block) => {
                assert_within(offset, bytes.len(), block.bytes(), "upload_at");
                // SAFETY: within the block, which the host maps.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), block.host().as_ptr().add(offset), bytes.len()) };
                Ok(())
            }
        }
    }
    pub(crate) fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(range) => range.download_at(offset, bytes),
            Memory::Block(block) => block.download_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.read_at(offset, bytes);
                Ok(())
            }
            Memory::HostNode(block) => {
                assert_within(offset, bytes.len(), block.bytes(), "download_at");
                // SAFETY: within the block, which the host maps.
                unsafe { std::ptr::copy_nonoverlapping(block.host().as_ptr().add(offset), bytes.as_mut_ptr(), bytes.len()) };
                Ok(())
            }
            Memory::Registered(block) => {
                assert_within(offset, bytes.len(), block.bytes(), "download_at");
                // SAFETY: within the block, which the host maps.
                unsafe { std::ptr::copy_nonoverlapping(block.host().as_ptr().add(offset), bytes.as_mut_ptr(), bytes.len()) };
                Ok(())
            }
        }
    }
}

/// This crate's own precondition: host access stays within the block.
fn assert_within(offset: usize, length: usize, bytes: usize, operation: &str) {
    assert!(
        offset.checked_add(length).is_some_and(|end| end <= bytes),
        "Buffer::{operation} precondition: [{offset}, {offset}+{length}) exceeds {bytes} bytes"
    );
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
