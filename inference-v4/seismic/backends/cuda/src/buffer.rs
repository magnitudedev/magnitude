//! CUDA storage exposed through the generic device service: device memory, or
//! pinned host memory mapped into the device address space.

use crate::driver::{Allocation, DriverError, HostMapped};
use std::sync::Arc;

#[derive(Clone)]
pub struct Buffer {
    memory: Arc<Memory>,
}

enum Memory {
    Device(Allocation),
    /// Host-written inputs: filled by the host without a driver call.
    Mapped(HostMapped),
}

impl Buffer {
    pub(crate) fn device(allocation: Allocation) -> Self {
        Self {
            memory: Arc::new(Memory::Device(allocation)),
        }
    }
    pub(crate) fn mapped(mapped: HostMapped) -> Self {
        Self {
            memory: Arc::new(Memory::Mapped(mapped)),
        }
    }
    pub fn len(&self) -> u64 {
        match &*self.memory {
            Memory::Device(allocation) => allocation.bytes as u64,
            Memory::Mapped(mapped) => mapped.bytes as u64,
        }
    }
    /// The address kernels use.
    pub fn pointer(&self) -> u64 {
        match &*self.memory {
            Memory::Device(allocation) => allocation.pointer,
            Memory::Mapped(mapped) => mapped.device,
        }
    }
    /// Synchronous host write. Device memory is written by a driver copy,
    /// which is ordered after all previously queued device work; mapped
    /// memory is written in place.
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(allocation) => allocation.upload_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.write_at(offset, bytes);
                Ok(())
            }
        }
    }
    /// Synchronous host read.
    pub(crate) fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(allocation) => allocation.download_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.read_at(offset, bytes);
                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
