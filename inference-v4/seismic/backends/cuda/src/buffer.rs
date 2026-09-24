//! CUDA storage exposed through the generic device service: device memory,
//! pinned host memory mapped into the device address space, or the backed
//! prefix of a reserved device address range.

use crate::driver::{self, Allocation, DriverError, HostMapped, Reservation};
use std::sync::Arc;

#[derive(Clone)]
pub struct Buffer {
    memory: Arc<Memory>,
}

enum Memory {
    Device(Allocation),
    /// Host-written inputs: filled by the host without a driver call.
    Mapped(HostMapped),
    /// The leading `bytes` of a reservation, which are backed. Buffers of
    /// one reservation share its base address.
    Reserved {
        reservation: Arc<Reservation>,
        bytes: usize,
    },
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
    pub(crate) fn reserved(reservation: Arc<Reservation>, bytes: usize) -> Self {
        Self {
            memory: Arc::new(Memory::Reserved { reservation, bytes }),
        }
    }
    /// Whether this buffer is the backed prefix of a reservation, which
    /// `Device::recommit` resizes in place.
    pub fn is_reserved(&self) -> bool {
        self.reservation().is_some()
    }
    /// The reservation this buffer is a backed prefix of.
    pub(crate) fn reservation(&self) -> Option<&Arc<Reservation>> {
        match &*self.memory {
            Memory::Reserved { reservation, .. } => Some(reservation),
            Memory::Device(_) | Memory::Mapped(_) => None,
        }
    }
    pub fn len(&self) -> u64 {
        match &*self.memory {
            Memory::Device(allocation) => allocation.bytes as u64,
            Memory::Mapped(mapped) => mapped.bytes as u64,
            Memory::Reserved { bytes, .. } => *bytes as u64,
        }
    }
    /// The address kernels use.
    pub fn pointer(&self) -> u64 {
        match &*self.memory {
            Memory::Device(allocation) => allocation.pointer,
            Memory::Mapped(mapped) => mapped.device,
            Memory::Reserved { reservation, .. } => reservation.base,
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
            Memory::Reserved {
                reservation,
                bytes: backed,
            } => {
                assert_within(offset, bytes.len(), *backed, "upload_at");
                driver::upload(reservation.context(), reservation.base + offset as u64, bytes)
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
            Memory::Reserved {
                reservation,
                bytes: backed,
            } => {
                assert_within(offset, bytes.len(), *backed, "download_at");
                driver::download(reservation.context(), reservation.base + offset as u64, bytes)
            }
        }
    }
}

/// This crate's own precondition: host access stays within the backed bytes.
fn assert_within(offset: usize, length: usize, backed: usize, operation: &str) {
    assert!(
        offset.checked_add(length).is_some_and(|end| end <= backed),
        "Buffer::{operation} precondition: [{offset}, {offset}+{length}) exceeds {backed} backed bytes"
    );
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
