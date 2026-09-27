//! Aligned host storage for one global allocation.
//!
//! A `Buffer` is one allocation of the host heap at the profile's
//! allocation alignment. Kernels receive its address through the launch
//! frame; the runtime reads and writes it through `DeviceService`. The
//! runtime's submission discipline is single-threaded per device: no host
//! access overlaps a launch that binds the buffer, which is what makes the
//! raw address a valid kernel operand.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// Why a host allocation could not be made: the only real failure of a
/// host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationFailure {
    /// The size and alignment do not form a valid host layout.
    Unrepresentable,
    /// The host allocator refused.
    OutOfMemory,
}

impl std::fmt::Display for AllocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unrepresentable => {
                f.write_str("CPU allocation size is not representable on the host")
            }
            Self::OutOfMemory => f.write_str("CPU allocation failed"),
        }
    }
}

struct Allocation {
    pointer: NonNull<u8>,
    layout: Layout,
}

// The allocation is plain bytes owned by this struct; sharing it across
// threads is sound because every access goes through the raw address under
// the submission discipline documented at the module level.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // `pointer` came from `alloc_zeroed(self.layout)` in `Buffer::new`.
            unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
        }
    }
}

/// One host allocation, shared by handle.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<Allocation>,
}

impl Buffer {
    /// A zeroed allocation of `bytes` at `alignment` (a power of two).
    pub fn new(bytes: u64, alignment: u64) -> Result<Self, AllocationFailure> {
        let size = usize::try_from(bytes).map_err(|_| AllocationFailure::Unrepresentable)?;
        let align = usize::try_from(alignment).map_err(|_| AllocationFailure::Unrepresentable)?;
        let layout =
            Layout::from_size_align(size, align).map_err(|_| AllocationFailure::Unrepresentable)?;
        let pointer = if size == 0 {
            NonNull::<u8>::dangling()
        } else {
            // A non-zero-sized layout, as `alloc_zeroed` requires.
            let raw = unsafe { alloc_zeroed(layout) };
            NonNull::new(raw).ok_or(AllocationFailure::OutOfMemory)?
        };
        Ok(Self {
            inner: Arc::new(Allocation { pointer, layout }),
        })
    }

    pub fn len(&self) -> u64 {
        self.inner.layout.size() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.inner.layout.size() == 0
    }

    /// The host address of byte 0. Valid while any handle is alive.
    pub fn data_pointer(&self) -> *mut u8 {
        self.inner.pointer.as_ptr()
    }

    /// Copies `bytes` into the buffer at `offset`. The range lies inside the
    /// allocation: the runtime sizes every transfer from the schema, so an
    /// out-of-range transfer is a violated precondition of this wrapper
    /// (§13.3.3).
    pub fn write(&self, offset: u64, bytes: &[u8]) {
        let range = self.range(offset, bytes.len());
        // In-range by `range`; the source is a separate host slice.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.data_pointer().add(range),
                bytes.len(),
            )
        };
    }

    /// Copies bytes out of the buffer at `offset` (same precondition as
    /// `write`).
    pub fn read(&self, offset: u64, into: &mut [u8]) {
        let range = self.range(offset, into.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data_pointer().add(range),
                into.as_mut_ptr(),
                into.len(),
            )
        };
    }

    fn range(&self, offset: u64, len: usize) -> usize {
        let end = offset.checked_add(len as u64);
        match end {
            Some(end) if end <= self.len() => offset as usize,
            _ => panic!(
                "Buffer precondition violated: transfer of {len} bytes at offset {offset} exceeds the {}-byte allocation",
                self.len()
            ),
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
