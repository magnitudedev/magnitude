//! Aligned host storage for one global allocation.
//!
//! A `Buffer` is one allocation of the host heap at the profile's
//! allocation alignment. Kernels receive its address through the launch
//! frame; the runtime reads and writes it through `DeviceService`. The
//! runtime's submission discipline is single-threaded per device: no host
//! access overlaps a launch that binds the buffer, which is what makes the
//! raw address a valid kernel operand.

use std::alloc::{Layout, alloc_zeroed, dealloc};
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
    /// The kernel refused to wire the allocation's pages.
    #[cfg(target_os = "macos")]
    WireRefused,
}

impl std::fmt::Display for AllocationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unrepresentable => {
                f.write_str("CPU allocation size is not representable on the host")
            }
            Self::OutOfMemory => f.write_str("CPU allocation failed"),
            #[cfg(target_os = "macos")]
            Self::WireRefused => f.write_str("CPU allocation could not be wired"),
        }
    }
}

/// On macOS every host buffer is wired, so the kernel never compresses or
/// swaps the engine's holdings. Wiring is per page: a wired buffer owns
/// whole pages, which no other allocation shares.
#[cfg(target_os = "macos")]
fn physical_layout(size: usize, align: usize) -> Result<Layout, AllocationFailure> {
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .map_err(|_| AllocationFailure::Unrepresentable)?;
    let size = size
        .max(1)
        .checked_next_multiple_of(page)
        .ok_or(AllocationFailure::Unrepresentable)?;
    Layout::from_size_align(size, align.max(page)).map_err(|_| AllocationFailure::Unrepresentable)
}

#[cfg(not(target_os = "macos"))]
fn physical_layout(size: usize, align: usize) -> Result<Layout, AllocationFailure> {
    Layout::from_size_align(size.max(1), align).map_err(|_| AllocationFailure::Unrepresentable)
}

struct Allocation {
    pointer: NonNull<u8>,
    /// The layout used to allocate and release the physical backing.
    physical_layout: Layout,
    /// The logical byte range exposed to the runtime and transfer methods.
    logical_bytes: u64,
}

// The allocation is plain bytes owned by this struct; sharing it across
// threads is sound because every access goes through the raw address under
// the submission discipline documented at the module level.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        // `pointer` came from `alloc_zeroed(self.physical_layout)` in
        // `Buffer::new`; physical storage is always non-zero-sized. The
        // allocator may reuse these pages, so they are unwired first.
        #[cfg(target_os = "macos")]
        unsafe {
            libc::munlock(self.pointer.as_ptr().cast(), self.physical_layout.size());
        }
        unsafe { dealloc(self.pointer.as_ptr(), self.physical_layout) };
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
        // A zero logical allocation still needs a live, aligned address for
        // typed native views. Keep that physical minimum private: the runtime
        // and transfer bounds continue to observe `logical_bytes`.
        let physical_layout = physical_layout(size, align)?;
        let raw = unsafe { alloc_zeroed(physical_layout) };
        let pointer = NonNull::new(raw).ok_or(AllocationFailure::OutOfMemory)?;
        #[cfg(target_os = "macos")]
        if unsafe { libc::mlock(raw.cast(), physical_layout.size()) } != 0 {
            unsafe { dealloc(raw, physical_layout) };
            return Err(AllocationFailure::WireRefused);
        }
        Ok(Self {
            inner: Arc::new(Allocation {
                pointer,
                physical_layout,
                logical_bytes: bytes,
            }),
        })
    }

    pub fn len(&self) -> u64 {
        self.inner.logical_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.inner.logical_bytes == 0
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

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn zero_logical_bytes_have_live_aligned_backing() {
        let buffer = Buffer::new(0, 64).expect("empty buffer");
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        let pointer = buffer.data_pointer() as usize;
        assert_ne!(pointer, 0);
        assert_eq!(pointer % 64, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_wired_buffer_owns_whole_pages() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let buffer = Buffer::new(page as u64 + 1, 64).expect("wired buffer");
        assert_eq!(buffer.len(), page as u64 + 1);
        assert_eq!(buffer.data_pointer() as usize % page, 0);
        assert_eq!(buffer.inner.physical_layout.size(), 2 * page);
        // The second page is the buffer's own: it reads as allocated zeros.
        let mut tail = [1];
        buffer.read(page as u64, &mut tail);
        assert_eq!(tail, [0]);
    }

    #[test]
    fn empty_transfers_are_bounded_by_logical_length() {
        let buffer = Buffer::new(0, 64).expect("empty buffer");
        buffer.write(0, &[]);
        let mut bytes = [];
        buffer.read(0, &mut bytes);
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
