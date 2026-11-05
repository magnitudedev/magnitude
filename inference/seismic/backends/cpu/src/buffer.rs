//! Aligned host storage for one global allocation.
//!
//! A `Buffer` is one allocation of the host heap at the profile's
//! allocation alignment. Kernels receive its address through the launch
//! frame; the runtime reads and writes it through `DeviceService`. The
//! runtime's submission discipline is single-threaded per device: no host
//! access overlaps a launch that binds the buffer, which is what makes the
//! raw address a valid kernel operand.

use seismic_reservation::{AllocationFailure, HostRange, PageSize, Pages, Reserved, Resizer};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::Arc;

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

enum Allocation {
    Heap(HeapAllocation),
    /// The committed prefix of a host reservation, resized in place by its
    /// [`Resizer`].
    Reserved(Arc<Reserved<Pages>>),
    /// Host memory another owner holds ([`Buffer::borrowed`]).
    Borrowed(BorrowedRegion),
}

/// `len` bytes at `pointer` that `_owner` keeps alive.
struct BorrowedRegion {
    pointer: NonNull<u8>,
    len: u64,
    _owner: Arc<dyn std::any::Any + Send + Sync>,
}

// The bytes are the owner's, which is `Send + Sync`; every access goes
// through the raw address under the submission discipline documented at the
// module level, as for the heap's.
unsafe impl Send for BorrowedRegion {}
unsafe impl Sync for BorrowedRegion {}

struct HeapAllocation {
    pointer: NonNull<u8>,
    /// The layout used to allocate and release the physical backing.
    physical_layout: Layout,
    /// The logical byte range exposed to the runtime and transfer methods.
    logical_bytes: u64,
}

// The allocation is plain bytes owned by this struct; sharing it across
// threads is sound because every access goes through the raw address under
// the submission discipline documented at the module level.
unsafe impl Send for HeapAllocation {}
unsafe impl Sync for HeapAllocation {}

impl Drop for HeapAllocation {
    fn drop(&mut self) {
        // `pointer` came from `alloc_zeroed(self.physical_layout)` in
        // `WritableBuffer::new`; physical storage is always non-zero-sized. The
        // allocator may reuse these pages, so they are unwired first.
        #[cfg(target_os = "macos")]
        unsafe {
            libc::munlock(self.pointer.as_ptr().cast(), self.physical_layout.size());
        }
        unsafe { dealloc(self.pointer.as_ptr(), self.physical_layout) };
    }
}

/// One host allocation, shared by handle: what device work binds and the
/// host reads. The host writes only a [`WritableBuffer`], the form of the
/// memory this crate allocates; a buffer over borrowed read-only memory
/// ([`Buffer::borrowed`]) is never one, so it has no host write path.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<Allocation>,
}

/// A [`Buffer`] over memory this crate allocated, which the host also
/// writes.
#[derive(Clone)]
pub struct WritableBuffer(Buffer);

impl WritableBuffer {
    /// Zeroed memory over a host address reservation of `reserved` bytes in
    /// `granule`-byte granules, with the leading `committed` bytes backed,
    /// and its resizer, which changes the backed prefix in place.
    pub fn reserved(
        committed: u64,
        reserved: u64,
        granule: u64,
    ) -> Result<(Self, Resizer<Pages>), AllocationFailure> {
        let range = HostRange::reserve(reserved.max(committed), granule, PageSize::Host)?;
        let (reservation, resizer) = Reserved::new(Pages(range), committed)?;
        let buffer = Buffer {
            inner: Arc::new(Allocation::Reserved(reservation)),
        };
        Ok((Self(buffer), resizer))
    }

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
        Ok(Self(Buffer {
            inner: Arc::new(Allocation::Heap(HeapAllocation {
                pointer,
                physical_layout,
                logical_bytes: bytes,
            })),
        }))
    }

    /// Copies `bytes` into the buffer at `offset`. The range lies inside the
    /// allocation: the runtime sizes every transfer from the schema, so an
    /// out-of-range transfer is a violated precondition of this wrapper
    /// (§13.3.3).
    pub fn write(&self, offset: u64, bytes: &[u8]) {
        let range = self.range(offset, bytes.len());
        // In-range by `range`; the source is a separate host slice.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data_pointer().add(range), bytes.len())
        };
    }
}

impl std::ops::Deref for WritableBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        &self.0
    }
}

impl std::borrow::Borrow<Buffer> for WritableBuffer {
    fn borrow(&self) -> &Buffer {
        &self.0
    }
}

impl Buffer {
    /// A buffer over `len` bytes of host memory at `pointer`, which `owner`
    /// holds: no bytes are copied or allocated, and the host only reads
    /// them.
    ///
    /// # Safety
    /// The bytes are readable for as long as `owner` lives, and nothing
    /// writes the ones a launch binding the buffer reads while it runs.
    pub unsafe fn borrowed(pointer: NonNull<u8>, len: u64, owner: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            inner: Arc::new(Allocation::Borrowed(BorrowedRegion { pointer, len, _owner: owner })),
        }
    }

    /// Logical bytes: a reserved buffer's committed bytes.
    pub fn len(&self) -> u64 {
        match &*self.inner {
            Allocation::Heap(heap) => heap.logical_bytes,
            Allocation::Reserved(reservation) => reservation.state().bytes,
            Allocation::Borrowed(region) => region.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host address of byte 0. Valid while any handle is alive.
    pub fn data_pointer(&self) -> *mut u8 {
        match &*self.inner {
            Allocation::Heap(heap) => heap.pointer.as_ptr(),
            Allocation::Reserved(reservation) => reservation.backing().0.base().as_ptr(),
            Allocation::Borrowed(region) => region.pointer.as_ptr(),
        }
    }

    /// Copies bytes out of the buffer at `offset` (same precondition as
    /// [`WritableBuffer::write`]).
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
    use super::{Buffer, WritableBuffer};

    #[test]
    fn zero_logical_bytes_have_live_aligned_backing() {
        let buffer = WritableBuffer::new(0, 64).expect("empty buffer");
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
        let buffer = WritableBuffer::new(page as u64 + 1, 64).expect("wired buffer");
        assert_eq!(buffer.len(), page as u64 + 1);
        assert_eq!(buffer.data_pointer() as usize % page, 0);
        let super::Allocation::Heap(heap) = &*buffer.inner else {
            unreachable!("WritableBuffer::new allocates from the heap")
        };
        assert_eq!(heap.physical_layout.size(), 2 * page);
        // The second page is the buffer's own: it reads as allocated zeros.
        let mut tail = [1];
        buffer.read(page as u64, &mut tail);
        assert_eq!(tail, [0]);
    }

    /// A borrowed buffer reads its owner's bytes in place; it has no host
    /// write path, which only a [`WritableBuffer`] offers.
    #[test]
    fn a_borrowed_buffer_reads_its_owners_bytes() {
        let owner = std::sync::Arc::new((0..16u8).collect::<Vec<_>>());
        let pointer = std::ptr::NonNull::new(owner.as_ptr().cast_mut()).unwrap();
        let buffer = unsafe { Buffer::borrowed(pointer, 16, owner.clone()) };
        assert_eq!(buffer.len(), 16);
        assert_eq!(buffer.data_pointer().cast_const(), owner.as_ptr());
        let mut bytes = [0; 4];
        buffer.read(12, &mut bytes);
        assert_eq!(bytes, [12, 13, 14, 15]);
    }

    #[test]
    fn empty_transfers_are_bounded_by_logical_length() {
        let buffer = WritableBuffer::new(0, 64).expect("empty buffer");
        buffer.write(0, &[]);
        let mut bytes = [];
        buffer.read(0, &mut bytes);
    }

    /// The process's physical footprint (macOS: `phys_footprint`, Linux: RSS).
    fn footprint() -> u64 {
        #[cfg(target_os = "macos")]
        unsafe {
            let mut usage = std::mem::zeroed::<libc::rusage_info_v2>();
            let status = libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                (&raw mut usage).cast(),
            );
            assert_eq!(status, 0);
            usage.ri_phys_footprint
        }
        #[cfg(not(target_os = "macos"))]
        {
            let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
            let pages: u64 = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
            pages * unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64
        }
    }

    /// A reserved buffer resizes in place: the address is fixed, kept bytes
    /// survive, bytes backed again read zero (also those a shrink left in a
    /// kept page), a shrink's pages stay until its release and then return
    /// to the system at once, and an abandoned growth backs nothing.
    #[test]
    fn a_reserved_buffer_resizes_in_place_and_releases_what_it_shrinks() {
        const MIB: u64 = 1 << 20;
        let page = seismic_reservation::host_page().unwrap();
        let (buffer, resizer) = WritableBuffer::reserved(256 * MIB, 4096 * MIB, page).expect("reservation");
        assert!(matches!(*buffer.inner, super::Allocation::Reserved(_)));
        let address = buffer.data_pointer();
        assert_eq!(buffer.len(), 256 * MIB);
        let pattern = |length: u64| (0..length).map(|index| (index % 251) as u8 + 1).collect::<Vec<_>>();
        buffer.write(0, &pattern(256 * MIB));

        let prepared = |resizer: super::Resizer<super::Pages>, bytes: u64| {
            let Ok(prepared) = resizer.prepare(bytes) else {
                panic!("a {bytes}-byte resize is refused");
            };
            prepared
        };
        let before = footprint();
        let resizer = prepared(resizer, 1024 * MIB).abandon();
        assert!(footprint().abs_diff(before) < 8 * MIB, "an abandoned growth backs nothing");
        let pending = prepared(resizer, 64 * MIB + 3).install();
        assert_eq!(buffer.len(), 64 * MIB + 3);
        assert!(footprint().abs_diff(before) < 8 * MIB, "an installed shrink releases nothing yet");
        let resizer = pending.release();
        let released = before.saturating_sub(footprint());
        eprintln!("a 192 MiB shrink lowered the footprint by {released} bytes");
        assert_eq!(buffer.data_pointer(), address);
        // The footprint falls by the released pages (other threads of the
        // test process may move it by a few MiB).
        assert!(
            released.abs_diff(192 * MIB) < 8 * MIB,
            "a 192 MiB shrink released {released} bytes"
        );
        let mut kept = vec![0; (64 * MIB + 3) as usize];
        buffer.read(0, &mut kept);
        assert!(kept == pattern(64 * MIB + 3), "kept bytes survive a shrink");

        let _resizer = prepared(resizer, 128 * MIB).install().release();
        assert_eq!(buffer.data_pointer(), address);
        let mut grown = vec![1; (128 * MIB) as usize];
        buffer.read(0, &mut grown);
        assert!(grown[..(64 * MIB + 3) as usize] == pattern(64 * MIB + 3)[..]);
        assert!(
            grown[(64 * MIB + 3) as usize..].iter().all(|byte| *byte == 0),
            "bytes backed again read zero"
        );
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
