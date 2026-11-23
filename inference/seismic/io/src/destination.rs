//! What a read lands in.

use std::ptr::NonNull;

/// Memory a read lands in: a stable region of bytes, whose alignment is its
/// base address's. A request holds each of its ranges' destinations
/// ([`crate::Range`]) until its last piece has ended, whatever becomes of
/// its handle, so a read into memory already freed has no expression.
///
/// # Safety
/// [`region`](Self::region) names the same bytes for the value's whole life,
/// and they are valid for writes while it lives.
pub unsafe trait Destination: Send + Sync {
    /// The region's first byte and its length.
    fn region(&self) -> (NonNull<u8>, usize);
}

/// Staging headroom needed beside any payload length so an arbitrary
/// unaligned file range `[offset, offset + length)` can land in an
/// [`crate::ALIGNMENT`]-aligned direct read without a bounce copy: up to
/// `ALIGNMENT - 1` leading bytes before `offset` and `ALIGNMENT - 1`
/// trailing bytes after `offset + length`.
pub const ALIGNED_HEADROOM: usize = 2 * crate::ALIGNMENT as usize;

/// A heap destination: `length` bytes whose first is [`crate::ALIGNMENT`]-aligned.
pub struct Buffer {
    base: NonNull<u8>,
    layout: std::alloc::Layout,
    length: usize,
}

// SAFETY: it owns its allocation; access to its bytes is governed by `&`/`&mut`
// and, for reads in flight, by [`Buffer::bytes`]'s contract.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(length: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(length.max(1), crate::ALIGNMENT as usize).expect("a buffer's layout");
        // SAFETY: the layout's size is nonzero.
        let base = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { base, layout, length }
    }

    /// A heap destination with [`ALIGNED_HEADROOM`] extra bytes beyond
    /// `payload_length`, so any unaligned range of `payload_length` bytes
    /// lands directly via [`crate::AlignedRange`] with zero bounce copy.
    pub fn with_headroom(payload_length: usize) -> Self {
        Self::new(payload_length.checked_add(ALIGNED_HEADROOM).expect("buffer length with headroom overflow"))
    }

    /// Its length in bytes.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Its bytes.
    ///
    /// # Safety
    /// No request that reads into it is in flight: each has ended
    /// ([`crate::Handle::wait`]).
    pub unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: `length` initialized bytes it owns, written by no read now.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr(), self.length) }
    }

    /// Its bytes, held alone.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `length` initialized bytes it owns, borrowed alone.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), self.length) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.base.as_ptr(), self.layout) }
    }
}

// SAFETY: its allocation never moves or changes length while it lives.
unsafe impl Destination for Buffer {
    fn region(&self) -> (NonNull<u8>, usize) {
        (self.base, self.length)
    }
}
