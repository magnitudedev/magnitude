//! Aligned, thread-affine host storage with retained byte views.
use std::{cell::RefCell, rc::Rc};

/// Why a host buffer operation failed. The runtime boundary maps these
/// into `ExecutionFailure::External` (allocation) or rejects the request
/// before submission (range/in-use facts of the caller's binding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferError {
    /// The requested size is not representable as a host allocation.
    SizeOverflow,
    /// The host allocation failed.
    Allocation,
    /// A view or transfer lies outside its parent view's bytes.
    OutOfRange,
    /// The backing storage is already borrowed for another operation.
    InUse,
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("CPU allocation size overflow"),
            Self::Allocation => f.write_str("CPU allocation failed"),
            Self::OutOfRange => f.write_str("CPU view or transfer exceeds its parent"),
            Self::InUse => f.write_str("CPU storage is already in use"),
        }
    }
}

impl std::error::Error for BufferError {}

#[derive(Clone)]
pub struct Buffer {
    pub(crate) storage: Rc<RefCell<Vec<u64>>>,
    pub(crate) offset: usize,
    len: usize,
}
impl Buffer {
    pub fn new(bytes: usize) -> Result<Self, BufferError> {
        let words = bytes
            .checked_add(7)
            .ok_or(BufferError::SizeOverflow)?
            / 8;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(words)
            .map_err(|_| BufferError::Allocation)?;
        storage.resize(words, 0);
        Ok(Self {
            storage: Rc::new(RefCell::new(storage)),
            offset: 0,
            len: bytes,
        })
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BufferError> {
        let buffer = Self::new(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, BufferError> {
        if range.start > range.end || range.end > self.len {
            return Err(BufferError::OutOfRange);
        }
        Ok(Self {
            storage: self.storage.clone(),
            offset: self
                .offset
                .checked_add(range.start)
                .ok_or(BufferError::SizeOverflow)?,
            len: range.end - range.start,
        })
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), BufferError> {
        if bytes.len() > self.len {
            return Err(BufferError::OutOfRange);
        }
        let mut storage = self
            .storage
            .try_borrow_mut()
            .map_err(|_| BufferError::InUse)?;
        // The checked view lies within the word allocation. No host slices into
        // that private allocation escape, so the input cannot alias it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                storage.as_mut_ptr().cast::<u8>().add(self.offset),
                bytes.len(),
            );
        }
        Ok(())
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), BufferError> {
        if bytes.len() > self.len {
            return Err(BufferError::OutOfRange);
        }
        let storage = self
            .storage
            .try_borrow()
            .map_err(|_| BufferError::InUse)?;
        // u64 has no invalid bit patterns. Reading initialized allocation bytes
        // is valid, and the caller cannot hold a slice into this private owner.
        unsafe {
            std::ptr::copy_nonoverlapping(
                storage.as_ptr().cast::<u8>().add(self.offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(())
    }
    /// The host pointer of this view's bytes. The Rc allocation outlives the
    /// view; no growth occurs while the kernel borrows it. Public so the
    /// runtime crate's CPU adapter can extract validated buffer pointers.
    ///
    /// The single-threaded borrow discipline of submission guarantees no
    /// overlapping borrow at this point; the caller owns that invariant.
    pub fn data_pointer(&self) -> *mut u8 {
        let mut storage = self
            .storage
            .try_borrow_mut()
            .expect("CPU storage is already in use");
        unsafe { storage.as_mut_ptr().cast::<u8>().add(self.offset) }
    }
}
