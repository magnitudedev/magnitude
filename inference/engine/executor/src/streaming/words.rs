//! Words the host and running device work share (spec 4.3).
//!
//! A step's routed entries read and write these words while the host
//! watches them, so access goes through a tensor's host window
//! (`Tensor::host_window`), not the synchronized transfers. The host side of
//! the cross-agent contract is acquire loads and release stores with
//! sequentially consistent fences; the device side is system-coherent
//! accesses under system-scope fences (`metal/lib/streaming/control.h`).

use seismic::Tensor;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, Ordering, fence};

/// An `i32` tensor's words, addressed by the host while the device runs.
pub(crate) struct SharedWords {
    /// Keeps the storage alive while `base` is used.
    _storage: Tensor,
    base: NonNull<AtomicI32>,
    len: usize,
}

// The words are only touched through atomics; the tensor keeps them alive.
unsafe impl Send for SharedWords {}
unsafe impl Sync for SharedWords {}

impl SharedWords {
    pub(crate) fn new(tensor: Tensor) -> Result<Self, String> {
        let (address, bytes) = tensor.host_window().map_err(|error| error.to_string())?;
        if address.as_ptr() as usize % std::mem::align_of::<AtomicI32>() != 0 {
            return Err("shared words are not 4-byte aligned".into());
        }
        let len = usize::try_from(bytes / 4).map_err(|_| "shared words exceed the host")?;
        Ok(Self {
            _storage: tensor,
            base: address.cast(),
            len,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn word(&self, index: usize) -> &AtomicI32 {
        assert!(index < self.len, "shared word {index} of {}", self.len);
        // In bounds and aligned; the storage outlives `self`.
        unsafe { &*self.base.as_ptr().add(index) }
    }

    /// A word as the device last published it: an acquire load, so what
    /// the device published before it is visible after it. Polling takes
    /// no full fence per word (fences contend with the device for the
    /// memory system).
    pub(crate) fn load(&self, index: usize) -> i32 {
        self.word(index).load(Ordering::Acquire)
    }

    /// Publish a word to device work that may be running.
    pub(crate) fn store(&self, index: usize, value: i32) {
        self.word(index).store(value, Ordering::Release);
        fence(Ordering::SeqCst);
    }

    /// Fill every word before any device work reads them.
    pub(crate) fn fill(&self, value: i32) {
        for index in 0..self.len {
            self.word(index).store(value, Ordering::Relaxed);
        }
        fence(Ordering::SeqCst);
    }
}

/// A tensor's bytes, written by the host's reads while the device runs
/// (staging). The device reads them with system-coherent loads.
pub(crate) struct SharedBytes {
    /// Keeps the storage alive while `base` is used.
    _storage: Tensor,
    base: NonNull<u8>,
    len: u64,
}

unsafe impl Send for SharedBytes {}
unsafe impl Sync for SharedBytes {}

impl SharedBytes {
    pub(crate) fn new(tensor: Tensor) -> Result<Self, String> {
        let (base, len) = tensor.host_window().map_err(|error| error.to_string())?;
        Ok(Self {
            _storage: tensor,
            base,
            len,
        })
    }

    /// The host address of byte `offset`, with `bytes` bytes in bounds after it.
    pub(crate) fn at(&self, offset: u64, bytes: u64) -> *mut u8 {
        assert!(
            offset.checked_add(bytes).is_some_and(|end| end <= self.len),
            "staging bytes [{offset}, +{bytes}) exceed {}",
            self.len
        );
        // In bounds; the storage outlives `self`.
        unsafe { self.base.as_ptr().add(offset as usize) }
    }
}
