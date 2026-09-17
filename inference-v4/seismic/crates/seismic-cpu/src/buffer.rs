//! Aligned, thread-affine host storage with retained byte views.
use std::{cell::RefCell, rc::Rc};
#[derive(Clone)]
pub struct Buffer {
    pub(crate) storage: Rc<RefCell<Vec<u64>>>,
    pub(crate) offset: usize,
    len: usize,
}
impl Buffer {
    pub fn new(bytes: usize) -> Result<Self, String> {
        let words = bytes.checked_add(7).ok_or("CPU allocation size overflow")? / 8;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(words)
            .map_err(|e| format!("CPU allocation failed: {e}"))?;
        storage.resize(words, 0);
        Ok(Self {
            storage: Rc::new(RefCell::new(storage)),
            offset: 0,
            len: bytes,
        })
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
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
    pub fn view(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len {
            return Err("CPU buffer view exceeds its parent".into());
        }
        Ok(Self {
            storage: self.storage.clone(),
            offset: self
                .offset
                .checked_add(range.start)
                .ok_or("CPU view offset overflow")?,
            len: range.end - range.start,
        })
    }
    pub fn write(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > self.len {
            return Err("CPU write exceeds buffer view".into());
        }
        let mut storage = self
            .storage
            .try_borrow_mut()
            .map_err(|_| "CPU storage is already in use")?;
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
    pub fn read(&self, bytes: &mut [u8]) -> Result<(), String> {
        if bytes.len() > self.len {
            return Err("CPU read exceeds buffer view".into());
        }
        let storage = self
            .storage
            .try_borrow()
            .map_err(|_| "CPU storage is already in use")?;
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
}
