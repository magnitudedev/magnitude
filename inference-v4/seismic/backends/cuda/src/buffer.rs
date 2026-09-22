//! CUDA device allocation exposed through the generic device service.

use crate::driver::Allocation;
use std::sync::Arc;

#[derive(Clone)]
pub struct Buffer {
    pub(crate) allocation: Arc<Allocation>,
}

impl Buffer {
    pub(crate) fn new(allocation: Allocation) -> Self {
        Self {
            allocation: Arc::new(allocation),
        }
    }
    pub fn len(&self) -> u64 {
        self.allocation.bytes as u64
    }
    pub(crate) fn pointer(&self) -> u64 {
        self.allocation.pointer
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
