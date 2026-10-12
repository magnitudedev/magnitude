//! Device storage as ranges of blocks (`seismic_compiler::blocks`): each
//! block is one `cuMemAlloc` of whole allocation granules, so a model's many
//! tensors cost no per-allocation driver rounding (under WDDM every
//! `cuMemAlloc` takes whole 2 MiB pages). A block is freed once it holds no
//! range; its unallocated space is held device memory the caller counts
//! with its charges.

use crate::driver::{self, Allocation, Context, DriverError};
use seismic_compiler::blocks::{block_bytes, FreeRanges, BLOCK_CAP_BYTES};
use seismic_compiler::errors::ExecutionError;
use std::sync::{Arc, Mutex, MutexGuard};

/// Every range starts at least this aligned.
const MINIMUM_ALIGNMENT: u64 = 256;

struct Block {
    allocation: Allocation,
    dedicated: bool,
    free: Mutex<FreeRanges>,
}

impl Block {
    fn ranges(&self) -> MutexGuard<'_, FreeRanges> {
        self.free
            .lock()
            .expect("block free list lock is never poisoned")
    }
}

/// One device's blocks.
pub(crate) struct Blocks {
    context: Arc<Context>,
    granularity: u64,
    blocks: Mutex<Vec<Arc<Block>>>,
}

impl Blocks {
    pub(crate) fn new(context: Arc<Context>) -> Self {
        Self {
            granularity: driver::allocation_granularity(&context),
            context,
            blocks: Mutex::new(Vec::new()),
        }
    }

    fn blocks(&self) -> MutexGuard<'_, Vec<Arc<Block>>> {
        self.blocks
            .lock()
            .expect("block list lock is never poisoned")
    }

    /// Unallocated space of every block.
    pub(crate) fn unallocated_bytes(&self) -> u64 {
        self.blocks()
            .iter()
            .map(|block| block.ranges().free_bytes())
            .sum()
    }

    /// A range of `bytes`. `room`, when given, is the device memory the
    /// caller may add, its blocks' free space already counted: a new block
    /// is sized within it, and an allocation that needs more is refused as a
    /// capacity shortfall.
    pub(crate) fn allocate(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        room: Option<u64>,
    ) -> Result<Range, ExecutionError> {
        let bytes = bytes.max(1);
        let alignment = alignment.max(MINIMUM_ALIGNMENT);
        let mut blocks = self.blocks();
        let found = blocks
            .iter()
            .filter(|block| !block.dedicated)
            .find_map(|block| {
                block
                    .ranges()
                    .take(bytes, alignment)
                    .map(|offset| (block.clone(), offset))
            });
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                let largest = blocks
                    .iter()
                    .filter(|block| !block.dedicated)
                    .map(|block| block.allocation.bytes as u64)
                    .max();
                let size = block_bytes(largest, BLOCK_CAP_BYTES, bytes, room, self.granularity)?;
                let allocation = Allocation::new(
                    &self.context,
                    usize::try_from(size).map_err(|_| {
                        ExecutionError::AllocationFailed(
                            "CUDA allocation exceeds host address space".into(),
                        )
                    })?,
                )
                .map_err(crate::executor::allocation_error)?;
                let block = Arc::new(Block {
                    allocation,
                    dedicated: bytes > BLOCK_CAP_BYTES,
                    free: Mutex::new(FreeRanges::new(size)),
                });
                let offset = block
                    .ranges()
                    .take(bytes, alignment)
                    .expect("a new block holds its first allocation");
                blocks.push(block.clone());
                (block, offset)
            }
        };
        Ok(Range {
            blocks: self.clone(),
            block,
            offset,
            bytes,
        })
    }

    /// Return a range, and free its block once it holds none.
    fn release(&self, range: &Range) {
        let mut blocks = self.blocks();
        if range.block.ranges().give(range.offset, range.bytes) {
            let index = blocks
                .iter()
                .position(|block| Arc::ptr_eq(block, &range.block))
                .expect("a live range's block belongs to its device");
            blocks.remove(index);
        }
    }
}

/// A range of a block, returned on drop.
pub(crate) struct Range {
    blocks: Arc<Blocks>,
    block: Arc<Block>,
    offset: u64,
    bytes: u64,
}

impl Range {
    pub(crate) fn pointer(&self) -> u64 {
        self.block.allocation.pointer + self.offset
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Precondition (this crate's own): the bytes lie within the range.
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        self.assert_within(offset, bytes.len(), "upload_at");
        driver::upload(&self.blocks.context, self.pointer() + offset as u64, bytes)
    }

    pub(crate) fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        self.assert_within(offset, bytes.len(), "download_at");
        driver::download(&self.blocks.context, self.pointer() + offset as u64, bytes)
    }

    fn assert_within(&self, offset: usize, length: usize, operation: &str) {
        assert!(
            (offset as u64)
                .checked_add(length as u64)
                .is_some_and(|end| end <= self.bytes),
            "Range::{operation} precondition: [{offset}, {offset}+{length}) exceeds {} bytes",
            self.bytes
        );
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        self.blocks.release(self);
    }
}
