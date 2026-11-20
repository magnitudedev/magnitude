//! Device storage as ranges of blocks (`seismic_compiler::blocks`): each
//! block is one `cuMemAlloc` of whole allocation granules, so a model's many
//! tensors cost no per-allocation driver rounding (under WDDM every
//! `cuMemAlloc` takes whole 2 MiB pages). A block is freed once it holds no
//! range. Each block is charged whole to the device store's own entry, and
//! each range is carved from it for the entry its allocation names: a
//! block's free space is the device's own bytes.

use crate::driver::{self, Allocation, Context, DriverError};
use seismic_compiler::blocks::{
    charge_block, first_fit, largest_shared, takes_own_block, FreeRanges, PackedBlock,
    BLOCK_CAP_BYTES,
};
use seismic_compiler::errors::ExecutionError;
use seismic_memory::entry::{Backed, CarvableCharge, Carved, Entry, Wiring};
use std::sync::{Arc, Mutex, MutexGuard};

/// Every range starts at least this aligned.
const MINIMUM_ALIGNMENT: u64 = 256;

struct Block {
    /// The driver's allocation with its bytes, on the device store's own
    /// entry, which its ranges are carved from.
    backed: Backed<Allocation, CarvableCharge>,
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

impl PackedBlock for Block {
    fn bytes(&self) -> u64 {
        self.backed.bytes as u64
    }
    fn own(&self) -> bool {
        self.dedicated
    }
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        self.ranges().take(bytes, alignment)
    }
}

/// One device's blocks.
pub(crate) struct Blocks {
    context: Arc<Context>,
    /// What every block is charged to: the device store's own entry.
    own: Arc<dyn Entry>,
    granularity: u64,
    blocks: Mutex<Vec<Arc<Block>>>,
}

impl Blocks {
    pub(crate) fn new(context: Arc<Context>, own: Arc<dyn Entry>) -> Result<Self, DriverError> {
        Ok(Self {
            granularity: driver::allocation_granularity(&context)?,
            context,
            own,
            blocks: Mutex::new(Vec::new()),
        })
    }

    fn blocks(&self) -> MutexGuard<'_, Vec<Arc<Block>>> {
        self.blocks
            .lock()
            .expect("block list lock is never poisoned")
    }

    /// A range of `bytes`, its bytes carved for `entry` from its block's.
    /// A new block is charged to the device's own entry before the driver
    /// allocates it ([`charge_block`]).
    pub(crate) fn allocate(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Range, ExecutionError> {
        let bytes = bytes.max(1);
        let alignment = alignment.max(MINIMUM_ALIGNMENT);
        let mut blocks = self.blocks();
        let found = first_fit(&blocks, bytes, alignment)
            .map(|(index, offset)| (blocks[index].clone(), offset));
        let (block, offset) = match found {
            Some(found) => found,
            None => {
                // The charge, the driver's allocation and its record are one
                // change.
                let change = self.own.change();
                let charge = charge_block(&change, &self.own, largest_shared(&blocks), BLOCK_CAP_BYTES, bytes, self.granularity)?;
                let backed = Backed::allocate(&change, charge, |charge| {
                    let size = charge.bytes();
                    Allocation::new(
                        &self.context,
                        usize::try_from(size).map_err(|_| {
                            ExecutionError::AllocationFailed(
                                "CUDA allocation exceeds host address space".into(),
                            )
                        })?,
                    )
                    .map_err(crate::executor::allocation_error(size, || crate::executor::device_free(&self.context)))
                })?
                .into_carvable();
                let size = backed.charge().bytes();
                let block = Arc::new(Block {
                    backed,
                    dedicated: takes_own_block(bytes, BLOCK_CAP_BYTES),
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
            carved: Some(block.backed.charge().carve(entry, bytes)),
            blocks: self.clone(),
            block,
            offset,
            bytes,
        })
    }

    /// One block of its own ([`crate::Device::allocate_block`]): at least
    /// `bytes`, whole allocation granules, its bytes as `fill` says
    /// (`SlabFill::Unwritten` skips synchronous zeroing so allocation never
    /// synchronizes the transfer stream), charged exactly to `entry` and freed
    /// as it drops, never shared with a range.
    pub(crate) fn allocate_block(&self, bytes: u64, entry: &Arc<dyn Entry>, fill: seismic_memory::SlabFill) -> Result<DeviceBlock, ExecutionError> {
        let size = bytes
            .max(1)
            .checked_next_multiple_of(self.granularity)
            .ok_or_else(|| ExecutionError::AllocationFailed("CUDA block size overflows".into()))?;
        let host = usize::try_from(size)
            .map_err(|_| ExecutionError::AllocationFailed("CUDA allocation exceeds host address space".into()))?;
        let refused = || crate::executor::allocation_error(size, || crate::executor::device_free(&self.context));
        let backed = Backed::take(entry, size, Wiring::Pageable, |_| {
            let allocation = Allocation::new(&self.context, host).map_err(refused())?;
            if fill == seismic_memory::SlabFill::Zeroed {
                driver::zero(&self.context, allocation.pointer, host).map_err(refused())?;
            }
            Ok::<_, ExecutionError>(allocation)
        })?;
        Ok(DeviceBlock { backed })
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
    /// Its bytes, carved for its allocation's entry from the block's;
    /// returned to the block before the range becomes allocatable again.
    carved: Option<Carved>,
    blocks: Arc<Blocks>,
    block: Arc<Block>,
    offset: u64,
    bytes: u64,
}

impl Range {
    pub(crate) fn pointer(&self) -> u64 {
        self.block.backed.pointer + self.offset
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
        assert_within(offset, length, self.bytes, &format!("Range::{operation}"));
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        drop(self.carved.take());
        self.blocks.release(self);
    }
}

/// A device block of its own ([`Blocks::allocate_block`]), freed as it
/// drops, its exact bytes charged to its entry while it lives.
pub(crate) struct DeviceBlock {
    backed: Backed<Allocation>,
}

impl DeviceBlock {
    pub(crate) fn pointer(&self) -> u64 {
        self.backed.pointer
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.backed.bytes as u64
    }

    /// Precondition (this crate's own): the bytes lie within the block.
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        assert_within(offset, bytes.len(), self.bytes(), "DeviceBlock::upload_at");
        driver::upload(&self.backed.context, self.pointer() + offset as u64, bytes)
    }

    pub(crate) fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        assert_within(offset, bytes.len(), self.bytes(), "DeviceBlock::download_at");
        driver::download(&self.backed.context, self.pointer() + offset as u64, bytes)
    }
}

/// This crate's own precondition: host access stays within `limit` bytes.
fn assert_within(offset: usize, length: usize, limit: u64, operation: &str) {
    assert!(
        (offset as u64)
            .checked_add(length as u64)
            .is_some_and(|end| end <= limit),
        "{operation} precondition: [{offset}, {offset}+{length}) exceeds {limit} bytes"
    );
}
