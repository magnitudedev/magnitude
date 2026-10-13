//! Block sub-allocation shared by the backends that own their device memory
//! (CUDA, Vulkan). Device memory is taken from the driver in blocks, and an
//! allocation is a range of a block, so a model's many small tensors cost
//! no per-allocation driver rounding. Blocks grow from 64 MiB to the cap; a
//! request beyond the cap gets a dedicated block. A block's unallocated
//! space is device usage no allocation charges, so a new block never exceeds
//! the room the caller grants, and the backend reports its blocks' free
//! space as held memory.

use crate::errors::ExecutionError;

pub const FIRST_BLOCK_BYTES: u64 = 64 << 20;
pub const BLOCK_CAP_BYTES: u64 = 1 << 30;

/// The free ranges `(offset, bytes)` of one block, sorted and coalesced.
#[derive(Debug)]
pub struct FreeRanges {
    bytes: u64,
    free: Vec<(u64, u64)>,
}

impl FreeRanges {
    pub fn new(bytes: u64) -> Self {
        Self {
            bytes,
            free: vec![(0, bytes)],
        }
    }

    /// The offset of a range of `bytes` aligned to `alignment`, first fit.
    pub fn take(&mut self, bytes: u64, alignment: u64) -> Option<u64> {
        let (index, start) = self
            .free
            .iter()
            .enumerate()
            .find_map(|(index, (offset, size))| {
                let start = offset.next_multiple_of(alignment);
                (start + bytes <= offset + size).then_some((index, start))
            })?;
        let (offset, size) = self.free.remove(index);
        if start + bytes < offset + size {
            self.free
                .insert(index, (start + bytes, offset + size - start - bytes));
        }
        if start > offset {
            self.free.insert(index, (offset, start - offset));
        }
        Some(start)
    }

    /// Return a range; true when the block is entirely free afterwards.
    pub fn give(&mut self, offset: u64, bytes: u64) -> bool {
        let free = &mut self.free;
        let index = free.partition_point(|(start, _)| *start < offset);
        free.insert(index, (offset, bytes));
        if index + 1 < free.len() && free[index].0 + free[index].1 == free[index + 1].0 {
            free[index].1 += free.remove(index + 1).1;
        }
        if index > 0 && free[index - 1].0 + free[index - 1].1 == free[index].0 {
            free[index - 1].1 += free.remove(index).1;
        }
        free.len() == 1 && free[0] == (0, self.bytes)
    }

    /// Unallocated bytes of the block.
    pub fn free_bytes(&self) -> u64 {
        self.free.iter().map(|(_, bytes)| bytes).sum()
    }
}

/// The size of a new block for an allocation of `bytes`: double the largest
/// block (64 MiB first) up to `cap`, in whole `granularity` units; an
/// allocation larger than that gets a block of its own size rounded up to
/// the granularity. When `room` is given and the doubled block does not fit
/// it, the block is the allocation's own size: a device near its limit holds
/// no free block space it was not asked for, so its usage stays with its
/// claims and clear of the planning reserve. An allocation that needs more
/// than the room is a capacity shortfall.
pub fn block_bytes(
    largest: Option<u64>,
    cap: u64,
    bytes: u64,
    room: Option<u64>,
    granularity: u64,
) -> Result<u64, ExecutionError> {
    let needed = bytes.next_multiple_of(granularity);
    let size = largest
        .map_or(FIRST_BLOCK_BYTES, |largest| largest * 2)
        .min(cap)
        .max(needed);
    match room {
        Some(room) if needed > room => Err(ExecutionError::AllocationCapacity {
            required: needed.into(),
            available: room,
        }),
        Some(room) if size > room => Ok(needed),
        Some(_) => Ok(size),
        None => Ok(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn ranges_are_taken_aligned_and_coalesce_when_returned() {
        let mut ranges = FreeRanges::new(64);
        assert_eq!(ranges.take(10, 16), Some(0));
        assert_eq!(ranges.take(10, 16), Some(16));
        assert_eq!(ranges.free_bytes(), 44);
        assert!(!ranges.give(0, 10));
        assert!(ranges.give(16, 10));
        assert_eq!(ranges.take(65, 1), None);
    }

    /// A new block never grows device usage past the room granted: near the
    /// limit it holds only its allocation (in whole granules), and an
    /// allocation beyond the room is refused with the room as available. A
    /// request beyond the cap takes its own size.
    #[test]
    fn new_blocks_stay_within_the_granted_room() {
        let cap = 1024 * MIB;
        let g = 2 * MIB;
        assert_eq!(
            block_bytes(None, cap, MIB, None, g).unwrap(),
            FIRST_BLOCK_BYTES
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, MIB, None, g).unwrap(),
            cap
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, MIB, Some(4096 * MIB), g).unwrap(),
            cap
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 300 * MIB, Some(401 * MIB), g).unwrap(),
            300 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 3 * MIB, Some(401 * MIB), g).unwrap(),
            4 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 400 * MIB, Some(400 * MIB), g).unwrap(),
            400 * MIB
        );
        assert_eq!(
            block_bytes(None, cap, 1500 * MIB + 1, None, g).unwrap(),
            1502 * MIB
        );
        assert!(matches!(
            block_bytes(Some(512 * MIB), cap, 401 * MIB, Some(400 * MIB), g),
            Err(ExecutionError::AllocationCapacity { available, .. }) if available == 400 * MIB
        ));
    }
}
