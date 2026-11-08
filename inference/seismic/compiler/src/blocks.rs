//! Block sub-allocation shared by the backends that own their device memory
//! (CUDA, Vulkan). Device memory is taken from the driver in blocks, and an
//! allocation is a range of a block, so a model's many small tensors cost
//! no per-allocation driver rounding. Blocks grow from 64 MiB to the cap; a
//! request of more than half the first block gets a block of its own. A block's unallocated
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

/// A block as the packing rule sees it: its size, whether it is an
/// allocation's own (`takes_own_block`), and its free ranges. The backends'
/// blocks and a replay's (`Packing`) are placed by the same rule.
pub trait PackedBlock {
    fn bytes(&self) -> u64;
    fn own(&self) -> bool;
    /// The offset of a range taken from the block's free ranges, first fit.
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64>;
}

impl<B: PackedBlock> PackedBlock for std::sync::Arc<B> {
    fn bytes(&self) -> u64 {
        (**self).bytes()
    }
    fn own(&self) -> bool {
        (**self).own()
    }
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        (**self).take(bytes, alignment)
    }
}

/// Where an allocation goes among `blocks`, in order: the first shared block
/// with a free range that fits it, and the range's offset.
pub fn first_fit<B: PackedBlock>(blocks: &[B], bytes: u64, alignment: u64) -> Option<(usize, u64)> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| !block.own())
        .find_map(|(index, block)| block.take(bytes, alignment).map(|offset| (index, offset)))
}

/// The largest shared block, the one a new shared block doubles.
pub fn largest_shared<B: PackedBlock>(blocks: &[B]) -> Option<u64> {
    blocks
        .iter()
        .filter(|block| !block.own())
        .map(PackedBlock::bytes)
        .max()
}

/// A replay of the packing rule over a sequence of allocations, holding
/// only the blocks' sizes and free ranges: the device memory a backend's
/// blocks hold for those allocations, and what of it no allocation uses,
/// without a device. No room is granted, so every new shared block is the
/// doubled one: the most free space the sequence leaves.
pub struct Packing {
    cap: u64,
    granularity: u64,
    minimum_alignment: u64,
    blocks: Vec<ReplayedBlock>,
}

struct ReplayedBlock {
    bytes: u64,
    own: bool,
    free: std::cell::RefCell<FreeRanges>,
}

impl PackedBlock for ReplayedBlock {
    fn bytes(&self) -> u64 {
        self.bytes
    }
    fn own(&self) -> bool {
        self.own
    }
    fn take(&self, bytes: u64, alignment: u64) -> Option<u64> {
        self.free.borrow_mut().take(bytes, alignment)
    }
}

impl Packing {
    /// The blocks of a backend whose blocks are at most `cap`, in whole
    /// `granularity` units, with ranges at least `minimum_alignment` aligned.
    pub fn new(cap: u64, granularity: u64, minimum_alignment: u64) -> Self {
        Self {
            cap,
            granularity,
            minimum_alignment,
            blocks: Vec::new(),
        }
    }

    /// Place an allocation of `bytes` aligned to `alignment`, as the
    /// backend's blocks would.
    pub fn allocate(&mut self, bytes: u64, alignment: u64) -> Result<(), ExecutionError> {
        let bytes = bytes.max(1);
        let alignment = alignment.max(self.minimum_alignment);
        if first_fit(&self.blocks, bytes, alignment).is_some() {
            return Ok(());
        }
        let size = block_bytes(
            largest_shared(&self.blocks),
            self.cap,
            bytes,
            None,
            self.granularity,
        )?;
        let mut free = FreeRanges::new(size);
        free.take(bytes, alignment)
            .expect("a new block holds its first allocation");
        self.blocks.push(ReplayedBlock {
            bytes: size,
            own: takes_own_block(bytes, self.cap),
            free: std::cell::RefCell::new(free),
        });
        Ok(())
    }

    /// The device memory the blocks hold.
    pub fn held_bytes(&self) -> u64 {
        self.blocks.iter().map(|block| block.bytes).sum()
    }

    /// What of the blocks no allocation uses.
    pub fn free_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| block.free.borrow().free_bytes())
            .sum()
    }

    /// What of the blocks no later allocation can use: the free space of
    /// the blocks allocations hold as their own, which `first_fit` never
    /// offers again. A shared block's free space is where the next
    /// allocation that fits it goes.
    pub fn stranded_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter(|block| block.own)
            .map(|block| block.free.borrow().free_bytes())
            .sum()
    }
}

/// Whether an allocation of `bytes` takes a block of its own, which nothing
/// else shares and no later block doubles: one of more than half the first
/// block (or of the cap). Sharing doubled blocks, a run of such allocations
/// (a model's expert stacks) would hold a tail of each block that fits none
/// of them and that the model's few smaller tensors never fill: 343 MiB
/// stacks two to a 1 GiB block held 5.0 GB free beside Nemotron 3.5
/// Lightning's 20.5 GB, and 450 MiB stacks 5.4 GB beside Qwen3.8 Flash
/// Next's 73.5 GB, against a one-block allowance.
pub fn takes_own_block(bytes: u64, cap: u64) -> bool {
    bytes > FIRST_BLOCK_BYTES.min(cap) / 2
}

/// The size of a new block for an allocation of `bytes`: double the largest
/// shared block (64 MiB first) up to `cap`, in whole `granularity` units,
/// so many smaller allocations cost one driver allocation. An allocation
/// that takes its own block (`takes_own_block`) gets its own size rounded up
/// to the granularity. When `room` is given and the doubled block does not
/// fit it, the block is the allocation's own size: a device near its limit
/// holds no free block space it was not asked for, so its usage stays with
/// its claims and clear of the planning reserve. An allocation that needs
/// more than the room is a capacity shortfall.
pub fn block_bytes(
    largest: Option<u64>,
    cap: u64,
    bytes: u64,
    room: Option<u64>,
    granularity: u64,
) -> Result<u64, ExecutionError> {
    let needed = bytes.next_multiple_of(granularity);
    let doubled = largest
        .map_or(FIRST_BLOCK_BYTES, |largest| largest * 2)
        .min(cap);
    let size = if takes_own_block(bytes, cap) || needed > doubled / 2 {
        needed
    } else {
        doubled
    };
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

    /// A replay holds what the backends' blocks would: a run of expert stacks
    /// each in a block of its own, the granule beyond each its only free
    /// space, and small tensors sharing a doubled block.
    #[test]
    fn a_replay_holds_each_stack_in_its_own_block_and_shares_the_small() {
        let g = 2 * MIB;
        let mut packing = Packing::new(BLOCK_CAP_BYTES, g, 256);
        for _ in 0..46 {
            packing.allocate(343 * MIB + MIB / 2, 256).unwrap();
        }
        assert_eq!(packing.held_bytes(), 46 * 344 * MIB);
        assert_eq!(packing.free_bytes(), 46 * (MIB / 2));
        assert_eq!(packing.stranded_bytes(), 46 * (MIB / 2));
        for _ in 0..100 {
            packing.allocate(MIB / 4, 256).unwrap();
        }
        assert_eq!(packing.held_bytes(), 46 * 344 * MIB + FIRST_BLOCK_BYTES);
        assert_eq!(
            packing.free_bytes(),
            46 * (MIB / 2) + FIRST_BLOCK_BYTES - 25 * MIB
        );
        // The shared block's tail is the next small allocation's.
        assert_eq!(packing.stranded_bytes(), 46 * (MIB / 2));
    }

    /// An allocation of more than half the first block takes a block of its
    /// own size, however large the shared blocks have grown: its tail would
    /// fit nothing its size. Half or less shares the doubled block.
    #[test]
    fn an_allocation_beyond_half_the_first_block_takes_its_own_size() {
        let cap = 1024 * MIB;
        let g = 2 * MIB;
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 756 * MIB + 1, None, g).unwrap(),
            758 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 343 * MIB + 1, None, g).unwrap(),
            344 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 32 * MIB, None, g).unwrap(),
            cap
        );
        assert_eq!(block_bytes(None, cap, 40 * MIB, None, g).unwrap(), 40 * MIB);
        assert!(takes_own_block(32 * MIB + 1, cap));
        assert!(!takes_own_block(32 * MIB, cap));
        assert!(takes_own_block(9 * MIB, 16 * MIB));
    }
}
