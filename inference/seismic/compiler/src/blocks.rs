//! Block sub-allocation shared by the backends that own their device memory
//! (CUDA, Vulkan). Device memory is taken from the driver in blocks, and an
//! allocation is a range of a block, so a model's many small tensors cost
//! no per-allocation driver rounding. Blocks grow from 64 MiB to the cap; a
//! request of more than half the first block gets a block of its own. A
//! block is charged whole to its backend's own ledger entry
//! ([`charge_block`]), and each range is carved from it for the entry the
//! allocation names, so a block's free space is the backend's own bytes.

use crate::errors::ExecutionError;
use seismic_memory::entry::{Change, Charge, Entry, Wiring};
use std::sync::Arc;

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
/// without a device. Every new shared block is the doubled one, as where
/// the backend's own entry admits it: the most free space the sequence
/// leaves.
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
            self.granularity,
        );
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
/// to the granularity.
pub fn block_bytes(largest: Option<u64>, cap: u64, bytes: u64, granularity: u64) -> u64 {
    let needed = bytes.next_multiple_of(granularity);
    let doubled = largest
        .map_or(FIRST_BLOCK_BYTES, |largest| largest * 2)
        .min(cap);
    if takes_own_block(bytes, cap) || needed > doubled / 2 {
        needed
    } else {
        doubled
    }
}

/// The charge of a new block for an allocation of `bytes`, taken on `own`
/// (the backend's own entry, which a block's free space is) before the
/// driver allocates it, in the `change` its caller holds open
/// ([`Entry::change`]) until the block is allocated and backed by the
/// charge ([`seismic_memory::entry::Backed::allocate`]): the block [`block_bytes`] sizes where `own` admits
/// it, else only the allocation's whole granules, which leave the block no
/// free space; refused when `own` admits not even those. The block's size
/// is the charge's bytes. Blocks are a discrete device's memory (CUDA's,
/// Vulkan's), which the host OS never wires: pageable.
pub fn charge_block(
    change: &Change,
    own: &Arc<dyn Entry>,
    largest: Option<u64>,
    cap: u64,
    bytes: u64,
    granularity: u64,
) -> Result<Charge, ExecutionError> {
    let size = block_bytes(largest, cap, bytes, granularity);
    let needed = bytes.next_multiple_of(granularity);
    Charge::take(change, own, size, Wiring::Pageable)
        .or_else(|refused| match size > needed {
            true => Charge::take(change, own, needed, Wiring::Pageable),
            false => Err(refused),
        })
        .map_err(ExecutionError::from)
}

/// Like [`charge_block`], but charges the first allocation's `bytes` directly
/// to `entry` (drawing on its already-committed claim) and only the uncarved
/// block remainder `size - bytes` to `own`. Returns `(remainder, first)`.
pub fn charge_block_for(
    change: &Change,
    own: &Arc<dyn Entry>,
    entry: &Arc<dyn Entry>,
    largest: Option<u64>,
    cap: u64,
    bytes: u64,
    granularity: u64,
) -> Result<(Charge, Charge), ExecutionError> {
    let first = Charge::take(change, entry, bytes, Wiring::Pageable).map_err(ExecutionError::from)?;
    let size = block_bytes(largest, cap, bytes, granularity);
    let needed = bytes.next_multiple_of(granularity);
    let take_slack = |total: u64| {
        let slack = total - bytes;
        if slack == 0 {
            Ok(Charge::took(change, own, 0, Wiring::Pageable))
        } else {
            Charge::take(change, own, slack, Wiring::Pageable)
        }
    };
    let mut candidate = size;
    let remainder = loop {
        match take_slack(candidate) {
            Ok(charge) => break charge,
            Err(_) if candidate > needed => {
                // Step down by half (whole granules) so a tight store still
                // opens a shared block that fits its room instead of falling
                // straight to per-tensor dedicated granules.
                let halved = (candidate / 2).next_multiple_of(granularity).max(needed);
                candidate = if halved < candidate { halved } else { needed };
            }
            Err(_) => {
                // Mandatory driver granularity rounding (`needed - bytes < granularity`)
                // for an already-admitted `first` allocation is block slack: record it
                // on `own` rather than refusing the admitted allocation.
                break Charge::took(change, own, needed - bytes, Wiring::Pageable);
            }
        }
    };
    Ok((remainder, first))
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

    /// A new shared block doubles the largest up to the cap; a request
    /// beyond the cap takes its own size in whole granules.
    #[test]
    fn new_blocks_double_up_to_the_cap() {
        let cap = 1024 * MIB;
        let g = 2 * MIB;
        assert_eq!(block_bytes(None, cap, MIB, g), FIRST_BLOCK_BYTES);
        assert_eq!(block_bytes(Some(512 * MIB), cap, MIB, g), cap);
        assert_eq!(block_bytes(Some(512 * MIB), cap, 300 * MIB, g), 300 * MIB);
        assert_eq!(block_bytes(None, cap, 1500 * MIB + 1, g), 1502 * MIB);
    }

    /// A new block never takes more than the backend's own entry admits:
    /// near its limit the block holds only its allocation (in whole
    /// granules), and an allocation beyond it is refused with what the
    /// entry had, nothing charged.
    #[test]
    fn a_new_block_is_charged_within_what_the_entry_admits() {
        let cap = 1024 * MIB;
        let g = 2 * MIB;
        let identity = seismic_topology::StoreIdentity::HostRam;
        let tally = seismic_memory::entry::Tally::limited(identity, 401 * MIB);
        let own: Arc<dyn Entry> = Arc::new(tally.clone());
        let block = charge_block(&own.change(), &own, Some(512 * MIB), cap, 3 * MIB, g).unwrap();
        assert_eq!(block.bytes(), 4 * MIB);
        assert_eq!(tally.held(), 4 * MIB);
        drop(block);
        assert!(matches!(
            charge_block(&own.change(), &own, Some(512 * MIB), cap, 402 * MIB, g),
            Err(ExecutionError::AllocationCapacity { available, .. }) if available == 401 * MIB
        ));
        assert_eq!(tally.held(), 0);
        tally.bound(u64::MAX);
        assert_eq!(charge_block(&own.change(), &own, Some(512 * MIB), cap, MIB, g).unwrap().bytes(), cap);
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
            block_bytes(Some(512 * MIB), cap, 756 * MIB + 1, g),
            758 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 343 * MIB + 1, g),
            344 * MIB
        );
        assert_eq!(
            block_bytes(Some(512 * MIB), cap, 32 * MIB, g),
            cap
        );
        assert_eq!(block_bytes(None, cap, 40 * MIB, g), 40 * MIB);
        assert!(takes_own_block(32 * MIB + 1, cap));
        assert!(!takes_own_block(32 * MIB, cap));
        assert!(takes_own_block(9 * MIB, 16 * MIB));
    }

    /// Under a tight store, `charge_block_for` charges `bytes` directly to
    /// `entry`, steps the shared block down by half to the largest power-of-two
    /// multiple of `granularity` whose slack `own` admits, and never refuses an
    /// admitted `first` allocation for mandatory granularity rounding.
    #[test]
    fn charge_block_for_steps_down_shared_block_and_tolerates_granularity_rounding() {
        let cap = 1024 * MIB;
        let g = 2 * MIB;
        let identity = seismic_topology::StoreIdentity::HostRam;
        let own_tally = seismic_memory::entry::Tally::limited(identity.clone(), 70 * MIB);
        let entry_tally = seismic_memory::entry::Tally::limited(identity, u64::MAX);
        let own: Arc<dyn Entry> = Arc::new(own_tally.clone());
        let entry: Arc<dyn Entry> = Arc::new(entry_tally.clone());

        // 70 MiB free on `own`: doubled 128 MiB is refused, halved 64 MiB fits.
        let (remainder, first) =
            charge_block_for(&own.change(), &own, &entry, Some(64 * MIB), cap, MIB / 4, g).unwrap();
        assert_eq!(first.bytes(), MIB / 4);
        assert_eq!(remainder.bytes(), 64 * MIB - MIB / 4);
        assert_eq!(entry_tally.held(), MIB / 4);
        assert_eq!(own_tally.held(), 64 * MIB - MIB / 4);
        drop((remainder, first));

        // 0 free on `own`: mandatory granularity rounding still succeeds via `took`.
        own_tally.bound(0);
        let (remainder, first) =
            charge_block_for(&own.change(), &own, &entry, Some(64 * MIB), cap, MIB / 4, g).unwrap();
        assert_eq!(first.bytes(), MIB / 4);
        assert_eq!(remainder.bytes(), g - MIB / 4);
    }
}
