//! A store's granule pool on a device: the storage slab stores
//! ([`crate::SlabTensor::in_pool`]) take their slabs from, one granule each.

use crate::{Charges, Contract, Device, TensorError};

pub use seismic_runtime::pool::{block_bytes, BlockSize, BlockSizing, NativeAllocations, SizingError, BLOCK_ALIGNMENT};

/// One store's granule pool on a device: blocks of the backend's plain
/// allocations, each charged to the pool's ledger entry, cut into granules
/// that slabs claim and release in process, beside the absent granule every
/// free slot names, charged to an entry of its own ([`PoolCharges`]).
/// Clones name the same pool.
#[derive(Clone)]
pub struct Pool {
    inner: seismic_runtime::pool::Pool,
}

/// The charges a pool's memory is made through, of one device: its blocks
/// through one, its absent granule through the other.
#[derive(Clone)]
pub struct PoolCharges {
    inner: seismic_runtime::pool::PoolCharges,
}

impl PoolCharges {
    /// Every byte of the pool made through `charges`: a tally's or a
    /// fixture's, which counts what the pool holds without telling its
    /// granules apart.
    pub fn one(charges: &Charges) -> Self {
        Self { inner: seismic_runtime::pool::PoolCharges::one(charges.inner()) }
    }

    /// The pool's blocks made through `blocks`, its absent granule through
    /// `absent`: a ledger's, which holds the absent granule apart from the
    /// granules a claim can take.
    ///
    /// # Panics
    /// Where they are charges of two devices.
    pub fn apart(blocks: &Charges, absent: &Charges) -> Self {
        Self { inner: seismic_runtime::pool::PoolCharges::apart(blocks.inner().clone(), absent.inner().clone()) }
    }
}

impl Pool {
    /// The granule pool of the store `contract` places in on `charges`'
    /// device, of `granule`-byte granules (a multiple of 256) in
    /// `block_bytes`-byte blocks ([`block_bytes`]), made through `charges`
    /// as every allocation is (their entries of that store, the kind they
    /// are confined to): a new pool, holding no block until it grows, which
    /// its clones share. A load opens one per store it places in; a fixture
    /// opens its own beside it.
    ///
    /// # Panics
    /// Through charges that allocate nothing in that store
    /// ([`Charges::none`], [`Device::charges_in`] of another store).
    pub fn open(charges: &PoolCharges, contract: Contract, granule: u64, block_bytes: u64) -> Result<Pool, TensorError> {
        seismic_runtime::pool::Pool::open(&charges.inner, contract, granule, block_bytes).map(|inner| Pool { inner })
    }
}

impl Device {
    /// How the device's backend makes its native allocations
    /// ([`BlockSizing::allocations`]), what a pool's blocks are sized by.
    pub fn native_allocations(&self) -> NativeAllocations {
        seismic_runtime::pool::Pool::native_allocations(self.inner())
    }
}

impl Pool {
    pub fn granule(&self) -> u64 {
        self.inner.granule()
    }

    pub fn block_bytes(&self) -> u64 {
        self.inner.block_bytes()
    }

    /// The bytes the pool holds in its store, as charged to its entries.
    pub fn held_bytes(&self) -> u64 {
        self.inner.held_bytes()
    }

    /// Granules claimable now, in process.
    pub fn free_granules(&self) -> usize {
        self.inner.free_granules()
    }

    /// Allocate one block, admitted by the pool's entry first. Refused with
    /// the entry's capacity error, nothing changes.
    pub fn grow(&self) -> Result<(), TensorError> {
        self.inner.grow()
    }

    /// Return `freed` bytes, what a shrink of the pool's holders freed for
    /// outside demand: every empty block, then, while those fall short of
    /// it, the emptiest block emptied (its live slabs copied into free
    /// granules elsewhere, their stores' tables repointed) and returned,
    /// until they reach it or the emptiest block cannot be emptied (a view
    /// holds one of its slabs). An emptying waits for
    /// the pool's work in flight (at most one flight) and holds off new work
    /// on the pool while it copies. The bytes returned.
    pub fn return_freed(&self, freed: u64) -> Result<u64, TensorError> {
        self.inner.return_freed(freed)
    }

    pub(crate) fn inner(&self) -> &seismic_runtime::pool::Pool {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::{Availability, Element, SlabFill, SlabRegion, SlabTensor};

    const GRANULE: u64 = 1 << 20;

    /// What a shrink frees for outside demand returns to the OS on every
    /// backend, whole blocks first and, where they fall short, the emptiest
    /// block emptied: its live slab copied into a free granule elsewhere,
    /// its table repointed, its bytes intact, and the block returned, its
    /// charge with it. Two blocks each keep one live slab of their granules;
    /// the newer is emptied into the older, which keeps both.
    #[test]
    fn a_shortfall_empties_the_emptiest_block_on_every_backend() {
        let devices = crate::Devices::discover(&[]).unwrap();
        let topology = devices.topology();
        let available = topology
            .executors()
            .iter()
            .filter(|info| matches!(info.availability(), Availability::Available))
            .map(|info| info.id())
            .collect::<Vec<_>>();
        for id in available {
            let tallies = devices.tallies(id);
            let stores = tallies.iter().map(|tally| Arc::new(tally.clone()) as Arc<dyn crate::memory::Store>).collect();
            let device = Device { inner: devices.inner.open_own(id, Default::default(), stores).unwrap() };
            let store = topology.first_store(device.info(), Contract::DeviceOnly).unwrap();
            let identity = topology.store_identity(store).unwrap();
            let tally = tallies.into_iter().find(|tally| crate::memory::Store::identity(tally) == &identity).unwrap();
            // At least two granules a block, so a block can hold one live
            // slab beside a free one.
            let allocations = device.native_allocations();
            let allocations = NativeAllocations { least: allocations.least.max(2 * GRANULE), ..allocations };
            let block = block_bytes(BlockSizing { granule: GRANULE, allocations }).unwrap().bytes;
            let per_block = block / GRANULE;
            let charges = PoolCharges::one(&device.charges_in(Contract::DeviceOnly, Arc::new(tally.clone())));
            let pool = Pool::open(&charges, Contract::DeviceOnly, GRANULE, block).unwrap();
            let regions = [SlabRegion { element: Element::f32(), row_shape: vec![GRANULE / 4] }];
            let slabs = SlabTensor::in_pool(&crate::Charges::tally(&device), &pool, 1, 2 * per_block, &regions).unwrap();
            for _ in 0..2 * per_block {
                slabs.add_slab(SlabFill::Zeroed).unwrap();
            }
            let held = pool.held_bytes();
            let (older, newer) = ((per_block - 1) as usize, (2 * per_block - 1) as usize);
            let marker = |slot: usize| vec![slot as u8 + 1; GRANULE as usize];
            for slot in [older, newer] {
                slabs.region_rows(0, slot as u64, 1).unwrap().write_from_host(&marker(slot)).unwrap();
            }
            for slot in (0..2 * per_block as usize).filter(|slot| ![older, newer].contains(slot)) {
                assert!(slabs.free_slab(slot).unwrap());
            }
            let freed = (2 * per_block - 2) * GRANULE;
            assert_eq!(pool.return_freed(0).unwrap(), 0, "each block keeps a live slab");
            let charged = tally.held();
            let returned = pool.return_freed(freed).unwrap();
            assert!(returned >= block, "the newer block returned: {returned} of {block}");
            assert_eq!(pool.held_bytes(), held - returned, "the pool holds one block less");
            assert_eq!(tally.held(), charged - returned, "its charge left with it");
            for slot in [older, newer] {
                assert_eq!(slabs.region_rows(0, slot as u64, 1).unwrap().read_to_host().unwrap(), marker(slot), "slab {slot} kept its bytes");
            }
            assert_eq!(slabs.backed(), [older, newer]);
        }
    }
}
