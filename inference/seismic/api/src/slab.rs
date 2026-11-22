//! Slab stores: logical tensors over fixed-size slabs, each slab one
//! granule of a store's pool ([`crate::Pool`]), bound through an address
//! table that device work reads as it runs.

use crate::{Charges, Element, Pool, Tensor, TensorError};
pub use seismic_runtime::pool::SlabFill;
use seismic_runtime::pool::{SlabGeometry, SlabStore};

/// One component region repeated for every row in a slab. `row_shape`
/// excludes the leading row axis.
#[derive(Clone, Debug)]
pub struct SlabRegion {
    pub element: Element,
    pub row_shape: Vec<u64>,
}

/// Device-independent physical layout of a slab store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabLayout {
    pub slab_bytes: u64,
    pub address_table_bytes: u64,
    /// The least multiple of every region's alignment in a slab: a slab
    /// whose storage starts at a multiple of it in its allocation lays its
    /// regions out aligned as one of its own allocation does.
    pub slab_alignment: u64,
}

impl SlabLayout {
    pub fn for_regions(
        rows_per_slab: u64,
        logical_rows: u64,
        regions: &[SlabRegion],
    ) -> Result<Self, TensorError> {
        let (bytes, _, capacity, slab_alignment) =
            layout_regions(rows_per_slab, logical_rows, regions)?;
        let region_count = u64::try_from(regions.len())
            .map_err(|_| TensorError::SlabLayout("region count exceeds u64".into()))?;
        let (address_table_bytes, _) =
            Tensor::canonical_layout(Element::u32(), &[region_count, capacity as u64, 2])?;
        Ok(Self {
            slab_bytes: bytes,
            address_table_bytes,
            slab_alignment,
        })
    }
}

/// A slab's bytes, each region's offset in it, the slots of the logical
/// rows, and the alignment every offset is a multiple of.
fn layout_regions(
    rows_per_slab: u64,
    logical_rows: u64,
    regions: &[SlabRegion],
) -> Result<(u64, Vec<u64>, usize, u64), TensorError> {
    if rows_per_slab == 0 || logical_rows == 0 || regions.is_empty() {
        return Err(TensorError::SlabLayout(
            "a slab needs rows, logical extent and at least one region".into(),
        ));
    }
    let capacity = usize::try_from(logical_rows.div_ceil(rows_per_slab))
        .map_err(|_| TensorError::SlabLayout("slab capacity exceeds host domain".into()))?;
    let mut offsets = Vec::with_capacity(regions.len());
    let mut bytes = 0u64;
    let mut slab_alignment = 1u64;
    for region in regions {
        if region.row_shape.contains(&0) {
            return Err(TensorError::SlabLayout(
                "region row dimensions must be positive".into(),
            ));
        }
        let extents = std::iter::once(rows_per_slab)
            .chain(region.row_shape.iter().copied())
            .collect::<Vec<_>>();
        let (length, alignment) = Tensor::canonical_layout(region.element, &extents)?;
        let alignment = alignment.max(seismic_runtime::pool::BLOCK_ALIGNMENT);
        // The least multiple of every region's alignment.
        let (mut a, mut b) = (slab_alignment, alignment);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        slab_alignment = slab_alignment / a * alignment;
        bytes = bytes
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or_else(|| TensorError::SlabLayout("slab offset overflows".into()))?;
        offsets.push(bytes);
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| TensorError::SlabLayout("slab size overflows".into()))?;
    }
    Ok((bytes, offsets, capacity, slab_alignment))
}

/// Slabs of fixed size backing a common logical row domain, each one granule
/// of a pool, claimed and released in process. Indices are stable while
/// backed, and the lowest free index is backed first. Each region's logical
/// tensor binds the region's address table, which stays valid as slabs are
/// added, freed and moved; a free slot names a zero slab of the pool, so a
/// masked load of it reads device memory of a whole slab.
pub struct SlabTensor {
    pool: Pool,
    store: SlabStore,
}

impl SlabTensor {
    /// A store of `regions`, `rows_per_slab` rows a slab over
    /// `logical_rows`, its slabs granules of `pool`, every slot free. Refused
    /// where a slab exceeds the pool's granule. Its address table is
    /// allocated through `charges`.
    pub fn in_pool(
        charges: &Charges,
        pool: &Pool,
        rows_per_slab: u64,
        logical_rows: u64,
        regions: &[SlabRegion],
    ) -> Result<Self, TensorError> {
        Self::in_pool_over(charges, pool, rows_per_slab, logical_rows, regions, Vec::new())
    }

    /// [`Self::in_pool`] whose leading slots are lent: `leading` holds one
    /// tensor per region, each `[lent * rows_per_slab, ...row_shape]` of its
    /// region's element, a contiguous canonical view of storage the caller
    /// made on the pool's device. Slots `0..lent` view those rows for the
    /// store's life and are never freed ([`Self::free_slab`] refuses them);
    /// device work bound to the store orders against that storage too.
    pub fn in_pool_over(
        charges: &Charges,
        pool: &Pool,
        rows_per_slab: u64,
        logical_rows: u64,
        regions: &[SlabRegion],
        leading: Vec<Tensor>,
    ) -> Result<Self, TensorError> {
        let (slab_bytes, offsets, capacity, slab_alignment) =
            layout_regions(rows_per_slab, logical_rows, regions)?;
        let geometry = SlabGeometry {
            rows_per_slab,
            logical_rows,
            capacity,
            slab_bytes,
            slab_alignment,
            regions: regions
                .iter()
                .zip(&offsets)
                .map(|(region, &offset)| (region.element.id(), region.row_shape.clone(), offset))
                .collect(),
        };
        let lent = leading.iter().map(|tensor| tensor.inner().clone()).collect::<Vec<_>>();
        let store = SlabStore::in_pool(charges.inner(), pool.inner(), geometry, &lent)?;
        Ok(Self { pool: pool.clone(), store })
    }

    /// The pool the store's slabs are granules of.
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn rows_per_slab(&self) -> u64 {
        self.store.geometry().rows_per_slab
    }

    pub fn logical_rows(&self) -> u64 {
        self.store.geometry().logical_rows
    }

    /// The bytes of one slab, its regions laid out within it.
    pub fn slab_bytes(&self) -> u64 {
        self.store.geometry().slab_bytes
    }

    pub fn capacity(&self) -> usize {
        self.store.geometry().capacity
    }

    pub fn region_count(&self) -> usize {
        self.store.geometry().regions.len()
    }

    pub fn region_offset(&self, index: usize) -> Option<u64> {
        self.store.geometry().regions.get(index).map(|(_, _, offset)| *offset)
    }

    /// The device bytes the store holds of its own: its address table. Its
    /// slabs are its pool's granules ([`Pool::held_bytes`]), its lent slabs
    /// their owner's.
    pub fn address_table_bytes(&self) -> u64 {
        self.capacity() as u64 * self.region_count() as u64 * 8
    }

    /// The leading slots whose slabs are lent ([`Self::in_pool_over`]).
    pub fn lent(&self) -> usize {
        self.store.lent()
    }

    /// The backed slots, in order.
    pub fn backed(&self) -> Vec<usize> {
        self.store.backed()
    }

    /// The number of backed slots, without allocating.
    pub fn backed_count(&self) -> usize {
        self.store.backed_count()
    }

    /// Whether slot `index` is backed by a slab, without creating a view.
    pub fn is_backed(&self, index: usize) -> bool {
        self.store.is_backed(index)
    }

    /// Back the lowest free slot with a granule of the pool, growing the
    /// pool by a block, admitted by its entry, where none is free: the slot.
    /// Refused where every slot is backed, or with the entry's capacity
    /// error where it refuses the block. Never waits on device work.
    pub fn add_slab(&self, fill: SlabFill) -> Result<usize, TensorError> {
        self.store.add_slab(fill)
    }

    /// Back the `count` lowest free slots with granules of the pool in one
    /// batch, growing the pool by blocks, admitted by its entry, where none
    /// are free: the slots in ascending order. Refused where fewer than
    /// `count` slots are free, or with the entry's capacity error where it
    /// refuses a block (on refusal, no slot is backed). Never waits on device
    /// work.
    pub fn add_slabs(&self, count: usize, fill: SlabFill) -> Result<Vec<usize>, TensorError> {
        self.store.add_slabs(count, fill)
    }

    /// Free slot `index`: its table entries name the pool's zero slab from
    /// now, and its granule returns to the pool once no view of it remains
    /// and the device work submitted by then completes, so work in flight
    /// reads it whole. Never waits on device work. False for a free slot;
    /// refused for a lent one.
    pub fn free_slab(&self, index: usize) -> Result<bool, TensorError> {
        self.store.free_slab(index)
    }

    /// Region `region`'s address table: `[capacity, 2]` u32, entry `i` the
    /// 64-bit device address of slot `i`'s rows of the region. A record port
    /// binds a one-row-a-slab region's records through it.
    pub fn address_table(&self, region: usize) -> Result<Tensor, TensorError> {
        self.store.table(region).map(Tensor::of_inner)
    }

    /// Region `region` of slot `index`'s slab, `[rows_per_slab,
    /// ...row_shape]`; none for a free slot. It holds the slab's bytes while
    /// it lives, freed or not.
    pub fn slab_region(&self, index: usize, region: usize) -> Result<Option<Tensor>, TensorError> {
        Ok(self.store.slab_region(index, region)?.map(Tensor::of_inner))
    }

    /// A view of rows `start..start + rows` of region `region`, which lie in
    /// one backed slab.
    pub fn region_rows(&self, region: usize, start: u64, rows: u64) -> Result<Tensor, TensorError> {
        let rows_per_slab = self.rows_per_slab();
        let end = start
            .checked_add(rows)
            .ok_or_else(|| TensorError::SlabLayout("span end overflows".into()))?;
        if rows == 0 || end > self.logical_rows() || (end - 1) / rows_per_slab != start / rows_per_slab {
            return Err(TensorError::SlabLayout("span must lie within one slab".into()));
        }
        let index = usize::try_from(start / rows_per_slab)
            .map_err(|_| TensorError::SlabLayout("slab index exceeds host domain".into()))?;
        let slab = self
            .slab_region(index, region)?
            .ok_or_else(|| TensorError::SlabLayout("span refers to a free slab".into()))?;
        let local = start % rows_per_slab;
        slab.slice_leading(local, local + rows)
    }

    /// The logical tensor of region `region`, `[logical_rows,
    /// ...row_shape]`. Its native binding is the region's address table,
    /// which device work reads as it runs: it stays valid as slabs are
    /// added, freed and moved.
    pub fn logical_region(&self, region: usize) -> Result<Tensor, TensorError> {
        self.store.logical_region(region).map(Tensor::of_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Availability, BackendName, Contract, Device, Devices};
    use std::sync::Arc;

    const GRANULE: u64 = 1 << 20;

    /// A DeviceOnly pool of `device`, of 1 MiB granules in 4 MiB blocks,
    /// charged to an unbounded entry of its store.
    fn pool(device: &Device) -> Pool {
        let topology = device.topology();
        let store = topology.first_store(device.info(), Contract::DeviceOnly).unwrap();
        let identity = topology.store_identity(store).unwrap();
        let entry: Arc<dyn crate::memory::Entry> = Arc::new(crate::memory::Tally::unbounded(identity));
        let charges = crate::PoolCharges::one(&device.charges_in(Contract::DeviceOnly, entry));
        Pool::open(&charges, Contract::DeviceOnly, GRANULE, 4 * GRANULE).unwrap()
    }

    fn cpu() -> Device {
        Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap()
    }

    fn available() -> Vec<Device> {
        let catalog = Devices::discover(&[]).unwrap();
        catalog
            .topology()
            .executors()
            .iter()
            .filter(|info| matches!(info.availability(), Availability::Available))
            .map(|info| catalog.open(info.id()).unwrap())
            .collect()
    }

    fn entry(tensor: &SlabTensor, region: usize, index: u64) -> u64 {
        let bytes = tensor
            .address_table(region)
            .unwrap()
            .slice_leading(index, index + 1)
            .unwrap()
            .read_to_host()
            .unwrap();
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    /// Slabs are granules named in the address table region by region; a
    /// freed slot names the pool's zero slab at once, beside live views of
    /// the store, and the lowest free slot is backed first.
    #[test]
    fn slabs_are_granules_and_the_lowest_free_slot_is_backed_first() {
        let device = cpu();
        let regions = [
            SlabRegion { element: Element::f32(), row_shape: vec![2] },
            SlabRegion { element: Element::f16(), row_shape: vec![3] },
        ];
        let layout = SlabLayout::for_regions(256, 768, &regions).unwrap();
        let tensor = SlabTensor::in_pool(&crate::Charges::tally(&device), &pool(&device), 256, 768, &regions).unwrap();
        assert_eq!(layout.slab_bytes, tensor.slab_bytes());
        assert_eq!(layout.address_table_bytes, tensor.address_table_bytes());
        let free = entry(&tensor, 0, 0);
        assert_eq!(entry(&tensor, 1, 0), free + tensor.region_offset(1).unwrap());
        let first = tensor.add_slab(SlabFill::Zeroed).unwrap();
        let second = tensor.add_slab(SlabFill::Zeroed).unwrap();
        assert_eq!((first, second), (0, 1));
        assert_eq!(tensor.backed(), [0, 1]);
        let slab = tensor.slab_region(first, 0).unwrap().unwrap();
        assert_eq!(slab.extents(), &[256, 2]);
        assert_eq!(entry(&tensor, 0, 0), slab.device_address());
        assert_eq!(
            entry(&tensor, 1, 0),
            tensor.slab_region(first, 1).unwrap().unwrap().device_address()
        );
        assert_eq!(
            entry(&tensor, 1, 0),
            slab.device_address() + tensor.region_offset(1).unwrap()
        );
        assert_eq!(slab.device_address() % 256, 0);
        let logical = tensor.logical_region(0).unwrap();
        let sibling = tensor.logical_region(1).unwrap();
        assert_eq!(logical.extents(), &[768, 2]);
        let reshaped = logical.reshape(&[768, 1, 2]).unwrap();
        assert_eq!(reshaped.extents(), &[768, 1, 2]);
        assert!(logical.reshape(&[384, 4]).is_err());
        assert_ne!(logical.descriptor().allocation, sibling.descriptor().allocation);
        // Each region's view describes its own logical tensor from 0, however
        // far into the address table its entries lie; whole-slab sub-views
        // place their range at `first * slab_bytes` in that region's space.
        for view in [&logical, &sibling] {
            let descriptor = view.descriptor();
            assert_eq!((descriptor.byte_offset, descriptor.byte_len), (0, view.byte_len()));
        }
        let first_two = logical.slice_leading(0, 512).unwrap();
        let last_one = logical.slice_leading(512, 768).unwrap();
        let nested_mid = first_two.slice_leading(256, 512).unwrap();
        assert_eq!(
            (first_two.descriptor().allocation, first_two.descriptor().byte_offset),
            (logical.descriptor().allocation, 0)
        );
        assert_eq!(
            (last_one.descriptor().allocation, last_one.descriptor().byte_offset),
            (logical.descriptor().allocation, first_two.byte_len())
        );
        assert_eq!(nested_mid.descriptor().byte_offset, 256 * 2 * 4);
        assert_eq!(logical.slice_leading(255, 256).unwrap().extents(), &[1, 2]);
        assert!(logical.slice_leading(255, 257).is_err());
        assert_eq!(logical.read_to_host().unwrap().len(), 768 * 2 * 4);
        assert_eq!(tensor.region_rows(0, 255, 1).unwrap().extents(), &[1, 2]);
        assert!(tensor.region_rows(0, 255, 2).is_err());
        // Freed beside live views of the store: its slot names the zero slab.
        assert!(tensor.free_slab(first).unwrap());
        assert!(!tensor.free_slab(first).unwrap());
        assert_eq!(entry(&tensor, 0, 0), free);
        assert_eq!(entry(&tensor, 0, 2), free);
        assert_eq!(entry(&tensor, 1, 0), free + tensor.region_offset(1).unwrap());
        assert!(tensor.slab_region(first, 0).unwrap().is_none());
        // The freed slab's view still reads its rows.
        assert_eq!(slab.read_to_host().unwrap().len(), 256 * 2 * 4);
        assert_eq!(tensor.add_slab(SlabFill::Zeroed).unwrap(), first);
        assert_ne!(entry(&tensor, 0, 0), free);
        drop((reshaped, logical, sibling));
    }

    #[test]
    fn typed_regions_work_on_every_available_backend() {
        for device in available() {
            let tensor = SlabTensor::in_pool(&crate::Charges::tally(&device), 
                &pool(&device),
                256,
                512,
                &[SlabRegion { element: Element::f32(), row_shape: vec![2] }],
            )
            .unwrap();
            tensor.add_slab(SlabFill::Zeroed).unwrap();
            let mut row = tensor.region_rows(0, 7, 1).unwrap();
            row.write_from_host(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
            assert_eq!(row.read_to_host().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
            assert!(tensor.free_slab(0).unwrap());
            drop(row);
            assert_eq!(tensor.add_slab(SlabFill::Zeroed).unwrap(), 0);
        }
    }

    /// A zeroed slab reads zero on every backend, whatever its granule held
    /// before.
    #[test]
    fn a_zeroed_slab_reads_zero_whatever_its_granule_held() {
        for device in available() {
            let store = SlabTensor::in_pool(&crate::Charges::tally(&device), 
                &pool(&device),
                256,
                256,
                &[SlabRegion { element: Element::f32(), row_shape: vec![1024] }],
            )
            .unwrap();
            for _ in 0..4 {
                let index = store.add_slab(SlabFill::Zeroed).unwrap();
                let mut rows = store.region_rows(0, 0, 256).unwrap();
                assert!(rows.read_to_host().unwrap().iter().all(|byte| *byte == 0));
                rows.write_from_host(&vec![0xa5; 256 * 1024 * 4]).unwrap();
                drop(rows);
                store.free_slab(index).unwrap();
            }
        }
    }

    #[test]
    fn final_slab_can_exceed_the_logical_extent() {
        let device = cpu();
        let tensor = SlabTensor::in_pool(&crate::Charges::tally(&device), 
            &pool(&device),
            256,
            300,
            &[SlabRegion { element: Element::f32(), row_shape: vec![1] }],
        )
        .unwrap();
        tensor.add_slab(SlabFill::Zeroed).unwrap();
        tensor.add_slab(SlabFill::Zeroed).unwrap();
        let logical = tensor.logical_region(0).unwrap();
        assert_eq!(logical.extents(), &[300, 1]);
        assert_eq!(logical.read_to_host().unwrap().len(), 300 * 4);
        assert_eq!(logical.slice_leading(299, 300).unwrap().extents(), &[1, 1]);
        assert!(logical.slice_leading(299, 301).is_err());
        let mut row = tensor.region_rows(0, 256, 1).unwrap();
        row.write_from_host(&17.0f32.to_le_bytes()).unwrap();
        assert_eq!(&logical.read_to_host().unwrap()[256 * 4..257 * 4], 17.0f32.to_le_bytes());
    }

    /// Lent leading slots view the caller's rows for the store's life: never
    /// freed, and the first slab added lies past them.
    #[test]
    fn lent_slots_view_the_callers_rows_and_are_never_freed() {
        let device = cpu();
        let regions = [SlabRegion { element: Element::f32(), row_shape: vec![2] }];
        let leading = Tensor::zeros(&crate::Charges::tally(&device), Element::f32(), &[2 * 256, 2]).unwrap();
        let tensor =
            SlabTensor::in_pool_over(&crate::Charges::tally(&device), &pool(&device), 256, 4 * 256, &regions, vec![leading.clone()]).unwrap();
        assert_eq!(tensor.lent(), 2);
        assert_eq!(tensor.backed(), [0, 1]);
        assert_eq!(entry(&tensor, 0, 0), leading.device_address());
        assert_eq!(entry(&tensor, 0, 1), leading.device_address() + 256 * 2 * 4);
        assert!(tensor.free_slab(1).is_err());
        assert_eq!(tensor.add_slab(SlabFill::Zeroed).unwrap(), 2);
        let wrong = Tensor::zeros(&crate::Charges::tally(&device), Element::f32(), &[255, 2]).unwrap();
        assert!(SlabTensor::in_pool_over(&crate::Charges::tally(&device), &pool(&device), 256, 4 * 256, &regions, vec![wrong]).is_err());
    }
}
