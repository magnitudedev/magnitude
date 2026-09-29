use crate::{Device, Element, Tensor, TensorError, TensorStorageObserver};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// One component region repeated for every row in a slab. `row_shape`
/// excludes the leading row axis.
#[derive(Clone, Debug)]
pub struct SlabRegion {
    pub element: Element,
    pub row_shape: Vec<u64>,
}

/// Device-independent physical charge of a fixed slab store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabLayout {
    pub slab_bytes: u64,
    pub address_table_bytes: u64,
}

impl SlabLayout {
    pub fn for_regions(
        rows_per_slab: u64,
        logical_rows: u64,
        regions: &[SlabRegion],
    ) -> Result<Self, TensorError> {
        let (bytes, _, capacity) = layout_regions(rows_per_slab, logical_rows, regions)?;
        let region_count = u64::try_from(regions.len())
            .map_err(|_| TensorError::SlabLayout("region count exceeds u64".into()))?;
        let (address_table_bytes, _) =
            Tensor::canonical_layout(Element::u32(), &[region_count, capacity as u64, 2])?;
        Ok(Self {
            slab_bytes: bytes,
            address_table_bytes,
        })
    }
}

fn layout_regions(
    rows_per_slab: u64,
    logical_rows: u64,
    regions: &[SlabRegion],
) -> Result<(u64, Vec<u64>, usize), TensorError> {
    if rows_per_slab == 0 || logical_rows == 0 || regions.is_empty() {
        return Err(TensorError::SlabLayout(
            "a slab needs rows, logical extent and at least one region".into(),
        ));
    }
    let capacity = usize::try_from(logical_rows.div_ceil(rows_per_slab))
        .map_err(|_| TensorError::SlabLayout("slab capacity exceeds host domain".into()))?;
    let mut offsets = Vec::with_capacity(regions.len());
    let mut bytes = 0u64;
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
        let alignment = alignment.max(256);
        bytes = bytes
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or_else(|| TensorError::SlabLayout("slab offset overflows".into()))?;
        offsets.push(bytes);
        bytes = bytes
            .checked_add(length)
            .ok_or_else(|| TensorError::SlabLayout("slab size overflows".into()))?;
    }
    Ok((bytes, offsets, capacity))
}

/// One physical allocation, with typed component views over disjoint regions.
/// Cloned views and submitted work retain the allocation after its slot is freed.
pub struct Slab {
    storage: Tensor,
    regions: Vec<Tensor>,
}

impl Slab {
    pub(crate) fn device_address(&self) -> u64 {
        self.storage.device_address()
    }

    pub fn region(&self, index: usize) -> Option<&Tensor> {
        self.regions.get(index)
    }

    pub fn observe_storage(&self) -> TensorStorageObserver {
        self.storage.observe_storage()
    }
}

/// Fixed-size allocations backing a common logical row domain. Indices are
/// stable while allocated, and the lowest freed index is reused first.
pub struct SlabTensor {
    device: Device,
    rows_per_slab: u64,
    logical_rows: u64,
    capacity: usize,
    regions: Vec<SlabRegion>,
    aliases: Vec<u64>,
    offsets: Vec<u64>,
    slab_bytes: u64,
    slabs: Vec<Option<Slab>>,
    addresses: Tensor,
    region_tables: Vec<Tensor>,
    address_words: Vec<u8>,
    lease: Arc<AtomicUsize>,
}

impl SlabTensor {
    pub fn new(
        device: &Device,
        rows_per_slab: u64,
        logical_rows: u64,
        regions: Vec<SlabRegion>,
    ) -> Result<Self, TensorError> {
        let (bytes, offsets, capacity) = layout_regions(rows_per_slab, logical_rows, &regions)?;
        let capacity_u64 = capacity as u64;
        let region_count = u64::try_from(regions.len())
            .map_err(|_| TensorError::SlabLayout("region count exceeds u64".into()))?;
        let table = Tensor::zeros(device, Element::u32(), &[region_count, capacity_u64, 2])?;
        let region_tables = (0..region_count)
            .map(|region| {
                table
                    .slice_leading(region, region + 1)?
                    .reshape(&[capacity_u64, 2])
            })
            .collect::<Result<Vec<_>, TensorError>>()?;
        let table_bytes = usize::try_from(table.byte_len())
            .map_err(|_| TensorError::SlabLayout("address table exceeds host domain".into()))?;
        Ok(Self {
            device: device.clone(),
            rows_per_slab,
            logical_rows,
            capacity,
            regions,
            aliases: (0..offsets.len())
                .map(|_| Tensor::new_slab_alias())
                .collect(),
            offsets,
            slab_bytes: bytes,
            slabs: Vec::new(),
            addresses: table,
            region_tables,
            address_words: vec![0; table_bytes],
            lease: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn rows_per_slab(&self) -> u64 {
        self.rows_per_slab
    }

    pub fn slab_bytes(&self) -> u64 {
        self.slab_bytes
    }

    /// The fixed address table plus currently backed slab allocations.
    pub fn storage_bytes(&self) -> u64 {
        self.addresses.storage_bytes() + self.slabs().count() as u64 * self.slab_bytes
    }

    pub fn slab_storage_observers(&self) -> impl Iterator<Item = TensorStorageObserver> + '_ {
        self.slabs().map(|(_, slab)| slab.observe_storage())
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Two little-endian u32 words per slab address, already offset to this
    /// component's region. All region tables share one fixed allocation.
    pub(crate) fn address_table(&self, region: usize) -> Option<&Tensor> {
        self.region_tables.get(region)
    }

    pub fn region_offset(&self, index: usize) -> Option<u64> {
        self.offsets.get(index).copied()
    }

    pub fn logical_rows(&self) -> u64 {
        self.logical_rows
    }

    pub fn backed_extent(&self) -> u64 {
        self.slabs.len() as u64 * self.rows_per_slab
    }

    pub fn slab(&self, index: usize) -> Option<&Slab> {
        self.slabs.get(index).and_then(Option::as_ref)
    }

    /// An ordinary typed view of a span wholly within one backed slab.
    /// The returned view retains that slab's physical allocation.
    pub fn region_rows(&self, region: usize, start: u64, rows: u64) -> Result<Tensor, TensorError> {
        let end = start
            .checked_add(rows)
            .ok_or_else(|| TensorError::SlabLayout("span end overflows".into()))?;
        if rows == 0
            || end > self.logical_rows
            || (end - 1) / self.rows_per_slab != start / self.rows_per_slab
        {
            return Err(TensorError::SlabLayout(
                "span must lie within one slab".into(),
            ));
        }
        let index = usize::try_from(start / self.rows_per_slab)
            .map_err(|_| TensorError::SlabLayout("slab index exceeds host domain".into()))?;
        let slab = self
            .slab(index)
            .ok_or_else(|| TensorError::SlabLayout("span refers to a free slab".into()))?;
        let component = slab
            .region(region)
            .ok_or_else(|| TensorError::SlabLayout("unknown component region".into()))?;
        let local = start % self.rows_per_slab;
        component.slice_leading(local, local + rows)
    }

    /// Test oracle for copying one row between backed slabs.
    #[cfg(test)]
    pub fn copy_row(&self, region: usize, from: u64, to: u64) -> Result<(), TensorError> {
        if from == to {
            return Ok(());
        }
        let source = self.region_rows(region, from, 1)?;
        let mut destination = self.region_rows(region, to, 1)?;
        destination.write_from_host(&source.read_to_host()?)
    }

    /// Logical tensor for a component region. Its native buffer is the fixed
    /// address table; the snapshot retains every slab that was backed when
    /// this view was formed. Form a fresh view after placement changes.
    pub fn logical_region(&self, region: usize) -> Result<Tensor, TensorError> {
        let specification = self
            .regions
            .get(region)
            .ok_or_else(|| TensorError::SlabLayout("unknown component region".into()))?;
        let views = (0..self.capacity)
            .map(|index| {
                self.slab(index)
                    .and_then(|slab| slab.region(region))
                    .cloned()
            })
            .collect();
        Tensor::slabbed(
            &self.region_tables[region],
            specification.element,
            self.rows_per_slab,
            self.logical_rows,
            self.aliases[region],
            views,
            self.lease.clone(),
        )
    }

    pub fn slabs(&self) -> impl Iterator<Item = (usize, &Slab)> {
        self.slabs
            .iter()
            .enumerate()
            .filter_map(|(index, slab)| slab.as_ref().map(|slab| (index, slab)))
    }

    /// Allocate exactly one slab. A failed allocation leaves all slots intact.
    pub fn add_slab(&mut self) -> Result<usize, TensorError> {
        self.ensure_unbound()?;
        let index = self
            .slabs
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.slabs.len());
        if index >= self.capacity {
            return Err(TensorError::SlabLayout("all slab slots are backed".into()));
        }
        let storage = Tensor::zeros(&self.device, Element::bool(), &[self.slab_bytes])?;
        storage.share_slab_access_with(&self.addresses);
        let mut views = Vec::with_capacity(self.regions.len());
        for (region, &offset) in self.regions.iter().zip(&self.offsets) {
            let extents = std::iter::once(self.rows_per_slab)
                .chain(region.row_shape.iter().copied())
                .collect::<Vec<_>>();
            views.push(storage.region(region.element, &extents, offset)?);
        }
        let slab = Slab {
            storage,
            regions: views,
        };
        slab.storage.register_slab();
        self.write_addresses(index, slab.device_address())?;
        if index < self.slabs.len() {
            self.slabs[index] = Some(slab);
            Ok(index)
        } else {
            self.slabs.push(Some(slab));
            Ok(index)
        }
    }

    /// Remove one slab. The observer reports when all device uses and views
    /// actually release its charge; removing a slot alone grants no credit.
    pub fn free_slab(
        &mut self,
        index: usize,
    ) -> Result<Option<TensorStorageObserver>, TensorError> {
        self.ensure_unbound()?;
        if self.slab(index).is_none() {
            return Ok(None);
        }
        self.write_addresses(index, 0)?;
        let slab = self.slabs[index].take().expect("checked backed slab");
        Ok(Some(slab.observe_storage()))
    }

    fn write_addresses(&mut self, index: usize, base: u64) -> Result<(), TensorError> {
        let mut next = self.address_words.clone();
        for (region, offset) in self.offsets.iter().enumerate() {
            let address = if base == 0 {
                0
            } else {
                base.checked_add(*offset)
                    .ok_or_else(|| TensorError::SlabLayout("slab address overflows".into()))?
            };
            let position = (region * self.capacity + index) * 8;
            next[position..position + 8].copy_from_slice(&address.to_le_bytes());
        }
        self.addresses.write_from_host(&next)?;
        self.address_words = next;
        Ok(())
    }

    fn ensure_unbound(&self) -> Result<(), TensorError> {
        if self.lease.load(Ordering::Acquire) != 0 {
            Err(TensorError::SlabLayout(
                "slab placement has a live binding".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Availability, BackendName, DeviceCatalog};

    #[test]
    fn slab_regions_share_one_charge_and_reuse_the_lowest_slot() {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        let regions = vec![
            SlabRegion {
                element: Element::f32(),
                row_shape: vec![2],
            },
            SlabRegion {
                element: Element::f16(),
                row_shape: vec![3],
            },
        ];
        let layout = SlabLayout::for_regions(256, 768, &regions).unwrap();
        let mut tensor = SlabTensor::new(&device, 256, 768, regions).unwrap();
        assert_eq!(layout.slab_bytes, tensor.slab_bytes());
        assert_eq!(layout.address_table_bytes, tensor.storage_bytes());
        let first = tensor.add_slab().unwrap();
        let second = tensor.add_slab().unwrap();
        assert_eq!((first, second), (0, 1));
        let address = tensor.slab(first).unwrap().device_address();
        assert_eq!(
            tensor
                .address_table(0)
                .unwrap()
                .slice_leading(0, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            address.to_le_bytes()
        );
        assert_eq!(
            tensor
                .address_table(1)
                .unwrap()
                .slice_leading(0, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            (address + tensor.region_offset(1).unwrap()).to_le_bytes()
        );
        assert_eq!(tensor.logical_rows(), 768);
        assert_eq!(tensor.backed_extent(), 512);
        let slab = tensor.slab(first).unwrap();
        assert_eq!(slab.region(0).unwrap().extents(), &[256, 2]);
        let logical = tensor.logical_region(0).unwrap();
        let sibling = tensor.logical_region(1).unwrap();
        assert_eq!(logical.extents(), &[768, 2]);
        let reshaped = logical.reshape(&[768, 1, 2]).unwrap();
        assert_eq!(reshaped.extents(), &[768, 1, 2]);
        assert!(logical.reshape(&[384, 4]).is_err());
        drop(reshaped);
        assert_ne!(
            logical.descriptor().allocation,
            sibling.descriptor().allocation
        );
        assert_eq!(logical.slice_leading(255, 256).unwrap().extents(), &[1, 2]);
        assert!(logical.slice_leading(255, 257).is_err());
        assert_eq!(logical.read_to_host().unwrap().len(), 768 * 2 * 4);
        assert_eq!(slab.region(1).unwrap().extents(), &[256, 3]);
        assert!(slab
            .region(0)
            .unwrap()
            .shares_allocation(slab.region(1).unwrap()));
        assert_eq!(
            slab.region(1).unwrap().device_address(),
            slab.device_address() + tensor.region_offset(1).unwrap()
        );
        assert_eq!(tensor.region_rows(0, 255, 1).unwrap().extents(), &[1, 2]);
        assert!(tensor.region_rows(0, 255, 2).is_err());
        let view = slab.region(0).unwrap().clone();
        assert!(tensor.free_slab(first).is_err());
        drop(logical);
        drop(sibling);
        let observer = tensor.free_slab(first).unwrap().unwrap();
        assert_eq!(
            tensor
                .address_table(0)
                .unwrap()
                .slice_leading(0, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            [0; 8]
        );
        assert!(observer.charged_bytes().is_some());
        assert_eq!(tensor.add_slab().unwrap(), first);
        assert!(!view.shares_allocation(tensor.slab(first).unwrap().region(0).unwrap()));
        drop(view);
        assert_eq!(observer.charged_bytes(), None);
    }

    #[test]
    fn typed_regions_work_on_every_available_backend() {
        let catalog = DeviceCatalog::discover().unwrap();
        for info in catalog.topology().devices() {
            if !matches!(info.availability, Availability::Available) {
                continue;
            }
            let device = catalog.open(info.id).unwrap();
            let mut tensor = SlabTensor::new(
                &device,
                256,
                512,
                vec![SlabRegion {
                    element: Element::f32(),
                    row_shape: vec![2],
                }],
            )
            .unwrap();
            tensor.add_slab().unwrap();
            let mut row = tensor.region_rows(0, 7, 1).unwrap();
            row.write_from_host(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
            assert_eq!(row.read_to_host().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
            let observer = tensor.free_slab(0).unwrap().unwrap();
            assert!(observer.charged_bytes().is_some());
            drop(row);
            assert_eq!(observer.charged_bytes(), None);
            assert_eq!(tensor.add_slab().unwrap(), 0);
        }
    }

    #[test]
    fn final_slab_can_exceed_the_logical_extent() {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        let mut tensor = SlabTensor::new(
            &device,
            256,
            300,
            vec![SlabRegion {
                element: Element::f32(),
                row_shape: vec![1],
            }],
        )
        .unwrap();
        tensor.add_slab().unwrap();
        tensor.add_slab().unwrap();
        let logical = tensor.logical_region(0).unwrap();
        assert_eq!(logical.extents(), &[300, 1]);
        assert_eq!(logical.read_to_host().unwrap().len(), 300 * 4);
        assert_eq!(logical.slice_leading(299, 300).unwrap().extents(), &[1, 1]);
        assert!(logical.slice_leading(299, 301).is_err());
        let mut source = tensor.region_rows(0, 255, 1).unwrap();
        source.write_from_host(&17.0f32.to_le_bytes()).unwrap();
        tensor.copy_row(0, 255, 256).unwrap();
        assert_eq!(
            tensor
                .region_rows(0, 256, 1)
                .unwrap()
                .read_to_host()
                .unwrap(),
            17.0f32.to_le_bytes()
        );
    }
}
