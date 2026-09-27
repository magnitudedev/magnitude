//! Address rows in a slab-stored tensor through its bound address table.

use crate::element::{Dense, Element, U32};
use crate::tensor::{Scalars, Tensor};
use std::marker::PhantomData;

/// A typed logical tensor whose leading axis resolves through a slab table.
#[derive(Clone, Copy)]
pub struct SlabTensor<'a, E: Element, const RANK: usize> {
    table: *const u64,
    rows_per_slab: usize,
    extents: [usize; RANK],
    strides: [usize; RANK],
    life: PhantomData<&'a E>,
}

// SAFETY: the same work-item read/write contract as `tensor::Tensor` applies.
unsafe impl<E: Element, const RANK: usize> Send for SlabTensor<'_, E, RANK> {}
unsafe impl<E: Element, const RANK: usize> Sync for SlabTensor<'_, E, RANK> {}

impl<'a, E: Element, const RANK: usize> SlabTensor<'a, E, RANK> {
    #[inline(always)]
    pub fn from_bound(bound: Tensor<'a, E, RANK>, rows_per_slab: usize) -> Self {
        assert!(RANK > 0 && rows_per_slab > 0);
        Self {
            table: bound.pointer().cast(),
            rows_per_slab,
            extents: bound.extents(),
            strides: bound.strides(),
            life: PhantomData,
        }
    }

    #[inline(always)]
    fn pointer(&self, index: [usize; RANK]) -> *mut E::Storage {
        debug_assert!(index.iter().zip(self.extents).all(|(i, n)| *i < n));
        let slab = index[0] / self.rows_per_slab;
        let mut offset = (index[0] % self.rows_per_slab) * self.strides[0];
        for axis in 1..RANK {
            offset += index[axis] * self.strides[axis];
        }
        // SAFETY: native launch validation owns every named slab and row.
        unsafe { region::<E::Storage>(self.table, slab).add(offset) }
    }

    #[inline(always)]
    pub fn row(&self, index: [usize; RANK]) -> &'a [E::Storage] {
        assert_eq!(self.strides[RANK - 1], 1);
        let mut start = index;
        start[RANK - 1] = 0;
        // SAFETY: the last axis is contiguous and within the bound region.
        unsafe { std::slice::from_raw_parts(self.pointer(start), self.extents[RANK - 1]) }
    }

    /// # Safety
    /// The caller gives each work item distinct writes and no concurrent readers.
    #[inline(always)]
    pub unsafe fn row_mut(&self, index: [usize; RANK]) -> &'a mut [E::Storage] {
        assert_eq!(self.strides[RANK - 1], 1);
        let mut start = index;
        start[RANK - 1] = 0;
        unsafe { std::slice::from_raw_parts_mut(self.pointer(start), self.extents[RANK - 1]) }
    }

    #[inline(always)]
    pub fn span(&self, index: [usize; RANK], len: usize) -> &'a [E::Storage] {
        assert_eq!(self.strides[RANK - 1], 1);
        assert!(index[RANK - 1] + len <= self.extents[RANK - 1]);
        unsafe { std::slice::from_raw_parts(self.pointer(index), len) }
    }

    /// # Safety
    /// The caller gives each work item distinct writes and no concurrent readers.
    #[inline(always)]
    pub unsafe fn span_mut(&self, index: [usize; RANK], len: usize) -> &'a mut [E::Storage] {
        assert_eq!(self.strides[RANK - 1], 1);
        assert!(index[RANK - 1] + len <= self.extents[RANK - 1]);
        unsafe { std::slice::from_raw_parts_mut(self.pointer(index), len) }
    }
}

impl<'a, const RANK: usize> SlabTensor<'a, U32, RANK> {
    #[inline(always)]
    pub fn from_scalars(bound: Scalars<'a, u32, RANK>, rows_per_slab: usize) -> Self {
        // SAFETY: U32 storage is u32 and the bound scalar view describes the
        // same logical extents, strides and address-table lifetime.
        let view = unsafe {
            Tensor::<U32, RANK>::from_raw(
                bound.pointer(),
                bound.extents().map(|extent| extent as u64),
                bound.strides().map(|stride| stride as u64),
            )
        };
        Self::from_bound(view, rows_per_slab)
    }
}

impl<E: Dense, const RANK: usize> SlabTensor<'_, E, RANK> {
    #[inline(always)]
    pub fn get(&self, index: [usize; RANK]) -> f32 {
        E::widen(unsafe { *self.pointer(index) })
    }

    /// # Safety
    /// The caller gives each work item distinct writes and no concurrent readers.
    #[inline(always)]
    pub unsafe fn set(&self, index: [usize; RANK], value: f32) {
        unsafe { *self.pointer(index) = E::narrow(value) };
    }
}

/// Resolve one component region. The table holds one native pointer per slab.
///
/// # Safety
/// `table` must address a bound table with a backed entry at `slab_index`.
#[inline(always)]
pub unsafe fn region<T>(table: *const u64, slab_index: usize) -> *mut T {
    unsafe { *table.add(slab_index) as usize as *mut T }
}

/// Resolve one row within a component region.
///
/// # Safety
/// `table` must have a backed slab for `index`, and that region must contain
/// `rows_per_slab` rows of `row_elements` values of `T`.
#[inline(always)]
pub unsafe fn row<T>(
    table: *const u64,
    index: usize,
    rows_per_slab: usize,
    row_elements: usize,
) -> *mut T {
    unsafe { region::<T>(table, index / rows_per_slab).add((index % rows_per_slab) * row_elements) }
}
