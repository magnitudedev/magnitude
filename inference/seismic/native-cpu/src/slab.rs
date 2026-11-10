//! Address rows in a slab-stored tensor through its bound address table.

use crate::element::{Dense, Element, U32};
use crate::tensor::{Scalars, Tensor};
use std::marker::PhantomData;

/// A typed logical tensor whose leading axis resolves through a slab table.
#[derive(Clone, Copy)]
pub struct SlabTensor<'a, E: Element, const RANK: usize> {
    table: *const u64,
    rows_per_slab: usize,
    /// `ceil(2^64 / rows_per_slab)`: a leading index's slab is
    /// `index * reciprocal >> 64`, exact for indices and slab sizes below
    /// 2^32 (Lemire, Kaser and Kurz, "Faster remainder by direct
    /// computation", 2019), so no row pays a division.
    reciprocal: u128,
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
        let extents = bound.extents();
        assert!(rows_per_slab <= u32::MAX as usize && extents[0] <= u32::MAX as usize);
        Self {
            table: bound.pointer().cast(),
            rows_per_slab,
            reciprocal: reciprocal(rows_per_slab),
            extents,
            strides: bound.strides(),
            life: PhantomData,
        }
    }

    #[inline(always)]
    fn pointer(&self, index: [usize; RANK]) -> *mut E::Storage {
        debug_assert!(index.iter().zip(self.extents).all(|(i, n)| *i < n));
        let slab = slab_of(index[0], self.reciprocal);
        let mut offset = (index[0] - slab * self.rows_per_slab) * self.strides[0];
        for axis in 1..RANK {
            offset += index[axis] * self.strides[axis];
        }
        // SAFETY: native launch validation owns every named slab and row.
        unsafe { region::<E::Storage>(self.table, slab).add(offset) }
    }

    /// A walk over consecutive leading indices from `first`, for this
    /// tensor and any other of its slab size ([`Self::row_at`]).
    #[inline(always)]
    pub fn walk(&self, first: usize) -> SlabWalk {
        SlabWalk::new(first, self.rows_per_slab, self.reciprocal)
    }

    /// The rows in one slab.
    #[inline(always)]
    pub fn rows_per_slab(&self) -> usize {
        self.rows_per_slab
    }

    /// [`Self::row`] at `index`, whose leading index is `walk`'s: its slab
    /// already resolved.
    #[inline(always)]
    pub fn row_at(&self, walk: &SlabWalk, index: [usize; RANK]) -> &'a [E::Storage] {
        assert_eq!(self.strides[RANK - 1], 1);
        debug_assert!(walk.rows_per_slab == self.rows_per_slab && index[0] == walk.index && walk.index < self.extents[0]);
        let mut offset = walk.row * self.strides[0];
        for axis in 1..RANK - 1 {
            offset += index[axis] * self.strides[axis];
        }
        // SAFETY: the walk's slab and row are those of its leading index,
        // which native launch validation owns; the last axis is contiguous.
        unsafe { std::slice::from_raw_parts(region::<E::Storage>(self.table, walk.slab).add(offset), self.extents[RANK - 1]) }
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

/// `ceil(2^64 / rows)` for [`slab_of`].
fn reciprocal(rows: usize) -> u128 {
    u128::from(u64::MAX) / rows as u128 + 1
}

/// A leading index of slab tensors of one slab size, walked one index at a
/// time: its slab resolved once ([`SlabTensor::walk`]), then stepped row by
/// row with no division.
#[derive(Clone, Copy)]
pub struct SlabWalk {
    rows_per_slab: usize,
    index: usize,
    slab: usize,
    row: usize,
}

impl SlabWalk {
    /// The walk at `first` over slabs of `rows_per_slab` rows, of
    /// [`reciprocal`] `reciprocal`.
    #[inline(always)]
    fn new(first: usize, rows_per_slab: usize, reciprocal: u128) -> Self {
        let slab = slab_of(first, reciprocal);
        Self { rows_per_slab, index: first, slab, row: first - slab * rows_per_slab }
    }

    /// The leading index.
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Steps to the next leading index.
    #[inline(always)]
    pub fn step(&mut self) {
        self.index += 1;
        self.row += 1;
        if self.row == self.rows_per_slab {
            self.slab += 1;
            self.row = 0;
        }
    }
}

/// `index / rows` for the `reciprocal` of `rows`, both below 2^32.
#[inline(always)]
fn slab_of(index: usize, reciprocal: u128) -> usize {
    ((index as u128 * reciprocal) >> 64) as usize
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

#[cfg(test)]
mod tests {
    use super::{reciprocal, slab_of, SlabWalk};

    #[test]
    fn slab_of_is_the_quotient_below_two_to_the_32() {
        let top = u32::MAX as usize;
        for rows in [1, 2, 3, 7, 255, 256, 257, 1000, 4096, 65_537, top - 1, top] {
            let inverse = reciprocal(rows);
            let indices = (0..4 * rows.min(1 << 12))
                .chain([rows - 1, rows, rows + 1, top / rows * rows, top - 1, top]);
            for index in indices.filter(|index| *index <= top) {
                assert_eq!(slab_of(index, inverse), index / rows, "{index} / {rows}");
            }
        }
    }

    #[test]
    fn a_walk_steps_through_the_slabs_its_first_index_resolves() {
        for rows in [1, 3, 48] {
            for first in [0, 1, 47, 48, 100] {
                let mut walk = SlabWalk::new(first, rows, reciprocal(rows));
                for index in first..first + 3 * rows {
                    assert_eq!((walk.index(), walk.slab, walk.row), (index, index / rows, index % rows));
                    walk.step();
                }
            }
        }
    }
}
