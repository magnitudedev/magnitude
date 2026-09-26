//! Typed views of the tensors a CPU kernel receives.
//!
//! A view carries its base address, extents and strides from the ABI words.
//! Element addressing is in elements of the view's storage type; rows of a
//! canonical view (unit innermost stride) are exact-length slices, so loops
//! over them carry no per-element bounds checks.
//!
//! Work items of one launch run concurrently. Reads through a view are safe;
//! writes go through `unsafe` accessors whose contract is the native one: no
//! two work items write the same element, and no work item reads an element
//! another writes in the same launch.

use crate::element::{Dense, Element};
use std::marker::PhantomData;

/// A tensor of dense elements `E` with rank `RANK`. Every element's storage
/// can be moved; a [`Dense`] element's values also read and write as `f32`.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a, E: Element, const RANK: usize> {
    base: *mut E::Storage,
    extents: [usize; RANK],
    strides: [usize; RANK],
    life: PhantomData<&'a E>,
}

// SAFETY: see the module contract; views only address storage the launch
// owns for its duration.
unsafe impl<E: Element, const RANK: usize> Send for Tensor<'_, E, RANK> {}
unsafe impl<E: Element, const RANK: usize> Sync for Tensor<'_, E, RANK> {}

/// A tensor of raw scalars of a concrete dtype (`f32`, `i32`, `u32`).
#[derive(Clone, Copy, Debug)]
pub struct Scalars<'a, T: Copy, const RANK: usize> {
    base: *mut T,
    extents: [usize; RANK],
    strides: [usize; RANK],
    life: PhantomData<&'a T>,
}

unsafe impl<T: Copy, const RANK: usize> Send for Scalars<'_, T, RANK> {}
unsafe impl<T: Copy, const RANK: usize> Sync for Scalars<'_, T, RANK> {}

macro_rules! addressing {
    ($view:ident, $item:ty, [$($bound:tt)*]) => {
        impl<'a, $($bound)*, const RANK: usize> $view<'a, $item, RANK> {
            /// A view of `base` with `extents` and `strides` in elements.
            ///
            /// # Safety
            /// Every addressed element lies in storage valid for `'a`.
            pub unsafe fn from_raw(base: *mut u8, extents: [u64; RANK], strides: [u64; RANK]) -> Self {
                Self {
                    base: base.cast(),
                    extents: extents.map(|extent| extent as usize),
                    strides: strides.map(|stride| stride as usize),
                    life: PhantomData,
                }
            }

            pub fn extents(&self) -> [usize; RANK] {
                self.extents
            }

            pub fn extent(&self, axis: usize) -> usize {
                self.extents[axis]
            }

            pub fn strides(&self) -> [usize; RANK] {
                self.strides
            }

            #[inline(always)]
            fn offset(&self, index: [usize; RANK]) -> usize {
                let mut offset = 0;
                for axis in 0..RANK {
                    debug_assert!(index[axis] < self.extents[axis], "index {index:?} outside {:?}", self.extents);
                    offset += index[axis] * self.strides[axis];
                }
                offset
            }

            /// The base address of the view.
            pub fn pointer(&self) -> *mut u8 {
                self.base.cast()
            }
        }
    };
}

addressing!(Tensor, E, [E: Element]);
addressing!(Scalars, T, [T: Copy]);

impl<E: Dense, const RANK: usize> Tensor<'_, E, RANK> {
    /// The element at `index`, widened to `f32`.
    #[inline(always)]
    pub fn get(&self, index: [usize; RANK]) -> f32 {
        // SAFETY: `from_raw`'s contract.
        E::widen(unsafe { *self.base.add(self.offset(index)) })
    }

    /// Stores `value` rounded to `E` at `index`.
    ///
    /// # Safety
    /// The module's write contract.
    #[inline(always)]
    pub unsafe fn set(&self, index: [usize; RANK], value: f32) {
        unsafe { *self.base.add(self.offset(index)) = E::narrow(value) };
    }
}

impl<'a, E: Element, const RANK: usize> Tensor<'a, E, RANK> {
    /// The innermost row at `index` (its last coordinate ignored), which
    /// must be contiguous.
    #[inline(always)]
    pub fn row(&self, index: [usize; RANK]) -> &'a [E::Storage] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a row view needs a unit innermost stride"
        );
        let mut start = index;
        start[RANK - 1] = 0;
        // SAFETY: `from_raw`'s contract; the row lies inside the view.
        unsafe {
            std::slice::from_raw_parts(
                self.base.add(self.offset_row(start)),
                self.extents[RANK - 1],
            )
        }
    }

    /// The same row, writable.
    ///
    /// # Safety
    /// The module's write contract, for every element of the row.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn row_mut(&self, index: [usize; RANK]) -> &'a mut [E::Storage] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a row view needs a unit innermost stride"
        );
        let mut start = index;
        start[RANK - 1] = 0;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.base.add(self.offset_row(start)),
                self.extents[RANK - 1],
            )
        }
    }

    #[inline(always)]
    fn offset_row(&self, index: [usize; RANK]) -> usize {
        let mut offset = 0;
        for axis in 0..RANK - 1 {
            debug_assert!(index[axis] < self.extents[axis]);
            offset += index[axis] * self.strides[axis];
        }
        offset
    }

    /// `length` contiguous elements of the innermost row from `index`.
    #[inline(always)]
    pub fn span(&self, index: [usize; RANK], length: usize) -> &'a [E::Storage] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a span needs a unit innermost stride"
        );
        assert!(
            index[RANK - 1] + length <= self.extents[RANK - 1],
            "span outside its row"
        );
        // SAFETY: `from_raw`'s contract; the span lies inside its row.
        unsafe { std::slice::from_raw_parts(self.base.add(self.offset(index)), length) }
    }

    /// The same span, writable: the way a work item writes its part of a row
    /// another work item also writes.
    ///
    /// # Safety
    /// The module's write contract, for every element of the span.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn span_mut(&self, index: [usize; RANK], length: usize) -> &'a mut [E::Storage] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a span needs a unit innermost stride"
        );
        assert!(
            index[RANK - 1] + length <= self.extents[RANK - 1],
            "span outside its row"
        );
        unsafe { std::slice::from_raw_parts_mut(self.base.add(self.offset(index)), length) }
    }
}

impl<'a, T: Copy, const RANK: usize> Scalars<'a, T, RANK> {
    #[inline(always)]
    pub fn get(&self, index: [usize; RANK]) -> T {
        // SAFETY: `from_raw`'s contract.
        unsafe { *self.base.add(self.offset(index)) }
    }

    /// # Safety
    /// The module's write contract.
    #[inline(always)]
    pub unsafe fn set(&self, index: [usize; RANK], value: T) {
        unsafe { *self.base.add(self.offset(index)) = value };
    }

    /// The innermost row at `index`, which must be contiguous.
    #[inline(always)]
    pub fn row(&self, index: [usize; RANK]) -> &'a [T] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a row view needs a unit innermost stride"
        );
        let mut start = index;
        start[RANK - 1] = 0;
        let offset = self.offset_row(start);
        unsafe { std::slice::from_raw_parts(self.base.add(offset), self.extents[RANK - 1]) }
    }

    /// # Safety
    /// The module's write contract, for every element of the row.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn row_mut(&self, index: [usize; RANK]) -> &'a mut [T] {
        assert_eq!(
            self.strides[RANK - 1],
            1,
            "a row view needs a unit innermost stride"
        );
        let mut start = index;
        start[RANK - 1] = 0;
        let offset = self.offset_row(start);
        unsafe { std::slice::from_raw_parts_mut(self.base.add(offset), self.extents[RANK - 1]) }
    }

    #[inline(always)]
    fn offset_row(&self, index: [usize; RANK]) -> usize {
        let mut offset = 0;
        for axis in 0..RANK - 1 {
            debug_assert!(index[axis] < self.extents[axis]);
            offset += index[axis] * self.strides[axis];
        }
        offset
    }
}

/// Call-private scratch bytes of one launch.
#[derive(Clone, Copy, Debug)]
pub struct Scratch<'a> {
    base: *mut u8,
    life: PhantomData<&'a u8>,
}

unsafe impl Send for Scratch<'_> {}
unsafe impl Sync for Scratch<'_> {}

impl<'a> Scratch<'a> {
    /// # Safety
    /// `base` is valid for `'a` and aligned to 256 bytes.
    pub unsafe fn from_raw(base: *mut u8) -> Self {
        Self {
            base,
            life: PhantomData,
        }
    }

    /// `length` elements of `T` from byte `offset`.
    ///
    /// # Safety
    /// The range lies inside the scratch, `offset` is aligned for `T`, and
    /// the module's write contract holds for the elements written.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut<T: Copy>(&self, offset: usize, length: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.base.add(offset).cast(), length) }
    }

    /// # Safety
    /// As [`Scratch::slice_mut`], for reading.
    pub unsafe fn slice<T: Copy>(&self, offset: usize, length: usize) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.base.add(offset).cast(), length) }
    }
}

/// The first `count` `f32` values of a work item's private bytes (which the
/// pool aligns for any scalar).
#[inline(always)]
pub fn floats(bytes: &mut [u8], count: usize) -> &mut [f32] {
    assert!(
        bytes.len() >= 4 * count,
        "{count} floats need {} private bytes; the launch has {}",
        4 * count,
        bytes.len()
    );
    // SAFETY: every bit pattern is an `f32`; `align_to_mut` places only
    // aligned whole values in the middle part.
    let (head, middle, _) = unsafe { bytes.align_to_mut::<f32>() };
    assert!(head.is_empty(), "private bytes are aligned for f32");
    &mut middle[..count]
}

/// The `f32` values of `row`.
#[inline(always)]
pub fn widen_row<E: Dense>(row: &[E::Storage], out: &mut [f32]) {
    let out = &mut out[..row.len()];
    for (value, target) in row.iter().zip(out) {
        *target = E::widen(*value);
    }
}

/// `values` rounded to `E` into `row`.
#[inline(always)]
pub fn narrow_row<E: Dense>(values: &[f32], row: &mut [E::Storage]) {
    let row = &mut row[..values.len()];
    for (value, target) in values.iter().zip(row) {
        *target = E::narrow(*value);
    }
}
