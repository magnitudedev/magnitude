//! Fixed-width vector values in the one typed kernel IR.
//!
//! Width is part of the Rust handle and the erased `ValueType`. Backends may
//! select native instructions for the value, but cannot change its lanes,
//! element type, masking semantics, or reduction order.

use super::ops::{ErasedValue, ValueType};
use super::BlockId;
use crate::identity::OwnerToken;
use crate::repr::VectorElement;
use std::fmt;
use std::marker::PhantomData;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId<T: VectorElement, const LANES: u16> {
    pub(crate) owner: OwnerToken,
    pub(crate) kernel: u32,
    pub(crate) block: BlockId,
    pub(crate) index: u32,
    marker: PhantomData<T>,
}

impl<T: VectorElement, const LANES: u16> Clone for VectorId<T, LANES> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: VectorElement, const LANES: u16> Copy for VectorId<T, LANES> {}
impl<T: VectorElement, const LANES: u16> fmt::Debug for VectorId<T, LANES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vector<{:?}; {LANES}>#{}", T::DTYPE, self.index)
    }
}

impl<T: VectorElement, const LANES: u16> VectorId<T, LANES> {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, block: BlockId, index: u32) -> Self {
        assert!(LANES > 0, "kernel vectors must have at least one lane");
        Self {
            owner,
            kernel,
            block,
            index,
            marker: PhantomData,
        }
    }

    pub(crate) fn erased(self) -> ErasedValue {
        ErasedValue::new(self.owner, self.kernel, self.block, self.index)
    }

    pub(crate) fn value_type() -> ValueType {
        assert!(LANES > 0, "kernel vectors must have at least one lane");
        ValueType::Vector {
            dtype: T::DTYPE,
            lanes: LANES,
        }
    }
}
