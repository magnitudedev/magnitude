use super::{MemoryReservation, TensorData};
use crate::ids::RepresentationId;
use crate::reference_math::ReferenceScalar;
use num_bigint::{BigInt, BigUint, ToBigUint};
use num_traits::ToPrimitive;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub(super) enum Backing {
    Argument(usize),
    Owned(Rc<RefCell<TensorData>>),
}

#[derive(Clone, Debug)]
pub(super) struct TensorMemory {
    pub(super) _backing: Option<Rc<MemoryReservation>>,
    pub(super) shape: Rc<MemoryReservation>,
    pub(super) positions: Rc<MemoryReservation>,
}

/// A logical strided view. Storage representations stay on the backing; a
/// view never manufactures a second physical interpretation.
#[derive(Clone, Debug)]
pub struct TensorValue {
    pub(super) backing: Backing,
    pub(super) representation: RepresentationId,
    pub(super) shape: Rc<Vec<usize>>,
    /// Backing-flat position of every logical row-major element. Keeping the
    /// logical index map explicit makes arbitrary compositions of
    /// slice/transpose/reshape exact without inventing backend view rules.
    pub(super) positions: Rc<Vec<usize>>,
    pub(super) memory: TensorMemory,
}

impl TensorValue {
    pub fn representation(&self) -> RepresentationId {
        self.representation
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) fn argument(
        id: usize,
        representation: RepresentationId,
        shape: &[usize],
        memory: TensorMemory,
    ) -> Self {
        Self {
            backing: Backing::Argument(id),
            representation,
            shape: Rc::new(shape.to_vec()),
            positions: Rc::new((0..shape.iter().product()).collect()),
            memory,
        }
    }

    pub(super) fn owned(tensor: TensorData, memory: TensorMemory) -> Self {
        let representation = tensor.representation();
        let shape = tensor.shape().to_vec();
        Self {
            backing: Backing::Owned(Rc::new(RefCell::new(tensor))),
            representation,
            positions: Rc::new((0..shape.iter().product()).collect()),
            memory,
            shape: Rc::new(shape),
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

pub(super) fn row_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

#[derive(Clone, Debug)]
pub enum Value {
    Scalar(ReferenceScalar),
    /// An exact signed mathematical quantity, independent of source word width.
    Integer(BigInt),
    Index(BigUint),
    Range(BigUint, BigUint),
    Tensor(TensorValue),
    Tuple(Vec<Value>),
    Void,
}

impl Value {
    pub(super) fn scalar(value: ReferenceScalar) -> Self {
        Self::Scalar(value)
    }

    pub(super) fn as_scalar(&self) -> ReferenceScalar {
        match self {
            Self::Scalar(value) => *value,
            _ => unreachable!("checked scalar operand is not a scalar value"),
        }
    }

    pub(super) fn as_nat(&self) -> BigUint {
        let natural = match self {
            Self::Index(value) => Some(value.clone()),
            Self::Integer(value) => value.to_biguint(),
            Self::Scalar(ReferenceScalar::U32(value)) => Some((*value).into()),
            Self::Scalar(ReferenceScalar::I32(value)) => value.to_biguint(),
            _ => unreachable!("checked natural operand is not a quantity or word"),
        };
        natural.unwrap_or_else(|| unreachable!("checked natural operand is negative"))
    }

    pub(super) fn as_nat_usize(&self) -> usize {
        self.as_nat()
            .to_usize()
            .unwrap_or_else(|| unreachable!("checked in-bounds natural exceeds the address width"))
    }

    pub(super) fn as_integer(&self) -> BigInt {
        match self {
            Self::Integer(value) => value.clone(),
            Self::Index(value) => BigInt::from(value.clone()),
            Self::Scalar(ReferenceScalar::I32(value)) => (*value).into(),
            Self::Scalar(ReferenceScalar::U32(value)) => (*value).into(),
            _ => unreachable!("checked integer operand is not a quantity or word"),
        }
    }

    pub(super) fn as_tensor(&self) -> &TensorValue {
        match self {
            Self::Tensor(value) => value,
            _ => unreachable!("checked tensor operand is not a tensor value"),
        }
    }
}
