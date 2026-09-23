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

    pub(super) fn as_scalar(&self) -> Result<ReferenceScalar, String> {
        match self {
            Self::Scalar(value) => Ok(*value),
            _ => Err("semantic value is not scalar".to_owned()),
        }
    }

    pub(super) fn as_nat(&self) -> Result<BigUint, String> {
        match self {
            Self::Index(value) => Ok(value.clone()),
            Self::Integer(value) => value.to_biguint().ok_or_else(|| "negative natural value".to_owned()),
            Self::Scalar(ReferenceScalar::U32(value)) => Ok((*value).into()),
            Self::Scalar(ReferenceScalar::I32(value)) => value.to_biguint().ok_or_else(|| "negative natural value".to_owned()),
            _ => Err("semantic value is not natural".to_owned()),
        }
    }

    pub(super) fn as_nat_usize(&self) -> Result<usize, String> {
        self.as_nat()?.to_usize().ok_or_else(|| "natural value exceeds address width".to_owned())
    }

    pub(super) fn as_integer(&self) -> Result<BigInt, String> {
        match self {
            Self::Integer(value) => Ok(value.clone()),
            Self::Index(value) => Ok(BigInt::from(value.clone())),
            Self::Scalar(ReferenceScalar::I32(value)) => Ok((*value).into()),
            Self::Scalar(ReferenceScalar::U32(value)) => Ok((*value).into()),
            _ => Err("semantic value is not integer".to_owned()),
        }
    }

    pub(super) fn as_tensor(&self) -> Result<&TensorValue, String> {
        match self {
            Self::Tensor(value) => Ok(value),
            _ => Err("semantic value is not a tensor".to_owned()),
        }
    }
}
