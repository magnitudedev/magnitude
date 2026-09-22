use super::TensorData;
use crate::ids::RepresentationId;
use crate::types::DType;
use std::cell::RefCell;
use std::rc::Rc;

pub(super) type Scalar = (DType, f64);

#[derive(Clone, Debug)]
pub(super) enum Backing {
    Argument(usize),
    Owned(Rc<RefCell<TensorData>>),
}

/// A logical strided view. Storage representations stay on the backing; a
/// view never manufactures a second physical interpretation.
#[derive(Clone, Debug)]
pub struct TensorValue {
    pub(super) backing: Backing,
    pub(super) representation: RepresentationId,
    pub(super) shape: Vec<usize>,
    /// Backing-flat position of every logical row-major element. Keeping the
    /// logical index map explicit makes arbitrary compositions of
    /// slice/transpose/reshape exact without inventing backend view rules.
    pub(super) positions: Vec<usize>,
}

impl TensorValue {
    pub(super) fn argument(id: usize, representation: RepresentationId, shape: &[usize]) -> Self {
        Self {
            backing: Backing::Argument(id),
            representation,
            shape: shape.to_vec(),
            positions: (0..shape.iter().product()).collect(),
        }
    }

    pub(super) fn owned(tensor: TensorData) -> Self {
        let representation = tensor.representation();
        let shape = tensor.shape().to_vec();
        Self {
            backing: Backing::Owned(Rc::new(RefCell::new(tensor))),
            representation,
            positions: (0..shape.iter().product()).collect(),
            shape,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub(super) fn flat_positions(&self) -> Vec<usize> {
        self.positions.clone()
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
    Scalar(DType, f64),
    Range(i64, i64),
    Tensor(TensorValue),
    Tuple(Vec<Value>),
    Void,
}

impl Value {
    pub(super) fn scalar(value: Scalar) -> Self {
        Self::Scalar(value.0, value.1)
    }

    pub(super) fn as_scalar(&self) -> Result<Scalar, String> {
        match self {
            Self::Scalar(dtype, value) => Ok((*dtype, *value)),
            _ => Err("semantic value is not scalar".to_owned()),
        }
    }

    pub(super) fn as_tensor(&self) -> Result<&TensorValue, String> {
        match self {
            Self::Tensor(value) => Ok(value),
            _ => Err("semantic value is not a tensor".to_owned()),
        }
    }
}
