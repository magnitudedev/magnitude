//! Runtime values of the reference interpreter.
use crate::types::DType;
use std::cell::RefCell;
use std::rc::Rc;

/// A scalar with the dtype it was produced at. Integers and bools are carried exactly.
pub(super) type S = (DType, f64);

#[derive(Debug)]
pub struct Dense {
    pub dtype: DType,
    pub data: Vec<f64>,
    pub init: Vec<bool>,
}

#[derive(Clone, Debug)]
pub enum Backing {
    /// Index into the interpreter's tensor table. Packed tensors are immutable, so a packed
    /// snapshot is a descriptor over its tensor.
    Tensor(usize),
    Owned(Rc<RefCell<Dense>>),
}

/// A strided selection of a backing store. Positions are zero-based.
#[derive(Clone, Debug)]
pub struct Shaped {
    pub backing: Backing,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Shaped {
    pub fn tensor(id: usize, shape: &[usize]) -> Shaped {
        Shaped {
            backing: Backing::Tensor(id),
            shape: shape.to_vec(),
            strides: row_major(shape),
            offset: 0,
        }
    }

    pub fn owned(dtype: DType, shape: Vec<usize>, data: Vec<f64>) -> Shaped {
        let init = vec![true; data.len()];
        Shaped::from_dense(shape, Dense { dtype, data, init })
    }

    pub fn uninit(dtype: DType, shape: Vec<usize>) -> Shaped {
        let n = shape.iter().product();
        Shaped::from_dense(
            shape,
            Dense {
                dtype,
                data: vec![f64::NAN; n],
                init: vec![false; n],
            },
        )
    }

    fn from_dense(shape: Vec<usize>, dense: Dense) -> Shaped {
        Shaped {
            strides: row_major(&shape),
            shape,
            offset: 0,
            backing: Backing::Owned(Rc::new(RefCell::new(dense))),
        }
    }

    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Flat backing positions in row-major order of this selection.
    pub fn flats(&self) -> Vec<usize> {
        let n = self.count();
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.shape.len()];
        let mut flat = self.offset;
        for _ in 0..n {
            out.push(flat);
            let mut k = idx.len();
            while k > 0 {
                k -= 1;
                idx[k] += 1;
                flat += self.strides[k];
                if idx[k] < self.shape[k] {
                    break;
                }
                flat -= self.strides[k] * idx[k];
                idx[k] = 0;
            }
        }
        out
    }

    /// Whether this selection is exactly its owned buffer in row-major order, unshared.
    pub fn exclusive(&self) -> bool {
        match &self.backing {
            Backing::Owned(rc) => {
                Rc::strong_count(rc) == 1
                    && self.offset == 0
                    && self.strides == row_major(&self.shape)
                    && rc.borrow().data.len() == self.count()
            }
            Backing::Tensor(_) => false,
        }
    }

    pub fn transposed(&self) -> Result<Shaped, String> {
        if self.shape.len() != 2 {
            return Err(format!("transpose of a rank-{} value", self.shape.len()));
        }
        Ok(Shaped {
            backing: self.backing.clone(),
            shape: vec![self.shape[1], self.shape[0]],
            strides: vec![self.strides[1], self.strides[0]],
            offset: self.offset,
        })
    }
}

pub(super) fn row_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[derive(Clone, Debug)]
pub enum Value {
    Scalar(DType, f64),
    Range(i64, i64),
    Tensor(Shaped),
    Tuple(Vec<Value>),
    Void,
}

impl Value {
    pub fn int(v: i64) -> Value {
        Value::Scalar(DType::I32, v as f64)
    }

    pub fn scalar(s: S) -> Value {
        Value::Scalar(s.0, s.1)
    }

    pub fn shaped(&self) -> Option<&Shaped> {
        match self {
            Value::Tensor(s) => Some(s),
            _ => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::Scalar(..) => "a scalar",
            Value::Range(..) => "a range",
            Value::Tensor(_) => "a tensor value",
            Value::Tuple(_) => "a tuple",
            Value::Void => "void",
        }
    }
}

pub(super) enum Flow {
    Next,
    Return(Value),
}
