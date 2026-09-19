//! Types of the execution IR: every extent is a concrete or symbolic number.

use crate::sym::Sym;
use crate::types::{DType, Elem};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shaped {
    pub shape: Vec<Sym>,
    pub elem: Elem,
    /// For a packed element type: the axis along which packets run. A tensor packs its last
    /// axis; views and transposes carry the axis along.
    pub packed_axis: Option<usize>,
}

impl Shaped {
    pub fn new(shape: Vec<Sym>, elem: Elem) -> Shaped {
        let packed_axis = match elem {
            Elem::Repr(_) => Some(shape.len().saturating_sub(1)),
            _ => None,
        };
        Shaped { shape, elem, packed_axis }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Scalar(DType),
    /// Global storage or a view of it.
    Tensor(Shaped),
    /// A logical block, or a view of one.
    Tile(Shaped),
    /// A fragment operand of a backend intrinsic (lowering scope only).
    Frag(Shaped),
    Tuple(Vec<Ty>),
    Void,
}

impl Ty {
    pub fn scalar(d: DType) -> Ty {
        Ty::Scalar(d)
    }

    pub fn shaped(&self) -> Option<&Shaped> {
        match self {
            Ty::Tensor(s) | Ty::Tile(s) | Ty::Frag(s) => Some(s),
            _ => None,
        }
    }

    pub fn rank(&self) -> Option<usize> {
        self.shaped().map(|s| s.shape.len())
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn shape(s: &Shaped) -> String {
            format!("[{}] {}", s.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "), s.elem)
        }
        match self {
            Ty::Scalar(d) => write!(f, "{}", d.name()),
            Ty::Tensor(s) => write!(f, "tensor{}", shape(s)),
            Ty::Tile(s) => write!(f, "tile{}", shape(s)),
            Ty::Frag(s) => write!(f, "frag{}", shape(s)),
            Ty::Tuple(items) => write!(f, "({})", items.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ")),
            Ty::Void => write!(f, "void"),
        }
    }
}
