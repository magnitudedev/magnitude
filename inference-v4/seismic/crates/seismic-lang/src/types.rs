//! Types of the language.

use crate::sym::Sym;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    BF16,
    F16,
    I32,
    U32,
    Bool,
}

impl DType {
    pub fn from_name(name: &str) -> Option<DType> {
        Some(match name {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "bool" => DType::Bool,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::BF16 | DType::F16)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }

    pub fn is_numeric(self) -> bool {
        self.is_float() || self.is_int()
    }

    pub fn bytes(self) -> u32 {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::BF16 | DType::F16 => 2,
            DType::Bool => 1,
        }
    }

    /// Widening for arithmetic between two dtypes: exact for the narrow floats, none across kinds.
    pub fn promote(a: DType, b: DType) -> Option<DType> {
        if a == b {
            return Some(a);
        }
        if a.is_float() && b.is_float() {
            return Some(DType::F32);
        }
        None
    }
}

/// Element type of a tensor or tile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    Dtype(DType),
    /// A packed representation such as `q4g64`. Reading an element yields its decoded value.
    Repr(String),
    /// A dtype parameter of the enclosing declaration (`T`, `U`).
    Param(String),
}

impl Elem {
    /// The dtype a read of one element produces at portable scope.
    pub fn read_dtype(&self) -> Option<DType> {
        match self {
            Elem::Dtype(d) => Some(*d),
            Elem::Repr(_) => Some(DType::F32),
            Elem::Param(_) => None,
        }
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Elem::Dtype(d) => write!(f, "{}", d.name()),
            Elem::Repr(r) => write!(f, "{r}"),
            Elem::Param(p) => write!(f, "{p}"),
        }
    }
}

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
