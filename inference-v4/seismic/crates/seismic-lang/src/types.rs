//! Types of structured Seismic. Semantic extents, structural extents and native
//! geometry are distinct authorities and never collapse into one integer.

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

/// A static region binder within one definition body (index into `sir::Body::slices`).
/// All dynamic visits of the binder share this identity and one numerical site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceId(pub u32);

/// A static region within one definition body (index into `sir::Body::regions`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub u32);

/// Extent of one axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    /// A problem dimension; numerically usable.
    Semantic(Sym),
    /// The width of a slice. Never a source number in portable code. A shape parameter of a
    /// helper bound to a structural argument is substituted by this at the call occurrence.
    Structural(SliceId),
}

impl Extent {
    pub fn semantic(&self) -> Option<&Sym> {
        match self {
            Extent::Semantic(s) => Some(s),
            Extent::Structural(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shaped {
    pub axes: Vec<Extent>,
    pub elem: Elem,
    /// For a packed representation: the axis along which packets run.
    pub packed_axis: Option<usize>,
}

impl Shaped {
    pub fn new(axes: Vec<Extent>, elem: Elem) -> Shaped {
        let packed_axis = match elem {
            Elem::Repr(_) => Some(axes.len().saturating_sub(1)),
            _ => None,
        };
        Shaped {
            axes,
            elem,
            packed_axis,
        }
    }

    pub fn rank(&self) -> usize {
        self.axes.len()
    }
}

/// One value per visit of the producing region, keeping its slice correspondence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultTy {
    /// The region that introduced the partition. A region traversing earlier results and
    /// yielding new values has a new `producer` but the same `origin`.
    pub origin: RegionId,
    pub producer: RegionId,
    /// The origin's binders, in order. Consumers must bind the same arity.
    pub binders: Vec<SliceId>,
    /// Member schema; structural axes refer to `binders` (or enclosing slices).
    pub member: Ty,
}

/// A backend-owned opaque operand (`metal.simdgroup_matrix(f32)`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeTy {
    pub target: String,
    pub name: String,
    pub shape: Vec<Sym>,
    pub elem: Option<Elem>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Scalar(DType),
    /// `i32` refined to `0 <= i < bound`.
    Index(Sym),
    /// External storage.
    Tensor(Shaped),
    /// Borrowed selection of a tensor or tile.
    View(Shaped),
    /// Owned logical block.
    Tile(Shaped),
    /// `range[N]`, retaining its semantic upper bound.
    Range(Sym),
    Slice(SliceId),
    /// Tile coordinate over a structural axis (binder of `owned`/`axis`). Coordinates over
    /// semantic axes are `Index`.
    Coord(SliceId),
    Result(Box<ResultTy>),
    Tuple(Vec<Ty>),
    Void,
    Native(NativeTy),
}

impl Ty {
    pub fn shaped(&self) -> Option<&Shaped> {
        match self {
            Ty::Tensor(s) | Ty::View(s) | Ty::Tile(s) => Some(s),
            _ => None,
        }
    }

    pub fn scalar_dtype(&self) -> Option<DType> {
        match self {
            Ty::Scalar(d) => Some(*d),
            Ty::Index(_) => Some(DType::I32),
            _ => None,
        }
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Extent::Semantic(s) => write!(f, "{s}"),
            Extent::Structural(s) => write!(f, "slice#{}", s.0),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn shape(s: &Shaped) -> String {
            format!(
                "[{}] {}",
                s.axes
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                s.elem
            )
        }
        match self {
            Ty::Scalar(d) => write!(f, "{}", d.name()),
            Ty::Index(n) => write!(f, "index[{n}]"),
            Ty::Tensor(s) => write!(f, "tensor{}", shape(s)),
            Ty::View(s) => write!(f, "view{}", shape(s)),
            Ty::Tile(s) => write!(f, "tile{}", shape(s)),
            Ty::Range(bound) => write!(f, "range[{bound}]"),
            Ty::Slice(s) => write!(f, "slice#{}", s.0),
            Ty::Coord(s) => write!(f, "coord(slice#{})", s.0),
            Ty::Result(r) => write!(f, "result#{}<{}>", r.origin.0, r.member),
            Ty::Tuple(items) => write!(
                f,
                "({})",
                items
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Ty::Void => write!(f, "void"),
            Ty::Native(n) => write!(f, "{}.{}", n.target, n.name),
        }
    }
}
