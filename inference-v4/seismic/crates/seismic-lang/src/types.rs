//! Canonical value types, paths, and the canonical leaf traversal.
//!
//! `ValueType` is the single canonical type language used by interfaces, calls,
//! the ABI, result allocation, and binding. Ownership (owned tensor / shared
//! view / exclusive mutable view) is a *signature* property carried next to the
//! type, never a type variant: a tensor is one semantic leaf however it is
//! accessed.

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

/// Element type of a tensor.
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

/// Identity of a runtime-determined extent. At the logical level a
/// `RuntimeExtent` carries its value and capacity; at the checked
/// level extents are symbolic over shape parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeExtentId(pub u32);

/// Extent of one axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtentExpr {
    Static(u64),
    /// A symbolic expression over shape parameters (the checked-level form).
    Sym(Sym),
    /// A runtime extent allocated by logical construction.
    Runtime(RuntimeExtentId),
}

impl ExtentExpr {
    pub fn as_static(&self) -> Option<u64> {
        match self {
            ExtentExpr::Static(n) => Some(*n),
            ExtentExpr::Sym(s) => s.as_constant().and_then(|c| u64::try_from(c).ok()),
            ExtentExpr::Runtime(_) => None,
        }
    }

    pub fn sym(&self) -> Option<&Sym> {
        match self {
            ExtentExpr::Sym(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for ExtentExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtentExpr::Static(n) => write!(f, "{n}"),
            ExtentExpr::Sym(s) => write!(f, "{s}"),
            ExtentExpr::Runtime(id) => write!(f, "runtime#{}", id.0),
        }
    }
}

/// A nonempty list; tuple components are never empty (an empty source result
/// canonicalizes to `Void`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NonEmpty<T>(Vec<T>);

impl<T> NonEmpty<T> {
    pub fn new(items: Vec<T>) -> Option<NonEmpty<T>> {
        (!items.is_empty()).then(|| NonEmpty(items))
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn first(&self) -> &T {
        &self.0[0]
    }
}

/// Semantic shape of a tensor value: one semantic leaf, one or more physical
/// planes when the element is a packed representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub axes: Vec<ExtentExpr>,
    pub elem: Elem,
    /// For a packed representation: the axis along which packets run.
    pub packed_axis: Option<usize>,
}

impl TensorType {
    pub fn new(axes: Vec<ExtentExpr>, elem: Elem) -> TensorType {
        let packed_axis = match elem {
            Elem::Repr(_) => Some(axes.len().saturating_sub(1)),
            _ => None,
        };
        TensorType {
            axes,
            elem,
            packed_axis,
        }
    }

    pub fn rank(&self) -> usize {
        self.axes.len()
    }
}

/// A backend-owned opaque value (`metal.simdgroup_matrix` fragments). Capability
/// values cannot cross portable boundaries or the public ABI.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapabilityValueType {
    pub target: String,
    pub name: String,
    pub shape: Vec<ExtentExpr>,
    pub elem: Option<Elem>,
}

/// The canonical value type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar(DType),
    /// `i32` refined to `0 <= i < bound`.
    Index {
        bound: ExtentExpr,
    },
    /// A bounded logical half-open range value.
    Range {
        bound: ExtentExpr,
    },
    Tensor(TensorType),
    Tuple(NonEmpty<ValueType>),
    CapabilityValue(CapabilityValueType),
    /// The canonical form of an empty result or tuple. Creates no graph data
    /// value or storage; a void boundary retains transport/completion only.
    Void,
}

impl ValueType {
    pub fn shaped(&self) -> Option<&TensorType> {
        match self {
            ValueType::Tensor(s) => Some(s),
            _ => None,
        }
    }

    pub fn scalar_dtype(&self) -> Option<DType> {
        match self {
            ValueType::Scalar(d) => Some(*d),
            ValueType::Index { .. } => Some(DType::I32),
            _ => None,
        }
    }

    pub fn is_void(&self) -> bool {
        matches!(self, ValueType::Void)
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn shape(s: &TensorType) -> String {
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
            ValueType::Scalar(d) => write!(f, "{}", d.name()),
            ValueType::Index { bound } => write!(f, "index[{bound}]"),
            ValueType::Range { bound } => write!(f, "range[{bound}]"),
            ValueType::Tensor(s) => write!(f, "tensor{}", shape(s)),
            ValueType::Tuple(items) => write!(
                f,
                "({})",
                items
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ValueType::CapabilityValue(n) => write!(f, "{}.{}", n.target, n.name),
            ValueType::Void => write!(f, "void"),
        }
    }
}

/// Ordinal path of one component of a value: tuple nesting and ordinal paths
/// are preserved; names never define identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ValuePath(pub Vec<u32>);

impl fmt::Display for ValuePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("()");
        }
        let joined = self.0.iter().map(|i| i.to_string()).collect::<Vec<_>>();
        joined.join(".").fmt(f)
    }
}

impl ValuePath {
    pub fn extend(&self, index: u32) -> ValuePath {
        let mut out = self.0.clone();
        out.push(index);
        ValuePath(out)
    }
}

/// One semantic leaf of a canonical type, as seen by interfaces, calls, the
/// ABI, result allocation, and binding. Range is one semantic leaf (two ABI
/// scalar fields); a tensor is one semantic leaf (one or more planes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf<'a> {
    Scalar(DType),
    Index(&'a ExtentExpr),
    Range(&'a ExtentExpr),
    Tensor(&'a TensorType),
}

/// The one canonical leaf traversal. `Err` names a capability value, which
/// cannot cross a portable boundary or the public ABI. `Void` has no leaves.
pub fn canonical_leaves(ty: &ValueType) -> Result<Vec<(ValuePath, Leaf<'_>)>, String> {
    let mut out = Vec::new();
    fn walk<'a>(
        ty: &'a ValueType,
        path: &ValuePath,
        out: &mut Vec<(ValuePath, Leaf<'a>)>,
    ) -> Result<(), String> {
        match ty {
            ValueType::Scalar(d) => out.push((path.clone(), Leaf::Scalar(*d))),
            ValueType::Index { bound } => out.push((path.clone(), Leaf::Index(bound))),
            ValueType::Range { bound } => out.push((path.clone(), Leaf::Range(bound))),
            ValueType::Tensor(s) => out.push((path.clone(), Leaf::Tensor(s))),
            ValueType::Tuple(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &path.extend(i as u32), out)?;
                }
            }
            ValueType::CapabilityValue(n) => {
                return Err(format!(
                    "a `{}.{}` capability value cannot cross a portable boundary or the public ABI",
                    n.target, n.name
                ))
            }
            ValueType::Void => {}
        }
        Ok(())
    }
    walk(ty, &ValuePath::default(), &mut out)?;
    Ok(out)
}
