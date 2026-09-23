//! Scalar dtypes (public) and the checker's canonical value types
//! (crate-private).
//!
//! `ValueType` is the type language of the checked representation: interfaces,
//! calls, and bodies. Its extents are `IntExpr` handles into the owning
//! definition's expression arena; it never leaves the crate. The public
//! monomorphized form is `entry::SemanticType`, built by the entry builder.
//! Ownership (owned tensor / shared borrow / exclusive borrow) is a signature
//! property carried next to the type, never a type variant.

use crate::expr::{ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, RepresentationId};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F32,
    BF16,
    F16,
    I32,
    U32,
    Bool,
}

impl DType {
    pub const ALL: [DType; 6] = [
        DType::F32,
        DType::BF16,
        DType::F16,
        DType::I32,
        DType::U32,
        DType::Bool,
    ];

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

    /// Widening for arithmetic between two dtypes: exact for the narrow
    /// floats, none across kinds.
    pub fn promote(a: DType, b: DType) -> Option<DType> {
        if a == b {
            return Some(a);
        }
        if a.is_float() && b.is_float() {
            return Some(DType::F32);
        }
        None
    }

    /// Wire ordinal, stable across builds.
    pub(crate) fn ordinal(self) -> u8 {
        match self {
            DType::F32 => 0,
            DType::BF16 => 1,
            DType::F16 => 2,
            DType::I32 => 3,
            DType::U32 => 4,
            DType::Bool => 5,
        }
    }

    pub(crate) fn from_ordinal(ordinal: u8) -> Option<DType> {
        DType::ALL.get(usize::from(ordinal)).copied()
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Element type of a tensor at checked scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Elem {
    Dtype(DType),
    /// A packed representation. Reading one element yields its decoded value.
    Repr(RepresentationId),
    /// An element parameter of the enclosing declaration (`T`, `U`); bound to
    /// a representation at monomorphization.
    Param(String),
}

impl Elem {
    /// The dtype a read of one element produces at portable scope. Packed
    /// elements decode to their registry `decoded` dtype; an element
    /// parameter reads as `f32` at portable scope.
    pub(crate) fn read_dtype(&self) -> DType {
        match self {
            Elem::Dtype(d) => *d,
            Elem::Repr(id) => crate::registry::representation_info(*id).decoded,
            Elem::Param(_) => DType::F32,
        }
    }

    /// The dense dtype an elementwise operation sees, or `None` for packed
    /// storage (which must be decoded first).
    pub(crate) fn dense_dtype(&self) -> Option<DType> {
        match self {
            Elem::Dtype(d) => Some(*d),
            Elem::Param(_) => Some(DType::F32),
            Elem::Repr(_) => None,
        }
    }

    pub(crate) fn is_packed(&self) -> bool {
        matches!(self, Elem::Repr(_))
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Elem::Dtype(d) => write!(f, "{}", d.name()),
            Elem::Repr(r) => write!(f, "{}", crate::registry::representation_info(*r).name),
            Elem::Param(p) => write!(f, "{p}"),
        }
    }
}

/// A nonempty list; tuple components are never empty (an empty source result
/// canonicalizes to `Void`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NonEmpty<T>(Vec<T>);

impl<T> NonEmpty<T> {
    pub(crate) fn new(items: Vec<T>) -> Option<NonEmpty<T>> {
        (!items.is_empty()).then(|| NonEmpty(items))
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

/// Semantic shape of a tensor value: one semantic leaf, one or more physical
/// planes when the element is a packed representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TensorType {
    /// Axis extents, symbolic over the definition's dimensions and body
    /// symbols, in the definition's arena.
    pub axes: Vec<IntExpr>,
    pub elem: Elem,
    /// For a packed representation: the axis along which packets run.
    pub packed_axis: Option<usize>,
}

impl TensorType {
    pub(crate) fn new(axes: Vec<IntExpr>, elem: Elem) -> TensorType {
        let packed_axis = match &elem {
            Elem::Repr(_) => axes.len().checked_sub(1),
            _ => None,
        };
        TensorType {
            axes,
            elem,
            packed_axis,
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.axes.len()
    }
}

/// The canonical value type of the checked representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ValueType {
    Scalar(DType),
    /// Exact signed mathematical quantity.
    Integer,
    /// Mathematical quantity refined to `0 <= i < bound`.
    Index {
        bound: IntExpr,
    },
    /// A bounded logical half-open range value.
    Range {
        bound: IntExpr,
    },
    Tensor(TensorType),
    Tuple(NonEmpty<ValueType>),
    /// A backend-opaque intrinsic value.
    Opaque {
        capability: CapabilityId,
        name: &'static str,
    },
    /// The canonical form of an empty result or tuple.
    Void,
}

impl ValueType {
    pub(crate) fn shaped(&self) -> Option<&TensorType> {
        match self {
            ValueType::Tensor(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn scalar_dtype(&self) -> Option<DType> {
        match self {
            ValueType::Scalar(d) => Some(*d),
            _ => None,
        }
    }

    pub(crate) fn is_void(&self) -> bool {
        matches!(self, ValueType::Void)
    }

    /// Every extent handle mentioned by this type, in traversal order.
    pub(crate) fn extents(&self, out: &mut Vec<IntExpr>) {
        match self {
            ValueType::Scalar(_) | ValueType::Integer | ValueType::Opaque { .. } | ValueType::Void => {}
            ValueType::Index { bound } | ValueType::Range { bound } => out.push(*bound),
            ValueType::Tensor(t) => out.extend(t.axes.iter().copied()),
            ValueType::Tuple(items) => items.iter().for_each(|item| item.extents(out)),
        }
    }
}

impl ValueType {
    /// The type with its shape expressions, in source spelling: for example
    /// `tensor[N - 1] f32` or `index[B]` (L23 c). `name` names the arena's
    /// symbols.
    pub(crate) fn with_shapes<'a>(
        &'a self,
        arena: &'a ExprArena,
        name: &'a dyn Fn(SymbolId) -> String,
    ) -> ShapedValueType<'a> {
        ShapedValueType {
            ty: self,
            extents: Some((arena, name)),
        }
    }
}

/// A `ValueType` rendered in source spelling. Without an arena, every shape
/// expression is written `_`.
pub(crate) struct ShapedValueType<'a> {
    ty: &'a ValueType,
    extents: Option<(&'a ExprArena, &'a dyn Fn(SymbolId) -> String)>,
}

impl ShapedValueType<'_> {
    fn extent(&self, extent: IntExpr) -> String {
        match self.extents {
            Some((arena, name)) => crate::check::prove::display(arena, extent, name),
            None => "_".to_string(),
        }
    }
}

impl fmt::Display for ShapedValueType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            ValueType::Scalar(dtype) => write!(f, "{dtype}"),
            ValueType::Integer => f.write_str("integer"),
            ValueType::Index { bound } => write!(f, "index[{}]", self.extent(*bound)),
            ValueType::Range { bound } => write!(f, "range[{}]", self.extent(*bound)),
            ValueType::Tensor(tensor) => {
                let axes = tensor
                    .axes
                    .iter()
                    .map(|axis| self.extent(*axis))
                    .collect::<Vec<_>>();
                write!(f, "tensor[{}] {}", axes.join(", "), tensor.elem)
            }
            ValueType::Tuple(items) => {
                f.write_str("(")?;
                for (index, item) in items.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    let item = ShapedValueType {
                        ty: item,
                        extents: self.extents,
                    };
                    write!(f, "{item}")?;
                }
                f.write_str(")")
            }
            ValueType::Opaque { name, .. } => write!(f, "opaque<{name}>"),
            ValueType::Void => f.write_str("void"),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ShapedValueType {
            ty: self,
            extents: None,
        }
        .fmt(f)
    }
}

/// Ordinal path of one component of a value: tuple nesting and ordinal paths
/// are preserved; names never define identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct ValuePath(pub Vec<u32>);

impl ValuePath {
    pub(crate) fn extend(&self, index: u32) -> ValuePath {
        let mut out = self.0.clone();
        out.push(index);
        ValuePath(out)
    }
}

/// One semantic leaf of a canonical type: a range is one leaf, a tensor is
/// one leaf. Opaque values are not leaves: they cannot cross a portable
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Leaf<'a> {
    Scalar(DType),
    Integer,
    Index(IntExpr),
    Range(IntExpr),
    Tensor(&'a TensorType),
}

/// The one canonical leaf traversal. `None` names an opaque value somewhere
/// in the type. `Void` has no leaves.
pub(crate) fn canonical_leaves(ty: &ValueType) -> Option<Vec<(ValuePath, Leaf<'_>)>> {
    fn walk<'a>(
        ty: &'a ValueType,
        path: &ValuePath,
        out: &mut Vec<(ValuePath, Leaf<'a>)>,
    ) -> Option<()> {
        match ty {
            ValueType::Scalar(d) => out.push((path.clone(), Leaf::Scalar(*d))),
            ValueType::Integer => out.push((path.clone(), Leaf::Integer)),
            ValueType::Index { bound } => out.push((path.clone(), Leaf::Index(*bound))),
            ValueType::Range { bound } => out.push((path.clone(), Leaf::Range(*bound))),
            ValueType::Tensor(s) => out.push((path.clone(), Leaf::Tensor(s))),
            ValueType::Tuple(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, &path.extend(i as u32), out)?;
                }
            }
            ValueType::Opaque { .. } => return None,
            ValueType::Void => {}
        }
        Some(())
    }
    let mut out = Vec::new();
    walk(ty, &ValuePath::default(), &mut out)?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_render_with_their_shape_expressions() {
        let mut arena = ExprArena::new();
        let (n, n_expr) = arena.template_dimension(0);
        let one = arena.int(1);
        let shorter = arena.int_sub(n_expr, one);
        let name = |symbol: SymbolId| if symbol == n { "N".to_string() } else { format!("{symbol:?}") };
        let tensor = |axis| ValueType::Tensor(TensorType::new(vec![axis], Elem::Dtype(DType::F32)));
        assert_eq!(tensor(n_expr).with_shapes(&arena, &name).to_string(), "tensor[N] f32");
        // Extents print in the prover's normal form, as every shape diagnostic does.
        assert_eq!(tensor(shorter).with_shapes(&arena, &name).to_string(), "tensor[-1 + N] f32");
        assert_eq!(
            ValueType::Index { bound: n_expr }.with_shapes(&arena, &name).to_string(),
            "index[N]"
        );
        assert_eq!(tensor(shorter).to_string(), "tensor[_] f32");
        assert_eq!(ValueType::Scalar(DType::I32).to_string(), "i32");
    }
}
