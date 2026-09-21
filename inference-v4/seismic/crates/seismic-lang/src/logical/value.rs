//! Logical values and logical storage ownership (package L1).
//!
//! C0 freezes this vocabulary; L1 owns its construction. A logical value has
//! exactly one exhaustive kind. A direct tensor value is either a computed
//! tensor or a view: the distinction is a variant of `TensorSource`, never
//! an `Option`. A view's base is itself exhaustive (`ViewBase`): one logical
//! storage, or one computed tensor value — a view of a computed tensor is a
//! legitimate tensor value and invents no storage. Constructors are private
//! to logical construction (`super::builder`); every other crate reads
//! through the immutable accessors. A tensor value carries no optional
//! provenance and a non-tensor value carries no view; a value of every graph
//! is defined exactly once in that graph's value table (`TaskGraph::value`).

use crate::types::{CapabilityValueType, DType, ExtentExpr, NonEmpty, TensorType, ValueType};

use super::boundary::BoundaryLeaf;
use super::{GraphValueId, LogicalStorageId, LogicalViewId};

/// Where a direct tensor value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorSource {
    /// Produced by an operation; has no logical storage. Whether it needs a
    /// physical spill is a later physical (D1) decision.
    Computed,
    /// A view, of logical storage or of a computed tensor value.
    View(LogicalViewId),
}

/// The ultimate origin a view reads. A view declared over another view
/// flattens to its operand's base, so every view names exactly one storage
/// or one computed tensor value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewBase {
    Storage(LogicalStorageId),
    /// A computed tensor value; a view of it is read-only and needs no
    /// physical spill decision of its own (D1 spills the base value).
    Value(GraphValueId),
}

/// The exhaustive kind of one logical value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GraphValueKind {
    Void,
    Scalar(DType),
    Index { bound: ExtentExpr },
    Range { bound: ExtentExpr },
    Tuple(NonEmpty<ValueType>),
    Capability(CapabilityValueType),
    Tensor { ty: TensorType, source: TensorSource },
}

/// One SSA graph value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphValue {
    id: GraphValueId,
    kind: GraphValueKind,
}

impl GraphValue {
    /// The only constructor; private to logical construction.
    pub(super) fn new(id: GraphValueId, kind: GraphValueKind) -> GraphValue {
        GraphValue { id, kind }
    }

    pub fn id(&self) -> GraphValueId {
        self.id
    }

    pub fn kind(&self) -> &GraphValueKind {
        &self.kind
    }

    /// The sole conversion back to the canonical type language.
    pub fn ty(&self) -> ValueType {
        match &self.kind {
            GraphValueKind::Void => ValueType::Void,
            GraphValueKind::Scalar(dtype) => ValueType::Scalar(*dtype),
            GraphValueKind::Index { bound } => ValueType::Index {
                bound: bound.clone(),
            },
            GraphValueKind::Range { bound } => ValueType::Range {
                bound: bound.clone(),
            },
            GraphValueKind::Tuple(items) => ValueType::Tuple(items.clone()),
            GraphValueKind::Capability(ty) => ValueType::CapabilityValue(ty.clone()),
            GraphValueKind::Tensor { ty, .. } => ValueType::Tensor(ty.clone()),
        }
    }

    /// The tensor source of a tensor value; `None` for every non-tensor kind
    /// (which is an exhaustive statement about the kind, not a missing fact).
    pub fn tensor_source(&self) -> Option<TensorSource> {
        match &self.kind {
            GraphValueKind::Tensor { source, .. } => Some(*source),
            GraphValueKind::Void
            | GraphValueKind::Scalar(_)
            | GraphValueKind::Index { .. }
            | GraphValueKind::Range { .. }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Capability(_) => None,
        }
    }
}

/// Who owns one logical storage. Returning a value is a boundary fact
/// (`LogicalBoundary::results`), never a storage provenance: there is no
/// `Result` owner.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicalStorageOwner {
    /// Storage of one tensor leaf of one interface parameter.
    Parameter(BoundaryLeaf),
    /// Storage allocated inside the graph that owns it.
    Local,
}
