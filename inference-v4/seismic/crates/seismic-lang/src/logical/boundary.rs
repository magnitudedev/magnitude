//! Logical boundary contracts (package L1).
//!
//! Every function alternative owns one explicit boundary contract generated
//! from its checked interface and normalized body. Roots and calls
//! instantiate the same schema: a root instantiates it with ABI residences, a
//! call instantiates it with caller value/state identities (package O1).
//! There is no root/open/call-specific path inference, no positional result
//! reconstruction, and no parameter/result cursor.
//!
//! `BoundaryLeaf` moves here from `seismic-realization::executable` because a
//! boundary leaf is a logical fact. The realization crate consumes it.
//!
//! A call instantiates the callee's `LogicalBoundary` schema with the
//! caller's own value/state identities as a `CallBoundary`: the same leaf
//! keys, the same ownership modes, the same separation of returned values
//! from final mutable states. Call results are produced values of the caller
//! (`GraphValueKind::Tensor { source: Computed }` for tensor leaves); no
//! caller-side storage is fabricated for them.

use crate::sir::ParamOwnership;
use crate::types::ValuePath;
use std::collections::BTreeMap;

use super::{GraphValueId, StateTokenId};

/// One canonical boundary leaf, qualified by its interface parameter so that
/// same-shaped parameters never collide. Result leaves are keyed by their
/// canonical path in the interface result type.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundaryLeaf {
    Input { param: u32, leaf: ValuePath },
    Result { leaf: ValuePath },
}

/// One input leaf of a boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalBoundaryInput {
    /// A plain value leaf (scalar, index, range, capability value).
    Value(GraphValueId),
    /// A tensor leaf: the value, its entry state, and the ownership mode the
    /// interface declares for it.
    Tensor {
        value: GraphValueId,
        state: StateTokenId,
        ownership: ParamOwnership,
    },
}

/// One result leaf of a boundary. Parameter pass-through, returned computed
/// values, and returned owned views all use this one representation; the
/// value identity is the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogicalBoundaryResult {
    pub value: GraphValueId,
}

/// The complete boundary contract of one implementation graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalBoundary {
    inputs: BTreeMap<BoundaryLeaf, LogicalBoundaryInput>,
    results: BTreeMap<BoundaryLeaf, LogicalBoundaryResult>,
    /// The final state of every exclusively borrowed tensor parameter leaf
    /// (`&mut`), keyed by its input leaf. Owned and shared parameters have no
    /// final state at the boundary.
    final_states: BTreeMap<BoundaryLeaf, StateTokenId>,
}

impl LogicalBoundary {
    /// The only constructor; private to logical construction, which
    /// establishes: every canonical interface leaf is present exactly once,
    /// every exclusive parameter leaf has exactly one final state, and void
    /// creates no leaf.
    pub(super) fn new(
        inputs: BTreeMap<BoundaryLeaf, LogicalBoundaryInput>,
        results: BTreeMap<BoundaryLeaf, LogicalBoundaryResult>,
        final_states: BTreeMap<BoundaryLeaf, StateTokenId>,
    ) -> LogicalBoundary {
        LogicalBoundary {
            inputs,
            results,
            final_states,
        }
    }

    pub fn inputs(&self) -> &BTreeMap<BoundaryLeaf, LogicalBoundaryInput> {
        &self.inputs
    }

    pub fn results(&self) -> &BTreeMap<BoundaryLeaf, LogicalBoundaryResult> {
        &self.results
    }

    pub fn final_states(&self) -> &BTreeMap<BoundaryLeaf, StateTokenId> {
        &self.final_states
    }
}

/// One input leaf of a call, with the caller's identities. Mirrors
/// `LogicalBoundaryInput`: a plain value leaf; a tensor leaf backed by caller
/// storage the call borrows or moves (with the state it consumes); or a
/// tensor leaf passed as a computed value the callee reads (shared) or moves
/// (owned) without any caller storage — the caller's computed tensor is a
/// legitimate argument, and whether it needs a physical spill is a later
/// physical (D1) decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallInput {
    Value(GraphValueId),
    Tensor {
        value: GraphValueId,
        state: StateTokenId,
        ownership: ParamOwnership,
    },
    /// A tensor leaf supplied as a computed value (directly or through a view
    /// of one): the callee receives the value itself. Admitted for shared
    /// reads and owned moves only; an exclusive borrow requires a place with
    /// storage.
    Computed {
        value: GraphValueId,
        ownership: ParamOwnership,
    },
}

/// The callee contract instantiated at one call: every interface input leaf,
/// every result leaf as a value the call produces at the caller, and the next
/// state of every exclusively borrowed caller storage. These are logical
/// facts consumed by occurrence formation (package O1); the fields are public
/// because they are read as data, and only logical construction creates a
/// `CallNode`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallBoundary {
    pub inputs: BTreeMap<BoundaryLeaf, CallInput>,
    pub results: BTreeMap<BoundaryLeaf, GraphValueId>,
    pub final_states: BTreeMap<BoundaryLeaf, StateTokenId>,
}
