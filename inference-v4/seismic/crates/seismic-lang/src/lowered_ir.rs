//! Lowered IR for one backend, sharing the common typed IR nodes.
//! Expansion decisions are retained with the function. Further execution choices
//! are not yet closed here; this representation is not Tuned IR.
use crate::ir::{Stmt, Var, VarId};
use crate::sym::Sym;
use crate::types::{Elem, Ty};
use std::collections::HashMap;

/// A function after inlining for one backend and one shape binding.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredIr {
    pub name: String,
    pub backend: String,
    pub params: Vec<(String, Ty)>,
    pub index_params: Vec<(String, Sym)>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
    /// Shape parameters and their concrete values.
    pub shapes: HashMap<String, i64>,
    /// Which lowering block was chosen for each call site, for inspection and the tuning cache.
    pub selections: Vec<Selection>,
    /// Validated decisions that produced this expanded program.
    pub decisions: Vec<DecisionRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub construct: String,
    pub shape_args: Vec<i64>,
    pub choice: Choice,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Choice {
    /// Index into the construct's blocks for this backend, in file order.
    Block(usize),
    Portable,
}

/// A legal optimization decision, with no performance ordering implied.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub kind: DecisionKind,
    pub alternatives: Alternatives,
}

/// Numeric execution domains are indexed symbolically. An axis with billions of
/// elements does not require billions of allocations to expose its legal pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Alternatives {
    Explicit(Vec<Alternative>),
    StreamCapacities(StreamCapacities),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCapacities {
    maximum: i64,
    count: usize,
}
impl From<Vec<Alternative>> for Alternatives {
    fn from(values: Vec<Alternative>) -> Self { Self::Explicit(values) }
}
impl Alternatives {
    pub fn stream_capacities(maximum: i64) -> Result<Self, String> {
        if maximum <= 0 { return Err("stream domain must have a positive capacity".into()); }
        let count = usize::try_from(maximum).map_err(|_| "stream domain cardinality overflow")?;
        Ok(Self::StreamCapacities(StreamCapacities { maximum, count }))
    }
    pub fn len(&self) -> usize {
        match self { Self::Explicit(v) => v.len(), Self::StreamCapacities(r) => r.count }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn get(&self, index: usize) -> Option<Alternative> {
        match self {
            Self::Explicit(v) => v.get(index).cloned(),
            // Whole-axis is the diagnostic baseline. Enumeration order is not
            // a performance preference and the tuner must cover every value.
            Self::StreamCapacities(r) if index < r.count => Some(Alternative::StreamCapacity(r.maximum - index as i64)),
            _ => None,
        }
    }
    pub fn contains(&self, alternative: &Alternative) -> bool {
        match (self, alternative) {
            (Self::Explicit(v), a) => v.contains(a),
            (Self::StreamCapacities(r), Alternative::StreamCapacity(n)) => (1..=r.maximum).contains(n),
            _ => false,
        }
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = Alternative> + ExactSizeIterator + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
    pub fn capacity_interval(&self) -> Option<(i64, i64)> {
        match self { Self::StreamCapacities(r) => Some((r.maximum, 1)), _ => None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecisionKind {
    Stream { piece: crate::sym::Atom, extent: Sym, maximum: i64 },
    Construct {
        name: String,
        shape_args: Vec<Sym>,
        element_args: Vec<Elem>,
    },
    Producer {
        variable: VarId,
        name: String,
        ty: Ty,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Alternative {
    StreamCapacity(i64),
    Body(Choice),
    Materialize,
    Recompute,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub domain: Decision,
    pub selected: Alternative,
}
