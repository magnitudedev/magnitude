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
    pub ownership: crate::composition::Ownership,
    /// Source parallel binding requirements, retained before output regrouping.
    pub alias_requirements: Vec<AliasRequirement>,
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

/// Parameter ordinals whose storage must be disjoint unless their exact typed
/// per-item regions were proved equal in the source computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasRequirement {
    pub left: usize,
    pub right: usize,
    pub exact_allowed: bool,
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
    OutputWidths { maximum: i64 },
    ReductionCuts { first: i64, last: i64 },
    ReductionSegments { maximum: i64 },
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
    pub fn output_widths(maximum: i64) -> Result<Self, String> {
        if maximum <= 0 || usize::try_from(maximum).is_err() {
            return Err("invalid independent output extent".into());
        }
        Ok(Self::OutputWidths { maximum })
    }
    pub fn reduction_segments(maximum:i64)->Result<Self,String> {if maximum<=0 || usize::try_from(maximum).is_err(){return Err("invalid fold segment extent".into());} Ok(Self::ReductionSegments{maximum})}
    pub fn reduction_cuts(first: i64, last: i64) -> Result<Self,String> {
        if first < 1 || first > last || usize::try_from(last-first+1).is_err() {return Err("invalid reduction cut domain".into());}
        Ok(Self::ReductionCuts{first,last})
    }
    pub fn stream_capacities(maximum: i64) -> Result<Self, String> {
        if maximum <= 0 { return Err("stream domain must have a positive capacity".into()); }
        let count = usize::try_from(maximum).map_err(|_| "stream domain cardinality overflow")?;
        Ok(Self::StreamCapacities(StreamCapacities { maximum, count }))
    }
    pub fn len(&self) -> usize {
        match self { Self::OutputWidths { maximum } => *maximum as usize, Self::ReductionSegments{maximum}=>*maximum as usize, Self::Explicit(v) => v.len(), Self::StreamCapacities(r) => r.count, Self::ReductionCuts{first,last} => (last-first+1) as usize }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn get(&self, index: usize) -> Option<Alternative> {
        match self {
            Self::Explicit(v) => v.get(index).cloned(),
            Self::OutputWidths { maximum } if index < *maximum as usize => Some(Alternative::OutputWidth(index as i64 + 1)),
            Self::ReductionSegments{maximum} if index<*maximum as usize=>Some(Alternative::ReductionSegment(*maximum-index as i64)),
            // Whole-axis is the diagnostic baseline. Enumeration order is not
            // a performance preference and the tuner must cover every value.
            Self::StreamCapacities(r) if index < r.count => Some(Alternative::StreamCapacity(r.maximum - index as i64)),
            Self::ReductionCuts{first,last} if index < (last-first+1) as usize => Some(Alternative::ReductionCut(first+index as i64)),
            _ => None,
        }
    }
    pub fn contains(&self, alternative: &Alternative) -> bool {
        match (self, alternative) {
            (Self::Explicit(v), a) => v.contains(a),
            (Self::OutputWidths { maximum }, Alternative::OutputWidth(n)) => (1..=*maximum).contains(n),
            (Self::ReductionSegments{maximum},Alternative::ReductionSegment(n))=>(1..=*maximum).contains(n),
            (Self::StreamCapacities(r), Alternative::StreamCapacity(n)) => (1..=r.maximum).contains(n),
            (Self::ReductionCuts{first,last}, Alternative::ReductionCut(n)) => (*first..=*last).contains(n),
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

/// One retained call whose independent output dimension participates in a
/// common source-coordinate group. Position distinguishes repeated calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputGroupCall {
    pub position: usize,
    pub construct: String,
    pub parameter: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecisionKind {
    OutputGroup { coordinate: VarId, extent: i64, calls: Vec<OutputGroupCall> },
    Representation { variable: VarId },
    ReductionSegments {extent:i64},
    Intermediate { variable: VarId, publication: usize },
    ReductionBranch { start: i64, end: i64, fields: Vec<Ty> },
    Reduction { merge: String, extent: Sym, fields: Vec<Ty> },
    ParallelFusion { boundary: usize, other: usize, left_domain: Vec<Sym>, right_domain: Vec<Sym> },
    StreamFusion { first: usize, second: usize, extent: Sym },
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
    OutputWidth(i64),
    /// Preserve the representation-owned compact packet storage.
    Encoded,
    /// Materialize exact decoded F32 values alongside the packet snapshot.
    Decoded,
    ReductionSegment(i64),
    ReductionCut(i64),
    ReductionTree(crate::reduction::structured::Tree),
    ParallelFusion { shared_axes: usize, refine_consumer: bool },
    RetainLocal,
    Separate,
    Fuse,
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
