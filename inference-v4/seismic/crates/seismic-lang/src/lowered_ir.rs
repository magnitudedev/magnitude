//! Lowered IR for one backend, sharing the common typed IR nodes.
//! Expansion decisions are retained with the function. Further execution choices
//! are not yet closed here; this representation is not Tuned IR.
use crate::ir::{Stmt, Var, VarId};
use crate::sym::Sym;
use crate::types::{Elem, Ty};
use std::collections::HashMap;

/// A function after inlining for one backend and one shape binding.
#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
    pub alternatives: Vec<Alternative>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecisionKind {
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
    Body(Choice),
    Materialize,
    Recompute,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub domain: Decision,
    pub selected: Alternative,
}

