//! Typed nodes of the execution IR: the statements and expressions `instantiate`
//! produces and a backend realization consumes.

use super::types::Ty;
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::syntax::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::types::{DType, Elem};

pub type VarId = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
    pub kind: VarKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VarKind {
    Param(usize),
    Local,
    /// A loop index; its value is the atom.
    Index(Atom),
}

/// Identity within one normalized execution artifact, never a source offset or
/// scheduling position. Transformations producing a new artifact reassign IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperationId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub id: Option<OperationId>,
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Parallel { vars: Vec<VarId>, extents: Vec<Sym>, body: Vec<Stmt> },
    /// Internal selected streaming execution. `piece` is the compiler-chosen extent
    /// along `axis`; ordinary source code cannot observe or author this node.
    /// After lowering, `capacity` is the static piece size when the axis extent is dynamic
    /// (the piece atom then stays symbolic and denotes the runtime extent of each piece).
    /// `modes` is unresolved before selection; afterward it contains one load mode per binding.
    LoadLoop { domain: IterationDomain, offset: Option<VarId>, modes: Option<Vec<LoadMode>>, vars: Vec<VarId>, views: Vec<Expr>, axes: Vec<usize>, piece: Atom, capacity: Option<i64>, body: Vec<Stmt> },
    Owned { vars: Vec<VarId>, tile: Expr, body: Vec<Stmt> },
    Range { var: VarId, lo: Sym, hi: Sym, body: Vec<Stmt> },
    /// Lowering scope: iterate `extent` across the subgroup's lanes, `width` consecutive per lane.
    Lanes { var: VarId, extent: Sym, width: i64, body: Vec<Stmt> },
    If { cond: Expr, then: Vec<Stmt>, els: Vec<Stmt> },
    Assign { target: Expr, op: AssignOp, value: Expr },
    Expr(Expr),
}

/// Logical iteration geometry independent of the transfers chosen for its body.
/// The view is evaluated only for metadata and extent guards, never materialized.
#[derive(Clone, Debug, PartialEq)]
pub struct IterationDomain {
    pub view: Expr,
    pub axis: usize,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
    /// Symbolic value for integer-typed expressions built from parameters, indices and literals.
    pub sym: Option<Sym>,
    pub span: Span,
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        // Typed expression identity is structural, not a floating-point
        // comparison: signed zero and NaN payloads are part of the source.
        let same_kind = match (&self.kind, &other.kind) {
            (ExprKind::Float(a), ExprKind::Float(b)) => a.to_bits() == b.to_bits(),
            (a, b) => a == b,
        };
        same_kind && self.ty == other.ty && self.sym == other.sym && self.span == other.span
    }
}

/// Selected realization of a value-semantic load. Borrowing requires a lifetime
/// proof; it does not change the program's observable snapshot semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMode { Materialize, Borrow }

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(i64),
    /// A shape parameter used as a value; `sym` carries it.
    ShapeParam(String),
    Float(f64),
    Bool(bool),
    Var(VarId),
    TileAlloc { shape: Vec<Sym>, dtype: Elem },
    /// Execution-stage load with its storage decision resolved.
    Load { view: Box<Expr>, mode: LoadMode },
    /// Indexing of a tensor or tile: a view, or an element when every axis is a point.
    Index { base: Box<Expr>, indices: Vec<Index> },
    Transpose(Box<Expr>),
    /// Lowering-scope accessor on a packed tile: `words`, `scale`, `bias`.
    Accessor { base: Box<Expr>, name: String },
    /// Lowering-scope lane distribution of a tile axis: `t.lanes(K)`.
    Lanes { base: Box<Expr>, extent: Sym },
    Builtin { name: Builtin, args: Vec<Expr> },
    /// Call of a function or construct with inferred shape and element arguments.
    Call { callee: String, shape_args: Vec<Sym>, elem_args: Vec<Elem>, args: Vec<Expr> },
    Intrinsic { op: crate::intrinsics::Operation, args: Vec<Expr> },
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Cast { dtype: DType, expr: Box<Expr> },
    Tuple(Vec<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    Point(Expr),
    Slice { start: Option<Expr>, end: Option<Expr> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    /// Compiler-internal scalar selection. All three arguments are evaluated;
    /// a true condition returns the second value, otherwise the third value.
    /// This is not source control flow and is deliberately absent from names.
    Select,
    Reshape,
    Load,
    Store,
    Atomic,
    Reduce,
    Extent,
    Fma,
    Exp,
    ExpFast,
    Rsqrt,
    Sqrt,
    Log,
    Sin,
    Cos,
    Abs,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Argmax,
}

impl ReduceOp {
    pub fn from_tag(tag: i64) -> Option<Self> {
        Some(match tag { 0 => Self::Sum, 1 => Self::Max, 2 => Self::Min, 3 => Self::Argmax, _ => return None })
    }
}
