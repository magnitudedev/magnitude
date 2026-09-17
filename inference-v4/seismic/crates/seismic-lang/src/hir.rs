//! Typed, resolved IR produced by the checker.

use crate::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::span::Span;
use crate::sym::{Atom, Sym};
use crate::types::{DType, Elem, Ty};

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

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub is_construct: bool,
    pub shape_params: Vec<String>,
    pub elem_params: Vec<String>,
    pub params: Vec<(String, Ty)>,
    /// Declared runtime index bounds, retained as part of the invocation contract.
    pub index_params: Vec<(String, Sym)>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lowering {
    pub construct: String,
    pub backend: String,
    /// Element parameters this block is specialized to; a block with bindings never covers the domain.
    pub elem_bindings: Vec<(String, Elem)>,
    pub vars: Vec<Var>,
    pub body: Vec<Stmt>,
    /// Constraints on shape parameters the body needs and the checker could not prove.
    /// Each is `expr >= 0`; an empty list means the block applies to the whole domain.
    pub residual: Vec<Sym>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Parallel { vars: Vec<VarId>, extents: Vec<Sym>, body: Vec<Stmt> },
    /// `for vars in load(views, over=axis)`; `piece` is the lowering-chosen extent along `axis`.
    /// After lowering, `capacity` is the static piece size when the axis extent is dynamic
    /// (the piece atom then stays symbolic and denotes the runtime extent of each piece).
    LoadLoop { vars: Vec<VarId>, views: Vec<Expr>, axis: usize, piece: Atom, capacity: Option<i64>, body: Vec<Stmt> },
    Owned { vars: Vec<VarId>, tile: Expr, body: Vec<Stmt> },
    Range { var: VarId, lo: Sym, hi: Sym, body: Vec<Stmt> },
    /// Lowering scope: iterate `extent` across the subgroup's lanes, `width` consecutive per lane.
    Lanes { var: VarId, extent: Sym, width: i64, body: Vec<Stmt> },
    If { cond: Expr, then: Vec<Stmt>, els: Vec<Stmt> },
    Assign { target: Expr, op: AssignOp, value: Expr },
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
    /// Symbolic value for integer-typed expressions built from parameters, indices and literals.
    pub sym: Option<Sym>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(i64),
    /// A shape parameter used as a value; `sym` carries it.
    ShapeParam(String),
    Float(f64),
    Bool(bool),
    Var(VarId),
    TileAlloc { shape: Vec<Sym>, dtype: Elem },
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
    Intrinsic { name: String, args: Vec<Expr> },
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

impl Builtin {
    pub fn from_name(name: &str) -> Option<Builtin> {
        Some(match name {
            "reshape" => Builtin::Reshape,
            "load" => Builtin::Load,
            "store" => Builtin::Store,
            "atomic" => Builtin::Atomic,
            "reduce" => Builtin::Reduce,
            "extent" => Builtin::Extent,
            "fma" => Builtin::Fma,
            "exp" => Builtin::Exp,
            "exp_fast" => Builtin::ExpFast,
            "rsqrt" => Builtin::Rsqrt,
            "sqrt" => Builtin::Sqrt,
            "log" => Builtin::Log,
            "sin" => Builtin::Sin,
            "cos" => Builtin::Cos,
            "abs" => Builtin::Abs,
            "max" => Builtin::Max,
            "min" => Builtin::Min,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Argmax,
}

impl ReduceOp {
    pub fn from_name(name: &str) -> Option<ReduceOp> {
        Some(match name {
            "sum" => ReduceOp::Sum,
            "max" => ReduceOp::Max,
            "min" => ReduceOp::Min,
            "argmax" => ReduceOp::Argmax,
            _ => return None,
        })
    }
}
