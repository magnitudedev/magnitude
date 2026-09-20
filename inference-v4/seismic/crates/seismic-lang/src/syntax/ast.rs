//! Abstract syntax of structured Seismic source files. See `docs/seismic/language.md`.

use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
}

impl AssignOp {
    pub fn text(self) -> &'static str {
        match self {
            AssignOp::Assign => "=",
            AssignOp::Add => "+=",
            AssignOp::Sub => "-=",
            AssignOp::Mul => "*=",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

impl UnaryOp {
    pub fn text(self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "not ",
            UnaryOp::BitNot => "~",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BitOr,
    BitXor,
    BitAnd,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl BinaryOp {
    pub fn text(self) -> &'static str {
        match self {
            BinaryOp::Or => "or",
            BinaryOp::And => "and",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::BitAnd => "&",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
        }
    }

    /// Binding power; higher binds tighter. All binary operators are left-associative.
    pub fn precedence(self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => 4,
            BinaryOp::BitOr => 5,
            BinaryOp::BitXor => 6,
            BinaryOp::BitAnd => 7,
            BinaryOp::Shl | BinaryOp::Shr => 8,
            BinaryOp::Add | BinaryOp::Sub => 9,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => 10,
        }
    }
}

/// Precedence of `not`, between `and` and comparisons.
pub const NOT_PRECEDENCE: u8 = 3;
/// Precedence of unary `-` and `~`, above every binary operator.
pub const UNARY_PRECEDENCE: u8 = 11;

#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    pub decls: Vec<Decl>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decl {
    Fn(FnDecl),
    Lower(LowerDecl),
}

impl Decl {
    pub fn name(&self) -> &Ident {
        match self {
            Decl::Fn(f) => &f.name,
            Decl::Lower(l) => &l.name,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Decl::Fn(f) => f.span,
            Decl::Lower(l) => l.span,
        }
    }
}

/// `fn name[shape](params) [-> result] [for target] [requires capability] [where pred]: body`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub signature: Signature,
    pub name: Ident,
    /// `None` is a portable function; `Some` restricts the function to that backend.
    pub target: Option<Ident>,
    /// Capability namespaces explicitly required by this backend-specific body.
    pub requires: Vec<CapabilityPath>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub shape: Vec<Ident>,
    pub params: Vec<Param>,
    /// `None` means `void`.
    pub result: Option<TypeExpr>,
    /// Conjuncts of the `where` clause.
    pub predicates: Vec<Expr>,
}

/// `lower name[..](..) [-> r] for target [where pred]: body`.
#[derive(Clone, Debug, PartialEq)]
pub struct LowerDecl {
    pub name: Ident,
    pub signature: Signature,
    pub target: Ident,
    /// Capability namespaces explicitly required by this lowering body.
    pub requires: Vec<CapabilityPath>,
    /// Conjuncts following `for target where`; for the long form these are also stored here,
    /// not in `signature.predicates`.
    pub predicates: Vec<Expr>,
    pub body: Block,
    pub span: Span,
}

/// A backend capability namespace such as `metal.matrix`.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityPath {
    pub backend: Ident,
    pub capability: Ident,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeExpr {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// `f32`, or an element parameter used as a scalar type (`T`).
    Scalar(Ident),
    /// `index[N]`
    Index(Box<Expr>),
    /// `range[N]`: a bounded logical half-open range.
    Range(Box<Expr>),
    /// `tensor[shape] elem`, `&tensor[shape] elem`, or `&mut tensor[shape] elem`.
    Shaped {
        head: ShapedHead,
        shape: Vec<Expr>,
        elem: Ident,
    },
    Tuple(Vec<TypeExpr>),
    Void,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapedHead {
    /// Owned logical tensor.
    Tensor,
    /// Shared logical tensor borrow (`&tensor`).
    SharedTensor,
    /// Exclusive mutable logical tensor borrow (`&mut tensor`).
    MutTensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

/// `let` target: `a`, `a, b`, `(a, b)`, nested.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Name(Ident),
    Tuple(Vec<Pattern>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Let {
        mutable: bool,
        pattern: Pattern,
        value: Expr,
    },
    /// `target op= value`; `target` may be a tuple of places for tuple assignment.
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// `for targets in iter`: `iter` is a bounded range or a range value.
    For {
        /// `false` is ordered `for`; `true` is independent `parallel for`.
        parallel: bool,
        targets: Vec<Ident>,
        iter: Expr,
        body: Block,
    },
    If {
        cond: Expr,
        then: Block,
        els: Option<Block>,
    },
    Return(Vec<Expr>),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(u64),
    Float(f64),
    Inf,
    Bool(bool),
    Name(Ident),
    Tuple(Vec<Expr>),
    /// `lo..hi`
    Range {
        lo: Box<Expr>,
        hi: Box<Expr>,
    },
    /// `tensor[shape] elem`: uninitialized owned logical tensor storage.
    Tensor {
        shape: Vec<Expr>,
        elem: Ident,
    },
    /// `f[R = 64](args)`
    Call {
        callee: Box<Expr>,
        bindings: Vec<(Ident, Expr)>,
        args: Vec<Arg>,
    },
    Index {
        base: Box<Expr>,
        indices: Vec<Index>,
    },
    /// `t.T`, `t.words`, `metal.name`
    Attr {
        base: Box<Expr>,
        name: Ident,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// A point, a slice binder or a tile coordinate; the checker distinguishes them.
    Expr(Expr),
    /// `lo:hi`, `lo:`, `:hi`, `:`
    Slice {
        start: Option<Expr>,
        end: Option<Expr>,
    },
}
