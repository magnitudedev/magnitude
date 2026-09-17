//! Abstract syntax of Seismic source files.

use crate::span::Span;

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
    Construct(FnDecl),
    Lower(LowerDecl),
}

impl Decl {
    pub fn name(&self) -> &Ident {
        match self {
            Decl::Fn(f) | Decl::Construct(f) => &f.name,
            Decl::Lower(l) => &l.name,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Decl::Fn(f) | Decl::Construct(f) => f.span,
            Decl::Lower(l) => l.span,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub name: Ident,
    pub shape: Vec<Ident>,
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LowerDecl {
    pub name: Ident,
    /// The construct's signature, restated so the block's body is readable on its own and
    /// checked against the declaration. A concrete element type where the construct has an
    /// element parameter specializes the block to that type.
    pub shape: Vec<Ident>,
    pub params: Vec<Param>,
    /// `None` means `lower name: portable`.
    pub body: Option<Block>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
}

/// `head[shape] elem`, e.g. `tensor[N, K] q4g64`, `tile[M, N] f32`, `f32`, `T`.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeExpr {
    pub head: Ident,
    pub shape: Vec<Expr>,
    pub elem: Option<Ident>,
    pub span: Span,
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

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    For { targets: Vec<Ident>, iter: Expr, body: Block },
    If { cond: Expr, then: Block, els: Option<Block> },
    Assign { target: Expr, op: AssignOp, value: Expr },
    Expr(Expr),
}

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
    /// `tile[shape] dtype`
    Tile { shape: Vec<Expr>, dtype: Ident },
    /// `f[R = 64](args)`: explicit shape bindings for parameters the arguments do not determine.
    Call { callee: Box<Expr>, bindings: Vec<(Ident, Expr)>, args: Vec<Arg> },
    Index { base: Box<Expr>, indices: Vec<Index> },
    Attr { base: Box<Expr>, name: Ident },
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Lambda { params: Vec<Ident>, body: Box<Expr> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    Expr(Expr),
    Slice { start: Option<Expr>, end: Option<Expr> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
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
