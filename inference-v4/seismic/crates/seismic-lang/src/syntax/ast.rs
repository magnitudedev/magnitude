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

/// `[admit] fn name[shape](params) [alias(..)] [-> result] [for target] [where pred]: body`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub admit: bool,
    pub signature: Signature,
    pub name: Ident,
    /// `None` is a portable function; `Some` restricts the function to that backend.
    pub target: Option<Ident>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub shape: Vec<Ident>,
    pub params: Vec<Param>,
    /// `alias(a, b)` pairs permitted to overlap.
    pub aliases: Vec<(Ident, Ident)>,
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
    /// Conjuncts following `for target where`; for the long form these are also stored here,
    /// not in `signature.predicates`.
    pub predicates: Vec<Expr>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    In,
    Out,
    Inout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub mode: Mode,
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
    /// `tensor[shape] elem`, `view[shape] elem`, `tile[shape] elem`
    Shaped {
        head: ShapedHead,
        shape: Vec<Expr>,
        elem: Ident,
    },
    Tuple(Vec<TypeExpr>),
    Void,
    /// `metal.simdgroup_matrix(f32)`: target namespace, type name, arguments.
    Native {
        target: Ident,
        name: Ident,
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapedHead {
    Tensor,
    View,
    Tile,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegionMode {
    Parallel,
    Ordered,
    Pipeline,
}

/// `mode [binders] in source: body [merge (l, r) identity e: body]`
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub mode: RegionMode,
    pub binders: Vec<Ident>,
    /// One expression, or the members of a parenthesized product `(d0, d1)`. Each is a
    /// domain `lo..hi`, an enclosing slice, or (single source only) a region result.
    pub sources: Vec<Expr>,
    pub body: Block,
    pub merge: Option<Merge>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Merge {
    pub left: Pattern,
    pub right: Pattern,
    pub identity: Expr,
    pub body: Block,
    pub span: Span,
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
    /// A region in statement position (no result).
    Region(Region),
    Stage {
        name: Ident,
        ports: Vec<Ident>,
        body: Block,
    },
    /// `for targets in iter`: `iter` is `lo..hi`, `owned(t)`, `axis(t, n)` or a slice name.
    For {
        targets: Vec<Ident>,
        iter: Expr,
        body: Block,
    },
    If {
        cond: Expr,
        then: Block,
        els: Option<Block>,
    },
    Publish {
        value: Expr,
        destination: Expr,
    },
    Yield(Vec<Expr>),
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
    /// `tile[shape] elem`
    Tile {
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
    /// A result-producing region; only as the value of `let`, `yield` or `return`.
    Region(Box<Region>),
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
