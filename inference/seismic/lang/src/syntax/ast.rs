//! Abstract syntax of structured Seismic source files. See `docs/seismic/language.md`.

use crate::span::Span;

/// `value`'s structure: its derived `Debug` rendering with every source
/// position (`Span { .. }`) removed, so two parses are equal in it exactly
/// when they are equal but for where their text sat. Every field of the
/// syntax tree is in it by construction. Free text that could hold `Span {`
/// is only a native declaration's asset path; identifiers cannot.
pub(crate) fn structure(value: &impl std::fmt::Debug) -> String {
    let text = format!("{value:?}");
    let mut out = String::new();
    let mut rest = text.as_str();
    while let Some(i) = rest.find("Span {") {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        rest = &rest[rest.find('}').map_or(rest.len(), |j| j + 1)..];
    }
    out + rest
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
    pub constants: Vec<ConstDecl>,
}

/// `const NAME = INTEGER`: a module constant. Every file of a source set
/// sees it, and an expression naming it is that integer literal; natives
/// read it as `SEISMIC_CONST_NAME`, and the generated Rust API exports it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstDecl {
    pub name: Ident,
    pub value: u64,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decl {
    Fn(FnDecl),
    Lower(LowerDecl),
    Native(NativeDecl),
}

impl Decl {
    pub fn name(&self) -> &Ident {
        match self {
            Decl::Fn(f) => &f.name,
            Decl::Lower(l) => &l.name,
            Decl::Native(n) => &n.function,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Decl::Fn(f) => f.span,
            Decl::Lower(l) => l.span,
            Decl::Native(n) => n.span,
        }
    }
}

/// A direct top-level backend implementation of an existing portable function.
///
/// ```text
/// native NAME for TARGET from "SOURCE":
///     static (DIM, ..)
///     params ([arithmetic] [form] NAME in [V, ..] default DEFAULT, ..)
///     elements (ELEMENT in [DTYPE, ..], ..)
///     let NAME = EXPR
///     where CONDITION
///     scratch NAME bytes (EXPR) [sync] [when CONDITION]
///     launch KERNEL [when CONDITION]:
///         params ([arithmetic] [form] NAME in [V, ..] default DEFAULT, ..)
///         where CONDITION
///         reads (PARAMETER, ..)
///         threadgroups (X, Y, Z)
///         grid (PARAMETER)
///         threads_per_threadgroup (X, Y, Z)
///         shared_bytes (EXPR)
///     repeat (EXPR):
///         launch ..
/// ```
///
/// A `CONDITION` is comparisons of natural-number expressions joined by
/// `and` and `or`, grouped with parentheses; a capability `device.NAME`
/// stands as a condition. Every expression may read `device.NAME`, a
/// property the target backend provides. A `DEFAULT` is `EXPR`, `least
/// where CONDITION` or `greatest where CONDITION`, over the device, the
/// static dimensions and the bound elements.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeDecl {
    pub function: Ident,
    pub target: Ident,
    pub source: String,
    /// Entry dimensions fixed at preparation.
    pub statics: Vec<Ident>,
    pub params: Vec<NativeParamDecl>,
    /// The dense representations a build-time compiled (CPU) form covers,
    /// per element parameter it monomorphizes.
    pub elements: Vec<NativeElementsDecl>,
    /// `records (PORT, ..)`: tensor ports the native reaches a record (a
    /// leading-axis row) at a time, each bound through a table of its
    /// records' device addresses.
    pub records: Vec<NativeRecordDecl>,
    /// Named natural-number expressions, in declaration order.
    pub terms: Vec<NativeTermDecl>,
    /// The `where` condition restricting admissible configurations.
    pub constraint: Option<Expr>,
    pub scratch: Vec<NativeScratchDecl>,
    /// Every launch in dispatch order, those of the `repeat` block included.
    pub launches: Vec<NativeLaunchDecl>,
    pub repeat: Option<NativeRepeatDecl>,
    pub span: Span,
}

/// `let NAME = EXPR`: a name for a natural-number expression over entry
/// dimensions, entry parameters and the terms before it. The declaration's
/// later expressions read it as that expression.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeTermDecl {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

/// `repeat (EXPR):` over consecutive launches: a call dispatches the block's
/// launches in order, `count` times.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeRepeatDecl {
    pub count: Expr,
    /// Ordinal of the block's first launch.
    pub first: usize,
    /// Number of launches in the block.
    pub launches: usize,
    pub span: Span,
}

/// One tuning parameter, its finite domain and its default:
/// `[arithmetic] [form] NAME in [V, ..] default DEFAULT`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeParamDecl {
    pub name: Ident,
    /// The parameter changes the arithmetic order of a row's result.
    pub arithmetic: bool,
    /// The parameter's values select structurally different algorithms.
    pub form: bool,
    pub values: Vec<u64>,
    pub default: NativeDefaultDecl,
    pub span: Span,
}

/// A parameter's default, over the device (`device.NAME`) and the static
/// dimensions: an expression whose value is one of the listed values, or
/// the least or greatest listed value at which a condition over the
/// parameter itself holds.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeDefaultDecl {
    /// `default EXPR`
    Value(Expr),
    /// `default least where CONDITION`
    Least(Expr),
    /// `default greatest where CONDITION`
    Greatest(Expr),
}

/// `NAME in [DTYPE, ..]`: the dense element types one element parameter of a
/// CPU form is compiled for.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeElementsDecl {
    pub name: Ident,
    pub dtypes: Vec<Ident>,
    pub span: Span,
}

/// A record port of a native declaration: a parameter by name, or `result N`
/// (the entry's result leaf `N`).
#[derive(Clone, Debug, PartialEq)]
pub enum NativeRecordDecl {
    Parameter(Ident),
    Result { ordinal: u64, span: Span },
}

/// Call-private device memory shared by the launches of one native call.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeScratchDecl {
    pub name: Ident,
    pub bytes: Expr,
    /// Arrival counters: every buffer of this kind is zero when a launch
    /// starts, and the call's kernels restore it to zero before it ends.
    pub sync: bool,
    /// The buffer is sized only when this condition holds.
    pub when: Option<Expr>,
    pub span: Span,
}

/// One ordered dispatch of a kernel function from the native source.
/// A launch condition's word: `PARAMETER` (word 0 of a `&tensor[1] i32`)
/// or `PARAMETER[INDEX]` (word INDEX of a `&tensor[N] i32`, INDEX a constant
/// below N).
#[derive(Clone, Debug, PartialEq)]
pub struct NativeConditionDecl {
    pub encoded: crate::checked::Encoded,
    pub word: Ident,
    /// The word's index, as written (a constant, which checking requires):
    /// none for the plain form.
    pub index: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeLaunchDecl {
    pub kernel: Ident,
    /// The launch runs only when this condition holds.
    pub when: Option<Expr>,
    /// Parameters local to this launch; names may be reused by other launches.
    pub params: Vec<NativeParamDecl>,
    /// The launch's `where`: restricts the admissible configurations of the
    /// parameters it reads, the launch's own and the entry's.
    pub constraint: Option<Expr>,
    /// Entry parameters read by the kernel but absent from launch expressions.
    pub reads: Vec<Ident>,
    /// The most groups the launch dispatches on each axis; with `grid`, the
    /// bound of the grid it reads.
    pub threadgroups: [Expr; 3],
    /// `grid (PARAMETER)`: the launch dispatches the grid an earlier launch
    /// wrote into this entry parameter (three `i32` group counts, each at
    /// most `threadgroups`'), read when the launch runs (indirect dispatch).
    pub grid: Option<Ident>,
    /// `if (WORD)` or `unless (WORD)` lines, a conjunction: the launch is
    /// encoded only while each such word of a host-shared `&tensor[N] i32`
    /// entry parameter the host writes between submissions is nonzero, or
    /// zero, read when the run is encoded.
    pub conditions: Vec<NativeConditionDecl>,
    pub threads_per_threadgroup: [Expr; 3],
    pub shared_bytes: Option<Expr>,
    pub span: Span,
}

/// `fn name[shape](params) [-> result] [requires capability] [where pred]: body`
#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    pub signature: Signature,
    pub name: Ident,
    /// Capability namespaces named by the declaration; a portable function
    /// requires none, and the resolver rejects any.
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
    /// A host port (`&host tensor`, `&host mut tensor`): a tensor the host
    /// reads (`write` false) or rewrites once it sees the launch, which the
    /// kernel never accesses.
    HostTensor { write: bool },
    /// An awaited port (`&await tensor`, `&await mut tensor`): a shared or
    /// (`write`) mutable borrow holding words the host writes while the
    /// launch waits for them. The launch parks: it reads them only once its
    /// host has answered, which each backend realizes its own way (a
    /// backend whose devices see host writes while work runs spins in the
    /// kernel; one whose devices do not ends the submission before the
    /// launch and resumes it on the host's signal).
    AwaitTensor { write: bool },
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
