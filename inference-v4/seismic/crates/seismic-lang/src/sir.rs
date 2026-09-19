//! Structured IR: the checked, typed program. Authored structure is authoritative:
//! regions, stages, producers, state and publications appear exactly as written.
//! Nothing here chooses an implementation, a grouping or a number.
//!
//! One `Definition` is a template. Calls name a contract (function name + argument
//! binding); which definition implements an occurrence is a selection decision made
//! over `family`, never here.

use super::syntax::ast::{AssignOp, BinaryOp, Mode, RegionMode, UnaryOp};
use super::types::{DType, Elem, Extent, RegionId, SliceId, Ty};
use crate::intrinsics::Operation;
use crate::span::Span;
use crate::sym::Sym;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefId(pub u32);

/// Index into `Body::vars`.
pub type VarId = usize;

/// Index into `Body::calls`: one static call occurrence within a definition body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallId(pub u32);

#[derive(Clone, Debug)]
pub struct Program {
    pub definitions: Vec<Definition>,
    pub families: Vec<ContractFamily>,
    /// Source files, for diagnostics: `(path, text)`.
    pub files: Vec<(String, String)>,
}

impl Program {
    pub fn definition(&self, id: DefId) -> &Definition {
        &self.definitions[id.0 as usize]
    }

    pub fn family_of(&self, id: DefId) -> &ContractFamily {
        &self.families[self.definition(id).family]
    }

    /// Resolve a name-only root request. Calls already carry an exact family index, but an
    /// embedding request has no argument-type selector and therefore must fail closed when
    /// several disjoint overload families share the name.
    pub fn family_index(&self, name: &str) -> Result<usize, String> {
        let mut matches = self
            .families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.name == name)
            .map(|(index, _)| index);
        let Some(first) = matches.next() else {
            return Err(format!("no linked function `{name}`"));
        };
        if matches.next().is_some() {
            return Err(format!(
                "ambiguous linked function `{name}`: a name-only root request matches multiple disjoint signatures"
            ));
        }
        Ok(first)
    }

    pub fn resolve_family(&self, name: &str) -> Result<&ContractFamily, String> {
        Ok(&self.families[self.family_index(name)?])
    }

    pub fn family(&self, name: &str) -> Option<&ContractFamily> {
        self.resolve_family(name).ok()
    }
}

/// A connected component of same-name implementations with overlapping applicability and a
/// compatible contract.
#[derive(Clone, Debug)]
pub struct ContractFamily {
    pub name: String,
    /// Portable and backend-specific `fn` bodies.
    pub bodies: Vec<DefId>,
    /// Backend `lower` bodies.
    pub lowerings: Vec<DefId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefKind {
    /// A `fn` body. `None` is portable; `Some` restricts it to one backend.
    Body { target: Option<String> },
    /// `lower … for target:` with a body.
    Lower { target: String },
}

impl DefKind {
    pub fn target(&self) -> Option<&str> {
        match self {
            DefKind::Body { target } => target.as_deref(),
            DefKind::Lower { target } => Some(target),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub kind: DefKind,
    /// Index into `Program::families`.
    pub family: usize,
    pub shape_params: Vec<String>,
    pub elem_params: Vec<String>,
    /// Concrete element types this definition fixes where the contract has a parameter.
    pub elem_bindings: Vec<(String, Elem)>,
    pub params: Vec<Param>,
    /// Parameter ordinal pairs permitted to alias.
    pub aliases: Vec<(usize, usize)>,
    pub result: Ty,
    /// Applicability: every predicate must hold.
    pub predicates: Vec<Predicate>,
    pub admit: bool,
    pub body: Body,
    /// Index into `Program::files`.
    pub file: usize,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub mode: Mode,
    pub ty: Ty,
    pub var: VarId,
}

/// A decidable applicability predicate over shape parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// `expr >= 0`
    NonNegative(Sym),
    /// `expr == 0` (equalities and divisibility `N % c == 0`)
    Zero(Sym),
    /// `expr != 0`
    NonZero(Sym),
    /// `full(P)`: when `P` is bound to a structural extent, its capacity divides the
    /// semantic extent of the slice's parent domain. Vacuous for semantic bindings.
    Full(String),
}

#[derive(Clone, Debug)]
pub struct Body {
    pub vars: Vec<Var>,
    pub slices: Vec<SliceDecl>,
    pub regions: Vec<RegionDecl>,
    pub calls: Vec<CallSite>,
    pub block: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub ty: Ty,
    pub kind: VarKind,
    /// Carries a partial-domain obligation (language.md section 8).
    pub partial: bool,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VarKind {
    Param(usize),
    /// `let`
    Value,
    /// `let mut`
    State,
    /// Region binder.
    Slice(SliceId),
    /// Stage input port.
    Port,
    /// Binder of `for i in lo..hi` (type `Index`/`i32`).
    RangeIndex,
    /// Binder of `owned`/`axis` (type `Index` over a semantic axis, `Coord` over a structural one).
    Coordinate,
    /// Binder of `for h in slice`: the semantic coordinate, a proven member of the slice.
    SliceMember(SliceId),
    /// Merge operand.
    MergeOperand,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceDecl {
    pub var: VarId,
    pub region: RegionId,
    pub parent: SliceParent,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SliceParent {
    /// A new partition of `lo..hi`. This binder is a numerical site.
    Domain { lo: Sym, hi: Sym },
    /// An explicit refinement of an enclosing slice. This binder is a numerical site.
    Refine(SliceId),
    /// Rebinding of a result member's slice. Not a site: geometry is inherited.
    Rebind(SliceId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegionDecl {
    pub mode: RegionMode,
    pub binders: Vec<SliceId>,
    /// Enclosing region, if any.
    pub parent: Option<RegionId>,
    pub span: Span,
}

/// One static call occurrence. `family` and the argument binding identify the contract;
/// applicable definitions are resolved per target and workload by `family` construction.
#[derive(Clone, Debug, PartialEq)]
pub struct CallSite {
    /// Index into `Program::families`.
    pub family: usize,
    /// Callee shape parameter name -> bound extent, for the *contract* parameter names of
    /// each candidate separately (candidates may name parameters differently).
    pub bindings: Vec<CandidateBinding>,
    pub result: Ty,
    pub span: Span,
}

/// How one candidate definition's parameters bind at a call occurrence. Candidates whose
/// unification failed are absent. Predicates are *not* evaluated here.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateBinding {
    pub definition: DefId,
    pub shape_args: Vec<(String, Extent)>,
    pub elem_args: Vec<(String, Elem)>,
    /// Argument expression ordinal for each callee parameter (named arguments resolved).
    pub arg_order: Vec<usize>,
    /// Element parameters of the CALLER that must equal these concrete element types for
    /// this candidate to apply: the callee fixes a concrete element (`bf16`, `q4g64`) where
    /// the caller's argument element is still a parameter. Decided at family construction
    /// against the caller template's element bindings.
    pub requires_elems: Vec<(String, Elem)>,
}

pub type Block = Vec<Stmt>;

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Var(VarId),
    Tuple(Vec<Pattern>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    /// `let`/`let mut`: one producer occurrence in this lexical scope.
    Bind {
        pattern: Pattern,
        value: Expr,
    },
    /// State update. For tuple targets all right-hand sides read old versions first.
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// A region in statement position.
    Region(Region),
    /// A maximal run of consecutive `stage` statements.
    Stages(Vec<Stage>),
    /// `for i in lo..hi`
    Range {
        var: VarId,
        lo: Expr,
        hi: Expr,
        body: Block,
    },
    /// `for i, j in owned(t)` (all axes) / `for k in axis(t, n)` (`axes == [n]`)
    Coordinates {
        vars: Vec<VarId>,
        of: Expr,
        axes: Vec<usize>,
        body: Block,
    },
    /// `for h in slice`
    Members {
        var: VarId,
        slice: SliceId,
        body: Block,
    },
    If {
        cond: Expr,
        then: Block,
        els: Block,
    },
    Publish {
        value: Expr,
        destination: Expr,
    },
    Yield(Vec<Expr>),
    Return(Vec<Expr>),
    /// A call evaluated for its `out`/`inout` effects.
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub id: RegionId,
    pub mode: RegionMode,
    pub binders: Vec<VarId>,
    pub source: RegionSource,
    pub body: Block,
    pub merge: Option<Merge>,
    /// `Some` when used as an expression without `merge`: the result type. With `merge`
    /// the expression's type is the merged member type.
    pub result: Option<Ty>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RegionSource {
    /// New partitions / refinements, one per binder; see each binder's `SliceDecl::parent`.
    Domains,
    /// Traversal of an earlier region result with its original slices.
    Results(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Merge {
    pub left: Pattern,
    pub right: Pattern,
    pub identity: Expr,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub name: String,
    pub ports: Vec<VarId>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
    /// Symbolic value of integer expressions over shape parameters and indices.
    pub sym: Option<Sym>,
    pub partial: bool,
    pub span: Span,
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        let same_kind = match (&self.kind, &other.kind) {
            (ExprKind::Float(a), ExprKind::Float(b)) => a.to_bits() == b.to_bits(),
            (a, b) => a == b,
        };
        same_kind && self.ty == other.ty && self.sym == other.sym && self.span == other.span
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// Scalar point (semantic coordinate; membership proved for structural axes).
    Point(Expr),
    /// Tile coordinate over a structural axis.
    Coord(VarId),
    /// Slice binder.
    Slice(SliceId),
    /// `lo:hi` semantic range; `None` is the axis bound.
    Range {
        start: Option<Expr>,
        end: Option<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Math {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Argmax,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Var(VarId),
    /// A shape parameter used as a value.
    ShapeParam(String),
    Tuple(Vec<Expr>),
    Field {
        base: Box<Expr>,
        index: usize,
    },
    /// `tile[shape] elem`: uninitialized owned tile.
    TileAlloc,
    /// `zeros_like` / `ones_like`: shape of the operand, given dtype, constant fill.
    Filled {
        like: Box<Expr>,
        value: f64,
    },
    /// Element read (all points) or view (otherwise) of a tensor, view, tile.
    Index {
        base: Box<Expr>,
        indices: Vec<Index>,
    },
    /// `results[p]` / `results[rows, cols]`
    Member {
        result: Box<Expr>,
        slices: Vec<SliceId>,
    },
    Transpose(Box<Expr>),
    Reshape {
        base: Box<Expr>,
        axes: Vec<Extent>,
    },
    /// Snapshot in the view's own representation.
    Load(Box<Expr>),
    /// Dense f32 tile of a packed view.
    Decode(Box<Expr>),
    /// Scalar cast, or elementwise read-and-convert of a tile/view (yields a tile).
    Cast {
        dtype: DType,
        expr: Box<Expr>,
    },
    /// Scalar or elementwise (tile operands, scalar broadcast).
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Math {
        op: Math,
        args: Vec<Expr>,
    },
    Select {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
    },
    /// `unordered` (`reduce(t, axis, sum, unordered=true)`, legal only inside an `admit fn`)
    /// permits reassociation: a backend may combine lane partials. The reference
    /// interpreter always accumulates in ascending index order.
    Reduce {
        value: Box<Expr>,
        axis: usize,
        op: ReduceOp,
        unordered: bool,
    },
    /// Semantic coordinate of a tile coordinate.
    CoordOf(VarId),
    /// `extent(v, axis)` on a semantic axis.
    ExtentOf {
        base: Box<Expr>,
        axis: usize,
    },
    Call {
        call: CallId,
        args: Vec<Expr>,
    },
    /// A result-producing region (with or without merge).
    Region(Box<Region>),
    // Target-dependent forms.
    Intrinsic {
        op: Operation,
        args: Vec<Expr>,
    },
    /// Packed plane accessor: `words`, `scale`, `bias`.
    Accessor {
        base: Box<Expr>,
        name: String,
    },
    /// `capacity(t, axis)` / `valid(t, axis)` under geometry authority.
    Geometry {
        base: Box<Expr>,
        axis: usize,
        valid: bool,
    },
    Atomic {
        op: BinaryOp,
        place: Box<Expr>,
        value: Box<Expr>,
    },
}
