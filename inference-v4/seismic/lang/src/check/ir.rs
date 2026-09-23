//! The crate-private checked representation.
//!
//! One `Definition` is a template checked once: its signature, `where`
//! predicates and body are symbolic over its own dimensions in its own
//! expression arena. A call names a family and, per candidate definition
//! that unifies with the arguments, how the callee's dimensions bind in the
//! caller's arena. The checked call graph is acyclic.
//!
//! Nothing here is constructible outside `check` and the bundle decoder.

use crate::expr::{ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, FamilyId, FunctionId, IntrinsicId, StableFunctionId};
use crate::intrinsics::PrimitiveId;
use crate::reference_math::ReferenceScalar;
use crate::registry::BackendName;
use crate::span::Span;
use crate::syntax::ast::AssignOp;
use crate::types::{Elem, ValueType};

/// A local of one checked body. Parameters occupy the first locals in
/// parameter order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LocalId(u32);

impl LocalId {
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for LocalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a definition is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefKind {
    /// A `fn` body. `None` is portable; `Some` restricts it to one backend.
    Body { target: Option<BackendName> },
    /// `lower … for target:` with a body.
    Lower { target: BackendName },
}

impl DefKind {
    pub(crate) fn target(self) -> Option<BackendName> {
        match self {
            DefKind::Body { target } => target,
            DefKind::Lower { target } => Some(target),
        }
    }

    pub(crate) fn is_portable_body(self) -> bool {
        matches!(self, DefKind::Body { target: None })
    }
}

/// Logical call ownership of a parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Ownership {
    Tuple(Vec<Ownership>),
    /// A plain value (scalar, index, range, tuple, opaque value).
    Value,
    /// An owned tensor that moves into the callee.
    Owned,
    /// A shared borrow (`&tensor`).
    Shared,
    /// An exclusive mutable borrow (`&mut tensor`).
    Exclusive,
}

/// One dimension (shape parameter) of a definition.
#[derive(Clone, Debug)]
pub(crate) struct Dimension {
    pub name: String,
    /// Its symbol in the definition's arena.
    pub symbol: SymbolId,
    /// Whether a `where` predicate admits the extent zero; otherwise the
    /// dimension is at least one.
    pub admits_zero: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Param {
    pub name: String,
    pub ownership: Ownership,
    pub ty: ValueType,
    pub local: LocalId,
    pub span: Span,
}

/// A decidable applicability predicate over dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Predicate {
    /// `expr >= 0`
    NonNegative(IntExpr),
    /// `expr == 0` (equalities and divisibility `N % c == 0`)
    Zero(IntExpr),
    /// `expr != 0`
    NonZero(IntExpr),
}

#[derive(Debug)]
pub(crate) struct Definition {
    pub stable: StableFunctionId,
    pub name: String,
    pub kind: DefKind,
    /// Capability namespaces explicitly declared by this body.
    pub requires: Vec<CapabilityId>,
    pub family: FamilyId,
    pub dimensions: Vec<Dimension>,
    pub elem_params: Vec<String>,
    /// Concrete elements this definition fixes where its family's contract
    /// has an element parameter.
    pub elem_bindings: Vec<(String, Elem)>,
    pub params: Vec<Param>,
    pub aliases: Vec<(usize, usize)>,
    pub result: ValueType,
    /// Applicability: every predicate must hold.
    pub predicates: Vec<Predicate>,
    pub body: Body,
    pub initialization: crate::initialization::InitializationContract,
    /// The one arena of every extent, bound and symbolic value above.
    pub arena: ExprArena,
    /// Index into the module's source files.
    pub file: usize,
    pub span: Span,
}

impl Definition {
    pub(crate) fn dimension_named(&self, name: &str) -> Option<usize> {
        self.dimensions.iter().position(|d| d.name == name)
    }
}

/// A connected component of same-name implementations with overlapping
/// applicability and a compatible contract.
#[derive(Clone, Debug)]
pub(crate) struct Family {
    pub name: String,
    /// Canonical source contract: stable parameter names, ordering, element
    /// parameters and the domain-defining predicates.
    pub contract: FunctionId,
    /// Portable and backend-specific `fn` bodies.
    pub bodies: Vec<FunctionId>,
    /// Backend `lower` bodies.
    pub lowerings: Vec<FunctionId>,
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct Local {
    pub ownership: super::ownership::ValueOwnership,
    pub name: String,
    pub ty: ValueType,
    pub mutable: bool,
    pub span: Span,
    /// The symbol standing for this local's runtime integer value (index
    /// parameters, loop binders, proven-bounded integers), when it has one.
    pub symbol: Option<SymbolId>,
}

#[derive(Clone, Debug)]
pub(crate) struct Body {
    pub locals: Vec<Local>,
    pub root: Block,
}

#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub statements: Vec<Stmt>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug)]
pub(crate) enum Terminator {
    /// Control continues with the enclosing construct.
    Continue,
    /// The function boundary's result values.
    Return(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub(crate) enum Stmt {
    Let {
        pattern: Pattern,
        mutable: bool,
        value: Expr,
    },
    Assign {
        place: Place,
        op: AssignOp,
        value: Expr,
        /// Fresh exact-value symbols for the installed local SSA versions.
        value_symbols: Vec<(LocalId, SymbolId)>,
        /// Checker-minted proof for each parallel-captured storage root.
        authorities: Vec<ExclusiveWriteCapability>,
    },
    Loop {
        kind: LoopKind,
        binder: LocalId,
        start: Expr,
        end: Expr,
        body: Block,
        /// Captured quantity/word locals: body parameter and exit result.
        value_symbols: Vec<(LocalId, SymbolId, SymbolId)>,
        initialization: crate::initialization::LoopInitialization,
    },
    If {
        condition: Expr,
        then_body: Block,
        else_body: Block,
        /// Active source value version mapped to each branch parameter.
        capture_symbols: Vec<(LocalId, SymbolId)>,
        /// Fresh source value version mapped to each changed join result.
        join_symbols: Vec<(LocalId, SymbolId)>,
    },
    Evaluate(Expr),
}

/// Source-level loop semantics, independent of any physical execution width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LoopKind {
    /// Ascending `for`; captured mutable values are carried across visits.
    Ordered,
    /// Independent `parallel for`; only proved-disjoint or atomic writes.
    Independent,
}

/// Opaque proof that a parallel write's selected regions are disjoint across
/// the exact enclosing logical participants. Only the checker constructs it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExclusiveWriteCapability {
    region: Box<Place>,
    participants: Box<[LocalId]>,
}

impl ExclusiveWriteCapability {
    pub(super) fn checked(region: Place, participants: Vec<LocalId>) -> Self {
        assert!(!participants.is_empty());
        Self {
            region: Box::new(region),
            participants: participants.into_boxed_slice(),
        }
    }
    pub(crate) fn region(&self) -> &Place {
        &self.region
    }
    pub(crate) fn participants(&self) -> &[LocalId] {
        &self.participants
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CheckedAtomicOrder {
    Relaxed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CheckedAtomicScope {
    Participant,
    Participants(Box<[LocalId]>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CheckedAtomicPublication {
    CommandCompletion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CheckedAssociationOutcome {
    Exact,
    Reassociated { accumulator: crate::types::DType },
}

/// Opaque, identity-bound authority for one checked atomic RMW.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AtomicCapability {
    region: Box<Place>,
    participants: Box<[LocalId]>,
    order: CheckedAtomicOrder,
    scope: CheckedAtomicScope,
    publication: CheckedAtomicPublication,
    outcome: CheckedAssociationOutcome,
}

impl AtomicCapability {
    pub(super) fn checked(
        region: Place,
        participants: Vec<LocalId>,
        outcome: CheckedAssociationOutcome,
    ) -> Self {
        let participants = participants.into_boxed_slice();
        let scope = if participants.is_empty() {
            CheckedAtomicScope::Participant
        } else {
            CheckedAtomicScope::Participants(participants.clone())
        };
        Self {
            region: Box::new(region),
            participants,
            order: CheckedAtomicOrder::Relaxed,
            scope,
            publication: CheckedAtomicPublication::CommandCompletion,
            outcome,
        }
    }
    pub(crate) fn region(&self) -> &Place {
        &self.region
    }
    pub(crate) fn participants(&self) -> &[LocalId] {
        &self.participants
    }
    pub(crate) fn order(&self) -> CheckedAtomicOrder {
        self.order
    }
    pub(crate) fn scope(&self) -> &CheckedAtomicScope {
        &self.scope
    }
    pub(crate) fn publication(&self) -> CheckedAtomicPublication {
        self.publication
    }
    pub(crate) fn outcome(&self) -> CheckedAssociationOutcome {
        self.outcome
    }
}

/// Opaque proof for an explicit barrier. Current source syntax has no barrier
/// form, so this can only become inhabited when the checker gains a construct
/// that proves a uniform cohort and its visibility domain together.
#[derive(Clone, Debug)]
pub(crate) struct BarrierCapability {
    cohort: Box<[LocalId]>,
    visibility: Box<[LocalId]>,
}

impl BarrierCapability {
    #[allow(dead_code)]
    pub(super) fn checked(cohort: Vec<LocalId>, visibility: Vec<LocalId>) -> Self {
        assert!(!cohort.is_empty());
        assert!(!visibility.is_empty());
        Self {
            cohort: cohort.into_boxed_slice(),
            visibility: visibility.into_boxed_slice(),
        }
    }
    #[allow(dead_code)]
    pub(crate) fn cohort(&self) -> &[LocalId] {
        &self.cohort
    }
    #[allow(dead_code)]
    pub(crate) fn visibility(&self) -> &[LocalId] {
        &self.visibility
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Pattern {
    Local(LocalId),
    Tuple(Vec<Pattern>),
}

/// A mutable place: a local's storage, an element/selection of it, or a
/// tuple of places (tuple assignment).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Place {
    Local(super::ownership::LocalPlace),
    Element {
        root: super::ownership::LocalPlace,
        indices: Vec<Index>,
    },
    Tuple(Vec<Place>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Index {
    Point {
        value: Expr,
        /// The checker could not prove this data-dependent bound statically.
        runtime_check: bool,
    },
    /// `lo:hi`; `None` bounds are the axis ends.
    Range {
        start: Option<Expr>,
        end: Option<Expr>,
        check_start: bool,
        check_order: bool,
        check_end: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Expr {
    pub kind: ExprKind,
    pub ty: ValueType,
    /// Symbolic value of integer expressions over dimensions and bounded
    /// integer locals, when the checker proved one.
    pub sym: Option<IntExpr>,
    pub span: Span,
}

impl Expr {
    pub(crate) fn new(kind: ExprKind, ty: ValueType, sym: Option<IntExpr>, span: Span) -> Expr {
        Expr {
            kind,
            ty,
            sym,
            span,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExprKind {
    Literal(ReferenceScalar),
    /// A dimension of the definition used as a value; ordinal into
    /// `Definition::dimensions`.
    Dimension(u32),
    Local(LocalId),
    Primitive {
        id: PrimitiveId,
        operands: Vec<Expr>,
    },
    Atomic {
        op: crate::intrinsics::AtomicOp,
        place: Box<Expr>,
        indices: Vec<Expr>,
        value: Box<Expr>,
        authority: AtomicCapability,
    },
    /// A named physical plane of a logical packed representation. This is a
    /// semantic view, not an arithmetic primitive.
    PlaneView {
        base: Box<Expr>,
        plane: u32,
    },
    Intrinsic {
        id: IntrinsicId,
        args: Vec<Expr>,
    },
    Call {
        call: Box<Call>,
        args: Vec<Expr>,
    },
}

/// One static call occurrence: the family it names and, per candidate
/// definition that unifies with the arguments, how its parameters bind.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Call {
    pub family: FamilyId,
    /// Authored shape bindings, in source evaluation order before value arguments.
    pub explicit_shapes: Vec<(String, Expr)>,
    /// Candidates whose unification succeeded, in definition order.
    /// Predicates are *not* evaluated here.
    pub candidates: Vec<Candidate>,
    pub span: Span,
}

/// How one candidate definition's parameters bind at a call occurrence.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub definition: FunctionId,
    /// The callee's dimensions, in the callee's declaration order, as
    /// expressions in the caller's arena.
    pub shape_args: Vec<IntExpr>,
    /// Ordered construction of the same dimensions from reached call inputs.
    pub shape_plan: Vec<(u32, ShapeBindingPlan)>,
    /// Callee element parameter -> element (possibly a caller parameter).
    pub elem_args: Vec<(String, Elem)>,
    /// Argument expression ordinal for each callee parameter.
    pub arg_order: Vec<usize>,
    /// Element parameters of the caller that must equal these concrete
    /// elements for this candidate to apply.
    pub requires_elems: Vec<(String, Elem)>,
    pub applicability_proven: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShapeBindingPlan {
    Explicit { binding: usize },
    /// The inverse of the selected formal axis is
    /// `(observed - formal_axis[dimension := 0]) / coefficient`.
    /// Every remaining dimension in that offset was captured earlier.
    Inferred {
        observation: ShapeObservation,
        /// This candidate definition's checked formal-axis expression, in
        /// its own definition arena. Replacing the selected dimension with
        /// zero yields the checker-selected inverse offset.
        formal_axis: IntExpr,
        coefficient: i64,
        prior_dimensions: Vec<u32>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShapeObservation {
    TensorAxis { parameter: usize, argument: usize, path: Vec<u32>, axis: u32 },
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------

pub(crate) fn walk_block<'a>(block: &'a Block, visit: &mut dyn FnMut(&'a Expr)) {
    for statement in &block.statements {
        walk_stmt(statement, visit);
    }
    if let Terminator::Return(values) = &block.terminator {
        values.iter().for_each(|v| walk_expr(v, visit));
    }
}

pub(crate) fn walk_stmt<'a>(statement: &'a Stmt, visit: &mut dyn FnMut(&'a Expr)) {
    match statement {
        Stmt::Let { value, .. } => walk_expr(value, visit),
        Stmt::Assign { place, value, .. } => {
            walk_place(place, visit);
            walk_expr(value, visit)
        }
        Stmt::Loop {
            start, end, body, ..
        } => {
            walk_expr(start, visit);
            walk_expr(end, visit);
            walk_block(body, visit);
        }
        Stmt::If {
            condition,
            then_body,
            else_body,
            ..
        } => {
            walk_expr(condition, visit);
            walk_block(then_body, visit);
            walk_block(else_body, visit);
        }
        Stmt::Evaluate(expr) => walk_expr(expr, visit),
    }
}

pub(crate) fn walk_place<'a>(place: &'a Place, visit: &mut dyn FnMut(&'a Expr)) {
    match place {
        Place::Local(_) => {}
        Place::Element { indices, .. } => {
            for index in indices {
                match index {
                    Index::Point { value, .. } => walk_expr(value, visit),
                    Index::Range { start, end, .. } => {
                        start.iter().chain(end).for_each(|e| walk_expr(e, visit))
                    }
                }
            }
        }
        Place::Tuple(places) => places.iter().for_each(|p| walk_place(p, visit)),
    }
}

pub(crate) fn walk_expr<'a>(expr: &'a Expr, visit: &mut dyn FnMut(&'a Expr)) {
    visit(expr);
    match &expr.kind {
        ExprKind::Primitive { operands, .. } => operands.iter().for_each(|o| walk_expr(o, visit)),
        ExprKind::Atomic {
            place,
            indices,
            value,
            ..
        } => {
            walk_expr(place, visit);
            indices.iter().for_each(|index| walk_expr(index, visit));
            walk_expr(value, visit);
        }
        ExprKind::PlaneView { base, .. } => walk_expr(base, visit),
        ExprKind::Intrinsic { args, .. } => {
            args.iter().for_each(|a| walk_expr(a, visit))
        }
        ExprKind::Call { call, args } => {
            for (_, value) in &call.explicit_shapes { walk_expr(value, visit); }
            args.iter().for_each(|a| walk_expr(a, visit));
        }
        ExprKind::Literal(_) | ExprKind::Dimension(_) | ExprKind::Local(_) => {}
    }
}

impl Body {
    /// Every call occurrence in this body, in evaluation order.
    pub(crate) fn calls(&self) -> Vec<&Call> {
        let mut out = Vec::new();
        walk_block(&self.root, &mut |expr: &Expr| {
            if let ExprKind::Call { call, .. } = &expr.kind {
                out.push(call.as_ref());
            }
        });
        out
    }

    /// Every static callee definition reachable from this body.
    pub(crate) fn callees(&self) -> Vec<FunctionId> {
        let mut out: Vec<FunctionId> = Vec::new();
        for call in self.calls() {
            for candidate in &call.candidates {
                if !out.contains(&candidate.definition) {
                    out.push(candidate.definition);
                }
            }
        }
        out
    }
}
