//! The crate-private checked representation.
//!
//! One `Definition` is a template checked once: its signature, `where`
//! conjuncts and body are symbolic over its own dimensions in its own
//! expression arena. A call names a family, the one argument order of its
//! members, the callee dimensions solved by the family's dimension plan, and
//! the members that apply at the call. The checked call graph is acyclic.
//!
//! Nothing here is constructible outside `check`.

use super::dimensions::DimensionPlan;
use super::ownership::LocalPlace;
use super::resolve::SignatureDimension;
use crate::checked::ElementDomain;
use crate::expr::{ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, FamilyId, FunctionId, IntrinsicId, StableFunctionId};
use crate::initialization::{InitializationContract, LoopInitialization, VisitSeparation};
use crate::intrinsics::{PrimitiveFailure, PrimitiveId};
use crate::reference_math::ReferenceScalar;
use crate::registry::BackendName;
use crate::span::Span;
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
}

impl std::fmt::Display for LocalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a definition is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefKind {
    /// A portable `fn` body.
    Body,
    /// `lower … for target:` with a body.
    Lower { target: BackendName },
}

impl DefKind {
    pub(crate) fn target(self) -> Option<BackendName> {
        match self {
            DefKind::Body => None,
            DefKind::Lower { target } => Some(target),
        }
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

impl Predicate {
    pub(crate) fn expression(self) -> IntExpr {
        match self {
            Self::NonNegative(e) | Self::Zero(e) | Self::NonZero(e) => e,
        }
    }
}

/// One conjunct of a `where` clause, with the source span of its comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WhereConjunct {
    pub predicate: Predicate,
    pub span: Span,
}

/// Where a definition may be placed relative to its call site's parallel
/// participants (L13).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Placement {
    /// A `WithinEnclosingParallel` intrinsic is used outside any `parallel
    /// for` of this body.
    pub requires_enclosing_parallel: bool,
    /// A `WholeTensor` intrinsic is used by this body.
    pub forbids_enclosing_parallel: bool,
    /// A cohort intrinsic is used outside any `parallel for` of this body.
    pub requires_uniform_call_site: bool,
}

impl Placement {
    /// Whether a call site with `context` definitely cannot place this
    /// definition. An enclosing parallel loop may still come from the
    /// caller's own call site, so that requirement is decided at entry.
    pub(crate) fn excluded_by(self, context: CallContext) -> bool {
        (self.forbids_enclosing_parallel && context.enclosing_parallel)
            || (self.requires_uniform_call_site && !context.participant_uniform)
    }
}

#[derive(Debug)]
pub(crate) struct Definition {
    pub stable: StableFunctionId,
    pub name: String,
    pub kind: DefKind,
    /// Capability namespaces explicitly declared by this body.
    pub requires: Vec<CapabilityId>,
    pub dimensions: Vec<SignatureDimension>,
    pub elem_params: Vec<String>,
    /// Concrete elements this definition fixes where its family's contract
    /// has an element parameter.
    pub elem_bindings: Vec<(String, Elem)>,
    pub params: Vec<Param>,
    pub result: ValueType,
    /// Applicability: every conjunct must hold.
    pub predicates: Vec<WhereConjunct>,
    pub body: Body,
    pub initialization: InitializationContract,
    /// The admissible bindings of each element parameter.
    pub element_domain: ElementDomain,
    pub placement: Placement,
    /// The one arena of every extent, bound and symbolic value above.
    pub arena: ExprArena,
    /// Index into the module's source files.
    pub file: usize,
    pub span: Span,
}

/// A connected component of same-name implementations with overlapping
/// applicability and a compatible contract.
#[derive(Clone, Debug)]
pub(crate) struct Family {
    pub name: String,
    /// The contract body: the first declared portable body. Its meaning is
    /// the family's meaning (L2).
    pub contract: FunctionId,
    /// Portable `fn` bodies.
    pub bodies: Vec<FunctionId>,
    /// Backend `lower` bodies.
    pub lowerings: Vec<FunctionId>,
    /// How every family dimension is obtained from observed input axes,
    /// over the contract's arena.
    pub dimension_plan: DimensionPlan,
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
    /// parameters, loop binders, word and quantity locals), when it has one.
    pub symbol: Option<SymbolId>,
}

#[derive(Clone, Debug)]
pub(crate) struct Body {
    pub locals: Vec<Local>,
    pub root: Block,
    /// The function's one exit: its result values, evaluated after the whole
    /// root block (L5).
    pub result: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub(crate) enum Stmt {
    Let {
        pattern: Pattern,
        value: Expr,
    },
    /// `place = value`; compound assignment is desugared by the checker.
    Assign {
        place: Place,
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
        initialization: LoopInitialization,
        /// L25 (I1)-(I3), recorded by the initialization pass.
        separation: VisitSeparation,
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

#[derive(Clone, Debug)]
pub(crate) enum Pattern {
    Local(LocalId),
    Tuple(Vec<Pattern>),
}

/// A mutable place: a local's storage, an element/selection of it, or a
/// tuple of places (tuple assignment).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Place {
    Local(LocalPlace),
    Element {
        root: LocalPlace,
        indices: Vec<Index>,
    },
    Tuple(Vec<Place>),
}

/// One axis of a selection. Every `check*` flag is `true` exactly when the
/// checker did not prove the corresponding bound.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Index {
    Point {
        value: Expr,
        runtime_check: bool,
    },
    /// `lo:hi`; `None` bounds are the axis ends.
    Range {
        start: Option<Expr>,
        end: Option<Expr>,
        check_start: bool,
        check_order: bool,
        check_end: bool,
        /// L24: a `s : s + w` range whose realized width is not proved `w`.
        check_width: bool,
    },
    /// An omitted trailing axis: the whole axis.
    Full,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Expr {
    pub kind: ExprKind,
    pub ty: ValueType,
    /// Symbolic value of integer expressions over dimensions and bounded
    /// integer locals, when it equals the value exactly.
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
        /// Whether the primitive's scalar recipe can fail at this site.
        failure: PrimitiveFailure,
    },
    /// An atomic update of one element of a tensor place (L12).
    Atomic {
        op: crate::intrinsics::AtomicOp,
        place: LocalPlace,
        indices: Vec<Index>,
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
        overload: IntrinsicOverload,
        args: Vec<Expr>,
    },
    Call {
        call: Box<Call>,
        args: Vec<Expr>,
    },
    /// L31: a data word the checker proved in range at an `index[B]` or
    /// `range[B]` position. Typed `index[B]`; its symbol is the word's.
    IndexPosition(Box<Expr>),
}

/// The rows of one capability intrinsic that are compatible with some
/// admissible binding of the call's operands. Never empty; entry
/// construction resolves exactly one row per specialization.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntrinsicOverload {
    pub capability: CapabilityId,
    pub name: &'static str,
    pub rows: Vec<IntrinsicId>,
}

/// The participant context of a call site within its own body (L13, L14).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CallContext {
    /// The call is under a `parallel for` of the calling body.
    pub enclosing_parallel: bool,
    /// Control at the call is uniform across the participants of the
    /// innermost enclosing `parallel for` (or of the body's own call site).
    pub participant_uniform: bool,
}

/// One static call occurrence.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Call {
    pub family: FamilyId,
    /// Explicit dimension bindings `f[D = e]`, in source order before the
    /// value arguments: dimension ordinal and its checked value.
    pub seeds: Vec<(u32, Expr)>,
    /// Argument expression ordinal for each family parameter (L3: one order
    /// for every member).
    pub arg_order: Vec<usize>,
    /// The family's dimensions at this call, in the caller's arena: the
    /// family dimension plan applied to the actual axes and seeds.
    pub dimensions: Vec<IntExpr>,
    pub context: CallContext,
    /// The members that apply at this call; the contract is always first.
    pub candidates: Vec<Candidate>,
    pub span: Span,
}

/// One family member that applies at a call: its element bindings are
/// compatible, its initialization contract applies here with guarantees
/// including the contract's, and its placement is not excluded.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub definition: FunctionId,
    /// Callee element parameter -> element (possibly a caller parameter).
    pub elem_args: Vec<(String, Elem)>,
    /// Element parameters of the caller that must equal these concrete
    /// elements for this candidate to apply.
    pub requires_elems: Vec<(String, Elem)>,
}
