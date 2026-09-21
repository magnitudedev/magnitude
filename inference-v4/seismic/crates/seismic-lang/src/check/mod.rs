//! The checker: resolution, contract families and target capabilities, typing
//! of every value kind through the intrinsic registry, ownership/moves,
//! borrow rules, loop carry and disjointness, initialization coverage, and
//! `where` predicates. Emits the checked representation directly.
//! Entry point is [`crate::checked::check_source`].
//!
//! Every symbolic integer is an `IntExpr` in the definition's private arena.

mod call;
mod entry_build;
mod expr;
pub(crate) mod ir;
mod prove;
pub(crate) mod resolve;
mod stmt;
mod xfer;

pub(crate) use entry_build::build_entry;

use self::ir::{
    Block as CheckedBlock, Body as CheckedBody, DefKind, Expr as CheckedExpr,
    ExprKind as CheckedExprKind, Index as CheckedIndex, Local as CheckedLocal, LocalId,
    Ownership as ParamOwnership, Predicate,
};
use crate::checked::EntryInfo;
use crate::expr::{ExprArena, IntExpr, SymbolId};
use crate::ids::{CapabilityId, ModuleHash, ModuleId, ProgramId, StableFunctionId};
use crate::intrinsics::PrimitiveId;
use crate::span::{Diagnostic, Span};
use crate::syntax::ast;
use crate::types::{Elem, TensorType, ValueType};
use prove::Facts;
use resolve::{BodySig, Declared, Located, Resolved};
use std::collections::{BTreeSet, HashMap, HashSet};

/// What a local binds, for the checker's ownership and storage analysis. Not
/// part of the checked representation: the canonical type records what a value
/// is, this records how its storage is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalKind {
    Param(usize),
    /// `let mut` state.
    State,
    /// An immutable `let` binding.
    Value,
    /// A loop binder.
    Binder,
}

/// How the storage of a checked expression is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ValueClass {
    Scalar,
    /// A computed dense value (elementwise results, snapshots, decodes, packed reads).
    Computed,
    /// Owned tensor storage (allocations, materializations, clones, call results).
    Owned,
    /// A borrowed view of storage (tensor/view parameters, selections of storage).
    Borrowed,
}

/// Per-definition facts callers need: how each shape parameter is used.
#[derive(Clone, Debug, Default)]
pub(crate) struct Summary {
    /// Shape parameters used as numbers (arithmetic, `extent`, range bounds).
    pub numeric: BTreeSet<String>,
    /// Shape parameters whose axis the body reduces over.
    pub reduces: BTreeSet<String>,
    /// `(callee definition, callee parameter, own parameter)`: passed along unchanged.
    pub passes: Vec<(usize, String, String)>,
    /// Exclusive tensor parameters definitely initialized on every path before return.
    pub full_init: BTreeSet<usize>,
    /// Exclusive own parameter forwarded to every candidate parameter at one call site.
    pub init_passes: Vec<(Vec<(usize, usize)>, usize)>,
}

pub(crate) struct Env<'a> {
    pub resolved: &'a Resolved<'a>,
    pub summaries: &'a [Summary],
    /// Second pass: summaries are complete.
    pub enforce: bool,
}

/// Loop context for the mutation summary and carry analysis.
pub(crate) struct LoopCtx {
    /// Locals declared before the loop (captured floor).
    pub floor: usize,
    /// Captured storage roots written by the body so far.
    pub writes: Vec<LocalId>,
    /// Captured storage roots written whole (not through the binder).
    pub whole_writes: Vec<LocalId>,
    /// Storage roots updated atomically.
    pub atomics: Vec<LocalId>,
}

pub(crate) struct Checker<'a> {
    pub env: &'a Env<'a>,
    pub def: usize,
    pub sig: BodySig,
    pub kind: DefKind,
    /// The target whose forms this body may name: a backend-specific function
    /// or lowering target.
    pub target: Option<crate::registry::BackendName>,
    pub requires: Vec<(CapabilityId, Span)>,
    pub used_capabilities: BTreeSet<CapabilityId>,
    pub locals: Vec<CheckedLocal>,
    pub kinds: Vec<LocalKind>,
    pub scopes: Vec<HashMap<String, LocalId>>,
    pub facts: Facts,
    pub symbols: HashMap<LocalId, SymbolId>,
    pub scalar_symbols: HashMap<LocalId, IntExpr>,
    pub unassigned: HashSet<LocalId>,
    /// Tiles the current complete-traversal loop assigns by its first write at
    /// the loop's own coordinates.
    pub pending_full_assign: Vec<(LocalId, Vec<LocalId>)>,
    /// Complete logical `0..axis` traversals may collectively initialize
    /// uninitialized storage.
    pub init_loop_depth: usize,
    /// Runtime-bounded range views: (start, end, parent extent, realized-length atom).
    pub dyn_views: Vec<(Option<CheckedExpr>, Option<CheckedExpr>, IntExpr, SymbolId)>,
    /// For each view binding: the storage root it selects.
    pub view_roots: HashMap<LocalId, LocalId>,
    /// For each `let`-bound view: how many writes had happened when it was bound.
    pub view_bound: HashMap<LocalId, usize>,
    pub mutated: Vec<LocalId>,
    /// Storage roots read so far.
    pub reads: Vec<LocalId>,
    /// Active independent (`parallel for`) loops: (captured floor, binder).
    pub logical_parallel: Vec<(usize, LocalId)>,
    /// Active loop contexts (innermost last).
    pub loops: Vec<LoopCtx>,
    /// Depth of enclosing loops; `return` is invalid inside.
    pub loop_depth: usize,
    pub summary: Summary,
    pub diagnostics: Vec<Diagnostic>,
    pub counter: usize,
    /// Names whose binding was rejected; uses of them are not reported again.
    pub poisoned: HashSet<String>,
    /// Owned tensor bindings consumed by a source-level move.
    pub moved: HashSet<LocalId>,
    /// Lexically live view borrows: binding -> (storage root, exclusive).
    pub borrows: HashMap<LocalId, (LocalId, bool)>,
    pub arena: ExprArena,
}

impl<'a> Checker<'a> {
    fn new(env: &'a Env<'a>, def: usize) -> Checker<'a> {
        let declared: &'a Declared<'a> = &env.resolved.declared[def];
        let target = declared.kind.target();
        let (sig, arena) = declared.sig.for_body();
        let mut c = Checker {
            env,
            def,
            sig,
            kind: declared.kind.clone(),
            target,
            requires: declared.requires.clone(),
            used_capabilities: BTreeSet::new(),
            locals: Vec::new(),
            kinds: Vec::new(),
            scopes: vec![HashMap::new()],
            facts: Facts::new(),
            symbols: HashMap::new(),
            scalar_symbols: HashMap::new(),
            unassigned: HashSet::new(),
            pending_full_assign: Vec::new(),
            init_loop_depth: 0,
            dyn_views: Vec::new(),
            view_roots: HashMap::new(),
            view_bound: HashMap::new(),
            mutated: Vec::new(),
            reads: Vec::new(),
            logical_parallel: Vec::new(),
            loops: Vec::new(),
            loop_depth: 0,
            summary: Summary::default(),
            diagnostics: Vec::new(),
            counter: 0,
            poisoned: HashSet::new(),
            moved: HashSet::new(),
            borrows: HashMap::new(),
            arena,
        };
        // Shape parameters are positive extents unless a `where` admits zero.
        for (ordinal, _p) in c.sig.shape_params.iter().enumerate() {
            let symbol = c.sig.shape_symbols[ordinal];
            let value = c.arena.int_symbol(symbol);
            let admits_zero = c.sig.predicates.iter().any(
                |q| matches!(q, Predicate::NonNegative(e) if prove::same(&c.arena, *e, value)),
            );
            let lower = c.arena.int(if admits_zero { 0 } else { 1 });
            c.facts.set_range_lower(symbol, lower);
        }
        for predicate in c.sig.predicates.clone() {
            match predicate {
                Predicate::NonNegative(e) => c.assume_nonneg(e),
                Predicate::Zero(e) => c.assume_zero(e),
                Predicate::NonZero(_) => {}
            }
        }
        for (i, p) in c.sig.params.clone().into_iter().enumerate() {
            let id = c.declare(
                &p.name,
                p.ty.clone(),
                p.span,
                LocalKind::Param(i),
                p.ownership == ParamOwnership::Exclusive,
            );
            if let ValueType::Index { bound } = &p.ty {
                let (_, symbol, _) = c.arena.loop_binder();
                let zero = c.arena.int(0);
                let one = c.arena.int(1);
                let upper = c.arena.int_sub(*bound, one);
                c.facts.set_range(symbol, zero, upper);
                c.symbols.insert(id, symbol);
                c.locals[id.index()].symbol = Some(symbol);
            }
            if let ValueType::Tensor(t) = &p.ty {
                if p.ownership == ParamOwnership::Owned {
                    // An owned tensor parameter starts initialized.
                    let _ = t;
                }
            }
        }
        c
    }

    pub fn error(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::new(span, message));
    }

    pub fn use_capability(&mut self, capability: &CapabilityId, span: Span, use_site: &str) {
        self.used_capabilities.insert(*capability);
        if !self
            .requires
            .iter()
            .any(|(declared, _)| declared == capability)
        {
            self.error(
                span,
                format!(
                    "{use_site} requires capability `{}.{}`; add `requires {}.{}` to this declaration",
                    crate::registry::capability_info(*capability).backend.as_str(),
                    crate::registry::capability_info(*capability).name,
                    crate::registry::capability_info(*capability).backend.as_str(),
                    crate::registry::capability_info(*capability).name
                ),
            );
        }
    }

    pub fn lookup(&self, name: &str) -> Option<LocalId> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    pub fn declare(
        &mut self,
        name: &str,
        ty: ValueType,
        span: Span,
        kind: LocalKind,
        mutable: bool,
    ) -> LocalId {
        let id = LocalId::new(
            u32::try_from(self.locals.len()).expect("definition has more than u32::MAX locals"),
        );
        self.locals.push(CheckedLocal {
            name: name.to_string(),
            ty,
            mutable,
            span,
            symbol: None,
        });
        self.kinds.push(kind);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), id);
        }
        id
    }

    pub fn fresh_symbol(&mut self, _base: &str) -> SymbolId {
        self.counter += 1;
        self.arena.loop_binder().1
    }

    /// Assume `e >= 0` as bounds on every atom with a unit coefficient.
    pub fn assume_nonneg(&mut self, e: IntExpr) {
        for symbol in prove::symbols(&self.arena, e) {
            match prove::linear_in(&mut self.arena, e, symbol) {
                Some((1, rest)) => {
                    let zero = self.arena.int(0);
                    let neg = self.arena.int_sub(zero, rest);
                    self.facts.add_lower(symbol, neg)
                }
                Some((-1, rest)) => self.facts.add_upper(symbol, rest),
                _ => {}
            }
        }
    }

    /// Assume `e == 0`: a zero fact, and both bounds on every atom with a unit coefficient.
    pub fn assume_zero(&mut self, e: IntExpr) {
        self.facts.assume_zero(&self.arena, e);
        self.assume_nonneg(e);
        let zero = self.arena.int(0);
        let neg = self.arena.int_sub(zero, e);
        self.assume_nonneg(neg);
    }

    fn is_shape_param(&self, name: &str) -> bool {
        self.sig.shape_params.iter().any(|p| p == name)
    }

    /// Record that the value of these shape parameters is observed as a number.
    pub fn numeric_use(&mut self, value: IntExpr) {
        for symbol in prove::symbols(&self.arena, value) {
            if let Some(ordinal) = self
                .sig
                .shape_symbols
                .iter()
                .position(|candidate| *candidate == symbol)
            {
                self.summary
                    .numeric
                    .insert(self.sig.shape_params[ordinal].clone());
            }
        }
    }

    /// Prove `e >= 0`. Shape-arithmetic needs become a diagnostic asking for a `where`.
    pub fn require_nonneg(&mut self, e: IntExpr, span: Span, what: &str) {
        if prove::nonneg(&self.arena, &self.facts, e) {
            return;
        }
        let rendered = prove::display(&self.arena, e, &|symbol| {
            self.sig
                .shape_symbols
                .iter()
                .position(|candidate| *candidate == symbol)
                .map(|ordinal| self.sig.shape_params[ordinal].clone())
                .unwrap_or_else(|| format!("{symbol:?}"))
        });
        self.error(span, format!("{what}: cannot prove `{rendered} >= 0`"));
    }

    // ---- types ----

    pub fn same_extent(&self, a: IntExpr, b: IntExpr) -> bool {
        prove::same(&self.arena, a, b)
    }

    pub fn same_axes(&self, a: &TensorType, b: &TensorType) -> bool {
        a.rank() == b.rank()
            && a.axes
                .iter()
                .zip(&b.axes)
                .all(|(x, y)| self.same_extent(*x, *y))
    }

    pub fn same_ty(&self, a: &ValueType, b: &ValueType) -> bool {
        match (a, b) {
            (ValueType::Tensor(x), ValueType::Tensor(y)) => {
                x.elem == y.elem && self.same_axes(x, y)
            }
            (ValueType::Index { bound: x }, ValueType::Index { bound: y })
            | (ValueType::Range { bound: x }, ValueType::Range { bound: y }) => {
                self.same_extent(*x, *y)
            }
            (ValueType::Tuple(x), ValueType::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| self.same_ty(p, q))
            }
            _ => a == b,
        }
    }

    /// Whether a value of type `value` may be installed into state of type
    /// `target` (floats round to the target's element type).
    pub fn assignable(&self, target: &ValueType, value: &ValueType) -> bool {
        match (target, value) {
            (ValueType::Scalar(a), _) => value
                .scalar_dtype()
                .is_some_and(|b| *a == b || (a.is_float() && b.is_float())),
            (ValueType::Tensor(a), ValueType::Tensor(b)) => {
                self.same_axes(a, b) && elem_rounds(&b.elem, &a.elem)
            }
            (ValueType::Tuple(a), ValueType::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| self.assignable(x, y))
            }
            _ => self.same_ty(target, value),
        }
    }

    // ---- target-dependent forms ----

    /// A target-dependent form. Legal only in a body with a target context.
    pub fn target_form(&mut self, span: Span, what: &str, namespace: Option<&str>) -> bool {
        let Some(target) = self.target.clone() else {
            self.error(span, format!("{what} is target-dependent and cannot appear in a portable body; write it in a `fn … for <target>` or `lower … for <target>` body"));
            return false;
        };
        if namespace
            .and_then(crate::registry::BackendName::parse)
            .is_some_and(|ns| ns != target)
        {
            self.error(
                span,
                format!(
                    "{what} belongs to target `{}` but this body is for `{}`",
                    namespace.unwrap_or_default(),
                    target.as_str()
                ),
            );
            return false;
        }
        true
    }

    // ---- storage class ----

    /// The storage root a place or value expression designates, if any.
    pub fn root_var(&self, e: &CheckedExpr) -> Option<LocalId> {
        match &e.kind {
            CheckedExprKind::Local(v) => Some(self.view_roots.get(v).copied().unwrap_or(*v)),
            CheckedExprKind::Primitive { id, operands } => match id {
                PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                    operands.first().and_then(|b| self.root_var(b))
                }
                _ => None,
            },
            CheckedExprKind::PlaneView { base, .. } => self.root_var(base),
            CheckedExprKind::Atomic { .. } => None,
            _ => None,
        }
    }

    /// How the storage of an expression is reached.
    pub fn class_of(&self, e: &CheckedExpr) -> ValueClass {
        match &e.kind {
            CheckedExprKind::Literal(_) => ValueClass::Scalar,
            CheckedExprKind::Dimension(_) => ValueClass::Scalar,
            CheckedExprKind::Local(v) => {
                if self.view_roots.contains_key(v) {
                    return ValueClass::Borrowed;
                }
                match self.kinds[v.index()] {
                    LocalKind::Param(i) => match self.sig.params[i].ownership {
                        ParamOwnership::Owned => {
                            if matches!(self.locals[v.index()].ty, ValueType::Tensor(_)) {
                                ValueClass::Owned
                            } else {
                                ValueClass::Scalar
                            }
                        }
                        ParamOwnership::Shared | ParamOwnership::Exclusive => {
                            if matches!(self.locals[v.index()].ty, ValueType::Tensor(_)) {
                                ValueClass::Borrowed
                            } else {
                                ValueClass::Scalar
                            }
                        }
                        ParamOwnership::Value => ValueClass::Scalar,
                    },
                    LocalKind::State => {
                        if matches!(self.locals[v.index()].ty, ValueType::Tensor(_)) {
                            ValueClass::Owned
                        } else {
                            ValueClass::Scalar
                        }
                    }
                    LocalKind::Value => ValueClass::Computed,
                    LocalKind::Binder => ValueClass::Scalar,
                }
            }
            CheckedExprKind::Primitive { id, .. } => match id {
                PrimitiveId::Constant(_) | PrimitiveId::Symbolic(_) => ValueClass::Scalar,
                PrimitiveId::ElementRead { .. } => ValueClass::Scalar,
                PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                    ValueClass::Borrowed
                }
                PrimitiveId::Load
                | PrimitiveId::Decode
                | PrimitiveId::Unary(_)
                | PrimitiveId::Binary(_)
                | PrimitiveId::Cast(_)
                | PrimitiveId::Math(_)
                | PrimitiveId::Select
                | PrimitiveId::Reduce { .. }
                | PrimitiveId::Extent { .. } => {
                    if e.ty.scalar_dtype().is_some() {
                        ValueClass::Scalar
                    } else {
                        ValueClass::Computed
                    }
                }
                PrimitiveId::TensorAlloc
                | PrimitiveId::Fill(_)
                | PrimitiveId::Materialize
                | PrimitiveId::Clone
                | PrimitiveId::RepresentationConvert(_) => ValueClass::Owned,
                PrimitiveId::TuplePack => ValueClass::Computed,
                PrimitiveId::TupleGet(_)
                | PrimitiveId::RangeMake
                | PrimitiveId::RangeStart
                | PrimitiveId::RangeEnd => ValueClass::Scalar,
                PrimitiveId::Atomic { .. } => ValueClass::Scalar,
            },
            CheckedExprKind::PlaneView { .. } => ValueClass::Borrowed,
            CheckedExprKind::Atomic { .. } => ValueClass::Scalar,
            CheckedExprKind::Intrinsic { .. } => ValueClass::Scalar,
            CheckedExprKind::Call { .. } => match &e.ty {
                ValueType::Tensor(_) => ValueClass::Owned,
                ValueType::Tuple(_) => ValueClass::Computed,
                _ => ValueClass::Scalar,
            },
        }
    }

    /// Whether writes may target the storage rooted at `id`.
    pub fn writable_root(&self, id: LocalId) -> bool {
        match self.kinds[id.index()] {
            LocalKind::Param(i) => self.sig.params[i].ownership == ParamOwnership::Exclusive,
            LocalKind::State => true,
            _ => false,
        }
    }

    /// Recover the exact participant identity whose disjointness proof was
    /// consumed by `write`. Sequential writes need no parallel authority.
    pub(crate) fn exclusive_write_authority(
        &self,
        root: LocalId,
        region: &ir::Place,
    ) -> Vec<ir::ExclusiveWriteCapability> {
        let participants = self
            .logical_parallel
            .iter()
            .filter(|(floor, _)| root.index() < *floor)
            .map(|(_, binder)| *binder)
            .collect::<Vec<_>>();
        if participants.is_empty() {
            Vec::new()
        } else {
            vec![ir::ExclusiveWriteCapability::checked(
                region.clone(),
                participants,
            )]
        }
    }

    // ---- effects ----

    /// Prove that distinct visits of the enclosing `parallel for` loops in
    /// `binders` write distinct elements through `indices`.
    ///
    /// A binder is proven by a point axis whose index is affine in it with a
    /// nonzero coefficient once every other binder on that axis is already
    /// proven; several unproven binders on one axis are proven together when
    /// their coefficients form a mixed radix over the binders' ranges (each
    /// coefficient exceeds the reach of the smaller ones). A slice
    /// `c*v + d : c*v + d + len` proves `v` when `len <= c`. Data-dependent,
    /// nonlinear, and unbounded indices prove nothing. Returns the first
    /// binder that stays unproven.
    fn disjoint_visits(
        &self,
        indices: &[CheckedIndex],
        binders: &[LocalId],
    ) -> Result<(), LocalId> {
        let symbols: Vec<_> = binders
            .iter()
            .map(|binder| self.symbols.get(binder).copied())
            .collect();
        if symbols.iter().any(Option::is_none) {
            return Err(binders[symbols
                .iter()
                .position(Option::is_none)
                .expect("missing symbol was observed")]);
        }
        let axes: Vec<_> = indices
            .iter()
            .map(|index| match index {
                CheckedIndex::Point { value, .. } => value
                    .sym
                    .map(prove::WriteAxis::Point)
                    .unwrap_or(prove::WriteAxis::Opaque),
                CheckedIndex::Range {
                    start: Some(start),
                    end: Some(end),
                    ..
                } => match (start.sym, end.sym) {
                    (Some(start), Some(end)) => prove::WriteAxis::Slice { start, end },
                    _ => prove::WriteAxis::Opaque,
                },
                CheckedIndex::Range { .. } => prove::WriteAxis::Opaque,
            })
            .collect();
        let symbols: Vec<_> = symbols.into_iter().flatten().collect();
        prove::disjoint_visits(&self.arena, &self.facts, &axes, &symbols)
            .map_err(|ordinal| binders[ordinal])
    }

    /// Check and record a write to the storage `place` designates. `whole` is
    /// an update of the state object itself (assignment, `inout` of the whole
    /// variable).
    pub fn write(
        &mut self,
        root: LocalId,
        binding: LocalId,
        indices: &[CheckedIndex],
        whole: bool,
        span: Span,
    ) -> Option<LocalId> {
        let binding_name = self.locals[binding.index()].name.clone();
        match &self.kinds[binding.index()] {
            LocalKind::Param(i) if self.sig.params[*i].ownership == ParamOwnership::Owned => {
                self.error(
                    span,
                    format!("`{binding_name}` is a read-only moved-in parameter; writing requires `&mut tensor`"),
                );
                return None;
            }
            LocalKind::Param(_) | LocalKind::State => {}
            LocalKind::Value if self.view_roots.contains_key(&binding) => {}
            _ => {
                self.error(
                    span,
                    format!("`{binding_name}` is not mutable state; only `let mut` bindings and `&mut tensor` parameters are written"),
                );
                return None;
            }
        }
        if !self.writable_root(root) {
            self.error(
                span,
                format!(
                    "`{}` is not writable storage; writing requires `let mut` state or a `&mut tensor` parameter",
                    self.locals[root.index()].name
                ),
            );
            return None;
        }
        if let ValueType::Tensor(tensor) = &self.locals[root.index()].ty {
            if let Elem::Repr(representation) = &tensor.elem {
                if crate::registry::representation_info(*representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
                {
                    self.error(
                        span,
                        format!(
                            "representation `{}` is decode-only and has no canonical write contract",
                            crate::registry::representation_info(*representation).name
                        ),
                    );
                    return None;
                }
            }
        }
        if self
            .borrows
            .iter()
            .any(|(borrow, (borrowed, _))| *borrowed == root && *borrow != binding)
        {
            self.error(
                span,
                format!(
                    "cannot mutate `{}` while a tensor borrow is live",
                    self.locals[root.index()].name
                ),
            );
            return None;
        }
        let capturing: Vec<LocalId> = self
            .logical_parallel
            .iter()
            .filter(|(floor, _)| root.index() < *floor)
            .map(|(_, binder)| *binder)
            .collect();
        if !capturing.is_empty() {
            if whole {
                self.error(
                    span,
                    "a `parallel for` body may mutate captured tensor storage only through an index that depends on its loop variable; a whole-value update is shared by every visit",
                );
                return None;
            }
            if let Err(binder) = self.disjoint_visits(indices, &capturing) {
                let binder_name = self.locals[binder.index()].name.clone();
                let root_name = self.locals[root.index()].name.clone();
                self.error(
                    span,
                    format!(
                        "a `parallel for` body may mutate captured tensor storage only through an index that depends on its loop variable injectively: distinct visits of `{binder_name}` are not proved to write distinct elements of `{root_name}`; index with `{binder_name}`, `c*{binder_name} + d` or `{binder_name}*c : ({binder_name}+1)*c`, or update with `atomic(add|max|min, place, value)`"
                    ),
                );
                return None;
            }
        }
        // Record the write in the enclosing loop contexts (carry/disjointness).
        for ctx in self.loops.iter_mut() {
            if root.index() < ctx.floor {
                ctx.writes.push(root);
                if whole {
                    ctx.whole_writes.push(root);
                }
            }
        }
        self.mutated.push(root);
        self.scalar_symbols.remove(&root);
        let tensor_effect = matches!(self.locals[root.index()].ty, ValueType::Tensor(_));
        let root_exprs = self.dyn_views.clone();
        let _ = root_exprs;
        self.dyn_views.retain(|(start, end, _, _)| {
            let mentions =
                |e: &Option<CheckedExpr>| e.as_ref().is_some_and(|b| expr::mentions_local(b, root));
            !tensor_effect && !(mentions(start) || mentions(end))
        });
        Some(root)
    }

    // ---- result ----

    fn finish(
        mut self,
        root: CheckedBlock,
        span: Span,
    ) -> (CheckedBody, BodySig, Summary, Vec<Diagnostic>, ExprArena) {
        let well_formed = self.diagnostics.is_empty();
        if well_formed
            && self.sig.result != ValueType::Void
            && !matches!(root.terminator, ir::Terminator::Return(_))
        {
            self.error(
                span,
                format!(
                    "`{}` returns {} but not every path ends in `return`",
                    self.sig.name, self.sig.result
                ),
            );
        }
        for (capability, declared_at) in self.requires.clone() {
            if !self.used_capabilities.contains(&capability) {
                self.error(
                    declared_at,
                    format!(
                        "capability `{}.{}` is required but not used directly or through a backend-specific helper",
                        crate::registry::capability_info(capability).backend.as_str(),
                        crate::registry::capability_info(capability).name
                    ),
                );
            }
        }
        let body = CheckedBody {
            locals: self.locals,
            root,
        };
        (body, self.sig, self.summary, self.diagnostics, self.arena)
    }
}

/// Whether publishing/assigning elements of `value` into storage of `target`
/// is a defined rounding.
pub(crate) fn elem_rounds(value: &Elem, target: &Elem) -> bool {
    match (value, target) {
        (Elem::Repr(a), Elem::Repr(b)) => a == b,
        (Elem::Repr(_), _) | (_, Elem::Repr(_)) => false,
        (Elem::Dtype(a), Elem::Dtype(b)) => a == b || (a.is_float() && b.is_float()),
        (Elem::Dtype(a), Elem::Param(_)) | (Elem::Param(_), Elem::Dtype(a)) => a.is_float(),
        (Elem::Param(_), Elem::Param(_)) => true,
    }
}

struct CheckedOutcome {
    body: CheckedBody,
    summary: Summary,
    diagnostics: Vec<Diagnostic>,
    arena: ExprArena,
    signature: BodySig,
}

fn check_definition(env: &Env, def: usize) -> CheckedOutcome {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    let root = c.block(declared.body);
    let facts = c.facts.clone();
    let (body, signature, mut summary, diagnostics, mut arena) = c.finish(root, declared.name_span);
    for (parameter, declared_param) in declared.sig.params.iter().enumerate() {
        if declared_param.ownership == ParamOwnership::Exclusive {
            if let ValueType::Tensor(shaped) = &body.locals[parameter].ty {
                if definitely_initializes(&mut arena, &body, parameter, shaped, &facts) {
                    summary.full_init.insert(parameter);
                }
            }
        }
    }
    CheckedOutcome {
        body,
        summary,
        diagnostics,
        arena,
        signature,
    }
}

/// Whether a body definitely initializes every element of `parameter`.
/// Parameters are the first locals of a checked body, in parameter order.
fn definitely_initializes(
    arena: &mut ExprArena,
    body: &CheckedBody,
    parameter: usize,
    shaped: &TensorType,
    facts: &Facts,
) -> bool {
    writes_cover(
        arena,
        &body.locals,
        &body.root,
        LocalId::new(u32::try_from(parameter).expect("parameter ordinal exceeds u32::MAX")),
        &shaped.axes,
        &[],
        facts,
    )
}

/// Whether the block writes every element of `local` on every path, under the
/// complete traversals in `loops` (binder, bound).
pub(crate) fn writes_cover(
    arena: &mut ExprArena,
    locals: &[CheckedLocal],
    block: &CheckedBlock,
    local: LocalId,
    extents: &[IntExpr],
    loops: &[(LocalId, IntExpr)],
    facts: &Facts,
) -> bool {
    fn point_covers(
        arena: &mut ExprArena,
        locals: &[CheckedLocal],
        point: &CheckedExpr,
        extent: IntExpr,
        loops: &[(LocalId, IntExpr)],
    ) -> bool {
        match &point.kind {
            CheckedExprKind::Local(var) => loops
                .iter()
                .any(|(loop_var, bound)| loop_var == var && prove::same(arena, *bound, extent)),
            _ => {
                if prove::constant(arena, extent) == Some(1)
                    && point.sym.is_some_and(|value| prove::is_zero(arena, value))
                {
                    return true;
                }
                let mut total = arena.int(1);
                let mut linear = arena.int(0);
                for (var, bound) in loops {
                    let Some(symbol) = locals[var.index()].symbol else {
                        return false;
                    };
                    let coordinate = arena.int_symbol(symbol);
                    let scaled = arena.int_mul(linear, *bound);
                    linear = arena.int_add(scaled, coordinate);
                    total = arena.int_mul(total, *bound);
                }
                prove::same(arena, total, extent)
                    && point
                        .sym
                        .is_some_and(|value| prove::same(arena, value, linear))
            }
        }
    }
    fn index_covers(
        arena: &mut ExprArena,
        locals: &[CheckedLocal],
        index: &CheckedIndex,
        extent: IntExpr,
        loops: &[(LocalId, IntExpr)],
        facts: &Facts,
    ) -> bool {
        match index {
            CheckedIndex::Point { value, .. } => point_covers(arena, locals, value, extent, loops),
            CheckedIndex::Range { start, end, .. } => {
                let full = start.as_ref().is_none_or(|start| {
                    start.sym.is_some_and(|value| prove::is_zero(arena, value))
                }) && end.as_ref().is_none_or(|end| {
                    end.sym
                        .is_some_and(|value| prove::same(arena, value, extent))
                });
                full || loops.iter().any(|(var, partitions)| {
                    let width = arena.int_div(extent, *partitions);
                    let covered = arena.int_mul(width, *partitions);
                    let difference = arena.int_sub(covered, extent);
                    if !prove::zero(arena, facts, difference) {
                        return false;
                    }
                    let Some(symbol) = locals[var.index()].symbol else {
                        return false;
                    };
                    let coordinate = arena.int_symbol(symbol);
                    let expected_start = arena.int_mul(coordinate, width);
                    let one = arena.int(1);
                    let next = arena.int_add(coordinate, one);
                    let expected_end = arena.int_mul(next, width);
                    start
                        .as_ref()
                        .and_then(|value| value.sym)
                        .is_some_and(|value| prove::same(arena, value, expected_start))
                        && end
                            .as_ref()
                            .and_then(|value| value.sym)
                            .is_some_and(|value| prove::same(arena, value, expected_end))
                })
            }
        }
    }
    fn place_covers(
        arena: &mut ExprArena,
        locals: &[CheckedLocal],
        place: &ir::Place,
        local: LocalId,
        extents: &[IntExpr],
        loops: &[(LocalId, IntExpr)],
        facts: &Facts,
    ) -> bool {
        let (root, indices) = match place {
            ir::Place::Local(root) => return *root == local,
            ir::Place::Element { root, indices } => (root, indices),
            ir::Place::Tuple(_) => return false,
        };
        if *root != local {
            return false;
        }
        if indices.len() > extents.len() {
            return false;
        }
        indices
            .iter()
            .zip(extents)
            .all(|(index, extent)| index_covers(arena, locals, index, *extent, loops, facts))
        // Omitted trailing indices denote the complete remaining tensor slice.
    }
    fn block_writes(
        arena: &mut ExprArena,
        locals: &[CheckedLocal],
        block: &CheckedBlock,
        local: LocalId,
        extents: &[IntExpr],
        loops: &[(LocalId, IntExpr)],
        facts: &Facts,
    ) -> bool {
        // Every path must write: an `if` covers only when both arms cover.
        for statement in &block.statements {
            match statement {
                ir::Stmt::Assign { place, .. } => {
                    if place_covers(arena, locals, place, local, extents, loops, facts) {
                        return true;
                    }
                }
                ir::Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if block_writes(arena, locals, then_body, local, extents, loops, facts)
                        && block_writes(arena, locals, else_body, local, extents, loops, facts)
                    {
                        return true;
                    }
                }
                ir::Stmt::Loop {
                    binder,
                    start,
                    end,
                    body,
                    ..
                } => {
                    let (Some(start), Some(end)) = (start.sym, end.sym) else {
                        continue;
                    };
                    if !prove::is_zero(arena, start) {
                        continue;
                    }
                    let mut nested = loops.to_vec();
                    nested.push((*binder, end));
                    if block_writes(arena, locals, body, local, extents, &nested, facts) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    block_writes(arena, locals, block, local, extents, loops, facts)
}

/// Close numeric and reduction uses over parameters passed along unchanged to callees.
fn close_summaries(summaries: &mut [Summary]) {
    loop {
        let mut changed = false;
        for i in 0..summaries.len() {
            for (callee, callee_param, own) in summaries[i].passes.clone() {
                if summaries[callee].numeric.contains(&callee_param)
                    && summaries[i].numeric.insert(own.clone())
                {
                    changed = true;
                }
                if summaries[callee].reduces.contains(&callee_param)
                    && summaries[i].reduces.insert(own)
                {
                    changed = true;
                }
            }
            for (candidates, own) in summaries[i].init_passes.clone() {
                if !candidates.is_empty()
                    && candidates
                        .iter()
                        .all(|(callee, parameter)| summaries[*callee].full_init.contains(parameter))
                    && summaries[i].full_init.insert(own)
                {
                    changed = true;
                }
            }
        }
        if !changed {
            return;
        }
    }
}

/// The checked static call graph must be acyclic; recursion is rejected before
/// specialization because execution families are finite.
fn reject_cycles(definitions: &[ir::Definition], diagnostics: &mut Vec<Located>) {
    let count = definitions.len();
    // 0 = unvisited, 1 = on stack, 2 = done.
    let mut state = vec![0u8; count];
    let mut stack: Vec<usize> = Vec::new();
    fn visit(
        d: usize,
        definitions: &[ir::Definition],
        state: &mut [u8],
        stack: &mut Vec<usize>,
        cycle: &mut Option<usize>,
    ) {
        match state[d] {
            2 => return,
            1 => {
                if cycle.is_none() {
                    *cycle = stack
                        .get(stack.iter().position(|&s| s == d).unwrap_or(0))
                        .copied();
                }
                return;
            }
            _ => {}
        }
        state[d] = 1;
        stack.push(d);
        for callee in definitions[d].body.callees() {
            let c = callee.index();
            if c < definitions.len() {
                visit(c, definitions, state, stack, cycle);
            }
        }
        stack.pop();
        state[d] = 2;
    }
    for d in 0..count {
        let mut cycle = None;
        visit(d, definitions, &mut state, &mut stack, &mut cycle);
        if let Some(cycle_root) = cycle {
            let definition = &definitions[cycle_root];
            diagnostics.push(Located {
                file: definition.file,
                diagnostic: crate::span::Diagnostic::new(
                    definition.span,
                    format!(
                        "`{}` participates in a recursive call chain: the checked call graph must be acyclic and recursion is rejected before specialization",
                        definition.name
                    ),
                ),
            });
        }
    }
}

/// Check every declared body of the closed program. Returns the definitions and families.
pub(crate) fn check_program(
    files: &[(usize, ast::File)],
    sources: &crate::checked::SourceSet,
    program: ProgramId,
    diagnostics: &mut Vec<Located>,
) -> (Vec<ir::Definition>, Vec<ir::Family>) {
    let resolved = resolve::resolve(files, program, diagnostics);
    let count = resolved.declared.len();

    // First pass: usage summaries only. Second pass: the checked bodies and diagnostics.
    let mut summaries = vec![Summary::default(); count];
    {
        let empty = vec![Summary::default(); count];
        let env = Env {
            resolved: &resolved,
            summaries: &empty,
            enforce: false,
        };
        for (def, summary) in summaries.iter_mut().enumerate() {
            *summary = check_definition(&env, def).summary;
        }
    }
    close_summaries(&mut summaries);
    let env = Env {
        resolved: &resolved,
        summaries: &summaries,
        enforce: true,
    };
    let mut definitions = Vec::with_capacity(count);
    for (def, declared) in resolved.declared.iter().enumerate() {
        let mut checked = check_definition(&env, def);
        diagnostics.extend(checked.diagnostics.into_iter().map(|diagnostic| Located {
            file: declared.file,
            diagnostic,
        }));
        let params = checked
            .signature
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| ir::Param {
                name: p.name.clone(),
                ownership: p.ownership,
                ty: p.ty.clone(),
                local: LocalId::new(
                    u32::try_from(i).expect("definition has more than u32::MAX parameters"),
                ),
                span: p.span,
            })
            .collect();
        let mut stable_hasher = sha2::Sha256::new();
        use sha2::Digest as _;
        stable_hasher.update(crate::registry::REGISTRY_REVISION.as_bytes());
        let source = &sources.files()[declared.file];
        stable_hasher.update(source.path.as_bytes());
        stable_hasher.update(source.text.as_bytes());
        stable_hasher.update((def as u64).to_le_bytes());
        let stable = StableFunctionId::new(stable_hasher.finalize().into());
        let dimensions = checked.signature.shape_params.iter().enumerate().map(|(ordinal, name)| {
            let symbol = checked.signature.shape_symbols[ordinal];
            let value = checked.arena.int_symbol(symbol);
            let admits_zero = checked.signature.predicates.iter().any(|predicate| {
                matches!(predicate, Predicate::NonNegative(expression) if prove::same(&checked.arena, *expression, value))
            });
            ir::Dimension { name: name.clone(), symbol, admits_zero }
        }).collect();
        definitions.push(ir::Definition {
            stable,
            name: declared.sig.name.clone(),
            kind: declared.kind.clone(),
            requires: declared
                .requires
                .iter()
                .map(|(capability, _)| *capability)
                .collect(),
            family: declared.family,
            dimensions,
            elem_params: declared.sig.elem_params.clone(),
            elem_bindings: declared.elem_bindings.clone(),
            params,
            aliases: checked.signature.aliases,
            result: checked.signature.result,
            predicates: checked.signature.predicates,
            body: checked.body,
            arena: checked.arena,
            file: declared.file,
            span: declared.span,
        });
    }
    if diagnostics.is_empty() {
        reject_cycles(&definitions, diagnostics);
    }
    (definitions, resolved.families)
}

pub(crate) fn check_closed(
    sources: crate::checked::SourceSet,
    module_id: ModuleId,
    program: ProgramId,
) -> Result<crate::checked::internals::Module, crate::checked::SourceError> {
    use crate::checked::{Diagnostics, SourceDiagnostic, SourceError};
    let mut parsed = Vec::new();
    let mut parse_diagnostics = Vec::new();
    for (file, source) in sources.files().iter().enumerate() {
        match crate::syntax::parse(&source.text) {
            Ok(ast) => parsed.push((file, ast)),
            Err(diagnostic) => parse_diagnostics.push(SourceDiagnostic {
                path: source.path.clone(),
                span: diagnostic.span,
                message: diagnostic.message,
            }),
        }
    }
    if let Some(diagnostics) = Diagnostics::new(parse_diagnostics) {
        return Err(SourceError::Parse(diagnostics));
    }

    let mut located = Vec::new();
    let (definitions, families) = check_program(&parsed, &sources, program, &mut located);
    if !located.is_empty() {
        located.sort_by_key(|item| (item.file, item.diagnostic.span.start));
        located
            .dedup_by(|left, right| left.file == right.file && left.diagnostic == right.diagnostic);
        let diagnostics = located
            .into_iter()
            .map(|item| SourceDiagnostic {
                path: sources.files()[item.file].path.clone(),
                span: item.diagnostic.span,
                message: item.diagnostic.message,
            })
            .collect();
        return Err(SourceError::Type(
            Diagnostics::new(diagnostics).expect("nonempty checker diagnostics disappeared"),
        ));
    }

    use sha2::Digest as _;
    let mut module_hasher = sha2::Sha256::new();
    module_hasher.update(crate::bundle::COMPILER_SEMANTIC_VERSION.as_bytes());
    module_hasher.update(crate::registry::REGISTRY_REVISION.as_bytes());
    for source in sources.files() {
        module_hasher.update((source.path.len() as u64).to_le_bytes());
        module_hasher.update(source.path.as_bytes());
        module_hasher.update((source.text.len() as u64).to_le_bytes());
        module_hasher.update(source.text.as_bytes());
    }
    let semantic_hash = ModuleHash::new(module_hasher.finalize().into());
    let mut entries = Vec::new();
    let mut entry_families = Vec::new();
    let mut entry_diagnostics = Vec::new();
    for (family_ordinal, family) in families.iter().enumerate() {
        let Some(contract) = definitions.get(family.contract.index()) else {
            panic!("checked family contract is outside the checked definition arena");
        };
        if !contract.kind.is_portable_body() {
            continue;
        }
        if let Err(message) = entry_build::validate_external_dimension_inference(contract) {
            entry_diagnostics.push(SourceDiagnostic {
                path: sources.files()[contract.file].path.clone(),
                span: contract.span,
                message,
            });
            continue;
        }
        let ordinal = u32::try_from(entries.len()).expect("module has more than u32::MAX entries");
        let id = crate::ids::EntryId::new(module_id, ordinal);
        let mut stable_hasher = sha2::Sha256::new();
        stable_hasher.update(semantic_hash.digest());
        stable_hasher.update((family_ordinal as u64).to_le_bytes());
        let stable = crate::ids::StableEntryId::new(stable_hasher.finalize().into());
        entries.push(entry_info(id, stable, contract));
        entry_families.push(family_ordinal);
    }
    if let Some(diagnostics) = Diagnostics::new(entry_diagnostics) {
        return Err(SourceError::Type(diagnostics));
    }
    Ok(crate::checked::internals::Module {
        id: module_id,
        template_program: program,
        semantic_hash,
        sources,
        entries,
        entry_families,
        definitions,
        families,
    })
}

fn element_summary(element: &Elem) -> crate::checked::ElementSummary {
    match element {
        Elem::Dtype(dtype) => crate::checked::ElementSummary::Fixed(dtype.name().to_owned()),
        Elem::Repr(representation) => crate::checked::ElementSummary::Fixed(
            crate::registry::representation_info(*representation)
                .name
                .to_owned(),
        ),
        Elem::Param(name) => crate::checked::ElementSummary::Parameter(name.clone()),
    }
}

fn entry_info(
    id: crate::ids::EntryId,
    stable: crate::ids::StableEntryId,
    definition: &ir::Definition,
) -> crate::checked::EntryInfo {
    use crate::checked::{
        ParameterSummary, ParameterSummaryKind, ResultSummary, ResultSummaryKind, TensorAccess,
    };
    fn flatten_parameter(
        source: u32,
        name: &str,
        ownership: ParamOwnership,
        ty: &ValueType,
        path: &mut Vec<u32>,
        output: &mut Vec<ParameterSummary>,
    ) {
        if let ValueType::Tuple(items) = ty {
            for (ordinal, item) in items.iter().enumerate() {
                path.push(u32::try_from(ordinal).expect("tuple has more than u32::MAX elements"));
                flatten_parameter(source, name, ownership, item, path, output);
                path.pop();
            }
            return;
        }
        let kind = match ty {
            ValueType::Tensor(tensor) => ParameterSummaryKind::Tensor {
                access: match ownership {
                    ParamOwnership::Owned | ParamOwnership::Value => TensorAccess::Owned,
                    ParamOwnership::Shared => TensorAccess::Shared,
                    ParamOwnership::Exclusive => TensorAccess::Mutable,
                },
                rank: u32::try_from(tensor.rank()).expect("tensor rank exceeds u32::MAX"),
                element: element_summary(&tensor.elem),
            },
            ValueType::Scalar(dtype) => ParameterSummaryKind::Scalar(*dtype),
            ValueType::Index { .. } => ParameterSummaryKind::Index,
            ValueType::Range { .. } => ParameterSummaryKind::Range,
            ValueType::Void => return,
            ValueType::Opaque { .. } => {
                panic!("backend-opaque value escaped a portable entry signature")
            }
            ValueType::Tuple(_) => unreachable!(),
        };
        output.push(ParameterSummary {
            source,
            path: path.clone(),
            name: name.to_owned(),
            kind,
        });
    }
    let mut parameters = Vec::new();
    for (ordinal, parameter) in definition.params.iter().enumerate() {
        flatten_parameter(
            u32::try_from(ordinal).expect("parameter count exceeds u32::MAX"),
            &parameter.name,
            parameter.ownership,
            &parameter.ty,
            &mut Vec::new(),
            &mut parameters,
        );
    }
    fn flatten(ty: &ValueType, path: &mut Vec<u32>, output: &mut Vec<ResultSummary>) {
        match ty {
            ValueType::Tuple(items) => {
                for (ordinal, item) in items.iter().enumerate() {
                    path.push(
                        u32::try_from(ordinal).expect("tuple has more than u32::MAX elements"),
                    );
                    flatten(item, path, output);
                    path.pop();
                }
            }
            ValueType::Tensor(tensor) => output.push(ResultSummary {
                path: path.clone(),
                kind: ResultSummaryKind::Tensor {
                    rank: u32::try_from(tensor.rank()).expect("tensor rank exceeds u32::MAX"),
                    element: element_summary(&tensor.elem),
                },
            }),
            ValueType::Scalar(dtype) => output.push(ResultSummary {
                path: path.clone(),
                kind: ResultSummaryKind::Scalar(*dtype),
            }),
            ValueType::Index { .. } => output.push(ResultSummary {
                path: path.clone(),
                kind: ResultSummaryKind::Index,
            }),
            ValueType::Range { .. } => output.push(ResultSummary {
                path: path.clone(),
                kind: ResultSummaryKind::Range,
            }),
            ValueType::Void => {}
            ValueType::Opaque { .. } => {
                panic!("backend-opaque result escaped an exported portable entry")
            }
        }
    }
    let mut results = Vec::new();
    flatten(&definition.result, &mut Vec::new(), &mut results);
    EntryInfo {
        id,
        stable,
        name: definition.name.clone(),
        element_parameters: definition.elem_params.clone(),
        parameters,
        results,
    }
}
