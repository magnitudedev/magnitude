//! The checker: resolution, contract families and target capabilities, typing
//! of every value kind through the intrinsic registry, ownership/moves,
//! borrow rules, loop carry and disjointness, initialization coverage, and
//! `where` predicates. Emits the checked representation directly.
//! Entry point is `program::compile`.
//!
//! Symbol naming in emitted `Sym`s: the atom `Param("{name}#{LocalId}")` is the
//! runtime value of body local `LocalId` (index parameters, loop binders);
//! `Param("@dyn#n")` is the realized length of a runtime-bounded range view.
//! Every other `Param` is a shape parameter of the definition.

mod call;
mod expr;
pub(crate) mod resolve;
mod stmt;

use crate::intrinsics::{CapabilityId, PrimitiveId};
use crate::sir::IntrinsicUse;
use crate::sir::{
    CheckedBlock, CheckedBody, CheckedExpr, CheckedIndex, CheckedLocal, ContractFamily, DefId,
    DefKind, LocalId, ParamOwnership, Predicate,
};
use crate::span::{Diagnostic, Span};
use crate::sym::{Atom, Facts, Prover, Sym};
use crate::syntax::ast;
use crate::types::{Elem, ExtentExpr, TensorType, ValueType};
use resolve::{Declared, Located, Resolved, Sig};
use std::collections::{BTreeSet, HashMap, HashSet};

/// The coefficient of `atom` in `s` when `s` is `c * atom + rest` with neither
/// `c` nor `rest` mentioning `atom` (zero when `s` is free of `atom`); `None`
/// when `s` is nonlinear in `atom` or mentions it inside a quotient or
/// remainder. `c` may be symbolic (`W * heads` has coefficient `W`).
fn linear_coefficient(s: &Sym, atom: &Atom) -> Option<Sym> {
    let mut coefficient = Sym::constant(0);
    let mut rest = Sym::constant(0);
    for (monomial, k) in s.monomials() {
        let mut term = Sym::constant(k);
        let mut power = 0u32;
        for (factor, multiplicity) in monomial {
            if factor == atom {
                power = *multiplicity;
                continue;
            }
            for _ in 0..*multiplicity {
                term = term.mul(&Sym::atom(factor.clone()));
            }
        }
        match power {
            0 => rest = rest.add(&term),
            1 => coefficient = coefficient.add(&term),
            _ => return None,
        }
    }
    if mentions_atom(&coefficient, atom) || mentions_atom(&rest, atom) {
        None
    } else {
        Some(coefficient)
    }
}

/// Whether the binders in `radix` (magnitude of coefficient, range width) can
/// be ordered so that each magnitude exceeds `reach`, the largest value the
/// binders before it can contribute: then `sum(c_k * v_k)` is injective over
/// the box of ranges, as digits of a mixed radix are. The last binder needs
/// no width. Backtracks over orders; `radix` is at most a few binders.
fn mixed_radix(
    radix: &[(Sym, Option<Sym>)],
    proves: &dyn Fn(&Sym) -> bool,
    used: &mut [bool],
    reach: Sym,
    one: &Sym,
) -> bool {
    let remaining = used.iter().filter(|u| !**u).count();
    if remaining == 0 {
        return true;
    }
    for i in 0..radix.len() {
        if used[i] {
            continue;
        }
        let (magnitude, width) = &radix[i];
        if !proves(&magnitude.sub(one).sub(&reach)) {
            continue;
        }
        used[i] = true;
        let ok = if remaining == 1 {
            true
        } else {
            match width {
                Some(width) => {
                    mixed_radix(radix, proves, used, reach.add(&width.mul(magnitude)), one)
                }
                None => false,
            }
        };
        if ok {
            return true;
        }
        used[i] = false;
    }
    false
}

/// Whether `s` mentions `target` anywhere, including inside quotient and
/// remainder atoms (`i / 2` mentions `i` without being linear in it).
fn mentions_atom(s: &Sym, target: &Atom) -> bool {
    s.atoms().iter().any(|atom| {
        atom == target
            || match atom {
                Atom::Quot(n, d) | Atom::Rem(n, d) => {
                    mentions_atom(n, target) || mentions_atom(d, target)
                }
                Atom::Param(_) => false,
            }
    })
}

/// The atom denoting the runtime value of body local `id`.
pub fn var_atom(name: &str, id: LocalId) -> Atom {
    Atom::Param(format!("{name}#{id}"))
}

/// The body local an atom name denotes, if it is a variable atom.
pub fn atom_var(name: &str) -> Option<LocalId> {
    if name.starts_with('@') {
        return None;
    }
    name.rsplit_once('#').and_then(|(_, id)| id.parse().ok())
}

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
    pub sig: &'a Sig,
    pub kind: DefKind,
    /// The target whose forms this body may name: a backend-specific function
    /// or lowering target.
    pub target: Option<String>,
    pub requires: Vec<(CapabilityId, Span)>,
    pub used_capabilities: BTreeSet<CapabilityId>,
    pub intrinsic_uses: Vec<IntrinsicUse>,
    pub locals: Vec<CheckedLocal>,
    pub kinds: Vec<LocalKind>,
    pub scopes: Vec<HashMap<String, LocalId>>,
    pub facts: Facts,
    pub atoms: HashMap<LocalId, Atom>,
    pub scalar_symbols: HashMap<LocalId, Sym>,
    pub unassigned: HashSet<LocalId>,
    /// Tiles the current complete-traversal loop assigns by its first write at
    /// the loop's own coordinates.
    pub pending_full_assign: Vec<(LocalId, Vec<LocalId>)>,
    /// Complete logical `0..axis` traversals may collectively initialize
    /// uninitialized storage.
    pub init_loop_depth: usize,
    /// Runtime-bounded range views: (start, end, parent extent, realized-length atom).
    pub dyn_views: Vec<(Option<CheckedExpr>, Option<CheckedExpr>, Sym, Atom)>,
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
}

impl<'a> Checker<'a> {
    fn new(env: &'a Env<'a>, def: usize) -> Checker<'a> {
        let declared: &'a Declared<'a> = &env.resolved.declared[def];
        let target = declared.kind.target().map(str::to_owned);
        let mut c = Checker {
            env,
            def,
            sig: &declared.sig,
            kind: declared.kind.clone(),
            target,
            requires: declared.requires.clone(),
            used_capabilities: BTreeSet::new(),
            intrinsic_uses: Vec::new(),
            locals: Vec::new(),
            kinds: Vec::new(),
            scopes: vec![HashMap::new()],
            facts: Facts::new(),
            atoms: HashMap::new(),
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
        };
        // Shape parameters are positive extents unless a `where` admits zero.
        for p in &c.sig.shape_params {
            let admits_zero = c
                .sig
                .predicates
                .iter()
                .any(|q| matches!(q, Predicate::NonNegative(e) if *e == Sym::param(p)));
            c.facts.set_range_lower(
                Atom::Param(p.clone()),
                Sym::constant(if admits_zero { 0 } else { 1 }),
            );
        }
        for predicate in &c.sig.predicates {
            match predicate {
                Predicate::NonNegative(e) => c.assume_nonneg(e),
                Predicate::Zero(e) => c.assume_zero(e),
                Predicate::NonZero(_) => {}
            }
        }
        for (i, p) in c.sig.params.iter().enumerate() {
            let id = c.declare(
                &p.name,
                p.ty.clone(),
                p.span,
                LocalKind::Param(i),
                p.ownership == ParamOwnership::Exclusive,
            );
            if let ValueType::Index { bound } = &p.ty {
                let bound = bound.sym().cloned().unwrap_or_else(|| Sym::constant(1));
                let atom = var_atom(&p.name, id);
                c.facts
                    .set_range(atom.clone(), Sym::constant(0), bound.sub(&Sym::constant(1)));
                c.atoms.insert(id, atom);
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
        self.used_capabilities.insert(capability.clone());
        if !self
            .requires
            .iter()
            .any(|(declared, _)| declared == capability)
        {
            self.error(
                span,
                format!(
                    "{use_site} requires capability `{}`; add `requires {}` to this declaration",
                    capability.path(),
                    capability.path()
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
        let id = self.locals.len();
        self.locals.push(CheckedLocal {
            name: name.to_string(),
            ty,
            mutable,
            span,
        });
        self.kinds.push(kind);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), id);
        }
        id
    }

    pub fn fresh_atom(&mut self, base: &str) -> Atom {
        self.counter += 1;
        Atom::Param(format!("@{base}#{}", self.counter))
    }

    pub fn prover(&self) -> Prover<'_> {
        Prover::new(&self.facts)
    }

    /// Assume `e >= 0` as bounds on every atom with a unit coefficient.
    pub fn assume_nonneg(&mut self, e: &Sym) {
        for atom in e.atoms() {
            match e.linear_in(&atom) {
                Some((1, rest)) => self.facts.add_lower(atom, rest.neg()),
                Some((-1, rest)) => self.facts.add_upper(atom, rest),
                _ => {}
            }
        }
    }

    /// Assume `e == 0`: a zero fact, and both bounds on every atom with a unit coefficient.
    pub fn assume_zero(&mut self, e: &Sym) {
        self.facts.assume_zero(e.clone());
        self.assume_nonneg(e);
        self.assume_nonneg(&e.neg());
    }

    fn is_shape_param(&self, name: &str) -> bool {
        self.sig.shape_params.iter().any(|p| p == name)
    }

    /// Record that the value of these shape parameters is observed as a number.
    pub fn numeric_use(&mut self, sym: &Sym) {
        for p in sym.params() {
            if self.is_shape_param(&p) {
                self.summary.numeric.insert(p);
            }
        }
    }

    /// Prove `e >= 0`. Shape-arithmetic needs become a diagnostic asking for a `where`.
    pub fn require_nonneg(&mut self, e: &Sym, span: Span, what: &str) {
        if self.prover().nonneg(e) {
            return;
        }
        let shape_only = |s: &Sym, c: &Checker| s.params().iter().all(|p| c.is_shape_param(p));
        let need = self.prover().interval_over(e, &|a| a.mentions_loop()).lo;
        if shape_only(&need, self) && !need.as_constant().is_some_and(|c| c < 0) {
            self.error(span, format!("{what}: cannot prove `{e} >= 0`; state the shape requirement with `where {need} >= 0`"));
        } else {
            self.error(span, format!("{what}: cannot prove `{e} >= 0`"));
        }
    }

    // ---- types ----

    pub fn same_extent(&self, a: &ExtentExpr, b: &ExtentExpr) -> bool {
        match (a, b) {
            (ExtentExpr::Static(x), ExtentExpr::Static(y)) => x == y,
            (ExtentExpr::Sym(x), ExtentExpr::Sym(y)) => self.prover().zero(&x.sub(y)),
            _ => false,
        }
    }

    pub fn same_axes(&self, a: &TensorType, b: &TensorType) -> bool {
        a.rank() == b.rank()
            && a.axes
                .iter()
                .zip(&b.axes)
                .all(|(x, y)| self.same_extent(x, y))
    }

    pub fn same_ty(&self, a: &ValueType, b: &ValueType) -> bool {
        match (a, b) {
            (ValueType::Tensor(x), ValueType::Tensor(y)) => {
                x.elem == y.elem && self.same_axes(x, y)
            }
            (ValueType::Index { bound: x }, ValueType::Index { bound: y })
            | (ValueType::Range { bound: x }, ValueType::Range { bound: y }) => {
                self.same_extent(x, y)
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
        if namespace.is_some_and(|ns| ns != target) {
            self.error(
                span,
                format!(
                    "{what} belongs to target `{}` but this body is for `{target}`",
                    namespace.unwrap_or_default()
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
                PrimitiveId::SliceView { .. }
                | PrimitiveId::Transpose
                | PrimitiveId::Reshape
                | PrimitiveId::PackedRead(_) => operands.first().and_then(|b| self.root_var(b)),
                _ => None,
            },
            _ => None,
        }
    }

    /// How the storage of an expression is reached.
    pub fn class_of(&self, e: &CheckedExpr) -> ValueClass {
        match &e.kind {
            CheckedExprKind::Literal(_) => ValueClass::Scalar,
            CheckedExprKind::Local(v) => {
                if self.view_roots.contains_key(v) {
                    return ValueClass::Borrowed;
                }
                match self.kinds[*v] {
                    LocalKind::Param(i) => match self.sig.params[i].ownership {
                        ParamOwnership::Owned => {
                            if matches!(self.locals[*v].ty, ValueType::Tensor(_)) {
                                ValueClass::Owned
                            } else {
                                ValueClass::Scalar
                            }
                        }
                        ParamOwnership::Shared | ParamOwnership::Exclusive => {
                            if matches!(self.locals[*v].ty, ValueType::Tensor(_)) {
                                ValueClass::Borrowed
                            } else {
                                ValueClass::Scalar
                            }
                        }
                        ParamOwnership::Value => ValueClass::Scalar,
                    },
                    LocalKind::State => {
                        if matches!(self.locals[*v].ty, ValueType::Tensor(_)) {
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
                PrimitiveId::ElementRead { .. } => ValueClass::Scalar,
                PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                    ValueClass::Borrowed
                }
                PrimitiveId::Load
                | PrimitiveId::Decode
                | PrimitiveId::PackedRead(_)
                | PrimitiveId::Unary(_)
                | PrimitiveId::Binary(_)
                | PrimitiveId::Cast(_)
                | PrimitiveId::Math(_)
                | PrimitiveId::Select
                | PrimitiveId::Reduce { .. }
                | PrimitiveId::Extent { .. }
                | PrimitiveId::ValidExtent { .. } => {
                    if e.ty.scalar_dtype().is_some() {
                        ValueClass::Scalar
                    } else {
                        ValueClass::Computed
                    }
                }
                PrimitiveId::TensorAlloc { .. }
                | PrimitiveId::Fill { .. }
                | PrimitiveId::Materialize
                | PrimitiveId::Clone => ValueClass::Owned,
                PrimitiveId::TuplePack => ValueClass::Computed,
                PrimitiveId::TupleGet(_)
                | PrimitiveId::RangeMake
                | PrimitiveId::RangeStart
                | PrimitiveId::RangeEnd => ValueClass::Scalar,
                PrimitiveId::ElementWrite { .. }
                | PrimitiveId::CopyInto
                | PrimitiveId::Atomic { .. } => ValueClass::Scalar,
            },
            CheckedExprKind::Capability { .. } => ValueClass::Scalar,
            CheckedExprKind::Call { .. } => match &e.ty {
                ValueType::Tensor(_) => ValueClass::Owned,
                ValueType::Tuple(_) => ValueClass::Computed,
                _ => ValueClass::Scalar,
            },
        }
    }

    /// Whether writes may target the storage rooted at `id`.
    pub fn writable_root(&self, id: LocalId) -> bool {
        match self.kinds[id] {
            LocalKind::Param(i) => self.sig.params[i].ownership == ParamOwnership::Exclusive,
            LocalKind::State => true,
            _ => false,
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
        enum Axis {
            Opaque,
            /// Coefficient of every binder (zero when absent).
            Point(Vec<Sym>),
            Slice {
                binder: usize,
                coefficient: Sym,
                length: Sym,
            },
        }
        let atoms: Vec<Option<Atom>> = binders.iter().map(|b| self.atoms.get(b).cloned()).collect();
        let mut axes = Vec::with_capacity(indices.len());
        for index in indices {
            axes.push(match index {
                CheckedIndex::Point(p) => match &p.sym {
                    Some(s) => {
                        let mut coefficients = Vec::with_capacity(atoms.len());
                        let mut opaque = false;
                        for atom in &atoms {
                            let Some(atom) = atom else {
                                coefficients.push(Sym::constant(0));
                                continue;
                            };
                            match linear_coefficient(s, atom) {
                                Some(c) => coefficients.push(c),
                                None => {
                                    opaque = true;
                                    break;
                                }
                            }
                        }
                        if opaque {
                            Axis::Opaque
                        } else {
                            Axis::Point(coefficients)
                        }
                    }
                    None => Axis::Opaque,
                },
                CheckedIndex::Range {
                    start: Some(start),
                    end: Some(end),
                } => match (&start.sym, &end.sym) {
                    (Some(start), Some(end)) => {
                        let length = end.sub(start);
                        let mut found = None;
                        let mut opaque = false;
                        for (k, atom) in atoms.iter().enumerate() {
                            let Some(atom) = atom else { continue };
                            if mentions_atom(&length, atom) {
                                opaque = true;
                                break;
                            }
                            match linear_coefficient(start, atom) {
                                Some(c) if c.is_zero() => {}
                                Some(c) if found.is_none() => found = Some((k, c)),
                                _ => {
                                    opaque = true;
                                    break;
                                }
                            }
                        }
                        match found {
                            Some((binder, coefficient)) if !opaque => Axis::Slice {
                                binder,
                                coefficient,
                                length,
                            },
                            _ => Axis::Opaque,
                        }
                    }
                    _ => Axis::Opaque,
                },
                CheckedIndex::Range { .. } => Axis::Opaque,
            });
        }
        let prover = self.prover();
        let one = Sym::constant(1);
        // The write executes inside every capturing loop, so each of their
        // ranges is nonempty: `upper - lower >= 0` is a fact the goal may spend
        // (`y[i * C + j]` needs `C >= 1`, which `j < C` supplies).
        let nonempty: Vec<Sym> = atoms
            .iter()
            .flatten()
            .filter_map(|atom| {
                let upper = self.facts.upper_of(atom)?;
                Some(upper.sub(&self.facts.lower_of(atom)))
            })
            .collect();
        let proves = |goal: &Sym| -> bool {
            prover.nonneg(goal) || nonempty.iter().any(|fact| prover.nonneg(&goal.sub(fact)))
        };
        // `|c|` when the prover knows the sign of `c` and `|c| >= 1`.
        let magnitude = |c: &Sym| -> Option<Sym> {
            if proves(&c.sub(&one)) {
                Some(c.clone())
            } else if proves(&c.neg().sub(&one)) {
                Some(c.neg())
            } else {
                None
            }
        };
        let mut proven = vec![false; binders.len()];
        loop {
            let mut progress = false;
            for axis in &axes {
                match axis {
                    Axis::Opaque => {}
                    Axis::Slice {
                        binder,
                        coefficient,
                        length,
                    } => {
                        // Visits `v != v'` write `[c*v + d, c*v + d + len)`, which are
                        // disjoint when `len <= c` (the checker already has `len >= 0`).
                        if !proven[*binder] && proves(&coefficient.sub(length)) {
                            proven[*binder] = true;
                            progress = true;
                        }
                    }
                    Axis::Point(coefficients) => {
                        let mut pending: Vec<(usize, Sym)> = Vec::new();
                        let mut usable = true;
                        for (k, c) in coefficients.iter().enumerate() {
                            if proven[k] || c.is_zero() {
                                continue;
                            }
                            match magnitude(c) {
                                Some(m) => pending.push((k, m)),
                                None => {
                                    usable = false;
                                    break;
                                }
                            }
                        }
                        if !usable || pending.is_empty() {
                            continue;
                        }
                        // Every proven binder on this axis is fixed between the
                        // two visits. The remaining ones are proven together when
                        // they can be ordered so that each magnitude exceeds the
                        // reach of the ones before it over their ranges (a mixed
                        // radix).
                        let radix: Vec<(Sym, Option<Sym>)> = pending
                            .iter()
                            .map(|(k, m)| {
                                let atom = atoms[*k]
                                    .as_ref()
                                    .expect("a binder with a coefficient has an atom");
                                let width = self
                                    .facts
                                    .upper_of(atom)
                                    .map(|upper| upper.sub(&self.facts.lower_of(atom)));
                                (m.clone(), width)
                            })
                            .collect();
                        let mut used = vec![false; radix.len()];
                        if mixed_radix(&radix, &proves, &mut used, Sym::constant(0), &one) {
                            for (k, _) in &pending {
                                proven[*k] = true;
                            }
                            progress = true;
                        }
                    }
                }
            }
            if !progress {
                break;
            }
        }
        match proven.iter().position(|p| !p) {
            Some(k) => Err(binders[k]),
            None => Ok(()),
        }
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
        let binding_name = self.locals[binding].name.clone();
        match &self.kinds[binding] {
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
                    self.locals[root].name
                ),
            );
            return None;
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
                    self.locals[root].name
                ),
            );
            return None;
        }
        let capturing: Vec<LocalId> = self
            .logical_parallel
            .iter()
            .filter(|(floor, _)| root < *floor)
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
                let binder_name = self.locals[binder].name.clone();
                let root_name = self.locals[root].name.clone();
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
            if root < ctx.floor {
                ctx.writes.push(root);
                if whole {
                    ctx.whole_writes.push(root);
                }
            }
        }
        self.mutated.push(root);
        self.scalar_symbols.remove(&root);
        let tensor_effect = matches!(self.locals[root].ty, ValueType::Tensor(_));
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
    ) -> (CheckedBody, Summary, Vec<IntrinsicUse>, Vec<Diagnostic>) {
        let well_formed = self.diagnostics.is_empty();
        if well_formed
            && self.sig.result != ValueType::Void
            && !matches!(root.terminator, crate::sir::BlockTerminator::Return(_))
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
                        "capability `{}` is required but not used directly or through a backend-specific helper",
                        capability.path()
                    ),
                );
            }
        }
        let body = CheckedBody {
            locals: self.locals,
            root,
        };
        (body, self.summary, self.intrinsic_uses, self.diagnostics)
    }
}

use crate::sir::CheckedExprKind;

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
    intrinsic_uses: Vec<IntrinsicUse>,
    diagnostics: Vec<Diagnostic>,
}

fn check_definition(env: &Env, def: usize) -> CheckedOutcome {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    let root = c.block(declared.body);
    let facts = c.facts.clone();
    let (body, mut summary, intrinsic_uses, diagnostics) = c.finish(root, declared.name_span);
    for (parameter, declared_param) in declared.sig.params.iter().enumerate() {
        if declared_param.ownership == ParamOwnership::Exclusive {
            if let ValueType::Tensor(shaped) = &declared_param.ty {
                if definitely_initializes(&body, parameter, shaped, &facts) {
                    summary.full_init.insert(parameter);
                }
            }
        }
    }
    CheckedOutcome {
        body,
        summary,
        intrinsic_uses,
        diagnostics,
    }
}

/// Whether a body definitely initializes every element of `parameter`.
/// Parameters are the first locals of a checked body, in parameter order.
fn definitely_initializes(
    body: &CheckedBody,
    parameter: usize,
    shaped: &TensorType,
    facts: &Facts,
) -> bool {
    writes_cover(
        &body.locals,
        &body.root,
        parameter,
        &shaped.axes,
        &[],
        facts,
    )
}

/// Whether the block writes every element of `local` on every path, under the
/// complete traversals in `loops` (binder, bound).
pub(crate) fn writes_cover(
    locals: &[CheckedLocal],
    block: &CheckedBlock,
    local: LocalId,
    extents: &[ExtentExpr],
    loops: &[(LocalId, Sym)],
    facts: &Facts,
) -> bool {
    fn point_covers(
        locals: &[CheckedLocal],
        point: &CheckedExpr,
        extent: &Sym,
        loops: &[(LocalId, Sym)],
    ) -> bool {
        match &point.kind {
            CheckedExprKind::Local(var) => loops
                .iter()
                .any(|(loop_var, bound)| loop_var == var && bound == extent),
            _ => {
                if extent.as_constant() == Some(1) && point.sym.as_ref().is_some_and(Sym::is_zero) {
                    return true;
                }
                // Nested full loops form a bijective mixed-radix enumeration of a
                // flattened axis: `h * W + i`, and its higher-rank generalization.
                let mut total = Sym::constant(1);
                let mut linear = Sym::constant(0);
                for (var, bound) in loops {
                    let atom = var_atom(&locals[*var].name, *var);
                    linear = linear.mul(bound).add(&Sym::atom(atom));
                    total = total.mul(bound);
                }
                &total == extent && point.sym.as_ref() == Some(&linear)
            }
        }
    }
    fn index_covers(
        locals: &[CheckedLocal],
        index: &crate::sir::CheckedIndex,
        extent: &Sym,
        loops: &[(LocalId, Sym)],
        facts: &Facts,
    ) -> bool {
        match index {
            crate::sir::CheckedIndex::Point(point) => point_covers(locals, point, extent, loops),
            crate::sir::CheckedIndex::Range { start, end } => {
                let full = start
                    .as_ref()
                    .is_none_or(|start| start.sym.as_ref().is_some_and(Sym::is_zero))
                    && end
                        .as_ref()
                        .is_none_or(|end| end.sym.as_ref() == Some(extent));
                full || loops.iter().any(|(var, partitions)| {
                    let width = extent.quot(partitions);
                    if !Prover::new(facts).zero(&width.mul(partitions).sub(extent)) {
                        return false;
                    }
                    let coordinate = Sym::atom(var_atom(&locals[*var].name, *var));
                    let expected_start = coordinate.mul(&width);
                    let expected_end = coordinate.add(&Sym::constant(1)).mul(&width);
                    start.as_ref().and_then(|value| value.sym.as_ref()) == Some(&expected_start)
                        && end.as_ref().and_then(|value| value.sym.as_ref()) == Some(&expected_end)
                })
            }
        }
    }
    fn place_covers(
        locals: &[CheckedLocal],
        place: &crate::sir::CheckedPlace,
        local: LocalId,
        extents: &[ExtentExpr],
        loops: &[(LocalId, Sym)],
        facts: &Facts,
    ) -> bool {
        let (root, indices) = match place {
            crate::sir::CheckedPlace::Local { root } => {
                return *root == local;
            }
            crate::sir::CheckedPlace::Element { root, indices } => (root, indices),
            crate::sir::CheckedPlace::Tuple(_) => return false,
        };
        if *root != local {
            return false;
        }
        if indices.len() > extents.len() {
            return false;
        }
        indices.iter().zip(extents).all(|(index, extent)| {
            let ExtentExpr::Sym(extent) = extent else {
                let ExtentExpr::Static(n) = extent else {
                    return false;
                };
                return index_covers(locals, index, &Sym::constant(*n as i64), loops, facts);
            };
            index_covers(locals, index, extent, loops, facts)
        })
        // Omitted trailing indices denote the complete remaining tensor slice.
    }
    fn block_writes(
        locals: &[CheckedLocal],
        block: &CheckedBlock,
        local: LocalId,
        extents: &[ExtentExpr],
        loops: &[(LocalId, Sym)],
        facts: &Facts,
    ) -> bool {
        // Every path must write: an `if` covers only when both arms cover.
        for statement in &block.statements {
            match statement {
                crate::sir::CheckedStmt::Assign { place, .. } => {
                    if place_covers(locals, place, local, extents, loops, facts) {
                        return true;
                    }
                }
                crate::sir::CheckedStmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if block_writes(locals, then_body, local, extents, loops, facts)
                        && block_writes(locals, else_body, local, extents, loops, facts)
                    {
                        return true;
                    }
                }
                crate::sir::CheckedStmt::Loop {
                    binder,
                    range,
                    body,
                    ..
                } => {
                    let (Some(start), Some(end)) = (range.start.sym.clone(), range.end.sym.clone())
                    else {
                        continue;
                    };
                    if !start.is_zero() {
                        continue;
                    }
                    let mut nested = loops.to_vec();
                    nested.push((*binder, end));
                    if block_writes(locals, body, local, extents, &nested, facts) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }
    block_writes(locals, block, local, extents, loops, facts)
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
fn reject_cycles(definitions: &[crate::sir::Definition], diagnostics: &mut Vec<Located>) {
    let count = definitions.len();
    // 0 = unvisited, 1 = on stack, 2 = done.
    let mut state = vec![0u8; count];
    let mut stack: Vec<usize> = Vec::new();
    fn visit(
        d: usize,
        definitions: &[crate::sir::Definition],
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
            let c = callee.0 as usize;
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
    diagnostics: &mut Vec<Located>,
) -> (Vec<crate::sir::Definition>, Vec<ContractFamily>) {
    let resolved = resolve::resolve(files, diagnostics);
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
        let checked = check_definition(&env, def);
        diagnostics.extend(checked.diagnostics.into_iter().map(|diagnostic| Located {
            file: declared.file,
            diagnostic,
        }));
        // Parameters are the first locals of a checked body.
        let params = declared
            .sig
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| crate::sir::Param {
                name: p.name.clone(),
                mode: p.mode,
                ownership: p.ownership,
                ty: p.ty.clone(),
                local: i,
            })
            .collect();
        definitions.push(crate::sir::Definition {
            id: DefId(def as u32),
            name: declared.sig.name.clone(),
            kind: declared.kind.clone(),
            requires: declared
                .requires
                .iter()
                .map(|(capability, _)| capability.clone())
                .collect(),
            intrinsic_uses: checked.intrinsic_uses,
            family: declared.family,
            shape_params: declared.sig.shape_params.clone(),
            elem_params: declared.sig.elem_params.clone(),
            elem_bindings: declared.elem_bindings.clone(),
            params,
            aliases: declared.sig.aliases.clone(),
            result: declared.sig.result.clone(),
            predicates: declared.sig.predicates.clone(),
            body: checked.body,
            file: declared.file,
            span: declared.span,
        });
    }
    if diagnostics.is_empty() {
        reject_cycles(&definitions, diagnostics);
    }
    (definitions, resolved.families)
}

#[cfg(test)]
mod tests {
    use crate::program::{compile, SourceFile};
    use crate::sir::{BlockTerminator, CheckedExprKind, CheckedStmt, LoopKind, Program};
    use crate::types::{DType, Elem, ValueType};

    fn check(sources: &[(&str, &str)]) -> Result<Program, String> {
        let files: Vec<SourceFile> = sources
            .iter()
            .map(|(path, text)| SourceFile {
                path: path.to_string(),
                text: text.to_string(),
            })
            .collect();
        compile(&files).map_err(|d| d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n"))
    }

    fn rejected(source: &str, rule: &str) {
        match check(&[("case.seismic", source)]) {
            Ok(_) => panic!("accepted, expected a diagnostic containing `{rule}`:\n{source}"),
            Err(rendered) => assert!(rendered.contains(rule), "expected `{rule}` in:\n{rendered}"),
        }
    }

    #[test]
    fn logical_call_sources_check() {
        let program = check(&[(
            "logical.seismic",
            "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
        let linear = program.family("linear").expect("linear is defined");
        let body = &program.definition(linear.bodies[0]).body;
        // Logical borrowed operands and the owned result bind M and N structurally.
        let call = &body.calls()[0];
        assert!(!call.bindings.is_empty());
        assert_eq!(call.bindings[0].shape_args.len(), 2);
        assert_eq!(call.bindings[0].arg_order, vec![0, 1, 2]);
    }

    #[test]
    fn returned_tuple_results_check() {
        check(&[(
            "tuple.seismic",
            "fn split[N](left: tensor[N] f32, right: tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):\n    return left, right\n\nfn swap[N](left: tensor[N] f32, right: tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):\n    let a, b = split(left, right)\n    return b, a\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn slice_width_query_is_a_logical_value() {
        check(&[("case.seismic", "fn f[M, N](x: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let v = x[:, 0]\n    let w = extent(v, 0)\n    return result\n")]).unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn borrowed_slice_cannot_outlive_exclusive_access() {
        rejected("fn f[N](x: &mut tensor[N] f32):\n    let slice = x[:]\n    x[0] = 1.0\n    let value = slice[0]\n", "tensor borrow is live");
    }

    #[test]
    fn mask_as_if_condition_rejected() {
        rejected("fn f[N](x: &tensor[N] f32, result: tensor[N] f32) -> tensor[N] f32:\n    let mask = f32(x) > 0.0\n    if mask:\n        return result\n    else:\n        return result\n", "a mask is a `bool` tile");
    }

    #[test]
    fn return_on_one_path_rejected() {
        rejected("fn f[N](result: tensor[N] f32, flag: bool) -> tensor[N] f32:\n    if flag:\n        return result\n", "not every path ends in `return`");
    }

    #[test]
    fn parallel_mutating_enclosing_state_rejected() {
        rejected("fn f[N](x: &tensor[N] f32) -> f32:\n    let mut total = f32(0.0)\n    parallel for i in 0..N:\n        total = total + x[i]\n    return total\n", "parallel for");
    }

    #[test]
    fn let_mut_is_required_for_assignment() {
        rejected(
            "fn f() -> f32:\n    let value = f32(1.0)\n    value = f32(2.0)\n    return value\n",
            "not mutable state",
        );
        check(&[("case.seismic", "fn f() -> f32:\n    let mut value = f32(1.0)\n    value = f32(2.0)\n    return value\n")])
            .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn let_mut_is_required_for_writable_borrows() {
        rejected(
            "fn f[N](y: tensor[N] f32) -> tensor[N] f32:\n    let alias = y\n    alias = clone(alias)\n    return alias\n",
            "not mutable state",
        );
        check(&[(
            "case.seismic",
            "fn f[N](y: tensor[N] f32) -> tensor[N] f32:\n    let mut alias = y\n    alias = clone(alias)\n    return alias\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
        rejected(
            "fn f[N](x: &tensor[N] f32):\n    x[0] = 0.0\n",
            "is not writable storage",
        );
    }

    #[test]
    fn backend_specific_call_capabilities_are_checked() {
        check(&[(
            "case.seismic",
            "fn helper(x: f32) -> f32 for metal:\n    return x + f32(1.0)\n\nfn caller(x: f32) -> f32 for metal:\n    return helper(x)\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
        rejected(
            "fn helper(x: f32) -> f32 for metal:\n    return x\n\nfn caller(x: f32) -> f32:\n    return helper(x)\n",
            "no implementation callable from portable code",
        );
        rejected(
            "fn helper(x: f32) -> f32 for cuda:\n    return x\n\nfn caller(x: f32) -> f32 for metal:\n    return helper(x)\n",
            "no implementation callable from metal code",
        );
        rejected(
            "fn helper(x: f32) -> f32:\n    return x\n\nfn helper(x: f32) -> f32 for metal:\n    return x + f32(1.0)\n",
            "separate helper, not an implementation of a portable family",
        );
    }

    #[test]
    fn implementation_contracts_preserve_shapes_elements_and_ownership() {
        rejected(
            "fn f[M, N](x: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return result\n\nlower f[M, N](x: &tensor[M, M] f32, result: tensor[M, N] f32) -> tensor[M, N] f32 for cpu:\n    return result\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn f[M](x: &tensor[M] f32, result: tensor[M] f32) -> tensor[M] f32:\n    return result\n\nlower f[M](x: tensor[M] f32, result: tensor[M] f32) -> tensor[M] f32 for cpu:\n    return result\n",
            "no definition of `f` has this parameter structure",
        );
        rejected(
            "fn f[M, N](x: tensor[M, N] f32) -> tensor[M, N] f32:\n    return x\n\nlower f[M, N](x: tensor[M, N] f32) -> tensor[N, M] f32 for cpu:\n    return x\n",
            "equivalent results",
        );

        check(&[(
            "case.seismic",
            "fn f[M](x: tensor[M] T) -> tensor[M] T:\n    return x\n\nlower f[M](x: tensor[M] bf16) -> tensor[M] bf16 for cpu:\n    return x\n",
        )])
        .unwrap_or_else(|error| panic!("concrete element specialization was rejected:\n{error}"));

        rejected(
            "fn split[M](x: tensor[M] T, w: &tensor[M] T) -> tensor[M] T:\n    return x\n\nlower split[M](x: tensor[M] U, w: &tensor[M] V) -> tensor[M] U for cpu:\n    return x\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn mixed[M](x: tensor[M] T, w: &tensor[M] T) -> tensor[M] T:\n    return x\n\nlower mixed[M](x: tensor[M] bf16, w: &tensor[M] U) -> tensor[M] bf16 for cpu:\n    return x\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn concrete[M](x: tensor[M] bf16) -> tensor[M] bf16:\n    return x\n\nlower concrete[M](x: tensor[M] T) -> tensor[M] T for cpu:\n    return x\n",
            "equivalent parameter shape and element relationships",
        );
        check(&[(
            "case.seismic",
            "fn converge[M](x: tensor[M] T, w: &tensor[M] U) -> tensor[M] T:\n    return x\n\nlower converge[M](x: tensor[M] V, w: &tensor[M] V) -> tensor[M] V for cpu:\n    return x\n",
        )])
        .unwrap_or_else(|error| panic!("independent element parameters could not converge:\n{error}"));
    }

    #[test]
    fn portable_body_cannot_name_target_intrinsics() {
        rejected(
            "fn f(x: f32) -> f32:\n    return metal.subgroup.simd_sum(x)\n",
            "cannot appear in a portable body",
        );
    }

    #[test]
    fn capability_requirements_are_explicit_exact_and_used() {
        rejected(
            "fn f(x: f32) -> f32 for metal:\n    return metal.subgroup.simd_sum(x)\n",
            "add `requires metal.subgroup`",
        );
        rejected(
            "fn f(x: f32) -> f32 for metal requires cuda.subgroup:\n    return x\n",
            "belongs to backend `cuda`",
        );
        rejected(
            "fn f(x: f32) -> f32 for metal requires metal.threads:\n    return x\n",
            "not a known capability namespace",
        );
        rejected(
            "fn f(x: f32) -> f32 for metal requires metal.subgroup:\n    return x\n",
            "required but not used",
        );
        rejected(
            "fn f(x: f32) -> f32 requires metal.subgroup:\n    return x\n",
            "portable functions cannot require",
        );

        let program = check(&[(
            "case.seismic",
            "fn f(x: f32) -> f32 for metal requires metal.subgroup:\n    return metal.subgroup.simd_max(x)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        let definition = &program.definitions[0];
        assert_eq!(definition.requires[0].path(), "metal.subgroup");
        assert_eq!(definition.intrinsic_uses.len(), 1);
        assert_eq!(
            definition.intrinsic_uses[0].id.path(),
            "metal.subgroup.simd_max"
        );
        assert_eq!(
            definition.intrinsic_uses[0].arguments,
            vec![ValueType::Scalar(DType::F32)]
        );
        assert_eq!(
            definition.intrinsic_uses[0].result,
            ValueType::Scalar(DType::F32)
        );
    }

    #[test]
    fn backend_helper_requirements_propagate_to_callers() {
        let leaf = "fn leaf(x: f32) -> f32 for metal requires metal.subgroup:\n    return metal.subgroup.simd_sum(x)\n\n";
        rejected(
            &format!("{leaf}fn caller(x: f32) -> f32 for metal:\n    return leaf(x)\n"),
            "backend-specific helper `leaf` requires capability `metal.subgroup`",
        );
        check(&[(
            "case.seismic",
            &format!(
                "{leaf}fn caller(x: f32) -> f32 for metal requires metal.subgroup:\n    return leaf(x)\n"
            ),
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn logical_matrix_intrinsic_records_typed_use() {
        let program = check(&[(
            "case.seismic",
            "fn mm[M, K, N](a: tensor[M, K] f16, b: tensor[K, N] f16) -> tensor[M, N] f32 for metal requires metal.matrix:\n    return metal.matrix.matmul(a, b, accumulation=f32)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        let used = &program.definitions[0].intrinsic_uses[0];
        assert_eq!(used.id.path(), "metal.matrix.matmul");
        assert_eq!(used.arguments.len(), 2);
        assert!(matches!(used.result, ValueType::Tensor(_)));
        assert_eq!(
            used.result.shaped().map(|shape| &shape.elem),
            Some(&Elem::Dtype(DType::F32))
        );
    }

    #[test]
    fn owned_tensors_move_and_copies_are_explicit() {
        rejected(
            "fn f[N](x: tensor[N] f32) -> tensor[N] f32:\n    let y = x\n    let z = x\n    return y\n",
            "use of moved owned tensor `x`",
        );
        rejected(
            "fn f[N](x: tensor[N] f32, seed: tensor[N] f32) -> tensor[N] f32:\n    let mut state = seed\n    state = x\n    let again = x\n    return state\n",
            "use of moved owned tensor `x`",
        );
        check(&[(
            "case.seismic",
            "fn f[N](x: tensor[N] f32) -> tensor[N] f32:\n    let copy = clone(x)\n    return x\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        check(&[(
            "case.seismic",
            "fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn call_ownership_distinguishes_move_shared_and_exclusive_access() {
        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn caller[N](x: tensor[N] f32):\n    take(x)\n    take(x)\n    return\n",
            "use of moved owned tensor `x`",
        );
        check(&[(
            "case.seismic",
            "fn read[N](x: &tensor[N] f32) -> f32:\n    return f32(x[0])\n\nfn caller[N](x: tensor[N] f32) -> f32:\n    return read(x) + read(x)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn conflict[N](write: &mut tensor[N] f32, read: &tensor[N] f32):\n    return\n\nfn caller[N](x: &mut tensor[N] f32):\n    conflict(x, x)\n",
            "overlapping tensor arguments",
        );
    }

    #[test]
    fn exclusive_call_requires_a_full_initialization_effect() {
        check(&[("case.seismic", "fn fill[N](x: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        x[i] = f32(i)\n\nfn make[N]() -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    fill(result)\n    return result\n")])
            .unwrap_or_else(|error| panic!("{error}"));
        check(&[("case.seismic", "fn fill2[M, N](x: &mut tensor[M, N] f32):\n    parallel for i in 0..M:\n        parallel for j in 0..N:\n            x[i, j] = f32(i + j)\n\nfn wrapper[M, N](x: &mut tensor[M, N] f32):\n    fill2(x)\n\nfn make[M, N]() -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    wrapper(result)\n    return result\n")])
            .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn partial[N](x: &mut tensor[N] f32):\n    x[0] = 1.0\n\nfn make[N]() -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    partial(result)\n    return result\n",
            "does not initialize that exclusive tensor on every path",
        );
        rejected(
            "fn conditional[N](x: &mut tensor[N] f32, yes: bool):\n    if yes:\n        for i in 0..N:\n            x[i] = f32(i)\n\nfn make[N](yes: bool) -> tensor[N] f32:\n    let mut result = tensor[N] f32\n    conditional(result, yes)\n    return result\n",
            "does not initialize that exclusive tensor on every path",
        );
    }

    #[test]
    fn full_initialization_summaries_cross_files_and_cover_logical_slices() {
        check(&[
            (
                "fill.seismic",
                "fn fill_rows[M, N](value: &tensor[N] f32, result: &mut tensor[M, N] f32):\n    parallel for row in 0..M:\n        result[row] = value\n\nfn fill_singleton[N](value: &tensor[N] f32, result: &mut tensor[1, N] f32):\n    parallel for col in 0..N:\n        result[0, col] = value[col]\n\nfn fill_slice[M, N](value: &tensor[M, N] f32, result: &mut tensor[M, N] f32):\n    result[0:M] = value\n",
            ),
            (
                "caller.seismic",
                "fn make_rows[M, N](value: &tensor[N] f32) -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    fill_rows(value, result)\n    return result\n\nfn make_singleton[N](value: &tensor[N] f32) -> tensor[1, N] f32:\n    let mut result = tensor[1, N] f32\n    fill_singleton(value, result)\n    return result\n\nfn make_slice[M, N](value: &tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    fill_slice(value, result)\n    return result\n",
            ),
        ])
        .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn full_initialization_proves_exact_affine_coverage_and_transparent_reshape() {
        check(&[
            (
                "fills.seismic",
                "fn fill_partitioned[G, P](value: &tensor[G] f32, result: &mut tensor[P * G] f32):\n    parallel for part in 0..P:\n        result[part * G:(part + 1) * G] = value\n\nfn fill_flat[H, W](value: f32, result: &mut tensor[H * W] f32):\n    parallel for h in 0..H:\n        parallel for i in 0..W:\n            result[h * W + i] = value\n",
            ),
            (
                "callers.seismic",
                "fn make_partitioned[G, P](value: &tensor[G] f32) -> tensor[P * G] f32:\n    let mut result = tensor[P * G] f32\n    fill_partitioned[P = P](value, result)\n    return result\n\nfn make_reshaped[H, W](value: f32) -> tensor[H, W] f32:\n    let mut result = tensor[H, W] f32\n    fill_flat[H = H, W = W](value, reshape(result, (H * W,)))\n    return result\n\nfn make_local[M, N](value: &tensor[N] f32) -> tensor[M, N] f32:\n    let mut result = tensor[M, N] f32\n    parallel for row in 0..M:\n        result[row] = value\n    let first = result[0, 0]\n    return result\n",
            ),
        ])
        .unwrap_or_else(|error| panic!("{error}"));

        rejected(
            "fn gap[G, P](value: &tensor[G] f32, result: &mut tensor[P * G] f32):\n    parallel for part in 0..P:\n        result[part * G:(part + 1) * G - 1] = value\n\nfn make[G, P](value: &tensor[G] f32) -> tensor[P * G] f32:\n    let mut result = tensor[P * G] f32\n    gap[P = P](value, result)\n    return result\n",
            "does not initialize that exclusive tensor on every path",
        );
    }

    #[test]
    fn branch_ownership_is_independent_and_joins_on_all_paths() {
        check(&[(
            "case.seismic",
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn caller[N](x: tensor[N] f32, choose: bool):\n    if choose:\n        take(x)\n    else:\n        take(x)\n    return\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));

        check(&[(
            "case.seismic",
            "fn caller[N](x: tensor[N] f32, choose: bool) -> tensor[N] f32:\n    if choose:\n        let a = clone(x)\n    else:\n        let b = clone(x)\n    return x\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));

        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn caller[N](x: tensor[N] f32, choose: bool) -> tensor[N] f32:\n    if choose:\n        take(x)\n    else:\n        let copy = clone(x)\n    return x\n",
            "use of moved owned tensor `x`",
        );
    }

    #[test]
    fn nested_branch_ownership_joins_recursively() {
        check(&[(
            "case.seismic",
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn caller[N](x: tensor[N] f32, outer: bool, inner: bool):\n    if outer:\n        if inner:\n            take(x)\n        else:\n            take(x)\n    else:\n        take(x)\n    return\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn loop_ownership_distinguishes_zero_one_and_repeated_execution() {
        check(&[(
            "case.seismic",
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn zero[N](x: tensor[N] f32) -> tensor[N] f32:\n    for i in 0..0:\n        take(x)\n    return x\n\nfn one[N](x: tensor[N] f32):\n    for i in 0..1:\n        take(x)\n    return\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));

        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn one[N](x: tensor[N] f32) -> tensor[N] f32:\n    for i in 0..1:\n        take(x)\n    return x\n",
            "use of moved owned tensor `x`",
        );
        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn many[N](x: tensor[N] f32):\n    for i in 0..2:\n        take(x)\n    return\n",
            "loop may repeat after moving captured owned tensor `x`",
        );
        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn unknown[N](x: tensor[N] f32):\n    for i in 0..N:\n        take(x)\n    return\n",
            "loop may repeat after moving captured owned tensor `x`",
        );
    }

    #[test]
    fn repeated_loop_accepts_restored_state_and_joins_nested_paths() {
        check(&[(
            "case.seismic",
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn restored[N](seed: tensor[N] f32, choose: bool) -> tensor[N] f32:\n    let mut state = clone(seed)\n    for i in 0..2:\n        take(state)\n        if choose:\n            state = clone(seed)\n        else:\n            state = clone(seed)\n    return state\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));

        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn not_restored[N](seed: tensor[N] f32, choose: bool):\n    let mut state = clone(seed)\n    for i in 0..2:\n        take(state)\n        if choose:\n            state = clone(seed)\n        else:\n            let copy = clone(seed)\n    return\n",
            "loop may repeat after moving captured owned tensor `state`",
        );
        rejected(
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn nested[N](x: tensor[N] f32):\n    for outer in 0..2:\n        for inner in 0..1:\n            take(x)\n    return\n",
            "loop may repeat after moving captured owned tensor `x`",
        );
    }

    #[test]
    fn lexical_slice_borrows_enforce_shared_and_exclusive_access() {
        check(&[(
            "case.seismic",
            "fn f[N](x: &tensor[N] f32) -> f32:\n    let a = x[:]\n    let b = x[:]\n    return f32(a[0]) + f32(b[0])\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn f[N](x: &mut tensor[N] f32):\n    let mut slice = x[:]\n    x[0] = 1.0\n",
            "exclusive tensor borrow is live",
        );
    }

    #[test]
    fn logical_parallel_for_requires_iteration_disjoint_writes() {
        check(&[(
            "case.seismic",
            "fn fill[N](x: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        x[i] = 1.0\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn fill[N](x: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        x[0] = 1.0\n",
            "depends on its loop variable",
        );
        rejected(
            "fn sum[N](x: &tensor[N] f32) -> f32:\n    let mut total = f32(0.0)\n    parallel for i in 0..N:\n        total = total + f32(x[i])\n    return total\n",
            "parallel for",
        );
    }

    #[test]
    fn range_types_are_checked_and_borrowed_results_are_rejected() {
        check(&[(
            "case.seismic",
            "fn bounded[N](span: range[N], at: index[N]):\n    return\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn bad[N](x: &tensor[N] f32) -> &tensor[N] f32:\n    return x\n",
            "borrowed tensors cannot be returned",
        );
    }

    #[test]
    fn bounded_ranges_and_loop_kind_survive_checked_sir() {
        let program = check(&[(
            "case.seismic",
            "fn loops[N](x: &mut tensor[N] f32, selected: range[N]):\n    for i in selected:\n        x[i] = 1.0\n    parallel for j in 0..N:\n        x[j] = 2.0\n    return\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        let definition = &program.definitions[0];
        assert!(matches!(definition.params[1].ty, ValueType::Range { .. }));
        assert!(matches!(
            definition.body.root.statements[0],
            CheckedStmt::Loop {
                kind: LoopKind::Ordered,
                ..
            }
        ));
        assert!(matches!(
            definition.body.root.statements[1],
            CheckedStmt::Loop {
                kind: LoopKind::Independent,
                ..
            }
        ));
        assert!(matches!(
            definition.body.root.terminator,
            BlockTerminator::Return(_)
        ));

        rejected(
            "fn bad[N](x: &mut tensor[N] f32):\n    for i in N..0:\n        x[i] = 1.0\n",
            "range must prove",
        );
        rejected(
            "fn bad[N](selected: range[N]) -> i32:\n    return selected + 1\n",
            "range",
        );
    }

    #[test]
    fn returns_inside_loops_are_rejected() {
        rejected(
            "fn f[N](x: &tensor[N] f32) -> f32:\n    for i in 0..N:\n        return x[i]\n    return x[0]\n",
            "return inside a loop",
        );
    }

    #[test]
    fn recursion_is_rejected_before_specialization() {
        rejected(
            "fn fact(n: i32) -> i32:\n    return fact(n - 1)\n",
            "recursive call chain",
        );
        rejected(
            "fn even(n: i32) -> i32:\n    return odd(n - 1)\n\nfn odd(n: i32) -> i32:\n    return even(n - 1)\n",
            "recursive call chain",
        );
    }

    #[test]
    fn scalar_cannot_bind_tensor_helper_parameter() {
        rejected("fn first[K](v: &tensor[K] f32) -> f32:\n    return v[0]\n\nfn f[N](x: &tensor[N] f32) -> f32:\n    return first(x[0])\n", "expects");
    }

    #[test]
    fn returned_tuple_arity_is_checked() {
        rejected("fn pair[N](a: tensor[N] f32, b: tensor[N] f32) -> (tensor[N] f32, tensor[N] f32):\n    return a, b\n\nfn f[N](a: tensor[N] f32, b: tensor[N] f32) -> tensor[N] f32:\n    let only = pair(a, b)\n    return only\n", "returns");
    }

    #[test]
    fn concrete_lowering_requires_caller_elem_and_unordered_is_numerical_policy() {
        let source = "fn mm[M, K](a: &tensor[M, K] T, into: tensor[M] f32) -> tensor[M] f32:\n    let mut result = into\n    for i in 0..M:\n        result[i] = result[i] + f32(a[i, 0])\n    return result\nlower mm[M, K](a: &tensor[M, K] bf16, into: tensor[M] f32) -> tensor[M] f32 for cpu:\n    let mut result = into\n    for i in 0..M:\n        result[i] = result[i] + f32(a[i, 0])\n    return result\nlower mm[M, K](a: &tensor[M, K] T, into: tensor[M] f32) -> tensor[M] f32 for cpu:\n    let mut result = into\n    for i in 0..M:\n        result[i] = result[i] + f32(a[i, 0])\n    return result\nfn g[M, K](x: &tensor[M, K] A, acc: tensor[M] f32) -> tensor[M] f32:\n    return mm(x, acc)\n";
        let program = check(&[("case.seismic", source)]).unwrap_or_else(|e| panic!("{e}"));
        let g = program
            .definitions
            .iter()
            .find(|d| d.name == "g")
            .expect("g is defined");
        let bindings = &g.body.calls()[0].bindings;
        let concrete = bindings
            .iter()
            .find(|b| !b.requires_elems.is_empty())
            .expect("the bf16 lowering is a provisional candidate");
        assert_eq!(
            concrete.requires_elems,
            vec![("A".to_string(), Elem::Dtype(DType::BF16))]
        );
        let generic = bindings
            .iter()
            .find(|b| b.requires_elems.is_empty() && !b.elem_args.is_empty())
            .expect("the generic lowering binds T");
        assert_eq!(
            generic.elem_args,
            vec![("T".to_string(), Elem::Param("A".to_string()))]
        );
        check(&[("unordered.seismic", "fn f[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum, unordered=true)\n")]).expect("unordered sum is a selectable numerical alternative");
        rejected("fn f[N](x: &tensor[N] f32) -> i32:\n    return reduce(f32(x), 0, argmax, unordered=true)\n", "never accepts `unordered`");
    }

    #[test]
    fn parallel_for_requires_disjoint_tensor_writes() {
        rejected("fn f[N](x: &tensor[N] f32, result: tensor[N] f32) -> tensor[N] f32:\n    let mut output = result\n    parallel for i in 0..N:\n        output[0] = x[i]\n    return output\n", "depends on its loop variable");
    }

    #[test]
    fn packed_storage_is_readable_but_not_writable() {
        rejected(
            "fn f[N](x: &mut tensor[N] q4g64, v: &tensor[N] q4g64):\n    x[0:64] = load(v)\n",
            "readable and decodable but not writable",
        );
        check(&[(
            "case.seismic",
            "fn f[N](x: &tensor[N] q4g64) -> f32:\n    let v = decode(x)\n    return reduce(v, 0, sum)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn floating_reduction_types_through_the_registry() {
        let program = check(&[(
            "case.seismic",
            "fn f[N](x: &tensor[N] f16) -> f32:\n    return reduce(f16(x), 0, sum)\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        let definition = &program.definitions[0];
        match &definition.body.root.terminator {
            BlockTerminator::Return(values) => {
                assert_eq!(values[0].ty, ValueType::Scalar(DType::F32));
                assert!(matches!(
                    values[0].kind,
                    CheckedExprKind::Primitive {
                        id: crate::intrinsics::PrimitiveId::Reduce { .. },
                        ..
                    }
                ));
            }
            BlockTerminator::Continue => panic!("expected a return terminator"),
        }
    }

    #[test]
    fn parallel_for_writes_are_proved_injective_in_every_binder() {
        // Affine, multi-axis, mixed-radix (`i * C + j` with `j < C`), and
        // slice (`i * W : (i + 1) * W`) indices prove disjoint visits.
        check(&[(
            "case.seismic",
            "fn injective[R, C, W](x: &tensor[R, C] f32, a: &mut tensor[R] f32, b: &mut tensor[2 * R + 1] f32, c: &mut tensor[R, C] f32, d: &mut tensor[R * C] f32, e: &mut tensor[1, R * W] f32) where C - W >= 0:\n    parallel for i in 0..R:\n        a[i] = f32(x[i, 0])\n        b[2 * i + 1] = f32(x[i, 0])\n        e[0, i * W : (i + 1) * W] = f32(x[i, 0 : W])\n        parallel for j in 0..C:\n            c[i, j] = f32(x[i, j])\n            d[i * C + j] = f32(x[i, j])\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        // Data-dependent, non-injective, and jointly non-injective indices do not.
        rejected(
            "fn histogram[N, E](routes: &tensor[N] i32, counts: &mut tensor[E] i32):\n    parallel for i in 0..N:\n        counts[routes[i]] = counts[routes[i]] + 1\n",
            "injectively",
        );
        rejected(
            "fn halve[N](x: &tensor[N] f32, y: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        y[i / 2] = f32(x[i])\n",
            "injectively",
        );
        rejected(
            "fn diagonal[R, C](x: &tensor[R, C] f32, y: &mut tensor[R + C] f32):\n    parallel for i in 0..R:\n        parallel for j in 0..C:\n            y[i + j] = f32(x[i, j])\n",
            "injectively",
        );
    }

    #[test]
    fn atomic_updates_are_portable_and_typed() {
        // `add`, `max` and `min` are portable, and the one admitted way to
        // update a place from colliding visits.
        check(&[(
            "case.seismic",
            "fn histogram[N, E](routes: &tensor[N] i32, counts: &mut tensor[E] i32):\n    parallel for i in 0..N:\n        atomic(add, counts[routes[i]], 1)\n\nfn best[N, E](values: &tensor[N] f32, routes: &tensor[N] i32, top: &mut tensor[E] f32):\n    parallel for i in 0..N:\n        atomic(max, top[routes[i]], f32(values[i]))\n\nfn least[N, E](values: &tensor[N] i32, routes: &tensor[N] i32, low: &mut tensor[E] i32):\n    parallel for i in 0..N:\n        atomic(min, low[routes[i]], i32(values[i]))\n",
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        rejected(
            "fn flags[N](x: &mut tensor[N] bool):\n    parallel for i in 0..N:\n        atomic(add, x[i], true)\n",
            "not bool",
        );
        rejected(
            "fn xor[N](x: &mut tensor[N] i32):\n    parallel for i in 0..N:\n        atomic(xor, x[i], 1)\n",
            "add, max or min",
        );
    }
}
