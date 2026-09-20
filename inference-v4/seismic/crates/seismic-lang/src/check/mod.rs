//! The checker: resolution, contract families and target capabilities, typing of every
//! value kind, slice opacity, region results, stages and ports, modes/alias/effects,
//! bounds (via `sym`), initialization, and partial-domain obligations. Emits `sir`.
//! Entry point is `program::compile`.
//!
//! Symbol naming in emitted `Sym`s: the atom `Param("{name}#{VarId}")` is the runtime value
//! of body variable `VarId` (index parameters, loop indices, slice members, coordinates,
//! `let`-bound domain bounds); `Param("@dyn#n")` is the extent of a runtime-bounded range
//! view; `Param("@capacity#s")`/`Param("@valid#s")` are geometry of slice `s`. Every other
//! `Param` is a shape parameter of the definition.

mod call;
mod expr;
pub(crate) mod resolve;
mod stmt;

use crate::intrinsics::CapabilityId;
use crate::sir::{
    self, CallSite, DefKind, IntrinsicUse, Predicate, RegionDecl, SliceDecl, SliceParent, Var,
    VarId, VarKind,
};
use crate::span::{Diagnostic, Span};
use crate::sym::{Atom, Facts, Prover, Sym};
use crate::syntax::ast;
use crate::sir::Mode;
use crate::types::{Extent, RegionId, ResultTy, Shaped, SliceId, Ty};
use resolve::{Declared, Located, Resolved, Sig};
use std::collections::{BTreeSet, HashMap, HashSet};

/// The atom denoting the runtime value of body variable `id`.
pub fn var_atom(name: &str, id: VarId) -> Atom {
    Atom::Param(format!("{name}#{id}"))
}

/// The body variable an atom name denotes, if it is a variable atom.
pub fn atom_var(name: &str) -> Option<VarId> {
    if name.starts_with('@') {
        return None;
    }
    name.rsplit_once('#').and_then(|(_, id)| id.parse().ok())
}

/// Per-definition facts callers need: how each shape parameter is used.
#[derive(Clone, Debug, Default)]
pub(crate) struct Summary {
    /// Shape parameters used as numbers (arithmetic, `extent`, range bounds, coordinates).
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
    /// Second pass: summaries are complete; reject structural bindings into numeric uses.
    pub enforce: bool,
}

/// The function return boundary.
#[derive(Clone, Debug)]
pub(crate) struct YieldCtx {
    /// The current path has produced its value.
    pub done: bool,
    /// Enclosing element loops since the boundary; a value per loop visit is not one per path.
    pub loops: usize,
}

pub(crate) struct Checker<'a> {
    pub env: &'a Env<'a>,
    pub def: usize,
    pub sig: &'a Sig,
    pub kind: DefKind,
    /// The target whose forms this body may name: a backend-specific function or lowering target.
    pub target: Option<String>,
    pub requires: Vec<(CapabilityId, Span)>,
    pub used_capabilities: BTreeSet<CapabilityId>,
    pub intrinsic_uses: Vec<IntrinsicUse>,
    pub vars: Vec<Var>,
    pub scopes: Vec<HashMap<String, VarId>>,
    pub slices: Vec<SliceDecl>,
    pub regions: Vec<RegionDecl>,
    pub calls: Vec<CallSite>,
    pub facts: Facts,
    pub atoms: HashMap<VarId, Atom>,
    pub scalar_symbols: HashMap<VarId, Sym>,
    pub unassigned: HashSet<VarId>,
    /// Tiles the current `owned` loop assigns by its first write at the loop's own coordinates.
    pub pending_full_assign: Vec<(VarId, Vec<VarId>)>,
    /// Active logical ranges may collectively initialize uninitialized storage;
    /// coverage is proved over the completed loop before that storage becomes readable.
    pub init_loop_depth: usize,
    pub dyn_slices: Vec<(Option<sir::Expr>, Option<sir::Expr>, Sym, Atom)>,
    pub yields: Vec<YieldCtx>,
    pub view_roots: HashMap<VarId, VarId>,
    /// For each `let`-bound view: how many writes had happened when it was bound.
    pub view_bound: HashMap<VarId, usize>,
    pub partial_origin: HashMap<VarId, RegionId>,
    pub result_partials: HashMap<RegionId, Vec<bool>>,
    pub mutated: Vec<VarId>,
    /// State and storage roots read so far (pipeline stage conflicts).
    pub reads: Vec<VarId>,
    /// Variable floors of the enclosing element loops over a structural extent.
    pub structural_loops: Vec<usize>,
    /// Temporary checker-side representation of independent logical loops until
    /// the structured IR gains an explicit `parallel for` node.
    pub logical_parallel: Vec<(usize, VarId)>,
    pub published: HashSet<VarId>,
    pub summary: Summary,
    pub diagnostics: Vec<Diagnostic>,
    pub counter: usize,
    /// Names whose binding was rejected; uses of them are not reported again.
    pub poisoned: HashSet<String>,
    /// Owned tensor bindings consumed by a source-level move.
    pub moved: HashSet<VarId>,
    /// Lexically live slice/view borrows: binding -> (storage root, exclusive).
    pub borrows: HashMap<VarId, (VarId, bool)>,
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
            vars: Vec::new(),
            scopes: vec![HashMap::new()],
            slices: Vec::new(),
            regions: Vec::new(),
            calls: Vec::new(),
            facts: Facts::new(),
            atoms: HashMap::new(),
            scalar_symbols: HashMap::new(),
            unassigned: HashSet::new(),
            pending_full_assign: Vec::new(),
            init_loop_depth: 0,
            dyn_slices: Vec::new(),
            yields: Vec::new(),
            view_roots: HashMap::new(),
            view_bound: HashMap::new(),
            partial_origin: HashMap::new(),
            result_partials: HashMap::new(),
            mutated: Vec::new(),
            reads: Vec::new(),
            structural_loops: Vec::new(),
            logical_parallel: Vec::new(),
            published: HashSet::new(),
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
                Predicate::NonZero(_) | Predicate::Full(_) => {}
            }
        }
        for (i, p) in c.sig.params.iter().enumerate() {
            let id = c.declare(&p.name, p.ty.clone(), p.span, VarKind::Param(i));
            if let Ty::Index(bound) = &p.ty {
                let atom = var_atom(&p.name, id);
                c.facts
                    .set_range(atom.clone(), Sym::constant(0), bound.sub(&Sym::constant(1)));
                c.atoms.insert(id, atom);
            }
            if p.mode == Mode::Out && matches!(p.ty, Ty::Tile(_)) {
                c.unassigned.insert(id);
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

    pub fn lookup(&self, name: &str) -> Option<VarId> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    pub fn declare(&mut self, name: &str, ty: Ty, span: Span, kind: VarKind) -> VarId {
        let id = self.vars.len();
        self.vars.push(Var {
            name: name.to_string(),
            ty,
            kind,
            partial: false,
            span,
        });
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

    // ---- slices ----

    pub fn slice_parent(&self, id: SliceId) -> &SliceParent {
        &self.slices[id.0 as usize].parent
    }

    /// The semantic domain a slice lies in.
    pub fn root_domain(&self, mut id: SliceId) -> (Sym, Sym) {
        loop {
            match self.slice_parent(id) {
                SliceParent::Domain { lo, hi } => return (lo.clone(), hi.clone()),
                SliceParent::Refine(parent) | SliceParent::Rebind(parent) => id = *parent,
            }
        }
    }

    /// Whether every coordinate of `inner` is a coordinate of `outer`: the same slice or an
    /// explicit refinement chain. Equal widths never establish this.
    pub fn within(&self, mut inner: SliceId, outer: SliceId) -> bool {
        loop {
            if inner == outer {
                return true;
            }
            match self.slice_parent(inner) {
                SliceParent::Refine(parent) => inner = *parent,
                _ => return false,
            }
        }
    }

    pub fn slice_name(&self, id: SliceId) -> String {
        self.vars[self.slices[id.0 as usize].var].name.clone()
    }

    // ---- types ----

    pub fn same_extent(&self, a: &Extent, b: &Extent) -> bool {
        match (a, b) {
            (Extent::Semantic(x), Extent::Semantic(y)) => self.prover().zero(&x.sub(y)),
            (Extent::Structural(x), Extent::Structural(y)) => x == y,
            _ => false,
        }
    }

    pub fn same_axes(&self, a: &Shaped, b: &Shaped) -> bool {
        a.rank() == b.rank()
            && a.axes
                .iter()
                .zip(&b.axes)
                .all(|(x, y)| self.same_extent(x, y))
    }

    pub fn same_ty(&self, a: &Ty, b: &Ty) -> bool {
        match (a, b) {
            (Ty::Tensor(x), Ty::Tensor(y))
            | (Ty::View(x), Ty::View(y))
            | (Ty::Tile(x), Ty::Tile(y)) => x.elem == y.elem && self.same_axes(x, y),
            (Ty::Index(x), Ty::Index(y)) => self.prover().zero(&x.sub(y)),
            (Ty::Range(x), Ty::Range(y)) => self.prover().zero(&x.sub(y)),
            (Ty::Tuple(x), Ty::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| self.same_ty(p, q))
            }
            (Ty::Result(x), Ty::Result(y)) => x.origin == y.origin && x.producer == y.producer,
            _ => a == b,
        }
    }

    /// Whether a value of type `value` may be installed into state of type `target`
    /// (floats round to the target's element type).
    pub fn assignable(&self, target: &Ty, value: &Ty) -> bool {
        match (target, value) {
            (Ty::Scalar(a), _) => value
                .scalar_dtype()
                .is_some_and(|b| *a == b || (a.is_float() && b.is_float())),
            (Ty::Tile(a), Ty::Tile(b)) => self.same_axes(a, b) && elem_rounds(&b.elem, &a.elem),
            (Ty::Tuple(a), Ty::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| self.assignable(x, y))
            }
            _ => self.same_ty(target, value),
        }
    }

    /// Replace slice identities in a type (result member rebinding).
    pub fn rebind_ty(&self, ty: &Ty, map: &[(SliceId, SliceId)]) -> Ty {
        let slice = |s: SliceId| {
            map.iter()
                .find(|(from, _)| *from == s)
                .map_or(s, |(_, to)| *to)
        };
        let shaped = |s: &Shaped| Shaped {
            axes: s
                .axes
                .iter()
                .map(|a| match a {
                    Extent::Structural(id) => Extent::Structural(slice(*id)),
                    other => other.clone(),
                })
                .collect(),
            elem: s.elem.clone(),
            packed_axis: s.packed_axis,
        };
        match ty {
            Ty::Tensor(s) => Ty::Tensor(shaped(s)),
            Ty::View(s) => Ty::View(shaped(s)),
            Ty::Tile(s) => Ty::Tile(shaped(s)),
            Ty::Slice(s) => Ty::Slice(slice(*s)),
            Ty::Coord(s) => Ty::Coord(slice(*s)),
            Ty::Tuple(items) => Ty::Tuple(items.iter().map(|t| self.rebind_ty(t, map)).collect()),
            Ty::Result(r) => Ty::Result(Box::new(ResultTy {
                origin: r.origin,
                producer: r.producer,
                binders: r.binders.clone(),
                member: self.rebind_ty(&r.member, map),
            })),
            other => other.clone(),
        }
    }

    // ---- target-dependent forms ----

    /// A form of language.md section 6. Legal only in a body with a target context.
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

    /// Whether this body has geometry authority (target code).
    pub fn geometry_authority(&self) -> bool {
        self.target.is_some()
    }

    // ---- partial-domain obligations ----

    /// Partial values are ordinary while a structural merge combines them or while code
    /// traverses the exact result partition they belong to. This is structural authority,
    /// independent of numerical precision.
    pub fn partial_free(&self) -> bool {
        false
    }

    pub fn forbid_partial(&mut self, e: &sir::Expr, use_: &str) {
        if e.partial && !self.partial_free() {
            self.error(e.span, format!("partial-domain value used as {use_}: a value yielded per slice or reduced over a structural axis may only be forwarded, stored in results, combined by `merge`, or accumulated into `let mut` state by `+`, `max`, or `min` inside a traversal of the same result"));
        }
    }

    // ---- effects ----

    /// The variable whose storage a place or view expression designates.
    pub fn root_var(&self, e: &sir::Expr) -> Option<VarId> {
        match &e.kind {
            sir::ExprKind::Var(v) => Some(self.view_roots.get(v).copied().unwrap_or(*v)),
            sir::ExprKind::Index { base, .. }
            | sir::ExprKind::Transpose(base)
            | sir::ExprKind::Reshape { base, .. }
            | sir::ExprKind::Accessor { base, .. } => self.root_var(base),
            _ => None,
        }
    }

    /// The binding through which a place is named, before following view aliases to their
    /// backing storage. Mutation requires permission from both this binding and the root.
    fn place_binding(&self, e: &sir::Expr) -> Option<VarId> {
        match &e.kind {
            sir::ExprKind::Var(v) => Some(*v),
            sir::ExprKind::Index { base, .. }
            | sir::ExprKind::Transpose(base)
            | sir::ExprKind::Reshape { base, .. }
            | sir::ExprKind::Accessor { base, .. } => self.place_binding(base),
            _ => None,
        }
    }

    /// Slices a place is selected by: directly (binder, member, coordinate) or through a
    /// data-dependent index computed from a member. Includes enclosing refinement parents.
    fn selecting_slices(&self, e: &sir::Expr, out: &mut HashSet<SliceId>) {
        fn mentions(c: &Checker, e: &sir::Expr, out: &mut HashSet<SliceId>) {
            match &e.kind {
                sir::ExprKind::Var(v) | sir::ExprKind::CoordOf(v) => {
                    match (&c.vars[*v].kind, &c.vars[*v].ty) {
                        (VarKind::SliceMember(s), _) | (_, Ty::Coord(s)) | (_, Ty::Slice(s)) => {
                            c.with_parents(*s, out)
                        }
                        _ => {}
                    }
                }
                sir::ExprKind::Index { base, indices } => {
                    mentions(c, base, out);
                    c.index_slices(indices, out);
                }
                sir::ExprKind::Binary { lhs, rhs, .. } => {
                    mentions(c, lhs, out);
                    mentions(c, rhs, out);
                }
                sir::ExprKind::Unary { expr, .. } | sir::ExprKind::Cast { expr, .. } => {
                    mentions(c, expr, out)
                }
                _ => {}
            }
        }
        match &e.kind {
            sir::ExprKind::Index { base, indices } => {
                for index in indices {
                    match index {
                        sir::Index::Point(p) => mentions(self, p, out),
                        sir::Index::Range { start, end } => {
                            start.iter().chain(end).for_each(|b| mentions(self, b, out))
                        }
                        _ => {}
                    }
                }
                self.index_slices(indices, out);
                self.selecting_slices(base, out);
            }
            sir::ExprKind::Transpose(base)
            | sir::ExprKind::Reshape { base, .. }
            | sir::ExprKind::Accessor { base, .. } => self.selecting_slices(base, out),
            _ => {}
        }
    }

    fn index_slices(&self, indices: &[sir::Index], out: &mut HashSet<SliceId>) {
        for index in indices {
            match index {
                sir::Index::Slice(s) => self.with_parents(*s, out),
                sir::Index::Coord(v) => {
                    if let Ty::Coord(s) = &self.vars[*v].ty {
                        self.with_parents(*s, out);
                    }
                }
                _ => {}
            }
        }
    }

    fn with_parents(&self, mut s: SliceId, out: &mut HashSet<SliceId>) {
        loop {
            out.insert(s);
            match self.slice_parent(s) {
                SliceParent::Refine(parent) => s = *parent,
                _ => return,
            }
        }
    }

    /// Check and record a write to the storage `place` designates. `whole` is an update of
    /// the state object itself (assignment, `inout` of the whole variable).
    pub fn write(&mut self, place: &sir::Expr, span: Span, whole: bool) -> Option<VarId> {
        let Some(binding) = self.place_binding(place) else {
            self.error(span, "a write needs a place: an `out`/`inout` parameter, local `let mut` state, or a mutable view of one");
            return None;
        };
        let binding_name = self.vars[binding].name.clone();
        match &self.vars[binding].kind {
            VarKind::Param(i) if self.sig.params[*i].mode == Mode::In => {
                self.error(
                    span,
                    format!("`{binding_name}` is a read-only parameter; writing requires `out` or `inout`"),
                );
                return None;
            }
            VarKind::Param(_) | VarKind::State => {}
            VarKind::Value => {
                self.error(span, format!("`{binding_name}` is an immutable `let` binding; a mutable view or local state is declared with `let mut`"));
                return None;
            }
            _ => {
                self.error(span, format!("`{binding_name}` is not mutable state"));
                return None;
            }
        }
        let Some(root) = self.root_var(place) else {
            self.error(span, "a write needs a place: an `out`/`inout` parameter, local `let mut` state, or a mutable view of one");
            return None;
        };
        if self
            .borrows
            .iter()
            .any(|(borrow, (borrowed, _))| *borrowed == root && *borrow != binding)
        {
            self.error(
                span,
                format!(
                    "cannot mutate `{}` while a tensor borrow is live",
                    self.vars[root].name
                ),
            );
            return None;
        }
        for (floor, index) in self.logical_parallel.clone().into_iter().rev() {
            if root >= floor {
                continue;
            }
            if whole || !expr::mentions_var(place, index) {
                self.error(
                    span,
                    format!(
                        "a `parallel for` body may mutate captured tensor storage only through an index that depends on its loop variable"
                    ),
                );
                return None;
            }
        }
        let name = self.vars[root].name.clone();
        match &self.vars[root].kind {
            VarKind::Param(i) if self.sig.params[*i].mode == Mode::In => {
                self.error(
                    span,
                    format!("`{name}` is a read-only parameter; writing requires `out` or `inout`"),
                );
                return None;
            }
            VarKind::Param(_) | VarKind::State => {}
            VarKind::Value => {
                self.error(span, format!("`{name}` is an immutable `let` value; mutable state is declared with `let mut`"));
                return None;
            }
            _ => {
                self.error(span, format!("`{name}` is not mutable state"));
                return None;
            }
        }
        let mut selected = HashSet::new();
        self.selecting_slices(place, &mut selected);
        // State carried across the coordinates of a slice is one aggregate per tuned piece.
        if whole && !self.partial_free() && self.structural_loops.iter().any(|floor| *floor > root)
        {
            self.vars[root].partial = true;
        }
        self.mutated.push(root);
        self.published.insert(root);
        self.scalar_symbols.remove(&root);
        let tensor_effect = matches!(self.vars[root].ty, Ty::Tensor(_) | Ty::View(_));
        self.dyn_slices.retain(|(start, end, _, _)| {
            !tensor_effect && !start.iter().chain(end).any(|b| expr::mentions_var(b, root))
        });
        Some(root)
    }

    // ---- result ----

    fn finish(
        mut self,
        block: sir::Block,
        span: Span,
    ) -> (sir::Body, Summary, Vec<IntrinsicUse>, Vec<Diagnostic>) {
        // Coverage of `out` parameters is judged on bodies that are otherwise well-formed.
        let well_formed = self.diagnostics.is_empty();
        for (i, p) in self.sig.params.iter().enumerate().filter(|_| well_formed) {
            if p.mode != Mode::Out {
                continue;
            }
            let Some(id) = self.vars.iter().position(|v| v.kind == VarKind::Param(i)) else {
                continue;
            };
            if self.unassigned.contains(&id) {
                self.error(p.span, format!("`out` tile `{}` is not assigned on every path; assign every element through `for … in owned({})`", p.name, p.name));
            } else if !self.published.contains(&id) {
                self.error(p.span, format!("`out` parameter `{}` is never published; every element of an `out` parameter must be written", p.name));
            }
        }
        if well_formed
            && self.sig.result != Ty::Void
            && !self.yields.first().is_some_and(|y| y.done)
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
        let body = sir::Body {
            vars: self.vars,
            slices: self.slices,
            regions: self.regions,
            calls: self.calls,
            block,
        };
        (body, self.summary, self.intrinsic_uses, self.diagnostics)
    }
}

/// Whether publishing/assigning elements of `value` into storage of `target` is a defined rounding.
pub(crate) fn elem_rounds(value: &crate::types::Elem, target: &crate::types::Elem) -> bool {
    use crate::types::Elem;
    match (value, target) {
        (Elem::Repr(a), Elem::Repr(b)) => a == b,
        (Elem::Repr(_), _) | (_, Elem::Repr(_)) => false,
        (Elem::Dtype(a), Elem::Dtype(b)) => a == b || (a.is_float() && b.is_float()),
        (Elem::Dtype(a), Elem::Param(_)) | (Elem::Param(_), Elem::Dtype(a)) => a.is_float(),
        (Elem::Param(_), Elem::Param(_)) => true,
    }
}

struct CheckedBody {
    body: sir::Body,
    summary: Summary,
    intrinsic_uses: Vec<IntrinsicUse>,
    diagnostics: Vec<Diagnostic>,
}

fn check_definition(env: &Env, def: usize) -> CheckedBody {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    c.yields.push(YieldCtx {
        done: false,
        loops: 0,
    });
    let stmts = c.block(declared.body);
    let facts = c.facts.clone();
    let (body, mut summary, intrinsic_uses, diagnostics) = c.finish(stmts, declared.name_span);
    for (parameter, declared_param) in declared.sig.params.iter().enumerate() {
        if declared_param.ownership == resolve::ParamOwnership::Exclusive {
            if let Ty::Tensor(shaped) | Ty::View(shaped) = &declared_param.ty {
                if definitely_initializes(&body, parameter, shaped, &facts) {
                    summary.full_init.insert(parameter);
                }
            }
        }
    }
    CheckedBody {
        body,
        summary,
        intrinsic_uses,
        diagnostics,
    }
}

fn definitely_initializes(
    body: &sir::Body,
    parameter: usize,
    shaped: &Shaped,
    facts: &Facts,
) -> bool {
    let Some(variable) = body
        .vars
        .iter()
        .position(|var| var.kind == sir::VarKind::Param(parameter))
    else {
        return false;
    };
    definitely_initializes_var(&body.vars, &body.block, variable, shaped, facts)
}

pub(crate) fn definitely_initializes_var(
    vars: &[sir::Var],
    block: &sir::Block,
    variable: VarId,
    shaped: &Shaped,
    facts: &Facts,
) -> bool {
    fn root(expr: &sir::Expr) -> Option<VarId> {
        match &expr.kind {
            sir::ExprKind::Var(var) => Some(*var),
            sir::ExprKind::Index { base, .. }
            | sir::ExprKind::Reshape { base, .. }
            | sir::ExprKind::Transpose(base)
            | sir::ExprKind::Accessor { base, .. } => root(base),
            _ => None,
        }
    }
    fn point_covers(
        vars: &[sir::Var],
        point: &sir::Expr,
        extent: &Sym,
        loops: &[(VarId, Sym)],
    ) -> bool {
        match &point.kind {
            sir::ExprKind::Var(var) => loops.iter().any(|(loop_var, bound)| loop_var == var && bound == extent),
            _ => {
                if extent.as_constant() == Some(1) && point.sym.as_ref().is_some_and(Sym::is_zero) {
                    return true;
                }
                // Nested full loops form a bijective mixed-radix enumeration of a
                // flattened axis: `h * W + i`, and its higher-rank generalization.
                let mut total = Sym::constant(1);
                let mut linear = Sym::constant(0);
                for (var, bound) in loops {
                    let atom = var_atom(&vars[*var].name, *var);
                    linear = linear.mul(bound).add(&Sym::atom(atom));
                    total = total.mul(bound);
                }
                &total == extent && point.sym.as_ref() == Some(&linear)
            }
        }
    }
    fn index_covers(
        vars: &[sir::Var],
        index: &sir::Index,
        extent: &Sym,
        loops: &[(VarId, Sym)],
        facts: &Facts,
    ) -> bool {
        match index {
            sir::Index::Point(point) => point_covers(vars, point, extent, loops),
            sir::Index::Range { start, end } => {
                let full = start.as_ref().is_none_or(|start| start.sym.as_ref().is_some_and(Sym::is_zero))
                    && end.as_ref().is_none_or(|end| end.sym.as_ref() == Some(extent));
                full || loops.iter().any(|(var, partitions)| {
                    let width = extent.quot(partitions);
                    if !Prover::new(facts).zero(&width.mul(partitions).sub(extent)) {
                        return false;
                    }
                    let coordinate = Sym::atom(var_atom(&vars[*var].name, *var));
                    let expected_start = coordinate.mul(&width);
                    let expected_end = coordinate.add(&Sym::constant(1)).mul(&width);
                    start.as_ref().and_then(|value| value.sym.as_ref()) == Some(&expected_start)
                        && end.as_ref().and_then(|value| value.sym.as_ref()) == Some(&expected_end)
                })
            }
            sir::Index::Coord(_) | sir::Index::Slice(_) => false,
        }
    }
    fn target_covers(
        vars: &[sir::Var],
        target: &sir::Expr,
        variable: VarId,
        extents: &[Extent],
        loops: &[(VarId, Sym)],
        facts: &Facts,
    ) -> bool {
        if root(target) != Some(variable) {
            return false;
        }
        let sir::ExprKind::Index { indices, .. } = &target.kind else {
            return matches!(target.kind, sir::ExprKind::Var(_));
        };
        if indices.len() > extents.len() {
            return false;
        }
        indices.iter().zip(extents).all(|(index, extent)| {
            let Extent::Semantic(extent) = extent else { return false };
            index_covers(vars, index, extent, loops, facts)
        })
        // Omitted trailing indices denote the complete remaining tensor slice.
    }
    fn writes(
        vars: &[sir::Var],
        block: &sir::Block,
        variable: VarId,
        extents: &[Extent],
        loops: &[(VarId, Sym)],
        facts: &Facts,
    ) -> bool {
        block.iter().any(|statement| match &statement.kind {
            sir::StmtKind::Assign { target, .. } | sir::StmtKind::Publish { destination: target, .. } => {
                target_covers(vars, target, variable, extents, loops, facts)
            }
            sir::StmtKind::If { then, els, .. } => {
                writes(vars, then, variable, extents, loops, facts)
                    && writes(vars, els, variable, extents, loops, facts)
            }
            sir::StmtKind::Range { var, lo, hi, value: None, body: nested, .. } => {
                let Some(bound) = hi.sym.clone().filter(|_| lo.sym.as_ref().is_some_and(Sym::is_zero)) else {
                    return false;
                };
                let mut nested_loops = loops.to_vec();
                nested_loops.push((*var, bound));
                writes(vars, nested, variable, extents, &nested_loops, facts)
            }
            _ => false,
        })
    }
    writes(vars, block, variable, &shaped.axes, &[], facts)
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
                    && candidates.iter().all(|(callee, parameter)| summaries[*callee].full_init.contains(parameter))
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

/// Check every declared body of the closed program. Returns the definitions and families.
pub(crate) fn check_program(
    files: &[(usize, ast::File)],
    diagnostics: &mut Vec<Located>,
) -> (Vec<sir::Definition>, Vec<sir::ContractFamily>) {
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
        // Parameters are the first variables of a checked body.
        let params = declared
            .sig
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| sir::Param {
                name: p.name.clone(),
                mode: p.mode,
                ty: p.ty.clone(),
                var: i,
            })
            .collect();
        definitions.push(sir::Definition {
            id: sir::DefId(def as u32),
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
    (definitions, resolved.families)
}

#[cfg(test)]
mod tests {
    use crate::program::{compile, SourceFile};
    use crate::sir::Program;

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
        let call = &body.calls[0];
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
        rejected("fn f[N](result: tensor[N] f32, flag: bool) -> tensor[N] f32:\n    if flag:\n        return result\n", "one path of this `if` yields/returns");
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
            "only tile elements are assigned",
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
            vec![crate::types::Ty::Scalar(crate::types::DType::F32)]
        );
        assert_eq!(
            definition.intrinsic_uses[0].result,
            crate::types::Ty::Scalar(crate::types::DType::F32)
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
        assert!(used.operation.produces_owned_result());
        assert_eq!(used.arguments.len(), 2);
        assert!(matches!(used.result, crate::types::Ty::Tensor(_)));
        assert_eq!(
            used.result.shaped().map(|shape| &shape.elem),
            Some(&crate::types::Elem::Dtype(crate::types::DType::F32))
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
            "fn take[N](x: tensor[N] f32):\n    return\n\nfn caller[N](x: tensor[N] f32) -> tensor[N] f32:\n    take(x)\n    return x\n",
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
        assert!(matches!(
            definition.params[1].ty,
            crate::types::Ty::Range(_)
        ));
        assert!(matches!(
            definition.body.block[0].kind,
            crate::sir::StmtKind::Range {
                kind: crate::sir::LoopKind::Ordered,
                ..
            }
        ));
        assert!(matches!(
            definition.body.block[1].kind,
            crate::sir::StmtKind::Range {
                kind: crate::sir::LoopKind::Parallel,
                ..
            }
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
        let bindings = &g.body.calls[0].bindings;
        let concrete = bindings
            .iter()
            .find(|b| !b.requires_elems.is_empty())
            .expect("the bf16 lowering is a provisional candidate");
        assert_eq!(
            concrete.requires_elems,
            vec![(
                "A".to_string(),
                crate::types::Elem::Dtype(crate::types::DType::BF16)
            )]
        );
        let generic = bindings
            .iter()
            .find(|b| b.requires_elems.is_empty() && !b.elem_args.is_empty())
            .expect("the generic lowering binds T");
        assert_eq!(
            generic.elem_args,
            vec![("T".to_string(), crate::types::Elem::Param("A".to_string()))]
        );
        check(&[("unordered.seismic", "fn f[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(x), 0, sum, unordered=true)\n")]).expect("unordered sum is a selectable numerical alternative");
        rejected("fn f[N](x: &tensor[N] f32) -> i32:\n    return reduce(f32(x), 0, argmax, unordered=true)\n", "never accepts `unordered`");
    }

    #[test]
    fn parallel_for_requires_disjoint_tensor_writes() {
        rejected("fn f[N](x: &tensor[N] f32, result: tensor[N] f32) -> tensor[N] f32:\n    let mut output = result\n    parallel for i in 0..N:\n        output[0] = x[i]\n    return output\n", "depends on its loop variable");
    }
}
