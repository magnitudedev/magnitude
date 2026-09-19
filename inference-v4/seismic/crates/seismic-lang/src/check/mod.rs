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

use crate::sir::{
    self, CallSite, DefKind, Predicate, RegionDecl, SliceDecl, SliceParent, Var, VarId, VarKind,
};
use crate::span::{Diagnostic, Span};
use crate::sym::{Atom, Facts, Prover, Sym};
use crate::syntax::ast::{self, Mode, RegionMode};
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
}

pub(crate) struct Env<'a> {
    pub resolved: &'a Resolved<'a>,
    pub summaries: &'a [Summary],
    /// Second pass: summaries are complete; reject structural bindings into numeric uses.
    pub enforce: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YieldKind {
    Function,
    Region,
    Stage,
    Merge,
}

/// One result boundary: a function (`return`), a region visit, a stage or a merge body (`yield`).
#[derive(Clone, Debug)]
pub(crate) struct YieldCtx {
    pub kind: YieldKind,
    /// The current path has produced its value.
    pub done: bool,
    /// Enclosing element loops since the boundary; a value per loop visit is not one per path.
    pub loops: usize,
    pub schema: Option<Vec<Ty>>,
    pub partials: Vec<bool>,
    /// A `yield` on this boundary was itself rejected; do not also report it missing.
    pub failed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FrameKind {
    Region {
        id: RegionId,
        mode: RegionMode,
        origin: Option<RegionId>,
    },
    Stage {
        pipeline: bool,
    },
    Merge,
}

/// An ownership scope. Variables with id below `floor` are enclosing.
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub kind: FrameKind,
    pub floor: usize,
}

pub(crate) struct Checker<'a> {
    pub env: &'a Env<'a>,
    pub def: usize,
    pub sig: &'a Sig,
    pub kind: DefKind,
    /// The target whose forms this body may name: a backend-specific function or lowering target.
    pub target: Option<String>,
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
    pub dyn_slices: Vec<(Option<sir::Expr>, Option<sir::Expr>, Sym, Atom)>,
    pub frames: Vec<Frame>,
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
    pub published: HashSet<VarId>,
    pub summary: Summary,
    pub diagnostics: Vec<Diagnostic>,
    pub counter: usize,
    /// Names whose binding was rejected; uses of them are not reported again.
    pub poisoned: HashSet<String>,
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
            dyn_slices: Vec::new(),
            frames: Vec::new(),
            yields: Vec::new(),
            view_roots: HashMap::new(),
            view_bound: HashMap::new(),
            partial_origin: HashMap::new(),
            result_partials: HashMap::new(),
            mutated: Vec::new(),
            reads: Vec::new(),
            structural_loops: Vec::new(),
            published: HashSet::new(),
            summary: Summary::default(),
            diagnostics: Vec::new(),
            counter: 0,
            poisoned: HashSet::new(),
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
        self.frames.iter().any(|frame| {
            matches!(frame.kind, FrameKind::Merge | FrameKind::Region { origin: Some(_), .. })
        })
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
        for frame in self.frames.clone().iter().rev().filter(|f| f.floor > root) {
            match &frame.kind {
                FrameKind::Merge => {
                    self.error(span, format!("a `merge` body has no externally visible effects; it cannot write `{name}`"));
                    return None;
                }
                FrameKind::Region {
                    id,
                    mode: RegionMode::Parallel,
                    ..
                } => {
                    if whole {
                        self.error(span, format!("a `parallel` body cannot mutate enclosing state `{name}`; yield per-slice values and combine them with `merge` or an ordered traversal"));
                        return None;
                    }
                    let binders = self.regions[id.0 as usize].binders.clone();
                    if let Some(missing) = binders.iter().find(|b| !selected.contains(b)) {
                        let binder = self.slice_name(*missing);
                        self.error(span, format!("cannot prove `parallel` visits write disjoint views of `{name}`: binder `{binder}` does not select the destination"));
                        return None;
                    }
                }
                _ => {}
            }
        }
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

    fn finish(mut self, block: sir::Block, span: Span) -> (sir::Body, Summary, Vec<Diagnostic>) {
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
        let body = sir::Body {
            vars: self.vars,
            slices: self.slices,
            regions: self.regions,
            calls: self.calls,
            block,
        };
        (body, self.summary, self.diagnostics)
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
    diagnostics: Vec<Diagnostic>,
}

fn check_definition(env: &Env, def: usize) -> CheckedBody {
    let declared = &env.resolved.declared[def];
    let mut c = Checker::new(env, def);
    c.yields.push(YieldCtx {
        kind: YieldKind::Function,
        done: false,
        loops: 0,
        schema: None,
        partials: Vec::new(),
        failed: false,
    });
    let stmts = c.block(declared.body);
    let (body, summary, diagnostics) = c.finish(stmts, declared.name_span);
    CheckedBody {
        body,
        summary,
        diagnostics,
    }
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
    fn reference_sources_check() {
        let program = check(&[
            (
                "rms_norm.seismic",
                include_str!("../../../../../seismic-std/lib/kernels/rms_norm.seismic"),
            ),
            (
                "linear.seismic",
                include_str!("../../../../../seismic-std/lib/kernels/linear.seismic"),
            ),
            (
                "matmul.seismic",
                include_str!("../../../../../seismic-std/lib/constructs/matmul.seismic"),
            ),
            (
                "matmul-cpu.seismic",
                include_str!("../../../../../seismic-std/lib/constructs/matmul-cpu.seismic"),
            ),
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        let linear = program.family("linear").expect("linear is defined");
        let body = &program.definition(linear.bodies[0]).body;
        // `matmul(load(x[rows, k]), load(weight[cols, k]), into=acc)` binds M, N, K structurally.
        let call = &body.calls[0];
        assert!(!call.bindings.is_empty());
        assert!(call.bindings[0]
            .shape_args
            .iter()
            .all(|(_, e)| matches!(e, crate::types::Extent::Structural(_))));
        assert_eq!(call.bindings[0].arg_order, vec![0, 1, 2]);
    }

    #[test]
    fn region_results_stages_and_merge_check() {
        check(&[(
            "sum.seismic",
            "fn sum_squares[N](x: tensor[N] f32, out y: tensor[1] f32):\n    stage prepare:\n        let partials = parallel [p] in 0..N:\n            let values = f32(x[p])\n            yield reduce(values * values, 0, sum)\n        yield partials\n    stage finish(partials):\n        let mut total = f32(0.0)\n        ordered [piece] in partials:\n            total = total + partials[piece]\n        publish total to y[0]\n\nfn merged[K](x: tensor[K] f32, out y: tensor[1] f32):\n    let t = parallel [part] in 0..K:\n        yield reduce(f32(x[part]), 0, sum)\n    merge (left, right) identity f32(0.0):\n        yield left + right\n    publish t to y[0]\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn slice_width_query_rejected() {
        rejected("fn f[M, N](x: tensor[M, N] f32, out y: tensor[M, N] f32):\n    parallel [cols] in 0..N:\n        let v = x[:, cols]\n        let w = extent(v, 1)\n        publish f32(v) to y[:, cols]\n", "structural extent is never a number");
    }

    #[test]
    fn unrelated_same_width_slices_rejected() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[N] f32):\n    parallel [a] in 0..N:\n        let t = f32(x[a])\n        ordered [b] in 0..N:\n            let u = f32(x[b])\n            let s = t + u\n        publish t to y[a]\n", "unrelated slices");
    }

    #[test]
    fn mask_as_if_condition_rejected() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[N] f32):\n    parallel [p] in 0..N:\n        let t = f32(x[p])\n        let m = t > 0.0\n        if m:\n            publish t to y[p]\n        else:\n            publish t * 2.0 to y[p]\n", "a mask is a `bool` tile");
    }

    #[test]
    fn yield_on_one_path_rejected() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[N] f32, flag: i32):\n    let r = parallel [p] in 0..N:\n        let t = f32(x[p])\n        if flag > 0:\n            yield t\n    parallel [p] in r:\n        publish r[p] to y[p]\n", "one path of this `if` yields");
    }

    #[test]
    fn parallel_mutating_enclosing_state_rejected() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[1] f32):\n    let mut total = f32(0.0)\n    parallel [p] in 0..N:\n        total = total + f32(1.0)\n    publish total to y[0]\n", "`parallel` body cannot mutate enclosing state");
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
    fn let_mut_is_required_for_writable_view_aliases() {
        rejected(
            "fn f[N](out y: tensor[N] f32):\n    let alias = y[:]\n    publish zeros_like(alias, dtype=f32) to alias\n",
            "immutable `let` binding",
        );
        check(&[(
            "case.seismic",
            "fn f[N](out y: tensor[N] f32):\n    let mut alias = y[:]\n    publish zeros_like(alias, dtype=f32) to alias\n",
        )])
        .unwrap_or_else(|e| panic!("{e}"));
        rejected(
            "fn f[N](x: tensor[N] f32):\n    let mut alias = x[:]\n    publish zeros_like(alias, dtype=f32) to alias\n",
            "read-only parameter",
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
    fn implementation_contracts_preserve_shapes_elements_and_aliases() {
        rejected(
            "fn f[M, N](x: tensor[M, N] f32, out y: tensor[M, N] f32):\n    publish x to y\n\nlower f[M, N](x: tensor[M, M] f32, out y: tensor[M, N] f32) for cpu:\n    publish x to y\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn f[M](x: tensor[M] f32, out y: tensor[M] f32) alias(x, y):\n    publish x to y\n\nlower f[M](x: tensor[M] f32, out y: tensor[M] f32) for cpu:\n    publish x to y\n",
            "identical `alias` permissions",
        );
        rejected(
            "fn f[M, N](x: tensor[M, N] f32) -> tile[M, N] f32:\n    return load(x)\n\nlower f[M, N](x: tensor[M, N] f32) -> tile[N, M] f32 for cpu:\n    return load(x.T)\n",
            "equivalent results",
        );

        check(&[(
            "case.seismic",
            "fn f[M](x: tensor[M] T, out y: tensor[M] T):\n    publish x to y\n\nlower f[M](x: tensor[M] bf16, out y: tensor[M] bf16) for cpu:\n    publish x to y\n",
        )])
        .unwrap_or_else(|error| panic!("concrete element specialization was rejected:\n{error}"));

        rejected(
            "fn split[M](x: tensor[M] T, w: tensor[M] T, out y: tensor[M] T):\n    publish x to y\n\nlower split[M](x: tensor[M] U, w: tensor[M] V, out y: tensor[M] U) for cpu:\n    publish x to y\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn mixed[M](x: tensor[M] T, w: tensor[M] T, out y: tensor[M] T):\n    publish x to y\n\nlower mixed[M](x: tensor[M] bf16, w: tensor[M] U, out y: tensor[M] bf16) for cpu:\n    publish x to y\n",
            "equivalent parameter shape and element relationships",
        );
        rejected(
            "fn concrete[M](x: tensor[M] bf16, out y: tensor[M] bf16):\n    publish x to y\n\nlower concrete[M](x: tensor[M] T, out y: tensor[M] T) for cpu:\n    publish x to y\n",
            "equivalent parameter shape and element relationships",
        );
        check(&[(
            "case.seismic",
            "fn converge[M](x: tensor[M] T, w: tensor[M] U, out y: tensor[M] T):\n    publish x to y\n\nlower converge[M](x: tensor[M] V, w: tensor[M] V, out y: tensor[M] V) for cpu:\n    publish x to y\n",
        )])
        .unwrap_or_else(|error| panic!("independent element parameters could not converge:\n{error}"));
    }

    #[test]
    fn portable_body_cannot_name_target_intrinsics() {
        rejected(
            "fn f(x: f32) -> f32:\n    return metal.simd_sum(x)\n",
            "cannot appear in a portable body",
        );
    }

    #[test]
    fn structural_binding_into_numeric_helper_rejected() {
        rejected("fn width[K](v: view[K] f32) -> i32:\n    return extent(v, 0)\n\nfn f[N](x: tensor[N] f32, out y: tensor[N] f32):\n    parallel [p] in 0..N:\n        let w = width(x[p])\n        publish f32(x[p]) to y[p]\n", "binds it to a structural extent");
    }

    #[test]
    fn result_member_of_different_origin_rejected() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[N] f32):\n    let r = parallel [p] in 0..N:\n        yield f32(x[p])\n    parallel [q] in 0..N:\n        publish r[q] to y[q]\n", "does not come from this result's origin");
    }

    #[test]
    fn concrete_lowering_requires_caller_elem_and_unordered_is_numerical_policy() {
        let source = "fn mm[M, K](a: tile[M, K] T, inout into: tile[M] f32):\n    for i in owned(into):\n        into[i] = into[i] + f32(a[i, 0])\nlower mm[M, K](a: tile[M, K] bf16, inout into: tile[M] f32) for cpu:\n    for i in owned(into):\n        into[i] = into[i] + f32(a[i, 0])\nlower mm[M, K](a: tile[M, K] T, inout into: tile[M] f32) for cpu:\n    for i in owned(into):\n        into[i] = into[i] + f32(a[i, 0])\nfn g[M, K](x: tensor[M, K] A, out y: tensor[M] f32):\n    let mut acc = reduce(f32(x), 1, sum, unordered=true)\n    mm(load(x), acc)\n    publish acc to y\n";
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
        check(&[("unordered.seismic", "fn f[N](x: tensor[N] f32, out y: tensor[1] f32):\n    publish reduce(f32(x), 0, sum, unordered=true) to y[0]\n")]).expect("unordered sum is a selectable numerical alternative");
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[1] i32):\n    publish reduce(f32(x), 0, argmax, unordered=true) to y[0]\n", "never accepts `unordered`");
    }

    #[test]
    fn partial_value_cannot_be_published() {
        rejected("fn f[N](x: tensor[N] f32, out y: tensor[N] f32):\n    parallel [p] in 0..N:\n        let t = f32(x[p])\n        let count = reduce(t, 0, sum)\n        publish t * count to y[p]\n", "partial-domain value");
    }
}
