//! Expressions: literals, names, operators, indexing (points, slices, coordinates,
//! ranges), result member selection, attributes, tile allocation and casts.

use super::Checker;
use crate::repr;
use crate::span::Span;
use crate::sir::{Expr, ExprKind, Index, VarId, VarKind};
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, UnaryOp};
use crate::types::{DType, Elem, Extent, Shaped, SliceId, Ty};
use crate::sym::{Atom, Sym};

/// Whether `e` reads variable `var`.
pub(crate) fn mentions_var(e: &Expr, var: VarId) -> bool {
    match &e.kind {
        ExprKind::Var(v) | ExprKind::CoordOf(v) => *v == var,
        ExprKind::Index { base, indices } => {
            mentions_var(base, var)
                || indices.iter().any(|i| match i {
                    Index::Point(p) => mentions_var(p, var),
                    Index::Coord(v) => *v == var,
                    Index::Range { start, end } => start.iter().chain(end).any(|b| mentions_var(b, var)),
                    Index::Slice(_) => false,
                })
        }
        ExprKind::Binary { lhs, rhs, .. } => mentions_var(lhs, var) || mentions_var(rhs, var),
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } | ExprKind::Transpose(expr) | ExprKind::Load(expr) | ExprKind::Decode(expr) => mentions_var(expr, var),
        ExprKind::Tuple(items) | ExprKind::Math { args: items, .. } | ExprKind::Call { args: items, .. } | ExprKind::Intrinsic { args: items, .. } => items.iter().any(|i| mentions_var(i, var)),
        ExprKind::ExtentOf { base, .. } | ExprKind::Geometry { base, .. } | ExprKind::Accessor { base, .. } | ExprKind::Reshape { base, .. } | ExprKind::Field { base, .. } => mentions_var(base, var),
        _ => false,
    }
}

/// The same runtime value, wherever it was written.
fn same_value(a: &Expr, b: &Expr) -> bool {
    match (&a.kind, &b.kind) {
        (ExprKind::Int(x), ExprKind::Int(y)) => x == y,
        (ExprKind::Var(x), ExprKind::Var(y)) => x == y,
        (ExprKind::ShapeParam(x), ExprKind::ShapeParam(y)) => x == y,
        (ExprKind::Binary { op: o, lhs: l, rhs: r }, ExprKind::Binary { op: p, lhs: m, rhs: s }) => o == p && same_value(l, m) && same_value(r, s),
        (ExprKind::Cast { dtype: d, expr: x }, ExprKind::Cast { dtype: e, expr: y }) => d == e && same_value(x, y),
        (ExprKind::Index { base: x, indices: i }, ExprKind::Index { base: y, indices: j }) => {
            same_value(x, y)
                && i.len() == j.len()
                && i.iter().zip(j).all(|(p, q)| match (p, q) {
                    (Index::Point(p), Index::Point(q)) => same_value(p, q),
                    (Index::Coord(p), Index::Coord(q)) => p == q,
                    (Index::Slice(p), Index::Slice(q)) => p == q,
                    _ => false,
                })
        }
        _ => a.sym.is_some() && a.sym == b.sym,
    }
}

fn same_bound(a: &Option<Expr>, b: &Option<Expr>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => same_value(a, b),
        _ => false,
    }
}

/// The width of `start:start + c` when `start` is a runtime value.
fn static_width(start: &Option<Expr>, end: &Option<Expr>) -> Option<Sym> {
    let (Some(start), Some(Expr { kind: ExprKind::Binary { op: BinaryOp::Add, lhs, rhs }, .. })) = (start, end) else { return None };
    let width = if same_value(lhs, start) {
        rhs.sym.clone()
    } else if same_value(rhs, start) {
        lhs.sym.clone()
    } else {
        None
    };
    width.filter(|w| w.as_constant().is_some_and(|c| c >= 0))
}

/// Element dtype a dense operand contributes to arithmetic; parameters read as `f32`.
fn dense_dtype(elem: &Elem) -> Option<DType> {
    match elem {
        Elem::Dtype(d) => Some(*d),
        Elem::Param(_) => Some(DType::F32),
        Elem::Repr(_) => None,
    }
}

impl<'a> Checker<'a> {
    pub fn expr(&mut self, e: &ast::Expr, expected: Option<&Ty>) -> Option<Expr> {
        self.expr_inner(e, expected, false)
    }

    pub fn scalar(&self, kind: ExprKind, dtype: DType, sym: Option<Sym>, span: Span) -> Expr {
        Expr { kind, ty: Ty::Scalar(dtype), sym, partial: false, span }
    }

    pub fn expr_inner(&mut self, e: &ast::Expr, expected: Option<&Ty>, allow_unassigned: bool) -> Option<Expr> {
        let span = e.span;
        // A literal adopts the scalar dtype (or tile element dtype) its context supplies.
        let context = expected.and_then(|t| t.scalar_dtype().or_else(|| t.shaped().and_then(|s| dense_dtype(&s.elem))));
        match &e.kind {
            A::Int(v) => {
                let dtype = context.filter(|d| d.is_numeric()).unwrap_or(DType::I32);
                let limit = if dtype == DType::U32 { u64::from(u32::MAX) } else { i32::MAX as u64 };
                if dtype.is_int() && *v > limit {
                    self.error(span, format!("integer literal does not fit {}", dtype.name()));
                    return None;
                }
                Some(if dtype.is_float() { self.scalar(ExprKind::Float(*v as f64), dtype, None, span) } else { self.scalar(ExprKind::Int(*v as i64), dtype, Some(Sym::constant(*v as i64)), span) })
            }
            A::Float(v) => Some(self.scalar(ExprKind::Float(*v), context.filter(|d| d.is_float()).unwrap_or(DType::F32), None, span)),
            A::Inf => Some(self.scalar(ExprKind::Float(f64::INFINITY), context.filter(|d| d.is_float()).unwrap_or(DType::F32), None, span)),
            A::Bool(b) => Some(self.scalar(ExprKind::Bool(*b), DType::Bool, None, span)),
            A::Name(n) => self.name(n, allow_unassigned),
            A::Tuple(items) => {
                let hints: Vec<Option<&Ty>> = match expected {
                    Some(Ty::Tuple(tys)) if tys.len() == items.len() => tys.iter().map(Some).collect(),
                    _ => vec![None; items.len()],
                };
                let mut out = Vec::new();
                for (item, hint) in items.iter().zip(hints) {
                    let item = self.expr(item, hint)?;
                    if item.ty == Ty::Void {
                        self.error(item.span, "`void` is not a tuple component");
                        return None;
                    }
                    out.push(item);
                }
                let ty = Ty::Tuple(out.iter().map(|e| e.ty.clone()).collect());
                let partial = out.iter().any(|e| e.partial);
                Some(Expr { kind: ExprKind::Tuple(out), ty, sym: None, partial, span })
            }
            A::Range { .. } => {
                self.error(span, "a domain `lo..hi` is written as the source of a region or a `for` loop; it is not data");
                None
            }
            A::Tile { shape, elem } => self.tile_alloc(shape, elem, span),
            A::Call { callee, bindings, args } => self.call(callee, bindings, args, expected, span),
            A::Index { base, indices } => {
                let base = self.expr_inner(base, None, allow_unassigned)?;
                self.index(base, indices, span)
            }
            A::Attr { base, name } => {
                let base = self.expr(base, None)?;
                self.attr(base, name, span)
            }
            A::Unary { op, expr } => self.unary(*op, expr, expected, span),
            A::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, expected, span),
            A::Region(_) => {
                self.error(span, "a result-producing region appears only as the value of `let`/`var`, `yield` or `return`");
                None
            }
        }
    }

    fn name(&mut self, n: &ast::Ident, allow_unassigned: bool) -> Option<Expr> {
        if let Some(id) = self.lookup(&n.name) {
            if self.unassigned.contains(&id) && !allow_unassigned {
                self.error(n.span, format!("`{}` is read before every element is assigned; an uninitialized tile cannot be read, yielded or published", n.name));
                return None;
            }
            let mut ty = self.vars[id].ty.clone();
            match (&self.vars[id].kind, &ty) {
                (_, Ty::Coord(_)) => {
                    if !self.target_form(n.span, "arithmetic on a tile coordinate", None) {
                        return None;
                    }
                    ty = Ty::Scalar(DType::I32);
                }
                // A coordinate over a semantic axis is an ordinary index value; observing it
                // as a number observes that axis's extent.
                // Target code computes with tile coordinates under geometry authority.
                (VarKind::Coordinate, Ty::Index(bound)) => {
                    let bound = bound.clone();
                    if self.geometry_authority() {
                        self.requires_target = self.target.clone();
                    } else {
                        self.numeric_use(&bound);
                    }
                }
                _ => {}
            }
            let sym = self.atoms.get(&id).map(|a| Sym::atom(a.clone())).or_else(|| self.scalar_symbols.get(&id).cloned());
            if !allow_unassigned {
                self.reads.push(self.view_roots.get(&id).copied().unwrap_or(id));
            }
            // A `let` of a view freezes the descriptor, not the data it borrows.
            if let (Some(root), Some(bound)) = (self.view_roots.get(&id).copied(), self.view_bound.get(&id).copied()) {
                if !allow_unassigned && self.mutated[bound..].contains(&root) {
                    let root = self.vars[root].name.clone();
                    self.error(n.span, format!("view `{}` borrows `{root}`, which was written after the view was bound; a `let` of a view is not a snapshot: take one with `load`, or select the view again after the write", n.name));
                    return None;
                }
            }
            return Some(Expr { kind: ExprKind::Var(id), ty, sym, partial: self.vars[id].partial, span: n.span });
        }
        if self.sig.shape_params.contains(&n.name) {
            let sym = Sym::param(&n.name);
            self.numeric_use(&sym);
            return Some(self.scalar(ExprKind::ShapeParam(n.name.clone()), DType::I32, Some(sym), n.span));
        }
        if !self.poisoned.contains(&n.name) {
            self.error(n.span, format!("`{}` is not declared", n.name));
        }
        None
    }

    /// A writable place: evaluated without reading it.
    pub fn place(&mut self, e: &ast::Expr) -> Option<Expr> {
        match &e.kind {
            A::Index { base, indices } => {
                let base = self.place(base)?;
                self.index(base, indices, e.span)
            }
            _ => self.expr_inner(e, None, true),
        }
    }

    fn tile_alloc(&mut self, shape: &[ast::Expr], elem: &ast::Ident, span: Span) -> Option<Expr> {
        if shape.is_empty() {
            self.error(span, "a tile needs a shape");
            return None;
        }
        let mut axes = Vec::new();
        for dim in shape {
            // A bare slice or shape parameter in type position is an axis identity, not a number.
            if let A::Name(n) = &dim.kind {
                match self.lookup(&n.name).map(|id| self.vars[id].ty.clone()) {
                    Some(Ty::Slice(slice)) => {
                        axes.push(Extent::Structural(slice));
                        continue;
                    }
                    None if self.sig.shape_params.contains(&n.name) => {
                        axes.push(Extent::Semantic(Sym::param(&n.name)));
                        continue;
                    }
                    _ => {}
                }
            }
            let d = self.expr(dim, Some(&Ty::Scalar(DType::I32)))?;
            let Some(sym) = d.sym.clone().filter(|_| d.ty.scalar_dtype() == Some(DType::I32)) else {
                self.error(d.span, "a tile extent is a slice or a symbolic integer expression");
                return None;
            };
            self.require_nonneg(&sym, d.span, "tile extent may be negative");
            axes.push(Extent::Semantic(sym));
        }
        let elem = if let Some(d) = DType::from_name(&elem.name) {
            Elem::Dtype(d)
        } else if self.sig.elem_params.contains(&elem.name) {
            Elem::Param(elem.name.clone())
        } else {
            self.error(elem.span, format!("`{}` is not a dtype or an element parameter of this declaration; encoded tiles are produced by `load`", elem.name));
            return None;
        };
        Some(Expr { kind: ExprKind::TileAlloc, ty: Ty::Tile(Shaped::new(axes, elem)), sym: None, partial: false, span })
    }

    // ---- operators ----

    /// Operands of an elementwise operation: scalars and dense tiles over identical axes.
    /// Returns the common tile shape (if any operand is a tile) and each operand's dtype.
    pub fn broadcast(&mut self, operands: &[&Expr], what: &str, span: Span) -> Option<(Option<Shaped>, Vec<DType>)> {
        let mut shape: Option<Shaped> = None;
        let mut dtypes = Vec::new();
        for operand in operands {
            match &operand.ty {
                Ty::Tile(s) => {
                    let Some(d) = dense_dtype(&s.elem) else {
                        self.error(operand.span, format!("{what} is not defined on encoded `{}` storage; decode it with `f32(v)` or `decode(v)`", s.elem));
                        return None;
                    };
                    match &shape {
                        Some(first) if !self.same_axes(first, s) => {
                            let first = Ty::Tile(first.clone());
                            self.error(span, format!("{what} is elementwise over identical axes: {first} vs {}; equal widths of unrelated slices establish nothing", operand.ty));
                            return None;
                        }
                        Some(_) => {}
                        None => shape = Some(s.clone()),
                    }
                    dtypes.push(d);
                }
                Ty::View(_) | Ty::Tensor(_) => {
                    self.error(operand.span, format!("{what} consumes tile values; a {} is borrowed storage: read it with `load(v)` or a cast such as `f32(v)`", operand.ty));
                    return None;
                }
                Ty::Slice(_) | Ty::Coord(_) | Ty::Domain => {
                    self.error(operand.span, format!("{what} on a slice: slices are opaque geometry with no numeric, ordering, equality or identity-observation operator"));
                    return None;
                }
                Ty::Result(_) => {
                    self.error(operand.span, format!("{what} on a region result: select a member with `results[p]` inside a traversal of the same result"));
                    return None;
                }
                other => match other.scalar_dtype() {
                    Some(d) => dtypes.push(d),
                    None => {
                        self.error(operand.span, format!("{what} is not defined on {other}"));
                        return None;
                    }
                },
            }
        }
        for operand in operands {
            self.forbid_partial(operand, "an arithmetic operand");
        }
        Some((shape, dtypes))
    }

    pub fn elementwise(&self, shape: Option<Shaped>, dtype: DType) -> Ty {
        match shape {
            Some(s) => Ty::Tile(Shaped::new(s.axes, Elem::Dtype(dtype))),
            None => Ty::Scalar(dtype),
        }
    }

    fn unary(&mut self, op: UnaryOp, inner: &ast::Expr, expected: Option<&Ty>, span: Span) -> Option<Expr> {
        let inner = self.expr(inner, expected)?;
        let (shape, dtypes) = self.broadcast(&[&inner], &format!("unary `{}`", op.text().trim()), span)?;
        let d = dtypes[0];
        let ok = match op {
            UnaryOp::Neg => d.is_numeric(),
            UnaryOp::Not => d == DType::Bool,
            UnaryOp::BitNot => d.is_int(),
        };
        if !ok {
            self.error(span, format!("unary `{}` is not defined on {}", op.text().trim(), d.name()));
            return None;
        }
        if let (UnaryOp::Neg, ExprKind::Float(v), None) = (op, &inner.kind, &shape) {
            return Some(self.scalar(ExprKind::Float(-*v), d, None, span));
        }
        let sym = match (op, &inner.sym) {
            (UnaryOp::Neg, Some(s)) => Some(s.neg()),
            _ => None,
        };
        let ty = self.elementwise(shape, d);
        Some(Expr { partial: inner.partial && self.partial_free(), kind: ExprKind::Unary { op, expr: Box::new(inner) }, ty, sym, span })
    }

    fn binary(&mut self, op: BinaryOp, lhs: &ast::Expr, rhs: &ast::Expr, expected: Option<&Ty>, span: Span) -> Option<Expr> {
        let is_cmp = matches!(op, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge);
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let hint = if is_cmp || is_logic { None } else { expected };
        let l0 = self.expr(lhs, hint)?;
        let r_hint = if is_logic || is_shift { None } else { Some(l0.ty.clone()) };
        let r = self.expr(rhs, r_hint.as_ref().or(hint))?;
        // A literal on the left adopts the right operand's dtype.
        let l = if matches!(lhs.kind, A::Int(_) | A::Float(_)) && !is_shift { self.expr(lhs, Some(&r.ty))? } else { l0 };
        self.binary_exprs(op, l, r, span)
    }

    pub fn binary_exprs(&mut self, op: BinaryOp, l: Expr, r: Expr, span: Span) -> Option<Expr> {
        let is_cmp = matches!(op, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge);
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let is_bit = matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor);
        let (shape, dtypes) = self.broadcast(&[&l, &r], &format!("`{}`", op.text()), span)?;
        let (a, b) = (dtypes[0], dtypes[1]);
        // Outside an admitted context a partial operand was just reported.
        let partial = (l.partial || r.partial) && self.partial_free();
        let dtype = if is_logic {
            if a != DType::Bool || b != DType::Bool {
                self.error(span, format!("`{}` needs bool operands, found {} and {}", op.text(), a.name(), b.name()));
                return None;
            }
            DType::Bool
        } else if is_shift {
            // A shift amount is any integer; the result has the shifted operand's type.
            if !a.is_int() || !b.is_int() {
                self.error(span, format!("`{}` needs integer operands, found {} and {}", op.text(), a.name(), b.name()));
                return None;
            }
            if r.sym.as_ref().and_then(Sym::as_constant).is_some_and(|n| !(0..32).contains(&n)) {
                self.error(r.span, "integer shift count must be in 0..32");
                return None;
            }
            a
        } else {
            let Some(d) = DType::promote(a, b) else {
                self.error(span, format!("`{}` between {} and {} needs an explicit cast", op.text(), a.name(), b.name()));
                return None;
            };
            if is_bit && !d.is_int() {
                self.error(span, format!("`{}` needs integer operands, found {}", op.text(), d.name()));
                return None;
            }
            if !d.is_numeric() && !(d == DType::Bool && matches!(op, BinaryOp::Eq | BinaryOp::Ne)) {
                self.error(span, format!("`{}` is not defined on {}", op.text(), d.name()));
                return None;
            }
            d
        };
        let sym = match (&l.sym, &r.sym) {
            (Some(x), Some(y)) if shape.is_none() && dtype.is_int() && !is_cmp => match op {
                BinaryOp::Add => Some(x.add(y)),
                BinaryOp::Sub => Some(x.sub(y)),
                BinaryOp::Mul => Some(x.mul(y)),
                BinaryOp::Div | BinaryOp::Rem => {
                    if !self.prover().nonneg(&y.sub(&Sym::constant(1))) {
                        self.error(r.span, format!("divisor `{y}` is not provably positive"));
                        return None;
                    }
                    Some(if op == BinaryOp::Div { x.quot(y) } else { x.rem(y) })
                }
                BinaryOp::Shl => y.as_constant().and_then(|c| {
                    let product = x.scale(1i64 << c);
                    let (minimum, maximum) = if a == DType::U32 { (0, i64::from(u32::MAX)) } else { (i64::from(i32::MIN), i64::from(i32::MAX)) };
                    (self.prover().le(&Sym::constant(minimum), &product) && self.prover().le(&product, &Sym::constant(maximum))).then_some(product)
                }),
                BinaryOp::Shr => y.as_constant().map(|c| x.quot(&Sym::constant(1 << c))),
                _ => None,
            },
            _ => None,
        };
        let ty = self.elementwise(shape, if is_cmp { DType::Bool } else { dtype });
        Some(Expr { kind: ExprKind::Binary { op, lhs: Box::new(l), rhs: Box::new(r) }, ty, sym, partial, span })
    }

    // ---- indexing ----

    /// Accept a bounds need that is provable, or that depends on runtime data (then it is
    /// a runtime-checked obligation, as for every data-dependent index).
    fn require_in_bounds(&mut self, e: &Sym, span: Span, what: &str) {
        let data_dependent = e.params().iter().any(|p| !self.sig.shape_params.contains(p) && self.facts.upper_of(&Atom::Param(p.clone())).is_none());
        if !data_dependent || self.prover().nonneg(e) {
            self.require_nonneg(e, span, what);
        }
    }

    fn index(&mut self, base: Expr, indices: &[ast::Index], span: Span) -> Option<Expr> {
        let shaped = match &base.ty {
            Ty::Tensor(s) | Ty::View(s) | Ty::Tile(s) => s.clone(),
            Ty::Result(_) => return self.member(base, indices, span),
            Ty::Slice(_) => {
                self.error(span, "a slice cannot be indexed (`s[0]`): it is opaque geometry, not a source-visible array of coordinates");
                return None;
            }
            Ty::Tuple(_) => {
                self.error(span, "indexing does not distribute over a tuple; destructure it explicitly and index the components");
                return None;
            }
            Ty::Native(n) => {
                self.error(span, format!("native value `{}.{}` is not indexable; use the target's load/store operations", n.target, n.name));
                return None;
            }
            other => {
                self.error(span, format!("cannot index a {other}"));
                return None;
            }
        };
        if indices.len() > shaped.rank() {
            self.error(span, format!("{} indices for rank {}", indices.len(), shaped.rank()));
            return None;
        }
        let mut axes = Vec::new();
        let mut out = Vec::new();
        let mut packed_axis = shaped.packed_axis;
        let point = |packed_axis: &mut Option<usize>, removed: usize| match *packed_axis {
            Some(p) if p == removed => *packed_axis = None,
            Some(p) if p > removed => *packed_axis = Some(p - 1),
            _ => {}
        };
        for (axis, index) in indices.iter().enumerate() {
            let extent = shaped.axes[axis].clone();
            let position = axes.len();
            match index {
                ast::Index::Expr(e) => match self.structural_index(e, &extent)? {
                    Some((index, Some(kept))) => {
                        axes.push(kept);
                        out.push(index);
                    }
                    Some((index, None)) => {
                        point(&mut packed_axis, position);
                        out.push(index);
                    }
                    None => {
                        let i = self.expr(e, Some(&Ty::Scalar(DType::I32)))?;
                        if i.ty.scalar_dtype() != Some(DType::I32) {
                            self.error(i.span, format!("a point index is an `i32`, found {}", i.ty));
                            return None;
                        }
                        self.forbid_partial(&i, "an index");
                        match (&extent, &i.sym) {
                            // Target code addresses its structural axes under geometry authority.
                            (Extent::Structural(_), _) if self.geometry_authority() => {
                                self.target_form(i.span, "a point index on a structural axis", None);
                            }
                            (Extent::Structural(slice), _) => {
                                let name = self.slice_name(*slice);
                                self.error(i.span, format!("this axis is structural (slice `{name}`): a semantic coordinate indexes it only with proved membership (`for h in {name}:`), and a literal cannot mean an element of whichever tuned slice was received"));
                                return None;
                            }
                            (Extent::Semantic(extent), Some(s)) => {
                                self.require_in_bounds(s, i.span, "index may be negative");
                                self.require_in_bounds(&extent.sub(s).sub(&Sym::constant(1)), i.span, &format!("index may exceed extent `{extent}`"));
                            }
                            // Data-dependent points keep a runtime bounds obligation.
                            (Extent::Semantic(_), None) => {}
                        }
                        point(&mut packed_axis, position);
                        out.push(Index::Point(i));
                    }
                },
                ast::Index::Slice { start: None, end: None } => {
                    axes.push(extent);
                    out.push(Index::Range { start: None, end: None });
                }
                ast::Index::Slice { start, end } => {
                    let Extent::Semantic(extent) = extent else {
                        self.error(span, "a `lo:hi` range selects semantic coordinates; this axis is structural and is selected by its slice, a refinement, a member coordinate or `:`");
                        return None;
                    };
                    let mut bounds = [None, None];
                    for (slot, bound) in bounds.iter_mut().zip([start, end]) {
                        if let Some(b) = bound {
                            let b = self.expr(b, Some(&Ty::Scalar(DType::I32)))?;
                            if b.ty.scalar_dtype() != Some(DType::I32) {
                                self.error(b.span, format!("a range bound is an `i32`, found {}", b.ty));
                                return None;
                            }
                            self.forbid_partial(&b, "a range bound");
                            *slot = Some(b);
                        }
                    }
                    let [start, end] = bounds;
                    let lo = start.as_ref().map_or(Some(Sym::constant(0)), |b| b.sym.clone());
                    let hi = end.as_ref().map_or(Some(extent.clone()), |b| b.sym.clone());
                    let kept = match (lo, hi, static_width(&start, &end)) {
                        (Some(lo), Some(hi), _) => {
                            self.require_in_bounds(&lo, span, "range start may be negative");
                            self.require_in_bounds(&hi.sub(&lo), span, "range may be reversed");
                            self.require_in_bounds(&extent.sub(&hi), span, &format!("range end may exceed extent `{extent}`"));
                            hi.sub(&lo)
                        }
                        // A runtime start with a static width: `t:t + c`.
                        (_, _, Some(width)) => width,
                        // Runtime bounds: the view is clamped to the axis at run time. Equal
                        // windows over unchanged inputs share one extent.
                        _ => {
                            let known = self.dyn_slices.iter().find(|(s, e, parent, _)| same_bound(s, &start) && same_bound(e, &end) && *parent == extent).map(|(_, _, _, atom)| atom.clone());
                            let atom = match known {
                                Some(atom) => atom,
                                None => {
                                    let atom = self.fresh_atom("dyn");
                                    self.facts.set_range(atom.clone(), Sym::constant(0), extent.clone());
                                    self.dyn_slices.push((start.clone(), end.clone(), extent.clone(), atom.clone()));
                                    atom
                                }
                            };
                            Sym::atom(atom)
                        }
                    };
                    axes.push(Extent::Semantic(kept));
                    out.push(Index::Range { start, end });
                }
            }
        }
        axes.extend(shaped.axes[indices.len()..].iter().cloned());
        let ty = if axes.is_empty() {
            Ty::Scalar(shaped.elem.read_dtype().unwrap_or(DType::F32))
        } else {
            Ty::View(Shaped { axes, elem: shaped.elem.clone(), packed_axis })
        };
        Some(Expr { partial: base.partial, kind: ExprKind::Index { base: Box::new(base), indices: out }, ty, sym: None, span })
    }

    /// An index that is a bare name of a slice, a tile coordinate, a slice member, or a
    /// coordinate of this very axis. Returns the index and the axis it keeps, if any.
    /// `Some(None)` means the name is an ordinary scalar expression.
    #[allow(clippy::type_complexity)]
    fn structural_index(&mut self, e: &ast::Expr, extent: &Extent) -> Option<Option<(Index, Option<Extent>)>> {
        let A::Name(n) = &e.kind else { return Some(None) };
        let Some(id) = self.lookup(&n.name) else { return Some(None) };
        let unrelated = |c: &mut Checker, have: SliceId, axis: SliceId| {
            let (have, axis) = (c.slice_name(have), c.slice_name(axis));
            c.error(e.span, format!("slice `{have}` is unrelated to this axis (slice `{axis}`): two slices are interchangeable only if they are the same binder, an alias or an explicit refinement; equal tuned widths mean nothing"));
        };
        match (self.vars[id].kind.clone(), self.vars[id].ty.clone()) {
            (_, Ty::Slice(slice)) => match extent {
                Extent::Semantic(extent) => {
                    let (lo, hi) = self.root_domain(slice);
                    self.require_in_bounds(&lo, e.span, "slice domain may start below the axis");
                    self.require_in_bounds(&extent.sub(&hi), e.span, &format!("slice domain may exceed extent `{extent}`"));
                    Some(Some((Index::Slice(slice), Some(Extent::Structural(slice)))))
                }
                Extent::Structural(axis) if self.within(slice, *axis) => Some(Some((Index::Slice(slice), Some(Extent::Structural(slice))))),
                Extent::Structural(axis) => {
                    unrelated(self, slice, *axis);
                    None
                }
            },
            (_, Ty::Coord(slice)) => match extent {
                Extent::Structural(axis) if slice == *axis => Some(Some((Index::Coord(id), None))),
                Extent::Structural(axis) => {
                    unrelated(self, slice, *axis);
                    None
                }
                Extent::Semantic(_) => {
                    self.error(e.span, format!("tile coordinate `{}` indexes tiles and views sharing its structural axis; use `coord({})` for its semantic coordinate", n.name, n.name));
                    None
                }
            },
            (VarKind::SliceMember(slice), _) => match extent {
                Extent::Structural(axis) if self.within(slice, *axis) => {
                    let sym = self.atoms.get(&id).map(|a| Sym::atom(a.clone()));
                    let point = Expr { kind: ExprKind::Var(id), ty: self.vars[id].ty.clone(), sym, partial: false, span: e.span };
                    Some(Some((Index::Point(point), None)))
                }
                Extent::Structural(axis) => {
                    unrelated(self, slice, *axis);
                    None
                }
                Extent::Semantic(_) => Some(None),
            },
            // A coordinate of an axis with this very extent shares the axis identity.
            (VarKind::Coordinate, Ty::Index(bound)) => match extent {
                Extent::Semantic(extent) if self.prover().zero(&bound.sub(extent)) => {
                    let sym = self.atoms.get(&id).map(|a| Sym::atom(a.clone()));
                    let point = Expr { kind: ExprKind::Var(id), ty: Ty::Index(bound), sym, partial: false, span: e.span };
                    Some(Some((Index::Point(point), None)))
                }
                _ => Some(None),
            },
            _ => Some(None),
        }
    }

    /// `results[p]`: the member yielded for exactly this slice of the same origin.
    fn member(&mut self, result: Expr, indices: &[ast::Index], span: Span) -> Option<Expr> {
        let Ty::Result(ty) = result.ty.clone() else { return None };
        if indices.len() != ty.binders.len() {
            self.error(span, format!("this result was produced over {} binders; member selection names exactly that many slices", ty.binders.len()));
            return None;
        }
        let mut slices = Vec::new();
        for (index, original) in indices.iter().zip(&ty.binders) {
            let slice = match index {
                ast::Index::Expr(ast::Expr { kind: A::Name(n), .. }) => match self.lookup(&n.name).map(|id| self.vars[id].ty.clone()) {
                    Some(Ty::Slice(slice)) => Some(slice),
                    _ => None,
                },
                _ => None,
            };
            let Some(slice) = slice else {
                self.error(span, "a region result is selected by the slices of a traversal of the same result (`results[p]`); it cannot be indexed by number, counted or flattened");
                return None;
            };
            let same_origin = matches!(self.slice_parent(slice), crate::sir::SliceParent::Rebind(o) if o == original);
            if !same_origin {
                let name = self.slice_name(slice);
                self.error(span, format!("slice `{name}` does not come from this result's origin: a member is selected only by a slice rebound from the same result (`parallel/ordered [p] in results:`); an unrelated partition of equal width establishes no identity"));
                return None;
            }
            slices.push((slice, *original));
        }
        let map: Vec<(SliceId, SliceId)> = slices.iter().map(|(slice, original)| (*original, *slice)).collect();
        let member = self.rebind_ty(&ty.member, &map);
        let partial = self.result_partials.get(&ty.producer).is_none_or(|flags| flags.iter().any(|p| *p));
        Some(Expr { kind: ExprKind::Member { result: Box::new(result), slices: slices.into_iter().map(|(s, _)| s).collect() }, ty: member, sym: None, partial, span })
    }

    fn attr(&mut self, base: Expr, name: &ast::Ident, span: Span) -> Option<Expr> {
        match name.name.as_str() {
            "T" => {
                let Some(shaped) = base.ty.shaped().filter(|s| s.rank() == 2).cloned() else {
                    self.error(span, format!("`.T` transposes a rank-2 tile or view, found {}", base.ty));
                    return None;
                };
                if shaped.packed_axis.is_some() {
                    self.error(span, "a packed tile or view cannot be transposed; packets run along its last axis");
                    return None;
                }
                let t = Shaped { axes: vec![shaped.axes[1].clone(), shaped.axes[0].clone()], elem: shaped.elem, packed_axis: None };
                let ty = if matches!(base.ty, Ty::Tile(_)) { Ty::Tile(t) } else { Ty::View(t) };
                Some(Expr { partial: base.partial, kind: ExprKind::Transpose(Box::new(base)), ty, sym: None, span })
            }
            "words" | "scale" | "bias" | "coefficients" | "scale_factor" | "bias_factor" => {
                if !self.target_form(span, &format!("packed accessor `.{}`", name.name), None) {
                    return None;
                }
                let packed = match &base.ty {
                    Ty::Tile(s) | Ty::View(s) => match &s.elem {
                        Elem::Repr(r) => repr::lookup(r).map(|rep| (s.clone(), rep)),
                        _ => None,
                    },
                    _ => None,
                };
                let Some((s, rep)) = packed else {
                    self.error(span, format!("`.{}` needs a packed tile or view, found {}", name.name, base.ty));
                    return None;
                };
                let Some(Extent::Semantic(k)) = s.packed_axis.and_then(|axis| s.axes.get(axis)).cloned() else {
                    self.error(span, "this packed value has no semantic packet axis left to expose");
                    return None;
                };
                let (extent, dtype) = if name.name == "scale" || name.name == "bias" {
                    if name.name == "bias" && !rep.has_bias() {
                        self.error(span, format!("`{}` has no bias", rep.name));
                        return None;
                    }
                    (rep.groups_extent(&k), rep.coefficient_dtype())
                } else {
                    let Some(plane) = rep.plane(&name.name) else {
                        self.error(span, format!("`{}` has no physical plane `{}`", rep.name, name.name));
                        return None;
                    };
                    (plane.extent(&k), plane.dtype())
                };
                let mut axes = s.axes.clone();
                if let Some(axis) = s.packed_axis {
                    axes[axis] = Extent::Semantic(extent);
                }
                let plane = Shaped::new(axes, Elem::Dtype(dtype));
                let ty = if matches!(base.ty, Ty::Tile(_)) { Ty::Tile(plane) } else { Ty::View(plane) };
                Some(Expr { kind: ExprKind::Accessor { base: Box::new(base), name: name.name.clone() }, ty, sym: None, partial: false, span })
            }
            other => {
                self.error(name.span, format!("unknown attribute `{other}`"));
                None
            }
        }
    }

    /// `f32(e)`: scalar cast, or read-and-convert of a tile/view/tensor (yields a tile).
    pub fn cast(&mut self, dtype: DType, args: &[ast::Arg], span: Span) -> Option<Expr> {
        let [ast::Arg { name: None, value }] = args else {
            self.error(span, format!("`{}(x)` takes one argument", dtype.name()));
            return None;
        };
        let hint = Ty::Scalar(dtype);
        let inner = self.expr(value, matches!(value.kind, A::Int(_) | A::Float(_) | A::Inf | A::Unary { .. }).then_some(&hint))?;
        if let Some(s) = inner.ty.shaped() {
            let ok = match &s.elem {
                Elem::Repr(_) => dtype == DType::F32,
                Elem::Dtype(from) => from.is_numeric() && dtype.is_numeric(),
                Elem::Param(_) => dtype.is_float(),
            };
            if !ok {
                self.error(span, format!("cannot convert {} to `{}` elements; packed values decode with `f32(v)`", inner.ty, dtype.name()));
                return None;
            }
            let ty = Ty::Tile(Shaped::new(s.axes.clone(), Elem::Dtype(dtype)));
            return Some(Expr { partial: inner.partial, kind: ExprKind::Cast { dtype, expr: Box::new(inner) }, ty, sym: None, span });
        }
        let Some(from) = inner.ty.scalar_dtype() else {
            self.error(span, format!("cannot cast {} to {}; slices, results and native values have no scalar conversion", inner.ty, dtype.name()));
            return None;
        };
        if !from.is_numeric() && from != DType::Bool || !dtype.is_numeric() {
            self.error(span, format!("cannot cast {} to {}", from.name(), dtype.name()));
            return None;
        }
        let sym = if dtype.is_int() && from.is_int() { inner.sym.clone() } else { None };
        Some(Expr { partial: inner.partial, kind: ExprKind::Cast { dtype, expr: Box::new(inner) }, ty: Ty::Scalar(dtype), sym, span })
    }
}
