//! Expressions: literals, names, operators, indexing (points and ranges),
//! attributes, tensor allocation and casts — all typed through the intrinsic
//! registry and emitted as registry primitives.

use super::ir::{
    Expr as CheckedExpr, ExprKind as CheckedExprKind, Index as CheckedIndex, Literal, LocalId,
};
use super::{Checker, ValueClass};
use crate::expr::IntExpr;
use crate::intrinsics::IndexSlot as Slot;
use crate::intrinsics::{primitive, PrimitiveId};
use crate::span::Span;
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, UnaryOp};
use crate::types::{DType, Elem, TensorType, ValueType};

/// Whether `e` reads local `local`.
pub(crate) fn mentions_local(e: &CheckedExpr, local: LocalId) -> bool {
    if let CheckedExprKind::Local(v) = &e.kind {
        return *v == local;
    }
    let mut found = false;
    walk(e, &mut |expr: &CheckedExpr| {
        if let CheckedExprKind::Local(v) = &expr.kind {
            found |= *v == local;
        }
    });
    found
}

fn walk(e: &CheckedExpr, visit: &mut dyn FnMut(&CheckedExpr)) {
    visit(e);
    match &e.kind {
        CheckedExprKind::Primitive { operands, .. } => operands.iter().for_each(|o| walk(o, visit)),
        CheckedExprKind::Atomic {
            place,
            indices,
            value,
            ..
        } => {
            walk(place, visit);
            indices.iter().for_each(|index| walk(index, visit));
            walk(value, visit);
        }
        CheckedExprKind::PlaneView { base, .. } => walk(base, visit),
        CheckedExprKind::Intrinsic { args, .. } => args.iter().for_each(|a| walk(a, visit)),
        CheckedExprKind::Call { args, .. } => args.iter().for_each(|a| walk(a, visit)),
        CheckedExprKind::Literal(_) | CheckedExprKind::Dimension(_) | CheckedExprKind::Local(_) => {
        }
    }
}

/// The same runtime value, wherever it was written.
fn same_value(a: &CheckedExpr, b: &CheckedExpr) -> bool {
    match (&a.kind, &b.kind) {
        (CheckedExprKind::Literal(Literal::Int(x)), CheckedExprKind::Literal(Literal::Int(y))) => {
            x == y
        }
        (CheckedExprKind::Local(x), CheckedExprKind::Local(y)) => x == y,
        (CheckedExprKind::Dimension(x), CheckedExprKind::Dimension(y)) => x == y,
        (
            CheckedExprKind::Primitive {
                id: left_id,
                operands: left,
            },
            CheckedExprKind::Primitive {
                id: right_id,
                operands: right,
            },
        ) => {
            left_id == right_id
                && left.len() == right.len()
                && left.iter().zip(right).all(|(x, y)| same_value(x, y))
        }
        (CheckedExprKind::Literal(x), CheckedExprKind::Literal(y)) => x == y,
        _ => false,
    }
}

fn same_bound(a: &Option<CheckedExpr>, b: &Option<CheckedExpr>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => same_value(a, b),
        _ => false,
    }
}

/// The width of `start:start + c` when `start` is a runtime value.
fn static_width(
    arena: &crate::expr::ExprArena,
    start: &Option<CheckedExpr>,
    end: &Option<CheckedExpr>,
) -> Option<IntExpr> {
    let (
        Some(start),
        Some(CheckedExpr {
            kind:
                CheckedExprKind::Primitive {
                    id: PrimitiveId::Binary(BinaryOp::Add),
                    operands,
                },
            ..
        }),
    ) = (start, end)
    else {
        return None;
    };
    if operands.len() != 2 {
        return None;
    }
    let width = if same_value(&operands[0], start) {
        operands[1].sym
    } else if same_value(&operands[1], start) {
        operands[0].sym
    } else {
        None
    };
    width.filter(|w| super::prove::constant(arena, *w).is_some_and(|c| c >= 0))
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
    pub fn expr(&mut self, e: &ast::Expr, expected: Option<&ValueType>) -> Option<CheckedExpr> {
        self.expr_inner(e, expected, false)
    }

    pub fn scalar_expr(
        &mut self,
        kind: CheckedExprKind,
        dtype: DType,
        sym: Option<IntExpr>,
        span: Span,
    ) -> CheckedExpr {
        CheckedExpr::new(kind, ValueType::Scalar(dtype), sym, span)
    }

    pub fn expr_inner(
        &mut self,
        e: &ast::Expr,
        expected: Option<&ValueType>,
        allow_unassigned: bool,
    ) -> Option<CheckedExpr> {
        let span = e.span;
        // A literal adopts the scalar dtype (or tensor element dtype) its context supplies.
        let context = expected.and_then(|t| {
            t.scalar_dtype()
                .or_else(|| t.shaped().and_then(|s| dense_dtype(&s.elem)))
        });
        match &e.kind {
            A::Int(v) => {
                let dtype = context.filter(|d| d.is_numeric()).unwrap_or(DType::I32);
                let limit = if dtype == DType::U32 {
                    u64::from(u32::MAX)
                } else {
                    i32::MAX as u64
                };
                if dtype.is_int() && *v > limit {
                    self.error(
                        span,
                        format!("integer literal does not fit {}", dtype.name()),
                    );
                    return None;
                }
                let symbolic = self.arena.int(*v as i64);
                Some(if dtype.is_float() {
                    self.scalar_expr(
                        CheckedExprKind::Literal(Literal::Float(*v as f64)),
                        dtype,
                        None,
                        span,
                    )
                } else {
                    self.scalar_expr(
                        CheckedExprKind::Literal(Literal::Int(*v as i64)),
                        dtype,
                        Some(symbolic),
                        span,
                    )
                })
            }
            A::Float(v) => Some(self.scalar_expr(
                CheckedExprKind::Literal(Literal::Float(*v)),
                context.filter(|d| d.is_float()).unwrap_or(DType::F32),
                None,
                span,
            )),
            A::Inf => Some(self.scalar_expr(
                CheckedExprKind::Literal(Literal::Float(f64::INFINITY)),
                context.filter(|d| d.is_float()).unwrap_or(DType::F32),
                None,
                span,
            )),
            A::Bool(b) => Some(self.scalar_expr(
                CheckedExprKind::Literal(Literal::Bool(*b)),
                DType::Bool,
                None,
                span,
            )),
            A::Name(n) => self.name(n, allow_unassigned),
            A::Tuple(items) => {
                let hints: Vec<Option<&ValueType>> = match expected {
                    Some(ValueType::Tuple(tys)) if tys.len() == items.len() => {
                        tys.iter().map(Some).collect()
                    }
                    _ => vec![None; items.len()],
                };
                let mut out = Vec::new();
                for (item, hint) in items.iter().zip(hints) {
                    let item = self.expr(item, hint)?;
                    if item.ty.is_void() {
                        self.error(item.span, "`void` is not a tuple component");
                        return None;
                    }
                    out.push(item);
                }
                if out.is_empty() {
                    self.error(e.span, "an empty tuple is not a value");
                    return None;
                }
                if out.len() == 1 {
                    return out.pop();
                }
                let operands = out.clone();
                let tys: Vec<ValueType> = out.iter().map(|e| e.ty.clone()).collect();
                let ty = ValueType::Tuple(
                    crate::types::NonEmpty::new(tys)
                        .unwrap_or_else(|| panic!("nonempty checked tuple lost all components")),
                );
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::TuplePack,
                        operands,
                    },
                    ty,
                    None,
                    span,
                ))
            }
            A::Range { lo, hi } => {
                let lo = self.expr(lo, Some(&ValueType::Scalar(DType::I32)))?;
                let hi = self.expr(hi, Some(&ValueType::Scalar(DType::I32)))?;
                let (Some(lo_sym), Some(hi_sym)) = (lo.sym, hi.sym) else {
                    self.error(span, "range bounds must be symbolic integers");
                    return None;
                };
                let bound = match expected {
                    Some(ValueType::Range { bound }) => Some(*bound),
                    _ => None,
                }
                .unwrap_or(hi_sym);
                let hi_after_lo = self.arena.int_sub(hi_sym, lo_sym);
                let bound_after_hi = self.arena.int_sub(bound, hi_sym);
                if !super::prove::nonneg(&self.arena, &self.facts, lo_sym)
                    || !super::prove::nonneg(&self.arena, &self.facts, hi_after_lo)
                    || !super::prove::nonneg(&self.arena, &self.facts, bound_after_hi)
                {
                    self.error(span, "range must prove `0 <= start <= end <= bound`");
                    return None;
                }
                self.numeric_use(lo_sym);
                self.numeric_use(hi_sym);
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: PrimitiveId::RangeMake,
                        operands: vec![lo, hi],
                    },
                    ValueType::Range { bound },
                    None,
                    span,
                ))
            }
            A::Tensor { shape, elem } => self.tensor_alloc(shape, elem, span),
            A::Call {
                callee,
                bindings,
                args,
            } => self.call(callee, bindings, args, expected, span),
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
        }
    }

    fn name(&mut self, n: &ast::Ident, allow_unassigned: bool) -> Option<CheckedExpr> {
        if let Some(id) = self.lookup(&n.name) {
            if self.moved.contains(&id) {
                self.error(n.span, format!("use of moved owned tensor `{}`", n.name));
                return None;
            }
            if !self.borrows.contains_key(&id)
                && self
                    .borrows
                    .values()
                    .any(|(root, exclusive)| *root == id && *exclusive)
            {
                self.error(
                    n.span,
                    format!(
                        "cannot access `{}` while an exclusive tensor borrow is live",
                        n.name
                    ),
                );
                return None;
            }
            if self.unassigned.contains(&id) && !allow_unassigned {
                self.error(n.span, format!("`{}` is read before every element is assigned; an uninitialized tensor cannot be read or returned", n.name));
                return None;
            }
            let ty = self.locals[id.index()].ty.clone();
            let sym = self
                .symbols
                .get(&id)
                .map(|symbol| self.arena.int_symbol(*symbol))
                .or_else(|| self.scalar_symbols.get(&id).copied());
            if !allow_unassigned {
                self.reads
                    .push(self.view_roots.get(&id).copied().unwrap_or(id));
            }
            // A `let` of a view freezes the descriptor, not the data it borrows.
            if let (Some(root), Some(bound)) = (
                self.view_roots.get(&id).copied(),
                self.view_bound.get(&id).copied(),
            ) {
                if !allow_unassigned && self.mutated[bound..].contains(&root) {
                    let root = self.locals[root.index()].name.clone();
                    self.error(n.span, format!("view `{}` borrows `{root}`, which was written after the view was bound; a `let` of a view is not a snapshot: take one with `load`, or select the view again after the write", n.name));
                    return None;
                }
            }
            if let ValueType::Index { bound } = &ty {
                self.numeric_use(*bound);
            }
            return Some(CheckedExpr::new(
                CheckedExprKind::Local(id),
                ty,
                sym,
                n.span,
            ));
        }
        if self.sig.shape_params.contains(&n.name) {
            let ordinal = self
                .sig
                .shape_params
                .iter()
                .position(|name| name == &n.name)
                .expect("shape parameter lookup disagrees with contains");
            let sym = self.arena.int_symbol(self.sig.shape_symbols[ordinal]);
            self.numeric_use(sym);
            return Some(self.scalar_expr(
                CheckedExprKind::Dimension(
                    u32::try_from(ordinal).expect("definition has more than u32::MAX dimensions"),
                ),
                DType::I32,
                Some(sym),
                n.span,
            ));
        }
        if !self.poisoned.contains(&n.name) {
            self.error(n.span, format!("`{}` is not declared", n.name));
        }
        None
    }

    /// A writable place: `(root, indices, selected type)`. Evaluated without
    /// reading the storage.
    pub fn place(&mut self, e: &ast::Expr) -> Option<(LocalId, Vec<CheckedIndex>, ValueType)> {
        match &e.kind {
            A::Index { base, indices } => {
                let A::Name(name) = &base.kind else {
                    self.error(
                        base.span,
                        "element assignment indexes a tensor variable directly",
                    );
                    return None;
                };
                let Some(id) = self.place_name(name)? else {
                    return None;
                };
                let ty = self.locals[id.index()].ty.clone();
                self.select_indices(id, &ty, indices, e.span)
            }
            A::Name(name) => {
                let Some(id) = self.place_name(name)? else {
                    return None;
                };
                Some((id, Vec::new(), self.locals[id.index()].ty.clone()))
            }
            _ => {
                self.error(
                    e.span,
                    "an assignment target is `let mut` state, a tensor element, or a tuple of state",
                );
                None
            }
        }
    }

    /// Resolve the name a place designates, with the same moved/borrow access
    /// rules as a value read (a place does not read the storage).
    fn place_name(&mut self, name: &ast::Ident) -> Option<Option<LocalId>> {
        let Some(id) = self.lookup(&name.name) else {
            self.error(name.span, format!("`{}` is not declared", name.name));
            return None;
        };
        if self.moved.contains(&id) {
            self.error(
                name.span,
                format!("use of moved owned tensor `{}`", name.name),
            );
            return None;
        }
        if !self.borrows.contains_key(&id)
            && self
                .borrows
                .values()
                .any(|(root, exclusive)| *root == id && *exclusive)
        {
            self.error(
                name.span,
                format!(
                    "cannot access `{}` while an exclusive tensor borrow is live",
                    name.name
                ),
            );
            return None;
        }
        Some(Some(id))
    }

    /// Check the indices of one selection against `ty`, returning the root, the
    /// checked indices and the selected type.
    fn select_indices(
        &mut self,
        root: LocalId,
        ty: &ValueType,
        indices: &[ast::Index],
        span: Span,
    ) -> Option<(LocalId, Vec<CheckedIndex>, ValueType)> {
        let shaped = match ty {
            ValueType::Tensor(s) => s.clone(),
            ValueType::Tuple(_) => {
                self.error(span, "indexing does not distribute over a tuple; destructure it explicitly and index the components");
                return None;
            }
            ValueType::Opaque { name, .. } => {
                self.error(span, format!("backend-opaque value `{name}` is not indexable; use its capability intrinsics"));
                return None;
            }
            other => {
                self.error(span, format!("cannot index a {other}"));
                return None;
            }
        };
        if indices.len() > shaped.rank() {
            self.error(
                span,
                format!("{} indices for rank {}", indices.len(), shaped.rank()),
            );
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
            let extent = shaped.axes[axis];
            let extent_sym = extent;
            let position = axes.len();
            match index {
                ast::Index::Expr(e) => {
                    let i = self.expr(e, Some(&ValueType::Scalar(DType::I32)))?;
                    if i.ty.scalar_dtype() != Some(DType::I32) {
                        self.error(i.span, format!("a point index is an `i32`, found {}", i.ty));
                        return None;
                    }
                    if let Some(s) = i.sym {
                        let lower_check =
                            self.require_in_bounds(s, i.span, "index may be negative");
                        let after = self.arena.int_sub(extent_sym, s);
                        let one = self.arena.int(1);
                        let last = self.arena.int_sub(after, one);
                        let upper_check = self.require_in_bounds(
                            last,
                            i.span,
                            "index may exceed its axis extent",
                        );
                        point(&mut packed_axis, position);
                        out.push(CheckedIndex::Point {
                            value: i,
                            runtime_check: lower_check || upper_check,
                        });
                    } else {
                        point(&mut packed_axis, position);
                        out.push(CheckedIndex::Point {
                            value: i,
                            runtime_check: true,
                        });
                    }
                }
                ast::Index::Slice {
                    start: None,
                    end: None,
                } => {
                    axes.push(extent);
                    out.push(CheckedIndex::Range {
                        start: None,
                        end: None,
                        check_start: false,
                        check_order: false,
                        check_end: false,
                    });
                }
                ast::Index::Slice { start, end } => {
                    let mut bounds = [None, None];
                    for (slot, bound) in bounds.iter_mut().zip([start, end]) {
                        if let Some(b) = bound {
                            let b = self.expr(b, Some(&ValueType::Scalar(DType::I32)))?;
                            if b.ty.scalar_dtype() != Some(DType::I32) {
                                self.error(
                                    b.span,
                                    format!("a range bound is an `i32`, found {}", b.ty),
                                );
                                return None;
                            }
                            *slot = Some(b);
                        }
                    }
                    let [start, end] = bounds;
                    let zero = self.arena.int(0);
                    let lo = start.as_ref().map_or(Some(zero), |b| b.sym);
                    let hi = end.as_ref().map_or(Some(extent_sym), |b| b.sym);
                    let mut checks = (false, false, false);
                    let kept = match (lo, hi, static_width(&self.arena, &start, &end)) {
                        (Some(lo), Some(hi), _) => {
                            checks.0 =
                                self.require_in_bounds(lo, span, "range start may be negative");
                            let width = self.arena.int_sub(hi, lo);
                            checks.1 = self.require_in_bounds(width, span, "range may be reversed");
                            let tail = self.arena.int_sub(extent_sym, hi);
                            checks.2 = self.require_in_bounds(
                                tail,
                                span,
                                "range end may exceed its axis extent",
                            );
                            width
                        }
                        // A runtime start with a static width: `t:t + c`.
                        (_, _, Some(width)) => width,
                        // Runtime bounds: the realized length is a runtime value,
                        // never clamped; out-of-bounds selections fail at runtime.
                        _ => {
                            let known = self
                                .dyn_views
                                .iter()
                                .find(|(s, e, parent, _)| {
                                    same_bound(s, &start)
                                        && same_bound(e, &end)
                                        && *parent == extent_sym
                                })
                                .map(|(_, _, _, symbol)| *symbol);
                            let symbol = match known {
                                Some(symbol) => symbol,
                                None => {
                                    let symbol = self.fresh_symbol("dyn");
                                    let zero = self.arena.int(0);
                                    self.facts.set_range(symbol, zero, extent_sym);
                                    self.dyn_views.push((
                                        start.clone(),
                                        end.clone(),
                                        extent_sym,
                                        symbol,
                                    ));
                                    symbol
                                }
                            };
                            self.arena.int_symbol(symbol)
                        }
                    };
                    self.numeric_use(kept);
                    axes.push(kept);
                    if lo.is_none() || hi.is_none() {
                        checks = (start.is_some(), true, end.is_some());
                    }
                    out.push(CheckedIndex::Range {
                        start,
                        end,
                        check_start: checks.0,
                        check_order: checks.1,
                        check_end: checks.2,
                    });
                }
            }
        }
        axes.extend(shaped.axes[indices.len()..].iter().cloned());
        let selected = if axes.is_empty() {
            ValueType::Scalar(shaped.elem.read_dtype())
        } else {
            ValueType::Tensor(TensorType {
                axes,
                elem: shaped.elem,
                packed_axis,
            })
        };
        Some((root, out, selected))
    }

    /// `t[i, j:k]`: a point read or a view selection, one registry primitive.
    fn index(
        &mut self,
        base: CheckedExpr,
        indices: &[ast::Index],
        span: Span,
    ) -> Option<CheckedExpr> {
        if base.ty.shaped().is_none() {
            // select_indices reports the precise diagnostic for this base type.
            self.select_indices(LocalId::new(0), &base.ty, indices, span)?;
            return None;
        }
        let (_, checked, selected) =
            self.select_indices(LocalId::new(0), &base.ty, indices, span)?;
        let arity = checked.len();
        let all_points = checked
            .iter()
            .all(|i| matches!(i, CheckedIndex::Point { .. }));
        let element =
            indices.len() == base.ty.shaped().map(|s| s.rank()).unwrap_or(0) && all_points;
        let mut operands = vec![base];
        for index in &checked {
            match index {
                CheckedIndex::Point { value, .. } => operands.push(value.clone()),
                CheckedIndex::Range { start, end, .. } => {
                    operands.extend(start.iter().chain(end).cloned());
                }
            }
        }
        let (id, ty) = if element {
            let dtype = match selected {
                ValueType::Scalar(d) => d,
                _ => DType::F32,
            };
            (
                PrimitiveId::ElementRead {
                    arity: u32::try_from(arity).expect("index arity exceeds u32::MAX"),
                },
                ValueType::Scalar(dtype),
            )
        } else {
            let slots = checked
                .iter()
                .map(|i| match i {
                    CheckedIndex::Point { .. } => Slot::Point,
                    CheckedIndex::Range { start, end, .. } => Slot::Range {
                        start: start.is_some(),
                        end: end.is_some(),
                    },
                })
                .collect();
            (PrimitiveId::SliceView { indices: slots }, selected)
        };
        let signature = primitive(&id);
        if !signature.accepts(&[operands[0].ty.clone()]) {
            self.error(
                span,
                format!("`{}` is not defined on {}", id, operands[0].ty),
            );
            return None;
        }
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive { id, operands },
            ty,
            None,
            span,
        ))
    }

    fn tensor_alloc(
        &mut self,
        shape: &[ast::Expr],
        elem: &ast::Ident,
        span: Span,
    ) -> Option<CheckedExpr> {
        if shape.is_empty() {
            self.error(span, "a tensor needs a shape");
            return None;
        }
        let mut axes = Vec::new();
        let mut operands = Vec::new();
        for dim in shape {
            let d = self.expr(dim, Some(&ValueType::Scalar(DType::I32)))?;
            let Some(sym) = d.sym.filter(|_| d.ty.scalar_dtype() == Some(DType::I32)) else {
                self.error(d.span, "a tensor extent is a symbolic integer expression");
                return None;
            };
            self.require_nonneg(sym, d.span, "tensor extent may be negative");
            self.numeric_use(sym);
            axes.push(sym);
            operands.push(d);
        }
        let element = if let Some(d) = DType::from_name(&elem.name) {
            Elem::Dtype(d)
        } else if self.sig.elem_params.contains(&elem.name) {
            Elem::Param(elem.name.clone())
        } else {
            self.error(elem.span, format!("`{}` is not a dtype or an element parameter of this declaration; encoded tensors are produced by `load`", elem.name));
            return None;
        };
        let id = PrimitiveId::TensorAlloc;
        let ty = ValueType::Tensor(TensorType::new(axes, element));
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive { id, operands },
            ty,
            None,
            span,
        ))
    }

    // ---- operators ----

    /// Operands of an elementwise operation: scalars and dense computed values
    /// over identical axes. Returns the common axes (if any operand is a tile)
    /// and each operand's dtype.
    pub fn broadcast(
        &mut self,
        operands: &[&CheckedExpr],
        what: &str,
        span: Span,
    ) -> Option<(Option<Vec<IntExpr>>, Vec<DType>)> {
        let mut axes: Option<Vec<IntExpr>> = None;
        let mut dtypes = Vec::new();
        for operand in operands {
            match &operand.ty {
                ValueType::Tensor(s) => {
                    let Some(d) = dense_dtype(&s.elem) else {
                        self.error(operand.span, format!("{what} is not defined on encoded `{}` storage; decode it with `f32(v)` or `decode(v)`", s.elem));
                        return None;
                    };
                    match &axes {
                        Some(first)
                            if !self
                                .same_axes(&TensorType::new(first.clone(), Elem::Dtype(d)), s) =>
                        {
                            let first = ValueType::Tensor(TensorType::new(
                                first.clone(),
                                Elem::Dtype(dtypes[0]),
                            ));
                            self.error(
                                span,
                                format!(
                                    "{what} is elementwise over identical axes: {first} vs {}",
                                    operand.ty
                                ),
                            );
                            return None;
                        }
                        Some(_) => {}
                        None => axes = Some(s.axes.clone()),
                    }
                    dtypes.push(d);
                }
                ValueType::Range { .. } => {
                    self.error(
                        operand.span,
                        format!(
                            "{what} is not defined on a bounded range; ranges are consumed by `for`"
                        ),
                    );
                    return None;
                }
                ValueType::Opaque { .. } => {
                    self.error(
                        operand.span,
                        format!("{what} is not defined on a capability value"),
                    );
                    return None;
                }
                ValueType::Void => {
                    self.error(operand.span, format!("{what} is not defined on void"));
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
        Some((axes, dtypes))
    }

    /// the result type from the registry.
    pub(crate) fn elementwise_primitive(
        &mut self,
        id: PrimitiveId,
        operands: Vec<CheckedExpr>,
        axes: Option<Vec<IntExpr>>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let signature = primitive(&id);
        let tys: Vec<ValueType> = match &axes {
            Some(axes) => operands
                .iter()
                .map(|o| match &o.ty {
                    ValueType::Tensor(s) => {
                        ValueType::Tensor(TensorType::new(axes.clone(), s.elem.clone()))
                    }
                    other => other.clone(),
                })
                .collect(),
            None => operands.iter().map(|o| o.ty.clone()).collect(),
        };
        let ty = match signature.result_type(&tys) {
            Some(ty) => ty,
            None => {
                self.error(
                    span,
                    format!("`{}` is not defined on these operand types", id),
                );
                return None;
            }
        };
        let sym = None;
        Some(CheckedExpr::new(
            CheckedExprKind::Primitive { id, operands },
            ty,
            sym,
            span,
        ))
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        inner: &ast::Expr,
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let inner = self.expr(inner, expected)?;
        let (axes, _) =
            self.broadcast(&[&inner], &format!("unary `{}`", op.text().trim()), span)?;
        if let (UnaryOp::Neg, CheckedExprKind::Literal(Literal::Float(v))) = (op, &inner.kind) {
            let dtype = match inner.ty {
                ValueType::Scalar(d) => d,
                _ => DType::F32,
            };
            return Some(self.scalar_expr(
                CheckedExprKind::Literal(Literal::Float(-*v)),
                dtype,
                None,
                span,
            ));
        }
        let sym = match (op, &inner.sym) {
            (UnaryOp::Neg, Some(s)) => {
                let zero = self.arena.int(0);
                Some(self.arena.int_sub(zero, *s))
            }
            _ => None,
        };
        let mut out =
            self.elementwise_primitive(PrimitiveId::Unary(op), vec![inner.clone()], axes, span)?;
        if out.ty.scalar_dtype() == Some(DType::I32) {
            out.sym = sym;
        }
        Some(out)
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let is_cmp = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let hint = if is_cmp || is_logic { None } else { expected };
        let l0 = self.expr(lhs, hint)?;
        let r_hint = if is_logic || is_shift {
            None
        } else {
            Some(l0.ty.clone())
        };
        let r = self.expr(rhs, r_hint.as_ref().or(hint))?;
        // A literal on the left adopts the right operand's dtype.
        let l = if matches!(lhs.kind, A::Int(_) | A::Float(_)) && !is_shift {
            self.expr(lhs, Some(&r.ty))?
        } else {
            l0
        };
        self.binary_exprs(op, l, r, span)
    }

    pub fn binary_exprs(
        &mut self,
        op: BinaryOp,
        l: CheckedExpr,
        r: CheckedExpr,
        span: Span,
    ) -> Option<CheckedExpr> {
        let is_cmp = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let is_bit = matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor);
        let (axes, dtypes) = self.broadcast(&[&l, &r], &format!("`{}`", op.text()), span)?;
        let (a, b) = (dtypes[0], dtypes[1]);
        if is_logic {
            if a != DType::Bool || b != DType::Bool {
                self.error(
                    span,
                    format!(
                        "`{}` needs bool operands, found {} and {}",
                        op.text(),
                        a.name(),
                        b.name()
                    ),
                );
                return None;
            }
        } else if is_shift {
            if !a.is_int() || !b.is_int() {
                self.error(
                    span,
                    format!(
                        "`{}` needs integer operands, found {} and {}",
                        op.text(),
                        a.name(),
                        b.name()
                    ),
                );
                return None;
            }
            if r.sym
                .and_then(|value| super::prove::constant(&self.arena, value))
                .is_some_and(|n| !(0..32).contains(&n))
            {
                self.error(r.span, "integer shift count must be in 0..32");
                return None;
            }
        } else {
            let Some(d) = DType::promote(a, b) else {
                self.error(
                    span,
                    format!(
                        "`{}` between {} and {} needs an explicit cast",
                        op.text(),
                        a.name(),
                        b.name()
                    ),
                );
                return None;
            };
            if is_bit && !d.is_int() {
                self.error(
                    span,
                    format!("`{}` needs integer operands, found {}", op.text(), d.name()),
                );
                return None;
            }
            if !d.is_numeric() && !(d == DType::Bool && matches!(op, BinaryOp::Eq | BinaryOp::Ne)) {
                self.error(
                    span,
                    format!("`{}` is not defined on {}", op.text(), d.name()),
                );
                return None;
            }
        }
        let sym = match (l.sym, r.sym) {
            (Some(x), Some(y))
                if axes.is_none() && l.ty.scalar_dtype().is_some_and(|d| d.is_int()) && !is_cmp =>
            {
                match op {
                    BinaryOp::Add => Some(self.arena.int_add(x, y)),
                    BinaryOp::Sub => Some(self.arena.int_sub(x, y)),
                    BinaryOp::Mul => Some(self.arena.int_mul(x, y)),
                    BinaryOp::Div | BinaryOp::Rem => {
                        let one = self.arena.int(1);
                        let positive = self.arena.int_sub(y, one);
                        if !super::prove::nonneg(&self.arena, &self.facts, positive) {
                            self.error(r.span, "divisor is not provably positive");
                            return None;
                        }
                        Some(if op == BinaryOp::Div {
                            self.arena.int_div(x, y)
                        } else {
                            self.arena.int_rem(x, y)
                        })
                    }
                    BinaryOp::Shl => super::prove::constant(&self.arena, y).and_then(|c| {
                        let scale = 1i64.checked_shl(u32::try_from(c).ok()?)?;
                        let scale_expr = self.arena.int(scale);
                        let product = self.arena.int_mul(x, scale_expr);
                        let (minimum, maximum) = if a == DType::U32 {
                            (0, i64::from(u32::MAX))
                        } else {
                            (i64::from(i32::MIN), i64::from(i32::MAX))
                        };
                        let minimum = self.arena.int(minimum);
                        let maximum = self.arena.int(maximum);
                        (super::prove::le(&mut self.arena, &self.facts, minimum, product)
                            && super::prove::le(&mut self.arena, &self.facts, product, maximum))
                        .then_some(product)
                    }),
                    BinaryOp::Shr => super::prove::constant(&self.arena, y).and_then(|c| {
                        let divisor = 1i64.checked_shl(u32::try_from(c).ok()?)?;
                        let divisor = self.arena.int(divisor);
                        Some(self.arena.int_div(x, divisor))
                    }),
                    _ => None,
                }
            }
            _ => None,
        };
        let mut out =
            self.elementwise_primitive(PrimitiveId::Binary(op), vec![l, r], axes, span)?;
        if !is_cmp && !is_logic && out.ty.scalar_dtype() == Some(DType::I32) {
            out.sym = sym;
        }
        Some(out)
    }

    // ---- attributes ----

    fn attr(&mut self, base: CheckedExpr, name: &ast::Ident, span: Span) -> Option<CheckedExpr> {
        match name.name.as_str() {
            "T" => {
                let Some(shaped) = base.ty.shaped().filter(|s| s.rank() == 2).cloned() else {
                    self.error(
                        span,
                        format!("`.T` transposes a rank-2 tensor or view, found {}", base.ty),
                    );
                    return None;
                };
                if shaped.packed_axis.is_some() {
                    self.error(span, "a packed tensor or view cannot be transposed; packets run along its last axis");
                    return None;
                }
                let t = TensorType {
                    axes: vec![shaped.axes[1].clone(), shaped.axes[0].clone()],
                    elem: shaped.elem,
                    packed_axis: None,
                };
                let id = PrimitiveId::Transpose;
                if !primitive(&id).accepts(&[base.ty.clone()]) {
                    self.error(span, format!("`.T` is not defined on {}", base.ty));
                    return None;
                }
                Some(CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id,
                        operands: vec![base],
                    },
                    ValueType::Tensor(t),
                    None,
                    span,
                ))
            }
            "words" | "scale" | "bias" | "coefficients" | "scale_factor" | "bias_factor" => {
                if !self.target_form(span, &format!("packed accessor `.{}`", name.name), None) {
                    return None;
                }
                let class = self.class_of(&base);
                if !matches!(class, ValueClass::Borrowed | ValueClass::Computed) {
                    self.error(
                        span,
                        format!(
                            "`.`{} needs a packed view or tile, found {}",
                            name.name, base.ty
                        ),
                    );
                    return None;
                }
                let packed = match &base.ty {
                    ValueType::Tensor(s) => match &s.elem {
                        Elem::Repr(r) => match &crate::registry::representation_info(*r).kind {
                            crate::registry::RepresentationKind::Packed(layout) => {
                                Some((s.clone(), *r, layout))
                            }
                            crate::registry::RepresentationKind::Dense(_)
                            | crate::registry::RepresentationKind::External(_) => None,
                        },
                        _ => None,
                    },
                    _ => None,
                };
                let Some((s, representation, layout)) = packed else {
                    self.error(
                        span,
                        format!(
                            "`.`{} needs a packed view or tile, found {}",
                            name.name, base.ty
                        ),
                    );
                    return None;
                };
                let Some(k) = s.packed_axis.and_then(|axis| s.axes.get(axis)).copied() else {
                    self.error(
                        span,
                        "this packed value has no semantic packet axis left to expose",
                    );
                    return None;
                };
                let Some((plane_ordinal, plane)) = layout
                    .planes
                    .iter()
                    .enumerate()
                    .find(|(_, plane)| plane.name == name.name)
                else {
                    self.error(
                        span,
                        format!(
                            "`{}` has no physical plane `{}`",
                            crate::registry::representation_info(representation).name,
                            name.name
                        ),
                    );
                    return None;
                };
                let group = i64::from(plane.group);
                let group_expr = self.arena.int(group);
                let adjustment = self.arena.int(group - 1);
                let adjusted = self.arena.int_add(k, adjustment);
                let groups = self.arena.int_div(adjusted, group_expr);
                let fields = self.arena.int(i64::from(plane.fields));
                let entries = self.arena.int_mul(groups, fields);
                let extent = match plane.encoding {
                    crate::registry::PlaneEncoding::Dense(_) => entries,
                    crate::registry::PlaneEncoding::Packed { bits, .. } => {
                        let bits = self.arena.int(i64::from(bits));
                        let total_bits = self.arena.int_mul(entries, bits);
                        let thirty_one = self.arena.int(31);
                        let adjusted = self.arena.int_add(total_bits, thirty_one);
                        let thirty_two = self.arena.int(32);
                        self.arena.int_div(adjusted, thirty_two)
                    }
                    crate::registry::PlaneEncoding::FloatCode { format } => {
                        let bits = self.arena.int(i64::from(format.bits()));
                        let total_bits = self.arena.int_mul(entries, bits);
                        let seven = self.arena.int(7);
                        let adjusted = self.arena.int_add(total_bits, seven);
                        let eight = self.arena.int(8);
                        self.arena.int_div(adjusted, eight)
                    }
                };
                let dtype = plane.storage_dtype;
                let mut axes = s.axes.clone();
                if let Some(axis) = s.packed_axis {
                    axes[axis] = extent;
                }
                let plane = TensorType::new(axes, Elem::Dtype(dtype));
                Some(CheckedExpr::new(
                    CheckedExprKind::PlaneView {
                        base: Box::new(base),
                        plane: u32::try_from(plane_ordinal)
                            .expect("representation has more than u32::MAX planes"),
                    },
                    ValueType::Tensor(plane),
                    None,
                    span,
                ))
            }
            other => {
                self.error(name.span, format!("unknown attribute `{other}`"));
                None
            }
        }
    }

    /// Accept a bounds need that is provable, or that depends on runtime data
    /// (then it is a runtime-checked obligation, as for every data-dependent
    /// index; nothing is ever clamped).
    fn require_in_bounds(&mut self, e: IntExpr, span: Span, what: &str) -> bool {
        let data_dependent = super::prove::symbols(&self.arena, e).iter().any(|symbol| {
            !self.sig.shape_symbols.contains(symbol) && self.facts.upper_of(*symbol).is_none()
        });
        if super::prove::nonneg(&self.arena, &self.facts, e) {
            false
        } else if data_dependent {
            true
        } else {
            self.require_nonneg(e, span, what);
            false
        }
    }

    /// `f32(e)`: scalar cast, or read-and-convert of a tensor value (yields a tile).
    pub fn cast(&mut self, dtype: DType, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        let [ast::Arg { name: None, value }] = args else {
            self.error(span, format!("`{}(x)` takes one argument", dtype.name()));
            return None;
        };
        let hint = ValueType::Scalar(dtype);
        let inner = self.expr(
            value,
            matches!(
                value.kind,
                A::Int(_) | A::Float(_) | A::Inf | A::Unary { .. }
            )
            .then_some(&hint),
        )?;
        if let Some(s) = inner.ty.shaped() {
            let ok = match &s.elem {
                Elem::Repr(_) => dtype == DType::F32,
                Elem::Dtype(from) => from.is_numeric() && dtype.is_numeric(),
                Elem::Param(_) => dtype.is_float(),
            };
            if !ok {
                self.error(
                    span,
                    format!(
                        "cannot convert {} to `{}` elements; packed values decode with `f32(v)`",
                        inner.ty,
                        dtype.name()
                    ),
                );
                return None;
            }
            let axes = Some(s.axes.clone());
            return self.elementwise_primitive(PrimitiveId::Cast(dtype), vec![inner], axes, span);
        }
        let Some(from) = inner.ty.scalar_dtype() else {
            self.error(
                span,
                format!(
                    "cannot cast {} to {}; capability values have no scalar conversion",
                    inner.ty,
                    dtype.name()
                ),
            );
            return None;
        };
        if !from.is_numeric() && from != DType::Bool || !dtype.is_numeric() {
            self.error(
                span,
                format!("cannot cast {} to {}", from.name(), dtype.name()),
            );
            return None;
        }
        let sym = if dtype.is_int() && from.is_int() {
            inner.sym.clone()
        } else {
            None
        };
        let mut out =
            self.elementwise_primitive(PrimitiveId::Cast(dtype), vec![inner], None, span)?;
        out.sym = sym;
        Some(out)
    }
}
