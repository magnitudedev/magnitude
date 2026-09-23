//! Expressions: literals, names, operators, indexing (points and ranges),
//! attributes, tensor allocation and casts — all typed through the intrinsic
//! registry and emitted as registry primitives.

use super::ir::{Expr as CheckedExpr, ExprKind as CheckedExprKind, Index as CheckedIndex, LocalId};
use super::{Checker, ValueClass};
use crate::checked::DiagnosticRule;
use crate::expr::IntExpr;
use crate::intrinsics::IndexSlot as Slot;
use crate::intrinsics::{primitive, PrimitiveFailure, PrimitiveId};
use crate::reference_math::{self, ReferenceScalar, ScalarOp};
use crate::span::Span;
use crate::syntax::ast::{self, BinaryOp, ExprKind as A, UnaryOp};
use crate::types::{DType, Elem, TensorType, ValueType};

/// Whether `e` reads local `local`.
pub(crate) fn mentions_local(e: &CheckedExpr, local: LocalId) -> bool {
    let mut found = false;
    walk(e, &mut |expr: &CheckedExpr| {
        if let CheckedExprKind::Local(v) = &expr.kind {
            found |= *v == local;
        }
    });
    found
}

fn walk_index(index: &CheckedIndex, visit: &mut dyn FnMut(&CheckedExpr)) {
    match index {
        CheckedIndex::Point { value, .. } => walk(value, visit),
        CheckedIndex::Range { start, end, .. } => {
            start.iter().chain(end).for_each(|bound| walk(bound, visit))
        }
        CheckedIndex::Full => {}
    }
}

fn walk(e: &CheckedExpr, visit: &mut dyn FnMut(&CheckedExpr)) {
    visit(e);
    match &e.kind {
        CheckedExprKind::Primitive { operands, .. } => operands.iter().for_each(|o| walk(o, visit)),
        CheckedExprKind::Atomic { indices, value, .. } => {
            indices.iter().for_each(|index| walk_index(index, visit));
            walk(value, visit);
        }
        CheckedExprKind::PlaneView { base, .. } | CheckedExprKind::IndexPosition(base) => {
            walk(base, visit)
        }
        CheckedExprKind::Intrinsic { args, .. } => args.iter().for_each(|a| walk(a, visit)),
        CheckedExprKind::Call { call, args } => {
            for (_, value) in &call.seeds {
                walk(value, visit);
            }
            args.iter().for_each(|a| walk(a, visit));
        }
        CheckedExprKind::Literal(_) | CheckedExprKind::Dimension(_) | CheckedExprKind::Local(_) => {
        }
    }
}

/// The same runtime value, wherever it was written.
fn same_value(a: &CheckedExpr, b: &CheckedExpr) -> bool {
    match (&a.kind, &b.kind) {
        (CheckedExprKind::Local(x), CheckedExprKind::Local(y)) => x == y,
        (CheckedExprKind::Dimension(x), CheckedExprKind::Dimension(y)) => x == y,
        (
            CheckedExprKind::Primitive {
                id: left_id,
                operands: left,
                ..
            },
            CheckedExprKind::Primitive {
                id: right_id,
                operands: right,
                ..
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

fn quantity(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Integer | ValueType::Index { .. })
}

fn word(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Scalar(DType::I32 | DType::U32))
}

/// L24: the width `w` of `s : s + w` (or `s : w + s`) when `w` is a
/// nonnegative constant or a quantity. A quantity written into a word
/// addition keeps its exact value as the width.
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
                    ..
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
        &operands[1]
    } else if same_value(&operands[1], start) {
        &operands[0]
    } else {
        return None;
    };
    let width = match &width.kind {
        CheckedExprKind::Primitive {
            id: PrimitiveId::Cast(_),
            operands,
            ..
        } if quantity(&operands[0].ty) => &operands[0],
        _ => width,
    };
    let value = width.sym?;
    (quantity(&width.ty) || super::prove::constant(arena, value).is_some_and(|c| c >= 0))
        .then_some(value)
}

/// The elements of an operand at an elementwise position: a scalar's dtype,
/// a dense tensor's element dtype, `f32` for an element parameter.
fn operand_dtype(ty: &ValueType) -> Option<DType> {
    match ty {
        ValueType::Tensor(tensor) => tensor.elem.dense_dtype(),
        other => other.scalar_dtype(),
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

    /// A checked primitive, with the checker's proof about its failures.
    pub fn primitive_expr(
        &mut self,
        id: PrimitiveId,
        operands: Vec<CheckedExpr>,
        ty: ValueType,
        sym: Option<IntExpr>,
        span: Span,
    ) -> CheckedExpr {
        let failure = self.primitive_failure(&id, &operands);
        CheckedExpr::new(
            CheckedExprKind::Primitive {
                id,
                operands,
                failure,
            },
            ty,
            sym,
            span,
        )
    }

    /// The only constructor of `PrimitiveFailure`: `ProvedAbsent` when the
    /// primitive's scalar recipe has no failure outputs, or when its divisor
    /// or shift count is proved in range here.
    pub fn primitive_failure(&mut self, id: &PrimitiveId, operands: &[CheckedExpr]) -> PrimitiveFailure {
        let proved = |proof: bool| {
            if proof {
                PrimitiveFailure::ProvedAbsent
            } else {
                PrimitiveFailure::Possible
            }
        };
        match id {
            PrimitiveId::Binary(BinaryOp::Div | BinaryOp::Rem) => {
                let divisor = &operands[1];
                let Some(d) = divisor.sym else {
                    return PrimitiveFailure::Possible;
                };
                let one = self.arena.int(1);
                let positive = self.arena.int_sub(d, one);
                if super::prove::nonneg(&self.arena, &self.facts, positive) {
                    return PrimitiveFailure::ProvedAbsent;
                }
                proved(quantity(&divisor.ty) && self.facts.nonzero(&mut self.arena, d))
            }
            PrimitiveId::Binary(BinaryOp::Shl | BinaryOp::Shr) => {
                let Some(c) = operands[1].sym else {
                    return PrimitiveFailure::Possible;
                };
                let thirty_one = self.arena.int(31);
                let headroom = self.arena.int_sub(thirty_one, c);
                proved(
                    super::prove::nonneg(&self.arena, &self.facts, c)
                        && super::prove::nonneg(&self.arena, &self.facts, headroom),
                )
            }
            _ => {
                let Some(operation) = reference_math::scalar_operation(id) else {
                    return PrimitiveFailure::ProvedAbsent;
                };
                let Some(types) = operands
                    .iter()
                    .map(|operand| operand_dtype(&operand.ty))
                    .collect::<Option<Vec<_>>>()
                else {
                    return PrimitiveFailure::ProvedAbsent;
                };
                proved(reference_math::scalar_recipe(operation, &types).failures().is_empty())
            }
        }
    }

    pub fn expr_inner(
        &mut self,
        e: &ast::Expr,
        expected: Option<&ValueType>,
        place_context: bool,
    ) -> Option<CheckedExpr> {
        let span = e.span;
        // A literal adopts the scalar dtype (or tensor element dtype) its context supplies.
        let context = expected.and_then(operand_dtype);
        match &e.kind {
            A::Int(v) => self.integer_literal(i128::from(*v), expected, context, span),
            A::Float(_) | A::Inf => {
                let dtype = context.filter(|d| d.is_float()).unwrap_or(DType::F32);
                let value = match &e.kind {
                    A::Float(value) => *value,
                    A::Inf => f64::INFINITY,
                    _ => unreachable!(),
                };
                Some(self.scalar_expr(
                    CheckedExprKind::Literal(reference_math::float_literal(dtype, value)),
                    dtype,
                    None,
                    span,
                ))
            }
            A::Bool(b) => Some(self.scalar_expr(
                CheckedExprKind::Literal(ReferenceScalar::Bool(*b)),
                DType::Bool,
                None,
                span,
            )),
            A::Name(n) => self.name(n, place_context),
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
                        self.error(DiagnosticRule::Type, item.span, "`void` is not a tuple component");
                        return None;
                    }
                    out.push(item);
                }
                if out.is_empty() {
                    self.error(DiagnosticRule::Type, e.span, "an empty tuple is not a value");
                    return None;
                }
                if out.len() == 1 {
                    return out.pop();
                }
                let tys: Vec<ValueType> = out.iter().map(|e| e.ty.clone()).collect();
                let ty = ValueType::Tuple(
                    crate::types::NonEmpty::new(tys)
                        .unwrap_or_else(|| panic!("nonempty checked tuple lost all components")),
                );
                Some(self.primitive_expr(PrimitiveId::TuplePack, out, ty, None, span))
            }
            A::Range { lo, hi } => {
                let lo = self.expr(lo, Some(&ValueType::Integer))?;
                let hi = self.expr(hi, Some(&ValueType::Integer))?;
                for endpoint in [&lo, &hi] {
                    if !quantity(&endpoint.ty) && !word(&endpoint.ty) {
                        self.error(
                            DiagnosticRule::Type,
                            endpoint.span,
                            format!("a range endpoint is an integer, found {}", self.shown(&endpoint.ty)),
                        );
                        return None;
                    }
                }
                // L31: `0 <= lo <= hi` over the endpoints' one symbol each.
                let lo_sym = self.position_symbol(&lo)?;
                let hi_sym = self.position_symbol(&hi)?;
                let bound = match expected {
                    Some(ValueType::Range { bound }) => *bound,
                    _ => hi_sym,
                };
                let order = self.arena.int_sub(hi_sym, lo_sym);
                let tail = self.arena.int_sub(bound, hi_sym);
                let mut unproved = Vec::new();
                if !super::prove::nonneg(&self.arena, &self.facts, lo_sym) {
                    unproved.push(format!("0 <= {}", self.text(lo.span)));
                }
                if !super::prove::nonneg(&self.arena, &self.facts, order) {
                    unproved.push(format!("{} <= {}", self.text(lo.span), self.text(hi.span)));
                }
                if !super::prove::nonneg(&self.arena, &self.facts, tail) {
                    unproved.push(format!("{} <= {}", self.text(hi.span), self.render(bound)));
                }
                if !unproved.is_empty() {
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "a range needs `0 <= start <= end`; prove {} in scope, e.g. with `if`",
                            unproved
                                .iter()
                                .map(|conjunct| format!("`{conjunct}`"))
                                .collect::<Vec<_>>()
                                .join(" and ")
                        ),
                    );
                    return None;
                }
                Some(self.primitive_expr(
                    PrimitiveId::RangeMake,
                    vec![lo, hi],
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
                let base = self.expr_inner(base, None, place_context)?;
                self.index(base, indices, span)
            }
            A::Attr { base, name } => {
                let base = self.expr(base, None)?;
                self.attr(base, name, span)
            }
            A::Unary {
                op: UnaryOp::Neg,
                expr: operand,
            } if matches!(operand.kind, A::Int(_)) => {
                let A::Int(v) = operand.kind else {
                    unreachable!()
                };
                // L7: a negated integer literal is range-checked after folding.
                self.integer_literal(-i128::from(v), expected, context, span)
            }
            A::Unary { op, expr } => self.unary(*op, expr, expected, span),
            A::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, expected, span),
        }
    }

    fn integer_literal(
        &mut self,
        value: i128,
        expected: Option<&ValueType>,
        context: Option<DType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        if matches!(expected, Some(ValueType::Integer | ValueType::Index { .. })) {
            let constant = i64::try_from(value).ok()?;
            let value = self.arena.int(constant);
            return Some(self.primitive_expr(
                PrimitiveId::Symbolic(value),
                vec![],
                ValueType::Integer,
                Some(value),
                span,
            ));
        }
        let dtype = context.filter(|d| d.is_numeric()).unwrap_or(DType::I32);
        let fits = match dtype {
            DType::I32 => i128::from(i32::MIN) <= value && value <= i128::from(i32::MAX),
            DType::U32 => 0 <= value && value <= i128::from(u32::MAX),
            _ => true,
        };
        if !fits {
            self.error(
                DiagnosticRule::Type,
                span,
                format!("integer literal does not fit {}", dtype.name()),
            );
            return None;
        }
        let symbolic = dtype
            .is_int()
            .then(|| self.arena.int(i64::try_from(value).expect("word literal fits i64")));
        Some(self.scalar_expr(
            CheckedExprKind::Literal(reference_math::integer_literal(dtype, value)),
            dtype,
            symbolic,
            span,
        ))
    }

    fn name(&mut self, n: &ast::Ident, place_context: bool) -> Option<CheckedExpr> {
        if let Some(id) = self.lookup(&n.name) {
            if self.locals[id.index()].ownership.has_moved() {
                self.error(
                    DiagnosticRule::Ownership,
                    n.span,
                    format!("use of moved owned tensor `{}`", n.name),
                );
                return None;
            }
            if self.exclusive_borrow_blocks(id) {
                self.error(
                    DiagnosticRule::Ownership,
                    n.span,
                    format!(
                        "cannot access `{}` while an exclusive tensor borrow is live",
                        n.name
                    ),
                );
                return None;
            }
            let ty = self.locals[id.index()].ty.clone();
            let sym = self
                .symbols
                .get(&id)
                .map(|symbol| self.arena.int_symbol(*symbol))
                .or_else(|| self.scalar_symbols.get(&id).copied());
            let _ = place_context;
            return Some(CheckedExpr::new(
                CheckedExprKind::Local(id),
                ty,
                sym,
                n.span,
            ));
        }
        if let Some(ordinal) = self
            .sig
            .dimensions
            .iter()
            .position(|dimension| dimension.name == n.name)
        {
            let sym = self.arena.int_symbol(self.sig.dimensions[ordinal].symbol);
            return Some(CheckedExpr::new(
                CheckedExprKind::Dimension(
                    u32::try_from(ordinal).expect("definition has more than u32::MAX dimensions"),
                ),
                ValueType::Integer,
                Some(sym),
                n.span,
            ));
        }
        if !self.poisoned.contains(&n.name) {
            self.error(
                DiagnosticRule::Resolution,
                n.span,
                format!("`{}` is not declared", n.name),
            );
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
                        DiagnosticRule::Type,
                        base.span,
                        "element assignment indexes a tensor variable directly",
                    );
                    return None;
                };
                let id = self.place_name(name)?;
                let ty = self.locals[id.index()].ty.clone();
                self.select_indices(id, &ty, indices, e.span)
            }
            A::Name(name) => {
                let id = self.place_name(name)?;
                Some((id, Vec::new(), self.locals[id.index()].ty.clone()))
            }
            _ => {
                self.error(
                    DiagnosticRule::Type,
                    e.span,
                    "an assignment target is `let mut` state, a tensor element, or a tuple of state",
                );
                None
            }
        }
    }

    /// Resolve the name a place designates, with the same moved/borrow access
    /// rules as a value read (a place does not read the storage).
    fn place_name(&mut self, name: &ast::Ident) -> Option<LocalId> {
        let Some(id) = self.lookup(&name.name) else {
            if !self.poisoned.contains(&name.name) {
                self.error(
                    DiagnosticRule::Resolution,
                    name.span,
                    format!("`{}` is not declared", name.name),
                );
            }
            return None;
        };
        if self.locals[id.index()].ownership.has_moved() {
            self.error(
                DiagnosticRule::Ownership,
                name.span,
                format!("use of moved owned tensor `{}`", name.name),
            );
            return None;
        }
        if self.exclusive_borrow_blocks(id) {
            self.error(
                DiagnosticRule::Ownership,
                name.span,
                format!(
                    "cannot access `{}` while an exclusive tensor borrow is live",
                    name.name
                ),
            );
            return None;
        }
        Some(id)
    }

    /// An integer at a quantity position, with its value's one symbol.
    fn index_value(&mut self, e: &ast::Expr, what: &str) -> Option<CheckedExpr> {
        let i = self.expr(e, Some(&ValueType::Integer))?;
        if !quantity(&i.ty) && !word(&i.ty) {
            self.error(
                DiagnosticRule::Type,
                i.span,
                format!("{what} is an integer, found {}", self.shown(&i.ty)),
            );
            return None;
        }
        Some(i)
    }

    /// Check the indices of one selection against `ty`, returning the root, the
    /// checked indices (one per axis, `Full` for omitted trailing axes) and the
    /// selected type. A bound the checker does not prove is either rejected or,
    /// when it depends on runtime data, flagged for a runtime check; after a
    /// checked access the continuation assumes the bounds (L18).
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
                self.error(DiagnosticRule::Type, span, "indexing does not distribute over a tuple; destructure it explicitly and index the components");
                return None;
            }
            ValueType::Opaque { name, .. } => {
                self.error(DiagnosticRule::Type, span, format!("backend-opaque value `{name}` is not indexable; use its capability intrinsics"));
                return None;
            }
            other => {
                self.error(DiagnosticRule::Type, span, format!("cannot index a {}", self.shown(other)));
                return None;
            }
        };
        if indices.len() > shaped.rank() {
            self.error(
                DiagnosticRule::Type,
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
        let mut assumptions = Vec::new();
        for (axis, index) in indices.iter().enumerate() {
            let extent = shaped.axes[axis];
            let position = axes.len();
            match index {
                ast::Index::Expr(e) => {
                    let i = self.index_value(e, "a point index")?;
                    let Some(s) = i.sym else {
                        point(&mut packed_axis, position);
                        out.push(CheckedIndex::Point {
                            value: i,
                            runtime_check: true,
                        });
                        continue;
                    };
                    let lower_check = self.require_in_bounds(s, i.span, "index may be negative");
                    let after = self.arena.int_sub(extent, s);
                    let one = self.arena.int(1);
                    let last = self.arena.int_sub(after, one);
                    let upper_check =
                        self.require_in_bounds(last, i.span, "index may exceed its axis extent");
                    if lower_check || upper_check {
                        assumptions.extend([s, last]);
                    }
                    point(&mut packed_axis, position);
                    out.push(CheckedIndex::Point {
                        value: i,
                        runtime_check: lower_check || upper_check,
                    });
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
                        check_width: false,
                    });
                }
                ast::Index::Slice { start, end } => {
                    let mut bounds = [None, None];
                    for (slot, bound) in bounds.iter_mut().zip([start, end]) {
                        if let Some(b) = bound {
                            *slot = Some(self.index_value(b, "a range bound")?);
                        }
                    }
                    let [start, end] = bounds;
                    let zero = self.arena.int(0);
                    let lo = start.as_ref().map_or(Some(zero), |b| b.sym);
                    let hi = end.as_ref().map_or(Some(extent), |b| b.sym);
                    let width = static_width(&self.arena, &start, &end);
                    let (kept, checks) = match (lo, hi) {
                        (Some(lo), Some(hi)) => {
                            let check_start = self.require_in_bounds(lo, span, "range start may be negative");
                            let order = self.arena.int_sub(hi, lo);
                            let check_order = self.require_in_bounds(order, span, "range may be reversed");
                            let tail = self.arena.int_sub(extent, hi);
                            let check_end = self.require_in_bounds(
                                tail,
                                span,
                                "range end may exceed its axis extent",
                            );
                            if check_start || check_order || check_end {
                                assumptions.extend([lo, order, tail]);
                            }
                            // L24: `s : s + w` has length `w`; a word
                            // addition that may wrap keeps a width check.
                            let check_width = width.is_some_and(|w| {
                                let realized = self.arena.int_sub(order, w);
                                !super::prove::zero(&self.arena, &self.facts, realized)
                            });
                            (width.unwrap_or(order), (check_start, check_order, check_end, check_width))
                        }
                        (_, _) => {
                            let kept = match width {
                                // A runtime start with a static width: `t:t + w`.
                                Some(width) => width,
                                // Runtime bounds: the realized length is a
                                // runtime value, never clamped.
                                None => {
                                    let known = self
                                        .dyn_views
                                        .iter()
                                        .find(|(s, e, parent, _)| {
                                            same_bound(s, &start)
                                                && same_bound(e, &end)
                                                && *parent == extent
                                        })
                                        .map(|(_, _, _, symbol)| *symbol);
                                    let symbol = match known {
                                        Some(symbol) => symbol,
                                        None => {
                                            let symbol = self.fresh_data_symbol();
                                            let zero = self.arena.int(0);
                                            self.facts.set_range(symbol, zero, extent);
                                            self.dyn_views.push((
                                                start.clone(),
                                                end.clone(),
                                                extent,
                                                symbol,
                                            ));
                                            symbol
                                        }
                                    };
                                    self.arena.int_symbol(symbol)
                                }
                            };
                            (kept, (start.is_some(), true, end.is_some(), width.is_some()))
                        }
                    };
                    axes.push(kept);
                    out.push(CheckedIndex::Range {
                        start,
                        end,
                        check_start: checks.0,
                        check_order: checks.1,
                        check_end: checks.2,
                        check_width: checks.3,
                    });
                }
            }
        }
        for _ in indices.len()..shaped.rank() {
            out.push(CheckedIndex::Full);
        }
        axes.extend(shaped.axes[indices.len()..].iter().cloned());
        // L18: the continuation of a checked access assumes its bounds.
        for fact in assumptions {
            self.assume_nonneg(fact);
        }
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
        let (_, checked, selected) = self.select_indices(LocalId::new(0), &base.ty, indices, span)?;
        let element = checked
            .iter()
            .all(|i| matches!(i, CheckedIndex::Point { .. }));
        let base_element = base
            .ty
            .shaped()
            .expect("a selected base is a tensor")
            .elem
            .clone();
        let mut operands = vec![base];
        for index in &checked {
            match index {
                CheckedIndex::Point { value, .. } => operands.push(value.clone()),
                CheckedIndex::Range { start, end, .. } => {
                    operands.extend(start.iter().chain(end).cloned());
                }
                CheckedIndex::Full => {}
            }
        }
        let (id, ty) = if element {
            self.elements.decoded_read(&base_element);
            let checks = checked
                .iter()
                .map(|index| match index {
                    CheckedIndex::Point { runtime_check, .. } => *runtime_check,
                    CheckedIndex::Range { .. } | CheckedIndex::Full => {
                        unreachable!("an element read selects points")
                    }
                })
                .collect();
            (PrimitiveId::ElementRead { checks }, selected)
        } else {
            let slots = checked
                .iter()
                .map(|i| match i {
                    CheckedIndex::Point { runtime_check, .. } => Slot::Point {
                        check: *runtime_check,
                    },
                    CheckedIndex::Range {
                        start,
                        end,
                        check_start,
                        check_order,
                        check_end,
                        check_width,
                    } => Slot::Range {
                        start: start.is_some(),
                        end: end.is_some(),
                        check_start: *check_start,
                        check_order: *check_order,
                        check_end: *check_end,
                        check_width: *check_width,
                    },
                    CheckedIndex::Full => Slot::Full,
                })
                .collect();
            (PrimitiveId::SliceView { indices: slots }, selected)
        };
        let signature = primitive(&id);
        if !signature.accepts(&[operands[0].ty.clone()]) {
            self.error(
                DiagnosticRule::Type,
                span,
                format!("`{}` is not defined on {}", id, self.shown(&operands[0].ty)),
            );
            return None;
        }
        let sym = (matches!(id, PrimitiveId::ElementRead { .. }) && word(&ty)).then(|| {
            let symbol = self.fresh_data_symbol();
            self.facts.assume_type(&mut self.arena, symbol, &ty);
            self.arena.int_symbol(symbol)
        });
        Some(self.primitive_expr(id, operands, ty, sym, span))
    }

    fn tensor_alloc(
        &mut self,
        shape: &[ast::Expr],
        elem: &ast::Ident,
        span: Span,
    ) -> Option<CheckedExpr> {
        if shape.is_empty() {
            self.error(DiagnosticRule::Type, span, "a tensor needs a shape");
            return None;
        }
        let mut axes = Vec::new();
        let mut operands = Vec::new();
        for dim in shape {
            let d = self.index_value(dim, "a tensor extent")?;
            // L31: an extent position proves `w >= 0` over the word's one symbol.
            let sym = self.position_symbol(&d)?;
            if !super::prove::nonneg(&self.arena, &self.facts, sym) {
                let text = self.text(d.span).to_owned();
                self.error(
                    DiagnosticRule::Type,
                    d.span,
                    format!("tensor extent may be negative: cannot prove `{text} >= 0`"),
                );
                return None;
            }
            axes.push(sym);
            operands.push(d);
        }
        let element = if let Some(d) = DType::from_name(&elem.name) {
            Elem::Dtype(d)
        } else if self.sig.elem_params.contains(&elem.name) {
            Elem::Param(elem.name.clone())
        } else {
            self.error(
                DiagnosticRule::Resolution,
                elem.span,
                format!("`{}` is not a dtype or an element parameter of this declaration", elem.name),
            );
            return None;
        };
        self.elements.stored(&element);
        let ty = ValueType::Tensor(TensorType::new(axes, element));
        Some(self.primitive_expr(PrimitiveId::TensorAlloc, operands, ty, None, span))
    }

    /// The one symbol an integer value has at a quantity position (I-90).
    /// A value with no exact symbol has no provable range.
    pub fn position_symbol(&mut self, value: &CheckedExpr) -> Option<IntExpr> {
        match value.sym {
            Some(sym) => Some(sym),
            None => {
                let text = self.text(value.span).to_owned();
                self.error(
                    DiagnosticRule::Type,
                    value.span,
                    format!(
                        "`{text}` has no exact integer value here; bind it with `let` and prove its range in scope, e.g. with `if`"
                    ),
                );
                None
            }
        }
    }

    // ---- operators ----

    /// Operands of an elementwise operation: scalars and dense computed values
    /// over identical axes. Returns the common axes (if any operand is a tile)
    /// and each operand's dtype. An element-parameter tensor operand is read
    /// as its decoded `f32` value.
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
                    let Some(d) = s.elem.dense_dtype() else {
                        self.error(
                            DiagnosticRule::Type,
                            operand.span,
                            format!("{what} is not defined on encoded `{}` storage; decode it with `f32(v)`", s.elem),
                        );
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
                                DiagnosticRule::Type,
                                span,
                                format!(
                                    "{what} is elementwise over identical axes: {} vs {}",
                                    self.shown(&first),
                                    self.shown(&operand.ty)
                                ),
                            );
                            return None;
                        }
                        Some(_) => {}
                        None => axes = Some(s.axes.clone()),
                    }
                    let element = s.elem.clone();
                    self.elements.decoded_read(&element);
                    dtypes.push(d);
                }
                ValueType::Range { .. } => {
                    self.error(
                        DiagnosticRule::Type,
                        operand.span,
                        format!(
                            "{what} is not defined on a bounded range; ranges are consumed by `for`"
                        ),
                    );
                    return None;
                }
                ValueType::Opaque { .. } => {
                    self.error(
                        DiagnosticRule::Type,
                        operand.span,
                        format!("{what} is not defined on a capability value"),
                    );
                    return None;
                }
                ValueType::Void => {
                    self.error(DiagnosticRule::Type, operand.span, format!("{what} is not defined on void"));
                    return None;
                }
                other => match other.scalar_dtype() {
                    Some(d) => dtypes.push(d),
                    None => {
                        self.error(
                            DiagnosticRule::Type,
                            operand.span,
                            format!("{what} is not defined on {}", self.shown(other)),
                        );
                        return None;
                    }
                },
            }
        }
        Some((axes, dtypes))
    }

    /// L10, §2.3.6 (2): every operand whose elements differ from `target`,
    /// and every element-parameter tensor operand, is wrapped in an explicit
    /// `Cast(target)`, elementwise for a tensor. No later layer promotes.
    pub fn promote_operands(&mut self, operands: Vec<CheckedExpr>, target: DType) -> Vec<CheckedExpr> {
        operands
            .into_iter()
            .map(|operand| {
                let (differs, ty) = match &operand.ty {
                    ValueType::Tensor(tensor) => (
                        tensor.elem != Elem::Dtype(target),
                        ValueType::Tensor(TensorType::new(tensor.axes.clone(), Elem::Dtype(target))),
                    ),
                    ValueType::Scalar(dtype) => (*dtype != target, ValueType::Scalar(target)),
                    _ => (false, operand.ty.clone()),
                };
                if !differs {
                    return operand;
                }
                let span = operand.span;
                self.primitive_expr(PrimitiveId::Cast(target), vec![operand], ty, None, span)
            })
            .collect()
    }

    /// The result type from the registry.
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
                    DiagnosticRule::Type,
                    span,
                    format!("`{}` is not defined on these operand types", id),
                );
                return None;
            }
        };
        Some(self.primitive_expr(id, operands, ty, None, span))
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        inner: &ast::Expr,
        expected: Option<&ValueType>,
        span: Span,
    ) -> Option<CheckedExpr> {
        let inner = self.expr(inner, expected)?;
        // L28: one table of operand domains.
        let domain = crate::intrinsics::unary_operand_domain(op);
        let admitted = match &inner.ty {
            ValueType::Integer | ValueType::Index { .. } => domain.admits_quantity(),
            ValueType::Tensor(tensor) => tensor
                .elem
                .dense_dtype()
                .is_some_and(|dtype| domain.admits_dtype(dtype)),
            other => other.scalar_dtype().is_some_and(|dtype| domain.admits_dtype(dtype)),
        };
        if !admitted {
            let hint = if op == UnaryOp::BitNot && inner.ty.scalar_dtype() == Some(DType::Bool) {
                "; for a boolean write `not c`"
            } else {
                ""
            };
            self.error(
                DiagnosticRule::Type,
                span,
                format!(
                    "`{}` is not defined on {}{hint}",
                    op.text().trim(),
                    self.shown(&inner.ty)
                ),
            );
            return None;
        }
        if quantity(&inner.ty) {
            let Some(value) = inner.sym else {
                self.error(DiagnosticRule::Type, span, "quantity negation requires an exact integer value");
                return None;
            };
            let zero = self.arena.int(0);
            let result = self.arena.int_sub(zero, value);
            return Some(self.primitive_expr(
                PrimitiveId::Unary(op),
                vec![inner],
                ValueType::Integer,
                Some(result),
                span,
            ));
        }
        let (axes, _) =
            self.broadcast(&[&inner], &format!("unary `{}`", op.text().trim()), span)?;
        if let (UnaryOp::Neg, CheckedExprKind::Literal(value)) = (op, &inner.kind) {
            if value.dtype().is_float() {
                let recipe = reference_math::scalar_recipe(ScalarOp::Unary(op), &[value.dtype()]);
                return Some(
                    self.scalar_expr(
                        CheckedExprKind::Literal(
                            reference_math::evaluate(&recipe, &[*value])
                                .expect("floating negation is total"),
                        ),
                        value.dtype(),
                        None,
                        span,
                    ),
                );
            }
        }
        let sym = inner.sym.filter(|_| axes.is_none()).and_then(|value| {
            let dtype = inner.ty.scalar_dtype()?;
            dtype.is_int().then(|| self.arena.scalar_integer(
                ScalarOp::Unary(op),
                &[(dtype, value)],
            ))
        });
        // An element-parameter tensor operand reads as its decoded f32 value.
        let inner = match operand_dtype(&inner.ty) {
            Some(dtype) if matches!(&inner.ty, ValueType::Tensor(t) if matches!(t.elem, Elem::Param(_))) => {
                self.promote_operands(vec![inner], dtype).remove(0)
            }
            _ => inner,
        };
        let mut out =
            self.elementwise_primitive(PrimitiveId::Unary(op), vec![inner], axes, span)?;
        if out.ty.scalar_dtype().is_some_and(DType::is_int) {
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
        mut l: CheckedExpr,
        mut r: CheckedExpr,
        span: Span,
    ) -> Option<CheckedExpr> {
        let is_cmp = matches!(
            op,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        );
        if quantity(&l.ty) || quantity(&r.ty) {
            if let ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) = l.ty {
                r = self.quantity_to_word(r, dtype);
            } else if let ValueType::Scalar(dtype @ (DType::I32 | DType::U32)) = r.ty {
                l = self.quantity_to_word(l, dtype);
            } else {
                let (Some(a), Some(b)) = (l.sym, r.sym) else {
                    self.error(DiagnosticRule::Type, span, "quantity operation requires exact integer values");
                    return None;
                };
                let sym = match op {
                    BinaryOp::Add => Some(self.arena.int_add(a, b)),
                    BinaryOp::Sub => Some(self.arena.int_sub(a, b)),
                    BinaryOp::Mul => Some(self.arena.int_mul(a, b)),
                    BinaryOp::Div => Some(self.arena.int_div(a, b)),
                    BinaryOp::Rem => Some(self.arena.int_rem(a, b)),
                    _ if is_cmp => None,
                    _ => {
                        self.error(
                            DiagnosticRule::Type,
                            span,
                            format!("`{}` needs a fixed-width scalar operand", op.text()),
                        );
                        return None;
                    }
                };
                return Some(self.primitive_expr(
                    PrimitiveId::Binary(op),
                    vec![l, r],
                    if is_cmp { ValueType::Scalar(DType::Bool) } else { ValueType::Integer },
                    sym,
                    span,
                ));
            }
        }
        let is_logic = matches!(op, BinaryOp::And | BinaryOp::Or);
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let is_bit = matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor);
        let (axes, dtypes) = self.broadcast(&[&l, &r], &format!("`{}`", op.text()), span)?;
        let (a, b) = (dtypes[0], dtypes[1]);
        let mut target = None;
        if is_logic {
            if a != DType::Bool || b != DType::Bool {
                self.error(
                    DiagnosticRule::Type,
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
                    DiagnosticRule::Type,
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
                self.error(DiagnosticRule::Type, r.span, "integer shift count must be in 0..32");
                return None;
            }
        } else {
            let Some(d) = DType::promote(a, b) else {
                self.error(
                    DiagnosticRule::Type,
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
                    DiagnosticRule::Type,
                    span,
                    format!("`{}` needs integer operands, found {}", op.text(), d.name()),
                );
                return None;
            }
            if !d.is_numeric() && !(d == DType::Bool && matches!(op, BinaryOp::Eq | BinaryOp::Ne)) {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    format!("`{}` is not defined on {}", op.text(), d.name()),
                );
                return None;
            }
            target = Some(d);
        }
        let sym = match (l.sym, r.sym) {
            (Some(x), Some(y)) if axes.is_none() && a.is_int() && b.is_int() && !is_cmp => {
                Some(self.arena.scalar_integer(ScalarOp::Binary(op), &[(a, x), (b, y)]))
            }
            _ => None,
        };
        let operands = match target {
            Some(target) => self.promote_operands(vec![l, r], target),
            None => vec![l, r],
        };
        let mut out = self.elementwise_primitive(PrimitiveId::Binary(op), operands, axes, span)?;
        if !is_cmp && !is_logic && out.ty.scalar_dtype().is_some_and(DType::is_int) {
            out.sym = sym;
        }
        Some(out)
    }

    /// The word projection of a quantity. The word keeps the quantity as its
    /// symbol only where the quantity is proved in the word's range; otherwise
    /// its symbol is the wrapped value.
    pub fn quantity_to_word(&mut self, value: CheckedExpr, dtype: DType) -> CheckedExpr {
        if !quantity(&value.ty) {
            return value;
        }
        let sym = value.sym.map(|integer| {
            let (lower, upper) = match dtype {
                DType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
                DType::U32 => (0, i64::from(u32::MAX)),
                _ => unreachable!("a quantity projects only to an integer word"),
            };
            let lower = self.arena.int(lower);
            let upper = self.arena.int(upper);
            let above = self.arena.int_sub(integer, lower);
            let below = self.arena.int_sub(upper, integer);
            if super::prove::nonneg(&self.arena, &self.facts, above)
                && super::prove::nonneg(&self.arena, &self.facts, below)
            {
                integer
            } else {
                self.arena.scalar_integer(ScalarOp::Cast(dtype), &[(dtype, integer)])
            }
        });
        let span = value.span;
        self.primitive_expr(
            PrimitiveId::Cast(dtype),
            vec![value],
            ValueType::Scalar(dtype),
            sym,
            span,
        )
    }

    // ---- attributes ----

    fn attr(&mut self, base: CheckedExpr, name: &ast::Ident, span: Span) -> Option<CheckedExpr> {
        match name.name.as_str() {
            "T" => {
                let Some(shaped) = base.ty.shaped().filter(|s| s.rank() == 2).cloned() else {
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!("`.T` transposes a rank-2 tensor or view, found {}", self.shown(&base.ty)),
                    );
                    return None;
                };
                if shaped.packed_axis.is_some() {
                    self.error(DiagnosticRule::Type, span, "a packed tensor or view cannot be transposed; packets run along its last axis");
                    return None;
                }
                let t = TensorType {
                    axes: vec![shaped.axes[1], shaped.axes[0]],
                    elem: shaped.elem,
                    packed_axis: None,
                };
                let id = PrimitiveId::Transpose;
                if !primitive(&id).accepts(&[base.ty.clone()]) {
                    self.error(DiagnosticRule::Type, span, format!("`.T` is not defined on {}", self.shown(&base.ty)));
                    return None;
                }
                Some(self.primitive_expr(id, vec![base], ValueType::Tensor(t), None, span))
            }
            "words" | "scale" | "bias" | "coefficients" | "scale_factor" | "bias_factor" => {
                if !self.target_form(span, &format!("packed accessor `.{}`", name.name), None) {
                    return None;
                }
                let class = self.class_of(&base);
                if !matches!(class, ValueClass::Borrowed | ValueClass::Computed) {
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "`.{}` needs a packed view or tile, found {}",
                            name.name,
                            self.shown(&base.ty)
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
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "`.{}` needs a packed view or tile, found {}",
                            name.name,
                            self.shown(&base.ty)
                        ),
                    );
                    return None;
                };
                let Some(k) = s.packed_axis.and_then(|axis| s.axes.get(axis)).copied() else {
                    self.error(
                        DiagnosticRule::Type,
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
                        DiagnosticRule::Type,
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
                self.error(DiagnosticRule::Type, name.span, format!("unknown attribute `{other}`"));
                None
            }
        }
    }

    /// Accept a bounds need that is provable, or that depends on runtime data
    /// (then it is a runtime-checked obligation, as for every data-dependent
    /// index; nothing is ever clamped). Returns whether a runtime check is
    /// needed.
    fn require_in_bounds(&mut self, e: IntExpr, span: Span, what: &str) -> bool {
        if super::prove::nonneg(&self.arena, &self.facts, e) {
            false
        } else if self.data_dependent(e) {
            true
        } else {
            self.require_nonneg(DiagnosticRule::Type, e, span, what);
            false
        }
    }

    /// `f32(e)`: scalar cast, or read-and-convert of a tensor value (yields a
    /// tile). Casts are uniform for scalars and tensors (L7): bool converts
    /// to 0/1, and a numeric value never converts to bool.
    pub fn cast(&mut self, dtype: DType, args: &[ast::Arg], span: Span) -> Option<CheckedExpr> {
        let [ast::Arg { name: None, value }] = args else {
            self.error(DiagnosticRule::Type, span, format!("`{}(x)` takes one argument", dtype.name()));
            return None;
        };
        if dtype == DType::Bool {
            self.error(
                DiagnosticRule::Type,
                span,
                "no conversion produces `bool`; write a comparison such as `x != 0`",
            );
            return None;
        }
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
                Elem::Dtype(from) => from.is_numeric() || *from == DType::Bool,
                Elem::Param(_) => dtype.is_float(),
            };
            if !ok {
                self.error(
                    DiagnosticRule::Type,
                    span,
                    format!(
                        "cannot convert {} to `{}` elements; packed values decode with `f32(v)`",
                        self.shown(&inner.ty),
                        dtype.name()
                    ),
                );
                return None;
            }
            let element = s.elem.clone();
            self.elements.decoded_read(&element);
            let axes = Some(s.axes.clone());
            return self.elementwise_primitive(PrimitiveId::Cast(dtype), vec![inner], axes, span);
        }
        if quantity(&inner.ty) {
            if dtype.is_int() {
                return Some(self.quantity_to_word(inner, dtype));
            }
            // L6: the exact integer rounded once to the float dtype.
            return Some(self.primitive_expr(
                PrimitiveId::Cast(dtype),
                vec![inner],
                ValueType::Scalar(dtype),
                None,
                span,
            ));
        }
        let Some(from) = inner.ty.scalar_dtype() else {
            self.error(
                DiagnosticRule::Type,
                span,
                format!(
                    "cannot cast {} to {}; capability values have no scalar conversion",
                    self.shown(&inner.ty),
                    dtype.name()
                ),
            );
            return None;
        };
        let sym = if dtype.is_int() && from.is_int() {
            inner.sym.map(|value| self.arena.scalar_integer(
                ScalarOp::Cast(dtype),
                &[(from, value)],
            ))
        } else {
            None
        };
        let mut out =
            self.elementwise_primitive(PrimitiveId::Cast(dtype), vec![inner], None, span)?;
        out.sym = sym;
        Some(out)
    }

    /// L31: a data word (or quantity) at an `index[bound]` position. Proves
    /// `0 <= w <= bound - 1` over the value's one symbol and marks the value.
    pub fn index_position(
        &mut self,
        value: CheckedExpr,
        bound: IntExpr,
        span: Span,
    ) -> Option<CheckedExpr> {
        let sym = self.position_symbol(&value)?;
        let one = self.arena.int(1);
        let last = self.arena.int_sub(bound, one);
        let headroom = self.arena.int_sub(last, sym);
        let text = self.text(value.span).to_owned();
        let mut unproved = Vec::new();
        if !super::prove::nonneg(&self.arena, &self.facts, sym) {
            unproved.push(format!("0 <= {text}"));
        }
        if !super::prove::nonneg(&self.arena, &self.facts, headroom) {
            unproved.push(format!("{text} <= {}", self.render(last)));
        }
        if !unproved.is_empty() {
            let kind = if word(&value.ty) { "a data word" } else { "a quantity" };
            let target = self.shown(&ValueType::Index { bound });
            self.error(
                DiagnosticRule::Type,
                span,
                format!(
                    "`{text}` is {kind} at an `{target}` position; prove {} in scope, e.g. with `if`",
                    unproved
                        .iter()
                        .map(|conjunct| format!("`{conjunct}`"))
                        .collect::<Vec<_>>()
                        .join(" and ")
                ),
            );
            return None;
        }
        Some(CheckedExpr::new(
            CheckedExprKind::IndexPosition(Box::new(value)),
            ValueType::Index { bound },
            Some(sym),
            span,
        ))
    }
}

#[cfg(test)]
mod literal_tests {
    use crate::checked::{check_source, SourceFile, SourceSet};
    use crate::entry::{ElementBindings, SemanticNodeView};
    use crate::interp::{Interpreter, OutcomeValue};
    use crate::intrinsics::PrimitiveId;
    use crate::reference_math::ReferenceScalar;

    #[test]
    fn checked_literals_keep_direct_quantization_through_reference_execution() {
        for (token, expected) in [
            ("1.0004882812500002", ReferenceScalar::F16(0x3c01)),
            ("1.0039062500000002", ReferenceScalar::BF16(0x3f81)),
            // Above the midpoint by one exact integer; conversion through F64
            // would lose that unit and incorrectly round down to 0x5a000000.
            ("9007199791611905", ReferenceScalar::F32(0x5a00_0001)),
            ("18446744073709551615", ReferenceScalar::F32(0x5f80_0000)),
            ("-0.0", ReferenceScalar::F16(0x8000)),
            ("-0.0", ReferenceScalar::BF16(0x8000)),
            ("-0.0", ReferenceScalar::F32(0x8000_0000)),
            ("-0.0000000000001", ReferenceScalar::F16(0x8000)),
            ("-inf", ReferenceScalar::F16(0xfc00)),
            ("inf", ReferenceScalar::BF16(0x7f80)),
            ("4294967295", ReferenceScalar::U32(u32::MAX)),
            ("-2147483648", ReferenceScalar::I32(i32::MIN)),
            ("true", ReferenceScalar::Bool(true)),
        ] {
            let dtype = expected.dtype();
            let module = check_source(SourceSet::new(vec![SourceFile {
                path: "literal-bits.seismic".into(),
                text: format!("fn probe() -> {dtype}:\n    return {token}\n"),
            }]))
            .unwrap_or_else(|error| panic!("{dtype} {token}: {error:?}"));
            let entry = module
                .entry(
                    module.entry_named("probe").unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            let program = entry.program();
            let body = program.function(program.family(program.root()).reference().function());
            let constants: Vec<_> = body
                .nodes(body.root())
                .filter_map(|(_, node)| match node.view() {
                    SemanticNodeView::Primitive {
                        primitive: PrimitiveId::Constant(value),
                        ..
                    } => Some(*value),
                    _ => None,
                })
                .collect();
            assert_eq!(constants, [expected], "checked {dtype} {token}");
            let outcome = Interpreter::new(&entry).run(&[]).unwrap();
            let result = outcome.results().next().unwrap();
            let OutcomeValue::Scalar(actual) = result.value() else {
                panic!("scalar result")
            };
            assert_eq!(actual, expected, "reference {dtype} {token}");
        }
    }
}
