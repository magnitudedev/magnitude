//! Statements: bindings, state updates, logical loops, and return path rules.

use super::resolve::ParamOwnership;
use super::{elem_rounds, var_atom, Checker};
use crate::sir::{self, Expr, ExprKind, Stmt, StmtKind, VarId, VarKind};
use crate::span::Span;
use crate::sym::Sym;
use crate::syntax::ast::{self, AssignOp, BinaryOp, ExprKind as A};
use crate::types::{DType, Extent, RegionId, Ty};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopCardinality {
    Zero,
    One,
    RepeatedOrUnknown,
}

impl<'a> Checker<'a> {
    pub fn push_scope(&mut self) {
        self.scopes.push(Default::default());
    }

    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            for variable in scope.values() {
                self.borrows.remove(variable);
            }
        }
    }

    pub fn block(&mut self, b: &ast::Block) -> Vec<Stmt> {
        let mut out = Vec::new();
        for s in &b.stmts {
            if self.yields.last().is_some_and(|y| y.done) {
                self.error(
                    s.span,
                    "unreachable statement: `yield`/`return` is terminal for its result boundary",
                );
            }
            if let Some(kind) = self.stmt(s) {
                out.push(Stmt { kind, span: s.span });
            }
        }
        out
    }

    fn scoped_block(&mut self, b: &ast::Block) -> Vec<Stmt> {
        self.push_scope();
        let out = self.block(b);
        self.pop_scope();
        out
    }

    fn stmt(&mut self, s: &ast::Stmt) -> Option<StmtKind> {
        match &s.kind {
            ast::StmtKind::Let {
                mutable,
                pattern,
                value,
            } => self.bind(pattern, value, *mutable),
            ast::StmtKind::Assign { target, op, value } => self.assign(target, *op, value),
            ast::StmtKind::For {
                parallel,
                targets,
                iter,
                body,
            } => {
                // State carried through a loop is not its pre-loop definition on every visit.
                self.dyn_slices.clear();
                if let Some(y) = self.yields.last_mut() {
                    y.loops += 1;
                }
                let kind = self.for_stmt(*parallel, targets, iter, body);
                if let Some(y) = self.yields.last_mut() {
                    y.loops -= 1;
                }
                kind
            }
            ast::StmtKind::If { cond, then, els } => self.if_stmt(cond, then, els.as_ref()),
            ast::StmtKind::Return(values) => self.return_stmt(values, s.span),
            ast::StmtKind::Expr(e) => {
                let e = self.expr(e, None)?;
                if e.ty != Ty::Void {
                    self.error(e.span, format!("an expression statement is a call evaluated for its `out`/`inout` effects; this value of type {} is unused", e.ty));
                }
                Some(StmtKind::Expr(e))
            }
        }
    }

    // ---- bindings ----

    /// A source value expression.
    fn value(&mut self, e: &ast::Expr, expected: Option<&Ty>) -> Option<Expr> {
        self.expr(e, expected)
    }

    /// Partial flags of the components of a tuple-typed value.
    fn component_partials(&self, e: &Expr, n: usize) -> Vec<bool> {
        match &e.kind {
            ExprKind::Tuple(items) if items.len() == n => items.iter().map(|i| i.partial).collect(),
            ExprKind::Member { result, .. } => match &result.ty {
                Ty::Result(r) => self
                    .result_partials
                    .get(&r.producer)
                    .filter(|p| p.len() == n)
                    .cloned()
                    .unwrap_or_else(|| vec![e.partial; n]),
                _ => vec![e.partial; n],
            },
            _ => vec![e.partial; n],
        }
    }

    fn partial_origin_of(&self, e: &Expr) -> Option<RegionId> {
        match &e.kind {
            ExprKind::Member { result, .. } => match &result.ty {
                Ty::Result(r) => Some(r.origin),
                _ => None,
            },
            ExprKind::Var(v) => self.partial_origin.get(v).copied(),
            ExprKind::Cast { expr, .. } => self.partial_origin_of(expr),
            _ => None,
        }
    }

    fn poison(&mut self, pattern: &ast::Pattern) {
        match pattern {
            ast::Pattern::Name(name) => {
                self.poisoned.insert(name.name.clone());
            }
            ast::Pattern::Tuple(items) => items.iter().for_each(|item| self.poison(item)),
        }
    }

    fn bind(&mut self, pattern: &ast::Pattern, value: &ast::Expr, state: bool) -> Option<StmtKind> {
        let Some(value) = self.value(value, None) else {
            self.poison(pattern);
            return None;
        };
        if value.ty == Ty::Void {
            self.error(value.span, "cannot bind a call that returns nothing");
            return None;
        }
        let origin = self.partial_origin_of(&value);
        let moved = matches!(value.ty, Ty::Tensor(_))
            .then(|| self.root_var(&value))
            .flatten();
        let pattern = self.destructure(pattern, &value.ty.clone(), &value, state, origin)?;
        if let Some(root) = moved {
            self.moved.insert(root);
        }
        Some(StmtKind::Bind { pattern, value })
    }

    fn destructure(
        &mut self,
        pattern: &ast::Pattern,
        ty: &Ty,
        value: &Expr,
        state: bool,
        origin: Option<RegionId>,
    ) -> Option<sir::Pattern> {
        match pattern {
            ast::Pattern::Name(name) => {
                if state && matches!(ty, Ty::Slice(_) | Ty::Coord(_) | Ty::Range(_)) {
                    self.error(name.span, format!("`let mut` declares mutable state or a mutable storage capability; a {ty} is a structural handle and is bound with `let`"));
                    return None;
                }
                if matches!(ty, Ty::Coord(_)) {
                    self.error(name.span, "a tile coordinate cannot be rebound; it indexes tiles sharing its axis or is converted with `coord(i)`");
                    return None;
                }
                let id = self.declare(
                    &name.name,
                    ty.clone(),
                    name.span,
                    if state {
                        VarKind::State
                    } else {
                        VarKind::Value
                    },
                );
                self.vars[id].partial = value.partial;
                if value.partial {
                    if let Some(origin) = origin {
                        self.partial_origin.insert(id, origin);
                    }
                }
                if matches!(value.kind, ExprKind::TileAlloc) {
                    self.unassigned.insert(id);
                }
                if matches!(ty, Ty::View(_)) {
                    if let Some(root) = self.root_var(value) {
                        let writable = matches!(
                            self.vars[root].kind,
                            VarKind::Param(i) if self.sig.params[i].mode != sir::Mode::In
                        ) || matches!(self.vars[root].kind, VarKind::State);
                        let exclusive = state && writable;
                        if self.borrows.values().any(|(borrowed, prior_exclusive)| {
                            *borrowed == root && (exclusive || *prior_exclusive)
                        }) {
                            self.error(
                                name.span,
                                format!(
                                    "tensor borrow of `{}` overlaps a live {} borrow",
                                    self.vars[root].name,
                                    if exclusive {
                                        "mutable"
                                    } else {
                                        "exclusive mutable"
                                    }
                                ),
                            );
                            return None;
                        }
                        self.view_roots.insert(id, root);
                        self.view_bound.insert(id, self.mutated.len());
                        self.borrows.insert(id, (root, exclusive));
                    }
                }
                if !state && ty.scalar_dtype() == Some(DType::I32) {
                    match (&value.sym, &value.kind) {
                        (Some(sym), _) => {
                            self.scalar_symbols.insert(id, sym.clone());
                        }
                        // A scalar argmax over a semantic axis is an index into that axis.
                        (
                            None,
                            ExprKind::Reduce {
                                value: reduced,
                                axis,
                                op: sir::ReduceOp::Argmax,
                                ..
                            },
                        ) => {
                            if let Some(Extent::Semantic(extent)) =
                                reduced.ty.shaped().and_then(|s| s.axes.get(*axis)).cloned()
                            {
                                let atom = var_atom(&name.name, id);
                                self.facts.set_range(
                                    atom.clone(),
                                    Sym::constant(0),
                                    extent.sub(&Sym::constant(1)),
                                );
                                self.atoms.insert(id, atom);
                            }
                        }
                        _ => {}
                    }
                }
                Some(sir::Pattern::Var(id))
            }
            ast::Pattern::Tuple(items) => {
                let Ty::Tuple(tys) = ty else {
                    self.error(
                        value.span,
                        format!("a tuple pattern destructures a tuple; this value is a {ty}"),
                    );
                    return None;
                };
                if tys.len() != items.len() {
                    self.error(
                        value.span,
                        format!(
                            "pattern binds {} names but the value has {} components",
                            items.len(),
                            tys.len()
                        ),
                    );
                    return None;
                }
                let partials = self.component_partials(value, tys.len());
                let parts: Vec<Option<&Expr>> = match &value.kind {
                    ExprKind::Tuple(parts) if parts.len() == tys.len() => {
                        parts.iter().map(Some).collect()
                    }
                    _ => vec![None; tys.len()],
                };
                let mut out = Vec::new();
                for (((item, ty), partial), part) in items.iter().zip(tys).zip(partials).zip(parts)
                {
                    let component = match part {
                        Some(part) => part.clone(),
                        None => Expr {
                            kind: ExprKind::Tuple(Vec::new()),
                            ty: ty.clone(),
                            sym: None,
                            partial,
                            span: value.span,
                        },
                    };
                    out.push(self.destructure(item, ty, &component, state, origin)?);
                }
                Some(sir::Pattern::Tuple(out))
            }
        }
    }

    // ---- state updates ----

    fn assign(&mut self, target: &ast::Expr, op: AssignOp, value: &ast::Expr) -> Option<StmtKind> {
        match &target.kind {
            A::Tuple(places) => {
                if op != AssignOp::Assign {
                    self.error(target.span, "tuple assignment uses `=`");
                    return None;
                }
                let mut targets = Vec::new();
                for place in places {
                    let A::Name(name) = &place.kind else {
                        self.error(
                            place.span,
                            "tuple assignment installs whole `let mut` state objects",
                        );
                        return None;
                    };
                    targets.push(self.state_place(name)?);
                }
                let expected = Ty::Tuple(targets.iter().map(|t| t.ty.clone()).collect());
                // All right-hand sides read the old versions. Accumulating a partial into
                // state is recognized per component of a tuple literal.
                let value = match &value.kind {
                    A::Tuple(parts) if parts.len() == targets.len() => {
                        let mut out = Vec::new();
                        for (part, place) in parts.iter().zip(&targets) {
                            let component = match self.accumulation(part, place) {
                                Some(built) => built?,
                                None => self.expr(part, Some(&place.ty))?,
                            };
                            if !place.partial {
                                self.forbid_partial(&component, "a state update");
                            }
                            out.push(component);
                        }
                        let ty = Ty::Tuple(out.iter().map(|e| e.ty.clone()).collect());
                        let partial = out.iter().any(|e| e.partial);
                        Expr {
                            kind: ExprKind::Tuple(out),
                            ty,
                            sym: None,
                            partial,
                            span: value.span,
                        }
                    }
                    _ => {
                        let value = self.expr(value, Some(&expected))?;
                        if !targets.iter().all(|t| t.partial) {
                            self.forbid_partial(&value, "a state update");
                        }
                        value
                    }
                };
                if !self.assignable(&expected, &value.ty) {
                    self.error(
                        value.span,
                        format!(
                            "tuple assignment expects {expected} but the value is {}",
                            value.ty
                        ),
                    );
                    return None;
                }
                for place in &targets {
                    self.write(place, place.span, true)?;
                }
                let target = Expr {
                    kind: ExprKind::Tuple(targets),
                    ty: expected,
                    sym: None,
                    partial: false,
                    span: target.span,
                };
                Some(StmtKind::Assign { target, op, value })
            }
            A::Name(name) => {
                let place = self.state_place(name)?;
                let ty = place.ty.clone();
                let value = match (op, self.accumulation_if(op, value, &place)) {
                    (_, Some(built)) => built?,
                    (_, None) => self.expr(value, Some(&ty))?,
                };
                if op == AssignOp::Assign {
                    if !self.assignable(&ty, &value.ty) {
                        self.error(
                            value.span,
                            format!(
                                "`{}` has type {ty} but the value has type {}",
                                name.name, value.ty
                            ),
                        );
                        return None;
                    }
                } else {
                    self.arith_assign(&ty, &value, op)?;
                }
                if !place.partial && !(op == AssignOp::Add && self.accumulates_into(&value)) {
                    self.forbid_partial(&value, "a state update");
                }
                let moved = (op == AssignOp::Assign && matches!(value.ty, Ty::Tensor(_)))
                    .then(|| self.root_var(&value))
                    .flatten();
                let root = self.write(&place, target.span, true)?;
                if op == AssignOp::Assign && matches!(ty, Ty::Tensor(_)) {
                    // Assignment reinitializes owned state, including state that was moved
                    // earlier on this path. The source is consumed separately below.
                    self.moved.remove(&root);
                }
                if let Some(source) = moved.filter(|source| *source != root) {
                    self.moved.insert(source);
                }
                self.unassigned.remove(&root);
                Some(StmtKind::Assign {
                    target: place,
                    op,
                    value,
                })
            }
            A::Index { base, .. } => {
                let A::Name(base_name) = &base.kind else {
                    self.error(
                        base.span,
                        "element assignment indexes a tile variable directly",
                    );
                    return None;
                };
                let Some(id) = self.lookup(&base_name.name) else {
                    self.error(
                        base_name.span,
                        format!("`{}` is not declared", base_name.name),
                    );
                    return None;
                };
                let logical_mut = matches!(
                    self.vars[id].kind,
                    VarKind::Param(parameter)
                        if self.sig.params[parameter].ownership == ParamOwnership::Exclusive
                ) || matches!(
                    (&self.vars[id].kind, &self.vars[id].ty),
                    (VarKind::State, Ty::Tensor(_))
                ) || self
                    .borrows
                    .get(&id)
                    .is_some_and(|(_, exclusive)| *exclusive);
                if !matches!(self.vars[id].ty, Ty::Tile(_)) && !logical_mut {
                    self.error(target.span, format!("only tile elements are assigned; `{}` is a {}. `publish` is the only write to tensor storage", base_name.name, self.vars[id].ty));
                    return None;
                }
                let pending = self
                    .pending_full_assign
                    .iter()
                    .find(|(v, _)| *v == id)
                    .map(|(_, axes)| axes.clone());
                if self.unassigned.contains(&id) && pending.is_none() && self.init_loop_depth == 0 {
                    self.error(target.span, format!("`{}` is written element-wise before it is initialized; assign every element through `for … in owned(…)`", base_name.name));
                    return None;
                }
                let place = self.place(target)?;
                let covers = op == AssignOp::Assign
                    && pending.as_ref().is_some_and(|axes| {
                        let ExprKind::Index { indices, .. } = &place.kind else {
                            return false;
                        };
                        indices.len() == axes.len()
                            && indices.iter().zip(axes).all(|(index, axis)| match index {
                                sir::Index::Coord(v) => v == axis,
                                sir::Index::Point(e) => {
                                    matches!(e.kind, ExprKind::Var(v) if v == *axis)
                                }
                                _ => false,
                            })
                    });
                let Ty::Scalar(dtype) = place.ty.clone() else {
                    if op != AssignOp::Assign {
                        self.error(target.span, "compound assignment requires a scalar element place");
                        return None;
                    }
                    let value = self.expr(value, None)?;
                    let compatible = match (&value.ty, &place.ty) {
                        (Ty::Tile(v) | Ty::View(v) | Ty::Tensor(v), Ty::Tensor(d) | Ty::View(d) | Ty::Tile(d)) => {
                            self.same_axes(v, d) && elem_rounds(&v.elem, &d.elem)
                        }
                        _ => false,
                    };
                    if !compatible {
                        self.error(value.span, format!("cannot assign {} to {}; convert explicitly", value.ty, place.ty));
                        return None;
                    }
                    self.forbid_partial(&value, "a tensor slice assignment");
                    let root = self.write(&place, target.span, false)?;
                    if covers || self.covers_whole(&place) {
                        self.unassigned.remove(&root);
                        self.pending_full_assign.retain(|(variable, _)| *variable != id);
                    }
                    return Some(StmtKind::Publish { value, destination: place });
                };
                let value = self.expr(value, Some(&Ty::Scalar(dtype)))?;
                let Some(vd) = value.ty.scalar_dtype() else {
                    self.error(
                        value.span,
                        format!(
                            "cannot assign {} to an element of dtype {}",
                            value.ty,
                            dtype.name()
                        ),
                    );
                    return None;
                };
                if op == AssignOp::Assign {
                    if !(vd == dtype || (vd.is_float() && dtype.is_float())) {
                        self.error(
                            value.span,
                            format!(
                                "cannot assign {} to an element of dtype {}; cast explicitly",
                                vd.name(),
                                dtype.name()
                            ),
                        );
                    }
                } else {
                    self.arith_assign(&Ty::Scalar(dtype), &value, op)?;
                }
                self.forbid_partial(&value, "an element assignment");
                self.write(&place, target.span, false)?;
                if covers {
                    self.unassigned.remove(&id);
                    self.pending_full_assign.retain(|(v, _)| *v != id);
                }
                Some(StmtKind::Assign {
                    target: place,
                    op,
                    value,
                })
            }
            _ => {
                self.error(
                    target.span,
                    "an assignment target is `let mut` state, a tile element, or a tuple of state",
                );
                None
            }
        }
    }

    fn state_place(&mut self, name: &ast::Ident) -> Option<Expr> {
        let Some(id) = self.lookup(&name.name) else {
            self.error(
                name.span,
                format!(
                    "`{}` is not declared; introduce state with `let mut`",
                    name.name
                ),
            );
            return None;
        };
        let ty = self.vars[id].ty.clone();
        match self.vars[id].kind {
            VarKind::State => {}
            VarKind::Param(i)
                if self.sig.params[i].mode != sir::Mode::In && matches!(ty, Ty::Tile(_)) => {}
            VarKind::Slice(_) => {
                self.error(
                    name.span,
                    format!(
                        "slice `{}` is immutable geometry and cannot be reassigned",
                        name.name
                    ),
                );
                return None;
            }
            _ => {
                self.error(name.span, format!("`{}` is not mutable state; only `let mut` bindings and `out`/`inout` tiles are assigned", name.name));
                return None;
            }
        }
        if matches!(ty, Ty::Result(_) | Ty::Native(_)) {
            self.error(
                name.span,
                format!("a {ty} is immutable once produced and cannot be reassigned"),
            );
            return None;
        }
        Some(Expr {
            kind: ExprKind::Var(id),
            ty,
            sym: None,
            partial: self.vars[id].partial,
            span: name.span,
        })
    }

    fn accumulation_if(
        &mut self,
        op: AssignOp,
        value: &ast::Expr,
        place: &Expr,
    ) -> Option<Option<Expr>> {
        if op == AssignOp::Assign {
            self.accumulation(value, place)
        } else {
            None
        }
    }

    /// `s = s + p`, `s = max(s, p)`, `s = min(s, p)`: with `p` a forwarded partial of the
    /// result being traversed this is the one admitted accumulation of partials into state
    /// outside a structural merge. `None` when `value` does not have this form.
    fn accumulation(&mut self, value: &ast::Expr, place: &Expr) -> Option<Option<Expr>> {
        let ExprKind::Var(state) = place.kind else {
            return None;
        };
        let is_state = |e: &ast::Expr, c: &Checker| matches!(&e.kind, A::Name(n) if c.lookup(&n.name) == Some(state));
        let (math, operands): (Option<sir::Math>, [&ast::Expr; 2]) = match &value.kind {
            A::Binary {
                op: BinaryOp::Add,
                lhs,
                rhs,
            } => (None, [lhs, rhs]),
            A::Call {
                callee,
                bindings,
                args,
            } if bindings.is_empty()
                && args.len() == 2
                && args.iter().all(|a| a.name.is_none()) =>
            {
                match &callee.kind {
                    A::Name(n) if n.name == "max" => {
                        (Some(sir::Math::Max), [&args[0].value, &args[1].value])
                    }
                    A::Name(n) if n.name == "min" => {
                        (Some(sir::Math::Min), [&args[0].value, &args[1].value])
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        let state_first = is_state(operands[0], self);
        if !state_first && !is_state(operands[1], self) {
            return None;
        }
        Some(self.accumulate(math, operands, state_first, place, value.span))
    }

    fn accumulate(
        &mut self,
        math: Option<sir::Math>,
        operands: [&ast::Expr; 2],
        state_first: bool,
        place: &Expr,
        span: Span,
    ) -> Option<Expr> {
        let mut other = self.expr(operands[usize::from(state_first)], Some(&place.ty))?;
        let state = self.expr(operands[usize::from(!state_first)], None)?;
        let admitted = other.partial && !self.partial_free();
        if admitted {
            if !self.accumulates_into(&other) {
                self.error(other.span, "a partial value accumulates into `let mut` state only inside a traversal of the result it belongs to (`ordered [p] in results:`)");
                return None;
            }
            // The accumulated state is whole once the traversal completes.
            other.partial = false;
        }
        let (lhs, rhs) = if state_first {
            (state, other)
        } else {
            (other, state)
        };
        match math {
            Some(op) => self.math_exprs(op, lhs, rhs, span),
            None => self.binary_exprs(BinaryOp::Add, lhs, rhs, span),
        }
    }

    /// Whether `e` is a forwarded partial of a result the current code is traversing.
    fn accumulates_into(&self, e: &Expr) -> bool {
        let _ = e;
        false
    }

    fn arith_assign(&mut self, target: &Ty, value: &Expr, op: AssignOp) -> Option<()> {
        if let (Ty::Tile(a), Ty::Tile(b)) = (target, &value.ty) {
            if !self.same_axes(a, b) || !elem_rounds(&b.elem, &a.elem) {
                self.error(
                    value.span,
                    format!(
                        "`{}` needs tiles over identical axes: {} vs {}",
                        op.text(),
                        target,
                        value.ty
                    ),
                );
                return None;
            }
            return Some(());
        }
        if let (Ty::Tile(a), Some(v)) = (target, value.ty.scalar_dtype()) {
            if a.elem
                .read_dtype()
                .is_none_or(|t| DType::promote(t, v).is_some())
            {
                return Some(());
            }
        }
        let (Some(t), Some(v)) = (target.scalar_dtype(), value.ty.scalar_dtype()) else {
            self.error(
                value.span,
                format!(
                    "`{}` needs numeric operands, found {} and {}",
                    op.text(),
                    target,
                    value.ty
                ),
            );
            return None;
        };
        if !t.is_numeric() || DType::promote(t, v).is_none() {
            self.error(
                value.span,
                format!(
                    "`{}` between {} and {} needs an explicit cast",
                    op.text(),
                    t.name(),
                    v.name()
                ),
            );
            return None;
        }
        Some(())
    }

    // ---- control ----

    fn if_stmt(
        &mut self,
        cond: &ast::Expr,
        then: &ast::Block,
        els: Option<&ast::Block>,
    ) -> Option<StmtKind> {
        let cond = self.expr(cond, Some(&Ty::Scalar(DType::Bool)))?;
        match &cond.ty {
            Ty::Scalar(DType::Bool) => {}
            Ty::Tile(s) if s.elem == crate::types::Elem::Dtype(DType::Bool) => {
                self.error(cond.span, "a mask is a `bool` tile, not a scalar condition; there is no implicit reduction. Use `select(mask, a, b)`");
                return None;
            }
            other => {
                self.error(
                    cond.span,
                    format!("an `if` condition is a scalar `bool`, found {other}"),
                );
                return None;
            }
        }
        self.forbid_partial(&cond, "a branch condition");
        let facts_before = self.facts.clone();
        let unassigned_before = self.unassigned.clone();
        let pending_before = self.pending_full_assign.clone();
        let symbols_before = self.scalar_symbols.clone();
        let moved_before = self.moved.clone();
        let done_before: Vec<bool> = self.yields.iter().map(|y| y.done).collect();

        self.assume(&cond, false);
        let then = self.scoped_block(then);
        self.facts = facts_before.clone();
        let unassigned_then = std::mem::replace(&mut self.unassigned, unassigned_before);
        let symbols_then = std::mem::replace(&mut self.scalar_symbols, symbols_before);
        let moved_then = std::mem::replace(&mut self.moved, moved_before.clone());
        let done_then: Vec<bool> = self.yields.iter().map(|y| y.done).collect();
        for (y, done) in self.yields.iter_mut().zip(&done_before) {
            y.done = *done;
        }
        self.pending_full_assign = pending_before.clone();

        let els = match els {
            Some(b) => {
                self.assume(&cond, true);
                self.scoped_block(b)
            }
            None => Vec::new(),
        };
        self.facts = facts_before;
        let moved_else = self.moved.clone();
        let done_else: Vec<bool> = self.yields.iter().map(|y| y.done).collect();
        if done_then != done_else {
            self.error(cond.span, "one path of this `if` yields/returns and the other does not; every path through a result boundary produces exactly one value with one schema");
        }
        self.unassigned.extend(unassigned_then);
        self.scalar_symbols
            .retain(|id, sym| symbols_then.get(id) == Some(sym));
        // A value is available after an `if` only when it is available on every
        // path. Both arms are checked from `moved_before`, so a move in one arm
        // cannot spuriously poison checking of the other arm.
        self.moved = moved_then.union(&moved_else).copied().collect();
        self.pending_full_assign = pending_before
            .into_iter()
            .filter(|(v, _)| self.unassigned.contains(v))
            .collect();
        Some(StmtKind::If { cond, then, els })
    }

    /// Path facts of a condition over symbolic integers (or of its negation).
    fn assume(&mut self, cond: &Expr, negate: bool) {
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            if let ExprKind::Unary {
                op: ast::UnaryOp::Not,
                expr,
            } = &cond.kind
            {
                self.assume(expr, !negate);
            }
            return;
        };
        match (op, negate) {
            (BinaryOp::And, false) | (BinaryOp::Or, true) => {
                self.assume(lhs, negate);
                self.assume(rhs, negate);
                return;
            }
            (BinaryOp::And, true) | (BinaryOp::Or, false) => return,
            _ => {}
        }
        let (Some(l), Some(r)) = (&lhs.sym, &rhs.sym) else {
            return;
        };
        let one = Sym::constant(1);
        match (op, negate) {
            (BinaryOp::Lt, false) | (BinaryOp::Ge, true) => self.assume_nonneg(&r.sub(l).sub(&one)),
            (BinaryOp::Le, false) | (BinaryOp::Gt, true) => self.assume_nonneg(&r.sub(l)),
            (BinaryOp::Gt, false) | (BinaryOp::Le, true) => self.assume_nonneg(&l.sub(r).sub(&one)),
            (BinaryOp::Ge, false) | (BinaryOp::Lt, true) => self.assume_nonneg(&l.sub(r)),
            (BinaryOp::Eq, false) | (BinaryOp::Ne, true) => self.assume_zero(&l.sub(r)),
            _ => {}
        }
    }

    fn for_stmt(
        &mut self,
        parallel: bool,
        targets: &[ast::Ident],
        iter: &ast::Expr,
        body: &ast::Block,
    ) -> Option<StmtKind> {
        match &iter.kind {
            A::Range { .. } | A::Name(_)
                if matches!(&iter.kind, A::Range { .. })
                    || matches!(
                        &iter.kind,
                        A::Name(name)
                            if self.lookup(&name.name).is_some_and(|id| matches!(self.vars[id].ty, Ty::Range(_)))
                    ) =>
            {
                let [target] = targets else {
                    self.error(iter.span, "a semantic range binds exactly one name");
                    return None;
                };
                let range = self.expr(iter, None)?;
                let Ty::Range(bound) = &range.ty else {
                    unreachable!("range syntax and range bindings check as range values")
                };
                let (lo_sym, hi_sym) = match &range.kind {
                    ExprKind::Range { lo, hi } => (
                        lo.sym
                            .clone()
                            .expect("checked range lower bound is symbolic"),
                        hi.sym
                            .clone()
                            .expect("checked range upper bound is symbolic"),
                    ),
                    _ => (Sym::constant(0), bound.clone()),
                };
                let cardinality = match hi_sym.sub(&lo_sym).as_constant() {
                    Some(n) if n <= 0 => LoopCardinality::Zero,
                    Some(1) => LoopCardinality::One,
                    _ => LoopCardinality::RepeatedOrUnknown,
                };
                let floor = self.vars.len();
                let moved_before = self.moved.clone();
                self.push_scope();
                let ty = Ty::Index(bound.clone());
                let var = self.declare(&target.name, ty, target.span, VarKind::RangeIndex);
                let atom = var_atom(&target.name, var);
                self.facts
                    .set_range(atom.clone(), lo_sym.clone(), hi_sym.sub(&Sym::constant(1)));
                self.atoms.insert(var, atom);
                if parallel {
                    self.logical_parallel.push((floor, var));
                }
                // A complete logical `0..axis` traversal carries the same definite-
                // initialization proof as the retired authored `owned(t)` iterator.
                let prior_pending = self.pending_full_assign.clone();
                let mut pending = Vec::new();
                for tensor in self.unassigned.iter().copied() {
                    let (Ty::Tensor(shaped) | Ty::Tile(shaped)) = &self.vars[tensor].ty else { continue };
                    let prefix = prior_pending.iter().find(|(candidate, _)| *candidate == tensor)
                        .map(|(_, axes)| axes.clone()).unwrap_or_default();
                    let axis = prefix.len();
                    let Some(Extent::Semantic(extent)) = shaped.axes.get(axis) else { continue };
                    if lo_sym.is_zero() && &hi_sym == extent {
                        let mut axes = prefix;
                        axes.push(var);
                        pending.push((tensor, axes));
                    }
                }
                self.pending_full_assign = pending;
                self.init_loop_depth += 1;
                let body = self.block(body);
                self.init_loop_depth -= 1;
                self.pending_full_assign = prior_pending;
                if parallel {
                    self.logical_parallel.pop();
                }
                self.pop_scope();
                self.finish_loop_ownership(moved_before, floor, cardinality, iter.span);
                let literal_bounds = match &range.kind {
                    ExprKind::Range { lo, hi } => Some((lo.as_ref().clone(), hi.as_ref().clone())),
                    _ => None,
                };
                let (lo, hi, value) = match literal_bounds {
                    Some((lo, hi)) => (lo, hi, None),
                    None => (
                        Expr {
                            kind: ExprKind::Int(0),
                            ty: Ty::Scalar(DType::I32),
                            sym: Some(Sym::constant(0)),
                            partial: false,
                            span: iter.span,
                        },
                        Expr {
                            kind: ExprKind::Int(0),
                            ty: Ty::Scalar(DType::I32),
                            sym: Some(bound.clone()),
                            partial: false,
                            span: iter.span,
                        },
                        Some(range),
                    ),
                };
                let kind = StmtKind::Range {
                    kind: if parallel {
                        sir::LoopKind::Parallel
                    } else {
                        sir::LoopKind::Ordered
                    },
                    var,
                    lo,
                    hi,
                    value,
                    body,
                };
                let loop_block = vec![Stmt {
                    kind: kind.clone(),
                    span: iter.span,
                }];
                let pending: Vec<_> = self.unassigned.iter().copied().collect();
                for variable in pending {
                    let Some(shaped) = self.vars[variable].ty.shaped() else { continue };
                    if super::definitely_initializes_var(
                        &self.vars,
                        &loop_block,
                        variable,
                        shaped,
                        &self.facts,
                    ) {
                        self.unassigned.remove(&variable);
                    }
                }
                Some(kind)
            }
            A::Name(name) => {
                if parallel {
                    self.error(iter.span, "`parallel for` currently requires an explicit bounded range such as `0..N`");
                    return None;
                }
                let [target] = targets else {
                    self.error(iter.span, "`for h in slice` binds exactly one name");
                    return None;
                };
                let slice = match self.lookup(&name.name).map(|id| self.vars[id].ty.clone()) {
                    Some(Ty::Slice(slice)) => slice,
                    Some(Ty::Result(_)) => {
                        self.error(iter.span, "a region result is traversed with `parallel`/`ordered [p] in results:`, which rebinds its slices; it is not a scalar iterator");
                        return None;
                    }
                    _ => {
                        self.error(
                            iter.span,
                            "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice",
                        );
                        return None;
                    }
                };
                let (lo, hi) = self.root_domain(slice);
                let floor = self.vars.len();
                let moved_before = self.moved.clone();
                self.push_scope();
                let ty = if self.prover().nonneg(&lo) {
                    Ty::Index(hi.clone())
                } else {
                    Ty::Scalar(DType::I32)
                };
                let var = self.declare(&target.name, ty, target.span, VarKind::SliceMember(slice));
                let atom = var_atom(&target.name, var);
                self.facts
                    .set_range(atom.clone(), lo, hi.sub(&Sym::constant(1)));
                self.atoms.insert(var, atom);
                self.structural_loops.push(var);
                let body = self.block(body);
                self.structural_loops.pop();
                self.pop_scope();
                self.finish_loop_ownership(
                    moved_before,
                    floor,
                    LoopCardinality::RepeatedOrUnknown,
                    iter.span,
                );
                Some(StmtKind::Members { var, slice, body })
            }
            A::Call {
                callee,
                bindings,
                args,
            } if bindings.is_empty() => {
                if parallel {
                    self.error(iter.span, "`parallel for` currently requires an explicit bounded range such as `0..N`");
                    return None;
                }
                let A::Name(callee) = &callee.kind else {
                    self.error(
                        iter.span,
                        "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice",
                    );
                    return None;
                };
                match callee.name.as_str() {
                    "owned" | "axis" => {
                        self.coordinates(callee.name == "owned", targets, args, body, iter.span)
                    }
                    "lanes" => {
                        if self.target_form(iter.span, "`lanes`", None) {
                            self.error(iter.span, "`lanes` has no structured IR form: participant distribution belongs to the selected mapping, and target code reads its participant with the target's index intrinsic");
                        }
                        None
                    }
                    other => {
                        self.error(callee.span, format!("`{other}` is not an iterator; `for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice"));
                        None
                    }
                }
            }
            _ => {
                self.error(
                    iter.span,
                    "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice",
                );
                None
            }
        }
    }

    fn coordinates(
        &mut self,
        owned: bool,
        targets: &[ast::Ident],
        args: &[ast::Arg],
        body: &ast::Block,
        span: Span,
    ) -> Option<StmtKind> {
        if args.iter().any(|a| a.name.is_some()) || args.len() != if owned { 1 } else { 2 } {
            self.error(
                span,
                if owned {
                    "`owned(t)` takes one tile or view"
                } else {
                    "`axis(t, n)` takes a tile or view and a constant axis"
                },
            );
            return None;
        }
        let of = self.expr_inner(&args[0].value, None, true)?;
        let Some(shaped) = of.ty.shaped().cloned() else {
            self.error(
                of.span,
                format!(
                    "coordinate loops range over a tile or view, found {}",
                    of.ty
                ),
            );
            return None;
        };
        let axes: Vec<usize> = if owned {
            (0..shaped.rank()).collect()
        } else {
            let axis = self.expr(&args[1].value, Some(&Ty::Scalar(DType::I32)))?;
            match axis
                .sym
                .as_ref()
                .and_then(Sym::as_constant)
                .and_then(|a| usize::try_from(a).ok())
                .filter(|a| *a < shaped.rank())
            {
                Some(a) => vec![a],
                None => {
                    self.error(
                        axis.span,
                        format!("`axis` needs a constant axis below rank {}", shaped.rank()),
                    );
                    return None;
                }
            }
        };
        if axes.len() != targets.len() {
            self.error(
                span,
                format!(
                    "the loop ranges over {} axes but binds {} names",
                    axes.len(),
                    targets.len()
                ),
            );
            return None;
        }
        let floor = self.vars.len();
        let moved_before = self.moved.clone();
        self.push_scope();
        let mut vars = Vec::new();
        for (target, axis) in targets.iter().zip(&axes) {
            let (ty, lo, hi) = match &shaped.axes[*axis] {
                Extent::Semantic(extent) => {
                    (Ty::Index(extent.clone()), Sym::constant(0), extent.clone())
                }
                Extent::Structural(slice) => {
                    let (lo, hi) = self.root_domain(*slice);
                    (Ty::Coord(*slice), lo, hi)
                }
            };
            let var = self.declare(&target.name, ty, target.span, VarKind::Coordinate);
            let atom = var_atom(&target.name, var);
            self.facts
                .set_range(atom.clone(), lo, hi.sub(&Sym::constant(1)));
            self.atoms.insert(var, atom);
            vars.push(var);
        }
        // Definite assignment: an `owned` loop whose first write is `t[coordinates] = …`
        // initializes `t` and any uninitialized tile over identical axes.
        let saved = if owned {
            let pending: Vec<(VarId, Vec<VarId>)> = self
                .unassigned
                .iter()
                .copied()
                .filter(|v| matches!(&self.vars[*v].ty, Ty::Tile(s) if self.same_axes(s, &shaped)))
                .map(|v| (v, vars.clone()))
                .collect();
            Some(std::mem::replace(&mut self.pending_full_assign, pending))
        } else {
            None
        };
        let structural = axes
            .iter()
            .any(|axis| matches!(shaped.axes[*axis], Extent::Structural(_)));
        if structural {
            self.structural_loops.push(vars[0]);
        }
        let body = self.block(body);
        if structural {
            self.structural_loops.pop();
        }
        if let Some(saved) = saved {
            self.pending_full_assign = saved;
        }
        self.pop_scope();
        self.finish_loop_ownership(
            moved_before,
            floor,
            LoopCardinality::RepeatedOrUnknown,
            span,
        );
        Some(StmtKind::Coordinates {
            vars,
            of,
            axes,
            body,
        })
    }

    /// Join ownership across a loop's zero/one/back-edge control flow.
    ///
    /// Bindings declared in the body are fresh on every iteration. A captured
    /// owned binding, however, must reach a repeated back-edge initialized. A
    /// statically empty loop preserves the entry state, and a statically
    /// single-iteration loop carries its exit state forward.
    fn finish_loop_ownership(
        &mut self,
        moved_before: HashSet<VarId>,
        captured_floor: usize,
        cardinality: LoopCardinality,
        span: Span,
    ) {
        let moved_after = self.moved.clone();
        match cardinality {
            LoopCardinality::Zero => self.moved = moved_before,
            LoopCardinality::One => {
                self.moved = moved_after
                    .into_iter()
                    .filter(|id| *id < captured_floor)
                    .collect();
            }
            LoopCardinality::RepeatedOrUnknown => {
                let mut consumed: Vec<_> = moved_after
                    .difference(&moved_before)
                    .copied()
                    .filter(|id| *id < captured_floor)
                    .collect();
                consumed.sort_unstable();
                for id in consumed {
                    self.error(
                        span,
                        format!(
                            "loop may repeat after moving captured owned tensor `{}`; reinitialize it before the iteration ends",
                            self.vars[id].name
                        ),
                    );
                }
                // A valid repeated body has the same captured ownership state at
                // its back-edge as at entry. Body-local bindings do not escape.
                self.moved = moved_before;
            }
        }
    }

    /// Whether a place selects every element of its root (`t`, `t[:, :]`).
    fn covers_whole(&self, place: &Expr) -> bool {
        match &place.kind {
            ExprKind::Var(_) => true,
            ExprKind::Index { base, indices } => {
                indices.iter().all(|i| {
                    matches!(
                        i,
                        sir::Index::Range {
                            start: None,
                            end: None
                        }
                    )
                }) && self.covers_whole(base)
            }
            _ => false,
        }
    }

    // ---- result boundaries ----

    fn yielded(&mut self, values: &[ast::Expr], expected: Option<Vec<Ty>>) -> Option<Vec<Expr>> {
        let mut out = Vec::new();
        for (i, v) in values.iter().enumerate() {
            let hint = expected.as_ref().and_then(|tys| tys.get(i)).cloned();
            let e = self.value(v, hint.as_ref())?;
            if e.ty == Ty::Void {
                self.error(e.span, "`void` is not a value");
                return None;
            }
            out.push(e);
        }
        Some(out)
    }

    /// A value leaving its scope cannot carry a live slice handle, a coordinate, or a view of
    /// storage that dies with the scope.
    fn check_escape(&mut self, e: &Expr, floor: usize, boundary: &str) -> bool {
        fn handle(ty: &Ty) -> bool {
            match ty {
                Ty::Slice(_) | Ty::Coord(_) | Ty::Range(_) => true,
                Ty::Tuple(items) => items.iter().any(handle),
                _ => false,
            }
        }
        if handle(&e.ty) {
            self.error(e.span, format!("a slice is a scoped borrow of its region and cannot escape through {boundary}; region results keep the correspondence instead"));
            return false;
        }
        let parts: Vec<&Expr> = match &e.kind {
            ExprKind::Tuple(items) => items.iter().collect(),
            _ => vec![e],
        };
        for part in parts {
            if matches!(part.ty, Ty::View(_)) {
                if let Some(root) = self.root_var(part) {
                    if root >= floor && matches!(self.vars[root].ty, Ty::Tile(_)) {
                        let name = self.vars[root].name.clone();
                        self.error(part.span, format!("this view borrows local tile `{name}`, which dies with its scope; {boundary} a tile (`load`) instead"));
                        return false;
                    }
                }
            }
        }
        true
    }

    fn return_stmt(&mut self, values: &[ast::Expr], span: Span) -> Option<StmtKind> {
        if self.yields.iter().any(|y| y.loops > 0) {
            self.error(
                span,
                "`return` inside an element loop is not one result per path; return after the loop",
            );
            return None;
        }
        let expected: Vec<Ty> = match &self.sig.result {
            Ty::Void => Vec::new(),
            Ty::Tuple(items) if values.len() != 1 => items.clone(),
            other => vec![other.clone()],
        };
        if values.len() != expected.len() {
            self.error(
                span,
                format!(
                    "`{}` returns {} but this `return` has {} values",
                    self.sig.name,
                    self.sig.result,
                    values.len()
                ),
            );
            return None;
        }
        let exprs = self.yielded(values, Some(expected.clone()))?;
        for (e, ty) in exprs.iter().zip(&expected) {
            if !self.check_escape(e, self.sig.params.len(), "`return`") {
                return None;
            }
            if !self.assignable(ty, &e.ty) {
                self.error(
                    e.span,
                    format!(
                        "`{}` returns {ty} but this value is {}",
                        self.sig.name, e.ty
                    ),
                );
                return None;
            }
            if e.partial {
                self.error(e.span, "a partial-domain value cannot be returned as a whole-domain result; combine it with `merge` or an ordered traversal");
            }
        }
        if let Some(function) = self.yields.first_mut() {
            function.done = true;
        }
        // A `return` from a terminal root stage also ends that stage's path.
        if let Some(last) = self.yields.last_mut() {
            last.done = true;
        }
        Some(StmtKind::Return(exprs))
    }

}
