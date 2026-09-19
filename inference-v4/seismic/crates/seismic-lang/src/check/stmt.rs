//! Statements: bindings, state updates, regions and their results, stages and ports,
//! element/coordinate loops, publication, `yield`/`return` path rules.

use super::{elem_rounds, var_atom, Checker, Frame, FrameKind, YieldCtx, YieldKind};
use crate::span::Span;
use crate::sir::{self, Expr, ExprKind, RegionDecl, SliceDecl, SliceParent, Stmt, StmtKind, VarId, VarKind};
use crate::syntax::ast::{self, AssignOp, BinaryOp, ExprKind as A, RegionMode};
use crate::types::{DType, Extent, RegionId, ResultTy, SliceId, Ty};
use crate::sym::Sym;
use std::collections::HashSet;

impl<'a> Checker<'a> {
    pub fn push_scope(&mut self) {
        self.scopes.push(Default::default());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn block(&mut self, b: &ast::Block) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.stmts.len() {
            let s = &b.stmts[i];
            if self.yields.last().is_some_and(|y| y.done) {
                self.error(s.span, "unreachable statement: `yield`/`return` is terminal for its result boundary");
            }
            if matches!(s.kind, ast::StmtKind::Stage { .. }) {
                let end = b.stmts[i..].iter().position(|s| !matches!(s.kind, ast::StmtKind::Stage { .. })).map_or(b.stmts.len(), |n| i + n);
                let stages = self.stages(&b.stmts[i..end], end == b.stmts.len());
                out.push(Stmt { kind: StmtKind::Stages(stages), span: s.span.to(b.stmts[end - 1].span) });
                i = end;
                continue;
            }
            if let Some(kind) = self.stmt(s) {
                out.push(Stmt { kind, span: s.span });
            }
            i += 1;
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
            ast::StmtKind::Let { pattern, value } => self.bind(pattern, value, false),
            ast::StmtKind::Var { pattern, value } => self.bind(pattern, value, true),
            ast::StmtKind::Assign { target, op, value } => self.assign(target, *op, value),
            ast::StmtKind::Region(region) => self.region(region, false).map(|(region, _, _)| StmtKind::Region(region)),
            ast::StmtKind::Stage { .. } => None,
            ast::StmtKind::For { targets, iter, body } => {
                // State carried through a loop is not its pre-loop definition on every visit.
                self.dyn_slices.clear();
                if let Some(y) = self.yields.last_mut() {
                    y.loops += 1;
                }
                let kind = self.for_stmt(targets, iter, body);
                if let Some(y) = self.yields.last_mut() {
                    y.loops -= 1;
                }
                kind
            }
            ast::StmtKind::If { cond, then, els } => self.if_stmt(cond, then, els.as_ref()),
            ast::StmtKind::Publish { value, destination } => self.publish(value, destination, s.span),
            ast::StmtKind::Yield(values) => self.yield_stmt(values, s.span),
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

    /// The value of `let`/`var`/`yield`/`return`: the only positions of a result-producing region.
    fn value(&mut self, e: &ast::Expr, expected: Option<&Ty>) -> Option<Expr> {
        let A::Region(region) = &e.kind else { return self.expr(e, expected) };
        let (region, ty, partial) = self.region(region, true)?;
        Some(Expr { kind: ExprKind::Region(Box::new(region)), ty, sym: None, partial, span: e.span })
    }

    /// Partial flags of the components of a tuple-typed value.
    fn component_partials(&self, e: &Expr, n: usize) -> Vec<bool> {
        match &e.kind {
            ExprKind::Tuple(items) if items.len() == n => items.iter().map(|i| i.partial).collect(),
            ExprKind::Member { result, .. } => match &result.ty {
                Ty::Result(r) => self.result_partials.get(&r.producer).filter(|p| p.len() == n).cloned().unwrap_or_else(|| vec![e.partial; n]),
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
        let pattern = self.destructure(pattern, &value.ty.clone(), &value, state, origin)?;
        Some(StmtKind::Bind { pattern, value })
    }

    fn destructure(&mut self, pattern: &ast::Pattern, ty: &Ty, value: &Expr, state: bool, origin: Option<RegionId>) -> Option<sir::Pattern> {
        match pattern {
            ast::Pattern::Name(name) => {
                if state && matches!(ty, Ty::Tensor(_) | Ty::View(_) | Ty::Slice(_) | Ty::Coord(_) | Ty::Domain) {
                    self.error(name.span, format!("`var` declares mutable value state; a {ty} is a borrowed or structural handle and is bound with `let`"));
                    return None;
                }
                if matches!(ty, Ty::Coord(_)) {
                    self.error(name.span, "a tile coordinate cannot be rebound; it indexes tiles sharing its axis or is converted with `coord(i)`");
                    return None;
                }
                let id = self.declare(&name.name, ty.clone(), name.span, if state { VarKind::State } else { VarKind::Value });
                self.vars[id].partial = value.partial;
                if value.partial {
                    if let Some(origin) = origin {
                        self.partial_origin.insert(id, origin);
                    }
                }
                if matches!(value.kind, ExprKind::TileAlloc) {
                    self.unassigned.insert(id);
                }
                if matches!(ty, Ty::View(_) | Ty::Tensor(_)) {
                    if let Some(root) = self.root_var(value) {
                        self.view_roots.insert(id, root);
                        self.view_bound.insert(id, self.mutated.len());
                    }
                }
                if !state && ty.scalar_dtype() == Some(DType::I32) {
                    match (&value.sym, &value.kind) {
                        (Some(sym), _) => {
                            self.scalar_symbols.insert(id, sym.clone());
                        }
                        // A scalar argmax over a semantic axis is an index into that axis.
                        (None, ExprKind::Reduce { value: reduced, axis, op: sir::ReduceOp::Argmax, .. }) => {
                            if let Some(Extent::Semantic(extent)) = reduced.ty.shaped().and_then(|s| s.axes.get(*axis)).cloned() {
                                let atom = var_atom(&name.name, id);
                                self.facts.set_range(atom.clone(), Sym::constant(0), extent.sub(&Sym::constant(1)));
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
                    self.error(value.span, format!("a tuple pattern destructures a tuple; this value is a {ty}"));
                    return None;
                };
                if tys.len() != items.len() {
                    self.error(value.span, format!("pattern binds {} names but the value has {} components", items.len(), tys.len()));
                    return None;
                }
                let partials = self.component_partials(value, tys.len());
                let parts: Vec<Option<&Expr>> = match &value.kind {
                    ExprKind::Tuple(parts) if parts.len() == tys.len() => parts.iter().map(Some).collect(),
                    _ => vec![None; tys.len()],
                };
                let mut out = Vec::new();
                for (((item, ty), partial), part) in items.iter().zip(tys).zip(partials).zip(parts) {
                    let component = match part {
                        Some(part) => part.clone(),
                        None => Expr { kind: ExprKind::Tuple(Vec::new()), ty: ty.clone(), sym: None, partial, span: value.span },
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
                        self.error(place.span, "tuple assignment installs whole `var` state objects");
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
                        Expr { kind: ExprKind::Tuple(out), ty, sym: None, partial, span: value.span }
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
                    self.error(value.span, format!("tuple assignment expects {expected} but the value is {}", value.ty));
                    return None;
                }
                for place in &targets {
                    self.write(place, place.span, true)?;
                }
                let target = Expr { kind: ExprKind::Tuple(targets), ty: expected, sym: None, partial: false, span: target.span };
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
                        self.error(value.span, format!("`{}` has type {ty} but the value has type {}", name.name, value.ty));
                        return None;
                    }
                } else {
                    self.arith_assign(&ty, &value, op)?;
                }
                if !place.partial && !(op == AssignOp::Add && self.accumulates_into(&value)) {
                    self.forbid_partial(&value, "a state update");
                }
                let root = self.write(&place, target.span, true)?;
                self.unassigned.remove(&root);
                Some(StmtKind::Assign { target: place, op, value })
            }
            A::Index { base, .. } => {
                let A::Name(base_name) = &base.kind else {
                    self.error(base.span, "element assignment indexes a tile variable directly");
                    return None;
                };
                let Some(id) = self.lookup(&base_name.name) else {
                    self.error(base_name.span, format!("`{}` is not declared", base_name.name));
                    return None;
                };
                if !matches!(self.vars[id].ty, Ty::Tile(_)) {
                    self.error(target.span, format!("only tile elements are assigned; `{}` is a {}. `publish` is the only write to tensor storage", base_name.name, self.vars[id].ty));
                    return None;
                }
                let pending = self.pending_full_assign.iter().find(|(v, _)| *v == id).map(|(_, axes)| axes.clone());
                if self.unassigned.contains(&id) && pending.is_none() {
                    self.error(target.span, format!("`{}` is written element-wise before it is initialized; assign every element through `for … in owned(…)`", base_name.name));
                    return None;
                }
                let place = self.place(target)?;
                let covers = op == AssignOp::Assign && pending.as_ref().is_some_and(|axes| {
                    let ExprKind::Index { indices, .. } = &place.kind else { return false };
                    indices.len() == axes.len() && indices.iter().zip(axes).all(|(index, axis)| match index {
                        sir::Index::Coord(v) => v == axis,
                        sir::Index::Point(e) => matches!(e.kind, ExprKind::Var(v) if v == *axis),
                        _ => false,
                    })
                });
                if self.unassigned.contains(&id) && !covers {
                    self.error(target.span, format!("`{}` is written before initialization; its first `owned` write must cover every element (`{}[i, …] = …` at the loop's own coordinates)", base_name.name, base_name.name));
                    return None;
                }
                let Ty::Scalar(dtype) = place.ty else {
                    self.error(target.span, "an assignment target selects a single element; write views with `publish`");
                    return None;
                };
                let value = self.expr(value, Some(&Ty::Scalar(dtype)))?;
                let Some(vd) = value.ty.scalar_dtype() else {
                    self.error(value.span, format!("cannot assign {} to an element of dtype {}", value.ty, dtype.name()));
                    return None;
                };
                if op == AssignOp::Assign {
                    if !(vd == dtype || (vd.is_float() && dtype.is_float())) {
                        self.error(value.span, format!("cannot assign {} to an element of dtype {}; cast explicitly", vd.name(), dtype.name()));
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
                Some(StmtKind::Assign { target: place, op, value })
            }
            _ => {
                self.error(target.span, "an assignment target is `var` state, a tile element, or a tuple of state");
                None
            }
        }
    }

    fn state_place(&mut self, name: &ast::Ident) -> Option<Expr> {
        let Some(id) = self.lookup(&name.name) else {
            self.error(name.span, format!("`{}` is not declared; introduce state with `var`", name.name));
            return None;
        };
        let ty = self.vars[id].ty.clone();
        match self.vars[id].kind {
            VarKind::State => {}
            VarKind::Param(i) if self.sig.params[i].mode != ast::Mode::In && matches!(ty, Ty::Tile(_)) => {}
            VarKind::Slice(_) => {
                self.error(name.span, format!("slice `{}` is immutable geometry and cannot be reassigned", name.name));
                return None;
            }
            _ => {
                self.error(name.span, format!("`{}` is not mutable state; only `var` bindings and `out`/`inout` tiles are assigned", name.name));
                return None;
            }
        }
        if matches!(ty, Ty::Result(_) | Ty::Native(_)) {
            self.error(name.span, format!("a {ty} is immutable once produced and cannot be reassigned"));
            return None;
        }
        Some(Expr { kind: ExprKind::Var(id), ty, sym: None, partial: self.vars[id].partial, span: name.span })
    }

    fn accumulation_if(&mut self, op: AssignOp, value: &ast::Expr, place: &Expr) -> Option<Option<Expr>> {
        if op == AssignOp::Assign { self.accumulation(value, place) } else { None }
    }

    /// `s = s + p`, `s = max(s, p)`, `s = min(s, p)`: with `p` a forwarded partial of the
    /// result being traversed this is the one admitted accumulation of partials into state
    /// outside an `admit fn`. `None` when `value` does not have this form.
    fn accumulation(&mut self, value: &ast::Expr, place: &Expr) -> Option<Option<Expr>> {
        let ExprKind::Var(state) = place.kind else { return None };
        let is_state = |e: &ast::Expr, c: &Checker| matches!(&e.kind, A::Name(n) if c.lookup(&n.name) == Some(state));
        let (math, operands): (Option<sir::Math>, [&ast::Expr; 2]) = match &value.kind {
            A::Binary { op: BinaryOp::Add, lhs, rhs } => (None, [lhs, rhs]),
            A::Call { callee, bindings, args } if bindings.is_empty() && args.len() == 2 && args.iter().all(|a| a.name.is_none()) => match &callee.kind {
                A::Name(n) if n.name == "max" => (Some(sir::Math::Max), [&args[0].value, &args[1].value]),
                A::Name(n) if n.name == "min" => (Some(sir::Math::Min), [&args[0].value, &args[1].value]),
                _ => return None,
            },
            _ => return None,
        };
        let state_first = is_state(operands[0], self);
        if !state_first && !is_state(operands[1], self) {
            return None;
        }
        Some(self.accumulate(math, operands, state_first, place, value.span))
    }

    fn accumulate(&mut self, math: Option<sir::Math>, operands: [&ast::Expr; 2], state_first: bool, place: &Expr, span: Span) -> Option<Expr> {
        let mut other = self.expr(operands[usize::from(state_first)], Some(&place.ty))?;
        let state = self.expr(operands[usize::from(!state_first)], None)?;
        let admitted = other.partial && !self.partial_free();
        if admitted {
            if !self.accumulates_into(&other) {
                self.error(other.span, "a partial value accumulates into `var` state only inside a traversal of the result it belongs to (`ordered [p] in results:`)");
                return None;
            }
            // The accumulated state is whole once the traversal completes.
            other.partial = false;
        }
        let (lhs, rhs) = if state_first { (state, other) } else { (other, state) };
        match math {
            Some(op) => self.math_exprs(op, lhs, rhs, span),
            None => self.binary_exprs(BinaryOp::Add, lhs, rhs, span),
        }
    }

    /// Whether `e` is a forwarded partial of a result the current code is traversing.
    fn accumulates_into(&self, e: &Expr) -> bool {
        e.partial && self.partial_origin_of(e).is_some_and(|origin| self.frames.iter().any(|f| matches!(&f.kind, FrameKind::Region { origin: Some(o), .. } if *o == origin)))
    }

    fn arith_assign(&mut self, target: &Ty, value: &Expr, op: AssignOp) -> Option<()> {
        if let (Ty::Tile(a), Ty::Tile(b)) = (target, &value.ty) {
            if !self.same_axes(a, b) || !elem_rounds(&b.elem, &a.elem) {
                self.error(value.span, format!("`{}` needs tiles over identical axes: {} vs {}", op.text(), target, value.ty));
                return None;
            }
            return Some(());
        }
        if let (Ty::Tile(a), Some(v)) = (target, value.ty.scalar_dtype()) {
            if a.elem.read_dtype().is_none_or(|t| DType::promote(t, v).is_some()) {
                return Some(());
            }
        }
        let (Some(t), Some(v)) = (target.scalar_dtype(), value.ty.scalar_dtype()) else {
            self.error(value.span, format!("`{}` needs numeric operands, found {} and {}", op.text(), target, value.ty));
            return None;
        };
        if !t.is_numeric() || DType::promote(t, v).is_none() {
            self.error(value.span, format!("`{}` between {} and {} needs an explicit cast", op.text(), t.name(), v.name()));
            return None;
        }
        Some(())
    }

    // ---- control ----

    fn if_stmt(&mut self, cond: &ast::Expr, then: &ast::Block, els: Option<&ast::Block>) -> Option<StmtKind> {
        let cond = self.expr(cond, Some(&Ty::Scalar(DType::Bool)))?;
        match &cond.ty {
            Ty::Scalar(DType::Bool) => {}
            Ty::Tile(s) if s.elem == crate::types::Elem::Dtype(DType::Bool) => {
                self.error(cond.span, "a mask is a `bool` tile, not a scalar condition; there is no implicit reduction. Use `select(mask, a, b)`");
                return None;
            }
            other => {
                self.error(cond.span, format!("an `if` condition is a scalar `bool`, found {other}"));
                return None;
            }
        }
        self.forbid_partial(&cond, "a branch condition");
        let facts_before = self.facts.clone();
        let unassigned_before = self.unassigned.clone();
        let pending_before = self.pending_full_assign.clone();
        let symbols_before = self.scalar_symbols.clone();
        let done_before: Vec<bool> = self.yields.iter().map(|y| y.done).collect();

        self.assume(&cond, false);
        let then = self.scoped_block(then);
        self.facts = facts_before.clone();
        let unassigned_then = std::mem::replace(&mut self.unassigned, unassigned_before);
        let symbols_then = std::mem::replace(&mut self.scalar_symbols, symbols_before);
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
        let done_else: Vec<bool> = self.yields.iter().map(|y| y.done).collect();
        if done_then != done_else {
            self.error(cond.span, "one path of this `if` yields/returns and the other does not; every path through a result boundary produces exactly one value with one schema");
        }
        self.unassigned.extend(unassigned_then);
        self.scalar_symbols.retain(|id, sym| symbols_then.get(id) == Some(sym));
        self.pending_full_assign = pending_before.into_iter().filter(|(v, _)| self.unassigned.contains(v)).collect();
        Some(StmtKind::If { cond, then, els })
    }

    /// Path facts of a condition over symbolic integers (or of its negation).
    fn assume(&mut self, cond: &Expr, negate: bool) {
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            if let ExprKind::Unary { op: ast::UnaryOp::Not, expr } = &cond.kind {
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
        let (Some(l), Some(r)) = (&lhs.sym, &rhs.sym) else { return };
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

    /// A domain or range bound: a symbolic integer, or an immutable `i32` variable whose
    /// runtime value becomes a symbol.
    fn bound(&mut self, e: &ast::Expr) -> Option<(Expr, Sym)> {
        let checked = self.expr(e, Some(&Ty::Scalar(DType::I32)))?;
        if checked.ty.scalar_dtype() != Some(DType::I32) {
            self.error(checked.span, format!("a domain bound is an `i32`, found {}", checked.ty));
            return None;
        }
        self.forbid_partial(&checked, "a domain bound");
        if let Some(sym) = checked.sym.clone() {
            return Some((checked, sym));
        }
        if let ExprKind::Var(v) = checked.kind {
            if matches!(self.vars[v].kind, VarKind::Value | VarKind::Param(_) | VarKind::Port) {
                let atom = var_atom(&self.vars[v].name, v);
                self.atoms.insert(v, atom.clone());
                return Some((checked, Sym::atom(atom)));
            }
        }
        self.error(checked.span, "a domain bound is a symbolic integer; bind a runtime bound with `let` first so the domain names one value");
        None
    }

    fn for_stmt(&mut self, targets: &[ast::Ident], iter: &ast::Expr, body: &ast::Block) -> Option<StmtKind> {
        match &iter.kind {
            A::Range { lo, hi } => {
                let [target] = targets else {
                    self.error(iter.span, "a semantic range binds exactly one name");
                    return None;
                };
                let (lo, lo_sym) = self.bound(lo)?;
                let (hi, hi_sym) = self.bound(hi)?;
                self.push_scope();
                let ty = if lo_sym.is_zero() { Ty::Index(hi_sym.clone()) } else { Ty::Scalar(DType::I32) };
                let var = self.declare(&target.name, ty, target.span, VarKind::RangeIndex);
                let atom = var_atom(&target.name, var);
                self.facts.set_range(atom.clone(), lo_sym, hi_sym.sub(&Sym::constant(1)));
                self.atoms.insert(var, atom);
                let body = self.block(body);
                self.pop_scope();
                Some(StmtKind::Range { var, lo, hi, body })
            }
            A::Name(name) => {
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
                        self.error(iter.span, "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice");
                        return None;
                    }
                };
                let (lo, hi) = self.root_domain(slice);
                self.push_scope();
                let ty = if self.prover().nonneg(&lo) { Ty::Index(hi.clone()) } else { Ty::Scalar(DType::I32) };
                let var = self.declare(&target.name, ty, target.span, VarKind::SliceMember(slice));
                let atom = var_atom(&target.name, var);
                self.facts.set_range(atom.clone(), lo, hi.sub(&Sym::constant(1)));
                self.atoms.insert(var, atom);
                self.structural_loops.push(var);
                let body = self.block(body);
                self.structural_loops.pop();
                self.pop_scope();
                Some(StmtKind::Members { var, slice, body })
            }
            A::Call { callee, bindings, args } if bindings.is_empty() => {
                let A::Name(callee) = &callee.kind else {
                    self.error(iter.span, "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice");
                    return None;
                };
                match callee.name.as_str() {
                    "owned" | "axis" => self.coordinates(callee.name == "owned", targets, args, body, iter.span),
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
                self.error(iter.span, "`for` iterates `lo..hi`, `owned(t)`, `axis(t, n)` or a slice");
                None
            }
        }
    }

    fn coordinates(&mut self, owned: bool, targets: &[ast::Ident], args: &[ast::Arg], body: &ast::Block, span: Span) -> Option<StmtKind> {
        if args.iter().any(|a| a.name.is_some()) || args.len() != if owned { 1 } else { 2 } {
            self.error(span, if owned { "`owned(t)` takes one tile or view" } else { "`axis(t, n)` takes a tile or view and a constant axis" });
            return None;
        }
        let of = self.expr_inner(&args[0].value, None, true)?;
        let Some(shaped) = of.ty.shaped().cloned() else {
            self.error(of.span, format!("coordinate loops range over a tile or view, found {}", of.ty));
            return None;
        };
        let axes: Vec<usize> = if owned {
            (0..shaped.rank()).collect()
        } else {
            let axis = self.expr(&args[1].value, Some(&Ty::Scalar(DType::I32)))?;
            match axis.sym.as_ref().and_then(Sym::as_constant).and_then(|a| usize::try_from(a).ok()).filter(|a| *a < shaped.rank()) {
                Some(a) => vec![a],
                None => {
                    self.error(axis.span, format!("`axis` needs a constant axis below rank {}", shaped.rank()));
                    return None;
                }
            }
        };
        if axes.len() != targets.len() {
            self.error(span, format!("the loop ranges over {} axes but binds {} names", axes.len(), targets.len()));
            return None;
        }
        self.push_scope();
        let mut vars = Vec::new();
        for (target, axis) in targets.iter().zip(&axes) {
            let (ty, lo, hi) = match &shaped.axes[*axis] {
                Extent::Semantic(extent) => (Ty::Index(extent.clone()), Sym::constant(0), extent.clone()),
                Extent::Structural(slice) => {
                    let (lo, hi) = self.root_domain(*slice);
                    (Ty::Coord(*slice), lo, hi)
                }
            };
            let var = self.declare(&target.name, ty, target.span, VarKind::Coordinate);
            let atom = var_atom(&target.name, var);
            self.facts.set_range(atom.clone(), lo, hi.sub(&Sym::constant(1)));
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
        let structural = axes.iter().any(|axis| matches!(shaped.axes[*axis], Extent::Structural(_)));
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
        Some(StmtKind::Coordinates { vars, of, axes, body })
    }

    // ---- publication ----

    fn publish(&mut self, value: &ast::Expr, destination: &ast::Expr, span: Span) -> Option<StmtKind> {
        let destination = self.place(destination)?;
        let hint = destination.ty.scalar_dtype().map(Ty::Scalar);
        let value = self.expr(value, hint.as_ref())?;
        self.forbid_partial(&value, "a published value");
        let ok = match (&value.ty, &destination.ty) {
            (v, Ty::Scalar(d)) => v.scalar_dtype().is_some_and(|v| v == *d || (v.is_float() && d.is_float())),
            (Ty::Tile(v) | Ty::View(v) | Ty::Tensor(v), Ty::Tensor(d) | Ty::View(d) | Ty::Tile(d)) => {
                if matches!(d.elem, crate::types::Elem::Repr(_)) && v.elem != d.elem {
                    self.error(span, format!("publishing into packed `{}` storage needs an explicit encode operation", d.elem));
                    return None;
                }
                if !self.same_axes(v, d) {
                    self.error(span, format!("`publish` needs matching domains: value {} into destination {}", value.ty, destination.ty));
                    return None;
                }
                elem_rounds(&v.elem, &d.elem)
            }
            _ => false,
        };
        if !ok {
            self.error(span, format!("cannot publish {} to {}; convert explicitly", value.ty, destination.ty));
            return None;
        }
        let whole = matches!(destination.kind, ExprKind::Var(_)) && matches!(destination.ty, Ty::Tile(_));
        let root = self.write(&destination, span, false)?;
        if whole || self.covers_whole(&destination) {
            self.unassigned.remove(&root);
        }
        Some(StmtKind::Publish { value, destination })
    }

    /// Whether a place selects every element of its root (`t`, `t[:, :]`).
    fn covers_whole(&self, place: &Expr) -> bool {
        match &place.kind {
            ExprKind::Var(_) => true,
            ExprKind::Index { base, indices } => indices.iter().all(|i| matches!(i, sir::Index::Range { start: None, end: None })) && self.covers_whole(base),
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
                Ty::Slice(_) | Ty::Coord(_) | Ty::Domain => true,
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

    fn yield_stmt(&mut self, values: &[ast::Expr], span: Span) -> Option<StmtKind> {
        let checked = self.checked_yield(values, span);
        if checked.is_none() {
            if let Some(ctx) = self.yields.last_mut() {
                ctx.done = true;
                ctx.failed = true;
            }
        }
        checked
    }

    fn checked_yield(&mut self, values: &[ast::Expr], span: Span) -> Option<StmtKind> {
        let Some(ctx) = self.yields.last().cloned() else { return None };
        if ctx.kind == YieldKind::Function {
            self.error(span, "`yield` needs a structural consumer: a next-stage port, a result-producing region, or a `merge` body");
            return None;
        }
        if ctx.loops > 0 {
            self.error(span, "`yield` inside an element loop would produce a variable number of values; a visit yields exactly once");
            return None;
        }
        if values.is_empty() {
            self.error(span, "`yield` needs a value");
            return None;
        }
        let exprs = self.yielded(values, ctx.schema.clone())?;
        let floor = self.frames.last().map_or(0, |f| f.floor);
        for e in &exprs {
            if !self.check_escape(e, floor, "`yield`") {
                return None;
            }
        }
        let tys: Vec<Ty> = exprs.iter().map(|e| e.ty.clone()).collect();
        let partials: Vec<bool> = exprs.iter().map(|e| e.partial).collect();
        self.record_yield(tys, partials, span);
        Some(StmtKind::Yield(exprs))
    }

    fn record_yield(&mut self, tys: Vec<Ty>, partials: Vec<bool>, span: Span) {
        let Some(index) = self.yields.len().checked_sub(1) else { return };
        if self.yields[index].done {
            self.error(span, "this path already yielded; every path yields exactly once");
        }
        match self.yields[index].schema.clone() {
            Some(schema) => {
                if schema.len() != tys.len() || !schema.iter().zip(&tys).all(|(a, b)| self.same_ty(a, b)) {
                    let show = |t: &[Ty]| t.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
                    self.error(span, format!("every path yields one schema: ({}) here but ({}) earlier", show(&tys), show(&schema)));
                }
                for (flag, partial) in self.yields[index].partials.iter_mut().zip(&partials) {
                    *flag |= *partial;
                }
            }
            None => {
                self.yields[index].schema = Some(tys);
                self.yields[index].partials = partials;
            }
        }
        self.yields[index].done = true;
    }

    fn return_stmt(&mut self, values: &[ast::Expr], span: Span) -> Option<StmtKind> {
        if self.frames.iter().any(|f| !matches!(f.kind, FrameKind::Stage { pipeline: false })) {
            self.error(span, "`return` leaves the function; it cannot appear inside a region, a pipeline stage or a `merge` body");
            return None;
        }
        if self.yields.iter().any(|y| y.loops > 0) {
            self.error(span, "`return` inside an element loop is not one result per path; return after the loop");
            return None;
        }
        let expected: Vec<Ty> = match &self.sig.result {
            Ty::Void => Vec::new(),
            Ty::Tuple(items) if values.len() != 1 => items.clone(),
            other => vec![other.clone()],
        };
        if values.len() != expected.len() {
            self.error(span, format!("`{}` returns {} but this `return` has {} values", self.sig.name, self.sig.result, values.len()));
            return None;
        }
        let exprs = self.yielded(values, Some(expected.clone()))?;
        for (e, ty) in exprs.iter().zip(&expected) {
            if !self.check_escape(e, self.sig.params.len(), "`return`") {
                return None;
            }
            if !self.assignable(ty, &e.ty) {
                self.error(e.span, format!("`{}` returns {ty} but this value is {}", self.sig.name, e.ty));
                return None;
            }
            if e.partial && !self.admit {
                self.error(e.span, "a partial-domain value cannot be returned as a whole-domain result; combine it with `merge`, an ordered traversal, or an `admit fn`");
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

    // ---- regions ----

    fn new_slice(&mut self, binder: &ast::Ident, region: RegionId, parent: SliceParent) -> (VarId, SliceId) {
        let slice = SliceId(self.slices.len() as u32);
        let var = self.declare(&binder.name, Ty::Slice(slice), binder.span, VarKind::Slice(slice));
        self.slices.push(SliceDecl { var, region, parent });
        (var, slice)
    }

    /// Returns the region, the type of the region expression, and whether that value is partial.
    fn region(&mut self, r: &ast::Region, as_expr: bool) -> Option<(sir::Region, Ty, bool)> {
        if as_expr && r.mode == RegionMode::Pipeline {
            self.error(r.span, "`pipeline` is never a result expression; its terminal stage updates state or publishes");
            return None;
        }
        if r.merge.is_some() && (r.mode != RegionMode::Parallel || !as_expr) {
            self.error(r.span, "`merge` combines the partials of a `parallel` region expression");
            return None;
        }
        if r.mode == RegionMode::Parallel && self.frames.iter().any(|f| matches!(f.kind, FrameKind::Stage { pipeline: true })) {
            self.error(r.span, "a pipeline stage cannot create new concurrent owners; use `ordered` refinement or restructure the enclosing regions");
            return None;
        }
        let id = RegionId(self.regions.len() as u32);
        let parent = self.frames.iter().rev().find_map(|f| match &f.kind {
            FrameKind::Region { id, .. } => Some(*id),
            _ => None,
        });
        self.regions.push(RegionDecl { mode: r.mode, binders: Vec::new(), parent, span: r.span });

        // Sources are evaluated in the enclosing scope, before the binders exist.
        enum Source {
            Parents(Vec<SliceParent>),
            Results(Expr, ResultTy),
        }
        let single_result = match r.sources.as_slice() {
            [source] if !matches!(source.kind, A::Range { .. }) => {
                let e = self.expr(source, None)?;
                match e.ty.clone() {
                    Ty::Result(result) => Some(Source::Results(e, *result)),
                    Ty::Slice(slice) => Some(Source::Parents(vec![SliceParent::Refine(slice)])),
                    other => {
                        self.error(source.span, format!("a region ranges over a domain `lo..hi`, an enclosing slice, or a region result; found {other}"));
                        return None;
                    }
                }
            }
            _ => None,
        };
        let source = match single_result {
            Some(source) => source,
            None => {
                let mut parents = Vec::new();
                for source in &r.sources {
                    match &source.kind {
                        A::Range { lo, hi } => {
                            let (_, lo) = self.bound(lo)?;
                            let (_, hi) = self.bound(hi)?;
                            let static_bounds = lo.params().iter().chain(hi.params().iter()).all(|p| !p.contains('#'));
                            if static_bounds {
                                self.require_nonneg(&lo, source.span, "domain start may be negative");
                                self.require_nonneg(&hi.sub(&lo), source.span, "domain may be reversed");
                            }
                            parents.push(SliceParent::Domain { lo, hi });
                        }
                        _ => match self.expr(source, None)?.ty {
                            Ty::Slice(slice) => parents.push(SliceParent::Refine(slice)),
                            Ty::Result(_) => {
                                self.error(source.span, "a region result is traversed alone, with its own binder arity; it cannot be one member of a product");
                                return None;
                            }
                            other => {
                                self.error(source.span, format!("a region ranges over a domain `lo..hi` or an enclosing slice; found {other}"));
                                return None;
                            }
                        },
                    }
                }
                Source::Parents(parents)
            }
        };

        let floor = self.vars.len();
        self.push_scope();
        let mut binders = Vec::new();
        let mut binder_slices = Vec::new();
        let (region_source, origin, rebound) = match source {
            Source::Parents(parents) => {
                if parents.len() != r.binders.len() {
                    self.error(r.span, format!("{} binders for {} domains; there is one domain per bound name", r.binders.len(), parents.len()));
                    self.pop_scope();
                    return None;
                }
                for (binder, parent) in r.binders.iter().zip(parents) {
                    let (var, slice) = self.new_slice(binder, id, parent);
                    binders.push(var);
                    binder_slices.push(slice);
                }
                (sir::RegionSource::Domains, None, Vec::new())
            }
            Source::Results(e, result) => {
                if result.binders.len() != r.binders.len() {
                    self.error(r.span, format!("this result was produced over {} binders and is traversed with exactly that arity, not {}; results are never flattened or regrouped", result.binders.len(), r.binders.len()));
                    self.pop_scope();
                    return None;
                }
                let mut rebound = Vec::new();
                for (binder, original) in r.binders.iter().zip(&result.binders) {
                    let (var, slice) = self.new_slice(binder, id, SliceParent::Rebind(*original));
                    binders.push(var);
                    binder_slices.push(slice);
                    rebound.push((slice, *original));
                }
                (sir::RegionSource::Results(Box::new(e)), Some((result.origin, result.binders.clone())), rebound)
            }
        };
        self.regions[id.0 as usize].binders = binder_slices.clone();

        self.frames.push(Frame { kind: FrameKind::Region { id, mode: r.mode, origin: origin.as_ref().map(|(o, _)| *o) }, floor });
        self.yields.push(YieldCtx { kind: YieldKind::Region, done: false, loops: 0, schema: None, partials: Vec::new(), failed: false });
        if r.mode == RegionMode::Pipeline {
            if let Some(other) = r.body.stmts.iter().find(|s| !matches!(s.kind, ast::StmtKind::Stage { .. })) {
                self.error(other.span, "a `pipeline` body consists only of stages");
            }
        }
        let body = self.block(&r.body);
        let ctx = self.yields.pop();
        self.frames.pop();
        self.pop_scope();
        let ctx = ctx?;

        if !as_expr {
            if ctx.schema.is_some() {
                self.error(r.span, "a statement region cannot collect or discard yielded members; bind it (`let results = …`) or attach a `merge`");
            }
            return Some((sir::Region { id, mode: r.mode, binders, source: region_source, body, merge: None, result: None }, Ty::Void, false));
        }
        if ctx.failed {
            return None;
        }
        let Some(schema) = ctx.schema.filter(|_| ctx.done) else {
            self.error(r.span, "a result-producing region yields exactly one value on every path through its body");
            return None;
        };
        let member = if schema.len() == 1 { schema[0].clone() } else { Ty::Tuple(schema.clone()) };
        // A yielded component is pointwise over the visit only if it is shaped by every binder;
        // anything else is one aggregate per tuned piece.
        let pointwise = |ty: &Ty| match ty {
            Ty::Tile(s) | Ty::View(s) => binder_slices.iter().all(|b| s.axes.contains(&Extent::Structural(*b))),
            Ty::Result(_) => true,
            _ => false,
        };
        let partials: Vec<bool> = schema.iter().zip(&ctx.partials).map(|(ty, partial)| *partial || !pointwise(ty)).collect();

        if let Some(merge) = &r.merge {
            let merge = self.merge(merge, &member, &partials)?;
            let region = sir::Region { id, mode: r.mode, binders, source: region_source, body, merge: Some(merge), result: None };
            return Some((region, member, false));
        }
        self.result_partials.insert(id, partials);
        let (origin, origin_binders) = origin.unwrap_or((id, binder_slices));
        // The member schema refers to the origin's binders, whoever traverses them.
        let member = self.rebind_ty(&member, &rebound);
        let ty = Ty::Result(Box::new(ResultTy { origin, producer: id, binders: origin_binders, member }));
        let region = sir::Region { id, mode: r.mode, binders, source: region_source, body, merge: None, result: Some(ty.clone()) };
        Some((region, ty, false))
    }

    fn merge(&mut self, m: &ast::Merge, member: &Ty, partials: &[bool]) -> Option<sir::Merge> {
        let floor = self.vars.len();
        self.frames.push(Frame { kind: FrameKind::Merge, floor });
        self.push_scope();
        let identity = self.expr(&m.identity, Some(member));
        let result = identity.and_then(|identity| {
            if !self.assignable(member, &identity.ty) {
                self.error(identity.span, format!("the `merge` identity must have the partial type {member}, found {}", identity.ty));
                return None;
            }
            let left = self.merge_operand(&m.left, member, partials, m.span)?;
            let right = self.merge_operand(&m.right, member, partials, m.span)?;
            let schema = match member {
                Ty::Tuple(items) => items.clone(),
                other => vec![other.clone()],
            };
            self.yields.push(YieldCtx { kind: YieldKind::Merge, done: false, loops: 0, schema: Some(schema), partials: vec![false; partials.len()], failed: false });
            let body = self.block(&m.body);
            let ctx = self.yields.pop()?;
            if ctx.failed {
                return None;
            }
            if !ctx.done {
                self.error(m.span, format!("a `merge` body yields the combined partial ({member}) on every path"));
                return None;
            }
            Some(sir::Merge { left, right, identity, body })
        });
        self.pop_scope();
        self.frames.pop();
        result
    }

    fn merge_operand(&mut self, pattern: &ast::Pattern, member: &Ty, partials: &[bool], span: Span) -> Option<sir::Pattern> {
        match (pattern, member) {
            (ast::Pattern::Name(name), _) => {
                let id = self.declare(&name.name, member.clone(), name.span, VarKind::MergeOperand);
                self.vars[id].partial = partials.iter().any(|p| *p);
                Some(sir::Pattern::Var(id))
            }
            (ast::Pattern::Tuple(items), Ty::Tuple(tys)) if items.len() == tys.len() => {
                let mut out = Vec::new();
                for (i, (item, ty)) in items.iter().zip(tys).enumerate() {
                    let flags = [partials.get(i).copied().unwrap_or(true)];
                    out.push(self.merge_operand(item, ty, &flags, span)?);
                }
                Some(sir::Pattern::Tuple(out))
            }
            _ => {
                self.error(span, format!("the `merge` operand pattern does not match the partial type {member}"));
                None
            }
        }
    }

    // ---- stages ----

    fn stages(&mut self, run: &[ast::Stmt], ends_block: bool) -> Vec<sir::Stage> {
        let pipeline = matches!(self.frames.last(), Some(Frame { kind: FrameKind::Region { mode: RegionMode::Pipeline, .. }, .. }));
        let pipeline_floor = self.frames.last().map_or(0, |f| f.floor);
        let mut out = Vec::new();
        let mut previous: Option<(Vec<Ty>, Vec<bool>)> = None;
        let mut updated: Vec<(VarId, String)> = Vec::new();
        let mut read: Vec<(HashSet<VarId>, String)> = Vec::new();
        for (ordinal, s) in run.iter().enumerate() {
            let ast::StmtKind::Stage { name, ports, body } = &s.kind else { continue };
            let last = ordinal + 1 == run.len();
            let floor = self.vars.len();
            self.push_scope();
            self.frames.push(Frame { kind: FrameKind::Stage { pipeline }, floor });
            let (tys, partials) = previous.take().unwrap_or_default();
            if tys.len() != ports.len() {
                let message = if ordinal == 0 {
                    format!("stage `{}` is first in its chain and has no predecessor to bind {} ports", name.name, ports.len())
                } else {
                    format!("stage `{}` binds {} ports but the previous stage yields {} values; ports bind the previous `yield` positionally", name.name, ports.len(), tys.len())
                };
                self.error(name.span, message);
            }
            let mut port_vars = Vec::new();
            for (i, port) in ports.iter().enumerate() {
                let Some(ty) = tys.get(i) else { break };
                let id = self.declare(&port.name, ty.clone(), port.span, VarKind::Port);
                self.vars[id].partial = partials.get(i).copied().unwrap_or(false);
                port_vars.push(id);
            }
            self.yields.push(YieldCtx { kind: YieldKind::Stage, done: false, loops: 0, schema: None, partials: Vec::new(), failed: false });
            let mutated_from = self.mutated.len();
            let reads_from = self.reads.len();
            let checked = self.block(body);
            let ctx = self.yields.pop();
            self.frames.pop();
            self.pop_scope();

            if pipeline {
                read.push((self.reads[reads_from..].iter().copied().filter(|v| *v < pipeline_floor && self.vars[*v].kind == VarKind::State).collect(), name.name.clone()));
                let states: HashSet<VarId> = self.mutated[mutated_from..].iter().copied().filter(|v| *v < pipeline_floor && self.vars[*v].kind == VarKind::State).collect();
                for state in states {
                    match updated.iter().find(|(v, _)| *v == state) {
                        Some((_, other)) => {
                            let state_name = self.vars[state].name.clone();
                            self.error(name.span, format!("state `{state_name}` is updated by stages `{other}` and `{}`; in a `pipeline` each state object has exactly one updating stage", name.name));
                        }
                        None => updated.push((state, name.name.clone())),
                    }
                }
            }
            if let Some(ctx) = ctx {
                match (ctx.schema, ctx.done, last) {
                    (Some(schema), true, false) => previous = Some((schema, ctx.partials)),
                    _ if ctx.failed => {}
                    (Some(_), false, _) => self.error(name.span, format!("stage `{}` yields on some paths only; every path yields its ports exactly once", name.name)),
                    (Some(schema), true, true) => {
                        // The terminal stage of a chain that ends a result-producing visit
                        // supplies that visit's result.
                        let region_visit = self.yields.last().is_some_and(|y| y.kind == YieldKind::Region) && !pipeline;
                        let returned = self.yields.first().is_some_and(|y| y.done) && self.yields.len() == 1;
                        if region_visit && ends_block {
                            self.record_yield(schema, ctx.partials, name.span);
                        } else if !returned {
                            self.error(name.span, format!("the `yield` of terminal stage `{}` has no structural consumer: no next stage binds it and the chain does not end a result-producing visit", name.name));
                        }
                    }
                    (None, _, _) => {}
                }
            }
            out.push(sir::Stage { name: name.name.clone(), ports: port_vars, body: checked, span: s.span });
        }
        // Stages of different visits overlap: only the updating stage may access its state.
        for (state, updater) in &updated {
            if let Some((_, reader)) = read.iter().find(|(states, reader)| reader != updater && states.contains(state)) {
                let state_name = self.vars[*state].name.clone();
                let span = run.first().map_or(Span::default(), |s| s.span);
                self.error(span, format!("stage `{reader}` reads state `{state_name}`, which stage `{updater}` updates; pipeline visits overlap, so only the updating stage may access that state"));
            }
        }
        out
    }
}
