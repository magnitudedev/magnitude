//! Statements: bindings, state updates, loops, branches, and return paths —
//! emitted as the section-5 checked statement set.

use super::ir::{
    Block as CheckedBlock, Expr as CheckedExpr, ExprKind as CheckedExprKind, Index as CheckedIndex,
    LoopKind, Ownership as ParamOwnership, Pattern, Place as CheckedPlace, Stmt as CheckedStmt,
    Terminator as BlockTerminator,
};
use super::{elem_rounds, Checker, LocalKind, ValueClass};
use crate::span::Span;
use crate::syntax::ast::{self, AssignOp, BinaryOp, ExprKind as A};
use crate::types::{DType, Elem, ValueType};
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

    /// Check one source block into a checked block with its terminator.
    pub fn block(&mut self, b: &ast::Block) -> CheckedBlock {
        let mut statements = Vec::new();
        let mut terminator = BlockTerminator::Continue;
        let mut terminal = false;
        for s in &b.stmts {
            if terminal {
                self.error(
                    s.span,
                    "unreachable statement: `return` is terminal for the function boundary",
                );
                continue;
            }
            match self.stmt(s) {
                Some(PartialStmt::Terminal(values)) => {
                    terminator = BlockTerminator::Return(values);
                    terminal = true;
                }
                Some(PartialStmt::Statement(CheckedStmt::If {
                    condition,
                    then_body,
                    else_body,
                })) => {
                    // An `if` whose arms both return ends this path; the arms
                    // carry their own result values.
                    let both = matches!(then_body.terminator, BlockTerminator::Return(_))
                        && matches!(else_body.terminator, BlockTerminator::Return(_));
                    statements.push(CheckedStmt::If {
                        condition,
                        then_body,
                        else_body,
                    });
                    if both {
                        terminal = true;
                    }
                }
                Some(PartialStmt::Statement(other)) => statements.push(other),
                None => {}
            }
        }
        if terminal && matches!(terminator, BlockTerminator::Continue) {
            terminator = BlockTerminator::Return(Vec::new());
        }
        CheckedBlock {
            statements,
            terminator,
        }
    }

    fn scoped_block(&mut self, b: &ast::Block) -> CheckedBlock {
        self.push_scope();
        let out = self.block(b);
        self.pop_scope();
        out
    }

    fn stmt(&mut self, s: &ast::Stmt) -> Option<PartialStmt> {
        match &s.kind {
            ast::StmtKind::Let {
                mutable,
                pattern,
                value,
            } => self
                .bind(pattern, value, *mutable)
                .map(PartialStmt::Statement),
            ast::StmtKind::Assign { target, op, value } => {
                self.assign(target, *op, value).map(PartialStmt::Statement)
            }
            ast::StmtKind::For {
                parallel,
                targets,
                iter,
                body,
            } => {
                self.dyn_views.clear();
                let kind = self.for_stmt(*parallel, targets, iter, body);
                kind.map(PartialStmt::Statement)
            }
            ast::StmtKind::If { cond, then, els } => self
                .if_stmt(cond, then, els.as_ref())
                .map(PartialStmt::Statement),
            ast::StmtKind::Return(values) => self.return_stmt(values, s.span),
            ast::StmtKind::Expr(e) => {
                let e = self.expr(e, None)?;
                if !e.ty.is_void() {
                    self.error(e.span, format!("an expression statement is a call evaluated for its `inout` effects; this value of type {} is unused", e.ty));
                }
                Some(PartialStmt::Statement(CheckedStmt::Evaluate(e)))
            }
        }
    }

    // ---- bindings ----

    fn poison(&mut self, pattern: &ast::Pattern) {
        match pattern {
            ast::Pattern::Name(name) => {
                self.poisoned.insert(name.name.clone());
            }
            ast::Pattern::Tuple(items) => items.iter().for_each(|item| self.poison(item)),
        }
    }

    fn bind(
        &mut self,
        pattern: &ast::Pattern,
        value: &ast::Expr,
        state: bool,
    ) -> Option<CheckedStmt> {
        let value = self.expr(value, None)?;
        if value.ty.is_void() {
            self.error(value.span, "cannot bind a call that returns nothing");
            self.poison(pattern);
            return None;
        }
        let moved = (matches!(value.ty, ValueType::Tensor(_))
            && matches!(self.class_of(&value), ValueClass::Owned))
        .then(|| self.root_var(&value))
        .flatten();
        let pattern = self.destructure(pattern, &value.ty.clone(), &value, state)?;
        if let Some(root) = moved {
            self.moved.insert(root);
        }
        Some(CheckedStmt::Let {
            pattern,
            mutable: state,
            value,
        })
    }

    fn destructure(
        &mut self,
        pattern: &ast::Pattern,
        ty: &ValueType,
        value: &CheckedExpr,
        state: bool,
    ) -> Option<Pattern> {
        match pattern {
            ast::Pattern::Name(name) => {
                let id = self.declare(
                    &name.name,
                    ty.clone(),
                    name.span,
                    if state {
                        LocalKind::State
                    } else {
                        LocalKind::Value
                    },
                    state,
                );
                if matches!(
                    value.kind,
                    CheckedExprKind::Primitive {
                        id: crate::intrinsics::PrimitiveId::TensorAlloc,
                        ..
                    }
                ) {
                    self.unassigned.insert(id);
                }
                // A view binding borrows the storage it selects.
                if matches!(self.class_of(value), ValueClass::Borrowed) {
                    if let Some(root) = self.root_var(value) {
                        let writable = self.writable_root(root);
                        let exclusive = state && writable;
                        if self.borrows.values().any(|(borrowed, prior_exclusive)| {
                            *borrowed == root && (exclusive || *prior_exclusive)
                        }) {
                            self.error(
                                name.span,
                                format!(
                                    "tensor borrow of `{}` overlaps a live {} borrow",
                                    self.locals[root.index()].name,
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
                    match value.sym {
                        Some(sym) => {
                            self.scalar_symbols.insert(id, sym);
                        }
                        None => {
                            // A scalar argmax over an axis is an index into that axis.
                            if let CheckedExprKind::Primitive {
                                id:
                                    crate::intrinsics::PrimitiveId::Reduce {
                                        op: crate::intrinsics::ReduceOp::Argmax,
                                        axis,
                                        ..
                                    },
                                operands,
                            } = &value.kind
                            {
                                if let Some(extent) = operands
                                    .first()
                                    .and_then(|o| o.ty.shaped())
                                    .and_then(|s| s.axes.get(*axis as usize))
                                {
                                    let (_, symbol, _) = self.arena.loop_binder();
                                    let zero = self.arena.int(0);
                                    let one = self.arena.int(1);
                                    let upper = self.arena.int_sub(*extent, one);
                                    self.facts.set_range(symbol, zero, upper);
                                    self.symbols.insert(id, symbol);
                                    self.locals[id.index()].symbol = Some(symbol);
                                }
                            }
                        }
                    }
                }
                Some(Pattern::Local(id))
            }
            ast::Pattern::Tuple(items) => {
                let ValueType::Tuple(tys) = ty else {
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
                let parts: Vec<Option<CheckedExpr>> = match &value.kind {
                    CheckedExprKind::Primitive {
                        id: crate::intrinsics::PrimitiveId::TuplePack,
                        operands,
                    } if operands.len() == tys.len() => {
                        operands.iter().cloned().map(Some).collect()
                    }
                    _ => vec![None; tys.len()],
                };
                let mut out = Vec::new();
                for (i, ((item, ty), part)) in items.iter().zip(tys.iter()).zip(parts).enumerate() {
                    let component = part.unwrap_or_else(|| {
                        CheckedExpr::new(
                            CheckedExprKind::Primitive {
                                id: crate::intrinsics::PrimitiveId::TupleGet(i as u32),
                                operands: vec![value.clone()],
                            },
                            ty.clone(),
                            None,
                            value.span,
                        )
                    });
                    out.push(self.destructure(item, ty, &component, state)?);
                }
                Some(Pattern::Tuple(out))
            }
        }
    }

    // ---- state updates ----

    fn assign(
        &mut self,
        target: &ast::Expr,
        op: AssignOp,
        value: &ast::Expr,
    ) -> Option<CheckedStmt> {
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
                    targets.push((
                        self.state_place(name)?,
                        self.locals[self.lookup(&name.name)?.index()].ty.clone(),
                    ));
                }
                let expected = ValueType::Tuple(
                    crate::types::NonEmpty::new(targets.iter().map(|(_, t)| t.clone()).collect())
                        .expect("tuple assignment has components"),
                );
                // All right-hand sides read the old versions.
                let value = match &value.kind {
                    A::Tuple(parts) if parts.len() == targets.len() => {
                        let mut out = Vec::new();
                        for (part, (_, ty)) in parts.iter().zip(&targets) {
                            let component = self.expr(part, Some(ty))?;
                            out.push(component);
                        }
                        let tys: Vec<ValueType> = out.iter().map(|e| e.ty.clone()).collect();
                        CheckedExpr::new(
                            CheckedExprKind::Primitive {
                                id: crate::intrinsics::PrimitiveId::TuplePack,
                                operands: out.clone(),
                            },
                            ValueType::Tuple(crate::types::NonEmpty::new(tys).expect("components")),
                            None,
                            value.span,
                        )
                    }
                    _ => self.expr(value, Some(&expected))?,
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
                let mut authorities = Vec::new();
                for (place, _) in &targets {
                    let root = self.write_place(place, true, target.span)?;
                    authorities.extend(self.exclusive_write_authority(root, place));
                }
                let places = targets.into_iter().map(|(p, _)| p).collect();
                Some(CheckedStmt::Assign {
                    place: CheckedPlace::Tuple(places),
                    op,
                    value,
                    authorities,
                })
            }
            A::Name(name) => {
                let place = self.state_place(name)?;
                let ty = match &place {
                    CheckedPlace::Local(root) => self.locals[root.index()].ty.clone(),
                    _ => unreachable!(),
                };
                let value = self.expr(value, Some(&ty))?;
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
                let moved = (op == AssignOp::Assign
                    && matches!(value.ty, ValueType::Tensor(_))
                    && matches!(self.class_of(&value), ValueClass::Owned))
                .then(|| self.root_var(&value))
                .flatten();
                let root = self.write_place(&place, true, target.span)?;
                if op == AssignOp::Assign && matches!(ty, ValueType::Tensor(_)) {
                    // Assignment reinitializes owned state, including state that
                    // was moved earlier on this path.
                    self.moved.remove(&root);
                }
                if let Some(source) = moved.filter(|source| *source != root) {
                    self.moved.insert(source);
                }
                self.unassigned.remove(&root);
                let authorities = self.exclusive_write_authority(root, &place);
                Some(CheckedStmt::Assign {
                    place,
                    op,
                    value,
                    authorities,
                })
            }
            A::Index { .. } => {
                let (root, indices, selected) = self.place(target)?;
                let binding = match &target.kind {
                    A::Index { base, .. } => match &base.kind {
                        A::Name(name) => self.lookup(&name.name),
                        _ => None,
                    },
                    _ => None,
                };
                let Some(binding) = binding else {
                    self.error(
                        target.span,
                        "element assignment indexes a tensor variable directly",
                    );
                    return None;
                };
                if !self.writable_root(self.root_var_local(binding)) {
                    self.error(target.span, format!("`{}` is not writable storage; writing requires `let mut` state or a `&mut tensor` parameter", self.locals[binding.index()].name));
                    return None;
                }
                // Packed representations are readable and decodable but never
                // writable; no reachable interpreter or emitter panic remains.
                if let Some(shaped) = self.locals[binding.index()].ty.shaped() {
                    if matches!(shaped.elem, Elem::Repr(_)) {
                        self.error(
                            target.span,
                            "packed representations are readable and decodable but not writable",
                        );
                        return None;
                    }
                }
                let pending = self
                    .pending_full_assign
                    .iter()
                    .find(|(v, _)| *v == root)
                    .map(|(_, axes)| axes.clone());
                if self.unassigned.contains(&root) && pending.is_none() && self.init_loop_depth == 0
                {
                    self.error(target.span, format!("`{}` is written element-wise before it is initialized; assign every element through a complete `for … in 0..extent` traversal", self.locals[root.index()].name));
                    return None;
                }
                let covers = op == AssignOp::Assign
                    && pending.as_ref().is_some_and(|axes| {
                        indices.len() == axes.len()
                            && indices.iter().zip(axes).all(|(index, axis)| match index {
                                CheckedIndex::Point { value: p, .. } => {
                                    matches!(&p.kind, CheckedExprKind::Local(v) if v == axis)
                                }
                                _ => false,
                            })
                    });
                match &selected {
                    ValueType::Scalar(dtype) => {
                        let dtype = *dtype;
                        let value = self.expr(value, Some(&ValueType::Scalar(dtype)))?;
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
                            self.arith_assign(&ValueType::Scalar(dtype), &value, op)?;
                        }
                        self.write(root, binding, &indices, false, target.span)?;
                        let checked_place = CheckedPlace::Element {
                            root,
                            indices: indices.clone(),
                        };
                        let authorities = self.exclusive_write_authority(root, &checked_place);
                        if covers {
                            self.unassigned.remove(&root);
                            self.pending_full_assign
                                .retain(|(variable, _)| *variable != root);
                        }
                        Some(CheckedStmt::Assign {
                            place: checked_place,
                            op,
                            value,
                            authorities,
                        })
                    }
                    ValueType::Tensor(_) => {
                        if op != AssignOp::Assign {
                            self.error(
                                target.span,
                                "compound assignment requires a scalar element place",
                            );
                            return None;
                        }
                        let value = self.expr(value, None)?;
                        let compatible = match (&value.ty, &selected) {
                            (ValueType::Tensor(v), ValueType::Tensor(d)) => {
                                self.same_axes(v, d) && elem_rounds(&v.elem, &d.elem)
                            }
                            _ => false,
                        };
                        if !compatible {
                            self.error(
                                value.span,
                                format!(
                                    "cannot assign {} to {}; convert explicitly",
                                    value.ty, selected
                                ),
                            );
                            return None;
                        }
                        self.write(root, binding, &indices, false, target.span)?;
                        let checked_place = CheckedPlace::Element {
                            root,
                            indices: indices.clone(),
                        };
                        let authorities = self.exclusive_write_authority(root, &checked_place);
                        if covers
                            || indices.iter().all(|i| {
                                matches!(
                                    i,
                                    CheckedIndex::Range {
                                        start: None,
                                        end: None,
                                        ..
                                    }
                                )
                            })
                        {
                            self.unassigned.remove(&root);
                            self.pending_full_assign
                                .retain(|(variable, _)| *variable != root);
                        }
                        Some(CheckedStmt::Assign {
                            place: checked_place,
                            op,
                            value,
                            authorities,
                        })
                    }
                    other => {
                        self.error(
                            target.span,
                            format!("cannot assign into a place of type {other}"),
                        );
                        None
                    }
                }
            }
            _ => {
                self.error(
                    target.span,
                    "an assignment target is `let mut` state, a tensor element, or a tuple of state",
                );
                None
            }
        }
    }

    /// Record a whole-variable write through a place.
    fn write_place(
        &mut self,
        place: &CheckedPlace,
        whole: bool,
        span: Span,
    ) -> Option<super::ir::LocalId> {
        match place {
            CheckedPlace::Local(root) => self.write(*root, *root, &[], whole, span),
            CheckedPlace::Element { root, indices } => {
                self.write(*root, *root, indices, whole, span)
            }
            CheckedPlace::Tuple(places) => {
                let mut last = None;
                for p in places {
                    last = self.write_place(p, whole, span);
                }
                last
            }
        }
    }

    fn state_place(&mut self, name: &ast::Ident) -> Option<CheckedPlace> {
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
        match self.kinds[id.index()] {
            LocalKind::State => {}
            LocalKind::Param(i) if self.sig.params[i].ownership == ParamOwnership::Exclusive => {}
            _ => {
                self.error(name.span, format!("`{}` is not mutable state; only `let mut` bindings and `&mut tensor` parameters are assigned", name.name));
                return None;
            }
        }
        if matches!(self.locals[id.index()].ty, ValueType::Opaque { .. }) {
            self.error(
                name.span,
                "a capability value is immutable once produced and cannot be reassigned",
            );
            return None;
        }
        Some(CheckedPlace::Local(id))
    }

    fn arith_assign(
        &mut self,
        target: &ValueType,
        value: &CheckedExpr,
        op: AssignOp,
    ) -> Option<()> {
        if let (ValueType::Tensor(a), ValueType::Tensor(b)) = (target, &value.ty) {
            if !self.same_axes(a, b) || !elem_rounds(&b.elem, &a.elem) {
                self.error(
                    value.span,
                    format!(
                        "`{}` needs tensors over identical axes: {} vs {}",
                        op.text(),
                        target,
                        value.ty
                    ),
                );
                return None;
            }
            return Some(());
        }
        if let (ValueType::Tensor(a), Some(v)) = (target, value.ty.scalar_dtype()) {
            if DType::promote(a.elem.read_dtype(), v).is_some() {
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
    ) -> Option<CheckedStmt> {
        let cond = self.expr(cond, Some(&ValueType::Scalar(DType::Bool)))?;
        match &cond.ty {
            ValueType::Scalar(DType::Bool) => {}
            ValueType::Tensor(s) if s.elem == Elem::Dtype(DType::Bool) => {
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
        let facts_before = self.facts.clone();
        let unassigned_before = self.unassigned.clone();
        let pending_before = self.pending_full_assign.clone();
        let symbols_before = self.scalar_symbols.clone();
        let moved_before = self.moved.clone();
        let reads_before = self.reads.clone();

        self.assume(&cond, false);
        let then_body = self.scoped_block(then);
        self.facts = facts_before.clone();
        let unassigned_then = std::mem::replace(&mut self.unassigned, unassigned_before);
        let symbols_then = std::mem::replace(&mut self.scalar_symbols, symbols_before);
        let moved_then = std::mem::replace(&mut self.moved, moved_before.clone());
        let reads_then = std::mem::replace(&mut self.reads, reads_before);
        self.pending_full_assign = pending_before.clone();

        let else_body = match els {
            Some(b) => {
                self.assume(&cond, true);
                self.scoped_block(b)
            }
            None => CheckedBlock {
                statements: Vec::new(),
                terminator: BlockTerminator::Continue,
            },
        };
        self.facts = facts_before;
        let moved_else = self.moved.clone();
        self.unassigned.extend(unassigned_then);
        self.reads.extend(reads_then);
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
        Some(CheckedStmt::If {
            condition: cond,
            then_body,
            else_body,
        })
    }

    /// Path facts of a condition over symbolic integers (or of its negation).
    fn assume(&mut self, cond: &CheckedExpr, negate: bool) {
        let (op, lhs, rhs) = match &cond.kind {
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Binary(op),
                operands,
            } if operands.len() == 2 => (*op, &operands[0], &operands[1]),
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Unary(crate::syntax::ast::UnaryOp::Not),
                operands,
            } if operands.len() == 1 => {
                self.assume(&operands[0].clone(), !negate);
                return;
            }
            _ => return,
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
        let (Some(l), Some(r)) = (lhs.sym, rhs.sym) else {
            return;
        };
        let one = self.arena.int(1);
        match (op, negate) {
            (BinaryOp::Lt, false) | (BinaryOp::Ge, true) => {
                let d = self.arena.int_sub(r, l);
                let d = self.arena.int_sub(d, one);
                self.assume_nonneg(d)
            }
            (BinaryOp::Le, false) | (BinaryOp::Gt, true) => {
                let d = self.arena.int_sub(r, l);
                self.assume_nonneg(d)
            }
            (BinaryOp::Gt, false) | (BinaryOp::Le, true) => {
                let d = self.arena.int_sub(l, r);
                let d = self.arena.int_sub(d, one);
                self.assume_nonneg(d)
            }
            (BinaryOp::Ge, false) | (BinaryOp::Lt, true) => {
                let d = self.arena.int_sub(l, r);
                self.assume_nonneg(d)
            }
            (BinaryOp::Eq, false) | (BinaryOp::Ne, true) => {
                let d = self.arena.int_sub(l, r);
                self.assume_zero(d)
            }
            _ => {}
        }
    }

    fn for_stmt(
        &mut self,
        parallel: bool,
        targets: &[ast::Ident],
        iter: &ast::Expr,
        body: &ast::Block,
    ) -> Option<CheckedStmt> {
        let is_range = matches!(iter.kind, A::Range { .. });
        let is_range_value = matches!(
            &iter.kind,
            A::Name(name)
                if self
                    .lookup(&name.name)
                    .is_some_and(|id| matches!(self.locals[id.index()].ty, ValueType::Range { .. }))
        );
        if !is_range && !is_range_value {
            self.error(
                iter.span,
                "`for` iterates a bounded range `lo..hi` or a `range[N]` value",
            );
            return None;
        }
        let [target] = targets else {
            self.error(iter.span, "a range binds exactly one name");
            return None;
        };
        let range = self.expr(iter, None)?;
        let ValueType::Range { bound } = range.ty.clone() else {
            self.error(iter.span, "a range loop source is a `range[N]` value");
            return None;
        };
        let bound = bound;
        let (range_exprs, lo_sym, hi_sym) = match &range.kind {
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::RangeMake,
                operands,
            } if operands.len() == 2 => {
                let lo = operands[0].sym.clone();
                let hi = operands[1].sym.clone();
                ((operands[0].clone(), operands[1].clone()), lo, hi)
            }
            _ => {
                let lo = CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: crate::intrinsics::PrimitiveId::RangeStart,
                        operands: vec![range.clone()],
                    },
                    ValueType::Scalar(DType::I32),
                    Some(self.arena.int(0)),
                    iter.span,
                );
                let hi = CheckedExpr::new(
                    CheckedExprKind::Primitive {
                        id: crate::intrinsics::PrimitiveId::RangeEnd,
                        operands: vec![range.clone()],
                    },
                    ValueType::Scalar(DType::I32),
                    Some(bound),
                    iter.span,
                );
                ((lo, hi), Some(self.arena.int(0)), Some(bound))
            }
        };
        let (Some(lo_sym), Some(hi_sym)) = (lo_sym, hi_sym) else {
            self.error(iter.span, "range bounds must be symbolic integers");
            return None;
        };
        let width = self.arena.int_sub(hi_sym, lo_sym);
        let cardinality = match super::prove::constant(&self.arena, width) {
            Some(n) if n <= 0 => LoopCardinality::Zero,
            Some(1) => LoopCardinality::One,
            _ => LoopCardinality::RepeatedOrUnknown,
        };
        let floor = self.locals.len();
        let moved_before = self.moved.clone();
        let reads_before = self.reads.clone();
        self.push_scope();
        let binder = self.declare(
            &target.name,
            ValueType::Index { bound: hi_sym },
            target.span,
            LocalKind::Binder,
            false,
        );
        let (_, symbol, _) = self.arena.loop_binder();
        let one = self.arena.int(1);
        let upper = self.arena.int_sub(hi_sym, one);
        self.facts.set_range(symbol, lo_sym, upper);
        self.symbols.insert(binder, symbol);
        self.locals[binder.index()].symbol = Some(symbol);
        if parallel {
            self.logical_parallel.push((floor, binder));
        }
        self.loops.push(super::LoopCtx {
            floor,
            writes: Vec::new(),
            whole_writes: Vec::new(),
            atomics: Vec::new(),
        });
        // A complete logical `0..axis` traversal may collectively initialize
        // uninitialized storage over that axis.
        let prior_pending = self.pending_full_assign.clone();
        let mut pending = Vec::new();
        for tensor in self.unassigned.iter().copied() {
            let ValueType::Tensor(shaped) = &self.locals[tensor.index()].ty else {
                continue;
            };
            let prefix = prior_pending
                .iter()
                .find(|(candidate, _)| *candidate == tensor)
                .map(|(_, axes)| axes.clone())
                .unwrap_or_default();
            let axis = prefix.len();
            let Some(extent) = shaped.axes.get(axis) else {
                continue;
            };
            if super::prove::is_zero(&self.arena, lo_sym)
                && super::prove::same(&self.arena, hi_sym, *extent)
            {
                let mut axes = prefix;
                axes.push(binder);
                pending.push((tensor, axes));
            }
        }
        self.pending_full_assign = pending;
        self.init_loop_depth += 1;
        self.loop_depth += 1;
        let body_block = self.block(body);
        self.loop_depth -= 1;
        self.init_loop_depth -= 1;
        self.pending_full_assign = prior_pending;
        let ctx = self.loops.pop().expect("loop context is balanced");
        if parallel {
            self.logical_parallel.pop();
        }
        self.pop_scope();
        self.finish_loop_ownership(moved_before, floor, cardinality, iter.span);
        self.reads = reads_before;
        let _ = ctx;
        let kind = if parallel {
            LoopKind::Independent
        } else {
            LoopKind::Ordered
        };
        let checked = CheckedStmt::Loop {
            kind,
            binder,
            start: range_exprs.0,
            end: range_exprs.1,
            body: body_block,
        };
        let loop_block = CheckedBlock {
            statements: vec![checked.clone()],
            terminator: BlockTerminator::Continue,
        };
        let pending: Vec<_> = self.unassigned.iter().copied().collect();
        for variable in pending {
            let Some(shaped) = self.locals[variable.index()].ty.shaped().cloned() else {
                continue;
            };
            if super::writes_cover(
                &mut self.arena,
                &self.locals,
                &loop_block,
                variable,
                &shaped.axes,
                &[],
                &self.facts,
            ) {
                self.unassigned.remove(&variable);
            }
        }
        Some(checked)
    }

    /// Join ownership across a loop's zero/one/back-edge control flow.
    ///
    /// Bindings declared in the body are fresh on every iteration. A captured
    /// owned binding, however, must reach a repeated back-edge initialized. A
    /// statically empty loop preserves the entry state, and a statically
    /// single-iteration loop carries its exit state forward.
    fn finish_loop_ownership(
        &mut self,
        moved_before: HashSet<super::ir::LocalId>,
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
                    .filter(|id| id.index() < captured_floor)
                    .collect();
            }
            LoopCardinality::RepeatedOrUnknown => {
                let mut consumed: Vec<_> = moved_after
                    .difference(&moved_before)
                    .copied()
                    .filter(|id| id.index() < captured_floor)
                    .collect();
                consumed.sort_unstable();
                for id in consumed {
                    self.error(
                        span,
                        format!(
                            "loop may repeat after moving captured owned tensor `{}`; reinitialize it before the iteration ends",
                            self.locals[id.index()].name
                        ),
                    );
                }
                // A valid repeated body has the same captured ownership state at
                // its back-edge as at entry. Body-local bindings do not escape.
                self.moved = moved_before;
            }
        }
    }

    // ---- result boundaries ----

    fn return_stmt(&mut self, values: &[ast::Expr], span: Span) -> Option<PartialStmt> {
        if self.loop_depth > 0 {
            self.error(
                span,
                "return inside a loop is not one result per path; return after the loop",
            );
            return None;
        }
        let expected: Vec<ValueType> = match &self.sig.result {
            ValueType::Void => Vec::new(),
            ValueType::Tuple(items) if values.len() != 1 => items.as_slice().to_vec(),
            other => vec![other.clone()],
        };
        if values.is_empty() && !self.sig.result.is_void() {
            self.error(
                span,
                format!(
                    "`{}` returns {} but this `return` has no values",
                    self.sig.name, self.sig.result
                ),
            );
            return None;
        }
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
        let mut exprs = Vec::new();
        for (i, v) in values.iter().enumerate() {
            let hint = expected.get(i).cloned();
            let e = self.expr(v, hint.as_ref())?;
            if e.ty.is_void() {
                self.error(e.span, "`void` is not a value");
                return None;
            }
            exprs.push(e);
        }
        for (e, ty) in exprs.iter().zip(&expected) {
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
        }
        Some(PartialStmt::Terminal(exprs))
    }
}

/// Every local referenced by a place's indices (points and range bounds, at
/// any depth): the places an independent loop may mutate through its binder.
/// A statement or a terminal return during block assembly.
enum PartialStmt {
    Statement(CheckedStmt),
    Terminal(Vec<CheckedExpr>),
}
