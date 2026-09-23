//! Statements: bindings, state updates, loops, branches, and the function's
//! one result — emitted as the checked statement set.

use super::ir::{
    Block as CheckedBlock, Expr as CheckedExpr, ExprKind as CheckedExprKind, LocalId, LoopKind,
    Ownership as ParamOwnership, Pattern, Place as CheckedPlace, Stmt as CheckedStmt,
};
use super::{elem_rounds, Checker, LocalKind};
use crate::checked::DiagnosticRule;
use crate::expr::{IntExpr, SymbolId};
use crate::initialization::{Condition, RegionOps};
use crate::span::Span;
use crate::syntax::ast::{self, AssignOp, BinaryOp, ExprKind as A, UnaryOp};
use crate::types::{DType, Elem, ValueType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopCardinality {
    Zero,
    One,
    RepeatedOrUnknown,
}

/// How a value enters its destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Destination {
    /// A `let mut` local takes the value itself: its declared type is kept
    /// exactly (L23 (a)).
    Rebind,
    /// Stored into existing elements or a result: floats round to the
    /// destination's element type.
    Install,
}

/// The checker's path facts reuse the initialization pass's one
/// condition-to-facts owner (G-C24-17).
struct Conditions<'a> {
    arena: &'a mut crate::expr::ExprArena,
}

impl RegionOps for Conditions<'_> {
    fn arena(&mut self) -> &mut crate::expr::ExprArena {
        self.arena
    }
    fn arena_ref(&self) -> &crate::expr::ExprArena {
        self.arena
    }
    fn fresh_variable(&mut self) -> SymbolId {
        self.arena.proof_variable(crate::expr::SymbolSort::Int)
    }
}

impl<'a> Checker<'a> {
    /// A new version of a mutable quantity or word local, carrying its
    /// declared type's bounds (L32 (a)). The path that reaches a version
    /// (an assignment, an `if` arm, a loop visit) decides its value at run
    /// time, so a bound its facts do not prove is checked where it is used.
    /// A version made by binding or assigning a value with a symbol equals
    /// that symbol (L31, I-90), so the value's facts hold for the local.
    fn fresh_integer_version(&mut self, local: LocalId, assigned: Option<IntExpr>) -> SymbolId {
        let ty = self.locals[local.index()].ty.clone();
        let symbol = self.fresh_data_symbol();
        self.facts.assume_type(&mut self.arena, symbol, &ty);
        let value = self.arena.int_symbol(symbol);
        if let Some(assigned) = assigned {
            let difference = self.arena.int_sub(value, assigned);
            self.assume_zero(difference);
        }
        self.scalar_symbols.insert(local, value);
        symbol
    }

    fn integer_value_type(ty: &ValueType) -> bool {
        matches!(ty, ValueType::Integer | ValueType::Index { .. })
            || ty.scalar_dtype().is_some_and(DType::is_int)
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Default::default());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// The function body: its root block and its one result (L5). `return`
    /// is legal only as the last statement of the top-level block.
    pub fn function_body(&mut self, b: &ast::Block, span: Span) -> (CheckedBlock, Vec<CheckedExpr>) {
        let (statements, tail) = match b.stmts.split_last() {
            Some((last, rest)) if matches!(last.kind, ast::StmtKind::Return(_)) => (rest, Some(last)),
            _ => (b.stmts.as_slice(), None),
        };
        let root = self.statements(statements);
        let result = match tail {
            Some(ast::Stmt {
                kind: ast::StmtKind::Return(values),
                span,
            }) => self.result_values(values, *span),
            Some(_) => unreachable!("the tail was matched as a return"),
            None => {
                if !self.sig.result.is_void() {
                    let result = self.shown(&self.sig.result.clone());
                    self.error(
                        DiagnosticRule::Type,
                        span,
                        format!(
                            "`{}` returns {result} but its body does not end in `return`",
                            self.sig.name
                        ),
                    );
                }
                Some(Vec::new())
            }
        };
        (root, result.unwrap_or_default())
    }

    /// Check one nested source block.
    pub fn block(&mut self, b: &ast::Block) -> CheckedBlock {
        self.block_depth += 1;
        let block = self.statements(&b.stmts);
        self.block_depth -= 1;
        block
    }

    fn statements(&mut self, stmts: &[ast::Stmt]) -> CheckedBlock {
        let mut statements = Vec::new();
        for s in stmts {
            if let Some(statement) = self.stmt(s) {
                statements.push(statement);
            }
        }
        CheckedBlock { statements }
    }

    fn scoped_block(&mut self, b: &ast::Block) -> CheckedBlock {
        self.push_scope();
        let out = self.block(b);
        self.pop_scope();
        out
    }

    fn stmt(&mut self, s: &ast::Stmt) -> Option<CheckedStmt> {
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
                self.dyn_views.clear();
                self.for_stmt(*parallel, targets, iter, body)
            }
            ast::StmtKind::If { cond, then, els } => self.if_stmt(cond, then, els.as_ref()),
            ast::StmtKind::Return(_) => {
                self.error(
                    DiagnosticRule::Syntax,
                    s.span,
                    "`return` is only the last statement of a function body; bind the value with `let mut result` before the `if` or loop, assign it in each branch, and `return result` at the end",
                );
                None
            }
            ast::StmtKind::Expr(e) => {
                let e = self.expr(e, None)?;
                if !e.ty.is_void() {
                    let shown = self.shown(&e.ty);
                    self.error(
                        DiagnosticRule::Type,
                        e.span,
                        format!("an expression statement is a call evaluated for its effects; this value of type {shown} is unused"),
                    );
                }
                Some(CheckedStmt::Evaluate(e))
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

    /// A `let`. Every failure poisons the pattern, so later uses of its
    /// names are not reported again.
    fn bind(&mut self, pattern: &ast::Pattern, value: &ast::Expr, state: bool) -> Option<CheckedStmt> {
        let bound = self.bind_value(pattern, value, state);
        if bound.is_none() {
            self.poison(pattern);
        }
        bound
    }

    fn bind_value(
        &mut self,
        pattern: &ast::Pattern,
        value: &ast::Expr,
        state: bool,
    ) -> Option<CheckedStmt> {
        // L31 (I-90): a bound word is its value's one symbol.
        let value = self.expr(value, None)?;
        let value = self.word_value(value);
        if value.ty.is_void() {
            self.error(DiagnosticRule::Type, value.span, "cannot bind a call that returns nothing");
            return None;
        }
        let moves = match self.binding_moves(&value) {
            Ok(moves) => moves,
            Err(error) => {
                self.error(DiagnosticRule::Ownership, value.span, error);
                return None;
            }
        };
        let pattern = self.destructure(pattern, &value.ty.clone(), &value, state)?;
        for place in moves {
            self.consume_place(&place);
        }
        Some(CheckedStmt::Let { pattern, value })
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
                let mut ownership = self.ownership(value);
                let mut failure = None;
                ownership.visit(&mut Vec::new(), &mut |_, leaf| {
                    if let super::ownership::TensorOwnership::Borrowed { owner, .. } = leaf {
                        let exclusive = state && self.writable_place(owner);
                        if self
                            .live_borrows()
                            .iter()
                            .any(|(_, borrowed, prior_exclusive)| {
                                borrowed.overlaps(owner) && (exclusive || *prior_exclusive)
                            })
                        {
                            failure = Some(format!(
                                "tensor borrow of `{}` overlaps a live {} borrow",
                                self.locals[owner.local.index()].name,
                                if exclusive {
                                    "mutable"
                                } else {
                                    "exclusive mutable"
                                }
                            ));
                        }
                    }
                });
                if let Some(message) = failure {
                    self.error(DiagnosticRule::Ownership, name.span, message);
                    return None;
                }
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
                fn install(
                    value: &mut super::ownership::ValueOwnership,
                    state: bool,
                    writable: &impl Fn(&super::ownership::LocalPlace) -> bool,
                ) {
                    use super::ownership::{TensorOwnership, ValueOwnership};
                    match value {
                        ValueOwnership::Tensor(TensorOwnership::Computed) if state => {
                            *value = ValueOwnership::Tensor(TensorOwnership::Owned { moved: false })
                        }
                        ValueOwnership::Tensor(TensorOwnership::Owned { moved }) => *moved = false,
                        ValueOwnership::Tensor(TensorOwnership::Borrowed { owner, exclusive }) => {
                            *exclusive = state && writable(owner)
                        }
                        ValueOwnership::Tuple(parts) => {
                            for part in parts {
                                install(part, state, writable);
                            }
                        }
                        _ => {}
                    }
                }
                install(&mut ownership, state, &|root| self.writable_place(root));
                self.locals[id.index()].ownership = ownership;
                if state && Self::integer_value_type(ty) {
                    let symbol = self.fresh_integer_version(id, value.sym);
                    self.locals[id.index()].symbol = Some(symbol);
                } else if !state && ty.scalar_dtype().is_some_and(DType::is_int) {
                    // I-90: a word local is one symbol, its runtime value.
                    let symbol = self.fresh_data_symbol();
                    self.facts.assume_type(&mut self.arena, symbol, ty);
                    if let Some(exact) = value.sym {
                        let local = self.arena.int_symbol(symbol);
                        let difference = self.arena.int_sub(local, exact);
                        self.assume_zero(difference);
                    }
                    self.symbols.insert(id, symbol);
                    self.locals[id.index()].symbol = Some(symbol);
                } else if !state && Self::integer_value_type(ty) {
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
                                ..
                            } = &value.kind
                            {
                                if let Some(extent) = operands
                                    .first()
                                    .and_then(|o| o.ty.shaped())
                                    .and_then(|s| s.axes.get(*axis as usize))
                                    .copied()
                                {
                                    let symbol = self.fresh_data_symbol();
                                    let zero = self.arena.int(0);
                                    let one = self.arena.int(1);
                                    let upper = self.arena.int_sub(extent, one);
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
                    let shown = self.shown(ty);
                    self.error(
                        DiagnosticRule::Type,
                        value.span,
                        format!("a tuple pattern destructures a tuple; this value is a {shown}"),
                    );
                    return None;
                };
                if tys.len() != items.len() {
                    self.error(
                        DiagnosticRule::Type,
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
                        ..
                    } if operands.len() == tys.len() => {
                        operands.iter().cloned().map(Some).collect()
                    }
                    _ => vec![None; tys.len()],
                };
                let mut out = Vec::new();
                for (i, ((item, ty), part)) in items.iter().zip(tys.iter()).zip(parts).enumerate() {
                    let component = match part {
                        Some(part) => part,
                        None => self.primitive_expr(
                            crate::intrinsics::PrimitiveId::TupleGet(i as u32),
                            vec![value.clone()],
                            ty.clone(),
                            None,
                            value.span,
                        ),
                    };
                    out.push(self.destructure(item, ty, &component, state)?);
                }
                Some(Pattern::Tuple(out))
            }
        }
    }

    // ---- state updates ----

    /// The value a destination of type `target` receives (L23, L31, L17).
    /// An `index[B]` destination takes a word or quantity proved in range, or
    /// an `index[k]` with `k <= B`.
    fn destination_value(
        &mut self,
        target: &ValueType,
        value: CheckedExpr,
        destination: Destination,
        what: &str,
    ) -> Option<CheckedExpr> {
        let mismatch = |checker: &mut Self, value: &CheckedExpr| {
            let target = checker.shown(target);
            let actual = checker.shown(&value.ty);
            let hint = match (&value.ty, target.as_str()) {
                (ValueType::Index { .. }, "i32" | "u32") => {
                    format!("; the word projection wraps, write `{target}(…)`")
                }
                _ => String::new(),
            };
            checker.error(
                DiagnosticRule::Type,
                value.span,
                format!("{what} {target} but the value has type {actual}{hint}"),
            );
        };
        match (target, &value.ty) {
            (ValueType::Index { bound }, ValueType::Index { bound: actual }) => {
                let slack = self.arena.int_sub(*bound, *actual);
                if super::prove::nonneg(&self.arena, &self.facts, slack) {
                    Some(value)
                } else {
                    let span = value.span;
                    self.index_position(value, *bound, span)
                }
            }
            (ValueType::Index { bound }, ValueType::Integer | ValueType::Scalar(DType::I32 | DType::U32)) => {
                let span = value.span;
                self.index_position(value, *bound, span)
            }
            (ValueType::Scalar(a), ValueType::Scalar(b))
                if *a == *b || (a.is_float() && b.is_float()) =>
            {
                Some(value)
            }
            (ValueType::Tensor(a), ValueType::Tensor(b)) => {
                let compatible = self.same_axes(a, b)
                    && match destination {
                        Destination::Rebind => a.elem == b.elem,
                        Destination::Install => elem_rounds(&b.elem, &a.elem),
                    };
                if !compatible {
                    mismatch(self, &value);
                    return None;
                }
                Some(value)
            }
            (ValueType::Tuple(targets), ValueType::Tuple(values)) if targets.len() == values.len() => {
                let targets = targets.as_slice().to_vec();
                let parts = (0..targets.len())
                    .map(|index| super::ownership::project(&value, index))
                    .collect::<Vec<_>>();
                let mut checked = Vec::new();
                for (target, part) in targets.iter().zip(parts) {
                    checked.push(self.destination_value(target, part, destination, what)?);
                }
                let tys = checked.iter().map(|part| part.ty.clone()).collect();
                let span = value.span;
                Some(self.primitive_expr(
                    crate::intrinsics::PrimitiveId::TuplePack,
                    checked,
                    ValueType::Tuple(crate::types::NonEmpty::new(tys).expect("components")),
                    None,
                    span,
                ))
            }
            _ if self.same_ty(target, &value.ty) => Some(value),
            _ => {
                mismatch(self, &value);
                None
            }
        }
    }

    /// `value` of a compound assignment `target op= value`: the desugared
    /// `target op value`, promoted like any binary operation.
    fn compound_value(
        &mut self,
        target: &ast::Expr,
        op: AssignOp,
        value: &ast::Expr,
        hint: Option<&ValueType>,
    ) -> Option<CheckedExpr> {
        let binary = match op {
            AssignOp::Add => BinaryOp::Add,
            AssignOp::Sub => BinaryOp::Sub,
            AssignOp::Mul => BinaryOp::Mul,
            AssignOp::Assign => unreachable!("plain assignment is not compound"),
        };
        let current = self.expr(target, None)?;
        let value = self.expr(value, hint.or(Some(&current.ty.clone())))?;
        let span = current.span.to(value.span);
        self.binary_exprs(binary, current, value, span)
    }

    fn assign(&mut self, target: &ast::Expr, op: AssignOp, value: &ast::Expr) -> Option<CheckedStmt> {
        match &target.kind {
            A::Tuple(places) => {
                if op != AssignOp::Assign {
                    self.error(DiagnosticRule::Type, target.span, "tuple assignment uses `=`");
                    return None;
                }
                let mut targets = Vec::new();
                for place in places {
                    let A::Name(name) = &place.kind else {
                        self.error(
                            DiagnosticRule::Type,
                            place.span,
                            "tuple assignment installs whole `let mut` state objects",
                        );
                        return None;
                    };
                    let checked = self.state_place(name)?;
                    let ty = self.locals[self.lookup(&name.name)?.index()].ty.clone();
                    targets.push((checked, ty));
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
                            out.push(self.expr(part, Some(ty))?);
                        }
                        let tys: Vec<ValueType> = out.iter().map(|e| e.ty.clone()).collect();
                        let span = value.span;
                        self.primitive_expr(
                            crate::intrinsics::PrimitiveId::TuplePack,
                            out,
                            ValueType::Tuple(crate::types::NonEmpty::new(tys).expect("components")),
                            None,
                            span,
                        )
                    }
                    _ => self.expr(value, Some(&expected))?,
                };
                let value = self.destination_value(
                    &expected,
                    value,
                    Destination::Rebind,
                    "the tuple assignment has type",
                )?;
                let mut authorities = Vec::new();
                for (place, _) in &targets {
                    let root = self.write_place(place, target.span)?;
                    authorities.extend(self.exclusive_write_authority(root, place));
                }
                let value_symbols = targets
                    .iter()
                    .filter_map(|(place, ty)| match place {
                        CheckedPlace::Local(root) if Self::integer_value_type(ty) => {
                            Some((root.local, self.fresh_integer_version(root.local, None)))
                        }
                        _ => None,
                    })
                    .collect();
                let place = CheckedPlace::Tuple(targets.into_iter().map(|(p, _)| p).collect());
                if let Err(error) = self.assignment_ownership(&place, &value) {
                    self.error(DiagnosticRule::Ownership, value.span, error);
                    return None;
                }
                Some(CheckedStmt::Assign {
                    place,
                    value,
                    value_symbols,
                    authorities,
                })
            }
            A::Name(name) => {
                let place = self.state_place(name)?;
                let ty = self.locals[self
                    .lookup(&name.name)
                    .expect("state place resolved")
                    .index()]
                .ty
                .clone();
                let value = if op == AssignOp::Assign {
                    self.expr(value, Some(&ty))?
                } else {
                    self.compound_value(target, op, value, Some(&ty))?
                };
                let destination = match place {
                    CheckedPlace::Element { .. } => Destination::Install,
                    _ => Destination::Rebind,
                };
                let value =
                    self.destination_value(&ty, value, destination, &format!("`{}` has type", name.name))?;
                let root = self.write_place(&place, target.span)?;
                if let Err(error) = self.assignment_ownership(&place, &value) {
                    self.error(DiagnosticRule::Ownership, value.span, error);
                    return None;
                }
                let authorities = self.exclusive_write_authority(root, &place);
                let value_symbols = Self::integer_value_type(&ty)
                    .then(|| (root, self.fresh_integer_version(root, value.sym)))
                    .into_iter()
                    .collect();
                Some(CheckedStmt::Assign {
                    place,
                    value,
                    value_symbols,
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
                        DiagnosticRule::Type,
                        target.span,
                        "element assignment indexes a tensor variable directly",
                    );
                    return None;
                };
                if !self.writable_root(self.root_var_local(binding)) {
                    let name = self.locals[binding.index()].name.clone();
                    self.error(
                        DiagnosticRule::Ownership,
                        target.span,
                        format!("`{name}` is not writable storage; writing requires `let mut` state or a `&mut tensor` parameter"),
                    );
                    return None;
                }
                // Packed representations are readable and decodable but never writable.
                if let Some(shaped) = self.locals[binding.index()].ty.shaped() {
                    if matches!(shaped.elem, Elem::Repr(_)) {
                        self.error(
                            DiagnosticRule::Type,
                            target.span,
                            "packed representations are readable and decodable but not writable",
                        );
                        return None;
                    }
                }
                if !matches!(selected, ValueType::Scalar(_) | ValueType::Tensor(_)) {
                    let shown = self.shown(&selected);
                    self.error(
                        DiagnosticRule::Type,
                        target.span,
                        format!("cannot assign into a place of type {shown}"),
                    );
                    return None;
                }
                if op != AssignOp::Assign && !matches!(selected, ValueType::Scalar(_)) {
                    self.error(
                        DiagnosticRule::Type,
                        target.span,
                        "compound assignment requires a scalar element place",
                    );
                    return None;
                }
                let value = if op == AssignOp::Assign {
                    let hint = selected.scalar_dtype().map(ValueType::Scalar);
                    self.expr(value, hint.as_ref())?
                } else {
                    self.compound_value(target, op, value, Some(&selected))?
                };
                let value = match (&selected, &value.ty) {
                    (ValueType::Scalar(dtype), ValueType::Scalar(actual))
                        if !(*actual == *dtype || (actual.is_float() && dtype.is_float())) =>
                    {
                        self.error(
                            DiagnosticRule::Type,
                            value.span,
                            format!(
                                "cannot assign {} to an element of dtype {}; cast explicitly",
                                actual.name(),
                                dtype.name()
                            ),
                        );
                        return None;
                    }
                    (ValueType::Scalar(dtype), ValueType::Index { .. } | ValueType::Integer) => {
                        let shown = self.shown(&value.ty);
                        self.error(
                            DiagnosticRule::Type,
                            value.span,
                            format!(
                                "cannot assign {shown} to an element of dtype {}; the word projection wraps, write `{}(…)`",
                                dtype.name(),
                                dtype.name()
                            ),
                        );
                        return None;
                    }
                    (ValueType::Scalar(dtype), other) if other.scalar_dtype().is_none() => {
                        let shown = self.shown(other);
                        self.error(
                            DiagnosticRule::Type,
                            value.span,
                            format!("cannot assign {shown} to an element of dtype {}", dtype.name()),
                        );
                        return None;
                    }
                    (ValueType::Scalar(_), _) => value,
                    _ => self.destination_value(
                        &selected,
                        value,
                        Destination::Install,
                        "this selection has type",
                    )?,
                };
                self.write(root, binding, target.span)?;
                let checked_place = CheckedPlace::Element {
                    root: super::ownership::LocalPlace::root(root),
                    indices,
                };
                let authorities = self.exclusive_write_authority(root, &checked_place);
                Some(CheckedStmt::Assign {
                    place: checked_place,
                    value,
                    value_symbols: Vec::new(),
                    authorities,
                })
            }
            _ => {
                self.error(
                    DiagnosticRule::Type,
                    target.span,
                    "an assignment target is `let mut` state, a tensor element, or a tuple of state",
                );
                None
            }
        }
    }

    /// Record a whole-variable write through a place.
    fn write_place(&mut self, place: &CheckedPlace, span: Span) -> Option<LocalId> {
        match place {
            CheckedPlace::Local(root) => {
                if self
                    .logical_parallel
                    .iter()
                    .any(|(floor, _)| root.local.index() < *floor)
                {
                    self.error(
                        DiagnosticRule::Independence,
                        span,
                        "a `parallel for` body cannot reassign captured state; use an explicit reduction, participant-local value, or disjoint tensor element writes",
                    );
                    return None;
                }
                self.write(root.local, root.local, span)
            }
            CheckedPlace::Element { root, .. } => {
                if !self.writable_place(root) {
                    self.error(DiagnosticRule::Ownership, span, "tensor place does not permit exclusive writes");
                    return None;
                }
                self.write(root.local, root.local, span)
            }
            CheckedPlace::Tuple(places) => {
                let mut last = None;
                for p in places {
                    last = self.write_place(p, span);
                }
                last
            }
        }
    }

    fn state_place(&mut self, name: &ast::Ident) -> Option<CheckedPlace> {
        let Some(id) = self.lookup(&name.name) else {
            if !self.poisoned.contains(&name.name) {
                self.error(
                    DiagnosticRule::Resolution,
                    name.span,
                    format!(
                        "`{}` is not declared; introduce state with `let mut`",
                        name.name
                    ),
                );
            }
            return None;
        };
        match self.kinds[id.index()] {
            LocalKind::State => {}
            LocalKind::Param(i) if self.sig.params[i].ownership == ParamOwnership::Exclusive => {}
            _ => {
                self.error(
                    DiagnosticRule::Ownership,
                    name.span,
                    format!("`{}` is not mutable state; only `let mut` bindings and `&mut tensor` parameters are assigned", name.name),
                );
                return None;
            }
        }
        if matches!(self.locals[id.index()].ty, ValueType::Opaque { .. }) {
            self.error(
                DiagnosticRule::Type,
                name.span,
                "a capability value is immutable once produced and cannot be reassigned",
            );
            return None;
        }
        fn target(
            value: &super::ownership::ValueOwnership,
            place: &mut super::ownership::LocalPlace,
        ) -> CheckedPlace {
            use super::ownership::{TensorOwnership, ValueOwnership};
            match value {
                ValueOwnership::Tuple(parts) => CheckedPlace::Tuple(
                    parts
                        .iter()
                        .enumerate()
                        .map(|(i, part)| {
                            place.path.push(i);
                            let target = target(part, place);
                            place.path.pop();
                            target
                        })
                        .collect(),
                ),
                // L23 (b): a view local is a place; assigning it stores
                // into the viewed elements.
                ValueOwnership::Tensor(TensorOwnership::Borrowed { .. }) => CheckedPlace::Element {
                    root: place.clone(),
                    indices: Vec::new(),
                },
                _ => CheckedPlace::Local(place.clone()),
            }
        }
        Some(target(
            &self.locals[id.index()].ownership,
            &mut super::ownership::LocalPlace::root(id),
        ))
    }

    // ---- control ----

    /// Symbols every participant of the innermost `parallel for` agrees on
    /// (L14): dimensions and binders of uniform ordered loops.
    fn uniform_symbol(&self, symbol: SymbolId) -> bool {
        self.sig.dimension_of(symbol).is_some() || self.uniform_binders.contains(&symbol)
    }

    fn uniform_value(&self, value: &CheckedExpr) -> bool {
        value.sym.is_some_and(|sym| {
            super::prove::symbols(&self.arena, sym)
                .into_iter()
                .all(|symbol| self.uniform_symbol(symbol))
        })
    }

    /// Whether a condition has the same value for every participant (L14).
    fn uniform_condition(&self, condition: &CheckedExpr) -> bool {
        match &condition.kind {
            CheckedExprKind::Literal(_) => true,
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Binary(BinaryOp::And | BinaryOp::Or),
                operands,
                ..
            } => operands.iter().all(|operand| self.uniform_condition(operand)),
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Unary(UnaryOp::Not),
                operands,
                ..
            } => self.uniform_condition(&operands[0]),
            CheckedExprKind::Primitive {
                id:
                    crate::intrinsics::PrimitiveId::Binary(
                        BinaryOp::Eq
                        | BinaryOp::Ne
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge,
                    ),
                operands,
                ..
            } => operands.iter().all(|operand| self.uniform_value(operand)),
            _ => false,
        }
    }

    /// Whether control here is uniform across the innermost `parallel for`'s
    /// participants (or the body's own call site).
    pub fn participant_uniform(&self) -> bool {
        self.divergence.last().copied() == Some(0)
    }

    fn if_stmt(&mut self, cond: &ast::Expr, then: &ast::Block, els: Option<&ast::Block>) -> Option<CheckedStmt> {
        let cond = self.expr(cond, Some(&ValueType::Scalar(DType::Bool)))?;
        match &cond.ty {
            ValueType::Scalar(DType::Bool) => {}
            ValueType::Tensor(s) if s.elem == Elem::Dtype(DType::Bool) => {
                self.error(DiagnosticRule::Type, cond.span, "a mask is a `bool` tile, not a scalar condition; there is no implicit reduction. Use `select(mask, a, b)`");
                return None;
            }
            other => {
                let shown = self.shown(other);
                self.error(
                    DiagnosticRule::Type,
                    cond.span,
                    format!("an `if` condition is a scalar `bool`, found {shown}"),
                );
                return None;
            }
        }
        let divergent = !self.uniform_condition(&cond);
        let facts_before = self.facts.clone();
        let symbols_before = self.scalar_symbols.clone();
        // Captures cover only the locals in scope at the `if` (G-A1-new-1).
        let capture_symbols = (0..self.locals.len())
            .filter_map(|ordinal| {
                let id = LocalId::new(ordinal as u32);
                if !self.in_scope(id) || !Self::integer_value_type(&self.locals[ordinal].ty) {
                    return None;
                }
                let symbol = self.symbols.get(&id).copied().or_else(|| {
                    symbols_before.get(&id).and_then(|value| match self.arena.view((*value).into()) {
                        crate::expr::NodeView::Symbol(symbol) => Some(symbol),
                        _ => None,
                    })
                })?;
                Some((id, symbol))
            })
            .collect::<Vec<_>>();
        let moved_before = self.moved_snapshot();
        if divergent {
            *self.divergence.last_mut().expect("a divergence frame") += 1;
        }

        self.assume(&cond, false);
        let then_body = self.scoped_block(then);
        let facts_then = std::mem::replace(&mut self.facts, facts_before.clone());
        let symbols_then = std::mem::replace(&mut self.scalar_symbols, symbols_before.clone());
        let moved_then = self.moved_snapshot();
        self.restore_moves(&moved_before);

        let else_body = match els {
            Some(b) => {
                self.assume(&cond, true);
                self.scoped_block(b)
            }
            None => CheckedBlock {
                statements: Vec::new(),
            },
        };
        if divergent {
            *self.divergence.last_mut().expect("a divergence frame") -= 1;
        }
        let facts_else = std::mem::replace(&mut self.facts, facts_before);
        let moved_else = self.moved_snapshot();
        let symbols_else = std::mem::replace(&mut self.scalar_symbols, symbols_before.clone());
        let mut join_symbols = Vec::new();
        for (id, before) in symbols_before {
            if !self.locals[id.index()].mutable
                || !Self::integer_value_type(&self.locals[id.index()].ty)
            {
                continue;
            }
            let then_value = symbols_then.get(&id).copied().unwrap_or(before);
            let else_value = symbols_else.get(&id).copied().unwrap_or(before);
            if then_value != before || else_value != before {
                let joined = self.fresh_integer_version(id, None);
                // L32 (b): the bounds common to both arms.
                let scope = self.scope_symbols();
                self.facts.join_bounds(
                    &mut self.arena,
                    joined,
                    [(&facts_then, then_value), (&facts_else, else_value)],
                    &|symbol| scope.contains(&symbol),
                );
                join_symbols.push((id, joined));
            }
        }
        // A value is available after an `if` only when it is available on every
        // path. Both arms are checked from `moved_before`, so a move in one arm
        // cannot spuriously poison checking of the other arm.
        self.restore_moves(&moved_then.union(&moved_else).cloned().collect());
        Some(CheckedStmt::If {
            condition: cond,
            then_body,
            else_body,
            capture_symbols,
            join_symbols,
        })
    }

    /// The symbols in scope here: dimensions, and the symbols of every local
    /// in scope.
    fn scope_symbols(&self) -> std::collections::HashSet<SymbolId> {
        let mut symbols = self
            .sig
            .dimensions
            .iter()
            .map(|dimension| dimension.symbol)
            .collect::<std::collections::HashSet<_>>();
        for ordinal in 0..self.locals.len() {
            let id = LocalId::new(ordinal as u32);
            if !self.in_scope(id) {
                continue;
            }
            symbols.extend(self.symbols.get(&id).copied());
            if let Some(value) = self.scalar_symbols.get(&id) {
                symbols.extend(super::prove::symbols(&self.arena, *value));
            }
        }
        symbols
    }

    /// The checked condition as a path condition.
    fn condition(&self, cond: &CheckedExpr) -> Condition {
        match &cond.kind {
            CheckedExprKind::Literal(crate::reference_math::ReferenceScalar::Bool(value)) => {
                Condition::Constant(*value)
            }
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Unary(UnaryOp::Not),
                operands,
                ..
            } => Condition::Not(Box::new(self.condition(&operands[0]))),
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::Binary(op),
                operands,
                ..
            } if operands.len() == 2 => match (op, operands[0].sym, operands[1].sym) {
                (BinaryOp::And, _, _) => Condition::And(
                    Box::new(self.condition(&operands[0])),
                    Box::new(self.condition(&operands[1])),
                ),
                (BinaryOp::Or, _, _) => Condition::Or(
                    Box::new(self.condition(&operands[0])),
                    Box::new(self.condition(&operands[1])),
                ),
                (
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge,
                    Some(l),
                    Some(r),
                ) => Condition::Compare(*op, l, r),
                _ => Condition::Version(0, Vec::new()),
            },
            _ => Condition::Version(0, Vec::new()),
        }
    }

    /// Path facts of a condition over symbolic integers (or of its negation).
    fn assume(&mut self, cond: &CheckedExpr, negate: bool) {
        let condition = self.condition(cond);
        let mut path = Vec::new();
        Conditions {
            arena: &mut self.arena,
        }
        .assume(&mut path, &mut self.facts, condition, !negate);
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
                DiagnosticRule::Type,
                iter.span,
                "`for` iterates a bounded range `lo..hi` or a `range[N]` value",
            );
            return None;
        }
        let [target] = targets else {
            self.error(DiagnosticRule::Type, iter.span, "a range binds exactly one name");
            return None;
        };
        let range = self.expr(iter, None)?;
        let ValueType::Range { bound } = range.ty.clone() else {
            self.error(DiagnosticRule::Type, iter.span, "a range loop source is a `range[N]` value");
            return None;
        };
        let (start, end, lo, hi, binder_bound) = match &range.kind {
            CheckedExprKind::Primitive {
                id: crate::intrinsics::PrimitiveId::RangeMake,
                operands,
                ..
            } if operands.len() == 2 => {
                let lo = operands[0].sym.expect("a checked range endpoint has its symbol");
                let hi = operands[1].sym.expect("a checked range endpoint has its symbol");
                (operands[0].clone(), operands[1].clone(), lo, hi, hi)
            }
            _ => {
                let start_symbol = self.fresh_symbol();
                let end_symbol = self.fresh_symbol();
                let start = self.arena.int_symbol(start_symbol);
                let end = self.arena.int_symbol(end_symbol);
                let zero = self.arena.int(0);
                self.facts.set_range(start_symbol, zero, bound);
                self.facts.set_range(end_symbol, start, bound);
                let lo = self.primitive_expr(
                    crate::intrinsics::PrimitiveId::RangeStart,
                    vec![range.clone()],
                    ValueType::Integer,
                    Some(start),
                    iter.span,
                );
                let hi = self.primitive_expr(
                    crate::intrinsics::PrimitiveId::RangeEnd,
                    vec![range.clone()],
                    ValueType::Integer,
                    Some(end),
                    iter.span,
                );
                // L17: a binder over `r: range[K]` has type `index[K]`.
                (lo, hi, start, end, bound)
            }
        };
        let width = self.arena.int_sub(hi, lo);
        let cardinality = match super::prove::constant(&self.arena, width) {
            Some(n) if n <= 0 => LoopCardinality::Zero,
            Some(1) => LoopCardinality::One,
            _ => LoopCardinality::RepeatedOrUnknown,
        };
        let uniform_bounds = self.uniform_value(&start) && self.uniform_value(&end);
        let data_bounds = self.data_dependent(lo) || self.data_dependent(hi);
        let floor = self.locals.len();
        let moved_before = self.moved_snapshot();
        let symbols_before = self.scalar_symbols.clone();
        let facts_before = self.facts.clone();
        let mut value_symbols = Vec::new();
        // The header carries exactly the mutable integer locals in scope
        // (G-A1-new-1).
        for ordinal in 0..floor {
            let id = LocalId::new(ordinal as u32);
            if self.in_scope(id)
                && self.locals[ordinal].mutable
                && Self::integer_value_type(&self.locals[ordinal].ty)
            {
                let header = self.fresh_integer_version(id, None);
                value_symbols.push((id, header, header));
            }
        }
        self.push_scope();
        let binder = self.declare(
            &target.name,
            ValueType::Index {
                bound: binder_bound,
            },
            target.span,
            LocalKind::Binder,
            false,
        );
        let (_, symbol, _) = self.arena.loop_binder();
        let one = self.arena.int(1);
        let upper = self.arena.int_sub(hi, one);
        self.facts.set_range(symbol, lo, upper);
        if data_bounds {
            self.data_symbols.insert(symbol);
        }
        self.symbols.insert(binder, symbol);
        self.locals[binder.index()].symbol = Some(symbol);
        if parallel {
            self.logical_parallel.push((floor, binder));
            self.divergence.push(0);
        } else if uniform_bounds && self.participant_uniform() {
            self.uniform_binders.insert(symbol);
        } else {
            *self.divergence.last_mut().expect("a divergence frame") += 1;
        }
        let body_block = self.block(body);
        if parallel {
            self.logical_parallel.pop();
            self.divergence.pop();
        } else if !self.uniform_binders.contains(&symbol) {
            *self.divergence.last_mut().expect("a divergence frame") -= 1;
        }
        self.pop_scope();
        self.finish_loop_ownership(moved_before, floor, cardinality, iter.span);
        // Facts learned in the body do not survive a loop that may not run.
        self.facts = facts_before;
        self.scalar_symbols = symbols_before;
        for (id, _, exit) in &mut value_symbols {
            *exit = self.fresh_integer_version(*id, None);
        }
        let kind = if parallel {
            LoopKind::Independent
        } else {
            LoopKind::Ordered
        };
        Some(CheckedStmt::Loop {
            kind,
            binder,
            start,
            end,
            body: body_block,
            value_symbols,
            initialization: crate::initialization::LoopInitialization::empty(
                crate::initialization::ParameterPath::root(self.sig.params.len() + binder.index()),
            ),
            separation: crate::initialization::VisitSeparation::Unrecorded,
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
        moved_before: std::collections::BTreeSet<super::ownership::LocalPlace>,
        captured_floor: usize,
        cardinality: LoopCardinality,
        span: Span,
    ) {
        let moved_after = self.moved_snapshot();
        match cardinality {
            LoopCardinality::Zero => self.restore_moves(&moved_before),
            LoopCardinality::One => {
                self.restore_moves(
                    &moved_after
                        .into_iter()
                        .filter(|place| place.local.index() < captured_floor)
                        .collect(),
                );
            }
            LoopCardinality::RepeatedOrUnknown => {
                let mut consumed: Vec<_> = moved_after
                    .difference(&moved_before)
                    .cloned()
                    .filter(|place| place.local.index() < captured_floor)
                    .collect();
                consumed.sort_unstable();
                for id in consumed {
                    let name = self.locals[id.local.index()].name.clone();
                    self.error(
                        DiagnosticRule::Ownership,
                        span,
                        format!(
                            "loop may repeat after moving captured owned tensor `{name}`; reinitialize it before the iteration ends"
                        ),
                    );
                }
                // A valid repeated body has the same captured ownership state at
                // its back-edge as at entry. Body-local bindings do not escape.
                self.restore_moves(&moved_before);
            }
        }
    }

    // ---- the result ----

    /// The values of the function's final `return`.
    fn result_values(&mut self, values: &[ast::Expr], span: Span) -> Option<Vec<CheckedExpr>> {
        let expected: Vec<ValueType> = match &self.sig.result {
            ValueType::Void => Vec::new(),
            ValueType::Tuple(items) if values.len() != 1 => items.as_slice().to_vec(),
            other => vec![other.clone()],
        };
        if values.len() != expected.len() {
            let result = self.shown(&self.sig.result.clone());
            self.error(
                DiagnosticRule::Type,
                span,
                format!(
                    "`{}` returns {result} but this `return` has {} values",
                    self.sig.name,
                    values.len()
                ),
            );
            return None;
        }
        let mut exprs = Vec::new();
        for (v, ty) in values.iter().zip(&expected) {
            let e = self.expr(v, Some(ty))?;
            if e.ty.is_void() {
                self.error(DiagnosticRule::Type, e.span, "`void` is not a value");
                return None;
            }
            exprs.push(e);
        }
        let mut returned_places = std::collections::BTreeSet::new();
        let mut result = Vec::new();
        for (e, ty) in exprs.into_iter().zip(&expected) {
            let (_, places) = match self.owned_consumption(&e) {
                Ok(value) => value,
                Err(error) => {
                    // L19: a result owns its backing.
                    self.error(
                        DiagnosticRule::Ownership,
                        e.span,
                        format!("{error}; return `to_owned(…)` of it, or drop it from the result"),
                    );
                    return None;
                }
            };
            for place in places {
                if !returned_places.insert(place) {
                    self.error(DiagnosticRule::Ownership, e.span, "owned tensor leaf is returned more than once");
                    return None;
                }
            }
            let name = format!("`{}` returns", self.sig.name);
            result.push(self.destination_value(ty, e, Destination::Install, &name)?);
        }
        Some(result)
    }

}
