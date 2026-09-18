//! Prove a contiguous union across one affine loop without visiting its points.
//! Endpoints are used only after symbolic affinity and invariant view geometry are
//! established. Holes, nonlinear indices and conditional tensor accesses fall back.
use super::*;
use seismic_lang::sym::Sym;
pub(super) fn sweep(
    body: &[Stmt],
    frame: &Frame,
    name: &str,
    lo: i64,
    hi: i64,
    budget: &mut usize,
) -> Option<Vec<(View, Mode)>> {
    if hi <= lo {
        return Some(Vec::new());
    }
    let mut accesses = Vec::new();
    collect(body, &mut accesses, budget)?;
    let mut first = frame.clone();
    first.env.insert(name.into(), lo);
    let mut last = frame.clone();
    last.env.insert(name.into(), hi.checked_sub(1)?);
    let mut result = Vec::new();
    for (expression, mode) in accesses {
        *budget = budget.checked_sub(1)?;
        affine_view(expression, frame, name)?;
        let a = view(expression, &first).ok()?;
        let b = view(expression, &last).ok()?;
        if a.parameter != b.parameter
            || a.elem != b.elem
            || a.shape != b.shape
            || a.strides != b.strides
        {
            return None;
        }
        let mut len = 1u64;
        for (size, stride) in a.shape.iter().zip(&a.strides).rev() {
            if *size == 0 {
                len = 0;
                break;
            }
            if *size != 1 && *stride != len {
                return None;
            }
            len = len.checked_mul(*size)?;
        }
        if len == 0 {
            continue;
        }
        let distance = a.offset.abs_diff(b.offset);
        let steps = u64::try_from(hi.checked_sub(lo)?.checked_sub(1)?).ok()?;
        if steps > 0 && (!distance.is_multiple_of(steps) || distance / steps > len) {
            return None;
        }
        let width = len.checked_add(distance)?;
        result.push((
            View {
                offset: a.offset.min(b.offset),
                shape: vec![width],
                strides: vec![1],
                ..a
            },
            mode,
        ));
    }
    Some(result)
}
fn affine_integer(e: &Expr, frame: &Frame, name: &str) -> Option<()> {
    let mut expression = e.sym.clone().or_else(|| {
        if let ExprKind::Int(n) = e.kind {
            Some(Sym::constant(n))
        } else {
            None
        }
    })?;
    for (parameter, value) in &frame.env {
        if parameter != name {
            expression = expression.subst(&Atom::Param(parameter.clone()), &Sym::constant(*value));
        }
    }
    for (monomial, _) in expression.monomials() {
        if monomial.is_empty() {
            continue;
        }
        if monomial.len() != 1 || monomial.get(&Atom::Param(name.into())) != Some(&1) {
            return None;
        }
    }
    Some(())
}
fn affine_view(e: &Expr, frame: &Frame, name: &str) -> Option<()> {
    match &e.kind {
        ExprKind::Var(id) => frame.views.contains_key(id).then_some(()),
        ExprKind::Transpose(base) => affine_view(base, frame, name),
        ExprKind::Index { base, indices } => {
            affine_view(base, frame, name)?;
            for index in indices {
                match index {
                    Index::Point(e) => affine_integer(e, frame, name)?,
                    Index::Slice { start, end } => {
                        for e in start.iter().chain(end) {
                            affine_integer(e, frame, name)?
                        }
                    }
                }
            }
            Some(())
        }
        _ => None,
    }
}
fn add<'a>(e: &'a Expr, mode: Mode, out: &mut Vec<(&'a Expr, Mode)>) -> Option<()> {
    if let ExprKind::Tuple(items) = &e.kind {
        for item in items {
            add(item, mode, out)?
        }
    } else {
        out.push((e, mode));
    }
    Some(())
}
fn collect<'a>(
    body: &'a [Stmt],
    out: &mut Vec<(&'a Expr, Mode)>,
    budget: &mut usize,
) -> Option<()> {
    for statement in body {
        *budget = budget.checked_sub(1)?;
        match &statement.kind {
            StmtKind::Reduction(r) => {
                for e in r.operands() {expression(e,out,budget)?;}
                if r.bodies().any(tensor_body) {return None;}
            }
            StmtKind::Assign { target, value, op } => {
                if matches!(target.ty, Ty::Tensor(_)) && matches!(target.kind, ExprKind::Var(_)) {
                    return None;
                }
                expression(value, out, budget)?;
                if let ExprKind::Index { base, .. } = &target.kind {
                    if matches!(base.ty, Ty::Tensor(_)) {
                        add(
                            target,
                            if *op == seismic_lang::ast::AssignOp::Assign {
                                Mode::Write
                            } else {
                                Mode::ReadWrite
                            },
                            out,
                        )?;
                    }
                }
            }
            StmtKind::Expr(e) => expression(e, out, budget)?,
            StmtKind::If { cond, then, els } => {
                if tensor_body(then) || tensor_body(els) {
                    return None;
                }
                expression(cond, out, budget)?;
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Lanes { body, .. } => {
                if tensor_body(body) {
                    return None;
                }
            }
            StmtKind::LoadLoop { .. } => return None,
        }
    }
    Some(())
}
fn expression<'a>(e: &'a Expr, out: &mut Vec<(&'a Expr, Mode)>, budget: &mut usize) -> Option<()> {
    *budget = budget.checked_sub(1)?;
    match &e.kind {
        ExprKind::Builtin { name, args } => {
            for arg in args {
                expression(arg, out, budget)?
            }
            match name {
                Builtin::Load => add(&args[0], Mode::Read, out)?,
                Builtin::Store => add(&args[1], Mode::Write, out)?,
                Builtin::Atomic => add(&args[0], Mode::ReadWrite, out)?,
                _ => {}
            }
        }
        ExprKind::Call { args, .. } => {
            if args.iter().any(tensor_expr) {
                return None;
            }
            for arg in args {
                expression(arg, out, budget)?
            }
        }
        ExprKind::Index { base, indices } => {
            expression(base, out, budget)?;
            for index in indices {
                match index {
                    Index::Point(e) => expression(e, out, budget)?,
                    Index::Slice { start, end } => {
                        for e in start.iter().chain(end) {
                            expression(e, out, budget)?
                        }
                    }
                }
            }
            if matches!(base.ty, Ty::Tensor(_)) && matches!(e.ty, Ty::Scalar(_)) {
                add(e, Mode::Read, out)?
            }
        }
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } | ExprKind::Transpose(expr) => {
            expression(expr, out, budget)?
        }
        ExprKind::Binary { op, lhs, rhs } => {
            if matches!(op, BinaryOp::And | BinaryOp::Or) && tensor_expr(rhs) {
                return None;
            }
            expression(lhs, out, budget)?;
            expression(rhs, out, budget)?;
        }
        ExprKind::Tuple(items) => {
            for item in items {
                expression(item, out, budget)?
            }
        }
        ExprKind::Intrinsic { .. } | ExprKind::Accessor { .. } | ExprKind::Lanes { .. } => {
            return None
        }
        _ => {}
    }
    Some(())
}
