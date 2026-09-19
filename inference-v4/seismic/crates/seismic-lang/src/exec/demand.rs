//! Element-data demand on the existing IR. Geometry remains a value even when
//! its backing elements are unused: evaluating a view still checks its bounds,
//! captures its endpoints and validates its layout at the original definition.
use super::ir::{Builtin, Expr, ExprKind, Index, Stmt, StmtKind, VarId};
use super::types::Ty;
use crate::syntax::ast::AssignOp;
use crate::types::Elem;
use std::collections::HashSet;

/// Variables whose element storage is needed by this body. This is a
/// conservative, definition-independent fixed point: if any version needs data,
/// every version keeps its ordinary storage realization. It does not authorize
/// omitting computations or effects; a separate rewrite must prove that legal.
pub fn data_variables(body: &[Stmt]) -> HashSet<VarId> {
    let mut data = HashSet::new();
    loop {
        let before = data.len();
        statements(body, &mut data);
        if data.len() == before {
            return data;
        }
    }
}

/// Whether an expression's shaped result has a geometry-only realization.
/// Unknown computations remain data-demanding even if their result is unused.
pub fn has_geometry(expr: &Expr) -> bool {
    if !expr
        .ty
        .shaped()
        .is_some_and(|s| matches!(s.elem, Elem::Dtype(_)))
    {
        return false;
    }
    match &expr.kind {
        ExprKind::Var(_) | ExprKind::TileAlloc { .. } => expr.ty.shaped().is_some(),
        ExprKind::Index { base, .. } => expr.ty.shaped().is_some() && has_geometry(base),
        ExprKind::Transpose(base) | ExprKind::Load { view: base, .. } => has_geometry(base),
        ExprKind::Builtin {
            name: Builtin::Load | Builtin::Reshape,
            args,
        } => args.first().is_some_and(has_geometry),
        _ => false,
    }
}

fn expression(expr: &Expr, elements: bool, data: &mut HashSet<VarId>) {
    let elements = elements || matches!(expr.ty, Ty::Scalar(_));
    match &expr.kind {
        ExprKind::Var(v) => {
            if elements {
                data.insert(*v);
            }
        }
        ExprKind::Index { base, indices } => {
            expression(base, elements, data);
            for index in indices {
                match index {
                    Index::Point(point) => expression(point, true, data),
                    Index::Slice { start, end } => {
                        for bound in start.iter().chain(end) {
                            expression(bound, true, data);
                        }
                    }
                }
            }
        }
        ExprKind::Transpose(base) | ExprKind::Load { view: base, .. } => {
            expression(base, elements, data);
        }
        ExprKind::Builtin {
            name: Builtin::Extent,
            args,
        } => {
            if let Some(view) = args.first() {
                expression(view, false, data);
            }
            for axis in args.iter().skip(1) {
                expression(axis, true, data);
            }
        }
        ExprKind::Builtin {
            name: Builtin::Load | Builtin::Reshape,
            args,
        } => {
            if let Some(view) = args.first() {
                expression(view, elements, data);
            }
            for dimension in args.iter().skip(1) {
                expression(dimension, true, data);
            }
        }
        ExprKind::Tuple(items) => {
            for item in items {
                expression(item, elements, data);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            expression(lhs, true, data);
            expression(rhs, true, data);
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast { expr, .. }
        | ExprKind::Accessor { base: expr, .. }
        | ExprKind::Lanes { base: expr, .. } => {
            expression(expr, true, data);
        }
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. } => {
            for argument in args {
                expression(argument, true, data);
            }
        }
        _ => {}
    }
}

fn statements(body: &[Stmt], data: &mut HashSet<VarId>) {
    for statement in body {
        match &statement.kind {
            StmtKind::Assign {
                target,
                op: AssignOp::Assign,
                value,
            } if matches!(target.kind, ExprKind::Var(_)) && target.ty.shaped().is_some() => {
                let ExprKind::Var(variable) = target.kind else {
                    unreachable!()
                };
                if !has_geometry(value) {
                    data.insert(variable);
                }
                expression(value, data.contains(&variable), data);
            }
            StmtKind::Assign { target, value, .. } => {
                expression(target, true, data);
                expression(value, true, data);
            }
            StmtKind::Expr(expr) => expression(expr, true, data),
            StmtKind::Owned { tile, body, .. } => {
                expression(tile, false, data);
                statements(body, data);
            }
            StmtKind::LoadLoop {
                domain,
                vars,
                views,
                body,
                ..
            } => {
                expression(&domain.view, false, data);
                for (variable, view) in vars.iter().zip(views) {
                    // A piece retains its geometry even when none of its elements
                    // are read. The fixed point propagates any body data use back
                    // through the transfer; unsupported geometry stays materialized.
                    if !has_geometry(view) {
                        data.insert(*variable);
                    }
                    expression(view, data.contains(variable), data);
                }
                statements(body, data);
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } => statements(body, data),
            StmtKind::If { cond, then, els } => {
                expression(cond, true, data);
                statements(then, data);
                statements(els, data);
            }
        }
    }
}
