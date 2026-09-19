//! Variables a statement of the execution IR writes.

use super::ir::{Builtin, Expr, ExprKind, Index, Stmt, StmtKind, Var, VarId};
use super::types::Ty;
use std::collections::HashSet;

/// Variables a statement assigns, directly or through an element write.
pub fn writes(s: &Stmt, out: &mut HashSet<VarId>) {
    match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            expression_writes(value, out);
            expression_writes(target, out);
            let mut node = target;
            loop {
                match &node.kind {
                    ExprKind::Var(v) => {
                        out.insert(*v);
                        return;
                    }
                    ExprKind::Index { base, .. } => node = base,
                    _ => return,
                }
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            if let ExprKind::Var(v) = tile.kind {
                out.insert(v);
            }
            for b in body {
                writes(b, out);
            }
        }
        StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Parallel { body, .. } => {
            for b in body {
                writes(b, out);
            }
        }
        StmtKind::If { then, els, .. } => {
            for b in then.iter().chain(els.iter()) {
                writes(b, out);
            }
        }
        StmtKind::Expr(expr) => expression_writes(expr, out),
    }
}

/// Values requiring distinct bindings when copying a computation. Publishing
/// through a tensor reference mutates its backing, not the reference itself.
/// Effect/dependency analysis must still use `writes`, including those backings.
pub fn value_writes(s: &Stmt, vars: &[Var], out: &mut HashSet<VarId>) {
    let mut changed = HashSet::new();
    writes(s, &mut changed);
    out.extend(changed.into_iter().filter(|&v| !matches!(vars[v].ty, Ty::Tensor(_))));
    fn bindings(s: &Stmt, out: &mut HashSet<VarId>) {
        match &s.kind {
            StmtKind::Assign { target: Expr { kind: ExprKind::Var(v), .. }, .. } => { out.insert(*v); }
            StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } | StmtKind::Parallel { body, .. } => {
                for statement in body { bindings(statement, out); }
            }
            StmtKind::If { then, els, .. } => { for statement in then.iter().chain(els) { bindings(statement, out); } }
            _ => {}
        }
    }
    bindings(s, out);
}

/// Intrinsics mutate value storage through typed operand effects even when
/// written as expression statements. They participate in the same dependency
/// closure as ordinary tile assignments.
fn expression_writes(expr: &Expr, out: &mut HashSet<VarId>) {
    fn root(expr: &Expr) -> Option<VarId> {
        match &expr.kind {
            ExprKind::Var(v) => Some(*v),
            ExprKind::Index { base, .. } | ExprKind::Transpose(base) => root(base),
            ExprKind::Builtin { name: Builtin::Reshape, args } => args.first().and_then(root),
            _ => None,
        }
    }
    match &expr.kind {
        ExprKind::Intrinsic { op, args } => {
            for &index in op.writes_arguments() {
                if let Some(v) = args.get(index).and_then(root) { out.insert(v); }
            }
            for argument in args { expression_writes(argument, out); }
        }
        ExprKind::Builtin { name: Builtin::Store, args } => {
            if let Some(v) = args.get(1).and_then(root) { out.insert(v); }
            for argument in args { expression_writes(argument, out); }
        }
        ExprKind::Builtin { name: Builtin::Atomic, args } => {
            if let Some(v) = args.first().and_then(root) { out.insert(v); }
            for argument in args { expression_writes(argument, out); }
        }
        ExprKind::Call { args, .. } => {
            // Unresolved callees have not established a read-only contract.
            for argument in args {
                if argument.ty.shaped().is_some() {
                    if let Some(v) = root(argument) { out.insert(v); }
                }
                expression_writes(argument, out);
            }
        }
        ExprKind::Builtin { args, .. } | ExprKind::Tuple(args) => {
            for argument in args { expression_writes(argument, out); }
        }
        ExprKind::Index { base, indices } => {
            expression_writes(base, out);
            for index in indices {
                match index {
                    Index::Point(expr) => expression_writes(expr, out),
                    Index::Slice { start, end } => for expr in start.iter().chain(end) { expression_writes(expr, out); },
                }
            }
        }
        ExprKind::Load { view: expr, .. } | ExprKind::Transpose(expr)
        | ExprKind::Accessor { base: expr, .. } | ExprKind::Lanes { base: expr, .. }
        | ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => expression_writes(expr, out),
        ExprKind::Binary { lhs, rhs, .. } => { expression_writes(lhs, out); expression_writes(rhs, out); }
        _ => {}
    }
}
