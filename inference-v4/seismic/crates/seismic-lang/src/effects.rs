//! Conservative lifetime facts used to establish legality, never performance scores.
use crate::ir::{Builtin, Expr, ExprKind, Index, Stmt, StmtKind, VarId};

/// A load may borrow its tensor backing only when no tensor store or unknown
/// effect can run before its final consumption in the defining lexical block.
/// Tile assignment is by value, so assigning the loaded tile consumes a snapshot.
/// Runtime bindings must separately uphold the checked parallel-independence facts.
pub fn load_can_borrow(body: &[Stmt], var: VarId) -> bool {
    fn definitions(body: &[Stmt], var: VarId) -> usize {
        body.iter()
            .map(|s| match &s.kind {
                StmtKind::Assign { target, .. } => {
                    usize::from(matches!(target.kind, ExprKind::Var(v) if v == var))
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => definitions(body, var),
                StmtKind::If { then, els, .. } => definitions(then, var) + definitions(els, var),
                _ => 0,
            })
            .sum()
    }
    fn find(body: &[Stmt], var: VarId) -> Option<bool> {
        if let Some(definition) = body.iter().position(|s| matches!(&s.kind, StmtKind::Assign { target,value,.. }
            if matches!(target.kind, ExprKind::Var(v) if v==var) && matches!(value.kind,ExprKind::Builtin {name:Builtin::Load,..} | ExprKind::Load {..}))) {
            let last = body.iter().rposition(|s|uses(s,var)).unwrap_or(definition);
            return Some(last <= definition || !body[definition+1..=last].iter().any(|s|tensor_effect(s)||tile_mutated(s,var)));
        }
        for (at, s) in body.iter().enumerate() {
            let found = match &s.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => find(body, var),
                StmtKind::If { then, els, .. } => {
                    find(then, var).or_else(|| find(els, var))
                }
                _ => None,
            };
            if let Some(valid) = found {
                // A lexical lifetime proof cannot admit a view escaping that scope.
                return Some(
                    valid
                        && !body
                            .iter()
                            .enumerate()
                            .any(|(i, s)| i != at && uses(s, var)),
                );
            }
        }
        None
    }
    definitions(body, var) == 1 && find(body, var) == Some(true)
}
/// A streamed binding's snapshot may borrow only when the complete piece body
/// preserves its backing memory and does not mutate the loaded tile.
pub fn stream_load_can_borrow(body: &[Stmt], var: VarId) -> bool {
    !body
        .iter()
        .any(|stmt| tensor_effect(stmt) || tile_mutated(stmt, var))
}

fn expressions(e: &Expr, predicate: &impl Fn(&Expr) -> bool) -> bool {
    if predicate(e) {
        return true;
    }
    match &e.kind {
        ExprKind::Index { base, indices } => {
            expressions(base, predicate)
                || indices.iter().any(|i| match i {
                    Index::Point(e) => expressions(e, predicate),
                    Index::Slice { start, end } => {
                        start.iter().chain(end).any(|e| expressions(e, predicate))
                    }
                })
        }
        ExprKind::Load { view: e, .. }
        | ExprKind::Transpose(e)
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. } => expressions(e, predicate),
        ExprKind::Binary { lhs, rhs, .. } => {
            expressions(lhs, predicate) || expressions(rhs, predicate)
        }
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => args.iter().any(|e| expressions(e, predicate)),
        _ => false,
    }
}
fn statement(s: &Stmt, predicate: &impl Fn(&Expr) -> bool) -> bool {
    match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            expressions(target, predicate) || expressions(value, predicate)
        }
        StmtKind::Expr(e) => expressions(e, predicate),
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. } => body.iter().any(|s| statement(s, predicate)),
        StmtKind::Owned { tile, body, .. } => {
            expressions(tile, predicate) || body.iter().any(|s| statement(s, predicate))
        }
        StmtKind::LoadLoop { views, body, .. } => {
            views.iter().any(|e| expressions(e, predicate))
                || body.iter().any(|s| statement(s, predicate))
        }
        StmtKind::If { cond, then, els } => {
            expressions(cond, predicate) || then.iter().chain(els).any(|s| statement(s, predicate))
        }
    }
}
fn uses(s: &Stmt, var: VarId) -> bool {
    statement(s, &|e| matches!(e.kind,ExprKind::Var(v) if v==var))
}
pub fn tensor_effect(s: &Stmt) -> bool {
    statement(s, &|e| match &e.kind {
        ExprKind::Call { .. }
        | ExprKind::Builtin {
            name: Builtin::Store | Builtin::Atomic,
            ..
        } => true,
        ExprKind::Intrinsic { op, .. } => op.writes_tensor_memory(),
        _ => false,
    })
}

/// Whether a statement can mutate a tile binding or its value storage. Tile copies
/// are independent; ordinary expression uses and read-only intrinsic parameters
/// do not create write effects. Unknown calls remain conservative.
pub fn tile_mutated(s: &Stmt, var: VarId) -> bool {
    let mentions = |e: &Expr| expressions(e, &|e| matches!(e.kind,ExprKind::Var(v) if v==var));
    if statement(s, &|e| match &e.kind {
        ExprKind::Call { args, .. } => args.iter().any(&mentions),
        ExprKind::Intrinsic { op, args } => op.writes_arguments().iter()
            .any(|a| args.get(*a).is_none_or(&mentions)),
        _ => false,
    }) {
        return true;
    }
    match &s.kind {
        StmtKind::Assign { target, .. } => mentions(target),
        StmtKind::Parallel { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::LoadLoop { body, .. }
        | StmtKind::Lanes { body, .. } => body.iter().any(|s| tile_mutated(s, var)),
        StmtKind::If { then, els, .. } => then
            .iter()
            .chain(els)
            .any(|s| tile_mutated(s, var)),
        _ => false,
    }
}
