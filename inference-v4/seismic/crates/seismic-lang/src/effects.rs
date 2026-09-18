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
pub fn stream_load_can_borrow(body: &[Stmt], var: VarId, source: &Expr) -> bool {
    let mut roots=Vec::new();
    fn root(e:&Expr)->Option<VarId>{match &e.kind {ExprKind::Var(v)=>Some(*v),ExprKind::Index{base,..}|ExprKind::Transpose(base)=>root(base),ExprKind::Builtin{name:Builtin::Reshape,args}=>root(&args[0]),_=>None}}
    let Some(v)=root(source) else {return false;};
    roots.push(v);
    !body
        .iter()
        .any(|stmt| tensor_effect(stmt) || tile_mutated(stmt, var) || roots.iter().any(|v|tile_mutated(stmt,*v)))
}

pub(crate) fn expressions(e: &Expr, predicate: &impl Fn(&Expr) -> bool) -> bool {
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
        StmtKind::Reduction(r) => r.operands().any(|e|expressions(e,predicate)) || r.bodies().flatten().any(|s|statement(s,predicate)),
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
        StmtKind::LoadLoop { domain, views, body, .. } => {
            expressions(&domain.view,predicate) || views.iter().any(|e| expressions(e, predicate))
                || body.iter().any(|s| statement(s, predicate))
        }
        StmtKind::If { cond, then, els } => {
            expressions(cond, predicate) || then.iter().chain(els).any(|s| statement(s, predicate))
        }
    }
}
pub fn uses(s: &Stmt, var: VarId) -> bool {
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

/// Whether omitting a checked portable expression can discard an observable
/// effect or numerical failure. Bounds rely on the original checked invocation
/// contract; numerical operations retain their source preconditions. Callers
/// separately establish that the value is unused or independently recomputed.
pub fn expression_can_be_omitted(expr: &Expr) -> bool {
    use crate::ast::BinaryOp::{Div, Rem, Shl, Shr};
    use crate::types::{Elem, Ty};
    !expressions(expr, &|e| match &e.kind {
        ExprKind::Call { .. }
        | ExprKind::Intrinsic { .. }
        | ExprKind::Builtin { name: Builtin::Store | Builtin::Atomic, .. } => true,
        // A reshape observes layout validity even if only its extent is used.
        // Type-compatible dimensions alone do not prove a borrowed view can
        // be reshaped without copying (e.g. a noncontiguous tensor slice).
        ExprKind::Builtin { name: Builtin::Reshape, .. } => true,
        // Symbolic points were checked under the retained source conditions.
        // Data-dependent points still carry an execution-time bounds check;
        // dropping a coordinate cannot silently drop that failure. Slices use
        // the language's clamped-window semantics, so they need no such rule.
        ExprKind::Index { indices, .. } => indices.iter().any(|index| {
            matches!(index, Index::Point(point) if point.sym.is_none())
        }),
        ExprKind::Binary { op, lhs, rhs } if matches!(op, Div | Rem | Shl | Shr) => {
            let dtype = match &e.ty {
                Ty::Scalar(d) => Some(*d),
                Ty::Tile(s) => match s.elem { Elem::Dtype(d) => Some(d), _ => None },
                _ => None,
            };
            let Some(dtype) = dtype.filter(|d| d.is_int()) else { return false; };
            let constant = |e: &Expr| e.sym.as_ref()?.as_constant()
                .map(|n| crate::numeric::integer_value(dtype, n as u32));
            match op {
                Div | Rem => !crate::numeric::integer_division_is_defined(dtype, constant(lhs), constant(rhs)),
                Shl | Shr => !crate::numeric::integer_shift_is_defined(constant(rhs)),
                _ => unreachable!(),
            }
        }
        _ => false,
    })
}

/// Whether a checked expression can use its symbolic value without evaluating
/// its expression tree. Shape queries establish view metadata even when their
/// result is statically known and their evaluation cannot otherwise fail.
pub fn can_substitute_symbolic_value(expr: &Expr) -> bool {
    expr.sym.is_some()
        && expression_can_be_omitted(expr)
        && !expressions(expr, &|e| {
            matches!(e.kind, ExprKind::Builtin { name: Builtin::Extent, .. })
        })
}

/// Whether a statement can mutate a tile binding or its value storage. Tile copies
/// are independent; ordinary expression uses and read-only intrinsic parameters
/// do not create write effects. Unknown calls remain conservative.
pub fn tile_mutated(s: &Stmt, var: VarId) -> bool {
    if matches!(&s.kind, StmtKind::Reduction(r) if r.state_variables().any(|v|v==var)) { return true; }
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

/// Tensor inputs are immutable only when no store can reach their backing.
/// Tensor views alias; loaded tiles and scalar values are independent snapshots.
/// Unresolved calls conservatively may write any tensor passed to them.
pub fn tensor_parameter_read_only(body: &[Stmt], root: VarId) -> bool {
    use std::collections::HashSet;
    fn backing(e: &Expr) -> Option<VarId> {
        match &e.kind {
            ExprKind::Var(v) => Some(*v),
            ExprKind::Index { base, .. } | ExprKind::Transpose(base) => backing(base),
            ExprKind::Builtin { name: Builtin::Reshape, args } => args.first().and_then(backing),
            _ => None,
        }
    }
    fn walk(body: &[Stmt], visit: &mut impl FnMut(&Stmt)) {
        for s in body {
            visit(s);
            match &s.kind {
                StmtKind::Range { body, .. } | StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Lanes { body, .. }
                | StmtKind::LoadLoop { body, .. } => walk(body, visit),
                StmtKind::If { then, els, .. } => { walk(then, visit); walk(els, visit); }
                StmtKind::Reduction(r) => for body in r.bodies() { walk(body, visit); },
                _ => {}
            }
        }
    }
    let mut aliases = HashSet::from([root]);
    loop {
        let before = aliases.len();
        walk(body, &mut |s| {
            if let StmtKind::Assign { target, value, .. } = &s.kind {
                if matches!(value.ty, crate::types::Ty::Tensor(_)) && backing(value).is_some_and(|v| aliases.contains(&v)) {
                    if let ExprKind::Var(v) = target.kind { aliases.insert(v); }
                }
            }
        });
        if before == aliases.len() { break; }
    }
    let aliases_input = |e: &Expr| backing(e).is_some_and(|v| aliases.contains(&v));
    let mut writes_element = false;
    walk(body, &mut |s| {
        if let StmtKind::Assign { target, .. } = &s.kind {
            if matches!(target.kind, ExprKind::Index { .. }) && aliases_input(target) { writes_element = true; }
        }
    });
    !writes_element && !body.iter().any(|s| statement(s, &|e| match &e.kind {
        ExprKind::Builtin { name: Builtin::Store, args } => args.get(1).is_none_or(&aliases_input),
        ExprKind::Builtin { name: Builtin::Atomic, args } => args.first().is_none_or(&aliases_input),
        ExprKind::Call { args, .. } => args.iter().any(&aliases_input),
        ExprKind::Intrinsic { op, args } => op.writes_arguments().iter().any(|&i| args.get(i).is_none_or(&aliases_input)),
        _ => false,
    }))
}
