//! Validity of the compiler's own rewrites, checked on the IR.
//!
//! A rewrite claims to preserve what a body computes. That claim is a property of the
//! statements, not of any input data, so it is decided here by reading the IR: no device, no
//! interpreter, no shapes. A rewrite whose precondition does not hold is refused with the
//! reason, at the point it would have been applied.
//!
//! The rule every rewrite in this file answers to: **a statement may be moved out of a
//! repeated region only if the region neither reads a value the statement makes stale nor
//! rewrites what the statement wrote.**

use crate::ir::{Expr, ExprKind, Index, Stmt, StmtKind, VarId, VarKind};
use crate::sym::Atom;
use std::collections::HashSet;

/// Why a rewrite is not valid here.
#[derive(Clone, Debug)]
pub struct Invalid {
    pub rewrite: &'static str,
    pub reason: String,
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the `{}` rewrite does not apply here: {}", self.rewrite, self.reason)
    }
}

/// Hoisting the first `split` statements of `body` out of a region that repeats the rest.
///
/// Valid when each hoisted statement is independent of the repeated index, and nothing in the
/// repeated part writes a variable a hoisted statement wrote: a value computed once must not
/// be one the repetition is supposed to reset or update.
pub fn check_hoist(body: &[Stmt], split: usize, inner: VarId, atom: &Atom, names: &dyn Fn(VarId) -> String) -> Result<(), Invalid> {
    let hoisted = &body[..split];
    let repeated = &body[split..];
    let mut tainted: HashSet<VarId> = HashSet::new();
    tainted.insert(inner);
    for (i, s) in hoisted.iter().enumerate() {
        if depends(s, &tainted, atom) {
            return Err(Invalid {
                rewrite: "hoist",
                reason: format!("statement {i} depends on `{}`, which varies across the region, so computing it once would change the result", names(inner)),
            });
        }
    }
    let mut written_in_repeat: HashSet<VarId> = HashSet::new();
    for s in repeated {
        writes(s, &mut written_in_repeat);
    }
    for (i, s) in hoisted.iter().enumerate() {
        let mut w = HashSet::new();
        writes(s, &mut w);
        if let Some(v) = w.iter().find(|v| written_in_repeat.contains(v)) {
            return Err(Invalid {
                rewrite: "hoist",
                reason: format!("statement {i} writes `{}`, which the repeated region also writes, so it must run for every index rather than once", names(*v)),
            });
        }
    }
    Ok(())
}

/// Which statements of `body` may be computed once for the whole region, rather than once per
/// index in it. Unlike a prefix, this selects individual statements, because an accumulator
/// that must be reset per index does not prevent the work feeding it from being shared.
///
/// A statement is selected when it is independent of the index, the region does not rewrite
/// what it wrote, and every statement it depends on is also selected. Order is preserved.
pub fn hoistable_set(body: &[Stmt], inner: VarId, atom: &Atom) -> Vec<bool> {
    let per_stmt: Vec<HashSet<VarId>> = body
        .iter()
        .map(|s| {
            let mut w = HashSet::new();
            writes(s, &mut w);
            w
        })
        .collect();
    // Start by assuming everything can be shared, then remove statements until the set is
    // stable. Shrinking only, so it converges and never re-admits something it rejected.
    let mut selected = vec![true; body.len()];
    loop {
        // Values that vary across the region: the index, and whatever the repeated part writes.
        let mut varying: HashSet<VarId> = HashSet::new();
        varying.insert(inner);
        for (i, _) in body.iter().enumerate() {
            if !selected[i] {
                for v in &per_stmt[i] {
                    varying.insert(*v);
                }
            }
        }
        let mut changed = false;
        for (i, s) in body.iter().enumerate() {
            if !selected[i] {
                continue;
            }
            // Reading something that varies, or writing something the region rewrites.
            let reads_varying = depends(s, &varying, atom);
            let rewritten = (0..body.len()).any(|j| !selected[j] && per_stmt[i].iter().any(|v| per_stmt[j].contains(v)));
            if reads_varying || rewritten {
                selected[i] = false;
                changed = true;
            }
        }
        if !changed {
            return selected;
        }
    }
}

/// Whether hoisting exactly the selected statements preserves what the body computes.
pub fn check_hoist_set(body: &[Stmt], selected: &[bool], inner: VarId, atom: &Atom, names: &dyn Fn(VarId) -> String) -> Result<(), Invalid> {
    let mut tainted: HashSet<VarId> = HashSet::new();
    tainted.insert(inner);
    for (i, s) in body.iter().enumerate() {
        if !selected[i] {
            let mut w = HashSet::new();
            writes(s, &mut w);
            for v in w {
                tainted.insert(v);
            }
        }
    }
    for (i, s) in body.iter().enumerate() {
        if !selected[i] {
            continue;
        }
        if depends(s, &tainted, atom) {
            return Err(Invalid {
                rewrite: "hoist",
                reason: format!("statement {i} depends on a value that varies across the region, so computing it once would change the result"),
            });
        }
        let mut w = HashSet::new();
        writes(s, &mut w);
        for (j, other) in body.iter().enumerate() {
            if selected[j] {
                continue;
            }
            let mut ow = HashSet::new();
            writes(other, &mut ow);
            if let Some(v) = w.iter().find(|v| ow.contains(v)) {
                return Err(Invalid {
                    rewrite: "hoist",
                    reason: format!("statement {i} writes `{}`, which statement {j} rewrites inside the region, so it must run for every index", names(*v)),
                });
            }
        }
    }
    Ok(())
}

/// The largest prefix of `body` that `check_hoist` accepts. Computed by the same rule the
/// check states, then checked, so a mistake in either is caught rather than silently applied.
pub fn hoistable_prefix(body: &[Stmt], inner: VarId, atom: &Atom) -> usize {
    let mut tainted: HashSet<VarId> = HashSet::new();
    tainted.insert(inner);
    let mut n = 0;
    for s in body {
        if depends(s, &tainted, atom) {
            break;
        }
        n += 1;
    }
    while n > 0 {
        let mut written_after: HashSet<VarId> = HashSet::new();
        for s in &body[n..] {
            writes(s, &mut written_after);
        }
        let mut w = HashSet::new();
        writes(&body[n - 1], &mut w);
        if w.iter().any(|v| written_after.contains(v)) {
            n -= 1;
        } else {
            break;
        }
    }
    n
}

/// Whether splitting a streamed range composes with one work item covering several index
/// tuples. It does not: a split item holds partial state that a later launch merges, but an
/// item covering several tuples would hold one such state per tuple while the carried tiles
/// exist once, outside the run. The two rewrites are therefore exclusive, and the model must
/// choose between them rather than emit code that computes something else.
pub fn check_split_with_multiplicity(per_item: i64) -> Result<(), Invalid> {
    if per_item > 1 {
        return Err(Invalid {
            rewrite: "split",
            reason: format!(
                "one work item already covers {per_item} index tuples, and a split item carries partial state per tuple that its merge cannot recover; apply one rewrite or the other"
            ),
        });
    }
    Ok(())
}

/// Splitting a streamed range across work items, merging the carried state afterwards.
///
/// Valid when the carried state's update is associative in the pieces: the loop body folds
/// each piece into the running state, so folding one part's state into another's is the same
/// operation. The check is that every carried tile is written by the loop and read by it,
/// and that nothing else the loop writes escapes it, since only carried state is merged.
pub fn check_split(loop_body: &[Stmt], carried: &[VarId], names: &dyn Fn(VarId) -> String) -> Result<(), Invalid> {
    let mut written = HashSet::new();
    for s in loop_body {
        writes(s, &mut written);
    }
    for v in carried {
        if !written.contains(v) {
            return Err(Invalid {
                rewrite: "split",
                reason: format!("`{}` is treated as carried state but the streamed loop never updates it", names(*v)),
            });
        }
    }
    // Anything else the loop writes and the tail reads would be lost, since only the carried
    // tiles are published and merged.
    let mut escaping: Vec<VarId> = written.iter().copied().filter(|v| !carried.contains(v)).collect();
    escaping.sort();
    for v in escaping {
        if is_local_temp(loop_body, v) {
            continue;
        }
        return Err(Invalid {
            rewrite: "split",
            reason: format!("the streamed loop writes `{}`, which is not carried state, so splitting the range would lose it", names(v)),
        });
    }
    Ok(())
}

/// Whether a variable is produced fresh by each iteration, so no part of it needs to survive
/// the split. Fresh means the loop gives it a whole new value before reading it: a tile
/// allocation or any assignment of the variable itself, as opposed to updating elements of a
/// value that came from outside.
fn is_local_temp(loop_body: &[Stmt], v: VarId) -> bool {
    fn defined_here(stmts: &[Stmt], v: VarId) -> bool {
        stmts.iter().any(|s| match &s.kind {
            StmtKind::Reduction(_) => false,
            // `x = ...` replaces the whole variable, so nothing of the previous value remains.
            StmtKind::Assign { target, op: crate::ast::AssignOp::Assign, .. } => {
                matches!(target.kind, ExprKind::Var(x) if x == v)
            }
            StmtKind::Assign { .. } => false,
            StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Parallel { body, .. } => defined_here(body, v),
            StmtKind::If { then, els, .. } => defined_here(then, v) || defined_here(els, v),
            StmtKind::Expr(_) => false,
        })
    }
    defined_here(loop_body, v)
}

/// Variables a statement assigns, directly or through an element write.
pub fn writes(s: &Stmt, out: &mut HashSet<VarId>) {
    match &s.kind {
        StmtKind::Reduction(r) => { out.extend(r.state_variables()); for s in r.bodies().flatten() {writes(s,out);} }
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
pub fn value_writes(s: &Stmt, vars: &[crate::ir::Var], out: &mut HashSet<VarId>) {
    let mut changed = HashSet::new();
    writes(s, &mut changed);
    out.extend(changed.into_iter().filter(|&v| !matches!(vars[v].ty, crate::types::Ty::Tensor(_))));
    fn bindings(s: &Stmt, out: &mut HashSet<VarId>) {
        match &s.kind {
            StmtKind::Assign { target: Expr { kind: ExprKind::Var(v), .. }, .. } => { out.insert(*v); }
            StmtKind::Reduction(r) => { for statement in r.bodies().flatten() { bindings(statement, out); } }
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
            ExprKind::Builtin { name: crate::ir::Builtin::Reshape, args } => args.first().and_then(root),
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
        ExprKind::Builtin { name: crate::ir::Builtin::Store, args } => {
            if let Some(v) = args.get(1).and_then(root) { out.insert(v); }
            for argument in args { expression_writes(argument, out); }
        }
        ExprKind::Builtin { name: crate::ir::Builtin::Atomic, args } => {
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

/// Whether a statement's value depends on an index, directly or through a tainted variable.
pub fn depends(s: &Stmt, tainted: &HashSet<VarId>, atom: &Atom) -> bool {
    match &s.kind {
        StmtKind::Reduction(r) => r.operands().any(|e|expr_depends(e,tainted,atom)) || r.extent().atoms().contains(atom) || r.bodies().flatten().any(|s|depends(s,tainted,atom)),
        StmtKind::Assign { target, value, .. } => expr_depends(target, tainted, atom) || expr_depends(value, tainted, atom),
        StmtKind::Expr(e) => expr_depends(e, tainted, atom),
        StmtKind::Owned { tile, body, .. } => expr_depends(tile, tainted, atom) || body.iter().any(|b| depends(b, tainted, atom)),
        StmtKind::Range { lo, hi, body, .. } => {
            lo.atoms().contains(atom) || hi.atoms().contains(atom) || body.iter().any(|b| depends(b, tainted, atom))
        }
        StmtKind::Lanes { extent, body, .. } => extent.atoms().contains(atom) || body.iter().any(|b| depends(b, tainted, atom)),
        StmtKind::LoadLoop { domain, views, body, .. } => {
            expr_depends(&domain.view,tainted,atom) || views.iter().any(|v| expr_depends(v, tainted, atom)) || body.iter().any(|b| depends(b, tainted, atom))
        }
        StmtKind::If { cond, then, els } => {
            expr_depends(cond, tainted, atom) || then.iter().chain(els.iter()).any(|b| depends(b, tainted, atom))
        }
        StmtKind::Parallel { .. } => true,
    }
}

fn expr_depends(e: &Expr, tainted: &HashSet<VarId>, atom: &Atom) -> bool {
    if let Some(sym) = &e.sym {
        if sym.atoms().contains(atom) {
            return true;
        }
    }
    match &e.kind {
        ExprKind::Var(v) => tainted.contains(v),
        ExprKind::Index { base, indices } => {
            expr_depends(base, tainted, atom)
                || indices.iter().any(|i| match i {
                    Index::Point(p) => expr_depends(p, tainted, atom),
                    Index::Slice { start, end } => {
                        start.as_ref().is_some_and(|x| expr_depends(x, tainted, atom)) || end.as_ref().is_some_and(|x| expr_depends(x, tainted, atom))
                    }
                })
        }
        ExprKind::Load { view: x, .. } | ExprKind::Transpose(x) | ExprKind::Accessor { base: x, .. } | ExprKind::Lanes { base: x, .. } | ExprKind::Unary { expr: x, .. } | ExprKind::Cast { expr: x, .. } => expr_depends(x, tainted, atom),
        ExprKind::Builtin { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Call { args, .. } | ExprKind::Tuple(args) => args.iter().any(|a| expr_depends(a, tainted, atom)),
        ExprKind::Binary { lhs, rhs, .. } => expr_depends(lhs, tainted, atom) || expr_depends(rhs, tainted, atom),
        _ => false,
    }
}

/// A variable's name, for messages.
pub fn namer(vars: &[crate::ir::Var]) -> impl Fn(VarId) -> String + '_ {
    move |v| vars.get(v).map(|x| x.name.clone()).unwrap_or_else(|| format!("#{v}"))
}

/// Unused today, kept so `VarKind` stays imported for future checks.
#[allow(dead_code)]
fn _kind(_k: &VarKind) {}
