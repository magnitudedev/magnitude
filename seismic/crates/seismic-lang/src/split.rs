//! Splitting a streamed reduction across work items.
//!
//! A `parallel` body that streams a range and carries state across the pieces is sequential
//! in that range. When the work items a kernel already has cannot fill the machine, the
//! model splits the range: each item takes a slice, and the partial states combine.
//!
//! The combine needs no separate expression. A streaming body merges each piece into the
//! running state as it goes, so the body already contains the merge rule; running it over
//! the partial results of the slices produces the same answer. This transform therefore
//! rewrites the range and leaves the arithmetic alone, storing each slice's state and
//! re-running the loop's own tail over those states.

use crate::hir::{Expr, ExprKind, Index, Stmt, StmtKind, VarId, VarKind};
use crate::sym::{Atom, Sym};
use crate::types::Ty;

/// A streamed range that carries state, and so can be split across items.
#[derive(Clone, Debug)]
pub struct Splittable {
    /// Index of the `parallel` statement in the kernel body.
    pub stmt: usize,
    /// Position of the `LoadLoop` within that block's statements.
    pub loop_at: usize,
    /// Tiles assigned before the loop and updated inside it.
    pub carried: Vec<VarId>,
    /// The range's start and end as written. Data-dependent bounds are ordinary
    /// expressions, so they are kept as expressions rather than symbolic values.
    pub lo: Expr,
    pub hi: Expr,
}

/// Find the streamed ranges of a kernel that carry state across pieces. These are the only
/// sequential work a `parallel` block has, and so the only thing a split can help.
pub fn splittable(body: &[Stmt], vars: &[Var]) -> Vec<Splittable> {
    let mut out = Vec::new();
    for (i, s) in body.iter().enumerate() {
        let StmtKind::Parallel { body: block, .. } = &s.kind else { continue };
        for (j, inner) in block.iter().enumerate() {
            let StmtKind::LoadLoop { views, axis, body: loop_body, .. } = &inner.kind else { continue };
            let carried = carried_tiles(&block[..j], loop_body, vars);
            let range = range_of(&views[0], *axis);
            if carried.is_empty() {
                continue;
            }
            let Some((lo, hi)) = range else { continue };
            out.push(Splittable { stmt: i, loop_at: j, carried, lo, hi });
        }
    }
    out
}

/// Tiles assigned before the loop and updated inside it: the loop's carried state.
fn carried_tiles(before: &[Stmt], loop_body: &[Stmt], vars: &[Var]) -> Vec<VarId> {
    let mut assigned = Vec::new();
    for s in before {
        collect_assigned(s, &mut assigned);
    }
    let mut updated = Vec::new();
    for s in loop_body {
        collect_assigned(s, &mut updated);
    }
    assigned
        .into_iter()
        .filter(|v| updated.contains(v) && matches!(vars[*v].ty, Ty::Tile(_)) && matches!(vars[*v].kind, VarKind::Local))
        .collect()
}

fn collect_assigned(s: &Stmt, out: &mut Vec<VarId>) {
    match &s.kind {
        StmtKind::Assign { target, .. } => {
            let mut node = target;
            loop {
                match &node.kind {
                    ExprKind::Var(v) => {
                        if !out.contains(v) {
                            out.push(*v);
                        }
                        return;
                    }
                    ExprKind::Index { base, .. } => node = base,
                    _ => return,
                }
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            if let ExprKind::Var(v) = tile.kind {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            for b in body {
                collect_assigned(b, out);
            }
        }
        StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } | StmtKind::Parallel { body, .. } => {
            for b in body {
                collect_assigned(b, out);
            }
        }
        StmtKind::If { then, els, .. } => {
            for b in then.iter().chain(els.iter()) {
                collect_assigned(b, out);
            }
        }
        StmtKind::Expr(_) => {}
    }
}

/// The `[lo, hi)` a streamed view covers along its axis, as expressions: the bounds may be
/// read from a tensor at run time, as an attention kernel's visible range is.
fn range_of(view: &Expr, axis: usize) -> Option<(Expr, Expr)> {
    let mut node = view;
    loop {
        match &node.kind {
            ExprKind::Tuple(items) => node = items.first()?,
            ExprKind::Transpose(inner) => node = inner,
            ExprKind::Index { base, indices } => {
                if let Some(Index::Slice { start: Some(lo), end: Some(hi) }) = indices.get(axis) {
                    return Some((lo.clone(), hi.clone()));
                }
                node = base;
            }
            _ => return None,
        }
    }
}

use crate::hir::Var;

/// Rewrite a splittable range so `parts` work items share it, each streaming a slice.
///
/// The `parallel` block gains an index over the parts. The streamed range narrows to that
/// part's slice: with `span = (hi - lo + parts - 1) / parts`, part `p` covers
/// `[lo + p * span, min(hi, lo + (p + 1) * span))`. Each item ends holding the carried
/// state for its slice, which the caller combines with the loop's own merge rule.
pub fn narrow_range(sp: &Splittable, body: &mut [Stmt], part: VarId, part_atom: &Atom, parts: i64) -> Result<(), String> {
    let StmtKind::Parallel { body: block, .. } = &mut body[sp.stmt].kind else {
        return Err("split target is not a parallel block".into());
    };
    let StmtKind::LoadLoop { views, axis, .. } = &mut block[sp.loop_at].kind else {
        return Err("split target is not a streaming loop".into());
    };
    let i32_ty = Ty::Scalar(crate::types::DType::I32);
    let p = Expr { kind: ExprKind::Var(part), ty: i32_ty.clone(), sym: Some(Sym::atom(part_atom.clone())), span: sp.lo.span };
    let n = |v: i64| Expr { kind: ExprKind::Int(v), ty: i32_ty.clone(), sym: Some(Sym::constant(v)), span: sp.lo.span };
    let bin = |op: crate::ast::BinaryOp, a: Expr, b: Expr| -> Expr {
        let sym = match (&a.sym, &b.sym) {
            (Some(x), Some(y)) => match op {
                crate::ast::BinaryOp::Add => Some(x.add(y)),
                crate::ast::BinaryOp::Sub => Some(x.sub(y)),
                crate::ast::BinaryOp::Mul => Some(x.mul(y)),
                crate::ast::BinaryOp::Div => Some(x.quot(y)),
                _ => None,
            },
            _ => None,
        };
        Expr { kind: ExprKind::Binary { op, lhs: Box::new(a), rhs: Box::new(b) }, ty: i32_ty.clone(), sym, span: sp.lo.span }
    };
    use crate::ast::BinaryOp::{Add, Div, Mul, Sub};
    // span = (hi - lo + parts - 1) / parts
    let width = bin(Sub, sp.hi.clone(), sp.lo.clone());
    let span = bin(Div, bin(Add, width, n(parts - 1)), n(parts));
    let start = bin(Add, sp.lo.clone(), bin(Mul, p.clone(), span.clone()));
    let unclamped = bin(Add, start.clone(), span);
    let end = Expr {
        kind: ExprKind::Builtin { name: crate::hir::Builtin::Min, args: vec![sp.hi.clone(), unclamped] },
        ty: i32_ty,
        sym: None,
        span: sp.lo.span,
    };
    for v in views.iter_mut() {
        replace_slice(v, *axis, &start, &end);
    }
    Ok(())
}

/// Replace the slice bounds on `axis` of every view in an expression, descending tuples.
fn replace_slice(view: &mut Expr, axis: usize, lo: &Expr, hi: &Expr) {
    match &mut view.kind {
        ExprKind::Tuple(items) => {
            for i in items {
                replace_slice(i, axis, lo, hi);
            }
        }
        ExprKind::Transpose(inner) => replace_slice(inner, axis, lo, hi),
        ExprKind::Index { base, indices } => {
            if let Some(slot @ Index::Slice { .. }) = indices.get_mut(axis) {
                *slot = Index::Slice { start: Some(lo.clone()), end: Some(hi.clone()) };
                return;
            }
            replace_slice(base, axis, lo, hi);
        }
        _ => {}
    }
}
