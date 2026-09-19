//! Structural candidates for splitting streamed state across work items.
//!
//! Finding a carried state is not proof that it can be split. The backend must
//! derive an applicable merge law, preserve the original runtime domain checks,
//! and represent publication and merge ordering before applying this transform.
//! Narrowing a view updates both its bounds and its checked extent metadata.

use crate::ir::{Expr, ExprKind, Index, Stmt, StmtKind, VarId, VarKind};
use crate::sym::{Atom, Sym};
use crate::types::Ty;

pub mod parameterized;

/// A streamed range with carried state requiring a merge proof before splitting.
#[derive(Clone, Debug)]
pub struct SplitCandidate {
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

/// Find streamed ranges with carried state. This does not establish a merge law.
pub fn split_candidates(body: &[Stmt], vars: &[Var]) -> Vec<SplitCandidate> {
    let mut out = Vec::new();
    for (i, s) in body.iter().enumerate() {
        let StmtKind::Parallel { body: block, .. } = &s.kind else {
            continue;
        };
        for (j, inner) in block.iter().enumerate() {
            let StmtKind::LoadLoop {
                domain,
                body: loop_body,
                ..
            } = &inner.kind
            else {
                continue;
            };
            let carried = carried_tiles(&block[..j], loop_body, vars);
            let range = range_of(&domain.view, domain.axis);
            if carried.is_empty() {
                continue;
            }
            let Some((lo, hi)) = range else { continue };
            out.push(SplitCandidate {
                stmt: i,
                loop_at: j,
                carried,
                lo,
                hi,
            });
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
        .filter(|v| {
            updated.contains(v)
                && matches!(vars[*v].ty, Ty::Tile(_))
                && matches!(vars[*v].kind, VarKind::Local)
        })
        .collect()
}

fn collect_assigned(s: &Stmt, out: &mut Vec<VarId>) {
    match &s.kind {
        StmtKind::Reduction(r) => {
            for v in r.state_variables() {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
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
        StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. }
        | StmtKind::LoadLoop { body, .. }
        | StmtKind::Parallel { body, .. } => {
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
fn input_axis(indices: &[Index], mut axis: usize) -> usize {
    for (at, index) in indices.iter().enumerate() {
        if matches!(index, Index::Point(_)) {
            continue;
        }
        if axis == 0 {
            return at;
        }
        axis -= 1;
    }
    indices.len() + axis
}

fn range_of(view: &Expr, mut axis: usize) -> Option<(Expr, Expr)> {
    let mut node = view;
    loop {
        match &node.kind {
            ExprKind::Tuple(items) => node = items.first()?,
            ExprKind::Transpose(inner) => {
                axis = 1usize.checked_sub(axis)?;
                node = inner;
            }
            ExprKind::Index { base, indices } => {
                axis = input_axis(indices, axis);
                if let Some(Index::Slice {
                    start: Some(lo),
                    end: Some(hi),
                }) = indices.get(axis)
                {
                    return Some((lo.clone(), hi.clone()));
                }
                node = base;
            }
            _ => return None,
        }
    }
}

use crate::ir::Var;

/// Rewrite a splittable range so `parts` work items share it, each streaming a slice.
///
/// The caller declares and binds the part index. The streamed range narrows to that
/// part's slice: with `span = (hi - lo + parts - 1) / parts`, part `p` covers
/// `[min(hi, lo + p * span), min(hi, lo + (p + 1) * span))`. Each item ends holding the carried
/// state for its slice, which the caller combines with the loop's own merge rule.
pub fn narrow_range(
    sp: &SplitCandidate,
    body: &mut [Stmt],
    vars: &mut Vec<Var>,
    part: VarId,
    part_atom: &Atom,
    parts: i64,
) -> Result<usize, String> {
    let StmtKind::Parallel { body: block, .. } = &mut body[sp.stmt].kind else {
        return Err("split target is not a parallel block".into());
    };
    let StmtKind::LoadLoop {
        views,
        axes,
        domain,
        offset,
        body: piece_body,
        ..
    } = &mut block[sp.loop_at].kind
    else {
        return Err("split target is not a streaming loop".into());
    };
    if parts <= 0 {
        return Err("split part count must be positive".into());
    }
    let mut origin = None;
    if views.len() != axes.len() {
        return Err("stream transfer axes disagree with bindings".into());
    }
    for (view_index, (v, axis)) in std::iter::once((&mut domain.view, domain.axis))
        .chain(views.iter_mut().zip(axes.iter().copied()))
        .enumerate()
    {
        let (lo, hi) =
            range_of(v, axis).ok_or("split requires an explicit range for every streamed view")?;
        let i32_ty = Ty::Scalar(crate::types::DType::I32);
        let p = Expr {
            kind: ExprKind::Var(part),
            ty: i32_ty.clone(),
            sym: Some(Sym::atom(part_atom.clone())),
            span: lo.span,
        };
        let n = |v: i64| Expr {
            kind: ExprKind::Int(v),
            ty: i32_ty.clone(),
            sym: Some(Sym::constant(v)),
            span: lo.span,
        };
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
            Expr {
                kind: ExprKind::Binary {
                    op,
                    lhs: Box::new(a),
                    rhs: Box::new(b),
                },
                ty: i32_ty.clone(),
                sym,
                span: lo.span,
            }
        };
        use crate::ast::BinaryOp::{Add, Div, Mul, Sub};
        // span = (hi - lo + parts - 1) / parts
        let width = bin(Sub, hi.clone(), lo.clone());
        let span = bin(Div, bin(Add, width, n(parts - 1)), n(parts));
        let raw_start = bin(Add, lo.clone(), bin(Mul, p.clone(), span.clone()));
        let start = Expr {
            kind: ExprKind::Builtin {
                name: crate::ir::Builtin::Min,
                args: vec![hi.clone(), raw_start],
            },
            ty: i32_ty.clone(),
            sym: None,
            span: lo.span,
        };
        let unclamped = bin(Add, start.clone(), span);
        let end = Expr {
            kind: ExprKind::Builtin {
                name: crate::ir::Builtin::Min,
                args: vec![hi.clone(), unclamped],
            },
            ty: i32_ty.clone(),
            sym: None,
            span: lo.span,
        };
        if view_index == 0 && offset.is_some() {
            origin = Some(bin(Sub, start.clone(), lo.clone()));
        }
        if !replace_slice(
            v,
            axis,
            &start,
            &end,
            &Sym::param(&format!("split_extent#{part}_{view_index}")),
        ) {
            return Err("split could not narrow every streamed view".into());
        }
    }
    if let (Some(index), Some(origin)) = (*offset, origin) {
        let VarKind::Index(atom) = &vars[index].kind else {
            return Err("stream offset is not an index binding".into());
        };
        let atom = atom.clone();
        let base = vars.len();
        let base_atom = Atom::Param(format!("split_origin#{base}"));
        let ty = Ty::Scalar(crate::types::DType::I32);
        let span = origin.span;
        vars.push(Var {
            name: format!("split_origin_{base}"),
            ty: ty.clone(),
            span,
            kind: VarKind::Index(base_atom.clone()),
        });
        let base_expr = Expr {
            kind: ExprKind::Var(base),
            ty: ty.clone(),
            sym: Some(Sym::atom(base_atom.clone())),
            span,
        };
        let index_expr = Expr {
            kind: ExprKind::Var(index),
            ty: ty.clone(),
            sym: Some(Sym::atom(atom.clone())),
            span,
        };
        let logical = Expr {
            kind: ExprKind::Binary {
                op: crate::ast::BinaryOp::Add,
                lhs: Box::new(index_expr),
                rhs: Box::new(base_expr.clone()),
            },
            ty,
            sym: Some(Sym::atom(atom.clone()).add(&Sym::atom(base_atom))),
            span,
        };
        crate::widen::replace_index(piece_body, index, &atom, &logical);
        block.insert(
            sp.loop_at,
            Stmt {
                id: None,
                span,
                kind: StmtKind::Assign {
                    target: base_expr,
                    op: crate::ast::AssignOp::Assign,
                    value: origin,
                },
            },
        );
        return Ok(sp.loop_at + 1);
    }
    Ok(sp.loop_at)
}

/// Replace the slice bounds on `axis` of every view in an expression, descending tuples.
fn replace_slice(view: &mut Expr, axis: usize, lo: &Expr, hi: &Expr, extent: &Sym) -> bool {
    let changed = match &mut view.kind {
        ExprKind::Tuple(items) => items
            .iter_mut()
            .all(|i| replace_slice(i, axis, lo, hi, extent)),
        ExprKind::Transpose(inner) => replace_slice(inner, 1 - axis, lo, hi, extent),
        ExprKind::Index { base, indices } => {
            let input = input_axis(indices, axis);
            if let Some(slot @ Index::Slice { .. }) = indices.get_mut(input) {
                *slot = Index::Slice {
                    start: Some(lo.clone()),
                    end: Some(hi.clone()),
                };
                true
            } else {
                replace_slice(base, input, lo, hi, extent)
            }
        }
        _ => false,
    };
    if changed {
        if let Ty::Tensor(shape) | Ty::Tile(shape) = &mut view.ty {
            shape.shape[axis] = extent.clone();
        }
    }
    changed
}
