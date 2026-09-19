//! Symbolic streamed-range partitioning. The split count remains the original
//! source parameter; this transformation retains one bounded merge topology
//! instead of materializing a kernel per count.
use super::{range_of, replace_slice, SplitCandidate};
use crate::{ast::AssignOp, ir::*, sym::{Atom, Sym}, types::{DType, Ty}};

fn symbolic(sym: Sym, span: crate::span::Span) -> Expr {
    let kind = sym.as_constant().map(ExprKind::Int)
        .unwrap_or_else(|| ExprKind::ShapeParam("split_geometry".into()));
    Expr { kind, ty: Ty::Scalar(DType::I32), sym: Some(sym), span }
}

/// Narrow a streamed range into a symbolic part slice and return the possibly
/// shifted loop position. All slice bounds and carried-state offsets reference
/// the same original `parts` value.
pub fn narrow_range(
    sp: &SplitCandidate,
    body: &mut [Stmt],
    vars: &mut Vec<Var>,
    part: VarId,
    part_atom: &Atom,
    parts: &Sym,
) -> Result<usize, String> {
    let StmtKind::Parallel { body: block, .. } = &mut body[sp.stmt].kind else {
        return Err("split target is not a parallel block".into());
    };
    let StmtKind::LoadLoop { views, axes, domain, offset, body: piece_body, .. } = &mut block[sp.loop_at].kind else {
        return Err("split target is not a streaming loop".into());
    };
    if views.len() != axes.len() { return Err("stream transfer axes disagree with bindings".into()); }
    let mut origin = None;
    for (view_index, (view, axis)) in std::iter::once((&mut domain.view, domain.axis))
        .chain(views.iter_mut().zip(axes.iter().copied())).enumerate() {
        let (lo, hi) = range_of(view, axis).ok_or("split requires an explicit range for every streamed view")?;
        let span = lo.span;
        let binary = |op, lhs: Expr, rhs: Expr| {
            let sym = lhs.sym.as_ref().zip(rhs.sym.as_ref()).and_then(|(left, right)| match op {
                crate::ast::BinaryOp::Add => Some(left.add(right)),
                crate::ast::BinaryOp::Sub => Some(left.sub(right)),
                crate::ast::BinaryOp::Mul => Some(left.mul(right)),
                crate::ast::BinaryOp::Div => Some(left.quot(right)),
                _ => None,
            });
            Expr { kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }, ty: Ty::Scalar(DType::I32), sym, span }
        };
        use crate::ast::BinaryOp::{Add, Sub, Mul, Div};
        let width = binary(Sub, hi.clone(), lo.clone());
        let chunk = binary(Div, binary(Add, width, symbolic(parts.sub(&Sym::constant(1)), span)), symbolic(parts.clone(), span));
        let part_value = Expr { kind: ExprKind::Var(part), ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(part_atom.clone())), span };
        let raw_start = binary(Add, lo.clone(), binary(Mul, part_value, chunk.clone()));
        let start = Expr { kind: ExprKind::Builtin { name: Builtin::Min, args: vec![hi.clone(), raw_start.clone()] },
            ty: Ty::Scalar(DType::I32), sym: None, span };
        let end = Expr { kind: ExprKind::Builtin { name: Builtin::Min, args: vec![hi.clone(), binary(Add, start.clone(), chunk)] },
            ty: Ty::Scalar(DType::I32), sym: None, span };
        if view_index == 0 && offset.is_some() {
            origin = Some(Expr { kind: ExprKind::Binary { op: crate::ast::BinaryOp::Sub,
                lhs: Box::new(start.clone()), rhs: Box::new(lo.clone()) }, ty: Ty::Scalar(DType::I32), sym: None, span });
        }
        if !replace_slice(view, axis, &start, &end, &Sym::param(&format!("split_extent#{part}_{view_index}"))) { return Err("split could not narrow every streamed view".into()); }
    }
    if let (Some(index), Some(origin)) = (*offset, origin) {
        let VarKind::Index(atom) = &vars[index].kind else { return Err("stream offset is not an index binding".into()); };
        let atom = atom.clone();
        let base = vars.len();
        let base_atom = Atom::Param(format!("split_origin#{base}"));
        let span = origin.span;
        vars.push(Var { name: format!("split_origin_{base}"), ty: Ty::Scalar(DType::I32), span, kind: VarKind::Index(base_atom.clone()) });
        let base_expr = Expr { kind: ExprKind::Var(base), ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(base_atom.clone())), span };
        let index_expr = Expr { kind: ExprKind::Var(index), ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(atom.clone())), span };
        let logical = Expr { kind: ExprKind::Binary { op: crate::ast::BinaryOp::Add, lhs: Box::new(index_expr), rhs: Box::new(base_expr.clone()) },
            ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(atom.clone()).add(&Sym::atom(base_atom))), span };
        crate::widen::replace_index(piece_body, index, &atom, &logical);
        block.insert(sp.loop_at, Stmt { id: None, span, kind: StmtKind::Assign { target: base_expr, op: AssignOp::Assign, value: origin } });
        Ok(sp.loop_at + 1)
    } else { Ok(sp.loop_at) }
}
