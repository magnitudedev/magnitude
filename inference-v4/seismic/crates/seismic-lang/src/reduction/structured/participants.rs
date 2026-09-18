//! A selected contiguous fold tree mapped to a complete subgroup. This refines
//! the existing source operation; step and merge arithmetic remain ordinary IR.
use super::*;
use crate::{ast::BinaryOp, intrinsics::Operation, lowered_ir::LoweredIr, types::Elem};
use std::collections::HashSet;

/// Lexical reduction sites that this finite participant form can represent.
/// The seed occupies lane zero, followed by one contiguous segment per lane.
pub fn candidates(function: &LoweredIr, lanes: u32) -> Vec<usize> {
    let mut sites = Vec::new();
    let mut site = 0;
    fn visit(
        body: &[Stmt],
        site: &mut usize,
        sites: &mut Vec<usize>,
        lanes: u32,
        independent: bool,
    ) {
        for s in body {
            match &s.kind {
                StmtKind::Reduction(r) => {
                    if independent && applicable(r, lanes) {
                        sites.push(*site);
                    }
                    *site += 1;
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. } => visit(body, site, sites, lanes, independent),
                StmtKind::Lanes { body, .. } => visit(body, site, sites, lanes, false),
                StmtKind::If { then, els, .. } => {
                    visit(then, site, sites, lanes, independent);
                    visit(els, site, sites, lanes, independent);
                }
                _ => {}
            }
        }
    }
    visit(&function.body, &mut site, &mut sites, lanes, true);
    sites
}
fn local_expr(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Intrinsic { .. } | ExprKind::Lanes { .. } | ExprKind::Call { .. } => false,
        ExprKind::Load { view, .. }
        | ExprKind::Transpose(view)
        | ExprKind::Accessor { base: view, .. }
        | ExprKind::Unary { expr: view, .. }
        | ExprKind::Cast { expr: view, .. } => local_expr(view),
        ExprKind::Index { base, indices } => {
            local_expr(base)
                && indices.iter().all(|i| match i {
                    Index::Point(e) => local_expr(e),
                    Index::Slice { start, end } => start.iter().chain(end).all(local_expr),
                })
        }
        ExprKind::Binary { lhs, rhs, .. } => local_expr(lhs) && local_expr(rhs),
        ExprKind::Builtin { args, .. } | ExprKind::Tuple(args) => args.iter().all(local_expr),
        _ => true,
    }
}
fn local_body(body: &[Stmt]) -> bool {
    body.iter().all(|s| match &s.kind {
        StmtKind::Lanes { .. } | StmtKind::Parallel { .. } | StmtKind::LoadLoop { .. } => false,
        StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } => local_body(body),
        StmtKind::Reduction(r) => r.operands().all(local_expr) && r.bodies().all(local_body),
        StmtKind::If { cond, then, els } => local_expr(cond) && local_body(then) && local_body(els),
        StmtKind::Assign { target, value, .. } => local_expr(target) && local_expr(value),
        StmtKind::Expr(e) => local_expr(e),
    })
}
fn applicable(r: &Reduction, lanes: u32) -> bool {
    if lanes < 2
        || !lanes.is_power_of_two()
        || r.ordered
        || r.step.is_none()
        || r.implementation.is_none()
        || !matches!(r.tree, Some(Tree::Pairwise | Tree::Explicit))
    {
        return false;
    }
    let Some(n) = r.extent().as_constant() else {
        return false;
    };
    let Some(segment) = r.segment.filter(|&s| s > 0 && s <= n) else {
        return false;
    };
    let count = n / segment + i64::from(n % segment != 0);
    count < i64::from(lanes)
        && r.bodies().all(local_body)
        && r.state.iter().all(|s| {
            s.ty.shaped()
                .is_some_and(|s| s.elem == Elem::Dtype(DType::F32))
        })
}

/// Selected site IDs come from `candidates` on this exact retained computation.
/// Unselected reductions retain their ordinary single-participant realization.
#[derive(Clone, Debug)]
pub struct Refinement {
    pub function: LoweredIr,
    /// Tile values with distinct state in each participant. Placement must
    /// preserve this ownership; cooperation on one shared value is not valid.
    pub private_values: Vec<usize>,
}
pub fn apply(function: &LoweredIr, selected: &[usize], lanes: u32) -> Result<Refinement, String> {
    let admitted = candidates(function, lanes);
    let chosen: HashSet<_> = selected.iter().copied().collect();
    if chosen.len() != selected.len() || chosen.iter().any(|i| !admitted.contains(i)) {
        return Err("participant fold site is outside the selected reduction family".into());
    }
    let mut result = function.clone();
    let mut site = 0;
    let mut private_values = HashSet::new();
    fn block(
        body: &mut Vec<Stmt>,
        vars: &mut Vec<Var>,
        site: &mut usize,
        chosen: &HashSet<usize>,
        lanes: u32,
        private_values: &mut HashSet<usize>,
    ) -> Result<(), String> {
        let mut output = Vec::new();
        for mut stmt in std::mem::take(body) {
            match &mut stmt.kind {
                StmtKind::Reduction(r) => {
                    let id = *site;
                    *site += 1;
                    if chosen.contains(&id) {
                        let generated = expand(r, vars, lanes)?;
                        private_tiles(&generated, private_values);
                        output.extend(generated);
                        continue;
                    }
                }
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => {
                    block(body, vars, site, chosen, lanes, private_values)?
                }
                StmtKind::If { then, els, .. } => {
                    block(then, vars, site, chosen, lanes, private_values)?;
                    block(els, vars, site, chosen, lanes, private_values)?;
                }
                _ => {}
            }
            output.push(stmt);
        }
        *body = output;
        Ok(())
    }
    block(
        &mut result.body,
        &mut result.vars,
        &mut site,
        &chosen,
        lanes,
        &mut private_values,
    )?;
    let mut private_values: Vec<_> = private_values.into_iter().collect();
    private_values.sort_unstable();
    Ok(Refinement {
        function: result,
        private_values,
    })
}
fn private_tiles(body: &[Stmt], values: &mut HashSet<usize>) {
    for s in body {
        match &s.kind {
            StmtKind::Assign { target, .. } => {
                let mut root = target;
                while let ExprKind::Index { base, .. }
                | ExprKind::Transpose(base)
                | ExprKind::Accessor { base, .. } = &root.kind
                {
                    root = base;
                }
                if let ExprKind::Var(id) = root.kind {
                    if matches!(root.ty, Ty::Tile(_)) {
                        values.insert(id);
                    }
                }
            }
            StmtKind::Owned { body, .. }
            | StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => private_tiles(body, values),
            StmtKind::If { then, els, .. } => {
                private_tiles(then, values);
                private_tiles(els, values);
            }
            _ => {}
        }
    }
}
fn intrinsic(operation: Operation, args: Vec<Expr>, dtype: DType, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Intrinsic {
            op: operation,
            args,
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    }
}
fn binary(op: BinaryOp, lhs: Expr, rhs: Expr, span: Span) -> Expr {
    let comparison = matches!(
        op,
        BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::And
    );
    Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ty: Ty::Scalar(if comparison { DType::Bool } else { DType::I32 }),
        sym: None,
        span,
    }
}
fn assign(target: Expr, value: Expr, span: Span) -> Stmt {
    stmt(
        StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        },
        span,
    )
}
fn shuffle_copy(b: &mut Builder<'_>, target: &Expr, value: &Expr, lane: &Expr) -> Stmt {
    let mut copy = b.copy(target, value);
    let StmtKind::Owned { body, .. } = &mut copy.kind else {
        unreachable!()
    };
    let StmtKind::Assign { value, .. } = &mut body[0].kind else {
        unreachable!()
    };
    *value = intrinsic(
        Operation::ShuffleIndex,
        vec![value.clone(), lane.clone()],
        DType::F32,
        b.span,
    );
    copy
}
fn expand(r: &Reduction, vars: &mut Vec<Var>, lanes: u32) -> Result<Vec<Stmt>, String> {
    if !applicable(r, lanes) {
        return Err("unsupported selected participant fold".into());
    }
    let parts = r.segments(vars)?;
    let span = r.span;
    let mut b = Builder { vars, span };
    let leaf_ids: HashSet<_> = parts
        .leaves
        .iter()
        .filter_map(|e| {
            if let ExprKind::Var(id) = e.kind {
                Some(id)
            } else {
                None
            }
        })
        .collect();
    // Distributed leaves stay in the participant's partial state. No logical
    // leaf array is physically allocated by this realization.
    let mut body:Vec<_>=parts.setup.into_iter().filter(|s|!matches!(&s.kind,StmtKind::Assign{target:Expr{kind:ExprKind::Var(id),..},..} if leaf_ids.contains(id))).collect();
    let lane = b.local(Ty::Scalar(DType::I32));
    body.push(assign(
        lane.clone(),
        intrinsic(Operation::LaneIndex, vec![], DType::I32, span),
        span,
    ));
    for (dst, identity) in parts.partial.iter().zip(&r.step.as_ref().unwrap().identity) {
        body.push(b.copy(dst, identity));
    }
    body.push(assign(
        parts.index.clone(),
        binary(BinaryOp::Sub, lane.clone(), integer(1, span), span),
        span,
    ));
    let valid = binary(
        BinaryOp::And,
        binary(BinaryOp::Gt, lane.clone(), integer(0, span), span),
        binary(BinaryOp::Le, lane.clone(), integer(parts.count, span), span),
        span,
    );
    body.push(stmt(
        StmtKind::If {
            cond: valid,
            then: parts.body,
            els: vec![],
        },
        span,
    ));
    let seed = parts
        .partial
        .iter()
        .zip(&r.state)
        .map(|(dst, src)| b.copy(dst, src))
        .collect();
    body.push(stmt(
        StmtKind::If {
            cond: binary(BinaryOp::Eq, lane.clone(), integer(0, span), span),
            then: seed,
            els: vec![],
        },
        span,
    ));
    let merge = r.implementation.as_ref().unwrap();
    for p in merge.left.iter().chain(&merge.right).chain(&merge.output) {
        body.push(b.allocate(p));
    }
    let combine = |b: &mut Builder<'_>, source: Expr, active: Expr, body: &mut Vec<Stmt>| {
        for ((left, right), value) in merge.left.iter().zip(&merge.right).zip(&parts.partial) {
            body.push(b.copy(left, value));
            body.push(shuffle_copy(b, right, value, &source));
        }
        let mut arithmetic = merge.body.clone();
        for (dst, src) in parts.partial.iter().zip(&merge.output) {
            arithmetic.push(b.copy(dst, src));
        }
        body.push(stmt(
            StmtKind::If {
                cond: active,
                then: arithmetic,
                els: vec![],
            },
            span,
        ));
    };
    match r.tree.unwrap() {
        Tree::Pairwise => {
            let mut stride = 1;
            while stride <= parts.count {
                let neighbor = binary(BinaryOp::Add, lane.clone(), integer(stride, span), span);
                let in_range = binary(
                    BinaryOp::Le,
                    neighbor.clone(),
                    integer(parts.count, span),
                    span,
                );
                let source = b.local(Ty::Scalar(DType::I32));
                body.push(assign(source.clone(), lane.clone(), span));
                body.push(stmt(
                    StmtKind::If {
                        cond: in_range.clone(),
                        then: vec![assign(source.clone(), neighbor, span)],
                        els: vec![],
                    },
                    span,
                ));
                let leader = binary(
                    BinaryOp::Eq,
                    binary(BinaryOp::Rem, lane.clone(), integer(stride * 2, span), span),
                    integer(0, span),
                    span,
                );
                combine(
                    &mut b,
                    source,
                    binary(BinaryOp::And, leader, in_range, span),
                    &mut body,
                );
                stride *= 2;
            }
        }
        Tree::Explicit => {
            // Parent-first branch storage is traversed in reverse: all children
            // are ready before their parent combines adjacent ranges.
            for branch in r.branches.iter().rev() {
                combine(
                    &mut b,
                    integer(branch.cut, span),
                    binary(
                        BinaryOp::Eq,
                        lane.clone(),
                        integer(branch.start, span),
                        span,
                    ),
                    &mut body,
                );
            }
        }
        Tree::Ordered => unreachable!(),
    }
    for (dst, value) in r.state.iter().zip(&parts.partial) {
        body.push(shuffle_copy(&mut b, dst, value, &integer(0, span)));
    }
    Ok(body)
}
