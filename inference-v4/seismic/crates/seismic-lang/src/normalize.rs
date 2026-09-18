//! Explicit value bindings for operations that need statement-level realization.
//! Bindings stay in their original control-flow and iteration scope. This pass
//! does not select a reduction algorithm, move work between iterations, or emit code.
use crate::{ast::AssignOp, ir::*};
pub mod loads;

/// Bind load and reduction expressions and return each original statement's new position
/// after its own prerequisite bindings. This preserves references to split loops.
pub fn bind_values(body: &mut Vec<Stmt>, vars: &mut Vec<Var>) -> Vec<usize> {
    let mut positions = Vec::with_capacity(body.len());
    let mut result = Vec::new();
    for mut statement in std::mem::take(body) {
        match &mut statement.kind {
            StmtKind::Assign { target, op, value } => {
                bind_expr(target, vars, &mut result, false);
                let already_bound =
                    *op == AssignOp::Assign && matches!(target.kind, ExprKind::Var(_));
                bind_expr(value, vars, &mut result, already_bound);
            }
            StmtKind::Expr(expr) => bind_expr(expr, vars, &mut result, false),
            StmtKind::If { cond, then, els } => {
                bind_expr(cond, vars, &mut result, false);
                bind_values(then, vars);
                bind_values(els, vars);
            }
            StmtKind::Owned { tile, body, .. } => {
                bind_expr(tile, vars, &mut result, false);
                bind_values(body, vars);
            }
            StmtKind::LoadLoop { views, body, .. } => {
                for view in views {
                    bind_expr(view, vars, &mut result, false);
                }
                bind_values(body, vars);
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } => {
                bind_values(body, vars);
            }
        }
        positions.push(result.len());
        result.push(statement);
    }
    *body = result;
    positions
}

/// Bind loads and reductions needed to evaluate a separately represented execution input,
/// such as a split's original domain validation.
pub fn bind_expression_values(expr: &mut Expr, vars: &mut Vec<Var>) -> Vec<Stmt> {
    let mut bindings = Vec::new();
    bind_expr(expr, vars, &mut bindings, false);
    bindings
}

/// Resolve ordinary and streamed load realization in the operations themselves.
/// The input must have explicit value bindings.
pub fn select_loads(body: &mut [Stmt], borrow_read_only: bool) {
    let modes = loads::sites(body).iter().map(|site| {
        if borrow_read_only && site.can_borrow { LoadMode::Borrow } else { LoadMode::Materialize }
    }).collect::<Vec<_>>();
    loads::resolve(body, &modes).expect("modes are selected from the checked domain");
}

fn bind_expr(expr: &mut Expr, vars: &mut Vec<Var>, bindings: &mut Vec<Stmt>, already_bound: bool) {
    match &mut expr.kind {
        ExprKind::Index { base, indices } => {
            bind_expr(base, vars, bindings, false);
            for index in indices {
                match index {
                    Index::Point(point) => bind_expr(point, vars, bindings, false),
                    Index::Slice { start, end } => {
                        for point in start.iter_mut().chain(end.iter_mut()) {
                            bind_expr(point, vars, bindings, false);
                        }
                    }
                }
            }
        }
        ExprKind::Load { view: base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Accessor { base, .. }
        | ExprKind::Lanes { base, .. }
        | ExprKind::Unary { expr: base, .. }
        | ExprKind::Cast { expr: base, .. } => bind_expr(base, vars, bindings, false),
        ExprKind::Binary { lhs, rhs, .. } => {
            bind_expr(lhs, vars, bindings, false);
            bind_expr(rhs, vars, bindings, false);
        }
        ExprKind::Builtin { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Tuple(args) => {
            for argument in args {
                bind_expr(argument, vars, bindings, false);
            }
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Var(_)
        | ExprKind::ShapeParam(_)
        | ExprKind::TileAlloc { .. } => {}
    }
    if !already_bound
        && matches!(
            expr.kind,
            ExprKind::Builtin {
                name: Builtin::Load | Builtin::Reduce,
                ..
            }
        )
    {
        let id = vars.len();
        vars.push(Var {
            name: if matches!(
                expr.kind,
                ExprKind::Builtin {
                    name: Builtin::Load,
                    ..
                }
            ) {
                "loaded_value"
            } else {
                "reduction_value"
            }
            .into(),
            ty: expr.ty.clone(),
            span: expr.span,
            kind: VarKind::Local,
        });
        let reference = Expr {
            kind: ExprKind::Var(id),
            ty: expr.ty.clone(),
            sym: None,
            span: expr.span,
        };
        let value = std::mem::replace(expr, reference.clone());
        bindings.push(Stmt {
            id: None,
            span: value.span,
            kind: StmtKind::Assign {
                target: reference,
                op: AssignOp::Assign,
                value,
            },
        });
    }
}

/// Represent a serial root as one rank-zero work item for parallel dispatch.
/// Existing outer parallel domains retain their phase boundaries.
pub fn work_domain(body: &mut Vec<Stmt>) {
    if body
        .iter()
        .any(|s| matches!(s.kind, StmtKind::Parallel { .. }))
    {
        return;
    }
    let span = body.first().map(|s| s.span).unwrap_or_default();
    let inner = std::mem::take(body);
    body.push(Stmt {
        id: None,
        kind: StmtKind::Parallel {
            vars: Vec::new(),
            extents: Vec::new(),
            body: inner,
        },
        span,
    });
}

/// Lift pure, loop-invariant reductions out of nonempty owned domains. This
/// exposes full-lane participation before lane ownership introduces tail masks.
/// Return old statement positions so execution-domain references remain valid.
pub fn lift_owned_reductions(body: &mut Vec<Stmt>) -> Vec<usize> {
    fn definitions(body: &[Stmt], ids: &mut std::collections::HashSet<VarId>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign {
                    target:
                        Expr {
                            kind: ExprKind::Var(v),
                            ..
                        },
                    ..
                } => {
                    ids.insert(*v);
                }
                StmtKind::Owned { vars, body, .. }
                | StmtKind::Parallel { vars, body, .. }
                | StmtKind::LoadLoop { vars, body, .. } => {
                    ids.extend(vars);
                    definitions(body, ids);
                }
                StmtKind::Range { var, body, .. } | StmtKind::Lanes { var, body, .. } => {
                    ids.insert(*var);
                    definitions(body, ids);
                }
                StmtKind::If { then, els, .. } => {
                    definitions(then, ids);
                    definitions(els, ids);
                }
                _ => {}
            }
        }
    }
    let mut result = Vec::new();
    let mut positions = Vec::new();
    for mut statement in std::mem::take(body) {
        match &mut statement.kind {
            StmtKind::Owned { tile, body, .. } => {
                lift_owned_reductions(body);
                let nonempty = tile.ty.shaped().is_some_and(|s| {
                    s.shape
                        .iter()
                        .all(|d| d.as_constant().is_some_and(|n| n > 0))
                });
                if nonempty
                    && !body
                        .iter()
                        .any(|s| crate::effects::tensor_effect(s))
                {
                    let mut defined = std::collections::HashSet::new();
                    definitions(body, &mut defined);
                    loop {
                        let Some(Stmt {
                            kind:
                                StmtKind::Assign {
                                    target:
                                        Expr {
                                            kind: ExprKind::Var(output),
                                            ..
                                        },
                                    op: AssignOp::Assign,
                                    value:
                                        Expr {
                                            kind:
                                                ExprKind::Builtin {
                                                    name: Builtin::Reduce,
                                                    args,
                                                },
                                            ..
                                        },
                                },
                            ..
                        }) = body.first()
                        else {
                            break;
                        };
                        let Some(Expr {
                            kind: ExprKind::Var(input),
                            ..
                        }) = args.first()
                        else {
                            break;
                        };
                        if defined.contains(input)
                            || args[1..]
                                .iter()
                                .any(|e| !matches!(e.kind, ExprKind::Int(_) | ExprKind::Bool(_)))
                            || body[1..].iter().any(|s| {
                                crate::effects::tile_mutated(s, *input)
                                    || crate::effects::tile_mutated(s, *output)
                            })
                        {
                            break;
                        }
                        defined.remove(output);
                        result.push(body.remove(0));
                    }
                }
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::LoadLoop { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } => {
                lift_owned_reductions(body);
            }
            StmtKind::If { then, els, .. } => {
                lift_owned_reductions(then);
                lift_owned_reductions(els);
            }
            _ => {}
        }
        positions.push(result.len());
        result.push(statement);
    }
    *body = result;
    positions
}

/// Assign unique identities after structural normalization. A shared counter
/// covers separately stored phase inputs as well as the main body.
pub fn identify(body: &mut [Stmt], next: &mut usize) {
    for statement in body {
        statement.id = Some(OperationId(*next));
        *next += 1;
        match &mut statement.kind {
            StmtKind::Parallel { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::LoadLoop { body, .. }
            | StmtKind::Lanes { body, .. } => identify(body, next),
            StmtKind::If { then, els, .. } => {
                identify(then, next);
                identify(els, next);
            }
            _ => {}
        }
    }
}

/// Remove statically empty range bodies before execution planning. Return each
/// original statement's new position; retained domain references can be remapped.
pub fn remove_empty_ranges(body: &mut Vec<Stmt>) -> Vec<usize> {
    let mut positions = Vec::with_capacity(body.len());
    let mut result = Vec::new();
    for mut statement in std::mem::take(body) {
        positions.push(result.len());
        if matches!(&statement.kind, StmtKind::Range { lo, hi, .. }
            if matches!((lo.as_constant(), hi.as_constant()), (Some(lo),Some(hi)) if lo >= hi))
        {
            continue;
        }
        match &mut statement.kind {
            StmtKind::Parallel { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::LoadLoop { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } => {
                remove_empty_ranges(body);
            }
            StmtKind::If { then, els, .. } => {
                remove_empty_ranges(then);
                remove_empty_ranges(els);
            }
            _ => {}
        }
        result.push(statement);
    }
    *body = result;
    positions
}
