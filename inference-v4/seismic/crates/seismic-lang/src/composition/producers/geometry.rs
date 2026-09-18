//! Capture a computed view's runtime coordinates as ordinary scalar IR values.
//! The retained view still owns bounds/layout checks and dynamic extent binding.
use super::*;

pub(super) fn needs_capture(view: &Expr) -> bool {
    match &view.kind {
        ExprKind::Index { base, indices } => {
            needs_capture(base)
                || indices.iter().any(|i| match i {
                    Index::Point(p) => !crate::effects::can_substitute_symbolic_value(p),
                    Index::Slice { start, end } => start
                        .iter()
                        .chain(end)
                        .any(|p| !crate::effects::can_substitute_symbolic_value(p)),
                })
        }
        ExprKind::Transpose(base) => needs_capture(base),
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => needs_capture(&args[0]),
        _ => false,
    }
}

pub(super) fn capture(view: &Expr, vars: &mut Vec<Var>) -> Option<(Expr, Vec<Stmt>)> {
    let mut statements = Vec::new();
    let (view, _) = capture_view(view, vars, &mut statements)?;
    Some((view, statements))
}

fn bind(value: Expr, vars: &mut Vec<Var>, body: &mut Vec<Stmt>) -> Option<Expr> {
    // Slice endpoints have this checked type. Wider unsigned point coordinates
    // need their own typed capture before this projection family can admit them.
    if value.ty != Ty::Scalar(DType::I32) {
        return None;
    }
    let id = vars.len();
    vars.push(Var {
        name: format!("view_coordinate_{id}"),
        ty: value.ty.clone(),
        span: value.span,
        kind: VarKind::Index(Atom::Param(format!("view_coordinate#{id}"))),
    });
    let target = variable(id, vars);
    body.push(Stmt {
        id: None,
        span: value.span,
        kind: StmtKind::Assign {
            target: target.clone(),
            op: AssignOp::Assign,
            value,
        },
    });
    Some(target)
}

fn extent(view: &Expr, axis: usize) -> Expr {
    Expr {
        kind: ExprKind::Builtin {
            name: Builtin::Extent,
            args: vec![view.clone(), symbol(Sym::constant(axis as i64), view.span)],
        },
        ty: Ty::Scalar(DType::I32),
        sym: Some(view.ty.shaped().unwrap().shape[axis].clone()),
        span: view.span,
    }
}
fn limit(name: Builtin, a: Expr, b: Expr) -> Expr {
    Expr {
        span: a.span,
        kind: ExprKind::Builtin {
            name,
            args: vec![a, b],
        },
        ty: Ty::Scalar(DType::I32),
        sym: None,
    }
}

fn capture_view(
    view: &Expr,
    vars: &mut Vec<Var>,
    body: &mut Vec<Stmt>,
) -> Option<(Expr, Vec<i64>)> {
    let mut captured = view.clone();
    let capacities = match &view.kind {
        ExprKind::Var(_) => view
            .ty
            .shaped()?
            .shape
            .iter()
            .map(|n| {
                n.as_constant()
                    .filter(|n| (0..=i64::from(i32::MAX)).contains(n))
            })
            .collect::<Option<Vec<_>>>()?,
        ExprKind::Transpose(base) => {
            let (base, mut capacities) = capture_view(base, vars, body)?;
            captured.kind = ExprKind::Transpose(Box::new(base));
            capacities.reverse();
            capacities
        }
        ExprKind::Index { base, indices } => {
            let (base, parent) = capture_view(base, vars, body)?;
            let mut normalized = Vec::new();
            let mut capacities = Vec::new();
            for (axis, capacity) in parent.iter().copied().enumerate() {
                match indices.get(axis) {
                    Some(Index::Point(point)) => {
                        let point = if crate::effects::can_substitute_symbolic_value(point) {
                            point.clone()
                        } else {
                            bind(point.clone(), vars, body)?
                        };
                        normalized.push(Index::Point(point));
                    }
                    Some(Index::Slice { start, end }) => {
                        let dynamic = start.iter().chain(end).any(|p| p.sym.is_none());
                        // Evaluate start before end, before clamping either one.
                        // Empty and reversed windows retain both evaluations.
                        let mut capture_bound = |p: &Expr| {
                            if crate::effects::can_substitute_symbolic_value(p) {
                                Some(p.clone())
                            } else {
                                bind(p.clone(), vars, body)
                            }
                        };
                        let start = match start {
                            Some(p) => Some(capture_bound(p)?),
                            None => None,
                        };
                        let end = match end {
                            Some(p) => Some(capture_bound(p)?),
                            None => None,
                        };
                        let (start, end) = if dynamic {
                            let zero = || symbol(Sym::constant(0), view.span);
                            let end = bind(
                                limit(
                                    Builtin::Min,
                                    limit(
                                        Builtin::Max,
                                        end.unwrap_or_else(|| extent(&base, axis)),
                                        zero(),
                                    ),
                                    extent(&base, axis),
                                ),
                                vars,
                                body,
                            )?;
                            let start = bind(
                                limit(
                                    Builtin::Min,
                                    limit(Builtin::Max, start.unwrap_or_else(zero), zero()),
                                    end.clone(),
                                ),
                                vars,
                                body,
                            )?;
                            (Some(start), Some(end))
                        } else {
                            (start, end)
                        };
                        normalized.push(Index::Slice { start, end });
                        capacities.push(capacity);
                    }
                    None => {
                        normalized.push(Index::Slice {
                            start: None,
                            end: None,
                        });
                        capacities.push(capacity);
                    }
                }
                // Evaluate this axis before reading the next axis's endpoints.
                // In particular a failed point guard cannot be postponed until
                // after a later endpoint load or integer operation.
                let source_shape = &base.ty.shaped()?.shape;
                let result = view.ty.shaped()?;
                let mut shape = result.shape[..capacities.len()].to_vec();
                shape.extend_from_slice(&source_shape[axis + 1..]);
                if shape.is_empty() {
                    return None;
                }
                let partial = Expr {
                    kind: ExprKind::Index {
                        base: Box::new(base.clone()),
                        indices: normalized.clone(),
                    },
                    ty: Ty::Tile(Shaped::new(shape, result.elem.clone())),
                    sym: None,
                    span: view.span,
                };
                body.push(Stmt {
                    id: None,
                    span: view.span,
                    kind: StmtKind::Expr(extent(&partial, 0)),
                });
            }
            captured.kind = ExprKind::Index {
                base: Box::new(base),
                indices: normalized,
            };
            if capacities.is_empty() {
                return None;
            }
            capacities
        }
        // Reshape validity and snapshot layout must be retained before admitting
        // dynamic rectangular producer maps through a reshape.
        _ => return None,
    };
    Some((captured, capacities))
}
