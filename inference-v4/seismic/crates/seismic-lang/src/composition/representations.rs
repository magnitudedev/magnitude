//! Storage choices for actual packed load values after logical decomposition.
//! A decoded cache is an ordinary F32 producer; packet access retains the original
//! binding. Both storage and decode arithmetic therefore survive into emission.
use super::*;

type Values = HashMap<VarId, Expr>;

pub(super) fn select(
    function: &mut LoweredIr,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    block(
        &mut function.body,
        &mut function.vars,
        &Values::new(),
        select,
    )
}

fn block(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    inherited: &Values,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let mut values = inherited.clone();
    let mut result = Vec::new();
    for mut statement in std::mem::take(body) {
        let binding = match &statement.kind {
            StmtKind::Assign {
                target:
                    Expr {
                        kind: ExprKind::Var(id),
                        ty: Ty::Tile(shape),
                        ..
                    },
                op: AssignOp::Assign,
                value,
            } if matches!(shape.elem, Elem::Repr(_)) => Some((*id, value.clone())),
            _ => None,
        };
        // A tile alias captures a value at this point, including its original
        // logical offsets. Its hidden decoded producer is immutable thereafter.
        let alias = binding
            .as_ref()
            .and_then(|(_, value)| logical_view(value, &values));
        match &mut statement.kind {
            StmtKind::Assign { target, value, .. } => {
                read(value, &values);
                // Assignment addresses retain their original physical owner.
                if let ExprKind::Index { indices, .. } = &mut target.kind {
                    for index in indices {
                        match index {
                            Index::Point(e) => read(e, &values),
                            Index::Slice { start, end } => {
                                for e in start.iter_mut().chain(end) {
                                    read(e, &values);
                                }
                            }
                        }
                    }
                }
            }
            StmtKind::Expr(e) => read(e, &values),
            StmtKind::Reduction(r) => {
                for input in &mut r.inputs {
                    // Reducing away the packed axis produces a logical dense
                    // leaf. Other axes may expose packets to the step helper.
                    if input
                        .ty
                        .shaped()
                        .is_some_and(|s| s.packed_axis == Some(r.axis))
                    {
                        if let Some(decoded) = logical_view(input, &values) {
                            *input = decoded;
                        }
                    }
                    read(input, &values);
                }
                for step in r.step.iter_mut() {
                    for e in &mut step.identity {
                        read(e, &values);
                    }
                }
            }
            StmtKind::If { cond, .. } => read(cond, &values),
            StmtKind::LoadLoop { domain, views, .. } => {
                read(&mut domain.view, &values);
                for e in views {
                    read(e, &values);
                }
            }
            _ => {}
        }
        let mut nested = values.clone();
        if matches!(
            statement.kind,
            StmtKind::Range { .. }
                | StmtKind::Parallel { .. }
                | StmtKind::Owned { .. }
                | StmtKind::Lanes { .. }
                | StmtKind::LoadLoop { .. }
        ) {
            // A loop body cannot reuse a pre-loop cache for a value carried and
            // mutated by another iteration.
            nested.retain(|&id, _| !crate::effects::tile_mutated(&statement, id));
        }
        if let StmtKind::LoadLoop {
            vars: bindings,
            body,
            ..
        } = &mut statement.kind
        {
            // These are actual load owners as well: each invocation creates its
            // own bounded snapshot, so decoding belongs inside the same loop.
            let mut prefix = Vec::new();
            for &variable in bindings.iter() {
                if matches!(
                    vars[variable].ty.shaped().map(|s| &s.elem),
                    Some(Elem::Repr(_))
                ) {
                    if let Some((cache, producer)) = choose(variable, vars, select)? {
                        prefix.extend(producer);
                        nested.insert(variable, cache);
                    }
                }
            }
            block(body, vars, &nested, select)?;
            prefix.append(body);
            *body = prefix;
        } else {
            let mut error = None;
            nested_mut(&mut statement, &mut |body| {
                if error.is_none() {
                    error = block(body, vars, &nested, select).err();
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
        }
        values.retain(|&id, _| !crate::effects::tile_mutated(&statement, id));
        if let Some((id, _)) = &binding {
            if let Some(alias) = alias {
                values.insert(*id, alias);
            }
        }
        let packed_load = binding.as_ref().filter(|(_, value)| {
            matches!(
                value.kind,
                ExprKind::Builtin {
                    name: Builtin::Load,
                    ..
                } | ExprKind::Load { .. }
            )
        });
        result.push(statement);
        if let Some((variable, _)) = packed_load {
            if let Some((cache, producer)) = choose(*variable, vars, select)? {
                result.extend(producer);
                values.insert(*variable, cache);
            }
        }
    }
    *body = result;
    Ok(())
}
fn choose(
    variable: VarId,
    vars: &mut Vec<Var>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<Option<(Expr, Vec<Stmt>)>, String> {
    let decision = Decision {
        kind: DecisionKind::Representation { variable },
        alternatives: vec![Alternative::Encoded, Alternative::Decoded].into(),
    };
    match select(&decision)? {
        Alternative::Encoded => Ok(None),
        Alternative::Decoded => decode(&super::variable(variable, vars), vars).map(Some),
        _ => Err("invalid packed value storage representation".into()),
    }
}

fn dense_type(ty: &Ty) -> Ty {
    let mut ty = ty.clone();
    if let Ty::Tile(shape) = &mut ty {
        shape.elem = Elem::Dtype(DType::F32);
        shape.packed_axis = None;
    }
    ty
}
fn logical_view(e: &Expr, values: &Values) -> Option<Expr> {
    let mut result = e.clone();
    result.kind = match &e.kind {
        ExprKind::Var(id) => return values.get(id).cloned(),
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(logical_view(base, values)?),
            indices: indices.clone(),
        },
        ExprKind::Transpose(base) => ExprKind::Transpose(Box::new(logical_view(base, values)?)),
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => {
            let mut args = args.clone();
            args[0] = logical_view(&args[0], values)?;
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            }
        }
        _ => return None,
    };
    result.ty = dense_type(&result.ty);
    Some(result)
}
fn read(e: &mut Expr, values: &Values) {
    if let ExprKind::Index { base, .. } = &e.kind {
        if matches!(e.ty, Ty::Scalar(DType::F32))
            && matches!(base.ty.shaped().map(|s| &s.elem), Some(Elem::Repr(_)))
        {
            if let Some(decoded) = logical_view(e, values) {
                *e = decoded;
            }
        }
    }
    if let ExprKind::Builtin {
        name: Builtin::Store,
        args,
    } = &mut e.kind
    {
        if let Some(decoded) = logical_view(&args[0], values) {
            args[0] = decoded;
        }
    }
    // Accessors expose packets of the original snapshot, not values of the
    // logical decoded cache. Their indices outside this node are still visited.
    if matches!(e.kind, ExprKind::Accessor { .. }) {
        return;
    }
    children_mut(e, &mut |child| read(child, values));
}
fn decode(source: &Expr, vars: &mut Vec<Var>) -> Result<(Expr, Vec<Stmt>), String> {
    let span = source.span;
    let Ty::Tile(shape) = dense_type(&source.ty) else {
        return Err("decoded cache requires a packed tile load owner".into());
    };
    let id = vars.len();
    vars.push(Var {
        name: format!("decoded_{id}"),
        ty: Ty::Tile(shape.clone()),
        kind: VarKind::Local,
        span,
    });
    let cache = super::variable(id, vars);
    let mut indices = Vec::new();
    for _ in &shape.shape {
        let id = vars.len();
        vars.push(Var {
            name: format!("decode_index_{id}"),
            ty: Ty::Scalar(DType::I32),
            kind: VarKind::Index(Atom::Param(format!("$decode_{id}"))),
            span,
        });
        indices.push(id);
    }
    let coordinates = indices
        .iter()
        .map(|&id| Index::Point(super::variable(id, vars)))
        .collect::<Vec<_>>();
    let element = |base: Expr| Expr {
        kind: ExprKind::Index {
            base: Box::new(base),
            indices: coordinates.clone(),
        },
        ty: Ty::Scalar(DType::F32),
        sym: None,
        span,
    };
    let assignment = Stmt {
        id: None,
        span,
        kind: StmtKind::Assign {
            target: element(cache.clone()),
            op: AssignOp::Assign,
            value: element(source.clone()),
        },
    };
    Ok((
        cache.clone(),
        vec![
            Stmt {
                id: None,
                span,
                kind: StmtKind::Assign {
                    target: cache.clone(),
                    op: AssignOp::Assign,
                    value: Expr {
                        kind: ExprKind::TileAlloc {
                            shape: shape.shape,
                            dtype: Elem::Dtype(DType::F32),
                        },
                        ty: cache.ty.clone(),
                        sym: None,
                        span,
                    },
                },
            },
            Stmt {
                id: None,
                span,
                kind: StmtKind::Owned {
                    vars: indices,
                    tile: cache,
                    body: vec![assignment],
                },
            },
        ],
    ))
}
