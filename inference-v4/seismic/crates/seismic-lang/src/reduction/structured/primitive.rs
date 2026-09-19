//! Standard reductions retain their numerical contract while joining the same
//! state/step/merge representation as source-defined coupled reductions.
use super::*;
use crate::{
    ast::BinaryOp,
    ir::ReduceOp,
    lowered_ir::{Alternative, Decision, DecisionKind},
    reduction::{Combination, Contract},
    types::Elem,
};

pub fn contract(e: &Expr) -> Option<Contract> {
    let ExprKind::Builtin {
        name: Builtin::Reduce,
        args,
    } = &e.kind
    else {
        return None;
    };
    let [input, _, operation, ..] = args.as_slice() else {
        return None;
    };
    let ExprKind::Int(operation) = operation.kind else {
        return None;
    };
    Some(Contract::new(
        ReduceOp::from_tag(operation)?,
        input.ty.shaped()?.elem.read_dtype()?,
        matches!(args.get(3).map(|e| &e.kind), Some(ExprKind::Bool(true))),
    ))
}

/// The ordinary source operation is retained as `merge` metadata; both helper
/// bodies are mechanical expansions of its existing numerical contract.
pub fn retain(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    select: &mut dyn FnMut(
        &crate::lowered_ir::Decision,
    ) -> Result<crate::lowered_ir::Alternative, String>,
) -> Result<(), String> {
    let mut result = Vec::new();
    for mut s in std::mem::take(body) {
        match &mut s.kind {
            StmtKind::Parallel { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => retain(body, vars, select)?,
            StmtKind::If { then, els, .. } => {
                retain(then, vars, select)?;
                retain(els, vars, select)?;
            }
            StmtKind::Reduction(r) => {
                for m in r.implementations_mut() {
                    retain(&mut m.body, vars, select)?;
                }
            }
            _ => {}
        }
        if let StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        } = &s.kind
        {
            if let Some(c) = contract(value) {
                let (mut prefix, mut reduction, suffix) = expand(value, target, c, vars)?;
                let d = Decision {
                    kind: DecisionKind::Reduction {
                        merge: reduction.merge_name().into(),
                        extent: reduction.extent().clone(),
                        fields: reduction.state.iter().map(|s| s.ty.clone()).collect(),
                    },
                    alternatives: reduction
                        .trees()
                        .into_iter()
                        .map(Alternative::ReductionTree)
                        .collect::<Vec<_>>()
                        .into(),
                };
                let Alternative::ReductionTree(tree) = select(&d)? else {
                    return Err("invalid reduction implementation".into());
                };
                reduction.select_tree(tree, select)?;
                result.append(&mut prefix);
                result.push(stmt(StmtKind::Reduction(Box::new(reduction)), s.span));
                result.extend(suffix);
                continue;
            }
        }
        result.push(s);
    }
    *body = result;
    Ok(())
}
pub(crate) fn expand(
    source: &Expr,
    target: &Expr,
    c: Contract,
    vars: &mut Vec<Var>,
) -> Result<(Vec<Stmt>, Reduction, Vec<Stmt>), String> {
    let ExprKind::Builtin { args, .. } = &source.kind else {
        unreachable!()
    };
    let axis = args[1]
        .sym
        .as_ref()
        .and_then(Sym::as_constant)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or("invalid reduction axis")?;
    let input = args[0].clone();
    let mut shape = input
        .ty
        .shaped()
        .ok_or("reduction input must be shaped")?
        .clone();
    if axis >= shape.shape.len() {
        return Err("reduction axis outside input".into());
    }
    let extent = shape.shape.remove(axis);
    if !c.allows_empty_axis() && extent.as_constant() == Some(0) {
        return Err("argmax requires a nonempty axis".into());
    }
    shape.elem = Elem::Dtype(c.input);
    shape.packed_axis = None;
    let ty = Ty::Tile(shape.clone());
    let mut b = Builder {
        vars,
        span: source.span,
    };
    let mut prefix = Vec::new();
    let state = b.alloc(&ty, &mut prefix);
    prefix.push(fill(&mut b, &state, |_| {
        literal(c.input, c.identity().value(), source.span)
    }));
    let identity = b.alloc(&ty, &mut prefix);
    prefix.push(b.copy(&identity, &state));
    let mut inputs = vec![input.clone()];
    let mut states = vec![state];
    let mut identities = vec![identity];
    if c.operation == ReduceOp::Argmax {
        shape.elem = Elem::Dtype(DType::I32);
        let index = b.alloc(&Ty::Tile(shape.clone()), &mut prefix);
        prefix.push(fill(&mut b, &index, |_| integer(0, source.span)));
        let zero = b.alloc(&Ty::Tile(shape), &mut prefix);
        prefix.push(b.copy(&zero, &index));
        states.push(index);
        identities.push(zero);
        let mut input_shape = input.ty.shaped().unwrap().clone();
        input_shape.elem = Elem::Dtype(DType::I32);
        input_shape.packed_axis = None;
        let indexes = b.alloc(&Ty::Tile(input_shape), &mut prefix);
        prefix.push(fill(&mut b, &indexes, |coordinates| {
            coordinates[axis].clone()
        }));
        inputs.push(indexes);
    }
    let merge = implementation(&mut b, &states, c);
    let step = implementation(&mut b, &states, c);
    let state_result = states.last().unwrap().clone();
    let value = if matches!(target.ty, Ty::Scalar(_)) {
        point(&state_result, &[], source.span)
    } else {
        state_result
    };
    let suffix = vec![stmt(
        StmtKind::Assign {
            target: target.clone(),
            op: AssignOp::Assign,
            value,
        },
        source.span,
    )];
    let reduction = Reduction {
        preparation: vec![InputPreparation::Direct; inputs.len()],
        unroll: 1,
        preparation_window: None,
        inputs,
        state: states,
        axis,
        merge: Callback::Source(source.clone()),
        ordered: c.ordered,
        span: source.span,
        implementation: Some(merge),
        tree: None,
        branches: vec![],
        step: Some(Step {
            state: super::StepState::Separate,
            operands: Vec::new(),
            identity: identities,
            call: Callback::Source(source.clone()),
            implementation: Some(step),
        }),
        segment: None,
    };
    Ok((prefix, reduction, suffix))
}
fn literal(dtype: DType, value: f64, span: Span) -> Expr {
    let kind = match dtype {
        DType::Bool => ExprKind::Bool(value != 0.0),
        DType::I32 | DType::U32 => ExprKind::Int(value as i64),
        _ => ExprKind::Float(value),
    };
    Expr {
        kind,
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    }
}
fn binary(op: BinaryOp, a: Expr, b: Expr, dtype: DType, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(a),
            rhs: Box::new(b),
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    }
}
fn point(tile: &Expr, coordinates: &[Expr], span: Span) -> Expr {
    Expr {
        kind: ExprKind::Index {
            base: Box::new(tile.clone()),
            indices: coordinates.iter().cloned().map(Index::Point).collect(),
        },
        ty: Ty::Scalar(tile.ty.shaped().unwrap().elem.read_dtype().unwrap()),
        sym: None,
        span,
    }
}
fn fill(b: &mut Builder<'_>, tile: &Expr, compute: impl FnOnce(&[Expr]) -> Expr) -> Stmt {
    let coordinates: Vec<_> = tile
        .ty
        .shaped()
        .unwrap()
        .shape
        .iter()
        .map(|_| b.index())
        .collect();
    let target = point(tile, &coordinates, b.span);
    let assignment = stmt(
        StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value: compute(&coordinates),
        },
        b.span,
    );
    stmt(
        StmtKind::Owned {
            vars: coordinates
                .iter()
                .map(|e| {
                    if let ExprKind::Var(v) = e.kind {
                        v
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
            tile: tile.clone(),
            body: vec![assignment],
        },
        b.span,
    )
}
fn implementation(b: &mut Builder<'_>, state: &[Expr], c: Contract) -> Merge {
    let left: Vec<_> = state.iter().map(|s| b.local(s.ty.clone())).collect();
    let right: Vec<_> = state.iter().map(|s| b.local(s.ty.clone())).collect();
    let output: Vec<_> = state.iter().map(|s| b.local(s.ty.clone())).collect();
    let coordinates: Vec<_> = state[0]
        .ty
        .shaped()
        .unwrap()
        .shape
        .iter()
        .map(|_| b.index())
        .collect();
    let span = b.span;
    let a = point(&left[0], &coordinates, span);
    let z = point(&right[0], &coordinates, span);
    let assign = |field: usize, value: Expr| {
        stmt(
            StmtKind::Assign {
                target: point(&output[field], &coordinates, span),
                op: AssignOp::Assign,
                value,
            },
            span,
        )
    };
    let body = match c.combination() {
        Combination::FloatingAdd | Combination::LogicalOr | Combination::LogicalAnd => {
            let op = match c.combination() {
                Combination::LogicalOr => BinaryOp::Or,
                Combination::LogicalAnd => BinaryOp::And,
                _ => BinaryOp::Add,
            };
            vec![assign(0, binary(op, a, z, c.input, span))]
        }
        Combination::Maximum | Combination::Minimum => vec![assign(
            0,
            Expr {
                kind: ExprKind::Builtin {
                    name: if c.combination() == Combination::Maximum {
                        Builtin::Max
                    } else {
                        Builtin::Min
                    },
                    args: vec![a, z],
                },
                ty: Ty::Scalar(c.input),
                sym: None,
                span,
            },
        )],
        Combination::SaturatingAdd => {
            let sum = b.local(Ty::Scalar(c.input));
            let binding = stmt(
                StmtKind::Assign {
                    target: sum.clone(),
                    op: AssignOp::Assign,
                    value: binary(BinaryOp::Add, a.clone(), z.clone(), c.input, span),
                },
                span,
            );
            let (overflow, saturated) = if c.input == DType::U32 {
                (
                    binary(BinaryOp::Lt, sum.clone(), a, DType::Bool, span),
                    vec![assign(0, literal(DType::U32, u32::MAX as f64, span))],
                )
            } else {
                let bits = binary(
                    BinaryOp::BitAnd,
                    binary(BinaryOp::BitXor, a.clone(), sum.clone(), c.input, span),
                    binary(BinaryOp::BitXor, z, sum.clone(), c.input, span),
                    c.input,
                    span,
                );
                let negative = binary(BinaryOp::Lt, a, integer(0, span), DType::Bool, span);
                let clamp = stmt(
                    StmtKind::If {
                        cond: negative,
                        then: vec![assign(0, literal(DType::I32, i32::MIN as f64, span))],
                        els: vec![assign(0, literal(DType::I32, i32::MAX as f64, span))],
                    },
                    span,
                );
                (
                    binary(BinaryOp::Lt, bits, integer(0, span), DType::Bool, span),
                    vec![clamp],
                )
            };
            vec![
                binding,
                stmt(
                    StmtKind::If {
                        cond: overflow,
                        then: saturated,
                        els: vec![assign(0, sum)],
                    },
                    span,
                ),
            ]
        }
        Combination::FirstMaximum => {
            let cond = if c.input == DType::Bool {
                let not = Expr {
                    kind: ExprKind::Unary {
                        op: crate::ast::UnaryOp::Not,
                        expr: Box::new(a),
                    },
                    ty: Ty::Scalar(DType::Bool),
                    sym: None,
                    span,
                };
                binary(BinaryOp::And, z, not, DType::Bool, span)
            } else {
                binary(BinaryOp::Gt, z, a, DType::Bool, span)
            };
            // Strict improvement retains the earlier index on ties and ignores
            // unordered floating comparisons exactly as the reference fold.
            // Both fields observe one comparison before either is published.
            let choose_right = b.local(Ty::Scalar(DType::Bool));
            let mut body = vec![stmt(StmtKind::Assign {
                target: choose_right.clone(), op: AssignOp::Assign, value: cond,
            }, span)];
            body.extend((0..2).map(|field| {
                let yes = point(&right[field], &coordinates, span);
                let no = point(&left[field], &coordinates, span);
                assign(field, Expr {
                    ty: yes.ty.clone(), sym: None, span,
                    kind: ExprKind::Builtin { name: Builtin::Select, args: vec![choose_right.clone(), yes, no] },
                })
            }));
            body
        }
    };
    Merge {
        left,
        right,
        output: output.clone(),
        body: vec![stmt(
            StmtKind::Owned {
                vars: coordinates
                    .iter()
                    .map(|e| {
                        if let ExprKind::Var(v) = e.kind {
                            v
                        } else {
                            unreachable!()
                        }
                    })
                    .collect(),
                tile: output[0].clone(),
                body,
            },
            span,
        )],
    }
}
