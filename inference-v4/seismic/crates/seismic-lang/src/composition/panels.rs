//! Buffer pure operand preparation across contiguous matrix iterations while
//! leaving the ordered fragment updates inside their original serial traversal.
//! The buffers and loops are ordinary IR consumed by storage and emission.
use super::*;
use crate::intrinsics::Operation;

struct Producer {
    output: VarId,
    indices: Vec<VarId>,
    definition: Vec<Stmt>,
    allocation: usize,
    owned: usize,
}
struct Panel {
    iteration: VarId,
    atom: Atom,
    lo: i64,
    hi: i64,
    producers: Vec<Producer>,
    remaining: Vec<Stmt>,
}

pub(super) fn select(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    choose: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for statement in body {
        if let Some(panel) = admit(statement, vars) {
            let decision = Decision {
                kind: DecisionKind::MatrixPanel {
                    iteration: panel.iteration,
                    iterations: panel.hi - panel.lo,
                    operands: panel.producers.iter().map(|p| p.output).collect(),
                },
                alternatives: crate::lowered_ir::Alternatives::MatrixPanelWidths {
                    maximum: panel.hi - panel.lo,
                },
            };
            let selected = choose(&decision)?;
            if !decision.alternatives.contains(&selected) {
                return Err("matrix panel width is outside its iteration domain".into());
            }
            let Alternative::MatrixPanelWidth(width) = selected else {
                unreachable!()
            };
            if width > 1 {
                *statement = materialize(panel, width, vars, statement.span)?;
            }
        }
        let mut error = None;
        nested_mut(statement, &mut |nested| {
            if error.is_none() {
                error = select(nested, vars, choose).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    Ok(())
}

fn admit(statement: &Stmt, vars: &[Var]) -> Option<Panel> {
    let StmtKind::Range { var, lo, hi, body } = &statement.kind else {
        return None;
    };
    let (lo, hi) = (lo.as_constant()?, hi.as_constant()?);
    if lo < 0 || hi > i64::from(i32::MAX) || hi - lo < 2 {
        return None;
    }
    let VarKind::Index(atom) = &vars[*var].kind else {
        return None;
    };
    let mut producers = Vec::new();
    let mut remaining = Vec::new();
    let mut allocations = HashMap::new();
    for (at, s) in body.iter().enumerate() {
        match &s.kind {
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
                            ExprKind::TileAlloc {
                                shape,
                                dtype: Elem::Dtype(_),
                            },
                        ..
                    },
            } => {
                let count = shape
                    .iter()
                    .try_fold(1i64, |n, s| n.checked_mul(s.as_constant()?.max(0)))?;
                count.checked_mul(hi - lo)?;
                if allocations.insert(*output, at).is_some() {
                    return None;
                }
            }
            StmtKind::Owned {
                vars: indices,
                tile:
                    Expr {
                        kind: ExprKind::Var(output),
                        ..
                    },
                body: producer,
            } => {
                let allocation = *allocations.get(output)?;
                if producers.iter().any(|p: &Producer| p.output == *output) {
                    return None;
                }
                let (target, definition) = crate::lower::producer_definition(
                    producer,
                    *output,
                    vars,
                    &body[..at],
                    &body[at + 1..],
                )?;
                let ExprKind::Index {
                    indices: points, ..
                } = &target.kind
                else {
                    return None;
                };
                if points.len() != indices.len() || !points.iter().zip(indices).all(|(p,v)| matches!(p, Index::Point(e) if matches!(e.kind, ExprKind::Var(i) if i == *v))) { return None; }
                producers.push(Producer {
                    output: *output,
                    indices: indices.clone(),
                    definition,
                    allocation,
                    owned: at,
                });
            }
            StmtKind::Assign {
                target,
                op: AssignOp::Assign,
                value:
                    Expr {
                        kind:
                            ExprKind::Intrinsic {
                                op: Operation::Matrix,
                                ..
                            },
                        ..
                    },
            } if matches!(target.ty, Ty::Frag(_)) => remaining.push(s.clone()),
            StmtKind::Expr(Expr {
                kind:
                    ExprKind::Intrinsic {
                        op:
                            Operation::MatrixLoad
                            | Operation::MatrixLoadTranspose
                            | Operation::MatrixMultiplyAccumulate,
                        ..
                    },
                ..
            }) => remaining.push(s.clone()),
            _ => return None,
        }
    }
    if producers.is_empty()
        || producers.len() != allocations.len()
        || !remaining.iter().any(|s| {
            matches!(
                s.kind,
                StmtKind::Expr(Expr {
                    kind: ExprKind::Intrinsic {
                        op: Operation::MatrixMultiplyAccumulate,
                        ..
                    },
                    ..
                })
            )
        })
    {
        return None;
    }
    let written = written(&remaining);
    let prepared: HashSet<_> = producers.iter().map(|p| p.output).collect();
    fn pure_reads(body: &[Stmt], forbidden: &HashSet<VarId>) -> bool {
        body.iter().all(|s| match &s.kind {
            StmtKind::Assign { value, .. } => !crate::effects::expressions(
                value,
                &|e| matches!(e.kind, ExprKind::Var(v) if forbidden.contains(&v)),
            ),
            StmtKind::If { cond, then, els } => {
                !crate::effects::expressions(
                    cond,
                    &|e| matches!(e.kind, ExprKind::Var(v) if forbidden.contains(&v)),
                ) && pure_reads(then, forbidden)
                    && pure_reads(els, forbidden)
            }
            _ => false,
        })
    }
    let forbidden = written
        .into_iter()
        .chain(prepared.iter().copied())
        .collect();
    if producers
        .iter()
        .any(|p| !pure_reads(&p.definition, &forbidden))
    {
        return None;
    }
    // Every prepared value is consumed only by read-only matrix loads after its
    // complete producer. No stores, scalar side effects or other tile mutations
    // can cross the panel's preparation boundary.
    for (at, s) in body.iter().enumerate() {
        for p in &producers {
            if at != p.allocation && at != p.owned && crate::effects::uses(s, p.output) {
                if at <= p.owned
                    || !matches!(&s.kind, StmtKind::Expr(Expr { kind: ExprKind::Intrinsic { op: Operation::MatrixLoad | Operation::MatrixLoadTranspose, args }, .. }) if matches!(args[1].kind, ExprKind::Var(v) if v == p.output))
                {
                    return None;
                }
            }
        }
    }
    Some(Panel {
        iteration: *var,
        atom: atom.clone(),
        lo,
        hi,
        producers,
        remaining,
    })
}

fn index(name: &str, vars: &mut Vec<Var>, span: Span) -> (VarId, Sym) {
    let id = vars.len();
    let atom = Atom::Param(format!("{name}#{id}"));
    vars.push(Var {
        name: format!("{name}_{id}"),
        ty: Ty::Scalar(DType::I32),
        kind: VarKind::Index(atom.clone()),
        span,
    });
    (id, Sym::atom(atom))
}
fn symbolic(value: Sym, span: Span) -> Expr {
    Expr {
        kind: ExprKind::ShapeParam(value.to_string()),
        ty: Ty::Scalar(DType::I32),
        sym: Some(value),
        span,
    }
}
fn reference(v: VarId, vars: &[Var], span: Span) -> Expr {
    Expr {
        kind: ExprKind::Var(v),
        ty: vars[v].ty.clone(),
        sym: None,
        span,
    }
}
fn condition(
    outer: VarId,
    slot: VarId,
    width: i64,
    lo: i64,
    hi: i64,
    vars: &[Var],
    span: Span,
) -> Expr {
    // The padded last panel can extend beyond I32 even though every consumed
    // source coordinate fits. Test its ordinal in U32 before evaluating the
    // original I32 index inside the live branch. The padded ordinal is less
    // than hi + width, which fits U32 for the admitted nonnegative I32 range.
    let integer = |n| Expr {
        kind: ExprKind::Int(n),
        ty: Ty::Scalar(DType::U32),
        sym: Some(Sym::constant(n)),
        span,
    };
    let cast = |v| Expr {
        kind: ExprKind::Cast {
            dtype: DType::U32,
            expr: Box::new(reference(v, vars, span)),
        },
        ty: Ty::Scalar(DType::U32),
        sym: None,
        span,
    };
    let binary = |op, lhs, rhs| Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ty: Ty::Scalar(DType::U32),
        sym: None,
        span,
    };
    let offset = binary(crate::ast::BinaryOp::Mul, cast(outer), integer(width));
    let offset = binary(crate::ast::BinaryOp::Add, offset, integer(lo));
    let offset = binary(crate::ast::BinaryOp::Add, offset, cast(slot));
    Expr {
        kind: ExprKind::Binary {
            op: crate::ast::BinaryOp::Lt,
            lhs: Box::new(offset),
            rhs: Box::new(integer(hi)),
        },
        ty: Ty::Scalar(DType::Bool),
        sym: None,
        span,
    }
}

fn substitute(body: &mut [Stmt], values: &HashMap<VarId, Expr>) {
    for s in body {
        match &mut s.kind {
            StmtKind::Assign { target, value, .. } => {
                *target = crate::lower::subst_vars(target, values, &HashMap::new());
                *value = crate::lower::subst_vars(value, values, &HashMap::new());
            }
            StmtKind::Expr(e) => *e = crate::lower::subst_vars(e, values, &HashMap::new()),
            StmtKind::If { cond, then, els } => {
                *cond = crate::lower::subst_vars(cond, values, &HashMap::new());
                substitute(then, values);
                substitute(els, values);
            }
            _ => unreachable!("panel definitions and fragment consumers were admitted above"),
        }
    }
}
fn panel_view(buffer: VarId, offset: Sym, original: Ty, vars: &[Var], span: Span) -> Expr {
    let mut indices = vec![Index::Point(symbolic(offset, span))];
    indices.extend(
        original
            .shaped()
            .unwrap()
            .shape
            .iter()
            .map(|n| Index::Slice {
                start: Some(symbolic(Sym::constant(0), span)),
                end: Some(symbolic(n.clone(), span)),
            }),
    );
    Expr {
        kind: ExprKind::Index {
            base: Box::new(reference(buffer, vars, span)),
            indices,
        },
        ty: original,
        sym: None,
        span,
    }
}
fn materialize(panel: Panel, width: i64, vars: &mut Vec<Var>, span: Span) -> Result<Stmt, String> {
    let (outer, outer_index) = index("matrix_panel", vars, span);
    let (inner, inner_index) = index("matrix_panel_item", vars, span);
    let base = outer_index.scale(width).add(&Sym::constant(panel.lo));
    let mut body = Vec::new();
    let mut replacements = HashMap::new();
    for p in panel.producers {
        let Ty::Tile(mut shape) = vars[p.output].ty.clone() else {
            unreachable!()
        };
        shape.shape.insert(0, Sym::constant(width));
        let buffer = vars.len();
        vars.push(Var {
            name: format!("matrix_panel_{}_{buffer}", vars[p.output].name),
            ty: Ty::Tile(shape.clone()),
            kind: VarKind::Local,
            span,
        });
        body.push(Stmt {
            id: None,
            span,
            kind: StmtKind::Assign {
                target: reference(buffer, vars, span),
                op: AssignOp::Assign,
                value: Expr {
                    kind: ExprKind::TileAlloc {
                        shape: shape.shape,
                        dtype: shape.elem,
                    },
                    ty: vars[buffer].ty.clone(),
                    sym: None,
                    span,
                },
            },
        });
        let (slot, slot_index) = index("matrix_panel_prepare", vars, span);
        let mut definition = p.definition;
        let logical = base.add(&slot_index);
        crate::widen::replace_index(
            &mut definition,
            panel.iteration,
            &panel.atom,
            &symbolic(logical.clone(), span),
        );
        substitute(
            &mut definition,
            &HashMap::from([(
                p.output,
                panel_view(buffer, slot_index, vars[p.output].ty.clone(), vars, span),
            )]),
        );
        let mut indices = vec![slot];
        indices.extend(p.indices);
        body.push(Stmt {
            id: None,
            span,
            kind: StmtKind::Owned {
                vars: indices,
                tile: reference(buffer, vars, span),
                body: vec![Stmt {
                    id: None,
                    span,
                    kind: StmtKind::If {
                        cond: condition(outer, slot, width, panel.lo, panel.hi, vars, span),
                        then: definition,
                        els: vec![],
                    },
                }],
            },
        });
        replacements.insert(
            p.output,
            panel_view(
                buffer,
                inner_index.clone(),
                vars[p.output].ty.clone(),
                vars,
                span,
            ),
        );
    }
    let mut remaining = panel.remaining;
    let logical = base.add(&inner_index);
    crate::widen::replace_index(
        &mut remaining,
        panel.iteration,
        &panel.atom,
        &symbolic(logical.clone(), span),
    );
    substitute(&mut remaining, &replacements);
    body.push(Stmt {
        id: None,
        span,
        kind: StmtKind::Range {
            var: inner,
            lo: Sym::constant(0),
            hi: Sym::constant(width),
            body: vec![Stmt {
                id: None,
                span,
                kind: StmtKind::If {
                    cond: condition(outer, inner, width, panel.lo, panel.hi, vars, span),
                    then: remaining,
                    els: vec![],
                },
            }],
        },
    });
    Ok(Stmt {
        id: None,
        span,
        kind: StmtKind::Range {
            var: outer,
            lo: Sym::constant(0),
            hi: Sym::constant((panel.hi - panel.lo + width - 1) / width),
            body,
        },
    })
}
