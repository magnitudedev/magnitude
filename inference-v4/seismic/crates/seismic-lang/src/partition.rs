//! Proven independent pointwise tile partitioning. No cost policy lives here.
//! Eligibility is deliberately explicit: dense equal-shape row views, same-index
//! scalar expressions, and no reductions, cross-element reads or control effects.
use crate::{
    ast::AssignOp,
    ir::*,
    lowered_ir::LoweredIr,
    sym::{Atom, Sym},
    types::{DType, Elem, Ty},
};
use std::collections::HashSet;

pub struct Partitioned {
    pub function: LoweredIr,
    /// Runtime must reject overlapping parameter storage unless these parameters
    /// have identical base addresses and identical dense element types.
    pub parameters: Vec<(usize, DType)>,
}

pub fn pointwise(function: &LoweredIr, piece: i64) -> Result<Partitioned, String> {
    if piece <= 0 {
        return Err("pointwise partition must be positive".into());
    }
    let mut f = function.clone();
    let mut parameters = HashSet::new();
    for phase in &mut f.body {
        let StmtKind::Parallel {
            vars,
            extents,
            body,
        } = &mut phase.kind
        else {
            return Err("pointwise partition requires parallel phases".into());
        };
        let mut tiles = HashSet::new();
        let mut width = None;
        let mut view_key = None;
        let mut stores = 0;
        for statement in body.iter() {
            match &statement.kind {
                StmtKind::Assign {
                    target,
                    op: AssignOp::Assign,
                    value,
                } => {
                    let ExprKind::Var(v) = target.kind else {
                        return Err("partition expects named tiles".into());
                    };
                    let Ty::Tile(t) = &target.ty else {
                        return Err("partition only supports tile assignments".into());
                    };
                    if t.shape.len() != 1 || !matches!(t.elem, Elem::Dtype(_)) {
                        return Err("partition requires dense rank-one tiles".into());
                    }
                    let n = t.shape[0]
                        .as_constant()
                        .ok_or("partition tile extent must be concrete")?;
                    if n <= 0 || n % piece != 0 || width.is_some_and(|w| w != n) {
                        return Err("partition must divide every equal tile extent".into());
                    }
                    width = Some(n);
                    match &value.kind {
                        ExprKind::TileAlloc { .. } => {}
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        } if args.len() == 1 => {
                            check_view(&args[0], &mut view_key, &mut parameters)?
                        }
                        _ => {
                            return Err(
                                "partition only supports direct loads and allocations".into()
                            )
                        }
                    }
                    if !tiles.insert(v) {
                        return Err("partition tile rebinding is unsupported".into());
                    }
                }
                StmtKind::Owned { vars, tile, body } if vars.len() == 1 => {
                    if !matches!(tile.kind,ExprKind::Var(v) if tiles.contains(&v)) {
                        return Err("partition owned domain must be a local tile".into());
                    }
                    for statement in body {
                        let StmtKind::Assign {
                            target,
                            op: AssignOp::Assign,
                            value,
                        } = &statement.kind
                        else {
                            return Err(
                                "partition owned body must assign independent elements".into()
                            );
                        };
                        element(target, vars[0], &tiles)?;
                        scalar(value, vars[0], &tiles)?;
                    }
                }
                StmtKind::Expr(Expr {
                    kind:
                        ExprKind::Builtin {
                            name: Builtin::Store,
                            args,
                        },
                    ..
                }) if args.len() == 2 => {
                    if !matches!(args[0].kind,ExprKind::Var(v) if tiles.contains(&v)) {
                        return Err("partition stores must publish local tiles".into());
                    }
                    check_view(&args[1], &mut view_key, &mut parameters)?;
                    stores += 1;
                }
                _ => return Err("partition cannot prove this phase pointwise independent".into()),
            }
        }
        if stores == 0 {
            return Err("partition phase has no publication".into());
        }
        let width = width.ok_or("partition phase has no tiles")?;
        let id = f.vars.len();
        let atom = Atom::Param(format!("partition#{id}"));
        f.vars.push(Var {
            name: "partition".into(),
            ty: Ty::Scalar(DType::I32),
            span: phase.span,
            kind: VarKind::Index(atom.clone()),
        });
        vars.push(id);
        extents.push(Sym::constant(width / piece));
        let start = Sym::atom(atom.clone()).scale(piece);
        for v in &tiles {
            resize(&mut f.vars[*v].ty, piece);
        }
        for statement in body {
            match &mut statement.kind {
                StmtKind::Assign { target, value, .. } => {
                    resize(&mut target.ty, piece);
                    resize(&mut value.ty, piece);
                    match &mut value.kind {
                        ExprKind::TileAlloc { shape, .. } => shape[0] = Sym::constant(piece),
                        ExprKind::Builtin { args, .. } => {
                            args[0] = slice(&args[0], id, &atom, &start, piece)
                        }
                        _ => unreachable!(),
                    }
                }
                StmtKind::Owned { tile, body, .. } => {
                    resize(&mut tile.ty, piece);
                    for statement in body {
                        if let StmtKind::Assign { target, value, .. } = &mut statement.kind {
                            resize_expr(target, piece);
                            resize_expr(value, piece);
                        }
                    }
                }
                StmtKind::Expr(Expr {
                    kind: ExprKind::Builtin { args, .. },
                    ..
                }) => {
                    resize(&mut args[0].ty, piece);
                    args[1] = slice(&args[1], id, &atom, &start, piece);
                }
                _ => unreachable!(),
            }
        }
    }
    let mut parameters = parameters
        .into_iter()
        .map(|i| {
            let Ty::Tensor(t) = &f.vars[i].ty else {
                unreachable!()
            };
            let Elem::Dtype(dtype) = t.elem else {
                unreachable!()
            };
            (i, dtype)
        })
        .collect::<Vec<_>>();
    parameters.sort_by_key(|(i, _)| *i);
    Ok(Partitioned {
        function: f,
        parameters,
    })
}
// Views must address the same logical element coordinates. Equal base pointers
// are therefore safe only with the same element width/type; shifted overlap is not.
type ViewKey = (Vec<Sym>, Vec<Sym>);
fn check_view(
    e: &Expr,
    key: &mut Option<ViewKey>,
    params: &mut HashSet<usize>,
) -> Result<(), String> {
    let (base, indices) = match &e.kind {
        ExprKind::Var(_) => (e, &[][..]),
        ExprKind::Index { base, indices } => (base.as_ref(), indices.as_slice()),
        _ => return Err("partition requires direct parameter row views".into()),
    };
    let ExprKind::Var(v) = base.kind else {
        return Err("partition view base must be a parameter".into());
    };
    let Ty::Tensor(t) = &base.ty else {
        return Err("partition view base must be a tensor".into());
    };
    if !matches!(t.elem, Elem::Dtype(_)) || t.shape.len() != indices.len() + 1 {
        return Err("partition requires dense row views".into());
    }
    let coordinates = indices
        .iter()
        .map(|i| match i {
            Index::Point(e) => e.sym.clone().ok_or("partition row index must be symbolic"),
            _ => Err("partition row slices are unsupported"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let actual = (t.shape.clone(), coordinates);
    if key.as_ref().is_some_and(|k| *k != actual) {
        return Err("partition row coordinates and shapes must match".into());
    }
    *key = Some(actual);
    params.insert(v);
    Ok(())
}
fn element(e: &Expr, index: VarId, tiles: &HashSet<VarId>) -> Result<(), String> {
    match &e.kind {
        ExprKind::Index { base, indices }
            if matches!(base.kind,ExprKind::Var(v) if tiles.contains(&v))
                && matches!(indices.as_slice(),[Index::Point(Expr{kind:ExprKind::Var(v),..})] if *v==index) =>
        {
            Ok(())
        }
        _ => Err("partition requires same-index tile access".into()),
    }
}
fn scalar(e: &Expr, index: VarId, tiles: &HashSet<VarId>) -> Result<(), String> {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => Ok(()),
        ExprKind::Index { .. } => element(e, index, tiles),
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => scalar(expr, index, tiles),
        ExprKind::Binary { lhs, rhs, .. } => {
            scalar(lhs, index, tiles)?;
            scalar(rhs, index, tiles)
        }
        ExprKind::Builtin {
            name:
                Builtin::Fma
                | Builtin::Exp
                | Builtin::ExpFast
                | Builtin::Rsqrt
                | Builtin::Sqrt
                | Builtin::Log
                | Builtin::Sin
                | Builtin::Cos
                | Builtin::Abs
                | Builtin::Max
                | Builtin::Min,
            args,
        } => {
            for arg in args {
                scalar(arg, index, tiles)?;
            }
            Ok(())
        }
        _ => Err("partition expression may depend on other elements or indices".into()),
    }
}
fn resize(ty: &mut Ty, piece: i64) {
    if let Ty::Tile(t) = ty {
        t.shape[0] = Sym::constant(piece);
    }
}
fn resize_expr(e: &mut Expr, piece: i64) {
    resize(&mut e.ty, piece);
    match &mut e.kind {
        ExprKind::Index { base, .. } => resize_expr(base, piece),
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => resize_expr(expr, piece),
        ExprKind::Binary { lhs, rhs, .. } => {
            resize_expr(lhs, piece);
            resize_expr(rhs, piece);
        }
        ExprKind::Builtin { args, .. } => {
            for arg in args {
                resize_expr(arg, piece);
            }
        }
        _ => {}
    }
}
fn slice(view: &Expr, id: VarId, atom: &Atom, start: &Sym, piece: i64) -> Expr {
    use crate::ast::BinaryOp;
    let literal = |n| Expr {
        kind: ExprKind::Int(n),
        ty: Ty::Scalar(DType::I32),
        sym: Some(Sym::constant(n)),
        span: view.span,
    };
    let index = Expr {
        kind: ExprKind::Var(id),
        ty: Ty::Scalar(DType::I32),
        sym: Some(Sym::atom(atom.clone())),
        span: view.span,
    };
    let begin = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(index),
            rhs: Box::new(literal(piece)),
        },
        ty: Ty::Scalar(DType::I32),
        sym: Some(start.clone()),
        span: view.span,
    };
    let end = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(begin.clone()),
            rhs: Box::new(literal(piece)),
        },
        ty: Ty::Scalar(DType::I32),
        sym: Some(start.add(&Sym::constant(piece))),
        span: view.span,
    };
    let Ty::Tensor(mut t) = view.ty.clone() else {
        unreachable!()
    };
    t.shape[0] = Sym::constant(piece);
    Expr {
        kind: ExprKind::Index {
            base: Box::new(view.clone()),
            indices: vec![Index::Slice {
                start: Some(begin),
                end: Some(end),
            }],
        },
        ty: Ty::Tensor(t),
        sym: None,
        span: view.span,
    }
}
