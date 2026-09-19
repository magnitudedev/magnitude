//! One pointwise partition definition over the original positive piece width.
//! Full pieces and the final remainder retain separate typed tile shapes;
//! changing width never constructs another complete function alternative.
use super::*;

pub struct Retained {
    pub function: LoweredIr,
    pub parameters: Vec<(usize, DType)>,
    pub indices: Vec<VarId>,
    pub aliases: Vec<(VarId, VarId)>,
}

pub fn apply(function: &LoweredIr, pieces: &[Sym]) -> Result<Retained, String> {
    // This is the existing independent-element and alias proof. Its transformed
    // function is discarded; only the proof's parameter obligations are used.
    let parameters = super::pointwise(function, 1)?.parameters;
    if pieces.len() != function.body.len() { return Err("one retained partition width is required per phase".into()); }
    let mut function = function.clone();
    let mut indices = Vec::new();
    let mut aliases = Vec::new();
    for (phase, piece) in function.body.iter_mut().zip(pieces) {
        let StmtKind::Parallel { vars, extents, body } = &mut phase.kind else { unreachable!() };
        let tiles = body.iter().filter_map(|statement| {
            let StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), ty: Ty::Tile(_), .. }, .. } = &statement.kind else { return None; };
            Some(*variable)
        }).collect::<HashSet<_>>();
        let width = tiles.iter().filter_map(|variable| function.vars[*variable].ty.shaped()?.shape.first()?.as_constant())
            .next().ok_or("retained partition has no proven tile extent")?;
        let index = function.vars.len();
        let atom = Atom::Param(format!("partition#{index}"));
        function.vars.push(Var { name: "partition".into(), ty: Ty::Scalar(DType::I32),
            span: phase.span, kind: VarKind::Index(atom.clone()) });
        indices.push(index); vars.push(index);
        // This header is the finite structural envelope. Dispatch uses the
        // original ceil(width/piece) value retained by the backend mapping.
        extents.push(Sym::constant(width));
        let complete = Sym::constant(width).quot(piece);
        let remainder = Sym::constant(width).rem(piece);
        let start = Sym::atom(atom.clone()).mul(piece);
        let (mut tail, copies) = crate::widen::copy_bindings(body, &mut function.vars);
        let tail_tiles = tiles.iter().map(|variable| copies.get(variable).copied().unwrap_or(*variable)).collect();
        aliases.extend(copies.into_iter());
        resize_body(body, &mut function.vars, &tiles, &start, piece);
        resize_body(&mut tail, &mut function.vars, &tail_tiles, &start, &remainder);
        let condition = Expr { kind: ExprKind::Binary { op: crate::ast::BinaryOp::Lt,
            lhs: Box::new(Expr { kind: ExprKind::Var(index), ty: Ty::Scalar(DType::I32), sym: Some(Sym::atom(atom)), span: phase.span }),
            rhs: Box::new(symbol(complete, phase.span)) }, ty: Ty::Scalar(DType::Bool), sym: None, span: phase.span };
        *body = vec![Stmt { id: None, span: phase.span, kind: StmtKind::If { cond: condition, then: std::mem::take(body), els: tail } }];
    }
    Ok(Retained { function, parameters, indices, aliases })
}

fn symbol(value: Sym, span: crate::span::Span) -> Expr {
    let kind = value.as_constant().map(ExprKind::Int).unwrap_or_else(|| ExprKind::ShapeParam("partition_geometry".into()));
    Expr { kind, ty: Ty::Scalar(DType::I32), sym: Some(value), span }
}
fn resize(ty: &mut Ty, piece: &Sym) {
    if let Ty::Tile(shape) = ty { shape.shape[0] = piece.clone(); }
}
fn expression(value: &mut Expr, piece: &Sym) {
    resize(&mut value.ty, piece);
    match &mut value.kind {
        ExprKind::Index { base, .. } | ExprKind::Unary { expr: base, .. } | ExprKind::Cast { expr: base, .. } => expression(base, piece),
        ExprKind::Binary { lhs, rhs, .. } => { expression(lhs, piece); expression(rhs, piece); },
        ExprKind::Builtin { args, .. } => for argument in args { expression(argument, piece); },
        _ => {},
    }
}
fn slice(view: &Expr, start: &Sym, piece: &Sym) -> Expr {
    let Ty::Tensor(mut shape) = view.ty.clone() else { unreachable!() };
    shape.shape[0] = piece.clone();
    Expr { kind: ExprKind::Index { base: Box::new(view.clone()), indices: vec![Index::Slice {
        start: Some(symbol(start.clone(), view.span)), end: Some(symbol(start.add(piece), view.span)),
    }] }, ty: Ty::Tensor(shape), sym: None, span: view.span }
}
fn resize_body(body: &mut [Stmt], vars: &mut [Var], tiles: &HashSet<VarId>, start: &Sym, piece: &Sym) {
    for &variable in tiles { resize(&mut vars[variable].ty, piece); }
    for statement in body {
        match &mut statement.kind {
            StmtKind::Assign { target, value, .. } => {
                if matches!(target.ty, Ty::Scalar(_)) { continue; }
                resize(&mut target.ty, piece); resize(&mut value.ty, piece);
                match &mut value.kind {
                    ExprKind::TileAlloc { shape, .. } => shape[0] = piece.clone(),
                    ExprKind::Builtin { args, .. } => args[0] = slice(&args[0], start, piece),
                    ExprKind::Var(_) => {},
                    _ => unreachable!(),
                }
            },
            StmtKind::Owned { tile, body, .. } => {
                resize(&mut tile.ty, piece);
                for statement in body {
                    if let StmtKind::Assign { target, value, .. } = &mut statement.kind { expression(target, piece); expression(value, piece); }
                }
            },
            StmtKind::Expr(Expr { kind: ExprKind::Builtin { args, .. }, .. }) => {
                resize(&mut args[0].ty, piece); args[1] = slice(&args[1], start, piece);
            },
            _ => unreachable!(),
        }
    }
}
