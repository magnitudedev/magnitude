//! Specialize an existing construct call to an independent output rectangle.
//! The checked portable definition supplies the legality facts; the resulting
//! call still owns all applicable backend implementations.
use super::*;
use std::collections::HashSet;

/// Independent dimensions of the single state/output parameter, derived from
/// its portable access maps. Both narrowing and grouping use this proof.
pub(super) struct OutputAxes {
    pub output: usize,
    pub dimensions: Vec<(usize, String)>,
}

pub(super) fn output_axes(
    program: &Program,
    function: &Function,
) -> Result<Option<OutputAxes>, String> {
    let (vars, body) = portable_body(program, function)?;
    let mut written = HashSet::new();
    for s in &body {
        crate::rewrite::writes(s, &mut written);
    }
    let outputs = written
        .into_iter()
        .filter(|v| matches!(vars[*v].kind, VarKind::Param(_)))
        .collect::<Vec<_>>();
    let [output] = outputs.as_slice() else {
        return Ok(None);
    };
    let Some(shape) = vars[*output].ty.shaped() else {
        return Ok(None);
    };
    let dimensions = shape
        .shape
        .iter()
        .enumerate()
        .filter_map(|(axis, extent)| {
            let parameter = function
                .shape_params
                .iter()
                .find(|p| *extent == Sym::param(p))?;
            let changed = HashSet::from([parameter.clone()]);
            independent_parameters(function, &vars, &body, *output, &changed)
                .then_some((axis, parameter.clone()))
        })
        .collect();
    let VarKind::Param(output) = vars[*output].kind else {
        unreachable!()
    };
    Ok(Some(OutputAxes { output, dimensions }))
}

fn independent_parameters(
    function: &Function,
    vars: &[Var],
    body: &[Stmt],
    output: VarId,
    changed: &HashSet<String>,
) -> bool {
    !function
        .index_params
        .iter()
        .any(|(_, bound)| changed.iter().any(|name| bound.params().contains(name)))
        && function.params.iter().all(|(_, ty)| {
            ty.shaped().is_none_or(|shape| {
                shape.shape.iter().all(|extent| {
                    changed
                        .iter()
                        .all(|name| !extent.params().contains(name) || *extent == Sym::param(name))
                })
            })
        })
        && independent(body, output, vars, changed)
}

pub(super) fn call(
    program: &Program,
    call: &Expr,
    output: VarId,
    view: &Expr,
    target: &Expr,
    vars: &mut Vec<Var>,
) -> Result<Option<Vec<Stmt>>, String> {
    let ExprKind::Call {
        callee,
        shape_args,
        elem_args,
        args,
    } = &call.kind
    else {
        return Ok(None);
    };
    let Some(function) = program
        .functions
        .iter()
        .find(|f| f.name == *callee && f.is_construct)
    else {
        return Ok(None);
    };
    let outputs = args
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.kind,ExprKind::Var(v) if v==output))
        .map(|(n, _)| n)
        .collect::<Vec<_>>();
    let [output_parameter] = outputs.as_slice() else {
        return Ok(None);
    };
    let output_parameter = *output_parameter;
    let source = vars[output]
        .ty
        .shaped()
        .ok_or("call output must be shaped")?;
    let Some(result) = target.ty.shaped() else {
        return Ok(None);
    };
    let Some(formal) = function.params[output_parameter].1.shaped() else {
        return Ok(None);
    };
    if source.shape.len() != result.shape.len() || source.elem != result.elem {
        return Ok(None);
    };
    let starts = match &view.kind {
        ExprKind::Index { base, indices } if matches!(base.kind,ExprKind::Var(v) if v==output) => {
            (0..source.shape.len())
                .map(|axis| match indices.get(axis) {
                    None | Some(Index::Slice { start: None, .. }) => Some(Sym::constant(0)),
                    Some(Index::Slice { start: Some(e), .. }) => e.sym.clone(),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        }
        _ => None,
    };
    let Some(starts) = starts else {
        return Ok(None);
    };
    let mut changed = HashMap::<String, (Sym, Sym)>::new();
    for (axis, (old, new)) in source.shape.iter().zip(&result.shape).enumerate() {
        if old == new && starts[axis].as_constant() == Some(0) {
            continue;
        }
        let Some(name) = function
            .shape_params
            .iter()
            .find(|p| formal.shape[axis] == Sym::param(p))
        else {
            return Ok(None);
        };
        if changed
            .insert(name.clone(), (starts[axis].clone(), new.clone()))
            .is_some()
        {
            return Ok(None);
        };
    }
    if changed.is_empty() {
        return Ok(None);
    };
    for (name, (start, count)) in &changed {
        for (axis, extent) in formal.shape.iter().enumerate() {
            if *extent == Sym::param(name)
                && (&starts[axis] != start || &result.shape[axis] != count)
            {
                return Ok(None);
            }
        }
    }
    if args
        .iter()
        .enumerate()
        .any(|(n, e)| n != output_parameter && mentions_output(e, output))
    {
        return Ok(None);
    }
    let (logical_vars, logical) = portable_body(program, function)?;
    if !independent_parameters(
        function,
        &logical_vars,
        &logical,
        output_parameter,
        &changed.keys().cloned().collect(),
    ) {
        return Ok(None);
    };
    let checkpoint = vars.len();
    let mut prefix = Vec::new();
    let mut parameters = Vec::new();
    for (position, ((_, formal), actual)) in function.params.iter().zip(args).enumerate() {
        if position == output_parameter {
            parameters.push(target.clone());
            continue;
        }
        let Some(shape) = formal.shaped() else {
            parameters.push(actual.clone());
            continue;
        };
        let mut projected = actual.clone();
        for (axis, extent) in shape.shape.iter().enumerate() {
            if let Some((start, count)) = changed
                .iter()
                .find_map(|(name, range)| (*extent == Sym::param(name)).then_some(range))
            {
                let (statement, value) = decomposition::slice_input(
                    &projected,
                    axis,
                    start.clone(),
                    count.clone(),
                    vars,
                )?;
                prefix.push(statement);
                projected = value;
            }
        }
        parameters.push(projected);
    }
    if parameters
        .iter()
        .enumerate()
        .any(|(n, e)| n != output_parameter && mentions_output(e, output))
    {
        vars.truncate(checkpoint);
        return Ok(None);
    };
    let shapes = function
        .shape_params
        .iter()
        .zip(shape_args)
        .map(|(p, n)| changed.get(p).map_or_else(|| n.clone(), |(_, n)| n.clone()))
        .collect();
    prefix.push(Stmt {
        id: None,
        span: call.span,
        kind: StmtKind::Expr(Expr {
            kind: ExprKind::Call {
                callee: callee.clone(),
                shape_args: shapes,
                elem_args: elem_args.clone(),
                args: parameters,
            },
            ty: call.ty.clone(),
            sym: None,
            span: call.span,
        }),
    });
    Ok(Some(prefix))
}
fn mentions_output(e: &Expr, output: VarId) -> bool {
    let mut found = false;
    walk_expr(e, &mut |e| {
        found |= matches!(e.kind,ExprKind::Var(v) if v==output)
    });
    found
}
fn independent(body: &[Stmt], output: VarId, vars: &[Var], changed: &HashSet<String>) -> bool {
    let mut written = HashSet::new();
    for s in body {
        crate::rewrite::writes(s, &mut written);
    }
    if written
        .iter()
        .any(|v| *v != output && matches!(vars[*v].kind, VarKind::Param(_)))
    {
        return false;
    }
    let output_shape = vars[output].ty.shaped().unwrap();
    for s in body {
        let StmtKind::Owned {
            vars: indices,
            tile,
            body,
        } = &s.kind
        else {
            return false;
        };
        if !matches!(tile.kind,ExprKind::Var(v) if v==output) {
            return false;
        }
        let varied = indices
            .iter()
            .zip(&output_shape.shape)
            .filter_map(|(id, n)| {
                changed
                    .iter()
                    .any(|p| n.params().contains(p))
                    .then_some(*id)
            })
            .collect::<HashSet<_>>();
        if !statements(body, output, indices, &varied, vars, changed) {
            return false;
        }
    }
    !body.is_empty()
}
fn sensitive(
    value: &Sym,
    varied: &HashSet<VarId>,
    vars: &[Var],
    changed: &HashSet<String>,
) -> bool {
    changed.iter().any(|p| value.params().contains(p))
        || varied.iter().any(|v| match &vars[*v].kind {
            VarKind::Index(atom) => value.atoms().contains(atom),
            _ => false,
        })
}
fn statements(
    body: &[Stmt],
    output: VarId,
    coordinates: &[VarId],
    varied: &HashSet<VarId>,
    vars: &[Var],
    changed: &HashSet<String>,
) -> bool {
    body.iter().all(|s| match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            expression(target, false, output, coordinates, varied, vars, changed)
                && expression(value, false, output, coordinates, varied, vars, changed)
        }
        StmtKind::Expr(e) => expression(e, false, output, coordinates, varied, vars, changed),
        StmtKind::Owned { tile, body, .. } => {
            expression(tile, false, output, coordinates, varied, vars, changed)
                && !tile
                    .ty
                    .shaped()
                    .is_some_and(|t| t.shape.iter().any(|n| sensitive(n, varied, vars, changed)))
                && statements(body, output, coordinates, varied, vars, changed)
        }
        StmtKind::Range { lo, hi, body, .. } => {
            !sensitive(lo, varied, vars, changed)
                && !sensitive(hi, varied, vars, changed)
                && statements(body, output, coordinates, varied, vars, changed)
        }
        StmtKind::If { cond, then, els } => {
            expression(cond, false, output, coordinates, varied, vars, changed)
                && statements(then, output, coordinates, varied, vars, changed)
                && statements(els, output, coordinates, varied, vars, changed)
        }
        StmtKind::Reduction(r) => {
            !sensitive(r.extent(), varied, vars, changed)
                && r.operands()
                    .all(|e| expression(e, false, output, coordinates, varied, vars, changed))
                && r.bodies()
                    .all(|b| statements(b, output, coordinates, varied, vars, changed))
        }
        _ => false,
    })
}
fn expression(
    e: &Expr,
    address: bool,
    output: VarId,
    coordinates: &[VarId],
    varied: &HashSet<VarId>,
    vars: &[Var],
    changed: &HashSet<String>,
) -> bool {
    if !crate::effects::expression_can_be_omitted(e) {
        return false;
    }
    if !address
        && matches!(e.ty, Ty::Scalar(_))
        && e.sym
            .as_ref()
            .is_some_and(|s| sensitive(s, varied, vars, changed))
    {
        return false;
    }
    match &e.kind {
        ExprKind::Var(v) if *v == output => false,
        ExprKind::Var(v) if varied.contains(v) => address,
        ExprKind::Var(_)
            if e.ty
                .shaped()
                .is_some_and(|t| t.shape.iter().any(|n| sensitive(n, varied, vars, changed))) =>
        {
            false
        }
        ExprKind::ShapeParam(_) => !e
            .sym
            .as_ref()
            .is_some_and(|s| changed.iter().any(|p| s.params().contains(p))),
        ExprKind::Index { base, indices } => {
            let output_access = matches!(base.kind,ExprKind::Var(v) if v==output);
            if output_access && (indices.len()!=coordinates.len() || !indices.iter().zip(coordinates).all(|(index,v)|matches!((index,&vars[*v].kind),(Index::Point(e),VarKind::Index(a)) if e.sym.as_ref()==Some(&Sym::atom(a.clone()))))){return false;}
            if !output_access
                && !matches!(base.kind, ExprKind::Var(_))
                && !expression(base, false, output, coordinates, varied, vars, changed)
            {
                return false;
            }
            let Some(shape) = base.ty.shaped() else {
                return false;
            };
            if indices.len() > shape.shape.len() {
                return false;
            }
            shape.shape.iter().enumerate().all(|(axis, extent)| {
                let index = indices.get(axis);
                if changed.iter().any(|name| extent.params().contains(name))
                    && !matches!(
                        index,
                        Some(Index::Point(_))
                            | Some(Index::Slice {
                                start: Some(_),
                                end: Some(_)
                            })
                    )
                {
                    return false;
                }
                let admissible = |value: &Expr, end: bool| {
                    let present = varied
                        .iter()
                        .filter(|v| mentions_output(value, **v))
                        .copied()
                        .collect::<Vec<_>>();
                    if present.is_empty() {
                        if changed.iter().any(|name| extent.params().contains(name)) {
                            return false;
                        }
                        return expression(value, true, output, coordinates, varied, vars, changed);
                    }
                    let [index] = present.as_slice() else {
                        return false;
                    };
                    let Some(output_axis) = coordinates.iter().position(|v| v == index) else {
                        return false;
                    };
                    let Some(output_shape) = vars[output].ty.shaped() else {
                        return false;
                    };
                    if *extent != output_shape.shape[output_axis] {
                        return false;
                    }
                    let VarKind::Index(atom) = &vars[*index].kind else {
                        return false;
                    };
                    let expected = Sym::atom(atom.clone()).add(&Sym::constant(i64::from(end)));
                    value.sym.as_ref() == Some(&expected)
                };
                match index {
                    Some(Index::Point(value)) => admissible(value, false),
                    Some(Index::Slice { start, end }) => {
                        start.as_ref().is_none_or(|value| admissible(value, false))
                            && end.as_ref().is_none_or(|value| admissible(value, true))
                    }
                    None => true,
                }
            })
        }
        ExprKind::Call { .. }
        | ExprKind::Intrinsic { .. }
        | ExprKind::Builtin {
            name: Builtin::Store | Builtin::Atomic,
            ..
        } => false,
        ExprKind::Builtin {
            name: Builtin::Extent,
            ..
        } if e
            .sym
            .as_ref()
            .is_some_and(|s| changed.iter().any(|p| s.params().contains(p))) =>
        {
            false
        }
        ExprKind::TileAlloc { shape, .. }
            if shape.iter().any(|n| sensitive(n, varied, vars, changed)) =>
        {
            false
        }
        _ => {
            let mut valid = true;
            walk_children(e, &mut |e| {
                valid &= expression(e, address, output, coordinates, varied, vars, changed)
            });
            valid
        }
    }
}
fn walk_children(e: &Expr, f: &mut impl FnMut(&Expr)) {
    match &e.kind {
        ExprKind::Load { view: e, .. }
        | ExprKind::Transpose(e)
        | ExprKind::Accessor { base: e, .. }
        | ExprKind::Lanes { base: e, .. }
        | ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. } => f(e),
        ExprKind::Binary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        ExprKind::Builtin { args, .. } | ExprKind::Tuple(args) => {
            for e in args {
                f(e);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scope, program::SourceFile};
    fn projected(expression: &str) -> bool {
        let parsed = if expression.contains("i / 2") {
            "out[i] = a[i]"
        } else {
            expression
        };
        let text = format!(
            "construct transform[N](a:tile[N] f32,out:tile[N] f32):\n  for i in owned(out):\n    {parsed}\n"
        );
        let mut program = crate::program::compile(
            &[SourceFile {
                path: "projection.seismic.portable".into(),
                scope: Scope::Portable,
                text,
            }],
            &[],
        )
        .unwrap();
        if expression.contains("i / 2") {
            let StmtKind::Owned { body, .. } = &mut program.functions[0].body[0].kind else {
                unreachable!()
            };
            let StmtKind::Assign { value, .. } = &mut body[0].kind else {
                unreachable!()
            };
            let ExprKind::Index { indices, .. } = &mut value.kind else {
                unreachable!()
            };
            let Index::Point(index) = &mut indices[0] else {
                unreachable!()
            };
            let two = Expr {
                kind: ExprKind::Int(2),
                ty: index.ty.clone(),
                sym: Some(Sym::constant(2)),
                span: index.span,
            };
            *index = Expr {
                kind: ExprKind::Binary {
                    op: crate::ast::BinaryOp::Div,
                    lhs: Box::new(index.clone()),
                    rhs: Box::new(two),
                },
                ty: index.ty.clone(),
                sym: index.sym.as_ref().map(|s| s.quot(&Sym::constant(2))),
                span: index.span,
            };
        }
        let span = program.functions[0].vars[0].span;
        let tile = |n| {
            Ty::Tile(Shaped {
                shape: vec![Sym::constant(n)],
                elem: Elem::Dtype(crate::types::DType::F32),
                packed_axis: None,
            })
        };
        let mut vars = vec![
            Var {
                name: "a".into(),
                ty: tile(8),
                kind: VarKind::Local,
                span,
            },
            Var {
                name: "out".into(),
                ty: tile(8),
                kind: VarKind::Local,
                span,
            },
            Var {
                name: "piece".into(),
                ty: tile(4),
                kind: VarKind::Local,
                span,
            },
        ];
        let variable = |v: usize| Expr {
            kind: ExprKind::Var(v),
            ty: vars[v].ty.clone(),
            sym: None,
            span,
        };
        let integer = |n| Expr {
            kind: ExprKind::Int(n),
            ty: Ty::Scalar(crate::types::DType::I32),
            sym: Some(Sym::constant(n)),
            span,
        };
        let call = Expr {
            kind: ExprKind::Call {
                callee: "transform".into(),
                shape_args: vec![Sym::constant(8)],
                elem_args: vec![],
                args: vec![variable(0), variable(1)],
            },
            ty: Ty::Void,
            sym: None,
            span,
        };
        let view = Expr {
            kind: ExprKind::Index {
                base: Box::new(variable(1)),
                indices: vec![Index::Slice {
                    start: Some(integer(2)),
                    end: Some(integer(6)),
                }],
            },
            ty: tile(4),
            sym: None,
            span,
        };
        let target = variable(2);
        super::call(&program, &call, 1, &view, &target, &mut vars)
            .unwrap()
            .is_some()
    }
    #[test]
    fn retained_call_specialization_requires_exact_corresponding_access() {
        assert!(projected("out[i] = a[i] * 2.0"));
        assert!(projected("out[i] = a[i + 0] * 2.0"));
        assert!(!projected("out[i] = a[0]"));
        assert!(!projected("out[i] = reduce(a,0,sum)"));
        assert!(!projected("out[i] = reduce(a[:],0,sum)"));
        assert!(!projected(
            "out[i] = 0.0\n    for k in range(i): out[i] += 1.0"
        ));
        assert!(!projected("out[i] = a[N - 1 - i]"));
        assert!(!projected("out[i] = a[i / 2]"));
        assert!(!projected("out[i] = a[i] * f32(N)"));
        assert!(!projected("a[i] = a[i] + 1.0; out[i] = a[i]"));
        assert!(!projected("out[i] = a[i] + f32(i)"));
    }
}

pub(super) fn effects(program: &Program, call: &Expr) -> Option<(HashSet<VarId>, bool)> {
    let ExprKind::Call { callee, args, .. } = &call.kind else {
        return None;
    };
    let function = program.functions.iter().find(|f| f.name == *callee)?;
    let (vars, body) = portable_body(program, function).ok()?;
    let mut written = HashSet::new();
    for statement in &body {
        crate::rewrite::writes(statement, &mut written);
    }
    fn root(e: &Expr) -> Option<VarId> {
        match &e.kind {
            ExprKind::Var(v) => Some(*v),
            ExprKind::Index { base, .. } | ExprKind::Transpose(base) => root(base),
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            } => root(&args[0]),
            _ => None,
        }
    }
    let mut actual = HashSet::new();
    for id in written {
        if let VarKind::Param(parameter) = vars[id].kind {
            actual.insert(root(args.get(parameter)?)?);
        }
    }
    Some((actual, body.iter().any(crate::effects::tensor_effect)))
}
