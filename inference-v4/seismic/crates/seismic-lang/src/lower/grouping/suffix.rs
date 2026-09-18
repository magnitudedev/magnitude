//! Compact the final independent epilogue without extending the admitted group.
//! Calls and their seeds remain retained; this changes only how their disjoint
//! outputs are consumed. Values captured from between calls are not reconstructed.
use super::*;

pub(super) fn outputs(region: &Region<'_>, body: &[Stmt]) -> Option<Vec<VarId>> {
    let last = region.calls.last()?.call;
    let mut outputs = Vec::new();
    for call in &region.calls {
        let StmtKind::Expr(Expr {
            kind: ExprKind::Call { args, .. },
            ..
        }) = &body[call.call].kind
        else {
            return None;
        };
        let ExprKind::Var(output) = args[call.output].kind else {
            return None;
        };
        if outputs.contains(&output)
            || body[call.call + 1..=last]
                .iter()
                .any(|s| crate::effects::uses(s, output))
        {
            return None;
        }
        outputs.push(output);
    }
    let mut prefix_values = HashSet::new();
    for s in &body[..=last] {
        crate::rewrite::writes(s, &mut prefix_values);
    }
    // Original parallel independence already covers effects and aliasing. This
    // narrower check requires every live value from its prefix to be supplied
    // by a retained call output. All other prefix evaluation is kept intact.
    if prefix_values.iter().any(|v| {
        !outputs.contains(v) && body[last + 1..].iter().any(|s| crate::effects::uses(s, *v))
    }) {
        return None;
    }
    Some(outputs)
}

pub(super) fn materialize(
    region: &Region<'_>,
    source: &[Stmt],
    outputs: &[Expr],
    factors: &[i64],
    bases: &[Sym],
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Vec<Stmt>, String> {
    let mut indices = Vec::new();
    let mut offsets = Vec::new();
    for _ in factors {
        let id = vars.len();
        let atom = Atom::Param(format!("group_epilogue#{id}"));
        vars.push(Var {
            name: format!("group_epilogue_{id}"),
            ty: Ty::Scalar(DType::I32),
            kind: VarKind::Index(atom.clone()),
            span,
        });
        indices.push(id);
        offsets.push(Sym::atom(atom));
    }
    let mut body = Vec::new();
    for (call, output) in region.calls.iter().zip(outputs) {
        let StmtKind::Expr(Expr {
            kind: ExprKind::Call { args, .. },
            ..
        }) = &source[call.call].kind
        else {
            unreachable!()
        };
        let target = &args[call.output];
        let shape = target
            .ty
            .shaped()
            .ok_or("grouped output lost its tile shape")?;
        let mut coordinates = Vec::new();
        for (axis, extent) in shape.shape.iter().enumerate() {
            let start = if let Some(group) = call.axes.iter().position(|a| a.output_axis == axis) {
                offsets[group].scale(
                    extent
                        .as_constant()
                        .ok_or("grouped output needs a static element extent")?,
                )
            } else {
                Sym::constant(0)
            };
            coordinates.push(Index::Slice {
                start: Some(symbol(start.clone(), span)),
                end: Some(symbol(start.add(extent), span)),
            });
        }
        let view = Expr {
            kind: ExprKind::Index {
                base: Box::new(output.clone()),
                indices: coordinates,
            },
            ty: target.ty.clone(),
            sym: None,
            span,
        };
        body.push(assign(
            target.clone(),
            Expr {
                kind: ExprKind::Builtin {
                    name: Builtin::Load,
                    args: vec![view],
                },
                ty: target.ty.clone(),
                sym: None,
                span,
            },
            span,
        ));
    }
    body.extend_from_slice(&source[region.calls.last().unwrap().call + 1..]);
    // Bind a single per-iteration local environment before substituting logical
    // coordinates, including any scalar temporaries and internal casts.
    let (mut body, _) = crate::widen::copy_bindings(&body, vars);
    for ((axis, base), offset) in region.axes.iter().zip(bases).zip(&offsets) {
        crate::widen::replace_index(
            &mut body,
            axis.coordinate,
            &axis.atom,
            &symbol(base.add(offset), span),
        );
    }
    for (&var, &factor) in indices.iter().zip(factors).rev() {
        body = vec![Stmt {
            id: None,
            span,
            kind: StmtKind::Range {
                var,
                lo: Sym::constant(0),
                hi: Sym::constant(factor),
                body,
            },
        }];
    }
    Ok(body)
}
