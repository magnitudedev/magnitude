//! Rectangular independent outputs become one retained construct invocation.
//! Geometry and independence come from the same checked maps used by projection;
//! ordinary tile copies preserve each source seed and publication conversion.
use super::*;
use crate::ast::AssignOp;
use crate::composition::View;
use crate::types::DType;
use std::collections::HashSet;

#[derive(Clone)]
struct Axis {
    coordinate: VarId,
    atom: Atom,
    extent: i64,
    output_axis: usize,
    parameter: String,
}
struct Region<'a> {
    function: &'a Function,
    call: usize,
    output: usize,
    axes: Vec<Axis>,
    loads: Vec<Option<Expr>>,
    removable: HashSet<usize>,
    aliases: Vec<AliasRequirement>,
}

pub(super) fn select(
    ir: &mut LoweredIr,
    program: &Program,
    choose: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    fn visit(
        body: &mut Vec<Stmt>,
        vars: &mut Vec<Var>,
        program: &Program,
        choose: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
        aliases: &mut Vec<AliasRequirement>,
        enclosing: &HashSet<VarId>,
    ) -> Result<(), String> {
        let mut result = Vec::new();
        let mut visible = enclosing.clone();
        for mut s in std::mem::take(body) {
            let mut written = HashSet::new();
            crate::rewrite::writes(&s, &mut written);
            if let StmtKind::Parallel {
                vars: coordinates,
                extents,
                body,
            } = &s.kind
            {
                if let Some(region) = region(program, coordinates, extents, body, vars, &visible)? {
                    // Preserve invocation legality from the source work-item
                    // geometry, including when all items become one group.
                    for requirement in &region.aliases {
                        if let Some(existing) = aliases
                            .iter_mut()
                            .find(|a| a.left == requirement.left && a.right == requirement.right)
                        {
                            existing.exact_allowed &= requirement.exact_allowed;
                        } else {
                            aliases.push(requirement.clone());
                        }
                    }
                    let widths = region
                        .axes
                        .iter()
                        .map(|axis| {
                            let domain = Decision {
                                kind: DecisionKind::OutputGroup {
                                    coordinate: axis.coordinate,
                                    extent: axis.extent,
                                    construct: region.function.name.clone(),
                                    parameter: axis.parameter.clone(),
                                },
                                alternatives: Alternatives::output_widths(axis.extent)?,
                            };
                            let Alternative::OutputWidth(width) = choose(&domain)? else {
                                return Err("output grouping needs an output width".into());
                            };
                            if !domain
                                .alternatives
                                .contains(&Alternative::OutputWidth(width))
                            {
                                return Err("output width outside independent domain".into());
                            }
                            Ok(width)
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    if widths.iter().any(|w| *w > 1) {
                        result.extend(materialize(&region, body, &widths, vars, s.span)?);
                        visible.extend(written);
                        continue;
                    }
                }
            }
            match &mut s.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => {
                    visit(body, vars, program, choose, aliases, &visible)?
                }
                StmtKind::If { then, els, .. } => {
                    visit(then, vars, program, choose, aliases, &visible)?;
                    visit(els, vars, program, choose, aliases, &visible)?;
                }
                _ => {}
            }
            visible.extend(written);
            result.push(s);
        }
        *body = result;
        Ok(())
    }
    visit(
        &mut ir.body,
        &mut ir.vars,
        program,
        choose,
        &mut ir.alias_requirements,
        &(0..ir.params.len()).collect(),
    )
}

fn region<'a>(
    program: &'a Program,
    coordinates: &[VarId],
    extents: &[Sym],
    body: &[Stmt],
    vars: &[Var],
    enclosing: &HashSet<VarId>,
) -> Result<Option<Region<'a>>, String> {
    let calls = body
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.kind {
            StmtKind::Expr(
                e @ Expr {
                    kind: ExprKind::Call { .. },
                    ..
                },
            ) => Some((i, e)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(call_index, call)] = calls.as_slice() else {
        return Ok(None);
    };
    let ExprKind::Call { callee, args, .. } = &call.kind else {
        unreachable!()
    };
    let Some(function) = program
        .functions
        .iter()
        .find(|f| f.is_construct && f.name == *callee)
    else {
        return Ok(None);
    };
    let Some(facts) = projection::output_axes(program, function)? else {
        return Ok(None);
    };
    let Some(output) = args.get(facts.output) else {
        return Ok(None);
    };
    if args.iter().any(|a| {
        matches!(a.ty, Ty::Tensor(_))
            || a.ty.shaped().is_some_and(|s| {
                s.shape
                    .iter()
                    .any(|n| n.as_constant().is_none_or(|n| n <= 0))
            })
    }) {
        return Ok(None);
    }
    let ExprKind::Var(output_id) = output.kind else {
        return Ok(None);
    };
    let Ty::Tile(output_shape) = &output.ty else {
        return Ok(None);
    };
    if output_shape
        .shape
        .iter()
        .any(|n| n.as_constant().is_none_or(|n| n <= 0))
    {
        return Ok(None);
    }
    if args
        .iter()
        .enumerate()
        .any(|(i, e)| i != facts.output && mentions(e, output_id))
    {
        return Ok(None);
    }
    // Local state must be recreated inside each source work item. This excludes
    // escaping state, mutable tensors, and unknown effects around the call.
    let mut local = HashSet::new();
    fn bindings(body: &[Stmt], local: &mut HashSet<VarId>) {
        for s in body {
            match &s.kind {
                StmtKind::Assign {
                    target:
                        Expr {
                            kind: ExprKind::Var(v),
                            ..
                        },
                    ..
                } => {
                    local.insert(*v);
                }
                StmtKind::Owned { body, .. } => bindings(body, local),
                _ => {}
            }
        }
    }
    bindings(body, &mut local);
    if !local.is_disjoint(enclosing) {
        return Ok(None);
    }
    if local.iter().any(|v| {
        !matches!(vars[*v].kind, VarKind::Local)
            || vars[*v]
                .ty
                .shaped()
                .is_some_and(|s| s.shape.iter().any(|n| n.as_constant().is_none()))
    }) {
        return Ok(None);
    }
    if !local.contains(&output_id)
        || body[..*call_index]
            .iter()
            .any(|s| !local_statement(s, &local, false))
        || body[*call_index + 1..]
            .iter()
            .any(|s| !local_statement(s, &local, true))
    {
        return Ok(None);
    }
    let publications = body[*call_index + 1..]
        .iter()
        .filter_map(|s| match &s.kind {
            StmtKind::Expr(Expr {
                kind:
                    ExprKind::Builtin {
                        name: Builtin::Store,
                        args,
                    },
                ..
            }) if args.len() == 2 => Some((&args[0], &args[1])),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(value, destination)] = publications.as_slice() else {
        return Ok(None);
    };
    if value
        .ty
        .shaped()
        .is_none_or(|s| s.shape != output_shape.shape)
    {
        return Ok(None);
    }
    let Some(publication) = View::of(destination) else {
        return Ok(None);
    };
    if publication.visible.len() != output_shape.shape.len() {
        return Ok(None);
    }
    let mut source = body
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != *call_index)
        .map(|(_, s)| s.clone())
        .collect::<Vec<_>>();
    for argument in args.iter().filter(|a| a.ty.shaped().is_none()) {
        source.push(Stmt {
            id: None,
            span: argument.span,
            kind: StmtKind::Expr(argument.clone()),
        });
    }
    let accesses = crate::composition::Accesses::of(&source);
    if accesses.unknown
        || accesses
            .accesses
            .iter()
            .any(|a| !matches!(vars[a.view.root].kind, VarKind::Param(_)))
    {
        return Ok(None);
    }
    let mut aliases = Vec::<AliasRequirement>::new();
    for (i, left) in accesses.accesses.iter().enumerate() {
        for right in &accesses.accesses[i + 1..] {
            if extents.iter().all(|n| n.as_constant() == Some(1)) {
                continue;
            }
            if !left.write && !right.write {
                continue;
            }
            let (VarKind::Param(a), VarKind::Param(b)) =
                (&vars[left.view.root].kind, &vars[right.view.root].kind)
            else {
                unreachable!()
            };
            let exact_allowed = vars[left.view.root].ty == vars[right.view.root].ty
                && left.view.axes == right.view.axes
                && left.view.visible == right.view.visible;
            if a == b {
                if !exact_allowed {
                    return Ok(None);
                }
                continue;
            }
            let (a, b) = ((*a).min(*b), (*a).max(*b));
            if let Some(prior) = aliases.iter_mut().find(|p| p.left == a && p.right == b) {
                prior.exact_allowed &= exact_allowed;
            } else {
                aliases.push(AliasRequirement {
                    left: a,
                    right: b,
                    exact_allowed,
                });
            }
        }
    }
    let mut axes = Vec::new();
    for (&coordinate, extent) in coordinates.iter().zip(extents) {
        let (VarKind::Index(atom), Some(extent)) = (&vars[coordinate].kind, extent.as_constant())
        else {
            return Ok(None);
        };
        if extent <= 0 {
            return Ok(None);
        }
        let mapped = facts
            .dimensions
            .iter()
            .filter(|(axis, _)| {
                let (start, count) = &publication.axes[publication.visible[*axis]];
                *count == output_shape.shape[*axis]
                    && start
                        .linear_in(atom)
                        .is_some_and(|(step, _)| Some(step) == count.as_constant())
            })
            .collect::<Vec<_>>();
        let [(output_axis, parameter)] = mapped.as_slice() else {
            return Ok(None);
        };
        if axes.iter().any(|a: &Axis| a.parameter == *parameter) {
            return Ok(None);
        }
        // A rectangular coordinate changes exactly one physical publication axis.
        if publication
            .axes
            .iter()
            .enumerate()
            .any(|(i, (start, count))| {
                count.atoms().contains(atom)
                    || (i != publication.visible[*output_axis] && start.atoms().contains(atom))
            })
        {
            return Ok(None);
        }
        axes.push(Axis {
            coordinate,
            atom: atom.clone(),
            extent,
            output_axis: *output_axis,
            parameter: parameter.clone(),
        });
    }
    let mut loads = Vec::new();
    let mut removable = HashSet::new();
    for (position, actual) in args.iter().enumerate() {
        if position == facts.output {
            loads.push(None);
            continue;
        }
        for axis in &axes {
            let formal_varies = function.params[position]
                .1
                .shaped()
                .is_some_and(|s| s.shape.iter().any(|n| *n == Sym::param(&axis.parameter)));
            let dependencies =
                crate::widen::plan(&body[..*call_index], axis.coordinate, &axis.atom, 2).wide_vars;
            if !formal_varies && depends(actual, &dependencies, axis.coordinate, &axis.atom) {
                return Ok(None);
            }
        }
        let definition = match actual.kind {
            ExprKind::Var(v) => body[..*call_index]
                .iter()
                .enumerate()
                .find_map(|(i, s)| match &s.kind {
                    StmtKind::Assign {
                        target:
                            Expr {
                                kind: ExprKind::Var(t),
                                ..
                            },
                        value,
                        ..
                    } if *t == v => match &value.kind {
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        } if args.len() == 1 => Some((i, args[0].clone())),
                        ExprKind::Load { view, .. } => Some((i, (**view).clone())),
                        _ => None,
                    },
                    _ => None,
                }),
            _ => None,
        };
        let union = definition.as_ref().and_then(|(definition, view)| {
            let ExprKind::Var(id) = actual.kind else {
                return None;
            };
            if body[*definition + 1..*call_index]
                .iter()
                .any(|s| crate::effects::tile_mutated(s, id))
            {
                return None;
            }
            let geometry = union_view(view, &function.params[position].1, &axes)?;
            // A local view alias must first be resolved in its lexical snapshot
            // environment. Until then keep its dense value through a tile copy.
            if !matches!(vars[geometry.root].kind, VarKind::Param(_)) {
                return None;
            }
            Some(view.clone())
        });
        if actual
            .ty
            .shaped()
            .is_some_and(|s| matches!(s.elem, Elem::Repr(_)))
            && union.is_none()
        {
            return Ok(None);
        }
        if let (Some(_), Some((definition, _)), ExprKind::Var(v)) =
            (&union, &definition, &actual.kind)
        {
            if !body
                .iter()
                .enumerate()
                .any(|(i, s)| i != *definition && i != *call_index && statement_mentions(s, *v))
            {
                removable.insert(*definition);
            }
        }
        loads.push(union);
    }
    Ok(Some(Region {
        function,
        call: *call_index,
        output: facts.output,
        axes,
        loads,
        removable,
        aliases,
    }))
}

fn local_statement(s: &Stmt, local: &HashSet<VarId>, publication: bool) -> bool {
    match &s.kind {
        StmtKind::Assign { target, value, .. } => {
            root(target).is_some_and(|v| local.contains(&v))
                && !matches!(target.ty, Ty::Tensor(_))
                && crate::effects::expression_can_be_omitted(target)
                && crate::effects::expression_can_be_omitted(value)
        }
        StmtKind::Owned { tile, body, .. } => {
            root(tile).is_some_and(|v| local.contains(&v))
                && body.iter().all(|s| local_statement(s, local, false))
        }
        StmtKind::Expr(Expr {
            kind:
                ExprKind::Builtin {
                    name: Builtin::Store,
                    args,
                },
            ..
        }) if publication => args.iter().all(crate::effects::expression_can_be_omitted),
        StmtKind::Expr(e) => crate::effects::expression_can_be_omitted(e),
        _ => false,
    }
}
fn root(e: &Expr) -> Option<VarId> {
    match &e.kind {
        ExprKind::Var(v) => Some(*v),
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => root(base),
        _ => None,
    }
}
fn mentions(e: &Expr, v: VarId) -> bool {
    let mut found = false;
    walk_expr(e, &mut |e| {
        found |= matches!(e.kind,ExprKind::Var(x) if x==v)
    });
    found
}
fn statement_mentions(s: &Stmt, v: VarId) -> bool {
    match &s.kind {
        StmtKind::Assign { target, value, .. } => mentions(target, v) || mentions(value, v),
        StmtKind::Expr(e) => mentions(e, v),
        StmtKind::Owned { tile, body, .. } => {
            mentions(tile, v) || body.iter().any(|s| statement_mentions(s, v))
        }
        _ => true,
    }
}
fn depends(e: &Expr, dependencies: &[VarId], coordinate: VarId, atom: &Atom) -> bool {
    let mut found = false;
    walk_expr(e, &mut |e| {
        found |= e.sym.as_ref().is_some_and(|s| s.atoms().contains(atom))
            || matches!(e.kind,ExprKind::Var(v) if v==coordinate||dependencies.contains(&v));
    });
    found
}

// Prove direct input snapshots tile adjacent rectangles. The normalized map also
// preserves transpose and singleton rank-drop; no tensor root-name conventions.
fn union_view(view: &Expr, formal: &Ty, axes: &[Axis]) -> Option<View> {
    if !crate::effects::expression_can_be_omitted(view) {
        return None;
    }
    let v = View::of(view)?;
    let shaped = formal.shaped()?;
    if shaped.shape.len() != v.visible.len() {
        return None;
    }
    for axis in axes {
        let affected = shaped
            .shape
            .iter()
            .enumerate()
            .filter(|(_, n)| **n == Sym::param(&axis.parameter))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if affected.len() > 1 {
            return None;
        }
        for (i, (start, count)) in v.axes.iter().enumerate() {
            if count.atoms().contains(&axis.atom) {
                return None;
            }
            if affected.first().is_some_and(|a| v.visible[*a] == i) {
                if start.linear_in(&axis.atom)?.0 != count.as_constant()? {
                    return None;
                }
            } else if start.atoms().contains(&axis.atom) {
                return None;
            }
        }
    }
    Some(v)
}

fn materialize(
    region: &Region<'_>,
    body: &[Stmt],
    widths: &[i64],
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Vec<Stmt>, String> {
    // Each combination is a disjoint full/tail rectangle with static call shapes.
    let mut rectangles = vec![Vec::<(i64, i64, i64)>::new()]; // group count, block width, source offset
    for (axis, &width) in region.axes.iter().zip(widths) {
        let mut next = Vec::new();
        for prefix in rectangles {
            let mut full = prefix.clone();
            full.push((axis.extent / width, width, 0));
            next.push(full);
            if axis.extent % width != 0 {
                let mut tail = prefix;
                tail.push((1, axis.extent % width, axis.extent / width * width));
                next.push(tail);
            }
        }
        rectangles = next;
    }
    let mut result = Vec::new();
    for rectangle in rectangles {
        let mut coordinates = Vec::new();
        let mut bases = Vec::new();
        let mut group_extents = Vec::new();
        let factors = rectangle.iter().map(|r| r.1).collect::<Vec<_>>();
        for (count, width, offset) in &rectangle {
            let id = vars.len();
            let atom = Atom::Param(format!("output_group#{id}"));
            vars.push(Var {
                name: format!("output_group_{id}"),
                ty: Ty::Scalar(DType::I32),
                span,
                kind: VarKind::Index(atom.clone()),
            });
            coordinates.push(id);
            group_extents.push(Sym::constant(*count));
            bases.push(Sym::atom(atom).scale(*width).add(&Sym::constant(*offset)));
        }
        let grouped = rectangle_body(region, body, &factors, &bases, vars, span)?;
        result.push(Stmt {
            id: None,
            span,
            kind: StmtKind::Parallel {
                vars: coordinates,
                extents: group_extents,
                body: grouped,
            },
        });
    }
    Ok(result)
}

fn rectangle_body(
    region: &Region<'_>,
    body: &[Stmt],
    factors: &[i64],
    bases: &[Sym],
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Vec<Stmt>, String> {
    let ExprKind::Call {
        callee,
        shape_args,
        elem_args,
        args,
    } = &match &body[region.call].kind {
        StmtKind::Expr(e) => e,
        _ => unreachable!(),
    }
    .kind
    else {
        unreachable!()
    };
    let mut slots = vec![Vec::<i64>::new()];
    for &factor in factors {
        slots = slots
            .into_iter()
            .flat_map(|p| {
                (0..factor).map(move |n| {
                    let mut q = p.clone();
                    q.push(n);
                    q
                })
            })
            .collect();
    }
    let filtered = body
        .iter()
        .enumerate()
        .filter(|(i, _)| !region.removable.contains(i))
        .map(|(_, s)| s.clone())
        .collect::<Vec<_>>();
    let at = region.call - region.removable.len();
    let mut prefixes = Vec::new();
    let mut suffixes = Vec::new();
    let mut arguments = Vec::new();
    for slot in &slots {
        let (mut copy, _) = crate::widen::copy_bindings(&filtered, vars);
        for ((axis, base), offset) in region.axes.iter().zip(bases).zip(slot) {
            crate::widen::replace_index(
                &mut copy,
                axis.coordinate,
                &axis.atom,
                &symbol(base.add(&Sym::constant(*offset)), span),
            );
        }
        let StmtKind::Expr(Expr {
            kind: ExprKind::Call { args, .. },
            ..
        }) = &copy[at].kind
        else {
            return Err("group call lost during binding copy".into());
        };
        arguments.push(args.clone());
        prefixes.extend(copy[..at].iter().cloned());
        suffixes.push(copy[at + 1..].to_vec());
    }
    let mut combined = Vec::new();
    let mut seed = Vec::new();
    for (position, actual) in args.iter().enumerate() {
        let Some(formal) = region.function.params[position].1.shaped() else {
            combined.push(arguments[0][position].clone());
            continue;
        };
        let mapping = formal
            .shape
            .iter()
            .map(|n| {
                region
                    .axes
                    .iter()
                    .position(|a| *n == Sym::param(&a.parameter))
            })
            .collect::<Vec<_>>();
        let mut shape = actual
            .ty
            .shaped()
            .ok_or("group operand lost shape")?
            .clone();
        for (n, a) in shape.shape.iter_mut().zip(&mapping) {
            if let Some(a) = a {
                *n = n.scale(factors[*a]);
            }
        }
        let ty = Ty::Tile(shape.clone());
        let target = local_var("group_operand", ty.clone(), vars, span);
        if let Some(source) = &region.loads[position] {
            let mut v = union_view(source, &region.function.params[position].1, &region.axes)
                .ok_or("group operand geometry changed")?;
            for (start, count) in &mut v.axes {
                for (axis, base) in region.axes.iter().zip(bases) {
                    *start = start.subst(&axis.atom, base);
                    *count = count.subst(&axis.atom, base);
                }
            }
            for (axis, mapped) in mapping.iter().enumerate() {
                if let Some(group) = mapped {
                    let root = v.visible[axis];
                    v.axes[root].1 = v.axes[root].1.scale(factors[*group]);
                }
            }
            let view = geometry(&v, &source.ty, vars, span)?;
            seed.push(assign(
                target.clone(),
                Expr {
                    kind: ExprKind::Builtin {
                        name: Builtin::Load,
                        args: vec![view],
                    },
                    ty,
                    sym: None,
                    span,
                },
                span,
            ));
        } else {
            seed.push(assign(
                target.clone(),
                Expr {
                    kind: ExprKind::TileAlloc {
                        shape: shape.shape.clone(),
                        dtype: shape.elem.clone(),
                    },
                    ty,
                    sym: None,
                    span,
                },
                span,
            ));
            let mut copied = HashSet::new();
            for (slot, arguments) in slots.iter().zip(&arguments) {
                let offset = mapping
                    .iter()
                    .enumerate()
                    .map(|(axis, g)| {
                        g.map_or(0, |g| {
                            slot[g]
                                * actual.ty.shaped().unwrap().shape[axis]
                                    .as_constant()
                                    .unwrap()
                        })
                    })
                    .collect::<Vec<_>>();
                if copied.insert(offset.clone()) {
                    seed.push(copy_tile(
                        &arguments[position],
                        &target,
                        &offset,
                        false,
                        vars,
                        span,
                    )?);
                }
            }
        }
        combined.push(target);
    }
    prefixes.extend(seed);
    let shapes = region
        .function
        .shape_params
        .iter()
        .zip(shape_args)
        .map(|(p, n)| {
            region
                .axes
                .iter()
                .position(|a| a.parameter == *p)
                .map_or_else(|| n.clone(), |a| n.scale(factors[a]))
        })
        .collect();
    prefixes.push(Stmt {
        id: None,
        span,
        kind: StmtKind::Expr(Expr {
            kind: ExprKind::Call {
                callee: callee.clone(),
                shape_args: shapes,
                elem_args: elem_args.clone(),
                args: combined.clone(),
            },
            ty: Ty::Void,
            sym: None,
            span,
        }),
    });
    for ((slot, arguments), suffix) in slots.iter().zip(&arguments).zip(suffixes) {
        let output = &arguments[region.output];
        let mut offset = vec![0; output.ty.shaped().unwrap().shape.len()];
        for (axis, index) in region.axes.iter().zip(slot) {
            offset[axis.output_axis] = index
                * output.ty.shaped().unwrap().shape[axis.output_axis]
                    .as_constant()
                    .unwrap();
        }
        prefixes.push(copy_tile(
            output,
            &combined[region.output],
            &offset,
            true,
            vars,
            span,
        )?);
        prefixes.extend(suffix);
    }
    Ok(prefixes)
}
fn geometry(
    view: &View,
    original: &Ty,
    vars: &[Var],
    span: crate::span::Span,
) -> Result<Expr, String> {
    let root = variable(view.root, vars);
    let mut shaped = root.ty.shaped().unwrap().clone();
    let mut indices = Vec::new();
    for (axis, (start, count)) in view.axes.iter().enumerate() {
        indices.push(if view.visible.contains(&axis) {
            Index::Slice {
                start: Some(symbol(start.clone(), span)),
                end: Some(symbol(start.add(count), span)),
            }
        } else {
            Index::Point(symbol(start.clone(), span))
        });
    }
    let physical = (0..view.axes.len())
        .filter(|a| view.visible.contains(a))
        .collect::<Vec<_>>();
    shaped.shape = physical.iter().map(|a| view.axes[*a].1.clone()).collect();
    shaped.packed_axis = shaped
        .packed_axis
        .and_then(|a| physical.iter().position(|p| *p == a));
    let tensor = matches!(original, Ty::Tensor(_));
    let mut expression = Expr {
        kind: ExprKind::Index {
            base: Box::new(root),
            indices,
        },
        ty: if tensor {
            Ty::Tensor(shaped.clone())
        } else {
            Ty::Tile(shaped.clone())
        },
        sym: None,
        span,
    };
    if physical != view.visible {
        if physical.iter().rev().copied().collect::<Vec<_>>() != view.visible {
            return Err("unsupported output operand permutation".into());
        }
        shaped.shape.reverse();
        shaped.packed_axis = shaped.packed_axis.map(|a| shaped.shape.len() - 1 - a);
        expression = Expr {
            kind: ExprKind::Transpose(Box::new(expression)),
            ty: if tensor {
                Ty::Tensor(shaped)
            } else {
                Ty::Tile(shaped)
            },
            sym: None,
            span,
        };
    }
    Ok(expression)
}
fn copy_tile(
    small: &Expr,
    large: &Expr,
    offset: &[i64],
    scatter: bool,
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Stmt, String> {
    let shape = small.ty.shaped().ok_or("group copy requires tile")?;
    let mut coordinates = Vec::new();
    let mut local = Vec::new();
    let mut global = Vec::new();
    for &offset in offset {
        let id = vars.len();
        let atom = Atom::Param(format!("group_element#{id}"));
        vars.push(Var {
            name: format!("group_element_{id}"),
            ty: Ty::Scalar(DType::I32),
            kind: VarKind::Index(atom.clone()),
            span,
        });
        coordinates.push(id);
        local.push(Index::Point(symbol(Sym::atom(atom.clone()), span)));
        global.push(Index::Point(symbol(
            Sym::atom(atom).add(&Sym::constant(offset)),
            span,
        )));
    }
    let scalar = Ty::Scalar(shape.elem.read_dtype().ok_or("unbound grouped element")?);
    let small = Expr {
        kind: ExprKind::Index {
            base: Box::new(small.clone()),
            indices: local,
        },
        ty: scalar.clone(),
        sym: None,
        span,
    };
    let large = Expr {
        kind: ExprKind::Index {
            base: Box::new(large.clone()),
            indices: global,
        },
        ty: scalar,
        sym: None,
        span,
    };
    Ok(Stmt {
        id: None,
        span,
        kind: StmtKind::Owned {
            vars: coordinates,
            tile: match &small.kind {
                ExprKind::Index { base, .. } => (**base).clone(),
                _ => unreachable!(),
            },
            body: vec![if scatter {
                assign(small, large, span)
            } else {
                assign(large, small, span)
            }],
        },
    })
}
fn variable(id: VarId, vars: &[Var]) -> Expr {
    Expr {
        kind: ExprKind::Var(id),
        ty: vars[id].ty.clone(),
        sym: match &vars[id].kind {
            VarKind::Index(a) => Some(Sym::atom(a.clone())),
            _ => None,
        },
        span: vars[id].span,
    }
}
fn local_var(name: &str, ty: Ty, vars: &mut Vec<Var>, span: crate::span::Span) -> Expr {
    let id = vars.len();
    vars.push(Var {
        name: format!("{name}_{id}"),
        ty,
        span,
        kind: VarKind::Local,
    });
    variable(id, vars)
}
fn symbol(sym: Sym, span: crate::span::Span) -> Expr {
    Expr {
        kind: ExprKind::ShapeParam(sym.to_string()),
        ty: Ty::Scalar(DType::I32),
        sym: Some(sym),
        span,
    }
}
fn assign(target: Expr, value: Expr, span: crate::span::Span) -> Stmt {
    Stmt {
        id: None,
        span,
        kind: StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        },
    }
}
