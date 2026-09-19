//! Rectangular independent outputs widen an ordered sequence of retained calls.
//! Geometry and independence come from the same checked maps used by projection;
//! ordinary tile copies preserve each source seed and publication conversion.
use super::*;
mod suffix;
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
struct CallRegion<'a> {
    function: &'a Function,
    call: usize,
    output: usize,
    axes: Vec<Axis>,
    loads: Vec<Option<Expr>>,
    versions: Vec<Option<usize>>,
}
struct Region<'a> {
    calls: Vec<CallRegion<'a>>,
    axes: Vec<Axis>,
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
                        .enumerate()
                        .map(|(dimension, axis)| {
                            let domain = Decision {
                                kind: DecisionKind::OutputGroup {
                                    coordinate: axis.coordinate,
                                    extent: axis.extent,
                                    calls: region
                                        .calls
                                        .iter()
                                        .map(|call| OutputGroupCall {
                                            position: call.call,
                                            construct: call.function.name.clone(),
                                            parameter: call.axes[dimension].parameter.clone(),
                                        })
                                        .collect(),
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
                        let compact = if let Some(outputs) = suffix::outputs(&region, body) {
                            let decision = Decision {
                                kind: DecisionKind::GroupEpilogue { outputs },
                                alternatives: vec![Alternative::GroupEpilogue(crate::lowered_ir::GroupEpilogue::Unrolled), Alternative::GroupEpilogue(crate::lowered_ir::GroupEpilogue::Serial)].into(),
                            };
                            let selected = choose(&decision)?;
                            if !decision.alternatives.contains(&selected) { return Err("invalid grouped epilogue construction".into()); }
                            selected == Alternative::GroupEpilogue(crate::lowered_ir::GroupEpilogue::Serial)
                        } else { false };
                        let mut grouped = materialize(&region, body, &widths, compact, vars, s.span)?;
                        if grouped.len() > 1 && concatenated_extent(&grouped).is_some() {
                            let domains = grouped
                                .iter()
                                .map(|s| match &s.kind {
                                    StmtKind::Parallel { extents, .. } => extents.clone(),
                                    _ => unreachable!(),
                                })
                                .collect();
                            let decision = Decision {
                                kind: DecisionKind::OutputRemainders { domains },
                                alternatives: vec![Alternative::Separate, Alternative::Concatenate]
                                    .into(),
                            };
                            match choose(&decision)? {
                                Alternative::Separate => {}
                                Alternative::Concatenate => {
                                    grouped = vec![concatenate(grouped, vars, s.span)?]
                                }
                                _ => return Err("invalid grouped remainder mapping".into()),
                            }
                        }
                        result.extend(grouped);
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
        .filter_map(|(position, statement)| match &statement.kind {
            StmtKind::Expr(
                call @ Expr {
                    kind: ExprKind::Call { .. },
                    ..
                },
            ) => Some((position, call)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let Some((last_call, _)) = calls.last() else {
        return Ok(None);
    };
    let positions = calls
        .iter()
        .map(|(position, _)| *position)
        .collect::<HashSet<_>>();
    let mut local = HashSet::new();
    fn bindings(body: &[Stmt], local: &mut HashSet<VarId>) {
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
                    local.insert(*v);
                }
                StmtKind::Owned { body, .. } => bindings(body, local),
                _ => {}
            }
        }
    }
    bindings(body, &mut local);
    if !local.is_disjoint(enclosing)
        || local.iter().any(|v| {
            !matches!(vars[*v].kind, VarKind::Local)
                || vars[*v]
                    .ty
                    .shaped()
                    .is_some_and(|s| s.shape.iter().any(|n| n.as_constant().is_none()))
        })
        || body.iter().enumerate().any(|(position, statement)| {
            !positions.contains(&position)
                && !local_statement(statement, &local, position > *last_call)
        })
    {
        return Ok(None);
    }
    let publications = body[*last_call + 1..]
        .iter()
        .filter_map(|statement| match &statement.kind {
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
    let Some(publication) = View::of(destination) else {
        return Ok(None);
    };

    // Retained calls have checked write effects. The ordinary syntactic write
    // collector cannot see their output until their bodies have been resolved.
    let mut writes = Vec::new();
    for statement in body {
        let mut written = HashSet::new();
        crate::rewrite::writes(statement, &mut written);
        if let StmtKind::Expr(
            call @ Expr {
                kind: ExprKind::Call { .. },
                ..
            },
        ) = &statement.kind
        {
            let Some((call_writes, external)) = call_effects(program, call) else {
                return Ok(None);
            };
            if external || !call_writes.is_subset(&local) {
                return Ok(None);
            }
            written.extend(call_writes);
        }
        writes.push(written);
    }
    let mut source = body
        .iter()
        .enumerate()
        .filter(|(position, _)| !positions.contains(position))
        .map(|(_, statement)| statement.clone())
        .collect::<Vec<_>>();
    for (_, call) in &calls {
        let ExprKind::Call { args, .. } = &call.kind else {
            unreachable!()
        };
        for argument in args
            .iter()
            .filter(|argument| argument.ty.shaped().is_none())
        {
            source.push(Stmt {
                id: None,
                span: argument.span,
                kind: StmtKind::Expr(argument.clone()),
            });
        }
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

    let mut regions = Vec::new();
    for (call_index, call) in &calls {
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
        let ExprKind::Var(output_id) = output.kind else {
            return Ok(None);
        };
        let Ty::Tile(output_shape) = &output.ty else {
            return Ok(None);
        };
        if !local.contains(&output_id)
            || writes[*call_index] != HashSet::from([output_id])
            || args.iter().any(|argument| {
                matches!(argument.ty, Ty::Tensor(_))
                    || !crate::effects::expression_can_be_omitted(argument)
                    || argument.ty.shaped().is_some_and(|s| {
                        s.shape
                            .iter()
                            .any(|n| n.as_constant().is_none_or(|n| n <= 0))
                    })
            })
            || args.iter().enumerate().any(|(position, argument)| {
                position != facts.output && mentions(argument, output_id)
            })
            || value
                .ty
                .shaped()
                .is_none_or(|s| s.shape != output_shape.shape)
            || publication.visible.len() != output_shape.shape.len()
        {
            return Ok(None);
        }
        let mut axes = Vec::new();
        for (&coordinate, extent) in coordinates.iter().zip(extents) {
            let (VarKind::Index(atom), Some(extent)) =
                (&vars[coordinate].kind, extent.as_constant())
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

        // Each source slot initializes one rectangular block. Repeating a
        // grouped parameter on multiple formal axes would expose cross-slot
        // blocks that the gather never initialized, including output seeds.
        if function.params.iter().any(|(_, ty)| {
            ty.shaped().is_some_and(|shape| {
                axes.iter().any(|axis| {
                    shape
                        .shape
                        .iter()
                        .filter(|n| **n == Sym::param(&axis.parameter))
                        .count()
                        > 1
                })
            })
        }) {
            return Ok(None);
        }

        let mut loads = Vec::new();
        let mut versions = Vec::new();
        for (position, actual) in args.iter().enumerate() {
            let version = root(actual).and_then(|variable| {
                writes[..*call_index]
                    .iter()
                    .rposition(|written| written.contains(&variable))
            });
            versions.push(version);
            if position == facts.output {
                loads.push(None);
                continue;
            }
            for axis in &axes {
                let formal_varies = function.params[position]
                    .1
                    .shaped()
                    .is_some_and(|s| s.shape.iter().any(|n| *n == Sym::param(&axis.parameter)));
                let dependencies = crate::widen::plan_with_writes(
                    &body[..*call_index],
                    axis.coordinate,
                    &axis.atom,
                    2,
                    &writes[..*call_index],
                )
                .wide_vars;
                if !formal_varies && depends(actual, &dependencies, axis.coordinate, &axis.atom) {
                    return Ok(None);
                }
            }
            let load = version
                .and_then(|definition| match (&actual.kind, &body[definition].kind) {
                    (
                        ExprKind::Var(v),
                        StmtKind::Assign {
                            target:
                                Expr {
                                    kind: ExprKind::Var(t),
                                    ..
                                },
                            value,
                            op: AssignOp::Assign,
                        },
                    ) if v == t => match &value.kind {
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        } if args.len() == 1 => Some(args[0].clone()),
                        ExprKind::Load { view, .. } => Some((**view).clone()),
                        _ => None,
                    },
                    _ => None,
                })
                .filter(|view| {
                    union_view(view, &function.params[position].1, &axes).is_some_and(|geometry| {
                        matches!(vars[geometry.root].kind, VarKind::Param(_))
                    })
                });
            if actual
                .ty
                .shaped()
                .is_some_and(|s| matches!(s.elem, Elem::Repr(_)))
                && load.is_none()
            {
                return Ok(None);
            }
            loads.push(load);
        }
        regions.push(CallRegion {
            function,
            call: *call_index,
            output: facts.output,
            axes,
            loads,
            versions,
        });
    }
    // Remove a direct snapshot only when every use is supplied by its proven
    // widened load. A later seed, local computation or publication keeps it.
    let mut removable = HashSet::new();
    for call in &regions {
        let ExprKind::Call { args, .. } = &calls
            .iter()
            .find(|(position, _)| *position == call.call)
            .unwrap()
            .1
            .kind
        else {
            unreachable!()
        };
        for (argument, (load, version)) in args.iter().zip(call.loads.iter().zip(&call.versions)) {
            let (ExprKind::Var(variable), Some(_), Some(definition)) =
                (&argument.kind, load, version)
            else {
                continue;
            };
            if body.iter().enumerate().all(|(position, statement)| {
                if position == *definition {
                    return true;
                }
                if let Some(call) = regions.iter().find(|call| call.call == position) {
                    let StmtKind::Expr(Expr {
                        kind: ExprKind::Call { args, .. },
                        ..
                    }) = &statement.kind
                    else {
                        unreachable!()
                    };
                    return args.iter().zip(&call.loads).all(|(argument, load)| {
                        !mentions(argument, *variable)
                            || matches!(argument.kind, ExprKind::Var(v) if v == *variable)
                                && load.is_some()
                    });
                }
                !statement_mentions(statement, *variable)
            }) {
                removable.insert(*definition);
            }
        }
    }
    Ok(Some(Region {
        axes: regions[0].axes.clone(),
        calls: regions,
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
    compact: bool,
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
        let grouped = rectangle_body(region, body, &factors, &bases, compact, vars, span)?;
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

/// All rectangles come from one already-proved independent source parallel
/// region. Concatenation changes only their work-item numbering; it does not
/// establish a new memory independence or alias assumption between source ops.
fn concatenated_extent(rectangles: &[Stmt]) -> Option<i64> {
    let total = rectangles.iter().try_fold(0i64, |sum, s| {
        let StmtKind::Parallel { extents, .. } = &s.kind else {
            return None;
        };
        let count = extents.iter().try_fold(1i64, |product, n| {
            let n = n.as_constant().filter(|&n| n > 0)?;
            product.checked_mul(n)
        })?;
        sum.checked_add(count)
    })?;
    (total <= i64::from(i32::MAX)).then_some(total)
}
fn concatenate(
    rectangles: Vec<Stmt>,
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Stmt, String> {
    let total = concatenated_extent(&rectangles)
        .ok_or("grouped remainder domain exceeds index capacity")?;
    let id = vars.len();
    let atom = Atom::Param(format!("output_region#{id}"));
    vars.push(Var {
        name: format!("output_region_{id}"),
        ty: Ty::Scalar(DType::I32),
        span,
        kind: VarKind::Index(atom.clone()),
    });
    let ordinal = Sym::atom(atom);
    let mut branches = Vec::new();
    let mut offset = 0;
    for rectangle in rectangles {
        let StmtKind::Parallel {
            vars: coordinates,
            extents,
            mut body,
        } = rectangle.kind
        else {
            unreachable!()
        };
        let local = ordinal.sub(&Sym::constant(offset));
        let mut stride = 1i64;
        for (&coordinate, extent) in coordinates.iter().zip(&extents).rev() {
            let extent = extent
                .as_constant()
                .ok_or("nonconstant grouped rectangle")?;
            let value = local
                .quot(&Sym::constant(stride))
                .rem(&Sym::constant(extent));
            let VarKind::Index(atom) = &vars[coordinate].kind else {
                return Err("grouped rectangle has no coordinate identity".into());
            };
            crate::widen::replace_index(&mut body, coordinate, atom, &symbol(value, span));
            stride = stride
                .checked_mul(extent)
                .ok_or("grouped rectangle extent overflow")?;
        }
        offset = offset
            .checked_add(stride)
            .ok_or("grouped remainder extent overflow")?;
        branches.push((offset, body));
    }
    let (_, mut body) = branches.pop().ok_or("empty grouped remainder mapping")?;
    for (end, then) in branches.into_iter().rev() {
        body = vec![Stmt {
            id: None,
            span,
            kind: StmtKind::If {
                cond: Expr {
                    kind: ExprKind::Binary {
                        op: crate::ast::BinaryOp::Lt,
                        lhs: Box::new(variable(id, vars)),
                        rhs: Box::new(symbol(Sym::constant(end), span)),
                    },
                    ty: Ty::Scalar(DType::Bool),
                    sym: None,
                    span,
                },
                then,
                els: body,
            },
        }];
    }
    Ok(Stmt {
        id: None,
        span,
        kind: StmtKind::Parallel {
            vars: vec![id],
            extents: vec![Sym::constant(total)],
            body,
        },
    })
}

fn rectangle_body(
    region: &Region<'_>,
    body: &[Stmt],
    factors: &[i64],
    bases: &[Sym],
    compact: bool,
    vars: &mut Vec<Var>,
    span: crate::span::Span,
) -> Result<Vec<Stmt>, String> {
    let mut slots = vec![Vec::<i64>::new()];
    for &factor in factors {
        slots = slots
            .into_iter()
            .flat_map(|prefix| {
                (0..factor).map(move |offset| {
                    let mut slot = prefix.clone();
                    slot.push(offset);
                    slot
                })
            })
            .collect();
    }
    let end = if compact { region.calls.last().unwrap().call + 1 } else { body.len() };
    let filtered = body[..end]
        .iter()
        .enumerate()
        .filter(|(position, _)| !region.removable.contains(position))
        .map(|(_, statement)| statement.clone())
        .collect::<Vec<_>>();
    // Each slot has one persistent local environment across all call stages.
    // Scattering immediately after a call exposes its writes to the next stage.
    let mut copies = Vec::new();
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
        copies.push(copy);
    }
    let mut result = Vec::new();
    let mut cursor = 0;
    // Reuse only an identical reaching snapshot with identical group axes.
    // Checked call writes participate in the version recorded at each use.
    let mut shared = Vec::<(VarId, Option<usize>, Vec<Option<usize>>, Expr)>::new();
    let mut outputs = Vec::new();
    for call in &region.calls {
        let StmtKind::Expr(Expr {
            kind:
                ExprKind::Call {
                    callee,
                    shape_args,
                    elem_args,
                    args,
                },
            ..
        }) = &body[call.call].kind
        else {
            unreachable!()
        };
        let at = call.call
            - region
                .removable
                .iter()
                .filter(|position| **position < call.call)
                .count();
        let mut arguments = Vec::new();
        for copy in &copies {
            result.extend(copy[cursor..at].iter().cloned());
            let StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) = &copy[at].kind
            else {
                return Err("group call lost during binding copy".into());
            };
            arguments.push(args.clone());
        }
        let mut combined = Vec::new();
        let mut seed = Vec::new();
        for (position, actual) in args.iter().enumerate() {
            let Some(formal) = call.function.params[position].1.shaped() else {
                combined.push(arguments[0][position].clone());
                continue;
            };
            let mapping = formal
                .shape
                .iter()
                .map(|n| {
                    call.axes
                        .iter()
                        .position(|a| *n == Sym::param(&a.parameter))
                })
                .collect::<Vec<_>>();
            let key = match actual.kind {
                ExprKind::Var(variable) if position != call.output => {
                    Some((variable, call.versions[position], mapping.clone()))
                }
                _ => None,
            };
            if let Some((variable, version, mapping)) = &key {
                if let Some((_, _, _, value)) = shared.iter().find(|(v, epoch, axes, _)| {
                    v == variable && epoch == version && axes == mapping
                }) {
                    combined.push(value.clone());
                    continue;
                }
            }
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
            if let Some(source) = &call.loads[position] {
                let mut v = union_view(source, &call.function.params[position].1, &call.axes)
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
            if let Some((variable, version, mapping)) = key {
                shared.push((variable, version, mapping, target.clone()));
            }
            combined.push(target);
        }

        result.extend(seed);
        let shapes = call
            .function
            .shape_params
            .iter()
            .zip(shape_args)
            .map(|(parameter, extent)| {
                call.axes
                    .iter()
                    .position(|axis| axis.parameter == *parameter)
                    .map_or_else(|| extent.clone(), |axis| extent.scale(factors[axis]))
            })
            .collect();
        result.push(Stmt {
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
        if compact { outputs.push(combined[call.output].clone()); }
        for (slot, arguments) in slots.iter().zip(&arguments).filter(|_| !compact) {
            let output = &arguments[call.output];
            let mut offset = vec![0; output.ty.shaped().unwrap().shape.len()];
            for (axis, index) in call.axes.iter().zip(slot) {
                offset[axis.output_axis] = index
                    * output.ty.shaped().unwrap().shape[axis.output_axis]
                        .as_constant()
                        .unwrap();
            }
            result.push(copy_tile(
                output,
                &combined[call.output],
                &offset,
                true,
                vars,
                span,
            )?);
        }
        cursor = at + 1;
    }
    if compact {
        result.extend(suffix::materialize(region, body, &outputs, factors, bases, vars, span)?);
    } else {
        for copy in copies { result.extend(copy[cursor..].iter().cloned()); }
    }
    Ok(result)
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
    for offset in offset {
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
            Sym::atom(atom).add(&Sym::constant(*offset)),
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
