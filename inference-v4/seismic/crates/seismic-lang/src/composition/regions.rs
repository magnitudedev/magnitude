//! Project an existing pure producer region onto a demanded rectangular domain.
//! Statements remain the computation: this analysis only restricts independent
//! output coordinates and renames their local storage. Reaching-value and motion
//! legality belong to the caller's existing producer analysis.
use super::*;

/// Call specialization is supplied by lowering, which owns the checked callee
/// semantics. Returning a body certifies that this output restriction preserves
/// all other observable effects; the region analysis never guesses call effects.
pub(super) type ProjectCall<'a> =
    dyn FnMut(&Expr, VarId, &Expr, &Expr, &mut Vec<Var>) -> Result<Option<Vec<Stmt>>, String> + 'a;

pub(super) fn project(
    region: &[Stmt],
    output: VarId,
    view: &Expr,
    target: &Expr,
    vars: &mut Vec<Var>,
    project_call: &mut ProjectCall<'_>,
) -> Result<Option<Vec<Stmt>>, String> {
    let Some(domain) = View::of(view).filter(|v| v.root == output) else {
        return Ok(None);
    };
    let Some(result) = target.ty.shaped() else {
        return Ok(None);
    };
    if !matches!(target.ty, Ty::Tile(_))
        || view.ty != target.ty
        || domain.visible.len() != result.shape.len()
        || domain
            .visible
            .iter()
            .zip(&result.shape)
            .any(|(&axis, n)| domain.axes[axis].1 != *n)
    {
        return Ok(None);
    }
    let rectangular = Ty::Tile(crate::types::Shaped::new(
        domain.axes.iter().map(|(_, n)| n.clone()).collect(),
        result.elem.clone(),
    ));
    let canonical = Expr {
        kind: ExprKind::Index {
            base: Box::new(variable(output, vars)),
            indices: domain
                .axes
                .iter()
                .map(|(start, n)| Index::Slice {
                    start: Some(symbol(start.clone(), view.span)),
                    end: Some(symbol(start.add(n), view.span)),
                })
                .collect(),
        },
        ty: rectangular.clone(),
        sym: None,
        span: view.span,
    };
    if domain.visible.iter().copied().eq(0..domain.axes.len()) {
        return project_rectangle(region, output, &canonical, target, vars, project_call);
    }
    // Keep the producer's rectangular computation and call shapes in source
    // axis order. A final ordinary value view restores the requested logical
    // order; removed point axes have extent one, so storage stays bounded by
    // exactly the demanded element count.
    let mut ordered = domain.visible.clone();
    ordered.sort_unstable();
    let reversed = ordered
        .iter()
        .rev()
        .copied()
        .eq(domain.visible.iter().copied());
    if domain.visible != ordered && !(reversed && ordered.len() == 2) {
        return Ok(None);
    }
    let checkpoint = vars.len();
    let temporary = fresh(vars, "projected_rectangle", rectangular, target.span);
    let temporary = variable(temporary, vars);
    let result = project_rectangle(region, output, &canonical, &temporary, vars, project_call);
    match result {
        Ok(Some(mut body)) => {
            let mut value = Expr {
                kind: ExprKind::Index {
                    base: Box::new(temporary),
                    indices: (0..domain.axes.len())
                        .map(|axis| {
                            if domain.visible.contains(&axis) {
                                Index::Slice {
                                    start: None,
                                    end: None,
                                }
                            } else {
                                Index::Point(symbol(Sym::constant(0), target.span))
                            }
                        })
                        .collect(),
                },
                ty: Ty::Tile(crate::types::Shaped::new(
                    ordered
                        .iter()
                        .map(|&axis| domain.axes[axis].1.clone())
                        .collect(),
                    target.ty.shaped().unwrap().elem.clone(),
                )),
                sym: None,
                span: target.span,
            };
            if domain.visible != ordered {
                value = Expr {
                    kind: ExprKind::Transpose(Box::new(value)),
                    ty: target.ty.clone(),
                    sym: None,
                    span: target.span,
                };
            }
            body.push(Stmt {
                id: None,
                span: target.span,
                kind: StmtKind::Assign {
                    target: target.clone(),
                    op: AssignOp::Assign,
                    value,
                },
            });
            Ok(Some(body))
        }
        other => {
            vars.truncate(checkpoint);
            other
        }
    }
}

fn project_rectangle(
    region: &[Stmt],
    output: VarId,
    view: &Expr,
    target: &Expr,
    vars: &mut Vec<Var>,
    project_call: &mut ProjectCall<'_>,
) -> Result<Option<Vec<Stmt>>, String> {
    let Some(starts) = admissible(region, output, view, target, vars) else {
        return Ok(None);
    };
    let checkpoint = vars.len();
    let result = (|| {
        let mut local_ids = HashSet::new();
        definitions(region, &mut local_ids);
        local_ids.remove(&output);
        let mut local_ids = local_ids.into_iter().collect::<Vec<_>>();
        local_ids.sort_unstable();
        let mut rename = HashMap::new();
        for old in local_ids {
            let id = vars.len();
            let mut v = vars[old].clone();
            v.name = format!("{}_projected_{id}", v.name);
            if matches!(v.kind, VarKind::Index(_)) {
                v.kind = VarKind::Index(Atom::Param(format!("projected#{id}")));
            }
            vars.push(v);
            rename.insert(old, id);
        }
        let mut atoms = index_substitutions(&rename, vars);
        let mut replacements = HashMap::new();
        let mut out = region.to_vec();
        // Stream extents are lexical binders too, though they have no VarId.
        // Keep the copied loop's binder and every dependent tile type together.
        let mut pieces = HashMap::new();
        for statement in &mut out {
            rename_pieces(statement, checkpoint, &mut pieces);
        }
        atoms.extend(pieces.into_iter().map(|(old, new)| (old, Sym::atom(new))));
        // Preserve the containing algorithmic loops and branch order. Only the
        // independent output coordinates are restricted to the requested view.
        if restrict(
            &mut out[1..],
            output,
            target,
            &starts,
            &rename,
            &mut replacements,
            &mut atoms,
            vars,
        )
        .is_none()
        {
            return Ok(None);
        }
        let shape = target.ty.shaped().expect("checked tile");
        out[0] = Stmt {
            id: None,
            span: target.span,
            kind: StmtKind::Assign {
                target: target.clone(),
                op: AssignOp::Assign,
                value: Expr {
                    kind: ExprKind::TileAlloc {
                        shape: shape.shape.clone(),
                        dtype: shape.elem.clone(),
                    },
                    ty: target.ty.clone(),
                    sym: None,
                    span: target.span,
                },
            },
        };
        for id in rename.values() {
            map_ty(&mut vars[*id].ty, &atoms);
        }
        for statement in &mut out[1..] {
            substitute(statement, &rename, &replacements, &atoms);
        }
        if !project_calls(&mut out, output, view, target, vars, project_call)? {
            return Ok(None);
        }
        Ok(Some(out))
    })();
    if !matches!(&result, Ok(Some(_))) {
        vars.truncate(checkpoint);
    }
    result
}

fn admissible(
    region: &[Stmt],
    output: VarId,
    view: &Expr,
    target: &Expr,
    vars: &[Var],
) -> Option<Vec<Sym>> {
    let ExprKind::Var(destination) = target.kind else {
        return None;
    };
    if output == destination {
        return None;
    }
    let source_shape = vars.get(output)?.ty.shaped()?;
    let target_shape = target.ty.shaped()?;
    if !matches!(vars[output].ty, Ty::Tile(_))
        || !matches!(target.ty, Ty::Tile(_))
        || !matches!(source_shape.elem, Elem::Dtype(_))
        || source_shape.elem != target_shape.elem
        || source_shape.shape.len() != target_shape.shape.len()
        || view.ty != target.ty
    {
        return None;
    }
    let starts = starts(view, output, &source_shape.shape)?;
    let [first, rest @ ..] = region else {
        return None;
    };
    if !matches!(&first.kind,StmtKind::Assign{target:Expr{kind:ExprKind::Var(v),..},op:AssignOp::Assign,value:Expr{kind:ExprKind::TileAlloc{..},..}} if *v==output)
        || rest.is_empty()
    {
        return None;
    }
    let mut locals = HashSet::new();
    definitions(rest, &mut locals);
    locals.remove(&output);
    if written(rest)
        .iter()
        .any(|v| *v != output && !locals.contains(v))
        || !fresh_locals(rest, &locals, &mut HashSet::new())
        || !independent_region(rest, output, starts.len(), vars, &references(rest))
    {
        return None;
    }
    Some(starts)
}

fn references(body: &[Stmt]) -> HashMap<VarId, usize> {
    let mut counts = HashMap::new();
    for s in body {
        visit_stmt(s, &mut |e| {
            walk(e, &mut |e| {
                if let ExprKind::Var(v) = e.kind {
                    *counts.entry(v).or_default() += 1;
                }
            })
        });
    }
    counts
}
fn independent_region(
    body: &[Stmt],
    output: VarId,
    rank: usize,
    vars: &[Var],
    region_references: &HashMap<VarId, usize>,
) -> bool {
    body.iter().all(|s| match &s.kind {
        StmtKind::Owned {
            vars: indices,
            tile,
            body,
        } if matches!(tile.kind,ExprKind::Var(v) if v==output) => {
            if indices.len() != rank {
                return false;
            }
            let Some(coordinates) = indices
                .iter()
                .map(|v| match &vars[*v].kind {
                    VarKind::Index(a) => Some(Sym::atom(a.clone())),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            let mut locals = HashSet::new();
            definitions(body, &mut locals);
            locals.extend(indices);
            let local_references = references(body);
            // A scalar written while visiting coordinates cannot be observed by
            // another phase. Even a plain assignment would otherwise expose the
            // last visited coordinate and change under output restriction.
            if locals.iter().any(|v| {
                *v != output
                    && region_references.get(v).copied().unwrap_or(0)
                        > local_references.get(v).copied().unwrap_or(0)
            }) {
                return false;
            }
            // State may carry between algorithmic iterations through the output
            // coordinate. Other state must be fresh for every coordinate.
            !written(body)
                .iter()
                .any(|v| *v != output && !locals.contains(v))
                && independent_output(body, output, Some(&coordinates))
                && fresh_locals(body, &locals, &mut indices.iter().copied().collect())
        }
        StmtKind::Range { body, .. } => {
            independent_region(body, output, rank, vars, region_references)
        }
        StmtKind::If { cond, then, els } => {
            independent_expression(cond, output, None)
                && independent_region(then, output, rank, vars, region_references)
                && independent_region(els, output, rank, vars, region_references)
        }
        StmtKind::Expr(Expr {
            kind: ExprKind::Call { .. },
            ..
        }) => true,
        _ => independent_output(std::slice::from_ref(s), output, None),
    })
}

fn restrict(
    body: &mut [Stmt],
    output: VarId,
    target: &Expr,
    starts: &[Sym],
    rename: &HashMap<VarId, VarId>,
    replacements: &mut HashMap<VarId, Expr>,
    atoms: &mut Vec<(Atom, Sym)>,
    vars: &[Var],
) -> Option<()> {
    for statement in body {
        match &mut statement.kind {
            StmtKind::Owned {
                vars: indices,
                tile,
                body,
            } if matches!(tile.kind, ExprKind::Var(v) if v == output) => {
                let local = indices
                    .iter()
                    .map(|v| variable(rename[v], vars))
                    .collect::<Vec<_>>();
                let coordinates = indices
                    .iter()
                    .map(|v| variable(*v, vars).sym.unwrap())
                    .collect::<Vec<_>>();
                for ((v, l), start) in indices.iter().zip(&local).zip(starts) {
                    let VarKind::Index(atom) = &vars[*v].kind else {
                        return None;
                    };
                    let global = start.add(l.sym.as_ref()?);
                    atoms.retain(|(a, _)| a != atom);
                    atoms.push((atom.clone(), global.clone()));
                    replacements.insert(*v, symbol(global, target.span));
                }
                for s in body {
                    replace_output(s, output, Some(&coordinates), target, &local)?;
                }
                *tile = target.clone();
            }
            StmtKind::Range { body, .. } => restrict(
                body,
                output,
                target,
                starts,
                rename,
                replacements,
                atoms,
                vars,
            )?,
            StmtKind::If { cond, then, els } => {
                replace_output_expression(cond, output, None, target, &[])?;
                restrict(
                    then,
                    output,
                    target,
                    starts,
                    rename,
                    replacements,
                    atoms,
                    vars,
                )?;
                restrict(
                    els,
                    output,
                    target,
                    starts,
                    rename,
                    replacements,
                    atoms,
                    vars,
                )?;
            }
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { .. },
                ..
            }) => {}
            _ => replace_output(statement, output, None, target, &[])?,
        }
    }
    Some(())
}

fn project_calls(
    body: &mut Vec<Stmt>,
    output: VarId,
    view: &Expr,
    target: &Expr,
    vars: &mut Vec<Var>,
    project_call: &mut ProjectCall<'_>,
) -> Result<bool, String> {
    let mut out = Vec::new();
    for mut statement in std::mem::take(body) {
        match &mut statement.kind {
            StmtKind::Expr(
                call @ Expr {
                    kind: ExprKind::Call { .. },
                    ..
                },
            ) => {
                let Some(projected) = project_call(call, output, view, target, vars)? else {
                    return Ok(false);
                };
                out.extend(projected);
                continue;
            }
            StmtKind::Range { body, .. } => {
                if !project_calls(body, output, view, target, vars, project_call)? {
                    return Ok(false);
                }
            }
            StmtKind::If { then, els, .. } => {
                if !project_calls(then, output, view, target, vars, project_call)?
                    || !project_calls(els, output, view, target, vars, project_call)?
                {
                    return Ok(false);
                }
            }
            _ => {}
        }
        out.push(statement);
    }
    *body = out;
    Ok(true)
}
fn starts(view: &Expr, output: VarId, shape: &[Sym]) -> Option<Vec<Sym>> {
    match &view.kind {
        ExprKind::Var(v) if *v == output => Some(vec![Sym::constant(0); shape.len()]),
        ExprKind::Index { base, indices } if matches!(base.kind,ExprKind::Var(v) if v==output) => {
            if indices.len() > shape.len() {
                return None;
            }
            (0..shape.len())
                .map(|i| match indices.get(i) {
                    None | Some(Index::Slice { start: None, .. }) => Some(Sym::constant(0)),
                    Some(Index::Slice { start: Some(e), .. }) => e.sym.clone(),
                    _ => None,
                })
                .collect()
        }
        _ => None,
    }
}
fn definitions(body: &[Stmt], out: &mut HashSet<VarId>) {
    for s in body {
        match &s.kind {
            StmtKind::Assign {
                target:
                    Expr {
                        kind: ExprKind::Var(v),
                        ..
                    },
                op: AssignOp::Assign,
                ..
            } => {
                out.insert(*v);
            }
            StmtKind::Owned { vars, body, .. } | StmtKind::Parallel { vars, body, .. } => {
                out.extend(vars);
                definitions(body, out);
            }
            StmtKind::Range { var, body, .. } | StmtKind::Lanes { var, body, .. } => {
                out.insert(*var);
                definitions(body, out);
            }
            StmtKind::LoadLoop {
                vars, offset, body, ..
            } => {
                out.extend(vars);
                out.extend(offset);
                definitions(body, out);
            }
            StmtKind::If { then, els, .. } => {
                definitions(then, out);
                definitions(els, out);
            }
            StmtKind::Reduction(r) => {
                for m in r.implementations() {
                    for p in m.left.iter().chain(&m.right).chain(&m.output) {
                        if let ExprKind::Var(v) = p.kind {
                            out.insert(v);
                        }
                    }
                    definitions(&m.body, out);
                }
            }
            _ => {}
        }
    }
}
fn rename_pieces(statement: &mut Stmt, namespace: usize, pieces: &mut HashMap<Atom, Atom>) {
    if let StmtKind::LoadLoop { piece, .. } = &mut statement.kind {
        let fresh = Atom::Param(format!("projected_piece#{namespace}#{}", pieces.len()));
        *piece = pieces.entry(piece.clone()).or_insert(fresh).clone();
    }
    nested_mut(statement, &mut |body| {
        for statement in body {
            rename_pieces(statement, namespace, pieces);
        }
    });
}
// A renamed temporary must be defined in this output coordinate before it is
// read. Otherwise an assignment such as counter=counter+1 carries state between
// coordinates and restricting the domain would change the computation.
fn fresh_locals(body: &[Stmt], locals: &HashSet<VarId>, defined: &mut HashSet<VarId>) -> bool {
    fn read(e: &Expr, locals: &HashSet<VarId>, defined: &HashSet<VarId>) -> bool {
        let mut valid = true;
        walk(e, &mut |e| {
            if let ExprKind::Var(v) = e.kind {
                valid &= !locals.contains(&v) || defined.contains(&v);
            }
        });
        valid
    }
    for s in body {
        match &s.kind {
            StmtKind::Assign { target, op, value } => {
                if !read(value, locals, defined) {
                    return false;
                }
                if let ExprKind::Var(v) = target.kind {
                    if *op != AssignOp::Assign && !defined.contains(&v) {
                        return false;
                    }
                    defined.insert(v);
                } else if !read(target, locals, defined) {
                    return false;
                }
            }
            StmtKind::Owned { vars, tile, body } => {
                if !read(tile, locals, defined) {
                    return false;
                }
                let mut inner = defined.clone();
                inner.extend(vars);
                if !fresh_locals(body, locals, &mut inner) {
                    return false;
                }
            }
            StmtKind::Range { var, body, .. } => {
                let mut inner = defined.clone();
                inner.insert(*var);
                if !fresh_locals(body, locals, &mut inner) {
                    return false;
                }
            }
            StmtKind::LoadLoop { domain, views, vars, offset, body, .. } => {
                if !read(&domain.view, locals, defined)
                    || !views.iter().all(|view| read(view, locals, defined))
                {
                    return false;
                }
                let mut inner = defined.clone();
                inner.extend(vars);
                inner.extend(offset);
                if !fresh_locals(body, locals, &mut inner) {
                    return false;
                }
            }
            StmtKind::If { cond, then, els } => {
                if !read(cond, locals, defined) {
                    return false;
                }
                let mut a = defined.clone();
                let mut b = defined.clone();
                if !fresh_locals(then, locals, &mut a) || !fresh_locals(els, locals, &mut b) {
                    return false;
                }
                defined.extend(a.intersection(&b));
            }
            StmtKind::Reduction(r) => {
                if !r.operands().all(|e| read(e, locals, defined)) {
                    return false;
                }
                for m in r.implementations() {
                    let mut inner = defined.clone();
                    for e in m.left.iter().chain(&m.right).chain(&m.output) {
                        if let ExprKind::Var(v) = e.kind {
                            inner.insert(v);
                        }
                    }
                    if !fresh_locals(&m.body, locals, &mut inner) {
                        return false;
                    }
                }
            }
            StmtKind::Expr(e) => {
                if !read(e, locals, defined) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

fn coordinate_access(e: &Expr, output: VarId, coordinates: &[Sym]) -> bool {
    let ExprKind::Index { base, indices } = &e.kind else {
        return false;
    };
    matches!(base.kind,ExprKind::Var(v) if v==output)
        && indices.len() == coordinates.len()
        && indices
            .iter()
            .zip(coordinates)
            .all(|(index, c)| match index {
                Index::Point(e) => e.sym.as_ref() == Some(c),
                Index::Slice {
                    start: Some(a),
                    end: Some(b),
                } => a.sym.as_ref() == Some(c) && b.sym.as_ref() == Some(&c.add(&Sym::constant(1))),
                _ => false,
            })
}
fn independent_expression(e: &Expr, output: VarId, coordinates: Option<&[Sym]>) -> bool {
    crate::effects::expression_can_be_omitted(e) && independent_accesses(e, output, coordinates)
}
fn independent_accesses(e: &Expr, output: VarId, coordinates: Option<&[Sym]>) -> bool {
    if coordinates.is_some_and(|c| coordinate_access(e, output, c)) {
        return true;
    }
    if let ExprKind::Builtin {
        name: Builtin::Extent,
        args,
    } = &e.kind
    {
        if matches!(args[0].kind,ExprKind::Var(v) if v==output) {
            return e.sym.is_some();
        }
    }
    if matches!(e.kind,ExprKind::Var(v) if v==output) {
        return false;
    }
    let mut valid = true;
    children(e, &mut |e| {
        valid &= independent_accesses(e, output, coordinates)
    });
    valid
}
fn independent_output(body: &[Stmt], output: VarId, coordinates: Option<&[Sym]>) -> bool {
    fn structure(s: &Stmt) -> bool {
        match &s.kind {
            StmtKind::Parallel { .. } | StmtKind::Lanes { .. } => false,
            StmtKind::LoadLoop { capacity, body, .. } => {
                capacity.is_some_and(|capacity| capacity > 0) && body.iter().all(structure)
            }
            StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } => {
                body.iter().all(structure)
            }
            StmtKind::If { then, els, .. } => then.iter().chain(els).all(structure),
            StmtKind::Reduction(r) => {
                r.tree.is_some()
                    && r.implementations().count() == 1 + usize::from(r.step.is_some())
                    && r.bodies().flatten().all(structure)
            }
            _ => true,
        }
    }
    body.iter().all(|s| {
        let mut valid = structure(s);
        visit_stmt(s, &mut |e| {
            valid &= independent_expression(e, output, coordinates)
        });
        valid
    })
}
fn replace_output_expression(
    e: &mut Expr,
    output: VarId,
    coordinates: Option<&[Sym]>,
    target: &Expr,
    local: &[Expr],
) -> Option<()> {
    if coordinates.is_some_and(|c| coordinate_access(e, output, c)) {
        let ExprKind::Index { indices, .. } = &e.kind else {
            unreachable!()
        };
        let indices = indices
            .iter()
            .zip(local)
            .map(|(index, l)| match index {
                Index::Point(_) => Index::Point(l.clone()),
                Index::Slice { .. } => Index::Slice {
                    start: Some(l.clone()),
                    end: Some(symbol(
                        l.sym.as_ref().unwrap().add(&Sym::constant(1)),
                        e.span,
                    )),
                },
            })
            .collect();
        e.kind = ExprKind::Index {
            base: Box::new(target.clone()),
            indices,
        };
        return Some(());
    }
    if let ExprKind::Builtin {
        name: Builtin::Extent,
        args,
    } = &e.kind
    {
        if matches!(args[0].kind,ExprKind::Var(v) if v==output) {
            *e = symbol(e.sym.clone()?, e.span);
            return Some(());
        }
    }
    let mut valid = Some(());
    children_mut(e, &mut |e| {
        valid =
            valid.and_then(|_| replace_output_expression(e, output, coordinates, target, local));
    });
    valid
}
fn replace_output(
    s: &mut Stmt,
    output: VarId,
    coordinates: Option<&[Sym]>,
    target: &Expr,
    local: &[Expr],
) -> Option<()> {
    let mut valid = Some(());
    direct_exprs_mut(s, &mut |e| {
        valid = valid.and_then(|_| replace_output_expression(e, output, coordinates, target, local))
    });
    nested_mut(s, &mut |body| {
        for s in body {
            valid = valid.and_then(|_| replace_output(s, output, coordinates, target, local));
        }
    });
    valid
}
fn substitute(
    s: &mut Stmt,
    rename: &HashMap<VarId, VarId>,
    values: &HashMap<VarId, Expr>,
    atoms: &[(Atom, Sym)],
) {
    // Substitute original index reads before renaming binder identities. Values
    // such as f32(i) must carry a real symbolic expression, not display metadata.
    let expression = |e: &mut Expr| {
        let mapped = values
            .iter()
            .map(|(v, e)| (*v, e.clone()))
            .collect::<HashMap<_, _>>();
        let sym = atoms
            .iter()
            .filter_map(|(a, s)| {
                if let Atom::Param(a) = a {
                    Some((a.clone(), Some(s.clone())))
                } else {
                    None
                }
            })
            .collect();
        *e = crate::lower::subst_vars(e, &mapped, &sym);
    };
    direct_exprs_mut(s, &mut |e| expression(e));
    if let StmtKind::Reduction(r) = &mut s.kind {
        if let Some(call) = r.merge.source_mut() { expression(call); }
        if let Some(step) = &mut r.step {
            if let Some(call) = step.call.source_mut() { expression(call); }
        }
        for m in r.implementations_mut() {
            for e in m.left.iter_mut().chain(&mut m.right).chain(&mut m.output) {
                expression(e);
            }
        }
    }
    // Mapping types and symbols is idempotent for fresh atoms; only original
    // bindings occur as keys. Body recursion and binder updates are explicit.
    direct_exprs_mut(s, &mut |e| map_expr(e, rename, atoms));
    match &mut s.kind {
        StmtKind::Owned { vars, .. } | StmtKind::Parallel { vars, .. } => {
            for v in vars {
                if let Some(n) = rename.get(v) {
                    *v = *n;
                }
            }
        }
        StmtKind::Range { var, lo, hi, .. } => {
            if let Some(n) = rename.get(var) {
                *var = *n;
            }
            map_sym(lo, atoms);
            map_sym(hi, atoms);
        }
        StmtKind::Lanes { var, extent, .. } => {
            if let Some(n) = rename.get(var) {
                *var = *n;
            }
            map_sym(extent, atoms);
        }
        StmtKind::LoadLoop { vars, offset, .. } => {
            for variable in vars.iter_mut().chain(offset.iter_mut()) {
                if let Some(new) = rename.get(variable) {
                    *variable = *new;
                }
            }
        }
        StmtKind::Reduction(r) => {
            if let Some(call) = r.merge.source_mut() { map_expr(call, rename, atoms); }
            if let Some(step) = &mut r.step {
                if let Some(call) = step.call.source_mut() { map_expr(call, rename, atoms); }
            }
            for m in r.implementations_mut() {
                for e in m.left.iter_mut().chain(&mut m.right).chain(&mut m.output) {
                    map_expr(e, rename, atoms);
                }
            }
        }
        _ => {}
    }
    nested_mut(s, &mut |body| {
        for s in body {
            substitute(s, rename, values, atoms);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        interp::{Arg, Interpreter, TensorData},
        program::{compile, Program, SourceFile},
        Scope,
    };
    fn fixture(update: &str) -> LoweredIr {
        let text=format!("fn entry(x:tensor[4,5] f16,out:tensor[2,2] f32):\n  counter = 7\n  t = load(x)\n  s = tile[4,5] f32\n  for i,j in owned(s): s[i,j] = f32(i-j)\n  for i,j in owned(s):\n{update}\n  y = s[1:3,2:4]\n  store(y,out)\n");
        source_fixture(text)
    }
    fn source_fixture(text: String) -> LoweredIr {
        let p = compile(
            &[SourceFile {
                path: "region.seismic.portable".into(),
                text,
                scope: Scope::Portable,
            }],
            &[],
        )
        .unwrap();
        crate::lower::lower(&p, "entry", "cpu", &HashMap::new()).unwrap()
    }
    fn execute(f: LoweredIr) -> Vec<u8> {
        let p = Program {
            functions: vec![Function {
                name: f.name,
                is_construct: false,
                shape_params: vec![],
                elem_params: vec![],
                params: f.params,
                index_params: f.index_params,
                vars: f.vars,
                body: f.body,
            }],
            lowerings: vec![],
            signatures: HashMap::new(),
        };
        let mut vm = Interpreter::new(&p);
        let x = vm.add_tensor(TensorData::dense(
            DType::F16,
            vec![4, 5],
            (0..20).map(|i| (i as f64 - 8.0) / 3.0).collect(),
        ));
        let out = vm.add_tensor(TensorData::dense(DType::F32, vec![2, 2], vec![0.0; 4]));
        vm.run(
            "entry",
            &[Arg::Tensor(x), Arg::Tensor(out)],
            &HashMap::new(),
        )
        .unwrap();
        vm.tensors[out].device_bytes().remove(0)
    }
    fn apply(f: &mut LoweredIr) -> bool {
        let source = f.vars.iter().position(|v| v.name == "s").unwrap();
        let start=f.body.iter().position(|s|matches!(&s.kind,StmtKind::Assign{target:Expr{kind:ExprKind::Var(v),..},..} if *v==source)).unwrap();
        let end=f.body.iter().position(|s|matches!(&s.kind,StmtKind::Assign{value:Expr{kind:ExprKind::Index{base,..},..},..} if matches!(base.kind,ExprKind::Var(v) if v==source))).unwrap();
        let StmtKind::Assign { target, value, .. } = &f.body[end].kind else {
            unreachable!()
        };
        let old = f.vars.len();
        if let Some(projected) = project(
            &f.body[start..end],
            source,
            value,
            target,
            &mut f.vars,
            &mut |_, _, _, _, _| Ok(None),
        )
        .unwrap()
        {
            f.body.splice(start..=end, projected);
            true
        } else {
            assert_eq!(f.vars.len(), old);
            false
        }
    }
    #[test]
    fn projected_reduction_regions_preserve_original_indices_seeds_and_precision() {
        let f=fixture("    row = t[i,:]\n    total = reduce(row,0,sum,ordered=true)\n    s[i,j] = fma(f32(i),f32(j),f32(total)+s[i,j])");
        let expected = execute(f.clone());
        let mut projected = f;
        assert!(apply(&mut projected));
        assert_eq!(execute(projected), expected);
    }
    #[test]
    fn projected_dynamic_reductions_rebind_stream_inputs_and_extents() {
        fn streams(body: &mut [Stmt], bindings: &mut Vec<(Atom, Vec<VarId>, Option<VarId>)>) {
            for statement in body {
                if let StmtKind::LoadLoop { piece, vars, offset, .. } = &statement.kind {
                    bindings.push((piece.clone(), vars.clone(), *offset));
                }
                nested_mut(statement, &mut |body| streams(body, bindings));
            }
        }
        let mut f = fixture("    row = t[i,:i32(t[i,j])]\n    total = reduce(row,0,sum,ordered=true)\n    s[i,j] = f32(total) + s[i,j]");
        let mut original = Vec::new();
        streams(&mut f.body, &mut original);
        assert!(!original.is_empty(), "the reduction must retain its dynamic stream");
        let expected = execute(f.clone());
        assert!(apply(&mut f));
        let mut projected = Vec::new();
        streams(&mut f.body, &mut projected);
        assert_eq!(projected.len(), original.len());
        for ((old_piece, old_vars, old_offset), (piece, vars, offset)) in original.iter().zip(&projected) {
            assert_ne!(piece, old_piece);
            assert!(vars.iter().all(|var| !old_vars.contains(var)));
            assert_ne!(offset, old_offset);
            assert!(vars.iter().all(|var| f.vars[*var].ty.shaped().unwrap().shape.contains(&Sym::atom(piece.clone()))));
        }
        assert_eq!(execute(f), expected);
    }
    #[test]
    fn projected_shape_queries_keep_the_original_logical_domain() {
        let f = fixture("    s[i,j] = s[i,j] + f32(extent(s,1)) + f32(i*11+j)");
        let expected = execute(f.clone());
        let mut projected = f;
        assert!(apply(&mut projected));
        assert_eq!(execute(projected), expected);
    }
    #[test]
    fn coordinate_carried_scalar_state_and_publications_prevent_projection() {
        for update in [
            "    counter = counter + 1\n    s[i,j] = f32(counter)",
            "    store(t,x)\n    s[i,j] = s[i,j] + 1.0",
        ] {
            let mut f = fixture(update);
            assert!(!apply(&mut f));
        }
    }
    #[test]
    fn reads_from_other_output_coordinates_are_not_independent() {
        let mut f = fixture("    s[i,j] = s[i,j] + s[0,0]");
        assert!(!apply(&mut f));
    }
    #[test]
    fn algorithmic_loops_and_branches_preserve_order_and_coordinate_meaning() {
        for iterations in [0, 1, 3, 4] {
            let f = source_fixture(format!(
                r#"fn entry(x:tensor[4,5] f16,out:tensor[2,2] f32):
  t = load(x)
  s = tile[4,5] f32
  for i,j in owned(s): s[i,j] = f32(i-j)
  bias = 1.0
  for k in range(0,{iterations}):
    bias = bias * 0.75
    row = t[k,:]
    total = reduce(row,0,sum,ordered=true)
    if k < 2:
      for i,j in owned(s): s[i,j] = f32(f16(s[i,j] + f32(total))) + bias + f32((j / 2) << 1)
    else:
      for i,j in owned(s): s[i,j] = fma(f32(i-j),bias,s[i,j])
  y = s[1:3,2:4]
  store(y,out)
"#
            ));
            let expected = execute(f.clone());
            let mut projected = f;
            assert!(apply(&mut projected), "iterations={iterations}");
            assert_eq!(execute(projected), expected, "iterations={iterations}");
        }
    }
    #[test]
    fn coordinate_written_state_cannot_escape_into_another_phase() {
        let mut f = source_fixture(
            r#"fn entry(x:tensor[4,5] f16,out:tensor[2,2] f32):
  s = tile[4,5] f32
  bias = 0.0
  for i,j in owned(s):
    bias = f32(i-j)
    s[i,j] = bias
  for i,j in owned(s): s[i,j] = s[i,j] + bias
  y = s[1:3,2:4]
  store(y,out)
"#
            .into(),
        );
        assert!(!apply(&mut f));
    }
    #[test]
    fn projection_does_not_remove_undischarged_integer_exceptions() {
        let mut f = fixture("    s[i,j] = s[i,j] + f32(i / i32(t[i,j]))");
        assert!(!apply(&mut f));
    }
}
