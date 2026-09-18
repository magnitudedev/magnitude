//! Pull a demanded logical subview through its reaching pure producers. Each
//! projected value remains a snapshot and retains every intermediate conversion.
use super::*;
mod geometry;

struct Projection {
    body: Vec<Stmt>,
    /// The reaching logical view uses captured coordinate bindings, so a later
    /// projection does not evaluate the original endpoint expressions again.
    view: Expr,
}
#[derive(Clone)]
enum Producer {
    Load(Expr),
    View(Expr),
    Region {
        output: VarId,
        body: Vec<Stmt>,
    },
    Pointwise {
        indices: Vec<VarId>,
        value: Expr,
        dtype: DType,
        region: Vec<Stmt>,
    },
}
impl Producer {
    fn mentions(&self, variable: VarId) -> bool {
        match self {
            Self::Load(e) | Self::View(e) => mentions(e, variable),
            Self::Pointwise { value, .. } => mentions(value, variable),
            Self::Region { output, body } => {
                *output != variable && body.iter().any(|s| crate::effects::uses(s, variable))
            }
        }
    }
    fn reads_external_memory(&self) -> bool {
        match self {
            Self::Load(_) => true,
            Self::Region { .. } => true,
            _ => false,
        }
    }
}
pub(super) fn select(
    f: &mut LoweredIr,
    program: &crate::program::Program,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
    project_call: &mut super::regions::ProjectCall<'_>,
) -> Result<(), String> {
    let mut removable = HashSet::new();
    block(
        &mut f.body,
        &mut f.vars,
        &mut HashMap::new(),
        select,
        &mut removable,
        project_call,
        program,
    )?;
    removable.retain(|v| removable_definitions(&f.body, *v, &f.vars, program, &f.body));
    loop {
        let dead = removable
            .iter()
            .copied()
            .filter(|v| !used_outside_definition(&f.body, *v, program))
            .collect::<HashSet<_>>();
        if dead.is_empty() {
            break;
        }
        remove(&mut f.body, &dead, program);
        removable.retain(|v| !dead.contains(v));
    }
    // A value's geometry may still be observed after all of its elements become
    // dead. Retain the definitions that capture shape/layout/endpoints; omit
    // only independently omittable element production. Backend realization uses
    // the same data-demand facts to keep those definitions without allocation.
    prune_geometry_producers(&mut f.body, &f.vars);
    Ok(())
}

fn prune_geometry_producers(body: &mut Vec<Stmt>, vars: &[Var]) {
    fn candidates(body: &[Stmt], root_body: &[Stmt], vars: &[Var], safe: &mut HashSet<VarId>, unsafe_: &mut HashSet<VarId>) {
        for (position, statement) in body.iter().enumerate() {
            match &statement.kind {
                StmtKind::Owned { tile, body: definition, .. } => {
                    if let ExprKind::Var(variable) = tile.kind {
                        let mut written = HashSet::new();
                        for statement in definition { crate::rewrite::writes(statement, &mut written); }
                        if matches!(vars[variable].kind, VarKind::Local)
                            && crate::lower::producer_value(definition, variable, vars,
                                &body[..position], &body[position + 1..]).is_some()
                            && written.iter().all(|v| *v == variable
                                || (!matches!(vars[*v].kind, VarKind::Param(_))
                                    && !uses_except(root_body, statement, *v)))
                        {
                            safe.insert(variable);
                        } else {
                            unsafe_.insert(variable);
                        }
                    }
                    candidates(definition, root_body, vars, safe, unsafe_);
                }
                StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::LoadLoop { body, .. } | StmtKind::Lanes { body, .. } => {
                    candidates(body, root_body, vars, safe, unsafe_);
                }
                StmtKind::If { then, els, .. } => {
                    candidates(then, root_body, vars, safe, unsafe_);
                    candidates(els, root_body, vars, safe, unsafe_);
                }
                StmtKind::Reduction(r) => {
                    for body in r.bodies() { candidates(body, root_body, vars, safe, unsafe_); }
                }
                _ => {}
            }
        }
    }
    fn omit(body: &mut Vec<Stmt>, variables: &HashSet<VarId>) {
        body.retain(|s| !matches!(&s.kind, StmtKind::Owned { tile, .. }
            if matches!(tile.kind, ExprKind::Var(v) if variables.contains(&v))));
        for statement in body { nested_mut(statement, &mut |child| omit(child, variables)); }
    }
    let mut safe = HashSet::new();
    let mut unsafe_ = HashSet::new();
    candidates(body, body, vars, &mut safe, &mut unsafe_);
    safe.retain(|v| !unsafe_.contains(v));
    if safe.is_empty() { return; }
    // Start with all candidate producers omitted, then restore every producer
    // whose data has a live consumer. Restored producers may demand more input
    // data, so repeat to a fixed point before changing the actual statements.
    loop {
        let mut projected = body.clone();
        omit(&mut projected, &safe);
        let data = crate::demand::data_variables(&projected);
        let before = safe.len();
        safe.retain(|v| !data.contains(v));
        if safe.len() == before { break; }
    }
    omit(body, &safe);
}
fn block(
    body: &mut Vec<Stmt>,
    vars: &mut Vec<Var>,
    available: &mut HashMap<VarId, Producer>,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
    removable: &mut HashSet<VarId>,
    project_call: &mut super::regions::ProjectCall<'_>,
    program: &crate::program::Program,
) -> Result<(), String> {
    let original = body.clone();
    let mut result = Vec::new();
    for (position, mut stmt) in std::mem::take(body).into_iter().enumerate() {
        // Shape queries need the stable logical view, not its materialized
        // data. The same reaching-version checks guard this substitution.
        direct_exprs_mut(&mut stmt,&mut |e|metadata(e,available));
        if let StmtKind::LoadLoop{domain,..}=&mut stmt.kind{
            if let Some(view)=tensor(&domain.view,available,0){domain.view=view;}
        }
        let mut projected = None;
        let mut projected_view = None;
        let mut updates = HashSet::new();
        updates.extend(facts(&stmt, program).0);
        let previous_regions = if matches!(
            stmt.kind,
            StmtKind::Owned { .. }
                | StmtKind::Range { .. }
                | StmtKind::If { .. }
                | StmtKind::Expr(Expr {
                    kind: ExprKind::Call { .. },
                    ..
                })
        ) {
            updates
                .iter()
                .filter_map(|v| match available.get(v) {
                    Some(Producer::Region { body, .. }) => Some((*v, body.clone())),
                    Some(Producer::Pointwise { region, .. }) if !region.is_empty() => {
                        Some((*v, region.clone()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if let StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        } = &stmt.kind
        {
            if let (ExprKind::Var(_), Ty::Tile(shape)) = (&target.kind, &target.ty) {
                if let Some(root) = root(value).filter(|v| available.contains_key(v)) {
                    let source = vars[root].ty.shaped().unwrap();
                    let size = |shape: &[Sym]| shape.iter().fold(Sym::constant(1), |a, n| a.mul(n));
                    if size(&shape.shape) != size(&source.shape)
                        && matches!(
                            value.kind,
                            ExprKind::Index { .. }
                                | ExprKind::Transpose(_)
                                | ExprKind::Builtin {
                                    name: Builtin::Reshape,
                                    ..
                                }
                        )
                    {
                        let mut projected_vars = vars.clone();
                        let candidate =
                            project(value, target, available, &mut projected_vars, project_call)?;
                        if candidate.is_some() {
                            let domain = Decision {
                                kind: DecisionKind::Producer {
                                    variable: root,
                                    name: vars[root].name.clone(),
                                    ty: vars[root].ty.clone(),
                                },
                                alternatives: vec![
                                    Alternative::Materialize,
                                    Alternative::Recompute,
                                ]
                                .into(),
                            };
                            match select(&domain)? {
                                Alternative::Materialize => {}
                                Alternative::Recompute => {
                                    *vars = projected_vars;
                                    let candidate = candidate.expect("admitted projection");
                                    projected_view = Some(candidate.view);
                                    let mut body = candidate.body;
                                    block(
                                        &mut body,
                                        vars,
                                        &mut available.clone(),
                                        select,
                                        removable,
                                        project_call,
                                        program,
                                    )?;
                                    projected = Some(body)
                                }
                                _ => return Err("invalid bounded producer realization".into()),
                            }
                        }
                    }
                }
            }
        }
        if let Some(view) = projected_view {
            if let StmtKind::Assign { value, .. } = &mut stmt.kind { *value = view; }
        }
        if let StmtKind::LoadLoop {
            offset,
            vars: bindings,
            views,
            axes,
            piece,
            modes,
            body: inner,
            ..
        } = &mut stmt.kind
        {
            // The original producer is a snapshot. Delaying its reads into a
            // repeated region requires stability for the entire region.
            let stable = stable_in(inner, available, program);
            if modes.is_none() {
                let mut new_views = Vec::new();
                let mut new_bindings = Vec::new();
                let mut new_axes = Vec::new();
                let mut prelude = Vec::new();
                for ((view, binding), axis) in views.iter().zip(bindings.iter()).zip(axes.iter()) {
                    let Some(variable) = root(view).filter(|v| stable.contains_key(v)) else {
                        new_views.push(view.clone());
                        new_bindings.push(*binding);
                        new_axes.push(*axis);
                        continue;
                    };
                    let mut candidate_vars = vars.clone();
                    let candidate = if let Some(source) = tensor(view, &stable, 0) {
                        Some((vec![source], vec![*binding], vec![*axis], Vec::new()))
                    } else if let Some(offset) = *offset {
                        stream_projection(
                            view,
                            *binding,
                            *axis,
                            piece,
                            offset,
                            &stable,
                            &mut candidate_vars,
                            project_call,
                        )?
                    } else {
                        None
                    };
                    let Some((sources, names, source_axes, computed)) = candidate else {
                        new_views.push(view.clone());
                        new_bindings.push(*binding);
                        new_axes.push(*axis);
                        continue;
                    };
                    let domain = Decision {
                        kind: DecisionKind::Producer {
                            variable,
                            name: vars[variable].name.clone(),
                            ty: vars[variable].ty.clone(),
                        },
                        alternatives: vec![Alternative::Materialize, Alternative::Recompute].into(),
                    };
                    match select(&domain)? {
                        Alternative::Materialize => {
                            new_views.push(view.clone());
                            new_bindings.push(*binding);
                            new_axes.push(*axis);
                        }
                        Alternative::Recompute => {
                            *vars = candidate_vars;
                            new_views.extend(sources);
                            new_bindings.extend(names);
                            new_axes.extend(source_axes);
                            prelude.extend(computed);
                        }
                        _ => return Err("invalid streamed producer realization".into()),
                    }
                }
                prelude.append(inner);
                *inner = prelude;
                *views = new_views;
                *bindings = new_bindings;
                *axes = new_axes;
            }
        }
        // Traverse the same reaching environment; effects inside a child end the
        // motion window there, and cannot leak a new definition out of its scope.
        let mut error = None;
        nested_mut(&mut stmt, &mut |child| {
            if error.is_none() {
                error = block(
                    child,
                    vars,
                    &mut stable_in(child, available, program),
                    select,
                    removable,
                    project_call,
                    program,
                )
                .err()
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        let mut changed = HashSet::new();
        let (written, effect) = facts(&stmt, program);
        changed.extend(written);
        loop {
            let invalid = available
                .iter()
                .filter_map(|(v, p)| {
                    (changed.contains(v)
                        || changed.iter().any(|v| p.mentions(*v))
                        || (effect && p.reads_external_memory()))
                    .then_some(*v)
                })
                .collect::<Vec<_>>();
            if invalid.is_empty() {
                break;
            }
            for v in invalid {
                available.remove(&v);
                changed.insert(v);
            }
        }
        for (output, mut region) in previous_regions {
            region.push(stmt.clone());
            available.insert(
                output,
                Producer::Region {
                    output,
                    body: region,
                },
            );
            removable.insert(output);
        }
        match &stmt.kind {
            StmtKind::Assign {
                target,
                op: AssignOp::Assign,
                value,
            } if matches!(target.ty, Ty::Tile(_)) => {
                if let ExprKind::Var(v) = target.kind {
                    if !matches!(vars[v].kind, VarKind::Local) {
                        result.extend(projected.unwrap_or_else(|| vec![stmt]));
                        continue;
                    }
                    let p = match &value.kind {
                        ExprKind::TileAlloc { .. } => Some(Producer::Region {
                            output: v,
                            body: vec![stmt.clone()],
                        }),
                        ExprKind::Builtin {
                            name: Builtin::Load,
                            args,
                        } => Some(Producer::Load(args[0].clone())),
                        ExprKind::Index { .. }
                        | ExprKind::Transpose(_)
                        | ExprKind::Var(_)
                        | ExprKind::Builtin {
                            name: Builtin::Reshape,
                            ..
                        } => Some(Producer::View(value.clone())),
                        _ => None,
                    };
                    if let Some(p) = p {
                        available.insert(v, p);
                        removable.insert(v);
                    }
                }
            }
            StmtKind::Owned {
                vars: indices,
                tile,
                body: producer,
            } => {
                if let ExprKind::Var(v) = tile.kind {
                    if !matches!(vars[v].kind, VarKind::Local) {
                        result.extend(projected.unwrap_or_else(|| vec![stmt]));
                        continue;
                    }
                    if let Some((target, value)) = crate::lower::producer_value(
                        producer,
                        v,
                        vars,
                        &original[..position],
                        &original[position + 1..],
                    ) {
                        let plain=match &target.kind {ExprKind::Index{base,indices:ix} if matches!(base.kind,ExprKind::Var(id) if id==v)=>ix.len()==indices.len()&&ix.iter().zip(indices).all(|(i,v)|matches!((i,&vars[*v].kind),(Index::Point(e),VarKind::Index(a)) if e.sym.as_ref()==Some(&Sym::atom(a.clone())))),_=>false};
                        if plain && !mentions(&value, v) {
                            if let Some(Elem::Dtype(dtype)) = tile.ty.shaped().map(|s| &s.elem) {
                                available.insert(
                                    v,
                                    Producer::Pointwise {
                                        indices: indices.clone(),
                                        value,
                                        dtype: *dtype,
                                        region: match available.get(&v) {
                                            Some(Producer::Region { body, .. }) => body.clone(),
                                            _ => Vec::new(),
                                        },
                                    },
                                );
                                removable.insert(v);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        result.extend(projected.unwrap_or_else(|| vec![stmt]));
    }
    *body = result;
    Ok(())
}
// A definition inherited from outside a repeated region cannot be replayed
// after a loop-carried change to any of its transitive inputs. Local definitions
// established while walking the child still become available normally.
fn stable_in(
    body: &[Stmt],
    available: &HashMap<VarId, Producer>,
    program: &crate::program::Program,
) -> HashMap<VarId, Producer> {
    let mut changed = HashSet::new();
    for s in body {
        changed.extend(facts(s, program).0);
    }
    let effects = body.iter().any(|s| facts(s, program).1);
    let mut stable = available.clone();
    loop {
        let invalid = stable
            .iter()
            .filter_map(|(v, p)| {
                (changed.contains(v)
                    || changed.iter().any(|v| p.mentions(*v))
                    || (effects && p.reads_external_memory()))
                .then_some(*v)
            })
            .collect::<Vec<_>>();
        if invalid.is_empty() {
            break;
        }
        for v in invalid {
            stable.remove(&v);
            changed.insert(v);
        }
    }
    stable
}

/// Project the pure element computation first, then use its actual access maps
/// to bind streamed operands. The global coordinate remains in arithmetic;
/// only the addressing of a selected operand is rebased into its current slice.
fn stream_projection(
    view: &Expr,
    binding: VarId,
    axis: usize,
    piece: &Atom,
    offset: VarId,
    available: &HashMap<VarId, Producer>,
    vars: &mut Vec<Var>,
    project_call: &mut super::regions::ProjectCall<'_>,
) -> Result<Option<(Vec<Expr>, Vec<VarId>, Vec<usize>, Vec<Stmt>)>, String> {
    let start = variable(offset, vars)
        .sym
        .ok_or("stream logical offset is not an index")?;
    let count = Sym::atom(piece.clone());
    let extent = view
        .ty
        .shaped()
        .ok_or("stream producer shape missing")?
        .shape[axis]
        .clone();
    let mut shape = view.ty.shaped().unwrap().clone();
    shape.shape[axis] = count.clone();
    let mut slices = vec![
        Index::Slice {
            start: None,
            end: None
        };
        shape.shape.len()
    ];
    slices[axis] = Index::Slice {
        start: Some(symbol(start.clone(), view.span)),
        end: Some(symbol(start.add(&count), view.span)),
    };
    let sliced = Expr {
        kind: ExprKind::Index {
            base: Box::new(view.clone()),
            indices: slices,
        },
        ty: Ty::Tile(shape),
        sym: None,
        span: view.span,
    };
    let target = variable(binding, vars);
    // A stream domain is evaluated outside the repeated piece. Raw dynamic
    // endpoints must have been captured by their reaching value definition;
    // do not move their evaluation into each iteration.
    if geometry::needs_capture(&expand_views(view, available, 0)) { return Ok(None); }
    let Some(projected) = project(&sliced, &target, available, vars, project_call)? else {
        return Ok(None);
    };
    let mut projected = projected.body;
    let Some(Stmt {
        kind:
            StmtKind::Owned {
                vars: coordinates,
                body,
                ..
            },
        ..
    }) = projected.last_mut()
    else {
        return Ok(Some((Vec::new(), Vec::new(), Vec::new(), projected)));
    };
    let logical = start.add(
        variable(coordinates[axis], vars)
            .sym
            .as_ref()
            .ok_or("projected coordinate missing")?,
    );
    let mut views: Vec<Expr> = Vec::new();
    let mut bindings = Vec::new();
    let mut axes = Vec::new();
    fn reads(
        e: &mut Expr,
        logical: &Sym,
        start: &Sym,
        extent: &Sym,
        count: &Sym,
        views: &mut Vec<Expr>,
        bindings: &mut Vec<VarId>,
        axes: &mut Vec<usize>,
        vars: &mut Vec<Var>,
    ) {
        if let ExprKind::Index { base, indices } = &mut e.kind {
            if matches!(e.ty, Ty::Scalar(_)) && indices.iter().all(|i| matches!(i, Index::Point(_)))
            {
                if let Some(shape) = base.ty.shaped() {
                    if let Some(axis)=(0..shape.shape.len()).find(|axis|shape.shape.get(*axis)==Some(extent) && matches!(indices.get(*axis),Some(Index::Point(c)) if c.sym.as_ref()==Some(logical))) {
                        let existing = views.iter().enumerate().position(|(i,v)| axes[i]==axis && same_expr(v, base));
                        let v = if let Some(i) = existing {
                            bindings[i]
                        } else {
                            let v = vars.len();
                            let mut shape = shape.clone();
                            shape.shape[axis] = count.clone();
                            vars.push(Var {
                                name: format!("producer_input_{v}"),
                                ty: Ty::Tile(shape),
                                span: base.span,
                                kind: VarKind::Local,
                            });
                            views.push((**base).clone());
                            bindings.push(v);
                            axes.push(axis);
                            v
                        };
                        *base = Box::new(variable(v, vars));
                        indices[axis] = Index::Point(symbol(logical.sub(start), e.span));
                        return;
                    }
                }
            }
        }
        children_mut(e, &mut |child| {
            reads(
                child, logical, start, extent, count, views, bindings, axes, vars,
            )
        });
    }
    for statement in body {
        if let StmtKind::Assign { value, .. } = &mut statement.kind {
            reads(
                value,
                &logical,
                &start,
                &extent,
                &count,
                &mut views,
                &mut bindings,
                &mut axes,
                vars,
            );
        }
    }
    Ok(Some((views, bindings, axes, projected)))
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
fn tensor(e: &Expr, available: &HashMap<VarId, Producer>, depth: usize) -> Option<Expr> {
    if depth > available.len() + 1 {
        return None;
    }
    if matches!(e.ty, Ty::Tensor(_)) {
        return Some(e.clone());
    }
    let kind = match &e.kind {
        ExprKind::Var(v) => {
            return match available.get(v)? {
                Producer::Load(view) => Some(view.clone()),
                Producer::View(view) => tensor(view, available, depth + 1),
                _ => None,
            };
        }
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(tensor(base, available, depth + 1)?),
            indices: indices.clone(),
        },
        ExprKind::Transpose(base) => {
            ExprKind::Transpose(Box::new(tensor(base, available, depth + 1)?))
        }
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => {
            let mut args = args.clone();
            args[0] = tensor(&args[0], available, depth + 1)?;
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            }
        }
        _ => return None,
    };
    Some(Expr {
        kind,
        ty: Ty::Tensor(e.ty.shaped()?.clone()),
        sym: None,
        span: e.span,
    })
}
fn metadata(e:&mut Expr,available:&HashMap<VarId,Producer>){
    if let ExprKind::Builtin{name:Builtin::Extent,args}=&mut e.kind{
        if let Some(view)=args.first().and_then(|v|tensor(v,available,0)){args[0]=view;}
    }
    children_mut(e,&mut |e|metadata(e,available));
}
fn expand_views(e: &Expr, available: &HashMap<VarId, Producer>, depth: usize) -> Expr {
    if depth > available.len() {
        return e.clone();
    }
    if let ExprKind::Var(v) = e.kind {
        if let Some(Producer::View(view)) = available.get(&v) {
            return expand_views(view, available, depth + 1);
        }
    }
    let mut expanded = e.clone();
    match &mut expanded.kind {
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => {
            **base = expand_views(base, available, depth + 1)
        }
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => args[0] = expand_views(&args[0], available, depth + 1),
        _ => {}
    }
    expanded
}
fn project(
    value: &Expr,
    target: &Expr,
    available: &HashMap<VarId, Producer>,
    vars: &mut Vec<Var>,
    project_call: &mut super::regions::ProjectCall<'_>,
) -> Result<Option<Projection>, String> {
    let span = target.span;
    // Follow only the reaching, still-stable snapshot definitions already owned
    // by this environment. Preserve each ordinary view operation and its shape.
    let value = expand_views(value, available, 0);
    if let Some(view) = tensor(&value, available, 0) {
        return Ok(Some(Projection { view: value, body: vec![Stmt {
            id: None,
            span,
            kind: StmtKind::Assign {
                target: target.clone(),
                op: AssignOp::Assign,
                value: Expr {
                    kind: ExprKind::Builtin {
                        name: Builtin::Load,
                        args: vec![view],
                    },
                    ty: target.ty.clone(),
                    sym: None,
                    span,
                },
            },
        }] }));
    }
    let (value, mut body) = if geometry::needs_capture(&value) {
        let Some(captured) = geometry::capture(&value, vars) else { return Ok(None); };
        captured
    } else { (value, Vec::new()) };
    if let Some(Producer::Region { output, body: region }) = root(&value).and_then(|v| available.get(&v)) {
        let Some(projected) = super::regions::project(region, *output, &value, target, vars, project_call)? else {
            return Ok(None);
        };
        body.extend(projected);
        return Ok(Some(Projection { body, view: value }));
    }
    let shape = &target
        .ty
        .shaped()
        .ok_or("projected producer must be shaped")?
        .shape;
    let Elem::Dtype(dtype) = target.ty.shaped().unwrap().elem else {
        return Err("computed packed producer cannot be projected as raw storage".into());
    };
    let mut indices = Vec::new();
    for _ in shape {
        let v = vars.len();
        vars.push(Var {
            name: format!("projection_{v}"),
            ty: Ty::Scalar(DType::I32),
            span,
            kind: VarKind::Index(Atom::Param(format!("projection#{v}"))),
        });
        indices.push(v);
    }
    let coords = indices
        .iter()
        .map(|v| variable(*v, vars))
        .collect::<Vec<_>>();
    let Some(rhs) = element(&value, &coords, available, vars, 0)? else {
        return Ok(None);
    };
    let lhs = Expr {
        kind: ExprKind::Index {
            base: Box::new(target.clone()),
            indices: coords.iter().cloned().map(Index::Point).collect(),
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span,
    };
    body.extend([
        Stmt {
            id: None,
            span,
            kind: StmtKind::Assign {
                target: target.clone(),
                op: AssignOp::Assign,
                value: Expr {
                    kind: ExprKind::TileAlloc {
                        shape: shape.clone(),
                        dtype: Elem::Dtype(dtype),
                    },
                    ty: target.ty.clone(),
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
                tile: target.clone(),
                body: vec![Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Assign {
                        target: lhs,
                        op: AssignOp::Assign,
                        value: cast(rhs, dtype),
                    },
                }],
            },
        },
    ]);
    Ok(Some(Projection { body, view: value }))
}
fn cast(e: Expr, dtype: DType) -> Expr {
    Expr {
        span: e.span,
        kind: ExprKind::Cast {
            dtype,
            expr: Box::new(e),
        },
        ty: Ty::Scalar(dtype),
        sym: None,
    }
}
fn element(
    e: &Expr,
    coords: &[Expr],
    available: &HashMap<VarId, Producer>,
    vars: &[Var],
    depth: usize,
) -> Result<Option<Expr>, String> {
    if depth > available.len() + 8 {
        return Err("cyclic logical producer".into());
    }
    if matches!(e.ty, Ty::Tensor(_)) {
        return read(e, coords).map(Some);
    }
    match &e.kind {
        ExprKind::Var(v) => {
            if let Some(p) = available.get(v) {
                match p {
                    Producer::Load(view) => return element(view, coords, &HashMap::new(), vars, 0),
                    Producer::View(view) => {
                        return element(view, coords, available, vars, depth + 1);
                    }
                    Producer::Region { .. } => return read(e, coords).map(Some),
                    Producer::Pointwise {
                        indices,
                        value,
                        dtype,
                        ..
                    } => {
                        let map = indices
                            .iter()
                            .copied()
                            .zip(coords.iter().cloned())
                            .collect::<HashMap<_, _>>();
                        let atoms = indices
                            .iter()
                            .zip(coords)
                            .filter_map(|(v, c)| match &vars[*v].kind {
                                VarKind::Index(Atom::Param(a)) => Some((a.clone(), c.sym.clone())),
                                _ => None,
                            })
                            .collect();
                        let mut value = crate::lower::subst_vars(value, &map, &atoms);
                        if !expand_elements(&mut value, available, vars, depth + 1)? {
                            return Ok(None);
                        }
                        return Ok(Some(cast(value, *dtype)));
                    }
                }
            }
        }
        ExprKind::Index { base, indices } => {
            // Symbolic coordinates can be substituted into a pointwise body.
            // Data-dependent views also establish clamped extents, evaluate
            // endpoints once, and enforce point guards. Until that geometry is
            // retained by this projection path, only its materialized form is
            // constructible. Tensor-rooted views above use ordinary view
            // evaluation and do not have this restriction.
            if indices.iter().any(|index| match index {
                Index::Point(point) => point.sym.is_none(),
                Index::Slice { start, end } => {
                    start.iter().chain(end).any(|bound| bound.sym.is_none())
                }
            }) {
                return Ok(None);
            }
            let mut mapped = Vec::new();
            let mut n = 0;
            for axis in 0..base
                .ty
                .shaped()
                .ok_or("projection base shape missing")?
                .shape
                .len()
            {
                match indices.get(axis) {
                    Some(Index::Point(e)) => mapped.push(e.clone()),
                    ix => {
                        let coordinate =
                            coords.get(n).ok_or("projection coordinate rank mismatch")?;
                        n += 1;
                        let coordinate = match ix {
                            Some(Index::Slice {
                                start: Some(start), ..
                            }) => {
                                let Some(coordinate) = coordinate.sym.as_ref() else {
                                    return Ok(None);
                                };
                                symbol(
                                    start
                                        .sym
                                        .as_ref()
                                        .expect("symbolic slice domain")
                                        .add(coordinate),
                                    e.span,
                                )
                            }
                            _ => coordinate.clone(),
                        };
                        mapped.push(coordinate);
                    }
                }
            }
            return element(base, &mapped, available, vars, depth + 1);
        }
        ExprKind::Transpose(base) => {
            let mut c = coords.to_vec();
            c.reverse();
            return element(base, &c, available, vars, depth + 1);
        }
        ExprKind::Builtin {
            name: Builtin::Reshape,
            args,
        } => {
            let mut linear = Sym::constant(0);
            for (c, n) in coords.iter().zip(&e.ty.shaped().unwrap().shape) {
                let Some(coordinate) = c.sym.as_ref() else {
                    return Ok(None);
                };
                linear = linear.mul(n).add(coordinate);
            }
            let mut c = Vec::new();
            for n in args[0].ty.shaped().unwrap().shape.iter().rev() {
                c.push(symbol(linear.rem(n), e.span));
                linear = linear.quot(n);
            }
            c.reverse();
            return element(&args[0], &c, available, vars, depth + 1);
        }
        _ => {}
    }
    read(e, coords).map(Some)
}
fn read(e: &Expr, coords: &[Expr]) -> Result<Expr, String> {
    let dtype =
        e.ty.shaped()
            .and_then(|s| s.elem.read_dtype())
            .ok_or("producer element has no readable type")?;
    Ok(Expr {
        kind: ExprKind::Index {
            base: Box::new(e.clone()),
            indices: coords.iter().cloned().map(Index::Point).collect(),
        },
        ty: Ty::Scalar(dtype),
        sym: None,
        span: e.span,
    })
}

fn expand_elements(
    e: &mut Expr,
    available: &HashMap<VarId, Producer>,
    vars: &[Var],
    depth: usize,
) -> Result<bool, String> {
    if let ExprKind::Index { base, indices } = &e.kind {
        if matches!(e.ty, Ty::Scalar(_)) && indices.iter().all(|i| matches!(i, Index::Point(_))) {
            let coords = indices
                .iter()
                .map(|i| match i {
                    Index::Point(e) => e.clone(),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            let Some(projected) = element(base, &coords, available, vars, depth)? else {
                return Ok(false);
            };
            *e = projected;
            return Ok(true);
        }
    }
    let mut error = None;
    let mut supported = true;
    children_mut(e, &mut |child| {
        if error.is_none() && supported {
            match expand_elements(child, available, vars, depth) {
                Ok(child_supported) => supported = child_supported,
                Err(child_error) => error = Some(child_error),
            }
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(supported)
}
fn removable_definitions(
    body: &[Stmt],
    variable: VarId,
    vars: &[Var],
    program: &crate::program::Program,
    root_body: &[Stmt],
) -> bool {
    body.iter().all(|s| match &s.kind {
        StmtKind::Assign { target, value, .. } if root(target) == Some(variable) => {
            matches!(target.kind, ExprKind::Var(_)) && {
                let mut effect = false;
                walk(value, &mut |e| {
                    effect |= matches!(
                        e.kind,
                        ExprKind::Call { .. }
                            | ExprKind::Intrinsic { .. }
                            | ExprKind::Builtin {
                                name: Builtin::Store | Builtin::Atomic,
                                ..
                            }
                    )
                });
                !effect
            }
        }
        StmtKind::Owned {
            tile,
            body: definition,
            ..
        } if matches!(tile.kind,ExprKind::Var(v) if v==variable) => {
            let mut written = HashSet::new();
            for statement in definition {
                crate::rewrite::writes(statement, &mut written);
            }
            !definition.iter().any(crate::effects::tensor_effect)
                && written.iter().all(|v| {
                    *v == variable
                        || (!matches!(vars[*v].kind, VarKind::Param(_))
                            && !uses_except(root_body, s, *v))
                })
        }
        StmtKind::Reduction(r) => {
            !r.state_variables().any(|v| v == variable)
                && r.bodies()
                    .all(|b| removable_definitions(b, variable, vars, program, root_body))
        }
        StmtKind::Range { body, .. }
        | StmtKind::Owned { body, .. }
        | StmtKind::Parallel { body, .. }
        | StmtKind::LoadLoop { body, .. }
        | StmtKind::Lanes { body, .. } => {
            removable_definitions(body, variable, vars, program, root_body)
        }
        StmtKind::If { then, els, .. } => {
            removable_definitions(then, variable, vars, program, root_body)
                && removable_definitions(els, variable, vars, program, root_body)
        }
        _ => true,
    })
}
fn used_outside_definition(
    body: &[Stmt],
    variable: VarId,
    program: &crate::program::Program,
) -> bool {
    body.iter().any(|s|match &s.kind{
        StmtKind::Assign{target,value,..} if matches!(target.kind,ExprKind::Var(v) if v==variable)=>mentions(value,variable),
        StmtKind::Owned{tile,..} if matches!(tile.kind,ExprKind::Var(v) if v==variable)=>false,
        StmtKind::Range{body,..}|StmtKind::Parallel{body,..}|StmtKind::Owned{body,..}|StmtKind::Lanes{body,..}=>used_outside_definition(body,variable,program),
        StmtKind::LoadLoop{domain,views,body,..}=>mentions(&domain.view,variable)||views.iter().any(|e|mentions(e,variable))||used_outside_definition(body,variable,program),
        StmtKind::If{cond,then,els}=>mentions(cond,variable)||used_outside_definition(then,variable,program)||used_outside_definition(els,variable,program),
        StmtKind::Expr(Expr{kind:ExprKind::Call{..},..}) if facts(s,program)==(HashSet::from([variable]),false)=>false,
        _=>crate::effects::uses(s,variable),
    })
}
fn remove(body: &mut Vec<Stmt>, dead: &HashSet<VarId>, program: &crate::program::Program) {
    body.retain(|s| match &s.kind {
        StmtKind::Assign { target, .. } => {
            !matches!(target.kind,ExprKind::Var(v) if dead.contains(&v))
        }
        StmtKind::Owned { tile, .. } => !matches!(tile.kind,ExprKind::Var(v) if dead.contains(&v)),
        StmtKind::Expr(Expr {
            kind: ExprKind::Call { .. },
            ..
        }) => {
            let (writes, effect) = facts(s, program);
            effect || writes.is_empty() || !writes.iter().all(|v| dead.contains(v))
        }
        _ => true,
    });
    for s in body {
        nested_mut(s, &mut |b| remove(b, dead, program));
    }
}

fn facts(statement: &Stmt, program: &crate::program::Program) -> (HashSet<VarId>, bool) {
    let mut written = HashSet::new();
    crate::rewrite::writes(statement, &mut written);
    let mut effect = false;
    visit_stmt(statement, &mut |e| {
        walk(e, &mut |e| match &e.kind {
            ExprKind::Call { args, .. } => match crate::lower::call_effects(program, e) {
                Some((writes, external)) => {
                    written.extend(writes);
                    effect |= external;
                }
                None => {
                    effect = true;
                    for e in args {
                        if let Some(v) = root(e) {
                            written.insert(v);
                        }
                    }
                }
            },
            ExprKind::Builtin {
                name: Builtin::Store | Builtin::Atomic,
                ..
            } => effect = true,
            ExprKind::Intrinsic { op, .. } => effect |= op.writes_tensor_memory(),
            _ => {}
        })
    });
    (written, effect)
}

fn uses_except(body: &[Stmt], excluded: &Stmt, variable: VarId) -> bool {
    body.iter().any(|statement| {
        if std::ptr::eq(statement, excluded) {
            return false;
        }
        let direct = match &statement.kind {
            StmtKind::Assign { target, value, .. } => {
                mentions(target, variable) || mentions(value, variable)
            }
            StmtKind::Expr(e) => mentions(e, variable),
            StmtKind::Owned { tile, .. } => mentions(tile, variable),
            StmtKind::LoadLoop { domain, views, .. } => {
                mentions(&domain.view, variable) || views.iter().any(|e| mentions(e, variable))
            }
            StmtKind::If { cond, .. } => mentions(cond, variable),
            StmtKind::Reduction(r) => r.operands().any(|e| mentions(e, variable)),
            _ => false,
        };
        direct
            || match &statement.kind {
                StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Parallel { body, .. }
                | StmtKind::LoadLoop { body, .. }
                | StmtKind::Lanes { body, .. } => uses_except(body, excluded, variable),
                StmtKind::If { then, els, .. } => {
                    uses_except(then, excluded, variable) || uses_except(els, excluded, variable)
                }
                StmtKind::Reduction(r) => r.bodies().any(|b| uses_except(b, excluded, variable)),
                _ => false,
            }
    })
}
