//! Lowering by inlining: for a target backend and concrete shapes, replace every
//! construct call by the selected `lower` block (or the construct's portable body)
//! until only primitives and intrinsics remain.

use crate::ir::*;
use crate::lowered_ir::*;
use crate::program::Program;
use crate::sym::{Atom, Sym};
use crate::types::{Elem, Shaped, Ty};
use std::collections::HashMap;
pub mod alternatives;
pub mod family;
mod decomposition;
mod grouping;
mod projection;
pub(crate) fn call_effects(
    program: &Program,
    call: &Expr,
) -> Option<(std::collections::HashSet<VarId>, bool)> {
    projection::effects(program, call)
}

/// Explicit restrictions on the lowering space.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Restrict every stream to this capacity. `None` exposes every capacity up
    /// to its proven backing extent. Diagnostic lowering chooses the whole axis.
    pub piece: Option<i64>,
    /// Invocation-owned intermediates, disjoint from every external argument.
    /// Their initial contents and final storage are not observable; required
    /// numerical conversion at each source publication remains observable.
    pub ownership: crate::composition::Ownership,
}

pub fn lower(
    program: &Program,
    name: &str,
    backend: &str,
    shapes: &HashMap<String, i64>,
) -> Result<LoweredIr, String> {
    lower_with(program, name, backend, shapes, &Options::default())
}

pub fn lower_with(
    program: &Program,
    name: &str,
    backend: &str,
    shapes: &HashMap<String, i64>,
    opts: &Options,
) -> Result<LoweredIr, String> {
    lower_specialized(program, name, backend, shapes, &HashMap::new(), opts)
}

/// Bind entry element parameters as well as shapes before choosing lowerings.
pub fn lower_specialized(
    program: &Program,
    name: &str,
    backend: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
    opts: &Options,
) -> Result<LoweredIr, String> {
    lower_selected(
        program,
        name,
        backend,
        shapes,
        elements,
        opts,
        &mut |decision| {
            // Deterministic diagnostic baseline only. It makes no optimality claim.
            decision
                .alternatives
                .get(0)
                .ok_or_else(|| format!("empty decision domain on `{backend}`: {:?}", decision.kind))
        },
    )
}

/// Expand only choices made by the caller, after checking their applicability.
/// Every construct decision is exposed, including singleton domains and portable
/// alternatives. An invalid choice fails; there is no silent preferred fallback.
#[allow(clippy::too_many_arguments)]
pub fn lower_selected(
    program: &Program,
    name: &str,
    backend: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
    opts: &Options,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<LoweredIr, String> {
    let (mut lowered, bodies) = prepare_selected(program, name, backend, shapes, elements, opts, select)?;
    let mut stage = LoweringStage::Bodies(bodies);
    loop {
        finish_stage(&mut lowered, program, &stage, select)?;
        let Some(next) = stage.next() else { break; };
        stage = next;
    }
    Ok(lowered)
}

/// Retained boundary after decomposition, partitioning and producer projection.
/// Backend bodies and composition refine these same calls and captured facts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_selected(
    program: &Program,
    name: &str,
    backend: &str,
    shapes: &HashMap<String, i64>,
    elements: &HashMap<String, Elem>,
    opts: &Options,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(LoweredIr, BodyResolution), String> {
    if opts.piece.is_some_and(|capacity| capacity <= 0) {
        return Err("stream piece capacity must be positive".into());
    }
    let f = program
        .functions
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("no function `{name}`"))?;
    let mut decisions = Vec::new();
    let mut recording = |domain: &Decision| {
        let selected = select(domain)?;
        if !domain.alternatives.contains(&selected) {
            return Err(format!(
                "selected alternative {selected:?} is not applicable to {:?}",
                domain.kind
            ));
        }
        decisions.push(DecisionRecord {
            domain: domain.clone(),
            selected: selected.clone(),
        });
        Ok(selected)
    };
    let mut ctx = Inliner {
        program,
        backend,
        select: &mut recording,
        selections: Vec::new(),
        counter: 0,
        opts: opts.clone(),
        piece_values: HashMap::new(),
        elements: elements.clone(),
        calls: CallStage::Retain,
        partitioning: std::collections::HashSet::new(),
        domains: HashMap::new(),
        view_domains: HashMap::new(),
    };
    let env: HashMap<String, Sym> = shapes
        .iter()
        .map(|(k, v)| (k.clone(), Sym::constant(*v)))
        .collect();
    for p in &f.shape_params {
        if !shapes.contains_key(p) {
            return Err(format!("shape parameter `{p}` of `{name}` is not bound"));
        }
    }
    crate::program::validate_element_bindings(f, elements)?;
    let mut vars: Vec<Var> = f
        .vars
        .iter()
        .map(|v| Var {
            ty: subst_elem_ty(&subst_ty(&v.ty, &env), elements),
            ..v.clone()
        })
        .collect();
    let body = ctx.inline_block(
        &f.body,
        &env,
        &HashMap::new(),
        &mut vars,
        &mut HashMap::new(),
        0,
    )?;
    let params = f
        .params
        .iter()
        .map(|(n, t)| (n.clone(), subst_elem_ty(&subst_ty(t, &env), elements)))
        .collect();
    let index_params = f
        .index_params
        .iter()
        .map(|(name, bound)| (name.clone(), subst_sym(bound, &env, &HashMap::new())))
        .collect();
    let selections = std::mem::take(&mut ctx.selections);
    let mut counter = ctx.counter;
    let mut piece_values = std::mem::take(&mut ctx.piece_values);
    let mut domains = std::mem::take(&mut ctx.domains);
    let mut view_domains = std::mem::take(&mut ctx.view_domains);
    drop(ctx);
    let mut lowered = LoweredIr {
        name: name.to_string(),
        backend: backend.to_string(),
        ownership: opts.ownership.clone(),
        alias_requirements: Vec::new(),
        params,
        index_params,
        vars,
        body,
        shapes: shapes.clone(),
        selections: Vec::new(),
        decisions: Vec::new(),
    };
    crate::normalize::bind_values(&mut lowered.body, &mut lowered.vars);
    crate::reduction::structured::primitive::retain(
        &mut lowered.body,
        &mut lowered.vars,
        &mut recording,
    )?;
    decomposition::select(&mut lowered, opts, &mut recording)?;
    grouping::select(&mut lowered, program, &mut recording)?;
    // Group logical output calls before choosing contraction capacities. Keep
    // the partitioned calls visible for producer projection and body selection.
    let mut partitioner = Inliner {
        program,
        backend,
        select: &mut recording,
        selections: Vec::new(),
        counter,
        opts: opts.clone(),
        piece_values,
        elements: elements.clone(),
        calls: CallStage::Partition,
        partitioning: std::collections::HashSet::new(),
        domains,
        view_domains,
    };
    partitioner.resolve_calls(&mut lowered.body, &mut lowered.vars, &[])?;
    counter = partitioner.counter;
    piece_values = std::mem::take(&mut partitioner.piece_values);
    domains = std::mem::take(&mut partitioner.domains);
    view_domains = std::mem::take(&mut partitioner.view_domains);
    drop(partitioner);
    crate::composition::project_producers(
        &mut lowered,
        program,
        &mut recording,
        &mut |call, output, view, target, vars| {
            projection::call(program, call, output, view, target, vars)
        },
    )?;
    lowered.selections = selections;
    drop(recording);
    lowered.decisions = decisions;
    crate::verify::lowered(&lowered, crate::verify::Stage::Decomposed)?;
    Ok((lowered, BodyResolution {
        counter,
        options: opts.clone(),
        elements: elements.clone(),
        piece_values,
        domains,
        view_domains,
    }))
}

/// Lexical lowering facts travel with the retained calls. They are the original
/// inliner's state, not a separately reconstructed interpretation of the source.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BodyResolution {
    counter: usize,
    options: Options,
    elements: HashMap<String, Elem>,
    piece_values: HashMap<String, Vec<i64>>,
    domains: HashMap<String, Vec<decomposition::Domain>>,
    view_domains: HashMap<Sym, (i64, IterationDomain)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LoweringStage {
    Bodies(BodyResolution),
    Values,
    Representations,
}
impl LoweringStage {
    pub(crate) fn next(&self) -> Option<Self> {
        match self {
            Self::Bodies(_) => Some(Self::Values),
            Self::Values => Some(Self::Representations),
            Self::Representations => None,
        }
    }
}

pub(crate) fn finish_stage(
    lowered: &mut LoweredIr,
    program: &Program,
    stage: &LoweringStage,
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    let mut decisions = std::mem::take(&mut lowered.decisions);
    let mut recording = |domain: &Decision| {
        let selected = select(domain)?;
        if !domain.alternatives.contains(&selected) {
            return Err(format!(
                "selected alternative {selected:?} is not applicable to {:?}",
                domain.kind
            ));
        }
        decisions.push(DecisionRecord {
            domain: domain.clone(),
            selected: selected.clone(),
        });
        Ok(selected)
    };
    match stage {
        LoweringStage::Bodies(state) => {
            let mut ctx = Inliner {
                program,
                backend: &lowered.backend,
                select: &mut recording,
                selections: Vec::new(),
                counter: state.counter,
                opts: state.options.clone(),
                piece_values: state.piece_values.clone(),
                elements: state.elements.clone(),
                calls: CallStage::Resolve,
                partitioning: std::collections::HashSet::new(),
                domains: state.domains.clone(),
                view_domains: state.view_domains.clone(),
            };
            ctx.resolve_calls(&mut lowered.body, &mut lowered.vars, &[])?;
            lowered.selections.extend(ctx.selections);
        }
        LoweringStage::Values => {
            let ownership = lowered.ownership.clone();
            crate::composition::select(
                lowered,
                program,
                &ownership,
                &mut recording,
                &mut |call, output, view, target, vars| {
                    projection::call(program, call, output, view, target, vars)
                },
            )?;
        }
        LoweringStage::Representations => {
            crate::composition::select_representations(lowered, &mut recording)?;
            select_producers(&mut lowered.body, &lowered.vars, &mut recording)?;
        }
    }
    lowered.decisions = decisions;
    crate::verify::lowered(lowered, crate::verify::Stage::Expanded)?;
    Ok(())
}

/// A slice cannot exceed its parent axis. Follow that structural bound rather
/// than assigning a performance-dependent default to a runtime-sized view.
pub fn view_axis_capacity(view: &Expr, axis: usize) -> Result<i64, String> {
    let shaped = view.ty.shaped().ok_or("stream requires a shaped view")?;
    let extent = shaped.shape.get(axis).ok_or("invalid stream axis")?;
    if let Some(n) = extent.as_constant() {
        return if n >= 0 {
            Ok(n)
        } else {
            Err("negative stream extent".into())
        };
    }
    match &view.kind {
        ExprKind::Index { base, indices } => {
            let rank = base
                .ty
                .shaped()
                .ok_or("indexed stream requires a shaped parent")?
                .shape
                .len();
            let parent_axis = (0..rank)
                .filter(|i| !matches!(indices.get(*i), Some(Index::Point(_))))
                .nth(axis)
                .ok_or("invalid indexed stream axis")?;
            view_axis_capacity(base, parent_axis)
        }
        ExprKind::Transpose(base) => {
            let rank = shaped.shape.len();
            view_axis_capacity(base, rank - 1 - axis)
        }
        _ => Err(format!(
            "stream axis `{extent}` has no proven static capacity"
        )),
    }
}

/// Construct calls stay semantic until output grouping, contraction partitioning,
/// and producer projection have all had access to their checked definitions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CallStage {
    /// Expand portable definitions to derive semantic facts, without decisions.
    Portable,
    /// Expand transparent helpers but retain logical construct invocations.
    Retain,
    /// Choose contraction domains while preserving the resulting calls.
    Partition,
    /// Select backend bodies after logical transformations.
    Resolve,
}

struct Inliner<'a> {
    program: &'a Program,
    backend: &'a str,
    select: &'a mut dyn FnMut(&Decision) -> Result<Alternative, String>,
    selections: Vec<Selection>,
    counter: usize,
    opts: Options,
    elements: HashMap<String, Elem>,
    calls: CallStage,
    partitioning: std::collections::HashSet<(String, String)>,
    domains: HashMap<String, Vec<decomposition::Domain>>,
    view_domains: HashMap<Sym, (i64, IterationDomain)>,
    /// Piece atoms over static extents: the concrete extents their pieces take (the capacity
    /// and the tail), so residuals can be decided exactly.
    piece_values: HashMap<String, Vec<i64>>,
}

/// Maps a callee's variable ids to expressions in the caller (parameters bound to arguments,
/// locals renamed into the caller's table).
type VarMap = HashMap<VarId, Expr>;

impl<'a> Inliner<'a> {
    fn inline_block(
        &mut self,
        stmts: &[Stmt],
        env: &HashMap<String, Sym>,
        vmap: &VarMap,
        vars: &mut Vec<Var>,
        atom_map: &mut HashMap<String, Atom>,
        depth: usize,
    ) -> Result<Vec<Stmt>, String> {
        if depth > 32 {
            return Err("lowering recursion too deep".into());
        }
        let saved_domains = self.view_domains.clone();
        let mut out = Vec::new();
        for s in stmts {
            out.extend(self.inline_stmt(s, env, vmap, vars, atom_map, depth)?);
        }
        self.view_domains = saved_domains;
        Ok(out)
    }

    fn inline_stmt(
        &mut self,
        s: &Stmt,
        env: &HashMap<String, Sym>,
        vmap: &VarMap,
        vars: &mut Vec<Var>,
        atom_map: &mut HashMap<String, Atom>,
        depth: usize,
    ) -> Result<Vec<Stmt>, String> {
        let span = s.span;
        let kind = match &s.kind {
            StmtKind::Reduction(_) => {
                return Err("a retained reduction is already specialized".into());
            }
            StmtKind::Parallel {
                vars: vs,
                extents,
                body,
            } => StmtKind::Parallel {
                vars: vs.iter().map(|v| remap_var(*v, vmap)).collect(),
                extents: extents
                    .iter()
                    .map(|e| subst_sym(e, env, atom_map))
                    .collect(),
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::LoadLoop {
                domain,
                offset,
                vars: vs,
                views,
                axes,
                piece,
                body,
                ..
            } => {
                let views: Vec<Expr> = views
                    .iter()
                    .map(|v| self.inline_expr(v, env, vmap, vars, atom_map))
                    .collect::<Result<_, _>>()?;
                let domain = IterationDomain {
                    view: self.inline_expr(&domain.view, env, vmap, vars, atom_map)?,
                    axis: domain.axis,
                };
                if self.calls == CallStage::Portable {
                    return Ok(vec![Stmt { id: None, span, kind: StmtKind::LoadLoop {
                        domain, offset: offset.map(|v| remap_var(v, vmap)), modes: None,
                        vars: vs.iter().map(|v| remap_var(*v, vmap)).collect(), views,
                        axes: axes.clone(), piece: remap_atom(piece, atom_map), capacity: None,
                        body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
                    } }]);
                }
                let extent = domain
                    .view
                    .ty
                    .shaped()
                    .and_then(|s| s.shape.get(domain.axis))
                    .ok_or("invalid iteration domain")?
                    .clone();
                let Atom::Param(pname) = piece else {
                    unreachable!()
                };
                let mut inner_env = env.clone();
                let maximum = match extent.as_constant() {
                    Some(n) if n >= 0 => n.max(1),
                    Some(_) => return Err("negative stream extent".into()),
                    None => view_axis_capacity(&domain.view, domain.axis)?.max(1),
                };
                let decision = Decision {
                    kind: DecisionKind::Stream {
                        piece: remap_atom(piece, atom_map),
                        extent: extent.clone(),
                        maximum,
                    },
                    alternatives: match self.opts.piece {
                        Some(n) => vec![Alternative::StreamCapacity(n)].into(),
                        None => Alternatives::stream_capacities(maximum)?,
                    },
                };
                let Alternative::StreamCapacity(selected) = (self.select)(&decision)? else {
                    return Err("stream decision requires a capacity".into());
                };
                let capacity = match (extent.as_constant(), Some(selected)) {
                    (Some(e), Some(c)) if e > c => {
                        // Static extent chunked: pieces of `c` and a tail of `e % c`.
                        let mut values = vec![c];
                        if e % c != 0 {
                            values.push(e % c);
                        }
                        self.piece_values.insert(pname.clone(), values);
                        Some(c)
                    }
                    (Some(_), _) => {
                        inner_env.insert(pname.clone(), extent);
                        None
                    }
                    // Dynamic extent: static pieces of a chosen capacity; the atom stays symbolic.
                    (None, explicit) => {
                        let bound = view_axis_capacity(&domain.view, domain.axis)?;
                        // Even an empty backing axis needs a positive loop step; the
                        // runtime domain remains empty and performs no reads.
                        Some(explicit.unwrap_or(bound.max(1)))
                    }
                };
                let new_vars: Vec<VarId> = vs.iter().map(|v| remap_var(*v, vmap)).collect();
                for v in &new_vars {
                    vars[*v].ty = subst_ty(&vars[*v].ty, &inner_env);
                }
                StmtKind::LoadLoop {
                    domain,
                    offset: offset.map(|v| remap_var(v, vmap)),
                    modes: None,
                    vars: new_vars,
                    views,
                    axes: axes.clone(),
                    piece: remap_atom(piece, atom_map),
                    capacity,
                    body: self.inline_block(body, &inner_env, vmap, vars, atom_map, depth)?,
                }
            }
            StmtKind::Owned {
                vars: vs,
                tile,
                body,
            } => StmtKind::Owned {
                vars: vs.iter().map(|v| remap_var(*v, vmap)).collect(),
                tile: self.inline_expr(tile, env, vmap, vars, atom_map)?,
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Range { var, lo, hi, body } => StmtKind::Range {
                var: remap_var(*var, vmap),
                lo: subst_sym(lo, env, atom_map),
                hi: subst_sym(hi, env, atom_map),
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Lanes {
                var,
                extent,
                width,
                body,
            } => StmtKind::Lanes {
                var: remap_var(*var, vmap),
                extent: subst_sym(extent, env, atom_map),
                width: *width,
                body: self.inline_block(body, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::If { cond, then, els } => StmtKind::If {
                cond: self.inline_expr(cond, env, vmap, vars, atom_map)?,
                then: self.inline_block(then, env, vmap, vars, atom_map, depth)?,
                els: self.inline_block(els, env, vmap, vars, atom_map, depth)?,
            },
            StmtKind::Assign { target, op, value } => StmtKind::Assign {
                target: self.inline_expr(target, env, vmap, vars, atom_map)?,
                op: *op,
                value: self.inline_expr(value, env, vmap, vars, atom_map)?,
            },
            StmtKind::Expr(e) => {
                if let Some(mut reduction) = crate::reduction::structured::Reduction::from_expr(e) {
                    for e in reduction.operands_mut() {
                        *e = self.inline_expr(e, env, vmap, vars, atom_map)?;
                    }
                    if self.calls == CallStage::Portable {
                        // Dependence analysis consumes the reduction contract;
                        // callback implementation and tree are still unresolved.
                        return Ok(vec![Stmt { id: None, span, kind: StmtKind::Reduction(Box::new(reduction)) }]);
                    }
                    if let ExprKind::Call {
                        shape_args,
                        elem_args,
                        ..
                    } = &mut reduction.merge.source_mut().unwrap().kind
                    {
                        *shape_args = shape_args
                            .iter()
                            .map(|s| subst_sym(s, env, atom_map))
                            .collect();
                        *elem_args = elem_args
                            .iter()
                            .map(|e| subst_elem(e, &self.elements))
                            .collect();
                    }
                    let parameters: Vec<_> = (0..3)
                        .flat_map(|_| reduction.state.iter())
                        .map(|state| {
                            let id = vars.len();
                            vars.push(Var {
                                name: format!("merge_{id}"),
                                ty: state.ty.clone(),
                                span,
                                kind: VarKind::Local,
                            });
                            Expr {
                                kind: ExprKind::Var(id),
                                ty: state.ty.clone(),
                                sym: None,
                                span,
                            }
                        })
                        .collect();
                    let ExprKind::Call {
                        callee,
                        shape_args,
                        elem_args,
                        ..
                    } = &reduction.merge.source_mut().unwrap().kind
                    else {
                        unreachable!()
                    };
                    let body = self.inline_call(
                        callee,
                        shape_args,
                        elem_args,
                        &parameters,
                        &HashMap::new(),
                        &HashMap::new(),
                        vars,
                        &mut HashMap::new(),
                        depth + 1,
                    )?;
                    let n = reduction.state.len();
                    reduction.implementation = Some(crate::reduction::structured::Merge {
                        left: parameters[..n].to_vec(),
                        right: parameters[n..2 * n].to_vec(),
                        output: parameters[2 * n..].to_vec(),
                        body,
                    });
                    if let Some(step) = &mut reduction.step {
                        if let ExprKind::Call {
                            shape_args,
                            elem_args,
                            ..
                        } = &mut step.call.source_mut().unwrap().kind
                        {
                            *shape_args = shape_args
                                .iter()
                                .map(|s| subst_sym(s, env, atom_map))
                                .collect();
                            *elem_args = elem_args
                                .iter()
                                .map(|e| subst_elem(e, &self.elements))
                                .collect();
                        }
                        let leaf_types: Vec<_> = reduction
                            .inputs
                            .iter()
                            .map(|e| {
                                crate::reduction::structured::slice(
                                    e,
                                    reduction.axis,
                                    &crate::reduction::structured::integer(0, span),
                                    span,
                                )
                                .ty
                            })
                            .collect();
                        let parameters: Vec<_> = reduction
                            .state
                            .iter()
                            .map(|e| &e.ty)
                            .chain(&leaf_types)
                            .chain(reduction.state.iter().map(|e| &e.ty))
                            .map(|ty| {
                                let id = vars.len();
                                vars.push(Var {
                                    name: format!("fold_{id}"),
                                    ty: ty.clone(),
                                    span,
                                    kind: VarKind::Local,
                                });
                                Expr {
                                    kind: ExprKind::Var(id),
                                    ty: ty.clone(),
                                    sym: None,
                                    span,
                                }
                            })
                            .collect();
                        let ExprKind::Call {
                            callee,
                            shape_args,
                            elem_args,
                            ..
                        } = &mut step.call.source_mut().unwrap().kind
                        else {
                            unreachable!()
                        };
                        // Removing a packed axis yields decoded scalar values.
                        // Specialize helper element parameters from those actual
                        // leaf operands, not the original physical input type.
                        let f = self
                            .program
                            .functions
                            .iter()
                            .find(|f| f.name == *callee)
                            .ok_or("missing fold step")?;
                        for (name, arg) in f.elem_params.iter().zip(elem_args.iter_mut()) {
                            for ((_, formal), actual) in f.params.iter().zip(&parameters) {
                                if matches!(formal.shaped().map(|s|&s.elem),Some(Elem::Param(p)) if p==name)
                                {
                                    *arg = actual
                                        .ty
                                        .shaped()
                                        .ok_or("fold helper parameter must be a tile")?
                                        .elem
                                        .clone();
                                    break;
                                }
                            }
                        }
                        let body = self.inline_call(
                            callee,
                            shape_args,
                            elem_args,
                            &parameters,
                            &HashMap::new(),
                            &HashMap::new(),
                            vars,
                            &mut HashMap::new(),
                            depth + 1,
                        )?;
                        let m = leaf_types.len();
                        step.implementation = Some(crate::reduction::structured::Merge {
                            left: parameters[..n].to_vec(),
                            right: parameters[n..n + m].to_vec(),
                            output: parameters[n + m..].to_vec(),
                            body,
                        });
                    }
                    let domain = Decision {
                        kind: DecisionKind::Reduction {
                            merge: reduction.merge_name().into(),
                            extent: reduction.extent().clone(),
                            fields: reduction.state.iter().map(|e| e.ty.clone()).collect(),
                        },
                        alternatives: reduction
                            .trees()
                            .into_iter()
                            .map(Alternative::ReductionTree)
                            .collect::<Vec<_>>()
                            .into(),
                    };
                    let Alternative::ReductionTree(tree) = (self.select)(&domain)? else {
                        return Err("invalid coupled reduction choice".into());
                    };
                    reduction.select_tree(tree, self.select)?;
                    return Ok(vec![Stmt {
                        id: None,
                        kind: StmtKind::Reduction(Box::new(reduction)),
                        span,
                    }]);
                }
                if let ExprKind::Call {
                    callee,
                    shape_args,
                    elem_args,
                    args,
                } = &e.kind
                {
                    let elem_args = elem_args
                        .iter()
                        .map(|e| subst_elem(e, &self.elements))
                        .collect::<Vec<_>>();
                    return self.inline_call(
                        callee, shape_args, &elem_args, args, env, vmap, vars, atom_map, depth,
                    );
                }
                StmtKind::Expr(self.inline_expr(e, env, vmap, vars, atom_map)?)
            }
        };
        Ok(vec![Stmt {
            id: None,
            kind,
            span,
        }])
    }

    #[allow(clippy::too_many_arguments)]
    fn inline_call(
        &mut self,
        callee: &str,
        shape_args: &[Sym],
        elem_args: &[Elem],
        args: &[Expr],
        env: &HashMap<String, Sym>,
        vmap: &VarMap,
        vars: &mut Vec<Var>,
        atom_map: &mut HashMap<String, Atom>,
        depth: usize,
    ) -> Result<Vec<Stmt>, String> {
        let f = self
            .program
            .functions
            .iter()
            .find(|f| f.name == callee)
            .ok_or_else(|| format!("no function `{callee}`"))?;
        // Concrete shape arguments in the caller's environment.
        // Shape arguments may stay symbolic when they carry a piece extent; a block whose
        // residual depends on such an argument is then not applicable and the portable body is used.
        let mut inner_env: HashMap<String, Sym> = HashMap::new();
        let mut concrete = Vec::new();
        for (p, s) in f.shape_params.iter().zip(shape_args) {
            let v = subst_sym(s, env, atom_map);
            concrete.push(v.as_constant().unwrap_or(-1));
            inner_env.insert(p.clone(), v);
        }
        let inlined_args: Vec<Expr> = args
            .iter()
            .map(|a| self.inline_expr(a, env, vmap, vars, atom_map))
            .collect::<Result<_, _>>()?;
        // Logical domain decisions precede backend-body specialization. The
        // domain comes from the visible portable state/dataflow, not a name.
        if f.is_construct && matches!(self.calls, CallStage::Partition | CallStage::Resolve) {
            if !self.domains.contains_key(callee) {
                let (logical_vars, logical_body) = portable_body(self.program, f)?;
                self.domains.insert(
                    callee.to_owned(),
                    decomposition::domains(f, &logical_vars, &logical_body),
                );
            }
            for domain in self.domains[callee].clone() {
                if self
                    .partitioning
                    .contains(&(callee.to_owned(), domain.parameter.clone()))
                {
                    continue;
                }
                let logical_extent = inner_env[&domain.parameter].clone();
                let source_domain = self.view_domains.get(&logical_extent).cloned();
                let maximum = logical_extent
                    .as_constant()
                    .or_else(|| source_domain.as_ref().map(|(capacity, _)| *capacity));
                let Some(maximum) = maximum.filter(|n| *n > 0) else {
                    continue;
                };
                let mut input_roots = std::collections::HashSet::new();
                let mut state_roots = std::collections::HashSet::new();
                for (argument, axis) in inlined_args.iter().zip(&domain.axes) {
                    let mut root = argument;
                    while let ExprKind::Index { base, .. } | ExprKind::Transpose(base) = &root.kind
                    {
                        root = base;
                    }
                    if let ExprKind::Var(v) = root.kind {
                        if axis.is_some() {
                            input_roots.insert(v);
                        } else {
                            state_roots.insert(v);
                        }
                    }
                }
                if !input_roots.is_disjoint(&state_roots) {
                    continue;
                }
                let piece = Atom::Param(format!("domain_{}", self.fresh()));
                let decision = Decision {
                    kind: DecisionKind::Stream {
                        piece,
                        extent: logical_extent.clone(),
                        maximum,
                    },
                    alternatives: match self.opts.piece {
                        Some(n) => vec![Alternative::StreamCapacity(n.min(maximum))].into(),
                        None => Alternatives::stream_capacities(maximum)?,
                    },
                };
                let Alternative::StreamCapacity(capacity) = (self.select)(&decision)? else {
                    return Err("logical decomposition needs a capacity".into());
                };
                if !decision
                    .alternatives
                    .contains(&Alternative::StreamCapacity(capacity))
                {
                    return Err("logical capacity outside domain".into());
                }
                let position = f
                    .shape_params
                    .iter()
                    .position(|p| p == &domain.parameter)
                    .unwrap();
                let shapes = f
                    .shape_params
                    .iter()
                    .map(|p| inner_env[p].clone())
                    .collect::<Vec<_>>();
                if let Some(extent) = logical_extent.as_constant() {
                    if capacity < extent {
                        return self.partition_call(
                            callee,
                            &shapes,
                            elem_args,
                            &inlined_args,
                            &domain.axes,
                            position,
                            extent,
                            capacity,
                            vars,
                            depth,
                        );
                    }
                } else if let Some((_, iteration)) = source_domain {
                    return self.partition_dynamic_call(
                        callee,
                        &shapes,
                        elem_args,
                        &inlined_args,
                        &domain.axes,
                        position,
                        iteration,
                        capacity,
                        vars,
                        depth,
                    );
                }
            }
        }
        if f.is_construct && matches!(self.calls, CallStage::Retain | CallStage::Partition) {
            let span = inlined_args
                .first()
                .map(|e| e.span)
                .or_else(|| f.body.first().map(|s| s.span))
                .unwrap_or_default();
            return Ok(vec![Stmt {
                id: None,
                span,
                kind: StmtKind::Expr(Expr {
                    kind: ExprKind::Call {
                        callee: callee.to_owned(),
                        shape_args: f
                            .shape_params
                            .iter()
                            .map(|p| inner_env[p].clone())
                            .collect(),
                        elem_args: elem_args.to_vec(),
                        args: inlined_args,
                    },
                    ty: Ty::Void,
                    sym: None,
                    span,
                }),
            }]);
        }
        // Choose a body.
        let (body, body_vars, choice) = if f.is_construct && self.calls == CallStage::Resolve {
            let blocks: Vec<&Lowering> = self
                .program
                .lowerings
                .iter()
                .filter(|l| l.construct == callee && l.backend == self.backend)
                .collect();
            let applicable = |l: &Lowering| -> bool {
                let elems_ok = l.elem_bindings.iter().all(|(p, e)| {
                    let idx = f.elem_params.iter().position(|x| x == p);
                    idx.map(|i| &elem_args[i] == e).unwrap_or(false)
                });
                // A residual over a chunked piece must hold for every extent the piece takes.
                let residual_ok = l.residual.iter().all(|r| {
                    let mut assignments: Vec<HashMap<String, i64>> = vec![HashMap::new()];
                    for p in r.params() {
                        let Some(s) = inner_env.get(&p) else {
                            return false;
                        };
                        let values: Vec<i64> = match s.as_constant() {
                            Some(v) => vec![v],
                            None => match single_piece(s) {
                                Some(atom) => match self.piece_values.get(&atom) {
                                    Some(vs) => vs.clone(),
                                    None => return false,
                                },
                                None => return false,
                            },
                        };
                        let mut next = Vec::new();
                        for a in &assignments {
                            for v in &values {
                                let mut b = a.clone();
                                b.insert(p.clone(), *v);
                                next.push(b);
                            }
                        }
                        assignments = next;
                    }
                    assignments.iter().all(|a| {
                        r.eval(&|p| a.get(p).copied())
                            .map(|v| v >= 0)
                            .unwrap_or(false)
                    })
                });
                elems_ok && residual_ok
            };
            let mut alternatives = Vec::new();
            let mut portable = false;
            for (i, block) in blocks.iter().enumerate() {
                if block.body.is_empty()
                    && block.residual.is_empty()
                    && block.elem_bindings.is_empty()
                {
                    portable = true;
                } else if applicable(block) {
                    alternatives.push(Alternative::Body(Choice::Block(i)));
                }
            }
            if portable {
                alternatives.push(Alternative::Body(Choice::Portable));
            }
            let decision = Decision {
                kind: DecisionKind::Construct {
                    name: callee.to_string(),
                    shape_args: f
                        .shape_params
                        .iter()
                        .map(|p| inner_env[p].clone())
                        .collect(),
                    element_args: elem_args.to_vec(),
                },
                alternatives: alternatives.into(),
            };
            if decision.alternatives.is_empty() {
                return Err(format!(
                    "no lowering of `{callee}` on `{}` applies to shapes {:?} and elements {:?}",
                    self.backend, concrete, elem_args
                ));
            }
            let choice = (self.select)(&decision)?;
            if !decision.alternatives.contains(&choice) {
                return Err(format!(
                    "selected lowering {choice:?} is not applicable to `{callee}`; legal alternatives: {:?}",
                    decision.alternatives
                ));
            }
            match choice {
                Alternative::Body(Choice::Block(i)) => (
                    blocks[i].body.clone(),
                    blocks[i].vars.clone(),
                    Choice::Block(i),
                ),
                Alternative::Body(Choice::Portable) => {
                    (f.body.clone(), f.vars.clone(), Choice::Portable)
                }
                _ => unreachable!("validated construct domain"),
            }
        } else {
            (f.body.clone(), f.vars.clone(), Choice::Portable)
        };
        self.selections.push(Selection {
            construct: callee.to_string(),
            shape_args: concrete,
            choice,
        });
        // Element substitution for the callee's generic element types.
        let mut elem_env: HashMap<String, Elem> = HashMap::new();
        for (p, e) in f.elem_params.iter().zip(elem_args) {
            elem_env.insert(p.clone(), e.clone());
        }
        // Build the variable map: parameters bind to argument expressions; locals are appended.
        let mut inner_map: VarMap = HashMap::new();
        let mut inner_atoms: HashMap<String, Atom> = HashMap::new();
        let mut snapshots = Vec::new();
        let mut writes = std::collections::HashSet::new();
        for statement in &body {
            crate::rewrite::writes(statement, &mut writes);
        }
        for (id, v) in body_vars.iter().enumerate() {
            // Bounded index parameters carry Index kind too; parameter identity
            // comes from the declaration, never from the kind of its value.
            if id < f.params.len() {
                let mut argument = inlined_args[id].clone();
                if let VarKind::Index(Atom::Param(name)) = &v.kind {
                    let value = argument.sym.clone().ok_or_else(|| {
                        format!(
                            "bounded index argument {} has no retained symbolic value",
                            f.params[id].0
                        )
                    })?;
                    inner_env.insert(name.clone(), value);
                } else if let Ty::Scalar(dtype) = v.ty {
                    if writes.contains(&id)
                        || !matches!(
                            argument.kind,
                            ExprKind::Var(_)
                                | ExprKind::Int(_)
                                | ExprKind::Float(_)
                                | ExprKind::Bool(_)
                        )
                    {
                        let new_id = vars.len();
                        vars.push(Var {
                            name: format!("{}_{}", v.name, self.fresh()),
                            ty: Ty::Scalar(dtype),
                            span: v.span,
                            kind: VarKind::Local,
                        });
                        let target = Expr {
                            kind: ExprKind::Var(new_id),
                            ty: Ty::Scalar(dtype),
                            sym: None,
                            span: v.span,
                        };
                        let value = Expr {
                            kind: ExprKind::Cast {
                                dtype,
                                expr: Box::new(argument),
                            },
                            ty: Ty::Scalar(dtype),
                            sym: None,
                            span: v.span,
                        };
                        snapshots.push(Stmt {
                            id: None,
                            kind: StmtKind::Assign {
                                target: target.clone(),
                                op: crate::ast::AssignOp::Assign,
                                value,
                            },
                            span: v.span,
                        });
                        argument = target;
                    }
                }
                inner_map.insert(id, argument);
                continue;
            }
            match &v.kind {
                VarKind::Param(i) => {
                    inner_map.insert(id, inlined_args[*i].clone());
                }
                VarKind::Local => {
                    let new_id = vars.len();
                    let ty = subst_elem_ty(&subst_ty(&v.ty, &inner_env), &elem_env);
                    vars.push(Var {
                        name: format!("{}_{}", v.name, self.fresh()),
                        ty: ty.clone(),
                        span: v.span,
                        kind: VarKind::Local,
                    });
                    inner_map.insert(
                        id,
                        Expr {
                            kind: ExprKind::Var(new_id),
                            ty,
                            sym: None,
                            span: v.span,
                        },
                    );
                }
                VarKind::Index(atom) => {
                    let new_id = vars.len();
                    let Atom::Param(name) = atom else {
                        unreachable!()
                    };
                    let new_atom = Atom::Param(format!("{name}_{}", self.fresh()));
                    inner_atoms.insert(name.clone(), new_atom.clone());
                    vars.push(Var {
                        name: format!("{}_{}", v.name, self.fresh()),
                        ty: v.ty.clone(),
                        span: v.span,
                        kind: VarKind::Index(new_atom.clone()),
                    });
                    inner_map.insert(
                        id,
                        Expr {
                            kind: ExprKind::Var(new_id),
                            ty: v.ty.clone(),
                            sym: Some(Sym::atom(new_atom)),
                            span: v.span,
                        },
                    );
                }
            }
        }
        let caller_elements = std::mem::replace(&mut self.elements, elem_env);
        let result = self.inline_block(
            &body,
            &inner_env,
            &inner_map,
            vars,
            &mut inner_atoms,
            depth + 1,
        );
        self.elements = caller_elements;
        result.map(|body| {
            snapshots.extend(body);
            snapshots
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn partition_call(
        &mut self,
        callee: &str,
        shapes: &[Sym],
        elements: &[Elem],
        args: &[Expr],
        axes: &[Option<usize>],
        dimension: usize,
        extent: i64,
        capacity: i64,
        vars: &mut Vec<Var>,
        depth: usize,
    ) -> Result<Vec<Stmt>, String> {
        let span = args[0].span;
        let mut result = Vec::new();
        let index = vars.len();
        let atom = Atom::Param(format!("partition_{}", self.fresh()));
        vars.push(Var {
            name: format!("partition_{index}"),
            ty: Ty::Scalar(crate::types::DType::I32),
            span,
            kind: VarKind::Index(atom.clone()),
        });
        for (start, count, repeated) in [
            (Sym::atom(atom).scale(capacity), capacity, true),
            (
                Sym::constant(extent / capacity * capacity),
                extent % capacity,
                false,
            ),
        ] {
            if count == 0 {
                continue;
            }
            let mut body = Vec::new();
            let mut arguments = Vec::new();
            for (argument, axis) in args.iter().zip(axes) {
                let Some(axis) = axis else {
                    arguments.push(argument.clone());
                    continue;
                };
                let (copy, target) = decomposition::slice_input(
                    argument,
                    *axis,
                    start.clone(),
                    Sym::constant(count),
                    vars,
                )?;
                body.push(copy);
                arguments.push(target);
            }
            let mut selected_shapes = shapes.to_vec();
            selected_shapes[dimension] = Sym::constant(count);
            let parameter = self
                .program
                .functions
                .iter()
                .find(|f| f.name == callee)
                .unwrap()
                .shape_params[dimension]
                .clone();
            let key = (callee.to_owned(), parameter);
            let inserted = self.partitioning.insert(key.clone());
            let expanded = self.inline_call(
                callee,
                &selected_shapes,
                elements,
                &arguments,
                &HashMap::new(),
                &HashMap::new(),
                vars,
                &mut HashMap::new(),
                depth + 1,
            );
            if inserted {
                self.partitioning.remove(&key);
            }
            body.extend(expanded?);
            if repeated {
                result.push(Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Range {
                        var: index,
                        lo: Sym::constant(0),
                        hi: Sym::constant(extent / capacity),
                        body,
                    },
                })
            } else {
                result.extend(body);
            }
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn partition_dynamic_call(
        &mut self,
        callee: &str,
        shapes: &[Sym],
        elements: &[Elem],
        args: &[Expr],
        axes: &[Option<usize>],
        dimension: usize,
        domain: IterationDomain,
        capacity: i64,
        vars: &mut Vec<Var>,
        depth: usize,
    ) -> Result<Vec<Stmt>, String> {
        let span = domain.view.span;
        let piece = Atom::Param(format!("domain_piece#{}", self.fresh()));
        let offset = vars.len();
        let offset_atom = Atom::Param(format!("domain_offset#{}", self.fresh()));
        vars.push(Var {
            name: format!("domain_offset_{offset}"),
            ty: Ty::Scalar(crate::types::DType::I32),
            span,
            kind: VarKind::Index(offset_atom),
        });
        let mut bindings = Vec::new();
        let mut views = Vec::new();
        let mut transfer_axes = Vec::new();
        let mut arguments = Vec::new();
        for (arg, axis) in args.iter().zip(axes) {
            let Some(axis) = axis else {
                arguments.push(arg.clone());
                continue;
            };
            let mut shape = arg
                .ty
                .shaped()
                .ok_or("contraction input must be shaped")?
                .clone();
            shape.shape[*axis] = Sym::atom(piece.clone());
            let v = vars.len();
            vars.push(Var {
                name: format!("domain_input_{v}"),
                ty: Ty::Tile(shape),
                span,
                kind: VarKind::Local,
            });
            arguments.push(Expr {
                kind: ExprKind::Var(v),
                ty: vars[v].ty.clone(),
                span,
                sym: None,
            });
            bindings.push(v);
            views.push(arg.clone());
            transfer_axes.push(*axis);
        }
        let mut shapes = shapes.to_vec();
        shapes[dimension] = Sym::atom(piece.clone());
        let parameter = self
            .program
            .functions
            .iter()
            .find(|f| f.name == callee)
            .unwrap()
            .shape_params[dimension]
            .clone();
        let key = (callee.to_owned(), parameter);
        let inserted = self.partitioning.insert(key.clone());
        let body = self.inline_call(
            callee,
            &shapes,
            elements,
            &arguments,
            &HashMap::new(),
            &HashMap::new(),
            vars,
            &mut HashMap::new(),
            depth + 1,
        );
        if inserted {
            self.partitioning.remove(&key);
        }
        Ok(vec![Stmt {
            id: None,
            span,
            kind: StmtKind::LoadLoop {
                domain,
                offset: Some(offset),
                modes: None,
                vars: bindings,
                views,
                axes: transfer_axes,
                piece,
                capacity: Some(capacity),
                body: body?,
            },
        }])
    }

    // Visit retained leaves in their lexical context. Partition first chooses
    // their logical domains; Resolve preserves those choices while selecting
    // bodies. Existing reductions and loops keep their identities.
    fn resolve_calls(
        &mut self,
        body: &mut Vec<Stmt>,
        vars: &mut Vec<Var>,
        context: &[Stmt],
    ) -> Result<(), String> {
        let saved_domains = self.view_domains.clone();
        let result = self.resolve_call_block(body, vars, context);
        self.view_domains = saved_domains;
        result
    }

    fn resolve_call_block(
        &mut self,
        body: &mut Vec<Stmt>,
        vars: &mut Vec<Var>,
        context: &[Stmt],
    ) -> Result<(), String> {
        let mut resolved = Vec::new();
        let mut context = context.to_vec();
        for mut statement in std::mem::take(body) {
            let mut visible = statement.clone();
            match &mut visible.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Lanes { body, .. }
                | StmtKind::LoadLoop { body, .. } => body.clear(),
                StmtKind::If { then, els, .. } => {
                    then.clear();
                    els.clear();
                }
                StmtKind::Reduction(r) => {
                    for m in r.implementations_mut() {
                        m.body.clear();
                    }
                }
                _ => {}
            }
            // Retaining a call does not retain the inliner's lexical facts.
            // Rebuild them from the already evaluated enclosing statements as
            // we visit this block; child scopes restore this reaching context.
            decomposition::direct_bounds(&visible, &mut self.view_domains);
            context.push(visible);
            if let StmtKind::Expr(Expr {
                kind:
                    ExprKind::Call {
                        callee,
                        shape_args,
                        elem_args,
                        args,
                    },
                ..
            }) = &statement.kind
            {
                let keys = self
                    .domains
                    .get(callee)
                    .into_iter()
                    .flatten()
                    .filter(|_| self.calls == CallStage::Resolve)
                    .map(|d| (callee.clone(), d.parameter.clone()))
                    .collect::<Vec<_>>();
                for key in &keys {
                    self.partitioning.insert(key.clone());
                }
                let expanded = self.inline_call(
                    callee,
                    shape_args,
                    elem_args,
                    args,
                    &HashMap::new(),
                    &HashMap::new(),
                    vars,
                    &mut HashMap::new(),
                    0,
                );
                for key in &keys {
                    self.partitioning.remove(key);
                }
                let mut expanded = expanded?;
                crate::normalize::bind_values(&mut expanded, vars);
                crate::reduction::structured::primitive::retain(&mut expanded, vars, self.select)?;
                decomposition::select_body(&mut expanded, vars, &context, &self.opts, self.select)?;
                resolved.extend(expanded);
                continue;
            }
            match &mut statement.kind {
                StmtKind::Parallel { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Lanes { body, .. }
                | StmtKind::LoadLoop { body, .. } => self.resolve_calls(body, vars, &context)?,
                StmtKind::If { then, els, .. } => {
                    self.resolve_calls(then, vars, &context)?;
                    self.resolve_calls(els, vars, &context)?;
                }
                StmtKind::Reduction(r) => {
                    for m in r.implementations_mut() {
                        self.resolve_calls(&mut m.body, vars, &context)?;
                    }
                }
                _ => {}
            }
            resolved.push(statement);
        }
        *body = resolved;
        Ok(())
    }

    fn fresh(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    fn inline_expr(
        &mut self,
        e: &Expr,
        env: &HashMap<String, Sym>,
        vmap: &VarMap,
        vars: &mut Vec<Var>,
        atom_map: &mut HashMap<String, Atom>,
    ) -> Result<Expr, String> {
        let mut ty = subst_elem_ty(&subst_ty(&e.ty, env), &self.elements);
        let sym = e.sym.as_ref().map(|s| subst_sym(s, env, atom_map));
        let span = e.span;
        let mut sub = |x: &Expr, this: &mut Self| this.inline_expr(x, env, vmap, vars, atom_map);
        let kind = match &e.kind {
            ExprKind::Load { .. } => {
                return Err(
                    "selected execution loads cannot appear in lowering definitions".into(),
                );
            }
            ExprKind::Var(id) => {
                if let Some(bound) = vmap.get(id) {
                    let mut b = bound.clone();
                    // A remapped index variable carries its renamed atom.
                    if let Some(s) = &sym {
                        if matches!(b.kind, ExprKind::Var(_)) && b.sym.is_some() {
                            b.sym = Some(s.clone());
                        }
                    }
                    return Ok(b);
                }
                ExprKind::Var(*id)
            }
            ExprKind::ShapeParam(p) => match sym.clone().and_then(|s| s.as_constant()) {
                Some(v) => ExprKind::Int(v),
                // A piece extent stays symbolic; the printer reads it from `sym`.
                None => ExprKind::ShapeParam(p.clone()),
            },
            ExprKind::Int(v) => ExprKind::Int(*v),
            ExprKind::Float(v) => ExprKind::Float(*v),
            ExprKind::Bool(b) => ExprKind::Bool(*b),
            ExprKind::TileAlloc { shape, dtype } => ExprKind::TileAlloc {
                shape: shape.iter().map(|s| subst_sym(s, env, atom_map)).collect(),
                dtype: match subst_elem(dtype, &self.elements) {
                    Elem::Repr(_) => {
                        return Err("local tile allocation requires a dense dtype".into());
                    }
                    dtype => dtype,
                },
            },
            ExprKind::Index { base, indices } => ExprKind::Index {
                base: Box::new(sub(base, self)?),
                indices: indices
                    .iter()
                    .map(|i| {
                        Ok(match i {
                            Index::Point(p) => {
                                Index::Point(self.inline_expr(p, env, vmap, vars, atom_map)?)
                            }
                            Index::Slice { start, end } => Index::Slice {
                                start: start
                                    .as_ref()
                                    .map(|x| self.inline_expr(x, env, vmap, vars, atom_map))
                                    .transpose()?,
                                end: end
                                    .as_ref()
                                    .map(|x| self.inline_expr(x, env, vmap, vars, atom_map))
                                    .transpose()?,
                            },
                        })
                    })
                    .collect::<Result<_, String>>()?,
            },
            ExprKind::Transpose(inner) => ExprKind::Transpose(Box::new(sub(inner, self)?)),
            ExprKind::Accessor { base, name } => ExprKind::Accessor {
                base: Box::new(sub(base, self)?),
                name: name.clone(),
            },
            ExprKind::Lanes { base, extent } => ExprKind::Lanes {
                base: Box::new(sub(base, self)?),
                extent: subst_sym(extent, env, atom_map),
            },
            ExprKind::Builtin { name, args } => {
                let args = args
                    .iter()
                    .map(|a| self.inline_expr(a, env, vmap, vars, atom_map))
                    .collect::<Result<Vec<_>, _>>()?;
                if *name == Builtin::Reshape
                    && matches!(args[0].ty.shaped().map(|s| &s.elem), Some(Elem::Repr(_)))
                {
                    return Err("reshape currently requires dense storage".into());
                }
                if *name == Builtin::Store {
                    let source = args[0].ty.shaped().ok_or("store source has no shape")?;
                    let target = args[1].ty.shaped().ok_or("store target has no shape")?;
                    if !matches!((&source.elem, &target.elem), (Elem::Dtype(a), Elem::Dtype(b)) if a == b || (a.is_float() && b.is_float()))
                    {
                        return Err(format!(
                            "store specialization cannot publish {} into {}",
                            source.elem, target.elem
                        ));
                    }
                }
                if *name == Builtin::Reduce
                    && matches!(args.get(2).map(|e| &e.kind), Some(ExprKind::Int(3)))
                {
                    let axis = args[1]
                        .sym
                        .as_ref()
                        .and_then(Sym::as_constant)
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or("argmax axis unresolved")?;
                    let extent = args[0]
                        .ty
                        .shaped()
                        .and_then(|s| s.shape.get(axis))
                        .and_then(Sym::as_constant)
                        .ok_or("argmax runtime nonempty-domain validation is not implemented")?;
                    if extent <= 0 {
                        return Err(
                            "argmax requires a nonempty axis after shape specialization".into()
                        );
                    }
                }
                ExprKind::Builtin { name: *name, args }
            }
            ExprKind::Call { .. } => {
                return Err(
                    "a call in expression position cannot be inlined; calls are statements".into(),
                );
            }
            ExprKind::Intrinsic { op: name, args } => ExprKind::Intrinsic {
                op: *name,
                args: args
                    .iter()
                    .map(|a| self.inline_expr(a, env, vmap, vars, atom_map))
                    .collect::<Result<_, _>>()?,
            },
            ExprKind::Unary { op, expr } => ExprKind::Unary {
                op: *op,
                expr: Box::new(sub(expr, self)?),
            },
            ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
                op: *op,
                lhs: Box::new(sub(lhs, self)?),
                rhs: Box::new(sub(rhs, self)?),
            },
            ExprKind::Cast { dtype, expr } => ExprKind::Cast {
                dtype: *dtype,
                expr: Box::new(sub(expr, self)?),
            },
            ExprKind::Tuple(items) => ExprKind::Tuple(
                items
                    .iter()
                    .map(|a| self.inline_expr(a, env, vmap, vars, atom_map))
                    .collect::<Result<_, _>>()?,
            ),
        };
        // A generic element read has a provisional scalar type before its
        // storage element parameter is bound. Scalar types contain no element
        // parameter for subst_elem_ty to replace, so derive the specialized
        // index type from the now-specialized storage. Explicit casts and
        // assignment conversions remain at their original use sites.
        if let ExprKind::Index { base, .. } = &kind {
            if matches!(ty, Ty::Scalar(_)) {
                ty = Ty::Scalar(base.ty.shaped()
                    .and_then(|shape| shape.elem.read_dtype())
                    .ok_or("indexed element type remains unresolved after specialization")?);
            }
        }
        let expression = Expr {
            kind,
            ty,
            sym,
            span,
        };
        decomposition::collect(&expression, &mut self.view_domains);
        Ok(expression)
    }
}

fn remap_var(v: VarId, vmap: &VarMap) -> VarId {
    match vmap.get(&v) {
        Some(Expr {
            kind: ExprKind::Var(id),
            ..
        }) => *id,
        Some(_) => panic!("loop variable bound to a non-variable"),
        None => v,
    }
}

fn remap_atom(a: &Atom, atom_map: &HashMap<String, Atom>) -> Atom {
    match a {
        Atom::Param(p) => atom_map.get(p).cloned().unwrap_or_else(|| a.clone()),
        _ => a.clone(),
    }
}

/// Substitute shape parameters by their bindings and rename loop atoms.
/// The piece atom name when a symbol is exactly one piece atom.
fn single_piece(s: &Sym) -> Option<String> {
    let params = s.params();
    if params.len() == 1 && *s == Sym::param(&params[0]) && params[0].contains('#') {
        Some(params[0].clone())
    } else {
        None
    }
}

pub fn subst_sym(s: &Sym, env: &HashMap<String, Sym>, atom_map: &HashMap<String, Atom>) -> Sym {
    let mut out = s.clone();
    for a in s.atoms() {
        match &a {
            Atom::Param(p) => {
                if let Some(v) = env.get(p) {
                    out = out.subst(&a, v);
                } else if let Some(n) = atom_map.get(p) {
                    out = out.subst(&a, &Sym::atom(n.clone()));
                }
            }
            Atom::Quot(n, d) => {
                let r = Sym::atom(Atom::Quot(
                    Box::new(subst_sym(n, env, atom_map)),
                    Box::new(subst_sym(d, env, atom_map)),
                ));
                let r = simplify_div(&r);
                out = out.subst(&a, &r);
            }
            Atom::Rem(n, d) => {
                let r = Sym::atom(Atom::Rem(
                    Box::new(subst_sym(n, env, atom_map)),
                    Box::new(subst_sym(d, env, atom_map)),
                ));
                let r = simplify_div(&r);
                out = out.subst(&a, &r);
            }
        }
    }
    out
}

/// Re-normalize a quotient or remainder atom whose parts may now be constants.
fn simplify_div(s: &Sym) -> Sym {
    if let [atom] = s.atoms().as_slice() {
        if *s == Sym::atom(atom.clone()) {
            match atom {
                Atom::Quot(n, d) => return n.quot(d),
                Atom::Rem(n, d) => return n.rem(d),
                _ => {}
            }
        }
    }
    s.clone()
}

pub fn subst_ty(t: &Ty, env: &HashMap<String, Sym>) -> Ty {
    let empty = HashMap::new();
    let f = |s: &Shaped| Shaped {
        shape: s.shape.iter().map(|d| subst_sym(d, env, &empty)).collect(),
        elem: s.elem.clone(),
        packed_axis: s.packed_axis,
    };
    match t {
        Ty::Tensor(s) => Ty::Tensor(f(s)),
        Ty::Tile(s) => Ty::Tile(f(s)),
        Ty::Frag(s) => Ty::Frag(f(s)),
        Ty::Tuple(items) => Ty::Tuple(items.iter().map(|i| subst_ty(i, env)).collect()),
        other => other.clone(),
    }
}

pub(crate) fn subst_elem(elem: &Elem, elems: &HashMap<String, Elem>) -> Elem {
    match elem {
        Elem::Param(p) => elems.get(p).cloned().unwrap_or_else(|| elem.clone()),
        other => other.clone(),
    }
}

pub fn subst_elem_ty(t: &Ty, elems: &HashMap<String, Elem>) -> Ty {
    let f = |s: &Shaped| {
        let elem = subst_elem(&s.elem, elems);
        let packed_axis = match (&elem, s.packed_axis) {
            (Elem::Repr(_), None) => Some(s.shape.len().saturating_sub(1)),
            (_, a) => a,
        };
        Shaped {
            shape: s.shape.clone(),
            elem,
            packed_axis,
        }
    };
    match t {
        Ty::Tensor(s) => Ty::Tensor(f(s)),
        Ty::Tile(s) => Ty::Tile(f(s)),
        Ty::Frag(s) => Ty::Frag(f(s)),
        Ty::Tuple(items) => Ty::Tuple(items.iter().map(|i| subst_elem_ty(i, elems)).collect()),
        other => other.clone(),
    }
}

/// Select legal producer materialization after expansion is complete. A producer
/// is offered once; nested expansion cannot override a previous retention decision.
fn select_producers(
    block: &mut Vec<Stmt>,
    vars: &[Var],
    select: &mut dyn FnMut(&Decision) -> Result<Alternative, String>,
) -> Result<(), String> {
    for statement in block.iter_mut() {
        match &mut statement.kind {
            StmtKind::Reduction(r) => {
                for merge in r.implementations_mut() {
                    select_producers(&mut merge.body, vars, select)?;
                }
            }
            StmtKind::Parallel { body, .. }
            | StmtKind::Range { body, .. }
            | StmtKind::Owned { body, .. }
            | StmtKind::Lanes { body, .. }
            | StmtKind::LoadLoop { body, .. } => select_producers(body, vars, select)?,
            StmtKind::If {
                then: then_body,
                els: else_body,
                ..
            } => {
                select_producers(then_body, vars, select)?;
                select_producers(else_body, vars, select)?;
            }
            _ => {}
        }
    }
    let mut producers = producer_candidates(block, vars);
    // Sort by the source variable identity; HashMap iteration must never affect
    // replay, candidate identity, or coverage of independent decisions.
    let mut candidates: Vec<_> = producers.keys().copied().collect();
    candidates.sort_unstable();
    for variable in candidates {
        let decision = Decision {
            kind: DecisionKind::Producer {
                variable,
                name: vars[variable].name.clone(),
                ty: vars[variable].ty.clone(),
            },
            alternatives: vec![Alternative::Materialize, Alternative::Recompute].into(),
        };
        match select(&decision)? {
            Alternative::Materialize => {
                producers.remove(&variable);
            }
            Alternative::Recompute => {}
            other => return Err(format!("invalid producer alternative {other:?}")),
        }
    }
    // Drop their definitions and rewrite the reads.
    block.retain(|s| match &s.kind {
        StmtKind::Owned { tile, .. } => {
            !matches!(tile.kind, ExprKind::Var(a) if producers.contains_key(&a))
        }
        StmtKind::Assign { target, value, .. } => {
            !(matches!(target.kind, ExprKind::Var(a) if producers.contains_key(&a))
                && matches!(value.kind, ExprKind::TileAlloc { .. }))
        }
        _ => true,
    });
    for s in block.iter_mut() {
        rewrite_reads_stmt(s, &producers, vars);
    }
    Ok(())
}

/// Complete local recomputation domain from pure producers and snapshot lifetimes.
fn producer_candidates(block: &[Stmt], vars: &[Var]) -> HashMap<VarId, (Vec<VarId>, Expr)> {
    let mut producers = pure_producer_candidates(block, vars);
    let mut other_uses: HashMap<VarId, usize> = HashMap::new();
    for statement in block {
        count_non_element_uses(statement, &producers, &mut other_uses);
    }
    producers.retain(|variable, _| other_uses.get(variable).copied().unwrap_or(0) == 0);
    producers
}

/// Reaching pure element definitions with stable source dependencies. Consumers
/// establish separately whether their demanded values can be projected from it.
fn pure_producer_candidates(block: &[Stmt], vars: &[Var]) -> HashMap<VarId, (Vec<VarId>, Expr)> {
    // Only a tile allocated in this block can be removed here. A write to a
    // loop-carried or enclosing tile escapes this block even without a local read.
    let allocated: std::collections::HashSet<VarId> = block
        .iter()
        .filter_map(|s| match &s.kind {
            StmtKind::Assign { target, value, .. }
                if matches!(value.kind, ExprKind::TileAlloc { .. }) =>
            {
                match target.kind {
                    ExprKind::Var(v) => Some(v),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    let mut writes = Writes::default();
    for statement in block.iter() {
        written_vars(statement, &mut writes);
    }
    // Producers at this level: `for i.. in owned(a): a[i..] = value`.
    let mut producers: HashMap<VarId, (Vec<VarId>, Expr)> = HashMap::new();
    for (position, s) in block.iter().enumerate() {
        if let StmtKind::Owned {
            vars: ivs,
            tile,
            body,
        } = &s.kind
        {
            let ExprKind::Var(a) = tile.kind else {
                continue;
            };
            if !matches!(vars[a].kind, VarKind::Local)
                || !allocated.contains(&a)
                || writes.variables.get(&a) != Some(&2)
            {
                continue;
            }
            let Some((target, value)) =
                producer_value(body, a, vars, &block[..position], &block[position + 1..])
            else {
                continue;
            };
            let ExprKind::Index { base, indices } = &target.kind else {
                continue;
            };
            if !matches!(base.kind, ExprKind::Var(b) if b == a) {
                continue;
            }
            let plain = indices
                .iter()
                .zip(ivs)
                .all(|(ix, v)| match (ix, &vars[*v].kind) {
                    (Index::Point(p), VarKind::Index(atom)) => {
                        p.sym.as_ref() == Some(&Sym::atom(atom.clone()))
                    }
                    _ => false,
                });
            if !plain || indices.len() != ivs.len() || mentions_var(&value, a) {
                continue;
            }
            // Recomputing a value later is legal only while all its source values
            // remain unchanged. Unknown memory effects conservatively prevent motion.
            let mut dependencies = std::collections::HashSet::new();
            walk_expr(&value, &mut |e| {
                if let ExprKind::Var(v) = e.kind {
                    if !ivs.contains(&v) {
                        dependencies.insert(v);
                    }
                }
            });
            let mut future_writes = Writes::default();
            for later in &block[position + 1..] {
                written_vars(later, &mut future_writes);
            }
            if future_writes.unknown
                || (future_writes.tensors
                    && dependencies
                        .iter()
                        .any(|v| matches!(vars[*v].ty, Ty::Tensor(_))))
                || dependencies
                    .iter()
                    .any(|v| future_writes.variables.contains_key(v))
            {
                continue;
            }
            producers.insert(a, (ivs.clone(), value));
        }
    }
    producers
}

#[derive(Default)]
struct Writes {
    variables: HashMap<VarId, usize>,
    tensors: bool,
    unknown: bool,
}

/// Extract a straight-line value from the same pure producer normalization
/// used by sharing. Conditional producers keep their original lazy control;
/// consumers requiring a single expression do not admit those definitions.
pub(crate) fn producer_value(
    body: &[Stmt], output: VarId, vars: &[Var], before: &[Stmt], after: &[Stmt],
) -> Option<(Expr, Expr)> {
    let (target, definition) = producer_definition(body, output, vars, before, after)?;
    let [Stmt { kind: StmtKind::Assign { value, .. }, .. }] = definition.as_slice() else { return None; };
    Some((target, value.clone()))
}

/// A complete pure point producer, represented by its existing statements.
/// Scalar temporaries retain their conversions; every conditional path must
/// assign the same element exactly once. No select expression eagerly evaluates
/// an unchosen read, and no new semantic graph is introduced.
pub(crate) fn producer_definition(
    body: &[Stmt], output: VarId, vars: &[Var], before: &[Stmt], after: &[Stmt],
) -> Option<(Expr, Vec<Stmt>)> {
    fn normalize(body: &[Stmt], output: VarId, vars: &[Var], before: &[Stmt], after: &[Stmt], inherited: &HashMap<VarId, Expr>) -> Option<(Expr, Vec<Stmt>)> {
        let mut scalars = inherited.clone();
        let mut target_value = None;
        let mut definition = Vec::new();
        let canonical = |kind| Stmt { id: None, span: crate::span::Span::default(), kind };
        for (position, statement) in body.iter().enumerate() {
            match &statement.kind {
                StmtKind::Assign { target, op: crate::ast::AssignOp::Assign, value } => {
                    let value = subst_vars(value, &scalars, &HashMap::new());
                    let substituted_target = subst_vars(target, &scalars, &HashMap::new());
                    if !crate::effects::expression_can_be_omitted(&value)
                        || !crate::effects::expression_can_be_omitted(&substituted_target) { return None; }
                    match &target.kind {
                        ExprKind::Var(v) if matches!(target.ty, Ty::Scalar(_)) && matches!(vars[*v].kind, VarKind::Local) => {
                            if before.iter().chain(after).any(|s| crate::effects::uses(s, *v)) { return None; }
                            let Ty::Scalar(dtype) = target.ty else { unreachable!() };
                            scalars.insert(*v, Expr { kind: ExprKind::Cast { dtype, expr: Box::new(value) }, ty: target.ty.clone(), sym: None, span: target.span });
                        }
                        ExprKind::Index { base, .. } if matches!(base.kind, ExprKind::Var(v) if v == output) => {
                            if target_value.is_some() { return None; }
                            let target = substituted_target;
                            target_value = Some(target.clone());
                            definition.push(canonical(StmtKind::Assign { target, op: crate::ast::AssignOp::Assign, value }));
                        }
                        _ => return None,
                    }
                }
                StmtKind::If { cond, then, els } if target_value.is_none() && position + 1 == body.len() => {
                    let condition = subst_vars(cond, &scalars, &HashMap::new());
                    if !crate::effects::expression_can_be_omitted(&condition) { return None; }
                    let (then_target, then) = normalize(then, output, vars, before, after, &scalars)?;
                    let (else_target, els) = normalize(els, output, vars, before, after, &scalars)?;
                    if crate::normalize::value_identity(&then_target) != crate::normalize::value_identity(&else_target) { return None; }
                    target_value = Some(then_target);
                    definition.push(canonical(StmtKind::If { cond: condition, then, els }));
                }
                _ => return None,
            }
        }
        Some((target_value?, definition))
    }
    normalize(body, output, vars, before, after, &HashMap::new())
}
/// Syntactic effects across nested control flow. Materialized tile values cannot
/// alias tensor storage, while tensor reads require an alias proof to cross stores.
fn written_vars(statement: &Stmt, writes: &mut Writes) {
    fn expression(e: &Expr, writes: &mut Writes) {
        walk_expr(e, &mut |e| {
            if matches!(
                e.kind,
                ExprKind::Intrinsic { .. }
                    | ExprKind::Call { .. }
                    | ExprKind::Builtin {
                        name: Builtin::Atomic,
                        ..
                    }
            ) {
                writes.unknown = true;
            }
            if matches!(
                e.kind,
                ExprKind::Builtin {
                    name: Builtin::Store,
                    ..
                }
            ) {
                writes.tensors = true;
            }
        });
    }
    match &statement.kind {
        StmtKind::Reduction(r) => {
            for v in r.state_variables() {
                *writes.variables.entry(v).or_default() += 1;
            }
            for child in r.bodies().flatten() {
                written_vars(child, writes);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            expression(value, writes);
            let mut root = target;
            while let ExprKind::Index { base, .. } | ExprKind::Transpose(base) = &root.kind {
                root = base;
            }
            if let ExprKind::Var(v) = root.kind {
                *writes.variables.entry(v).or_default() += 1;
            }
        }
        StmtKind::Expr(e) => expression(e, writes),
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. } => {
            for child in body {
                written_vars(child, writes);
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            expression(tile, writes);
            for child in body {
                written_vars(child, writes);
            }
        }
        StmtKind::LoadLoop {
            domain,
            views,
            body,
            ..
        } => {
            expression(&domain.view, writes);
            for view in views {
                expression(view, writes);
            }
            for child in body {
                written_vars(child, writes);
            }
        }
        StmtKind::If { cond, then, els } => {
            expression(cond, writes);
            for child in then.iter().chain(els) {
                written_vars(child, writes);
            }
        }
    }
}

fn mentions_var(e: &Expr, a: VarId) -> bool {
    let mut found = false;
    walk_expr(e, &mut |x| {
        if matches!(x.kind, ExprKind::Var(v) if v == a) {
            found = true;
        }
    });
    found
}

fn walk_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Index { base, indices } => {
            walk_expr(base, f);
            for i in indices {
                match i {
                    Index::Point(p) => walk_expr(p, f),
                    Index::Slice { start, end } => {
                        if let Some(x) = start {
                            walk_expr(x, f)
                        }
                        if let Some(x) = end {
                            walk_expr(x, f)
                        }
                    }
                }
            }
        }
        ExprKind::Transpose(x)
        | ExprKind::Accessor { base: x, .. }
        | ExprKind::Lanes { base: x, .. }
        | ExprKind::Unary { expr: x, .. }
        | ExprKind::Cast { expr: x, .. } => walk_expr(x, f),
        ExprKind::Builtin { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Tuple(args) => {
            for a in args {
                walk_expr(a, f);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        _ => {}
    }
}

/// Uses of a producer tile other than all-point element reads, its own definition excluded.
fn count_non_element_uses(
    s: &Stmt,
    producers: &HashMap<VarId, (Vec<VarId>, Expr)>,
    out: &mut HashMap<VarId, usize>,
) {
    let expr_uses = |e: &Expr, out: &mut HashMap<VarId, usize>| {
        // Walk manually so element reads of producers are not descended into as bare vars.
        fn go(
            e: &Expr,
            producers: &HashMap<VarId, (Vec<VarId>, Expr)>,
            out: &mut HashMap<VarId, usize>,
        ) {
            match &e.kind {
                ExprKind::Var(v) => {
                    if producers.contains_key(v) {
                        *out.entry(*v).or_insert(0) += 1;
                    }
                }
                ExprKind::Index { base, indices } => {
                    let all_points = indices.iter().all(|i| matches!(i, Index::Point(_)));
                    match &base.kind {
                        ExprKind::Var(v) if producers.contains_key(v) && all_points => {}
                        _ => go(base, producers, out),
                    }
                    for i in indices {
                        match i {
                            Index::Point(p) => go(p, producers, out),
                            Index::Slice { start, end } => {
                                if let Some(x) = start {
                                    go(x, producers, out)
                                }
                                if let Some(x) = end {
                                    go(x, producers, out)
                                }
                            }
                        }
                    }
                }
                ExprKind::Transpose(x)
                | ExprKind::Accessor { base: x, .. }
                | ExprKind::Lanes { base: x, .. }
                | ExprKind::Unary { expr: x, .. }
                | ExprKind::Cast { expr: x, .. } => go(x, producers, out),
                ExprKind::Builtin { args, .. }
                | ExprKind::Intrinsic { args, .. }
                | ExprKind::Call { args, .. }
                | ExprKind::Tuple(args) => {
                    for a in args {
                        go(a, producers, out);
                    }
                }
                ExprKind::Binary { lhs, rhs, .. } => {
                    go(lhs, producers, out);
                    go(rhs, producers, out);
                }
                _ => {}
            }
        }
        go(e, producers, out);
    };
    match &s.kind {
        StmtKind::Reduction(r) => {
            for e in r.operands() {
                expr_uses(e, out);
            }
            for s in r.bodies().flatten() {
                count_non_element_uses(s, producers, out);
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            // The producer's own loop is its definition; any other owned loop over it is a use.
            let own = matches!(tile.kind, ExprKind::Var(a) if producers.contains_key(&a));
            if !own {
                expr_uses(tile, out);
                for b in body {
                    count_non_element_uses(b, producers, out);
                }
            }
        }
        StmtKind::Assign { target, value, .. } => {
            let alloc = matches!(target.kind, ExprKind::Var(a) if producers.contains_key(&a))
                && matches!(value.kind, ExprKind::TileAlloc { .. });
            if !alloc {
                // An element write elsewhere is a use that forbids inlining.
                if let ExprKind::Index { base, .. } = &target.kind {
                    if let ExprKind::Var(a) = base.kind {
                        if producers.contains_key(&a) {
                            *out.entry(a).or_insert(0) += 1;
                        }
                    }
                } else {
                    expr_uses(target, out);
                }
                expr_uses(value, out);
            }
        }
        StmtKind::Expr(e) => expr_uses(e, out),
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. } => {
            for b in body {
                count_non_element_uses(b, producers, out);
            }
        }
        StmtKind::LoadLoop {
            domain,
            views,
            body,
            ..
        } => {
            expr_uses(&domain.view, out);
            for v in views {
                expr_uses(v, out);
            }
            for b in body {
                count_non_element_uses(b, producers, out);
            }
        }
        StmtKind::If { cond, then, els } => {
            expr_uses(cond, out);
            for b in then.iter().chain(els.iter()) {
                count_non_element_uses(b, producers, out);
            }
        }
    }
}

fn rewrite_reads_stmt(s: &mut Stmt, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, vars: &[Var]) {
    match &mut s.kind {
        StmtKind::Reduction(r) => {
            for e in &mut r.inputs {
                rewrite_reads(e, producers, vars);
            }
            for merge in r.implementations_mut() {
                for s in &mut merge.body {
                    rewrite_reads_stmt(s, producers, vars);
                }
            }
        }
        StmtKind::Owned { tile, body, .. } => {
            rewrite_reads(tile, producers, vars);
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            rewrite_reads(target, producers, vars);
            rewrite_reads(value, producers, vars);
        }
        StmtKind::Expr(e) => rewrite_reads(e, producers, vars),
        StmtKind::Parallel { body, .. }
        | StmtKind::Range { body, .. }
        | StmtKind::Lanes { body, .. } => {
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::LoadLoop {
            domain,
            views,
            body,
            ..
        } => {
            rewrite_reads(&mut domain.view, producers, vars);
            for v in views {
                rewrite_reads(v, producers, vars);
            }
            for b in body {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
        StmtKind::If { cond, then, els } => {
            rewrite_reads(cond, producers, vars);
            for b in then.iter_mut().chain(els.iter_mut()) {
                rewrite_reads_stmt(b, producers, vars);
            }
        }
    }
}

fn rewrite_reads(e: &mut Expr, producers: &HashMap<VarId, (Vec<VarId>, Expr)>, vars: &[Var]) {
    if let ExprKind::Index { base, indices } = &e.kind {
        if let ExprKind::Var(a) = base.kind {
            if let Some((ivs, value)) = producers.get(&a) {
                if indices.iter().all(|i| matches!(i, Index::Point(_))) {
                    let points: Vec<Expr> = indices
                        .iter()
                        .map(|i| match i {
                            Index::Point(p) => {
                                let mut p = p.clone();
                                rewrite_reads(&mut p, producers, vars);
                                p
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    let mut map: HashMap<VarId, Expr> = HashMap::new();
                    let mut atoms: HashMap<String, Option<Sym>> = HashMap::new();
                    for (v, p) in ivs.iter().zip(points) {
                        if let VarKind::Index(Atom::Param(name)) = &vars[*v].kind {
                            atoms.insert(name.clone(), p.sym.clone());
                        }
                        map.insert(*v, p);
                    }
                    let mut replaced = subst_vars(value, &map, &atoms);
                    rewrite_reads(&mut replaced, producers, vars);
                    // Preserve the eliminated tile's publication precision, even
                    // when its producer computed a wider intermediate expression.
                    if let Ty::Tile(tile) = &vars[a].ty {
                        if let Elem::Dtype(dtype) = tile.elem {
                            replaced = Expr {
                                kind: ExprKind::Cast {
                                    dtype,
                                    expr: Box::new(replaced),
                                },
                                ty: Ty::Scalar(dtype),
                                sym: None,
                                span: e.span,
                            };
                        }
                    }
                    if let Ty::Scalar(dtype) = e.ty {
                        if replaced.ty != e.ty {
                            replaced = Expr {
                                kind: ExprKind::Cast {
                                    dtype,
                                    expr: Box::new(replaced),
                                },
                                ty: e.ty.clone(),
                                sym: None,
                                span: e.span,
                            };
                        }
                    }
                    *e = replaced;
                    return;
                }
            }
        }
    }
    match &mut e.kind {
        ExprKind::Index { base, indices } => {
            rewrite_reads(base, producers, vars);
            for i in indices {
                match i {
                    Index::Point(p) => rewrite_reads(p, producers, vars),
                    Index::Slice { start, end } => {
                        if let Some(x) = start {
                            rewrite_reads(x, producers, vars)
                        }
                        if let Some(x) = end {
                            rewrite_reads(x, producers, vars)
                        }
                    }
                }
            }
        }
        ExprKind::Transpose(x)
        | ExprKind::Accessor { base: x, .. }
        | ExprKind::Lanes { base: x, .. }
        | ExprKind::Unary { expr: x, .. }
        | ExprKind::Cast { expr: x, .. } => rewrite_reads(x, producers, vars),
        ExprKind::Builtin { args, .. }
        | ExprKind::Intrinsic { args, .. }
        | ExprKind::Call { args, .. }
        | ExprKind::Tuple(args) => {
            for a in args {
                rewrite_reads(a, producers, vars);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            rewrite_reads(lhs, producers, vars);
            rewrite_reads(rhs, producers, vars);
        }
        _ => {}
    }
}

/// Substitute loop-index variables by expressions, in kinds and in symbolic values.
pub(crate) fn subst_vars(
    e: &Expr,
    map: &HashMap<VarId, Expr>,
    atoms: &HashMap<String, Option<Sym>>,
) -> Expr {
    if let ExprKind::Var(v) = e.kind {
        if let Some(r) = map.get(&v) {
            return r.clone();
        }
    }
    let sym = e.sym.as_ref().and_then(|s| {
        let mut out = s.clone();
        for (name, value) in atoms {
            let atom = Atom::Param(name.clone());
            if out.atoms().contains(&atom) {
                match value {
                    Some(v) => out = out.subst(&atom, v),
                    None => return None,
                }
            }
        }
        Some(out)
    });
    let sub = |x: &Expr| subst_vars(x, map, atoms);
    let kind = match &e.kind {
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(sub(base)),
            indices: indices
                .iter()
                .map(|i| match i {
                    Index::Point(p) => Index::Point(sub(p)),
                    Index::Slice { start, end } => Index::Slice {
                        start: start.as_ref().map(sub),
                        end: end.as_ref().map(sub),
                    },
                })
                .collect(),
        },
        ExprKind::Transpose(x) => ExprKind::Transpose(Box::new(sub(x))),
        ExprKind::Accessor { base, name } => ExprKind::Accessor {
            base: Box::new(sub(base)),
            name: name.clone(),
        },
        ExprKind::Lanes { base, extent } => ExprKind::Lanes {
            base: Box::new(sub(base)),
            extent: extent.clone(),
        },
        ExprKind::Builtin { name, args } => ExprKind::Builtin {
            name: *name,
            args: args.iter().map(sub).collect(),
        },
        ExprKind::Intrinsic { op: name, args } => ExprKind::Intrinsic {
            op: *name,
            args: args.iter().map(sub).collect(),
        },
        ExprKind::Call {
            callee,
            shape_args,
            elem_args,
            args,
        } => ExprKind::Call {
            callee: callee.clone(),
            shape_args: shape_args.clone(),
            elem_args: elem_args.clone(),
            args: args.iter().map(sub).collect(),
        },
        ExprKind::Tuple(items) => ExprKind::Tuple(items.iter().map(sub).collect()),
        ExprKind::Unary { op, expr } => ExprKind::Unary {
            op: *op,
            expr: Box::new(sub(expr)),
        },
        ExprKind::Cast { dtype, expr } => ExprKind::Cast {
            dtype: *dtype,
            expr: Box::new(sub(expr)),
        },
        ExprKind::Binary { op, lhs, rhs } => ExprKind::Binary {
            op: *op,
            lhs: Box::new(sub(lhs)),
            rhs: Box::new(sub(rhs)),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        ty: e.ty.clone(),
        sym,
        span: e.span,
    }
}

/// Inline transparent helpers using their checked portable definitions for
/// dependence analysis. This is a derived view of the same source, not a model.
fn portable_body(program: &Program, f: &Function) -> Result<(Vec<Var>, Vec<Stmt>), String> {
    let mut select = |_: &Decision| Err("portable semantic expansion attempted an implementation decision".to_owned());
    let mut ctx = Inliner {
        program,
        backend: "",
        select: &mut select,
        selections: Vec::new(),
        counter: 0,
        opts: Options::default(),
        elements: HashMap::new(),
        piece_values: HashMap::new(),
        calls: CallStage::Portable,
        partitioning: std::collections::HashSet::new(),
        domains: HashMap::new(),
        view_domains: HashMap::new(),
    };
    let mut vars = f.vars.clone();
    let body = ctx.inline_block(
        &f.body,
        &HashMap::new(),
        &HashMap::new(),
        &mut vars,
        &mut HashMap::new(),
        0,
    )?;
    Ok((vars, body))
}
