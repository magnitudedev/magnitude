//! Checked source-order launch boundaries and invocation-owned value handoffs.
//! A serial region executes once. Its values are published once and reloaded
//! after completion, never recomputed independently by parallel consumers.
use seismic_lang::{
    ast::AssignOp,
    ir::*,
    lowered_ir::LoweredIr,
    span::Span,
    sym::Sym,
    types::{DType, Elem, Shaped, Ty},
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedValue {
    pub variable: VarId,
    /// Appended tensor parameter ordinal in the transformed function.
    pub parameter: usize,
    pub name: String,
    pub dtype: DType,
    pub elements: u64,
    pub bytes: usize,
    pub producer: usize,
    pub consumers: Vec<usize>,
    /// Ordered publications, including subsequent serial updates.
    pub writers: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Phase {
    /// Physical completion is required before this phase begins.
    pub predecessor: Option<usize>,
    pub inputs: Vec<VarId>,
    pub outputs: Vec<VarId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhasePlan {
    pub function: LoweredIr,
    pub phases: Vec<Phase>,
    pub retained: Vec<RetainedValue>,
    pub source_parameters: usize,
    pub handoffs: Vec<PhaseHandoff>,
}
/// Invocation-owned restores and publications shared by every local
/// implementation alternative of one source phase.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseHandoff {
    pub restores: Vec<Stmt>,
    pub publications: Vec<Stmt>,
}

/// Applicability never turns a missing realization into an infeasibility proof.
/// Malformed checked IR is an error; a well-formed program whose phase storage
/// or work domain is not represented yet retains an explicit unresolved reason.
pub enum Applicability {
    Supported(PhasePlan),
    Unresolved { reason: String },
}
#[derive(Debug)]
enum FormationError {
    Unresolved(String),
    Invalid(String),
}
impl std::fmt::Display for FormationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved(reason) | Self::Invalid(reason) => formatter.write_str(reason),
        }
    }
}
impl From<String> for FormationError {
    fn from(reason: String) -> Self {
        Self::Invalid(reason)
    }
}
impl From<&str> for FormationError {
    fn from(reason: &str) -> Self {
        Self::Invalid(reason.into())
    }
}
pub fn assess(source: &LoweredIr) -> Result<Applicability, String> {
    seismic_lang::verify::lowered(source, seismic_lang::verify::Stage::Expanded)?;
    match construct_checked(source) {
        Ok(plan) => Ok(Applicability::Supported(plan)),
        Err(FormationError::Unresolved(reason)) => Ok(Applicability::Unresolved { reason }),
        Err(FormationError::Invalid(reason)) => Err(reason),
    }
}

/// Construct one legal materialized phase realization. Parallel-local values
/// cannot escape their owner domain, and parallel consumers cannot mutate a
/// broadcast snapshot without a separately established ownership/merge rule.
pub fn construct(source: &LoweredIr) -> Result<PhasePlan, String> {
    construct_checked(source).map_err(|error| error.to_string())
}
/// Compiler parameters have fixed values within one selected program and
/// declared finite bounds while phase formation retains the whole family.
pub fn construct_parameterized(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>) -> Result<PhasePlan, String> {
    construct_retained(source, numeric, &BTreeSet::new())
}
/// Compiler predicates describe disjoint source alternatives within one
/// retained template. Their conditional definitions remain visible to later
/// regions guarded by the same original decisions. Concrete reconstruction
/// verifies the selected source with the ordinary lexical scope rules.
pub fn construct_retained(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>, selectors: &BTreeSet<VarId>) -> Result<PhasePlan, String> {
    construct_with_parameters(source, numeric, selectors).map_err(|error| error.to_string())
}
fn construct_checked(source: &LoweredIr) -> Result<PhasePlan, FormationError> {
    construct_with_parameters(source, &BTreeMap::new(), &BTreeSet::new())
}
fn construct_with_parameters(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>, selectors: &BTreeSet<VarId>) -> Result<PhasePlan, FormationError> {
    let mut function = work_domains_with_parameters(source, numeric, selectors)?;
    let mut parameters = parameter_variables(&function)?;
    for &selector in selectors {
        if function.vars.get(selector).is_none_or(|variable| variable.ty != Ty::Scalar(DType::Bool)) {
            return Err("retained compiler selector requires a boolean binding".into());
        }
        parameters.insert(selector);
    }
    let mut available = parameters.clone();
    let mut producers = BTreeMap::new();
    let mut phases = Vec::new();
    let mut captures: BTreeMap<VarId, (usize, Vec<usize>)> = BTreeMap::new();
    let mut writes = Vec::new();
    for (index, root) in function.body.iter().enumerate() {
        let StmtKind::Parallel {
            vars,
            extents,
            body,
        } = &root.kind
        else {
            unreachable!()
        };
        if vars.len() != extents.len() {
            return Err("phase index/extent arity differs".into());
        }
        if extents
            .iter()
            .any(|e| e.as_constant().is_none_or(|n| n < 0))
        {
            return Err(FormationError::Unresolved(
                "phase work domain must have static nonnegative extents".into(),
            ));
        }
        let mut bound = parameters.clone();
        bound.extend(vars);
        let mut scope = Scope {
            function: &function,
            numeric,
            selectors,
            inputs: BTreeSet::new(),
            symbols: BTreeSet::new(),
        };
        scope.body(body, &mut bound)?;
        if let Some(symbol) = scope.unbound_symbol(&bound) {
            return Err(FormationError::Unresolved(format!(
                "phase {index} needs an explicit captured value for runtime symbol `{symbol}`"
            )));
        }
        let mut changed = HashSet::new();
        for statement in body {
            seismic_lang::rewrite::value_writes(statement, &function.vars, &mut changed);
        }
        // Reassignment of an existing tile copies into its captured geometry.
        // Even a complete overwrite must retain the earlier shape checks.
        scope
            .inputs
            .extend(changed.iter().copied().filter(|variable| {
                available.contains(variable) && matches!(function.vars[*variable].ty, Ty::Tile(_))
            }));
        let inputs: Vec<_> = scope.inputs.into_iter().collect();
        for &variable in &inputs {
            if !available.contains(&variable) {
                return Err(format!(
                    "phase {index} reads `{}` outside its defining scope",
                    function.vars[variable].name
                )
                .into());
            }
            let producer = *producers
                .get(&variable)
                .ok_or("phase input has no completed value producer")?;
            captures
                .entry(variable)
                .or_insert_with(|| (producer, Vec::new()))
                .1
                .push(index);
        }
        if !vars.is_empty()
            && available.iter().any(|v| {
                !parameters.contains(v)
                    && !matches!(function.vars[*v].ty, Ty::Tensor(_))
                    && changed.contains(v)
            })
        {
            return Err(FormationError::Unresolved(format!(
                "phase {index} mutates a retained value without an inter-item ownership or merge proof"
            )));
        }
        // Only source serial scopes publish bindings. Iteration-local values
        // stay in their domain even when its extent happens to equal one.
        let outputs = if vars.is_empty() {
            bound
                .difference(&parameters)
                .copied()
                .filter(|v| !inputs.contains(v) || changed.contains(v))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for &variable in &outputs {
            available.insert(variable);
            producers.entry(variable).or_insert(index);
        }
        phases.push(Phase {
            predecessor: index.checked_sub(1),
            inputs,
            outputs,
        });
        writes.push(changed);
    }
    let source_parameters = source.params.len();
    let mut retained = Vec::new();
    let mut restores = vec![Vec::new(); phases.len()];
    let mut publications = vec![Vec::new(); phases.len()];
    for (variable, (producer, consumers)) in captures {
        // Handoffs belong to the same original alternatives as the value they
        // carry. Keep those paths before adding any synthetic extent/bridge
        // bindings, whose identities have no independent source activity.
        let paths: Vec<_> = function.body.iter().map(|root| {
            let StmtKind::Parallel { body, .. } = &root.kind else { unreachable!() };
            capture_paths(body, variable, &function.vars, selectors)
        }).collect();
        let ty = function.vars[variable].ty.clone();
        let span = function.vars[variable].span;
        let logical = match &ty {
            Ty::Scalar(dtype) => Shaped::new(vec![Sym::constant(1)], Elem::Dtype(*dtype)),
            Ty::Tile(shape) | Ty::Frag(shape) if matches!(shape.elem, Elem::Dtype(_)) => {
                shape.clone()
            }
            _ => {
                return Err(FormationError::Unresolved(format!(
                    "cross-phase value `{}` requires retained value storage, found {ty}",
                    function.vars[variable].name
                )));
            }
        };
        let last = *consumers.last().unwrap();
        let writers: Vec<_> = (producer..last)
            .filter(|&phase| writes[phase].contains(&variable))
            .collect();
        let mut physical = logical.clone();
        let mut dimensions = Vec::new();
        for (axis, extent) in logical.shape.iter().enumerate() {
            if extent.as_constant().is_some() {
                dimensions.push(None);
                continue;
            }
            let capacity = match extent.eval_interval(&|name| numeric.get(name).copied()) {
                Some((minimum, maximum)) if minimum >= 0 => maximum,
                _ => variable_capacity(variable, axis, &function, &mut BTreeSet::new())?,
            };
            physical.shape[axis] = Sym::constant(capacity);
            let dimension = function.vars.len();
            function.vars.push(Var {
                name: format!("phase_extent_{variable}_{axis}"),
                ty: Ty::Scalar(DType::I32),
                span,
                kind: VarKind::Local,
            });
            let storage = allocate(
                &mut function,
                &mut retained,
                dimension,
                Shaped::new(vec![Sym::constant(1)], Elem::Dtype(DType::I32)),
                producer,
                &consumers,
                &writers,
            )?;
            for &phase in &consumers {
                restores[phase].extend(guard_handoff(&paths[phase].reads, vec![restore(
                    dimension,
                    element(storage, &function.vars, span),
                    &function.vars,
                    span,
                )], &function.vars, span));
                phases[phase].inputs.push(dimension);
            }
            for &phase in &writers {
                let value = Expr {
                    kind: ExprKind::Builtin {
                        name: Builtin::Extent,
                        args: vec![
                            reference(variable, &function.vars, span),
                            integer(axis as i64, span),
                        ],
                    },
                    ty: Ty::Scalar(DType::I32),
                    sym: None,
                    span,
                };
                let definition = Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Assign {
                        target: reference(dimension, &function.vars, span),
                        op: AssignOp::Assign,
                        value,
                    },
                };
                publications[phase].extend(guard_handoff(&paths[phase].writes, vec![definition, publish(
                    dimension,
                    element(storage, &function.vars, span),
                    &function.vars,
                    span,
                )], &function.vars, span));
                phases[phase].outputs.push(dimension);
            }
            dimensions.push(Some(dimension));
        }
        let storage = allocate(
            &mut function,
            &mut retained,
            variable,
            physical,
            producer,
            &consumers,
            &writers,
        )?;
        if let Ty::Frag(shape) = &ty {
            let Elem::Dtype(dtype) = shape.elem else {
                unreachable!()
            };
            let bridge = function.vars.len();
            function.vars.push(Var {
                name: format!("phase_fragment_{variable}"),
                ty: Ty::Tile(shape.clone()),
                span,
                kind: VarKind::Local,
            });
            let memory = reference(storage, &function.vars, span);
            for &phase in &consumers {
                let allocation = Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Assign {
                        target: reference(variable, &function.vars, span),
                        op: AssignOp::Assign,
                        value: Expr {
                            kind: ExprKind::Intrinsic {
                                op: seismic_lang::intrinsics::Operation::Matrix,
                                args: vec![Expr {
                                    kind: ExprKind::Int(0),
                                    ty: Ty::Scalar(dtype),
                                    sym: None,
                                    span,
                                }],
                            },
                            ty: ty.clone(),
                            sym: None,
                            span,
                        },
                    },
                };
                restores[phase].extend(guard_handoff(&paths[phase].reads, vec![
                    restore(bridge, memory.clone(), &function.vars, span),
                    allocation,
                    fragment_transfer(
                    seismic_lang::intrinsics::Operation::MatrixLoad,
                    variable,
                    bridge,
                    &function.vars,
                    span,
                )], &function.vars, span));
            }
            for &phase in &writers {
                let allocation = Stmt {
                    id: None,
                    span,
                    kind: StmtKind::Assign {
                        target: reference(bridge, &function.vars, span),
                        op: AssignOp::Assign,
                        value: Expr {
                            kind: ExprKind::TileAlloc {
                                shape: shape.shape.clone(),
                                dtype: shape.elem.clone(),
                            },
                            ty: Ty::Tile(shape.clone()),
                            sym: None,
                            span,
                        },
                    },
                };
                publications[phase].extend(guard_handoff(&paths[phase].writes, vec![allocation, fragment_transfer(
                    seismic_lang::intrinsics::Operation::MatrixStore,
                    variable,
                    bridge,
                    &function.vars,
                    span,
                ), publish(bridge, memory.clone(), &function.vars, span)], &function.vars, span));
            }
            continue;
        }
        let view = if matches!(ty, Ty::Scalar(_)) {
            element(storage, &function.vars, span)
        } else if dimensions.iter().all(Option::is_none) {
            reference(storage, &function.vars, span)
        } else {
            Expr {
                kind: ExprKind::Index {
                    base: Box::new(reference(storage, &function.vars, span)),
                    indices: dimensions
                        .iter()
                        .map(|dimension| Index::Slice {
                            start: None,
                            end: dimension.map(|v| reference(v, &function.vars, span)),
                        })
                        .collect(),
                },
                ty: Ty::Tensor(logical),
                sym: None,
                span,
            }
        };
        for &phase in &consumers {
            restores[phase].extend(guard_handoff(&paths[phase].reads,
                vec![restore(variable, view.clone(), &function.vars, span)], &function.vars, span));
        }
        for &phase in &writers {
            publications[phase].extend(guard_handoff(&paths[phase].writes,
                vec![publish(variable, view.clone(), &function.vars, span)], &function.vars, span));
        }
    }
    let handoffs = restores.iter().zip(&publications).map(|(restores, publications)|
        PhaseHandoff { restores: restores.clone(), publications: publications.clone() }).collect();
    for (phase, root) in function.body.iter_mut().enumerate() {
        let StmtKind::Parallel { body, .. } = &mut root.kind else {
            unreachable!()
        };
        let mut completed = std::mem::take(&mut restores[phase]);
        completed.append(body);
        completed.append(&mut publications[phase]);
        *body = completed;
    }
    // Each launch is independently scoped. Check the exact transformed phase
    // bodies, rather than trusting the capture bookkeeping alone.
    verify_with_parameters(&function, numeric, selectors)?;
    Ok(PhasePlan {
        function,
        phases,
        retained,
        source_parameters,
        handoffs,
    })
}

type CompilerPath = Vec<(VarId, bool)>;

#[derive(Default)]
struct CapturePaths {
    reads: Vec<CompilerPath>,
    writes: Vec<CompilerPath>,
}

/// Project source activity without turning runtime control into a compiler
/// choice. Inspect headers separately so an inner alternative does not make
/// the enclosing path look unconditionally active.
fn capture_paths(body: &[Stmt], variable: VarId, vars: &[Var], selectors: &BTreeSet<VarId>) -> CapturePaths {
    fn visit(body: &[Stmt], variable: VarId, vars: &[Var], selectors: &BTreeSet<VarId>, path: &CompilerPath, out: &mut CapturePaths) {
        for statement in body {
            let mut header = statement.clone();
            match &mut header.kind {
                StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => body.clear(),
                StmtKind::If { then, els, .. } => { then.clear(); els.clear(); }
                StmtKind::Reduction(reduction) => {
                    for implementation in reduction.implementations_mut() { implementation.body.clear(); }
                }
                StmtKind::Assign { .. } | StmtKind::Expr(_) => {}
            }
            let mut written = HashSet::new();
            seismic_lang::rewrite::value_writes(&header, vars, &mut written);
            if written.contains(&variable) { out.writes.push(path.clone()); }
            // A complete scalar assignment establishes its value. A tile
            // assignment also consumes the retained destination geometry.
            let reads = match &header.kind {
                StmtKind::Assign { target: Expr { kind: ExprKind::Var(v), .. }, op: AssignOp::Assign, value }
                    if *v == variable && !matches!(vars[variable].ty, Ty::Tile(_)) => {
                        seismic_lang::effects::uses(&Stmt { id: None, span: header.span, kind: StmtKind::Expr(value.clone()) }, variable)
                    }
                _ => seismic_lang::effects::uses(&header, variable),
            };
            if reads { out.reads.push(path.clone()); }
            match &statement.kind {
                StmtKind::If { cond, then, els } => {
                    let selector = match cond.kind {
                        ExprKind::Var(selector) if selectors.contains(&selector) => Some(selector),
                        _ => None,
                    };
                    if let Some(selector) = selector {
                        for (branch, truth) in [(then, true), (els, false)] {
                            let mut nested = path.clone();
                            match nested.binary_search_by_key(&selector, |&(v, _)| v) {
                                Ok(index) if nested[index].1 != truth => continue,
                                Ok(_) => {}
                                Err(index) => nested.insert(index, (selector, truth)),
                            }
                            visit(branch, variable, vars, selectors, &nested, out);
                        }
                    } else {
                        visit(then, variable, vars, selectors, path, out);
                        visit(els, variable, vars, selectors, path, out);
                    }
                }
                StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, variable, vars, selectors, path, out),
                StmtKind::Reduction(reduction) => {
                    for body in reduction.bodies() { visit(body, variable, vars, selectors, path, out); }
                }
                StmtKind::Assign { .. } | StmtKind::Expr(_) => {}
            }
        }
    }
    let mut out = CapturePaths::default();
    visit(body, variable, vars, selectors, &Vec::new(), &mut out);
    out.reads.sort();
    out.reads.dedup();
    out.writes.sort();
    out.writes.dedup();
    out
}

/// A union of original paths executes one handoff, even when the source uses
/// the value in multiple overlapping alternatives. Branches test the original
/// selector identities, so selection removes inactive transfers with the source.
fn guard_handoff(paths: &[CompilerPath], body: Vec<Stmt>, vars: &[Var], span: Span) -> Vec<Stmt> {
    if paths.iter().any(Vec::is_empty) { return body; }
    let Some(&(selector, _)) = paths.first().and_then(|path| path.first()) else { return Vec::new() };
    let branch = |truth| {
        let paths: Vec<_> = paths.iter().filter_map(|path| {
            match path.binary_search_by_key(&selector, |&(v, _)| v) {
                Ok(index) if path[index].1 != truth => None,
                Ok(index) => {
                    let mut path = path.clone();
                    path.remove(index);
                    Some(path)
                }
                Err(_) => Some(path.clone()),
            }
        }).collect();
        guard_handoff(&paths, body.clone(), vars, span)
    };
    vec![Stmt { id: None, span, kind: StmtKind::If {
        cond: reference(selector, vars, span), then: branch(true), els: branch(false),
    } }]
}

fn fragment_transfer(
    operation: seismic_lang::intrinsics::Operation,
    fragment: VarId,
    tile: VarId,
    vars: &[Var],
    span: Span,
) -> Stmt {
    Stmt {
        id: None,
        span,
        kind: StmtKind::Expr(Expr {
            kind: ExprKind::Intrinsic {
                op: operation,
                args: vec![
                    reference(fragment, vars, span),
                    reference(tile, vars, span),
                    integer(0, span),
                    integer(0, span),
                ],
            },
            ty: Ty::Void,
            sym: None,
            span,
        }),
    }
}

/// Tensor values are captured views, not snapshots of their referents. Retain
/// their evaluated coordinates as scalars, then reconstruct the same view in
/// each consuming launch. In particular, a coordinate loaded from a mutable
/// tensor is read once at the original view definition.
struct ViewRecipe {
    variable: VarId,
    value: Expr,
    path: CompilerPath,
    /// Ordinary bool values evaluated in the producing serial phase. These
    /// never join the original compiler selectors or change family coverage.
    runtime: Vec<(VarId, bool)>,
}

fn close_views(function: &mut LoweredIr, selectors: &BTreeSet<VarId>) -> Result<Vec<usize>, FormationError> {
    let mut recipes: Vec<ViewRecipe> = Vec::new();
    let original = function.body.clone();
    let required = retained_views(&original, &function.vars)?;
    let mut prefixes = Vec::new();
    for phase in 0..function.body.len() {
        let StmtKind::Parallel {
            vars: indices,
            extents,
            body,
        } = &mut function.body[phase].kind
        else {
            unreachable!()
        };
        let mut demands: BTreeMap<VarId, Vec<CompilerPath>> = BTreeMap::new();
        for recipe in &recipes {
            demands.entry(recipe.variable).or_insert_with(|| {
                capture_paths(body, recipe.variable, &function.vars, selectors).reads
            });
        }
        let mut symbols = control_paths(body, selectors);
        for symbol in extents.iter().flat_map(Sym::params) {
            symbols.entry(symbol).or_default().push(Vec::new());
        }
        for (symbol, paths) in &symbols {
            let owners = recipes
                .iter()
                .filter(|recipe| {
                    recipe.value.ty.shaped().is_some_and(|shape| {
                        shape
                            .shape
                            .iter()
                            .any(|extent| extent.params().contains(symbol))
                    })
                })
                .collect::<Vec<_>>();
            if let Some(recipe) = owners.first().copied() {
                // Several source versions of one view retain their source
                // order. Distinct view identities must still agree on the
                // meaning of a shared runtime geometry symbol.
                if owners.iter().any(|other| other.variable != recipe.variable && other.value != recipe.value) {
                    return Err(FormationError::Unresolved(format!(
                        "runtime control symbol `{symbol}` has multiple captured geometry versions"
                    )));
                }
                demands.entry(recipe.variable).or_default().extend(paths.iter().cloned());
            }
        }
        // Resolve dependencies backwards through their original definition
        // order. An alias observes exactly the backing version available when
        // it was defined, including when a later source assignment rebinds it.
        let mut definitions = Vec::new();
        for recipe in recipes.iter().rev() {
            let mut paths: Vec<_> = demands.get(&recipe.variable).into_iter().flatten()
                .filter_map(|path| intersect_paths(path, &recipe.path)).collect();
            paths.sort();
            paths.dedup();
            if paths.is_empty() { continue; }
            for variable in view_backings(&recipe.value) {
                let demand = demands.entry(variable).or_default();
                demand.extend(paths.iter().cloned());
                demand.sort();
                demand.dedup();
            }
            let span = function.vars[recipe.variable].span;
            let mut definition = vec![Stmt {
                id: None, span, kind: StmtKind::Assign {
                    target: reference(recipe.variable, &function.vars, span),
                    op: AssignOp::Assign, value: recipe.value.clone(),
                },
            }];
            for &(predicate, truth) in recipe.runtime.iter().rev() {
                let (then, els) = if truth { (definition, Vec::new()) } else { (Vec::new(), definition) };
                definition = vec![Stmt { id: None, span, kind: StmtKind::If {
                    cond: reference(predicate, &function.vars, span), then, els,
                } }];
            }
            definitions.push(guard_handoff(&paths, definition, &function.vars, span));
        }
        let mut prefix: Vec<_> = definitions.into_iter().rev().flatten().collect();
        prefixes.push(prefix.len());
        if indices.is_empty() {
            let mut defaults = Vec::new();
            let frozen = freeze_views(std::mem::take(body), &Vec::new(), &[], &required,
                &mut function.vars, &mut recipes, selectors, &mut defaults)?;
            prefix.extend(defaults);
            prefix.extend(frozen);
        } else {
            prefix.append(body);
        }
        *body = prefix;
    }
    Ok(prefixes)
}

fn intersect_paths(left: &CompilerPath, right: &CompilerPath) -> Option<CompilerPath> {
    let mut path = left.clone();
    for &(selector, truth) in right {
        match path.binary_search_by_key(&selector, |&(v, _)| v) {
            Ok(index) if path[index].1 != truth => return None,
            Ok(_) => {}
            Err(index) => path.insert(index, (selector, truth)),
        }
    }
    Some(path)
}

/// Only the backing chain belongs to a view recipe. Coordinate expressions
/// execute at the source definition and are captured separately as values.
fn view_backings(expr: &Expr) -> Vec<VarId> {
    match &expr.kind {
        ExprKind::Var(variable) => vec![*variable],
        ExprKind::Index { base, .. } | ExprKind::Transpose(base) => view_backings(base),
        ExprKind::Builtin { name: Builtin::Reshape, args } => args.first().map_or_else(Vec::new, view_backings),
        _ => Vec::new(),
    }
}

/// Close the set of escaping views over local backing definitions before
/// freezing them, so a local alias chain can cross a launch as one snapshot.
fn retained_views(body: &[Stmt], vars: &[Var]) -> Result<BTreeSet<VarId>, FormationError> {
    fn definitions<'a>(body: &'a [Stmt], phase: usize, serial: bool, out: &mut Vec<(usize, VarId, &'a Expr, bool)>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), ty: Ty::Tensor(_), .. }, op: AssignOp::Assign, value } => {
                    out.push((phase, *variable, value, serial));
                }
                StmtKind::If { then, els, .. } => {
                    definitions(then, phase, serial, out);
                    definitions(els, phase, serial, out);
                }
                StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => definitions(body, phase, false, out),
                StmtKind::Reduction(reduction) => {
                    for body in reduction.bodies() { definitions(body, phase, false, out); }
                }
                _ => {}
            }
        }
    }
    let mut all = Vec::new();
    for (phase, root) in body.iter().enumerate() {
        let StmtKind::Parallel { vars, body, .. } = &root.kind else { unreachable!() };
        // Parallel-local view assignments never publish a source binding.
        // This includes reconstruction prefixes from an earlier domain pass.
        if vars.is_empty() { definitions(body, phase, true, &mut all); }
    }
    let symbols: Vec<_> = body.iter().map(|root| control_symbols(std::slice::from_ref(root))).collect();
    let mut required = BTreeSet::new();
    for &(phase, variable, _, serial) in &all {
        if body[phase + 1..].iter().enumerate().any(|(later, root)| {
            seismic_lang::effects::uses(root, variable) || (serial && vars[variable].ty.shaped().is_some_and(|shape| {
                shape.shape.iter().flat_map(Sym::params).any(|symbol| symbols[phase + 1 + later].contains(&symbol))
            }))
        }) { required.insert(variable); }
    }
    loop {
        let mut dependencies = BTreeSet::new();
        for &(_, variable, value, _) in &all {
            if required.contains(&variable) {
                dependencies.extend(view_backings(value).into_iter().filter(|&v| !matches!(vars[v].kind, VarKind::Param(_))));
            }
        }
        let previous = required.len();
        required.extend(dependencies);
        if previous == required.len() { break; }
    }
    if all.iter().any(|(_, variable, _, serial)| required.contains(variable) && !serial) {
        return Err(FormationError::Unresolved("retained tensor view definition inside an iteration requires an explicit escaping binding".into()));
    }
    Ok(required)
}

fn freeze_views(body: Vec<Stmt>, path: &CompilerPath, runtime: &[(VarId, bool)], required: &BTreeSet<VarId>, vars: &mut Vec<Var>, recipes: &mut Vec<ViewRecipe>, selectors: &BTreeSet<VarId>, defaults: &mut Vec<Stmt>) -> Result<Vec<Stmt>, FormationError> {
    let mut frozen = Vec::new();
    for mut statement in body {
        match &mut statement.kind {
            StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), ty: Ty::Tensor(_), .. }, op: AssignOp::Assign, value }
                if required.contains(variable) => {
                    let first = vars.len();
                    let recipe = freeze_view(value, vars, recipes, &mut frozen)?;
                    if !runtime.is_empty() {
                        // A coordinate exists only when its original branch
                        // executes. Give its private handoff slot a harmless
                        // value on the other path; reconstruction remains
                        // guarded and never observes that inactive coordinate.
                        for coordinate in first..vars.len() {
                            if matches!(vars[coordinate].ty, Ty::Scalar(DType::I32 | DType::U32)) {
                                default_capture(coordinate, path, vars, defaults);
                            }
                        }
                    }
                    recipes.push(ViewRecipe { variable: *variable, value: recipe, path: path.clone(), runtime: runtime.to_vec() });
                }
            StmtKind::If { cond, then, els } => {
                let selector = match cond.kind {
                    ExprKind::Var(selector) if selectors.contains(&selector) => Some(selector),
                    _ => None,
                };
                if let Some(selector) = selector {
                    for (branch, truth) in [(then, true), (els, false)] {
                        if let Some(path) = intersect_paths(path, &vec![(selector, truth)]) {
                            *branch = freeze_views(std::mem::take(branch), &path, runtime, required, vars, recipes, selectors, defaults)?;
                        }
                    }
                } else if contains_retained_view(then, required) || contains_retained_view(els, required) {
                    let predicate = vars.len();
                    vars.push(Var { name: format!("phase_predicate_{predicate}"), ty: Ty::Scalar(DType::Bool),
                        span: cond.span, kind: VarKind::Local });
                    default_capture(predicate, path, vars, defaults);
                    let target = reference(predicate, vars, cond.span);
                    let value = std::mem::replace(cond, target.clone());
                    frozen.push(Stmt { id: None, span: value.span, kind: StmtKind::Assign {
                        target, op: AssignOp::Assign, value,
                    } });
                    for (branch, truth) in [(then, true), (els, false)] {
                        let mut nested = runtime.to_vec();
                        nested.push((predicate, truth));
                        *branch = freeze_views(std::mem::take(branch), path, &nested, required, vars, recipes, selectors, defaults)?;
                    }
                }
            }
            _ => {}
        }
        frozen.push(statement);
    }
    Ok(frozen)
}

fn contains_retained_view(body: &[Stmt], required: &BTreeSet<VarId>) -> bool {
    body.iter().any(|statement| match &statement.kind {
        StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), ty: Ty::Tensor(_), .. }, op: AssignOp::Assign, .. } => required.contains(variable),
        StmtKind::If { then, els, .. } => contains_retained_view(then, required) || contains_retained_view(els, required),
        _ => false,
    })
}

/// Only compiler-generated scalar slots receive defaults. Original source
/// predicates and coordinates are still evaluated at their original location,
/// and compiler-inactive alternatives allocate no publication or restore.
fn default_capture(variable: VarId, path: &CompilerPath, vars: &[Var], defaults: &mut Vec<Stmt>) {
    let span = vars[variable].span;
    let value = Expr {
        kind: if vars[variable].ty == Ty::Scalar(DType::Bool) { ExprKind::Bool(false) } else { ExprKind::Int(0) },
        ty: vars[variable].ty.clone(), sym: None, span,
    };
    defaults.extend(guard_handoff(std::slice::from_ref(path), vec![Stmt { id: None, span, kind: StmtKind::Assign {
        target: reference(variable, vars, span), op: AssignOp::Assign, value,
    } }], vars, span));
}

fn control_paths(body: &[Stmt], selectors: &BTreeSet<VarId>) -> BTreeMap<String, Vec<CompilerPath>> {
    fn visit(body: &[Stmt], path: &CompilerPath, selectors: &BTreeSet<VarId>, out: &mut BTreeMap<String, Vec<CompilerPath>>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Range { lo, hi, .. } => {
                    for symbol in lo.params().into_iter().chain(hi.params()) { out.entry(symbol).or_default().push(path.clone()); }
                }
                StmtKind::Lanes { extent, .. } => {
                    for symbol in extent.params() { out.entry(symbol).or_default().push(path.clone()); }
                }
                StmtKind::Parallel { extents, .. } => {
                    for symbol in extents.iter().flat_map(Sym::params) { out.entry(symbol).or_default().push(path.clone()); }
                }
                _ => {}
            }
            match &statement.kind {
                StmtKind::If { cond, then, els } => {
                    let selector = match cond.kind {
                        ExprKind::Var(selector) if selectors.contains(&selector) => Some(selector),
                        _ => None,
                    };
                    if let Some(selector) = selector {
                        for (branch, truth) in [(then, true), (els, false)] {
                            if let Some(path) = intersect_paths(path, &vec![(selector, truth)]) { visit(branch, &path, selectors, out); }
                        }
                    } else { visit(then, path, selectors, out); visit(els, path, selectors, out); }
                }
                StmtKind::Parallel { body, .. } | StmtKind::LoadLoop { body, .. }
                | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. } => visit(body, path, selectors, out),
                StmtKind::Reduction(reduction) => {
                    for body in reduction.bodies() { visit(body, path, selectors, out); }
                }
                _ => {}
            }
        }
    }
    let mut paths = BTreeMap::new();
    visit(body, &Vec::new(), selectors, &mut paths);
    paths
}

fn control_symbols(body: &[Stmt]) -> BTreeSet<String> {
    control_paths(body, &BTreeSet::new()).into_keys().collect()
}

/// Physical work domains retain a structural capacity and guard their logical
/// runtime extent. This is shared by dispatch-domain choice construction and
/// full phase formation, so mapping choices see the same bounded domain.
pub fn work_domains(source: &LoweredIr) -> Result<LoweredIr, String> {
    work_domains_checked(source).map_err(|error| error.to_string())
}
pub fn work_domains_parameterized(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>) -> Result<LoweredIr, String> {
    work_domains_retained(source, numeric, &BTreeSet::new())
}
pub fn work_domains_retained(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>, selectors: &BTreeSet<VarId>) -> Result<LoweredIr, String> {
    work_domains_with_parameters(source, numeric, selectors).map_err(|error| error.to_string())
}
fn work_domains_checked(source: &LoweredIr) -> Result<LoweredIr, FormationError> {
    work_domains_with_parameters(source, &BTreeMap::new(), &BTreeSet::new())
}
fn work_domains_with_parameters(source: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>, selectors: &BTreeSet<VarId>) -> Result<LoweredIr, FormationError> {
    let mut function = source.clone();
    seismic_lang::normalize::work_domain(&mut function.body);
    let prefixes = close_views(&mut function, selectors)?;
    for phase in 0..function.body.len() {
        let StmtKind::Parallel { vars, extents, .. } = &function.body[phase].kind else {
            unreachable!()
        };
        let mut bounded = Vec::new();
        let mut condition = None;
        for (&variable, extent) in vars.iter().zip(extents) {
            if let Some(value) = extent.as_constant() {
                bounded.push(Sym::constant(value));
                continue;
            }
            let capacity = extent_capacity(extent, &function, numeric)?;
            bounded.push(Sym::constant(capacity));
            let span = function.vars[variable].span;
            let limit = Expr {
                kind: ExprKind::ShapeParam(format!("phase_{phase}_extent")),
                ty: Ty::Scalar(DType::I32),
                sym: Some(extent.clone()),
                span,
            };
            let active = Expr {
                kind: ExprKind::Binary {
                    op: seismic_lang::ast::BinaryOp::Lt,
                    lhs: Box::new(reference(variable, &function.vars, span)),
                    rhs: Box::new(limit),
                },
                ty: Ty::Scalar(DType::Bool),
                sym: None,
                span,
            };
            condition = Some(match condition {
                None => active,
                Some(previous) => Expr {
                    kind: ExprKind::Binary {
                        op: seismic_lang::ast::BinaryOp::And,
                        lhs: Box::new(previous),
                        rhs: Box::new(active),
                    },
                    ty: Ty::Scalar(DType::Bool),
                    sym: None,
                    span,
                },
            });
        }
        if let Some(cond) = condition {
            let StmtKind::Parallel { extents, body, .. } = &mut function.body[phase].kind else {
                unreachable!()
            };
            *extents = bounded;
            let then = body.split_off(prefixes[phase]);
            body.push(Stmt {
                id: None,
                span: cond.span,
                kind: StmtKind::If {
                    cond,
                    then,
                    els: Vec::new(),
                },
            });
        }
    }
    Ok(function)
}
fn extent_capacity(extent: &Sym, function: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>) -> Result<i64, FormationError> {
    let mut intervals = BTreeMap::new();
    for symbol in extent.params() {
        if let Some(&bounds) = numeric.get(&symbol) { intervals.insert(symbol, bounds); continue; }
        if let Some(&value) = function.shapes.get(&symbol) {
            intervals.insert(symbol, (value, value));
            continue;
        }
        if let Some((_, upper)) = function
            .index_params
            .iter()
            .find(|(name, _)| *name == symbol)
        {
            if let Some(upper) = upper.as_constant() {
                intervals.insert(symbol, (0, upper.saturating_sub(1)));
                continue;
            }
        }
        let mut capacity = None;
        for (variable, var) in function.vars.iter().enumerate() {
            if let Some(shape) = var.ty.shaped() {
                for (axis, dimension) in shape.shape.iter().enumerate() {
                    if dimension == &Sym::param(&symbol) {
                        match variable_capacity(variable, axis, function, &mut BTreeSet::new()) {
                            Ok(bound) => {
                                capacity = Some(
                                    capacity.map_or(bound, |previous: i64| previous.max(bound)),
                                )
                            }
                            Err(FormationError::Unresolved(_)) => {}
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
        }
        let capacity = capacity.ok_or_else(|| {
            FormationError::Unresolved(format!(
                "parallel extent `{extent}` has no structural bound for `{symbol}`"
            ))
        })?;
        intervals.insert(symbol, (0, capacity));
    }
    let (minimum, maximum) = extent
        .eval_interval(&|name| intervals.get(name).copied())
        .ok_or_else(|| {
            FormationError::Unresolved(format!("parallel extent `{extent}` has no finite capacity"))
        })?;
    if minimum < 0 || maximum > i64::from(i32::MAX) {
        return Err(FormationError::Unresolved(format!(
            "parallel extent `{extent}` does not have a nonnegative i32 capacity"
        )));
    }
    Ok(maximum)
}

fn freeze_view(
    expr: &mut Expr,
    vars: &mut Vec<Var>,
    recipes: &[ViewRecipe],
    setup: &mut Vec<Stmt>,
) -> Result<Expr, FormationError> {
    let mut recipe = expr.clone();
    match (&mut expr.kind, &mut recipe.kind) {
        (ExprKind::Var(variable), _) => {
            if recipes.iter().any(|recipe| recipe.variable == *variable) {
                return Ok(recipe);
            }
            if !matches!(
                vars.get(*variable)
                    .ok_or("retained view has an invalid variable identity")?
                    .kind,
                VarKind::Param(_)
            ) {
                return Err(FormationError::Unresolved(
                    "retained tensor view has no captured backing identity".into(),
                ));
            }
        }
        (
            ExprKind::Index { base, indices },
            ExprKind::Index {
                base: retained_base,
                indices: retained_indices,
            },
        ) => {
            **retained_base = freeze_view(base, vars, recipes, setup)?;
            capture_geometry(base, vars, setup);
            for index in indices.iter_mut() {
                match index {
                    Index::Point(point) => freeze_coordinate(point, vars, setup)?,
                    Index::Slice { start, end } => {
                        for value in start.iter_mut().chain(end) {
                            freeze_coordinate(value, vars, setup)?;
                        }
                    }
                }
            }
            *retained_indices = indices.clone();
        }
        (ExprKind::Transpose(base), ExprKind::Transpose(retained_base)) => {
            **retained_base = freeze_view(base, vars, recipes, setup)?;
        }
        (
            ExprKind::Builtin {
                name: Builtin::Reshape,
                args,
            },
            ExprKind::Builtin {
                args: retained_args,
                ..
            },
        ) => {
            let (base, dimensions) = args
                .split_first_mut()
                .ok_or("retained reshape has no backing")?;
            let retained_base = freeze_view(base, vars, recipes, setup)?;
            capture_geometry(base, vars, setup);
            for dimension in dimensions {
                freeze_coordinate(dimension, vars, setup)?;
            }
            *retained_args = args.clone();
            retained_args[0] = retained_base;
        }
        _ => {
            return Err(FormationError::Unresolved(
                "retained tensor view needs ordinary indexing, transpose, or reshape geometry"
                    .into(),
            ));
        }
    }
    Ok(recipe)
}
fn capture_geometry(expr: &mut Expr, vars: &mut Vec<Var>, setup: &mut Vec<Stmt>) {
    if matches!(expr.kind, ExprKind::Var(_)) {
        return;
    }
    let variable = vars.len();
    vars.push(Var {
        name: format!("phase_geometry_{variable}"),
        ty: expr.ty.clone(),
        span: expr.span,
        kind: VarKind::Local,
    });
    let target = reference(variable, vars, expr.span);
    let value = std::mem::replace(expr, target.clone());
    setup.push(Stmt {
        id: None,
        span: value.span,
        kind: StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        },
    });
}

fn freeze_coordinate(
    expr: &mut Expr,
    vars: &mut Vec<Var>,
    setup: &mut Vec<Stmt>,
) -> Result<(), String> {
    if matches!(expr.kind, ExprKind::Int(_) | ExprKind::ShapeParam(_)) {
        return Ok(());
    }
    if !matches!(expr.ty, Ty::Scalar(DType::I32 | DType::U32)) {
        return Err("retained view coordinate must have integer type".into());
    }
    let variable = vars.len();
    vars.push(Var {
        name: format!("phase_coordinate_{variable}"),
        ty: expr.ty.clone(),
        span: expr.span,
        kind: VarKind::Local,
    });
    let target = reference(variable, vars, expr.span);
    let value = std::mem::replace(expr, target.clone());
    setup.push(Stmt {
        id: None,
        span: value.span,
        kind: StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        },
    });
    Ok(())
}

fn parameter_variables(function: &LoweredIr) -> Result<BTreeSet<VarId>, String> {
    let mut result = BTreeSet::new();
    for (id, var) in function.vars.iter().enumerate() {
        if let VarKind::Param(parameter) = var.kind {
            if function
                .params
                .get(parameter)
                .is_none_or(|(_, ty)| ty != &var.ty)
            {
                return Err(format!(
                    "variable `{}` disagrees with its parameter type",
                    var.name
                ));
            }
            result.insert(id);
        }
    }
    Ok(result)
}

/// Validate independent launch scope, variable types, condition types, and
/// participation index bindings. Backend collective checks refine this contract.
pub fn verify(function: &LoweredIr) -> Result<(), String> {
    verify_parameterized(function, &BTreeMap::new())
}
pub fn verify_parameterized(function: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>) -> Result<(), String> {
    verify_with_parameters(function, numeric, &BTreeSet::new())
}
fn verify_with_parameters(function: &LoweredIr, numeric: &BTreeMap<String, (i64, i64)>, selectors: &BTreeSet<VarId>) -> Result<(), String> {
    let mut parameters = parameter_variables(function)?;
    parameters.extend(selectors);
    for (phase, statement) in function.body.iter().enumerate() {
        let StmtKind::Parallel {
            vars,
            extents,
            body,
        } = &statement.kind
        else {
            return Err("execution phase is not a work domain".into());
        };
        if vars.len() != extents.len() {
            return Err("phase index/extent arity differs".into());
        }
        let mut bound = parameters.clone();
        let mut scope = Scope {
            function,
            numeric,
            selectors,
            inputs: BTreeSet::new(),
            symbols: BTreeSet::new(),
        };
        scope.indices(vars, &mut bound)?;
        scope.body(body, &mut bound)?;
        if let Some(symbol) = scope.unbound_symbol(&bound) {
            return Err(format!(
                "phase {phase} has an unbound runtime symbol `{symbol}`"
            ));
        }
        if !scope.inputs.is_empty() {
            return Err(format!(
                "phase {phase} has unbound inputs {:?}",
                scope.inputs
            ));
        }
    }
    Ok(())
}

fn reference(variable: VarId, vars: &[Var], span: Span) -> Expr {
    Expr {
        kind: ExprKind::Var(variable),
        ty: vars[variable].ty.clone(),
        sym: None,
        span,
    }
}
fn element(storage: VarId, vars: &[Var], span: Span) -> Expr {
    let Ty::Tensor(shape) = &vars[storage].ty else {
        unreachable!()
    };
    Expr {
        kind: ExprKind::Index {
            base: Box::new(reference(storage, vars, span)),
            indices: vec![Index::Point(Expr {
                kind: ExprKind::Int(0),
                ty: Ty::Scalar(DType::I32),
                sym: Some(Sym::constant(0)),
                span,
            })],
        },
        ty: Ty::Scalar(shape.elem.read_dtype().unwrap()),
        sym: None,
        span,
    }
}
fn integer(value: i64, span: Span) -> Expr {
    Expr {
        kind: ExprKind::Int(value),
        ty: Ty::Scalar(DType::I32),
        sym: Some(Sym::constant(value)),
        span,
    }
}
fn restore(variable: VarId, view: Expr, vars: &[Var], span: Span) -> Stmt {
    let value = if matches!(vars[variable].ty, Ty::Scalar(_)) {
        view
    } else {
        Expr {
            kind: ExprKind::Load {
                view: Box::new(view),
                mode: LoadMode::Materialize,
            },
            ty: vars[variable].ty.clone(),
            sym: None,
            span,
        }
    };
    Stmt {
        id: None,
        span,
        kind: StmtKind::Assign {
            target: reference(variable, vars, span),
            op: AssignOp::Assign,
            value,
        },
    }
}
fn publish(variable: VarId, view: Expr, vars: &[Var], span: Span) -> Stmt {
    let value = reference(variable, vars, span);
    Stmt {
        id: None,
        span,
        kind: if matches!(vars[variable].ty, Ty::Scalar(_)) {
            StmtKind::Assign {
                target: view,
                op: AssignOp::Assign,
                value,
            }
        } else {
            StmtKind::Expr(Expr {
                kind: ExprKind::Builtin {
                    name: Builtin::Store,
                    args: vec![value, view],
                },
                ty: Ty::Void,
                sym: None,
                span,
            })
        },
    }
}
fn allocate(
    function: &mut LoweredIr,
    retained: &mut Vec<RetainedValue>,
    variable: VarId,
    shape: Shaped,
    producer: usize,
    consumers: &[usize],
    writers: &[usize],
) -> Result<VarId, String> {
    let Elem::Dtype(dtype) = shape.elem else {
        return Err("retained storage requires resolved dense planes".into());
    };
    let elements = shape.shape.iter().try_fold(1u64, |n, extent| {
        extent
            .as_constant()
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|extent| n.checked_mul(extent))
            .ok_or("retained value capacity must be static and fit u64")
    })?;
    let bytes = elements
        .checked_mul(u64::from(dtype.bytes()))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("retained value byte capacity overflow")?;
    let parameter = function.params.len();
    let mut name = format!("__seismic_retained_{variable}");
    while function.params.iter().any(|(n, _)| n == &name) {
        name.push('_');
    }
    let buffer_ty = Ty::Tensor(shape);
    function.params.push((name.clone(), buffer_ty.clone()));
    function.ownership.intermediates.insert(name.clone());
    let storage = function.vars.len();
    function.vars.push(Var {
        name: name.clone(),
        ty: buffer_ty,
        span: function.vars[variable].span,
        kind: VarKind::Param(parameter),
    });
    retained.push(RetainedValue {
        variable,
        parameter,
        name,
        dtype,
        elements,
        bytes,
        producer,
        consumers: consumers.to_vec(),
        writers: writers.to_vec(),
    });
    Ok(storage)
}

fn variable_capacity(
    variable: VarId,
    axis: usize,
    function: &LoweredIr,
    visiting: &mut BTreeSet<VarId>,
) -> Result<i64, FormationError> {
    let extent = function.vars[variable]
        .ty
        .shaped()
        .and_then(|s| s.shape.get(axis))
        .ok_or("retained value axis is absent")?;
    if let Some(n) = extent.as_constant() {
        return Ok(n);
    }
    if !visiting.insert(variable) {
        return Err(FormationError::Unresolved(
            "retained shape capacity depends on its own value".into(),
        ));
    }
    fn definitions<'a>(body: &'a [Stmt], variable: VarId, out: &mut Vec<&'a Expr>) {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign {
                    target:
                        Expr {
                            kind: ExprKind::Var(v),
                            ..
                        },
                    op: AssignOp::Assign,
                    value,
                } if *v == variable => out.push(value),
                StmtKind::Parallel { body, .. }
                | StmtKind::Owned { body, .. }
                | StmtKind::Range { body, .. }
                | StmtKind::Lanes { body, .. }
                | StmtKind::LoadLoop { body, .. } => definitions(body, variable, out),
                StmtKind::If { then, els, .. } => {
                    definitions(then, variable, out);
                    definitions(els, variable, out);
                }
                _ => {}
            }
        }
    }
    let mut values = Vec::new();
    definitions(&function.body, variable, &mut values);
    let result = values
        .into_iter()
        .map(|expr| expression_capacity(expr, axis, function, visiting))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| {
            FormationError::Unresolved("retained runtime shape has no bounded definition".into())
        });
    visiting.remove(&variable);
    Ok(result?)
}
fn expression_capacity(
    expr: &Expr,
    axis: usize,
    function: &LoweredIr,
    visiting: &mut BTreeSet<VarId>,
) -> Result<i64, FormationError> {
    let shape = expr
        .ty
        .shaped()
        .ok_or("retained capacity needs a shaped value")?;
    if let Some(n) = shape.shape.get(axis).and_then(Sym::as_constant) {
        return Ok(n);
    }
    match &expr.kind {
        ExprKind::Var(v) => variable_capacity(*v, axis, function, visiting),
        ExprKind::Load { view, .. } => expression_capacity(view, axis, function, visiting),
        ExprKind::Builtin {
            name: Builtin::Load,
            args,
        } => expression_capacity(&args[0], axis, function, visiting),
        ExprKind::Transpose(base) => {
            expression_capacity(base, shape.shape.len() - 1 - axis, function, visiting)
        }
        ExprKind::Index { base, indices } => {
            let rank = base
                .ty
                .shaped()
                .ok_or("retained view needs a shaped parent")?
                .shape
                .len();
            let parent = (0..rank)
                .filter(|i| !matches!(indices.get(*i), Some(Index::Point(_))))
                .nth(axis)
                .ok_or("retained view axis is absent")?;
            expression_capacity(base, parent, function, visiting)
        }
        _ => Err(FormationError::Unresolved(
            "retained runtime shape has no structural capacity proof".into(),
        )),
    }
}

struct Scope<'a> {
    function: &'a LoweredIr,
    numeric: &'a BTreeMap<String, (i64, i64)>,
    selectors: &'a BTreeSet<VarId>,
    inputs: BTreeSet<VarId>,
    symbols: BTreeSet<String>,
}
impl Scope<'_> {
    fn symbol_bound(&self, symbol: &str, bound: &BTreeSet<VarId>) -> bool {
        self.function.shapes.contains_key(symbol)
            || self.numeric.contains_key(symbol)
            || self.function.index_params.iter().any(|(name, _)| name == symbol)
            || bound.iter().chain(&self.inputs).any(|&v| {
                self.function.vars.get(v).is_some_and(|var| {
                    matches!(&var.kind, VarKind::Index(seismic_lang::sym::Atom::Param(name)) if name == symbol)
                        || var.ty.shaped().is_some_and(|shape| shape.shape.iter().any(|extent| extent.params().iter().any(|name| name == symbol)))
                })
            })
    }
    fn use_symbols(&mut self, value: &Sym, bound: &BTreeSet<VarId>) {
        for name in value.params() {
            if !self.symbol_bound(&name, bound) {
                self.symbols.insert(name);
            }
        }
    }
    fn unbound_symbol<'a>(&'a self, bound: &BTreeSet<VarId>) -> Option<&'a str> {
        self.symbols
            .iter()
            .find(|symbol| !self.symbol_bound(symbol, bound))
            .map(String::as_str)
    }
    fn variable(&self, id: VarId) -> Result<&Var, String> {
        self.function
            .vars
            .get(id)
            .ok_or_else(|| format!("invalid variable identity {id}"))
    }
    fn indices(&self, vars: &[VarId], bound: &mut BTreeSet<VarId>) -> Result<(), String> {
        for &v in vars {
            let var = self.variable(v)?;
            if !matches!(var.kind, VarKind::Index(_)) || var.ty != Ty::Scalar(DType::I32) {
                return Err("execution index requires a typed i32 index binding".into());
            }
            bound.insert(v);
        }
        Ok(())
    }
    fn expr(&mut self, expr: &Expr, bound: &BTreeSet<VarId>) -> Result<(), String> {
        match &expr.kind {
            ExprKind::ShapeParam(symbol) => {
                if let Some(value) = &expr.sym {
                    self.use_symbols(value, bound);
                } else if !self.symbol_bound(symbol, bound) {
                    self.symbols.insert(symbol.clone());
                }
            }
            ExprKind::Var(v) => {
                let variable = self.variable(*v)?;
                if variable.ty != expr.ty {
                    return Err(format!(
                        "variable `{}` reference type differs from its binding",
                        variable.name
                    ));
                }
                if !bound.contains(v) {
                    self.inputs.insert(*v);
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base, bound)?;
                for index in indices {
                    match index {
                        Index::Point(point) => self.expr(point, bound)?,
                        Index::Slice { start, end } => {
                            for value in start.iter().chain(end) {
                                self.expr(value, bound)?;
                            }
                        }
                    }
                }
            }
            ExprKind::Load { view: expr, .. }
            | ExprKind::Transpose(expr)
            | ExprKind::Accessor { base: expr, .. }
            | ExprKind::Lanes { base: expr, .. }
            | ExprKind::Unary { expr, .. }
            | ExprKind::Cast { expr, .. } => self.expr(expr, bound)?,
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs, bound)?;
                self.expr(rhs, bound)?;
            }
            ExprKind::Builtin { args, .. }
            | ExprKind::Call { args, .. }
            | ExprKind::Intrinsic { args, .. }
            | ExprKind::Tuple(args) => {
                for argument in args {
                    self.expr(argument, bound)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn body(&mut self, body: &[Stmt], bound: &mut BTreeSet<VarId>) -> Result<(), String> {
        for statement in body {
            match &statement.kind {
                StmtKind::Assign { target, op, value } => {
                    self.expr(value, bound)?;
                    if let ExprKind::Var(v) = target.kind {
                        let variable = self.variable(v)?;
                        if variable.ty != target.ty {
                            return Err("assignment target type differs from binding".into());
                        }
                        if *op != AssignOp::Assign {
                            self.expr(target, bound)?;
                        }
                        bound.insert(v);
                    } else {
                        self.expr(target, bound)?;
                    }
                }
                StmtKind::Expr(expr) => self.expr(expr, bound)?,
                StmtKind::If { cond, then, els } => {
                    if cond.ty != Ty::Scalar(DType::Bool) {
                        return Err("execution condition must be boolean".into());
                    }
                    self.expr(cond, bound)?;
                    let mut yes = bound.clone();
                    let mut no = bound.clone();
                    self.body(then, &mut yes)?;
                    self.body(els, &mut no)?;
                    if matches!(cond.kind, ExprKind::Var(variable) if self.selectors.contains(&variable)) {
                        bound.extend(yes);
                        bound.extend(no);
                    }
                }
                StmtKind::Parallel {
                    vars,
                    extents,
                    body,
                } => {
                    if vars.len() != extents.len() {
                        return Err("parallel index/extent arity differs".into());
                    }
                    let mut inner = bound.clone();
                    self.indices(vars, &mut inner)?;
                    self.body(body, &mut inner)?;
                }
                StmtKind::Owned { vars, tile, body } => {
                    self.expr(tile, bound)?;
                    let mut inner = bound.clone();
                    self.indices(vars, &mut inner)?;
                    self.body(body, &mut inner)?;
                }
                StmtKind::Range { var, lo, hi, body } => {
                    self.use_symbols(lo, bound);
                    self.use_symbols(hi, bound);
                    let mut inner = bound.clone();
                    self.indices(&[*var], &mut inner)?;
                    self.body(body, &mut inner)?;
                    if let VarKind::Index(seismic_lang::sym::Atom::Param(name)) =
                        self.variable(*var)?.kind.clone()
                    {
                        self.symbols.remove(&name);
                    }
                }
                StmtKind::Lanes {
                    var, extent, body, ..
                } => {
                    self.use_symbols(extent, bound);
                    let mut inner = bound.clone();
                    self.indices(&[*var], &mut inner)?;
                    self.body(body, &mut inner)?;
                    if let VarKind::Index(seismic_lang::sym::Atom::Param(name)) =
                        self.variable(*var)?.kind.clone()
                    {
                        self.symbols.remove(&name);
                    }
                }
                StmtKind::LoadLoop {
                    domain,
                    offset,
                    vars,
                    views,
                    axes,
                    body,
                    ..
                } => {
                    if vars.len() != views.len() || vars.len() != axes.len() {
                        return Err("stream binding arity differs".into());
                    }
                    self.expr(&domain.view, bound)?;
                    for view in views {
                        self.expr(view, bound)?;
                    }
                    let mut inner = bound.clone();
                    inner.extend(vars);
                    inner.extend(offset);
                    self.body(body, &mut inner)?;
                }
                StmtKind::Reduction(reduction) => {
                    for operand in reduction.operands() {
                        self.expr(operand, bound)?;
                    }
                    for implementation in reduction.implementations() {
                        let mut inner = bound.clone();
                        for value in implementation
                            .left
                            .iter()
                            .chain(&implementation.right)
                            .chain(&implementation.output)
                        {
                            if let ExprKind::Var(v) = value.kind {
                                inner.insert(v);
                            }
                        }
                        self.body(&implementation.body, &mut inner)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lowered(source: &str) -> LoweredIr {
        let program = seismic_lang::program::compile(
            &[seismic_lang::program::SourceFile {
                path: "phases.seismic.portable".into(),
                scope: seismic_lang::Scope::Portable,
                text: source.into(),
            }],
            &[],
        )
        .unwrap_or_else(|errors| {
            panic!(
                "{}",
                errors
                    .iter()
                    .map(|e| e.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        });
        seismic_lang::lower::lower(&program, "evaluate", "cpu", &Default::default()).unwrap()
    }
    #[test]
    fn serial_values_are_published_once_and_serial_updates_publish_new_versions() {
        let function = lowered(
            "fn evaluate(x:tensor[8] f32,out:tensor[8] f32):\n  a = x[0]\n  snapshot = load(x)\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = snapshot[row] + a\n    store(y,out[row:row+1])\n  a += 1.0\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = a\n    store(y,out[row:row+1])\n",
        );
        let plan = construct(&function).unwrap();
        assert_eq!(plan.phases.len(), 4);
        assert_eq!(plan.retained.len(), 2);
        let scalar = plan.retained.iter().find(|v| v.elements == 1).unwrap();
        assert_eq!(scalar.producer, 0);
        assert_eq!(scalar.consumers, [1, 2, 3]);
        assert_eq!(scalar.writers, [0, 2]);
        let tile = plan.retained.iter().find(|v| v.elements == 8).unwrap();
        assert_eq!(tile.consumers, [1]);
        assert_eq!(plan.function.params.len(), function.params.len() + 2);
        verify(&plan.function).unwrap();
    }
    #[test]
    fn captured_view_retains_coordinate_reads_without_copying_its_referent() {
        let function = lowered(
            "fn evaluate(x:tensor[8] f32,bounds:tensor[2] i32,out:tensor[2] i32):\n  view = x[bounds[0]:bounds[1]]\n  for row in parallel:\n    y = tile[1] i32\n    for i in owned(y): y[i] = extent(view,0)\n    store(y,out[row:row+1])\n",
        );
        let plan = construct(&function).unwrap();
        assert_eq!(plan.retained.len(), 2);
        assert!(
            plan.retained
                .iter()
                .all(|v| v.dtype == DType::I32 && v.elements == 1)
        );
        let StmtKind::Parallel { body, .. } = &plan.function.body[1].kind else {
            unreachable!()
        };
        let bounds = function
            .vars
            .iter()
            .position(|v| v.name == "bounds")
            .unwrap();
        assert!(!body.iter().any(|s| seismic_lang::effects::uses(s, bounds)));
        verify(&plan.function).unwrap();
    }
    #[test]
    fn geometry_used_only_by_a_symbolic_range_is_a_phase_input() {
        let function = lowered(
            "fn evaluate(x:tensor[8] f32,bounds:tensor[2] i32,out:tensor[2] i32):\n  view = x[bounds[0]:bounds[1]]\n  for row in parallel:\n    count = 0\n    for j in range(extent(view,0)): count += 1\n    y = tile[1] i32\n    for i in owned(y): y[i] = count\n    store(y,out[row:row+1])\n",
        );
        let plan = construct(&function).unwrap();
        assert_eq!(plan.retained.len(), 2);
        let StmtKind::Parallel { body, .. } = &plan.function.body[1].kind else {
            unreachable!()
        };
        let view = function.vars.iter().position(|v| v.name == "view").unwrap();
        assert!(body.iter().any(|s| matches!(&s.kind, StmtKind::Assign { target: Expr { kind: ExprKind::Var(v), .. }, .. } if *v == view)));
        verify(&plan.function).unwrap();
    }
    #[test]
    fn runtime_tile_shape_has_retained_lengths_and_structural_capacity() {
        let function = lowered(
            "fn evaluate(x:tensor[8] f32,bounds:tensor[2] i32,out:tensor[2] f32):\n  snapshot = load(x[bounds[0]:bounds[1]])\n  for row in parallel:\n    result = tile[1] f32\n    for i in owned(result): result[i] = reduce(snapshot,0,sum)\n    store(result,out[row:row+1])\n",
        );
        let plan = construct(&function).unwrap();
        assert!(
            plan.retained
                .iter()
                .any(|v| v.dtype == DType::I32 && v.elements == 1)
        );
        assert!(
            plan.retained
                .iter()
                .any(|v| v.dtype == DType::F32 && v.elements == 8)
        );
        verify(&plan.function).unwrap();
    }
    #[test]
    fn phase_scope_rejects_an_iteration_value_escaping_to_the_next_launch() {
        let mut function = lowered(
            "fn evaluate(x:tensor[8] f32,out:tensor[8] f32):\n  for row in parallel:\n    value = load(x[row:row+1])\n    store(value,out[row:row+1])\n",
        );
        let StmtKind::Parallel { body, .. } = &function.body[0].kind else {
            unreachable!()
        };
        let StmtKind::Assign { target, .. } = &body[0].kind else {
            unreachable!()
        };
        function.body.push(Stmt {
            id: None,
            span: target.span,
            kind: StmtKind::Expr(target.clone()),
        });
        let error = construct(&function).unwrap_err();
        assert!(error.contains("outside its defining scope"), "{error}");
    }
}
