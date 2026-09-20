use super::*;
use crate::family::{Candidate, CandidateRef, OccurrenceId, Requirement, SiteId, SiteKind};
use crate::sir::{ExprKind as SExpr, StmtKind as SStmt};
use crate::sym::Atom;
use crate::types::SliceId;

pub(super) fn populate(
    program: &Program,
    family: &family::Family,
    logical: &mut LogicalProgram,
) -> Result<(), String> {
    logical.choices.clear();
    logical.task_graphs.clear();
    logical.extents.clear();
    logical.structural_constraints.clear();
    logical.structural_refinements.clear();
    logical.extent_bounds.clear();
    let mut candidate_symbols = BTreeMap::<CandidateRef, StructuralSymbols>::new();

    for occurrence in &family.occurrences {
        let choice = ChoiceId(occurrence.id.0);
        if choice.0 as usize != logical.choices.len() {
            return Err("family occurrences are not in stable pre-order".into());
        }
        let first = occurrence
            .candidates
            .first()
            .ok_or_else(|| format!("choice#{} has no applicable implementation", choice.0))?;
        let environments = occurrence
            .candidates
            .iter()
            .enumerate()
            .map(|(alternative, candidate)| {
                let owner = CandidateRef {
                    occurrence: occurrence.id,
                    candidate: alternative as u32,
                };
                let parent = occurrence
                    .parent
                    .map(|parent| {
                        candidate_symbols.get(&parent).ok_or_else(|| {
                            format!("choice#{} precedes its symbolic parent", occurrence.id.0)
                        })
                    })
                    .transpose()?;
                let caller_result = match (occurrence.parent, occurrence.call, parent) {
                    (Some(parent_ref), Some(call), Some(parent_symbols)) => {
                        let parent_candidate = family.candidate(parent_ref);
                        let parent_template = family.template(parent_candidate.template);
                        let parent_definition = program.definition(parent_template.definition);
                        let call = parent_definition
                            .body
                            .calls
                            .get(call.0 as usize)
                            .ok_or_else(|| {
                                "nested call is absent from its parent body".to_string()
                            })?;
                        Some(body_type(
                            &call.result,
                            &parent_template.shapes,
                            &parent_template.elems,
                            parent_symbols,
                        )?)
                    }
                    _ => None,
                };
                let template = family.template(candidate.template);
                let definition = program.definition(template.definition);
                let symbols = structural_symbols(
                    family,
                    owner,
                    candidate,
                    parent,
                    definition,
                    caller_result.as_ref(),
                )?;
                candidate_symbols.insert(owner, symbols.clone());
                Ok(symbols)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let first_template = family.template(first.template);
        let first_definition = program.definition(first_template.definition);
        let first_structural = &environments[0];
        let (interface, inputs, results) = branch_interface(
            first_definition,
            &first_template.shapes,
            &first_template.elems,
            &first_structural,
            if occurrence.id.0 == 0 {
                Some(logical)
            } else {
                None
            },
        )?;

        let mut alternatives = Vec::with_capacity(occurrence.candidates.len());
        for (alternative_ordinal, candidate) in occurrence.candidates.iter().enumerate() {
            let template = family.template(candidate.template);
            let definition = program.definition(template.definition);
            let structural = environments[alternative_ordinal].clone();
            let (candidate_interface, candidate_inputs, candidate_results) = branch_interface(
                definition,
                &template.shapes,
                &template.elems,
                &structural,
                if occurrence.id.0 == 0 {
                    Some(logical)
                } else {
                    None
                },
            )?;
            if candidate_interface != interface {
                return Err(format!(
                    "choice#{} implementation #{} changes its specialized logical interface",
                    choice.0, definition.id.0
                ));
            }
            let input_remap = input_remap(program, family, occurrence.id, candidate, definition)?;
            let task_graph_id = TaskGraphId(logical.task_graphs.len() as u32);
            for (symbol, bound) in runtime_extent_bounds(definition, &template.shapes, &structural)?
            {
                if let Some(previous) = logical.extent_bounds.insert(symbol.clone(), bound.clone())
                {
                    if previous != bound {
                        return Err(format!(
                            "runtime extent `{symbol}` has inconsistent capacity bounds"
                        ));
                    }
                }
            }
            let mut translator = Translator::new(
                family,
                CandidateRef {
                    occurrence: occurrence.id,
                    candidate: alternative_ordinal as u32,
                },
                definition,
                &template.shapes,
                &template.elems,
                structural,
                candidate_results.clone(),
            )?;
            let body = translator.block(&definition.body.block)?;
            let mut runtime_extents = std::mem::take(&mut translator.runtime_extents);
            for (port, input) in candidate_inputs.iter().enumerate() {
                let Type::Tensor(tensor) = &input.ty else {
                    continue;
                };
                for (axis, extent) in tensor.shape.iter().enumerate() {
                    for atom in extent.atoms() {
                        let Atom::Param(symbol) = atom else {
                            continue;
                        };
                        if !symbol.starts_with("@runtime.") || runtime_extents.contains_key(&symbol)
                        {
                            continue;
                        }
                        let Some((coefficient, rest)) =
                            extent.linear_in(&Atom::Param(symbol.clone()))
                        else {
                            continue;
                        };
                        let Some(rest) = rest.as_constant() else {
                            continue;
                        };
                        if coefficient <= 0 {
                            continue;
                        }
                        let scalar_ty = Type::Scalar(DType::I32);
                        let mut value = LogicalExpr {
                            ty: scalar_ty.clone(),
                            kind: LogicalExprKind::Extent {
                                base: Box::new(LogicalExpr {
                                    ty: input.ty.clone(),
                                    kind: LogicalExprKind::Value(ValueRef::Input(port as u32)),
                                    span: Span::default(),
                                }),
                                axis,
                            },
                            span: Span::default(),
                        };
                        if rest != 0 {
                            value = LogicalExpr {
                                ty: scalar_ty.clone(),
                                kind: LogicalExprKind::Binary {
                                    op: crate::syntax::ast::BinaryOp::Sub,
                                    lhs: Box::new(value),
                                    rhs: Box::new(LogicalExpr {
                                        ty: scalar_ty.clone(),
                                        kind: LogicalExprKind::Int(rest),
                                        span: Span::default(),
                                    }),
                                },
                                span: Span::default(),
                            };
                        }
                        if coefficient != 1 {
                            value = LogicalExpr {
                                ty: scalar_ty,
                                kind: LogicalExprKind::Binary {
                                    op: crate::syntax::ast::BinaryOp::Div,
                                    lhs: Box::new(value),
                                    rhs: Box::new(LogicalExpr {
                                        ty: Type::Scalar(DType::I32),
                                        kind: LogicalExprKind::Int(coefficient),
                                        span: Span::default(),
                                    }),
                                },
                                span: Span::default(),
                            };
                        }
                        runtime_extents.insert(symbol, value);
                    }
                }
            }
            let task_capabilities = definition
                .requires
                .iter()
                .map(|id| id.path())
                .collect::<BTreeSet<_>>();
            let normalized = normalize_body(
                body,
                &mut translator.value_types,
                &translator.views,
                translator.var_storage.clone(),
                &candidate_inputs,
                &candidate_results,
                &task_capabilities,
                &candidate.numerical_effects,
            )?;
            logical.task_graphs.push(LogicalTaskGraph {
                id: task_graph_id,
                choice,
                alternative: alternative_ordinal as u32,
                inputs: candidate_inputs,
                results: candidate_results,
                input_remap,
                result_remap: (0..results.len() as u32).collect(),
                values: translator.value_types,
                storage: translator.storage,
                views: translator.views,
                runtime_extents,
                operands: normalized.operands,
                tasks: normalized.tasks,
                calls: normalized.calls,
                dependencies: normalized.dependencies,
            });
            alternatives.push(Alternative {
                definition: definition.id.0,
                interface: interface.clone(),
                capabilities: task_capabilities,
                numerical_effects: candidate.numerical_effects.clone(),
                task_graph: task_graph_id,
            });

            for site in &candidate.sites {
                let site = family.sites.get(site.0 as usize).ok_or_else(|| {
                    format!("candidate references absent logical extent site#{}", site.0)
                })?;
                logical.extents.push(LogicalExtent::Structural {
                    choice,
                    alternative: alternative_ordinal as u32,
                    site: site.id.0,
                    upper_bound: site.extent,
                });
            }
            let active_if = LogicalAlternativeRef {
                choice,
                alternative: alternative_ordinal as u32,
            };
            for requirement in &candidate.requirements {
                logical.structural_constraints.push(StructuralConstraint {
                    active_if,
                    site: logical_site_ref(family, *requirement_site(requirement))?,
                    kind: match requirement {
                        Requirement::Multiple { unit, .. } => {
                            StructuralConstraintKind::Multiple(*unit)
                        }
                        Requirement::AtLeast { value, .. } => {
                            StructuralConstraintKind::AtLeast(*value)
                        }
                        Requirement::AtMost { value, .. } => {
                            StructuralConstraintKind::AtMost(*value)
                        }
                        Requirement::Equal { value, .. } => StructuralConstraintKind::Equal(*value),
                        Requirement::Divides { extent, .. } => {
                            StructuralConstraintKind::Divides(*extent)
                        }
                    },
                });
            }
        }
        logical.choices.push(Choice {
            occurrence: occurrence.id.0,
            interface,
            inputs,
            results,
            alternatives,
        });
    }
    logical.structural_refinements = family
        .refinements
        .iter()
        .map(|(refinement, refined)| {
            Ok(StructuralRefinement {
                refinement: logical_site_ref(family, *refinement)?,
                refined: logical_site_ref(family, *refined)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    logical.entry_choice = ChoiceId(0);
    for (ordinal, value) in logical.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::RangeParameter { .. }) {
            logical
                .extents
                .push(LogicalExtent::Dynamic(ValueId(ordinal as u32)));
        }
    }
    let interface_types = logical
        .choices
        .iter()
        .flat_map(|choice| {
            choice
                .interface
                .inputs
                .iter()
                .chain(choice.interface.results.iter())
        })
        .cloned()
        .collect::<Vec<_>>();
    for ty in &interface_types {
        record_type_extents(ty, &mut logical.extents);
    }
    Ok(())
}

fn requirement_site(requirement: &Requirement) -> &SiteId {
    match requirement {
        Requirement::Multiple { site, .. }
        | Requirement::AtLeast { site, .. }
        | Requirement::AtMost { site, .. }
        | Requirement::Equal { site, .. }
        | Requirement::Divides { site, .. } => site,
    }
}

fn logical_site_ref(family: &family::Family, site: SiteId) -> Result<StructuralSiteRef, String> {
    let site = family
        .sites
        .get(site.0 as usize)
        .ok_or_else(|| format!("logical constraint references absent site#{}", site.0))?;
    Ok(StructuralSiteRef {
        choice: ChoiceId(site.owner.occurrence.0),
        alternative: site.owner.candidate,
        site: site.id.0,
    })
}

#[derive(Clone, Debug, Default)]
struct StructuralSymbols {
    params: BTreeMap<String, Sym>,
    slices: BTreeMap<SliceId, Sym>,
    runtime: BTreeMap<String, Sym>,
    runtime_prefix: String,
}

impl StructuralSymbols {
    fn substitute(&self, sym: &Sym, shapes: &BTreeMap<String, i64>) -> Sym {
        substitute_sym(sym, &|name| {
            shapes
                .get(name)
                .copied()
                .map(Sym::constant)
                .or_else(|| self.params.get(name).cloned())
                .or_else(|| self.runtime.get(name).cloned())
                // Checker variable atoms are unique only within one authored
                // definition (`name#VarId`). A definition may occur more than
                // once in one logical program, so retaining that spelling
                // would alias unrelated loop indices and scalar temporaries.
                // The candidate occurrence is their logical owner.
                .or_else(|| {
                    crate::check::atom_var(name)
                        .map(|var| Sym::param(&format!("{}.var{var}", self.runtime_prefix)))
                })
                .or_else(|| {
                    name.strip_prefix("@dyn#")
                        .map(|suffix| Sym::param(&format!("{}.{}", self.runtime_prefix, suffix)))
                })
        })
    }
}

fn substitute_sym(sym: &Sym, env: &dyn Fn(&str) -> Option<Sym>) -> Sym {
    let mut out = Sym::constant(0);
    for (monomial, coefficient) in sym.monomials() {
        let mut term = Sym::constant(coefficient);
        for (atom, power) in monomial {
            let factor = match atom {
                Atom::Param(name) => env(name).unwrap_or_else(|| Sym::atom(atom.clone())),
                Atom::Quot(numerator, denominator) => {
                    substitute_sym(numerator, env).quot(&substitute_sym(denominator, env))
                }
                Atom::Rem(numerator, denominator) => {
                    substitute_sym(numerator, env).rem(&substitute_sym(denominator, env))
                }
            };
            for _ in 0..*power {
                term = term.mul(&factor);
            }
        }
        out = out.add(&term);
    }
    out
}

/// Give every structural extent one occurrence-wide identity. A callee shape
/// parameter and the caller slice bound to it must name the same symbol; raw
/// source parameter names (`K`, `N`, ...) are definition-local and therefore
/// cannot be used as logical identities.
fn structural_symbols(
    family: &family::Family,
    owner: CandidateRef,
    candidate: &Candidate,
    parent: Option<&StructuralSymbols>,
    definition: &sir::Definition,
    caller_result: Option<&Type>,
) -> Result<StructuralSymbols, String> {
    let mut symbols = StructuralSymbols {
        params: BTreeMap::new(),
        slices: BTreeMap::new(),
        runtime: BTreeMap::new(),
        runtime_prefix: format!(
            "@runtime.choice{}.alternative{}",
            owner.occurrence.0, owner.candidate
        ),
    };
    for (name, site) in &candidate.structural {
        symbols
            .params
            .insert(name.clone(), Sym::param(&format!("@site{}", site.0 .0)));
    }
    for (name, extent) in &candidate.dynamic {
        let extent = parent
            .map(|parent| parent.substitute(extent, &BTreeMap::new()))
            .unwrap_or_else(|| extent.clone());
        symbols.params.insert(name.clone(), extent);
    }
    for site in &candidate.sites {
        let site = family
            .sites
            .get(site.0 as usize)
            .ok_or_else(|| format!("candidate references absent structural site#{}", site.0))?;
        let slice = match site.kind {
            SiteKind::Width { slice, .. } | SiteKind::Parts { slice, .. } => slice,
        };
        symbols
            .slices
            .insert(slice, Sym::param(&format!("@site{}", site.id.0)));
    }
    if let Some(caller_result) = caller_result {
        bind_result_symbols(
            &definition.result,
            caller_result,
            &mut symbols.params,
            &mut symbols.runtime,
        )?;
    }
    Ok(symbols)
}

/// A dynamic result extent is dependent call-interface data.  The checker
/// creates a definition-local atom for the callee body and another at the call
/// expression; specializing the occurrence makes that equality explicit by
/// substituting the callee atom with the caller's logical identity.
fn bind_result_symbols(
    authored: &Ty,
    caller: &Type,
    params: &mut BTreeMap<String, Sym>,
    runtime: &mut BTreeMap<String, Sym>,
) -> Result<(), String> {
    match (authored, caller) {
        (Ty::Tensor(value) | Ty::View(value) | Ty::Tile(value), Type::Tensor(caller)) => {
            if value.axes.len() != caller.shape.len() {
                return Err("dynamic call result changes rank".into());
            }
            for (axis, caller) in value.axes.iter().zip(&caller.shape) {
                let Extent::Semantic(authored) = axis else {
                    continue;
                };
                if let [Atom::Param(name)] = authored.atoms().as_slice() {
                    if authored != &Sym::param(name) {
                        continue;
                    }
                    let previous = if name.starts_with("@dyn#") {
                        runtime.insert(name.clone(), caller.clone())
                    } else if params.contains_key(name) {
                        params.insert(name.clone(), caller.clone())
                    } else {
                        None
                    };
                    if let Some(previous) = previous {
                        if previous != *caller {
                            // The call result is the authoritative dependent
                            // instantiation of a runtime-valued result parameter.
                            // Static/structural disagreements remain ordinary
                            // interface mismatches below.
                            if previous.as_constant().is_some()
                                || caller.as_constant().is_some()
                                || name.starts_with("@site")
                            {
                                return Err(format!(
                                    "result symbol `{name}` disagrees with its caller extent"
                                ));
                            }
                        }
                    }
                }
            }
        }
        (Ty::Tuple(authored), Type::Tuple(caller)) if authored.len() == caller.len() => {
            for (authored, caller) in authored.iter().zip(caller) {
                bind_result_symbols(authored, caller, params, runtime)?;
            }
        }
        (Ty::Result(result), caller) => {
            bind_result_symbols(&result.member, caller, params, runtime)?
        }
        _ => {}
    }
    Ok(())
}

fn each_expression_in<'a>(expression: &'a sir::Expr, visit: &mut dyn FnMut(&'a sir::Expr)) {
    use sir::{ExprKind, Index, RegionSource};
    fn region<'a>(region: &'a sir::Region, visit: &mut dyn FnMut(&'a sir::Expr)) {
        if let RegionSource::Results(value) = &region.source {
            each_expression_in(value, visit);
        }
        each_expression(&region.body, visit);
        if let Some(merge) = &region.merge {
            each_expression_in(&merge.identity, visit);
            each_expression(&merge.body, visit);
        }
    }
    visit(expression);
    match &expression.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Var(_)
        | ExprKind::ShapeParam(_)
        | ExprKind::TileAlloc
        | ExprKind::CoordOf(_) => {}
        ExprKind::Tuple(values)
        | ExprKind::Math { args: values, .. }
        | ExprKind::Call { args: values, .. }
        | ExprKind::Intrinsic { args: values, .. } => {
            for value in values {
                each_expression_in(value, visit);
            }
        }
        ExprKind::Range { lo, hi } => {
            each_expression_in(lo, visit);
            each_expression_in(hi, visit);
        }
        ExprKind::Field { base, .. }
        | ExprKind::Filled { like: base, .. }
        | ExprKind::Member { result: base, .. }
        | ExprKind::Transpose(base)
        | ExprKind::Reshape { base, .. }
        | ExprKind::Load(base)
        | ExprKind::Decode(base)
        | ExprKind::Cast { expr: base, .. }
        | ExprKind::Unary { expr: base, .. }
        | ExprKind::Reduce { value: base, .. }
        | ExprKind::ExtentOf { base, .. }
        | ExprKind::Accessor { base, .. }
        | ExprKind::Geometry { base, .. } => each_expression_in(base, visit),
        ExprKind::Index { base, indices } => {
            each_expression_in(base, visit);
            for index in indices {
                match index {
                    Index::Point(value) => each_expression_in(value, visit),
                    Index::Range { start, end } => {
                        for value in start.iter().chain(end) {
                            each_expression_in(value, visit);
                        }
                    }
                    Index::Coord(_) | Index::Slice(_) => {}
                }
            }
        }
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Atomic {
            place: lhs,
            value: rhs,
            ..
        } => {
            each_expression_in(lhs, visit);
            each_expression_in(rhs, visit);
        }
        ExprKind::Select { cond, then, els } => {
            each_expression_in(cond, visit);
            each_expression_in(then, visit);
            each_expression_in(els, visit);
        }
        ExprKind::Region(value) => region(value, visit),
    }
}

fn each_expression<'a>(block: &'a [sir::Stmt], visit: &mut dyn FnMut(&'a sir::Expr)) {
    use sir::StmtKind;
    for statement in block {
        match &statement.kind {
            StmtKind::Bind { value, .. } => each_expression_in(value, visit),
            StmtKind::Assign { target, value, .. } => {
                each_expression_in(target, visit);
                each_expression_in(value, visit);
            }
            StmtKind::Region(region) => {
                if let sir::RegionSource::Results(value) = &region.source {
                    each_expression_in(value, visit);
                }
                each_expression(&region.body, visit);
                if let Some(merge) = &region.merge {
                    each_expression_in(&merge.identity, visit);
                    each_expression(&merge.body, visit);
                }
            }
            StmtKind::Stages(stages) => {
                for stage in stages {
                    each_expression(&stage.body, visit);
                }
            }
            StmtKind::Range { lo, hi, body, .. } => {
                each_expression_in(lo, visit);
                each_expression_in(hi, visit);
                each_expression(body, visit);
            }
            StmtKind::Coordinates { of, body, .. } => {
                each_expression_in(of, visit);
                each_expression(body, visit);
            }
            StmtKind::Members { body, .. } => each_expression(body, visit),
            StmtKind::If { cond, then, els } => {
                each_expression_in(cond, visit);
                each_expression(then, visit);
                each_expression(els, visit);
            }
            StmtKind::Publish { value, destination } => {
                each_expression_in(value, visit);
                each_expression_in(destination, visit);
            }
            StmtKind::Yield(values) | StmtKind::Return(values) => {
                for value in values {
                    each_expression_in(value, visit);
                }
            }
            StmtKind::Expr(value) => each_expression_in(value, visit),
        }
    }
}

fn runtime_extent_bounds(
    definition: &sir::Definition,
    shapes: &BTreeMap<String, i64>,
    symbols: &StructuralSymbols,
) -> Result<BTreeMap<String, Sym>, String> {
    let mut bounds = BTreeMap::new();
    let mut failure = None;

    // A range binder is a runtime integer with a statically checked exclusive
    // upper bound. Shapes constructed inside the loop may depend affinely on
    // that binder (for example an attention prefix `[heads, row + 1]`). Record
    // the binder's inclusive capacity here so physical layout can bound those
    // shapes without replacing the runtime value.
    fn range_bounds(
        definition: &sir::Definition,
        block: &[sir::Stmt],
        shapes: &BTreeMap<String, i64>,
        symbols: &StructuralSymbols,
        bounds: &mut BTreeMap<String, Sym>,
    ) -> Result<(), String> {
        use sir::StmtKind;
        for statement in block {
            match &statement.kind {
                StmtKind::Range { var, hi, body, .. } => {
                    if let Some(hi) = &hi.sym {
                        let variable = definition
                            .body
                            .vars
                            .get(*var)
                            .ok_or("range binder names an absent body variable")?;
                        let local = Sym::atom(crate::check::var_atom(&variable.name, *var));
                        let canonical = symbols.substitute(&local, shapes);
                        let atoms = canonical.atoms();
                        let [Atom::Param(symbol)] = atoms.as_slice() else {
                            return Err("range binder has no canonical runtime identity".into());
                        };
                        if canonical != Sym::param(symbol) {
                            return Err("range binder runtime identity is not one symbol".into());
                        }
                        let bound = symbols.substitute(hi, shapes).sub(&Sym::constant(1));
                        if let Some(previous) = bounds.insert(symbol.clone(), bound.clone()) {
                            if previous != bound {
                                return Err(format!(
                                    "range binder `{symbol}` has inconsistent capacity bounds"
                                ));
                            }
                        }
                    }
                    range_bounds(definition, body, shapes, symbols, bounds)?;
                }
                StmtKind::Region(region) => {
                    range_bounds(definition, &region.body, shapes, symbols, bounds)?;
                    if let Some(merge) = &region.merge {
                        range_bounds(definition, &merge.body, shapes, symbols, bounds)?;
                    }
                }
                StmtKind::Stages(stages) => {
                    for stage in stages {
                        range_bounds(definition, &stage.body, shapes, symbols, bounds)?;
                    }
                }
                StmtKind::Coordinates { body, .. } | StmtKind::Members { body, .. } => {
                    range_bounds(definition, body, shapes, symbols, bounds)?;
                }
                StmtKind::If { then, els, .. } => {
                    range_bounds(definition, then, shapes, symbols, bounds)?;
                    range_bounds(definition, els, shapes, symbols, bounds)?;
                }
                StmtKind::Bind { .. }
                | StmtKind::Assign { .. }
                | StmtKind::Publish { .. }
                | StmtKind::Yield(_)
                | StmtKind::Return(_)
                | StmtKind::Expr(_) => {}
            }
        }
        Ok(())
    }

    range_bounds(
        definition,
        &definition.body.block,
        shapes,
        symbols,
        &mut bounds,
    )?;
    each_expression(&definition.body.block, &mut |expression| {
        if failure.is_some() {
            return;
        }
        let SExpr::Index { base, indices } = &expression.kind else {
            return;
        };
        let (Some(source), Some(result)) = (base.ty.shaped(), expression.ty.shaped()) else {
            return;
        };
        let kept = (0..source.axes.len())
            .filter(|axis| {
                !matches!(
                    indices.get(*axis),
                    Some(sir::Index::Point(_) | sir::Index::Coord(_))
                )
            })
            .collect::<Vec<_>>();
        if kept.len() != result.axes.len() {
            failure = Some("runtime range view changes rank inconsistently".to_string());
            return;
        }
        for (result_axis, source_axis) in kept.into_iter().enumerate() {
            if !matches!(indices.get(source_axis), Some(sir::Index::Range { .. })) {
                continue;
            }
            let Extent::Semantic(runtime) = &result.axes[result_axis] else {
                continue;
            };
            let runtime_atoms = runtime.atoms();
            let [Atom::Param(local)] = runtime_atoms.as_slice() else {
                continue;
            };
            if runtime != &Sym::param(local) || !local.starts_with("@dyn#") {
                continue;
            }
            let runtime = symbols.substitute(runtime, shapes);
            let canonical_atoms = runtime.atoms();
            let [Atom::Param(canonical)] = canonical_atoms.as_slice() else {
                failure = Some("runtime extent identity is not one canonical symbol".to_string());
                return;
            };
            let bound = match &source.axes[source_axis] {
                Extent::Semantic(bound) => symbols.substitute(bound, shapes),
                Extent::Structural(slice) => match symbols.slices.get(slice) {
                    Some(bound) => bound.clone(),
                    None => {
                        failure = Some(format!(
                            "runtime extent source slice#{} has no structural identity",
                            slice.0
                        ));
                        return;
                    }
                },
            };
            if let Some(previous) = bounds.insert(canonical.clone(), bound.clone()) {
                if previous != bound {
                    failure = Some(format!(
                        "runtime extent `{canonical}` has two source-axis bounds"
                    ));
                    return;
                }
            }
        }
    });
    failure.map_or(Ok(bounds), Err)
}

fn record_type_extents(ty: &Type, out: &mut Vec<LogicalExtent>) {
    match ty {
        Type::Tensor(tensor) => {
            for extent in &tensor.shape {
                let value = match extent.as_constant() {
                    Some(value) => LogicalExtent::Static(value),
                    None => LogicalExtent::Symbolic(extent.clone()),
                };
                if !out.contains(&value) {
                    out.push(value);
                }
            }
        }
        Type::Tuple(items) => {
            for item in items {
                record_type_extents(item, out);
            }
        }
        _ => {}
    }
}

fn branch_interface(
    definition: &sir::Definition,
    shapes: &BTreeMap<String, i64>,
    elems: &BTreeMap<String, Elem>,
    structural: &StructuralSymbols,
    root: Option<&LogicalProgram>,
) -> Result<(Interface, Vec<Port>, Vec<Port>), String> {
    let inputs = definition
        .params
        .iter()
        .map(|param| body_type(&param.ty, shapes, elems, structural))
        .collect::<Result<Vec<_>, _>>()?;
    let result = body_type(&definition.result, shapes, elems, structural)?;
    let results = if result == Type::Void {
        vec![]
    } else {
        vec![result]
    };
    let mut input_ports = Vec::new();
    let mut port_effects = Vec::new();
    for (ordinal, (param, ty)) in definition.params.iter().zip(&inputs).enumerate() {
        input_ports.push(Port {
            path: vec![ordinal as u32],
            ty: ty.clone(),
            access: shaped_access(ty, param.mode),
        });
        add_port_effects(
            ty,
            ordinal as u32,
            &mut Vec::new(),
            param.mode,
            &mut port_effects,
        );
    }
    let mut result_ports = Vec::new();
    for ty in &results {
        flatten_ports(
            ty,
            &mut Vec::new(),
            true,
            &mut result_ports,
            &mut port_effects,
        );
    }
    let effects = root.map_or_else(Vec::new, |program| {
        let mut effects = Vec::new();
        for (ordinal, storage) in program.storage.iter().enumerate() {
            let id = StorageId(ordinal as u32);
            match storage.origin {
                StorageOrigin::Parameter { ordinal, .. } => {
                    match definition.params[ordinal as usize].mode {
                        Mode::In => effects.push(Effect::Read(id)),
                        Mode::Out => effects.push(Effect::Write(id)),
                        Mode::Inout => {
                            effects.push(Effect::Read(id));
                            effects.push(Effect::Write(id));
                        }
                    }
                }
                StorageOrigin::Result { .. } => {
                    effects.push(Effect::Write(id));
                    effects.push(Effect::Move(id));
                }
                StorageOrigin::Owned => {}
            }
        }
        effects
    });
    Ok((
        Interface {
            inputs,
            results,
            effects,
            port_effects,
        },
        input_ports,
        result_ports,
    ))
}

fn shaped_access(ty: &Type, mode: Mode) -> Option<Access> {
    matches!(ty, Type::Tensor(_) | Type::Tuple(_)).then_some(if mode == Mode::In {
        Access::Shared
    } else {
        Access::Exclusive
    })
}

fn add_port_effects(
    ty: &Type,
    port: u32,
    path: &mut Vec<u32>,
    mode: Mode,
    out: &mut Vec<PortEffect>,
) {
    match ty {
        Type::Tensor(_) => match mode {
            Mode::In => out.push(PortEffect::Read {
                port,
                path: path.clone(),
            }),
            Mode::Out => out.push(PortEffect::Write {
                port,
                path: path.clone(),
            }),
            Mode::Inout => {
                out.push(PortEffect::Read {
                    port,
                    path: path.clone(),
                });
                out.push(PortEffect::Write {
                    port,
                    path: path.clone(),
                });
            }
        },
        Type::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push(index as u32);
                add_port_effects(item, port, path, mode, out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn flatten_ports(
    ty: &Type,
    path: &mut Vec<u32>,
    result: bool,
    out: &mut Vec<Port>,
    effects: &mut Vec<PortEffect>,
) {
    match ty {
        Type::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push(index as u32);
                flatten_ports(item, path, result, out, effects);
                path.pop();
            }
        }
        Type::Void => {}
        _ => {
            let port = out.len() as u32;
            out.push(Port {
                path: path.clone(),
                ty: ty.clone(),
                access: matches!(ty, Type::Tensor(_)).then_some(Access::Exclusive),
            });
            if result && matches!(ty, Type::Tensor(_)) {
                effects.push(PortEffect::Write {
                    port,
                    path: path.clone(),
                });
                effects.push(PortEffect::MoveResult {
                    port,
                    path: path.clone(),
                });
            }
        }
    }
}

fn input_remap(
    program: &Program,
    family: &family::Family,
    occurrence: OccurrenceId,
    candidate: &family::Candidate,
    definition: &sir::Definition,
) -> Result<Vec<u32>, String> {
    if occurrence.0 == 0 {
        return Ok((0..definition.params.len() as u32).collect());
    }
    let call = family
        .occurrence(occurrence)
        .call
        .ok_or_else(|| format!("nested choice#{} has no call", occurrence.0))?;
    let parent = family
        .occurrence(occurrence)
        .parent
        .ok_or_else(|| format!("nested choice#{} has no parent", occurrence.0))?;
    let parent_template = family.template(family.candidate(parent).template);
    let site = program
        .definition(parent_template.definition)
        .body
        .calls
        .get(call.0 as usize)
        .ok_or_else(|| format!("call#{} is absent", call.0))?;
    site.bindings
        .iter()
        .find(|binding| binding.definition == candidate.via)
        .map(|binding| {
            binding
                .arg_order
                .iter()
                .map(|ordinal| *ordinal as u32)
                .collect()
        })
        .ok_or_else(|| {
            format!(
                "choice#{} candidate #{} has no argument remapping",
                occurrence.0, candidate.via.0
            )
        })
}

fn body_type(
    ty: &Ty,
    shapes: &BTreeMap<String, i64>,
    elems: &BTreeMap<String, Elem>,
    structural: &StructuralSymbols,
) -> Result<Type, String> {
    fn shaped(
        value: &Shaped,
        shapes: &BTreeMap<String, i64>,
        elems: &BTreeMap<String, Elem>,
        structural: &StructuralSymbols,
    ) -> Result<TensorType, String> {
        let shape = value
            .axes
            .iter()
            .map(|axis| -> Result<Sym, String> {
                match axis {
                    Extent::Semantic(sym) => Ok(structural.substitute(sym, shapes)),
                    Extent::Structural(slice) => {
                        structural.slices.get(slice).cloned().ok_or_else(|| {
                            format!("structural slice#{} has no logical site", slice.0)
                        })
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let elem = match &value.elem {
            Elem::Param(name) => elems
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unbound logical element `{name}`"))?,
            elem => elem.clone(),
        };
        Ok(TensorType { shape, elem })
    }
    Ok(match ty {
        Ty::Scalar(dtype) => Type::Scalar(*dtype),
        Ty::Index(bound) | Ty::Range(bound) => {
            let bound = structural.substitute(bound, shapes);
            if matches!(ty, Ty::Range(_)) {
                Type::Range { bound }
            } else {
                Type::Index { bound }
            }
        }
        Ty::Tensor(value) | Ty::View(value) | Ty::Tile(value) => {
            Type::Tensor(shaped(value, shapes, elems, structural)?)
        }
        Ty::Slice(slice) | Ty::Coord(slice) => Type::Index {
            bound: structural
                .slices
                .get(slice)
                .cloned()
                .ok_or_else(|| format!("structural slice#{} has no logical site", slice.0))?,
        },
        Ty::Tuple(items) => Type::Tuple(
            items
                .iter()
                .map(|item| body_type(item, shapes, elems, structural))
                .collect::<Result<_, _>>()?,
        ),
        Ty::Void => Type::Void,
        Ty::Native(native) => Type::CapabilityValue {
            target: native.target.clone(),
            name: native.name.clone(),
            shape: native.shape.clone(),
            elem: native.elem.clone(),
        },
        Ty::Result(result) => body_type(&result.member, shapes, elems, structural)?,
    })
}

fn has_expression_boundary(operation: &LogicalOperation) -> bool {
    fn index(index: &LogicalIndex) -> bool {
        match index {
            LogicalIndex::Point(value) => expr(value),
            LogicalIndex::Range { start, end } => start.iter().chain(end).any(|value| expr(value)),
            _ => false,
        }
    }
    fn expr(value: &LogicalExpr) -> bool {
        match &value.kind {
            LogicalExprKind::Call { .. }
            | LogicalExprKind::Reduce { .. }
            | LogicalExprKind::Region(_) => true,
            LogicalExprKind::Tuple(values)
            | LogicalExprKind::Math { args: values, .. }
            | LogicalExprKind::Intrinsic { args: values, .. } => values.iter().any(expr),
            LogicalExprKind::Range(a, b)
            | LogicalExprKind::Binary { lhs: a, rhs: b, .. }
            | LogicalExprKind::Atomic {
                place: a, value: b, ..
            } => expr(a) || expr(b),
            LogicalExprKind::Field(value, _)
            | LogicalExprKind::View { base: value, .. }
            | LogicalExprKind::Snapshot { source: value, .. }
            | LogicalExprKind::Decode { source: value, .. }
            | LogicalExprKind::Materialize { value, .. }
            | LogicalExprKind::Cast { expr: value, .. }
            | LogicalExprKind::Unary { expr: value, .. }
            | LogicalExprKind::Extent { base: value, .. }
            | LogicalExprKind::Accessor { base: value, .. }
            | LogicalExprKind::Geometry { base: value, .. }
            | LogicalExprKind::Filled { like: value, .. } => expr(value),
            LogicalExprKind::Index { base, indices } => expr(base) || indices.iter().any(index),
            LogicalExprKind::Select { cond, then, els } => expr(cond) || expr(then) || expr(els),
            _ => false,
        }
    }
    match &operation.kind {
        LogicalOperationKind::Bind { value, .. }
        | LogicalOperationKind::Expr(value)
        | LogicalOperationKind::Reduction { value, .. } => expr(value),
        LogicalOperationKind::ConditionalMerge { .. } => false,
        LogicalOperationKind::Assign { target, value, .. }
        | LogicalOperationKind::Publish {
            value,
            destination: target,
        } => expr(target) || expr(value),
        LogicalOperationKind::If {
            condition,
            then,
            els,
        } => expr(condition) || has_block_boundary(then) || has_block_boundary(els),
        LogicalOperationKind::Yield(values) => values.iter().any(expr),
        LogicalOperationKind::Return(writes) => writes.iter().any(|write| expr(&write.value)),
        LogicalOperationKind::Region(_)
        | LogicalOperationKind::Stages(_)
        | LogicalOperationKind::For { .. }
        | LogicalOperationKind::Coordinates { .. }
        | LogicalOperationKind::Members { .. } => true,
    }
}

fn has_block_boundary(body: &LogicalBlock) -> bool {
    body.iter().any(has_expression_boundary)
}

fn flatten_operand_types(ty: &Type, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, Type)>) {
    match ty {
        Type::Tuple(items) => {
            for (ordinal, item) in items.iter().enumerate() {
                path.push(ordinal as u32);
                flatten_operand_types(item, path, out);
                path.pop();
            }
        }
        Type::Void => {}
        _ => out.push((path.clone(), ty.clone())),
    }
}

fn collect_operation_values(operation: &LogicalOperation, out: &mut BTreeSet<ValueRef>) {
    match &operation.kind {
        LogicalOperationKind::Bind { value, .. }
        | LogicalOperationKind::Expr(value)
        | LogicalOperationKind::Reduction { value, .. } => collect_expr_values(value, out),
        LogicalOperationKind::Assign { target, value, .. }
        | LogicalOperationKind::Publish {
            value,
            destination: target,
        } => {
            collect_expr_values(target, out);
            collect_expr_values(value, out);
        }
        LogicalOperationKind::If {
            condition,
            then,
            els,
        } => {
            collect_expr_values(condition, out);
            for operation in then.iter().chain(els) {
                collect_operation_values(operation, out);
            }
        }
        LogicalOperationKind::Yield(values) => {
            for value in values {
                collect_expr_values(value, out);
            }
        }
        LogicalOperationKind::Return(writes) => {
            for write in writes {
                collect_expr_values(&write.value, out);
            }
        }
        LogicalOperationKind::ConditionalMerge { cases, .. } => {
            for case in cases {
                for predicate in &case.predicates {
                    collect_expr_values(&predicate.condition, out);
                }
            }
        }
        LogicalOperationKind::Region(_)
        | LogicalOperationKind::Stages(_)
        | LogicalOperationKind::For { .. }
        | LogicalOperationKind::Coordinates { .. }
        | LogicalOperationKind::Members { .. } => {}
    }
}

fn collect_expr_values(expression: &LogicalExpr, out: &mut BTreeSet<ValueRef>) {
    let mut collect = |value: &LogicalExpr| collect_expr_values(value, out);
    match &expression.kind {
        LogicalExprKind::Value(value) => {
            out.insert(value.clone());
        }
        LogicalExprKind::Coordinate(value) => {
            out.insert(ValueRef::Local(*value));
        }
        LogicalExprKind::Tuple(values)
        | LogicalExprKind::Math { args: values, .. }
        | LogicalExprKind::Intrinsic { args: values, .. }
        | LogicalExprKind::Call { args: values, .. } => {
            for value in values {
                collect(value);
            }
        }
        LogicalExprKind::Range(a, b)
        | LogicalExprKind::Binary { lhs: a, rhs: b, .. }
        | LogicalExprKind::Atomic {
            place: a, value: b, ..
        } => {
            collect(a);
            collect(b);
        }
        LogicalExprKind::Field(value, _)
        | LogicalExprKind::View { base: value, .. }
        | LogicalExprKind::Snapshot { source: value, .. }
        | LogicalExprKind::Decode { source: value, .. }
        | LogicalExprKind::Materialize { value, .. }
        | LogicalExprKind::Cast { expr: value, .. }
        | LogicalExprKind::Unary { expr: value, .. }
        | LogicalExprKind::Reduce { value, .. }
        | LogicalExprKind::Extent { base: value, .. }
        | LogicalExprKind::Accessor { base: value, .. }
        | LogicalExprKind::Geometry { base: value, .. }
        | LogicalExprKind::Filled { like: value, .. } => collect(value),
        LogicalExprKind::Index { base, indices } => {
            collect(base);
            for index in indices {
                match index {
                    LogicalIndex::Point(value) => collect_expr_values(value, out),
                    LogicalIndex::Coordinate(value) => {
                        out.insert(*value);
                    }
                    LogicalIndex::Range { start, end } => {
                        for value in start.iter().chain(end) {
                            collect_expr_values(value, out);
                        }
                    }
                    LogicalIndex::Slice(_) => {}
                }
            }
        }
        LogicalExprKind::Select { cond, then, els } => {
            collect(cond);
            collect(then);
            collect(els);
        }
        LogicalExprKind::Region(region) => {
            for operation in &region.body {
                collect_operation_values(operation, out);
            }
        }
        _ => {}
    }
}

struct NormalizedGraph {
    operands: Vec<LogicalOperand>,
    tasks: Vec<LogicalTask>,
    calls: Vec<LogicalCallBoundary>,
    dependencies: Vec<LogicalDependency>,
}

fn normalize_body(
    body: LogicalBlock,
    values: &mut Vec<Type>,
    views: &[LocalView],
    value_storage: Vec<Option<StorageRef>>,
    inputs: &[Port],
    results: &[Port],
    capabilities: &BTreeSet<String>,
    numerical_semantics: &[NumericalEffect],
) -> Result<NormalizedGraph, String> {
    let mut normalizer = GraphNormalizer {
        values,
        views,
        value_storage,
        inputs,
        results,
        capabilities,
        numerical_semantics: numerical_semantics.to_vec(),
        operands: Vec::new(),
        producers: Vec::new(),
        tasks: Vec::new(),
        calls: Vec::new(),
        dependencies: Vec::new(),
        output_cases: BTreeMap::new(),
        last: None,
    };
    normalizer.block(body, LogicalDomain::default())?;
    normalizer.finish_outputs()?;
    Ok(NormalizedGraph {
        operands: normalizer.operands,
        tasks: normalizer.tasks,
        calls: normalizer.calls,
        dependencies: normalizer.dependencies,
    })
}

struct GraphNormalizer<'a> {
    values: &'a mut Vec<Type>,
    views: &'a [LocalView],
    value_storage: Vec<Option<StorageRef>>,
    inputs: &'a [Port],
    results: &'a [Port],
    capabilities: &'a BTreeSet<String>,
    numerical_semantics: Vec<NumericalEffect>,
    operands: Vec<LogicalOperand>,
    producers: Vec<Option<LogicalEndpoint>>,
    tasks: Vec<LogicalTask>,
    calls: Vec<LogicalCallBoundary>,
    dependencies: Vec<LogicalDependency>,
    output_cases: BTreeMap<u32, Vec<(Vec<LogicalPredicate>, OperandId)>>,
    last: Option<LogicalEndpoint>,
}

impl GraphNormalizer<'_> {
    fn value(&mut self, ty: Type) -> LocalValueId {
        let id = LocalValueId(self.values.len() as u32);
        self.values.push(ty);
        self.value_storage.push(None);
        id
    }

    fn operand(&mut self, ty: Type, value: LocalValueId, storage: Option<StorageRef>) -> OperandId {
        let id = OperandId(self.operands.len() as u32);
        self.operands.push(LogicalOperand {
            id,
            ty,
            value: ValueRef::Local(value),
            storage,
        });
        self.producers.push(None);
        id
    }

    fn sequence(&mut self, next: LogicalEndpoint) {
        if let Some(previous) = self.last.replace(next) {
            let id = DependencyId(self.dependencies.len() as u32);
            self.dependencies.push(LogicalDependency {
                id,
                from: previous,
                to: next,
                kind: LogicalDependencyKind::Control,
            });
        }
    }

    fn task(
        &mut self,
        domain: LogicalDomain,
        operations: LogicalBlock,
        inputs: Vec<OperandId>,
        outputs: Vec<OperandId>,
        span: Span,
    ) -> TaskId {
        let id = TaskId(self.tasks.len() as u32);
        let effects = self.operation_effects(&operations);
        self.tasks.push(LogicalTask {
            id,
            domain,
            body: ScalarBody { operations },
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            effects: effects.clone(),
            capabilities: self.capabilities.clone(),
            numerical_semantics: std::mem::take(&mut self.numerical_semantics),
            span,
        });
        self.sequence(LogicalEndpoint::Task(id));
        self.connect_operands(&inputs, LogicalEndpoint::Task(id));
        for output in outputs {
            self.producers[output.index()] = Some(LogicalEndpoint::Task(id));
        }
        for access in effects.accesses {
            let (from, to) = match (&access.storage, access.kind) {
                (StorageRef::Input { port, .. }, LogicalAccessKind::Read) => {
                    (LogicalEndpoint::Input(*port), LogicalEndpoint::Task(id))
                }
                (
                    StorageRef::Result { port, .. },
                    LogicalAccessKind::Write | LogicalAccessKind::Move,
                ) => (LogicalEndpoint::Task(id), LogicalEndpoint::Output(*port)),
                _ => continue,
            };
            let kind = match access.kind {
                LogicalAccessKind::Move => LogicalDependencyKind::Ownership(access.storage),
                _ => LogicalDependencyKind::Effect(access.storage),
            };
            if !self
                .dependencies
                .iter()
                .any(|edge| edge.from == from && edge.to == to && edge.kind == kind)
            {
                let edge = DependencyId(self.dependencies.len() as u32);
                self.dependencies.push(LogicalDependency {
                    id: edge,
                    from,
                    to,
                    kind,
                });
            }
        }
        id
    }

    fn connect_operands(&mut self, inputs: &[OperandId], consumer: LogicalEndpoint) {
        for operand in inputs {
            let Some(producer) = self.producers[operand.index()] else {
                continue;
            };
            if self.dependencies.iter().any(|edge| {
                edge.from == producer
                    && edge.to == consumer
                    && edge.kind == LogicalDependencyKind::Value(*operand)
            }) {
                continue;
            }
            let id = DependencyId(self.dependencies.len() as u32);
            self.dependencies.push(LogicalDependency {
                id,
                from: producer,
                to: consumer,
                kind: LogicalDependencyKind::Value(*operand),
            });
        }
    }

    fn block(&mut self, body: LogicalBlock, domain: LogicalDomain) -> Result<(), String> {
        let mut scalar = Vec::new();
        for operation in body {
            match operation.kind {
                LogicalOperationKind::For {
                    independent,
                    binder,
                    lo,
                    hi,
                    source,
                    body,
                } => {
                    self.flush_scalar(&domain, &mut scalar);
                    let lo = self.expr(lo, &domain)?;
                    let hi = self.expr(hi, &domain)?;
                    let runtime = source.map(|value| self.expr(value, &domain)).transpose()?;
                    let mut nested = domain.clone();
                    nested.axes.push(LogicalAxis {
                        binders: vec![binder],
                        order: if independent {
                            AxisOrder::Independent
                        } else {
                            AxisOrder::Ordered
                        },
                        source: LogicalAxisSource::Range { lo, hi, runtime },
                    });
                    self.block(body, nested)?;
                }
                LogicalOperationKind::Coordinates {
                    binders,
                    value,
                    axes,
                    body,
                } => {
                    self.flush_scalar(&domain, &mut scalar);
                    let value = self.expr(value, &domain)?;
                    let mut nested = domain.clone();
                    nested.axes.push(LogicalAxis {
                        binders,
                        order: self.axis_order(&domain),
                        source: LogicalAxisSource::Coordinates { value, axes },
                    });
                    self.block(body, nested)?;
                }
                LogicalOperationKind::Members {
                    binder,
                    slice,
                    body,
                } => {
                    self.flush_scalar(&domain, &mut scalar);
                    let mut nested = domain.clone();
                    nested.axes.push(LogicalAxis {
                        binders: vec![binder],
                        order: self.axis_order(&domain),
                        source: LogicalAxisSource::Members { slice },
                    });
                    self.block(body, nested)?;
                }
                LogicalOperationKind::Stages(stages) => {
                    self.flush_scalar(&domain, &mut scalar);
                    for stage in stages {
                        self.block(stage.body, domain.clone())?;
                    }
                }
                LogicalOperationKind::If {
                    condition,
                    then,
                    els,
                } if has_block_boundary(&then) || has_block_boundary(&els) => {
                    self.flush_scalar(&domain, &mut scalar);
                    let condition = self.expr(condition, &domain)?;
                    let mut when_true = domain.clone();
                    when_true.predicates.push(LogicalPredicate {
                        condition: condition.clone(),
                        when_true: true,
                    });
                    self.block(then, when_true)?;
                    let mut when_false = domain.clone();
                    when_false.predicates.push(LogicalPredicate {
                        condition,
                        when_true: false,
                    });
                    self.block(els, when_false)?;
                }
                LogicalOperationKind::Region(region) => {
                    self.flush_scalar(&domain, &mut scalar);
                    self.region(region, domain.clone())?;
                }
                kind => {
                    let operation = LogicalOperation {
                        kind,
                        span: operation.span,
                    };
                    if has_expression_boundary(&operation) {
                        self.flush_scalar(&domain, &mut scalar);
                    }
                    let operation = self.scalar_operation(operation, &domain)?;
                    scalar.push(operation);
                }
            }
        }
        self.flush_scalar(&domain, &mut scalar);
        Ok(())
    }

    fn flush_scalar(&mut self, domain: &LogicalDomain, operations: &mut LogicalBlock) {
        if operations.is_empty() {
            return;
        }
        let body = std::mem::take(operations);
        let span = body
            .first()
            .map(|operation| operation.span)
            .unwrap_or_default();
        let inputs = self.operand_inputs(&body);
        let returned = body
            .iter()
            .filter_map(|operation| match &operation.kind {
                LogicalOperationKind::Return(writes) => Some(writes.clone()),
                _ => None,
            })
            .flatten()
            .collect::<Vec<_>>();
        let mut outputs = Vec::new();
        let mut output_ports = Vec::new();
        for write in returned {
            let value = match &write.value.kind {
                LogicalExprKind::Value(ValueRef::Local(value)) => *value,
                _ => self.value(write.value.ty.clone()),
            };
            let operand = self.operand(
                write.value.ty.clone(),
                value,
                self.expr_storage(&write.value),
            );
            outputs.push(operand);
            output_ports.push((operand, write.port));
        }
        let task = self.task(domain.clone(), body, inputs, outputs, span);
        for (operand, port) in output_ports {
            let _ = task;
            self.output_cases
                .entry(port)
                .or_default()
                .push((domain.predicates.clone(), operand));
        }
    }

    fn finish_outputs(&mut self) -> Result<(), String> {
        for (port, cases) in std::mem::take(&mut self.output_cases) {
            let (producer, operand) = if cases.len() == 1 {
                let operand = cases[0].1;
                (
                    self.producers[operand.index()]
                        .ok_or_else(|| format!("output#{port} value has no producer"))?,
                    operand,
                )
            } else {
                let first = self
                    .operands
                    .get(cases[0].1.index())
                    .ok_or_else(|| format!("output#{port} has no cases"))?
                    .clone();
                if cases
                    .iter()
                    .any(|(_, operand)| self.operands[operand.index()].ty != first.ty)
                {
                    return Err(format!(
                        "output#{port} conditional cases have different types"
                    ));
                }
                let binder = self.value(first.ty.clone());
                let output = self.operand(first.ty.clone(), binder, first.storage.clone());
                let inputs = cases
                    .iter()
                    .map(|(_, operand)| *operand)
                    .collect::<Vec<_>>();
                let operation = LogicalOperation {
                    kind: LogicalOperationKind::ConditionalMerge {
                        binder,
                        cases: cases
                            .into_iter()
                            .map(|(predicates, value)| LogicalConditionalCase { predicates, value })
                            .collect(),
                    },
                    span: Span::default(),
                };
                let task = self.task(
                    LogicalDomain::default(),
                    vec![operation],
                    inputs,
                    vec![output],
                    Span::default(),
                );
                (LogicalEndpoint::Task(task), output)
            };
            let id = DependencyId(self.dependencies.len() as u32);
            self.dependencies.push(LogicalDependency {
                id,
                from: producer,
                to: LogicalEndpoint::Output(port),
                kind: LogicalDependencyKind::Value(operand),
            });
        }
        Ok(())
    }

    fn operand_inputs(&mut self, body: &LogicalBlock) -> Vec<OperandId> {
        let mut values = BTreeSet::new();
        for operation in body {
            collect_operation_values(operation, &mut values);
        }
        for value in values.iter().filter(|value| !matches!(value, ValueRef::Local(_))) {
            if self.operands.iter().any(|operand| &operand.value == value) {
                continue;
            }
            let (ty, storage, producer) = match value {
                ValueRef::Input(port) => {
                    let Some(input) = self.inputs.get(*port as usize) else {
                        continue;
                    };
                    (
                        input.ty.clone(),
                        matches!(input.ty, Type::Tensor(_)).then_some(StorageRef::Input {
                            port: *port,
                            path: Vec::new(),
                        }),
                        Some(LogicalEndpoint::Input(*port)),
                    )
                }
                ValueRef::Result(port) => {
                    let Some(result) = self.results.get(*port as usize) else {
                        continue;
                    };
                    (
                        result.ty.clone(),
                        matches!(result.ty, Type::Tensor(_)).then_some(StorageRef::Result {
                            port: *port,
                            path: Vec::new(),
                        }),
                        None,
                    )
                }
                ValueRef::Local(_) => unreachable!(),
            };
            let id = OperandId(self.operands.len() as u32);
            self.operands.push(LogicalOperand {
                id,
                ty,
                value: value.clone(),
                storage,
            });
            self.producers.push(producer);
        }
        let mut operands = self
            .operands
            .iter()
            .filter(|operand| {
                values.contains(&operand.value)
                    && (!matches!(operand.value, ValueRef::Local(_))
                        || self.producers[operand.id.index()].is_some())
            })
            .map(|operand| operand.id)
            .collect::<BTreeSet<_>>();
        for call in &self.calls {
            if values.contains(&self.operands[call.result.index()].value)
            {
                operands.extend(call.outputs.iter().copied());
            }
        }
        operands.into_iter().collect()
    }

    fn axis_order(&self, domain: &LogicalDomain) -> AxisOrder {
        match domain.regions.last().map(|region| region.mode) {
            Some(sir::RegionMode::Parallel | sir::RegionMode::Pipeline) => AxisOrder::Independent,
            _ => AxisOrder::Ordered,
        }
    }

    fn region(&mut self, region: LogicalRegion, domain: LogicalDomain) -> Result<(), String> {
        let source = region
            .source
            .map(|value| self.expr(*value, &domain))
            .transpose()?;
        let merge = region
            .merge
            .as_ref()
            .map(|merge| -> Result<_, String> {
                Ok(LogicalMergeSignature {
                    left: merge.left.clone(),
                    right: merge.right.clone(),
                    identity: self.expr(merge.identity.clone(), &domain)?,
                })
            })
            .transpose()?;
        let mut nested = domain;
        nested.regions.push(LogicalRegionFrame {
            id: region.id,
            mode: region.mode,
            binders: region.binders,
            source,
            merge,
            result: region.result,
        });
        self.block(region.body, nested.clone())?;
        if let Some(merge) = region.merge {
            self.block(merge.body, nested)?;
        }
        Ok(())
    }

    fn scalar_operation(
        &mut self,
        operation: LogicalOperation,
        domain: &LogicalDomain,
    ) -> Result<LogicalOperation, String> {
        let span = operation.span;
        let kind = match operation.kind {
            LogicalOperationKind::Bind { pattern, value } => LogicalOperationKind::Bind {
                pattern,
                value: self.expr(value, domain)?,
            },
            LogicalOperationKind::Assign { target, op, value } => LogicalOperationKind::Assign {
                target: self.expr(target, domain)?,
                op,
                value: self.expr(value, domain)?,
            },
            LogicalOperationKind::If {
                condition,
                then,
                els,
            } => LogicalOperationKind::If {
                condition: self.expr(condition, domain)?,
                then: self.scalar_nested(then, domain)?,
                els: self.scalar_nested(els, domain)?,
            },
            LogicalOperationKind::Publish { value, destination } => LogicalOperationKind::Publish {
                value: self.expr(value, domain)?,
                destination: self.expr(destination, domain)?,
            },
            LogicalOperationKind::Yield(values) => LogicalOperationKind::Yield(
                values
                    .into_iter()
                    .map(|value| self.expr(value, domain))
                    .collect::<Result<_, _>>()?,
            ),
            LogicalOperationKind::Return(writes) => LogicalOperationKind::Return(
                writes
                    .into_iter()
                    .map(|write| {
                        Ok(ResultWrite {
                            value: self.expr(write.value, domain)?,
                            ..write
                        })
                    })
                    .collect::<Result<_, String>>()?,
            ),
            LogicalOperationKind::Expr(value) => {
                LogicalOperationKind::Expr(self.expr(value, domain)?)
            }
            LogicalOperationKind::Reduction {
                binder,
                value,
                axis,
                op,
                unordered,
            } => LogicalOperationKind::Reduction {
                binder,
                value: self.expr(value, domain)?,
                axis,
                op,
                unordered,
            },
            LogicalOperationKind::ConditionalMerge { binder, cases } => {
                LogicalOperationKind::ConditionalMerge { binder, cases }
            }
            LogicalOperationKind::Region(_)
            | LogicalOperationKind::Stages(_)
            | LogicalOperationKind::For { .. }
            | LogicalOperationKind::Coordinates { .. }
            | LogicalOperationKind::Members { .. } => {
                return Err("scheduling construct reached scalar normalization".into())
            }
        };
        Ok(LogicalOperation { kind, span })
    }

    fn scalar_nested(
        &mut self,
        body: LogicalBlock,
        domain: &LogicalDomain,
    ) -> Result<LogicalBlock, String> {
        body.into_iter()
            .map(|operation| self.scalar_operation(operation, domain))
            .collect()
    }

    fn expr(
        &mut self,
        expression: LogicalExpr,
        domain: &LogicalDomain,
    ) -> Result<LogicalExpr, String> {
        let ty = expression.ty.clone();
        let span = expression.span;
        let kind = match expression.kind {
            LogicalExprKind::Call {
                choice,
                args,
                results,
            } => {
                let mut inputs = Vec::new();
                for argument in args {
                    let argument = self.expr(argument, domain)?;
                    let value = self.value(argument.ty.clone());
                    let operand =
                        self.operand(argument.ty.clone(), value, self.expr_storage(&argument));
                    self.task(
                        domain.clone(),
                        vec![LogicalOperation {
                            kind: LogicalOperationKind::Bind {
                                pattern: LogicalPattern::Value(value),
                                value: argument,
                            },
                            span,
                        }],
                        Vec::new(),
                        vec![operand],
                        span,
                    );
                    inputs.push(operand);
                }
                let value = self.value(ty.clone());
                let storage = (!matches!(ty, Type::Tuple(_)))
                    .then(|| {
                        results
                            .first()
                            .map(|result| StorageRef::Local(result.storage))
                    })
                    .flatten();
                self.value_storage[value.index()] = storage.clone();
                let result = self.operand(ty.clone(), value, storage);
                let mut leaves = Vec::new();
                flatten_operand_types(&ty, &mut Vec::new(), &mut leaves);
                let mut outputs = Vec::with_capacity(leaves.len());
                for (path, leaf_ty) in leaves {
                    if path.is_empty() {
                        outputs.push(result);
                        continue;
                    }
                    let leaf_value = self.value(leaf_ty.clone());
                    let leaf_storage = results
                        .iter()
                        .find(|candidate| candidate.path == path)
                        .map(|candidate| StorageRef::Local(candidate.storage));
                    self.value_storage[leaf_value.index()] = leaf_storage.clone();
                    outputs.push(self.operand(leaf_ty, leaf_value, leaf_storage));
                }
                let id = CallBoundaryId(self.calls.len() as u32);
                let mut accesses = inputs
                    .iter()
                    .filter_map(|operand| self.operands[operand.index()].storage.clone())
                    .map(|storage| LogicalAccess {
                        storage,
                        view: None,
                        kind: LogicalAccessKind::Read,
                    })
                    .collect::<Vec<_>>();
                for result in &results {
                    accesses.push(LogicalAccess {
                        storage: StorageRef::Local(result.storage),
                        view: None,
                        kind: LogicalAccessKind::Write,
                    });
                }
                self.calls.push(LogicalCallBoundary {
                    id,
                    choice,
                    inputs,
                    result,
                    outputs: outputs.clone(),
                    domain: domain.clone(),
                    results,
                    effects: EffectFootprint { accesses },
                    span,
                });
                self.sequence(LogicalEndpoint::Call(id));
                let inputs = self.calls[id.index()].inputs.clone();
                self.connect_operands(&inputs, LogicalEndpoint::Call(id));
                self.producers[result.index()] = Some(LogicalEndpoint::Call(id));
                for output in outputs {
                    self.producers[output.index()] = Some(LogicalEndpoint::Call(id));
                }
                LogicalExprKind::Value(ValueRef::Local(value))
            }
            LogicalExprKind::Reduce {
                value,
                axis,
                op,
                unordered,
            } => {
                let value = self.expr(*value, domain)?;
                let binder = self.value(ty.clone());
                let input_value = self.value(value.ty.clone());
                let input = self.operand(value.ty.clone(), input_value, self.expr_storage(&value));
                self.task(
                    domain.clone(),
                    vec![LogicalOperation {
                        kind: LogicalOperationKind::Bind {
                            pattern: LogicalPattern::Value(input_value),
                            value,
                        },
                        span,
                    }],
                    Vec::new(),
                    vec![input],
                    span,
                );
                let output = self.operand(ty.clone(), binder, None);
                self.task(
                    domain.clone(),
                    vec![LogicalOperation {
                        kind: LogicalOperationKind::Reduction {
                            binder,
                            value: LogicalExpr {
                                ty: self.values[input_value.index()].clone(),
                                kind: LogicalExprKind::Value(ValueRef::Local(input_value)),
                                span,
                            },
                            axis,
                            op,
                            unordered,
                        },
                        span,
                    }],
                    vec![input],
                    vec![output],
                    span,
                );
                LogicalExprKind::Value(ValueRef::Local(binder))
            }
            LogicalExprKind::Region(region) => {
                self.region(*region, domain.clone())?;
                let value = self.value(ty.clone());
                let output = self.operand(ty.clone(), value, None);
                if let Some(last) = self.last {
                    if let Some(task) = self.tasks.last_mut() {
                        task.outputs.push(output);
                    } else if let Some(call) = self.calls.last_mut() {
                        call.outputs.push(output);
                    }
                    self.producers[output.index()] = Some(last);
                    let _ = last;
                }
                LogicalExprKind::Value(ValueRef::Local(value))
            }
            kind => self.scalar_expr_kind(kind, domain)?,
        };
        Ok(LogicalExpr { ty, kind, span })
    }

    fn scalar_expr_kind(
        &mut self,
        kind: LogicalExprKind,
        domain: &LogicalDomain,
    ) -> Result<LogicalExprKind, String> {
        let boxed =
            |this: &mut Self, value: Box<LogicalExpr>| this.expr(*value, domain).map(Box::new);
        Ok(match kind {
            LogicalExprKind::Tuple(values) => LogicalExprKind::Tuple(
                values
                    .into_iter()
                    .map(|value| self.expr(value, domain))
                    .collect::<Result<_, _>>()?,
            ),
            LogicalExprKind::Range(a, b) => {
                LogicalExprKind::Range(boxed(self, a)?, boxed(self, b)?)
            }
            LogicalExprKind::Field(value, index) => {
                LogicalExprKind::Field(boxed(self, value)?, index)
            }
            LogicalExprKind::Filled {
                storage,
                like,
                value,
            } => LogicalExprKind::Filled {
                storage,
                like: boxed(self, like)?,
                value,
            },
            LogicalExprKind::View { view, base } => LogicalExprKind::View {
                view,
                base: boxed(self, base)?,
            },
            LogicalExprKind::Index { base, indices } => LogicalExprKind::Index {
                base: boxed(self, base)?,
                indices: indices
                    .into_iter()
                    .map(|index| self.index(index, domain))
                    .collect::<Result<_, _>>()?,
            },
            LogicalExprKind::Snapshot { storage, source } => LogicalExprKind::Snapshot {
                storage,
                source: boxed(self, source)?,
            },
            LogicalExprKind::Decode { storage, source } => LogicalExprKind::Decode {
                storage,
                source: boxed(self, source)?,
            },
            LogicalExprKind::Materialize { storage, value } => LogicalExprKind::Materialize {
                storage,
                value: boxed(self, value)?,
            },
            LogicalExprKind::Cast { dtype, expr } => LogicalExprKind::Cast {
                dtype,
                expr: boxed(self, expr)?,
            },
            LogicalExprKind::Unary { op, expr } => LogicalExprKind::Unary {
                op,
                expr: boxed(self, expr)?,
            },
            LogicalExprKind::Binary { op, lhs, rhs } => LogicalExprKind::Binary {
                op,
                lhs: boxed(self, lhs)?,
                rhs: boxed(self, rhs)?,
            },
            LogicalExprKind::Math { op, args } => LogicalExprKind::Math {
                op,
                args: args
                    .into_iter()
                    .map(|value| self.expr(value, domain))
                    .collect::<Result<_, _>>()?,
            },
            LogicalExprKind::Select { cond, then, els } => LogicalExprKind::Select {
                cond: boxed(self, cond)?,
                then: boxed(self, then)?,
                els: boxed(self, els)?,
            },
            LogicalExprKind::Extent { base, axis } => LogicalExprKind::Extent {
                base: boxed(self, base)?,
                axis,
            },
            LogicalExprKind::Intrinsic { operation, args } => LogicalExprKind::Intrinsic {
                operation,
                args: args
                    .into_iter()
                    .map(|value| self.expr(value, domain))
                    .collect::<Result<_, _>>()?,
            },
            LogicalExprKind::Accessor { base, name } => LogicalExprKind::Accessor {
                base: boxed(self, base)?,
                name,
            },
            LogicalExprKind::Geometry { base, axis, valid } => LogicalExprKind::Geometry {
                base: boxed(self, base)?,
                axis,
                valid,
            },
            LogicalExprKind::Atomic { op, place, value } => LogicalExprKind::Atomic {
                op,
                place: boxed(self, place)?,
                value: boxed(self, value)?,
            },
            LogicalExprKind::Call { .. }
            | LogicalExprKind::Reduce { .. }
            | LogicalExprKind::Region(_) => {
                unreachable!("scheduling expression handled before scalar recursion")
            }
            other => other,
        })
    }

    fn index(
        &mut self,
        index: LogicalIndex,
        domain: &LogicalDomain,
    ) -> Result<LogicalIndex, String> {
        Ok(match index {
            LogicalIndex::Point(value) => LogicalIndex::Point(Box::new(self.expr(*value, domain)?)),
            LogicalIndex::Range { start, end } => LogicalIndex::Range {
                start: start
                    .map(|value| self.expr(*value, domain).map(Box::new))
                    .transpose()?,
                end: end
                    .map(|value| self.expr(*value, domain).map(Box::new))
                    .transpose()?,
            },
            other => other,
        })
    }

    fn expr_storage(&self, expression: &LogicalExpr) -> Option<StorageRef> {
        match &expression.kind {
            LogicalExprKind::Construct { storage }
            | LogicalExprKind::Filled { storage, .. }
            | LogicalExprKind::Snapshot { storage, .. }
            | LogicalExprKind::Decode { storage, .. }
            | LogicalExprKind::Materialize { storage, .. } => Some(StorageRef::Local(*storage)),
            LogicalExprKind::View { view, .. } => self
                .views
                .get(view.index())
                .map(|view| view.storage.clone()),
            LogicalExprKind::Value(ValueRef::Input(port))
                if matches!(expression.ty, Type::Tensor(_)) =>
            {
                Some(StorageRef::Input {
                    port: *port,
                    path: Vec::new(),
                })
            }
            LogicalExprKind::Value(ValueRef::Result(port))
                if matches!(expression.ty, Type::Tensor(_)) =>
            {
                Some(StorageRef::Result {
                    port: *port,
                    path: Vec::new(),
                })
            }
            LogicalExprKind::Value(ValueRef::Local(value)) => {
                self.value_storage.get(value.index()).cloned().flatten()
            }
            LogicalExprKind::Field(base, _)
            | LogicalExprKind::Index { base, .. }
            | LogicalExprKind::Accessor { base, .. }
            | LogicalExprKind::Geometry { base, .. } => self.expr_storage(base),
            _ => None,
        }
    }

    fn operation_effects(&self, operations: &LogicalBlock) -> EffectFootprint {
        let mut accesses = Vec::new();
        for operation in operations {
            match &operation.kind {
                LogicalOperationKind::Bind { value, .. }
                | LogicalOperationKind::Expr(value)
                | LogicalOperationKind::Reduction { value, .. } => {
                    self.read_expr(value, &mut accesses)
                }
                LogicalOperationKind::Assign { target, value, .. } => {
                    self.write_expr(target, &mut accesses);
                    self.read_expr(value, &mut accesses);
                }
                LogicalOperationKind::If {
                    condition,
                    then,
                    els,
                } => {
                    self.read_expr(condition, &mut accesses);
                    accesses.extend(self.operation_effects(then).accesses);
                    accesses.extend(self.operation_effects(els).accesses);
                }
                LogicalOperationKind::Publish { value, destination } => {
                    self.read_expr(value, &mut accesses);
                    self.write_expr(destination, &mut accesses);
                }
                LogicalOperationKind::Yield(values) => {
                    for value in values {
                        self.read_expr(value, &mut accesses);
                    }
                }
                LogicalOperationKind::Return(writes) => {
                    for write in writes {
                        self.read_expr(&write.value, &mut accesses);
                        if write.transfer {
                            if let Some(storage) = self.expr_storage(&write.value) {
                                accesses.push(LogicalAccess {
                                    storage,
                                    view: None,
                                    kind: LogicalAccessKind::Move,
                                });
                            }
                        }
                        if matches!(write.value.ty, Type::Tensor(_)) {
                            accesses.push(LogicalAccess {
                                storage: StorageRef::Result {
                                    port: write.port,
                                    path: Vec::new(),
                                },
                                view: None,
                                kind: if write.transfer {
                                    LogicalAccessKind::Move
                                } else {
                                    LogicalAccessKind::Write
                                },
                            });
                        }
                    }
                }
                LogicalOperationKind::ConditionalMerge { .. } => {}
                LogicalOperationKind::Region(_)
                | LogicalOperationKind::Stages(_)
                | LogicalOperationKind::For { .. }
                | LogicalOperationKind::Coordinates { .. }
                | LogicalOperationKind::Members { .. } => {}
            }
        }
        accesses.sort_by_key(|access| (access.storage.clone(), access.view, access.kind as u8));
        accesses.dedup();
        EffectFootprint { accesses }
    }

    fn write_expr(&self, expression: &LogicalExpr, accesses: &mut Vec<LogicalAccess>) {
        if let Some(storage) = self.expr_storage(expression) {
            accesses.push(LogicalAccess {
                storage,
                view: None,
                kind: LogicalAccessKind::Write,
            });
        }
        self.read_children(expression, accesses);
    }

    fn read_expr(&self, expression: &LogicalExpr, accesses: &mut Vec<LogicalAccess>) {
        if let Some(storage) = self.expr_storage(expression) {
            accesses.push(LogicalAccess {
                storage,
                view: None,
                kind: LogicalAccessKind::Read,
            });
        }
        self.read_children(expression, accesses);
    }

    fn read_children(&self, expression: &LogicalExpr, accesses: &mut Vec<LogicalAccess>) {
        let mut read = |value: &LogicalExpr| self.read_expr(value, accesses);
        match &expression.kind {
            LogicalExprKind::Tuple(values)
            | LogicalExprKind::Math { args: values, .. }
            | LogicalExprKind::Intrinsic { args: values, .. } => {
                for value in values {
                    read(value);
                }
            }
            LogicalExprKind::Range(a, b)
            | LogicalExprKind::Binary { lhs: a, rhs: b, .. }
            | LogicalExprKind::Atomic {
                place: a, value: b, ..
            } => {
                read(a);
                read(b);
            }
            LogicalExprKind::Field(value, _)
            | LogicalExprKind::View { base: value, .. }
            | LogicalExprKind::Snapshot { source: value, .. }
            | LogicalExprKind::Decode { source: value, .. }
            | LogicalExprKind::Materialize { value, .. }
            | LogicalExprKind::Cast { expr: value, .. }
            | LogicalExprKind::Unary { expr: value, .. }
            | LogicalExprKind::Reduce { value, .. }
            | LogicalExprKind::Extent { base: value, .. }
            | LogicalExprKind::Accessor { base: value, .. }
            | LogicalExprKind::Geometry { base: value, .. }
            | LogicalExprKind::Filled { like: value, .. } => read(value),
            LogicalExprKind::Index { base, indices } => {
                read(base);
                for index in indices {
                    match index {
                        LogicalIndex::Point(value) => read(value),
                        LogicalIndex::Range { start, end } => {
                            for value in start.iter().chain(end) {
                                read(value);
                            }
                        }
                        _ => {}
                    }
                }
            }
            LogicalExprKind::Select { cond, then, els } => {
                read(cond);
                read(then);
                read(els);
            }
            LogicalExprKind::Region(_)
            | LogicalExprKind::Call { .. }
            | LogicalExprKind::Int(_)
            | LogicalExprKind::Float(_)
            | LogicalExprKind::Bool(_)
            | LogicalExprKind::Value(_)
            | LogicalExprKind::Shape(_)
            | LogicalExprKind::Construct { .. }
            | LogicalExprKind::Coordinate(_) => {}
        }
    }
}

struct Translator<'a> {
    definition: &'a sir::Definition,
    shapes: &'a BTreeMap<String, i64>,
    elems: &'a BTreeMap<String, Elem>,
    structural: StructuralSymbols,
    results: Vec<Port>,
    value_types: Vec<Type>,
    storage: Vec<LocalStorage>,
    views: Vec<LocalView>,
    runtime_extents: BTreeMap<String, LogicalExpr>,
    var_storage: Vec<Option<StorageRef>>,
    calls: BTreeMap<sir::CallId, ChoiceId>,
}

impl<'a> Translator<'a> {
    fn new(
        family: &'a family::Family,
        owner: CandidateRef,
        definition: &'a sir::Definition,
        shapes: &'a BTreeMap<String, i64>,
        elems: &'a BTreeMap<String, Elem>,
        structural: StructuralSymbols,
        results: Vec<Port>,
    ) -> Result<Self, String> {
        let value_types = definition
            .body
            .vars
            .iter()
            .map(|var| body_type(&var.ty, shapes, elems, &structural))
            .collect::<Result<Vec<_>, _>>()?;
        let calls = family
            .candidate(owner)
            .children
            .iter()
            .map(|child| {
                let occurrence = family.occurrence(*child);
                Ok((
                    occurrence
                        .call
                        .ok_or_else(|| format!("child choice#{} has no call", child.0))?,
                    ChoiceId(child.0),
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            definition,
            shapes,
            elems,
            structural,
            results,
            value_types,
            storage: Vec::new(),
            views: Vec::new(),
            runtime_extents: BTreeMap::new(),
            var_storage: vec![None; definition.body.vars.len()],
            calls,
        })
    }

    fn block(&mut self, block: &[sir::Stmt]) -> Result<LogicalBlock, String> {
        block.iter().map(|stmt| self.statement(stmt)).collect()
    }

    fn statement(&mut self, stmt: &sir::Stmt) -> Result<LogicalOperation, String> {
        let kind = match &stmt.kind {
            SStmt::Bind { pattern, value } => {
                let value = self.expr(value)?;
                let pattern = self.pattern(pattern);
                self.bind_storage(pattern.clone(), &value);
                LogicalOperationKind::Bind { pattern, value }
            }
            SStmt::Assign { target, op, value } => LogicalOperationKind::Assign {
                target: self.expr(target)?,
                op: *op,
                value: self.expr(value)?,
            },
            SStmt::Region(region) => LogicalOperationKind::Region(self.region(region)?),
            SStmt::Stages(stages) => LogicalOperationKind::Stages(
                stages
                    .iter()
                    .map(|stage| {
                        Ok(LogicalStage {
                            name: stage.name.clone(),
                            ports: stage
                                .ports
                                .iter()
                                .map(|id| LocalValueId(*id as u32))
                                .collect(),
                            body: self.block(&stage.body)?,
                            span: stage.span,
                        })
                    })
                    .collect::<Result<_, String>>()?,
            ),
            SStmt::Range {
                kind,
                var,
                lo,
                hi,
                value,
                body,
            } => {
                self.bind_runtime_binder(*var)?;
                LogicalOperationKind::For {
                    independent: *kind == sir::LoopKind::Parallel,
                    binder: LocalValueId(*var as u32),
                    lo: self.expr(lo)?,
                    hi: self.expr(hi)?,
                    source: value.as_ref().map(|v| self.expr(v)).transpose()?,
                    body: self.block(body)?,
                }
            }
            SStmt::Coordinates {
                vars,
                of,
                axes,
                body,
            } => LogicalOperationKind::Coordinates {
                binders: vars.iter().map(|id| LocalValueId(*id as u32)).collect(),
                value: self.expr(of)?,
                axes: axes.clone(),
                body: self.block(body)?,
            },
            SStmt::Members { var, slice, body } => LogicalOperationKind::Members {
                binder: LocalValueId(*var as u32),
                slice: slice.0,
                body: self.block(body)?,
            },
            SStmt::If { cond, then, els } => LogicalOperationKind::If {
                condition: self.expr(cond)?,
                then: self.block(then)?,
                els: self.block(els)?,
            },
            SStmt::Publish { value, destination } => LogicalOperationKind::Publish {
                value: self.expr(value)?,
                destination: self.expr(destination)?,
            },
            SStmt::Yield(values) => LogicalOperationKind::Yield(
                values
                    .iter()
                    .map(|value| self.expr(value))
                    .collect::<Result<_, _>>()?,
            ),
            SStmt::Return(values) => LogicalOperationKind::Return(self.result_writes(values)?),
            SStmt::Expr(expr) => LogicalOperationKind::Expr(self.expr(expr)?),
        };
        Ok(LogicalOperation {
            kind,
            span: stmt.span,
        })
    }

    fn pattern(&self, pattern: &sir::Pattern) -> LogicalPattern {
        match pattern {
            sir::Pattern::Var(id) => LogicalPattern::Value(LocalValueId(*id as u32)),
            sir::Pattern::Tuple(items) => {
                LogicalPattern::Tuple(items.iter().map(|item| self.pattern(item)).collect())
            }
        }
    }

    /// Record the value source for a checker variable that survives in a
    /// runtime-valued shape. The occurrence-qualified symbol is an identity;
    /// this binding is its value provenance. Keeping the two separate avoids
    /// either parsing symbol names in backends or substituting a capacity for
    /// the value at execution time.
    fn bind_runtime_binder(&mut self, var: usize) -> Result<(), String> {
        let variable = self
            .definition
            .body
            .vars
            .get(var)
            .ok_or("runtime extent binder names an absent body variable")?;
        let authored = Sym::atom(crate::check::var_atom(&variable.name, var));
        let canonical = self.structural.substitute(&authored, self.shapes);
        let atoms = canonical.atoms();
        let [Atom::Param(symbol)] = atoms.as_slice() else {
            return Ok(());
        };
        if canonical != Sym::param(symbol) || !symbol.starts_with("@runtime.") {
            return Ok(());
        }
        let ty = self
            .value_types
            .get(var)
            .cloned()
            .ok_or("runtime extent binder has no logical value type")?;
        self.runtime_extents
            .entry(symbol.clone())
            .or_insert(LogicalExpr {
                ty,
                kind: LogicalExprKind::Value(ValueRef::Local(LocalValueId(var as u32))),
                span: Span::default(),
            });
        Ok(())
    }
    fn bind_storage(&mut self, pattern: LogicalPattern, value: &LogicalExpr) {
        match pattern {
            LogicalPattern::Value(id) => self.var_storage[id.0 as usize] = self.expr_storage(value),
            LogicalPattern::Tuple(items) => {
                if let LogicalExprKind::Tuple(values) = &value.kind {
                    for (item, value) in items.into_iter().zip(values) {
                        self.bind_storage(item, value);
                    }
                } else {
                    for (index, item) in items.into_iter().enumerate() {
                        let Ok(ty) = field_type(&value.ty, index) else {
                            continue;
                        };
                        self.bind_storage(
                            item,
                            &LogicalExpr {
                                ty,
                                kind: LogicalExprKind::Field(Box::new(value.clone()), index),
                                span: value.span,
                            },
                        );
                    }
                }
            }
        }
    }

    fn result_writes(&mut self, values: &[sir::Expr]) -> Result<Vec<ResultWrite>, String> {
        let expressions = values
            .iter()
            .map(|value| self.expr(value))
            .collect::<Result<Vec<_>, _>>()?;
        let root = if expressions.len() == 1 {
            expressions.into_iter().next().unwrap()
        } else {
            LogicalExpr {
                ty: Type::Tuple(expressions.iter().map(|v| v.ty.clone()).collect()),
                kind: LogicalExprKind::Tuple(expressions),
                span: Span::default(),
            }
        };
        let mut writes = Vec::new();
        for (port, result) in self.results.clone().into_iter().enumerate() {
            let mut value = root.clone();
            for index in &result.path {
                value = LogicalExpr {
                    ty: field_type(&value.ty, *index as usize)?,
                    kind: LogicalExprKind::Field(Box::new(value), *index as usize),
                    span: root.span,
                };
            }
            writes.push(ResultWrite {
                port: port as u32,
                path: result.path,
                transfer: matches!(result.ty, Type::Tensor(_)),
                value,
            });
        }
        Ok(writes)
    }

    fn region(&mut self, region: &sir::Region) -> Result<LogicalRegion, String> {
        let source = match &region.source {
            sir::RegionSource::Domains => None,
            sir::RegionSource::Results(value) => Some(Box::new(self.expr(value)?)),
        };
        let merge = if let Some(merge) = &region.merge {
            Some(LogicalMerge {
                left: self.pattern(&merge.left),
                right: self.pattern(&merge.right),
                identity: self.expr(&merge.identity)?,
                body: self.block(&merge.body)?,
            })
        } else {
            None
        };
        Ok(LogicalRegion {
            id: region.id.0,
            mode: region.mode,
            binders: region
                .binders
                .iter()
                .map(|id| LocalValueId(*id as u32))
                .collect(),
            source,
            body: self.block(&region.body)?,
            merge,
            result: region
                .result
                .as_ref()
                .map(|ty| body_type(ty, self.shapes, self.elems, &self.structural))
                .transpose()?,
        })
    }

    fn owned_storage(
        &mut self,
        ty: &Type,
        origin: LocalStorageOrigin,
    ) -> Result<LocalStorageId, String> {
        let Type::Tensor(ty) = ty else {
            return Err("owned logical value is not tensor-shaped".into());
        };
        let id = LocalStorageId(self.storage.len() as u32);
        self.storage.push(LocalStorage {
            ty: ty.clone(),
            origin,
        });
        Ok(id)
    }

    fn call_result_storage(
        &mut self,
        ty: &Type,
        choice: ChoiceId,
        path: &mut Vec<u32>,
        output: &mut Vec<CallResultStorage>,
    ) -> Result<(), String> {
        match ty {
            Type::Tensor(_) => {
                let storage = self.owned_storage(
                    ty,
                    LocalStorageOrigin::CallResult {
                        choice,
                        path: path.clone(),
                    },
                )?;
                output.push(CallResultStorage {
                    path: path.clone(),
                    storage,
                });
            }
            Type::Tuple(items) => {
                for (ordinal, item) in items.iter().enumerate() {
                    path.push(ordinal as u32);
                    self.call_result_storage(item, choice, path, output)?;
                    path.pop();
                }
            }
            Type::Scalar(_) | Type::Index { .. } | Type::Void => {}
            Type::Range { .. } | Type::CapabilityValue { .. } => {
                return Err("a call cannot return a range or capability value".into());
            }
        }
        Ok(())
    }

    fn expr(&mut self, expr: &sir::Expr) -> Result<LogicalExpr, String> {
        let ty = body_type(&expr.ty, self.shapes, self.elems, &self.structural)?;
        let kind = match &expr.kind {
            SExpr::Int(v) => LogicalExprKind::Int(*v),
            SExpr::Float(v) => LogicalExprKind::Float(v.to_bits()),
            SExpr::Bool(v) => LogicalExprKind::Bool(*v),
            SExpr::Var(id) => {
                let var = &self.definition.body.vars[*id];
                match var.kind {
                    sir::VarKind::Param(port) => {
                        LogicalExprKind::Value(ValueRef::Input(port as u32))
                    }
                    _ => LogicalExprKind::Value(ValueRef::Local(LocalValueId(*id as u32))),
                }
            }
            SExpr::ShapeParam(name) => LogicalExprKind::Shape(
                self.shapes
                    .get(name)
                    .copied()
                    .map(Sym::constant)
                    .or_else(|| self.structural.params.get(name).cloned())
                    .unwrap_or_else(|| self.structural.substitute(&Sym::param(name), self.shapes)),
            ),
            SExpr::Tuple(items) => LogicalExprKind::Tuple(
                items
                    .iter()
                    .map(|item| self.expr(item))
                    .collect::<Result<_, _>>()?,
            ),
            SExpr::Range { lo, hi } => {
                LogicalExprKind::Range(Box::new(self.expr(lo)?), Box::new(self.expr(hi)?))
            }
            SExpr::Field { base, index } => {
                LogicalExprKind::Field(Box::new(self.expr(base)?), *index)
            }
            SExpr::TileAlloc => LogicalExprKind::Construct {
                storage: self.owned_storage(&ty, LocalStorageOrigin::Constructed)?,
            },
            SExpr::Filled { like, value } => {
                let like = self.expr(like)?;
                let storage = self.owned_storage(&ty, LocalStorageOrigin::Constructed)?;
                LogicalExprKind::Filled {
                    storage,
                    like: Box::new(like),
                    value: value.to_bits(),
                }
            }
            SExpr::Index { base, indices } => {
                let base = self.expr(base)?;
                let indices = indices
                    .iter()
                    .map(|index| self.index(index))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Type::Tensor(result) = &ty {
                    let mut result_axis = 0usize;
                    for (source_axis, index) in indices.iter().enumerate() {
                        if matches!(index, LogicalIndex::Point(_) | LogicalIndex::Coordinate(_)) {
                            continue;
                        }
                        if let LogicalIndex::Range { start, end } = index {
                            if let Some(shape) = result.shape.get(result_axis) {
                                let atoms = shape.atoms();
                                if let [Atom::Param(symbol)] = atoms.as_slice() {
                                    if shape == &Sym::param(symbol)
                                        && symbol.starts_with("@runtime.")
                                    {
                                        let scalar_ty = Type::Scalar(DType::I32);
                                        let start =
                                            start.as_deref().cloned().unwrap_or(LogicalExpr {
                                                ty: scalar_ty.clone(),
                                                kind: LogicalExprKind::Int(0),
                                                span: expr.span,
                                            });
                                        let end = end.as_deref().cloned().unwrap_or(LogicalExpr {
                                            ty: scalar_ty.clone(),
                                            kind: LogicalExprKind::Extent {
                                                base: Box::new(base.clone()),
                                                axis: source_axis,
                                            },
                                            span: expr.span,
                                        });
                                        let value = LogicalExpr {
                                            ty: scalar_ty,
                                            kind: LogicalExprKind::Binary {
                                                op: crate::syntax::ast::BinaryOp::Sub,
                                                lhs: Box::new(end),
                                                rhs: Box::new(start),
                                            },
                                            span: expr.span,
                                        };
                                        self.runtime_extents.entry(symbol.clone()).or_insert(value);
                                    }
                                }
                            }
                        }
                        result_axis += 1;
                    }
                }
                LogicalExprKind::Index {
                    base: Box::new(base),
                    indices,
                }
            }
            SExpr::Member { result, slices } => LogicalExprKind::Index {
                base: Box::new(self.expr(result)?),
                indices: slices
                    .iter()
                    .map(|slice| LogicalIndex::Slice(slice.0))
                    .collect(),
            },
            SExpr::Transpose(base) => {
                self.view_expr(base, &ty, |rank| LocalViewTransform::Transpose {
                    permutation: (0..rank as u32).rev().collect(),
                })?
            }
            SExpr::Reshape { base, .. } => {
                let source = body_type(&base.ty, self.shapes, self.elems, &self.structural)?;
                let Type::Tensor(source) = source else {
                    return Err("reshape source is not shaped".into());
                };
                self.view_expr(base, &ty, |_| LocalViewTransform::Reshape {
                    source_shape: source.shape,
                })?
            }
            SExpr::Load(source) => {
                let source = self.expr(source)?;
                let source_storage = self
                    .expr_storage(&source)
                    .ok_or_else(|| "logical snapshot source has no storage identity".to_string())?;
                let storage = self.owned_storage(
                    &ty,
                    LocalStorageOrigin::Snapshot {
                        source: source_storage,
                    },
                )?;
                LogicalExprKind::Snapshot {
                    storage,
                    source: Box::new(source),
                }
            }
            SExpr::Decode(source) => {
                let source = self.expr(source)?;
                let source_storage = self
                    .expr_storage(&source)
                    .ok_or_else(|| "logical decode source has no storage identity".to_string())?;
                let storage = self.owned_storage(
                    &ty,
                    LocalStorageOrigin::Snapshot {
                        source: source_storage,
                    },
                )?;
                LogicalExprKind::Decode {
                    storage,
                    source: Box::new(source),
                }
            }
            SExpr::Cast { dtype, expr } => LogicalExprKind::Cast {
                dtype: *dtype,
                expr: Box::new(self.expr(expr)?),
            },
            SExpr::Unary { op, expr } => LogicalExprKind::Unary {
                op: *op,
                expr: Box::new(self.expr(expr)?),
            },
            SExpr::Binary { op, lhs, rhs } => LogicalExprKind::Binary {
                op: *op,
                lhs: Box::new(self.expr(lhs)?),
                rhs: Box::new(self.expr(rhs)?),
            },
            SExpr::Math { op, args } => LogicalExprKind::Math {
                op: *op,
                args: args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<_, _>>()?,
            },
            SExpr::Select { cond, then, els } => LogicalExprKind::Select {
                cond: Box::new(self.expr(cond)?),
                then: Box::new(self.expr(then)?),
                els: Box::new(self.expr(els)?),
            },
            SExpr::Reduce {
                value,
                axis,
                op,
                unordered,
            } => LogicalExprKind::Reduce {
                value: Box::new(self.expr(value)?),
                axis: *axis,
                op: *op,
                unordered: *unordered,
            },
            SExpr::CoordOf(id) => LogicalExprKind::Coordinate(LocalValueId(*id as u32)),
            SExpr::ExtentOf { base, axis } => LogicalExprKind::Extent {
                base: Box::new(self.expr(base)?),
                axis: *axis,
            },
            SExpr::Call { call, args } => {
                let choice = *self
                    .calls
                    .get(call)
                    .ok_or_else(|| format!("call#{} has no nested logical choice", call.0))?;
                let mut results = Vec::new();
                self.call_result_storage(&ty, choice, &mut Vec::new(), &mut results)?;
                LogicalExprKind::Call {
                    choice,
                    args: args
                        .iter()
                        .map(|arg| self.expr(arg))
                        .collect::<Result<_, _>>()?,
                    results,
                }
            }
            SExpr::Region(region) => LogicalExprKind::Region(Box::new(self.region(region)?)),
            SExpr::Intrinsic { op, args } => LogicalExprKind::Intrinsic {
                operation: *op,
                args: args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<_, _>>()?,
            },
            SExpr::Accessor { base, name } => LogicalExprKind::Accessor {
                base: Box::new(self.expr(base)?),
                name: name.clone(),
            },
            SExpr::Geometry { base, axis, valid } => LogicalExprKind::Geometry {
                base: Box::new(self.expr(base)?),
                axis: *axis,
                valid: *valid,
            },
            SExpr::Atomic { op, place, value } => LogicalExprKind::Atomic {
                op: *op,
                place: Box::new(self.expr(place)?),
                value: Box::new(self.expr(value)?),
            },
        };
        let mut translated = LogicalExpr {
            ty: ty.clone(),
            kind,
            span: expr.span,
        };
        let owns_result = matches!(expr.ty, Ty::Tile(_))
            || matches!(&expr.kind, SExpr::Intrinsic { op, .. } if op.produces_owned_result());
        if owns_result && self.expr_storage(&translated).is_none() {
            let storage = self.owned_storage(&ty, LocalStorageOrigin::Constructed)?;
            translated = LogicalExpr {
                ty,
                kind: LogicalExprKind::Materialize {
                    storage,
                    value: Box::new(translated),
                },
                span: expr.span,
            };
        }
        Ok(translated)
    }

    fn view_expr(
        &mut self,
        base: &sir::Expr,
        ty: &Type,
        transform: impl FnOnce(usize) -> LocalViewTransform,
    ) -> Result<LogicalExprKind, String> {
        let base = self.expr(base)?;
        let storage = self
            .expr_storage(&base)
            .ok_or_else(|| "logical view source has no storage identity".to_string())?;
        let Type::Tensor(tensor) = ty else {
            return Err("logical view is not shaped".into());
        };
        let id = LocalViewId(self.views.len() as u32);
        self.views.push(LocalView {
            storage,
            shape: tensor.shape.clone(),
            elem: tensor.elem.clone(),
            access: Access::Exclusive,
            transform: transform(tensor.shape.len()),
        });
        Ok(LogicalExprKind::View {
            view: id,
            base: Box::new(base),
        })
    }

    fn index(&mut self, index: &sir::Index) -> Result<LogicalIndex, String> {
        Ok(match index {
            sir::Index::Point(value) => LogicalIndex::Point(Box::new(self.expr(value)?)),
            sir::Index::Coord(id) => LogicalIndex::Coordinate(LocalValueId(*id as u32)),
            sir::Index::Slice(slice) => LogicalIndex::Slice(slice.0),
            sir::Index::Range { start, end } => LogicalIndex::Range {
                start: start
                    .as_ref()
                    .map(|v| self.expr(v).map(Box::new))
                    .transpose()?,
                end: end
                    .as_ref()
                    .map(|v| self.expr(v).map(Box::new))
                    .transpose()?,
            },
        })
    }

    fn expr_storage(&self, expr: &LogicalExpr) -> Option<StorageRef> {
        fn call_result(expr: &LogicalExpr, mut path: Vec<u32>) -> Option<LocalStorageId> {
            match &expr.kind {
                LogicalExprKind::Call { results, .. } => results
                    .iter()
                    .find(|result| result.path == path)
                    .map(|result| result.storage),
                LogicalExprKind::Field(base, index) => {
                    path.insert(0, *index as u32);
                    call_result(base, path)
                }
                _ => None,
            }
        }
        if let Some(storage) = call_result(expr, Vec::new()) {
            return Some(StorageRef::Local(storage));
        }
        match expr.kind {
            LogicalExprKind::Construct { storage }
            | LogicalExprKind::Filled { storage, .. }
            | LogicalExprKind::Snapshot { storage, .. }
            | LogicalExprKind::Decode { storage, .. }
            | LogicalExprKind::Materialize { storage, .. } => Some(StorageRef::Local(storage)),
            LogicalExprKind::View { view, .. } => self
                .views
                .get(view.0 as usize)
                .map(|view| view.storage.clone()),
            LogicalExprKind::Value(ValueRef::Input(port)) if matches!(expr.ty, Type::Tensor(_)) => {
                Some(StorageRef::Input {
                    port,
                    path: Vec::new(),
                })
            }
            LogicalExprKind::Value(ValueRef::Result(port))
                if matches!(expr.ty, Type::Tensor(_)) =>
            {
                Some(StorageRef::Result {
                    port,
                    path: Vec::new(),
                })
            }
            LogicalExprKind::Value(ValueRef::Local(id)) => {
                self.var_storage.get(id.0 as usize).cloned().flatten()
            }
            LogicalExprKind::Field(ref base, index) => {
                self.expr_storage(base).map(|storage| match storage {
                    StorageRef::Input { port, mut path } => {
                        path.push(index as u32);
                        StorageRef::Input { port, path }
                    }
                    StorageRef::Result { port, mut path } => {
                        path.push(index as u32);
                        StorageRef::Result { port, path }
                    }
                    StorageRef::Local(storage) => StorageRef::Local(storage),
                })
            }
            LogicalExprKind::Index { ref base, .. }
            | LogicalExprKind::Accessor { ref base, .. }
            | LogicalExprKind::Geometry { ref base, .. } => self.expr_storage(base),
            _ => None,
        }
    }
}

fn field_type(ty: &Type, index: usize) -> Result<Type, String> {
    match ty {
        Type::Tuple(items) => items
            .get(index)
            .cloned()
            .ok_or_else(|| format!("tuple has no field {index}")),
        _ => Err(format!("field {index} of non-tuple logical value")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_variable_extents_are_owned_by_their_candidate_occurrence() {
        let first = StructuralSymbols {
            runtime_prefix: "@runtime.choice3.alternative0".into(),
            ..StructuralSymbols::default()
        };
        let second = StructuralSymbols {
            runtime_prefix: "@runtime.choice9.alternative0".into(),
            ..StructuralSymbols::default()
        };
        let authored = Sym::param("row#4").add(&Sym::constant(1));

        let first_extent = first.substitute(&authored, &BTreeMap::new());
        let second_extent = second.substitute(&authored, &BTreeMap::new());

        assert_eq!(
            first_extent,
            Sym::param("@runtime.choice3.alternative0.var4").add(&Sym::constant(1))
        );
        assert_eq!(
            second_extent,
            Sym::param("@runtime.choice9.alternative0.var4").add(&Sym::constant(1))
        );
        assert_ne!(first_extent, second_extent);
    }
}
