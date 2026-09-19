use super::*;
use crate::{ast::AssignOp, lower, program::Program, types::{Elem, Ty}};
use std::collections::{HashMap, HashSet};
mod partition;
mod callbacks;
mod preparation;
mod panels;
mod fusion;
mod producers;

#[derive(Clone, Default)]
struct Context {
    shapes: HashMap<String, Sym>,
    elements: HashMap<String, Elem>,
    variables: HashMap<VarId, Expr>,
}

struct Builder<'a> {
    program: &'a Program,
    options: &'a lower::Options,
    family: ExecutionFamily,
    calls: Vec<String>,
    bounds: HashMap<Sym, (i64, IterationDomain)>,
    partitioning: HashSet<(String, String)>,
    packet_aligned: HashSet<VarId>,
}

pub(super) fn construct(input: lower::alternatives::Specialization<'_>) -> Result<ExecutionFamily, String> {
    let source = input.program.functions.iter().find(|f| f.name == input.entry)
        .ok_or_else(|| format!("no function `{}`", input.entry))?;
    if input.options.piece.is_some_and(|n| n <= 0) {
        return Err("stream piece capacity must be positive".into());
    }
    for shape in &source.shape_params {
        if !input.shapes.contains_key(shape) {
            return Err(format!("shape parameter `{shape}` of `{}` is not bound", input.entry));
        }
    }
    crate::program::validate_element_bindings(source, input.elements)?;
    let context = Context {
        shapes: input.shapes.iter().map(|(name, &value)| (name.clone(), Sym::constant(value))).collect(),
        elements: input.elements.clone(),
        variables: HashMap::new(),
    };
    let ty = |value: &Ty| lower::subst_elem_ty(&lower::subst_ty(value, &context.shapes), &context.elements);
    let template = LoweredIr {
        name: input.entry.into(), backend: input.backend.into(), ownership: input.options.ownership.clone(),
        alias_requirements: Vec::new(), params: source.params.iter().map(|(name, value)| (name.clone(), ty(value))).collect(),
        index_params: source.index_params.iter().map(|(name, value)| (name.clone(), context.symbol(value))).collect(),
        vars: source.vars.iter().map(|var| Var { ty: ty(&var.ty), ..var.clone() }).collect(),
        body: Vec::new(), shapes: input.shapes.clone(), selections: Vec::new(), decisions: Vec::new(),
    };
    template.ownership.validate(&template)?;
    let mut builder = Builder { program: input.program, options: input.options,
        family: empty(template), calls: vec![input.entry.into()], bounds: HashMap::new(), partitioning: HashSet::new(), packet_aligned: HashSet::new() };
    let occurrence = OccurrenceId { definition: input.entry.into(), topology: Vec::new() };
    builder.family.root = builder.block(&source.body, &context, &occurrence, &Guard::default())?;
    builder.finish();
    Ok(builder.family)
}

fn empty(template: LoweredIr) -> ExecutionFamily {
    ExecutionFamily { template, root: RegionId(0), regions: Vec::new(), decisions: Vec::new(),
        obligations: Vec::new(), requirements: Vec::new(), dependencies: Vec::new(), snapshots: Vec::new() }
}

pub(super) fn fixed(function: &LoweredIr) -> Result<ExecutionFamily, String> {
    crate::verify::lowered(function, crate::verify::Stage::Expanded)?;
    let mut template = function.clone();
    let mut body = std::mem::take(&mut template.body);
    crate::normalize::bind_values(&mut body, &mut template.vars);
    let program = Program { functions: Vec::new(), lowerings: Vec::new(), signatures: HashMap::new() };
    let options = lower::Options { piece: None, ownership: template.ownership.clone() };
    let occurrence = OccurrenceId { definition: function.name.clone(), topology: Vec::new() };
    let mut builder = Builder { program: &program, options: &options, family: empty(template), calls: Vec::new(), bounds: HashMap::new(), partitioning: HashSet::new(), packet_aligned: HashSet::new() };
    builder.family.root = builder.fixed_block(&body, &occurrence);
    builder.finish();
    Ok(builder.family)
}

impl Context {
    fn symbol(&self, value: &Sym) -> Sym { lower::subst_sym(value, &self.shapes, &HashMap::new()) }
    fn variable(&self, value: VarId) -> Result<VarId, String> {
        match self.variables.get(&value) {
            Some(Expr { kind: ExprKind::Var(id), .. }) => Ok(*id),
            Some(_) => Err("control binding must retain a variable identity".into()),
            None => Ok(value),
        }
    }
}

impl Builder<'_> {
    fn expression(&mut self, expression: &Expr, context: &Context) -> Result<Expr, String> {
        // Share the checked expression specialization, not any decision pass.
        // Callback call expressions carry typed arguments but are not expanded.
        if let ExprKind::Call { callee, shape_args, elem_args, args } = &expression.kind {
            return Ok(Expr { kind: ExprKind::Call { callee: callee.clone(),
                shape_args: shape_args.iter().map(|s| context.symbol(s)).collect(),
                elem_args: elem_args.iter().map(|e| lower::subst_elem(e, &context.elements)).collect(),
                args: args.iter().map(|arg| self.expression(arg, context)).collect::<Result<_, _>>()?,
            }, ty: lower::subst_elem_ty(&lower::subst_ty(&expression.ty, &context.shapes), &context.elements),
                sym: expression.sym.as_ref().map(|s| context.symbol(s)), span: expression.span });
        }
        // Structured reductions contain callback expressions, so specialize
        // their operands recursively before the scalar expression visitor.
        if let ExprKind::Builtin { name: Builtin::Reduce, args } = &expression.kind {
            if args.iter().any(|e| matches!(e.kind, ExprKind::Call { .. } | ExprKind::Tuple(_))) {
                let args = args.iter().map(|e| self.expression(e, context)).collect::<Result<_, _>>()?;
                return Ok(Expr { kind: ExprKind::Builtin { name: Builtin::Reduce, args },
                    ty: lower::subst_elem_ty(&lower::subst_ty(&expression.ty, &context.shapes), &context.elements),
                    sym: expression.sym.as_ref().map(|s| context.symbol(s)), span: expression.span });
            }
        }
        if let ExprKind::Tuple(elements) = &expression.kind {
            return Ok(Expr { kind: ExprKind::Tuple(elements.iter().map(|e| self.expression(e, context)).collect::<Result<_, _>>()?),
                ty: lower::subst_elem_ty(&lower::subst_ty(&expression.ty, &context.shapes), &context.elements),
                sym: expression.sym.as_ref().map(|s| context.symbol(s)), span: expression.span });
        }
        let mut forbidden = |_: &Decision| Err("expression specialization attempted an implementation decision".to_owned());
        let mut inliner = lower::Inliner {
            program: self.program, backend: &self.family.template.backend, select: &mut forbidden,
            selections: Vec::new(), counter: 0, opts: self.options.clone(), elements: context.elements.clone(),
            calls: lower::CallStage::Retain, piece_values: HashMap::new(), partitioning: HashSet::new(),
            domains: HashMap::new(), view_domains: HashMap::new(),
        };
        inliner.inline_expr(expression, &context.shapes, &context.variables, &mut self.family.template.vars, &mut HashMap::new())
    }

    fn obligation(&mut self, occurrence: &OccurrenceId, guard: &Guard, class: DecisionClass, reason: impl Into<String>) {
        let obligation = CoverageObligation { occurrence: occurrence.clone(), guard: guard.clone(), class, reason: reason.into() };
        if !self.family.obligations.contains(&obligation) { self.family.obligations.push(obligation); }
    }

    fn decision(&mut self, occurrence: &OccurrenceId, guard: &Guard, domain: Decision, slot: usize) -> Result<DecisionId, String> {
        if domain.alternatives.is_empty() { return Err(format!("empty retained decision domain: {:?}", domain.kind)); }
        let id = DecisionId { occurrence: occurrence.clone(), class: DecisionClass::of(&domain.kind), slot };
        if self.family.decisions.iter().any(|d| d.id == id) { return Err(format!("duplicate semantic decision {id:?}")); }
        let numeric = (domain.alternatives.numeric().is_some() || matches!(domain.kind, DecisionKind::Stream { .. })).then(|| NumericParameter {
            id: ParameterId(id.clone()), atom: Atom::Param(format!("family#numeric#{}", self.family.decisions.len())),
        });
        self.family.decisions.push(FamilyDecision { id: id.clone(), domain, guard: guard.clone(), numeric });
        Ok(id)
    }

    fn push(&mut self, occurrence: &OccurrenceId, guard: &Guard, kind: RegionKind, effects: Effects) -> RegionId {
        let id = RegionId(self.family.regions.len());
        self.family.regions.push(Region { occurrence: occurrence.clone(), guard: guard.clone(), effects, kind });
        id
    }

    fn sequence(&mut self, children: Vec<RegionId>, occurrence: &OccurrenceId, guard: &Guard) -> RegionId {
        let mut effects = Effects::default();
        for &child in &children { merge_effects(&mut effects, &self.family.regions[child.0].effects); }
        for (right_position, &right) in children.iter().enumerate() {
            for &left in &children[..right_position] {
                let a = &self.family.regions[left.0].effects;
                let b = &self.family.regions[right.0].effects;
                for &variable in a.writes.intersection(&b.reads).chain(a.reads.intersection(&b.writes)).chain(a.writes.intersection(&b.writes)) {
                    self.family.dependencies.push(Dependency { from: left, to: right, kind: EdgeKind::Value(variable) });
                }
                if a.tensor_effect || b.tensor_effect {
                    self.family.dependencies.push(Dependency { from: left, to: right, kind: EdgeKind::Effect });
                }
            }
        }
        let phases = children.iter().copied().filter(|child| matches!(self.family.regions[child.0].kind,
            RegionKind::Repeated { repetition: Repetition { order: RepeatOrder::Parallel, .. }, .. })).collect::<Vec<_>>();
        for pair in phases.windows(2) {
            self.family.dependencies.push(Dependency { from: pair[0], to: pair[1], kind: EdgeKind::Phase });
        }
        self.push(occurrence, guard, RegionKind::Sequence(children), effects)
    }

    fn block(&mut self, body: &[Stmt], context: &Context, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let enclosing_bounds = self.bounds.clone();
        let enclosing_packets = self.packet_aligned.clone();
        self.check_composition(body, context, occurrence, guard);
        let mut children = Vec::with_capacity(body.len());
        for (position, statement) in body.iter().enumerate() {
            let child = child_origin(occurrence, "statement", position, operation(statement));
            children.push(self.statement(statement, context, &child, guard)?);
        }
        let result = self.compose(children, occurrence, guard)?;
        self.bounds = enclosing_bounds;
        self.packet_aligned = enclosing_packets;
        Ok(result)
    }

    fn statement(&mut self, source: &Stmt, context: &Context, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut header = Stmt { id: None, span: source.span, kind: source.kind.clone() };
        match &source.kind {
            StmtKind::Expr(Expr { kind: ExprKind::Call { .. }, .. }) => {
                let StmtKind::Expr(call) = &source.kind else { unreachable!() };
                let call = self.expression(call, context)?;
                self.call(&call, occurrence, guard)
            }
            StmtKind::Parallel { vars, extents, body } => {
                let vars = vars.iter().map(|&v| context.variable(v)).collect::<Result<Vec<_>, _>>()?;
                let extents = extents.iter().map(|s| context.symbol(s)).collect::<Vec<_>>();
                header.kind = StmtKind::Parallel { vars: vars.clone(), extents: extents.clone(), body: Vec::new() };
                let body = self.block(body, context, occurrence, guard)?;
                self.repeated(header, body, RepeatOrder::Parallel, vars, extents, occurrence, guard)
            }
            StmtKind::Range { var, lo, hi, body } => {
                let var = context.variable(*var)?;
                let lo = context.symbol(lo); let hi = context.symbol(hi);
                let count = hi.sub(&lo);
                header.kind = StmtKind::Range { var, lo, hi, body: Vec::new() };
                let body = self.block(body, context, occurrence, guard)?;
                let repeated = self.repeated(header, body, RepeatOrder::Serial, vec![var], vec![count], occurrence, guard)?;
                self.matrix_panel(repeated, occurrence, guard)
            }
            StmtKind::Lanes { var, extent, width, body } => {
                let var = context.variable(*var)?; let extent = context.symbol(extent);
                header.kind = StmtKind::Lanes { var, extent: extent.clone(), width: *width, body: Vec::new() };
                let body = self.block(body, context, occurrence, guard)?;
                self.repeated(header, body, RepeatOrder::Parallel, vec![var], vec![extent], occurrence, guard)
            }
            StmtKind::Owned { vars, tile, body } => {
                let vars = vars.iter().map(|&v| context.variable(v)).collect::<Result<Vec<_>, _>>()?;
                let tile = self.expression(tile, context)?;
                let counts = tile.ty.shaped().map(|t| t.shape.clone()).unwrap_or_default();
                header.kind = StmtKind::Owned { vars: vars.clone(), tile, body: Vec::new() };
                let body = self.block(body, context, occurrence, guard)?;
                self.repeated(header, body, RepeatOrder::Parallel, vars, counts, occurrence, guard)
            }
            StmtKind::If { cond, then, els } => {
                header.kind = StmtKind::If { cond: self.expression(cond, context)?, then: Vec::new(), els: Vec::new() };
                let then_origin = child_origin(occurrence, "then", 0, "runtime condition".into());
                let else_origin = child_origin(occurrence, "else", 0, "runtime condition".into());
                let then = self.block(then, context, &then_origin, guard)?;
                let els = self.block(els, context, &else_origin, guard)?;
                let mut effects = effects(&header, self.family.template.vars.len());
                merge_effects(&mut effects, &self.family.regions[then.0].effects);
                merge_effects(&mut effects, &self.family.regions[els.0].effects);
                Ok(self.push(occurrence, guard, RegionKind::Conditional { header, then, els }, effects))
            }
            StmtKind::LoadLoop { domain, offset, vars, views, axes, piece, body, .. } => {
                let domain = IterationDomain { view: self.expression(&domain.view, context)?, axis: domain.axis };
                let extent = domain.view.ty.shaped().and_then(|s| s.shape.get(domain.axis)).ok_or("invalid stream domain")?.clone();
                let maximum = lower::view_axis_capacity(&domain.view, domain.axis)?.max(1);
                let piece = match piece { Atom::Param(_) => Atom::Param(format!("family#piece#{}", self.family.decisions.len())),
                    _ => return Err("stream piece must be a named atom".into()) };
                let capacity = self.decision(occurrence, guard, Decision {
                    kind: DecisionKind::Stream { piece: piece.clone(), extent: extent.clone(), maximum },
                    alternatives: match self.options.piece { Some(n) => vec![Alternative::StreamCapacity(n)].into(),
                        None => Alternatives::stream_capacities(maximum)? },
                }, 0)?;
                let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
                let capacity_sym = Sym::atom(parameter.atom.clone());
                let geometry = StreamGeometry { parameter, extent: extent.clone(), capacity: capacity_sym.clone(),
                    complete_pieces: extent.quot(&capacity_sym), tail_extent: extent.rem(&capacity_sym),
                    visits: extent.add(&capacity_sym).sub(&Sym::constant(1)).quot(&capacity_sym), piece: piece.clone() };
                let views = views.iter().map(|e| self.expression(e, context)).collect::<Result<Vec<_>, _>>()?;
                let vars = vars.iter().map(|&v| context.variable(v)).collect::<Result<Vec<_>, _>>()?;
                let mut inside = context.clone();
                let StmtKind::LoadLoop { piece: Atom::Param(original_piece), .. } = &source.kind else { unreachable!() };
                inside.shapes.insert(original_piece.clone(), Sym::atom(piece.clone()));
                for &variable in &vars {
                    self.family.template.vars[variable].ty = lower::subst_ty(&self.family.template.vars[variable].ty, &inside.shapes);
                }
                header.kind = StmtKind::LoadLoop { domain, offset: offset.map(|v| context.variable(v)).transpose()?,
                    modes: None, vars, views, axes: axes.clone(), piece, capacity: None, body: Vec::new() };
                let body = self.block(body, &inside, occurrence, guard)?;
                let mut summary = effects(&header, self.family.template.vars.len());
                merge_effects(&mut summary, &self.family.regions[body.0].effects);
                Ok(self.push(occurrence, guard, RegionKind::Stream { header, body, capacity, geometry }, summary))
            }
            StmtKind::Assign { target, op, value } => {
                header.kind = StmtKind::Assign { target: self.expression(target, context)?, op: *op, value: self.expression(value, context)? };
                self.leaf(header, occurrence, guard)
            }
            StmtKind::Expr(value) => {
                let value = self.expression(value, context)?;
                if let Some(operation) = crate::reduction::structured::Reduction::from_expr(&value) {
                    return self.coupled_reduction(operation, occurrence, guard);
                }
                header.kind = StmtKind::Expr(value);
                self.leaf(header, occurrence, guard)
            }
            StmtKind::Reduction(_) => Err("portable family input contains an already retained reduction; use from_lowered for a selected artifact".into()),
        }
    }

    fn repeated(&mut self, header: Stmt, body: RegionId, order: RepeatOrder, indices: Vec<VarId>, counts: Vec<Sym>,
        occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut summary = effects(&header, self.family.template.vars.len());
        let child = &self.family.regions[body.0].effects;
        let carried = if order == RepeatOrder::Serial { child.reads.intersection(&child.writes).copied()
            .filter(|v| !indices.contains(v)).collect() } else { BTreeSet::new() };
        merge_effects(&mut summary, child);
        for &variable in &carried {
            self.family.dependencies.push(Dependency { from: body, to: body, kind: EdgeKind::Recurrence(variable) });
        }
        Ok(self.push(occurrence, guard, RegionKind::Repeated { header, body,
            repetition: Repetition { order, indices, counts, carried } }, summary))
    }

    fn leaf(&mut self, source: Stmt, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let mut body = vec![source];
        crate::normalize::bind_values(&mut body, &mut self.family.template.vars);
        let mut children = Vec::new();
        for (position, statement) in body.into_iter().enumerate() {
            lower::decomposition::direct_bounds(&statement, &mut self.bounds);
            let aligned_binding = match &statement.kind {
                StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, op: AssignOp::Assign, value }
                    if crate::composition::packet_aligned(value, &self.family.template.vars, &self.packet_aligned) => Some(*variable),
                _ => None,
            };
            self.packet_aligned.retain(|&variable| !crate::effects::tile_mutated(&statement, variable));
            if let Some(variable) = aligned_binding { self.packet_aligned.insert(variable); }
            let origin = child_origin(occurrence, "value", position, operation(&statement));
            if let StmtKind::Assign { target, op: AssignOp::Assign, value } = &statement.kind {
                if let Some(contract) = crate::reduction::structured::primitive::contract(value) {
                    let (prefix, reduction, suffix) = crate::reduction::structured::primitive::expand(value, target, contract, &mut self.family.template.vars)?;
                    for (index, statement) in prefix.into_iter().enumerate() {
                        let setup = child_origin(&origin, "reduction.setup", index, operation(&statement));
                        let summary = effects(&statement, self.family.template.vars.len());
                        children.push(self.push(&setup, guard, RegionKind::Statement(statement), summary));
                    }
                    children.push(self.reduction(reduction, &origin, guard)?);
                    for (index, statement) in suffix.into_iter().enumerate() {
                        let finish = child_origin(&origin, "reduction.result", index, operation(&statement));
                        let summary = effects(&statement, self.family.template.vars.len());
                        children.push(self.push(&finish, guard, RegionKind::Statement(statement), summary));
                    }
                    continue;
                }
            }
            let has_reduce = match &statement.kind {
                StmtKind::Assign { value, .. } | StmtKind::Expr(value) => crate::effects::expressions(value,
                    &|e| matches!(e.kind, ExprKind::Builtin { name: Builtin::Reduce, .. })),
                _ => false,
            };
            if has_reduce {
                self.obligation(&origin, guard, DecisionClass::Reduction,
                    "source reduction requires retained tree/segment/branch and fold preparation transformations");
            }
            let packed = match &statement.kind {
                StmtKind::Assign { value, .. } | StmtKind::Expr(value) => crate::effects::expressions(value,
                    &|e| matches!(e.ty.shaped().map(|s| &s.elem), Some(Elem::Repr(_)))),
                _ => false,
            };
            if packed {
                self.obligation(&origin, guard, DecisionClass::Representation,
                    "packed source value requires retained representation/packet-decoding alternatives");
            }
            if let StmtKind::Assign { target, value, .. } = &statement.kind {
                let mut source = value;
                while let ExprKind::Index { base, .. } | ExprKind::Transpose(base) = &source.kind { source = base; }
                if let ExprKind::Var(variable) = source.kind {
                    if matches!(self.family.template.vars[variable].kind, VarKind::Local)
                        && matches!(target.ty, Ty::Tile(_)) && matches!(value.kind, ExprKind::Index { .. } | ExprKind::Transpose(_))
                        && target.ty.shaped().zip(source.ty.shaped()).is_some_and(|(target, source)|
                            target.shape.iter().fold(Sym::constant(1), |n, axis| n.mul(axis))
                                != source.shape.iter().fold(Sym::constant(1), |n, axis| n.mul(axis))) {
                        self.obligation(&origin, guard, DecisionClass::Producer,
                            "bounded producer projection needs retained reaching-view, recomputation and publication lifetime regions");
                    }
                }
            }
            let summary = effects(&statement, self.family.template.vars.len());
            let snapshot = match &statement.kind {
                StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, value, .. } => {
                    match &value.kind {
                        ExprKind::Builtin { name: Builtin::Load, args } => args.first().map(|source| (*variable, source.clone())),
                        ExprKind::Load { view, .. } => Some((*variable, (**view).clone())),
                        _ => None,
                    }
                },
                _ => None,
            };
            let id = self.push(&origin, guard, RegionKind::Statement(statement), summary);
            if let Some((variable, source)) = snapshot {
                self.family.snapshots.push(Snapshot { producer: id, variable, source, guard: guard.clone(), consumers: Vec::new() });
            }
            children.push(id);
        }
        Ok(self.sequence(children, occurrence, guard))
    }

    fn reduction(&mut self, mut operation: crate::reduction::structured::Reduction, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        use crate::reduction::structured::{StepOperand, StepState, Tree};
        let mut trees = operation.trees();
        let parameterized_extent = operation.extent().as_constant().is_none() && self.compile_parameter_expression(operation.extent());
        let maximum_extent = self.numeric_bound(operation.extent());
        if parameterized_extent && !operation.ordered && maximum_extent.is_some_and(|n| n > 0 && n < i64::MAX) {
            trees = vec![Tree::Ordered, Tree::Pairwise, Tree::Explicit, Tree::SeedThenPairwise];
        }
        let tree = self.decision(occurrence, guard, Decision { kind: DecisionKind::Reduction {
            merge: operation.merge_name().into(), extent: operation.extent().clone(), fields: operation.state.iter().map(|s| s.ty.clone()).collect(),
        }, alternatives: trees.iter().copied().map(Alternative::ReductionTree).collect::<Vec<_>>().into() }, 0)?;
        let mut segments = Vec::new();
        let mut preparation = Vec::new();
        let mut explicit = None;
        for (ordinal, &alternative) in trees.iter().enumerate() {
            let active = guard.with(tree.clone(), ordinal);
            if alternative != Tree::Ordered && parameterized_extent {
                self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: operation.extent().sub(&Sym::constant(1)) });
            }
            if alternative != Tree::Ordered && operation.step.is_some() {
                let extent = maximum_extent.ok_or("segmented source family needs a finite bound")?;
                let id = self.decision(occurrence, &active, Decision { kind: DecisionKind::ReductionSegments { extent },
                    alternatives: Alternatives::reduction_segments(extent)? }, ordinal)?;
                let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
                let capacity = Sym::atom(parameter.atom.clone());
                let extent = operation.extent().clone();
                self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: extent.sub(&capacity) });
                let geometry = StreamGeometry { parameter, extent: extent.clone(), capacity: capacity.clone(),
                    complete_pieces: extent.quot(&capacity), tail_extent: extent.rem(&capacity),
                    visits: extent.add(&capacity).sub(&Sym::constant(1)).quot(&capacity),
                    piece: Atom::Param(format!("family#segment#{}", self.family.decisions.len())),
                };
                preparation.push(self.fold_preparation(&operation, &geometry, ordinal, occurrence, &active)?);
                segments.push((active.clone(), id, geometry));
            }
            if alternative == Tree::Explicit {
                let maximum = maximum_extent.ok_or("explicit reduction tree needs a finite source extent bound")?;
                let leaves = if operation.step.is_some() {
                    let geometry = segments.iter().find(|(segment_guard, _, _)| segment_guard == &active)
                        .map(|(_, _, geometry)| geometry.visits.add(&Sym::constant(1)))
                        .ok_or("explicit segmented tree has no retained geometry")?;
                    geometry
                } else {
                    operation.extent().add(&Sym::constant(1))
                };
                let maximum_leaves = maximum.checked_add(1).ok_or("explicit reduction leaf count overflow")?;
                explicit = Some(self.explicit_tree(occurrence, &active, leaves, maximum_leaves,
                    operation.state.iter().map(|state| state.ty.clone()).collect())?);
            }
        }
        let mut operands = Vec::new();
        let mut state = None;
        if let Some(step) = &mut operation.step {
            if let Some(implementation) = &step.implementation {
                step.operands = vec![StepOperand::Private; implementation.right.len()];
                for (input, parameter) in implementation.right.iter().enumerate() {
                    if implementation.can_view_operand(input) {
                        let decision = self.decision(occurrence, guard, Decision { kind: DecisionKind::FoldOperand { input, ty: parameter.ty.clone() },
                            alternatives: vec![Alternative::StepOperand(StepOperand::Private), Alternative::StepOperand(StepOperand::View)].into() }, input)?;
                        operands.push((input, decision));
                    }
                }
                if implementation.can_retain_state() {
                    state = Some(self.decision(occurrence, guard, Decision { kind: DecisionKind::FoldState { fields: operation.state.iter().map(|s| s.ty.clone()).collect() },
                        alternatives: vec![Alternative::StepState(StepState::Separate), Alternative::StepState(StepState::Retained)].into() }, 0)?);
                }
            }
        }
        let mut dynamic_decomposition = None;
        let decomposition = if let Some(maximum) = maximum_extent.filter(|n| *n > 0) {
            let extent = operation.extent().clone();
            let ordinal = trees.iter().position(|tree| *tree == Tree::Ordered).ok_or("ordered source reduction is missing")?;
            let mut active = guard.with(tree.clone(), ordinal);
            if parameterized_extent { active = self.predicate(&active, extent.sub(&Sym::constant(1)))?; }
            let piece = Atom::Param(format!("family#piece#{}", self.family.decisions.len()));
            let decision = self.decision(occurrence, &active, Decision { kind: DecisionKind::Stream {
                piece: piece.clone(), extent: extent.clone(), maximum,
            }, alternatives: self.partition_capacities(&extent, maximum)? }, 0)?;
            let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
            let capacity = Sym::atom(parameter.atom.clone());
            self.constrain_partition_capacity(&extent, &capacity, &active)?;
            let geometry = StreamGeometry { parameter, extent: extent.clone(), capacity: capacity.clone(),
                complete_pieces: extent.quot(&capacity), tail_extent: extent.rem(&capacity),
                visits: extent.add(&capacity).sub(&Sym::constant(1)).quot(&capacity), piece };
            let index = self.family.template.vars.len();
            let atom = Atom::Param(format!("family#partition#{index}"));
            self.family.template.vars.push(Var { name: format!("partition_{index}"), ty: Ty::Scalar(crate::types::DType::I32),
                span: operation.span, kind: VarKind::Index(atom.clone()) });
            let mut piece = |start: Sym, count: Sym| -> Result<ReductionPiece, String> {
                let mut setup = Vec::new(); let mut part = operation.clone(); part.inputs.clear();
                for input in &operation.inputs {
                    let (binding, value) = lower::decomposition::slice_input(input, operation.axis, start.clone(), count.clone(), &mut self.family.template.vars)?;
                    setup.push(binding); part.inputs.push(value);
                }
                Ok(ReductionPiece { setup, operation: part })
            };
            let full = piece(Sym::atom(atom).mul(&capacity), capacity.clone())?;
            let tail = piece(geometry.tail_start(), geometry.tail_extent.clone())?;
            Some(ReductionDecomposition { guard: active, decision, index, whole_condition: capacity.sub(&extent), geometry, full, tail })
        } else {
            if operation.extent().as_constant().is_none() {
                if let Some((maximum, domain)) = self.bounds.get(operation.extent()).cloned().filter(|(bound, _)| *bound > 0) {
                    let ordinal = trees.iter().position(|tree| *tree == Tree::Ordered).ok_or("ordered source reduction is missing")?;
                    let active = guard.with(tree.clone(), ordinal);
                    let piece = Atom::Param(format!("family#piece#{}", self.family.decisions.len()));
                    let extent = operation.extent().clone();
                    let decision = self.decision(occurrence, &active, Decision { kind: DecisionKind::Stream {
                        piece: piece.clone(), extent: extent.clone(), maximum,
                    }, alternatives: match self.options.piece { Some(capacity) => vec![Alternative::StreamCapacity(capacity.min(maximum))].into(),
                        None => Alternatives::stream_capacities(maximum)? } }, 0)?;
                    let parameter = self.family.decisions.last().unwrap().numeric.clone().unwrap();
                    let capacity = Sym::atom(parameter.atom.clone());
                    let geometry = StreamGeometry { parameter, extent: extent.clone(), capacity: capacity.clone(),
                        complete_pieces: extent.quot(&capacity), tail_extent: extent.rem(&capacity),
                        visits: extent.add(&capacity).sub(&Sym::constant(1)).quot(&capacity), piece: piece.clone() };
                    let offset = self.family.template.vars.len();
                    self.family.template.vars.push(Var { name: format!("dynamic_start_{offset}"), ty: Ty::Scalar(crate::types::DType::I32),
                        span: operation.span, kind: VarKind::Index(Atom::Param(format!("family#start#{offset}"))) });
                    let mut part = operation.clone(); part.inputs.clear();
                    let mut bindings = Vec::new();
                    for source in &operation.inputs {
                        let mut shaped = source.ty.shaped().ok_or("dynamic reduction input has no shape")?.clone();
                        shaped.shape[operation.axis] = Sym::atom(piece.clone());
                        let ty = Ty::Tile(shaped);
                        let variable = self.family.template.vars.len();
                        self.family.template.vars.push(Var { name: format!("dynamic_piece_{variable}"), ty: ty.clone(), span: operation.span, kind: VarKind::Local });
                        bindings.push(variable);
                        part.inputs.push(Expr { kind: ExprKind::Var(variable), ty, sym: None, span: operation.span });
                    }
                    let header = Stmt { id: None, span: operation.span, kind: StmtKind::LoadLoop {
                        domain, offset: Some(offset), modes: None, vars: bindings, views: operation.inputs.clone(),
                        axes: vec![operation.axis; operation.inputs.len()], piece, capacity: None, body: Vec::new(),
                    } };
                    dynamic_decomposition = Some(DynamicReductionDecomposition { guard: active, decision, geometry, header, operation: part });
                } else {
                    self.obligation(occurrence, guard, DecisionClass::Stream,
                        "dynamic reduction decomposition has no retained captured-extent provenance and finite backing bound");
                }
            }
            None
        };
        let summary = effects(&Stmt { id: None, span: operation.span, kind: StmtKind::Reduction(Box::new(operation.clone())) }, self.family.template.vars.len());
        Ok(self.push(occurrence, guard, RegionKind::Reduction(Box::new(ReductionFamily { operation, tree, segments, operands, state, decomposition, dynamic_decomposition, callbacks: Vec::new(), preparation, explicit })), summary))
    }

    /// Retain one monotone postorder frontier per merge. For leaves L, merge i
    /// emits the exclusive leaf endpoint f_i. The arithmetic requirements below
    /// make the sequence a compact bijection with ordered binary trees:
    /// i+2 <= f_i <= L, f_i >= f_(i-1), and the last active f_i == L.
    fn explicit_tree(&mut self, occurrence: &OccurrenceId, guard: &Guard, leaves: Sym, maximum_leaves: i64,
        fields: Vec<Ty>) -> Result<ExplicitReductionFamily, String> {
        if maximum_leaves < 2 { return Err("explicit reduction tree needs at least two leaves".into()); }
        let mut merges = Vec::new();
        let mut previous = Sym::constant(1);
        for merge in 0..usize::try_from(maximum_leaves - 1).map_err(|_| "explicit reduction merge count overflow")? {
            let merge_i = i64::try_from(merge).map_err(|_| "explicit reduction merge index overflow")?;
            let active = self.predicate(guard, leaves.sub(&Sym::constant(merge_i + 2)))?;
            let decision = self.decision(occurrence, &active, Decision {
                kind: DecisionKind::ReductionFrontier { merge, leaves: leaves.clone(), maximum: maximum_leaves, fields: fields.clone() },
                alternatives: Alternatives::reduction_frontiers(merge_i + 2, maximum_leaves)?,
            }, merge)?;
            let frontier = self.family.decisions.last().and_then(|entry| entry.numeric.as_ref())
                .map(|parameter| Sym::atom(parameter.atom.clone()))
                .ok_or("explicit frontier is not numeric")?;
            self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: leaves.clone().sub(&frontier) });
            self.family.requirements.push(Requirement { guard: active.clone(), nonnegative: frontier.clone().sub(&previous) });
            // Exactly one frontier is final for a concrete leaf count. The
            // conjunction of this guard and active means leaves == merge+2.
            let final_guard = self.predicate(&active, Sym::constant(merge_i + 2).sub(&leaves))?;
            self.family.requirements.push(Requirement { guard: final_guard, nonnegative: frontier.clone().sub(&leaves) });
            merges.push(ExplicitReductionMerge { guard: active, frontier: decision });
            previous = frontier;
        }
        Ok(ExplicitReductionFamily { guard: guard.clone(), leaves, merges })
    }

    fn call(&mut self, call: &Expr, occurrence: &OccurrenceId, guard: &Guard) -> Result<RegionId, String> {
        let ExprKind::Call { callee, shape_args, elem_args, args } = &call.kind else { unreachable!() };
        let function = self.program.functions.iter().find(|f| &f.name == callee)
            .ok_or_else(|| format!("undefined retained callee `{callee}`"))?.clone();
        if self.calls.contains(callee) {
            return Err(format!("recursive execution family definition `{callee}`"));
        }
        if shape_args.len() != function.shape_params.len() || elem_args.len() != function.elem_params.len() || args.len() != function.params.len() {
            return Err(format!("retained call `{callee}` has inconsistent checked arguments"));
        }
        let context = Context { shapes: function.shape_params.iter().cloned().zip(shape_args.iter().cloned()).collect(),
            elements: function.elem_params.iter().cloned().zip(elem_args.iter().cloned()).collect(), variables: HashMap::new() };
        if function.is_construct {
            if let Some(partition) = self.partition_call(call, &function, occurrence, guard)? { return Ok(partition); }
        }
        let mut common = Vec::new();
        let mut arguments = args.clone();
        // Evaluate scalar arguments at their original call occurrence, before
        // selecting a body. All body alternatives share these snapshots.
        for (index, argument) in arguments.iter_mut().enumerate() {
            if let Ty::Scalar(dtype) = argument.ty {
                if matches!(function.vars[index].kind, VarKind::Index(_)) { continue; }
                let variable = self.family.template.vars.len();
                self.family.template.vars.push(Var { name: format!("call_{}_{}", callee, variable), ty: argument.ty.clone(), span: call.span, kind: VarKind::Local });
                let target = Expr { kind: ExprKind::Var(variable), ty: argument.ty.clone(), sym: None, span: call.span };
                let value = Expr { kind: ExprKind::Cast { dtype, expr: Box::new(argument.clone()) }, ty: argument.ty.clone(), sym: None, span: call.span };
                let origin = child_origin(occurrence, "argument", index, function.params[index].0.clone());
                common.push(self.leaf(Stmt { id: None, span: call.span, kind: StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value } }, &origin, guard)?);
                *argument = target;
            }
        }
        self.calls.push(callee.clone());
        if function.is_construct {
            let blocks = self.program.lowerings.iter().filter(|l| l.construct == *callee && l.backend == self.family.template.backend).cloned().collect::<Vec<_>>();
            let mut bodies = Vec::new();
            let mut portable = false;
            for (index, block) in blocks.into_iter().enumerate() {
                if block.body.is_empty() && block.residual.is_empty() && block.elem_bindings.is_empty() { portable = true; continue; }
                if !block.elem_bindings.iter().all(|(name, elem)| context.elements.get(name) == Some(elem)) { continue; }
                let residual = block.residual.iter().map(|r| context.symbol(r)).collect::<Vec<_>>();
                if residual.iter().any(|r| r.as_constant().is_some_and(|n| n < 0)) { continue; }
                bodies.push((Choice::Block(index), block.vars, block.body, residual));
            }
            if portable { bodies.push((Choice::Portable, function.vars.clone(), function.body.clone(), Vec::new())); }
            let decision = self.decision(occurrence, guard, Decision { kind: DecisionKind::Construct {
                name: callee.clone(), shape_args: shape_args.clone(), element_args: elem_args.clone(),
            }, alternatives: bodies.iter().map(|(choice, ..)| Alternative::Body(choice.clone())).collect::<Vec<_>>().into() }, 0)?;
            let mut arms = Vec::with_capacity(bodies.len());
            let mut summary = Effects::default();
            for (ordinal, (choice, vars, body, residual)) in bodies.into_iter().enumerate() {
                let arm_guard = guard.with(decision.clone(), ordinal);
                let mut origin = child_origin(occurrence, "body", ordinal, format!("{callee}:{choice:?}"));
                origin.definition = callee.clone();
                for residual in residual {
                    if residual.as_constant().is_none() {
                        self.family.requirements.push(Requirement { guard: arm_guard.clone(), nonnegative: residual.clone() });
                        if !self.compile_parameter_expression(&residual) {
                            self.obligation(&origin, &arm_guard, DecisionClass::Construct,
                                "body residual needs universal validity over runtime repetition extents");
                        }
                    }
                }
                let context = self.bind_call(&function, &vars, &arguments, &context, &origin)?;
                let arm = self.block(&body, &context, &origin, &arm_guard)?;
                merge_effects(&mut summary, &self.family.regions[arm.0].effects);
                arms.push(arm);
            }
            common.push(self.push(occurrence, guard, RegionKind::Choice { decision, arms }, summary));
        } else {
            let mut origin = child_origin(occurrence, "helper", 0, callee.clone());
            origin.definition = callee.clone();
            let context = self.bind_call(&function, &function.vars, &arguments, &context, &origin)?;
            common.push(self.block(&function.body, &context, &origin, guard)?);
        }
        self.calls.pop();
        Ok(self.sequence(common, occurrence, guard))
    }

    fn bind_call(&mut self, function: &Function, variables: &[Var], args: &[Expr], outer: &Context, _occurrence: &OccurrenceId) -> Result<Context, String> {
        let mut context = outer.clone();
        // Rename index atoms before specializing any local type that uses one.
        for (id, variable) in variables.iter().enumerate() {
            if let VarKind::Index(Atom::Param(name)) = &variable.kind {
                if id < function.params.len() {
                    context.shapes.insert(name.clone(), args[id].sym.clone().ok_or("bounded call argument has no symbolic value")?);
                } else {
                    context.shapes.insert(name.clone(), Sym::param(&format!("family#index#{}#{id}", self.family.template.vars.len())));
                }
            }
        }
        for (id, variable) in variables.iter().enumerate() {
            if id < function.params.len() { context.variables.insert(id, args[id].clone()); continue; }
            let ty = lower::subst_elem_ty(&lower::subst_ty(&variable.ty, &context.shapes), &context.elements);
            let global = self.family.template.vars.len();
            let kind = match &variable.kind {
                VarKind::Index(Atom::Param(name)) => {
                    let atom = context.shapes[name].atoms().into_iter().next().ok_or("missing retained index atom")?;
                    VarKind::Index(atom)
                },
                VarKind::Index(_) => return Err("index binding must be a named atom".into()),
                VarKind::Local => VarKind::Local,
                VarKind::Param(index) => { context.variables.insert(id, args[*index].clone()); continue; },
            };
            let sym = match &kind { VarKind::Index(atom) => Some(Sym::atom(atom.clone())), _ => None };
            self.family.template.vars.push(Var { name: format!("{}_{}", variable.name, global), ty: ty.clone(), span: variable.span, kind });
            context.variables.insert(id, Expr { kind: ExprKind::Var(global), ty, sym, span: variable.span });
        }
        Ok(context)
    }

    fn check_composition(&mut self, body: &[Stmt], context: &Context, occurrence: &OccurrenceId, guard: &Guard) {
        let mut reductions = 0;
        for statement in body {
            match &statement.kind {
                StmtKind::Parallel { extents, body, .. } => {
                    // Grouping widens construct outputs along a parallel axis.
                    // Empty/singleton domains have no nonidentity width, and
                    // ordinary function calls have no grouped-output contract.
                    // Other candidate regions still retain their obligation.
                    let wider = extents.iter().any(|extent| context.symbol(extent).as_constant().is_none_or(|extent| extent > 1));
                    let constructs = body.iter().filter_map(|s| match &s.kind {
                        StmtKind::Expr(Expr { kind: ExprKind::Call { callee, .. }, .. }) => Some(callee),
                        _ => None,
                    }).collect::<Vec<_>>();
                    if wider && !constructs.is_empty() && constructs.iter().all(|callee|
                        self.program.functions.iter().any(|function| function.is_construct && &function.name == *callee)) {
                        self.obligation(occurrence, guard, DecisionClass::OutputGroup,
                            "independent construct outputs require retained grouping, remainder and epilogue transformations");
                    }
                },
                StmtKind::Reduction(_) => reductions += 1,
                _ => {},
            }
        }
        for (count, class) in [(reductions, DecisionClass::ReductionFusion)] {
            if count > 1 { self.obligation(occurrence, guard, class,
                "sibling regions require conditional fusion compatibility and effect constraints"); }
        }
        if !self.options.ownership.intermediates.is_empty() {
            self.obligation(occurrence, guard, DecisionClass::Intermediate,
                "invocation-owned publications require retained producer projection and storage lifetimes");
        }
    }

    fn fixed_block(&mut self, body: &[Stmt], occurrence: &OccurrenceId) -> RegionId {
        let guard = Guard::default();
        let children = body.iter().enumerate().map(|(index, statement)| {
            let origin = child_origin(occurrence, "fixed", index, operation(statement));
            // A diagnostic artifact is already closed; its exact typed body is
            // retained, without reintroducing source optimization freedom.
            self.push(&origin, &guard, RegionKind::Statement(statement.clone()), effects(statement, self.family.template.vars.len()))
        }).collect();
        self.sequence(children, occurrence, &guard)
    }

    fn finish(&mut self) {
        for snapshot in &mut self.family.snapshots {
            snapshot.consumers = self.family.regions.iter().enumerate().filter_map(|(index, region)| {
                (index != snapshot.producer.0 && matches!(region.kind, RegionKind::Statement(_))
                    && region.effects.reads.contains(&snapshot.variable)).then_some(RegionId(index))
            }).collect();
        }
    }
}

fn merge_effects(target: &mut Effects, source: &Effects) {
    target.reads.extend(&source.reads); target.writes.extend(&source.writes); target.tensor_effect |= source.tensor_effect;
}
fn effects(statement: &Stmt, variables: usize) -> Effects {
    let mut writes = HashSet::new(); crate::rewrite::writes(statement, &mut writes);
    Effects { reads: (0..variables).filter(|&v| crate::effects::uses(statement, v)).collect(),
        writes: writes.into_iter().collect(), tensor_effect: crate::effects::tensor_effect(statement) }
}
fn child_origin(parent: &OccurrenceId, role: &str, sibling: usize, operation: String) -> OccurrenceId {
    let mut result = parent.clone(); result.topology.push(Origin { role: role.into(), sibling, operation }); result
}
fn operation(statement: &Stmt) -> String {
    // Full typed topology is part of source identity. Spans do not determine
    // identity, so moving the same definition between source files is harmless.
    fn expression(e: &Expr) -> String { format!("{:?}", crate::normalize::value_identity(e)) }
    match &statement.kind {
        StmtKind::Assign { target, op, value } => format!("assign:{op:?}:{}:{}", expression(target), expression(value)),
        StmtKind::Expr(value) => expression(value),
        StmtKind::Parallel { vars, extents, .. } => format!("parallel:{vars:?}:{extents:?}"),
        StmtKind::Range { var, lo, hi, .. } => format!("range:{var}:{lo}:{hi}"),
        StmtKind::Lanes { var, extent, width, .. } => format!("lanes:{var}:{extent}:{width}"),
        StmtKind::Owned { vars, tile, .. } => format!("owned:{vars:?}:{}", expression(tile)),
        StmtKind::LoadLoop { domain, views, axes, .. } => format!("stream:{}:{}:{axes:?}:{:?}", expression(&domain.view), domain.axis, views.iter().map(expression).collect::<Vec<_>>()),
        StmtKind::If { cond, .. } => format!("if:{}", expression(cond)),
        StmtKind::Reduction(reduction) => format!("reduction:{}:{}:{:?}", reduction.merge_name(), reduction.extent(), reduction.ordered),
    }
}
