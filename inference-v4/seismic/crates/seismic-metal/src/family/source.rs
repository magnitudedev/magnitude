//! Typed source regions and their original compile-time bindings. Branches are
//! retained locally, so independent source choices never multiply whole kernels.
use magnitude_solver::model::{ModelBuilder, VarId};
use seismic_accounting::algebra::Value;
use seismic_compiler::tuner::source;
use seismic_lang::{family::{Guard, RegionId, RegionKind}, ir::*, lowered_ir::LoweredIr,
    sym::{Atom, Sym}, types::{DType, Ty}};
use std::collections::{BTreeMap, HashMap};

/// Deterministic specialization of the retained backend body. Operation and
/// variable identities survive selection so storage and terminal bindings keep
/// referring to the original definitions accounted by the shared model.
pub(crate) struct Selection<'a> {
    pub parameters: HashMap<String, Sym>,
    predicates: BTreeMap<seismic_lang::ir::VarId, bool>,
    layout: &'a super::layout::Family,
    values: &'a [i64],
}
impl<'a> Selection<'a> {
    pub(crate) fn new(function: &LoweredIr, parameters: &BTreeMap<String, Value>,
        layout: &'a super::layout::Family, values: &'a [i64]) -> Result<Self, String> {
        let parameters = parameters.iter().map(|(name, original)| {
            let value = *values.get(original.id().0).ok_or("missing original Metal source operand")?;
            let unsigned = u64::try_from(value).map_err(|_| "negative Metal source operand")?;
            if unsigned < original.bounds().0 || unsigned > original.bounds().1 { return Err("Metal source operand is outside its original domain".into()); }
            Ok((name.clone(), Sym::constant(value)))
        }).collect::<Result<_, String>>()?;
        let mut predicates = BTreeMap::new();
        for (variable, definition) in function.vars.iter().enumerate() {
            let Some(original) = layout.source_predicates.get(&crate::msl::variable_symbol(definition, variable)) else { continue; };
            let value = match values.get(original.0) {
                Some(0) => false,
                Some(1) => true,
                _ => return Err("missing or invalid original Metal compiler predicate".into()),
            };
            predicates.insert(variable, value);
        }
        Ok(Self { parameters, predicates, layout, values })
    }
    pub(crate) fn predicate(&self, symbol: &str) -> Result<bool, String> {
        let original = self.layout.source_predicates.get(symbol).ok_or("selected source predicate has no original definition")?;
        match self.values.get(original.0) {
            Some(0) => Ok(false), Some(1) => Ok(true),
            _ => Err("missing or invalid selected source predicate".into()),
        }
    }
    pub(crate) fn phase_active(&self, phase: usize) -> Result<bool, String> {
        self.layout.phase_predicates.get(phase).and_then(Option::as_deref).map_or(Ok(true), |symbol| self.predicate(symbol))
    }
    pub(crate) fn body(&self, body: &[Stmt]) -> Result<Vec<Stmt>, String> {
        let mut selected = Vec::new();
        for statement in body {
            if matches!(&statement.kind, StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, .. }
                if self.predicates.contains_key(variable)) { continue; }
            if let StmtKind::If { cond: Expr { kind: ExprKind::Var(variable), .. }, then, els } = &statement.kind {
                if let Some(&active) = self.predicates.get(variable) {
                    selected.extend(self.body(if active { then } else { els })?);
                    continue;
                }
            }
            let mut statement = statement.clone();
            let operation = statement.id;
            match &mut statement.kind {
                StmtKind::Assign { target, value, .. } => {
                    if let (Some(operation), ExprKind::Var(variable)) = (operation, &target.kind) {
                        if let Some(choice) = self.layout.load_sites.get(&(operation, *variable)) {
                            let mode = *choice.selected(self.values)?;
                            match &mut value.kind {
                                ExprKind::Load { mode: selected, .. } => *selected = mode,
                                ExprKind::Builtin { name: Builtin::Load, args } if args.len() == 1 => {
                                    value.kind = ExprKind::Load { view: Box::new(args[0].clone()), mode };
                                },
                                _ => return Err("selected load no longer refers to its original operation".into()),
                            }
                        }
                    }
                    self.expression(target); self.expression(value);
                },
                StmtKind::Expr(value) => self.expression(value),
                StmtKind::If { cond, then, els } => {
                    self.expression(cond); *then = self.body(then)?; *els = self.body(els)?;
                },
                StmtKind::Parallel { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => *body = self.body(body)?,
                StmtKind::Owned { tile, body, .. } => { self.expression(tile); *body = self.body(body)?; },
                StmtKind::LoadLoop { domain, vars, views, modes, body, .. } => {
                    self.expression(&mut domain.view);
                    for view in views { self.expression(view); }
                    let operation = operation.ok_or("selected stream lacks its original operation identity")?;
                    let original = modes.as_ref().ok_or("retained stream has no typed load realization")?;
                    if original.len() != vars.len() { return Err("retained stream load arity differs from its bindings".into()); }
                    *modes = Some(vars.iter().zip(original).map(|(&variable, &mode)| {
                        self.layout.load_sites.get(&(operation, variable)).map_or(Ok(mode), |choice| choice.selected(self.values).copied())
                    }).collect::<Result<Vec<_>, _>>()?);
                    *body = self.body(body)?;
                },
                StmtKind::Reduction(_) => return Err("retained Metal function still contains unexpanded fold ownership".into()),
            }
            selected.push(statement);
        }
        Ok(selected)
    }
    fn expression(&self, value: &mut Expr) {
        match &mut value.kind {
            ExprKind::Var(variable) => if let Some(&selected) = self.predicates.get(variable) {
                value.kind = ExprKind::Bool(selected); value.sym = None;
            },
            ExprKind::Index { base, indices } => {
                self.expression(base);
                for index in indices {
                    match index {
                        Index::Point(point) => self.expression(point),
                        Index::Slice { start, end } => for bound in start.iter_mut().chain(end) { self.expression(bound); },
                    }
                }
            },
            ExprKind::Load { view: base, .. } | ExprKind::Transpose(base) | ExprKind::Accessor { base, .. }
            | ExprKind::Lanes { base, .. } | ExprKind::Unary { expr: base, .. } | ExprKind::Cast { expr: base, .. } => self.expression(base),
            ExprKind::Binary { lhs, rhs, .. } => { self.expression(lhs); self.expression(rhs); },
            ExprKind::Builtin { args, .. } | ExprKind::Call { args, .. } | ExprKind::Intrinsic { args, .. } | ExprKind::Tuple(args) => {
                for argument in args { self.expression(argument); }
            },
            ExprKind::Int(_) | ExprKind::ShapeParam(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::TileAlloc { .. } => {},
        }
    }
}

#[derive(Clone)]
pub(crate) struct Predicate {
    /// Typed source-local variable. Emission recognizes it through this binding,
    /// never by parsing a variable name or selecting its placeholder initializer.
    pub variable: seismic_lang::ir::VarId,
    pub presence: VarId,
}
#[derive(Clone)]
pub(crate) struct Stream {
    pub operation: OperationId,
    pub capacity: Value,
    pub expression: Sym,
}
#[derive(Clone)]
pub(crate) struct Template {
    pub function: LoweredIr,
    pub predicates: Vec<Predicate>,
    predicate_variables: HashMap<VarId, seismic_lang::ir::VarId>,
    pub parameters: BTreeMap<String, Value>,
    /// Exact derived equations over these original parameter names. Independent
    /// source choices have bindings only, never manufactured definitions.
    pub numeric_definitions: BTreeMap<String, Sym>,
    pub streams: Vec<Stream>,
    /// Presence of each retained source-order launch, including alternatives
    /// with different phase counts. Absent phases retain storage identity but
    /// contribute no dispatch, submission or operation occurrence.
    pub phase_presence: Vec<Option<VarId>>,
}
impl Template {
    pub(crate) fn construct(builder: &mut ModelBuilder, source: &mut source::Binding) -> Result<Self, String> {
        let mut template = Self { function: source.family().template().clone(), predicates: Vec::new(), predicate_variables: HashMap::new(),
            parameters: source.parameters().clone(), numeric_definitions: BTreeMap::new(),
            streams: Vec::new(), phase_presence: Vec::new() };
        let root = source.family().root();
        template.function.body = template.region(builder, source, root)?;
        template.retain_phases(builder)?;
        let mut operation = 0;
        identify(&mut template.function.body, &mut operation, &mut template.streams);
        Ok(template)
    }
    fn retain_phases(&mut self, builder: &mut ModelBuilder) -> Result<(), String> {
        let selectors = self.predicates.iter().map(|predicate| (predicate.variable, predicate.presence)).collect::<BTreeMap<_, _>>();
        let mut phases = Vec::new();
        collect_phases(std::mem::take(&mut self.function.body), &[], &selectors, &mut phases)?;
        if phases.is_empty() {
            phases.push((Stmt { id: None, span: Default::default(), kind: StmtKind::Parallel { vars: Vec::new(), extents: Vec::new(), body: Vec::new() } }, Vec::new()));
        }
        for (mut phase, guards) in phases {
            let StmtKind::Parallel { vars, body, .. } = &mut phase.kind else { unreachable!() };
            strip_initializers(body, &selectors);
            let presence = phase_presence(builder, body, vars.is_empty(), &guards, &selectors);
            let mut declarations = Vec::new();
            for &variable in selectors.keys() {
                let span = self.function.vars[variable].span;
                declarations.push(Stmt { id: None, span, kind: StmtKind::Assign {
                    target: Expr { kind: ExprKind::Var(variable), ty: Ty::Scalar(DType::Bool), sym: None, span },
                    op: seismic_lang::ast::AssignOp::Assign,
                    value: Expr { kind: ExprKind::Bool(false), ty: Ty::Scalar(DType::Bool), sym: None, span },
                } });
            }
            declarations.append(body); *body = declarations;
            self.function.body.push(phase); self.phase_presence.push(presence);
        }
        Ok(())
    }
    fn predicate(&mut self, builder: &mut ModelBuilder, source: &mut source::Binding, guard: &Guard,
        then: Vec<Stmt>, els: Vec<Stmt>) -> Result<Vec<Stmt>, String> {
        let Some(presence) = source.presence(builder, guard)? else { return Ok(then); };
        Ok(self.predicate_body(presence, then, els))
    }
    fn predicate_body(&mut self, presence: VarId, then: Vec<Stmt>, els: Vec<Stmt>) -> Vec<Stmt> {
        // One original activation has one immutable source binding. Reusing it
        // preserves branch facts across visits and preparation alternatives.
        if let Some(&variable) = self.predicate_variables.get(&presence) {
            let span = then.first().or(els.first()).map(|statement| statement.span).unwrap_or_default();
            let cond = Expr { kind: ExprKind::Var(variable), ty: Ty::Scalar(DType::Bool), sym: None, span };
            return vec![Stmt { id: None, span, kind: StmtKind::If { cond, then, els } }];
        }
        let (body, predicate) = predicate_region(&mut self.function.vars, presence, then, els);
        self.predicate_variables.insert(presence, predicate.variable);
        self.predicates.push(predicate); body
    }
    fn region(&mut self, builder: &mut ModelBuilder, source: &mut source::Binding, id: RegionId) -> Result<Vec<Stmt>, String> {
        self.region_under(builder, source, id, &Guard::default())
    }
    fn region_under(&mut self, builder: &mut ModelBuilder, source: &mut source::Binding, id: RegionId, parent: &Guard) -> Result<Vec<Stmt>, String> {
        let region = source.family().regions().get(id.0).ok_or("unknown retained Metal source region")?.clone();
        let guard = region.guard.clone();
        let body = match region.kind {
            RegionKind::Sequence(children) => {
                let mut body = Vec::new();
                for child in children { body.extend(self.region_under(builder, source, child, &guard)?); }
                Ok(body)
            },
            RegionKind::Statement(statement) => Ok(vec![statement]),
            RegionKind::Choice { decision, arms } => {
                if arms.is_empty() { return Err("empty retained source choice".into()); }
                let mut body = Vec::new();
                // An original choice executes exactly one arm. Retain that
                // control-flow fact as an if/else chain so lifetime analysis
                // cannot combine a borrowed binding with a sibling's writes.
                for (ordinal, arm) in arms.into_iter().enumerate().rev() {
                    let mut arm_guard = guard.clone(); arm_guard.choices.push((decision.clone(), ordinal));
                    let then = self.region_under(builder, source, arm, &arm_guard)?;
                    body = self.predicate(builder, source, &arm_guard, then, body)?;
                }
                Ok(body)
            },
            RegionKind::Repeated { mut header, body, .. } => {
                let selected = self.region_under(builder, source, body, &guard)?;
                match &mut header.kind {
                    StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. }
                    | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } => *body = selected,
                    _ => return Err("invalid retained source repetition header".into()),
                }
                Ok(vec![header])
            },
            RegionKind::Replicated { index, count, body } => {
                let variable = self.function.vars.get(index).ok_or("retained copy index is absent")?.clone();
                let VarKind::Index(atom) = variable.kind else { return Err("retained copy index has no symbolic identity".into()); };
                let original = self.region_under(builder, source, body, &guard)?;
                let maximum = count.eval_interval(&|name| self.parameters.get(name).and_then(|value| {
                    let (minimum, maximum) = value.bounds(); Some((i64::try_from(minimum).ok()?, i64::try_from(maximum).ok()?))
                })).map(|(_, maximum)| maximum).ok_or("retained static replication has no finite original count bound")?;
                if maximum < 0 { return Err("retained static replication has a negative count bound".into()); }
                let mut expressions = seismic_compiler::tuner::expressions::Expressions::new(self.parameters.clone());
                let mut body = Vec::new();
                let active = source.presence(builder, &guard)?;
                for ordinal in 0..maximum {
                    let mut append = |builder: &mut ModelBuilder| expressions.predicate(builder, "metal.source.copy", &count.sub(&Sym::constant(ordinal + 1)));
                    let presence = match active {
                        Some(active) => builder.when(magnitude_solver::model::Literal::new(active, 1), append),
                        None => append(builder),
                    }.map_err(|error| error.to_string())?;
                    let occurrence = seismic_lang::widen::parameterized::replicate(&original, index, &atom, ordinal, variable.span);
                    body.extend(self.predicate_body(presence, occurrence, Vec::new()));
                }
                Ok(body)
            },
            RegionKind::Conditional { mut header, then, els } => {
                let then_guard = source.family().regions()[then.0].guard.clone();
                let else_guard = source.family().regions()[els.0].guard.clone();
                let yes = self.region_under(builder, source, then, &then_guard)?;
                let no = self.region_under(builder, source, els, &else_guard)?;
                let StmtKind::If { then, els, .. } = &mut header.kind else { return Err("invalid retained conditional header".into()); };
                *then = yes.clone(); *els = no.clone();
                let both = if else_guard != guard {
                    self.predicate(builder, source, &else_guard, vec![header], yes)?
                } else { vec![header] };
                if then_guard != guard {
                    let no = if else_guard != guard { self.predicate(builder, source, &else_guard, no, Vec::new())? } else { no };
                    self.predicate(builder, source, &then_guard, both, no)
                } else { Ok(both) }
            },
            RegionKind::Stream { mut header, body, geometry, .. } => {
                let selected = self.region_under(builder, source, body, &guard)?;
                let Atom::Param(parameter) = &geometry.parameter.atom else { return Err("stream capacity has no named parameter".into()); };
                let capacity = *self.parameters.get(parameter).ok_or("stream capacity missing from shared source model")?;
                // The marker is local to retained construction and replaced by
                // one fresh operation identity after the complete union exists.
                let operation = OperationId(usize::MAX - self.streams.len());
                header.id = Some(operation);
                let StmtKind::LoadLoop { body, .. } = &mut header.kind else { return Err("invalid retained stream header".into()); };
                *body = selected;
                self.streams.push(Stream { operation, capacity, expression: geometry.capacity });
                Ok(vec![header])
            },
            RegionKind::Reduction(_) => Err("retained Metal reduction requires expanded typed source regions".into()),
        }?;
        if &guard == parent { Ok(body) } else { self.predicate(builder, source, &guard, body, Vec::new()) }
    }
}

/// Serial source alternatives can share one retained phase. That phase exists
/// exactly when one of its compiler-guarded bodies exists, even when none of
/// those alternatives contains an explicit parallel header.
fn phase_presence(builder: &mut ModelBuilder, body: &[Stmt], serial: bool, enclosing: &[magnitude_solver::model::Literal],
    selectors: &BTreeMap<seismic_lang::ir::VarId, VarId>) -> Option<VarId> {
    use magnitude_solver::model::{Constraint, Domain, LinearTerm, Literal};
    fn paths(body: &[Stmt], enclosing: &[Literal], selectors: &BTreeMap<seismic_lang::ir::VarId, VarId>, output: &mut Vec<Vec<Literal>>) {
        for statement in body {
            if let StmtKind::If { cond: Expr { kind: ExprKind::Var(variable), .. }, then, els } = &statement.kind {
                if let Some(&predicate) = selectors.get(variable) {
                    let mut guard = enclosing.to_vec(); guard.push(Literal::new(predicate, 1));
                    paths(then, &guard, selectors, output);
                    *guard.last_mut().unwrap() = Literal::new(predicate, 0);
                    paths(els, &guard, selectors, output);
                    continue;
                }
            }
            output.push(enclosing.to_vec());
        }
    }
    let mut alternatives = Vec::new();
    if serial { paths(body, enclosing, selectors, &mut alternatives); }
    else { alternatives.push(enclosing.to_vec()); }
    if alternatives.iter().any(Vec::is_empty) { return None; }
    for guard in &mut alternatives { guard.sort_by_key(|literal| (literal.variable, literal.value)); guard.dedup(); }
    alternatives.sort_by(|left, right| left.iter().map(|literal| (literal.variable, literal.value))
        .cmp(right.iter().map(|literal| (literal.variable, literal.value))));
    alternatives.dedup();
    let mut negations = BTreeMap::new();
    let mut arms = Vec::new();
    for guard in alternatives {
        let inputs = guard.into_iter().map(|guard| {
            if guard.value == 1 { guard.variable } else { *negations.entry(guard.variable).or_insert_with(|| {
                let inverse = builder.variable("metal.source.phase.absent", Domain::boolean());
                builder.constraint(Constraint::NotEqual { left: inverse, right: guard.variable }); inverse
            }) }
        }).collect::<Vec<_>>();
        let active = builder.variable("metal.source.phase.body", Domain::boolean());
        builder.constraint(Constraint::BoolAnd { output: active, inputs });
        arms.push(active);
    }
    if arms.len() == 1 { return arms.pop(); }
    let active = builder.variable("metal.source.phase.active", Domain::boolean());
    let mut terms = vec![LinearTerm::new(active, 1)];
    for arm in arms {
        builder.constraint(Constraint::Implies { premise: Literal::new(arm, 1), consequence: Literal::new(active, 1) });
        terms.push(LinearTerm::new(arm, -1));
    }
    builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
    Some(active)
}

fn contains_parallel(body: &[Stmt]) -> bool {
    body.iter().any(|statement| match &statement.kind {
        StmtKind::Parallel { .. } => true,
        StmtKind::If { then, els, .. } => contains_parallel(then) || contains_parallel(els),
        _ => false,
    })
}
fn collect_phases(body: Vec<Stmt>, guards: &[magnitude_solver::model::Literal], selectors: &BTreeMap<seismic_lang::ir::VarId, VarId>,
    phases: &mut Vec<(Stmt, Vec<magnitude_solver::model::Literal>)>) -> Result<(), String> {
    let mut serial = Vec::new();
    let flush = |serial: &mut Vec<Stmt>, phases: &mut Vec<(Stmt, Vec<magnitude_solver::model::Literal>)>| {
        if serial.is_empty() { return; }
        let span = serial[0].span;
        phases.push((Stmt { id: None, span, kind: StmtKind::Parallel { vars: Vec::new(), extents: Vec::new(), body: std::mem::take(serial) } }, guards.to_vec()));
    };
    for statement in body {
        match statement.kind {
            StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, .. } if selectors.contains_key(&variable) => {},
            StmtKind::Parallel { .. } => { flush(&mut serial, phases); phases.push((statement, guards.to_vec())); },
            StmtKind::If { ref cond, ref then, ref els } if contains_parallel(then) || contains_parallel(els) => {
                let ExprKind::Var(variable) = cond.kind else { return Err("runtime branch containing parallel phases requires an explicit retained control publication".into()); };
                let presence = *selectors.get(&variable).ok_or("runtime branch containing parallel phases has no retained compiler selector")?;
                flush(&mut serial, phases);
                let StmtKind::If { then, els, .. } = statement.kind else { unreachable!() };
                let mut active = guards.to_vec(); active.push(magnitude_solver::model::Literal::new(presence, 1));
                collect_phases(then, &active, selectors, phases)?;
                *active.last_mut().unwrap() = magnitude_solver::model::Literal::new(presence, 0);
                collect_phases(els, &active, selectors, phases)?;
            },
            _ => serial.push(statement),
        }
    }
    flush(&mut serial, phases); Ok(())
}
fn strip_initializers(body: &mut Vec<Stmt>, selectors: &BTreeMap<seismic_lang::ir::VarId, VarId>) {
    body.retain(|statement| !matches!(&statement.kind,
        StmtKind::Assign { target: Expr { kind: ExprKind::Var(variable), .. }, .. } if selectors.contains_key(variable)));
    for statement in body {
        match &mut statement.kind {
            StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } => strip_initializers(body, selectors),
            StmtKind::If { then, els, .. } => { strip_initializers(then, selectors); strip_initializers(els, selectors); },
            _ => {},
        }
    }
}

pub(crate) fn predicate_region(vars: &mut Vec<Var>, presence: VarId, then: Vec<Stmt>, els: Vec<Stmt>) -> (Vec<Stmt>, Predicate) {
    let span = then.first().or(els.first()).map(|statement| statement.span).unwrap_or_default();
    let variable = vars.len();
    vars.push(Var { name: format!("source_choice_{variable}"), ty: Ty::Scalar(DType::Bool), kind: VarKind::Local, span });
    let value = Expr { kind: ExprKind::Var(variable), ty: Ty::Scalar(DType::Bool), sym: None, span };
    let body = vec![Stmt { id: None, span, kind: StmtKind::Assign {
        target: value.clone(), op: seismic_lang::ast::AssignOp::Assign,
        value: Expr { kind: ExprKind::Bool(false), ty: Ty::Scalar(DType::Bool), sym: None, span },
    } }, Stmt { id: None, span, kind: StmtKind::If { cond: value, then, els } }];
    (body, Predicate { variable, presence })
}

fn identify(body: &mut [Stmt], next: &mut usize, streams: &mut [Stream]) {
    for statement in body {
        let previous = statement.id;
        let id = OperationId(*next); *next += 1;
        statement.id = Some(id);
        if let Some(stream) = streams.iter_mut().find(|stream| Some(stream.operation) == previous) { stream.operation = id; }
        match &mut statement.kind {
            StmtKind::Parallel { body, .. } | StmtKind::Owned { body, .. } | StmtKind::Range { body, .. }
            | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } => identify(body, next, streams),
            StmtKind::If { then, els, .. } => { identify(then, next, streams); identify(els, next, streams); },
            StmtKind::Reduction(_) | StmtKind::Assign { .. } | StmtKind::Expr(_) => {},
        }
    }
}
