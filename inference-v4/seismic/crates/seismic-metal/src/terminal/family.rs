//! Retained local terminal implementations. Transfer alternatives share their
//! surrounding computation; loop widths stay numeric rather than expanding a
//! separate terminal program for every combination of widths.
use super::{Program, Site, Statement, transfer, traversal};
use magnitude_solver::model::{Arithmetic, Constraint, Domain, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Algebra, Symbolic, Value};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
pub(crate) enum Decision {
    Transfer(transfer::Choice),
    Traversal(traversal::Choice),
}
impl Decision {
    fn len(&self) -> usize {
        match self { Self::Transfer(choice) => choice.len(), Self::Traversal(choice) => choice.len() }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Definition {
    pub identity: String,
    pub decision: Decision,
}
#[derive(Clone, Debug)]
pub(crate) enum Region {
    Sequence(Vec<Region>),
    Statement(Site),
    Choice { decision: usize, alternatives: Vec<Region> },
    /// A source or implementation decision already owned by the enclosing
    /// model. It changes retained code, never introduces a runtime branch.
    Select { guard: Literal, yes: Box<Region>, no: Box<Region> },
    Guarded { guard: Literal, body: Box<Region> },
    Loop { header: Site, decision: Option<usize>, body: Box<Region> },
    Scope { header: Site, body: Box<Region>, alternative: Option<Box<Region>> },
}
pub(crate) struct Family {
    definitions: Vec<Definition>,
    launches: Vec<Region>,
}
#[derive(Clone)]
pub(crate) struct Parameter {
    pub ordinal: VarId,
    pub width: Value,
    /// Original finite loop count divided by the original selected width.
    pub complete: Option<Value>,
    pub tail: Option<Value>,
}
#[derive(Clone)]
pub(crate) struct Binding {
    name: String,
    family: Arc<Family>,
    pub parameters: Vec<Parameter>,
    selectors: BTreeMap<(VarId, i64), VarId>,
}
impl Family {
    pub(crate) fn new(program: &Program) -> Result<Arc<Self>, String> {
        let transfers = transfer::regions(program)?;
        let mut family = Self { definitions: Vec::new(), launches: Vec::new() };
        for (launch, (body, transfers)) in program.launches.iter().zip(&transfers).enumerate() {
            let region = family.region(body, 0, body.len(), launch,
                &format!("launch{launch}"), transfers)?;
            family.launches.push(region);
        }
        Ok(Arc::new(family))
    }
    /// Local stage producers can compose ordinary regions with their original
    /// solver predicates without selecting complete downstream implementations.
    pub(crate) fn from_regions(definitions: Vec<Definition>, launches: Vec<Region>) -> Arc<Self> {
        Arc::new(Self { definitions, launches })
    }
    /// Promote emitted source-selector scopes into compile-time choices. The
    /// placeholder declarations disappear together with their branch service.
    pub(crate) fn with_selectors(program: &Program, selectors: &BTreeMap<String, VarId>) -> Result<Arc<Self>, String> {
        fn selector(expression: &super::Expression, selectors: &BTreeMap<String, VarId>) -> Option<VarId> {
            match expression {
                super::Expression::Variable(name, _) => selectors.get(name).copied(),
                super::Expression::Cast(_, expression) => selector(expression, selectors),
                _ => None,
            }
        }
        fn promote(region: Region, selectors: &BTreeMap<String, VarId>) -> Region {
            match region {
                Region::Sequence(children) => Region::Sequence(children.into_iter().map(|child| promote(child, selectors)).collect()),
                Region::Statement(site) if matches!(&site.statement, Statement::Let {name,..} | Statement::Assign {name,..} if selectors.contains_key(name)) => Region::Sequence(Vec::new()),
                Region::Scope { header, body, alternative } => {
                    let body = Box::new(promote(*body, selectors));
                    let alternative = alternative.map(|body| Box::new(promote(*body, selectors)));
                    if let Statement::If(expression) = &header.statement {
                        if let Some(variable) = selector(expression, selectors) {
                            return Region::Select { guard: Literal::new(variable, 1), yes: body,
                                no: alternative.unwrap_or_else(|| Box::new(Region::Sequence(Vec::new()))) };
                        }
                    }
                    Region::Scope { header, body, alternative }
                },
                Region::Choice { decision, alternatives } => Region::Choice { decision, alternatives: alternatives.into_iter().map(|child| promote(child, selectors)).collect() },
                Region::Loop { header, decision, body } => Region::Loop { header, decision, body: Box::new(promote(*body, selectors)) },
                other => other,
            }
        }
        let family = Self::new(program)?;
        Ok(Arc::new(Self { definitions: family.definitions.clone(), launches: family.launches.iter().cloned().map(|region| promote(region, selectors)).collect() }))
    }
    pub(crate) fn definitions(&self) -> &[Definition] { &self.definitions }
    pub(crate) fn launches(&self) -> &[Region] { &self.launches }
    fn define(&mut self, identity: String, decision: Decision) -> usize {
        let index = self.definitions.len();
        self.definitions.push(Definition { identity, decision });
        index
    }
    fn region(&mut self, body: &[Site], start: usize, end: usize, launch: usize,
        context: &str, transfers: &[transfer::Region]) -> Result<Region, String> {
        let mut regions = Vec::new();
        let mut at = start;
        while at < end {
            if let Some(transfer) = transfers.iter().find(|region| region.range.start == at) {
                if transfer.range.end > end { return Err("terminal implementation crosses its lexical scope".into()); }
                let identity = format!("{context}.operation{:?}.site{at}.transfer", body[at].operation);
                let decision = self.define(identity.clone(), Decision::Transfer(transfer.choice.clone()));
                let mut alternatives = Vec::new();
                for (ordinal, alternative) in transfer.alternatives.iter().enumerate() {
                    alternatives.push(self.region(alternative, 0, alternative.len(), launch,
                        &format!("{identity}.alternative{ordinal}"), &[])?);
                }
                regions.push(Region::Choice { decision, alternatives });
                at = transfer.range.end;
                continue;
            }
            let site = &body[at];
            match &site.statement {
                Statement::For { .. } => {
                    let close = traversal::close(body, at)?;
                    if close >= end { return Err("terminal loop crosses retained region".into()); }
                    let decision = traversal::iterations(body, at)?.filter(|&count| count > 1).map(|iterations| {
                        self.define(format!("{context}.operation{:?}.site{at}.traversal", site.operation),
                            Decision::Traversal(traversal::Choice { launch, site: at, operation: site.operation, iterations }))
                    });
                    let inner = self.region(body, at + 1, close, launch, context, transfers)?;
                    regions.push(Region::Loop { header: site.clone(), decision, body: Box::new(inner) });
                    at = close + 1;
                }
                Statement::If(_) | Statement::Scope => {
                    let close = traversal::close(body, at)?;
                    if close >= end { return Err("terminal scope crosses retained region".into()); }
                    let mut alternative = None;
                    let mut depth = 0usize;
                    for (index, site) in body.iter().enumerate().take(close).skip(at + 1) {
                        match site.statement {
                            Statement::If(_) | Statement::For { .. } | Statement::Scope => depth += 1,
                            Statement::End => depth -= 1,
                            Statement::Else if depth == 0 => alternative = Some(index),
                            _ => {},
                        }
                    }
                    let inner = self.region(body, at + 1, alternative.unwrap_or(close), launch, context, transfers)?;
                    let alternative = alternative.map(|at| self.region(body, at + 1, close, launch, context, transfers).map(Box::new)).transpose()?;
                    regions.push(Region::Scope { header: site.clone(), body: Box::new(inner), alternative });
                    at = close + 1;
                }
                Statement::Else | Statement::End => return Err("unowned terminal scope delimiter".into()),
                _ => { regions.push(Region::Statement(site.clone())); at += 1; },
            }
        }
        Ok(Region::Sequence(regions))
    }
    pub(crate) fn append(self: &Arc<Self>, builder: &mut ModelBuilder, name: &str) -> Result<Binding, String> {
        let mut parameters = (0..self.definitions.len()).map(|_| None).collect::<Vec<_>>();
        let mut selectors = BTreeMap::new();
        fn collect(region: &Region, builder: &mut ModelBuilder, name: &str, selectors: &mut BTreeMap<(VarId, i64), VarId>) {
            match region {
                Region::Sequence(children) | Region::Choice { alternatives: children, .. } => for child in children { collect(child, builder, name, selectors); },
                Region::Select { guard, yes, no } => {
                    bind_selector(builder, name, *guard, selectors);
                    collect(yes, builder, name, selectors); collect(no, builder, name, selectors);
                },
                Region::Guarded { guard, body } => {
                    bind_selector(builder, name, *guard, selectors); collect(body, builder, name, selectors);
                },
                Region::Loop { body, .. } => collect(body, builder, name, selectors),
                Region::Scope { body, alternative, .. } => {
                    collect(body, builder, name, selectors);
                    if let Some(alternative) = alternative { collect(alternative, builder, name, selectors); }
                },
                Region::Statement(_) => {},
            }
        }
        // Bind a predicate before entering its local arms. Reusing one predicate
        // in several scopes must not make its defining equality conditional on
        // whichever scope happened to be encountered first.
        for launch in &self.launches { collect(launch, builder, name, &mut selectors); }
        fn append(region: &Region, family: &Family, builder: &mut ModelBuilder, name: &str,
            parameters: &mut [Option<Parameter>], selectors: &mut BTreeMap<(VarId, i64), VarId>) -> Result<(), String> {
            let mut parameter = |index: usize, builder: &mut ModelBuilder| -> Result<VarId, String> {
                if parameters[index].is_some() { return Err("terminal decision has multiple defining regions".into()); }
                let definition = &family.definitions[index];
                let maximum = i64::try_from(definition.decision.len().checked_sub(1).ok_or("empty terminal implementation domain")?)
                    .map_err(|_| "terminal implementation domain exceeds shared integers")?;
                let prefix = format!("{name}.{}", definition.identity);
                let domain = Domain::interval(0, maximum).map_err(|e| e.to_string())?;
                let ordinal = builder.local_variable(format!("{prefix}.ordinal"), domain.clone()).map_err(|e| e.to_string())?;
                let mut algebra = Symbolic::new(builder, &prefix);
                let one = algebra.constant(1).map_err(|e| e.to_string())?;
                let width = algebra.sum(Value::binding(ordinal, &domain).map_err(|e| e.to_string())?, one).map_err(|e| e.to_string())?;
                let (complete, tail) = if let Decision::Traversal(choice) = &definition.decision {
                    let count = algebra.constant(choice.iterations as u64).map_err(|e| e.to_string())?;
                    let quotient_domain = Domain::interval(1, maximum + 1).map_err(|e| e.to_string())?;
                    let remainder_domain = Domain::interval(0, maximum).map_err(|e| e.to_string())?;
                    let quotient = builder.local_variable(format!("{prefix}.complete"), quotient_domain.clone()).map_err(|e| e.to_string())?;
                    let remainder = builder.local_variable(format!("{prefix}.tail"), remainder_domain.clone()).map_err(|e| e.to_string())?;
                    builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem { numerator: count.id(), denominator: width.id(), quotient, remainder }));
                    let complete = Value::binding(quotient, &quotient_domain).map_err(|e| e.to_string())?;
                    let tail = Value::binding(remainder, &remainder_domain).map_err(|e| e.to_string())?;
                    (Some(complete), Some(tail))
                } else { (None, None) };
                parameters[index] = Some(Parameter { ordinal, width, complete, tail });
                Ok(ordinal)
            };
            match region {
                Region::Statement(_) => {},
                Region::Sequence(children) => for child in children { append(child, family, builder, name, parameters, selectors)?; },
                Region::Choice { decision, alternatives } => {
                    let variable = parameter(*decision, builder)?;
                    for (ordinal, alternative) in alternatives.iter().enumerate() {
                        builder.when(Literal::new(variable, ordinal as i64), |builder|
                            append(alternative, family, builder, name, parameters, selectors))?;
                    }
                },
                Region::Select { guard, yes, no } => {
                    let active = bind_selector(builder, name, *guard, selectors);
                    builder.when(Literal::new(active, 1), |builder| append(yes, family, builder, name, parameters, selectors))?;
                    builder.when(Literal::new(active, 0), |builder| append(no, family, builder, name, parameters, selectors))?;
                },
                Region::Guarded { guard, body } => {
                    let active = bind_selector(builder, name, *guard, selectors);
                    builder.when(Literal::new(active, 1), |builder| append(body, family, builder, name, parameters, selectors))?;
                },
                Region::Loop { decision, body, .. } => {
                    if let Some(decision) = decision { parameter(*decision, builder)?; }
                    append(body, family, builder, name, parameters, selectors)?;
                },
                Region::Scope { body, alternative, .. } => {
                    append(body, family, builder, name, parameters, selectors)?;
                    if let Some(alternative) = alternative { append(alternative, family, builder, name, parameters, selectors)?; }
                },
            }
            Ok(())
        }
        for launch in &self.launches { append(launch, self, builder, name, &mut parameters, &mut selectors)?; }
        Ok(Binding { name: name.into(), family: self.clone(), selectors, parameters: parameters.into_iter()
            .map(|binding| binding.ok_or_else(|| "terminal decision has no defining region".to_string()))
            .collect::<Result<Vec<_>, _>>()? })
    }
}
fn bind_selector(builder: &mut ModelBuilder, name: &str, guard: Literal, selectors: &mut BTreeMap<(VarId, i64), VarId>) -> VarId {
    if let Some(&active) = selectors.get(&(guard.variable, guard.value)) { return active; }
    let active = builder.variable(format!("{name}.selector{}.{}", guard.variable.0, guard.value), Domain::boolean());
    let expected = builder.variable(format!("{name}.selector.literal"), Domain::singleton(guard.value));
    builder.guarded_constraint(vec![Literal::new(active, 1)], Constraint::Equal { left: guard.variable, right: expected });
    builder.guarded_constraint(vec![Literal::new(active, 0)], Constraint::NotEqual { left: guard.variable, right: expected });
    selectors.insert((guard.variable, guard.value), active); active
}
impl Binding {
    pub(crate) fn selector(&self, guard: Literal) -> Result<VarId, String> {
        self.selectors.get(&(guard.variable, guard.value)).copied().ok_or_else(|| "retained external selector has no binding".into())
    }
    pub(crate) fn family(&self) -> &Arc<Family> { &self.family }
    pub(crate) fn instantiate(&self, values: &[i64]) -> Result<(Program, Vec<seismic_compiler::tuner::family::Decision>), String> {
        fn region(node: &Region, binding: &Binding, values: &[i64], decisions: &mut Vec<seismic_compiler::tuner::family::Decision>) -> Result<Vec<Site>, String> {
            let selected = |index: usize, decisions: &mut Vec<seismic_compiler::tuner::family::Decision>| -> Result<usize, String> {
                let definition = &binding.family.definitions[index];
                let value = *values.get(binding.parameters[index].ordinal.0).ok_or("missing terminal parameter")?;
                let ordinal = usize::try_from(value).map_err(|_| "negative terminal parameter")?;
                if ordinal >= definition.decision.len() { return Err("terminal parameter outside original domain".into()); }
                decisions.push(seismic_compiler::tuner::family::Decision { identity: format!("{}.{}", binding.name, definition.identity), value });
                Ok(ordinal)
            };
            match node {
                Region::Statement(site) => Ok(vec![site.clone()]),
                Region::Sequence(children) => {
                    let mut output = Vec::new();
                    for child in children { output.extend(region(child, binding, values, decisions)?); }
                    Ok(output)
                },
                Region::Choice { decision, alternatives } => region(&alternatives[selected(*decision, decisions)?], binding, values, decisions),
                Region::Select { guard, yes, no } => {
                    let selected = *values.get(guard.variable.0).ok_or("missing external terminal decision")? == guard.value;
                    region(if selected {yes} else {no}, binding, values, decisions)
                },
                Region::Guarded { guard, body } => {
                    if *values.get(guard.variable.0).ok_or("missing external terminal decision")? == guard.value {
                        region(body, binding, values, decisions)
                    } else { Ok(Vec::new()) }
                },
                Region::Loop { header, decision, body } => {
                    let inner = region(body, binding, values, decisions)?;
                    let mut output = vec![header.clone()]; output.extend(inner);
                    output.push(Site { operation: header.operation, statement: Statement::End });
                    if let Some(index) = decision {
                        let width = selected(*index, decisions)? + 1;
                        let Decision::Traversal(original) = &binding.family.definitions[*index].decision else { return Err("loop parameter has a nontraversal definition".into()); };
                        let mut choice = original.clone(); choice.site = 0;
                        traversal::apply(&mut output, choice.launch, &[traversal::Selection { choice, width }])?;
                    }
                    Ok(output)
                },
                Region::Scope { header, body, alternative } => {
                    let mut output = vec![header.clone()]; output.extend(region(body, binding, values, decisions)?);
                    if let Some(alternative) = alternative {
                        output.push(Site { operation: header.operation, statement: Statement::Else });
                        output.extend(region(alternative, binding, values, decisions)?);
                    }
                    output.push(Site { operation: header.operation, statement: Statement::End });
                    Ok(output)
                },
            }
        }
        let mut decisions = Vec::new();
        let launches = self.family.launches.iter().map(|launch| region(launch, self, values, &mut decisions)).collect::<Result<Vec<_>, _>>()?;
        Ok((Program { launches }, decisions))
    }
}
