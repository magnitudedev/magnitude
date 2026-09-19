//! Interpretation of retained terminal regions. Compile-time guards annotate
//! optional operations; runtime predicates retain their ordinary branch service.
use super::*;
use crate::terminal::family::{Binding, Decision, Region};
use magnitude_solver::model::{Arithmetic, Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Algebra, Symbolic, Value};

#[derive(Clone)]
struct State {
    env: BTreeMap<String, Values>,
    ranges: BTreeMap<String, ranges::Ranges>,
    affine: BTreeMap<String, affine::Values>,
    memory: Memory,
    active: u32,
    alive: u32,
    returned: Values,
    returned_facts: Facts,
    symbols: super::parameters::Symbols,
}
impl State {
    fn alternatives(branches: Vec<(Vec<Literal>, Self)>, choice: Option<(VarId, usize)>) -> Result<Self, DerivationError> {
        let mut joined = branches.first().ok_or("terminal choice has no retained branch")?.1.clone();
        for (_, branch) in branches.iter().skip(1) { joined.join(branch)?; }
        for (guards, branch) in &branches {
            joined.memory.join_alternative(&branch.memory, guards);
            joined.symbols.join_alternative(&branch.symbols, &branch.env, &branch.ranges, &branch.affine,
                &joined.env, &joined.ranges, &joined.affine, guards);
        }
        if let Some((variable, cardinality)) = choice {
            joined.memory.collapse_choice(variable, cardinality);
            joined.symbols.collapse_choice(variable, cardinality);
        }
        Ok(joined)
    }
    fn capture(state: &Derivation<'_>, symbols: &super::parameters::Symbols) -> Self {
        Self { env: state.env.clone(), ranges: state.ranges.clone(), affine: state.affine.clone(),
            memory: state.memory.clone(), active: state.active, alive: state.alive,
            returned: state.returned, returned_facts: state.returned_facts.clone(), symbols: symbols.clone() }
    }
    fn restore(&self, state: &mut Derivation<'_>, symbols: &mut super::parameters::Symbols) {
        *symbols = self.symbols.clone();
        state.env = self.env.clone(); state.ranges = self.ranges.clone(); state.affine = self.affine.clone();
        state.memory = self.memory.clone(); state.active = self.active; state.alive = self.alive;
        state.returned = self.returned; state.returned_facts = self.returned_facts.clone(); state.facts.clear(); state.assumptions.clear();
    }
    fn join(&mut self, other: &Self) -> Result<(), DerivationError> {
        if self.active != other.active || self.alive != other.alive {
            return Err(DerivationError::Unsupported("terminal alternatives have different runtime participation".into()));
        }
        self.env.retain(|name, values| {
            let Some(right) = other.env.get(name) else { return false; };
            for lane in 0..32 { if values[lane] != right[lane] { values[lane] = None; } }
            true
        });
        self.ranges.retain(|name, values| {
            let Some(right) = other.ranges.get(name) else { return false; };
            for lane in 0..32 { values[lane] = values[lane].zip(right[lane]).map(|((a,b),(c,d))| (a.min(c), b.max(d))); }
            true
        });
        self.affine.retain(|name, values| {
            let Some(right) = other.affine.get(name) else { return false; };
            for lane in 0..32 { if values[lane] != right[lane] { values[lane] = None; } }
            true
        });
        self.memory.join(&other.memory);
        self.symbols.join(&other.symbols);
        for lane in 0..32 {
            if self.returned[lane] != other.returned[lane] { self.returned[lane] = None; }
            self.returned_facts.ranges[lane] = self.returned_facts.ranges[lane].zip(other.returned_facts.ranges[lane])
                .map(|((a,b),(c,d))| (a.min(c), b.max(d)));
            if self.returned_facts.affine[lane] != other.returned_facts.affine[lane] { self.returned_facts.affine[lane] = None; }
        }
        Ok(())
    }
}

/// The same static implementation parameter is visited by many logical items.
/// Retain its arithmetic once under its defining activation, while operation
/// occurrences and their memory dependencies remain distinct.
#[derive(Default)]
pub(super) struct Predicates {
    above: BTreeMap<(Vec<(VarId, i64)>, VarId, u64), VarId>,
    equal: BTreeMap<(Vec<(VarId, i64)>, VarId, i64), VarId>,
    all: BTreeMap<(Vec<(VarId, i64)>, Vec<VarId>), VarId>,
    modulo: BTreeMap<(Vec<(VarId, i64)>, u64, VarId), VarId>,
    products: BTreeMap<(Vec<(VarId, i64)>, VarId, VarId), Value>,
}
pub(super) struct Recorder<'a> {
    pub builder: &'a mut ModelBuilder,
    pub binding: &'a Binding,
    pub guards: Vec<Literal>,
    pub presence: &'a mut Vec<Vec<Literal>>,
    pub predicates: &'a mut Predicates,
    pub symbols: &'a mut super::parameters::Symbols,
}
impl Recorder<'_> {
    fn count(state: &Derivation<'_>) -> usize {
        match &state.sink { Sink::Schedule { model, .. } => model.operations.len(), _ => unreachable!() }
    }
    fn capture<T>(&mut self, state: &mut Derivation<'_>, action: impl FnOnce(&mut Derivation<'_>) -> Result<T, DerivationError>) -> Result<T, DerivationError> {
        let result = action(state);
        self.presence.resize(Self::count(state), self.guards.clone());
        if let Sink::Schedule { model, .. } = &mut state.sink {
            for reason in model.unmapped.drain(..) {
                self.builder.obligation(self.guards.clone(), magnitude_solver::model::ObligationKind::Analysis, reason);
            }
        }
        result
    }
    fn join(&mut self, state: &mut Derivation<'_>, predecessors: Vec<usize>) -> Result<(), DerivationError> {
        let Sink::Schedule { model, .. } = &mut state.sink else { unreachable!() };
        if model.operations.len() >= state.limits.operations { return Err(DerivationError::Exhausted(DerivationLimit::Operations(state.limits.operations))); }
        let index = model.operations.len();
        model.operations.push(Operation { name: format!("{} region join {index}", state.scope), predecessors,
            start_predecessors: Vec::new(), latency: 0, reservations: Vec::new() });
        self.presence.push(self.guards.clone()); state.last = Some(index); Ok(())
    }
    fn optional(&mut self, state: &mut Derivation<'_>, guard: Literal,
        action: impl FnOnce(&mut Recorder<'_>, &mut Derivation<'_>) -> Result<(), DerivationError>) -> Result<(), DerivationError> {
        if let Some(known) = self.guards.iter().find(|known| known.variable == guard.variable) {
            return if known.value == guard.value { action(self, state) } else { Ok(()) };
        }
        let before = state.last;
        let mut guards = self.guards.clone(); guards.push(guard);
        let binding = self.binding; let presence = &mut *self.presence;
        let predicates = &mut *self.predicates; let symbols = &mut *self.symbols;
        self.builder.when(guard, |builder|
            action(&mut Recorder { builder, binding, guards, presence, predicates, symbols }, state))?;
        let after = state.last;
        if after != before { self.join(state, before.into_iter().chain(after).collect())?; }
        Ok(())
    }
    fn primitive(&mut self, state: &mut Derivation<'_>, primitive: Primitive) -> Result<(), DerivationError> {
        self.capture(state, |state| state.issue(primitive, u64::from(state.active.count_ones())).map(|_| ()))
    }
    fn expression(&mut self, state: &mut Derivation<'_>, expression: &Expression) -> Result<Values, DerivationError> {
        state.assumptions.clear();
        self.symbols.annotate(state, expression);
        self.capture(state, |state| state.expr(expression))?
            .map_err(|reason| DerivationError::Unsupported(format!("{}: {}: {reason}", state.scope, expression.render())))
    }
    fn test(&mut self, state: &mut Derivation<'_>) -> Result<(), DerivationError> {
        self.primitive(state, Primitive::Binary { operation: BinaryOp::Lt, ty: Type::I32 })?;
        self.primitive(state, Primitive::Branch)
    }
    fn context(&self) -> Vec<(VarId, i64)> {
        let mut guards = self.guards.iter().map(|literal| (literal.variable, literal.value)).collect::<Vec<_>>();
        guards.sort_unstable(); guards.dedup(); guards
    }
    fn boolean(&mut self, label: &str) -> VarId {
        self.builder.variable(format!("metal.terminal.trace.{}.{}", self.presence.len(), label), Domain::boolean())
    }
    fn above(&mut self, value: Value, threshold: u64) -> VarId {
        let key = (self.context(), value.id(), threshold);
        if let Some(&result) = self.predicates.above.get(&key) { return result; }
        let result = self.boolean("above");
        self.builder.guarded_constraint(vec![Literal::new(result, 1)], Constraint::LinearLe { terms: vec![LinearTerm::new(value.id(), -1)], rhs: -i128::from(threshold) - 1 });
        self.builder.guarded_constraint(vec![Literal::new(result, 0)], Constraint::LinearLe { terms: vec![LinearTerm::new(value.id(), 1)], rhs: i128::from(threshold) });
        self.predicates.above.insert(key, result); result
    }
    fn equal(&mut self, value: VarId, constant: i64) -> VarId {
        let key = (self.context(), value, constant);
        if let Some(&result) = self.predicates.equal.get(&key) { return result; }
        let result = self.boolean("equal");
        let expected = self.builder.variable("metal.terminal.trace.literal", Domain::singleton(constant));
        self.builder.guarded_constraint(vec![Literal::new(result, 1)], Constraint::Equal { left: value, right: expected });
        self.builder.guarded_constraint(vec![Literal::new(result, 0)], Constraint::NotEqual { left: value, right: expected });
        self.predicates.equal.insert(key, result); result
    }
    fn all(&mut self, mut inputs: Vec<VarId>) -> VarId {
        inputs.sort_unstable(); inputs.dedup();
        if inputs.len() == 1 { return inputs[0]; }
        let key = (self.context(), inputs.clone());
        if let Some(&output) = self.predicates.all.get(&key) { return output; }
        let output = self.boolean("all"); self.builder.constraint(Constraint::BoolAnd { output, inputs });
        self.predicates.all.insert(key, output); output
    }
    fn product(&mut self, left: Value, right: Value) -> Result<Value, DerivationError> {
        let key = (self.context(), left.id().min(right.id()), left.id().max(right.id()));
        if let Some(&output) = self.predicates.products.get(&key) { return Ok(output); }
        let output = Symbolic::new(self.builder, "metal.terminal.trace.product").product(left, right).map_err(|e| e.to_string())?;
        self.predicates.products.insert(key, output); Ok(output)
    }
    fn modulo(&mut self, numerator: u64, divisor: Value) -> Result<VarId, DerivationError> {
        let key = (self.context(), numerator, divisor.id());
        if let Some(&result) = self.predicates.modulo.get(&key) { return Ok(result); }
        let numerator = i64::try_from(numerator).map_err(|_| "terminal iteration exceeds shared integer")?;
        let source = self.builder.variable("metal.terminal.trace.ordinal", Domain::singleton(numerator));
        let quotient = self.builder.variable("metal.terminal.trace.quotient", Domain::interval(0, numerator).map_err(|e| e.to_string())?);
        let remainder = self.builder.variable("metal.terminal.trace.remainder", Domain::interval(0, divisor.bounds().1 as i64 - 1).map_err(|e| e.to_string())?);
        self.builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem { numerator: source, denominator: divisor.id(), quotient, remainder }));
        self.predicates.modulo.insert(key, remainder); Ok(remainder)
    }
    /// The outer sequence includes the complete continuation of an early
    /// kernel return. Falling off its end also ends every lane's execution;
    /// this is a control-flow boundary, not an additional return instruction.
    pub fn launch(&mut self, state: &mut Derivation<'_>, region: &Region) -> Result<(), DerivationError> {
        if let Region::Sequence(children) = region {
            state.memory.assume(&self.guards);
            self.symbols.assume(state, &self.guards);
            self.sequence(state, children, true)
        } else {
            self.region(state, region)?;
            state.active = 0;
            state.alive = 0;
            Ok(())
        }
    }
    pub fn region(&mut self, state: &mut Derivation<'_>, region: &Region) -> Result<(), DerivationError> {
        if state.active == 0 { return Ok(()); }
        state.memory.assume(&self.guards);
        self.symbols.assume(state, &self.guards);
        match region {
            Region::Sequence(children) => self.sequence(state, children, false)?,
            Region::Statement(site) => {
                self.symbols.annotate_statement(state, &site.statement);
                let definition = match &site.statement {
                    Statement::Let {name, value, ..} | Statement::Assign {name, value} => Some((name.clone(), self.symbols.values(state, value))),
                    _ => None,
                };
                self.capture(state, |state| state.block(std::slice::from_ref(site), 0, 1))?
                    .map_err(|reason| DerivationError::Unsupported(format!("{}: {}: {reason}", state.scope, site.statement.render())))?;
                if let Some((name, symbols)) = definition { self.symbols.assign(state, &name, symbols); }
            },
            Region::Choice { decision, alternatives } => {
                let before = State::capture(state, self.symbols);
                let predecessor = state.last;
                let variable = self.binding.parameters[*decision].ordinal;
                if let Some(known) = self.guards.iter().find(|known| known.variable == variable) {
                    let ordinal = usize::try_from(known.value).map_err(|_| "negative terminal choice ordinal")?;
                    let alternative = alternatives.get(ordinal).ok_or("terminal choice ordinal is outside its original domain")?;
                    return self.region(state, alternative);
                }
                let mut branches = Vec::new();
                let mut ends = predecessor.into_iter().collect::<Vec<_>>();
                for (ordinal, alternative) in alternatives.iter().enumerate() {
                    before.restore(state, self.symbols); state.last = predecessor;
                    let guard = Literal::new(variable, ordinal as i64);
                    let mut guards = self.guards.clone(); guards.push(guard);
                    let activation = guards.clone();
                    let binding = self.binding;
                    let presence = &mut *self.presence;
                    let predicates = &mut *self.predicates;
                    let symbols = &mut *self.symbols;
                    self.builder.when(guard, |builder| {
                        Recorder { builder, binding, guards, presence, predicates, symbols }.region(state, alternative)
                    })?;
                    ends.extend(state.last);
                    branches.push((activation, State::capture(state, self.symbols)));
                }
                State::alternatives(branches, Some((variable, alternatives.len())))?.restore(state, self.symbols);
                ends.sort_unstable(); ends.dedup(); self.join(state, ends)?;
            },
            Region::Select { guard, yes, no } => self.external(state, *guard, yes, Some(no))?,
            Region::Guarded { guard, body } => self.external(state, *guard, body, None)?,
            Region::Loop { header, decision: Some(decision), body } => self.traversal(state, header, *decision, body)?,
            Region::Loop { header, decision: None, body } => self.runtime_loop(state, header, body)?,
            Region::Scope { header, body, alternative } => {
                let outer = state.active;
                if let Statement::If(condition) = &header.statement {
                    let values = self.expression(state, condition)?;
                    let yes = match state.predicate(values) {
                        Ok(yes) => yes,
                        Err(reason) => {
                            if let Some(regions) = self.symbols.participation_regions(state, self.builder, condition)? {
                                self.primitive(state, Primitive::Branch)?;
                                self.participating(state, regions, body, alternative.as_deref())?;
                                return Ok(());
                            }
                            if let Some(selector) = self.symbols.predicate(state, self.builder, condition)? {
                                self.primitive(state, Primitive::Branch)?;
                                self.conditional(state, selector, body, alternative.as_deref())?;
                                return Ok(());
                            }
                            return Err(DerivationError::Unsupported(format!("{}: if ({}): {reason}", state.scope, condition.render())));
                        },
                    };
                    self.primitive(state, Primitive::Branch)?;
                    state.active = yes; self.region(state, body)?;
                    state.active = outer & !yes & state.alive;
                    if let Some(alternative) = alternative { self.region(state, alternative)?; }
                    state.active = outer & state.alive;
                } else {
                    let mut names = BTreeSet::new();
                    local_names(body, &mut names);
                    let saved_symbols = self.symbols.clone();
                    let saved_memory = state.memory.clone();
                    let saved = names.into_iter().map(|name| {
                        let values = (state.env.get(&name).copied(), state.ranges.get(&name).copied(), state.affine.get(&name).cloned());
                        (name, values)
                    }).collect::<Vec<_>>();
                    self.region(state, body)?;
                    for (name, values) in saved {
                        restore_name(state, &name, values);
                        self.symbols.restore_local(&name, &saved_symbols);
                        state.memory.restore_local(&name, &saved_memory);
                    }
                }
            },
        }
        Ok(())
    }
    fn sequence(&mut self, state: &mut Derivation<'_>, children: &[Region], launch_end: bool) -> Result<(), DerivationError> {
        self.sequence_body(state, children, launch_end)?;
        if launch_end {
            state.active = 0;
            state.alive = 0;
        }
        Ok(())
    }
    fn sequence_body(&mut self, state: &mut Derivation<'_>, children: &[Region], launch_end: bool) -> Result<(), DerivationError> {
        for (index, child) in children.iter().enumerate() {
            if state.active == 0 { break; }
            if let Region::Statement(crate::terminal::Site { statement: Statement::ReturnIf(condition), .. }) = child {
                if let Some(regions) = self.symbols.participation_regions(state, self.builder, condition)? {
                    // The continuation executes under the same arithmetic
                    // participation facts as the return decision. A Boolean
                    // model guard alone loses, for example, slot < work_count
                    // when deriving the live work item's coordinates.
                    self.expression(state, condition)?;
                    self.primitive(state, Primitive::Branch)?;
                    self.with_participation(state, regions, |recorder, state, returning, outer| {
                        state.active = outer & returning;
                        if state.active != 0 { recorder.primitive(state, Primitive::Return)?; }
                        state.alive &= !state.active;
                        state.active = outer & state.alive;
                        recorder.sequence(state, &children[index + 1..], launch_end)
                    })?;
                    return Ok(());
                }
                if let Some(selector) = self.symbols.predicate(state, self.builder, condition)? {
                    // Return participation guards the entire continuation. A
                    // padding slot is still charged for its original prologue.
                    self.expression(state, condition)?;
                    self.primitive(state, Primitive::Branch)?;
                    let before = State::capture(state, self.symbols);
                    let predecessor = state.last;
                    let mut ends = predecessor.into_iter().collect::<Vec<_>>();
                    let mut branches = Vec::new();
                    for returning in [true, false] {
                        before.restore(state, self.symbols); state.last = predecessor;
                        let guard = Literal::new(selector, i64::from(returning));
                        let mut guards = self.guards.clone(); guards.push(guard);
                        let activation = guards.clone();
                        let binding = self.binding; let presence = &mut *self.presence;
                        let predicates = &mut *self.predicates; let symbols = &mut *self.symbols;
                        self.builder.when(guard, |builder| -> Result<(), DerivationError> {
                            let mut recorder = Recorder { builder, binding, guards, presence, predicates, symbols };
                            if returning {
                                recorder.primitive(state, Primitive::Return)?;
                                state.alive &= !state.active;
                                state.active = 0;
                            }
                            recorder.sequence(state, &children[index + 1..], launch_end)
                        })?;
                        ends.extend(state.last);
                        branches.push((activation, State::capture(state, self.symbols)));
                    }
                    State::alternatives(branches, Some((selector, 2)))?.restore(state, self.symbols);
                    ends.sort_unstable(); ends.dedup(); self.join(state, ends)?;
                    return Ok(());
                }
            }
            self.region(state, child)?;
        }
        Ok(())
    }
    fn external(&mut self, state: &mut Derivation<'_>, guard: Literal, yes: &Region, no: Option<&Region>) -> Result<(), DerivationError> {
        let selector = self.binding.selector(guard)?;
        self.conditional(state, selector, yes, no)
    }
    fn participating(&mut self, state: &mut Derivation<'_>, regions: Vec<super::parameters::Participation>,
        yes: &Region, no: Option<&Region>) -> Result<(), DerivationError> {
        self.with_participation(state, regions, |recorder, state, mask, outer| {
            state.active = outer & mask;
            recorder.region(state, yes)?;
            state.active = outer & !mask & state.alive;
            if let Some(no) = no { recorder.region(state, no)?; }
            Ok(())
        })
    }
    fn with_participation(&mut self, state: &mut Derivation<'_>, regions: Vec<super::parameters::Participation>,
        mut action: impl FnMut(&mut Recorder<'_>, &mut Derivation<'_>, u32, u32) -> Result<(), DerivationError>) -> Result<(), DerivationError> {
        let before = State::capture(state, self.symbols);
        let predecessor = state.last;
        let mut branches = Vec::new();
        let mut ends = predecessor.into_iter().collect::<Vec<_>>();
        for region in regions {
            before.restore(state, self.symbols); state.last = predecessor;
            self.symbols.assume_participation(&region);
            let guard = Literal::new(region.active, 1);
            let mut guards = self.guards.clone(); guards.push(guard);
            let activation = guards.clone();
            let binding = self.binding; let presence = &mut *self.presence;
            let predicates = &mut *self.predicates; let symbols = &mut *self.symbols;
            let mut impossible = false;
            self.builder.when(guard, |builder| -> Result<(), DerivationError> {
                let mut recorder = Recorder { builder, binding, guards, presence, predicates, symbols };
                state.memory.assume(&recorder.guards);
                recorder.symbols.assume(state, &recorder.guards);
                if recorder.symbols.inconsistent() {
                    recorder.builder.constraint(Constraint::LinearLe { terms: Vec::new(), rhs: -1 });
                    impossible = true;
                    return Ok(());
                }
                action(&mut recorder, state, region.mask, before.active)?;
                state.active = before.active & state.alive;
                Ok(())
            })?;
            if impossible { continue; }
            ends.extend(state.last);
            branches.push((activation, State::capture(state, self.symbols)));
        }
        if branches.is_empty() {
            self.builder.constraint(Constraint::LinearLe { terms: Vec::new(), rhs: -1 });
            before.restore(state, self.symbols);
            state.last = predecessor;
            return Ok(());
        }
        State::alternatives(branches, None)?.restore(state, self.symbols);
        ends.sort_unstable(); ends.dedup(); self.join(state, ends)
    }
    fn conditional(&mut self, state: &mut Derivation<'_>, selector: VarId, yes: &Region, no: Option<&Region>) -> Result<(), DerivationError> {
        // A retained arm can re-enter the same original compiler predicate.
        // Its enclosing literal already decides this branch; deriving the
        // contradictory arm loses bindings from an unreachable source path.
        if let Some(known) = self.guards.iter().find(|known| known.variable == selector) {
            return match known.value {
                1 => self.region(state, yes),
                0 => no.map_or(Ok(()), |body| self.region(state, body)),
                _ => Err("retained Boolean selector has a non-Boolean activation".into()),
            };
        }
        let before = State::capture(state, self.symbols);
        let predecessor = state.last;
        let mut branches = Vec::new();
        let mut ends = predecessor.into_iter().collect::<Vec<_>>();
        for (selected, body) in [(true, Some(yes)), (false, no)] {
            before.restore(state, self.symbols); state.last = predecessor;
            let literal = Literal::new(selector, i64::from(selected));
            let mut guards = self.guards.clone(); guards.push(literal);
            let activation = guards.clone();
            if let Some(body) = body {
                let binding = self.binding;
                let presence = &mut *self.presence;
                let predicates = &mut *self.predicates;
                    let symbols = &mut *self.symbols;
                self.builder.when(literal, |builder|
                    Recorder { builder, binding, guards, presence, predicates, symbols }.region(state, body))?;
            }
            ends.extend(state.last);
            branches.push((activation, State::capture(state, self.symbols)));
        }
        State::alternatives(branches, Some((selector, 2)))?.restore(state, self.symbols);
        ends.sort_unstable(); ends.dedup(); self.join(state, ends)
    }
    fn runtime_loop(&mut self, state: &mut Derivation<'_>, header: &crate::terminal::Site, body: &Region) -> Result<(), DerivationError> {
        let Statement::For { name, start, end, step } = &header.statement else { return Err("retained loop has no loop header".into()); };
        if *step <= 0 { return Err("nonpositive retained loop step".into()); }
        if self.symbols.interval(state, end).is_some_and(|(lo, hi)| lo != hi)
            || self.symbols.interval(state, start).is_some_and(|(lo, hi)| lo != hi) {
            return self.parameter_loop(state, name, start, end, *step, body);
        }
        let outer = state.active;
        let mut index = self.expression(state, start)?;
        let saved = (state.env.get(name).copied(), state.ranges.get(name).copied(), state.affine.get(name).cloned());
        let saved_symbols = self.symbols.clone();
        self.symbols.remove(name);
        loop {
            state.assign(name, index);
            let limit = self.expression(state, end)?;
            let mut live = 0;
            for lane in 0..32 {
                if state.active & (1 << lane) != 0 {
                    let (Some(index), Some(limit)) = (index[lane], limit[lane]) else {
                        return Err(DerivationError::Unsupported(format!(
                            "{}: {}: lane {lane} has index {:?}, limit {:?}; symbolic origin {:?}, limit {:?}",
                            state.scope, header.statement.render(), index[lane], limit[lane],
                            self.symbols.interval(state, start), self.symbols.interval(state, end))));
                    };
                    if (index as i32) < (limit as i32) { live |= 1 << lane; }
                }
            }
            self.test(state)?; state.active = live;
            if live == 0 { break; }
            self.region(state, body)?;
            if state.active == 0 { break; }
            self.primitive(state, Primitive::Binary { operation: BinaryOp::Add, ty: Type::I32 })?;
            for lane in 0..32 {
                if state.active & (1 << lane) != 0 { index[lane] = index[lane].and_then(|value| (value as i32).checked_add(*step as i32)).map(|value| value as u32 as u64); }
            }
        }
        state.active = outer & state.alive;
        restore_name(state, name, saved);
        self.symbols.restore_local(name, &saved_symbols);
        Ok(())
    }
    fn parameter_loop(&mut self, state: &mut Derivation<'_>, name: &str, start: &Expression, end: &Expression, step: i64, body: &Region) -> Result<(), DerivationError> {
        let (minimum, _) = self.symbols.interval(state, start).ok_or_else(|| DerivationError::Unsupported("numeric loop has no bounded original origin".into()))?;
        let (_, maximum) = self.symbols.interval(state, end).ok_or_else(|| DerivationError::Unsupported("numeric loop has no bounded original limit".into()))?;
        let distance = i128::from(maximum).saturating_sub(i128::from(minimum)).max(0);
        let count = u64::try_from((distance + i128::from(step) - 1) / i128::from(step)).map_err(|_| "numeric loop occurrence range overflow")?;
        if count > state.limits.instructions as u64 { return Err(DerivationError::Exhausted(DerivationLimit::Instructions(state.limits.instructions))); }
        let initial = self.symbols.values(state, start);
        let saved_symbols = self.symbols.clone();
        let saved = (state.env.get(name).copied(), state.ranges.get(name).copied(), state.affine.get(name).cloned());
        let initial_values = self.expression(state, start)?;
        let mut previous: Option<[Option<super::parameters::Predicate>; 32]> = None;
        for ordinal in 0..=count {
            let offset = i64::try_from(i128::from(ordinal) * i128::from(step)).map_err(|_| "numeric loop induction exceeds integer range")?;
            let indices = initial_values.map(|value| value.and_then(|value| (value as i32).checked_add(i32::try_from(offset).ok()?)).map(|value| value as u32 as u64));
            state.assign(name, indices);
            let definition = self.symbols.offset(&initial, offset);
            self.symbols.assign(state, name, definition);
            let condition = Expression::binary(BinaryOp::Lt, Expression::variable(name, Type::I32), end.clone(), Type::Bool);
            let current = self.symbols.participation(state, &condition);
            let testing = previous.as_ref().cloned().unwrap_or_else(|| std::array::from_fn(|lane|
                (state.active & (1 << lane) != 0).then_some(super::parameters::Predicate::Constant(true))));
            let regions = self.symbols.participation_regions_for(state.active, self.builder, &testing)?
                .ok_or_else(|| DerivationError::Unsupported("numeric loop tests need retained symbolic lane geometry".into()))?;
            self.with_participation(state, regions, |this, state, mask, outer| {
                state.active = outer & mask;
                if state.active == 0 { return Ok(()); }
                this.expression(state, end)?; this.test(state)?;
                if ordinal == count { return Ok(()); }
                let regions = this.symbols.participation_regions_for(state.active, this.builder, &current)?
                    .ok_or_else(|| DerivationError::Unsupported("numeric loop body needs retained symbolic lane geometry".into()))?;
                this.with_participation(state, regions, |this, state, mask, outer| {
                    state.active = outer & mask;
                    if state.active == 0 { return Ok(()); }
                    this.region(state, body)?;
                    if state.active != 0 { this.primitive(state, Primitive::Binary { operation: BinaryOp::Add, ty: Type::I32 })?; }
                    Ok(())
                })
            })?;
            previous = Some(current);
        }
        restore_name(state, name, saved); self.symbols.restore_local(name, &saved_symbols);
        Ok(())
    }
    fn traversal(&mut self, state: &mut Derivation<'_>, header: &crate::terminal::Site, decision: usize, body: &Region) -> Result<(), DerivationError> {
        let Statement::For { name, start: Expression::Integer(first, Type::I32), step, .. } = &header.statement else { return Err("retained traversal lost its constant origin".into()); };
        let Decision::Traversal(definition) = &self.binding.family().definitions()[decision].decision else { return Err("retained traversal has a different decision kind".into()); };
        let iterations = definition.iterations;
        let parameter = &self.binding.parameters[decision];
        let width = parameter.width;
        let complete = parameter.complete.ok_or("retained traversal has no quotient")?;
        let scalar = self.equal(parameter.ordinal, 0);
        let unrolled = self.above(width, 1);
        let chunked = self.above(complete, 1);
        let single = self.equal(complete.id(), 1);
        let repeated = self.all(vec![unrolled, chunked]);
        let once = self.all(vec![unrolled, single]);
        let covered = self.product(complete, width)?;
        // Numeric operands stay unresolved. These integer expressions have
        // invariant primitive services; the selected quotient/remainder above
        // determines precisely which occurrences execute. Their resulting
        // source coordinate is the original logical iteration below.
        let width_name = format!("seismic_family_width_{}__", width.id().0);
        let chunk_name = format!("seismic_family_chunk_{}__", width.id().0);
        let offset_name = format!("seismic_family_offset_{}__", width.id().0);
        for operand in [&width_name, &chunk_name, &offset_name] { state.env.insert(operand.clone(), [None; 32]); }
        let saved = (state.env.get(name).copied(), state.ranges.get(name).copied(), state.affine.get(name).cloned());
        let outer = state.active;
        let saved_symbols = self.symbols.clone();
        self.symbols.remove(name);
        for ordinal in 0..iterations {
            if state.active == 0 { break; }
            let main = self.above(covered, ordinal as u64);
            let remainder = self.modulo(ordinal as u64, width)?;
            let first_in_chunk = self.equal(remainder, 0);
            let chunk_begin = self.all(vec![repeated, main, first_in_chunk]);
            self.optional(state, Literal::new(scalar, 1), |this, state| this.test(state))?;
            self.optional(state, Literal::new(chunk_begin, 1), |this, state| this.test(state))?;
            let repeated_coordinate = self.all(vec![repeated, main]);
            self.optional(state, Literal::new(repeated_coordinate, 1), |this, state| {
                let coordinate = crate::terminal::traversal::coordinate(*first, *step,
                    Expression::variable(&width_name, Type::I64), Some(Expression::variable(&chunk_name, Type::I32)), Expression::variable(&offset_name, Type::I64));
                this.expression(state, &coordinate).map(|_| ())
            })?;
            let single_coordinate = self.all(vec![once, main]);
            self.optional(state, Literal::new(single_coordinate, 1), |this, state| {
                let coordinate = crate::terminal::traversal::coordinate(*first, *step,
                    Expression::variable(&width_name, Type::I64), None, Expression::variable(&offset_name, Type::I64));
                this.expression(state, &coordinate).map(|_| ())
            })?;
            let value = first.checked_add((ordinal as i64).checked_mul(*step).ok_or("retained traversal coordinate overflow")?).ok_or("retained traversal coordinate overflow")?;
            state.assign(name, [Some(value as i32 as u32 as u64); 32]);
            self.region(state, body)?;
            if state.active == 0 { break; }
            self.optional(state, Literal::new(scalar, 1), |this, state| this.primitive(state, Primitive::Binary { operation: BinaryOp::Add, ty: Type::I32 }))?;
            let next_remainder = self.modulo(ordinal as u64 + 1, width)?;
            let last_in_chunk = self.equal(next_remainder, 0);
            let chunk_end = self.all(vec![repeated, main, last_in_chunk]);
            self.optional(state, Literal::new(chunk_end, 1), |this, state| this.primitive(state, Primitive::Binary { operation: BinaryOp::Add, ty: Type::I32 }))?;
            let main_end = self.equal(covered.id(), ordinal as i64 + 1);
            let complete_test = self.all(vec![repeated, main_end]);
            self.optional(state, Literal::new(complete_test, 1), |this, state| this.test(state))?;
        }
        if state.active != 0 { self.optional(state, Literal::new(scalar, 1), |this, state| this.test(state))?; }
        state.active = outer & state.alive;
        restore_name(state, name, saved);
        self.symbols.restore_local(name, &saved_symbols);
        for operand in [&width_name, &chunk_name, &offset_name] { state.env.remove(operand); }
        Ok(())
    }
}
fn local_names(region: &Region, out: &mut BTreeSet<String>) {
    match region {
        Region::Sequence(children) => for child in children { local_names(child, out); },
        Region::Choice { alternatives, .. } => for alternative in alternatives { local_names(alternative, out); },
        Region::Statement(site) => match &site.statement {
            Statement::Let { name, .. } | Statement::Array { name, .. } | Statement::Pointer { name, .. } | Statement::Fragment { name, .. } => { out.insert(name.clone()); },
            Statement::VectorRead { name, components, .. } => for component in 0..*components { out.insert(format!("{name}[{component}]")); },
            _ => {},
        },
        Region::Select { yes, no, .. } => { local_names(yes, out); local_names(no, out); },
        Region::Guarded { body, .. } => local_names(body, out),
        Region::Loop { .. } | Region::Scope { .. } => {},
    }
}
fn restore_name(state: &mut Derivation<'_>, name: &str, saved: (Option<Values>, Option<ranges::Ranges>, Option<affine::Values>)) {
    match saved.0 { Some(value) => { state.env.insert(name.into(), value); }, None => { state.env.remove(name); } }
    match saved.1 { Some(value) => { state.ranges.insert(name.into(), value); }, None => { state.ranges.remove(name); } }
    match saved.2 { Some(value) => { state.affine.insert(name.into(), value); }, None => { state.affine.remove(name); } }
    state.facts.clear();
    state.assumptions.clear();
}
